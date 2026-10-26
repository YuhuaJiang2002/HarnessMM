#!/usr/bin/env python3
"""Run only the outcome-independent first frozen TRAIN task using native Harbor.

Later paired orchestration can use
``integrations.deepseek_harness.harbor_agent:DeepSeekHarborAgent`` directly.
This preflight entry point never opens SEALED_TEST.json or selects by outcome.
It requires an isolated Harbor 0.18.0 environment on the authorized A800 host.
Use ``--detach`` over SSH: the wrapper itself must survive a lost SSH connection.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.deepseek_harness.harbor_agent import (  # noqa: E402
    HARBOR_VERSION,
    MODEL_NAME,
    canonical_hash,
    file_hash,
    known_nonnegative,
    runtime_identity,
    sum_known,
    utc_now,
    write_json,
)

CAMPAIGN = Path("/opt/phiagent/work/deepseek-harness/20260929-v1")
PARTITION_SHA256 = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
TERMINAL_REVISION = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
AUTHORIZED_HOST = "harnessmm-api-worker"


def task_file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def select_first_training_task(partition_dir: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = partition_dir / "SPLIT_MANIFEST.json"
    if file_hash(manifest_path) != expected_sha256:
        raise ValueError("Frozen partition manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["benchmark"] != "terminal-bench-2.0"
        or manifest["population"] != 89
        or manifest["counts"] != {"train": 18, "validation": 18, "test": 53}
        or manifest["split_seed"] != 20260928
        or manifest["score_or_outcome_used"] is not False
    ):
        raise ValueError("Expected the complete pinned 18/18/53 outcome-independent partition")
    splits = manifest["splits"]
    all_ids = [unit for split in ("train", "validation", "test") for unit in splits[split]]
    if len(all_ids) != 89 or len(set(all_ids)) != 89:
        raise ValueError("Partition membership is incomplete or overlapping")
    if any(len(splits[name]) != manifest["counts"][name] for name in splits):
        raise ValueError("Partition counts do not match membership")
    first_id = splits["train"][0]
    prepared = json.loads((partition_dir / "TRAIN_VALIDATION.json").read_text())
    if [item["id"] for item in prepared["train"]] != splits["train"]:
        raise ValueError("Prepared training order differs from the frozen manifest")
    task = prepared["train"][0]
    if task["id"] != first_id:
        raise ValueError("The task is not the first frozen TRAIN task")
    directory = Path(task["path"]).resolve()
    hashes = task_file_hashes(directory)
    digest = canonical_hash(hashes)
    record = next(item for item in manifest["records"] if item["id"] == first_id)
    if digest != task["sha256"] or digest != record["payload_sha256"]:
        raise ValueError("Official task source does not match the frozen partition")
    if not {"task.toml", "instruction.md"}.issubset(hashes):
        raise ValueError("The selected task has no native Harbor definition")
    return {**task, "path": str(directory), "source_files": hashes}


def read_command(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        process = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)
        return {
            "argv": argv,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error_type": type(exc).__name__, "error": str(exc)}


def resource_inventory() -> dict[str, Any]:
    commands = {
        "gpu_inventory": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        "gpu_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        "memory": ["free", "-b"],
        "disk": ["df", "-h", "/path/to/data", "/opt/phiagent"],
        "docker": [
            "docker", "info", "--format",
            "ServerVersion={{.ServerVersion}}\nDriver={{.Driver}}\n"
            "DockerRootDir={{.DockerRootDir}}\nNCPU={{.NCPU}}\nMemTotal={{.MemTotal}}\n"
            "SecurityOptions={{json .SecurityOptions}}",
        ],
        "containers": [
            "docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}",
        ],
    }
    return {
        "timestamp": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_gpus": [],
        "compute": "CPU Docker task and remote official DeepSeek inference; no GPU allocation",
        "observations": {name: read_command(argv) for name, argv in commands.items()},
    }


def freeze_source(root: Path, destination: Path) -> dict[str, str]:
    files = sorted((root / "integrations" / "deepseek_harness").glob("*.py"))
    files += sorted((root / "integrations" / "deepseek_harness").glob("*.mjs"))
    files += [
        root / "scripts" / "run_deepseek_terminal.py",
        root / "requirements" / "deepseek-harness-terminal.txt",
        root / "tests" / "test_deepseek_terminal_adapter.py",
        root / "phiagent" / "__init__.py",
        root / "phiagent" / "harness" / "deepseek_bridge.py",
        root / "phiagent" / "harness" / "loopback_http.py",
        root / "phiagent" / "data_engine" / "provenance.py",
        root / "scripts" / "serve_deepseek_durable_capture.py",
    ]
    # This namespace subset avoids importing unrelated planners from harness/__init__.py.
    hashes = {}
    for source in files:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = file_hash(target)
    return hashes


def build_native_job(
    run_dir: Path,
    task_path: Path,
    agent_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "job_name": "native",
        "jobs_dir": str(run_dir / "harbor"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "force_build": False, "delete": True},
        "verifier": {"disable": False},
        "agents": [{
            "name": None,
            "import_path": "integrations.deepseek_harness.harbor_agent:DeepSeekHarborAgent",
            "model_name": MODEL_NAME,
            "kwargs": agent_kwargs,
        }],
        "tasks": [{"path": str(task_path)}],
    }


def merge_attempt_accounting(
    current: dict[str, Any],
    predecessors: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    for summary in predecessors:
        for attempt in summary["attempts"]:
            identity = attempt["attempt_id"]
            if identity in by_id and by_id[identity] != attempt:
                raise ValueError("Conflicting accounting for an inherited attempt")
            by_id[identity] = attempt
    identity = current["attempt_id"]
    if identity in by_id:
        raise ValueError("A continuation must use a new attempt directory")
    by_id[identity] = current
    attempts = list(by_id.values())
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "total_tokens",
        "prompt_tokens_including_cache",
    )
    known_subtotals = {}
    for key in fields:
        samples = []
        for attempt in attempts:
            reported = attempt.get("known_reported_usage", {}).get(key)
            value = known_nonnegative(reported)
            if value is None:
                value = known_nonnegative(attempt.get("usage", {}).get(key))
            if value is not None:
                samples.append(value)
        known_subtotals[key] = sum(samples) if samples else None
    return {
        "attempts": attempts,
        "cumulative_usage": {
            key: sum_known([attempt.get("usage", {}).get(key) for attempt in attempts])
            for key in fields
        },
        "known_usage_subtotals": known_subtotals,
        "unknown_usage_attempts": {
            key: [
                attempt["attempt_id"] for attempt in attempts
                if known_nonnegative(attempt.get("usage", {}).get(key)) is None
            ]
            for key in fields
        },
        "cumulative_token_limit": None,
        "unknown_is_zero": False,
        "accounting_unit": "unique SDK provider usage events; unique inherited attempt IDs",
    }


def summarize_native_trial(
    run_dir: Path,
    predecessors: list[dict[str, Any]],
) -> dict[str, Any]:
    results = []
    for path in sorted((run_dir / "harbor" / "native").glob("*/result.json")):
        value = json.loads(path.read_text())
        if "task_name" in value:
            results.append((path, value))
    if len(results) > 1:
        raise ValueError("The preflight must contain exactly one native trial")
    path, trial = results[0] if results else (None, {})
    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    context = trial.get("agent_result") or {}
    metadata = context.get("metadata") or {}
    usage = metadata.get("deepseek_usage", {})
    known_usage = metadata.get("deepseek_known_reported_usage", {})
    provider_never_started = (
        "agent_execution" in trial and trial["agent_execution"] is None
        and trial.get("config", {}).get("agent", {}).get("import_path")
        == "integrations.deepseek_harness.harbor_agent:DeepSeekHarborAgent"
    )
    if provider_never_started:
        usage = {key: 0 for key in (
            "input_tokens", "output_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens", "total_tokens", "prompt_tokens_including_cache",
        )}
        known_usage = dict(usage)
    exception = trial.get("exception_info")
    valid_rewards = isinstance(rewards, dict) and bool(rewards) and all(
        type(value) in (int, float) and math.isfinite(value) for value in rewards.values()
    )
    status = "valid_native_verdict" if valid_rewards else "no_native_verdict"
    if exception and exception.get("exception_type") not in (None, "AgentTimeoutError"):
        status = "native_trial_error"
    current = {
        "attempt_id": str(run_dir.resolve()),
        "native_trial_result": str(path) if path else None,
        "status": status,
        "usage": {
            key: known_nonnegative(usage.get(key))
            for key in (
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "total_tokens", "prompt_tokens_including_cache",
            )
        },
        "known_reported_usage": known_usage,
        "provider_never_started": provider_never_started,
        "usage_evidence": (
            "Official trial agent_execution is null; this adapter invokes the SDK only in run(), "
            "never during environment or agent setup"
        ) if provider_never_started else "SDK provider receipts; missing values remain unknown",
    }
    return {
        "schema": "deepseek-terminal-training-preflight-v1",
        "timestamp": utc_now(),
        "task_id": trial.get("task_name"),
        "split": "train",
        "purpose": "Training/operability only; not a final paired comparison or leaderboard score",
        "native_rewards": rewards,
        "valid_for_comparison": status == "valid_native_verdict",
        "native_exception": exception,
        "native_trial_result": str(path) if path else None,
        "status": status,
        "agent_context": context,
        **merge_attempt_accounting(current, predecessors),
    }


def validate_acceptance(receipt: dict[str, Any], runtime: dict[str, str], source: Path) -> None:
    identity = runtime_identity(runtime)
    expected = {
        "status": "passed",
        "upstream_commit": identity["upstream_commit"],
        "upstream_tag": identity["upstream_tag"],
        "profile": "sdk",
        "provider": "deepseek-official",
        "model": MODEL_NAME,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Official SDK acceptance has not passed for the pinned campaign identity")
    required_checks = {"official_runtime", "tool_catalog_restriction", "tool_bridge", "usage_accounting"}
    if any(receipt.get("checks", {}).get(key) is not True for key in required_checks):
        raise ValueError("SDK acceptance is missing an essential execution/isolation/accounting gate")
    if receipt.get("runtime_manifest_sha256") != identity["runtime_manifest_sha256"]:
        raise ValueError("The SDK runtime changed after acceptance")
    for filename in ("runner.py", "bridge_plugin.mjs", "chisel_plugin.mjs", "cot_jev.mjs"):
        current = file_hash(source / "integrations" / "deepseek_harness" / filename)
        if receipt.get("adapter_hashes", {}).get(filename) != current:
            raise ValueError(f"SDK source changed after acceptance: {filename}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--detach", action="store_true",
                        help="Detach the entire wrapper; persist launcher PID/log beside the run")
    parser.add_argument("--partition-dir", type=Path, default=CAMPAIGN / "terminal-partition-v1")
    parser.add_argument("--partition-sha256", default=PARTITION_SHA256)
    parser.add_argument("--runtime-config", type=Path, required=True,
                        help="JSON object: source_root, dsh_bin, node_bin, manifest")
    parser.add_argument("--runner-python", type=Path, required=True)
    parser.add_argument("--sdk-private-root", type=Path,
                        help="New sidecar below campaign/runs for an accepted worker's output allowlist")
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument("--source-git-state", type=Path,
                        help="Original checkout Git receipt when running from an untracked remote mirror")
    parser.add_argument("--api-key-file", type=Path,
                        default=Path("/path/to/private-config/credentials/deepseek-harness-20260929"))
    parser.add_argument("--max-output-tokens", type=int, default=32768,
                        help="Official per-request output capacity, never a cumulative token budget")
    parser.add_argument("--predecessor", type=Path, action="append", default=[],
                        help="Prior failed SUMMARY.json; every provider cost is inherited exactly once")
    return parser.parse_args(argv)


def detached_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--run-dir", str(args.run_dir.resolve()),
        "--partition-dir", str(args.partition_dir.resolve()),
        "--partition-sha256", args.partition_sha256,
        "--runtime-config", str(args.runtime_config.resolve()),
        "--runner-python", str(args.runner_python.absolute()),
        "--acceptance-receipt", str(args.acceptance_receipt.resolve()),
        "--api-key-file", str(args.api_key_file.resolve()),
        "--max-output-tokens", str(args.max_output_tokens),
    ]
    if args.source_git_state:
        command.extend(["--source-git-state", str(args.source_git_state.resolve())])
    if args.sdk_private_root:
        command.extend(["--sdk-private-root", str(args.sdk_private_root.resolve())])
    for predecessor in args.predecessor:
        command.extend(["--predecessor", str(predecessor.resolve())])
    return command


def launch_detached(args: argparse.Namespace) -> dict[str, Any]:
    if socket.gethostname() != AUTHORIZED_HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Detached launches require a800-4 and explicit CUDA_VISIBLE_DEVICES=''")
    run_dir = args.run_dir.resolve()
    allowed_root = CAMPAIGN / "terminal-training-preflight"
    if not run_dir.is_relative_to(allowed_root) or run_dir == allowed_root:
        raise ValueError(f"Use a new run subdirectory below {allowed_root}")
    if run_dir.exists():
        raise FileExistsError("Never reuse a frozen experiment directory")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = run_dir.with_name(run_dir.name + ".launcher.log")
    receipt_path = run_dir.with_name(run_dir.name + ".launcher.json")
    if receipt_path.exists():
        raise FileExistsError("This launch identity already has a preserved receipt")
    command = detached_command(args)
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    receipt = {
        "schema": "deepseek-terminal-detached-launch-v1",
        "started_at": utc_now(),
        "hostname": socket.gethostname(),
        "wrapper_pid": process.pid,
        "new_session": True,
        "stdin": "DEVNULL",
        "log": str(log_path),
        "run_dir": str(run_dir),
        "argv": command,
        "state": "starting",
        "startup_wait_is_not_a_trial_deadline": True,
    }
    write_json(receipt_path, receipt)
    deadline = time.monotonic() + 10
    while True:
        returncode = process.poll()
        if returncode is not None:
            receipt.update(state="exited", returncode=returncode)
            break
        if (run_dir / "RUN_MANIFEST.json").exists():
            receipt.update(state="running", run_manifest_observed=True)
            break
        if time.monotonic() >= deadline:
            receipt.update(state="starting", run_manifest_observed=False)
            break
        time.sleep(0.1)
    receipt["checked_at"] = utc_now()
    receipt["receipt_path"] = str(receipt_path)
    write_json(receipt_path, receipt)
    if receipt["state"] == "exited" and receipt["returncode"] != 0:
        raise RuntimeError(f"Detached wrapper failed at startup; preserved log: {log_path}")
    return receipt


def run_first_training(args: argparse.Namespace) -> dict[str, Any]:
    if socket.gethostname() != AUTHORIZED_HOST:
        raise RuntimeError("This benchmark entry point runs only on a800-4 (harnessmm-api-worker)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' explicitly before launching")
    if importlib.metadata.version("harbor") != HARBOR_VERSION:
        raise RuntimeError(f"Use the isolated terminal-runtime with Harbor {HARBOR_VERSION}")
    run_dir = args.run_dir.resolve()
    allowed_root = CAMPAIGN / "terminal-training-preflight"
    if not run_dir.is_relative_to(allowed_root) or run_dir == allowed_root:
        raise ValueError(f"Use a new run subdirectory below {allowed_root}")
    if run_dir.exists():
        raise FileExistsError("Never reuse a frozen experiment directory")
    runtime = json.loads(args.runtime_config.read_text())
    if set(runtime) != {"source_root", "dsh_bin", "node_bin", "manifest"}:
        raise ValueError("runtime-config must contain the official worker's four runtime paths")
    if any(not Path(path).is_absolute() for path in runtime.values()):
        raise ValueError("Every official runtime path must be absolute")
    acceptance = json.loads(args.acceptance_receipt.read_text())
    validate_acceptance(acceptance, runtime, ROOT)
    if not args.runner_python.is_absolute() or not args.runner_python.is_file():
        raise ValueError("runner-python must be an existing absolute SDK interpreter path")
    if args.sdk_private_root:
        private_root = args.sdk_private_root.resolve()
        if not private_root.is_relative_to(CAMPAIGN / "runs") or private_root == CAMPAIGN / "runs":
            raise ValueError("SDK sidecars must be below the accepted worker's campaign/runs allowlist")
        if private_root.exists():
            raise FileExistsError("The SDK private sidecar must be a new directory")
    task = select_first_training_task(args.partition_dir, args.partition_sha256)
    predecessors = [json.loads(path.read_text()) for path in args.predecessor]
    inherited = {
        attempt["attempt_id"]
        for summary in predecessors for attempt in summary.get("attempts", [])
    }
    if len(inherited) >= 2:
        raise ValueError("At most two distinct first-training attempts are authorized")
    for summary in predecessors:
        if summary.get("valid_for_comparison") or summary.get("status") == "valid_native_verdict":
            raise ValueError("A valid first-training verdict already exists; do not repeat it")
        if summary.get("task_id") not in (None, task["id"]) or summary.get("split") != "train":
            raise ValueError("Only this first-training trial may be an inherited predecessor")
    allowed_root.mkdir(parents=True, exist_ok=True)
    with (allowed_root / ".first-train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior_runs = list(allowed_root.glob("*/RUN_MANIFEST.json"))
        for previous in prior_runs:
            if str(previous.parent.resolve()) not in inherited:
                raise ValueError(
                    f"A previous attempt exists and must be accounted for: {previous.parent}"
                )
        if len(prior_runs) >= 2:
            raise ValueError("At most two distinct first-training attempts are authorized")
        run_dir.mkdir(mode=0o700)
        try:
            return _run_frozen_trial(args, run_dir, task, runtime, acceptance, predecessors)
        except Exception as exc:
            if not (run_dir / "SUMMARY.json").exists():
                summary = summarize_native_trial(run_dir, predecessors)
                summary.update({
                    "task_id": task["id"],
                    "status": "preparation_error",
                    "valid_for_comparison": False,
                    "launcher_failure": {"type": type(exc).__name__, "message": str(exc)},
                })
                write_json(run_dir / "SUMMARY.json", summary)
            raise


def _run_frozen_trial(
    args: argparse.Namespace,
    run_dir: Path,
    task: dict[str, Any],
    runtime: dict[str, str],
    acceptance: dict[str, Any],
    predecessors: list[dict[str, Any]],
) -> dict[str, Any]:
    write_json(run_dir / "RUN_MANIFEST.json", {
        "schema": "deepseek-terminal-training-preflight-v1",
        "created_at": utc_now(),
        "attempt_id": str(run_dir),
        "task_id": task["id"],
        "selection": "First frozen TRAIN task in outcome-independent hash order",
        "purpose": "training/operability",
        "arm": "baseline",
        "benchmark": "terminal-bench-2.0",
        "dataset_revision": TERMINAL_REVISION,
        "harbor_version": HARBOR_VERSION,
        "official_deepseek_revision": acceptance["upstream_commit"],
        "official_deepseek_tag": acceptance["upstream_tag"],
        "official_sdk_distribution_version": acceptance.get("sdk_distribution_version"),
        "official_runtime_launch_mode": acceptance.get("runtime_launch_mode"),
        "partition_sha256": args.partition_sha256,
        "task_sha256": task["sha256"],
        "split_seed": 20260928,
        "counts": {"train": 18, "validation": 18, "test": 53},
        "sealed_test_opened": False,
        "cumulative_token_limit": None,
        "predecessors": [str(path.resolve()) for path in args.predecessor],
        "detached_launcher_receipt": str(
            run_dir.with_name(run_dir.name + ".launcher.json")
        ) if run_dir.with_name(run_dir.name + ".launcher.json").exists() else None,
    })
    snapshot_root = run_dir / "source"
    source_hashes = freeze_source(ROOT, snapshot_root)
    write_json(run_dir / "SOURCE_SHA256.json", source_hashes)
    validate_acceptance(acceptance, runtime, snapshot_root)
    source_paths = sorted(source_hashes)
    git_state = {
        "head": read_command(["git", "rev-parse", "HEAD"], ROOT),
        "scoped_status": read_command(["git", "status", "--porcelain", "--", *source_paths], ROOT),
        "scope": source_paths,
    }
    if args.source_git_state:
        git_state["original_checkout"] = json.loads(args.source_git_state.read_text())
    write_json(run_dir / "GIT_STATE.json", git_state)
    write_json(run_dir / "RESOURCE_INVENTORY.json", resource_inventory())
    write_json(run_dir / "PACKAGES.json", {
        "python": sys.version,
        "executable": sys.executable,
        "packages": sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
        ),
    })
    write_json(run_dir / "SDK_ACCEPTANCE.json", acceptance)
    shutil.copy2(Path(runtime["manifest"]), run_dir / "SDK_RUNTIME_MANIFEST.json")
    shutil.copy2(args.partition_dir / "SPLIT_MANIFEST.json", run_dir / "SPLIT_MANIFEST.json")
    task_path = run_dir / "task-source" / task["id"]
    shutil.copytree(task["path"], task_path)
    if task_file_hashes(task_path) != task["source_files"]:
        raise ValueError("Task source changed while freezing the run")
    write_json(run_dir / "TASK_SOURCE_SHA256.json", task["source_files"])
    private_root = args.sdk_private_root.resolve() if args.sdk_private_root else run_dir / "private"
    private_root.mkdir(parents=True, mode=0o700)
    if args.sdk_private_root:
        (run_dir / "private").symlink_to(private_root, target_is_directory=True)
        write_json(run_dir / "SDK_ARTIFACT_REFERENCE.json", {
            "private_root": str(private_root),
            "linked_from": str(run_dir / "private"),
            "reason": "Preserve exact accepted SDK worker bytes and its existing output allowlist",
        })
    source_revision = git_state["head"].get("stdout", "").strip() or None
    if not source_revision and isinstance(git_state.get("original_checkout"), dict):
        source_revision = git_state["original_checkout"].get("head")
    agent_kwargs = {
        "runner_python": str(args.runner_python),
        "runner_script": str(snapshot_root / "integrations" / "deepseek_harness" / "runner.py"),
        "runtime": runtime,
        "api_key_file": str(args.api_key_file),
        "private_root": str(private_root),
        "max_tokens": args.max_output_tokens,
        "seed": 20260928,
        "arm": "baseline",
        "chisel": None,
        "source_revision": source_revision,
    }
    from harbor.models.job.config import JobConfig

    job = JobConfig.model_validate(build_native_job(run_dir, task_path, agent_kwargs))
    job_path = run_dir / "harbor-job.json"
    write_json(job_path, job.model_dump(mode="json"))
    command = [str(Path(sys.executable).parent / "harbor"), "run", "--config", str(job_path)]
    write_json(run_dir / "COMMANDS.json", {
        "launcher_argv": sys.argv,
        "native_argv": command,
        "cwd": str(run_dir),
        "PYTHONPATH": str(snapshot_root),
        "CUDA_VISIBLE_DEVICES": "",
        "cumulative_token_limit": None,
    })
    cache_dir = run_dir / "build-cache"
    cache_dir.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": str(snapshot_root),
        "CUDA_VISIBLE_DEVICES": "",
        "TMPDIR": str(cache_dir),
    }
    returncode = None
    failure = None
    process = None
    try:
        with (run_dir / "harbor.log").open("wb") as log:
            process = subprocess.Popen(
                command, cwd=run_dir, env=environment, stdout=log, stderr=log,
                start_new_session=True,
            )
            write_json(run_dir / "PROCESS.json", {
                "wrapper_pid": os.getpid(), "native_harbor_pid": process.pid, "started_at": utc_now(),
            })
            returncode = process.wait()
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            returncode = process.returncode
        summary = summarize_native_trial(run_dir, predecessors)
        summary["task_id"] = task["id"]
        summary["harbor_returncode"] = returncode
        summary["launcher_failure"] = failure
        summary["task_source_unchanged"] = task_file_hashes(task_path) == task["source_files"]
        if returncode != 0 or failure or not summary["task_source_unchanged"]:
            summary["status"] = "native_trial_error"
            summary["valid_for_comparison"] = False
        write_json(run_dir / "SUMMARY.json", summary)
        if args.sdk_private_root:
            write_json(run_dir / "SDK_ARTIFACT_SHA256.json", {
                path.relative_to(private_root).as_posix(): file_hash(path)
                for path in sorted(private_root.rglob("*"))
                if path.is_file() and not path.is_symlink()
                and "cache" not in path.relative_to(private_root).parts
                and "process-cache" not in path.relative_to(private_root).parts
            })
        write_json(run_dir / "ARTIFACT_SHA256.json", {
            path.relative_to(run_dir).as_posix(): file_hash(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and not path.is_symlink()
            and "build-cache" not in path.relative_to(run_dir).parts
            and "process-cache" not in path.relative_to(run_dir).parts
            and "cache" not in path.relative_to(run_dir).parts
            and path.name != "ARTIFACT_SHA256.json"
        })
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detach:
        print(json.dumps(launch_detached(args), sort_keys=True))
        return 0
    summary = run_first_training(args)
    print(json.dumps({
        "run_dir": str(args.run_dir.resolve()),
        "status": summary["status"],
        "native_rewards": summary["native_rewards"],
        "cumulative_usage": summary["cumulative_usage"],
    }, sort_keys=True))
    return 0 if summary["valid_for_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
