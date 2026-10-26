#!/usr/bin/env python3
"""New shared-parent full18 TRAIN baselines; no inherited outcomes or held-out runs.

Run on the authorized A800 CPU/API host with --config, --track, --run-dir and
--detach. The detached controller owns the complete TRAIN queue and a durable
heartbeat. --resume only harvests already-dispatched work; it never replays an
ambiguous actor. The first native gate belongs to the same 18 observations.
The resulting CAPABILITY_GATE.json is required before a future CHISEL stage.
Raw task bytes and Harbor-rendered task prompt receipts are frozen separately.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import utc_now, write_json_atomic  # noqa: E402
from phiagent.harness.benchmark_partition import (  # noqa: E402
    canonical_sha256, validate_partition,
)
from phiagent.harness.industrial_common_prompt import (  # noqa: E402
    BENCHMARK, PROMPT_SHA256, PROMPT_VERSION, prompt_manifest, prompt_receipt,
)
from phiagent.harness.industrial_capacity import task_admission, validate_workers  # noqa: E402
from scripts.run_codex_terminal import executed_native_verdict, native_reward  # noqa: E402
from scripts.run_deepseek_terminal import (  # noqa: E402
    build_native_job, read_command, resource_inventory, task_file_hashes,
)
from scripts.run_deepseek_terminal_campaign import canonical_usage  # noqa: E402
from scripts.serve_codex_deepseek import usage_summary  # noqa: E402

SCHEMA = "industrial-terminal-capability/v1"
HOSTNAME = "harnessmm-api-worker"
PARTITION_SHA256 = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
RUN_ROOT = Path("/path/to/data/phiagent/work/deepseek-harness/industrial-capability/20260930-v1")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
SOURCE_ENTRIES = (
    "scripts/run_industrial_capability_baselines.py",
    "scripts/run_codex_context_control.py",
    "scripts/industrial_terminal_agents.py",
    "scripts/run_codex_terminal.py",
    "scripts/recover_terminal_verifier.py",
    "scripts/recover_codex_sam_verifier.py",
    "integrations/deepseek_harness/runner.py",
    "phiagent/__init__.py",
    "tests/test_industrial_capability.py",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict) -> None:
    expected = {
        "schema": SCHEMA, "benchmark": BENCHMARK, "phase": "train",
        "methods": ["codex", "deepseek"], "model": "deepseek-flash",
        "counts": {"train": 18, "validation": 18, "test": 53},
        "partition_sha256": PARTITION_SHA256, "split_seed": 20260928,
        "execution_seed": 20260928, "cumulative_token_limit": None,
        "token_usage_mode": "unlimited", "sampling_seed_supported": False,
        "historical_baseline_reuse": False, "automatic_rsi_execution": False,
        "authorized_hostname": HOSTNAME, "run_root": str(RUN_ROOT),
        "CUDA_VISIBLE_DEVICES": "", "harbor_version": "0.18.0",
        "capability_gate": {"required_task_count": 18, "minimum_successes": 9},
        "dataset_revision": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Only a new, unlimited, shared-parent complete18 TRAIN campaign is allowed")
    prompt_manifest(
        config["benchmark"], config["shared_prompt_version"], config["shared_prompt_sha256"],
    )
    if (
        config.get("minimum_free_disk_gib", 0) < 8
        or config.get("minimum_available_memory_gib", 0) < 8
        or not 0 < config.get("heartbeat_seconds", 0) <= 30
        or not 0 < config.get("capacity_audit_seconds", 0) <= 300
    ):
        raise ValueError("Preserve native capacity reserves and bounded, timely monitoring")
    validate_workers(config.get("maximum_workers"))
    if config["deepseek"]["max_output_tokens"] != 32768:
        raise ValueError("The native SDK per-response output capacity is unchanged")


def train_records(manifest: dict, config: dict) -> list[dict]:
    validate_config(config)
    validate_partition(manifest)
    if (
        manifest["benchmark"] != BENCHMARK or manifest["population"] != 89
        or manifest["counts"] != config["counts"]
        or manifest["split_seed"] != config["split_seed"]
        or manifest["score_or_outcome_used"] is not False
    ):
        raise ValueError("Full outcome-independent 18/18/53 partition required")
    records = {row["id"]: row for row in manifest["records"]}
    return [records[task_id] for task_id in manifest["splits"]["train"]]


def source_files(root: Path = ROOT, *, extra_entries: tuple[str, ...] = ()) -> list[str]:
    """Freeze the local import closure, without vendoring installed runtimes."""
    pending, found = [*SOURCE_ENTRIES, *extra_entries], set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        found.add(relative)
        source = root / relative
        package = relative.removesuffix(".py").split("/")[:-1]
        for node in ast.walk(ast.parse(source.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join([*prefix, *([node.module] if node.module else [])])
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            for module in modules:
                if module.split(".")[0] not in ("phiagent", "scripts", "integrations"):
                    continue
                name = module.replace(".", "/") + ".py"
                if (root / name).is_file() and name not in found:
                    pending.append(name)
    found.update({
        "integrations/deepseek_harness/bridge_plugin.mjs",
        "integrations/deepseek_harness/cot_jev.mjs",
        "integrations/deepseek_harness/jev_v2.mjs",
        "integrations/deepseek_harness/sar_v2.mjs",
        "integrations/deepseek_harness/sar_admission.mjs",
        "integrations/deepseek_harness/chisel_plugin.mjs",
        "integrations/deepseek_harness/session_worker.py",
        "integrations/deepseek_harness/observation_policy.mjs",
        "integrations/deepseek_harness/action_evidence_checkpoint.mjs",
        "requirements/deepseek-harness-terminal.txt",
        "configs/industrial_terminal_capability_v1.json",
    })
    return sorted(found)


def task_prompt_receipt(task: Path, expected_files: dict) -> dict:
    if task_file_hashes(task) != expected_files:
        raise ValueError("Frozen native task bytes changed before prompt preparation")
    from harbor.models.task.task import Task

    # Harbor's instruction reader may remove leading blank lines from the raw file.
    return prompt_receipt(Task(task_dir=task).instruction)[1]


def require_remote() -> None:
    if sys.platform != "linux" or socket.gethostname() != HOSTNAME:
        raise RuntimeError("Native execution is authorized only on the A800 CPU/API host")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' explicitly; no GPU is allocated")


def validate_run_path(path: Path) -> None:
    if not path.is_absolute() or not path.is_relative_to(RUN_ROOT) or path == RUN_ROOT:
        raise ValueError("Use a new directory below the authorized data-volume campaign root")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink output roots are not allowed")


def storage_capacity(config: dict, run: Path) -> dict:
    targets = (run, run / "private", run / "process-cache", Path(config["docker_root"]))
    records = []
    for path in targets:
        existing = path
        while not existing.exists():
            existing = existing.parent
        disk = shutil.disk_usage(existing)
        records.append({
            "path": str(path), "device": existing.stat().st_dev,
            "free_bytes": disk.free, "minimum_free_bytes": config["minimum_free_disk_gib"] * 2**30,
        })
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    return {
        "at": utc_now(), "writable_filesystems": records,
        "available_memory_bytes": available,
        "system_root_free_bytes": shutil.disk_usage("/").free,
        "system_root_is_new_payload_target": False,
        "reserve_unchanged_gib": config["minimum_free_disk_gib"],
        "ready": all(row["free_bytes"] >= row["minimum_free_bytes"] for row in records)
        and available >= config["minimum_available_memory_gib"] * 2**30,
    }


def task_capacity(config: dict, run: Path, task_paths: list[Path]) -> dict:
    import tomllib

    tasks = [tomllib.loads((path / "task.toml").read_text()) for path in task_paths]
    return {
        **task_admission(
            storage_capacity(config, run), tasks,
            cpu_count=os.cpu_count() or 1, load_average=os.getloadavg(),
            minimum_available_memory_gib=config["minimum_available_memory_gib"],
        ),
        "at": utc_now(),
        "native_task_paths": [str(path) for path in task_paths],
    }


def check_sources(run: Path) -> None:
    for name, expected in load(run / "FROZEN_IDENTITY.json").items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen campaign identity changed: {name}")
    for relative, expected in load(run / "SOURCE_SHA256.json").items():
        if digest(run / "source" / relative) != expected:
            raise ValueError(f"Frozen campaign source changed: {relative}")
    pins = load(run / "NATIVE_RUNTIME.json")
    for filename, expected in pins["files"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"Pinned native runtime changed: {filename}")


def prepare(
    config: dict, track: str, run: Path, *, extra_source_files: tuple[str, ...] = (),
) -> None:
    require_remote()
    validate_config(config)
    config = {
        **config, "require_executed_native_tests": True,
        "deepseek_trial_namespace": "resolved_attempt_path_sha256_20",
    }
    validate_run_path(run)
    if track not in config["methods"]:
        raise ValueError("Only native baseline tracks may execute in this campaign")
    if track == "codex" and not config["codex"].get("provider_ledger"):
        raise ValueError("A new data-volume metering ledger is required before Codex launch")
    partition_path = Path(config["partition_dir"]) / "SPLIT_MANIFEST.json"
    if digest(partition_path) != config["partition_sha256"]:
        raise ValueError("Frozen partition bytes changed")
    manifest = load(partition_path)
    records = train_records(manifest, config)
    if importlib.metadata.version("harbor") != config["harbor_version"]:
        raise ValueError("Pinned Harbor 0.18.0 is required")
    import harbor

    harbor_root = Path(harbor.__file__).parent
    native_files = {
        str(harbor_root / name): digest(harbor_root / name)
        for name in ("verifier/verifier.py", "trial/trial.py", "environments/docker/docker.py",
                     "agents/installed/codex.py", "models/task/task.py")
    }
    if track == "codex":
        from phiagent.harness.codex_runtime import CODEX_REVISION, CODEX_SHA256

        native_files[config["codex"]["binary"]] = digest(Path(config["codex"]["binary"]))
        if (
            native_files[config["codex"]["binary"]] != CODEX_SHA256
            or config["codex"]["binary_sha256"] != CODEX_SHA256
            or config["codex"]["revision"] != CODEX_REVISION
            or native_files[str(harbor_root / "agents/installed/codex.py")]
            != config["codex"]["harbor_adapter_sha256"]
        ):
            raise ValueError("The official Codex binary or Harbor adapter pin differs")
    else:
        from integrations.deepseek_harness.harbor_agent import runtime_identity

        identity = runtime_identity(config["deepseek"]["runtime"])
        if (
            identity["upstream_commit"] != config["deepseek"]["official_revision"]
            or identity["upstream_tag"] != config["deepseek"]["official_tag"]
        ):
            raise ValueError("The official DeepSeek SDK identity differs")
        runtime_manifest = config["deepseek"]["runtime"]["manifest"]
        native_files[runtime_manifest] = digest(Path(runtime_manifest))
    observed_docker_root = subprocess.check_output(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True, timeout=30,
    ).strip()
    if Path(observed_docker_root).resolve() != Path(config["docker_root"]).resolve():
        raise ValueError("Actual Docker storage differs from the declared capacity guard")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    (run / "private").mkdir(mode=0o700)
    (run / "process-cache").mkdir(mode=0o700)
    write_json_atomic(run / "CONFIG.json", {**config, "track": track})
    write_json_atomic(run / "SHARED_PROMPT.json", prompt_manifest())
    shutil.copyfile(partition_path, run / "SPLIT_MANIFEST.json")
    write_json_atomic(run / "NATIVE_RUNTIME.json", {
        "harbor_version": config["harbor_version"], "harbor_revision": config["harbor_revision"],
        "files": native_files, "native_system_and_loop": "unchanged",
    })
    hashes = {}
    for name in sorted(set(source_files()) | set(extra_source_files)):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Additional frozen sources must be relative workspace files")
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = digest(destination)
    write_json_atomic(run / "SOURCE_SHA256.json", hashes)
    frozen_tasks, frozen_prompts = {}, {}
    for record in records:
        task = Path(config["task_root"]) / record["id"]
        files = task_file_hashes(task)
        if canonical_sha256(files) != record["payload_sha256"]:
            raise ValueError("TRAIN source differs from its frozen official record")
        frozen_tasks[record["id"]] = files
        frozen_prompts[record["id"]] = task_prompt_receipt(task, files)
    write_json_atomic(run / "TASK_SOURCE_SHA256.json", frozen_tasks)
    write_json_atomic(run / "TASK_PROMPT_RECEIPTS.json", frozen_prompts)
    write_json_atomic(run / "VERIFIER_ENVIRONMENT.json", config["verifier_environment"])
    write_json_atomic(run / "PROVENANCE.json", {
        "at": utc_now(), "hostname": socket.gethostname(), "python": sys.version,
        "executable": sys.executable, "seed": config["execution_seed"], "argv": sys.argv,
        "git_head": read_command(["git", "rev-parse", "HEAD"], ROOT),
        "scoped_git_status": read_command(["git", "status", "--short", "--", *hashes], ROOT),
        "upload_git_state": load(ROOT / "UPLOAD_GIT_STATE.json")
        if (ROOT / "UPLOAD_GIT_STATE.json").is_file() else None,
        "packages": sorted((dist.metadata["Name"], dist.version)
                           for dist in importlib.metadata.distributions()),
    })
    write_json_atomic(run / "RESOURCE_INVENTORY.json", resource_inventory())
    write_json_atomic(run / "CAPACITY.json", storage_capacity(config, run))
    write_json_atomic(run / "LEASE.json", {
        "hostname": socket.gethostname(), "selected_gpus": [], "CUDA_VISIBLE_DEVICES": "",
        "resource_class": "CPU Docker tasks + approved DeepSeek API",
        "maximum_workers": config["maximum_workers"], "owner_run": str(run),
        "successor_task_id": "complete18-train-then-capability-gated-chisel",
        "old_campaigns_mutated": False,
    })
    write_json_atomic(run / "FROZEN_IDENTITY.json", {
        name: digest(run / name) for name in (
            "CONFIG.json", "SHARED_PROMPT.json", "SPLIT_MANIFEST.json",
            "NATIVE_RUNTIME.json", "SOURCE_SHA256.json", "TASK_SOURCE_SHA256.json",
            "TASK_PROMPT_RECEIPTS.json", "VERIFIER_ENVIRONMENT.json",
        )
    })


def count(value: Any) -> bool:
    return type(value) is int and value >= 0


def usage_record(track: str, raw: dict) -> dict:
    if track == "codex":
        complete = raw.get("usage_status") == "MEASURED"
        known = raw.get("known_usage") or {}
        usage = {field: known.get(field) if complete else None for field in TOKEN_FIELDS}
        evidence = {
            "provider_call_count": raw.get("calls"),
            "provider_receipts": raw.get("receipts", []),
            "unresolved_call_ids": raw.get("unresolved_call_ids", []),
            "cached_input_tokens": raw.get("cached_input_tokens"),
            "reasoning_output_tokens": raw.get("reasoning_output_tokens"),
            "failed_calls": raw.get("failed_calls", []),
        }
        if complete and raw.get("provider_models") != ["deepseek-flash"]:
            raise ValueError("Native provider model differs from the common actor")
    elif track == "deepseek":
        complete = raw.get("deepseek_usage_complete") is True
        usage = canonical_usage(raw.get("deepseek_usage"), complete)
        known = raw.get("deepseek_known_reported_usage") or {}
        if "prompt_tokens_including_cache" in known:
            known = {**known, "input_tokens": known["prompt_tokens_including_cache"]}
        evidence = {
            "provider_call_count": raw.get("provider_attempt_count"),
            "sdk_worker_dir": raw.get("sdk_worker_dir"),
            "private_artifacts": raw.get("private_artifacts"),
        }
    else:
        raise ValueError("Unsupported native track")
    if complete and (
        not all(count(usage.get(field)) for field in TOKEN_FIELDS)
        or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
    ):
        raise ValueError("Native usage is missing counters or double-counts cached input")
    return {
        "usage_status": "MEASURED" if complete else "UNKNOWN",
        "usage": {field: usage.get(field) for field in TOKEN_FIELDS},
        "known_subtotal": {field: known.get(field) for field in TOKEN_FIELDS},
        "cost_usd": None, "cost_usd_status": "provider_did_not_report_usd",
        **evidence, "cumulative_token_limit": None,
    }


def reduce_usage(attempts: list[dict]) -> dict:
    by_id = {}
    for row in attempts:
        key = row["attempt_id"]
        if key in by_id and by_id[key] != row:
            raise ValueError("Conflicting evidence for one actual native attempt")
        by_id[key] = row
    rows = list(by_id.values())
    unknown = [row["attempt_id"] for row in rows if row["cost"]["usage_status"] != "MEASURED"]
    totals = {}
    for field in TOKEN_FIELDS:
        values = [
            row["cost"]["usage"].get(field)
            if row["cost"]["usage_status"] == "MEASURED"
            else row["cost"]["known_subtotal"].get(field)
            for row in rows
        ]
        totals[field] = {
            "known_subtotal": sum(value for value in values if count(value)),
            "complete_total": sum(values) if all(count(value) for value in values) and not unknown
            else None,
        }
    return {
        "unique_native_attempts": len(rows), "unknown_attempt_ids": unknown,
        "tokens": totals, "cost_usd": None if rows else 0,
        "accounting_unit": "Each actual native attempt once, including failed/dispatched calls",
        "not_invoked": {"optimizer": 0, "validation": 0, "test": 0, "model_judge": 0},
        "retry_policy": "No automatic actor replay; late receipts update the same attempt identity",
    }


def capability_gate(config: dict, track: str, train_ids: list[str], attempts: list[dict]) -> dict:
    validate_config(config)
    if track not in config["methods"] or len(train_ids) != 18 or len(set(train_ids)) != 18:
        raise ValueError("Capability gate needs one native track and all18 unique TRAIN tasks")
    rows = {}
    for row in attempts:
        if (
            row["track"] != track or row["phase"] != "train" or row["task_id"] not in train_ids
            or row["shared_prompt_sha256"] != config["shared_prompt_sha256"]
            or row["partition_sha256"] != config["partition_sha256"]
            or row["task_id"] in rows
        ):
            raise ValueError("Mismatched, duplicate, historical, or held-out gate observation")
        rows[row["task_id"]] = row
    scored = [row for row in rows.values() if row["status"] == "SCORED"]
    if any(type(row["native_score"]) not in (int, float) or row["native_score"] not in (0, 1)
           for row in scored):
        raise ValueError("Capability gate requires genuine binary native verdicts")
    successes = sum(row["native_score"] == 1 for row in scored)
    full = len(scored) == 18
    passed = full and successes >= config["capability_gate"]["minimum_successes"]
    return {
        "status": "PASSED" if passed else ("INSUFFICIENT_CAPABILITY" if full else "WAITING_FULL_TRAIN"),
        "track": track, "phase": "train", "native_scored": len(scored), "required": 18,
        "native_successes": successes, "minimum_successes": 9,
        "success_rate": successes / 18 if full else None,
        "rsi_ready": passed, "automatic_rsi_execution": False,
        "shared_prompt_version": PROMPT_VERSION, "shared_prompt_sha256": PROMPT_SHA256,
        "partition_sha256": config["partition_sha256"],
        "native_gate_counted_once": True,
        "first_native_gate": rows.get(train_ids[0]),
        "comparison": "CHISEL must descend from this same strengthened native parent",
        "scope": "TRAIN capability, not final held-out accuracy or prompt-improvement evidence",
    }


def require_rsi_parent(gate: dict, config: dict, track: str) -> None:
    validate_config(config)
    if (
        gate.get("status") != "PASSED" or gate.get("rsi_ready") is not True
        or gate.get("track") != track or gate.get("phase") != "train"
        or gate.get("native_scored") != 18 or gate.get("native_successes", -1) < 9
        or gate.get("shared_prompt_sha256") != config["shared_prompt_sha256"]
        or gate.get("partition_sha256") != config["partition_sha256"]
    ):
        raise ValueError("RSI requires the complete capable TRAIN parent with the identical prompt")


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def process_live(receipt: dict) -> bool:
    return bool(receipt.get("start_ticks")) and (
        process_identity(receipt["pid"]) == receipt["start_ticks"]
    )


def settled_verifier_pending(row: dict) -> bool:
    usage = row.get("cost", {}).get("usage")
    return (
        row.get("phase") == "train" and row.get("status") == "BLOCKED"
        and row.get("native_score") is None
        and row.get("error") == "Official verifier did not supply its unique native reward"
        and (row.get("native_exception") or {}).get("exception_type")
        in ("AgentTimeoutError", "VerifierTimeoutError")
        and row.get("cost", {}).get("usage_status") == "MEASURED"
        and isinstance(usage, dict)
        and all(type(usage.get(key)) is int and usage[key] >= 0 for key in TOKEN_FIELDS)
        and usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"]
        and usage["total_tokens"] > 0
        and isinstance(row.get("native_result"), str)
        and isinstance(row.get("native_result_sha256"), str)
        and len(row["native_result_sha256"]) == 64
    )


def refresh_retained(previous: dict[str, dict], updated: tuple[dict, ...]) -> dict[str, dict]:
    current = {row["task_id"]: deepcopy(row) for row in updated}
    if len(current) != len(updated) or set(current) != set(previous):
        raise ValueError("Verifier settlement must preserve every retained native task")
    for task, row in current.items():
        original = previous[task]
        if any(row.get(key) != original.get(key) for key in (
            "attempt_id", "task_id", "track", "phase", "cost",
            "shared_prompt_sha256", "partition_sha256",
        )):
            raise ValueError("Verifier settlement changed the paid actor or its exact-once cost")
        if original["status"] == "SCORED" and (
            row.get("status") != "SCORED" or row.get("native_score") != original["native_score"]
        ):
            raise ValueError("A settled native score cannot be replaced or lost")
        if row["status"] != "SCORED" and not settled_verifier_pending(row):
            raise ValueError("An unresolved actor cannot be reclassified as a verifier-only wait")
    return current


def deepseek_trial_name(directory: Path) -> str:
    from scripts.recover_terminal_verifier import container_name

    if not directory.is_absolute():
        raise ValueError("Native trial identity requires an absolute attempt directory")
    return container_name("ds-cap-", directory)


def deepseek_trial_config(job: dict, directory: Path) -> dict:
    from scripts.recover_terminal_verifier import cached_trial_config

    trial_name = deepseek_trial_name(directory)
    if load(directory / "TRIAL_ID.json")["trial_id"] != trial_name:
        raise ValueError("Native trial identity differs from its frozen attempt directory")
    return cached_trial_config(job, trial_name)


def trial_command(config: dict, track: str, run: Path, directory: Path, task_id: str) -> list[str]:
    task = directory / "task-source" / task_id
    shutil.copytree(Path(config["task_root"]) / task_id, task)
    expected_files = load(run / "TASK_SOURCE_SHA256.json")[task_id]
    if task_file_hashes(task) != expected_files:
        raise ValueError("Native task bytes changed while creating the isolated source snapshot")
    prompts_path = run / "TASK_PROMPT_RECEIPTS.json"
    if prompts_path.is_file() and (
        task_prompt_receipt(task, expected_files) != load(prompts_path).get(task_id)
    ):
        raise ValueError("Pinned Harbor task instruction changed before native dispatch")
    source = run / "source"
    if track == "codex":
        runtime = config["codex"]
        trial_id = "cap-" + hashlib.sha256(str(directory).encode()).hexdigest()[:32]
        write_json_atomic(directory / "TRIAL_ID.json", {"trial_id": trial_id})
        command = [
            runtime["python"], "-u", str(source / "scripts/run_codex_terminal.py"),
            "--task", str(task), "--partition", str(run / "SPLIT_MANIFEST.json"),
            "--phase", "train", "--output", str(directory / "native-run"),
            "--trial-id", trial_id, "--provider-url", runtime["provider_url"],
            "--provider-ledger", runtime["provider_ledger"],
            "--relay-secret", runtime["relay_secret_file"], "--codex-binary", runtime["binary"],
            "--method", "codex", "--repetition-seed", str(config["execution_seed"]),
            "--common-prompt-version", config["shared_prompt_version"],
            "--common-prompt-sha256", config["shared_prompt_sha256"],
            "--verifier-environment", str(run / "VERIFIER_ENVIRONMENT.json"),
            "--provider-settlement-timeout", str(runtime["provider_settlement_timeout_seconds"]),
        ]
        if task_id == "sam-cell-seg":
            command += ["--sam-verifier-recovery", config["sam_verifier_recovery"]]
        cache = config.get("native_verifier_caches", {}).get(task_id)
        if cache:
            command += ["--native-verifier-cache", cache]
        return command
    native = config["deepseek"]
    write_json_atomic(directory / "TRIAL_ID.json", {
        "trial_id": deepseek_trial_name(directory),
        "namespace": "resolved_attempt_path_sha256_20",
        "attempt_directory": str(directory),
    })
    kwargs = {
        "runner_python": native["sdk_python"],
        "runner_script": str(source / "integrations/deepseek_harness/runner.py"),
        "runtime": native["runtime"], "api_key_file": native["api_key_file"],
        "private_root": str(run / "private" / directory.name),
        "max_tokens": native["max_output_tokens"], "seed": config["execution_seed"],
        "arm": "baseline", "chisel": None, "source_revision": native["official_revision"],
        "shared_prompt_version": config["shared_prompt_version"],
        "shared_prompt_sha256": config["shared_prompt_sha256"], "benchmark": BENCHMARK,
    }
    if config.get("provider_durable_capture") is True:
        kwargs["provider_durable_capture"] = True
    job = build_native_job(directory, task, kwargs)
    job["agents"][0]["import_path"] = "scripts.industrial_terminal_agents:SharedPromptDeepSeek"
    job["verifier"]["env"] = config["verifier_environment"]
    job["environment"].update(delete=False, kwargs={"keep_containers": True})
    write_json_atomic(directory / "HARBOR_JOB.json", job)
    return [
        native["python"], "-u", str(source / "scripts/run_industrial_capability_baselines.py"),
        "--run-dir", str(run), "--native-directory", str(directory),
    ]


def sam_cache_environment(original: dict[str, str], cache_url: str) -> dict[str, str]:
    from scripts.recover_codex_sam_verifier import dependency_env

    return {
        **dependency_env(cache_url), **original,
        "BASH_ENV": "/opt/sam-recovery/cache.env", "UV_OFFLINE": "1",
    }


async def deepseek_native(run: Path, directory: Path) -> None:
    require_remote()
    config = load(run / "CONFIG.json")
    validate_config(config)
    check_sources(run)
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial, TrialEvent
    from scripts.recover_terminal_verifier import (
        install_cache, record_owned_container,
        release_stopped_trial_network, remove_scored_container, verified_cache,
    )

    job = load(directory / "HARBOR_JOB.json")
    task = Path(job["tasks"][0]["path"])
    expected = load(run / "TASK_SOURCE_SHA256.json")[task.name]
    if task_file_hashes(task) != expected:
        raise ValueError("Frozen native task source changed")
    native_config = deepseek_trial_config(job, directory)
    cache = config.get("native_verifier_caches", {}).get(task.name)
    sam = config.get("sam_verifier_recovery") if task.name == "sam-cell-seg" else None
    if cache:
        prepared = verified_cache(Path(cache), task)
        native_config["verifier"]["env"] = {
            **native_config["verifier"].get("env", {}),
            **prepared["verifier_environment"], "UV_OFFLINE": "1",
        }
    if sam:
        from scripts.recover_codex_sam_verifier import verified_cache as sam_cache

        _, prepared = sam_cache(Path(sam))
        native_config["verifier"]["env"] = sam_cache_environment(
            native_config["verifier"].get("env", {}), prepared["dependency_cache_url"],
        )
    write_json_atomic(directory / "NATIVE_CONFIG.json", native_config)
    trial = await Trial.create(TrialConfig.model_validate(native_config))

    async def record_container(event):
        await record_owned_container(trial.agent_environment, directory)

    async def install_dependencies(event):
        if config.get("sar_public_verifier_cache") is True:
            from scripts.prepare_sar_public_verifier_cache import prepare
            updates = await prepare(trial.agent_environment, directory, task)
            trial.config.verifier.env.update(updates)
            write_json_atomic(directory / "VERIFIER_ENV_AFTER_CACHE.json", trial.config.verifier.env)
        if cache:
            await install_cache(trial.agent_environment, Path(cache), task, directory)
        if sam:
            from scripts.recover_codex_sam_verifier import prepare_verifier_cache

            await prepare_verifier_cache(
                trial.agent_environment, Path(sam), directory,
                transport_environment=native_config["verifier"]["env"],
            )

    trial.add_hook(TrialEvent.AGENT_START, record_container)
    trial.add_hook(TrialEvent.VERIFICATION_START, install_dependencies)
    result = (await trial.run()).model_dump(mode="json")
    write_json_atomic(directory / "native_result.json", result)
    try:
        await release_stopped_trial_network(directory)
        if config.get("require_executed_native_tests") is True:
            verdict = executed_native_verdict(result, directory / "harbor/native")
            write_json_atomic(directory / "VERIFIER_ACCEPTANCE.json", verdict)
        else:
            native_reward(result)
        await remove_scored_container(directory)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        write_json_atomic(directory / "RETENTION.json", {
            "unscored_workspace_preserved": True, "cleanup_or_verdict_error": str(exc),
        })


def harvest(config: dict, track: str, run: Path, directory: Path, task_id: str) -> dict:
    native_path = directory / ("native-run/native_result.json" if track == "codex" else "native_result.json")
    if not native_path.is_file():
        pattern = "native-run/native/*/result.json" if track == "codex" else "harbor/native/*/result.json"
        candidates = list(directory.glob(pattern))
        if len(candidates) == 1:
            native_path = candidates[0]
    result = load(native_path) if native_path.is_file() else {}
    receipt_paths = list(directory.glob("**/SHARED_PROMPT.json"))
    valid_prompt = False
    prompt_error = "Native adapter did not record the shared prompt"
    if len(receipt_paths) == 1:
        receipt = load(receipt_paths[0])
        prompts_path = run / "TASK_PROMPT_RECEIPTS.json"
        if prompts_path.is_file():
            task_matches = receipt == load(prompts_path).get(task_id)
        else:
            expected = load(run / "TASK_SOURCE_SHA256.json")[task_id]["instruction.md"]
            task_matches = receipt.get("original_task_sha256") == expected
        valid_prompt = (
            receipt.get("sha256") == config["shared_prompt_sha256"]
            and task_matches
            and receipt.get("shared_block_count") == 1
            and receipt.get("original_task_preserved") is True
        )
        prompt_error = None if valid_prompt else "Native shared prompt/task hash mismatch"
    try:
        score = native_reward(result)
        if config.get("require_executed_native_tests") is True:
            native_directory = directory / ("native-run/native" if track == "codex" else "harbor/native")
            verdict = executed_native_verdict(result, native_directory)
            write_json_atomic(directory / "VERIFIER_ACCEPTANCE.json", verdict)
        if not valid_prompt:
            raise ValueError(prompt_error)
        if result.get("task_name") != task_id:
            raise ValueError("Native scored task identity differs")
        status, error = "SCORED", None
    except ValueError as exc:
        score, status, error = None, "BLOCKED", str(exc)
    if track == "codex":
        identity_path = directory / "TRIAL_ID.json"
        raw_usage = usage_summary(
            Path(config["codex"]["provider_ledger"]), load(identity_path)["trial_id"],
        ) if identity_path.is_file() else {}
    else:
        raw_usage = (result.get("agent_result") or {}).get("metadata") or {}
    call_costs = []
    usage_reconciliation = None
    if track == "codex":
        for path in raw_usage.get("receipts", []):
            receipt_path = Path(path)
            if not receipt_path.is_file():
                continue
            receipt = load(receipt_path)
            call_costs.append({
                "call_id": str(receipt_path.parent), "role": "actor",
                "usage": {key: (receipt.get("usage") or {}).get(key) for key in TOKEN_FIELDS},
                "usage_status": receipt.get("usage_status"),
                "model": receipt.get("model"), "http_status": receipt.get("http_status"),
                "failure_classification": receipt.get("failure_classification"),
                "receipt": path, "receipt_sha256": digest(receipt_path),
            })
    elif raw_usage.get("sdk_worker_dir"):
        from integrations.deepseek_harness.harbor_agent import (
            normalize_runner_usage, recover_runner_accounting,
        )

        worker = Path(raw_usage["sdk_worker_dir"])
        if not worker.resolve().is_relative_to((run / "private").resolve()):
            raise ValueError("SDK accounting escaped this new run's private workspace")
        accounting = recover_runner_accounting(worker)
        usage_reconciliation = {
            "source": accounting.get("source"),
            "original_reported_usage": raw_usage.get("deepseek_usage"),
            "original_reported_complete": raw_usage.get("deepseek_usage_complete"),
            "original_provider_attempt_count": raw_usage.get("provider_attempt_count"),
            "unknown_reasons": accounting.get("unknown_reasons"),
        }
        raw_usage = {
            **raw_usage,
            "deepseek_usage_complete": accounting["complete"],
            "deepseek_usage": normalize_runner_usage(accounting.get("usage")),
            "deepseek_known_reported_usage": normalize_runner_usage(accounting.get("known_reported_usage")),
            "provider_attempt_count": accounting["attempt_count"],
            "usage_unknown_reasons": accounting["unknown_reasons"],
        }
        for attempt in accounting.get("attempts", []):
            normalized = normalize_runner_usage(attempt.get("usage"))
            call_costs.append({
                "call_id": str(worker) + "#" + attempt["attempt_id"], "role": "actor",
                "purpose": attempt.get("start", {}).get("purpose"),
                "model": attempt.get("start", {}).get("model"),
                "usage": {**normalized, "input_tokens": normalized["prompt_tokens_including_cache"]},
                "ended": attempt.get("ended"),
                "finish_reason": (attempt.get("finish") or {}).get("reason"),
                "stream_receipts": str(worker / "llm-streams.jsonl"),
            })
    if len({row["call_id"] for row in call_costs}) != len(call_costs):
        raise ValueError("Duplicate actual provider call in accounting evidence")
    write_json_atomic(directory / "PROVIDER_CALLS.json", {
        "calls": call_costs, "native_actor_attempt_id": str(directory),
        "roles": {"actor": len(call_costs), "optimizer": 0, "model_judge": 0},
        "wrapper_retry_calls": 0,
        "native_retry_attribution": "Retained per-call finish/usage and raw native receipts; never imputed",
        "aggregate_policy": "Details of the same actor total, not an additional cost to add",
    })
    row = {
        "attempt_id": str(directory), "track": track, "phase": "train", "task_id": task_id,
        "status": status, "native_score": score, "error": error,
        "native_result": str(native_path) if native_path.is_file() else None,
        "native_result_sha256": digest(native_path) if native_path.is_file() else None,
        "shared_prompt_sha256": config["shared_prompt_sha256"],
        "shared_prompt_receipts": [str(path) for path in receipt_paths],
        "partition_sha256": config["partition_sha256"],
        "cost": usage_record(track, raw_usage), "historical_outcome_reused": False,
        "provider_call_ledger": str(directory / "PROVIDER_CALLS.json"),
        "stage": config.get("execution_stage", "training_baseline_capability"),
        "native_exception": result.get("exception_info"),
        "usage_reconciliation": usage_reconciliation,
    }
    write_json_atomic(directory / "RESULT.json", row)
    return row


def report(config: dict, track: str, run: Path, train_ids: list[str], attempts: list[dict], active: dict) -> dict:
    gate = capability_gate(config, track, train_ids, attempts)
    terminal = len(attempts) == 18 and not active
    summary = {
        "schema": SCHEMA, "at": utc_now(), "pid": os.getpid(), "track": track,
        "status": ("COMPLETE" if gate["native_scored"] == 18 else "BLOCKED") if terminal else "RUNNING",
        "phase": "train", "coverage": {"scored": gate["native_scored"], "expected": 18,
                                       "successes": gate["native_successes"]},
        "shared_prompt": prompt_manifest(), "capability_gate": gate,
        "attempts": attempts, "active": active, "accounting": reduce_usage(attempts),
        "next_stage": config["next_stage"], "automatic_rsi_execution": False,
        "first_gate_is_part_of_full_train": True, "cumulative_token_limit": None,
    }
    write_json_atomic(run / "REPORT.json", summary)
    write_json_atomic(run / "CAPABILITY_GATE.json", gate)
    write_json_atomic(run / "QUEUE.json", {
        "phase": "train", "train_order": train_ids,
        "pending": [task for task in train_ids if task not in {
            row["task_id"] for row in attempts
        } and task not in active],
        "active": active, "terminal": terminal,
        "successor": {"stage": "chisel_train", "ready": gate["rsi_ready"],
                      "required_parent_receipt": str(run / "CAPABILITY_GATE.json"),
                      "common_prompt_sha256": PROMPT_SHA256,
                      "auto_launch": False, "no_test_access": True},
    })
    return summary


def controller(
    run: Path, *,
    command_builder: Callable[[dict, str, Path, Path, str], list[str]] | None = None,
    pause_on_failure: bool = False,
    report_observer: Callable[[dict], None] | None = None,
    stop_file: Path | None = None,
    retained_attempts: tuple[dict, ...] = (),
    admission_check: Callable[[Path, str, list[str]], dict] | None = None,
    defer_settled_verifier_failures: bool = False,
    retained_attempts_loader: Callable[[], tuple[dict, ...]] | None = None,
) -> dict:
    if type(pause_on_failure) is not bool or type(defer_settled_verifier_failures) is not bool:
        raise ValueError("The native submission failure policy must be explicit")
    require_remote()
    config = load(run / "CONFIG.json")
    validate_config(config)
    track = config["track"]
    train_ids = [row["id"] for row in train_records(load(run / "SPLIT_MANIFEST.json"), config)]
    retained = {row["task_id"]: deepcopy(row) for row in retained_attempts}
    if len(retained) != len(retained_attempts):
        raise ValueError("A retained TRAIN task may appear only once")
    if retained:
        capability_gate(config, track, train_ids, list(retained.values()))
        for row in retained.values():
            pending_verifier = (
                defer_settled_verifier_failures and settled_verifier_pending(row)
                and row.get("verifier_pending") is True
            )
            if (row["cost"]["usage_status"] != "MEASURED"
                    or row["status"] != "SCORED" and not pending_verifier):
                raise ValueError("Only source-verified, scored and metered attempts may be retained")
    processes = {}
    last_audit = 0.0
    lock = (run / "CONTROLLER.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json_atomic(run / "OWNER.json", {
        "pid": os.getpid(), "start_ticks": process_identity(os.getpid()),
        "hostname": socket.gethostname(), "at": utc_now(), "run": str(run),
    })
    while True:
        check_sources(run)
        if retained_attempts_loader is not None:
            retained = refresh_retained(retained, retained_attempts_loader())
        if time.monotonic() - last_audit >= config["capacity_audit_seconds"]:
            inventory = resource_inventory()
            write_json_atomic(run / "RESOURCE_LATEST.json", inventory)
            with (run / "RESOURCE_HISTORY.jsonl").open("a") as stream:
                stream.write(json.dumps(inventory, sort_keys=True) + "\n")
            last_audit = time.monotonic()
        capacity = storage_capacity(config, run)
        write_json_atomic(run / "CAPACITY.json", capacity)
        attempts, active, pending = [], {}, []
        for index, task_id in enumerate(train_ids):
            directory = run / "tasks" / f"{index:02d}-{task_id}"
            if task_id in retained:
                if directory.exists():
                    raise ValueError("A retained task also has a new attempt directory")
                attempts.append(deepcopy(retained[task_id]))
                continue
            if not directory.exists():
                pending.append((task_id, directory))
                continue
            process = processes.get(task_id)
            if process is not None:
                exit_code = process.poll()
                if exit_code is not None and not (directory / "PROCESS_EXIT.json").exists():
                    write_json_atomic(directory / "PROCESS_EXIT.json", {
                        "returncode": exit_code, "at": utc_now(),
                    })
            receipt_path = directory / "PROCESS.json"
            receipt = load(receipt_path) if receipt_path.is_file() else {}
            if receipt and process_live(receipt):
                active[task_id] = {**receipt, "directory": str(directory)}
            else:
                attempts.append(harvest(config, track, run, directory, task_id))
        summary = report(config, track, run, train_ids, attempts, active)
        if report_observer is not None:
            report_observer(deepcopy(summary))
        stopped = stop_file is not None and stop_file.exists()
        failures = [
            {"task_id": row["task_id"], "attempt_id": row["attempt_id"],
             "status": row["status"], "usage_status": row["cost"]["usage_status"],
             "error": row.get("error")}
            for row in attempts
            if (row["status"] != "SCORED" or row["cost"]["usage_status"] != "MEASURED")
            and not (defer_settled_verifier_failures and settled_verifier_pending(row))
        ] if pause_on_failure else []
        deferred = [
            row for row in attempts
            if defer_settled_verifier_failures and settled_verifier_pending(row)
        ]
        if deferred:
            write_json_atomic(run / "VERIFIER_QUEUE.json", {
                "at": utc_now(), "pending": deferred, "additional_actor_calls": 0,
                "native_scores_not_imputed": True,
                "optimizer_and_heldout_gate": "All native verdicts and provider costs required",
                "independent_unstarted_train_tasks_may_continue": True,
            })
        if failures or stopped:
            write_json_atomic(run / "SUBMISSION_GATE.json", {
                "at": utc_now(), "status": "USER_STOP_DRAINING" if stopped else "FROZEN_NATIVE_FAILURE",
                "failures": failures, "active_actors_draining": active,
                "pending_tasks_preserved": [task for task, _ in pending],
                "actor_replay_permitted": False,
            })
            if not active:
                summary = {
                    **summary, "status": "STOPPED" if stopped else "BLOCKED",
                    "submissions_frozen": True,
                    "pending_tasks_preserved": [task for task, _ in pending],
                }
                write_json_atomic(run / "REPORT.json", summary)
                write_json_atomic(run / "LEASE_RELEASED.json", {
                    "at": utc_now(), "pid": os.getpid(), "status": summary["status"],
                    "selected_gpus": [], "successor": "Native failure diagnosis; no actor replay",
                })
                return summary
        if not pending and not active:
            write_json_atomic(run / "LEASE_RELEASED.json", {
                "at": utc_now(), "pid": os.getpid(), "status": summary["status"],
                "selected_gpus": [], "successor": str(run / "CAPABILITY_GATE.json"),
            })
            return summary
        scored_gate = any(row["status"] == "SCORED" for row in attempts)
        workers = config["maximum_workers"] if scored_gate else 1
        while pending and len(active) < workers and capacity["ready"] and not failures and not stopped:
            if stop_file is not None and stop_file.exists():
                break
            if admission_check is not None or workers > 2:
                admission = (
                    admission_check(run, pending[0][0], list(active))
                    if admission_check is not None else task_capacity(
                        config, run,
                        [Path(config["task_root"]) / task for task in [*active, pending[0][0]]],
                    )
                )
                write_json_atomic(run / "NEXT_TASK_ADMISSION.json", admission)
                if admission.get("ready") is not True:
                    break
            task_id, directory = pending.pop(0)
            directory.mkdir(parents=True, exist_ok=False)
            try:
                command = (command_builder or trial_command)(
                    config, track, run, directory, task_id,
                )
                cache = directory / "process-cache"
                cache.mkdir()
                write_json_atomic(directory / "COMMAND.json", {
                    "argv": command, "cwd": str(directory), "CUDA_VISIBLE_DEVICES": "",
                    "TMPDIR": str(cache), "cumulative_token_limit": None,
                })
                with (directory / "process.log").open("xb") as log:
                    process = subprocess.Popen(
                        command, cwd=directory, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                        env={**os.environ, "PYTHONPATH": str(run / "source"),
                             "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(cache),
                             "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                processes[task_id] = process
                receipt = {"pid": process.pid, "start_ticks": process_identity(process.pid),
                           "started_at": utc_now(), "controller_pid": os.getpid()}
                write_json_atomic(directory / "PROCESS.json", receipt)
                active[task_id] = {**receipt, "directory": str(directory)}
            except Exception as exc:
                write_json_atomic(directory / "LAUNCH_FAILURE.json", {
                    "at": utc_now(), "type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(), "actor_replayed": False,
                })
                break
        summary = report(config, track, run, train_ids, attempts, active)
        if report_observer is not None:
            report_observer(deepcopy(summary))
        time.sleep(config["heartbeat_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--track", choices=("codex", "deepseek"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--provider-ledger", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_remote()
    run = args.run_dir.absolute()
    validate_run_path(run)
    os.umask(0o077)
    if args.native_directory:
        if not args.native_directory.is_relative_to(run / "tasks"):
            raise ValueError("A native trial must belong to its frozen campaign")
        asyncio.run(deepseek_native(run, args.native_directory))
        return 0
    if not args.resume and not args.worker:
        if not args.config or not args.track:
            parser.error("A new run requires --config and --track")
        config = load(args.config)
        if args.provider_ledger:
            config["codex"]["provider_ledger"] = str(args.provider_ledger.absolute())
        prepare(config, args.track, run)
    else:
        check_sources(run)
    if args.detach and not args.worker:
        command = [
            sys.executable, "-u", str(run / "source" / "scripts/run_industrial_capability_baselines.py"),
            "--run-dir", str(run), "--worker",
        ]
        with (run / "controller.log").open("ab") as log:
            process = subprocess.Popen(
                command, cwd=run, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
                env={**os.environ, "TMPDIR": str(run / "process-cache"), "CUDA_VISIBLE_DEVICES": "",
                     "PYTHONDONTWRITEBYTECODE": "1"},
            )
        write_json_atomic(run / "LAUNCH.json", {
            "pid": process.pid, "start_ticks": process_identity(process.pid),
            "argv": command, "at": utc_now(), "detached": True,
        })
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(f"Controller startup failed; inspect {run / 'controller.log'}")
        print(json.dumps({"status": "STARTED", "pid": process.pid, "run": str(run)}), flush=True)
        return 0
    try:
        result = controller(run)
    except Exception as exc:
        write_json_atomic(run / "CONTROLLER_FAILURE.json", {
            "at": utc_now(), "type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "actions_replayed": False,
            "resume": "Fix a diagnosed infrastructure blocker, then --resume --detach on frozen source",
        })
        raise
    print(json.dumps({key: result[key] for key in ("status", "track", "coverage")}))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
