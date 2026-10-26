#!/usr/bin/env python3
"""Exactly one fresh Codex attempt for each authorized unscored infrastructure failure.

These are prospective replacement attempts, not restored workspaces. The native
configuration comes from the original trial; only isolated paths/IDs and verified
dependency-cache environment change. All original paid attempts remain ancestors.
No native controller, common prompt, scorer, model, or context policy is edited.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
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
import traceback

PARENT = Path(
    "/path/to/data/phiagent/work/deepseek-harness/industrial-capability/20260930-v1/train-codex-v1"
)
DATA = Path("/path/to/data/phiagent/work/harness-terminal-recovery")
TASKS = {"sqlite-with-gcov": "07-sqlite-with-gcov",
         "adaptive-rejection-sampler": "13-adaptive-rejection-sampler"}
PROMPT = "f2755a7b715e692b6fe794cdf7b316dfb58b301759dade0bf3f67c2ae02e57e5"
PARTITION = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
SCHEMA = "codex-capability-prospective-infrastructure-continuation/v1"
ADAPTIVE_CACHE = DATA / (
    "20260930-v1/adaptive-candidate-recovery-v1/dependencies/"
    "adaptive-rejection-sampler/PREPARED.json"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".writing")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def original_eligible(proof: dict) -> None:
    need(proof.get("task_id") in TASKS, "Task was not authorized for a fresh actor attempt")
    need(proof.get("original_native_tests_executed") is False,
         "An already-evaluated task cannot receive a replacement")
    need(proof.get("original_native_reward") == 0
         and proof.get("dependency_failure_confirmed") is True,
         "Only the specified permanently unscored dependency failures may be replaced")
    need(proof.get("original_actor_exited") is True and proof.get("original_actor_active") is False,
         "An original actor still owns this task")
    need(proof.get("workspace_absent") is True and proof.get("snapshot_image_ids") == []
         and proof.get("retained_workspace_files") == [], "Retained actor work must not be replayed")
    usage = proof.get("settled_usage") or {}
    need(usage.get("usage_status") == "MEASURED" and usage.get("calls", 0) > 0
         and usage.get("unresolved_call_ids") == []
         and usage.get("provider_models") == ["deepseek-flash"]
         and proof.get("all_provider_receipts_finished") is True,
         "Original provider requests are incomplete, ambiguous, or still in flight")
    need(proof.get("shared_prompt_sha256") == PROMPT
         and proof.get("partition_sha256") == PARTITION, "Different original parent identity")


def replacement_config(template: dict, task: Path, attempt: Path, trial_id: str,
                       cache_environment: dict) -> dict:
    need(template["agent"]["model_name"] == "deepseek-flash", "Do not change the actor backbone")
    kwargs = template["agent"]["kwargs"]
    need(template["agent"]["import_path"] == "scripts.industrial_terminal_agents:SharedPromptCodex"
         and kwargs.get("shared_prompt_sha256") == PROMPT and kwargs.get("residual") == "",
         "Replacement must use the identical strengthened native parent")
    need(template.get("timeout_multiplier") == 1, "Keep native task limits")
    result = deepcopy(template)
    result["task"]["path"] = str(task)
    result["trial_name"] = trial_id
    result["trials_dir"] = str(attempt / "native-run/native")
    result["agent"]["kwargs"]["trial_id"] = trial_id
    result["verifier"]["env"] = {
        **template["verifier"].get("env", {}), **cache_environment, "UV_OFFLINE": "1",
    }
    return result


def validate_plan(plan: dict, run: Path) -> None:
    need(plan.get("schema") == SCHEMA and plan.get("track") == "codex"
         and plan.get("phase") == "train", "Not a declared Codex TRAIN continuation")
    need(plan.get("original_parent_run") == str(PARENT)
         and plan.get("parent_run") == str(run), "Different continuation parent")
    need(plan.get("shared_prompt_sha256") == PROMPT and plan.get("partition_sha256") == PARTITION,
         "Different shared prompt or partition")
    need(plan.get("maximum_actor_attempts_per_task") == 1
         and plan.get("selection_policy") == "one_prospective_replacement_not_best_of",
         "Only one prospective attempt is authorized")
    need(set(plan.get("tasks", {})) == set(TASKS), "The replacement task set changed")
    for task_id, item in plan["tasks"].items():
        need(item["original_attempt"] == str(PARENT / "tasks" / TASKS[task_id]),
             "Different original task attempt")
        need(Path(item["replacement_attempt"]) == run / "tasks" / task_id,
             "Replacement attempt escaped its new task directory")
        need(item.get("fresh_actor_attempt") is True and item.get("restored_original_workspace") is False,
             "Fresh attempts must not be represented as restored workspaces")


def authorize_host(run: Path) -> None:
    need(socket.gethostname() == "harnessmm-api-worker", "Native continuation is remote-only on A800-4")
    need(run.resolve().is_relative_to(DATA) and run.resolve() != DATA,
         "Use a new data-volume continuation directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(PARENT / "source"))


def validate_parent() -> None:
    for name, expected in read(PARENT / "FROZEN_IDENTITY.json").items():
        need(sha(PARENT / name) == expected, f"Original frozen identity changed: {name}")
    for name, expected in read(PARENT / "SOURCE_SHA256.json").items():
        need(sha(PARENT / "source" / name) == expected, f"Original native source changed: {name}")
    for path, expected in read(PARENT / "NATIVE_RUNTIME.json")["files"].items():
        need(sha(Path(path)) == expected, f"Original native runtime changed: {path}")


def proof_for(task_id: str) -> dict:
    from harbor.models.trial.config import TrialConfig
    from scripts.serve_codex_deepseek import usage_summary

    attempt = PARENT / "tasks" / TASKS[task_id]
    row = read(attempt / "RESULT.json")
    native_path = Path(row["native_result"])
    native = read(native_path)
    need(sha(native_path) == row["native_result_sha256"], "Original native result changed")
    template_path = attempt / "native-run/CONFIG.json"
    need(TrialConfig.model_validate(read(template_path)).model_dump(mode="json") == native["config"],
         "Original native template does not match the executed configuration")
    ctrf = list(attempt.glob("**/verifier/ctrf.json"))
    stdout = list(attempt.glob("**/verifier/test-stdout.txt"))
    need(len(stdout) == 1, "Missing original native infrastructure diagnosis")
    diagnostic = stdout[0].read_text()
    dependency_error = (
        "Failed to download `pytest==8.4.1`" in diagnostic if task_id == "sqlite-with-gcov"
        else "uvx: command not found" in diagnostic and "lock-frontend" in diagnostic
    )
    container_path = attempt / "native-run/NATIVE_CONTAINER.json"
    container = read(container_path)
    need(container.get("captured_before_actor") is True, "Original workspace identity was not captured")
    inspected = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", container["container_id"]],
        capture_output=True, text=True, timeout=30,
    )
    images_command = [
        "docker", "image", "ls", "--no-trunc", "--filter",
        "label=com.docker.compose.project=" + container["network_owner"]["project"],
        "--format", "{{.ID}}",
    ]
    images = subprocess.check_output(images_command, text=True, timeout=30).splitlines()
    artifacts = [
        str(p) for p in attempt.glob("native-run/native/*/artifacts/**/*")
        if p.is_file() and p.name != "manifest.json"
    ]
    trial_id = read(attempt / "TRIAL_ID.json")["trial_id"]
    usage = usage_summary(Path(read(PARENT / "CONFIG.json")["codex"]["provider_ledger"]), trial_id)
    receipts = [Path(path) for path in usage["receipts"]]
    active = read(PARENT / "REPORT.json").get("active", {})
    proof = {
        "at": now(), "task_id": task_id, "original_attempt": str(attempt),
        "original_result": ref(attempt / "RESULT.json"), "original_native_result": ref(native_path),
        "native_template": ref(attempt / "native-run/CONFIG.json"),
        "original_native_reward": (native.get("verifier_result") or {}).get("rewards", {}).get("reward"),
        "original_native_tests_executed": bool(ctrf),
        "native_dependency_failure_log": ref(stdout[0]),
        "dependency_failure_confirmed": dependency_error,
        "original_actor_exited": (attempt / "PROCESS_EXIT.json").is_file(),
        "original_actor_exit": read(attempt / "PROCESS_EXIT.json"),
        "original_actor_active": task_id in active,
        "container_receipt": ref(container_path), "container_id": container["container_id"],
        "workspace_absent": inspected.returncode != 0 and "no such object" in inspected.stderr.lower(),
        "workspace_probe": {"returncode": inspected.returncode, "stderr": inspected.stderr},
        "snapshot_probe_command": images_command, "snapshot_image_ids": images,
        "retained_workspace_files": artifacts, "settled_usage": usage,
        "all_provider_receipts_finished": all(read(p).get("finished_at") for p in receipts),
        "provider_receipts": [ref(p) for p in receipts],
        "shared_prompt_sha256": row["shared_prompt_sha256"],
        "partition_sha256": row["partition_sha256"],
        "original_actor_cost": row["cost"], "new_actor_calls": 0,
    }
    original_eligible(proof)
    need(row["cost"]["usage"] == usage["known_usage"]
         and row["cost"]["provider_call_count"] == usage["calls"],
         "Original paid costs do not reconcile to all settled requests")
    return proof


def capacity(run: Path) -> dict:
    paths = (run, Path("/path/to/data/docker"), Path(read(PARENT / "CONFIG.json")["codex"]["provider_ledger"]))
    storage = {}
    for path in paths:
        stats = os.statvfs(path)
        available = stats.f_bavail * stats.f_frsize
        need(available >= 24 * 1024**3, f"Keep 8GiB reserve plus declared continuation footprint: {path}")
        storage[str(path)] = available
    return {
        "at": now(), "hostname": socket.gethostname(), "CUDA_VISIBLE_DEVICES": "",
        "selected_gpus": [], "storage_available_bytes": storage,
        "gpu_inventory": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader",
        ], text=True, timeout=30),
        "memory": Path("/proc/meminfo").read_text(), "load_average": os.getloadavg(),
        "placement": "A800 CPU/Docker fixture and existing private DeepSeek gateway; no GPU allocation",
    }


def prepare(run: Path) -> dict:
    authorize_host(run)
    validate_parent()
    need(not (run / "PROSPECTIVE_REPLACEMENTS.json").exists(), "Continuation plan already frozen")
    for existing in DATA.glob("2026*-codex-infrastructure-continuation-*/PROSPECTIVE_REPLACEMENTS.json"):
        need(existing.parent == run, f"These original attempts already have a continuation: {existing}")
    need((run / "SOURCE_GIT_STATE.json").is_file(), "Stage source hashes and Git state before preparation")
    run.mkdir(parents=True, exist_ok=True)
    (run / "proofs").mkdir()
    (run / "tasks").mkdir()
    save(run / "RESOURCES.json", capacity(run))
    tasks = {}
    for task_id in TASKS:
        proof = proof_for(task_id)
        path = run / "proofs" / (task_id + ".json")
        save(path, proof)
        tasks[task_id] = {
            "original_attempt": proof["original_attempt"], "original_proof": ref(path),
            "replacement_attempt": str(run / "tasks" / task_id),
            "dependency_cache": str(ADAPTIVE_CACHE if task_id == "adaptive-rejection-sampler"
                                    else run / "dependencies/sqlite-with-gcov/PREPARED.json"),
            "fresh_actor_attempt": True, "restored_original_workspace": False,
        }
    plan = {
        "schema": SCHEMA, "created_at": now(), "track": "codex", "phase": "train",
        "parent_run": str(run), "original_parent_run": str(PARENT), "tasks": tasks,
        "shared_prompt_sha256": PROMPT, "partition_sha256": PARTITION,
        "maximum_actor_attempts_per_task": 1,
        "selection_policy": "one_prospective_replacement_not_best_of",
        "cost_policy": "Every old and new paid actor attempt retained and counted exactly once",
        "authorization": "2026-10-01 prospective replacements of two permanently unscored Codex failures",
        "source": ref(Path(__file__).resolve()), "original_frozen_identity": ref(PARENT / "FROZEN_IDENTITY.json"),
        "source_git_state": ref(run / "SOURCE_GIT_STATE.json"),
        "execution_seed": read(PARENT / "CONFIG.json")["execution_seed"],
        "provider_sampling_seed_supported": False,
        "cumulative_token_limit": None, "other_native_owners_untouched": True,
    }
    validate_plan(plan, run)
    save(run / "PROSPECTIVE_REPLACEMENTS.json", plan)
    save(run / "FROZEN_PLAN.json", ref(run / "PROSPECTIVE_REPLACEMENTS.json"))
    return plan


async def native_attempt(run: Path, task_id: str) -> None:
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial, TrialEvent
    from scripts.recover_terminal_verifier import (
        install_cache, record_owned_container, release_stopped_trial_network, verified_cache,
    )
    from scripts.run_industrial_capability_baselines import harvest
    from scripts.run_codex_terminal import settle_provider_usage

    plan = read(run / "PROSPECTIVE_REPLACEMENTS.json")
    validate_plan(plan, run)
    need(sha(run / "PROSPECTIVE_REPLACEMENTS.json") == read(run / "FROZEN_PLAN.json")["sha256"],
         "Prospective selection manifest changed")
    validate_parent()
    item = plan["tasks"][task_id]
    attempt = Path(item["replacement_attempt"])
    proof = read(Path(item["original_proof"]["path"]))
    need(sha(Path(item["original_proof"]["path"])) == item["original_proof"]["sha256"],
         "Original failure proof changed")
    original_eligible(proof)
    fresh = proof_for(task_id)
    need(fresh["original_result"] == proof["original_result"]
         and fresh["provider_receipts"] == proof["provider_receipts"], "Original attempt changed since freeze")
    cache = Path(item["dependency_cache"])
    while not cache.is_file():
        state_path = cache.parent / "STATE.json"
        state = read(state_path) if state_path.is_file() else {}
        save(attempt / "STATE.json", {
            "status": "WAITING_VERIFIED_DEPENDENCIES", "at": now(), "pid": os.getpid(),
            "task_id": task_id, "cache": str(cache), "upstream": state, "additional_actor_calls": 0,
        })
        need(state.get("status") not in ("BLOCKED", "COMPLETE"), "Dependency preparation ended without a cache")
        need(type(state.get("pid")) is int and (Path("/proc") / str(state["pid"])).exists(),
             "Dependency preparation has no live owner; no actor was started")
        await asyncio.sleep(30)
    old_task = Path(item["original_attempt"]) / "task-source" / task_id
    prepared = verified_cache(cache, old_task)
    save(attempt / "RESOURCES.json", capacity(run))
    task = attempt / "task-source" / task_id
    shutil.copytree(old_task, task)
    expected = read(PARENT / "TASK_SOURCE_SHA256.json")[task_id]
    need({p.relative_to(task).as_posix(): sha(p) for p in task.rglob("*") if p.is_file()} == expected,
         "New native task bytes differ from the original task")
    template_path = Path(item["original_attempt"]) / "native-run/CONFIG.json"
    need(sha(template_path) == proof["native_template"]["sha256"], "Original native template changed")
    template = read(template_path)
    trial_id = "cap-" + hashlib.sha256(str(attempt).encode()).hexdigest()[:32]
    parent_config = read(PARENT / "CONFIG.json")
    ledger = Path(parent_config["codex"]["provider_ledger"])
    need(not (ledger / "calls" / trial_id).exists(), "Replacement provider identity was already used")
    output = attempt / "native-run"
    output.mkdir()
    config = replacement_config(template, task, attempt, trial_id, prepared["verifier_environment"])
    save(attempt / "TRIAL_ID.json", {"trial_id": trial_id})
    save(output / "CONFIG.json", config)
    start = {
        "at": now(), "task_id": task_id, "attempt_id": str(attempt),
        "prospective_manifest": ref(run / "PROSPECTIVE_REPLACEMENTS.json"),
        "original_proof": item["original_proof"], "dependency_cache": ref(cache),
        "dependency_archive": {"path": prepared["archive"], "sha256": prepared["archive_sha256"]},
        "original_native_template": proof["native_template"], "native_config": ref(output / "CONFIG.json"),
        "source": ref(Path(__file__).resolve()), "fresh_actor_attempt": True,
        "restored_original_workspace": False, "maximum_actor_attempts": 1,
        "framework": "Pinned official Harbor Trial + unchanged SharedPromptCodex/official Codex loop",
        "cleanup_policy": "Retain this new workspace even if native verification is unscored",
        "native_verifier_deadline_changed": False, "cumulative_token_limit": None,
        "execution_seed": plan["execution_seed"], "provider_sampling_seed_supported": False,
        "package_versions": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
    }
    with (attempt / "ACTOR_ATTEMPT_START.json").open("x") as stream:
        json.dump(start, stream, indent=2, sort_keys=True)
    state = {"status": "STARTING_NATIVE_ENVIRONMENT", "at": now(), "pid": os.getpid(),
             "task_id": task_id, "fresh_actor_attempts": 1, "native_trials": 1}
    save(attempt / "STATE.json", state)
    trial = await Trial.create(TrialConfig.model_validate(config))

    async def retain(event):
        need(event.task_name == task_id, "Unexpected native actor task")
        await record_owned_container(trial.agent_environment, output)
        state.update(status="NATIVE_ACTOR_RUNNING", at=now())
        save(attempt / "STATE.json", state)

    async def verifier_cache(event):
        need(event.task_name == task_id, "Unexpected native verifier task")
        need(sha(cache) == start["dependency_cache"]["sha256"], "Verified dependency receipt changed")
        await install_cache(trial.agent_environment, cache, task, output)
        state.update(status="NATIVE_VERIFIER_RUNNING", at=now())
        save(attempt / "STATE.json", state)

    trial.add_hook(TrialEvent.AGENT_START, retain)
    trial.add_hook(TrialEvent.VERIFICATION_START, verifier_cache)
    result = (await trial.run()).model_dump(mode="json")
    save(output / "native_result.json", result)
    save(attempt / "NATIVE_FINISHED.json", {"at": now(), "native_result": ref(output / "native_result.json")})
    try:
        await release_stopped_trial_network(output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        save(attempt / "NETWORK_RELEASE_FAILURE.json", {"at": now(), "error": str(error)})
        print(f"Native network release failed: {error}", file=sys.stderr, flush=True)
    state.update(status="SETTLING_PROVIDER_COSTS", at=now())
    save(attempt / "STATE.json", state)
    try:
        await settle_provider_usage(
            ledger, trial_id, output, parent_config["codex"]["provider_settlement_timeout_seconds"],
        )
    finally:
        row = harvest(parent_config, "codex", PARENT, attempt, task_id)
        save(attempt / "CONTINUATION_RESULT.json", {
            "at": now(), "new_attempt": ref(attempt / "RESULT.json"),
            "original_attempt": item["original_attempt"], "fresh_actor_attempt": True,
            "raw_native_status": row["status"], "cost": row["cost"],
            "admission": "Only the separate source-bound native-test auditor may admit this verdict",
        })
    state.update(status="NATIVE_FINISHED_AWAITING_ADMISSION", at=now())
    save(attempt / "STATE.json", state)


def worker(run: Path, task_id: str) -> None:
    authorize_host(run)
    need(task_id in TASKS, "Task is not an authorized infrastructure continuation")
    attempt = run / "tasks" / task_id
    attempt.mkdir(exist_ok=False)
    lock = (attempt / "ACTOR.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ticks = (Path("/proc") / str(os.getpid()) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    save(attempt / "OWNER.json", {
        "pid": os.getpid(), "start_ticks": ticks, "at": now(), "task_id": task_id, "argv": sys.argv,
    })
    save(attempt / "COMMAND.json", {"argv": sys.argv, "source": ref(Path(__file__).resolve())})
    try:
        asyncio.run(native_attempt(run, task_id))
    except Exception as error:
        save(attempt / "STATE.json", {
            "status": "BLOCKED", "at": now(), "pid": os.getpid(), "task_id": task_id,
            "error_type": type(error).__name__, "error": str(error),
            "fresh_actor_attempts": int((attempt / "ACTOR_ATTEMPT_START.json").is_file()),
        })
        (attempt / "FAILURE.txt").write_text(traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--task-id", choices=tuple(TASKS))
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.run_dir), sort_keys=True))
    else:
        need(args.task_id is not None, "Choose one authorized task; no automatic retries")
        worker(args.run_dir, args.task_id)


if __name__ == "__main__":
    main()
