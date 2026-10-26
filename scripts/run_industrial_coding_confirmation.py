#!/usr/bin/env python3
"""Confirm a TRAIN-derived coding instruction with native matched held-out trials.

Four frozen arms isolate the learned instruction and native compaction. Complete
validation18 selects a fixed candidate before any TEST53x5 actor can start.
Failures preserve their actors and block new submissions, never become zeros.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import capture_provenance, utc_now, write_json_atomic  # noqa: E402
from phiagent.harness import industrial_coding_confirmation as protocol  # noqa: E402
from scripts import run_industrial_capability_baselines as native  # noqa: E402
from scripts.run_codex_context_control import source_bound_parent_evidence  # noqa: E402
from scripts.run_codex_terminal import executed_native_verdict  # noqa: E402
from scripts.serve_codex_deepseek import usage_summary  # noqa: E402

RUN_ROOT = Path("/path/to/data/phiagent/work/industrial-coding-confirmation")
SOURCE_ENTRIES = (
    "scripts/run_industrial_coding_confirmation.py",
    "phiagent/harness/industrial_coding_confirmation.py",
    "tests/test_industrial_coding_confirmation.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha(path: Path) -> str:
    return native.digest(path)


def verified_input(entry: dict) -> Path:
    if set(entry) != {"path", "sha256"}:
        raise ValueError("Every input requires an exact source path and file SHA256")
    path = Path(entry["path"])
    if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Input capsules require absolute nonsymlink paths")
    if file_sha(path) != entry["sha256"]:
        raise ValueError(f"Frozen input changed: {path}")
    return path


def require_run_path(run: Path) -> None:
    if (not run.is_absolute() or run == RUN_ROOT or not run.is_relative_to(RUN_ROOT)
            or ".." in run.parts or any(part.is_symlink() for part in (run, *run.parents))):
        raise ValueError("Confirmation requires a fresh explicit data-volume run directory")


def candidate_training_rows(report: dict, candidate: dict, task_ids: list[str]) -> list[dict]:
    instruction = protocol.instruction_suffix(candidate["instruction"])
    rows = []
    seen = set()
    for entry in report["ledger"]:
        if (entry["kind"] != "executor" or entry["phase"] != "train"
                or entry["arm"] != candidate["id"]):
            continue
        task = entry["task_id"]
        if task not in task_ids or task in seen or entry["seed"] != 20260928:
            raise ValueError("Candidate TRAIN population or repetition identity differs")
        seen.add(task)
        result = report["observations"][entry["request_id"]]
        path = Path(result["provenance"]["native_record"])
        if file_sha(path) != result["provenance"]["native_record_sha256"]:
            raise ValueError("TRAIN actor receipt changed")
        record = read(path)
        if (
            record["phase"] != "train" or record["task_id"] != task
            or record["method"] != "chisel_codex" or record["model"] != "deepseek-flash"
            or record["residual_sha256"] != protocol.text_sha256(instruction)
            or record["seed"] != 20260928
            or record["context_control"]["active_context_trigger_tokens"] != 996147
            or record["shared_prompt"]["sha256"] != native.PROMPT_SHA256
        ):
            raise ValueError("TRAIN instruction, native runtime or matched prompt differs")
        native_path = Path(record["native_result"])
        native_result = read(native_path)
        verdict = executed_native_verdict(native_result, path.parent / "native")
        if verdict["score"] != result["native_score"] or verdict["score"] != record["score"]:
            raise ValueError("Stored TRAIN score differs from executed native verification")
        usage = record["provider_usage"]
        if (
            usage["usage_status"] != "MEASURED"
            or usage["known_usage"]["total_tokens"] != result["usage"]["total_tokens"]
            or usage["provider_models"] != ["deepseek-flash"]
        ):
            raise ValueError("Complete same-backbone TRAIN provider accounting is required")
        row = {
            "arm": candidate["id"], "task_id": task, "seed": entry["seed"], "status": "SCORED",
            "score": verdict["score"], "usage": {
                key: result["usage"][key] for key in protocol.TOKEN_KEYS
            },
            "executed_native_tests_verified": True,
            "actor_receipt": {"path": str(path), "sha256": file_sha(path)},
            "native_receipt": {"path": str(native_path), "sha256": file_sha(native_path)},
            "native_verifier": verdict,
        }
        protocol.verify_measured_row(row)
        rows.append(row)
    if seen != set(task_ids):
        raise ValueError("Every one of the18 frozen TRAIN tasks must have genuine candidate evidence")
    return rows


def select_training_instruction(report: dict, task_ids: list[str]) -> tuple[dict, dict]:
    if (report["state"] != "trained" or report["selection"] is not None or report["test"] is not None):
        raise ValueError("Only completed, unselected TRAIN evidence may select an instruction")
    candidates, evidence = [], {}
    for candidate in report["candidate_bank"]:
        if candidate["id"] == "noop":
            continue
        if (candidate["origin"] != "optimizer_callback_train_only"
                or candidate["training_tasks"] != 18
                or candidate["paired_training_observations"] != 18
                or not candidate["instruction"].strip()):
            raise ValueError("Candidate was not derived exclusively from the complete TRAIN parent")
        rows = candidate_training_rows(report, candidate, task_ids)
        summary = protocol.summarize_arm(rows, task_ids, [20260928], candidate["id"])
        evidence[candidate["id"]] = {"summary": summary, "rows": rows}
        candidates.append(candidate)
    if not candidates:
        raise ValueError("There is no verified TRAIN-derived candidate to confirm")
    selected = min(candidates, key=lambda value: (
        -evidence[value["id"]]["summary"]["mean_native_score"],
        evidence[value["id"]]["summary"]["actor_usage"]["total_tokens"], value["id"],
    ))
    return deepcopy(selected), evidence


def prepare(config_path: Path) -> Path:
    native.require_remote()
    config = read(config_path)
    native.validate_workers(config.get("maximum_workers"))
    if (
        config["schema"] != protocol.SCHEMA or config["cumulative_token_limit"] is not None
        or config["test_repetitions"] != 5
        or config["model"] != "deepseek-flash" or config["minimum_free_disk_gib"] < 8
    ):
        raise ValueError("Preserve the authorized model, native worker policy and unlimited campaign")
    run = Path(config["run_directory"])
    require_run_path(run)
    if run.exists():
        raise FileExistsError("New confirmation settings require a new run")
    bundle = read(ROOT / "SOURCE_BUNDLE.json")
    for name, checksum in bundle["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or file_sha(ROOT / name) != checksum:
            raise ValueError("New confirmation source capsule changed")
    template_path = verified_input(config["inputs"]["template"])
    report_path = verified_input(config["inputs"]["training_report"])
    template = read(template_path)
    partition_directory = Path(template["benchmarks"]["terminal-bench-2.0"]["partition_directory"])
    partition_path = partition_directory / "SPLIT_MANIFEST.json"
    partition_sha = file_sha(partition_path)
    if partition_sha != native.PARTITION_SHA256:
        raise ValueError("The canonical complete18/18/53 partition changed")
    partition = read(partition_path)
    protocol.validate_partition(partition)
    parent = source_bound_parent_evidence(config["parent_binding"], "codex")
    if (
        parent["ready"] is not True or parent["native_scored"] != 18
        or parent["native_successes"] < 9
        or parent["report"]["selected_execution_accounting"]["tokens"]["total_tokens"][
            "complete_total"
        ] is None
    ):
        raise ValueError("Confirmation requires the corrected, capable, fully accounted native parent")
    report = read(report_path)
    candidate, training = select_training_instruction(report, partition["splits"]["train"])
    selected_summary = training[candidate["id"]]["summary"]
    if (
        selected_summary["successes"] <= parent["native_successes"]
        or selected_summary["actor_usage"]["total_tokens"] >=
        parent["report"]["selected_execution_accounting"]["tokens"]["total_tokens"]["complete_total"]
    ):
        raise ValueError("The selected coding instruction does not improve the corrected TRAIN parent")
    inherited = read(Path(config["parent_binding"]["parent_run"]) / "CONFIG.json")
    inherited["maximum_workers"] = config["maximum_workers"]
    native.validate_config(inherited)
    if not native.storage_capacity(inherited, run)["ready"]:
        raise RuntimeError("Native storage/memory admission is blocked; do not bypass reserves")
    run.mkdir(parents=True, mode=0o700)
    (run / "process-cache").mkdir()
    for name in bundle["files"]:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    instruction = candidate["instruction"]
    residual = protocol.instruction_suffix(instruction)
    contract = {
        "schema": protocol.SCHEMA, "benchmark": "terminal-bench-2.0",
        "model": "deepseek-flash", "arms": protocol.ARMS,
        "partition": partition, "partition_sha256": partition_sha,
        "shared_prompt_sha256": native.PROMPT_SHA256, "shared_prompt_version": native.PROMPT_VERSION,
        "instruction_sha256": protocol.text_sha256(instruction),
        "applied_residual_sha256": protocol.text_sha256(residual),
        "candidate_id": candidate["id"], "validation_seeds": [20260928],
        "test_seeds": list(range(20260928, 20260933)),
        "maximum_workers": config["maximum_workers"],
        "cumulative_token_limit": None, "token_usage_mode": "unlimited",
        "test_selection_from_outcomes": False, "optimization_from_validation": False,
        "method_identity": "TRAIN-derived industrial instruction/context study",
        "cot_jev_core_claim": False,
        "primary_test_contrast": "Validation-selected fixed candidate versus both native controls",
        "all_mechanism_arms_retained_for_test": True,
        "native_timeouts_and_resources": "Unmodified task.toml",
        "model_sampling_seed_supported": False,
        "prior_exposure": partition["prior_exposure"],
    }
    protocol.validate_contract(contract)
    for name, value in (
        ("CONFIG.json", config), ("NATIVE_CONFIG.json", inherited),
        ("PROTOCOL.json", contract), ("PARENT_ADMISSION.json", parent),
        ("CORRECTED_TRAIN.json", training),
        ("SOURCE_SHA256.json", bundle["files"]), ("STAGING_PROVENANCE.json", bundle),
        ("NATIVE_RUNTIME.json", read(Path(config["parent_binding"]["parent_run"]) / "NATIVE_RUNTIME.json")),
        ("FROZEN_INSTRUCTION.json", {
            "candidate": candidate, "selected_from": "Complete TRAIN only; highest native quality "
            "then lowest complete actor tokens", "router_from_invalid_legacy_labels_reused": False,
            "training_report": config["inputs"]["training_report"],
            "new_optimizer_calls": 0,
        }),
        ("RESOURCE_PRELAUNCH.json", native.resource_inventory()),
    ):
        write_json_atomic(run / name, value)
    (run / "INSTRUCTION.txt").write_bytes(residual.encode("utf-8"))
    shutil.copyfile(partition_path, run / "SPLIT_MANIFEST.json")
    shutil.copyfile(partition_directory / "TRAIN_VALIDATION.json", run / "TRAIN_VALIDATION.json")
    # Keep only the sealed metadata path; TEST task contents are never opened during preparation.
    write_json_atomic(run / "TEST_METADATA_BINDING.json", {
        "path": str(partition_directory / "SEALED_TEST.json"),
        "sha256": file_sha(partition_directory / "SEALED_TEST.json"),
    })
    frozen_names = (
        "CONFIG.json", "NATIVE_CONFIG.json", "PROTOCOL.json", "PARENT_ADMISSION.json",
        "CORRECTED_TRAIN.json", "SOURCE_SHA256.json", "STAGING_PROVENANCE.json",
        "FROZEN_INSTRUCTION.json", "INSTRUCTION.txt", "SPLIT_MANIFEST.json",
        "TRAIN_VALIDATION.json", "TEST_METADATA_BINDING.json", "NATIVE_RUNTIME.json",
    )
    write_json_atomic(run / "FROZEN_IDENTITY.json", {
        name: file_sha(run / name) for name in frozen_names
    })
    write_json_atomic(run / "PROVENANCE.json", capture_provenance(ROOT, sys.argv, 20260928))
    write_json_atomic(run / "STATE.json", {
        "status": "PREPARED_VALIDATION", "at": utc_now(), "schema": protocol.SCHEMA,
        "test_opened": False, "cumulative_token_limit": None,
    })
    return run


def check_run(run: Path) -> None:
    for name, checksum in read(run / "FROZEN_IDENTITY.json").items():
        if file_sha(run / name) != checksum:
            raise ValueError(f"Frozen confirmation identity changed: {name}")
    for name, checksum in read(run / "SOURCE_SHA256.json").items():
        if file_sha(run / "source" / name) != checksum:
            raise ValueError(f"Frozen confirmation source changed: {name}")
    for name, checksum in read(run / "NATIVE_RUNTIME.json")["files"].items():
        if file_sha(Path(name)) != checksum:
            raise ValueError(f"Pinned official native runtime changed: {name}")


def phase_plan(run: Path, phase: str) -> list[dict]:
    contract = read(run / "PROTOCOL.json")
    protocol.validate_contract(contract)
    if phase == "validation":
        rows = read(run / "TRAIN_VALIDATION.json")["validation"]
    elif phase == "test":
        release = read(run / "TEST_RELEASE.json")
        native.validate_config(read(run / "NATIVE_CONFIG.json"))
        if release["status"] != "FROZEN_FOR_TEST":
            raise ValueError("TEST remains sealed")
        binding = read(run / "TEST_METADATA_BINDING.json")
        source = Path(binding["path"])
        if file_sha(source) != binding["sha256"]:
            raise ValueError("Sealed TEST population metadata changed")
        rows = read(source)["test"]
    else:
        raise ValueError("Confirmation cannot optimize on TRAIN or inspect an undeclared phase")
    ids = contract["partition"]["splits"][phase]
    paths = {row["id"]: row["path"] for row in rows}
    if len(paths) != len(rows) or set(paths) != set(ids):
        raise ValueError("Native phase metadata omits or duplicates frozen task identities")
    arms = list(protocol.ARMS)
    jobs = []
    for seed in contract[phase + "_seeds"]:
        for index, task in enumerate(ids):
            order = arms[index % len(arms):] + arms[:index % len(arms)]
            for arm in order:
                identity = f"{phase}\0{arm}\0{seed}\0{task}\0{run}"
                trial_id = "ic-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
                jobs.append({
                    "phase": phase, "arm": arm, "seed": seed, "task_id": task,
                    "task_path": paths[task], "trial_id": trial_id,
                    "directory": str(run / "trials" / phase / arm / str(seed) / task),
                })
    if len({job["trial_id"] for job in jobs}) != len(jobs):
        raise ValueError("Independent native trial namespaces collide")
    return jobs


def trial_command(run: Path, job: dict) -> list[str]:
    config, contract = read(run / "NATIVE_CONFIG.json"), read(run / "PROTOCOL.json")
    arm = protocol.ARMS[job["arm"]]
    directory = Path(job["directory"])
    actor = config["codex"]
    command = [
        actor["python"], "-B", "-u", str(run / "source/scripts/run_codex_terminal.py"),
        "--task", job["task_path"], "--partition", str(run / "SPLIT_MANIFEST.json"),
        "--phase", job["phase"], "--output", str(directory / "native-run"),
        "--trial-id", job["trial_id"], "--provider-url", actor["provider_url"],
        "--provider-ledger", actor["provider_ledger"],
        "--relay-secret", actor["relay_secret_file"], "--codex-binary", actor["binary"],
        "--method", arm["method"], "--context-compaction-threshold",
        str(arm["context_compaction_threshold"]), "--repetition-seed", str(job["seed"]),
        "--common-prompt-version", contract["shared_prompt_version"],
        "--common-prompt-sha256", contract["shared_prompt_sha256"],
        "--verifier-environment", str(run / "VERIFIER_ENVIRONMENT.json"),
        "--provider-settlement-timeout", str(actor["provider_settlement_timeout_seconds"]),
        "--confirmation-contract", str(run / "PROTOCOL.json"),
        "--confirmation-contract-sha256", file_sha(run / "PROTOCOL.json"),
        "--confirmation-arm", job["arm"],
    ]
    if arm["instruction"]:
        command += ["--residual-file", str(run / "INSTRUCTION.txt")]
    if job["task_id"] == "sam-cell-seg":
        command += ["--sam-verifier-recovery", config["sam_verifier_recovery"]]
    cache = config.get("native_verifier_caches", {}).get(job["task_id"])
    if cache:
        command += ["--native-verifier-cache", cache]
    if job["phase"] == "test":
        command += ["--freeze", str(run / "TEST_RELEASE.json")]
    return command


def launch_trial(run: Path, job: dict) -> subprocess.Popen:
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    command = trial_command(run, job)
    cache = directory / "process-cache"
    cache.mkdir()
    write_json_atomic(directory / "INTENT.json", {
        **job, "argv": command, "at": utc_now(), "actor_replay_permitted": False,
    })
    with (directory / "process.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=directory, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(run / "source"),
                 "TMPDIR": str(cache), "PYTHONDONTWRITEBYTECODE": "1"},
        )
    write_json_atomic(directory / "PROCESS.json", {
        "pid": process.pid, "start_ticks": native.process_identity(process.pid),
        "started_at": utc_now(), "controller_pid": os.getpid(),
    })
    return process


def harvest_trial(run: Path, job: dict) -> dict:
    directory = Path(job["directory"])
    config = read(run / "NATIVE_CONFIG.json")
    provider = usage_summary(Path(config["codex"]["provider_ledger"]), job["trial_id"])
    output = directory / "native-run"
    path = output / "RUN.json"
    row = {
        **{key: job[key] for key in ("phase", "arm", "seed", "task_id", "trial_id")},
        "status": "BLOCKED", "score": None, "usage": {
            key: provider.get("known_usage", {}).get(key) for key in protocol.TOKEN_KEYS
        },
        "usage_complete": provider.get("usage_status") == "MEASURED",
        "provider_call_count": provider.get("calls"),
        "provider_receipts": provider.get("receipts", []),
        "executed_native_tests_verified": False,
        "attempt_directory": str(directory), "actor_replayed": False,
    }
    try:
        record = read(path)
        contract = read(run / "PROTOCOL.json")
        if (
            record["trial_id"] != job["trial_id"] or record["task_id"] != job["task_id"]
            or record["seed"] != job["seed"] or record["phase"] != job["phase"]
            or record["method"] != protocol.ARMS[job["arm"]]["method"]
            or record["confirmation"]["contract_sha256"] != file_sha(run / "PROTOCOL.json")
            or record["confirmation"]["arm"] != job["arm"]
            or record["partition_sha256"] != contract["partition_sha256"]
            or record["shared_prompt"]["sha256"] != contract["shared_prompt_sha256"]
        ):
            raise ValueError("Native receipt differs from its frozen method/task/phase contract")
        native_path = output / "native_result.json"
        result = read(native_path)
        verdict = executed_native_verdict(result, output / "native")
        if (
            record["status"] != "SCORED"
            or verdict["score"] != record["score"]
            or provider["usage_status"] != "MEASURED"
            or provider["provider_models"] != ["deepseek-flash"]
            or provider["known_usage"] != record["provider_usage"]["known_usage"]
        ):
            raise ValueError("Native scoring or provider settlement is incomplete")
        row.update(
            status="SCORED", score=verdict["score"], executed_native_tests_verified=True,
            native_verifier=verdict,
            native_result={"path": str(native_path), "sha256": file_sha(native_path)},
            actor_result={"path": str(path), "sha256": file_sha(path)},
        )
        protocol.verify_measured_row(row)
    except (OSError, ValueError, KeyError) as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    write_json_atomic(directory / "OUTCOME.json", row)
    return row


def stage_status(run: Path, plan: list[dict], processes: dict[int, subprocess.Popen]) -> tuple:
    rows, active, pending = [], [], []
    for job in plan:
        directory = Path(job["directory"])
        if not directory.exists():
            pending.append(job)
            continue
        process_path = directory / "PROCESS.json"
        if not process_path.exists():
            raise RuntimeError(f"AMBIGUOUS_NATIVE_LAUNCH_NO_REPLAY: {directory}")
        owner = read(process_path)
        process = processes.get(owner["pid"])
        if process is not None and process.poll() is not None:
            write_json_atomic(directory / "PROCESS_EXIT.json", {
                "at": utc_now(), "returncode": process.returncode,
            })
        if native.process_live(owner):
            active.append({**job, "owner": owner})
        else:
            rows.append(harvest_trial(run, job))
    return rows, active, pending


def run_stage(run: Path, phase: str) -> dict:
    plan = phase_plan(run, phase)
    contract = read(run / "PROTOCOL.json")
    native_config = read(run / "NATIVE_CONFIG.json")
    processes: dict[int, subprocess.Popen] = {}
    last_capacity = 0.0
    while True:
        check_run(run)
        rows, active, pending = stage_status(run, plan, processes)
        failures = [row for row in rows if row["status"] != "SCORED"]
        capacity = native.storage_capacity(native_config, run)
        if time.monotonic() - last_capacity >= 300:
            write_json_atomic(run / "RESOURCE_CURRENT.json", native.resource_inventory())
            last_capacity = time.monotonic()
        write_json_atomic(run / f"{phase.upper()}_ROWS.json", rows)
        write_json_atomic(run / "STATE.json", {
            "schema": protocol.SCHEMA, "at": utc_now(), "pid": os.getpid(), "phase": phase,
            "status": "DRAINING_NATIVE_FAILURE" if failures else "RUNNING_" + phase.upper(),
            "active": active, "pending": len(pending), "settled": len(rows),
            "scored": sum(row["status"] == "SCORED" for row in rows),
            "expected": len(plan), "native_failures": failures,
            "capacity": capacity, "cumulative_token_limit": None,
            "test_opened": phase == "test", "successor": "Complete validation; immutable TEST release "
            "only after the fixed quality/cost gate; then source-bound publication event",
        })
        if failures and not active:
            raise RuntimeError("Native verdict/accounting failure: no further actors dispatched")
        if (run / "STOP").exists() and not active:
            raise RuntimeError("Explicit user stop preserved; no active actor replayed")
        if not active and not pending:
            ids = contract["partition"]["splits"][phase]
            seeds = contract[phase + "_seeds"]
            summaries = {
                arm: protocol.summarize_arm([row for row in rows if row["arm"] == arm], ids, seeds, arm)
                for arm in protocol.ARMS
            }
            result = {"phase": phase, "summaries": summaries, "rows": rows,
                      "complete": True, "new_optimizer_calls": 0, "at": utc_now()}
            write_json_atomic(run / f"{phase.upper()}_COMPLETE.json", result)
            return result
        if not failures and not (run / "STOP").exists() and capacity["ready"]:
            free = contract["maximum_workers"] - len(active)
            admitted = list(active)
            for job in pending[:free]:
                capacity = native.task_capacity(
                    native_config, run,
                    [Path(item["task_path"]) for item in [*admitted, job]],
                )
                write_json_atomic(run / "NEXT_TASK_ADMISSION.json", capacity)
                if not capacity["ready"]:
                    break
                process = launch_trial(run, job)
                processes[process.pid] = process
                admitted.append(job)
        time.sleep(15)


def release_test(run: Path, validation: dict) -> dict:
    selection = protocol.validation_selection(validation["summaries"])
    write_json_atomic(run / "VALIDATION_SELECTION.json", selection)
    if selection["selected_arm"] is None:
        return selection
    files = [
        {"path": str(run / name), "sha256": file_sha(run / name)}
        for name in ("PROTOCOL.json", "INSTRUCTION.txt", "FROZEN_INSTRUCTION.json",
                     "VALIDATION_SELECTION.json", "FROZEN_IDENTITY.json")
    ]
    files += [
        {"path": str(run / "source" / name), "sha256": checksum}
        for name, checksum in read(run / "SOURCE_SHA256.json").items()
    ]
    write_json_atomic(run / "TEST_RELEASE.json", {
        **selection, "at": utc_now(),
        "partition_sha256": read(run / "PROTOCOL.json")["partition_sha256"],
        "files": files, "all_four_method_policies_unchanged": True,
    })
    return selection


def worker(run: Path) -> int:
    native.require_remote()
    require_run_path(run)
    check_run(run)
    if ROOT != run / "source":
        raise ValueError("Use the confirmation's own frozen source capsule")
    with (run / "OWNER.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json_atomic(run / "OWNER.json", {
            "at": utc_now(), "pid": os.getpid(), "start_ticks": native.process_identity(os.getpid()),
            "run": str(run), "hostname": socket.gethostname(), "selected_gpus": [],
        })
        inherited = read(run / "NATIVE_CONFIG.json")
        write_json_atomic(run / "VERIFIER_ENVIRONMENT.json", inherited["verifier_environment"])
        try:
            validation = run_stage(run, "validation")
            selected = release_test(run, validation)
            if selected["selected_arm"] is None:
                write_json_atomic(run / "STATE.json", {
                    "schema": protocol.SCHEMA, "at": utc_now(), "status": selected["status"],
                    "test_opened": False, "selection": selected,
                })
                return 0
            test = run_stage(run, "test")
            contract = read(run / "PROTOCOL.json")
            comparisons = protocol.paired_comparisons(
                {arm: [row for row in test["rows"] if row["arm"] == arm] for arm in protocol.ARMS},
                contract["partition"]["splits"]["test"], contract["test_seeds"],
                selected["selected_arm"],
            )
            write_json_atomic(run / "COMPARISON.json", comparisons)
            write_json_atomic(run / "PUBLICATION_READY.json", {
                "at": utc_now(), "result": str(run / "COMPARISON.json"),
                "result_sha256": file_sha(run / "COMPARISON.json"),
                "complete_official_native_metrics": True,
                "positive_result_qualified": comparisons["positive_complete_comparison"],
                "training_and_optimizer_accounting": str(run / "CORRECTED_TRAIN.json"),
                "publication_requires_independent_source_audit": True,
                "cloud_sync_required_after_manuscript_update": True,
                "cot_jev_core_claim": False,
            })
            write_json_atomic(run / "STATE.json", {
                "schema": protocol.SCHEMA, "at": utc_now(), "status": "COMPLETE_PAIRED_TEST",
                "test_opened": True, "comparison": str(run / "COMPARISON.json"),
                "positive_result_qualified": comparisons["positive_complete_comparison"],
            })
            return 0
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            write_json_atomic(run / "FAILURE.json", {
                "at": utc_now(), "type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "actor_replayed": False,
                "token_ceiling_triggered": False,
            })
            write_json_atomic(run / "STATE.json", {
                "schema": protocol.SCHEMA, "at": utc_now(), "status": "BLOCKED",
                "failure": str(run / "FAILURE.json"),
                "test_opened": (run / "TEST_RELEASE.json").exists(),
            })
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    native.require_remote()
    if args.worker:
        if args.run_dir is None:
            parser.error("--worker requires --run-dir")
        return worker(args.run_dir)
    if args.config is None:
        parser.error("A new confirmation requires --config")
    run = prepare(args.config)
    if not args.detach:
        raise ValueError("Run the prepared source capsule with --worker, or use --detach")
    config = read(run / "NATIVE_CONFIG.json")
    command = [
        config["codex"]["python"], "-B", "-u",
        str(run / "source/scripts/run_industrial_coding_confirmation.py"),
        "--run-dir", str(run), "--worker",
    ]
    with (run / "controller.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=run, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(run / "source"),
                 "TMPDIR": str(run / "process-cache"), "PYTHONDONTWRITEBYTECODE": "1"},
        )
    write_json_atomic(run / "LAUNCH.json", {
        "at": utc_now(), "pid": process.pid, "start_ticks": native.process_identity(process.pid),
        "argv": command, "detached": True,
    })
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f"Confirmation startup failed; inspect {run / 'controller.log'}")
    print(json.dumps({"run": str(run), "pid": process.pid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
