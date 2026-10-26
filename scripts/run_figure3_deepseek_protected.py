#!/usr/bin/env python3
"""Protected VAL/TEST orchestration around a TRAIN owner's frozen native adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phiagent.harness.figure3_repetitions import test_seeds  # noqa: E402

ARMS = ("native", "cot_jev")
FIELDS = ("input_tokens", "output_tokens", "total_tokens")
SEEDS = (2026092801, 2026092802, 2026092803, 2026092804, 2026092805)
PARTITION_SHA = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
API_MODEL = "deepseek-flash"
FALLBACK_MODEL = "Qwen/Qwen3.5-4B"
REMOTE_HOSTNAME = "harnessmm-api-worker"
RUN_ROOT = Path("/path/to/data/phiagent/work/deepseek-harness/industrial-capability/20260930-v1")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in {"Z", "T", "t", "X", "x"} else None
    except FileNotFoundError:
        return None


def live_process(receipt_: dict) -> bool:
    return (
        type(receipt_.get("pid")) is int and receipt_.get("start_ticks") is not None
        and process_identity(receipt_["pid"]) == receipt_["start_ticks"]
    )


def verified(reference_: dict) -> dict:
    path = Path(reference_["path"])
    if sha(path) != reference_["sha256"]:
        raise ValueError(f"An owner-provided immutable receipt changed: {path}")
    return read(path)


def usage_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and all(type(value.get(field)) is int and value[field] >= 0 for field in FIELDS)
        and value["input_tokens"] + value["output_tokens"] == value["total_tokens"]
    )


def reporting_policy(value: dict) -> str:
    policy = value.get("heldout_outcome_policy", "qualified_only")
    if policy not in ("qualified_only", "all_outcomes"):
        raise ValueError("Declare qualified_only or all_outcomes held-out reporting")
    if policy == "all_outcomes":
        authorization = value.get("outcome_reporting_authorization", {})
        timestamp = authorization.get("user_directive_at")
        if not isinstance(timestamp, str):
            raise ValueError("All-outcome TEST requires explicit prospective user authorization")
        directive = datetime.fromisoformat(timestamp)
        if (
            directive.tzinfo is None
            or authorization.get("retain_complete_null_and_negative_test") is not True
            or authorization.get("test_feedback_to_optimizer") is not False
        ):
            raise ValueError("Preserve prospective authorization and the protected TEST boundary")
    return policy


def owner_gate(handoff: dict, report: dict) -> dict:
    """Consume the owner's audit, never rerun TRAIN or derive its proposals."""
    policy = reporting_policy(handoff)
    if policy != reporting_policy(report):
        raise ValueError("The TRAIN audit and owner handoff must declare the same reporting policy")
    if (
        handoff.get("schema") != "figure3-deepseek-protected-owner-handoff/v1"
        or handoff.get("protected_pipeline_already_owned") is not False
        or handoff.get("ready_for_protected_validation") is not True
        or handoff.get("cumulative_token_limit") is not None
        or handoff.get("token_usage_mode") != "unlimited"
        or handoff.get("runtime_unlimited_verified_by_owner") is not True
        or handoff.get("completion_adapter_forwarding_verified") is not True
        or handoff.get("partition", {}).get("sha256") != PARTITION_SHA
        or handoff.get("component_identity") != "native_cot_jev"
        or handoff.get("zero_decoding_readout") is not False
        or handoff.get("training_feedback_only") is not True
        or handoff.get("validation_or_test_feedback_used") is not False
        or report.get("phase") != "train"
        or report.get("complete") is not True
        or report.get("eligible_for_protected_validation") is not True
        or report.get("native_fast_execution_proofs", 0) <= 0
        or report.get("cot_only_fast_calls") != 0
        or report.get("errors")
    ):
        raise ValueError("Wait for the existing owner's fully audited TRAIN handoff")
    if handoff.get("include_cot_only_control") is True:
        if set(handoff.get("native_roots", {})) != {"native", "cot_only", "cot_jev"}:
            raise ValueError("All three matched frozen native configurations are required")
        for candidate in handoff["train_candidates"]:
            artifact = verified(candidate["artifact"])
            if artifact.get("provenance", {}).get("author_result") != handoff.get("bank_author_receipt"):
                raise ValueError("Charge the actual author receipt of the TRAIN-frozen bank")
        from scripts.validate_figure3_deepseek_manifest import verified_bank_author_usage
        verified_bank_author_usage(handoff, verified)
    model = handoff.get("model_identity", {}).get("model")
    if model == FALLBACK_MODEL:
        identity = handoff["model_identity"]
        authorization = handoff.get("fallback_authorization", {})
        if (
            authorization.get("reason") not in {"insufficient_balance", "insufficient_quota"}
            or authorization.get("provider_response") is None
            or identity.get("pinned_revision") is None
            or identity.get("gpu_selection") is None
            or handoff.get("fresh_matched_fallback_train") is not True
        ):
            raise ValueError("4B requires explicit quota exhaustion and fresh pinned matched TRAIN")
    elif model != API_MODEL:
        raise ValueError("No generic transport error,9B or27B backbone substitution")
    summaries = report.get("arms", {})
    for arm in ("native", "cot_only", "cot_jev"):
        row = summaries.get(arm, {})
        if (
            row.get("observations") != 18 or row.get("complete_verified") is not True
            or type(row.get("successes")) is not int or not 0 <= row["successes"] <= 18
            or not usage_valid(row.get("actor_tokens"))
        ):
            raise ValueError("A partial, unmetered or unmatched TRAIN arm cannot open protected data")
    baseline, candidate = summaries["native"], summaries["cot_jev"]
    controls = [summaries[arm] for arm in ("native", "cot_only")]
    minimum = handoff.get("minimum_token_reduction_percent", 30)
    target = handoff.get("target_token_reduction_percent", 40)
    if (
        type(minimum) is not int or not 30 <= minimum < 100
        or type(target) is not int or not max(40, minimum) <= target < 100
    ):
        raise ValueError("Preserve the registered30% minimum and40% target outcome gates")
    if baseline["successes"] < 9:
        raise ValueError("A complete capable native TRAIN parent is required before held-out evaluation")
    positive_train_gate = (
        all(candidate["successes"] >= control["successes"] for control in controls)
        and all(
            candidate["actor_tokens"]["total_tokens"] < control["actor_tokens"]["total_tokens"]
            and candidate["actor_tokens"]["total_tokens"] * 100
            <= control["actor_tokens"]["total_tokens"] * (100 - minimum)
            for control in controls
        )
    )
    if policy == "qualified_only" and not positive_train_gate:
        raise ValueError("The native capability/quality/30% raw actor-cost gate against both controls failed")
    if (
        not isinstance(handoff.get("train_candidates"), list) or not handoff["train_candidates"]
        or len({row["sha256"] for row in handoff["train_candidates"]}) != len(handoff["train_candidates"])
        or handoff.get("policy", {}).get("enabled") is not True
        or handoff.get("maximum_workers") not in range(1, 9)
    ):
        raise ValueError("Declare TRAIN-frozen candidates, genuine native policy and an admitted queue")
    return {
        "passed": True, "model": model, "native_successes": baseline["successes"],
        "heldout_outcome_policy": policy, "positive_train_gate": positive_train_gate,
        "minimum_token_reduction_percent": minimum, "target_token_reduction_percent": target,
        "target_token_reduction_met_against_both_controls": all(
            candidate["actor_tokens"]["total_tokens"] * 100
            <= control["actor_tokens"]["total_tokens"] * (100 - target)
            for control in controls
        ),
    }


def protected_branches(candidates: list[str], settings: dict) -> list[tuple[str, str | None]]:
    enabled = settings.get("include_cot_only_control", False)
    if type(enabled) is not bool:
        raise ValueError("Declare the prospective matched CoT-only held-out control")
    return [("native", None), *(("cot_only", value) for value in candidates if enabled),
            *(("cot_jev", value) for value in candidates)]


def summary_key(arm: str, candidate: str | None) -> str:
    return "native" if arm == "native" else (
        "cot_only:" + candidate if arm == "cot_only" else candidate)


def task_plan(phase: str, task_ids: list[str], candidates: list[str], settings: dict | None = None) -> list[dict]:
    expected = 18 if phase == "validation" else 53 if phase == "test" else None
    if expected is None or len(task_ids) != expected or len(set(task_ids)) != expected:
        raise ValueError("Preserve complete canonical VAL18/TEST53 membership")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Use unique TRAIN-frozen candidates")
    if phase == "test" and len(candidates) != 1:
        raise ValueError("TEST measures exactly one validation-selected frozen candidate")
    seeds = [20260928] if phase == "validation" else test_seeds(settings or {}, SEEDS)
    return [
        {
            "phase": phase, "task_id": task, "seed": seed, "repetition": repetition,
            "arm": arm, "candidate_sha256": candidate,
            "slot": f"{phase}/{arm}/{candidate or 'baseline'}/{task}/{seed}",
        }
        for repetition, seed in enumerate(seeds)
        for task in task_ids
        for arm, candidate in protected_branches(candidates, settings or {})
    ]


def stage_summary(plan: list[dict], rows: list[dict]) -> dict:
    expected = {row["slot"] for row in plan}
    observed = [row["slot"] for row in rows]
    if len(set(observed)) != len(observed) or not set(observed).issubset(expected):
        raise ValueError("A protected task/seed/condition was counted twice or changed")
    summaries = {}
    call_ids = []
    complete = set(observed) == expected
    for row in rows:
        valid = (
            row.get("status") == "SCORED" and type(row.get("native_score")) in (int, float)
            and row["native_score"] in (0, 1)
            and row.get("cost", {}).get("usage_status") == "MEASURED"
            and usage_valid(row.get("cost", {}).get("usage"))
            and row.get("native_scorer_executed") is True
            and row.get("physical_calls_reconcile") is True
        )
        complete = complete and valid
        key = summary_key(row["arm"], row["candidate_sha256"])
        summary = summaries.setdefault(key, {
            "observations": 0, "successes": 0, "actor_tokens": {field: 0 for field in FIELDS},
            "native_fast_execution_proofs": 0,
        })
        summary["observations"] += 1
        if valid:
            summary["successes"] += int(row["native_score"])
            for field in FIELDS:
                summary["actor_tokens"][field] += row["cost"]["usage"][field]
        summary["native_fast_execution_proofs"] += len(
            (row.get("jev_audit") or {}).get("native_fast_execution_proofs", []),
        )
        call_ids.extend(row.get("physical_call_ids", []))
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("An actual provider call appeared in multiple protected task results")
    return {
        "complete": complete, "expected": len(plan), "observations": len(rows),
        "summaries": summaries if complete else None, "test_metrics_admitted": False,
        "unknown_or_unscored_slots": [
            row["slot"] for row in rows
            if row.get("status") != "SCORED" or row.get("cost", {}).get("usage_status") != "MEASURED"
            or row.get("native_scorer_executed") is not True
            or row.get("physical_calls_reconcile") is not True
        ],
        "cumulative_token_limit": None,
    }


def select_validation(
    summary: dict, frozen_candidates: list[str], *, outcome_policy: str = "qualified_only",
    include_cot_only_control: bool = False,
) -> str | None:
    if outcome_policy not in ("qualified_only", "all_outcomes"):
        raise ValueError("Declare a known held-out outcome policy")
    if summary.get("complete") is not True:
        return None
    rows = summary["summaries"]
    expected = {"native", *frozen_candidates}
    if include_cot_only_control:
        expected.update("cot_only:" + key for key in frozen_candidates)
    if set(rows) != expected:
        raise ValueError("Validation may select only the predeclared TRAIN-frozen bank")
    baseline = rows["native"]
    eligible = list(frozen_candidates) if outcome_policy == "all_outcomes" else [
        key for key in frozen_candidates
        if rows[key]["successes"] >= baseline["successes"]
        and rows[key]["actor_tokens"]["total_tokens"] < baseline["actor_tokens"]["total_tokens"]
        and rows[key]["native_fast_execution_proofs"] > 0
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda key: (
        -rows[key]["successes"], rows[key]["actor_tokens"]["total_tokens"], key,
    ))


def copy_verified(source: Path, destination: Path, expected: str) -> None:
    if sha(source) != expected:
        raise ValueError(f"The TRAIN owner source changed before protected freeze: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_owner_module(arm_root: Path, module_name: str):
    sys.path.insert(0, str(arm_root / "source"))
    return importlib.import_module(module_name)


def require_remote() -> None:
    if socket.gethostname() != REMOTE_HOSTNAME:
        raise RuntimeError("Every native trial/experiment/protected stage runs on authorized A800-4")


def prepare_stage(run: Path, handoff: dict, phase: str, selected: str | None = None) -> Path:
    require_remote()
    partition = verified(handoff["partition"])
    hashes = [row["sha256"] for row in handoff["train_candidates"]]
    candidates = [selected] if selected is not None else hashes
    plan = task_plan(phase, partition["splits"][phase], candidates, handoff)
    stage = run / phase
    stage.mkdir(mode=0o700)
    write(stage / "PLAN.json", plan)
    write(stage / "STAGE.json", {
        "phase": phase, "candidates": candidates, "model_identity": handoff["model_identity"],
        "task_ids": partition["splits"][phase], "partition": handoff["partition"],
        "shared_prompt_sha256": handoff["shared_prompt_sha256"],
        "provider_sampling_seed_supported": False, "cumulative_token_limit": None,
        "token_usage_mode": "unlimited", "no_cross_task_policy_updates": True,
    })
    branches = protected_branches(candidates, handoff)
    for arm, candidate in branches:
        root = stage / summary_key(arm, candidate).replace(":", "-")
        root.mkdir()
        (root / "private").mkdir(mode=0o700)
        (root / "process-cache").mkdir()
        descriptor = handoff["native_roots"][arm]
        config = verified(descriptor["config"])
        sources = verified(descriptor["source_manifest"])
        origin = Path(descriptor["config"]["path"]).parent
        for relative, checksum in sources.items():
            name = Path(relative)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("A native source closure escaped its frozen root")
            copy_verified(origin / "source" / name, root / "source" / name, checksum)
        for helper in handoff.get("additional_receipt_audit_sources", []):
            name = Path(helper["relative_path"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("A handed-off receipt auditor escaped its source closure")
            copy_verified(Path(helper["path"]), root / "source" / name, helper["sha256"])
            sources[str(name)] = helper["sha256"]
        module = load_owner_module(root, handoff["native_module"])
        config.update(
            protected_phase=phase, execution_stage=f"figure3_protected_{phase}",
            study_arm=arm, maximum_workers=handoff["maximum_workers"],
        )
        # The native library's legacy phase annotation is not a learning controller.
        # Only this orchestrator chooses the disjoint held-out membership.
        write(root / "CONFIG.json", {**config, "native_adapter_metadata_phase": config.get("phase")})
        write(root / "SOURCE_SHA256.json", sources)
        for name in ("NATIVE_RUNTIME.json", "SHARED_PROMPT.json", "VERIFIER_ENVIRONMENT.json"):
            shutil.copy2(origin / name, root / name)
        shutil.copyfile(Path(handoff["partition"]["path"]), root / "SPLIT_MANIFEST.json")
        frozen_tasks, prompts = {}, {}
        records = verified(handoff["canonical_task_records"])
        record_map = {record["id"]: record for record in records["records"]}
        for task_id in partition["splits"][phase]:
            task = Path(config["task_root"]) / task_id
            files = module.task_file_hashes(task)
            record = record_map[task_id]
            if module.canonical_sha256(files) != record["payload_sha256"]:
                raise ValueError("A protected official task definition differs from the canonical payload")
            frozen_tasks[task_id] = files
            prompts[task_id] = module.task_prompt_receipt(task, files)
        write(root / "TASK_SOURCE_SHA256.json", frozen_tasks)
        write(root / "TASK_PROMPT_RECEIPTS.json", prompts)
        write(root / "FROZEN_IDENTITY.json", {
            name: sha(root / name) for name in (
                "CONFIG.json", "SOURCE_SHA256.json", "NATIVE_RUNTIME.json", "SHARED_PROMPT.json",
                "VERIFIER_ENVIRONMENT.json", "SPLIT_MANIFEST.json",
                "TASK_SOURCE_SHA256.json", "TASK_PROMPT_RECEIPTS.json",
            )
        })
    write(stage / "PROVENANCE.json", {
        "at": now(), "hostname": socket.gethostname(), "seed": 20260928,
        "execution_seeds": [20260928] if phase == "validation" else test_seeds(handoff, SEEDS),
        "argv": sys.argv, "python": sys.version,
        "packages": sorted((dist.metadata["Name"], dist.version)
                           for dist in importlib.metadata.distributions()),
        "source_owner": handoff["train_owner"], "handoff_source": reference(run / "OWNER_HANDOFF.json"),
        "Git_state": handoff["git_state"], "CUDA_VISIBLE_DEVICES": "", "selected_gpus": [],
        "resource_class": "Native Docker CPU tasks plus owner-qualified shared API",
        "native_limits_and_task_definitions_unchanged": True,
    })
    return stage


def arm_root(stage: Path, item: dict) -> Path:
    return stage / summary_key(item["arm"], item["candidate_sha256"]).replace(":", "-")


def case_root(stage: Path, item: dict) -> Path:
    return arm_root(stage, item) / "tasks" / f"{item['task_id']}--{item['seed']}"


def launch_case(stage: Path, handoff: dict, item: dict) -> subprocess.Popen:
    root, directory = arm_root(stage, item), case_root(stage, item)
    directory.mkdir(parents=True)
    config = read(root / "CONFIG.json")
    module = load_owner_module(root, handoff["native_module"])
    command = module.trial_command(config, "deepseek", root, directory, item["task_id"])
    job_path = directory / "HARBOR_JOB.json"
    job = read(job_path)
    kwargs = job["agents"][0]["kwargs"]
    kwargs["seed"] = item["seed"]
    if item["arm"] in ("cot_only", "cot_jev"):
        candidate = next(row for row in handoff["train_candidates"]
                         if row["sha256"] == item["candidate_sha256"])
        artifact = verified(candidate["artifact"])
        instruction = directory / "instruction.txt"
        instruction.write_text(artifact["text"])
        if sha(instruction) != candidate["sha256"]:
            raise ValueError("A selected TRAIN-frozen instruction changed")
        kwargs.update(
            arm="chisel", cot_jev={**handoff["policy"], "enabled": item["arm"] == "cot_jev"},
            chisel={
                "path": str(instruction), "sha256": candidate["sha256"],
                "artifact_kind": "training-derived", "provenance": artifact["provenance"],
            },
        )
    write(job_path, job)
    cache = directory / "process-cache"
    cache.mkdir()
    write(directory / "COMMAND.json", {
        "argv": command, "phase": item["phase"], "seed": item["seed"],
        "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(cache), "cumulative_token_limit": None,
    })
    with (directory / "process.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=directory, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(root / "source"), "CUDA_VISIBLE_DEVICES": "",
                 "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(cache), "PYTHONUNBUFFERED": "1"},
        )
    write(directory / "PROCESS.json", {
        "pid": process.pid, "start_ticks": process_identity(process.pid),
        "at": now(), "slot": item["slot"],
    })
    return process


def harvest_case(stage: Path, handoff: dict, item: dict) -> dict:
    root, directory = arm_root(stage, item), case_root(stage, item)
    module = load_owner_module(root, handoff["native_module"])
    row = module.harvest(read(root / "CONFIG.json"), "deepseek", root, directory, item["task_id"])
    ledger = read(directory / "PROVIDER_CALLS.json")
    calls = ledger["calls"]
    total = {field: 0 for field in FIELDS}
    reconciled = bool(calls)
    for call in calls:
        valid = (
            call.get("ended") is True
            and call.get("model") == handoff["model_identity"]["model"]
            and usage_valid(call.get("usage"))
        )
        reconciled = reconciled and valid
        if valid:
            for field in FIELDS:
                total[field] += call["usage"][field]
    reconciled = reconciled and total == row["cost"]["usage"]
    verifier_path = directory / "VERIFIER_ACCEPTANCE.json"
    verdict = read(verifier_path) if verifier_path.is_file() else {}
    native_tests = verdict.get("executed_native_tests", {}).get("tests", 0)
    audited = None
    streams = sorted({call["stream_receipts"] for call in calls if call.get("stream_receipts")})
    if item["arm"] in ("cot_only", "cot_jev") and len(streams) == 1:
        auditor = handoff["jev_receipt_auditor"]
        audit_module = load_owner_module(root, auditor["module"])
        stream = Path(streams[0])
        try:
            records = [json.loads(line) for line in stream.read_text().splitlines()]
            audited = getattr(audit_module, auditor["function"])(records, read(stream.parent / "result.json"))
            if item["arm"] == "cot_only" and audited.get("fast_provider_calls", 0) != 0:
                raise ValueError("The CoT-only matched control made an undeclared fast call")
        except ValueError as error:
            write(directory / "JEV_AUDIT_BOUNDARY.json", {
                "error": str(error), "all_actual_costs_and_native_failure_preserved": True,
                "original_stream": reference(stream),
                "mechanism_audit_complete": False,
                "native_fast_execution_inferred": False,
            })
    write(directory / "JEV_RECEIPT_AUDIT.json", audited or {
        "all_physical_provider_costs_reconcile": reconciled,
        "total_tokens": total["total_tokens"] if reconciled else None,
        "native_fast_execution_proofs": [],
        "native_sdk_completion_not_inferred": True,
    })
    protected = {
        **row, **item, "phase": item["phase"], "arm": item["arm"],
        "phase_qualification": {
            "native_adapter_metadata_phase": row["phase"],
            "actual_canonical_partition": item["phase"],
            "native_learning_controller_invoked": False,
        },
        "native_scorer_executed": (
            type(native_tests) is int and native_tests > 0 and verdict.get("score") == row["native_score"]
        ),
        "physical_calls_reconcile": reconciled,
        "physical_call_ids": [call["call_id"] for call in calls], "jev_audit": audited,
        "native_adapter_result": reference(directory / "RESULT.json"),
    }
    write(directory / "PROTECTED_RESULT.json", protected)
    return protected


def docker_network_admission(maximum_bridge_networks: int = 26) -> dict:
    """Wait for subnet capacity without removing any network or stopping an owner."""
    output = subprocess.check_output([
        "docker", "network", "ls", "--filter", "driver=bridge", "--format", "{{.ID}}",
    ], text=True, timeout=30)
    count = len(output.split())
    return {"ready": count < maximum_bridge_networks, "active_bridge_networks": count,
            "maximum_bridge_networks": maximum_bridge_networks,
            "networks_removed": 0, "owners_preempted": 0}


def run_stage(stage: Path, handoff: dict) -> dict:
    plan = read(stage / "PLAN.json")
    processes, results = {}, {}
    dispatched = set()
    last_resource = 0.0
    while True:
        for item in plan:
            directory = case_root(stage, item)
            if (directory / "PROCESS.json").is_file():
                dispatched.add(item["slot"])
            process = processes.get(item["slot"])
            receipt_path = directory / "PROCESS.json"
            receipt_ = read(receipt_path) if receipt_path.is_file() else {}
            exited = (
                process is not None and process.poll() is not None
            ) or (receipt_ and not live_process(receipt_))
            if exited and item["slot"] not in results:
                write(case_root(stage, item) / "PROCESS_EXIT.json", {
                    "returncode": process.returncode if process is not None else None, "at": now(),
                    "reattached_without_actor_replay": process is None,
                })
                results[item["slot"]] = harvest_case(stage, handoff, item)
        rows = list(results.values())
        summary = stage_summary(plan, rows)
        active = {}
        for item in plan:
            path = case_root(stage, item) / "PROCESS.json"
            if path.is_file():
                receipt_ = read(path)
                if live_process(receipt_):
                    active[item["slot"]] = receipt_["pid"]
        write(stage / "STATUS.json", {
            **summary, "at": now(), "active": active, "pid": os.getpid(),
            "status": "COMPLETE" if summary["complete"] else "RUNNING",
            "selected_gpus": [], "CUDA_VISIBLE_DEVICES": "",
            "actor_replay_permitted": False,
        })
        if not active and summary["unknown_or_unscored_slots"]:
            write(stage / "BLOCKER.json", {
                "status": "NATIVE_INFRASTRUCTURE_OR_USAGE_BLOCKED",
                "slots": summary["unknown_or_unscored_slots"],
                "new_submissions_frozen": True, "actual_actor_calls_replayed": False,
            })
            return summary
        if len(results) == len(plan) and not active:
            return summary
        pending = [item for item in plan if item["slot"] not in dispatched]
        while (
            pending and len(active) < handoff["maximum_workers"]
            and not summary["unknown_or_unscored_slots"]
        ):
            item = pending.pop(0)
            root = arm_root(stage, item)
            native = load_owner_module(root, handoff["native_module"])
            config = read(root / "CONFIG.json")
            active_items = [row for row in plan if row["slot"] in active]
            task_paths = [
                Path(config["task_root"]) / row["task_id"] for row in [*active_items, item]
            ]
            capacity = native.task_capacity(config, root, task_paths)
            network = docker_network_admission()
            capacity["docker_network_capacity"] = network
            capacity["ready"] = capacity.get("ready") is True and network["ready"]
            if time.monotonic() - last_resource >= 300:
                write(stage / "RESOURCE_LATEST.json", capacity)
                with (stage / "RESOURCE_HISTORY.jsonl").open("a") as stream:
                    stream.write(json.dumps({"at": now(), **capacity}, sort_keys=True) + "\n")
                last_resource = time.monotonic()
            write(stage / "NEXT_TASK_ADMISSION.json", capacity)
            if capacity.get("ready") is not True or not native.storage_capacity(config, root)["ready"]:
                break
            process = launch_case(stage, handoff, item)
            processes[item["slot"]] = process
            dispatched.add(item["slot"])
            active[item["slot"]] = process.pid
            # Let Compose allocate its network before evaluating another slot.
            break
        time.sleep(5)


def stage_manifest(stage: Path, handoff: dict, candidate: str) -> dict:
    phase = read(stage / "STAGE.json")["phase"]
    result = {
        "schema": f"figure3-deepseek-terminal-{phase}/v1", "phase": phase,
        "benchmark": "terminal-bench-2.0", "model": handoff["model_identity"]["model"],
        "split_seed": 20260928, "shared_prompt_sha256": handoff["shared_prompt_sha256"],
        "execution_seeds": [20260928] if phase == "validation" else test_seeds(handoff, SEEDS),
        "provider_sampling_seed_supported": False, "native_limits_unchanged": True,
        "selected_candidate_sha256": candidate, "partition": handoff["partition"],
        "cumulative_token_limit": None, "token_usage_mode": "unlimited", "arms": {},
        "arm_identity": {}, "include_cot_only_control": handoff.get("include_cot_only_control", False),
        "bank_author_receipt": handoff.get("bank_author_receipt"),
    }
    if handoff.get("test_repetition_policy") is not None:
        result["test_repetition_policy"] = handoff["test_repetition_policy"]
    for arm in (("native", "cot_only", "cot_jev") if handoff.get("include_cot_only_control") else ARMS):
        selected = [
            item for item in read(stage / "PLAN.json")
            if item["arm"] == arm and item["candidate_sha256"] == (candidate if arm != "native" else None)
        ]
        root = arm_root(stage, selected[0])
        result["arm_identity"][arm] = {
            "config": reference(root / "CONFIG.json"),
            "source_manifest": reference(root / "SOURCE_SHA256.json"),
        }
        result["arms"][arm] = [
            {
                "task_id": item["task_id"], "seed": item["seed"], "repetition": item["repetition"],
                "result": reference(case_root(stage, item) / "PROTECTED_RESULT.json"),
                "native_record": reference(case_root(stage, item) / "native_result.json"),
                "provider_calls": reference(case_root(stage, item) / "PROVIDER_CALLS.json"),
                "verifier_acceptance": reference(case_root(stage, item) / "VERIFIER_ACCEPTANCE.json"),
                "jev_receipt_audit": reference(case_root(stage, item) / "JEV_RECEIPT_AUDIT.json"),
            } for item in selected
        ]
    return result


def execute(handoff_path: Path, config: dict, run: Path) -> Path | None:
    require_remote()
    handoff = read(handoff_path)
    policy = reporting_policy(handoff)
    if policy != reporting_policy(config):
        raise ValueError("Changed outcome reporting requires a new frozen controller and owner handoff")
    if handoff.get("include_cot_only_control", False) != config.get("include_cot_only_control", False):
        raise ValueError("Freeze the same matched CoT-only population before protected access")
    if test_seeds(handoff, SEEDS) != test_seeds(config, SEEDS):
        raise ValueError("Freeze the same prospective TEST repetition policy in handoff and controller")
    if handoff.get("protected_pipeline_already_owned") is True:
        raise ValueError("Adopt the existing owner's protected pipeline; do not create a competitor")
    admission = verified(handoff["train_admission"])
    owner_gate(handoff, admission)
    if handoff.get("fully_audited_train_receipt") is None:
        raise ValueError("The TRAIN owner must supply its final source-qualified audit receipt")
    verified(handoff["fully_audited_train_receipt"])
    if handoff["model_identity"]["model"] == FALLBACK_MODEL:
        verified(handoff["fallback_authorization"]["provider_response"])
        verified(handoff["model_identity"]["gpu_selection"])
        if FALLBACK_MODEL not in handoff.get("native_supported_models", []):
            raise ValueError("Wait for the owner's source-qualified pinned4B-compatible native bridge")
    if not run.is_relative_to(RUN_ROOT) or run == RUN_ROOT:
        raise ValueError("Use a new protected experiment directory on the authorized data volume")
    run.mkdir(mode=0o700, exist_ok=True)
    owner_copy = run / "OWNER_HANDOFF.json"
    if owner_copy.exists() and sha(owner_copy) != sha(handoff_path):
        raise ValueError("The protected continuation handoff changed after its immutable claim")
    if not owner_copy.exists():
        shutil.copyfile(handoff_path, owner_copy)
    write(run / "CONFIG.json", config)
    write(run / "LEASE.json", {
        "pid": os.getpid(), "start_ticks": process_identity(os.getpid()),
        "at": now(), "selected_gpus": [], "CUDA_VISIBLE_DEVICES": "",
        "resource_class": "CPU/API", "maximum_workers": handoff["maximum_workers"],
        "successor_task_id": "protected-validation18-then-frozen-test53x5",
        "source_owner_train_mutations": 0,
    })
    validation = run / "validation"
    if not (validation / "PLAN.json").is_file():
        validation = prepare_stage(run, handoff, "validation")
    elif not (validation / "PROVENANCE.json").is_file():
        raise ValueError("Preserve a partially prepared stage; repair it in a new source-frozen continuation")
    summary = run_stage(validation, handoff)
    hashes = [row["sha256"] for row in handoff["train_candidates"]]
    selected = select_validation(summary, hashes, outcome_policy=policy,
                                 include_cot_only_control=handoff.get("include_cot_only_control", False))
    if selected is None:
        write(run / "STATUS.json", {
            "at": now(), "status": "VALIDATION_NO_ADMISSIBLE_TRAIN_FROZEN_CANDIDATE",
            "summary": summary, "test_opened": False, "train_revisions_from_validation": False,
            "next_dependency": "TRAIN owner may conduct a new TRAIN-only research iteration",
        })
        return run
    validation_manifest = stage_manifest(validation, handoff, selected)
    write(validation / "RESULT_MANIFEST.json", validation_manifest)
    freeze = {
        "at": now(), "frozen_before_test": True, "test_feedback_used": False,
        "selected_candidate_sha256": selected, "train_candidate_sha256": hashes,
        "validation_manifest": reference(validation / "RESULT_MANIFEST.json"),
        "source_hashes": validation_manifest["arm_identity"],
        "test_execution_seeds": test_seeds(handoff, SEEDS),
        "test_repetition_policy": handoff.get("test_repetition_policy"),
        "heldout_outcome_policy": policy,
        "outcome_reporting_authorization": handoff.get("outcome_reporting_authorization"),
        "selection_rule": "Complete validation: native quality first, complete actor tokens second, "
                          "lexical tie break; retain the selected complete TEST regardless of sign",
    }
    write(run / "CANDIDATE_FREEZE.json", freeze)
    test = run / "test"
    if not (test / "PLAN.json").is_file():
        test = prepare_stage(run, handoff, "test", selected)
    elif not (test / "PROVENANCE.json").is_file():
        raise ValueError("Preserve a partially prepared TEST; do not replay paid actors")
    final = run_stage(test, handoff)
    if final["complete"] is not True:
        write(run / "STATUS.json", {
            "at": now(), "status": "TEST_SOURCE_OR_USAGE_PENDING", "summary": final,
            "complete_test_available": False,
        })
        return run
    manifest = stage_manifest(test, handoff, selected)
    manifest.update(
        train_admission=handoff["train_admission"],
        protected_validation=reference(validation / "RESULT_MANIFEST.json"),
        candidate_freeze=reference(run / "CANDIDATE_FREEZE.json"),
        separate_development_accounting=handoff["separate_development_accounting"],
        heldout_outcome_policy=policy,
        outcome_reporting_authorization=handoff.get("outcome_reporting_authorization"),
    )
    write(test / "RESULT_MANIFEST.json", manifest)
    from scripts.validate_figure3_deepseek_manifest import validate

    publication = validate(manifest)
    write(test / "PUBLICATION.json", publication)
    write(run / "STATUS.json", {
        "at": now(), "status": "COMPLETE_NATIVE_TEST", "complete_test_available": True,
        "publication": reference(test / "PUBLICATION.json"),
        "null_and_negative_results_retained": True, "test_supplied_revisions": False,
        "cumulative_token_limit": None,
    })
    return run


def watch(config: dict) -> int:
    require_remote()
    output, handoff_path = Path(config["state_directory"]), Path(config["owner_handoff"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "OWNER.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claimed_path = output / "CLAIMED_HANDOFF.json"
        while True:
            state = {
                "at": now(), "pid": os.getpid(), "status": "WAITING_FULL_TRAIN_OWNER_HANDOFF",
                "handoff_path": str(handoff_path), "benchmark_or_model_calls": 0,
                "complete_test_available": False, "cumulative_token_limit": None,
                "training_owner_implementation_rechecked": False,
            }
            if handoff_path.is_file():
                try:
                    claim = read(claimed_path) if claimed_path.is_file() else None
                    if claim is not None and claim["handoff_sha256"] != sha(handoff_path):
                        raise ValueError("Do not switch a claimed protected run to another owner's source")
                    if claim is None:
                        handoff = read(handoff_path)
                        owner_gate(handoff, verified(handoff["train_admission"]))
                        run = RUN_ROOT / (
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-figure3-protected"
                        )
                        claim = {
                            "at": now(), "handoff_sha256": sha(handoff_path), "run": str(run),
                            "controller_source_sha256": sha(Path(__file__)),
                        }
                        write(claimed_path, claim)
                    if claim["controller_source_sha256"] != sha(Path(__file__)):
                        raise ValueError("A changed controller requires a source-frozen continuation, not blind resume")
                    run = execute(handoff_path, config, Path(claim["run"]))
                    state.update(
                        status="PROTECTED_PIPELINE_TERMINAL", run=str(run),
                        terminal=read(run / "STATUS.json") if run is not None else None,
                    )
                    write(output / "STATUS.json", state)
                    return 0
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                    state.update(status="OWNER_HANDOFF_OR_PROTECTED_STAGE_BLOCKED", error=str(error))
            write(output / "STATUS.json", state)
            time.sleep(config.get("poll_seconds", 30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    require_remote()
    os.umask(0o077)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = read(args.config)
    if not args.watch:
        parser.error("Use the detached owner-gated --watch coordinator")
    return watch(config)


if __name__ == "__main__":
    raise SystemExit(main())
