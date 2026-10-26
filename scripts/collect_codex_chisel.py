#!/usr/bin/env python3
"""Audit native paired test receipts and collect the Codex comparison status."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import write_json_atomic  # noqa: E402
from scripts.run_codex_terminal import native_reward  # noqa: E402
from scripts.serve_codex_deepseek import native_response_usage, usage_summary  # noqa: E402
from scripts.collect_deepseek_harness_progress import controller_running  # noqa: E402
from scripts.run_codex_chisel_terminal import registered_provider_ledger  # noqa: E402


def latest_version(parent: Path, prefix: str, *, default: int = 1) -> Path:
    candidates = []
    for state in parent.glob(prefix + "*/STATE.json"):
        match = re.fullmatch(re.escape(prefix) + r"([1-9][0-9]*)", state.parent.name)
        if match:
            candidates.append((int(match.group(1)), state.parent))
    return max(candidates, default=(default, parent / f"{prefix}{default}"))[1]


def validate_terminal_test_source(path: Path, root: Path, comparison: Path, config: dict) -> None:
    test_roots = (root / "terminal/test", comparison / "trials/test")
    if any(path.resolve().is_relative_to(directory.resolve()) for directory in test_roots):
        return
    entry = config.get("retained_test_records", {}).get(str(path))
    previous = Path(config.get("frozen_test_predecessor", ""))
    if (
        not entry or previous.parent != root / "terminal"
        or not re.fullmatch(r"chisel-comparison-v[1-9][0-9]*", previous.name)
        or previous == comparison
        or not path.resolve().is_relative_to((previous / "trials/test").resolve())
        or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
    ):
        raise ValueError("A final result references an artifact outside this frozen test lineage")
    policy = json.loads((previous / "POLICY.json").read_text())
    if (
        policy != json.loads((comparison / "POLICY.json").read_text())
        or entry["policy_sha256"] != policy["policy_sha256"]
    ):
        raise ValueError("Retained test outcome belongs to a different frozen policy")


def native_training_progress(index: Path, expected: int) -> dict:
    admitted = json.loads(index.read_text())
    if admitted["status"] != "VERIFIED_TRAIN_BASELINE_INDEX" or len(admitted["tasks"]) != expected:
        raise ValueError("The current training index is incomplete or unverified")
    scores = {}
    tokens = 0
    for task_id, entry in admitted["tasks"].items():
        path = Path(entry["result"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("A verified training receipt changed")
        row = json.loads(path.read_text())
        usage = row["provider_usage"]
        if (
            row["status"] != "SCORED" or row["task_id"] != task_id or row["phase"] != "train"
            or row["method"] != "codex" or usage["usage_status"] != "MEASURED"
            or type(row["score"]) not in (int, float) or row["score"] not in (0, 1)
            or type(usage["known_usage"]["total_tokens"]) is not int
            or usage["known_usage"]["total_tokens"] < 0
        ):
            raise ValueError("Training coverage requires actual native scores and measured usage")
        scores[task_id] = row["score"]
        tokens += usage["known_usage"]["total_tokens"]
    return {
        "status": admitted["status"], "scored": len(scores), "expected": expected,
        "active": [], "failed": {}, "scores": scores,
        "known_baseline_acquisition_tokens": tokens, "baseline_index": str(index),
        "scope": "Optimization training data; not final held-out performance",
    }


def paired_summary(baseline: list[dict], chisel: list[dict], expected: set[tuple]) -> dict:
    indexed = []
    for rows in (baseline, chisel):
        by_pair = {(row["task_id"], row["seed"]): row for row in rows}
        if len(by_pair) != len(rows) or set(by_pair) != expected:
            raise ValueError("Final paired coverage is duplicated, incomplete or mismatched")
        for row in rows:
            if (
                type(row["score"]) not in (int, float) or not math.isfinite(row["score"])
                or not 0 <= row["score"] <= 1
                or type(row["total_tokens"]) is not int or row["total_tokens"] < 0
            ):
                raise ValueError("Only measured native scores and token counts are admitted")
        indexed.append(by_pair)
    if not expected:
        raise ValueError("A nonempty final population is required")
    groups = {}
    for pair in sorted(expected):
        left, right = indexed[0][pair], indexed[1][pair]
        if left["unit_id"] != right["unit_id"]:
            raise ValueError("Matched rows must retain the same independent unit")
        group = groups.setdefault(left["unit_id"], {"score_delta": 0.0, "count": 0})
        group["score_delta"] += right["score"] - left["score"]
        group["count"] += 1
    units = list(groups.values())
    generator = random.Random(20260928)
    samples = []
    for _ in range(10000):
        drawn = generator.choices(units, k=len(units))
        samples.append(math.fsum(row["score_delta"] for row in drawn) /
                       sum(row["count"] for row in drawn))
    samples.sort()
    methods = {}
    for name, rows in zip(("codex", "chisel_codex"), (baseline, chisel)):
        methods[name] = {
            "observations": len(rows), "independent_units": len(groups),
            "native_mean_score": math.fsum(row["score"] for row in rows) / len(rows),
            "total_execution_tokens": sum(row["total_tokens"] for row in rows),
            "mean_execution_tokens": sum(row["total_tokens"] for row in rows) / len(rows),
        }
    base_tokens = methods["codex"]["total_execution_tokens"]
    return {
        "status": "VERIFIED_PAIRED_TEST", "methods": methods,
        "score_delta": methods["chisel_codex"]["native_mean_score"] - methods["codex"]["native_mean_score"],
        "paired_unit_bootstrap_ci95": [samples[249], samples[9749]],
        "bootstrap_samples": 10000, "bootstrap_seed": 20260928,
        "token_reduction_fraction": (
            1 - methods["chisel_codex"]["total_execution_tokens"] / base_tokens
            if base_tokens else None
        ),
    }


def audit_terminal(root: Path, config: dict) -> dict:
    comparison = latest_version(root / "terminal", "chisel-comparison-v")
    configuration = comparison / "CONFIG.json"
    if configuration.is_file():
        current = json.loads(configuration.read_text())
        if (
            Path(current["root"]).resolve() != root.resolve()
            or current.get("terminal_comparison_id", "chisel-comparison-v1") != comparison.name
        ):
            raise ValueError("Terminal configuration belongs to a different comparison")
        config = current
    state_path = comparison / "STATE.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "NOT_STARTED"}
    acquisition = root / "terminal/training-acquisition/STATE.json"
    training = json.loads(acquisition.read_text()) if acquisition.exists() else None
    output = {
        "status": state["status"], "final_comparison": None,
        "test_opened": state.get("test_opened", False),
        "training": None if training is None else {
            "status": training["status"], "scored": len(training["completed"]),
            "expected": training["expected"], "active": sorted(training["active"]),
            "failed": training["failed"],
            "scores": {key: row["score"] for key, row in training["completed"].items()},
            "known_baseline_acquisition_tokens": sum(
                row["provider_usage"]["total_tokens"] for row in training["completed"].values()
            ),
            "scope": "Optimization training data; not final held-out performance",
        },
        "controller_state_path": str(state_path),
        "controller_state": state,
        "controller_running": controller_running(state.get("pid"), (
            "resume_frozen_codex_terminal_test.py" if config.get("frozen_test_predecessor")
            else "run_codex_chisel_terminal.py"
        )),
    }
    if state.get("error"):
        output["blocker"] = state["error"]
    recovery = root / "terminal/recovery-v1"
    if (recovery / "PROTOCOL.json").exists():
        protocol = json.loads((recovery / "PROTOCOL.json").read_text())
        output["infrastructure_recovery"] = {
            "protocol": str(recovery / "PROTOCOL.json"),
            "failed_attempts": protocol["failed_attempts"],
            "successor": json.loads((recovery / "SUCCESSOR.json").read_text())
            if (recovery / "SUCCESSOR.json").exists() else None,
            "retry_states": {
                path.parent.name: {
                    key: record.get(key) for key in ("status", "score", "error", "provider_usage")
                }
                for path in recovery.glob("train/baseline/r0/*/RUN.json")
                for record in [json.loads(path.read_text())]
            },
        }
    reconciliation = root / "terminal/accounting-reconciliation-v1/STATE.json"
    if reconciliation.exists():
        value = json.loads(reconciliation.read_text())
        output["late_usage_reconciliation"] = {
            "status": value["status"], "completed": len(value["completed"]),
            "pending": value["pending"], "additional_model_calls": 0,
            "successor_pid": value.get("successor_pid"),
        }
    sam_handoff = latest_version(root / "terminal", "sam-native-handoff-v") / "STATE.json"
    if sam_handoff.exists():
        handoff = json.loads(sam_handoff.read_text())
        output["sam_native_handoff"] = handoff
        output["sam_native_handoff_path"] = str(sam_handoff)
        if (
            state["status"] in ("NOT_STARTED", "BLOCKED")
            and handoff["status"] == "WAITING_FOR_NATIVE_SAM_SCORE"
            and controller_running(handoff.get("pid"), "watch_codex_sam_handoff.py") is True
        ):
            output.update(
                status="WAITING_FOR_NATIVE_SAM_SCORE", historical_controller_state=state,
                controller_state=handoff, controller_state_path=str(sam_handoff),
                controller_running=True,
            )
    index = sam_handoff.parent / "BASELINE_INDEX.json"
    if index.exists():
        admitted = json.loads(index.read_text())
        output["historical_training_acquisition"] = output["training"]
        output["training"] = native_training_progress(
            index, config["benchmarks"]["terminal-bench-2.0"]["counts"]["train"],
        )
        output["native_training_coverage"] = {
            "status": admitted["status"], "scored": len(admitted["tasks"]),
            "expected": config["benchmarks"]["terminal-bench-2.0"]["counts"]["train"],
            "source": str(index),
        }
    elif reconciliation.exists():
        admitted = json.loads(reconciliation.read_text())
        output["native_training_coverage"] = {
            "status": "AWAITING_NATIVE_VERDICTS", "scored": len(admitted["completed"]),
            "pending": admitted["pending"], "source": str(reconciliation),
        }
    report_path = comparison / "CONTROLLER_REPORT.json"
    if state["status"] != "COMPLETE" or not report_path.exists():
        return output
    report = json.loads(report_path.read_text())
    terminal = config["benchmarks"]["terminal-bench-2.0"]
    partition = json.loads(
        (Path(terminal["partition_directory"]) / "SPLIT_MANIFEST.json").read_text()
    )
    expected = {
        (task, 20260928 + repetition)
        for task in partition["splits"]["test"]
        for repetition in range(terminal["test_repetitions"])
    }
    collected = {}
    for arm, method in (("baseline", "codex"), ("chisel", "chisel_codex")):
        rows = []
        for row in report["test"][arm]["rows"]:
            result = row["result"]
            path = Path(result["provenance"]["native_record"])
            validate_terminal_test_source(path, root, comparison, config)
            if hashlib.sha256(path.read_bytes()).hexdigest() != result["provenance"]["native_record_sha256"]:
                raise ValueError("A native final result changed after recording")
            native_record = json.loads(path.read_text())
            native = json.loads((path.parent / "native_result.json").read_text())
            native_receipts = native_record["provider_usage"]["receipts"]
            ledger = Path(native_receipts[0]).parents[3]
            if not registered_provider_ledger(root, Path(config["provider"]["ledger"]), ledger):
                raise ValueError("Final provider ledger is outside this campaign")
            usage = usage_summary(ledger, native_record["trial_id"])
            if (
                native_record["status"] != "SCORED" or native_record["phase"] != "test"
                or native_record["method"] != method or usage["usage_status"] != "MEASURED"
                or usage["provider_models"] != ["deepseek-flash"]
                or usage["known_usage"]["total_tokens"] != result["usage"]["total_tokens"]
                or native_reward(native) != result["native_score"]
            ):
                raise ValueError("Final native score, method or provider usage failed reconciliation")
            rows.append({
                "task_id": row["task_id"], "unit_id": row["unit_id"], "seed": row["seed"],
                "score": result["native_score"], "total_tokens": result["usage"]["total_tokens"],
            })
        collected[arm] = rows
    output["final_comparison"] = paired_summary(
        collected["baseline"], collected["chisel"], expected
    )
    output["optimization_and_selection_accounting"] = report["accounting"]
    policy = json.loads((comparison / "POLICY.json").read_text())
    output["selected_policy"] = {
        "candidate_id": policy["candidate"]["id"],
        "residual_enabled": policy["candidate"]["enabled"],
        "test_residual_applications": report["test"]["chisel"]["residual_applications"],
        "policy_sha256": policy["policy_sha256"],
    }
    return output


def audit_vqa(root: Path) -> dict:
    original = Path("/path/to/data/phiagent/work/codex-chisel/20260929-v1/vqav2-comparison-v1")
    work = latest_version(original.parent, "vqav2-comparison-v")
    state_path = work / "STATE.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "NOT_STARTED"}
    pilot_path = root / (
        "vqav2/train/baseline/r0/20260929T082045Z-native-training-pilot/ACCEPTANCE.json"
    )
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    result = {
        "status": state["status"], "final_comparison": None,
        "dataset_interpretation": "Provisional VQA=VQAv2",
        "metric": "Official VQA soft accuracy, not binary task success",
        "controller_state": state, "controller_root": str(work),
        "controller_state_path": str(state_path),
        "controller_running": controller_running(state.get("pid"), "run_codex_vqav2_campaign.py"),
        "training_pilot": pilot,
    }
    lifecycle = original.parent / "vqav2-lifecycle-recovery-v6"
    if (lifecycle / "STATE.json").exists():
        result["lifecycle_recovery"] = json.loads((lifecycle / "STATE.json").read_text())
        inherited = lifecycle / "INHERITED_RESULTS.json"
        if inherited.exists():
            prior = json.loads(inherited.read_text())
            result["retained_training_observations"] = len(prior["retained"])
            result["preserved_lifecycle_failures"] = prior["preserved_failures"]
    image_queue = original.with_name("vqav2-comparison-v4-queue") / "STATE.json"
    if image_queue.exists():
        result["image_runtime_handoff"] = json.loads(image_queue.read_text())
        if work.name in {f"vqav2-comparison-v{version}" for version in (1, 2, 3)}:
            result["status"] = result["image_runtime_handoff"]["status"]
    context_regression = original.with_name("vqav2-context-regression-v5")
    if (context_regression / "STATE.json").exists():
        result["native_context_guard"] = json.loads((context_regression / "STATE.json").read_text())
        accepted = context_regression / "ACCEPTANCE.json"
        if accepted.exists():
            result["native_context_guard_acceptance"] = json.loads(accepted.read_text())
        if work.name in {f"vqav2-comparison-v{version}" for version in (1, 2, 3, 4)}:
            result["status"] = result["native_context_guard"]["status"]
    quarantine = original / "INTEGRITY_QUARANTINE.json"
    if quarantine.exists():
        result["quarantined_predecessor"] = json.loads(quarantine.read_text())
        result["historical_pilot_reused"] = False
        if work == original:
            result["status"] = "QUARANTINED"
            return result
    network = root / "preflight/vqa-internal-network-v1/ACCEPTANCE.json"
    if network.exists():
        network_receipt = json.loads(network.read_text())
        result["network_isolation"] = {
            "status": network_receipt["status"],
            "network": network_receipt["network_inspect"]["Name"],
            "internal": network_receipt["network_inspect"]["Internal"],
            "external_egress_test": "EXTERNAL_EGRESS_BLOCKED" in network_receipt["stdout"],
            "receipt": str(network),
        }
    config = work / "CONFIG.json"
    if config.exists():
        recorded = json.loads(config.read_text())
        result["image_counts"] = recorded["counts"]
        result["question_counts"] = recorded["question_counts"]
        result["optimizer_units_per_iteration"] = recorded["optimizer_max_units"]
    events = work / "EVENTS.jsonl"
    if events.exists():
        rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        if len({row["request_id"] for row in rows}) != len(rows):
            raise ValueError("VQA callback accounting contains duplicate request IDs")
        result["completed_callback_usage"] = {
            "callbacks": len(rows),
            "known_total_tokens": sum(row["usage"].get("total_tokens") or 0 for row in rows),
            "unknown_usage_callbacks": sum(row["usage"].get("total_tokens") is None for row in rows),
            "scope": "All completed callback stages; final execution and optimization remain separate",
        }
    if state["status"] != "COMPLETE":
        return result
    report = json.loads((work / "CONTROLLER_REPORT.json").read_text())
    partition = json.loads((
        Path("/opt/phiagent/work/deepseek-harness/20260929-v1/assets/vqav2/partition.json")
    ).read_text())
    expected = {(str(qid), 20260928) for qid in partition["question_ids"]["test"]}
    grouped = {}
    for arm in ("baseline", "chisel"):
        verified = []
        for row in report["test"][arm]["rows"]:
            callback = row["result"]
            path = Path(callback["provenance"]["native_receipt"])
            if not path.resolve().is_relative_to((work / "trials/test").resolve()):
                raise ValueError("VQA test receipt is outside the frozen test directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != callback["provenance"]["native_receipt_sha256"]:
                raise ValueError("VQA native receipt changed after scoring")
            native = json.loads(path.read_text())
            score = json.loads((path.parent / "OFFICIAL_SCORE.json").read_text())
            receipts = native["provider_usage"]["receipts"]
            usage = usage_summary(Path(receipts[0]).parents[3], native["trial_id"])
            if (
                usage["usage_status"] != "MEASURED"
                or usage["known_usage"]["total_tokens"] != callback["usage"]["total_tokens"]
                or score["official_vqa_accuracy_01"] != callback["native_score"]
                or native.get("docker_network") != "phiagent-codex-vqa-20260929"
                or native.get("network_admission", {}).get("Internal") is not True
            ):
                raise ValueError("VQA native scoring or usage failed independent reconciliation")
            verified.append({
                "task_id": row["task_id"], "unit_id": row["unit_id"], "seed": row["seed"],
                "score": callback["native_score"], "total_tokens": callback["usage"]["total_tokens"],
            })
        grouped[arm] = verified
    result["final_comparison"] = paired_summary(grouped["baseline"], grouped["chisel"], expected)
    result["optimization_and_selection_accounting"] = report["accounting"]
    policy = json.loads((work / "POLICY.json").read_text())
    result["selected_policy"] = {
        "candidate_id": policy["candidate"]["id"],
        "residual_enabled": policy["candidate"]["enabled"],
        "test_residual_applications": report["test"]["chisel"]["residual_applications"],
        "policy_sha256": policy["policy_sha256"],
    }
    result["unanswered_test_timeouts"] = {
        arm: sum(row["result"]["execution_status"] == "timeout" for row in report["test"][arm]["rows"])
        for arm in ("baseline", "chisel")
    }
    return result


def audit_gaia(root: Path) -> dict:
    trials = list((root / "gaia2/train/baseline/r0").glob("*/RESULT.json"))
    receipts = [json.loads(path.read_text()) for path in trials]
    queues = sorted(
        (root / "gaia2").glob("native-pilot-queue-v*/STATE.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    queue = json.loads(queues[-1].read_text()) if queues else None
    transfer = root / "gaia2/assets/native-filesystem-v3.TRANSFER.json"
    verified_assets = json.loads(transfer.read_text()) if transfer.exists() else None
    full = latest_version(root / "gaia2", "full-comparison-v") / "STATE.json"
    full_state = json.loads(full.read_text()) if full.exists() else None
    controller_script = (
        "resume_frozen_codex_gaia2_test.py"
        if (full.parent / "FROZEN_TEST_CONTINUATION.json").is_file()
        else "run_codex_gaia2_campaign.py"
    )
    result = {
        "status": full_state["status"] if full_state else (
            queue["status"] if queue else "NATIVE_ASSET_PREPARATION"
        ),
        "final_comparison": None, "pilot_receipts": receipts,
        "native_pilot_queue": queue, "full_comparison_state": full_state,
        "controller_state_path": str(full),
        "controller_running": controller_running(
            full_state.get("pid") if full_state else None, controller_script,
        ),
        "observation_protocol_diagnostics": [
            str(path) for path in (root / "gaia2").glob("full-comparison-v*/OBSERVATION_PROTOCOL_SUPERSEDED.json")
        ],
        "complete_filesystem": None if verified_assets is None else {
            "status": verified_assets["status"], "files": len(verified_assets["records"]),
            "unverified": verified_assets.get("unverified"), "path": verified_assets["target"],
        },
        "assets": [
            {"path": str(path), "status": value["status"],
             "verified": len(value.get("verified", {})), "failed": len(value.get("failed", {}))}
            for path in (root / "gaia2/assets").glob("*native-assets/STATE.json")
            for value in [json.loads(path.read_text())]
        ],
    }
    superseded = full.parent / "OBSERVATION_PROTOCOL_SUPERSEDED.json"
    if superseded.exists():
        result.update(
            status="PROTOCOL_SUPERSEDED",
            protocol_superseded=json.loads(superseded.read_text()),
        )
        return result
    if not full_state or full_state["status"] != "COMPLETE":
        return result
    work = full.parent
    report = json.loads((work / "CONTROLLER_REPORT.json").read_text())
    selected = json.loads((work / "POLICY.json").read_text())
    command = json.loads((work / "provenance.json").read_text())["command"]
    partition_path = Path(command[command.index("--partition") + 1])
    if hashlib.sha256(partition_path.read_bytes()).hexdigest() != selected["sources"]["external"]["partition"]:
        raise ValueError("The native GAIA2 partition changed after policy selection")
    partition = json.loads(partition_path.read_text())
    seeds = selected["config"]["test_execution_seeds"]
    expected = {(unit, seed) for unit in partition["splits"]["test"] for seed in seeds}
    if len(partition["splits"]["test"]) != 96 or len(seeds) != 3 or len(expected) != 96 * 3:
        raise ValueError("GAIA2 must preserve the full96 scenarios and all three test repetitions")
    matched = {}
    judge_usage = {}
    test_roots = [work / "trials/test"]
    continuation_path = work / "FROZEN_TEST_CONTINUATION.json"
    continuation = json.loads(continuation_path.read_text()) if continuation_path.exists() else None
    if continuation:
        if (
            continuation["policy_sha256"] != selected["policy_sha256"]
            or continuation["source_sha256"] != selected["source_sha256"]
            or continuation["policy_or_output_limit_changes"] is not False
        ):
            raise ValueError("Frozen GAIA2 test continuation changed its method")
        previous = Path(continuation["previous"])
        if not previous.resolve().is_relative_to((root / "gaia2").resolve()):
            raise ValueError("Frozen test predecessor is outside this campaign")
        test_roots.append(previous / "trials/test")
        result["frozen_test_continuation"] = continuation
    native_failures = {"baseline": 0, "chisel": 0}
    for arm in ("baseline", "chisel"):
        rows = []
        judges = []
        for row in report["test"][arm]["rows"]:
            callback = row["result"]
            path = Path(callback["provenance"]["native_record"])
            if not any(path.resolve().is_relative_to(directory.resolve()) for directory in test_roots):
                raise ValueError("GAIA2 final result is outside its frozen test directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != callback["provenance"]["native_record_sha256"]:
                raise ValueError("GAIA2 native receipt changed")
            native = json.loads(path.read_text())
            admission_path = callback["provenance"].get("native_failure_admission")
            if admission_path:
                admitted_path = Path(admission_path)
                if not admitted_path.resolve().is_relative_to((work / "native-failure-admissions").resolve()):
                    raise ValueError("Native failure admission is outside the frozen continuation")
                if hashlib.sha256(admitted_path.read_bytes()).hexdigest() != callback["provenance"]["native_failure_admission_sha256"]:
                    raise ValueError("Native failure admission changed")
                admitted = json.loads(admitted_path.read_text())
                if (
                    admitted["status"] != "VERIFIED_NATIVE_AGENT_FAILURE"
                    or admitted["policy_sha256"] != selected["policy_sha256"]
                    or admitted["source_sha256"] != selected["source_sha256"]
                    or admitted["additional_model_calls"] != 0
                    or admitted["failed_trial_resampled"] is not False
                    or admitted["output_limits_changed"] is not False
                    or admitted["original_result_sha256"] != callback["provenance"]["native_record_sha256"]
                ):
                    raise ValueError("Native failure was altered or resampled")
                native_path = Path(admitted["native_result"])
                if hashlib.sha256(native_path.read_bytes()).hexdigest() != admitted["native_result_sha256"]:
                    raise ValueError("Recorded native failure result changed")
                scored = json.loads(native_path.read_text())
                if scored["success"] is not False or not scored.get("exception"):
                    raise ValueError("Missing actual native failure verdict")
                actor_path = Path(admitted["actor_result"])
                if hashlib.sha256(actor_path.read_bytes()).hexdigest() != admitted["actor_result_sha256"]:
                    raise ValueError("The failed actor receipt changed")
                rule = Path(admitted["official_rule_source"])
                if hashlib.sha256(rule.read_bytes()).hexdigest() != admitted["official_rule_sha256"]:
                    raise ValueError("The official native failure aggregation rule changed")
                actor = json.loads(actor_path.read_text())
                receipts = admitted["provider_usage"]["receipts"]
                judge = admitted["judge_accounting"]
                native_failures[arm] += 1
                native_admitted = True
            else:
                scored = json.loads(Path(native["outputs"]["native_result"]).read_text())
                actor = json.loads((Path(native["outputs"]["actor_run"]) / "RUN.json").read_text())
                receipts = actor["provider_usage"]["receipts"]
                judge = native["provider_usage"]["judge"]
                native_admitted = native["status"] == "COMPLETED" and native["split"] == "test"
            usage = usage_summary(Path(receipts[0]).parents[3], actor["trial_id"])
            if (
                not native_admitted
                or type(scored["success"]) is not bool
                or int(scored["success"]) != callback["native_score"]
                or usage["usage_status"] != "MEASURED"
                or usage["known_usage"]["total_tokens"] != callback["usage"]["total_tokens"]
                or judge["complete"] is not True
            ):
                raise ValueError("GAIA2 native score or role-separated usage failed reconciliation")
            rows.append({
                "task_id": row["task_id"], "unit_id": row["unit_id"], "seed": row["seed"],
                "score": callback["native_score"], "total_tokens": callback["usage"]["total_tokens"],
            })
            judges.append(judge["usage"])
        matched[arm] = rows
        judge_usage[arm] = {
            name: sum(record[name] for record in judges)
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
    result["final_comparison"] = paired_summary(matched["baseline"], matched["chisel"], expected)
    result["final_judge_usage_separate"] = judge_usage
    result["native_agent_failures_preserved"] = native_failures
    result["optimization_and_selection_accounting"] = report["accounting"]
    result["policy_sha256"] = selected["policy_sha256"]
    result["selected_policy"] = {
        "candidate_id": selected["candidate"]["id"],
        "residual_enabled": selected["candidate"]["enabled"],
        "test_residual_applications": report["test"]["chisel"]["residual_applications"],
        "policy_sha256": selected["policy_sha256"],
    }
    return result


def collect(root: Path, config: dict) -> dict:
    setup = []
    preflight_roots = (
        root / "preflight",
        Path("/path/to/data/phiagent/work/codex-chisel/20260929-v1/preflight"),
    )
    result_paths = sorted({
        path for directory in preflight_roots for path in directory.glob("*/RESULT.json")
    })
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        if result.get("status") in ("NATIVE_RESPONSES_ACCEPTED", "NATIVE_JSON_ACCEPTED"):
            raw_path = result_path.parent / "native-responses/response.txt"
            response = json.loads(raw_path.read_text())
            usage = native_response_usage(response)
            if result["usage"] != usage:
                raise ValueError("Capability usage does not match the raw native response")
            setup.append({
                "scope": result["status"], "usage": usage, "receipt": str(result_path),
            })
        elif result_path.parent.name.endswith("-codex-native-image-grid"):
            for index, item in enumerate(result.get("results", [])):
                usage = item.get("provider_usage")
                if usage and usage["usage_status"] == "MEASURED":
                    setup.append({
                        "scope": "NATIVE_CODEX_VISUAL_CAPABILITY",
                        "case": index, "usage": usage["known_usage"],
                        "passed": item["passed"], "receipt": str(result_path),
                    })
        elif result_path.parent.name.endswith("-codex-native-context-compaction"):
            usage = result.get("provider_usage")
            if usage and usage["usage_status"] == "MEASURED":
                setup.append({
                    "scope": "NATIVE_CODEX_CONTEXT_COMPACTION",
                    "usage": usage["known_usage"], "receipt": str(result_path),
                    "status": result["status"], "cumulative_token_limit": None,
                })
        elif result_path.parent.name.endswith("-disconnected-client-capture"):
            usage = result.get("provider_usage")
            if usage and usage["usage_status"] == "MEASURED":
                setup.append({
                    "scope": "DISCONNECTED_CLIENT_NATIVE_USAGE_CAPTURE",
                    "usage": usage["known_usage"], "receipt": str(result_path),
                    "status": result["status"],
                })
    for result_path in sorted((root / "preflight").glob("*/ACCEPTANCE.json")):
        result = json.loads(result_path.read_text())
        usage = result.get("provider_usage")
        if usage and usage["usage_status"] == "MEASURED":
            setup.append({
                "scope": "ACTUAL_CODEX_TOOL_CAPABILITY", "usage": usage["known_usage"],
                "receipt": str(result_path),
            })
    storage_states = sorted(
        (root / "storage-relocation").glob("*/STATE.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    return {
        "schema": "phiagent-codex-chisel-comparison-report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "root": str(root), "model": config["provider"]["model"],
        "codex_version": config["codex"]["version"],
        "wire_api": "native_responses", "cumulative_token_limit": None,
        "setup": {"known_total_tokens": sum(row["usage"]["total_tokens"] for row in setup),
                  "calls_or_checks": setup, "shared_preflight_tokens_recounted": False},
        "benchmarks": {
            "terminal-bench-2.0": audit_terminal(root, config),
            "gaia2-mini": audit_gaia(root),
            "vqav2": audit_vqa(root),
        },
        "evidence_storage": (
            json.loads(storage_states[-1].read_text()) if storage_states else None
        ),
        "headline_improvement_claim": None,
    }


def write_comparison_csv(report: dict, path: Path):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "benchmark", "method", "status", "native_mean_score", "mean_execution_tokens",
            "total_execution_tokens", "observations",
        ))
        writer.writeheader()
        for benchmark, entry in report["benchmarks"].items():
            final = entry["final_comparison"]
            for method in ("codex", "chisel_codex"):
                values = final["methods"][method] if final else {}
                writer.writerow({
                    "benchmark": benchmark, "method": method, "status": entry["status"],
                    **{key: values.get(key) for key in (
                        "native_mean_score", "mean_execution_tokens",
                        "total_execution_tokens", "observations",
                    )},
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Collect and audit remote benchmark results on A800-4")
    config = json.loads(args.config.read_text())
    report = collect(Path(config["root"]), config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / "COMPARISON.json", report)
    write_comparison_csv(report, args.output / "COMPARISON.csv")
    print(json.dumps({
        "report": str(args.output / "COMPARISON.json"),
        "states": {key: value["status"] for key, value in report["benchmarks"].items()},
    }))


if __name__ == "__main__":
    main()
