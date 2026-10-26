"""Frozen coding confirmation contracts and complete-population evidence gates."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition
from phiagent.harness.industrial_capacity import validate_workers
from phiagent.harness.figure3_repetitions import test_seeds

SCHEMA = "industrial-coding-confirmation/v1"
ARMS = {
    "native": {"method": "codex", "context_compaction_threshold": 996147, "instruction": False},
    "native_compaction": {
        "method": "codex_confirmation_compaction",
        "context_compaction_threshold": 65536, "instruction": False,
    },
    "instruction": {
        "method": "chisel_codex", "context_compaction_threshold": 996147, "instruction": True,
    },
    "instruction_compaction": {
        "method": "chisel_codex_confirmation_compaction",
        "context_compaction_threshold": 65536, "instruction": True,
    },
}
CANDIDATE_ARMS = ("instruction", "instruction_compaction")
CONTROL_ARMS = ("native", "native_compaction")
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instruction_suffix(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty frozen TRAIN-derived instruction is required")
    return "[CHISEL task-local residual]\n" + instruction


def validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("model") != "deepseek-flash"
        or canonical_sha256(contract.get("arms")) != canonical_sha256(ARMS)
        or "cumulative_token_limit" not in contract
        or contract.get("cumulative_token_limit") is not None
        or contract.get("token_usage_mode") != "unlimited"
        or contract.get("test_selection_from_outcomes") is not False
        or contract.get("optimization_from_validation") is not False
        or contract.get("method_identity") != "TRAIN-derived industrial instruction/context study"
        or contract.get("cot_jev_core_claim") is not False
    ):
        raise ValueError("Confirmation identity, arms or information boundary changed")
    validate_partition(contract["partition"])
    if (
        contract["partition"]["benchmark"] != "terminal-bench-2.0"
        or contract["partition"]["counts"] != {"train": 18, "validation": 18, "test": 53}
        or contract["partition"]["independent_unit"] != "task"
        or contract["validation_seeds"] != [20260928]
        or contract["test_seeds"] != test_seeds(contract, range(20260928, 20260933))
    ):
        raise ValueError("Preserve the complete18/18/53 partition and existing repetition policy")
    validate_workers(contract.get("maximum_workers"))
    for key in ("partition_sha256", "shared_prompt_sha256", "instruction_sha256",
                "applied_residual_sha256"):
        value = contract.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"A frozen SHA256 is required: {key}")


def validate_trial_binding(
    contract: Mapping[str, Any], *, arm: str, method: str, threshold: int,
    residual: str, phase: str, task_id: str, seed: int, partition_sha256: str,
    shared_prompt_sha256: str,
) -> None:
    validate_contract(contract)
    if arm not in ARMS or phase not in ("validation", "test"):
        raise ValueError("Only a declared held-out confirmation arm may execute")
    expected = ARMS[arm]
    if (
        method != expected["method"]
        or type(threshold) is not int or threshold != expected["context_compaction_threshold"]
        or type(seed) is not int or seed not in contract[phase + "_seeds"]
        or task_id not in contract["partition"]["splits"][phase]
        or partition_sha256 != contract["partition_sha256"]
        or shared_prompt_sha256 != contract["shared_prompt_sha256"]
    ):
        raise ValueError("Native trial differs from the frozen confirmation contract")
    checksum = contract["applied_residual_sha256"] if expected["instruction"] else text_sha256("")
    if text_sha256(residual) != checksum:
        raise ValueError("Native residual differs from the exact frozen TRAIN instruction")


def verify_measured_row(row: Mapping[str, Any]) -> None:
    usage = row.get("usage")
    if (
        row.get("status") != "SCORED"
        or type(row.get("score")) not in (int, float) or row["score"] not in (0, 1)
        or not isinstance(usage, Mapping)
        or any(type(usage.get(key)) is not int or usage[key] < 0 for key in TOKEN_KEYS)
        or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
        or usage["total_tokens"] <= 0
        or row.get("executed_native_tests_verified") is not True
    ):
        raise ValueError("Only genuine executed native scores and complete provider usage are valid")


def development_accounting(parent: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    totals = {"optimizer": dict.fromkeys(TOKEN_KEYS, 0),
              "candidate_train_execution": dict.fromkeys(TOKEN_KEYS, 0)}
    counts = dict.fromkeys(totals, 0)
    identities = set()
    inherited = parent["accounting"]["tokens"]
    for key in TOKEN_KEYS:
        value = inherited[key]["complete_total"]
        if type(value) is not int or value < 0:
            raise ValueError("Shared corrected TRAIN acquisition cost must be complete")
    for entry in report["ledger"]:
        if entry["request_id"] in identities:
            raise ValueError("A development callback was charged more than once")
        identities.add(entry["request_id"])
        if entry["phase"] != "train":
            raise ValueError("Held-out callbacks cannot enter optimizer/development accounting")
        if entry["kind"] == "executor" and entry["arm"] == "baseline":
            continue
        role = "optimizer" if entry["kind"] == "optimizer" else "candidate_train_execution"
        if entry["kind"] not in ("optimizer", "executor"):
            raise ValueError("Unknown paid development role")
        usage = report["observations"][entry["request_id"]]["usage"]
        if (any(type(usage.get(key)) is not int or usage[key] < 0 for key in TOKEN_KEYS)
                or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]):
            raise ValueError("Unknown or inconsistent development usage cannot be zero-imputed")
        for key in TOKEN_KEYS:
            totals[role][key] += usage[key]
        counts[role] += 1
    parent_usage = {key: inherited[key]["complete_total"] for key in TOKEN_KEYS}
    return {
        "shared_parent_acquisition": {
            "usage": parent_usage, "actual_actor_attempts": len(parent["all_actor_attempts"]),
            "invalid_original_attempts_retained": True, "charged_once": True,
        },
        "optimizer": {"usage": totals["optimizer"], "callbacks": counts["optimizer"]},
        "candidate_train_execution": {
            "usage": totals["candidate_train_execution"],
            "callbacks": counts["candidate_train_execution"],
        },
        "all_development_usage": {
            key: parent_usage[key] + sum(role[key] for role in totals.values()) for key in TOKEN_KEYS
        },
        "cache_reads_included_in_input_tokens": True, "cost_usd": None,
        "validation_and_test_execution_excluded": True,
        "baseline_reference_rows_not_charged_again": True,
    }


def summarize_arm(
    rows: Sequence[Mapping[str, Any]], task_ids: Sequence[str], seeds: Sequence[int], arm: str,
) -> dict[str, Any]:
    expected = {(task, seed) for task in task_ids for seed in seeds}
    if not expected or len(set(task_ids)) != len(task_ids) or len(set(seeds)) != len(seeds):
        raise ValueError("A complete independent task population and unique seeds are required")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        pair = row["task_id"], row["seed"]
        if row["arm"] != arm or pair not in expected or pair in indexed:
            raise ValueError("Duplicate, mismatched or out-of-population native outcome")
        verify_measured_row(row)
        indexed[pair] = row
    if set(indexed) != expected:
        raise ValueError("Incomplete native task/repetition population")
    by_task = {
        task: math.fsum(indexed[(task, seed)]["score"] for seed in seeds) / len(seeds)
        for task in task_ids
    }
    return {
        "arm": arm, "observations": len(rows), "independent_tasks": len(task_ids),
        "repetitions": len(seeds), "successes": sum(row["score"] for row in rows),
        "mean_native_score": math.fsum(by_task.values()) / len(task_ids),
        "actor_usage": {key: sum(row["usage"][key] for row in rows) for key in TOKEN_KEYS},
        "task_scores": by_task, "complete_native_scores_and_cost": True,
    }


def validation_selection(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(summaries) != set(ARMS):
        raise ValueError("All frozen controls and mechanism arms are required before selection")
    for arm, summary in summaries.items():
        if (summary.get("arm") != arm or summary.get("complete_native_scores_and_cost") is not True
                or summary.get("independent_tasks") != 18 or summary.get("repetitions") != 1):
            raise ValueError("Selection requires all four complete validation18 populations")
    reference_quality = max(summaries[arm]["mean_native_score"] for arm in CONTROL_ARMS)
    reference_tokens = min(summaries[arm]["actor_usage"]["total_tokens"] for arm in CONTROL_ARMS)
    eligible = [
        arm for arm in CANDIDATE_ARMS
        if summaries[arm]["mean_native_score"] > reference_quality
        and 0 < summaries[arm]["actor_usage"]["total_tokens"] < reference_tokens
    ]
    selected = min(eligible, key=lambda arm: (
        -summaries[arm]["mean_native_score"],
        summaries[arm]["actor_usage"]["total_tokens"], arm,
    )) if eligible else None
    return {
        "status": "FROZEN_FOR_TEST" if selected else "REJECTED_COMPLETE_VALIDATION",
        "selected_arm": selected, "eligible_arms": eligible,
        "selection_rule": "Strictly higher native quality and lower tokens than both native controls; "
                          "then highest quality, lowest tokens and lexical arm tie-break",
        "training_instruction_modified": False, "validation_feedback_to_optimizer": False,
        "test_outcomes_used": False,
    }


def paired_comparisons(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], task_ids: Sequence[str],
    seeds: Sequence[int], selected: str,
) -> dict[str, Any]:
    from scripts.collect_codex_chisel import paired_summary

    if selected not in CANDIDATE_ARMS or set(rows_by_arm) != set(ARMS):
        raise ValueError("An unchanged validation-selected candidate and all four TEST arms are required")
    expected = {(task, seed) for task in task_ids for seed in seeds}
    summaries = {
        arm: summarize_arm(rows, task_ids, seeds, arm) for arm, rows in rows_by_arm.items()
    }

    def paired_rows(arm: str) -> list[dict[str, Any]]:
        return [
            {"task_id": row["task_id"], "unit_id": row["task_id"], "seed": row["seed"],
             "score": row["score"], "total_tokens": row["usage"]["total_tokens"]}
            for row in rows_by_arm[arm]
        ]

    contrasts = {
        arm: paired_summary(paired_rows(arm), paired_rows(selected), expected)
        for arm in ARMS if arm != selected
    }
    strongest_quality = max(
        summaries[arm]["mean_native_score"] for arm in ARMS if arm != selected
    )
    lowest_tokens = min(
        summaries[arm]["actor_usage"]["total_tokens"] for arm in ARMS if arm != selected
    )
    positive = (
        summaries[selected]["mean_native_score"] > strongest_quality
        and summaries[selected]["actor_usage"]["total_tokens"] < lowest_tokens
    )
    return {
        "schema": SCHEMA, "phase": "test", "selected_arm": selected,
        "summaries": summaries, "paired_task_cluster_bootstrap": contrasts,
        "positive_complete_comparison": positive,
        "official_success_strictly_better_than_all_compared_arms": (
            summaries[selected]["mean_native_score"] > strongest_quality
        ),
        "all_native_failures_and_compaction_calls_in_cost": True,
        "optimizer_and_training_costs_separate": True,
        "scope": "Full custom53-task TEST, five native repetitions; industrial instruction/context "
                 "descendant, not an isolated CoT/Jev mechanism result",
        "cot_jev_core_claim": False,
        "result_sha256": canonical_sha256({"summaries": summaries, "contrasts": contrasts,
                                          "selected": selected}),
    }
