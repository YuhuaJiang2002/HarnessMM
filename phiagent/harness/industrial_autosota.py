"""TRAIN-only AutoSOTA-workflow adaptation for native industrial token reduction."""

from __future__ import annotations

import math
from typing import Any, Mapping

from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition
from phiagent.harness.industrial_capacity import validate_workers
from phiagent.harness.industrial_coding_confirmation import TOKEN_KEYS, text_sha256

SCHEMA = "industrial-autosota-token-search/v1"
METHOD = "chisel_codex_autosota_compaction"
REFERENCE = {
    "repository": "https://github.com/tsinghua-fib-lab/AutoSOTA",
    "revision": "d39ac9e03cf999589be196f06a26cc8449bb3031",
    "documentation": "cli_guide.md",
    "identity": "AutoSOTA-workflow-adapted search; not an official AutoSOTA CLI execution",
}
ARMS = {
    "compact_32k": {"method": METHOD, "context_compaction_threshold": 32768},
    "compact_16k": {"method": METHOD, "context_compaction_threshold": 16384},
    "compact_96k": {"method": METHOD, "context_compaction_threshold": 98304},
}
SEARCH_TREE = {
    "rounds": 2,
    "first_arm": "compact_32k",
    "after_complete_train": {
        "quality_preserved_and_incumbent_tokens_lower": "compact_16k",
        "otherwise": "compact_96k",
    },
    "percentage_target_is_not_an_early_stop": True,
}
RUBRIC = {
    "guardrails": {
        "complete_matched_train_tasks": 18,
        "minimum_native_successes": "max(native_parent, TRAIN-derived incumbent)",
        "model": "deepseek-flash",
        "shared_prompt_and_native_scorer_unchanged": True,
        "full_input_output_compaction_retry_and_failure_usage": True,
        "validation_and_test_feedback_to_search": False,
    },
    "objective": {
        "direction": "minimize",
        "metric": "complete_task_actor_input_plus_output_tokens",
        "minimum_reduction_fraction_against_parent": 0.30,
        "target_reduction_fraction_against_parent": 0.40,
        "strictly_fewer_tokens_than_train_incumbent": True,
    },
    "development_accounting": "Inherited acquisition and all search attempts, charged once",
}


def validate_contract(contract: Mapping[str, Any]) -> None:
    fixed = {
        "schema": SCHEMA,
        "phase": "train",
        "model": "deepseek-flash",
        "method_identity": REFERENCE["identity"],
        "reference": REFERENCE,
        "arms": ARMS,
        "search_tree": SEARCH_TREE,
        "rubric": RUBRIC,
        "execution_seeds": [20260928],
        "cumulative_token_limit": None,
        "token_usage_mode": "unlimited",
        "validation_access": False,
        "test_access": False,
        "cot_jev_core_claim": False,
        "official_autosota_cli_executed": False,
    }
    if any(key not in contract or canonical_sha256(contract[key]) != canonical_sha256(value)
           for key, value in fixed.items()):
        raise ValueError("AutoSOTA search objective, provenance or TRAIN boundary changed")
    validate_partition(contract["partition"])
    if (
        contract["partition"]["benchmark"] != "terminal-bench-2.0"
        or contract["partition"]["counts"] != {"train": 18, "validation": 18, "test": 53}
        or contract["partition"]["independent_unit"] != "task"
    ):
        raise ValueError("Require the complete native18/18/53 split")
    validate_workers(contract.get("maximum_workers"))
    for key in ("partition_sha256", "shared_prompt_sha256", "applied_residual_sha256"):
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
    if arm not in ARMS or phase != "train":
        raise ValueError("The optimization search cannot execute validation or test")
    expected = ARMS[arm]
    if (
        method != expected["method"]
        or type(threshold) is not int or threshold != expected["context_compaction_threshold"]
        or type(seed) is not int or seed not in contract["execution_seeds"]
        or task_id not in contract["partition"]["splits"]["train"]
        or partition_sha256 != contract["partition_sha256"]
        or shared_prompt_sha256 != contract["shared_prompt_sha256"]
        or not isinstance(residual, str) or not residual.strip()
        or text_sha256(residual) != contract["applied_residual_sha256"]
    ):
        raise ValueError("Native AutoSOTA trial differs from its frozen TRAIN contract")


def validate_summary(summary: Mapping[str, Any]) -> None:
    scores, usage = summary.get("task_scores"), summary.get("actor_usage")
    if (
        summary.get("complete_native_scores_and_cost") is not True
        or any(type(summary.get(key)) is not int or summary[key] != count
               for key, count in (("independent_tasks", 18), ("observations", 18), ("repetitions", 1)))
        or not isinstance(scores, Mapping) or len(scores) != 18
        or any(not isinstance(task, str) or not task for task in scores)
        or any(type(score) not in (int, float) or not math.isfinite(score)
               or score not in (0, 1) for score in scores.values())
        or not isinstance(usage, Mapping)
        or any(type(usage.get(key)) is not int or usage[key] < 0 for key in TOKEN_KEYS)
        or usage["total_tokens"] <= 0
        or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
        or type(summary.get("successes")) not in (int, float)
        or summary["successes"] != sum(scores.values())
        or summary.get("mean_native_score") != sum(scores.values()) / 18
    ):
        raise ValueError("Require all18 native verdicts and complete, reconciled execution tokens")


def evaluate_rubric(
    parent: Mapping[str, Any], incumbent: Mapping[str, Any], candidate: Mapping[str, Any],
) -> dict[str, Any]:
    for summary in (parent, incumbent, candidate):
        validate_summary(summary)
    if (set(parent["task_scores"]) != set(incumbent["task_scores"])
            or set(parent["task_scores"]) != set(candidate["task_scores"])):
        raise ValueError("Every search arm must retain the identical independent TRAIN population")
    if parent["successes"] < 9:
        raise ValueError("A token search requires the complete capable native parent")
    parent_tokens = parent["actor_usage"]["total_tokens"]
    incumbent_tokens = incumbent["actor_usage"]["total_tokens"]
    candidate_tokens = candidate["actor_usage"]["total_tokens"]
    quality_preserved = candidate["successes"] >= max(parent["successes"], incumbent["successes"])
    incumbent_saving = 1 - candidate_tokens / incumbent_tokens
    parent_saving = 1 - candidate_tokens / parent_tokens
    eligible = (
        quality_preserved and incumbent_saving > 0
        and 10 * candidate_tokens <= 7 * parent_tokens
    )
    return {
        "phase": "train", "candidate_eligible_for_protected_validation": eligible,
        "quality_preserved": quality_preserved,
        "required_native_successes": max(parent["successes"], incumbent["successes"]),
        "candidate_native_successes": candidate["successes"],
        "parent_token_reduction_fraction": parent_saving,
        "incumbent_token_reduction_fraction": incumbent_saving,
        "target_40_percent_met": 10 * candidate_tokens <= 6 * parent_tokens,
        "complete_task_tokens": candidate_tokens,
        "publication_benefit_claim_admitted": False,
        "validation_or_test_used": False,
    }


def reflect_next_arm(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    if (
        evaluation.get("phase") != "train"
        or evaluation.get("validation_or_test_used") is not False
        or type(evaluation.get("quality_preserved")) is not bool
        or type(evaluation.get("incumbent_token_reduction_fraction")) not in (int, float)
        or not math.isfinite(evaluation["incumbent_token_reduction_fraction"])
    ):
        raise ValueError("Reflection requires a complete TRAIN-only rubric evaluation")
    promising = (
        evaluation["quality_preserved"] and evaluation["incumbent_token_reduction_fraction"] > 0
    )
    branch = "quality_preserved_and_incumbent_tokens_lower" if promising else "otherwise"
    return {
        "next_arm": SEARCH_TREE["after_complete_train"][branch],
        "train_evidence_only": True, "new_optimizer_model_calls": 0,
        "reason": (
            "Native quality survived and tokens fell; test a shorter active-history trigger"
            if promising else
            "Aggressive compaction missed quality or cost; test a longer trigger"
        ),
        "percentage_target_does_not_skip_round_two": True,
    }
