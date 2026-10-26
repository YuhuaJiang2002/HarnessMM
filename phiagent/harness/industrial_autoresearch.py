"""TRAIN-only AutoResearch decisions and exact-once verifier continuation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from phiagent.harness import industrial_autosota as rubric
from phiagent.harness.industrial_capacity import validate_workers
from phiagent.harness.industrial_coding_confirmation import TOKEN_KEYS, text_sha256

SCHEMA = "industrial-autoresearch/v1"
REFERENCE = {
    "repository": "https://github.com/karpathy/autoresearch",
    "revision": "228791fb499afffb54b46200aca536f79142f117",
    "documentation": "program.md",
    "official_autoresearch_experiment_executed": False,
}


def validate_config(config: Mapping[str, Any]) -> None:
    fixed = {
        "schema": SCHEMA,
        "method_identity": "AutoResearch-style quality-guarded native instruction search",
        "reference": REFERENCE,
        "phase": "train",
        "model": "deepseek-flash",
        "benchmark": "terminal-bench-2.0",
        "split_counts": {"train": 18, "validation": 18, "test": 53},
        "split_seed": 20260928,
        "execution_seeds": [20260928],
        "test_seeds": list(range(20260928, 20260933)),
        "context_compaction_threshold": 996147,
        "cumulative_token_limit": None,
        "token_usage_mode": "unlimited",
        "validation_feedback_to_optimizer": False,
        "test_feedback_to_optimizer": False,
        "cot_jev_core_claim": False,
    }
    if any(key not in config or config[key] != value for key, value in fixed.items()):
        raise ValueError("AutoResearch identity, complete population or native limits changed")
    validate_workers(config.get("maximum_workers"))
    if (
        config["recovery"]["actor_replay_permitted"] is not False
        or config["recovery"]["preserve_actor_timeout"] is not True
        or config["recovery"]["additional_actor_calls_for_verifier_recovery"] != 0
        or config["train_keep_rule"]["minimum_parent_relative_token_reduction"] != 0.30
        or config["train_keep_rule"]["target_parent_relative_token_reduction"] != 0.40
        or config["train_keep_rule"]["percentage_target_is_not_an_early_stop"] is not True
        or config["loop"]["incomplete_results_cannot_change_incumbent"] is not True
        or config["publication_gate"]["complete_native_test_population_per_arm"] != 265
        or config["publication_gate"]["training_result_cannot_replace_figure3_test_cell"] is not True
        or config["publication_gate"]["requires_official_quality_better_than_every_compared_arm"]
        is not True
    ):
        raise ValueError("Preserve recovery, quality-first selection and held-out publication gates")


def resume_plan(
    report: Mapping[str, Any], recovery: Mapping[str, Any], task_ids: Sequence[str],
) -> dict[str, Any]:
    if (
        report.get("status") != "BLOCKED" or report.get("active")
        or len(task_ids) != 18 or len(set(task_ids)) != 18
        or report.get("coverage", {}).get("expected") != 18
    ):
        raise ValueError("Resume only a drained complete-population TRAIN controller")
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Retained native attempts are required")
    identities, tasks = set(), set()
    blocked = []
    for row in attempts:
        task, identity = row["task_id"], row["attempt_id"]
        usage = row["cost"].get("usage")
        if (
            task not in task_ids or task in tasks or identity in identities
            or row.get("phase") != "train" or row.get("track") != "codex"
            or row["cost"].get("usage_status") != "MEASURED"
            or not isinstance(usage, Mapping)
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in TOKEN_KEYS)
            or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
            or usage["total_tokens"] <= 0
        ):
            raise ValueError("Retain unique, TRAIN-only, completely metered native attempts")
        tasks.add(task)
        identities.add(identity)
        if row["status"] != "SCORED":
            blocked.append(row)
        elif type(row.get("native_score")) not in (int, float) or row["native_score"] not in (0, 1):
            raise ValueError("A scored retained task needs its binary native reward")
    if len(blocked) != 1:
        raise ValueError("This continuation repairs exactly one identified unscored native attempt")
    original = blocked[0]
    if (
        original["status"] != "BLOCKED" or original["native_score"] is not None
        or original["error"] != "Official verifier did not supply its unique native reward"
        or recovery.get("status") != "NATIVE_VERIFIER_RECOVERED"
        or recovery.get("task_id") != original["task_id"]
        or recovery.get("phase") != "train"
        or recovery.get("callback") != original["attempt_id"]
        or type(recovery.get("additional_actor_calls")) is not int
        or recovery["additional_actor_calls"] != 0
        or "cumulative_token_limit" not in recovery or recovery["cumulative_token_limit"] is not None
        or recovery.get("original_actor_usage") != original["cost"]
        or recovery.get("original_native_result_sha256") != original["native_result_sha256"]
        or type(recovery.get("score")) not in (int, float) or recovery["score"] not in (0, 1)
        or recovery.get("original_actor_exception") != original.get("native_exception")
    ):
        raise ValueError("Recovery must bind the same paid actor, original exception and actual native reward")
    corrected = {
        **deepcopy(original), "status": "SCORED", "native_score": recovery["score"],
        "native_result": recovery["native_result"],
        "native_result_sha256": recovery["native_result_sha256"],
        "error": None, "historical_outcome_reused": True,
        "verifier_only_recovery": True,
    }
    retained = [
        corrected if row["task_id"] == original["task_id"] else deepcopy(row)
        for row in attempts
    ]
    return {
        "phase": "train", "retained_attempts": retained,
        "recovered_task": original["task_id"],
        "unstarted_tasks": [task for task in task_ids if task not in tasks],
        "retained_actor_usage": {
            key: sum(row["cost"]["usage"][key] for row in attempts) for key in TOKEN_KEYS
        },
        "additional_actor_calls_for_recovered_verdict": 0,
        "original_actor_timeout_preserved": True,
        "validation_or_test_used": False,
        "cumulative_token_limit": None,
    }


def decision(
    parent: Mapping[str, Any], incumbent: Mapping[str, Any], candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result = rubric.evaluate_rubric(parent, incumbent, candidate)
    keep = result["quality_preserved"] and result["incumbent_token_reduction_fraction"] > 0
    return {
        **result, "status": "keep" if keep else "discard",
        "incumbent_changed": keep, "publication_benefit_claim_admitted": False,
        "rule": "Complete TRAIN quality cannot regress; tokens must strictly decrease",
    }


def validate_instruction(candidate: Mapping[str, Any]) -> None:
    instruction = candidate.get("instruction")
    if (
        not isinstance(instruction, str) or not instruction.strip()
        or candidate.get("id") != text_sha256(instruction)
        or candidate.get("origin") != "optimizer_callback_train_only"
        or candidate.get("training_tasks") != 18
        or candidate.get("paired_training_observations") != 18
    ):
        raise ValueError("A retained instruction requires its unchanged TRAIN-derived identity")
