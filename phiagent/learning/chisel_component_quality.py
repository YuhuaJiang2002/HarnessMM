"""Outcome-blind component controls around the existing CHISEL ridge head."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

from phiagent.learning.residual_meta_harness import (
    FeatureSpec,
    canonical_hash,
    deterministic_group_folds,
)
from phiagent.learning.residual_meta_training import _fit_ridge, _predict, _validate_examples


def _arm(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row["arms"][name]
    if type(value["exact_match"]) is not int or value["exact_match"] not in (0, 1):
        raise ValueError("training requires official binary exact-match outcomes")
    if type(value["tokens"]) is not int or value["tokens"] <= 0:
        raise ValueError("training requires measured positive token usage")
    return value


def select_quality_threshold(
    matrix: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    *,
    minimum_support: int,
    maximum_harmful_rate: float,
    maximum_token_delta: float,
) -> dict[str, Any]:
    """Use the historical full-policy objective, without splitting score ties."""
    if (not matrix or len({row["id"] for row in matrix}) != len(matrix)
            or set(scores) != {row["id"] for row in matrix}):
        raise ValueError("threshold score and training task identities differ")
    if (type(minimum_support) is not int or not 1 <= minimum_support <= len(matrix)
            or not 0 <= maximum_harmful_rate <= 1
            or not math.isfinite(maximum_token_delta)
            or any(not math.isfinite(value) for value in scores.values())):
        raise ValueError("invalid threshold gates or nonfinite scores")
    changes = {}
    successes = harmful = 0
    token_delta = 0.0
    for row in matrix:
        default, edit, parent = (_arm(row, name) for name in ("delete", "modify", "skillopt"))
        parent_success = parent["exact_match"]
        left, right = default["exact_match"], edit["exact_match"]
        default_cost = max(-1.0, min(1.0, (default["tokens"] - parent["tokens"]) / parent["tokens"]))
        edit_cost = max(-1.0, min(1.0, (edit["tokens"] - parent["tokens"]) / parent["tokens"]))
        successes += left
        harmful += int(not left and parent_success)
        token_delta += default_cost
        changes[row["id"]] = (
            right - left,
            int(not right and parent_success) - int(not left and parent_success),
            edit_cost - default_cost,
        )
    ordered = sorted(scores, key=lambda task: (-scores[task], task))
    best = None
    cursor = 0
    count = len(matrix)
    while cursor < count:
        threshold = scores[ordered[cursor]]
        end = cursor
        while end < count and scores[ordered[end]] == threshold:
            delta, harm, cost = changes[ordered[end]]
            successes += delta
            harmful += harm
            token_delta += cost
            end += 1
        if (end >= minimum_support and harmful / count <= maximum_harmful_rate
                and token_delta / count <= maximum_token_delta):
            candidate = {
                "status": "THRESHOLD_SELECTED",
                "score_threshold": threshold,
                "override_support": end,
                "training_successes": successes,
                "harmful_rate": harmful / count,
                "mean_normalized_token_delta": token_delta / count,
            }
            key = (-successes, harmful, token_delta, -threshold)
            if best is None or key < best[0]:
                best = (key, candidate)
        cursor = end
    if best is None:
        return {
            "status": "NO_ADMISSIBLE_THRESHOLD_ABSTAIN",
            "score_threshold": None,
            "override_support": 0,
            "reason": "No score-tie-preserving threshold satisfies all frozen gates.",
        }
    return best[1]


def fit_quality_heads(
    matrix: Sequence[Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, FeatureSpec],
    folds: int,
    seed: int,
    ridge: float,
    minimum_support: int,
    maximum_harmful_rate: float,
    maximum_token_delta: float,
) -> dict[str, Any]:
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    matrix = sorted(matrix, key=lambda row: row["id"])
    groups = {}
    examples = []
    for row in matrix:
        if not isinstance(row.get("group"), str) or not row["group"]:
            raise ValueError("every training task requires a question group")
        groups[row["id"]] = row["group"]
        left, right = (_arm(row, name) for name in ("delete", "modify"))
        examples.append({
            "task_id": row["id"],
            "features": row["features"],
            "utility": right["exact_match"] - left["exact_match"],
            "harmful": bool(left["exact_match"] and not right["exact_match"]),
            "token_delta": max(-1.0, min(1.0, (right["tokens"] - left["tokens"]) / left["tokens"])),
        })
    rows = _validate_examples(examples, feature_schema)
    assignments = deterministic_group_folds(sorted(set(groups.values())), folds=folds, seed=seed)
    oof = {}
    for fold in range(folds):
        training = [row for row in rows if assignments[groups[row[0]]] != fold]
        held = [row for row in rows if assignments[groups[row[0]]] == fold]
        intercept, weights = _fit_ridge(training, ridge=ridge)
        for task, features, *_ in held:
            oof[task] = _predict(intercept, weights, features)
    intercept, weights = _fit_ridge(rows, ridge=ridge)
    in_sample = {task: _predict(intercept, weights, features) for task, features, *_ in rows}
    settings = {
        "minimum_support": minimum_support,
        "maximum_harmful_rate": maximum_harmful_rate,
        "maximum_token_delta": maximum_token_delta,
    }
    thresholds = {
        "oof": select_quality_threshold(matrix, oof, **settings),
        "in_sample": select_quality_threshold(matrix, in_sample, **settings),
    }
    result = {
        "schema_version": "phiagent-chisel-component-heads/v1",
        "seed": seed,
        "folds": folds,
        "ridge": ridge,
        "feature_names": sorted(feature_schema),
        "intercept": intercept,
        "weights": weights,
        "thresholds": thresholds,
        "training_tasks": len(rows),
        "training_groups": len(assignments),
        "training_group_folds": assignments,
        "training_task_groups": groups,
        "oof_scores_sha256": canonical_hash(oof),
        "in_sample_scores_sha256": canonical_hash(in_sample),
        "shared_final_head": True,
        "fits_computed_once_and_shared_between_controls": folds + 1,
        "threshold_settings": settings,
    }
    result["canonical_sha256"] = canonical_hash(result)
    return result


def matched_random_overrides(
    public_items: Sequence[Mapping[str, Any]],
    selected: set[str],
    *,
    seed: int,
) -> set[str]:
    by_id = {item["id"]: item for item in public_items}
    if len(by_id) != len(public_items) or not selected <= by_id.keys():
        raise ValueError("random control identities are duplicated or absent")
    strata: dict[int, list[str]] = defaultdict(list)
    for item in public_items:
        strata[len(item["context"]) // 512].append(item["id"])
    result = set()
    for stratum, tasks in sorted(strata.items()):
        count = sum(task in selected for task in tasks)
        ordered = sorted(tasks, key=lambda task: (
            hashlib.sha256(f"chisel-quality-random-v1\0{seed}\0{stratum}\0{task}".encode()).digest(),
            task,
        ))
        result.update(ordered[:count])
    return result


def freeze_quality_routes(
    head: Mapping[str, Any],
    public_items: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, FeatureSpec],
    seed: int,
) -> dict[str, Any]:
    unsigned = dict(head)
    if unsigned.pop("canonical_sha256", None) != canonical_hash(unsigned):
        raise ValueError("component head hash differs")
    if head["feature_names"] != sorted(feature_schema):
        raise ValueError("evaluation feature schema differs")
    task_ids = {item["id"] for item in public_items}
    if len(task_ids) != len(public_items) or not task_ids or set(features) != task_ids:
        raise ValueError("evaluation task/feature identities differ")
    evaluation_groups = {item["group"] for item in public_items}
    if evaluation_groups & head["training_group_folds"].keys():
        raise ValueError("training/evaluation question-group leakage")
    rows = _validate_examples([
        {"task_id": item["id"], "features": features[item["id"]],
         "utility": 0, "harmful": False, "token_delta": 0}
        for item in public_items
    ], feature_schema)
    scores = {
        task: _predict(head["intercept"], head["weights"], vector)
        for task, vector, *_ in rows
    }
    policies = {}
    diagnostics = {}
    for kind in ("oof", "in_sample"):
        threshold = head["thresholds"][kind]["score_threshold"]
        selected = set() if threshold is None else {
            task for task, score in scores.items() if score >= threshold
        }
        random = matched_random_overrides(public_items, selected, seed=seed)
        policies[f"chisel_{kind}"] = sorted(selected)
        policies[f"matched_random_{kind}"] = sorted(random)
        policies[f"residual_placebo_{kind}"] = sorted(selected)
        diagnostics[kind] = {
            "threshold_status": head["thresholds"][kind]["status"],
            "override_tasks": len(selected),
            "random_override_tasks": len(random),
            "input_length_stratum_counts_matched": True,
        }
    ranked = sorted(scores, key=lambda task: (-scores[task], task))
    equal_count_control = set(ranked[:len(policies["chisel_oof"])])
    if equal_count_control != set(policies["chisel_oof"]):
        raise RuntimeError("shared-head equal-count route identity failed")
    result = {
        "schema_version": "phiagent-chisel-component-routes/v1",
        "status": "FROZEN_WITHOUT_EVALUATION_ANSWERS",
        "head_sha256": head["canonical_sha256"],
        "public_items_sha256": canonical_hash(public_items),
        "seed": seed,
        "tasks": len(task_ids),
        "question_groups": len(evaluation_groups),
        "override_task_ids": policies,
        "diagnostics": diagnostics,
        "oof_in_sample_route_disagreements": len(
            set(policies["chisel_oof"]) ^ set(policies["chisel_in_sample"])
        ),
        "crossfit_equal_override_count_control": {
            "alias_of": "chisel_oof",
            "routes_identical": True,
            "independent_generated_arm": False,
            "reason": "Cross-fitting changes threshold selection, not the shared final head or ranking.",
        },
    }
    result["canonical_sha256"] = canonical_hash(result)
    return result
