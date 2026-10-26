"""Training-only ADD candidate selection, with frozen input-only QA routes."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

from phiagent.learning.residual_meta_harness import (
    FeatureSpec,
    canonical_hash,
    deterministic_group_folds,
    feature_schema_hash,
)
from phiagent.learning.residual_meta_training import _fit_ridge, _predict, _validate_examples


SCHEMA = "phiagent-qa-add-candidates/v1"
CONTRACT = "## Answer Contract\n"
ADD_SOURCE = "## Meta-Harness ADD patch\n"
ADD_HEADER = "## Supplemental Evidence Check\n"
FAMILIES = {"dm": ("modify",), "da": ("add",), "dam": ("modify", "add")}
TRAIN_ARMS = ("delete", "modify", "add", "skillopt")


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "canonical_sha256": canonical_hash(value)}


def _verified(value: Mapping[str, Any], kind: str) -> None:
    unsigned = dict(value)
    if unsigned.pop("canonical_sha256", None) != canonical_hash(unsigned):
        raise ValueError(f"{kind} artifact hash differs")
    if value.get("schema_version") != SCHEMA or value.get("kind") != kind:
        raise ValueError(f"unexpected {kind} artifact identity")


def build_add_profiles(
    delete_prompt: str, modify_prompt: str, historical_add_prompt: str,
) -> dict[str, str]:
    """Insert the pinned historical ADD rule without importing its answer format."""
    if (delete_prompt.count(CONTRACT) != 1 or modify_prompt.count(CONTRACT) != 1
            or historical_add_prompt.count(ADD_SOURCE) != 1):
        raise ValueError("profiles require unique answer contracts and one historical ADD patch")
    prefix, contract = delete_prompt.split(CONTRACT)
    if not contract.strip() or modify_prompt.split(CONTRACT)[1] != contract:
        raise ValueError("DELETE and MODIFY answer contracts must be identical")
    patch = historical_add_prompt.split(ADD_SOURCE)[1].strip()
    if not patch or any(line.startswith("#") for line in patch.splitlines()):
        raise ValueError("ADD source must contain one nonempty rule block, not another section")
    if ADD_HEADER in delete_prompt:
        raise ValueError("DELETE already contains the supplemental ADD block")
    insertion = ADD_HEADER + patch + "\n\n"
    added = prefix + insertion + CONTRACT + contract
    if added.replace(insertion, "", 1) != delete_prompt:
        raise RuntimeError("ADD changed retained DELETE bytes")
    return {"delete": delete_prompt, "modify": modify_prompt, "add": added}


def _arm(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row["arms"][name]
    if (type(value["exact_match"]) is not int or value["exact_match"] not in (0, 1)
            or isinstance(value["f1"], bool) or not isinstance(value["f1"], (int, float))
            or not math.isfinite(value["f1"]) or not 0 <= value["f1"] <= 1
            or type(value["tokens"]) is not int or value["tokens"] <= 0):
        raise ValueError("arms require official binary EM, F1 in [0,1], and measured token usage")
    return value


def _matrix(matrix: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not matrix:
        raise ValueError("candidate training/calibration matrix is empty")
    seen = set()
    for row in matrix:
        if (not isinstance(row["id"], str) or not row["id"] or row["id"] in seen
                or not isinstance(row["group"], str) or not row["group"]):
            raise ValueError("matrix requires unique task IDs and nonempty question groups")
        seen.add(row["id"])
        for name in TRAIN_ARMS:
            _arm(row, name)
    return sorted(matrix, key=lambda row: row["id"])


def _winner(scores: Mapping[str, Mapping[str, float]], task: str, family: str) -> str:
    # The historical MODIFY branch wins an exact tie; runtime cost is not observed.
    return max(FAMILIES[family], key=lambda arm: scores[arm][task])


def _threshold(
    matrix: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, float]],
    family: str,
    *,
    minimum_support: int,
    maximum_harmful_rate: float,
    maximum_token_delta: float,
) -> dict[str, Any]:
    baseline_em = sum(_arm(row, "delete")["exact_match"] for row in matrix)
    baseline_f1 = math.fsum(_arm(row, "delete")["f1"] for row in matrix)
    baseline_tokens = sum(_arm(row, "delete")["tokens"] for row in matrix)
    successes, f1 = baseline_em, baseline_f1
    tokens = baseline_tokens
    harmful = sum(not _arm(row, "delete")["exact_match"]
                  and _arm(row, "skillopt")["exact_match"] for row in matrix)
    normalized_cost = math.fsum(
        max(-1.0, min(1.0, (_arm(row, "delete")["tokens"] - _arm(row, "skillopt")["tokens"])
                      / _arm(row, "skillopt")["tokens"]))
        for row in matrix
    )
    winners = {row["id"]: _winner(scores, row["id"], family) for row in matrix}
    ordered = sorted(matrix, key=lambda row: (-scores[winners[row["id"]]][row["id"]], row["id"]))
    counts: Counter[str] = Counter()
    selected_groups: dict[str, set[str]] = defaultdict(set)
    best = None
    cursor = 0
    while cursor < len(ordered):
        threshold = scores[winners[ordered[cursor]["id"]]][ordered[cursor]["id"]]
        if threshold <= 0:
            break
        end = cursor
        while end < len(ordered):
            row = ordered[end]
            arm = winners[row["id"]]
            if scores[arm][row["id"]] != threshold:
                break
            old, new, parent = (_arm(row, name) for name in ("delete", arm, "skillopt"))
            successes += new["exact_match"] - old["exact_match"]
            f1 += new["f1"] - old["f1"]
            tokens += new["tokens"] - old["tokens"]
            harmful += (int(not new["exact_match"] and parent["exact_match"])
                        - int(not old["exact_match"] and parent["exact_match"]))
            normalized_cost += (
                max(-1.0, min(1.0, (new["tokens"] - parent["tokens"]) / parent["tokens"]))
                - max(-1.0, min(1.0, (old["tokens"] - parent["tokens"]) / parent["tokens"]))
            )
            counts[arm] += 1
            selected_groups[arm].add(row["group"])
            end += 1
        improves_default = (
            successes > baseline_em or f1 > baseline_f1 + 1e-10
            or (abs(f1 - baseline_f1) <= 1e-10 and tokens < baseline_tokens)
        )
        if (improves_default
                and all(len(groups) >= minimum_support for groups in selected_groups.values())
                and successes >= baseline_em and f1 >= baseline_f1 - 1e-10
                and harmful / len(matrix) <= maximum_harmful_rate
                and normalized_cost / len(matrix) <= maximum_token_delta):
            candidate = {
                "status": "TRAINING_THRESHOLD_FROZEN",
                "score_threshold": threshold,
                "override_support": end,
                "arm_supports": dict(sorted(counts.items())),
                "arm_group_supports": {
                    arm: len(groups) for arm, groups in sorted(selected_groups.items())
                },
                "training_successes": successes,
                "training_f1_sum": f1,
                "training_total_tokens": tokens,
                "harmful_rate_vs_skillopt": harmful / len(matrix),
                "mean_normalized_token_delta_vs_skillopt": normalized_cost / len(matrix),
            }
            key = (-successes, -round(f1, 10), tokens, counts["add"], end, -threshold)
            if best is None or key < best[0]:
                best = (key, candidate)
        cursor = end
    if best is not None:
        return best[1]
    return {
        "status": "NO_ADMISSIBLE_THRESHOLD_ABSTAIN",
        "score_threshold": None,
        "override_support": 0,
        "arm_supports": {},
        "arm_group_supports": {},
        "reason": "No positive, tie-preserving OOF threshold passes every frozen training gate.",
        "abstention_is_not_a_quality_certificate": True,
    }


def fit_candidate_policies(
    matrix: Sequence[Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, FeatureSpec],
    folds: int = 5,
    seed: int = 20260920,
    ridge: float = 1.0,
    minimum_support: int = 400,
    maximum_harmful_rate: float = 0.05,
    maximum_token_delta: float = -0.08,
) -> dict[str, Any]:
    matrix = _matrix(matrix)
    if (type(folds) is not int or folds < 2 or type(seed) is not int
            or isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0
            or type(minimum_support) is not int
            or not 1 <= minimum_support <= len({row["group"] for row in matrix})
            or not math.isfinite(maximum_harmful_rate) or not 0 <= maximum_harmful_rate <= 1
            or not math.isfinite(maximum_token_delta) or not -1 <= maximum_token_delta <= 1):
        raise ValueError("invalid frozen candidate fitting or selection settings")
    if any(set(row["features"]) != set(feature_schema) for row in matrix):
        raise ValueError("training features differ from the frozen public feature allowlist")
    groups = {row["id"]: row["group"] for row in matrix}
    assignments = deterministic_group_folds(sorted(set(groups.values())), folds=folds, seed=seed)
    scores, heads = {}, {}
    for arm in ("modify", "add"):
        examples = [
            {
                "task_id": row["id"], "features": row["features"],
                "utility": _arm(row, arm)["exact_match"] - _arm(row, "delete")["exact_match"],
                "harmful": bool(_arm(row, "delete")["exact_match"]
                                and not _arm(row, arm)["exact_match"]),
                "token_delta": max(-1.0, min(
                    1.0, (_arm(row, arm)["tokens"] - _arm(row, "delete")["tokens"])
                    / _arm(row, "delete")["tokens"],
                )),
            }
            for row in matrix
        ]
        rows = _validate_examples(examples, feature_schema)
        oof = {}
        for fold in range(folds):
            training = [row for row in rows if assignments[groups[row[0]]] != fold]
            held = [row for row in rows if assignments[groups[row[0]]] == fold]
            intercept, weights = _fit_ridge(training, ridge=ridge)
            for task, vector, *_ in held:
                oof[task] = _predict(intercept, weights, vector)
        if set(oof) != set(groups) or any(not math.isfinite(value) for value in oof.values()):
            raise ValueError("incomplete or nonfinite out-of-fold predictions")
        intercept, weights = _fit_ridge(rows, ridge=ridge)
        scores[arm] = oof
        heads[arm] = {
            "intercept": intercept, "weights": weights,
            "oof_scores_sha256": canonical_hash(oof),
            "target": f"official_EM({arm}) - official_EM(delete)",
        }
    settings = {
        "minimum_support": minimum_support,
        "maximum_harmful_rate": maximum_harmful_rate,
        "maximum_token_delta": maximum_token_delta,
    }
    return _signed({
        "schema_version": SCHEMA, "kind": "fit",
        "status": "FROZEN_FROM_TRAINING_ONLY",
        "seed": seed, "folds": folds, "ridge": ridge,
        "feature_names": sorted(feature_schema),
        "feature_schema_sha256": feature_schema_hash(feature_schema),
        "training_matrix_sha256": canonical_hash(matrix),
        "training_tasks": len(matrix), "training_task_groups": groups,
        "training_group_folds": assignments,
        "training_fixed_totals": {
            arm: {
                "exact_matches": sum(_arm(row, arm)["exact_match"] for row in matrix),
                "f1_sum": math.fsum(_arm(row, arm)["f1"] for row in matrix),
                "total_tokens": sum(_arm(row, arm)["tokens"] for row in matrix),
            }
            for arm in TRAIN_ARMS
        },
        "heads": heads,
        "thresholds": {
            family: _threshold(matrix, scores, family, **settings) for family in FAMILIES
        },
        "threshold_settings": settings,
        "support_unit": "normalized whole-question group",
        "selection_rule": (
            "Positive OOF uplift; minimum whole-question-group support per active action; no loss of "
            "training EM/F1 versus DELETE and strictly improve its EM/F1/token lexicographic "
            "objective; harm and normalized token gates versus SkillOpt. Maximize EM, then F1, "
            "minimize total tokens, ADD count, and overrides. Preserve score ties."
        ),
        "primary_family": "dam", "no_add_control": "dm",
        "exact_score_tie": "modify", "router_model_calls": 0, "extra_actor_calls": 0,
        "same_modify_head_shared_by_dm_and_dam": True,
    })


def freeze_candidate_routes(
    fitted: Mapping[str, Any],
    public_items: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, FeatureSpec],
    seed: int,
) -> dict[str, Any]:
    _verified(fitted, "fit")
    if type(seed) is not int:
        raise ValueError("routing seed must be an integer")
    if (fitted["feature_names"] != sorted(feature_schema)
            or fitted["feature_schema_sha256"] != feature_schema_hash(feature_schema)):
        raise ValueError("routing feature definition differs from training")
    allowed = {"id", "group", "question", "context"}
    if not public_items or any(set(item) != allowed for item in public_items):
        raise ValueError("routing accepts only ID, group, question and context, never outcomes")
    if any(not all(isinstance(item[key], str) for key in allowed)
           or not item["id"] or not item["group"] for item in public_items):
        raise ValueError("routing public fields must be strings with nonempty IDs and groups")
    groups = {item["id"]: item["group"] for item in public_items}
    if len(groups) != len(public_items) or set(features) != set(groups):
        raise ValueError("routing task identities are duplicated or incomplete")
    if (set(groups) & fitted["training_task_groups"].keys()
            or set(groups.values()) & fitted["training_group_folds"].keys()):
        raise ValueError("training/routing task or question-group leakage")
    if any(set(value) != set(feature_schema) for value in features.values()):
        raise ValueError("routing features differ from the public feature allowlist")
    vectors = _validate_examples([
        {"task_id": task, "features": features[task], "utility": 0,
         "harmful": False, "token_delta": 0}
        for task in sorted(groups)
    ], feature_schema)
    scores = {
        arm: {task: _predict(head["intercept"], head["weights"], vector)
              for task, vector, *_ in vectors}
        for arm, head in fitted["heads"].items()
    }
    if any(not math.isfinite(value) for arm in scores.values() for value in arm.values()):
        raise ValueError("routing produced nonfinite scores")
    routes = {}
    for family in FAMILIES:
        threshold = fitted["thresholds"][family]["score_threshold"]
        routes[family] = {}
        for task in sorted(groups):
            arm = _winner(scores, task, family)
            routes[family][task] = (
                arm if threshold is not None and scores[arm][task] >= threshold else "delete"
            )
    strata: dict[int, list[str]] = defaultdict(list)
    for item in public_items:
        strata[len(item["context"]) // 512].append(item["id"])
    random_routes = {}
    for stratum, tasks in sorted(strata.items()):
        actions = sorted(routes["dam"][task] for task in tasks)
        ordered = sorted(tasks, key=lambda task: (
            hashlib.sha256(f"qa-add-random-v1\0{seed}\0{stratum}\0{task}".encode()).digest(), task,
        ))
        random_routes.update(zip(ordered, actions, strict=True))
    routes["matched_random_dam"] = random_routes
    routes["neutral_add_dam"] = {
        task: "neutral_add" if arm == "add" else arm for task, arm in routes["dam"].items()
    }
    return _signed({
        "schema_version": SCHEMA, "kind": "routes",
        "status": "FROZEN_WITHOUT_ROUTING_LABELS_OR_RESPONSES",
        "fit_sha256": fitted["canonical_sha256"], "seed": seed,
        "public_items_sha256": canonical_hash(sorted(public_items, key=lambda item: item["id"])),
        "features_sha256": canonical_hash(features),
        "tasks": len(groups), "task_groups": groups,
        "routes": routes,
        "action_counts": {name: dict(sorted(Counter(route.values()).items()))
                          for name, route in routes.items()},
        "random_control": "Exact action counts within outcome-blind 512-character context bins.",
        "random_realized_output_cost_matched": False,
        "aliases": [name for name, route in routes.items()
                    if name != "dam" and route == routes["dam"]],
    })


def select_calibrated_candidate(
    matrix: Sequence[Mapping[str, Any]],
    frozen_routes: Mapping[str, Any],
    *,
    minimum_add_support: int = 100,
) -> dict[str, Any]:
    """Accept/reject DAM without changing either trained policy or any threshold."""
    _verified(frozen_routes, "routes")
    matrix = _matrix(matrix)
    if type(minimum_add_support) is not int or minimum_add_support < 1:
        raise ValueError("calibration ADD support must be a positive integer")
    groups = {row["id"]: row["group"] for row in matrix}
    if groups != frozen_routes["task_groups"]:
        raise ValueError("calibration matrix differs from the pre-frozen routing population")
    totals = {}
    for family in ("dm", "dam"):
        chosen = [_arm(row, frozen_routes["routes"][family][row["id"]]) for row in matrix]
        totals[family] = {
            "tasks": len(chosen),
            "exact_matches": sum(arm["exact_match"] for arm in chosen),
            "f1_sum": math.fsum(arm["f1"] for arm in chosen),
            "total_tokens": sum(arm["tokens"] for arm in chosen),
        }
    baseline, candidate = totals["dm"], totals["dam"]
    add_count = sum(arm == "add" for arm in frozen_routes["routes"]["dam"].values())
    add_groups = {
        groups[task] for task, arm in frozen_routes["routes"]["dam"].items() if arm == "add"
    }
    checks = {
        "minimum_add_support": len(add_groups) >= minimum_add_support,
        "strict_em_improvement": candidate["exact_matches"] > baseline["exact_matches"],
        "strict_f1_improvement": candidate["f1_sum"] > baseline["f1_sum"] + 1e-10,
        "no_token_increase": candidate["total_tokens"] <= baseline["total_tokens"],
    }
    passed = all(checks.values())
    return _signed({
        "schema_version": SCHEMA, "kind": "calibration",
        "status": "CALIBRATION_SCREEN_PASSED" if passed else "CALIBRATION_REJECTED_KEEP_DM",
        "fit_sha256": frozen_routes["fit_sha256"],
        "routes_sha256": frozen_routes["canonical_sha256"],
        "matrix_sha256": canonical_hash(matrix),
        "selected_family": "dam" if passed else "dm",
        "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
        "totals": totals, "add_support": add_count, "add_group_support": len(add_groups),
        "minimum_add_support": minimum_add_support,
        "support_unit": "normalized whole-question group",
        "thresholds_retuned": False, "statistical_superiority_certified": False,
        "existing_published_release_replaced": False,
    })
