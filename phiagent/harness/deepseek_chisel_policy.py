"""Train-only residual fitting for the official DeepSeek Harness adapter.

The public-text features and cross-fitted trainer are shared with the existing
CHISEL controller. Native execution, split access and provider accounting belong
to the benchmark runner; this module does not execute a model or a benchmark.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math

from phiagent.harness.codex_chisel import _feature_schema, public_text_features
from phiagent.evaluation.edgebench_native import (
    compare_native_outcomes, validate_native_outcome,
)
from phiagent.learning.residual_meta_harness import canonical_hash
from phiagent.learning.residual_meta_training import fit_cross_fitted_linear_head


SPLIT_SEED = 20260928


def validate_proposal(
    instruction: str, supporting_task_ids: Sequence[str], training_task_ids: Sequence[str],
) -> dict:
    if not isinstance(instruction, str):
        raise ValueError("The residual must be a string")
    if (
        isinstance(supporting_task_ids, str)
        or not isinstance(supporting_task_ids, Sequence)
        or any(not isinstance(value, str) for value in supporting_task_ids)
        or len(set(supporting_task_ids)) != len(supporting_task_ids)
        or not set(supporting_task_ids) <= set(training_task_ids)
    ):
        raise ValueError("Residual support must contain distinct TRAIN task IDs only")
    instruction = instruction.strip()
    if (
        len(instruction) > 1200 or len(instruction.split()) > 180 or "\x00" in instruction
        or bool(instruction) != bool(supporting_task_ids)
    ):
        raise ValueError("A sparse residual requires training support; empty text must abstain")
    return {
        "id": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "instruction": instruction,
        "supporting_task_ids": sorted(supporting_task_ids),
        "origin": "deepseek-harness-train-only-optimizer",
    }


def measured_pair(row: Mapping) -> tuple[float | None, int]:
    if "native_outcome" in row:
        value = row["native_outcome"]
        checked = validate_native_outcome(
            value, task_id=value["task_id"], score_direction=value["score_direction"],
            selection_policy=value["selection_policy"],
            normalization_declared=value["normalization_declared"],
            infrastructure_ok=value["infrastructure_ok"],
        )
        tokens = row.get("total_tokens")
        if checked["availability"] != "observed" or (
            type(tokens) is not int or tokens < 0 or row.get("usage_complete") is not True
        ):
            raise ValueError("Complete observed native outcome and measured actor tokens are required")
        return checked["score"], tokens
    score, tokens = row.get("native_score"), row.get("total_tokens")
    if (
        type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
        or type(tokens) is not int or tokens < 0 or row.get("usage_complete") is not True
    ):
        raise ValueError("Complete native [0,1] score and measured actor tokens are required")
    return float(score), tokens


def fit_candidate(
    proposal: Mapping,
    tasks: Sequence[Mapping],
    baseline: Mapping[str, Mapping],
    treatment: Mapping[str, Mapping],
    *,
    native_preferences: bool = False,
) -> dict:
    ids = [task["task_id"] for task in tasks]
    if (
        not ids or len(set(ids)) != len(ids) or set(baseline) != set(ids)
        or set(treatment) != set(ids)
    ):
        raise ValueError("Candidate fitting requires the complete paired TRAIN task population")
    checked = validate_proposal(proposal["instruction"], proposal["supporting_task_ids"], ids)
    if not checked["instruction"]:
        raise ValueError("An abstention has no residual head to fit")
    features = {task["task_id"]: public_text_features(task["public_prompt"]) for task in tasks}
    groups: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        if not isinstance(task["unit_id"], str) or not task["unit_id"]:
            raise ValueError("Every task requires an independent unit")
        groups[task["unit_id"]].append(task["task_id"])
    if len(groups) < 2:
        raise ValueError("Cross-fitting requires at least two independent TRAIN units")
    schema = _feature_schema()
    examples = []
    for unit, members in sorted(groups.items()):
        pairs = [(measured_pair(baseline[key]), measured_pair(treatment[key])) for key in members]
        if native_preferences:
            comparisons = [
                compare_native_outcomes(
                    baseline[key]["native_outcome"], treatment[key]["native_outcome"],
                    purpose="train_routing",
                ) for key in members
            ]
            if not all(value["resolved"] for value in comparisons):
                raise ValueError("Unresolved native task ordering cannot supply routing supervision")
            differences = [{"win": 1.0, "tie": 0.0, "loss": -1.0}[value["ordinal"]]
                           for value in comparisons]
        else:
            differences = [right[0] - left[0] for left, right in pairs]
        examples.append({
            "task_id": unit,
            "features": {
                name: math.fsum(features[key][name] for key in members) / len(members)
                for name in sorted(schema)
            },
            "utility": math.fsum(differences) / len(pairs),
            "harmful": any(value < 0 for value in differences),
            "token_delta": math.fsum(
                max(-1.0, min(1.0, (right[1] - left[1]) / max(1, left[1])))
                for left, right in pairs
            ) / len(pairs),
        })
    head = fit_cross_fitted_linear_head(
        examples, feature_schema=schema, folds=2, seed=SPLIT_SEED,
        ridge=1.0, minimum_override_support=2,
    )
    admitted = head["status"] == "DISCOVERY_CANDIDATE" and head["oof_harmful_rate"] == 0
    candidate = {
        **checked, "head": head, "enabled": admitted,
        "admission": "positive_harmless_training_oof" if admitted else "no_admissible_training_gain",
        "training_tasks": len(tasks), "training_units": len(groups),
        "paired_evidence_sha256": canonical_hash({"baseline": baseline, "treatment": treatment}),
        "routing_utility": (
            "Ordinal native task preference; internal TRAIN utility, not a benchmark aggregate"
            if native_preferences else "Paired normalized native-score difference"
        ),
        "feature_bounds": {
            name: [min(value[name] for value in features.values()),
                   max(value[name] for value in features.values())]
            for name in sorted(schema)
        },
    }
    candidate["candidate_sha256"] = canonical_hash(candidate)
    return candidate


def verify_candidate(candidate: Mapping) -> None:
    unsigned = dict(candidate)
    declared = unsigned.pop("candidate_sha256", None)
    if declared != canonical_hash(unsigned):
        raise ValueError("Frozen CHISEL candidate hash mismatch")


def seal_stage(document: Mapping, phase: str) -> dict:
    if phase not in {"validation", "test"}:
        raise ValueError("Only completed training or validation can release a later phase")
    result = dict(document)
    result.pop("release_sha256", None)
    result["release_phase"] = phase
    result["schema"] = "phiagent-deepseek-chisel-phase-release/v1"
    result["release_sha256"] = canonical_hash(result)
    return result


def verify_stage(document: Mapping, phase: str, partition_sha256: str, now: float) -> None:
    unsigned = dict(document)
    declared = unsigned.pop("release_sha256", None)
    if declared != canonical_hash(unsigned):
        raise ValueError("Phase-release hash mismatch")
    if (
        document.get("schema") != "phiagent-deepseek-chisel-phase-release/v1"
        or document.get("release_phase") != phase
        or document.get("partition_sha256") != partition_sha256
        or document.get("training_finalized") is not True
        or type(document.get("frozen_at_unix")) not in (int, float)
        or not math.isfinite(document["frozen_at_unix"])
        or not document["frozen_at_unix"] <= now
        or not isinstance(document.get("source_files"), dict) or not document["source_files"]
    ):
        raise ValueError("Missing or contradictory pre-execution phase/source freeze")
    for candidate in document.get("candidates", []):
        verify_candidate(candidate)
    if phase == "test":
        selection = document.get("selection")
        if not isinstance(selection, dict) or document.get("frozen_before_test") is not True:
            raise ValueError("Test requires a completed frozen validation selection")
        unsigned_selection = dict(selection)
        selection_hash = unsigned_selection.pop("selection_sha256", None)
        if selection_hash != canonical_hash(unsigned_selection):
            raise ValueError("Frozen validation selection hash mismatch")
        selected = selection["selected_candidate_id"]
        candidates = {candidate["id"]: candidate for candidate in document.get("candidates", [])}
        if selected != "baseline" and (
            selected not in candidates or not candidates[selected]["enabled"]
            or selection["selected_candidate_sha256"] != candidates[selected]["candidate_sha256"]
        ):
            raise ValueError("The selected test policy is not an admitted frozen candidate")


def route(candidate: Mapping | None, public_prompt: str) -> dict:
    if candidate is None:
        return {"instruction": "", "candidate_id": None, "reason": "baseline_or_abstention"}
    verify_candidate(candidate)
    if not candidate["enabled"]:
        return {"instruction": "", "candidate_id": None, "reason": "unadmitted_training_candidate"}
    features = public_text_features(public_prompt)
    if any(
        not bounds[0] <= features[name] <= bounds[1]
        for name, bounds in candidate["feature_bounds"].items()
    ):
        return {"instruction": "", "candidate_id": None, "reason": "outside_training_support"}
    head, schema = candidate["head"], _feature_schema()
    score = head["intercept"] + math.fsum(
        head["weights"][name] * schema[name].normalize(features[name]) for name in sorted(schema)
    )
    active = score >= head["score_threshold"]
    return {
        "instruction": candidate["instruction"] if active else "",
        "candidate_id": candidate["id"] if active else None,
        "reason": "cross_fitted_advantage" if active else "below_cross_fitted_threshold",
        "score": score, "threshold": head["score_threshold"],
    }


def select_validation(
    candidates: Sequence[Mapping],
    results: Mapping[str, Mapping[str, Mapping]],
    expected_task_ids: Sequence[str],
    *,
    native_preferences: bool = False,
) -> dict:
    expected = set(expected_task_ids)
    if not expected or len(expected) != len(expected_task_ids):
        raise ValueError("Validation requires a unique complete task population")
    eligible = {}
    for candidate in candidates:
        verify_candidate(candidate)
        if candidate["id"] in eligible:
            raise ValueError("Duplicate validation candidate")
        if candidate["enabled"]:
            eligible[candidate["id"]] = candidate
    if set(results) != {"baseline", *eligible}:
        raise ValueError("Validation must evaluate baseline and every admitted frozen candidate")
    if native_preferences:
        summaries = {}
        for ident, rows in results.items():
            if set(rows) != expected:
                raise ValueError("Incomplete or out-of-split validation population")
            for value in rows.values():
                measured_pair(value)
            comparisons = [
                compare_native_outcomes(
                    results["baseline"][key]["native_outcome"], rows[key]["native_outcome"],
                    purpose="validation_selection",
                ) for key in sorted(expected)
            ]
            if not all(value["resolved"] for value in comparisons):
                raise ValueError("Complete native task orderings are required for policy selection")
            summaries[ident] = {
                "per_task_selection_diagnostics": comparisons,
                "pareto_improves_baseline": (
                    all(value["ordinal"] != "loss" for value in comparisons)
                    and any(value["ordinal"] == "win" for value in comparisons)
                ),
                "total_tokens": sum(value["total_tokens"] for value in rows.values()),
            }
        contenders = [
            ident for ident in eligible if summaries[ident]["pareto_improves_baseline"]
        ]
        winner = min(contenders, key=lambda ident: (
            summaries[ident]["total_tokens"], ident,
        )) if contenders else "baseline"
        selection = {
            "selected_candidate_id": winner,
            "selected_candidate_sha256": (
                eligible[winner]["candidate_sha256"] if winner != "baseline" else None
            ),
            "rule": "No native task loss and at least one gain; then total tokens and hash",
            "summaries": summaries, "validation_results_sha256": canonical_hash(results),
            "validation_used_for_optimization": False, "test_opened": False,
            "quality_aggregate": None, "internal_selection_only": True,
        }
        selection["selection_sha256"] = canonical_hash(selection)
        return selection
    summaries = {}
    for ident, rows in results.items():
        if set(rows) != expected:
            raise ValueError("Incomplete or out-of-split validation population")
        values = [measured_pair(rows[key]) for key in sorted(expected)]
        summaries[ident] = {
            "mean_native_score": math.fsum(score for score, _ in values) / len(values),
            "total_tokens": sum(tokens for _, tokens in values),
        }
    contenders = [
        ident for ident in eligible
        if summaries[ident]["mean_native_score"] > summaries["baseline"]["mean_native_score"] + 1e-12
    ]
    winner = min(contenders, key=lambda ident: (
        -summaries[ident]["mean_native_score"], summaries[ident]["total_tokens"], ident,
    )) if contenders else "baseline"
    selection = {
        "selected_candidate_id": winner,
        "selected_candidate_sha256": (
            eligible[winner]["candidate_sha256"] if winner != "baseline" else None
        ),
        "rule": "complete paired native mean; strict gain over baseline; then tokens and hash",
        "summaries": summaries, "validation_results_sha256": canonical_hash(results),
        "validation_used_for_optimization": False, "test_opened": False,
    }
    selection["selection_sha256"] = canonical_hash(selection)
    return selection
