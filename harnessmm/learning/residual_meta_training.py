"""Task-grouped cross-fitting for low-capacity residual Meta-Harness heads."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Mapping, Sequence

from .residual_meta_harness import (
    ALLOWED_FEATURE_NAMES,
    MAX_FEATURES,
    FeatureSpec,
    PolicyValidationError,
    canonical_hash,
    deterministic_group_folds,
    feature_schema_hash,
)


TRAINING_SCHEMA_VERSION = "phiagent-residual-meta-harness-training/v1"


def _validate_examples(
    examples: Sequence[Mapping[str, Any]],
    feature_schema: Mapping[str, FeatureSpec],
) -> list[tuple[str, list[float], float, bool, float]]:
    if not examples:
        raise ValueError("residual training examples are empty")
    if not feature_schema or len(feature_schema) > MAX_FEATURES:
        raise ValueError(
            f"feature schema must contain 1..{MAX_FEATURES} features"
        )
    unknown_features = set(feature_schema) - ALLOWED_FEATURE_NAMES
    if unknown_features:
        raise ValueError(
            "routing features are outside the frozen allowlist: "
            f"{sorted(unknown_features)}"
        )
    rows = []
    seen = set()
    names = tuple(sorted(feature_schema))
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping) or set(example) != {
            "task_id",
            "features",
            "utility",
            "harmful",
            "token_delta",
        }:
            raise ValueError(f"training example {index} fields differ")
        task_id = example["task_id"]
        features = example["features"]
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen
            or not isinstance(features, Mapping)
        ):
            raise ValueError("training examples require unique task IDs and features")
        seen.add(task_id)
        missing = set(feature_schema) - set(features)
        if missing:
            raise ValueError(f"training example lacks features: {sorted(missing)}")
        try:
            vector = [
                feature_schema[name].normalize(features[name]) for name in names
            ]
        except PolicyValidationError as error:
            raise ValueError(f"training feature contract failed: {error}") from error
        utility = example["utility"]
        token_delta = example["token_delta"]
        if (
            isinstance(utility, bool)
            or not isinstance(utility, (int, float))
            or not math.isfinite(float(utility))
            or not -1 <= float(utility) <= 1
            or isinstance(token_delta, bool)
            or not isinstance(token_delta, (int, float))
            or not math.isfinite(float(token_delta))
            or not -1 <= float(token_delta) <= 1
            or type(example["harmful"]) is not bool
        ):
            raise ValueError("training outcomes must be finite task-level values")
        rows.append(
            (
                task_id,
                vector,
                float(utility),
                bool(example["harmful"]),
                float(token_delta),
            )
        )
    return rows


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    size = len(target)
    augmented = [
        [*map(float, matrix[row]), float(target[row])] for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge system is numerically singular")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row],
                        augmented[column],
                        strict=True,
                    )
                ]
    return [augmented[row][-1] for row in range(size)]


def _fit_ridge(
    rows: Sequence[tuple[str, list[float], float, bool, float]],
    *,
    ridge: float,
) -> tuple[float, list[float]]:
    if not rows:
        raise ValueError("ridge fit has no task rows")
    dimension = len(rows[0][1]) + 1
    matrix = [[0.0] * dimension for _ in range(dimension)]
    target = [0.0] * dimension
    for _, features, utility, _, _ in rows:
        vector = [1.0, *features]
        for left in range(dimension):
            target[left] += vector[left] * utility
            for right in range(dimension):
                matrix[left][right] += vector[left] * vector[right]
    for index in range(1, dimension):
        matrix[index][index] += ridge
    coefficients = _solve(matrix, target)
    return coefficients[0], coefficients[1:]


def _predict(intercept: float, weights: Sequence[float], features: Sequence[float]) -> float:
    return intercept + math.fsum(
        weight * feature
        for weight, feature in zip(weights, features, strict=True)
    )


def fit_cross_fitted_linear_head(
    examples: Sequence[Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, FeatureSpec],
    folds: int = 5,
    seed: int = 20260908,
    ridge: float = 1.0,
    minimum_override_support: int,
) -> dict[str, Any]:
    """Fit a residual head and select its threshold from OOF predictions only."""

    if (
        not math.isfinite(ridge)
        or ridge <= 0
        or isinstance(minimum_override_support, bool)
        or not isinstance(minimum_override_support, int)
        or minimum_override_support < 1
    ):
        raise ValueError("ridge/support settings are invalid")
    rows = _validate_examples(examples, feature_schema)
    if minimum_override_support > len(rows):
        raise ValueError("minimum override support exceeds task count")
    assignments = deterministic_group_folds(
        [row[0] for row in rows],
        folds=folds,
        seed=seed,
    )
    out_of_fold = []
    for fold in range(folds):
        training = [row for row in rows if assignments[row[0]] != fold]
        held = [row for row in rows if assignments[row[0]] == fold]
        intercept, weights = _fit_ridge(training, ridge=ridge)
        for task_id, features, utility, harmful, token_delta in held:
            out_of_fold.append(
                {
                    "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                    "fold": fold,
                    "score": _predict(intercept, weights, features),
                    "utility": utility,
                    "harmful": harmful,
                    "token_delta": token_delta,
                }
            )
    if len(out_of_fold) != len(rows):
        raise RuntimeError("cross-fitting did not score every task exactly once")
    candidates = []
    for threshold in sorted(
        {float(row["score"]) for row in out_of_fold},
        reverse=True,
    ):
        selected = [
            row for row in out_of_fold if float(row["score"]) >= threshold
        ]
        if len(selected) < minimum_override_support:
            continue
        candidates.append(
            {
                "score_threshold": threshold,
                "override_support": len(selected),
                "mean_utility": statistics.fmean(
                    float(row["utility"]) for row in selected
                ),
                "harmful_rate": statistics.fmean(
                    bool(row["harmful"]) for row in selected
                ),
                "mean_token_delta": statistics.fmean(
                    float(row["token_delta"]) for row in selected
                ),
            }
        )
    if not candidates:
        raise RuntimeError("no out-of-fold threshold meets minimum support")
    selected_threshold = min(
        candidates,
        key=lambda row: (
            -float(row["mean_utility"]),
            float(row["harmful_rate"]),
            float(row["mean_token_delta"]),
            -float(row["score_threshold"]),
        ),
    )
    final_intercept, final_weights = _fit_ridge(rows, ridge=ridge)
    feature_names = tuple(sorted(feature_schema))
    status = (
        "DISCOVERY_CANDIDATE"
        if float(selected_threshold["mean_utility"]) > 0
        else "NO_POSITIVE_DISCOVERY_OVERRIDE"
    )
    result: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": status,
        "training_task_count": len(rows),
        "grouped_crossfit_folds": folds,
        "group_key": "task_id",
        "all_stages_and_seeds_grouped_by_task": True,
        "feature_names": list(feature_names),
        "feature_schema_sha256": feature_schema_hash(feature_schema),
        "ridge": ridge,
        "intercept": final_intercept,
        "weights": {
            name: final_weights[index]
            for index, name in enumerate(feature_names)
        },
        "score_threshold": float(selected_threshold["score_threshold"]),
        "oof_override_support": int(selected_threshold["override_support"]),
        "oof_mean_utility": float(selected_threshold["mean_utility"]),
        "oof_harmful_rate": float(selected_threshold["harmful_rate"]),
        "oof_mean_token_delta": float(
            selected_threshold["mean_token_delta"]
        ),
        "oof_predictions_sha256": canonical_hash(
            sorted(
                out_of_fold,
                key=lambda row: (str(row["task_sha256"]), int(row["fold"])),
            )
        ),
        "training_task_folds": {
            hashlib.sha256(task_id.encode()).hexdigest(): fold
            for task_id, fold in sorted(assignments.items())
        },
    }
    result["training_sha256"] = canonical_hash(result)
    return result


def select_calibration_records(
    training_result: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, FeatureSpec],
) -> list[dict[str, Any]]:
    """Apply a frozen discovery model to disjoint calibration task records."""

    if (
        training_result.get("schema_version") != TRAINING_SCHEMA_VERSION
        or training_result.get("status") != "DISCOVERY_CANDIDATE"
    ):
        raise ValueError("training result is not an eligible discovery candidate")
    unsigned = dict(training_result)
    declared = unsigned.pop("training_sha256", None)
    if declared != canonical_hash(unsigned):
        raise ValueError("training result self-hash mismatch")
    names = tuple(training_result["feature_names"])
    if (
        names != tuple(sorted(feature_schema))
        or training_result.get("feature_schema_sha256")
        != feature_schema_hash(feature_schema)
    ):
        raise ValueError("calibration feature schema differs from training")
    rows = _validate_examples(examples, feature_schema)
    training_hashes = set(training_result["training_task_folds"])
    calibration_hashes = {
        hashlib.sha256(task_id.encode()).hexdigest()
        for task_id, *_ in rows
    }
    if training_hashes & calibration_hashes:
        raise ValueError("calibration tasks overlap discovery tasks")
    intercept = float(training_result["intercept"])
    weights = [float(training_result["weights"][name]) for name in names]
    threshold = float(training_result["score_threshold"])
    selected = []
    for task_id, features, utility, harmful, token_delta in rows:
        if _predict(intercept, weights, features) >= threshold:
            selected.append(
                {
                    "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                    "utility": utility,
                    "harmful": harmful,
                    "token_delta": token_delta,
                }
            )
    return selected


__all__ = [
    "TRAINING_SCHEMA_VERSION",
    "fit_cross_fitted_linear_head",
    "select_calibration_records",
]
