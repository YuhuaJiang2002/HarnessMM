"""Optional CPU training and dependency-free inference for bounded residual trees."""

from __future__ import annotations

import importlib.metadata
import math
import struct
from typing import Mapping, Sequence

from phiagent.learning.qa_postdraft_recheck import DraftObservation, features, validate_outcome


SCHEMA = "phiagent-qa-postdraft-tree-head/v1"


def fit_tree_head(
    observations: Sequence[DraftObservation], targets: Mapping[str, dict], *,
    mode: str, draft: str, reference: str, f1_weight: float, max_depth: int,
    min_samples_leaf: int, seed: int, expected_version: str,
) -> dict:
    if (not observations or len({row.id for row in observations}) != len(observations)
            or set(targets) != {row.id for row in observations}
            or type(max_depth) is not int or not 1 <= max_depth <= 3
            or type(min_samples_leaf) is not int or min_samples_leaf < 2
            or type(seed) is not int or type(f1_weight) not in (int, float)
            or not math.isfinite(f1_weight) or not 0 <= f1_weight <= 1):
        raise ValueError("complete fit data and prospectively bounded tree settings are required")
    try:
        version = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Install the pinned optional QA tree requirements in a private CPU runtime") from error
    if version != expected_version:
        raise RuntimeError("The optional tree-training package differs from the frozen version")
    from sklearn.tree import DecisionTreeRegressor

    names = sorted(features(observations[0], mode=mode))
    x, y = [], []
    for observation in observations:
        target = targets[observation.id]
        if target["group"] != observation.group:
            raise ValueError("Tree-fit task and group identities differ")
        first, second = target["methods"][draft], target["methods"][reference]
        validate_outcome(first)
        validate_outcome(second)
        values = features(observation, mode=mode)
        x.append([values[name] for name in names])
        y.append(second["exact_match"] - first["exact_match"] + f1_weight * (second["f1"] - first["f1"]))
    fitted = DecisionTreeRegressor(
        criterion="squared_error", max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, random_state=seed,
    ).fit(x, y)
    tree = fitted.tree_
    head = {
        "schema_version": SCHEMA, "mode": mode, "feature_names": names,
        "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "seed": seed,
        "f1_gain_weight": f1_weight, "training_package": f"scikit-learn=={version}",
        "fit_tasks": len(observations), "fit_groups": len({row.group for row in observations}),
        "input_comparison_dtype": "float32",
        "children_left": tree.children_left.tolist(), "children_right": tree.children_right.tolist(),
        "split_feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
        "ids_groups_or_target_fields_used_as_features": False,
    }
    compiled = [predict_tree(observation, head) for observation in observations]
    if compiled != fitted.predict(x).tolist():
        raise RuntimeError("Dependency-free compiled predictions differ from the pinned training backend")
    head["training_compiled_predictions_match"] = True
    return head


def predict_tree(observation: DraftObservation, head: dict) -> float:
    values = features(observation, mode=head["mode"])
    names = head["feature_names"]
    count = len(head["value"])
    if (head["schema_version"] != SCHEMA or head["input_comparison_dtype"] != "float32"
            or names != sorted(values) or not 1 <= count <= 15
            or any(len(head[key]) != count for key in (
                "children_left", "children_right", "split_feature", "threshold",
            )) or any(type(value) not in (int, float) or not math.isfinite(value)
                      for value in [*head["threshold"], *head["value"]])):
        raise ValueError("The compiled tree must retain its exact public feature and numeric schema")
    # scikit-learn converts prediction inputs to float32 before comparing its stored thresholds.
    vector = [struct.unpack("<f", struct.pack("<f", values[name]))[0] for name in names]
    node = 0
    for _ in range(count):
        left, right, index = head["children_left"][node], head["children_right"][node], head["split_feature"][node]
        if any(type(value) is not int for value in (left, right, index)):
            raise ValueError("Compiled tree node references must be exact integers")
        if left == right == -1:
            return head["value"][node]
        if not (0 <= left < count and 0 <= right < count and 0 <= index < len(vector)):
            raise ValueError("Invalid compiled tree child or feature")
        node = left if vector[index] <= head["threshold"][node] else right
    raise ValueError("Compiled tree contains a cycle instead of a bounded leaf path")
