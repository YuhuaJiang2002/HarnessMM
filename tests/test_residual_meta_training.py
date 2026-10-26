from __future__ import annotations

import pytest

from phiagent.learning.residual_meta_harness import FeatureKind, FeatureSpec
from phiagent.learning.residual_meta_training import (
    fit_cross_fitted_linear_head,
    select_calibration_records,
)


FEATURES = {
    "recovery_signal": FeatureSpec(kind=FeatureKind.BOOLEAN),
}


def examples(prefix: str, count: int) -> list[dict]:
    return [
        {
            "task_id": f"{prefix}-{index}",
            "features": {"recovery_signal": index % 2 == 0},
            "utility": 0.6 if index % 2 == 0 else -0.6,
            "harmful": index % 2 != 0,
            "token_delta": -0.2 if index % 2 == 0 else 0.2,
        }
        for index in range(count)
    ]


def test_cross_fitted_head_discovers_sparse_positive_residual() -> None:
    result = fit_cross_fitted_linear_head(
        examples("train", 200),
        feature_schema=FEATURES,
        minimum_override_support=80,
    )

    assert result["status"] == "DISCOVERY_CANDIDATE"
    assert result["training_task_count"] == 200
    assert result["oof_override_support"] == 100
    assert result["oof_mean_utility"] == 0.6
    assert result["oof_harmful_rate"] == 0.0
    assert len(result["training_task_folds"]) == 200


def test_frozen_head_selects_disjoint_calibration_tasks() -> None:
    result = fit_cross_fitted_linear_head(
        examples("train", 200),
        feature_schema=FEATURES,
        minimum_override_support=80,
    )
    selected = select_calibration_records(
        result,
        examples("calibration", 120),
        feature_schema=FEATURES,
    )

    assert len(selected) == 60
    assert all(row["utility"] == 0.6 for row in selected)
    assert all(row["harmful"] is False for row in selected)


@pytest.mark.parametrize(
    "feature_name",
    ["instance_id", "task_identifier", "task_sha256"],
)
def test_trainer_rejects_raw_task_id_features(feature_name: str) -> None:
    with pytest.raises(ValueError, match="frozen allowlist"):
        fit_cross_fitted_linear_head(
            [
                {
                    **row,
                    "features": {feature_name: index},
                }
                for index, row in enumerate(examples("train", 20))
            ],
            feature_schema={
                feature_name: FeatureSpec(
                    kind=FeatureKind.NUMERIC,
                    mean=10,
                    scale=5,
                    minimum=0,
                    maximum=20,
                )
            },
            minimum_override_support=5,
        )
