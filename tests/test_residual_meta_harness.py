from __future__ import annotations

import copy
import hashlib
import math

import pytest

from phiagent.learning.residual_meta_harness import (
    POLICY_SCHEMA_VERSION,
    Operation,
    PolicySession,
    PolicyValidationError,
    build_evidence_document,
    calibration_bounds,
    canonical_hash,
    certificate_from_evidence,
    clopper_pearson_lower,
    clopper_pearson_upper,
    deterministic_group_folds,
    effective_sample_size,
    head_spec_hash,
    parse_policy,
    policy_to_dict,
)
from phiagent.learning.stage_aware_adm import (
    PolicySession as StagePolicySession,
    default_policy_value,
    parse_policy as parse_stage_policy,
)


def policy_value() -> dict:
    extractor_sha256 = "d" * 64
    feature_schema = {
        "recovery_signal": {"kind": "boolean"},
        "turn_fraction": {
            "kind": "numeric",
            "mean": 0.5,
            "scale": 0.25,
            "minimum": 0.0,
            "maximum": 1.0,
        },
    }
    spec_hash = head_spec_hash(
        head_id="recover-delete",
        stage="RECOVER",
        operation="DELETE",
        intercept=0.0,
        weights={
            "recovery_signal": 1.0,
            "turn_fraction": 0.1,
        },
        score_threshold=0.5,
        feature_schema=feature_schema,
        feature_extractor_sha256=extractor_sha256,
    )
    training = {
        hashlib.sha256(f"train-{index}".encode()).hexdigest(): index % 5 for index in range(200)
    }
    calibration = [
        {
            "task_sha256": hashlib.sha256(f"calibration-{index}".encode()).hexdigest(),
            "utility": 0.5,
            "harmful": False,
            "token_delta": -0.5,
        }
        for index in range(80)
    ]
    evidence = build_evidence_document(
        head_id="recover-delete",
        training_task_folds=training,
        calibration_records=calibration,
        source_matrix_sha256="b" * 64,
        head_spec_sha256=spec_hash,
        training_result_sha256="c" * 64,
    )
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": "residual-test-v1",
        "benchmark": "Terminal-Bench",
        "feature_extractor_sha256": extractor_sha256,
        "max_overrides_per_task": 1,
        "defaults_by_stage": {
            "INITIAL": "MODIFY",
            "EXECUTE": "MODIFY",
            "RECOVER": "MODIFY",
            "VERIFY": "ADD",
        },
        "feature_schema": feature_schema,
        "safety_gate": {
            "minimum_training_tasks": 100,
            "minimum_calibration_tasks": 60,
            "minimum_effective_sample_size": 50,
            "minimum_override_support": 50,
            "maximum_harmful_rate_upper": 0.1,
            "minimum_utility_lcb": 0.01,
            "maximum_token_delta_upper": 0.0,
            "maximum_extra_actor_calls": 0,
            "maximum_router_model_calls": 0,
        },
        "override_heads": [
            {
                "head_id": "recover-delete",
                "stage": "RECOVER",
                "operation": "DELETE",
                "intercept": 0.0,
                "weights": {
                    "recovery_signal": 1.0,
                    "turn_fraction": 0.1,
                },
                "score_threshold": 0.5,
                "certificate": certificate_from_evidence(evidence),
            }
        ],
        "evidence_documents": {"recover-delete": evidence},
    }


def test_policy_round_trip_and_self_hash() -> None:
    policy = parse_policy(policy_value())
    encoded = policy_to_dict(policy)

    assert encoded["policy_sha256"] == canonical_hash(
        {key: value for key, value in encoded.items() if key != "policy_sha256"}
    )
    assert parse_policy(encoded) == policy


def test_residual_override_requires_threshold_and_respects_budget() -> None:
    session = PolicySession(
        parse_policy(policy_value()),
        task_id="task-1",
        benchmark="Terminal-Bench",
    )

    initial = session.choose(
        stage="INITIAL",
        features={"recovery_signal": False, "turn_fraction": 0.0},
    )
    recovery = session.choose(
        stage="RECOVER",
        features={"recovery_signal": True, "turn_fraction": 0.6},
    )
    second_recovery = session.choose(
        stage="RECOVER",
        features={"recovery_signal": True, "turn_fraction": 0.8},
    )

    assert initial.used_default
    assert initial.operation is Operation.MODIFY
    assert recovery.operation is Operation.DELETE
    assert recovery.selected_head_id == "recover-delete"
    assert recovery.trace["router_model_calls"] == 0
    assert recovery.trace["extra_actor_calls"] == 0
    assert second_recovery.used_default
    assert second_recovery.reason == ("atomic residual head was already evaluated for this task")


def test_missing_or_out_of_distribution_features_fail_to_default() -> None:
    policy = parse_policy(policy_value())

    missing = PolicySession(
        policy,
        task_id="missing",
        benchmark="Terminal-Bench",
    ).choose(stage="RECOVER", features={"recovery_signal": True})
    shifted = PolicySession(
        policy,
        task_id="shifted",
        benchmark="Terminal-Bench",
    ).choose(
        stage="RECOVER",
        features={"recovery_signal": True, "turn_fraction": 1.5},
    )

    assert missing.used_default and "missing calibrated features" in missing.reason
    assert shifted.used_default and "feature contract fallback" in shifted.reason


def test_atomic_head_is_not_reconsidered_after_stage_reentry() -> None:
    session = PolicySession(
        parse_policy(policy_value()),
        task_id="reentry",
        benchmark="Terminal-Bench",
    )

    first = session.choose(
        stage="RECOVER",
        features={"recovery_signal": False, "turn_fraction": 0.1},
    )
    session.choose(
        stage="EXECUTE",
        features={"recovery_signal": False, "turn_fraction": 0.2},
    )
    second = session.choose(
        stage="RECOVER",
        features={"recovery_signal": True, "turn_fraction": 0.9},
    )

    assert first.used_default
    assert second.used_default
    assert second.reason == "atomic residual head was already evaluated for this task"


def test_outcome_features_and_unsafe_certificates_are_rejected() -> None:
    policy = parse_policy(policy_value())
    session = PolicySession(
        policy,
        task_id="unsafe",
        benchmark="Terminal-Bench",
    )
    with pytest.raises(PolicyValidationError, match="forbidden runtime field"):
        session.choose(
            stage="RECOVER",
            features={
                "recovery_signal": True,
                "turn_fraction": 0.5,
                "reward": 1,
            },
        )

    unsafe = copy.deepcopy(policy_value())
    unsafe["override_heads"][0]["certificate"]["router_model_calls"] = 1
    with pytest.raises(PolicyValidationError, match="raw evidence"):
        parse_policy(unsafe)

    impossible = copy.deepcopy(policy_value())
    impossible["override_heads"][0]["certificate"]["utility_lcb"] = 1.0
    with pytest.raises(PolicyValidationError, match="raw evidence"):
        parse_policy(impossible)

    rebound = copy.deepcopy(policy_value())
    rebound["override_heads"][0]["score_threshold"] = 0.4
    with pytest.raises(PolicyValidationError, match="bind the deployed head"):
        parse_policy(rebound)

    renormalized = copy.deepcopy(policy_value())
    renormalized["feature_schema"]["turn_fraction"]["scale"] = 0.01
    with pytest.raises(PolicyValidationError, match="bind the deployed head"):
        parse_policy(renormalized)


@pytest.mark.parametrize(
    "feature_name",
    ["instance_id", "task_identifier", "task_sha256"],
)
def test_policy_rejects_identifier_feature_aliases(feature_name: str) -> None:
    value = policy_value()
    value["feature_schema"][feature_name] = value["feature_schema"].pop("turn_fraction")
    value["override_heads"][0]["weights"][feature_name] = value["override_heads"][0]["weights"].pop(
        "turn_fraction"
    )

    with pytest.raises(PolicyValidationError, match="frozen runtime allowlist"):
        parse_policy(value)


def test_group_folds_are_balanced_and_task_disjoint() -> None:
    groups = [f"task-{index}" for index in range(17)]
    first = deterministic_group_folds(groups, folds=5, seed=7)
    second = deterministic_group_folds(list(reversed(groups)), folds=5, seed=7)
    counts = [list(first.values()).count(fold) for fold in range(5)]

    assert first == second
    assert max(counts) - min(counts) <= 1
    assert set(first) == set(groups)


def test_exact_harm_bound_and_effective_sample_size() -> None:
    upper = clopper_pearson_upper(0, 59)
    lower = clopper_pearson_lower(59, 59)

    assert upper == pytest.approx(1 - 0.05 ** (1 / 59), abs=1e-12)
    assert lower == pytest.approx(0.05 ** (1 / 59), abs=1e-12)
    assert upper < 0.05
    assert effective_sample_size([1.0] * 10) == pytest.approx(10.0)
    assert effective_sample_size([1.0, 3.0]) < 2.0


def test_calibration_bounds_use_task_level_vectors() -> None:
    bounds = calibration_bounds(
        [0.5] * 200,
        [False] * 200,
        [-0.5] * 200,
    )

    assert bounds["override_support"] == 200
    assert bounds["harmful_count"] == 0
    assert 0 < float(bounds["harmful_rate_upper"]) < 0.02
    assert float(bounds["utility_lcb"]) > 0
    assert float(bounds["token_delta_upper"]) < 0
    assert math.isfinite(float(bounds["utility_lcb"]))


def test_paired_exact_utility_bound_uses_wins_and_losses() -> None:
    utilities = [1.0] * 90 + [-1.0] * 60 + [0.0] * 1850
    bounds = calibration_bounds(
        utilities,
        [False] * len(utilities),
        [-0.15] * len(utilities),
        bound_method="paired_exact_bonferroni",
    )

    assert -0.02 < float(bounds["utility_lcb"]) < 0.0
    assert float(bounds["token_delta_upper"]) < 0.0


def test_paired_exact_utility_bound_rejects_continuous_utility() -> None:
    with pytest.raises(ValueError, match="utilities"):
        calibration_bounds(
            [0.5] * 200,
            [False] * 200,
            [-0.5] * 200,
            bound_method="paired_exact_bonferroni",
        )


def test_full_policy_evidence_separates_calibration_and_override_counts() -> None:
    feature_schema = {"recovery_signal": {"kind": "boolean"}}
    spec_hash = head_spec_hash(
        head_id="full-policy-add",
        stage="INITIAL",
        operation="ADD",
        intercept=0.0,
        weights={"recovery_signal": 1.0},
        score_threshold=0.5,
        feature_schema=feature_schema,
        feature_extractor_sha256="d" * 64,
    )
    evidence = build_evidence_document(
        head_id="full-policy-add",
        training_task_folds={
            hashlib.sha256(f"train-{index}".encode()).hexdigest(): index % 2 for index in range(10)
        },
        calibration_records=[
            {
                "task_sha256": hashlib.sha256(f"calibration-{index}".encode()).hexdigest(),
                "utility": 0,
                "harmful": False,
                "token_delta": -0.5,
                "overridden": index < 4,
            }
            for index in range(20)
        ],
        source_matrix_sha256="b" * 64,
        head_spec_sha256=spec_hash,
        training_result_sha256="c" * 64,
        bound_method="paired_exact_bonferroni",
        reference_policy_id="skillopt-v1",
    )

    certificate = certificate_from_evidence(evidence)

    assert certificate["calibration_task_count"] == 20
    assert certificate["effective_sample_size"] == 20
    assert certificate["override_support"] == 4

    policy = parse_policy(
        {
            "schema_version": POLICY_SCHEMA_VERSION,
            "policy_id": "full-policy-test-v1",
            "benchmark": "SearchQA",
            "feature_extractor_sha256": "d" * 64,
            "max_overrides_per_task": 1,
            "defaults_by_stage": {"INITIAL": "DELETE"},
            "feature_schema": feature_schema,
            "safety_gate": {
                "minimum_training_tasks": 10,
                "minimum_calibration_tasks": 20,
                "minimum_effective_sample_size": 20,
                "minimum_override_support": 4,
                "maximum_harmful_rate_upper": 1.0,
                "minimum_utility_lcb": -1.0,
                "maximum_token_delta_upper": 1.0,
                "maximum_extra_actor_calls": 0,
                "maximum_router_model_calls": 0,
            },
            "override_heads": [
                {
                    "head_id": "full-policy-add",
                    "stage": "INITIAL",
                    "operation": "ADD",
                    "intercept": 0.0,
                    "weights": {"recovery_signal": 1.0},
                    "score_threshold": 0.5,
                    "certificate": certificate,
                }
            ],
            "evidence_documents": {"full-policy-add": evidence},
        }
    )

    assert policy.override_heads[0].certificate.override_support == 4


def test_stage_aware_residual_mode_uses_local_proposal_without_controller() -> None:
    stage_policy = parse_stage_policy(default_policy_value("terminal", mode="residual"))
    session = StagePolicySession(
        stage_policy,
        task_id="task-1",
        task_context={"benchmark": "terminal", "family": "debugging"},
    )

    selected = session.choose(
        stage="INITIAL",
        proposed_operation="DELETE",
        features={"turn": 0},
    )
    fallback = session.choose(
        stage="RECOVER",
        proposed_operation=None,
        features={"turn": 1},
    )

    assert selected.operation.value == "DELETE"
    assert selected.source == "residual"
    assert fallback.operation.value == "MODIFY"
    assert fallback.source == "residual_fallback"
