from __future__ import annotations

import pytest

from phiagent.learning.chisel import (
    CalibrationCandidate,
    select_domain_policy,
)


def test_domain_shift_gate_selects_success_first_then_tokens() -> None:
    decision = select_domain_policy(
        [
            CalibrationCandidate("retrieval", 810, 2_799_103, 0.02),
            CalibrationCandidate("linear", 811, 2_965_320, 0.02),
            CalibrationCandidate("global-delete", 812, 2_794_711, 0.02),
        ],
        external_successes={
            "no-skill": 768,
            "skillopt": 784,
            "autosaddler": 754,
        },
        reference_policy_id="skillopt",
        reference_tokens=3_373_837,
        minimum_token_saving=0.05,
        maximum_harmful_rate=0.05,
    )

    assert decision.selected_policy_id == "global-delete"
    assert decision.passed is True
    assert decision.token_saving_vs_reference > 0.17


def test_domain_shift_gate_fails_closed_on_external_tie() -> None:
    decision = select_domain_policy(
        [CalibrationCandidate("candidate", 10, 80, 0.0)],
        external_successes={"skillopt": 10},
        reference_policy_id="skillopt",
        reference_tokens=100,
        minimum_token_saving=0.05,
        maximum_harmful_rate=0.05,
    )

    assert decision.passed is False
    assert "strictly_best_external" in decision.reason


def test_domain_shift_gate_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="invalid"):
        select_domain_policy(
            [
                CalibrationCandidate("same", 1, 1, 0.0),
                CalibrationCandidate("same", 2, 2, 0.0),
            ],
            external_successes={"skillopt": 1},
            reference_policy_id="skillopt",
            reference_tokens=2,
            minimum_token_saving=0.0,
            maximum_harmful_rate=1.0,
        )
