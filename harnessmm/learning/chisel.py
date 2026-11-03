"""CHISEL calibration-time policy selection and domain-shift abstention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


METHOD_NAME = "CHISEL"
METHOD_EXPANSION = "Certified Harness Interventions via Sparse Editing and Local routing"


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    policy_id: str
    successes: int
    total_tokens: int
    harmful_rate: float

    def __post_init__(self) -> None:
        if (
            not self.policy_id
            or isinstance(self.successes, bool)
            or self.successes < 0
            or isinstance(self.total_tokens, bool)
            or self.total_tokens < 0
            or not math.isfinite(self.harmful_rate)
            or not 0.0 <= self.harmful_rate <= 1.0
        ):
            raise ValueError("invalid CHISEL calibration candidate")


@dataclass(frozen=True, slots=True)
class DomainShiftDecision:
    selected_policy_id: str
    passed: bool
    reason: str
    token_saving_vs_reference: float
    candidate_ids: tuple[str, ...]
    external_baseline_ids: tuple[str, ...]


def select_domain_policy(
    candidates: Sequence[CalibrationCandidate],
    *,
    external_successes: Mapping[str, int],
    reference_policy_id: str,
    reference_tokens: int,
    minimum_token_saving: float,
    maximum_harmful_rate: float,
) -> DomainShiftDecision:
    """Select one predeclared candidate from calibration evidence only."""

    if (
        not candidates
        or len({candidate.policy_id for candidate in candidates}) != len(candidates)
        or not external_successes
        or reference_policy_id not in external_successes
        or isinstance(reference_tokens, bool)
        or reference_tokens <= 0
        or not math.isfinite(minimum_token_saving)
        or not 0.0 <= minimum_token_saving <= 1.0
        or not math.isfinite(maximum_harmful_rate)
        or not 0.0 <= maximum_harmful_rate <= 1.0
        or any(
            not policy_id
            or isinstance(successes, bool)
            or not isinstance(successes, int)
            or successes < 0
            for policy_id, successes in external_successes.items()
        )
    ):
        raise ValueError("invalid CHISEL domain-shift selection inputs")

    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate.successes,
            candidate.total_tokens,
            candidate.policy_id,
        ),
    )
    token_saving = 1.0 - selected.total_tokens / reference_tokens
    external_maximum = max(external_successes.values())
    checks = {
        "strictly_best_external": selected.successes > external_maximum,
        "token_saving": token_saving >= minimum_token_saving,
        "harmful_rate": selected.harmful_rate <= maximum_harmful_rate,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return DomainShiftDecision(
        selected_policy_id=selected.policy_id,
        passed=not failed,
        reason=(
            "calibration gates passed"
            if not failed
            else "calibration gates failed: " + ", ".join(failed)
        ),
        token_saving_vs_reference=token_saving,
        candidate_ids=tuple(candidate.policy_id for candidate in candidates),
        external_baseline_ids=tuple(sorted(external_successes)),
    )


__all__ = [
    "METHOD_EXPANSION",
    "METHOD_NAME",
    "CalibrationCandidate",
    "DomainShiftDecision",
    "select_domain_policy",
]
