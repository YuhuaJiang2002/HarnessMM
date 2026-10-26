"""Constrained GSPO-style optimization for finite harness edit sequences."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PolicyState:
    """Finite sequence-policy state and its capability Lagrange multiplier."""

    logits: tuple[tuple[str, float], ...]
    lagrange_multiplier: float
    update_index: int = 0

    def __post_init__(self) -> None:
        names = [name for name, _ in self.logits]
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("policy logits require unique sorted action IDs")
        if any(not math.isfinite(value) for _, value in self.logits):
            raise ValueError("policy logits must be finite")
        if not math.isfinite(self.lagrange_multiplier) or self.lagrange_multiplier < 0:
            raise ValueError("lagrange multiplier must be finite and non-negative")
        if self.update_index < 0:
            raise ValueError("update index must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return dict(self.logits)


@dataclass(frozen=True)
class CandidateOutcome:
    """One sequence-level capability and token outcome."""

    action_id: str
    successes: int
    tasks: int
    total_tokens: int
    reference_successes: int
    reference_tokens: int

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action ID must be non-empty")
        if (
            self.tasks <= 0
            or not 0 <= self.successes <= self.tasks
            or not 0 <= self.reference_successes <= self.tasks
        ):
            raise ValueError("success counts must lie inside the task count")
        if self.total_tokens <= 0 or self.reference_tokens <= 0:
            raise ValueError("token counts must be positive")

    @property
    def retention(self) -> float:
        if self.reference_successes == 0:
            return 1.0 if self.successes == 0 else math.inf
        return self.successes / self.reference_successes

    @property
    def token_saving_rate(self) -> float:
        return 1.0 - self.total_tokens / self.reference_tokens

    def violation(self, minimum_retention: float) -> float:
        return max(0.0, minimum_retention - self.retention)


def softmax(logits: Mapping[str, float]) -> dict[str, float]:
    """Return stable probabilities for a finite categorical policy."""

    if not logits:
        raise ValueError("logits must be non-empty")
    maximum = max(float(value) for value in logits.values())
    weights = {name: math.exp(float(value) - maximum) for name, value in logits.items()}
    denominator = sum(weights.values())
    return {name: value / denominator for name, value in weights.items()}


def normalized_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Compute group-relative advantages with population normalization."""

    if not rewards:
        raise ValueError("rewards must be non-empty")
    mean = sum(float(value) for value in rewards) / len(rewards)
    variance = sum((float(value) - mean) ** 2 for value in rewards) / len(rewards)
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return tuple(0.0 for _ in rewards)
    return tuple((float(value) - mean) / scale for value in rewards)


def constrained_sequence_reward(
    outcome: CandidateOutcome,
    *,
    minimum_retention: float,
    lagrange_multiplier: float,
) -> float:
    """Reward token savings and penalize capability-constraint violation."""

    return outcome.token_saving_rate - lagrange_multiplier * outcome.violation(minimum_retention)


def gspo_update(
    state: PolicyState,
    outcomes: Sequence[CandidateOutcome],
    *,
    minimum_retention: float,
    policy_epochs: int = 4,
    learning_rate: float = 0.25,
    clip_epsilon: float = 0.2,
    lagrange_learning_rate: float = 1.0,
) -> tuple[PolicyState, dict[str, object]]:
    """Apply clipped sequence-level group updates to a finite action policy.

    This is a tabular GSPO-style controller: each complete component-edit
    sequence is one categorical action, so probability ratios and clipping are
    defined at sequence level rather than per token.
    """

    if policy_epochs <= 0 or learning_rate <= 0:
        raise ValueError("policy epochs and learning rate must be positive")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must lie in (0, 1)")
    logits = state.as_dict()
    outcome_ids = [outcome.action_id for outcome in outcomes]
    if len(outcome_ids) != len(set(outcome_ids)) or set(outcome_ids) != set(logits):
        raise ValueError("outcomes must cover every policy action exactly once")
    behavior = softmax(logits)
    rewards = [
        constrained_sequence_reward(
            outcome,
            minimum_retention=minimum_retention,
            lagrange_multiplier=state.lagrange_multiplier,
        )
        for outcome in outcomes
    ]
    advantages = normalized_advantages(rewards)
    epoch_records = []
    for epoch in range(1, policy_epochs + 1):
        probabilities = softmax(logits)
        updates = {}
        for outcome, advantage in zip(outcomes, advantages, strict=True):
            ratio = probabilities[outcome.action_id] / behavior[outcome.action_id]
            clipped_ratio = min(
                1.0 + clip_epsilon,
                max(1.0 - clip_epsilon, ratio),
            )
            surrogate = min(
                ratio * advantage,
                clipped_ratio * advantage,
            )
            updates[outcome.action_id] = learning_rate * surrogate
        centered = sum(updates.values()) / len(updates)
        logits = {
            action_id: logits[action_id] + update - centered
            for action_id, update in updates.items()
        }
        epoch_records.append(
            {
                "epoch": epoch,
                "probabilities_before": probabilities,
                "updates": updates,
                "probabilities_after": softmax(logits),
            }
        )
    mean_violation = sum(outcome.violation(minimum_retention) for outcome in outcomes) / len(
        outcomes
    )
    lagrange = max(
        0.0,
        state.lagrange_multiplier + lagrange_learning_rate * mean_violation,
    )
    updated = PolicyState(
        logits=tuple(sorted(logits.items())),
        lagrange_multiplier=lagrange,
        update_index=state.update_index + 1,
    )
    audit = {
        "algorithm": "constrained_gspo_style_tabular_sequence_policy",
        "minimum_retention": minimum_retention,
        "behavior_probabilities": behavior,
        "rewards": dict(zip(outcome_ids, rewards, strict=True)),
        "advantages": dict(zip(outcome_ids, advantages, strict=True)),
        "mean_constraint_violation": mean_violation,
        "lagrange_before": state.lagrange_multiplier,
        "lagrange_after": lagrange,
        "policy_epochs": policy_epochs,
        "clip_epsilon": clip_epsilon,
        "learning_rate": learning_rate,
        "epoch_records": epoch_records,
        "final_probabilities": softmax(updated.as_dict()),
    }
    return updated, audit


def select_feasible_candidate(
    outcomes: Sequence[CandidateOutcome],
    *,
    minimum_retention: float,
) -> CandidateOutcome:
    """Select minimum-token feasible action, breaking ties by capability."""

    feasible = [outcome for outcome in outcomes if outcome.retention >= minimum_retention]
    if not feasible:
        raise ValueError("no candidate satisfies the retention constraint")
    return min(
        feasible,
        key=lambda outcome: (
            outcome.total_tokens,
            -outcome.successes,
            outcome.action_id,
        ),
    )


def select_masked_policy_candidate(
    outcomes: Sequence[CandidateOutcome],
    probabilities: Mapping[str, float],
    *,
    minimum_retention: float,
    minimum_token_saving: float = -math.inf,
) -> CandidateOutcome:
    """Select the highest-policy-probability action after hard constraint masking."""

    feasible = [
        outcome
        for outcome in outcomes
        if outcome.retention >= minimum_retention
        and outcome.token_saving_rate >= minimum_token_saving
    ]
    if not feasible:
        raise ValueError("no candidate satisfies the retention constraint")
    if set(probabilities) != {outcome.action_id for outcome in outcomes}:
        raise ValueError("probabilities must cover every candidate outcome")
    return max(
        feasible,
        key=lambda outcome: (
            float(probabilities[outcome.action_id]),
            outcome.token_saving_rate,
            outcome.successes,
            outcome.action_id,
        ),
    )


__all__ = [
    "CandidateOutcome",
    "PolicyState",
    "constrained_sequence_reward",
    "gspo_update",
    "normalized_advantages",
    "select_feasible_candidate",
    "select_masked_policy_candidate",
    "softmax",
]
