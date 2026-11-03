"""Benchmark-agnostic risk/coverage operators for skill evolution.

The module deliberately knows nothing about ALFWorld, LiveMathematicianBench,
or any other adapter.  It consumes only aligned unit identifiers, task-group
labels, binary outcomes, serialized traces, and a prompt budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TraceDatum:
    unit_id: str
    group_id: str
    success: bool
    text: str

    def __post_init__(self) -> None:
        if not self.unit_id or not self.group_id:
            raise ValueError("trace unit_id and group_id must be non-empty")
        if not self.text:
            raise ValueError("trace text must be non-empty")


@dataclass(frozen=True)
class RiskCoverageSelection:
    selected_unit_ids: tuple[str, ...]
    selected_failure_ids: tuple[str, ...]
    selected_success_ids: tuple[str, ...]
    group_weights: Mapping[str, float]
    group_failure_rates: Mapping[str, float]
    objective_value: float
    used_characters: int
    character_budget: int
    maximum_items: int
    facility_location_weight: float
    group_saturation_weight: float
    failure_weight: float
    entropic_step_size: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustAcceptanceDecision:
    accepted: bool
    empirical_gain: float
    robust_gain: float
    complexity_penalty: float
    penalized_robust_gain: float
    group_gains: Mapping[str, float]
    adversarial_group_weights: Mapping[str, float]
    kl_radius: float
    minimum_robust_gain: float
    require_strict_empirical_improvement: bool
    require_no_group_regression: bool
    gates: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedOutcomeAudit:
    total_units: int
    reference_successes: int
    candidate_successes: int
    beneficial_flip_ids: tuple[str, ...]
    harmful_flip_ids: tuple[str, ...]
    preserved_success_ids: tuple[str, ...]
    unchanged_failure_ids: tuple[str, ...]
    net_success_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtectedSuccessAcceptanceDecision:
    accepted: bool
    minimum_beneficial_flips: int
    maximum_harmful_flips: int
    require_strict_score_improvement: bool
    paired_outcomes: PairedOutcomeAudit
    gates: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_outcome_audit(
    *,
    unit_ids: Sequence[str],
    reference_outcomes: Sequence[bool | int],
    candidate_outcomes: Sequence[bool | int],
) -> PairedOutcomeAudit:
    """Classify paired outcome changes without hiding solved-unit regressions."""
    lengths = {len(unit_ids), len(reference_outcomes), len(candidate_outcomes)}
    if len(lengths) != 1 or not unit_ids:
        raise ValueError("paired outcome inputs must be aligned and non-empty")
    normalized_ids = tuple(str(unit_id).strip() for unit_id in unit_ids)
    if any(not unit_id for unit_id in normalized_ids):
        raise ValueError("unit_ids must be non-empty")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("unit_ids must be unique")

    reference = tuple(bool(outcome) for outcome in reference_outcomes)
    candidate = tuple(bool(outcome) for outcome in candidate_outcomes)
    beneficial = tuple(
        unit_id
        for unit_id, before, after in zip(
            normalized_ids, reference, candidate, strict=True
        )
        if not before and after
    )
    harmful = tuple(
        unit_id
        for unit_id, before, after in zip(
            normalized_ids, reference, candidate, strict=True
        )
        if before and not after
    )
    preserved = tuple(
        unit_id
        for unit_id, before, after in zip(
            normalized_ids, reference, candidate, strict=True
        )
        if before and after
    )
    unchanged_failure = tuple(
        unit_id
        for unit_id, before, after in zip(
            normalized_ids, reference, candidate, strict=True
        )
        if not before and not after
    )
    reference_successes = sum(reference)
    candidate_successes = sum(candidate)
    return PairedOutcomeAudit(
        total_units=len(normalized_ids),
        reference_successes=reference_successes,
        candidate_successes=candidate_successes,
        beneficial_flip_ids=beneficial,
        harmful_flip_ids=harmful,
        preserved_success_ids=preserved,
        unchanged_failure_ids=unchanged_failure,
        net_success_delta=candidate_successes - reference_successes,
    )


def protected_success_candidate_acceptance(
    *,
    unit_ids: Sequence[str],
    reference_outcomes: Sequence[bool | int],
    candidate_outcomes: Sequence[bool | int],
    changed: bool,
    minimum_beneficial_flips: int = 1,
    maximum_harmful_flips: int = 0,
    require_strict_score_improvement: bool = True,
) -> ProtectedSuccessAcceptanceDecision:
    """Require a candidate to expand, rather than trade away, solved coverage."""
    if minimum_beneficial_flips < 0:
        raise ValueError("minimum_beneficial_flips must be non-negative")
    if maximum_harmful_flips < 0:
        raise ValueError("maximum_harmful_flips must be non-negative")
    paired = paired_outcome_audit(
        unit_ids=unit_ids,
        reference_outcomes=reference_outcomes,
        candidate_outcomes=candidate_outcomes,
    )
    gates = {
        "candidate_changed": bool(changed),
        "beneficial_flip_floor": (
            len(paired.beneficial_flip_ids) >= minimum_beneficial_flips
        ),
        "protected_success_regression_budget": (
            len(paired.harmful_flip_ids) <= maximum_harmful_flips
        ),
        "strict_score_improvement": (
            paired.net_success_delta > 0
            if require_strict_score_improvement
            else paired.net_success_delta >= 0
        ),
    }
    return ProtectedSuccessAcceptanceDecision(
        accepted=all(gates.values()),
        minimum_beneficial_flips=minimum_beneficial_flips,
        maximum_harmful_flips=maximum_harmful_flips,
        require_strict_score_improvement=require_strict_score_improvement,
        paired_outcomes=paired,
        gates=gates,
    )


def canonical_trace_text(row: Mapping[str, Any]) -> str:
    """Serialize one adapter-normalized trace without benchmark assumptions."""
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalized_trace_group(row: Mapping[str, Any]) -> str:
    """Read the adapter's generic task-group label and fail closed if absent."""
    for field in ("task_group", "task_type", "group"):
        value = " ".join(str(row.get(field) or "").split())
        if value:
            return value
    raise ValueError("trace row lacks task_group/task_type/group")


def trace_data_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[TraceDatum, ...]:
    data = []
    for row in rows:
        unit_id = str(row.get("id") or "").strip()
        data.append(
            TraceDatum(
                unit_id=unit_id,
                group_id=normalized_trace_group(row),
                success=bool(row.get("hard")),
                text=canonical_trace_text(row),
            )
        )
    if not data or len({row.unit_id for row in data}) != len(data):
        raise ValueError("trace rows must have non-empty unique ids")
    return tuple(data)


def entropic_group_update(
    traces: Sequence[TraceDatum],
    *,
    step_size: float,
    previous_weights: Mapping[str, float] | None = None,
    beta_smoothing: float = 0.5,
) -> tuple[dict[str, float], dict[str, float]]:
    """Exponentiated-gradient ascent on smoothed per-group failure risk."""
    if not traces:
        raise ValueError("at least one trace is required")
    if not math.isfinite(step_size) or step_size < 0.0:
        raise ValueError("entropic step_size must be finite and non-negative")
    if not math.isfinite(beta_smoothing) or beta_smoothing <= 0.0:
        raise ValueError("beta_smoothing must be finite and positive")
    groups = sorted({row.group_id for row in traces})
    counts = {group: 0 for group in groups}
    failures = {group: 0 for group in groups}
    for row in traces:
        counts[row.group_id] += 1
        failures[row.group_id] += int(not row.success)
    risks = {
        group: (failures[group] + beta_smoothing)
        / (counts[group] + 2.0 * beta_smoothing)
        for group in groups
    }
    if previous_weights is None:
        prior = {group: 1.0 / len(groups) for group in groups}
    else:
        if set(previous_weights) != set(groups):
            raise ValueError("previous group weights do not match observed groups")
        if any(float(previous_weights[group]) <= 0.0 for group in groups):
            raise ValueError("previous group weights must be positive")
        total = sum(float(previous_weights[group]) for group in groups)
        prior = {group: float(previous_weights[group]) / total for group in groups}
    logits = {
        group: math.log(prior[group]) + step_size * risks[group]
        for group in groups
    }
    maximum = max(logits.values())
    unnormalized = {
        group: math.exp(logits[group] - maximum) for group in groups
    }
    normalizer = sum(unnormalized.values())
    weights = {
        group: unnormalized[group] / normalizer for group in groups
    }
    return weights, risks


def _token_shingles(text: str) -> frozenset[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())
    if not tokens:
        return frozenset({"<empty>"})
    if len(tokens) == 1:
        return frozenset(tokens)
    return frozenset(f"{left}\u241f{right}" for left, right in zip(tokens, tokens[1:]))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def risk_coverage_trace_selection(
    traces: Sequence[TraceDatum],
    *,
    character_budget: int,
    maximum_items: int,
    max_characters_per_trace: int,
    entropic_step_size: float,
    previous_group_weights: Mapping[str, float] | None = None,
    facility_location_weight: float = 1.0,
    group_saturation_weight: float = 0.25,
    failure_weight: float = 2.0,
    same_group_similarity: float = 0.2,
) -> RiskCoverageSelection:
    """Greedily maximize a monotone risk-weighted submodular coreset.

    The objective is a non-negative sum of facility-location coverage and a
    concave per-group saturation term.  Selection is deterministic and obeys
    both the item-cardinality and serialized-character budgets.
    """
    if not traces or len({row.unit_id for row in traces}) != len(traces):
        raise ValueError("traces must be non-empty with unique unit ids")
    if character_budget <= 0 or maximum_items <= 0 or max_characters_per_trace <= 0:
        raise ValueError("selection budgets must be positive")
    parameters = (
        facility_location_weight,
        group_saturation_weight,
        failure_weight,
        same_group_similarity,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in parameters):
        raise ValueError("selection weights must be finite and non-negative")
    if same_group_similarity > 1.0:
        raise ValueError("same_group_similarity must not exceed one")
    group_weights, group_risks = entropic_group_update(
        traces,
        step_size=entropic_step_size,
        previous_weights=previous_group_weights,
    )
    group_counts = {
        group: sum(row.group_id == group for row in traces)
        for group in group_weights
    }
    demand_weights = [
        group_weights[row.group_id]
        / group_counts[row.group_id]
        * (failure_weight if not row.success else 1.0)
        for row in traces
    ]
    demand_normalizer = sum(demand_weights)
    demand_weights = [value / demand_normalizer for value in demand_weights]
    shingles = [_token_shingles(row.text) for row in traces]
    similarities = [
        [
            (1.0 - same_group_similarity) * _jaccard(shingles[i], shingles[j])
            + same_group_similarity * float(traces[i].group_id == traces[j].group_id)
            for j in range(len(traces))
        ]
        for i in range(len(traces))
    ]
    costs = [min(len(row.text), max_characters_per_trace) for row in traces]
    selected: list[int] = []
    covered = [0.0] * len(traces)
    selected_groups: dict[str, int] = {group: 0 for group in group_weights}
    used = 0
    objective = 0.0
    remaining = set(range(len(traces)))
    while remaining and len(selected) < maximum_items:
        choices: list[tuple[float, float, str, int, float]] = []
        for candidate in remaining:
            if used + costs[candidate] > character_budget:
                continue
            coverage_gain = sum(
                demand_weights[index]
                * max(0.0, similarities[index][candidate] - covered[index])
                for index in range(len(traces))
            )
            group = traces[candidate].group_id
            group_gain = (
                group_weights[group]
                * (
                    math.sqrt(selected_groups[group] + 1.0)
                    - math.sqrt(selected_groups[group])
                )
            )
            marginal = (
                facility_location_weight * coverage_gain
                + group_saturation_weight * group_gain
            )
            choices.append(
                (
                    marginal / max(costs[candidate], 1),
                    marginal,
                    traces[candidate].unit_id,
                    candidate,
                    coverage_gain,
                )
            )
        if not choices:
            break
        _, marginal, _, chosen, _ = min(
            choices, key=lambda row: (-row[0], -row[1], row[2])
        )
        selected.append(chosen)
        remaining.remove(chosen)
        used += costs[chosen]
        objective += marginal
        selected_groups[traces[chosen].group_id] += 1
        covered = [
            max(covered[index], similarities[index][chosen])
            for index in range(len(traces))
        ]
    if not selected:
        raise ValueError("no trace fits the selection budget")
    selected_rows = [traces[index] for index in selected]
    return RiskCoverageSelection(
        selected_unit_ids=tuple(row.unit_id for row in selected_rows),
        selected_failure_ids=tuple(row.unit_id for row in selected_rows if not row.success),
        selected_success_ids=tuple(row.unit_id for row in selected_rows if row.success),
        group_weights=group_weights,
        group_failure_rates=group_risks,
        objective_value=objective,
        used_characters=used,
        character_budget=character_budget,
        maximum_items=maximum_items,
        facility_location_weight=facility_location_weight,
        group_saturation_weight=group_saturation_weight,
        failure_weight=failure_weight,
        entropic_step_size=entropic_step_size,
    )


def _kl_divergence(weights: Sequence[float], prior: Sequence[float]) -> float:
    return sum(
        weight * math.log(weight / base)
        for weight, base in zip(weights, prior, strict=True)
        if weight > 0.0
    )


def kl_robust_group_value(
    group_values: Mapping[str, float], *, kl_radius: float
) -> tuple[float, dict[str, float]]:
    """Return the worst group mixture in a KL ball around a uniform prior."""
    if not group_values:
        raise ValueError("group_values must be non-empty")
    if not math.isfinite(kl_radius) or kl_radius < 0.0:
        raise ValueError("kl_radius must be finite and non-negative")
    groups = sorted(group_values)
    values = [float(group_values[group]) for group in groups]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("group values must be finite")
    prior = [1.0 / len(groups)] * len(groups)
    if kl_radius == 0.0 or max(values) == min(values):
        return sum(values) / len(values), dict(zip(groups, prior, strict=True))
    minimum = min(values)
    minimum_indices = [index for index, value in enumerate(values) if value == minimum]
    limiting_kl = math.log(len(groups) / len(minimum_indices))
    if kl_radius >= limiting_kl - 1e-12:
        weights = [
            1.0 / len(minimum_indices) if index in minimum_indices else 0.0
            for index in range(len(groups))
        ]
        return minimum, dict(zip(groups, weights, strict=True))

    def tilted(temperature: float) -> list[float]:
        logits = [-value / temperature for value in values]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        total = sum(weights)
        return [weight / total for weight in weights]

    low = 1e-12
    high = 1.0
    while _kl_divergence(tilted(high), prior) > kl_radius:
        high *= 2.0
    for _ in range(160):
        middle = (low + high) / 2.0
        if _kl_divergence(tilted(middle), prior) > kl_radius:
            low = middle
        else:
            high = middle
    weights = tilted(high)
    value = sum(weight * group_value for weight, group_value in zip(weights, values, strict=True))
    return value, dict(zip(groups, weights, strict=True))


def robust_candidate_acceptance(
    *,
    unit_ids: Sequence[str],
    group_ids: Sequence[str],
    reference_outcomes: Sequence[bool | int],
    candidate_outcomes: Sequence[bool | int],
    changed: bool,
    edit_lexemes: int,
    prompt_lexeme_budget: int,
    kl_radius: float,
    complexity_weight: float,
    minimum_robust_gain: float = 0.0,
    require_strict_empirical_improvement: bool = True,
    require_no_group_regression: bool = True,
) -> RobustAcceptanceDecision:
    """Accept a skill edit only when its KL-robust paired utility clears cost."""
    lengths = {
        len(unit_ids), len(group_ids), len(reference_outcomes), len(candidate_outcomes)
    }
    if len(lengths) != 1 or not unit_ids:
        raise ValueError("paired acceptance inputs must be aligned and non-empty")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_ids must be unique")
    if any(not str(group).strip() for group in group_ids):
        raise ValueError("group_ids must be non-empty")
    if edit_lexemes < 0 or prompt_lexeme_budget <= 0:
        raise ValueError("edit and prompt lexeme counts are invalid")
    if not math.isfinite(complexity_weight) or complexity_weight < 0.0:
        raise ValueError("complexity_weight must be finite and non-negative")
    differences = [
        float(bool(candidate)) - float(bool(reference))
        for reference, candidate in zip(
            reference_outcomes, candidate_outcomes, strict=True
        )
    ]
    empirical_gain = sum(differences) / len(differences)
    groups = sorted(set(group_ids))
    group_gains = {
        group: sum(
            difference
            for difference, observed_group in zip(differences, group_ids, strict=True)
            if observed_group == group
        )
        / sum(observed_group == group for observed_group in group_ids)
        for group in groups
    }
    robust_gain, adversarial = kl_robust_group_value(
        group_gains, kl_radius=kl_radius
    )
    penalty = complexity_weight * edit_lexemes / prompt_lexeme_budget
    penalized = robust_gain - penalty
    gates = {
        "candidate_changed": bool(changed),
        "strict_empirical_improvement": (
            empirical_gain > 0.0
            if require_strict_empirical_improvement
            else empirical_gain >= 0.0
        ),
        "no_group_regression": (
            min(group_gains.values()) >= 0.0
            if require_no_group_regression
            else True
        ),
        "robust_gain_clears_complexity": penalized >= minimum_robust_gain,
    }
    return RobustAcceptanceDecision(
        accepted=all(gates.values()),
        empirical_gain=empirical_gain,
        robust_gain=robust_gain,
        complexity_penalty=penalty,
        penalized_robust_gain=penalized,
        group_gains=group_gains,
        adversarial_group_weights=adversarial,
        kl_radius=kl_radius,
        minimum_robust_gain=minimum_robust_gain,
        require_strict_empirical_improvement=require_strict_empirical_improvement,
        require_no_group_regression=require_no_group_regression,
        gates=gates,
    )


__all__ = [
    "PairedOutcomeAudit",
    "ProtectedSuccessAcceptanceDecision",
    "RiskCoverageSelection",
    "RobustAcceptanceDecision",
    "TraceDatum",
    "canonical_trace_text",
    "entropic_group_update",
    "kl_robust_group_value",
    "normalized_trace_group",
    "paired_outcome_audit",
    "protected_success_candidate_acceptance",
    "risk_coverage_trace_selection",
    "robust_candidate_acceptance",
    "trace_data_from_rows",
]
