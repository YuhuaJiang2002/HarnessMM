"""RRSI-inspired search gates, independent of models and benchmark loaders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from statistics import mean, stdev
from typing import Mapping, Sequence


REFERENCE = {
    "repository": "https://github.com/google-research/rrsi",
    "revision": "be50316e1db05914068a973f322770ef08ed7ba1",
    "paper": "https://arxiv.org/abs/2609.24972",
    "identity": "RRSI-inspired adaptation; official RRSI outer loop not executed",
}
COMPONENTS = frozenset({
    "instruction", "control_flow", "context", "configuration",
    "tool", "skill", "memory", "subagent",
})
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def edit_budget(round_index: int, rounds: int, minimum: int, maximum: int) -> int:
    values = (round_index, rounds, minimum, maximum)
    if (any(type(value) is not int for value in values) or rounds < 1
            or not 0 <= round_index <= rounds or not 1 <= minimum <= maximum):
        raise ValueError("Require an explicit, valid annealed independent-edit budget")
    value = minimum + (maximum - minimum) * (
        1 + math.cos(math.pi * round_index / rounds)
    ) / 2
    return math.ceil(round(value, 9))


@dataclass(frozen=True)
class Edit:
    component: str
    hypothesis: str
    trigger: str
    regression_risk: str
    operation: str

    def __post_init__(self) -> None:
        if (self.component not in COMPONENTS or self.operation not in ("add", "modify", "prune")
                or any(not isinstance(text, str) or not text.strip()
                       for text in (self.hypothesis, self.trigger, self.regression_risk))):
            raise ValueError("Each independent edit needs a mechanism, trigger and regression risk")

    @property
    def identity(self) -> str:
        return digest(self.__dict__)


def screen_proposal(
    edits: Sequence[Edit], runtime_diff: str, *, budget: int,
    prior_identities: Sequence[str], forbidden_literals: Sequence[str],
    protected_patterns: Sequence[str] = (),
) -> dict:
    if type(budget) is not int or budget < 1 or not isinstance(runtime_diff, str):
        raise ValueError("An explicit edit budget and runtime diff are required")
    if (any(not isinstance(value, str) or not value for value in forbidden_literals)
            or any(not isinstance(value, str) or not value for value in protected_patterns)):
        raise ValueError("Critic literals and patterns cannot be empty")
    reasons = []
    identities = [edit.identity for edit in edits]
    if not edits or len(edits) > budget or len(set(identities)) != len(identities):
        reasons.append("empty_duplicate_or_over_budget_edit_set")
    if not runtime_diff.strip():
        reasons.append("empty_runtime_change")
    if set(identities) & set(prior_identities):
        reasons.append("previously_measured_hypothesis_requires_new_evidence")
    if any(literal in runtime_diff for literal in forbidden_literals):
        reasons.append("task_specific_literal_in_runtime_change")
    if any(re.search(pattern, runtime_diff) for pattern in protected_patterns):
        reasons.append("protected_runtime_surface_changed")
    if re.search(r"AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,}", runtime_diff):
        reasons.append("credential_like_literal_in_runtime_change")
    return {
        "status": "REJECTED_PRECHECK" if reasons else "PASS_DETERMINISTIC_PRECHECK",
        "reasons": reasons, "declared_edits": [edit.__dict__ for edit in edits],
        "edit_identities": identities, "runtime_diff_sha256":
        hashlib.sha256(runtime_diff.encode()).hexdigest(),
        "semantic_critic_still_required": not reasons,
        "reference": REFERENCE, "paper_admitted": False,
    }


@dataclass(frozen=True)
class Observation:
    unit: str
    repetition: int
    scores: Mapping[str, float]
    actor_tokens: Mapping[str, int]
    all_provider_tokens: Mapping[str, int]
    weight: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.unit, str) or not self.unit
                or type(self.repetition) is not int or self.repetition < 1
                or not self.scores or type(self.weight) is not int or self.weight < 1):
            raise ValueError("An independent unit, repetition and native endpoints are required")
        for name, score in self.scores.items():
            if not isinstance(name, str) or not name or not 0 <= _finite(score, name) <= 1:
                raise ValueError("Native endpoint observations must be in [0, 1]")
        for usage in (self.actor_tokens, self.all_provider_tokens):
            if (set(usage) != set(TOKEN_FIELDS)
                    or any(type(usage[key]) is not int or usage[key] < 0 for key in TOKEN_FIELDS)
                    or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]):
                raise ValueError("Complete input, output and total token usage must reconcile")
        if any(self.all_provider_tokens[key] < self.actor_tokens[key] for key in TOKEN_FIELDS):
            raise ValueError("All-provider usage cannot omit actor usage")
        if self.actor_tokens["total_tokens"] <= 0:
            raise ValueError("A native execution requires positive measured actor usage")


def matched(
    rows: Sequence[Observation], units: Sequence[str], repetitions: Sequence[int],
    endpoints: Sequence[str],
) -> dict[tuple[str, int], Observation]:
    if (not units or len(set(units)) != len(units) or not repetitions
            or len(set(repetitions)) != len(repetitions)
            or any(type(value) is not int or value < 1 for value in repetitions)
            or not endpoints or len(set(endpoints)) != len(endpoints)):
        raise ValueError("Freeze unique population, repetitions and native endpoints")
    expected = {(unit, repetition) for unit in units for repetition in repetitions}
    observed = {(row.unit, row.repetition): row for row in rows}
    if (len(rows) != len(expected) or len(observed) != len(rows)
            or set(observed) != expected
            or any(set(row.scores) != set(endpoints) for row in rows)):
        raise ValueError("Partial, duplicated or unmatched native populations are not evidence")
    if any(len({observed[unit, repeat].weight for repeat in repetitions}) != 1
           for unit in units):
        raise ValueError("Repeated evaluations must preserve every context-group denominator")
    return observed


def quality(rows: Sequence[Observation], endpoint: str) -> float:
    return sum(row.scores[endpoint] * row.weight for row in rows) / sum(
        row.weight for row in rows
    )


def noise_band(
    base: Sequence[Observation], units: Sequence[str], repetitions: Sequence[int],
    endpoint: str, *, z: float = 2.0,
) -> dict:
    observed = matched(base, units, repetitions, (endpoint,))
    if len(repetitions) < 2:
        raise ValueError("One execution per task cannot calibrate repeated-execution noise")
    if _finite(z, "noise multiplier") <= 0:
        raise ValueError("The frozen noise multiplier must be positive")
    scores = [
        quality([observed[unit, repeat] for unit in units], endpoint)
        for repeat in repetitions
    ]
    delta = z * math.sqrt(2) * stdev(scores)
    return {
        "delta": delta, "z": z, "endpoint": endpoint,
        "method": "repeated unchanged parent on identical independent units",
        "base_scores_per_repetition": scores,
        "independent_units": len(units), "repetitions": list(repetitions),
        "new_model_calls": 0,
        "zero_band_is_observed_not_imputed": delta == 0,
    }


def selection_cost_rule(
    quality_delta: float, relative_cost_delta: float, delta: float, *,
    beta0: float, beta1: float, quality_weight: float, cost_weight: float,
) -> dict:
    values = (quality_delta, relative_cost_delta, delta, beta0, beta1,
              quality_weight, cost_weight)
    if (any(_finite(value, "selection parameter") != value for value in values)
            or delta < 0 or beta0 < 0 or beta1 < 0
            or quality_weight <= 0 or cost_weight <= 0):
        raise ValueError("Freeze finite, nonnegative regularization and positive objective weights")
    if quality_delta > delta:
        allowance = beta0 + beta1 * quality_delta
        return {
            "passed": relative_cost_delta <= allowance,
            "rule": "measured_quality_gain_pays_for_extra_execution_cost",
            "allowed_relative_cost_delta": allowance,
        }
    value = quality_weight * quality_delta - cost_weight * relative_cost_delta
    return {
        "passed": value > 0, "rule": "within_noise_band_quality_cost_improvement",
        "objective_delta": value,
        "structural_novelty_is_not_a_quality_or_cost_credit": True,
    }


def history_directives(history: Sequence[Mapping], *, window: int, delta: float) -> dict:
    if type(window) is not int or window < 1 or _finite(delta, "noise band") < 0:
        raise ValueError("Freeze a positive history window and a calibrated noise band")
    measured = []
    seen = set()
    for record in history:
        identity = record["candidate_id"]
        edits = record["edits"]
        if (not isinstance(identity, str) or not identity or identity in seen
                or not edits or type(record["accepted"]) is not bool
                or record["phase"] != "train"):
            raise ValueError("History must contain distinct, TRAIN-only candidate verdicts")
        for edit in edits:
            Edit(**edit)
        _finite(record["quality_delta"], "measured quality delta")
        _finite(record["relative_actor_cost_delta"], "measured cost delta")
        if record.get("complete_native_population") is not True:
            raise ValueError("An incomplete candidate cannot become a measured search verdict")
        seen.add(identity)
        measured.append(record)
    tried = {edit["component"] for record in measured for edit in record["edits"]}
    recent = measured[-window:]
    stalled = len(recent) == window and not any(
        record["accepted"] and record["quality_delta"] > delta for record in recent
    )
    prune = sorted({
        edit["component"] for record in recent
        if record["accepted"] and record["quality_delta"] <= delta
        and record["relative_actor_cost_delta"] > 0
        for edit in record["edits"]
    })
    return {
        "full_history_candidate_ids": [record["candidate_id"] for record in measured],
        "prior_edit_identities": [
            Edit(**edit).identity for record in measured for edit in record["edits"]
        ],
        "stalled": stalled, "untried_components": sorted(COMPONENTS - tried),
        "propose_measured_removal_of_components": prune,
        "never_mutate_the_frozen_incumbent_during_pruning": True,
        "validation_and_test_feedback_used": False,
    }


def stability_selection(
    candidate: Sequence[Observation], incumbent: Sequence[Observation],
    units: Sequence[str], repetitions: Sequence[int], endpoints: Sequence[str], *,
    primary_endpoint: str, high_water_quality: Mapping[str, float],
    calibrated_noise: Mapping[str, float], beta0: float, beta1: float,
) -> dict:
    left = matched(candidate, units, repetitions, endpoints)
    right = matched(incumbent, units, repetitions, endpoints)
    if (primary_endpoint not in endpoints or set(high_water_quality) != set(endpoints)
            or set(calibrated_noise) != set(endpoints)
            or any(not 0 <= _finite(value, "high-water quality") <= 1
                   for value in high_water_quality.values())
            or any(_finite(value, "calibrated noise") < 0
                   for value in calibrated_noise.values())
            or any(left[key].weight != right[key].weight for key in left)):
        raise ValueError("Explicit calibrated bands and identical native endpoints are required")
    current = {name: quality(incumbent, name) for name in endpoints}
    proposed = {name: quality(candidate, name) for name in endpoints}
    if any(high_water_quality[name] < current[name] for name in endpoints):
        raise ValueError("The quality high-water record cannot omit the incumbent")
    score_delta = proposed[primary_endpoint] - current[primary_endpoint]
    token_delta = (
        sum(row.actor_tokens["total_tokens"] for row in candidate)
        / sum(row.actor_tokens["total_tokens"] for row in incumbent) - 1
    )
    rule = selection_cost_rule(
        score_delta, token_delta, calibrated_noise[primary_endpoint],
        beta0=beta0, beta1=beta1, quality_weight=1.0, cost_weight=1.0,
    )
    floor_pass = all(
        proposed[name] >= high_water_quality[name] - calibrated_noise[name]
        for name in endpoints
    )
    point_pass = all(proposed[name] >= current[name] for name in endpoints)
    accepted = floor_pass and point_pass and rule["passed"]
    return {
        "candidate_quality": proposed, "incumbent_quality": current,
        "noise_adjusted_high_water_floor_passed": floor_pass,
        "noncompensatory_point_quality_passed": point_pass,
        "quality_delta": score_delta, "relative_actor_cost_delta": token_delta,
        "cost_rule": rule, "candidate_for_protected_validation": accepted,
        "decision": "RETAIN_CANDIDATE" if accepted else "RETAIN_INCUMBENT",
        "quality_cost_tradeoff": accepted and token_delta > 0,
        "joint_point_quality_cost_gain": accepted and score_delta > 0 and token_delta < 0,
        "reference": REFERENCE, "paper_admitted": False,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def paired_uncertainty(
    candidate: Sequence[Observation], control: Sequence[Observation],
    units: Sequence[str], repetitions: Sequence[int], endpoints: Sequence[str],
    *, draws: int, seed: int,
) -> dict:
    left = matched(candidate, units, repetitions, endpoints)
    right = matched(control, units, repetitions, endpoints)
    if any(left[key].weight != right[key].weight for key in left):
        raise ValueError("Paired context groups must contain the same independent questions")
    if type(draws) is not int or draws < 100 or type(seed) is not int:
        raise ValueError("Freeze at least100 paired cluster-bootstrap draws and an integer seed")
    values = {
        unit: {
            "deltas": {
                name: mean(
                    left[unit, repeat].scores[name] - right[unit, repeat].scores[name]
                    for repeat in repetitions
                ) for name in endpoints
            },
            "weight": left[unit, repetitions[0]].weight,
            "candidate_actor": sum(left[unit, repeat].actor_tokens["total_tokens"]
                                   for repeat in repetitions),
            "control_actor": sum(right[unit, repeat].actor_tokens["total_tokens"]
                                 for repeat in repetitions),
            "candidate_all": sum(left[unit, repeat].all_provider_tokens["total_tokens"]
                                 for repeat in repetitions),
            "control_all": sum(right[unit, repeat].all_provider_tokens["total_tokens"]
                               for repeat in repetitions),
        } for unit in units
    }
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {name: [] for name in endpoints}
    ratios: dict[str, list[float]] = {"actor": [], "all_provider": []}
    for _ in range(draws):
        sampled = [values[units[rng.randrange(len(units))]] for _ in units]
        for name in endpoints:
            deltas[name].append(
                sum(row["deltas"][name] * row["weight"] for row in sampled)
                / sum(row["weight"] for row in sampled)
            )
        ratios["actor"].append(
            sum(row["candidate_actor"] for row in sampled)
            / sum(row["control_actor"] for row in sampled)
        )
        ratios["all_provider"].append(
            sum(row["candidate_all"] for row in sampled)
            / sum(row["control_all"] for row in sampled)
        )
    return {
        "quality_delta_ci95": {
            name: [_percentile(samples, .025), _percentile(samples, .975)]
            for name, samples in deltas.items()
        },
        "token_ratio_ci95": {
            name: [_percentile(samples, .025), _percentile(samples, .975)]
            for name, samples in ratios.items()
        },
        "independent_unit": "frozen task/scenario/context group",
        "units": len(units), "repetitions": list(repetitions),
        "draws": draws, "seed": seed,
        "scope": "paired population uncertainty, not repeated-execution noise calibration",
    }


def efficiency_gate(
    candidate: Sequence[Observation], controls: Mapping[str, Sequence[Observation]],
    units: Sequence[str], repetitions: Sequence[int], endpoints: Sequence[str],
    *, required_controls: Sequence[str], actor_ratio: float, draws: int, seed: int,
) -> dict:
    if (not required_controls or len(set(required_controls)) != len(required_controls)
            or set(controls) != set(required_controls)
            or not 0 < _finite(actor_ratio, "actor ratio") <= 1):
        raise ValueError("Every named matched control and a frozen cost objective are required")
    matched(candidate, units, repetitions, endpoints)
    candidate_quality = {
        name: quality(candidate, name) for name in endpoints
    }
    comparisons = {}
    for name in required_controls:
        rows = controls[name]
        matched(rows, units, repetitions, endpoints)
        control_quality = {key: quality(rows, key) for key in endpoints}
        uncertainty = paired_uncertainty(
            candidate, rows, units, repetitions, endpoints, draws=draws, seed=seed,
        )
        ratio = (
            sum(row.actor_tokens["total_tokens"] for row in candidate)
            / sum(row.actor_tokens["total_tokens"] for row in rows)
        )
        comparisons[name] = {
            "quality_delta": {
                key: candidate_quality[key] - control_quality[key] for key in endpoints
            },
            "point_quality_preserved": all(
                candidate_quality[key] >= control_quality[key] for key in endpoints
            ),
            "actor_token_ratio": ratio, "cost_objective_met": ratio <= actor_ratio,
            "uncertainty": uncertainty,
            "zero_margin_paired_noninferiority": all(
                uncertainty["quality_delta_ci95"][key][0] >= 0 for key in endpoints
            ),
        }
    point_pass = all(
        row["point_quality_preserved"] and row["cost_objective_met"]
        for row in comparisons.values()
    )
    return {
        "status": "PASS_TRAIN_POINT_GATE" if point_pass else "REJECTED_TRAIN_POINT_GATE",
        "comparisons": comparisons, "candidate_quality": candidate_quality,
        "frozen_actor_ratio": actor_ratio, "reference": REFERENCE,
        "candidate_for_protected_validation": point_pass,
        "quality_and_cost_confirmed_with_paired_uncertainty": point_pass and all(
            row["zero_margin_paired_noninferiority"]
            and row["uncertainty"]["token_ratio_ci95"]["actor"][1] <= actor_ratio
            for row in comparisons.values()
        ),
        "paper_admitted": False, "automatic_test_promotion": False,
    }
