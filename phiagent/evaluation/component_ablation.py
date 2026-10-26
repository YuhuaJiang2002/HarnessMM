"""Matched task-level summaries for binary component ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import math
import random
import statistics
from typing import Any


def exact_mcnemar(left_only: int, right_only: int) -> float:
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    if not differences or resamples <= 0:
        raise ValueError("paired bootstrap requires data and positive resamples")
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        statistics.fmean(
            differences[rng.randrange(size)] for _ in range(size)
        )
        for _ in range(resamples)
    )
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _arm_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scores = [int(row["score"]) for row in rows]
    tokens = sum(int(row.get("total_tokens", 0)) for row in rows)
    calls = sum(int(row.get("model_calls", 0)) for row in rows)
    return {
        "units": len(rows),
        "successes": sum(scores),
        "success_rate": statistics.fmean(scores),
        "total_tokens": tokens,
        "tokens_per_unit": tokens / len(rows),
        "model_calls": calls,
        "calls_per_unit": calls / len(rows),
    }


def _paired_contrast(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    left_scores = [int(row["score"]) for row in left]
    right_scores = [int(row["score"]) for row in right]
    differences = [
        left_score - right_score
        for left_score, right_score in zip(
            left_scores,
            right_scores,
            strict=True,
        )
    ]
    left_only = sum(value == 1 for value in differences)
    right_only = sum(value == -1 for value in differences)
    left_tokens = sum(int(row.get("total_tokens", 0)) for row in left)
    right_tokens = sum(int(row.get("total_tokens", 0)) for row in right)
    left_calls = sum(int(row.get("model_calls", 0)) for row in left)
    right_calls = sum(int(row.get("model_calls", 0)) for row in right)
    return {
        "mean_score_difference": statistics.fmean(differences),
        "paired_ci95": list(
            paired_bootstrap_interval(
                differences,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
        ),
        "left_only": left_only,
        "right_only": right_only,
        "ties": len(differences) - left_only - right_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar(
            left_only,
            right_only,
        ),
        "token_change_fraction": (
            left_tokens / right_tokens - 1.0
            if right_tokens
            else None
        ),
        "call_change_fraction": (
            left_calls / right_calls - 1.0
            if right_calls
            else None
        ),
    }


def _apply_holm(rows: Sequence[dict[str, Any]]) -> None:
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(
        sorted(
            range(count),
            key=lambda value: (
                float(rows[value]["mcnemar_exact_two_sided_p"]),
                value,
            ),
        )
    ):
        adjusted = min(
            1.0,
            (count - rank)
            * float(rows[index]["mcnemar_exact_two_sided_p"]),
        )
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["holm_reject_0_05"] = running <= 0.05


def summarize_component_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    units: Sequence[str],
    arm_components: Mapping[str, Sequence[str]],
    component_names: Sequence[str],
    full_arm: str,
    empty_arm: str,
    control_arms: Sequence[str] = (),
    bootstrap_seed: int = 20260907,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Summarize a matched binary-score ablation without imputing cells."""

    expected_units = tuple(units)
    components = tuple(component_names)
    component_set = frozenset(components)
    if (
        not expected_units
        or len(set(expected_units)) != len(expected_units)
        or not components
        or len(component_set) != len(components)
    ):
        raise ValueError("units and components must be non-empty and unique")
    specs = {
        str(arm): frozenset(str(value) for value in values)
        for arm, values in arm_components.items()
    }
    if (
        full_arm not in specs
        or empty_arm not in specs
        or specs[full_arm] != component_set
        or specs[empty_arm]
        or any(not values <= component_set for values in specs.values())
    ):
        raise ValueError("full/empty arm component identities differ")
    if not set(control_arms) <= set(specs):
        raise ValueError("control arms must exist in arm_components")
    causal_specs = {
        arm: values
        for arm, values in specs.items()
        if arm not in set(control_arms)
    }
    if len(set(causal_specs.values())) != len(causal_specs):
        raise ValueError("non-control arms must have unique component sets")

    matrix: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        arm = str(row["arm"])
        unit = str(row["unit"])
        if arm not in specs or unit not in expected_units:
            raise ValueError(f"unexpected ablation cell: {arm}/{unit}")
        if row.get("valid") is not True or row.get("score") not in {
            0,
            1,
        }:
            raise ValueError(f"invalid ablation cell: {arm}/{unit}")
        key = (arm, unit)
        if key in matrix:
            raise ValueError(f"duplicate ablation cell: {arm}/{unit}")
        matrix[key] = row
    expected = {
        (arm, unit) for arm in specs for unit in expected_units
    }
    if set(matrix) != expected:
        raise ValueError(
            f"ablation matrix coverage differs: missing={sorted(expected-set(matrix))}"
        )

    ordered = {
        arm: [matrix[(arm, unit)] for unit in expected_units]
        for arm in specs
    }
    summaries = {
        arm: {
            "components": sorted(specs[arm]),
            **_arm_summary(arm_rows),
        }
        for arm, arm_rows in ordered.items()
    }
    full_minus: dict[str, Any] = {}
    one_only: dict[str, Any] = {}
    for index, component in enumerate(components):
        minus_matches = [
            arm
            for arm, values in causal_specs.items()
            if values == component_set - {component}
        ]
        only_matches = [
            arm
            for arm, values in causal_specs.items()
            if values == {component}
        ]
        if len(minus_matches) != 1 or len(only_matches) != 1:
            raise ValueError(
                f"component {component} lacks exactly one minus/only arm"
            )
        minus_arm = minus_matches[0]
        only_arm = only_matches[0]
        full_contrast = _paired_contrast(
            ordered[full_arm],
            ordered[minus_arm],
            bootstrap_seed=bootstrap_seed + index * 2,
            bootstrap_resamples=bootstrap_resamples,
        )
        isolated_contrast = _paired_contrast(
            ordered[only_arm],
            ordered[empty_arm],
            bootstrap_seed=bootstrap_seed + index * 2 + 1,
            bootstrap_resamples=bootstrap_resamples,
        )
        full_minus[component] = {
            "full_arm": full_arm,
            "ablated_arm": minus_arm,
            **full_contrast,
        }
        one_only[component] = {
            "component_arm": only_arm,
            "empty_arm": empty_arm,
            **isolated_contrast,
        }
    primary_contrasts = [
        full_minus[component] for component in components
    ] + [one_only[component] for component in components]
    _apply_holm(primary_contrasts)

    interactions: dict[str, Any] = {}
    for left, right in combinations(components, 2):
        matches = {values: arm for arm, values in causal_specs.items()}
        required = (
            frozenset(),
            frozenset({left}),
            frozenset({right}),
            frozenset({left, right}),
        )
        if not all(values in matches for values in required):
            continue
        per_unit = []
        for unit in expected_units:
            empty = int(matrix[(matches[frozenset()], unit)]["score"])
            left_score = int(
                matrix[(matches[frozenset({left})], unit)]["score"]
            )
            right_score = int(
                matrix[(matches[frozenset({right})], unit)]["score"]
            )
            pair = int(
                matrix[(matches[frozenset({left, right})], unit)][
                    "score"
                ]
            )
            per_unit.append(pair - left_score - right_score + empty)
        interactions[f"{left}+{right}"] = {
            "mean_difference_in_differences": statistics.fmean(per_unit),
            "paired_ci95": list(
                paired_bootstrap_interval(
                    per_unit,
                    seed=bootstrap_seed
                    + len(components) * 2
                    + len(interactions),
                    resamples=bootstrap_resamples,
                )
            ),
        }

    controls = {
        arm: _paired_contrast(
            ordered[full_arm],
            ordered[arm],
            bootstrap_seed=bootstrap_seed
            + len(components) * 2
            + len(interactions)
            + index,
            bootstrap_resamples=bootstrap_resamples,
        )
        for index, arm in enumerate(control_arms)
    }
    _apply_holm(list(controls.values()))
    usefulness = {}
    for component in components:
        conditional = full_minus[component]
        isolated = one_only[component]
        conditional_delta = float(conditional["mean_score_difference"])
        isolated_delta = float(isolated["mean_score_difference"])
        conditional_confirmed = float(conditional["paired_ci95"][0]) > 0
        isolated_confirmed = float(isolated["paired_ci95"][0]) > 0
        conditional_familywise = (
            conditional["holm_reject_0_05"]
            and conditional["left_only"] > conditional["right_only"]
        )
        isolated_familywise = (
            isolated["holm_reject_0_05"]
            and isolated["left_only"] > isolated["right_only"]
        )
        usefulness[component] = {
            "full_context_status": (
                "confirmed_positive"
                if conditional_confirmed
                else "directional_positive"
                if conditional_delta > 0
                else "directional_negative"
                if conditional_delta < 0
                else "no_point_effect"
            ),
            "isolated_status": (
                "confirmed_positive"
                if isolated_confirmed
                else "directional_positive"
                if isolated_delta > 0
                else "directional_negative"
                if isolated_delta < 0
                else "no_point_effect"
            ),
            "protected_success_contribution": (
                conditional["left_only"] > 0
                and conditional["right_only"] == 0
            ),
            "bootstrap_positive_full_context": conditional_confirmed,
            "bootstrap_positive_isolation": isolated_confirmed,
            "directional_useful": (
                conditional_delta > 0 or isolated_delta > 0
            ),
            "supported_useful": (
                conditional_familywise or isolated_familywise
            ),
        }
    return {
        "independent_unit": "official task",
        "units": list(expected_units),
        "components": list(components),
        "full_arm": full_arm,
        "empty_arm": empty_arm,
        "arm_summaries": summaries,
        "full_minus_component": full_minus,
        "component_only": one_only,
        "available_two_way_interactions": interactions,
        "matched_controls_vs_full": controls,
        "component_usefulness": usefulness,
        "bootstrap": {
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
            "cluster": "official task",
        },
        "multiplicity": {
            "primary_family": (
                "all Full-minus-component and component-only exact "
                "McNemar contrasts"
            ),
            "correction": "Holm",
            "alpha": 0.05,
            "interactions": "exploratory_unadjusted",
            "matched_controls": "separate_Holm_family",
        },
    }


__all__ = [
    "exact_mcnemar",
    "paired_bootstrap_interval",
    "summarize_component_ablation",
]
