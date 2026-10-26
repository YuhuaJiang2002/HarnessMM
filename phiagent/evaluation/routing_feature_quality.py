"""Quality-gated token accounting for complete, paired routing outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from phiagent.evaluation.chisel_component_quality import paired_finite_sample_lower_bound


def aggregate_routing_outcomes(
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Mapping[str, str],
) -> dict[str, dict[str, int | float]]:
    if not methods or not groups:
        raise ValueError("complete matched methods and question groups are required")
    totals = {}
    for name, rows in methods.items():
        if set(rows) != set(groups):
            raise ValueError(f"incomplete matched task coverage: {name}")
        for row in rows.values():
            if (
                type(row["exact_match"]) is not int or row["exact_match"] not in (0, 1)
                or type(row["f1"]) not in (int, float)
                or not math.isfinite(row["f1"]) or not 0 <= row["f1"] <= 1
                or any(type(row[key]) is not int or row[key] < 0
                       for key in ("prompt_tokens", "completion_tokens", "tokens"))
                or row["prompt_tokens"] <= 0
                or row["tokens"] != row["prompt_tokens"] + row["completion_tokens"]
            ):
                raise ValueError("official scores and complete additive token receipts are required")
        totals[name] = {
            "tasks": len(rows),
            "successes": sum(row["exact_match"] for row in rows.values()),
            "exact_match": 100 * sum(row["exact_match"] for row in rows.values()) / len(rows),
            "f1": 100 * math.fsum(row["f1"] for row in rows.values()) / len(rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows.values()),
            "completion_tokens": sum(row["completion_tokens"] for row in rows.values()),
            "total_tokens": sum(row["tokens"] for row in rows.values()),
        }
    return totals


def quality_gated_token_savings(
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Mapping[str, str],
    *,
    comparisons: Sequence[tuple[str, str]],
    generation_seed: int,
    alpha: float = 0.05,
    family_size: int | None = None,
) -> dict[str, Any]:
    if type(generation_seed) is not int:
        raise ValueError("the original generation seed must be an integer")
    if (
        not comparisons
        or len(set(comparisons)) != len(comparisons)
        or any(left == right or left not in methods or right not in methods
               for left, right in comparisons)
    ):
        raise ValueError("distinct, declared candidate/reference comparisons are required")
    totals = aggregate_routing_outcomes(methods, groups)
    if family_size is None:
        family_size = 2 * len(comparisons)
    if type(family_size) is not int or family_size < 2 * len(comparisons):
        raise ValueError("the simultaneous family cannot omit a declared comparison endpoint")
    effects = []
    for candidate, reference in comparisons:
        bounds = {}
        for metric in ("exact_match", "f1"):
            bounds[metric] = paired_finite_sample_lower_bound(
                {generation_seed: {task: row[metric] for task, row in methods[candidate].items()}},
                {generation_seed: {task: row[metric] for task, row in methods[reference].items()}},
                groups,
                seeds=[generation_seed],
                score_upper_bound=1,
                family_size=family_size,
                alpha=alpha,
                binary=metric == "exact_match",
            )
        quality_pass = all(
            value["one_sided_lower_bound"] >= 0 for value in bounds.values()
        )
        saved = totals[reference]["total_tokens"] - totals[candidate]["total_tokens"]
        reduction = 100 * saved / totals[reference]["total_tokens"]
        effects.append({
            "candidate": candidate,
            "reference": reference,
            "bounds": bounds,
            "observed_quality_nonregression": all(
                value["observed_difference"] >= 0 for value in bounds.values()
            ),
            "zero_loss_simultaneous_bound_passed": quality_pass,
            "total_tokens_saved": saved,
            "observed_token_reduction_pct": reduction,
            "quality_gated_token_reduction_pct": reduction if quality_pass else None,
            "quality_gated_cost_win": quality_pass and saved > 0,
            "decision": (
                "QUALITY_BOUND_AND_COST_PASS" if quality_pass and saved > 0
                else "QUALITY_BOUND_PASS_NO_TOKEN_SAVING" if quality_pass
                else "QUALITY_UNESTABLISHED_NO_SAVING_CREDIT"
            ),
        })
    return {
        "schema_version": "phiagent-routing-feature-quality/v1",
        "methods": totals,
        "comparisons": effects,
        "statistics": {
            "independent_unit": "Normalized-question group",
            "generation_seed": generation_seed,
            "familywise_alpha": alpha,
            "family_size": family_size,
            "success_loss_tolerance": 0.0,
            "f1_loss_tolerance": 0.0,
            "token_metric": "1 - sum(candidate prompt+completion) / sum(reference prompt+completion)",
            "all_failed_answers_counted_in_cost": True,
            "router_training_seeds_are_distinct_policies_not_independent_tasks": True,
        },
        "claim_boundary": (
            "Conditional numerical bounds, not per-question guarantees. Freshness, frozen "
            "selection, source protocol, and independent groups require a separate audit. "
            "Observed ties do not prove zero loss. No paper admission is implied."
        ),
    }
