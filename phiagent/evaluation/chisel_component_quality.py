"""Paired whole-question inference for binary EM and fractional official F1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import itertools
import math
import random
from typing import Any

from phiagent.evaluation.component_ablation import _quantile
from phiagent.evaluation.polyglot_self_evolution_statistics import holm_adjust
from phiagent.learning.residual_meta_harness import clopper_pearson_lower, clopper_pearson_upper


def quality_matched_costs(methods: Mapping[str, Mapping[str, float]], references: Sequence[str]) -> dict[str, Any]:
    if (not methods or not references or len(set(references)) != len(references)
            or any(reference not in methods for reference in references)):
        raise ValueError("complete declared quality references are required")
    counts = set()
    for method in methods.values():
        if (type(method["tasks"]) is not int or method["tasks"] <= 0
                or type(method["total_tokens"]) is not int or method["total_tokens"] <= 0
                or any(type(method[metric]) not in (int, float) or not math.isfinite(method[metric])
                       or not 0 <= method[metric] <= 100 for metric in ("exact_match", "f1"))):
            raise ValueError("finite official scores and complete positive task/token counts are required")
        counts.add(method["tasks"])
    if len(counts) != 1:
        raise ValueError("matched complete task counts are required")
    floor = {metric: max(methods[reference][metric] for reference in references) for metric in ("exact_match", "f1")}
    points = {}
    for arm, method in methods.items():
        quality_pass = all(method[metric] + 1e-12 >= floor[metric] for metric in floor)
        cheaper = all(method["total_tokens"] < methods[reference]["total_tokens"] for reference in references)
        dominated_by = [
            other for other, comparison in methods.items() if other != arm
            and comparison["total_tokens"] <= method["total_tokens"]
            and all(comparison[metric] + 1e-12 >= method[metric] for metric in floor)
            and (comparison["total_tokens"] < method["total_tokens"]
                 or any(comparison[metric] > method[metric] + 1e-12 for metric in floor))
        ]
        points[arm] = {
            **method, "quality_floor_pass": quality_pass,
            "cheaper_than_every_reference": cheaper,
            "observed_same_quality_cost_win": quality_pass and cheaper,
            "dominated_by": sorted(dominated_by),
            "quality_delta_from_floor_pp": {metric: method[metric] - floor[metric] for metric in floor},
            "token_reduction_pct_by_reference": {
                reference: 100 * (1 - method["total_tokens"] / methods[reference]["total_tokens"])
                for reference in references
            },
        }
    feasible = sorted(
        (arm for arm in points if points[arm]["quality_floor_pass"]),
        key=lambda arm: (points[arm]["total_tokens"], arm),
    )
    cheapest = points[feasible[0]]["total_tokens"] if feasible else None
    return {
        "status": "DESCRIPTIVE_QUALITY_MATCHED_COST_FRONTIER",
        "references": list(references), "quality_floor": floor, "points": points,
        "quality_feasible_arms": feasible,
        "cheapest_quality_feasible_arms": [arm for arm in feasible if points[arm]["total_tokens"] == cheapest],
        "observed_pareto_arms": sorted(arm for arm in points if not points[arm]["dominated_by"]),
        "statistical_noninferiority_established": False, "paper_admitted": False,
        "claim_boundary": (
            "Complete matched-cohort point estimates only. Reference identities and both official metrics "
            "are prospectively fixed; no success margin is relaxed and no statistical or public-benchmark claim is admitted."
        ),
    }


def fixed_input_token_saving_bound(
    candidate_prompt_tokens: Mapping[str, int],
    reference_prompt_tokens: Mapping[str, int],
    *,
    output_cap: int,
    context_length: int,
) -> dict[str, Any]:
    if not candidate_prompt_tokens or set(candidate_prompt_tokens) != set(reference_prompt_tokens):
        raise ValueError("complete paired prompt-token coverage is required")
    if (type(output_cap) is not int or output_cap < 0
            or type(context_length) is not int or context_length <= output_cap):
        raise ValueError("fixed nonnegative output cap and positive context limit are required")
    for counts in (candidate_prompt_tokens, reference_prompt_tokens):
        if any(type(count) is not int or count <= 0 or count + output_cap > context_length
               for count in counts.values()):
            raise ValueError("exact positive prompt lengths must fit the full output reservation")
    saving = {
        task: reference_prompt_tokens[task] - candidate_prompt_tokens[task] - output_cap
        for task in sorted(candidate_prompt_tokens)
    }
    total_lower = sum(saving.values())
    return {
        "method": "FIXED_INPUT_PROMPT_DIFFERENCE_MINUS_OUTPUT_CAP",
        "tasks": len(saving), "output_cap": output_cap, "context_length": context_length,
        "per_task_saving_lower": saving, "minimum_task_saving_lower": min(saving.values()),
        "total_saving_lower": total_lower,
        "total_reduction_pct_lower": 100 * total_lower / sum(reference_prompt_tokens.values()),
        "every_frozen_task_strictly_cheaper": min(saving.values()) > 0,
        "frozen_cohort_strictly_cheaper": total_lower > 0,
        "conditions": (
            "One successful call per method/input, exact attested prompt tokenization, "
            "nonnegative baseline completion, and candidate completion no larger than the fixed cap. "
            "No omitted retries, tools, routing or extra inference calls."
        ),
        "claim_boundary": (
            "Deterministic token-budget statement for these frozen inputs only. "
            "No success guarantee or distribution-wide claim; optimization/acquisition costs are separate."
        ),
    }


def paired_finite_sample_lower_bound(
    candidate: Mapping[int, Mapping[str, float]],
    reference: Mapping[int, Mapping[str, float]],
    groups: Mapping[str, str],
    *,
    seeds: Sequence[int],
    score_upper_bound: float,
    family_size: int,
    alpha: float = 0.05,
    binary: bool = False,
) -> dict[str, Any]:
    if (not seeds or len(set(seeds)) != len(seeds) or set(candidate) != set(seeds)
            or set(reference) != set(seeds)):
        raise ValueError("all and only the declared seeds are required")
    if (not groups or any(not isinstance(group, str) or not group for group in groups.values())
            or type(family_size) is not int or family_size < 1
            or not math.isfinite(alpha) or not 0 < alpha < 1
            or type(score_upper_bound) not in (float, int)
            or not math.isfinite(score_upper_bound) or score_upper_bound <= 0):
        raise ValueError("nonempty groups, a fixed score bound, and a valid simultaneous family are required")
    if binary and score_upper_bound != 1:
        raise ValueError("binary official success uses an upper bound of one")
    sizes: dict[str, int] = defaultdict(int)
    differences: dict[str, float] = defaultdict(float)
    for group in groups.values():
        sizes[group] += 1
    for seed in seeds:
        if set(candidate[seed]) != set(groups) or set(reference[seed]) != set(groups):
            raise ValueError("paired seed/task coverage differs; missing results cannot be dropped")
        for outcomes in (candidate[seed], reference[seed]):
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= score_upper_bound
                   or (binary and value not in (0, 1)) for value in outcomes.values()):
                raise ValueError("observed scores violate the prospectively fixed score bound")
        for task in sorted(groups):
            differences[groups[task]] += candidate[seed][task] - reference[seed][task]
    count = len(groups)
    mean = math.fsum(differences.values()) / (count * len(seeds))
    bound_alpha = alpha / family_size
    weights_squared = math.fsum((size / count) ** 2 for size in sizes.values())
    lower = max(-score_upper_bound, mean - score_upper_bound * math.sqrt(
        2 * weights_squared * math.log(1 / bound_alpha),
    ))
    method = "WEIGHTED_QUESTION_GROUP_HOEFFDING_ONE_SIDED_BONFERRONI"
    gains = losses = None
    if binary and len(seeds) == 1 and all(size == 1 for size in sizes.values()):
        gains = sum(value > 0 for value in differences.values())
        losses = sum(value < 0 for value in differences.values())
        lower = clopper_pearson_lower(gains, count, alpha=bound_alpha / 2) - clopper_pearson_upper(
            losses, count, alpha=bound_alpha / 2,
        )
        method = "PAIRED_GAIN_MINUS_LOSS_CLOPPER_PEARSON_BONFERRONI"
    return {
        "method": method, "tasks": count, "seeds": list(seeds),
        "independent_question_groups": len(sizes),
        "task_seed_pairs": count * len(seeds), "task_seed_pairs_are_not_independent": True,
        "effective_question_groups": 1 / weights_squared,
        "observed_difference": mean, "one_sided_lower_bound": lower,
        "family_size": family_size, "familywise_alpha": alpha, "alpha_per_bound": bound_alpha,
        "score_upper_bound": score_upper_bound, "gains": gains, "losses": losses,
        "zero_loss_bound_passed": lower > 0,
        "assumptions": (
            "Frozen methods, sample size, endpoints, family and bounds before outcomes; independent question groups. "
            "Hoeffding targets the fixed task-weighted expected difference with outcome-independent group weights; "
            "exact binomial bounds additionally assume identically distributed independent task pairs. "
            "Seeds and related tasks remain within a group. No optional stopping."
        ),
        "claim_boundary": (
            "Numerical bound only: independence and pre-registration must be audited separately. "
            "Exposed development data cannot establish confirmation or a per-question guarantee."
        ),
    }


def _resample(
    differences: Sequence[float],
    sizes: Sequence[int],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> tuple[list[float], float, str]:
    if not any(differences):
        return [0.0, 0.0], 1.0, "EXACT_ZERO_PAIRED_DIFFERENCES"
    count = len(differences)
    estimates = []
    rng = random.Random(seed)
    if count <= 32:
        for _ in range(bootstrap_samples):
            selected = [rng.randrange(count) for _ in range(count)]
            estimates.append(
                math.fsum(differences[index] for index in selected)
                / sum(sizes[index] for index in selected)
            )
    else:
        import numpy as np

        generator = np.random.default_rng(seed)
        values, weights = np.asarray(differences), np.asarray(sizes)
        for offset in range(0, bootstrap_samples, 128):
            indices = generator.integers(
                0, count, size=(min(128, bootstrap_samples - offset), count),
            )
            estimates.extend((values[indices].sum(axis=1) / weights[indices].sum(axis=1)).tolist())
    nonzero = [value for value in differences if value]
    observed = abs(math.fsum(nonzero))
    if len(nonzero) <= 18:
        extreme = sum(
            abs(math.fsum(sign * value for sign, value in zip(signs, nonzero))) >= observed - 1e-12
            for signs in itertools.product((-1, 1), repeat=len(nonzero))
        )
        pvalue = extreme / (2 ** len(nonzero))
        method = "EXACT_TWO_SIDED_QUESTION_SIGN_RANDOMIZATION"
    else:
        import numpy as np

        generator = np.random.default_rng(seed + 1)
        values = np.asarray(nonzero)
        extreme = 0
        for offset in range(0, randomization_samples, 256):
            signs = generator.integers(
                0, 2, size=(min(256, randomization_samples - offset), len(nonzero)), dtype=np.int8,
            ) * 2 - 1
            extreme += int((np.abs((signs * values).sum(axis=1)) >= observed - 1e-12).sum())
        pvalue = (extreme + 1) / (randomization_samples + 1)
        method = "MONTE_CARLO_TWO_SIDED_QUESTION_SIGN_RANDOMIZATION_PLUS_ONE"
    estimates.sort()
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)], pvalue, method


def paired_quality_effect(
    candidate: Mapping[int, Mapping[str, float]],
    reference: Mapping[int, Mapping[str, float]],
    groups: Mapping[str, str],
    *,
    seeds: Sequence[int],
    bootstrap_samples: int = 10000,
    randomization_samples: int = 100000,
    statistics_seed: int = 20260913,
) -> dict[str, Any]:
    if (not seeds or len(set(seeds)) != len(seeds) or set(candidate) != set(seeds)
            or set(reference) != set(seeds)):
        raise ValueError("all and only the declared seeds are required")
    if (not groups or any(not isinstance(group, str) or not group for group in groups.values())
            or bootstrap_samples < 1 or randomization_samples < 1):
        raise ValueError("nonempty question groups and positive resample counts are required")
    differences: dict[str, float] = defaultdict(float)
    sizes: dict[str, int] = defaultdict(int)
    for group in groups.values():
        sizes[group] += 1
    per_seed = []
    for seed in seeds:
        if set(candidate[seed]) != set(groups) or set(reference[seed]) != set(groups):
            raise ValueError("paired seed/task coverage differs")
        for outcomes in (candidate[seed], reference[seed]):
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in outcomes.values()):
                raise ValueError("official task scores must be finite values in [0, 1]")
        for task in sorted(groups):
            differences[groups[task]] += candidate[seed][task] - reference[seed][task]
        left = math.fsum(candidate[seed].values()) / len(groups)
        right = math.fsum(reference[seed].values()) / len(groups)
        per_seed.append({"seed": seed, "candidate_mean": left, "reference_mean": right, "delta": left - right})
    ordered = sorted(sizes)
    interval, pvalue, method = _resample(
        [differences[group] / len(seeds) for group in ordered],
        [sizes[group] for group in ordered],
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        seed=statistics_seed,
    )
    delta = math.fsum(differences.values()) / (len(groups) * len(seeds))
    return {
        "tasks": len(groups),
        "independent_question_groups": len(sizes),
        "seeds": list(seeds),
        "task_seed_pairs": len(groups) * len(seeds),
        "task_seed_pairs_are_not_independent": True,
        "candidate_mean": math.fsum(row["candidate_mean"] for row in per_seed) / len(seeds),
        "reference_mean": math.fsum(row["reference_mean"] for row in per_seed) / len(seeds),
        "delta": delta,
        "paired_question_bootstrap_ci95": interval,
        "two_sided_question_randomization_p": pvalue,
        "randomization_method": method,
        "bootstrap_samples": bootstrap_samples,
        "randomization_samples_maximum": randomization_samples,
        "per_seed": per_seed,
        "inference_boundary": "Conditional on this complete cohort and frozen policies; prior exposure is reported separately and is not erased by seeds or folds.",
    }


def adjust_primary_family(
    contrasts: Mapping[str, dict[str, Any]],
    *,
    expected_size: int,
    alpha: float = 0.05,
) -> None:
    if len(contrasts) != expected_size or not 0 < alpha < 1:
        raise ValueError("the complete preregistered primary family is required")
    pvalues = {key: row["two_sided_question_randomization_p"] for key, row in contrasts.items()}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in pvalues.values()):
        raise ValueError("invalid primary p-value")
    adjusted = holm_adjust(pvalues)
    for key, row in contrasts.items():
        row["holm_adjusted_p"] = adjusted[key]
        row["positive_quality_contribution_supported"] = (
            row["delta"] > 0 and row["paired_question_bootstrap_ci95"][0] > 0
            and adjusted[key] <= alpha
        )
