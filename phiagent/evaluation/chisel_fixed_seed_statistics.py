"""Matched fixed-seed component statistics with whole-floorplan resampling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from phiagent.evaluation.polyglot_self_evolution_statistics import (
    paired_cluster_bootstrap_interval,
    paired_cluster_randomization_pvalue,
)


def fixed_seed_effect(
    candidate: Mapping[int, Mapping[str, int]],
    reference: Mapping[int, Mapping[str, int]],
    groups: Mapping[str, str],
    *,
    declared_seeds: Sequence[int] = (20260911, 20260912, 20260913),
    expected_tasks: int = 134,
    bootstrap_samples: int = 10000,
    randomization_samples: int = 100000,
    statistics_seed: int = 20260911,
) -> dict[str, object]:
    seeds = tuple(declared_seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or set(candidate) != set(seeds) or set(reference) != set(seeds)):
        raise ValueError("every and only the prospectively declared seed must be present")
    tasks = set(groups)
    if len(tasks) != expected_tasks or any(not isinstance(group, str) or not group for group in groups.values()):
        raise ValueError("complete task identities and nonempty floorplan groups are required")
    differences: dict[str, int] = defaultdict(int)
    sizes: dict[str, int] = defaultdict(int)
    per_seed = []
    gains = losses = 0
    for task in tasks:
        sizes[groups[task]] += 1
    for seed in seeds:
        if set(candidate[seed]) != tasks or set(reference[seed]) != tasks:
            raise ValueError("seed/method task coverage differs; never fill a missing score with zero")
        for outcomes in (candidate[seed], reference[seed]):
            if any(type(value) is not int or value not in (0, 1) for value in outcomes.values()):
                raise ValueError("only verified native binary outcomes are permitted")
        for task in sorted(tasks):
            delta = candidate[seed][task] - reference[seed][task]
            differences[groups[task]] += delta
            gains += delta == 1
            losses += delta == -1
        left, right = sum(candidate[seed].values()), sum(reference[seed].values())
        per_seed.append({
            "seed": seed, "tasks": len(tasks), "candidate_successes": left,
            "reference_successes": right, "delta_successes": left - right,
        })
    ordered_groups = sorted(sizes)
    cluster_differences = [differences[group] for group in ordered_groups]
    interval = paired_cluster_bootstrap_interval(
        cluster_differences, cluster_sizes=[sizes[group] for group in ordered_groups],
        seeds_per_task=len(seeds), samples=bootstrap_samples, seed=statistics_seed,
    )
    pvalue = paired_cluster_randomization_pvalue(
        cluster_differences, samples=randomization_samples, seed=statistics_seed,
    )
    oriented = cluster_differences if sum(cluster_differences) >= 0 else [-value for value in cluster_differences]
    two_sided = min(1.0, 2 * paired_cluster_randomization_pvalue(
        oriented, samples=randomization_samples, seed=statistics_seed,
    ))
    denominator = len(tasks) * len(seeds)
    candidate_successes = sum(sum(candidate[seed].values()) for seed in seeds)
    reference_successes = sum(sum(reference[seed].values()) for seed in seeds)
    return {
        "tasks": len(tasks), "seeds": list(seeds), "independent_floorplan_clusters": len(sizes),
        "observed_task_seed_pairs": denominator, "task_seed_pairs_are_not_independent": True,
        "candidate_successes": candidate_successes, "reference_successes": reference_successes,
        "candidate_official_mean_success": candidate_successes / denominator,
        "reference_official_mean_success": reference_successes / denominator,
        "delta_rate": (candidate_successes - reference_successes) / denominator,
        "candidate_only_task_seed_pairs_descriptive": gains,
        "reference_only_task_seed_pairs_descriptive": losses,
        "descriptive_floorplan_bootstrap_ci95": list(interval),
        "one_sided_floorplan_sign_randomization_p": pvalue,
        "two_sided_floorplan_sign_randomization_p": two_sided,
        "per_seed": per_seed,
        "observed_positive_mean_difference": candidate_successes > reference_successes,
        "floorplan_interval_excludes_no_gain": interval[0] > 0,
        "bootstrap_samples": bootstrap_samples, "randomization_samples_maximum": randomization_samples,
        "statistics_seed": statistics_seed,
        "inference_boundary": "Historically exposed cohort; all seeds remain inside the same floorplan clusters. Not fresh confirmation or a global SOTA comparison.",
    }
