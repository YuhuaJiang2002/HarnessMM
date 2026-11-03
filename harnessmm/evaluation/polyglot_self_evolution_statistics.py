"""Paired task-cluster statistics for the same-model Polyglot comparison."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


def mean_success(outcomes: Mapping[int, Mapping[str, bool]], task_ids: Sequence[str]) -> float:
    values = [int(outcomes[seed][task_id]) for task_id in task_ids for seed in sorted(outcomes)]
    return sum(values) / len(values)


def task_cluster_differences(
    left: Mapping[int, Mapping[str, bool]],
    right: Mapping[int, Mapping[str, bool]],
    task_ids: Sequence[str],
) -> list[int]:
    if set(left) != set(right):
        raise ValueError("paired methods do not have the same seeds")
    return [
        sum(int(left[seed][task_id]) - int(right[seed][task_id]) for seed in sorted(left))
        for task_id in task_ids
    ]


def paired_cluster_randomization_pvalue(
    cluster_differences: Sequence[int],
    *,
    samples: int = 100_000,
    seed: int = 20260830,
) -> float:
    """One-sided paired sign-randomization test over task clusters."""

    nonzero = [int(value) for value in cluster_differences if value]
    observed = sum(nonzero)
    if observed <= 0:
        return 1.0
    if len(nonzero) <= 20:
        extreme = 0
        total = 1 << len(nonzero)
        for mask in range(total):
            statistic = sum(
                value if mask & (1 << index) else -value for index, value in enumerate(nonzero)
            )
            extreme += int(statistic >= observed)
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        statistic = sum(value if rng.getrandbits(1) else -value for value in nonzero)
        extreme += int(statistic >= observed)
    return (extreme + 1) / (samples + 1)


def paired_cluster_bootstrap_interval(
    cluster_differences: Sequence[int],
    *,
    seeds_per_task: int,
    cluster_sizes: Sequence[int] | None = None,
    samples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260830,
) -> tuple[float, float]:
    if not cluster_differences or seeds_per_task < 1:
        raise ValueError("cluster bootstrap inputs are empty")
    if cluster_sizes is not None and (
        len(cluster_sizes) != len(cluster_differences)
        or any(type(size) is not int or size < 1 for size in cluster_sizes)
    ):
        raise ValueError("cluster sizes must be matched positive integer task counts")
    rng = random.Random(seed)
    count = len(cluster_differences)
    estimates = []
    denominator = count * seeds_per_task
    for _ in range(samples):
        if cluster_sizes is None:
            estimates.append(
                sum(cluster_differences[rng.randrange(count)] for _ in range(count)) / denominator
            )
        else:
            selected = [rng.randrange(count) for _ in range(count)]
            estimates.append(
                sum(cluster_differences[index] for index in selected)
                / (sum(cluster_sizes[index] for index in selected) * seeds_per_task)
            )
    estimates.sort()
    lower_index = max(0, math.floor((alpha / 2) * samples))
    upper_index = min(samples - 1, math.ceil((1 - alpha / 2) * samples) - 1)
    return estimates[lower_index], estimates[upper_index]


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return adjusted


__all__ = [
    "holm_adjust",
    "mean_success",
    "paired_cluster_bootstrap_interval",
    "paired_cluster_randomization_pvalue",
    "task_cluster_differences",
]
