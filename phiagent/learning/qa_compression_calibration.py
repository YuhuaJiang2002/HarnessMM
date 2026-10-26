"""Outcome-table validation and a prospectively fixed compression-ratio selection rule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from phiagent.learning.qa_interaction_compiler import Outcome
from phiagent.learning.qa_meta_rsi import digest


RATIOS = ("0.25", "0.5", "0.75", "1.0")


def select_compression_ratio(
    outcomes: Mapping[str, Mapping[int, Mapping[str, Outcome]]], *,
    task_ids: Sequence[str], repeat_seeds: Sequence[int],
) -> dict:
    tasks, seeds = tuple(task_ids), tuple(repeat_seeds)
    if (
        isinstance(task_ids, (str, bytes)) or not tasks or len(tasks) != len(set(tasks))
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(seeds) != 4 or len(seeds) != len(set(seeds))
        or any(type(seed) is not int for seed in seeds)
        or set(outcomes) != set(RATIOS)
    ):
        raise ValueError("ratio calibration requires all four ratios and four declared complete repeats")
    reports, observed = {}, {}
    for ratio in RATIOS:
        blocks = outcomes[ratio]
        if (
            set(blocks) != set(seeds) or any(type(seed) is not int for seed in blocks)
            or any(set(blocks[seed]) != set(tasks) for seed in seeds)
        ):
            raise ValueError("every ratio/repeat must cover the same complete calibration task inventory")
        if any(not isinstance(blocks[seed][task], Outcome) for seed in seeds for task in tasks):
            raise ValueError("only explicitly validated official per-question outcomes are accepted")
        for task in tasks:
            if len({blocks[seed][task].prompt_tokens for seed in seeds}) != 1:
                raise ValueError("a fixed ratio changed its native input across repeat seeds")
        reports[ratio] = {
            endpoint: math.fsum(
                math.fsum(getattr(blocks[seed][task], endpoint) for seed in seeds) / len(seeds)
                for task in tasks
            ) / len(tasks)
            for endpoint in ("exact_match", "f1")
        }
        reports[ratio]["total_tokens"] = sum(
            blocks[seed][task].total_tokens for seed in seeds for task in tasks
        )
        reports[ratio]["mean_total_tokens"] = reports[ratio]["total_tokens"] / (len(tasks) * len(seeds))
        observed[ratio] = {
            str(seed): {
                task: {
                    "exact_match": blocks[seed][task].exact_match, "f1": blocks[seed][task].f1,
                    "prompt_tokens": blocks[seed][task].prompt_tokens,
                    "completion_tokens": blocks[seed][task].completion_tokens,
                } for task in sorted(tasks)
            } for seed in seeds
        }
    reference = reports["1.0"]
    eligible = [
        ratio for ratio in RATIOS
        if all(reports[ratio][endpoint] >= reference[endpoint] for endpoint in ("exact_match", "f1"))
    ]
    selected = min(eligible, key=lambda ratio: (reports[ratio]["total_tokens"], RATIOS.index(ratio)))
    return {
        "schema_version": "phiagent-qa-compression-ratio-calibration/v1",
        "status": "CALIBRATION_RATIO_SELECTED_REQUIRES_OFFICIAL_EVALUATION",
        "selected_ratio": selected, "selected_passthrough": selected == "1.0",
        "ratios": list(RATIOS), "eligible_ratios": eligible, "reports": reports,
        "tasks": len(tasks), "repeat_seeds": list(seeds),
        "actor_requests": len(tasks) * len(seeds) * len(RATIOS),
        "all_ratio_actor_tokens": sum(report["total_tokens"] for report in reports.values()),
        "observations_sha256": digest(observed),
        "quality_scale": "official per-question EM/F1 in [0,1]; repeat mean inside task, then task mean",
        "reference": "fresh same-bank ratio 1.0, byte-identical full DELETE",
        "tie_rule": "minimum actual total tokens, then declared ratio order",
        "statistical_noninferiority_established": False,
        "complete_public_benchmark_result": False, "paper_admitted": False,
    }
