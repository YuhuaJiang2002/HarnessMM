"""Independent per-benchmark quality/cost gates over verified aggregate evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


def _sha(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("a lower-case SHA-256 evidence binding is required")


@dataclass(frozen=True, slots=True)
class VerifiedAggregate:
    method: str
    benchmark: str
    cohort_sha256: str
    matched_protocol_sha256: str
    verification_sha256: str
    tasks: int
    metrics: Mapping[str, float]
    prompt_tokens: int
    completion_tokens: int
    overhead_tokens: int
    unknown_cost_requests: int

    def __post_init__(self) -> None:
        if not self.method or not self.benchmark:
            raise ValueError("benchmark and method identities are required")
        for digest in (self.cohort_sha256, self.matched_protocol_sha256, self.verification_sha256):
            _sha(digest)
        for value in (
            self.tasks, self.prompt_tokens, self.completion_tokens,
            self.overhead_tokens, self.unknown_cost_requests,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("task and complete token counts must be nonnegative integers")
        if self.tasks == 0 or not self.metrics:
            raise ValueError("a complete nonempty metric/task cohort is required")
        if any(not name or type(value) not in (int, float) or not math.isfinite(value)
               for name, value in self.metrics.items()):
            raise ValueError("official metrics must have names and finite real values")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.overhead_tokens


def benchmark_dual_gate(
    runs: Sequence[VerifiedAggregate],
    *,
    candidate: str,
    incumbent: str,
    autosaddler: str,
    metric_directions: Mapping[str, str],
    expected_quality_references: Sequence[str],
    full_official_protocol: bool,
    independent_confirmation: bool,
    simultaneous_lower_bounds: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    if not runs or not metric_directions or any(
        direction not in {"higher", "lower"} for direction in metric_directions.values()
    ):
        raise ValueError("verified runs and every official metric direction are required")
    if type(full_official_protocol) is not bool or type(independent_confirmation) is not bool:
        raise ValueError("evaluation scope and confirmation must be explicit boolean flags")
    by_name = {run.method: run for run in runs}
    references = tuple(expected_quality_references)
    if (
        len(by_name) != len(runs) or len(set(references)) != len(references)
        or set(references) != set(by_name) - {candidate}
        or candidate not in by_name or incumbent not in references or autosaddler not in references
    ):
        raise ValueError("all declared controls, incumbent and AutoSaddler must be present exactly once")
    chosen = by_name[candidate]
    for run in runs:
        if (
            run.benchmark != chosen.benchmark or run.tasks != chosen.tasks
            or run.cohort_sha256 != chosen.cohort_sha256
            or run.matched_protocol_sha256 != chosen.matched_protocol_sha256
            or set(run.metrics) != set(metric_directions)
        ):
            raise ValueError("cohort, task count, model/tool/scoring protocol or official metrics differ")
    quality_differences = {
        reference: {
            metric: (chosen.metrics[metric] - by_name[reference].metrics[metric])
            * (1 if direction == "higher" else -1)
            for metric, direction in metric_directions.items()
        }
        for reference in references
    }
    quality = all(delta > 0 for row in quality_differences.values() for delta in row.values())
    cost_known = all(run.unknown_cost_requests == 0 for run in runs)
    denominator = by_name[autosaddler].total_tokens
    if denominator <= 0:
        raise ValueError("AutoSaddler must have a measured positive complete token denominator")
    cost = chosen.total_tokens < denominator if cost_known else None
    ratio = chosen.total_tokens / denominator if cost_known else None
    statistical = None
    if simultaneous_lower_bounds is not None:
        expected = {reference: set(metric_directions) for reference in references}
        expected[autosaddler] = expected[autosaddler] | {"tokens_saved"}
        if (
            set(simultaneous_lower_bounds) != set(expected)
            or any(set(simultaneous_lower_bounds[name]) != keys for name, keys in expected.items())
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for row in simultaneous_lower_bounds.values() for value in row.values())
        ):
            raise ValueError("the complete simultaneous quality and AutoSaddler cost bounds are required")
        statistical = all(value > 0 for row in simultaneous_lower_bounds.values() for value in row.values())
    point = quality and cost is True
    return {
        "benchmark": chosen.benchmark,
        "candidate": candidate, "incumbent": incumbent, "cost_reference": autosaddler,
        "tasks": chosen.tasks, "cohort_sha256": chosen.cohort_sha256,
        "matched_protocol_sha256": chosen.matched_protocol_sha256,
        "quality_references": list(references),
        "direction_adjusted_quality_differences": quality_differences,
        "quality_strictly_improved": quality,
        "complete_cost_known": cost_known,
        "tokens_strictly_reduced": cost,
        "autosaddler_token_ratio": ratio,
        "observed_point_gate_passed": point,
        "simultaneous_bounds_passed": statistical,
        "full_official_protocol": full_official_protocol,
        "independent_confirmation": independent_confirmation,
        "admission_prerequisites_passed": (
            point and statistical is True and full_official_protocol and independent_confirmation
        ),
        "verification_sha256": {run.method: run.verification_sha256 for run in runs},
        "unknown_cost_is_not_zero": True,
    }


def campaign_dual_gate(
    results: Mapping[str, Mapping], *, expected_benchmarks: Sequence[str],
) -> dict:
    expected = tuple(expected_benchmarks)
    if not expected or len(expected) != len(set(expected)) or set(results) - set(expected):
        raise ValueError("a unique prospective benchmark inventory is required")
    for name, result in results.items():
        if result["benchmark"] != name:
            raise ValueError("a benchmark result cannot be relabeled as another")
    missing = [name for name in expected if name not in results]
    failed = [name for name in expected if name in results and
              results[name]["observed_point_gate_passed"] is not True]
    return {
        "expected_benchmarks": list(expected), "missing_benchmarks": missing,
        "failed_point_benchmarks": failed,
        "all_benchmark_point_gates_passed": not missing and not failed,
        "all_admission_prerequisites_passed": (
            not missing and all(results[name]["admission_prerequisites_passed"] is True for name in expected)
        ),
        "cross_benchmark_averaging_used": False,
    }
