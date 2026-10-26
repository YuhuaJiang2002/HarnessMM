"""Fabricated aggregate-only tests, without datasets, benchmarks or evaluators."""

from dataclasses import replace

import pytest

from phiagent.evaluation.dual_quality_cost import (
    VerifiedAggregate, benchmark_dual_gate, campaign_dual_gate,
)


def runs():
    base = VerifiedAggregate("auto", "synthetic", "a" * 64, "b" * 64, "c" * 64,
                             100, {"success": 70, "error": 3}, 800, 200, 0, 0)
    return [
        base,
        replace(base, method="current", metrics={"success": 72, "error": 2}),
        replace(base, method="new", metrics={"success": 73, "error": 1}, prompt_tokens=700),
    ]


def gate(values=None, **kwargs):
    return benchmark_dual_gate(
        runs() if values is None else values, candidate="new", incumbent="current",
        autosaddler="auto", metric_directions={"success": "higher", "error": "lower"},
        expected_quality_references=("auto", "current"),
        full_official_protocol=False, independent_confirmation=False, **kwargs,
    )


def test_strict_dual_gate_keeps_point_and_public_admission_separate():
    result = gate()
    assert result["observed_point_gate_passed"]
    assert result["autosaddler_token_ratio"] == 0.9
    assert result["direction_adjusted_quality_differences"]["current"]["error"] == 1
    assert not result["admission_prerequisites_passed"]


def test_overheads_cannot_be_hidden_from_token_comparison():
    values = runs()
    values[-1] = replace(values[-1], overhead_tokens=100)
    assert gate(values)["tokens_strictly_reduced"] is False
    values[-1] = replace(values[-1], unknown_cost_requests=1)
    assert gate(values)["autosaddler_token_ratio"] is None
    assert not gate(values)["observed_point_gate_passed"]


def test_quality_ties_are_not_improvements():
    values = runs()
    values[-1] = replace(values[-1], metrics=values[-2].metrics)
    assert not gate(values)["quality_strictly_improved"]


@pytest.mark.parametrize("field,value", [
    ("tasks", 99), ("cohort_sha256", "d" * 64), ("matched_protocol_sha256", "e" * 64),
    ("benchmark", "different"), ("metrics", {"success": 70}),
])
def test_mismatched_evidence_is_not_a_valid_comparison(field, value):
    values = runs()
    values[-1] = replace(values[-1], **{field: value})
    with pytest.raises(ValueError):
        gate(values)


def test_complete_declared_comparator_set_is_mandatory():
    with pytest.raises(ValueError):
        gate(runs()[1:])
    with pytest.raises(ValueError):
        gate(runs() + [replace(runs()[0], method="omitted_control")])


def test_uncertainty_must_cover_both_quality_and_complete_cost():
    bounds = {"auto": {"success": 0.1, "error": 0.1, "tokens_saved": 1},
              "current": {"success": 0.1, "error": 0.1}}
    assert gate(simultaneous_lower_bounds=bounds)["simultaneous_bounds_passed"]
    bounds["auto"]["tokens_saved"] = 0
    assert not gate(simultaneous_lower_bounds=bounds)["simultaneous_bounds_passed"]
    bounds["auto"].pop("tokens_saved")
    with pytest.raises(ValueError):
        gate(simultaneous_lower_bounds=bounds)


def test_missing_benchmarks_cannot_be_replaced_by_favorable_average():
    result = gate()
    campaign = campaign_dual_gate({"synthetic": result}, expected_benchmarks=("synthetic", "missing"))
    assert campaign["missing_benchmarks"] == ["missing"]
    assert not campaign["all_benchmark_point_gates_passed"]
    assert not campaign["cross_benchmark_averaging_used"]
    complete = campaign_dual_gate({"synthetic": result}, expected_benchmarks=("synthetic",))
    assert complete["all_benchmark_point_gates_passed"]
    assert not complete["all_admission_prerequisites_passed"]
