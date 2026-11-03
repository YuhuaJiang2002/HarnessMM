"""Validation helpers for the residual Meta-Harness V2 experiment design."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist
from typing import Any, Mapping


PROTOCOL_SCHEMA_VERSION = "phiagent-residual-meta-harness-experiment/v1"
EXPECTED_BENCHMARKS = frozenset(
    {"SearchQA", "Terminal-Bench", "VAB-WebArena-Lite"}
)
REQUIRED_CONTROLS = frozenset(
    {
        "static_incumbent",
        "matched_random_override",
        "direct_four_way_classifier",
        "residual_without_cross_fitting",
        "residual_without_hierarchical_shrinkage",
        "residual_without_safety_gate",
        "runtime_llm_controller",
    }
)
MINIMUM_SPLITS = {
    "SearchQA": {
        "discovery_tasks": 8_000,
        "calibration_tasks": 2_000,
        "confirmation_tasks": 5_000,
    },
    "Terminal-Bench": {
        "discovery_tasks": 256,
        "calibration_tasks": 96,
        "confirmation_tasks": 80,
    },
    "VAB-WebArena-Lite": {
        "discovery_tasks": 480,
        "calibration_tasks": 80,
        "confirmation_tasks": 80,
    },
}


class ProtocolValidationError(ValueError):
    """Raised when a residual Meta-Harness experiment design is unsafe."""


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def paired_mcnemar_sample_size(
    *,
    effect: float,
    discordance_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    family_size: int = 1,
) -> int:
    """Return a conservative normal-approximation task count for paired success.

    The calculation is a planning aid, not the final exact test. ``effect`` is
    the expected paired success-rate delta and ``discordance_rate`` is the
    expected fraction of tasks on which the two methods differ.
    """

    if (
        not all(
            math.isfinite(value)
            for value in (effect, discordance_rate, alpha, power)
        )
        or not 0 < effect < discordance_rate <= 1
        or not 0 < alpha < 1
        or not 0.5 < power < 1
        or isinstance(family_size, bool)
        or not isinstance(family_size, int)
        or family_size < 1
    ):
        raise ValueError("paired sample-size inputs are invalid")
    adjusted_alpha = alpha / family_size
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - adjusted_alpha)
    z_power = normal.inv_cdf(power)
    numerator = (
        z_alpha * math.sqrt(discordance_rate)
        + z_power * math.sqrt(discordance_rate - effect * effect)
    )
    return math.ceil((numerator / effect) ** 2)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ProtocolValidationError(
            f"{label} fields differ: "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def validate_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a self-hashed, pre-run residual Meta-Harness design."""

    if not isinstance(value, Mapping):
        raise ProtocolValidationError("protocol must be an object")
    top_keys = frozenset(
        {
            "schema_version",
            "status",
            "objective",
            "architecture",
            "cross_fitting",
            "benchmarks",
            "controls",
            "statistics",
            "promotion_gates",
            "execution",
            "literature",
            "canonical_sha256",
        }
    )
    _require_exact_keys(value, top_keys, "protocol")
    unsigned = dict(value)
    declared = unsigned.pop("canonical_sha256")
    if value["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolValidationError("unsupported protocol schema")
    if value["status"] != "DESIGN_READY_TASK_INVENTORY_PENDING":
        raise ProtocolValidationError("protocol status is not pre-run design-ready")
    if declared != canonical_hash(unsigned):
        raise ProtocolValidationError("protocol canonical_sha256 mismatch")

    architecture = _require_mapping(value["architecture"], "architecture")
    if (
        architecture.get("policy_class") != "residual_uplift_with_abstention"
        or architecture.get("static_incumbent_required") is not True
        or architecture.get("maximum_features") != 8
        or architecture.get("maximum_override_heads_per_policy") != 1
        or architecture.get("maximum_overrides_per_task") != 1
        or architecture.get("router_model_calls") != 0
        or architecture.get("maximum_extra_actor_calls") != 0
        or architecture.get("online_repair_calls") != 0
    ):
        raise ProtocolValidationError("architecture violates residual zero-call design")

    cross_fitting = _require_mapping(value["cross_fitting"], "cross_fitting")
    if (
        cross_fitting.get("folds") != 5
        or cross_fitting.get("group_key") != "task_id"
        or cross_fitting.get("all_seeds_and_stages_stay_in_task_fold") is not True
        or cross_fitting.get("leave_one_family_out_stress_test") is not True
        or cross_fitting.get("task_ids_as_features") is not False
        or cross_fitting.get("trajectory_steps_as_independent_units") is not False
    ):
        raise ProtocolValidationError("cross-fitting does not preserve task independence")

    benchmarks = _require_mapping(value["benchmarks"], "benchmarks")
    if set(benchmarks) != EXPECTED_BENCHMARKS:
        raise ProtocolValidationError("benchmark registry differs")
    split_summary = {}
    for benchmark, minima in MINIMUM_SPLITS.items():
        design = _require_mapping(benchmarks[benchmark], f"benchmarks.{benchmark}")
        if (
            design.get("independent_unit") != "complete_task"
            or design.get("seeds_nested_within_task") is not True
            or design.get("split_task_ids_pairwise_disjoint") is not True
            or design.get("gold_or_evaluator_visible_to_router") is not False
        ):
            raise ProtocolValidationError(
                f"{benchmark} does not preserve the information boundary"
            )
        splits = _require_mapping(design.get("splits"), f"{benchmark}.splits")
        for name, minimum in minima.items():
            count = splits.get(name)
            if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
                raise ProtocolValidationError(
                    f"{benchmark}.{name} is below the frozen minimum {minimum}"
                )
        if benchmark == "VAB-WebArena-Lite":
            target_count = (
                int(splits["calibration_tasks"])
                + int(splits["confirmation_tasks"])
                + int(splits.get("target_reserve_tasks", 0))
            )
            if target_count > 165:
                raise ProtocolValidationError(
                    "VAB target calibration/confirmation/reserve exceeds 165 tasks"
                )
            if design.get("exclude_vab_tasks_from_source_discovery") is not True:
                raise ProtocolValidationError(
                    "WebArena source discovery may overlap the VAB target"
                )
        split_summary[benchmark] = {
            name: int(splits[name]) for name in minima
        }

    controls = value["controls"]
    if not isinstance(controls, list) or set(controls) != REQUIRED_CONTROLS:
        raise ProtocolValidationError("required matched controls are incomplete")

    statistics = _require_mapping(value["statistics"], "statistics")
    if (
        statistics.get("primary_unit") != "task"
        or statistics.get("paired_official_success") is not True
        or statistics.get("task_clustered_uncertainty") is not True
        or statistics.get("holm_family_size") != 3
        or statistics.get("report_effect_size_and_interval") is not True
        or statistics.get("report_override_coverage") is not True
        or statistics.get("report_all_in_tokens_calls_latency") is not True
    ):
        raise ProtocolValidationError("statistical contract is incomplete")

    gates = _require_mapping(value["promotion_gates"], "promotion_gates")
    if (
        gates.get("router_and_extra_actor_calls_equal_zero") is not True
        or gates.get("confirmation_frozen_before_outcomes") is not True
        or gates.get("default_on_insufficient_support_or_shift") is not True
        or gates.get("subgroup_floor_required") is not True
        or gates.get("fresh_confirmation_required") is not True
    ):
        raise ProtocolValidationError("promotion gates are incomplete")

    execution = _require_mapping(value["execution"], "execution")
    if (
        execution.get("remote_only") is not True
        or execution.get("prefer_h200") is not True
        or execution.get("new_experiment_directory_per_run") is not True
        or execution.get("gpu_selection_saved") is not True
        or execution.get("frozen_commands_and_provenance_saved") is not True
    ):
        raise ProtocolValidationError("remote execution contract is incomplete")

    literature = value["literature"]
    if not isinstance(literature, list) or len(literature) < 6 or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("title"), str)
        or not isinstance(row.get("url"), str)
        or not str(row["url"]).startswith("https://")
        for row in literature
    ):
        raise ProtocolValidationError("literature provenance is incomplete")

    return {
        "status": "PASS",
        "canonical_sha256": declared,
        "benchmarks": split_summary,
        "required_controls": sorted(REQUIRED_CONTROLS),
        "router_model_calls": 0,
        "maximum_extra_actor_calls": 0,
    }


__all__ = [
    "EXPECTED_BENCHMARKS",
    "MINIMUM_SPLITS",
    "PROTOCOL_SCHEMA_VERSION",
    "REQUIRED_CONTROLS",
    "ProtocolValidationError",
    "canonical_hash",
    "paired_mcnemar_sample_size",
    "validate_protocol",
]
