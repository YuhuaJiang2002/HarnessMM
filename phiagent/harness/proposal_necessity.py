"""Audits whether repeated model proposals are necessary for a benchmark result.

This module is deliberately dependency-free.  It distinguishes a program that
reads state while executing from a model that changes its proposed program when
the generation input changes.  Those are different claims: a single frozen
program can be state-reactive at runtime while making repeated online model
calls unnecessary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


SCHEMA_VERSION = "1.0.0"


def canonical_sha256(value: object) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_task_replay(
    records: Sequence[Mapping[str, Any]],
    *,
    frozen_artifact: object,
    generation_input_ids: Sequence[str],
    environment_instance_ids: Sequence[str],
) -> dict[str, Any]:
    """Summarize proposal variation and exact frozen-artifact replay evidence.

    A replay outcome is counted only when the previously frozen artifact is
    exactly identical to the artifact that was actually executed in that trial.
    This is an artifact-identity observation, not an unexecuted simulator claim.
    """

    if not records:
        raise ValueError("at least one record is required")
    if len(generation_input_ids) != len(records):
        raise ValueError("generation_input_ids must align with records")
    if len(environment_instance_ids) != len(records):
        raise ValueError("environment_instance_ids must align with records")

    artifact_hashes = [canonical_sha256(record["artifact"]) for record in records]
    frozen_hash = canonical_sha256(frozen_artifact)
    source_matches = [artifact_hash == frozen_hash for artifact_hash in artifact_hashes]
    exact_replay_successes = sum(
        bool(record["success"]) for record, matched in zip(records, source_matches) if matched
    )
    actual_successes = sum(bool(record["success"]) for record in records)
    trials = len(records)
    unique_artifacts = len(set(artifact_hashes))
    unique_generation_inputs = len(set(generation_input_ids))
    unique_environment_instances = len(set(environment_instance_ids))
    model_calls = sum(int(record.get("model_calls", 0)) for record in records)
    total_tokens = sum(int(record.get("total_tokens", 0)) for record in records)
    model_latency_seconds = sum(
        float(record.get("model_latency_seconds", 0.0)) for record in records
    )
    all_match = all(source_matches)

    return {
        "trials": trials,
        "unique_generation_inputs": unique_generation_inputs,
        "unique_environment_instances": unique_environment_instances,
        "unique_proposed_artifacts": unique_artifacts,
        "proposal_variation_fraction": (
            0.0 if trials <= 1 else (unique_artifacts - 1) / (trials - 1)
        ),
        "generation_input_varies": unique_generation_inputs > 1,
        "model_proposal_varies": unique_artifacts > 1,
        "frozen_artifact_sha256": frozen_hash,
        "evaluation_artifact_sha256": sorted(set(artifact_hashes)),
        "frozen_artifact_matches_all_evaluation_trials": all_match,
        "exact_replay_trials": sum(source_matches),
        "actual_successes": actual_successes,
        "exact_replay_observed_successes": exact_replay_successes,
        "exact_replay_observed_success_rate": (
            exact_replay_successes / trials if all_match else None
        ),
        "online_model_cost": {
            "model_calls": model_calls,
            "total_tokens": total_tokens,
            "model_latency_seconds": model_latency_seconds,
        },
        "exact_frozen_replay_cost": {
            "model_calls": 0,
            "total_tokens": 0,
            "model_latency_seconds": 0.0,
        },
        "online_model_necessary_for_recorded_artifact": not all_match,
        "claim_boundary": (
            "Exact replay success is copied only for trials whose actually executed artifact "
            "is byte/canonical identical to the previously frozen artifact. This tests online "
            "proposal necessity, not whether the program itself reads state at execution time."
        ),
    }


def necessity_decision(
    suites: Sequence[Mapping[str, Any]],
    *,
    minimum_suites: int,
    minimum_task_groups: int,
    minimum_environment_instances_per_group: int,
    minimum_varying_proposal_groups: int,
    require_no_exact_frozen_replay_group: bool,
) -> dict[str, Any]:
    """Apply hard gates for an online model-planning necessity claim."""

    groups = [group for suite in suites for group in suite["task_groups"]]
    varying_groups = [group for group in groups if group["model_proposal_varies"]]
    exact_replay_groups = [
        group
        for group in groups
        if group["frozen_artifact_matches_all_evaluation_trials"]
    ]
    gates = {
        "minimum_independent_suites": len(suites) >= minimum_suites,
        "minimum_task_groups": len(groups) >= minimum_task_groups,
        "minimum_environment_instances_per_group": all(
            group["unique_environment_instances"]
            >= minimum_environment_instances_per_group
            for group in groups
        ),
        "minimum_varying_proposal_groups": len(varying_groups)
        >= minimum_varying_proposal_groups,
        "no_exact_frozen_replay_group": (
            not exact_replay_groups if require_no_exact_frozen_replay_group else True
        ),
    }
    promote = all(gates.values())
    return {
        "promote_online_model_planning_claim": promote,
        "reason": "all_necessity_gates_pass" if promote else "hard_gate_failed",
        "requirements": {
            "minimum_independent_suites": minimum_suites,
            "minimum_task_groups": minimum_task_groups,
            "minimum_environment_instances_per_group": (
                minimum_environment_instances_per_group
            ),
            "minimum_varying_proposal_groups": minimum_varying_proposal_groups,
            "require_no_exact_frozen_replay_group": (
                require_no_exact_frozen_replay_group
            ),
        },
        "gates": gates,
        "varying_proposal_groups": [
            f"{group['suite']}:{group['task_family']}" for group in varying_groups
        ],
        "exact_frozen_replay_groups": [
            f"{group['suite']}:{group['task_family']}" for group in exact_replay_groups
        ],
    }


__all__ = [
    "SCHEMA_VERSION",
    "canonical_sha256",
    "necessity_decision",
    "summarize_task_replay",
]
