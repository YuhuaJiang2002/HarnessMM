"""Scenario-level GAIA2 admission gates for the public 20/20/60 re-partition."""
from __future__ import annotations

import hashlib
import json

from phiagent.harness.benchmark_partition import build_partition, validate_partition


METHODS = ("native", "skillopt", "autosaddler", "chisel")
REPETITIONS = (1, 2, 3)
COUNTS = {"train": 32, "validation": 32, "test": 96}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def partition_gate(manifest):
    validate_partition(manifest)
    if (manifest["benchmark"] != "gaia2-mini" or manifest["split_seed"] != 20260928
            or manifest["counts"] != COUNTS or manifest["independent_unit"] != "scenario"):
        raise ValueError("Expected the exact 160-scenario GAIA2 32/32/96 partition")
    rebuilt = build_partition(manifest["records"], benchmark="gaia2-mini", seed=20260928)
    if rebuilt["splits"] != manifest["splits"]:
        raise ValueError("Partition is not the frozen outcome-independent hash rank")
    return {"status": "PASS", "counts": COUNTS, "repetitions": list(REPETITIONS),
            "final_executions_per_method": 288, "public_resplit_not_unseen": True}


def items_gate(items, manifest, split, *, complete=True):
    partition_gate(manifest)
    if split not in COUNTS:
        raise ValueError("Unknown split")
    ids = [item["id"] for item in items]
    allowed = set(manifest["splits"][split])
    if len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError("Duplicate or cross-split scenario")
    if complete and set(ids) != allowed:
        raise ValueError("Incomplete scenario population")
    records = {item["id"]: item for item in manifest["records"]}
    for item in items:
        if item["sha256"] != records[item["id"]]["payload_sha256"]:
            raise ValueError("Scenario payload differs from the partition")


def native_population_gate(rows, ids):
    expected = {(ident, rep) for ident in ids for rep in REPETITIONS}
    actual = [(row["base_scenario_id"], row["run_number"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Native executions must cover each scenario and all three repetitions once")
    if any(row.get("has_exception") or type(row.get("success_bool")) is not bool for row in rows):
        raise ValueError("Unresolved native results cannot enter selection")
    return {"executions": len(rows), "successes": sum(row["success_bool"] for row in rows)}


def choose_frozen_candidate(candidates, evaluations, validation_ids):
    """Only scalar native quality/cost selects among already-frozen train candidates."""
    if not candidates or set(candidates) != set(evaluations):
        raise ValueError("Every frozen candidate needs a complete validation population")
    scalars = {}
    for ident, candidate in candidates.items():
        if candidate["skill_sha256"] != hashlib.sha256(candidate["skill"].encode()).hexdigest():
            raise ValueError("Candidate changed after training freeze")
        evaluation = evaluations[ident]
        result = native_population_gate(evaluation["rows"], validation_ids)
        scalars[ident] = {**result, "actor_tokens": evaluation["actor_tokens"]}
    return choose_scalar_candidate(set(candidates), scalars, len(validation_ids) * len(REPETITIONS))


def choose_scalar_candidate(candidate_ids, evaluations, expected_executions):
    """Rank independently audited complete scalars with the same native tie-break."""
    if not candidate_ids or set(candidate_ids) != set(evaluations):
        raise ValueError("Every frozen candidate needs a complete validation population")
    if type(expected_executions) is not int or expected_executions <= 0:
        raise ValueError("A positive complete population is required")
    scores = []
    for ident, evaluation in evaluations.items():
        if type(evaluation["actor_tokens"]) is not int or evaluation["actor_tokens"] < 0:
            raise ValueError("Selection cost must be accounted")
        if (type(evaluation["executions"]) is not int
                or evaluation["executions"] != expected_executions
                or type(evaluation["successes"]) is not int
                or not 0 <= evaluation["successes"] <= expected_executions):
            raise ValueError("Selection requires valid complete native success counts")
        scores.append((-evaluation["successes"], evaluation["actor_tokens"], ident))
    return min(scores)[2]


def test_gate(manifest, selections):
    partition_gate(manifest)
    if set(selections) != set(METHODS):
        raise ValueError("All four methods must freeze before any test access")
    for method, selection in selections.items():
        if (selection["method"] != method or selection["partition_sha256"] != digest(manifest)
                or selection.get("test_accessed") is not False
                or not selection.get("validation_complete")
                or not selection.get("training_complete")
                or not selection.get("skill_sha256")):
            raise ValueError("Unverified method freeze")
    return {"status": "PASS", "partition_sha256": digest(manifest),
            "selection_sha256": {key: digest(value) for key, value in selections.items()}}
