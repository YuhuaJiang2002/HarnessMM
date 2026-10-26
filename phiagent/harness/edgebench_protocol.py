"""Pure partition, native-score and usage contracts for the public EdgeBench release."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re

from integrations.deepseek_harness.runner import normalize_usage
from phiagent.harness.benchmark_partition import build_partition, canonical_sha256


PUBLIC_TASK_COUNT = 51
SOURCE_REVISION = "6e4e5f4aa087a9121aa09a6059e5484e5102c6ff"
DATASET_REVISION = "6cc5a7f1b3288ce52484ad828177b8cc86b05b75"


def public_task_identity(query: str, cwd: str, work_image_tag: str) -> str:
    if any(not isinstance(value, str) or not value for value in (query, cwd, work_image_tag)):
        raise ValueError("Public identity requires the native query, workspace and work-image binding")
    return canonical_sha256({"agent_query": query, "cwd": cwd, "work_image_tag": work_image_tag})


def grouped_partition(records: Sequence[Mapping], groups: Mapping[str, Sequence[str]]) -> dict:
    for row in records:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"id", "payload_sha256", "public_task_sha256"}
            or not isinstance(row["id"], str) or not row["id"]
            or any(
                not isinstance(row[name], str) or not re.fullmatch(r"[0-9a-f]{64}", row[name])
                for name in ("payload_sha256", "public_task_sha256")
            )
        ):
            raise ValueError("Partition records require only public IDs and source SHA256 bindings")
    by_id = {row["id"]: dict(row) for row in records}
    if len(records) != PUBLIC_TASK_COUNT or len(by_id) != PUBLIC_TASK_COUNT:
        raise ValueError("The complete distinct 51-task public EdgeBench release is required")
    assigned = {}
    members = {}
    for name, values in sorted(groups.items()):
        if (
            not isinstance(name, str) or not name or isinstance(values, str)
            or not isinstance(values, Sequence)
            or len(values) < 2 or len(set(values)) != len(values)
        ):
            raise ValueError("A provenance group requires a name and distinct related tasks")
        unit = "group:" + name
        members[unit] = sorted(values)
        for ident in values:
            if ident not in by_id or ident in assigned:
                raise ValueError("Unknown or multiply grouped public task")
            assigned[ident] = unit
    for ident in sorted(by_id):
        if ident not in assigned:
            assigned[ident] = "task:" + ident
            members[assigned[ident]] = [ident]
    public_owners = {}
    for ident, row in by_id.items():
        digest = row["public_task_sha256"]
        previous = public_owners.setdefault(digest, assigned[ident])
        if previous != assigned[ident]:
            raise ValueError("Identical public tasks cross provenance groups")
    grouped = [
        {
            "id": unit,
            "payload_sha256": canonical_sha256([by_id[ident] for ident in values]),
            "public_task_sha256": canonical_sha256(sorted(
                by_id[ident]["public_task_sha256"] for ident in values
            )),
        }
        for unit, values in sorted(members.items())
    ]
    manifest = build_partition(
        grouped, benchmark="edgebench-public51-provenance-v1", seed=20260928,
        rounding="largest_remainder", independent_unit="public provenance group",
    )
    manifest["prior_exposure"] = (
        "Public release; no claim that tasks were absent from model pretraining or prior project work."
    )
    task_splits = {
        split: [ident for unit in units for ident in members[unit]]
        for split, units in manifest["splits"].items()
    }
    result = {
        "schema": "phiagent-edgebench-public51-partition/v1",
        "source_revision": SOURCE_REVISION, "dataset_revision": DATASET_REVISION,
        "public_task_count": PUBLIC_TASK_COUNT, "paper_task_count": 134,
        "group_partition": manifest,
        "group_members": members, "task_to_group": assigned,
        "task_splits": task_splits,
        "task_counts": {split: len(values) for split, values in task_splits.items()},
        "task_records": sorted(by_id.values(), key=lambda row: row["id"]),
        "public_identity": "Native query plus public workspace and work-image tag; no judge fields",
        "label": "Custom group-disjoint20/20/60 over all51 public tasks, rounded by independent unit",
        "score_or_outcome_used": False,
    }
    result["partition_sha256"] = canonical_sha256(result)
    return result


def verify_partition(partition: Mapping, groups: Mapping[str, Sequence[str]]) -> None:
    rebuilt = grouped_partition(partition["task_records"], groups)
    if rebuilt != partition:
        raise ValueError("EdgeBench partition differs from its frozen metadata-only assignment")


def actor_usage(sdk_usage: Mapping | None, *, complete: bool) -> dict:
    if type(complete) is not bool:
        raise ValueError("Usage completeness must be explicit")
    normalized = normalize_usage(sdk_usage)
    if not complete or normalized is None:
        return {"usage_complete": False, "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "known_sdk_usage": dict(sdk_usage) if sdk_usage else None}
    return {
        "usage_complete": True,
        "input_tokens": normalized["totalTokens"] - normalized["outputTokens"],
        "output_tokens": normalized["outputTokens"], "total_tokens": normalized["totalTokens"],
        "cache_read_tokens": normalized.get("cacheReadTokens"),
        "cache_write_tokens": normalized.get("cacheWriteTokens"),
        "reasoning_tokens": normalized.get("reasoningTokens"),
    }


def native_quality(report: Mapping, *, infrastructure_ok: bool) -> dict:
    if type(infrastructure_ok) is not bool:
        raise ValueError("Infrastructure validity must be explicit")
    score = report.get("score_0_100")
    if score is not None and (
        type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100
    ):
        raise ValueError("Expected the native finite score_0_100, not an invented solve rate")
    if not infrastructure_ok and score is not None:
        raise ValueError("An infrastructure failure cannot supply an ordinary native score")
    return {
        "native_score": score / 100 if infrastructure_ok and score is not None else None,
        "score_0_100": score,
        "native_valid": report.get("valid"),
        "native_pass_rate": report.get("pass_rate"),
        "native_raw_score": report.get("score"),
        "native_peak_score": report.get("peak_score"),
        "binary_task_success": None,
        "infrastructure_ok": infrastructure_ok,
    }
