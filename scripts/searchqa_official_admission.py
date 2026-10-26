"""Pure population, identity and cost gates for the SearchQA named controls."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


CONTROLS = ("no_skill", "skillopt", "autosaddler", "chisel_current")
OPTIMIZERS = {
    "skillopt": "79124b37e9a6371e13b753f8bcd7adb1e493ade1",
    "autosaddler": "30e20ce004486c58e7ee97c66182a8d0d41ec90e",
}
MODEL = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
COUNTS = {"train": 3395, "validation": 3395, "test": 10190}
USAGE = ("prompt_tokens", "completion_tokens", "total_tokens")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def population(partition: dict) -> dict[str, set[str]]:
    splits = {name: set(partition[name]["ids"]) for name in COUNTS}
    for name, ids in splits.items():
        if len(ids) != COUNTS[name] or len(ids) != len(partition[name]["ids"]):
            raise ValueError("The complete canonical split is required: " + name)
        groups = partition[name]["groups"]
        flattened = [key for group in groups for key in partition["groups"][group]]
        if len(groups) != len(set(groups)) or len(flattened) != len(ids) or set(flattened) != ids:
            raise ValueError("Independent groups do not cover the split exactly once")
    if any(splits[a] & splits[b] for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test"),
    )):
        raise ValueError("The train/validation/test populations overlap")
    if set(partition["groups"]) != {
        group for name in COUNTS for group in partition[name]["groups"]
    }:
        raise ValueError("The partition has missing or extra independent groups")
    return splits


def optimizer_roles(partition: dict) -> dict[str, list[str]]:
    """Partition only primary TRAIN; official internal selection is not validation."""
    ids = population(partition)["train"]
    groups = partition["train"]["groups"]
    cut = len(groups) * 4 // 5
    roles = {
        "train": sorted(key for group in groups[:cut] for key in partition["groups"][group]),
        "development": sorted(key for group in groups[cut:] for key in partition["groups"][group]),
    }
    if set(roles["train"]) & set(roles["development"]) or set().union(
        *(set(value) for value in roles.values())
    ) != ids:
        raise ValueError("Internal optimizer roles must partition primary TRAIN")
    return roles


def checked_usage(value: dict) -> dict:
    if (any(type(value.get(key)) is not int or value[key] < 0 for key in USAGE)
            or value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]):
        raise ValueError("Native input/output usage must be complete and additive")
    return {key: value[key] for key in USAGE}


def summarize(rows: list[dict], expected_ids: set[str]) -> dict:
    if len(rows) != len(expected_ids) or {row["id"] for row in rows} != expected_ids:
        raise ValueError("A paired complete population, not a favorable subset, is required")
    totals = dict.fromkeys(USAGE, 0)
    for row in rows:
        if (row.get("em") not in (0, 1) or not isinstance(row.get("f1"), (int, float))
                or not math.isfinite(row["f1"]) or not 0 <= row["f1"] <= 1):
            raise ValueError("Native EM/F1 is incomplete")
        for key, value in checked_usage(row["usage"]).items():
            totals[key] += value
    count = len(rows)
    if not count:
        raise ValueError("An empty evaluation is not complete")
    return {
        "tasks": count,
        "em_percent": 100 * sum(row["em"] for row in rows) / count,
        "f1_percent": 100 * sum(row["f1"] for row in rows) / count,
        "mean_actor_tokens": totals["total_tokens"] / count,
        "actor_input_tokens": totals["prompt_tokens"],
        "actor_output_tokens": totals["completion_tokens"],
        "total_actor_tokens": totals["total_tokens"],
    }


def identity_gate(method: str, identity: dict) -> None:
    if method not in OPTIMIZERS:
        return
    if (identity.get("official_engine_revision") != OPTIMIZERS[method]
            or identity.get("official_engine_returned") is not True
            or identity.get("all_configured_iterations_completed") is not True
            or identity.get("primary_feedback_split") != "train"
            or identity.get("external_validation_calls") != 0
            or identity.get("external_test_calls") != 0
            or identity.get("model_repository") != MODEL[0]
            or identity.get("model_revision") != MODEL[1]
            or not identity.get("selected_policy_sha256")
            or not identity.get("source_freeze_sha256")
            or identity.get("unknown_usage_calls") != 0):
        raise ValueError("A genuine completed, matched, TRAIN-only official optimizer is required")


def admission(candidate: dict, controls: dict, identities: dict) -> dict:
    missing = [name for name in CONTROLS if name not in controls]
    result = {
        "required_named_controls": list(CONTROLS),
        "internal_ablations_set_deployment_goal": False,
        "target_token_ratio": 0.8,
        "minimum_native_quality_delta": 0,
        "missing_named_controls": missing,
        "paper_admitted": False,
        "gate_passed": False,
    }
    if missing:
        return {**result, "status": "BLOCKED_MISSING_OFFICIAL_CONTROLS"}
    for method in CONTROLS:
        identity_gate(method, identities.get(method, {}))
        if controls[method]["tasks"] != candidate["tasks"]:
            raise ValueError("Named controls have different population sizes")
    reference = min(CONTROLS, key=lambda name: controls[name]["mean_actor_tokens"])
    denominator = controls[reference]["mean_actor_tokens"]
    if denominator <= 0:
        raise ValueError("Named-control native token cost must be positive")
    em = max(controls[name]["em_percent"] for name in CONTROLS)
    f1 = max(controls[name]["f1_percent"] for name in CONTROLS)
    quality = candidate["em_percent"] >= em and candidate["f1_percent"] >= f1
    ratio = candidate["mean_actor_tokens"] / denominator
    passed = quality and ratio <= 0.8
    return {
        **result,
        "status": "PASS_NAMED_CONTROL_GATE" if passed else "FAIL_NAMED_CONTROL_GATE",
        "cheapest_named_control": reference,
        "quality_goals": {"em_percent": em, "f1_percent": f1},
        "maximum_mean_actor_tokens": 0.8 * denominator,
        "candidate_token_ratio": ratio,
        "quality_passed": quality,
        "token_passed": ratio <= 0.8,
        "gate_passed": passed,
    }
