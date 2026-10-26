"""Validation-only boundaries and complete-population scalar reporting."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def validation_shard(data: Mapping, manifest: Mapping, indices: Sequence[int]) -> None:
    from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition

    validate_partition(manifest)
    if manifest["benchmark"] != "aime2025" or manifest["counts"] != {
        "train": 6, "validation": 6, "test": 18,
    }:
        raise RuntimeError("Wrong canonical validation partition")
    if set(data) != {"train", "development", "test"} or data["train"] or data["test"]:
        raise RuntimeError("Validation worker may contain validation payloads only")
    if list(indices) not in ([0, 1, 2], [3, 4, 5]):
        raise RuntimeError("Validation shards must be the predefined disjoint halves")
    expected = [manifest["splits"]["validation"][index] for index in indices]
    if [item["id"] for item in data["development"]] != expected:
        raise RuntimeError("Wrong validation shard membership or order")
    hashes = {item["id"]: item["payload_sha256"] for item in manifest["records"]}
    if any(canonical_sha256(item) != hashes[item["id"]] for item in data["development"]):
        raise RuntimeError("Canonical validation payload changed")


def complete_validation_summary(shards: Sequence[Mapping], expected_ids: Sequence[str]) -> dict:
    if len(shards) != 2 or len(expected_ids) != 6 or len(set(expected_ids)) != 6:
        raise RuntimeError("A complete two-shard validation audit is required")
    ids, correct, tokens, calls = [], 0, 0, 0
    for shard in shards:
        if shard["status"] != "PASS" or shard["unknown_usage_calls"]:
            raise RuntimeError("Validation shard is not independently admitted")
        if (len(shard["validation_ids"]) != 3 or type(shard["correct"]) is not int
                or not 0 <= shard["correct"] <= 3 or type(shard["validation_tokens"]) is not int
                or shard["validation_tokens"] < 0 or shard["new_calls"] != 3):
            raise RuntimeError("Invalid validation shard summary")
        ids.extend(shard["validation_ids"])
        correct += shard["correct"]
        tokens += shard["validation_tokens"]
        calls += shard["new_calls"]
    if len(ids) != 6 or len(set(ids)) != 6 or set(ids) != set(expected_ids) or calls != 6:
        raise RuntimeError("Missing or duplicated validation execution")
    return {"stage": "validation", "tasks": 6, "correct": correct,
            "validation_tokens": tokens, "new_solver_calls": calls,
            "test_calls": 0, "unknown_usage_calls": 0}


def exact_request(model: str, user_template: str, question: str, system_prompt: str,
                  seed: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_template.format(prompt=question)}],
        "max_tokens": 32768, "temperature": 0, "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
