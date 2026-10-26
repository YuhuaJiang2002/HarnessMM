#!/usr/bin/env python3
"""Durable complete TRAIN -> CHISEL -> VAL -> freeze -> paired GAIA2 TEST.

New: --config CONFIG --pilot-summary SUMMARY --run-dir NEW --detach
Resume: --run-dir EXISTING --resume --detach
Frozen TEST continuation: --continue-from PREDECESSOR --run-dir NEW_DATA0 --detach

Only committed callback receipts are replayed on resume. An owned, still-live
child is harvested, never relaunched. Unresolved native actions block the
campaign and retain their provider costs and evidence. A frozen TEST successor
may retry settled SDK transport failures or attested malformed wire JSON in a
new private native world; it never repairs the provider's tool arguments.
Output-limited actors are not resampled: future limits terminate normally for
native scoring, and prior capped trajectories use official judge-only validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deepseek_gaia2 import (  # noqa: E402
    ARE_VERSION, CAMPAIGN_ROOT, KEY_FILE, MODEL, OFFICIAL_REVISION, PARTITION_SHA256,
    USAGE_FIELDS, GateError, add_usage, canonical_hash, file_hash, freeze_source,
    initial_public_task, load, owned_path, persist, provenance, require_remote,
    resource_inventory, sdk_output_limit, sdk_usage, utc_now, verify_hashes,
)

SCHEMA = "deepseek-gaia2-campaign/v1"


class ParentTrainComplete(BaseException):
    """Stop at complete native TRAIN evidence without making an optimizer call."""


NATIVE_CONFIG_FIELDS = (
    "are_python", "sdk_python", "runtime_template", "filesystem", "filesystem_manifest_sha256",
    "dataset_revision", "official_revision", "model", "wire_api", "are_version",
    "max_output_tokens", "native_max_turns", "scenario_timeout_seconds", "cumulative_token_limit",
)


def native_config(config: dict) -> dict:
    fields = {key: config[key] for key in NATIVE_CONFIG_FIELDS}
    if "provider_durable_capture" in config:
        if config["provider_durable_capture"] is not True:
            raise GateError("Provider durable capture is an explicit opt-in revision")
        fields["provider_durable_capture"] = True
    if "native_execution_contract" in config:
        from phiagent.harness.gaia2_execution_contract import validate_contract

        fields["native_execution_contract"] = validate_contract(config["native_execution_contract"])
    if "shared_gaia2_prompt" in config:
        from phiagent.harness.gaia2_common_prompt import validate_prompt_identity

        fields["shared_gaia2_prompt"] = validate_prompt_identity(config["shared_gaia2_prompt"])
    if "native_observation_encoding" in config:
        encoding = config["native_observation_encoding"]
        if encoding not in ("canonical_json", "native_text"):
            raise GateError("Unsupported native observation encoding")
        fields["native_observation_encoding"] = encoding
    return fields


def validate_config(config: dict) -> None:
    required = {
        "schema": SCHEMA, "campaign_root": str(CAMPAIGN_ROOT),
        "model": MODEL, "official_revision": OFFICIAL_REVISION, "are_version": ARE_VERSION,
        "wire_api": "anthropic_messages", "partition_sha256": PARTITION_SHA256,
        "counts": {"train": 32, "validation": 32, "test": 96}, "split_seed": 20260928,
        "execution_seeds": [20260928],
        "test_execution_seeds": [2026092801, 2026092802, 2026092803],
        "executor_workers": 2, "optimization_iterations": 2, "optimizer_max_units": None,
        "optimizer_trace_max_characters_per_result": 3500,
        "application": "system_section", "cumulative_token_limit": None,
        "sampling_seed_supported": False, "scenario_timeout_seconds": 1860,
    }
    if "native_execution_contract" in config:
        from phiagent.harness.gaia2_execution_contract import validate_contract

        contract = validate_contract(config["native_execution_contract"])
        required["scenario_timeout_seconds"] = contract["outer_wall_timeout_seconds"]
        required["native_max_turns"] = contract["unspecified_user_turn_limit"]
    for key, expected in required.items():
        if key not in config or config[key] != expected:
            raise GateError(f"Campaign configuration changed the frozen protocol: {key}")
    if not 0 < config["heartbeat_seconds"] <= 300 or config["capacity_audit_seconds"] != 300:
        raise GateError("A five-minute heartbeat and resource audit are required")
    if "shared_gaia2_prompt" in config or "native_observation_encoding" in config:
        native_config(config)
    if "minimum_parent_train_successes" in config:
        minimum = config["minimum_parent_train_successes"]
        if (
            type(minimum) is not int or not 1 <= minimum <= 32
            or "shared_gaia2_prompt" not in config
        ):
            raise GateError("A new shared-parent capability gate needs 1--32 TRAIN successes")


def parent_capability_gate(request: dict, config: dict, train_ids: set[str]) -> dict | None:
    minimum = config.get("minimum_parent_train_successes")
    if minimum is None:
        return None
    validate_config(config)
    rows = request["training_evidence"]
    if (
        request.get("phase") != "train"
        or len(rows) != len(train_ids) or len(train_ids) != 32
        or {row["task"]["task_id"] for row in rows} != train_ids
    ):
        raise GateError("Parent capability must use every TRAIN scenario exactly once")
    baseline = [row["baseline"] for row in rows]
    if any(
        row.get("execution_status") != "completed"
        or type(row.get("native_score")) not in (int, float)
        or row["native_score"] not in (0, 1)
        for row in baseline
    ):
        raise GateError("Parent capability needs complete native binary TRAIN verdicts")
    successes = sum(int(row["native_score"]) for row in baseline)
    return {
        "schema": "deepseek-gaia2-shared-parent-capability/v1",
        "scope": "TRAIN32", "independent_scenarios": 32,
        "successes": successes, "success_percent": 100 * successes / 32,
        "minimum_successes": minimum, "passed": successes >= minimum,
        "shared_gaia2_prompt": native_config(config)["shared_gaia2_prompt"],
        "task_ids": sorted(train_ids), "validation_or_test_feedback": False,
        "selection_or_optimization_from_test": False,
        "cumulative_token_limit": None,
    }


def compact_public_trace(path: Path, *, limit: int = 3500) -> str:
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if value.get("kind") not in (
                "native_tool_result", "native_tool_error", "native_argument_error",
                "native_notification", "harness_final_response_delivered",
            ):
                continue
            record = {key: value[key] for key in (
                "kind", "tool", "arguments", "observation", "message", "notification_type",
                "timestamp", "conclude_turn", "error_type", "error", "errors",
            ) if key in value}
            observation = record.get("observation")
            if isinstance(observation, str) and len(observation) > 1800:
                record["observation"] = observation[:1500] + "\n[public trace excerpt]\n" + observation[-300:]
                record["observation_original_characters"] = len(observation)
            rows.append(record)
    selected = rows if len(rows) <= 10 else rows[:5] + rows[-5:]
    while selected and len(json.dumps(selected, ensure_ascii=False)) > limit:
        selected.pop(len(selected) // 2)
    return json.dumps({
        "public_actor_trace": selected, "public_event_count": len(rows),
        "selection": "Deterministic first/last public events, with declared observation excerpts",
        "full_public_trace_sha256": file_hash(path) if path.is_file() else None,
        "judge_prompts_or_private_oracle_evidence": False,
    }, ensure_ascii=False)


def optimizer_payload(request: dict, train_ids: set[str]) -> dict:
    evidence = request["training_evidence"]
    identifiers = [row["task"]["task_id"] for row in evidence]
    if (request.get("phase") != "train" or set(identifiers) != train_ids
            or len(identifiers) != len(train_ids)):
        raise GateError("Optimizer requires the complete TRAIN population exactly once")

    def public_result(result: dict) -> dict:
        return {
            "native_score": result["native_score"],
            "execution_status": result["execution_status"],
            "complete_task_actor_tokens": result["usage"]["total_tokens"],
            "public_actor_trace": json.loads(result["feedback"]) if result.get("feedback") else None,
        }

    instructions = {row["id"]: row["instruction"] for row in request["previous_candidates"]}
    return {
        "objective": request["objective"] + (
            "\nReturn ONLY a JSON object with exactly two keys: instruction (string), "
            "supporting_task_ids (array of supplied TRAIN task IDs). No markdown. "
            f"At most {request['limits']['max_characters']} characters and "
            f"{request['limits']['max_words']} words. An empty instruction is an abstention."
        ),
        "training_evidence": [
            {"task": row["task"], "seed": row["seed"], "baseline": public_result(row["baseline"]),
             "candidates": {
                 name: {"instruction": instructions.get(name), **public_result(value)}
                 for name, value in row["candidates"].items()
             }}
            for row in evidence
        ],
    }


def parse_proposal(text: Any, train_ids: set[str], limits: dict) -> dict:
    if not isinstance(text, str):
        raise GateError("Optimizer did not return a textual JSON proposal")
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"instruction", "supporting_task_ids"}:
        raise GateError("Optimizer proposal fields differ from the frozen contract")
    instruction, support = value["instruction"], value["supporting_task_ids"]
    if (not isinstance(instruction, str) or "\0" in instruction
            or len(instruction) > limits["max_characters"]
            or len(instruction.split()) > limits["max_words"]
            or not isinstance(support, list) or any(not isinstance(key, str) for key in support)
            or len(set(support)) != len(support) or not set(support) <= train_ids
            or (instruction.strip() and not support)):
        raise GateError("Optimizer proposal violates the sparse TRAIN-only support contract")
    return value


def native_callback_result(request: dict, summary: dict, trial: Path) -> dict:
    accounting = summary.get("provider_usage") or {}
    valid = summary.get("valid_for_comparison") is True and (
        type(summary.get("success")) is bool
        and summary.get("native_score") == int(summary["success"])
    )
    result = {
        "request_id": request["request_id"], "task_id": request["task"]["task_id"],
        "identity": request["identity"], "execution_status": "completed" if valid else "blocked",
        "native_score": summary.get("native_score"), "success": summary.get("success"),
        "usage": summary.get("execution_usage") or accounting.get("combined") or dict.fromkeys(USAGE_FIELDS),
        "provenance": {
            "summary": str(trial / "SUMMARY.json"), "summary_sha256": file_hash(trial / "SUMMARY.json"),
            "native_result": summary.get("native_result"), "official_runtime": summary["runtime"],
            "provider_usage": accounting, "sampling_seed_supported": False,
            "seed_role": "Paired repetition label; not a claim of seeded provider sampling",
            "SDK_H0_for_empty_residual": not request.get("system_prompt_append"),
            "outer_arm": request["arm"], "cumulative_token_limit": None,
            "retained_native": summary.get("retained_native"),
        },
    }
    if request["phase"] == "train":
        result["feedback"] = compact_public_trace(trial / "native/backend/PUBLIC_BRIDGE.jsonl")
    return result


def retained_parent_trial(config: dict, request: dict, records: dict) -> Path | None:
    """Adopt all compatible settled TRAIN parents after a transport-only repair."""
    if request["phase"] != "train" or request["arm"] != "baseline":
        return None
    rows = config.get("retained_parent_train") or {}
    entry = rows.get(request["task"]["task_id"])
    if entry is None:
        return None
    if (config.get("provider_durable_capture") is not True or request["phase"] != "train"
            or request["arm"] != "baseline" or request.get("system_prompt_append")
            or request.get("residual")):
        raise GateError("Retained parent reuse is restricted to an unchanged TRAIN baseline")
    path = Path(entry["summary"])
    if file_hash(path) != entry["summary_sha256"]:
        raise GateError("Retained parent summary changed")
    trial = path.parent
    summary, old_config, old_job = load(path), load(trial / "CONFIG.json"), load(trial / "JOB.json")
    current_protocol, old_protocol = native_config(config), native_config(old_config)
    current_protocol.pop("provider_durable_capture", None)
    old_protocol.pop("provider_durable_capture", None)
    actor_path = trial / "sdk/result.json"
    native_path = Path(summary["native_result"])
    for name, source in (("actor", actor_path), ("native", native_path),
                         ("config", trial / "CONFIG.json"), ("job", trial / "JOB.json")):
        if file_hash(source) != entry[name + "_sha256"]:
            raise GateError("Retained parent evidence changed")
    actor, native = load(actor_path), load(native_path)
    if (current_protocol != old_protocol or old_job.get("chisel") is not None
            or old_job["task_id"] != request["task"]["task_id"]
            or old_job["phase"] != "train" or old_job["arm"] != "baseline"
            or old_job["seed"] != request["seed"]
            or old_job["public_prompt"] != request["prompt"]
            or old_job["scenario_sha256"] != records[old_job["task_id"]]["payload_sha256"]
            or summary.get("valid_for_comparison") is not True
            or type(native.get("success")) is not bool or native.get("exception") is not None
            or summary.get("native_score") != int(native["success"])
            or (summary.get("provider_usage") or {}).get("complete") is not True
            or (actor.get("status") != "completed" and not sdk_output_limit(actor))
            or actor.get("closed") is not True
            or actor.get("runtime_closed") is not True or actor.get("usage_complete") is not True
            or (summary["provider_usage"].get("actor") or {}).get("usage") != sdk_usage(actor)):
        raise GateError("Retained parent native protocol, verdict or accounting differs")
    return trial


def trial_attempts(summary: dict, trial: Path, stage: str) -> list[dict]:
    accounting = summary.get("provider_usage") or {}
    return [
        {
            "attempt_id": str(trial / role), "stage": stage + "_" + role,
            "usage": (accounting.get(role) or {}).get("usage") or dict.fromkeys(USAGE_FIELDS),
            "provider_attempt_count": (accounting.get(role) or {}).get("attempts"),
            "provider_native_usage": (accounting.get(role) or {}).get("provider_native_usage"),
            "known_reported_usage": (accounting.get(role) or {}).get("known_reported_usage"),
            "reasoning_tokens": (accounting.get(role) or {}).get("reasoning_tokens"),
        } for role in ("actor", "judge")
    ]


def unsettled_attempts(directory: Path, kind: str, phase: str) -> list[dict]:
    """Account for paid work even if a native process dies before publishing its verdict."""
    trial = directory / "trial"
    if kind == "executor" and (trial / "SUMMARY.json").is_file():
        return trial_attempts(load(trial / "SUMMARY.json"), trial, phase)
    sdk = trial / "sdk" if kind == "executor" else directory / "sdk"
    result = load(sdk / "result.json") if (sdk / "result.json").is_file() else {}
    accounting = load(sdk / "usage.json") if (sdk / "usage.json").is_file() else {}
    stream_path = sdk / "llm-streams.jsonl"
    if stream_path.is_file():
        from integrations.deepseek_harness.runner import summarize_streams

        records = []
        lines = stream_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise GateError("An interior SDK provider receipt is corrupt") from None
        if any(row.get("kind") == "llm/start" for row in records):
            accounting = summarize_streams(records)
    if not result and not accounting:
        return []
    complete = result.get("closed") is True and result.get("usage_complete") is True
    actor = {
        "attempt_id": str(trial / "actor") if kind == "executor" else str(sdk),
        "stage": phase + "_actor" if kind == "executor" else "training_optimizer",
        "usage": sdk_usage(result) if complete else dict.fromkeys(USAGE_FIELDS),
        "provider_attempt_count": accounting.get("attempt_count"),
        "provider_native_usage": result.get("usage"),
        "known_reported_usage": accounting.get("known_reported_usage"),
        "reasoning_tokens": (result.get("usage") or {}).get("reasoningTokens"),
        "callback_settled": False,
    }
    if kind != "executor":
        return [actor]
    usage, known, judge_complete = [], [], True
    for path in (trial / "judge").glob("*/STARTED.json"):
        response = path.with_name("RESULT.json")
        if response.is_file() and load(response).get("accounting_complete") is True:
            value = load(response)["usage"]
            usage.append(value)
            known.append(value)
        else:
            usage.append(dict.fromkeys(USAGE_FIELDS))
            judge_complete = False
    return [actor, {
        "attempt_id": str(trial / "judge"), "stage": phase + "_judge",
        "usage": add_usage(usage) if complete and judge_complete else dict.fromkeys(USAGE_FIELDS),
        "provider_attempt_count": len(usage), "provider_native_usage": None,
        "known_reported_usage": add_usage(known), "reasoning_tokens": None,
        "callback_settled": False,
    }]


def unique_attempts(rows: list[dict]) -> list[dict]:
    found = {}
    for row in rows:
        old = found.get(row["attempt_id"])
        if old is not None and old != row:
            raise GateError("Conflicting cost receipts share one native attempt identity")
        found[row["attempt_id"]] = row
    return list(found.values())


def usage_summary(rows: list[dict]) -> dict:
    rows = unique_attempts(rows)

    def known(row: dict, key: str):
        value = row["usage"].get(key)
        if value is not None:
            return value
        raw = row.get("known_reported_usage") or {}
        if "totalTokens" in raw:
            converted = sdk_usage({"usage_complete": True, "usage": raw})
            return converted.get(key) or 0
        return raw.get(key) or 0

    return {
        "unique_attempts": len(rows), "measured": add_usage([row["usage"] for row in rows]),
        "known_subtotals": {
            key: sum(known(row, key) for row in rows)
            for key in USAGE_FIELDS
        },
        "unknown_attempts": [row["attempt_id"] for row in rows
                             if row["usage"].get("total_tokens") is None],
        "provider_attempts_known_subtotal": sum(
            row["provider_attempt_count"] for row in rows
            if type(row.get("provider_attempt_count")) is int
        ),
        "unknown_is_zero": False, "cumulative_token_limit": None,
    }


def complete_test(rows: list[dict], ids: set[str], seeds: list[int]) -> bool:
    actual = {}
    for row in rows:
        request, result = row["request"], row["result"]
        if request["phase"] != "test":
            continue
        key = (request["task"]["task_id"], request["seed"], request["arm"])
        if key in actual:
            return False
        actual[key] = result
    expected = {(ident, seed, arm) for ident in ids for seed in seeds
                for arm in ("baseline", "chisel")}
    return set(actual) == expected and all(
        value["execution_status"] == "completed"
        and type(value["success"]) is bool
        and value["native_score"] == int(value["success"])
        and value["usage"]["total_tokens"] is not None for value in actual.values()
    )


def verified_callback(directory: Path) -> tuple[dict, dict, list[dict]]:
    """Read an old callback without changing its committed files."""
    seal = load(directory / "COMMITTED.json")
    for name in ("REQUEST.json", "RESULT.json", "USAGE.json"):
        if file_hash(directory / name) != seal[name]:
            raise GateError(f"Predecessor callback seal changed: {directory / name}")
    request, result, usage = (load(directory / name)
                              for name in ("REQUEST.json", "RESULT.json", "USAGE.json"))
    if (result["request_id"] != request["request_id"] or result["identity"] != request["identity"]
            or usage["usage"] != result["usage"]):
        raise GateError("Predecessor callback identity/accounting mismatch")
    return request, result, usage["attempts"]


def valid_observation(result: dict) -> bool:
    return (result.get("execution_status") == "completed"
            and type(result.get("success")) is bool
            and result.get("native_score") == int(result["success"]))


def shared_execution_request(adapted: dict) -> dict:
    """Reverse only the existing DeepSeek ABI rename to verify the original request seal."""
    value = deepcopy(adapted)
    value.pop("request_id", None)
    value["schema"] = value.pop("controller_origin_schema")
    value["codex_prompt"] = value.pop("prompt")
    value["agents_md_append"] = ""
    value["contract"].pop("context_application")
    return value


def frozen_test_plan(policy: dict, template: dict, tasks: list[dict]) -> list[list[dict]]:
    from phiagent.harness.deepseek_chisel import adapt_request

    candidate = policy["candidate"]
    if (candidate["id"] != "noop" or candidate["instruction"] or candidate["enabled"]
            or candidate["candidate_sha256"] != policy["baseline_policy_sha256"]):
        raise GateError("This continuation admits only the already-frozen NOOP policy")
    if (template["phase"] != "test" or template["run_id"] != policy["run_id"]
            or template["final_policy_sha256"] != policy["policy_sha256"]
            or template["source_sha256"] != policy["source_sha256"]
            or canonical_hash(shared_execution_request(template)) != template["request_id"]):
        raise GateError("Original TEST callback differs from the frozen policy/request identity")
    base = shared_execution_request(template)
    groups = []
    for task in sorted(tasks, key=lambda row: row["task_id"]):
        for seed in policy["config"]["test_execution_seeds"]:
            order = ["baseline", "chisel"]
            if bytes.fromhex(canonical_hash([policy["run_id"], task["task_id"], seed]))[0] % 2:
                order.reverse()
            pair = []
            for arm in order:
                shared = deepcopy(base)
                shared.update(
                    task=task, seed=seed, arm=arm, codex_prompt=task["public_prompt"],
                    pair_id=canonical_hash([policy["run_id"], "test", task["task_id"], seed]),
                    policy_sha256=candidate["candidate_sha256"],
                    candidate_sha256=candidate["candidate_sha256"],
                    system_prompt_append="", residual=None,
                )
                shared["request_id"] = canonical_hash(shared)
                pair.append(adapt_request(shared, optimizer=False))
            groups.append(pair)
    return groups


def malformed_tool_json_evidence(trial: Path, failure: dict) -> dict:
    stream = trial / "sdk/llm-streams.jsonl"
    expected = {
        "code": "MALFORMED_RESPONSE",
        "message": "DeepSeek Messages stream: tool input is invalid JSON",
    }
    if failure != expected or not stream.is_file():
        raise GateError("Malformed SDK retry requires saved invalid-JSON wire evidence")
    records = [json.loads(line) for line in stream.read_text().splitlines()]
    starts = [row for row in records if row.get("kind") == "llm/start"]
    if not starts or not isinstance(starts[-1].get("attempt_id"), str):
        raise GateError("Malformed SDK response has no identified final provider attempt")
    attempt_id = starts[-1]["attempt_id"]
    attempt = [row for row in records if row.get("attempt_id") == attempt_id]
    if (sum(row.get("kind") == "llm/start" for row in attempt) != 1
            or sum(row.get("kind") == "llm/end" for row in attempt) != 1
            or attempt[-1].get("kind") != "llm/end"):
        raise GateError("Malformed SDK provider attempt is not durably settled")
    start_index = records.index(starts[-1])
    if any(row.get("kind") == "bridge/tool/start" for row in records[start_index:]):
        raise GateError("Malformed provider response dispatched a native tool; do not replay")
    chunks = [row.get("chunk", {}) for row in attempt if row.get("kind") == "llm/chunk"]
    finishes = [chunk for chunk in chunks if chunk.get("type") == "finish"]
    if (len(finishes) != 1
            or finishes[0].get("reason") != {"kind": "error", "failure": expected}):
        raise GateError("Saved provider failure differs from the closed SDK result")
    invalid = []

    def reject_constant(value: str):
        raise ValueError(f"Non-JSON constant: {value}")

    for chunk in chunks:
        block = chunk.get("block", {})
        if chunk.get("type") != "block-end" or block.get("type") != "tool-call":
            continue
        arguments = block.get("arguments")
        if not isinstance(arguments, str):
            raise GateError("Saved tool arguments lack the original wire JSON text")
        try:
            json.loads(arguments, parse_constant=reject_constant)
        except ValueError as error:
            invalid.append({
                "index": chunk.get("index"), "tool_call_id": block.get("id"),
                "tool_name": block.get("name"),
                "arguments_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
                "json_error": str(error),
            })
    if not invalid:
        raise GateError("The failed provider response contains no attested malformed tool JSON")
    return {
        "provider_attempt_id": attempt_id, "stream": str(stream),
        "stream_sha256": file_hash(stream), "failure": expected,
        "invalid_tool_arguments": invalid, "failed_response_tools_dispatched": 0,
        "provider_output_rewritten": False, "new_model_or_judge_calls": 0,
    }


def transport_retry_admission(directory: Path) -> dict:
    """Retry only a closed, attested SDK wire failure with no unsettled native actions."""
    trial = directory / "trial"
    result = load(trial / "sdk/result.json")
    native = load(trial / "native/RESULT.json")
    reason = result.get("finish_details") or {}
    failure = reason.get("error") or {}
    code = failure.get("code")
    if (
        result.get("closed") is not True or result.get("runtime_closed") is not True
        or result.get("status") != "failed"
        or code not in ("TRANSPORT", "MALFORMED_RESPONSE")
        or native.get("exception") is None
    ):
        raise GateError("Unresolved predecessor is not a settled SDK transport/wire failure")
    calls = settled_native_calls(directory)
    evidence = malformed_tool_json_evidence(trial, failure) if code == "MALFORMED_RESPONSE" else None
    usage = load(trial / "sdk/usage.json")
    return {
        "classification": (
            "terminal_official_sdk_messages_malformed_tool_json" if evidence
            else "terminal_official_sdk_messages_transport_failure"
        ),
        "sdk_result": str(trial / "sdk/result.json"),
        "sdk_result_sha256": file_hash(trial / "sdk/result.json"),
        "native_verdict_unresolved": True, "native_calls_started": len(calls),
        "native_calls_settled": len(calls), "live_native_actions_replayed": False,
        "malformed_response_evidence": evidence,
        "provider_attempts": usage.get("attempt_count"),
        "known_reported_usage": usage.get("known_reported_usage"),
        "unknown_reasons": usage.get("unknown_reasons"), "unknown_is_zero": False,
        "retry": "New private native world and official SDK session; original task/seed/arm",
    }


def settled_native_calls(directory: Path) -> list[Path]:
    process = load(directory / "PROCESS.json")
    stat_path = Path(f"/proc/{process['pid']}/stat")
    if stat_path.exists():
        fields = stat_path.read_text().split()
        if fields[21] == process["start_ticks"] and fields[2] != "Z":
            raise GateError("The predecessor still owns a live native process")
    calls = list((directory / "trial/native/bridge/calls").glob("*/STARTED.json"))
    if any(not path.with_name("RESULT.json").is_file() for path in calls):
        raise GateError("A native mutation has no settled receipt; do not replay it")
    return calls


def output_limit_verification_admission(directory: Path) -> dict:
    from integrations.deepseek_harness.runner import summarize_streams

    trial = directory / "trial"
    request = load(directory / "REQUEST.json")
    actor = load(trial / "sdk/result.json")
    native = load(trial / "native/RESULT.json")
    summary = load(trial / "SUMMARY.json")
    if (
        not sdk_output_limit(actor) or actor.get("closed") is not True
        or actor.get("runtime_closed") is not True
        or not str(native.get("exception") or "").startswith("Official SDK native turn failed:")
        or summary.get("native_score") is not None
        or (summary.get("provider_usage") or {}).get("complete") is not True
    ):
        raise GateError("Offline recovery requires a closed, fully metered SDK output-limit finish")
    calls = settled_native_calls(directory)
    stream = trial / "sdk/llm-streams.jsonl"
    usage = summarize_streams([json.loads(line) for line in stream.read_text().splitlines()])
    saved_usage = load(trial / "sdk/usage.json")
    limit = load(trial / "CONFIG.json")["max_output_tokens"]
    if (
        usage["complete"] is not True or usage["usage"] != actor["usage"]
        or usage["usage"] != saved_usage.get("usage") or not usage["attempts"]
        or usage["attempts"][-1]["usage"]["outputTokens"] != limit
        or usage["attempts"][-1].get("finish", {}).get("reason") != {"kind": "max-tokens"}
    ):
        raise GateError("Saved native output-limit usage is not conserved")
    trace = Path(native["export_path"])
    if trace.is_symlink() or not trace.is_relative_to(trial / "native/traces"):
        raise GateError("Retained native trace is outside its original private trial")
    exported = load(trace)
    definition = exported.get("metadata", {}).get("definition", {})
    if (definition.get("scenario_id") != request["task"]["task_id"]
            or not isinstance(exported.get("completed_events"), list)):
        raise GateError("Retained native trace identity or completed events are missing")
    paths = {
        "request": directory / "REQUEST.json", "summary": trial / "SUMMARY.json",
        "sdk_result": trial / "sdk/result.json", "sdk_usage": trial / "sdk/usage.json",
        "stream": stream, "native_result": trial / "native/RESULT.json", "trace": trace,
        "config": trial / "CONFIG.json",
    }
    return {
        "classification": "native_output_limit_requires_offline_validation",
        "evidence": {name: {"path": str(path), "sha256": file_hash(path)}
                     for name, path in paths.items()},
        "native_calls_settled": len(calls), "completed_events": len(exported["completed_events"]),
        "native_output_limit": limit, "provider_attempts": usage["attempt_count"],
        "original_usage": summary["provider_usage"]["combined"], "new_actor_calls": 0,
        "tool_actions_replayed": False, "output_cap_resampling_permitted": False,
        "scoring": "Unmodified ARE ScenarioRunner judge_only on retained completed events",
    }


def validate_wire_retry_policy(admissions: dict, prior_admissions: list[dict]) -> dict:
    maximum = 2
    history = {}
    for prior in prior_admissions:
        for proof in prior.values():
            kind = proof["classification"]
            history[kind] = history.get(kind, 0) + 1
    proposed = {}
    for proof in admissions.values():
        kind = proof["classification"]
        proposed[kind] = proposed.get(kind, 0) + 1
    if any(history.get(kind, 0) + count > maximum for kind, count in proposed.items()):
        raise GateError("Unchanged SDK wire failure reached its two-retry limit; no further resampling")
    return {
        "maximum_unchanged_wire_failure_retries": maximum,
        "prior_retry_counts_by_classification": history,
        "new_retry_counts_by_classification": proposed,
        "output_limit_actor_retries": 0, "valid_native_zero_or_one_retries": 0,
        "policy_or_provider_output_rewriting": False,
    }


def prepare(config: dict, root: Path, pilot: Path, history: list[Path]) -> None:
    require_remote()
    validate_config(config)
    owned_path(root).mkdir(parents=True, exist_ok=False, mode=0o700)
    os.umask(0o077)
    frozen = {**config, "pilot_summary": str(pilot.resolve()),
              "pilot_summary_sha256": file_hash(pilot),
              "historical_trial_summaries": [
                  {"path": str(path.resolve()), "sha256": file_hash(path)} for path in history
              ]}
    persist(root / "CONFIG.json", frozen, immutable=True)
    hashes = freeze_source(root / "source")
    persist(root / "SOURCE_SHA256.json", hashes, immutable=True)
    persist(root / "PREPARATION.json", provenance(ROOT, sys.argv, config["split_seed"]),
            immutable=True)
    persist(root / "RESOURCE_SELECTION.json", resource_inventory(), immutable=True)


class Campaign:
    def __init__(self, root: Path):
        self.root, self.source = root, root / "source"
        self.config = load(root / "CONFIG.json")
        self.hashes = load(root / "SOURCE_SHA256.json")
        self.partition_root = Path(self.config["partition_dir"])
        self.partition = load(self.partition_root / "SPLIT_MANIFEST.json")
        self.records = {row["id"]: row for row in self.partition["records"]}
        self.runtime = load(Path(self.config["runtime_template"]))["runtime"]
        self.identity = {
            "runtime": "official_deepseek_harness_sdk", "wire_api": "anthropic_messages",
            "model": MODEL, "official_revision": OFFICIAL_REVISION,
            "runtime_manifest_sha256": file_hash(Path(self.runtime["manifest"])),
            "transport": "official_native_anthropic_messages",
        }
        self.native_hashes = {
            name: digest for name, digest in self.hashes.items()
            if name.startswith("integrations/deepseek_harness/")
            or name in ("scripts/run_deepseek_gaia2.py", "phiagent/harness/deepseek_bridge.py",
                        "phiagent/data_engine/provenance.py")
        }
        self.protocol_hash = canonical_hash({
            "native_sources": self.native_hashes, "identity": self.identity,
            "config": native_config(self.config),
        })
        self.phase = "preflight"
        self.controller = None
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.active: dict[str, dict] = {}
        self.pilot_trial = Path(self.config["pilot_summary"]).parent

    def check_sources(self) -> None:
        verify_hashes(self.source, self.hashes)
        if hasattr(self, "native_source"):
            verify_hashes(self.native_source, self.scorer_hashes)
        if file_hash(Path(self.runtime["manifest"])) != self.identity["runtime_manifest_sha256"]:
            raise GateError("Official runtime manifest changed")

    def preflight(self) -> None:
        require_remote()
        validate_config(self.config)
        self.check_sources()
        from phiagent.harness.benchmark_partition import validate_partition
        validate_partition(self.partition)
        for name, field in (
            ("SPLIT_MANIFEST.json", "partition_sha256"),
            ("TRAIN_VALIDATION.json", "train_validation_sha256"),
            ("SEALED_TEST.json", "sealed_test_sha256"),
        ):
            if file_hash(self.partition_root / name) != self.config[field]:
                raise GateError(f"Frozen split artifact changed: {name}")
        if self.partition["counts"] != self.config["counts"]:
            raise GateError("Complete GAIA2-mini population is required")
        manifest_path = Path(self.config["filesystem_manifest"])
        if file_hash(manifest_path) != self.config["filesystem_manifest_sha256"]:
            raise GateError("Frozen public filesystem manifest changed")
        prefix = "gaia2_filesystem/demo_filesystem/"
        assets = [row for row in load(manifest_path)["files"] if row["path"].startswith(prefix)]
        verified_bytes = 0
        for row in assets:
            relative = Path(row["path"][len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise GateError("Invalid frozen public asset path")
            path = Path(self.config["filesystem"]) / relative
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]
                    or file_hash(path) != row["sha256"]):
                raise GateError(f"Frozen public filesystem asset changed: {relative}")
            verified_bytes += row["bytes"]
        if (len(assets) != self.config["filesystem_file_count"]
                or verified_bytes != self.config["filesystem_bytes"]):
            raise GateError("Full official filesystem population is required")
        distribution = importlib.metadata.distribution("meta-agents-research-environments")
        if distribution.version != ARE_VERSION:
            raise GateError("Native ARE version differs from the pilot")
        self.native_source = Path(distribution.locate_file("are/simulation"))
        self.scorer_hashes = {
            path.relative_to(self.native_source).as_posix(): file_hash(path)
            for path in sorted(self.native_source.rglob("*.py"))
        }
        persist(self.root / "NATIVE_SOURCE_SHA256.json", self.scorer_hashes, immutable=True)
        index = []
        for ident, record in self.records.items():
            path = self.partition_root / "scenarios" / (ident + ".json")
            if file_hash(path) != record["payload_sha256"]:
                raise GateError("Frozen scenario source changed: " + ident)
            text = initial_public_task(load(path))
            index.append({"task_id": ident, "unit_id": ident,
                          "public_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()})
        self.index = index
        persist(self.root / "PUBLIC_HASH_INDEX.json", {
            "tasks": index, "purpose": "Trusted initial-public-task hash commitment only",
            "test_public_text_read_to_hash": True, "test_text_supplied_to_controller": False,
            "future_user_turns_used": False, "oracle_events_used": False,
            "partition_record_public_hash_is_not_used_for_routing": True,
        }, immutable=True)
        self.import_pilot()
        persist(self.root / "PREFLIGHT.json", {
            "status": "PASSED", "counts": self.partition["counts"], "model": MODEL,
            "native_protocol_sha256": self.protocol_hash,
            "filesystem_files": len(assets), "filesystem_bytes": verified_bytes,
            "first_train_real_native_verdict_verified": True, "test_opened": False,
            "cumulative_token_limit": None,
        }, immutable=True)

    def import_pilot(self) -> None:
        path = Path(self.config["pilot_summary"])
        if file_hash(path) != self.config["pilot_summary_sha256"]:
            raise GateError("Pilot receipt changed")
        summary = load(path)
        self.pilot = summary
        pilot_root = self.pilot_trial.parent
        pilot_hashes = load(pilot_root / "SOURCE_SHA256.json")
        if any(pilot_hashes.get(name) != digest for name, digest in self.native_hashes.items()):
            raise GateError("Pilot native/SDK source differs; require a new real first-TRAIN gate")
        pilot_config = load(self.pilot_trial / "CONFIG.json")
        if native_config(pilot_config) != native_config(self.config):
            raise GateError("Pilot execution settings differ from the campaign")
        if (
            summary.get("valid_for_comparison") is not True
            or summary["task_id"] != self.partition["splits"]["train"][0]
            or summary["phase"] != "train" or summary["arm"] != "baseline"
            or summary["seed"] != self.config["execution_seeds"][0]
            or summary["runtime"] != self.identity["runtime"]
            or (summary.get("provider_usage") or {}).get("complete") is not True
        ):
            raise GateError("The first hash-ranked real TRAIN integration gate has not passed")
        persist(self.root / "PILOT_IMPORT.json", {
            "summary": str(path), "sha256": file_hash(path), "eligible_training_reuse": True,
            "cost_role": "First TRAIN baseline acquisition, charged exactly once by its callback",
            "native_protocol_sha256": self.protocol_hash,
        }, immutable=True)
        history = []
        for row in self.config["historical_trial_summaries"]:
            receipt = Path(row["path"])
            if file_hash(receipt) != row["sha256"] or receipt == path:
                raise GateError("Historical cost receipt changed or duplicates the accepted pilot")
            history += trial_attempts(load(receipt), receipt.parent, "inherited_integration")
        persist(self.root / "INHERITED_ATTEMPTS.json", unique_attempts(history), immutable=True)

    def tasks(self, phase: str) -> list[dict]:
        if phase == "test":
            for name in ("TRAIN_COMPLETE.json", "VALIDATION_COMPLETE.json", "FROZEN_POLICY.json"):
                if not (self.root / name).is_file():
                    raise GateError("Test remains sealed until all automatic phase gates pass")
            self.check_sources()
            entries = load(self.partition_root / "SEALED_TEST.json")["test"]
        else:
            entries = load(self.partition_root / "TRAIN_VALIDATION.json")[phase]
        if [row["id"] for row in entries] != self.partition["splits"][phase]:
            raise GateError("Phase must use its complete frozen population in planned order")
        return [
            {"task_id": row["id"], "unit_id": row["id"],
             "public_prompt": initial_public_task(load(Path(row["path"])))}
            for row in entries
        ]

    def capacity(self) -> None:
        while True:
            free = shutil.disk_usage(self.root).free / 2**30
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(memory["MemAvailable"].split()[0]) / 2**20
            if (free >= self.config["minimum_free_disk_gib"]
                    and available >= self.config["minimum_available_memory_gib"]):
                return
            persist(self.root / "CAPACITY_WAIT.json", {
                "at": utc_now(), "free_disk_gib": free, "available_memory_gib": available,
                "action": "Waiting; no unrelated process changes",
            })
            time.sleep(60)

    def process(self, directory: Path, command: list[str], kind: str) -> None:
        cache = directory / "process-cache"
        cache.mkdir(exist_ok=True)
        persist(directory / "COMMAND.json", {"argv": command, "cwd": str(self.source),
                                             "CUDA_VISIBLE_DEVICES": ""}, immutable=True)
        with (directory / "process.log").open("xb") as stream:
            process = subprocess.Popen(
                command, cwd=self.source, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(cache),
                     "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                     "PYTHONPATH": str(self.source)},
            )
        stat_path = Path(f"/proc/{process.pid}/stat")
        persist(directory / "PROCESS.json", {
            "pid": process.pid, "start_ticks": stat_path.read_text().split()[21],
            "started_at": utc_now(),
        }, immutable=True)
        with self.lock:
            self.active[directory.name] = {
                "kind": kind, "pid": process.pid, "directory": str(directory),
                "started_at": utc_now(),
            }
        try:
            code = process.wait()
            persist(directory / "PROCESS_EXIT.json", {"returncode": code}, immutable=True)
        finally:
            with self.lock:
                self.active.pop(directory.name, None)

    def harvest(self, directory: Path) -> None:
        path = directory / "PROCESS.json"
        if not path.is_file():
            raise GateError(f"Ambiguous callback launch; never repeat native actions: {directory}")
        record = load(path)
        proc = Path(f"/proc/{record['pid']}/stat")
        while proc.exists():
            try:
                fields = proc.read_text().split()
            except FileNotFoundError:
                break
            if fields[21] != record["start_ticks"] or fields[2] == "Z":
                break
            time.sleep(2)

    def callback_directory(self, kind: str, request: dict) -> tuple[Path, bool]:
        directory = self.root / "callbacks" / kind / request["request_id"]
        fresh = not directory.exists()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        persist(directory / "REQUEST.json", request, immutable=True)
        return directory, fresh

    def committed(self, directory: Path) -> dict | None:
        if not (directory / "RESULT.json").is_file() or not (directory / "USAGE.json").is_file():
            return None
        result = load(directory / "RESULT.json")
        request = load(directory / "REQUEST.json")
        if result["request_id"] != request["request_id"] or result["identity"] != request["identity"]:
            raise GateError("Committed callback identity mismatch")
        if result["usage"] != load(directory / "USAGE.json")["usage"]:
            raise GateError("Committed usage differs from callback result")
        seal = {name: file_hash(directory / name)
                for name in ("REQUEST.json", "RESULT.json", "USAGE.json")}
        persist(directory / "COMMITTED.json", seal, immutable=True)
        return result

    def commit(self, directory: Path, result: dict, attempts: list[dict]) -> None:
        persist(directory / "USAGE.json", {"usage": result["usage"], "attempts": attempts,
                                           "unknown_is_zero": False}, immutable=True)
        persist(directory / "RESULT.json", result, immutable=True)
        self.committed(directory)
        self.snapshot()

    @staticmethod
    def assert_completed(result: dict, *, executor: bool) -> dict:
        if (result["execution_status"] != "completed"
                or result["usage"]["total_tokens"] is None
                or (executor and (type(result["success"]) is not bool
                                  or result["native_score"] is None))):
            raise GateError("Callback is unresolved; retained costs cannot authorize the next phase")
        return result

    def descriptor(self, request: dict, directory: Path) -> dict | None:
        if request["prompt"] != request["task"]["public_prompt"]:
            raise GateError("The executor must preserve the original public task")
        text = request.get("system_prompt_append", "")
        residual = request.get("residual")
        if not text:
            if residual is not None:
                raise GateError("A no-op must have no residual metadata")
            return None
        if (not residual or residual["instruction"] != text
                or residual["application"] != "system_section"):
            raise GateError("Only one additive system-section residual is permitted")
        proof = []
        for path in (self.root / "callbacks/optimizer").glob("*/RESULT.json"):
            value = load(path)
            if value["execution_status"] == "completed" and value["instruction"].strip() == text:
                request_path = path.with_name("REQUEST.json")
                proof.append({
                    "optimizer_result_sha256": file_hash(path),
                    "optimizer_request_sha256": file_hash(request_path),
                    "provider_prompt_sha256": file_hash(path.with_name("PROVIDER_PROMPT.json")),
                    "supporting_task_ids": value["supporting_task_ids"],
                    "optimizer_task_ids": sorted(
                        row["task"]["task_id"] for row in load(request_path)["training_evidence"]
                    ),
                    "provider_usage": value["usage"],
                })
        if not proof:
            raise GateError("A nonempty residual needs a real completed TRAIN optimizer receipt")
        artifact = directory / "instruction.txt"
        if artifact.exists():
            if artifact.read_text() != text:
                raise GateError("Frozen residual artifact changed")
        else:
            with artifact.open("x") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        descriptor = {
            "path": str(artifact), "sha256": file_hash(artifact), "artifact_kind": "training-derived",
            "provenance": {"method": "CHISEL", "application": "system_section",
                           "partition_sha256": PARTITION_SHA256, "optimizer_receipts": proof,
                           "candidate_sha256": request["candidate_sha256"],
                           "final_policy_sha256": request.get("final_policy_sha256"),
                           "source_sha256": request.get("source_sha256")},
        }
        persist(directory / "INSTRUCTION_PROVENANCE.json", descriptor, immutable=True)
        return descriptor

    def execute(self, request: dict) -> dict:
        directory, fresh = self.callback_directory("executor", request)
        prior = self.committed(directory)
        if prior is not None:
            return self.assert_completed(prior, executor=True)
        self.check_sources()
        phase, ident = request["phase"], request["task"]["task_id"]
        if ident not in self.partition["splits"][phase]:
            raise GateError("Task does not belong to the requested phase")
        treatment = self.descriptor(request, directory)
        inherited = (
            phase == "train" and request["arm"] == "baseline" and treatment is None
            and ident == self.partition["splits"]["train"][0]
            and request["seed"] == self.config["execution_seeds"][0]
        )
        trial = self.pilot_trial if inherited else directory / "trial"
        retained_parent = None if inherited else retained_parent_trial(self.config, request, self.records)
        if retained_parent is not None:
            trial = retained_parent
            persist(directory / "RETAINED_PARENT_ADOPTION.json", {
                "summary": str(trial / "SUMMARY.json"),
                "summary_sha256": file_hash(trial / "SUMMARY.json"),
                "actor_replays": 0, "native_scores_replayed": 0,
                "eligibility": "identical native execution, prompt, task, seed, backbone; transport-only repair",
                "provider_capture_on_original_actor": False,
            }, immutable=True)
        if not inherited and retained_parent is None:
            job = {
                "task_id": ident, "phase": phase, "arm": request["arm"], "seed": request["seed"],
                "scenario": str(self.partition_root / "scenarios" / (ident + ".json")),
                "scenario_sha256": self.records[ident]["payload_sha256"],
                "public_prompt": request["prompt"], "chisel": treatment,
                "partition_sha256": PARTITION_SHA256,
                "native_protocol_sha256": self.protocol_hash,
                "candidate_sha256": request["candidate_sha256"],
                "final_policy_sha256": request.get("final_policy_sha256"),
                "source_sha256": request.get("source_sha256"),
                "freeze_manifest": str(self.root / "FROZEN_POLICY.json") if phase == "test" else None,
                "source_revision": self.config.get("source_revision"),
            }
            retained = getattr(self, "verifier_recoveries", {}).get(request["request_id"])
            if retained is not None:
                job["retained_native"] = retained
            persist(directory / "NATIVE_JOB.json", job, immutable=True)
            if not (trial / "SUMMARY.json").is_file():
                if fresh:
                    self.capacity()
                    self.process(directory, [
                        self.config["are_python"], "-u",
                        str(self.source / "scripts/run_deepseek_gaia2.py"),
                        "--config", str(self.root / "CONFIG.json"),
                        "--job", str(directory / "NATIVE_JOB.json"), "--run-dir", str(trial),
                    ], "official_deepseek_native_episode")
                else:
                    self.harvest(directory)
        if not (trial / "SUMMARY.json").is_file():
            raise GateError(f"Native callback has no settled receipt; never relaunch: {trial}")
        summary = load(trial / "SUMMARY.json")
        if summary["task_id"] != ident or summary["phase"] != phase or summary["seed"] != request["seed"]:
            raise GateError("Native callback source identity differs")
        result = native_callback_result(request, summary, trial)
        attempts = trial_attempts(summary, trial, phase)
        self.commit(directory, result, attempts)
        return self.assert_completed(result, executor=True)

    def optimize(self, request: dict) -> dict:
        if self.config.get("parent_train_only") is True:
            self.check_sources()
            capability = parent_capability_gate(
                request, self.config, set(self.partition["splits"]["train"]),
            )
            if capability is None:
                raise GateError("Parent-only execution requires the complete native capability gate")
            persist(self.root / "PARENT_CAPABILITY_GATE.json", capability, immutable=True)
            persist(self.root / "PARENT_TRAIN_COMPLETE.json", {
                "capability": capability, "optimizer_calls": 0,
                "validation_opened": False, "test_opened": False,
                "next": "matched native/CoT-only/CHISEL TRAIN with native fast proof"
                        if capability["passed"] else "TRAIN-only diagnosis of this frozen parent",
                "cumulative_token_limit": None,
            }, immutable=True)
            raise ParentTrainComplete()
        directory, fresh = self.callback_directory("optimizer", request)
        prior = self.committed(directory)
        if prior is not None:
            return self.assert_completed(prior, executor=False)
        self.check_sources()
        train_ids = set(self.partition["splits"]["train"])
        payload = optimizer_payload(request, train_ids)
        capability = parent_capability_gate(request, self.config, train_ids)
        if capability is not None:
            persist(self.root / "PARENT_CAPABILITY_GATE.json", capability, immutable=True)
            if not capability["passed"]:
                raise GateError(
                    "Shared TRAIN parent is not yet capable: "
                    f"{capability['successes']}/32 native successes; "
                    f"requires {capability['minimum_successes']}/32 before RSI or held-out runs",
                )
        persist(directory / "PROVIDER_PROMPT.json", payload, immutable=True)
        private = directory / "sdk"
        template = load(Path(self.config["runtime_template"]))
        template.update({
            "purpose": "benchmark", "run_dir": str(private),
            "session_id": "gaia-optimizer-" + request["request_id"][:35],
            "seed": self.config["split_seed"], "model": MODEL,
            "max_tokens": self.config["max_output_tokens"], "api_key_file": str(KEY_FILE),
            "tools": [], "bridge": None, "chisel": None, "arm": "baseline",
            "prompt": json.dumps(payload, sort_keys=True, ensure_ascii=False),
        })
        template.pop("cwd", None)
        persist(directory / "SDK_JOB.json", template, immutable=True)
        if not (private / "result.json").is_file():
            if fresh:
                self.capacity()
                self.process(directory, [
                    self.config["sdk_python"], "-u",
                    str(self.source / "integrations/deepseek_harness/runner.py"),
                    "--job", str(directory / "SDK_JOB.json"),
                ], "official_deepseek_train_optimizer")
            else:
                self.harvest(directory)
        if not (private / "result.json").is_file():
            raise GateError("Official optimizer lacks a settled receipt; no replay")
        native = load(private / "result.json")
        usage = sdk_usage(native)
        proposal = {"instruction": "", "supporting_task_ids": []}
        failure = None
        try:
            if native.get("status") != "completed" or native.get("usage_complete") is not True:
                raise GateError("Official optimizer generation did not complete")
            proposal = parse_proposal(native.get("final_response"), train_ids, request["limits"])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        result = {
            "request_id": request["request_id"], "identity": request["identity"],
            "execution_status": "blocked" if failure else "completed",
            **proposal, "usage": usage,
            "provenance": {
                "sdk_result": str(private / "result.json"),
                "sdk_result_sha256": file_hash(private / "result.json"),
                "provider_prompt_sha256": file_hash(directory / "PROVIDER_PROMPT.json"),
                "actual_optimizer_task_ids": sorted(train_ids), "optimizer_data_size": 32,
                "tools": [], "failed_proposal": failure, "cumulative_token_limit": None,
            },
        }
        accounting = load(private / "usage.json") if (private / "usage.json").is_file() else {}
        self.commit(directory, result, [{
            "attempt_id": str(private), "stage": "training_optimizer", "usage": usage,
            "provider_attempt_count": accounting.get("attempt_count"),
            "provider_native_usage": native.get("usage"),
            "known_reported_usage": accounting.get("known_reported_usage"),
            "reasoning_tokens": (native.get("usage") or {}).get("reasoningTokens"),
        }])
        return self.assert_completed(result, executor=False)

    def guarded(self, kind: str, request: dict) -> dict:
        try:
            return self.execute(request) if kind == "executor" else self.optimize(request)
        except Exception as error:
            directory = self.root / "callbacks" / kind / request["request_id"]
            persist(directory / "FAILURE.json", {
                "at": utc_now(), "type": type(error).__name__, "message": str(error),
                "native_actions_replayed": False, "cumulative_token_limit": None,
            })
            raise

    def snapshot(self) -> dict:
        with self.lock:
            rows, attempts = [], []
            for path in sorted((self.root / "callbacks").glob("*/*/COMMITTED.json")):
                directory = path.parent
                request, result = load(directory / "REQUEST.json"), self.committed(directory)
                rows.append({"request": request, "result": result})
                attempts += load(directory / "USAGE.json")["attempts"]
            inherited = self.root / "INHERITED_ATTEMPTS.json"
            if inherited.is_file():
                attempts += load(inherited)
            for path in (self.root / "callbacks").glob("*/*/REQUEST.json"):
                directory = path.parent
                if (directory / "COMMITTED.json").is_file():
                    continue
                request = load(path)
                attempts += unsettled_attempts(directory, directory.parent.name, request["phase"])
            attempts = unique_attempts(attempts)
            complete = complete_test(rows, set(self.partition["splits"]["test"]),
                                     self.config["test_execution_seeds"])
            stages = sorted({row["stage"] for row in attempts})
            expected = 96 * 3 * 2
            result = {
                "schema": SCHEMA, "at": utc_now(), "pid": os.getpid(), "phase": self.phase,
                "coverage": {phase: sum(row["request"]["phase"] == phase
                                        and "task" in row["request"] for row in rows)
                             for phase in ("train", "validation", "test")},
                "expected_final_test_observations": expected, "final_test_complete": complete,
                "final_native_means": {
                    arm: sum(row["result"]["native_score"] for row in rows
                             if row["request"]["phase"] == "test"
                             and row["request"]["arm"] == arm) / (expected // 2) if complete else None
                    for arm in ("baseline", "chisel")
                },
                "costs_by_stage": {
                    stage: usage_summary([row for row in attempts if row["stage"] == stage])
                    for stage in stages
                },
                "cumulative_unique_attempts": usage_summary(attempts),
                "optimizer_data_size": 32, "executor_workers": 2,
                "active": list(self.active.values()), "cumulative_token_limit": None,
                "benchmark_improvement_claim": None,
                "test_opened": (self.root / "FROZEN_POLICY.json").is_file() and self.phase == "test",
            }
            persist(self.root / "SUMMARY.json", result)
            return result

    def write_heartbeat(self) -> None:
        with self.lock:
            active = list(self.active.values())
        persist(self.root / "HEARTBEAT.json", {
            "at": utc_now(), "pid": os.getpid(), "phase": self.phase, "active": active,
            "controller_status": self.phase if self.phase in ("complete", "blocked") else "running",
            "successor": {
                "train": "full32-validation", "validation": "policy/source-freeze",
                "freeze": "sealed96x3-paired-test", "test": "next-independent-pair",
            }.get(self.phase),
            "CUDA_VISIBLE_DEVICES": "", "selected_gpus": [],
            "executor_workers": 2, "cumulative_token_limit": None,
        })

    def heartbeat(self) -> None:
        last_audit = 0.0
        while not self.stop.is_set():
            try:
                if time.monotonic() - last_audit >= self.config["capacity_audit_seconds"]:
                    persist(self.root / "capacity" / f"{time.time_ns()}.json",
                            resource_inventory(), immutable=True)
                    last_audit = time.monotonic()
                self.write_heartbeat()
                self.snapshot()
            except Exception as error:
                persist(self.root / "HEARTBEAT_ERROR.json", {
                    "at": utc_now(), "type": type(error).__name__, "message": str(error),
                })
            self.stop.wait(self.config["heartbeat_seconds"])

    def lifecycle(self) -> int:
        self.preflight()
        from phiagent.harness.deepseek_chisel import ChiselConfig, DeepSeekChiselController
        self.controller = DeepSeekChiselController(
            partition=self.partition, task_index=self.index,
            execute=lambda request: self.guarded("executor", request),
            optimize=lambda request: self.guarded("optimizer", request),
            execution_identity=self.identity, optimizer_identity=self.identity,
            run_id=self.root.name, config=ChiselConfig(
                application="system_section", optimization_iterations=2, executor_workers=2,
                execution_seeds=tuple(self.config["execution_seeds"]),
                test_execution_seeds=tuple(self.config["test_execution_seeds"]),
                native_metric="official_are_scenario_task_success",
            ),
        )
        heartbeat = threading.Thread(target=self.heartbeat, name="gaia-heartbeat", daemon=True)
        heartbeat.start()
        try:
            self.phase = "train"
            report = self.controller.train(self.tasks("train"))
            persist(self.root / "TRAIN_COMPLETE.json", report, immutable=True)
            self.phase = "validation"
            selection = self.controller.select_validation(self.tasks("validation"))
            if not all(row["complete_native_scores"] for row in selection["results"]):
                raise GateError("All validation scalars are required before source/policy freeze")
            persist(self.root / "VALIDATION_COMPLETE.json", selection, immutable=True)
            self.phase = "freeze"
            self.check_sources()
            policy = self.controller.freeze(source_hashes={
                "deepseek_harness": self.identity["runtime_manifest_sha256"],
                "executor": self.protocol_hash,
                "optimizer": canonical_hash({
                    "source": self.hashes["scripts/run_deepseek_gaia2_campaign.py"],
                    "identity": self.identity, "data": "all32_train_only",
                }),
                "scorer": canonical_hash({"are_sources": self.scorer_hashes,
                                          "partition": PARTITION_SHA256}),
                "all_frozen_sources": canonical_hash(self.hashes),
            })
            persist(self.root / "FROZEN_POLICY.json", policy.to_dict(), immutable=True)
            self.phase = "test"
            test = self.controller.evaluate_test(
                lambda: self.tasks("test"), policy_sha256=policy.policy_sha256,
                source_sha256=policy.source_sha256,
            )
            if not self.snapshot()["final_test_complete"]:
                raise GateError("Completion requires all 576 paired native verdicts and usage receipts")
            persist(self.root / "TEST_COMPLETE.json", test, immutable=True)
            self.phase = "complete"
            return 0
        except ParentTrainComplete:
            self.phase = "parent_complete"
            return 0
        finally:
            self.stop.set()
            heartbeat.join(timeout=65)
            persist(self.root / "CONTROLLER_REPORT.json", self.controller.report())
            self.snapshot()


def prepare_test_continuation(predecessor: Path, root: Path) -> None:
    require_remote()
    old_config = load(predecessor / "CONFIG.json")
    validate_config(old_config)
    origin = Path(old_config.get("frozen_origin_root", str(predecessor)))
    policy = load(origin / "FROZEN_POLICY.json")
    if policy["candidate"]["id"] != "noop":
        raise GateError("This recovery must preserve the declared frozen NOOP policy")
    verify_hashes(origin / "source", load(origin / "SOURCE_SHA256.json"))
    chain = list(old_config.get("predecessor_roots", [])) + [str(predecessor)]
    if len(set(chain)) != len(chain):
        raise GateError("Repeated predecessor would double-charge evidence")
    owned_path(root).mkdir(parents=True, exist_ok=False, mode=0o700)
    os.umask(0o077)
    config = {
        **old_config, "sealed_test_continuation": True, "frozen_origin_root": str(origin),
        "predecessor_roots": chain, "origin_policy_sha256": policy["policy_sha256"],
        "origin_policy_source_sha256": policy["source_sha256"],
        "artifact_root": str(root), "cumulative_token_limit": None,
    }
    persist(root / "CONFIG.json", config, immutable=True)
    hashes = freeze_source(root / "source")
    original = load(origin / "SOURCE_SHA256.json")
    allowed = {"scripts/run_deepseek_gaia2.py", "scripts/run_deepseek_gaia2_campaign.py",
               "tests/test_deepseek_gaia2_runner.py", "SOURCE_PROVENANCE.json"}
    differences = {name: {"original": original.get(name), "continuation": hashes.get(name)}
                   for name in set(original) | set(hashes) if original.get(name) != hashes.get(name)}
    if set(differences) - allowed:
        raise GateError("Recovery changed a frozen SDK, native backend, shared learner or configuration")
    persist(root / "SOURCE_SHA256.json", hashes, immutable=True)
    shutil.copyfile(origin / "FROZEN_POLICY.json", root / "FROZEN_POLICY.json")
    persist(root / "REPAIR_FREEZE.json", {
        "origin": str(origin), "predecessors": chain,
        "policy_sha256": policy["policy_sha256"], "policy_source_sha256": policy["source_sha256"],
        "original_policy_file_sha256": file_hash(origin / "FROZEN_POLICY.json"),
        "source_differences": differences,
        "changes": ["Allow the already-authorized data0 SDK artifact root",
                    "Resume only the frozen TEST plan from settled callback receipts",
                    "Attest malformed provider tool JSON without rewriting it or changing the SDK",
                    "Retry only settled SDK wire failures in new private native episodes",
                    "Retain unresolved historical provider costs without inventing zeros",
                    "Publish a terminal heartbeat when the controller completes or blocks",
                    "Treat metered native output limits as actor termination, then score normally",
                    "Score capped predecessor traces through official judge_only with no actor calls",
                    "Bound unchanged wire-failure retries to two per classification"],
        "unchanged": ["policy", "held-out membership", "paired seeds and arm order", "model",
                      "108 native capabilities", "measured clock", "native scorer", "SDK sources"],
        "training_or_selection_invoked": False, "cumulative_token_limit": None,
    }, immutable=True)
    persist(root / "PREPARATION.json", provenance(ROOT, sys.argv, config["split_seed"]),
            immutable=True)
    persist(root / "RESOURCE_SELECTION.json", resource_inventory(), immutable=True)


class FrozenTestContinuation(Campaign):
    """Read-only frozen-policy evaluator; never calls train/select/freeze."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.origin = Path(self.config["frozen_origin_root"])
        self.predecessors = [Path(path) for path in self.config["predecessor_roots"]]
        self.observations: dict[str, dict] = {}
        self.old_requests: dict[str, dict] = {}
        self.inherited_attempts: list[dict] = []
        self.adopted: dict[str, dict] = {}
        self.retry_admissions: dict[str, dict] = {}
        self.verifier_recoveries: dict[str, dict] = {}
        self.origin_lock = None
        self.policy = None

    @staticmethod
    def harvest_native(directory: Path, request: dict) -> dict | None:
        trial = directory / "trial"
        native_path, summary_path = trial / "native/RESULT.json", trial / "SUMMARY.json"
        if not native_path.is_file() or not summary_path.is_file():
            return None
        native, summary = load(native_path), load(summary_path)
        if native.get("exception") is not None or type(native.get("success")) is not bool:
            return None
        if (summary["task_id"] != request["task"]["task_id"]
                or summary["seed"] != request["seed"] or summary["phase"] != "test"):
            raise GateError("Settled native outcome differs from the requested TEST unit")
        recovered = {
            **summary, "valid_for_comparison": True, "native_score": int(native["success"]),
            "success": native["success"],
        }
        result = native_callback_result(request, recovered, trial)
        result["provenance"]["native_verdict_harvest"] = {
            "native_result": str(native_path), "native_result_sha256": file_hash(native_path),
            "original_summary_sha256": file_hash(summary_path),
            "new_model_or_judge_calls": 0, "unknown_usage_not_a_zero": True,
        }
        return result

    def preflight(self) -> None:
        from phiagent.harness.codex_chisel import FrozenPolicy
        from phiagent.harness.benchmark_partition import validate_partition

        require_remote()
        validate_config(self.config)
        self.origin_lock = (self.origin / "OWNER.lock").open("r")
        fcntl.flock(self.origin_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.check_sources()
        original_hashes = load(self.origin / "SOURCE_SHA256.json")
        verify_hashes(self.origin / "source", original_hashes)
        for path in self.predecessors:
            verify_hashes(path / "source", load(path / "SOURCE_SHA256.json"))
        self.policy = FrozenPolicy.from_dict(load(self.root / "FROZEN_POLICY.json"))
        document = self.policy.to_dict()
        if (self.policy.policy_sha256 != self.config["origin_policy_sha256"]
                or self.policy.source_sha256 != self.config["origin_policy_source_sha256"]
                or file_hash(self.root / "FROZEN_POLICY.json")
                != file_hash(self.origin / "FROZEN_POLICY.json")
                or self.identity != document["sources"]["execution_identity"]):
            raise GateError("Frozen policy/source/model identity changed")
        for name in ("TRAIN_COMPLETE.json", "VALIDATION_COMPLETE.json"):
            if not (self.origin / name).is_file():
                raise GateError("Original pre-test phase gate is absent")
        validate_partition(self.partition)
        for filename, key in (("SPLIT_MANIFEST.json", "partition_sha256"),
                              ("TRAIN_VALIDATION.json", "train_validation_sha256"),
                              ("SEALED_TEST.json", "sealed_test_sha256")):
            if file_hash(self.partition_root / filename) != self.config[key]:
                raise GateError("Frozen dataset membership changed")
        self.index = load(self.origin / "PUBLIC_HASH_INDEX.json")["tasks"]
        by_id = {row["task_id"]: row for row in self.index}
        if (canonical_hash(by_id) != document["task_index_sha256"]
                or canonical_hash(self.partition) != document["partition_sha256"]):
            raise GateError("Frozen public-task index or partition changed")
        distribution = importlib.metadata.distribution("meta-agents-research-environments")
        if distribution.version != ARE_VERSION:
            raise GateError("Native ARE version changed")
        self.native_source = Path(distribution.locate_file("are/simulation"))
        self.scorer_hashes = load(self.origin / "NATIVE_SOURCE_SHA256.json")
        verify_hashes(self.native_source, self.scorer_hashes)
        manifest = Path(self.config["filesystem_manifest"])
        if file_hash(manifest) != self.config["filesystem_manifest_sha256"]:
            raise GateError("Native public filesystem manifest changed")
        prefix = "gaia2_filesystem/demo_filesystem/"
        asset_rows = [row for row in load(manifest)["files"] if row["path"].startswith(prefix)]
        for row in asset_rows:
            path = Path(self.config["filesystem"]) / row["path"][len(prefix):]
            if path.is_symlink() or file_hash(path) != row["sha256"]:
                raise GateError("Frozen native filesystem changed")
        if (len(asset_rows) != 294 or sum(row["bytes"] for row in asset_rows)
                != self.config["filesystem_bytes"]):
            raise GateError("Native filesystem population differs")
        for predecessor in self.predecessors:
            inherited = predecessor / "INHERITED_ATTEMPTS.json"
            if inherited.is_file():
                self.inherited_attempts.extend(load(inherited))
            for path in sorted((predecessor / "callbacks").glob("*/*/REQUEST.json")):
                directory = path.parent
                if not (directory / "COMMITTED.json").is_file():
                    self.harvest(directory)
                if (directory / "COMMITTED.json").is_file():
                    request, result, attempts = verified_callback(directory)
                else:
                    request = load(path)
                    result = self.harvest_native(directory, request) if request["phase"] == "test" else None
                    if result is None:
                        raise GateError("Uncommitted native work cannot be replayed or silently dropped")
                    attempts = unsettled_attempts(directory, directory.parent.name, request["phase"])
                self.inherited_attempts.extend(attempts)
                if request["phase"] != "test":
                    continue
                ident = request["request_id"]
                if ident in self.old_requests and self.old_requests[ident] != request:
                    raise GateError("TEST identity was reused with conflicting inputs")
                self.old_requests[ident] = request
                if not valid_observation(result):
                    result = self.harvest_native(directory, request)
                if result is not None and valid_observation(result):
                    if ident in self.observations and self.observations[ident] != result:
                        raise GateError("Multiple distinct completed observations exist for one TEST unit")
                    self.observations[ident] = result
                    self.adopted[ident] = {
                        "callback": str(directory), "request_sha256": file_hash(path),
                        "result_sha256": canonical_hash(result),
                        "native_verdict_reused_without_model_calls": True,
                    }
                else:
                    actor = load(directory / "trial/sdk/result.json")
                    if sdk_output_limit(actor):
                        self.verifier_recoveries[ident] = output_limit_verification_admission(directory)
                    else:
                        self.retry_admissions[ident] = transport_retry_admission(directory)
        self.inherited_attempts = unique_attempts(self.inherited_attempts)
        for ident in self.observations:
            self.retry_admissions.pop(ident, None)
            self.verifier_recoveries.pop(ident, None)
        retry_policy = validate_wire_retry_policy(self.retry_admissions, [
            load(path / "RETRY_ADMISSION.json") for path in self.predecessors
            if (path / "RETRY_ADMISSION.json").is_file()
        ])
        entries = load(self.partition_root / "SEALED_TEST.json")["test"]
        if [row["id"] for row in entries] != self.partition["splits"]["test"]:
            raise GateError("Complete held-out membership must remain unchanged")
        tasks = []
        for row in entries:
            path = Path(row["path"])
            if file_hash(path) != self.records[row["id"]]["payload_sha256"]:
                raise GateError("Frozen held-out scenario bytes changed")
            prompt = initial_public_task(load(path))
            if hashlib.sha256(prompt.encode()).hexdigest() != by_id[row["id"]]["public_prompt_sha256"]:
                raise GateError("Frozen initial public task changed")
            tasks.append({"task_id": row["id"], "unit_id": row["id"], "public_prompt": prompt})
        if not self.old_requests:
            raise GateError("TEST continuation needs an original sealed TEST request")
        self.groups = frozen_test_plan(document, next(iter(self.old_requests.values())), tasks)
        self.planned = {q["request_id"]: q for group in self.groups for q in group}
        if len(self.planned) != 576:
            raise GateError("The full 96 x 3 x 2 TEST plan is required")
        for ident, request in self.old_requests.items():
            if self.planned.get(ident) != request:
                raise GateError("Continuation changed an original TEST request or its paired arm order")
        persist(self.root / "TEST_PLAN.json", self.groups, immutable=True)
        persist(self.root / "ADOPTED_OBSERVATIONS.json", self.adopted, immutable=True)
        persist(self.root / "RETRY_ADMISSION.json", self.retry_admissions, immutable=True)
        persist(self.root / "VERIFIER_RECOVERY.json", self.verifier_recoveries, immutable=True)
        persist(self.root / "RETRY_POLICY.json", retry_policy, immutable=True)
        persist(self.root / "INHERITED_ATTEMPTS.json", self.inherited_attempts, immutable=True)
        persist(self.root / "PHASE_GATE_REFERENCES.json", {
            name: {"path": str(self.origin / name), "sha256": file_hash(self.origin / name)}
            for name in ("TRAIN_COMPLETE.json", "VALIDATION_COMPLETE.json", "FROZEN_POLICY.json")
        }, immutable=True)
        persist(self.root / "PREFLIGHT.json", {
            "status": "PASSED", "mode": "frozen_test_only",
            "policy_sha256": self.policy.policy_sha256,
            "policy_source_sha256": self.policy.source_sha256,
            "inherited_completed_test_observations": len(self.observations),
            "settled_transport_retries": sorted(
                ident for ident, proof in self.retry_admissions.items()
                if proof["classification"] == "terminal_official_sdk_messages_transport_failure"
            ),
            "settled_infrastructure_retries": {
                ident: proof["classification"] for ident, proof in self.retry_admissions.items()
            },
            "retained_output_limit_verifications": sorted(self.verifier_recoveries),
            "output_limit_actor_resampling": False,
            "expected_test_observations": 576, "independent_pairs": 288,
            "native_capabilities": 108, "training_or_reselection": False,
            "raw_artifacts_duplicated": False, "cumulative_token_limit": None,
        }, immutable=True)

    def execute(self, request: dict) -> dict:
        if self.policy is None or self.planned.get(request["request_id"]) != request:
            raise GateError("Only the exact frozen TEST plan may execute")
        ident = request["request_id"]
        if ident in self.observations:
            return self.observations[ident]
        if (ident in self.old_requests and ident not in self.retry_admissions
                and ident not in self.verifier_recoveries):
            raise GateError("An unresolved predecessor lacks safe retry admission")
        try:
            result = super().execute(request)
        except GateError:
            directory = self.root / "callbacks/executor" / ident
            result = self.harvest_native(directory, request)
            if result is None:
                raise
            persist(self.root / "harvested" / (ident + ".json"), result, immutable=True)
        if not valid_observation(result):
            raise GateError("A new native TEST verdict remains unresolved")
        with self.lock:
            self.observations[ident] = result
        self.snapshot()
        return result

    def snapshot(self) -> dict:
        with self.lock:
            observations = dict(self.observations)
            attempts = list(self.inherited_attempts)
            for path in sorted((self.root / "callbacks").glob("*/*/REQUEST.json")):
                directory = path.parent
                request = load(path)
                if (directory / "COMMITTED.json").is_file():
                    _, result, incurred = verified_callback(directory)
                    attempts.extend(incurred)
                    if valid_observation(result):
                        observations[request["request_id"]] = result
                else:
                    attempts.extend(unsettled_attempts(directory, directory.parent.name, "test"))
                harvested = self.root / "harvested" / (request["request_id"] + ".json")
                if harvested.is_file():
                    observations[request["request_id"]] = load(harvested)
            attempts = unique_attempts(attempts)
            planned = getattr(self, "planned", {})
            complete = len(observations) == 576 and set(observations) == set(planned)
            stages = sorted({row["stage"] for row in attempts})
            summary = {
                "schema": SCHEMA, "at": utc_now(), "pid": os.getpid(), "phase": self.phase,
                "mode": "frozen_test_only_continuation", "origin": str(self.origin),
                "coverage": {"train": 96, "validation": 32, "test": len(observations)},
                "inherited_completed_test": len(self.adopted),
                "new_completed_test": len(set(observations) - set(self.adopted)),
                "expected_final_test_observations": 576, "final_test_complete": complete,
                "policy_sha256": self.config["origin_policy_sha256"],
                "policy_source_sha256": self.config["origin_policy_source_sha256"],
                "training_or_reselection": False, "test_opened": True,
                "final_native_means": {
                    arm: (sum(value["native_score"] for ident, value in observations.items()
                              if planned[ident]["arm"] == arm) / 288 if complete else None)
                    for arm in ("baseline", "chisel")
                },
                "costs_by_stage": {stage: usage_summary([
                    row for row in attempts if row["stage"] == stage
                ]) for stage in stages},
                "cumulative_unique_attempts": usage_summary(attempts),
                "active": list(self.active.values()), "cumulative_token_limit": None,
            }
            persist(self.root / "SUMMARY.json", summary)
            return summary

    def lifecycle(self) -> int:
        heartbeat = None
        try:
            self.preflight()
            self.phase = "test"
            heartbeat = threading.Thread(target=self.heartbeat, name="sealed-test-heartbeat", daemon=True)
            heartbeat.start()
            pending_groups = [
                group for group in self.groups
                if any(q["request_id"] not in self.observations for q in group)
            ]

            def run_pair(group):
                for request in group:
                    self.guarded("executor", request)

            failure = None
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = {}
                iterator = iter(pending_groups)
                for _ in range(2):
                    group = next(iterator, None)
                    if group is not None:
                        pending[pool.submit(run_pair, group)] = group
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        try:
                            future.result()
                        except Exception as error:
                            failure = failure or error
                    if failure is None:
                        for _ in done:
                            group = next(iterator, None)
                            if group is not None:
                                pending[pool.submit(run_pair, group)] = group
                if failure is not None:
                    raise failure
            self.check_sources()
            self.policy.verify_sources()
            if not self.snapshot()["final_test_complete"]:
                raise GateError("Frozen TEST remains incomplete")
            persist(self.root / "TEST_COMPLETE.json", {
                "policy_sha256": self.policy.policy_sha256,
                "source_sha256": self.policy.source_sha256, "observations": self.observations,
                "selection_or_optimization_from_test": False,
            }, immutable=True)
            self.phase = "complete"
            return 0
        finally:
            self.stop.set()
            if heartbeat:
                heartbeat.join(timeout=65)
            self.snapshot()
            if self.origin_lock:
                self.origin_lock.close()


def launch(args) -> dict:
    root = owned_path(args.run_dir.resolve())
    if args.resume:
        config = load(root / "CONFIG.json")
        validate_config(config)
        verify_hashes(root / "source", load(root / "SOURCE_SHA256.json"))
        with (root / "OWNER.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
    elif args.continue_from:
        prepare_test_continuation(args.continue_from.resolve(), root)
        config = load(root / "CONFIG.json")
    else:
        config = load(args.config)
        prepare(config, root, args.pilot_summary, args.prior_summary)
    command = [
        config["are_python"], "-u", str(root / "source/scripts/run_deepseek_gaia2_campaign.py"),
        "--run-dir", str(root), "--worker",
    ]
    log = root / f"controller-{time.time_ns()}.log"
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command, cwd=root / "source", stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "source")},
        )
    receipt = {"pid": process.pid, "command": command, "at": utc_now(), "log": str(log),
               "run_dir": str(root), "resume": args.resume, "start_new_session": True}
    persist(root / f"LAUNCH-{time.time_ns()}.json", receipt, immutable=True)
    persist(root / "PID.json", receipt)
    time.sleep(3)
    if process.poll() is not None:
        raise GateError(f"Controller exited during launch; see {log}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pilot-summary", type=Path)
    parser.add_argument("--prior-summary", action="append", type=Path, default=[])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require_remote()
    if args.detach and not args.worker:
        if not args.resume and not args.continue_from and (
            args.config is None or args.pilot_summary is None
        ):
            parser.error("New campaigns require --config and a real --pilot-summary")
        print(json.dumps(launch(args), sort_keys=True))
        return 0
    if not args.worker or ROOT != args.run_dir / "source":
        parser.error("Use --detach; workers must execute their own frozen source")
    with (args.run_dir / "OWNER.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = load(args.run_dir / "CONFIG.json")
        campaign = (FrozenTestContinuation(args.run_dir)
                    if config.get("sealed_test_continuation") else Campaign(args.run_dir))
        try:
            return campaign.lifecycle()
        except BaseException as error:
            campaign.phase = "blocked"
            persist(args.run_dir / "BLOCKER.json", {
                "at": utc_now(), "type": type(error).__name__, "message": str(error),
                "native_actions_replayed": False, "cumulative_token_limit": None,
                "resume": "Replay committed callbacks; never retry an unresolved native mutation",
            })
            campaign.snapshot()
            raise
        finally:
            campaign.write_heartbeat()


if __name__ == "__main__":
    raise SystemExit(main())
