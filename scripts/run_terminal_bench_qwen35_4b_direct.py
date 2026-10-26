#!/usr/bin/env python3
"""Run frozen Terminal-Bench 2.0 trials against a Qwen OpenAI-compatible service."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import time
import tomllib
import traceback
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import phiagent.learning.stage_aware_adm as stage_aware_adm_module
import phiagent.learning.residual_meta_harness as residual_meta_harness_module
from phiagent.learning.residual_meta_harness import (
    PolicySession as ResidualPolicySession,
    parse_policy as parse_residual_policy,
    policy_to_dict as residual_policy_to_dict,
)
from phiagent.learning.stage_aware_adm import (
    PolicySession as ADMPolicySession,
    Stage as ADMStage,
    compact_messages as compact_adm_messages,
    controller_messages as adm_controller_messages,
    infer_stage as infer_adm_stage,
    operation_response_format as adm_operation_response_format,
    parse_operation_response as parse_adm_operation_response,
    parse_policy as parse_adm_policy,
    policy_to_dict as adm_policy_to_dict,
)


EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "exec_in_sandbox",
        "description": (
            "Run one non-interactive shell command in the task container. "
            "Use it to inspect files, edit files, run programs, and validate the solution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

BASE_SYSTEM_PROMPT = """\
You are an autonomous terminal agent solving one task in an isolated Linux container.
Use exec_in_sandbox to inspect and modify the task environment. Work carefully and
persist until the requested artifacts and behavior are complete. Run relevant checks
before finishing. Do not merely describe commands or a solution: execute the work in
the container. When the task is complete, return a short final response.
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def repair_parent_sha256(config: Mapping[str, Any]) -> str:
    """Bind behavior, not service paths or the optional repair component."""

    agent = deepcopy(config["agent"])
    agent.pop("counterfactual_repair", None)
    agent.pop("public_progress", None)
    agent.pop("evidence_contracts", None)
    agent.pop("consumer_evidence", None)
    agent.pop("action_capsules", None)
    observed_budgets = {
        task["name"]: task.get("runtime", {}) for task in config.get("tasks", [])
    }
    budgets = config.get("chisel_provenance", {}).get(
        "task_runtime_budgets", observed_budgets
    )
    if any(budgets.get(name) != budget for name, budget in observed_budgets.items()):
        raise ValueError("task runtime differs from the frozen common repair budget")
    behavior = {
            "agent": agent,
            "model": {
                "repository": config["model"]["repository"],
                "revision": config["model"]["revision"],
                "temperature": float(config["model"]["temperature"]),
                "seed": int(config["model"]["seed"]),
                "max_output_tokens": int(config["model"]["max_output_tokens"]),
            },
            "official_source_revision": config.get("official_source_revision"),
            "mode": config.get("mode", "agent"),
            "task_runtime_budgets": budgets,
        }
    if config.get("actor_transport") is not None:
        behavior["actor_transport"] = dict(config["actor_transport"])
    return canonical_sha256(behavior)


def public_progress_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        relative: file_sha256(root / relative)
        for relative in (
            "scripts/terminal_public_progress.py",
            "phiagent/learning/public_recurrence.py",
            "phiagent/learning/counterfactual_repair.py",
        )
    }


def evidence_contract_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        relative: file_sha256(root / relative)
        for relative in (
            "scripts/terminal_evidence_contracts.py",
            "phiagent/learning/evidence_contracts.py",
        )
    }


def consumer_evidence_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        relative: file_sha256(root / relative)
        for relative in (
            "scripts/terminal_consumer_evidence.py",
            "phiagent/learning/consumer_evidence.py",
            "phiagent/learning/evidence_contracts.py",
        )
    }


def prepare_actor_transport(
    config: Mapping[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    transport = config.get("actor_transport")
    if transport is None:
        return payload, None
    if not isinstance(transport, Mapping) or set(transport) != {
        "tool_id_normalization", "normalizer_sha256"
    }:
        raise ValueError("actor_transport must be the closed common normalization contract")
    if transport["tool_id_normalization"] != "first-definition-linkage/v1":
        raise ValueError("unsupported common tool-ID normalization")
    import scripts.terminal_prefix_fidelity as normalizer

    source_hash = file_sha256(Path(normalizer.__file__).resolve())
    if transport["normalizer_sha256"] != source_hash:
        raise RuntimeError("common actor normalizer differs from its frozen source")
    normalized, mapping = normalizer.normalize_tool_ids(payload)
    return normalized, {
        "schema_version": "phiagent-terminal-common-transport-trace/v1",
        "source_sha256": source_hash,
        "original_payload_sha256": canonical_sha256(payload),
        "transmitted_payload_sha256": canonical_sha256(normalized),
        "opaque_linkage_mapping": mapping,
        "semantic_commands_or_observations_changed": False,
        "model_determinism_claim": False,
    }


def residual_feature_extractor_sha256() -> str:
    """Bind residual features to the runner and stage-inference sources."""

    return canonical_sha256(
        {
            "runner": file_sha256(Path(__file__).resolve()),
            "stage_aware_adm": file_sha256(
                Path(stage_aware_adm_module.__file__).resolve()
            ),
        }
    )


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def tree_sha256(root: Path) -> str:
    return canonical_sha256(tree_manifest(root))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_nvidia_csv(command: list[str], field_names: list[str]) -> list[dict[str, Any]]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(field_names):
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        rows.append(dict(zip(field_names, values, strict=True)))
    return rows


def gpu_preflight(
    config: Mapping[str, Any],
    service_process_ids: set[int],
) -> dict[str, Any]:
    inventory = parse_nvidia_csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        [
            "index",
            "uuid",
            "name",
            "memory_total_mib",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_percent",
        ],
    )
    processes = parse_nvidia_csv(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        ["pid", "gpu_uuid", "used_memory_mib"],
    )
    expected_index = str(config["host"]["physical_gpu_index"])
    expected_uuid = str(config["host"]["expected_gpu_uuid"])
    selected = [
        row
        for row in inventory
        if row["index"] == expected_index and row["uuid"] == expected_uuid
    ]
    service_rows = [
        row
        for row in processes
        if int(row["pid"]) in service_process_ids and row["gpu_uuid"] == expected_uuid
    ]
    if len(selected) != 1:
        raise RuntimeError("expected physical GPU index/UUID pair is unavailable")
    if not service_rows:
        raise RuntimeError("no model service process is resident on the expected GPU")
    return {
        "captured_at_utc": utc_now(),
        "selected_physical_gpu": selected[0],
        "model_service_processes": service_rows,
        "model_service_process_ids": sorted(service_process_ids),
        "inventory": inventory,
        "compute_processes": processes,
    }


def post_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read().decode())
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from model service: {body}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("model service returned a non-object")
    return value


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        value = json.loads(response.read().decode())
    if not isinstance(value, dict):
        raise RuntimeError("model service returned a non-object")
    return value


def model_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise RuntimeError("assistant content must be a string or null")
    return content


def model_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = message.get("tool_calls")
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError("assistant tool_calls must be an array of objects")
    return value


def truncate_tool_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n\n...[tool output truncated by harness]...\n\n"
    remaining = limit - len(marker)
    return value[: remaining // 2] + marker + value[-(remaining - remaining // 2) :]


def execute_tool(
    container_name: str,
    command: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    inner_timeout_seconds = max(1, math.ceil(timeout_seconds))
    docker_command = [
        "docker",
        "exec",
        container_name,
        "timeout",
        "--signal=TERM",
        "--kill-after=5s",
        f"{inner_timeout_seconds}s",
        "bash",
        "-lc",
        command,
    ]
    try:
        completed = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=inner_timeout_seconds + 15,
            check=False,
        )
        return {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "duration_seconds": time.monotonic() - started,
            "command": command,
            "docker_command": docker_command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": completed.returncode in {124, 137},
            "timeout_enforced_inside_container": True,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        return {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "duration_seconds": time.monotonic() - started,
            "command": command,
            "docker_command": docker_command,
            "exit_code": 124,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timed_out": True,
            "timeout_enforced_inside_container": True,
            "outer_docker_timeout": True,
        }


def tool_result_for_model(result: Mapping[str, Any], limit: int) -> str:
    text = (
        f"exit_code={result['exit_code']}\n"
        f"timed_out={str(result['timed_out']).lower()}\n"
        f"stdout:\n{result['stdout']}\n"
        f"stderr:\n{result['stderr']}"
    )
    return truncate_tool_output(text, limit)


def parse_tool_command(tool_call: Mapping[str, Any]) -> tuple[str, str]:
    call_id = tool_call.get("id")
    function = tool_call.get("function")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool call has no nonempty id")
    if not isinstance(function, Mapping):
        raise ValueError("tool call has no function object")
    if function.get("name") != "exec_in_sandbox":
        raise ValueError(f"unsupported tool: {function.get('name')!r}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("tool arguments are not JSON text")
    value = json.loads(arguments)
    if not isinstance(value, Mapping) or not isinstance(value.get("command"), str):
        raise ValueError("exec_in_sandbox requires a string command")
    if not value["command"].strip():
        raise ValueError("exec_in_sandbox command is empty")
    return call_id, value["command"]


def command_fingerprint(command: str) -> str:
    return hashlib.sha256(command.strip().encode()).hexdigest()


def command_shape_fingerprint(command: str) -> str:
    normalized = re.sub(
        r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\n.*?\n\1",
        "<<HEREDOC",
        command,
        flags=re.DOTALL,
    )
    normalized = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", normalized)
    normalized = re.sub(r"\b\d+\b", "<N>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def duplicate_command_rejection(
    *,
    command: str,
    previous_command_fingerprint: str | None,
    policy: str,
) -> str | None:
    if policy == "allow":
        return None
    if policy != "reject_consecutive_exact":
        raise ValueError(f"unsupported duplicate command policy: {policy}")
    if command_fingerprint(command) != previous_command_fingerprint:
        return None
    return (
        "Harness policy rejected an exact consecutive duplicate command. "
        "Inspect the prior result, issue a materially different command, or validate "
        "the current artifact instead of repeating the same action."
    )


def context_overflow_retry_tokens(
    error: str,
    *,
    current_max_tokens: int,
    margin_tokens: int,
) -> int | None:
    context_match = re.search(r"maximum context length is (\d+) tokens", error)
    input_match = re.search(r"prompt contains at least (\d+) input tokens", error)
    if context_match is None or input_match is None:
        return None
    if margin_tokens < 0:
        raise ValueError("context overflow margin must be non-negative")
    context_tokens = int(context_match.group(1))
    input_tokens = int(input_match.group(1))
    reported_headroom = context_tokens - input_tokens - margin_tokens
    retry_tokens = min(reported_headroom, max(1, current_max_tokens // 2))
    if retry_tokens < 1 or retry_tokens >= current_max_tokens:
        return None
    return retry_tokens


def compact_tool_messages_for_context(
    messages: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("context tool-message limit must be positive")
    compacted = []
    for message in messages:
        row = dict(message)
        if row.get("role") == "tool":
            row["content"] = truncate_tool_output(str(row.get("content") or ""), limit)
        compacted.append(row)
    return compacted


def compact_tool_history(
    messages: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    preserve_recent: int,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("history tool-message limit must be positive")
    if preserve_recent < 0:
        raise ValueError("preserve_recent must be non-negative")
    tool_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    ]
    preserved = set(tool_indices[-preserve_recent:]) if preserve_recent else set()
    compacted = []
    for index, message in enumerate(messages):
        row = dict(message)
        if row.get("role") == "tool" and index not in preserved:
            row["content"] = truncate_tool_output(str(row.get("content") or ""), limit)
        compacted.append(row)
    return compacted


def parse_dynamic_adm_profiles(value: object) -> dict[str, dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "ADD",
        "DELETE",
        "MODIFY",
    }:
        raise TypeError("agent.dynamic_adm_profiles must define ADD/DELETE/MODIFY")
    required = {
        "system_prompt_suffix",
        "duplicate_command_policy",
        "no_progress_policy",
        "no_progress_patience",
        "no_progress_reminder",
        "reserve_validation_turns",
        "validation_reminder",
        "history_tool_characters",
        "history_preserve_recent",
    }
    parsed = {}
    for operation, raw in value.items():
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise TypeError(f"dynamic ADM {operation} profile fields differ")
        profile = dict(raw)
        if not isinstance(profile["system_prompt_suffix"], str):
            raise TypeError("dynamic ADM profile suffix must be a string")
        if profile["duplicate_command_policy"] not in {
            "allow",
            "reject_consecutive_exact",
        }:
            raise ValueError("dynamic ADM duplicate-command policy is invalid")
        if profile["no_progress_policy"] not in {
            "off",
            "remind",
            "remind_then_terminate",
        }:
            raise ValueError("dynamic ADM no-progress policy is invalid")
        for name in (
            "no_progress_patience",
            "reserve_validation_turns",
            "history_tool_characters",
            "history_preserve_recent",
        ):
            if type(profile[name]) is not int or profile[name] < 0:
                raise TypeError(f"dynamic ADM {name} must be a non-negative integer")
        for name in ("no_progress_reminder", "validation_reminder"):
            if not isinstance(profile[name], str):
                raise TypeError(f"dynamic ADM {name} must be a string")
        if (
            profile["no_progress_policy"] != "off"
            and (
                profile["no_progress_patience"] < 1
                or not profile["no_progress_reminder"].strip()
            )
        ):
            raise ValueError("dynamic ADM no-progress profile is incomplete")
        if (
            profile["reserve_validation_turns"]
            and not profile["validation_reminder"].strip()
        ):
            raise ValueError("dynamic ADM validation profile is incomplete")
        parsed[str(operation)] = profile
    return parsed


def request_adm_operation(
    *,
    model: Mapping[str, Any],
    instruction: str,
    session: ADMPolicySession,
    stage: object,
    recent_messages: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
) -> tuple[str | None, dict[str, Any]]:
    """Ask the same frozen actor model for one bounded meta-operation."""

    recent = json.dumps(
        [dict(message) for message in recent_messages[-4:]],
        ensure_ascii=False,
        sort_keys=True,
    )[-4000:]
    payload = {
        "model": model["request_model_id"],
        "messages": adm_controller_messages(
            session,
            stage=stage,
            task_text=instruction,
            trajectory_text=recent,
        ),
        "temperature": 0,
        "max_tokens": int(model.get("dynamic_adm_controller_max_tokens", 64)),
        "seed": int(model["seed"]),
        "response_format": adm_operation_response_format(),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = post_json(
        str(model["chat_completions_url"]),
        payload,
        timeout=timeout_seconds,
    )
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    message = first.get("message") if isinstance(first, Mapping) else None
    content = model_content(message) if isinstance(message, Mapping) else ""
    operation, parse_error = parse_adm_operation_response(content)
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise RuntimeError("dynamic ADM controller response lacks exact token usage")
    response_model = response.get("model")
    if response_model != model["expected_response_model_id"]:
        raise RuntimeError("dynamic ADM controller returned an unexpected model")
    usage_complete = all(type(usage.get(key)) is int and usage[key] >= 0
                         for key in ("prompt_tokens", "completion_tokens"))
    if "total_tokens" in usage:
        usage_complete = (usage_complete and type(usage["total_tokens"]) is int
                          and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"])
    record = {
        "request_sha256": canonical_sha256(payload),
        "response_sha256": canonical_sha256(response),
        "response_model": response_model,
        "raw_content": content,
        "parsed_operation": operation.value if operation is not None else None,
        "parse_error": parse_error,
        "usage_complete": usage_complete,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0),
        },
    }
    return (
        operation.value if operation is not None else None,
        record,
    )


def run_agent(
    *,
    config: Mapping[str, Any],
    instruction: str,
    container_name: str,
    trial_dir: Path,
    agent_timeout_seconds: float,
    task_spec: Mapping[str, Any] | None = None,
    tool_executor: Callable[[str, str, float], dict[str, Any]] | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    model = deepcopy(config["model"])
    agent = deepcopy(config["agent"])
    execute = tool_executor or execute_tool
    solpi_session = None
    solpi_value = agent.get("solpi")
    if solpi_value is not None:
        from scripts.terminal_solpi import TerminalSoLPiSession

        if not isinstance(solpi_value, Mapping):
            raise ValueError("solpi requires an explicit configuration object")
        if (solpi_value.get("observation_pack") or solpi_value.get("action_fusion")) and (
            any(agent.get(name) is not None for name in (
                "dual_terminal", "action_capsules", "evidence_contracts", "consumer_evidence",
            )) or agent.get("context_overflow_policy", {}).get("max_retries", 0)
        ):
            raise ValueError("solpi must not overlap committed-context adapters or lossy overflow retries")
        solpi_session = TerminalSoLPiSession(solpi_value, trial_dir)
    recovery_value = agent.get("recovery_routing")
    if recovery_value is not None:
        from scripts.terminal_recovery_routing import TerminalRecoveryRoutingSession

        if (any(agent.get(name) is not None for name in (
                "dual_terminal", "action_capsules", "evidence_contracts", "consumer_evidence",
                "jev", "counterfactual_repair", "public_progress", "dynamic_adm_policy",
                "residual_meta_policy",
        )) or agent.get("history_policy")
                or agent.get("context_overflow_policy", {}).get("max_retries", 0)
                or solpi_session is not None and (
                    solpi_session.pack is not None or solpi_session.action_fusion)):
            raise ValueError("Recovery routing must not overlap another history/action intervention")
        solpi_session = TerminalRecoveryRoutingSession(
            recovery_value, trial_dir, instruction=instruction,
        )
    jev_session = None
    if agent.get("jev") is not None:
        from scripts.terminal_jev import TerminalJevSession

        if (any(agent.get(name) is not None for name in (
                "dual_terminal", "action_capsules", "evidence_contracts", "consumer_evidence",
                "counterfactual_repair", "public_progress", "dynamic_adm_policy", "residual_meta_policy",
            )) or agent.get("tool_policy", {}).get("no_progress_policy", "off") != "off"
                or agent.get("context_overflow_policy", {}).get("max_retries", 0)
                or (solpi_session is not None and solpi_session.action_fusion)):
            raise ValueError("Jev continuation requires static controls and no overlapping action controller")
        jev_session = TerminalJevSession(agent["jev"])
    dual_session = None
    dual_value = agent.get("dual_terminal")
    if dual_value is not None:
        from scripts.dual_terminal_completion import DualTerminalSession

        if not isinstance(dual_value, Mapping) or set(dual_value) != {"mode", "config"}:
            raise ValueError("dual_terminal requires a closed frozen configuration")
        if any(agent.get(name) is not None for name in (
            "action_capsules", "counterfactual_repair", "public_progress",
            "evidence_contracts", "consumer_evidence", "dynamic_adm_policy", "residual_meta_policy",
        )) or agent.get("history_policy") or agent.get("context_overflow_policy", {}).get("max_retries", 0):
            raise ValueError("dual_terminal must not overlap legacy recovery or lossy history policies")
        dual_tool_policy = agent.get("tool_policy", {})
        if (
            dual_tool_policy.get("duplicate_command_policy", "allow") != "allow"
            or dual_tool_policy.get("no_progress_policy", "off") != "off"
            or dual_tool_policy.get("max_duplicate_rejections", 0)
            or dual_tool_policy.get("reserve_validation_turns", 0)
        ):
            raise ValueError("dual_terminal must not restore legacy guards or forced termination")
        dual_session = DualTerminalSession(instruction=instruction, **dual_value)
    capsule_session = None
    capsule_value = agent.get("action_capsules")
    if capsule_value is not None:
        from scripts.terminal_action_capsules import TerminalActionCapsules

        if not isinstance(capsule_value, Mapping) or set(capsule_value) != {"mode", "config"}:
            raise ValueError("action_capsules requires a closed frozen configuration")
        if any(agent.get(name) is not None for name in ("evidence_contracts", "consumer_evidence")):
            raise ValueError("action capsules must not replace evidence-contract snapshots")
        capsule_session = TerminalActionCapsules(**capsule_value)
    repair_session = None
    repair_value = agent.get("counterfactual_repair")
    progress_session = None
    progress_value = agent.get("public_progress")
    if progress_value is not None:
        from scripts.terminal_public_progress import PublicProgressSession

        if repair_value is not None or not isinstance(progress_value, Mapping):
            raise ValueError("public progress requires one explicit sidecar, not overlapping repairs")
        progress_session = PublicProgressSession(
            progress_value["patch"], mode=progress_value["mode"],
            task_id=str((task_spec or {}).get("name", trial_dir.name)),
            expected_parent_sha256=repair_parent_sha256(config),
            seed=int(model["seed"]), config=progress_value["config"],
        )
    if repair_value is not None:
        from phiagent.learning.counterfactual_repair import (
            RepairMode,
            RepairSession,
            parse_patch,
        )

        if not isinstance(repair_value, Mapping):
            raise TypeError("agent.counterfactual_repair must be an object")
        repair_session = RepairSession(
            parse_patch(repair_value["patch"]),
            mode=RepairMode(repair_value["mode"]),
            task_id=str((task_spec or {}).get("name", trial_dir.name)),
            expected_parent_sha256=repair_parent_sha256(config),
            seed=int(model["seed"]),
            random_window=int(repair_value.get("random_window", 8)),
        )
    evidence_session = None
    evidence_value = agent.get("evidence_contracts")
    if evidence_value is not None:
        from scripts.terminal_evidence_contracts import TerminalEvidenceContractSession

        if not isinstance(evidence_value, Mapping):
            raise TypeError("agent.evidence_contracts must be an explicit configuration object")
        if (repair_session is not None or progress_session is not None
                or agent.get("dynamic_adm_policy") is not None or agent.get("residual_meta_policy") is not None):
            raise ValueError("Evidence-contract control must not overlap another adaptive or repair controller")
        if agent.get("context_overflow_policy", {}).get("max_retries", 0):
            raise ValueError("Evidence-contract input must not be silently rewritten after its preservation check")
        evidence_session = TerminalEvidenceContractSession(
            instruction=instruction, mode=evidence_value["mode"], budget_bytes=evidence_value["budget_bytes"],
        )
    consumer_session = None
    consumer_value = agent.get("consumer_evidence")
    consumer_tools = []
    consumer_supplement = ""
    if consumer_value is not None:
        from scripts.terminal_consumer_evidence import (
            REQUEST_TOOL, SYSTEM_SUPPLEMENT, TerminalConsumerSession,
        )

        if not isinstance(consumer_value, Mapping) or set(consumer_value) != {
            "mode", "budget_bytes", "acquisition_enabled",
        }:
            raise ValueError("consumer_evidence requires one closed, explicit interface configuration")
        if any(agent.get(name) is not None for name in (
            "counterfactual_repair", "public_progress", "evidence_contracts",
            "dynamic_adm_policy", "residual_meta_policy",
        )):
            raise ValueError("Consumer evidence must not overlap another adaptive or repair controller")
        if agent.get("context_overflow_policy", {}).get("max_retries", 0):
            raise ValueError("Consumer evidence snapshots must not be rewritten after commitment")
        consumer_session = TerminalConsumerSession(**consumer_value)
        consumer_tools = [REQUEST_TOOL]
        consumer_supplement = SYSTEM_SUPPLEMENT
    runtime_override = (
        task_spec.get("runtime", {}) if isinstance(task_spec, Mapping) else {}
    )
    if not isinstance(runtime_override, Mapping):
        raise TypeError("task runtime override must be an object")
    for field in ("max_turns", "max_tool_output_characters"):
        if field in runtime_override:
            agent[field] = int(runtime_override[field])
    if "max_output_tokens" in runtime_override:
        model["max_output_tokens"] = int(runtime_override["max_output_tokens"])
    if "max_tool_calls" in runtime_override:
        agent.setdefault("tool_policy", {})["max_tool_calls"] = runtime_override["max_tool_calls"]
    if "reserve_validation_turns" in runtime_override:
        agent.setdefault("tool_policy", {})
        agent["tool_policy"]["reserve_validation_turns"] = int(
            runtime_override["reserve_validation_turns"]
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": str(agent["system_prompt"]) + consumer_supplement},
        {"role": "user", "content": instruction},
    ]
    started_at_utc = utc_now()
    started = time.monotonic()
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    tool_calls_count = 0
    tool_budget_rejections = 0
    environment_tool_calls_attempted = 0
    environment_tool_calls_completed = 0
    consumer_requests_total = 0
    final_content = ""
    stop_reason = "max_turns"
    protocol_failures: list[dict[str, Any]] = []
    policy_events: list[dict[str, Any]] = []
    tool_policy = agent.get("tool_policy", {})
    if not isinstance(tool_policy, Mapping):
        raise TypeError("agent.tool_policy must be an object")
    maximum_tool_calls = tool_policy.get("max_tool_calls")
    if maximum_tool_calls is not None and (
        type(maximum_tool_calls) is not int or maximum_tool_calls < 1
    ):
        raise ValueError("max_tool_calls must be a positive integer when specified")
    if solpi_session is not None and solpi_session.action_fusion and maximum_tool_calls is not None:
        raise ValueError("Action Fusion requires an uncapped command budget; do not bypass legacy tool-call caps")
    duplicate_policy = str(tool_policy.get("duplicate_command_policy", "allow"))
    if duplicate_policy not in {"allow", "reject_consecutive_exact"}:
        raise ValueError(f"unsupported duplicate command policy: {duplicate_policy}")
    reserve_validation_turns = int(tool_policy.get("reserve_validation_turns", 0))
    if reserve_validation_turns < 0 or reserve_validation_turns >= int(
        agent["max_turns"]
    ):
        raise ValueError("reserve_validation_turns must be within the turn budget")
    validation_reminder = str(tool_policy.get("validation_reminder", "")).strip()
    if reserve_validation_turns and not validation_reminder:
        raise ValueError("validation_reminder is required when turns are reserved")
    validation_reminder_inserted = False
    previous_command_fingerprint: str | None = None
    duplicate_command_rejections = 0
    max_duplicate_rejections = int(tool_policy.get("max_duplicate_rejections", 0))
    if max_duplicate_rejections < 0:
        raise ValueError("max_duplicate_rejections must be non-negative")
    duplicate_budget_triggered = False
    no_progress_policy = str(tool_policy.get("no_progress_policy", "off"))
    if no_progress_policy not in {"off", "remind", "remind_then_terminate"}:
        raise ValueError(f"unsupported no-progress policy: {no_progress_policy}")
    no_progress_patience = int(tool_policy.get("no_progress_patience", 0))
    if no_progress_policy != "off" and no_progress_patience < 1:
        raise ValueError("no_progress_patience must be positive when enabled")
    no_progress_reminder = str(
        tool_policy.get("no_progress_reminder", "")
    ).strip()
    if no_progress_policy != "off" and not no_progress_reminder:
        raise ValueError("no_progress_reminder is required when enabled")
    previous_command_shape: str | None = None
    consecutive_no_progress_events = 0
    no_progress_events = 0
    no_progress_reminders = 0
    no_progress_termination_triggered = False
    model_protocol_retries = int(tool_policy.get("model_protocol_retries", 0))
    if model_protocol_retries < 0 or model_protocol_retries > 3:
        raise ValueError("model_protocol_retries must be in 0..3")
    model_protocol_retry_count = 0
    history_policy = agent.get("history_policy", {})
    if not isinstance(history_policy, Mapping):
        raise TypeError("agent.history_policy must be an object")
    history_tool_characters = int(history_policy.get("tool_message_characters", 0))
    history_preserve_recent = int(
        history_policy.get("preserve_recent_tool_messages", 0)
    )
    if history_tool_characters < 0 or history_preserve_recent < 0:
        raise ValueError("history policy values must be non-negative")
    if history_preserve_recent and not history_tool_characters:
        raise ValueError("history tool-message limit is required")
    history_compaction_turns = 0
    context_policy = agent.get("context_overflow_policy", {})
    if not isinstance(context_policy, Mapping):
        raise TypeError("agent.context_overflow_policy must be an object")
    context_max_retries = int(context_policy.get("max_retries", 0))
    context_margin_tokens = int(context_policy.get("margin_tokens", 0))
    context_tool_message_characters = int(
        context_policy.get("tool_message_characters", 0)
    )
    if context_max_retries < 0 or context_margin_tokens < 0:
        raise ValueError("context overflow policy values must be non-negative")
    if context_max_retries and context_tool_message_characters < 1:
        raise ValueError(
            "tool_message_characters is required for context overflow retries"
        )
    context_budget_events: list[dict[str, Any]] = []
    dynamic_policy_value = agent.get("dynamic_adm_policy")
    dynamic_session: ADMPolicySession | None = None
    dynamic_decisions: list[dict[str, Any]] = []
    dynamic_controller_calls: list[dict[str, Any]] = []
    residual_policy_value = agent.get("residual_meta_policy")
    if residual_policy_value is not None and dynamic_policy_value is None:
        raise ValueError(
            "residual_meta_policy requires a residual-mode dynamic_adm_policy"
        )
    residual_session: ResidualPolicySession | None = None
    residual_decisions: list[dict[str, Any]] = []
    residual_default_active = False
    dynamic_compact_next = False
    last_adm_stage = None
    active_adm_decision = None
    initial_adm_profile: str | None = None
    previous_recovery_signal = False
    dynamic_profiles = parse_dynamic_adm_profiles(
        agent.get("dynamic_adm_profiles")
    )

    def activate_initial_profile(operation: str) -> None:
        nonlocal duplicate_policy
        nonlocal history_preserve_recent
        nonlocal history_tool_characters
        nonlocal history_policy
        nonlocal no_progress_patience
        nonlocal no_progress_policy
        nonlocal no_progress_reminder
        nonlocal reserve_validation_turns
        nonlocal tool_policy
        nonlocal validation_reminder
        nonlocal initial_adm_profile

        if dynamic_profiles is None:
            return
        profile = dynamic_profiles[operation]
        messages[0]["content"] = (
            str(messages[0]["content"]) + profile["system_prompt_suffix"]
        )
        duplicate_policy = str(profile["duplicate_command_policy"])
        no_progress_policy = str(profile["no_progress_policy"])
        no_progress_patience = int(profile["no_progress_patience"])
        no_progress_reminder = str(profile["no_progress_reminder"])
        reserve_validation_turns = int(profile["reserve_validation_turns"])
        validation_reminder = str(profile["validation_reminder"])
        history_tool_characters = int(profile["history_tool_characters"])
        history_preserve_recent = int(profile["history_preserve_recent"])
        initial_adm_profile = operation
        tool_policy = {
            "duplicate_command_policy": duplicate_policy,
            "no_progress_policy": no_progress_policy,
            "no_progress_patience": no_progress_patience,
            "no_progress_reminder": no_progress_reminder,
            "reserve_validation_turns": reserve_validation_turns,
            "validation_reminder": validation_reminder,
            "model_protocol_retries": model_protocol_retries,
        }
        history_policy = {
            "tool_message_characters": history_tool_characters,
            "preserve_recent_tool_messages": history_preserve_recent,
        }
        agent["tool_policy"] = tool_policy
        agent["history_policy"] = history_policy
        policy_events.append(
            {
                "event": "dynamic_adm_initial_profile",
                "operation": initial_adm_profile,
                "system_prompt_sha256": hashlib.sha256(
                    str(messages[0]["content"]).encode()
                ).hexdigest(),
                "tool_policy": tool_policy,
                "history_policy": history_policy,
            }
        )

    if dynamic_policy_value is not None:
        if not isinstance(dynamic_policy_value, Mapping):
            raise TypeError("agent.dynamic_adm_policy must be an object")
        dynamic_policy = parse_adm_policy(dynamic_policy_value)
        task_context = {
            "benchmark": "Terminal-Bench 2.0",
            "task_name": (
                str(task_spec.get("name"))
                if isinstance(task_spec, Mapping) and task_spec.get("name")
                else "unknown"
            ),
            "family": (
                str(task_spec.get("category", "unknown"))
                if isinstance(task_spec, Mapping)
                else "unknown"
            ),
            "difficulty": (
                str(task_spec.get("difficulty", "unknown"))
                if isinstance(task_spec, Mapping)
                else "unknown"
            ),
            "repository_history_task": bool(
                re.search(
                    r"\b(?:git|repository|commit|history|redact|sanitize)\b",
                    instruction,
                    re.IGNORECASE,
                )
            ),
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        }
        dynamic_session = ADMPolicySession(
            dynamic_policy,
            task_id=str(task_context["task_name"]),
            task_context=task_context,
        )
        if residual_policy_value is not None:
            if not isinstance(residual_policy_value, Mapping):
                raise TypeError("agent.residual_meta_policy must be an object")
            if dynamic_policy.mode != "residual":
                raise ValueError(
                    "residual_meta_policy requires dynamic ADM mode residual"
                )
            residual_policy = parse_residual_policy(residual_policy_value)
            if (
                residual_policy.feature_extractor_sha256
                != residual_feature_extractor_sha256()
            ):
                raise ValueError(
                    "residual feature extractor differs from frozen policy"
                )
            observed_defaults = {
                stage: operation.value
                for stage, operation in residual_policy.defaults_by_stage.items()
            }
            if (
                set(observed_defaults) != {stage.value for stage in ADMStage}
                or set(observed_defaults.values()) != {"KEEP"}
                or any(
                    head.operation.value == "KEEP"
                    for head in residual_policy.override_heads
                )
            ):
                raise ValueError(
                    "residual integration requires KEEP defaults and ADM overrides"
                )
            residual_session = ResidualPolicySession(
                residual_policy,
                task_id=str(task_context["task_name"]),
                benchmark="Terminal-Bench",
            )
        elif dynamic_policy.mode == "residual":
            raise ValueError(
                "dynamic ADM mode residual requires residual_meta_policy"
            )
        if dynamic_policy.mode != "model":
            proposed_initial = None
            apply_initial_edit = True
            if residual_session is not None:
                residual_initial = residual_session.choose(
                    stage="INITIAL",
                    features={
                        "turn_fraction": 0.0,
                        "remaining_fraction": 1.0,
                        "recovery_signal": False,
                        "no_progress_fraction": 0.0,
                        "action_fraction": 0.0,
                        "family_debugging": task_context["family"] == "debugging",
                        "family_security": task_context["family"] == "security",
                        "repository_history_task": bool(
                            task_context["repository_history_task"]
                        ),
                    },
                )
                residual_decisions.append(dict(residual_initial.trace))
                if residual_initial.used_default:
                    apply_initial_edit = False
                    residual_default_active = True
                    last_adm_stage = ADMStage.INITIAL
                else:
                    proposed_initial = residual_initial.operation.value
            if apply_initial_edit:
                active_adm_decision = dynamic_session.choose(
                    stage="INITIAL",
                    proposed_operation=proposed_initial,
                    features={
                        "turn_index": 0,
                        "remaining_turns": int(agent["max_turns"]),
                        "action_count": 0,
                        "previous_recovery_signal": False,
                        "consecutive_no_progress_events": 0,
                    },
                )
                dynamic_decisions.append(dict(active_adm_decision.trace))
                dynamic_compact_next = active_adm_decision.compact_history
                last_adm_stage = active_adm_decision.stage
        if active_adm_decision is not None:
            activate_initial_profile(active_adm_decision.operation.value)

    for turn_index in range(int(agent["max_turns"])):
        if cancellation_check is not None:
            cancellation_check()
        remaining = agent_timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "agent_timeout"
            break
        if (
            reserve_validation_turns
            and not validation_reminder_inserted
            and turn_index == int(agent["max_turns"]) - reserve_validation_turns
        ):
            messages.append({"role": "user", "content": validation_reminder})
            policy_events.append(
                {
                    "event": "validation_reminder_inserted",
                    "turn": turn_index + 1,
                    "reserved_turns": reserve_validation_turns,
                    "message_sha256": hashlib.sha256(
                        validation_reminder.encode()
                    ).hexdigest(),
                }
            )
            validation_reminder_inserted = True
        current_adm_stage = None
        payload_messages = messages
        if dynamic_session is not None:
            current_adm_stage = infer_adm_stage(
                turn_index=turn_index,
                max_turns=int(agent["max_turns"]),
                action_count=tool_calls_count,
                recovery_signal=previous_recovery_signal,
                verification_reserve=min(
                    max(1, reserve_validation_turns or 3),
                    int(agent["max_turns"]) - 1,
                ),
            )
            if current_adm_stage != last_adm_stage:
                proposed_adm_operation = None
                controller_call = None
                residual_decision = None
                if dynamic_session.policy.mode == "model":
                    if solpi_session is not None:
                        solpi_session.record_request()
                        write_json(trial_dir / "agent/solpi.json", solpi_session.report())
                    proposed_adm_operation, controller_call = request_adm_operation(
                        model=model,
                        instruction=instruction,
                        session=dynamic_session,
                        stage=current_adm_stage,
                        recent_messages=messages,
                        timeout_seconds=min(
                            float(model["request_timeout_seconds"]),
                            remaining,
                        ),
                    )
                    if solpi_session is not None:
                        solpi_session.record_response(
                            controller_call["usage"], role="controller",
                            complete=controller_call["usage_complete"],
                        )
                    controller_call.update(
                        {
                            "turn": turn_index + 1,
                            "stage": current_adm_stage.value,
                        }
                    )
                    dynamic_controller_calls.append(controller_call)
                    controller_usage = controller_call["usage"]
                    input_tokens += int(controller_usage["prompt_tokens"])
                    output_tokens += int(controller_usage["completion_tokens"])
                elif residual_session is not None:
                    max_turns = int(agent["max_turns"])
                    residual_decision = residual_session.choose(
                        stage=current_adm_stage.value,
                        features={
                            "turn_fraction": turn_index / max_turns,
                            "remaining_fraction": (
                                max_turns - turn_index
                            )
                            / max_turns,
                            "recovery_signal": previous_recovery_signal,
                            "no_progress_fraction": min(
                                1.0,
                                consecutive_no_progress_events / 3,
                            ),
                            "action_fraction": min(
                                1.0,
                                tool_calls_count / max_turns,
                            ),
                            "family_debugging": (
                                dynamic_session.task_context.get("family")
                                == "debugging"
                            ),
                            "family_security": (
                                dynamic_session.task_context.get("family")
                                == "security"
                            ),
                            "repository_history_task": bool(
                                dynamic_session.task_context.get(
                                    "repository_history_task"
                                )
                            ),
                        },
                    )
                    residual_decisions.append(dict(residual_decision.trace))
                    if not residual_decision.used_default:
                        proposed_adm_operation = residual_decision.operation.value
                if residual_decision is not None and residual_decision.used_default:
                    active_adm_decision = None
                    dynamic_compact_next = False
                    residual_default_active = True
                    last_adm_stage = current_adm_stage
                    policy_events.append(
                        {
                            "event": "residual_meta_default",
                            "turn": turn_index + 1,
                            "stage": current_adm_stage.value,
                            "effective_operation": "KEEP",
                            "trace_sha256": residual_decision.trace[
                                "trace_sha256"
                            ],
                        }
                    )
                else:
                    residual_default_active = False
                    active_adm_decision = dynamic_session.choose(
                        stage=current_adm_stage,
                        proposed_operation=proposed_adm_operation,
                        features={
                            "turn_index": turn_index,
                            "remaining_turns": (
                                int(agent["max_turns"]) - turn_index
                            ),
                            "action_count": tool_calls_count,
                            "previous_recovery_signal": previous_recovery_signal,
                            "consecutive_no_progress_events": (
                                consecutive_no_progress_events
                            ),
                        },
                    )
                    dynamic_decisions.append(dict(active_adm_decision.trace))
                    dynamic_compact_next = active_adm_decision.compact_history
                    last_adm_stage = current_adm_stage
                    if (
                        current_adm_stage.value == "INITIAL"
                        and initial_adm_profile is None
                        and dynamic_profiles is not None
                    ):
                        activate_initial_profile(
                            active_adm_decision.operation.value
                        )
                    policy_events.append(
                        {
                            "event": "dynamic_adm_decision",
                            "turn": turn_index + 1,
                            "stage": current_adm_stage.value,
                            "proposed_operation": proposed_adm_operation,
                            "effective_operation": (
                                active_adm_decision.operation.value
                            ),
                            "decision_source": active_adm_decision.source,
                            "trace_sha256": active_adm_decision.trace[
                                "trace_sha256"
                            ],
                            "controller_response_sha256": (
                                controller_call["response_sha256"]
                                if controller_call is not None
                                else None
                            ),
                        }
                    )
            if residual_default_active:
                payload_messages = [dict(message) for message in messages]
            elif dynamic_compact_next:
                payload_messages = compact_adm_messages(
                    messages,
                    tool_message_characters=(
                        dynamic_session.policy.delete_tool_message_characters
                    ),
                    preserve_recent_tool_messages=(
                        dynamic_session.policy.delete_preserve_recent_tool_messages
                    ),
                )
                dynamic_compact_next = False
            else:
                payload_messages = [dict(message) for message in messages]
            active_instruction = (
                ""
                if residual_default_active
                else dynamic_session.active_instruction(current_adm_stage)
            )
            if active_instruction:
                payload_messages.append(
                    {
                        "role": "user",
                        "content": active_instruction,
                    }
                )
        history_compacted_this_turn = False
        if history_tool_characters:
            before_history_compaction = payload_messages
            payload_messages = compact_tool_history(
                payload_messages,
                limit=history_tool_characters,
                preserve_recent=history_preserve_recent,
            )
            if payload_messages != before_history_compaction:
                history_compaction_turns += 1
                history_compacted_this_turn = True
        if evidence_session is not None:
            evidence_message = evidence_session.before_actor(
                full_public_messages=messages, visible_messages=payload_messages, turn_index=turn_index,
            )
            if evidence_session.fallback_full_context:
                payload_messages = [dict(message) for message in messages]
                if history_compacted_this_turn:
                    history_compaction_turns -= 1
            elif evidence_message is not None:
                payload_messages = [*payload_messages, {"role": "user", "content": evidence_message}]
            write_json(trial_dir / "agent/evidence-contract-traces.json", evidence_session.traces)
        if repair_session is not None:
            repair_decision = repair_session.before_actor(
                turn_index=turn_index,
                remaining_steps=int(agent["max_turns"]) - turn_index,
                stage=(
                    current_adm_stage.value
                    if current_adm_stage is not None
                    else "EXECUTE"
                ),
            )
            if repair_decision.message is not None:
                payload_messages = [
                    *payload_messages,
                    {"role": "user", "content": repair_decision.message},
                ]
            write_json(
                trial_dir / "agent" / "repair-traces.json",
                list(repair_session.traces),
            )
        if progress_session is not None:
            progress_decision = progress_session.before_actor(
                turn_index=turn_index,
                remaining_steps=int(agent["max_turns"]) - turn_index,
                stage=current_adm_stage.value if current_adm_stage is not None else "EXECUTE",
            )
            if progress_decision.message is not None:
                payload_messages = [
                    *payload_messages,
                    {"role": "user", "content": progress_decision.message},
                ]
            write_json(
                trial_dir / "agent" / "public-progress-traces.json",
                progress_session.traces,
            )
        if consumer_session is not None:
            consumer_session.before_actor(payload_messages, turn_index=turn_index)
        if capsule_session is not None:
            payload_messages = capsule_session.prepare(
                full_public_messages=messages, visible_messages=payload_messages,
                turn_index=turn_index,
            )
            write_json(trial_dir / "agent/action-capsule-traces.json", capsule_session.traces)
        if dual_session is not None:
            payload_messages = dual_session.prepare(payload_messages, turn_index=turn_index)
            write_json(trial_dir / "agent/dual-terminal-traces.json", dual_session.traces)
        actor_tools = [EXEC_TOOL, *consumer_tools]
        if solpi_session is not None:
            payload_messages = solpi_session.project(payload_messages)
            actor_tools = solpi_session.tools(actor_tools)
            write_json(trial_dir / "agent/solpi.json", solpi_session.report())
        if jev_session is not None:
            actor_tools = jev_session.tools(actor_tools)
        payload = {
            "model": model["request_model_id"],
            "messages": payload_messages,
            "tools": actor_tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": float(model["temperature"]),
            "max_tokens": int(model["max_output_tokens"]),
            "seed": int(model["seed"]),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        turn_dir = trial_dir / "agent" / f"turn-{turn_index + 1:03d}"
        turn_dir.mkdir(parents=True)
        original_payload = payload
        payload, transport_trace = prepare_actor_transport(config, payload)
        if transport_trace is not None:
            write_json(turn_dir / "transport-original-request.json", original_payload)
            write_json(turn_dir / "transport-identity.json", transport_trace)
        write_json(turn_dir / "request.json", payload)
        request_attempt = 0
        while True:
            try:
                if cancellation_check is not None:
                    cancellation_check()
                if solpi_session is not None:
                    solpi_session.record_request()
                    write_json(trial_dir / "agent/solpi.json", solpi_session.report())
                response = post_json(
                    str(model["chat_completions_url"]),
                    payload,
                    timeout=min(float(model["request_timeout_seconds"]), remaining),
                )
                break
            except RuntimeError as exc:
                retry_tokens = context_overflow_retry_tokens(
                    str(exc),
                    current_max_tokens=int(payload["max_tokens"]),
                    margin_tokens=context_margin_tokens,
                )
                if (
                    retry_tokens is None
                    or request_attempt >= context_max_retries
                ):
                    raise
                request_attempt += 1
                event = {
                    "event": "context_overflow_retry",
                    "turn": turn_index + 1,
                    "attempt": request_attempt,
                    "previous_max_tokens": int(payload["max_tokens"]),
                    "retry_max_tokens": retry_tokens,
                    "margin_tokens": context_margin_tokens,
                    "error": str(exc),
                }
                context_budget_events.append(event)
                write_json(
                    turn_dir / f"context-overflow-{request_attempt:02d}.json",
                    event,
                )
                payload = {
                    **payload,
                    "max_tokens": retry_tokens,
                    "messages": compact_tool_messages_for_context(
                        payload["messages"],
                        context_tool_message_characters,
                    ),
                }
                write_json(
                    turn_dir / f"request-retry-{request_attempt:02d}.json",
                    payload,
                )
        write_json(turn_dir / "response.json", response)
        if solpi_session is not None:
            solpi_session.record_response(response.get("usage"))
            write_json(trial_dir / "agent/solpi.json", solpi_session.report())
        if cancellation_check is not None:
            cancellation_check()
        protocol_failure = response.get("phiagent_protocol_failure")
        if isinstance(protocol_failure, Mapping):
            protocol_failures.append(dict(protocol_failure))
        if response.get("model") != model["expected_response_model_id"]:
            raise RuntimeError("live completion returned an unexpected model identity")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("model response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
            raise RuntimeError("model response choice has no assistant message")
        raw_assistant_message = choice["message"]
        assistant_message = {
            "role": "assistant",
            "content": raw_assistant_message.get("content"),
        }
        if raw_assistant_message.get("tool_calls") is not None:
            assistant_message["tool_calls"] = raw_assistant_message["tool_calls"]
        messages.append(assistant_message)
        usage = response.get("usage", {})
        if isinstance(usage, Mapping):
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details", {})
            if isinstance(details, Mapping):
                cached_tokens += int(details.get("cached_tokens") or 0)

        tool_calls = model_tool_calls(assistant_message)
        harness_completion_call_id = None
        if not tool_calls:
            final_content = model_content(assistant_message)
            if (
                isinstance(protocol_failure, Mapping)
                and model_protocol_retry_count < model_protocol_retries
            ):
                model_protocol_retry_count += 1
                previous_recovery_signal = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous command response was not valid strict JSON. "
                            "Correct the response format: issue one short "
                            "exec_in_sandbox tool call with no prose."
                        ),
                    }
                )
                policy_events.append(
                    {
                        "event": "model_protocol_retry",
                        "turn": turn_index + 1,
                        "retry_number": model_protocol_retry_count,
                        "maximum": model_protocol_retries,
                    }
                )
                continue
            completion_call = None
            if (
                dual_session is not None and not isinstance(protocol_failure, Mapping)
                and choice.get("finish_reason") != "length"
                and (maximum_tool_calls is None or tool_calls_count < maximum_tool_calls)
            ):
                completion_call = dual_session.completion_call(
                    turn_index=turn_index,
                    remaining_turns=int(agent["max_turns"]) - turn_index - 1,
                    used_call_ids=[
                        call["id"] for message in messages
                        for call in message.get("tool_calls", [])
                        if isinstance(call, Mapping) and isinstance(call.get("id"), str)
                    ],
                )
                write_json(trial_dir / "agent/dual-terminal-traces.json", dual_session.traces)
            if completion_call is None:
                stop_reason = (
                    "model_protocol_failure"
                    if isinstance(protocol_failure, Mapping)
                    else str(choice.get("finish_reason") or "assistant_final")
                )
                break
            harness_completion_call_id = completion_call["id"]
            write_json(turn_dir / "harness-completion-proposal.json", {
                "origin": "dual_terminal_harness_not_model",
                "original_response_preserved": True, "tool_call": completion_call,
            })
            messages.append({
                "role": "assistant",
                "content": "Harness-origin check of output files explicitly required by the task.",
                "tool_calls": [completion_call],
            })
            tool_calls = [completion_call]

        turn_recovery_signal = False
        for call_index, tool_call in enumerate(tool_calls, start=1):
            if cancellation_check is not None:
                cancellation_check()
            tool_calls_count += 1
            tool_dir = turn_dir / f"tool-{call_index:02d}"
            tool_dir.mkdir()
            write_json(tool_dir / "call.json", tool_call)
            consumer_resolution = None
            try:
                executable_call = tool_call
                function = tool_call.get("function", {})
                if (solpi_session is not None and isinstance(function, Mapping)
                        and function.get("name") == "ObsRecall"):
                    call_id = tool_call.get("id")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("tool call has no nonempty id")
                    if maximum_tool_calls is not None and tool_calls_count > maximum_tool_calls:
                        tool_budget_rejections += 1
                        raise ValueError("Matched per-case tool-call budget exhausted; recall not executed.")
                    content = solpi_session.recall(tool_call)
                    write_json(tool_dir / "result.json", {
                        "result_kind": "solpi_observation_recall", "command": None,
                        "docker_command": None, "exit_code": None, "timed_out": False,
                        "environment_action_executed": False, "content": content,
                    })
                    messages.append({
                        "role": "tool", "name": "ObsRecall", "tool_call_id": call_id,
                        "content": content,
                    })
                    write_json(trial_dir / "agent/solpi.json", solpi_session.report())
                    continue
                if (consumer_session is not None and isinstance(function, Mapping)
                        and function.get("name") == "request_public_evidence"):
                    consumer_requests_total += 1
                    committed = consumer_session.commit_call(tool_call)
                    write_json(tool_dir / "consumer-commit.json", {
                        "request": committed.as_dict(), "request_sha256": committed.sha256,
                        "at": utc_now(), "archive_lookup_has_started": False,
                    })
                    consumer_resolution = consumer_session.resolve(committed)
                    write_json(tool_dir / "consumer-resolution.json", consumer_session.traces[-1])
                    write_json(trial_dir / "agent/consumer-evidence-traces.json", consumer_session.traces)
                    if consumer_resolution.action is None:
                        content = consumer_session.feedback(consumer_resolution)
                        write_json(tool_dir / "result.json", {
                            "result_kind": "harness_evidence_response", "command": None,
                            "docker_command": None, "exit_code": None, "timed_out": False,
                            "environment_action_executed": False,
                            "status": consumer_resolution.status, "content": content,
                        })
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call["id"], "content": content,
                        })
                        continue
                    executable_call = {
                        "id": tool_call["id"], "type": "function",
                        "function": {
                            "name": consumer_resolution.action.tool,
                            "arguments": consumer_resolution.action.arguments_json,
                        },
                    }
                call_id, command = parse_tool_command(executable_call)
                validation_command = (
                    solpi_session.validation_command(executable_call)
                    if solpi_session is not None else None
                )
                followups = jev_session.parse(executable_call) if jev_session is not None else []
                tool_cap_rejected = (
                    maximum_tool_calls is not None and tool_calls_count > maximum_tool_calls
                )
                rejection = (
                    "Matched per-case tool-call budget exhausted; no environment action executed."
                    if tool_cap_rejected
                    else duplicate_command_rejection(
                        command=command,
                        previous_command_fingerprint=previous_command_fingerprint,
                        policy=(
                            "allow"
                            if active_adm_decision is not None
                            and active_adm_decision.operation.value == "DELETE"
                            else duplicate_policy
                        ),
                    )
                )
                fingerprint = command_fingerprint(command)
                if no_progress_policy != "off":
                    shape_fingerprint = command_shape_fingerprint(command)
                    if shape_fingerprint == previous_command_shape:
                        consecutive_no_progress_events += 1
                        no_progress_events += 1
                        policy_events.append(
                            {
                                "event": "semantic_no_progress_detected",
                                "turn": turn_index + 1,
                                "tool_call": call_index,
                                "command_shape_fingerprint": shape_fingerprint,
                                "consecutive": consecutive_no_progress_events,
                            }
                        )
                    else:
                        consecutive_no_progress_events = 0
                    previous_command_shape = shape_fingerprint
                if rejection is not None:
                    if tool_cap_rejected:
                        tool_budget_rejections += 1
                    else:
                        duplicate_command_rejections += 1
                    result = {
                        "started_at_utc": utc_now(),
                        "finished_at_utc": utc_now(),
                        "duration_seconds": 0.0,
                        "command": command,
                        "command_fingerprint": fingerprint,
                        "docker_command": None,
                        "exit_code": None,
                        "stdout": "",
                        "stderr": rejection,
                        "timed_out": False,
                        "policy_rejected": True,
                        "policy": "max_tool_calls" if tool_cap_rejected else duplicate_policy,
                    }
                    content = rejection
                    policy_events.append(
                        {
                            "event": "tool_budget_rejected" if tool_cap_rejected else "duplicate_command_rejected",
                            "turn": turn_index + 1,
                            "tool_call": call_index,
                            "command_fingerprint": fingerprint,
                        }
                    )
                    if (
                        not tool_cap_rejected
                        and max_duplicate_rejections
                        and duplicate_command_rejections
                        >= max_duplicate_rejections
                    ):
                        duplicate_budget_triggered = True
                        policy_events.append(
                            {
                                "event": "duplicate_command_budget_exhausted",
                                "turn": turn_index + 1,
                                "duplicate_command_rejections": (
                                    duplicate_command_rejections
                                ),
                                "maximum": max_duplicate_rejections,
                            }
                        )
                else:
                    if consumer_resolution is not None:
                        consumer_session.mark_action_attempt(
                            request_sha256=consumer_resolution.request_sha256, command=command,
                        )
                        write_json(trial_dir / "agent/consumer-evidence-traces.json", consumer_session.traces)
                    tool_started = time.monotonic()
                    tool_timeout = float(agent["tool_timeout_seconds"])
                    if validation_command is not None or followups:
                        tool_timeout = min(tool_timeout, agent_timeout_seconds - (tool_started - started))
                        if tool_timeout <= 0:
                            raise ValueError("Agent deadline exhausted before fused command; no action executed")
                    environment_tool_calls_attempted += 1
                    result = execute(
                        container_name,
                        command,
                        tool_timeout,
                    )
                    environment_tool_calls_completed += 1
                    if validation_command is not None:
                        validation_result = None
                        remaining_validation = min(
                            tool_timeout - (time.monotonic() - tool_started),
                            agent_timeout_seconds - (time.monotonic() - started),
                        )
                        if result["exit_code"] == 0 and not result["timed_out"] and remaining_validation > 0:
                            if cancellation_check is not None:
                                cancellation_check()
                            environment_tool_calls_attempted += 1
                            validation_result = execute(container_name, validation_command, remaining_validation)
                            environment_tool_calls_completed += 1
                        result = solpi_session.combine_results(result, validation_result, validation_command)
                    result["command_fingerprint"] = fingerprint
                    output_limit = int(agent["max_tool_output_characters"])
                    if validation_command is not None:
                        content = "Primary command:\n" + tool_result_for_model(result["primary_result"], output_limit)
                        content += "\nValidation command:\n" + (
                            tool_result_for_model(result["validation_result"], output_limit)
                            if result["validation_result"] is not None
                            else "status=" + result["validation_status"]
                        )
                    else:
                        content = tool_result_for_model(result, output_limit)
                    if followups:
                        primary_result = dict(result)
                        write_json(tool_dir / "primary-result.json", primary_result)
                        following_results = []
                        gates = []
                        previous_result = primary_result
                        stopped_reason = "plan_exhausted"
                        content = "Primary command:\n" + content
                        for followup_index, step in enumerate(followups, start=1):
                            remaining_followup = min(
                                tool_timeout - (time.monotonic() - tool_started),
                                agent_timeout_seconds - (time.monotonic() - started),
                            )
                            visible = tool_result_for_model(previous_result, output_limit)
                            complete_limit = len(str(previous_result["stdout"])) + len(str(previous_result["stderr"])) + 128
                            complete_output = visible == tool_result_for_model(previous_result, complete_limit)
                            public_result = previous_result if complete_output else {
                                **previous_result, "stdout": None, "stderr": None,
                            }
                            gate = jev_session.gate(step, public_result, remaining_seconds=remaining_followup)
                            gate["checks"]["output_not_truncated"] = complete_output
                            gates.append(gate)
                            if not gate["execute"]:
                                stopped_reason = "public_gate_abstained"
                                break
                            if cancellation_check is not None:
                                cancellation_check()
                            tool_calls_count += 1
                            followup_rejection = duplicate_command_rejection(
                                command=step["command"], previous_command_fingerprint=fingerprint,
                                policy=duplicate_policy,
                            )
                            followup_cap = maximum_tool_calls is not None and tool_calls_count > maximum_tool_calls
                            if followup_cap or followup_rejection is not None:
                                if followup_cap:
                                    tool_budget_rejections += 1
                                    stopped_reason = "tool_budget_rejected"
                                else:
                                    duplicate_command_rejections += 1
                                    duplicate_budget_triggered = bool(max_duplicate_rejections and
                                        duplicate_command_rejections >= max_duplicate_rejections)
                                    stopped_reason = "duplicate_command_rejected"
                                policy_events.append({"event": stopped_reason, "turn": turn_index + 1,
                                    "tool_call": call_index, "followup": followup_index,
                                    "command_fingerprint": command_fingerprint(step["command"])})
                                break
                            write_json(tool_dir / f"followup-{followup_index:02d}-request.json", {
                                "origin": "model_declared_followup", "parent_tool_call_id": call_id,
                                "command": step["command"], "remaining_shared_seconds": remaining_followup,
                                "gate": gate,
                            })
                            environment_tool_calls_attempted += 1
                            following = execute(container_name, step["command"], remaining_followup)
                            environment_tool_calls_completed += 1
                            fingerprint = command_fingerprint(step["command"])
                            following["command_fingerprint"] = fingerprint
                            following_results.append(following)
                            write_json(tool_dir / f"followup-{followup_index:02d}-result.json", following)
                            content += f"\nFollowup {followup_index}:\n" + tool_result_for_model(following, output_limit)
                            previous_result = following
                            if following["exit_code"] != 0 or following["timed_out"]:
                                stopped_reason = "followup_failed"
                                break
                            if jev_session.condition_target == "self":
                                complete_limit = len(str(following["stdout"])) + len(str(following["stderr"])) + 128
                                complete_output = tool_result_for_model(following, output_limit) == tool_result_for_model(following, complete_limit)
                                checked_output = following if complete_output else {**following, "stdout": None, "stderr": None}
                                checked = jev_session.postcondition(step, checked_output)
                                gates.append(checked)
                                if not checked["execute"]:
                                    stopped_reason = "postcondition_failed"
                                    break
                        trace = jev_session.record(tool_call_id=call_id, declared=followups,
                            executed=following_results, gates=gates, stopped_reason=stopped_reason)
                        result = {**primary_result, "primary_result": primary_result,
                                  "followup_results": following_results, "continuation": trace,
                                  "exit_code": previous_result["exit_code"], "timed_out": previous_result["timed_out"],
                                  "command_fingerprint": fingerprint}
                        content += "\nContinuation status:\n" + json.dumps({
                            "declared": len(followups), "executed": len(following_results),
                            "stopped_reason": stopped_reason,
                            "failed_checks": [name for gate in gates for name, passed in gate["checks"].items() if not passed],
                        }, sort_keys=True)
                        turn_recovery_signal = turn_recovery_signal or stopped_reason != "plan_exhausted"
                        write_json(trial_dir / "agent/jev.json", jev_session.report())
                    if consumer_session is not None:
                        content = consumer_session.observe_environment(
                            command=command, public_content=content,
                        )
                        write_json(trial_dir / "agent/consumer-public-reports.json", [
                            {"sequence": report.sequence, "payload_json": report.payload_json}
                            for report in consumer_session.reports
                        ])
                turn_recovery_signal = turn_recovery_signal or bool(
                    result.get("policy_rejected")
                    or result.get("timed_out")
                    or result.get("validation_failed")
                )
                previous_command_fingerprint = fingerprint
            except (ValueError, json.JSONDecodeError) as exc:
                call_id = str(tool_call.get("id") or f"invalid-{turn_index}-{call_index}")
                result = {
                    "started_at_utc": utc_now(),
                    "finished_at_utc": utc_now(),
                    "duration_seconds": 0.0,
                    "command": None,
                    "docker_command": None,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": str(exc),
                    "timed_out": False,
                    "protocol_error": True,
                }
                content = f"Tool call rejected: {exc}"
                turn_recovery_signal = True
            if consumer_resolution is not None:
                result["consumer_request_sha256"] = consumer_resolution.request_sha256
                result["consumer_action_kind"] = (
                    "explicit_observation" if consumer_resolution.requires_charged_observation
                    else "committed_branch"
                )
            if dual_session is not None:
                result["dual_terminal_origin"] = (
                    "harness_completion_check"
                    if call_id == harness_completion_call_id else "model"
                )
                content = truncate_tool_output(dual_session.observe(
                    call_id=call_id, public_content=content,
                    executed=not bool(result.get("protocol_error") or result.get("policy_rejected")),
                ), int(agent["max_tool_output_characters"]))
                write_json(trial_dir / "agent/dual-terminal-traces.json", dual_session.traces)
            write_json(tool_dir / "result.json", result)
            if repair_session is not None:
                repair_session.observe(
                    action=str(result.get("command") or "invalid_tool_call"),
                    observation=content,
                    tool_error=bool(
                        result.get("exit_code") not in (None, 0)
                        or result.get("timed_out")
                        or result.get("protocol_error")
                        or result.get("policy_rejected")
                    ),
                )
            if progress_session is not None:
                progress_session.observe(
                    action=str(result.get("command") or "invalid_tool_call"),
                    observation=content,
                    tool_error=bool(
                        result.get("exit_code") not in (None, 0)
                        or result.get("timed_out")
                        or result.get("protocol_error")
                        or result.get("policy_rejected")
                    ),
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        previous_recovery_signal = turn_recovery_signal
        if (
            no_progress_policy != "off"
            and consecutive_no_progress_events >= no_progress_patience
        ):
            if no_progress_policy == "remind" or no_progress_reminders == 0:
                messages.append({"role": "user", "content": no_progress_reminder})
                no_progress_reminders += 1
                consecutive_no_progress_events = 0
                policy_events.append(
                    {
                        "event": "no_progress_reminder_inserted",
                        "turn": turn_index + 1,
                        "reminder_number": no_progress_reminders,
                    }
                )
            else:
                no_progress_termination_triggered = True
                policy_events.append(
                    {
                        "event": "no_progress_termination",
                        "turn": turn_index + 1,
                        "no_progress_events": no_progress_events,
                    }
                )
        if duplicate_budget_triggered:
            stop_reason = "duplicate_command_budget"
            break
        if no_progress_termination_triggered:
            stop_reason = "no_progress_termination"
            break

    write_json(trial_dir / "agent" / "transcript.json", messages)
    write_json(trial_dir / "agent" / "policy-events.json", policy_events)
    if solpi_session is not None:
        write_json(trial_dir / "agent/solpi.json", solpi_session.report())
    if jev_session is not None:
        write_json(trial_dir / "agent/jev.json", jev_session.report())
    return {
        "started_at_utc": started_at_utc,
        "duration_seconds": time.monotonic() - started,
        "model_calls": (
            len(list((trial_dir / "agent").glob("turn-*/response.json")))
            + len(dynamic_controller_calls)
        ),
        "tool_calls": tool_calls_count,
        **({
            "maximum_tool_calls": maximum_tool_calls,
            "tool_budget_rejections": tool_budget_rejections,
        } if maximum_tool_calls is not None else {}),
        "environment_tool_calls_attempted": environment_tool_calls_attempted,
        "environment_tool_calls_completed": environment_tool_calls_completed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens,
        "stop_reason": stop_reason,
        "final_content": final_content,
        "protocol_failures": protocol_failures,
        "model_protocol_retry_count": model_protocol_retry_count,
        "tool_policy": dict(tool_policy),
        "duplicate_command_rejections": duplicate_command_rejections,
        "duplicate_budget_triggered": duplicate_budget_triggered,
        "no_progress_events": no_progress_events,
        "no_progress_reminders": no_progress_reminders,
        "no_progress_termination_triggered": no_progress_termination_triggered,
        "validation_reminder_inserted": validation_reminder_inserted,
        "history_policy": dict(history_policy),
        "history_compaction_turns": history_compaction_turns,
        "effective_runtime": {
            "max_turns": int(agent["max_turns"]),
            "max_output_tokens": int(model["max_output_tokens"]),
            "max_tool_output_characters": int(
                agent["max_tool_output_characters"]
            ),
            "reserve_validation_turns": reserve_validation_turns,
        },
        "context_overflow_policy": dict(context_policy),
        "context_overflow_retries": len(context_budget_events),
        "context_budget_events": context_budget_events,
        "dynamic_adm_policy": (
            adm_policy_to_dict(dynamic_session.policy)
            if dynamic_session is not None
            else None
        ),
        "dynamic_adm_decisions": dynamic_decisions,
        "dynamic_adm_controller_calls": dynamic_controller_calls,
        "residual_meta_policy": (
            residual_policy_to_dict(residual_session.policy)
            if residual_session is not None
            else None
        ),
        "residual_meta_decisions": residual_decisions,
        "residual_meta_router_model_calls": 0,
        "dynamic_adm_initial_profile": initial_adm_profile,
        "dynamic_adm_model_decisions": sum(
            row["decision_source"] == "model" for row in dynamic_decisions
        ),
        "dynamic_adm_fallback_decisions": sum(
            row["decision_source"] == "model_fallback"
            for row in dynamic_decisions
        ),
        "policy_events_sha256": file_sha256(
            trial_dir / "agent" / "policy-events.json"
        ),
        "transcript_sha256": file_sha256(trial_dir / "agent" / "transcript.json"),
        **({"solpi": solpi_session.report()} if solpi_session is not None else {}),
        **({"jev": jev_session.report()} if jev_session is not None else {}),
        **({"dual_terminal": dual_session.summary()} if dual_session is not None else {}),
        **(
            {
                "counterfactual_repair": {
                    "mode": repair_value["mode"],
                    "parent_harness_sha256": repair_parent_sha256(config),
                    "traces": list(repair_session.traces),
                    "activation_count": sum(
                        bool(trace["activated_now"])
                        for trace in repair_session.traces
                    ),
                }
            }
            if repair_session is not None
            else {}
        ),
        **(
            {
                "public_progress": {
                    "mode": progress_value["mode"],
                    "parent_harness_sha256": repair_parent_sha256(config),
                    "traces": progress_session.traces,
                    "activation_count": sum(
                        bool(trace["activated_now"]) for trace in progress_session.traces
                    ),
                    "information_boundary": "Existing public tool-message contents after tool-output cap; no raw untruncated stdout or verifier results.",
                }
            }
            if progress_session is not None
            else {}
        ),
        **(
            {
                "evidence_contracts": {
                    "mode": evidence_value["mode"],
                    "traces": evidence_session.traces,
                    "message_turns": sum(trace["message_sha256"] is not None for trace in evidence_session.traces),
                    "fallback_full_context_turns": sum(trace["fallback_full_parent_context"] for trace in evidence_session.traces),
                    "information_boundary": "Only already-public actor transcript and proposed outgoing context; no raw uncapped tool result or verifier input.",
                }
            }
            if evidence_session is not None
            else {}
        ),
        **(
            {"consumer_evidence": {
                **consumer_session.summary(), "requested_tool_calls": consumer_requests_total,
            }}
            if consumer_session is not None else {}
        ),
        **(
            {"action_capsules": {
                "mode": capsule_value["mode"],
                "config": capsule_value["config"],
                "traces": capsule_session.traces,
                "information_boundary": "Only already-public capped tool messages and commands; no grading input.",
                "extra_model_calls": 0,
                "extra_environment_actions": 0,
            }}
            if capsule_session is not None else {}
        ),
    }


def sanitized_container_name(experiment: Path, task_name: str) -> str:
    value = f"phiagent-{experiment.name}-{task_name}"
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)[:120]


def container_network_arguments(config: Mapping[str, Any]) -> list[str]:
    mode = config.get("container_network_mode")
    if mode is None:
        return []
    if mode != "none":
        raise ValueError("only explicitly network-isolated container adaptation is supported")
    return ["--network", "none"]


def docker_image_id(image: str) -> str:
    return command_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"])


def run_verifier(
    *,
    config: Mapping[str, Any],
    container_name: str,
    timeout_seconds: float,
    trial_dir: Path,
) -> dict[str, Any]:
    command = ["docker", "exec"]
    if config.get("verifier_runtime"):
        command.extend(
            [
                "--env",
                "HTTP_PROXY=http://127.0.0.1:9",
                "--env",
                "HTTPS_PROXY=http://127.0.0.1:9",
                "--env",
                "http_proxy=http://127.0.0.1:9",
                "--env",
                "https_proxy=http://127.0.0.1:9",
                "--env",
                "NO_PROXY=127.0.0.1,localhost",
                "--env",
                "no_proxy=127.0.0.1,localhost",
                "--env",
                "UV_OFFLINE=true",
                "--env",
                "UV_FIND_LINKS=/verifier-wheels",
                "--env",
                "UV_PYTHON_DOWNLOADS=never",
                "--env",
                "UV_CACHE_DIR=/root/.cache/uv",
                "--env",
                "PIP_NO_INDEX=1",
                "--env",
                "PIP_FIND_LINKS=/verifier-wheels",
                "--env",
                "PIP_DISABLE_PIP_VERSION_CHECK=1",
                "--env",
                (
                    "PATH=/root/.local/bin:/verifier-python/bin:/usr/local/sbin:"
                    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ),
                "--env",
                "LD_LIBRARY_PATH=/verifier-python/lib",
            ]
        )
    command.extend([container_name, "bash", "/tests/test.sh"])
    started = time.monotonic()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    (trial_dir / "verifier.stdout.log").write_text(completed.stdout)
    (trial_dir / "verifier.stderr.log").write_text(completed.stderr)
    reward_path = trial_dir / "logs" / "verifier" / "reward.txt"
    if not reward_path.is_file():
        raise RuntimeError(
            f"official verifier produced no reward.txt (exit {completed.returncode})"
        )
    reward_text = reward_path.read_text().strip()
    if reward_text not in {"0", "1"}:
        raise RuntimeError(f"official verifier returned nonbinary reward {reward_text!r}")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "duration_seconds": time.monotonic() - started,
        "reward": int(reward_text),
        "reward_path": str(reward_path),
        "reward_sha256": file_sha256(reward_path),
        "stdout_sha256": file_sha256(trial_dir / "verifier.stdout.log"),
        "stderr_sha256": file_sha256(trial_dir / "verifier.stderr.log"),
    }


def prepare_offline_verifier(
    *,
    container_name: str,
    trial_dir: Path,
    expected_uv_version: str,
    install_debs: bool,
    install_curl_stub: bool,
) -> dict[str, Any]:
    setup_commands = [
        "install -d /root/.local/bin",
        "install -m 0755 /verifier-bin/uv /root/.local/bin/uv",
        "install -m 0755 /verifier-bin/uvx /root/.local/bin/uvx",
        (
            "printf '%s\\n' 'export PATH=\"$HOME/.local/bin:$PATH\"' "
            "> /root/.local/bin/env"
        ),
    ]
    if install_debs:
        setup_commands.append("dpkg -i /verifier-debs/*.deb")
    if install_curl_stub:
        setup_commands.extend(
            [
                (
                    "printf '%s\\n' '#!/bin/sh' "
                    "'echo verifier network disabled >&2' 'exit 1' "
                    "> /root/.local/bin/curl"
                ),
                "chmod 0755 /root/.local/bin/curl",
            ]
        )
    setup_commands.extend(
        [
            "/root/.local/bin/uv --version",
            "test -x /root/.local/bin/uvx",
            "test -f /root/.local/bin/env",
        ]
    )
    command = [
        "docker",
        "exec",
        container_name,
        "bash",
        "-lc",
        " && ".join(setup_commands),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180.0,
        check=False,
    )
    (trial_dir / "verifier-preinstall.stdout.log").write_text(completed.stdout)
    (trial_dir / "verifier-preinstall.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"offline verifier prerequisite install failed with {completed.returncode}"
        )
    version_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith("uv ")
    ]
    if version_lines != [expected_uv_version]:
        raise RuntimeError(
            f"offline verifier uv identity mismatch: {version_lines!r}"
        )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "uv_version": version_lines[0],
        "stdout_sha256": file_sha256(
            trial_dir / "verifier-preinstall.stdout.log"
        ),
        "stderr_sha256": file_sha256(
            trial_dir / "verifier-preinstall.stderr.log"
        ),
    }


def copy_directory_to_container(
    *,
    source: Path,
    container_name: str,
    destination: str,
) -> list[list[str]]:
    commands = [
        [
            "docker",
            "exec",
            container_name,
            "rm",
            "-rf",
            destination,
        ],
        [
            "docker",
            "exec",
            container_name,
            "mkdir",
            "-p",
            destination,
        ],
        [
            "docker",
            "cp",
            f"{source}/.",
            f"{container_name}:{destination}",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    return commands


def stage_verifier(
    *,
    config: Mapping[str, Any],
    task_name: str,
    task_dir: Path,
    container_name: str,
    trial_dir: Path,
) -> dict[str, Any]:
    copied = {
        "/tests": copy_directory_to_container(
            source=task_dir / "tests",
            container_name=container_name,
            destination="/tests",
        )
    }
    verifier_runtime = config.get("verifier_runtime")
    if not isinstance(verifier_runtime, Mapping):
        result = {"copied_directories": copied}
        write_json(trial_dir / "verifier-staging.json", result)
        return result

    runtime_root = Path(str(verifier_runtime["root"]))
    task_assets = verifier_runtime["tasks"][task_name]
    sources = {
        "/verifier-bin": runtime_root / str(verifier_runtime["bin_dir"]),
        "/verifier-wheels": runtime_root / str(task_assets["wheelhouse"]),
    }
    if task_assets.get("deb_dir"):
        sources["/verifier-debs"] = runtime_root / str(task_assets["deb_dir"])
    if task_assets.get("python_dir"):
        sources["/verifier-python"] = runtime_root / str(task_assets["python_dir"])
    for fixture in task_assets.get("fixtures", []):
        sources[str(fixture["destination"])] = runtime_root / str(
            fixture["source"]
        )
    for destination, source in sources.items():
        copied[destination] = copy_directory_to_container(
            source=source,
            container_name=container_name,
            destination=destination,
        )
    result = {"copied_directories": copied}
    write_json(trial_dir / "verifier-staging.json", result)
    return result


def run_trial(
    *,
    config: Mapping[str, Any],
    experiment: Path,
    dataset: Path,
    task_spec: Mapping[str, Any],
) -> dict[str, Any]:
    task_name = str(task_spec["name"])
    task_dir = dataset / task_name
    trial_dir = experiment / "trials" / task_name
    trial_dir.mkdir(parents=True)
    logs_dir = trial_dir / "logs" / "verifier"
    logs_dir.mkdir(parents=True)
    logs_dir.chmod(0o777)
    container_name = sanitized_container_name(
        Path(str(config.get("container_name_prefix", experiment.name))), task_name
    )
    task_config = tomllib.loads((task_dir / "task.toml").read_text())
    environment = task_config["environment"]
    image = str(task_spec["docker_image"])
    result: dict[str, Any] = {
        "task": task_name,
        "started_at_utc": utc_now(),
        "source_tree_sha256": tree_sha256(task_dir),
        "docker_image": image,
        "docker_image_id": docker_image_id(image),
        "container_name": container_name,
        "agent_result": None,
        "verifier_result": None,
        "exception": None,
    }
    if result["source_tree_sha256"] != task_spec["source_tree_sha256"]:
        raise RuntimeError(f"source tree changed before trial: {task_name}")
    if result["docker_image_id"] != task_spec["docker_image_id"]:
        raise RuntimeError(f"Docker image changed before trial: {task_name}")

    run_command = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--cpus",
        str(environment["cpus"]),
        "--memory",
        str(environment["memory"]),
        "--mount",
        f"type=bind,src={logs_dir},dst=/logs/verifier",
    ]
    run_command.extend(container_network_arguments(config))
    verifier_runtime = config.get("verifier_runtime")
    if isinstance(verifier_runtime, Mapping):
        uv_cache = trial_dir / "uv-cache"
        uv_cache.mkdir()
        run_command.extend(
            [
                "--mount",
                f"type=bind,src={uv_cache},dst=/root/.cache/uv",
            ]
        )
    if config["mode"] == "solution-parity":
        run_command.extend(
            [
                "--mount",
                f"type=bind,src={task_dir / 'solution'},dst=/solution,readonly",
            ]
        )
    run_command.extend([image, "sleep", "infinity"])
    write_json(trial_dir / "container-run.json", {"command": run_command})

    container_started = False
    try:
        subprocess.run(run_command, check=True, capture_output=True, text=True)
        container_started = True
        inspect = json.loads(command_output(["docker", "inspect", container_name]))
        write_json(trial_dir / "container-inspect-before.json", inspect)
        forbidden_paths = [
            "/tests",
            "/verifier-bin",
            "/verifier-wheels",
            "/verifier-debs",
            "/verifier-python",
        ]
        if config["mode"] == "agent":
            forbidden_paths.append("/solution")
        hidden_asset_command = [
            "docker",
            "exec",
            container_name,
            "sh",
            "-c",
            " && ".join(f"test ! -e {path}" for path in forbidden_paths),
        ]
        hidden_asset_check = subprocess.run(
            hidden_asset_command,
            capture_output=True,
            text=True,
            check=False,
        )
        result["pre_agent_hidden_asset_audit"] = {
            "command": hidden_asset_command,
            "forbidden_paths": forbidden_paths,
            "exit_code": hidden_asset_check.returncode,
            "stdout": hidden_asset_check.stdout,
            "stderr": hidden_asset_check.stderr,
            "passed": hidden_asset_check.returncode == 0,
        }
        if hidden_asset_check.returncode != 0:
            raise RuntimeError("verifier-only assets are visible before agent execution")

        if config["mode"] == "solution-parity":
            solution_command = [
                "docker",
                "exec",
            ]
            solution_environment = config.get("solution_environment", {})
            if not isinstance(solution_environment, Mapping):
                raise TypeError("solution_environment must be an object")
            allowed_environment = {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
                "PIP_INDEX_URL",
            }
            if set(solution_environment) - allowed_environment:
                raise ValueError("solution_environment contains unsupported keys")
            for name in sorted(solution_environment):
                solution_command.extend(
                    ["--env", f"{name}={solution_environment[name]}"]
                )
            solution_command.extend(
                [
                    container_name,
                    "bash",
                    f"/solution/{task_spec['solution_entrypoint']}",
                ]
            )
            completed = subprocess.run(
                solution_command,
                capture_output=True,
                text=True,
                timeout=float(task_config["agent"]["timeout_sec"]),
                check=False,
            )
            (trial_dir / "solution.stdout.log").write_text(completed.stdout)
            (trial_dir / "solution.stderr.log").write_text(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"official solution failed with exit code {completed.returncode}"
                )
            result["agent_result"] = {
                "mode": "official_solution",
                "command": solution_command,
                "exit_code": completed.returncode,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
        elif config["mode"] == "agent":
            instruction = (task_dir / "instruction.md").read_text()
            result["agent_result"] = run_agent(
                config=config,
                instruction=instruction,
                container_name=container_name,
                trial_dir=trial_dir,
                agent_timeout_seconds=float(task_config["agent"]["timeout_sec"]),
                task_spec=task_spec,
            )
        else:
            raise ValueError(f"unsupported run mode: {config['mode']}")

        result["verifier_staging"] = stage_verifier(
            config=config,
            task_name=task_name,
            task_dir=task_dir,
            container_name=container_name,
            trial_dir=trial_dir,
        )
        if isinstance(verifier_runtime, Mapping):
            task_assets = verifier_runtime["tasks"][task_name]
            result["verifier_preinstall"] = prepare_offline_verifier(
                container_name=container_name,
                trial_dir=trial_dir,
                expected_uv_version=str(verifier_runtime["uv_version"]),
                install_debs=bool(task_assets.get("deb_dir")),
                install_curl_stub=bool(task_assets.get("curl_stub", False)),
            )
        result["verifier_result"] = run_verifier(
            config=config,
            container_name=container_name,
            timeout_seconds=float(task_config["verifier"]["timeout_sec"]),
            trial_dir=trial_dir,
        )
        inspect_after = json.loads(command_output(["docker", "inspect", container_name]))
        write_json(trial_dir / "container-inspect-after.json", inspect_after)
    except Exception as exc:  # Trial exceptions must remain distinct from reward zero.
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if container_started:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                check=False,
            )
        result["finished_at_utc"] = utc_now()
        result["valid_binary_reward"] = (
            result["exception"] is None
            and isinstance(result["verifier_result"], Mapping)
            and result["verifier_result"].get("reward") in {0, 1}
        )
        write_json(trial_dir / "result.json", result)
    return result


def verify_verifier_runtime(
    config: Mapping[str, Any],
    experiment: Path,
) -> None:
    runtime = config.get("verifier_runtime")
    if runtime is None:
        return
    if not isinstance(runtime, Mapping):
        raise TypeError("verifier_runtime must be an object")
    root = Path(str(runtime["root"]))
    manifest = []
    for relative, expected_sha256 in sorted(runtime.get("files_sha256", {}).items()):
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing verifier runtime file: {path}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"verifier runtime file changed: {relative}")
        manifest.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": observed_sha256,
            }
        )
    group_hashes = {}
    for relative, expected_sha256 in sorted(
        runtime.get("groups_sha256", {}).items()
    ):
        path = root / str(relative)
        if not path.is_dir():
            raise FileNotFoundError(f"missing verifier runtime group: {path}")
        group_manifest = tree_manifest(path)
        root_relative_manifest = [
            {
                **row,
                "path": str(Path(str(relative)) / str(row["path"])),
            }
            for row in group_manifest
        ]
        observed_sha256 = canonical_sha256(root_relative_manifest)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"verifier runtime group changed: {relative}")
        group_hashes[str(relative)] = observed_sha256
        manifest.extend(root_relative_manifest)
    if not manifest:
        raise ValueError("verifier_runtime must freeze at least one file or group")
    write_json(
        experiment / "verifier-runtime-manifest.json",
        {
            "root": str(root),
            "uv_version": runtime["uv_version"],
            "groups_sha256": group_hashes,
            "files": manifest,
        },
    )


def verify_model_listing(
    model: Mapping[str, Any], listing: Mapping[str, Any]
) -> None:
    matching = [
        row for row in listing.get("data", [])
        if isinstance(row, Mapping) and row.get("id") == model["request_model_id"]
    ]
    if not matching:
        raise RuntimeError("live model listing does not contain the frozen request ID")
    expected = model.get("expected_max_model_len")
    if expected is not None:
        if type(expected) is not int or expected < 1:
            raise ValueError("expected_max_model_len must be a positive integer")
        if any(row.get("max_model_len") != expected for row in matching):
            raise RuntimeError("live model context capacity differs from frozen model binding")


def snapshot_and_verify_service(
    config: Mapping[str, Any],
    experiment: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    service_dir = Path(str(config["model"]["service_dir"]))
    snapshot_dir = experiment / "model_service_snapshot"
    snapshot_dir.mkdir()
    for name, expected_sha256 in config["model"]["service_files_sha256"].items():
        source = service_dir / name
        if file_sha256(source) != expected_sha256:
            raise RuntimeError(f"model service evidence changed: {name}")
        shutil.copy2(source, snapshot_dir / name)
    service = json.loads((service_dir / "service.json").read_text())
    if service.get("request_model_id") != config["model"]["request_model_id"]:
        raise RuntimeError("model service request identity differs from frozen config")
    if service.get("expected_model_id") != config["model"]["expected_response_model_id"]:
        raise RuntimeError("model service response identity differs from frozen config")

    service_kind = str(config["model"].get("service_kind", "local_docker_gpu"))
    if service_kind == "local_docker_gpu":
        container_name = str(service["container_name"])
        container = json.loads(
            command_output(["docker", "inspect", container_name])
        )
        if not isinstance(container, list) or len(container) != 1:
            raise RuntimeError("model service container inspect returned an invalid result")
        container_row = container[0]
        if (
            container_row.get("Id") != service.get("container_id")
            or container_row.get("State", {}).get("Status") != "running"
        ):
            raise RuntimeError("model service container identity or state changed")
        service_main_pid = int(container_row["State"]["Pid"])
        top_output = command_output(["docker", "top", container_name, "-eo", "pid"])
        service_process_ids = {
            int(line.strip())
            for line in top_output.splitlines()[1:]
            if line.strip().isdigit()
        }
        if service_main_pid not in service_process_ids:
            raise RuntimeError("model service main PID is absent from docker top")
        gpu = gpu_preflight(config, service_process_ids)
        write_json(experiment / "model-container-inspect.json", container)
    elif service_kind == "remote_relay":
        service_main_pid = int(service["proxy_pid"])
        if not process_is_alive(service_main_pid):
            raise RuntimeError("model relay proxy is not alive")
        remote_audit = json.loads(
            (service_dir / str(service["remote_gpu_preflight_file"])).read_text()
        )
        if not isinstance(remote_audit, Mapping):
            raise RuntimeError("remote GPU preflight is not an object")
        selected = remote_audit.get("selected_physical_gpu")
        model_host = config.get("model_host")
        if not isinstance(selected, Mapping) or not isinstance(model_host, Mapping):
            raise RuntimeError("remote model GPU identity is unavailable")
        if (
            int(selected.get("index", -1)) != int(model_host["physical_gpu_index"])
            or selected.get("uuid") != model_host["expected_gpu_uuid"]
            or remote_audit.get("hostname") != model_host["hostname"]
        ):
            raise RuntimeError("remote model GPU preflight differs from frozen identity")
        gpu = dict(remote_audit)
        gpu["relay_proxy_pid"] = service_main_pid
        gpu["relay_proxy_alive"] = True
    else:
        raise ValueError(f"unsupported model service kind: {service_kind}")
    models = get_json(str(config["model"]["models_url"]), timeout=30.0)
    verify_model_listing(config["model"], models)
    probe = post_json(
        str(config["model"]["chat_completions_url"]),
        {
            "model": config["model"]["request_model_id"],
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "temperature": 0,
            "max_tokens": 16,
            "seed": int(config["model"]["seed"]),
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60.0,
    )
    if probe.get("model") != config["model"]["expected_response_model_id"]:
        raise RuntimeError("live probe returned an unexpected model identity")
    write_json(experiment / "model-listing.json", models)
    write_json(experiment / "model-probe.json", probe)
    write_json(experiment / "gpu-preflight.json", gpu)
    service["_live_main_pid"] = service_main_pid
    service["_live_process_ids"] = (
        sorted(service_process_ids) if service_kind == "local_docker_gpu" else []
    )
    service["_service_kind"] = service_kind
    return service, gpu


def configure_executor_cuda(config: Mapping[str, Any]) -> str:
    if config["model"].get("service_kind", "local_docker_gpu") == "remote_relay":
        from scripts.terminal_cpu_runtime import require_cpu_executor

        require_cpu_executor()
        return ""
    model_host = config.get("model_host", config["host"])
    selected = str(model_host["physical_gpu_index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


def run_experiment(experiment: Path) -> tuple[dict[str, Any], int]:
    config_path = experiment / "config.json"
    reservation_path = experiment / "reservation.json"
    git_state_path = experiment / "git-state.json"
    if not all(
        path.is_file() for path in (config_path, reservation_path, git_state_path)
    ):
        raise FileNotFoundError(
            "experiment must contain config.json, reservation.json, and git-state.json"
        )
    config = json.loads(config_path.read_text())
    if config.get("status") != "FROZEN_BEFORE_FIRST_RUN":
        raise RuntimeError("experiment config is not frozen")
    if (
        config["agent"].get("residual_meta_policy") is not None
        and config["agent"].get("dynamic_adm_policy") is None
    ):
        raise RuntimeError(
            "residual_meta_policy requires a residual-mode dynamic_adm_policy"
        )
    runner = Path(__file__).resolve()
    if file_sha256(runner) != config["runner_sha256"]:
        raise RuntimeError("runner differs from frozen SHA-256")
    source_snapshot = experiment / "source_snapshot"
    source_snapshot.mkdir()
    shutil.copy2(runner, source_snapshot / runner.name)
    if config.get("actor_transport") is not None:
        import scripts.terminal_prefix_fidelity as normalizer

        source = Path(normalizer.__file__).resolve()
        if file_sha256(source) != config["actor_transport"]["normalizer_sha256"]:
            raise RuntimeError("common transport normalizer differs from frozen source")
        shutil.copy2(source, source_snapshot / source.name)
    if config["agent"].get("counterfactual_repair") is not None:
        import phiagent.learning.counterfactual_repair as repair_module

        repair_source = Path(repair_module.__file__).resolve()
        if file_sha256(repair_source) != config.get("counterfactual_repair_source_sha256"):
            raise RuntimeError("counterfactual repair source differs from frozen SHA-256")
        shutil.copy2(repair_source, source_snapshot / repair_source.name)
    if config["agent"].get("public_progress") is not None:
        if config["agent"].get("counterfactual_repair") is not None:
            raise ValueError("public progress and legacy repair sidecars cannot overlap")
        progress_sources = public_progress_source_hashes()
        if config.get("public_progress_source_hashes") != progress_sources:
            raise RuntimeError("public progress adapter/shared predicate/runtime sources differ from freeze")
        for relative in progress_sources:
            source = Path(__file__).resolve().parents[1] / relative
            shutil.copy2(source, source_snapshot / source.name)
    if config["agent"].get("action_capsules") is not None:
        from scripts import terminal_action_capsules

        source = Path(terminal_action_capsules.__file__).resolve()
        if file_sha256(source) != config.get("action_capsules_source_sha256"):
            raise RuntimeError("action capsule source differs from frozen SHA-256")
        shutil.copy2(source, source_snapshot / source.name)
    if config["agent"].get("dual_terminal") is not None:
        from scripts import dual_terminal_completion

        source = Path(dual_terminal_completion.__file__).resolve()
        if file_sha256(source) != config.get("dual_terminal_source_sha256"):
            raise RuntimeError("dual_terminal source differs from frozen SHA-256")
        shutil.copy2(source, source_snapshot / source.name)
    if config["agent"].get("evidence_contracts") is not None:
        if any(config["agent"].get(name) is not None for name in (
            "counterfactual_repair", "public_progress", "dynamic_adm_policy", "residual_meta_policy",
            "consumer_evidence",
        )):
            raise ValueError("Evidence contracts require one non-overlapping controller")
        sources = evidence_contract_source_hashes()
        if config.get("evidence_contract_source_hashes") != sources:
            raise RuntimeError("Evidence contract query adapter/checker source differs from the frozen configuration")
        for relative in sources:
            source = Path(__file__).resolve().parents[1] / relative
            shutil.copy2(source, source_snapshot / source.name)
    if config["agent"].get("consumer_evidence") is not None:
        if any(config["agent"].get(name) is not None for name in (
            "counterfactual_repair", "public_progress", "evidence_contracts",
            "dynamic_adm_policy", "residual_meta_policy",
        )):
            raise ValueError("Consumer evidence requires one non-overlapping controller")
        sources = consumer_evidence_source_hashes()
        if config.get("consumer_evidence_source_hashes") != sources:
            raise RuntimeError("Consumer evidence interface/checker source differs from the frozen configuration")
        for relative in sources:
            source = Path(__file__).resolve().parents[1] / relative
            shutil.copy2(source, source_snapshot / source.name)
    dynamic_adm_source = Path(stage_aware_adm_module.__file__).resolve()
    residual_meta_source = Path(
        residual_meta_harness_module.__file__
    ).resolve()
    if config["agent"].get("dynamic_adm_policy") is not None:
        expected_dynamic_source = config.get("dynamic_adm_source_sha256")
        if (
            not isinstance(expected_dynamic_source, str)
            or file_sha256(dynamic_adm_source) != expected_dynamic_source
        ):
            raise RuntimeError("dynamic ADM source differs from frozen SHA-256")
        shutil.copy2(dynamic_adm_source, source_snapshot / dynamic_adm_source.name)
    if config["agent"].get("residual_meta_policy") is not None:
        expected_residual_source = config.get("residual_meta_source_sha256")
        if (
            not isinstance(expected_residual_source, str)
            or file_sha256(residual_meta_source) != expected_residual_source
        ):
            raise RuntimeError(
                "residual Meta-Harness source differs from frozen SHA-256"
            )
        shutil.copy2(
            residual_meta_source,
            source_snapshot / residual_meta_source.name,
        )

    executor_cuda = configure_executor_cuda(config)
    service, gpu = snapshot_and_verify_service(config, experiment)
    verify_verifier_runtime(config, experiment)

    source_root = Path(str(config["source_root"]))
    dataset = experiment / "dataset"
    dataset.mkdir()
    dataset_manifest = {}
    task_specs = list(config["tasks"])
    for task_spec in task_specs:
        task_name = str(task_spec["name"])
        source_task = source_root / task_name
        observed_sha256 = tree_sha256(source_task)
        if observed_sha256 != task_spec["source_tree_sha256"]:
            raise RuntimeError(f"frozen source tree mismatch: {task_name}")
        shutil.copytree(source_task, dataset / task_name)
        dataset_manifest[task_name] = {
            "tree_sha256": observed_sha256,
            "files": tree_manifest(dataset / task_name),
        }
        if docker_image_id(str(task_spec["docker_image"])) != task_spec["docker_image_id"]:
            raise RuntimeError(f"frozen Docker image mismatch: {task_name}")
    write_json(experiment / "dataset-manifest.json", dataset_manifest)

    runtime = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": command_output(["python3", "--version"]),
        "docker_client": command_output(
            ["docker", "version", "--format", "{{.Client.Version}}"]
        ),
        "docker_server": command_output(
            ["docker", "version", "--format", "{{.Server.Version}}"]
        ),
        "docker_context": command_output(["docker", "context", "show"]),
        "cuda_visible_devices": executor_cuda,
        "remote_model_host": config.get("model_host") if service["_service_kind"] == "remote_relay" else None,
        "remote_GPU_index_is_not_local_device_access": service["_service_kind"] == "remote_relay",
        "model_service_kind": service["_service_kind"],
        "model_service_main_pid": int(service["_live_main_pid"]),
        "model_service_process_ids": service["_live_process_ids"],
        "selected_gpu": gpu["selected_physical_gpu"],
    }
    packages = command_output(["python3", "-m", "pip", "freeze", "--all"])
    (experiment / "packages.txt").write_text(packages + "\n")
    write_json(experiment / "runtime.json", runtime)
    write_json(
        experiment / "command.json",
        {
            "started_at_utc": utc_now(),
            "command": [
                "python3",
                str(runner),
                "--experiment-dir",
                str(experiment),
            ],
            "environment": {
                "CUDA_VISIBLE_DEVICES": executor_cuda,
            },
        },
    )

    max_workers = int(config.get("max_concurrent_trials", 1))
    if max_workers < 1:
        raise ValueError("max_concurrent_trials must be positive")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_trial,
                config=config,
                experiment=experiment,
                dataset=dataset,
                task_spec=task_spec,
            )
            for task_spec in task_specs
        ]
        trials = [future.result() for future in futures]
    valid = [trial for trial in trials if trial["valid_binary_reward"]]
    successes = sum(
        trial["verifier_result"]["reward"] == 1
        for trial in valid
        if isinstance(trial["verifier_result"], Mapping)
    )
    exact_expected_tasks = [trial["task"] for trial in trials] == [
        str(task["name"]) for task in task_specs
    ]
    gates = {
        "exact_expected_tasks": exact_expected_tasks,
        "all_trials_have_valid_binary_rewards": len(valid) == len(task_specs),
        "minimum_successes": successes
        >= int(config["acceptance"]["minimum_successes"]),
        "model_service_alive_after": process_is_alive(
            int(service["_live_main_pid"])
        ),
    }
    if config["mode"] == "agent":
        gates["all_agent_usage_nonempty"] = all(
            isinstance(trial["agent_result"], Mapping)
            and int(trial["agent_result"].get("model_calls") or 0) > 0
            and int(trial["agent_result"].get("input_tokens") or 0) > 0
            for trial in trials
        )
        gates["all_tool_traces_nonempty"] = all(
            isinstance(trial["agent_result"], Mapping)
            and int(trial["agent_result"].get("tool_calls") or 0) > 0
            for trial in trials
        )
    accepted = all(gates.values())
    result = {
        "schema_version": "phiagent-terminalbench-direct-result/v1",
        "status": "WORKING" if accepted else "PARTIAL",
        "finished_at_utc": utc_now(),
        "benchmark": "Terminal-Bench 2.0",
        "official_source_revision": config["official_source_revision"],
        "mode": config["mode"],
        "method": config["method"],
        "tasks": [str(task["name"]) for task in task_specs],
        "trials": trials,
        "valid_binary_reward_cells": len(valid),
        "successes": successes,
        "total_tokens": sum(
            int(trial["agent_result"].get("total_tokens") or 0)
            for trial in trials
            if isinstance(trial["agent_result"], Mapping)
        ),
        "acceptance": {"accepted": accepted, "gates": gates},
        "claim_boundary": config["claim_boundary"],
        "evidence": {
            "config_sha256": file_sha256(config_path),
            "reservation_sha256": file_sha256(reservation_path),
            "git_state_sha256": file_sha256(git_state_path),
            "runner_sha256": file_sha256(runner),
            "dataset_manifest_sha256": file_sha256(
                experiment / "dataset-manifest.json"
            ),
            "runtime_sha256": file_sha256(experiment / "runtime.json"),
            "packages_sha256": file_sha256(experiment / "packages.txt"),
            "gpu_preflight_sha256": file_sha256(
                experiment / "gpu-preflight.json"
            ),
            "dynamic_adm_source_sha256": (
                file_sha256(dynamic_adm_source)
                if config["agent"].get("dynamic_adm_policy") is not None
                else None
            ),
            "residual_meta_source_sha256": (
                file_sha256(residual_meta_source)
                if config["agent"].get("residual_meta_policy") is not None
                else None
            ),
        },
    }
    verifier_manifest = experiment / "verifier-runtime-manifest.json"
    if verifier_manifest.is_file():
        result["evidence"]["verifier_runtime_manifest_sha256"] = file_sha256(
            verifier_manifest
        )
    result["result_sha256"] = canonical_sha256(result)
    write_json(experiment / "results.json", result)
    return result, 0 if accepted else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    result, exit_code = run_experiment(args.experiment_dir.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
