"""Compact public actor inputs without changing native tools, receipts or scoring."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from phiagent.harness.gaia2_calendar_preread_v1 import CalendarPrereadReceiptEngine
from phiagent.harness.gaia2_jev import action_object


POLICY = "public_catalogue_and_history_compaction_v1"
CATALOGUE_HEADER = (
    "Tool records are JSON arrays [name, description, output_type, arguments].\n"
    "Each argument is [name, type, description, remaining_schema_fields]. "
    "All original fields and defaults are retained; an empty remaining object "
    "does not supply a default.\n"
)
OBSERVATION = re.compile(
    r"^(\[OUTPUT OF STEP (?P<step>[0-9]+)\] Observation:\n\*\*\*\n)"
    r"(?P<body>.*?)(\n\*\*\*\n?)$", re.DOTALL,
)


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def pack_catalogue(records: list[dict[str, Any]]) -> str:
    rows = []
    names = set()
    for record in records:
        if set(record) != {"name", "description", "output_type", "inputs"}:
            raise ValueError("A tool record must preserve all four native catalogue fields")
        name = record["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Tool names must be nonempty and unique")
        names.add(name)
        if not isinstance(record["description"], str) or not isinstance(record["inputs"], dict):
            raise ValueError("Native tool descriptions and input schemas are required")
        arguments = []
        for argument, schema in record["inputs"].items():
            if (not isinstance(argument, str) or not isinstance(schema, dict)
                    or "type" not in schema or "description" not in schema):
                raise ValueError("Argument schemas must name their type and description")
            remaining = {key: value for key, value in schema.items()
                         if key not in ("type", "description")}
            arguments.append([argument, schema["type"], schema["description"], remaining])
        rows.append(encoded([name, record["description"], record["output_type"], arguments]))
    text = CATALOGUE_HEADER + "\n".join(rows)
    if unpack_catalogue(text) != records:
        raise ValueError("The compact catalogue failed its exact schema round trip")
    return text


def unpack_catalogue(text: str) -> list[dict[str, Any]]:
    if not text.startswith(CATALOGUE_HEADER):
        raise ValueError("Unknown compact catalogue encoding")
    records = []
    for line in text[len(CATALOGUE_HEADER):].splitlines():
        name, description, output_type, arguments = json.loads(line)
        inputs = {}
        for argument, kind, detail, remaining in arguments:
            if argument in inputs or set(remaining).intersection(("type", "description")):
                raise ValueError("Ambiguous compact argument schema")
            inputs[argument] = {"type": kind, "description": detail, **remaining}
        records.append({
            "name": name, "description": description, "output_type": output_type, "inputs": inputs,
        })
    return records


@dataclass(frozen=True)
class Projection:
    messages: list[dict[str, Any]]
    audit: dict[str, Any]


def project_public_messages(
    messages: list[dict[str, Any]],
    *,
    native_catalogue: str,
    records: list[dict[str, Any]],
) -> Projection:
    if (not messages or messages[0].get("role") != "system"
            or not isinstance(messages[0].get("content"), str)):
        raise ValueError("The native actor system message is required")
    projected = deepcopy(messages)
    changes = []
    compact = pack_catalogue(records)
    system = messages[0]["content"]
    catalogue_status = "native_catalogue_not_uniquely_bound"
    if native_catalogue and system.count(native_catalogue) == 1:
        catalogue_status = "compact_catalogue_not_shorter"
        if len(compact) < len(native_catalogue):
            projected[0]["content"] = system.replace(native_catalogue, compact, 1)
            catalogue_status = "exact_schema_roundtrip"
            changes.append({
                "message": 0, "operation": "tool_catalogue_encoding",
                "native_catalogue_sha256": fingerprint(native_catalogue),
                "schema_sha256": fingerprint(records), "tools": len(records),
                "native_characters": len(native_catalogue), "compact_characters": len(compact),
            })
    seen: dict[str, tuple[int, str]] = {}
    for index in range(len(messages) - 1, 0, -1):
        message = messages[index]
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        match = OBSERVATION.fullmatch(content)
        if match is None or len(match["body"]) < 512:
            continue
        body = match["body"]
        if body in seen:
            retained, step = seen[body]
            replacement = (
                f"[Exact duplicate of the public observation retained at step {step}; "
                f"sha256={hashlib.sha256(body.encode()).hexdigest()}]"
            )
            projected[index]["content"] = (
                content[:match.start("body")] + replacement + content[match.end("body"):]
            )
            changes.append({
                "message": index, "operation": "exact_observation_reference",
                "retained_message": retained, "retained_step": step,
                "observation_sha256": hashlib.sha256(body.encode()).hexdigest(),
            })
        else:
            seen[body] = (index, match["step"])
    for index, message in enumerate(messages[:-2]):
        content = message.get("content")
        if (message.get("role") != "assistant" or not isinstance(content, str)
                or not content.startswith("Thought:") or content.count("Action:") != 1
                or action_object(content) is None):
            continue
        action = "Action:" + content.split("Action:", 1)[1]
        replacement = "Thought: [Earlier deliberation omitted; exact action retained.]\n" + action
        if len(replacement) < len(content):
            projected[index]["content"] = replacement
            changes.append({
                "message": index, "operation": "retain_exact_action",
                "action_sha256": fingerprint(action),
            })
    applied = bool(changes)
    return Projection(projected if applied else messages, {
        "policy": POLICY, "applied": applied, "catalogue_status": catalogue_status,
        "changes": changes, "source_messages_sha256": fingerprint(messages),
        "projected_messages_sha256": fingerprint(projected),
        "source_characters": sum(len(str(row.get("content", ""))) for row in messages),
        "projected_characters": sum(len(str(row.get("content", ""))) for row in projected),
        "public_observations_deleted": 0, "native_tools_removed": 0,
        "native_logs_mutated": False, "new_model_calls": 0,
    })


class CompactCalendarReceiptEngine(CalendarPrereadReceiptEngine):
    def __init__(self, agent, provider, *, skill, record, execute_followups,
                 projector=project_public_messages, policy=POLICY):
        last_projection: dict[str, Any] = {"applied": False}

        def compact_provider(messages, **kwargs):
            from are.simulation.tool_box import get_tool_description_with_args

            tools = list(agent.action_executor.tools.values())
            native = "\n".join(get_tool_description_with_args(tool) for tool in tools)
            records = [
                {"name": tool.name, "description": tool.description,
                 "output_type": tool.output_type, "inputs": tool.inputs}
                for tool in tools
            ]
            projection = projector(
                messages, native_catalogue=native, records=records,
            )
            last_projection.clear()
            last_projection.update(projection.audit)
            record({"source": "public_input_compaction", "agent_id": agent.agent_id,
                    **projection.audit})
            return provider(projection.messages, **kwargs)

        def record_projection(value):
            if last_projection["applied"] and value.get("source") in (
                "receipt_graph_model", "receipt_graph_model_calendar_preread",
                "calendar_availability_preread_v1",
            ):
                value = {**value, "actor_input_projection": policy,
                         "native_fallback_unchanged": False,
                         "native_messages_forwarded_by_identity": False,
                         "native_messages_kwargs_unchanged": False,
                         "native_kwargs_unmodified": True}
            record(value)

        super().__init__(agent, compact_provider, skill=skill, record=record_projection,
                         execute_followups=execute_followups)


def install_adapter(config, directory, role, mode, skill):
    if (config.get("stage") != "train-candidate" or config.get("evaluation_split") != "train"
            or mode != "jev" or not config.get("receipt_execution")):
        raise RuntimeError("This prospective compaction candidate is admitted to TRAIN only")
    from phiagent.harness import gaia2_202060_progress as progress

    progress.ReceiptEngine = CompactCalendarReceiptEngine
    progress.install_adapter(config, directory, role, mode, skill)
