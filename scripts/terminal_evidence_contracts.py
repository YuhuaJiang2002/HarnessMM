"""Historical-report query adapter for contract-preserving Terminal context repair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from phiagent.learning.evidence_contracts import (
    CapsuleEntry, Obligation, Observation, PublicHistory, QueryScope, canonical_json,
    encoded_capsule, plan_context_repair,
)


VERSION = "terminal-historical-evidence-queries/v1"
MODES = ("checked_obligations", "typed_recency")
MAX_FAILURE_QUERIES = 2
MAX_TASK_PATH_QUERIES = 2
MAX_EXCERPT_CHARACTERS = 192
MAX_COMMAND_CHARACTERS = 96
CAPSULE_BUDGET_BYTES = 2304
PATH = re.compile(r"/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+")
DIAGNOSTIC = re.compile(r"\b(?:error|failed|failure|traceback|exception|assertion|not found|missing)\b", re.I)


@dataclass(frozen=True)
class PublicToolEvent:
    event_id: str
    sequence: int
    command: str | None
    content: str


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def bounded_excerpt(value: str, needle: str | None = None) -> str:
    if len(value) <= MAX_EXCERPT_CHARACTERS:
        return value
    index = value.find(needle) if needle is not None else -1
    if index >= 0:
        begin = max(0, index - 32)
        return value[begin:begin + MAX_EXCERPT_CHARACTERS]
    half = MAX_EXCERPT_CHARACTERS // 2
    return value[:half] + value[-half:]


def public_events(messages: Sequence[Mapping[str, object]]) -> tuple[PublicToolEvent, ...]:
    commands = {}
    events = []
    for message in messages:
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("public tool-call declarations must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("public tool-call declarations must be structured objects")
                identity = call.get("id")
                if not isinstance(identity, str) or identity in commands:
                    raise ValueError("public assistant tool-call identities must be unique")
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    raise ValueError("public tool function declaration is malformed")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                command = arguments.get("command") if isinstance(arguments, dict) else None
                commands[identity] = command if isinstance(command, str) else None
        elif message.get("role") == "tool":
            identity, content = message.get("tool_call_id"), message.get("content")
            if not isinstance(identity, str) or identity not in commands or not isinstance(content, str):
                raise ValueError("public tool output lacks its actor-visible linkage or text")
            if any(event.event_id == identity for event in events):
                raise ValueError("duplicate public tool result")
            events.append(PublicToolEvent(identity, len(events), commands[identity], content))
    return tuple(events)


def parse_public_status(content: str) -> tuple[int | None, bool] | None:
    match = re.match(r"\Aexit_code=(-?\d+|None)\ntimed_out=(true|false)\nstdout:\n", content)
    if match is None or "\nstderr:\n" not in content:
        return None
    return (None if match[1] == "None" else int(match[1]), match[2] == "true")


def declared_task_paths(instruction: str) -> tuple[str, ...]:
    quoted = re.findall(r"`([^`]+)`", instruction)
    candidates = {value for value in quoted if PATH.fullmatch(value)}
    unquoted = re.sub(r"`[^`]*`", "", instruction)
    candidates.update(match[0].rstrip(".,;:") for match in PATH.finditer(unquoted))
    return tuple(sorted(
        (path for path in candidates if len(path) <= 128 and path not in ("/app", "/tmp", "/")),
        key=lambda path: (-len(path), path),
    )[:MAX_TASK_PATH_QUERIES])


def command_key(command: str) -> str:
    return "invocation:" + digest(command)


def project_events(events: Sequence[PublicToolEvent], paths: Sequence[str], canonical_ids=None) -> PublicHistory:
    projected = []
    sequence = 0
    for event in events:
        identity = canonical_ids[event.event_id] if canonical_ids is not None else event.event_id
        status = parse_public_status(event.content)
        if event.command is not None:
            value = None
            if status is not None:
                lines = [line for line in event.content.splitlines() if DIAGNOSTIC.search(line)]
                excerpt = "\n".join(lines[:2]) if lines else event.content
                value = {
                    "reported_command_prefix": event.command[:MAX_COMMAND_CHARACTERS],
                    "reported_exit_code": status[0], "reported_timed_out": status[1],
                    "public_diagnostic_excerpt": bounded_excerpt(excerpt),
                    "excerpt_is_complete_output": excerpt == event.content and len(excerpt) <= MAX_EXCERPT_CHARACTERS,
                    "current_workspace_truth_claimed": False,
                }
            projected.append(Observation(
                identity + ":invocation", sequence, command_key(event.command),
                canonical_json(value) if value is not None else None, digest(event.content),
            ))
            sequence += 1
        for path in paths:
            if path not in event.content:
                continue
            lines = [line for line in event.content.splitlines() if path in line]
            value = {
                "declared_task_literal": path,
                "public_mention": bounded_excerpt(lines[-1], path),
                "file_existence_or_goal_completion_claimed": False,
            }
            projected.append(Observation(
                identity + ":path:" + digest(path)[:12], sequence, "mention:" + path,
                canonical_json(value), digest(event.content),
            ))
            sequence += 1
    return PublicHistory(projected)


def obligations_for_history(events: Sequence[PublicToolEvent], paths: Sequence[str]) -> tuple[Obligation, ...]:
    failures = {}
    for event in events:
        status = parse_public_status(event.content)
        if event.command is None or status is None:
            continue
        key = command_key(event.command)
        if status[0] not in (None, 0) or status[1]:
            failures[key] = event.sequence
    keys = sorted(failures, key=lambda key: (-failures[key], key))[:MAX_FAILURE_QUERIES]
    queries = [Obligation("failure-report:" + key.removeprefix("invocation:")[:12], key, QueryScope.LATEST)
               for key in keys]
    queries.extend(Obligation("task-mention:" + digest(path)[:12], "mention:" + path, QueryScope.LATEST) for path in paths)
    return tuple(queries)


class TerminalEvidenceContractSession:
    """Fixed historical-query prototype; its usefulness is an empirical question."""

    def __init__(self, *, instruction: str, mode: str, budget_bytes: int = CAPSULE_BUDGET_BYTES):
        if mode not in MODES or budget_bytes != CAPSULE_BUDGET_BYTES:
            raise ValueError("only the prospectively fixed modes and capsule byte budget are supported")
        self.mode = mode
        self.paths = declared_task_paths(instruction)
        self.budget_bytes = budget_bytes
        self.entries = ()
        self.traces = []
        self.fallback_full_context = False
        self._last_turn = -1

    def before_actor(self, *, full_public_messages, visible_messages, turn_index: int):
        if type(turn_index) is not int or turn_index <= self._last_turn:
            raise ValueError("use exactly one increasing evidence-contract decision per logical actor turn")
        self._last_turn = turn_index
        events = public_events(full_public_messages)
        visible_events = public_events(visible_messages)
        identities = {event.event_id: f"public-event-{index:06d}" for index, event in enumerate(events)}
        if any(event.event_id not in identities for event in visible_events):
            raise ValueError("visible context contains a tool event outside the recorded public history")
        queries = obligations_for_history(events, self.paths)
        history = project_events(events, self.paths, identities)
        visible = project_events(visible_events, self.paths, identities)
        plan = plan_context_repair(history, visible, self.entries, queries, budget_bytes=self.budget_bytes)
        self.fallback_full_context = self.mode == "checked_obligations" and not plan.admitted
        message = None
        if self.mode == "checked_obligations":
            if plan.admitted:
                self.entries = plan.entries
                if self.entries:
                    message = encoded_capsule(self.entries)
        else:
            records = []
            seen = set()
            for observation in reversed(history.observations):
                if observation.key in seen:
                    continue
                seen.add(observation.key)
                query_id = ("task-mention:" + digest(observation.key.removeprefix("mention:"))[:12]
                            if observation.key.startswith("mention:") else
                            "failure-report:" + observation.key.removeprefix("invocation:")[:12])
                query = Obligation(query_id, observation.key, QueryScope.LATEST)
                answer = history.answer(query)
                if not answer.known:
                    continue
                record = CapsuleEntry(query_id, answer)
                if len(encoded_capsule((*records, record)).encode()) > self.budget_bytes:
                    break
                records.append(record)
            if records and any(gap.value == "retention_or_representation_gap" for _, gap in plan.gaps):
                message = encoded_capsule(records)
        self.traces.append({
            "schema_version": VERSION, "turn_index": turn_index, "mode": self.mode,
            "queries": [query.query_id for query in queries],
            "gaps": {key: value.value for key, value in plan.gaps},
            "plan_admitted": plan.admitted, "plan_reason": plan.reason,
            "fallback_full_parent_context": self.fallback_full_context,
            "edits": ([{"operation": edit.kind.value, "query_id": edit.query_id} for edit in plan.edits]
                      if self.mode == "checked_obligations" else []),
            "edit_semantics": "checked_transaction" if self.mode == "checked_obligations" else "fixed_recency_not_adaptive_edits",
            "capsule_bytes": len(message.encode()) if message is not None else 0,
            "message_sha256": digest(message) if message is not None else None,
            "new_model_tool_environment_calls": 0,
            "unsupported_public_event_ids": [
                event.event_id for event in events
                if event.command is None or parse_public_status(event.content) is None
            ],
            "historical_reports_not_state_or_quality_certificates": True,
        })
        return message
