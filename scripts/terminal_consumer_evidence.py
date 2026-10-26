"""Opt-in historical evidence requests with precommitted Terminal action branches."""

from __future__ import annotations

from dataclasses import asdict
import json
import re
import shlex
from collections.abc import Mapping

from phiagent.learning.consumer_evidence import (
    MODES, VERSION, CommittedBranch, ConsumerResolution, PublicAction, PublicReport,
    commit_branch, resolve_branch,
)
from phiagent.learning.evidence_contracts import canonical_json


CAPSULE_BUDGET_BYTES = 2304
MAX_COMMAND_CHARACTERS = 4000
REPORT_HEADER = re.compile(r"\Apublic_report_id=(0|[1-9][0-9]*)\n")
ACTION_SCHEMA = {
    "type": "object", "properties": {"command": {
        "type": "string", "maxLength": MAX_COMMAND_CHARACTERS, "pattern": r"^[^\r\n]*$",
    }},
    "required": ["command"], "additionalProperties": False,
}
REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": "request_public_evidence",
        "description": (
            "Resolve a literal value from a previously observed public report for two committed "
            "pending shell actions. Evidence selects a branch; it is not executed as code. "
            "Unavailable or uninterpretable evidence is UNKNOWN, not the different branch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "report_sequence": {"type": "integer", "minimum": 0},
                "selector": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["json_pointer", "exact_span"]},
                        "pointer": {"type": "string", "enum": ["/text", "/command"]},
                        "prefix": {"type": "string"}, "suffix": {"type": "string"},
                        "occurrence": {"type": "integer", "minimum": 0},
                    },
                    "required": ["kind", "pointer", "prefix", "suffix", "occurrence"],
                    "additionalProperties": False,
                },
                "equals": {},
                "when_equal": ACTION_SCHEMA, "when_different": ACTION_SCHEMA,
                "on_unknown": {
                    "anyOf": [
                        {"type": "string", "enum": ["REPLAN"]},
                        {
                            "type": "object", "properties": {"observe": ACTION_SCHEMA},
                            "required": ["observe"], "additionalProperties": False,
                        },
                    ],
                },
            },
            "required": [
                "report_sequence", "selector", "equals", "when_equal", "when_different", "on_unknown",
            ],
            "additionalProperties": False,
        },
    },
}
SYSTEM_SUPPLEMENT = """
Public tool reports have a chronological public_report_id. They describe historical
reports, not guaranteed current file state. Old output may be shortened in your context.
Use request_public_evidence only if a pending operation depends on a specific reported
value. Commit a selector and two different useful shell actions before lookup. Do not
request arbitrary deleted bytes, merely echo the answer, or treat an output pathname
in the task as evidence that the file exists. The /text field is exactly the previously
shown report body; /command is its issued command. exact_span selects text after the
specified occurrence of prefix and before the following suffix; all delimiters are literal.
json_pointer copies the complete selected field and uses empty prefix/suffix and occurrence 0.
UNKNOWN returns REPLAN or runs only your explicit ordinary observation branch, which is
charged to the same step/time budget. The request may return advisory evidence without
executing the pending action; inspect the returned status and continue using ordinary tools.
"""


def validate_terminal_action(value: Mapping[str, object]) -> PublicAction:
    if not isinstance(value, Mapping) or set(value) != {"command"}:
        raise ValueError("a registered Terminal branch requires exactly one command")
    command = value["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("a registered Terminal branch requires a nonempty shell command")
    if "\x00" in command:
        raise ValueError("shell commands cannot contain a null byte")
    if len(command) > MAX_COMMAND_CHARACTERS or "\r" in command or "\n" in command:
        raise ValueError("pending actions must preserve the common 4000-character single-line transport cap")
    return PublicAction("exec_in_sandbox", canonical_json({"command": command.strip()}))


def obvious_nonconsumer(action: PublicAction) -> bool:
    command = json.loads(action.arguments_json)["command"]
    if any(character in command for character in "|;&`$<>\n"):
        return False
    tokens = shlex.split(command)
    return bool(tokens) and tokens[0] in {":", "true", "false", "echo", "printf", "sleep"}


class TerminalConsumerSession:
    def __init__(self, *, mode: str, budget_bytes: int, acquisition_enabled: bool):
        if mode not in MODES or type(budget_bytes) is not int or budget_bytes != CAPSULE_BUDGET_BYTES:
            raise ValueError("use one frozen consumer mode and the explicit 2304-byte witness budget")
        if type(acquisition_enabled) is not bool:
            raise ValueError("acquisition_enabled must be an explicit boolean")
        self.mode = mode
        self.budget_bytes = budget_bytes
        self.acquisition_enabled = acquisition_enabled
        self.reports = []
        self.actor_reports = ()
        self.visible_reports = ()
        self.entries = ()
        self.traces = []
        self.visible_projection_failures = []
        self.last_turn = -1

    def before_actor(self, visible_messages, *, turn_index: int):
        if type(turn_index) is not int or turn_index <= self.last_turn:
            raise ValueError("one increasing consumer snapshot is required per actor turn")
        self.last_turn = turn_index
        self.actor_reports = tuple(self.reports)
        visible = []
        for message in visible_messages:
            if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                continue
            content = message["content"]
            header = REPORT_HEADER.match(content)
            if header is None:
                continue
            sequence = int(header[1])
            if sequence >= len(self.actor_reports):
                raise ValueError("visible report identity was not issued by this session")
            declaration, separator, body = content[header.end():].partition("\n")
            payload = None
            if declaration.startswith("public_command_json=") and separator:
                try:
                    command = json.loads(declaration.removeprefix("public_command_json="))
                except json.JSONDecodeError:
                    command = None
                if isinstance(command, str):
                    payload = {"command": command, "text": body}
            if payload is None:
                self.visible_projection_failures.append({
                    "turn": turn_index, "report_sequence": sequence,
                    "reason": "public_header_not_interpretable_after_context_edit",
                })
                payload = {"uninterpretable_public_header": True}
            visible.append(PublicReport(sequence, canonical_json(payload)))
        sequences = [report.sequence for report in visible]
        if sequences != sorted(set(sequences)):
            raise ValueError("visible public report identities were duplicated or reordered")
        self.visible_reports = tuple(visible)

    def observe_environment(self, *, command: str, public_content: str) -> str:
        sequence = len(self.reports)
        self.reports.append(PublicReport(
            sequence, canonical_json({"command": command, "text": public_content}),
        ))
        return f"public_report_id={sequence}\npublic_command_json={canonical_json(command)}\n{public_content}"

    def commit_call(self, tool_call: Mapping[str, object]) -> CommittedBranch:
        if self.last_turn < 0:
            raise ValueError("a consumer request must follow an actual actor input snapshot")
        function = tool_call.get("function")
        if not isinstance(tool_call.get("id"), str) or not isinstance(function, Mapping):
            raise ValueError("an evidence request requires its public tool-call linkage")
        if function.get("name") != "request_public_evidence":
            raise ValueError("unsupported evidence-request tool")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("evidence-request arguments must be a JSON string")
        request = commit_branch(json.loads(arguments), validate_action=validate_terminal_action)
        if request.selector.pointer not in ("/text", "/command"):
            raise ValueError("Terminal requests may select only public text or its issued command")
        if obvious_nonconsumer(request.when_equal) and obvious_nonconsumer(request.when_different):
            raise ValueError("logging or no-op-only branches do not qualify as evidence consumers")
        return request

    def resolve(self, request: CommittedBranch) -> ConsumerResolution:
        resolution = resolve_branch(
            request, reports=self.actor_reports, visible_reports=self.visible_reports,
            previous_repair=self.entries, budget_bytes=self.budget_bytes, mode=self.mode,
            acquisition_enabled=self.acquisition_enabled,
        )
        admitted = resolution.repair.admitted and resolution.status != "BUDGET_REFUSED_REPLAN"
        if self.mode == "bound_checked" and admitted:
            self.entries = resolution.repair.entries
        trace = {
            "schema_version": VERSION, "turn": self.last_turn, "mode": self.mode,
            "request": request.as_dict(), "request_sha256": request.sha256,
            "resolution": asdict(resolution),
            "actual_environment_action_attempted": False,
            "snapshot_report_count": len(self.actor_reports),
            "future_same_response_tool_outputs_available_to_resolver": False,
            "actual_managed_edits": (
                [asdict(edit) for edit in resolution.repair.edits]
                if self.mode == "bound_checked" and admitted else []
            ),
            "witness_capsule_bytes_are_not_actor_tokens": True,
            "reference_repair_plan_is_diagnostic_only_for_lookup_controls": self.mode != "bound_checked",
        }
        self.traces.append(trace)
        return resolution

    def mark_action_attempt(self, *, request_sha256: str, command: str):
        if not self.traces or self.traces[-1]["request_sha256"] != request_sha256:
            raise ValueError("the actual action must be linked to its most recent committed request")
        trace = self.traces[-1]
        action = trace["resolution"]["action"]
        if action is None or json.loads(action["arguments_json"])["command"] != command:
            raise ValueError("the environment command differs from its selected committed action")
        trace["actual_environment_action_attempted"] = True

    @staticmethod
    def feedback(resolution: ConsumerResolution) -> str:
        return canonical_json({
            "status": resolution.status, "request_sha256": resolution.request_sha256,
            "historical_evidence": json.loads(resolution.answer_capsule) if resolution.answer_capsule else None,
            "environment_action_executed": False,
            "next": "Use an ordinary tool or revise the plan; no action ran for this response.",
        })

    def summary(self) -> dict:
        return {
            "mode": self.mode, "acquisition_enabled": self.acquisition_enabled,
            "valid_committed_requests": len(self.traces),
            "bound_environment_action_attempts": sum(
                trace["actual_environment_action_attempted"] and trace["resolution"]["consumer_bound"]
                for trace in self.traces
            ),
            "explicit_acquisition_attempts": sum(
                trace["actual_environment_action_attempted"]
                and trace["resolution"]["requires_charged_observation"]
                for trace in self.traces
            ),
            "public_reports": len(self.reports),
            "public_archive_bytes": sum(len(report.payload_json.encode()) for report in self.reports),
            "visible_projection_failures": self.visible_projection_failures,
            "traces": self.traces,
            "information_boundary": "Previously shown, output-capped public reports only; no raw uncapped stdout, verifier input, or future same-response tool outputs.",
            "quality_or_semantic_necessity_claimed": False,
        }
