"""Public-byte deduplication and charged, instruction-only output-file checks.

No task catalog, reference solution, scorer, or host filesystem is consulted.
Metadata checks are a completion aid, never a substitute for native reward.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re
import shlex
from typing import Mapping, Sequence


VERSION = "dual-terminal-completion/v1"
MODES = ("dedup", "completion", "full")
CONFIG = {
    "preserve_recent_tool_messages": 2,
    "maximum_completion_checks": 2,
    "maximum_output_paths": 8,
}
_PATH = r"/[A-Za-z0-9_+.-]+(?:/[A-Za-z0-9_+.-]+)*"
_OUTPUT = re.compile(
    r"\b(?:create|write|save|store|output|produce|place|generate)\b"
    r"(?P<context>[^.\n;!?]{0,120}?)"
    r"(?P<quote>[`\"']?)(?P<path>" + _PATH + r")(?P=quote)"
    r"(?=$|[\s.,;:!?`\"')\]])",
    re.IGNORECASE,
)
_PROTECTED_PARTS = {
    "tests", "test", "solution", "solutions", "verifier", "reward", "rewards",
    ".git", "proc", "sys", "dev", "run", "etc",
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()


def required_output_paths(instruction: str) -> tuple[str, ...]:
    """Only explicit absolute output filenames in positive imperative clauses."""
    paths = []
    for match in _OUTPUT.finditer(instruction):
        before = re.split(r"[\n.;!?]", instruction[:match.start()])[-1].lower()
        context = match["context"].lower()
        path = match["path"]
        parts = PurePosixPath(path).parts
        if (
            re.search(r"\b(?:not|never|don't|avoid|example|e\.g|if|unless)\b", before)
            or re.search(r"\b(?:from|read|input|example|directory|folder|optionally)\b", context)
            or any(part.lower() in _PROTECTED_PARTS or part in (".", "..") for part in parts)
            or str(PurePosixPath(path)) != path
            or (
                "." not in PurePosixPath(path).name.strip(".")
                and not re.search(r"\b(?:file|script|binary|executable)\b", context)
            )
            or path.endswith(".")
        ):
            continue
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def deduplicate_public_tools(messages: Sequence[Mapping], *, preserve_recent: int) -> tuple[list, list]:
    """Replace old exact duplicate observations, retaining their newest full body."""
    result = deepcopy(list(messages))
    pending, rows, ids = {}, [], set()
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            if pending:
                raise ValueError("incomplete public tool-call group")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("malformed public tool-call group")
            for call in calls:
                if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                    raise ValueError("malformed public tool-call function")
                call_id, function = call.get("id"), call.get("function", {})
                if not isinstance(call_id, str) or not call_id or call_id in ids:
                    raise ValueError("ambiguous public tool-call identity")
                ids.add(call_id)
                pending[call_id] = function
        elif message.get("role") == "tool":
            call_id, content = message.get("tool_call_id"), message.get("content")
            if call_id not in pending or not isinstance(content, str):
                raise ValueError("unlinked public tool feedback")
            function = pending.pop(call_id)
            if function.get("name") == "exec_in_sandbox":
                rows.append((index, call_id, digest([function, content])))
        elif pending:
            raise ValueError("incomplete public tool-call group")
    if pending:
        raise ValueError("incomplete public tool-call group")
    protected = {row[0] for row in rows[-preserve_recent:]} if preserve_recent else set()
    latest, traces = {}, []
    for index, call_id, key in reversed(rows):
        if key not in latest:
            latest[key] = (index, call_id)
            continue
        if index in protected:
            continue
        anchor, anchor_id = latest[key]
        content = messages[index]["content"]
        reference = json.dumps({
            "dual_terminal_exact_public_duplicate": {
                "retained_message_index": anchor,
                "public_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "meaning": (
                    "Zero-based outgoing message index retains identical command/output bytes; "
                    "historical observations, not filesystem stasis."
                ),
            },
        }, sort_keys=True, separators=(",", ":"))
        if len(reference) >= len(content):
            continue
        result[index]["content"] = reference
        traces.append({
            "message_index": index, "retained_message_index": anchor,
            "tool_call_id": call_id, "retained_tool_call_id": anchor_id,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "characters_removed": len(content) - len(reference),
        })
    return result, list(reversed(traces))


def output_check_command(paths: Sequence[str]) -> str:
    """Use the existing shell tool: no interpreter dependency or file-content reads."""
    if not paths:
        raise ValueError("an explicit output file is required")
    commands = []
    for index, path in enumerate(paths):
        quoted = shlex.quote(path)
        commands.append(
            f"if [ -f {quoted} ]; then printf 'dual_terminal_file:{index}:present\\n'; "
            f"else printf 'dual_terminal_file:{index}:missing_or_not_regular\\n'; fi"
        )
    return "; ".join(commands)


class DualTerminalSession:
    def __init__(self, *, instruction: str, mode: str, config: Mapping):
        if mode not in MODES or dict(config) != CONFIG:
            raise ValueError("unknown dual_terminal mode or unfrozen configuration")
        self.mode = mode
        self.paths = required_output_paths(instruction)
        if len(self.paths) > CONFIG["maximum_output_paths"]:
            self.paths = ()
        self.traces: list[dict] = []
        self.checks = 0
        self.last_checked_action_count: int | None = None
        self.action_count = 0
        self.pending_call: str | None = None

    def prepare(self, messages: Sequence[Mapping], *, turn_index: int) -> list:
        result, references, fallback = deepcopy(list(messages)), [], None
        if self.mode in ("dedup", "full"):
            try:
                result, references = deduplicate_public_tools(
                    messages, preserve_recent=CONFIG["preserve_recent_tool_messages"],
                )
            except ValueError as exc:
                fallback = str(exc)
        self.traces.append({
            "event": "public_context", "turn_index": turn_index,
            "input_sha256": digest(messages), "output_sha256": digest(result),
            "exact_duplicate_references": references,
            "novel_tool_output_removed": False, "tools_skipped": 0,
            "full_context_fallback": fallback,
        })
        return result

    def completion_call(
        self, *, turn_index: int, remaining_turns: int, used_call_ids: Sequence[str] = (),
    ) -> dict | None:
        reason = None
        if self.mode == "dedup":
            reason = "completion_ablation"
        elif not self.paths:
            reason = "no_unambiguous_explicit_output_file"
        elif remaining_turns < 1:
            reason = "no_remaining_actor_turn_for_charged_feedback"
        elif self.checks >= CONFIG["maximum_completion_checks"]:
            reason = "completion_check_limit"
        elif self.last_checked_action_count == self.action_count:
            reason = "no_action_since_previous_check"
        if reason:
            self.traces.append({"event": "completion_check_not_inserted", "reason": reason})
            return None
        self.checks += 1
        self.last_checked_action_count = self.action_count
        self.pending_call = f"dual-terminal-check-{turn_index + 1}-{self.checks}"
        while self.pending_call in used_call_ids:
            self.pending_call += "-harness"
        call = {
            "id": self.pending_call, "type": "function",
            "function": {
                "name": "exec_in_sandbox",
                "arguments": json.dumps({"command": output_check_command(self.paths)}),
            },
        }
        self.traces.append({
            "event": "completion_check_requested", "turn_index": turn_index,
            "origin": "harness_not_model", "call": deepcopy(call),
            "required_paths_from_public_instruction": list(self.paths),
            "charged_environment_action": True,
            "official_reward_or_file_content_access": False,
        })
        return call

    def observe(self, *, call_id: str, public_content: str, executed: bool) -> str:
        if call_id != self.pending_call:
            if executed:
                self.action_count += 1
            return public_content
        self.pending_call = None
        parsed = re.fullmatch(
            r"exit_code=0\ntimed_out=false\nstdout:\n"
            r"(?P<rows>(?:dual_terminal_file:\d+:(?:present|missing_or_not_regular)\n)+)"
            r"\nstderr:\n",
            public_content,
        ) if executed else None
        missing = []
        statuses = {}
        if parsed:
            for index, status in re.findall(r"dual_terminal_file:(\d+):(\w+)", parsed["rows"]):
                if int(index) in statuses:
                    parsed = None
                    break
                statuses[int(index)] = status
            if set(statuses) != set(range(len(self.paths))):
                parsed = None
        if parsed:
            missing = [path for index, path in enumerate(self.paths)
                       if statuses[index] != "present"]
        self.traces.append({
            "event": "completion_check_observed", "executed": executed,
            "public_feedback_sha256": hashlib.sha256(public_content.encode()).hexdigest(),
            "metadata_parse_valid": bool(parsed), "missing_required_files": missing,
            "task_success_inferred": False,
        })
        if not parsed:
            guidance = "Output-file metadata check inconclusive; use ordinary public task checks."
        elif missing:
            guidance = (
                "Required output files missing or not regular: " + json.dumps(missing)
                + ". Repair only these explicit deliverables using the task instruction and "
                "public evidence, then use the remaining ordinary tool/turn budget to verify."
            )
        else:
            guidance = (
                "Explicit output files exist. This does not establish correct contents or task "
                "success; use the task's public requirements to decide whether further work is needed."
            )
        return public_content + "\n[dual_terminal harness-origin completion feedback]\n" + guidance

    def summary(self) -> dict:
        return {
            "version": VERSION, "mode": self.mode, "checks_requested": self.checks,
            "required_output_paths": list(self.paths), "traces": deepcopy(self.traces),
            "official_reward_substituted": False, "forced_model_abort": False,
        }
