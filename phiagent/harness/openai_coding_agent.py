"""Bounded coding-agent transport for OpenAI-compatible local model servers.

The official Meta-Harness release delegates proposal generation to a coding
agent wrapper.  This module is a provider adapter for that boundary: it keeps
the released Meta-Harness outer loop unchanged while allowing a Qwen or other
OpenAI-compatible model to use the same filesystem-oriented tool surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import glob as glob_module
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest
import uuid

from phiagent.harness.observation_pack import ObservationPack


@dataclass
class CodingToolCall:
    name: str
    tool_id: str
    input: dict[str, Any]
    output: str = ""
    is_error: bool = False


@dataclass
class OpenAICodingSessionResult:
    prompt: str
    text: str
    tool_calls: list[CodingToolCall]
    files_read: dict[str, dict[str, int]]
    files_written: dict[str, dict[str, int]]
    token_usage: dict[str, int]
    duration_seconds: float
    model: str
    session_id: str
    exit_code: int
    cost_usd: float
    raw_events: list[dict[str, Any]]
    command: list[str] | None = None
    cwd: str | None = None
    stderr: str = ""
    skill: list[dict[str, str]] | None = None
    name: str | None = None
    log_dir: str | None = None
    efficiency: dict[str, Any] | None = None
    usage_accounting: dict[str, Any] | None = None

    def show(self) -> None:
        if self.exit_code != 0:
            print(f"  FAILED (exit={self.exit_code}) {(self.stderr or '')[:300]}")
            return
        for call in self.tool_calls:
            arg = (
                call.input.get("file_path")
                or call.input.get("pattern")
                or call.input.get("command")
                or call.input.get("prompt")
                or ""
            )
            suffix = " ERR" if call.is_error else ""
            print(f"  tool: {call.name}({str(arg)[:120]}){suffix}")
        if self.text.strip():
            print(f"  text: {self.text.strip().replace(chr(10), ' ')[:200]}")
        print(
            f"  {self.token_usage.get('input_tokens', 0)}in/"
            f"{self.token_usage.get('output_tokens', 0)}out  "
            f"{self.duration_seconds:.1f}s"
        )


class OpenAIChatTransport:
    """Small dependency-free client for one OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        retries: int = 2,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                req = urlrequest.Request(
                    self.endpoint, data=encoded, headers=headers, method="POST"
                )
                with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("chat response must be one JSON object")
                return payload
            except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.retries:
                    break
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")


def _tool_schema(name: str, *, action_fusion: bool = False) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {
        "Read": {
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 4000},
            },
            "additionalProperties": False,
        },
        "Glob": {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "Grep": {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "Write": {
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "Edit": {
            "type": "object",
            "required": ["file_path", "old_string", "new_string"],
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "Bash": {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            "additionalProperties": False,
        },
        "Agent": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
                "description": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "ObsRecall": {
            "type": "object",
            "required": ["observation_id"],
            "properties": {
                "observation_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8000},
            },
            "additionalProperties": False,
        },
    }
    descriptions = {
        "Read": "Read a UTF-8 text file inside the workspace.",
        "Glob": "List workspace paths matching a glob pattern.",
        "Grep": "Search text files in the workspace with a regular expression.",
        "Write": "Create or replace one UTF-8 text file in the workspace.",
        "Edit": "Replace an exact string in one workspace text file.",
        "Bash": "Run one bounded shell command in the workspace.",
        "Agent": "Ask a bounded subagent to analyze or implement one scoped workspace task.",
        "ObsRecall": (
            "Recall an exact archived tool observation by its session-local ID. "
            "offset and limit count characters; follow next_offset for more pages."
        ),
    }
    if name not in schemas:
        raise ValueError(f"unsupported coding tool: {name}")
    if action_fusion and name in {"Write", "Edit"}:
        schemas[name]["properties"]["validation"] = schemas["Bash"]
        descriptions[name] += (
            " When a follow-up check is already known, include validation to run it "
            "after a successful edit in this same call. Bash restrictions still apply."
        )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": descriptions[name],
            "parameters": schemas[name],
        },
    }


class CodingToolRuntime:
    def __init__(
        self, cwd: Path, *, max_output_characters: int | None = 60000,
        action_fusion: bool = False,
    ) -> None:
        self.cwd = cwd.resolve()
        self.max_output_characters = max_output_characters
        self.action_fusion = action_fusion
        self.fused_validations = 0

    def _path(self, value: object) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("file path must be a non-empty string")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        candidate = candidate.resolve(strict=False)
        try:
            candidate.relative_to(self.cwd)
        except ValueError as exc:
            raise PermissionError("path escapes coding-agent workspace") from exc
        return candidate

    def _bounded(self, text: str) -> str:
        if self.max_output_characters is None or len(text) <= self.max_output_characters:
            return text
        return text[: self.max_output_characters] + "\n...[output truncated]"

    @staticmethod
    def _bash_arguments(arguments: Mapping[str, Any]) -> tuple[str, int]:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise TypeError("Bash command must be a non-empty string")
        forbidden = re.compile(
            r"(?:harbor\s+run|run_eval\.sh|terminal[-_ ]bench|\bdocker\b|"
            r"\bcurl\b|\bwget\b|git\s+clone)",
            re.IGNORECASE,
        )
        if forbidden.search(command):
            raise PermissionError("command crosses the frozen proposer/evaluator boundary")
        timeout = min(300, max(1, int(arguments.get("timeout_seconds", 120))))
        return command, timeout

    def _run_bash(self, arguments: Mapping[str, Any]) -> tuple[str, int]:
        command, timeout = self._bash_arguments(arguments)
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = completed.stdout
        if completed.stderr:
            output += "\n[stderr]\n" + completed.stderr
        output += f"\n[exit_code={completed.returncode}]"
        return output, completed.returncode

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name in {"Write", "Edit"} and "validation" in arguments:
            if not self.action_fusion:
                raise PermissionError("fused validation requires action_fusion and Bash permission")
            validation = arguments["validation"]
            if (
                not isinstance(validation, Mapping)
                or set(validation) - {"command", "timeout_seconds"}
            ):
                raise TypeError("validation must contain only Bash command and timeout_seconds")
            self._bash_arguments(validation)
            edit_receipt = self.execute(name, {
                key: value for key, value in arguments.items() if key != "validation"
            })
            self.fused_validations += 1
            try:
                output, returncode = self._run_bash(validation)
            except subprocess.TimeoutExpired as exc:
                partial = []
                for label, payload in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                    if payload:
                        text = (
                            payload.decode("utf-8", errors="replace")
                            if isinstance(payload, bytes) else str(payload)
                        )
                        partial.append(f"[{label}]\n{text}")
                raise RuntimeError(self._bounded(
                    f"{edit_receipt}\n[validation timed out after {exc.timeout}s; "
                    "edit remains applied]\n"
                    + "\n".join(partial)
                )) from exc
            except OSError as exc:
                raise RuntimeError(f"{edit_receipt}\n[validation did not run: {exc}]") from exc
            if returncode != 0:
                raise RuntimeError(self._bounded(
                    f"{edit_receipt}\n[validation failed; exit_code={returncode}; "
                    f"edit remains applied]\n{output}"
                ))
            return self._bounded(f"{edit_receipt}\n[validation]\n{output}")
        if name == "Read":
            path = self._path(arguments.get("file_path"))
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            offset = max(1, int(arguments.get("offset", 1)))
            limit = min(4000, max(1, int(arguments.get("limit", 2000))))
            selected = lines[offset - 1 : offset - 1 + limit]
            return self._bounded(
                "\n".join(
                    f"{line_no:6d}\t{line}"
                    for line_no, line in enumerate(selected, start=offset)
                )
            )
        if name == "Glob":
            base = self._path(arguments.get("path", "."))
            pattern = str(arguments.get("pattern", ""))
            matches = sorted(
                Path(value).resolve(strict=False)
                for value in glob_module.glob(str(base / pattern), recursive=True)
            )
            relative = []
            for match in matches[:2000]:
                try:
                    relative.append(str(match.relative_to(self.cwd)))
                except ValueError:
                    continue
            return self._bounded("\n".join(relative))
        if name == "Grep":
            regex = re.compile(str(arguments.get("pattern", "")))
            base = self._path(arguments.get("path", "."))
            file_glob = str(arguments.get("glob", "**/*"))
            paths = [base] if base.is_file() else sorted(base.glob(file_glob))
            rows: list[str] = []
            for path in paths:
                if not path.is_file() or len(rows) >= 2000:
                    continue
                try:
                    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
                except (OSError, UnicodeError):
                    continue
                for line_number, line in enumerate(lines, start=1):
                    if regex.search(line):
                        rows.append(
                            f"{path.relative_to(self.cwd)}:{line_number}:{line}"
                        )
                        if len(rows) >= 2000:
                            break
            return self._bounded("\n".join(rows))
        if name == "Write":
            path = self._path(arguments.get("file_path"))
            content = arguments.get("content")
            if not isinstance(content, str):
                raise TypeError("Write content must be a string")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"wrote {path.relative_to(self.cwd)} ({len(content)} characters)"
        if name == "Edit":
            path = self._path(arguments.get("file_path"))
            old = arguments.get("old_string")
            new = arguments.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                raise TypeError("Edit strings must be strings and old_string non-empty")
            content = path.read_text(encoding="utf-8")
            count = content.count(old)
            if count == 0:
                raise ValueError("old_string was not found")
            replace_all = bool(arguments.get("replace_all", False))
            if count > 1 and not replace_all:
                raise ValueError("old_string is not unique; set replace_all=true")
            updated = content.replace(old, new, -1 if replace_all else 1)
            path.write_text(updated, encoding="utf-8")
            return f"edited {path.relative_to(self.cwd)} ({count if replace_all else 1} replacement(s))"
        if name == "Bash":
            output, _returncode = self._run_bash(arguments)
            return self._bounded(output)
        raise ValueError(f"unsupported coding tool: {name}")


def _load_skills(paths: Sequence[str] | None) -> list[dict[str, str]]:
    loaded = []
    for raw in paths or ():
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            candidates = [path / "SKILL.md", path / "skill.md"]
            path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if path.is_file():
            loaded.append(
                {"path": str(path), "name": path.parent.name, "content": path.read_text(encoding="utf-8")}
            )
    return loaded


def _safe_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat response lacks choices[0].message")
    return dict(message)


def _has_complete_usage(response: Mapping[str, Any]) -> bool:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return False
    counts = (
        usage.get("prompt_tokens", usage.get("input_tokens")),
        usage.get("completion_tokens", usage.get("output_tokens")),
    )
    return all(type(count) is int and count >= 0 for count in counts)


def _write_log(result: OpenAICodingSessionResult, log_dir: str | None) -> None:
    if not log_dir:
        return
    directory = Path(log_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", result.name or "session")
    path = directory / f"{stamp}-{slug}-{result.session_id[:8]}.json"
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_openai_coding_agent(
    *,
    prompt: str,
    model: str,
    transport: Any,
    allowed_tools: Sequence[str],
    cwd: str,
    system_prompt: str | None = None,
    skills: Sequence[str] | None = None,
    timeout_seconds: float = 2400,
    max_turns: int = 80,
    max_tokens: int = 32768,
    temperature: float = 1.0,
    reasoning_effort: str | None = "xhigh",
    log_dir: str | None = None,
    name: str | None = None,
    progress: bool | Callable[[dict[str, Any], list[CodingToolCall]], None] = True,
    observation_pack: bool = False,
    action_fusion: bool = False,
) -> OpenAICodingSessionResult:
    start = time.monotonic()
    for flag, value in (("observation_pack", observation_pack), ("action_fusion", action_fusion)):
        if type(value) is not bool:
            raise TypeError(f"{flag} must be a boolean")
    session_id = str(uuid.uuid4())
    root = Path(cwd).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    skill_rows = _load_skills(skills)
    skill_text = "\n\n".join(
        f"## Skill: {row['name']}\n{row['content']}" for row in skill_rows
    )
    system = (
        "You are the proposal coding agent inside a frozen Meta-Harness outer loop. "
        "Inspect prior scores and traces, edit candidate harness source, and write the "
        "exact pending_eval.json requested by the task. The benchmark evaluator is "
        "outside your authority; never invoke it or inspect hidden task data. Use tools "
        "for every claimed filesystem change.\n\n"
        + (skill_text + "\n\n" if skill_text else "")
        + (system_prompt or "")
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if observation_pack and not log_dir:
        raise ValueError("observation_pack requires log_dir for persistent evidence")
    if not observation_pack and "ObsRecall" in allowed_tools:
        raise ValueError("ObsRecall requires observation_pack")
    recall_tools = ("ObsRecall",) if observation_pack else ()
    allowed = tuple(dict.fromkeys((*allowed_tools, *recall_tools)))
    fusion_enabled = action_fusion and "Bash" in allowed
    tool_schemas = [_tool_schema(tool, action_fusion=fusion_enabled) for tool in allowed]
    pack = (
        ObservationPack(Path(log_dir).expanduser().resolve() / session_id / "observations")
        if observation_pack and log_dir else None
    )
    runtime = CodingToolRuntime(
        root, max_output_characters=None if pack else 60000, action_fusion=fusion_enabled,
    )
    calls: list[CodingToolCall] = []
    raw_events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    usage_complete = True
    reported_responses = 0
    delegated_sessions: list[dict[str, Any]] = []
    stderr = ""
    exit_code = 124

    for turn in range(max_turns):
        if time.monotonic() - start >= timeout_seconds:
            stderr = f"session timed out after {timeout_seconds} seconds"
            break
        response = transport.complete(
            model=model,
            messages=pack.project(messages) if pack else messages,
            tools=tool_schemas,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        raw_events.append(response)
        reported_responses += 1
        usage_complete = usage_complete and _has_complete_usage(response)
        response_usage = response.get("usage", {})
        if isinstance(response_usage, Mapping):
            usage["input_tokens"] += int(
                response_usage.get("prompt_tokens", response_usage.get("input_tokens", 0)) or 0
            )
            usage["output_tokens"] += int(
                response_usage.get("completion_tokens", response_usage.get("output_tokens", 0)) or 0
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            stderr = "chat response lacks choices"
            exit_code = 1
            break
        message = _safe_message(choices[0])
        content = message.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            messages.append({"role": "assistant", "content": content or ""})
            exit_code = 0
            break
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls,
        }
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        if isinstance(reasoning, str) and reasoning:
            assistant_message["reasoning_content"] = reasoning
        messages.append(assistant_message)
        for raw_call in tool_calls:
            if not isinstance(raw_call, Mapping):
                continue
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                continue
            tool_name = str(function.get("name", ""))
            tool_id = str(raw_call.get("id", f"call-{len(calls)}"))
            raw_arguments = function.get("arguments", "{}")
            arguments: dict[str, Any] = {}
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else dict(raw_arguments)
                )
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be an object")
                if tool_name not in allowed:
                    raise PermissionError(f"tool {tool_name} is not allowed")
                if tool_name == "Agent":
                    sub_prompt = str(arguments.get("prompt", ""))
                    sub_result = run_openai_coding_agent(
                        prompt=sub_prompt,
                        model=model,
                        transport=transport,
                        allowed_tools=tuple(
                            value for value in allowed if value != "Agent"
                        ),
                        cwd=str(root),
                        system_prompt=(
                            "Complete only the scoped subtask. You may edit files when "
                            "the subtask explicitly requests implementation, but never "
                            "run the benchmark evaluator or inspect hidden task data."
                        ),
                        timeout_seconds=min(600, max(30, timeout_seconds - (time.monotonic() - start))),
                        max_turns=24,
                        max_tokens=min(max_tokens, 16384),
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        progress=False,
                        log_dir=log_dir if observation_pack or action_fusion else None,
                        observation_pack=observation_pack,
                        action_fusion=action_fusion,
                    )
                    for key in usage:
                        usage[key] += sub_result.token_usage[key]
                    child_accounting = sub_result.usage_accounting or {}
                    reported_responses += child_accounting.get("reported_model_responses", 0)
                    usage_complete = usage_complete and child_accounting.get(
                        "complete_for_reported_responses", False,
                    )
                    delegated_sessions.append({
                        "session_id": sub_result.session_id,
                        "model": sub_result.model,
                        "token_usage": dict(sub_result.token_usage),
                        "exit_code": sub_result.exit_code,
                    })
                    output = sub_result.text or sub_result.stderr
                    is_error = sub_result.exit_code != 0
                elif tool_name == "ObsRecall" and pack is not None:
                    output = pack.recall(**arguments)
                    is_error = False
                else:
                    output = runtime.execute(tool_name, arguments)
                    is_error = False
            except Exception as exc:  # tool failures are observations, not session crashes
                output = f"{type(exc).__name__}: {exc}"
                is_error = True
            record = CodingToolCall(
                name=tool_name,
                tool_id=tool_id,
                input=arguments,
                output=output,
                is_error=is_error,
            )
            calls.append(record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": output,
                }
            )
            if progress is True:
                suffix = " ERR" if is_error else ""
                print(f"  tool: {tool_name}{suffix}", flush=True)
            elif callable(progress):
                progress(response, calls)
    else:
        stderr = f"session exhausted max_turns={max_turns}"

    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    for call in calls:
        path = call.input.get("file_path")
        if call.name == "Read" and isinstance(path, str):
            row = files_read.setdefault(path, {"reads": 0, "lines": 0})
            row["reads"] += 1
            row["lines"] += len(call.output.splitlines())
        elif call.name in {"Write", "Edit"} and isinstance(path, str):
            row = files_written.setdefault(path, {"lines_written": 0})
            content = call.input.get("content", call.input.get("new_string", ""))
            row["lines_written"] += str(content).count("\n") + 1

    result = OpenAICodingSessionResult(
        prompt=prompt,
        text="\n".join(text_parts),
        tool_calls=calls,
        files_read=files_read,
        files_written=files_written,
        token_usage=usage,
        duration_seconds=time.monotonic() - start,
        model=model,
        session_id=session_id,
        exit_code=exit_code,
        cost_usd=0.0,
        raw_events=raw_events,
        command=["openai-compatible", transport.endpoint if hasattr(transport, "endpoint") else "injected"],
        cwd=str(root),
        stderr=stderr,
        skill=skill_rows or None,
        name=name,
        log_dir=log_dir,
        efficiency={
            "schema_version": 1,
            "implementation": "PhiAgent SoL-Pi-inspired adaptation",
            "reference_revision": "bd005888b9b8a3fcdb511feb91fc27d3dfa8f2b1",
            "observation_pack": pack.report() if pack else {"enabled": False},
            "action_fusion": {
                "requested": action_fusion,
                "enabled": fusion_enabled,
                "validation_attempts": runtime.fused_validations,
            },
        } if observation_pack or action_fusion else None,
        usage_accounting={
            "scope": "returned_responses_including_delegated_sessions",
            "complete_for_reported_responses": usage_complete,
            "reported_model_responses": reported_responses,
            "failed_transport_attempt_usage": "unknown",
            "delegated_sessions": delegated_sessions,
        },
    )
    _write_log(result, log_dir)
    return result


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def coding_efficiency_from_environment() -> dict[str, bool]:
    return {
        "observation_pack": _environment_flag("META_HARNESS_OBSERVATION_PACK"),
        "action_fusion": _environment_flag("META_HARNESS_ACTION_FUSION"),
    }


def run_meta_harness_compatible(
    prompt: str,
    model: str = "unused",
    allowed_tools: Sequence[str] | None = None,
    tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    cwd: str | None = None,
    log_dir: str | None = None,
    name: str | None = None,
    system_prompt: str | None = None,
    skill_path: str | None = None,
    skills: Sequence[str] | None = None,
    skill_dir: str | None = None,
    timeout_seconds: float | None = None,
    disable_skills: bool = True,
    disable_mcp: bool = True,
    progress: bool | Callable[[dict[str, Any], list[CodingToolCall]], None] = True,
    effort: str | None = None,
) -> OpenAICodingSessionResult:
    """Drop-in ``claude_wrapper.run`` adapter used by the official release."""

    del model, skill_dir, disable_skills, disable_mcp
    base_url = os.environ.get("META_HARNESS_OPENAI_BASE_URL", "")
    wire_model = os.environ.get("META_HARNESS_OPENAI_MODEL", "")
    if not base_url or not wire_model:
        raise RuntimeError(
            "META_HARNESS_OPENAI_BASE_URL and META_HARNESS_OPENAI_MODEL are required"
        )
    efficiency = coding_efficiency_from_environment()
    if efficiency["observation_pack"] and "ObsRecall" in (disallowed_tools or ()):
        raise ValueError("observation_pack requires ObsRecall permission")
    effective_tools = list(tools or allowed_tools or ("Read", "Glob", "Grep", "Write", "Edit", "Bash"))
    for denied in disallowed_tools or ():
        effective_tools = [value for value in effective_tools if value != denied]
    skill_paths = list(skills or ())
    if skill_path:
        skill_paths.append(skill_path)
    transport = OpenAIChatTransport(
        base_url=base_url,
        api_key=os.environ.get("META_HARNESS_OPENAI_API_KEY", "EMPTY"),
        timeout_seconds=float(os.environ.get("META_HARNESS_REQUEST_TIMEOUT_SECONDS", "600")),
    )
    return run_openai_coding_agent(
        prompt=prompt,
        model=wire_model,
        transport=transport,
        allowed_tools=effective_tools,
        cwd=cwd or os.getcwd(),
        system_prompt=system_prompt,
        skills=skill_paths,
        timeout_seconds=timeout_seconds or 2400,
        max_turns=int(os.environ.get("META_HARNESS_MAX_PROPOSER_TURNS", "80")),
        max_tokens=int(os.environ.get("META_HARNESS_MAX_OUTPUT_TOKENS", "32768")),
        temperature=float(os.environ.get("META_HARNESS_TEMPERATURE", "1.0")),
        reasoning_effort=effort or os.environ.get("META_HARNESS_REASONING_EFFORT", "xhigh"),
        log_dir=log_dir,
        name=name,
        progress=progress,
        **efficiency,
    )


__all__ = [
    "CodingToolCall",
    "CodingToolRuntime",
    "OpenAIChatTransport",
    "OpenAICodingSessionResult",
    "coding_efficiency_from_environment",
    "run_meta_harness_compatible",
    "run_openai_coding_agent",
]
