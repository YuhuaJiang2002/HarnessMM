"""Exact deployment-token accounting for recorded OpenAI-compatible actor calls.

The module has no model-framework imports. Callers provide the exact tokenizer
used by the actor service; API usage remains authoritative for end-to-end input
and output totals, while tokenizer offset mappings attribute transmitted text
to non-overlapping content categories. Chat-template framing and any content
that cannot be attributed remain explicit ``unknown_tokens``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence


ALFWORLD_SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED Embodied Environment."
)
SKILL_PROMPT_PREFIX = (
    "\n\n## Skill Knowledge\n"
    "Below is a skill document with learned strategies. "
    "Use these guidelines to inform your decisions:\n\n"
)
SKILL_PROMPT_SUFFIX = "\n"
BREAKDOWN_FIELDS = (
    "base_system_tokens",
    "task_tokens",
    "active_harness_tokens",
    "active_skill_tokens",
    "wiki_or_memory_tokens",
    "tool_schema_tokens",
    "trajectory_history_tokens",
    "other_tokens",
    "unknown_tokens",
)


class OffsetTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, Any]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


class JsonOffsetTokenizer:
    """Small adapter around the exact Hugging Face ``tokenizer.json`` file."""

    def __init__(self, root: Path) -> None:
        tokenizer_path = root.expanduser().resolve() / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"actor tokenizer.json is absent: {tokenizer_path}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "exact actor decomposition requires the existing 'tokenizers' package"
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        ).ids

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, Any]:
        encoding = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        result: dict[str, Any] = {"input_ids": encoding.ids}
        if return_offsets_mapping:
            result["offset_mapping"] = encoding.offsets
        return result


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def token_count(tokenizer: OffsetTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_skill_prompt(skill_content: str) -> str:
    content = str(skill_content or "").strip()
    if not content:
        return ""
    return SKILL_PROMPT_PREFIX + content + SKILL_PROMPT_SUFFIX


def usage_fields(response: Mapping[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage") or {}
    if not isinstance(usage, Mapping):
        raise TypeError("actor response usage must be an object")
    actor_input = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    actor_output = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    actor_total = int(usage.get("total_tokens") or 0) or actor_input + actor_output
    if min(actor_input, actor_output, actor_total) < 0:
        raise ValueError("actor token usage cannot be negative")
    if actor_total != actor_input + actor_output:
        raise RuntimeError("actor total token usage is not input plus output")
    return actor_input, actor_output, actor_total


def assistant_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("actor response contains no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("actor response choice contains no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("actor response content is not text")
    return content


def tokenizer_file_audit(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    )
    records = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        records.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise FileNotFoundError(f"actor tokenizer files are absent: {root}")
    payload = {
        "schema_version": "phiagent-actor-tokenizer-audit/v1",
        "tokenizer_root": str(root),
        "files": records,
    }
    return {**payload, "audit_sha256": canonical_hash(payload)}


@dataclass(frozen=True)
class RecordedActorCall:
    index: int
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    request_path: Path
    response_path: Path
    elapsed_seconds: float


class ActorCallRecorder(AbstractContextManager["ActorCallRecorder"]):
    """Wrap one backend's private HTTP function without changing its payload."""

    def __init__(self, backend: Any, root: Path) -> None:
        original = getattr(backend, "_post_chat_completion", None)
        if not callable(original):
            raise AttributeError("actor backend has no _post_chat_completion hook")
        self.backend = backend
        self.root = root.expanduser().resolve()
        self.original: Callable[..., Mapping[str, Any]] = original
        self._lock = threading.Lock()
        self._episode_dir: Path | None = None
        self._index = 0
        self._installed = False

    def __enter__(self) -> "ActorCallRecorder":
        if self._installed:
            raise RuntimeError("actor recorder is already installed")

        def wrapped(
            payload: MutableMapping[str, Any],
            timeout: float | None,
            config: Any,
        ) -> Mapping[str, Any]:
            with self._lock:
                episode_dir = self._episode_dir
                if episode_dir is None:
                    raise RuntimeError("actor call occurred outside an episode scope")
                self._index += 1
                index = self._index
                call_dir = episode_dir / f"call-{index:04d}"
                call_dir.mkdir(parents=True, exist_ok=False)
                write_json(call_dir / "request.json", payload)
            started = time.monotonic()
            try:
                response = self.original(payload, timeout, config)
            except BaseException as exc:
                write_json(
                    call_dir / "error.json",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                raise
            elapsed = time.monotonic() - started
            write_json(call_dir / "response.json", response)
            write_json(
                call_dir / "audit.json",
                {
                    "index": index,
                    "elapsed_seconds": elapsed,
                    "request_sha256": hashlib.sha256(
                        (call_dir / "request.json").read_bytes()
                    ).hexdigest(),
                    "response_sha256": hashlib.sha256(
                        (call_dir / "response.json").read_bytes()
                    ).hexdigest(),
                },
            )
            return response

        setattr(self.backend, "_post_chat_completion", wrapped)
        self._installed = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        setattr(self.backend, "_post_chat_completion", self.original)
        self._installed = False
        self._episode_dir = None

    def start_episode(self, task_id: str, output_dir: Path) -> None:
        if not self._installed:
            raise RuntimeError("actor recorder is not installed")
        if self._episode_dir is not None:
            raise RuntimeError("previous actor episode was not finalized")
        safe_id = str(task_id).replace("/", "_").replace("\\", "_")
        self._episode_dir = output_dir.expanduser().resolve() / safe_id
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        self._index = 0

    def finish_episode(self) -> list[RecordedActorCall]:
        episode_dir = self._episode_dir
        if episode_dir is None:
            raise RuntimeError("no actor episode is active")
        calls = load_recorded_calls(episode_dir)
        self._episode_dir = None
        self._index = 0
        return calls


def load_recorded_calls(root: Path) -> list[RecordedActorCall]:
    calls = []
    for call_dir in sorted(root.glob("call-*")):
        request_path = call_dir / "request.json"
        response_path = call_dir / "response.json"
        audit_path = call_dir / "audit.json"
        if not request_path.is_file() or not response_path.is_file():
            raise RuntimeError(f"incomplete actor call recording: {call_dir}")
        audit = (
            json.loads(audit_path.read_text(encoding="utf-8"))
            if audit_path.is_file()
            else {}
        )
        calls.append(
            RecordedActorCall(
                index=int(call_dir.name.split("-")[-1]),
                request=json.loads(request_path.read_text(encoding="utf-8")),
                response=json.loads(response_path.read_text(encoding="utf-8")),
                request_path=request_path,
                response_path=response_path,
                elapsed_seconds=float(audit.get("elapsed_seconds") or 0.0),
            )
        )
    return calls


def _offsets(tokenizer: OffsetTokenizer, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw = encoded.get("offset_mapping")
    if raw is None:
        raise RuntimeError("actor tokenizer does not expose offset mappings")
    if raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
        raw = raw[0]
    return [(int(value[0]), int(value[1])) for value in raw]


def _mark(
    labels: list[str],
    text: str,
    needle: str,
    category: str,
    *,
    start: int = 0,
    overwrite: bool = False,
) -> int | None:
    if not needle:
        return None
    index = text.find(needle, start)
    if index < 0:
        return None
    end = index + len(needle)
    for position in range(index, end):
        if overwrite or labels[position] == "other_tokens":
            labels[position] = category
    return index


def _message_categories(
    *,
    tokenizer: OffsetTokenizer,
    role: str,
    content: str,
    skill_content: str,
    task_text: str | None,
    prior_steps: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, Any]]:
    counts = {
        "base_system_tokens": 0,
        "task_tokens": 0,
        "active_skill_tokens": 0,
        "other_harness_tokens": 0,
        "trajectory_history_tokens": 0,
        "other_tokens": 0,
    }
    if not content:
        return counts, {"task_occurrences": 0, "history_matches": 0}
    if role == "system":
        counts["base_system_tokens"] = token_count(tokenizer, content)
        return counts, {"task_occurrences": 0, "history_matches": 0}
    if role in {"assistant", "tool"}:
        counts["trajectory_history_tokens"] = token_count(tokenizer, content)
        return counts, {"task_occurrences": 0, "history_matches": 1}

    labels = ["other_tokens"] * len(content)
    skill_prompt = build_skill_prompt(skill_content)
    if skill_prompt and content.startswith(skill_prompt):
        prefix_end = len(SKILL_PROMPT_PREFIX)
        skill_end = prefix_end + len(str(skill_content or "").strip())
        for position in range(0, prefix_end):
            labels[position] = "other_harness_tokens"
        for position in range(prefix_end, skill_end):
            labels[position] = "active_skill_tokens"
        for position in range(skill_end, len(skill_prompt)):
            labels[position] = "other_harness_tokens"

    task_occurrences = 0
    if task_text:
        cursor = 0
        while True:
            matched = _mark(labels, content, task_text, "task_tokens", start=cursor)
            if matched is None:
                break
            task_occurrences += 1
            cursor = matched + len(task_text)

    history_matches = 0
    history_values = []
    for step in prior_steps:
        for key in ("model_response", "env_feedback"):
            value = str(step.get(key) or "")
            if value:
                history_values.append(value)
    for value in sorted(set(history_values), key=len, reverse=True):
        cursor = 0
        while True:
            matched = _mark(
                labels,
                content,
                value,
                "trajectory_history_tokens",
                start=cursor,
            )
            if matched is None:
                break
            history_matches += 1
            cursor = matched + len(value)

    for start, end in _offsets(tokenizer, content):
        if end <= start:
            continue
        category = labels[min(start, len(labels) - 1)]
        counts[category] += 1
    return counts, {
        "task_occurrences": task_occurrences,
        "history_matches": history_matches,
    }


def extract_alfworld_task_text(
    source_row: Mapping[str, Any],
    *,
    data_root: Path | None = None,
) -> str | None:
    for key in ("task_description", "instruction", "task"):
        value = str(source_row.get(key) or "").strip()
        if value:
            return value
    gamefile = str(source_row.get("gamefile") or "").strip()
    if not gamefile:
        return None
    path = Path(gamefile).expanduser()
    if not path.is_absolute() and data_root is not None:
        path = data_root.expanduser().resolve() / path
    trajectory = path.parent / "traj_data.json"
    if not trajectory.is_file():
        return None
    value = json.loads(trajectory.read_text(encoding="utf-8"))
    annotations = ((value.get("turk_annotations") or {}).get("anns") or [])
    for annotation in annotations:
        text = str((annotation or {}).get("task_desc") or "").strip()
        if text:
            return text
    return None


def count_skill_artifact(
    skill_root: Path,
    tokenizer: OffsetTokenizer,
    *,
    active_skill_content: str,
) -> dict[str, Any]:
    root = skill_root.expanduser().resolve()
    skill_files = sorted(root.glob("*/SKILL.md"))
    nonempty = [
        (path, path.read_text(encoding="utf-8").strip())
        for path in skill_files
        if path.read_text(encoding="utf-8").strip()
    ]
    total_skill_bank_tokens = sum(token_count(tokenizer, text) for _, text in nonempty)
    artifact_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    return {
        "total_skill_bank_tokens": total_skill_bank_tokens,
        "static_total_skill_tokens": total_skill_bank_tokens,
        "num_total_skills": len(nonempty),
        "num_active_skills": len(nonempty) if active_skill_content.strip() else 0,
        "artifact_bytes": artifact_bytes,
        "artifact_bytes_are_deployment_cost": False,
    }


def account_actor_episode(
    *,
    tokenizer: OffsetTokenizer,
    calls: Sequence[RecordedActorCall],
    conversation: Sequence[Mapping[str, Any]],
    skill_content: str,
    task_text: str | None,
    static_artifact: Mapping[str, Any],
    actor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not calls:
        raise ValueError("actor episode contains no recorded calls")
    sums = {
        "actor_input_tokens": 0,
        "actor_output_tokens": 0,
        "actor_total_tokens": 0,
        "base_system_tokens": 0,
        "task_tokens": 0,
        "active_skill_tokens": 0,
        "other_harness_tokens": 0,
        "wiki_or_memory_tokens": 0,
        "tool_schema_tokens": 0,
        "trajectory_history_tokens": 0,
        "other_tokens": 0,
        "unknown_tokens": 0,
    }
    call_rows = []
    total_task_occurrences = 0
    total_history_matches = 0
    for call_offset, call in enumerate(calls):
        actor_input, actor_output, actor_total = usage_fields(call.response)
        sums["actor_input_tokens"] += actor_input
        sums["actor_output_tokens"] += actor_output
        sums["actor_total_tokens"] += actor_total
        messages = call.request.get("messages") or []
        if not isinstance(messages, list):
            raise TypeError("actor request messages must be a list")
        attributed = 0
        per_call = {
            "base_system_tokens": 0,
            "task_tokens": 0,
            "active_skill_tokens": 0,
            "other_harness_tokens": 0,
            "trajectory_history_tokens": 0,
            "other_tokens": 0,
            "wiki_or_memory_tokens": 0,
            "tool_schema_tokens": 0,
        }
        prior_steps = conversation[:call_offset]
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("actor message must be an object")
            role = str(message.get("role") or "")
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, sort_keys=True)
            categories, audit = _message_categories(
                tokenizer=tokenizer,
                role=role,
                content=content,
                skill_content=skill_content,
                task_text=task_text,
                prior_steps=prior_steps,
            )
            for key, value in categories.items():
                per_call[key] += int(value)
            total_task_occurrences += int(audit["task_occurrences"])
            total_history_matches += int(audit["history_matches"])
        tools = call.request.get("tools")
        if tools:
            per_call["tool_schema_tokens"] = token_count(
                tokenizer,
                json.dumps(
                    tools,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        attributed = sum(per_call.values())
        if attributed > actor_input:
            raise RuntimeError(
                "decomposed actor content exceeds authoritative API input tokens"
            )
        unknown = actor_input - attributed
        for key, value in per_call.items():
            sums[key] += int(value)
        sums["unknown_tokens"] += unknown
        call_rows.append(
            {
                "call_index": call.index,
                "actor_input_tokens": actor_input,
                "actor_output_tokens": actor_output,
                "actor_total_tokens": actor_total,
                **per_call,
                "unknown_tokens": unknown,
                "elapsed_seconds": call.elapsed_seconds,
                "request_path": str(call.request_path),
                "response_path": str(call.response_path),
            }
        )

    sums["active_harness_tokens"] = (
        sums["base_system_tokens"]
        + sums["active_skill_tokens"]
        + sums["other_harness_tokens"]
        + sums["wiki_or_memory_tokens"]
        + sums["tool_schema_tokens"]
    )
    static_skill_prompt = build_skill_prompt(skill_content)
    static_harness_tokens = token_count(tokenizer, ALFWORLD_SYSTEM_PROMPT)
    if static_skill_prompt:
        static_harness_tokens += token_count(tokenizer, static_skill_prompt)
    effective_payload = {
        **dict(actor_identity),
        "system_prompt": ALFWORLD_SYSTEM_PROMPT,
        "active_skill_content": skill_content,
        "tool_schema": None,
    }
    status = (
        "exact_api_totals_exact_tokenizer_content_partial_semantic_decomposition"
        if task_text is not None
        else "exact_api_totals_exact_tokenizer_content_task_unknown"
    )
    return {
        **sums,
        **dict(static_artifact),
        "static_harness_tokens": static_harness_tokens,
        "effective_agent_hash": canonical_hash(effective_payload),
        "actor_call_count": len(calls),
        "task_text_available": task_text is not None,
        "task_text_occurrences": total_task_occurrences,
        "trajectory_history_matches": total_history_matches,
        "decomposition_status": status,
        "call_rows": call_rows,
    }
