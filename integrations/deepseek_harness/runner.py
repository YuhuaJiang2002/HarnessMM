#!/usr/bin/env python3
"""Run the pinned OFFICIAL DeepSeek Harness SDK, without replacing its agent loop.

Runtime execution is restricted to ``ssh a800-4`` (hostname harnessmm-api-worker).
Importing this module and testing its pure validators/reducers needs no SDK.
Install the official source's ``python/sdk`` (distribution deepseek-harness-sdk),
NOT the unrelated PyPI distribution named deepseek-harness.

JSON job interface (all paths absolute; run_dir must not already exist)::

    {
      "schema_version": 1,
      "run_dir": "/opt/phiagent/work/deepseek-harness/20260929-v1/runs/unique",
      "purpose": "synthetic",
      "arm": "baseline",
      "session_id": "unique-session",
      "seed": 20260929,
      "prompt": "Reply with SDK_OK.",
      "model": "deepseek-flash",
      "max_tokens": 32768,
      "api_key_file": "/path/to/private-config/credentials/deepseek-harness-20260929",
      "runtime": {
        "source_root": "/absolute/pinned/official/source",
        "dsh_bin": "/absolute/pinned/official/source/apps/cli/lib/bin.js",
        "node_bin": "/absolute/isolated/node/bin/node",
        "manifest": "/absolute/runtime-provenance.json"
      },
      "tools": [],
      "bridge": null,
      "chisel": null,
      "source_revision": "parent-checkout-revision"
    }

``prompt`` also accepts official text/image content parts. An image is
{"type":"image","data":"BASE64","mimeType":"image/jpeg"} (not a data URL).
``purpose`` is synthetic or benchmark; this runner does not choose datasets,
train CHISEL, score results, or authorize new benchmark runs.
Optional ``cwd`` names an existing empty directory within an authorized root, shared
by both arms if desired. It changes the SDK agent's workspace only; subprocess
cwd, home, persistence, and artifacts remain isolated per run.

Tool jobs supply ``tools`` as a fixed list of
{"name": "...", "description": "...", "parameters": {JSON Schema}} and
``bridge`` as {"url":"http://127.0.0.1:PORT",
"secret_file":"/absolute/owner-only/file","notifications":false,
"generation_timing":false}. The bridge owns native execution, deduplication,
and cancellation; generated commands are never executed by these plugins.
Optional ``native_observation_encoding`` is ``canonical_json`` (the legacy
default) or ``native_text``. The latter forwards original native string
observations literally, without adding JSON-string quotes or escaped newlines.
It requires a tool bridge and must be frozen identically for paired arms.
Notifications receive {"session_id":...}; clock start receives session_id and
attempt_id, and clock end additionally receives elapsed_seconds. All endpoints
are POSTs with the per-run bridge Bearer credential.
Treatment jobs use ``arm``="chisel" and
``chisel``={"path":"/absolute/frozen.txt","sha256":"64 lower-case hex digits",
"artifact_kind":"training-derived","provenance":{...}}.
Synthetic plugin verification may use artifact_kind="synthetic-fixture";
that is not trained CHISEL and is rejected for benchmark jobs.
Optional ``benchmark_context`` is an empty-or-text string containing only shared
public native context (for example initial native time and the scenario's
additional system prompt). Pass identical bytes to paired arms. It is frozen as
``frozen-benchmark-context.txt``, hashed in provenance, and added literally through
``benchmark:context`` (order 10200, interpolate=false), independently of CHISEL's
``chisel:frozen`` section (order 10300). Neither section overrides the official
system prompt. Empty/absent context adds no section. ``run(prompt)`` still receives
the original native user input unchanged; subsequent native time/user updates
belong in notifications, not a rewritten initial prompt.

Runtime manifest requires ``upstream_commit``, ``source_tree_sha256`` (mapping
relative file paths to hashes, including pnpm-lock.yaml and SDK Python files),
and ``dsh_sha256``. Optional ``build`` records acquisition/build commands.
Source and runtime hashes are checked before any credential is read.
CLI: python runner.py --job /absolute/job.json
Acceptance: python runner.py --job /absolute/synthetic-template.json --synthetic-acceptance
The acceptance command requires purpose=synthetic and creates two child runs:
baseline text and a synthetic-fixture treatment with a loopback echo tool. Its
random tool receipt is never supplied in the prompt. This is operability only.
Artifact export: ``python runner.py --collect-from /ABS/RUN --collect-to /ABS/NEW``.
Export uses the finalized hash allowlist, not recursive directory copying.
API-key paths, auth.secret/bridge.secret, managed credential files, symlinks,
caches, and files containing the in-memory run credentials are excluded.
Authorized run roots are the existing campaign's ``runs/`` and
``/path/to/data/phiagent/work/deepseek-harness/``. A run must be a strict descendant,
not the root itself. Traversal and symlink components are rejected. New run
directories and any new parents are private (0700); existing parents are not
chmodded. The A800-only execution guard applies equally to both roots.

Persistent Python ABI (import this only in the isolated SDK environment)::

    with OfficialHarnessSession(template) as session:
        first = session.run("First native user turn")
        second = session.run("Next native user turn")
    episode = session.result

The template may omit ``prompt``. Every run uses its fixed session_id and the
same official SDK process/history. ``run`` returns the existing run_job result
shape with turn-local ``usage``; ``session.result`` and root result.json/usage.json
cover the entire episode. ``run_job(job)`` remains a one-turn convenience API.
Inputs and turn results are immutable under turns/000001/, turns/000002/, etc.
Each records byte spans into the append-only raw stream/notification logs.
Root summaries are snapshots until ``closed`` is true; only close finalizes the
artifact allowlist. SDK durable event totals are cross-checks, never added to
the all-attempt stream total. A concluded native tool turn is not episode end.
``stop_requested`` from native notifications rejects the next official SDK
step without another provider call; it is a successful environment stop,
reported alongside the unchanged SDK finish reason. Image notifications retain
official inline image parts. Unsupported content fails explicitly.

JSONL IPC ABI v1, for ARE processes with an incompatible Pydantic version::

    CUDA_VISIBLE_DEVICES='' runtime-sdk/bin/python STAGING/runner.py --session-worker
    {"id":"init","op":"init","job":{...template...}}
    {"id":"turn-1","op":"run","prompt":"First native user turn"}
    {"id":"turn-2","op":"run","prompt":"Next native user turn"}
    {"id":"done","op":"shutdown"}

Each stdout line is exactly one JSON response with protocol_version=1, id, op,
and ok. init returns session_id/run_dir/runtime_pid; run returns ``result``.
Wait for init and each run reply before the next run. IDs are unique per worker;
overlapping runs and duplicate IDs are rejected, never replayed. ``cancel`` and
``shutdown`` are terminal: they close the owned SDK, interrupt an active turn,
reap the child, finalize accounting, and return the episode result. An active
run reply may precede their acknowledgement. EOF also closes/reaps the SDK.
No SDK import occurs in the parent or until worker init. Stdout is protocol-only;
diagnostics go to stderr. This protocol imposes no cumulative token/turn limit.
Feature acceptance is ``tests/deepseek_harness_remote_session_acceptance.py``:
its versioned result, raw two-turn receipts, zero-generation environment-stop
turn, child-shutdown proof, and separate historical/current cost ledgers are
under ``/path/to/data/phiagent/work/deepseek-harness/session-extension-20260929-v1/``.
The backward-compatible text run is
``/opt/phiagent/work/deepseek-harness/20260929-v1/runs/session-extension-legacy-20260929-v1``.
``acceptance.json`` and ``SOURCE_MANIFEST.json`` identify the exact results and
tested source hashes; this synthetic fixture is not benchmark/trained-CHISEL evidence.
Shared-context regression receipts are under
``/path/to/data/phiagent/work/deepseek-harness/session-context-20260929-v1/``;
they additionally verify identical literal context in both arms while preserving
the initial user prompt and CHISEL's separate frozen section.

Each session creates a fresh isolated cwd/home, frozen sources/config,
raw SDK notifications, every LLM stream, usage disclosure, and run result.
Both arms disable host instruction discovery, filesystem/office skills,
workspace dependency provisioning, host shell/filesystem/Web tools, subagent
tools, workflows, and plugin-management tools. The core SDK, agent loop, LLM,
tool registry, and image attachment service remain enabled; only the explicit
bridge catalog is allowed in each agent's scope.
No cumulative token or turn ceiling is imposed. max_tokens is the official
per-request output capacity, not an experiment-wide token budget.
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import secrets as random_secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen


UPSTREAM_COMMIT = "4878cdabd87d4041bdaff61d04c966883b9fd07a"
UPSTREAM_TAG = "dsh-v0.2.0-rc.1"
PROVIDER = "deepseek-official"
BASE_URL = "https://api.deepseek.com/anthropic"
REMOTE_HOSTNAME = "harnessmm-api-worker"
CAMPAIGN_ROOT = Path("/opt/phiagent/work/deepseek-harness/20260929-v1")
DATA0_ROOT = Path("/path/to/data/phiagent/work/deepseek-harness")
MAX_SAFE_INTEGER = 2**53 - 1
PLUGIN_FILES = ("bridge_plugin.mjs", "chisel_plugin.mjs", "cot_jev.mjs", "jev_v2.mjs", "sar_v2.mjs", "sar_admission.mjs")
ADAPTER_FILES = ("runner.py", "session_worker.py", *PLUGIN_FILES)
CLOSED_NATIVE_DISABLED_PLUGINS = (
    "agent-instructions",
    "skill-filesystem",
    "skill-office",
    "workspace-dependencies",
    "tool-skill",
    "tool-web",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-subagent-control",
    "tool-subagent-list-agents",
    "tool-workflow",
    "tool-plugin-manager",
    "tool-bash",
    "tool-pwsh",
    "tool-fs",
    "tool-fs-search",
)
PRIVATE_ARTIFACT_NAMES = {
    "auth.secret", "bridge.secret", ".credentials.yaml", ".credentials.yml", ".env",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_SAFE_INTEGER


def normalize_usage(usage: Any) -> dict[str, int] | None:
    """Equivalent of upstream normalizeUsage; cache buckets are disjoint.

    An absent optional bucket is not silently replaced by zero. An exact
    total permits missing cache buckets, but never a contradictory total.
    """
    if not isinstance(usage, dict):
        return None
    if not all(_count(usage.get(key)) for key in ("inputTokens", "outputTokens")):
        return None
    optional = ("cacheReadTokens", "cacheWriteTokens", "reasoningTokens", "totalTokens")
    if any(key in usage and not _count(usage[key]) for key in optional):
        return None
    if usage.get("reasoningTokens", 0) > usage["outputTokens"]:
        return None
    known_prompt = usage["inputTokens"] + sum(
        usage[key] for key in ("cacheReadTokens", "cacheWriteTokens") if key in usage)
    if known_prompt > MAX_SAFE_INTEGER:
        return None
    both_cache = all(key in usage for key in ("cacheReadTokens", "cacheWriteTokens"))
    if "totalTokens" in usage:
        total = usage["totalTokens"]
        exact_prompt = total - usage["outputTokens"]
        if exact_prompt < known_prompt or (both_cache and exact_prompt != known_prompt):
            return None
    else:
        if not both_cache:
            return None
        total = known_prompt + usage["outputTokens"]
        if total > MAX_SAFE_INTEGER:
            return None
    result = {key: usage[key] for key in ("inputTokens", "outputTokens", *optional)
              if key in usage}
    result["totalTokens"] = total
    return result


def aggregate_usage(samples: list[dict[str, int]]) -> dict[str, int] | None:
    if not samples:
        return None
    result: dict[str, int] = {}
    for key in ("inputTokens", "outputTokens", "totalTokens",
                "cacheReadTokens", "cacheWriteTokens", "reasoningTokens"):
        if all(key in sample for sample in samples):
            total = sum(sample[key] for sample in samples)
            if total > MAX_SAFE_INTEGER:
                return None
            result[key] = total
    return result


def _last_stream_usage(stream: Any) -> Any:
    if not isinstance(stream, list):
        return None
    for record in reversed(stream):
        if not isinstance(record, dict) or record.get("type") != "chunk":
            continue
        chunk = record.get("chunk")
        if isinstance(chunk, dict) and chunk.get("type") == "usage":
            return chunk.get("usage")
    return None


def derive_turn_token_usage(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Port of pinned packages/llm/token-meter/src/turn-usage.ts state machine.

    Fold *complete*, single-turn durable events. Retried/failed attempts count
    once; assistant/message and assistant/attempt are not additive snapshots.
    Unknown, inconsistent, or incomplete accounting returns None, not zero.
    """
    state: dict[str, Any] = {"kind": "idle"}
    turn = None
    saw_end = False
    attempts: list[dict[str, int]] = []
    routes: list[dict[str, str] | None] = []

    def same(data: dict[str, Any]) -> bool:
        return state.get("turn") == data.get("turn") and state.get("step") == data.get("step")

    def close_open(route: dict[str, str] | None = None) -> bool:
        if state["kind"] != "open":
            return False
        sample = normalize_usage(state.get("sample"))
        if sample is None:
            return False
        attempts.append(sample)
        routes.append(route)
        return True

    for event in events:
        kind, data = event.get("type"), event.get("data", {})
        if not isinstance(data, dict):
            return None
        if kind == "turn/start":
            if turn is not None or state["kind"] != "idle":
                return None
            turn = data.get("turn")
            continue
        if turn is None:
            return None
        if kind == "turn/end":
            if data.get("turn") != turn or state["kind"] != "idle" or saw_end:
                return None
            saw_end = True
            continue
        if saw_end:
            return None
        if kind == "step/start":
            if data.get("turn") != turn or state["kind"] != "idle":
                return None
            state = {"kind": "open", "turn": turn, "step": data.get("step")}
        elif kind == "llm/retry-started":
            if (data.get("turn") != turn or state["kind"] != "settled"
                    or state.get("by") != "retry" or not same(data)):
                return None
            state = {"kind": "open", "turn": turn, "step": data.get("step")}
        elif kind in ("assistant/attempt", "assistant/message"):
            if data.get("turn") != turn or state["kind"] != "open" or not same(data):
                return None
            sample = (data.get("usage") if kind == "assistant/message" else None)
            if sample is None:
                sample = _last_stream_usage(data.get("stream"))
            if sample is not None:
                state["sample"] = sample
            route = None
            source = data.get("message", {}).get("source", {})
            if (kind == "assistant/message" and isinstance(source, dict)
                    and isinstance(source.get("provider"), str) and source["provider"]
                    and isinstance(source.get("model"), str) and source["model"]):
                route = {"provider": source["provider"], "model": source["model"]}
            if not close_open(route):
                return None
            state = {"kind": "finishClosed" if kind == "assistant/attempt" else "settled",
                     "turn": turn, "step": data.get("step"), "by": "message"}
        elif kind == "llm/retry":
            if data.get("turn") != turn or state["kind"] == "idle" or not same(data):
                return None
            if state["kind"] == "settled" or (state["kind"] == "open" and not close_open()):
                return None
            state = {"kind": "settled", "turn": turn, "step": data.get("step"), "by": "retry"}
        elif kind == "step/end":
            if data.get("turn") != turn or state["kind"] == "idle" or not same(data):
                return None
            if state["kind"] == "open" and not close_open():
                return None
            state = {"kind": "idle"}
    result = aggregate_usage(attempts) if saw_end and state["kind"] == "idle" else None
    if result is None:
        return None
    result["uncachedInputTokens"] = result.pop("inputTokens")
    if all(route is not None for route in routes):
        result["routes"] = list({canonical_json(route): route for route in routes}.values())
    return result


def summarize_streams(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Account from exactly one source: innermost LLM stream receipts.

    SDK final messages are retained for cross-checking, never added to these
    totals. All invocations, including compaction/auxiliary calls, are included.
    """
    attempts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        attempt_id = record.get("attempt_id")
        if not isinstance(attempt_id, str):
            continue
        kind = record.get("kind")
        if kind == "llm/start":
            if attempt_id in attempts:
                errors.append(f"duplicate-start:{attempt_id}")
            attempts[attempt_id] = {"attempt_id": attempt_id, "start": record,
                                    "usage_raw": None, "ended": False, "request_ids": []}
        elif attempt_id not in attempts:
            errors.append(f"missing-start:{attempt_id}")
        elif kind == "llm/chunk":
            chunk = record.get("chunk", {})
            if isinstance(chunk, dict) and chunk.get("type") == "usage":
                attempts[attempt_id]["usage_raw"] = chunk.get("usage")
            if isinstance(chunk, dict) and chunk.get("type") == "finish":
                attempt = attempts[attempt_id]
                attempt["finish"] = chunk
                failure = chunk.get("reason", {}).get("failure", {})
                if isinstance(failure, dict) and failure.get("requestId"):
                    attempt["request_ids"].append(failure["requestId"])
                replay = chunk.get("replayState", {})
                if isinstance(replay, dict) and "response" in replay:
                    attempt["raw_response_metadata"] = replay["response"]
        elif kind == "llm/end":
            if attempts[attempt_id]["ended"]:
                errors.append(f"duplicate-end:{attempt_id}")
            attempts[attempt_id]["ended"] = True
            attempts[attempt_id]["end"] = record
    samples = []
    requests = {
        record.get("decision_id"): record.get("native_attempt_id")
        for record in records if record.get("kind") == "jev/request"
    }
    fast_fallbacks: set[str] = set()
    for record in records:
        if record.get("kind") != "jev/slow-fallback":
            continue
        identity = record.get("decision_id")
        if (not isinstance(identity, str) or identity in fast_fallbacks
                or requests.get(identity) not in attempts):
            errors.append("unbound-or-duplicate-fast-fallback")
        else:
            fast_fallbacks.add(identity)
    deterministic = set()
    compiled = {r.get("decision_id") for r in records
                if r.get("kind") == "jev/compiled-native-action"}
    for record in records:
        if record.get("kind") != "sar/deterministic-commit":
            continue
        ident = record.get("native_attempt_id")
        if ident not in attempts or ident in deterministic or record.get("decision_id") not in compiled:
            errors.append("unbound-or-duplicate-deterministic-commit")
        else:
            deterministic.add(ident)
    for attempt in attempts.values():
        sample = normalize_usage(attempt["usage_raw"])
        attempt["usage"] = sample
        attempt["physical_provider_request"] = attempt["attempt_id"] not in deterministic
        if attempt["attempt_id"] in deterministic and (sample is None or sample.get("totalTokens") != 0):
            errors.append("deterministic-commit-has-nonzero-or-unknown-usage")
        if sample is None or not attempt["ended"]:
            errors.append(f"incomplete-attempt:{attempt['attempt_id']}")
        elif sample is not None:
            samples.append(sample)
    total = aggregate_usage(samples) if not errors and attempts else None
    return {"source": "all-llm-stream-receipts-not-sdk-message-sums",
            "complete": total is not None, "usage": total,
            "known_reported_usage": aggregate_usage(samples),
            "attempt_count": len(attempts), "attempts": list(attempts.values()),
            "model_call_attempt_count": len(attempts) - len(deterministic) + len(fast_fallbacks),
            "deterministic_callback_count": len(deterministic),
            "unknown_reasons": errors or ([] if attempts else ["no-llm-attempts"]),
            "money": None, "money_status": "not-derived-without-versioned-provider-prices"}


def durable_turn_disclosures(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: dict[str, list[dict[str, Any]]] = {}
    disclosed = []
    seen: dict[tuple[str, int], str] = {}
    for receipt in receipts:
        session_id = receipt.get("session_id")
        event = receipt.get("event")
        if not isinstance(session_id, str) or not isinstance(event, dict):
            continue
        sequence = event.get("seq")
        if _count(sequence):
            key = (session_id, sequence)
            fingerprint = canonical_json(event)
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError("contradictory durable events at the same session sequence")
                continue
            seen[key] = fingerprint
        if event.get("type") == "turn/start":
            if session_id in turns:
                disclosed.append({"session_id": session_id, "usage": None,
                                  "reason": "new-turn-before-prior-turn-ended"})
            turns[session_id] = []
        if session_id in turns:
            turns[session_id].append(event)
            if event.get("type") == "turn/end":
                group = turns.pop(session_id)
                disclosed.append({"session_id": session_id,
                                  "turn": group[0].get("data", {}).get("turn"),
                                  "usage": derive_turn_token_usage(group)})
    for session_id in turns:
        disclosed.append({"session_id": session_id, "usage": None,
                          "reason": "incomplete-turn"})
    return disclosed


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value)


def _no_symlink_path(path: Path, label: str) -> Path:
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain traversal")
    absolute = path.absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError(f"{label} must not contain symlink components")
    return absolute


def authorized_run_path(value: Any) -> Path:
    path = _no_symlink_path(_absolute_path(value, "run_dir"), "run_dir")
    roots = (CAMPAIGN_ROOT / "runs", DATA0_ROOT)
    if not any(path != root and path.is_relative_to(root) for root in roots):
        raise ValueError(f"run_dir must be below {roots[0]} or {roots[1]}")
    return path


def _create_private_directory(path: Path) -> None:
    """Create a fresh directory with no symlink-following, including its parents."""
    path = _no_symlink_path(path, "directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for index, part in enumerate(path.parts[1:], 1):
            last = index == len(path.parts) - 1
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                if last:
                    raise
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("new run/export directory must be owned by this uid and mode 0700")
    finally:
        os.close(descriptor)


def validate_prompt(prompt: Any) -> str | list[dict[str, Any]]:
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be text or nonempty official content parts")
    for part in prompt:
        if not isinstance(part, dict):
            raise ValueError("prompt parts must be objects")
        if part.get("type") == "text":
            if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                raise ValueError("invalid text part")
        elif part.get("type") == "image":
            if (set(part) != {"type", "data", "mimeType"}
                    or part["mimeType"] not in ("image/jpeg", "image/png",
                                                "image/gif", "image/webp")):
                raise ValueError("invalid image part")
            try:
                if not base64.b64decode(part["data"], validate=True):
                    raise ValueError("empty image")
            except (ValueError, TypeError) as exc:
                raise ValueError("image data must be nonempty base64, not a URL") from exc
        else:
            raise ValueError("unsupported prompt content type")
    return prompt


def validate_observation_arm(arm: str, chisel: Any, policy: Any) -> None:
    if arm not in ("baseline", "chisel", "context_only", "chisel_context"):
        raise ValueError("Unknown native industrial arm")
    if arm in ("baseline", "context_only") and chisel is not None:
        raise ValueError("The native parent/context-only arm must not receive CHISEL instructions")
    if arm in ("chisel", "chisel_context") and not chisel:
        raise ValueError("CHISEL requires its frozen training-derived artifact")
    if arm not in ("context_only", "chisel_context"):
        if policy is not None:
            raise ValueError("Observation policy requires an explicitly identified context arm")
        return
    keys = {
        "version", "minimum_characters", "full_exposures", "keep_recent",
        "preview_characters", "page_characters",
    }
    checkpoint = None
    if isinstance(policy, dict) and policy.get("version") == 2:
        keys.add("action_evidence_checkpoint")
        checkpoint = policy.get("action_evidence_checkpoint")
        checkpoint_keys = {
            "version", "retain_tokens", "minimum_saved_tokens", "minimum_reasoning_characters",
        }
        if (not isinstance(checkpoint, dict) or set(checkpoint) != checkpoint_keys
                or any(type(value) is not int or not 1 <= value <= MAX_SAFE_INTEGER
                       for value in checkpoint.values())
                or checkpoint["version"] != 1 or checkpoint["retain_tokens"] < 1024
                or checkpoint["minimum_saved_tokens"] < 256
                or checkpoint["minimum_reasoning_characters"] < 1024):
            raise ValueError("Invalid frozen action-evidence checkpoint policy")
    if (not isinstance(policy, dict) or set(policy) != keys
            or any(type(value) is not int or not 1 <= value <= MAX_SAFE_INTEGER
                   for key, value in policy.items() if key != "action_evidence_checkpoint")
            or policy["version"] not in (1, 2) or policy["minimum_characters"] < 1024
            or policy["full_exposures"] < 2 or policy["page_characters"] > 8000
            or 2 * policy["preview_characters"] >= policy["minimum_characters"]):
        raise ValueError("Invalid frozen native observation policy")


def validate_job(job: Any, *, require_prompt: bool = True) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    allowed = {"schema_version", "run_dir", "purpose", "arm", "session_id", "seed",
               "prompt", "model", "max_tokens", "api_key_file", "runtime", "tools",
               "bridge", "chisel", "source_revision", "cwd", "benchmark_context",
               "observation_policy", "cot_jev", "native_observation_encoding", "provider_capture", "jev_v2"}
    if set(job) - allowed:
        raise ValueError(f"unsupported job fields: {sorted(set(job) - allowed)}")
    if type(job.get("schema_version")) is not int or job["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    authorized_run_path(job.get("run_dir"))
    if job.get("cwd") is not None:
        cwd = _no_symlink_path(_absolute_path(job["cwd"], "cwd"), "cwd")
        if not any(cwd.is_relative_to(root) for root in (CAMPAIGN_ROOT, DATA0_ROOT)):
            raise ValueError("explicit cwd must remain within the campaign or authorized data0 root")
    if job.get("purpose") not in ("synthetic", "benchmark"):
        raise ValueError("purpose must be synthetic or benchmark")
    validate_observation_arm(job.get("arm"), job.get("chisel"), job.get("observation_policy"))
    if not isinstance(job.get("session_id"), str) or not re.fullmatch(
            r"[a-zA-Z0-9_.-]{1,120}", job["session_id"]):
        raise ValueError("session_id must be a nonempty safe identifier")
    if not _count(job.get("seed")):
        raise ValueError("seed must be a nonnegative safe integer")
    if not isinstance(job.get("model"), str) or not re.fullmatch(
            r"deepseek-[a-zA-Z0-9_.-]+", job["model"]):
        raise ValueError("an explicit official DeepSeek model alias is required")
    if not _count(job.get("max_tokens")) or job["max_tokens"] == 0:
        raise ValueError("max_tokens must be a positive per-request capacity")
    _absolute_path(job.get("api_key_file"), "api_key_file")
    capture = job.get("provider_capture")
    if capture is not None:
        if not isinstance(capture, dict) or set(capture) != {"base_url", "fast_url"}:
            raise ValueError("provider_capture requires base_url and fast_url")
        slow, fast = (urlsplit(capture[key]) for key in ("base_url", "fast_url"))
        for endpoint in (slow, fast):
            if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
                    or endpoint.port is None or endpoint.username or endpoint.password
                    or endpoint.query or endpoint.fragment):
                raise ValueError("Provider capture must use an authenticated loopback transport")
        if (slow.netloc != fast.netloc or not re.fullmatch(
                r"/[A-Za-z0-9][A-Za-z0-9_-]{0,159}/anthropic", slow.path)
                or fast.path != slow.path.removesuffix("/anthropic") + "/chat/completions"
                or job["model"] != "deepseek-flash"):
            raise ValueError("Provider capture routes or frozen model differ")
    runtime = job.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
            "source_root", "dsh_bin", "node_bin", "manifest"}:
        raise ValueError("runtime requires source_root, dsh_bin, node_bin, manifest only")
    for key, value in runtime.items():
        _absolute_path(value, f"runtime.{key}")
    if require_prompt or "prompt" in job:
        validate_prompt(job.get("prompt"))
    if "benchmark_context" in job and not isinstance(job["benchmark_context"], str):
        raise ValueError("benchmark_context must be empty or literal text")
    if "native_observation_encoding" in job and job["native_observation_encoding"] not in (
            "canonical_json", "native_text"):
        raise ValueError("Unsupported native observation encoding")
    tools = job.get("tools")
    if not isinstance(tools, list):
        raise ValueError("tools must be an explicit list (empty for VQA/text)")
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"name", "description", "parameters"}:
            raise ValueError("tool requires name, description, parameters only")
        name = tool["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", name):
            raise ValueError("invalid tool name")
        if name in names:
            raise ValueError("duplicate tool name")
        names.add(name)
        if not isinstance(tool["description"], str) or not isinstance(tool["parameters"], dict):
            raise ValueError("tool description/schema must be string/object")
        if tool["parameters"].get("type") != "object":
            raise ValueError("tool parameters must be an object JSON schema")
    bridge = job.get("bridge")
    if bridge is None:
        if "native_observation_encoding" in job:
            raise ValueError("Native observation encoding requires the native tool bridge")
        if tools:
            raise ValueError("tools require a loopback bridge")
    else:
        if not isinstance(bridge, dict) or set(bridge) != {
                "url", "secret_file", "notifications", "generation_timing"}:
            raise ValueError("bridge requires url, secret_file, notifications, generation_timing")
        parsed = urlsplit(bridge["url"])
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.port is None or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("bridge must be a literal 127.0.0.1 HTTP origin")
        _absolute_path(bridge["secret_file"], "bridge.secret_file")
        if any(type(bridge[key]) is not bool for key in ("notifications", "generation_timing")):
            raise ValueError("bridge optional modes must be booleans")
    treatment = job.get("chisel")
    if job.get("observation_policy") is not None and (
            names != {"terminal"} or bridge is None
            or bridge["notifications"] or bridge["generation_timing"]):
        raise ValueError("The observation-policy pilot requires the native terminal-only bridge")
    if job.get("cot_jev") is not None:
        policy = job["cot_jev"]
        native_apps = isinstance(policy, dict) and policy.get("version") in (3, 4)
        if (job["arm"] != "chisel" or bridge is None or not names
                or (not native_apps and (names != {"terminal"}
                    or bridge["notifications"] or bridge["generation_timing"]))
                or (native_apps and job.get("native_observation_encoding") != "native_text")
                or job.get("observation_policy") is not None):
            raise ValueError("CoT/Jev requires the unchanged terminal-only CHISEL arm")
        if (not isinstance(policy, dict) or set(policy) != {
                "version", "enabled", "maximum_candidates", "maximum_fast_steps",
                "maximum_public_bytes"}
                or type(policy["version"]) is not int or policy["version"] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
                or type(policy["enabled"]) is not bool
                or any(type(policy[key]) is not int or not 1 <= policy[key] <= 16
                       for key in ("maximum_candidates", "maximum_fast_steps"))
                or type(policy["maximum_public_bytes"]) is not int
                or not 1024 <= policy["maximum_public_bytes"] <= 65536):
            raise ValueError("Invalid frozen CoT/Jev policy")
    if job.get("jev_v2") is not None:
        if not job.get("cot_jev", {}).get("enabled") or job["jev_v2"].get("arm") not in ("A3", "A4", "A5", "A6"):
            raise ValueError("Jev-v2 requires an active, independently frozen experimental arm")
        if job["jev_v2"].get("sar", {}).get("mode") == "train_screen" and job.get("purpose") != "synthetic":
            # Protected TRAIN ownership is checked by the ARE-to-SDK adapter;
            # an exploratory prior must never qualify a held-out deployment.
            if job["jev_v2"].get("sar", {}).get("execution_phase") != "train":
                raise ValueError("Exploratory SAR priors are TRAIN-only")
    if job["arm"] in ("baseline", "context_only"):
        if treatment is not None:
            raise ValueError("baseline must not receive a CHISEL artifact")
    else:
        if not isinstance(treatment, dict) or set(treatment) != {
                "path", "sha256", "artifact_kind", "provenance"}:
            raise ValueError("chisel requires path, sha256, artifact_kind, provenance")
        _absolute_path(treatment["path"], "chisel.path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(treatment["sha256"])):
            raise ValueError("invalid frozen CHISEL sha256")
        kinds = {"training-derived"}
        if job["purpose"] == "synthetic":
            kinds.add("synthetic-fixture")
        if treatment["artifact_kind"] not in kinds or not isinstance(treatment["provenance"], dict):
            raise ValueError("benchmark CHISEL must be an identified training-derived artifact")
        if job["purpose"] == "benchmark" and (
                not treatment["provenance"] or treatment["provenance"].get("synthetic")
                or treatment["provenance"].get("not_trained_chisel")):
            raise ValueError("synthetic or absent provenance is not training-derived evidence")
    return job


def read_owner_secret(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("credential must be an owner-only regular file owned by this uid")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as handle:
            value = handle.read(16385).strip()
        if not value or len(value) > 16384 or "\n" in value or "\r" in value:
            raise ValueError("credential must be one nonempty line")
        return value
    finally:
        os.close(fd)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED_CREDENTIAL]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item, secrets) for key, item in value.items()}
    return value


def _benchmark_context_config(job: dict[str, Any], run_dir: Path) -> dict[str, str] | None:
    if "benchmark_context" not in job:
        return None
    return {
        "path": str(run_dir / "frozen-benchmark-context.txt"),
        "sha256": hashlib.sha256(job["benchmark_context"].encode("utf-8")).hexdigest(),
    }


def build_patch(job: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """JSON is YAML-compatible; each patched config is a *whole-row* replacement."""
    plugins = [{
        "id": "phiagent-benchmark-bridge",
        "name": str(run_dir / "source" / "bridge_plugin.mjs"),
        "config": {"jobFile": str(run_dir / "job.json"),
                   "receiptsFile": str(run_dir / "llm-streams.jsonl"),
                   "bridgeTransport": ("native-http" if {tool["name"] for tool in job["tools"]}
                                       == {"terminal"} else "fetch"),
                   "messageModule": str(Path(job["runtime"]["source_root"])
                                        / "packages/llm/llm/lib/types/message.js")},
    }]
    context = _benchmark_context_config(job, run_dir)
    if context is not None:
        plugins[0]["config"]["benchmarkContext"] = context
    if job["arm"] in ("chisel", "chisel_context"):
        plugins.append({
            "id": "phiagent-chisel-frozen",
            "name": str(run_dir / "source" / "chisel_plugin.mjs"),
            "config": {"path": str(run_dir / "frozen-chisel.txt"),
                       "sha256": job["chisel"]["sha256"]},
        })
    if job.get("observation_policy") is not None:
        plugins.append({
            "id": "phiagent-native-observation-policy",
            "name": str(run_dir / "source" / "observation_policy.mjs"),
            "config": {
                "jobFile": str(run_dir / "job.json"),
                "receiptsFile": str(run_dir / "context-events.jsonl"),
                "archiveDirectory": str(run_dir / "observation-archive"),
                "messageModule": plugins[0]["config"]["messageModule"],
            },
        })
        if job["observation_policy"].get("action_evidence_checkpoint") is not None:
            plugins[-1]["config"]["regionModule"] = str(
                Path(job["runtime"]["source_root"])
                / "packages/compaction/compaction-basic/lib/types/region.js"
            )
    return [
        {"id": "session-log-deepseek", "config": {"enabled": False}},
        {"id": "session-telemetry-otel", "disabled": True},
        {"id": "plugin-package-inventory-deepseek", "config": {"enabled": False}},
        *({"id": plugin_id, "disabled": True}
          for plugin_id in CLOSED_NATIVE_DISABLED_PLUGINS),
        {"id": "session-persistence-jsonl",
         "config": {"root": str(run_dir / "sessions"), "compression": "none"}},
        {"id": "permission", "config": {
            "presets": {"benchmark": {"sandbox": "workspace-write", "approval": "never"}},
            "defaultPreset": "benchmark"}},
        {"insert": plugins},
    ]


def verify_runtime(runtime: dict[str, str]) -> dict[str, Any]:
    root = Path(runtime["source_root"]).resolve()
    manifest = json.loads(Path(runtime["manifest"]).read_text(encoding="utf-8"))
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("runtime manifest does not identify the pinned official commit")
    files = manifest.get("source_tree_sha256")
    required = {"pnpm-lock.yaml", "python/sdk/pyproject.toml",
                "python/sdk/src/deepseek_harness/api.py",
                "python/sdk/src/deepseek_harness/client.py"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("runtime manifest is missing lockfile/SDK source hashes")
    runtime_files = manifest.get("runtime_artifact_sha256", {})
    if not isinstance(runtime_files, dict):
        raise ValueError("runtime artifact hashes must be an object")
    for relative, digest in {**files, **runtime_files}.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError(f"official source hash mismatch: {relative}")
    if sha256_file(Path(runtime["dsh_bin"])) != manifest.get("dsh_sha256"):
        raise ValueError("dsh executable differs from the frozen runtime manifest")
    if "node_sha256" in manifest and sha256_file(Path(runtime["node_bin"])) != manifest["node_sha256"]:
        raise ValueError("isolated Node executable differs from the runtime manifest")
    return manifest


def verify_installed_sdk(source_root: Path) -> None:
    spec = importlib.util.find_spec("deepseek_harness")
    if spec is None or spec.origin is None:
        raise RuntimeError("install python/sdk from the pinned official source first")
    installed = Path(spec.origin).parent
    source = source_root / "python/sdk/src/deepseek_harness"
    for path in source.glob("*.py"):
        if sha256_file(installed / path.name) != sha256_file(path):
            raise RuntimeError(f"installed SDK file differs from official pin: {path.name}")


def _command_receipt(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=30, check=False)
        return {"argv": argv, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": type(exc).__name__}


def _load_jsonl(path: Path, start: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open("rb") as handle:
        handle.seek(start)
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def artifact_exclusion_reason(
    path: Path, run_dir: Path, secret_files: tuple[Path, ...] = (),
) -> str | None:
    """Pure policy plus filesystem checks, reusable by native artifact collectors."""
    root = run_dir.resolve()
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "outside-run"
    if ".." in relative.parts or path.is_symlink():
        return "symlink-or-traversal"
    if not path.resolve().is_relative_to(root):
        return "outside-run"
    if any(path.resolve() == secret.resolve() for secret in secret_files):
        return "credential-path"
    if (path.name.casefold() in PRIVATE_ARTIFACT_NAMES
            or path.name.casefold().startswith(".env.")):
        return "credential-file"
    if any(part in {"cache", "__pycache__"} for part in relative.parts):
        return "cache"
    if any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)):
        return "symlink-or-traversal"
    return None if path.is_file() else "not-regular-file"


def _job_secret_files(job: dict[str, Any]) -> tuple[Path, ...]:
    values = [job.get("api_key_file")]
    bridge = job.get("bridge")
    if isinstance(bridge, dict):
        values.append(bridge.get("secret_file"))
    return tuple(Path(value) for value in values if isinstance(value, str) and value)


def _credential_free_digest(path: Path, secrets: tuple[str, ...]) -> str | None:
    needles = tuple(value.encode("utf-8") for value in secrets if value)
    overlap = max((len(value) for value in needles), default=1) - 1
    tail = b""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sample = tail + chunk
            if any(value in sample for value in needles):
                return None
            digest.update(chunk)
            tail = sample[-overlap:] if overlap else b""
    return digest.hexdigest()


def write_artifact_manifest(
    run_dir: Path, job: dict[str, Any], secrets: tuple[str, ...] = (),
) -> dict[str, str]:
    root = run_dir.resolve()
    hashes: dict[str, str] = {}
    excluded: dict[str, str] = {}
    credential_files = _job_secret_files(job)
    for path in sorted(root.rglob("*")):
        if path.name in {"ARTIFACT_SHA256.json", "ARTIFACT_EXCLUSIONS.json"}:
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        reason = artifact_exclusion_reason(path, root, credential_files)
        if reason is None:
            digest = _credential_free_digest(path, secrets)
            if digest is not None:
                hashes[relative] = digest
                continue
            reason = "credential-content"
        excluded[relative] = reason
    _write_json(root / "ARTIFACT_EXCLUSIONS.json", excluded)
    hashes["ARTIFACT_EXCLUSIONS.json"] = sha256_file(root / "ARTIFACT_EXCLUSIONS.json")
    _write_json(root / "ARTIFACT_SHA256.json", hashes)
    return hashes


def collect_run_artifacts(run_dir: Path, destination: Path) -> dict[str, Any]:
    """Copy only finalized, credential-safe, hash-verified artifacts.

    Collection is allowed locally; it never imports the SDK or invokes a model.
    For remote collection, stage this filtered export on A800 before transfer.
    Never recursively copy a live native bridge or SDK run directory instead.
    """
    root = _no_symlink_path(run_dir, "artifact source")
    destination = _no_symlink_path(destination, "artifact destination")
    if destination == root or destination.is_relative_to(root):
        raise ValueError("artifact destination must be outside the source run")
    manifest = json.loads((root / "ARTIFACT_SHA256.json").read_text(encoding="utf-8"))
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    secret_files = _job_secret_files(job)
    selected: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for relative, digest in manifest.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid artifact manifest entry")
        if relative == "ARTIFACT_SHA256.json":
            continue
        path = root / relative
        reason = artifact_exclusion_reason(path, root, secret_files)
        if reason is not None:
            excluded[relative] = reason
            continue
        if sha256_file(path) != digest:
            raise ValueError(f"artifact changed after finalization: {relative}")
        selected[relative] = digest
    _create_private_directory(destination)
    for relative, digest in selected.items():
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"artifact changed during collection: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(data)
        target.chmod(0o600)
    _write_json(destination / "ARTIFACT_SHA256.json", selected)
    _write_json(destination / "COLLECTION.json", {
        "source_run": str(root), "copied_files": len(selected), "excluded_files": excluded,
        "method": "finalized-hash-allowlist-not-recursive-copy",
    })
    return {"destination": str(destination), "copied_files": len(selected),
            "excluded_files": excluded}


class OfficialHarnessSession:
    """One pinned SDK process and native scenario, with independently metered turns."""

    def __init__(self, template: dict[str, Any]) -> None:
        self.job = copy.deepcopy(validate_job(template, require_prompt=False))
        self.run_dir = authorized_run_path(self.job["run_dir"])
        self.result: dict[str, Any] = {}
        self._harness: Any = None
        self._process: Any = None
        self._secrets: tuple[str, ...] = ()
        self._source_files: tuple[str, ...] = ()
        self._events: list[dict[str, Any]] = []
        self._turns: list[dict[str, Any]] = []
        self._reserved = self._started = self._closing = self._closed = False
        self._stop_requested = False
        self._start_error: dict[str, str] | None = None
        self._close_error: dict[str, str] | None = None
        self._started_at = time.time()
        self._cancel_requested = threading.Event()
        self._start_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._close_lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self.job["session_id"]

    @property
    def runtime_pid(self) -> int | None:
        return getattr(self._process, "pid", None)

    def redact(self, value: Any) -> Any:
        return _redact(value, self._secrets)

    def _error(self, exc: Exception) -> dict[str, str]:
        return {"type": type(exc).__name__, "message": self.redact(str(exc))}

    def __enter__(self) -> OfficialHarnessSession:
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def start(self) -> None:
        with self._start_lock:
            if self._closed or self._closing:
                raise RuntimeError("the SDK session is closed")
            if self._started:
                return
            if socket.gethostname() != REMOTE_HOSTNAME:
                raise RuntimeError("SDK execution is authorized only on a800-4 (harnessmm-api-worker)")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("set CUDA_VISIBLE_DEVICES='' explicitly; this is a CPU/API-only run")
            os.umask(0o077)
            authorized_run_path(str(self.run_dir))
            _create_private_directory(self.run_dir)
            self._reserved = True
            try:
                for child in ("source", "cwd", "home", "sessions", "cache", "turns"):
                    (self.run_dir / child).mkdir(mode=0o700)
                _write_json(self.run_dir / "job.json", self.job)
                _write_json(self.run_dir / "commands.json", {
                    "argv": sys.argv, "cwd": str(Path.cwd()),
                    "launch": "CUDA_VISIBLE_DEVICES='' python runner.py --job JOB "
                              "(or --session-worker)",
                    "session_api": "OfficialHarnessSession-v1",
                    "cumulative_token_limit": None, "turn_limit": None,
                    "seed": self.job["seed"]})
                source = Path(__file__).resolve().parent
                self._source_files = tuple(
                    name for name in ADAPTER_FILES
                    if name != "session_worker.py" or (source / name).is_file())
                if self.job.get("observation_policy") is not None:
                    self._source_files += ("observation_policy.mjs",)
                    if self.job["observation_policy"].get("action_evidence_checkpoint") is not None:
                        self._source_files += ("action_evidence_checkpoint.mjs",)
                for filename in self._source_files:
                    shutil.copy2(source / filename,
                                 self.run_dir / "source" / filename)
                self._prepare_runtime()
                self._harness.start()
                self._process = getattr(self._harness.client, "_proc", None)
                self._started = True
                self._save()
            except Exception as exc:
                self._start_error = self._error(exc)
                self.close()
                raise RuntimeError(self._start_error["message"]) from None

    def _prepare_runtime(self) -> None:
        job, run_dir = self.job, self.run_dir
        local_source = Path(__file__).resolve().parent
        workspace = Path(job.get("cwd") or run_dir / "cwd")
        node = Path(job["runtime"]["node_bin"])
        node_version = _command_receipt([str(node), "--version"])
        disk = shutil.disk_usage(run_dir)
        context_config = _benchmark_context_config(job, run_dir)
        _write_json(run_dir / "provenance.json", {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "executable": sys.executable,
            "packages": sorted({dist.metadata["Name"]: dist.version
                                for dist in importlib.metadata.distributions()}.items()),
            "node": node_version, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_tag": UPSTREAM_TAG, "source_revision": job.get("source_revision"),
            "adapter_hashes": {filename: sha256_file(run_dir / "source" / filename)
                               for filename in self._source_files},
            "git": _command_receipt(["git", "rev-parse", "HEAD"], local_source),
            "gpu_selection": {"CUDA_VISIBLE_DEVICES": "", "cpu_api_only": True},
            "gpu_state": _command_receipt(["nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader"]),
            "provider": PROVIDER, "base_url": BASE_URL,
            "provider_capture": job.get("provider_capture"),
            "requested_model_alias": job["model"], "immutable_model_revision": None,
            "profile": "sdk", "purpose": job["purpose"], "arm": job["arm"],
            "seed": job["seed"], "cumulative_token_limit": None,
            "seed_role": "job/partition provenance; the official SDK has no provider seed argument",
            "agent_cwd": str(workspace), "runtime_cwd": str(run_dir / "cwd"),
            "shared_empty_cwd": bool(job.get("cwd")),
            "session_api": "OfficialHarnessSession-v1",
            "authorized_run_root": str(DATA0_ROOT if run_dir.is_relative_to(DATA0_ROOT)
                                       else CAMPAIGN_ROOT / "runs"),
            "disk_at_start": {"total_bytes": disk.total, "free_bytes": disk.free},
            "benchmark_context": context_config,
        })
        _no_symlink_path(workspace, "SDK cwd")
        if not workspace.is_dir() or any(workspace.iterdir()):
            raise ValueError("SDK cwd must be an existing empty non-symlink directory")
        version = node_version.get("stdout", "").strip().lstrip("v")
        version_parts = tuple(int(part) for part in version.split(".")) if re.fullmatch(
            r"\d+\.\d+\.\d+", version) else ()
        if not (len(version_parts) == 3 and
                ((version_parts[0] == 22 and version_parts >= (22, 19, 0))
                 or version_parts[0] >= 24)):
            raise RuntimeError("isolated Node must satisfy ^22.19.0 || >=24.0.0")
        manifest = verify_runtime(job["runtime"])
        _write_json(run_dir / "runtime-manifest.json", manifest)
        if job["arm"] in ("chisel", "chisel_context"):
            original = Path(job["chisel"]["path"])
            if sha256_file(original) != job["chisel"]["sha256"]:
                raise ValueError("frozen CHISEL artifact does not match declared SHA256")
            shutil.copyfile(original, run_dir / "frozen-chisel.txt")
        if context_config is not None:
            context_path = Path(context_config["path"])
            context_path.write_text(job["benchmark_context"], encoding="utf-8")
            context_path.chmod(0o400)
        patch_path = run_dir / "benchmark.patch.yml"
        _write_json(patch_path, build_patch(job, run_dir))
        verify_installed_sdk(Path(job["runtime"]["source_root"]))
        api_key = read_owner_secret(Path(job["api_key_file"]))
        self._secrets = (api_key,)
        bridge_secret = (read_owner_secret(Path(job["bridge"]["secret_file"]))
                         if job.get("bridge") else "")
        self._secrets = (api_key, bridge_secret)
        if bridge_secret and random_secrets.compare_digest(api_key, bridge_secret):
            raise ValueError("bridge authentication must use a distinct per-run credential")
        # The SDK imports only here: source editing and pure unit tests are keyless.
        from deepseek_harness import DeepSeekHarness
        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": str(node.parent) + os.pathsep + os.environ.get("PATH", ""),
            "HOME": str(run_dir / "home"), "TMPDIR": str(run_dir / "cache"),
            "DSH_HOME": str(run_dir / "home"), "DSH_PRIMARY_RUNTIME": "",
            "OTEL_SDK_DISABLED": "true",
        }
        if job.get("cot_jev") is not None:
            env["DEEPSEEK_API_KEY"] = api_key
            if job["cot_jev"]["version"] >= 8:
                priors = Path(os.environ["PHIAGENT_SAR_TRAIN_PRIORS"])
                prior_sha = os.environ["PHIAGENT_SAR_TRAIN_PRIORS_SHA256"]
                if sha256_file(priors) != prior_sha:
                    raise ValueError("Frozen TRAIN admission priors changed")
                env["PHIAGENT_SAR_TRAIN_PRIORS"] = str(priors)
                env["PHIAGENT_SAR_TRAIN_PRIORS_SHA256"] = prior_sha
            if job.get("provider_capture") is not None:
                env["PHIAGENT_DEEPSEEK_FAST_URL"] = job["provider_capture"]["fast_url"]

        self._harness = DeepSeekHarness(
            provider=PROVIDER, model=job["model"], profile="sdk",
            base_url=(job.get("provider_capture") or {}).get("base_url", BASE_URL),
            max_tokens=job["max_tokens"], api_key=api_key,
            cwd=str(workspace), runtime_cwd=str(run_dir / "cwd"),
            dsh_home=str(run_dir / "home"), dsh_bin=job["runtime"]["dsh_bin"],
            patches=(str(patch_path),), env=env,
            initialize_timeout_seconds=120.0, request_timeout_seconds=None,
            shutdown_timeout_seconds=15.0,
        )

    def _notification_received(self, notification: Any) -> None:
        raw = dataclasses.asdict(notification) if dataclasses.is_dataclass(notification) else notification
        raw = self.redact(raw)
        with (self.run_dir / "notifications.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(raw) + "\n")
            handle.flush()
        if isinstance(raw, dict) and raw.get("method") == "session.event":
            payload = raw.get("payload", {})
            if isinstance(payload.get("event"), dict):
                self._events.append({"session_id": payload.get("sessionId"), "event": payload["event"]})

    def _position(self, filename: str) -> int:
        path = self.run_dir / filename
        return path.stat().st_size if path.exists() else 0

    @staticmethod
    def _accounting(streams: list[dict[str, Any]], events: list[dict[str, Any]],
                    *, empty_is_known: bool = False) -> dict[str, Any]:
        accounting = summarize_streams(streams)
        if empty_is_known and not accounting["attempt_count"] and not any(
                item.get("kind", "").startswith("llm/") for item in streams):
            zero = dict.fromkeys(("inputTokens", "outputTokens", "totalTokens",
                                  "cacheReadTokens", "cacheWriteTokens"), 0)
            accounting.update(complete=True, usage=zero, known_reported_usage=zero,
                              unknown_reasons=[], no_generation=True)
        accounting["durable_turn_disclosures"] = durable_turn_disclosures(events)
        return accounting

    def run(self, prompt: str | list[dict[str, Any]]) -> dict[str, Any]:
        prompt = copy.deepcopy(validate_prompt(prompt))
        self.start()
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("only one native turn may be active per SDK session")
        try:
            if self._closed or self._closing or self._cancel_requested.is_set():
                raise RuntimeError("the SDK session is closing or cancelled")
            if self._stop_requested:
                raise RuntimeError("the native environment has stopped; close this episode")
            index = len(self._turns) + 1
            turn_dir = self.run_dir / "turns" / f"{index:06d}"
            turn_dir.mkdir(mode=0o700)
            stream_start = self._position("llm-streams.jsonl")
            notification_start = self._position("notifications.jsonl")
            event_start = len(self._events)
            started = time.time()
            turn: dict[str, Any] = {
                "status": "failed", "session_id": self.session_id, "turn_index": index,
                "started_at_unix": started, "finish_reason": None, "final_response": None,
                "run_dir": str(self.run_dir), "turn_dir": str(turn_dir), "usage_scope": "turn",
            }
            _write_json(turn_dir / "input.json", self.redact({
                "session_id": self.session_id, "turn_index": index, "prompt": prompt}))
            try:
                sdk_result = self._harness.run(
                    prompt, session_id=self.session_id, on_notification=self._notification_received)
                turn.update({
                    "status": "completed" if sdk_result.finish_reason == "completed" else "failed",
                    "finish_reason": sdk_result.finish_reason,
                    "final_response": self.redact(sdk_result.final_response),
                })
                for event in reversed(sdk_result.events):
                    if event.get("type") == "turn/end":
                        turn["finish_details"] = self.redact(event.get("data", {}).get("reason"))
                        break
                raw_result = self.redact(dataclasses.asdict(sdk_result))
                _write_json(turn_dir / "sdk-result.json", raw_result)
                _write_json(self.run_dir / "sdk-result.json", raw_result)
            except Exception as exc:
                turn["error"] = self._error(exc)
            finally:
                events = self._events[event_start:]
                _write_json(turn_dir / "events.json", events)
                try:
                    streams = _load_jsonl(self.run_dir / "llm-streams.jsonl", stream_start)
                    stopped = any(item.get("kind") == "bridge/environment-stop" for item in streams)
                    self._stop_requested = self._stop_requested or stopped
                    if stopped and "error" not in turn and turn["finish_reason"] in (
                            "blocked", "completed"):
                        turn["status"] = "completed"
                    accounting = self._accounting(
                        streams, events, empty_is_known=stopped and any(
                            item["event"].get("type") == "turn/end" for item in events))
                except (ValueError, OSError) as exc:
                    accounting = {"complete": False, "usage": None,
                                  "unknown_reasons": [f"receipt-read-failed:{type(exc).__name__}"]}
                turn.update(
                    elapsed_seconds=time.time() - started,
                    usage_complete=accounting["complete"], usage=accounting["usage"],
                    stop_requested=self._stop_requested,
                    cancelled=self._cancel_requested.is_set(),
                    stream_span_bytes=[stream_start, self._position("llm-streams.jsonl")],
                    notification_span_bytes=[
                        notification_start, self._position("notifications.jsonl")])
                _write_json(turn_dir / "usage.json", self.redact(accounting))
                _write_json(turn_dir / "result.json", turn)
                self._turns.append(turn)
                self._save()
            return copy.deepcopy(turn)
        finally:
            self._run_lock.release()

    def _save(self) -> None:
        if not self._reserved:
            return
        _write_json(self.run_dir / "events.json", self._events)
        try:
            accounting = self._accounting(
                _load_jsonl(self.run_dir / "llm-streams.jsonl"), self._events,
                empty_is_known=self._started and (
                    not self._turns or all(turn.get("usage_complete") for turn in self._turns)))
        except (ValueError, OSError) as exc:
            accounting = {"complete": False, "usage": None,
                          "unknown_reasons": [f"receipt-read-failed:{type(exc).__name__}"]}
        _write_json(self.run_dir / "usage.json", self.redact(accounting))
        failed = self._start_error or self._close_error or any(
            turn["status"] != "completed" for turn in self._turns)
        status = "failed" if failed else "completed"
        if self._cancel_requested.is_set() and not self._close_error:
            status = "cancelled"
        last = copy.deepcopy(self._turns[-1]) if self._turns else {
            "finish_reason": None, "final_response": None}
        self.result = {
            **last, "status": status, "session_id": self.session_id,
            "started_at_unix": self._started_at, "elapsed_seconds": time.time() - self._started_at,
            "usage_scope": "episode", "usage_complete": accounting["complete"],
            "usage": accounting["usage"], "run_dir": str(self.run_dir),
            "turn_count": len(self._turns), "turns": copy.deepcopy(self._turns),
            "stop_requested": self._stop_requested, "closed": self._closed,
            "cancelled": self._cancel_requested.is_set(), "runtime_pid": self.runtime_pid,
        }
        if self._start_error:
            self.result["error"] = self._start_error
        if self._close_error:
            self.result["close_error"] = self._close_error
        if self._closed:
            self.result["runtime_closed"] = (
                self._process.poll() is not None if self._process is not None
                else self._close_error is None)
        _write_json(self.run_dir / "result.json", self.result)

    def cancel(self) -> dict[str, Any]:
        """Terminal cancellation; never reuse an interrupted native mutation."""
        if not self._closed:
            self._cancel_requested.set()
        return self.close()

    def close(self) -> dict[str, Any]:
        with self._close_lock:
            if self._closed:
                return copy.deepcopy(self.result)
            self._closing = True
            if self._run_lock.locked():
                self._cancel_requested.set()
            if self._harness is not None:
                self._process = self._process or getattr(self._harness.client, "_proc", None)
                try:
                    self._harness.close()
                except Exception as exc:
                    self._close_error = self._error(exc)
                if self._process is not None and self._process.poll() is None:
                    try:
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            self._process.kill()
                            self._process.wait()
                    except Exception as exc:
                        self._close_error = self._error(exc)
            with self._run_lock:
                self._closed = True
                self._save()
                if self._reserved:
                    _write_json(self.run_dir / "shutdown.json", {
                        "runtime_pid": self.runtime_pid,
                        "returncode": self._process.poll() if self._process is not None else None,
                        "runtime_closed": self.result.get("runtime_closed"),
                        "cancelled": self._cancel_requested.is_set(),
                        "close_error": self._close_error,
                    })
                    write_artifact_manifest(self.run_dir, self.job, self._secrets)
            return copy.deepcopy(self.result)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible one-turn entry point, sharing the persistent lifecycle."""
    validate_job(job)
    session = OfficialHarnessSession(job)
    try:
        session.start()
        session.run(job["prompt"])
    except Exception as exc:
        if not session._reserved:
            raise
        if session._start_error is None:
            session._start_error = session._error(exc)
    finally:
        session.close()
    return session.result


def run_synthetic_acceptance(template: dict[str, Any]) -> dict[str, Any]:
    """Two real provider calls through the SDK; this never loads a benchmark."""
    validate_job(template)
    if template["purpose"] != "synthetic":
        raise ValueError("synthetic acceptance does not accept benchmark jobs")
    if socket.gethostname() != REMOTE_HOSTNAME or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("synthetic acceptance requires A800-4 and CUDA_VISIBLE_DEVICES=''")
    os.umask(0o077)
    suite = authorized_run_path(template["run_dir"])
    _create_private_directory(suite)
    _write_json(suite / "template.json", template)
    _write_json(suite / "commands.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
    secret_file = suite / "bridge.secret"
    token = random_secrets.token_urlsafe(32)
    secret_file.write_text(token, encoding="utf-8")
    secret_file.chmod(0o600)
    challenge = "BRIDGE_RECEIPT_" + random_secrets.token_hex(12)
    bridge_requests: list[dict[str, Any]] = []
    delivered_notice = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def respond(self, status: int, value: Any) -> None:
            encoded = canonical_json(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self.respond(200 if self.path == "/health" else 404, {"ready": True})

        def do_POST(self) -> None:
            nonlocal delivered_notice
            if not random_secrets.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.respond(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                self.respond(400, {"error": "invalid request size"})
                return
            payload = json.loads(self.rfile.read(length))
            bridge_requests.append({"path": self.path, "payload": payload, "at": time.time()})
            if self.path == "/tool":
                if (payload.get("name") != "synthetic_echo" or not payload.get("call_id")
                        or payload.get("arguments") != {"value": "BRIDGE_PROBE"}):
                    self.respond(400, {"error": "unexpected synthetic tool invocation"})
                    return
                self.respond(200, {"observation": {"receipt": challenge, "echo": "BRIDGE_PROBE"},
                                   "conclude_turn": False})
            elif self.path == "/notifications":
                messages = [] if delivered_notice else ["Synthetic environment is ready."]
                delivered_notice = True
                self.respond(200, {"messages": messages, "stop_requested": False})
            elif self.path in ("/generation/start", "/generation/end"):
                self.respond(200, {"ok": True})
            else:
                self.respond(404, {"error": "unknown synthetic endpoint"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bridge_url = f"http://127.0.0.1:{server.server_address[1]}"
    outcomes = []
    checks: dict[str, bool] = {}
    failure = None
    try:
        with urlopen(bridge_url + "/health", timeout=5) as response:
            assert json.load(response)["ready"] is True
        text_job = {**template, "run_dir": str(suite / "text"),
                    "session_id": template["session_id"] + "-text", "arm": "baseline",
                    "prompt": "Reply with exactly SDK_TEXT_OK. Do not call any tools.",
                    "tools": [], "bridge": None, "chisel": None}
        outcomes.append(run_job(text_job))
        fixture = suite / "synthetic-frozen-fixture.txt"
        fixture.write_text(
            "Synthetic adapter acceptance fixture, not trained CHISEL. "
            "When the synthetic_echo tool returns a receipt, give that receipt verbatim.\n",
            encoding="utf-8")
        tool_job = {**template, "run_dir": str(suite / "tool"),
                    "session_id": template["session_id"] + "-tool", "arm": "chisel",
                    "prompt": 'Call synthetic_echo exactly once with {"value":"BRIDGE_PROBE"}. '
                              "Then reply with exactly the receipt from its observation. "
                              "Do not invent the receipt or answer before calling the tool.",
                    "tools": [{"name": "synthetic_echo",
                               "description": "Echo a synthetic value and return a fresh receipt.",
                               "parameters": {"type": "object", "properties": {
                                   "value": {"type": "string", "const": "BRIDGE_PROBE"}},
                                   "required": ["value"], "additionalProperties": False}}],
                    "bridge": {"url": bridge_url, "secret_file": str(secret_file),
                               "notifications": True, "generation_timing": True},
                    "chisel": {"path": str(fixture), "sha256": sha256_file(fixture),
                               "artifact_kind": "synthetic-fixture",
                               "provenance": {"synthetic": True, "not_trained_chisel": True}}}
        outcomes.append(run_job(tool_job))
        text_records = _load_jsonl(suite / "text" / "llm-streams.jsonl")
        tool_records = _load_jsonl(suite / "tool" / "llm-streams.jsonl")
        text_llm = [item for item in text_records if item["kind"] == "llm/start"]
        tool_llm = [item for item in tool_records if item["kind"] == "llm/start"]
        starts = [item for item in bridge_requests if item["path"] == "/generation/start"]
        ends = [item for item in bridge_requests if item["path"] == "/generation/end"]
        tool_calls = [item for item in bridge_requests if item["path"] == "/tool"]
        checks = {
            "text_completed": outcomes[0]["status"] == "completed",
            "text_exact_reply": (outcomes[0]["final_response"] or "").strip() == "SDK_TEXT_OK",
            "tool_completed": outcomes[1]["status"] == "completed",
            "unprompted_native_receipt": (outcomes[1]["final_response"] or "").strip() == challenge,
            "one_native_tool_call": len(tool_calls) == 1,
            "text_usage_complete": outcomes[0]["usage_complete"],
            "tool_usage_complete": outcomes[1]["usage_complete"],
            "text_no_host_tools": bool(text_llm) and all(
                not item.get("tool_names") for item in text_llm),
            "tool_exact_catalog": bool(tool_llm) and all(
                item.get("tool_names") == ["synthetic_echo"] for item in tool_llm),
            "notification_delivered": delivered_notice and any(
                "Synthetic environment is ready." in canonical_json(item.get("request", {}))
                for item in tool_llm),
            "native_clock_paired": bool(starts) and len(starts) == len(ends)
                and {item["payload"]["attempt_id"] for item in starts}
                == {item["payload"]["attempt_id"] for item in ends}
                and all(item["payload"].get("elapsed_seconds", -1) >= 0 for item in ends),
            "frozen_section_present": any(
                "Synthetic adapter acceptance fixture, not trained CHISEL."
                in canonical_json(item.get("request", {}))
                for item in tool_records if item["kind"] == "llm/start"),
        }
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        secret_file.unlink(missing_ok=True)
        _write_json(suite / "bridge-requests.json", bridge_requests)
    result = {"accepted": bool(checks) and all(checks.values()) and failure is None,
              "checks": checks, "error": failure, "runs": outcomes,
              "suite_dir": str(suite), "scope": "synthetic-sdk-operability-not-benchmark-evidence"}
    _write_json(suite / "acceptance.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--job", type=Path)
    operation.add_argument("--collect-from", type=Path)
    operation.add_argument("--session-worker", action="store_true")
    parser.add_argument("--collect-to", type=Path)
    parser.add_argument("--synthetic-acceptance", action="store_true")
    args = parser.parse_args(argv)
    if args.session_worker:
        if args.collect_to is not None or args.synthetic_acceptance:
            parser.error("--session-worker cannot collect artifacts or run legacy acceptance")
        if __package__:
            from .session_worker import main as worker_main
        else:
            from session_worker import main as worker_main
        return worker_main()
    if args.collect_from is not None:
        if args.collect_to is None or args.synthetic_acceptance:
            parser.error("--collect-from requires --collect-to and cannot run acceptance")
        print(canonical_json(collect_run_artifacts(args.collect_from, args.collect_to)))
        return 0
    if args.collect_to is not None:
        parser.error("--collect-to requires --collect-from")
    job = json.loads(args.job.read_text(encoding="utf-8"))
    result = run_synthetic_acceptance(job) if args.synthetic_acceptance else run_job(job)
    print(canonical_json(result))
    return 0 if (result.get("accepted") if args.synthetic_acceptance
                 else result["status"] == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
