"""Official Codex 0.142.5 + an explicit native-ARE MCP benchmark adapter.

This module never imports ARE at import time. ``run_episode`` is remote-only.
The actor is the unmodified official Codex executable, not an ARE/ReAct agent.
ARE owns scenario preprocessing, app calls, event scheduling, and validation.
Both comparison arms use this implementation; only residual instructions differ.
Residual files contain only the learned instruction. The common launcher writes
it to task-local AGENTS.md; the task prompt and original Codex system stay unchanged.

Controller contract
-------------------
``--provider-url`` is the parent metering root, e.g. ``http://172.17.0.1:28160``
on the authorized A800 Docker bridge (loopback roots are also supported).
``--relay-secret-file`` is the owner-private HMAC relay secret, not an API key.
The adapter uses separate ``<trial-id>-actor`` and ``<trial-id>-judge`` URLs and
ephemeral derived tokens. ``--judge-url`` optionally selects another metering
root. Both actor and judge use DeepSeek's native
``https://api.deepseek.com/responses`` endpoint through byte-preserving,
credential-holding metering. There is no Responses-to-Chat adapter.
The native ARE judge, prompts, and decisions are unchanged.
The bridge must report real provider usage on every completed response. The HMAC
secret and evaluator's files never enter the Codex namespace. Failed/uncertain
requests are retained, never retried here, and never assigned fabricated costs.
Actor request/response bytes and tool schemas are preserved. Codex configuration
disables host-execution tools; a fail-closed check admits native MCP tools plus
Codex's harmless internal planning/resource/image tools in the empty namespace.
Both Responses and the native ``/responses/compact`` route are supported.

``--partition-manifest`` uses phiagent-20-20-60/v1: gaia2-mini, seed 20260928,
160 scenarios, exactly 32/32/96. The default is alongside ``scenarios/``.
``--freeze-manifest`` is required, including for development:

    {
      "schema": "phiagent-codex-gaia2-freeze/v1",
      "frozen_at_utc": "...",
      "dataset_revision": "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99",
      "source_parquet_sha256": "<GAIA2_MINI_SHA256 below; prepared by controller>",
      "partition_sha256": "<file SHA256>",
      "filesystem_sha256": "<tree_sha256(filesystem)>",
      "codex_sha256": "<resolved official static ELF SHA256>",
      "are_source_sha256": "<tree_sha256(are/simulation, '*.py')>",
      "source_files": {
        "scripts/run_codex_gaia2.py": "<SHA256>",
        "phiagent/harness/codex_gaia2.py": "<SHA256>",
        "phiagent/harness/benchmark_partition.py": "<SHA256>",
        "phiagent/harness/codex_runtime.py": "<SHA256>",
        "scripts/serve_codex_deepseek.py": "<SHA256>"
      },
      "bridge_source_files": {"/absolute/path/to/bridge.py": "<SHA256>"},
      "native_protocol_receipt": {
        "path": "/absolute/path/to/native-protocol/RESULT.json",
        "sha256": "<receipt SHA256>"
      },
      "source_revision": "<repository commit, required for a copied checkout>",
      "model": "deepseek-flash",
      "policies": {
        "plain_codex": {"residual_sha256": null},
        "chisel_codex": {"residual_sha256": "<SHA256>"}
      },
      "test_authorized": false
    }

All listed hashes are checked before opening the selected scenario. Test access
additionally requires ``test_authorized: true`` and both frozen policy hashes.
Training candidates may each have a new freeze record, never a rewritten run.
``tree_sha256`` hashes a canonical sorted relative-path -> SHA256 mapping and
rejects symlinks. Frozen source files are copied into each new private run.
For this campaign, reuse the already measured receipt at
``/opt/phiagent/work/codex-chisel/20260929-v1/preflight/20260929T063057Z-codex-deepseek-protocol/RESULT.json``
(SHA256 ``3ff1eff7d8dd20317f99ae850a8f2c171c5cec473d5708465dcf37d88227d30c``).
It records native Responses acceptance and 36/17/53 input/output/total tokens,
not a benchmark result. Its shared setup usage is never charged to each episode.
The subsequent official native tool check is preserved separately at
``/opt/phiagent/work/codex-chisel/20260929-v1/preflight/codex-cli-native-tools-v2/ACCEPTANCE.json``,
SHA256 ``f45a7eb634f5c8bd95e830228fdeba621fdc40039217ad7a82bbdea056682fc5``:
three provider calls, 25385 input / 264 output / 25649 total tokens, and 16640
cached input tokens. This is working Codex tool execution, not GAIA benchmark
evidence; do not repeat or charge it to the first training scenario.

Common-runtime extensions required before any scenario is opened:

* ``mcp_servers: Mapping[str, Mapping[str, Any]]``: merge native MCP configuration.
* ``config_overrides: Mapping[str, Any]``: dotted native config overrides, limited
  here to disabling non-benchmark capabilities; never replace system instructions.
* ``stop_requested: Callable[[], str | None]``: poll while running, stop only the
  owned container, and return ``status='STOPPED'`` plus ``stop_reason`` and measured
  provider usage for a native stop, even if CLI completion usage is unavailable.
  Fast exits racing with the callback need the same explicit stop attribution.

The common launcher must put scratch files under its own workspace, not global
temporary directories. Existing host networking is sufficient for loopback MCP.
The adapter owns only MCP/native clock/scoring and a byte-preserving timing relay;
Docker, Codex configuration, AGENTS.md, subprocesses and cleanup belong exclusively
to ``phiagent.harness.codex_runtime.run_codex_trial``. Missing extensions produce
an explicit BLOCKED result, never a substitute subprocess implementation.

Prerequisites: Linux A800-4; pinned ARE 1.1.0; complete frozen GAIA2 filesystem;
the extended common Docker launcher; the original Codex static binary; frozen
assets/policies; an operable, metered native Responses service.
No benchmark, provider request, or ARE import is needed to run the synthetic tests.
"""

from __future__ import annotations

import base64
import ast
from collections.abc import Callable, Mapping
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from phiagent.harness.benchmark_partition import build_partition, validate_partition
from phiagent.harness.gaia2_execution_contract import (
    WALL_TIMEOUT_SECONDS, aligned_contract, judge_response_text, user_turn_limit,
)

CODEX_VERSION = "0.142.5"
CODEX_SOURCE_REVISION = "26de83050b20f7e0ee211b9739e52ae00ce8032a"
ARE_VERSION = "1.1.0"
GAIA2_REVISION = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
GAIA2_MINI_SHA256 = "88430154e09bb728c666bbbf367bb805c8a84f8bdee427e7d469887022ab71fe"
MODEL = "deepseek-flash"
SAFE_CODEX_TOOLS = {
    "update_plan", "view_image", "list_mcp_resources", "list_mcp_resource_templates",
    "read_mcp_resource", "request_user_input",
}
SPLIT_SEED = 20260928
REMOTE_HOST = "harnessmm-api-worker"
SOURCE_FILES = (
    "scripts/run_codex_gaia2.py",
    "phiagent/harness/codex_gaia2.py",
    "phiagent/harness/benchmark_partition.py",
    "phiagent/harness/codex_runtime.py",
    "phiagent/harness/gaia2_execution_contract.py",
    "scripts/serve_codex_deepseek.py",
)
NATIVE_BANK_SOURCE_FILES = (
    "phiagent/harness/codex_native_app_bank.py",
    "phiagent/harness/codex_jev_v2.py",
    "integrations/deepseek_harness/codex_bank_contract.mjs",
    "integrations/deepseek_harness/cot_jev.mjs",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ITERATIONS = 80
REMOVED_AUI_TOOLS = {
    "AgentUserInterface__get_last_message_from_user",
    "AgentUserInterface__get_last_message_from_agent",
    "AgentUserInterface__get_last_unread_messages",
    "AgentUserInterface__get_all_messages",
}
ADAPTER_INSTRUCTIONS = """You are operating the native ARE task environment.
Use only the are MCP tools. They execute the actual environment's app tools.
The observe tool and every app-tool result include currently delivered user
messages and environment notifications; these are your task and observations.
Send user-facing responses with AgentUserInterface__send_message_to_user.
Use SystemApp__wait_for_notification when waiting for the user or an event.
After sending a response, observe again if the episode has not finished.
Once episode_finished is true, end your response without further tool calls.
Your final Codex text is a transcript only; it is not an environment action.
Preserve the user's constraints and native tool descriptions.
"""


class GateError(RuntimeError):
    """An unmet integration/integrity prerequisite, never a task-success score."""


class NativeEpisodeEnded(RuntimeError):
    """An official environment/iteration/turn limit has ended the episode."""


def metered_native_judge_request(recorder, body):
    """Retry only a fully metered truncated evaluator response, with identical native input."""
    start = len(recorder.calls)
    retry_usage = []
    for attempt in range(3):
        try:
            wire, response, usage = recorder.request('judge', body)
            fees = [*retry_usage, usage]
            return wire, response, {key: sum(row[key] for row in fees)
                for key in ('input_tokens', 'output_tokens', 'total_tokens')}
        except GateError as error:
            call = recorder.calls[-1] if len(recorder.calls) > start else {}
            path = recorder.root / 'judge' / f"call-{call.get('sequence', 0):06d}" / 'response.json'
            try:
                raw = read_json(path)
            except (OSError, ValueError):
                raise error
            usage = call.get('usage') or {}
            if (attempt == 2 or body.get('stream') is not False
                    or body.get('max_output_tokens') != 4096
                    or call.get('role') != 'judge' or call.get('http_status') != 200
                    or call.get('error') != 'Native Responses returned an incomplete or failed response'
                    or raw.get('status') != 'incomplete'
                    or (raw.get('incomplete_details') or {}).get('reason') != 'max_output_tokens'
                    or usage.get('output_tokens') != 4096
                    or any(type(usage.get(key)) is not int or usage[key] < 0
                        for key in ('input_tokens', 'output_tokens', 'total_tokens'))
                    or usage['input_tokens'] + usage['output_tokens'] != usage['total_tokens']):
                raise
            # The original failed receipt and its fees remain in recorder.calls.
            # This clears only the recognized evaluator failure for a fresh identical request.
            recorder.failure = None
            retry_usage.append(usage)
    raise AssertionError('Unreachable evaluator retry state')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def tree_sha256(root: Path, pattern: str = "*") -> str:
    if not root.is_dir() or root.is_symlink():
        raise GateError(f"A real frozen directory is required: {root}")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise GateError(f"Frozen trees must not contain symlinks: {root}")
    records = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths
        if path.is_file() and path.match(pattern)
    }
    if not records:
        raise GateError(f"Frozen tree is empty: {root}")
    return canonical_sha256(records)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise GateError(f"Missing or invalid frozen SHA256: {label}")
    return value


def verify_file(path: Path, expected: Any, label: str) -> None:
    if file_sha256(path) != require_hash(expected, label):
        raise GateError(f"Frozen file changed: {label}")


def loopback_url(value: str) -> str:
    """Admit only loopback and the specifically authorized A800 Docker gateway."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1", "172.17.0.1")
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GateError("Use an explicit campaign-local HTTP bridge port, without URL credentials")
    return value.rstrip("/")


@dataclass(frozen=True)
class RunConfig:
    scenario: Path
    filesystem: Path
    output: Path
    codex_bin: Path
    provider_url: str
    freeze_manifest: Path
    partition_manifest: Path
    model: str = MODEL
    residual_file: Path | None = None
    judge_url: str | None = None
    relay_secret_file: Path | None = None
    trial_id: str | None = None
    native_bank_policy: dict[str, Any] | None = None
    jev_v2: dict[str, Any] | None = None

    def public_config(self) -> dict[str, Any]:
        value = {
            key: str(item) if isinstance(item, Path) else item
            for key, item in asdict(self).items()
            if key != "relay_secret_file"
        }
        return {
            **value,
            "runtime": f"official Codex CLI {CODEX_VERSION}",
            "wire_api": "responses",
            "transport": "byte_preserving_metering_proxy",
            "native_upstream": "https://api.deepseek.com/responses",
            "adapter": "native ARE AppTools over MCP; official ScenarioRunner/scorer",
            "method": "chisel_codex" if self.residual_file is not None else "plain_codex",
            "split_seed": SPLIT_SEED,
            "maximum_tokens": None,
            "token_usage_mode": "unlimited",
            "simulated_generation_time_mode": "measured",
            "codex_source_revision": CODEX_SOURCE_REVISION,
            "relay_auth_configured": self.relay_secret_file is not None,
            "trial_ids": {role: trial_identity(self, role) for role in ("actor", "judge")},
            "residual_application": "task_local_agents_md",
        }


def trial_identity(config: RunConfig, role: str) -> str:
    base = config.trial_id or (
        "gaia2-" + hashlib.sha256(str(config.output.resolve()).encode()).hexdigest()[:24]
    )
    ident = base + "-" + role
    if role not in ("actor", "judge") or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", ident) is None:
        raise GateError("Use a unique valid trial ID with space for the actor/judge suffix")
    return ident


def resolve_codex(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        if stream.read(4) == b"\x7fELF":
            return resolved
    if resolved.name != "codex.js":
        raise GateError("Expected the original Codex ELF or official npm codex.js launcher")
    package = resolved.parent.parent
    metadata = read_json(package / "package.json")
    if metadata.get("name") != "@openai/codex" or metadata.get("version") != CODEX_VERSION:
        raise GateError("The official npm Codex package must be exactly 0.142.5")
    for vendor in (
        package / "node_modules/@openai/codex-linux-x64/vendor",
        package / "vendor",
    ):
        candidate = vendor / "x86_64-unknown-linux-musl/bin/codex"
        if candidate.is_file():
            with candidate.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    return candidate.resolve()
    raise GateError("The original official static Linux Codex payload is missing")


def verify_freeze(
    config: RunConfig, executable: Path, native_source: Path, code_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Do not read scenario bytes until *all* source, policy and asset gates pass."""
    freeze = read_json(config.freeze_manifest)
    if freeze.get("schema") != "phiagent-codex-gaia2-freeze/v1":
        raise GateError("A phiagent-codex-gaia2-freeze/v1 freeze record is required")
    if not freeze.get("frozen_at_utc") or freeze.get("model") != config.model:
        raise GateError("Freeze timestamp/model does not match this run")
    if (
        freeze.get("dataset_revision") != GAIA2_REVISION
        or freeze.get("source_parquet_sha256") != GAIA2_MINI_SHA256
    ):
        raise GateError("Use the controller-prepared pinned official complete GAIA2-mini source")
    if config.model != MODEL:
        raise GateError("The authorized backbone is deepseek-flash")
    verify_file(config.partition_manifest, freeze.get("partition_sha256"), "partition")
    partition = read_json(config.partition_manifest)
    validate_partition(partition)
    if (
        partition.get("benchmark") != "gaia2-mini"
        or partition.get("split_seed") != SPLIT_SEED
        or partition.get("population") != 160
        or partition.get("counts") != {"train": 32, "validation": 32, "test": 96}
    ):
        raise GateError("Use the shared frozen GAIA2-mini 32/32/96 partition, seed 20260928")
    expected = build_partition(
        partition["records"], benchmark="gaia2-mini", seed=SPLIT_SEED
    )
    if partition["splits"] != expected["splits"]:
        raise GateError("Membership differs from the outcome-independent shared partition")
    ident = config.scenario.stem
    memberships = [name for name, ids in partition["splits"].items() if ident in ids]
    if len(memberships) != 1 or config.scenario.suffix != ".json":
        raise GateError("The scenario filename must identify exactly one frozen scenario")
    split = memberships[0]
    if config.jev_v2 is not None:
        if split != "train" or config.native_bank_policy is None or config.residual_file is None:
            raise GateError("Codex Jev-v2 requires its own non-noop frozen TRAIN intervention")
        if freeze.get("jev_v2") != config.jev_v2:
            raise GateError("Codex Jev-v2 routing differs from the source freeze")
    policies = freeze.get("policies", {})
    if freeze.get("native_bank_policy") != config.native_bank_policy:
        raise GateError("Native bank policy differs from the source-frozen configuration")
    if config.native_bank_policy is not None:
        from phiagent.harness.codex_native_app_bank import validate_policy
        validate_policy(config.native_bank_policy)
        if config.residual_file is None:
            raise GateError("A native bank condition requires its TRAIN-authored frozen skill")
    if policies.get("plain_codex", {}).get("residual_sha256", "missing") is not None:
        raise GateError("The plain Codex policy must have no residual")
    method = "chisel_codex" if config.residual_file is not None else "plain_codex"
    residual = ""
    if config.residual_file is not None:
        verify_file(
            config.residual_file,
            policies.get("chisel_codex", {}).get("residual_sha256"),
            "CHISEL residual",
        )
        residual = config.residual_file.read_text()
        if not residual.strip():
            raise GateError("A CHISEL residual must be nonempty")
    if split == "test":
        if freeze.get("test_authorized") is not True:
            raise GateError("Held-out test access is sealed until method/source selection freezes")
        require_hash(policies.get("chisel_codex", {}).get("residual_sha256"), "selected CHISEL")
    source_files = freeze.get("source_files", {})
    if not all(name in source_files for name in SOURCE_FILES):
        raise GateError("The complete adapter source closure must be frozen")
    if config.native_bank_policy is not None and not all(
        name in source_files for name in NATIVE_BANK_SOURCE_FILES
    ):
        raise GateError("Freeze the complete native bank compiler and controller source closure")
    for name, digest in source_files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise GateError("Adapter source freeze paths must be repository-relative")
        verify_file(code_root / relative, digest, name)
    bridge_files = freeze.get("bridge_source_files", {})
    if not isinstance(bridge_files, dict) or not bridge_files:
        raise GateError("Freeze the campaign bridge source, not merely its URL")
    for name, digest in bridge_files.items():
        if not Path(name).is_absolute():
            raise GateError("Bridge source paths must be absolute")
        verify_file(Path(name), digest, "bridge source")
    evidence = freeze.get("native_protocol_receipt", {})
    receipt_path = Path(evidence.get("path", ""))
    if not receipt_path.is_absolute():
        raise GateError("Freeze the already measured native Responses capability receipt")
    verify_file(receipt_path, evidence.get("sha256"), "native Responses capability receipt")
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "NATIVE_RESPONSES_ACCEPTED"
        or receipt.get("native_responses_http_status") != 200
        or receipt.get("model") != config.model
        or receipt.get("benchmark_calls") != 0
        or receipt.get("benchmark_evidence") is not False
    ):
        raise GateError("Native Responses operability evidence does not match this campaign")
    normalized_usage(receipt.get("usage", {}), "shared_setup")
    verify_file(executable, freeze.get("codex_sha256"), "official Codex payload")
    if tree_sha256(native_source, "*.py") != require_hash(
        freeze.get("are_source_sha256"), "ARE source"
    ):
        raise GateError("Pinned native ARE source differs from its freeze")
    if tree_sha256(config.filesystem) != require_hash(
        freeze.get("filesystem_sha256"), "GAIA2 filesystem"
    ):
        raise GateError("The complete frozen GAIA2 filesystem has changed")
    record = next(row for row in partition["records"] if row["id"] == ident)
    verify_file(config.scenario, record["payload_sha256"], "selected scenario")
    return {
        "scenario_id": ident,
        "split": split,
        "method": method,
        "residual": residual,
        "scenario_sha256": record["payload_sha256"],
        "freeze_sha256": file_sha256(config.freeze_manifest),
        "partition_sha256": file_sha256(config.partition_manifest),
        "source_files": source_files,
        "bridge_source_files": bridge_files,
        "native_protocol_receipt": {
            "path": str(receipt_path), "sha256": evidence["sha256"],
            "status": receipt["status"], "usage": receipt["usage"],
            "shared_setup_cost_recounted": False, "benchmark_evidence": False,
        },
    }


def normalized_usage(raw: Mapping[str, Any], role: str) -> dict[str, Any]:
    names = ("input_tokens", "output_tokens", "total_tokens")
    values = [raw.get(name) for name in names]
    if any(type(value) is not int or value < 0 for value in values):
        raise GateError(f"{role} provider usage is missing, invalid, or unreported")
    if values[0] + values[1] != values[2]:
        raise GateError(f"{role} provider token counts do not reconcile")
    cached = (raw.get("input_tokens_details") or {}).get("cached_tokens")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= values[0]):
        raise GateError(f"{role} cached-token accounting is invalid")
    return {
        "input_tokens": values[0],
        "output_tokens": values[1],
        "total_tokens": values[2],
        "cached_input_tokens": cached,
        "cost_usd": None,
        "cost_status": "not_reported; do not interpret as zero",
    }


def judge_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the private native judge's API envelope, not its prompt or metric."""
    converted = []
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": part["text"],
                })
            elif part.get("type") == "image_url" and role == "user":
                parts.append({"type": "input_image", "image_url": part["image_url"]["url"]})
            else:
                raise GateError("Unsupported native judge message part; no prompt was dropped")
        converted.append({"role": role, "content": parts})
    return converted


def responses_text(response: dict[str, Any]) -> str:
    text = "".join(
        part["text"]
        for item in response.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    )
    if not text:
        raise GateError("Native judge Responses result has no textual verdict")
    return text


def response_events(raw: bytes) -> list[dict[str, Any]]:
    events = []
    for block in raw.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in block.splitlines()
                         if line.startswith("data:"))
        if data and data != "[DONE]":
            event = json.loads(data)
            if not isinstance(event, dict):
                raise GateError("A Responses SSE event must be an object")
            events.append(event)
    return events


def completed_response(raw: bytes) -> dict[str, Any]:
    """Require a real terminal Responses SSE receipt, not a text-only success."""
    completed = []
    for event in response_events(raw):
        if event.get("type") in ("error", "response.failed", "response.incomplete"):
            raise GateError("The actor provider stream failed or was incomplete; inspect raw receipt")
        if event.get("type") == "response.completed":
            completed.append(event.get("response"))
    if len(completed) != 1 or not isinstance(completed[0], dict):
        raise GateError("A single response.completed receipt with usage is required")
    if completed[0].get("status") != "completed":
        raise GateError("The Responses receipt is not a completed response")
    return completed[0]


def validate_response_tools(
    raw: bytes, completed: dict[str, Any], allowed: set[str]
) -> None:
    output = completed.get("output", [])
    if not isinstance(output, list):
        raise GateError("Responses output is not an item list")
    items = output + [
        event["item"] for event in response_events(raw)
        if event.get("type") in ("response.output_item.added", "response.output_item.done")
        and isinstance(event.get("item"), dict)
    ]
    for item in items:
        kind = item.get("type")
        if kind == "function_call":
            name, namespace = item.get("name"), item.get("namespace")
            qualified = f"{namespace}.{name}" if namespace else name
            aliases = {qualified}
            if not namespace and isinstance(name, str):
                if "::" in name:
                    aliases.add(name.replace("::", ".", 1))
                if name.startswith("mcp__are__"):
                    aliases.add("mcp__are." + name.removeprefix("mcp__are__"))
            native_namespace = namespace == "mcp__are" or (
                not namespace and isinstance(name, str)
                and name.startswith(("mcp__are::", "mcp__are__", "mcp__are."))
            )
            if not aliases.intersection(allowed) and not native_namespace:
                raise GateError("Responses escaped the native tool capability boundary")
        if kind not in ("message", "reasoning", "function_call"):
            raise GateError("Responses attempted a non-native tool or unsupported output item")


def native_tool_names(tools: list[dict[str, Any]], allowed: set[str]) -> set[str]:
    offered = set()

    def visit(items, namespace=""):
        for tool in items:
            kind, name = tool.get("type"), tool.get("name")
            if not isinstance(name, str) or not name:
                raise GateError("Codex tool declaration is missing its name")
            if kind == "namespace":
                children = tool.get("tools")
                if namespace or not isinstance(children, list):
                    raise GateError("Unsupported nested Codex tool namespace")
                visit(children, name)
                continue
            canonical = name if not namespace or namespace == "functions" else namespace + "__" + name
            if kind != "function" or canonical not in allowed:
                raise GateError(
                    "Codex offered a tool outside the isolated native capability boundary: "
                    + json.dumps({"type": kind, "namespace": namespace, "name": name})
                )
            offered.add(namespace + "." + name if namespace else name)
            if canonical == name:
                offered.add(name)

    visit(tools)
    return offered


def native_only_request(body: dict[str, Any], allowed: set[str], model: str) -> dict[str, Any]:
    if body.get("model") != model or body.get("stream") is not True:
        raise GateError("Codex must use the frozen model and its native streaming Responses API")
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        raise GateError("Unexpected Codex Responses tool representation")
    native_tool_names(tools, allowed)
    return body


class ProviderRecorder:
    def __init__(self, root: Path, config: RunConfig):
        self.root, self.config = root, config
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.opener = build_opener(ProxyHandler({}))
        self.failure: str | None = None
        if config.relay_secret_file is None:
            raise GateError("The common launcher's owner-private relay secret is required")
        from scripts.serve_codex_deepseek import read_credential, trial_token

        secret = read_credential(config.relay_secret_file)
        self.trial_ids = {role: trial_identity(config, role) for role in ("actor", "judge")}
        self.tokens = {role: trial_token(secret, ident) for role, ident in self.trial_ids.items()}
        self._redactions = (secret, *self.tokens.values())

    def redact(self, message: str) -> str:
        for token in self._redactions:
            message = message.replace(token, "[REDACTED]")
        return message

    def trial_url(self, role: str) -> str:
        base = self.config.provider_url if role == "actor" else (
            self.config.judge_url or self.config.provider_url
        )
        return base.rstrip("/") + "/" + self.trial_ids[role]

    def proxy_usage(self, role: str) -> bytes:
        request = Request(
            self.trial_url(role) + "/usage",
            headers={"Authorization": "Bearer " + self.tokens[role]},
        )
        with self.opener.open(request, timeout=30) as response:
            raw = response.read()
        if json.loads(raw).get("trial_id") != self.trial_ids[role]:
            raise GateError("The metering proxy returned usage for a different trial")
        return raw

    def request(
        self, role: str, body: dict[str, Any], allowed: set[str] | None = None,
        *, wire_body: bytes | None = None, compact: bool = False, native_fast: bool = False,
    ) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        if native_fast:
            if compact or self.config.native_bank_policy is None or not self.config.native_bank_policy["enabled"]:
                raise GateError("Native fast calls require their enabled frozen bank policy")
            from scripts.serve_codex_deepseek import validate_native_fast_request
            validate_native_fast_request(body)
        streaming = not compact and body.get("stream") is True
        payload = wire_body if wire_body is not None else json.dumps(body).encode()
        if wire_body is not None and json.loads(wire_body) != body:
            raise GateError("Request bytes differ from the admitted native Codex request")
        with self.lock:
            sequence = len(self.calls) + 1
            call = {"sequence": sequence, "role": role, "started_at_utc": now(),
                    "status": "PENDING", "usage": None, "trial_id": self.trial_ids[role]}
            self.calls.append(call)
        folder = self.root / role / f"call-{sequence:06d}"
        folder.mkdir(parents=True)
        (folder / "request.json").write_bytes(payload)
        call["request_sha256"] = hashlib.sha256(payload).hexdigest()
        call["wire_api"] = "chat/completions" if native_fast else "responses"
        call["native_fast_decision"] = native_fast
        call["body_rewritten"] = False
        write_json(folder / "receipt.json", call)
        endpoint = "/chat/completions" if native_fast else ("/responses/compact" if compact else "/responses")
        url = self.trial_url(role) + endpoint
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if streaming else "application/json",
            "X-Phiagent-Role": role,
            "X-Phiagent-Run-Id": str(self.config.output.resolve()),
            "X-Phiagent-Request-Id": canonical_sha256({
                "run": str(self.config.output.resolve()), "role": role, "sequence": sequence,
            }),
        }
        if role in self.tokens:
            headers["Authorization"] = "Bearer " + self.tokens[role]
        start = time.monotonic()
        try:
            request = Request(url, data=payload, headers=headers)
            with self.opener.open(request, timeout=1860) as response:
                raw = response.read()
                call["http_status"] = response.status
                call["content_type"] = response.headers.get("Content-Type")
            (folder / ("response.sse" if streaming else "response.json")).write_bytes(raw)
            parsed = completed_response(raw) if streaming else json.loads(raw)
            if native_fast:
                from scripts.serve_codex_deepseek import native_chat_usage
                usage = normalized_usage(native_chat_usage(parsed), role)
                call["native_usage"] = parsed.get("usage")
            else:
                usage = normalized_usage(parsed.get("usage", {}), role)
            call["usage"] = usage
            if not compact and not native_fast and parsed.get("status") != "completed":
                raise GateError("Native Responses returned an incomplete or failed response")
            if parsed.get("model") not in (None, self.config.model):
                raise GateError("The provider returned an unexpected model identity")
            if role == "actor" and not compact and not native_fast:
                validate_response_tools(raw, parsed, allowed or set())
            call["status"] = "COMPLETED"
            return raw, parsed, usage
        except Exception as error:
            if isinstance(error, HTTPError):
                (folder / "http-error.txt").write_text(
                    self.redact(error.read().decode(errors="replace"))
                )
                call["http_status"] = error.code
            call["status"] = "FAILED"
            call["error_type"] = type(error).__name__
            call["error"] = self.redact(str(error))
            self.failure = f"{role} request {sequence}: {call['error']}"
            raise GateError(self.failure) from error
        finally:
            call["elapsed_seconds"] = time.monotonic() - start
            call["finished_at_utc"] = now()
            write_json(folder / "receipt.json", call)
            write_json(self.root / "usage.json", self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            groups = {}
            for role in ("actor", "judge"):
                calls = [call for call in self.calls if call["role"] == role]
                reported = [call["usage"] for call in calls if call["usage"] is not None]
                complete = all(call["usage"] is not None for call in calls)
                sums = {
                    key: sum(usage[key] for usage in reported)
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                }
                groups[role] = {
                    "trial_id": self.trial_ids[role],
                    "requests": len(calls), "complete": complete,
                    "usage": sums if complete else None,
                    "known_usage_subtotal": sums,
                    "cached_input_tokens": (
                        sum(usage["cached_input_tokens"] for usage in reported)
                        if complete and all(usage["cached_input_tokens"] is not None
                                            for usage in reported) else None
                    ),
                    "cache_write_tokens": None if calls else 0,
                    "unresolved_calls": [call["sequence"] for call in calls
                                         if call["usage"] is None],
                    "failed_calls": [call["sequence"] for call in calls if call["status"] == "FAILED"],
                    "cost_usd": None if calls else 0,
                    "cost_status": "not_reported" if calls else "verified_no_calls",
                }
            return groups


def public_content(value: Any, attachments: list[Any] | None = None) -> list[dict[str, Any]]:
    """Serialize only actual native tool return values and delivered attachments."""
    native_types = sys.modules.get("are.simulation.agents.llm.types")
    # Email records also have content/attachments; their paths remain textual metadata.
    if native_types is not None and isinstance(value, native_types.MMObservation):
        attachments = list(value.attachments)
        value = value.content
    content = [{"type": "text", "text": str(value)}]
    for attachment in attachments or []:
        if not hasattr(attachment, "mime") or not hasattr(attachment, "base64_data"):
            raise GateError("Native image attachments require typed MIME and base64 fields")
        mime = attachment.mime
        if not isinstance(mime, str) or not mime.startswith("image/"):
            raise GateError(f"Native attachment transport is not implemented for {mime}")
        data = attachment.base64_data
        if isinstance(data, bytes):
            data = data.decode("ascii")
        base64.b64decode(data, validate=True)
        content.append({"type": "image", "mimeType": mime, "data": data})
    return content


def mcp_name(public_name: str) -> str:
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,52}", public_name) and public_name != "observe":
        return public_name
    return "native_" + hashlib.sha256(public_name.encode()).hexdigest()[:24]


class NativeSurface:
    """A native observation/action boundary. It contains no language-model loop."""

    def __init__(
        self, env: Any, notification_system: Any, tools: list[dict[str, Any]],
        callables: Mapping[str, Callable], original_names: Mapping[str, str],
        expected_turns: int | None, audit_path: Path,
    ):
        self.env, self.notifications = env, notification_system
        self.tools, self.callables, self.original_names = tools, callables, original_names
        self.expected_turns, self.audit_path = expected_turns, audit_path
        self.lock = threading.RLock()
        self.generations = 0
        self.turns = 0
        self.stop_reason: str | None = None
        self.failure: str | None = None
        self.pending_attachments: list[dict[str, Any]] = []
        self.generating = False
        self.public_notifications: list[dict[str, Any]] = []

    @property
    def finished(self) -> bool:
        state = getattr(self.env.state, "value", self.env.state)
        return (
            str(state).lower() not in ("running", "paused")
            or (self.expected_turns is not None and self.turns >= self.expected_turns)
            or self.stop_reason is not None
        )

    def observe(self) -> list[dict[str, Any]]:
        timestamp = self.env.time_manager.time()
        delivered = self.notifications.message_queue.get_by_timestamp(
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
        )
        messages, images = [], []
        for notification in delivered:
            kind = getattr(notification.message_type, "value", notification.message_type)
            if kind in ("USER_MESSAGE", "ENVIRONMENT_NOTIFICATION"):
                messages.append({
                    "type": kind, "message": notification.message,
                    "timestamp": notification.timestamp.isoformat(),
                })
                if messages[-1] not in self.public_notifications:
                    self.public_notifications.append(dict(messages[-1]))
                images.extend(notification.attachments)
            elif kind == "ENVIRONMENT_STOP":
                self.stop_reason = "native_environment_stop"
            else:
                raise GateError("Unknown native observable notification type")
        content = public_content(json.dumps({
            "simulation_time": timestamp,
            "messages": messages,
            "episode_finished": self.finished,
        }, ensure_ascii=False), images)
        pending, self.pending_attachments = self.pending_attachments, []
        return content + pending

    def audit(self, record: dict[str, Any]) -> None:
        with self.audit_path.open("a") as stream:
            stream.write(json.dumps({"at_utc": now(), **record}, ensure_ascii=False) + "\n")

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.generating:
                raise GateError("Native tools cannot execute during a pending model generation")
            if name not in self.callables and name != "observe":
                result = {
                    "content": public_content(f"Unknown native MCP tool: {name}"),
                    "isError": True,
                }
                self.audit({"tool": name, "arguments": arguments, "result": result})
                return result
            if not isinstance(arguments, dict):
                raise GateError("MCP tool arguments must be an object")
            if name == "observe" and arguments:
                raise GateError("The native observation adapter takes no arguments")
            before = self.observe()
            if self.finished or name == "observe":
                result = {"content": before, "isError": False}
            else:
                try:
                    value = self.callables[name](**arguments)
                except Exception as error:
                    result = {"content": public_content(str(error)), "isError": True}
                else:
                    if self.original_names[name] == "AgentUserInterface__send_message_to_user":
                        self.turns += 1
                    result = {"content": public_content(value), "isError": False}
                result["content"] = before + result["content"] + self.observe()
            self.audit({"tool": name, "arguments": arguments, "result": result})
            return result

    @contextmanager
    def generation(self):
        with self.lock:
            if self.generating:
                raise GateError("Concurrent actor generations violate the native sequential protocol")
            if self.finished:
                if self.stop_reason is None:
                    self.stop_reason = (
                        "native_turn_limit" if self.expected_turns is not None and self.turns >= self.expected_turns
                        else "native_environment_stop"
                    )
                raise NativeEpisodeEnded(self.stop_reason)
            if self.generations >= NATIVE_ITERATIONS:
                sender = next(
                    (name for name, original in self.original_names.items()
                     if original == "AgentUserInterface__send_message_to_user"), None
                )
                if sender is None:
                    raise GateError("Native iteration-limit user notification tool is missing")
                # Exactly the pinned ARE termination action, not an invented fallback answer.
                self.callables[sender](
                    content=f"Max iterations ({NATIVE_ITERATIONS}) reached. Stopping."
                )
                self.stop_reason = "native_max_iterations"
                raise NativeEpisodeEnded(self.stop_reason)
            self.generations += 1
            self.env.pause()
            self.generating = True
            start = time.monotonic()
        try:
            yield
        finally:
            with self.lock:
                try:
                    self.env.resume_with_offset(time.monotonic() - start)
                finally:
                    self.generating = False


def native_surface(env: Any, scenario: Any, audit_path: Path) -> NativeSurface:
    from pydantic import ConfigDict, create_model
    from are.simulation.tool_utils import AppToolAdapter

    tools = [{
        "name": "observe",
        "description": "Read currently delivered native user messages and notifications.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    }]
    callables, originals = {}, {}
    aui_found = False
    for native in scenario.get_tools():
        if native.name.startswith("AgentUserInterface__"):
            native.class_instance.wait_for_user_response = False
            aui_found = True
        if native.name in REMOVED_AUI_TOOLS:
            continue
        adapter = AppToolAdapter(native)
        name = mcp_name(adapter.name)
        if name in callables:
            raise GateError("Native MCP tool-name collision")
        fields = {
            arg.name: (
                arg.type_obj if arg.type_obj is not None else Any,
                arg.default if arg.has_default else ...,
            )
            for arg in native.args
        }
        model = create_model(name + "Args", __config__=ConfigDict(extra="forbid"), **fields)
        schema = model.model_json_schema()
        for arg in native.args:
            if arg.description:
                schema["properties"][arg.name]["description"] = arg.description
        tools.append({"name": name, "description": adapter.description, "inputSchema": schema})
        callables[name], originals[name] = native, native.name
    if not aui_found:
        raise GateError("Native scored scenario did not supply a user interface")
    turns = user_turn_limit(scenario.nb_turns, None)
    return NativeSurface(
        env, env.notification_system, tools, callables, originals,
        turns, audit_path,
    )


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, recorder: ProviderRecorder):
        self.recorder = recorder
        self.surface: NativeSurface | None = None
        self.mcp_admitted = False
        self.bank = None
        self.token = recorder.tokens["actor"]
        self.actor_path = "/" + recorder.trial_ids["actor"]
        super().__init__(("127.0.0.1", 0), AdapterHandler)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def native_bank(self):
        if self.recorder.config.native_bank_policy is None:
            return None
        if self.bank is None:
            from phiagent.harness.codex_native_app_bank import NativeBankBoundary, fast_request
            surface = self.surface
            if surface is None:
                raise GateError("Native bank requires the original ARE surface")

            def judge(packet):
                _, parsed, _ = self.recorder.request("actor", fast_request(packet), native_fast=True)
                return parsed

            boundary = NativeBankBoundary
            extra = {}
            if self.recorder.config.jev_v2 is not None:
                from phiagent.harness.codex_jev_v2 import CodexJevV2
                boundary = CodexJevV2
                extra = {"v2": self.recorder.config.jev_v2}
            self.bank = boundary(
                surface.tools, enabled=self.recorder.config.native_bank_policy["enabled"],
                judge=judge, execute=surface.call, generation=surface.generation,
                finished=lambda: surface.finished, audit=surface.audit, **extra,
            )
            self.bank.native_notifications = lambda: surface.public_notifications
        return self.bank

    def rpc(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request.get("jsonrpc") != "2.0":
            raise GateError("MCP requires JSON-RPC 2.0")
        method, params = request.get("method"), request.get("params", {})
        if method in ("notifications/initialized", "notifications/cancelled"):
            return None
        if method == "initialize":
            version = params.get("protocolVersion")
            if version not in ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"):
                version = "2025-03-26"
            result = {
                "protocolVersion": version, "capabilities": {"tools": {}},
                "serverInfo": {"name": "phiagent-native-are", "version": "1"},
            }
        elif method == "ping":
            result = {}
        elif method in ("resources/list", "resources/templates/list", "prompts/list"):
            key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates",
                   "prompts/list": "prompts"}[method]
            result = {key: []}
        elif method == "tools/list" and self.surface is not None:
            self.mcp_admitted = True
            result = {"tools": self.surface.tools}
        elif method == "tools/call" and self.surface is not None:
            bank = self.native_bank()
            result = (bank.call if bank is not None else self.surface.call)(
                params.get("name"), params.get("arguments", {})
            )
        else:
            return {"jsonrpc": "2.0", "id": request.get("id"),
                    "error": {"code": -32601, "message": "Method not available"}}
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:
        pass

    def respond(self, status: int, payload: bytes = b"", mime: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def authorized(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.server.token
        )

    def do_GET(self) -> None:
        if not self.authorized():
            self.respond(401)
        elif self.path == "/health":
            self.respond(200, b'{"ready":true}')
        elif self.path == self.server.actor_path + "/usage":
            try:
                self.respond(200, self.server.recorder.proxy_usage("actor"))
            except Exception as error:
                message = self.server.recorder.redact(str(error))
                self.server.recorder.failure = message
                self.respond(502, json.dumps({"error": message}).encode())
        else:
            self.respond(405)

    def do_DELETE(self) -> None:
        self.respond(405 if self.authorized() else 401)

    def do_POST(self) -> None:
        if not self.authorized():
            self.respond(401)
            return
        try:
            wire_body = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(wire_body)
            if self.path == "/mcp":
                result = self.server.rpc(body)
                self.respond(202 if result is None else 200,
                             b"" if result is None else json.dumps(result).encode())
            elif self.path in (
                self.server.actor_path + "/responses",
                self.server.actor_path + "/responses/compact",
            ) and (
                self.server.surface is not None
            ):
                if not self.server.mcp_admitted:
                    raise GateError("Codex must register the required ARE MCP server first")
                surface = self.server.surface
                allowed = SAFE_CODEX_TOOLS | {"mcp__are__" + tool["name"] for tool in surface.tools}
                compact = self.path.endswith("/compact")
                if compact:
                    if body.get("model") != self.server.recorder.config.model:
                        raise GateError("Native compaction must use the frozen actor model")
                else:
                    native_only_request(body, allowed, self.server.recorder.config.model)
                offered = native_tool_names(body.get("tools", []), allowed) if not compact else set()
                surface.audit({"capability_boundary": {
                    "offered_tool_names": sorted(offered),
                    "body_rewritten": False, "native_compaction": compact,
                }})
                with surface.generation():
                    raw, parsed, _ = self.server.recorder.request(
                        "actor", body, offered, wire_body=wire_body, compact=compact
                    )
                if not compact:
                    bank = self.server.native_bank()
                    if bank is not None:
                        from phiagent.harness.codex_native_app_bank import public_user_text
                        bank.register(parsed, public_task=public_user_text(body))
                self.respond(200, raw, "application/json" if compact else "text/event-stream")
            else:
                if self.path.startswith(self.server.actor_path + "/"):
                    self.server.recorder.failure = (
                        f"Unsupported native Codex route {self.path}; no transport fallback was used"
                    )
                self.respond(404)
        except NativeEpisodeEnded as error:
            self.server.surface.stop_reason = str(error)
            self.respond(409, json.dumps({"error": str(error)}).encode())
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            message = self.server.recorder.redact(str(error))
            self.server.recorder.failure = message
            if self.server.surface is not None:
                self.server.surface.failure = message
            self.respond(502, json.dumps({"error": message}).encode())


def require_common_runtime(executor: Callable | None = None) -> Callable:
    """Static interface admission only: this never starts Codex or Docker."""
    if executor is None:
        from phiagent.harness.codex_runtime import run_codex_trial
        executor = run_codex_trial
    required = {"mcp_servers", "config_overrides", "stop_requested"}
    missing = required - set(inspect.signature(executor).parameters)
    if missing:
        raise GateError(
            "Common run_codex_trial requires GAIA extensions: " + ", ".join(sorted(missing))
        )
    source = ast.parse(textwrap.dedent(inspect.getsource(executor)))
    if any(
        isinstance(node, ast.Constant) and isinstance(node.value, str)
        and re.search(r"(?:^|[= ])/(?:var/)?tmp(?:[/: ]|$)", node.value)
        for node in ast.walk(source)
    ):
        raise GateError("Common runtime must use workspace-local scratch, not global temporary paths")
    if inspect.iscoroutinefunction(executor):
        raise GateError("The common runtime contract must be synchronous")
    return executor


def mcp_runtime_settings(server: AdapterServer, scenario_timeout: int) -> dict[str, Any]:
    def stop_requested():
        if server.recorder.failure or server.surface.failure:
            return "infrastructure_failure"
        if not server.surface.finished:
            return None
        if server.surface.stop_reason is None:
            server.surface.stop_reason = (
                "native_turn_limit" if server.surface.expected_turns is not None and server.surface.turns >= server.surface.expected_turns
                else "native_environment_stop"
            )
        return server.surface.stop_reason

    overrides = {
        f"features.{name}": False for name in (
            "shell_tool", "unified_exec", "shell_snapshot", "apply_patch_freeform",
            "apps", "plugins", "hooks", "codex_hooks", "multi_agent", "collab",
            "memory_tool", "memories", "browser_use", "computer_use", "image_generation",
            "search_tool", "tool_search", "remote_models", "responses_websockets",
            "responses_websockets_v2", "skill_mcp_dependency_install",
            "enable_request_compression", "token_budget", "rollout_budget", "goals",
        )
    }
    overrides.update(web_search="disabled", check_for_update_on_startup=False)
    return {
        "mcp_servers": {"are": {
            "url": server.base_url + "/mcp",
            "bearer_token_env_var": "CODEX_PROVIDER_TOKEN",
            "required": True, "startup_timeout_sec": 60,
            "tool_timeout_sec": scenario_timeout, "supports_parallel_tool_calls": False,
        }},
        "config_overrides": overrides,
        "stop_requested": stop_requested,
    }


def run_common_actor(
    config: RunConfig, executable: Path, server: AdapterServer, prompt: str, residual: str,
    scenario_timeout: int, *, executor: Callable | None = None,
) -> dict[str, Any]:
    launch = require_common_runtime(executor)
    result = launch(
        output=config.output / "codex", prompt=prompt,
        trial_id=server.recorder.trial_ids["actor"], provider_url=server.base_url,
        relay_secret_file=config.relay_secret_file, codex_binary=executable,
        images=(), residual=residual, timeout_seconds=scenario_timeout,
        **mcp_runtime_settings(server, scenario_timeout),
    )
    if not isinstance(result, dict):
        raise GateError("Common run_codex_trial did not return its documented result object")
    write_json(config.output / "private/codex-process.json", result)
    if server.recorder.failure or server.surface.failure:
        raise GateError(server.recorder.failure or server.surface.failure)
    if result.get("status") != "COMPLETED":
        stopped = (
            result.get("status") == "STOPPED" and server.surface.finished
            and result.get("stop_reason") == server.surface.stop_reason
            and result.get("stop_reason") in {
                "native_turn_limit", "native_environment_stop", "native_max_iterations",
            }
        )
        if not stopped:
            raise GateError("Common Codex runtime did not complete or attest a native protocol stop")
    measured = result.get("provider_usage", {})
    captured = server.recorder.snapshot()["actor"]
    if (
        measured.get("usage_status") != "MEASURED" or captured["requests"] == 0
        or not captured["complete"] or measured.get("known_usage") != captured["usage"]
    ):
        raise GateError("Common provider ledger and native actor receipts do not reconcile")
    return result


def run_native(
    config: RunConfig, executable: Path, admission: dict[str, Any], recorder: ProviderRecorder
) -> dict[str, Any]:
    # These imports can initialize native filesystem state. They occur only after
    # remote authorization, source/policy admission, and run-local storage setup.
    from are.simulation.agents.agent_config_builder import AgentConfigBuilder
    from are.simulation.agents.agent_execution_result import AgentExecutionResult
    from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine, LiteLLMModelConfig
    from are.simulation.benchmark.scenario_loader import load_scenario
    from are.simulation.scenario_runner import ScenarioRunner
    from are.simulation.scenarios.config import ScenarioRunnerConfig
    from are.simulation.scenarios.scenario_imported_from_json.utils import (
        get_scenario_duration, preprocess_scenario,
    )
    from are.simulation.validation.configs import GraphPerEventJudgeConfig

    class MeteredNativeJudge(LiteLLMEngine):
        def chat_completion(self, messages, stop_sequences=(), **_kwargs):
            converted = [self._convert_message_to_litellm_format(item) for item in messages]
            body = {
                "model": config.model, "stream": False, "temperature": 0,
                "max_output_tokens": 4096, "input": judge_responses_input(converted),
            }
            _, raw, usage = metered_native_judge_request(recorder, body)
            text = responses_text(raw)
            if not isinstance(text, str):
                raise GateError("Native judge provider returned no textual verdict")
            # The native judge parses [[True]]/[[False]] case-sensitively.
            # Figure 2 preserves provider text; actor boolean rewriting must
            # never be applied to evaluator verdicts.
            text = judge_response_text(text, stop_sequences)
            return text, {
                "prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
            }

    class CodexConfigBuilder(AgentConfigBuilder):
        def build(self, agent_name, **kwargs):
            if agent_name != "codex_mcp":
                raise GateError("This adapter admits only the official Codex actor")
            # Reuse the runner's configuration shape, never its default agent/loop.
            return super().build(agent_name="default", **kwargs)

    server = AdapterServer(recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with build_opener(ProxyHandler({})).open(Request(
            server.base_url + "/health",
            headers={"Authorization": "Bearer " + server.token},
        ), timeout=5) as response:
            if response.status != 200 or not thread.is_alive():
                raise GateError("The run-local adapter is not responsive")

        class CodexAgent(RunnableARESimulationAgent):
            def __init__(self, env):
                self.env = env

            def run_scenario(self, scenario, notification_system=None, initial_agent_logs=None):
                if initial_agent_logs:
                    raise GateError("This new-run adapter does not replay prior agent trajectories")
                surface = native_surface(
                    self.env, scenario, config.output / "private/public-observations.jsonl"
                )
                server.surface = surface
                write_json(config.output / "private/native-tools.json", {
                    "tools": surface.tools, "original_names": surface.original_names,
                })
                # Native notifications, never scenario JSON/ground truth, supply the task.
                initial = surface.observe()
                prompt = ADAPTER_INSTRUCTIONS + "\nComplete the currently delivered native task.\n"
                if config.native_bank_policy is not None:
                    from phiagent.harness.codex_native_app_bank import instruction
                    prompt += instruction(config.native_bank_policy) + "\n"
                if scenario.additional_system_prompt:
                    prompt += scenario.additional_system_prompt + "\n"
                surface.pending_attachments = [item for item in initial if item["type"] != "text"]
                prompt += "\n".join(item["text"] for item in initial if item["type"] == "text")
                if surface.pending_attachments:
                    prompt += "\nNative task images are available from observe; inspect them there."
                actor = run_common_actor(
                    config, executable, server, prompt, admission["residual"],
                    WALL_TIMEOUT_SECONDS,
                )
                return AgentExecutionResult(output=actor.get("final_answer"), metadata=actor)

        class CodexBuilder:
            def build(self, agent_config, env=None, mock_responses=None):
                if env is None or mock_responses is not None:
                    raise GateError("A real native environment, not mock actor responses, is required")
                return CodexAgent(env)

        scenario, replay = load_scenario(
            config.scenario.read_text(), str(config.scenario), load_completed_events=False
        )
        if scenario is None or replay:
            raise GateError("Official native scenario import failed or attempted trajectory replay")
        if scenario.scenario_id != admission["scenario_id"]:
            raise GateError("Native scenario identity does not match the frozen partition")
        runner_config = ScenarioRunnerConfig(
            agent="codex_mcp", model=config.model, model_provider="local", export=True,
            output_dir=str(config.output / "private/native-traces"),
            simulated_generation_time_mode="measured",
            max_turns=None,
        )
        judge = MeteredNativeJudge(LiteLLMModelConfig(
            model_name=config.model, provider="local",
            endpoint=recorder.trial_url("judge"),
        ))
        preprocess_scenario(
            scenario, judge_config=GraphPerEventJudgeConfig(engine=judge),
            max_scenario_duration=get_scenario_duration(
                scenario, runner_config.max_time_scenario_duration,
                runner_config.max_scenario_duration,
            ),
            offline_validation=False,
        )
        if scenario.judge is None:
            raise GateError("The native scenario has no official oracle-backed judge")
        write_json(config.output / "private/native-config.json", {
            "config": runner_config.model_dump(mode="json"),
            "scenario_seed": scenario.seed, "scenario_duration": scenario.duration,
            "scenario_time_increment": scenario.time_increment_in_seconds,
            "native_max_iterations": NATIVE_ITERATIONS,
            "outer_wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
            "native_execution_contract": aligned_contract(),
            "judge": {
                "implementation": "official GraphPerEventJudge",
                "model": config.model, "wire_api": "responses",
                "max_output_tokens": 4096, "temperature": 0,
            },
        })
        result = ScenarioRunner(
            agent_builder=CodexBuilder(), agent_config_builder=CodexConfigBuilder()
        ).run(runner_config, scenario)
        native = {
            "success": result.success,
            "exception": recorder.redact(str(result.exception)) if result.exception else None,
            "rationale": result.rationale, "duration": result.duration,
            "export_path": result.export_path,
        }
        write_json(config.output / "private/native-result.json", native)
        if recorder.failure:
            raise GateError(recorder.failure)
        if result.exception is not None or type(result.success) is not bool:
            raise GateError("Official native validation did not produce a valid boolean verdict")
        return native
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def owned_episode_output(path: Path) -> Path:
    output = path.resolve()
    shared_deepseek = Path("/opt/phiagent/work/deepseek-harness/20260929-v1")
    comparison_root = Path("/path/to/data/phiagent/work/industrial-gate-comparison/codex")
    legacy = output.is_relative_to("/opt/phiagent/work") and not output.is_relative_to(shared_deepseek)
    comparison = output != comparison_root and output.is_relative_to(comparison_root)
    if not (legacy or comparison):
        raise GateError("Use a new campaign run outside the shared active deepseek-harness checkout")
    return output


def run_episode(config: RunConfig) -> dict[str, Any]:
    if sys.platform != "linux" or socket.gethostname() != REMOTE_HOST:
        raise GateError("Execution is restricted to the authorized A800-4; no local benchmark runs")
    loopback_url(config.provider_url)
    loopback_url(config.judge_url or config.provider_url)
    if urlsplit(config.provider_url).path.rstrip("/") or (
        config.judge_url and urlsplit(config.judge_url).path.rstrip("/")
    ):
        raise GateError("Pass a metering root URL; this adapter adds separate actor/judge trial IDs")
    output = owned_episode_output(config.output)
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=False)
    (output / "private").mkdir()
    write_json(output / "CONFIG.json", config.public_config())
    write_json(output / "STARTED.json", {
        "at_utc": now(), "hostname": socket.gethostname(), "python": sys.version,
        "command": sys.argv, "state": "PREFLIGHT",
        "source_files": {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES},
        "dataset_revision": GAIA2_REVISION, "source_parquet_sha256": GAIA2_MINI_SHA256,
        "split_seed": SPLIT_SEED,
    })
    result = {
        "status": "BLOCKED", "native_score": None, "metric": "official ARE task success",
        "method": "chisel_codex" if config.residual_file is not None else "plain_codex",
        "runtime": f"official Codex CLI {CODEX_VERSION}",
        "output": str(output), "provider_usage": None, "started_at_utc": now(),
    }
    recorder = None
    previous_tempdir = tempfile.tempdir
    try:
        sys.dont_write_bytecode = True
        if any(name.startswith("are.simulation") for name in sys.modules):
            raise GateError("Use a fresh CLI process; shared or previously imported ARE state is forbidden")
        require_common_runtime()
        if config.relay_secret_file is None:
            raise GateError("Provide the common runtime's owner-private HMAC relay secret")
        distribution = importlib.metadata.distribution("meta-agents-research-environments")
        if distribution.version != ARE_VERSION:
            raise GateError("Use pinned meta-agents-research-environments==1.1.0")
        native_source = Path(distribution.locate_file("are/simulation"))
        executable = resolve_codex(config.codex_bin)
        version = subprocess.check_output(
            [str(executable), "--version"], text=True, timeout=30, env={"PATH": os.defpath}
        ).strip()
        if version != f"codex-cli {CODEX_VERSION}":
            raise GateError("The actual Codex executable is not version 0.142.5")
        admission = verify_freeze(config, executable, native_source)
        write_json(output / "ADMISSION.json", {
            key: value for key, value in admission.items() if key != "residual"
        })
        for name in admission["source_files"]:
            target = output / "private/source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / name, target)
        shutil.copyfile(config.freeze_manifest, output / "private/FREEZE.json")
        shutil.copyfile(config.partition_manifest, output / "private/PARTITION.json")
        shutil.copyfile(
            admission["native_protocol_receipt"]["path"],
            output / "private/native-protocol-receipt.json",
        )
        revision = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True, capture_output=True,
        )
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *SOURCE_FILES],
            text=True, capture_output=True,
        )
        source_revision = revision.stdout.strip() if revision.returncode == 0 else (
            read_json(config.freeze_manifest).get("source_revision")
        )
        if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
            raise GateError("A copied checkout must retain its original frozen source revision")
        write_json(output / "PROVENANCE.json", {
            "hostname": socket.gethostname(), "python": sys.version,
            "source_revision": source_revision,
            "source_revision_origin": "git" if revision.returncode == 0 else "controller_freeze",
            "source_status": status.stdout if status.returncode == 0 else "copied checkout",
            "packages": {dist.metadata["Name"]: dist.version
                         for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
            "codex_version": version, "codex_payload": str(executable),
            "codex_sha256": file_sha256(executable),
            "are_version": ARE_VERSION, "split_seed": SPLIT_SEED,
            "source_files": admission["source_files"],
            "bridge_source_files": admission["bridge_source_files"],
            "native_protocol_receipt": admission["native_protocol_receipt"],
            "configuration_sha256": canonical_sha256(config.public_config()),
        })
        inventory = {}
        for name, command in (
            ("gpus", ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                      "--format=csv"]),
            ("gpu_processes", ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                              "--format=csv"]),
            ("memory", ["free", "-h"]), ("load", ["uptime"]), ("leases", ["lslocks", "--json"]),
        ):
            probe = subprocess.run(command, text=True, capture_output=True, timeout=30)
            inventory[name] = {"returncode": probe.returncode, "stdout": probe.stdout,
                               "stderr": probe.stderr}
        write_json(output / "RESOURCE_SELECTION.json", {
            "selected_gpu": None, "CUDA_VISIBLE_DEVICES": "", "inventory": inventory,
            "reason": "CPU native ARE with hosted DeepSeek; no GPU lease or unrelated job changes",
        })
        storage = output / "private/native-storage"
        storage.mkdir()
        tempfile.tempdir = str(storage)
        os.environ.update(
            TMPDIR=str(storage), FS_PATH=str(storage / "fs-states"),
            DEMO_FS_PATH=str(config.filesystem.resolve()), CUDA_VISIBLE_DEVICES="",
            PYTHON_DOTENV_DISABLED="1",
        )
        recorder = ProviderRecorder(output / "private/provider", config)
        logging.basicConfig(filename=output / "private/native.log", level=logging.INFO, force=True)
        with (output / "private/native-stdout.log").open("w") as native_stdout:
            with redirect_stdout(native_stdout):
                native = run_native(config, executable, admission, recorder)
        verify_freeze(config, executable, native_source)
        usage = recorder.snapshot()
        if not all(value["complete"] for value in usage.values()):
            raise GateError("Provider accounting is incomplete; native verdict is not admitted")
        actor = read_json(output / "private/codex-process.json")
        result.update(
            status="COMPLETED", native_score=int(native["success"]),
            native_success=native["success"], split=admission["split"],
            scenario_id=admission["scenario_id"],
            outputs={
                "native_result": str(output / "private/native-result.json"),
                "native_traces": str(output / "private/native-traces"),
                "actor_events": actor["events_path"],
                "public_observations": str(output / "private/public-observations.jsonl"),
                "actor_run": actor["run_directory"],
                "actor_workspace": actor["workspace"],
                "provider_receipts": str(output / "private/provider"),
            },
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["blocker"] = recorder.redact(str(error)) if recorder else str(error)
    finally:
        tempfile.tempdir = previous_tempdir
        result["provider_usage"] = recorder.snapshot() if recorder else {
            "actor": {"requests": 0, "status": "not_started"},
            "judge": {"requests": 0, "status": "not_started"},
        }
        result["finished_at_utc"] = now()
        write_json(output / "RESULT.json", result)
    return result
