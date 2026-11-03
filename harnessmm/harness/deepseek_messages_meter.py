"""Durable, byte-preserving transport for DeepSeek's Anthropic Messages endpoint.

Use ``with DeepSeekMessagesMeter(root, upstream_key) as meter`` and give the SDK
``base_url=meter.url, api_key=meter.client_key``. The key is supplied in memory;
this module does not read credentials. Only POST /v1/messages (optionally with
?beta=true) is admitted, to https://api.deepseek.com/anthropic/v1/messages.

Call ``seal_admission()`` at native task stop, then close the SDK, then call
``close()`` / ``drain()`` outside the benchmark deadline. Sealing does no I/O and
does not wait for or close existing requests. close/drain also seal admission,
then wait for admitted HTTP exchanges, including disconnected SDK responses.
They never retry.
The timeout is an ordinary socket inactivity timeout, not a token budget or
benchmark deadline. Call close outside the benchmark's action/deadline loop.

Each exchange has an atomic RECEIPT.json and owner-only request.body and
response.body files. Hashes cover HTTP entity bytes, without transfer framing.
Bodies are unchanged, except that unsafe credential-bearing bodies and
unsupported content encodings are withheld with an explicit failure receipt.
Only identity content encoding and Content-Length request framing are supported.
Receipts are fsynced before dispatch and response bytes are fsynced as they arrive.
Process/host death cannot keep a socket alive; unfinished receipts remain unknown.

snapshot() (also read_meter_snapshot(root), without a credential) counts each
physical send attempt once. ``usage`` is exact only when ``complete`` is true.
``known_usage`` aggregates only billing-complete requests, in native SDK camelCase
units; partial counters stay in individual receipts. Missing counters are never
filled with zeros. inputTokens, cacheReadTokens and cacheWriteTokens are disjoint;
outputTokens already includes reasoningTokens.

``recover_meter_snapshot(root, sidecar=path)`` is offline-only: it verifies saved
response hashes and can recover historical INVALID_NATIVE_USAGE receipts caused
by conserved prompt/cache reclassification. It never edits source receipts or
bodies. The optional sidecar must be new and outside the provider root. While a
meter is live, its report covers only the captured request population, not future
requests. In-flight, corrupt, failed and unsupported exchanges remain unknown.
"""

from __future__ import annotations

import codecs
from datetime import datetime, timezone
import hashlib
import hmac
from http.client import HTTPException, HTTPSConnection, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import socket
import ssl
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid


_UPSTREAM_HOST = "api.deepseek.com"
_UPSTREAM_PATH = "/anthropic/v1/messages"
_UPSTREAM_URL = "https://" + _UPSTREAM_HOST + _UPSTREAM_PATH
_MAX_SAFE_INTEGER = 2**53 - 1
_COUNTERS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_read_input_tokens": "cacheReadTokens",
    "cache_creation_input_tokens": "cacheWriteTokens",
    "reasoning_tokens": "reasoningTokens",
    "total_tokens": "totalTokens",
}
_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
_AUTH_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}
_VENDOR_IDS = {"request-id", "x-request-id", "x-ds-request-id", "x-ds-trace-id"}
_PROMPT_COUNTERS = ("inputTokens", "cacheReadTokens", "cacheWriteTokens")
_RECOVERY_POLICY = "deepseek-messages-conserved-prompt-cache-snapshot-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_file(path: Path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with _private_file(pending) as output:
            output.write((json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, path)
        _sync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def _normalized_usage(values: dict[str, int]) -> dict[str, int] | None:
    if any(
        key not in _COUNTERS.values()
        or type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER
        for key, value in values.items()
    ):
        return None
    if not all(key in values for key in ("inputTokens", "outputTokens")):
        return None
    if values.get("reasoningTokens", 0) > values["outputTokens"]:
        return None
    prompt = values["inputTokens"] + sum(
        values[key] for key in ("cacheReadTokens", "cacheWriteTokens") if key in values
    )
    both_cache = all(key in values for key in ("cacheReadTokens", "cacheWriteTokens"))
    total = values.get("totalTokens")
    if total is None:
        if not both_cache:
            return None
        total = prompt + values["outputTokens"]
    exact_prompt = total - values["outputTokens"]
    if (
        prompt > _MAX_SAFE_INTEGER or total > _MAX_SAFE_INTEGER
        or exact_prompt < prompt or (both_cache and exact_prompt != prompt)
    ):
        return None
    return {**values, "totalTokens": total}


def _aggregate(samples: list[dict[str, int]]) -> dict[str, int] | None:
    if not samples:
        return None
    result = {
        key: sum(sample[key] for sample in samples)
        for key in _COUNTERS.values() if all(key in sample for sample in samples)
    }
    return result if all(value <= _MAX_SAFE_INTEGER for value in result.values()) else None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member")
        result[key] = value
    return result


class _Usage:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.invalid = False
        self.prompt_reclassifications: list[dict[str, Any]] = []

    def merge(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            self.invalid = True
            return
        raw = dict(raw)
        details = raw.get("output_tokens_details")
        if isinstance(details, dict) and "reasoning_tokens" in details:
            if "reasoning_tokens" in raw and raw["reasoning_tokens"] != details["reasoning_tokens"]:
                self.invalid = True
            raw["reasoning_tokens"] = details["reasoning_tokens"]
        supplied = {
            target: raw[source] for source, target in _COUNTERS.items() if source in raw
        }
        complete_prompt = all(
            key in self.values and type(supplied.get(key)) is int
            and 0 <= supplied[key] <= _MAX_SAFE_INTEGER
            for key in _PROMPT_COUNTERS
        )
        reclassified = (
            complete_prompt
            and any(supplied[key] < self.values[key] for key in _PROMPT_COUNTERS)
            and sum(supplied[key] for key in _PROMPT_COUNTERS)
            == sum(self.values[key] for key in _PROMPT_COUNTERS)
        )
        if reclassified:
            self.prompt_reclassifications.append({
                "previous": {key: self.values[key] for key in _PROMPT_COUNTERS},
                "replacement": {key: supplied[key] for key in _PROMPT_COUNTERS},
                "prompt_tokens": sum(supplied[key] for key in _PROMPT_COUNTERS),
            })
        for source, target in _COUNTERS.items():
            if source not in raw:
                continue
            value = raw[source]
            if (
                type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER
                or (
                    value < self.values.get(target, 0)
                    and not (reclassified and target in _PROMPT_COUNTERS)
                )
            ):
                self.invalid = True
            else:
                # Final cache classification may lower uncached input while
                # preserving total prompt tokens. Replace; never add or take max.
                self.values[target] = value

    def normalized(self) -> dict[str, int] | None:
        return None if self.invalid else _normalized_usage(self.values)


class _NativeResponse:
    """Parse a separate copy; never reserialize the forwarded response."""

    def __init__(self, content_type: str) -> None:
        self.sse = content_type.split(";", 1)[0].strip().lower() == "text/event-stream"
        self.usage = _Usage()
        self.model: str | None = None
        self.message_id: str | None = None
        self.terminal = False
        self.error: str | None = None
        self._started = False
        self._final_output = False
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self._line = ""
        self._skip_lf = False
        self._data: list[str] = []
        self._event = ""
        self._json = bytearray()
        self.usage_events: list[dict[str, Any]] = []

    def _merge_usage(self, kind: str, raw: Any) -> None:
        self.usage.merge(raw)
        self.usage_events.append({
            "event": kind,
            "counters": {
                key: raw[key] for key in _COUNTERS
                if isinstance(raw, dict) and type(raw.get(key)) is int
            },
        })

    def _identity(self, message: dict[str, Any]) -> None:
        for field, attribute in (("model", "model"), ("id", "message_id")):
            if isinstance(message.get(field), str):
                setattr(self, attribute, message[field])

    def _frame(self) -> None:
        if not self._data:
            self._event = ""
            return
        raw, event = "\n".join(self._data), self._event
        self._data, self._event = [], ""
        try:
            value = json.loads(raw, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            self.error = "INVALID_SSE_JSON"
            return
        if not isinstance(value, dict) or not isinstance(value.get("type"), str):
            self.error = "INVALID_SSE_EVENT"
            return
        kind = value["type"]
        if event and event != kind:
            self.error = "CONFLICTING_SSE_EVENT_TYPE"
            return
        if kind == "error":
            self.error = "NATIVE_PROVIDER_ERROR"
            return
        if kind == "ping":
            return
        if self.terminal:
            self.error = "EVENT_AFTER_MESSAGE_STOP"
            return
        if kind == "message_start":
            message = value.get("message")
            if self._started or not isinstance(message, dict) or message.get("type") != "message":
                self.error = "INVALID_MESSAGE_START"
                return
            self._started = True
            self._identity(message)
            self._merge_usage(kind, message.get("usage"))
        elif kind in {"message_delta", "message_stop"}:
            if not self._started:
                self.error = "MESSAGE_WITHOUT_START"
                return
            if "usage" in value:
                self._merge_usage(kind, value["usage"])
                if isinstance(value["usage"], dict) and "output_tokens" in value["usage"]:
                    self._final_output = True
            if kind == "message_stop":
                self.terminal = True

    def _accept_line(self) -> None:
        line, self._line = self._line, ""
        if not line:
            self._frame()
        elif not line.startswith(":"):
            name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if name == "data":
                self._data.append(value)
            elif name == "event":
                self._event = value

    def feed(self, chunk: bytes) -> None:
        if self.error:
            return
        if not self.sse:
            self._json.extend(chunk)
            return
        try:
            text = self._decoder.decode(chunk)
        except UnicodeError:
            self.error = "INVALID_RESPONSE_UTF8"
            return
        for char in text:
            if self._skip_lf:
                self._skip_lf = False
                if char == "\n":
                    continue
            if char in "\r\n":
                self._accept_line()
                self._skip_lf = char == "\r"
            else:
                self._line += char
            if self.error:
                break

    def finish(self) -> None:
        if self.error:
            return
        if self.sse:
            try:
                self._decoder.decode(b"", final=True)
            except UnicodeError:
                self.error = "INVALID_RESPONSE_UTF8"
            if self._line or self._data:
                self.error = self.error or "UNTERMINATED_SSE_EVENT"
            if not self.terminal or not self._final_output:
                self.error = self.error or "INCOMPLETE_NATIVE_RESPONSE"
            return
        try:
            value = json.loads(self._json, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError):
            self.error = "INVALID_RESPONSE_JSON"
            return
        if not isinstance(value, dict):
            self.error = "INVALID_NATIVE_RESPONSE"
        elif value.get("type") == "error" or value.get("error") is not None:
            self.error = "NATIVE_PROVIDER_ERROR"
        elif value.get("type") != "message":
            self.error = "INVALID_NATIVE_RESPONSE"
        else:
            self._identity(value)
            self.usage.merge(value.get("usage"))
            self.terminal = isinstance(value.get("stop_reason"), str) and bool(value["stop_reason"])
            if not self.terminal:
                self.error = "INCOMPLETE_NATIVE_RESPONSE"


class _SecretGuard:
    """Hold a short tail so an echoed key cannot leak across read boundaries."""

    def __init__(self, keys: tuple[bytes, ...]) -> None:
        self.keys = keys
        self.tail = b""
        self.blocked = False
        self.hold = max(map(len, keys)) - 1

    def feed(self, raw: bytes, *, final: bool = False) -> bytes:
        if self.blocked:
            return b""
        raw = self.tail + raw
        matches = [raw.find(key) for key in self.keys if key in raw]
        if matches:
            self.blocked = True
            self.tail = b""
            return raw[:min(matches)]
        keep = 0 if final else min(self.hold, len(raw))
        self.tail = raw[-keep:] if keep else b""
        return raw[:-keep] if keep else raw


def read_meter_snapshot(root: Path) -> dict[str, Any]:
    """Read durable receipts, including unknown attempts left by a dead process."""
    root = Path(root)
    recorder_errors = []
    try:
        manifest = json.loads((root / "INSTANCE.json").read_bytes())
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError
        recorder_errors.extend(manifest.get("recorder_errors", []))
    except (OSError, ValueError, TypeError):
        recorder_errors.append("MISSING_OR_INVALID_MANIFEST")
    records = []
    paths = sorted((root / "requests").glob("*/RECEIPT.json"))
    directories = list((root / "requests").glob("*"))
    unreadable = max(0, len(directories) - len(paths))
    for path in paths:
        try:
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError
            records.append(value)
        except (OSError, ValueError):
            unreadable += 1
    attempted = [record for record in records if record.get("upstream_attempted")]
    measured = [
        record for record in attempted
        if record.get("billing_complete") and isinstance(record.get("usage"), dict)
        and _normalized_usage(record["usage"]) == record["usage"]
        and record.get("transport_complete") and record.get("terminal_received")
        and record.get("response_body_preserved")
        and record.get("finished_at") is not None and not record.get("failure_classification")
    ]
    known = _aggregate([record["usage"] for record in measured])
    pending = sum(record.get("finished_at") is None for record in records)
    unknown = len(attempted) - len(measured) + unreadable
    failed = sum(bool(record.get("failure_classification")) for record in records)
    complete = not (
        unknown or pending or unreadable or failed or recorder_errors
        or (measured and known is None)
    )
    return {
        "complete": complete,
        "usage": known if complete else None,
        "known_usage": known,
        "request_count": len(attempted) + unreadable,
        "unknown_request_count": unknown,
        "pending_request_count": pending,
        "rejected_request_count": sum(
            not record.get("upstream_attempted") and record.get("finished_at") is not None
            for record in records
        ),
        "failed_request_count": failed,
        "unreadable_receipt_count": unreadable,
        "recorder_errors": recorder_errors,
        "receipt_paths": [str(path) for path in paths],
        "response_body_paths": [
            record["response_body_path"] for record in records if "response_body_path" in record
        ],
        "provider_models": sorted({record["model"] for record in records if record.get("model")}),
    }


def _recover_receipt(directory: Path, instance_id: str | None) -> dict[str, Any]:
    """Return usage metadata only, never a response, prompt, header or credential."""
    receipt_path = directory / "RECEIPT.json"
    body_path = directory / "response.body"
    result: dict[str, Any] = {
        "request_id": directory.name, "receipt_path": str(receipt_path),
        "response_body_path": str(body_path), "status": "UNKNOWN",
        "reason": "UNREADABLE_RECEIPT", "usage": None, "upstream_attempted": True,
        "pending": False, "original_billing_complete": False,
    }
    try:
        if directory.is_symlink() or receipt_path.is_symlink():
            raise ValueError
        source = receipt_path.read_bytes()
        record = json.loads(source, object_pairs_hook=_unique_object)
        if not isinstance(record, dict):
            raise ValueError
    except (OSError, ValueError, RecursionError):
        return result
    result["receipt_sha256"] = hashlib.sha256(source).hexdigest()
    attempted = record.get("upstream_attempted")
    result["upstream_attempted"] = attempted if type(attempted) is bool else True
    result["pending"] = record.get("finished_at") is None

    def audit() -> None:
        if (
            type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("request_id") != directory.name or not instance_id
            or record.get("instance_id") != instance_id
        ):
            result.update(reason="RECEIPT_IDENTITY_MISMATCH", upstream_attempted=True)
            return
        if type(attempted) is not bool:
            result["reason"] = "INVALID_ATTEMPT_STATE"
            return
        if result["pending"]:
            result["reason"] = "IN_FLIGHT"
            return
        if not result["upstream_attempted"]:
            result.update(status="NOT_DISPATCHED", reason="NO_UPSTREAM_ATTEMPT")
            return
        complete = all(record.get(key) is True for key in (
            "upstream_request_sent", "transport_complete", "terminal_received",
            "response_body_preserved",
        ))
        status = record.get("http_status")
        if (
            not complete or type(status) is not int or not 200 <= status < 300
            or not isinstance(record.get("finished_at"), str) or not record["finished_at"]
            or record.get("body_rewritten") is not False
            or record.get("upstream_url") != _UPSTREAM_URL or record.get("error_type")
        ):
            result["reason"] = "SOURCE_NOT_SUCCESSFULLY_COMPLETED"
            return
        usage = record.get("usage")
        if (
            record.get("billing_complete") is True and not record.get("failure_classification")
            and isinstance(usage, dict) and _normalized_usage(usage) == usage
        ):
            result.update(
                status="MEASURED", reason="UNCHANGED_METER_RECEIPT", usage=dict(usage),
                original_billing_complete=True,
            )
            return
        if (
            record.get("failure_classification") != "INVALID_NATIVE_USAGE"
            or record.get("billing_complete") is not False or usage is not None
        ):
            result["reason"] = "FAILURE_NOT_ALLOWLISTED"
            return
        if (
            record.get("content_encoding") != "identity"
            or not isinstance(record.get("content_type"), str)
            or record["content_type"].split(";", 1)[0].strip().lower() != "text/event-stream"
        ):
            result["reason"] = "UNSUPPORTED_RESPONSE_ENCODING_OR_TYPE"
            return
        try:
            if body_path.is_symlink():
                raise ValueError
            before = body_path.stat()
            raw = body_path.read_bytes()
        except (OSError, ValueError):
            result["reason"] = "UNREADABLE_RESPONSE_BODY"
            return
        digest = hashlib.sha256(raw).hexdigest()
        result["response_sha256"] = digest
        result["response_bytes"] = len(raw)
        if (
            digest != record.get("response_sha256")
            or digest != record.get("recorded_response_sha256")
            or any(
                type(record.get(key)) is not int or record[key] != len(raw)
                for key in ("response_bytes", "recorded_response_bytes")
            )
            or (
                record.get("expected_response_bytes") is not None
                and (
                    type(record["expected_response_bytes"]) is not int
                    or record["expected_response_bytes"] != len(raw)
                )
            )
        ):
            result["reason"] = "RESPONSE_HASH_OR_LENGTH_MISMATCH"
            return
        parser = _NativeResponse(record["content_type"])
        for offset in range(0, len(raw), 65536):
            parser.feed(raw[offset:offset + 65536])
        parser.finish()
        try:
            after = body_path.stat()
            unchanged = (before.st_ino, before.st_size, before.st_mtime_ns) == (
                after.st_ino, after.st_size, after.st_mtime_ns,
            )
        except OSError:
            unchanged = False
        if not unchanged:
            result["reason"] = "RESPONSE_CHANGED_DURING_AUDIT"
            return
        normalized = parser.usage.normalized()
        if parser.error or not parser.terminal or normalized is None:
            result["reason"] = "NATIVE_PROTOCOL_OR_USAGE_UNRESOLVED"
            return
        if not parser.usage.prompt_reclassifications:
            result["reason"] = "NO_CONSERVED_PROMPT_RECLASSIFICATION"
            return
        result.update(
            status="RECOVERED", reason="CONSERVED_PROMPT_CACHE_RECLASSIFICATION",
            usage=normalized, usage_events=parser.usage_events,
            prompt_reclassifications=parser.usage.prompt_reclassifications,
        )

    audit()
    try:
        if receipt_path.read_bytes() != source:
            raise ValueError
    except (OSError, ValueError):
        result.update(
            status="UNKNOWN", reason="RECEIPT_CHANGED_DURING_AUDIT", usage=None,
            original_billing_complete=False, upstream_attempted=True,
        )
    return result


def recover_meter_snapshot(root: Path, *, sidecar: Path | None = None) -> dict[str, Any]:
    """Recover only hash-verified, complete conserved prompt/cache snapshots.

    Anthropic's streaming SDK replaces supplied message_delta input/cache counts;
    output is cumulative and includes reasoning. See its accumulate_event:
    https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py

    This narrower recovery requires a full prompt-bucket snapshot with unchanged
    prompt total before allowing any bucket to decrease. Output/reasoning
    regressions, missing counters, errors and absent terminals remain unknown.
    ``known_usage`` replaces, rather than adds to, the original meter aggregate:
    every captured physical request contributes at most once. ``recovered_usage``
    is only the newly recovered subset. Original measured receipts are retained;
    only recovery candidates have their saved response bodies reparsed.

    No network, SDK, credential, benchmark or evaluator is used. Sources are
    read-only, live bodies are skipped, and changed receipts stay unknown.
    The optional external sidecar is atomic, owner-only and never overwritten.
    """
    root = Path(root).resolve()
    destination = Path(sidecar).absolute() if sidecar is not None else None
    if destination is not None:
        resolved = destination.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError("Recovery sidecar must be outside the source provider root")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Recovery sidecar already exists")
    started_at = _now()
    errors = []
    manifest_path = root / "INSTANCE.json"
    manifest_sha256 = None
    instance_id = None
    state = None
    try:
        if manifest_path.is_symlink():
            raise ValueError
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw, object_pairs_hook=_unique_object)
        if (
            not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("instance_id"), str) or not manifest["instance_id"]
            or manifest.get("upstream_url") != _UPSTREAM_URL
        ):
            raise ValueError
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        instance_id = manifest["instance_id"]
        state = manifest.get("state") if manifest.get("state") in {"OPEN", "CLOSED"} else None
        if manifest.get("recorder_errors"):
            errors.append("SOURCE_RECORDER_ERRORS")
    except (OSError, ValueError, TypeError, RecursionError):
        errors.append("MISSING_OR_INVALID_MANIFEST")
    try:
        directories = sorted((root / "requests").iterdir())
    except OSError:
        directories = []
        errors.append("UNREADABLE_REQUEST_DIRECTORY")
    entries = [_recover_receipt(directory, instance_id) for directory in directories]
    measured = [entry for entry in entries if entry["status"] == "MEASURED"]
    recovered = [entry for entry in entries if entry["status"] == "RECOVERED"]
    known = _aggregate([entry["usage"] for entry in measured + recovered])
    recovered_usage = _aggregate([entry["usage"] for entry in recovered])
    attempts = sum(entry["upstream_attempted"] for entry in entries)
    unknown = sum(
        entry["upstream_attempted"] and entry["status"] == "UNKNOWN" for entry in entries
    )
    pending = sum(entry["pending"] for entry in entries)
    unaccounted = sum(entry["status"] not in {"MEASURED", "RECOVERED"} for entry in entries)
    if (measured or recovered) and known is None:
        errors.append("AGGREGATE_USAGE_OVERFLOW")
    complete = not (errors or unaccounted or pending)
    report: dict[str, Any] = {
        "schema_version": 1, "recovery_policy": _RECOVERY_POLICY,
        "source_root": str(root), "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest_sha256, "source_meter_state": state,
        "scope": "captured_request_population_only",
        "started_at": started_at, "finished_at": _now(),
        "complete": complete, "usage": known if complete else None, "known_usage": known,
        "original_known_usage": _aggregate([entry["usage"] for entry in measured]),
        "recovered_usage": recovered_usage,
        "request_count": attempts, "unknown_request_count": unknown,
        "pending_request_count": pending, "recovered_request_count": len(recovered),
        "original_billing_complete_request_count": len(measured),
        "original_unknown_request_count": attempts - len(measured),
        "rejected_request_count": sum(entry["status"] == "NOT_DISPATCHED" for entry in entries),
        "recorder_errors": errors, "entries": entries,
    }
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        pending_path = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".pending")
        try:
            with _private_file(pending_path) as output:
                output.write((json.dumps(report, ensure_ascii=True, indent=2) + "\n").encode())
                output.flush()
                os.fsync(output.fileno())
            os.link(pending_path, destination)  # Publish atomically without replacing any file.
            _sync_directory(destination.parent)
        finally:
            pending_path.unlink(missing_ok=True)
    return report


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = False

    def handle_error(self, _request, _client_address) -> None:
        # Never print exception values or HTTP headers through the server's logger.
        self.meter._record_error("HANDLER_EXCEPTION")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.meter.client_timeout)

    def log_message(self, *_args) -> None:
        pass

    def send_error(self, code, message=None, explain=None) -> None:
        self.server.meter._local_error(self, code, "UNSUPPORTED_HTTP_REQUEST")

    def do_POST(self) -> None:
        self.server.meter._handle(self)


class DeepSeekMessagesMeter:
    """One fresh receipt root, loopback listener, and ephemeral SDK credential.

    ``start``/``open`` are idempotent until close and never reopen sealed
    admission. A closed instance cannot restart. A root must be new or empty,
    so older billing evidence is never overwritten. close/drain are idempotent
    and return the final snapshot.
    """

    def __init__(
        self, root: Path, upstream_key: str, *, timeout: float = 1800,
        client_timeout: float = 30,
    ) -> None:
        if (
            not isinstance(upstream_key, str) or not upstream_key
            or any(not 33 <= ord(char) <= 126 for char in upstream_key)
        ):
            raise ValueError("upstream_key must be a nonempty printable ASCII credential")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            for value in (timeout, client_timeout)
        ):
            raise ValueError("HTTP timeouts must be finite positive numbers")
        self.root = Path(root).absolute()
        self.timeout = timeout
        self.client_timeout = client_timeout
        self._upstream_key = upstream_key
        self._client_key = secrets.token_urlsafe(32)
        self._keys = (upstream_key.encode(), self._client_key.encode())
        self._lock = threading.RLock()
        self._admission_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._accepting = False
        self._sealed = False
        self._closed = False
        self._errors: list[str] = []
        self._manifest: dict[str, Any] = {}

    @property
    def client_key(self) -> str:
        return self._client_key

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Start the meter before requesting its URL")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _safe(self, value: str) -> str:
        for key in (self._upstream_key, self._client_key):
            value = value.replace(key, "[REDACTED]")
        return value

    def _record_error(self, kind: str) -> None:
        with self._lock:
            self._errors.append(kind)

    def start(self) -> DeepSeekMessagesMeter:
        with self._lock:
            if self._closed:
                raise RuntimeError("A closed meter cannot restart")
            if self._server is not None:
                return self
            with self._admission_lock:
                if self._sealed:
                    raise RuntimeError("A sealed meter cannot start")
            if self.root.is_symlink():
                raise ValueError("Meter root must not be a symlink")
            self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
            if any(self.root.iterdir()):
                raise FileExistsError("Meter root must be new or empty")
            self.root.chmod(0o700)
            (self.root / "requests").mkdir(mode=0o700)
            self._manifest = {
                "schema_version": 1, "instance_id": uuid.uuid4().hex,
                "created_at": _now(), "upstream_url": _UPSTREAM_URL,
                "bind": "127.0.0.1", "transport": "native_anthropic_messages",
                "http_timeout_seconds": self.timeout, "automatic_retries": 0,
                "content_encoding": "identity", "state": "OPEN",
            }
            _atomic_json(self.root / "INSTANCE.json", self._manifest)
            server = _Server(("127.0.0.1", 0), _Handler)
            server.meter = self
            self._server = server
            with self._admission_lock:
                self._accepting = not self._sealed
            self._thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.05},
                name="deepseek-messages-meter", daemon=False,
            )
            self._thread.start()
            return self

    def open(self) -> DeepSeekMessagesMeter:
        return self.start()

    def __enter__(self) -> DeepSeekMessagesMeter:
        return self.start()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def snapshot(self) -> dict[str, Any]:
        result = read_meter_snapshot(self.root)
        with self._lock:
            result["recorder_errors"] = sorted(set(result["recorder_errors"]) | set(self._errors))
        if result["recorder_errors"]:
            result.update(complete=False, usage=None)
        return result

    def seal_admission(self) -> None:
        """Permanently reject new upstream admissions without waiting for I/O.

        Already-admitted exchanges continue unchanged. A request still reading
        its client body or preparing its durable dispatch receipt is not yet
        admitted and will receive 503 without contacting the provider.
        """
        with self._admission_lock:
            self._sealed = True
            self._accepting = False

    def close(self) -> dict[str, Any]:
        self.seal_admission()
        with self._close_lock:
            with self._lock:
                self._closed = True
            if self._server is not None and self._thread is not None:
                self._server.shutdown()
                self._server.server_close()  # Joins non-daemon handlers, not just the accept loop.
                self._thread.join()
                self._thread = None
                self._manifest.update(
                    state="CLOSED", closed_at=_now(), recorder_errors=list(self._errors)
                )
                try:
                    _atomic_json(self.root / "INSTANCE.json", self._manifest)
                except OSError:
                    self._record_error("MANIFEST_WRITE_FAILED")
            return self.snapshot()

    def drain(self) -> dict[str, Any]:
        return self.close()

    @staticmethod
    def _disconnected(record: dict[str, Any]) -> None:
        if not record["client_disconnected"]:
            record.update(client_disconnected=True, client_disconnected_at=_now())

    def _headers(
        self, handler: _Handler, status: int, headers: list[tuple[str, str]],
        record: dict[str, Any],
    ) -> None:
        record["downstream_status"] = status
        handler.close_connection = True
        try:
            handler.send_response_only(status)
            for name, value in headers:
                handler.send_header(name, value)
            handler.send_header("Connection", "close")
            handler.end_headers()
        except OSError:
            self._disconnected(record)

    def _send(self, handler: _Handler, data: bytes, record: dict[str, Any]) -> None:
        if data and not record["client_disconnected"]:
            try:
                handler.wfile.write(data)
                handler.wfile.flush()
            except OSError:
                self._disconnected(record)

    def _local_error(
        self, handler: _Handler, status: int, kind: str,
        record: dict[str, Any] | None = None,
    ) -> None:
        record = record if record is not None else {"client_disconnected": False}
        body = json.dumps({"type": "error", "error": {"type": "meter_error", "message": kind}}).encode()
        self._headers(handler, status, [
            ("Content-Type", "application/json"), ("Content-Length", str(len(body))),
        ], record)
        self._send(handler, body, record)

    def _filtered_headers(self, headers) -> list[tuple[str, str]]:
        values = list(headers)
        excluded = _HOP_HEADERS | _AUTH_HEADERS | {"host", "content-length"}
        for name, value in values:
            if name.lower() == "connection":
                excluded |= {part.strip().lower() for part in value.split(",")}
        return [(name, value) for name, value in values if name.lower() not in excluded]

    def _authorized(self, handler: _Handler) -> bool:
        values = handler.headers.get_all("x-api-key", [])
        bearer = handler.headers.get_all("authorization", [])
        if len(values) > 1 or len(bearer) > 1 or not (values or bearer):
            return False
        checks = [(value, self._client_key) for value in values]
        checks += [(value, "Bearer " + self._client_key) for value in bearer]
        return all(
            hmac.compare_digest(actual.encode("utf-8"), expected.encode("utf-8"))
            for actual, expected in checks
        )

    def _handle(self, handler: _Handler) -> None:
        if not self._authorized(handler):
            self._local_error(handler, 403, "INVALID_METER_AUTHENTICATION")
            return
        try:
            endpoint = urlsplit(handler.path)
            valid_path = (
                not endpoint.scheme and not endpoint.netloc and not endpoint.fragment
                and endpoint.path == "/v1/messages" and endpoint.query in ("", "beta=true")
            )
        except ValueError:
            valid_path = False
        if not valid_path:
            self._local_error(handler, 404, "ONLY_NATIVE_MESSAGES_ARE_ADMITTED")
            return
        with self._admission_lock:
            accepting = self._accepting
        if not accepting:
            self._local_error(handler, 503, "METER_ADMISSION_SEALED")
            return
        request_id = uuid.uuid4().hex
        directory = self.root / "requests" / request_id
        directory.mkdir(mode=0o700)
        _sync_directory(directory.parent)
        receipt_path = directory / "RECEIPT.json"
        headers = self._filtered_headers(handler.headers.raw_items())
        record: dict[str, Any] = {
            "schema_version": 1, "request_id": request_id,
            "instance_id": self._manifest["instance_id"], "started_at": _now(),
            "upstream_url": _UPSTREAM_URL, "body_rewritten": False,
            "upstream_path": _UPSTREAM_PATH + ("?" + endpoint.query if endpoint.query else ""),
            "upstream_attempted": False, "upstream_request_sent": False,
            "client_disconnected": False, "http_status": None,
            "transport_complete": False, "terminal_received": False,
            "billing_complete": False, "usage_status": "UNKNOWN", "usage": None,
            "known_usage": {}, "request_headers": [
                [self._safe(name), self._safe(value)] for name, value in headers
            ],
            "request_body_path": str(directory / "request.body"),
            "response_body_path": str(directory / "response.body"),
            "request_body_preserved": False, "response_body_preserved": False,
            "finished_at": None,
        }
        started = time.monotonic()
        response_digest = hashlib.sha256()
        stored_digest = hashlib.sha256()
        record.update(response_bytes=0, recorded_response_bytes=0)
        connection = None
        output = None
        parser = None
        guard = _SecretGuard(self._keys)
        downstream_started = False
        recording = True

        def persist() -> None:
            _atomic_json(receipt_path, record)

        def persist_dispatched() -> None:
            try:
                persist()
            except OSError:
                record["failure_classification"] = "RECEIPT_WRITE_FAILED"
                self._record_error("RECEIPT_WRITE_FAILED")

        def capture(chunk: bytes, *, final: bool = False) -> None:
            nonlocal recording
            response_digest.update(chunk)
            record["response_bytes"] += len(chunk)
            if parser is None:
                return
            parser.feed(chunk)
            safe = guard.feed(chunk, final=final)
            if safe and recording:
                try:
                    output.write(safe)
                    output.flush()
                    os.fsync(output.fileno())
                    stored_digest.update(safe)
                    record["recorded_response_bytes"] += len(safe)
                except OSError:
                    recording = False
                    record["failure_classification"] = "RESPONSE_RECORDING_FAILED"
                    self._record_error("RESPONSE_RECORDING_FAILED")
            self._send(handler, safe, record)
            if guard.blocked:
                record["failure_classification"] = "CREDENTIAL_IN_PROVIDER_BODY"
                if not record["client_disconnected"]:
                    try:
                        handler.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self._disconnected(record)

        try:
            persist()
            with _private_file(directory / "request.body") as request_output:
                output = _private_file(directory / "response.body")
                lengths = handler.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                    or handler.headers.get("Transfer-Encoding") is not None
                ):
                    record["failure_classification"] = "UNSUPPORTED_REQUEST_FRAMING"
                    self._local_error(handler, 411, record["failure_classification"], record)
                    return
                length = int(lengths[0])
                raw = handler.rfile.read(length)
                record.update(request_bytes=len(raw), request_sha256=hashlib.sha256(raw).hexdigest())
                encoding = ",".join(
                    handler.headers.get_all("Content-Encoding", ["identity"])
                ).strip().lower()
                if encoding != "identity":
                    record["failure_classification"] = "UNSUPPORTED_REQUEST_CONTENT_ENCODING"
                    self._local_error(handler, 415, record["failure_classification"], record)
                    return
                if any(key in raw for key in self._keys):
                    record["failure_classification"] = "CREDENTIAL_IN_REQUEST_BODY"
                    self._local_error(handler, 400, record["failure_classification"], record)
                    return
                request_output.write(raw)
                request_output.flush()
                os.fsync(request_output.fileno())
                record["request_body_preserved"] = True
                if len(raw) != length:
                    record["failure_classification"] = "INCOMPLETE_REQUEST_BODY"
                    self._local_error(handler, 400, record["failure_classification"], record)
                    return
            if any(self._safe(item) != item for pair in headers for item in pair):
                record["failure_classification"] = "CREDENTIAL_IN_REQUEST_HEADER"
                self._local_error(handler, 400, record["failure_classification"], record)
                return
            try:
                request_value = json.loads(raw)
                model = request_value.get("model") if isinstance(request_value, dict) else None
                if isinstance(model, str):
                    record["requested_model"] = self._safe(model)
            except (ValueError, UnicodeError, RecursionError):
                pass  # Invalid native requests still receive the provider's unmodified failure.
            with self._admission_lock:
                accepting = self._accepting
            if not accepting:
                record["failure_classification"] = "CANCELLED_BEFORE_DISPATCH"
                self._local_error(handler, 503, record["failure_classification"], record)
                return
            record.update(upstream_attempted=True, dispatched_at=_now())
            try:
                persist()
            except OSError:
                record.update(
                    upstream_attempted=False, failure_classification="DISPATCH_RECORDING_FAILED"
                )
                raise
            # Only this memory-only gate races with seal_admission; neither disk
            # persistence nor socket operations may hold the admission lock.
            with self._admission_lock:
                admitted = self._accepting
            if not admitted:
                record.update(
                    upstream_attempted=False, failure_classification="CANCELLED_BEFORE_DISPATCH"
                )
                self._local_error(handler, 503, record["failure_classification"], record)
                return
            connection = HTTPSConnection(
                _UPSTREAM_HOST, port=443, timeout=self.timeout, context=ssl.create_default_context()
            )
            connection.putrequest("POST", record["upstream_path"], skip_accept_encoding=True)
            for name, value in headers:
                connection.putheader(name, value)
            connection.putheader("Content-Length", str(len(raw)))
            connection.putheader("x-api-key", self._upstream_key)
            connection.putheader("Connection", "close")
            connection.endheaders(raw)
            record["upstream_request_sent"] = True
            persist_dispatched()
            response = connection.getresponse()
            record.update(http_status=response.status, response_headers_at=_now())
            upstream_headers = response.getheaders()
            filtered = self._filtered_headers(upstream_headers)
            record["response_headers"] = [
                [self._safe(name), self._safe(value)] for name, value in filtered
            ]
            record["vendor_request_ids"] = {
                name.lower(): self._safe(value)
                for name, value in upstream_headers if name.lower() in _VENDOR_IDS
            }
            content_type = response.getheader("Content-Type", "")
            encoding = response.getheader("Content-Encoding", "identity").strip().lower()
            expected_length = response.length
            record.update(
                content_type=self._safe(content_type), content_encoding=self._safe(encoding),
                expected_response_bytes=expected_length,
            )
            if any(self._safe(item) != item for pair in filtered for item in pair):
                record["failure_classification"] = "CREDENTIAL_IN_PROVIDER_HEADER"
            elif encoding != "identity":
                record["failure_classification"] = "UNSUPPORTED_RESPONSE_CONTENT_ENCODING"
            else:
                parser = _NativeResponse(content_type)
            if parser is None:
                # Unknown encodings cannot be checked for an encoded credential echo.
                record["response_body_omitted"] = record["failure_classification"]
                self._local_error(handler, 502, record["failure_classification"], record)
            else:
                if expected_length is not None:
                    filtered.append(("Content-Length", str(expected_length)))
                self._headers(handler, response.status, filtered, record)
            downstream_started = True
            persist_dispatched()
            while True:
                try:
                    chunk = response.read1(65536)
                except IncompleteRead as error:
                    capture(error.partial)
                    raise
                if not chunk:
                    break
                capture(chunk)
            if expected_length is not None and record["response_bytes"] != expected_length:
                raise IncompleteRead(b"")
            record["transport_complete"] = True
        except (OSError, HTTPException, ValueError) as error:
            record.update(
                failure_classification=record.get("failure_classification") or "TRANSPORT_ERROR",
                error_type=type(error).__name__,
            )
            if not downstream_started:
                self._local_error(handler, 502, "UPSTREAM_TRANSPORT_FAILED", record)
        finally:
            if connection is not None:
                connection.close()
            if parser is not None:
                capture(b"", final=True)
                parser.finish()
                record.update(
                    terminal_received=parser.terminal,
                    model=self._safe(parser.model) if parser.model is not None else None,
                    message_id=self._safe(parser.message_id) if parser.message_id is not None else None,
                    known_usage=parser.usage.normalized() or dict(parser.usage.values),
                )
                failure = record.get("failure_classification")
                if not failure:
                    if not 200 <= record["http_status"] < 300:
                        failure = "UPSTREAM_HTTP_ERROR"
                    elif parser.error:
                        failure = parser.error
                    elif parser.usage.invalid:
                        failure = "INVALID_NATIVE_USAGE"
                    elif parser.usage.normalized() is None:
                        failure = "INCOMPLETE_NATIVE_USAGE"
                if failure:
                    record["failure_classification"] = failure
                record["response_body_preserved"] = not guard.blocked and recording
                record["billing_complete"] = bool(
                    not failure and record["transport_complete"] and parser.terminal
                    and record["response_body_preserved"]
                )
                if record["billing_complete"]:
                    record.update(usage=parser.usage.normalized(), usage_status="MEASURED")
                elif record["known_usage"]:
                    record["usage_status"] = "PARTIAL"
            if output is not None:
                try:
                    output.close()
                except OSError:
                    self._record_error("RESPONSE_FILE_CLOSE_FAILED")
                    record.update(billing_complete=False, usage=None, usage_status="UNKNOWN")
            record.update(
                response_sha256=response_digest.hexdigest(),
                recorded_response_sha256=stored_digest.hexdigest(),
                finished_at=_now(), elapsed_seconds=time.monotonic() - started,
            )
            try:
                persist()
            except OSError:
                self._record_error("RECEIPT_WRITE_FAILED")
