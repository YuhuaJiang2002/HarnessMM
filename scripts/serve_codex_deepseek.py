#!/usr/bin/env python3
"""Byte-preserving, metered native Responses transport for remote Codex trials.

The upstream credential never reaches an agent container. Each trial uses an
HMAC-derived relay token and /<trial_id> as its provider base URL. This server
does not change prompts, tools, responses, sampling, or model behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPException
import json
import os
from pathlib import Path
import queue
import re
import secrets
import socket
import stat
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid
from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import write_json_atomic  # noqa: E402
from phiagent.experiments import create_frozen_experiment_directory  # noqa: E402

TRIAL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}
BODY_LIMIT_REJECTION = b"Failed to buffer the request body: length limit exceeded"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deliver_to_client(operation: Callable[[], None], receipt: dict) -> bool:
    """A departed client must not interrupt capture of paid upstream usage."""
    if receipt.get("client_disconnected"):
        return False
    try:
        operation()
        return True
    except OSError as error:
        receipt.update(
            client_disconnected=True,
            client_delivery_error={"type": type(error).__name__, "message": str(error)},
        )
        return False


class ResponseDelivery:
    """Drain the provider independently of a slow or departed downstream client."""

    def __init__(self, begin, send, disconnect, receipt: dict, *, queue_chunks: int = 64):
        self.begin, self.send, self.disconnect = begin, send, disconnect
        self.receipt = receipt
        self.chunks = queue.Queue(maxsize=queue_chunks)
        self.finished = threading.Event()
        self.worker = threading.Thread(target=self._deliver, daemon=True)
        self.worker.start()

    def _deliver(self):
        if not deliver_to_client(self.begin, self.receipt):
            return
        while not self.receipt.get("client_disconnected"):
            try:
                chunk = self.chunks.get(timeout=0.1)
            except queue.Empty:
                if self.finished.is_set():
                    return
                continue
            if not deliver_to_client(lambda: self.send(chunk), self.receipt):
                return

    def _disconnect(self, reason):
        self.receipt.update(
            client_disconnected=True,
            client_delivery_error={"type": "DownstreamBackpressure", "message": reason},
        )
        try:
            self.disconnect()
        except OSError:
            pass

    def submit(self, chunk: bytes):
        if self.receipt.get("client_disconnected"):
            return
        try:
            self.chunks.put_nowait(chunk)
        except queue.Full:
            self._disconnect("Client delivery buffer filled; upstream capture continues")

    def close(self, *, drain_timeout: float = 16):
        self.finished.set()
        self.worker.join(timeout=drain_timeout)
        if self.worker.is_alive():
            self._disconnect("Client delivery did not finish; upstream capture is preserved")
            self.worker.join(timeout=1)
        self.receipt["client_delivery_finished"] = not self.worker.is_alive()


def trial_token(secret: str, trial_id: str) -> str:
    if not TRIAL_PATTERN.fullmatch(trial_id):
        raise ValueError("Invalid trial identifier")
    return hmac.new(secret.encode(), trial_id.encode(), hashlib.sha256).hexdigest()


def read_credential(path: Path) -> str:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_uid != os.geteuid()
    ):
        raise ValueError("Credential must be an owner-only regular file owned by this user")
    value = path.read_text().strip()
    if not value:
        raise ValueError("Empty credential")
    return value


def native_response_usage(response: dict) -> dict:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Missing native Responses usage")
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise ValueError("Invalid native Responses token count: " + field)
    if usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]:
        raise ValueError("Native Responses tokens do not reconcile")
    for details, field, bound in (
        ("input_tokens_details", "cached_tokens", "input_tokens"),
        ("output_tokens_details", "reasoning_tokens", "output_tokens"),
    ):
        detail = usage.get(details)
        if detail is not None:
            if not isinstance(detail, dict):
                raise ValueError("Invalid native token details")
            if field in detail and (
                type(detail[field]) is not int or not 0 <= detail[field] <= usage[bound]
            ):
                raise ValueError("Invalid native token breakdown: " + field)
    return usage


def terminal_response(raw: bytes, content_type: str) -> dict:
    if "text/event-stream" not in content_type:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Provider response is not an object")
        return result
    terminals = []
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for frame in text.split("\n\n"):
        data = "\n".join(
            line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("type") in TERMINAL_EVENTS:
            terminals.append(event["response"])
    if len(terminals) != 1:
        raise ValueError("Expected exactly one terminal Responses event")
    return terminals[0]


def validate_native_fast_request(body: dict) -> None:
    """Admit only a charged label decision; preserve its exact upstream bytes."""
    expected = {"model", "temperature", "max_tokens", "thinking", "stream",
                "logprobs", "top_logprobs", "messages"}
    if (set(body) != expected or body.get("model") != "deepseek-flash"
            or body.get("temperature") != 0 or type(body.get("max_tokens")) is not int
            or body["max_tokens"] != 1 or body.get("thinking") != {"type": "disabled"}
            or body.get("stream") is not False or body.get("logprobs") is not True
            or body.get("top_logprobs") != 20):
        raise ValueError("Only the frozen one-token native fast decision is admitted")
    messages = body.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or [m.get("role") for m in messages if isinstance(m, dict)] != ["system", "user"]
            or any(set(m) != {"role", "content"} or not isinstance(m["content"], str)
                   for m in messages)):
        raise ValueError("Fast decisions require two public text messages")


def native_chat_usage(response: dict) -> dict:
    """Normalize native Chat counters without adding cache hits twice."""
    usage = response.get("usage", {})
    keys = ("prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in keys):
        raise ValueError("Native fast usage is incomplete")
    if (usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
            or usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
            != usage["prompt_tokens"]):
        raise ValueError("Native fast usage counters disagree")
    return native_response_usage({"usage": {
        "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "input_tokens_details": {"cached_tokens": usage["prompt_cache_hit_tokens"]},
    }})


def native_receipt_fields(raw: bytes, content_type: str, http_status: int,
                          *, endpoint: str = "/responses") -> dict:
    """Classify provider failures without turning absent native usage into zero."""
    fields = {"usage_status": "UNKNOWN", "response_sha256": hashlib.sha256(raw).hexdigest()}
    if http_status == 413 and raw.strip() == BODY_LIMIT_REJECTION:
        fields.update(
            failure_classification="UPSTREAM_REQUEST_BODY_LIMIT",
            native_request_stage="rejected_while_buffering_body",
            usage_evidence="HTTP request dispatched; provider returned no native usage counters",
        )
        return fields
    try:
        terminal = json.loads(raw) if endpoint == "/chat/completions" else terminal_response(raw, content_type)
    except (ValueError, KeyError, TypeError) as error:
        fields.update(
            failure_classification=(
                "UPSTREAM_HTTP_ERROR" if http_status >= 400 else "INCOMPLETE_NATIVE_RESPONSE"
            ),
            native_response_parse_error_type=type(error).__name__,
            native_response_parse_error=str(error),
        )
        return fields
    fields.update({
        "response_id": terminal.get("id"), "model": terminal.get("model"),
        "response_status": terminal.get("status"),
        "system_fingerprint": terminal.get("system_fingerprint"),
    })
    if isinstance(terminal.get("error"), dict):
        fields["native_error"] = {
            key: terminal["error"].get(key) for key in ("type", "code", "param", "message")
        }
        fields["failure_classification"] = "NATIVE_PROVIDER_ERROR"
    try:
        usage = native_chat_usage(terminal) if endpoint == "/chat/completions" else native_response_usage(terminal)
        fields.update(usage=usage, usage_status="MEASURED")
        if endpoint == "/chat/completions":
            fields["native_usage"] = terminal["usage"]
    except ValueError as error:
        fields["native_usage_error"] = str(error)
    return fields


def usage_summary(root: Path, trial_id: str) -> dict:
    if not TRIAL_PATTERN.fullmatch(trial_id):
        raise ValueError("Invalid trial identifier")
    directory = root / "calls" / trial_id
    calls = sorted(directory.iterdir()) if directory.exists() else []
    totals = dict.fromkeys(("input_tokens", "output_tokens", "total_tokens"), 0)
    cached = reasoning = 0
    missing_breakdown = set()
    unresolved = []
    receipts = []
    failed_calls = []
    for call in calls:
        receipt_path = call / "RECEIPT.json"
        if not receipt_path.exists():
            unresolved.append(call.name)
            continue
        receipt = json.loads(receipt_path.read_text())
        receipts.append(receipt)
        if (
            (receipt.get("http_status") or 0) >= 400
            or receipt.get("failure_classification") or receipt.get("error_type")
        ):
            failed_calls.append({
                "call_id": call.name, "http_status": receipt.get("http_status"),
                "provider_dispatched": receipt.get("provider_dispatched"),
                "failure_classification": receipt.get("failure_classification"),
                "error_type": receipt.get("error_type"),
                "usage_status": receipt.get("usage_status"),
                "receipt": str(receipt_path),
            })
        if receipt.get("usage_status") != "MEASURED":
            unresolved.append(call.name)
            continue
        usage = native_response_usage({"usage": receipt["usage"]})
        for name in totals:
            totals[name] += usage[name]
        for detail, field in (
            ("input_tokens_details", "cached_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ):
            if field not in (usage.get(detail) or {}):
                missing_breakdown.add(field)
        cached += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    return {
        "trial_id": trial_id,
        "usage_status": "MEASURED" if calls and not unresolved else "UNRESOLVED",
        "calls": len(calls),
        "known_usage": totals,
        "cached_input_tokens": None if "cached_tokens" in missing_breakdown else cached,
        "reasoning_output_tokens": None if "reasoning_tokens" in missing_breakdown else reasoning,
        "unresolved_call_ids": unresolved,
        "failed_calls": failed_calls,
        "provider_models": sorted({r["model"] for r in receipts if r.get("model")}),
        "model_response_ids": [r["response_id"] for r in receipts if r.get("response_id")],
        "receipts": [str(call / "RECEIPT.json") for call in calls],
    }


def make_server(root: Path, provider_key: str, relay_secret: str, *, bind: str, port: int,
                allow_native_fast: bool = False, terminal_bank=None):
    opener = build_opener(ProxyHandler({}))
    semaphore = threading.BoundedSemaphore(8)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def respond(self, status: int, result: dict):
            raw = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def identity(self):
            parts = self.path.split("/")
            if len(parts) < 3 or not TRIAL_PATTERN.fullmatch(parts[1]):
                return None
            ident = parts[1]
            expected = "Bearer " + trial_token(relay_secret, ident)
            if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                return None
            return ident, "/" + "/".join(parts[2:])

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {
                    "status": "READY", "hostname": socket.gethostname(),
                    "provider": "https://api.deepseek.com", "model": "deepseek-flash",
                    "transport": "native-responses-byte-preserving", "root": str(root),
                    "native_fast_decisions": allow_native_fast,
                })
                return
            identity = self.identity()
            if identity is None:
                self.respond(403, {"error": "Invalid trial authorization"})
            elif identity[1] == "/usage":
                self.respond(200, usage_summary(root, identity[0]))
            else:
                self.respond(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            identity = self.identity()
            if identity is None:
                self.respond(403, {"error": "Invalid trial authorization"})
                return
            ident, endpoint = identity
            admitted = ("/responses", "/responses/compact") + (("/chat/completions",) if allow_native_fast else ())
            if endpoint not in admitted:
                self.respond(404, {"error": "Only native Responses endpoints are admitted"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError("Invalid request length")
                if self.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Compressed requests are not supported by this recorder")
                self.connection.settimeout(1800)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(raw)
                if body.get("model") != "deepseek-flash":
                    raise ValueError("Model differs from the frozen provider identity")
                if endpoint == "/chat/completions":
                    validate_native_fast_request(body)
            except (ValueError, TimeoutError) as error:
                self.respond(400, {"error": str(error), "provider_dispatched": False})
                return
            if endpoint == "/responses" and terminal_bank is not None:
                try:
                    compiled = terminal_bank(ident, body, raw)
                except (OSError, ValueError, RuntimeError) as error:
                    self.respond(502, {"error": "Native terminal bank decision failed",
                                       "failure_type": type(error).__name__})
                    return
                if compiled is not None:
                    rendered, content_type = compiled
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(rendered)))
                    self.end_headers()
                    self.wfile.write(rendered)
                    return
            if not semaphore.acquire(timeout=1800):
                self.respond(503, {"error": "Provider concurrency queue timed out"})
                return
            call = root / "calls" / ident / uuid.uuid4().hex
            call.mkdir(parents=True)
            (call / "request.json").write_bytes(raw)
            receipt = {
                "trial_id": ident, "call_id": call.name, "endpoint": endpoint,
                "started_at": now(), "request_sha256": hashlib.sha256(raw).hexdigest(),
                "request_bytes": len(raw), "upstream_request_bytes": len(raw),
                "usage_status": "UNKNOWN", "body_rewritten": False,
                "provider_dispatched": False, "client_disconnected": False,
            }
            started = time.monotonic()
            delivery = None
            try:
                headers = {
                    "Authorization": "Bearer " + provider_key,
                    "Content-Type": "application/json",
                }
                for name in ("Accept", "User-Agent", "OpenAI-Beta", "Originator"):
                    if self.headers.get(name):
                        headers[name] = self.headers[name]
                request = Request(
                    "https://api.deepseek.com" + endpoint, data=raw,
                    headers=headers, method="POST",
                )
                receipt["provider_dispatched"] = True
                write_json_atomic(call / "DISPATCH.json", receipt)
                try:
                    response = opener.open(request, timeout=1800)
                except HTTPError as error:
                    response = error
                with response, (call / "response.body").open("wb") as output:
                    content_type = response.headers.get("Content-Type", "application/json")
                    receipt.update(
                        http_status=response.status, content_type=content_type,
                        trace_id=response.headers.get("x-ds-trace-id"),
                    )
                    self.connection.settimeout(15)

                    def begin_delivery():
                        self.send_response(response.status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Connection", "close")
                        if endpoint == "/chat/completions":
                            self.send_header("X-PhiAgent-Call-ID", call.name)
                        self.end_headers()

                    self.close_connection = True

                    def send_chunk(chunk):
                        self.wfile.write(chunk)
                        self.wfile.flush()

                    delivery = ResponseDelivery(
                        begin_delivery, send_chunk,
                        lambda: self.connection.shutdown(socket.SHUT_RDWR), receipt,
                    )
                    while chunk := response.read1(65536):
                        output.write(chunk)
                        output.flush()
                        delivery.submit(chunk)
                    os.fsync(output.fileno())
                captured = (call / "response.body").read_bytes()
                fields = native_receipt_fields(captured, content_type, response.status, endpoint=endpoint)
                receipt.update(json.loads(json.dumps(fields).replace(provider_key, "[REDACTED]")))
            except (OSError, URLError, HTTPException, ValueError, KeyError) as error:
                receipt["error"] = str(error).replace(provider_key, "[REDACTED]")
                receipt["error_type"] = type(error).__name__
                if isinstance(error, URLError) and isinstance(error.reason, ConnectionResetError):
                    receipt["failure_classification"] = "UPSTREAM_CONNECTION_RESET"
                if delivery is None and not receipt["client_disconnected"]:
                    deliver_to_client(
                        lambda: self.respond(502, {"error": "Provider transport failed", "call_id": call.name}),
                        receipt,
                    )
            finally:
                if delivery is not None:
                    delivery.close()
                captured_path = call / "response.body"
                if captured_path.exists() and receipt.get("usage_status") != "MEASURED":
                    fields = native_receipt_fields(
                        captured_path.read_bytes(), receipt.get("content_type", ""),
                        receipt.get("http_status") or 0,
                        endpoint=endpoint,
                    )
                    receipt.update(json.loads(json.dumps(fields).replace(provider_key, "[REDACTED]")))
                receipt.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
                write_json_atomic(call / "RECEIPT.json", receipt)
                semaphore.release()

    server = ThreadingHTTPServer((bind, port), Handler)
    server.daemon_threads = False
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--relay-secret", type=Path, required=True)
    parser.add_argument(
        "--bind", choices=("127.0.0.1", "172.17.0.1", "172.29.253.1"), default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=28160)
    parser.add_argument("--allow-native-fast", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Run this service only on the authorized A800-4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.umask(0o077)
    provider_key = read_credential(args.credential)
    if not args.relay_secret.exists():
        with args.relay_secret.open("x") as stream:
            stream.write(secrets.token_hex(32) + "\n")
        args.relay_secret.chmod(0o600)
    relay_secret = read_credential(args.relay_secret)
    config = {
        "seed": 20260928, "model": "deepseek-flash", "maximum_tokens": None,
        "token_usage_mode": "unlimited", "transport_retries": 0,
        "upstream": "https://api.deepseek.com", "bind": args.bind, "port": args.port,
        "credential_file": str(args.credential), "relay_secret_file": str(args.relay_secret),
        "CUDA_VISIBLE_DEVICES": "", "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "native_fast_decisions": args.allow_native_fast,
    }
    root = create_frozen_experiment_directory(
        args.root, "codex-native-responses", config, repo_root=ROOT
    ).path
    server = make_server(root, provider_key, relay_secret, bind=args.bind, port=args.port,
                         allow_native_fast=args.allow_native_fast)
    write_json_atomic(root / "SERVICE.json", {
        **config, "status": "READY", "hostname": socket.gethostname(), "pid": os.getpid(),
        "started_at": now(), "argv": sys.argv,
        "resource_inventory": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv",
        ], text=True, timeout=30),
    })
    print(json.dumps({"status": "READY", "root": str(root), "pid": os.getpid()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
