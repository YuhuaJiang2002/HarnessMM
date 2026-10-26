"""Authenticated, exact-once bridge from the official DSH loop to native tools."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Protocol

from phiagent.data_engine.provenance import write_json_atomic
from phiagent.harness.loopback_http import FetchCompatibleHTTPServer


class NativeBackend(Protocol):
    def call(self, name: str, arguments: dict) -> dict: ...
    def notifications(self) -> list[str | list[dict]]: ...
    def generation_start(self) -> None: ...
    def generation_end(self, elapsed_seconds: float) -> None: ...


class BridgeError(RuntimeError):
    pass


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


class ToolBridge(AbstractContextManager):
    def __init__(self, root: Path, backend: NativeBackend, catalog: list[dict]):
        self.root, self.backend, self.catalog = root, backend, catalog
        self.names = {tool["name"] for tool in catalog}
        if len(self.names) != len(catalog) or any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("Native tool names must be unique nonempty strings")
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.secret = secrets.token_urlsafe(32)
        self.secret_file = root / "bridge.secret"
        descriptor = os.open(self.secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(self.secret + "\n")
        self.server = FetchCompatibleHTTPServer(("127.0.0.1", 0), BridgeHandler)
        self.server.bridge = self
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, name="dsh-native-bridge")
        self.generation_active = False
        self.blocked = False
        write_json_atomic(root / "CATALOG.json", catalog)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.secret_file.unlink()
        return False

    def dispatch(self, path: str, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("Bridge request must be a JSON object")
        with self.lock:
            if self.blocked:
                raise BridgeError("Native bridge stopped after an unresolved infrastructure failure")
            if path == "/tool":
                return self.tool(request)
            if path == "/notifications":
                messages = self.backend.notifications()
                return {
                    "messages": messages,
                    "stop_requested": bool(getattr(self.backend, "stop_requested", False)),
                }
            if path == "/generation/start":
                if self.generation_active:
                    raise BridgeError("Overlapping model-generation clock intervals")
                self.backend.generation_start()
                self.generation_active = True
                return {"status": "PAUSED_FOR_GENERATION"}
            if path == "/generation/end":
                elapsed = request.get("elapsed_seconds")
                if type(elapsed) not in (int, float) or not 0 <= elapsed < float("inf"):
                    raise ValueError("A finite measured generation duration is required")
                if not self.generation_active:
                    raise BridgeError("No active model-generation clock interval")
                self.backend.generation_end(float(elapsed))
                self.generation_active = False
                return {"status": "RESUMED_WITH_MEASURED_OFFSET"}
            if path == "/health":
                return {"status": "READY", "tool_count": len(self.catalog)}
            raise ValueError("Unknown bridge endpoint")

    def tool(self, request: dict) -> dict:
        if set(request) != {"session_id", "call_id", "name", "arguments"}:
            raise ValueError("Tool request fields differ from the frozen bridge contract")
        if any(not isinstance(request[field], str) or not request[field] for field in ("session_id", "call_id", "name")):
            raise ValueError("Tool identity fields must be nonempty strings")
        if request["name"] not in self.names or not isinstance(request["arguments"], dict):
            raise ValueError("Unknown native tool or non-object arguments")
        ident = canonical_hash([request["session_id"], request["call_id"]])
        folder = self.root / "calls" / ident
        payload_hash = canonical_hash(request)
        start_file, result_file = folder / "STARTED.json", folder / "RESULT.json"
        if start_file.exists():
            old = json.loads(start_file.read_text())
            if old["request_sha256"] != payload_hash:
                raise BridgeError("Tool call identity was reused with different arguments")
            if result_file.exists():
                return json.loads(result_file.read_text())
            self.blocked = True
            raise BridgeError("Previous native mutation has no settled receipt; do not replay it")
        write_json_atomic(start_file, {
            "at_unix": time.time(), "request_sha256": payload_hash, "request": request,
        })
        with start_file.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            result = self.backend.call(request["name"], request["arguments"])
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("observation"), str)
                or type(result.get("conclude_turn")) is not bool
            ):
                raise BridgeError("Native tool returned an invalid observation contract")
            canonical_hash(result)
            write_json_atomic(result_file, result)
            return result
        except Exception as error:
            self.blocked = True
            write_json_atomic(folder / "FAILURE.json", {
                "at_unix": time.time(), "type": type(error).__name__,
                "error": str(error).replace(self.secret, "[REDACTED]"),
                "mutation_replay_allowed": False,
            })
            raise


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        return None

    def do_POST(self) -> None:
        bridge = self.server.bridge
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + bridge.secret,
        ):
            self.reply(403, {"error": "Authenticated native bridge access required"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4 * 1024 * 1024:
                raise ValueError("Invalid bridge request size")
            self.connection.settimeout(30)
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("Truncated bridge request")
            request = json.loads(raw)
            result = bridge.dispatch(self.path, request)
        except (ValueError, KeyError) as error:
            self.reply(400, {"error": str(error).replace(bridge.secret, "[REDACTED]")})
            return
        except Exception as error:
            write_json_atomic(bridge.root / "BRIDGE_FAILURE.json", {
                "type": type(error).__name__,
                "error": str(error).replace(bridge.secret, "[REDACTED]"),
                "path": self.path,
            })
            self.reply(500, {"error": "Native bridge failure; see preserved failure receipt"})
            return
        self.reply(200, result)

    def reply(self, status: int, result: dict) -> None:
        encoded = json.dumps(result, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            # A settled mutation stays on disk even when its acknowledgement is lost.
            return
