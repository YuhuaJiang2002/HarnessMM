"""Keep the official DSH session and credential on the host, tools in native SForge."""

from __future__ import annotations

import dataclasses
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sys
import threading
import time
from typing import Any

from integrations.deepseek_harness import runner
from integrations.deepseek_harness.harbor_agent import (
    TERMINAL_TOOL as HARBOR_TERMINAL_TOOL, terminal_arguments,
)
from phiagent.data_engine.provenance import capture_provenance, write_json_atomic
from phiagent.harness.deepseek_bridge import ToolBridge
from phiagent.harness.deepseek_messages_meter import DeepSeekMessagesMeter


TERMINAL_TOOL = {
    **HARBOR_TERMINAL_TOOL,
    "description": (
        "Run a non-interactive shell command inside this task's isolated native SForge "
        "work container. Filesystem changes persist. Use sforge-submit for native "
        "agent-visible evaluation feedback; the official task deadline applies."
    ),
}


def docker_terminal_command(arguments: dict, default_cwd: str) -> tuple[list[str], str, int | None]:
    checked = terminal_arguments(arguments)
    return (
        ["bash", "-lc", checked["command"]],
        checked.get("cwd", default_cwd), checked.get("timeout_sec"),
    )


class NativeTools:
    def __init__(self, backend: Any, handle: Any, cwd: str, cancel: threading.Event, root: Path):
        self.backend, self.handle, self.cwd, self.cancel, self.root = backend, handle, cwd, cancel, root

    def call(self, name: str, arguments: dict) -> dict:
        if name != "terminal":
            raise ValueError("Only the native isolated terminal tool is registered")
        try:
            command, cwd, timeout = docker_terminal_command(arguments, self.cwd)
        except ValueError as error:
            return {"observation": json.dumps({
                "error_type": "InvalidArguments", "error": str(error), "executed": False,
            }), "conclude_turn": False}
        if self.cancel.is_set():
            raise RuntimeError("Native task has ended; no further mutations are allowed")
        result = self.backend.exec_run_with_timeout(
            self.handle, command, timeout=timeout, workdir=cwd,
            environment={"CUDA_VISIBLE_DEVICES": ""}, shutdown_event=self.cancel,
            log_file=None, stream_to_stdout=False, log_append=False, on_chunk=None,
        )
        output = result.output.decode(errors="replace") if isinstance(result.output, bytes) else result.output
        return {"observation": json.dumps({
            "output": output, "exit_code": result.exit_code,
            "timed_out": result.timed_out, "elapsed_seconds": result.elapsed_seconds,
        }), "conclude_turn": False}

    def notifications(self) -> list[str]:
        return []

    def generation_start(self) -> None:
        raise RuntimeError("EdgeBench uses native wall-clock time, not a simulated generation clock")

    def generation_end(self, elapsed_seconds: float) -> None:
        raise RuntimeError("EdgeBench uses native wall-clock time, not a simulated generation clock")


class OfficialSession:
    def __init__(self, job: dict, backend: Any, handle: Any, cwd: str, cancel: threading.Event):
        if job["tools"] != [TERMINAL_TOOL]:
            raise ValueError("EdgeBench must expose exactly its isolated native terminal catalog")
        runner.validate_job({**job, "tools": [], "bridge": None})
        if socket.gethostname() != runner.REMOTE_HOSTNAME or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Official EdgeBench SDK execution is A800 CPU/API-only")
        self.job, self.root, self.cancel = job, Path(job["run_dir"]), cancel
        self.root.mkdir(parents=True, exist_ok=False)
        for name in ("source", "home", "cwd", "cache", "sessions"):
            (self.root / name).mkdir()
        for name in ("runner.py", *runner.PLUGIN_FILES):
            shutil.copy2(Path(runner.__file__).parent / name, self.root / "source" / name)
        manifest = runner.verify_runtime(job["runtime"])
        runner.verify_installed_sdk(Path(job["runtime"]["source_root"]))
        write_json_atomic(self.root / "runtime-manifest.json", manifest)
        write_json_atomic(self.root / "provenance.json", {
            **capture_provenance(Path(__file__).resolve().parents[2], sys.argv, job["seed"]),
            "upstream_commit": runner.UPSTREAM_COMMIT, "upstream_tag": runner.UPSTREAM_TAG,
            "profile": "sdk", "cumulative_token_limit": None, "selected_gpus": [],
            "seed_role": "Paired repetition identity, not a provider sampling seed",
            "native_work_container": handle.id,
        })
        if job["chisel"] is not None:
            if runner.sha256_file(Path(job["chisel"]["path"])) != job["chisel"]["sha256"]:
                raise ValueError("CHISEL artifact changed before native execution")
            shutil.copy2(job["chisel"]["path"], self.root / "frozen-chisel.txt")
        key = runner.read_owner_secret(Path(job["api_key_file"]))
        self.turns: list[dict] = []
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.closed = False
        node = Path(job["runtime"]["node_bin"])
        from deepseek_harness import DeepSeekHarness

        self.tools = NativeTools(backend, handle, cwd, cancel, self.root)
        self.bridge = ToolBridge(self.root / "bridge", self.tools, [TERMINAL_TOOL])
        self.meter = DeepSeekMessagesMeter(self.root / "provider", key)
        self.secrets = (key, self.bridge.secret, self.meter.client_key)
        self.harness = None
        self.bridge.__enter__()
        try:
            self.meter.start()
            write_json_atomic(self.root / "PROVIDER_TRANSPORT.json", {
                "transport": "Byte-preserving native Anthropic Messages with independent usage drain",
                "upstream": runner.BASE_URL, "meter_url": self.meter.url,
                "request_body_modified": False, "automatic_retries": 0,
                "provider_key_in_sdk": False, "provider_key_in_work_container": False,
            })
            job["bridge"] = {
                "url": self.bridge.url, "secret_file": str(self.bridge.secret_file),
                "notifications": False, "generation_timing": False,
            }
            runner.validate_job(job)
            write_json_atomic(self.root / "job.json", job)
            patch = self.root / "benchmark.patch.yml"
            write_json_atomic(patch, runner.build_patch(job, self.root))
            self.harness = DeepSeekHarness(
                provider=runner.PROVIDER, model=job["model"], profile="sdk",
                base_url=self.meter.url, max_tokens=job["max_tokens"], api_key=self.meter.client_key,
                cwd=str(self.root / "cwd"), runtime_cwd=str(self.root / "cwd"),
                dsh_home=str(self.root / "home"), dsh_bin=job["runtime"]["dsh_bin"],
                patches=(str(patch),), env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "PATH": str(node.parent) + os.pathsep + os.environ["PATH"],
                    "HOME": str(self.root / "home"), "TMPDIR": str(self.root / "cache"),
                    "DSH_HOME": str(self.root / "home"), "DSH_PRIMARY_RUNTIME": "",
                    "OTEL_SDK_DISABLED": "true",
                },
                initialize_timeout_seconds=120, request_timeout_seconds=None,
                shutdown_timeout_seconds=15,
            )
            self.harness.__enter__()
        except Exception as error:
            write_json_atomic(self.root / "INITIALIZATION_FAILURE.json", {
                "type": type(error).__name__, "message": runner._redact(str(error), self.secrets),
            })
            if self.harness is not None:
                self.harness.close()
            self.meter.close()
            self.bridge.__exit__(None, None, None)
            raise

    def notification(self, value: Any) -> None:
        raw = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
        raw = runner._redact(raw, self.secrets)
        with (self.root / "notifications.jsonl").open("a") as stream:
            stream.write(runner.canonical_json(raw) + "\n")
            stream.flush()
        if isinstance(raw, dict) and raw.get("method") == "session.event":
            payload = raw.get("payload", {})
            if isinstance(payload.get("event"), dict):
                self.events.append({"session_id": payload.get("sessionId"), "event": payload["event"]})

    def run(self, prompt: str) -> dict:
        started = time.time()
        turn: dict = {"started_at_unix": started, "status": "failed"}
        try:
            if self.closed or self.cancel.is_set():
                raise RuntimeError("The native task was cancelled")
            result = self.harness.run(
                prompt, session_id=self.job["session_id"], on_notification=self.notification,
            )
            turn.update(
                status="completed" if result.finish_reason == "completed" else "failed",
                finish_reason=result.finish_reason,
                final_response=runner._redact(result.final_response, self.secrets),
            )
            write_json_atomic(
                self.root / f"sdk-turn-{len(self.turns):04d}.json",
                runner._redact(dataclasses.asdict(result), self.secrets),
            )
        except Exception as error:
            turn["error"] = {
                "type": type(error).__name__, "message": runner._redact(str(error), self.secrets),
            }
            raise
        finally:
            turn["elapsed_seconds"] = time.time() - started
            self.turns.append(turn)
            self.save()
        return turn

    def save(self) -> dict:
        with self.lock:
            streams = runner._load_jsonl(self.root / "llm-streams.jsonl")
            sdk_usage = runner.summarize_streams(streams)
            provider = self.meter.snapshot()
            differences = {}
            if sdk_usage["complete"] and provider["complete"] and sdk_usage["usage"] and provider["usage"]:
                for name in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens"):
                    if name in sdk_usage["usage"] and name in provider["usage"]:
                        if sdk_usage["usage"][name] != provider["usage"][name]:
                            differences[name] = {
                                "sdk": sdk_usage["usage"][name], "provider": provider["usage"][name],
                            }
            usage = {
                "complete": provider["complete"] and not differences,
                "usage": provider["usage"] if not differences else None,
                "known_reported_usage": provider["known_usage"],
                "attempt_count": provider["request_count"],
                "unknown_request_count": provider["unknown_request_count"],
                "source": "durable native provider HTTP receipts, including post-disconnect completion",
                "sdk_provider_differences": differences,
            }
            write_json_atomic(self.root / "events.json", self.events)
            write_json_atomic(self.root / "SDK_USAGE.json", sdk_usage)
            write_json_atomic(self.root / "PROVIDER_USAGE.json", provider)
            write_json_atomic(self.root / "usage.json", usage)
            result = {
                "turns": self.turns, "usage": usage["usage"], "usage_complete": usage["complete"],
                "cancelled": self.cancel.is_set(), "run_dir": str(self.root),
            }
            write_json_atomic(self.root / "result.json", result)
            return result

    def close(self) -> None:
        with self.close_lock:
            if self.closed:
                return
            self.closed = True
            self.cancel.set()
            self.meter.seal_admission()
            try:
                self.harness.close()
            finally:
                try:
                    self.bridge.__exit__(None, None, None)
                finally:
                    self.meter.close()
                    self.save()


class HostRelay:
    def __init__(self, job: dict, *, gateway: str, cwd: str, expected_prompt: str):
        self.job, self.cwd, self.expected_prompt = job, cwd, expected_prompt
        self.key = secrets.token_urlsafe(32)
        self.cancel = threading.Event()
        self.session: OfficialSession | None = None
        self.bound: tuple[Any, Any] | None = None
        self.judge_token: str | None = None
        self.call_lock = threading.Lock()
        self.initial_started = False
        self.errors: list[dict] = []
        self.stop_requested_at: float | None = None
        self.closed = False
        self.close_lock = threading.Lock()
        self.server = ThreadingHTTPServer(("0.0.0.0", 0), RelayHandler)
        self.server.relay = self
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, name="edgebench-host-relay")
        self.url = f"http://{gateway}:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def bind(self, backend: Any, handle: Any, log_dir: Path) -> None:
        if self.bound is not None:
            raise RuntimeError("Relay cannot change its native work-container binding")
        self.bound = backend, handle
        container = backend.client.containers.get(handle.id)
        container.reload()
        environment = dict(value.split("=", 1) for value in container.attrs["Config"]["Env"])
        self.judge_token = environment["SFORGE_TOKEN"]
        write_json_atomic(log_dir / "DSH_BINDING.json", {
            "container_id": handle.id, "container_name": handle.name,
            "sdk_run_dir": self.job["run_dir"], "provider_key_inside_container": False,
        })

    def execute(self, prompt: str, initial: bool) -> dict:
        if self.bound is None or self.cancel.is_set():
            raise RuntimeError("Native work container is unavailable or cancelled")
        if initial:
            if self.initial_started or prompt != self.expected_prompt:
                raise ValueError("Initial prompt must be the single unchanged native SForge prompt")
            self.initial_started = True
        elif not self.initial_started or prompt != "Continue working.":
            raise ValueError("Resume requires the existing official SDK session")
        if self.session is None:
            backend, handle = self.bound
            self.session = OfficialSession(self.job, backend, handle, self.cwd, self.cancel)
        return self.session.run(prompt)

    def close(self) -> None:
        with self.close_lock:
            if self.closed:
                return
            self.closed = True
            if self.stop_requested_at is None:
                self.stop_requested_at = time.time()
            self.cancel.set()
            if self.session is not None:
                self.session.close()
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()


class RelayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        return None

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ready"}\n')

    def do_POST(self) -> None:
        relay = self.server.relay
        if not hmac.compare_digest(self.headers.get("X-Relay-Run", ""), relay.key):
            self.send_error(403)
            return
        if self.path not in {"/run", "/resume"}:
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 2 * 1024 * 1024 or not relay.call_lock.acquire(blocking=False):
            self.send_error(409)
            return
        outcome: dict = {}
        finished = threading.Event()
        try:
            self.connection.settimeout(30)
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("Truncated native prompt")
            prompt = raw.decode("utf-8")

            def execute():
                try:
                    outcome["result"] = relay.execute(prompt, self.path == "/run")
                except Exception as error:
                    exposed = relay.session.secrets if relay.session is not None else (relay.key,)
                    receipt = {
                        "at_unix": time.time(),
                        "type": type(error).__name__,
                        "message": (
                            runner._redact(str(error), exposed) if relay.session is not None
                            else "SDK initialization or native prompt contract failed"
                        ),
                    }
                    relay.errors.append(receipt)
                    write_json_atomic(Path(relay.job["run_dir"]) / "RELAY_FAILURES.json", relay.errors)
                    outcome["error"] = {
                        "type": type(error).__name__, "message": "See the preserved host failure receipt",
                    }
                    relay.cancel.set()
                    if relay.session is not None:
                        relay.session.close()
                finally:
                    finished.set()

            worker = threading.Thread(target=execute, name="edgebench-official-sdk-turn")
            worker.start()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                while not finished.wait(1):
                    self.wfile.write(b'{"status":"running"}\n')
                    self.wfile.flush()
                self.wfile.write(json.dumps(outcome).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                if relay.stop_requested_at is None:
                    relay.stop_requested_at = time.time()
                relay.cancel.set()
                if relay.session is not None:
                    relay.session.close()
            finally:
                worker.join()
        finally:
            relay.call_lock.release()


def make_agent(config: Any, relay: HostRelay):
    from sforge.harness.agent.base import Agent

    class DeepSeekHostAgent(Agent):
        name = "deepseek-harness-host-adapter"
        install_cmds = []
        api_key_env = "UNUSED_PROVIDER_KEY"
        api_base_env = "EDGEBENCH_RELAY_URL"
        model_env = "PHI_DSH_MODEL"
        run_cmd = (
            'curl --noproxy "*" --fail --silent --show-error --no-buffer '
            '-H "Content-Type: text/plain" -H "X-Relay-Run: $EDGEBENCH_RELAY_RUN" '
            '--data-binary @"{prompt_file}" "$EDGEBENCH_RELAY_URL/run"'
        )
        resume_cmd = (
            'curl --noproxy "*" --fail --silent --show-error --no-buffer '
            '-H "X-Relay-Run: $EDGEBENCH_RELAY_RUN" '
            '--data-binary "Continue working." "$EDGEBENCH_RELAY_URL/resume"'
        )

        def augment_env(self, env, model):
            if not env.get(self.api_base_env):
                raise RuntimeError("Native SForge did not supply the relay endpoint")
            env["EDGEBENCH_RELAY_RUN"] = relay.key

        def install_stop_hook(self, backend, handle, log_dir, logger):
            relay.bind(backend, handle, log_dir)

    return DeepSeekHostAgent(config)
