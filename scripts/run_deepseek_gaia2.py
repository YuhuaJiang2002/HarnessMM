#!/usr/bin/env python3
"""Official DeepSeek Harness / native ARE adapter, restricted to A800 CPU/API.

The ARE process runs in a mount/PID sandbox. The SDK stays in its separately
pinned Python environment and owns every reasoning step, tool decision and
conversation turn. An authenticated supervisor relays its JSONL session ABI
and meters the unchanged native judge's OpenAI-compatible HTTP requests.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAMPAIGN_ROOT = Path("/opt/phiagent/work/deepseek-harness/20260929-v1")
DATA0_ROOT = Path("/path/to/data/phiagent/work/deepseek-harness")
AUTHORIZED_HOST = "harnessmm-api-worker"
MODEL = "deepseek-flash"
OFFICIAL_REVISION = "4878cdabd87d4041bdaff61d04c966883b9fd07a"
ARE_VERSION = "1.1.0"
PARTITION_SHA256 = "5549b4a72340ae1d4cc883e419dc9d947af17b7a3f69fd1d60950c17108bebec"
KEY_FILE = Path(os.environ.get("PHIAGENT_JEV_V2_KEY_FILE", "/path/to/private-config/credentials/deepseek-harness-20260929"))
JUDGE_UPSTREAM = "https://api.deepseek.com/chat/completions"
TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens",
)
USAGE_FIELDS = (*TOKEN_FIELDS, "cost_usd")
BASE_SOURCE_FILES = (
    "SOURCE_PROVENANCE.json",
    "scripts/run_deepseek_gaia2.py", "scripts/run_deepseek_gaia2_campaign.py",
    "configs/deepseek_gaia2_campaign_v1.json", "tests/test_deepseek_gaia2_runner.py",
    "phiagent/__init__.py", "phiagent/data_engine/__init__.py",
    "phiagent/data_engine/provenance.py", "phiagent/harness/__init__.py",
    "phiagent/harness/task_reasoning.py", "phiagent/harness/tshirt_fold_strategy.py",
    "phiagent/harness/tshirt_positive_reference.py",
    "phiagent/harness/deepseek_bridge.py", "phiagent/harness/deepseek_chisel.py",
    "phiagent/harness/loopback_http.py",
    "phiagent/harness/gaia2_common_prompt.py",
    "phiagent/harness/gaia2_execution_contract.py",
    "phiagent/harness/codex_chisel.py", "phiagent/harness/benchmark_partition.py",
    "phiagent/learning/__init__.py", "phiagent/learning/residual_meta_harness.py",
    "phiagent/learning/residual_meta_training.py",
    "scripts/advance_deepseek_gaia2_common_parent.py",
    "scripts/serve_deepseek_durable_capture.py",
    "tests/test_deepseek_gaia2_common_parent.py",
    "tests/deepseek_harness_plugin.test.mjs", "tests/deepseek_cot_jev.test.mjs",
)


class GateError(RuntimeError):
    pass


def native_app_sdk_template(template: dict, job: dict, config: dict) -> dict:
    """Carry the registered bank intervention across the ARE-to-SDK boundary."""
    result = dict(template)
    if job.get("cot_jev") is not None:
        from integrations.deepseek_harness.cot_jev_policy import validate_native_app_policy
        result["cot_jev"] = validate_native_app_policy(
            job["cot_jev"], treatment=job.get("chisel"),
            encoding=config.get("native_observation_encoding"),
        )
    else:
        result.pop("cot_jev", None)
    if job.get("jev_v2") is not None:
        if job.get("phase") != "train":
            raise GateError("Jev-v2 experimental execution is TRAIN-only until qualification")
        result["jev_v2"] = dict(job["jev_v2"])
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def persist(path: Path, value: Any, *, immutable: bool = False) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise GateError(f"Immutable artifact differs: {path}")
        return
    staging = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.writing")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if immutable:
            os.link(staging, path)
        else:
            staging.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        staging.unlink(missing_ok=True)


def require_remote() -> None:
    if (sys.platform != "linux" or socket.gethostname() != AUTHORIZED_HOST
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise GateError("Only A800-4 with explicit CUDA_VISIBLE_DEVICES='' may execute")


def owned_path(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise GateError("A new absolute owned run path is required")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise GateError("Run paths cannot contain symlinks")
    if not any(path.is_relative_to(root) and path != root
               for root in (CAMPAIGN_ROOT / "runs", DATA0_ROOT)):
        raise GateError("GAIA2 artifacts must be inside an authorized private run root")
    return path


def command_receipt(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"argv": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": command, "error": str(error)}


def resource_inventory() -> dict[str, Any]:
    return {
        "at": utc_now(), "hostname": socket.gethostname(), "CUDA_VISIBLE_DEVICES": "",
        "selected_gpus": [], "mode": "CPU/API; no GPU allocation or unrelated job changes",
        "gpus": command_receipt(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,"
                                "utilization.gpu", "--format=csv"]),
        "gpu_processes": command_receipt([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv",
        ]),
        "memory": command_receipt(["free", "-b"]), "load": command_receipt(["uptime"]),
        "leases": command_receipt(["lslocks", "--json"]),
    }


def source_files(root: Path = ROOT) -> tuple[str, ...]:
    names = set(BASE_SOURCE_FILES)
    names.update(path.relative_to(root).as_posix()
                 for path in (root / "integrations/deepseek_harness").iterdir()
                 if path.suffix in (".py", ".mjs") and path.is_file())
    return tuple(sorted(name for name in names if (root / name).is_file()))


def freeze_source(destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in source_files():
        source = ROOT / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = file_hash(target)
    if not all(name in hashes for name in (
        "scripts/run_deepseek_gaia2.py", "integrations/deepseek_harness/session_worker.py",
        "integrations/deepseek_harness/gaia2_backend.py",
    )):
        raise GateError("The accepted persistent SDK/native backend sources are missing")
    return hashes


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        if not (root / name).is_file() or file_hash(root / name) != digest:
            raise GateError(f"Frozen source changed: {name}")


def provenance(root: Path, command: list[str], seed: int) -> dict[str, Any]:
    return {
        "at": utc_now(), "hostname": socket.gethostname(), "python": sys.version,
        "executable": sys.executable, "command": command, "seed": seed,
        "git_head": command_receipt(["git", "rev-parse", "HEAD"], cwd=root),
        "git_status": command_receipt(
            ["git", "--no-pager", "status", "--short", "--", *source_files(root)], cwd=root,
        ),
        "packages": {dist.metadata["Name"]: dist.version
                     for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
        "cumulative_token_limit": None, "sampling_seed_supported": False,
    }


def initial_public_task(scenario: dict[str, Any]) -> str:
    """The first delivered USER content only; never future turns or oracle events."""
    for event in scenario.get("events", []):
        if event.get("event_type") != "USER":
            continue
        action = event.get("action", {})
        if (action.get("app"), action.get("function")) != (
            "AgentUserInterface", "send_message_to_agent",
        ):
            raise GateError("Unsupported native initial public task action")
        values = [arg["value"] for arg in action.get("args", []) if arg.get("name") == "content"]
        if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
            raise GateError("Native first USER event needs one nonempty textual content")
        return values[0]
    raise GateError("Scenario has no initial public USER event")


def initial_prompt(backend, expected: str, *, render_user=None) -> str | list[dict[str, Any]]:
    """Take one actually delivered user message; leave other events for SDK hooks."""
    env = backend.environment
    while not backend.stop_requested:
        messages = env.notification_system.message_queue.get_by_timestamp(
            datetime.fromtimestamp(env.time_manager.time(), timezone.utc),
        )
        first = next((item for item in messages if item.message_type.value == "USER_MESSAGE"), None)
        for item in messages:
            if item is not first:
                env.notification_system.message_queue.put(item)
        if first is not None:
            delivered = render_user(expected, first.timestamp) if render_user else expected
            if first.message != delivered:
                raise GateError("Delivered initial task differs from its public-only commitment")
            backend.record({
                "kind": "native_notification", "notification_type": "USER_MESSAGE",
                "message": first.message, "timestamp": first.timestamp.isoformat(),
                "attachment_count": len(first.attachments), "initial_task": True,
            })
            parts = [{"type": "text", "text": expected}]
            for attachment in first.attachments:
                if not attachment.mime.startswith("image/"):
                    raise GateError("Native non-image attachment requires an explicit SDK adapter")
                data = attachment.base64_data
                if isinstance(data, bytes):
                    data = data.decode("ascii")
                base64.b64decode(data, validate=True)
                parts.append({"type": "image", "data": data, "mimeType": attachment.mime})
            return parts if first.attachments else expected
        if any(item.message_type.value == "ENVIRONMENT_STOP" for item in messages):
            raise GateError("Native environment stopped before the first user message")
        time.sleep(0.1)
    raise GateError("Native environment stopped before the initial task")


def sdk_usage(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("usage") or {}
    if result.get("usage_complete") is not True:
        return dict.fromkeys(USAGE_FIELDS)
    output, total = raw.get("outputTokens"), raw.get("totalTokens")
    if type(output) is not int or type(total) is not int or not 0 <= output <= total:
        raise GateError("SDK complete-usage receipt is inconsistent")
    return {
        "input_tokens": total - output, "output_tokens": output, "total_tokens": total,
        "cache_read_tokens": raw.get("cacheReadTokens"),
        "cache_write_tokens": raw.get("cacheWriteTokens"), "cost_usd": None,
    }


def sdk_output_limit(result: dict[str, Any]) -> bool:
    """A metered native output limit ends the actor, not the benchmark scorer."""
    capped = (
        result.get("status") == "failed" and result.get("finish_reason") == "max-tokens"
        and result.get("finish_details") == {"kind": "max-tokens"}
        and result.get("usage_complete") is True and not result.get("error")
        and not result.get("cancelled")
    )
    if capped:
        sdk_usage(result)
    return capped


def retained_episode(job: dict) -> dict | None:
    proof = job.get("retained_native")
    if proof is None:
        return None
    if proof.get("classification") != "native_output_limit_requires_offline_validation":
        raise GateError("Only an attested output-limited episode may use retained native scoring")
    for item in proof["evidence"].values():
        path = Path(item["path"])
        if path.is_symlink() or file_hash(path) != item["sha256"]:
            raise GateError("Retained native evidence changed")
    evidence = proof["evidence"]
    request = load(Path(evidence["request"]["path"]))
    actor = load(Path(evidence["sdk_result"]["path"]))
    summary = load(Path(evidence["summary"]["path"]))
    if (
        not sdk_output_limit(actor) or actor.get("closed") is not True
        or actor.get("runtime_closed") is not True
        or request["task"]["task_id"] != job["task_id"]
        or any(request[key] != job[key] for key in ("phase", "arm", "seed"))
        or request.get("final_policy_sha256") != job.get("final_policy_sha256")
        or request.get("source_sha256") != job.get("source_sha256")
        or summary.get("native_score") is not None
        or (summary.get("provider_usage") or {}).get("complete") is not True
    ):
        raise GateError("Retained output-limited episode differs from the frozen request")
    return summary


def chat_usage(response: dict[str, Any]) -> dict[str, Any]:
    raw = response.get("usage") or {}
    values = [raw.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    if (any(type(value) is not int or value < 0 for value in values)
            or values[0] + values[1] != values[2]):
        raise GateError("Judge response omitted consistent provider-native usage")
    hit = raw.get("prompt_cache_hit_tokens")
    miss = raw.get("prompt_cache_miss_tokens")
    if hit is not None and (type(hit) is not int or not 0 <= hit <= values[0]):
        raise GateError("Judge cache-read disclosure is inconsistent")
    if miss is not None and hit is not None and (
        type(miss) is not int or miss < 0 or hit + miss != values[0]
    ):
        raise GateError("Judge cache hit/miss disclosures do not partition prompt tokens")
    return {
        "input_tokens": values[0], "output_tokens": values[1], "total_tokens": values[2],
        "cache_read_tokens": hit, "cache_write_tokens": raw.get("cache_creation_input_tokens"),
        "cost_usd": None,
    }


def add_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: (sum(row[key] for row in rows)
              if all(row.get(key) is not None for row in rows) else None)
        for key in USAGE_FIELDS
    }


def read_secret(path: Path) -> str:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise GateError("A regular owner-only credential file is required")
    value = path.read_text().strip()
    if not value or "\n" in value:
        raise GateError("Invalid credential file")
    return value


def captured_actor_usage(root: Path, actor: dict, sdk_accounting: dict) -> dict:
    """Reconcile every physical wire request without replacing SDK receipts."""
    from scripts.serve_deepseek_durable_capture import native_usage

    rows, values, errors = [], [], []
    for path in sorted((root / "actor").glob("*/RECEIPT.json")):
        receipt = load(path)
        rows.append({"path": str(path), "sha256": file_hash(path),
                     "call_id": receipt.get("call_id"), "route": receipt.get("route")})
        try:
            body, raw = path.with_name("request.body"), path.with_name("response.body")
            if (receipt.get("status") != "MEASURED" or receipt.get("body_rewritten") is not False
                    or file_hash(body) != receipt["request_sha256"]
                    or file_hash(raw) != receipt["response_sha256"]):
                raise GateError("Physical provider response is incomplete or changed")
            usage = native_usage(raw.read_bytes(), receipt["route"], bool(load(body).get("stream")))
            if usage != receipt.get("usage"):
                raise GateError("Stored wire usage differs from raw provider counters")
            values.append(sdk_usage({"usage_complete": True, "usage": usage}))
        except (OSError, KeyError, ValueError, GateError) as error:
            errors.append(f"{path.name}:{type(error).__name__}")
    if actor.get("closed") is not True or actor.get("runtime_closed") is not True:
        errors.append("native-sdk-not-closed")
    expected = sdk_accounting.get("model_call_attempt_count", sdk_accounting.get("attempt_count"))
    if type(expected) is not int or expected != len(rows):
        errors.append("physical-request-count-does-not-reconcile-with-sdk")
    known = add_usage(values)
    if actor.get("usage_complete") is True and known != sdk_usage(actor):
        errors.append("complete-sdk-and-wire-counters-differ")
    complete = not errors
    return {"complete": complete, "attempts": len(rows), "receipts": rows,
            "usage": known if complete else dict.fromkeys(USAGE_FIELDS),
            "known_reported_usage": known, "errors": errors, "unknown_is_zero": False,
            "authority": "raw complete provider responses; independent SDK cross-check"}


def model_failure_scoring_trace(native: dict, actor: dict, output: Path, task_id: str) -> dict | None:
    """Admit a closed model failure to the real scorer, never a wrapper verdict."""
    if (actor.get("closed") is not True or actor.get("runtime_closed") is not True
            or actor.get("cancelled") or native.get("success") not in (None, False)
            or not str(native.get("exception") or "").startswith("Official SDK native turn failed:")):
        return None
    malformed = "DeepSeek Messages stream: tool input is invalid JSON" in str(native["exception"])
    if not malformed and not sdk_output_limit(actor):
        return None
    trace = Path(native.get("export_path") or "")
    if (not trace.is_absolute() or trace.is_symlink()
            or not trace.is_relative_to(output / "native/traces") or not trace.is_file()):
        raise GateError("Failed native actor has no source-bound retained trace")
    exported = load(trace)
    if (exported.get("metadata", {}).get("definition", {}).get("scenario_id") != task_id
            or not isinstance(exported.get("completed_events"), list)):
        raise GateError("Failed native actor trace identity or completed events differ")
    calls = list((output / "native/bridge/calls").glob("*/STARTED.json"))
    if any(not path.with_name("RESULT.json").is_file() for path in calls):
        raise GateError("An unsettled native mutation cannot enter retained scoring")
    return {"path": str(trace), "sha256": file_hash(trace),
            "classification": "native_model_malformed_json" if malformed else "native_output_limit",
            "new_actor_calls": 0, "tool_actions_replayed": False}


def load_native_scoring_scenario(loader, scenario_path: Path, retained: dict | None):
    """Use retained completed events with the original frozen native oracle."""
    source = Path(retained["path"]) if retained else scenario_path
    if retained and file_hash(source) != retained["sha256"]:
        raise GateError("Retained native trace changed inside the read-only sandbox")
    scenario, replay = loader(source.read_text(), str(source), load_completed_events=bool(retained))
    if retained:
        original, unused = loader(scenario_path.read_text(), str(scenario_path), load_completed_events=False)
        if (original is None or scenario is None or original.scenario_id != scenario.scenario_id or unused):
            raise GateError("The retained scorer must use the original frozen native oracle")
        scenario = original
    return scenario, replay


class SessionIPC(AbstractContextManager):
    """Persistent official SDK worker; the ARE interpreter never imports its SDK."""

    def __init__(self, python: str, runner: Path, directory: Path, *, timeout_seconds=2100):
        self.timeout_seconds = timeout_seconds
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        (directory / "cache").mkdir()
        (directory / "home").mkdir()
        self.sequence = 0
        self.lock = threading.Lock()
        self.pending: dict[str, queue.Queue] = {}
        self.closed = False
        self.result = None
        self.log = (directory / "worker.stderr.log").open("xb")
        command = [python, "-u", str(runner), "--session-worker"]
        persist(directory / "COMMAND.json", {"argv": command, "CUDA_VISIBLE_DEVICES": "",
                                             "cumulative_token_limit": None}, immutable=True)
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, cwd=directory,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": str(directory / "home"),
                 "TMPDIR": str(directory / "cache"), "PYTHONPATH": str(runner.parents[2]),
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                 "CUDA_VISIBLE_DEVICES": ""},
        )
        persist(directory / "PROCESS.json", {"pid": self.process.pid}, immutable=True)
        self.reader = threading.Thread(target=self._read, name="official-sdk-jsonl", daemon=True)
        self.reader.start()

    def _read(self) -> None:
        error = "SDK worker exited without a settled reply"
        try:
            for line in self.process.stdout:
                reply = json.loads(line)
                if reply.get("protocol_version") != 1:
                    raise GateError("Unsupported official SDK IPC protocol")
                with self.lock:
                    target = self.pending.get(reply.get("id"))
                if target is None:
                    raise GateError("Unmatched SDK IPC reply; no replay is permitted")
                target.put(reply)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                for target in self.pending.values():
                    target.put({"ok": False, "error": error})

    def exchange(self, op: str, **payload) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                raise GateError("The SDK worker is closed")
            self.sequence += 1
            ident = f"request-{self.sequence}"
            target: queue.Queue = queue.Queue()
            self.pending[ident] = target
            request = {"id": ident, "op": op, **payload}
            self.process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
            self.process.stdin.flush()
        try:
            reply = target.get(timeout=self.timeout_seconds)
            if reply.get("id") not in (None, ident) or reply.get("ok") is not True:
                raise GateError(f"Official SDK {op} failed: {reply.get('error')}")
            return reply
        except queue.Empty:
            raise GateError("SDK IPC exceeded the native scenario timeout; no replay") from None
        finally:
            with self.lock:
                self.pending.pop(ident, None)

    def close(self) -> dict[str, Any] | None:
        if self.closed:
            return self.result
        try:
            if self.process.poll() is None:
                self.result = self.exchange("shutdown").get("result")
        finally:
            self.closed = True
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.reader.join(timeout=5)
            self.log.close()
        return self.result

    def __exit__(self, *_args):
        self.close()
        return False


class Supervisor(AbstractContextManager):
    """Host-only SDK and raw judge transport, never exposed as model tools."""

    def __init__(self, config: dict, job: dict, output: Path):
        self.config, self.job, self.output = config, job, output
        self.secret = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.judge_sequence = 0
        self.worker: SessionIPC | None = None
        self.actor = None
        self.failure = None
        self.capture = None
        self.capture_closed = False
        self.capture_secret_path = output / "provider-capture.secret"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SupervisorHandler)
        self.server.owner = self
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.secret_path = output / "native" / "supervisor.secret"
        with self.secret_path.open("x") as stream:
            stream.write(self.secret + "\n")
        self.secret_path.chmod(0o600)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        rpc(self.url, self.secret, "/health", {})
        return self

    def session(self, operation: str, request: dict) -> dict:
        if operation == "init":
            if set(request) != {"tools", "bridge", "public_system_context"}:
                raise GateError("Invalid native SDK initialization fields")
            with self.lock:
                if self.worker is not None:
                    raise GateError("Only one persistent SDK session may own a native episode")
                template = load(Path(self.config["runtime_template"]))
                template.update({
                    "purpose": "benchmark", "run_dir": str(self.output / "sdk"),
                    "session_id": "gaia-" + canonical_hash(str(self.output))[:40],
                    "seed": self.job["seed"], "model": self.config["model"],
                    "max_tokens": self.config["max_output_tokens"],
                    "api_key_file": str(KEY_FILE), "tools": request["tools"],
                    "bridge": request["bridge"],
                    "benchmark_context": request["public_system_context"],
                    "arm": "chisel" if self.job.get("chisel") else "baseline",
                    "chisel": self.job.get("chisel"),
                    "source_revision": self.job.get("source_revision"),
                })
                template.pop("prompt", None)
                template.pop("cwd", None)
                template = native_app_sdk_template(template, self.job, self.config)
                if self.config.get("provider_durable_capture") is True:
                    from scripts.serve_deepseek_durable_capture import start_capture

                    capture_secret = secrets.token_urlsafe(32)
                    self.capture_secret_path.write_text(capture_secret + "\n")
                    self.capture_secret_path.chmod(0o600)
                    self.capture = start_capture(
                        self.output / "provider-capture", read_secret(KEY_FILE), capture_secret,
                        attribution=({'execution_id': str(self.output), 'task_id': self.job['task_id'],
                          'phase':self.job['phase'], 'arm':self.job.get('ablation_arm',self.job['arm'])}
                          if self.config.get('jev_cost_instrumentation') else None),
                        execution_guards=(self.job['jev_v2']['guards']
                          if self.job.get('jev_v2',{}).get('arm') == 'A6' else None),
                    )
                    origin = f"http://127.0.0.1:{self.capture.server_port}/actor"
                    template.update(api_key_file=str(self.capture_secret_path), provider_capture={
                        "base_url": origin + "/anthropic", "fast_url": origin + "/chat/completions",
                    })
                if "native_observation_encoding" in self.config:
                    template["native_observation_encoding"] = self.config[
                        "native_observation_encoding"
                    ]
                self.worker = SessionIPC(
                    self.config["sdk_python"], ROOT / "integrations/deepseek_harness/runner.py",
                    self.output / "ipc",
                    timeout_seconds=self.config["scenario_timeout_seconds"] + 240,
                )
                return self.worker.exchange("init", job=template)
        if self.worker is None:
            raise GateError("SDK session has not been initialized")
        if operation == "run" and set(request) == {"prompt"}:
            return self.worker.exchange("run", prompt=request["prompt"])
        if operation == "shutdown" and not request:
            self.actor = self.worker.close()
            return {"result": self.actor}
        raise GateError("Invalid SDK session operation")

    def judge(self, body: bytes) -> tuple[int, bytes]:
        request = json.loads(body)
        if request.get("model") != MODEL or request.get("stream") is True:
            raise GateError("Only the frozen native non-streaming DeepSeek judge is admitted")
        with self.lock:
            self.judge_sequence += 1
            sequence = self.judge_sequence
        folder = self.output / "judge" / f"{sequence:06d}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "request.json").write_bytes(body)
        persist(folder / "STARTED.json", {
            "at": utc_now(), "role": "judge", "endpoint": JUDGE_UPSTREAM,
            "request_sha256": hashlib.sha256(body).hexdigest(), "provider_attempt": sequence,
        }, immutable=True)
        key = read_secret(KEY_FILE)
        status_code, raw, headers = None, b"", {}
        try:
            outgoing = Request(JUDGE_UPSTREAM, data=body, headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json",
            }, method="POST")
            try:
                with build_opener(ProxyHandler({})).open(outgoing, timeout=180) as response:
                    status_code, raw = response.status, response.read()
                    headers = dict(response.headers.items())
            except HTTPError as error:
                status_code, raw = error.code, error.read()
                headers = dict(error.headers.items())
            (folder / "response.json").write_bytes(raw)
            decoded = json.loads(raw)
            usage = chat_usage(decoded) if status_code == 200 else dict.fromkeys(USAGE_FIELDS)
            persist(folder / "RESULT.json", {
                "status": status_code, "at": utc_now(), "usage": usage,
                "provider_native_usage": decoded.get("usage"),
                "response_model": decoded.get("model"),
                "system_fingerprint": decoded.get("system_fingerprint"),
                "headers": {name: value for name, value in headers.items()
                            if name.lower() not in ("set-cookie", "authorization")},
                "accounting_complete": status_code == 200,
            }, immutable=True)
            if status_code != 200:
                self.failure = "Native judge provider request failed; its costs are retained"
            return status_code, raw
        except Exception as error:
            self.failure = f"Judge transport/accounting: {type(error).__name__}"
            persist(folder / "FAILURE.json", {
                "type": type(error).__name__, "message": str(error).replace(key, "[REDACTED]"),
                "usage": None, "status": status_code, "retry_by_adapter": False,
            }, immutable=True)
            raise

    def accounting(self) -> dict[str, Any]:
        actor_path = self.output / "sdk/result.json"
        actor = load(actor_path) if actor_path.is_file() else (self.actor or {})
        actor_never_started = self.worker is None and not (self.output / "sdk").exists()
        actor_accounting_path = self.output / "sdk/usage.json"
        actor_accounting = load(actor_accounting_path) if actor_accounting_path.is_file() else {}
        judge_rows = []
        judge_complete = True
        for started in sorted((self.output / "judge").glob("*/STARTED.json")):
            result = started.with_name("RESULT.json")
            if not result.is_file():
                judge_complete = False
                judge_rows.append(dict.fromkeys(USAGE_FIELDS))
            else:
                row = load(result)
                judge_rows.append(row["usage"])
                judge_complete &= row.get("accounting_complete") is True
        actor_usage = dict.fromkeys(USAGE_FIELDS, 0) if actor_never_started else sdk_usage(actor)
        actor_complete = actor_never_started or actor.get("usage_complete") is True
        wire = None
        if self.capture is not None:
            wire = captured_actor_usage(self.output / "provider-capture", actor, actor_accounting)
            actor_usage, actor_complete = wire["usage"], wire["complete"]
        judge_usage = add_usage(judge_rows)
        known_judges = [load(path)["usage"] for path in (self.output / "judge").glob("*/RESULT.json")
                       if load(path).get("accounting_complete") is True]
        judge_reasoning = [
            ((load(path).get("provider_native_usage") or {}).get("completion_tokens_details")
             or {}).get("reasoning_tokens")
            for path in (self.output / "judge").glob("*/RESULT.json")
        ]
        return {
            "actor": {
                "usage": actor_usage,
                "complete": actor_complete,
                "attempts": (wire["attempts"] if wire is not None else
                             (0 if actor_never_started else actor_accounting.get("attempt_count"))),
                "durable_wire_evidence": wire,
                "provider_native_usage": actor.get("usage"),
                "known_reported_usage": actor_accounting.get("known_reported_usage"),
                "reasoning_tokens": (actor.get("usage") or {}).get("reasoningTokens"),
                "provider_never_started": actor_never_started,
            },
            "judge": {
                "usage": judge_usage, "complete": judge_complete, "attempts": len(judge_rows),
                "known_usage": add_usage(known_judges),
                "reasoning_tokens": (sum(judge_reasoning) if judge_complete and all(
                    type(value) is int and value >= 0 for value in judge_reasoning
                ) else None),
            },
            "combined": add_usage([actor_usage, judge_usage]),
            "complete": actor_complete and judge_complete,
            "unknown_is_zero": False, "actor_and_judge_are_disjoint": True,
            "money_status": "Not derived without versioned provider pricing",
        }

    def settle_actor(self):
        if self.worker is not None:
            self.actor = self.worker.close()
        if self.capture is not None and not self.capture_closed:
            self.capture.shutdown()
            self.capture.server_close()
            self.capture_closed = True
            self.capture_secret_path.unlink(missing_ok=True)

    def __exit__(self, *_args):
        try:
            self.settle_actor()
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=10)
            self.secret_path.unlink(missing_ok=True)
        return False


class SupervisorHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        owner: Supervisor = self.server.owner
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + owner.secret):
            self.reply(403, {"error": "Owner-only supervisor authentication required"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 64 * 1024 * 1024:
                raise GateError("Invalid native RPC payload size")
            self.connection.settimeout(owner.config["scenario_timeout_seconds"] + 240)
            body = self.rfile.read(size)
            if len(body) != size:
                raise GateError("Truncated native RPC payload")
            if self.path == "/judge/v1/chat/completions":
                status_code, raw = owner.judge(body)
                self.reply(status_code, raw)
            elif self.path.startswith("/session/"):
                self.reply(200, owner.session(self.path.removeprefix("/session/"), json.loads(body)))
            elif self.path == "/health":
                self.reply(200, {"status": "READY", "runtime": "official_deepseek_harness_sdk"})
            else:
                self.reply(404, {"error": "Unknown endpoint"})
        except Exception as error:
            owner.failure = f"Supervisor: {type(error).__name__}: {error}"
            self.reply(500, {"error": owner.failure.replace(owner.secret, "[REDACTED]")})

    def reply(self, status_code: int, value: Any):
        data = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def rpc(url: str, secret: str, path: str, value: dict, *, timeout_seconds=2100) -> dict:
    request = Request(url + path, data=json.dumps(value, ensure_ascii=False).encode(), headers={
        "Authorization": "Bearer " + secret, "Content-Type": "application/json",
    }, method="POST")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").replace(secret, "[REDACTED]")
        raise GateError(f"Native supervisor HTTP {error.code}: {detail}") from None


def filesystem_boundary(environment, native: Path, scenario_path: Path) -> dict:
    """Check the actual native filesystem, not a replacement file-tool surface."""
    scratch = (native / "cache").resolve()
    systems = {}
    for tool in environment.get_tools():
        app = tool.class_instance
        if hasattr(app, "tmpdir") and hasattr(app, "_validate_path"):
            systems[id(app)] = app
    checks = []
    for app in systems.values():
        root = Path(app.tmpdir).resolve()
        if not root.is_relative_to(scratch) or root == scratch:
            raise GateError("Native filesystem was not created inside the fresh owned cache")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise GateError("Native filesystem contains a symlink outside the frozen asset contract")
        candidates = ["/", "/path/to/private-config/credentials/" + KEY_FILE.name,
                      str(scenario_path), "/proc/self/environ", "../../" + KEY_FILE.name,
                      "../" + root.name + "_prefix-check"]
        for candidate in candidates:
            try:
                normalized = Path(app._validate_path(candidate)).resolve()
            except PermissionError:
                continue
            if not normalized.is_relative_to(scratch):
                raise GateError("Native filesystem normalization escaped its isolated scratch mount")
            if normalized == scenario_path or normalized == KEY_FILE:
                raise GateError("Native filesystem could reach evaluator/credential paths")
        checks.append({"app": app.name, "root": str(root), "owned_boundary": str(scratch)})
    if KEY_FILE.exists():
        raise GateError("The provider credential must not be mounted into native ARE")
    if not checks:
        raise GateError("Expected native Files sandbox is absent")
    return {
        "status": "PASSED", "filesystems": checks, "provider_credential_mounted": False,
        "native_tools_unchanged": True, "source_readonly": True, "assets_readonly": True,
        "mount_and_pid_isolation": True, "private_scratch_only_writable": True,
    }


def sandbox_command(
    config: dict, native: Path, scenario: Path, retained_trace: Path | None = None,
) -> list[str]:
    runtime = Path(config["are_python"]).parent.parent
    source = ROOT
    command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid",
               "--unshare-ipc", "--unshare-uts", "--cap-drop", "ALL", "--clearenv"]
    for name in ("/usr", "/lib", "/lib64"):
        if Path(name).exists():
            command += ["--ro-bind", name, name]
    command += ["--proc", "/proc", "--dev", "/dev"]
    for name in ("/etc/ssl/certs", "/etc/resolv.conf", "/etc/localtime",
                 "/etc/passwd", "/etc/group", "/etc/hosts", "/etc/mime.types"):
        if Path(name).exists():
            command += ["--ro-bind", name, name]
    for path in (runtime, source, Path(config["filesystem"]), scenario):
        command += ["--ro-bind", str(path), str(path)]
    if retained_trace is not None:
        command += ["--ro-bind", str(retained_trace), str(retained_trace)]
    command += ["--bind", str(native), str(native)]
    environment = {
        "PATH": "/usr/bin:/bin", "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "", "HOME": str(native / "home"),
        "TMPDIR": str(native / "cache"), "FS_PATH": str(native / "fs-states"),
        "DEMO_FS_PATH": config["filesystem"], "PYTHON_DOTENV_DISABLED": "1",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(native / "cache"), "HF_HOME": str(native / "cache/huggingface"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DO_NOT_TRACK": "1",
    }
    for key, value in environment.items():
        command += ["--setenv", key, value]
    return command + [
        "--chdir", str(native), "--", config["are_python"], "-u",
        str(source / "scripts/run_deepseek_gaia2.py"), "--native-job", str(native / "JOB.json"),
    ]


def run_native(job_path: Path) -> int:
    require_remote()
    job = load(job_path)
    native = job_path.parent
    secret = read_secret(native / "supervisor.secret")
    url = job["supervisor_url"]
    scenario_path = Path(job["scenario"])
    if any(name.startswith("deepseek_harness") for name in sys.modules):
        raise GateError("SDK modules must never enter the incompatible native ARE interpreter")
    from are.simulation.agents.agent_config_builder import AgentConfigBuilder
    from are.simulation.agents.agent_execution_result import AgentExecutionResult
    from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
    from are.simulation.agents.default_agent.prompts import get_notification_system_prompt
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine, LiteLLMModelConfig
    from are.simulation.apps.agent_user_interface import AUIMessage, Sender
    from are.simulation.benchmark.scenario_loader import load_scenario
    from are.simulation.scenario_runner import ScenarioRunner
    from are.simulation.scenarios.config import ScenarioRunnerConfig
    from are.simulation.scenarios.scenario_imported_from_json.utils import (
        get_scenario_duration, preprocess_scenario,
    )
    from are.simulation.validation.configs import GraphPerEventJudgeConfig
    from integrations.deepseek_harness.gaia2_backend import Gaia2Backend
    from phiagent.harness.deepseek_bridge import ToolBridge
    from phiagent.harness.gaia2_common_prompt import public_context

    class NativeConfig(AgentConfigBuilder):
        def build(self, agent_name, **kwargs):
            if agent_name != "official_deepseek_harness_sdk":
                raise GateError("This runner admits only the official DeepSeek actor")
            return super().build(agent_name="default", **kwargs)

    class NativeAgent(RunnableARESimulationAgent):
        def __init__(self, env):
            self.env = env

        def run_scenario(self, scenario, notification_system=None, initial_agent_logs=None):
            if initial_agent_logs:
                raise GateError("A fresh native trial cannot replay an old trajectory")
            backend = Gaia2Backend(self.env, native / "backend")
            persist(native / "FILESYSTEM_BOUNDARY.json",
                    filesystem_boundary(self.env, native, scenario_path), immutable=True)
            prompt = initial_prompt(
                backend, job["public_prompt"], render_user=lambda content, timestamp: str(
                    AUIMessage(sender=Sender.USER, content=content, timestamp=timestamp.timestamp(),
                               time_read=timestamp.timestamp()),
                ),
            )
            current = datetime.fromtimestamp(self.env.time_manager.time(), timezone.utc).isoformat()
            context, common_receipt = public_context(
                current,
                get_notification_system_prompt(notification_system, scenario.apps),
                scenario.additional_system_prompt,
                job.get("shared_gaia2_prompt"),
            )
            persist(native / "PUBLIC_CONTEXT.json", {
                "text": context, "initial_public_prompt_sha256": hashlib.sha256(
                    job["public_prompt"].encode()).hexdigest(),
                "future_user_events_in_prompt": False, "oracle_events_in_prompt": False,
                **({"shared_prompt": common_receipt} if common_receipt is not None else {}),
            }, immutable=True)
            from phiagent.harness.gaia2_execution_contract import user_turn_limit

            limit = user_turn_limit(scenario.nb_turns, job["native_max_turns"])
            final, session_started, turns, finish_kind = None, False, [], "completed"
            with ToolBridge(native / "bridge", backend, backend.catalog) as bridge:
                self.env.pause()
                try:
                    rpc(url, secret, "/session/init", {
                        "tools": backend.catalog,
                        "bridge": {"url": bridge.url, "secret_file": str(bridge.secret_file),
                                   "notifications": True, "generation_timing": True},
                        "public_system_context": context,
                    })
                    session_started = True
                finally:
                    self.env.resume_with_offset(0.0)
                    backend.record({"kind": "initialization_clock",
                                    "convention": "setup_paused_zero_offset"})
                try:
                    while not backend.stop_requested and (
                        limit is None or backend.answered_turns < limit
                    ):
                        if prompt is None:
                            messages = backend.notifications()
                            if backend.stop_requested:
                                break
                            if not messages:
                                time.sleep(0.1)
                                continue
                            parts = []
                            for message in messages:
                                parts.extend([{"type": "text", "text": message}]
                                             if isinstance(message, str) else message)
                            prompt = parts
                        backend.turn_state = "running"
                        turn = rpc(url, secret, "/session/run", {"prompt": prompt},
                                   timeout_seconds=job.get("scenario_timeout_seconds", 1860) + 240)["result"]
                        turns.append(turn)
                        persist(native / "SDK_TURNS.json", turns)
                        if turn.get("status") != "completed":
                            if sdk_output_limit(turn):
                                finish_kind = "max-tokens"
                                backend.record({
                                    "kind": "native_actor_output_limit", "actor_resampled": False,
                                    "native_scoring_required": True,
                                })
                                break
                            detail = turn.get("error") or turn.get("finish_details") or {
                                "status": turn.get("status"),
                            }
                            raise GateError(
                                "Official SDK native turn failed: "
                                + json.dumps(detail, sort_keys=True, ensure_ascii=False),
                            )
                        final = turn.get("final_response")
                        if not backend.stop_requested and not turn.get("stop_requested"):
                            backend.finish_response(final)
                        if turn.get("stop_requested"):
                            break
                        prompt = None
                finally:
                    if session_started:
                        closed = rpc(url, secret, "/session/shutdown", {})["result"]
                        persist(native / "SDK_EPISODE.json", closed)
            persist(native / "ACTOR_TERMINATION.json", {
                "scenario_nb_turns": scenario.nb_turns,
                "fallback_user_turn_limit": job["native_max_turns"],
                "termination_reason": (
                    "native_environment_stop" if backend.stop_requested
                    else "native_user_turn_limit" if limit is not None and backend.answered_turns >= limit
                    else finish_kind
                ),
                "native_max_user_turns": limit, "answered_turns": backend.answered_turns,
                "sdk_turns": len(turns), "environment_stop": backend.stop_requested,
                "SystemApp_wait_is_episode_end": False,
                "same_official_session_across_turns": True,
                "finish_kind": finish_kind, "output_limit_retries": 0,
            }, immutable=True)
            return AgentExecutionResult(output=final)

    class NativeBuilder:
        def build(self, agent_config, env=None, mock_responses=None):
            if job.get("retained_trace"):
                raise GateError("Retained native validation must never instantiate an actor")
            if env is None or mock_responses is not None:
                raise GateError("A real env-bound native tool surface is required")
            return NativeAgent(env)

    retained = job.get("retained_trace")
    scenario, replay = load_native_scoring_scenario(load_scenario, scenario_path, retained)
    if (scenario is None or scenario.scenario_id != job["task_id"]
            or (not retained and replay) or (retained and replay is None)):
        raise GateError("Native scenario load or frozen identity failed")
    config = ScenarioRunnerConfig(
        agent="official_deepseek_harness_sdk", model=MODEL, model_provider="local",
        export=True, output_dir=str(native / "traces"),
        simulated_generation_time_mode="measured",
        judge_only=bool(retained),
        max_turns=job["native_max_turns"],
    )
    judge = LiteLLMEngine(LiteLLMModelConfig(
        model_name="openai/" + MODEL, provider="local",
        endpoint=url + "/judge/v1", api_key=secret,
    ))
    if job.get("native_execution_contract") is not None:
        from types import MethodType
        from phiagent.harness.gaia2_execution_contract import (
            native_judge_completion, validate_contract,
        )

        validate_contract(job["native_execution_contract"])
        judge.chat_completion = MethodType(native_judge_completion, judge)
    preprocess_scenario(
        scenario, judge_config=GraphPerEventJudgeConfig(engine=judge),
        max_scenario_duration=get_scenario_duration(
            scenario, config.max_time_scenario_duration, config.max_scenario_duration,
        ),
        offline_validation=bool(retained),
    )
    if scenario.judge is None:
        raise GateError("Official graph-per-event judge was not initialized")
    persist(native / "NATIVE_CONFIG.json", {
        "config": config.model_dump(mode="json"), "scenario_seed": scenario.seed,
        "scenario_duration": scenario.duration, "native_nb_turns": scenario.nb_turns,
        "judge": {"implementation": "unmodified LiteLLMEngine/GraphPerEventJudge",
                  "provider_wire_api": "native_openai_chat_completions", "model": MODEL},
        "retained_trace": retained, "new_actor_calls": 0 if retained else None,
        "native_execution_contract": job.get("native_execution_contract"),
        "retained_completed_events": len(replay) if retained else None,
    }, immutable=True)
    result = ScenarioRunner(
        agent_builder=NativeBuilder(), agent_config_builder=NativeConfig(),
    ).run(config, scenario, completed_events=replay if retained else None)
    persist(native / "RESULT.json", {
        "success": result.success, "exception": str(result.exception) if result.exception else None,
        "exception_type": type(result.exception).__name__ if result.exception else None,
        "rationale": result.rationale, "duration": result.duration, "export_path": result.export_path,
    }, immutable=True)
    return 0 if type(result.success) is bool and result.exception is None else 2


def run_episode(config: dict, job: dict, output: Path) -> dict:
    require_remote()
    owned_path(output).mkdir(parents=True, exist_ok=False, mode=0o700)
    os.umask(0o077)
    for name in ("native", "native/cache", "native/home", "native/fs-states"):
        (output / name).mkdir(exist_ok=True, mode=0o700)
    persist(output / "JOB.json", job, immutable=True)
    persist(output / "CONFIG.json", config, immutable=True)
    persist(output / "RESOURCE_SELECTION.json", resource_inventory(), immutable=True)
    persist(output / "PROVENANCE.json", provenance(ROOT, sys.argv, job["seed"]), immutable=True)
    result = {
        "status": "BLOCKED", "task_id": job["task_id"], "phase": job["phase"],
        "arm": job["arm"], "seed": job["seed"], "native_score": None, "success": None,
        "runtime": "official_deepseek_harness_sdk", "model": MODEL,
        "actor_wire_api": "anthropic_messages", "judge_wire_api": "chat_completions",
        "started_at": utc_now(), "output": str(output), "valid_for_comparison": False,
        "cumulative_token_limit": None, "sampling_seed_supported": False,
    }
    supervisor, inherited_episode = None, None
    try:
        if importlib.metadata.version("meta-agents-research-environments") != ARE_VERSION:
            raise GateError("Use the pinned native ARE 1.1.0 runtime")
        if config.get("cumulative_token_limit") is not None or config["model"] != MODEL:
            raise GateError("Only the frozen DeepSeek model and unlimited cumulative usage are valid")
        if file_hash(Path(config["filesystem_manifest"])) != config["filesystem_manifest_sha256"]:
            raise GateError("The pinned native filesystem manifest changed")
        scenario_path = Path(job["scenario"])
        if file_hash(scenario_path) != job["scenario_sha256"]:
            raise GateError("Frozen native scenario changed")
        if initial_public_task(load(scenario_path)) != job["public_prompt"]:
            raise GateError("Actor task is not the first public USER content")
        inherited_episode = retained_episode(job)
        if job["phase"] == "test":
            freeze = load(Path(job["freeze_manifest"]))
            if (freeze["policy_sha256"] != job["final_policy_sha256"]
                    or freeze["source_sha256"] != job["source_sha256"]):
                raise GateError("Sealed test lacks its exact final policy/source freeze")
        with Supervisor(config, job, output) as supervisor:
            native_job = {
                "supervisor_url": supervisor.url, "scenario": str(scenario_path),
                "task_id": job["task_id"], "public_prompt": job["public_prompt"],
                "native_max_turns": config["native_max_turns"],
                "scenario_timeout_seconds": config["scenario_timeout_seconds"],
            }
            if "native_execution_contract" in config:
                from phiagent.harness.gaia2_execution_contract import validate_contract

                native_job["native_execution_contract"] = validate_contract(config["native_execution_contract"])
            if "shared_gaia2_prompt" in config:
                from phiagent.harness.gaia2_common_prompt import validate_prompt_identity

                native_job["shared_gaia2_prompt"] = validate_prompt_identity(
                    config["shared_gaia2_prompt"],
                )
            retained_trace = None
            if inherited_episode is not None:
                proof = job["retained_native"]
                native_job["retained_trace"] = proof["evidence"]["trace"]
                retained_trace = Path(proof["evidence"]["trace"]["path"])
            persist(output / "native/JOB.json", native_job, immutable=True)
            command = sandbox_command(config, output / "native", scenario_path, retained_trace)
            persist(output / "COMMAND.json", {"argv": command, "cumulative_token_limit": None},
                    immutable=True)
            with (output / "native-process.log").open("xb") as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            persist(output / "NATIVE_PROCESS.json", {
                "pid": process.pid, "started_at": utc_now(),
                "outer_wall_timeout_seconds": config["scenario_timeout_seconds"],
            }, immutable=True)
            try:
                code = process.wait(timeout=config["scenario_timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                persist(output / "OUTER_TIMEOUT.json", {
                    "limit_seconds": config["scenario_timeout_seconds"],
                    "termination_reason": "supervisor_wall_timeout", "actor_replayed": False,
                    "native_score": None,
                }, immutable=True)
                raise GateError(
                    f"Native scenario exceeded registered {config['scenario_timeout_seconds']}-second supervisor timeout"
                ) from None
            persist(output / "PROCESS_EXIT.json", {"returncode": code}, immutable=True)
            native_result = output / "native/RESULT.json"
            if not native_result.is_file():
                raise GateError("Native ARE exited without a durable official verdict")
            native = load(native_result)
            model_failure_scored = False
            if (config.get("provider_durable_capture") is True
                    and inherited_episode is None and native.get("exception") is not None):
                supervisor.settle_actor()
                actor = load(output / "sdk/result.json")
                proof = model_failure_scoring_trace(native, actor, output, job["task_id"])
                if proof is not None:
                    if (supervisor.accounting().get("actor") or {}).get("complete") is not True:
                        raise GateError("Retained model failure requires complete physical actor accounting")
                    scorer = output / "native-model-failure-score"
                    for directory in (scorer, scorer / "cache", scorer / "home", scorer / "fs-states"):
                        directory.mkdir(mode=0o700)
                    (scorer / "supervisor.secret").write_text(supervisor.secret + "\n")
                    (scorer / "supervisor.secret").chmod(0o600)
                    score_job = {**native_job, "retained_trace": {
                        "path": proof["path"], "sha256": proof["sha256"],
                    }}
                    persist(scorer / "JOB.json", score_job, immutable=True)
                    persist(scorer / "ADMISSION.json", {
                        **proof, "original_native_result": str(native_result),
                        "original_native_result_sha256": file_hash(native_result),
                        "original_actor_sha256": file_hash(output / "sdk/result.json"),
                    }, immutable=True)
                    score_command = sandbox_command(config, scorer, scenario_path, Path(proof["path"]))
                    persist(scorer / "COMMAND.json", {"argv": score_command}, immutable=True)
                    with (scorer / "process.log").open("xb") as stream:
                        score_process = subprocess.Popen(score_command, stdin=subprocess.DEVNULL,
                                                         stdout=stream, stderr=subprocess.STDOUT)
                        try:
                            score_code = score_process.wait(timeout=config["scenario_timeout_seconds"])
                        except subprocess.TimeoutExpired:
                            score_process.terminate()
                            score_process.wait(timeout=30)
                            raise GateError("Native retained-event scoring exceeded its registered wall time")
                    persist(scorer / "PROCESS_EXIT.json", {"returncode": score_code}, immutable=True)
                    (scorer / "supervisor.secret").unlink(missing_ok=True)
                    native_result = scorer / "RESULT.json"
                    native = load(native_result)
                    result["native_model_failure_scoring"] = proof
                    model_failure_scored = True
            result["native_result"] = str(native_result)
            result["native_exception"] = native.get("exception")
            result["native_raw_success"] = native.get("success")
            if native.get("exception") is None and type(native.get("success")) is bool:
                result.update(native_score=int(native["success"]), success=native["success"])
            if native.get("exception") is not None or type(native.get("success")) is not bool:
                raise GateError("Native validation was unresolved; null verdict is infrastructure")
            if supervisor.failure:
                raise GateError(supervisor.failure)
            if inherited_episode is None:
                actor = load(output / "sdk/result.json")
                if ((actor.get("status") != "completed" and not sdk_output_limit(actor)
                     and not model_failure_scored)
                        or actor.get("closed") is not True
                        or actor.get("runtime_closed") is not True):
                    raise GateError("The official SDK episode did not close successfully")
            accounting = supervisor.accounting()
            if accounting["complete"] is not True:
                raise GateError("Actor/judge accounting is incomplete; never replace it with zero")
            result.update(status="COMPLETED", valid_for_comparison=True,
                          actor_result=(
                              job["retained_native"]["evidence"]["sdk_result"]["path"]
                              if inherited_episode is not None else str(output / "sdk/result.json")
                          ))
    except Exception as error:
        result.update(error_type=type(error).__name__, blocker=str(error))
    finally:
        if supervisor is not None:
            try:
                result["provider_usage"] = supervisor.accounting()
            except Exception as error:
                result["provider_usage"] = None
                result["accounting_error"] = f"{type(error).__name__}: {error}"
                result["status"], result["valid_for_comparison"] = "BLOCKED", False
        else:
            result["provider_usage"] = None
        if inherited_episode is not None:
            result["retained_native"] = job["retained_native"]
            result["execution_usage"] = add_usage([
                inherited_episode["provider_usage"]["combined"],
                (result.get("provider_usage") or {}).get("combined") or dict.fromkeys(USAGE_FIELDS),
            ])
            result["new_actor_calls"] = 0
        result["finished_at"] = utc_now()
        persist(output / "SUMMARY.json", result, immutable=True)
    return result


def first_training_job(config: dict) -> dict:
    partition = Path(config["partition_dir"])
    if file_hash(partition / "SPLIT_MANIFEST.json") != config["partition_sha256"]:
        raise GateError("Frozen population partition changed")
    if file_hash(partition / "TRAIN_VALIDATION.json") != config["train_validation_sha256"]:
        raise GateError("Frozen development partition changed")
    manifest = load(partition / "SPLIT_MANIFEST.json")
    tasks = load(partition / "TRAIN_VALIDATION.json")["train"]
    task = tasks[0]
    if task["id"] != manifest["splits"]["train"][0] or task["id"] != "scenario_universe_30_erta5d":
        raise GateError("Acceptance must use the first hash-ranked TRAIN scenario")
    return {
        "task_id": task["id"], "scenario": task["path"], "scenario_sha256": task["sha256"],
        "public_prompt": initial_public_task(load(Path(task["path"]))), "phase": "train",
        "arm": "baseline", "seed": config["execution_seeds"][0], "chisel": None,
        "source_revision": config.get("source_revision"),
        "partition_sha256": config["partition_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--first-training", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--native-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require_remote()
    if args.native_job:
        try:
            return run_native(args.native_job)
        except Exception as error:
            persist(args.native_job.parent / "ADAPTER_FAILURE.json", {
                "type": type(error).__name__, "message": str(error),
            }, immutable=True)
            raise
    if args.config is None or args.run_dir is None or (not args.job and not args.first_training):
        parser.error("Use --config, --run-dir, and either --job or --first-training")
    config = load(args.config)
    if args.detach:
        root = owned_path(args.run_dir)
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        hashes = freeze_source(root / "source")
        persist(root / "SOURCE_SHA256.json", hashes, immutable=True)
        persist(root / "CONFIG.json", config, immutable=True)
        persist(root / "PREPARATION.json", provenance(ROOT, sys.argv, config["split_seed"]),
                immutable=True)
        job = first_training_job(config) if args.first_training else load(args.job)
        persist(root / "JOB.json", job, immutable=True)
        command = [
            config["are_python"], "-u", str(root / "source/scripts/run_deepseek_gaia2.py"),
            "--config", str(root / "CONFIG.json"), "--job", str(root / "JOB.json"),
            "--run-dir", str(root / "trial"),
        ]
        with (root / "launcher.log").open("xb") as stream:
            process = subprocess.Popen(
                command, cwd=root / "source", stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        receipt = {"pid": process.pid, "command": command, "started_at": utc_now(),
                   "run_dir": str(root), "trial": str(root / "trial")}
        persist(root / "PID.json", receipt, immutable=True)
        time.sleep(2)
        if process.poll() is not None:
            raise GateError(f"First TRAIN worker exited; inspect {root / 'launcher.log'}")
        print(json.dumps(receipt, sort_keys=True))
        return 0
    job = first_training_job(config) if args.first_training else load(args.job)
    result = run_episode(config, job, args.run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid_for_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
