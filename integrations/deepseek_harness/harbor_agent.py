"""Official DeepSeek SDK tools restricted to one native Harbor Docker environment.

Import path: ``integrations.deepseek_harness.harbor_agent:DeepSeekHarborAgent``.
The SDK worker and its bearer credential live outside Harbor's mounted agent logs.
Harbor, not this adapter, owns task timeouts, container lifecycle, and verification.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import secrets
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phiagent.harness.deepseek_bridge import ToolBridge

try:
    from harbor.agents.base import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "harbor" and not (exc.name or "").startswith("harbor."):
        raise

    class BaseAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "Install requirements/deepseek-harness-terminal.txt in a remote terminal runtime"
            )


HARBOR_VERSION = "0.18.0"
MODEL_NAME = "deepseek-flash"
LOGGER = logging.getLogger(__name__)
TERMINAL_TOOL = {
    "name": "terminal",
    "description": (
        "Run a shell command inside this task's isolated Harbor container. "
        "Use cwd for a container working directory. Commands are non-interactive; "
        "filesystem changes persist between calls. The task's native deadline applies."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "cwd": {"type": "string", "minLength": 1},
            "timeout_sec": {"type": "integer", "minimum": 1},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".writing")
    staging.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    staging.replace(path)


def runtime_identity(runtime: dict[str, str]) -> dict[str, str]:
    from integrations.deepseek_harness.runner import UPSTREAM_COMMIT, UPSTREAM_TAG

    if not isinstance(UPSTREAM_COMMIT, str) or not re.fullmatch(r"[a-f0-9]{40}", UPSTREAM_COMMIT):
        raise ValueError("The SDK runner must pin a verified official GitHub commit")
    if not isinstance(UPSTREAM_TAG, str) or not UPSTREAM_TAG:
        raise ValueError("The official SDK runner must record its verified GitHub tag")
    manifest_path = Path(runtime["manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("SDK runtime manifest and frozen official runner commit differ")
    if manifest.get("upstream_tag", UPSTREAM_TAG) != UPSTREAM_TAG:
        raise ValueError("SDK runtime manifest and frozen official runner tag differ")
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tag": UPSTREAM_TAG,
        "runtime_manifest_sha256": file_hash(manifest_path),
    }


def terminal_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    if set(arguments) - {"command", "cwd", "timeout_sec"}:
        raise ValueError("Only command, cwd, and timeout_sec are allowed")
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise ValueError("command must be a nonempty string without NUL")
    result: dict[str, Any] = {"command": command}
    if "cwd" in arguments:
        cwd = arguments["cwd"]
        if not isinstance(cwd, str) or not cwd or "\x00" in cwd:
            raise ValueError("cwd must be a nonempty container path without NUL")
        result["cwd"] = cwd
    if "timeout_sec" in arguments:
        timeout = arguments["timeout_sec"]
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout_sec must be a positive integer")
        result["timeout_sec"] = timeout
    return result


def terminal_environment(value: dict[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    allowed = {
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy",
        "UV_INSTALLER_GITHUB_BASE_URL", "UV_PYTHON_INSTALL_MIRROR",
        "UV_INDEX_URL", "PIP_INDEX_URL",
    }
    if (
        not isinstance(value, dict) or set(value) - allowed
        or any(not isinstance(item, str) or not item or "\x00" in item for item in value.values())
    ):
        raise ValueError("Terminal environment may contain only explicit public dependency routing")
    for upper, lower in (("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy"),
                         ("NO_PROXY", "no_proxy")):
        if upper in value and lower in value and value[upper] != value[lower]:
            raise ValueError("Upper/lowercase public dependency routes must agree")
    return dict(value)


class TerminalBackend:
    """Synchronous shared-bridge backend dispatching onto Harbor's asyncio loop."""

    def __init__(
        self, environment: Any, private_dir: Path,
        command_environment: dict[str, str] | None = None,
    ):
        self.environment = environment
        self.command_environment = {
            **terminal_environment(command_environment), "CUDA_VISIBLE_DEVICES": "",
        }
        private_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = private_dir / "terminal-exec.jsonl"
        self.loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._state_lock = threading.Lock()
        self._pending: set[concurrent.futures.Future] = set()
        self._tasks: set[asyncio.Task] = set()
        self._closing = False
        from integrations.deepseek_harness.shared_parent_guard import SharedParentGuard
        self.execution_guard = SharedParentGuard(str(private_dir.parent / "sdk-run" / "cwd"))

    def _record(self, event: dict[str, Any]) -> None:
        with self.audit_path.open("a") as stream:
            stream.write(json.dumps({"timestamp": utc_now(), **event}, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != TERMINAL_TOOL["name"]:
            raise ValueError("Tool is not in the terminal catalog")
        try:
            arguments = terminal_arguments(arguments)
        except ValueError as exc:
            return {
                "observation": json.dumps({
                    "error_type": "InvalidArguments", "error": str(exc), "executed": False,
                }),
                "conclude_turn": False,
            }
        if threading.get_ident() == self._loop_thread_id:
            raise RuntimeError("The synchronous terminal backend must run outside the Harbor loop")
        with self._state_lock:
            if self._closing:
                raise RuntimeError("Terminal backend is closing; mutation outcome may be unknown")
            future = asyncio.run_coroutine_threadsafe(self._execute(arguments), self.loop)
            self._pending.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError as exc:
            raise RuntimeError("Native terminal execution cancelled; mutation outcome unknown") from exc
        finally:
            with self._state_lock:
                self._pending.discard(future)

    async def _execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            if self._closing:
                raise RuntimeError("Terminal backend stopped before command dispatch")
            if self.execution_guard.container_cwd is None:
                discovery = await self.environment.exec(command="pwd", env=self.command_environment)
                discovered = (discovery.stdout or "").strip()
                if discovery.return_code != 0 or not discovered.startswith("/") or "\n" in discovered:
                    raise RuntimeError("Cannot discover native container working directory")
                self.execution_guard.container_cwd = discovered
                self._record({"event": "container_cwd_discovery", "container_cwd": discovered})
            original_arguments = dict(arguments)
            arguments, recovery = self.execution_guard.resolve(arguments)
            if recovery:
                self._record({"event": "shared_parent_recovery", "arguments": original_arguments,
                              "reason": recovery, "executed": False})
                return {"observation": json.dumps({"return_code": 125, "stdout": "",
                        "stderr": recovery, "executed": False, "replan_required": True}),
                        "conclude_turn": False}
            if arguments != original_arguments:
                self._record({"event": "host_container_translation", "original_arguments": original_arguments,
                              "effective_arguments": arguments})
            self._record({
                "event": "exec_start", "arguments": arguments,
                "command_environment": self.command_environment,
            })
            from integrations.deepseek_harness.shared_shell_execution import wrapped_command, cleanup_command
            import uuid
            wrapped, marker = wrapped_command(arguments['command'], arguments.get('timeout_sec'), uuid.uuid4().hex)
            effective = {**arguments, 'command': wrapped}
            if arguments.get('timeout_sec') is not None:
                effective['timeout_sec'] = arguments['timeout_sec'] + 10
            try:
                result = await self.environment.exec(**effective, env=self.command_environment)
            except asyncio.CancelledError:
                await asyncio.shield(self.environment.exec(command=cleanup_command(marker), env=self.command_environment, timeout_sec=10))
                raise

            observation = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
            }
            self._record({"event": "exec_result", "observation": observation})
            self.execution_guard.observe(arguments, observation)
            return {"observation": json.dumps(observation, ensure_ascii=False), "conclude_turn": False}
        except asyncio.CancelledError:
            self._record({
                "event": "exec_cancelled",
                "execution_status": "unknown_after_cancellation",
                "mutation_replay_allowed": False,
            })
            raise
        except Exception as exc:
            self._record({
                "event": "exec_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "execution_status": "unknown_after_error",
                "mutation_replay_allowed": False,
            })
            raise
        finally:
            self._tasks.discard(task)

    async def close(self) -> None:
        with self._state_lock:
            self._closing = True
            for future in self._pending:
                future.cancel()
        await asyncio.sleep(0)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def notifications(self) -> list[str]:
        return []

    def generation_start(self) -> None:
        return None

    def generation_end(self, elapsed_seconds: float) -> None:
        return None


@asynccontextmanager
async def terminal_bridge(
    environment: Any, private_dir: Path, command_environment: dict[str, str] | None = None,
):
    backend = TerminalBackend(environment, private_dir / "native-backend", command_environment)
    bridge = ToolBridge(private_dir / "bridge", backend, [TERMINAL_TOOL])
    bridge.__enter__()
    try:
        yield bridge
    finally:
        await backend.close()
        await asyncio.to_thread(bridge.__exit__, None, None, None)


def known_nonnegative(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def sum_known(values: list[Any]) -> int | None:
    normalized = [known_nonnegative(value) for value in values]
    return sum(normalized) if normalized and all(v is not None for v in normalized) else None


def validate_container_isolation(container: dict[str, Any], private_dir: Path) -> None:
    host = container["HostConfig"]
    if host.get("Privileged") or host.get("NetworkMode") == "host" or host.get("PidMode") == "host":
        raise ValueError("The task container must not expose privileged host execution")
    for mount in container.get("Mounts") or []:
        source = mount.get("Source") or ""
        target = mount.get("Destination") or ""
        if "docker.sock" in source or "docker.sock" in target:
            raise ValueError("The Docker socket must not be mounted into the task container")
        if source and private_dir.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError("The SDK private workspace must not be mounted into the task container")
    if host.get("DeviceRequests"):
        raise ValueError("This CPU-only campaign must not request container GPUs")
    if any("nvidia" in (item.get("PathOnHost") or "") for item in host.get("Devices") or []):
        raise ValueError("This CPU-only campaign must not mount NVIDIA devices")


async def inspect_owned_container(environment: Any, private_dir: Path) -> dict[str, Any]:
    from harbor.environments.docker.docker import MAIN_SERVICE_NAME

    # This read-only command uses Harbor's own project identity, never a model-supplied ID.
    result = await environment._run_docker_compose_command(["ps", "-q", MAIN_SERVICE_NAME])
    container_ids = [
        line.strip() for line in (result.stdout or "").splitlines()
        if re.fullmatch(r"[a-f0-9]{12,64}", line.strip())
    ]
    if len(container_ids) != 1:
        raise RuntimeError("Expected exactly one owned native Harbor main container")
    command = ["docker", "inspect", container_ids[0]]
    inspected = await asyncio.to_thread(
        subprocess.run, command, capture_output=True, text=True, timeout=30, check=True
    )
    container = json.loads(inspected.stdout)[0]
    validate_container_isolation(container, private_dir)
    host = container["HostConfig"]
    requested = environment.task_env_config.model_dump(mode="json")
    return {
        "container_id": container["Id"],
        "name": container["Name"],
        "image_id": container["Image"],
        "image_reference": container.get("Config", {}).get("Image"),
        "host_config": {
            key: host.get(key)
            for key in (
                "NanoCpus", "CpuQuota", "CpuPeriod", "Memory", "MemorySwap", "StorageOpt",
                "Privileged", "NetworkMode", "PidMode", "DeviceRequests", "Devices", "Runtime",
            )
        },
        "mounts": [
            {key: mount.get(key) for key in ("Type", "Source", "Destination", "RW")}
            for mount in container.get("Mounts") or []
        ],
        "requested": {
            key: requested.get(key) for key in (
                "cpus", "memory", "memory_mb", "storage", "storage_mb", "gpus",
                "build_timeout_sec", "docker_image",
            ) if key in requested
        },
        "native_capabilities": environment.resource_capabilities().model_dump(mode="json"),
        "cpu_limit_configured": bool(host.get("NanoCpus") or (host.get("CpuQuota") or 0) > 0),
        "memory_limit_configured": bool(host.get("Memory")),
        "storage_enforcement": {
            "enforced": False,
            "reason": "Harbor 0.18.0 Docker supports CPU and memory limits, not storage quotas",
            "official_request_preserved": True,
        },
    }


async def stop_owned_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def normalize_runner_usage(raw: Any) -> dict[str, int | None]:
    raw = raw if isinstance(raw, dict) else {}
    result = {
        "input_tokens": known_nonnegative(raw.get("inputTokens")),
        "output_tokens": known_nonnegative(raw.get("outputTokens")),
        "cache_read_input_tokens": known_nonnegative(raw.get("cacheReadTokens")),
        "cache_creation_input_tokens": known_nonnegative(raw.get("cacheWriteTokens")),
        "total_tokens": known_nonnegative(raw.get("totalTokens")),
    }
    prompt = sum_known([
        result["input_tokens"], result["cache_read_input_tokens"],
        result["cache_creation_input_tokens"],
    ])
    total, output = result["total_tokens"], result["output_tokens"]
    if total is not None and output is not None and total >= output:
        exact_prompt = total - output
        if prompt is not None and prompt != exact_prompt:
            raise ValueError("Provider total contradicts the disjoint input/cache/output buckets")
        prompt = exact_prompt
    result["prompt_tokens_including_cache"] = prompt
    return result


def build_worker_job(
    *,
    instruction: str,
    session_id: str,
    private_dir: Path,
    bridge: ToolBridge,
    runtime: dict[str, str],
    api_key_file: str,
    max_tokens: int,
    seed: int,
    source_revision: str | None,
    arm: str = "baseline",
    chisel: dict[str, Any] | None = None,
    observation_policy: dict[str, Any] | None = None,
    cot_jev: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not bridge.url.startswith("http://127.0.0.1:"):
        raise ValueError("The terminal bridge has not started")
    from integrations.deepseek_harness.runner import validate_observation_arm

    validate_observation_arm(arm, chisel, observation_policy)
    job = {
        "schema_version": 1,
        "run_dir": str(private_dir / "sdk-run"),
        "purpose": "benchmark",
        "arm": arm,
        "session_id": session_id,
        "seed": seed,
        "prompt": instruction,
        "model": MODEL_NAME,
        "max_tokens": max_tokens,
        "api_key_file": api_key_file,
        "runtime": runtime,
        "tools": [TERMINAL_TOOL],
        "bridge": {
            "url": bridge.url,
            "secret_file": str(bridge.secret_file),
            "notifications": False,
            "generation_timing": False,
        },
        "chisel": chisel,
        "source_revision": source_revision,
    }
    if observation_policy is not None:
        job["observation_policy"] = dict(observation_policy)
    if cot_jev is not None:
        from integrations.deepseek_harness.runner import validate_job

        job["cot_jev"] = dict(cot_jev)
        validate_job(job)
    return job


def recover_runner_accounting(worker_dir: Path) -> dict[str, Any]:
    accounting_path = worker_dir / "usage.json"
    errors, stored = [], None
    if accounting_path.exists():
        try:
            stored = json.loads(accounting_path.read_text())
            if not isinstance(stored, dict):
                raise ValueError("Accounting must be an object")
        except (ValueError, OSError) as exc:
            stored = None
            errors.append(f"incomplete-accounting-file:{type(exc).__name__}")
            LOGGER.warning("Native accounting snapshot is unreadable in %s: %s", worker_dir, exc)
    from integrations.deepseek_harness.runner import normalize_usage, summarize_streams

    records = []
    stream_path = worker_dir / "llm-streams.jsonl"
    if not stream_path.exists() and stored is not None:
        return {**stored, "stream_cross_check": "UNAVAILABLE_NO_STREAM_FILE"}
    if stream_path.exists():
        with stream_path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Native stream receipt must be an object")
                    records.append(record)
                except ValueError as exc:
                    errors.append(f"incomplete-stream-receipt-line:{number}")
                    LOGGER.warning("Incomplete native stream receipt in %s at line %s: %s",
                                   worker_dir, number, exc)
    accounting = summarize_streams(records)
    if (not records and not errors and stored is not None
            and stored.get("no_generation") is True
            and stored.get("attempt_count") == 0):
        return {**stored, "stream_cross_check": "MATCHED_NO_GENERATION"}
    if stored is not None:
        stored_usage = normalize_usage(stored.get("usage"))
        matched = (
            stored.get("complete") is accounting["complete"]
            and stored_usage == accounting["usage"]
            and stored.get("attempt_count") == accounting["attempt_count"]
        )
        accounting["snapshot_reconciliation"] = {
            "status": "MATCHED" if matched else "STALE_SNAPSHOT_REPLACED_BY_RAW_RECEIPTS",
            "stored_usage": stored_usage,
            "stored_complete": stored.get("complete"),
            "stored_attempt_count": stored.get("attempt_count"),
            "raw_attempt_count": accounting["attempt_count"],
            "stored_path": str(accounting_path),
            "raw_path": str(stream_path),
            "raw_receipts_counted_once": True,
        }
        if not matched:
            LOGGER.warning(
                "Replacing stale native accounting snapshot in %s with %s recorded attempts",
                worker_dir, accounting["attempt_count"],
            )
    if errors:
        accounting["complete"] = False
        accounting["usage"] = None
        accounting["unknown_reasons"].extend(errors)
    accounting["stream_cross_check"] = "RAW_ALL_ATTEMPTS"
    accounting["recovered_after_worker_interruption"] = True
    closed = worker_dir.parent / 'PROVIDER_CAPTURE_CLOSED.json'
    if closed.is_file():
        proof = json.loads(closed.read_text())
        if proof.get('owned_process_group_closed') is not True:
            raise ValueError('Native provider capture lacks an actual closed actor group')
        from scripts.serve_deepseek_durable_capture import audited_physical_accounting

        physical = audited_physical_accounting(worker_dir.parent / 'provider-wire', accounting)
        write_json(worker_dir.parent / 'RAW_PROVIDER_ACCOUNTING.json', physical)
        return physical
    return accounting


class DeepSeekHarborAgent(BaseAgent):
    command_environment: dict[str, str] | None = None
    """Harbor BaseAgent using the pinned official SDK runner, with no local agent loop."""

    def __init__(
        self,
        logs_dir: Path,
        *,
        runner_python: str,
        runner_script: str,
        runtime: dict[str, str],
        api_key_file: str,
        private_root: str,
        model_name: str | None = MODEL_NAME,
        max_tokens: int = 32768,
        seed: int = 20260928,
        arm: str = "baseline",
        chisel: dict[str, Any] | None = None,
        observation_policy: dict[str, Any] | None = None,
        cot_jev: dict[str, Any] | None = None,
        usage_identity: dict[str, Any] | None = None,
        provider_durable_capture: bool = False,
        command_environment: dict[str, str] | None = None,
        source_revision: str | None = None,
        **kwargs: Any,
    ):
        if importlib.metadata.version("harbor") != HARBOR_VERSION:
            raise ValueError(f"This adapter requires official Harbor {HARBOR_VERSION}")
        if model_name not in (MODEL_NAME, f"deepseek-official/{MODEL_NAME}"):
            raise ValueError(f"The paired campaign requires {MODEL_NAME}")
        from integrations.deepseek_harness.runner import validate_observation_arm

        validate_observation_arm(arm, chisel, observation_policy)
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be the positive per-request output capacity")
        for path in (runner_python, runner_script, private_root, api_key_file):
            if not Path(path).is_absolute():
                raise ValueError("Runner and private artifact paths must be absolute")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.runner_python = runner_python
        self.runner_script = runner_script
        self.runtime = runtime
        self._runtime_identity = runtime_identity(runtime)
        self.api_key_file = api_key_file
        self.private_root = Path(private_root).resolve()
        self.max_tokens = max_tokens
        self.seed = seed
        self.arm = arm
        self.chisel = chisel
        self.observation_policy = (
            dict(observation_policy) if observation_policy is not None else None
        )
        self.cot_jev = dict(cot_jev) if cot_jev is not None else None
        self.usage_identity = dict(usage_identity or {})
        if type(provider_durable_capture) is not bool:
            raise ValueError('Provider durable capture must be an explicit boolean')
        self.provider_durable_capture = provider_durable_capture
        self.command_environment = terminal_environment(command_environment)
        self.source_revision = source_revision
        self._session_id = "terminal-" + secrets.token_hex(12)
        self._private_dir: Path | None = None

    @staticmethod
    def name() -> str:
        return "official-deepseek-harness-terminal"

    def version(self) -> str:
        return f"{self._runtime_identity['upstream_tag']}+harbor-{HARBOR_VERSION}"

    async def setup(self, environment: Any) -> None:
        from harbor.environments.docker.docker import DockerEnvironment

        if type(environment) is not DockerEnvironment:
            raise ValueError("This adapter requires the official native Harbor Docker environment")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Set CUDA_VISIBLE_DEVICES='' for this CPU/API-only campaign")
        if self.private_root.is_relative_to(self.logs_dir.resolve()):
            raise ValueError("The SDK private workspace must be outside container-mounted agent logs")
        self._private_dir = self.private_root / self._session_id
        self._private_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        receipt = await inspect_owned_container(environment, self._private_dir)
        receipt.update({
            "timestamp": utc_now(),
            "harbor_version": HARBOR_VERSION,
            "native_session_id": environment.session_id,
            "sdk_session_id": self._session_id,
            "official_upstream_revision": self._runtime_identity["upstream_commit"],
            "official_upstream_tag": self._runtime_identity["upstream_tag"],
            "runtime_manifest_sha256": self._runtime_identity["runtime_manifest_sha256"],
            "arm": self.arm,
            "observation_policy": self.observation_policy,
            "command_environment": self.command_environment,
            "model": MODEL_NAME,
            "tool_catalog": [TERMINAL_TOOL],
            "CUDA_VISIBLE_DEVICES": "",
            "selected_gpus": [],
            "cumulative_token_limit": None,
            "per_request_output_capacity": self.max_tokens,
            "trajectory_format": "official SDK session events and LLM streams; terminal call audit",
        })
        write_json(self._private_dir / "HARBOR_ENVIRONMENT.json", receipt)
        write_json(self.logs_dir / "adapter-provenance.json", {
            "private_artifacts": str(self._private_dir),
            "environment_receipt_sha256": file_hash(self._private_dir / "HARBOR_ENVIRONMENT.json"),
        })

    def _populate_context(self, context: Any) -> None:
        if self._private_dir is None:
            return
        worker_dir = self._private_dir / "sdk-run"
        accounting = recover_runner_accounting(worker_dir)
        usage = normalize_runner_usage(accounting.get("usage"))
        known = normalize_runner_usage(accounting.get("known_reported_usage"))
        context.n_input_tokens = usage["prompt_tokens_including_cache"]
        context.n_output_tokens = usage["output_tokens"]
        context.n_cache_tokens = sum_known([
            usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"],
        ])
        context.cost_usd = None
        context.metadata = {
            **(context.metadata or {}),
            "deepseek_usage": usage,
            "deepseek_known_reported_usage": known,
            "deepseek_usage_complete": accounting.get("complete", False),
            "usage_bucket_semantics": (
                "input_tokens excludes disjoint cache-read/cache-write buckets; "
                "Harbor n_input_tokens includes both; n_cache_tokens is their sum"
            ),
            "usage_source": accounting.get("source"),
            "usage_unknown_reasons": accounting.get("unknown_reasons", []),
            "provider_attempt_count": accounting.get(
                "model_call_attempt_count", accounting.get("attempt_count"),
            ),
            "native_stream_attempt_count": accounting.get("attempt_count"),
            "private_artifacts": str(self._private_dir),
            "sdk_worker_dir": str(worker_dir),
            "cumulative_token_limit": None,
        }
        write_json(self._private_dir / "TOKEN_ACCOUNTING.json", {
            "usage": usage, "known_reported_usage": known, "provider_accounting": accounting,
        })
        write_json(self.logs_dir / "token-accounting.json", {
            "usage": usage,
            "known_reported_usage": known,
            "complete": accounting.get("complete", False),
            "trusted_private_receipt": str(self._private_dir / "TOKEN_ACCOUNTING.json"),
        })

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if self._private_dir is None:
            raise RuntimeError("Harbor must call setup before run")
        process = None
        capture = None
        capture_secret_file = None
        try:
            async with terminal_bridge(
                environment, self._private_dir, self.command_environment,
            ) as bridge:
                job = build_worker_job(
                    instruction=instruction,
                    session_id=self._session_id,
                    private_dir=self._private_dir,
                    bridge=bridge,
                    runtime=self.runtime,
                    api_key_file=self.api_key_file,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    source_revision=self.source_revision,
                    arm=self.arm,
                    chisel=self.chisel,
                    observation_policy=self.observation_policy,
                    cot_jev=self.cot_jev,
                )
                if self.provider_durable_capture:
                    from integrations.deepseek_harness.runner import read_owner_secret
                    from scripts.serve_deepseek_durable_capture import start_capture

                    capture_secret = secrets.token_urlsafe(32)
                    capture_secret_file = self._private_dir / 'capture.secret'
                    capture_secret_file.write_text(capture_secret + '\n')
                    capture_secret_file.chmod(0o600)
                    capture = start_capture(self._private_dir / 'provider-wire',
                        read_owner_secret(Path(self.api_key_file)), capture_secret, attribution=self.usage_identity)
                    origin = f'http://127.0.0.1:{capture.server_port}/actor'
                    job.update(api_key_file=str(capture_secret_file), provider_capture={
                        'base_url': origin + '/anthropic', 'fast_url': origin + '/chat/completions'})
                job_path = self._private_dir / "worker-job.json"
                write_json(job_path, job)
                command = [self.runner_python, self.runner_script, "--job", str(job_path)]
                write_json(self._private_dir / "COMMAND.json", {
                    "argv": command, "cwd": str(self._private_dir),
                    "CUDA_VISIBLE_DEVICES": "", "cumulative_token_limit": None,
                })
                cache_dir = self._private_dir / "process-cache"
                cache_dir.mkdir()
                child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(cache_dir)}
                with (
                    (self._private_dir / "runner.stdout.log").open("wb") as stdout,
                    (self._private_dir / "runner.stderr.log").open("wb") as stderr,
                ):
                    process = await asyncio.create_subprocess_exec(
                        *command, cwd=self._private_dir, env=child_env, stdout=stdout, stderr=stderr,
                        start_new_session=True,
                    )
                    await process.wait()
                result_path = self._private_dir / "sdk-run" / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else None
                if (
                    bridge.blocked or not result
                    or result.get("status") != "completed" or process.returncode != 0
                ):
                    raise RuntimeError(
                        f"Official SDK worker did not complete; preserve {self._private_dir}"
                    )
                write_json(self.logs_dir / "official-sdk-result.json", result)
        finally:
            if process is not None:
                await stop_owned_process(process)
            if capture is not None:
                await asyncio.to_thread(capture.shutdown)
                await asyncio.to_thread(capture.server_close)
                group_closed = process is None
                if process is not None:
                    try:
                        os.killpg(process.pid, 0)
                        group_closed = False
                    except ProcessLookupError:
                        group_closed = True
                write_json(self._private_dir / 'PROVIDER_CAPTURE_CLOSED.json', {
                    'owned_process_group_closed': group_closed,
                    'owned_actor_pid': process.pid if process is not None else None,
                    'actor_returncode': process.returncode if process is not None else None,
                    'provider_never_started': process is None,
                    'native_actor_limits_unchanged': True, 'automatic_actor_retries': 0,
                    'capture_drained_to_native_provider_end': True})
                capture_secret_file.unlink(missing_ok=True)
            self._populate_context(context)

    def populate_context_post_run(self, context: Any) -> None:
        self._populate_context(context)
