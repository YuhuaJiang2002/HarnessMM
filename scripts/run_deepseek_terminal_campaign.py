#!/usr/bin/env python3
"""Durable full TB2 CHISEL/official-DeepSeek lifecycle, on the authorized A800 only.

Launch with ``CUDA_VISIBLE_DEVICES='' terminal-runtime/bin/python
source/scripts/run_deepseek_terminal_campaign.py --config CONFIG --run-dir NEW --detach``.
The launcher freezes the accepted native adapter and shared learner, then detaches
the entire TRAIN -> optimizer -> VAL -> freeze -> sealed paired TEST controller.
``--resume --detach`` on that frozen run replays committed callbacks, not actions.
An uncommitted callback can only harvest its already-existing native result;
ambiguous actions block instead of being repeated. No cumulative token ceiling.

SUMMARY.json/CSV remain explicitly partial until all 530 final native verdicts
exist. Optimizer, development, final execution and inherited setup costs are
separate. The SDK is the actor and optimizer; this file supplies orchestration,
public TRAIN evidence and official Harbor scoring, never a replacement agent.

A settled SDK per-response ``max-tokens`` stop still proceeds to the native
verifier. Its output capacity and task deadline are unchanged. New source
continuations can import equivalent completed TRAIN baselines and all predecessor
costs. Diagnosed unscored infrastructure failures whose workspaces were deleted
may receive a new isolated acquisition. Their unknown mutation outcomes and
provider costs remain recorded; no live action or scored outcome is replayed.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import csv
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.deepseek_harness.harbor_agent import (  # noqa: E402
    DeepSeekHarborAgent, canonical_hash, file_hash, normalize_runner_usage,
    recover_runner_accounting, utc_now, TerminalBackend, TERMINAL_TOOL,
    build_worker_job, stop_owned_process,
)
from phiagent.harness.deepseek_bridge import ToolBridge  # noqa: E402
from scripts.terminal_network_lifecycle import (  # noqa: E402
    attest_undispatched_network_failure, network_owner, retire_networks,
)
from scripts.run_deepseek_terminal import (  # noqa: E402
    AUTHORIZED_HOST, CAMPAIGN, HARBOR_VERSION, MODEL_NAME, PARTITION_SHA256,
    TERMINAL_REVISION, build_native_job, freeze_source, read_command, resource_inventory,
    summarize_native_trial, task_file_hashes, validate_acceptance,
)

SCHEMA = "deepseek-terminal-campaign/v1"
UPSTREAM = "4878cdabd87d4041bdaff61d04c966883b9fd07a"
TAG = "dsh-v0.2.0-rc.1"
USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "total_tokens", "cost_usd",
)
NATIVE_FILES = (
    "integrations/deepseek_harness/harbor_agent.py",
    "integrations/deepseek_harness/runner.py",
    "integrations/deepseek_harness/bridge_plugin.mjs",
    "integrations/deepseek_harness/cot_jev.mjs",
    "integrations/deepseek_harness/chisel_plugin.mjs",
    "integrations/deepseek_harness/observation_policy.mjs",
    "integrations/deepseek_harness/action_evidence_checkpoint.mjs",
    "phiagent/harness/deepseek_bridge.py",
    "phiagent/data_engine/provenance.py",
    "scripts/run_deepseek_terminal.py",
)
CONTROLLER_FILES = (
    "scripts/run_deepseek_terminal_campaign.py",
    "scripts/industrial_terminal_agents.py",
    "scripts/codex_terminal_agent.py",
    "scripts/serve_codex_deepseek.py",
    "scripts/prepare_deepseek_terminal_recovery.py",
    "scripts/verify_deepseek_terminal_shutdown.py",
    "scripts/recover_terminal_verifier.py",
    "scripts/terminal_network_lifecycle.py",
    "tests/test_deepseek_terminal_campaign.py",
    "tests/test_deepseek_terminal_recovery.py",
    "tests/test_recover_terminal_verifier.py",
    "configs/deepseek_terminal_campaign_v1.json",
    "configs/deepseek_terminal_campaign_v4.json",
    "configs/deepseek_terminal_campaign_v5.json",
    "configs/deepseek_terminal_campaign_v8.json",
    "phiagent/harness/deepseek_chisel.py",
    "phiagent/harness/codex_chisel.py",
    "phiagent/harness/codex_runtime.py",
    "phiagent/harness/benchmark_partition.py",
    "phiagent/harness/industrial_common_prompt.py",
    "phiagent/experiments.py",
    "phiagent/learning/residual_meta_harness.py",
    "phiagent/learning/residual_meta_training.py",
)
PRIVATE_TRACE = re.compile(
    r"(?i)(?:/tests(?:/|\b)|/verifier(?:/|\b)|test_outputs\.py|reward\.txt|"
    r"solution\.(?:sh|py)|\boracle\b|api[_-]?key|authorization|bearer\s)"
)


class CampaignBlocked(RuntimeError):
    pass


def require_train_settlement(report: dict[str, Any]) -> None:
    observations = report.get("observations")
    if not isinstance(observations, dict) or not observations:
        raise CampaignBlocked("Complete TRAIN observation and usage gate is closed")
    for request_id, result in observations.items():
        if "task_id" in result and result.get("native_score") is None:
            raise CampaignBlocked(f"Complete TRAIN native-verdict gate is closed: {request_id}")
        usage = result.get("usage")
        if (
            not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0
                   for key in ("input_tokens", "output_tokens", "total_tokens"))
            or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
        ):
            raise CampaignBlocked(f"Complete TRAIN provider-usage gate is closed: {request_id}")


class NativeTerminalBackend(TerminalBackend):
    """Report native command timeouts without replaying an unresolved command."""

    def __init__(
        self, environment: Any, private_dir: Path,
        command_environment: dict[str, str] | None = None,
    ):
        super().__init__(environment, private_dir, command_environment)
        self._unresolved_commands: set[str] = set()

    async def _execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        key = canonical_hash({name: arguments.get(name) for name in ("command", "cwd")})
        if key in self._unresolved_commands:
            self._record({"event": "uncertain_command_replay_refused", "command_sha256": key})
            return {"observation": json.dumps({
                "error_type": "UnresolvedCommandReplayRefused",
                "execution_status": "not_reexecuted",
                "error": "This command previously timed out; its mutation outcome is unknown.",
                "mutation_replay_allowed": False,
            }), "conclude_turn": False}
        try:
            return await super()._execute(arguments)
        except RuntimeError as exc:
            if not re.fullmatch(r"Command timed out after \d+(?:\.\d+)? seconds", str(exc)):
                raise
            self._unresolved_commands.add(key)
            self._record({
                "event": "native_command_timeout_observed", "command_sha256": key,
                "execution_status": "unknown_after_timeout", "mutation_replay_allowed": False,
                "native_timeout_unchanged": True,
            })
            return {"observation": json.dumps({
                "error_type": "NativeCommandTimeout", "error": str(exc),
                "execution_status": "unknown_after_timeout", "return_code": None,
                "mutation_replay_allowed": False,
            }), "conclude_turn": False}


class QuiescentTerminalBridge(ToolBridge):
    """Do not release cancellation errors to an SDK that can still request a new turn."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.teardown_started = threading.Event()
        self.worker_stopped = threading.Event()

    def tool(self, request: dict) -> dict:
        try:
            return super().tool(request)
        except Exception:
            if self.teardown_started.is_set():
                self.worker_stopped.wait()
            raise


def live_sdk_group(group: int) -> bool:
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            value = (path / "stat").read_text()
            fields = value[value.rfind(")") + 2:].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


async def quiesce_sdk_worker(process: asyncio.subprocess.Process | None) -> None:
    if process is None:
        return
    started = time.monotonic()
    await stop_owned_process(process)
    # The Python wrapper can exit before its SDK child finishes signal teardown.
    while live_sdk_group(process.pid):
        if time.monotonic() - started >= 10:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        await asyncio.sleep(0.02)


@asynccontextmanager
async def campaign_terminal_bridge(
    environment: Any, private_dir: Path,
    stop_worker: Callable[[], Awaitable[None]] | None = None,
    command_environment: dict[str, str] | None = None,
):
    backend = NativeTerminalBackend(
        environment, private_dir / "native-backend", command_environment,
    )
    bridge = QuiescentTerminalBridge(private_dir / "bridge", backend, [TERMINAL_TOOL])
    bridge.__enter__()
    try:
        yield bridge
    finally:
        bridge.teardown_started.set()
        try:
            await backend.close()
            if stop_worker is not None:
                await stop_worker()
        finally:
            bridge.worker_stopped.set()
            await asyncio.to_thread(bridge.__exit__, None, None, None)


def settled_output_stop(result: dict[str, Any]) -> bool:
    return (
        result.get("status") == "failed"
        and result.get("finish_reason") == "max-tokens"
        and result.get("finish_details") == {"kind": "max-tokens"}
        and result.get("usage_complete") is True
        and not result.get("error")
        and not result.get("close_error")
    )


class NativeCompletionAgent(DeepSeekHarborAgent):
    """Use the accepted SDK actor with quiescent cancellation and native verifier handoff."""

    async def _run_worker(self, instruction: str, environment: Any, context: Any) -> None:
        if self._private_dir is None:
            raise RuntimeError("Harbor must call setup before run")
        private, process, capture = self._private_dir, None, None
        capture_secret_file = None

        async def stop_worker() -> None:
            await quiesce_sdk_worker(process)
            persist(private / "SDK_SHUTDOWN.json", {
                "at": utc_now(), "worker_pid": process.pid if process is not None else None,
                "returncode": process.returncode if process is not None else None,
                "native_commands_cancelled_before_worker_wait": True,
                "cancellation_responses_released_after_sdk_stop": True,
                "native_task_deadline_extended": False,
                "cumulative_token_limit": None,
            }, immutable=True)

        try:
            async with campaign_terminal_bridge(
                environment, private, stop_worker, self.command_environment,
            ) as bridge:
                job = build_worker_job(
                    instruction=instruction, session_id=self._session_id, private_dir=private,
                    bridge=bridge, runtime=self.runtime, api_key_file=self.api_key_file,
                    max_tokens=self.max_tokens, seed=self.seed, source_revision=self.source_revision,
                    arm=self.arm, chisel=self.chisel, observation_policy=self.observation_policy,
                    cot_jev=self.cot_jev,
                )
                if getattr(self, 'provider_durable_capture', False):
                    import secrets
                    from integrations.deepseek_harness.runner import read_owner_secret
                    from scripts.serve_deepseek_durable_capture import start_capture

                    secret = secrets.token_urlsafe(32)
                    capture_secret_file = private / 'capture.secret'
                    capture_secret_file.write_text(secret + '\n')
                    capture_secret_file.chmod(0o600)
                    capture = start_capture(private / 'provider-wire',
                        read_owner_secret(Path(self.api_key_file)), secret, attribution=self.usage_identity)
                    origin = f'http://127.0.0.1:{capture.server_port}/actor'
                    job.update(api_key_file=str(capture_secret_file), provider_capture={
                        'base_url': origin + '/anthropic', 'fast_url': origin + '/chat/completions'})
                job_path = private / "worker-job.json"
                persist(job_path, job, immutable=True)
                command = [self.runner_python, self.runner_script, "--job", str(job_path)]
                persist(private / "COMMAND.json", {
                    "argv": command, "cwd": str(private),
                    "CUDA_VISIBLE_DEVICES": "", "cumulative_token_limit": None,
                }, immutable=True)
                cache = private / "process-cache"
                cache.mkdir()
                with (
                    (private / "runner.stdout.log").open("wb") as stdout,
                    (private / "runner.stderr.log").open("wb") as stderr,
                ):
                    process = await asyncio.create_subprocess_exec(
                        *command, cwd=private,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(cache)},
                        stdout=stdout, stderr=stderr, start_new_session=True,
                    )
                    await process.wait()
                result_path = private / "sdk-run/result.json"
                result = load(result_path) if result_path.exists() else None
                if (
                    bridge.blocked or not result or result.get("status") != "completed"
                    or process.returncode != 0
                ):
                    raise RuntimeError(f"Official SDK worker did not complete; preserve {private}")
                persist(self.logs_dir / "official-sdk-result.json", result)
        finally:
            if capture is not None:
                await quiesce_sdk_worker(process)
                await asyncio.to_thread(capture.shutdown)
                await asyncio.to_thread(capture.server_close)
                persist(private / 'PROVIDER_CAPTURE_CLOSED.json', {
                    'owned_process_group_closed': process is None or not live_sdk_group(process.pid),
                    'owned_actor_pid': process.pid if process is not None else None,
                    'actor_returncode': process.returncode if process is not None else None,
                    'provider_never_started': process is None,
                    'native_actor_limits_unchanged': True, 'automatic_actor_retries': 0,
                    'capture_drained_to_native_provider_end': True}, immutable=True)
                capture_secret_file.unlink(missing_ok=True)
            self._populate_context(context)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        try:
            await self._run_worker(instruction, environment, context)
        except RuntimeError as exc:
            if not str(exc).startswith("Official SDK worker did not complete;"):
                raise
            private = self._private_dir
            result_path = private / "sdk-run/result.json" if private else None
            if result_path is None or not result_path.is_file():
                raise
            result = load(result_path)
            if not settled_output_stop(result):
                raise
            sdk_usage(result)
            if (context.metadata or {}).get("deepseek_usage_complete") is not True:
                raise
            context.metadata.update({
                "native_actor_stop": "sdk_per_response_output_limit",
                "sdk_finish_reason": "max-tokens",
                "native_verifier_required": True,
                "native_deadline_extended": False,
                "provider_calls_added_by_completion_adapter": 0,
            })
            persist(private / "NATIVE_ACTOR_STOP.json", {
                **{key: context.metadata[key] for key in (
                    "native_actor_stop", "sdk_finish_reason", "native_verifier_required",
                    "native_deadline_extended", "provider_calls_added_by_completion_adapter",
                )},
                "sdk_result_sha256": file_hash(result_path),
                "per_request_output_capacity": self.max_tokens,
                "cumulative_token_limit": None,
            }, immutable=True)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def serve_dependencies(manifest_path: Path, port: int) -> None:
    """Serve only independently hash-verified public release assets, never run files."""
    manifest = load(manifest_path)
    routes = {}
    for row in manifest["assets"]:
        path = Path(row["path"])
        if (
            path.is_symlink() or not path.resolve().is_relative_to(manifest_path.parent.resolve())
            or path.stat().st_size != row["size"] or file_hash(path) != row["sha256"]
        ):
            raise CampaignBlocked("Public dependency archive differs from its upstream digest")
        routes[row["route"]] = path

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def answer(self, body: bool) -> None:
            route = unquote(urlsplit(self.path).path)
            if route == "/wheels/":
                content = ("\n".join(
                    f'<a href="{row["route"]}#sha256={row["sha256"]}">'
                    f'{Path(row["path"]).name}</a>'
                    for row in manifest["assets"] if row["route"].startswith("/wheels/")
                ) + "\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                if body:
                    self.wfile.write(content)
                return
            path = routes.get(route)
            if path is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            if body:
                try:
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            self.answer(True)

        def do_HEAD(self):
            self.answer(False)

    server = ThreadingHTTPServer(("172.17.0.1", port), Handler)
    persist(manifest_path.parent / "DEPENDENCY_SERVER.json", {
        "pid": os.getpid(), "hostname": socket.gethostname(), "port": server.server_port,
        "started_at": utc_now(), "manifest_sha256": file_hash(manifest_path),
        "public_release_assets_only": True, "routes": sorted(routes),
    })
    server.serve_forever()


def verify_dependencies(config_path: Path, run: Path) -> dict[str, Any]:
    """Exercise only the real native image's runtime prerequisites, not data or scoring."""
    import tomllib
    config = load(config_path)
    validate_config(config)
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs = run / "verifier-logs"
    logs.mkdir()
    native_config = tomllib.loads(
        (Path(config["task_root"]) / "count-dataset-tokens/task.toml").read_text()
    )
    image = native_config["environment"]["docker_image"]
    container = f"dsh-runtime-prerequisite-{os.getpid()}"
    package_check = (
        "import importlib.metadata as m,json,sys;"
        "assert sys.version_info[:2]==(3,13);"
        "assert m.version('pytest')=='8.4.1';"
        "assert m.version('pytest-json-ctrf')=='0.3.5';"
        "print('NATIVE_DEPENDENCIES_OK '+json.dumps("
        "{'python':sys.version,'pytest':m.version('pytest'),"
        "'pytest_json_ctrf':m.version('pytest-json-ctrf')}))"
    )
    script = (
        "set -euo pipefail\n"
        "apt-get update\napt-get install -y curl\n"
        "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n"
        'source "$HOME/.local/bin/env"\nuv --version\n'
        f"uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 python -c '{package_check.replace(chr(39), chr(34))}'\n"
    )
    command = [
        "docker", "run", "--name", container, "--rm", "--cpus", "1", "--memory", "2g",
        "--env", "CUDA_VISIBLE_DEVICES=",
        "--mount", f"type=bind,source={logs},target=/logs/verifier",
    ]
    for key, value in sorted(config["verifier_environment"].items()):
        command.extend(["--env", f"{key}={value}"])
    command.extend(["--entrypoint", "/bin/bash", image, "-lc", script])
    persist(run / "COMMANDS.json", {
        "argv": command, "purpose": "Native image runtime prerequisites only",
        "model_calls": 0, "benchmark_data_or_evaluator_invoked": False,
        "driver_sha256": file_hash(Path(__file__)),
    }, immutable=True)
    persist(run / "RESOURCE_INVENTORY.json", resource_inventory(), immutable=True)
    persist(run / "CONFIG.json", config, immutable=True)
    persist(run / "IMAGE.json", read_command(["docker", "image", "inspect", image]), immutable=True)
    started = time.monotonic()
    failure = None
    code = None
    try:
        with (run / "probe.log").open("xb") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, close_fds=True,
            )
            persist(run / "PROCESS.json", {"pid": process.pid, "container": container},
                    immutable=True)
            code = process.wait(timeout=880)
    except subprocess.TimeoutExpired:
        failure = "Dependency prerequisite check exceeded its separate 880-second infrastructure gate"
        subprocess.run(["docker", "rm", "--force", container], capture_output=True, timeout=30)
        process.wait(timeout=30)
    text = (run / "probe.log").read_text()
    passed = code == 0 and "NATIVE_DEPENDENCIES_OK " in text and bool(
        re.search(r"(?m)^uv 0\.9\.5(?:\s|$)", text)
    )
    receipt = {
        "status": "passed" if passed else "blocked", "at": utc_now(), "run_dir": str(run),
        "elapsed_seconds": time.monotonic() - started, "returncode": code, "failure": failure,
        "image": image, "cpu_limit": 1, "memory_limit": "2G",
        "verifier_environment": config["verifier_environment"],
        "manifest_sha256": file_hash(Path(config["dependency_manifest"])),
        "model_calls": 0, "benchmark_data_or_evaluator_invoked": False,
        "official_task_and_verifier_deadlines_changed": False,
        "probe_log_sha256": file_hash(run / "probe.log"),
    }
    persist(run / "ACCEPTANCE.json", receipt, immutable=True)
    if passed:
        persist(Path(config["dependency_acceptance"]), receipt, immutable=True)
    return receipt


def persist(path: Path, value: Any, *, immutable: bool = False) -> None:
    """Durable atomic commit; a committed callback can never be overwritten."""
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise CampaignBlocked(f"Immutable receipt conflict: {path}")
        return
    staging = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.writing")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if immutable:
            os.link(staging, path)
            staging.unlink()
        else:
            staging.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if staging.exists():
            staging.unlink()


def valid_counter(value: Any) -> bool:
    return type(value) is int and value >= 0


def canonical_usage(raw: dict[str, Any] | None, complete: bool) -> dict[str, Any]:
    result = dict.fromkeys(USAGE_FIELDS)
    if not complete:
        return result
    raw = raw or {}
    result.update({
        "input_tokens": raw.get("prompt_tokens_including_cache"),
        "output_tokens": raw.get("output_tokens"),
        "cache_read_tokens": raw.get("cache_read_input_tokens"),
        "cache_write_tokens": raw.get("cache_creation_input_tokens"),
        "total_tokens": raw.get("total_tokens"),
    })
    if not all(valid_counter(result[key]) for key in USAGE_FIELDS if key != "cost_usd"):
        raise CampaignBlocked("Complete provider usage is missing measured counters")
    if result["input_tokens"] + result["output_tokens"] != result["total_tokens"]:
        raise CampaignBlocked("Provider token counters contradict one another")
    if result["cache_read_tokens"] + result["cache_write_tokens"] > result["input_tokens"]:
        raise CampaignBlocked("Disjoint cache buckets exceed canonical input")
    return result


def sdk_usage(result: dict[str, Any]) -> dict[str, Any]:
    return canonical_usage(normalize_runner_usage(result.get("usage")),
                           result.get("usage_complete") is True)


def settled_verifier_skip(directory: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Authorize only this diagnosed, fully settled adapter failure, not an outcome rerun."""
    provenance = result.get("provenance") or {}
    if result.get("native_score") is not None or result["usage"]["total_tokens"] is None:
        raise CampaignBlocked("Only a measured, unscored infrastructure failure is recoverable")
    worker = Path(provenance["sdk_worker_dir"])
    native_path = Path(provenance["native_trial_result"])
    native, sdk = load(native_path), load(worker / "result.json")
    exception = native.get("exception_info") or {}
    if (
        not settled_output_stop(sdk) or native.get("verifier_result") is not None
        or exception.get("exception_type") != "RuntimeError"
        or not str(exception.get("exception_message", "")).startswith(
            "Official SDK worker did not complete;"
        )
        or sdk_usage(sdk) != result["usage"]
    ):
        raise CampaignBlocked("Predecessor failure is not the diagnosed verifier-skipped output stop")
    accounting = load(worker / "usage.json")
    starts, ends, tool_starts, tool_ends = [], [], 0, 0
    with (worker / "llm-streams.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "llm/start":
                starts.append(row["attempt_id"])
            elif row.get("kind") == "llm/end":
                if row.get("completed") is not True:
                    raise CampaignBlocked("An unsettled provider attempt cannot be replayed")
                ends.append(row["attempt_id"])
            elif row.get("kind") == "bridge/tool/start":
                tool_starts += 1
            elif row.get("kind") == "bridge/tool/end":
                tool_ends += 1
    terminal = [
        json.loads(line) for line in
        (worker.parent / "native-backend/terminal-exec.jsonl").read_text().splitlines()
    ]
    native_starts = sum(row.get("event") == "exec_start" for row in terminal)
    native_ends = sum(row.get("event") == "exec_result" for row in terminal)
    if (
        accounting.get("complete") is not True or accounting.get("unknown_reasons")
        or not starts or len(starts) != len(set(starts)) or sorted(starts) != sorted(ends)
        or len(ends) != len(set(ends)) or accounting.get("attempt_count") != len(starts)
        or native_starts != native_ends or tool_starts != tool_ends
        or native_starts != tool_starts
    ):
        raise CampaignBlocked("Ambiguous provider/native actions are not eligible for re-execution")
    environment = load(worker.parent / "HARBOR_ENVIRONMENT.json")
    container = environment["container_id"]
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise CampaignBlocked("Invalid owned native container identity")
    observed = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", container],
        capture_output=True, text=True, timeout=30,
    )
    if observed.returncode == 0:
        raise CampaignBlocked("Retained native workspace requires verifier-only recovery, not re-execution")
    absent = observed.stderr.lower()
    if container not in absent or not any(
        phrase in absent for phrase in ("no such object", "no such container")
    ):
        raise CampaignBlocked("Cannot verify that the failed actor's native workspace was deleted")
    if any(path.name != "manifest.json" for path in (native_path.parent / "artifacts").rglob("*")
           if path.is_file()):
        raise CampaignBlocked("Preserved actor artifacts need verifier-only recovery review")
    return {
        "reason": "settled_output_stop_skipped_native_verifier",
        "sdk_finish_reason": "max-tokens", "settled_provider_attempts": len(starts),
        "settled_native_calls": native_ends, "owned_container_deleted": True,
        "native_verifier_ran": False, "native_caps_changed": False,
        "old_callback": str(directory), "sdk_result_sha256": file_hash(worker / "result.json"),
        "native_result_sha256": file_hash(native_path),
        "provider_usage_sha256": file_hash(worker / "usage.json"),
        "native_action_audit_sha256": file_hash(worker.parent / "native-backend/terminal-exec.jsonl"),
        "all_failed_attempt_costs_retained": True,
    }


def retired_runtime_failure(directory: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Keep aborted accounting unknown; never resume an old provider or native call."""
    if result.get("native_score") is not None:
        raise CampaignBlocked("A completed native outcome must not be re-executed")
    provenance = result["provenance"]
    worker = Path(provenance["sdk_worker_dir"])
    private = worker.parent
    native_path = Path(provenance["native_trial_result"])
    native = load(native_path)
    exception = native.get("exception_info") or {}
    events = [json.loads(line) for line in
              (private / "native-backend/terminal-exec.jsonl").read_text().splitlines()]
    failures = [row for row in events if row.get("event") == "exec_failure"]
    if not failures or any(
        row.get("error_type") != "RuntimeError"
        or not re.fullmatch(r"Command timed out after \d+(?:\.\d+)? seconds", row.get("error", ""))
        for row in failures
    ):
        raise CampaignBlocked("Old native failure is outside the diagnosed command-timeout repair")
    starts, ends, aborted = [], [], []
    for line in (worker / "llm-streams.jsonl").open():
        row = json.loads(line)
        if row.get("kind") == "llm/start":
            starts.append(row["attempt_id"])
        elif row.get("kind") == "llm/end":
            ends.append(row["attempt_id"])
            if row.get("completed") is not True:
                aborted.append(row["attempt_id"])
    if sorted(starts) != sorted(ends) or len(starts) != len(set(starts)):
        raise CampaignBlocked("An old provider request still lacks a terminal transport event")
    native_starts = sum(row.get("event") == "exec_start" for row in events)
    terminal_events = sum(row.get("event") in ("exec_result", "exec_failure") for row in events)
    if native_starts != terminal_events:
        raise CampaignBlocked("An old native call still lacks a terminal client event")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if str(private).encode() in (proc / "cmdline").read_bytes():
                raise CampaignBlocked(f"Old SDK worker is still live: {proc.name}")
            if (proc / "cwd").resolve().is_relative_to(private):
                raise CampaignBlocked(f"Old SDK child is still live: {proc.name}")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    container = load(private / "HARBOR_ENVIRONMENT.json")["container_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", container):
        raise CampaignBlocked("Invalid retired native container identity")
    inspected = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", container],
        capture_output=True, text=True, timeout=30,
    )
    if inspected.returncode == 0:
        raise CampaignBlocked("Retained workspace needs verifier-only recovery, not a fresh actor")
    if container not in inspected.stderr or "no such " not in inspected.stderr.lower():
        raise CampaignBlocked("Cannot verify retirement of the failed native workspace")
    if any(
        path.name != "manifest.json"
        for path in (native_path.parent / "artifacts").rglob("*") if path.is_file()
    ):
        raise CampaignBlocked("Retained actor artifacts require verifier-only recovery review")
    sdk = load(worker / "result.json") if (worker / "result.json").is_file() else {}
    trial_log = (native_path.parent / "trial.log").read_text()
    stdout_path = native_path.parent / "verifier/test-stdout.txt"
    verifier_stdout = stdout_path.read_text() if stdout_path.is_file() else ""
    if (
        exception.get("exception_type") == "RuntimeError"
        and str(exception.get("exception_message", "")).startswith("Official SDK worker did not complete;")
        and sdk.get("status") == "completed" and sdk.get("usage_complete") is True
        and native.get("verifier_result") is None
        and (private / "bridge/BRIDGE_FAILURE.json").is_file()
    ):
        reason = "ordinary_native_command_timeout_fenced_bridge_and_skipped_verifier"
    elif (
        exception.get("exception_type") == "AgentTimeoutError"
        and "VerifierTimeoutError" in trial_log
        and "downloading uv 0.9.5" in verifier_stdout
        and native.get("verifier_result") is None
    ):
        reason = "native_verifier_dependency_download_timed_out_after_actor_deadline"
    else:
        raise CampaignBlocked("A model failure or unknown verifier failure is not an infrastructure retry")
    accounting = recover_runner_accounting(worker)
    measured = canonical_usage(
        normalize_runner_usage(accounting.get("usage")), accounting.get("complete") is True,
    )
    known = normalize_runner_usage(accounting.get("known_reported_usage"))
    if measured != result["usage"] or known != provenance.get("known_reported_usage"):
        raise CampaignBlocked("Recovered provider accounting differs from its immutable failed ledger")
    return {
        "reason": reason, "old_callback": str(directory),
        "original_provider_attempt_ids": starts, "aborted_provider_attempt_ids": aborted,
        "aborted_attempt_usage": "unknown, retained; never filled with zero",
        "native_timeout_failures": len(failures),
        "native_mutation_outcomes": "unknown for timeout calls, confined to the deleted old workspace",
        "old_sdk_processes_absent": True, "old_container_deleted": True,
        "recovery": "Fresh isolated trial after runtime repair, never reissue old call/session IDs",
        "native_caps_changed": False,
        "native_result_sha256": file_hash(native_path),
        "native_action_audit_sha256": file_hash(private / "native-backend/terminal-exec.jsonl"),
        "provider_receipts_sha256": file_hash(worker / "llm-streams.jsonl"),
        "provider_usage_complete": accounting.get("complete") is True,
        "provider_attempt_count": accounting.get("attempt_count"),
        "provider_unknown_reasons": accounting.get("unknown_reasons", []),
        "usage": measured, "known_reported_usage": known,
        "all_failed_attempt_costs_retained": True,
    }


def native_callback_result(request: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metadata = (summary.get("agent_context") or {}).get("metadata") or {}
    reward = (summary.get("native_rewards") or {}).get("reward")
    if type(reward) not in (int, float) or not math.isfinite(reward) or reward not in (0, 1):
        reward = None
    valid = summary.get("valid_for_comparison") is True and reward is not None
    complete = metadata.get("deepseek_usage_complete") is True
    usage = canonical_usage(metadata.get("deepseek_usage"), complete)
    return {
        "request_id": request["request_id"], "task_id": request["task"]["task_id"],
        "identity": request["identity"],
        "execution_status": "completed" if valid else "blocked",
        "native_score": reward if valid else None,
        "success": bool(reward) if valid else None, "usage": usage,
        "provenance": {
            "native_status": summary.get("status"),
            "native_exception": summary.get("native_exception"),
            "native_actor_stop": metadata.get("native_actor_stop"),
            "sdk_finish_reason": metadata.get("sdk_finish_reason"),
            "native_trial_result": summary.get("native_trial_result"),
            "native_timeout_is_scored_when_verifier_reward_exists": True,
            "usage_complete": complete,
            "known_reported_usage": metadata.get("deepseek_known_reported_usage"),
            "provider_attempt_count": metadata.get("provider_attempt_count"),
            "sampling_seed_supported": False,
            "seed_semantics": "Fixed paired repetition label, not seeded model sampling",
            "sdk_worker_dir": metadata.get("sdk_worker_dir"),
            "private_artifacts": metadata.get("private_artifacts"),
            "cumulative_token_limit": None,
        },
    }


def recovered_verifier_summary(directory: Path, receipt_path: Path, scorer_sha256: str) -> dict:
    import tomllib

    receipt = load(receipt_path)
    original = load(directory / "NATIVE_SUMMARY.json")
    old_callback = load(directory / "RESULT.json")
    old_native = load(Path(original["native_trial_result"]))
    native_path = Path(receipt["native_result"])
    if (
        receipt.get("status") != "NATIVE_VERIFIER_RECOVERED"
        or Path(receipt["callback"]).resolve() != directory.resolve()
        or receipt.get("callback_sha256") != file_hash(directory / "RESULT.json")
        or receipt.get("original_native_result") != original["native_trial_result"]
        or receipt.get("original_native_result_sha256")
        != file_hash(Path(original["native_trial_result"]))
        or receipt.get("native_result_sha256") != file_hash(native_path)
        or receipt.get("official_verifier_sha256") != scorer_sha256
        or receipt.get("additional_actor_calls") != 0
        or receipt.get("original_actor_usage") != old_callback["usage"]
    ):
        raise CampaignBlocked("Native verifier recovery is not bound to the retained execution")
    task = directory / "task-source" / original["task_id"]
    limit = tomllib.loads((task / "task.toml").read_text())["verifier"]["timeout_sec"]
    if receipt.get("official_verifier_timeout_seconds") != limit:
        raise CampaignBlocked("Recovered verifier changed its original task deadline")
    native = load(native_path)
    rewards = (native.get("verifier_result") or {}).get("rewards")
    old_exception = old_native.get("exception_info")
    native_exception = native.get("exception_info")
    exception_transition_valid = (
        native_exception == old_exception
        or (
            isinstance(old_exception, dict)
            and old_exception.get("exception_type") == "VerifierTimeoutError"
            and native_exception is None
        )
    )
    if (
        not isinstance(rewards, dict) or set(rewards) != {"reward"}
        or type(rewards["reward"]) not in (int, float) or rewards["reward"] not in (0, 1)
        or not exception_transition_valid
        or native.get("task_name") != original["task_id"]
        or receipt.get("task_id") != original["task_id"]
        or receipt.get("score") != rewards["reward"]
        or native.get("agent_result") != old_native.get("agent_result")
        or native.get("agent_execution") != old_native.get("agent_execution")
    ):
        raise CampaignBlocked("Recovered verifier did not produce the exact native task verdict")
    result = deepcopy(original)
    result.update(
        native_rewards=rewards, valid_for_comparison=True,
        native_exception=native_exception,
        native_trial_result=str(native_path), status="valid_native_verdict",
        verifier_recovery={"receipt": str(receipt_path), "sha256": file_hash(receipt_path)},
    )
    return result


def recovered_callback(
    directory: Path, result: dict[str, Any], recovery: dict[str, str], scorer_sha256: str,
) -> dict[str, Any]:
    original = Path(recovery["callback"])
    if (
        file_hash(original / "REQUEST.json") != file_hash(directory / "REQUEST.json")
        or file_hash(original / "RESULT.json") != file_hash(directory / "RESULT.json")
        or load(directory / "REQUEST.json")["phase"] != "train"
        or result.get("native_score") is not None
    ):
        raise CampaignBlocked("A callback recovery must preserve the exact unscored TRAIN request")
    summary = recovered_verifier_summary(original, Path(recovery["receipt"]), scorer_sha256)
    verified = native_callback_result(load(directory / "REQUEST.json"), summary)
    return {
        **result,
        **{key: verified[key] for key in ("execution_status", "native_score", "success", "usage")},
        "provenance": {
            **result["provenance"], **verified["provenance"],
            "original_callback": str(original), "verifier_recovery": recovery["receipt"],
            "verifier_recovery_sha256": file_hash(Path(recovery["receipt"])),
        },
    }


def compact_public_trace(path: Path, *, events: int = 12, characters: int = 7000) -> str:
    """Only the actor's terminal bridge audit, never verifier output or SDK secrets."""
    if not path.is_file():
        return json.dumps({"public_actor_events": [], "trace_available": False})
    public = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "exec_start":
            args = row.get("arguments") or {}
            item = {"event": "terminal_command", "command": args.get("command", ""),
                    "cwd": args.get("cwd"), "timeout_sec": args.get("timeout_sec")}
        elif row.get("event") == "exec_result":
            observation = row.get("observation") or {}
            item = {"event": "terminal_result",
                    "return_code": observation.get("return_code"),
                    "stdout": str(observation.get("stdout") or "")[:1000],
                    "stderr": str(observation.get("stderr") or "")[:500]}
        elif row.get("event") in (
            "exec_failure", "native_command_timeout_observed", "uncertain_command_replay_refused",
        ):
            item = {"event": row["event"], "error_type": row.get("error_type"),
                    "error": str(row.get("error") or "")[:500],
                    "execution_status": row.get("execution_status"),
                    "mutation_replay_allowed": False}
        else:
            continue
        if PRIVATE_TRACE.search(json.dumps(item)):
            continue
        public.append(item)
    chosen = public if len(public) <= events else public[:events // 2] + public[-events // 2:]
    while chosen and len(json.dumps(chosen, ensure_ascii=False)) > characters:
        chosen.pop(len(chosen) // 2)
    return json.dumps({
        "public_actor_events": chosen, "trace_available": True,
        "public_event_count": len(public), "selection": "first/last public terminal events",
        "verifier_or_oracle_contents_included": False,
    }, ensure_ascii=False)


def optimizer_payload(request: dict[str, Any], train_ids: set[str]) -> dict[str, Any]:
    evidence = request["training_evidence"]
    if request["phase"] != "train" or {
        row["task"]["task_id"] for row in evidence
    } != train_ids or len(evidence) != len(train_ids):
        raise CampaignBlocked("Optimizer requires all and only the 18 TRAIN observations")
    if any(
        result.get("native_score") is None
        for row in evidence for result in [row["baseline"], *row["candidates"].values()]
    ):
        raise CampaignBlocked("Complete TRAIN native-verdict gate is closed; optimizer not invoked")
    instructions = {row["id"]: row["instruction"]
                    for row in request.get("previous_candidates", [])}

    def public_result(result: dict[str, Any]) -> dict[str, Any]:
        feedback = result.get("feedback", "")
        return {"native_score": result["native_score"],
                "execution_status": result["execution_status"],
                "public_actor_trace": json.loads(feedback) if feedback else None}

    return {
        "objective": (
            request["objective"] + "\nReturn ONLY a JSON object with exactly instruction "
            "(string) and supporting_task_ids (array of supplied TRAIN IDs). "
            f"Maximum {request['limits']['max_characters']} characters and "
            f"{request['limits']['max_words']} words. An empty instruction is an abstention. "
            "Do not request tools or include markdown."
        ),
        "training_evidence": [
            {"task": row["task"], "seed": row["seed"],
             "baseline": public_result(row["baseline"]),
             "candidates": {
                 key: {"instruction": instructions.get(key), **public_result(value)}
                 for key, value in row["candidates"].items()
             }} for row in evidence
        ],
    }


def parse_proposal(response: Any, train_ids: set[str], limits: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, str):
        raise ValueError("Optimizer final_response is not text")
    value = json.loads(response)
    if not isinstance(value, dict) or set(value) != {"instruction", "supporting_task_ids"}:
        raise ValueError("Optimizer response must contain only instruction/supporting_task_ids")
    text, support = value["instruction"], value["supporting_task_ids"]
    if (
        not isinstance(text, str) or not isinstance(support, list)
        or any(not isinstance(key, str) for key in support)
        or len(set(support)) != len(support) or not set(support) <= train_ids
        or (text.strip() and not support)
        or len(text) > limits["max_characters"] or len(text.split()) > limits["max_words"]
        or "\0" in text
    ):
        raise ValueError("Optimizer proposal violates sparse TRAIN-only support contract")
    return value


def semantic_key(request: dict[str, Any], task_hash: str, protocol_hash: str) -> str:
    """No cross-phase/no-op pooling, and test acquisitions stay run/policy specific."""
    value = {
        "phase": request["phase"], "arm": request["arm"], "seed": request["seed"],
        "task_id": request["task"]["task_id"], "task_sha256": task_hash,
        "prompt": request["prompt"], "residual": request.get("system_prompt_append", ""),
        "identity": request["identity"], "native_protocol_sha256": protocol_hash,
    }
    if request["phase"] != "train" or request["arm"] != "baseline":
        value.update(run_id=request["run_id"], candidate_sha256=request["candidate_sha256"],
                     final_policy_sha256=request.get("final_policy_sha256"),
                     source_sha256=request.get("source_sha256"))
    return canonical_hash(value)


def sum_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def known(row: dict[str, Any], key: str) -> int | float:
        value = row["usage"].get(key)
        if value is not None:
            return value
        raw = row.get("known_reported_usage") or {}
        normalized = normalize_runner_usage(raw) if "inputTokens" in raw else raw
        native_key = {
            "input_tokens": "prompt_tokens_including_cache",
            "cache_read_tokens": "cache_read_input_tokens",
            "cache_write_tokens": "cache_creation_input_tokens",
        }.get(key, key)
        value = normalized.get(native_key)
        return value if value is not None else 0

    return {
        "attempts": len(rows),
        "provider_attempts_known_subtotal": sum(
            row["provider_attempt_count"] for row in rows
            if valid_counter(row.get("provider_attempt_count"))
        ),
        "measured": {key: sum(row["usage"][key] for row in rows)
                     if all(row["usage"].get(key) is not None for row in rows) else None
                     for key in USAGE_FIELDS},
        "known_subtotals": {
            key: sum(known(row, key) for row in rows)
            for key in USAGE_FIELDS
        },
        "unknown_attempts": [row["attempt_id"] for row in rows
                             if row["usage"].get("total_tokens") is None],
    }


def unique_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = {}
    for row in rows:
        key = row["attempt_id"]
        if key in found and found[key]["usage"] != row["usage"]:
            raise CampaignBlocked(f"Conflicting exact-once cost receipt: {key}")
        if key not in found or found[key].get("stage") == "inherited_setup":
            found[key] = row
    return [found[key] for key in sorted(found)]


def complete_test(rows: list[dict[str, Any]], test_ids: set[str], seeds: list[int]) -> bool:
    expected = {(task, seed, arm) for task in test_ids for seed in seeds
                for arm in ("baseline", "chisel")}
    observed = []
    for row in rows:
        request, result = row["request"], row["result"]
        if request["phase"] != "test":
            continue
        if result["native_score"] is None:
            return False
        observed.append((request["task"]["task_id"], request["seed"], request["arm"]))
    return len(observed) == len(set(observed)) and set(observed) == expected


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "schema": SCHEMA, "authorized_hostname": AUTHORIZED_HOST,
        "dataset_revision": TERMINAL_REVISION, "official_revision": UPSTREAM,
        "official_tag": TAG, "partition_sha256": PARTITION_SHA256,
        "model": MODEL_NAME, "wire_api": "anthropic_messages", "harbor_version": HARBOR_VERSION,
        "counts": {"train": 18, "validation": 18, "test": 53}, "split_seed": 20260928,
        "execution_seeds": [20260928], "optimization_iterations": 2,
        "application": "system_section", "cumulative_token_limit": None,
        "optimizer_max_units": None, "sampling_seed_supported": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Campaign identity, population or unlimited lifecycle differs")
    if type(config.get("executor_workers")) is not int or config["executor_workers"] not in (2, 4, 8):
        raise ValueError("Declare two, four or eight independent native executor workers")
    seeds = config.get("test_execution_seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(type(seed) is not int for seed in seeds):
        raise ValueError("Both TEST arms require five fixed repetition labels")
    if not 0 < config["heartbeat_seconds"] <= 60 or not 0 < config["capacity_audit_seconds"] <= 300:
        raise ValueError("Heartbeat/capacity audits must remain durable and timely")
    shared = config.get("shared_parent_prompt")
    if shared is not None:
        from phiagent.harness.industrial_common_prompt import prompt_manifest

        if shared != {
            key: prompt_manifest(
                shared.get("benchmark"), shared.get("version"), shared.get("sha256"),
            )[key]
            for key in ("benchmark", "version", "sha256")
        }:
            raise ValueError("Shared parent prompt identity differs from the frozen public brief")


def shutdown_runtime_hash(source: Path) -> str:
    names = {
        "NativeTerminalBackend", "QuiescentTerminalBridge", "live_sdk_group",
        "quiesce_sdk_worker", "campaign_terminal_bridge", "settled_output_stop",
        "NativeCompletionAgent", "canonical_usage", "sdk_usage", "load", "persist",
    }
    nodes = [
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
    ]
    if {node.name for node in nodes} != names:
        raise CampaignBlocked("The complete accepted SDK shutdown implementation is missing")
    return canonical_hash([ast.dump(node, include_attributes=False) for node in nodes])


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        if file_hash(root / relative) != expected:
            raise CampaignBlocked(f"Frozen source changed: {relative}")


def prepare(config: dict[str, Any], run: Path) -> None:
    validate_config(config)
    if socket.gethostname() != AUTHORIZED_HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Only a800-4 CPU/API execution with CUDA_VISIBLE_DEVICES='' is authorized")
    if not run.is_relative_to(CAMPAIGN / "terminal-campaigns") or run == CAMPAIGN / "terminal-campaigns":
        raise ValueError("Reserve a new directory below campaign/terminal-campaigns")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    persist(run / "CONFIG.json", config, immutable=True)
    persist(run / "RUNTIME_CONFIG.json", load(Path(config["runtime_template"]))["runtime"],
            immutable=True)
    source = run / "source"
    hashes = freeze_source(Path(config["native_source_root"]), source)
    for relative in CONTROLLER_FILES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        hashes[relative] = file_hash(target)
    persist(run / "SOURCE_SHA256.json", hashes, immutable=True)
    persist(run / "GIT_STATE.json", {
        "head": read_command(["git", "rev-parse", "HEAD"], ROOT),
        "scoped_status": read_command(
            ["git", "status", "--porcelain", "--", *CONTROLLER_FILES], ROOT),
        "authoritative_working_tree_hashes": hashes,
        "native_source_root": config["native_source_root"],
        "native_origin_git_state": load(Path(config["native_source_root"]).parent / "GIT_STATE.json"),
        "uploaded_source_git_state": (
            load(ROOT / "CAMPAIGN_UPLOAD_GIT_STATE.json")
            if (ROOT / "CAMPAIGN_UPLOAD_GIT_STATE.json").is_file() else None
        ),
    }, immutable=True)
    persist(run / "COMMANDS.json", {
        "launcher_argv": sys.argv, "cwd": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
        "cumulative_token_limit": None,
        "continuation": "Automatic TRAIN -> optimizer -> VAL -> freeze -> paired sealed TEST",
    }, immutable=True)
    private = CAMPAIGN / "runs" / f"terminal-campaign-{run.name}"
    private.mkdir(mode=0o700)
    (run / "private").symlink_to(private, target_is_directory=True)
    persist(run / "PRIVATE_ROOT.json", {"path": str(private)}, immutable=True)
    persist(run / "RESOURCE_INVENTORY.json", resource_inventory(), immutable=True)
    persist(run / "PACKAGES.json", {
        "python": sys.version, "executable": sys.executable, "hostname": socket.gethostname(),
        "packages": sorted((item.metadata["Name"], item.version)
                           for item in importlib.metadata.distributions()),
    }, immutable=True)
    if config.get("callback_predecessor"):
        previous = Path(config["callback_predecessor"])
        state = load(previous / "SUMMARY.json")
        if (
            not previous.resolve().is_relative_to(CAMPAIGN / "terminal-campaigns")
            or state["status"] != "blocked" or state.get("active")
            or state["coverage"]["test"] or state["coverage"]["validation"]
        ):
            raise CampaignBlocked("Callback continuation requires a quiescent TRAIN-only predecessor")
        proc = Path("/proc") / str(state["pid"]) / "stat"
        if proc.is_file() and proc.read_text().rsplit(")", 1)[1].split()[0] != "Z":
            raise CampaignBlocked("The predecessor controller has not exited")
        verify_hashes(previous / "source", load(previous / "SOURCE_SHA256.json"))
        previous_config = load(previous / "CONFIG.json")
        logical_id = previous_config.get("logical_run_id", previous.name)
        if config.get("logical_run_id") != logical_id:
            raise CampaignBlocked("Retained callback continuation changed its logical run identity")
        for path in (previous / "callbacks").glob("*/*/REQUEST.json"):
            request = load(path)
            if (
                request["phase"] != "train" or request["run_id"] != logical_id
                or not (path.parent / "COMMITTED.json").is_file()
            ):
                raise CampaignBlocked("Only immutable settled TRAIN callbacks can continue")
        skipped = config.get("uninvoked_optimizer_gates", {})
        undispatched = config.get("undispatched_executor_requests", {})
        for request_id, proof in skipped.items():
            directory = previous / "callbacks/optimizer" / request_id
            private = Path(load(previous / "PRIVATE_ROOT.json")["path"])
            if (
                proof.get("optimizer_invoked") is not False
                or proof["committed_sha256"] != file_hash(directory / "COMMITTED.json")
                or (private / "optimizer" / request_id).exists()
                or not load(directory / "CALLBACK_FAILURE.json").get("message", "").startswith(
                    "Complete TRAIN native-verdict gate"
                )
            ):
                raise CampaignBlocked("Only a proven pre-invocation optimizer gate may be retried")
        for request_id, proof in undispatched.items():
            directory = previous / "callbacks/executor" / request_id
            native_path = Path(proof["native_result"])
            if (
                proof["committed_sha256"] != file_hash(directory / "COMMITTED.json")
                or proof["native_result_sha256"] != file_hash(native_path)
            ):
                raise CampaignBlocked("Unstarted native failure attestation changed")
            attest_undispatched_network_failure(
                load(native_path), load(directory / "REQUEST.json")["task"]["task_id"],
            )

        def omit_uninvoked(directory, names):
            if Path(directory) == previous / "callbacks/optimizer":
                return set(names) & set(skipped)
            if Path(directory) == previous / "callbacks/executor":
                return set(names) & set(undispatched)
            return set()

        shutil.copytree(previous / "callbacks", run / "callbacks", ignore=omit_uninvoked)
        persist(run / "CALLBACK_PREDECESSOR.json", {
            "source": str(previous), "source_manifest_sha256": file_hash(previous / "SOURCE_SHA256.json"),
            "logical_run_id": config["logical_run_id"], "completed_executions_repeated": False,
            "failure_overlays": config.get("callback_verifier_recoveries", {}),
            "uninvoked_optimizer_gates": skipped,
            "undispatched_executor_requests": undispatched,
        }, immutable=True)


class Campaign:
    def __init__(self, run: Path):
        self.run = run
        self.config = load(run / "CONFIG.json")
        self.source = run / "source"
        self.hashes = load(run / "SOURCE_SHA256.json")
        self.private = Path(load(run / "PRIVATE_ROOT.json")["path"])
        self.runtime = load(run / "RUNTIME_CONFIG.json")
        self.partition = load(Path(self.config["partition_dir"]) / "SPLIT_MANIFEST.json")
        self.records = {row["id"]: row for row in self.partition["records"]}
        self.identity = {
            "runtime": "official_deepseek_harness_sdk", "wire_api": "anthropic_messages",
            "model": MODEL_NAME, "official_revision": UPSTREAM, "official_tag": TAG,
            "runtime_manifest_sha256": file_hash(Path(self.runtime["manifest"])),
            "transport": "official_native_anthropic_messages",
        }
        if self.config.get("shared_parent_prompt"):
            self.identity["shared_prompt_sha256"] = self.config["shared_parent_prompt"]["sha256"]
        self.protocol = {
            "identity": self.identity, "harbor_version": HARBOR_VERSION,
            "dataset_revision": TERMINAL_REVISION, "partition_sha256": PARTITION_SHA256,
            "native_sources": {name: self.hashes[name] for name in NATIVE_FILES},
            "max_output_tokens": self.config["max_output_tokens"],
            "native_timeout_multiplier": 1.0, "native_attempts": 1, "native_retries": 0,
            "tools": "unchanged accepted Harbor terminal bridge", "CUDA_VISIBLE_DEVICES": "",
            "shared_parent_prompt": self.config.get("shared_parent_prompt"),
        }
        self.protocol_hash = canonical_hash(self.protocol)
        self.phase = "preflight"
        self.lock = threading.RLock()
        self.active: dict[str, dict[str, Any]] = {}
        self.stop = threading.Event()
        self.controller = None
        self.pilot = None
        self.baseline_cache: dict[str, dict[str, Any]] = {}

    def preflight(self) -> None:
        validate_config(self.config)
        if socket.gethostname() != AUTHORIZED_HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise CampaignBlocked("Campaign cannot execute outside the authorized CPU/API host")
        if importlib.metadata.version("harbor") != HARBOR_VERSION:
            raise CampaignBlocked("The native Harbor version changed")
        verify_hashes(self.source, self.hashes)
        partition = Path(self.config["partition_dir"])
        for filename, key in (
            ("SPLIT_MANIFEST.json", "partition_sha256"),
            ("TRAIN_VALIDATION.json", "train_validation_sha256"),
            ("SEALED_TEST.json", "sealed_test_sha256"),
        ):
            if file_hash(partition / filename) != self.config[key]:
                raise CampaignBlocked(f"Frozen partition file changed: {filename}")
        if self.partition["counts"] != self.config["counts"]:
            raise CampaignBlocked("Partition population differs")
        from phiagent.harness.benchmark_partition import validate_partition
        from integrations.deepseek_harness.runner import verify_runtime
        validate_partition(self.partition)
        acceptance = load(Path(self.config["acceptance_receipt"]))
        validate_acceptance(acceptance, self.runtime, self.source)
        original_acceptance = Path(acceptance["actual_synthetic_acceptance"])
        if file_hash(original_acceptance) != acceptance["actual_synthetic_acceptance_sha256"]:
            raise CampaignBlocked("Accepted SDK receipt changed")
        verify_runtime(self.runtime)
        source_receipt = load(Path(self.config["native_source_receipt"]))
        if not source_receipt.get("verified") or source_receipt["upstream_revision"] != TERMINAL_REVISION:
            raise CampaignBlocked("Native task-source verification is absent")
        task_hashes = {}
        for task_id, record in self.records.items():
            directory = Path(self.config["task_root"]) / task_id
            hashes = task_file_hashes(directory)
            if canonical_hash(hashes) != record["payload_sha256"]:
                raise CampaignBlocked(f"Pinned native task source changed: {task_id}")
            if hashes.get("instruction.md") != record["public_task_sha256"]:
                raise CampaignBlocked(f"Public prompt commitment changed: {task_id}")
            task_hashes[task_id] = hashes
        persist(self.run / "TASK_SOURCE_SHA256.json", task_hashes, immutable=True)
        persist(self.run / "SPLIT_MANIFEST.json", self.partition, immutable=True)
        persist(self.run / "NATIVE_PROTOCOL.json", self.protocol, immutable=True)
        persist(self.run / "SDK_ACCEPTANCE.json", acceptance, immutable=True)
        persist(self.run / "SDK_RUNTIME_MANIFEST.json", load(Path(self.runtime["manifest"])),
                immutable=True)
        harbor = Path(importlib.metadata.distribution("harbor").locate_file("harbor"))
        scorer = {path.relative_to(harbor).as_posix(): file_hash(path)
                  for subdir in ("verifier", "trial", "models/task", "models/trial")
                  for path in sorted((harbor / subdir).rglob("*.py"))}
        if not scorer:
            raise CampaignBlocked("Native scorer source files are absent")
        persist(self.run / "SCORER_SHA256.json", scorer, immutable=True)
        self.harbor_root = harbor
        self.scorer_hashes = scorer
        if self.config.get("native_verifier_caches"):
            from scripts.recover_terminal_verifier import verified_cache

            for task_id, manifest_path in self.config["native_verifier_caches"].items():
                if task_id not in self.records:
                    raise CampaignBlocked("Verifier cache task is outside the frozen population")
                verified_cache(Path(manifest_path), Path(self.config["task_root"]) / task_id)
            persist(self.run / "NATIVE_VERIFIER_CACHES.json", {
                task_id: {"manifest": path, "sha256": file_hash(Path(path))}
                for task_id, path in self.config["native_verifier_caches"].items()
            }, immutable=True)
        if self.config.get("dependency_manifest"):
            manifest_path = Path(self.config["dependency_manifest"])
            manifest = load(manifest_path)
            acceptance = load(Path(self.config["dependency_acceptance"]))
            if (
                acceptance.get("status") != "passed"
                or acceptance.get("manifest_sha256") != file_hash(manifest_path)
                or acceptance.get("verifier_environment") != self.config["verifier_environment"]
            ):
                raise CampaignBlocked("Native verifier dependency acceptance differs from the frozen plan")
            opener = build_opener(ProxyHandler({}))
            for asset in manifest["assets"]:
                url = self.config["dependency_base_url"] + asset["route"]
                with opener.open(Request(url, method="HEAD"), timeout=15) as response:
                    if int(response.headers["Content-Length"]) != asset["size"]:
                        raise CampaignBlocked("Public dependency mirror changed")
            persist(self.run / "DEPENDENCY_MANIFEST.json", manifest, immutable=True)
            persist(self.run / "DEPENDENCY_ACCEPTANCE.json", acceptance, immutable=True)
        if self.config.get("sdk_shutdown_acceptance"):
            shutdown = load(Path(self.config["sdk_shutdown_acceptance"]))
            accepted_source = Path(self.config.get(
                "sdk_shutdown_source", self.source / "scripts/run_deepseek_terminal_campaign.py",
            ))
            if (
                shutdown.get("status") != "passed"
                or shutdown.get("controller_sha256") != file_hash(accepted_source)
                or shutdown_runtime_hash(accepted_source) != shutdown_runtime_hash(
                    self.source / "scripts/run_deepseek_terminal_campaign.py"
                )
                or shutdown.get("official_revision") != UPSTREAM
                or shutdown.get("model") != MODEL_NAME
                or shutdown.get("provider_attempts_started_after_cancellation") != []
                or shutdown.get("usage_complete") is not True
                or file_hash(Path(shutdown["usage_receipt"])) != shutdown["usage_receipt_sha256"]
            ):
                raise CampaignBlocked("The frozen SDK shutdown path lacks its real-provider acceptance")
            persist(self.run / "SDK_SHUTDOWN_ACCEPTANCE.json", shutdown, immutable=True)
            persist(self.run / "SDK_SHUTDOWN_SOURCE_EQUIVALENCE.json", {
                "accepted_source": str(accepted_source),
                "accepted_source_sha256": file_hash(accepted_source),
                "current_source_sha256": self.hashes["scripts/run_deepseek_terminal_campaign.py"],
                "shutdown_runtime_sha256": shutdown_runtime_hash(accepted_source),
                "native_adapter_files_unchanged": self.protocol["native_sources"],
                "additional_acceptance_provider_calls": 0,
            }, immutable=True)
        self.import_pilot()
        self.import_predecessors()
        persist(self.run / "COMPLETION_ADAPTER.json", {
            "import_path": "scripts.run_deepseek_terminal_campaign:NativeCompletionAgent",
            "source_sha256": self.hashes["scripts/run_deepseek_terminal_campaign.py"],
            "change": (
                "Native command timeouts become explicit unknown-outcome observations without "
                "bridge poisoning or replay; settled SDK output stops reach the native verifier; "
                "unscored containers are retained; cancellation responses are withheld until "
                "the SDK process group stops, without delaying native command cancellation"
            ),
            "native_actor_protocol_sha256": self.protocol_hash,
            "native_task_deadline_extended": False, "new_provider_calls_on_stop": 0,
            "native_command_timeout": "Public unknown-outcome observation, not a permanently fenced bridge",
            "unresolved_identical_commands": "Refused without executing them again",
            "verifier_environment": self.config.get("verifier_environment", {}),
            "failed_native_containers": "Preserved stopped; only scored owned containers removed",
            "unknown_cost_does_not_fill_or_suppress_native_scores": True,
            "cumulative_token_limit": None,
        }, immutable=True)
        persist(self.run / "PREFLIGHT.json", {
            "status": "passed", "hostname": socket.gethostname(),
            "native_protocol_sha256": self.protocol_hash,
            "dataset_revision": TERMINAL_REVISION, "task_files": sum(map(len, task_hashes.values())),
            "CUDA_VISIBLE_DEVICES": "", "selected_gpus": [], "cumulative_token_limit": None,
            "test_prompts_opened": False,
        }, immutable=True)

    def import_pilot(self) -> None:
        path = Path(self.config["pilot_summary"])
        if file_hash(path) != self.config["pilot_summary_sha256"]:
            raise CampaignBlocked("Completed pilot receipt changed")
        summary = load(path)
        root = path.parent
        manifest = load(root / "RUN_MANIFEST.json")
        previous = load(root / "SOURCE_SHA256.json")
        differences = [name for name in NATIVE_FILES if previous.get(name) != self.hashes[name]]
        worker = load(next((root / "private").glob("*/worker-job.json")))
        expected = {
            "model": MODEL_NAME, "runtime": self.runtime, "max_tokens": self.config["max_output_tokens"],
            "seed": 20260928, "arm": "baseline", "chisel": None,
        }
        differences.extend(key for key, value in expected.items() if worker.get(key) != value)
        for key, value in {
            "task_sha256": self.records[summary["task_id"]]["payload_sha256"],
            "partition_sha256": PARTITION_SHA256, "harbor_version": HARBOR_VERSION,
            "official_deepseek_revision": UPSTREAM, "dataset_revision": TERMINAL_REVISION,
        }.items():
            if manifest.get(key) != value:
                differences.append(key)
        eligible = (not differences and summary.get("valid_for_comparison") is True
                    and summary["task_id"] in self.partition["splits"]["train"])
        self.pilot = summary if eligible else None
        persist(self.run / "PILOT_IMPORT.json", {
            "source": str(path), "sha256": file_hash(path), "eligible_training_reuse": eligible,
            "source_or_protocol_mismatches": differences,
            "cost_role": "TRAIN acquisition once if equivalent; otherwise separate inherited setup",
        }, immutable=True)
        rows = []
        for attempt in summary["attempts"]:
            usage = canonical_usage(attempt.get("usage"), all(
                attempt.get("usage", {}).get(key) is not None
                for key in ("prompt_tokens_including_cache", "output_tokens", "total_tokens",
                            "cache_read_input_tokens", "cache_creation_input_tokens")
            ))
            provider_count = (
                0 if attempt.get("provider_never_started") else
                summary["agent_context"]["metadata"].get("provider_attempt_count")
                if attempt.get("native_trial_result") == summary.get("native_trial_result") else None
            )
            rows.append({**attempt, "stage": "inherited_setup", "usage": usage,
                         "provider_attempt_count": provider_count})
        persist(self.run / "INHERITED_ATTEMPTS.json", unique_attempts(rows), immutable=True)
        # SDK operability attempts are shared setup, not task or optimizer execution.
        shared = []
        for usage_path in sorted((CAMPAIGN / "runs").glob(
            "synthetic-sdk-acceptance-20260929-v*/*/usage.json"
        )):
            value = load(usage_path)
            shared.append({
                "attempt_id": str(usage_path.parent), "stage": "shared_sdk_setup",
                "usage": canonical_usage(normalize_runner_usage(value.get("usage")),
                                         value.get("complete") is True),
                "usage_receipt": str(usage_path), "usage_sha256": file_hash(usage_path),
                "known_reported_usage": value.get("known_reported_usage"),
                "provider_attempt_count": value.get("attempt_count"),
            })
        acceptances = [self.config["sdk_shutdown_acceptance"]] if self.config.get(
            "sdk_shutdown_acceptance"
        ) else []
        acceptances.extend(self.config.get("additional_sdk_setup_acceptances", []))
        for acceptance in acceptances:
            shutdown = load(Path(acceptance))
            usage_path = Path(shutdown["usage_receipt"])
            if file_hash(usage_path) != shutdown["usage_receipt_sha256"]:
                raise CampaignBlocked("Retained SDK setup usage changed")
            value = load(usage_path)
            shared.append({
                "attempt_id": shutdown["worker_dir"], "stage": "shared_sdk_setup",
                "usage": canonical_usage(normalize_runner_usage(value.get("usage")),
                                         value.get("complete") is True),
                "usage_receipt": str(usage_path), "usage_sha256": file_hash(usage_path),
                "known_reported_usage": value.get("known_reported_usage"),
                "provider_attempt_count": value.get("attempt_count"),
                "purpose": "Real-SDK cancellation acceptance; no benchmark data or scorer",
            })
        persist(self.run / "SHARED_SDK_SETUP.json", unique_attempts(shared), immutable=True)

    def import_predecessors(self) -> None:
        attempts, imports = [], []
        if self.config.get("callback_predecessor"):
            previous = Path(self.config["callback_predecessor"])
            if load(previous / "NATIVE_PROTOCOL.json") != self.protocol:
                raise CampaignBlocked("Retained TRAIN callbacks changed their actor/native protocol")
            for name in (
                "INHERITED_ATTEMPTS.json", "SHARED_SDK_SETUP.json", "PREDECESSOR_ATTEMPTS.json",
            ):
                if (previous / name).is_file():
                    attempts.extend(load(previous / name))
            for proof in self.config.get("uninvoked_optimizer_gates", {}).values():
                attempts.extend({
                    **row, "stage": "inherited_infrastructure",
                } for row in load(Path(proof["callback"]) / "USAGE.json")["attempts"])
            for proof in self.config.get("retired_incomplete_executor_attempts", {}).values():
                callback = Path(proof["callback"])
                checks = {
                    callback / "COMMITTED.json": proof["committed_sha256"],
                    Path(proof["native_result"]): proof["native_result_sha256"],
                    Path(proof["sdk_result"]): proof["sdk_result_sha256"],
                    Path(proof["usage_receipt"]): proof["usage_receipt_sha256"],
                }
                if (
                    proof.get("actor_completed") is not False
                    or proof.get("usage_complete") is not False
                    or proof.get("retry_reason") != "official_sdk_worker_did_not_complete"
                    or any(not path.is_file() or file_hash(path) != expected
                           for path, expected in checks.items())
                ):
                    raise CampaignBlocked("Retired incomplete actor evidence changed")
                attempts.extend({
                    **row, "stage": "inherited_infrastructure",
                    "unknown_usage_preserved": row["usage"]["total_tokens"] is None,
                } for row in load(callback / "USAGE.json")["attempts"])
            for proof in self.config.get("undispatched_executor_requests", {}).values():
                attempts.extend({
                    **row, "stage": "inherited_infrastructure",
                    "zero_call_attestation": proof,
                } for row in load(Path(proof["callback"]) / "USAGE.json")["attempts"])
        for name in self.config.get("predecessor_roots", []):
            root = Path(name).resolve()
            if not root.is_relative_to(CAMPAIGN / "terminal-campaigns") or root == self.run:
                raise CampaignBlocked("Invalid predecessor campaign identity")
            with (root / ".controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            if load(root / "NATIVE_PROTOCOL.json") != self.protocol:
                raise CampaignBlocked("Cannot pool a changed predecessor actor/model/native protocol")
            verify_hashes(root / "source", load(root / "SOURCE_SHA256.json"))
            for filename in (
                "INHERITED_ATTEMPTS.json", "SHARED_SDK_SETUP.json", "PREDECESSOR_ATTEMPTS.json",
            ):
                if (root / filename).is_file():
                    attempts.extend(load(root / filename))
            for request_path in sorted((root / "callbacks").glob("*/*/REQUEST.json")):
                directory = request_path.parent
                request = load(request_path)
                result = self.committed(directory)
                if result is None:
                    raise CampaignBlocked(f"Ambiguous predecessor callback is not replayable: {directory}")
                if directory.parent.name == "optimizer":
                    worker = Path(load(root / "PRIVATE_ROOT.json")["path"]) / "optimizer"
                    if result["execution_status"] != "blocked" or worker.exists():
                        raise CampaignBlocked("Cannot reuse or discard an invoked predecessor optimizer")
                    attempts.extend({
                        **row, "stage": "inherited_infrastructure",
                    } for row in load(directory / "USAGE.json")["attempts"])
                    imports.append({
                        "callback": str(directory), "request_id": request["request_id"],
                        "action": "preserve_pre_optimizer_native_verdict_gate_failure",
                        "optimizer_invoked": False,
                    })
                    continue
                if request["phase"] != "train" or request["arm"] != "baseline":
                    raise CampaignBlocked("Changed-source continuation accepts TRAIN baselines only")
                recovered_summary = None
                recovery = self.config.get("native_verifier_recoveries", {}).get(str(directory))
                if recovery:
                    if result["native_score"] is not None:
                        raise CampaignBlocked("A scored native result cannot be replaced by recovery")
                    recovered_summary = recovered_verifier_summary(
                        directory, Path(recovery), self.scorer_hashes["verifier/verifier.py"],
                    )
                    result = native_callback_result(request, recovered_summary)
                if (directory / "PROCESS.json").is_file():
                    if not (directory / "PROCESS_EXIT.json").is_file():
                        raise CampaignBlocked(f"Predecessor has no settled process exit: {directory}")
                    process = load(directory / "PROCESS.json")
                    proc = Path(f"/proc/{process['pid']}/stat")
                    if proc.is_file() and proc.read_text().split()[21] == process["proc_start_ticks"]:
                        raise CampaignBlocked(f"Predecessor native process is still live: {directory}")
                elif not (directory / "REUSE.json").is_file():
                    raise CampaignBlocked(f"Predecessor has neither an execution nor a reuse receipt: {directory}")
                task_id = request["task"]["task_id"]
                if request["identity"] != self.identity or task_id not in self.partition["splits"]["train"]:
                    raise CampaignBlocked("Predecessor baseline identity or TRAIN membership differs")
                key = semantic_key(request, self.records[task_id]["payload_sha256"], self.protocol_hash)
                if load(directory / "SEMANTICS.json")["semantic_sha256"] != key:
                    raise CampaignBlocked("Predecessor semantic commitment differs")
                rows = load(directory / "USAGE.json")["attempts"]
                item = {
                    "task_id": task_id, "callback": str(directory),
                    "request_id": request["request_id"], "semantic_sha256": key,
                    "committed_sha256": file_hash(directory / "COMMITTED.json"),
                }
                if result["native_score"] is not None:
                    self.assert_result(result, executor=True)
                    summary = recovered_summary or load(directory / "NATIVE_SUMMARY.json")
                    recomputed = native_callback_result(request, summary)
                    if any(recomputed[field] != result[field] for field in (
                        "native_score", "usage", "identity", "execution_status",
                    )):
                        raise CampaignBlocked("Predecessor callback and native verdict disagree")
                    if key in self.baseline_cache:
                        previous = self.baseline_cache[key]
                        if (
                            previous["summary"]["native_trial_result"] != summary["native_trial_result"]
                            or {row["attempt_id"]: row["usage"] for row in previous["attempts"]}
                            != {row["attempt_id"]: row["usage"] for row in rows}
                        ):
                            raise CampaignBlocked("Distinct baseline acquisitions need explicit reconciliation")
                    else:
                        self.baseline_cache[key] = {
                            "summary": summary, "attempts": rows, "source": str(directory),
                            "committed_sha256": item["committed_sha256"],
                        }
                    item["action"] = "reuse_completed_baseline_once"
                    if recovery:
                        item.update(
                            action="reuse_retained_actor_with_recovered_native_verifier",
                            native_verifier_recovery=recovery,
                            native_verifier_recovery_sha256=file_hash(Path(recovery)),
                            additional_actor_calls=0,
                        )
                    attempts.extend({**row, "stage": "inherited_training_executor"} for row in rows)
                else:
                    sdk_path = Path(result["provenance"]["sdk_worker_dir"]) / "result.json"
                    sdk = load(sdk_path) if sdk_path.is_file() else {}
                    proof = (
                        settled_verifier_skip(directory, result) if settled_output_stop(sdk)
                        else retired_runtime_failure(directory, result)
                    )
                    if proof.get("reason") == "native_verifier_dependency_download_timed_out_after_actor_deadline":
                        if not (self.run / "DEPENDENCY_ACCEPTANCE.json").is_file():
                            raise CampaignBlocked("Verifier download repair has not passed its native-image gate")
                    item.update(action="new_isolated_execution_after_diagnosed_infrastructure_failure",
                                recovery_proof=proof)
                    attempts.extend({**row, "stage": "inherited_infrastructure"} for row in rows)
                imports.append(item)
        persist(self.run / "PREDECESSOR_IMPORT.json", {
            "roots": self.config.get("predecessor_roots", []),
            "callbacks": imports, "native_actor_protocol_sha256": self.protocol_hash,
            "completed_outcomes_reexecuted": False, "ambiguous_actions_replayed": False,
            "failure_repair": (
                "Native timeout handoff, unchanged verifier with verified dependency transport, "
                "retained unscored workspaces and exact-once measured/unknown failed costs"
            ),
        }, immutable=True)
        persist(self.run / "PREDECESSOR_ATTEMPTS.json", unique_attempts(attempts), immutable=True)

    def tasks(self, phase: str) -> list[dict[str, str]]:
        directory = Path(self.config["partition_dir"])
        if phase == "test":
            if not (self.run / "FROZEN_POLICY.json").is_file():
                raise CampaignBlocked("Sealed TEST release requires a persisted frozen policy")
            policy = load(self.run / "FROZEN_POLICY.json")
            persist(self.run / "TEST_RELEASE.json", {
                "policy_sha256": policy["policy_sha256"],
                "source_sha256": policy["source_sha256"],
                "sealed_test_sha256": self.config["sealed_test_sha256"],
                "test_execution_seeds": self.config["test_execution_seeds"],
            }, immutable=True)
            prepared = load(directory / "SEALED_TEST.json")
            items = prepared["test"] if isinstance(prepared, dict) else prepared
        else:
            items = load(directory / "TRAIN_VALIDATION.json")[phase]
        if [row["id"] for row in items] != self.partition["splits"][phase]:
            raise CampaignBlocked(f"{phase} membership/order changed")
        tasks = []
        for row in items:
            path = Path(row["path"])
            if path.resolve() != (Path(self.config["task_root"]) / row["id"]).resolve():
                raise CampaignBlocked("Native task path differs from frozen population")
            text = (path / "instruction.md").read_text()
            if hashlib.sha256(text.encode()).hexdigest() != self.records[row["id"]]["public_task_sha256"]:
                raise CampaignBlocked("Public prompt changed")
            tasks.append({"task_id": row["id"], "unit_id": row["id"], "public_prompt": text})
        return tasks

    def callback_directory(self, kind: str, request: dict[str, Any]) -> tuple[Path, bool]:
        path = self.run / "callbacks" / kind / request["request_id"]
        fresh = not path.exists()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        persist(path / "REQUEST.json", request, immutable=True)
        return path, fresh

    def check_sources(self) -> None:
        verify_hashes(self.source, self.hashes)
        verify_hashes(self.harbor_root, self.scorer_hashes)
        if file_hash(Path(self.runtime["manifest"])) != self.identity["runtime_manifest_sha256"]:
            raise CampaignBlocked("Official SDK runtime manifest changed")
        if self.config.get("dependency_manifest"):
            manifest = load(Path(self.config["dependency_manifest"]))
            if manifest != load(self.run / "DEPENDENCY_MANIFEST.json"):
                raise CampaignBlocked("Verifier dependency manifest changed")
            for asset in manifest["assets"]:
                if file_hash(Path(asset["path"])) != asset["sha256"]:
                    raise CampaignBlocked("A public verifier dependency archive changed")

    def wait_capacity(self) -> None:
        while True:
            free = shutil.disk_usage(self.run).free / 2**30
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(memory["MemAvailable"].split()[0]) / 2**20
            if (free >= self.config["minimum_free_root_gib"]
                    and available >= self.config["minimum_available_memory_gib"]):
                return
            persist(self.run / "CAPACITY_WAIT.json", {
                "at": utc_now(), "free_root_gib": free, "available_memory_gib": available,
                "reason": "Waiting for native CPU/memory/disk capacity; no unrelated jobs preempted",
            })
            time.sleep(60)

    def process(self, directory: Path, command: list[str], *, kind: str) -> int:
        cache = directory / "process-cache"
        cache.mkdir(exist_ok=True)
        persist(directory / "COMMAND.json", {
            "argv": command, "cwd": str(directory), "PYTHONPATH": str(self.source),
            "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(cache), "cumulative_token_limit": None,
        }, immutable=True)
        with (directory / "process.log").open("xb") as log:
            process = subprocess.Popen(
                command, cwd=directory,
                env={**os.environ, "PYTHONPATH": str(self.source), "CUDA_VISIBLE_DEVICES": "",
                     "TMPDIR": str(cache), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        persist(directory / "PROCESS.json", {
            "pid": process.pid, "controller_pid": os.getpid(), "started_at": utc_now(),
            "proc_start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split()[21],
        }, immutable=True)
        with self.lock:
            self.active[directory.name] = {
                "kind": kind, "pid": process.pid, "directory": str(directory),
                "started_at": utc_now(),
            }
        try:
            code = process.wait()
            persist(directory / "PROCESS_EXIT.json", {"returncode": code}, immutable=True)
            return code
        finally:
            with self.lock:
                self.active.pop(directory.name, None)

    def harvest_only(self, directory: Path) -> None:
        """A crash may leave an independent native process; wait, never re-submit it."""
        path = directory / "PROCESS.json"
        if not path.is_file():
            raise CampaignBlocked(f"Uncommitted callback has ambiguous launch state: {directory}")
        receipt = load(path)
        proc = Path(f"/proc/{receipt['pid']}/stat")
        while proc.exists():
            try:
                fields = proc.read_text().split()
            except FileNotFoundError:
                break
            if fields[21] != receipt["proc_start_ticks"] or fields[2] == "Z":
                break
            time.sleep(5)

    def descriptor(self, request: dict[str, Any], directory: Path) -> dict[str, Any] | None:
        instruction = request.get("system_prompt_append", "")
        if request["prompt"] != request["task"]["public_prompt"]:
            raise CampaignBlocked("system_section must preserve the original native prompt")
        residual = request.get("residual")
        if not instruction:
            if residual is not None:
                raise CampaignBlocked("No-op has nonempty residual metadata")
            return None
        if not residual or residual["instruction"] != instruction or residual["application"] != "system_section":
            raise CampaignBlocked("Residual application metadata differs")
        proofs = []
        for result_path in (self.run / "callbacks" / "optimizer").glob("*/RESULT.json"):
            result = load(result_path)
            if result["execution_status"] == "completed" and result["instruction"].strip() == instruction:
                req = load(result_path.parent / "REQUEST.json")
                proofs.append({
                    "optimizer_request_id": req["request_id"], "iteration": req["iteration"],
                    "optimizer_result_sha256": file_hash(result_path),
                    "optimizer_request_sha256": file_hash(result_path.parent / "REQUEST.json"),
                    "optimizer_payload_sha256": file_hash(result_path.parent / "PROVIDER_PROMPT.json"),
                    "supporting_task_ids": result["supporting_task_ids"],
                    "optimizer_task_ids": sorted(row["task"]["task_id"]
                                                 for row in req["training_evidence"]),
                    "provider_usage": result["usage"],
                })
        if not proofs:
            raise CampaignBlocked("A residual needs an actual completed TRAIN optimizer receipt")
        artifact = directory / "instruction.txt"
        if artifact.exists():
            if artifact.read_text() != instruction:
                raise CampaignBlocked("Instruction artifact changed")
        else:
            with artifact.open("x") as stream:
                stream.write(instruction)
                stream.flush()
                os.fsync(stream.fileno())
        descriptor = {
            "path": str(artifact), "sha256": file_hash(artifact),
            "artifact_kind": "training-derived",
            "provenance": {"method": "CHISEL", "application": "system_section",
                           "partition_sha256": PARTITION_SHA256, "optimizer_receipts": proofs,
                           "candidate_sha256": request["candidate_sha256"],
                           "final_policy_sha256": request.get("final_policy_sha256"),
                           "source_sha256": request.get("source_sha256")},
        }
        persist(directory / "INSTRUCTION_PROVENANCE.json", descriptor, immutable=True)
        return descriptor

    def commit_result(self, directory: Path, result: dict[str, Any],
                      attempts: list[dict[str, Any]]) -> None:
        persist(directory / "USAGE.json", {
            "usage": result["usage"], "attempts": attempts, "unknown_is_zero": False,
        }, immutable=True)
        persist(directory / "RESULT.json", result, immutable=True)
        persist(directory / "COMMITTED.json", {
            "request_sha256": file_hash(directory / "REQUEST.json"),
            "result_sha256": file_hash(directory / "RESULT.json"),
            "usage_sha256": file_hash(directory / "USAGE.json"),
        }, immutable=True)

    def committed(self, directory: Path) -> dict[str, Any] | None:
        if not (directory / "COMMITTED.json").is_file():
            if not all((directory / name).is_file() for name in ("RESULT.json", "USAGE.json")):
                return None
            request, result = load(directory / "REQUEST.json"), load(directory / "RESULT.json")
            if (result.get("request_id") != request["request_id"]
                    or result.get("identity") != request["identity"]
                    or load(directory / "USAGE.json")["usage"] != result["usage"]):
                raise CampaignBlocked("Uncommitted result does not match its immutable request")
            self.commit_result(directory, result, load(directory / "USAGE.json")["attempts"])
        receipt = load(directory / "COMMITTED.json")
        for name in ("request", "result", "usage"):
            if file_hash(directory / f"{name.upper()}.json") != receipt[f"{name}_sha256"]:
                raise CampaignBlocked("Completed callback receipt changed")
        result = load(directory / "RESULT.json")
        overlay = self.config.get("callback_verifier_recoveries", {}).get(directory.name)
        if overlay:
            result = recovered_callback(
                directory, result, overlay, self.scorer_hashes["verifier/verifier.py"],
            )
        return result

    def guarded_callback(self, kind: str, request: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.execute(request) if kind == "executor" else self.optimize(request)
        except Exception as exc:
            directory, _ = self.callback_directory(kind, request)
            persist(directory / "CALLBACK_FAILURE.json", {
                "type": type(exc).__name__, "message": str(exc),
                "request_id": request["request_id"], "actions_replayed": False,
            })
            if self.committed(directory) is None:
                usage = dict.fromkeys(USAGE_FIELDS)
                attempt_id = str(directory)
                known, count = None, None
                if kind == "optimizer":
                    worker = self.private / "optimizer" / request["request_id"]
                    attempt_id = str(worker)
                    if worker.is_dir():
                        accounting = recover_runner_accounting(worker)
                        usage = canonical_usage(
                            normalize_runner_usage(accounting.get("usage")),
                            accounting.get("complete") is True,
                        )
                        known = accounting.get("known_reported_usage")
                        count = accounting.get("attempt_count")
                result = {
                    "request_id": request["request_id"], "identity": request["identity"],
                    "execution_status": "blocked", "usage": usage,
                    "provenance": {"callback_failure": str(directory / "CALLBACK_FAILURE.json"),
                                   "known_reported_usage": known, "provider_attempt_count": count},
                }
                if kind == "executor":
                    result.update(task_id=request["task"]["task_id"], native_score=None, success=None)
                else:
                    result.update(instruction="", supporting_task_ids=[])
                self.commit_result(directory, result, [{
                    "attempt_id": attempt_id, "stage": f"blocked_{kind}", "usage": usage,
                    "known_reported_usage": known, "provider_attempt_count": count,
                    "request_id": request["request_id"],
                }])
            raise

    def assert_result(self, result: dict[str, Any], *, executor: bool) -> dict[str, Any]:
        if executor and result["native_score"] is None:
            raise CampaignBlocked("Missing native verdict; no zero substitution or action replay")
        if not executor and result["execution_status"] == "blocked":
            raise CampaignBlocked("Optimizer callback remains blocked; ambiguous calls not repeated")
        return result

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        self.check_sources()
        directory, fresh = self.callback_directory("executor", request)
        prior = self.committed(directory)
        if prior is not None:
            return prior
        task_id = request["task"]["task_id"]
        key = semantic_key(request, self.records[task_id]["payload_sha256"], self.protocol_hash)
        descriptor = self.descriptor(request, directory)
        persist(directory / "SEMANTICS.json", {
            "semantic_sha256": key, "native_protocol_sha256": self.protocol_hash,
            "outer_arm": request["arm"], "sdk_arm": "chisel" if descriptor else "baseline",
            "no_op_is_not_a_synthetic_treatment": descriptor is None,
        }, immutable=True)
        reused = (self.pilot is not None and request["phase"] == "train"
                  and request["arm"] == "baseline" and task_id == self.pilot["task_id"]
                  and request["seed"] == 20260928 and descriptor is None)
        cached = self.baseline_cache.get(key)
        if cached is not None:
            summary = cached["summary"]
            persist(directory / "REUSE.json", {
                "source": cached["source"], "committed_sha256": cached["committed_sha256"],
                "semantic_sha256": key, "native_actor_protocol_verified": True,
                "completion_adapter_change": "Does not affect this already scored baseline",
            }, immutable=True)
        elif reused:
            summary = self.pilot
            persist(directory / "REUSE.json", {
                "source": self.config["pilot_summary"], "sha256": self.config["pilot_summary_sha256"],
                "semantic_sha256": key, "source_and_protocol_verified": True,
            }, immutable=True)
        else:
            if fresh:
                self.wait_capacity()
                task_path = directory / "task-source" / task_id
                shutil.copytree(Path(self.config["task_root"]) / task_id, task_path)
                expected = load(self.run / "TASK_SOURCE_SHA256.json")[task_id]
                if task_file_hashes(task_path) != expected:
                    raise CampaignBlocked("Native task bytes changed during snapshot")
                private = self.private / "executor" / request["request_id"]
                private.mkdir(parents=True, mode=0o700)
                kwargs = {
                    "runner_python": self.config["sdk_python"],
                    "runner_script": str(self.source / "integrations/deepseek_harness/runner.py"),
                    "runtime": self.runtime, "api_key_file": self.config["api_key_file"],
                    "private_root": str(private), "max_tokens": self.config["max_output_tokens"],
                    "seed": request["seed"], "arm": "chisel" if descriptor else "baseline",
                    "chisel": descriptor, "source_revision": UPSTREAM,
                }
                shared = self.config.get("shared_parent_prompt")
                if shared is not None:
                    kwargs.update(
                        shared_prompt_version=shared["version"],
                        shared_prompt_sha256=shared["sha256"],
                        benchmark=shared["benchmark"],
                    )
                from harbor.models.job.config import JobConfig
                native_job = build_native_job(directory, task_path, kwargs)
                native_job["agents"][0]["import_path"] = (
                    "scripts.industrial_terminal_agents:SharedPromptDeepSeek"
                    if shared is not None else
                    "scripts.run_deepseek_terminal_campaign:NativeCompletionAgent"
                )
                native_job["verifier"]["env"] = self.config.get("verifier_environment", {})
                native_job["environment"]["kwargs"] = {"keep_containers": True}
                job = JobConfig.model_validate(native_job)
                job_path = directory / "HARBOR_JOB.json"
                persist(job_path, job.model_dump(mode="json"), immutable=True)
                native_cache = self.config.get("native_verifier_caches", {}).get(task_id)
                if native_cache:
                    command = [
                        self.config["terminal_python"],
                        str(self.source / "scripts/recover_terminal_verifier.py"),
                        "trial", "--native-job", str(job_path),
                        "--cache", native_cache, "--output", str(directory),
                    ]
                else:
                    command = [
                        str(Path(self.config["terminal_python"]).parent / "harbor"),
                        "run", "--config", str(job_path),
                    ]
                self.process(directory, command, kind="native_executor")
            else:
                self.harvest_only(directory)
            summary = summarize_native_trial(directory, [])
            summary["purpose"] = "Full campaign native callback; phase is in REQUEST.json"
            summary["split"] = request["phase"]
            persist(directory / "NATIVE_SUMMARY.json", summary, immutable=True)
        if cached is not None or reused:
            persist(directory / "NATIVE_SUMMARY.json", summary, immutable=True)
        result = native_callback_result(request, summary)
        result["provenance"].update(
            semantic_sha256=key, native_protocol_sha256=self.protocol_hash,
            outer_arm=request["arm"], sdk_arm="chisel" if descriptor else "baseline",
            reused_completed_pilot=reused,
            reused_predecessor_callback=cached["source"] if cached else None,
        )
        if request["phase"] == "train":
            private_dir = result["provenance"].get("private_artifacts")
            result["feedback"] = compact_public_trace(
                Path(private_dir) / "native-backend/terminal-exec.jsonl"
            ) if private_dir else json.dumps({"public_actor_events": [], "trace_available": False})
        stage = ("training_executor" if request["phase"] == "train" else
                 "validation_selection" if request["phase"] == "validation" else
                 f"final_{request['arm']}")
        attempts = []
        cached_counts = {
            row["attempt_id"]: row.get("provider_attempt_count")
            for row in cached["attempts"]
        } if cached is not None else {}
        for attempt in summary["attempts"]:
            measured = attempt.get("usage", {})
            usage = canonical_usage(measured, all(
                measured.get(field) is not None for field in (
                    "prompt_tokens_including_cache", "output_tokens", "total_tokens",
                    "cache_read_input_tokens", "cache_creation_input_tokens",
                )
            ))
            attempts.append({**attempt, "usage": usage, "stage": stage,
                             "request_id": request["request_id"],
                             "provider_attempt_count": (
                                 cached_counts[attempt["attempt_id"]]
                                 if attempt["attempt_id"] in cached_counts else
                                 0 if attempt.get("provider_never_started") else
                                 result["provenance"].get("provider_attempt_count")
                                 if attempt.get("native_trial_result") == summary.get("native_trial_result")
                                 else None
                             )})
        self.commit_result(directory, result, attempts)
        if not reused and cached is None and result["provenance"].get("private_artifacts"):
            private = Path(result["provenance"]["private_artifacts"])
            container = load(private / "HARBOR_ENVIRONMENT.json")["container_id"]
            info = subprocess.run(
                ["docker", "inspect", "--format", "{{json .}}", container],
                capture_output=True, text=True, timeout=30, check=True,
            )
            retire_networks(
                network_owner(json.loads(info.stdout)), directory / "NETWORK_CLEANUP.json",
            )
        if result["native_score"] is not None and result["usage"]["total_tokens"] is not None:
            persist(self.run / "semantic-cache" / f"{key}.json", {
                "semantic_sha256": key, "callback": str(directory),
                "result_sha256": file_hash(directory / "RESULT.json"),
            }, immutable=True)
        self.snapshot()
        if result["native_score"] is None:
            persist(directory / "UNRESOLVED_NATIVE.json", {
                "native_score": None, "reason": "No official native verdict",
                "workspace_retained": not reused and cached is None,
                "phase_gate": "No optimizer/selection/final completion with missing native scores",
                "native_exception": summary.get("native_exception"),
            }, immutable=True)
            return result
        if not reused and cached is None:
            self.cleanup_scored_container(directory, result)
        return result

    def cleanup_scored_container(self, directory: Path, result: dict[str, Any]) -> None:
        private = result.get("provenance", {}).get("private_artifacts")
        if not private:
            return
        receipt = load(Path(private) / "HARBOR_ENVIRONMENT.json")
        container = receipt["container_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", container):
            raise CampaignBlocked("Invalid owned scored container identity")
        observed = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", container],
            capture_output=True, text=True, timeout=30,
        )
        if observed.returncode != 0:
            return
        if json.loads(observed.stdout).get("Running"):
            raise CampaignBlocked("Refusing to remove a still-running scored container")
        command = ["docker", "rm", "--volumes", container]
        cleanup = subprocess.run(command, capture_output=True, text=True, timeout=60)
        persist(directory / "SCORED_CONTAINER_CLEANUP.json", {
            "container_id": container, "argv": command, "returncode": cleanup.returncode,
            "native_verdict_preserved": True, "result_sha256": file_hash(directory / "RESULT.json"),
        }, immutable=True)

    def optimize(self, request: dict[str, Any]) -> dict[str, Any]:
        self.check_sources()
        directory, fresh = self.callback_directory("optimizer", request)
        prior = self.committed(directory)
        if prior is not None:
            return self.assert_result(prior, executor=False)
        train_ids = set(self.partition["splits"]["train"])
        payload = optimizer_payload(request, train_ids)
        persist(directory / "PROVIDER_PROMPT.json", payload, immutable=True)
        private = self.private / "optimizer" / request["request_id"]
        job = {
            "schema_version": 1, "run_dir": str(private), "purpose": "benchmark",
            "arm": "baseline", "chisel": None, "session_id": f"optimizer-{request['request_id'][:24]}",
            "seed": 20260928, "model": MODEL_NAME, "max_tokens": self.config["max_output_tokens"],
            "prompt": json.dumps(payload, ensure_ascii=False), "tools": [], "bridge": None,
            "api_key_file": self.config["api_key_file"], "runtime": self.runtime,
            "source_revision": UPSTREAM,
        }
        persist(directory / "SDK_JOB.json", job, immutable=True)
        persist(directory / "OPTIMIZER_DATA.json", {
            "task_ids": sorted(train_ids), "task_count": len(train_ids), "split": "train",
            "iteration": request["iteration"], "tools": [],
            "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
            "validation_or_test_evidence": False, "official_verifier_source": False,
        }, immutable=True)
        if fresh:
            self.process(directory, [
                self.config["sdk_python"],
                str(self.source / "integrations/deepseek_harness/runner.py"),
                "--job", str(directory / "SDK_JOB.json"),
            ], kind="official_sdk_optimizer")
        else:
            self.harvest_only(directory)
        if not (private / "result.json").is_file():
            raise CampaignBlocked("Optimizer has no settled SDK result; ambiguous call not repeated")
        native = load(private / "result.json")
        usage = sdk_usage(native)
        status, proposal, failure = "failed", {"instruction": "", "supporting_task_ids": []}, None
        try:
            if native.get("status") != "completed":
                raise ValueError("Official SDK optimizer did not complete")
            proposal = parse_proposal(native.get("final_response"), train_ids, request["limits"])
            status = "completed"
        except (ValueError, TypeError) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
        result = {
            "request_id": request["request_id"], "identity": request["identity"],
            "execution_status": status, **proposal, "usage": usage,
            "provenance": {
                "sdk_result": str(private / "result.json"),
                "sdk_result_sha256": file_hash(private / "result.json"),
                "usage_complete": native.get("usage_complete") is True,
                "failed_proposal": failure, "actual_optimizer_task_ids": sorted(train_ids),
                "optimizer_data_size": len(train_ids), "tools": [],
                "cumulative_token_limit": None,
            },
        }
        accounting = load(private / "usage.json") if (private / "usage.json").is_file() else {}
        self.commit_result(directory, result, [{
            "attempt_id": str(private), "stage": "training_optimizer", "usage": usage,
            "provider_attempt_count": accounting.get("attempt_count"),
            "known_reported_usage": accounting.get("known_reported_usage"),
            "request_id": request["request_id"],
        }])
        self.snapshot()
        return self.assert_result(result, executor=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            rows, attempts = [], []
            for receipt in sorted((self.run / "callbacks").glob("*/*/COMMITTED.json")):
                directory = receipt.parent
                request, result = load(directory / "REQUEST.json"), self.committed(directory)
                rows.append({"request": request, "result": result})
                attempts.extend(load(directory / "USAGE.json")["attempts"])
            for filename in (
                "INHERITED_ATTEMPTS.json", "SHARED_SDK_SETUP.json", "PREDECESSOR_ATTEMPTS.json",
            ):
                if (self.run / filename).is_file():
                    attempts.extend(load(self.run / filename))
            attempts = unique_attempts(attempts)
            complete = complete_test(rows, set(self.partition["splits"]["test"]),
                                     self.config["test_execution_seeds"])
            stages = sorted({row["stage"] for row in attempts})
            output = {
                "schema": SCHEMA, "updated_at": utc_now(), "pid": os.getpid(),
                "status": self.phase, "final_test_complete": complete,
                "test_opened": (self.run / "TEST_OPENED.json").is_file() or any(
                    row["request"]["phase"] == "test" for row in rows
                ),
                "coverage": {
                    phase: sum(row["request"]["phase"] == phase
                               and "task_id" in row["result"]
                               and row["result"]["native_score"] is not None for row in rows)
                    for phase in ("train", "validation", "test")
                },
                "native_callback_counts": {
                    phase: sum(row["request"]["phase"] == phase
                               and "task_id" in row["result"] for row in rows)
                    for phase in ("train", "validation", "test")
                },
                "expected_final_test_observations": 530,
                "final_native_means": {
                    arm: (sum(row["result"]["native_score"] for row in rows
                              if row["request"]["phase"] == "test"
                              and row["request"]["arm"] == arm) / 265 if complete else None)
                    for arm in ("baseline", "chisel")
                },
                "execution_costs_complete": all(
                    row["result"]["usage"]["total_tokens"] is not None
                    for row in rows if "task_id" in row["result"]
                ),
                "unresolved_native_trials": [
                    row["request"]["request_id"] for row in rows
                    if "task_id" in row["result"] and row["result"]["native_score"] is None
                ],
                "costs_by_stage": {
                    stage: sum_usage([row for row in attempts if row["stage"] == stage])
                    for stage in stages
                },
                "cumulative_unique_attempts": sum_usage(attempts),
                "accounting_identity": "Unique native attempt/SDK run path; provider retries included",
                "optimizer_data_size": 18, "cumulative_token_limit": None,
                "active": list(self.active.values()), "benchmark_improvement_claim": None,
                "coverage_label": "full paired held-out test" if complete else "partial coverage",
            }
            persist(self.run / "SUMMARY.json", output)
            csv_stream = io.StringIO()
            writer = csv.DictWriter(csv_stream, fieldnames=[
                "phase", "arm", "task_id", "seed", "request_id", "native_score",
                "execution_status", "input_tokens", "output_tokens", "total_tokens",
            ])
            writer.writeheader()
            for row in rows:
                request, result = row["request"], row["result"]
                writer.writerow({
                    "phase": request["phase"], "arm": request.get("arm", "optimizer"),
                    "task_id": request.get("task", {}).get("task_id"), "seed": request.get("seed"),
                    "request_id": request["request_id"], "native_score": result.get("native_score"),
                    "execution_status": result["execution_status"],
                    **{key: result["usage"][key] for key in (
                        "input_tokens", "output_tokens", "total_tokens",
                    )},
                })
            path = self.run / ".SUMMARY.csv.writing"
            path.write_text(csv_stream.getvalue())
            path.replace(self.run / "SUMMARY.csv")
            return output

    def heartbeat(self) -> None:
        last_audit = 0.0
        while not self.stop.is_set():
            try:
                now = time.monotonic()
                if now - last_audit >= self.config["capacity_audit_seconds"]:
                    inventory = resource_inventory()
                    persist(self.run / "capacity" / f"{time.time_ns()}.json", inventory,
                            immutable=True)
                    last_audit = time.monotonic()
                with self.lock:
                    active = list(self.active.values())
                activity = []
                for path in self.private.glob("executor/*/*/native-backend/terminal-exec.jsonl"):
                    activity.append({
                        "path": str(path), "bytes": path.stat().st_size,
                        "mtime": path.stat().st_mtime,
                    })
                persist(self.run / "HEARTBEAT.json", {
                    "at": utc_now(), "pid": os.getpid(), "phase": self.phase,
                    "active": active, "recent_actor_traces": sorted(
                        activity, key=lambda item: item["mtime"], reverse=True)[:4],
                    "successor": {
                        "train": "full-validation", "validation": "policy/source-freeze",
                        "freeze": "sealed-53x5-paired-test", "test": "next-independent-test-pair",
                    }.get(self.phase, "train"),
                    "executor_workers": self.config["executor_workers"],
                    "CUDA_VISIBLE_DEVICES": "", "selected_gpus": [],
                    "cumulative_token_limit": None,
                })
                self.snapshot()
            except Exception as exc:
                persist(self.run / "HEARTBEAT_ERROR.json", {
                    "at": utc_now(), "type": type(exc).__name__, "message": str(exc),
                })
            self.stop.wait(self.config["heartbeat_seconds"])

    def lifecycle(self) -> int:
        self.preflight()
        from phiagent.harness.deepseek_chisel import ChiselConfig, DeepSeekChiselController
        config = ChiselConfig(
            application="system_section", optimization_iterations=2,
            executor_workers=self.config["executor_workers"],
            execution_seeds=tuple(self.config["execution_seeds"]),
            test_execution_seeds=tuple(self.config["test_execution_seeds"]),
            native_metric="official_task_verifier_binary_reward",
        )
        self.controller = DeepSeekChiselController(
            partition=self.partition,
            task_index=[{"task_id": key, "unit_id": key,
                         "public_prompt_sha256": value["public_task_sha256"]}
                        for key, value in self.records.items()],
            execute=lambda request: self.guarded_callback("executor", request),
            optimize=lambda request: self.guarded_callback("optimizer", request),
            execution_identity=self.identity,
            optimizer_identity=self.identity,
            run_id=self.config.get("logical_run_id", self.run.name), config=config,
        )
        thread = threading.Thread(target=self.heartbeat, name="durable-heartbeat", daemon=True)
        thread.start()
        try:
            self.phase = "train"
            report = self.controller.train(self.tasks("train"))
            require_train_settlement(report)
            persist(self.run / "TRAIN_COMPLETE.json", report, immutable=True)
            self.phase = "validation"
            selected = self.controller.select_validation(self.tasks("validation"))
            if not all(row["complete_native_scores"] for row in selected["results"]):
                raise CampaignBlocked("Validation is incomplete; freeze remains closed")
            persist(self.run / "VALIDATION_COMPLETE.json", selected, immutable=True)
            self.phase = "freeze"
            self.check_sources()
            policy = self.controller.freeze(source_hashes={
                "deepseek_harness": self.identity["runtime_manifest_sha256"],
                "executor": canonical_hash({
                    "native_actor_protocol": self.protocol_hash,
                    "completion_adapter": self.hashes["scripts/run_deepseek_terminal_campaign.py"],
                    "complete_frozen_source": canonical_hash(self.hashes),
                }),
                "optimizer": canonical_hash({
                    "driver": self.hashes["scripts/run_deepseek_terminal_campaign.py"],
                    "runtime": self.identity, "data": "all 18 TRAIN tasks only",
                }),
                "scorer": canonical_hash({
                    "harbor": self.scorer_hashes,
                    "native_tasks": load(self.run / "TASK_SOURCE_SHA256.json"),
                    "verifier_environment": self.config.get("verifier_environment", {}),
                    "dependency_manifest": (
                        load(self.run / "DEPENDENCY_MANIFEST.json")
                        if (self.run / "DEPENDENCY_MANIFEST.json").exists() else None
                    ),
                    "native_verifier_caches": (
                        load(self.run / "NATIVE_VERIFIER_CACHES.json")
                        if (self.run / "NATIVE_VERIFIER_CACHES.json").exists() else None
                    ),
                }),
            })
            persist(self.run / "FROZEN_POLICY.json", policy.to_dict(), immutable=True)
            self.phase = "test"

            def load_test():
                persist(self.run / "TEST_OPENED.json", {
                    "policy_sha256": policy.policy_sha256,
                    "source_sha256": policy.source_sha256,
                    "partition_sha256": self.config["partition_sha256"],
                }, immutable=True)
                return self.tasks("test")

            result = self.controller.evaluate_test(
                load_test, policy_sha256=policy.policy_sha256,
                source_sha256=policy.source_sha256,
            )
            summary = self.snapshot()
            if not summary["final_test_complete"]:
                raise CampaignBlocked("Final completion requires every paired official verdict")
            persist(self.run / "TEST_COMPLETE.json", result, immutable=True)
            self.phase = "complete"
            return 0
        finally:
            self.stop.set()
            thread.join(timeout=65)
            if self.controller is not None:
                persist(self.run / "CONTROLLER_REPORT.json", self.controller.report())
            self.snapshot()


def active_comparisons(
    root: Path, config: dict[str, Any], *, proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    conflicts = []
    for receipt in root.glob("*/PID.json"):
        directory = receipt.parent
        config_path = directory / "CONFIG.json"
        if not config_path.is_file():
            continue
        previous = load(config_path)
        if any(previous.get(key) != config.get(key) for key in (
            "schema", "partition_sha256", "model", "wire_api",
            "execution_seeds", "test_execution_seeds",
        )):
            continue
        identity = load(receipt)
        try:
            argv = (proc_root / str(identity["pid"]) / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        script = str(directory / "source/scripts/run_deepseek_terminal_campaign.py").encode()
        if script in argv and str(directory).encode() in argv and b"--worker" in argv:
            conflicts.append({"pid": identity["pid"], "run_dir": str(directory)})
    return conflicts


def launch(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run_dir.resolve()
    requested_config = (
        load(run / "CONFIG.json") if args.resume else load(args.config.resolve())
    )
    conflicts = active_comparisons(CAMPAIGN / "terminal-campaigns", requested_config)
    if conflicts:
        raise CampaignBlocked(
            "A matching native comparison is already active; do not submit duplicate actors: "
            + json.dumps(conflicts, sort_keys=True)
        )
    if args.resume:
        if not (run / "CONFIG.json").is_file():
            raise ValueError("Resume requires an existing frozen campaign")
        config = load(run / "CONFIG.json")
        validate_config(config)
        verify_hashes(run / "source", load(run / "SOURCE_SHA256.json"))
        with (run / ".controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    else:
        config = load(args.config.resolve())
        prepare(config, run)
    command = [
        config["terminal_python"], str(run / "source/scripts/run_deepseek_terminal_campaign.py"),
        "--run-dir", str(run), "--worker",
    ]
    if args.resume:
        command.append("--resume")
    receipt_path = run / f"LAUNCH-{time.time_ns()}.json"
    log = run / f"controller-{time.time_ns()}.log"
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command, cwd=run / "source",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1",
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(run / "source")},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    receipt = {
        "at": utc_now(), "pid": process.pid, "command": command, "run_dir": str(run),
        "log": str(log), "start_new_session": True, "stdin": "DEVNULL", "resume": args.resume,
    }
    persist(receipt_path, receipt, immutable=True)
    persist(run / "PID.json", receipt)
    time.sleep(2)
    if process.poll() is not None:
        raise CampaignBlocked(f"Controller exited during launch; preserved log: {log}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--serve-dependencies", type=Path, metavar="MANIFEST")
    parser.add_argument("--dependency-port", type=int, default=28363)
    parser.add_argument("--verify-dependencies", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if socket.gethostname() != AUTHORIZED_HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Only the authorized A800 host with CUDA_VISIBLE_DEVICES='' may execute")
    if args.serve_dependencies:
        serve_dependencies(args.serve_dependencies.resolve(), args.dependency_port)
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required")
    if args.verify_dependencies:
        if args.config is None:
            parser.error("Dependency validation requires --config")
        receipt = verify_dependencies(args.config.resolve(), args.run_dir.resolve())
        print(json.dumps(receipt, sort_keys=True))
        return 0 if receipt["status"] == "passed" else 2
    if args.detach and not args.worker:
        if not args.resume and args.config is None:
            parser.error("A new campaign needs --config")
        print(json.dumps(launch(args), sort_keys=True))
        return 0
    if not args.worker:
        parser.error("Use --detach to persist the whole lifecycle")
    run = args.run_dir.resolve()
    if ROOT != run / "source":
        parser.error("Workers must execute the campaign's frozen source")
    with (
        (CAMPAIGN / "terminal-campaigns/.campaign.lock").open("a") as pool_lock,
        (run / ".controller.lock").open("a") as lock,
    ):
        fcntl.flock(pool_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for previous_lock in (CAMPAIGN / "terminal-campaigns").glob("*/.controller.lock"):
            if previous_lock.parent == run:
                continue
            with previous_lock.open("a") as previous:
                fcntl.flock(previous, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(previous, fcntl.LOCK_UN)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        persist(CAMPAIGN / "terminal-campaigns/OWNER.json", {
            "pid": os.getpid(), "run_dir": str(run), "started_at": utc_now(),
            "proc_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
            "source_sha256": file_hash(Path(__file__)),
            "CUDA_VISIBLE_DEVICES": "", "cumulative_token_limit": None,
        })
        campaign = Campaign(run)
        try:
            return campaign.lifecycle()
        except BaseException as exc:
            campaign.phase = "blocked"
            persist(run / "BLOCKER.json", {
                "at": utc_now(), "type": type(exc).__name__, "message": str(exc),
                "ambiguous_actions_replayed": False, "cumulative_token_limit": None,
            })
            persist(run / "HEARTBEAT.json", {
                "at": utc_now(), "pid": os.getpid(), "phase": "blocked", "active": [],
                "terminal": True, "blocker": str(run / "BLOCKER.json"),
                "cumulative_token_limit": None,
            })
            campaign.snapshot()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
