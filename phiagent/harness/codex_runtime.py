"""Run the pinned, unmodified Codex CLI in an isolated remote Docker workspace.

``run_codex_trial`` accepts public task text/images and at most one learned
residual. It never mounts evaluator data or the hosted provider credential.
Native provider receipts, rather than CLI estimates, are the cost authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Sequence
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic
from scripts.serve_codex_deepseek import read_credential, trial_token

CODEX_VERSION = "codex-cli 0.142.5"
CODEX_SHA256 = "ac06f492f3ded7a8e2f36dc961e3cc5276a3c4841a2695d4681d0557c5b30e41"
CODEX_REVISION = "26de83050b20f7e0ee211b9739e52ae00ce8032a"
BASE_IMAGE = "nvidia/cuda@sha256:457a4076c56025f51217bff647ca631c7880ad3dbf546b03728ba98297ebbc22"
VQA_NETWORK = "phiagent-codex-vqa-20260929"
VQA_GATEWAY = "172.29.253.1"


def codex_config(
    provider_base_url: str, *,
    mcp_servers: Mapping[str, Mapping] | None = None,
    config_overrides: Mapping[str, object] | None = None,
    context_compaction_threshold: int = 996147,
) -> str:
    if type(context_compaction_threshold) is not int or not 1024 <= context_compaction_threshold < 1048576:
        raise ValueError("Native active-context compaction threshold is outside model capacity")
    parsed = urlsplit(provider_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "172.17.0.1", VQA_GATEWAY)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Codex must use the credential-holding local metering service")
    features = {"multi_agent": False}
    top_level = {}
    for name, value in (config_overrides or {}).items():
        if name.startswith("features.") and re.fullmatch(r"features\.[a-z][a-z0-9_]*", name):
            if value is not False:
                raise ValueError("Benchmark capability overrides may only disable features")
            features[name.removeprefix("features.")] = False
        elif (name, value) in (("web_search", "disabled"), ("check_for_update_on_startup", False)):
            if name != "web_search":
                top_level[name] = value
        else:
            raise ValueError("Unsupported benchmark capability override: " + name)
    extra_top_level = "".join(
        f"{name} = {json.dumps(value)}\n" for name, value in top_level.items()
    )
    configuration = f"""model = "deepseek-flash"
model_provider = "deepseek_native"
model_context_window = 1048576
model_auto_compact_token_limit = {context_compaction_threshold}
model_supports_reasoning_summaries = false
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
{extra_top_level}

[model_providers.deepseek_native]
name = "DeepSeek native Responses via byte-preserving metering"
base_url = {json.dumps(provider_base_url)}
wire_api = "responses"
env_key = "CODEX_PROVIDER_TOKEN"
requires_openai_auth = false
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 1800000

[features]
"""
    configuration += "".join(
        f"{name} = {json.dumps(value)}\n" for name, value in features.items()
    )
    fields = {
        "url": str, "bearer_token_env_var": str, "required": bool,
        "startup_timeout_sec": (int, float), "tool_timeout_sec": (int, float),
        "supports_parallel_tool_calls": bool,
    }
    for name, settings in (mcp_servers or {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(settings, Mapping):
            raise ValueError("Invalid MCP server configuration")
        if "url" not in settings or set(settings) - set(fields):
            raise ValueError("Only run-local HTTP MCP servers are permitted")
        endpoint = urlsplit(settings["url"])
        if (
            endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "172.17.0.1")
            or endpoint.username or endpoint.password or endpoint.fragment or endpoint.query
        ):
            raise ValueError("MCP server must be a local authenticated endpoint")
        configuration += f"\n[mcp_servers.{name}]\n"
        for key, value in settings.items():
            if type(value) not in (
                fields[key] if isinstance(fields[key], tuple) else (fields[key],)
            ):
                raise ValueError("Invalid MCP configuration value: " + key)
            if key == "bearer_token_env_var" and value != "CODEX_PROVIDER_TOKEN":
                raise ValueError("MCP may use only the trial-scoped relay credential")
            if key.endswith("_timeout_sec") and value <= 0:
                raise ValueError("MCP timeout must be positive")
            configuration += f"{key} = {json.dumps(value)}\n"
    return configuration


def validate_private_network(metadata: dict, provider_url: str) -> None:
    if (
        metadata.get("Name") != VQA_NETWORK or metadata.get("Internal") is not True
        or metadata.get("Driver") != "bridge"
        or metadata.get("Labels", {}).get("phiagent.campaign") != "codex-chisel-20260929"
        or not any(
            row.get("Gateway") == VQA_GATEWAY and row.get("Subnet") == "172.29.253.0/24"
            for row in metadata.get("IPAM", {}).get("Config", [])
        )
        or urlsplit(provider_url).hostname != VQA_GATEWAY
    ):
        raise ValueError("VQA requires its verified internal network and host-only metering gateway")


def reap_stopped_container_client(process, name: str) -> dict:
    try:
        process.wait(timeout=30)
        return {"escalation": "none"}
    except subprocess.TimeoutExpired:
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State.Running}}", name],
            capture_output=True, text=True, timeout=30, check=False,
        )
        stopped = (
            inspected.returncode == 0 and inspected.stdout.strip() == "false"
        ) or (
            inspected.returncode != 0 and "no such object" in inspected.stderr.lower()
            and name in inspected.stderr
        )
        if not stopped:
            raise RuntimeError(
                "Owned Docker client stalled but container shutdown is unverified: " + name
            )
        process.terminate()
        escalation = "terminate_after_container_stopped"
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            escalation = "kill_client_after_container_stopped"
        return {
            "escalation": escalation, "container_confirmed_stopped": True,
            "inspect_returncode": inspected.returncode,
        }


def run_container_process(
    command: list[str], *, name: str, prompt: str, environment: dict[str, str],
    stdout, stderr, timeout_seconds: float, stop_requested: Callable[[], str | None] | None,
    termination_receipt: Path | None = None,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    stop_reason = None
    timed_out = False
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
        text=True, env=environment,
    ) as process:
        try:
            process.stdin.write(prompt)
            process.stdin.close()
            while process.poll() is None:
                stop_reason = stop_requested() if stop_requested else None
                timed_out = time.monotonic() >= deadline
                if stop_reason is not None or timed_out:
                    break
                time.sleep(0.2)
            if stop_reason is None and stop_requested:
                stop_reason = stop_requested()
            termination = None
            if process.poll() is None:
                if termination_receipt is not None:
                    write_json_atomic(termination_receipt, {
                        "status": "STOP_REQUESTED", "container_name": name,
                        "docker_client_pid": process.pid, "timed_out": timed_out,
                        "stop_reason": stop_reason, "timeout_seconds": timeout_seconds,
                        "at_utc": datetime.now(timezone.utc).isoformat(),
                    })
                stopped = subprocess.run(
                    ["docker", "stop", "--time", "5", name],
                    capture_output=True, text=True, timeout=120, check=False,
                )
                termination = {
                    "returncode": stopped.returncode, "stdout": stopped.stdout,
                    "stderr": stopped.stderr,
                }
                termination["client_reaping"] = reap_stopped_container_client(process, name)
                if termination_receipt is not None:
                    write_json_atomic(termination_receipt, {
                        "status": "STOP_COMPLETED", "container_name": name,
                        "docker_client_pid": process.pid, "timed_out": timed_out,
                        "stop_reason": stop_reason, "termination": termination,
                        "at_utc": datetime.now(timezone.utc).isoformat(),
                    })
            return {
                "returncode": process.returncode, "stop_reason": stop_reason,
                "timed_out": timed_out, "container_termination": termination,
            }
        except BaseException:
            if process.poll() is None:
                subprocess.run(
                    ["docker", "stop", "--time", "5", name],
                    capture_output=True, timeout=120, check=False,
                )
                reap_stopped_container_client(process, name)
            raise


def cli_usage(events: Sequence[dict]) -> dict | None:
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if not completed:
        return None
    totals: dict[str, int] = {}
    for event in completed:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("Completed Codex turn is missing usage")
        for name, value in usage.items():
            if type(value) is not int or value < 0:
                raise ValueError("Codex CLI usage must contain nonnegative integers")
            totals[name] = totals.get(name, 0) + value
    return totals


def provider_usage(provider_url: str, trial_id: str, token: str) -> dict:
    opener = build_opener(ProxyHandler({}))
    request = Request(
        f"{provider_url.rstrip('/')}/{trial_id}/usage",
        headers={"Authorization": "Bearer " + token},
    )
    deadline = time.monotonic() + 1810
    while True:
        with opener.open(request, timeout=30) as response:
            usage = json.load(response)
        if usage["usage_status"] == "MEASURED":
            return usage
        unresolved = set(usage.get("unresolved_call_ids", []))
        receipt_paths = [Path(path) for path in usage.get("receipts", [])]
        pending = any(
            path.parent.name in unresolved and not path.exists() for path in receipt_paths
        )
        if not pending or time.monotonic() >= deadline:
            return usage
        time.sleep(0.5)


def run_codex_trial(
    *,
    output: Path,
    prompt: str,
    trial_id: str,
    provider_url: str,
    relay_secret_file: Path,
    codex_binary: Path,
    images: Sequence[Path] = (),
    residual: str = "",
    timeout_seconds: float = 1200,
    container_image: str = BASE_IMAGE,
    mcp_servers: Mapping[str, Mapping] | None = None,
    config_overrides: Mapping[str, object] | None = None,
    stop_requested: Callable[[], str | None] | None = None,
    docker_network: str = "host",
    context_compaction_threshold: int = 996147,
) -> dict:
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Codex experiment execution is restricted to authorized A800-4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for hosted API execution")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty public task prompt is required")
    if not isinstance(residual, str) or timeout_seconds <= 0:
        raise ValueError("Invalid residual or execution timeout")
    if not re.fullmatch(r"(?:[a-z0-9./_-]+@)?sha256:[0-9a-f]{64}", container_image):
        raise ValueError("Container image must be pinned by digest")
    binary_hash = hashlib.sha256(codex_binary.read_bytes()).hexdigest()
    if binary_hash != CODEX_SHA256:
        raise ValueError("Codex binary differs from the pinned official release")
    version = subprocess.check_output(
        [str(codex_binary), "--version"], text=True, timeout=30
    ).strip()
    if version != CODEX_VERSION:
        raise ValueError("Codex version differs from the pinned official release")
    token = trial_token(read_credential(relay_secret_file), trial_id)
    network_metadata = None
    if docker_network != "host":
        if docker_network != VQA_NETWORK:
            raise ValueError("Unregistered isolated benchmark network")
        networks = json.loads(subprocess.check_output(
            ["docker", "network", "inspect", docker_network], text=True, timeout=30
        ))
        if len(networks) != 1:
            raise ValueError("The isolated network has no unique Docker identity")
        network_metadata = networks[0]
        validate_private_network(network_metadata, provider_url)
    config = codex_config(
        f"{provider_url.rstrip('/')}/{trial_id}",
        mcp_servers=mcp_servers, config_overrides=config_overrides,
        context_compaction_threshold=context_compaction_threshold,
    )
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    workspace = output / "workspace"
    home = output / "codex-home"
    workspace.mkdir()
    home.mkdir()
    scratch = home / "runtime-tmp"
    scratch.mkdir()
    (home / "config.toml").write_text(config)
    (output / "prompt.txt").write_text(prompt)
    if residual:
        (workspace / "AGENTS.md").write_text(residual.rstrip() + "\n")
    image_paths = []
    for index, image in enumerate(images):
        extension = image.suffix.lower()
        if extension not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            raise ValueError("Unsupported image extension")
        target = workspace / f"input-{index}{extension}"
        shutil.copyfile(image, target)
        image_paths.append("/workspace/" + target.name)
    name = "codex-chisel-" + uuid.uuid4().hex[:20]
    command = [
        "docker", "run", "--rm", "--name", name,
        "--network", docker_network, "--cpus", "1", "--memory", "2g", "--pids-limit", "256",
        "--read-only",
        "--mount", f"type=bind,src={workspace},dst=/workspace",
        "--mount", f"type=bind,src={home},dst=/codex-home",
        "--mount", f"type=bind,src={codex_binary.resolve()},dst=/usr/local/bin/codex,readonly",
        "-e", "CODEX_HOME=/codex-home", "-e", "CODEX_PROVIDER_TOKEN",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NO_PROXY=127.0.0.1,localhost",
        "-e", "NVIDIA_VISIBLE_DEVICES=void",
        "-e", "TMPDIR=/codex-home/runtime-tmp",
        "-e", "TMP=/codex-home/runtime-tmp", "-e", "TEMP=/codex-home/runtime-tmp",
        "-w", "/workspace", "--entrypoint", "/usr/local/bin/codex", "-i",
        container_image, "exec", "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check", "--json", "-C", "/workspace",
        "--output-last-message", "/workspace/last-message.txt",
    ]
    for image_path in image_paths:
        command.extend(("--image", image_path))
    command.append("-")
    record = {
        "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "trial_id": trial_id, "hostname": socket.gethostname(), "command": command,
        "codex_version": version, "codex_revision": CODEX_REVISION,
        "codex_binary_sha256": binary_hash, "container_image": container_image,
        "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "residual_sha256": hashlib.sha256(residual.encode()).hexdigest(),
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": "deepseek-flash", "maximum_tokens": None, "token_usage_mode": "unlimited",
        "workspace": str(workspace), "run_directory": str(output),
        "scratch_directory": str(scratch),
        "container_scratch_directory": "/codex-home/runtime-tmp",
        "events_path": str(output / "events.jsonl"), "CUDA_VISIBLE_DEVICES": "",
        "selected_gpu": None, "reason": "Hosted model API; CPU-only isolated task executor",
        "seed": 20260928, "provider_sampling_seed_supported": False,
        "mcp_servers": dict(mcp_servers or {}), "config_overrides": dict(config_overrides or {}),
        "native_stop_callback": stop_requested is not None,
        "docker_network": docker_network, "network_admission": network_metadata,
        "active_context_compaction_threshold": context_compaction_threshold,
        "compaction_threshold_is_not_a_cumulative_token_budget": True,
        "images": [
            {"path": path, "sha256": hashlib.sha256(
                (workspace / Path(path).name).read_bytes()
            ).hexdigest()} for path in image_paths
        ],
    }
    write_json_atomic(output / "RUN.json", record)
    write_json_atomic(output / "provenance.json", capture_provenance(
        Path(__file__).resolve().parents[2], command, 20260928
    ))
    write_json_atomic(output / "resources.json", {
        "gpus": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv",
        ], text=True, timeout=30),
        "model_gpu_selected": None,
        "container_mounts": [str(workspace), str(home), str(codex_binary)],
    })
    try:
        with (output / "events.jsonl").open("w") as stdout, (
            output / "stderr.log"
        ).open("w") as stderr:
            execution = run_container_process(
                command, name=name, prompt=prompt, stdout=stdout, stderr=stderr,
                environment={**os.environ, "CODEX_PROVIDER_TOKEN": token, "CUDA_VISIBLE_DEVICES": ""},
                timeout_seconds=timeout_seconds, stop_requested=stop_requested,
                termination_receipt=output / "TERMINATION.json",
            )
        record.update(execution)
        events = [
            json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        record["native_cli_usage"] = cli_usage(events)
        answer = workspace / "last-message.txt"
        record["final_answer"] = answer.read_text() if answer.exists() else None
        record["provider_usage"] = provider_usage(provider_url, trial_id, token)
        failed = any(event.get("type") in ("turn.failed", "error") for event in events)
        record["status"] = "COMPLETED" if (
            execution["returncode"] == 0 and not failed and record["final_answer"] is not None
            and record["native_cli_usage"] is not None
            and record["provider_usage"]["usage_status"] == "MEASURED"
        ) else "BLOCKED"
        if execution["stop_reason"] is not None:
            record["status"] = "STOPPED"
        elif execution["timed_out"]:
            record.update(status="TIMEOUT", error="Codex execution reached its protocol timeout")
        if record["status"] == "BLOCKED":
            record["error"] = "Codex execution or complete native provider accounting failed"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record.update(
            status="BLOCKED", error_type=type(error).__name__,
            error=str(error).replace(token, "[REDACTED]"),
        )
        try:
            record["provider_usage"] = provider_usage(provider_url, trial_id, token)
        except (OSError, ValueError) as usage_error:
            record["accounting_error"] = {
                "type": type(usage_error).__name__,
                "message": str(usage_error).replace(token, "[REDACTED]"),
            }
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(output / "RUN.json", record)
    return record
