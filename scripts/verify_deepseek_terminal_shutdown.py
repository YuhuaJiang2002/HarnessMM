#!/usr/bin/env python3
"""A800-only real-SDK cancellation check, with no benchmark data or scoring."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_deepseek_terminal_campaign as campaign  # noqa: E402
from scripts.run_deepseek_terminal import resource_inventory  # noqa: E402


def jsonl_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def verify(config_path: Path, run: Path) -> dict:
    config = campaign.load(config_path)
    campaign.validate_config(config)
    if socket.gethostname() != campaign.AUTHORIZED_HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Real SDK acceptance requires a800-4 and CUDA_VISIBLE_DEVICES=''")
    if not run.is_relative_to(campaign.CAMPAIGN / "runs") or run == campaign.CAMPAIGN / "runs":
        raise ValueError("Use a fresh synthetic acceptance directory under campaign/runs")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    private = run / "private"
    private.mkdir(mode=0o700)
    logs = run / "agent-logs"
    logs.mkdir(mode=0o700)
    runtime = campaign.load(Path(config["runtime_template"]))["runtime"]
    campaign.persist(run / "CONFIG.json", config, immutable=True)
    campaign.persist(run / "RESOURCE_INVENTORY.json", resource_inventory(), immutable=True)
    campaign.persist(run / "PACKAGES.json", {
        "python": sys.version, "executable": sys.executable, "hostname": socket.gethostname(),
        "packages": sorted((item.metadata["Name"], item.version)
                           for item in importlib.metadata.distributions()),
    }, immutable=True)
    campaign.persist(run / "COMMANDS.json", {
        "argv": sys.argv, "cwd": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
        "provider": "official native DeepSeek Messages", "model": config["model"],
        "official_revision": config["official_revision"], "seed": config["split_seed"],
        "benchmark_data_loaded": False, "benchmark_scorer_invoked": False,
        "native_benchmark_deadlines_changed": False, "cumulative_token_limit": None,
        "stop_condition": "Cancel once the SDK's first synthetic terminal call is awaiting completion",
    }, immutable=True)
    campaign.persist(run / "SOURCE.json", {
        "verification_script_sha256": campaign.file_hash(Path(__file__)),
        "controller_sha256": campaign.file_hash(ROOT / "scripts/run_deepseek_terminal_campaign.py"),
        "sdk_runner_sha256": campaign.file_hash(
            ROOT / "integrations/deepseek_harness/runner.py"
        ),
        "runtime_manifest_sha256": campaign.file_hash(Path(runtime["manifest"])),
        "upload_git_state": campaign.load(ROOT / "CAMPAIGN_UPLOAD_GIT_STATE.json"),
        "override": "Only worker-job purpose is synthetic instead of benchmark",
    }, immutable=True)
    entered = asyncio.Event()

    class SyntheticTerminal:
        cancelled = False

        async def exec(self, **arguments):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    original_builder = campaign.build_worker_job

    def synthetic_job(**kwargs):
        return {**original_builder(**kwargs), "purpose": "synthetic"}

    campaign.build_worker_job = synthetic_job
    agent = campaign.NativeCompletionAgent(
        logs_dir=logs, runner_python=config["sdk_python"],
        runner_script=str(ROOT / "integrations/deepseek_harness/runner.py"),
        runtime=runtime, api_key_file=config["api_key_file"], private_root=str(private),
        max_tokens=config["max_output_tokens"], seed=config["split_seed"],
        source_revision=config["official_revision"],
    )
    agent._private_dir = private
    environment = SyntheticTerminal()
    context = SimpleNamespace(
        metadata={}, n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None, cost_usd=None,
    )
    prompt = (
        "This is a synthetic SDK infrastructure check, not a benchmark task. "
        "Call terminal_exec once with command 'printf SYNTHETIC_CANCELLATION_ACCEPTANCE'. "
        "The controlled test backend will wait without executing a shell command. "
        "Do not answer before calling the tool."
    )
    actor = asyncio.create_task(agent.run(prompt, environment, context))
    gate = asyncio.create_task(entered.wait())
    failure = None
    cancelled = False
    try:
        completed, _ = await asyncio.wait({actor, gate}, timeout=300,
                                         return_when=asyncio.FIRST_COMPLETED)
        if gate not in completed:
            failure = "The real SDK did not enter the controlled tool within the operability window"
        actor.cancel()
        try:
            await actor
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    finally:
        gate.cancel()
        await asyncio.gather(gate, return_exceptions=True)
        campaign.build_worker_job = original_builder
    accounting = campaign.recover_runner_accounting(private / "sdk-run")
    campaign.persist(run / "RECOVERED_USAGE.json", accounting, immutable=True)
    audit_path = private / "native-backend/terminal-exec.jsonl"
    audit = jsonl_records(audit_path)
    native_cancel = next(
        (row["timestamp"] for row in audit if row.get("event") == "exec_cancelled"), None,
    )
    starts = [
        row for row in jsonl_records(private / "sdk-run/llm-streams.jsonl")
        if row.get("kind") == "llm/start"
    ]
    late = [
        row["attempt_id"] for row in starts
        if native_cancel and datetime.fromisoformat(row["at"].replace("Z", "+00:00"))
        >= datetime.fromisoformat(native_cancel)
    ]
    shutdown_path = private / "SDK_SHUTDOWN.json"
    shutdown = campaign.load(shutdown_path) if shutdown_path.is_file() else {}
    complete = accounting.get("complete") is True
    worker_pid = shutdown.get("worker_pid")
    sdk_group_stopped = isinstance(worker_pid, int) and not campaign.live_sdk_group(worker_pid)
    if failure is None and not audit_path.is_file():
        failure = "The real SDK stopped before creating the controlled native tool audit"
    passed = (
        failure is None and cancelled and environment.cancelled and native_cancel is not None
        and bool(starts) and not late and complete
        and shutdown.get("cancellation_responses_released_after_sdk_stop") is True
        and sdk_group_stopped
    )
    receipt = {
        "status": "passed" if passed else "blocked", "checked_at": campaign.utc_now(),
        "run_dir": str(run), "worker_dir": str(private / "sdk-run"),
        "evidence_role": "SDK adapter cancellation operability; not a public benchmark outcome",
        "controller_sha256": campaign.file_hash(ROOT / "scripts/run_deepseek_terminal_campaign.py"),
        "official_revision": config["official_revision"], "model": config["model"],
        "failure": failure, "native_cancelled_at": native_cancel,
        "provider_attempts": len(starts), "provider_attempts_started_after_cancellation": late,
        "usage_complete": complete,
        "usage": campaign.canonical_usage(
            campaign.normalize_runner_usage(accounting.get("usage")), complete,
        ),
        "usage_receipt": str(run / "RECOVERED_USAGE.json"),
        "usage_receipt_sha256": campaign.file_hash(run / "RECOVERED_USAGE.json"),
        "native_commands_executed": 0, "benchmark_data_loaded": False,
        "benchmark_scorer_invoked": False, "selected_gpus": [], "cumulative_token_limit": None,
        "native_benchmark_deadlines_changed": False, "sdk_group_stopped": sdk_group_stopped,
    }
    campaign.persist(run / "ACCEPTANCE.json", receipt, immutable=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(verify(args.config.resolve(), args.run_dir.resolve()))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
