#!/usr/bin/env python3
"""Launch a new shared-parent GAIA2 pilot and its complete frozen successor."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.harness.gaia2_common_prompt import prompt_identity  # noqa: E402
from scripts import run_deepseek_gaia2 as native  # noqa: E402
from scripts import run_deepseek_gaia2_campaign as campaign  # noqa: E402


def successor_config(
    original: dict, account: dict, source_revision: str,
    observation_encoding: str | None = None,
    execution_alignment: bool = False,
) -> dict:
    campaign.validate_config(original)
    if account.get("http_status") != 200 or account.get("provider_account_available") is not True:
        raise native.GateError("Verify the actual DeepSeek account before launching this successor")
    value = {
        **original,
        "source_revision": source_revision,
        "shared_gaia2_prompt": prompt_identity(),
        "minimum_parent_train_successes": 8,
        "parent_protocol": {
            "name": "new common native-app brief; original and CHISEL receive identical bytes",
            "design_evidence": "complete prior TRAIN32 public actor traces only",
            "prior_complete_train_successes": 1,
            "historical_actor_results_reused": False,
            "test_feedback_used_for_revision": False,
            "capability_before_rsi": "at least 8/32 native TRAIN successes",
        },
        "account_availability": account,
    }
    if observation_encoding is not None:
        value["native_observation_encoding"] = observation_encoding
        value["parent_protocol"]["observation_transport_revision"] = (
            "Original native tool text, without an additional JSON-string wrapper"
        )
    if execution_alignment:
        from phiagent.harness.gaia2_execution_contract import aligned_contract

        contract = aligned_contract()
        value.update(native_execution_contract=contract,
                     native_max_turns=contract["unspecified_user_turn_limit"],
                     scenario_timeout_seconds=contract["outer_wall_timeout_seconds"])
        value["parent_protocol"]["execution_repair"] = contract
    if original.get("provider_durable_capture") is True:
        value["parent_protocol"].update(
            historical_actor_results_reused=bool(original.get("retained_parent_train")),
            transport_revision="byte-exact provider capture continues to native final usage after SDK disconnect",
            retained_parent_tasks=len(original.get("retained_parent_train") or {}),
            original_unknown_development_costs_preserved=True,
        )
    campaign.validate_config(value)
    return value


def launch(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=True, timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    receipt = json.loads(completed.stdout)
    if type(receipt.get("pid")) is not int or receipt["pid"] <= 0:
        raise native.GateError("Detached child did not return its real owner PID")
    os.kill(receipt["pid"], 0)
    return receipt


def prepare(
    root: Path, config_path: Path, availability: Path, source_revision: str,
    observation_encoding: str | None = None,
    execution_alignment: bool = False,
) -> None:
    native.require_remote()
    native.owned_path(root).mkdir(parents=True, exist_ok=False, mode=0o700)
    original, account = native.load(config_path), native.load(availability)
    config = successor_config(original, account, source_revision, observation_encoding,
                              execution_alignment)
    native.persist(root / "CONFIG.json", config, immutable=True)
    hashes = native.freeze_source(root / "source")
    native.persist(root / "SOURCE_SHA256.json", hashes, immutable=True)
    native.persist(root / "RESOURCE_SELECTION.json", native.resource_inventory(), immutable=True)
    native.persist(root / "PREPARATION.json", {
        **native.provenance(ROOT, sys.argv, config["split_seed"]),
        "original_config": {"path": str(config_path), "sha256": native.file_hash(config_path)},
        "availability": {"path": str(availability), "sha256": native.file_hash(availability)},
        "source_revision": source_revision,
        "local_benchmark_or_model_calls": 0,
        "successor_task_id": "complete32-parent-then-train-rsi-val-freeze-test",
    }, immutable=True)
    pilot_command = [
        config["are_python"], "-u", str(root / "source/scripts/run_deepseek_gaia2.py"),
        "--config", str(root / "CONFIG.json"), "--first-training",
        "--run-dir", str(root / "pilot"), "--detach",
    ]
    pilot = launch(pilot_command, root / "source")
    native.persist(root / "PILOT_OWNER.json", pilot, immutable=True)
    worker_command = [
        config["are_python"], "-u",
        str(root / "source/scripts/advance_deepseek_gaia2_common_parent.py"),
        "--run-dir", str(root), "--worker",
    ]
    with (root / "advancer.log").open("xb") as stream:
        process = subprocess.Popen(
            worker_command, cwd=root / "source", stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    receipt = {
        "pid": process.pid, "command": worker_command, "started_at": native.utc_now(),
        "run_dir": str(root), "pilot_owner_pid": pilot["pid"],
        "successor_task_id": "complete32-parent-then-train-rsi-val-freeze-test",
    }
    native.persist(root / "OWNER.json", receipt, immutable=True)
    time.sleep(2)
    if process.poll() is not None:
        raise native.GateError("Advancer exited before durable monitoring; inspect advancer.log")
    print(json.dumps(receipt, sort_keys=True))


def worker(root: Path) -> int:
    native.require_remote()
    config = native.load(root / "CONFIG.json")
    campaign.validate_config(config)
    native.verify_hashes(root / "source", native.load(root / "SOURCE_SHA256.json"))
    with (root / "OWNER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase, last_capacity = "first-native-train-gate", 0.0
        while True:
            if time.monotonic() - last_capacity >= 300:
                native.persist(
                    root / "capacity" / f"{time.time_ns()}.json",
                    native.resource_inventory(), immutable=True,
                )
                last_capacity = time.monotonic()
            native.persist(root / "STATE.json", {
                "at_utc": native.utc_now(), "owner_pid": os.getpid(), "phase": phase,
                "CUDA_VISIBLE_DEVICES": "", "selected_gpus": [],
                "cumulative_token_limit": None, "shared_gaia2_prompt": prompt_identity(),
                "successor_task_id": "complete32-parent-then-train-rsi-val-freeze-test",
            })
            pilot_summary = root / "pilot/trial/SUMMARY.json"
            if phase == "first-native-train-gate" and pilot_summary.is_file():
                summary = native.load(pilot_summary)
                if summary.get("valid_for_comparison") is not True:
                    raise native.GateError(
                        "First native TRAIN gate failed; retain its costs and diagnose: "
                        + str(summary.get("blocker")),
                    )
                command = [
                    config["are_python"], "-u",
                    str(root / "source/scripts/run_deepseek_gaia2_campaign.py"),
                    "--config", str(root / "CONFIG.json"),
                    "--pilot-summary", str(pilot_summary),
                    "--run-dir", str(root / "campaign"), "--detach",
                ]
                native.persist(root / "CAMPAIGN_COMMAND.json", {"argv": command}, immutable=True)
                native.persist(
                    root / "CAMPAIGN_OWNER.json", launch(command, root / "source"), immutable=True,
                )
                phase = "complete32-parent-then-train-rsi-val-freeze-test"
            if phase != "first-native-train-gate":
                path = root / "campaign/SUMMARY.json"
                if path.is_file():
                    summary = native.load(path)
                    native.persist(root / "LATEST_CAMPAIGN.json", summary)
                    if summary.get("phase") in ("complete", "parent_complete", "blocked"):
                        native.persist(root / "TERMINAL.json", {
                            "at_utc": native.utc_now(), "phase": summary["phase"],
                            "final_test_complete": summary.get("final_test_complete") is True,
                            "source": str(path), "source_sha256": native.file_hash(path),
                            "cumulative_token_limit": None,
                        }, immutable=True)
                        return 0 if summary["phase"] in ("complete", "parent_complete") else 2
            time.sleep(config["heartbeat_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--availability", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--observation-encoding", choices=("canonical_json", "native_text"))
    parser.add_argument("--align-figure2-execution", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    native.require_remote()
    if args.worker:
        try:
            return worker(args.run_dir)
        except Exception as error:
            native.persist(args.run_dir / "FAILURE.json", {
                "at_utc": native.utc_now(), "type": type(error).__name__, "message": str(error),
                "cumulative_token_limit": None, "new_actor_replay_permitted": False,
            }, immutable=True)
            raise
    if args.config is None or args.availability is None or not args.source_revision:
        parser.error("New successors need --config, --availability and --source-revision")
    prepare(
        args.run_dir, args.config, args.availability, args.source_revision,
        args.observation_encoding, args.align_figure2_execution,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
