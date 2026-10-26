#!/usr/bin/env python3
"""Paired full-split GAIA2 campaign around the verified native Codex/MCP adapter."""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.harness.codex_chisel import (  # noqa: E402
    ChiselConfig, CodexChiselController, execution_result_from_receipt,
)
from phiagent.harness.benchmark_partition import canonical_sha256  # noqa: E402
from phiagent.harness.codex_gaia2 import SOURCE_FILES  # noqa: E402
from phiagent.harness.codex_runtime import CODEX_SHA256  # noqa: E402
from scripts.run_codex_chisel_terminal import TerminalComparison  # noqa: E402
from scripts.serve_codex_deepseek import read_credential, usage_summary  # noqa: E402


def public_route_prompt(scenario: dict) -> str:
    users = [event for event in scenario["events"] if event.get("event_type") == "USER"]
    if not users:
        raise ValueError("Native scenario has no public USER event")
    action = users[0]["action"]
    public = {name: action.get(name) for name in ("app", "function", "args")}
    return json.dumps(public, ensure_ascii=False, sort_keys=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_gaia_result(config: dict, request: dict) -> Path | None:
    record = config.get('retained_gaia_trial_results', {}).get(request['request_id'])
    if record is None:
        return None
    path = Path(record['path'])
    if request['phase'] != 'train' or digest(path) != record['sha256']:
        raise ValueError('Retained native scorer recovery changed or entered held-out evaluation')
    result = json.loads(path.read_text())
    recovery = result.get('native_scorer_recovery') or {}
    original = Path(recovery.get('original_trial', '')) / 'RESULT.json'
    if (recovery.get('new_actor_calls') != 0
            or recovery.get('judge_prompt_parser_model_and_output_limit_unchanged') is not True
            or digest(original) != recovery.get('paid_result_sha256')
            or result.get('scenario_id') != request['task']['task_id']
            or result.get('split') != 'train' or result.get('status') != 'COMPLETED'):
        raise ValueError('Retained scorer recovery lacks the identical paid TRAIN actor')
    return path


class GAIAComparison(TerminalComparison):
    def __init__(self, config: dict, work: Path, partition: Path, freeze: Path, pilot: Path):
        self.config = {**config, "benchmark_tag": "gaia"}
        self.root, self.work = Path(config["root"]), work
        work.mkdir(parents=True, exist_ok=True)
        self.partition_path = partition
        self.partition = json.loads(partition.read_text())
        self.base_freeze = json.loads(freeze.read_text())
        self.pilot = pilot
        self.provider_url = config["provider"]["base_url"]
        self.ledger_root = Path(config["provider"]["ledger"])
        self.relay_secret_file = Path(config["provider"]["relay_secret_file"])
        self.relay_secret = read_credential(self.relay_secret_file)
        self.request_dir = work / "requests"
        self.request_dir.mkdir(exist_ok=True)
        self.callback_results = work / "callback-results"
        self.callback_results.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.progress = {
            "status": "WAITING_FOR_VERIFIED_NATIVE_PILOT", "pid": os.getpid(),
            "callbacks_completed": 0, "maximum_tokens": None, "token_usage_mode": "unlimited",
            "test_opened": False,
        }
        self.freeze = deepcopy(self.base_freeze)
        self.freeze["source_files"] = {name: digest(ROOT / name) for name in SOURCE_FILES}

    def replay_callback(self, request: dict):
        continuation = self.config.get("training_continuation")
        if continuation is not None:
            if request["phase"] != "train":
                return None
            ident = request["request_id"]
            expected = continuation["callback_sha256"].get(ident)
            if expected is None:
                return None
            path = Path(continuation["root"]) / "callback-results" / (ident + ".json")
            if digest(path) != expected:
                raise ValueError("Paid training continuation receipt changed")
            result = json.loads(path.read_text())
            if result["request_id"] != ident or result["identity"] != request["identity"]:
                raise ValueError("Paid training continuation identity changed")
            return self.complete_callback(request, result)
        replay = self.config.get("frozen_learning_replay")
        if not replay:
            return None
        ident = request["request_id"]
        path = Path(replay["root"]) / "callback-results" / (ident + ".json")
        if not path.exists():
            if request["phase"] == "train":
                raise RuntimeError("Frozen training replay is incomplete; no new optimizer or training calls allowed")
            return None
        if digest(path) != replay["callback_sha256"].get(ident):
            raise ValueError("Frozen callback bytes changed")
        result = json.loads(path.read_text())
        if result["request_id"] != ident:
            raise ValueError("Frozen callback identity mismatch")
        return self.complete_callback(request, result)

    def optimize(self, request: dict):
        replayed = self.replay_callback(request)
        return replayed if replayed is not None else super().optimize(request)

    def scenario_path(self, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError("Unsafe scenario identifier")
        return self.partition_path.parent / "scenarios" / (name + ".json")

    def phase_tasks(self, phase: str):
        return [
            {"task_id": name, "unit_id": name,
             "public_prompt": public_route_prompt(json.loads(self.scenario_path(name).read_text()))}
            for name in self.partition["splits"][phase]
        ]

    def task_index(self):
        index = []
        for row in self.partition["records"]:
            prompt = public_route_prompt(json.loads(self.scenario_path(row["id"]).read_text()))
            index.append({
                "task_id": row["id"], "unit_id": row["id"],
                "public_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
        write_json_atomic(self.work / "PUBLIC_HASH_INDEX.json", {
            "tasks": index, "purpose": "Trusted metadata commitment; first USER event only",
            "future_user_events_used": False, "oracle_or_outcomes_used": False,
            "test_public_text_read_to_hash": True, "model_calls": 0,
        })
        return index

    def execute(self, request: dict):
        ident = request["request_id"]
        cached = self.callback_results / (ident + ".json")
        if cached.exists():
            return json.loads(cached.read_text())
        replayed = self.replay_callback(request)
        if replayed is not None:
            return replayed
        phase, task_id = request["phase"], request["task"]["task_id"]
        baseline = request["arm"] in ("baseline", "noop")
        residual = request["agents_md_append"]
        if request["codex_prompt"] != request["task"]["public_prompt"] or baseline and residual:
            raise ValueError("GAIA2 changes only the routed task-local AGENTS.md instruction")
        first = self.partition["splits"]["train"][0]
        inherited_rows = self.config.get("inherited_gaia_training", {})
        inherited = phase == "train" and baseline and request["seed"] == 20260928 and (
            task_id == first or task_id in inherited_rows
        )
        if inherited and task_id in inherited_rows:
            inherited_path = Path(inherited_rows[task_id]["path"])
            if digest(inherited_path) != inherited_rows[task_id]["sha256"]:
                raise ValueError("Inherited native GAIA2 training record changed")
            output = inherited_path.parent
        elif inherited:
            output = self.pilot
        else:
            output = self.work / "trials" / phase / ident
        result_path = output / "RESULT.json"
        recovered = retained_gaia_result(self.config, request)
        if recovered is not None:
            result_path = recovered
        if not result_path.exists():
            freeze = deepcopy(self.freeze)
            freeze["test_authorized"] = phase == "test"
            freeze["policies"]["chisel_codex"] = {
                "residual_sha256": hashlib.sha256(residual.encode()).hexdigest(),
            }
            if phase == "test":
                selected = json.loads((self.work / "POLICY.json").read_text())
                if selected["policy_sha256"] != request["final_policy_sha256"]:
                    raise ValueError("Test callback differs from the selected frozen policy")
                freeze["controller_policy_sha256"] = selected["policy_sha256"]
                freeze["controller_source_sha256"] = selected["source_sha256"]
            freeze_path = self.request_dir / (ident + ".freeze.json")
            write_json_atomic(freeze_path, freeze)
            command = [
                sys.executable, "-u", str(ROOT / "scripts/run_codex_gaia2.py"),
                "--scenario", str(self.scenario_path(task_id)),
                "--filesystem", self.config["gaia_filesystem"], "--output", str(output),
                "--partition-manifest", str(self.partition_path),
                "--freeze-manifest", str(freeze_path),
                "--provider-url", self.provider_url,
                "--relay-secret-file", str(self.relay_secret_file),
                "--codex-bin", self.config["codex"]["binary"],
                "--trial-id", "gaia-" + phase + "-" + ident[:40] + self.config.get("execution_suffix", ""),
            ]
            if residual:
                residual_path = self.request_dir / (ident + ".residual.txt")
                residual_path.write_text(residual)
                command.extend(("--residual-file", str(residual_path)))
            with (self.request_dir / (ident + ".log")).open("x") as log:
                subprocess.run(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=False,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                )
        if not result_path.exists():
            raise RuntimeError("Native GAIA2 adapter exited without a durable result")
        native = json.loads(result_path.read_text())
        if native["status"] != "COMPLETED" or native["scenario_id"] != task_id or native["split"] != phase:
            raise RuntimeError("Native GAIA2 execution/scoring remains blocked; retain all costs")
        actor_path = Path(native["outputs"]["actor_run"]) / "RUN.json"
        actor = json.loads(actor_path.read_text())
        if actor["status"] == "STOPPED":
            actor = {**actor, "status": "COMPLETED"}
        if actor["residual_sha256"] != hashlib.sha256(residual.encode()).hexdigest():
            raise ValueError("Native actor did not apply exactly the declared residual")
        receipts = actor["provider_usage"]["receipts"]
        usage = usage_summary(Path(receipts[0]).parents[3], actor["trial_id"])
        if usage["usage_status"] != "MEASURED" or usage["known_usage"] != actor["provider_usage"]["known_usage"]:
            raise ValueError("GAIA2 actor usage does not reconcile with the native provider ledger")
        feedback = None
        if phase == "train":
            observation_path = Path(native["outputs"]["public_observations"])
            feedback = (
                f"Official native task success: {native['native_score']}.\n"
                + observation_path.read_text(errors="replace")[-6000:]
            )
        result = execution_result_from_receipt(
            request, actor, native_score=native["native_score"],
            success=native["native_success"],
            feedback=feedback,
        )
        result["provenance"].update(
            native_record=str(result_path), native_record_sha256=digest(result_path),
            native_judge_accounting=native["provider_usage"]["judge"],
            inherited_training_pilot=inherited, actor_prompt="unchanged native ARE observation delivery",
            router_input="first public USER event only",
        )
        return self.complete_callback(request, result)

    def run(self):
        with (self.work / "OWNER.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.save()
            while not (self.pilot / "RESULT.json").exists():
                time.sleep(30)
            pilot = json.loads((self.pilot / "RESULT.json").read_text())
            if pilot["status"] != "COMPLETED":
                raise RuntimeError("Native GAIA2 acceptance did not pass; full campaign remains gated")
            write_json_atomic(self.work / "CONFIG.json", self.config)
            write_json_atomic(self.work / "provenance.json", capture_provenance(ROOT, sys.argv, 20260928))
            controller = CodexChiselController(
                partition=self.partition, task_index=self.task_index(),
                execute=self.execute, optimize=self.optimize,
                execution_identity={
                    "model": "deepseek-flash", "runtime": "official_codex_cli",
                    "codex_version": "0.142.5", "wire_api": "responses",
                    "transport": "byte_preserving_metering_proxy", "benchmark": "gaia2-mini",
                },
                optimizer_identity={"model": "deepseek-flash", "runtime": "native_responses"},
                run_id=self.config.get("gaia_run_id", "20260929-codex-chisel-gaia2-v1"),
                config=ChiselConfig(
                    application="agents_md", native_metric="official_gaia2_task_success",
                    execution_seeds=(20260928,), test_execution_seeds=(20260928, 20260929, 20260930),
                    executor_workers=2,
                ),
            )
            try:
                self.progress["status"] = "FULL_TRAINING"
                self.save()
                controller.train(self.phase_tasks("train"))
                replay = self.config.get("frozen_learning_replay")
                if replay and canonical_sha256(controller.candidate_bank) != replay["candidate_bank_sha256"]:
                    raise RuntimeError("Restored training-derived candidate bank differs from the frozen original")
                write_json_atomic(self.work / "TRAINING_REPORT.json", controller.report())
                self.progress["status"] = "VALIDATION_SELECTION"
                self.save()
                selection = controller.select_validation(self.phase_tasks("validation"))
                write_json_atomic(self.work / "VALIDATION_SELECTION.json", selection)
                if any(
                    not row["complete_native_scores"] or row["measured_total_tokens"] is None
                    for row in selection["results"]
                ):
                    raise RuntimeError("Validation accounting incomplete; held-out test stays sealed")
                policy = controller.freeze(source_hashes={
                    "codex_cli": CODEX_SHA256, "executor": digest(ROOT / "phiagent/harness/codex_gaia2.py"),
                    "optimizer": digest(ROOT / "scripts/run_codex_chisel_terminal.py"),
                    "scorer": self.freeze["are_source_sha256"], "adapter": digest(Path(__file__)),
                    "partition": digest(self.partition_path),
                })
                write_json_atomic(self.work / "POLICY.json", policy.to_dict())

                def load_test():
                    self.progress.update(status="PAIRED_TEST", test_opened=True)
                    self.save()
                    return self.phase_tasks("test")

                report = controller.evaluate_test(
                    load_test, policy_sha256=policy.policy_sha256, source_sha256=policy.source_sha256
                )
                write_json_atomic(self.work / "PAIRED_TEST.json", report)
                self.progress["status"] = "COMPLETE"
            except Exception as error:
                self.progress.update(status="BLOCKED", error_type=type(error).__name__, error=str(error))
                (self.work / "failure.txt").write_text(traceback.format_exc())
                raise
            finally:
                report = controller.report()
                report["predecessor_accounting"] = self.config.get("predecessor_gaia_accounting")
                write_json_atomic(self.work / "CONTROLLER_REPORT.json", report)
                self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--filesystem", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Native GAIA2 comparison is authorized only on A800-4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.umask(0o077)
    config = json.loads(args.config.read_text())
    config["gaia_filesystem"] = str(args.filesystem)
    comparison = GAIAComparison(config, args.work, args.partition, args.freeze, args.pilot)
    try:
        comparison.run()
    except Exception as error:
        comparison.progress.update(status="BLOCKED", error_type=type(error).__name__, error=str(error))
        comparison.save()
        raise


if __name__ == "__main__":
    main()
