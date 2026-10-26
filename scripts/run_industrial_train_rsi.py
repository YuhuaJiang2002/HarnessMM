#!/usr/bin/env python3
"""Stage or execute an owner-admitted, source-bound industrial TRAIN RSI operation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.harness.codex_chisel import ChiselConfig  # noqa: E402
from phiagent.harness.industrial_common_prompt import prompt_manifest  # noqa: E402
from phiagent.harness.industrial_train_rsi import (  # noqa: E402
    IDENTITY, SCHEMA, IndustrialTrainRSIController, quality_cost_gate,
)

NEW_FILES = (
    "phiagent/harness/industrial_train_rsi.py",
    "scripts/run_industrial_train_rsi.py",
    "configs/industrial_terminal_train_rsi_v1.json",
    "tests/test_industrial_train_rsi.py",
    "scripts/industrial_deepseek_train_adapter.py",
)
CORE_FILES = (
    "phiagent/__init__.py",
    "phiagent/harness/__init__.py",
    "phiagent/harness/task_reasoning.py",
    "phiagent/harness/tshirt_fold_strategy.py",
    "phiagent/harness/tshirt_positive_reference.py",
    "phiagent/harness/benchmark_partition.py",
    "phiagent/harness/codex_chisel.py",
    "phiagent/harness/industrial_common_prompt.py",
    "phiagent/harness/industrial_efficiency_rsi.py",
    "phiagent/learning/residual_meta_harness.py",
    "phiagent/learning/residual_meta_training.py",
    "phiagent/data_engine/provenance.py",
    "tests/test_codex_chisel.py",
    "scripts/run_industrial_capability_baselines.py",
    "scripts/run_codex_terminal.py",
)
REMOTE_PREFIX = Path("/path/to/data/phiagent/work/industrial-train-rsi")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def execution_summary(rows):
    totals = [row["usage"]["total_tokens"] for row in rows]
    return {
        "task_ids": [row["task_id"] for row in rows],
        "native_successes": sum(row["success"] is True for row in rows),
        "native_scored": sum(type(row["native_score"]) in (int, float)
                             and row["native_score"] in (0, 1) for row in rows),
        "all_attempt_actor_control_tokens": (
            sum(totals) if all(type(value) is int for value in totals) else None
        ),
        "all_attempt_usage_complete": len(rows) == 18 and all(type(value) is int for value in totals),
        "residual_applications": sum(row["residual_applied"] for row in rows),
    }


def remote_only():
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Native experiments require authorized A800-4 CPU/API execution")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for this unchanged hosted backbone")


def resource_gate(run):
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemAvailable:", "MemTotal:"))
    }
    inventory = {
        "at": now(), "hostname": socket.gethostname(), "gpu_required": False,
        "CUDA_VISIBLE_DEVICES": "", "memory_bytes": memory,
        "free_output_bytes": shutil.disk_usage(run).free,
        "hosted_backbone_unchanged": True,
        "load": subprocess.check_output(["uptime"], text=True, timeout=15).strip(),
    }
    write_json_atomic(run / "RESOURCE_ADMISSION.json", inventory)
    if memory["MemAvailable"] < 8 * 1024 ** 3 or inventory["free_output_bytes"] < 8 * 1024 ** 3:
        raise RuntimeError("Keep native 8 GiB memory/storage reserves; do not bypass resource guards")


def verify_admission(admission, settings, harness):
    """Consume the native owner's admission; do not redo its infrastructure audit."""
    if (admission.get("schema") != "industrial-terminal-train-admission/v1"
            or admission.get("status") != "READY"
            or admission.get("harness") != harness
            or admission.get("phase") != "train"
            or admission.get("validation_or_test_read") is not False
            or admission.get("exclusive_successor_owner") != settings["owner"]
            or admission.get("native_owner_handoff") is not True):
        raise RuntimeError("Await native owner's READY, exclusive, TRAIN-only source-bound admission")
    parent = admission["parent"]
    expected = settings["parents"][harness]
    if any(parent.get(key) != value for key, value in expected.items()):
        raise ValueError("Owner admission differs from the recorded capable parent lineage")
    if parent["native_scored"] != 18 or parent["native_successes"] < 9:
        raise ValueError("Require the established complete capable native TRAIN parent")
    contract = admission["shared_contract"]
    prompt_manifest(
        "terminal-bench-2.0", contract["shared_prompt_version"], contract["shared_prompt_sha256"],
    )
    runtime = "official_codex_cli" if harness == "codex" else "official_deepseek_harness_sdk"
    wire = "responses" if harness == "codex" else "anthropic_messages"
    if (contract["model"] != "deepseek-flash" or contract["runtime"] != runtime
            or contract["wire_api"] != wire
            or contract.get("native_output_time_tool_contracts_unchanged") is not True
            or admission.get("all_attempt_actor_control_usage") is not True
            or admission.get("official_native_verifier") is not True
            or admission.get("train_payload_count") != 18
            or not admission.get("callback_factory")
            or not admission.get("source_sha256")):
        raise RuntimeError("Native model/interface/verifier/all-attempt callback gates are not ready")
    identity = admission["execution_identity"]
    if (identity.get("model") != contract["model"] or identity.get("runtime") != runtime
            or identity.get("wire_api") != wire
            or identity.get("shared_prompt_sha256") != contract["shared_prompt_sha256"]
            or admission["optimizer_identity"].get("model") != contract["model"]):
        raise ValueError("Actor and proposer must use the exact frozen model/interface identity")
    approval = admission.get("callback_owner_approval")
    if (not isinstance(approval, dict) or not isinstance(approval.get("path"), str)
            or not isinstance(approval.get("sha256"), str)):
        raise RuntimeError("Capability READY is not exclusive native callback-owner approval")
    path = Path(approval["path"])
    if not path.is_file() or sha(path) != approval["sha256"]:
        raise ValueError("Native callback-owner approval artifact is absent or changed")
    release = load(path)
    if (release.get("schema") != "industrial-native-callback-owner-approval/v1"
            or release.get("status") != "READY"
            or release.get("scope") != "exclusive_industrial_train_callback_factory"
            or not isinstance(release.get("issued_by"), str)
            or not release["issued_by"]
            or release["issued_by"] == settings["owner"]
            or release.get("exclusive_successor_owner") != settings["owner"]
            or release.get("harness") != harness or release.get("phase") != "train"):
        raise RuntimeError("Require separate parent/native callback-owner approval, not self-release")
    bound_fields = (
        "callback_factory", "source_root", "source_sha256", "train_input_sha256",
        "parent", "shared_contract", "execution_identity", "optimizer_identity",
    )
    if any(field not in admission or release.get(field) != admission[field]
           for field in bound_fields):
        raise ValueError("Callback-owner approval does not bind these exact sources and TRAIN inputs")
    return deepcopy(contract)


class DurableTrainCallbacks:
    def __init__(self, root, adapter, *, identity, parent_summary=None):
        self.root, self.adapter, self.identity = root, adapter, identity
        self.parent_summary = parent_summary
        self.lock = threading.Lock()
        self.active = {}
        self.completed = []
        self.monitor_error = None

    def record_result(self, kind, request, result):
        with self.lock:
            self.completed.append({
                "kind": kind, "arm": request.get("arm"),
                "task_id": result.get("task_id"), "native_score": result.get("native_score"),
                "success": result.get("success"), "usage": result["usage"],
                "residual_applied": request.get("residual") is not None,
                "execution_status": result["execution_status"],
                "request_id": request["request_id"],
            })

    def invoke(self, kind, request):
        if request["phase"] != "train":
            raise ValueError("Do not dispatch or load any protected validation/test request")
        if self.monitor_error is not None:
            raise RuntimeError(f"Persistent heartbeat failed: {self.monitor_error}")
        folder = self.root / "callbacks" / kind / request["request_id"]
        if folder.exists():
            if load(folder / "REQUEST.json") != request:
                raise ValueError("A callback identity cannot be reused with different bytes")
            if not (folder / "COMMITTED.json").is_file():
                raise RuntimeError("Unresolved native dispatch requires its infrastructure owner")
            seal = load(folder / "COMMITTED.json")
            if (sha(folder / "REQUEST.json") != seal["request_sha256"]
                    or sha(folder / "RESULT.json") != seal["result_sha256"]):
                raise ValueError("Committed native callback bytes changed")
            result = load(folder / "RESULT.json")
            self.record_result(kind, request, result)
            return result
        folder.mkdir(parents=True)
        write_json_atomic(folder / "REQUEST.json", request)
        with self.lock:
            self.active[request["request_id"]] = {
                "kind": kind, "task_id": request.get("task", {}).get("task_id"),
                "arm": request.get("arm"), "started_at": now(),
            }
        try:
            result = getattr(self.adapter, "execute" if kind == "executor" else "optimize")(request)
            write_json_atomic(folder / "RESULT.json", result)
            if (result.get("identity") != request["identity"]
                    or (kind == "executor" and request["identity"] != self.identity)):
                raise ValueError("Native callback changed its declared model/interface identity")
            usage = result.get("usage", {})
            if (type(usage.get("total_tokens")) is not int
                    or usage.get("input_tokens") is None or usage.get("output_tokens") is None
                    or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]):
                raise RuntimeError("Missing all-attempt usage is unresolved, never zero")
            write_json_atomic(folder / "COMMITTED.json", {
                "request_sha256": sha(folder / "REQUEST.json"),
                "result_sha256": sha(folder / "RESULT.json"),
            })
            self.record_result(kind, request, result)
            return result
        except Exception as error:
            write_json_atomic(folder / "ERROR.json", {
                "at": now(), "type": type(error).__name__, "message": str(error),
                "native_receipts_preserved": True, "retry_owner": "native infrastructure owner",
            })
            raise
        finally:
            with self.lock:
                self.active.pop(request["request_id"], None)

    def execute(self, request):
        return self.invoke("executor", request)

    def optimize(self, request):
        if self.parent_summary is not None:
            rows = [row for row in self.completed if row["arm"] == "baseline"]
            parent = self.parent_summary
            if (len(rows) != 18 or {row["task_id"] for row in rows} != set(parent["task_ids"])
                    or sum(row["success"] is True for row in rows) != parent["native_successes"]
                    or sum(row["usage"]["total_tokens"] for row in rows)
                    != parent["all_attempt_actor_control_tokens"]):
                raise RuntimeError("Native parent callback omitted quality or all-attempt costs; "
                                   "block before paid proposal")
        return self.invoke("optimizer", request)


def stage(run, config, *, harness):
    if run.exists():
        raise ValueError("Changed method/settings require a fresh run directory")
    settings = load(config)
    if settings["schema"] != SCHEMA or harness not in settings["parents"]:
        raise ValueError("Use the frozen industrial TRAIN settings")
    run.mkdir(parents=True)
    for name in (*CORE_FILES, *NEW_FILES):
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json_atomic(run / "SOURCE_SHA256.json", {
        name: sha(run / "source" / name) for name in (*CORE_FILES, *NEW_FILES)
    })
    write_json_atomic(run / "STAGING_PROVENANCE.json", capture_provenance(ROOT, sys.argv, 20260928))
    write_json_atomic(run / "RSI_CONFIG.json", {**settings, "harness": harness, "run_id": run.name})
    write_json_atomic(run / "SETTINGS_SHA256.json", {
        "RSI_CONFIG.json": sha(run / "RSI_CONFIG.json"),
    })
    write_json_atomic(run / "STATE.json", {
        "schema": SCHEMA, "status": "STAGED_AWAITING_NATIVE_OWNER_ADMISSION",
        "at": now(), "run_root": str(run), "pid": None,
        "component_identity": IDENTITY, "new_native_requests": 0,
        "candidate_native_scored": 0, "candidate_native_successes": 0,
        "candidate_all_attempt_tokens": None,
        "cumulative_token_limit": None, "validation_read": False, "test_read": False,
        "dependency": settings["admission_dependency"],
        "command": [
            "env", "CUDA_VISIBLE_DEVICES=", settings["python"][harness],
            str(run / "source/scripts/run_industrial_train_rsi.py"),
            "run", "--run-dir", str(run),
        ],
    })


def run_training(run):
    remote_only()
    if not run.is_absolute() or not run.is_relative_to(REMOTE_PREFIX) or ROOT != run / "source":
        raise ValueError("Execute only a fresh remote run's own frozen source snapshot")
    settings = load(run / "RSI_CONFIG.json")
    if sha(run / "RSI_CONFIG.json") != load(run / "SETTINGS_SHA256.json")["RSI_CONFIG.json"]:
        raise ValueError("The frozen RSI settings changed")
    admission_path = run / "ADMISSION.json"
    if not admission_path.is_file():
        raise RuntimeError("Missing ADMISSION.json: native owner must release its sealed callbacks")
    admission = load(admission_path)
    contract = verify_admission(admission, settings, settings["harness"])
    sources = load(run / "SOURCE_SHA256.json")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("The staged coding RSI source changed")
    native_source = Path(admission["source_root"])
    for name, digest in admission["source_sha256"].items():
        if sha(native_source / name) != digest:
            raise ValueError("Owner-admitted native callback source changed")
    if sha(run / "TRAIN_INPUT.json") != admission["train_input_sha256"]:
        raise ValueError("The complete TRAIN-only input payload changed")
    data = load(run / "TRAIN_INPUT.json")
    if (data.get("phase") != "train" or len(data["tasks"]) != 18
            or {row["task_id"] for row in data["tasks"]} != set(data["partition"]["splits"]["train"])
            or {key: len(value) for key, value in data["partition"]["splits"].items()}
            != {"train": 18, "validation": 18, "test": 53}):
        raise ValueError("Require all 18 TRAIN identities and only public TRAIN payloads")
    parent_summary = data["parent_summary"]
    parent = admission["parent"]
    if (set(parent_summary["task_ids"]) != set(data["partition"]["splits"]["train"])
            or parent_summary["native_scored"] != parent["native_scored"]
            or parent_summary["native_successes"] != parent["native_successes"]
            or parent_summary["all_attempt_actor_control_tokens"] != parent["all_attempt_tokens"]
            or parent_summary["all_attempt_usage_complete"] is not True):
        raise ValueError("Parent comparator must retain all paid attempts, not just selected18")
    with (run / "OWNER.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource_gate(run)
        sys.path.insert(0, str(native_source))
        module, factory = admission["callback_factory"].split(":")
        adapter = getattr(importlib.import_module(module), factory)(
            run=run, admission=admission, train_input=data,
        )
        callbacks = DurableTrainCallbacks(
            run, adapter, identity=admission["execution_identity"], parent_summary=parent_summary,
        )
        controller = IndustrialTrainRSIController(
            partition=data["partition"], task_index=data["task_index"],
            execute=callbacks.execute, optimize=callbacks.optimize,
            execution_identity=admission["execution_identity"],
            optimizer_identity=admission["optimizer_identity"], run_id=run.name,
            config=ChiselConfig(
                optimization_iterations=1, executor_workers=1,
                execution_seeds=(20260928,), native_metric="official_terminal_bench_binary_reward",
                application="prompt_suffix" if settings["harness"] == "codex" else "system_section",
            ),
        )
        stop = threading.Event()
        state = {"schema": SCHEMA, "status": "TRAIN", "pid": os.getpid(), "run_root": str(run)}

        def save():
            with callbacks.lock:
                write_json_atomic(run / "STATE.json", {
                    **state, "at": now(), "active": deepcopy(callbacks.active),
                    "completed": deepcopy(callbacks.completed), "component_identity": IDENTITY,
                    "cumulative_token_limit": None, "validation_read": False, "test_read": False,
                })

        def heartbeat():
            while not stop.wait(settings["heartbeat_seconds"]):
                try:
                    save()
                except OSError as error:
                    callbacks.monitor_error = str(error)
                    write_json_atomic(run / "HEARTBEAT_ERROR.json", {
                        "at": now(), "type": type(error).__name__, "message": str(error),
                    })
                    stop.set()

        write_json_atomic(run / "PROVENANCE.json", capture_provenance(ROOT, sys.argv, 20260928))
        write_json_atomic(run / "ADMISSION_CONSUMED.json", {
            "admission_sha256": sha(admission_path), "native_sources": admission["source_sha256"],
            "shared_contract": contract, "all_attempt_costs_retained": True,
        })
        save()
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        try:
            report = controller.train(data["tasks"])
            for name, digest in sources.items():
                if sha(ROOT / name) != digest:
                    raise ValueError("Frozen source changed during TRAIN")
            write_json_atomic(run / "TRAIN_COMPLETE.json", report)
            frozen = controller.freeze_train_bank(sources=sources, shared_contract=contract)
            write_json_atomic(run / "TRAIN_FREEZE.json", frozen)
            with callbacks.lock:
                rows = deepcopy(callbacks.completed)
            comparisons = {}
            for candidate in controller.candidate_bank:
                if candidate["id"] == "noop":
                    continue
                observed = [row for row in rows if row["arm"] == candidate["id"]]
                summary = execution_summary(observed)
                comparisons[candidate["id"]] = {
                    "raw_train_intervention": summary,
                    "gate": quality_cost_gate(parent_summary, summary),
                    "routed_policy_execution": "Not evaluated; do not relabel forced TRAIN intervention",
                }
            write_json_atomic(run / "TRAIN_GATES.json", {
                "comparisons": comparisons, "development_cost": report["accounting"],
                "execution_claim": None,
                "gate_scope": "Forced TRAIN instruction intervention, not routed-policy efficiency",
                "next_gate": "Native routed TRAIN operation, then separate frozen protected validation",
            })
            enabled = [row for row in controller.candidate_bank if row["enabled"]]
            if enabled:
                selected = min(enabled, key=lambda row: (
                    -comparisons[row["id"]]["raw_train_intervention"]["native_successes"],
                    comparisons[row["id"]]["raw_train_intervention"]["all_attempt_actor_control_tokens"],
                    row["candidate_sha256"],
                ))
                arm = "routed-" + selected["id"]
                state["status"] = "FROZEN_ROUTED_TRAIN"
                save()
                routed = controller.evaluate_frozen_train(data["tasks"], selected["id"])
                with callbacks.lock:
                    routed_rows = [deepcopy(row) for row in callbacks.completed if row["arm"] == arm]
                summary = execution_summary(routed_rows)
                from phiagent.harness import codex_chisel as core

                ledger = controller.report()["ledger"]
                write_json_atomic(run / "ROUTED_TRAIN_COMPARISON.json", {
                    "summary": summary, "native_batch": routed,
                    "gate": quality_cost_gate(parent_summary, summary),
                    "inherited_parent": core._usage_summary([
                        row for row in ledger if row["arm"] == "baseline"
                    ]),
                    "development_usage": core._usage_summary([
                        row for row in ledger if row["arm"] not in ("baseline", arm)
                    ]),
                    "runtime_actor_control_usage": core._usage_summary([
                        row for row in ledger if row["arm"] == arm
                    ]),
                    "train_freeze_sha256": frozen["train_freeze_sha256"],
                    "next_gate": "Separate protected validation of this unchanged frozen method",
                })
                write_json_atomic(run / "CONTROLLER_REPORT.json", controller.report())
            state["status"] = "COMPLETE_TRAIN_FROZEN"
        except Exception as error:
            state.update(status="BLOCKED", error={"type": type(error).__name__, "message": str(error)})
            write_json_atomic(run / "CONTROLLER_REPORT.json", controller.report())
            raise
        finally:
            stop.set()
            monitor.join(timeout=settings["heartbeat_seconds"] + 1)
            save()


def wait_for_admission(run):
    """Completion-triggered handoff; no native call precedes the owner's release."""
    remote_only()
    if ROOT != run / "source" or not run.is_relative_to(REMOTE_PREFIX):
        raise ValueError("Watch only this operation's own frozen remote run")
    settings = load(run / "RSI_CONFIG.json")
    if sha(run / "RSI_CONFIG.json") != load(run / "SETTINGS_SHA256.json")["RSI_CONFIG.json"]:
        raise ValueError("The frozen admission-watcher settings changed")
    with (run / "HANDOFF_OWNER.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            write_json_atomic(run / "STATE.json", {
                "schema": SCHEMA, "status": "WAITING_NATIVE_OWNER_ADMISSION",
                "pid": os.getpid(), "run_root": str(run), "at": now(),
                "component_identity": IDENTITY, "new_native_requests": 0,
                "candidate_native_scored": 0, "candidate_native_successes": 0,
                "candidate_all_attempt_tokens": None, "cumulative_token_limit": None,
                "validation_read": False, "test_read": False,
                "dependency": settings["admission_dependency"],
            })
            if (run / "STOP").is_file():
                write_json_atomic(run / "HANDOFF_STOPPED.json", {"at": now(), "pid": os.getpid()})
                return
            if (run / "ADMISSION.json").is_file() and (run / "TRAIN_INPUT.json").is_file():
                try:
                    verify_admission(load(run / "ADMISSION.json"), settings, settings["harness"])
                    write_json_atomic(run / "HANDOFF_TRIGGERED.json", {
                        "at": now(), "pid": os.getpid(), "command": sys.argv,
                        "admission_sha256": sha(run / "ADMISSION.json"),
                    })
                    run_training(run)
                    return
                except Exception as error:
                    write_json_atomic(run / "HANDOFF_ERROR.json", {
                        "at": now(), "type": type(error).__name__, "message": str(error),
                        "automatic_retry": False,
                    })
                    raise
            threading.Event().wait(settings["heartbeat_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "run", "watch"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--harness", choices=("codex", "deepseek"))
    args = parser.parse_args()
    if args.operation == "stage":
        if args.config is None or args.harness is None:
            parser.error("stage requires --config and --harness")
        stage(args.run_dir, args.config, harness=args.harness)
    elif args.operation == "run":
        run_training(args.run_dir)
    else:
        wait_for_admission(args.run_dir)


if __name__ == "__main__":
    main()
