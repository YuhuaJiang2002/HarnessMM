#!/usr/bin/env python3
"""Remote, durable TRAIN/validation RSI using the original industrial GAIA2 adapters.

CONFIG.json preserves the native runtime template. RSI_CONFIG.json governs this
new four-round efficiency search. No historical TEST outcome is loaded, and this
runner cannot release TEST. Existing native runs and their sources are untouched.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import write_json_atomic  # noqa: E402

HOST = "harnessmm-api-worker"
CODEX = Path("/opt/phiagent/work/codex-chisel/20260929-v1/gaia2/full-comparison-v4")
DEEPSEEK = Path(
    "/opt/phiagent/work/deepseek-harness/20260929-v1/runs/"
    "gaia2-chisel-campaign-20260929T112727Z-v1"
)
NEW_FILES = (
    "scripts/run_industrial_efficiency_rsi.py",
    "scripts/run_deepseek_gaia2_campaign.py",
    "phiagent/harness/industrial_efficiency_rsi.py",
    "phiagent/harness/codex_chisel.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote_only():
    if sys.platform != "linux" or socket.gethostname() != HOST:
        raise RuntimeError("Industrial native experiments must run on authorized A800-4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This API-only campaign requires explicit CUDA_VISIBLE_DEVICES=''")


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=30)
    return {"argv": args, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def capacity(root):
    data = {
        "at": now(), "hostname": socket.gethostname(), "CUDA_VISIBLE_DEVICES": "",
        "selected_gpus": [],
        "reason": "Pinned industrial DeepSeek API runtime; H200 substitution would change the backbone",
        "gpu": command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                        "--format=csv,noheader"]),
        "gpu_processes": command([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader",
        ]),
        "memory": command(["free", "-b"]), "leases": command(["lslocks", "--json"]),
        "load": command(["uptime"]), "free_disk_bytes": shutil.disk_usage(root).free,
    }
    write_json_atomic(root / "capacity" / f"{time.time_ns()}.json", data)
    return data


def validate_output_root(root, harness):
    prefix = (
        Path("/opt/phiagent/work/codex-chisel/20260929-v1/gaia2")
        if harness == "codex" else DEEPSEEK.parent
    )
    if (not root.is_absolute() or ".." in root.parts or root == prefix
            or not root.is_relative_to(prefix)):
        raise ValueError("Run directory must satisfy the original native adapter's output-root contract")
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("New runs cannot use symlinked output roots")


def prepare(root, harness, staging, pool, continuation=None):
    remote_only()
    validate_output_root(root, harness)
    if root.exists():
        raise RuntimeError("New source/settings require a new run directory")
    if continuation is not None and harness != "deepseek":
        raise ValueError("This prospective recovery is restricted to the stopped DeepSeek lineage")
    predecessor = continuation or (CODEX if harness == "codex" else DEEPSEEK)
    previous_config = load(predecessor / "CONFIG.json")
    if harness == "codex":
        argv = load(predecessor / "provenance.json")["command"]
        old_source = Path(argv[0]).parents[1]
        native = {name: argv[argv.index("--" + name) + 1]
                  for name in ("partition", "freeze", "pilot")}
    else:
        old_source = predecessor / "source"
        native = {}
    root.mkdir(parents=True, mode=0o700)
    shutil.copytree(old_source, root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    if harness == "deepseek":
        codex_command = load(CODEX / "provenance.json")["command"]
        proposer_source = Path(codex_command[0]).parents[1]
        for path in proposer_source.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            destination = root / "source" / path.relative_to(proposer_source)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    for name in NEW_FILES:
        if continuation is not None and name not in (
                "scripts/run_industrial_efficiency_rsi.py",
                "phiagent/harness/industrial_efficiency_rsi.py"):
            continue
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging / name, destination)
    native_config = deepcopy(previous_config)
    for key in ("frozen_learning_replay", "inherited_gaia_training"):
        native_config.pop(key, None)
    native_config["efficiency_successor_owner"] = "run_industrial_efficiency_rsi.py"
    write_json_atomic(root / "CONFIG.json", native_config)
    settings = {
        "schema": "industrial-efficiency-rsi/v1", "harness": harness, "created_at": now(),
        "run_id": root.name, "predecessor": str(predecessor), "native": native,
        "native_source": str(old_source), "source_base_revision": previous_config.get(
            "source_revision", "a468ee05b5c842d0fcd6d8af241f7b312c2e42f3"),
        "optimization_iterations": 4, "executor_workers": 2,
        "max_residual_characters": 1200, "max_residual_words": 180,
        "execution_seeds": [2026093001], "test_access": False,
        "partition": {"train": 32, "validation": 32, "test": 96},
        "cumulative_token_limit": None, "provider_sampling_seed_supported": False,
        "cost_objective": "native quality nonregression and strictly lower total actor tokens",
        "proposal_trace_scope": "TRAIN only, native deterministic excerpts; complete raw traces retained",
        "native_runtime_template_optimizer_fields_unused": True,
        "historical_costs": "Preserved in predecessor; not relabeled as new paid traffic",
        "output_root_contract": "Original native adapter root restrictions retained unchanged",
        "stop_file": str(root / "STOP"),
        "successor_task": "Next feedback-driven TRAIN proposal, then protected validation selection",
        "deepseek_optimizer_transport": "native_responses_json_object",
        "deepseek_controller_cost_scope": "actor_only_with_judge_separate",
    }
    if continuation is not None:
        if load(continuation / "STATE.json")["state"] != "BLOCKED":
            raise ValueError("Do not take over a healthy or running native campaign")
        reuse, failed_attempts = {}, []
        for seal_path in sorted((continuation / "callbacks").glob("*/*/COMMITTED.json")):
            folder = seal_path.parent
            for name, digest in load(seal_path).items():
                if sha(folder / name) != digest:
                    raise RuntimeError("Predecessor callback changed")
            request, result, usage = (load(folder / name)
                                      for name in ("REQUEST.json", "RESULT.json", "USAGE.json"))
            if request["phase"] != "train":
                raise ValueError("Only TRAIN receipts may enter this recovery")
            if folder.parent.name == "executor":
                if (request["arm"] != "baseline" or request["residual"] is not None
                        or result["execution_status"] != "completed"
                        or request["seed"] != settings["execution_seeds"][0]):
                    raise ValueError("Only the unchanged complete TRAIN baseline may be reused")
                ident = request["task"]["task_id"]
                if ident in reuse:
                    raise ValueError("A baseline task would be charged twice")
                reuse[ident] = {"directory": str(folder), "seal_sha256": sha(seal_path)}
            else:
                if result["usage"]["total_tokens"] is None:
                    raise ValueError("Recover the failed proposer's actual usage before continuing")
                failed_attempts.extend(usage["attempts"])
        if len(reuse) != 32:
            raise ValueError("Recovery requires all 32 native baseline receipts")
        reuse_record = {
            "previous": str(continuation), "baseline": reuse,
            "failed_proposer_attempts": failed_attempts,
            "failed_proposer_tokens": sum(row["usage"]["total_tokens"] for row in failed_attempts),
            "new_baseline_model_calls": 0, "native_actor_or_scorer_changed": False,
            "reason": "Separate actor/judge costs and replace unstructured SDK proposal text "
                      "with same-model native Responses JSON mode",
        }
        write_json_atomic(root / "PAID_REUSE.json", reuse_record)
        settings["paid_reuse_sha256"] = sha(root / "PAID_REUSE.json")
    write_json_atomic(root / "RSI_CONFIG.json", settings)
    source_hashes = {
        path.relative_to(root / "source").as_posix(): sha(path)
        for path in sorted((root / "source").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    write_json_atomic(root / "SOURCE_SHA256.json", source_hashes)
    for name, path in (("POOL_SELECTION.json", pool),):
        shutil.copyfile(path, root / name)
    write_json_atomic(root / "PREPARATION.json", {
        "at": now(), "hostname": socket.gethostname(), "command": sys.argv,
        "git": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "source_revision": settings["source_base_revision"],
        "native_config_source": str(predecessor / "CONFIG.json"),
        "native_config_source_sha256": sha(predecessor / "CONFIG.json"),
        "new_settings_sha256": sha(root / "RSI_CONFIG.json"),
        "packages": {dist.metadata["Name"]: dist.version
                     for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
        "python": sys.version, "executable": sys.executable, "seed": 2026093001,
        "cumulative_token_limit": None,
    })
    capacity(root)
    print(json.dumps({"prepared": str(root), "harness": harness,
                      "sources": len(source_hashes), "test_access": False}))


def worker(root):
    remote_only()
    from phiagent.harness.codex_chisel import ChiselConfig
    from phiagent.harness.industrial_efficiency_rsi import (
        IndustrialEfficiencyController, actor_only_result,
    )

    settings = load(root / "RSI_CONFIG.json")
    if settings["test_access"] is not False or settings["cumulative_token_limit"] is not None:
        raise ValueError("This successor forbids TEST access and cumulative token ceilings")
    hashes = load(root / "SOURCE_SHA256.json")
    for name, digest in hashes.items():
        if sha(root / "source" / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    if sha(root / "RSI_CONFIG.json") != load(root / "PREPARATION.json")["new_settings_sha256"]:
        raise RuntimeError("Frozen RSI settings changed")
    stop = threading.Event()
    active = {}
    active_lock = threading.RLock()
    status = {"state": "initializing", "at": now(), "pid": os.getpid(),
              "test_opened": False, "cumulative_token_limit": None}
    controller = None
    reuse = load(root / "PAID_REUSE.json") if (root / "PAID_REUSE.json").exists() else None
    if reuse is not None and sha(root / "PAID_REUSE.json") != settings["paid_reuse_sha256"]:
        raise RuntimeError("Paid TRAIN reuse manifest changed")

    def save():
        with active_lock:
            write_json_atomic(root / "STATE.json", {
                **status, "at": now(), "active": list(active.values()),
                "successor_task": settings["successor_task"],
                "controller_state": controller.state if controller else None,
                "callbacks_completed": len(controller._ledger) - len(active) if controller else 0,
            })
            if controller:
                report = controller.report()
                if reuse is not None:
                    prior = reuse["failed_proposer_tokens"]
                    usage = report["accounting"]["all_calls"]["usage"]["total_tokens"]
                    report["predecessor_accounting"] = {
                        "source": str(root / "PAID_REUSE.json"),
                        "failed_proposer_tokens": prior,
                        "baseline_reused_and_charged_once": True,
                        "old_failed_proposer_cost_not_discarded": True,
                        "known_actor_and_proposer_tokens_including_predecessor":
                            usage["measured_subtotal"] + prior,
                        "actor_and_proposer_tokens_including_predecessor":
                            None if usage["total"] is None else usage["total"] + prior,
                    }
                write_json_atomic(root / "CONTROLLER_REPORT.json", report)

    def heartbeat():
        previous_capacity = 0
        while not stop.is_set():
            try:
                if time.monotonic() - previous_capacity >= 300:
                    capacity(root)
                    previous_capacity = time.monotonic()
                save()
            except Exception as error:
                write_json_atomic(root / "MONITOR_ERROR.json", {
                    "at": now(), "type": type(error).__name__, "message": str(error),
                })
            stop.wait(30)

    def guarded(callback, request, kind):
        if settings.get("closeout_only"):
            directory = root / "callbacks" / kind / request["request_id"]
            if (request["request_id"] not in settings["closed_callback_ids"]
                    or not (directory / "COMMITTED.json").is_file()):
                raise RuntimeError("Closeout forbids any new actor or proposer dispatch")
        if (root / "STOP").exists():
            raise RuntimeError("User stop requested; active native calls drain without new submissions")
        if request["phase"] == "test":
            raise RuntimeError("Opened historical TEST is excluded from this successor")
        with active_lock:
            active[request["request_id"]] = {
                "request_id": request["request_id"], "kind": kind, "phase": request["phase"],
                "task_id": request.get("task", {}).get("task_id"), "started_at": now(),
            }
        try:
            return callback(request)
        finally:
            with active_lock:
                active.pop(request["request_id"], None)

    with (root / "OWNER.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        if settings["harness"] == "codex":
            from scripts.run_codex_gaia2_campaign import GAIAComparison
            adapter = GAIAComparison(
                load(root / "CONFIG.json"), root / "native",
                *(Path(settings["native"][name]) for name in ("partition", "freeze", "pilot")),
            )
            tasks = adapter.phase_tasks
            index = adapter.task_index()
            identity = {
                "model": "deepseek-flash", "runtime": "official_codex_cli", "codex_version": "0.142.5",
                "wire_api": "responses", "transport": "byte_preserving_metering_proxy",
                "benchmark": "gaia2-mini",
            }
            optimize_identity = {"model": "deepseek-flash", "runtime": "native_responses"}
            execute, optimize = adapter.execute, adapter.optimize
            application = "agents_md"
            native_name, native_hash = "codex_cli", sha(Path(adapter.config["codex"]["binary"]))
        else:
            from scripts.run_deepseek_gaia2_campaign import Campaign
            from phiagent.harness.deepseek_chisel import adapt_request
            from scripts.run_codex_gaia2_campaign import GAIAComparison

            adapter = Campaign(root)
            adapter.preflight()
            tasks, index = adapter.tasks, adapter.index
            identity = adapter.identity
            optimize_identity = {"model": "deepseek-flash", "runtime": "native_responses",
                                 "wire_api": "responses"}
            codex_command = load(CODEX / "provenance.json")["command"]
            proposer_config = load(CODEX / "CONFIG.json")
            proposer_config["benchmark_tag"] = "dsh-efficiency"
            proposer = GAIAComparison(
                proposer_config, root / "responses-proposer",
                *(Path(codex_command[codex_command.index("--" + name) + 1])
                  for name in ("partition", "freeze", "pilot")),
            )
            proposer.config["benchmark_tag"] = "dsh-efficiency"

            def execute(request):
                adapter.phase = request["phase"]
                adapted = adapt_request(request, optimizer=False)
                ident = request["task"]["task_id"]
                if reuse is not None and request["phase"] == "train" and request["arm"] == "baseline":
                    previous = Path(reuse["baseline"][ident]["directory"])
                    if sha(previous / "COMMITTED.json") != reuse["baseline"][ident]["seal_sha256"]:
                        raise RuntimeError("Native baseline reuse seal changed")
                    from scripts.run_deepseek_gaia2_campaign import verified_callback
                    old_request, old_result, attempts = verified_callback(previous)
                    if (old_request["task"] != adapted["task"]
                            or old_request["seed"] != adapted["seed"]
                            or old_request["identity"] != adapted["identity"]
                            or adapted["residual"] is not None):
                        raise ValueError("Reused baseline must preserve task, model, seed and native input")
                    directory, _ = adapter.callback_directory("executor", adapted)
                    result = deepcopy(old_result)
                    result["request_id"] = adapted["request_id"]
                    result["provenance"]["paid_train_reuse"] = {
                        "source": str(previous), "original_request_id": old_request["request_id"],
                        "new_model_calls": 0,
                    }
                    adapter.commit(directory, result, attempts)
                else:
                    result = adapter.guarded("executor", adapted)
                return actor_only_result(result)

            def optimize(request):
                adapter.phase = "train"
                directory, _ = adapter.callback_directory("optimizer", request)
                existing = adapter.committed(directory)
                if existing is not None:
                    return existing
                result = proposer.optimize(request)
                payload = proposer.work / "optimizer" / request["request_id"] / "REQUEST.json"
                if sha(payload) != result["provenance"]["request_sha256"]:
                    raise RuntimeError("Native Responses optimizer prompt changed")
                shutil.copyfile(payload, directory / "PROVIDER_PROMPT.json")
                adapter.commit(directory, result, [{
                    "attempt_id": str(payload.parent), "stage": "train_optimizer",
                    "usage": result["usage"], "provider_attempt_count": 1,
                    "provider_native_usage": result["provenance"]["provider_usage"],
                    "known_reported_usage": result["usage"], "reasoning_tokens": None,
                }])
                return result
            application = "system_section"
            native_name = "deepseek_harness"
            native_hash = adapter.identity["runtime_manifest_sha256"]
        controller = IndustrialEfficiencyController(
            partition=adapter.partition, task_index=index,
            execute=lambda request: guarded(execute, request, "executor"),
            optimize=lambda request: guarded(optimize, request, "optimizer"),
            execution_identity=identity, optimizer_identity=optimize_identity,
            run_id=settings["run_id"], config=ChiselConfig(
                application=application, native_metric="official_gaia2_task_success",
                optimization_iterations=settings["optimization_iterations"],
                executor_workers=settings["executor_workers"],
                execution_seeds=tuple(settings["execution_seeds"]),
                test_execution_seeds=(2026093001, 2026093002, 2026093003),
            ),
        )
        monitor = threading.Thread(target=heartbeat, name="efficiency-rsi-monitor", daemon=True)
        monitor.start()
        try:
            status["state"] = "TRAIN"
            controller.train(tasks("train"))
            write_json_atomic(root / "TRAIN_COMPLETE.json", controller.report())
            admitted = [row for row in controller.candidate_bank if row["enabled"]]
            if admitted:
                status["state"] = "VALIDATION"
                selection = controller.select_validation(tasks("validation"))
                write_json_atomic(root / "VALIDATION_SELECTION.json", selection)
                policy = controller.freeze(source_hashes={
                    native_name: native_hash, "executor": sha(Path(__file__)),
                    "optimizer": sha(root / "source/phiagent/harness/industrial_efficiency_rsi.py"),
                    "scorer": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                    "all_sources": sha(root / "SOURCE_SHA256.json"),
                })
                write_json_atomic(root / "FROZEN_POLICY.json", policy.to_dict())
                status["state"] = "COMPLETE_VALIDATION_TEST_NOT_AUTHORIZED"
            else:
                status["state"] = "COMPLETE_TRAIN_NO_ADMISSIBLE_SAVING"
            status["accepted_candidates"] = len(admitted)
        except Exception as error:
            status.update(state="BLOCKED", error_type=type(error).__name__, error=str(error))
            write_json_atomic(root / "FAILURE.json", {
                **status, "traceback": traceback.format_exc(), "at": now(),
                "retry": "No blind replay; preserve pending native actions and measured costs",
            })
            raise
        finally:
            stop.set()
            monitor.join(timeout=65)
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare", choices=("codex", "deepseek"))
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--pool-manifest", type=Path)
    parser.add_argument("--continue-from", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.staging is None or args.pool_manifest is None:
            parser.error("Preparation needs --staging and --pool-manifest")
        prepare(args.run_dir, args.prepare, args.staging, args.pool_manifest, args.continue_from)
    else:
        worker(args.run_dir)


if __name__ == "__main__":
    main()
