#!/usr/bin/env python3
"""One new complete official loop after a proved XML integer-decoding repair.

The terminal predecessor is preserved, not reopened or rewritten. Only exact
native TRAIN payloads can be reused. Training produces a frozen candidate;
external validation/test require a separate all-method qualified release.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from scripts import run_chisel_interactive_complete_optimizers as base  # noqa: E402
from scripts import chisel_alfworld_native_cache as cache  # noqa: E402

n = base.native


def prepare(config_path):
    n.remote_only()
    campaign = n.read(config_path)
    parent = Path(campaign["parent_run"])
    config = n.read(parent / "config.json")
    original = n.read(Path(config["prepared_directory"]) / "BINDING.json")
    n.verify_signed(original, "binding_sha256")
    n.require(config["method"] == "autosaddler" and config["autosaddler_iterations"] == 5,
              "Different official optimizer recipe")
    source = n.read(ROOT / "SOURCE_PROVENANCE.json")
    output = Path(campaign["prepared_directory"])
    output.mkdir(parents=True, exist_ok=False)
    binding = deepcopy(original)
    binding.pop("binding_sha256")
    binding.update(
        source_hashes=source["source_hashes"], source_provenance=source,
        predecessor_binding_sha256=original["binding_sha256"],
        continuation_kind="new_complete_official_run_from_seed_after_evidenced_adapter_bug",
        checkpoint_resume_decision="The predecessor has RunCompleted; supported resume would return "
        "the cached seed. Reopening it would rewrite terminal/retry history, so preserve it and "
        "start exactly one identified complete loop using unchanged scientific settings.",
    )
    binding["binding_sha256"] = n.digest(binding)
    n.write(output / "BINDING.json", binding)
    config.update(
        run_directory=campaign["run_directory"], prepared_directory=str(output),
        source_root=str(ROOT), binding_sha256=binding["binding_sha256"],
        gateway_port=campaign["gateway_port"], parent_run=str(parent),
        repair_gate=campaign["repair_gate"],
        native_cache_manifest=str(output / "NATIVE_CACHE.json"),
        training_only=True, explicit_test_release_required=True,
    )
    n.write(output / "config.json", config)
    cached = cache.prepare(parent, config, binding, ROOT, output / "NATIVE_CACHE.json")
    gate = Path(campaign["repair_gate"])
    lineage = n.read(gate / "LINEAGE_COST_COMPONENTS_V2.json")
    n.require(lineage["status"] == "PASS_EXACT_ONCE_COMPLETE_OPTIMIZATION_LINEAGE",
              "Predecessor cost lineage is unresolved")
    n.write(output / "INHERITED_ACCOUNTING.json", {
        "source": str(gate / "LINEAGE_COST_COMPONENTS_V2.json"),
        "sha256": n.file_hash(gate / "LINEAGE_COST_COMPONENTS_V2.json"),
        "components": lineage, "diagnostic_gate_calls_replayed": False,
        "inherited_once": True,
    })
    n.write(output / "PREPARED.json", {
        "status": "READY_ONE_CORRECTED_OFFICIAL_TRAIN_LOOP",
        "binding_sha256": binding["binding_sha256"], "cached_native_cells": len(cached["entries"]),
        "cache_sha256": cached["cache_sha256"], "model_calls": 0,
        "primary_counts": binding["primary_counts"],
        "fit": len(binding["optimizer_feedback"]["train"]),
        "internal_feedback": len(binding["optimizer_feedback"]["development"]),
    })
    print(json.dumps(n.read(output / "PREPARED.json")), flush=True)


def physical_costs(root, inherited):
    actor = dict.fromkeys(base.ACTOR_KEYS, 0)
    actor_calls = 0
    logical_cached = 0
    for path in (root / "autosaddler").rglob("output.json"):
        result = n.read(path)
        if result.get("schema_version") != "phiagent-alfworld-candidate-eval-result/v1":
            continue
        n.verify_signed(result, "canonical_sha256")
        for key in actor:
            actor[key] += result["new_physical_usage"][key]
        actor_calls += sum(row["calls"] for row in result["rows"] if "execution_reuse" not in row)
        logical_cached += result["reused_native_cells"]
    return {"new_native_actor_usage": actor, "new_native_actor_calls": actor_calls,
            "reused_logical_native_cells": logical_cached,
            "inherited_physical_tokens": inherited[
                "optimization_total_including_native_feedback_and_repair"],
            "cached_physical_cost_charged_again": False}


def official_return_audit(store, iterations):
    events = [json.loads(line) for line in (store / "events.jsonl").read_text().splitlines()]
    completed = [event["payload"] for event in events if event["event_type"] == "IterationCompleted"]
    n.require(sorted(row["iteration"] for row in completed) == list(range(iterations)),
              "Missing official terminal rounds")
    n.require(sum(event["event_type"] == "RunCompleted" for event in events) == 1,
              "Official engine did not complete")
    pending = set()
    for event in events:
        if event["event_type"] == "DeferredWorkScheduled":
            pending.add(event["payload"]["obligation_id"])
        elif event["event_type"] in ("DeferredWorkCompleted", "DeferredWorkAbandoned"):
            n.require(event["payload"]["obligation_id"] in pending, "Invalid deferred-work history")
            pending.remove(event["payload"]["obligation_id"])
    n.require(not pending, "Official deferred work is still pending")
    return {
        "status": "PASS_OFFICIAL_PROTOCOL_EXECUTION",
        "iterations": completed,
        "failed_sessions": [event for event in events if event["event_type"] == "SessionFailed"],
        "abandoned_deferred_work": [event for event in events
                                    if event["event_type"] == "DeferredWorkAbandoned"],
        "success_or_accepted_update_is_not_required_for_completion": True,
        "events_sha256": n.file_hash(store / "events.jsonl"),
    }


def run(config_path):
    config = n.read(config_path)
    n.remote_only(config["authorized_hostname"])
    root = Path(config["run_directory"])
    root.mkdir(parents=True, exist_ok=False)
    n.write(root / "config.json", config)
    binding = n.read(Path(config["prepared_directory"]) / "BINDING.json")
    n.verify_signed(binding, "binding_sha256")
    for name, checksum in binding["source_hashes"].items():
        n.require(n.file_hash(ROOT / name) == checksum, f"Corrected source changed: {name}")
    auditor_path = Path("/opt/phiagent/work/chisel-interactive/20260929-live-gate-audit-v1/"
                        "audit_chisel_interactive_gate.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("corrected_resource_audit", auditor_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    parent = Path(config["parent_run"])
    selected = audit.gpu_selection(parent)
    n.write(root / "GPU_SELECTION.json", {
        **selected, "borrowed_owned_model_service": True, "new_model_process": False,
        "new_client_owner_pid": os.getpid(), "CUDA_VISIBLE_DEVICES": "0",
    })
    os.environ.update(CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NO_PROXY="127.0.0.1,localhost,214.40.221.1,214.40.221.2")
    n.write(root / "PID.json", {"pid": os.getpid(), "at_utc": n.now(),
                                "hostname": socket.gethostname(), "argv": sys.argv})
    n.write(root / "PROVENANCE.json", {
        "source": binding["source_provenance"], "argv": sys.argv, "hostname": socket.gethostname(),
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        "seed": 20260928, "wire_seed": 9401,
        "frozen_parent_config_sha256": n.file_hash(parent / "config.json"),
        "shared_existing_service": True,
    })
    inherited_record = n.read(Path(config["prepared_directory"]) / "INHERITED_ACCOUNTING.json")
    n.write(root / "INHERITED_ACCOUNTING.json", inherited_record)
    inherited = inherited_record["components"]
    gateway = base.Gateway(config, root)
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    stop = threading.Event()

    def monitor():
        audit_at = 0
        while not stop.is_set():
            try:
                costs = physical_costs(root, inherited)
                provider = gateway.summary()
                n.write(root / "HEARTBEAT.json", {
                    "at_utc": n.now(), "provider": provider, "cost": costs,
                    "state": n.read(root / "STATE.json") if (root / "STATE.json").exists() else {},
                    "test_sealed": not (root / "TEST_SEAL.json").exists(),
                })
                if time.monotonic() >= audit_at:
                    n.append(root / "CAPACITY_AUDIT.jsonl", n.resource_audit())
                    audit_at = time.monotonic() + 300
            except Exception as error:
                n.write(root / "MONITOR_ERROR.json", {"error": str(error),
                                                      "traceback": traceback.format_exc()})
            stop.wait(min(120, max(1, audit_at - time.monotonic())))

    threading.Thread(target=monitor, daemon=True).start()
    try:
        n.write(root / "STATE.json", {"status": "STARTING_ONE_CORRECTED_OFFICIAL_LOOP", "at_utc": n.now()})
        study = base.Study(config, binding, root, gateway)
        skill = study.autosaddler()
        gateway.optimizer_open = False
        proof = official_return_audit(root / "autosaddler/native-train-only",
                                     config["autosaddler_iterations"])
        n.write(root / "OFFICIAL_COMPLETION_AUDIT.json", proof)
        policy = base.global_policy(skill)
        candidate = {
            "method": "autosaddler", "policy": policy, "binding_sha256": binding["binding_sha256"],
            "base_comparison_binding_sha256": binding["predecessor_binding_sha256"],
            "source_hashes": binding["source_hashes"],
            "official_engine_revision": base.REVISIONS["autosaddler"],
            "at_utc": n.now(), "external_validation_seen": False, "test_seen": False,
        }
        candidate["candidate_sha256"] = n.digest(candidate)
        n.write(root / "CANDIDATE.json", candidate)
        costs = physical_costs(root, inherited)
        provider = gateway.summary()
        n.require(not provider["unknown_usage_calls"] and not provider["failed_calls"],
                  "New provider costs are not fully reported")
        costs.update(
            new_optimizer_provider=provider,
            physical_lineage_tokens=costs["inherited_physical_tokens"] +
            costs["new_native_actor_usage"]["total_tokens"] + provider["total_tokens"],
            exact_once=True, final_execution_tokens_not_included=True,
        )
        n.write(root / "COST_AUDIT.json", costs)
        n.write(root / "TRAIN_READY.json", {
            "status": "COMPLETE_CORRECTED_OFFICIAL_TRAIN_LOOP",
            "candidate_sha256": candidate["candidate_sha256"], "at_utc": n.now(),
            "failed_sessions": len(proof["failed_sessions"]),
            "abandoned_deferred_work": len(proof["abandoned_deferred_work"]),
            "physical_lineage_tokens": costs["physical_lineage_tokens"],
        })
        n.write(root / "STATE.json", {"status": "TRAIN_COMPLETE_ALL_METHOD_RELEASE_REQUIRED",
                                      "at_utc": n.now(), "test_opened": False})
    except BaseException as error:
        n.write(root / "FAILURE.json", {"error": str(error), "traceback": traceback.format_exc(),
                                        "at_utc": n.now(), "no_automatic_retry": True})
        raise
    finally:
        stop.set()
        n.write(root / "NEW_PROVIDER_USAGE.json", gateway.summary())
        gateway.shutdown()
        gateway.server_close()
        n.write(root / "RETIREMENT.json", {"at_utc": n.now(), "borrowed_model_service_stopped": False,
                                          "frozen_parent_changed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "external/AutoSaddler/src"), str(ROOT / "external/SkillOpt"),
                    str(ROOT / "integrations/autosaddler_phiagent/src")]
    {"prepare": prepare, "run": run}[args.command](args.config)


if __name__ == "__main__":
    main()
