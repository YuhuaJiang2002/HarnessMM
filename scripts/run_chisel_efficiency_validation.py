#!/usr/bin/env python3
"""Validate the exact TRAIN-selected memory_guarded64 policy against one frozen reference."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_chisel_efficiency_train as training  # noqa: E402

common, native, core = training.common, training.native, training.core
SCHEMA = "chisel-efficiency-validation/v1"
METHOD = "memory_guarded64"
LABEL = "CHISEL memory_guarded64 / TRAIN-selected validation"
TRAIN_SOURCE = "scripts/run_chisel_efficiency_train.py"
PROMPT_SOURCE = "phiagent/learning/alfworld_modular_rsi.py"
REFERENCE_FILES = (
    "RESULT.json", "INDEPENDENT_COMPLETION_AUDIT.json", "POLICY_FREEZE.json",
    "config.json", "manifest.json", "CHECKPOINT.json",
    "validation_gate-RESULT.json", "validation-RESULT.json",
    "validation_gate-rows.jsonl", "validation-rows.jsonl",
)
MODEL_FIELDS = ("repository", "revision", "served_name", "max_model_len", "enable_thinking")
ACTOR_SOURCES = ("scripts/run_chisel_repair_alfworld.py",
                 "scripts/run_alfworld_autosaddler_worker.py")


def validate_contract(config: dict, worker: dict, freeze: dict, protocol: dict) -> dict:
    native.require(config["schema_version"] == SCHEMA and config["method"] == METHOD
                   and config["implementation_label"] == LABEL, "Wrong validation identity")
    native.require(config["scope"] == "validation" and config["validation_tasks"] == 710
                   and config["first_episode_tasks"] == 1, "Require all710 validation tasks")
    native.require(config["test_access"] is False
                   and config["optimization_from_validation"] is False
                   and config["reuse_reference"] is True
                   and type(config["pinned_resource_only"]) is bool,
                   "No test access, policy updates or reference reruns; resource policy must be explicit")
    native.require(config["minimum_token_reduction"] == 0.10
                   and config["quality_loss_allowance"] == 0
                   and config["bootstrap_replicates"] == 10000
                   and config["bootstrap_seed"] == 20260929,
                   "Prospective comparison or uncertainty settings changed")
    parent = common.validate_frozen_instruction_contract(config, worker, freeze, protocol)
    selected = training.policies_from(parent)[METHOD]
    native.verify_signed(selected, "harness_sha256")
    native.require(selected["harness_sha256"] == config["selected_policy_sha256"],
                   "TRAIN-selected policy changed")
    return selected


def pinned_json(entry: dict) -> dict:
    path = Path(entry["path"])
    native.require(native.file_hash(path) == entry["sha256"], f"Frozen evidence changed: {path}")
    return native.read(path)


def validate_selection(config: dict, selected: dict, protocol: dict) -> dict:
    entries = config["training_evidence"]
    comparison, audit, freeze = (pinned_json(entries[key]) for key in ("comparison", "audit", "freeze"))
    native.require(comparison["status"] == "COMPLETE_TRAIN_ONLY_PROBE"
                   and comparison["scope"] == "training_only"
                   and comparison["tasks"] == 96 and comparison["episodes"] == 384
                   and comparison["selected_for_frozen_validation"] == METHOD
                   and comparison["candidates"][METHOD]["eligible_for_frozen_validation"] is True
                   and comparison["validation_result_available"] is False
                   and comparison["test_result_available"] is False,
                   "A complete TRAIN-only selection is required")
    native.require(audit["status"] == "PASS_COMPLETE_TRAIN_RAW_RECONSTRUCTION"
                   and audit["tasks"] == 96 and audit["episodes"] == 384
                   and audit["train_comparison_sha256"] == entries["comparison"]["sha256"]
                   and audit["validation_or_test_inputs_used"] is False
                   and audit["all_actions_model_generated"] is True,
                   "Independent TRAIN audit is missing or bound to different evidence")
    for method in training.PROFILES:
        for key in ("tasks", "successes", "calls", *common.USAGE_KEYS):
            native.require(comparison["summary"][method][key] == audit["summary"][method][key],
                           "TRAIN comparison and raw audit disagree")
    summaries = comparison["summary"]
    floor = max(summaries[name]["successes"] for name in ("reference", "no_skill_memory"))
    eligible = [
        name for name in ("recurrence_memory", METHOD)
        if summaries[name]["successes"] >= floor
        and summaries[name]["total_tokens"] <= 0.90 * summaries["reference"]["total_tokens"]
    ]
    ranked = sorted(eligible, key=lambda name: (
        -summaries[name]["successes"], summaries[name]["total_tokens"], name))
    native.require(ranked and ranked[0] == METHOD,
                   "Stored TRAIN selection does not reproduce the predefined quality/cost rule")
    native.verify_signed(freeze, "freeze_sha256")
    native.require(freeze["scope"] == "train" and freeze["test_access"] is False
                   and freeze["validation_access"] is False
                   and freeze["partition_sha256"] == config["partition_sha256"]
                   and freeze["policies"][METHOD] == selected,
                   "The selected policy does not match the completed TRAIN freeze")
    parent = {"skills": selected["skills"]}
    parent["harness_sha256"] = native.digest(parent)
    native.require(freeze["policies"] == training.policies_from(parent),
                   "TRAIN references and selected candidate used different frozen skills")
    ids = freeze["all_probe_ids"]
    native.require(len(ids) == len(set(ids)) == 96
                   and set(ids) <= {item["id"] for item in protocol["splits"]["train"]},
                   "Training selection escaped the frozen TRAIN partition")
    for name, checksum in freeze["source_hashes"].items():
        native.require(native.file_hash(ROOT / name) == checksum,
                       f"Completed TRAIN implementation changed: {name}")
    for name, checksum in config["policy_implementation_hashes"].items():
        native.require(freeze["source_hashes"][name] == checksum,
                       f"Policy implementation pin differs from TRAIN: {name}")
    return freeze


def validate_rows(rows: list[dict], items: list[dict], method: str, policy_hash: str) -> None:
    expected = {item["id"]: item for item in items}
    native.require(len(rows) == len(expected) == len({row["id"] for row in rows})
                   and {row["id"] for row in rows} == set(expected),
                   "Incomplete, duplicate or foreign comparison tasks")
    for row in rows:
        native.require(row["gamefile"] == expected[row["id"]]["gamefile"]
                       and row["method"] == method and row["seed"] == 9401
                       and row["model_id"] == "qwen35-9b@main"
                       and row["model_revision"] == native.REVISION
                       and row["parent_harness_sha256"] == policy_hash,
                       "Task/gamefile/seed/model/policy equivalence failed")
        native.require(row["valid"] is True and row["cost_complete"] is True
                       and type(row["official_success"]) is int and row["official_success"] in (0, 1)
                       and 0 < row["turns"] <= 50
                       and row["calls"] == row["attempted_calls"] == row["turns"],
                       "Invalid native result or non-model action")
        usage = row["usage"]
        native.require(all(type(usage[k]) is int and usage[k] >= 0 for k in common.USAGE_KEYS)
                       and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
                       and usage["total_tokens"] > 0, "Incomplete model-token accounting")


def check_reference_metadata(config, runtime, old_freeze, selected, items, bundle):
    result, audit, freeze = (bundle[key] for key in
                            ("RESULT.json", "INDEPENDENT_COMPLETION_AUDIT.json", "POLICY_FREEZE.json"))
    native.verify_signed(result, "result_sha256")
    native.verify_signed(freeze, "freeze_sha256")
    native.verify_signed(bundle["manifest.json"], "manifest_sha256")
    native.require(result["status"] == "COMPLETE_VALIDATION_ONLY_DIAGNOSTIC"
                   and result["freeze_sha256"] == freeze["freeze_sha256"]
                   and result["policy_sha256"] == config["diagnostic_policy_sha256"]
                   and result["partition_sha256"] == config["partition_sha256"]
                   and result["test_access"] is False and result["official_test_calls"] == 0
                   and result["first_episode_counted_once"] is True,
                   "Reference is incomplete or belongs to another frozen comparison")
    native.require(audit["status"] == "PASS_COMPLETE_RAW_RECONSTRUCTION"
                   and audit["result_file_sha256"] == config["reference"]["files"]["RESULT.json"]
                   and audit["policy_sha256"] == result["policy_sha256"]
                   and audit["tasks"] == audit["native_row_checks"] == 710
                   and audit["provider_calls_checked"] == audit["calls"] == result["summary"]["calls"]
                   and audit["all_actions_model_generated"] is True
                   and audit["first_episode_exact_once"] is True and audit["test_access"] is False
                   and audit["successes"] == result["summary"]["successes"],
                   "Reference independent raw reconstruction is missing or mismatched")
    native.require(freeze["partition_sha256"] == config["partition_sha256"]
                   and freeze["membership_sha256"] == config["membership_sha256"]
                   and freeze["native_freeze_sha256"] == config["native_freeze_sha256"]
                   and freeze["skills_sha256"] == config["skills_sha256"]
                   and bundle["CHECKPOINT.json"] == old_freeze["checkpoint"],
                   "Reference checkpoint, skills or partition differs")
    parent = freeze["policies"][common.METHOD]
    native.verify_signed(parent, "harness_sha256")
    native.require(parent["harness_sha256"] == config["diagnostic_policy_sha256"]
                   and parent["skills"] == selected["skills"]
                   and training.policies_from(parent)[METHOD] == selected,
                   "Reference and candidate do not share the exact frozen instructions")
    reference_config, manifest = bundle["config.json"], bundle["manifest.json"]
    native.require(reference_config["scope"] == "validation"
                   and reference_config["episode_budget"] == runtime["episode_budget"] == common.BUDGET
                   and manifest["budget"] == common.BUDGET
                   and manifest["alfworld_environment_mode"] == "eval_out_of_distribution"
                   and reference_config["episode_seed"] == runtime["episode_seed"] == 9401
                   and all(reference_config["model"][key] == runtime["model"][key]
                           for key in MODEL_FIELDS), "Reference model/budget/history interface differs")
    inventory = manifest["task_inventory"]
    native.require(len(inventory) == len(items) == len(freeze["validation_ids"]) == 710
                   and len({item["id"] for item in inventory}) == 710
                   and {item["id"]: item["gamefile"] for item in inventory}
                   == {item["id"]: item["gamefile"] for item in items}
                   and set(freeze["validation_ids"]) == {item["id"] for item in items},
                   "Reference validation population differs")
    for name in ACTOR_SOURCES:
        native.require(freeze["source_hashes"][name] == native.file_hash(ROOT / name),
                       f"Common reference actor changed: {name}")
    for key in common.USAGE_KEYS:
        native.require(audit["usage"][key] == result["summary"][key]
                       == result["cost"]["own_usage"][key], "Reference token receipts disagree")
    native.require(result["cost"]["all_in_calls"] == result["summary"]["calls"]
                   and result["cost"]["all_in_total_tokens"] == result["summary"]["total_tokens"]
                   and result["cumulative_token_mode"] == "unlimited",
                   "Reference complete call/token scope differs")
    return result


def load_reference(config, runtime, old_freeze, selected, items):
    spec = config["reference"]
    native.require(set(spec["files"]) == set(REFERENCE_FILES), "Incomplete reference evidence inventory")
    root = Path(spec["root"])
    bundle, rows = {}, []
    for name, checksum in spec["files"].items():
        native.require(native.file_hash(root / name) == checksum, f"Reference evidence changed: {name}")
        if name.endswith(".json"):
            bundle[name] = native.read(root / name)
    result = check_reference_metadata(config, runtime, old_freeze, selected, items, bundle)
    for stage, count in (("validation_gate", 1), ("validation", 709)):
        receipt = bundle[f"{stage}-RESULT.json"]
        native.require(receipt["status"] == "PASS" and receipt["completed"] == count
                       and receipt["rows_sha256"] == spec["files"][f"{stage}-rows.jsonl"],
                       "Reference first-gate/bulk receipt mismatch")
        part = common.read_rows(root, stage)
        native.require(len(part) == count, "Reference stage is incomplete")
        rows.extend(part)
    validate_rows(rows, items, common.METHOD, config["diagnostic_policy_sha256"])
    native.require(native.summarize(rows)[common.METHOD] == result["summary"],
                   "Reference rows do not reproduce the independently audited totals")
    native.require(native.digest(sorted(row["id"] for row in rows)) == result["validation_ids_sha256"],
                   "Reference ID checksum changed")
    return rows, {
        "root": str(root), "files": spec["files"], "summary": result["summary"],
        "independent_audit_status": bundle["INDEPENDENT_COMPLETION_AUDIT.json"]["status"],
        "policy_sha256": result["policy_sha256"], "reuse_count": 1,
        "new_reference_episodes": 0, "new_reference_model_calls": 0,
        "earlier_failed_reference_attempts_not_pooled": True,
    }


def paired_comparison(candidate, reference, items, *, replicates=10000, seed=20260929):
    ids = [item["id"] for item in items]
    native.require(len(ids) == len(set(ids)) == len(candidate) == len(reference)
                   and {row["id"] for row in candidate} == set(ids)
                   and {row["id"] for row in reference} == set(ids), "Unmatched paired populations")
    left, right = ({row["id"]: row for row in rows} for rows in (reference, candidate))
    groups = defaultdict(lambda: [0, 0, 0, 0, 0])
    gains = losses = 0
    for item in items:
        a, b = left[item["id"]], right[item["id"]]
        native.require(a["gamefile"] == b["gamefile"] == item["gamefile"],
                       "Paired gamefiles differ")
        gains += b["official_success"] > a["official_success"]
        losses += b["official_success"] < a["official_success"]
        bucket = groups[core.repair.floorplan(item["gamefile"])]
        values = (1, a["official_success"], b["official_success"],
                  a["usage"]["total_tokens"], b["usage"]["total_tokens"])
        for index, value in enumerate(values):
            bucket[index] += value

    def metrics(totals):
        n, r_success, c_success, r_tokens, c_tokens = totals
        native.require(n > 0 and r_tokens > 0, "Undefined paired cost ratio")
        ratio = c_tokens / r_tokens
        return {"reference_success_percent": 100 * r_success / n,
                "candidate_success_percent": 100 * c_success / n,
                "success_delta_pp": 100 * (c_success - r_success) / n,
                "total_token_ratio": ratio, "token_reduction_percent": 100 * (1 - ratio)}

    clusters = [groups[key] for key in sorted(groups)]
    point = metrics([sum(group[i] for group in clusters) for i in range(5)])
    native.require(type(replicates) is int and replicates > 0, "Invalid bootstrap replication count")
    draws = {key: [] for key in point}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = metrics([sum(group[i] for group in sampled) for i in range(5)])
        for key, value in values.items():
            draws[key].append(value)

    def percentile(values, fraction):
        ordered = sorted(values)
        location = (len(ordered) - 1) * fraction
        lo = int(location)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (location - lo)

    return {
        "tasks": len(ids), "gains": gains, "losses": losses, **point,
        "floorplan_cluster_95_intervals": {
            key: [percentile(values, 0.025), percentile(values, 0.975)]
            for key, values in draws.items()},
        "sampling_unit": "Floorplan; paired reference/candidate tasks kept together",
        "floorplan_clusters": len(clusters), "bootstrap_replicates": replicates,
        "bootstrap_seed": seed, "performance_guarantee_established": False,
    }


def validation_plan(protocol):
    stages = common.validation_plan(protocol)
    for stage in stages:
        for job in stage["jobs"]:
            job["method"] = METHOD
    return stages


def run(config_path: Path, lease_descriptors: list[int] | None = None):
    native.remote_only(socket.gethostname())
    config = native.read(config_path)
    native.require(config["authorized_hostname"] == socket.gethostname(), "Wrong authorized host")
    root = Path(config["run_directory"])
    native.require(root.is_absolute()
                   and root.parent == Path("/opt/phiagent/runs/chisel-efficiency-validation"),
                   "Use a new validation-only run identity")
    root.mkdir(parents=True, exist_ok=False)
    native.write(root / "STUDY_CONFIG.json", config)
    native.write(root / "PID.json", {"pid": os.getpid(), "hostname": socket.gethostname(),
                                   "argv": sys.argv, "at_utc": native.now()})
    process, descriptors, logfile = None, list(lease_descriptors or []), None
    try:
        worker, old_freeze, protocol, adapter = common.load_inputs(config, root)
        selected = validate_contract(config, worker, old_freeze, protocol)
        train_freeze = validate_selection(config, selected, protocol)
        native.require(Path(sys.prefix).resolve() == Path(worker["actor_python"]).parent.parent.resolve(),
                       "Use the frozen native actor Python")
        runtime = common.runtime_config(config, worker, root)
        runtime.update(schema_version=SCHEMA, method=METHOD, implementation_label=LABEL)
        # Keep IPC inside this new run and short enough for vLLM's UUID socket suffix.
        scratch = root / "ipc"
        native.require(len(str(scratch).encode()) + 37 <= 107, "Use a shorter new run identity for IPC")
        scratch.mkdir()
        tempfile.tempdir = str(scratch)
        os.environ.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(scratch), TEMP=str(scratch),
                          TMP=str(scratch), NO_PROXY="127.0.0.1,localhost",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          PYTHONDONTWRITEBYTECODE="1")
        sys.dont_write_bytecode = True
        sources = common.freeze_sources(config, old_freeze, root)
        for name in (TRAIN_SOURCE, str(Path(__file__).resolve().relative_to(ROOT))):
            sources[name] = native.file_hash(ROOT / name)
            destination = root / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        native.require(all(sources.get(name) == checksum
                           for name, checksum in train_freeze["source_hashes"].items()),
                       "Validation does not reuse the exact completed TRAIN implementation")
        stages = validation_plan(protocol)
        items = [job["item"] for stage in stages for job in stage["jobs"]]
        freeze = {
            "policies": {METHOD: selected}, "source_hashes": sources, "scope": "validation",
            "training_evidence": config["training_evidence"],
            "selected_from_train_only": True, "training_policy_sha256": selected["harness_sha256"],
            "partition_sha256": config["partition_sha256"], "skills_sha256": config["skills_sha256"],
            "validation_ids": [item["id"] for item in items], "test_access": False,
            "optimization_from_validation": False, "reference_reuse": config["reference"],
            "bootstrap_seed": config["bootstrap_seed"], "bootstrap_replicates": 10000,
            "performance_guarantee_established": False, "paper_admitted": False,
        }
        freeze["freeze_sha256"] = native.digest(freeze)
        native.write(root / "POLICY_FREEZE.json", freeze)
        reference_rows, reuse = load_reference(config, runtime, old_freeze, selected, items)
        native.write(root / "REFERENCE_REUSE.json", reuse)
        native.write(root / "config.json", runtime)
        native.write(root / "usage-ledger.json", common.new_ledger())
        native.write(root / "IPC_DIRECTORY.json", {"path": str(scratch), "inside_new_run": True})
        native.write(root / "PROVENANCE.json", {
            "source_hashes": sources, "hostname": socket.gethostname(), "argv": sys.argv,
            "source_git_base": config["source_git_base"], "seed": 9401,
            "git_head": core.repair.capture(["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT),
            "git_status": core.repair.capture(
                ["git", "--no-pager", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT),
            "actor_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
            "model_packages": core.repair.capture([runtime["model"]["python"], "-m", "pip", "freeze"]),
        })
        native.write(root / "PRELAUNCH_RESOURCE_AUDIT.json", native.resource_audit())
        common.verify_assets(runtime, protocol, adapter, items, root, data_scope="validation")
        manifest = {"schema_version": core.repair.SCHEMA, "budget": runtime["episode_budget"],
                    "alfworld_environment_mode": "eval_out_of_distribution", "seed": 20260928,
                    "task_inventory": items, "scope": "validation"}
        manifest["manifest_sha256"] = native.digest(manifest)
        native.write(root / "manifest.json", manifest)
        for directory in runtime["resource"]["lease_roots"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        inherited = descriptors if lease_descriptors is not None else None
        descriptors = []
        process, descriptors, logfile = core.start_service(root, runtime, lease_descriptors=inherited)
        endpoint = f"http://127.0.0.1:{runtime['resource']['port']}/v1"
        core.await_service(process, endpoint, runtime["startup_timeout_seconds"])
        binding = {"hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
                   "adapter_source_hashes": sources, "model": {
                       "id": runtime["model"]["served_name"], "revision": native.REVISION,
                       "endpoints": [endpoint]},
                   "data_root": runtime["data_root"], "skillopt_root": runtime["skillopt_root"],
                   "gpu": {"visible_devices": "", "service_selection": str(root / "GPU_SELECTION.json")},
                   "ledger": str(root / "usage-ledger.json"), "max_model_len": 32768}
        binding["binding_sha256"] = native.digest(binding)
        native.write(root / "binding.json", binding)
        rows = []
        for stage in stages:
            native.run_stage(root, runtime, freeze, stage,
                             worker_initializer=training.initialize_training_worker)
            batch = common.read_rows(root, stage["stage"])
            validate_rows(batch, [job["item"] for job in stage["jobs"]], METHOD,
                          selected["harness_sha256"])
            rows.extend(batch)
            if stage["stage"] == "validation_gate":
                native.write(root / "FIRST_EPISODE_GATE.json", {
                    "status": "PASS_EXACT_TRAIN_POLICY_NATIVE_AND_ACCOUNTING",
                    "success_required": False, "included_once": True,
                    "cost": training.audit_rows(root, rows, freeze["policies"])})
        cost = training.audit_rows(root, rows, freeze["policies"])
        validate_rows(rows, items, METHOD, selected["harness_sha256"])
        for name, checksum in sources.items():
            native.require(native.file_hash(ROOT / name) == checksum, "Source changed during validation")
        for name, checksum in config["reference"]["files"].items():
            native.require(native.file_hash(Path(config["reference"]["root"]) / name) == checksum,
                           "Frozen reference changed during validation")
        paired = paired_comparison(rows, reference_rows, items,
                                   replicates=config["bootstrap_replicates"],
                                   seed=config["bootstrap_seed"])
        native.write(root / "COST_AUDIT.json", cost)
        summary = native.summarize(rows)[METHOD]
        result = {
            "schema_version": SCHEMA, "status": "COMPLETE_FROZEN_POLICY_VALIDATION",
            "implementation_label": LABEL, "method": METHOD, "at_utc": native.now(),
            "freeze_sha256": freeze["freeze_sha256"], "policy_sha256": selected["harness_sha256"],
            "partition_sha256": config["partition_sha256"], "summary": summary, "cost": cost,
            "reference_reuse": reuse, "paired": paired,
            "component_token_ratios": {
                key: summary[key] / reuse["summary"][key] for key in common.USAGE_KEYS},
            "validation_point_gate": {
                "quality_nondecrease": paired["success_delta_pp"] >= 0,
                "at_least10_percent_token_reduction": paired["total_token_ratio"] <= 0.90},
            "new_reference_episodes": 0, "new_optimizer_calls": 0,
            "official_test_calls": 0, "test_access": False, "optimization_from_validation": False,
            "first_episode_counted_once": True, "failed_native_episodes_retained": True,
            "cumulative_token_mode": "unlimited", "performance_guarantee_established": False,
            "strongest_external_baselines_pending": True, "final_test_admission_pending": True,
            "paper_admitted": False,
        }
        result["result_sha256"] = native.digest(result)
        native.write(root / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "summary": summary, "paired": paired}), flush=True)
    except BaseException as error:
        ledger = root / "usage-ledger.json"
        native.write(root / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": native.now(),
            "ledger": native.read(ledger) if ledger.exists() else None,
            "failed_attempts_preserved": True, "unknown_usage_not_zero_imputed": True,
            "automatic_retries": 0, "new_reference_episodes": 0, "test_access": False,
            "paper_admitted": False})
        raise
    finally:
        common.stop_owned_service(process, descriptors, logfile)
        native.write(root / "RETIREMENT.json", {
            "at_utc": native.now(), "owned_service_pid": process.pid if process else None,
            "service_exit_code": process.returncode if process else None,
            "owned_lease_descriptors_closed": len(descriptors), "unrelated_processes_stopped": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lease-fds", type=int, nargs="*")
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}; preserve this attempt")

    signal.signal(signal.SIGTERM, interrupted)
    run(args.config, args.lease_fds)
