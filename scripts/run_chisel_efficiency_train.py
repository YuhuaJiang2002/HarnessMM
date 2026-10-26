#!/usr/bin/env python3
"""Training-only paired reasoning-cost study; every action still requires the common actor."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_chisel_instruction_only_validation as common  # noqa: E402
from scripts import run_chisel_interactive_shared226 as native  # noqa: E402
from scripts import run_alfworld_modular_rsi as core  # noqa: E402
from phiagent.learning.alfworld_modular_rsi import compact_action_contract, public_memory_lines  # noqa: E402
from scripts.alfworld_public_progress import UnsupportedPublicState  # noqa: E402

SCHEMA = "chisel-training-reasoning-cost/v1"
PROFILES = ("reference", "recurrence_memory", "memory_guarded64", "no_skill_memory")


def choose_brief(profile: str, public_history: list[dict]) -> tuple[bool, str]:
    if profile == "reference":
        return False, "matched_full_reasoning_reference"
    if profile in ("recurrence_memory", "no_skill_memory"):
        return False, "full_reasoning_with_public_memory"
    if profile != "memory_guarded64":
        raise ValueError("Unknown frozen efficiency profile")
    if not public_history:
        return False, "full_reasoning_initialization"
    if public_history[-1]["environment_command_error"]:
        return False, "full_reasoning_after_public_tool_error"
    current = (public_history[-1]["action"], public_history[-1]["observation"])
    if any((row["action"], row["observation"]) == current for row in public_history[-5:-1]):
        return False, "full_reasoning_after_public_recurrence"
    return True, "concise64_on_ordinary_public_step"


def memory_active(public_history: list[dict]) -> bool:
    pairs = [(row["action"], row["observation"]) for row in public_history]
    return (any(row["environment_command_error"] for row in public_history)
            or len(set(pairs)) < len(pairs))


def training_metered_call(binding: dict, **kwargs):
    if kwargs["stage"] != "actor":
        raise ValueError("The training study permits only the common actor")
    episode = kwargs["output"].parent.parent
    job = native.read(episode / "job.json")
    profile = job["policy"]["efficiency_profile"]
    source = episode / "conversation.json"
    history = native.read(source) if source.exists() else []
    public = [{key: row[key] for key in ("action", "observation", "environment_command_error")}
              for row in history]
    use_brief, reason = choose_brief(profile, public)
    original = deepcopy(kwargs["messages"])

    def count_prompt(text):
        return core.native_count(binding, messages=[original[0], {"role": "user", "content": text}])

    transformation = {"changed": False, "reason": reason, "additional_model_calls": 0}
    if use_brief:
        text, transformation = compact_action_contract(
            original[1]["content"], count_tokens=count_prompt, reasoning_word_limit=64)
        kwargs["messages"] = [original[0], {"role": "user", "content": text}]
        native.require(transformation["changed"], "Pinned native response contract was not found")
    memory = {"active": False, "status": "not_triggered", "lines": 0}
    if profile != "reference" and memory_active(public):
        initial = native.read(episode / "initial_observation.json")["observation"]
        try:
            lines = public_memory_lines(initial, public)
        except UnsupportedPublicState as error:
            memory = {"active": False, "status": "unsupported_public_wording",
                      "reason": str(error), "lines": 0}
        else:
            text = kwargs["messages"][1]["content"]
            text += ("\n\nObserved public memory (partial; missing information is unknown; "
                     "choose the action yourself):\n" + "\n".join(lines))
            kwargs["messages"] = [original[0], {"role": "user", "content": text}]
            memory = {"active": True, "status": "public_facts_only", "lines": len(lines)}
    native.write(episode / "efficiency-trace" / f"{kwargs['output'].name}.json", {
        "profile": profile, "reasoning_route": reason, "brief_requested": use_brief,
        "transformation": transformation, "original_messages": original,
        "transformed_messages": kwargs["messages"], "public_history_steps": len(public),
        "public_memory": memory,
        "scoring_fields_accessed": False, "task_action_selected_in_code": False,
    })
    return core.metered_call(binding, **kwargs)


def initialize_training_worker():
    native.initialize_worker()
    core.repair.provider_call = training_metered_call


def training_items(protocol: dict, per_family: int, selection_seed: int) -> list[dict]:
    native.verify_partition(protocol)
    native.require(type(per_family) is int and per_family > 0, "Invalid training probe size")
    selected = []
    for family in core.TASK_FAMILIES:
        rows = [item for item in protocol["splits"]["train"] if item["task_type"] == family]
        native.require(len(rows) >= per_family, "Insufficient training tasks in a family")
        selected.extend(sorted(rows, key=lambda item: native.digest(
            ["efficiency-train", selection_seed, item["id"]]))[:per_family])
    return sorted(selected, key=lambda item: native.digest(["paired-order", selection_seed, item["id"]]))


def policies_from(parent: dict) -> dict:
    native.require(set(parent) == {"skills", "harness_sha256"}, "No task solver may enter this study")
    policies = {}
    for profile in PROFILES:
        skills = ({key: "" for key in parent["skills"]} if profile == "no_skill_memory"
                  else deepcopy(parent["skills"]))
        policy = {"skills": skills, "efficiency_profile": profile}
        policy["harness_sha256"] = native.digest(policy)
        policies[profile] = policy
    return policies


def validate_contract(config: dict, worker: dict, freeze: dict, protocol: dict) -> dict:
    native.require(config["schema_version"] == SCHEMA and config["scope"] == "train",
                   "This study may use training data only")
    native.require(config["test_access"] is False and config["validation_access"] is False,
                   "Validation and test task access are forbidden")
    native.require(config["profiles"] == list(PROFILES), "Frozen candidate set changed")
    native.require(config["reasoning_word_limit"] == 64, "Reasoning profile changed after freezing")
    native.require(config["minimum_token_reduction"] == 0.10 and config["quality_loss_allowance"] == 0,
                   "The declared quality/cost gate changed")
    native.require(0 <= config["shard_index"] < config["shard_count"], "Invalid task shard")
    return common.validate_frozen_instruction_contract(config, worker, freeze, protocol)


def train_gate(rows: list[dict], items: list[dict], minimum_saving: float) -> dict:
    for row in rows:
        native.require(row["valid"] is True and row["cost_complete"] is True,
                       "Invalid or unaccounted training episode")
        usage = row["usage"]
        native.require(type(row["official_success"]) is int and row["official_success"] in (0, 1)
                       and all(type(usage[key]) is int and usage[key] >= 0 for key in common.USAGE_KEYS)
                       and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
                       "Native score or complete provider usage is invalid")
    grouped = {profile: {row["id"]: row for row in rows if row["method"] == profile}
               for profile in PROFILES}
    expected = {item["id"] for item in items}
    native.require(len(rows) == len(expected) * len(PROFILES)
                   and all(set(value) == expected for value in grouped.values()),
                   "Only the complete matched training population may be compared")
    parent = grouped["reference"]
    summaries = native.summarize(rows)
    candidates = {}
    for profile in ("recurrence_memory", "memory_guarded64"):
        candidate = grouped[profile]
        floor = max(summaries["reference"]["successes"], summaries["no_skill_memory"]["successes"])
        quality = summaries[profile]["successes"] >= floor
        ratio = summaries[profile]["total_tokens"] / summaries["reference"]["total_tokens"]
        candidates[profile] = {
            "training_quality_not_lower": quality, "total_token_ratio": ratio,
            "quality_floor_successes": floor,
            "quality_floor_sources": ["reference", "no_skill_memory"],
            "training_cost_gate": ratio <= 1 - minimum_saving,
            "eligible_for_frozen_validation": quality and ratio <= 1 - minimum_saving,
            "gains": sum(candidate[key]["official_success"] > parent[key]["official_success"] for key in expected),
            "losses": sum(candidate[key]["official_success"] < parent[key]["official_success"] for key in expected),
            "performance_guarantee_established": False, "paper_admitted": False,
        }
    return {"summary": summaries, "candidates": candidates, "scope": "training_only",
            "validation_or_test_result": False}


def audit_rows(run: Path, rows: list[dict], policies: dict) -> dict:
    for row in rows:
        root = Path(row["artifact_dir"])
        native.require(native.read(root / "job.json")["policy"] == policies[row["method"]],
                       "Executed candidate policy changed")
        native.require(row["calls"] == row["attempted_calls"] == row["turns"],
                       "Every action must have exactly one accounted actor call")
        native.require(len(list((root / "efficiency-trace").glob("*.json"))) == row["turns"],
                       "Incomplete response-contract trace")
    return common.audit_actor_cost(run, rows)


def run(config_path: Path, lease_descriptors: list[int] | None = None):
    native.remote_only(socket.gethostname())
    config = native.read(config_path)
    native.require(config["authorized_hostname"] == socket.gethostname(), "Configured host differs")
    run_dir = Path(config["run_directory"])
    native.require(run_dir.is_absolute()
                   and run_dir.parent.parent == Path("/opt/phiagent/runs/chisel-efficiency-train")
                   and run_dir.name == f"shard-{config['shard_index']:02d}",
                   "Use a new training-study directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    native.write(run_dir / "STUDY_CONFIG.json", config)
    native.write(run_dir / "PID.json", {"pid": os.getpid(), "hostname": socket.gethostname(),
                                      "argv": sys.argv, "at_utc": native.now()})
    process, descriptors, logfile = None, list(lease_descriptors or []), None
    try:
        worker, old_freeze, protocol, adapter = common.load_inputs(config, run_dir)
        parent = validate_contract(config, worker, old_freeze, protocol)
        native.require(Path(sys.prefix).resolve() == Path(worker["actor_python"]).parent.parent.resolve(),
                       "Use the frozen native actor Python")
        selected = training_items(protocol, config["tasks_per_family"], config["selection_seed"])
        items = selected[config["shard_index"]::config["shard_count"]]
        native.require(bool(items), "Empty training shard")
        policies = policies_from(parent)
        runtime = common.runtime_config(config, worker, run_dir)
        runtime.update(schema_version=SCHEMA, method="chisel_efficiency_train", scope="train",
                       implementation_label="Training-only model-based response-contract study")
        scratch = common.ipc_scratch(run_dir)
        os.environ.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(scratch), TEMP=str(scratch),
                          TMP=str(scratch), NO_PROXY="127.0.0.1,localhost",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          PYTHONDONTWRITEBYTECODE="1")
        sources = common.freeze_sources(config, old_freeze, run_dir)
        name = str(Path(__file__).resolve().relative_to(ROOT))
        sources[name] = native.file_hash(ROOT / name)
        (run_dir / "source" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, run_dir / "source" / name)
        native.write(run_dir / "config.json", runtime)
        native.write(run_dir / "usage-ledger.json", common.new_ledger())
        native.write(run_dir / "PROVENANCE.json", {
            "source_hashes": sources, "hostname": socket.gethostname(), "argv": sys.argv,
            "source_git_base": config["source_git_base"], "seed": 9401,
            "actor_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
            "model_packages": core.repair.capture([runtime["model"]["python"], "-m", "pip", "freeze"]),
        })
        common.verify_assets(runtime, protocol, adapter, items, run_dir, data_scope="train")
        freeze = {"policies": policies, "scope": "train", "selection_seed": config["selection_seed"],
                  "all_probe_ids": [item["id"] for item in selected],
                  "shard_ids": [item["id"] for item in items], "shard_index": config["shard_index"],
                  "shard_count": config["shard_count"], "partition_sha256": config["partition_sha256"],
                  "source_hashes": sources, "test_access": False, "validation_access": False,
                  "candidate_origin": "Historical TRAIN-only recurrence-memory gains; retain full reasoning or use a guarded64-word contract, not the failed12-word contract",
                  "task_solver_in_code": False, "paper_admitted": False}
        freeze["freeze_sha256"] = native.digest(freeze)
        native.write(run_dir / "POLICY_FREEZE.json", freeze)
        manifest = {"schema_version": core.repair.SCHEMA, "budget": runtime["episode_budget"],
                    "alfworld_environment_mode": "eval_out_of_distribution", "seed": 20260928,
                    "task_inventory": items, "scope": "train"}
        manifest["manifest_sha256"] = native.digest(manifest)
        native.write(run_dir / "manifest.json", manifest)
        for directory in runtime["resource"]["lease_roots"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        inherited = descriptors if lease_descriptors is not None else None
        descriptors = []
        process, descriptors, logfile = core.start_service(run_dir, runtime, lease_descriptors=inherited)
        endpoint = f"http://127.0.0.1:{runtime['resource']['port']}/v1"
        core.await_service(process, endpoint, runtime["startup_timeout_seconds"])
        binding = {"hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
                   "adapter_source_hashes": sources, "model": {
                       "id": runtime["model"]["served_name"], "revision": native.REVISION, "endpoints": [endpoint]},
                   "data_root": runtime["data_root"], "skillopt_root": runtime["skillopt_root"],
                   "gpu": {"visible_devices": "", "service_selection": str(run_dir / "GPU_SELECTION.json")},
                   "ledger": str(run_dir / "usage-ledger.json"), "max_model_len": 32768}
        binding["binding_sha256"] = native.digest(binding)
        native.write(run_dir / "binding.json", binding)
        plans = (
            {"stage": "train_gate", "jobs": [{"item": items[0], "method": profile} for profile in PROFILES]},
            {"stage": "train_probe", "jobs": [
                {"item": item, "method": profile}
                for item in items[1:]
                for profile in sorted(PROFILES, key=lambda name: native.digest(
                    ["counterbalance", config["selection_seed"], item["id"], name]))
            ]},
        )
        rows = []
        for stage in plans:
            native.run_stage(run_dir, runtime, freeze, stage, worker_initializer=initialize_training_worker)
            rows.extend(common.read_rows(run_dir, stage["stage"]))
            if stage["stage"] == "train_gate":
                cost = audit_rows(run_dir, rows, policies)
                native.write(run_dir / "FIRST_PAIRED_GATE.json", {
                    "status": "PASS_NATIVE_AND_ACCOUNTING", "tasks": 1, "conditions": len(PROFILES),
                    "success_required": False, "included_once": True, "cost": cost})
        cost = audit_rows(run_dir, rows, policies)
        result = train_gate(rows, items, config["minimum_token_reduction"])
        result.update(status="COMPLETE_TRAIN_SHARD", shard_index=config["shard_index"],
                      shard_count=config["shard_count"], tasks=len(items), episodes=len(rows),
                      cost=cost, freeze_sha256=freeze["freeze_sha256"], paper_admitted=False)
        native.write(run_dir / "RESULT.json", result)
    except BaseException as error:
        native.write(run_dir / "FAILURE.json", {
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "ledger": native.read(run_dir / "usage-ledger.json")
            if (run_dir / "usage-ledger.json").exists() else None,
            "automatic_retries": 0, "validation_or_test_result": False, "paper_admitted": False})
        raise
    finally:
        common.stop_owned_service(process, descriptors, logfile)
        native.write(run_dir / "RETIREMENT.json", {
            "owned_service_pid": process.pid if process else None, "unrelated_processes_stopped": 0,
            "at_utc": native.now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lease-fds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}; preserve this attempt")

    signal.signal(signal.SIGTERM, interrupted)
    run(args.config.resolve(), args.lease_fds)
