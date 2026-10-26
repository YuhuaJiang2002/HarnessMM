#!/usr/bin/env python3
"""Fresh matched ALFWorld controller removal and four frozen instruction policies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_chisel_interactive_shared226 as native  # noqa: E402
from scripts import run_chisel_instruction_only_validation as diagnostic  # noqa: E402

SCHEMA = "alfworld-controller-removal-fresh226/v1"
METHODS = ("chisel_skills", "chisel_controller", "no_skill", "skillopt", "autosaddler")
SOURCE_KEYS = ("parent", "no_stall_gate", "baseline_no_skill",
               "baseline_skillopt", "baseline_autosaddler")
COUNTS = {"train": 710, "validation": 710, "test": 2133}


def frozen_policies(archived: dict, protocol: dict) -> dict:
    policies = {name: deepcopy(archived["policies"][key])
                for name, key in zip(METHODS, SOURCE_KEYS, strict=True)}
    full = policies["chisel_controller"]
    native.require(full == protocol["policies"]["no_stall_gate"],
                   "The challenged controller must be rerun unchanged")
    native.require(set(full) == {"skills", "local_discriminator", "harness_sha256"},
                   "Unexpected controller fields")
    native.verify_signed(full, "harness_sha256")
    stripped = {key: value for key, value in full.items()
                if key not in ("local_discriminator", "harness_sha256")}
    stripped["harness_sha256"] = native.digest(stripped)
    native.require(policies["chisel_skills"] == stripped,
                   "Controller removal must preserve the exact instruction bytes")
    for name in ("chisel_skills", "no_skill", "skillopt", "autosaddler"):
        policy = policies[name]
        native.require(set(policy) <= {"skills", "harness_sha256", "arm", "label"}
                       and "skills" in policy, "An instruction arm contains another controller")
        native.require(set(policy["skills"]) == set(full["skills"]),
                       "Task-family instructions differ in coverage")
    native.require(all(value == "" for value in policies["no_skill"]["skills"].values()),
                   "No Skill must have empty instructions")
    return policies


def shard_items(protocol: dict, split: str, index: int, count: int) -> list[dict]:
    native.require(split in COUNTS and type(index) is int and type(count) is int
                   and 0 <= index < count, "Invalid shard or split")
    return native.ranked_items(protocol, split)[index::count]


def phase_plan(items: list[dict], name: str) -> dict:
    native.require(len({item["id"] for item in items}) == len(items), "Duplicate task identities")
    return {"stage": name, "jobs": [{"method": method, "item": item}
                                   for item in items for method in METHODS]}


def summarize_exact(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        measured = result.setdefault(row["method"], {
            "tasks": 0, "successes": 0, "calls": 0, "turns": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        })
        measured["tasks"] += 1
        measured["successes"] += row["official_success"]
        measured["calls"] += row["calls"]
        measured["turns"] += row["turns"]
        for key in diagnostic.USAGE_KEYS:
            measured[key] += row["usage"][key]
    for measured in result.values():
        measured["success_percent"] = 100 * measured["successes"] / measured["tasks"]
        measured["tokens_per_task"] = measured["total_tokens"] / measured["tasks"]
    return result


def validate_rows(rows: list[dict], items: list[dict]) -> None:
    expected = {(item["id"], method) for item in items for method in METHODS}
    native.require(len(rows) == len(expected)
                   and {(row["id"], row["method"]) for row in rows} == expected,
                   "Incomplete, duplicate or extra matched episodes")
    paths = {item["id"]: item["gamefile"] for item in items}
    for row in rows:
        native.require(row["gamefile"] == paths[row["id"]] and row["seed"] == 9401
                       and row["valid"] and row["cost_complete"], "Invalid episode identity")
        native.require(row["official_success"] in (0, 1) and 0 < row["turns"] <= 50,
                       "Invalid native episode outcome")
        native.require(row["calls"] == row["attempted_calls"], "Unaccounted model attempt")
        if row["method"] != "chisel_controller":
            native.require(row["calls"] == row["turns"],
                           "An instruction-only arm bypassed the common actor")
        usage = row["usage"]
        native.require(all(type(usage.get(key)) is int and usage[key] >= 0
                           for key in diagnostic.USAGE_KEYS)
                       and usage["total_tokens"] == usage["prompt_tokens"]
                       + usage["completion_tokens"], "Invalid provider accounting")


def verify_assets(config: dict, protocol: dict, items: list[dict]) -> None:
    parents = {str(Path(item["gamefile"]).parent) for item in items}
    selected = {name: digest for name, digest in protocol["data_hashes"].items()
                if str(Path(name).parent) in parents}
    native.require(all(item["gamefile"] in selected for item in items), "Missing task assets")
    for name, digest in selected.items():
        native.require(native.file_hash(Path(config["data_root"]) / name) == digest,
                       f"Frozen native task asset changed: {name}")


def replay_one(config: dict, row: dict, output: str) -> dict:
    from scripts.audit_alfworld_controller_forensics import replay_episode
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    previous = Path.cwd()
    try:
        os.chdir(path)
        record = replay_episode(config, {"episode": row["artifact_dir"]}, path)
    finally:
        os.chdir(previous)
    native.require(record["native_success"] == row["official_success"],
                   "Independent native replay changed the outcome")
    record["status"] = "PASS"
    native.write(path / "RESULT.json", record)
    return record


def execute_phase(run: Path, config: dict, freeze: dict, protocol: dict,
                  name: str, items: list[dict], previous_rows: list[dict]) -> list[dict]:
    verify_assets(config, protocol, items)
    native.run_stage(run, config, freeze, phase_plan(items, name))
    rows = diagnostic.read_rows(run, name)
    validate_rows(rows, items)
    audit_config = {
        "data_root": config["data_root"], "skillopt_root": config["skillopt_root"],
        "frozen_source": str(ROOT), "step_limit": 50,
    }
    audits = {}
    native.write(run / "PROGRESS.json", {
        "status": "INDEPENDENT_NATIVE_REPLAY", "stage": name,
        "completed": 0, "expected": len(rows), "at_utc": native.now(),
    })
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        pending = {}
        for row in rows:
            path = run / f"{name}-native-audit" / row["method"] / row["id"].replace(":", "_")
            pending[pool.submit(replay_one, audit_config, row, str(path))] = path
        for future in as_completed(pending):
            record = future.result()
            path = pending[future] / "RESULT.json"
            audits[str(path.relative_to(run))] = native.file_hash(path)
            native.require(record["status"] == "PASS", "Native replay failed")
            native.write(run / "PROGRESS.json", {
                "status": "INDEPENDENT_NATIVE_REPLAY", "stage": name,
                "completed": len(audits), "expected": len(rows), "at_utc": native.now(),
            })
    cost = diagnostic.audit_actor_cost(run, previous_rows + rows)
    native.write(run / f"{name}-AUDIT.json", {
        "status": "PASS_NATIVE_REPLAY_AND_ACCOUNTING", "stage": name,
        "freeze_sha256": freeze["freeze_sha256"], "tasks_per_method": len(items),
        "episodes": len(rows), "summary": summarize_exact(rows), "cumulative_cost": cost,
        "native_episode_audits": audits, "rows_sha256": native.file_hash(run / f"{name}-rows.jsonl"),
        "new_audit_model_calls": 0, "at_utc": native.now(),
    })
    return rows


def wait_release(path: Path, run: Path, freeze_sha: str, process, phase: str) -> None:
    while not path.exists():
        native.require(process.poll() is None, "Owned model service exited while waiting")
        native.write(run / "HEARTBEAT.json", {
            "status": "WAITING_GLOBAL_GATE", "phase": phase, "at_utc": native.now(),
            "successor_task_id": phase,
        })
        time.sleep(15)
    record = native.read(path)
    native.require(record["status"] == "PASS" and record["freeze_sha256"] == freeze_sha
                   and record["score_dependent_selection"] is False,
                   "Invalid global release; no outcome-dependent selection is permitted")


def reserve_device(config: dict, output: Path) -> list[int]:
    from scripts import run_alfworld_modular_rsi as core
    resource = config["resource"]
    while True:
        inventory = core.repair.capture([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"])
        processes = core.repair.capture([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
        native.require(inventory["returncode"] == processes["returncode"] == 0,
                       "GPU resource inspection failed")
        line = next(line for line in inventory["output"].splitlines()
                    if int(line.split(",")[0]) == resource["gpu_index"])
        native.require(resource["gpu_uuid"] in line, "Selected physical GPU identity changed")
        available = (int(line.split(",")[2].strip().split()[0]) < 100
                     and resource["gpu_uuid"] not in processes["output"])
        storage_ready = shutil.disk_usage(output).free >= 4 * 1024 ** 3
        if available and storage_ready:
            try:
                return core.acquire_service_leases(resource)
            except BlockingIOError:
                available = False
        native.write(output / "HEARTBEAT.json", {
            "status": "WAITING_RESOURCE_LEASE" if storage_ready else "WAITING_STORAGE",
            "at_utc": native.now(), "gpu": resource["gpu_index"],
            "inventory": inventory, "compute_processes": processes,
            "storage_free_bytes": shutil.disk_usage(output).free,
            "successor_task_id": "native-episode-operability-gate",
        })
        time.sleep(15)


def run(config_path: Path) -> None:
    config = native.read(config_path)
    native.remote_only(config["authorized_hostname"])
    native.require(config["schema_version"] == SCHEMA, "Unknown rerun contract")
    native.require(config["episode_budget"] == diagnostic.BUDGET,
                   "The common actor or native 50-step budget changed")
    native.require(config["model"]["repository"] == "Qwen/Qwen3.5-9B"
                   and config["model"]["revision"] == native.REVISION
                   and config["model"]["served_name"] == "qwen35-9b@main"
                   and config["model"]["max_model_len"] == 32768
                   and config["model"]["enable_thinking"] is False,
                   "The common frozen model configuration changed")
    native.require(config["cumulative_token_mode"] == "unlimited"
                   and all(config[key] is None for key in (
                       "maximum_total_provider_tokens", "maximum_request_attempts",
                       "maximum_run_seconds")), "Discretionary cumulative ceilings are forbidden")
    native.require(diagnostic.HOST_GPU_UUIDS[config["authorized_hostname"]][
        config["resource"]["gpu_index"]] == config["resource"]["gpu_uuid"],
        "Physical GPU identity differs from its authorized host")
    protocol = native.read(Path(config["partition_path"]))
    partition = native.verify_partition(protocol)
    native.require(partition["partition_sha256"] == config["partition_sha256"]
                   and partition["membership_sha256"] == config["membership_sha256"]
                   and partition["counts"] == COUNTS, "Frozen 20/20/60 membership changed")
    freeze = native.read(Path(config["freeze_path"]))
    native.verify_signed(freeze, "freeze_sha256")
    native.require(freeze["freeze_sha256"] == config["freeze_sha256"]
                   and tuple(freeze["methods"]) == METHODS, "Frozen method comparison changed")
    native.require(all(native.file_hash(ROOT / name) == checksum
                       for name, checksum in freeze["source_hashes"].items()),
                   "Private source snapshot changed")
    native.require(Path(sys.prefix).resolve()
                   == Path(config["actor_python"]).parent.parent.resolve(),
                   "Use the frozen actor Python")
    output = Path(config["run_directory"])
    output.mkdir(parents=True, exist_ok=False)
    native.write(output / "config.json", config)
    native.write(output / "POLICY_FREEZE.json", freeze)
    process, descriptors, logfile = None, [], None
    try:
        from scripts import run_alfworld_modular_rsi as core
        from scripts.alfworld_jev_runtime import bind_local_labels
        scratch = diagnostic.ipc_scratch(output)
        os.environ.update(CUDA_VISIBLE_DEVICES="", NO_PROXY="127.0.0.1,localhost",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                          TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
        sys.dont_write_bytecode = True
        native.write(output / "PROVENANCE.json", {
            "argv": sys.argv, "hostname": socket.gethostname(), "seed": 9401,
            "source_hashes": freeze["source_hashes"], "at_utc": native.now(),
            "config_sha256": native.file_hash(config_path),
            "source_git_state": config["source_git_state"],
            "actor_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
            "model_packages": core.repair.capture([config["model"]["python"], "-m", "pip", "freeze"]),
        })
        for name, checksum in protocol["checkpoint"]["files"].items():
            native.require(native.file_hash(Path(config["model"]["directory"]) / name) == checksum,
                           f"Pinned checkpoint changed: {name}")
        receipt = native.read(Path(config["native_adapter_path"]))
        native.require(len(receipt["files"]) == receipt["files_count"] == 182,
                       "Incomplete native adapter source receipt")
        for name, entry in receipt["files"].items():
            native.require(native.file_hash(Path(config["skillopt_root"]) / name) == entry["sha256"],
                           f"Native adapter changed: {name}")
        native.write(output / "CHECKPOINT.json", protocol["checkpoint"])
        native.write(output / "NATIVE_ADAPTER_SOURCE.json", receipt)
        native.write(output / "usage-ledger.json", diagnostic.new_ledger())
        owned = {split: shard_items(protocol, split, config["shard_index"], config["shard_count"])
                 for split in ("validation", "test")}
        train = native.gate_items(protocol) if config["shard_index"] == 0 else []
        manifest = {
            "schema_version": core.repair.SCHEMA, "budget": config["episode_budget"],
            "alfworld_environment_mode": "eval_out_of_distribution",
            "task_inventory": train + owned["validation"] + owned["test"],
            "protocol_sha256": config["partition_sha256"], "seed": 20260928,
            "exposure": "Previously exposed public TRAIN pool; fresh frozen-policy replication",
        }
        manifest["manifest_sha256"] = native.digest(manifest)
        native.write(output / "manifest.json", manifest)
        native.write(output / "PRELAUNCH_RESOURCE_AUDIT.json", native.resource_audit())
        for folder in config["resource"]["lease_roots"]:
            Path(folder).mkdir(parents=True, exist_ok=True)
        held = reserve_device(config, output)
        process, descriptors, logfile = core.start_service(output, config, lease_descriptors=held)
        endpoint = f"http://127.0.0.1:{config['resource']['port']}/v1"
        core.await_service(process, endpoint, config["startup_timeout_seconds"])
        binding = {
            "hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
            "adapter_source_hashes": freeze["source_hashes"],
            "model": {"id": config["model"]["served_name"], "revision": native.REVISION,
                      "endpoints": [endpoint]},
            "gpu": {"visible_devices": "", "service_selection": str(output / "GPU_SELECTION.json")},
            "skillopt_root": config["skillopt_root"], "data_root": config["data_root"],
            "ledger": str(output / "usage-ledger.json"), "max_model_len": 32768,
            "local_discriminator_logprobs_mode": "processed_logprobs", "typesafe_api_used": False,
        }
        binding["local_label_token_ids"] = bind_local_labels(binding, output / "LOCAL_TOKENIZER.json")
        binding["binding_sha256"] = native.digest(binding)
        native.write(output / "binding.json", binding)
        rows = []
        if train:
            rows.extend(execute_phase(output, config, freeze, protocol, "train_gate", train, rows))
            native.write(output / "TRAIN_READY.json", {
                "status": "PASS", "freeze_sha256": freeze["freeze_sha256"],
                "audit_sha256": native.file_hash(output / "train_gate-AUDIT.json"),
                "success_required": False, "at_utc": native.now(),
            })
        prepared = Path(config["release_directory"])
        wait_release(prepared / "VALIDATION_RELEASE.json", output, freeze["freeze_sha256"],
                     process, "validation")
        rows.extend(execute_phase(output, config, freeze, protocol, "validation",
                                  owned["validation"], rows))
        native.write(output / "VALIDATION_READY.json", {
            "status": "PASS", "freeze_sha256": freeze["freeze_sha256"],
            "tasks_per_method": len(owned["validation"]),
            "audit_sha256": native.file_hash(output / "validation-AUDIT.json"),
            "at_utc": native.now(),
        })
        wait_release(prepared / "TEST_RELEASE.json", output, freeze["freeze_sha256"],
                     process, "test")
        rows.extend(execute_phase(output, config, freeze, protocol, "test", owned["test"], rows))
        native.require(all(native.file_hash(ROOT / name) == checksum
                           for name, checksum in freeze["source_hashes"].items()),
                       "Source changed during execution")
        native.write(output / "RESULT.json", {
            "status": "COMPLETE_NATIVE_REPLAY_AND_ACCOUNTING", "at_utc": native.now(),
            "freeze_sha256": freeze["freeze_sha256"], "partition": partition,
            "shard_index": config["shard_index"], "shard_count": config["shard_count"],
            "validation": native.read(output / "validation-AUDIT.json"),
            "test": native.read(output / "test-AUDIT.json"),
            "cumulative_cost": diagnostic.audit_actor_cost(output, rows),
            "new_optimizer_calls": 0, "new_optimizer_run_claimed": False,
            "fresh_episodes": True, "historical_results_reused": False,
            "successor_task_id": "cross-node-aggregate-and-paper-integrity-audit",
        })
    except BaseException as error:
        native.write(output / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": native.now(),
            "failed_attempts_preserved": True, "automatic_retries": 0,
            "unknown_usage_not_zero_imputed": True,
        })
        raise
    finally:
        diagnostic.stop_owned_service(process, descriptors, logfile)
        native.write(output / "RETIREMENT.json", {
            "at_utc": native.now(), "service_pid": process.pid if process else None,
            "service_exit_code": process.returncode if process else None,
            "unrelated_processes_stopped": 0,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
