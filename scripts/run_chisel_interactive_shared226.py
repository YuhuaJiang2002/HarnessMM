#!/usr/bin/env python3
"""Native, remote-only CHISEL shared-controller validation on the frozen TRAIN pool.

The original partition, native simulator, action projection and 50-step cap are
unchanged. Optimizer feedback is TRAIN-only. A missing complete optimizer blocks
the four-method test release, never becomes a prompt-checkpoint baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "chisel-interactive-shared226/v1"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAIN_METHODS = ("no_skill", "skillopt", "autosaddler", "chisel_shared_support_first")
NATIVE_METHODS = ("no_skill", "chisel_shared_support_first")
DIAGNOSTIC = "legacy_stall_diagnostic"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".pending-{os.getpid()}")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    pending.replace(path)


def append(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def remote_only(hostname: str | None = None) -> None:
    require(sys.platform == "linux", "Benchmark execution is authorized-remote-only")
    require(socket.gethostname() in (
        "harnessmm-worker-1", "harnessmm-worker-2",
    ), "Host is not an authorized H200 server")
    if hostname:
        require(socket.gethostname() == hostname, "Wrong assigned host")


def verify_signed(value: dict, field: str) -> None:
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}),
            f"Invalid {field}")


def scenario(item: dict) -> str:
    return Path(item["gamefile"]).parent.parent.as_posix()


def verify_partition(protocol: dict, *, expected_total: int = 3553) -> dict:
    verify_signed(protocol, "protocol_sha256")
    require(protocol["seed"] == 20260928 and protocol["split_ratio"] == [2, 2, 6],
            "Frozen partition seed/ratio differs")
    require(protocol["source_population"] == "complete_official_train_inventory",
            "Must retain the complete official TRAIN source pool")
    require(protocol["step_limit"] == 50, "Native 50-step cap changed")
    splits = protocol["splits"]
    require(set(splits) == {"train", "validation", "test"}, "Missing partition")
    target = expected_total // 5
    require([len(splits[k]) for k in ("train", "validation", "test")] ==
            [target, target, expected_total - 2 * target], "Frozen population counts differ")
    all_items = [item for items in splits.values() for item in items]
    require(len({item["id"] for item in all_items}) == expected_total, "Task IDs overlap")
    require(len({item["gamefile"] for item in all_items}) == expected_total, "Paths overlap")
    require(all(item["gamefile"].startswith("json_2.1.1/train/") for item in all_items),
            "An official evaluation split entered this custom TRAIN-pool split")
    groups = {name: {scenario(item) for item in items} for name, items in splits.items()}
    require(not groups["train"] & groups["validation"]
            and not (groups["train"] | groups["validation"]) & groups["test"],
            "Goal/scene trial variants leaked across partitions")
    require(set(protocol["protected_gamefiles"]) <=
            {item["gamefile"] for item in splits["train"]},
            "Known predecessor exposure escaped TRAIN")
    return {
        "status": "PASS", "partition_sha256": protocol["protocol_sha256"],
        "membership_sha256": digest(splits), "population": expected_total,
        "counts": {k: len(v) for k, v in splits.items()},
        "groups": {k: len(v) for k, v in groups.items()},
        "protected_predecessor_gamefiles": len(protocol["protected_gamefiles"]),
        "label": "custom 20/20/60 test from the complete official TRAIN pool",
        "official_valid_unseen": False,
        "prior_exposure": "Historical public results and the predecessor split were opened; "
                          "known development goal/scene groups remain in TRAIN.",
    }


def policies_from_partition(protocol: dict) -> dict:
    shared = deepcopy(protocol["policies"]["no_stall_gate"])
    legacy = deepcopy(protocol["policies"]["no_periodic_refresh"])
    for policy in (shared, legacy):
        verify_signed(policy, "harness_sha256")
        require(policy["local_discriminator"]["deterministic_action_selection"] is True,
                "This track validates the supported deterministic local controller")
        require(policy["local_discriminator"]["model_revision"] == REVISION,
                "Historical policy backbone differs")
    require(shared["local_discriminator"]["periodic_cot_refresh"] is False,
            "Support-first must not restore periodic refresh")
    no_skill = {"skills": {family: "" for family in shared["skills"]}}
    no_skill["harness_sha256"] = digest(no_skill)
    return {"no_skill": no_skill, "chisel_shared_support_first": shared, DIAGNOSTIC: legacy}


def ranked_items(protocol: dict, split: str) -> list[dict]:
    return sorted(protocol["splits"][split],
                  key=lambda item: digest([20260928, "shared226-order", item["id"]]))


def gate_items(protocol: dict) -> list[dict]:
    by_family = {}
    for item in ranked_items(protocol, "train"):
        by_family.setdefault(item["task_type"], item)
    return [by_family[family] for family in sorted(by_family)]


def diagnostic_items(protocol: dict, count: int) -> list[dict]:
    gate = gate_items(protocol)
    selected = {item["id"] for item in gate}
    return (gate + [item for item in ranked_items(protocol, "train")
                    if item["id"] not in selected])[:count]


def plan_tasks(protocol: dict, index: int, total: int, diagnostic_count: int) -> list[dict]:
    require(0 <= index < total and total > 0, "Invalid shard")
    gate = gate_items(protocol)
    gate_ids = {item["id"] for item in gate}
    diagnostic = diagnostic_items(protocol, diagnostic_count)
    stages = [
        ("native_gate", gate, (*NATIVE_METHODS, DIAGNOSTIC)),
        ("train", [item for item in ranked_items(protocol, "train")
                   if item["id"] not in gate_ids], NATIVE_METHODS),
        ("diagnostic_train", [item for item in diagnostic if item["id"] not in gate_ids],
         (DIAGNOSTIC,)),
        ("validation", ranked_items(protocol, "validation"), NATIVE_METHODS),
    ]
    result = []
    for stage, items, methods in stages:
        jobs = [{"stage": stage, "item": item, "method": method}
                for item in items[index::total] for method in methods]
        result.append({"stage": stage, "jobs": jobs})
    return result


def resource_audit() -> dict:
    def capture(command):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30)
        return {"returncode": result.returncode, "output": result.stdout}
    return {
        "at_utc": now(), "hostname": socket.gethostname(),
        "gpus": capture(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader"]),
        "processes": capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                              "--format=csv,noheader"]),
        "owners": capture(["ps", "-eo", "user,pid,ppid,etimes,comm"]),
        "leases": capture(["lslocks", "-n", "-o", "PID,COMMAND,PATH"]),
        "memory": capture(["free", "-m"]), "disk": capture(["df", "-h", "/opt/phiagent"]),
    }


def prepare(config_path: Path) -> None:
    config = read(config_path)
    remote_only(config["preparation_hostname"])
    source = Path(config["predecessor_prepared"])
    protocol = read(source / "PROTOCOL.json")
    partition = verify_partition(protocol)
    require(protocol["protocol_sha256"] == config["predecessor_protocol_sha256"],
            "A different partition was substituted")
    require(protocol["checkpoint"]["revision"] == REVISION, "Checkpoint revision differs")
    from scripts import run_alfworld_modular_rsi as core
    from phiagent.learning import supported_local_decision

    require(hasattr(supported_local_decision, "SupportedCandidates")
            and hasattr(supported_local_decision, "execute_supported_decision"),
            "The shared supported-local-decision API is not ready")
    output = Path(config["prepared_directory"])
    output.mkdir(parents=True, exist_ok=False)
    names = tuple(dict.fromkeys((
        *core.SOURCE_FILES, "phiagent/learning/supported_local_decision.py",
        "scripts/run_chisel_interactive_shared226.py",
    )))
    hashes = {name: file_hash(ROOT / name) for name in names}
    methods = {
        "no_skill": {"identity": "No Skill; native CoT actor; no local controller",
                     "optimizer": "none", "train_population": 710},
        "chisel_shared_support_first": {
            "identity": "CHISEL shared supported-local-decision controller; support-first no_stall_gate",
            "candidate_origin": "Frozen predecessor TRAIN-derived family skills",
            "train_population": 710, "new_optimizer_calls": 0,
            "candidate_selection": "One predeclared train-derived candidate; no validation feedback",
            "historical_freeze_sha256": protocol["historical_freeze_sha256"],
        },
        "skillopt": {
            "identity": "Pinned complete official SkillOpt + explicit native ALFWorld adapter",
            "revision": config["optimizers"]["skillopt"]["revision"],
            "status": "BLOCKED_NEW_TRAIN_ONLY_COMPLETE_OPTIMIZER_BINDING",
            "reason": "Existing complete optimizer preparation is bound to a different "
                      "TRAIN24/DEV36 and Qwen3.8 protocol; no old prompt artifact is substituted.",
            "train_population": 710, "cost": None,
        },
        "autosaddler": {
            "identity": "Pinned complete official AutoSaddler V2 + explicit native ALFWorld adapter",
            "revision": config["optimizers"]["autosaddler"]["revision"],
            "status": "BLOCKED_NEW_TRAIN_ONLY_COMPLETE_OPTIMIZER_BINDING",
            "reason": "A new native evaluator binding and TRAIN-only internal development "
                      "feedback must be verified before this complete optimizer can run.",
            "train_population": 710, "cost": None,
        },
        DIAGNOSTIC: {
            "identity": "Shared-core legacy stall scheduling diagnostic, not a main comparator",
            "only_changed_setting": "Recovery waits for public stall instead of support-first",
            "training_cases": config["diagnostic_train_count"], "test_admitted": False,
        },
    }
    freeze = {
        "schema_version": SCHEMA, "created_at": now(), "partition": partition,
        "source_hashes": hashes, "checkpoint": protocol["checkpoint"],
        "policies": policies_from_partition(protocol), "methods": methods,
        "model_revision": REVISION, "actor_seed": 9401, "step_limit": 50,
        "optimizer_feedback": "TRAIN only; validation per-case feedback is not an optimizer input",
        "test_release": "All four complete methods, complete external validation, "
                        "frozen source/policies and accounted provider costs required",
        "positive_innovation_gate": "CHISEL official success strictly exceeds every main comparator",
        "cumulative_token_mode": "unlimited", "maximum_total_provider_tokens": None,
        "legacy_parity": "Remote archived public-trace replay only; regression, not benchmark gain",
        "source_provenance": read(ROOT / "SOURCE_PROVENANCE.json"),
    }
    freeze["freeze_sha256"] = digest(freeze)
    write(output / "FROZEN.json", freeze)
    write(output / "PARTITION.json", protocol)
    write(output / "PARTITION_AUDIT.json", partition)
    write(output / "METHODS.json", methods)
    old_config = read(source / "worker-00.json")
    workers = []
    for index, node in enumerate(config["nodes"]):
        worker = deepcopy(old_config)
        worker.update(
            schema_version=SCHEMA, authorized_hostname=node["hostname"],
            run_directory=f"{config['run_root']}/shard-{index:02d}",
            prepared_directory=str(output), freeze_sha256=freeze["freeze_sha256"],
            shard_index=index, shard_count=len(config["nodes"]),
            workers=config["workers_per_gpu"], maximum_total_provider_tokens=None,
            cumulative_token_mode="unlimited", maximum_request_attempts=None,
            maximum_run_seconds=None, diagnostic_train_count=config["diagnostic_train_count"],
            actor_python=node.get("actor_python", old_config["actor_python"]),
            metadata_address=node["address"], metadata_port=config["metadata_port_base"] + index,
            peers=[{"address": entry["address"], "port": config["metadata_port_base"] + rank}
                   for rank, entry in enumerate(config["nodes"])],
        )
        worker["model"].update(directory=node["model_directory"], python=node["model_python"])
        worker["resource"].update(
            host_alias=node["address"], gpu_index=node["gpu_index"],
            gpu_uuid=node["gpu_uuid"], port=config["port_base"] + index,
        )
        worker["episode_budget"].update(max_steps=50, seed=9401, temperature=0, provider_retries=0)
        worker["module_variants"] = ["cot_local"]
        path = output / f"worker-{index:02d}.json"
        write(path, worker)
        workers.append({"index": index, "host": node["address"], "hostname": node["hostname"],
                        "config": str(path), "run": worker["run_directory"],
                        "python": worker["actor_python"], "source": str(ROOT),
                        "metadata_port": worker["metadata_port"]})
    write(output / "QUEUE.json", {
        "updated_at": now(),
        "ready": ["native_gate", "train", "diagnostic_train"],
        "dependent": [
            {"stage": "validation", "depends_on": "Both complete native TRAIN stages"},
            {"stage": "skillopt_complete_optimization", "status": methods["skillopt"]["status"]},
            {"stage": "autosaddler_complete_optimization", "status": methods["autosaddler"]["status"]},
            {"stage": "four_method_test", "depends_on": "All four frozen complete method candidates"},
        ],
        "source_changes_require_new_run": True,
    })
    write(output / "PREPARED.json", {"config": config, "workers": workers,
                                    "freeze_sha256": freeze["freeze_sha256"]})
    print(json.dumps({"status": "PREPARED", "partition": partition,
                      "freeze_sha256": freeze["freeze_sha256"]}), flush=True)


def reserve_unlimited(state: dict, amount: int, *, current_time: float) -> None:
    require(type(amount) is int and amount > 0, "Invalid native request reservation")
    require(state["maximum_tokens"] is None and state["cumulative_token_mode"] == "unlimited",
            "Discretionary cumulative-token termination must be disabled")
    require(not state["cost_unknown"], "An unaccounted provider failure requires diagnosis")
    state["reserved_tokens"] += amount
    state["attempts"] += 1


def initialize_worker() -> None:
    from scripts import run_alfworld_modular_rsi as core
    core.reserve_request = reserve_unlimited
    core.initialize_worker()


def validate_row(row: dict, item: dict, method: str, *, shared_control: bool | None = None,
                 episode_seed: int = 9401) -> None:
    require(row["id"] == item["id"] and row["gamefile"] == item["gamefile"],
            "Native task identity changed")
    require(type(episode_seed) is int and 0 <= episode_seed < 2**32
            and row["method"] == method and row["seed"] == episode_seed, "Method/seed changed")
    require(row["valid"] and row["cost_complete"], f"Invalid native episode: {row}")
    require(row["official_success"] in (0, 1) and 0 < row["turns"] <= 50,
            "Invalid native terminal or turn count")
    usage = row["usage"]
    require(all(type(usage.get(k)) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"],
            "Incomplete provider cost")
    conversation = read(Path(row["artifact_dir"]) / "conversation.json")
    require(len(conversation) == row["turns"], "Incomplete native trace")
    terminal = conversation[-1]
    require(row["official_success"] == int(bool(terminal["done"]) and bool(terminal["official_won"])),
            "Score differs from native won AND done")
    if shared_control is None:
        shared_control = method != "no_skill"
    if shared_control:
        decisions = [read(path) for path in sorted(
            (Path(row["artifact_dir"]) / "decisions").glob("*.json"))]
        require(len(decisions) == row["turns"], "Shared controller trace is incomplete")
        for decision in decisions:
            trace = decision["supported_local_decision"]
            if trace["route"] == "commit":
                require(trace["committed_action"] in decision["native_actions"]
                        and decision["route"] == "deterministic",
                        "Shared commitment escaped native supported actions")
            else:
                require(decision["route"] == "cot" and decision["cot_calls"] == 1,
                        "Shared fallback did not invoke the native CoT actor exactly once")


def summarize(records: list[dict]) -> dict:
    by_method = {}
    for row in records:
        target = by_method.setdefault(row["method"], {
            "tasks": 0, "successes": 0, "calls": 0, "total_tokens": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "turns": 0,
        })
        target["tasks"] += 1
        target["successes"] += row["official_success"]
        target["calls"] += row["calls"]
        target["turns"] += row["turns"]
        for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
            target[key] += row["usage"][key]
    for value in by_method.values():
        value["native_success_rate"] = value["successes"] / value["tasks"]
    return by_method


def run_stage(run: Path, config: dict, freeze: dict, stage: dict, *,
              worker_initializer=initialize_worker, episode_seed: int = 9401) -> dict:
    from scripts import run_alfworld_modular_rsi as core
    require(type(episode_seed) is int and 0 <= episode_seed < 2**32, "Invalid frozen episode seed")
    records = []
    name = stage["stage"]
    jobs = [{
        "study": str(run), "phase": name, "arm": entry["method"], "item": entry["item"],
        "policy": freeze["policies"][entry["method"]], "endpoint_index": 0,
        "verify_native_interface": True,
        "alfworld_modular_rsi": {"max_steps": 50, "episode_seed": episode_seed},
    } for entry in stage["jobs"]]
    write(run / f"{name}-PLAN.json", jobs)
    write(run / "PROGRESS.json", {"status": "STARTING_NATIVE", "stage": name,
                                  "completed": 0, "expected": len(jobs), "at_utc": now()})
    by_key = {(job["item"]["id"], job["arm"]): job for job in jobs}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=worker_initializer) as pool:
        pending = {pool.submit(core.repair.run_episode, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            row = future.result()
            append(run / f"{name}-rows.jsonl", row)
            validate_row(row, job["item"], job["arm"],
                         shared_control="local_discriminator" in job["policy"], episode_seed=episode_seed)
            records.append(row)
            write(run / "PROGRESS.json", {
                "status": "RUNNING", "stage": name, "completed": len(records),
                "expected": len(jobs), "at_utc": now(), "summary": summarize(records),
                "usage": core.ledger_snapshot(run / "usage-ledger.json"),
            })
            if not (run / "FIRST_EPISODE.json").exists():
                write(run / "FIRST_EPISODE.json", {
                    "status": "PASS_NATIVE_TERMINAL", "at_utc": now(), "row": row,
                    "native_terminal": read(Path(row["artifact_dir"]) / "conversation.json")[-1],
                })
    require({(row["id"], row["method"]) for row in records} == set(by_key)
            and len(records) == len(by_key), "Incomplete or duplicate stage population")
    result = {"status": "PASS", "stage": name, "completed": len(records),
              "summary": summarize(records), "elapsed_seconds": time.monotonic() - started,
              "rows_sha256": file_hash(run / f"{name}-rows.jsonl") if records else None,
              "at_utc": now()}
    write(run / f"{name}-RESULT.json", result)
    return result


def worker(config_path: Path) -> None:
    config = read(config_path)
    remote_only(config["authorized_hostname"])
    require(config["maximum_total_provider_tokens"] is None
            and config["cumulative_token_mode"] == "unlimited", "Token ceiling is forbidden")
    prepared = Path(config["prepared_directory"])
    freeze = read(prepared / "FROZEN.json")
    verify_signed(freeze, "freeze_sha256")
    require(freeze["freeze_sha256"] == config["freeze_sha256"], "Wrong frozen source")
    for name, checksum in freeze["source_hashes"].items():
        require(file_hash(ROOT / name) == checksum, f"Source changed: {name}")
    protocol = read(prepared / "PARTITION.json")
    verify_partition(protocol)
    require(protocol["protocol_sha256"] == freeze["partition"]["partition_sha256"],
            "Partition no longer matches source freeze")
    run = Path(config["run_directory"])
    run.mkdir(parents=True, exist_ok=False)
    write(run / "config.json", config)
    write(run / "FROZEN.json", freeze)
    write(run / "PID.json", {"pid": os.getpid(), "hostname": socket.gethostname(), "at_utc": now()})
    os.environ.update(CUDA_VISIBLE_DEVICES="", NO_PROXY="127.0.0.1,localhost",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from scripts import run_alfworld_modular_rsi as core
    from scripts.alfworld_jev_runtime import bind_local_labels
    r = core.repair
    write(run / "PROVENANCE.json", {
        "argv": sys.argv, "hostname": socket.gethostname(), "at_utc": now(),
        "seed": 9401, "source_hashes": freeze["source_hashes"],
        "source_provenance": freeze["source_provenance"],
        "actor_packages": r.capture([sys.executable, "-m", "pip", "freeze"]),
        "model_packages": r.capture([config["model"]["python"], "-m", "pip", "freeze"]),
    })
    process, descriptors, logfile = None, [], None
    try:
        write(run / "PRELAUNCH_RESOURCE_AUDIT.json", resource_audit())
        for name, checksum in freeze["checkpoint"]["files"].items():
            require(file_hash(Path(config["model"]["directory"]) / name) == checksum,
                    f"Exact checkpoint bytes differ: {name}")
        write(run / "CHECKPOINT.json", freeze["checkpoint"])
        stages = plan_tasks(protocol, config["shard_index"], config["shard_count"],
                            config["diagnostic_train_count"])
        owned = {job["item"]["id"]: job["item"] for stage in stages for job in stage["jobs"]}
        for item in owned.values():
            for asset in (Path(config["data_root"]) / item["gamefile"]).parent.iterdir():
                if asset.is_file():
                    require(file_hash(asset) == protocol["data_hashes"].get(
                        str(asset.relative_to(config["data_root"]))),
                        f"Frozen native task asset differs: {asset}")
        manifest = {
            "schema_version": r.SCHEMA, "budget": config["episode_budget"],
            "alfworld_environment_mode": "eval_out_of_distribution", "seed": 20260928,
            "task_inventory": list(owned.values()), "exposure": freeze["partition"]["label"],
            "protocol_sha256": protocol["protocol_sha256"],
        }
        manifest["manifest_sha256"] = digest(manifest)
        write(run / "manifest.json", manifest)
        write(run / "usage-ledger.json", {
            "used_tokens": 0, "reserved_tokens": 0, "attempts": 0, "accounted_attempts": 0,
            "cost_unknown": False, "maximum_tokens": None, "maximum_attempts": None,
            "deadline_unix": None, "cumulative_token_mode": "unlimited",
        })
        for root in config["resource"]["lease_roots"]:
            Path(root).mkdir(parents=True, exist_ok=True)
        process, descriptors, logfile = core.start_service(run, config)
        endpoint = f"http://127.0.0.1:{config['resource']['port']}/v1"
        core.await_service(process, endpoint, config["startup_timeout_seconds"])
        binding = {
            "hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
            "adapter_source_hashes": freeze["source_hashes"],
            "model": {"id": config["model"]["served_name"], "revision": REVISION,
                      "endpoints": [endpoint]},
            "gpu": {"visible_devices": "", "service_selection": str(run / "GPU_SELECTION.json")},
            "skillopt_root": config["skillopt_root"], "data_root": config["data_root"],
            "ledger": str(run / "usage-ledger.json"), "max_model_len": config["model"]["max_model_len"],
            "local_discriminator_logprobs_mode": "processed_logprobs", "typesafe_api_used": False,
        }
        binding["local_label_token_ids"] = bind_local_labels(binding, run / "LOCAL_TOKENIZER.json")
        binding["binding_sha256"] = digest(binding)
        write(run / "binding.json", binding)
        results = []
        for stage in stages:
            if stage["stage"] == "validation":
                write(run / "TRAIN_READY.json", {
                    "status": "PASS", "freeze_sha256": freeze["freeze_sha256"], "at_utc": now(),
                    "candidate_hashes": {k: v["harness_sha256"]
                                         for k, v in freeze["policies"].items()},
                })
                release = prepared / "VALIDATION_RELEASE.json"
                while not release.exists():
                    require(process.poll() is None, "Owned provider exited before validation")
                    peers = []
                    for peer in config["peers"]:
                        try:
                            status = fetch_json(f"http://{peer['address']}:{peer['port']}/status")
                            require("FAILURE" not in status, "Peer failed; validation stays sealed")
                            peers.append(status["TRAIN_READY"])
                        except (OSError, KeyError):
                            break
                    if len(peers) == len(config["peers"]):
                        require(all(peer["status"] == "PASS" and peer["freeze_sha256"] ==
                                    freeze["freeze_sha256"] for peer in peers),
                                "A peer has a different training/candidate freeze")
                        write(release, {
                            "status": "ALL_NATIVE_TRAIN_PASS", "freeze_sha256": freeze["freeze_sha256"],
                            "at_utc": now(), "peer_receipts": peers,
                            "validation_feedback_returned_to_training": False,
                        })
                        break
                    write(run / "PROGRESS.json", {"status": "WAITING_ALL_TRAIN",
                                                  "stage": "validation", "at_utc": now()})
                    time.sleep(10)
                require(read(release)["freeze_sha256"] == freeze["freeze_sha256"],
                        "Validation opened with a different candidate freeze")
            results.append(run_stage(run, config, freeze, stage))
        write(run / "NATIVE_COMPLETE.json", {
            "status": "NATIVE_TRAIN_VALIDATION_COMPLETE_TEST_BLOCKED",
            "freeze_sha256": freeze["freeze_sha256"], "stages": results,
            "usage": core.ledger_snapshot(run / "usage-ledger.json"), "at_utc": now(),
            "test_opened": False,
            "incomplete_baselines": ["skillopt", "autosaddler"],
            "positive_innovation_admitted": False,
        })
    except BaseException as error:
        write(run / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": now(),
            "automatic_retries": 0, "retry_requires_relevant_fix_and_new_run": True,
        })
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        for descriptor in descriptors:
            os.close(descriptor)
        if logfile is not None:
            logfile.close()
        write(run / "RETIREMENT.json", {
            "at_utc": now(), "service_exit_code": process.returncode if process else None,
            "audit": resource_audit(),
        })


def fetch_json(url: str) -> dict:
    with build_opener(ProxyHandler({})).open(url, timeout=60) as response:
        return json.load(response)


def metadata_service(config_path: Path) -> None:
    """Read-only, two-host allowlisted metadata; never executes remote commands."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    config = read(config_path)
    remote_only(config["authorized_hostname"])
    prepared = Path(config["prepared_directory"])
    run = Path(config["run_directory"])
    names = ("PROGRESS", "FIRST_EPISODE", "TRAIN_READY", "NATIVE_COMPLETE", "FAILURE", "RETIREMENT")
    allowed_files = {
        "FROZEN.json", "PARTITION.json", "PARTITION_AUDIT.json", "METHODS.json",
        "PREPARED.json", "QUEUE.json", "worker-00.json", "worker-01.json",
    }
    allowed_addresses = {peer["address"] for peer in config["peers"]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0] not in allowed_addresses:
                self.send_error(403)
                return
            if self.path == "/status":
                value = {name: read(run / f"{name}.json")
                         for name in names if (run / f"{name}.json").exists()}
                value["service"] = {"hostname": socket.gethostname(), "pid": os.getpid(),
                                    "at_utc": now(), "read_only": True}
            elif self.path == "/resource-audit":
                value = resource_audit()
            elif self.path.startswith("/prepared/") and self.path[10:] in allowed_files:
                value = read(prepared / self.path[10:])
            else:
                self.send_error(404)
                return
            body = json.dumps(value, sort_keys=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    write(prepared / f"METADATA_SERVICE-{config['shard_index']:02d}.json", {
        "pid": os.getpid(), "hostname": socket.gethostname(), "at_utc": now(),
        "address": config["metadata_address"], "port": config["metadata_port"],
        "allowed_addresses": sorted(allowed_addresses),
    })
    server = ThreadingHTTPServer((config["metadata_address"], config["metadata_port"]), Handler)
    server.timeout = 30
    try:
        while True:
            server.handle_request()
            retired = run / "RETIREMENT.json"
            if retired.exists() and time.time() - retired.stat().st_mtime > 600:
                return
    finally:
        server.server_close()


def monitor(prepared_path: Path) -> None:
    remote_only()
    prepared = read(prepared_path)
    root = prepared_path.parent / "monitor"
    root.mkdir(exist_ok=False)
    write(root / "PID.json", {"pid": os.getpid(), "hostname": socket.gethostname(), "at_utc": now()})
    previous = {}
    next_audit = 0
    while True:
        statuses = {}
        for worker in prepared["workers"]:
            try:
                status = fetch_json(f"http://{worker['host']}:{worker['metadata_port']}/status")
            except Exception as error:
                status = {"STATUS_READ_ERROR": str(error)}
            key = str(worker["index"])
            statuses[key] = status
            state = ("FAILED" if "FAILURE" in status else "COMPLETE" if "NATIVE_COMPLETE" in status
                     else status.get("PROGRESS", {}).get("status", "STARTING"))
            if previous.get(key) != state:
                append(root / "TRANSITIONS.jsonl", {"at_utc": now(), "worker": key,
                                                    "previous": previous.get(key), "state": state})
                previous[key] = state
        if time.monotonic() >= next_audit:
            audits = {}
            for worker in prepared["workers"]:
                try:
                    audits[str(worker["index"])] = fetch_json(
                        f"http://{worker['host']}:{worker['metadata_port']}/resource-audit")
                except Exception as error:
                    audits[str(worker["index"])] = {"error": str(error)}
            append(root / "CAPACITY_AUDIT.jsonl", {"at_utc": now(), "nodes": audits})
            next_audit = time.monotonic() + 300
        write(root / "HEARTBEAT.json", {
            "at_utc": now(), "transition_interval_seconds": 120, "resource_audit_seconds": 300,
            "workers": statuses, "next_stage_queue": read(prepared_path.parent / "QUEUE.json"),
            "test_release": "BLOCKED_INCOMPLETE_COMPLETE_OPTIMIZERS",
        })
        if all("RETIREMENT" in value for value in statuses.values()):
            write(root / "TERMINAL.json", {
                "at_utc": now(), "states": previous, "test_opened": False,
                "blocker": "Complete SkillOpt/AutoSaddler training-only adapters not yet admitted",
                "positive_innovation_admitted": False,
            })
            return
        time.sleep(min(120, max(1, next_audit - time.monotonic())))


def replay(config_path: Path) -> None:
    """Replay recorded public inputs/responses; no simulator, provider or scoring."""
    remote_only()
    config = read(config_path)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    legacy_path = Path(config["legacy_runtime"])
    require(file_hash(legacy_path) == config["legacy_runtime_sha256"], "Legacy source changed")
    spec = importlib.util.spec_from_file_location("archived_alfworld_runtime", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    from scripts import alfworld_jev_runtime as shared
    jobs = sorted(Path(config["archive_root"]).glob(config["job_glob"]),
                  key=lambda path: digest(["public-replay", str(path)]))
    jobs = jobs[:config["maximum_episodes"]]
    require(bool(jobs), "No archived native public traces")
    totals = {"episodes": 0, "decisions": 0, "commits": 0, "fallbacks": 0,
              "new_model_calls": 0, "new_benchmark_episodes": 0}
    inputs = {}
    for index, path in enumerate(jobs):
        episode = path.parent
        job = read(path)
        conversation = read(episode / "conversation.json")
        result = read(episode / "result.json")
        policy = job["policy"]
        protocol = policy["local_discriminator"]
        require(protocol["deterministic_action_selection"], "Replay requires deterministic controller")
        skill = policy["skills"][job["item"]["task_type"]]
        records = [read(episode / "decisions" / f"{turn:03d}.json")
                   for turn in range(len(conversation))]
        current = {}

        def cot(prompt, _output):
            require(prompt == current["decision"]["native_prompt"], "Replayed native prompt differs")
            return current["step"]["model_response"], current["step"]["usage"]

        def no_model(*_args, **_kwargs):
            raise AssertionError("Replay attempted a new model selection")

        actors = [module.InterleavedActor(protocol, cot=cot, jev=no_model)
                  for module in (legacy, shared)]
        for turn, (step, decision) in enumerate(zip(conversation, records)):
            current.update(step=step, decision=decision)
            kwargs = {
                "turn": turn, "task": result["task_description"], "prompt": decision["native_prompt"],
                "skill": skill, "observation": decision["public_observation"],
                "actions": decision["native_actions"],
                "history": [{"action": value["action"], "observation": value["observation"]}
                            for value in conversation[:turn]],
            }
            old, new = [actor.step(**kwargs, output=output / f"episode-{index:03d}" / str(turn))
                        for actor in actors]
            new_trace = new["decision"].pop("supported_local_decision")
            require(old == new, f"Legacy/shared decision divergence: {path} turn={turn}")
            require(old["text"] == step["model_response"] and old["usage"] == step["usage"],
                    f"Replay differs from the archived transition: {path} turn={turn}")
            totals["decisions"] += 1
            totals["commits" if new_trace["route"] == "commit" else "fallbacks"] += 1
        totals["episodes"] += 1
        inputs[str(episode)] = {
            "job_sha256": file_hash(path), "conversation_sha256": file_hash(episode / "conversation.json"),
            "public_decisions_sha256": digest(records),
        }
        write(output / "PROGRESS.json", totals)
    write(output / "RESULT.json", {
        "status": "PASS", "kind": "REGRESSION_ONLY_NOT_BENCHMARK_IMPROVEMENT",
        "at_utc": now(), **totals, "legacy_runtime_sha256": file_hash(legacy_path),
        "shared_runtime_sha256": file_hash(Path(shared.__file__)),
        "public_trace_inputs": inputs, "no_outcome_feedback_to_training": True,
    })
    print(json.dumps({"status": "PASS", **totals}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "worker", "monitor", "metadata-service", "replay", "audit-resources"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "audit-resources":
        remote_only()
        print(json.dumps(resource_audit()))
    else:
        require(args.config is not None, "--config is required")
        {"prepare": prepare, "worker": worker, "monitor": monitor,
         "metadata-service": metadata_service, "replay": replay}[args.command](args.config)


if __name__ == "__main__":
    main()
