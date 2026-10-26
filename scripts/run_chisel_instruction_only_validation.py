#!/usr/bin/env python3
"""Validate frozen CHISEL instructions without its controller on H200-2.

This is a new validation-only diagnostic, not a new optimizer or a replacement
for the frozen CHISEL results. The first native episode is an operability gate
and counts exactly once within the complete 710-task validation population.
Neither its success nor later outcomes can select or change the policy.

Run with the actor Python recorded in the pinned shared226 worker config:
  python -B -u scripts/run_chisel_instruction_only_validation.py --config CONFIG
See docs/CHISEL_INSTRUCTION_ONLY_VALIDATION.md for the remote launch contract.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_chisel_interactive_shared226 as native  # noqa: E402

SCHEMA = "chisel-instruction-only-validation/v1"
METHOD = "chisel_instruction_only_diagnostic"
LABEL = "CHISEL instruction-only diagnostic / controller removed"
HOST = "harnessmm-worker-2"
GPU_UUID = "GPU-d56c67a2-d981-4798-197b-1d9746e4f22a"
GPU_UUIDS = {
    0: "GPU-fc1d96a1-dd3e-3e77-c18b-4ae819c7b5d1",
    1: "GPU-b220235e-a0f6-465e-c5a8-b240327e441b",
    2: GPU_UUID,
    3: "GPU-a882be53-cec9-3edd-4831-2859574774cc",
    4: "GPU-9143b9e1-1c00-9b97-a98d-82682d94c2b1",
    5: "GPU-d3887b61-b63d-8c64-cf49-baaffe4792e9",
    6: "GPU-657eebdb-c1d7-afdf-9849-2e95efc6cf61",
    7: "GPU-e6a1ec65-8fed-25ef-5976-eea33add523d",
}
HOST_GPU_UUIDS = {
    HOST: GPU_UUIDS,
    "harnessmm-worker-1": {
        0: "GPU-fff23e0e-9414-4319-0726-0dd6c627992b",
        1: "GPU-4fc2eebc-77eb-e505-a99c-afa06bc82d19",
        2: "GPU-3105bf40-69e9-3577-57d6-dc389411707b",
        3: "GPU-6963b4db-f9a5-8be6-941c-90e28b372751",
        4: "GPU-c2a89dff-4b80-5b01-45b0-f97a1572f374",
        5: "GPU-0f9154be-76ae-88ad-5d16-6f08d61508da",
        6: "GPU-88484b23-694f-c332-46ec-46dbb8051d16",
        7: "GPU-018f8807-db17-5e57-1c43-adc5f8a79ee9",
    },
}
COUNTS = {"train": 710, "validation": 710, "test": 2133}
BUDGET = {
    "history_length": 2, "max_completion_tokens": 2048, "max_steps": 50,
    "provider_retries": 0, "request_timeout_seconds": 180, "seed": 9401, "temperature": 0,
}
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
SHARED_SOURCE = "scripts/run_chisel_interactive_shared226.py"


def instruction_policy(parent: dict) -> dict:
    native.verify_signed(parent, "harness_sha256")
    native.require(set(parent) == {"skills", "local_discriminator", "harness_sha256"},
                   "Unexpected parent policy fields; do not silently retain another controller")
    native.require(parent["skills"] and all(
        isinstance(value, str) and value for value in parent["skills"].values()),
        "The diagnostic requires the frozen nonempty CHISEL instructions")
    policy = deepcopy(parent)
    del policy["local_discriminator"]
    del policy["harness_sha256"]
    policy["harness_sha256"] = native.digest(policy)
    native.require(policy["skills"] == parent["skills"], "Frozen instruction bytes changed")
    return policy


def validate_contract(config: dict, worker: dict, freeze: dict, protocol: dict) -> dict:
    native.require(config["schema_version"] == SCHEMA and config["method"] == METHOD
                   and config["implementation_label"] == LABEL, "Diagnostic identity changed")
    native.require(config["scope"] == "validation" and config["validation_tasks"] == 710
                   and config["first_episode_tasks"] == 1, "Only complete validation710 is allowed")
    native.require(config["test_access"] is False
                   and config["optimization_from_validation"] is False,
                   "Test access and validation-driven optimization are forbidden")
    return validate_frozen_instruction_contract(config, worker, freeze, protocol)


def validate_frozen_instruction_contract(config: dict, worker: dict, freeze: dict, protocol: dict) -> dict:
    native.require(config["cumulative_token_mode"] == "unlimited"
                   and config["maximum_total_provider_tokens"] is None
                   and config["maximum_request_attempts"] is None
                   and config["maximum_run_seconds"] is None, "Discretionary ceilings are forbidden")
    index = config["resource"]["gpu_index"]
    host = config["authorized_hostname"]
    native.require(host in HOST_GPU_UUIDS and type(index) is int
                   and index in HOST_GPU_UUIDS[host]
                   and config["resource"]["gpu_uuid"] == HOST_GPU_UUIDS[host][index],
                   "Wrong authorized physical GPU")
    if "model_runtime" in config:
        native.require(set(config["model_runtime"]) == {"python", "directory"}
                       and all(isinstance(value, str) and Path(value).is_absolute()
                               for value in config["model_runtime"].values()),
                       "Only absolute serving-Python and checkpoint-cache paths may be rebound")
    native.require(type(config["workers"]) is int and type(worker["workers"]) is int
                   and 0 < config["workers"] <= worker["workers"],
                   "Unexpected worker count or overcommit")
    native.require(worker["episode_budget"] == BUDGET, "Common native actor budget changed")
    model = worker["model"]
    native.require(model["repository"] == "Qwen/Qwen3.5-9B"
                   and model["revision"] == native.REVISION
                   and model["served_name"] == "qwen35-9b@main"
                   and model["max_model_len"] == 32768
                   and model["enable_thinking"] is False, "Common backbone or context changed")
    native.verify_signed(freeze, "freeze_sha256")
    partition = native.verify_partition(protocol)
    native.require(freeze["freeze_sha256"] == config["native_freeze_sha256"]
                   and partition["partition_sha256"] == config["partition_sha256"]
                   and partition["membership_sha256"] == config["membership_sha256"]
                   and partition["counts"] == COUNTS, "Frozen partition or source differs")
    native.require(freeze["partition"]["partition_sha256"] == config["partition_sha256"]
                   and freeze["checkpoint"] == protocol["checkpoint"]
                   and freeze["checkpoint"]["revision"] == native.REVISION,
                   "Checkpoint or native freeze does not belong to the partition")
    parent = freeze["policies"]["chisel_shared_support_first"]
    native.require(parent == protocol["policies"]["no_stall_gate"]
                   and parent["harness_sha256"] == config["parent_policy_sha256"]
                   and native.digest(parent["skills"]) == config["skills_sha256"],
                   "The frozen CHISEL skill source changed")
    policy = instruction_policy(parent)
    native.require(policy["harness_sha256"] == config["diagnostic_policy_sha256"],
                   "The instruction-only policy differs from its prospective freeze")
    return policy


def validation_plan(protocol: dict) -> list[dict]:
    items = native.ranked_items(protocol, "validation")
    native.require(len(items) == 710 and len({item["id"] for item in items}) == 710,
                   "Validation must contain every frozen task exactly once")
    return [
        {"stage": name, "jobs": [{"method": METHOD, "item": item} for item in rows]}
        for name, rows in (("validation_gate", items[:1]), ("validation", items[1:]))
    ]


def new_ledger() -> dict:
    return {
        "used_tokens": 0, "reserved_tokens": 0, "attempts": 0, "accounted_attempts": 0,
        "cost_unknown": False, "maximum_tokens": None, "maximum_attempts": None,
        "deadline_unix": None, "cumulative_token_mode": "unlimited",
    }


def validate_actor_rows(rows: list[dict], items: list[dict]) -> None:
    expected = {item["id"]: item for item in items}
    native.require(len(items) > 0 and len(expected) == len(items),
                   "Validation population must be nonempty with unique task IDs")
    native.require(len(rows) == len(expected)
                   and {row["id"] for row in rows} == set(expected),
                   "Incomplete, duplicated or out-of-scope validation population")
    for row in rows:
        native.require(row["gamefile"] == expected[row["id"]]["gamefile"]
                       and row["method"] == METHOD
                       and type(row["seed"]) is int and row["seed"] == 9401,
                       "Native task, diagnostic method or seed changed")
        native.require(row["valid"] is True and row["cost_complete"] is True
                       and type(row["official_success"]) is int
                       and row["official_success"] in (0, 1), "Invalid native episode")
        native.require(all(type(row[key]) is int for key in ("turns", "calls", "attempted_calls"))
                       and 0 < row["turns"] <= 50
                       and row["calls"] == row["attempted_calls"] == row["turns"],
                       "Instruction-only requires one accounted model call for every action")
        usage = row["usage"]
        native.require(all(type(usage.get(key)) is int and usage[key] >= 0
                           for key in USAGE_KEYS)
                       and usage["prompt_tokens"] + usage["completion_tokens"]
                       == usage["total_tokens"] and usage["total_tokens"] > 0,
                       "Unknown, missing or inconsistent actor usage")


def read_rows(run: Path, stage: str) -> list[dict]:
    return [json.loads(line) for line in (run / f"{stage}-rows.jsonl").read_text().splitlines()
            if line.strip()]


def load_inputs(config: dict, run: Path) -> tuple[dict, dict, dict, dict]:
    native.require(isinstance(config["inputs"], dict)
                   and set(config["inputs"]) == {"worker", "freeze", "partition", "native_adapter"},
                   "Frozen input names differ from the validation-only contract")
    values, provenance = {}, {}
    for name, source in config["inputs"].items():
        path = Path(source["path"])
        native.require(native.file_hash(path) == source["sha256"], f"Frozen input changed: {path}")
        values[name] = native.read(path)
        provenance[name] = source
        if name != "partition":
            native.write(run / "inputs" / f"{name}.json", values[name])
    native.write(run / "INPUT_PROVENANCE.json", provenance)
    return values["worker"], values["freeze"], values["partition"], values["native_adapter"]


def runtime_config(config: dict, worker: dict, run: Path) -> dict:
    resource = {**deepcopy(worker["resource"]), **config["resource"]}
    model = {**deepcopy(worker["model"]), **config.get("model_runtime", {})}
    native.require(resource["lease_roots"], "The common GPU lease roots are required")
    return {
        "schema_version": SCHEMA, "method": METHOD, "implementation_label": LABEL,
        "authorized_hostname": config["authorized_hostname"], "run_directory": str(run), "scope": "validation",
        "model": model, "resource": resource,
        "skillopt_root": worker["skillopt_root"], "data_root": worker["data_root"],
        "actor_python": worker["actor_python"], "workers": config["workers"],
        "episode_budget": deepcopy(BUDGET), "seed": 20260928, "episode_seed": 9401,
        "startup_timeout_seconds": worker["startup_timeout_seconds"],
        "request_timeout_seconds": 180, "module_variants": [], "prior_run_directories": [],
        "maximum_total_provider_tokens": None, "maximum_request_attempts": None,
        "maximum_run_seconds": None, "cumulative_token_mode": "unlimited",
        "test_access": False, "optimization_from_validation": False,
    }


def ipc_scratch(run: Path) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="ci-ipc-", dir="/tmp"))
    native.require(len(str(scratch).encode()) + 1 + 36 <= 107,
                   "vLLM UUID socket paths must fit the Unix-domain socket limit")
    native.write(run / "IPC_DIRECTORY.json", {
        "path": str(scratch), "purpose": "Short model-service IPC socket namespace",
        "unix_socket_path_limit_bytes": 107, "uuid_suffix_bytes": 37,
        "preserved_for_run_diagnostics": True,
    })
    return scratch


def freeze_sources(config: dict, prior: dict, run: Path) -> dict:
    checksums = {}
    for name, expected in prior["source_hashes"].items():
        if name == SHARED_SOURCE:
            expected = config["shared_helper_sha256"]
        if name == "scripts/run_alfworld_modular_rsi.py" and "service_helper_sha256" in config:
            expected = config["service_helper_sha256"]
        if name == "phiagent/learning/alfworld_modular_rsi.py" and "prompt_module_sha256" in config:
            expected = config["prompt_module_sha256"]
        native.require(native.file_hash(ROOT / name) == expected, f"Common source changed: {name}")
        checksums[name] = expected
    native.require(SHARED_SOURCE in checksums, "Missing common actor-stage source binding")
    name = str(Path(__file__).resolve().relative_to(ROOT))
    checksums[name] = native.file_hash(ROOT / name)
    for name, expected in checksums.items():
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        native.require(native.file_hash(destination) == expected, "Source changed while freezing")
    return checksums


def verify_assets(runtime: dict, protocol: dict, receipt: dict, items: list[dict], run: Path,
                  *, data_scope: str = "validation") -> None:
    native.require(data_scope in ("train", "validation"), "Test assets are not permitted")
    allowed = {item["gamefile"] for item in protocol["splits"][data_scope]}
    native.require(len({item["id"] for item in items}) == len(items)
                   and {item["gamefile"] for item in items} <= allowed,
                   "Requested assets escape their frozen data partition")
    native.require(receipt["source_cache"] == runtime["skillopt_root"]
                   and len(receipt["files"]) == receipt["files_count"] == 182,
                   "The native adapter cache differs from the predecessor")
    for name, entry in receipt["files"].items():
        native.require(native.file_hash(Path(runtime["skillopt_root"]) / name) == entry["sha256"],
                       f"Frozen native adapter changed: {name}")
    native.write(run / "NATIVE_ADAPTER_SOURCE.json", receipt)
    for name, checksum in protocol["checkpoint"]["files"].items():
        native.require(native.file_hash(Path(runtime["model"]["directory"]) / name) == checksum,
                       f"Exact checkpoint bytes changed: {name}")
    native.write(run / "CHECKPOINT.json", protocol["checkpoint"])
    selected = {}
    parents = {str(Path(item["gamefile"]).parent) for item in items}
    for name, checksum in protocol["data_hashes"].items():
        if str(Path(name).parent) not in parents:
            continue
        native.require(native.file_hash(Path(runtime["data_root"]) / name) == checksum,
                       f"Frozen validation asset changed: {name}")
        selected[name] = checksum
    native.require(all(item["gamefile"] in selected for item in items),
                   "Validation task data hashes are incomplete")
    data_manifest = {
        "files": selected, "tasks": len(items), "partition_sha256": protocol["protocol_sha256"],
        "scope": data_scope, "other_partition_task_assets_read": False,
    }
    data_manifest["train_or_test_task_assets_read" if data_scope == "validation"
                  else "validation_or_test_task_assets_read"] = False
    native.write(run / f"{data_scope.upper()}_DATA_MANIFEST.json", data_manifest)


def audit_actor_cost(run: Path, rows: list[dict]) -> dict:
    from scripts.audit_alfworld_modular_rsi import audit_cost_tree

    cost = audit_cost_tree(run, set())
    usage = {key: sum(row["usage"][key] for row in rows) for key in USAGE_KEYS}
    native.require(cost["own_usage"] == usage
                   and cost["all_in_calls"] == sum(row["calls"] for row in rows),
                   "Raw provider calls, usage ledger and episode costs differ")
    return cost


def stop_owned_service(process, descriptors: list[int], logfile) -> None:
    with ExitStack() as cleanup:
        if logfile is not None:
            cleanup.callback(logfile.close)
        for descriptor in reversed(descriptors):
            cleanup.callback(os.close, descriptor)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)


def run(config_path: Path, *, lease_descriptors: list[int] | None = None) -> None:
    native.remote_only(socket.gethostname())
    native.require(socket.gethostname() in HOST_GPU_UUIDS, "Only the authorized H200 hosts are allowed")
    config = native.read(config_path)
    native.require(config["authorized_hostname"] == socket.gethostname(), "Configured host differs")
    run_dir = Path(config["run_directory"])
    native.require(run_dir.is_absolute()
                   and run_dir.parent == Path("/opt/phiagent/runs/chisel-instruction-only"),
                   "Use a new instruction-only run directory, never an existing experiment")
    run_dir.mkdir(parents=True, exist_ok=False)
    native.write(run_dir / "DIAGNOSTIC_CONFIG.json", config)
    native.write(run_dir / "PID.json", {
        "pid": os.getpid(), "hostname": socket.gethostname(), "at_utc": native.now(),
        "argv": sys.argv, "config_sha256": native.file_hash(config_path),
    })
    process, descriptors, logfile = None, list(lease_descriptors or []), None
    try:
        worker, old_freeze, protocol, adapter = load_inputs(config, run_dir)
        policy = validate_contract(config, worker, old_freeze, protocol)
        native.require(Path(sys.prefix).resolve() == Path(worker["actor_python"]).parent.parent.resolve(),
                       f"Launch with the frozen actor Python: {worker['actor_python']}")
        runtime = runtime_config(config, worker, run_dir)
        scratch = ipc_scratch(run_dir)
        os.environ.update(
            CUDA_VISIBLE_DEVICES="", NO_PROXY="127.0.0.1,localhost",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
            TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
        )
        sys.dont_write_bytecode = True
        from scripts import run_alfworld_modular_rsi as core

        sources = freeze_sources(config, old_freeze, run_dir)
        native.write(run_dir / "config.json", runtime)
        native.write(run_dir / "usage-ledger.json", new_ledger())
        native.write(run_dir / "PROVENANCE.json", {
            "argv": sys.argv, "hostname": socket.gethostname(), "at_utc": native.now(),
            "python": sys.executable, "seed": 9401, "partition_seed": 20260928,
            "source_hashes": sources, "source_git_base": config["source_git_base"],
            "git_head": core.repair.capture(["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT),
            "git_status": core.repair.capture(
                ["git", "--no-pager", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT),
            "git_diff": core.repair.capture(["git", "--no-pager", "diff", "--binary"], cwd=ROOT),
            "actor_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
            "model_packages": core.repair.capture([runtime["model"]["python"], "-m", "pip", "freeze"]),
        })
        stages = validation_plan(protocol)
        items = [job["item"] for stage in stages for job in stage["jobs"]]
        freeze = {
            "schema_version": SCHEMA, "implementation_label": LABEL, "at_utc": native.now(),
            "policies": {METHOD: policy}, "parent_policy_sha256": config["parent_policy_sha256"],
            "native_freeze_sha256": config["native_freeze_sha256"],
            "skills_sha256": config["skills_sha256"], "source_hashes": sources,
            "partition_sha256": config["partition_sha256"],
            "membership_sha256": config["membership_sha256"], "primary_counts": COUNTS,
            "validation_ids": [item["id"] for item in items], "test_access": False,
            "first_episode_selection": "First shared226 hash-ranked validation ID; no outcome selection",
            "new_optimizer_calls": 0, "completed_new_optimizer": False,
            "optimization_from_validation": False, "paper_admitted": False,
        }
        freeze["freeze_sha256"] = native.digest(freeze)
        native.write(run_dir / "POLICY_FREEZE.json", freeze)
        native.write(run_dir / "PRELAUNCH_RESOURCE_AUDIT.json", native.resource_audit())
        verify_assets(runtime, protocol, adapter, items, run_dir)
        manifest = {
            "schema_version": core.repair.SCHEMA, "budget": runtime["episode_budget"],
            "alfworld_environment_mode": "eval_out_of_distribution",
            "seed": 20260928, "task_inventory": items, "scope": "validation",
            "protocol_sha256": config["partition_sha256"],
        }
        manifest["manifest_sha256"] = native.digest(manifest)
        native.write(run_dir / "manifest.json", manifest)
        for directory in runtime["resource"]["lease_roots"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        inherited = descriptors if lease_descriptors is not None else None
        descriptors = []
        process, descriptors, logfile = core.start_service(run_dir, runtime, lease_descriptors=inherited)
        endpoint = f"http://127.0.0.1:{runtime['resource']['port']}/v1"
        core.await_service(process, endpoint, runtime["startup_timeout_seconds"])
        binding = {
            "hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
            "adapter_source_hashes": sources,
            "model": {"id": runtime["model"]["served_name"], "revision": native.REVISION,
                      "endpoints": [endpoint]},
            "gpu": {"visible_devices": "", "service_selection": str(run_dir / "GPU_SELECTION.json")},
            "skillopt_root": runtime["skillopt_root"], "data_root": runtime["data_root"],
            "ledger": str(run_dir / "usage-ledger.json"), "max_model_len": 32768,
        }
        binding["binding_sha256"] = native.digest(binding)
        native.write(run_dir / "binding.json", binding)
        rows = []
        for stage in stages:
            native.run_stage(run_dir, runtime, freeze, stage)
            batch = read_rows(run_dir, stage["stage"])
            validate_actor_rows(batch, [job["item"] for job in stage["jobs"]])
            rows.extend(batch)
            if stage["stage"] == "validation_gate":
                cost = audit_actor_cost(run_dir, rows)
                native.write(run_dir / "FIRST_EPISODE_GATE.json", {
                    "status": "PASS_NATIVE_ACTOR_AND_ACCOUNTING", "row": batch[0],
                    "cost": cost, "included_once_in_validation710": True,
                    "success_required": False, "policy_sha256": policy["harness_sha256"],
                    "local_controller_present": False, "new_optimizer_calls": 0,
                })
        validate_actor_rows(rows, items)
        cost = audit_actor_cost(run_dir, rows)
        for name, checksum in sources.items():
            native.require(native.file_hash(ROOT / name) == checksum, "Source changed during execution")
        native.write(run_dir / "COST_AUDIT.json", cost)
        result = {
            "schema_version": SCHEMA, "status": "COMPLETE_VALIDATION_ONLY_DIAGNOSTIC",
            "method": METHOD, "implementation_label": LABEL, "at_utc": native.now(),
            "summary": native.summarize(rows)[METHOD], "cost": cost,
            "freeze_sha256": freeze["freeze_sha256"], "partition_sha256": config["partition_sha256"],
            "policy_sha256": policy["harness_sha256"], "skills_sha256": config["skills_sha256"],
            "validation_ids_sha256": native.digest(sorted(row["id"] for row in rows)),
            "first_episode_counted_once": True, "failed_native_episodes_retained": True,
            "new_optimizer_calls": 0, "completed_new_optimizer": False, "official_test_calls": 0,
            "test_access": False, "optimization_from_validation": False,
            "historical_optimizer_costs_included": False,
            "cumulative_token_mode": "unlimited", "paper_admitted": False,
        }
        result["result_sha256"] = native.digest(result)
        native.write(run_dir / "RESULT.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        ledger_path = run_dir / "usage-ledger.json"
        native.write(run_dir / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": native.now(),
            "ledger": native.read(ledger_path) if ledger_path.exists() else None,
            "raw_requests": len(list(run_dir.rglob("request.json"))),
            "raw_responses": len(list(run_dir.rglob("response.json"))),
            "failed_attempts_preserved": True, "unknown_usage_not_zero_imputed": True,
            "automatic_retries": 0, "test_access": False, "paper_admitted": False,
        })
        raise
    finally:
        stop_owned_service(process, descriptors, logfile)
        native.write(run_dir / "RETIREMENT.json", {
            "at_utc": native.now(), "owned_service_pid": process.pid if process else None,
            "service_exit_code": process.returncode if process else None,
            "owned_lease_descriptors_closed": len(descriptors), "unrelated_processes_stopped": 0,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lease-fds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}; preserve this attempt")

    signal.signal(signal.SIGTERM, interrupted)
    run(args.config, lease_descriptors=args.lease_fds)


if __name__ == "__main__":
    main()
