#!/usr/bin/env python3
"""Finite authority-gated four-method test preparation, paired-shard execution and collection.

The source command reads registry/gate metadata and code only. Every command
that can read test identities, assets or outcomes requires separate signed
parent authority first. No old test outcomes or optimizer runs are reused.
"""
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
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_chisel_efficiency_validation as validation  # noqa: E402

common, native, core, training = (
    validation.common, validation.native, validation.core, validation.training)
SCHEMA = "chisel-four-method-final-test/v1"
AUTHORITY_SCHEMA = "chisel-four-method-test-authority/v1"
METHODS = ("no_skill", "skillopt", "autosaddler", "memory_guarded64")
EXPOSURE = "historical_stall226_previously_exposed_custom_test"
EXTRA_SOURCES = (
    "phiagent/learning/supported_local_decision.py",
    common.SHARED_SOURCE, "scripts/run_chisel_instruction_only_validation.py",
    "scripts/run_chisel_efficiency_train.py", "scripts/run_chisel_efficiency_validation.py",
    "scripts/run_chisel_four_method_test.py",
)


def source_names():
    return tuple(dict.fromkeys((*core.SOURCE_FILES, *EXTRA_SOURCES)))


def check_template(config):
    native.require(config["schema_version"] == SCHEMA and config["test_tasks"] == 2133
                   and tuple(config["methods"]) == METHODS, "Wrong final-test comparison")
    native.require(config["episode_budget"] == common.BUDGET
                   and config["model_revision"] == native.REVISION
                   and config["actor_context_tokens"] == 32768,
                   "Common model/seed/native budget changed")
    native.require(config["reuse_old_test_outcomes"] is False
                   and config["test_population_exposure"] == EXPOSURE,
                   "No outcome reuse or unseen-population relabeling is permitted")
    native.require(config["cumulative_token_mode"] == "unlimited"
                   and all(config[key] is None for key in (
                       "maximum_total_provider_tokens", "maximum_request_attempts", "maximum_run_seconds")),
                   "Discretionary cumulative ceilings are forbidden")
    native.require(type(config["workers_per_gpu"]) is int and 1 <= config["workers_per_gpu"] <= 8,
                   "Invalid per-GPU actor concurrency")
    nodes = config["nodes"]
    native.require(2 <= len(nodes) <= 16
                   and {node["hostname"] for node in nodes} == set(common.HOST_GPU_UUIDS),
                   "The paired shard plan must use both authorized H200 hosts")
    devices, ports = set(), set()
    for node in nodes:
        host, index = node["hostname"], node["gpu_index"]
        native.require(type(index) is int and common.HOST_GPU_UUIDS[host].get(index) == node["gpu_uuid"],
                       "Unapproved physical GPU/UUID")
        native.require((host, index) not in devices and (host, node["port"]) not in ports,
                       "Duplicate GPU or service port in the shard plan")
        native.require(all(Path(node[key]).is_absolute() for key in ("model_directory", "model_python")),
                       "Use explicit existing model cache/runtime paths")
        devices.add((host, index))
        ports.add((host, node["port"]))
    root = Path(config["run_root"])
    native.require(root.is_absolute() and root.parent == Path("/opt/phiagent/runs/chisel-four-test"),
                   "Use a new, short four-method run root")
    native.require(len(str(root / "s15" / "ipc").encode()) + 37 <= 107,
                   "Run identity is too long for vLLM IPC sockets")


def check_registry(config, registry, gate):
    native.verify_signed(registry, "registry_sha256")
    native.require(registry["registry_sha256"] == config["registry_sha256"]
                   and registry["partition_sha256"] == config["partition_sha256"]
                   and registry["test_tasks"] == 2133 and registry["validation_tasks"] == 710
                   and set(registry["methods"]) == set(METHODS),
                   "Wrong qualified registry, methods or partition")
    native.require(gate["registry_sha256"] == registry["registry_sha256"]
                   and {key: value for key, value in gate.items() if key != "registry_sha256"}
                   == registry["validation_gate"]
                   and gate["passed"] is True and not gate["evidence_errors"]
                   and gate["status"] == "VALIDATION_QUALIFIED_AWAITING_SEPARATE_TEST_AUTHORITY"
                   and gate["test_release_authorized"] is False and registry["paper_admitted"] is False,
                   "Validation qualification cannot substitute for separate test authority")
    for method in METHODS:
        entry = registry["methods"][method]
        native.require(entry["validation_status"] == "COMPLETE_METADATA_VERIFIED",
                       "A comparator has incomplete validation")
        policy = entry["policy"]
        native.verify_signed(policy, "harness_sha256")
        native.require(policy["harness_sha256"] == config["policy_hashes"][method],
                       "Registered frozen policy changed")
        expected = {"skills", "harness_sha256"}
        if method == validation.METHOD:
            expected.add("efficiency_profile")
            native.require(policy["efficiency_profile"] == validation.METHOD,
                           "Wrong TRAIN-selected profile")
        native.require(set(policy) == expected, "A handwritten solver or policy extension was introduced")
        if method == "no_skill":
            native.require(all(text == "" for text in policy["skills"].values()), "No Skill changed")
        expected_function = ("initialize_training_worker" if method == validation.METHOD else "initialize_worker")
        expected_module = ("scripts.run_chisel_efficiency_train" if method == validation.METHOD
                           else "scripts.run_chisel_interactive_shared226")
        native.require(entry["worker_initializer"]["function"] == expected_function
                       and entry["worker_initializer"]["module"] == expected_module,
                       "Wrong per-method initializer declaration")


def metadata_binding(config_path, registry_path, gate_path):
    config = native.read(config_path)
    check_template(config)
    native.require(native.file_hash(registry_path) == config["registry_file_sha256"]
                   and native.file_hash(gate_path) == config["gate_file_sha256"],
                   "Qualified registry/gate bytes changed")
    registry, gate = native.read(registry_path), native.read(gate_path)
    check_registry(config, registry, gate)
    return config, registry, gate


def source_manifest(config_path: Path, registry_path: Path, gate_path: Path, output: Path):
    config, registry, _gate = metadata_binding(config_path, registry_path, gate_path)
    native.require(not output.exists(), "Never overwrite a source freeze")
    hashes = {name: native.file_hash(ROOT / name) for name in source_names()}
    for name, checksum in {**registry["executor_source_hashes"], **config["protected_source_hashes"]}.items():
        native.require(hashes[name] == checksum, f"Frozen policy/actor source changed: {name}")
    value = {
        "schema_version": SCHEMA, "kind": "SOURCE_FREEZE_NOT_AUTHORITY", "at_utc": native.now(),
        "comparison_id": config["comparison_id"], "template_sha256": native.file_hash(config_path),
        "template_canonical_sha256": native.digest(config),
        "registry_sha256": registry["registry_sha256"],
        "registry_file_sha256": native.file_hash(registry_path), "gate_file_sha256": native.file_hash(gate_path),
        "source_hashes": hashes, "resource_plan_sha256": native.digest(config["nodes"]),
        "partition_sha256": config["partition_sha256"],
        "partition_file_sha256": config["inputs"]["partition"]["sha256"],
        "test_data_read": False, "test_execution_authorized": False,
    }
    value["source_manifest_sha256"] = native.digest(value)
    native.write(output, value)
    return value


def check_authority(config, registry, gate, source, authority, template_hash):
    check_template(config)
    check_registry(config, registry, gate)
    native.verify_signed(source, "source_manifest_sha256")
    native.verify_signed(authority, "authority_sha256")
    native.require(source["kind"] == "SOURCE_FREEZE_NOT_AUTHORITY"
                   and source["template_sha256"] == template_hash
                   and source["template_canonical_sha256"] == native.digest(config)
                   and source["registry_sha256"] == registry["registry_sha256"]
                   and source["registry_file_sha256"] == config["registry_file_sha256"]
                   and source["gate_file_sha256"] == config["gate_file_sha256"]
                   and source["resource_plan_sha256"] == native.digest(config["nodes"])
                   and source["partition_sha256"] == config["partition_sha256"]
                   and source["partition_file_sha256"] == config["inputs"]["partition"]["sha256"]
                   and set(source["source_hashes"]) == set(source_names()),
                   "Source freeze does not bind this template/registry/gate/resource plan")
    native.require(authority["schema_version"] == AUTHORITY_SCHEMA
                   and authority["issuer"] == "parent"
                   and authority["test_execution_authorized"] is True
                   and authority["exposure_review_completed"] is True
                   and authority["comparison_id"] == config["comparison_id"]
                   and authority["registry_sha256"] == registry["registry_sha256"]
                   and authority["registry_file_sha256"] == config["registry_file_sha256"]
                   and authority["gate_file_sha256"] == config["gate_file_sha256"]
                   and authority["source_manifest_sha256"] == source["source_manifest_sha256"]
                   and authority["template_sha256"] == template_hash
                   and authority["resource_plan_sha256"] == native.digest(config["nodes"]),
                   "Separate signed parent test authority is missing or bound elsewhere")
    native.require(tuple(authority["methods"]) == METHODS
                   and authority["policy_hashes"] == config["policy_hashes"]
                   and authority["partition_sha256"] == config["partition_sha256"]
                   and authority["partition_file_sha256"] == config["inputs"]["partition"]["sha256"]
                   and authority["test_tasks"] == 2133
                   and authority["episode_budget"] == common.BUDGET
                   and authority["model_revision"] == native.REVISION
                   and authority["test_population_exposure"] == EXPOSURE
                   and authority["reuse_old_test_outcomes"] is False
                   and authority["modify_original_runs_or_seals"] is False,
                   "Authority cannot change policy, population, exposure, budget or reuse boundaries")
    native.require(isinstance(authority["issued_at_utc"], str) and authority["issued_at_utc"]
                   and isinstance(authority["exposure_review_note"], str) and authority["exposure_review_note"],
                   "Parent exposure review must be explicit")
    for name, checksum in source["source_hashes"].items():
        native.require(native.file_hash(ROOT / name) == checksum, f"Authorized source changed: {name}")


def authorize(config_path, registry_path, gate_path, source_path, authority_path):
    config, registry, gate = metadata_binding(config_path, registry_path, gate_path)
    source, authority = native.read(source_path), native.read(authority_path)
    check_authority(config, registry, gate, source, authority, native.file_hash(config_path))
    return config, registry, gate, source, authority


def paired_shards(test_items, count):
    native.require(type(count) is int and 1 <= count <= len(test_items), "Invalid shard count")
    native.require(len(test_items) == len({item["id"] for item in test_items})
                   == len({item["gamefile"] for item in test_items}), "Duplicate test identities")
    result = [test_items[index::count] for index in range(count)]
    native.require(sum(map(len, result)) == len(test_items), "Incomplete paired partition")
    return result


def method_order(comparison_id, shard_index):
    return sorted(METHODS, key=lambda method: native.digest(
        ["four-method-test-order", comparison_id, shard_index, method]))


def initializer_for(method):
    native.require(method in METHODS, "Old handwritten CHISEL is not an executable method")
    return training.initialize_training_worker if method == validation.METHOD else native.initialize_worker


def prepare(config_path, registry_path, gate_path, source_path, authority_path, output):
    # No partition bytes or task assets are read before this separate authority check.
    config, registry, gate, source, authority = authorize(
        config_path, registry_path, gate_path, source_path, authority_path)
    native.remote_only(socket.gethostname())
    native.require(not output.exists(), "Never overwrite a prepared test identity")
    output.mkdir(parents=True, exist_ok=False)
    try:
        worker, old_freeze, protocol, adapter = common.load_inputs(config, output)
        native.verify_signed(old_freeze, "freeze_sha256")
        partition = native.verify_partition(protocol)
        native.require(partition["partition_sha256"] == config["partition_sha256"]
                       and partition["counts"] == common.COUNTS
                       and old_freeze["freeze_sha256"] == config["native_freeze_sha256"]
                       and old_freeze["checkpoint"] == protocol["checkpoint"]
                       and protocol["checkpoint"]["revision"] == native.REVISION
                       and native.digest(protocol["checkpoint"]["files"]) == config["checkpoint_map_sha256"],
                       "The authorized frozen710/710/2133 partition/checkpoint changed")
        native.require(worker["episode_budget"] == common.BUDGET
                       and worker["model"]["revision"] == native.REVISION
                       and worker["model"]["max_model_len"] == 32768,
                       "Frozen common actor configuration differs")
        test_items = native.ranked_items(protocol, "test")
        native.require(len(test_items) == 2133, "Never substitute a partial final-test population")
        shards = paired_shards(test_items, len(config["nodes"]))
        prepared = {
            "schema_version": SCHEMA, "comparison_id": config["comparison_id"],
            "authority_sha256": authority["authority_sha256"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "registry_sha256": registry["registry_sha256"], "partition_sha256": config["partition_sha256"],
            "test_population_exposure": EXPOSURE, "reuse_old_test_outcomes": False,
            "methods": list(METHODS), "test_tasks": 2133, "expected_episodes": 4 * 2133,
            "test_identity_sha256": native.digest(sorted(
                (item["id"], item["gamefile"]) for item in test_items)),
            "shards": [],
        }
        for index, (node, items) in enumerate(zip(config["nodes"], shards)):
            run_dir = str(Path(config["run_root"]) / f"s{index:02d}")
            value = {
                "shard_index": index, "shard_count": len(shards), "node": node, "items": items,
                "run_directory": run_dir, "method_order": method_order(config["comparison_id"], index),
                "task_ids_sha256": native.digest(sorted(item["id"] for item in items)),
                "authority_sha256": authority["authority_sha256"],
                "registry_sha256": registry["registry_sha256"],
                "source_manifest_sha256": source["source_manifest_sha256"],
            }
            value["shard_sha256"] = native.digest(value)
            native.write(output / f"shard-{index:02d}.json", value)
            prepared["shards"].append({key: value[key] for key in (
                "shard_index", "run_directory", "task_ids_sha256", "shard_sha256", "node")})
        prepared["prepared_sha256"] = native.digest(prepared)
        for name, value in (
            ("TEMPLATE.json", config), ("REGISTRY.json", registry), ("RELEASE_GATE.json", gate),
            ("SOURCE_FREEZE.json", source), ("AUTHORITY.json", authority),
            ("PREPARED.json", prepared),
        ):
            native.write(output / name, value)
        for destination, input_name in (
            ("PARTITION.json", "partition"), ("WORKER_BASE.json", "worker"),
            ("NATIVE_ADAPTER_SOURCE.json", "native_adapter"),
        ):
            shutil.copyfile(config["inputs"][input_name]["path"], output / destination)
            native.require(native.file_hash(output / destination) == config["inputs"][input_name]["sha256"],
                           "A frozen input changed during preparation")
        return prepared
    except BaseException as error:
        native.write(output / "PREPARATION_FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(),
            "authority_sha256": authority["authority_sha256"], "new_model_calls": 0})
        raise


def authorized_prepared(prepared_root, authority_path):
    # These five files contain policy/source/authority metadata, not test identities.
    config = native.read(prepared_root / "TEMPLATE.json")
    registry = native.read(prepared_root / "REGISTRY.json")
    gate = native.read(prepared_root / "RELEASE_GATE.json")
    source = native.read(prepared_root / "SOURCE_FREEZE.json")
    authority = native.read(authority_path)
    check_authority(config, registry, gate, source, authority, source["template_sha256"])
    native.require(native.digest(config) == source["template_canonical_sha256"],
                   "Prepared template differs from the authorized original")
    prepared = native.read(prepared_root / "PREPARED.json")
    native.verify_signed(prepared, "prepared_sha256")
    native.require(prepared["authority_sha256"] == authority["authority_sha256"]
                   and prepared["registry_sha256"] == registry["registry_sha256"]
                   and prepared["source_manifest_sha256"] == source["source_manifest_sha256"]
                   and prepared["partition_sha256"] == config["partition_sha256"]
                   and prepared["test_tasks"] == 2133 and prepared["expected_episodes"] == 4 * 2133
                   and tuple(prepared["methods"]) == METHODS
                   and prepared["test_population_exposure"] == EXPOSURE
                   and prepared["reuse_old_test_outcomes"] is False
                   and len(prepared["shards"]) == len(config["nodes"]),
                   "Prepared test identity is not authorized by this parent")
    for index, shard in enumerate(prepared["shards"]):
        native.require(shard["shard_index"] == index and shard["node"] == config["nodes"][index]
                       and shard["run_directory"] == str(Path(config["run_root"]) / f"s{index:02d}"),
                       "Prepared resource/shard mapping changed")
    return config, registry, source, authority, prepared


def verify_test_assets(runtime, protocol, adapter, items, output):
    allowed = {item["id"]: item["gamefile"] for item in protocol["splits"]["test"]}
    native.require(all(allowed.get(item["id"]) == item["gamefile"] for item in items),
                   "A task escaped the authorized test partition")
    native.require(adapter["source_cache"] == runtime["skillopt_root"]
                   and len(adapter["files"]) == adapter["files_count"] == 182,
                   "Native adapter identity differs")
    for name, info in adapter["files"].items():
        native.require(native.file_hash(Path(runtime["skillopt_root"]) / name) == info["sha256"],
                       f"Native adapter bytes changed: {name}")
    for name, checksum in protocol["checkpoint"]["files"].items():
        native.require(native.file_hash(Path(runtime["model"]["directory"]) / name) == checksum,
                       f"Checkpoint bytes changed: {name}")
    parents = {str(Path(item["gamefile"]).parent) for item in items}
    selected = {}
    for name, checksum in protocol["data_hashes"].items():
        if str(Path(name).parent) in parents:
            native.require(native.file_hash(Path(runtime["data_root"]) / name) == checksum,
                           f"Authorized test asset changed: {name}")
            selected[name] = checksum
    native.require(all(item["gamefile"] in selected for item in items), "Incomplete test asset manifest")
    native.write(output / "TEST_DATA_MANIFEST.json", {
        "files": selected, "tasks": len(items), "scope": "test",
        "exposure": EXPOSURE, "partition_sha256": protocol["protocol_sha256"]})
    native.write(output / "CHECKPOINT.json", protocol["checkpoint"])
    native.write(output / "NATIVE_ADAPTER_SOURCE.json", adapter)


def method_audit(root, method, rows, items, policies):
    policy = policies[method]
    validation.validate_rows(rows, items, method, policy["harness_sha256"])
    for row in rows:
        episode = Path(row["artifact_dir"])
        native.require(episode.resolve().is_relative_to(root.resolve()) and "execution_reuse" not in row,
                       "Old results or foreign episode paths entered this new test")
        native.require(native.read(episode / "job.json")["policy"] == policy,
                       "An executed policy differs from the exact registry policy")
    return (training.audit_rows(root, rows, policies) if method == validation.METHOD
            else common.audit_actor_cost(root, rows))


def worker(prepared_root: Path, authority_path: Path, index: int, lease_descriptors=None):
    native.remote_only(socket.gethostname())
    config, registry, source, authority, prepared = authorized_prepared(prepared_root, authority_path)
    native.require(type(index) is int and 0 <= index < len(prepared["shards"]), "Invalid shard index")
    shard = native.read(prepared_root / f"shard-{index:02d}.json")
    native.verify_signed(shard, "shard_sha256")
    native.require(0 <= index < len(prepared["shards"])
                   and shard["shard_sha256"] == prepared["shards"][index]["shard_sha256"]
                   and shard["node"]["hostname"] == socket.gethostname(),
                   "Wrong authorized physical host/shard")
    root = Path(shard["run_directory"])
    native.require(root == Path(config["run_root"]) / f"s{index:02d}", "Run identity changed")
    root.mkdir(parents=True, exist_ok=False)
    native.write(root / "PID.json", {"pid": os.getpid(), "hostname": socket.gethostname(),
                                   "argv": sys.argv, "at_utc": native.now()})
    process, descriptors, logfile = None, list(lease_descriptors or []), None
    method_roots = {}
    try:
        for name, key in (("PARTITION.json", "partition"), ("WORKER_BASE.json", "worker"),
                          ("NATIVE_ADAPTER_SOURCE.json", "native_adapter")):
            native.require(native.file_hash(prepared_root / name) == config["inputs"][key]["sha256"],
                           f"Prepared immutable input changed: {name}")
        protocol = native.read(prepared_root / "PARTITION.json")
        native.verify_partition(protocol)
        native.require(protocol["protocol_sha256"] == config["partition_sha256"], "Partition changed")
        expected = native.ranked_items(protocol, "test")[index::len(config["nodes"])]
        native.require(shard["items"] == expected and shard["method_order"]
                       == method_order(config["comparison_id"], index), "Paired shard assignment changed")
        base = native.read(prepared_root / "WORKER_BASE.json")
        native.require(Path(sys.prefix).resolve() == Path(base["actor_python"]).parent.parent.resolve(),
                       "Use the existing frozen native actor Python")
        node = shard["node"]
        worker_config = {**config, "authorized_hostname": node["hostname"],
                         "workers": config["workers_per_gpu"],
                         "resource": {"host_alias": node["host_alias"], "gpu_index": node["gpu_index"],
                                      "gpu_uuid": node["gpu_uuid"], "port": node["port"]},
                         "model_runtime": {"python": node["model_python"], "directory": node["model_directory"]}}
        runtime = common.runtime_config(worker_config, base, root)
        runtime.update(schema_version=SCHEMA, method="four_method_test", scope="test",
                       test_access=True, implementation_label="New registry four-method final test")
        scratch = root / "ipc"
        native.require(len(str(scratch).encode()) + 37 <= 107, "IPC path exceeds Unix socket capacity")
        scratch.mkdir()
        tempfile.tempdir = str(scratch)
        os.environ.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(scratch), TEMP=str(scratch),
                          TMP=str(scratch), NO_PROXY="127.0.0.1,localhost",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        sys.dont_write_bytecode = True
        for name, checksum in source["source_hashes"].items():
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            native.require(native.file_hash(target) == checksum, "Source changed during snapshot")
        for name, value in (("AUTHORITY.json", authority), ("REGISTRY.json", registry),
                            ("SOURCE_FREEZE.json", source), ("SHARD.json", shard), ("config.json", runtime)):
            native.write(root / name, value)
        native.write(root / "PROVENANCE.json", {
            "argv": sys.argv, "hostname": socket.gethostname(), "at_utc": native.now(),
            "source_hashes": source["source_hashes"], "authority_sha256": authority["authority_sha256"],
            "registry_sha256": registry["registry_sha256"], "seed": 9401,
            "test_population_exposure": EXPOSURE, "reuse_old_test_outcomes": False,
            "git_head": core.repair.capture(["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT),
            "git_status": core.repair.capture(["git", "--no-pager", "status", "--porcelain",
                                              "--untracked-files=no"], cwd=ROOT),
            "actor_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
            "model_packages": core.repair.capture([runtime["model"]["python"], "-m", "pip", "freeze"]),
        })
        native.write(root / "PRELAUNCH_RESOURCE_AUDIT.json", native.resource_audit())
        adapter = native.read(prepared_root / "NATIVE_ADAPTER_SOURCE.json")
        verify_test_assets(runtime, protocol, adapter, expected, root)
        for directory in runtime["resource"]["lease_roots"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        inherited = descriptors if lease_descriptors is not None else None
        descriptors = []
        process, descriptors, logfile = core.start_service(root, runtime, lease_descriptors=inherited)
        endpoint = f"http://127.0.0.1:{runtime['resource']['port']}/v1"
        core.await_service(process, endpoint, runtime["startup_timeout_seconds"])
        policies = {name: registry["methods"][name]["policy"] for name in METHODS}
        freeze = {"policies": policies}
        records = {name: [] for name in METHODS}
        costs = {}
        for method in METHODS:
            method_root = root / method
            method_root.mkdir()
            method_roots[method] = method_root
            method_runtime = {**deepcopy(runtime), "method": method, "run_directory": str(method_root)}
            manifest = {"schema_version": core.repair.SCHEMA, "budget": common.BUDGET,
                        "alfworld_environment_mode": "eval_out_of_distribution", "seed": 20260928,
                        "task_inventory": expected, "scope": "test",
                        "authority_sha256": authority["authority_sha256"]}
            manifest["manifest_sha256"] = native.digest(manifest)
            native.write(method_root / "config.json", method_runtime)
            native.write(method_root / "manifest.json", manifest)
            native.write(method_root / "usage-ledger.json", common.new_ledger())
            binding = {"hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
                       "adapter_source_hashes": source["source_hashes"],
                       "model": {"id": runtime["model"]["served_name"], "revision": native.REVISION,
                                 "endpoints": [endpoint]},
                       "data_root": runtime["data_root"], "skillopt_root": runtime["skillopt_root"],
                       "gpu": {"visible_devices": "", "service_selection": str(root / "GPU_SELECTION.json")},
                       "ledger": str(method_root / "usage-ledger.json"), "max_model_len": 32768}
            binding["binding_sha256"] = native.digest(binding)
            native.write(method_root / "binding.json", binding)
        for phase, selected in (("test_gate", expected[:1]), ("test", expected[1:])):
            for method in shard["method_order"]:
                method_root = method_roots[method]
                stage = {"stage": phase, "jobs": [{"method": method, "item": item} for item in selected]}
                native.run_stage(method_root, runtime, freeze, stage, worker_initializer=initializer_for(method))
                records[method].extend(common.read_rows(method_root, phase))
                audited_items = expected[:1] if phase == "test_gate" else expected
                costs[method] = method_audit(method_root, method, records[method], audited_items, policies)
            if phase == "test_gate":
                native.write(root / "FIRST_PAIRED_GATE.json", {
                    "status": "PASS_FOUR_NATIVE_ACTORS_AND_ACCOUNTING", "tasks": 1, "methods": list(METHODS),
                    "success_required": False, "included_once_in_assigned_test_tasks": True, "costs": costs})
        rows = [row for method in METHODS for row in records[method]]
        for name, checksum in source["source_hashes"].items():
            native.require(native.file_hash(ROOT / name) == checksum, "Authorized source changed during test")
        result = {
            "schema_version": SCHEMA, "status": "COMPLETE_PAIRED_TEST_SHARD",
            "registry_sha256": registry["registry_sha256"], "authority_sha256": authority["authority_sha256"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "prepared_sha256": prepared["prepared_sha256"], "shard_sha256": shard["shard_sha256"],
            "shard_index": index, "tasks": len(expected), "episodes": len(rows),
            "summary": native.summarize(rows), "costs": costs, "node": node,
            "policy_hashes": {name: policies[name]["harness_sha256"] for name in METHODS},
            "phase_row_hashes": {
                f"{method}/{phase}-rows.jsonl": native.file_hash(root / method / f"{phase}-rows.jsonl")
                for method in METHODS for phase in ("test_gate", "test")},
            "method_ledger_sha256": {
                method: native.file_hash(method_roots[method] / "usage-ledger.json") for method in METHODS},
            "gpu_selection_sha256": native.file_hash(root / "GPU_SELECTION.json"),
            "test_population_exposure": EXPOSURE, "reuse_old_test_outcomes": False,
            "paper_admitted": False, "performance_guarantee_established": False,
        }
        result["result_sha256"] = native.digest(result)
        native.write(root / "RESULT.json", result)
    except BaseException as error:
        native.write(root / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": native.now(),
            "method_ledgers": {name: native.read(path / "usage-ledger.json")
                               for name, path in method_roots.items() if (path / "usage-ledger.json").exists()},
            "authority_sha256": authority["authority_sha256"], "automatic_retries": 0,
            "failed_attempts_preserved": True, "unknown_usage_not_zero_imputed": True})
        raise
    finally:
        common.stop_owned_service(process, descriptors, logfile)
        native.write(root / "RETIREMENT.json", {
            "at_utc": native.now(), "owned_service_pid": process.pid if process else None,
            "service_exit_code": process.returncode if process else None,
            "owned_lease_descriptors_closed": len(descriptors), "unrelated_processes_stopped": 0})


def aggregate_results(prepared, registry, shard_results):
    native.require(len(shard_results) == len(prepared["shards"])
                   and {row["shard_index"] for row in shard_results}
                   == set(range(len(prepared["shards"]))), "Missing or duplicate test shards")
    summaries = {method: {"tasks": 0, "successes": 0, "calls": 0, "turns": 0,
                          **dict.fromkeys(common.USAGE_KEYS, 0)} for method in METHODS}
    for result in shard_results:
        native.verify_signed(result, "result_sha256")
        index = result["shard_index"]
        native.require(result["status"] == "COMPLETE_PAIRED_TEST_SHARD"
                       and result["registry_sha256"] == registry["registry_sha256"]
                       and result["prepared_sha256"] == prepared["prepared_sha256"]
                       and result["authority_sha256"] == prepared["authority_sha256"]
                       and result["source_manifest_sha256"] == prepared["source_manifest_sha256"]
                       and result["shard_sha256"] == prepared["shards"][index]["shard_sha256"]
                       and result["node"] == prepared["shards"][index]["node"]
                       and result["policy_hashes"] == {
                           name: registry["methods"][name]["policy"]["harness_sha256"] for name in METHODS}
                       and result["test_population_exposure"] == EXPOSURE
                       and result["reuse_old_test_outcomes"] is False,
                       "Test shard identity, policy or exposure differs")
        native.require(set(result["summary"]) == set(result["costs"]) == set(METHODS),
                       "An undeclared method entered the result")
        for method, total in summaries.items():
            summary = result["summary"][method]
            native.require(summary["tasks"] == result["tasks"]
                           and result["episodes"] == 4 * result["tasks"]
                           and all(type(summary[key]) is int and summary[key] >= 0 for key in total)
                           and summary["successes"] <= summary["tasks"]
                           and summary["calls"] == summary["turns"]
                           and summary["prompt_tokens"] + summary["completion_tokens"]
                           == summary["total_tokens"]
                           and result["costs"][method]["all_in_calls"] == summary["calls"]
                           and result["costs"][method]["all_in_total_tokens"] == summary["total_tokens"],
                           "Unpaired method/task coverage or incomplete costs")
            for key in total:
                total[key] += summary[key]
    for total in summaries.values():
        native.require(total["tasks"] == 2133, "A method lacks the complete2133-task test")
        total["native_success_rate"] = total["successes"] / total["tasks"]
    return summaries


def collect(prepared_root, authority_path, runs_root, output):
    config, registry, source, authority, prepared = authorized_prepared(prepared_root, authority_path)
    native.remote_only(socket.gethostname())
    native.require(not output.exists(), "Never overwrite a test collection")
    results = []
    covered = {}
    for shard in prepared["shards"]:
        root = runs_root / Path(shard["run_directory"]).name
        result = native.read(root / "RESULT.json")
        assignment = native.read(prepared_root / f"shard-{shard['shard_index']:02d}.json")
        native.verify_signed(assignment, "shard_sha256")
        native.require(assignment["shard_sha256"] == shard["shard_sha256"], "Assignment changed")
        native.require(not set(covered) & {item["id"] for item in assignment["items"]},
                       "A test task was executed in more than one paired shard")
        covered.update({item["id"]: item["gamefile"] for item in assignment["items"]})
        for method in METHODS:
            rows = []
            for phase in ("test_gate", "test"):
                name = f"{method}/{phase}-rows.jsonl"
                native.require(native.file_hash(root / name) == result["phase_row_hashes"][name],
                               "Saved test rows changed after shard completion")
                rows.extend(common.read_rows(root / method, phase))
            validation.validate_rows(rows, assignment["items"], method, config["policy_hashes"][method])
            native.require(all("execution_reuse" not in row
                               and Path(row["artifact_dir"]).is_relative_to(
                                   Path(shard["run_directory"]) / method) for row in rows),
                           "Old or foreign outcomes cannot enter the first new test")
            native.require(native.summarize(rows)[method] == result["summary"][method],
                           "Shard summary differs from its complete saved rows")
            ledger_path = root / method / "usage-ledger.json"
            native.require(native.file_hash(ledger_path) == result["method_ledger_sha256"][method],
                           "Completed method ledger changed")
            ledger = native.read(ledger_path)
            native.require(ledger["cost_unknown"] is False and ledger["reserved_tokens"] == 0
                           and ledger["used_tokens"] == result["summary"][method]["total_tokens"]
                           and ledger["attempts"] == ledger["accounted_attempts"]
                           == result["summary"][method]["calls"], "Unsettled or mismatched method ledger")
        native.require(native.file_hash(root / "GPU_SELECTION.json") == result["gpu_selection_sha256"],
                       "Physical GPU selection receipt changed")
        results.append(result)
    native.require(len(covered) == len(set(covered.values())) == 2133
                   and native.digest(sorted(covered.items())) == prepared["test_identity_sha256"],
                   "Collected test identities differ from the complete frozen population")
    summaries = aggregate_results(prepared, registry, results)
    value = {
        "schema_version": SCHEMA, "status": "COMPLETE_NEW_FOUR_METHOD_TEST",
        "registry_sha256": registry["registry_sha256"], "authority_sha256": authority["authority_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "prepared_sha256": prepared["prepared_sha256"], "summary": summaries,
        "test_population_exposure": EXPOSURE, "reuse_old_test_outcomes": False,
        "optimizer_costs_remain_separate": True, "paper_admitted": False,
        "performance_guarantee_established": False,
        "shard_result_sha256": [row["result_sha256"] for row in results],
    }
    value["result_sha256"] = native.digest(value)
    native.write(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("source", "prepare"):
        sub = commands.add_parser(name)
        for flag in ("config", "registry", "gate", "output"):
            sub.add_argument("--" + flag, type=Path, required=True)
        if name == "prepare":
            sub.add_argument("--source-freeze", type=Path, required=True)
            sub.add_argument("--authority", type=Path, required=True)
    for name in ("worker", "collect"):
        sub = commands.add_parser(name)
        sub.add_argument("--prepared", type=Path, required=True)
        sub.add_argument("--authority", type=Path, required=True)
        if name == "worker":
            sub.add_argument("--shard", type=int, required=True)
            sub.add_argument("--lease-fds", type=int, nargs="*")
        else:
            sub.add_argument("--runs-root", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source":
        result = source_manifest(args.config, args.registry, args.gate, args.output)
    elif args.command == "prepare":
        result = prepare(args.config, args.registry, args.gate, args.source_freeze, args.authority, args.output)
    elif args.command == "worker":
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"Received signal {signum}; preserve this new attempt")
        signal.signal(signal.SIGTERM, interrupted)
        worker(args.prepared, args.authority, args.shard, args.lease_fds)
        return
    else:
        result = collect(args.prepared, args.authority, args.runs_root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key in (
        "kind", "status", "source_manifest_sha256", "prepared_sha256", "result_sha256")}), flush=True)


if __name__ == "__main__":
    main()
