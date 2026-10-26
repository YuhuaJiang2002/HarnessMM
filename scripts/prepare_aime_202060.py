"""Prepare a new isolated AIME 6/6/18 run without opening the sealed test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def open_frozen_test(partition_root, freeze_root, manifest_sha256, *, read_json=read, sha_file=sha):
    from phiagent.harness.aime_202060 import validate_policy_freeze

    frozen_path = freeze_root / "POLICY_FREEZE.json"
    frozen = read_json(frozen_path)
    proof = read_json(freeze_root / "VERIFIED.json")
    validate_policy_freeze(frozen, manifest_sha256)
    if proof["status"] != "PASS" or proof["policy_freeze_sha256"] != sha_file(frozen_path):
        raise RuntimeError("Final policy freeze audit failed before opening test")
    for bound in frozen["policies"].values():
        if sha_file(Path(bound["path"])) != bound["policy_sha256"]:
            raise RuntimeError("An all-method policy changed before test access")
    return frozen, read_json(partition_root / "SEALED_TEST.json")["test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=[
        "train_baseline", "skillopt", "autosaddler", "no_skill_validation",
        "chisel_learn", "chisel_train", "chisel_validation",
        "final",
    ], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--shard", type=int, choices=[0, 1])
    parser.add_argument("--baseline-root", type=Path, action="append", default=[])
    parser.add_argument("--library-root", type=Path)
    parser.add_argument("--freeze-root", type=Path)
    parser.add_argument("--method", choices=["no_skill", "skillopt", "autosaddler", "chisel"])
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--recover-learning-from", type=Path)
    parser.add_argument(
        "--partition-root", type=Path,
        default=Path("/opt/phiagent/work/df-202060/20260928-partition-v2/aime2025"),
    )
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-worker-2":
        raise RuntimeError("Only the authorized H200-2 runtime may prepare this campaign")
    if Path(args.name).name != args.name or not args.name.startswith("20260928-split202060-"):
        raise ValueError("Use a new 20260928-split202060-* run directory")
    if args.gpu not in (2, 3, 5, 6, 7):
        raise ValueError("This AIME campaign may lease only authorized GPUs2,3,5,6,7")
    if (args.mode in ("train_baseline", "chisel_train", "chisel_validation")) != (args.shard is not None):
        raise ValueError("Paired training/evaluation requires a predetermined even/odd shard")
    parent = Path("/opt/phiagent/work/aime-comparison")
    root = parent / args.name
    if root.exists():
        raise FileExistsError(root)
    bundle = Path(__file__).resolve().parents[1]
    manifest = read(bundle / "BUNDLE.json")
    for name, digest in manifest["files_sha256"].items():
        if sha(bundle / name) != digest:
            raise RuntimeError("Campaign source bundle drift: " + name)
    split_path = args.partition_root / "SPLIT_MANIFEST.json"
    if sha(split_path) != args.manifest_sha256:
        raise RuntimeError("Parent-verified split manifest changed")
    source = read(args.partition_root / "SOURCE.json")
    if source["manifest_sha256"] != args.manifest_sha256:
        raise RuntimeError("Parent partition provenance disagrees")
    inventory = {
        key: subprocess.check_output(command, text=True)
        for key, command in (
            ("gpus", ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader,nounits"]),
            ("processes", ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"]),
            ("owners", ["ps", "-eo", "user,pid,ppid,etimes,comm"]),
            ("leases", ["lslocks", "--json"]), ("memory", ["free", "-m"]), ("load", ["uptime"]),
        )
    }
    device = next(
        [part.strip() for part in line.split(",")] for line in inventory["gpus"].splitlines()
        if int(line.split(",")[0]) == args.gpu
    )
    if not args.wait_for_gpu and (int(device[2]) >= 1024 or device[1] in inventory["processes"]):
        raise RuntimeError("Requested GPU is not free; do not preempt existing work")
    template = parent / "20260928-aime-sk-v2"
    root.mkdir()
    shutil.copytree(
        template / "source", root / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
    )
    for name in manifest["files_sha256"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / name, target)
    sys.path.insert(0, str(root / "source"))
    from phiagent.harness.aime_202060 import validate_data
    from scripts.aime_client_admission import serial_configuration

    partition = read(split_path)
    data = read(args.partition_root / "TRAIN_VALIDATION.json")
    validate_data(data, partition)
    final_freeze = None
    if args.mode == "final":
        if args.freeze_root is None or not args.freeze_root.is_relative_to(parent) or args.method is None:
            raise ValueError("Final execution requires the all-method policy freeze and one method")
        final_freeze, data["test"] = open_frozen_test(
            args.partition_root, args.freeze_root, args.manifest_sha256,
        )
        validate_data(data, partition, final=True)
        shutil.copy2(args.freeze_root / "POLICY_FREEZE.json", root / "POLICY_FREEZE.json")
    for name in ("SPLIT_MANIFEST.json", "TRAIN_VALIDATION.json", "SOURCE.json"):
        shutil.copy2(args.partition_root / name, root / name)
    write(root / "DATA.json", data)
    old = read(template / "STARTED.json")["config"]
    config = {
        key: old[key] for key in (
            "runtime", "hostname", "model", "scorer_sha256", "scorer_revision", "scorer_repository",
        )
    }
    config.update(
        experiment_id=args.name, run_directory=str(root), mode=args.mode,
        seed=20260926, split_seed=20260928, manifest_sha256=args.manifest_sha256,
        benchmark_label="AIME2025 archived-public resplit train6 validation6",
        prior_exposure="Archived public resplit, not fresh unseen data",
        require_reproducible_inference=True, strict_optimizer_completion=True,
        optimizer_context_repair=True, optimizer_workspace_repair=True,
        optimizer_structured_finalization=True, recovery_session_retries=1,
        authorization="2026-09-28 user required common20/20/60 and training-grounded CHISEL improvement",
        resource={
            "preferred_gpu_indices": [args.gpu], "model_port": args.port,
            "capture_port": args.port + 1, "max_num_seqs": 1,
            "lease_root": "/opt/phiagent/gpu-leases", "gpu_memory_utilization": 0.8,
        },
        budget={
            "token_usage_mode": "unlimited", "maximum_tokens": None,
            "maximum_queue_seconds": 21600, "maximum_wall_seconds": 604800,
            "maximum_requests": 2000000, "transport_retries": 0,
            "unknown_usage_policy": "Preserve every uncertain call; block new selection on uncertainty",
        },
        datasets={"2025": old["datasets"]["2025"]},
        historical_cost_bindings=[],
    )
    serial_configuration(config)
    if args.mode.startswith("chisel_") or (args.mode == "final" and args.method == "chisel"):
        config["structured_output_backend"] = "xgrammar"
    if args.recover_learning_from is not None:
        prior = args.recover_learning_from.resolve()
        if args.mode != "chisel_learn" or not prior.is_relative_to(parent):
            raise ValueError("Only an authorized same-partition failed learner can be recovered")
        previous = read(prior / "STARTED.json")["config"]
        ledger_path = prior / "PROVIDER_FINAL.json"
        ledger = read(ledger_path)
        if (
            previous["mode"] != "chisel_learn"
            or previous["manifest_sha256"] != args.manifest_sha256
            or previous["model"] != config["model"] or previous["seed"] != config["seed"]
            or sha(prior / "DATA.json") != sha(root / "DATA.json")
            or not (prior / "FAILURE.json").exists()
            or ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]
        ):
            raise RuntimeError("Failed-learner recovery evidence is incomplete or mismatched")
        config["learning_recovery"] = {"root": str(prior), "ledger_sha256": sha(ledger_path)}
        config["historical_cost_bindings"] = [{
            "ledger_path": str(ledger_path), "sha256": sha(ledger_path),
            "purpose": "Preserved same-partition learner failure and valid diagnosis checkpoints",
        }]
    if final_freeze is not None:
        extension = ".json" if args.method == "chisel" else ".md"
        name = "selected_policy" + extension
        shutil.copy2(final_freeze["policies"][args.method]["path"], root / name)
        config.update(
            method=args.method, policy_freeze_sha256=sha(root / "POLICY_FREEZE.json"),
            selected_policy_file=name,
        )
    if args.mode == "train_baseline":
        config["assigned_train_ids"] = partition["splits"]["train"][args.shard::2]
        config["training_shards"] = {
            "count": 2, "index": args.shard, "method": "even/odd manifest-order positions",
            "union_is_complete_train6": True,
        }
    if args.mode == "chisel_learn":
        if len(args.baseline_root) != 2 or len(set(args.baseline_root)) != 2:
            raise ValueError("Constraint learning requires both distinct audited training shards")
        config["baseline_roots"] = []
        covered = []
        for location in args.baseline_root:
            if not location.is_relative_to(parent):
                raise ValueError("Baseline evidence must remain on the authorized server")
            proof, result = read(location / "VERIFIED.json"), read(location / "RESULT.json")
            if (
                proof["status"] != "PASS" or proof["result_sha256"] != sha(location / "RESULT.json")
                or result["manifest_sha256"] != args.manifest_sha256
                or result["mode"] != "train_baseline"
            ):
                raise RuntimeError("Training baseline gate failed")
            covered.extend(result["assigned_ids"])
            config["baseline_roots"].append({
                "root": str(location), "result_sha256": sha(location / "RESULT.json"),
                "verified_sha256": sha(location / "VERIFIED.json"),
            })
        if len(covered) != 6 or set(covered) != set(partition["splits"]["train"]):
            raise RuntimeError("Training shard union is not the complete frozen six")
    if args.mode in ("chisel_train", "chisel_validation"):
        location = args.library_root
        if location is None or not location.is_relative_to(parent):
            raise ValueError("An audited training-derived candidate library is required")
        proof, result = read(location / "VERIFIED.json"), read(location / "RESULT.json")
        if (
            proof["status"] != "PASS" or proof["result_sha256"] != sha(location / "RESULT.json")
            or result["manifest_sha256"] != args.manifest_sha256 or result["mode"] != "chisel_learn"
        ):
            raise RuntimeError("Training-only library audit failed")
        path = location / "POLICY_LIBRARY.json"
        config["library"] = {"path": str(path), "sha256": sha(path), "verified_sha256": sha(location / "VERIFIED.json")}
        split_name = "train" if args.mode == "chisel_train" else "validation"
        config["assigned_ids"] = partition["splits"][split_name][args.shard::2]
    gate = read(template / "REPRODUCIBILITY_GATE.json")
    helper = "scripts/aime_reproducible_runtime.py"
    if (
        gate["remote_acceptance"]["status"] != "PASS"
        or gate["remote_acceptance"]["target_repeats"] < 12
        or sha(root / "source" / helper) != gate["files_sha256"][helper]
    ):
        raise RuntimeError("Accepted deterministic runtime is missing or changed")
    shutil.copy2(template / "REPRODUCIBILITY_GATE.json", root / "REPRODUCIBILITY_GATE.json")
    config_path = root / "source/configs/aime_202060.json"
    config["_config_path"] = str(config_path)
    write(config_path, config)
    freeze = read(template / "SOURCE_FREEZE.json")
    for name in manifest["files_sha256"]:
        freeze["files_sha256"][name] = sha(root / "source" / name)
    freeze["files_sha256"]["configs/aime_202060.json"] = sha(config_path)
    for name in freeze["files_sha256"]:
        freeze["files_sha256"][name] = sha(root / "source" / name)
    freeze.update(
        git_state=manifest["git_state"], bundle_sha256=sha(bundle / "BUNDLE.json"),
        historical_results_reused=False, partition_source_sha256=sha(root / "SOURCE.json"),
    )
    write(root / "SOURCE_FREEZE.json", freeze)
    write(root / "PREPARED.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "dataset_hash": sha(root / "DATA.json"),
        "manifest_sha256": args.manifest_sha256,
        "counts": {"train": 6, "validation": 6, "test": 18 if args.mode == "final" else 0},
        "final_population_sealed": 18, "new_model_calls": 0, "old_results_reused": False,
        "historical_costs_inherited": False,
    })
    write(root / "RESOURCE_MANIFEST.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "selected_gpu": args.gpu,
        "gpu_uuid": device[1], "inventory": inventory, "both_node_snapshot": manifest["pool_snapshot"],
        "lease": "Canonical index+UUID locks acquired and rechecked before model startup",
        "successor": "Train-grounded CHISEL or frozen-policy validation after this run completes",
    })
    entrypoint = (
        "run_aime_final_202060.py" if args.mode == "final" else
        "run_aime_chisel_202060.py" if args.mode.startswith("chisel_") else "run_aime_202060.py"
    )
    command = [
        config["runtime"], "-u", str(root / "source/scripts" / entrypoint),
        "--config", str(config_path),
    ]
    work = root / "controller-work"
    work.mkdir()
    with (root / "controller.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=root / "source", stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(work), TMP=str(work), TEMP=str(work)),
        )
    write(root / "LAUNCH.json", {"pid": process.pid, "command": command, "gpu": args.gpu})
    print(json.dumps({"root": str(root), "pid": process.pid, "gpu": args.gpu, "mode": args.mode}))


if __name__ == "__main__":
    main()
