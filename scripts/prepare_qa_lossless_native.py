#!/usr/bin/env python3
"""Bind the existing exact 4B cache to a new CPU-only QA runtime manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

from scripts.compact_h200_runtime import digest
from scripts.verify_cvpr_qwen35_cache import verify_file


HOST = "harnessmm-worker-2"
REPOSITORY = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GATE_ROOT = Path("/opt/phiagent/runs/cvpr_second_backbone_gate_20260918v1")
GATE_SHA = "e80b2c543493b16f83a0bbb6646b7effefdd5fd993eece169ea0914d49f73ef4"
EXPECTED_SHA = "d53eab9ca8be6014d2ba04db13b83bf3edc68d75f4ece8eb2519e2d40f422971"
GPU_UUID = "GPU-fc1d96a1-dd3e-3e77-c18b-4ae819c7b5d1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require_remote() -> None:
    if socket.gethostname() != HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Native preparation requires the authorized CPU-only H200-2 host")


def runtime_manifest(model: Path, expected: dict) -> dict:
    if expected["repository"] != REPOSITORY or expected["target_revision"] != REVISION:
        raise ValueError("Native checkpoint identity differs from the prospective backbone")
    entries = expected["files"]
    names = [entry["name"] for entry in entries]
    if not entries or len(names) != len(set(names)):
        raise ValueError("Explicit unique native checkpoint filenames are required")
    verified = []
    for entry in entries:
        result = verify_file(model, entry)
        if result["match"] is not True:
            raise ValueError(f"Native cache differs from its public pin: {entry['name']}")
        path = model / entry["name"]
        if not path.resolve().is_relative_to(model.resolve()):
            raise ValueError("Native checkpoint file escapes its declared model root")
        verified.append({
            "path": entry["name"], "size": result["actual_size"],
            "sha256": result["actual_hash"] if entry["kind"] == "sha256" else digest(path),
            "public_hash_kind": entry["kind"], "public_hash": entry["hash"],
        })
    return {
        "canonical_repository": REPOSITORY,
        "canonical_huggingface_revision": REVISION,
        "files": verified,
        "scope": "Existing exact cache, rehashed for native text-only QA; no weights copied or loaded.",
    }


def prepare(output: Path, git_head: Path, git_status: Path) -> dict:
    require_remote()
    if output.resolve().parent != Path("/opt/phiagent/runs"):
        raise ValueError("Native preparation requires a new direct child of /opt/phiagent/runs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copyfile(git_head, output / "git-head.txt")
        shutil.copyfile(git_status, output / "git-status.txt")
        sources = [
            Path(__file__),
            Path(__file__).with_name("compact_h200_runtime.py"),
            Path(__file__).with_name("verify_cvpr_qwen35_cache.py"),
        ]
        source_hashes = {}
        for source in sources:
            shutil.copyfile(source, output / source.name)
            source_hashes[source.name] = digest(source)
        save(output / "STARTED.json", {
            "at_utc": now(), "argv": sys.argv, "hostname": socket.gethostname(),
            "python": sys.version, "pid": os.getpid(), "CUDA_VISIBLE_DEVICES": "",
            "randomness": "none; deterministic cache verification",
            "git_head_sha256": digest(output / "git-head.txt"),
            "git_status_sha256": digest(output / "git-status.txt"),
            "source_sha256": source_hashes,
            "packages": {
                item.metadata["Name"]: item.version
                for item in importlib.metadata.distributions() if item.metadata["Name"]
            },
            "model_calls": 0, "benchmark_or_evaluator_calls": 0, "gpu_allocations": 0,
        })
        gate_path = GATE_ROOT / "GATE.json"
        expected_path = GATE_ROOT / "source/cvpr_second_backbone_expected_manifest.json"
        if digest(gate_path) != GATE_SHA or digest(expected_path) != EXPECTED_SHA:
            raise ValueError("The prior cache gate or publisher manifest binding changed")
        gate = json.loads(gate_path.read_text())
        expected = json.loads(expected_path.read_text())
        if (
            gate["repository"] != REPOSITORY or gate["revision"] != REVISION
            or not gate["all_repository_files_match"] or len(expected["files"]) != 14
            or gate["model_root"] != expected["model_root"]
            or gate["runtime_root"] != expected["runtime_root"]
        ):
            raise ValueError("Exact native model/runtime source identity is incomplete")
        model = Path(gate["model_root"])
        manifest = runtime_manifest(model, expected)
        manifest.update(prior_gate_sha256=GATE_SHA, publisher_manifest_sha256=EXPECTED_SHA)
        manifest_path = output / "native-runtime-manifest.json"
        save(manifest_path, manifest)
        for item in gate["runtime_source_hashes"].values():
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("Pinned native model implementation or registry changed")
        for kind, fields in (
            ("gpu", "index,uuid,name,memory.used,memory.total,utilization.gpu"),
            ("compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory"),
        ):
            argv = ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"]
            observed = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30)
            save(output / f"{kind}-observation.json", {
                "at_utc": now(), "argv": argv, "output": observed.stdout,
                "read_only": True, "establishes_live_lease": False,
            })
        resource = {
            "binding_status": "CACHE_READY_REQUIRES_FRESH_GPU_AND_LEASE",
            "selected_host_alias": "h200-2", "selected_hostname": HOST,
            "physical_gpu_index": 0, "gpu_uuid": GPU_UUID, "hardware_name": "H200",
            "port": 19918, "lease_root": "/opt/phiagent/gpu-leases",
            "service_root": "/opt/phiagent/runs/ql0918s1",
            "fresh_GPU_and_exclusive_lease_required_before_launch": True,
            "no_other_study_service_reuse": True,
            "native_runtime": {
                "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
                "model_path": str(model),
                "python_executable": str(Path(gate["runtime_root"]) / "bin/python"),
                "native_driver_only": True, "a800_compatibility_paths_permitted": False,
                "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 3},
                "packages": gate["package_versions"],
            },
        }
        save(output / "RESOURCE.json", resource)
        result = {
            "status": "PINNED_QA_NATIVE_MANIFEST_READY_NOT_A_MODEL_RUN",
            "at_utc": now(), "hostname": HOST, "root": str(output),
            "repository": REPOSITORY, "revision": REVISION,
            "resource_sha256": digest(output / "RESOURCE.json"),
            "native_manifest_sha256": digest(manifest_path),
            "repository_files_verified": len(manifest["files"]),
            "model_calls": 0, "evaluator_calls": 0, "gpu_allocations": 0,
            "benchmark_inputs_read": False, "checkpoint_or_runtime_modified": False,
            "successor_task_id": "frozen-routing-controls",
        }
        save(output / "RESULT.json", result)
        return result
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
        save(output / "FAILED.json", {
            "at_utc": now(), "error": traceback.format_exc(),
            "model_calls": 0, "evaluator_calls": 0, "gpu_allocations": 0,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head", type=Path, required=True)
    parser.add_argument("--git-status", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.git_head, args.git_status), sort_keys=True))


if __name__ == "__main__":
    main()
