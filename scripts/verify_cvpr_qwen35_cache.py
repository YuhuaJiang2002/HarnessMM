#!/usr/bin/env python3
"""Read-only remote verification of cached Qwen3.5 inference assets against a public pin."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_file(directory, entry):
    name = entry["name"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Cache verification accepts only explicit leaf filenames")
    if entry["kind"] not in {"sha256", "git_blob_sha1"}:
        raise ValueError("Unsupported public artifact hash kind")
    path = directory / name
    if not path.is_file():
        return {**entry, "status": "MISSING", "match": False}
    size = path.stat().st_size
    digest = hashlib.sha256() if entry["kind"] == "sha256" else hashlib.sha1(
        b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    match = size == entry["size"] and actual == entry["hash"]
    return {**entry, "actual_size": size, "actual_hash": actual, "match": match,
            "status": "MATCH" if match else "MISMATCH"}


def vision_headers(directory, entries):
    result = {}
    for entry in entries:
        if not entry["name"].endswith(".safetensors") or not entry["match"]:
            continue
        with (directory / entry["name"]).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= 64 * 1024 * 1024:
                raise ValueError("Invalid bounded safetensors header length")
            header = json.loads(stream.read(size))
        names = [name for name in header if name.startswith(("model.visual.", "visual."))]
        result[entry["name"]] = {"vision_tensor_count": len(names), "example_names": names[:3]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--git-head", type=Path, required=True)
    parser.add_argument("--git-status", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["root"]).resolve()
    if (socket.gethostname() != config["hostname"] or config["hostname"] != "harnessmm-worker-2" or
            os.environ.get("CUDA_VISIBLE_DEVICES") != "" or
            root.parent != Path("/opt/phiagent/runs/cvpr-second-backbone")):
        raise RuntimeError("Cache integrity requires a new authorized H200-2 CPU directory")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, root / "config.json")
    shutil.copyfile(__file__, root / "verify_cvpr_qwen35_cache.py")
    shutil.copyfile(args.git_head, root / "git-head.txt")
    shutil.copyfile(args.git_status, root / "git-status.txt")
    began = time.monotonic()
    save(root / "STARTED.json", {
        "argv": sys.argv, "hostname": socket.gethostname(), "at_unix": time.time(),
        "python": sys.version, "pid": os.getpid(), "randomness": "none",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_calls": 0, "evaluator_calls": 0, "gpu_allocations": 0,
    })
    try:
        names = [entry["name"] for entry in config["files"]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate target asset filenames")
        directory = Path(config["model_root"])
        results = []
        for entry in config["files"]:
            if time.monotonic() - began > config["maximum_seconds"]:
                raise TimeoutError("Bounded read-only cache verification deadline reached")
            result = verify_file(directory, entry)
            results.append(result)
            with (root / "observations.jsonl").open("a") as stream:
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(result, sort_keys=True), flush=True)
        headers = vision_headers(directory, results)
        python = str(Path(config["runtime_root"]) / "bin/python")
        command = [python, "-c", "import importlib.metadata,json,sys; print(json.dumps("
                   "{'python':sys.version,'packages':{name:importlib.metadata.version(name) "
                   "for name in ['torch','transformers','vllm']}}))"]
        versions = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
        save(root / "runtime-command.json", {"argv": command, "CUDA_VISIBLE_DEVICES": ""})
        save(root / "runtime-versions.json", json.loads(versions.stdout))
        inference_match = all(row["match"] for row in results if row["inference"])
        result = {
            "status": "CACHE_INFERENCE_ASSETS_MATCH_TARGET" if inference_match else "CACHE_HASH_MISMATCH",
            "root": str(root), "target_revision": config["target_revision"],
            "previously_declared_cache_revision": config["previously_declared_cache_revision"],
            "metadata_url": config["metadata_url"], "files": results, "headers": headers,
            "all_repository_files_match": all(row["match"] for row in results),
            "all_inference_files_match": inference_match,
            "elapsed_seconds": time.monotonic() - began,
            "model_cache_modified": False, "native_execution_tested": False,
            "model_calls": 0, "evaluator_calls": 0, "gpu_allocations": 0,
        }
        save(root / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "root": str(root),
                          "mismatches": [row["name"] for row in results if not row["match"]]},
                         sort_keys=True), flush=True)
        return 0 if inference_match else 1
    except BaseException:
        save(root / "FAILED.json", {"at_unix": time.time(), "error": traceback.format_exc(),
                                   "model_calls": 0, "gpu_allocations": 0})
        raise


if __name__ == "__main__":
    sys.exit(main())
