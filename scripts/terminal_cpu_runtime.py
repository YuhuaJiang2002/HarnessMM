#!/usr/bin/env python3
"""Common CPU-executor environment invariant for remote-model Terminal clients."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys


CPU_RUNTIME = "terminal-remote-model-cpu-runtime/v1"


def cpu_child_environment(inherited=None):
    environment = dict(os.environ if inherited is None else inherited)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def require_cpu_executor(environment=None):
    environment = os.environ if environment is None else environment
    if environment.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("remote-model CPU executor requires explicit CUDA_VISIBLE_DEVICES=''")


def executor_provenance(model_host, environment=None):
    require_cpu_executor(environment)
    return {
        "cpu_runtime_invariant": CPU_RUNTIME,
        "local_CUDA_VISIBLE_DEVICES": "",
        "remote_model_host": dict(model_host),
        "remote_GPU_index_is_not_local_device_access": True,
    }


def attest(output):
    if platform.system() != "Linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("real-process CPU attestation runs only on authorized a800-4")
    output.mkdir(parents=True, exist_ok=False)
    before = dict(os.environ)
    inherited = {**before, "CUDA_VISIBLE_DEVICES": "7"}
    code = (
        "import json,os;"
        "print(json.dumps({'pid':os.getpid(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),"
        "'cwd':os.getcwd()}))"
    )
    argv = [sys.executable, "-c", code]
    completed = subprocess.run(argv, env=cpu_child_environment(inherited),
                               cwd=output, capture_output=True, text=True, timeout=15, check=True)
    child = json.loads(completed.stdout)
    report = {
        "schema_version": "phiagent-terminal-cpu-environment-attestation/v1",
        "at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "argv": argv, "simulated_inherited_CUDA_VISIBLE_DEVICES": "7", "child": child,
        "caller_environment_unchanged": dict(os.environ) == before,
        "child_exited": completed.returncode == 0, "child_has_empty_cuda": child["CUDA_VISIBLE_DEVICES"] == "",
        "model_environment_benchmark_or_replay_calls": 0,
    }
    report["passed"] = report["caller_environment_unchanged"] and report["child_exited"] and report["child_has_empty_cuda"]
    with (output / "ATTESTATION.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("CPU environment attestation failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attest-dir", type=Path, required=True)
    args = parser.parse_args()
    attest(args.attest_dir.resolve())


if __name__ == "__main__":
    main()
