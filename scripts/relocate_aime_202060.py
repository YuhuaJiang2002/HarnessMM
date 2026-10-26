"""Retire one unused waiting unit and launch its immutable resource successor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

PARENT = Path("/opt/phiagent/work/aime-comparison")
MODES = {
    "train_baseline", "skillopt", "autosaddler", "no_skill_validation",
    "chisel_learn", "chisel_train", "chisel_validation", "final",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def can_retire(snapshot):
    required = {
        "alive", "matching_command", "waiting", "gpu_selected",
        "model_owner", "provider_calls", "provider_ledger", "result",
    }
    if not required.issubset(snapshot):
        raise ValueError("Incomplete retirement evidence")
    return (
        snapshot["alive"] and snapshot["matching_command"] and snapshot["waiting"]
        and not snapshot["gpu_selected"] and not snapshot["model_owner"]
        and snapshot["provider_calls"] == 0
        and not snapshot["provider_ledger"] and not snapshot["result"]
    )


def alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def snapshot(root, pid):
    running = alive(pid)
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes() if running else b""
    return {
        "alive": running, "matching_command": str(root / "source/scripts").encode() in command,
        "waiting": (root / "WAITING_FOR_GPU.json").exists(),
        "gpu_selected": (root / "GPU_SELECTION.json").exists(),
        "model_owner": (root / "service/OWNER.json").exists(),
        "provider_calls": len(list((root / "provider").glob("call-*"))),
        "provider_ledger": (root / "PROVIDER_LEDGER.json").exists(),
        "result": (root / "RESULT.json").exists(),
    }


def mode_arguments(config, root):
    mode = config["mode"]
    if mode == "train_baseline":
        return ["--shard", str(config["training_shards"]["index"])]
    if mode == "chisel_learn":
        return [part for binding in config["baseline_roots"]
                for part in ("--baseline-root", binding["root"])]
    if mode in ("chisel_train", "chisel_validation"):
        manifest = read(root / "SPLIT_MANIFEST.json")
        split = "train" if mode == "chisel_train" else "validation"
        matches = [index for index in (0, 1)
                   if manifest["splits"][split][index::2] == config["assigned_ids"]]
        if len(matches) != 1:
            raise RuntimeError("The frozen candidate shard no longer matches the partition")
        return [
            "--shard", str(matches[0]), "--library-root",
            str(Path(config["library"]["path"]).parent),
        ]
    if mode == "final":
        freeze = read(root / "POLICY_FREEZE.json")
        return [
            "--method", config["method"], "--freeze-root",
            str(Path(freeze["policies"]["no_skill"]["path"]).parent),
        ]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), default=2)
    args = parser.parse_args()
    if socket.gethostname() != "harnessmm-worker-2":
        raise RuntimeError("Resource migration is restricted to the authorized H200-2 host")
    old = args.old_root.resolve()
    if not old.is_relative_to(PARENT) or not old.name.startswith("20260928-split202060-"):
        raise ValueError("Refuse to retire a different campaign")
    if Path(args.name).name != args.name or not args.name.startswith("20260928-split202060-"):
        raise ValueError("Use a new immutable successor name")
    new = PARENT / args.name
    if old.name.rsplit("-v", 1)[0] != new.name.rsplit("-v", 1)[0]:
        raise ValueError("A resource successor must preserve the same logical unit")
    if new.exists():
        raise FileExistsError("Successor exists; never duplicate a running unit")
    config = read(old / "STARTED.json")["config"]
    if config["mode"] not in MODES or Path(config["run_directory"]).resolve() != old:
        raise ValueError("Unexpected waiting experiment")
    pid = read(old / "LAUNCH.json")["pid"]
    if type(pid) is not int or pid < 2:
        raise ValueError("Invalid owned process ID")
    bundle = read(args.bundle / "BUNDLE.json")
    if any(sha(args.bundle / name) != digest for name, digest in bundle["files_sha256"].items()):
        raise RuntimeError("Successor source bundle changed")
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    if int(gpu[2]) >= 128 or gpu[1].strip() in apps:
        raise RuntimeError(f"GPU{args.gpu} is occupied; predecessor remains untouched")
    before = snapshot(old, pid)
    if not can_retire(before):
        raise RuntimeError("Refuse to retire a unit with GPU allocation, calls, or results")
    bindings = {
        str(path): sha(path) for path in
        (old / "SOURCE_FREEZE.json", old / "DATA.json", Path(config["_config_path"]))
    }
    extra = mode_arguments(config, old)
    os.kill(pid, signal.SIGSTOP)
    try:
        if not can_retire(snapshot(old, pid)):
            raise RuntimeError("Unit reached resources while retirement was being checked")
        os.kill(pid, signal.SIGTERM)
    finally:
        os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + 45
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if alive(pid):
        raise RuntimeError("Predecessor has not retired; successor not launched")
    if any(sha(Path(path)) != digest for path, digest in bindings.items()):
        raise RuntimeError("Retirement changed frozen predecessor source or inputs")
    command = [
        "python3", str(args.bundle / "scripts/prepare_aime_202060.py"),
        "--name", args.name, "--mode", config["mode"], "--gpu", str(args.gpu),
        "--port", str(config["resource"]["model_port"]),
        "--manifest-sha256", config["manifest_sha256"],
        "--partition-root", "/opt/phiagent/work/df-202060/20260928-partition-v2/aime2025",
        "--wait-for-gpu", *extra,
    ]
    receipt = {
        "at_utc": datetime.now(timezone.utc).isoformat(), "predecessor": str(old),
        "predecessor_pid": pid, "predecessor_snapshot": before, "predecessor_retired": True,
        "frozen_predecessor_sha256": bindings, "successor": str(new), "command": command,
        "new_model_calls_in_predecessor": 0,
    }
    logs = PARENT / "20260928-split202060-relocation-events-v1"
    logs.mkdir(exist_ok=True)
    event = logs / (args.name + ".json")
    with event.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    event.write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("Successor preparation failed; preserved retirement evidence: " + str(event))
    successor = read(new / "source/configs/aime_202060.json")
    for key in ("mode", "model", "seed", "budget", "manifest_sha256", "assigned_train_ids", "assigned_ids"):
        if successor.get(key) != config.get(key):
            raise RuntimeError("Scientific configuration changed during resource-only migration: " + key)
    if sha(new / "DATA.json") != bindings[str(old / "DATA.json")]:
        raise RuntimeError("Resource migration changed the evaluation population")
    receipt["launch"] = json.loads(result.stdout)
    (new / "RESOURCE_SUCCESSION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
