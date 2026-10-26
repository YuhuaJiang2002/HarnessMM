#!/usr/bin/env python3
"""Run the bounded 4B QA diagnostic while holding every shared physical-GPU guard."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback

from scripts import cvpr_execution_crossover_leases as guard
from scripts import run_compact_harness_refinement as runner
from scripts.prepare_qa_lossless_native import HOST, GPU_UUID, REPOSITORY, REVISION


GUARD_ROOT = Path("/opt/phiagent/runs/ql0918g1")
FORMAT_GUARD_ROOT = Path("/opt/phiagent/runs/ql0918g2")
COMPRESSION_GUARD_ROOT = Path("/opt/phiagent/runs/cp0918g1")
PRIORITY_COMPRESSION_GUARD_ROOT = Path("/opt/phiagent/runs/cp0918g2")
COMPRESSION_HOST = "harnessmm-worker-1"
COMPRESSION_GPU_UUID = "GPU-6963b4db-f9a5-8be6-941c-90e28b372751"
LEGACY_ROOT = Path("/tmp/phiagent-gpu-leases")


def validate_scope(
    config: dict, campaign: Path, *, format_successor: bool = False, profile_compression: bool = False,
) -> None:
    if format_successor and profile_compression:
        raise ValueError("Independent campaigns cannot be composed by launch flags")
    resource, model, budget = config["resource"], config["model"], config["budget"]
    host, index, uuid = (
        (COMPRESSION_HOST, 3, COMPRESSION_GPU_UUID) if profile_compression else (HOST, 0, GPU_UUID)
    )
    repository, revision = (
        ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
        if profile_compression else (REPOSITORY, REVISION)
    )
    expected_cap = 2048 if format_successor else 512
    compression_guard = (
        PRIORITY_COMPRESSION_GUARD_ROOT
        if config.get("schema_version") == "phiagent-qa-profile-compression/v2"
        else COMPRESSION_GUARD_ROOT
    )
    expected_guard = compression_guard if profile_compression else (
        FORMAT_GUARD_ROOT if format_successor else GUARD_ROOT
    )
    maximum_seconds = (
        3500 if config.get("schema_version") == "phiagent-qa-profile-compression/v2" else 5400
    ) if profile_compression else (7200 if format_successor else 6600)
    if (
        resource["selected_hostname"] != host or resource["physical_gpu_index"] != index
        or resource["gpu_uuid"] != uuid or resource["hardware_name"] != "H200"
        or model["repository"] != repository or model["revision"] != revision
        or model["max_output_tokens"] != expected_cap or model["request_seed"] != 20260918
        or model["temperature"] != 0 or model["enable_thinking"] is not False
        or resource["lease_root"] != str(campaign / "native-locks")
        or resource.get("outer_physical_guard_root") != str(expected_guard)
        or budget["maximum_physical_gpus"] != 1
        or not 180 < budget["maximum_allocation_seconds"] <= maximum_seconds
        or not 0 < budget["maximum_real_request_attempts"] <= (5000 if profile_compression else 4096)
        or not 0 < budget["maximum_total_provider_tokens"] <= (20000000 if profile_compression else 16000000)
        or budget["transport_retries"] != 0
        or resource["binding_status"] != "BOUND_AFTER_FRESH_AUDIT"
    ):
        raise ValueError("Frozen model, private inner locks, shared outer guard, or resource ceilings differ")


def discover_roots() -> list[str]:
    result = subprocess.run([
        "find", "/opt/phiagent", "-xdev", "-type", "d", "-iname", "*gpu*lease*",
    ], text=True, capture_output=True, check=True, timeout=120)
    roots = set(guard.LEASE_ROOTS) | set(result.stdout.splitlines())
    for value in roots:
        path = Path(value)
        if (
            not path.is_absolute() or path.is_symlink()
            or not path.resolve().is_relative_to("/opt/phiagent")
            or (path.exists() and not path.is_dir())
        ):
            raise ValueError("Missing or unexplained GPU lease namespace")
    return sorted(roots)


def legacy_history(raw: str) -> list[dict]:
    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        pieces = line.split(" ")
        if len(pieces) != 2 or not pieces[0].startswith("pid=") or not pieces[1].startswith("started="):
            raise ValueError("Unrecognized legacy flock history; do not discard its contents")
        pid_text = pieces[0][4:]
        if not pid_text.isascii() or not pid_text.isdecimal() or pid_text.startswith("0"):
            raise ValueError("Invalid legacy flock history PID")
        timestamp = pieces[1][8:]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Legacy flock history timestamp requires a time zone")
        records.append({"pid": int(pid_text), "started": timestamp})
    return records


def legacy_aliases(root: Path, index: int, physical_uuid: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Legacy flock namespace is missing or symbolic")
    path = root / f"physical-gpu-{index}.lock"
    unexpected = [
        item for item in root.iterdir()
        if guard.selected_alias(item.name, index, physical_uuid) and item != path
    ]
    if unexpected:
        raise ValueError("Unexplained additional aliases in the legacy flock namespace")
    return path


def acquire_legacy_flock(root: Path, index: int, physical_uuid: str, stack: ExitStack) -> dict:
    path = legacy_aliases(root, index, physical_uuid)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    stream = stack.enter_context(os.fdopen(fd, "rb"))
    identity = os.fstat(fd)
    if not stat.S_ISREG(identity.st_mode):
        raise ValueError("Legacy physical-GPU lock is not a regular file")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise guard.GuardBusy(f"Legacy physical-GPU flock is held: {path}") from error
    raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Legacy physical-GPU history exceeds its inspection bound")
    records = legacy_history(raw.decode("utf-8"))
    # Legacy consumers arbitrate this append-only history by flock, not by its past PIDs.
    return {
        "path": str(path), "device": identity.st_dev, "inode": identity.st_ino,
        "history_sha256": hashlib.sha256(raw).hexdigest(), "history_records": records,
        "exclusive_flock_held": True, "historical_bytes_modified": False,
        "historical_pids_are_not_a_current_reservation": True,
        "contract_reference": "scripts/run_ewmbench_pilot_metrics.py: physical-GPU a+ flock lifetime",
    }


def verify_legacy_binding(record: dict, root: Path, index: int, physical_uuid: str) -> None:
    path = legacy_aliases(root, index, physical_uuid)
    identity = path.stat()
    if (
        str(path) != record["path"] or identity.st_dev != record["device"]
        or identity.st_ino != record["inode"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != record["history_sha256"]
    ):
        raise guard.GuardBusy("Legacy physical-GPU lock identity or history changed while held")


def parse_snapshot(snapshot: dict, index: int) -> tuple[str, str, dict]:
    raw, compute = snapshot["outputs"]
    rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6 or int(rows[0][0]) != index:
        raise ValueError("Selected GPU telemetry is incomplete")
    row = rows[0]
    device = {
        "index": index, "uuid": row[1], "name": row[2],
        "memory_used_mib": float(row[3]), "memory_total_mib": float(row[4]),
        "utilization_percent": float(row[5]),
    }
    if (
        any(not math.isfinite(device[key]) for key in (
            "memory_used_mib", "memory_total_mib", "utilization_percent",
        ))
        or not 0 <= device["memory_used_mib"] <= device["memory_total_mib"]
        or device["memory_total_mib"] <= 0 or not 0 <= device["utilization_percent"] <= 100
    ):
        raise ValueError("Selected GPU telemetry is nonfinite or outside physical bounds")
    return raw, compute, {index: device}


def wait_for_strict_idle(query, index: int, physical_uuid: str, *, maximum_seconds: float = 60) -> None:
    deadline = time.monotonic() + maximum_seconds
    consecutive = 0
    while True:
        try:
            guard.verify_idle(query(), index, physical_uuid)
        except guard.GuardBusy:
            consecutive = 0
        else:
            consecutive += 1
            if consecutive == 2:
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise guard.GuardBusy("GPU did not satisfy two unchanged strict idle checks before the deadline")
        time.sleep(min(2, remaining))


def run(
    campaign: Path, config_path: Path, *, format_successor: bool = False, profile_compression: bool = False,
) -> None:
    host, index, uuid = (
        (COMPRESSION_HOST, 3, COMPRESSION_GPU_UUID) if profile_compression else (HOST, 0, GPU_UUID)
    )
    if socket.gethostname() != host or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("The guarded QA supervisor requires its authorized CPU-only H200-2 or H200-1")
    config = runner.read_json(config_path)
    validate_scope(config, campaign, format_successor=format_successor, profile_compression=profile_compression)
    protocol = runner.mixture_helpers(config)
    expected_protocol = "scripts.qa_profile_compression_protocol" if profile_compression else (
        "scripts.qa_lossless_format_protocol" if format_successor
        else "scripts.qa_lossless_routing_protocol"
    )
    if protocol is None or protocol.__name__ != expected_protocol:
        raise ValueError("Only the explicitly selected, validated opt-in diagnostic is authorized")
    if format_successor:
        protocol.verify_prior_debits(config)
    if profile_compression:
        protocol.verify_predecessor(config)
    compression_guard = (
        PRIORITY_COMPRESSION_GUARD_ROOT
        if config.get("schema_version") == "phiagent-qa-profile-compression/v2"
        else COMPRESSION_GUARD_ROOT
    )
    guard_root = compression_guard if profile_compression else (
        FORMAT_GUARD_ROOT if format_successor else GUARD_ROOT
    )
    allocation_seconds = config["budget"]["maximum_allocation_seconds"] if (
        format_successor or profile_compression
    ) else 7200
    frozen = runner.read_json(campaign / "SOURCE_FROZEN.json")
    if frozen["config_sha256"] != runner.sha(config_path):
        raise ValueError("The experiment configuration changed after source/data freezing")
    guard_root.mkdir(parents=True, exist_ok=False)
    native_locks = campaign / "native-locks"
    native_locks.mkdir(exist_ok=False)
    runner.write_json(guard_root / "STARTED.json", {
        "at_utc": runner.now(), "argv": sys.argv, "hostname": socket.gethostname(),
        "pid": os.getpid(), "config_sha256": runner.sha(config_path),
        "source_sha256": runner.sha(Path(__file__)),
        "guard_helper_sha256": runner.sha(Path(guard.__file__)),
        "source_frozen_sha256": runner.sha(campaign / "SOURCE_FROZEN.json"),
        "maximum_total_guard_seconds": allocation_seconds,
        "format_successor": format_successor,
        "profile_compression": profile_compression,
        "private_inner_locks_replace_no_shared_physical_guard": True,
    })
    query_number = 0

    def query():
        nonlocal query_number
        query_number += 1
        path = guard_root / f"gpu-observation-{query_number:03d}.json"
        runner.gpu_snapshot(config, path, require_idle=False)
        return parse_snapshot(runner.read_json(path), index)

    def deadline_reached(_number, _frame):
        raise RuntimeError("The cumulative QA GPU guard deadline reached its cleanup reserve")

    lease = None
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, deadline_reached)
    with ExitStack() as stack:
        try:
            reservation_started = time.monotonic()
            legacy = acquire_legacy_flock(LEGACY_ROOT, index, uuid, stack)
            runner.write_json(guard_root / "LEGACY_FLOCK_OWNED.json", legacy)
            roots = discover_roots()
            paths = guard.discover_paths(index, uuid, roots)
            runner.write_json(guard_root / "ROOT_INVENTORY.json", {
                "at_utc": runner.now(), "roots": roots,
                "aliases": [str(path) for path in paths],
                "separate_legacy_flock_namespace": str(LEGACY_ROOT),
                "legacy_history_preserved": True,
                "no_notifications_or_owner_messages_assumed": True,
            })
            lease = guard.acquire_gpu_guard(
                index=index, physical_uuid=uuid, authorized={index: uuid},
                hostname=host, run=campaign, service=guard_root, stack=stack,
                query_gpu=query, port=config["resource"]["port"], allocation_seconds=allocation_seconds,
                successor_task_id="independent-compression-confirmation" if profile_compression
                else "format-and-cap-controls", roots=roots,
            )
            remaining = allocation_seconds - (time.monotonic() - reservation_started) - 180
            if remaining <= 0:
                raise RuntimeError("Resource preparation consumed the cumulative guard budget")
            signal.setitimer(signal.ITIMER_REAL, remaining)
            if discover_roots() != roots or guard.discover_paths(index, uuid, roots) != paths:
                raise guard.GuardBusy("Lease namespaces changed during the atomic reservation")
            verify_legacy_binding(legacy, LEGACY_ROOT, index, uuid)
            lease.record.update({
                "required_CUDA_VISIBLE_DEVICES": str(index),
                "lease_paths": [str(path) for path in lease.paths],
                "private_inner_lock_directory": str(native_locks),
                "all_shared_physical_guards_held": True,
                "legacy_flock_receipt": str(guard_root / "LEGACY_FLOCK_OWNED.json"),
            })
            lease.write(lease.record)
            runner.write_json(guard_root / "OWNED.json", lease.record)
            print(json.dumps({
                "status": "ALL_PHYSICAL_GUARDS_OWNED_STARTING_BOUNDED_QA",
                "host": host, "physical_gpu": index, "gpu_uuid": uuid,
                "campaign": str(campaign), "guard_root": str(guard_root),
            }), flush=True)
            runner.run_campaign(campaign, config_path)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
            runner.write_json(guard_root / "FAILED.json", {
                "at_utc": runner.now(), "error": traceback.format_exc(),
                "owned_guard_acquired": lease is not None,
                "failure_does_not_reset_campaign_budget": True,
            })
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
            if lease is not None:
                wait_for_strict_idle(query, index, uuid)
                retirement = {
                    **lease.record, "owner": None, "last_owner": lease.record["owner"],
                    "status": "RETIRED", "retired_at_unix": time.time(),
                    "allocation_seconds": time.monotonic() - reservation_started,
                    "canonical_guard_seconds": time.monotonic() - lease.allocated_monotonic,
                    "allocation_includes_legacy_guard_and_native_preparation": True,
                    "native_retirement_path": str(campaign / "RETIREMENT.json"),
                    "physical_gpu_verified_idle_before_unlock": True,
                }
                lease.write(retirement)
                runner.write_json(guard_root / "RETIRED.json", retirement)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--format-successor", action="store_true")
    parser.add_argument("--profile-compression", action="store_true")
    args = parser.parse_args()
    run(args.campaign, args.config, format_successor=args.format_successor,
        profile_compression=args.profile_compression)


if __name__ == "__main__":
    main()
