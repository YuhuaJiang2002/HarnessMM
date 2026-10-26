#!/usr/bin/env python3
"""Reusable fail-closed physical-GPU guard; imports no model or benchmark code."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
import uuid as uuid_module


LEASE_ROOTS = (
    "/opt/phiagent/gpu-leases",
    "/opt/phiagent/runs/video-skillopt-gpu-leases",
    "/opt/phiagent/runs/video-skillopt/gpu-leases",
    "/opt/phiagent/runs/video-three-track-minimax-rv2v/gpu-leases",
)
RELEASED = {"RETIRED", "RELEASED", "FREE"}
REFERENCE_FIELDS = {"lease_path", "lease_paths", "lock_path", "lock_paths", "lock_file"}


class GuardBusy(RuntimeError):
    """A live owner, held lock, or nonidle physical GPU forbids allocation."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate lease key: {key}")
        result[key] = value
    return result


def identity_key(owner: dict | None):
    if owner is None:
        return None
    if type(owner.get("pid")) is not int or owner["pid"] <= 0:
        raise ValueError("Unknown legacy lease PID")
    ticks, boot = owner.get("start_time_ticks"), owner.get("boot_id")
    if ticks is not None and (type(ticks) is not int or ticks < 0):
        raise ValueError("Invalid process birth ticks")
    if boot is not None:
        if not isinstance(boot, str) or str(uuid_module.UUID(boot)) != boot.lower():
            raise ValueError("Invalid process boot identity")
    return (owner["pid"], ticks, boot.lower()) if ticks is not None and boot is not None else None


def parse_records(raw: str) -> list[dict]:
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    records, position = [], 0
    while position < len(raw):
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position == len(raw):
            break
        record, position = decoder.raw_decode(raw, position)
        if not isinstance(record, dict):
            raise ValueError("Unknown legacy lease record format")
        owner = record.get("owner", record)
        if owner is None:
            if record.get("status") not in RELEASED:
                raise ValueError("Unknown ownerless legacy lease state")
        elif not isinstance(owner, dict):
            raise ValueError("Unknown legacy lease owner identity")
        else:
            identity_key(owner)
        records.append(record)
    return records


def process_identity(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        status = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {
        "pid": pid, "state": fields[0], "ppid": int(fields[1]),
        "start_time_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uid": int(next(line for line in status.splitlines() if line.startswith("Uid:")).split()[1]),
        "hostname": socket.gethostname(),
    }


def active_owners(records: list[dict], observe=process_identity, hostname=None) -> list[dict]:
    expected_hostname = hostname or socket.gethostname()
    retired = set()
    result = []
    for record in reversed(records):
        owner = record.get("owner", record)
        if owner is None:
            last = record.get("last_owner")
            if isinstance(last, dict):
                key = identity_key(last)
                if key is not None and isinstance(record.get("run"), str):
                    retired.add((key, record["run"]))
            continue
        key = identity_key(owner)
        if key is not None and (key, record.get("run")) in retired:
            continue
        if owner.get("hostname", record.get("hostname")) not in (None, expected_hostname):
            raise ValueError("Lease owner belongs to a different/unknown host")
        current = observe(owner["pid"])
        if current is None or current["state"] == "Z":
            continue
        if owner.get("hostname") not in (None, current.get("hostname")):
            raise ValueError("Lease owner belongs to a different/unknown host")
        if (owner.get("boot_id") is not None
                and owner["boot_id"].lower() != current["boot_id"].lower()
                or owner.get("start_time_ticks") is not None
                and owner["start_time_ticks"] != current["start_time_ticks"]):
            continue
        result.append({"recorded_owner": owner, "observed_owner": current})
    return result


def canonical_paths(index: int, physical_uuid: str, roots=LEASE_ROOTS) -> list[Path]:
    return [
        Path(root) / name for root in roots
        for name in (f"physical-gpu-{index}.lock", f"physical-{physical_uuid}.lock",
                     f"physical-GPU-{physical_uuid}.lock")
    ]


def selected_alias(name: str, index: int, physical_uuid: str) -> bool:
    return (physical_uuid.lower() in name.lower()
            or re.search(rf"(^|[-_.])gpu[-_]?{index}($|[-_.])", name, re.IGNORECASE) is not None
            or re.match(rf"^{index}[.]", name) is not None)


def reference_paths(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REFERENCE_FIELDS and item is not None:
                items = [item] if isinstance(item, str) else item
                if not isinstance(items, list) or not all(isinstance(path, str) for path in items):
                    raise ValueError(f"Malformed referenced lock paths: {key}")
                found.update(items)
            elif isinstance(item, (dict, list)):
                found.update(reference_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.update(reference_paths(item))
    return found


def allowed_path(path: Path, roots: tuple[Path, ...], index: int, physical_uuid=None) -> Path:
    if not path.is_absolute():
        raise ValueError("Referenced lock path is not absolute")
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Referenced lock escapes the explicitly permitted roots")
    names = path.name + "/" + resolved.name
    uuid_pattern = r"GPU-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})"
    without_uuids = re.sub(uuid_pattern, "GPU-UUID", names, flags=re.IGNORECASE)
    indices = re.findall(r"(?:^|[-_./])gpu[-_]?(\d+)(?=$|[-_./])", without_uuids, re.IGNORECASE)
    if any(int(other) != index for other in indices):
        raise ValueError("Cross-device lock reference needs an explicit multi-GPU coordinator")
    identifiers = re.findall(uuid_pattern, names, re.IGNORECASE)
    if physical_uuid and any("gpu-" + value.lower() != physical_uuid.lower() for value in identifiers):
        raise ValueError("Cross-UUID lock reference needs an explicit multi-GPU coordinator")
    return resolved


def discover_paths(index: int, physical_uuid: str, roots=LEASE_ROOTS) -> list[Path]:
    permitted = tuple(Path(root).resolve() for root in roots)
    paths = {allowed_path(path, permitted, index, physical_uuid)
             for path in canonical_paths(index, physical_uuid, permitted)}
    for root in permitted:
        if root.is_dir():
            for path in root.iterdir():
                if selected_alias(path.name, index, physical_uuid):
                    paths.add(allowed_path(path, permitted, index, physical_uuid))
    inspected = set()
    while paths - inspected:
        if len(paths) > 128:
            raise ValueError("Unexpectedly large selected-GPU lock graph")
        for path in sorted(paths - inspected):
            inspected.add(path)
            if not path.exists():
                continue
            if not path.is_file():
                raise ValueError("GPU lease is not a regular file")
            records = parse_records(path.read_bytes().decode("utf-8"))
            paths.update(allowed_path(Path(ref), permitted, index, physical_uuid)
                         for ref in reference_paths(records))
    return sorted(paths)


def verify_idle(snapshot, index: int, physical_uuid: str) -> dict:
    _, compute, devices = snapshot
    device = devices[index]
    if device["uuid"] != physical_uuid or "H200" not in device["name"]:
        raise ValueError("Physical index/UUID/H200 identity mismatch")
    if (device["memory_used_mib"] > 100 or device["utilization_percent"] != 0
            or physical_uuid in compute):
        raise GuardBusy(f"GPU{index} is occupied or not verified idle")
    return device


def ipc_path_bytes(work: Path) -> int:
    return len(os.fsencode(str(work / ("0" * 36))))


@dataclass
class GpuLease:
    device: dict
    paths: list[Path]
    handles: list
    record: dict
    allocated_monotonic: float

    def write(self, record: dict) -> None:
        for stream, append in self.handles:
            if append:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
            else:
                stream.seek(0)
                stream.truncate()
            stream.write(json.dumps(record, sort_keys=True).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())


def acquire_gpu_guard(*, index: int, physical_uuid: str, authorized: dict[int, str],
                      hostname: str, run: Path, service: Path, stack: ExitStack,
                      query_gpu, port: int, allocation_seconds: int, successor_task_id: str,
                      roots=LEASE_ROOTS) -> GpuLease:
    if (socket.gethostname() != hostname or type(index) is not int
            or authorized.get(index) != physical_uuid):
        raise ValueError("Explicit host/physical-GPU authorization does not match")
    if not set(LEASE_ROOTS).issubset({str(root) for root in roots}):
        raise ValueError("A caller cannot omit mandatory cross-directory lease roots")
    if ipc_path_bytes(service / "work") > 100:
        raise ValueError("Project TMPDIR plus IPC UUID exceeds the pre-lease byte budget")
    before = query_gpu()
    (service / "gpu-before.csv").write_text(before[0])
    (service / "compute-before.csv").write_text(before[1])
    verify_idle(before, index, physical_uuid)
    paths = discover_paths(index, physical_uuid, roots)
    previous, handles, bindings = {}, [], []
    with ExitStack() as provisional:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
            fd = os.open(path, flags, 0o600)
            stream = provisional.enter_context(os.fdopen(fd, "r+b"))
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("GPU lease is not a regular file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise GuardBusy(f"Exclusive lease already held: {path}") from error
            raw = stream.read()
            records = parse_records(raw.decode("utf-8"))
            live = active_owners(records)
            previous[str(path)] = {
                "raw": raw.decode("utf-8"), "sha256": hashlib.sha256(raw).hexdigest(),
                "records": records, "live_owners": live,
            }
            if live:
                raise GuardBusy(f"Live legacy owner: {path}")
            handles.append((stream, len(records) > 1))
            observed = os.fstat(fd)
            bindings.append({"path": str(path), "device": observed.st_dev, "inode": observed.st_ino})
        if discover_paths(index, physical_uuid, roots) != paths:
            raise GuardBusy("Selected-GPU alias/reference graph changed during locking")
        for binding in bindings:
            observed = Path(binding["path"]).stat()
            if (observed.st_dev, observed.st_ino) != (binding["device"], binding["inode"]):
                raise GuardBusy("Locked file was replaced during allocation")
        after = query_gpu()
        device = verify_idle(after, index, physical_uuid)
        with socket.socket() as endpoint:
            endpoint.bind(("127.0.0.1", port))
        owner = process_identity(os.getpid())
        if owner is None:
            raise RuntimeError("Cannot bind the current supervisor's process identity")
        started = time.monotonic()
        record = {
            "owner": owner, "hostname": hostname, "run": str(run),
            "physical_index": index, "physical_uuid": physical_uuid,
            "authorized_physical_gpus": authorized,
            "required_CUDA_VISIBLE_DEVICES": physical_uuid,
            "at_unix": time.time(), "hard_deadline_unix": time.time() + allocation_seconds,
            "successor_task_id": successor_task_id, "guard_protocol": "cross-directory-gpu-guard/v2",
        }
        proof = {
            "status": "PASS", "at_unix": time.time(), "owner": owner,
            "physical_index": index, "physical_uuid": physical_uuid,
            "roots": list(roots), "paths": [str(path) for path in paths],
            "previous_records": previous, "file_identity_bindings": bindings,
            "all_exclusive_locks_held_before_gpu_launch": True,
            "fresh_gpu_before_and_after_all_locks": True,
            "project_ipc_path_bytes_with_uuid": ipc_path_bytes(service / "work"),
        }
        with (service / "full-lease-preflight.json").open("x") as output:
            json.dump(proof, output, indent=2, sort_keys=True)
        (service / "gpu-at-allocation.csv").write_text(after[0])
        (service / "compute-at-allocation.csv").write_text(after[1])
        guard = GpuLease(device, paths, handles, record, started)
        guard.write(record)
        stack.enter_context(provisional.pop_all())
        return guard
