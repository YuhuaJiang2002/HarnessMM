"""CPU/API scheduling limits without changing native task resource contracts."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

WORKER_COUNTS = (1, 2, 4, 8)


def validate_workers(value: object) -> int:
    if type(value) is not int or value not in WORKER_COUNTS:
        raise ValueError("Declare one, two, four or eight native CPU/API workers")
    return value


def declared_bytes(value: object) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)?[GMgm]", value):
        raise ValueError("Native memory/storage must be a finite nonnegative G or M size")
    number = float(value[:-1])
    if not math.isfinite(number):
        raise ValueError("Native memory/storage must be finite")
    return int(number * {"G": 2**30, "M": 2**20}[value[-1].upper()])


def task_admission(
    capacity: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    *,
    cpu_count: int,
    load_average: Sequence[float],
    minimum_available_memory_gib: int,
) -> dict[str, Any]:
    if (
        type(cpu_count) is not int or cpu_count <= 0
        or len(load_average) != 3
        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in load_average)
        or type(minimum_available_memory_gib) is not int
        or minimum_available_memory_gib < 8
        or type(capacity.get("ready")) is not bool
    ):
        raise ValueError("Require measured native CPU/load and unchanged memory reserves")
    filesystems = capacity.get("writable_filesystems")
    available = capacity.get("available_memory_bytes")
    if (
        not isinstance(filesystems, list) or not filesystems
        or type(available) is not int or available < 0
        or any(
            not isinstance(row, Mapping)
            or type(row.get("free_bytes")) is not int or row["free_bytes"] < 0
            or type(row.get("minimum_free_bytes")) is not int
            or row["minimum_free_bytes"] < 8 * 2**30
            for row in filesystems
        )
    ):
        raise ValueError("Require measured writable-filesystem capacity and native reserves")
    storage, memory, cpus = 0, 0, 0
    for task in tasks:
        environment = task.get("environment")
        if not isinstance(environment, Mapping):
            raise ValueError("Every task must retain its declared native environment")
        task_cpus = environment.get("cpus", 1)
        if type(task_cpus) is not int or task_cpus <= 0:
            raise ValueError("Native task CPU allocation must be a positive integer")
        cpus += task_cpus
        storage += declared_bytes(environment.get("storage", "10G"))
        memory += declared_bytes(environment.get("memory", "2G"))
    free = min(row["free_bytes"] for row in filesystems)
    storage_ready = all(
        row["free_bytes"] >= row["minimum_free_bytes"] + storage for row in filesystems
    )
    ready = (
        capacity["ready"] and storage_ready
        and available >= minimum_available_memory_gib * 2**30 + memory
        and load_average[0] + cpus < cpu_count * 0.9
    )
    return {
        **capacity,
        "ready": ready,
        "cpu_count": cpu_count,
        "load_average": list(load_average),
        "candidate_and_active_cpus": cpus,
        "candidate_declared_storage_bytes": storage,
        "reserve_after_candidate_bytes": free - storage,
        "candidate_and_active_memory_bytes": memory,
        "memory_available_bytes": available,
        "selected_gpus": [],
        "CUDA_VISIBLE_DEVICES": "",
        "resource_gate_does_not_preempt_or_raise_native_limits": True,
    }
