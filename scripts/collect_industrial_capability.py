#!/usr/bin/env python3
"""Read-only collection/monitoring of the new A800 shared-parent TRAIN campaigns.

Use --watch for durable local result collection; it never launches a benchmark,
reads old campaigns, or accesses a held-out result. Native remote controllers
own task scheduling. LATEST.json and material EVENTS.jsonl expose their progress.
New prepared runs verify the full frozen Harbor-rendered prompt receipt; legacy
runs retain their original raw-instruction hash contract without reinterpretation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import write_json_atomic  # noqa: E402
from phiagent.harness.industrial_common_prompt import PROMPT_SHA256  # noqa: E402

REMOTE_ROOT = "/path/to/data/phiagent/work/deepseek-harness/industrial-capability/20260930-v1"
ARTIFACT_ROOT = ROOT / "artifacts/industrial-capability/20260930-v1"


def aggregate_provider_usage(calls: list[dict]) -> dict[str, int | None]:
    usage = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [(call.get("usage") or {}).get(field) for call in calls]
        usage[field] = (
            sum(values)
            if all(type(value) is int and value >= 0 for value in values)
            else None
        )
    return usage


def read_run_packet(run: Path) -> dict:
    """Read one native TRAIN run without executing or rewriting its artifacts."""
    item = {"run": str(run)}
    for filename in (
        "CONFIG.json",
        "SHARED_PROMPT.json",
        "FROZEN_IDENTITY.json",
        "SOURCE_SHA256.json",
        "TASK_SOURCE_SHA256.json",
        "TASK_PROMPT_RECEIPTS.json",
        "SPLIT_MANIFEST.json",
        "NATIVE_RUNTIME.json",
        "REPORT.json",
        "OWNER.json",
        "LAUNCH.json",
        "QUEUE.json",
        "CAPABILITY_GATE.json",
        "CAPACITY.json",
        "CONTROLLER_FAILURE.json",
    ):
        path = run / filename
        if path.is_file():
            item[filename] = json.loads(path.read_text())
    if item.get("CONFIG.json", {}).get("phase") != "train":
        raise ValueError("Read-only native collection requires a TRAIN run")
    train_ids = set(item["SPLIT_MANIFEST.json"]["splits"]["train"])
    owner = item.get("OWNER.json", {})
    try:
        stat = (Path("/proc") / str(owner["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        item["controller_alive"] = stat[0] != "Z" and stat[19] == owner["start_ticks"]
    except (KeyError, FileNotFoundError):
        item["controller_alive"] = False
    item["frozen_identity_valid"] = all(
        hashlib.sha256((run / filename).read_bytes()).hexdigest() == expected
        for filename, expected in item.get("FROZEN_IDENTITY.json", {}).items()
    )
    evidence = []
    for observation in item.get("REPORT.json", {}).get("attempts", []):
        if observation.get("phase") != "train" or observation["task_id"] not in train_ids:
            raise ValueError("Native evidence is outside the declared TRAIN population")
        native = observation.get("native_result")
        row = {"task_id": observation["task_id"], "prompts": []}
        if native:
            path = Path(native)
            if not path.resolve().is_relative_to(run.resolve()):
                raise ValueError("Native evidence escaped new run")
            value = json.loads(path.read_text())
            row.update(
                native_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rewards=(value.get("verifier_result") or {}).get("rewards"),
                exception_type=(value.get("exception_info") or {}).get("exception_type"),
                native_task_name=value.get("task_name"),
            )
        for prompt in observation.get("shared_prompt_receipts", []):
            path = Path(prompt)
            if not path.resolve().is_relative_to(run.resolve()):
                raise ValueError("Prompt receipt escaped new run")
            row["prompts"].append(json.loads(path.read_text()))
        call_ledger = observation.get("provider_call_ledger")
        if call_ledger:
            path = Path(call_ledger)
            if not path.resolve().is_relative_to(run.resolve()):
                raise ValueError("Cost evidence escaped new run")
            calls = json.loads(path.read_text())["calls"]
            row["unique_provider_calls"] = len({call["call_id"] for call in calls}) == len(calls)
            row["provider_call_count"] = len(calls)
            row["provider_usage"] = aggregate_provider_usage(calls)
        evidence.append(row)
    item["native_evidence"] = evidence
    return item


def remote_program(names: list[str]) -> str:
    if (
        len(names) != 2
        or len(set(names)) != 2
        or not any(n.startswith("train-codex-") for n in names)
        or not any(n.startswith("train-deepseek-") for n in names)
        or any(not re.fullmatch(r"train-(?:codex|deepseek)-v[1-9][0-9]*", n) for n in names)
    ):
        raise ValueError("Collect only the named new shared-parent TRAIN tracks")
    return (
        "from pathlib import Path\nimport hashlib,json\n"
        + inspect.getsource(aggregate_provider_usage)
        + inspect.getsource(read_run_packet)
        + f"\nroot=Path({REMOTE_ROOT!r})\n"
        + f"packet={{name:read_run_packet(root/name) for name in {names!r}}}\n"
        + "print(json.dumps(packet,sort_keys=True))\n"
    )


def verify_packet(packet: dict) -> dict:
    tracks, first_prompts = {}, []
    for name, item in packet.items():
        report = item.get("REPORT.json") or {}
        source_prompt = item.get("SHARED_PROMPT.json") or {}
        if (
            source_prompt.get("sha256") != PROMPT_SHA256
            or hashlib.sha256(source_prompt.get("text", "").encode()).hexdigest() != PROMPT_SHA256
            or item.get("frozen_identity_valid") is not True
        ):
            raise ValueError("Collected campaign has a different shared parent")
        prepared_prompts = item.get("TASK_PROMPT_RECEIPTS.json")
        if prepared_prompts is not None and (
            not isinstance(prepared_prompts, dict)
            or "TASK_PROMPT_RECEIPTS.json" not in item.get("FROZEN_IDENTITY.json", {})
        ):
            raise ValueError("Native task prompt receipts must belong to the frozen identity")
        observations = report.get("attempts", [])
        evidence = {row["task_id"]: row for row in item.get("native_evidence", [])}
        scored = []
        for row in observations:
            if row["status"] != "SCORED":
                continue
            native = evidence[row["task_id"]]
            prompts = native["prompts"]
            if prepared_prompts is not None:
                task_matches = len(prompts) == 1 and prompts[0] == prepared_prompts.get(
                    row["task_id"]
                )
            else:
                raw_hash = (
                    item.get("TASK_SOURCE_SHA256.json", {})
                    .get(
                        row["task_id"],
                        {},
                    )
                    .get("instruction.md")
                )
                task_matches = (
                    raw_hash is not None
                    and len(prompts) == 1
                    and prompts[0].get("original_task_sha256") == raw_hash
                )
            if (
                native.get("native_result_sha256") != row["native_result_sha256"]
                or native.get("native_task_name") != row["task_id"]
                or native.get("rewards") != {"reward": row["native_score"]}
                or native.get("exception_type") not in (None, "AgentTimeoutError")
                or len(prompts) != 1
                or prompts[0].get("sha256") != PROMPT_SHA256
                or not task_matches
                or prompts[0].get("shared_block_count") != 1
                or prompts[0].get("original_task_preserved") is not True
            ):
                raise ValueError(
                    "Native score or exactly-once shared-prompt evidence failed verification"
                )
            if (row.get("cost") or {}).get("usage_status") == "MEASURED" and (
                native.get("unique_provider_calls") is not True
                or native.get("provider_usage") != row["cost"]["usage"]
                or native.get("provider_call_count") != row["cost"]["provider_call_count"]
            ):
                raise ValueError(
                    "Native per-call costs do not reconcile with the one-time actor total"
                )
            scored.append(row)
        first = next((row for row in scored if row["task_id"] == "headless-terminal"), None)
        if first:
            first_prompts.append(evidence[first["task_id"]]["prompts"][0])
        tracks[name] = {
            "run": item["run"],
            "controller_alive": item.get("controller_alive", False),
            "controller_pid": (item.get("OWNER.json") or {}).get("pid"),
            "status": report.get("status", "STARTING"),
            "native_scored": len(scored),
            "native_successes": sum(row["native_score"] for row in scored),
            "expected_train": 18,
            "active": report.get("active", {}),
            "first_native_gate": first,
            "accounting": report.get("accounting"),
            "accounting_scope": "Completed/blocked native attempts; active provider receipts remain remote",
            "capability_gate": item.get("CAPABILITY_GATE.json"),
            "controller_failure": item.get("CONTROLLER_FAILURE.json"),
        }
    same_prompt = len(first_prompts) == 2 and first_prompts[0] == first_prompts[1]
    if len(first_prompts) == 2 and not same_prompt:
        raise ValueError("Codex and DeepSeek received different common task content")
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "shared_prompt_sha256": PROMPT_SHA256,
        "tracks": tracks,
        "both_native_gates_verified": len(first_prompts) == 2,
        "same_first_task_prompt_verified": same_prompt,
        "scope": "New shared-prompt TRAIN observations only; no accuracy-improvement claim",
    }


def collect(host: str, control_path: str, names: list[str]) -> dict:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=600",
        "-o",
        "ControlPath=" + control_path,
        host,
        "python3 -",
    ]
    result = subprocess.run(
        command,
        input=remote_program(names),
        capture_output=True,
        text=True,
        timeout=90,
        cwd=ROOT,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="a800-4", choices=("a800-4",))
    parser.add_argument(
        "--control-path", default="artifacts/industrial-capability/20260930-v1/a800.sock"
    )
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "monitor")
    parser.add_argument("--codex-run", default="train-codex-v1")
    parser.add_argument("--deepseek-run", default="train-deepseek-v1")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 15 <= args.interval <= 300:
        parser.error("Use a 15–300 second read-only monitor interval")
    if not args.output.absolute().is_relative_to(ARTIFACT_ROOT):
        parser.error("Keep collection under the declared local capability artifact directory")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "COLLECTOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json_atomic(
        args.output / "MONITOR_OWNER.json",
        {
            "pid": os.getpid(),
            "argv": sys.argv,
            "watch": args.watch,
            "at": datetime.now(timezone.utc).isoformat(),
            "read_only_remote": True,
            "native_launch_authority": False,
        },
    )
    last_event = None
    while True:
        try:
            packet = collect(args.host, args.control_path, [args.codex_run, args.deepseek_run])
            summary = verify_packet(packet)
            write_json_atomic(args.output / "NATIVE_EVIDENCE.json", packet)
            write_json_atomic(args.output / "LATEST.json", summary)
            material = {
                key: {
                    field: row[field]
                    for field in (
                        "controller_alive",
                        "status",
                        "native_scored",
                        "native_successes",
                        "first_native_gate",
                        "accounting",
                        "controller_failure",
                    )
                }
                for key, row in summary["tracks"].items()
            }
            event_hash = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
            if event_hash != last_event:
                with (args.output / "EVENTS.jsonl").open("a") as stream:
                    stream.write(json.dumps(summary, sort_keys=True) + "\n")
                print(
                    json.dumps(
                        {
                            "at": summary["at"],
                            "both_native_gates_verified": summary["both_native_gates_verified"],
                            "tracks": {
                                name: {
                                    key: row[key]
                                    for key in (
                                        "status",
                                        "controller_alive",
                                        "native_scored",
                                        "native_successes",
                                    )
                                }
                                for name, row in summary["tracks"].items()
                            },
                        }
                    ),
                    flush=True,
                )
                last_event = event_hash
            if not args.watch or all(
                not row["controller_alive"] for row in summary["tracks"].values()
            ):
                return 0
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            write_json_atomic(
                args.output / "COLLECTION_FAILURE.json",
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "remote_actions_taken": 0,
                },
            )
            if not args.watch:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
