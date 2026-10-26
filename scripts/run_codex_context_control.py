#!/usr/bin/env python3
"""Queue a native Codex compaction control behind the same full18 TRAIN parent.

This is an explicit industrial baseline control, not a learned CHISEL method.
The existing native scheduler, task scorer, prompt, model and usage meter are
reused. A smaller active-context trigger never terminates an experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import utc_now, write_json_atomic  # noqa: E402
from scripts import audit_industrial_capability_recovery as recovery  # noqa: E402
from scripts import run_industrial_capability_baselines as native  # noqa: E402
from scripts.collect_industrial_capability import read_run_packet, verify_packet  # noqa: E402
from scripts.run_codex_terminal import context_control  # noqa: E402

SCHEMA = "codex-native-context-control/v2"
METHOD = "codex_native_compaction"
SOURCES = (
    "scripts/run_codex_context_control.py",
    "scripts/collect_industrial_capability.py",
    "scripts/audit_industrial_capability_recovery.py",
    "scripts/continue_codex_capability_infrastructure.py",
    "tests/test_codex_context_control.py",
)


def replacement_binding(value: object) -> tuple[Path, str]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not isinstance(value["sha256"], str)):
        raise ValueError("A prospective parent replacement requires its path and SHA256")
    path = Path(value["path"])
    if (not path.is_absolute() or ".." in path.parts
            or not path.is_relative_to("/path/to/data/phiagent/work/harness-terminal-recovery")
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("Invalid prospective parent replacement binding")
    return path, value["sha256"]


def validate_control(config: dict) -> None:
    expected = {
        "schema": SCHEMA, "method": METHOD, "phase": "train",
        "cumulative_token_limit": None, "context_compaction_threshold": 65536,
        "baseline_reexecution": False, "maximum_workers": 2,
    }
    if any(type(config.get(key)) is not type(value) or config.get(key) != value
           for key, value in expected.items()):
        raise ValueError("Only the explicit unlimited full-TRAIN native compaction control is allowed")
    for key in ("parent_run", "run_directory"):
        path = Path(config[key])
        native.validate_run_path(path)
    if config["parent_run"] == config["run_directory"]:
        raise ValueError("The control must not modify or reuse its parent's output directory")
    if Path(config["parent_run"]) != recovery.REMOTE / "train-codex-v1":
        raise ValueError("Require the registered source-bound Codex parent")
    replacement_binding(config.get("parent_replacement_manifest"))
    roots = config.get("parent_recovery_roots")
    if (not isinstance(roots, list) or not roots
            or any(not isinstance(root, str) or not Path(root).is_absolute()
                   or ".." in Path(root).parts
                   or not Path(root).is_relative_to("/path/to/data/phiagent/work/harness-terminal-recovery")
                   for root in roots)
            or len(set(roots)) != len(roots)):
        raise ValueError("Explicit unique native-verifier recovery roots are required")
    context_control(METHOD, config["context_compaction_threshold"], "", "train")


def control_command(command: list[str], threshold: int) -> list[str]:
    context_control(METHOD, threshold, "", "train")
    if (command.count("--method") != 1 or command.count("--phase") != 1
            or "--residual-file" in command or "--context-compaction-threshold" in command):
        raise ValueError("The control requires a clean native baseline command")
    result = list(command)
    index = result.index("--method") + 1
    if index >= len(result) or result[index] != "codex":
        raise ValueError("Do not relabel an existing CHISEL or control command")
    phase = result.index("--phase") + 1
    if phase >= len(result) or result[phase] != "train":
        raise ValueError("The compaction control cannot launch held-out work")
    result[index] = METHOD
    result += ["--context-compaction-threshold", str(threshold)]
    return result


def recovered_parent(
    track: dict, config: dict, inherited: dict, ids: list[str], *, track_name: str = "codex",
) -> dict:
    if track_name not in ("codex", "deepseek"):
        raise ValueError("Unknown native industrial parent track")
    if track["run"] != config["parent_run"]:
        raise ValueError("Recovered admission belongs to a different native parent")
    recomputed = recovery.merged_gate(track_name, ids, track["rows"], not track["identity_errors"])
    if any(type(track["gate"].get(key)) is not type(value)
           or track["gate"].get(key) != value for key, value in recomputed.items()):
        raise ValueError("Source-bound parent gate does not reconcile with its admitted rows")
    capable = recomputed["complete_valid_capable_parent"]
    if capable:
        native.require_rsi_parent(recomputed, inherited, track_name)
    live = track["controller"]["alive"]
    if type(live) is not bool:
        raise ValueError("Parent owner liveness must be explicit")
    ready = capable and not live
    rows = [
        {**row, "status": "SCORED" if row["admitted"] else "BLOCKED"}
        for row in track["rows"]
    ]
    all_attempts = track.get("all_actor_attempts", track["rows"])
    selected_cost = (
        recovery.aggregate_costs(rows, required_attempts=18)
        if "all_actor_attempts" in track else track["accounting"]
    )
    return {
        "ready": ready,
        "status": "VERIFIED_SOURCE_BOUND_PARENT" if ready else (
            "PARENT_DRAINING" if capable else recomputed["status"]
        ),
        "native_scored": recomputed["native_scored"],
        "native_successes": recomputed["native_successes"],
        "owner_live": live, "required": 18, "minimum_successes": 9,
        "source_bound_gate": dict(track["gate"]),
        "effective_parent_run": track.get("parent_run", track["run"]),
        "blocked_tasks": [
            {"task_id": row["task_id"], "blockers": row["blockers"]}
            for row in rows if not row["admitted"]
        ],
        "report": {
            "attempts": rows, "active": {},
            "all_actor_attempts": all_attempts,
            "coverage": {"expected": 18, "scored": recomputed["native_scored"],
                         "successes": recomputed["native_successes"]},
            "accounting": track["accounting"],
            "selected_execution_accounting": selected_cost,
        },
    }


def source_bound_parent_evidence(config: dict, track_name: str) -> dict:
    if track_name not in ("codex", "deepseek"):
        raise ValueError("Unknown native industrial parent track")
    parent = Path(config["parent_run"])
    if parent != recovery.REMOTE / f"train-{track_name}-v1":
        raise ValueError("Require the registered source-bound native parent")
    native.check_sources(parent)
    inherited = native.load(parent / "CONFIG.json")
    native.validate_config(inherited)
    if inherited.get("track") != track_name:
        raise ValueError("The control must descend from its own shared-prompt native parent")
    ids = [
        row["id"] for row in native.train_records(native.load(parent / "SPLIT_MANIFEST.json"), inherited)
    ]
    evidence = {}
    recoveries = recovery.recovery_inventory(config["parent_recovery_roots"], evidence)
    track = recovery.audit_track(track_name, recoveries, evidence)
    if config.get("parent_replacement_manifest") is not None:
        if track_name != "codex":
            raise ValueError("The declared prospective replacements belong only to Codex")
        manifest, checksum = replacement_binding(config["parent_replacement_manifest"])
        track = recovery.apply_prospective_manifest(track, manifest, checksum, evidence)
    return {
        **recovered_parent(track, config, inherited, ids, track_name=track_name),
        "audit_source_sha256": recovery.digest(Path(recovery.__file__)),
        "source_evidence": evidence,
    }


def parent_evidence(config: dict) -> dict:
    return source_bound_parent_evidence(config, "codex")


def comparison_summary(baseline: dict, control: dict) -> dict:
    for report in (baseline, control):
        rows = report["attempts"]
        if len({row["task_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate native task outcomes cannot enter a paired comparison")
    complete = all(
        len(report["attempts"]) == report["coverage"]["scored"] == 18
        and not report.get("active") for report in (baseline, control)
    )
    if complete and (
        {row["task_id"] for row in baseline["attempts"]}
        != {row["task_id"] for row in control["attempts"]}
    ):
        raise ValueError("Native control and parent have different task populations")
    costs = {
        label: report["accounting"]["tokens"]["total_tokens"]["complete_total"]
        for label, report in (("baseline", baseline), (METHOD, control))
    }
    measured = complete and all(type(value) is int and value >= 0 for value in costs.values())
    saving = 1 - costs[METHOD] / costs["baseline"] if measured and costs["baseline"] > 0 else None
    selected_costs = {
        label: report.get("selected_execution_accounting", report["accounting"])[
            "tokens"
        ]["total_tokens"]["complete_total"]
        for label, report in (("baseline", baseline), (METHOD, control))
    }
    selected_measured = complete and all(
        type(value) is int and value >= 0 for value in selected_costs.values()
    )
    selected_saving = (
        1 - selected_costs[METHOD] / selected_costs["baseline"]
        if selected_measured and selected_costs["baseline"] > 0 else None
    )
    return {
        "schema": SCHEMA, "phase": "train", "complete_native_population": complete,
        "baseline": baseline["coverage"], METHOD: control["coverage"],
        "actor_tokens_including_native_compaction": costs,
        "total_token_reduction": saving,
        "selected_native_execution_tokens": selected_costs,
        "selected_native_execution_token_reduction": selected_saving,
        "actor_attempt_records": {
            label: len(report.get("all_actor_attempts", report["attempts"]))
            for label, report in (("baseline", baseline), (METHOD, control))
        },
        "cost_scope": (
            "All-in costs retain every actor attempt; selected execution costs retain one "
            "valid native attempt per task, including native failures. Superseded infrastructure "
            "attempts remain in all-in costs and are not attributed to context efficiency."
        ),
        "native_success_delta": (
            control["coverage"]["successes"] - baseline["coverage"]["successes"]
        ) if complete else None,
        "cumulative_token_limit": None,
        "implementation_label": "official Codex native compaction control; not CHISEL",
        "test_access": False, "baseline_reexecuted": False,
        "unknown_costs_not_zero_imputed": True,
    }


def initialize(config: dict) -> Path:
    native.require_remote()
    validate_control(config)
    run = Path(config["run_directory"])
    parent = Path(config["parent_run"])
    native.check_sources(parent)
    inherited = native.load(parent / "CONFIG.json")
    if inherited.get("track") != "codex":
        raise ValueError("A Codex parent is required")
    inherited["require_executed_native_tests"] = True
    native.prepare(inherited, "codex", run, extra_source_files=SOURCES)
    write_json_atomic(run / "CONTROL.json", config)
    frozen = native.load(run / "FROZEN_IDENTITY.json")
    frozen["CONTROL.json"] = native.digest(run / "CONTROL.json")
    write_json_atomic(run / "FROZEN_IDENTITY.json", frozen)
    return run


def worker(run: Path) -> int:
    native.require_remote()
    native.check_sources(run)
    config = native.load(run / "CONTROL.json")
    validate_control(config)
    if run != Path(config["run_directory"]):
        raise ValueError("Control identity differs from its declared run directory")
    with (run / "CONTROL_OWNER.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json_atomic(run / "CONTROL_OWNER.json", {
            "pid": os.getpid(), "start_ticks": native.process_identity(os.getpid()),
            "at": utc_now(), "run": str(run),
        })
        last_audit = 0.0
        try:
            while True:
                native.check_sources(run)
                if time.monotonic() - last_audit >= 300:
                    write_json_atomic(run / "RESOURCE_LATEST.json", native.resource_inventory())
                    last_audit = time.monotonic()
                parent = parent_evidence(config)
                write_json_atomic(run / "STATE.json", {
                    "at": utc_now(), "schema": SCHEMA, "pid": os.getpid(),
                    "status": "PARENT_ADMITTED" if parent["ready"] else "WAITING_CAPABILITY",
                    "parent": {key: value for key, value in parent.items()
                               if key not in ("report", "source_evidence")},
                    "waiter_model_calls": 0, "cumulative_token_limit": None,
                    "method": METHOD, "test_access": False,
                })
                if (run / "STOP").exists():
                    write_json_atomic(run / "STATE.json", {
                        "at": utc_now(), "schema": SCHEMA, "status": "STOPPED", "model_calls": 0,
                    })
                    return 0
                if parent["ready"]:
                    break
                if parent.get("status") in ("INSUFFICIENT_CAPABILITY", "BLOCKED_IDENTITY"):
                    raise RuntimeError("Native parent did not pass its source-bound capability gate")
                time.sleep(30)
            write_json_atomic(run / "PARENT_ADMISSION.json", parent)

            def build(inherited, track, root, directory, task):
                return control_command(
                    native.trial_command(inherited, track, root, directory, task),
                    config["context_compaction_threshold"],
                )

            write_json_atomic(run / "STATE.json", {
                "at": utc_now(), "schema": SCHEMA, "status": "RUNNING_NATIVE_CONTROL",
                "pid": os.getpid(), "method": METHOD, "report": str(run / "REPORT.json"),
                "cumulative_token_limit": None, "test_access": False,
            })
            report = native.controller(run, command_builder=build)
            verify_packet({"control": read_run_packet(run)})
            summary = comparison_summary(parent["report"], report)
            write_json_atomic(run / "COMPARISON.json", summary)
            write_json_atomic(run / "STATE.json", {
                "at": utc_now(), "schema": SCHEMA,
                "status": "COMPLETE" if summary["complete_native_population"] else "BLOCKED",
                "comparison": str(run / "COMPARISON.json"), "method": METHOD,
            })
            return 0 if summary["complete_native_population"] else 2
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            write_json_atomic(run / "FAILURE.json", {
                "at": utc_now(), "type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "actor_replayed": False,
            })
            write_json_atomic(run / "STATE.json", {
                "at": utc_now(), "schema": SCHEMA, "status": "BLOCKED",
                "failure": str(run / "FAILURE.json"),
            })
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    native.require_remote()
    if args.worker:
        if args.run_dir is None:
            parser.error("--worker requires --run-dir")
        return worker(args.run_dir)
    if args.config is None:
        parser.error("New controls require --config")
    run = initialize(native.load(args.config))
    if not args.detach:
        return worker(run)
    inherited = native.load(run / "CONFIG.json")
    command = [
        inherited["codex"]["python"], "-B", "-u",
        str(run / "source/scripts/run_codex_context_control.py"),
        "--run-dir", str(run), "--worker",
    ]
    with (run / "controller.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=run, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                 "PYTHONPATH": str(run / "source"), "TMPDIR": str(run / "process-cache"),
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
    write_json_atomic(run / "LAUNCH.json", {
        "at": utc_now(), "pid": process.pid, "start_ticks": native.process_identity(process.pid),
        "argv": command, "detached": True,
    })
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f"Control startup failed; inspect {run / 'controller.log'}")
    print(json.dumps({"run": str(run), "pid": process.pid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
