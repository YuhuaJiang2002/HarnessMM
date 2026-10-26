#!/usr/bin/env python3
"""Zero-generation Terminal prefix/transport fidelity and scoped reset eligibility."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any


V14_SHA256 = "6ed0aaa60d071818b5062a43a90727bf7e7452aa87cc19b2d4b28f5bfd9b207c"
CASES = ("large-scale-text-editing", "sanitize-git-repo", "multi-source-data-merger", "circuit-fibsqrt")


class FidelityError(ValueError):
    pass


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def freeze(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def normalize_tool_ids(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize only structured linkage IDs, before the unchanged v14 serializer."""
    result = deepcopy(request)
    mapping: dict[str, str] = {}
    for message in result.get("messages", []):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                identity = call.get("id")
                if not isinstance(identity, str) or not identity or identity in mapping:
                    raise FidelityError("missing or duplicate tool-call definition ID")
                mapping[identity] = f"tool-call-{len(mapping) + 1:06d}"
                call["id"] = mapping[identity]
        elif message.get("role") == "tool":
            identity = message.get("tool_call_id")
            if identity not in mapping:
                raise FidelityError("orphan tool result: complete linkage history is required")
            message["tool_call_id"] = mapping[identity]
    return result, mapping


class ParentOutputShadow:
    """Archive/live-parent output sharing, not an environment reset or model resample."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = deepcopy(frames)
        self.cursor = 0
        self.branched = False

    def take(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.branched or self.cursor >= len(self.frames):
            raise FidelityError("shadow exhausted or intervention boundary reached")
        frame = self.frames[self.cursor]
        if normalize_tool_ids(request)[0] != normalize_tool_ids(frame["request"])[0]:
            raise FidelityError("semantic no-edit drift before intervention")
        self.cursor += 1
        return deepcopy({"response": frame["response"], "public_feedback": frame["public_feedback"]})

    def intervention_boundary(self) -> dict[str, Any]:
        if self.branched:
            raise FidelityError("only one declared intervention boundary")
        self.branched = True
        return {
            "next_turn": self.cursor + 1, "shared_parent_turns": self.cursor,
            "requires_verified_environment_branch": True, "model_calls": 0,
            "fallback_without_restore_certificate": "prospective whole-task randomization",
        }


def scoped_read_target(command: str) -> tuple[str | None, str]:
    if re.search(r"/(?:tests|solution|verifier[^/\s]*)\b", command):
        return None, "protected grader/solution path"
    if "expected.csv" in command:
        return None, "reference-file exposure ambiguous; do not inspect without classification"
    match = re.fullmatch(r"cat (/app/[A-Za-z0-9_./-]+)", command)
    if match and ".." not in Path(match.group(1)).parts:
        return match.group(1), "single immutable public-file read"
    return None, "outside declared single-file read contract; arbitrary shell/import side effects unproven"


def command(argv: list[str], *, timeout: float = 30) -> dict[str, Any]:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"argv": argv, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def locate(study: Path, phase: str, arm: str, task: str) -> Path:
    paths = list((study / "runs" / f"{phase}-{arm}").glob(f"replica-*/trials/{task}"))
    if len(paths) != 1:
        raise FidelityError("parent task archive is missing or ambiguous")
    return paths[0]


def public_frames(trial: Path) -> list[dict[str, Any]]:
    transcript = load(trial / "agent/transcript.json")
    observed = {
        message["tool_call_id"]: message["content"]
        for message in transcript if message.get("role") == "tool"
    }
    frames = []
    for directory in sorted((trial / "agent").glob("turn-*")):
        if not (directory / "response.json").is_file():
            break
        feedback = []
        for path in sorted(directory.glob("tool-*/result.json")):
            value = load(path)
            identity = load(path.parent / "call.json").get("id")
            if identity not in observed:
                raise FidelityError("exact actor-visible public feedback is unavailable")
            feedback.append({
                "command": value.get("command"),
                "actor_observation": observed[identity],
            })
        frames.append({"request": load(directory / "request.json"),
                       "response": load(directory / "response.json"), "public_feedback": feedback})
    return frames


def archive_gate(study: Path, proxy: Path, output: Path) -> dict[str, Any]:
    if sha(proxy) != V14_SHA256:
        raise FidelityError("v14 serializer source differs")
    specification = importlib.util.spec_from_file_location("frozen_terminal_v14", proxy)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    parent = public_frames(locate(study, "discovery", "parent", "break-filter-js-from-html"))
    inactive = public_frames(locate(study, "discovery", "conditional", "break-filter-js-from-html"))
    comparisons = []
    for turn, (a, b) in enumerate(zip(parent, inactive), 1):
        na, ma = normalize_tool_ids(a["request"])
        nb, mb = normalize_tool_ids(b["request"])
        fa, _ = module._tool_call_shim_request(na)
        fb, _ = module._tool_call_shim_request(nb)
        comparisons.append({
            "turn": turn, "raw_equal": a["request"] == b["request"],
            "normalized_equal": na == nb, "v14_after_common_normalization_equal": fa == fb,
            "parent_mapping": ma, "inactive_mapping": mb,
            "raw_parent_request_sha256": canonical(a["request"]),
            "raw_inactive_request_sha256": canonical(b["request"]),
        })
    shadow = ParentOutputShadow(parent)
    shared = []
    for frame in parent[:3]:
        replay = shadow.take(frame["request"])
        shared.append({
            "response_exact": replay["response"] == frame["response"],
            "public_feedback_exact": replay["public_feedback"] == frame["public_feedback"],
            "response_sha256": canonical(replay["response"]),
        })
    report = {
        "schema_version": "phiagent-terminal-prefix-fidelity/v1", "comparisons": comparisons,
        "first_raw_difference": next((r["turn"] for r in comparisons if not r["raw_equal"]), None),
        "first_normalized_semantic_difference": next((r["turn"] for r in comparisons if not r["normalized_equal"]), None),
        "shadow": shared, "boundary": shadow.intervention_boundary(),
        "shadow_exact": all(r["response_exact"] and r["public_feedback_exact"] for r in shared),
        "normalization_scope": "assistant.tool_calls[].id and tool.tool_call_id only",
        "semantic_strings_arguments_and_observations_rewritten": False,
        "common_to_all_methods_required": True, "model_determinism_claim": False,
        "model_calls": 0, "grading_data_read": False,
        "scope": "Archived public parent-prefix fidelity; not world-state replay or a benchmark score",
    }
    freeze(output / "archive-prefix-gate.json", report)
    return report


def inspect_case(study: Path, output: Path, task: str) -> dict[str, Any]:
    trial = locate(study, "evaluation", "parent", task)
    config = load(trial.parents[1] / "config.json")
    spec = next(item for item in config["tasks"] if item["name"] == task)
    call_path = trial / "agent/turn-001/tool-01/call.json"
    public_command = json.loads(load(call_path)["function"]["arguments"])["command"]
    target, reason = scoped_read_target(public_command)
    directory = output / "cases" / task
    directory.mkdir(parents=True)
    row = {
        "task": task, "prefix_tool_calls": 1, "parent_call_sha256": sha(call_path),
        "parent_command": public_command, "scoped_read_target": target,
        "prefix_replay_admission": reason, "full_task_same_state_reset_eligible": False,
        "full_task_exclusion": "Unrestricted exec_in_sandbox admits process, memory, clock, volume and network observations; filesystem equality is insufficient.",
        "official_image": spec["docker_image"], "expected_image_id": spec["docker_image_id"],
        "model_calls": 0, "verifier_calls": 0,
    }
    inspect = command(["docker", "image", "inspect", spec["docker_image"]])
    freeze(directory / "image-inspect-command.json", inspect)
    if inspect["exit_code"]:
        row["status"] = "EXCLUDED_IMAGE_UNAVAILABLE"
        return row
    image = json.loads(inspect["stdout"])[0]
    if image["Id"] != spec["docker_image_id"] or image["Config"].get("Volumes"):
        row["status"] = "EXCLUDED_IMAGE_ID_OR_DECLARED_VOLUMES"
        return row
    containers = []
    checks = []
    try:
        for replica in range(2 if target else 1):
            name = f"phiagent-fidelity-{output.name}-{task}-{replica}"[:120]
            argv = [
                "docker", "create", "--name", name, "--label", f"phiagent.fidelity={output}",
                "--network", "none", "--read-only", "--cpus", "1", "--memory", "1g",
                spec["docker_image"], "sleep", "infinity",
            ]
            created = command(argv)
            freeze(directory / f"create-{replica}.json", created)
            if created["exit_code"]:
                raise FidelityError("isolated CPU inspection container could not be created")
            identifier = created["stdout"].strip()
            containers.append(identifier)
            started = command(["docker", "start", identifier])
            freeze(directory / f"start-{replica}.json", started)
            if started["exit_code"]:
                raise FidelityError("isolated CPU inspection container could not start")
            info = json.loads(command(["docker", "inspect", identifier])["stdout"])[0]
            hidden = command(["docker", "exec", identifier, "sh", "-c",
                              "test ! -e /tests && test ! -e /solution && test -z \"$(find / -maxdepth 1 -name 'verifier*' -print)\""])
            processes = command(["docker", "top", identifier, "-eo", "comm,args"])
            check = {
                "container_id": identifier, "hidden_assets_absent": hidden["exit_code"] == 0,
                "mounts": info["Mounts"], "network": info["HostConfig"]["NetworkMode"],
                "readonly_rootfs": info["HostConfig"]["ReadonlyRootfs"],
                "device_requests": info["HostConfig"].get("DeviceRequests"),
                "process_inventory": processes,
                "process_memory_restored": False, "network_or_volume_state_restored": False,
            }
            if target:
                file_hash = command(["docker", "exec", identifier, "sha256sum", target])
                check["public_file_hash"] = file_hash
                if file_hash["exit_code"] or not check["hidden_assets_absent"] or info["Mounts"]:
                    raise FidelityError("scoped read prerequisites failed")
                read = command(["docker", "exec", identifier, "cat", target], timeout=10)
                freeze(directory / f"public-read-{replica}.json", read)
                check["read_stdout_sha256"] = hashlib.sha256(read["stdout"].encode()).hexdigest()
                check["read_exit_code"] = read["exit_code"]
            checks.append(check)
        row["checks"] = checks
        row["status"] = "INSPECTED_RESET_BLOCKED"
        if target:
            archived = load(trial / "agent/turn-001/tool-01/result.json")
            row["scoped_public_read_equal_across_fresh_images"] = (
                checks[0]["public_file_hash"]["stdout"] == checks[1]["public_file_hash"]["stdout"]
                and checks[0]["read_stdout_sha256"] == checks[1]["read_stdout_sha256"]
            )
            row["scoped_public_read_matches_archived_parent"] = (
                checks[0]["read_stdout_sha256"] == hashlib.sha256(archived["stdout"].encode()).hexdigest()
            )
        row["true_snapshot_restore_performed"] = False
    except Exception as error:
        row["status"] = "EXCLUDED_DIAGNOSTIC_FAILURE"
        row["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        cleanup = []
        for identifier in containers:
            current = command(["docker", "inspect", identifier])
            if current["exit_code"] == 0:
                info = json.loads(current["stdout"])[0]
                if info["Id"] != identifier or info["Config"]["Labels"].get("phiagent.fidelity") != str(output):
                    raise FidelityError("refusing cleanup of a container with different ownership")
                cleanup.append(command(["docker", "rm", "-f", identifier]))
        freeze(directory / "cleanup.json", cleanup)
        row["owned_containers_removed"] = all(item["exit_code"] == 0 for item in cleanup)
    freeze(directory / "gate.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v14-source", type=Path, required=True)
    parser.add_argument("--git-state", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "harnessmm-api-worker":
        raise FidelityError("real-prefix and CPU-image gates run only on authorized a800-4")
    args.output.mkdir(parents=True, exist_ok=False)
    freeze(args.output / "config.json", {
        "schema_version": "phiagent-terminal-fidelity-phase1/v1", "cases": list(CASES),
        "reference": "fixed parent5/18; no old-compiled/static substitution",
        "exposure": "OPENED DEVELOPMENT", "prefix_calls": 1, "seed": 0,
        "model_calls": 0, "verifier_calls": 0, "source_study": str(args.study),
        "created_at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "source_sha256": sha(Path(__file__)), "argv": sys.argv,
        "actor_service_launches": 0, "future_performance_context": 131072,
    })
    freeze(args.output / "git-state.json", load(args.git_state))
    freeze(args.output / "packages.json", command([sys.executable, "-m", "pip", "freeze", "--all"]))
    archive = archive_gate(args.study, args.v14_source, args.output)
    cases = [inspect_case(args.study, args.output, task) for task in CASES]
    result = {
        "archive_gate": archive, "cases": cases, "model_calls": 0, "verifier_calls": 0,
        "same_state_task_branch": "BLOCKED",
        "alternative": "Exact parent-output sharing before an intervention; prospective whole-task randomization when a verified environment branch is unavailable.",
        "not_a_benchmark_score": True, "method_gain_claim": False,
        "all_owned_containers_removed": all(row.get("owned_containers_removed", True) for row in cases),
    }
    freeze(args.output / "GATE.json", result)
    print(json.dumps({"gate": str(args.output / "GATE.json"), "same_state_task_branch": "BLOCKED",
                      "archive_shadow_exact": archive["shadow_exact"],
                      "first_normalized_semantic_difference": archive["first_normalized_semantic_difference"],
                      "cases": [{key: row.get(key) for key in ("task", "status", "scoped_public_read_equal_across_fresh_images",
                                "scoped_public_read_matches_archived_parent")} for row in cases],
                      "model_calls": 0, "all_owned_containers_removed": result["all_owned_containers_removed"]}, indent=2))


if __name__ == "__main__":
    main()
