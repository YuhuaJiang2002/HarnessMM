#!/usr/bin/env python3
"""Retire only empty Docker networks belonging to an attested native trial."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic  # noqa: E402


def network_owner(container: dict) -> dict:
    project = (container.get("Config", {}).get("Labels") or {}).get("com.docker.compose.project")
    if not project:
        return {"project": None, "networks": [], "scope": "not_a_compose_trial"}
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
        raise ValueError("Invalid attested Docker Compose project")
    networks = [
        {"id": value["NetworkID"], "name": name}
        for name, value in container["NetworkSettings"]["Networks"].items()
        if value.get("NetworkID")
    ]
    return {"project": project, "networks": networks}


def attested_project(receipt: dict) -> str:
    name = receipt["name"].lstrip("/")
    if (
        not re.fullmatch(r"[a-f0-9]{64}", receipt["container_id"])
        or not name.endswith("-main-1")
    ):
        raise ValueError("Unrecognized native main-container attestation")
    project = name.removesuffix("-main-1")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
        raise ValueError("Invalid archived native project identity")
    return project


def retire_attested_networks(receipt_path: Path, output: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    project = attested_project(receipt)
    ids = subprocess.check_output([
        "docker", "network", "ls", "-q", "--no-trunc",
        "--filter", "label=com.docker.compose.project=" + project,
    ], text=True, timeout=30).split()
    return retire_networks({
        "project": project, "networks": [{"id": identity} for identity in ids],
        "attestation": {
            "path": str(receipt_path), "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
    }, output)


def validate_empty_trial_network(network: dict, identity: str, project: str) -> bool:
    if (
        not re.fullmatch(r"[a-f0-9]{64}", identity)
        or network.get("Id") != identity or network.get("Driver") != "bridge"
        or network.get("Scope") != "local"
        or (network.get("Labels") or {}).get("com.docker.compose.project") != project
    ):
        raise ValueError("Network identity or native trial ownership differs")
    return not network.get("Containers")


def attest_undispatched_network_failure(native: dict, task_id: str) -> dict:
    agent = native.get("config", {}).get("agent", {})
    error = native.get("exception_info") or {}
    if (
        native.get("task_name") != task_id
        or "agent_execution" not in native or native["agent_execution"] is not None
        or native.get("agent_setup") is not None
        or native.get("verifier_result") is not None
        or "all predefined address pools have been fully subnetted"
        not in error.get("exception_message", "")
        or agent.get("import_path") not in {
            "scripts.run_deepseek_terminal_campaign:NativeCompletionAgent",
            "scripts.codex_terminal_agent:MeteredCodex",
            "scripts.industrial_terminal_agents:SharedPromptDeepSeek",
            "scripts.industrial_terminal_agents:SharedPromptCodex",
        }
    ):
        raise ValueError("Only a native pre-agent subnet-allocation failure can be retried")
    return {
        "classification": "docker_address_pool_exhausted_before_agent_setup",
        "task_id": task_id, "actor_calls": 0, "provider_calls": 0,
        "native_score": None, "score_imputed": False, "actor_trajectory_replayed": False,
    }


def retire_networks(owner: dict, output: Path) -> dict:
    record = {
        "at": datetime.now(timezone.utc).isoformat(), "owner": owner,
        "removed": [], "retained": [], "already_absent": [],
        "unrelated_networks_changed": False, "containers_or_data_removed": False,
    }
    write_json_atomic(output, {**record, "status": "INSPECTING"})
    try:
        for item in owner["networks"]:
            identity = item["id"]
            if not re.fullmatch(r"[a-f0-9]{64}", identity):
                raise ValueError("An attested full Docker network ID is required")
            inspected = subprocess.run(
                ["docker", "network", "inspect", identity], capture_output=True, text=True,
                timeout=30, check=False,
            )
            if inspected.returncode:
                if "not found" in inspected.stderr.lower() and identity in inspected.stderr:
                    record["already_absent"].append(identity)
                    continue
                raise RuntimeError(f"Network inspection failed: {inspected.stderr}")
            rows = json.loads(inspected.stdout)
            if len(rows) != 1:
                raise ValueError("Network inspection did not return one native identity")
            if not validate_empty_trial_network(rows[0], identity, owner["project"]):
                record["retained"].append({"id": identity, "reason": "connected_container_endpoints"})
                continue
            removed = subprocess.run(
                ["docker", "network", "rm", identity], capture_output=True, text=True,
                timeout=30, check=False,
            )
            if removed.returncode:
                raise RuntimeError(f"Owned network retirement failed: {removed.stderr}")
            record["removed"].append(identity)
            write_json_atomic(output, {**record, "status": "RETIRING"})
        record["status"] = "COMPLETE"
        write_json_atomic(output, record)
        return record
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write_json_atomic(output, {
            **record, "status": "BLOCKED", "error_type": type(error).__name__, "error": str(error),
        })
        raise


def registry_projects() -> dict[str, dict]:
    sources = list(Path("/opt/phiagent/work/deepseek-harness/20260929-v1/runs").glob(
        "terminal-campaign-*/executor/*/terminal-*/HARBOR_ENVIRONMENT.json"
    ))
    sources += list(Path("/opt/phiagent/work/codex-chisel/20260929-v1/terminal").glob(
        "chisel-comparison-v*/trials/*/*/*/*/NATIVE_CONTAINER.json"
    ))
    records = {}
    for path in sources:
        value = json.loads(path.read_text())
        project = attested_project(value)
        records[project] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "container_id": value["container_id"],
        }
    return records


def reclaim(output: Path) -> dict:
    if socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Network recovery is restricted to A800-4")
    if not output.resolve().is_relative_to("/path/to/data/phiagent/work/harness-terminal-recovery"):
        raise ValueError("Use a new campaign-owned recovery directory")
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "provenance.json", capture_provenance(ROOT, sys.argv, 20260928))
    registry = registry_projects()
    identities = subprocess.check_output(["docker", "network", "ls", "-q"], text=True).split()
    inventory = json.loads(subprocess.check_output(
        ["docker", "network", "inspect", *identities], text=True,
    ))
    write_json_atomic(output / "BEFORE.json", inventory)
    recovered = []
    for network in inventory:
        project = (network.get("Labels") or {}).get("com.docker.compose.project")
        if project not in registry or not validate_empty_trial_network(network, network["Id"], project):
            continue
        containers = subprocess.check_output([
            "docker", "ps", "-q", "--filter", "label=com.docker.compose.project=" + project,
        ], text=True, timeout=30).split()
        if containers:
            continue
        owner = {"project": project, "networks": [{"id": network["Id"], "name": network["Name"]}],
                 "attestation": registry[project]}
        recovered.append(retire_networks(owner, output / (network["Id"] + ".json")))
    result = {
        "status": "VERIFIED_EMPTY_NATIVE_NETWORKS_RETIRED",
        "at": datetime.now(timezone.utc).isoformat(), "networks_before": len(inventory),
        "retired_networks": sum(len(row["removed"]) for row in recovered),
        "attested_projects": len(registry), "runtime_or_policy_changes": False,
        "containers_or_data_removed": False, "active_workers_stopped": False,
        "retirements": recovered,
    }
    write_json_atomic(output / "RECOVERY.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reclaim(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
