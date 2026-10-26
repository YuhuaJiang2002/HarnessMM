#!/usr/bin/env python3
"""Matched, remote-only ALFWorld CHISEL-Repair whole-task development pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "phiagent-chisel-repair-alfworld/v1"
BASELINES = ("no_skill", "skillopt", "autosaddler", "old_ours")
PATCH_ARMS = ("static_patch", "conditional_patch", "random_time")
MODES = {
    "static_patch": "STATIC_PATCH",
    "conditional_patch": "CONDITIONAL_PATCH",
    "random_time": "RANDOM_TIME",
}
SKILLOPT_CHECKPOINT = "c8219398bf587dae9c33f832f97ed749ea0de978ef8627840df139e0fc3fff20"
SKILLOPT_COMMIT = "eb8c1e7bcbccdd80f9d422f12018fcd8e84ce19a"
SYSTEM = "You are an expert agent operating in the ALFRED Embodied Environment."
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
EXPOSURE = "OPENED_TASK_GROUPED_DEVELOPMENT"
TOOL_ERROR_SIGNAL_VERSION = "alfworld-public-command-feedback/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    pending.replace(path)


def capture(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(
        list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return {"command": list(command), "returncode": result.returncode, "output": result.stdout}


def floorplan(gamefile: str) -> str:
    path = Path(gamefile)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("json_2.1.1", "train"):
        raise ValueError("Only frozen relative ALFWorld TRAIN gamefiles are permitted")
    match = re.search(r"-(\d+)$", path.parent.parent.name)
    if not match:
        raise ValueError("Cannot infer room/floorplan group from TRAIN task directory")
    return f"FloorPlan{int(match.group(1))}"


def split_tasks(items: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    if not items or len({row["id"] for row in items}) != len(items):
        raise ValueError("Task IDs must be nonempty and unique")
    if len({row["gamefile"] for row in items}) != len(items):
        raise ValueError("Duplicate gamefile")
    groups: dict[str, list[str]] = {}
    for row in items:
        groups.setdefault(floorplan(str(row["gamefile"])), []).append(str(row["id"]))
    ranked = sorted(groups, key=lambda group: digest([seed, "discovery-floorplan-v1", group]))
    target = max(1, round(len(items) * 0.2))
    chosen: list[str] = []
    discovery_groups: list[str] = []
    for group in ranked:
        if len(chosen) >= target:
            break
        chosen.extend(groups[group])
        discovery_groups.append(group)
    selected = set(chosen)
    return {
        "rule": "Hash-rank complete floorplans; take groups until >=round(0.2*N); no outcomes",
        "seed": seed,
        "discovery": [row["id"] for row in items if row["id"] in selected],
        "evaluation": [row["id"] for row in items if row["id"] not in selected],
        "discovery_groups": discovery_groups,
        "evaluation_groups": [group for group in ranked if group not in discovery_groups],
    }


def freeze(args: argparse.Namespace) -> int:
    """Source-only manifest preparation; never imports a benchmark or reads scores."""
    from phiagent.learning.alfworld_autosaddler_pfn import render_task_skill

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source_experiment.resolve()
    transfer = args.skillopt_experiment.resolve()
    items_path = source / "splits/confirmation/val/items.json"
    items = read_json(items_path)
    if len(items) != 60:
        raise ValueError("This bounded opened-development pilot requires complete previous60")
    split = split_tasks(items, args.seed)
    if len(split["evaluation"]) < 36:
        raise ValueError("Too few floorplan-disjoint evaluation tasks")
    families = sorted({item["task_type"] for item in items})
    skills = {
        "no_skill": {family: "" for family in families},
    }
    skillopt_request = read_json(transfer / "request.json")
    skill = skillopt_request["components"]["global_skill"]
    if hashlib.sha256(skill.encode()).hexdigest() != SKILLOPT_CHECKPOINT:
        raise ValueError("SkillOpt released checkpoint differs")
    skills["skillopt"] = {family: skill for family in families}
    dependencies = [items_path, transfer / "request.json", source / "manifest.json"]
    labels = {
        "no_skill": "No Skill; new matched actor rollout",
        "skillopt": "SkillOpt released GPT-5.5-optimized artifact transfer; no new outer loop",
        "autosaddler": "AutoSaddler frozen optimized routed artifact transfer; no new outer loop",
        "old_ours": "Old Ours frozen routed skill transfer; common 50-step cap replaces old 39/49/50 caps",
    }
    old_caps: dict[str, Any] = {}
    for arm, old_name in (("autosaddler", "autosaddler"), ("old_ours", "ours_adm")):
        request_path = source / "evaluations/confirmation" / old_name / "request.json"
        request = read_json(request_path)
        dependencies.append(request_path)
        skills[arm] = {}
        for family in families:
            policy = request["family_policies"][family]
            skills[arm][family] = render_task_skill(policy["components"], family)[0]
            if arm == "old_ours":
                old_caps[family] = policy["policy_id"]
    baseline_records = {}
    for arm in BASELINES:
        policy = {"arm": arm, "label": labels[arm], "skills": skills[arm]}
        policy["harness_sha256"] = digest(policy)
        baseline_records[arm] = policy
    manifest = {
        "schema_version": SCHEMA, "created_at": now(), "exposure": EXPOSURE,
        "scientific_status": "UNVALIDATED_NEW_WHOLE_TASK_INTERVENTION_PILOT",
        "task_inventory": [
            {**item, "group_id": floorplan(item["gamefile"])} for item in items
        ],
        "split": split, "baselines": baseline_records,
        "budget": {
            "max_steps": 50, "max_completion_tokens": 2048, "temperature": 0.0,
            "history_length": 2, "request_timeout_seconds": 180,
            "provider_retries": 0, "candidate_count": 1, "proposer_calls": 1,
            "proposer_max_completion_tokens": 4096,
            "seed": args.seed, "random_window": 8,
            "memory_characters": 1500, "max_entries": 4,
        },
        "selection_rules": {
            "parent": "Discovery all-task official success descending; exact total tokens ascending; arm ID",
            "candidate": "One genuinely new served-LLM directive from selected-parent discovery failures only",
            "compiler": "Success first, then exact tokens; KEEP beats unsupported ties; CONDITIONAL must beat STATIC and RANDOM in discovery success",
            "stopping": "One candidate, one pass per task/arm; no valid-cell retries; no evaluation retuning",
        },
        "metric": {
            "name": "ALFWorld official episode success (TextWorld info.won)",
            "type": "binary_task_success", "aggregation": "all frozen task IDs, equal episode weight",
            "failure_mask": "Infrastructure-invalid is null, never zero; report coverage and bounds",
        },
        "identity_boundary": {
            "old_actor_not_matched": "Historical Qwen3.5-9B@main scores are not new27B controls",
            "old_ours_frozen_policy_ids": old_caps,
            "same_state_branching": False,
            "intervention": "Paired task/seed whole-task reruns; hash-randomized task-arm schedule",
            "random_control": "Common budget caps, not exact realized token/activation-count matching",
            "source_split": "TRAIN; previously opened60; no official valid_unseen calls",
            "fresh_confirmation": False, "certificate": False, "sota_claim": False,
            "new_outer_loop_baseline": False,
            "skillopt_source_commit": SKILLOPT_COMMIT,
        },
        "source_artifacts": [
            {"path": str(path), "sha256": file_hash(path)} for path in dependencies
        ],
        "source_git": capture(["git", "rev-parse", "HEAD"], cwd=ROOT),
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    shutil.copy2(__file__, output / "run_chisel_repair_alfworld.py")
    write_json(output / "readiness.json", {
        "status": "AWAITING_REMOTE_BIND", "manifest_sha256": manifest["manifest_sha256"],
        "discovery_tasks": len(split["discovery"]), "evaluation_tasks": len(split["evaluation"]),
        "opened_tasks": 60, "group_overlap": [],
        "remaining_gates": ["Pinned actor endpoint/revision/GPU provenance", "Remote runtime/data audit"],
        "command": (
            "python scripts/run_chisel_repair_alfworld.py bind --study-dir STUDY "
            "--skillopt-root SKILLOPT --data-root DATA --base-url URL --model-id MODEL "
            "--model-repository REPO --model-revision COMMIT --gpu-manifest GPU_JSON "
            "--visible-devices INDICES --authorized-hostname HOST; "
            "python scripts/run_chisel_repair_alfworld.py run --study-dir STUDY "
            "--phase discovery_baselines --workers 12"
        ),
    })
    print(json.dumps({"status": "FROZEN", "study_dir": str(output),
                      "discovery": len(split["discovery"]), "evaluation": len(split["evaluation"])}))
    return 0


def verify_manifest(study: Path) -> dict[str, Any]:
    value = read_json(study / "manifest.json")
    if value.get("manifest_sha256") != digest({k: v for k, v in value.items() if k != "manifest_sha256"}):
        raise ValueError("Frozen study manifest hash mismatch")
    return value


def remote_only(expected_host: str | None = None) -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Benchmark execution is remote-only; Linux authorized server required")
    if not expected_host or socket.gethostname() != expected_host:
        raise RuntimeError("Authorized executor hostname must match exactly")


def endpoint_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid actor endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must contain no credentials/query/fragment")
    return value.rstrip("/") if value.rstrip("/").endswith("/v1") else value.rstrip("/") + "/v1"


def get_models(base_url: str) -> dict[str, Any]:
    with urlopen(endpoint_url(base_url) + "/models", timeout=20) as response:
        return json.load(response)


def runtime_files(root: Path) -> None:
    workspace = root / "runtime_files"
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(workspace)
    tempfile.tempdir = str(workspace)


def alfworld_environment_mode(manifest: Mapping[str, Any]) -> tuple[str, bool]:
    mode = manifest.get("alfworld_environment_mode", "train")
    if mode not in ("train", "eval_out_of_distribution"):
        raise ValueError("Only the declared TRAIN or official valid_unseen ALF environment mode is supported")
    return mode, mode == "train"


def runtime_check(args: argparse.Namespace) -> int:
    remote_only(args.authorized_hostname)
    study = args.study_dir.resolve()
    manifest = verify_manifest(study)
    output = study / "cpu_runtime_preflight"
    output.mkdir(exist_ok=False)
    runtime_files(output)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ALFWORLD_DATA"] = str(args.data_root.resolve())
    os.environ["ALFWORLD_WORKER_START_METHOD"] = "fork"
    sys.path.insert(0, str(args.skillopt_root.resolve()))
    sys.path.insert(0, str(ROOT / "scripts"))
    write_json(output / "provenance.json", {
        "hostname": socket.gethostname(), "command": sys.argv, "python": sys.executable,
        "created_at": now(), "manifest_sha256": manifest["manifest_sha256"],
        "runner_sha256": file_hash(Path(__file__)), "CUDA_VISIBLE_DEVICES": "",
        "packages": capture([sys.executable, "-m", "pip", "freeze"]),
        "model_calls": 0, "seed": manifest["budget"]["seed"],
        "source_git_head": manifest["source_git"],
    })
    from run_alfworld_autosaddler_worker import install_alfworld_unused_expert_guard
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    from phiagent.learning.counterfactual_repair import parse_patch  # noqa: F401

    guard = install_alfworld_unused_expert_guard()
    first = next(item for item in manifest["task_inventory"]
                 if item["id"] == manifest["split"]["discovery"][0])
    env = None
    try:
        env = build_alfworld_env(1, "train", manifest["budget"]["seed"], True, [first["gamefile"]])
        observation, infos = env.reset({})
        actual_game = str(infos[0].get("extra.gamefile", ""))
        if not actual_game.endswith(first["gamefile"]):
            raise ValueError("Simulator reset loaded a different gamefile")
        write_json(output / "result.json", {
            "status": "CPU_RUNTIME_RESET_PASS", "id": first["id"], "gamefile": actual_game,
            "gamefile_sha256": file_hash(args.data_root / first["gamefile"]),
            "public_observation": observation["anchor"][0], "model_calls": 0,
            "infrastructure_guard": guard, "same_state_clone_validated": False,
            "created_at": now(),
        })
    except Exception as error:
        write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error)[:800],
        })
        raise
    finally:
        if env is not None:
            env.envs.close()
    print(json.dumps({"status": "CPU_RUNTIME_RESET_PASS", "study_dir": str(study)}))
    return 0


def binding_readiness(manifest: Mapping[str, Any], binding: Mapping[str, Any], study: Path) -> dict:
    mode, is_train = alfworld_environment_mode(manifest)
    result = {
        "status": "READY_FOR_MATCHED_DISCOVERY" if is_train else "READY_FOR_FROZEN_OFFICIAL_COHORT",
        "bound_at": now(), "binding_sha256": binding["binding_sha256"], "model_calls": 0,
    }
    if is_train:
        result.update({
            "discovery_tasks": len(manifest["split"]["discovery"]),
            "evaluation_tasks": len(manifest["split"]["evaluation"]),
            "command": [
                sys.executable, str(Path(__file__).resolve()), "run", "--study-dir", str(study),
                "--phase", "discovery_baselines", "--workers", "12",
            ],
        })
    else:
        result.update({
            "environment_mode": mode,
            "task_count": len(manifest["task_inventory"]),
            "task_counts_by_split": {
                name: len(items) for name, items in manifest["split"].items()
                if isinstance(items, list)
            },
            "execution_owner": "The frozen full-cohort runner, not the legacy discovery command",
        })
    return result


def bind(args: argparse.Namespace) -> int:
    remote_only(args.authorized_hostname)
    study = args.study_dir.resolve()
    manifest = verify_manifest(study)
    if (study / "binding.json").exists():
        raise FileExistsError("A bound study cannot be rebound")
    audit_root = study / "remote_readiness"
    audit_root.mkdir(exist_ok=False)
    runtime_files(audit_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    os.environ["ALFWORLD_DATA"] = str(args.data_root.resolve())
    os.environ["ALFWORLD_WORKER_START_METHOD"] = "fork"
    sys.path.insert(0, str(args.skillopt_root.resolve()))
    gpu = capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                   "--format=csv,noheader,nounits"])
    write_json(audit_root / "gpu_inventory.json", gpu)
    if gpu["returncode"] != 0:
        raise RuntimeError("GPU inventory inspection failed")
    devices = args.visible_devices.split(",")
    inventory = {}
    for line in gpu["output"].splitlines():
        if line.strip():
            index, uuid, *_ = (part.strip() for part in line.split(","))
            inventory[index] = inventory[uuid] = line.strip()
    if (not devices or any(device not in inventory for device in devices)
            or len({inventory[device] for device in devices}) != len(devices)):
        raise ValueError("Parent-selected physical GPU indices/UUIDs are absent or duplicated")
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.model_revision):
        raise ValueError("Model revision must be immutable commit/content hash, not main")
    provenance = read_json(args.gpu_manifest)
    if not provenance:
        raise ValueError("Parent GPU/model provenance must be nonempty")
    shutil.copy2(args.gpu_manifest, audit_root / "parent_gpu_manifest.json")
    for name in ("alfworld", "textworld", "gymnasium", "omegaconf", "numpy"):
        importlib.metadata.version(name)
    packages = dict(sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]
    ))
    frozen_skillopt = study / "source/SkillOpt"
    shutil.copytree(
        args.skillopt_root / "skillopt", frozen_skillopt / "skillopt",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*", ".DS_Store"),
    )
    sys.path.insert(0, str(frozen_skillopt))
    source_hashes = {}
    for path in sorted((frozen_skillopt / "skillopt").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml") and not path.name.startswith("._"):
            source_hashes[str(path.relative_to(frozen_skillopt))] = file_hash(path)
    installed_sources = {}
    for name in ("alfworld", "textworld"):
        specification = __import__(name).__file__
        package_root = Path(specification).parent
        installed_sources[name] = {
            str(path.relative_to(package_root)): file_hash(path)
            for path in sorted(package_root.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".so")
        }
    write_json(audit_root / "installed_scorer_source_hashes.json", installed_sources)
    data_hashes = {}
    for item in manifest["task_inventory"]:
        game = args.data_root / item["gamefile"]
        if not game.is_file():
            raise FileNotFoundError(game)
        data_hashes[item["gamefile"]] = file_hash(game)
        for path in sorted(game.parent.iterdir()):
            if path.is_file() and path.suffix in (".json", ".pddl", ".z8", ".ulx"):
                data_hashes[str(path.relative_to(args.data_root))] = file_hash(path)
        metadata = read_json(game.parent / "traj_data.json")
        scene = str(metadata.get("scene", {}).get("floor_plan", ""))
        if scene and re.sub(r"^FloorPlan", "", scene) != item["group_id"].removeprefix("FloorPlan"):
            raise ValueError("Directory-derived floorplan disagrees with scene metadata")
    for path in sorted((args.data_root / "logic").iterdir()):
        if path.is_file():
            data_hashes[str(path.relative_to(args.data_root))] = file_hash(path)
    endpoints = [endpoint_url(url) for url in args.base_url]
    responses = [get_models(url) for url in endpoints]
    if any(args.model_id not in {entry["id"] for entry in result.get("data", [])} for result in responses):
        raise RuntimeError("Served model list disagrees with frozen actor model ID")
    runner_hashes = {"scripts/run_chisel_repair_alfworld.py": file_hash(Path(__file__))}
    for name in ("phiagent/learning/counterfactual_repair.py",
                 "phiagent/learning/stage_aware_adm.py",
                 "phiagent/learning/alfworld_autosaddler_pfn.py",
                 "scripts/run_alfworld_autosaddler_worker.py"):
        path = ROOT / name
        runner_hashes[name] = file_hash(path)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_alfworld_autosaddler_worker import install_alfworld_unused_expert_guard
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    from phiagent.learning.counterfactual_repair import parse_patch  # noqa: F401

    guard = install_alfworld_unused_expert_guard()
    eval_dataset, is_train = alfworld_environment_mode(manifest)
    first = (next(item for item in manifest["task_inventory"]
                  if item["id"] == manifest["split"]["discovery"][0])
             if is_train else manifest["task_inventory"][0])
    check_env = build_alfworld_env(
        1, eval_dataset, manifest["budget"]["seed"], is_train, [first["gamefile"]],
    )
    try:
        observation, infos = check_env.reset({})
        actual_game = str(infos[0].get("extra.gamefile", ""))
        if not actual_game.endswith(first["gamefile"]):
            raise ValueError("Simulator reset loaded a different gamefile")
        write_json(audit_root / "environment_reset.json", {
            "id": first["id"], "gamefile": actual_game,
            "public_observation": observation["anchor"][0], "model_calls": 0,
            "infrastructure_guard": guard, "same_state_clone_validated": False,
        })
    finally:
        check_env.envs.close()
    binding = {
        "schema_version": SCHEMA, "bound_at": now(), "manifest_sha256": manifest["manifest_sha256"],
        "hostname": socket.gethostname(), "python": sys.executable,
        "skillopt_root": str(frozen_skillopt), "data_root": str(args.data_root.resolve()),
        "model": {"id": args.model_id, "repository": args.model_repository,
                  "revision": args.model_revision, "endpoints": endpoints,
                  "thinking": "disabled", "transport": "OpenAI-compatible chat completions"},
        "gpu": {"visible_devices": args.visible_devices,
                "selected_physical_devices": [inventory[device] for device in devices],
                "parent_selection_sha256": file_hash(args.gpu_manifest),
                "executor_owns_model_service": False},
        "source_hashes": source_hashes, "data_hashes": data_hashes,
        "alfworld_environment_mode": eval_dataset,
        "tool_error_signal_version": TOOL_ERROR_SIGNAL_VERSION,
        "installed_scorer_sources_sha256": digest(installed_sources),
        "adapter_source_hashes": runner_hashes, "packages": packages,
        "git_head": capture(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "git_diff": capture(["git", "diff", "--", "scripts/run_chisel_repair_alfworld.py",
                             "phiagent/learning/counterfactual_repair.py"], cwd=ROOT),
        "command": sys.argv, "environment": {"CUDA_VISIBLE_DEVICES": args.visible_devices},
    }
    binding["harness_hashes"] = {
        arm: digest({"policy": policy["harness_sha256"], "budget": manifest["budget"],
                     "alfworld_environment_mode": eval_dataset,
                     "model": binding["model"], "adapter_source_hashes": runner_hashes})
        for arm, policy in manifest["baselines"].items()
    }
    binding["binding_sha256"] = digest(binding)
    write_json(study / "binding.json", binding)
    write_json(audit_root / "models.json", responses)
    readiness = binding_readiness(manifest, binding, study)
    write_json(study / "readiness.json", readiness)
    print(json.dumps({"status": readiness["status"], "study_dir": str(study)}))
    return 0


def load_bound(study: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_manifest(study)
    binding = read_json(study / "binding.json")
    remote_only(binding["hostname"])
    if binding.get("binding_sha256") != digest({k: v for k, v in binding.items() if k != "binding_sha256"}):
        raise ValueError("Runtime binding hash mismatch")
    if binding["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Runtime belongs to a different frozen study")
    for name, expected_hash in binding["adapter_source_hashes"].items():
        if file_hash(ROOT / name) != expected_hash:
            raise ValueError("Frozen execution source changed after runtime binding")
    return manifest, binding


def provider_call(
    binding: Mapping[str, Any], *, endpoint_index: int, messages: Sequence[Mapping[str, str]],
    max_tokens: int, seed: int, output: Path, timeout: int, stage: str,
    response_schema: Mapping[str, Any] | None = None,
    decision_token_ids: Sequence[int] | None = None,
) -> tuple[str, dict[str, int]]:
    output.mkdir(parents=True, exist_ok=False)
    model = binding["model"]
    base_url = model["endpoints"][endpoint_index % len(model["endpoints"])]
    payload = {
        "model": model["id"], "messages": list(messages), "temperature": 0,
        "max_tokens": max_tokens, "seed": seed, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "alfworld_module_control", "strict": True, "schema": dict(response_schema),
        }}
    if decision_token_ids is not None:
        if (stage != "local" or max_tokens != 1 or response_schema is not None
                or not 2 <= len(decision_token_ids) <= 63
                or len(set(decision_token_ids)) != len(decision_token_ids)
                or any(type(token) is not int or token < 0 for token in decision_token_ids)):
            raise ValueError("Single-token decisions require a bounded unique native token set")
        payload.update(allowed_token_ids=list(decision_token_ids), logprobs=True,
                       top_logprobs=len(decision_token_ids))
    write_json(output / "request.json", {"endpoint": base_url, "stage": stage, "body": payload})
    started = time.monotonic()
    request = Request(base_url + "/chat/completions", json.dumps(payload).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
        (output / "response.json").write_text(raw)
        value = json.loads(raw)
        usage = value.get("usage") or {}
        if any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_KEYS):
            raise ValueError("Missing/nonintegral actual provider token usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("Inconsistent provider token accounting")
        write_json(output / "accounting.json", {
            "usage": {key: usage[key] for key in USAGE_KEYS}, "calls": 1,
            "elapsed_seconds": time.monotonic() - started, "model": value.get("model"),
            "finish_reason": (value.get("choices") or [{}])[0].get("finish_reason"),
            "valid": True,
        })
        if value.get("model") != model["id"]:
            raise ValueError("Provider response model differs from exact requested actor")
        content = value["choices"][0]["message"].get("content") or ""
        return str(content).strip(), {key: usage[key] for key in USAGE_KEYS}
    except Exception as error:
        if isinstance(error, HTTPError):
            try:
                (output / "http_error_body.txt").write_text(error.read().decode(errors="replace"))
            finally:
                error.close()
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "http_status": error.code if isinstance(error, HTTPError) else None,
            "elapsed_seconds": time.monotonic() - started, "valid": False,
            "cost_unknown": not (output / "accounting.json").exists(),
        })
        raise


UNMETERED_PROVIDER_CALL = provider_call


def tool_problem(observation: str) -> bool:
    """Recognize public command-error feedback, not the projection's XML format flag."""
    return bool(re.search(
        r"nothing happens|cannot |can't |not possible|don't understand|invalid action|not a valid",
        observation, re.IGNORECASE,
    ))


def task_call_ledger(output: Path, attempted_calls: int) -> dict[str, Any]:
    accounted = [read_json(path) for path in sorted((output / "calls").glob("*/accounting.json"))]
    return {
        "usage": {key: sum(record["usage"][key] for record in accounted) for key in USAGE_KEYS},
        "calls": attempted_calls, "accounted_calls": len(accounted),
        "cost_complete": len(accounted) == attempted_calls,
    }


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def run_episode(job: Mapping[str, Any]) -> dict[str, Any]:
    study = Path(job["study"])
    manifest, binding = load_bound(study)
    item = job["item"]
    arm = job["arm"]
    budget = manifest["budget"]
    output = study / job["phase"] / arm / digest(item["id"])[:16]
    output.mkdir(parents=True, exist_ok=False)
    runtime_files(output)
    os.chdir(output)
    seed = budget["seed"] + int(digest(["episode-seed", item["id"]])[:7], 16)
    if "alfworld_modular_rsi" in job:
        from phiagent.learning.alfworld_modular_rsi import episode_settings

        budget, seed = episode_settings(budget, job["alfworld_modular_rsi"])
    row = {
        "id": item["id"], "group_id": item["group_id"], "gamefile": item["gamefile"],
        "task_type": item["task_type"], "method": arm, "phase": job["phase"],
        "seed": seed, "official_success": None, "valid": False, "failure_type": None,
        "usage": {key: 0 for key in USAGE_KEYS}, "calls": 0, "attempted_calls": 0,
        "cost_complete": True, "turns": 0, "elapsed_seconds": 0,
        "cpu_seconds": 0, "environment_step_seconds": 0, "router_model_calls": 0,
        "model_id": binding["model"]["id"], "model_revision": binding["model"]["revision"],
        "manifest_sha256": manifest["manifest_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "artifact_dir": str(output), "parent_harness_sha256": job["policy"]["harness_sha256"],
        "repair_mode": MODES.get(arm, "KEEP_PARENT"),
        "tool_error_signal_version": TOOL_ERROR_SIGNAL_VERSION,
    }
    if "alfworld_modular_rsi" in job:
        row["alfworld_modular_rsi"] = dict(job["alfworld_modular_rsi"])
    write_json(output / "job.json", dict(job))
    env = None
    started = time.monotonic()
    cpu_started = cpu_seconds()
    conversation = []
    session = None
    jev_driver = None
    progress_session = False
    frontier_session = False
    try:
        if "jev" in job["policy"] or "local_discriminator" in job["policy"]:
            from scripts.alfworld_jev_runtime import EpisodeDriver

            if "jev" in job["policy"] and "local_discriminator" in job["policy"]:
                raise ValueError("External Jev and local discriminator cannot be combined")
            if arm in PATCH_ARMS or any(key.startswith(("alfworld_progress", "alfworld_search_frontier",
                    "alfworld_skill_compaction", "dual_alfworld")) for key in job):
                raise ValueError("Jev interleaving cannot overlap other harness interventions")
            protocol = job["policy"].get("local_discriminator", job["policy"].get("jev"))
            jev_driver = EpisodeDriver(binding=binding, budget=budget, protocol=protocol,
                                       output=output, seed=seed)
            row["decision_backend"] = protocol.get("backend", "cot_jev_interleaved")
        if job.get("dual_alfworld_fact_carryover") and "alfworld_progress" not in job:
            raise ValueError("Fact carryover requires the unchanged incumbent progress tracker")
        os.environ["CUDA_VISIBLE_DEVICES"] = binding["gpu"]["visible_devices"]
        os.environ["ALFWORLD_DATA"] = binding["data_root"]
        os.environ["ALFWORLD_WORKER_START_METHOD"] = "fork"
        sys.path.insert(0, binding["skillopt_root"])
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_alfworld_autosaddler_worker import install_alfworld_unused_expert_guard
        from skillopt.envs.alfworld.rollout import (
            _build_skill_prompt, _extract_action, build_alfworld_env,
        )

        row["infrastructure_guard"] = install_alfworld_unused_expert_guard()
        if arm in PATCH_ARMS:
            from phiagent.learning.counterfactual_repair import RepairMode, RepairSession, parse_patch

            patch = parse_patch(read_json(study / "proposal/patch.json"))
            session = RepairSession(
                patch, mode=RepairMode(MODES[arm]), task_id=item["id"],
                expected_parent_sha256=job["policy"]["harness_sha256"],
                seed=budget["seed"], random_window=budget["random_window"],
            )
        eval_dataset, is_train = alfworld_environment_mode(manifest)
        row["alfworld_environment_mode"] = eval_dataset
        env = build_alfworld_env(1, eval_dataset, seed, is_train, [item["gamefile"]])
        obs, infos = env.reset({})
        if job.get("verify_native_interface") or job.get("dual_alfworld_verify_native_interface"):
            if not str(infos[0].get("extra.gamefile", "")).endswith(item["gamefile"]):
                raise ValueError("Dual ALFWorld reset changed the frozen native gamefile")
            if not is_train and "extra.expert_plan" in infos[0]:
                raise ValueError("Training-only expert plan leaked into official evaluation")
        initial = obs["anchor"][0]
        row["task_description"] = initial.split("Your task is to: ", 1)[-1].strip()
        write_json(output / "initial_observation.json", {"observation": initial})
        if "alfworld_search_frontier" in job:
            from alfworld_search_frontier import SearchFrontierSession, VERSION

            if "alfworld_progress" in job or session is not None:
                raise ValueError("Frontier and legacy repair sessions cannot overlap")
            frontier = job["alfworld_search_frontier"]
            session = SearchFrontierSession(
                read_json(Path(frontier["patch_path"])), mode=frontier["mode"], task_id=item["id"],
                expected_parent_sha256=job["policy"]["harness_sha256"], seed=budget["seed"],
                task_text=row["task_description"], initial_observation=initial, config=frontier["config"],
            )
            progress_session = frontier_session = True
            row["repair_mode"] = frontier["mode"]
            row["public_progress_version"] = VERSION
        if "alfworld_progress" in job:
            from alfworld_public_progress import PublicProgressSession, TRACKER_VERSION

            progress = job["alfworld_progress"]
            progress_class = PublicProgressSession
            if job.get("dual_alfworld_fact_carryover"):
                from dual_alfworld_capsule import FactCarryoverSession, VERSION

                if job["dual_alfworld_fact_carryover"] != VERSION:
                    raise ValueError("Unknown dual ALFWorld fact-carryover version")
                progress_class = FactCarryoverSession
            session = progress_class(
                read_json(Path(progress["patch_path"])), mode=progress["mode"], task_id=item["id"],
                expected_parent_sha256=job["policy"]["harness_sha256"],
                seed=budget["seed"], task_text=row["task_description"], initial_observation=initial,
                config=progress["config"],
            )
            progress_session = True
            row["repair_mode"] = progress["mode"]
            row["public_progress_version"] = TRACKER_VERSION
        skill = job["policy"]["skills"][item["task_type"]]
        if job.get("dual_alfworld_policy_capsule"):
            from dual_alfworld_capsule import VERSION, compact_policy

            if job["dual_alfworld_policy_capsule"] != VERSION or job.get("alfworld_skill_compaction"):
                raise ValueError("Unknown or overlapping dual ALFWorld policy capsule")
            skill, record = compact_policy(skill, row["task_description"])
            row["dual_alfworld_policy_capsule"] = record
            write_json(output / "dual_alfworld_policy_capsule.json", record)
        if job.get("alfworld_skill_compaction"):
            from alfworld_skill_compaction import VERSION, compilation_record, compile_skill
            from alfworld_extractive_compaction import VERSION as EXTRACTIVE_VERSION

            if job["alfworld_skill_compaction"] == EXTRACTIVE_VERSION:
                from alfworld_extractive_compaction import compilation_record, compile_skill
            elif job["alfworld_skill_compaction"] != VERSION:
                raise ValueError("Unknown ALFWorld skill-compaction version")
            compiled = compile_skill(skill, row["task_description"])
            row["skill_compaction"] = compilation_record(skill, compiled)
            write_json(output / "skill_compaction.json", row["skill_compaction"])
            skill = compiled.text
        skill_prompt = _build_skill_prompt(skill)
        won = False
        done = False
        for turn in range(budget["max_steps"]):
            prompt = (skill_prompt + "\n" if skill_prompt else "") + obs["text"][0]
            decision = None
            if session is not None:
                if frontier_session:
                    decision = session.before_actor(
                        turn_index=turn, remaining_steps=budget["max_steps"] - turn,
                        public_prompt=obs["text"][0], stage="EXECUTE",
                    )
                else:
                    decision = session.before_actor(
                        turn_index=turn, remaining_steps=budget["max_steps"] - turn, stage="EXECUTE",
                    )
                if decision.message:
                    prompt += "\n\n" + decision.message
            if job.get("dual_alfworld_record_native_input"):
                from dual_alfworld_capsule import VERSION

                if job["dual_alfworld_record_native_input"] != VERSION:
                    raise ValueError("Unknown dual ALFWorld input-audit version")
                native_commands = None
                if job.get("dual_alfworld_verify_native_interface"):
                    native_commands = [
                        command for command in env.envs.get_admissible_commands[0] if command != "help"
                    ]
                    block = "\n ".join(f"'{command}'" for command in native_commands)
                    marker = "Your admissible actions of the current situation are: ["
                    if (marker not in obs["text"][0]
                            or obs["text"][0].rsplit(marker, 1)[1].split("].", 1)[0] != block):
                        raise ValueError("Native actor-visible admissible commands differ from environment API")
                write_json(output / "native_inputs" / f"{turn:03d}.json", {
                    "native_text": obs["text"][0],
                    "native_text_sha256": hashlib.sha256(obs["text"][0].encode()).hexdigest(),
                    "skill_prompt": skill_prompt,
                    "appended_message": decision.message if decision is not None else None,
                    "composed_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "native_input_truncated": False,
                    "native_admissible_commands": native_commands,
                })
            hybrid_decision = None
            if jev_driver is not None:
                native_commands = [command for command in env.envs.get_admissible_commands[0] if command != "help"]
                block = "\n ".join(f"'{command}'" for command in native_commands)
                marker = "Your admissible actions of the current situation are: ["
                if marker not in obs["text"][0] or obs["text"][0].rsplit(marker, 1)[1].split("].", 1)[0] != block:
                    raise ValueError("Jev options differ from the original public admissible-command list")
                hybrid = jev_driver.step(turn=turn, task=row["task_description"], prompt=prompt,
                    skill=skill, observation=str(obs["anchor"][0]),
                    actions=native_commands, history=[{"action": step["action"], "observation": step["observation"]}
                                                       for step in conversation])
                response, usage, hybrid_decision = hybrid["text"], hybrid["usage"], hybrid["decision"]
                actual_calls = (hybrid_decision["cot_calls"] + hybrid_decision["jev_calls"]
                                + hybrid_decision.get("local_calls", 0))
                row["attempted_calls"] += actual_calls
                row["calls"] += actual_calls
            else:
                row["attempted_calls"] += 1
                response, usage = provider_call(
                    binding, endpoint_index=job["endpoint_index"],
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    max_tokens=budget["max_completion_tokens"], seed=seed,
                    output=output / "calls" / f"{turn:03d}",
                    timeout=budget["request_timeout_seconds"], stage="actor",
                )
                row["calls"] += 1
            for key in USAGE_KEYS:
                row["usage"][key] += usage[key]
            actor_response = response
            if not response:
                response = "<think>empty model response</think><action>look</action>"
            elif _extract_action(response) is None:
                response = "<think>missing action tag</think><action>look</action>"
            action = (_extract_action(response) or "").lower()
            step_started = time.monotonic()
            obs, rewards, dones, infos = env.step([response])
            row["environment_step_seconds"] += time.monotonic() - step_started
            feedback = str(obs["anchor"][0])
            projection_format_valid = bool(infos[0].get("is_action_valid", False))
            environment_command_error = tool_problem(feedback)
            if session is not None:
                session.observe(
                    action or "<empty-action>", feedback,
                    tool_error=environment_command_error,
                )
            done = bool(dones[0])
            won = bool(infos[0].get("won", False)) if done else False
            conversation.append({
                "turn": turn, "action": action, "model_response": actor_response,
                "observation": feedback,
                "projection_format_valid": projection_format_valid,
                "environment_command_error": environment_command_error,
                "done": done,
                "official_won": won, "usage": usage,
                "repair": decision.trace if decision is not None else None,
            })
            if hybrid_decision is not None:
                conversation[-1]["hybrid_decision"] = hybrid_decision
                conversation[-1]["response_kind"] = (
                    "native_action_wrapper_not_generated_cot" if hybrid_decision["route"] in ("jev", "local", "deterministic") else "cot_model_response"
                )
            row["turns"] = turn + 1
            write_json(output / "conversation.json", conversation)
            if done:
                break
        row.update(valid=True, official_success=int(won),
                   outcome_type="official_episode_success",
                   stop_reason="environment_done" if done else "common_step_cap",
                   hard=int(won), agent_ok=True)
    except Exception as error:
        row["failure_type"] = type(error).__name__
        row["failure_message"] = str(error)[:800]
        row["cost_complete"] = row["attempted_calls"] == row["calls"]
        row["outcome_type"] = "infrastructure_invalid"
        row["agent_ok"] = False
        write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error)[:800],
            "turn": row["turns"], "attempted_calls": row["attempted_calls"],
        })
    finally:
        if env is not None:
            try:
                env.envs.close()
            except Exception as error:
                write_json(output / "close_failure.json", {"type": type(error).__name__})
        if session is not None:
            write_json(output / "repair_trace.json", session.traces)
            if progress_session:
                write_json(output / "public_progress.json", session.public_progress_snapshot())
        row.update(jev_driver.ledger() if jev_driver is not None else task_call_ledger(output, row["attempted_calls"]))
        row["elapsed_seconds"] = time.monotonic() - started
        row["cpu_seconds"] = cpu_seconds() - cpu_started
        row["tool_calls"] = row["turns"]
        write_json(output / "result.json", row)
    return row


def arm_summary(rows: Sequence[Mapping[str, Any]], total: int) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    successes = sum(row["official_success"] for row in valid)
    complete = len(valid) == total
    return {
        "tasks": total, "completed_cells": len(rows), "valid_tasks": len(valid),
        "invalid_tasks": len(rows) - len(valid), "successes_observed_valid": successes,
        "successes": successes if complete else None,
        "official_success_rate": successes / total if complete else None,
        "coverage": len(valid) / total,
        "all_task_success_bounds": [successes / total, (successes + total - len(valid)) / total],
        "usage": {key: sum(row["usage"][key] for row in rows) for key in USAGE_KEYS},
        "calls": sum(row["calls"] for row in rows),
        "attempted_calls": sum(row["attempted_calls"] for row in rows),
        "cost_complete": len(rows) == total and all(row["cost_complete"] for row in rows),
        "elapsed_seconds_sum": sum(row["elapsed_seconds"] for row in rows),
        "cpu_seconds_sum": sum(row.get("cpu_seconds", 0) for row in rows),
        "tool_calls": sum(row.get("tool_calls", 0) for row in rows),
        "valid_for_selection": complete and all(row["cost_complete"] for row in rows),
    }


def choose_parent(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    if set(summaries) != set(BASELINES) or not all(s["valid_for_selection"] for s in summaries.values()):
        raise ValueError("Parent selection requires every frozen baseline and task, with complete costs")
    return min(BASELINES, key=lambda arm: (
        -summaries[arm]["successes"], summaries[arm]["usage"]["total_tokens"], arm,
    ))


def compile_choice(summaries: Mapping[str, Mapping[str, Any]], parent: str) -> str:
    keep = summaries[parent]
    if not all(summaries[arm]["valid_for_selection"] for arm in (parent, *PATCH_ARMS)):
        return "KEEP_PARENT"
    static = summaries["static_patch"]
    conditional = summaries["conditional_patch"]
    random = summaries["random_time"]
    if (conditional["successes"] > max(keep["successes"], static["successes"], random["successes"])):
        return "CONDITIONAL_PATCH"
    if static["successes"] > keep["successes"] or (
        static["successes"] == keep["successes"]
        and static["usage"]["total_tokens"] < keep["usage"]["total_tokens"]
    ):
        return "STATIC_PATCH"
    return "KEEP_PARENT"


def run(args: argparse.Namespace) -> int:
    study = args.study_dir.resolve()
    manifest, binding = load_bound(study)
    if not 1 <= args.workers <= 24:
        raise ValueError("Bounded ALFWorld worker count must be in 1..24")
    phase = args.phase
    target = study / phase
    target.mkdir(exist_ok=False)
    split_name = "evaluation" if phase == "evaluation" else "discovery"
    items = [item for item in manifest["task_inventory"] if item["id"] in manifest["split"][split_name]]
    policies = {
        arm: {**policy, "harness_sha256": binding["harness_hashes"][arm]}
        for arm, policy in manifest["baselines"].items()
    }
    arms = list(BASELINES)
    parent = None
    if phase != "discovery_baselines":
        parent = read_json(study / "parent_selection.json")["selected_parent"]
        if not (study / "proposal/patch.json").is_file():
            raise ValueError("A frozen new served-LLM patch is required")
        for arm in PATCH_ARMS:
            policies[arm] = policies[parent]
        arms = list(PATCH_ARMS) if phase == "discovery_patches" else [*BASELINES, *PATCH_ARMS]
        if phase == "evaluation" and not (study / "compiler.json").is_file():
            raise ValueError("Discovery compiler choice must be frozen before evaluation")
    jobs = [
        {"study": str(study), "phase": phase, "arm": arm, "item": item,
         "policy": policies[arm], "endpoint_index": int(digest([item["id"], arm])[:7], 16)}
        for item in items for arm in arms
    ]
    jobs.sort(key=lambda job: digest([manifest["budget"]["seed"], phase, job["item"]["id"], job["arm"]]))
    write_json(target / "frozen_schedule.json", {
        "created_at": now(), "command": sys.argv, "workers": args.workers,
        "manifest_sha256": manifest["manifest_sha256"], "binding_sha256": binding["binding_sha256"],
        "cells": [{"id": job["item"]["id"], "arm": job["arm"]} for job in jobs],
    })
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_episode, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as error:
                write_json(target / f"worker_failure-{digest([job['arm'], job['item']['id']])[:16]}.json",
                           {"task_id": job["item"]["id"], "arm": job["arm"],
                            "type": type(error).__name__, "message": str(error)[:500]})
                row = {
                    "id": job["item"]["id"], "group_id": job["item"]["group_id"],
                    "gamefile": job["item"]["gamefile"], "method": job["arm"], "phase": phase,
                    "valid": False, "official_success": None, "outcome_type": "infrastructure_invalid",
                    "failure_type": type(error).__name__, "failure_message": str(error)[:500],
                    "usage": dict.fromkeys(USAGE_KEYS, 0), "calls": 0, "attempted_calls": 0,
                    "cost_complete": False, "call_count_unknown": True, "elapsed_seconds": 0,
                    "model_id": binding["model"]["id"], "model_revision": binding["model"]["revision"],
                }
            rows.append(row)
            write_json(target / "heartbeat.json", {
                "at": now(), "completed": len(rows), "total": len(jobs),
                "valid": sum(row["valid"] for row in rows),
            })
            print(json.dumps({"phase": phase, "completed": len(rows), "total": len(jobs),
                              "method": row["method"], "id": row["id"],
                              "valid": row["valid"], "success": row["official_success"]}), flush=True)
    summaries = {
        arm: arm_summary([row for row in rows if row["method"] == arm], len(items)) for arm in arms
    }
    result = {
        "schema_version": SCHEMA, "phase": phase, "exposure": EXPOSURE,
        "completed_at": now(), "task_ids": [item["id"] for item in items],
        "model": binding["model"], "manifest_sha256": manifest["manifest_sha256"],
        "binding_sha256": binding["binding_sha256"], "summaries": summaries,
        "rows": sorted(rows, key=lambda row: (row["method"], row["id"])),
    }
    write_json(target / "result.json", result)
    if phase == "discovery_baselines":
        try:
            parent = choose_parent(summaries)
        except ValueError:
            return 2
        write_json(study / "parent_selection.json", {
            "selected_at": now(), "selected_parent": parent,
            "parent_harness_sha256": policies[parent]["harness_sha256"],
            "rule": manifest["selection_rules"]["parent"],
            "discovery_result_sha256": file_hash(target / "result.json"),
            "selection_history": summaries, "evaluation_opened": False,
        })
    elif phase == "discovery_patches":
        baseline_summaries = read_json(study / "discovery_baselines/result.json")["summaries"]
        summaries.update(baseline_summaries)
        write_json(study / "compiler.json", {
            "compiled_at": now(), "mode": compile_choice(summaries, parent),
            "parent": parent, "patch_sha256": file_hash(study / "proposal/patch.json"),
            "rule": manifest["selection_rules"]["compiler"], "selection_history": summaries,
            "evaluation_opened": False, "claim": "Development-selected pilot policy, not a certificate",
        })
    return 0 if all(summary["valid_for_selection"] for summary in summaries.values()) else 2


def campaign(args: argparse.Namespace) -> int:
    """Execute the fixed campaign once; failed gates never trigger outcome retries."""
    for phase in ("discovery_baselines", "discovery_patches", "evaluation"):
        if phase == "discovery_patches":
            propose(args)
        args.phase = phase
        code = run(args)
        summarize(args)
        if code:
            return code
    return 0


def extract_json_object(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            result, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    raise ValueError("Proposer returned no JSON object")


def propose(args: argparse.Namespace) -> int:
    study = args.study_dir.resolve()
    manifest, binding = load_bound(study)
    from phiagent.learning.counterfactual_repair import parse_patch

    parent_record = read_json(study / "parent_selection.json")
    parent = parent_record["selected_parent"]
    output = study / "proposal"
    output.mkdir(exist_ok=False)
    discovery = read_json(study / "discovery_baselines/result.json")
    failures = [
        row for row in discovery["rows"]
        if row["method"] == parent and row["valid"] and row["official_success"] == 0
    ]
    failures.sort(key=lambda row: digest([manifest["budget"]["seed"], row["id"]]))
    packets = []
    for row in failures[:6]:
        conversation = read_json(Path(row["artifact_dir"]) / "conversation.json")
        public_steps = [
            {key: step[key] for key in (
                "turn", "action", "observation",
                "projection_format_valid", "environment_command_error",
            )}
            for step in conversation
        ]
        for step in public_steps:
            step["observation"] = step["observation"][:600]
        packets.append({
            "task_description": row["task_description"],
            "first_steps": public_steps[:4], "last_steps": public_steps[-8:],
        })
    write_json(output / "discovery_public_packets.json", {
        "discovery_ids": [row["id"] for row in failures[:6]], "packets": packets,
        "selection": "Hash-ranked selected-parent discovery failures; no evaluation traces/outcomes",
    })
    policy = manifest["baselines"][parent]
    instructions = (
        "Propose exactly ONE genuinely NEW generic recovery-memory directive for an ALFWorld actor. "
        "Use ONLY these public discovery failure traces. Do not include answers, task IDs, concrete object "
        "instance numbers, scene numbers, or memorized task-specific locations. Do not change the goal, "
        "scorer, tools, actor budget, tool parser, or environment. Repair one reusable failure mechanism "
        "with bounded memory of observed state and failed actions, using ordinary environment feedback. "
        "projection_format_valid describes response-tag formatting "
        "only, not command execution; environment_command_error is a separate feedback-derived signal. "
        "It must add substance beyond the "
        "parent's frozen skill, not merely rename it. Return JSON with directive (1..1800 chars), "
        "trigger_kind (any_problem/tool_error/stagnation/repeat), minimum_count (1..3), and rationale. "
        "No additional candidates. If evidence is weak still provide a diagnostic generic hypothesis "
        "and explicitly say evidence is weak.\n\n"
        + json.dumps({"parent_skills": policy["skills"], "public_discovery_failures": packets},
                     ensure_ascii=False)
    )
    budget = manifest["budget"]
    content, usage = provider_call(
        binding, endpoint_index=0,
        messages=[{"role": "system", "content": "You synthesize bounded, outcome-blind harness repairs."},
                  {"role": "user", "content": instructions}],
        max_tokens=budget["proposer_max_completion_tokens"], seed=budget["seed"],
        output=output / "llm_call", timeout=budget["request_timeout_seconds"], stage="proposer",
    )
    proposal = extract_json_object(content)
    directive = str(proposal["directive"]).strip()
    if (not directive or len(directive) > 1800 or
            any(directive in text for text in policy["skills"].values()) or
            re.search(r"FloorPlan\d|trial_T\d|set-edit-confirmation:|game\.tw-pddl", directive)):
        raise ValueError("Proposed directive violates novelty/identity contract")
    minimum = proposal.get("minimum_count")
    if type(minimum) is not int or not 1 <= minimum <= 3:
        raise ValueError("Trigger count outside frozen proposal contract")
    patch = {
        "schema_version": "phiagent-counterfactual-repair-patch/v1",
        "patch_id": "alfworld-discovery-new-recovery-v1",
        "parent_harness_sha256": parent_record["parent_harness_sha256"],
        "component": "recovery_memory", "operation": "ADD", "directive": directive,
        "trigger": {"kind": proposal["trigger_kind"], "minimum_count": minimum},
        "memory_characters": budget["memory_characters"], "max_entries": budget["max_entries"],
    }
    parse_patch(patch)
    write_json(output / "patch.json", patch)
    write_json(output / "selection.json", {
        "at": now(), "candidate_count": 1, "usage": usage, "calls": 1,
        "parent": parent, "rationale": proposal.get("rationale"),
        "discovery_failed_tasks": len(failures), "evaluation_opened": False,
        "status": "FROZEN_NEW_DIAGNOSTIC_HYPOTHESIS_NOT_VALIDATED",
    })
    print(json.dumps({"status": "NEW_PATCH_FROZEN", "parent": parent, "usage": usage}))
    return 0


def summarize(args: argparse.Namespace) -> int:
    study = args.study_dir.resolve()
    manifest = verify_manifest(study)
    binding = read_json(study / "binding.json")
    phases = {}
    for phase in ("discovery_baselines", "discovery_patches", "evaluation"):
        if (study / phase / "result.json").is_file():
            phases[phase] = read_json(study / phase / "result.json")
    parent = read_json(study / "parent_selection.json") if (study / "parent_selection.json").is_file() else None
    compiler = read_json(study / "compiler.json") if (study / "compiler.json").is_file() else None
    aliases = {}
    if parent:
        aliases["strong_parent"] = parent["selected_parent"]
    if compiler:
        aliases["compiled_choice"] = {
            "KEEP_PARENT": parent["selected_parent"], "STATIC_PATCH": "static_patch",
            "CONDITIONAL_PATCH": "conditional_patch",
        }[compiler["mode"]]
    summary = {
        "schema_version": SCHEMA, "created_at": now(), "benchmark": "ALFWorld",
        "exposure": EXPOSURE, "scientific_status": "UNVALIDATED_DEVELOPMENT_PILOT",
        "metric": manifest["metric"], "task_inventory": manifest["task_inventory"],
        "split": manifest["split"], "model": binding["model"], "phases": phases,
        "method_identity": {
            arm: {"label": record["label"], "policy_sha256": record["harness_sha256"],
                  "bound_harness_sha256": binding["harness_hashes"][arm]}
            for arm, record in manifest["baselines"].items()
        },
        "aliases": aliases, "alias_semantics": "Same frozen policy and same cells, not additional runs",
        "compiler": compiler, "parent_selection": parent,
        "identity_boundary": manifest["identity_boundary"],
        "artifacts": {"study_dir": str(study), "manifest_sha256": manifest["manifest_sha256"],
                      "binding_sha256": binding["binding_sha256"]},
        "proposer": read_json(study / "proposal/selection.json") if (study / "proposal/selection.json").is_file() else None,
        "statistics_unit": "Paired complete task; cluster/resample at group_id (floorplan)",
        "same_prefix_causal_branching": False, "official_valid_unseen_calls": 0,
    }
    write_json(study / "SUMMARY.json", summary)
    print(json.dumps({"status": "SUMMARY_SAVED", "phases": list(phases), "aliases": aliases}))
    return 0


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("freeze")
    prep.add_argument("--source-experiment", type=Path, required=True)
    prep.add_argument("--skillopt-experiment", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=20260910)
    binding = commands.add_parser("bind")
    binding.add_argument("--study-dir", type=Path, required=True)
    binding.add_argument("--skillopt-root", type=Path, required=True)
    binding.add_argument("--data-root", type=Path, required=True)
    binding.add_argument("--base-url", action="append", required=True)
    binding.add_argument("--model-id", required=True)
    binding.add_argument("--model-repository", required=True)
    binding.add_argument("--model-revision", required=True)
    binding.add_argument("--gpu-manifest", type=Path, required=True)
    binding.add_argument("--visible-devices", required=True)
    binding.add_argument("--authorized-hostname", required=True)
    check = commands.add_parser("runtime-check")
    check.add_argument("--study-dir", type=Path, required=True)
    check.add_argument("--skillopt-root", type=Path, required=True)
    check.add_argument("--data-root", type=Path, required=True)
    check.add_argument("--authorized-hostname", required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--study-dir", type=Path, required=True)
    execution.add_argument("--phase", choices=("discovery_baselines", "discovery_patches", "evaluation"), required=True)
    execution.add_argument("--workers", type=int, default=12)
    complete = commands.add_parser("campaign")
    complete.add_argument("--study-dir", type=Path, required=True)
    complete.add_argument("--workers", type=int, default=12)
    for name in ("propose", "summarize"):
        child = commands.add_parser(name)
        child.add_argument("--study-dir", type=Path, required=True)
    return cli


def main() -> int:
    args = parser().parse_args()
    return {"freeze": freeze, "runtime-check": runtime_check, "bind": bind, "run": run,
            "campaign": campaign, "propose": propose,
            "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
