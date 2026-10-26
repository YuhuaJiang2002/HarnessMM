#!/usr/bin/env python3
"""Freeze a full89 trial plan and actor variants without executing a benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_terminal_bench_qwen35_4b_direct import BASE_SYSTEM_PROMPT


ARM_SPECS = (
    ("no_skill", "no_skill", False, False),
    ("skillopt", "skillopt", False, False),
    ("autosaddler", "autosaddler", False, False),
    ("chisel", "chisel", False, False),
    ("autosaddler_chisel", "autosaddler_chisel", False, False),
    ("autosaddler_chisel_pack", "autosaddler_chisel", True, False),
    ("autosaddler_chisel_fusion", "autosaddler_chisel", False, True),
    ("autosaddler_chisel_solpi", "autosaddler_chisel", True, True),
)


def sha256_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != "phiagent-terminal-chisel-solpi-full89/v1":
        raise ValueError("unknown Terminal-Bench SoL-Pi protocol")
    benchmark = protocol["benchmark"]
    expected = {
        "dataset": "terminal-bench@2.0", "revision": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
        "task_count": 89, "evaluation_attempts_per_task": 5,
        "evaluation_expected_trials_per_agent": 445, "timeout_multiplier": 1.0, "max_retries": 0,
    }
    if any(benchmark.get(key) != value for key, value in expected.items()):
        raise ValueError("the full89x5 official protocol must remain unchanged")
    if any(benchmark.get(key) is not None for key in (
        "agent_timeout_overrides", "verifier_timeout_overrides", "resource_overrides",
        "turn_or_token_caps", "task_filters",
    )):
        raise ValueError("task filtering, timeout/resource overrides and additional benchmark-driver caps are forbidden")
    if (protocol["model"].get("repository") != "Qwen/Qwen3.8-27B"
            or protocol["model"].get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"):
        raise ValueError("the frozen Qwen model identity must remain unchanged")
    if protocol["common_runtime"]["context_overflow_policy"] != {"max_retries": 0}:
        raise ValueError("the matched runtime must disable legacy lossy overflow and output backoff")
    seeds = protocol["model"]["repeat_seeds"]
    if len(seeds) != 5 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 5:
        raise ValueError("five distinct frozen repeat seeds are required")
    observed = [(arm["id"], arm["parent"], arm["observation_pack"], arm["action_fusion"])
                for arm in protocol["arms"]]
    if observed != list(ARM_SPECS) or any(type(flag) is not bool for arm in observed for flag in arm[2:]):
        raise ValueError("the complete baseline and 2x2 ablation matrix is required")
    if protocol["maximum_evaluation_trials"] != 3560:
        raise ValueError("all eight full89x5 arms must be budgeted")
    if protocol["statistics"]["independent_task_count"] != 89 or protocol["statistics"]["repeats_are_independent_tasks"]:
        raise ValueError("the independent statistical unit is the task, not each repeat")
    if protocol["execution"]["launch_ready"] or not protocol["execution"]["remote_only"]:
        raise ValueError("preparation is remote-only and cannot attest native launch readiness")


def build_trial_plan(task_ids: Sequence[str], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    if any(not isinstance(task, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", task) for task in task_ids):
        raise ValueError("task IDs must be safe single path components")
    if len(task_ids) != 89 or len(set(task_ids)) != 89:
        raise ValueError("exactly 89 unique task IDs are required")
    plan = []
    arms = [arm["id"] for arm in protocol["arms"]]
    for repeat, seed in enumerate(protocol["model"]["repeat_seeds"], start=1):
        for task_index, task in enumerate(sorted(task_ids)):
            offset = (task_index + repeat - 1) % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                plan.append({
                    "arm": arm, "task_id": task, "repeat": repeat, "repeat_seed": seed,
                    "trial_id": f"{arm}/{task}/repeat-{repeat:02d}-seed-{seed}",
                    "status": "NOT_STARTED", "reward": None, "input_tokens": None, "output_tokens": None,
                })
    return plan


def validate_parent_configs(configs: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]) -> None:
    if set(configs) != {"no_skill", "skillopt", "autosaddler", "chisel"}:
        raise ValueError("four frozen parent configurations are required")
    expected_model = configs["no_skill"]["model"]
    for name, config in configs.items():
        model = config["model"]
        if any(model.get(key) != protocol["model"][key] for key in ("repository", "revision", "temperature")):
            raise ValueError(f"parent model identity differs from protocol: {name}")
        if model != expected_model:
            raise ValueError(f"parent model transport or decoding differs across arms: {name}")
        if config.get("protocol_file_sha256") != protocol["parents"]["historical_protocol_sha256"]:
            raise ValueError(f"parent is not bound to the frozen historical protocol: {name}")
        if config.get("official_source_revision") != protocol["benchmark"]["revision"]:
            raise ValueError(f"parent benchmark revision differs: {name}")


def compose_actor_variants(
    parents: Mapping[str, Mapping[str, Any]], *, base_system_prompt: str = BASE_SYSTEM_PROMPT,
    common_context_overflow_policy: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if set(parents) != {"no_skill", "skillopt", "autosaddler", "chisel"}:
        raise ValueError("four explicit frozen parent actors are required")
    actors = {name: deepcopy(dict(config)) for name, config in parents.items()}
    if actors["no_skill"].get("system_prompt") != base_system_prompt:
        raise ValueError("No Skill must match the frozen base system prompt")
    autosaddler = actors["autosaddler"]
    chisel = actors["chisel"]
    if any(actor.get("solpi") is not None for actor in actors.values()):
        raise ValueError("parent actors must not already contain SoL-Pi modifications")
    baseline_runtime = {key: value for key, value in actors["no_skill"].items() if key != "system_prompt"}
    auto_runtime = {key: value for key, value in autosaddler.items() if key != "system_prompt"}
    if baseline_runtime != auto_runtime:
        raise ValueError("this adapter only composes a frozen AutoSaddler strategy, not unbound runtime changes")
    if (not isinstance(chisel.get("system_prompt"), str)
            or not chisel["system_prompt"].startswith(base_system_prompt)):
        raise ValueError("CHISEL must retain the exact base prompt before its strategy suffix")
    combined = deepcopy(chisel)
    suffix = chisel["system_prompt"][len(base_system_prompt):]
    separator = "\n\n" if autosaddler["system_prompt"] != base_system_prompt and suffix else ""
    combined["system_prompt"] = autosaddler["system_prompt"] + separator + suffix
    actors["autosaddler_chisel"] = combined
    variants = {}
    for name, parent, pack, fusion in ARM_SPECS:
        variants[name] = deepcopy(actors[parent])
        variants[name]["solpi"] = {"observation_pack": pack, "action_fusion": fusion}
        if common_context_overflow_policy is not None:
            if common_context_overflow_policy != {"max_retries": 0}:
                raise ValueError("common migration must disable lossy overflow retries")
            variants[name]["context_overflow_policy"] = dict(common_context_overflow_policy)
    return variants


def preparation_blockers(variants: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers = [
        "NATIVE_HARBOR_AGENT_AND_RESOURCE_BACKEND_NOT_BOUND",
        "FRESH_OFFICIAL_IMAGE_AND_VERIFIER_READINESS_REQUIRED",
        "PROVIDER_TOOL_USAGE_AND_CAMPAIGN_BUDGET_NOT_BOUND",
        "FRESH_PHYSICAL_GPU_SELECTION_REQUIRED",
    ]
    for name, agent in variants.items():
        if agent.get("context_overflow_policy", {}).get("max_retries", 0):
            blockers.append(f"LEGACY_LOSSY_OVERFLOW_REQUIRES_MATCHED_PROTOCOL_MIGRATION:{name}")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-state", required=True, type=Path)
    for name in ("no-skill", "skillopt", "autosaddler", "chisel"):
        parser.add_argument(f"--{name}-config", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or socket.gethostname() != "harnessmm-api-worker":
        parser.error("real full89 metadata preparation is restricted to authorized a800-4")
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("source_identity_pass") or not manifest.get("registry_identity_pass"):
        raise ValueError("a verified full89 source and registry manifest is required")
    configs = {name: getattr(args, f"{name}_config") for name in ("no_skill", "skillopt", "autosaddler", "chisel")}
    parent_configs = {name: json.loads(file_path.read_text()) for name, file_path in configs.items()}
    validate_parent_configs(parent_configs, protocol)
    parents = {name: config["agent"] for name, config in parent_configs.items()}
    variants = compose_actor_variants(
        parents, common_context_overflow_policy=protocol["common_runtime"]["context_overflow_policy"],
    )
    plan = build_trial_plan(manifest["expected_task_ids"], protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "frozen"
    frozen.mkdir()
    for name, file_path in {"protocol": args.protocol, "manifest": args.manifest, "git-state": args.git_state, **configs}.items():
        shutil.copy2(file_path, frozen / f"{name}.json")
    sources = (
        "scripts/prepare_terminal_bench_solpi_campaign.py", "scripts/terminal_solpi.py",
        "scripts/run_terminal_bench_qwen35_4b_direct.py", "scripts/run_terminal_bench_autosaddler_dev.py",
        "phiagent/harness/observation_pack.py", "phiagent/learning/stage_aware_adm.py",
        "phiagent/learning/residual_meta_harness.py",
    )
    for relative in sources:
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    result = {
        "schema_version": protocol["schema_version"], "status": "PREPARED_BLOCKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "python": sys.version, "command": sys.argv, "seeds": protocol["model"]["repeat_seeds"],
        "model": protocol["model"], "common_runtime": protocol["common_runtime"],
        "independent_tasks": 89, "trials_per_arm": 445, "planned_trials": len(plan),
        "actor_calls": 0, "proposer_calls": 0, "verifier_calls": 0, "gpu_calls": 0,
        "frozen_files_sha256": {file_path.name: sha256_file(file_path) for file_path in sorted(frozen.iterdir())},
        "source_files_sha256": {relative: sha256_file(ROOT / relative) for relative in sources},
        "autosaddler_equals_no_skill": parents["autosaddler"] == parents["no_skill"],
        "combined_equals_chisel": variants["autosaddler_chisel"] == variants["chisel"],
        "blockers": preparation_blockers(variants),
    }
    for filename, value in (("actor-variants.json", variants), ("trial-plan.json", plan), ("PREPARATION.json", result)):
        with (args.output / filename).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "independent_tasks", "trials_per_arm", "planned_trials", "blockers")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
