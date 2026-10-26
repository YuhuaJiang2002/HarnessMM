#!/usr/bin/env python3
"""Prepare AA configuration locally, and opened diagnostic inputs remotely only."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from scripts.prepare_chisel_component_quality_data import canonical_hash, write_json
from scripts.qa_aa_calibration_protocol import (
    ARMS, INCUMBENT, PHASES, SCHEMA, select_public_items, sha, validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = "/opt/phiagent/work/compact-harness-refinement/20260916-chisel-sourcefield-v2/data"
PARENT_MANIFEST_SHA256 = "b98e4f8a08fd7edb98b282f992f1957b0888b9bd01c1e893e24c214e7ad6cadf"
CONFIG_NAME = "configs/qa_aa_calibration_v1.json"


def build_config(base: dict) -> dict:
    if base["schema_version"] != "phiagent-qa-chisel-preservation/v1":
        raise ValueError("AA must retain the verified Original/model lineage")
    config = copy.deepcopy(base)
    for key in (
        "capability_routing", "preservation", "zero_provider_cost_arms", "factorial",
        "routing_boundary", "selection_history", "parent_negative_boundary",
    ):
        config.pop(key, None)
    config.update({
        "schema_version": SCHEMA,
        "campaign_id": "compact-harness-refinement/20260917-aa-calibration-v1",
        "authorization": "2026-09-17 user authorized small Original-only A/A calibration. No compression, new candidate optimization, expanded benchmark or automatic follow-up runtime experiment.",
        "baseline_identity": "Every phase uses exact archived Original CHISEL head, DELETE/MODIFY prompts, complete context and identical model request bytes. Phase labels describe client scheduling, not different methods.",
        "status": "PROSPECTIVE_BOUNDED_AA_MEASUREMENT_CALIBRATION",
        "report_title": "Original CHISEL A/A: repeated serial and concurrent requests",
        "hypothesis": "Identical Original requests may produce differing outputs under repeated client schedules; this diagnostic measures variation without claiming a specific runtime cause.",
        "arms": list(ARMS), "comparison_arms": list(ARMS),
        "candidate_arms": [], "selection_eligible_arms": [],
        "primary_candidate": INCUMBENT, "primary_reference": INCUMBENT, "cost_reference": INCUMBENT,
        "calibration": {
            "phases": [
                {"arm": arm, "mode": mode, "client_concurrency": concurrency, "order": order}
                for arm, mode, concurrency, order in PHASES
            ],
            "order_salt": "20260917-aa-client-order-v1",
            "same_body_for_all_repeats": True, "historical_outputs_reused": False,
            "runtime_batch_invariant": "0",
            "server_microbatch_layout_controlled": False,
            "automatic_candidate_experiment": False,
            "fixed_order_means": "Same ordered groups of up to16client submissions with a barrier; actual server microbatches are not controlled.",
            "cache_and_time_boundary": "One fresh service, unchanged production runtime. Cache state and elapsed time can differ, so schedule association is not an isolated GPU-kernel causal result.",
            "raw_comparison": "Assistant content bytes only, excluding provider IDs/timestamps; extracted answer, official EM/F1, finish reason and usage are separate endpoints.",
            "no_early_stop_on_observed_disagreement": True,
        },
        "statistics": {
            "primary_endpoint": "question_group_with_any_extracted_answer_change_across_six_requests",
            "familywise_alpha": 0.05, "simultaneous_bounds": 1,
            "statistical_unit": "question_group_not_repeat_or_request",
            "all_fifteen_pairwise_comparisons": "descriptive_only",
            "no_optional_stopping": True,
            "zero_observed_changes_is_not_population_determinism": True,
        },
        "advancement": {
            "automatic_official_scale_up_permitted": False,
            "no_production_default_change": True,
            "point_target": "Diagnostic agreement counts only, not a method-quality point gate.",
            "statistical_target": "Report sensitivity using unique question groups; do not treat repeated requests as independent data.",
            "primary_blocked_action": "Any changed original head, prompt or model binding blocks the run.",
            "failed_point_action": "Report all variation and failures; never restart requests or choose a favorable repeat.",
        },
    })
    config["data"] = {
        "mode": "opened_development_aa_calibration",
        "parent_data": PARENT, "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "component_data_manifest_sha256": base["data"]["component_data_manifest_sha256"],
        "parent_tasks": 512, "parent_question_groups": 512,
        "question_groups": 128, "maximum_tasks": 160,
        "selection_salt": "20260917-aa128-opened-group-selection-v1",
        "historical_outcomes_exposed": True, "selection_uses_outcomes": False,
        "claim_boundary": "Label-blind subset of already opened official-training-source development groups. Not independent quality confirmation, full public evaluation, or a global determinism guarantee.",
        "exclude_prior_training": [],
    }
    config["budget"].update({
        "maximum_real_request_attempts": 960, "maximum_total_provider_tokens": 4000000,
        "maximum_allocation_seconds": 2700, "execution_workers": 16,
        "maximum_inflight_requests": 16, "maximum_synthetic_request_attempts": 1,
        "maximum_physical_gpus": 1, "transport_retries": 0,
    })
    config["model"]["request_seed"] = 20260918
    config["resource"].update({
        "service_root": "/opt/phiagent/runs/qaa0917v1", "port": 19676,
        "binding_status": "REQUIRES_FRESH_AUDIT",
        "selection_reason": "Fresh H200 physical GPU and dual exclusive leases are required; no existing service reuse.",
    })
    config["resource"].pop("pre_freeze_observation", None)
    validate_config(config)
    return config


def bind_resource(config: dict, observation: dict) -> None:
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(observation["at_utc"])).total_seconds()
    resource = config["resource"]
    if (
        not 0 <= age <= 180 or observation["hostname"] != resource["selected_hostname"]
        or observation["physical_gpu_index"] != resource["physical_gpu_index"]
        or observation["gpu_uuid"] != resource["gpu_uuid"]
        or observation["memory_used_mib"] >= 1024 or observation["utilization_percent"] > 5
        or "H200" not in observation["hardware_name"] or observation["compute_pids"]
        or observation["port"] != resource["port"] or observation["port_in_use"] is not False
        or observation["new_campaign_absent"] is not True or observation["new_service_absent"] is not True
    ):
        raise ValueError("A fresh idle physical H200 and unused campaign/service/port are required")
    resource.update({
        "binding_status": "BOUND_AFTER_FRESH_AUDIT", "pre_freeze_observation": observation,
        "selection_reason": "Fresh idle H200 observation bound before source freeze; existing controller will recheck under both physical GPU leases.",
    })


def prepare_data(config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_config(config)
    if socket.gethostname() != config["resource"]["selected_hostname"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("AA benchmark input preparation is remote-only on the authorized CPU client")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "EXECUTION.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "command": sys.argv, "python": sys.version, "CUDA_VISIBLE_DEVICES": "",
        "physical_gpu_selected": None, "model_calls": 0, "evaluator_calls": 0,
        "seed": config["data"]["selection_salt"],
        "packages": {entry.metadata["Name"]: entry.version for entry in importlib.metadata.distributions()
                     if entry.metadata["Name"]},
    })
    (output / "gpu-observation.csv").write_text(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv",
    ], text=True, timeout=30))
    origin = json.loads((ROOT / "LOCAL_FREEZE.json").read_text())
    for relative, expected in origin["files_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Unsealed AA source overlay changed: {relative}")
    parent = Path(config["data"]["parent_data"])
    if sha(parent / "manifest.json") != config["data"]["parent_manifest_sha256"]:
        raise RuntimeError("The opened parent data manifest differs")
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if parent_manifest["canonical_sha256"] != canonical_hash({
        key: value for key, value in parent_manifest.items() if key != "canonical_sha256"
    }):
        raise RuntimeError("Parent manifest self-hash changed")
    for name in ("public.jsonl", "answers.json", "mrqa_official_eval.py"):
        if sha(parent / name) != parent_manifest["files_sha256"][name]:
            raise RuntimeError(f"Parent source artifact changed: {name}")
    public = [json.loads(line) for line in (parent / "public.jsonl").read_text().splitlines()]
    if len(public) != config["data"]["parent_tasks"] or len({item["group"] for item in public}) != config["data"]["parent_question_groups"]:
        raise RuntimeError("The opened parent pool inventory differs")
    selected = select_public_items(
        public, count=config["data"]["question_groups"], salt=config["data"]["selection_salt"],
    )
    if len(selected) > config["data"]["maximum_tasks"]:
        raise RuntimeError("Whole selected groups exceed the task budget; no truncation is permitted")
    selection = {
        "groups": sorted({item["group"] for item in selected}),
        "task_ids": sorted(item["id"] for item in selected),
        "parent_manifest_sha256": config["data"]["parent_manifest_sha256"],
        "selection_salt": config["data"]["selection_salt"],
        "accepted_answers_consulted_for_selection": False,
        "prior_outcome_labels_consulted_for_selection": False,
        "historical_outcomes_exposed": True, "independent_quality_confirmation": False,
        "model_or_scorer_calls_before_freeze": 0,
    }
    write_json(output / "selection-freeze.json", selection)
    with (output / "public.jsonl").open("x", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    answers = json.loads((parent / "answers.json").read_text())
    chosen_answers = {item["id"]: answers[item["id"]] for item in selected}
    if any(not values or not isinstance(values, list) or any(not isinstance(value, str) for value in values)
           for values in chosen_answers.values()):
        raise RuntimeError("Selected official training answers are incomplete")
    write_json(output / "answers.json", chosen_answers)
    (output / "answers.json").chmod(0o600)
    shutil.copyfile(parent / "mrqa_official_eval.py", output / "mrqa_official_eval.py")
    if sha(output / "mrqa_official_eval.py") != config["scoring"]["source_sha256"]:
        raise RuntimeError("The official scorer revision differs")
    shutil.copyfile(config_path, output / "config.json")
    manifest = {
        "schema_version": "phiagent-opened-aa-data/v1",
        "status": "OPENED_AA_INPUTS_FROZEN_NO_NEW_OUTCOMES",
        "config_sha256": sha(config_path),
        "parent_manifest_sha256": config["data"]["parent_manifest_sha256"],
        "source_manifest_sha256": config["data"]["component_data_manifest_sha256"],
        "task_count": len(selected), "question_group_count": len(selection["groups"]),
        "historical_outcomes_exposed": True, "independent_quality_confirmation": False,
        "selection_uses_outcomes": False,
        "answers_packaged_only_after_selection_freeze": True,
        "new_model_calls": 0, "evaluator_calls": 0,
        "files_sha256": {
            name: sha(output / name) for name in (
                "public.jsonl", "answers.json", "selection-freeze.json", "mrqa_official_eval.py", "config.json",
            )
        },
    }
    write_json(output / "manifest.json", {**manifest, "canonical_sha256": canonical_hash(manifest)})
    print(json.dumps({"status": manifest["status"], "tasks": len(selected),
                      "groups": len(selection["groups"]), "new_model_calls": 0}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    configuration = sub.add_parser("config")
    configuration.add_argument("--output", type=Path, required=True)
    configuration.add_argument("--gpu-observation", type=Path, required=True)
    data = sub.add_parser("data")
    data.add_argument("--config", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "config":
        config = build_config(json.loads((ROOT / "configs/qa_chisel_sourcefield_v2.json").read_text()))
        bind_resource(config, json.loads(args.gpu_observation.read_text()))
        write_json(args.output, config)
        print(json.dumps({"status": "AA_CONFIG_FROZEN_NO_BENCHMARK_READS", "path": str(args.output)}))
    else:
        try:
            prepare_data(args.config, args.output)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            if args.output.is_dir() and not (args.output / "FAILURE.json").exists():
                write_json(args.output / "FAILURE.json", {
                    "type": type(error).__name__, "message": str(error),
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                })
            raise


if __name__ == "__main__":
    main()
