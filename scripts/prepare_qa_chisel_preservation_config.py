#!/usr/bin/env python3
"""Freeze configuration/source metadata only; never read benchmark items locally."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from scripts.qa_chisel_preservation_protocol import (
    ARMS, CANDIDATES, IMPLEMENTATION_FILES, INCUMBENT, LIGHT, SCHEMA, SOURCE_FIELD, validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/qa_capability_core_v3.json"
def build_configuration(base: dict) -> dict:
    if base["schema_version"] != "phiagent-qa-capability-core/v3":
        raise ValueError("Preservation must retain the exact matched Original/core reference lineage")
    config = copy.deepcopy(base)
    config.update({
        "schema_version": SCHEMA,
        "campaign_id": "compact-harness-refinement/20260916-chisel-preservation-v1",
        "status": "PROSPECTIVE_NEW_GROUP_PRESERVATION_SCREEN",
        "authorization": "2026-09-16 user authorized strict source fields plus fresh Original fallback and conservative light prompt compression, tested separately on new groups using an authorized H200. No combined policy or full-benchmark scale-up is authorized by this screen.",
        "report_title": "Original CHISEL Preservation: Fresh Source Fields and Light Compression",
        "hypothesis": "Source-bound direct answers and conservative rule rewording may separately lower complete inference tokens while preserving Original CHISEL EM/F1.",
        "arms": list(ARMS), "comparison_arms": list(ARMS),
        "candidate_arms": list(CANDIDATES), "primary_candidate": SOURCE_FIELD,
        "selection_eligible_arms": [SOURCE_FIELD],
        "primary_reference": INCUMBENT, "cost_reference": INCUMBENT,
        "zero_provider_cost_arms": [SOURCE_FIELD],
        "preservation": {
            "mode": "isolated_both",
            "minimum_total_token_saving": {SOURCE_FIELD: 0.0, LIGHT: 0.1},
            "light_total_token_saving_target_band": [0.1, 0.15],
            "minimum_light_prompt_saving_before_generation": 0.1,
            "fresh_model_fallbacks_required": True, "combine_interventions": False,
            "preserve_original_head_context_and_output_contract": True,
            "per_task_regressions_must_be_reported": True,
            "zero_per_task_regressions_is_separate_from_aggregate_rate": True,
            "source_text_is_factual_certificate": False,
            "implementation_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in IMPLEMENTATION_FILES
            },
        },
        "parent_protocol_sha256": "447363893b4923c05294823ea83e232005c6a903915b2b1c85beb46a4eedfb29",
        "parent_result_sha256": "0de745a8fc2cf4094454b7221b9f5d5146301de4187bdbdb05987f3d26a8f06f",
        "parent_negative_boundary": "The previous real 514-task matrix loses18 net successes with Core-only. Opened source-field results used cached fallbacks and are motivation only, never fresh preservation evidence.",
        "selection_history": {
            "previous_514_task_outcomes_were_opened": True,
            "light_candidate_frozen_without_new_outcomes": True,
            "source_field_rule_reused_without_new_outcome_tuning": True,
        },
    })
    config["data"].update({
        "question_groups": 512, "maximum_tasks": 640,
        "selection_salt": "20260916-chisel-preservation512-v1",
        "claim_boundary": "New whole official-training-source groups disjoint from all listed cohorts; complete historical freshness remains unverified. Not full16980official evaluation.",
    })
    config["data"]["exclude_prior_training"].append({
        "selection_freeze_path": "/opt/phiagent/work/compact-harness-refinement/20260916-core-confirmation-v3/data/selection-freeze.json",
        "sha256": "ba060f185b9346d8760d92a34ae661f79b4029a7237fae523ef2372086d7f4aa",
        "manifest_path": "/opt/phiagent/work/compact-harness-refinement/20260916-core-confirmation-v3/data/manifest.json",
        "manifest_sha256": "59032e099d1f3ff7aa58c1c4d76d3df6d757c76510c716150fd73b37f9320e4b",
    })
    config["model"]["request_seed"] = 20260918
    config["budget"].update({
        "maximum_physical_gpus": 1, "maximum_allocation_seconds": 9000,
        "maximum_real_request_attempts": 8960,
        "maximum_total_provider_tokens": 24000000,
        "execution_workers": 16, "maximum_inflight_requests": 16,
    })
    config["statistics"].update({
        "primary_references": [INCUMBENT],
        "primary_metrics": ["exact_match", "f1", "target_token_slack"],
        "factorial_quality_metrics": [],
        "simultaneous_bounds": 6, "noninferiority_margin": 0,
        "family_accounting": "Two separately frozen candidates times EM,F1,and declared cost-target slack; six simultaneous bounds clustered by question group.",
        "token_ratio_bound": "Source fields require any strict total-token reduction; light compression requires at least10% total reduction. Cost slack is (1-minimum_saving)*Original_tokens-candidate_tokens.",
        "all_other_comparisons": "All12 prior methods are rerun and reported; superiority and paper admission remain separate from this Original-preservation engineering question.",
    })
    config["factorial"] = {
        "design": "Two isolated interventions versus the unchanged Original; no composition",
        "contrasts": {"source_fields": [SOURCE_FIELD, INCUMBENT], "light_rules": [LIGHT, INCUMBENT]},
    }
    config["resource"].update({
        "binding_status": "REQUIRES_FRESH_CAPACITY_AUDIT_BEFORE_FREEZE",
        "service_root": "/opt/phiagent/runs/qcp0916v1", "port": 19674,
        "selection_reason": "A fresh GPU7 observation and the two existing exclusive physical-GPU leases are required before launch; old observations do not reserve capacity.",
    })
    config["advancement"].update({
        "point_target": "Each separately reported candidate must preserve Original EM/F1 and meet its own full prompt+completion saving threshold. Record paired regressions, not just net successes.",
        "statistical_target": "Zero-loss simultaneous EM/F1 lower bounds and nonnegative declared cost-target slack; numerical training-source bounds do not certify historical freshness.",
        "failed_point_action": "Preserve the full negative matrix; do not retune these groups or reuse its answers as fresh confirmation. A combined or revised policy needs a separate frozen protocol.",
        "automatic_official_scale_up_permitted": False,
    })
    config["routing_boundary"] = {
        "original_trained_head_unchanged": True,
        "all_original_context_retained": True,
        "source_field_selection_inputs": ["original question", "original context"],
        "light_compression_selection_inputs": ["frozen original DELETE/MODIFY route"],
        "source_field_fallback": "A separate new model request with exact Original bytes",
        "quality_certificate": False, "no_actor_labels_or_outputs": True,
        "router_model_calls": 0,
    }
    validate_config(config)
    return config


def source_field_only(config: dict, blocker_path: Path) -> dict:
    blocker = json.loads(blocker_path.read_text())
    if (
        blocker["status"] != "BLOCKED_INPUT_TOKEN_READINESS"
        or blocker["model_calls"] != 0 or blocker["benchmark_evaluator_calls"] != 0
        or blocker["minimum_light_prompt_saving"] != 0.1
        or not 0 <= blocker["observed_light_prompt_saving"] < 0.1
    ):
        raise ValueError("Only a documented no-outcome light-budget failure permits this follow-through")
    config = copy.deepcopy(config)
    config.update({
        "campaign_id": "compact-harness-refinement/20260916-chisel-sourcefield-v2",
        "report_title": "Original CHISEL Preservation: Fresh Source Fields Only",
        "arms": [arm for arm in config["arms"] if arm != LIGHT],
        "comparison_arms": [arm for arm in config["comparison_arms"] if arm != LIGHT],
        "candidate_arms": [SOURCE_FIELD],
    })
    config["preservation"].update({
        "mode": "source_field_only_after_light_budget_block",
        "light_readiness_blocker": {
            **blocker, "evidence_sha256": hashlib.sha256(blocker_path.read_bytes()).hexdigest(),
        },
    })
    config["data"].update({
        "selection_salt": "20260916-chisel-sourcefield512-v2",
    })
    config["data"]["exclude_prior_training"].append(blocker["input_only_cohort_exclusion"])
    config["resource"].update({"service_root": "/opt/phiagent/runs/qcs0916v2", "port": 19675})
    config["statistics"].update({
        "simultaneous_bounds": 3,
        "family_accounting": "One prospective source-field candidate times EM,F1,and cost slack. The separate light candidate failed input-only readiness before any outcome.",
    })
    config["factorial"]["contrasts"].pop("light_rules")
    config["budget"]["maximum_real_request_attempts"] = 8320
    validate_config(config)
    return config


def bind_gpu(config: dict, observation: dict) -> None:
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(observation["at_utc"])).total_seconds()
    if (
        not 0 <= age <= 180
        or observation["hostname"] != config["resource"]["selected_hostname"]
        or type(observation["physical_gpu_index"]) is not int
        or not 0 <= observation["physical_gpu_index"] <= 7
        or not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", observation["gpu_uuid"])
        or "H200" not in observation["hardware_name"]
        or observation["memory_used_mib"] >= 1024 or observation["utilization_percent"] > 5
        or observation["compute_pids"] or observation["port"] != config["resource"]["port"]
        or observation["port_in_use"] is not False
        or observation["new_campaign_absent"] is not True
        or observation["new_service_absent"] is not True
    ):
        raise ValueError("Fresh authorized H200 physical-GPU/service/port readiness is required")
    config["resource"].update({
        "binding_status": "BOUND_AFTER_FRESH_AUDIT",
        "physical_gpu_index": observation["physical_gpu_index"],
        "gpu_uuid": observation["gpu_uuid"],
        "pre_freeze_observation": observation,
        "selection_reason": "Fresh GPU/port/service observation saved here; final dual leases and GPU recheck remain mandatory in the existing controller.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-observation", type=Path, required=True)
    parser.add_argument("--light-readiness-blocker", type=Path)
    args = parser.parse_args()
    config = build_configuration(json.loads(BASE.read_text()))
    if args.light_readiness_blocker is not None:
        config = source_field_only(config, args.light_readiness_blocker)
    bind_gpu(config, json.loads(args.gpu_observation.read_text()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "CONFIGURATION_FROZEN_NO_BENCHMARK_READS", "path": str(args.output)}))


if __name__ == "__main__":
    main()
