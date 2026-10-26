#!/usr/bin/env python3
"""Prepare source/configuration metadata only; no benchmark data or model execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.qa_capability_cost_protocol import (
    CORE_CONFIRMATION_SCHEMA, FACTORIAL, MIXTURE_ARMS, SCHEMA, validate_config,
)


BASE_CONFIG = Path("configs/qa_harness_mixture_v1_bound.json")
BASE_SHA256 = "64ef96c80a9927eff6b3d3c4a543536ec60ba9bae11eaf216cd308c17f8c0d68"


def build_configuration(base: dict, *, confirm_core: bool = False) -> dict:
    config = copy.deepcopy(base)
    config.update({
        "schema_version": SCHEMA,
        "campaign_id": "compact-harness-refinement/20260916-capability-cost-v2",
        "status": "PROSPECTIVE_FULL_EVIDENCE_CAPABILITY_ROUTING",
        "authorization": "2026-09-16 user requested better official quality than current Ours and reduced total tokens relative to AutoSaddler on every benchmark. This SearchQA screen is one independent entry, not an all-benchmark success claim.",
        "report_title": "Full-Evidence Capability Routing: Quality Up, AutoSaddler Tokens Down",
        "hypothesis": "Preserving all task evidence and a mandatory capability skeleton while selecting optional execution rules can improve answer quality without the repeated long system prompt cost of the original method.",
        "parent_protocol_sha256": BASE_SHA256,
        "parent_result_sha256": "aafe71b368994506d591c3e7b71d8007830d93c2ebaf3f9a176a26203a836830",
        "parent_negative_boundary": "The preceding micro-context mixture kept7.53%tokens but lost15.625ppEM and12.4498ppF1. This new study never truncates C, never uses lexical coverage as a sufficiency gate, and retains the same512output cap in every arm.",
        "arms": ["no_skill", "skillopt", "autosaddler", "delete", "original_chisel", *MIXTURE_ARMS],
        "candidate_arms": list(MIXTURE_ARMS),
        "comparison_arms": ["no_skill", "skillopt", "autosaddler", "delete", "original_chisel", *MIXTURE_ARMS],
        "selection_eligible_arms": ["capability_routed"],
        "primary_candidate": "capability_routed",
        "cost_reference": "autosaddler",
        "capability_routing": {
            "enabled": True, "optional_top_k": 2, "preserve_full_context": True,
            "preserve_output_cap": True, "random_optional_count_matched_per_task": True,
        },
        "factorial": {
            "neither": "answer_only", "core_only": "core_only",
            "optional_only": "routed_without_core", "both": "capability_routed",
            "contrasts": FACTORIAL,
        },
    })
    config.pop("mixture_router")
    config.pop("count_matched_random")
    config["data"].update({
        "question_groups": 256, "maximum_tasks": 320,
        "selection_salt": "20260916-source-preserving-capability-v2",
        "claim_boundary": "Official MRQA training-source development, whole groups disjoint from listed known cohorts; historical freshness elsewhere unverified. Not a full16980official evaluation.",
    })
    config["data"]["exclude_prior_training"].append({
        "selection_freeze_path": "/opt/phiagent/work/compact-harness-refinement/20260915-harness-mixture-v1/data/selection-freeze.json",
        "sha256": "2946d9f23b33041f273340ba28b5746cf4ed113c56a05046fad71954033599b9",
        "manifest_path": "/opt/phiagent/work/compact-harness-refinement/20260915-harness-mixture-v1/data/manifest.json",
        "manifest_sha256": "99cf688ca1ca066fabff8d434f373d7a1d7e2d5bad7ac66f57cd1cc557e056e7",
    })
    config["model"]["request_seed"] = 20260916
    config["budget"].update({
        "maximum_allocation_seconds": 7200, "maximum_real_request_attempts": 3840,
        "maximum_total_provider_tokens": 9000000,
    })
    config["resource"].update({
        "binding_status": "REQUIRES_FRESH_CAPACITY_AUDIT_BEFORE_FREEZE",
        "service_root": "/opt/phiagent/runs/qcc0916v2", "port": 19672,
        "selection_reason": "Prospective H200-1 GPU7 identity; fresh observation and dual exclusive leases are mandatory before launch. A prior free observation is not a reservation.",
    })
    config["statistics"].update({
        "primary_references": [arm for arm in config["arms"] if arm != config["primary_candidate"]],
        "simultaneous_bounds": 37,
        "family_accounting": "33 primary candidate/reference EM,F1,tokens endpoints plus8factorialEM/F1 endpoints minus4shared endpoints=37.",
        "token_ratio_bound": "No10%threshold. Cost must be strictly below AutoSaddler; the paired token-difference lower bound must be positive for a numerical cost certificate.",
    })
    config["routing_boundary"] = {
        "selection_inputs": ["original question", "original context", "frozen skill library", "seed"],
        "all_original_context_retained": True, "all_output_caps_identical": True,
        "mandatory_core": ["requested answer type", "joint evidence matching", "minimal unambiguous form", "source wording"],
        "optional_selection": "At most two question-tagged source-derived skill clauses. Core clauses remain active for primary/all/random regardless of optional routing.",
        "random_control": "Same mandatory core and exactly the same optional skill count as primary on each task; randomized clause identity, not identical measured token cost.",
        "quality_certificate": False, "no_actor_labels_or_outputs": True, "router_model_calls": 0,
    }
    config["advancement"].update({
        "point_target": "Strictly better EM and F1 than every compared method, including current OriginalCHISEL and simple answer-only, with total prompt+completion tokens strictly below matched AutoSaddler.",
        "statistical_target": "Separate simultaneous lower bounds >0 for every quality comparison and primary-minus-AutoSaddler token saving; source freshness conditions remain unestablished.",
        "failed_point_action": "Retain negative evidence and do not silently retune this cohort. Any successor needs a separate frozen protocol, complete controls and independent data.",
    })
    if confirm_core:
        config.update({
            "schema_version": CORE_CONFIRMATION_SCHEMA,
            "campaign_id": "compact-harness-refinement/20260916-core-confirmation-v3",
            "status": "PROSPECTIVE_DISJOINT_TRAIN512_CORE_ONLY_REPLICATION",
            "primary_candidate": "core_only", "selection_eligible_arms": ["core_only"],
            "report_title": "Frozen Mandatory Core: Disjoint Training-Source Replication",
            "hypothesis": "The mandatory four-clause core, not optional skill routing, may improve both official MRQA EM and F1 while reducing total inference tokens relative to AutoSaddler.",
            "selection_history": {
                "source_result_sha256": "76d9bde7f392e4eec70e157f26c17e60896c9772aa12bf02eec063cb54f4261c",
                "source_verification_sha256": "689bcf112602384b90524faab3a709256be8ef3c1e04a2cb5260571be67896a9",
                "source_scope": "Previously opened256-group official-training-source development matrix.",
                "selection": "Core-only had206/256 correct, versus205 for optional routing and201 for currentOriginalCHISEL; corecost338313 versusAuto454329. This is post-hoc selection for a new frozen replication, not retroactive success of the original primary.",
                "optional_routing_innovation_admitted": False,
            },
        })
        config["data"].update({
            "question_groups": 512, "maximum_tasks": 640,
            "selection_salt": "20260916-core-only-disjoint-replication-v3",
        })
        config["data"]["exclude_prior_training"].append({
            "selection_freeze_path": "/opt/phiagent/work/compact-harness-refinement/20260916-capability-cost-v2/data/selection-freeze.json",
            "sha256": None,
            "manifest_path": "/opt/phiagent/work/compact-harness-refinement/20260916-capability-cost-v2/data/manifest.json",
            "manifest_sha256": "4edaa46c66e7485897b91343c0fedb028dade553268d955d4e3f9c71739527ad",
        })
        config["model"]["request_seed"] = 20260917
        config["budget"].update({
            "maximum_allocation_seconds": 9000, "maximum_real_request_attempts": 7680,
            "maximum_total_provider_tokens": 18000000,
        })
        config["resource"].update({"service_root": "/opt/phiagent/runs/qcc0916v3", "port": 19673})
        config["statistics"].update({
            "primary_references": [arm for arm in config["arms"] if arm != "core_only"],
            "simultaneous_bounds": 39,
            "family_accounting": "33core-only/reference EM,F1,tokens endpoints plus8factorial endpoints minus2shared endpoints=39.",
        })
    validate_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-core", action="store_true")
    args = parser.parse_args()
    raw = BASE_CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise RuntimeError("The exact parent protocol changed")
    config = build_configuration(json.loads(raw), confirm_core=args.confirm_core)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": "CONFIGURATION_ONLY_NO_BENCHMARK_CALLS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
