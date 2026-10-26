#!/usr/bin/env python3
"""Build and remotely prepare a new source-bound profile-compression study."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.prepare_chisel_component_quality_data import canonical_hash
from scripts.qa_profile_compression_protocol import (
    ARMS, COMPRESSION, PINNED_MODEL, PRIMARY, PRIORITY_ARMS, PRIORITY_COMPRESSION,
    PRIORITY_PRIMARY, PRIORITY_SCHEMA, SCHEMA, statistical_plan, validate_config,
)


def build_configuration(base: dict, resource: dict, *, additional_exclusions: list[dict]) -> dict:
    config = deepcopy(base)
    for key in ("routing_control", "count_matched_random", "data_preparation",
                "compression", "factorial", "preservation", "capability_routing", "calibration"):
        config.pop(key, None)
    config.update({
        "schema_version": SCHEMA,
        "campaign_id": "compact-harness-refinement/cp0918a1",
        "status": "FROZEN_SOURCE_COMPRESSION_HYPOTHESIS_NOT_VALIDATED",
        "report_title": "DELETE/MODIFY and shared-instruction compression with zero-loss gates",
        "authorization": "2026-09-18T17:28+08:00 user explicitly requested actual profile/common-instruction compression without relaxing the no-loss threshold.",
        "arms": list(ARMS), "comparison_arms": list(ARMS),
        "candidate_arms": [], "selection_eligible_arms": [],
        "primary_candidate": PRIMARY, "primary_reference": "original_chisel",
        "cost_reference": "autosaddler", "quality_margin_pp": 0,
        "compression": deepcopy(COMPRESSION), "resource": deepcopy(resource),
        "paper_admitted": False, "retrieval_contribution_established": False,
        "retained_method": None,
        "frozen_incumbent_sha256": canonical_hash(base["incumbent"]),
        "baseline_identity": "Exact archived prompts and frozen original router; not complete optimizer reproductions. Compact profiles are new source-bound candidates, not a relabeling of h0.",
        "hypothesis": "Consolidating normalization and shared evidence/target rules can remove the observed input-cost floor; behavioral no-loss remains a separately tested hypothesis.",
        "campaign_limits": {
            "maximum_all_provider_requests": 9000, "maximum_actual_provider_tokens": 30000000,
            "maximum_cumulative_gpu_seconds": 7200,
            "deadline_utc": "2026-09-18T13:28:53+00:00",
            "previous_campaign_debits_not_reset_or_relabelled": True,
            "continuation_requires_new_frozen_config_and_complete_prior_debits": True,
        },
    })
    config["model"].update({
        "repository": PINNED_MODEL[0], "revision": PINNED_MODEL[1],
        "request_seed": 20260918, "max_output_tokens": 512,
    })
    config["backbone_identity"] = {
        "kind": "primary_same_model", "cross_backbone_scores_may_be_pooled": False,
        "current_paper_release_sha256": base["backbone_identity"]["current_paper_release_sha256"],
    }
    data = config["data"]
    prior = {entry["selection_freeze_path"]: entry for entry in data["exclude_prior_training"]}
    for entry in additional_exclusions:
        name = entry["selection_freeze_path"]
        if name in prior and prior[name] != entry:
            raise ValueError("A previously frozen exclusion binding cannot change")
        if name not in prior:
            data["exclude_prior_training"].append(deepcopy(entry))
            prior[name] = entry
    data.update({
        "question_groups": 256, "maximum_tasks": 320, "preparation_host": "h200-1",
        "selection_salt": "qa-compact-normalization-common-20260918-source-only-v1",
        "mode": "disjoint_prepared_training", "selection_uses_answers": False,
        "claim_boundary": "New TRAIN-source whole-group compression development; no full public benchmark or fresh confirmation claim.",
    })
    config["budget"].update({
        "maximum_physical_gpus": 1, "maximum_allocation_seconds": 5400,
        "maximum_real_request_attempts": 4160, "maximum_synthetic_request_attempts": 1,
        "maximum_total_provider_tokens": 20000000, "execution_workers": 16,
        "maximum_inflight_requests": 16, "transport_retries": 0,
    })
    config["statistics"] = statistical_plan(config)
    validate_config(config)
    return config


def build_priority_configuration(base: dict, resource: dict, predecessor: dict) -> dict:
    if base["schema_version"] != SCHEMA:
        raise ValueError("The source-bound first compression protocol is the required predecessor")
    config = deepcopy(base)
    config.update({
        "schema_version": PRIORITY_SCHEMA, "campaign_id": "compact-harness-refinement/cp0918a2",
        "report_title": "Evidence/type priority in source-bound compact profiles",
        "primary_candidate": PRIORITY_PRIMARY,
        "arms": list(PRIORITY_ARMS), "comparison_arms": list(PRIORITY_ARMS),
        "compression": deepcopy(PRIORITY_COMPRESSION), "parent_development": deepcopy(predecessor),
        "resource": deepcopy(resource),
        "hypothesis": "Restoring evidence/type-before-normalization priority may repair the combined-compression regressions without restoring long prose. The first failed gate is not changed.",
    })
    config["data"]["selection_salt"] = "qa-profile-priority-new-groups-20260918-v2"
    config["data"]["exclude_prior_training"].append(deepcopy(predecessor["selection_exclusion"]))
    config["budget"].update(maximum_real_request_attempts=3200, maximum_allocation_seconds=3500)
    config["statistics"] = statistical_plan(config)
    validate_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "build-priority", "prepare-data"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--origin", type=Path)
    parser.add_argument("--predecessor", type=Path)
    args = parser.parse_args()
    if args.mode in {"build", "build-priority"}:
        base, resource = json.loads(args.base.read_text()), json.loads(args.resource.read_text())
        config = (
            build_priority_configuration(base, resource, json.loads(args.predecessor.read_text()))
            if args.mode == "build-priority" else build_configuration(
                base, resource, additional_exclusions=json.loads(args.exclusions.read_text()),
            )
        )
        with args.output.open("x") as stream:
            json.dump(config, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "CONFIG_ONLY_NO_BENCHMARK_ACCESS", "path": str(args.output)}))
    else:
        from scripts.prepare_compact_harness_training_pilot import prepare
        from scripts.run_compact_harness_refinement import require_remote

        config = json.loads(args.config.read_text())
        validate_config(config)
        require_remote(config)
        prepare(args.config, Path(config["data"]["source_manifest"]), args.origin, args.output)


if __name__ == "__main__":
    main()
