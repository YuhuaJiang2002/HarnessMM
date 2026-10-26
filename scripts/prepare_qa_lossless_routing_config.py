#!/usr/bin/env python3
"""Build a new frozen-Original diagnostic; never inherit stale resource/profile bindings.

First derive the canonical profile binding from the parent's audited config and
five source prompt files (no benchmark data). On the parent's source host::

    python -m scripts.prepare_qa_lossless_routing_config bind-profiles \\
      --parent-config /opt/phiagent/staging/qa-lossless-source-h2002-20260918-r2/contents/parent/parent-config.json \\
      --parent-config-sha256 63d68261bcc4c8a63c8e3071e1cac4a3f15e5c9a402bd29ba9129d185aa4aa4c \\
      --prompt-dir /opt/phiagent/staging/qa-lossless-source-h2002-20260918-r2/contents/source/prompts \\
      --output NEW_CANONICAL_PROFILE_BINDING.json

Build locally from source metadata only::

    python -m scripts.prepare_qa_lossless_routing_config build \\
      --base outputs/qa-lossless-token/20260918-v1/PARENT_CONFIG.json \\
      --base-sha256 63d68261bcc4c8a63c8e3071e1cac4a3f15e5c9a402bd29ba9129d185aa4aa4c \\
      --canonical-profile-binding configs/canonical-qa-profiles.json \\
      --resource-binding configs/fresh-h200-resource.json \\
      --campaign-id compact-harness-refinement/UNIQUE-routing-diagnostic \\
      --selection-salt qa-lossless-route-controls-20260918-v1 \\
      --groups 128 --maximum-tasks 160 --max-output-tokens 512 --seed 20260918 \\
      --backbone second_backbone_transfer \\
      --output configs/UNIQUE-routing-diagnostic.json

The profile binding is an object with schema_version
``phiagent-qa-canonical-profile-binding/v1``, paper_release_sha256, prompt_sha256
(no_skill, skillopt, autosaddler, delete, modify), and source_manifest containing
source_path (or null if prestaged), relative_path
``canonical-profile-binding/source-manifest.json``, sha256, and hash_field
(``prompt_sha256`` for parent-config.json, or ``source_sha256``/``files_sha256``
for a source manifest). It must actually bind all five prompts. Parent supplies
the audited parent-config hash; bind-profiles verifies every actual prompt byte.
The parent verified that this current configuration's SkillOpt hash is identical
to the capability configuration's b2228... hash; no mismatch is assumed. The
builder also adds the parent-verified capability-v2 and core-v3 TRAIN exclusions,
without changing either frozen historical configuration.
The current parent uses the preservation schema; its old intervention fields
are not carried into this diagnostic. An unresolved capability-v2 exclusion is
filled only with the exact selection hash separately supplied by the parent.
The capability-v2 config remains the default fallback base when --base is omitted.

The resource file is exactly the freshly audited ``resource`` object expected by
the existing native H200 lifecycle, including native_runtime/model manifest.
No native checkpoint, package, physical GPU, or prompt hash is invented here.
The command above is the explicitly separate 4B transfer. The native resource
object must name freshly audited H200-2 GPU0 and the exact 4B manifest; the
prospective placement in the parent campaign is not a live GPU lease.
Use --backbone primary_same_model only for a separately placed exact 27B run.

For metadata preparation before the live GPU audit, build with
``--allow-pending-resource`` and the exact cache-ready RESOURCE.json. It keeps
``CACHE_READY_REQUIRES_FRESH_GPU_AND_LEASE`` unchanged and cannot pass execution
dispatch or freeze-local. Optionally require its supplied file hash with
``--resource-binding-sha256``. Rebuild into a new final configuration from the
parent's fresh-audit resource object before freezing; do not edit a frozen config.

Data preparation retains the separately declared data.preparation_host from the
source metadata. It may run on H200-1 while the frozen actor runs on H200-2;
the data must then be transferred server-to-server with its manifest unchanged.

In a NEW remote private source snapshot, overlay the lifecycle OVERLAY_FILES
union the protocol OVERLAY_FILES; retain the base origin, canonical prompts,
and six exact incumbent artifacts. Use the lifecycle's freeze-local first.
Then, on the declared data-preparation host with CUDA_VISIBLE_DEVICES='', run::

    python -m scripts.prepare_qa_lossless_routing_config prepare-data \\
      --config configs/UNIQUE-routing-diagnostic.json --source-manifest EXACT \\
      --origin DATA_ORIGIN.json --output-dir NEW_PREPARATION_DIRECTORY

Stage NEW_PREPARATION_DIRECTORY/data as CAMPAIGN/data-input and use the unchanged
``scripts.run_compact_harness_refinement seal`` command with
--campaign CAMPAIGN --config configs/UNIQUE-routing-diagnostic.json.
For this first 4B campaign, launch through the parent's outer guard, NOT directly
through the inner runner::

    python -m scripts.run_qa_lossless_guarded \\
      --campaign /opt/phiagent/work/compact-harness-refinement/20260918-lossless-routing-v1 \\
      --config configs/UNIQUE-routing-diagnostic.json

Before freezing, the supplied resource object must have
lease_root=CAMPAIGN/native-locks (private inner locks),
outer_physical_guard_root=/opt/phiagent/runs/ql0918g1, and
service_root=/opt/phiagent/runs/ql0918s1. These metadata fields are preserved
unchanged. The child's 6600-second maximum leaves 600 seconds within the
parent's 7200-second cumulative guard ceiling. The outer guard owns all shared
physical locks; using shared locks again inside would self-deadlock.
The new schema is registered directly with the unchanged whole-group sampler;
the wrapper writes its data into NEW_PREPARATION_DIRECTORY/data. Older explicit
sampling-only adapter configurations remain readable without being relabeled.
Lifecycle launch enforces physical GPU selection, leases and a new service.
After completion ``collect --experiment CAMPAIGN/experiment --output NEW.json``
reconciles raw calls and already-scored outputs without opening benchmark data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from scripts.qa_lossless_routing_protocol import (
    AA, ARMS, DIRECT_SAMPLER, INCUMBENT, MAX_CHILD_ALLOCATION_SECONDS, MODELS,
    PAPER_RELEASE, PROFILE_BINDING_SCHEMA, PROFILES, RANDOM, RESULT_SCOPE,
    ROUTING_CONTROL, SCHEMA, STATISTICS, collect_run, file_sha, prepare_data,
    validate_config, validate_profile_binding,
)
from scripts.prepare_chisel_component_quality_data import canonical_hash, write_json
from scripts.run_chisel_component_quality import ProtocolError


DEFAULT_BASE = Path("configs/qa_capability_cost_v2.json")
BASE_SCHEMAS = {"phiagent-qa-capability-cost/v2", "phiagent-qa-chisel-preservation/v1"}
DEFAULT_SEED = 20260918
DEFAULT_CAP = 512
DEFAULT_BOUNDARIES = (512, 1024, 2048, 4096, 8192, 16384)
CURRENT_PARENT_CONFIG = Path("/opt/phiagent/runs/co0917s4/parent/parent-config.json")
CURRENT_PARENT_CONFIG_SHA256 = "63d68261bcc4c8a63c8e3071e1cac4a3f15e5c9a402bd29ba9129d185aa4aa4c"
CURRENT_SOURCE_PROMPTS = Path("/opt/phiagent/runs/co0917s4/source/prompts")
PARENT_VERIFIED_PRIOR_TRAINING = (
    {
        "selection_freeze_path": "/opt/phiagent/work/compact-harness-refinement/20260916-capability-cost-v2/data/selection-freeze.json",
        "sha256": "f989ef0240292ffe8a4192e27c8acc6ab332e5d95033efcd3c7bbc3aa889e8ec",
    },
    {
        "selection_freeze_path": "/opt/phiagent/work/compact-harness-refinement/20260916-core-confirmation-v3/data/selection-freeze.json",
        "sha256": "ba060f185b9346d8760d92a34ae661f79b4029a7237fae523ef2372086d7f4aa",
    },
)


def _retains_prior_binding(original: dict, current: dict) -> bool:
    expected = deepcopy(original)
    if expected.get("sha256") is None:
        for verified in PARENT_VERIFIED_PRIOR_TRAINING:
            if expected["selection_freeze_path"] == verified["selection_freeze_path"]:
                expected["sha256"] = verified["sha256"]
    return current == original or current == expected


def _add_parent_verified_exclusions(data: dict) -> list[dict]:
    exclusions = data["exclude_prior_training"]
    resolved = []
    for verified in PARENT_VERIFIED_PRIOR_TRAINING:
        existing = [
            row for row in exclusions
            if row["selection_freeze_path"] == verified["selection_freeze_path"]
        ]
        if len(existing) > 1 or existing and existing[0].get("sha256") not in {None, verified["sha256"]}:
            raise ProtocolError("A parent-verified prior TRAIN exclusion is duplicated or has a different hash")
        if existing and existing[0].get("sha256") is None:
            existing[0]["sha256"] = verified["sha256"]
            resolved.append({
                **verified, "prior_sha256": None,
                "basis": "Exact selection-freeze hash separately verified and supplied by the parent",
            })
        elif not existing:
            exclusions.append(deepcopy(verified))
    return resolved


def build_configuration(
    base: dict, *, canonical_profiles: dict, resource: dict,
    campaign_id: str, selection_salt: str, groups: int = 128,
    maximum_tasks: int | None = None, seed: int = DEFAULT_SEED,
    max_output_tokens: int = DEFAULT_CAP, backbone: str = "primary_same_model",
    context_token_boundaries: tuple[int, ...] = DEFAULT_BOUNDARIES,
    data_binding: dict | None = None,
    allow_pending_resource: bool = False,
) -> dict:
    validate_profile_binding(canonical_profiles)
    if (
        base.get("schema_version") not in BASE_SCHEMAS
        or base["incumbent"]["status"] != "BOUND"
        or base["incumbent"]["arm"] != INCUMBENT
        or backbone not in MODELS or type(groups) is not int or not 1 <= groups <= 256
        or not isinstance(campaign_id, str)
        or not campaign_id.startswith("compact-harness-refinement/")
        or campaign_id == base.get("campaign_id")
        or any(part in {"", ".", ".."} for part in campaign_id.split("/"))
        or not isinstance(selection_salt, str) or not selection_salt
        or selection_salt == base["data"]["selection_salt"]
        or type(allow_pending_resource) is not bool
    ):
        raise ProtocolError("A verified capability/current-parent base and a new prospective campaign/sampling identity are required")
    if any(canonical_profiles["prompt_sha256"][arm] != base["prompt_sha256"][arm]
           for arm in ("delete", "modify")):
        raise ProtocolError("Canonical binding cannot substitute either archived DELETE or MODIFY profile")
    data = deepcopy(base["data"] if data_binding is None else data_binding)
    if data_binding is not None:
        for name in (
            "component_data_manifest_sha256", "train_archive_sha256",
            "train_archive_publisher_md5", "official_train_task_count",
        ):
            if data[name] != base["data"][name]:
                raise ProtocolError(f"The public TRAIN source binding cannot change: {name}")
        existing = base["data"]["exclude_prior_training"]
        if any(not any(_retains_prior_binding(entry, current) for current in data["exclude_prior_training"])
               for entry in existing):
            raise ProtocolError("Parent data metadata must retain every prior exclusion from the base")
    resolved_exclusions = _add_parent_verified_exclusions(data)
    if maximum_tasks is None:
        maximum_tasks = min(511, (groups * 5 + 3) // 4)
    data.update({
        "question_groups": groups, "maximum_tasks": maximum_tasks,
        "selection_salt": selection_salt,
        "claim_boundary": RESULT_SCOPE, "mode": "disjoint_prepared_training",
        "historical_freshness_certified": False, "selection_uses_answers": False,
        "preserve_whole_question_groups": True,
        "preserve_full_original_context_and_question_in_source_artifacts": True,
    })
    config = {
        "schema_version": SCHEMA, "campaign_id": campaign_id,
        "status": (
            "PROSPECTIVE_CONFIG_CACHE_READY_NOT_EXECUTION_AUTHORIZED"
            if resource.get("binding_status") == "CACHE_READY_REQUIRES_FRESH_GPU_AND_LEASE"
            else "PROSPECTIVE_FROZEN_ORIGINAL_TRAIN_ROUTING_DIAGNOSTIC"
        ),
        "report_title": "Frozen Original CHISEL: paired quality, actual tokens and routing controls",
        "hypothesis": "Test, rather than assume, zero average quality damage and actual token saving.",
        "authorization": "Bounded TRAIN-source frozen-router diagnostic only; no method substitution or retuning.",
        "baseline_identity": (
            "No Skill, SkillOpt and AutoSaddler are exact supplied current canonical prompt profiles, "
            "not complete optimizer executions. DELETE/MODIFY and Original's head/features/threshold "
            "are unchanged archived artifacts. Every one of the eight arms makes new provider calls."
        ),
        "arms": list(ARMS), "comparison_arms": list(ARMS),
        "candidate_arms": [], "selection_eligible_arms": [],
        "primary_candidate": INCUMBENT, "primary_reference": INCUMBENT,
        "cost_reference": "autosaddler",
        "routing_control": deepcopy(ROUTING_CONTROL),
        "count_matched_random": {
            "source_arm": INCUMBENT, "control_arm": RANDOM,
            "stratification": "original_context_exact_token_count",
            "context_token_boundaries": list(context_token_boundaries), "seed": seed,
            "preserve_realized_profile_counts": True,
            "assignments_frozen_before_answers_or_outcomes": True,
            "actual_token_equality_assumed": False,
        },
        "incumbent": deepcopy(base["incumbent"]),
        "frozen_incumbent_sha256": canonical_hash(base["incumbent"]),
        "canonical_profiles": deepcopy(canonical_profiles),
        "prompt_sha256": deepcopy(canonical_profiles["prompt_sha256"]),
        "archived_routing_prompt_sha256": {
            arm: base["prompt_sha256"][arm] for arm in ("delete", "modify")
        },
        "model": {
            **deepcopy(base["model"]), "repository": MODELS[backbone][0],
            "revision": MODELS[backbone][1], "max_output_tokens": max_output_tokens,
            "request_seed": seed,
        },
        "backbone_identity": {
            "kind": backbone, "cross_backbone_scores_may_be_pooled": False,
            "current_paper_release_sha256": PAPER_RELEASE,
        },
        "budget": {
            "maximum_real_request_attempts": maximum_tasks * len(ARMS),
            "maximum_synthetic_request_attempts": 1, "maximum_total_provider_tokens": 16_000_000,
            "maximum_allocation_seconds": MAX_CHILD_ALLOCATION_SECONDS, "maximum_physical_gpus": 1,
            "maximum_inflight_requests": 16, "execution_workers": 16, "transport_retries": 0,
        },
        "data": data, "data_preparation": deepcopy(DIRECT_SAMPLER),
        "resource": deepcopy(resource), "statistics": deepcopy(STATISTICS),
        "scoring": deepcopy(base["scoring"]),
        "confirmation": {
            "status": "NOT_AUTHORIZED_BY_THIS_DIAGNOSTIC",
            "evaluation_answers_or_responses_may_be_opened": False,
        },
        "advancement": {
            "automatic_official_scale_up_permitted": False,
            "failed_point_action": "Retain negative evidence; no cutoff retuning, arm dropping or answer reuse.",
            "point_target": "Original EM/F1 >= all six non-AA controls with strictly fewer actual tokens than AutoSaddler.",
            "statistical_target": "42 simultaneous directional finite-sample bounds; diagnostics, not a no-loss claim.",
            "aa_role": f"{AA} is a fresh byte-identical request, not an independent quality reference.",
        },
        "configuration_derivation": {
            "base_config_canonical_sha256": canonical_hash(base),
            "base_schema_version": base["schema_version"],
            "profile_binding_canonical_sha256": canonical_hash(canonical_profiles),
            "resource_binding_canonical_sha256": canonical_hash(resource),
            "resource_admission_pending": resource.get("binding_status") != "BOUND_AFTER_FRESH_AUDIT",
            "profile_changes_from_base": {
                arm: {"base_sha256": base["prompt_sha256"][arm], "canonical_sha256": expected}
                for arm, expected in canonical_profiles["prompt_sha256"].items()
                if expected != base["prompt_sha256"][arm]
            },
            "historical_skillopt_sha256": base["prompt_sha256"]["skillopt"],
            "skillopt_profile_hash_status": (
                "MATCHES_CAPABILITY_BASE"
                if canonical_profiles["prompt_sha256"]["skillopt"] == base["prompt_sha256"]["skillopt"]
                else "EXPLICIT_SUPPLIED_CANONICAL_PROFILE_CHANGE"
            ),
            "historical_skillopt_may_be_used_without_canonical_binding": False,
            "additional_parent_verified_prior_training": [
                deepcopy(entry) for entry in PARENT_VERIFIED_PRIOR_TRAINING
            ],
            "resolved_parent_prior_training": resolved_exclusions,
        },
        "paper_admitted": False, "retrieval_contribution_established": False,
    }
    validate_config(config, require_resource_audit=not allow_pending_resource)
    return config


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object: {path}")
    return value


def bind_profiles(parent_config: Path, expected_sha256: str, prompt_dir: Path) -> dict:
    """Require both audited configuration provenance and matching actual prompt bytes."""
    if file_sha(parent_config) != expected_sha256:
        raise ProtocolError("The audited current parent-config hash differs")
    parent = _read(parent_config)
    declared = parent.get("prompt_sha256", {})
    if not isinstance(declared, dict) or not set(PROFILES) <= set(declared):
        raise ProtocolError("The audited current parent configuration must bind all five profiles")
    hashes = {}
    for arm in PROFILES:
        actual = file_sha(prompt_dir / f"{arm}.system.txt")
        if actual != declared[arm]:
            raise ProtocolError(f"Audited current parent profile and actual source prompt differ: {arm}")
        hashes[arm] = actual
    binding = {
        "schema_version": PROFILE_BINDING_SCHEMA,
        "paper_release_sha256": PAPER_RELEASE,
        "prompt_sha256": hashes,
        "source_manifest": {
            "source_path": str(parent_config.resolve()),
            "relative_path": "canonical-profile-binding/source-manifest.json",
            "sha256": expected_sha256, "hash_field": "prompt_sha256",
        },
    }
    validate_profile_binding(binding)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    profiles = commands.add_parser("bind-profiles", help="verify the audited current parent and actual prompt bytes")
    profiles.add_argument("--parent-config", type=Path, default=CURRENT_PARENT_CONFIG)
    profiles.add_argument("--parent-config-sha256", default=CURRENT_PARENT_CONFIG_SHA256)
    profiles.add_argument("--prompt-dir", type=Path, default=CURRENT_SOURCE_PROMPTS)
    profiles.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build", help="source/configuration metadata only; no benchmark access")
    build.add_argument("--base", type=Path, default=DEFAULT_BASE)
    build.add_argument("--base-sha256", help="mandatory when supplying a nondefault verified parent config")
    build.add_argument("--canonical-profile-binding", type=Path, required=True)
    build.add_argument("--resource-binding", type=Path, required=True)
    build.add_argument("--resource-binding-sha256", help="require an exact parent-supplied RESOURCE.json file hash")
    build.add_argument("--allow-pending-resource", action="store_true",
                       help="metadata-only cache-ready config; cannot freeze or execute before fresh audit")
    build.add_argument("--data-binding", type=Path, help="complete parent data metadata, with expanded exact exclusions")
    build.add_argument("--campaign-id", required=True)
    build.add_argument("--selection-salt", required=True)
    build.add_argument("--groups", type=int, choices=(128, 256), default=128)
    build.add_argument("--maximum-tasks", type=int)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--max-output-tokens", type=int, default=DEFAULT_CAP)
    build.add_argument("--backbone", choices=tuple(MODELS), default="primary_same_model")
    build.add_argument("--output", type=Path, required=True)
    preparing = commands.add_parser("prepare-data", help="remote-only registered schema, unchanged whole-group sampler")
    preparing.add_argument("--config", type=Path, required=True)
    preparing.add_argument("--source-manifest", type=Path, required=True)
    preparing.add_argument("--origin", type=Path, required=True)
    preparing.add_argument("--output-dir", type=Path, required=True)
    collect = commands.add_parser("collect", help="read completed scored artifacts; no evaluator/data loading")
    collect.add_argument("--experiment", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bind-profiles":
        binding = bind_profiles(args.parent_config, args.parent_config_sha256, args.prompt_dir)
        write_json(args.output, binding)
        print(json.dumps({"status": "AUDITED_PARENT_AND_ALL_FIVE_PROMPT_BYTES_BOUND",
                          "binding": str(args.output), "sha256": file_sha(args.output)}))
    elif args.command == "build":
        actual = file_sha(args.base)
        if (args.base.resolve() != DEFAULT_BASE.resolve() and args.base_sha256 is None
                or args.base_sha256 is not None and actual != args.base_sha256):
            parser.error("An exact verified --base-sha256 is required for a nondefault parent config")
        if args.resource_binding_sha256 is not None and file_sha(args.resource_binding) != args.resource_binding_sha256:
            parser.error("The exact parent-supplied RESOURCE.json hash differs")
        config = build_configuration(
            _read(args.base), canonical_profiles=_read(args.canonical_profile_binding),
            resource=_read(args.resource_binding), campaign_id=args.campaign_id,
            selection_salt=args.selection_salt, groups=args.groups, maximum_tasks=args.maximum_tasks,
            seed=args.seed, max_output_tokens=args.max_output_tokens, backbone=args.backbone,
            data_binding=_read(args.data_binding) if args.data_binding else None,
            allow_pending_resource=args.allow_pending_resource,
        )
        config["configuration_derivation"].update({
            "base_config_path": str(args.base), "base_config_sha256": actual,
            "canonical_profile_binding_file_sha256": file_sha(args.canonical_profile_binding),
            "resource_binding_file_sha256": file_sha(args.resource_binding),
        })
        write_json(args.output, config)
        print(json.dumps({"status": (
                              "PENDING_RESOURCE_CONFIG_ONLY_CANNOT_FREEZE_OR_EXECUTE"
                              if config["configuration_derivation"]["resource_admission_pending"]
                              else "CONFIGURATION_ONLY_NO_DATA_OR_MODEL_ACCESS"
                          ),
                          "config": str(args.output), "sha256": file_sha(args.output)}))
        print("SkillOpt profile hash status: "
              f"{config['configuration_derivation']['skillopt_profile_hash_status']}; "
              f"supplied canonical {config['prompt_sha256']['skillopt']}. "
              "Actual prompt bytes must still pass bind-profiles/staging checks.", file=sys.stderr)
    elif args.command == "prepare-data":
        prepare_data(args.config, args.source_manifest, args.origin, args.output_dir)
        print(json.dumps({"status": "TRAIN_SOURCE_SELECTION_READY_NOT_A_RESULT",
                          "data": str(args.output_dir / "data")}))
    else:
        write_json(args.output, collect_run(args.experiment))
        print(json.dumps({"status": "COMPLETED_CALL_ARTIFACTS_COLLECTED", "output": str(args.output)}))


if __name__ == "__main__":
    main()
