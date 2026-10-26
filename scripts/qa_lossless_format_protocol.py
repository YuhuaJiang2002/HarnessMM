"""Twelve-arm TRAIN answer-format/output-cap successor; no model or data loading on import.

REGISTRATION (parent applies after the shared eight-arm worker finishes):
1. At the top of run_compact_harness_training_pilot.mixture_helpers add:
       if config.get("schema_version") == "phiagent-qa-lossless-format/v1":
           from scripts import qa_lossless_format_protocol
           qa_lossless_format_protocol.validate_config(config)
           return qa_lossless_format_protocol
2. Add that schema to prepare_compact_harness_training_pilot.prepare's allowlist.
   The successor normally uses copy_prepared_data at seal, NOT fresh cohort selection.
3. Add that schema to verify_qa_harness_mixture.replay_matrix's dispatch allowlist.
   No new global-cap patch is needed on the inspected shared runner: preparation
   budget, request intent, and response validation already call output_cap(cell, config).
   If those hooks are absent in the final shared revision, STOP; do not clamp to 512.
   The synthetic runner-interface test below checks these three call sites.

run_compact_harness_refinement already dispatches fixed_prompt_hashes,
prompt_artifact_names, stage_incumbent, OVERLAY_FILES and copy_prepared_data.
The pilot already dispatches prepare_cells, execution_batches, attest_runtime,
diagnostic_bounds, target_gate and result_diagnostics. No shared file is edited here.

Config builder, source/config metadata only (safe locally):
    python -m scripts.qa_lossless_format_protocol --base BASE.json \
      --bindings FORMAT_BINDINGS.json --output NEW_CONFIG.json
BASE supplies exact archived prompt/incumbent/scorer/model/native/resource bindings.
BINDINGS has run_root, campaign_id, order_seed, prepared_manifest_sha256,
selection_freeze_sha256, public_sha256, question_groups_sha256, prior_attempts, and
all_prior_attempts_including_failures_declared=true (an explicit parent attestation).
Each prior_attempt supplies root, progress_path, progress_sha256, retirement_path,
retirement_sha256, and expected {calls,tokens,gpu_seconds}. Progress must account for
all real AND synthetic attempts (attempted, known_real_tokens, synthetic_requests,
synthetic_tokens, unknown_cost_requests, pending_http). Include every first-run and
failure debit. Missing/unknown accounting is a blocker, not zero. All supplied
debits are rehashed/recomputed remotely during seal, preparation and runtime preflight.

Stage the four OWN_FILES plus OVERLAY_FILES' existing, unchanged dependency closure;
include parent's final registered runner/verifier/prepare files in its source freeze.
Use a NEW campaign/source, copy the exact prepared128 cohort into data-input, then
the existing refinement freeze-local/seal/run commands with --config NEW_CONFIG.
The run inherits all existing physical GPU selection, exclusive lease, native model
attestation, service ownership, raw storage, scoring and retirement guards.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket

from phiagent.learning import qa_lossless_format_factorial as design


SCHEMA = "phiagent-qa-lossless-format/v1"
RESULT_STATUS = "COMPLETE_TRAIN_FORMAT_CAP_DIAGNOSTIC_NOT_CONFIRMATION"
DEVELOPMENT_STATUS = "FORMAT_CAP_DIAGNOSTIC_ONLY_NO_AUTOMATIC_ADVANCEMENT"
OWN_FILES = (
    "phiagent/learning/qa_lossless_format_factorial.py",
    "scripts/qa_lossless_format_protocol.py",
    "tests/test_qa_lossless_format_factorial.py",
    "tests/test_qa_lossless_format_protocol.py",
)
# Existing baseline adapter closure; never depend on the unfinished routing protocol.
OVERLAY_FILES = (*OWN_FILES, "scripts/qa_harness_mixture_protocol.py",
                 "scripts/prepare_chisel_component_quality_data.py",
                 "scripts/run_chisel_component_quality.py",
                 "scripts/probe_chisel_model_fidelity.py",
                 "scripts/run_searchqa_residual_meta_campaign.py",
                 "phiagent/evaluation/chisel_component_quality.py",
                 "phiagent/evaluation/component_ablation.py",
                 "phiagent/evaluation/polyglot_self_evolution_statistics.py",
                 "phiagent/evaluation/__init__.py",
                 "phiagent/learning/chisel_component_quality.py",
                 "phiagent/learning/residual_meta_harness.py",
                 "phiagent/learning/residual_meta_training.py",
                 "phiagent/learning/searchqa_grounded_features.py",
                 "phiagent/learning/stage_aware_adm.py",
                 "phiagent/harness/task_reasoning.py",
                 "phiagent/harness/tshirt_fold_strategy.py",
                 "phiagent/harness/tshirt_positive_reference.py",
                 "scripts/verify_qa_harness_mixture.py",
                 "scripts/run_qa_lossless_guarded.py",
                 "scripts/cvpr_execution_crossover_leases.py",
                 "scripts/prepare_qa_lossless_native.py",
                 "scripts/verify_cvpr_qwen35_cache.py",
                 "tests/test_qa_lossless_guarded.py",
                 "tests/test_qa_lossless_native.py")
LIMITS = {"calls": 4096, "tokens": 16000000, "gpu_seconds": 7200}
PROFILES = ("skillopt", "delete", "modify")
PRIMARY = design.arm_name("original_chisel", "unchanged", 512)
MODEL_REVISIONS = {
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen/Qwen3.5-4B": None,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound(path, expected):
    require(is_sha(expected) and sha(path) == expected, f"immutable artifact changed: {path}")
    return json.loads(path.read_text())


def remote(config):
    require(socket.gethostname() == config["resource"]["selected_hostname"]
            and socket.gethostname() in {"harnessmm-worker-1", "harnessmm-worker-2"}
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "benchmark artifacts require authorized remote CPU controller and explicit empty CUDA")


def fixed_prompt_hashes(config):
    return {name: config["prompt_sha256"][name] for name in PROFILES}


def prompt_artifact_names(_config):
    return list(PROFILES)


def statistical_plan(_config):
    return {"contrasts": design.contrasts(), "metrics": list(design.METRICS),
            "family_size": 90, "alpha": 0.05, "reference": PRIMARY}


def expected_totals(attempts):
    require(isinstance(attempts, list) and attempts, "first study and every failure debit are required")
    require(len({row["root"] for row in attempts}) == len(attempts), "duplicate prior-run debit")
    total = {"calls": 0, "tokens": 0, "gpu_seconds": 0.0}
    for entry in attempts:
        for name in ("progress", "retirement"):
            require(is_sha(entry[f"{name}_sha256"]) and isinstance(entry[f"{name}_path"], str),
                    "exact progress/retirement byte bindings are required")
        costs = entry["expected"]
        require(set(costs) == set(LIMITS)
                and all(type(costs[key]) is int and costs[key] >= 0 for key in ("calls", "tokens"))
                and type(costs["gpu_seconds"]) in (int, float)
                and math.isfinite(costs["gpu_seconds"]) and costs["gpu_seconds"] >= 0,
                "complete positive/known cumulative debits required")
        for key in total:
            total[key] += costs[key]
    return total


def validate_config(config):
    settings = config["format_factorial"]
    require(config["schema_version"] == SCHEMA
            and config["arms"] == config["comparison_arms"] == list(design.ARMS)
            and config["candidate_arms"] == config["selection_eligible_arms"] == []
            and config["primary_candidate"] == config["primary_reference"] == PRIMARY
            and config["cost_reference"] == PRIMARY
            and config["incumbent"]["status"] == "BOUND"
            and config["incumbent"]["arm"] == "original_chisel",
            "exact twelve-arm factorial and original head identity required")
    require(settings == {
        "bases": list(design.BASES), "conditions": list(design.CONDITIONS), "caps": list(design.CAPS),
        "answer_first_text": design.ANSWER_FIRST, "generic_header": design.GENERIC_HEADER,
        "padding_unit": design.PADDING_UNIT, "generic_neutrality_assumed": False,
        "route_once_from_original_inputs": True, "fresh_requests_for_all_cells": True,
        "reuse_AA_or_historical_outputs": False,
    }, "frozen text, full-native matching, route reuse or factorial identity changed")
    model = config["model"]
    require(model["repository"] in MODEL_REVISIONS
            and isinstance(model["revision"], str) and len(model["revision"]) == 40
            and all(c in "0123456789abcdef" for c in model["revision"])
            and (MODEL_REVISIONS[model["repository"]] is None
                 or model["revision"] == MODEL_REVISIONS[model["repository"]])
            and model["max_output_tokens"] == 2048 and model["context_length"] == 32768
            and type(model["request_seed"]) is int and model["temperature"] == 0
            and model["enable_thinking"] is False,
            "each model needs a separate exact revision/native binding, seed and cap ceiling")
    require(config["model_identity_sha256"] == design.canonical_hash(model)
            and config["native_identity_sha256"] == design.canonical_hash(config["resource"]["native_runtime"])
            and config["baseline_identity_sha256"] == design.canonical_hash({
                "incumbent": config["incumbent"], "prompts": fixed_prompt_hashes(config),
            }), "frozen model/native/base provenance changed")
    require(all(is_sha(value) for value in fixed_prompt_hashes(config).values())
            and config["scoring"]["revision"] == "b86bbcc63352786f0ca246b76191f062dce4b52b"
            and config["scoring"]["source_sha256"]
            == "02a7855398e7133952ee374582fcc90a8b6dcf6216e0759347f29487074ee992"
            and config["scoring"]["metrics"] == ["exact_match", "f1"],
            "archived prompts and unmodified official scorer must be pinned")
    require(config["data"]["question_groups"] == 128 and config["data"]["maximum_tasks"] == 160
            and config["data"]["mode"] == "same_prepared_train_format_diagnostic"
            and all(is_sha(config["data"][key]) for key in (
                "parent_manifest_sha256", "selection_freeze_sha256", "public_sha256",
                "question_groups_sha256",
            )), "reuse the identical whole prepared128 cohort, never select another subset")
    require(config["campaign_budget"]["limits"] == LIMITS
            and config["campaign_budget"]["all_prior_attempts_including_failures_declared"] is True,
            "cumulative 4096-call/7200-GPU-second/16M-token campaign limits changed")
    prior = expected_totals(config["campaign_budget"]["prior_attempts"])
    budget = config["budget"]
    require(budget["maximum_real_request_attempts"] == min(1920, LIMITS["calls"] - prior["calls"] - 1)
            and budget["maximum_total_provider_tokens"] == LIMITS["tokens"] - prior["tokens"]
            and budget["maximum_allocation_seconds"] == math.floor(LIMITS["gpu_seconds"] - prior["gpu_seconds"])
            and min(budget[key] for key in ("maximum_real_request_attempts",
                                           "maximum_total_provider_tokens", "maximum_allocation_seconds")) > 0
            and budget["maximum_physical_gpus"] == 1 and budget["transport_retries"] == 0
            and budget["maximum_synthetic_request_attempts"] == 1
            and type(budget["execution_workers"]) is int
            and 1 <= budget["execution_workers"] <= budget["maximum_inflight_requests"] <= 16,
            "remaining campaign request/token/GPU-time budgets or no-retry policy changed")
    require(config["statistics"] == statistical_plan(config)
            and type(config["dispatch_seed"]) is int and isinstance(config["campaign_id"], str)
            and bool(config["campaign_id"]) and Path(config["run_root"]).is_absolute()
            and config["resource"]["service_root"] == str(Path(config["run_root"]) / "service")
            and config["scope"] == "EXPOSED_TRAIN_DIAGNOSTIC_NOT_OFFICIAL_BENCHMARK_OR_CONFIRMATION"
            and config["paper_admitted"] is False
            and config["run_root"] not in {row["root"] for row in config["campaign_budget"]["prior_attempts"]}
            and not any(config.get(key) for key in (
                "routing_control", "mixture_router", "capability_routing", "preservation",
                "context_selection", "rule_compression", "length_matched_controls", "calibration",
            )), "separate diagnostic scope, dispatch freeze and unchanged experimental gates required")


def verify_prior_debits(config):
    remote(config)
    for entry in config["campaign_budget"]["prior_attempts"]:
        progress = read_bound(Path(entry["progress_path"]), entry["progress_sha256"])
        retirement = read_bound(Path(entry["retirement_path"]), entry["retirement_sha256"])
        require(progress["unknown_cost_requests"] == 0 and progress["pending_http"] == 0,
                "unresolved prior/failure requests cannot be charged as zero")
        fields = ("attempted", "known_real_tokens", "synthetic_requests", "synthetic_tokens")
        require(all(type(progress[key]) is int and progress[key] >= 0 for key in fields),
                "prior accounting must include real and synthetic/admission usage exactly")
        require(retirement["status"] == "RETIRED" and retirement["owned_live_processes"] == []
                and retirement["leases_released"] is True, "a prior allocation has not safely retired")
        actual = {
            "calls": progress["attempted"] + progress["synthetic_requests"],
            "tokens": progress["known_real_tokens"] + progress["synthetic_tokens"],
            "gpu_seconds": retirement["allocation_seconds"],
        }
        require(actual == entry["expected"], "prior/failure cumulative debit does not replay")
    return expected_totals(config["campaign_budget"]["prior_attempts"])


def stage_incumbent(source, config):
    validate_config(config)
    remote(config)
    verify_prior_debits(config)
    from scripts.qa_harness_mixture_protocol import stage_incumbent_artifacts

    return stage_incumbent_artifacts(source, config)


def output_cap(cell, config):
    require(cell["arm"] in design.FACTORS, "unknown factorial cell")
    cap = design.FACTORS[cell["arm"]][2]
    require(type(cell["max_output_tokens"]) is int and cell["max_output_tokens"] == cap
            and type(cell["body"]["max_tokens"]) is int and cell["body"]["max_tokens"] == cap
            and type(cell["expected_prompt_tokens"]) is int and cell["expected_prompt_tokens"] > 0
            and cell["expected_prompt_tokens"] + cap <= config["model"]["context_length"],
            "per-cell cap changed or complete untruncated input cannot fit; never clamp")
    return cap


def exact_native_preflight(cells, config):
    prior = expected_totals(config["campaign_budget"]["prior_attempts"])
    design.randomized_order(cells, config["dispatch_seed"], config["campaign_id"])
    real_tokens = sum(cell["expected_prompt_tokens"] + output_cap(cell, config) for cell in cells)
    admission_reserve = config["model"]["context_length"]
    require(len(cells) <= config["budget"]["maximum_real_request_attempts"]
            and real_tokens + admission_reserve <= config["budget"]["maximum_total_provider_tokens"],
            "complete native matrix exceeds remaining campaign budget; no tasks or caps may be dropped")
    return {
        "prior_all_attempt_debits": prior, "new_real_calls": len(cells),
        "new_admission_calls_reserved": 1, "new_worst_case_real_tokens": real_tokens,
        "new_admission_tokens_reserved": admission_reserve,
        "cumulative_calls_upper": prior["calls"] + len(cells) + 1,
        "cumulative_tokens_upper": prior["tokens"] + real_tokens + admission_reserve,
        "cumulative_gpu_seconds_upper": prior["gpu_seconds"] + config["budget"]["maximum_allocation_seconds"],
        "frozen_before_outputs": True, "discarded_tasks": 0,
    }


def prepare_cells(tokenizer, items, prompts, config, source):
    validate_config(config)
    remote(config)
    verify_prior_debits(config)
    require(items and len(items) <= 160 and len({item["id"] for item in items}) == len(items)
            and len({item["group"] for item in items}) == 128
            and design.canonical_hash(sorted({item["group"] for item in items}))
            == config["data"]["question_groups_sha256"]
            and all(set(item) == {"id", "group", "question", "context"} for item in items),
            "exact complete prepared128 group inventory required")
    require(set(prompts) == set(PROFILES)
            and all(hashlib.sha256(prompts[name].encode()).hexdigest() == expected
                    for name, expected in fixed_prompt_hashes(config).items()),
            "an archived baseline prompt changed")
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel
    from scripts.run_chisel_component_quality import build_messages, neutral_padding, prompt_length

    router = FrozenOriginalChisel(source, config)
    cells = []
    for item in sorted(items, key=lambda row: row["id"]):
        route = router.route(item)
        require(route["variant"] in {"delete", "modify"} and route["refit"] is False
                and route["answers_or_actor_responses_used"] is False,
                "the archived Original head must route once from unchanged inputs")
        for base in design.BASES:
            profile = "skillopt" if base == "skillopt" else route["variant"]
            original = prompts[profile]
            target = original + design.ANSWER_FIRST
            # Reuse the existing full native chat-template matcher, not word/tokenizer.encode counts.
            generic = neutral_padding(
                tokenizer, item, {"modify": target, "delete": original},
                {"neutral_control": {"padding_header": design.GENERIC_HEADER,
                                     "padding_unit": design.PADDING_UNIT}},
            )
            systems = {"unchanged": original, "answer_first": target, "generic_matched": generic["system"]}
            target_count = prompt_length(tokenizer, build_messages(target, item))
            require(generic["prompt_tokens"] == target_count, "generic complete native count mismatch")
            for condition in design.CONDITIONS:
                system = systems[condition]
                messages = build_messages(system, item)
                count = prompt_length(tokenizer, messages)
                require(condition == "unchanged" or count == target_count,
                        "answer-first/generic native equality changed after rendering")
                for cap in design.CAPS:
                    arm = design.arm_name(base, condition, cap)
                    cell = {
                        "cell": design.canonical_hash([
                            config["campaign_id"], config["model_identity_sha256"], item["id"], arm,
                        ]),
                        "task": item["id"], "group": item["group"], "arm": arm,
                        "expected_prompt_tokens": count, "max_output_tokens": cap,
                        "incumbent_route": deepcopy(route), "reference_arm": PRIMARY,
                        "format_receipt": {
                            "base": base, "condition": condition, "cap": cap, "profile": profile,
                            "archived_base_sha256": fixed_prompt_hashes(config)[profile],
                            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                            "input_question_sha256": hashlib.sha256(item["question"].encode()).hexdigest(),
                            "input_context_sha256": hashlib.sha256(item["context"].encode()).hexdigest(),
                            "generic_padding_units": generic["padding_units"],
                            "target_native_prompt_tokens": target_count, "generic_neutrality_assumed": False,
                            "baseline_method_relabelled_as_modified_control": False,
                        },
                        "body": {"model": config["model"]["repository"], "messages": messages,
                                 "temperature": config["model"]["temperature"], "max_tokens": cap,
                                 "seed": config["model"]["request_seed"],
                                 "chat_template_kwargs": {"enable_thinking": False}},
                    }
                    output_cap(cell, config)
                    cells.append(cell)
    cells.sort(key=lambda cell: cell["cell"])
    schedule = design.randomized_order(cells, config["dispatch_seed"], config["campaign_id"])
    return cells, {
        "status": "COMPLETE_FORMAT_CAP_FACTORIAL_FROZEN_BEFORE_OUTPUTS",
        "route_evaluations": len(items), "unique_original_tasks": len(items),
        "independent_question_groups": 128, "fresh_requests": len(cells),
        "source_and_context_transforms": 0, "router_model_calls": 0,
        "dispatch_cells_sha256": design.canonical_hash([cell["cell"] for cell in schedule]),
        "native_preflight": exact_native_preflight(cells, config),
        "generic_neutrality_assumed": False, "historical_outputs_reused": False,
        "model_identity_sha256": config["model_identity_sha256"],
    }


def execution_batches(cells, config):
    validate_config(config)
    exact_native_preflight(cells, config)
    ordered = design.randomized_order(cells, config["dispatch_seed"], config["campaign_id"])
    workers = config["budget"]["execution_workers"]
    return [ordered[offset:offset + workers] for offset in range(0, len(ordered), workers)]


def attest_runtime(binding, config):
    validate_config(config)
    prior = verify_prior_debits(config)
    require(binding["hostname"] == config["resource"]["selected_hostname"]
            and binding["physical_gpu_index"] == config["resource"]["physical_gpu_index"]
            and binding["gpu_uuid"] == config["resource"]["gpu_uuid"]
            and binding["revision"] == config["model"]["revision"]
            and binding["model_repository"] == config["model"]["repository"],
            "runtime model/GPU binding differs from the separately frozen model identity")
    remaining = (datetime.fromisoformat(binding["allocation_deadline_utc"])
                 - datetime.now(timezone.utc)).total_seconds()
    require(0 < remaining <= config["budget"]["maximum_allocation_seconds"],
            "runtime allocation exceeds remaining cumulative GPU-time budget")
    return {"status": "PRIOR_ALL_ATTEMPT_DEBITS_REPLAYED", "prior_debits": prior,
            "remaining_allocation_deadline_seconds": remaining,
            "shared_gpu_guards_required_and_unchanged": True}


def diagnostic_bounds(scores, groups, config):
    validate_config(config)
    return design.factorial_statistics(scores, groups, context_length=config["model"]["context_length"])


def target_gate(methods, config, bounds=None):
    require(set(methods) == set(design.ARMS)
            and len({row["tasks"] for row in methods.values()}) == 1,
            "complete twelve-arm results required")
    return {"status": "TRAIN_FORMAT_CAP_DIAGNOSTIC_ONLY", "primary_baseline_status": "BOUND",
            "point_target_passed": False,
            "statistical_target_passed": False, "paper_admitted": False,
            "automatic_official_scale_up_permitted": False, "production_defaults_changed": False,
            "independent_confirmation": False, "optimizer_reproduction": False}


def result_diagnostics(planned, scores, config):
    groups = {cell["task"]: cell["group"] for cell in planned}
    exact_native_preflight(planned, config)
    return {"format_factorial": diagnostic_bounds(scores, groups, config),
            "model_identity_sha256": config["model_identity_sha256"],
            "campaign_budget": config["campaign_budget"], "no_AA_repeats_imported": True}


def replay_matrix(tokenizer, public, planned, source, config):
    remote(config)
    prompts = {name: (source / f"prompts/{name}.system.txt").read_text() for name in PROFILES}
    rebuilt, summary = prepare_cells(tokenizer, public, prompts, config, source)
    require(rebuilt == planned, "full native factorial body, route, factor or order replay differs")
    return {"complete_request_and_receipt_replay": True, "summary": summary,
            "all_original_evidence_retained": True, "per_cell_output_caps_verified": True,
            "no_refit_or_outcome_selection": True}


def copy_prepared_data(config_path, source, destination):
    config = json.loads(config_path.read_text())
    remote(config)
    validate_config(config)
    verify_prior_debits(config)
    require(not destination.exists(), "a successor needs a new data directory")
    manifest = read_bound(source / "manifest.json", config["data"]["parent_manifest_sha256"])
    require(manifest["canonical_sha256"] == design.canonical_hash({
        key: value for key, value in manifest.items() if key != "canonical_sha256"
    }) and manifest["schema_version"] == "phiagent-compact-harness-training-data/v1"
        and manifest["question_group_count"] == 128 and 128 <= manifest["task_count"] <= 160,
        "only the exact complete prepared TRAIN128 source is admissible")
    require(manifest["files_sha256"]["selection-freeze.json"] == config["data"]["selection_freeze_sha256"]
            and manifest["files_sha256"]["public.jsonl"] == config["data"]["public_sha256"]
            and manifest["files_sha256"]["mrqa_official_eval.py"] == config["scoring"]["source_sha256"],
            "cohort identity, source bytes or official scorer changed")
    require(set(manifest["files_sha256"]) == {
        "public.jsonl", "answers.json", "selection-freeze.json", "mrqa_official_eval.py", "config.json",
    }, "prepared five-file source schema changed")
    for name, expected in manifest["files_sha256"].items():
        require(Path(name).name == name and not (source / name).is_symlink()
                and sha(source / name) == expected, "prepared source file changed or escapes snapshot")
    selection = json.loads((source / "selection-freeze.json").read_text())
    require(len(selection["groups"]) == len(set(selection["groups"])) == 128
            and design.canonical_hash(sorted(selection["groups"])) == config["data"]["question_groups_sha256"],
            "the prior prepared question groups must be reused without reselection")
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest["files_sha256"]:
        if name != "config.json":
            shutil.copyfile(source / name, destination / name)
    shutil.copyfile(config_path, destination / "config.json")
    (destination / "answers.json").chmod(0o600)
    successor = {key: value for key, value in manifest.items() if key != "canonical_sha256"}
    successor.update({
        "config_sha256": sha(config_path), "parent_manifest_sha256": config["data"]["parent_manifest_sha256"],
        "same_prepared_cohort_reused": True, "independent_confirmation": False,
        "historical_training_outcomes_exposed": True,
        "independence_fields_are_inherited_exclusion_receipts_not_successor_freshness": True,
        "files_sha256": {name: sha(destination / name) for name in manifest["files_sha256"]},
    })
    successor["canonical_sha256"] = design.canonical_hash(successor)
    with (destination / "manifest.json").open("x") as stream:
        json.dump(successor, stream, sort_keys=True, indent=2)


def build_configuration(base, bindings):
    require(bindings["all_prior_attempts_including_failures_declared"] is True,
            "parent must explicitly attest the complete first-run and failure debit inventory")
    config = deepcopy(base)
    for key in ("routing_control", "mixture_router", "capability_routing", "preservation",
                "context_selection", "rule_compression", "length_matched_controls",
                "calibration", "count_matched_random", "cost_source_design"):
        config.pop(key, None)
    config.update({
        "schema_version": SCHEMA, "run_root": bindings["run_root"],
        "campaign_id": bindings["campaign_id"], "dispatch_seed": bindings["order_seed"],
        "scope": "EXPOSED_TRAIN_DIAGNOSTIC_NOT_OFFICIAL_BENCHMARK_OR_CONFIRMATION",
        "arms": list(design.ARMS), "comparison_arms": list(design.ARMS),
        "candidate_arms": [], "selection_eligible_arms": [],
        "primary_candidate": PRIMARY, "primary_reference": PRIMARY, "cost_reference": PRIMARY,
        "paper_admitted": False, "parent_config_identity_sha256": design.canonical_hash(base),
        "format_factorial": {
            "bases": list(design.BASES), "conditions": list(design.CONDITIONS), "caps": list(design.CAPS),
            "answer_first_text": design.ANSWER_FIRST, "generic_header": design.GENERIC_HEADER,
            "padding_unit": design.PADDING_UNIT, "generic_neutrality_assumed": False,
            "route_once_from_original_inputs": True, "fresh_requests_for_all_cells": True,
            "reuse_AA_or_historical_outputs": False,
        },
        "campaign_budget": {"limits": dict(LIMITS), "prior_attempts": deepcopy(bindings["prior_attempts"]),
                            "all_prior_attempts_including_failures_declared": True},
    })
    config["data"].update({
        "mode": "same_prepared_train_format_diagnostic", "question_groups": 128, "maximum_tasks": 160,
        "parent_manifest_sha256": bindings["prepared_manifest_sha256"],
        **{key: bindings[key] for key in (
            "selection_freeze_sha256", "public_sha256", "question_groups_sha256",
        )},
    })
    config["model"]["max_output_tokens"] = max(design.CAPS)
    config["resource"]["service_root"] = str(Path(bindings["run_root"]) / "service")
    config["report_title"] = "TRAIN Answer-Format by Output-Cap Diagnostic"
    config["status"] = "PROSPECTIVE_FORMAT_CAP_FACTORIAL_NOT_CONFIRMATION"
    config["model_identity_sha256"] = design.canonical_hash(config["model"])
    config["native_identity_sha256"] = design.canonical_hash(config["resource"]["native_runtime"])
    config["baseline_identity_sha256"] = design.canonical_hash({
        "incumbent": config["incumbent"], "prompts": fixed_prompt_hashes(config),
    })
    prior = expected_totals(bindings["prior_attempts"])
    config["budget"].update({
        "maximum_real_request_attempts": min(1920, LIMITS["calls"] - prior["calls"] - 1),
        "maximum_total_provider_tokens": LIMITS["tokens"] - prior["tokens"],
        "maximum_allocation_seconds": math.floor(LIMITS["gpu_seconds"] - prior["gpu_seconds"]),
        "maximum_physical_gpus": 1, "maximum_synthetic_request_attempts": 1, "transport_retries": 0,
    })
    config["statistics"] = statistical_plan(config)
    validate_config(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = build_configuration(json.loads(args.base.read_text()), json.loads(args.bindings.read_text()))
    with args.output.open("x") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
