#!/usr/bin/env python3
"""Remote-only independent replay of frozen mixtures, raw bytes, and MRQA scores."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import io
import json
import os
from pathlib import Path
import socket
import sys

from scripts.prepare_compact_harness_training_pilot import (
    prior_exclusion_replica_bindings, prior_training_groups, resolve_prior_training_bindings,
)
from scripts.qa_harness_mixture_protocol import (
    OVERLAY_FILES, fixed_prompt_hashes, prepare_cells, prompt_artifact_names, validate_methods,
)
from scripts.run_chisel_component_quality import (
    official_scorer, prompt_length, score_rows, strict_response, valid_usage,
)
from scripts.run_compact_harness_training_pilot import (
    canonical_hash, checked, now, read_json, require_cpu_client, scorer_runtime_mode,
    sha, signed, validate_model, verify_sources, write_json,
    mixture_helpers,
)

SELF_PATH = "scripts/verify_qa_harness_mixture.py"
AMENDMENT_SCHEMA = "phiagent-qa-mixture-verifier-amendment/v1"
EXCLUSION_RECEIPT_AMENDMENT = "capability-v2-direct-exclusion-receipt"
AMENDMENT_REASONS = {
    "source-execution-json-reserialization-binding": (
        "Preparation reserializes JSON without preserving the source file bytes. "
        "Bind source, data-copy, and execution byte hashes separately and "
        "require type-sensitive canonical JSON equality; do not change any outcomes."
    ),
    EXCLUSION_RECEIPT_AMENDMENT: (
        "The initial capability-v2 preparer used the directly hashed prior exclusion "
        "list but omitted its duplicate resolved-list annotation. Reconstruct only "
        "the already frozen, directly hashed exclusions and verify every file and "
        "selected-group disjointness. No method, scoring, selection or threshold changes."
    ),
}


def validate_config(config: dict) -> None:
    if mixture_helpers(config) is None:
        raise RuntimeError("A registered mixture protocol is required")


def output_cap(cell: dict, config: dict) -> int:
    return mixture_helpers(config).output_cap(cell, config)


def statistical_plan(config: dict) -> dict:
    return mixture_helpers(config).statistical_plan(config)


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    return mixture_helpers(config).target_gate(methods, config, bounds)


def diagnostic_bounds(scores: dict, groups: dict, config: dict) -> dict:
    return mixture_helpers(config).diagnostic_bounds(scores, groups, config)


def configuration_binding(
    source: Path, data_root: Path, origin: dict, frozen: dict,
    prepared: dict, data: dict, config: dict,
) -> dict:
    source_hash = frozen["config_sha256"]
    paths = [
        relative for relative, digest in origin["source_sha256"].items()
        if digest == source_hash
    ]
    if not paths:
        raise RuntimeError("The source origin does not bind the pre-generation configuration")
    canonical = canonical_hash(config)
    for relative in paths:
        path = source / relative
        if (not path.resolve().is_relative_to(source.resolve())
                or sha(path) != source_hash or canonical_hash(read_json(path)) != canonical):
            raise RuntimeError("The source configuration differs from the executed configuration")
    data_copy_hash = data["files_sha256"]["config.json"]
    if (data["config_sha256"] != source_hash
            or sha(data_root / "config.json") != data_copy_hash
            or canonical_hash(read_json(data_root / "config.json")) != canonical):
        raise RuntimeError("Data selection and execution used different configurations")
    return {
        "source_config_sha256": source_hash,
        "data_protocol_source_config_sha256": data["config_sha256"],
        "data_config_copy_sha256": data_copy_hash,
        "execution_config_sha256": prepared["config_sha256"],
        "canonical_config_sha256": canonical,
        "source_paths": sorted(paths),
        "source_and_execution_canonical_json_equal": True,
        "byte_reserialization_recorded_not_waived": (
            source_hash != prepared["config_sha256"] or data_copy_hash != source_hash
        ),
    }


def audit_dependency_binding(
    root: Path, audit_source: Path, origin: dict, audit_files: set[str],
    amendment: dict | None,
) -> dict:
    changed = {
        relative: {
            "generation_sha256": origin["source_sha256"][relative],
            "audit_sha256": sha(audit_source / relative),
        }
        for relative in audit_files
        if sha(audit_source / relative) != origin["source_sha256"][relative]
    }
    if not changed:
        if amendment is not None:
            raise RuntimeError("An unchanged verifier must not consume an amendment")
        return {"status": "BYTE_IDENTICAL_FROZEN_VERIFIER_AND_DEPENDENCIES"}
    if set(changed) != {SELF_PATH}:
        raise RuntimeError("A verifier amendment cannot change routing, scoring, or generation dependencies")
    if amendment is None:
        raise RuntimeError("A changed verifier requires a pinned self-only amendment")
    if (
        amendment["schema_version"] != AMENDMENT_SCHEMA
        or amendment["reason_code"] not in AMENDMENT_REASONS
        or amendment["allowed_modified_files"] != [SELF_PATH]
        or amendment["generation_verifier_sha256"] != changed[SELF_PATH]["generation_sha256"]
        or amendment["audit_verifier_sha256"] != changed[SELF_PATH]["audit_sha256"]
        or amendment["result_sha256"] != sha(root / "RESULT.json")
        or amendment["prepared_sha256"] != sha(root / "PREPARED.json")
        or amendment["source_freeze_sha256"] != sha(root.parent / "SOURCE_FROZEN.json")
    ):
        raise RuntimeError("The verifier amendment does not bind this exact completed run and code")
    return {
        "status": "PINNED_SELF_ONLY_VERIFIER_AMENDMENT",
        "changed_files": changed,
        "amendment": amendment,
        "all_routing_scoring_and_generation_dependencies_byte_identical": True,
    }


def freeze_verifier_amendment(
    root: Path, output: Path, *, reason_code: str = "source-execution-json-reserialization-binding",
    quality_results_previously_visible: bool = False,
) -> None:
    require_cpu_client()
    if reason_code not in AMENDMENT_REASONS or type(quality_results_previously_visible) is not bool:
        raise ValueError("an explicit registered audit repair and observation boundary are required")
    prepared = checked(root / "PREPARED.json")
    origin = read_json(root / "origin.json")
    source = Path(prepared["source"])
    if canonical_hash(read_json(source / "origin.json")) != canonical_hash(origin):
        raise RuntimeError("Cannot amend an audit with mismatched generation origins")
    verify_sources(source, origin)
    write_json(output, signed({
        "schema_version": AMENDMENT_SCHEMA,
        "at_utc": now(), "hostname": socket.gethostname(), "command": sys.argv,
        "reason_code": reason_code,
        "reason": AMENDMENT_REASONS[reason_code],
        "allowed_modified_files": [SELF_PATH],
        "generation_verifier_sha256": origin["source_sha256"][SELF_PATH],
        "audit_verifier_sha256": sha(Path(__file__)),
        "result_sha256": sha(root / "RESULT.json"),
        "prepared_sha256": sha(root / "PREPARED.json"),
        "source_freeze_sha256": sha(root.parent / "SOURCE_FROZEN.json"),
        "benchmark_scores_parsed_by_amendment_builder": False,
        "quality_results_previously_visible": quality_results_previously_visible,
        "model_calls": 0, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "paper_admitted": False,
    }))


def prior_exclusion_annotation(
    selection: dict, config: dict, resolved: list[dict], amendment: dict | None,
) -> dict:
    if "resolved_prior_training_exclusions" in selection:
        if canonical_hash(selection["resolved_prior_training_exclusions"]) != canonical_hash(resolved):
            raise RuntimeError("The recorded resolved prior exclusion list differs")
        return {"status": "RECORDED_RESOLVED_EXCLUSIONS_VERIFIED", "resolved": resolved}
    specifications = config["data"]["exclude_prior_training"]
    if (
        amendment is None or amendment["reason_code"] != EXCLUSION_RECEIPT_AMENDMENT
        or config["schema_version"] != "phiagent-qa-capability-cost/v2"
        or canonical_hash(specifications) != canonical_hash(resolved)
        or canonical_hash(selection["prior_training_exclusions"]) != canonical_hash(specifications)
        or any(
            not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            for entry in specifications
        )
    ):
        raise RuntimeError("Missing resolved exclusions require the pinned direct-hash v2 audit amendment")
    return {
        "status": "PINNED_DIRECT_EXCLUSION_ANNOTATION_RECONSTRUCTED",
        "resolved": resolved,
        "selection_and_data_bytes_changed": False,
        "all_exclusion_file_hashes_and_group_disjointness_checked_separately": True,
    }


def replay_matrix(
    tokenizer, public: list[dict], planned: list[dict], source: Path, config: dict,
) -> dict:
    validate_config(config)
    if config["schema_version"] in {
        "phiagent-qa-capability-cost/v2", "phiagent-qa-capability-core/v3",
        "phiagent-qa-chisel-preservation/v1",
        "phiagent-qa-lossless-routing/v1", "phiagent-qa-lossless-format/v1",
        "phiagent-qa-profile-compression/v1",
        "phiagent-qa-profile-compression/v2",
    }:
        return mixture_helpers(config).replay_matrix(tokenizer, public, planned, source, config)
    arms = prompt_artifact_names(config)
    for arm, expected in fixed_prompt_hashes(config).items():
        if sha(source / f"prompts/{arm}.system.txt") != expected:
            raise RuntimeError(f"Frozen baseline prompt changed: {arm}")
    prompts = {arm: (source / f"prompts/{arm}.system.txt").read_text(encoding="utf-8") for arm in arms}
    rebuilt, summary = prepare_cells(tokenizer, public, prompts, config, source)
    if len(rebuilt) != len(planned):
        raise RuntimeError("The planned matrix is incomplete or has extra cells")
    for frozen, reconstructed in zip(planned, rebuilt, strict=True):
        if frozen != reconstructed:
            raise RuntimeError(f"Independent public-input route/context/prompt/cap replay differs: {frozen.get('cell')}")
        output_cap(frozen, config)
    by_pair = {(cell["task"], cell["arm"]): cell for cell in rebuilt}
    expected_pairs = {(item["id"], arm) for item in public for arm in config["arms"]}
    if set(by_pair) != expected_pairs or len(by_pair) != len(rebuilt):
        raise RuntimeError("Every matched arm must contain every complete task exactly once")
    count_matches = {}
    for item in public:
        task = item["id"]
        original, random_cell = by_pair[task, "full_mixture"], by_pair[task, "random_mixture"]
        receipt = random_cell["random_assignment"]
        count = len(tokenizer.encode(item["context"], add_special_tokens=False))
        stratum = f"context_tokens_bucket_{sum(count >= boundary for boundary in config['count_matched_random']['context_token_boundaries'])}"
        if (receipt["stratum"] != stratum or receipt["source_context_tokens"] != count
                or receipt["source_harness"] != original["mixture_receipt"]["selected_harness"]
                or receipt["assigned_harness"] != random_cell["mixture_receipt"]["selected_harness"]
                or receipt["original_policy_sha256"] != original["mixture_receipt"]["policy_sha256"]
                or receipt["effective_policy_sha256"] != random_cell["mixture_receipt"]["policy_sha256"]
                or receipt["answers_or_actor_responses_used"] is not False
                or receipt["override"] != {
                    "harness_mode": "fixed", "fixed_harness_id": receipt["assigned_harness"], "skill_mode": "random",
                }):
            raise RuntimeError("Random assignment/source policy/override binding differs")
        counters = count_matches.setdefault(stratum, {"full_mixture": Counter(), "random_mixture": Counter()})
        counters["full_mixture"][receipt["source_harness"]] += 1
        counters["random_mixture"][receipt["assigned_harness"]] += 1
    if any(counts["full_mixture"] != counts["random_mixture"] for counts in count_matches.values()):
        raise RuntimeError("The random control does not preserve realized per-stratum harness counts")
    materialized = {}
    for name, (left, right) in config["factorial"]["contrasts"].items():
        pairs = [(by_pair[item["id"], left], by_pair[item["id"], right]) for item in public]
        materialized[name] = {
            "left": left, "right": right, "tasks": len(pairs),
            "different_requests": sum(lcell["body"] != rcell["body"] for lcell, rcell in pairs),
            "different_output_caps": sum(
                lcell["max_output_tokens"] != rcell["max_output_tokens"] for lcell, rcell in pairs
            ),
            "different_skill_sets": sum(
                lcell["mixture_receipt"]["selected_skills"] != rcell["mixture_receipt"]["selected_skills"]
                for lcell, rcell in pairs
            ),
            "different_contexts": sum(
                lcell["mixture_receipt"]["rendered_context_sha256"] != rcell["mixture_receipt"]["rendered_context_sha256"]
                for lcell, rcell in pairs
            ),
        }
    return {
        "summary": summary, "factorial_materialization": materialized,
        "count_matched_random_verified": {
            stratum: {arm: dict(sorted(counts.items())) for arm, counts in methods.items()}
            for stratum, methods in sorted(count_matches.items())
        },
        "all_material_receipt_fields_exact": True, "per_cell_output_caps_verified": True,
        "source_offsets_tokenizer_and_full_messages_reconstructed": True,
        "answers_or_actor_outcomes_used_for_reconstruction": False,
        "source_provenance_is_not_answer_sufficiency": True,
    }


def replay_raw_calls(root: Path, planned: list[dict], config: dict) -> tuple[dict, dict]:
    actual_dirs = {path.name for path in (root / "calls").iterdir() if path.is_dir()}
    if actual_dirs != {cell["cell"] for cell in planned}:
        raise RuntimeError("Missing or extra raw call directories")
    rows, hashes = {arm: {} for arm in config["arms"]}, {}
    helpers = mixture_helpers(config)
    for cell in planned:
        directory = root / "calls" / cell["cell"]
        if (directory / "request-failure.json").exists():
            raise RuntimeError("A failed request cannot enter a complete matched comparison")
        if cell.get("execution_kind", "model") != "model":
            if (not hasattr(helpers, "verify_direct_cell")
                    or cell["execution_kind"] != helpers.DIRECT_EXECUTION_KIND):
                raise RuntimeError("Unregistered native execution cannot bypass raw-provider verification")
            saved, native_hashes = helpers.verify_direct_cell(cell, directory, config)
            if cell["task"] in rows[cell["arm"]]:
                raise RuntimeError("Duplicate native task in an arm")
            rows[cell["arm"]][cell["task"]] = saved
            hashes.update({
                str((directory / name).relative_to(root)): digest
                for name, digest in native_hashes.items()
            })
            continue
        expected_body = json.dumps(cell["body"], ensure_ascii=False, separators=(",", ":")).encode()
        if (directory / "request.body").read_bytes() != expected_body:
            raise RuntimeError("Actual request bytes differ from the frozen cell")
        intent = read_json(directory / "intent.json")
        if (intent["cell"] != cell["cell"] or intent["task"] != cell["task"]
                or intent["arm"] != cell["arm"]
                or intent["expected_prompt_tokens"] != cell["expected_prompt_tokens"]
                or intent["max_output_tokens"] != output_cap(cell, config)
                or intent["body_sha256"] != sha(directory / "request.body")):
            raise RuntimeError("Saved intent identity/token cap differs from the frozen request")
        if read_json(directory / "transport.json")["status"] != 200:
            raise RuntimeError("A failed HTTP attempt invalidates the whole comparison")
        raw = json.loads((directory / "response.body").read_bytes())
        observed = strict_response(
            raw, model=config["model"]["repository"], expected_prompt_tokens=cell["expected_prompt_tokens"],
        )
        if observed["usage"]["completion_tokens"] > output_cap(cell, config):
            raise RuntimeError("Actual completion exceeds this cell's selected harness cap")
        saved = read_json(directory / "record.json")
        if (saved["task"] != cell["task"] or saved["arm"] != cell["arm"]
                or saved["group"] != cell["group"] or saved["raw"] != raw
                or saved["usage"] != observed["usage"]
                or saved["finish_reason"] != observed["finish_reason"]
                or saved["empty_answer"] != observed["empty_answer"]
                or saved["response_sha256"] != sha(directory / "response.body")):
            raise RuntimeError("Saved task record differs from original provider bytes")
        if cell["task"] in rows[cell["arm"]]:
            raise RuntimeError("Duplicate task in an arm")
        rows[cell["arm"]][cell["task"]] = saved
        for name in ("request.body", "response.body", "intent.json", "transport.json", "record.json"):
            hashes[str((directory / name).relative_to(root))] = sha(directory / name)
    return rows, hashes


def rescore(rows: dict, answers: dict, scorer, config: dict | None = None) -> tuple[dict, dict, dict]:
    methods, all_scores, all_predictions = {}, {}, {}
    helpers = mixture_helpers(config) if config is not None else None
    for arm, records in rows.items():
        per_task = (
            helpers.score_records(records, answers, scorer)
            if helpers is not None and hasattr(helpers, "score_records")
            else score_rows(records, answers, scorer)
        )
        predictions = {task: row["prediction"] for task, row in per_task.items()}
        metrics = scorer.evaluate(answers, predictions, skip_no_answer=False)
        methods[arm] = {
            "tasks": len(per_task), "exact_match": float(metrics["exact_match"]), "f1": float(metrics["f1"]),
            "total_tokens": sum(row["tokens"] for row in per_task.values()),
            "prompt_tokens": sum(row["prompt_tokens"] for row in per_task.values()),
            "completion_tokens": sum(row["completion_tokens"] for row in per_task.values()),
            "empty_answers": sum(row["empty_answer"] for row in per_task.values()),
            "length_stops": sum(row["finish_reason"] == "length" for row in per_task.values()),
        }
        all_scores[arm], all_predictions[arm] = per_task, predictions
    return methods, all_scores, all_predictions


def verify_accounting(root: Path, methods: dict, planned: list[dict], config: dict, tokenizer) -> dict:
    result = checked(root / "RESULT.json")
    drained = checked(root / "GENERATION_DRAINED.json")
    campaign = root.parent
    binding_path = campaign / "binding.admitted.json"
    binding = read_json(binding_path)
    if (sha(binding_path) != result["service_binding_sha256"]
            or binding["model_spec_sha256"] != canonical_hash(read_json(root / "model-spec.json"))
            or binding["hostname"] != config["resource"]["selected_hostname"]
            or binding["physical_gpu_index"] != config["resource"]["physical_gpu_index"]
            or binding["gpu_uuid"] != config["resource"]["gpu_uuid"]):
        raise RuntimeError("The actual service/model/physical-GPU binding differs")
    start = read_json(root / "RUN_STARTED.json")
    if start["service_binding"] != binding or start["source_preparation_sha256"] != sha(root / "PREPARED.json"):
        raise RuntimeError("Generation did not start from the frozen preparation and binding")
    admission = read_json(campaign / "admission-request.body")
    if (admission["model"] != config["model"]["repository"]
            or admission["seed"] != config["model"]["request_seed"]
            or admission["max_tokens"] != 32 or admission["temperature"] != 0
            or admission["chat_template_kwargs"] != {"enable_thinking": False}
            or admission["messages"] != [
                {"role": "user", "content": "Return only the number 42 inside <answer>...</answer>."}
            ]):
        raise RuntimeError("Synthetic admission was not the separate declared one-call capability probe")
    raw_admission = read_json(campaign / "admission-response.body")
    admission_observed = strict_response(
        raw_admission, model=config["model"]["repository"],
        expected_prompt_tokens=prompt_length(tokenizer, admission["messages"]),
    )
    usage = valid_usage(raw_admission)
    if (read_json(campaign / "admission-transport.json")["status"] != 200
            or usage is None or admission_observed["usage"]["completion_tokens"] > 32):
        raise RuntimeError("Synthetic admission lacks valid charged usage within its cap")
    acquisition = read_json(campaign / "ADMISSION_USAGE.json")
    if (acquisition["usage"] != usage or acquisition["requests"] != 1
            or acquisition["unknown_cost_requests"] != 0
            or binding["synthetic_request_attempts"] != 1
            or binding["synthetic_total_tokens"] != usage["total_tokens"]):
        raise RuntimeError("Synthetic cost was omitted, duplicated, or changed")
    real_tokens = sum(row["total_tokens"] for row in methods.values())
    direct = sum(cell.get("execution_kind") == "deterministic_source_field" for cell in planned)
    for counters in (result["calls"], drained):
        if (counters["attempted"] != len(planned) or counters["completed"] != len(planned)
                or counters["pending_http"] != 0 or counters["unknown_cost_requests"] != 0
                or counters["known_real_tokens"] != real_tokens
                or counters["synthetic_requests"] != 1
                or counters["synthetic_tokens"] != usage["total_tokens"]):
            raise RuntimeError("Complete real/synthetic request and additive token accounting differs")
        if config["schema_version"] == "phiagent-qa-chisel-preservation/v1" and (
            counters["counter_unit"] != "logical_cells"
            or counters["provider_requests_attempted"] != len(planned) - direct
            or counters["provider_requests_completed"] != len(planned) - direct
            or counters["deterministic_outputs"] != direct
        ):
            raise RuntimeError("Native outputs were counted as provider requests or omitted")
    if (not drained["generation_complete"] or not drained["server_completion_verified"]
            or config["budget"]["maximum_synthetic_request_attempts"] < 1
            or len(planned) > config["budget"]["maximum_real_request_attempts"]
            or real_tokens + usage["total_tokens"] > config["budget"]["maximum_total_provider_tokens"]):
        raise RuntimeError("The frozen complete-matrix request or token budget was not respected")
    retirement = read_json(campaign / "RETIREMENT.json")
    if (retirement["status"] != "RETIRED" or retirement["owned_live_processes"]
            or retirement["leases_released"] is not True
            or type(retirement["allocation_seconds"]) not in (int, float)
            or not 0 <= retirement["allocation_seconds"] <= config["budget"]["maximum_allocation_seconds"]):
        raise RuntimeError("Service retirement or total bounded GPU allocation is not verified")
    selection = read_json(campaign / "GPU_SELECTION.json")
    gpu_rows = list(csv.reader(io.StringIO(selection["outputs"][0]), skipinitialspace=True))
    gpu_processes = list(csv.reader(io.StringIO(selection["outputs"][1]), skipinitialspace=True))
    if (selection["hostname"] != config["resource"]["selected_hostname"]
            or len(gpu_rows) != 1 or len(gpu_rows[0]) != 6 or gpu_processes
            or int(gpu_rows[0][0]) != config["resource"]["physical_gpu_index"]
            or gpu_rows[0][1] != config["resource"]["gpu_uuid"]
            or config["resource"]["hardware_name"] not in gpu_rows[0][2]
            or float(gpu_rows[0][3]) >= 1024 or float(gpu_rows[0][5]) > 5):
        raise RuntimeError("Saved prelaunch physical GPU selection differs")
    expected = {
        "selected_path_tokens": {arm: row["total_tokens"] for arm, row in methods.items()},
        "real_matrix_acquisition_tokens": real_tokens,
        "synthetic_acquisition_tokens": usage["total_tokens"],
        "all_in_acquisition_tokens": real_tokens + usage["total_tokens"],
        "router_provider_tokens": 0, "selected_path_is_all_in_acquisition_cost": False,
    }
    if result["cost_accounting"] != expected:
        raise RuntimeError("Selected-path cost was confused with complete study acquisition")
    return {
        **expected, "real_requests": len(planned) - direct, "synthetic_requests": 1,
        **({"logical_cells": len(planned), "deterministic_source_outputs": direct}
           if config["schema_version"] == "phiagent-qa-chisel-preservation/v1" else {}),
        "unknown_cost_requests": 0, "pending_requests": 0,
        "reserved_completion_tokens_by_arm": {
            arm: dict(sorted(Counter(output_cap(cell, config) for cell in planned if cell["arm"] == arm).items()))
            for arm in config["arms"]
        },
        "allocation_seconds": retirement["allocation_seconds"],
        "service_binding_sha256": sha(binding_path), "gpu_selection_sha256": sha(campaign / "GPU_SELECTION.json"),
        "retirement_sha256": sha(campaign / "RETIREMENT.json"),
    }


def verify(root: Path, output: Path, *, verifier_amendment: Path | None = None) -> dict:
    require_cpu_client()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(root / "config.json")
    validate_config(config)
    write_json(output / "EXECUTION.json", {
        "at_utc": now(), "hostname": socket.gethostname(), "python": sys.version, "command": sys.argv,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "selected_physical_gpu": None,
        "purpose": "Remote CPU-only independent saved-byte official-score replay; no new model requests",
        "seed": config["model"]["request_seed"], "config_sha256": sha(root / "config.json"),
        "source_sha256": sha(Path(__file__)), "new_model_calls": 0,
        "packages": {entry.metadata["Name"]: entry.version for entry in importlib.metadata.distributions()
                     if entry.metadata["Name"]},
    })
    for name in ("FAILURE.json", "CLIENT_STOPPED.json"):
        if (root / name).exists():
            raise RuntimeError("Failure evidence exists; do not score a partial or repaired matrix")
    if (root.parent / "RUN_FAILURE.json").exists():
        raise RuntimeError("The campaign owner recorded a failed run")
    prepared = checked(root / "PREPARED.json")
    for name, key in (("config.json", "config_sha256"), ("model-spec.json", "model_spec_sha256"),
                      ("requests.json", "requests_sha256")):
        if sha(root / name) != prepared[key]:
            raise RuntimeError(f"Frozen experiment input changed: {name}")
    source, data_root = Path(prepared["source"]), Path(prepared["data"])
    origin = read_json(root / "origin.json")
    if canonical_hash(read_json(source / "origin.json")) != canonical_hash(origin):
        raise RuntimeError("The source and generation origin snapshots differ")
    verify_sources(source, origin)
    audit_source = Path(__file__).resolve().parents[1]
    audit_files = {
        *OVERLAY_FILES,
        *mixture_helpers(config).OVERLAY_FILES,
        "scripts/run_compact_harness_training_pilot.py",
        "scripts/run_chisel_component_quality.py",
        "scripts/prepare_compact_harness_training_pilot.py",
        "scripts/prepare_chisel_component_quality_data.py",
        "phiagent/learning/qa_context_evidence.py",
        "phiagent/learning/public_evidence_retention.py",
        "phiagent/learning/sparse_retrieval.py",
        "phiagent/learning/compact_qa_contracts.py",
        "phiagent/evaluation/chisel_component_quality.py",
    }
    amendment = checked(verifier_amendment) if verifier_amendment is not None else None
    dependency_binding = audit_dependency_binding(
        root, audit_source, origin, audit_files, amendment,
    )
    frozen = read_json(root.parent / "SOURCE_FROZEN.json")
    if (frozen["origin_sha256"] != sha(source / "origin.json")
            or frozen["data_manifest_sha256"] != prepared["data_manifest_sha256"]
            or frozen["models_called"] != 0 or frozen["new_confirmation_answers_opened"] is not False):
        raise RuntimeError("The pre-generation source/data freeze binding differs")
    if sha(data_root / "manifest.json") != prepared["data_manifest_sha256"]:
        raise RuntimeError("The frozen prepared data manifest changed")
    data = checked(data_root / "manifest.json")
    config_binding = configuration_binding(source, data_root, origin, frozen, prepared, data, config)
    write_json(output / "CONFIGURATION_BINDING.json", config_binding)
    write_json(output / "AUDIT_DEPENDENCY_BINDING.json", dependency_binding)
    if (data["source_manifest_sha256"] != config["data"]["component_data_manifest_sha256"]
            or data["question_group_count"] != config["data"]["question_groups"]
            or not 0 < data["task_count"] <= config["data"]["maximum_tasks"]):
        raise RuntimeError("The declared whole-group training-source cohort differs")
    for name in ("public.jsonl", "selection-freeze.json", "mrqa_official_eval.py", "config.json"):
        if sha(data_root / name) != data["files_sha256"][name]:
            raise RuntimeError(f"Frozen pre-answer input changed: {name}")
    required_independence = (
        "group_disjoint_from_component_training", "group_disjoint_from_component_calibration",
        "group_disjoint_from_full_official_development", "group_disjoint_from_prior_training",
    )
    if (any(data["independence"].get(key) is not True for key in required_independence)
            or data["independence"]["evaluation_answers_opened"]
            or data["independence"]["evaluation_responses_opened"]):
        raise RuntimeError("Prepared data lacks the declared independence evidence")
    public = [json.loads(line) for line in (data_root / "public.jsonl").read_text().splitlines()]
    groups = {item["id"]: item["group"] for item in public}
    selection = read_json(data_root / "selection-freeze.json")
    resolved = resolve_prior_training_bindings(config["data"]["exclude_prior_training"])
    cross_host = (
        config["schema_version"] in {
            "phiagent-qa-lossless-routing/v1", "phiagent-qa-lossless-format/v1",
        }
        and config["data"]["preparation_host"] != config["resource"]["selected_host_alias"]
    )
    replicas = prior_exclusion_replica_bindings(source, resolved, required=cross_host)
    locations = {entry["original_path"]: source / entry["relative_path"] for entry in replicas}
    inspected = [
        {**entry, "selection_freeze_path": str(locations[entry["selection_freeze_path"]])}
        if entry["selection_freeze_path"] in locations else entry
        for entry in resolved
    ]
    prior_groups = prior_training_groups(inspected)
    exclusion_annotation = prior_exclusion_annotation(selection, config, resolved, amendment)
    if replicas:
        exclusion_annotation = {**exclusion_annotation, "hash_identical_private_replicas": replicas}
    write_json(output / "PRIOR_EXCLUSION_BINDING.json", exclusion_annotation)
    if (len(groups) != len(public) or len(groups) != prepared["tasks"] or len(groups) != data["task_count"]
            or len(set(groups.values())) != prepared["groups"]
            or set(groups) != set(selection["task_ids"])
            or len(selection["task_ids"]) != len(groups)
            or set(groups.values()) != set(selection["groups"])
            or set(groups.values()) & prior_groups
            or selection["prior_training_exclusions"] != config["data"]["exclude_prior_training"]
            or selection["selection_salt"] != config["data"]["selection_salt"]
            or selection["accepted_answers_consulted_for_selection"] is not False
            or selection["model_or_scorer_calls_before_freeze"] != 0):
        raise RuntimeError("Exact selected task/group identities or prior-cohort exclusions differ")
    spec = read_json(root / "model-spec.json")
    validate_model(config, spec)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_path"], local_files_only=True, trust_remote_code=False,
    )
    planned = read_json(root / "requests.json")
    replayed = replay_matrix(tokenizer, public, planned, source, config)
    recorded_preparation = read_json(root / "MIXTURE_PREPARATION.json")
    if any(recorded_preparation[key] != value for key, value in replayed["summary"].items()):
        raise RuntimeError("Mixture preparation cost/source totals differ from independent replay")
    worst_case = sum(cell["expected_prompt_tokens"] + output_cap(cell, config) for cell in planned)
    synthetic_reserve = config["budget"]["maximum_synthetic_request_attempts"] * config["model"]["context_length"]
    if (prepared["requests"] != len(planned)
            or prepared["worst_case_real_tokens"] != worst_case
            or prepared["reserved_synthetic_tokens"] != synthetic_reserve
            or worst_case + synthetic_reserve > config["budget"]["maximum_total_provider_tokens"]
            or prepared["actual_maximum_inflight_requests"] != config["budget"]["execution_workers"]
            or prepared["model_calls"] != 0 or prepared["training_answers_parsed"] is not False
            or prepared["evaluation_answers_or_responses_opened"] is not False
            or prepared["scorer_runtime_mode"] != scorer_runtime_mode(config)):
        raise RuntimeError("Per-cell reservation, request budget, or pre-answer preparation contract differs")
    write_json(output / "ROUTING_VERIFIED.json", replayed)
    # Outcomes and answer contents are opened only after route reconstruction.
    rows, raw_hashes = replay_raw_calls(root, planned, config)
    if sha(data_root / "answers.json") != data["files_sha256"]["answers.json"]:
        raise RuntimeError("Official training answer bytes changed")
    answers = read_json(data_root / "answers.json")
    if set(answers) != set(groups):
        raise RuntimeError("The official answer cohort is incomplete or different")
    scorer = official_scorer({"scorer": {
        "path": str(data_root / "mrqa_official_eval.py"),
        "revision": config["scoring"]["revision"], "sha256": config["scoring"]["source_sha256"],
    }}, runtime=scorer_runtime_mode(config))
    methods, scores, predictions = rescore(rows, answers, scorer, config)
    validate_methods(methods, config)
    result = checked(root / "RESULT.json")
    if (result["schema_version"] != "phiagent-qa-harness-mixture-result/v1"
            or result["methods"] != methods or result["paper_admitted"] is not False
            or result["retrieval_contribution_established"] is not False
            or result["data_manifest_sha256"] != prepared["data_manifest_sha256"]
            or result["config_sha256"] != prepared["config_sha256"]
            or result["official_aggregate_evaluations"] != len(config["arms"])
            or result["per_task_metric_evaluations"] != 2 * len(groups) * len(config["arms"])
            or result["evaluation_answers_or_responses_opened"] is not False
            or result["all_inputs_full_context"] != all(
                cell.get("mixture_receipt", {}).get("identity_context_transform", True) for cell in planned
            )):
        raise RuntimeError("Official aggregate metrics, accounting, or evidence boundaries differ")
    for arm in config["arms"]:
        if (read_json(root / f"{arm}.scores.json") != scores[arm]
                or read_json(root / f"{arm}.predictions.json") != predictions[arm]):
            raise RuntimeError(f"Official per-task metric/prediction replay differs: {arm}")
    helpers = mixture_helpers(config)
    if hasattr(helpers, "result_diagnostics"):
        for key, value in helpers.result_diagnostics(planned, scores, config).items():
            if result.get(key) != value:
                raise RuntimeError(f"Native/provider accounting or paired preservation differs: {key}")
    bounds = diagnostic_bounds(scores, groups, config)
    gate = target_gate(methods, config, bounds)
    if (result["target_gate"] != gate
            or result["mixture_bounds_sha256"] != sha(root / "MIXTURE_BOUNDS.json")
            or checked(root / "MIXTURE_BOUNDS.json")["bounds"] != bounds):
        raise RuntimeError("The matched 10%-token point or finite-sample no-loss gate differs")
    eligible = [config["primary_candidate"]] if (
        gate["point_target_passed"] is True and gate["statistical_target_passed"] is True
    ) else []
    expected_advancement = {
        arm: {"eligible_for_larger_training_study": arm in eligible, "paper_admitted": False}
        for arm in config["candidate_arms"]
    }
    if result["advancement"] != expected_advancement:
        raise RuntimeError("Reported advancement bypassed the paired primary gates")
    decision = read_json(root.parent / "DEVELOPMENT_DECISION.json")
    if (decision["eligible_candidates"] != eligible
            or decision["selected_candidate"] != (eligible[0] if eligible else None)
            or decision["result_sha256"] != sha(root / "RESULT.json")
            or decision["target_gate"] != gate or decision["confirmation_opened"]
            or decision["statistical_guarantee_established"] or decision["paper_admitted"]
            or decision["automatic_official_scale_up_permitted"]):
        raise RuntimeError("Development decision changed the target, primary identity, or claim boundary")
    accounting = verify_accounting(root, methods, planned, config, tokenizer)
    write_json(output / "TOKEN_ACCOUNTING.json", accounting)
    plan = statistical_plan(config)
    write_json(output / "DIAGNOSTIC_BOUNDS.json", {
        "candidate": config["primary_candidate"], "reference": plan["reference"],
        "family_size": plan["family_size"], "familywise_alpha": plan["alpha"], "bounds": bounds,
        "target_gate": gate, "statistical_guarantee_established": False, "paper_admitted": False,
        "claim_boundary": "Numerical training-source bounds are not independent full-public-benchmark evidence",
    })
    value = signed({
        "schema_version": "phiagent-qa-harness-mixture-verification/v1",
        "status": "INDEPENDENT_HARNESS_MIXTURE_REPLAY_VERIFIED",
        "at_utc": now(), "hostname": socket.gethostname(),
        "config_sha256": sha(root / "config.json"), "result_sha256": sha(root / "RESULT.json"),
        "configuration_binding": config_binding, "audit_dependency_binding": dependency_binding,
        "prior_exclusion_binding": exclusion_annotation,
        "requests_sha256": sha(root / "requests.json"), "generation_origin_sha256": sha(root / "origin.json"),
        "data_manifest_sha256": sha(data_root / "manifest.json"), "methods": methods,
        "requests_verified": len(planned), "real_provider_tokens": accounting["real_matrix_acquisition_tokens"],
        "synthetic_provider_tokens": accounting["synthetic_acquisition_tokens"],
        "all_in_acquisition_tokens": accounting["all_in_acquisition_tokens"],
        "target_gate": gate, "independently_eligible": eligible,
        "routing_replay": replayed, "resolved_prior_training_exclusions": resolved,
        "raw_files_sha256": raw_hashes,
        "scores_sha256": {arm: sha(root / f"{arm}.scores.json") for arm in config["arms"]},
        "audit_source_files_sha256": {relative: sha(audit_source / relative) for relative in sorted(audit_files)},
        "model_calls": 0, "paper_admitted": False, "statistical_guarantee_established": False,
        "limits": "Complete matched training-source matrix only; no full official benchmark, optimizer, or production claim.",
    })
    write_json(output / "VERIFIED.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-for-retirement", type=Path)
    parser.add_argument("--verifier-amendment", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new verification directory; never overwrite an audit attempt")
    if args.wait_for_retirement is not None:
        from scripts.verify_compact_harness_training_pilot import await_retirement

        await_retirement(args.experiment, args.wait_for_retirement)
    try:
        result = verify(args.experiment, args.output_dir, verifier_amendment=args.verifier_amendment)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if args.output_dir.is_dir():
            write_json(args.output_dir / "FAILURE.json", {
                "at_utc": now(), "type": type(error).__name__, "message": str(error), "new_model_calls": 0,
            })
        raise
    print(json.dumps({
        "status": result["status"], "requests": result["requests_verified"],
        "target_gate": result["target_gate"], "paper_admitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
