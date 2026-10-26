"""Opt-in TRAIN-source controls for the exact frozen Original CHISEL router.

``lossless`` names the zero-margin hypothesis, not an established property.
No route is fitted, no answer is cached, and no positive paper claim is admitted.
See ``python -m scripts.prepare_qa_lossless_routing_config --help`` for building
the execution configuration and remotely preparing its sampling-only adapter.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket

from phiagent.harness.mixture_router import count_matched_random_routes
from scripts.prepare_chisel_component_quality_data import (
    canonical_hash, question_group, write_json,
)
from scripts.qa_harness_mixture_protocol import (
    BASELINES, INCUMBENT, OVERLAY_FILES as INCUMBENT_FILES,
    FrozenOriginalChisel, artifact_paths, file_sha, fixed_prompt_hashes as incumbent_prompt_hashes,
    stage_incumbent_artifacts, text_sha, validate_methods,
)
from scripts.run_chisel_component_quality import (
    ProtocolError, build_messages, prompt_length, score_rows, valid_usage,
)


SCHEMA = "phiagent-qa-lossless-routing/v1"
PROFILE_BINDING_SCHEMA = "phiagent-qa-canonical-profile-binding/v1"
PAPER_RELEASE = "7b5fd298ea9249b08832cc2da9c356d310d832de9d5b1c1cb860625c10f6df54"
RANDOM = "count_matched_random_route"
AA = "original_aa"
ARMS = (*BASELINES, INCUMBENT, "modify_fixed", RANDOM, AA)
PROFILES = (*BASELINES, "modify")
REFERENCES = tuple(arm for arm in ARMS if arm != INCUMBENT)
QUALITY_REFERENCES = tuple(arm for arm in REFERENCES if arm != AA)
MODELS = {
    "primary_same_model": (
        "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ),
    "second_backbone_transfer": (
        "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
}
DATA_HOSTS = {"h200-1": "harnessmm-worker-1", "h200-2": "harnessmm-worker-2"}
RESULT_STATUS = "COMPLETE_FROZEN_ORIGINAL_ROUTING_TRAIN_DIAGNOSTIC"
RESULT_SCOPE = (
    "Bounded, task-weighted official-MRQA-scored TRAIN-source diagnostic of the "
    "archived Original CHISEL router; not a full benchmark, independent confirmation, "
    "new method, or cross-backbone pooled result."
)
DEVELOPMENT_STATUS = "DIAGNOSTIC_COMPLETE_NO_METHOD_PROMOTION_OR_CUTOFF_RETUNING"
RECORD_CLIENT_TIMING = True
MAX_CHILD_ALLOCATION_SECONDS = 6600
OVERLAY_FILES = tuple(dict.fromkeys((
    *INCUMBENT_FILES,
    "scripts/qa_lossless_routing_protocol.py",
    "scripts/prepare_qa_lossless_routing_config.py",
    "tests/test_qa_lossless_routing_protocol.py",
    "scripts/run_qa_lossless_guarded.py",
    "scripts/cvpr_execution_crossover_leases.py",
    "scripts/prepare_qa_lossless_native.py",
    "scripts/verify_cvpr_qwen35_cache.py",
    "tests/test_qa_lossless_guarded.py",
    "tests/test_qa_lossless_native.py",
)))
ROUTING_CONTROL = {
    "enabled": True,
    "router": "exact_archived_original_chisel",
    "profile_inventory": ["delete", "modify"],
    "route_inputs": ["question", "context"],
    "preserve_full_source_context_question_and_output_contract": True,
    "same_model_seed_temperature_output_cap_and_calls_per_task": True,
    "original_aa_body_identical_fresh_call": True,
    "all_fixed_and_matched_controls_are_new_requests": True,
    "cached_profile_answers_used": False,
    "refit_or_cutoff_retuning": False,
    "labels_outcomes_or_future_fields_used": False,
    "router_model_calls": 0,
}
STATISTICS = {
    "primary_references": list(REFERENCES),
    "primary_metrics": ["exact_match", "f1", "tokens"],
    "quality_gate_references": list(QUALITY_REFERENCES),
    "familywise_alpha": 0.05,
    "simultaneous_bounds": len(REFERENCES) * 3 * 2,
    "noninferiority_margin": 0,
    "statistical_unit": "whole_normalized_question_group",
    "estimand": "task_weighted_paired_difference",
    "interval": "existing_finite_sample_bounds_both_directions_bonferroni",
    "aa_is_a_repeatability_diagnostic_not_an_independent_sample": True,
    "no_optional_stopping": True,
    "zero_difference_bootstrap_is_not_a_certificate": True,
    "nonsignificance_is_not_equality": True,
}
SAMPLER_SCHEMA = "phiagent-compact-harness-training-pilot/v1"
SAMPLER_ADAPTER = {
    "schema_version": "phiagent-qa-lossless-sampling-adapter/v1",
    "sampler": "scripts.prepare_compact_harness_training_pilot.prepare",
    "sampler_transport_schema": SAMPLER_SCHEMA,
    "sampling_view_cannot_execute_model_requests": True,
    "sampler_and_group_selection_unchanged": True,
    "scope": "TRAIN-source selection only; execution retains the new routing-control schema",
}
DIRECT_SAMPLER = {
    "schema_version": "phiagent-qa-lossless-direct-sampling/v1",
    "sampler": "scripts.prepare_compact_harness_training_pilot.prepare",
    "execution_config_used_directly": True,
    "sampler_and_group_selection_unchanged": True,
    "scope": "TRAIN-source selection only; not an experiment or independent confirmation",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _hash(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _integer(value, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _signed(value: dict) -> dict:
    return {**value, "canonical_sha256": canonical_hash(value)}


def _checked(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("canonical_sha256") == canonical_hash({
        key: item for key, item in value.items() if key != "canonical_sha256"
    }), f"Self-hashed artifact differs: {path.name}")
    return value


def body_bytes(body: dict) -> bytes:
    """Match the existing runner's wire serialization, not sorted-key JSON."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_profile_binding(binding: dict) -> None:
    _require(set(binding) == {
        "schema_version", "paper_release_sha256", "source_manifest", "prompt_sha256",
    }, "An explicit canonical parent profile binding is required")
    _require(
        binding["schema_version"] == PROFILE_BINDING_SCHEMA
        and binding["paper_release_sha256"] == PAPER_RELEASE
        and set(binding["prompt_sha256"]) == set(PROFILES)
        and all(_hash(value) for value in binding["prompt_sha256"].values()),
        "Canonical profiles must bind all five exact parent artifacts and the current release",
    )
    manifest = binding["source_manifest"]
    _require(
        set(manifest) == {"source_path", "relative_path", "sha256", "hash_field"}
        and manifest["relative_path"] == "canonical-profile-binding/source-manifest.json"
        and _hash(manifest["sha256"])
        and manifest["hash_field"] in {"source_sha256", "files_sha256", "prompt_sha256"}
        and (manifest["source_path"] is None or (
            isinstance(manifest["source_path"], str) and bool(manifest["source_path"])
        )),
        "Canonical profile provenance requires an exact supplied source-manifest hash",
    )


def validate_config(config: dict, *, require_resource_audit: bool = True) -> None:
    """Fail closed on opt-in identity, scientific contract, and bounded resource metadata."""
    try:
        _validate_config(config, require_resource_audit=require_resource_audit)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ProtocolError(f"Incomplete routing-control configuration: {error}") from error


def _validate_config(config: dict, *, require_resource_audit: bool = True) -> None:
    _require(
        config.get("schema_version") == SCHEMA
        and config["arms"] == list(ARMS) and config["comparison_arms"] == list(ARMS)
        and config["candidate_arms"] == [] and config["selection_eligible_arms"] == []
        and config["primary_candidate"] == config["primary_reference"] == INCUMBENT
        and config["cost_reference"] == "autosaddler"
        and config["routing_control"] == ROUTING_CONTROL
        and config["data_preparation"] in (DIRECT_SAMPLER, SAMPLER_ADAPTER)
        and config["paper_admitted"] is False
        and config["confirmation"]["status"] == "NOT_AUTHORIZED_BY_THIS_DIAGNOSTIC"
        and config["confirmation"]["evaluation_answers_or_responses_may_be_opened"] is False,
        "The complete eight-arm, opt-in frozen Original diagnostic contract changed",
    )
    _require(not any(config.get(key) is not None for key in (
        "mixture_router", "capability_routing", "preservation", "calibration",
        "context_selection", "rule_compression", "length_matched_controls", "factorial",
        "cutoff", "threshold", "route_overrides", "cached_answers",
    )), "No other intervention, cutoff override, or answer cache may compose with this protocol")
    incumbent = config["incumbent"]
    _require(
        incumbent["status"] == "BOUND" and incumbent["arm"] == INCUMBENT
        and incumbent["allow_delete_diagnostic"] is False
        and config["frozen_incumbent_sha256"] == canonical_hash(incumbent),
        "The literal BOUND incumbent must be retained; no fallback, refit, or relabeling",
    )
    required = {"training", "routes", "protocol", "harness_source", "training_source", "runner_source"}
    _require(set(incumbent["artifacts"]) == required
             and all(_hash(row["sha256"]) for row in incumbent["artifacts"].values()),
             "All six immutable incumbent artifacts need exact hashes")
    paths = artifact_paths(Path("."), config)
    _require(len({str(path) for path in paths.values()}) == len(required),
             "Incumbent artifact paths must be distinct")
    _require(type(incumbent["historical_display"]["score_threshold"]) in (int, float)
             and math.isfinite(incumbent["historical_display"]["score_threshold"]),
             "The archived threshold must be explicit and finite")
    validate_profile_binding(config["canonical_profiles"])
    _require(config["prompt_sha256"] == config["canonical_profiles"]["prompt_sha256"],
             "Execution profiles must equal the explicitly supplied canonical parent profiles")
    _require(config["archived_routing_prompt_sha256"] == {
        name: config["prompt_sha256"][name] for name in ("delete", "modify")
    }, "DELETE and MODIFY cannot change with a baseline profile correction")
    model, budget, data = config["model"], config["budget"], config["data"]
    identity = config["backbone_identity"]
    _require(
        identity["kind"] in MODELS
        and (model["repository"], model["revision"]) == MODELS[identity["kind"]]
        and identity["cross_backbone_scores_may_be_pooled"] is False
        and identity["current_paper_release_sha256"] == PAPER_RELEASE
        and model["dtype"] == "bfloat16" and model["tensor_parallel_size"] == 1
        and type(model["temperature"]) in (int, float) and model["temperature"] == 0
        and model["enable_thinking"] is False
        and _integer(model["request_seed"], 0, 2**32 - 1)
        and _integer(model["context_length"], 2, 131072)
        and _integer(model["max_output_tokens"], 1, model["context_length"] - 1),
        "A single pinned backbone, seed, native context, and positive common output cap are required",
    )
    _require(
        _integer(budget["maximum_real_request_attempts"], len(ARMS), 4095)
        and budget["maximum_synthetic_request_attempts"] == 1
        and budget["maximum_physical_gpus"] == 1
        and _integer(budget["maximum_allocation_seconds"], 1, MAX_CHILD_ALLOCATION_SECONDS)
        and _integer(budget["maximum_total_provider_tokens"], 1, 16_000_000)
        and _integer(budget["maximum_inflight_requests"], 1, 16)
        and _integer(budget["execution_workers"], 1, budget["maximum_inflight_requests"])
        and type(budget["transport_retries"]) is int and budget["transport_retries"] == 0,
        "The child must stay within 4096 calls, 16M tokens, 6600 GPU-seconds and 16 inflight; "
        "600 seconds remain reserved within the parent's 7200-second guard ceiling",
    )
    _require(
        data["mode"] == "disjoint_prepared_training"
        and _integer(data["question_groups"], 1, 256)
        and _integer(data["maximum_tasks"], data["question_groups"], 511)
        and data["maximum_tasks"] * len(ARMS) <= budget["maximum_real_request_attempts"]
        and data["preserve_whole_question_groups"] is True
        and data["preparation_host"] in DATA_HOSTS
        and data["preserve_full_original_context_and_question_in_source_artifacts"] is True
        and data["selection_uses_answers"] is False
        and data["historical_freshness_certified"] is False
        and isinstance(data["selection_salt"], str) and bool(data["selection_salt"])
        and _hash(data["component_data_manifest_sha256"])
        and _hash(data["train_archive_sha256"]),
        "A bounded whole-group TRAIN-source selection, never an independent confirmation, is required",
    )
    _require(all(_hash(entry["sha256"]) for entry in data["exclude_prior_training"]),
             "Prior selection-freeze hashes must be resolved prospectively by the parent")
    control = config["count_matched_random"]
    _require(set(control) == {
        "source_arm", "control_arm", "stratification", "context_token_boundaries", "seed",
        "preserve_realized_profile_counts", "assignments_frozen_before_answers_or_outcomes",
        "actual_token_equality_assumed",
    }, "Random assignment accepts only predeclared input/route-inventory fields")
    boundaries = control["context_token_boundaries"]
    _require(
        control["source_arm"] == INCUMBENT and control["control_arm"] == RANDOM
        and control["stratification"] == "original_context_exact_token_count"
        and type(control["seed"]) is int and control["seed"] == model["request_seed"]
        and control["preserve_realized_profile_counts"] is True
        and control["assignments_frozen_before_answers_or_outcomes"] is True
        and control["actual_token_equality_assumed"] is False
        and isinstance(boundaries, list) and bool(boundaries)
        and all(_integer(boundary, 1, model["context_length"]) for boundary in boundaries)
        and boundaries == sorted(set(boundaries)),
        "Random routes require frozen native context-length strata and exact DELETE/MODIFY counts",
    )
    _require(
        config["scoring"]["metrics"] == ["exact_match", "f1"]
        and config["scoring"]["repository"] == "mrqa/MRQA-Shared-Task-2019"
        and _hash(config["scoring"]["source_sha256"])
        and config["scoring"]["score_only_complete_paired_matrix"] is True
        and config["scoring"]["empty_and_length_stopped_answers_are_retained_and_charged"] is True,
        "Unchanged official MRQA EM/F1 and complete known-usage scoring are required",
    )
    resource = config["resource"]
    allowed_resource_status = {"BOUND_AFTER_FRESH_AUDIT"}
    if not require_resource_audit:
        allowed_resource_status.add("CACHE_READY_REQUIRES_FRESH_GPU_AND_LEASE")
    _require(
        resource["binding_status"] in allowed_resource_status
        and resource["hardware_name"] == "H200"
        and _integer(resource["physical_gpu_index"], 0, 255)
        and isinstance(resource["gpu_uuid"], str) and resource["gpu_uuid"].startswith("GPU-")
        and all(isinstance(resource[name], str) and resource[name] for name in (
            "selected_hostname", "service_root", "lease_root",
        ))
        and (resource["binding_status"] != "BOUND_AFTER_FRESH_AUDIT"
             or isinstance(resource.get("selection_reason"), str) and bool(resource["selection_reason"]))
        and _integer(resource["port"], 1024, 65535)
        and resource["fresh_GPU_and_exclusive_lease_required_before_launch"] is True
        and resource["no_other_study_service_reuse"] is True
        and _hash(resource["native_runtime"]["manifest_sha256"])
        and all(isinstance(resource["native_runtime"][name], str)
                and resource["native_runtime"][name] for name in (
                    "manifest_path", "model_path", "python_executable",
                )),
        "Parent-supplied freshly audited native H200 resource and model-manifest bindings are required",
    )
    statistical_plan(config)


def statistical_plan(config: dict) -> dict:
    _require(config["statistics"] == STATISTICS,
             "The zero-margin complete simultaneous endpoint family cannot change")
    return {
        "contrasts": [(INCUMBENT, reference, metric)
                      for reference in REFERENCES for metric in ("exact_match", "f1", "tokens")],
        "family_size": STATISTICS["simultaneous_bounds"],
        "alpha": STATISTICS["familywise_alpha"], "reference": INCUMBENT,
    }


def fixed_prompt_hashes(config: dict) -> dict:
    return incumbent_prompt_hashes(config)


def prompt_artifact_names(_config: dict) -> list[str]:
    # Empty candidate_arms prevents lifecycle creation of unrelated mixture prompts.
    return list(PROFILES)


def stage_incumbent(source: Path, config: dict) -> list[str]:
    from scripts.prepare_compact_harness_training_pilot import prior_exclusion_replica_bindings

    validate_config(config)
    files = stage_incumbent_artifacts(source, config)
    record = config["canonical_profiles"]["source_manifest"]
    destination = source / record["relative_path"]
    _require(destination.resolve().is_relative_to(source.resolve()),
             "Canonical profile provenance must remain inside the new source snapshot")
    if not destination.exists():
        _require(bool(record["source_path"]), "Canonical parent source manifest needs prestaging")
        original = Path(record["source_path"])
        _require(file_sha(original) == record["sha256"], "Canonical parent manifest hash differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    _require(file_sha(destination) == record["sha256"], "Staged canonical manifest hash differs")
    manifest = json.loads(destination.read_text(encoding="utf-8"))
    _require(all(
        manifest[record["hash_field"]].get(
            arm if record["hash_field"] == "prompt_sha256" else f"prompts/{arm}.system.txt"
        ) == expected
        and file_sha(source / f"prompts/{arm}.system.txt") == expected
        for arm, expected in fixed_prompt_hashes(config).items()
    ), "Canonical parent profile inventory differs; historical SkillOpt is not a substitute")
    FrozenOriginalChisel(source, config)
    replicas = prior_exclusion_replica_bindings(
        source, config["data"]["exclude_prior_training"],
        required=config["data"]["preparation_host"] != config["resource"]["selected_host_alias"],
    )
    return [*files, record["relative_path"], *(entry["relative_path"] for entry in replicas)]


def _validate_items(items: list[dict], config: dict) -> None:
    _require(bool(items) and all(
        isinstance(item, dict) and set(item) == {"id", "group", "question", "context"}
        and all(isinstance(item[name], str) and bool(item[name]) for name in item)
        for item in items
    ), "Only complete public id/group/question/context inputs are allowed; no labels or future fields")
    _require(len({item["id"] for item in items}) == len(items),
             "Duplicate task IDs cannot enter the paired matrix")
    _require(all(item["group"] == question_group(item["question"]) for item in items),
             "Grouping must use the unchanged normalized-question grouping function")
    _require(len({item["group"] for item in items}) == config["data"]["question_groups"]
             and len(items) <= config["data"]["maximum_tasks"],
             "The complete predeclared group inventory and task budget are required")


def _validate_route(route: dict, config: dict) -> None:
    _require(
        set(route) == {
            "status", "variant", "score", "threshold", "features", "normalized_features",
            "source_artifacts_sha256", "refit", "answers_or_actor_responses_used",
            "router_model_calls", "request_protocol",
        }
        and route["status"] == "EXACT_FROZEN_ORIGINAL_ROUTER_RERUN"
        and route["variant"] in {"delete", "modify"}
        and type(route["score"]) in (int, float) and math.isfinite(route["score"])
        and route["threshold"] == config["incumbent"]["historical_display"]["score_threshold"]
        and route["variant"] == ("modify" if route["score"] >= route["threshold"] else "delete")
        and route["source_artifacts_sha256"] == {
            key: value["sha256"] for key, value in config["incumbent"]["artifacts"].items()
        }
        and route["refit"] is False and route["answers_or_actor_responses_used"] is False
        and route["router_model_calls"] == 0,
        "The exact archived feature/head/threshold decision must be rerun without outcome access",
    )


def prepare_cells(tokenizer, items: list[dict], prompts: dict, config: dict, source: Path):
    validate_config(config)
    _validate_items(items, config)
    _require(set(prompts) == set(PROFILES) and all(
        isinstance(prompts[name], str) and text_sha(prompts[name]) == expected
        for name, expected in fixed_prompt_hashes(config).items()
    ), "Only the exact five hash-bound canonical system profiles may be used")
    router = FrozenOriginalChisel(source, config)
    routes, context_tokens, strata = {}, {}, {}
    for item in items:
        task = item["id"]
        routes[task] = router.route(item)
        _validate_route(routes[task], config)
        tokens = tokenizer.encode(item["context"], add_special_tokens=False)
        _require(isinstance(tokens, list) and all(type(token) is int for token in tokens),
                 "Native context stratification requires exact tokenizer IDs")
        context_tokens[task] = len(tokens)
        strata[task] = f"context_tokens_bucket_{sum(len(tokens) >= boundary for boundary in config['count_matched_random']['context_token_boundaries'])}"
    inventory = {task: route["variant"] for task, route in routes.items()}
    assignments = count_matched_random_routes(
        inventory, strata=strata, seed=config["count_matched_random"]["seed"],
    )
    assignment_binding = {
        "source_routes_sha256": canonical_hash(inventory),
        "strata_sha256": canonical_hash(strata),
        "assignments_sha256": canonical_hash(assignments),
        "policy_sha256": canonical_hash(config["count_matched_random"]),
    }
    planned = []
    for item in items:
        task = item["id"]
        for arm in ARMS:
            profile = (
                inventory[task] if arm in {INCUMBENT, AA}
                else "modify" if arm == "modify_fixed"
                else assignments[task] if arm == RANDOM else arm
            )
            messages = build_messages(prompts[profile], item)
            cap = config["model"]["max_output_tokens"]
            body = {
                "model": config["model"]["repository"], "messages": messages,
                "temperature": 0, "max_tokens": cap, "seed": config["model"]["request_seed"],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            # write_json sorts keys recursively; bind the bytes the runner reloads and sends.
            body = json.loads(json.dumps(body, sort_keys=True, ensure_ascii=False))
            planned.append({
                "cell": canonical_hash([task, arm, config["model"]["request_seed"]]),
                "task": task, "group": item["group"], "arm": arm,
                "expected_prompt_tokens": prompt_length(tokenizer, messages),
                "max_output_tokens": cap, "reference_arm": INCUMBENT,
                "request_profile": profile, "request_profile_sha256": text_sha(prompts[profile]),
                "request_body_sha256": hashlib.sha256(body_bytes(body)).hexdigest(),
                "input_sha256": canonical_hash(item),
                "incumbent_route": deepcopy(routes[task]),
                "routing_control_receipt": {
                    "source_context_tokens": context_tokens[task], "stratum": strata[task],
                    "original_profile": inventory[task], "random_profile": assignments[task],
                    **assignment_binding,
                    "answers_outcomes_or_future_fields_used": False,
                    "count_matching_is_not_actual_token_matching": True,
                },
                "body": body,
            })
    planned.sort(key=lambda cell: cell["cell"])
    return planned, preparation_summary(planned, config)


def output_cap(cell: dict, config: dict) -> int:
    body, cap = cell["body"], config["model"]["max_output_tokens"]
    _require(
        cell["arm"] in ARMS and type(cell["max_output_tokens"]) is int
        and cell["max_output_tokens"] == cap and type(body["max_tokens"]) is int
        and body["max_tokens"] == cap
        and set(body) == {
            "model", "messages", "temperature", "max_tokens", "seed", "chat_template_kwargs",
        }
        and body["model"] == config["model"]["repository"]
        and type(body["temperature"]) in (int, float) and body["temperature"] == 0
        and type(body["seed"]) is int and body["seed"] == config["model"]["request_seed"]
        and body["chat_template_kwargs"] == {"enable_thinking": False}
        and _integer(cell["expected_prompt_tokens"], 1, config["model"]["context_length"] - cap)
        and cell.get("execution_kind", "model") == "model",
        "Every arm must execute a newly generated, same-model full-context request with the same cap",
    )
    _require(
        len(body["messages"]) == 2
        and [message.get("role") for message in body["messages"]] == ["system", "user"]
        and all(set(message) == {"role", "content"} and isinstance(message["content"], str)
                for message in body["messages"])
        and cell["request_profile"] in PROFILES
        and cell["request_profile_sha256"] == config["prompt_sha256"][cell["request_profile"]]
        and text_sha(body["messages"][0]["content"]) == cell["request_profile_sha256"]
        and hashlib.sha256(body_bytes(body)).hexdigest() == cell["request_body_sha256"],
        "The immutable profile, information envelope, or actual request bytes changed",
    )
    return cap


def validate_matrix(planned: list[dict], config: dict) -> dict:
    validate_config(config)
    _require(bool(planned), "An empty request matrix is invalid")
    cells, ids = {}, set()
    for cell in planned:
        _require(set(cell) == {
            "cell", "task", "group", "arm", "expected_prompt_tokens", "max_output_tokens",
            "reference_arm", "request_profile", "request_profile_sha256", "request_body_sha256",
            "input_sha256", "incumbent_route", "routing_control_receipt", "body",
        }, "Planned cells cannot include undeclared labels, future fields or execution overrides")
        output_cap(cell, config)
        _validate_route(cell["incumbent_route"], config)
        pair = cell["task"], cell["arm"]
        _require(pair not in cells and cell["cell"] not in ids
                 and cell["cell"] == canonical_hash([*pair, config["model"]["request_seed"]]),
                 "Duplicate or altered cell/task/arm identities")
        cells[pair] = cell
        ids.add(cell["cell"])
    tasks = {task for task, _arm in cells}
    _require(set(cells) == {(task, arm) for task in tasks for arm in ARMS},
             "Every unique task must have all eight newly executed arms")
    groups, inventory, strata, assignments = {}, {}, {}, {}
    for task in sorted(tasks):
        original = cells[task, INCUMBENT]
        groups[task] = original["group"]
        _require(isinstance(task, str) and bool(task)
                 and isinstance(groups[task], str) and bool(groups[task]),
                 "Task/group identities must be nonempty strings")
        receipt = original["routing_control_receipt"]
        _require(set(receipt) == {
            "source_context_tokens", "stratum", "original_profile", "random_profile",
            "source_routes_sha256", "strata_sha256", "assignments_sha256", "policy_sha256",
            "answers_outcomes_or_future_fields_used", "count_matching_is_not_actual_token_matching",
        }, "Routing receipts cannot include labels, outcomes or future information")
        inventory[task] = original["incumbent_route"]["variant"]
        strata[task], assignments[task] = receipt["stratum"], receipt["random_profile"]
        count = receipt["source_context_tokens"]
        _require(_integer(count, 0, config["model"]["context_length"])
                 and strata[task] == f"context_tokens_bucket_{sum(count >= boundary for boundary in config['count_matched_random']['context_token_boundaries'])}"
                 and receipt["original_profile"] == inventory[task]
                 and receipt["answers_outcomes_or_future_fields_used"] is False
                 and receipt["count_matching_is_not_actual_token_matching"] is True,
                 "Random routing must retain its predeclared input-only length stratum")
        for arm in ARMS:
            cell = cells[task, arm]
            expected = (
                inventory[task] if arm in {INCUMBENT, AA} else "modify" if arm == "modify_fixed"
                else assignments[task] if arm == RANDOM else arm
            )
            _require(
                cell["group"] == groups[task] and cell["input_sha256"] == original["input_sha256"]
                and cell["body"]["messages"][1] == original["body"]["messages"][1]
                and cell["incumbent_route"] == original["incumbent_route"]
                and cell["routing_control_receipt"] == receipt
                and cell["request_profile"] == expected,
                "Task grouping, source information, or a fixed/matched routing control changed",
            )
        _require(body_bytes(cells[task, AA]["body"]) == body_bytes(original["body"])
                 and cells[task, AA]["cell"] != original["cell"]
                 and cells[task, AA]["expected_prompt_tokens"] == original["expected_prompt_tokens"],
                 "Original A/A must have byte-identical requests but a separate fresh call identity")
    _require(len(set(groups.values())) == config["data"]["question_groups"]
             and len(tasks) <= config["data"]["maximum_tasks"],
             "The paired request group count differs from the frozen cohort")
    _require(assignments == count_matched_random_routes(
        inventory, strata=strata, seed=config["count_matched_random"]["seed"],
    ), "Count-matched assignment cannot be changed after the input-only seed was frozen")
    bindings = {
        "source_routes_sha256": canonical_hash(inventory), "strata_sha256": canonical_hash(strata),
        "assignments_sha256": canonical_hash(assignments),
        "policy_sha256": canonical_hash(config["count_matched_random"]),
    }
    _require(all(all(cell["routing_control_receipt"][key] == value
                     for key, value in bindings.items()) for cell in planned),
             "Cohort-level route/stratum/assignment hashes differ")
    reserved = sum(cell["expected_prompt_tokens"] + output_cap(cell, config) for cell in planned)
    _require(len(planned) <= config["budget"]["maximum_real_request_attempts"]
             and reserved + config["model"]["context_length"]
             <= config["budget"]["maximum_total_provider_tokens"],
             "Complete paired requests and the synthetic reserve exceed the frozen budget")
    return {"cells": cells, "groups": groups, "routes": inventory,
            "strata": strata, "assignments": assignments}


def _matching_summary(matrix: dict, prompt_tokens: dict) -> dict:
    strata, routes, assignments = (matrix[name] for name in ("strata", "routes", "assignments"))
    result = {}
    for stratum in sorted(set(strata.values())):
        tasks = [task for task in strata if strata[task] == stratum]
        original, random = Counter(routes[task] for task in tasks), Counter(
            assignments[task] for task in tasks
        )
        _require(original == random, "DELETE/MODIFY assignment counts differ within a stratum")
        original_tokens = sum(prompt_tokens[task, INCUMBENT] for task in tasks)
        random_tokens = sum(prompt_tokens[task, RANDOM] for task in tasks)
        result[stratum] = {
            "tasks": len(tasks), "original_profile_counts": dict(sorted(original.items())),
            "random_profile_counts": dict(sorted(random.items())),
            "original_prompt_tokens": original_tokens, "random_prompt_tokens": random_tokens,
            "random_minus_original_prompt_tokens": random_tokens - original_tokens,
            "full_request_input_tokens_equal": random_tokens == original_tokens,
        }
    original_total = sum(prompt_tokens[task, INCUMBENT] for task in strata)
    random_total = sum(prompt_tokens[task, RANDOM] for task in strata)
    return {
        "strata": result, "exact_profile_counts_matched_within_every_stratum": True,
        "changed_assignments": sum(routes[task] != assignments[task] for task in strata),
        "original_prompt_tokens": original_total, "random_prompt_tokens": random_total,
        "random_minus_original_prompt_tokens": random_total - original_total,
        "full_request_input_token_totals_equal": original_total == random_total,
        "full_request_input_tokens_equal_in_every_stratum": all(
            row["full_request_input_tokens_equal"] for row in result.values()
        ),
        "exact_actual_total_cost_matching_by_design": False,
        "realized_output_tokens_are_not_constrained_equal": True,
    }


def preparation_summary(planned: list[dict], config: dict) -> dict:
    matrix = validate_matrix(planned, config)
    return {
        "status": "FROZEN_ORIGINAL_AND_FRESH_CONTROLS_PREPARED_BEFORE_OUTCOMES",
        "primary_baseline_status": "BOUND", "reference_arm": INCUMBENT,
        "tasks": len(matrix["groups"]), "question_groups": len(set(matrix["groups"].values())),
        "requests": len(planned), "unique_context_transforms": 0,
        "router_model_calls": 0, "router_provider_tokens": 0,
        "selection_uses_answers_or_actor_responses": False,
        "original_aa_byte_identical_separate_request": True,
        "count_matched_random": {
            "measurement": "full_native_chat_template_tokenizer_not_provider_usage_yet",
            **_matching_summary(matrix, {
                pair: cell["expected_prompt_tokens"] for pair, cell in matrix["cells"].items()
            }),
        },
        "methods": {
            arm: {
                "tasks": len(matrix["groups"]),
                "prompt_tokens": sum(cell["expected_prompt_tokens"]
                                     for cell in planned if cell["arm"] == arm),
                "reserved_completion_tokens": len(matrix["groups"]) * config["model"]["max_output_tokens"],
                "selected_profiles": dict(Counter(
                    cell["request_profile"] for cell in planned if cell["arm"] == arm
                )),
            } for arm in ARMS
        },
        "scope": RESULT_SCOPE, "no_loss_claim_established": False,
    }


def score_records(records: dict, answers: dict, scorer) -> dict:
    _require(set(records) == set(answers) and bool(records), "Partial scoring is forbidden")
    for task, record in records.items():
        usage = valid_usage(record["raw"])
        _require(usage is not None and usage == record["usage"] and usage["prompt_tokens"] > 0,
                 "Unknown, noninteger, missing, or nonadditive provider usage invalidates the matrix")
        _require(record["task"] == task and record["arm"] in ARMS
                 and record["finish_reason"] in {"stop", "length"},
                 "A newly executed model record with its correct task/arm identity is required")
    result = score_rows(records, answers, scorer)
    for task, row in result.items():
        record = records[task]
        content = record["raw"]["choices"][0]["message"]["content"]
        _require(isinstance(content, str), "Raw provider output must be text, including empty output")
        row.update({
            "task": task, "group": record["group"], "arm": record["arm"],
            "provider_model": record["raw"]["model"],
            "raw_content_sha256": text_sha(content), "response_sha256": record["response_sha256"],
        })
    return result


def _validate_scores(scores: dict, groups: dict, config: dict) -> None:
    _require(bool(groups) and set(scores) == set(ARMS)
             and all(set(rows) == set(groups) for rows in scores.values()),
             "Every paired task in all eight arms is required; omissions cannot be imputed")
    _require(len(set(groups.values())) == config["data"]["question_groups"],
             "Statistical groups must match the prospective whole-group cohort")
    for arm, rows in scores.items():
        for task, row in rows.items():
            _require(
                row["task"] == task and row["arm"] == arm and row["group"] == groups[task]
                and isinstance(groups[task], str) and bool(groups[task])
                and row["provider_model"] == config["model"]["repository"]
                and type(row["exact_match"]) is int and row["exact_match"] in (0, 1)
                and type(row["f1"]) in (int, float) and math.isfinite(row["f1"])
                and 0 <= row["f1"] <= 1
                and isinstance(row["prediction"], str) and _hash(row["raw_content_sha256"])
                and _hash(row["response_sha256"])
                and row["finish_reason"] in {"stop", "length"}
                and type(row["empty_answer"]) is bool
                and row["empty_answer"] == (row["prediction"] == "")
                and _integer(row["prompt_tokens"], 1, config["model"]["context_length"])
                and _integer(row["completion_tokens"], 0, config["model"]["max_output_tokens"])
                and _integer(row["tokens"], 1, config["model"]["context_length"])
                and row["tokens"] == row["prompt_tokens"] + row["completion_tokens"],
                "Paired scores require official finite EM/F1, grouping, model identity and exact actual usage",
            )


def diagnostic_bounds(scores: dict, groups: dict[str, str], config: dict) -> dict:
    from phiagent.evaluation.chisel_component_quality import paired_finite_sample_lower_bound

    validate_config(config)
    _validate_scores(scores, groups, config)
    plan, seed, bounds = statistical_plan(config), config["model"]["request_seed"], {}
    for left, right, metric in plan["contrasts"]:
        lvalues = {task: row[metric] for task, row in scores[left].items()}
        rvalues = {task: row[metric] for task, row in scores[right].items()}
        if metric == "tokens":
            lvalues, rvalues = rvalues, lvalues
        arguments = {
            "groups": groups, "seeds": (seed,),
            "score_upper_bound": config["model"]["context_length"] if metric == "tokens" else 1,
            "family_size": plan["family_size"], "alpha": plan["alpha"],
            "binary": metric == "exact_match",
        }
        lower = paired_finite_sample_lower_bound({seed: lvalues}, {seed: rvalues}, **arguments)
        reverse = paired_finite_sample_lower_bound({seed: rvalues}, {seed: lvalues}, **arguments)
        bounds[f"{left}_vs_{right}_{metric}"] = {
            **lower, "simultaneous_upper_bound": -reverse["one_sided_lower_bound"],
            "reverse_bound": reverse,
            "direction": "reference_minus_original_tokens" if metric == "tokens" else "original_minus_reference",
            "all_observed_differences_zero": lvalues == rvalues,
            "zero_difference_bootstrap_certificate_used": False,
            "population_no_loss_claim_established": False,
        }
    return bounds


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    validate_config(config)
    validate_methods(methods, config)
    candidate, reference = methods[INCUMBENT], methods[config["cost_reference"]]
    quality = all(candidate[metric] >= methods[arm][metric]
                  for arm in QUALITY_REFERENCES for metric in ("exact_match", "f1"))
    cheaper = candidate["total_tokens"] < reference["total_tokens"]
    point = quality and cheaper
    numerical = None
    if bounds is not None:
        expected = {f"{left}_vs_{right}_{metric}"
                    for left, right, metric in statistical_plan(config)["contrasts"]}
        _require(set(bounds) == expected, "The full simultaneous uncertainty family is required")
        for bound in bounds.values():
            _require(
                bound["family_size"] == STATISTICS["simultaneous_bounds"]
                and bound["familywise_alpha"] == 0.05
                and type(bound["one_sided_lower_bound"]) in (int, float)
                and math.isfinite(bound["one_sided_lower_bound"])
                and bound["zero_difference_bootstrap_certificate_used"] is False
                and bound["method"] in {
                    "WEIGHTED_QUESTION_GROUP_HOEFFDING_ONE_SIDED_BONFERRONI",
                    "PAIRED_GAIN_MINUS_LOSS_CLOPPER_PEARSON_BONFERRONI",
                }
                and (not bound["all_observed_differences_zero"]
                     or bound["one_sided_lower_bound"] < 0),
                "Degenerate equality intervals cannot certify a zero-margin no-loss hypothesis",
            )
        endpoints = [f"{INCUMBENT}_vs_{arm}_{metric}"
                     for arm in QUALITY_REFERENCES for metric in ("exact_match", "f1")]
        numerical = point and all(bounds[key]["one_sided_lower_bound"] >= 0 for key in endpoints)
        numerical = numerical and bounds[
            f"{INCUMBENT}_vs_{config['cost_reference']}_tokens"
        ]["one_sided_lower_bound"] > 0
    return {
        "status": "OBSERVED_ZERO_MARGIN_QUALITY_AND_TOKEN_SAVING_POINT_PASS" if point
                  else "OBSERVED_ZERO_MARGIN_QUALITY_AND_TOKEN_SAVING_POINT_FAIL",
        "candidate": INCUMBENT, "primary_reference": INCUMBENT, "primary_baseline_status": "BOUND",
        "cost_reference": config["cost_reference"], "quality_references": list(QUALITY_REFERENCES),
        "quality_margin": 0, "observed_no_average_quality_damage": quality,
        "actual_total_tokens_strictly_below_cost_reference": cheaper,
        "observed_total_token_ratio": candidate["total_tokens"] / reference["total_tokens"],
        "point_target_passed": point, "simultaneous_numerical_target_passed": numerical,
        "statistical_target_passed": False,
        "statistical_bound_conditions_established": False,
        "no_loss_claim_established": False, "per_task_nonregression_guaranteed": False,
        "aa_is_not_a_quality_gate_reference": True,
        "no_candidate_promotion_in_this_diagnostic": True,
        "automatic_official_scale_up_permitted": False, "paper_admitted": False,
        "production_defaults_changed": False, "comparison_scope": RESULT_SCOPE,
    }


def result_diagnostics(planned: list[dict], scores: dict, config: dict) -> dict:
    matrix = validate_matrix(planned, config)
    groups, cells = matrix["groups"], matrix["cells"]
    _validate_scores(scores, groups, config)
    _require(all(
        scores[arm][task]["prompt_tokens"] == cell["expected_prompt_tokens"]
        for (task, arm), cell in cells.items()
    ), "Actual full-request provider prompt usage differs from the frozen native tokenizer")
    paired = {}
    for reference in REFERENCES:
        tasks = {}
        for task in sorted(groups):
            original, other = scores[INCUMBENT][task], scores[reference][task]
            tasks[task] = {
                "group": groups[task], "original_profile": cells[task, INCUMBENT]["request_profile"],
                "reference_profile": cells[task, reference]["request_profile"],
                "exact_match_difference": original["exact_match"] - other["exact_match"],
                "f1_difference": original["f1"] - other["f1"],
                "reference_minus_original_prompt_tokens": other["prompt_tokens"] - original["prompt_tokens"],
                "reference_minus_original_completion_tokens": other["completion_tokens"] - original["completion_tokens"],
                "reference_minus_original_total_tokens": other["tokens"] - original["tokens"],
                "original_usage": {key: original[key] for key in ("prompt_tokens", "completion_tokens", "tokens")},
                "reference_usage": {key: other[key] for key in ("prompt_tokens", "completion_tokens", "tokens")},
                "extracted_answer_equal": original["prediction"] == other["prediction"],
                "raw_content_equal": original["raw_content_sha256"] == other["raw_content_sha256"],
            }
        paired[reference] = {
            "per_task": tasks,
            "em_gains": sum(row["exact_match_difference"] > 0 for row in tasks.values()),
            "em_losses": sum(row["exact_match_difference"] < 0 for row in tasks.values()),
            "em_difference_pp": 100 * math.fsum(row["exact_match_difference"] for row in tasks.values()) / len(tasks),
            "f1_difference_pp": 100 * math.fsum(row["f1_difference"] for row in tasks.values()) / len(tasks),
            "actual_total_tokens_saved": sum(row["reference_minus_original_total_tokens"] for row in tasks.values()),
        }
    aa_tasks = paired[AA]["per_task"]
    differences = {
        name: [task for task, row in aa_tasks.items() if not row[field]]
        for name, field in (("raw_content", "raw_content_equal"), ("extracted_answer", "extracted_answer_equal"))
    }
    matching = _matching_summary(matrix, {
        (task, arm): row["prompt_tokens"] for arm, rows in scores.items() for task, row in rows.items()
    })
    return {
        "routing_control_diagnostics": {
            "schema_version": SCHEMA, "scope": RESULT_SCOPE,
            "backbone_identity": config["backbone_identity"], "model": config["model"],
            "paired_official_metrics_and_actual_usage": paired,
            "native_provider_prompt_token_matching": {
                "measurement": "actual_provider_full_request_prompt_tokens",
                "status": "INPUT_TOTALS_MATCHED" if matching["full_request_input_tokens_equal_in_every_stratum"]
                          else "INPUT_TOKEN_MISMATCH_REPORTED_NOT_COST_MATCHED",
                **matching,
            },
            "original_aa": {
                "request_bodies_byte_identical": True, "separate_call_identities": True,
                "historical_results_reused": False,
                "changed_task_ids": differences,
                "changed_question_groups": {
                    field: len({groups[task] for task in tasks}) for field, tasks in differences.items()
                },
                "em_difference_pp": paired[AA]["em_difference_pp"],
                "f1_difference_pp": paired[AA]["f1_difference_pp"],
                "actual_total_tokens_saved": paired[AA]["actual_total_tokens_saved"],
                "nonsignificance_used_as_equality": False, "population_determinism_established": False,
            },
            "paper_admitted": False, "no_loss_claim_established": False,
        },
    }


def replay_matrix(tokenizer, public, planned, source, config) -> dict:
    prompts = {name: (source / f"prompts/{name}.system.txt").read_text(encoding="utf-8")
               for name in prompt_artifact_names(config)}
    rebuilt, summary = prepare_cells(tokenizer, public, prompts, config, source)
    _require(rebuilt == planned, "Frozen Original routing/request/assignment replay differs")
    return {
        "summary": summary, "complete_request_and_receipt_replay": True,
        "original_aa_byte_identical_separate_request": True,
        "all_original_evidence_and_output_caps_retained": True,
        "answers_or_actor_outcomes_used_for_reconstruction": False,
    }


def sampling_configuration(config: dict) -> dict:
    """A disclosed metadata-only transport view for the unchanged existing sampler."""
    validate_config(config)
    return {
        "schema_version": SAMPLER_SCHEMA,
        "purpose": "SAMPLING_ONLY_NOT_A_MODEL_EXECUTION_CONFIGURATION",
        "routing_execution_schema": SCHEMA, "routing_execution_config_canonical_sha256": canonical_hash(config),
        "data": deepcopy(config["data"]), "scoring": deepcopy(config["scoring"]),
    }


def require_data_preparation_host(config: dict) -> None:
    expected = DATA_HOSTS[config["data"]["preparation_host"]]
    if socket.gethostname() != expected or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Data preparation requires its explicitly declared remote CPU host")


def prepare_data(config_path: Path, source_manifest: Path, origin: Path, output: Path) -> None:
    """Remote only. Run the registered schema directly; retain the older explicit adapter."""
    from scripts import prepare_compact_harness_training_pilot as sampler

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    require_data_preparation_host(config)
    sampler.require_remote()
    output.mkdir(parents=True, exist_ok=False)
    try:
        if config["data_preparation"] == DIRECT_SAMPLER:
            sampler.prepare(config_path, source_manifest, origin, output / "data")
            validate_prepared_data(config_path, output / "data")
            return
        view = output / "sampler-config.json"
        write_json(view, sampling_configuration(config))
        sampled = output / "sampler-output"
        sampler.prepare(view, source_manifest, origin, sampled)
        original = _checked(sampled / "manifest.json")
        _require(original["config_sha256"] == file_sha(view),
                 "Sampling-only manifest/config binding differs")
        destination = output / "data"
        destination.mkdir()
        for filename, expected in original["files_sha256"].items():
            _require(Path(filename).name == filename and file_sha(sampled / filename) == expected,
                     "The unchanged sampler output has a changed artifact")
            if filename != "config.json":
                shutil.copyfile(sampled / filename, destination / filename)
        shutil.copyfile(view, destination / "sampler-config.json")
        shutil.copyfile(sampled / "manifest.json", destination / "sampler-manifest.json")
        shutil.copyfile(config_path, destination / "config.json")
        manifest = {key: value for key, value in original.items() if key != "canonical_sha256"}
        manifest.update({
            "config_sha256": file_sha(config_path),
            "sampling_adapter": SAMPLER_ADAPTER,
            "unchanged_sampler_manifest_sha256": file_sha(sampled / "manifest.json"),
            "claim_boundary": RESULT_SCOPE,
            "files_sha256": {
                **original["files_sha256"], "config.json": file_sha(config_path),
                "sampler-config.json": file_sha(destination / "sampler-config.json"),
                "sampler-manifest.json": file_sha(destination / "sampler-manifest.json"),
            },
        })
        write_json(destination / "manifest.json", _signed(manifest))
        validate_prepared_data(config_path, destination)
    except Exception as error:
        write_json(output / "FAILURE.json", {"type": type(error).__name__, "message": str(error)})
        raise


def validate_prepared_data(config_path: Path, source: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    manifest = _checked(source / "manifest.json")
    _require(
        manifest["config_sha256"] == file_sha(config_path)
        and manifest["source_manifest_sha256"] == config["data"]["component_data_manifest_sha256"]
        and manifest["question_group_count"] == config["data"]["question_groups"]
        and 0 < manifest["task_count"] <= config["data"]["maximum_tasks"]
        and (config["data_preparation"] == DIRECT_SAMPLER
             or manifest.get("sampling_adapter") == SAMPLER_ADAPTER),
        "Prepared TRAIN-source sampling/execution configuration binding differs",
    )
    for filename, expected in manifest["files_sha256"].items():
        _require(Path(filename).name == filename and not (source / filename).is_symlink()
                 and file_sha(source / filename) == expected, "Prepared input hash differs")
    _require(file_sha(source / "config.json") == file_sha(config_path),
             "Prepared execution configuration bytes differ")
    _require(manifest["files_sha256"]["mrqa_official_eval.py"] == config["scoring"]["source_sha256"],
             "Prepared scorer bytes differ from the frozen official scorer")
    if config["data_preparation"] == SAMPLER_ADAPTER:
        original = _checked(source / "sampler-manifest.json")
        _require(
            file_sha(source / "sampler-manifest.json") == manifest["unchanged_sampler_manifest_sha256"]
            and original["config_sha256"] == file_sha(source / "sampler-config.json")
            and json.loads((source / "sampler-config.json").read_text()) == sampling_configuration(config)
            and all(manifest["files_sha256"][name] == expected
                    for name, expected in original["files_sha256"].items() if name != "config.json"),
            "The sampling-only adapter cannot change selected tasks, answers, or scorer bytes",
        )
    independence = manifest["independence"]
    _require(all(independence[name] is True for name in (
        "group_disjoint_from_component_training", "group_disjoint_from_component_calibration",
        "group_disjoint_from_full_official_development", "group_disjoint_from_prior_training",
    )) and independence["evaluation_answers_opened"] is False
             and independence["evaluation_responses_opened"] is False
             and independence["historical_freshness_certified"] is False,
             "Required disjoint-group evidence is missing; historical freshness remains unestablished")
    selection = json.loads((source / "selection-freeze.json").read_text())
    _require(
        selection["prior_training_exclusions"] == config["data"]["exclude_prior_training"]
        and selection["selection_salt"] == config["data"]["selection_salt"]
        and selection["accepted_answers_consulted_for_selection"] is False
        and selection["model_or_scorer_calls_before_freeze"] == 0,
        "Selection must be frozen before outcomes under the prospectively declared salt/exclusions",
    )
    items = [json.loads(line) for line in (source / "public.jsonl").read_text().splitlines()]
    _validate_items(items, config)
    _require(manifest["task_count"] == len(items)
             and sorted(selection["task_ids"]) == sorted(item["id"] for item in items)
             and sorted(selection["groups"]) == sorted({item["group"] for item in items}),
             "Whole-group selection and materialized public task inventory differ")
    return manifest


def copy_prepared_data(config_path: Path, source: Path, destination: Path) -> None:
    validate_prepared_data(config_path, source)
    shutil.copytree(source, destination)
    (destination / "answers.json").chmod(0o600)


def collect_run(experiment: Path) -> dict:
    """Audit completed request/score artifacts only; never load data, models, or a scorer."""
    prepared = _checked(experiment / "PREPARED.json")
    for name, field in (
        ("config.json", "config_sha256"), ("requests.json", "requests_sha256"),
        ("model-spec.json", "model_spec_sha256"),
    ):
        _require(file_sha(experiment / name) == prepared[field], "Prepared experiment binding differs")
    config = json.loads((experiment / "config.json").read_text())
    planned = json.loads((experiment / "requests.json").read_text())
    matrix = validate_matrix(planned, config)
    drained = _checked(experiment / "GENERATION_DRAINED.json")
    _require(
        drained["generation_complete"] is True and drained["server_completion_verified"] is True
        and drained["attempted"] == drained["completed"] == len(planned)
        and drained["unknown_cost_requests"] == drained["pending_http"] == 0
        and drained["failure"] is None,
        "A failed, incomplete, retried, or unknown-cost matrix cannot be collected as complete",
    )
    expected_calls = {cell["cell"] for cell in planned}
    _require({path.name for path in (experiment / "calls").iterdir()} == expected_calls,
             "Call artifacts must contain exactly the complete fresh eight-arm matrix")
    scores = {arm: json.loads((experiment / f"{arm}.scores.json").read_text()) for arm in ARMS}
    _validate_scores(scores, matrix["groups"], config)
    response_ids = set()
    for (task, arm), cell in matrix["cells"].items():
        directory = experiment / "calls" / cell["cell"]
        _require((directory / "request.body").read_bytes() == body_bytes(cell["body"]),
                 "Fresh-call wire request differs from the frozen body")
        intent = json.loads((directory / "intent.json").read_text())
        _require(intent["cell"] == cell["cell"] and intent["task"] == task and intent["arm"] == arm
                 and intent["body_sha256"] == cell["request_body_sha256"],
                 "Separate original/control request intents are required")
        transport = json.loads((directory / "transport.json").read_text())
        record = json.loads((directory / "record.json").read_text())
        raw = json.loads((directory / "response.body").read_text())
        row = scores[arm][task]
        _require(
            transport["status"] == 200 and valid_usage(raw) is not None
            and record["raw"] == raw and record["usage"] == valid_usage(raw)
            and record["task"] == task and record["arm"] == arm and record["group"] == cell["group"]
            and file_sha(directory / "response.body") == record["response_sha256"] == row["response_sha256"]
            and valid_usage(raw) == {
                "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"],
                "total_tokens": row["tokens"],
            } and raw["model"] == config["model"]["repository"],
            "Fresh provider transport, raw response, identity, or known actual usage differs",
        )
        if raw.get("id") is not None:
            _require(isinstance(raw["id"], str) and bool(raw["id"]) and raw["id"] not in response_ids,
                     "A provider response ID was reused across fresh calls")
            response_ids.add(raw["id"])
    actual = sum(row["tokens"] for rows in scores.values() for row in rows.values())
    _require(actual == drained["known_real_tokens"]
             and _integer(drained["synthetic_requests"], 0, 1)
             and _integer(drained["synthetic_tokens"], 0, config["model"]["context_length"])
             and actual + drained["synthetic_tokens"] <= config["budget"]["maximum_total_provider_tokens"],
             "Measured real and synthetic provider totals must remain complete and additive")
    methods = {
        arm: {
            "tasks": len(rows),
            "exact_match": 100 * math.fsum(row["exact_match"] for row in rows.values()) / len(rows),
            "f1": 100 * math.fsum(row["f1"] for row in rows.values()) / len(rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows.values()),
            "completion_tokens": sum(row["completion_tokens"] for row in rows.values()),
            "total_tokens": sum(row["tokens"] for row in rows.values()),
        } for arm, rows in scores.items()
    }
    bounds = diagnostic_bounds(scores, matrix["groups"], config)
    result = _checked(experiment / "RESULT.json")
    _require(result["config_sha256"] == file_sha(experiment / "config.json")
             and result["paper_admitted"] is False, "Completed result configuration/scope differs")
    for arm, method in methods.items():
        for key, value in method.items():
            observed = result["methods"][arm][key]
            _require(math.isclose(observed, value, rel_tol=0, abs_tol=1e-10)
                     if key in {"exact_match", "f1"} else observed == value,
                     "Stored official per-task scores and aggregate result differ")
    return _signed({
        "schema_version": SCHEMA, "status": "COMPLETE_CALL_ARTIFACTS_RECONCILED",
        "scope": RESULT_SCOPE, "config_sha256": file_sha(experiment / "config.json"),
        "result_sha256": file_sha(experiment / "RESULT.json"),
        "fresh_request_artifacts_verified": len(planned),
        "real_provider_tokens": actual, "synthetic_provider_tokens": drained["synthetic_tokens"],
        "all_in_acquisition_tokens": actual + drained["synthetic_tokens"],
        "selected_path_tokens_are_not_all_in_campaign_cost": True,
        "methods": methods, "bounds": bounds, "target_gate": target_gate(methods, config, bounds),
        **result_diagnostics(planned, scores, config),
        "official_scores_recomputed": False,
        "collection_scope": "Reconciles existing scored outputs/calls; not a substitute for scorer or GPU-lifecycle audit",
        "paper_admitted": False, "no_loss_claim_established": False,
    })
