"""Source-preserving DELETE/MODIFY normalization/common-block compression study."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import socket

from phiagent.learning import qa_profile_compression as design
from scripts.prepare_chisel_component_quality_data import canonical_hash
from scripts.qa_harness_mixture_protocol import (
    FrozenOriginalChisel, OVERLAY_FILES as INCUMBENT_FILES,
    stage_incumbent_artifacts, validate_methods,
)
from scripts.run_chisel_component_quality import ProtocolError, build_messages, prompt_length


SCHEMA = "phiagent-qa-profile-compression/v1"
PRIORITY_SCHEMA = "phiagent-qa-profile-compression/v2"
PRIMARY = "compact_both"
PRIORITY_PRIMARY = "compact_priority"
ARMS = (
    "original_chisel", "original_aa", "delete", "modify_fixed", "skillopt", "autosaddler",
    "answer_only", "compact_normalization", "compact_common", PRIMARY, "compact_aa",
    "compact_delete", "compact_modify",
)
PRIORITY_ARMS = (
    "original_chisel", "delete", "modify_fixed", "skillopt", "autosaddler", "answer_only",
    "compact_legacy", PRIORITY_PRIMARY, "compact_priority_without_guard", "compact_priority_aa",
)
PROFILES = ("no_skill", "skillopt", "autosaddler", "delete", "modify")
QUALITY_REFERENCES = tuple(arm for arm in ARMS if arm not in {PRIMARY, "original_aa", "compact_aa"})
COST_REFERENCES = ("original_chisel", "delete", "autosaddler")
FACTORIAL = (
    ("compact_normalization", "original_chisel"),
    ("compact_common", "original_chisel"),
    (PRIMARY, "compact_normalization"), (PRIMARY, "compact_common"),
    ("original_chisel", "original_aa"), (PRIMARY, "compact_aa"),
)
RESULT_STATUS = "COMPLETE_PROFILE_COMPRESSION_TRAIN_DIAGNOSTIC"
DEVELOPMENT_STATUS = "SOURCE_COMPRESSION_SCREEN_COMPLETE_NO_AUTOMATIC_PROMOTION"
RESULT_SCOPE = "Prospectively frozen TRAIN-source profile compression; not full benchmark, independent confirmation or a no-loss certificate."
RECORD_CLIENT_TIMING = True
OVERLAY_FILES = tuple(dict.fromkeys((
    *INCUMBENT_FILES,
    "phiagent/learning/qa_profile_compression.py",
    "scripts/qa_profile_compression_protocol.py",
    "scripts/prepare_qa_profile_compression.py",
    "tests/test_qa_profile_compression.py",
    "scripts/run_qa_lossless_guarded.py",
    "scripts/cvpr_execution_crossover_leases.py",
    "scripts/prepare_qa_lossless_native.py",
    "scripts/verify_cvpr_qwen35_cache.py",
    "tests/test_qa_lossless_guarded.py",
    "scripts/verify_qa_harness_mixture.py",
)))
PINNED_MODEL = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
COMPRESSION = {
    "normalization_component": "Compress source-bound profile-specific normalization, including every MODIFY exception.",
    "common_component": "Consolidate shared evidence/type/snippet/trap rules and redundant role/section preambles.",
    "full_source_context_preserved": True,
    "question_and_answer_contract_preserved": True,
    "head_features_threshold_and_route_unchanged": True,
    "maximum_output_tokens_unchanged": 512,
    "minimum_harness_marginal_saving_fraction": 0.60,
    "full_candidate_input_below_autosaddler_each_task": True,
    "native_token_saving_is_not_quality_equivalence": True,
}
PRIORITY_COMPRESSION = {
    **COMPRESSION,
    "priority": "Resolve evidence/target identity before shortening; explicit source forms override defaults.",
    "development_informed_successor": True,
    "legacy_combined_compression_remains_a_fresh_control": True,
    "new_disjoint_development_cohort_required": True,
}


def is_priority(config: dict) -> bool:
    return config["schema_version"] == PRIORITY_SCHEMA


def arms_for(config: dict) -> tuple[str, ...]:
    return PRIORITY_ARMS if is_priority(config) else ARMS


def primary_for(config: dict) -> str:
    return PRIORITY_PRIMARY if is_priority(config) else PRIMARY


def quality_references(config: dict) -> tuple[str, ...]:
    return tuple(arm for arm in arms_for(config)
                 if arm not in {primary_for(config), "original_aa", "compact_aa", "compact_priority_aa"})


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProtocolError(reason)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_prompt_hashes(config: dict) -> dict:
    return {name: config["prompt_sha256"][name] for name in PROFILES}


def prompt_artifact_names(_config: dict) -> list[str]:
    return list(PROFILES)


def contrasts(config: dict | None = None) -> list[tuple[str, str, str]]:
    if config is not None and is_priority(config):
        return list(dict.fromkeys(
            [(PRIORITY_PRIMARY, arm, metric) for arm in quality_references(config)
             for metric in ("exact_match", "f1")]
            + [(PRIORITY_PRIMARY, arm, "tokens") for arm in COST_REFERENCES]
            + [(PRIORITY_PRIMARY, arm, metric)
               for arm in ("compact_legacy", "compact_priority_without_guard", "compact_priority_aa")
               for metric in ("exact_match", "f1", "tokens")]
        ))
    return list(dict.fromkeys(
        [(PRIMARY, arm, metric) for arm in QUALITY_REFERENCES for metric in ("exact_match", "f1")]
        + [(PRIMARY, arm, "tokens") for arm in COST_REFERENCES]
        + [(left, right, metric) for left, right in FACTORIAL
           for metric in ("exact_match", "f1", "tokens")]
    ))


def statistical_plan(config: dict) -> dict:
    family = contrasts(config)
    return {"contrasts": [list(row) for row in family], "family_size": 2 * len(family), "alpha": 0.05,
            "reference": "original_chisel"}


def validate_config(config: dict) -> None:
    try:
        priority = is_priority(config)
        arms = arms_for(config)
        require(
            config["schema_version"] in {SCHEMA, PRIORITY_SCHEMA} and config["arms"] == list(arms)
            and config["comparison_arms"] == list(arms)
            and config["candidate_arms"] == config["selection_eligible_arms"] == []
            and config["primary_candidate"] == primary_for(config)
            and config["primary_reference"] == "original_chisel"
            and config["cost_reference"] == "autosaddler"
            and config["compression"] == (PRIORITY_COMPRESSION if priority else COMPRESSION)
            and config["statistics"] == statistical_plan(config)
            and config["quality_margin_pp"] == 0 and config["paper_admitted"] is False,
            "The frozen component factorial, zero-loss criterion and claim boundary changed",
        )
        require(config["incumbent"]["status"] == "BOUND"
                and config["incumbent"]["arm"] == "original_chisel"
                and canonical_hash(config["incumbent"]) == config["frozen_incumbent_sha256"],
                "The original head/features/threshold and source artifacts cannot change")
        require(all(config["prompt_sha256"][name] == expected
                    for name, expected in design.SOURCE_HASHES.items()),
                "Both original profile byte identities are required")
        model = config["model"]
        require((model["repository"], model["revision"]) == PINNED_MODEL
                and model["max_output_tokens"] == 512 and model["context_length"] == 32768
                and model["temperature"] == 0 and model["enable_thinking"] is False
                and type(model["request_seed"]) is int,
                "The common model, unchanged output cap and deterministic decoding are frozen")
        budget, data, resource = config["budget"], config["data"], config["resource"]
        require(data["mode"] == "disjoint_prepared_training"
                and data["question_groups"] == 256 and data["maximum_tasks"] == 320
                and data["preparation_host"] == "h200-1"
                and data["selection_uses_answers"] is False
                and data["historical_freshness_certified"] is False
                and data["preserve_whole_question_groups"] is True
                and data["preserve_full_original_context_and_question_in_source_artifacts"] is True
                and all(isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                        for row in data["exclude_prior_training"]),
                "Exact whole-group development and prior-group exclusion boundaries are required")
        require(budget["maximum_physical_gpus"] == 1
                and data["maximum_tasks"] * len(arms) <= budget["maximum_real_request_attempts"] <= 5000
                and budget["maximum_synthetic_request_attempts"] == 1
                and budget["transport_retries"] == 0
                and 0 < budget["maximum_total_provider_tokens"] <= 20000000
                and 180 < budget["maximum_allocation_seconds"] <= (3500 if priority else 5400)
                and 1 <= budget["execution_workers"] <= budget["maximum_inflight_requests"] <= 16,
                "The new bounded no-retry development budget changed")
        require(resource["binding_status"] == "BOUND_AFTER_FRESH_AUDIT"
                and resource["selected_host_alias"] == "h200-1"
                and resource["selected_hostname"] == "harnessmm-worker-1"
                and resource["physical_gpu_index"] == 3
                and resource["gpu_uuid"] == "GPU-6963b4db-f9a5-8be6-941c-90e28b372751"
                and resource["fresh_GPU_and_exclusive_lease_required_before_launch"] is True,
                "The exact authorized H200 placement and live guard requirement must be explicit")
        require(config["scoring"]["source_sha256"]
                == "02a7855398e7133952ee374582fcc90a8b6dcf6216e0759347f29487074ee992"
                and config["scoring"]["metrics"] == ["exact_match", "f1"],
                "The official evaluation metric cannot change")
        if priority:
            require(config["parent_development"]["result_sha256"]
                    and config["parent_development"]["audit_sha256"]
                    and config["parent_development"]["old_point_gate_remains_failed"] is True
                    and config["parent_development"]["response_reuse_permitted"] is False
                    and config["campaign_limits"]["maximum_cumulative_gpu_seconds"] == 7200
                    and config["campaign_limits"]["maximum_all_provider_requests"] == 9000
                    and config["campaign_limits"]["maximum_actual_provider_tokens"] == 30000000,
                    "The failed predecessor, new responses and unchanged cumulative limits are required")
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(f"Incomplete profile-compression protocol: {error}") from error


def source_profiles(source: Path, config: dict) -> dict[str, str]:
    result = {}
    for name, digest in fixed_prompt_hashes(config).items():
        path = source / "prompts" / f"{name}.system.txt"
        require(file_sha(path) == digest, "An archived baseline prompt changed")
        result[name] = path.read_text()
    return result


def verify_predecessor(config: dict) -> dict:
    if not is_priority(config):
        return {"prior_debits": None}
    require(socket.gethostname() == "harnessmm-worker-1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Predecessor result/accounting checks require authorized remote CPU")
    binding = config["parent_development"]
    records = {}
    for name in ("result", "audit", "progress", "retirement"):
        path = Path(binding[f"{name}_path"])
        require(file_sha(path) == binding[f"{name}_sha256"], f"Predecessor {name} bytes changed")
        records[name] = json.loads(path.read_text())
    previous, audit, progress, retirement = (records[name] for name in (
        "result", "audit", "progress", "retirement",
    ))
    require(audit["status"] == "INDEPENDENT_HARNESS_MIXTURE_REPLAY_VERIFIED"
            and audit["result_sha256"] == binding["result_sha256"]
            and previous["target_gate"]["point_target_passed"] is False
            and progress["attempted"] == progress["completed"] == 3328
            and progress["unknown_cost_requests"] == progress["pending_http"] == 0
            and retirement["status"] == "RETIRED"
            and retirement["physical_gpu_verified_idle_before_unlock"] is True,
            "The full failed predecessor and safe resource retirement must be retained")
    debits = {
        "calls": progress["attempted"] + progress["synthetic_requests"],
        "tokens": progress["known_real_tokens"] + progress["synthetic_tokens"],
        "gpu_seconds": retirement["allocation_seconds"],
    }
    require(debits == binding["all_in_debits"], "Prior/failure costs cannot be reset")
    limits, budget = config["campaign_limits"], config["budget"]
    require(
        debits["calls"] + budget["maximum_real_request_attempts"] + 1 <= limits["maximum_all_provider_requests"]
        and debits["tokens"] + budget["maximum_total_provider_tokens"] <= limits["maximum_actual_provider_tokens"]
        and debits["gpu_seconds"] + budget["maximum_allocation_seconds"] <= limits["maximum_cumulative_gpu_seconds"],
        "The successor exceeds the original campaign's remaining all-in budget",
    )
    return {"prior_debits": debits, "previous_gate_remains_failed": True}


def stage_incumbent(source: Path, config: dict) -> list[str]:
    validate_config(config)
    verify_predecessor(config)
    files = stage_incumbent_artifacts(source, config)
    profiles = source_profiles(source, config)
    compiled, mapping = profile_texts(profiles, config)
    artifact = source / "PROFILE_COMPRESSION_MAP.json"
    with artifact.open("x") as stream:
        json.dump(mapping, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    for variant, members in compiled.items():
        for profile, text in members.items():
            path = source / "compressed-profiles" / f"{variant}-{profile}.txt"
            path.parent.mkdir(exist_ok=True)
            with path.open("x") as stream:
                stream.write(text)
            files.append(str(path.relative_to(source)))
    FrozenOriginalChisel(source, config)
    return [*files, "PROFILE_COMPRESSION_MAP.json"]


def profile_texts(prompts: dict, config: dict | None = None) -> tuple[dict, dict]:
    originals = {name: prompts[name] for name in design.SOURCE_HASHES}
    compiled, mapping = design.compile_profiles(originals)
    if config is not None and is_priority(config):
        successor, mapping = design.compile_priority_profiles(originals)
        return {"both": compiled["both"], **successor}, mapping
    return compiled, mapping


def system_for(arm: str, route: str, prompts: dict, compressed: dict) -> str:
    if arm in {"compact_priority", "compact_priority_aa"}:
        return compressed["priority"][route]
    if arm == "compact_priority_without_guard":
        return compressed["without_priority"][route]
    if arm == "compact_legacy":
        return compressed["both"][route]
    if arm in {"original_chisel", "original_aa"}:
        return prompts[route]
    if arm in {"delete", "skillopt", "autosaddler"}:
        return prompts[arm]
    if arm == "modify_fixed":
        return prompts["modify"]
    if arm == "answer_only":
        return design.answer_only_prompt()
    if arm in {"compact_both", "compact_aa", "compact_delete", "compact_modify"}:
        profile = arm.removeprefix("compact_") if arm in {"compact_delete", "compact_modify"} else route
        return compressed["both"][profile]
    if arm in {"compact_normalization", "compact_common"}:
        return compressed[arm.removeprefix("compact_")][route]
    raise ProtocolError("Undeclared compression arm")


def prepare_cells(tokenizer, items: list[dict], prompts: dict, config: dict, source: Path):
    validate_config(config)
    verify_predecessor(config)
    require(items and len(items) <= config["data"]["maximum_tasks"]
            and len({item["id"] for item in items}) == len(items)
            and len({item["group"] for item in items}) == config["data"]["question_groups"]
            and all(set(item) == {"id", "group", "question", "context"} for item in items),
            "Every unchanged source question/context and all whole groups are required")
    require(set(prompts) == set(PROFILES) and all(
        design.text_sha(prompts[name]) == digest for name, digest in fixed_prompt_hashes(config).items()
    ), "Only exact archived prompts may define the unmodified comparison arms")
    compressed, mapping = profile_texts(prompts, config)
    primary, arms = primary_for(config), arms_for(config)
    router = FrozenOriginalChisel(source, config)
    planned = []
    for item in sorted(items, key=lambda value: value["id"]):
        route = router.route(item)
        require(route["variant"] in design.SOURCE_HASHES and route["refit"] is False
                and route["answers_or_actor_responses_used"] is False
                and route["threshold"] == config["incumbent"]["historical_display"]["score_threshold"],
                "Routing is fixed from original inputs, not trained or threshold-tuned")
        for arm in arms:
            system = system_for(arm, route["variant"], prompts, compressed)
            body = {
                "model": config["model"]["repository"], "messages": build_messages(system, item),
                "temperature": 0, "max_tokens": 512, "seed": config["model"]["request_seed"],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            body = json.loads(json.dumps(body, sort_keys=True, ensure_ascii=False))
            cell = {
                "cell": canonical_hash([item["id"], arm, config["model"]["request_seed"]]),
                "task": item["id"], "group": item["group"], "arm": arm, "body": body,
                "expected_prompt_tokens": prompt_length(tokenizer, body["messages"]),
                "max_output_tokens": 512, "reference_arm": "original_chisel",
                "incumbent_route": deepcopy(route),
                "compression_receipt": {
                    "source_profile": route["variant"], "system_sha256": design.text_sha(system),
                    "question_sha256": design.text_sha(item["question"]),
                    "context_sha256": design.text_sha(item["context"]),
                    "map_canonical_sha256": canonical_hash(mapping),
                    "source_clause_coverage_does_not_certify_no_loss": True,
                },
            }
            output_cap(cell, config)
            planned.append(cell)
    planned.sort(key=lambda cell: cell["cell"])
    counts = {(cell["task"], cell["arm"]): cell["expected_prompt_tokens"] for cell in planned}
    both_savings, marginal, auto_savings = [], [], []
    by_pair = {(cell["task"], cell["arm"]): cell for cell in planned}
    for item in items:
        task = item["id"]
        original, both, short = (counts[task, arm] for arm in ("original_chisel", primary, "answer_only"))
        require(original > short and both < original, "The full native request was not compressed")
        both_savings.append(original - both)
        marginal.append((original - both) / (original - short))
        auto_savings.append(counts[task, "autosaddler"] - both)
        aa = "compact_priority_aa" if is_priority(config) else "compact_aa"
        require(by_pair[task, primary]["body"] == by_pair[task, aa]["body"],
                "Candidate A/A requests must be byte-identical but independently executed")
        if not is_priority(config):
            require(by_pair[task, "original_chisel"]["body"] == by_pair[task, "original_aa"]["body"],
                    "Original A/A requests must remain byte-identical")
    require(min(marginal) >= COMPRESSION["minimum_harness_marginal_saving_fraction"]
            and min(auto_savings) > 0, "Compression cannot clear the native input-cost floor")
    summary = {
        "status": "SOURCE_BOUND_PROFILE_AND_COMMON_COMPRESSION_FROZEN",
        "tasks": len(items), "groups": len({item["group"] for item in items}),
        "requests": len(planned), "map_canonical_sha256": canonical_hash(mapping),
        "minimum_native_input_tokens_saved": min(both_savings),
        "minimum_harness_marginal_saving_fraction": min(marginal),
        "minimum_native_input_tokens_below_autosaddler": min(auto_savings),
        "native_input_tokens_by_arm": {
            arm: sum(counts[item["id"], arm] for item in items) for arm in arms
        },
        "frozen_route_counts": dict(Counter(router_cell["incumbent_route"]["variant"]
                                           for router_cell in planned if router_cell["arm"] == primary)),
        "source_context_changed": False, "output_caps_changed": False, "router_refit": False,
        "actual_usage_or_quality_not_yet_measured": True,
    }
    return planned, summary


def output_cap(cell: dict, config: dict) -> int:
    body = cell["body"]
    require(cell["arm"] in arms_for(config) and cell["max_output_tokens"] == body["max_tokens"] == 512
            and body["model"] == config["model"]["repository"]
            and body["temperature"] == 0 and body["seed"] == config["model"]["request_seed"]
            and body["chat_template_kwargs"] == {"enable_thinking": False}
            and type(cell["expected_prompt_tokens"]) is int
            and 0 < cell["expected_prompt_tokens"] <= config["model"]["context_length"] - 512,
            "A changed cap, model or truncated/oversized native input cannot enter this comparison")
    return 512


def execution_batches(cells: list[dict], config: dict) -> list[list[dict]]:
    ordered = sorted(cells, key=lambda cell: canonical_hash([config["campaign_id"], cell["cell"]]))
    workers = config["budget"]["execution_workers"]
    return [ordered[start:start + workers] for start in range(0, len(ordered), workers)]


def attest_runtime(binding: dict, config: dict) -> dict:
    validate_config(config)
    require(binding["hostname"] == config["resource"]["selected_hostname"]
            and binding["physical_gpu_index"] == config["resource"]["physical_gpu_index"]
            and binding["gpu_uuid"] == config["resource"]["gpu_uuid"]
            and binding["model_repository"] == config["model"]["repository"]
            and binding["revision"] == config["model"]["revision"],
            "Live native runtime must match the frozen model and exact physical GPU")
    return {"status": "FROZEN_NATIVE_IDENTITY_AND_CUMULATIVE_DEBITS_VERIFIED",
            **verify_predecessor(config)}


def diagnostic_bounds(scores: dict, groups: dict, config: dict) -> dict:
    from phiagent.evaluation.chisel_component_quality import paired_finite_sample_lower_bound

    validate_config(config)
    require(set(scores) == set(arms_for(config)) and all(set(rows) == set(groups) for rows in scores.values())
            and len(set(groups.values())) == config["data"]["question_groups"],
            "Every paired task/arm is needed for whole-group uncertainty")
    for rows in scores.values():
        for row in rows.values():
            require(type(row["exact_match"]) is int and row["exact_match"] in (0, 1)
                    and math.isfinite(row["f1"]) and 0 <= row["f1"] <= 1
                    and type(row["tokens"]) is int and row["tokens"] > 0,
                    "Unknown or invalid official quality/token observations")
    result = {}
    plan, seed = statistical_plan(config), config["model"]["request_seed"]
    for left, right, metric in plan["contrasts"]:
        lhs = {task: row[metric] for task, row in scores[left].items()}
        rhs = {task: row[metric] for task, row in scores[right].items()}
        if metric == "tokens":
            lhs, rhs = rhs, lhs
        arguments = {
            "groups": groups, "seeds": (seed,), "binary": metric == "exact_match",
            "score_upper_bound": config["model"]["context_length"] if metric == "tokens" else 1,
            "family_size": plan["family_size"], "alpha": 0.05,
        }
        lower = paired_finite_sample_lower_bound({seed: lhs}, {seed: rhs}, **arguments)
        reverse = paired_finite_sample_lower_bound({seed: rhs}, {seed: lhs}, **arguments)
        result[f"{left}_vs_{right}_{metric}"] = {
            **lower, "simultaneous_upper_bound": -reverse["one_sided_lower_bound"],
            "zero_difference_is_not_equivalence": True, "population_no_loss_claim": False,
        }
    return result


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    validate_config(config)
    validate_methods(methods, config)
    primary, references = primary_for(config), quality_references(config)
    candidate = methods[primary]
    quality = all(candidate[metric] >= methods[arm][metric]
                  for arm in references for metric in ("exact_match", "f1"))
    cost = all(candidate["total_tokens"] < methods[arm]["total_tokens"] for arm in COST_REFERENCES)
    numerical = None
    if bounds is not None:
        expected = {f"{a}_vs_{b}_{metric}" for a, b, metric in contrasts(config)}
        require(set(bounds) == expected, "No primary or ablation endpoint may be omitted")
        numerical = quality and cost and all(
            bounds[f"{primary}_vs_{arm}_{metric}"]["one_sided_lower_bound"] >= 0
            for arm in references for metric in ("exact_match", "f1")
        ) and all(bounds[f"{primary}_vs_{arm}_tokens"]["one_sided_lower_bound"] > 0
                  for arm in COST_REFERENCES)
    return {
        "status": "ZERO_LOSS_ACTUAL_COST_POINT_PASS" if quality and cost else "ZERO_LOSS_ACTUAL_COST_POINT_FAIL",
        "candidate": primary, "primary_reference": "original_chisel", "primary_baseline_status": "BOUND",
        "quality_margin_pp": 0, "quality_references": list(references),
        "cost_references": list(COST_REFERENCES),
        "observed_no_quality_damage_against_all_controls": quality,
        "observed_actual_tokens_saved_against_all_cost_references": cost,
        "point_target_passed": quality and cost, "simultaneous_numerical_target_passed": numerical,
        "statistical_target_passed": False, "statistical_bound_conditions_established": False,
        "paper_admitted": False, "automatic_official_scale_up_permitted": False,
        "production_defaults_changed": False, "independent_confirmation": False,
        "source_coverage_is_not_quality_equivalence": True,
    }


def result_diagnostics(planned: list[dict], scores: dict, config: dict) -> dict:
    return {"profile_compression": {
        "design": config["compression"], "no_loss_claim_established": False,
        "actual_tokens": {
            arm: {
                "input": sum(row["prompt_tokens"] for row in scores[arm].values()),
                "output": sum(row["completion_tokens"] for row in scores[arm].values()),
                "length_stops": sum(row["finish_reason"] == "length" for row in scores[arm].values()),
            } for arm in arms_for(config)
        },
        "native_full_input_counts_match_actual": all(
            cell["expected_prompt_tokens"] == scores[cell["arm"]][cell["task"]]["prompt_tokens"]
            for cell in planned
        ),
        "fresh_calls": len(planned),
    }}


def replay_matrix(tokenizer, public: list[dict], planned: list[dict], source: Path, config: dict) -> dict:
    rebuilt, summary = prepare_cells(tokenizer, public, source_profiles(source, config), config, source)
    require(rebuilt == planned, "Source-bound profile compression/raw request replay differs")
    return {"status": "SOURCE_BOUND_PROFILE_COMPRESSION_REPLAY_VERIFIED",
            "summary": summary, "requests": len(rebuilt)}


def copy_prepared_data(config_path: Path, source: Path, destination: Path) -> None:
    from scripts.run_compact_harness_refinement import copy_prepared_training

    copy_prepared_training(config_path, source, destination)
