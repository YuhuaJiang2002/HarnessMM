"""Prospective full-evidence QA quality/cost protocol, separate from the failed 10% study."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from phiagent.learning.qa_capability_routing import CAPABILITY_MODES, capability_request
from phiagent.learning.qa_harness_mixture import mixture_arm_prompt
from scripts.prepare_chisel_component_quality_data import canonical_hash
from scripts.qa_harness_mixture_protocol import (
    BASELINES, INCUMBENT, OVERLAY_FILES as LEGACY_OVERLAY_FILES,
    FrozenOriginalChisel, fixed_prompt_hashes as fixed_prompt_hashes, prompt_artifact_names,
    stage_incumbent_artifacts, validate_methods,
)
from scripts.run_chisel_component_quality import ProtocolError, build_messages, prompt_length


SCHEMA = "phiagent-qa-capability-cost/v2"
CORE_CONFIRMATION_SCHEMA = "phiagent-qa-capability-core/v3"
MIXTURE_ARMS = (
    "capability_routed", "core_only", "routed_without_core", "answer_only",
    "all_capabilities", "count_matched_random", "legacy_nine_all",
)
FACTORIAL = {
    "core_without_optional": ["core_only", "answer_only"],
    "core_with_optional": ["capability_routed", "routed_without_core"],
    "optional_without_core": ["routed_without_core", "answer_only"],
    "optional_with_core": ["capability_routed", "core_only"],
}
OVERLAY_FILES = (
    *LEGACY_OVERLAY_FILES,
    "phiagent/learning/qa_capability_routing.py",
    "scripts/qa_capability_cost_protocol.py",
    "scripts/prepare_qa_capability_cost_config.py",
    "tests/test_qa_capability_routing.py",
    "tests/test_qa_capability_cost_protocol.py",
)


def validate_config(config: dict) -> None:
    expected = [*BASELINES, INCUMBENT, *MIXTURE_ARMS]
    primary = "core_only" if config.get("schema_version") == CORE_CONFIRMATION_SCHEMA else "capability_routed"
    if (
        config.get("schema_version") not in {SCHEMA, CORE_CONFIRMATION_SCHEMA}
        or config["incumbent"]["status"] != "BOUND"
        or config["incumbent"]["arm"] != INCUMBENT
        or config["arms"] != expected or config["comparison_arms"] != expected
        or config["candidate_arms"] != list(MIXTURE_ARMS)
        or config["primary_candidate"] != primary
        or config["primary_reference"] != INCUMBENT
        or config["cost_reference"] != "autosaddler"
        or config["selection_eligible_arms"] != [primary]
        or config["factorial"]["contrasts"] != FACTORIAL
        or config.get("mixture_router") is not None
        or any(config.get(key) for key in ("context_selection", "rule_compression", "length_matched_controls"))
        or config["capability_routing"] != {
            "enabled": True, "optional_top_k": 2, "preserve_full_context": True,
            "preserve_output_cap": True, "random_optional_count_matched_per_task": True,
        }
        or config["model"]["temperature"] != 0 or config["model"]["enable_thinking"] is not False
        or config["budget"]["transport_retries"] != 0 or config["paper_admitted"] is not False
    ):
        raise ProtocolError("The prospective full-evidence capability/cost contract changed")
    statistical_plan(config)


def statistical_plan(config: dict) -> dict:
    candidate = config["primary_candidate"]
    references = [arm for arm in config["arms"] if arm != candidate]
    settings = config["statistics"]
    if (
        settings["primary_references"] != references
        or settings["primary_metrics"] != ["exact_match", "f1", "tokens"]
        or settings["factorial_quality_metrics"] != ["exact_match", "f1"]
        or settings["familywise_alpha"] != 0.05 or settings["noninferiority_margin"] != 0
    ):
        raise ProtocolError("The full simultaneous quality/cost family must be frozen")
    pairs = [(candidate, reference, metric) for reference in references
             for metric in settings["primary_metrics"]]
    pairs += [(left, right, metric) for left, right in FACTORIAL.values()
              for metric in settings["factorial_quality_metrics"]]
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) != settings["simultaneous_bounds"]:
        raise ProtocolError("Declared simultaneous endpoints do not match the complete family")
    return {"contrasts": pairs, "family_size": len(pairs), "alpha": 0.05, "reference": INCUMBENT}


def stage_incumbent(source: Path, config: dict) -> list[str]:
    validate_config(config)
    return stage_incumbent_artifacts(source, config)


def prepare_cells(tokenizer, items: list[dict], prompts: dict, config: dict, source: Path):
    validate_config(config)
    return prepare_comparison_cells(tokenizer, items, prompts, config, source)


def prepare_comparison_cells(tokenizer, items: list[dict], prompts: dict, config: dict, source: Path):
    """Materialize the unchanged twelve full-context reference methods."""
    arms = (*BASELINES, INCUMBENT, *MIXTURE_ARMS)
    if (
        not items or len({item["id"] for item in items}) != len(items)
        or any(set(item) != {"id", "group", "question", "context"} for item in items)
        or set(MIXTURE_ARMS) != set(CAPABILITY_MODES)
        or any(prompts[arm] != mixture_arm_prompt() for arm in MIXTURE_ARMS)
    ):
        raise ProtocolError("Complete unique public inputs and declared candidate artifacts are required")
    incumbent = FrozenOriginalChisel(source, config)

    def count(system, question, context):
        return prompt_length(tokenizer, build_messages(system, {"question": question, "context": context}))

    planned = []
    for item in items:
        route = incumbent.route(item)
        common = {
            "question": item["question"], "context": item["context"],
            "no_skill_system": prompts["no_skill"], "delete_system": prompts["delete"],
            "seed": config["model"]["request_seed"],
            "max_output_tokens": config["model"]["max_output_tokens"],
            "optional_top_k": config["capability_routing"]["optional_top_k"],
            "count_prompt": count,
        }
        primary = capability_request(mode="capability_routed", **common)
        for arm in arms:
            receipt = None
            if arm in MIXTURE_ARMS:
                request = primary if arm == "capability_routed" else capability_request(
                    mode=arm,
                    **({"random_optional_count": primary.receipt["selected_optional_count"]}
                       if arm == "count_matched_random" else {}),
                    **common,
                )
                system, receipt = request.system_prompt, request.receipt
                if request.context != item["context"]:
                    raise ProtocolError("Source-preserving candidates cannot remove evidence")
            else:
                system = prompts[route["variant"]] if arm == INCUMBENT else prompts[arm]
            messages = build_messages(system, item)
            prompt_tokens = prompt_length(tokenizer, messages)
            cap = config["model"]["max_output_tokens"]
            if prompt_tokens + cap > config["model"]["context_length"]:
                raise ProtocolError("The full, untruncated request exceeds model context")
            planned.append({
                "cell": canonical_hash([item["id"], arm, config["model"]["request_seed"]]),
                "task": item["id"], "group": item["group"], "arm": arm,
                "expected_prompt_tokens": prompt_tokens, "max_output_tokens": cap,
                "reference_arm": INCUMBENT, "incumbent_route": route,
                **({"mixture_receipt": receipt} if receipt is not None else {}),
                "body": {
                    "model": config["model"]["repository"], "messages": messages,
                    "temperature": 0, "max_tokens": cap, "seed": config["model"]["request_seed"],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            })
    planned.sort(key=lambda cell: cell["cell"])
    summary = {
        "status": "FULL_EVIDENCE_CAPABILITY_ROUTES_FROZEN_BEFORE_OUTCOMES",
        "primary_baseline_status": "BOUND", "reference_arm": INCUMBENT,
        "cost_reference": "autosaddler", "unique_context_transforms": 0,
        "router_model_calls": 0, "router_provider_tokens": 0,
        "selection_uses_answers_or_actor_responses": False,
        "lexical_signals_are_quality_certificates": False,
        "methods": {
            arm: {
                "tasks": len(items),
                "prompt_tokens": sum(cell["expected_prompt_tokens"] for cell in planned if cell["arm"] == arm),
                "reserved_completion_tokens": len(items) * config["model"]["max_output_tokens"],
                "selected_harnesses": dict(Counter(
                    cell["mixture_receipt"]["selected_harness"] for cell in planned
                    if cell["arm"] == arm and "mixture_receipt" in cell
                )),
                "incumbent_fallbacks": 0,
            }
            for arm in arms
        },
    }
    return planned, summary


def output_cap(cell: dict, config: dict) -> int:
    cap = config["model"]["max_output_tokens"]
    if cell["max_output_tokens"] != cap or cell["body"]["max_tokens"] != cap:
        raise ProtocolError("All arms must retain the same original output cap")
    if cell["arm"] in MIXTURE_ARMS and cell["mixture_receipt"]["max_output_tokens"] != cap:
        raise ProtocolError("The capability receipt has a different output cap")
    return cap


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    validate_methods(methods, config)
    candidate, cost_reference = config["primary_candidate"], config["cost_reference"]
    references = [arm for arm in config["comparison_arms"] if arm != candidate]
    quality = all(methods[candidate][metric] > methods[arm][metric]
                  for arm in references for metric in ("exact_match", "f1"))
    cost = methods[candidate]["total_tokens"] < methods[cost_reference]["total_tokens"]
    point = quality and cost
    endpoints = [f"{candidate}_vs_{arm}_{metric}"
                 for arm in references for metric in ("exact_match", "f1")]
    endpoints.append(f"{candidate}_vs_{cost_reference}_tokens")
    statistical = None if bounds is None else point and all(
        bounds[key]["one_sided_lower_bound"] > 0 for key in endpoints
    )
    return {
        "status": "DUAL_QUALITY_COST_POINT_PASS" if point else "DUAL_QUALITY_COST_POINT_FAIL",
        "candidate": candidate, "primary_reference": INCUMBENT, "primary_baseline_status": "BOUND",
        "cost_reference": cost_reference,
        "observed_total_token_ratio": methods[candidate]["total_tokens"] / methods[cost_reference]["total_tokens"],
        "quality_strictly_above_every_compared_method": quality,
        "tokens_strictly_below_autosaddler": cost,
        "point_target_passed": point, "statistical_target_passed": statistical,
        "metric_differences_pp": {
            metric: methods[candidate][metric] - methods[INCUMBENT][metric]
            for metric in ("exact_match", "f1")
        },
        "all_quality_references": references,
        "statistical_bound_conditions_established": False,
        "automatic_official_scale_up_permitted": False, "paper_admitted": False,
        "production_defaults_changed": False,
        "comparison_scope": "Disjoint official-training development; no full-benchmark claim",
    }


def diagnostic_bounds(scores: dict, groups: dict[str, str], config: dict) -> dict:
    from phiagent.evaluation.chisel_component_quality import paired_finite_sample_lower_bound

    if set(scores) != set(config["arms"]) or any(set(rows) != set(groups) for rows in scores.values()):
        raise ProtocolError("Every paired task and declared method is required")
    seed = config["model"]["request_seed"]
    plan = statistical_plan(config)
    result = {}
    for left, right, metric in plan["contrasts"]:
        lvalues = {task: row[metric] for task, row in scores[left].items()}
        rvalues = {task: row[metric] for task, row in scores[right].items()}
        if metric == "tokens":
            lvalues, rvalues = rvalues, lvalues
        result[f"{left}_vs_{right}_{metric}"] = paired_finite_sample_lower_bound(
            {seed: lvalues}, {seed: rvalues}, groups, seeds=(seed,),
            score_upper_bound=config["model"]["context_length"] if metric == "tokens" else 1,
            family_size=plan["family_size"], alpha=plan["alpha"], binary=metric == "exact_match",
        )
    return result


def replay_matrix(tokenizer, public, planned, source, config):
    prompts = {arm: (source / f"prompts/{arm}.system.txt").read_text()
               for arm in prompt_artifact_names(config)}
    rebuilt, summary = prepare_cells(tokenizer, public, prompts, config, source)
    if rebuilt != planned:
        raise RuntimeError("Full-evidence capability request replay differs from frozen bytes")
    by_pair = {(cell["task"], cell["arm"]): cell for cell in rebuilt}
    for item in public:
        primary = by_pair[item["id"], "capability_routed"]["mixture_receipt"]
        random = by_pair[item["id"], "count_matched_random"]["mixture_receipt"]
        if primary["selected_optional_count"] != random["selected_optional_count"]:
            raise RuntimeError("Random optional-skill counts are not exactly matched per input")
    return {
        "summary": summary, "complete_request_and_receipt_replay": True,
        "random_optional_skill_counts_matched_per_task": True,
        "all_original_evidence_retained": True, "all_original_output_caps_retained": True,
        "answers_or_actor_outcomes_used_for_reconstruction": False,
    }
