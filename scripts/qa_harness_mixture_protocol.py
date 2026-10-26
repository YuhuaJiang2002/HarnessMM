"""Opt-in mixture protocol, frozen incumbent adapter, and matched-cohort gates.

These helpers do not load datasets, models, or scorers. Benchmark execution stays
in the existing remote pilot; receipts are reconstructed from public inputs only.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

from scripts.prepare_chisel_component_quality_data import canonical_hash
from scripts.run_chisel_component_quality import ProtocolError, build_messages, prompt_length


SCHEMA = "phiagent-qa-harness-mixture/v1"
BASELINES = ("no_skill", "skillopt", "autosaddler", "delete")
INCUMBENT = "original_chisel"
MIXTURE_ARMS = (
    "full_mixture", "without_harness_router", "without_skill_router",
    "without_both_routers", "random_mixture", "fixed_micro_no_skills",
)
FACTORIAL = {
    "harness_without_skill_router": ["without_skill_router", "without_both_routers"],
    "harness_with_skill_router": ["full_mixture", "without_harness_router"],
    "skill_without_harness_router": ["without_harness_router", "without_both_routers"],
    "skill_with_harness_router": ["full_mixture", "without_skill_router"],
}
IMPORT_CLOSURE_FILES = (
    "phiagent/harness/task_reasoning.py",
    "phiagent/harness/tshirt_fold_strategy.py",
    "phiagent/harness/tshirt_positive_reference.py",
    "phiagent/evaluation/__init__.py",
    "phiagent/evaluation/component_ablation.py",
    "phiagent/evaluation/polyglot_self_evolution_statistics.py",
    "phiagent/learning/chisel_component_quality.py",
    "phiagent/learning/residual_meta_harness.py",
    "phiagent/learning/residual_meta_training.py",
    "phiagent/learning/searchqa_grounded_features.py",
    "phiagent/learning/stage_aware_adm.py",
    "scripts/run_chisel_component_quality.py",
    "scripts/probe_chisel_model_fidelity.py",
    "scripts/run_searchqa_residual_meta_campaign.py",
)
OVERLAY_FILES = (
    "phiagent/harness/__init__.py", "phiagent/harness/mixture_router.py",
    "phiagent/learning/qa_harness_mixture.py",
    "scripts/qa_harness_mixture_protocol.py", "scripts/verify_qa_harness_mixture.py",
    "tests/test_mixture_router.py", "tests/test_qa_harness_mixture.py",
    "tests/test_qa_harness_mixture_protocol.py",
    *IMPORT_CLOSURE_FILES,
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def enabled(config: dict) -> bool:
    declared = config.get("mixture_router")
    if declared is None:
        if config.get("schema_version") == SCHEMA:
            raise ProtocolError("The mixture protocol requires an explicit router declaration")
        return False
    if config.get("schema_version") != SCHEMA or declared.get("enabled") is not True:
        raise ProtocolError("Mixture routing requires its own explicitly enabled protocol")
    return True


def validate_config(config: dict) -> None:
    if not enabled(config):
        raise ProtocolError("This helper accepts only the opt-in mixture protocol")
    if any(config.get(key) for key in ("context_selection", "rule_compression", "length_matched_controls")):
        raise ProtocolError("Mixture routing cannot silently compose legacy interventions")
    settings, incumbent = config["mixture_router"], config["incumbent"]
    if incumbent["status"] not in {"BOUND", "BLOCKED"}:
        raise ProtocolError("The primary incumbent must be explicitly BOUND or BLOCKED before freezing")
    expected = [*BASELINES, *([INCUMBENT] if incumbent["status"] == "BOUND" else []), *MIXTURE_ARMS]
    if (incumbent["arm"] != INCUMBENT or incumbent["diagnostic_reference"] != "delete"
            or config["arms"] != expected or config["candidate_arms"] != list(MIXTURE_ARMS)
            or config["comparison_arms"] != expected
            or set(settings["arms"]) != set(MIXTURE_ARMS)
            or config["primary_candidate"] != "full_mixture"
            or config["primary_reference"] != INCUMBENT
            or config["selection_eligible_arms"] != ["full_mixture"]):
        raise ProtocolError("The complete frozen controls, ablations, and primary identity are required")
    expected_modes = {
        "full_mixture": {"harness_mode": "routed", "skill_mode": "routed"},
        "without_harness_router": {"harness_mode": "fixed", "fixed_harness_id": "micro", "skill_mode": "routed"},
        "without_skill_router": {"harness_mode": "routed", "skill_mode": "all"},
        "without_both_routers": {"harness_mode": "fixed", "fixed_harness_id": "micro", "skill_mode": "all"},
        "random_mixture": {"harness_mode": "random", "skill_mode": "random"},
        "fixed_micro_no_skills": {"harness_mode": "fixed", "fixed_harness_id": "micro", "skill_mode": "none"},
    }
    if settings["arms"] != expected_modes or config["factorial"]["contrasts"] != FACTORIAL:
        raise ProtocolError("The harness/skill 2x2 and matched controls must retain their declared meaning")
    if (type(settings["token_ratio_target"]) not in (float, int)
            or settings["token_ratio_target"] != 0.1
            or config["paper_admitted"] is not False
            or config["budget"]["transport_retries"] != 0
            or config["model"]["temperature"] != 0
            or config["model"]["enable_thinking"] is not False):
        raise ProtocolError("The 10% target, zero retries, deterministic model, and evidence boundary are fixed")
    if incumbent["status"] == "BLOCKED" and (
        not incumbent.get("blocked_reason") or incumbent.get("allow_delete_diagnostic") is not True
    ):
        raise ProtocolError("A blocked primary requires an explicit reason and opt-in DELETE diagnostic")
    control = config["count_matched_random"]
    boundaries = control["context_token_boundaries"]
    if (control["source_arm"] != "full_mixture" or control["control_arm"] != "random_mixture"
            or control["stratification"] != "original_context_exact_token_count"
            or type(control["seed"]) is not int
            or control["seed"] != config["model"]["request_seed"]
            or control["preserve_realized_profile_counts"] is not True
            or control["assignments_frozen_before_answers_or_outcomes"] is not True
            or not isinstance(boundaries, list) or not boundaries
            or any(type(boundary) is not int or boundary <= 0 for boundary in boundaries)
            or boundaries != sorted(set(boundaries))):
        raise ProtocolError("Random harness controls require frozen input-length strata and exact profile counts")
    statistical_plan(config)


def statistical_plan(config: dict) -> dict:
    candidate = config["primary_candidate"]
    references = [arm for arm in config["arms"] if arm != candidate]
    statistics = config["statistics"]
    if (statistics["primary_references"] != references
            or statistics["primary_metrics"] != ["exact_match", "f1", "tokens"]
            or statistics["factorial_quality_metrics"] != ["exact_match", "f1"]
            or statistics["familywise_alpha"] != 0.05
            or statistics["noninferiority_margin"] != 0):
        raise ProtocolError("The prospective simultaneous family and zero-loss margin changed")
    primary = [(candidate, reference, metric) for reference in references
               for metric in statistics["primary_metrics"]]
    factorial = [(left, right, metric) for left, right in config["factorial"]["contrasts"].values()
                 for metric in statistics["factorial_quality_metrics"]]
    contrasts = list(dict.fromkeys(primary + factorial))
    reference = INCUMBENT if config["incumbent"]["status"] == "BOUND" else "delete"
    contrasts.append((candidate, reference, "target_token_slack"))
    if statistics["simultaneous_bounds"] != len(contrasts):
        raise ProtocolError("The complete unique endpoint family, including the token-ratio bound, differs")
    return {"contrasts": contrasts, "family_size": len(contrasts),
            "alpha": statistics["familywise_alpha"], "reference": reference}


def fixed_prompt_hashes(config: dict) -> dict[str, str]:
    arms = (*BASELINES, *(("modify",) if config["incumbent"]["status"] == "BOUND" else ()))
    return {arm: config["prompt_sha256"][arm] for arm in arms}


def prompt_artifact_names(config: dict) -> list[str]:
    return [arm for arm in dict.fromkeys([*config["arms"], *fixed_prompt_hashes(config)]) if arm != INCUMBENT]


def validate_retention_receipt(receipt: dict) -> None:
    if (not isinstance(receipt["coverage_definition"], str) or not receipt["coverage_definition"]
            or receipt["absent_query_words_are_not_claimed_satisfied"] is not True
            or receipt["lexical_gate_is_performance_certificate"] is not False
            or receipt["performance_nonregression_established"] is not False):
        raise ProtocolError("Source-anchor retention must not be presented as factual or quality certification")
    for candidate in receipt["candidates"]:
        if candidate["harness_id"] == "incumbent":
            continue
        available, retained = candidate["available_query_anchors"], candidate["retained_query_anchors"]
        if (not isinstance(available, list) or not isinstance(retained, list)
                or any(not isinstance(term, str) or not term for term in available + retained)
                or available != sorted(set(available)) or retained != sorted(set(retained))
                or not set(retained) <= set(available)):
            raise ProtocolError("Retained query anchors must be a unique subset of source-present anchors")
        expected = len(retained) / len(available) if available else 0.0
        if (type(candidate["lexical_coverage"]) not in (int, float)
                or candidate["lexical_coverage"] != expected
                or type(candidate["all_query_weighted_coverage"]) not in (int, float)
                or not math.isfinite(candidate["all_query_weighted_coverage"])
                or not 0 <= candidate["all_query_weighted_coverage"] <= 1):
            raise ProtocolError("Retention coverage must use source-present anchors, not all-query weights")


def artifact_paths(source: Path, config: dict) -> dict[str, Path]:
    paths = {}
    for key, record in config["incumbent"]["artifacts"].items():
        relative = Path(record["relative_path"])
        path = source / relative
        if relative.is_absolute() or not path.resolve().is_relative_to(source.resolve()):
            raise ProtocolError("Frozen incumbent artifacts must stay inside the new source snapshot")
        paths[key] = path
    return paths


def stage_incumbent(source: Path, config: dict) -> list[str]:
    """Copy hash-bound historical artifacts only into the new private snapshot."""
    validate_config(config)
    return stage_incumbent_artifacts(source, config)


def stage_incumbent_artifacts(source: Path, config: dict) -> list[str]:
    if config["incumbent"]["status"] == "BLOCKED":
        return []
    paths = artifact_paths(source, config)
    for key, destination in paths.items():
        record = config["incumbent"]["artifacts"][key]
        if destination.exists():
            if file_sha(destination) != record["sha256"]:
                raise ProtocolError(f"An existing incumbent artifact differs: {key}")
            continue
        if not record.get("source_path"):
            raise ProtocolError(f"Primary incumbent BLOCKED: exact artifact needs a source path or prestaging: {key}")
        original = Path(record["source_path"])
        if file_sha(original) != record["sha256"]:
            raise ProtocolError(f"Primary incumbent BLOCKED: exact historical artifact differs: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    return [path.relative_to(source).as_posix() for path in paths.values()]


def _self_hashed(path: Path, field: str = "canonical_sha256") -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(field) != canonical_hash({key: item for key, item in value.items() if key != field}):
        raise ProtocolError(f"Historical incumbent self-hash changed: {path.name}")
    return value


class FrozenOriginalChisel:
    """Load the exact archived feature and prediction code, never refit its head."""

    def __init__(self, source: Path, config: dict) -> None:
        paths = artifact_paths(source, config)
        required = {"training", "routes", "protocol", "harness_source", "training_source", "runner_source"}
        if set(paths) != required:
            raise ProtocolError("The exact original head, route/protocol records, and three source files are required")
        for key, path in paths.items():
            if file_sha(path) != config["incumbent"]["artifacts"][key]["sha256"]:
                raise ProtocolError(f"Primary incumbent BLOCKED: artifact hash differs: {key}")
        training = _self_hashed(paths["training"], "training_sha256")
        routes, protocol = _self_hashed(paths["routes"]), _self_hashed(paths["protocol"])
        original_sources = protocol["source_state"]["source_files_sha256"]
        source_keys = {
            "harness_source": "phiagent/learning/residual_meta_harness.py",
            "training_source": "phiagent/learning/residual_meta_training.py",
            "runner_source": "scripts/run_searchqa_residual_meta_campaign.py",
        }
        if any(file_sha(paths[key]) != original_sources[relative] for key, relative in source_keys.items()):
            raise ProtocolError("Current routing code cannot replace the hash-bound historical extractor")
        if (routes["evaluation"]["protocol_sha256"] != protocol["canonical_sha256"]
                or routes["training"]["training_file_sha256"] != file_sha(paths["training"])
                or routes["training"]["training_result_sha256"] != training["training_sha256"]
                or training["feature_schema_sha256"] != protocol["feature_schema_sha256"]
                or routes["threshold_ablations"]["chisel_full"]["score_threshold"] != training["score_threshold"]
                or routes["policies"]["chisel_full"]["default_variant"] != "delete"
                or routes["policies"]["chisel_full"]["override_variant"] != "modify"
                or routes["evaluation"]["labels_read"] is not False
                or training["status"] != "DISCOVERY_CANDIDATE"):
            raise ProtocolError("Historical routing, threshold, and trained-head lineage differ")
        for arm in ("delete", "modify"):
            if file_sha(source / f"prompts/{arm}.system.txt") != protocol["response_matrix"]["prompts"][arm]["sha256"]:
                raise ProtocolError(f"Exact original incumbent prompt differs: {arm}")
        # Restore the live package namespace after importing the archived trio.
        # Returned functions retain their original module globals, not new code.
        names = {
            "harness_source": "phiagent.learning.residual_meta_harness",
            "training_source": "phiagent.learning.residual_meta_training",
            "runner_source": "scripts.run_searchqa_residual_meta_campaign",
        }
        previous = {name: sys.modules.get(name) for name in names.values()}
        previous_path = list(sys.path)
        modules = {}
        try:
            for key, name in names.items():
                spec = importlib.util.spec_from_file_location(name, paths[key])
                if spec is None or spec.loader is None:
                    raise ProtocolError(f"Exact original source is not importable: {key}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[key] = module
        finally:
            sys.path[:] = previous_path
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.campaign = modules["runner_source"]
        self.training_module = modules["training_source"]
        harness = modules["harness_source"]
        if harness.feature_schema_hash(self.campaign.FEATURE_SCHEMA) != training["feature_schema_sha256"]:
            raise ProtocolError("The archived feature schema no longer matches the trained head")
        extractor = canonical_hash({
            "runner_sha256": file_sha(paths["runner_source"]),
            "feature_schema_sha256": training["feature_schema_sha256"],
            "feature_extractor": "task_features/v1",
        })
        if extractor != protocol["feature_extractor_sha256"]:
            raise ProtocolError("The exact archived feature-extractor binding differs")
        self.training = training
        self.names = tuple(sorted(self.campaign.FEATURE_SCHEMA))
        if training["feature_names"] != list(self.names) or set(training["weights"]) != set(self.names):
            raise ProtocolError("Original head feature inventory differs")
        self.bindings = {key: file_sha(path) for key, path in paths.items()}

    def route(self, item: dict) -> dict:
        features = self.campaign.task_features({"question": item["question"], "context": item["context"]})
        vector = [self.campaign.FEATURE_SCHEMA[name].normalize(features[name]) for name in self.names]
        score = self.training_module._predict(
            self.training["intercept"], [self.training["weights"][name] for name in self.names], vector,
        )
        if not math.isfinite(score):
            raise ProtocolError("The exact original routing score is not finite")
        variant = "modify" if score >= self.training["score_threshold"] else "delete"
        return {
            "status": "EXACT_FROZEN_ORIGINAL_ROUTER_RERUN",
            "variant": variant, "score": score, "threshold": self.training["score_threshold"],
            "features": features, "normalized_features": vector,
            "source_artifacts_sha256": self.bindings, "refit": False,
            "answers_or_actor_responses_used": False, "router_model_calls": 0,
            "request_protocol": "Common full-source MRQA messages; historical totals are not a denominator",
        }


def prepare_cells(tokenizer: Any, items: list[dict], prompts: dict, config: dict, source: Path) -> tuple[list[dict], dict]:
    from phiagent.harness.mixture_router import count_matched_random_routes
    from phiagent.learning.qa_harness_mixture import build_mixture_request, mixture_arm_prompt

    validate_config(config)
    ids = [item["id"] for item in items]
    if (not ids or len(ids) != len(set(ids))
            or any(set(item) != {"id", "group", "question", "context"} for item in items)):
        raise ProtocolError("Complete unique public task identities are required before routing")
    if any(prompts[arm] != mixture_arm_prompt() for arm in MIXTURE_ARMS):
        raise ProtocolError("The minimal mixture prompt artifact changed before preparation")
    incumbent = FrozenOriginalChisel(source, config) if config["incumbent"]["status"] == "BOUND" else None
    reference_arm = INCUMBENT if incumbent is not None else "delete"
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(system: str, question: str, context: str) -> int:
        return prompt_length(tokenizer, build_messages(system, {"question": question, "context": context}))

    planned, cache = [], {}
    incumbent_routes, full_plans, context_tokens = {}, {}, {}
    for item in items:
        incumbent_routes[item["id"]] = incumbent.route(item) if incumbent is not None else {
            "status": "PRIMARY_BASELINE_BLOCKED_DELETE_DIAGNOSTIC_ONLY",
            "variant": "delete", "blocked_reason": config["incumbent"]["blocked_reason"],
            "answers_or_actor_responses_used": False, "router_model_calls": 0,
        }
        full_plans[item["id"]] = build_mixture_request(
            question=item["question"], context=item["context"],
            reference_system=prompts[incumbent_routes[item["id"]]["variant"]],
            reference_max_output_tokens=config["model"]["max_output_tokens"],
            settings=config["mixture_router"], arm="full_mixture", count_tokens=count_tokens,
            count_prompt=count_prompt, seed=config["model"]["request_seed"], context_cache=cache,
        )
        context_tokens[item["id"]] = count_tokens(item["context"])
    control = config["count_matched_random"]
    strata = {
        task: f"context_tokens_bucket_{sum(count >= boundary for boundary in control['context_token_boundaries'])}"
        for task, count in context_tokens.items()
    }
    full_routes = {task: plan.receipt["selected_harness"] for task, plan in full_plans.items()}
    assignments = count_matched_random_routes(full_routes, strata=strata, seed=control["seed"])
    cohort_binding = {
        "source_routes_sha256": canonical_hash(full_routes),
        "strata_sha256": canonical_hash(strata),
        "assignments_sha256": canonical_hash(assignments),
        "stratification_policy_sha256": canonical_hash(control),
    }
    for item in items:
        route = incumbent_routes[item["id"]]
        reference_system = prompts[route["variant"]]
        for arm in config["arms"]:
            cap = config["model"]["max_output_tokens"]
            context = item["context"]
            system = reference_system if arm == INCUMBENT else prompts[arm]
            receipt, assignment_receipt = None, None
            if arm in MIXTURE_ARMS:
                settings = config["mixture_router"]
                if arm == "random_mixture":
                    override = {
                        "harness_mode": "fixed", "fixed_harness_id": assignments[item["id"]],
                        "skill_mode": "random",
                    }
                    settings = {**settings, "arms": {**settings["arms"], arm: override}}
                plan = full_plans[item["id"]] if arm == "full_mixture" else build_mixture_request(
                    question=item["question"], context=context, reference_system=reference_system,
                    reference_max_output_tokens=cap, settings=settings, arm=arm, count_tokens=count_tokens,
                    count_prompt=count_prompt, seed=config["model"]["request_seed"], context_cache=cache,
                )
                system, context, cap = plan.system_prompt, plan.context, plan.max_output_tokens
                receipt = json.loads(json.dumps(plan.receipt, sort_keys=True, allow_nan=False))
                validate_retention_receipt(receipt)
                if arm == "random_mixture":
                    if receipt["selected_harness"] != assignments[item["id"]]:
                        raise ProtocolError("A random control did not realize its frozen count-matched assignment")
                    assignment_receipt = {
                        "schema_version": "phiagent-qa-count-matched-harness-assignment/v1",
                        "task": item["id"], "source_arm": "full_mixture", "control_arm": arm,
                        "stratum": strata[item["id"]], "source_context_tokens": context_tokens[item["id"]],
                        "source_harness": full_routes[item["id"]], "assigned_harness": assignments[item["id"]],
                        "seed": control["seed"], **cohort_binding,
                        "original_policy_sha256": full_plans[item["id"]].receipt["policy_sha256"],
                        "effective_policy_sha256": receipt["policy_sha256"],
                        "original_arm_policy": config["mixture_router"]["arms"][arm], "override": override,
                        "answers_or_actor_responses_used": False,
                        "profile_counts_matched_not_actual_token_expenditure": True,
                    }
                    assignment_receipt["canonical_sha256"] = canonical_hash(assignment_receipt)
            messages = build_messages(system, {"question": item["question"], "context": context})
            count = prompt_length(tokenizer, messages)
            if type(cap) is not int or cap <= 0 or count + cap > config["model"]["context_length"]:
                raise ProtocolError("A per-cell output reservation exceeds the full model context")
            if receipt is not None and (
                receipt["expected_prompt_tokens"] != count or receipt["max_output_tokens"] != cap
                or receipt["router_model_calls"] != 0 or receipt["router_provider_tokens"] != 0
                or receipt["answers_or_actor_responses_used"] is not False
            ):
                raise ProtocolError("Mixture receipt differs from the actual zero-call request")
            planned.append({
                "cell": canonical_hash([item["id"], arm, config["model"]["request_seed"]]),
                "task": item["id"], "group": item["group"], "arm": arm,
                "expected_prompt_tokens": count, "max_output_tokens": cap,
                "reference_arm": reference_arm, "incumbent_route": route,
                **({"mixture_receipt": receipt} if receipt is not None else {}),
                **({"random_assignment": assignment_receipt} if assignment_receipt is not None else {}),
                "body": {
                    "model": config["model"]["repository"], "messages": messages,
                    "temperature": 0, "max_tokens": cap, "seed": config["model"]["request_seed"],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            })
    planned.sort(key=lambda cell: cell["cell"])
    return planned, preparation_summary(planned, config, len(cache))


def preparation_summary(planned: list[dict], config: dict, transforms: int) -> dict:
    full_routes = {
        cell["task"]: cell["mixture_receipt"]["selected_harness"]
        for cell in planned if cell["arm"] == "full_mixture"
    }
    random_cells = {cell["task"]: cell for cell in planned if cell["arm"] == "random_mixture"}
    if set(full_routes) != set(random_cells):
        raise ProtocolError("Count-matched random controls require the full reference task cohort")
    strata = {task: cell["random_assignment"]["stratum"] for task, cell in random_cells.items()}
    matched_counts = {}
    for stratum in sorted(set(strata.values())):
        tasks = [task for task in strata if strata[task] == stratum]
        original = Counter(full_routes[task] for task in tasks)
        actual = Counter(random_cells[task]["mixture_receipt"]["selected_harness"] for task in tasks)
        if original != actual:
            raise ProtocolError("Realized random harness counts differ within an input-only stratum")
        matched_counts[stratum] = {
            "tasks": len(tasks), "full_mixture": dict(sorted(original.items())),
            "random_mixture": dict(sorted(actual.items())),
        }
    return {
        "status": "MIXTURE_AND_REFERENCE_REQUESTS_FROZEN_BEFORE_ANSWERS_OR_MODEL_CALLS",
        "primary_baseline_status": config["incumbent"]["status"],
        "reference_arm": INCUMBENT if config["incumbent"]["status"] == "BOUND" else "delete",
        "unique_context_transforms": transforms,
        "count_matched_random": {
            "status": "EXACT_REALIZED_PROFILE_COUNTS_MATCHED_BEFORE_OUTCOMES",
            "stratification": config["count_matched_random"]["stratification"],
            "context_token_boundaries": config["count_matched_random"]["context_token_boundaries"],
            "seed": config["count_matched_random"]["seed"], "strata": matched_counts,
            "changed_assignments": sum(
                full_routes[task] != cell["mixture_receipt"]["selected_harness"]
                for task, cell in random_cells.items()
            ),
            "actual_token_expenditure_is_matched": False,
            "answers_or_actor_responses_used": False,
        },
        "methods": {
            arm: {
                "tasks": sum(cell["arm"] == arm for cell in planned),
                "prompt_tokens": sum(cell["expected_prompt_tokens"] for cell in planned if cell["arm"] == arm),
                "reserved_completion_tokens": sum(cell["max_output_tokens"] for cell in planned if cell["arm"] == arm),
                "selected_harnesses": dict(Counter(
                    cell["mixture_receipt"]["selected_harness"] for cell in planned
                    if cell["arm"] == arm and "mixture_receipt" in cell
                )),
                "incumbent_fallbacks": sum(
                    cell["mixture_receipt"]["identity_incumbent_request"] for cell in planned
                    if cell["arm"] == arm and "mixture_receipt" in cell
                ),
            } for arm in config["arms"]
        },
        "router_model_calls": 0, "router_provider_tokens": 0,
        "selection_uses_answers_or_actor_responses": False,
        "lexical_signals_are_quality_certificates": False,
    }


def output_cap(cell: dict, config: dict) -> int:
    cap = cell["max_output_tokens"]
    permitted = {config["model"]["max_output_tokens"], *(
        profile["max_output_tokens"] for profile in config["mixture_router"]["profiles"]
    )}
    if type(cap) is not int or cap not in permitted or cell["body"]["max_tokens"] != cap:
        raise ProtocolError("Frozen per-cell output cap differs from its actual request")
    if cell["arm"] in MIXTURE_ARMS:
        if cell["mixture_receipt"]["max_output_tokens"] != cap:
            raise ProtocolError("Per-cell cap differs from the selected harness receipt")
    elif cap != config["model"]["max_output_tokens"]:
        raise ProtocolError("The unchanged full-context baseline cap differs")
    return cap


def validate_methods(methods: dict, config: dict) -> None:
    if set(methods) != set(config["arms"]):
        raise ProtocolError("All matched-arm aggregates are required; a partial matrix is invalid")
    counts = set()
    for arm, method in methods.items():
        if any(type(method[key]) is not int or method[key] < 0 for key in (
            "tasks", "prompt_tokens", "completion_tokens", "total_tokens",
        )):
            raise ProtocolError("Exact integer task and token totals are required")
        zero_provider_cost = (
            config.get("schema_version") == "phiagent-qa-chisel-preservation/v1"
            and arm in config.get("zero_provider_cost_arms", [])
            and method["prompt_tokens"] == method["completion_tokens"] == method["total_tokens"] == 0
        )
        if (method["tasks"] == 0 or (method["prompt_tokens"] <= 0 and not zero_provider_cost)
                or method["total_tokens"] != method["prompt_tokens"] + method["completion_tokens"]):
            raise ProtocolError("All requests, including empty/length stops, require additive token accounting")
        if any(type(method[metric]) not in (int, float) or not math.isfinite(method[metric])
               or not 0 <= method[metric] <= 100 for metric in ("exact_match", "f1")):
            raise ProtocolError("Finite official MRQA EM/F1 percentages are required")
        counts.add(method["tasks"])
    if len(counts) != 1:
        raise ProtocolError("Complete identical task counts are required for every arm")


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    validate_methods(methods, config)
    candidate = config["primary_candidate"]
    target = Fraction(str(config["mixture_router"]["token_ratio_target"]))
    available = config["incumbent"]["status"] == "BOUND"
    reference = INCUMBENT if available else "delete"
    numerator, denominator = methods[candidate]["total_tokens"], methods[reference]["total_tokens"]
    quality = all(methods[candidate][metric] >= methods[reference][metric] for metric in ("exact_match", "f1"))
    cost = numerator * target.denominator <= denominator * target.numerator
    point = quality and cost
    endpoint_keys = [f"{candidate}_vs_{reference}_{metric}"
                     for metric in ("exact_match", "f1", "target_token_slack")]
    statistical = None if bounds is None else all(
        bounds[key]["one_sided_lower_bound"] >= 0 for key in endpoint_keys
    )
    return {
        "candidate": candidate, "primary_reference": INCUMBENT,
        "primary_baseline_status": config["incumbent"]["status"],
        "status": "MATCHED_PRIMARY_POINT_TARGET_PASS" if available and point else
                  "MATCHED_PRIMARY_POINT_TARGET_FAIL" if available else "BLOCKED_PRIMARY_BASELINE",
        "token_ratio_target": float(target),
        "observed_total_token_ratio": numerator / denominator if available else None,
        "point_target_passed": point if available else None,
        "statistical_target_passed": (point and statistical) if available and statistical is not None else None,
        "statistical_bound_conditions_established": False,
        "diagnostic_reference": None if available else reference,
        "diagnostic_total_token_ratio": None if available else numerator / denominator,
        "diagnostic_point_target_passed": None if available else point,
        "diagnostic_statistical_bound_passed": None if available else statistical,
        "metric_differences_pp": {metric: methods[candidate][metric] - methods[reference][metric]
                                  for metric in ("exact_match", "f1")},
        "comparison_scope": "Matched official-MRQA-scored training-source rerun; never historical sensitivity totals",
        "automatic_official_scale_up_permitted": False,
        "paper_admitted": False, "production_defaults_changed": False,
    }


def diagnostic_bounds(scores: dict, groups: dict[str, str], config: dict) -> dict:
    from phiagent.evaluation.chisel_component_quality import paired_finite_sample_lower_bound

    if set(scores) != set(config["arms"]) or any(set(rows) != set(groups) for rows in scores.values()):
        raise ProtocolError("Bounds require every declared task in every matched arm")
    plan = statistical_plan(config)
    seed = config["model"]["request_seed"]
    bounds = {}
    for left, right, metric in plan["contrasts"]:
        source_metric = "tokens" if metric == "target_token_slack" else metric
        lvalues = {task: row[source_metric] for task, row in scores[left].items()}
        rvalues = {task: row[source_metric] for task, row in scores[right].items()}
        if metric in {"tokens", "target_token_slack"}:
            if metric == "target_token_slack":
                rvalues = {task: value * config["mixture_router"]["token_ratio_target"]
                           for task, value in rvalues.items()}
            lvalues, rvalues = rvalues, lvalues
        bounds[f"{left}_vs_{right}_{metric}"] = paired_finite_sample_lower_bound(
            {seed: lvalues}, {seed: rvalues}, groups, seeds=(seed,),
            score_upper_bound=config["model"]["context_length"] if source_metric == "tokens" else 1,
            family_size=plan["family_size"], alpha=plan["alpha"], binary=metric == "exact_match",
        )
    return bounds
