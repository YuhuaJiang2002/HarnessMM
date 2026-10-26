"""Opt-in CHISEL preservation study with native source fields and fresh fallbacks."""

from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import socket
import time

from phiagent.learning.qa_chisel_light_compression import light_compression_request
from phiagent.learning.qa_source_answer_fields import source_answer_field
from scripts.prepare_chisel_component_quality_data import canonical_hash, write_json
from scripts.qa_capability_cost_protocol import (
    BASELINES, INCUMBENT, MIXTURE_ARMS, OVERLAY_FILES as CAPABILITY_FILES,
    prepare_comparison_cells,
)
from scripts.qa_harness_mixture_protocol import (
    fixed_prompt_hashes as fixed_prompt_hashes,
    prompt_artifact_names as prompt_artifact_names,
    stage_incumbent_artifacts,
    validate_methods,
)
from scripts.run_chisel_component_quality import (
    ProtocolError, build_messages, prompt_length, score_rows,
)


SCHEMA = "phiagent-qa-chisel-preservation/v1"
SOURCE_FIELD = "source_field_original"
LIGHT = "light_original"
CANDIDATES = (SOURCE_FIELD, LIGHT)
COMPARISONS = (*BASELINES, INCUMBENT, *MIXTURE_ARMS)
ARMS = (*COMPARISONS, *CANDIDATES)
DIRECT_EXECUTION_KIND = "deterministic_source_field"
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
NATIVE_FILES = ("decision.json", "output.txt", "execution.json", "record.json")
IMPLEMENTATION_FILES = (
    "phiagent/learning/qa_source_answer_fields.py",
    "phiagent/learning/qa_chisel_light_compression.py",
)


class LightReadinessError(ProtocolError):
    def __init__(self, *, original_tokens: int, light_tokens: int, tasks: int):
        saving = 1 - light_tokens / original_tokens
        self.readiness_receipt = {
            "status": "BLOCKED_INPUT_TOKEN_READINESS", "tasks": tasks,
            "original_prompt_tokens": original_tokens, "light_prompt_tokens": light_tokens,
            "minimum_light_prompt_saving": 0.1, "observed_light_prompt_saving": saving,
            "model_calls": 0, "evaluator_calls": 0, "benchmark_evaluator_calls": 0,
            "call_count_scope": "input-token readiness computation only",
            "basis": "Exact pinned full chat-template tokens on newly frozen public inputs",
            "quality_measured": False, "success_preservation_established": False,
        }
        super().__init__(
            f"Light prompt saving {saving:.6%} misses the frozen 10% input-only readiness threshold"
        )


OVERLAY_FILES = (
    *CAPABILITY_FILES,
    "phiagent/learning/qa_chisel_light_compression.py",
    "phiagent/learning/qa_source_answer_fields.py",
    "scripts/qa_chisel_preservation_protocol.py",
    "scripts/prepare_qa_chisel_preservation_config.py",
    "scripts/render_qa_chisel_preservation.py",
    "scripts/render_qa_capability_cost.py",
    "scripts/render_qa_harness_mixture.py",
    "phiagent/evaluation/dual_quality_cost.py",
    "tests/test_qa_chisel_light_compression.py",
    "tests/test_qa_source_answer_fields.py",
    "tests/test_qa_chisel_preservation_protocol.py",
    "tests/test_qa_chisel_preservation_report.py",
    "tests/test_qa_harness_mixture_audit_binding.py",
)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def active_candidates(config: dict) -> tuple[str, ...]:
    mode = config["preservation"]["mode"]
    if mode == "isolated_both":
        return CANDIDATES
    if mode == "source_field_only_after_light_budget_block":
        blocker = config["preservation"]["light_readiness_blocker"]
        if (
            blocker["status"] != "BLOCKED_INPUT_TOKEN_READINESS"
            or blocker["model_calls"] != 0 or blocker["benchmark_evaluator_calls"] != 0
            or blocker["minimum_light_prompt_saving"] != 0.1
            or not 0 <= blocker["observed_light_prompt_saving"] < 0.1
            or len(blocker["evidence_sha256"]) != 64
        ):
            raise ProtocolError("Source-only follow-through requires a pinned no-outcome light-budget failure")
        return (SOURCE_FIELD,)
    raise ProtocolError("Unknown preservation study mode")


def validate_config(config: dict) -> None:
    preservation = config["preservation"]
    candidates = active_candidates(config)
    arms = (*COMPARISONS, *candidates)
    if (
        config.get("schema_version") != SCHEMA
        or config["arms"] != list(arms) or config["comparison_arms"] != list(arms)
        or config["candidate_arms"] != list(candidates)
        or config["primary_candidate"] != SOURCE_FIELD
        or config["selection_eligible_arms"] != [SOURCE_FIELD]
        or config["primary_reference"] != INCUMBENT or config["cost_reference"] != INCUMBENT
        or config["incumbent"]["status"] != "BOUND"
        or config["incumbent"]["arm"] != INCUMBENT
        or config["zero_provider_cost_arms"] != [SOURCE_FIELD]
        or config["model"]["temperature"] != 0
        or config["model"]["enable_thinking"] is not False
        or config["model"]["max_output_tokens"] != 512
        or config["budget"]["transport_retries"] != 0
        or config["paper_admitted"] is not False
        or any(config.get(key) for key in (
            "mixture_router", "context_selection", "rule_compression", "length_matched_controls",
        ))
        or config["capability_routing"] != {
            "enabled": True, "optional_top_k": 2, "preserve_full_context": True,
            "preserve_output_cap": True, "random_optional_count_matched_per_task": True,
        }
        or preservation["minimum_total_token_saving"] != {SOURCE_FIELD: 0.0, LIGHT: 0.1}
        or preservation["light_total_token_saving_target_band"] != [0.1, 0.15]
        or preservation["minimum_light_prompt_saving_before_generation"] != 0.1
        or preservation["fresh_model_fallbacks_required"] is not True
        or preservation["combine_interventions"] is not False
        or preservation["preserve_original_head_context_and_output_contract"] is not True
        or preservation["per_task_regressions_must_be_reported"] is not True
        or preservation["source_text_is_factual_certificate"] is not False
        or set(preservation["implementation_sha256"]) != set(IMPLEMENTATION_FILES)
    ):
        raise ProtocolError("The isolated, fresh-inference CHISEL preservation protocol changed")
    for relative, expected in preservation["implementation_sha256"].items():
        path = Path(__file__).resolve().parents[1] / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ProtocolError(f"Frozen preservation implementation changed: {relative}")
    statistical_plan(config)


def statistical_plan(config: dict) -> dict:
    contrasts = [
        (candidate, INCUMBENT, metric) for candidate in active_candidates(config)
        for metric in ("exact_match", "f1", "target_token_slack")
    ]
    settings = config["statistics"]
    if (
        settings["primary_references"] != [INCUMBENT]
        or settings["primary_metrics"] != ["exact_match", "f1", "target_token_slack"]
        or settings["familywise_alpha"] != 0.05
        or settings["noninferiority_margin"] != 0
        or settings["simultaneous_bounds"] != len(contrasts)
    ):
        raise ProtocolError("Every active candidate requires the frozen zero-loss endpoint family")
    return {"contrasts": contrasts, "family_size": len(contrasts),
            "alpha": 0.05, "reference": INCUMBENT}


def stage_incumbent(source: Path, config: dict) -> list[str]:
    validate_config(config)
    files = stage_incumbent_artifacts(source, config)
    mapping = {}
    for profile in ("delete", "modify"):
        original = (source / f"prompts/{profile}.system.txt").read_text()
        compressed = light_compression_request(system_prompt=original, profile=profile)
        relative = f"policies/light_original/{profile}.system.txt"
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(compressed.system_prompt)
        files.append(relative)
        mapping[profile] = compressed.receipt
    write_json(source / "LIGHT_COMPRESSION_MAP.json", mapping)
    return [*files, "LIGHT_COMPRESSION_MAP.json"]


def prepare_cells(tokenizer, items: list[dict], prompts: dict, config: dict, source: Path):
    validate_config(config)
    planned, comparison_summary = prepare_comparison_cells(
        tokenizer, items, prompts, config, source,
    )
    originals = {cell["task"]: cell for cell in planned if cell["arm"] == INCUMBENT}
    for item in items:
        original = originals[item["id"]]
        guard = source_answer_field(question=item["question"], context=item["context"])
        for arm in active_candidates(config):
            cell = copy.deepcopy(original)
            cell.update({
                "cell": canonical_hash([item["id"], arm, config["model"]["request_seed"]]),
                "arm": arm,
                "preservation_receipt": {
                    "schema_version": SCHEMA,
                    "identity_context_transform": True,
                    "original_route_sha256": canonical_hash(original["incumbent_route"]),
                    "reference_prompt_tokens": original["expected_prompt_tokens"],
                    "reference_output_cap": config["model"]["max_output_tokens"],
                    "responses_or_official_answers_used": False,
                    "cached_fallback_response_permitted": False,
                },
            })
            if arm == SOURCE_FIELD:
                cell["preservation_receipt"]["source_field"] = guard
                if guard["status"] == "UNIQUE_SUPPORTED_SOURCE_FIELD":
                    cell.pop("body")
                    cell.update({
                        "execution_kind": DIRECT_EXECUTION_KIND,
                        "expected_prompt_tokens": 0, "max_output_tokens": 0,
                        "source_input": {"question": item["question"], "context": item["context"]},
                    })
            else:
                profile = original["incumbent_route"]["variant"]
                compressed = light_compression_request(system_prompt=prompts[profile], profile=profile)
                messages = build_messages(compressed.system_prompt, item)
                count = prompt_length(tokenizer, messages)
                if count >= original["expected_prompt_tokens"]:
                    raise ProtocolError("Every light-compression model request must be genuinely shorter")
                cell["body"]["messages"] = messages
                cell["expected_prompt_tokens"] = count
                cell["preservation_receipt"]["light_compression"] = compressed.receipt
            output_cap(cell, config)
            planned.append(cell)
    planned.sort(key=lambda cell: cell["cell"])
    original_tokens = sum(cell["expected_prompt_tokens"] for cell in originals.values())
    light_tokens = sum(cell["expected_prompt_tokens"] for cell in planned if cell["arm"] == LIGHT)
    saving = 1 - light_tokens / original_tokens if LIGHT in active_candidates(config) else None
    required = Fraction(str(config["preservation"]["minimum_light_prompt_saving_before_generation"]))
    if saving is not None and (
        (original_tokens - light_tokens) * required.denominator < original_tokens * required.numerator
    ):
        raise LightReadinessError(
            original_tokens=original_tokens, light_tokens=light_tokens, tasks=len(items),
        )
    return planned, {
        "status": "ISOLATED_PRESERVATION_ROUTES_FROZEN_BEFORE_OUTCOMES",
        "primary_baseline_status": "BOUND", "reference_arm": INCUMBENT,
        "cost_reference": INCUMBENT, "comparison_preparation": comparison_summary,
        "light_prompt_saving_fraction": saving,
        "total_token_target_requires_actual_completion_usage": True,
        "methods": {
            arm: {
                "tasks": sum(cell["arm"] == arm for cell in planned),
                "prompt_tokens": sum(cell["expected_prompt_tokens"] for cell in planned if cell["arm"] == arm),
                "reserved_completion_tokens": sum(output_cap(cell, config) for cell in planned if cell["arm"] == arm),
                "execution_kinds": dict(Counter(
                    cell.get("execution_kind", "model") for cell in planned if cell["arm"] == arm
                )),
            } for arm in config["arms"]
        },
        "router_model_calls": 0, "router_provider_tokens": 0,
        "selection_uses_answers_or_actor_responses": False,
        "semantic_or_behavioral_fidelity_established": False,
        "all_model_fallbacks_newly_requested": True,
    }


def direct_answer(cell: dict, config: dict) -> str:
    if (
        config["schema_version"] != SCHEMA or cell["arm"] != SOURCE_FIELD
        or cell.get("execution_kind") != DIRECT_EXECUTION_KIND
        or "body" in cell or cell["expected_prompt_tokens"] != 0
        or cell["max_output_tokens"] != 0
    ):
        raise ProtocolError("A native source-field result cannot masquerade as a provider call")
    guard = source_answer_field(**cell["source_input"])
    if (
        guard != cell["preservation_receipt"]["source_field"]
        or guard["status"] != "UNIQUE_SUPPORTED_SOURCE_FIELD"
        or not isinstance(guard["answer"], str) or not guard["answer"]
    ):
        raise ProtocolError("The runtime source-field decision differs from its pre-outcome freeze")
    return guard["answer"]


def output_cap(cell: dict, config: dict) -> int:
    if cell.get("execution_kind", "model") == DIRECT_EXECUTION_KIND:
        direct_answer(cell, config)
        return 0
    if cell.get("execution_kind", "model") != "model":
        raise ProtocolError("Unknown execution kind")
    cap = config["model"]["max_output_tokens"]
    if cell["max_output_tokens"] != cap or cell["body"]["max_tokens"] != cap:
        raise ProtocolError("Every actual model request retains the original output cap")
    if cell["expected_prompt_tokens"] + cap > config["model"]["context_length"]:
        raise ProtocolError("A model request exceeds the full, untruncated context limit")
    return cap


def direct_record(cell: dict, answer: str) -> dict:
    return {
        "task": cell["task"], "group": cell["group"], "arm": cell["arm"],
        "execution_kind": DIRECT_EXECUTION_KIND, "provider_requests": 0,
        "usage_source": "no_provider_request", "usage": dict(ZERO_USAGE),
        "prediction": answer, "finish_reason": DIRECT_EXECUTION_KIND, "empty_answer": False,
        "output_sha256": text_sha(f"<answer>{answer}</answer>"),
        "decision_sha256": canonical_hash(cell),
    }


def perform_direct_cell(cell: dict, directory: Path, config: dict) -> dict:
    started = time.monotonic()
    answer = direct_answer(cell, config)
    write_json(directory / "decision.json", cell)
    with (directory / "output.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"<answer>{answer}</answer>")
    record = direct_record(cell, answer)
    write_json(directory / "record.json", record)
    write_json(directory / "execution.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "execution_kind": DIRECT_EXECUTION_KIND, "provider_requests": 0,
        "elapsed_seconds": time.monotonic() - started,
        "source_field_recomputed_at_execution": True,
        "cached_model_response_used": False, "provider_usage_fabricated": False,
    })
    return {"usage": dict(ZERO_USAGE), "record": record, "failure": None}


def verify_direct_cell(cell: dict, directory: Path, config: dict) -> tuple[dict, dict]:
    answer = direct_answer(cell, config)
    if {path.name for path in directory.iterdir()} != set(NATIVE_FILES):
        raise RuntimeError("Native result files are missing, extra, or contain a provider transcript")
    if json.loads((directory / "decision.json").read_text()) != cell:
        raise RuntimeError("Native decision bytes do not identify the frozen source input")
    if (directory / "output.txt").read_text() != f"<answer>{answer}</answer>":
        raise RuntimeError("Native output is not the uniquely supported source field")
    record = json.loads((directory / "record.json").read_text())
    if record != direct_record(cell, answer):
        raise RuntimeError("Native output/accounting record changed")
    execution = json.loads((directory / "execution.json").read_text())
    elapsed = execution["elapsed_seconds"]
    if (
        execution["execution_kind"] != DIRECT_EXECUTION_KIND
        or execution["provider_requests"] != 0
        or execution["source_field_recomputed_at_execution"] is not True
        or execution["cached_model_response_used"] is not False
        or execution["provider_usage_fabricated"] is not False
        or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
    ):
        raise RuntimeError("Native execution identity or timing differs")
    return record, {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in NATIVE_FILES
    }


def score_records(records: dict, answers: dict, scorer) -> dict:
    if set(records) != set(answers):
        raise ProtocolError("A native/model method still requires every task and official answer")
    model = {
        task: row for task, row in records.items()
        if row.get("execution_kind", "model") == "model"
    }
    result = score_rows(model, {task: answers[task] for task in model}, scorer)
    for task, row in records.items():
        if task in model:
            continue
        if (
            row.get("execution_kind") != DIRECT_EXECUTION_KIND
            or row["usage"] != ZERO_USAGE or row["provider_requests"] != 0
            or row["usage_source"] != "no_provider_request"
            or "raw" in row or not row["prediction"]
        ):
            raise ProtocolError("Native scoring requires explicit, zero-provider-cost output evidence")
        prediction = row["prediction"]
        result[task] = {
            "prediction": prediction,
            "exact_match": int(scorer.metric_max_over_ground_truths(
                scorer.exact_match_score, prediction, answers[task],
            )),
            "f1": float(scorer.metric_max_over_ground_truths(
                scorer.f1_score, prediction, answers[task],
            )),
            "tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "finish_reason": DIRECT_EXECUTION_KIND, "empty_answer": False,
            "execution_kind": DIRECT_EXECUTION_KIND, "provider_requests": 0,
        }
    return {task: result[task] for task in records}


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    validate_methods(methods, config)
    original = methods[INCUMBENT]
    candidates = {}
    for arm in active_candidates(config):
        current = methods[arm]
        minimum = config["preservation"]["minimum_total_token_saving"][arm]
        required = Fraction(str(minimum))
        ratio = current["total_tokens"] / original["total_tokens"]
        quality = all(current[metric] >= original[metric] for metric in ("exact_match", "f1"))
        cost = (
            current["total_tokens"] < original["total_tokens"]
            and current["total_tokens"] * required.denominator
            <= original["total_tokens"] * (required.denominator - required.numerator)
        )
        point = quality and cost
        endpoints = [f"{arm}_vs_{INCUMBENT}_{metric}"
                     for metric in ("exact_match", "f1", "target_token_slack")]
        statistical = None if bounds is None else point and all(
            bounds[key]["one_sided_lower_bound"] >= 0 for key in endpoints
        )
        candidates[arm] = {
            "quality_preserved_vs_original_point": quality,
            "total_token_target_passed": cost,
            "observed_total_token_ratio": ratio,
            "minimum_total_token_saving": minimum,
            "point_target_passed": point, "statistical_target_passed": statistical,
            "metric_differences_pp": {
                metric: current[metric] - original[metric] for metric in ("exact_match", "f1")
            },
            "quality_strictly_above_every_compared_method": all(
                current[metric] > methods[other][metric] for other in config["arms"] if other != arm
                for metric in ("exact_match", "f1")
            ),
            "tokens_strictly_below_autosaddler": current["total_tokens"] < methods["autosaddler"]["total_tokens"],
        }
    primary = candidates[SOURCE_FIELD]
    return {
        **primary, "status": (
            "PRESERVATION_POINT_PASS" if primary["point_target_passed"] else "PRESERVATION_POINT_FAIL"
        ),
        "candidate": SOURCE_FIELD, "primary_reference": INCUMBENT,
        "primary_baseline_status": "BOUND", "cost_reference": INCUMBENT,
        "candidate_gates": candidates,
        "all_quality_references": [arm for arm in config["arms"] if arm != SOURCE_FIELD],
        "statistical_bound_conditions_established": False,
        "automatic_official_scale_up_permitted": False, "paper_admitted": False,
        "production_defaults_changed": False,
        "comparison_scope": "New groups disjoint from listed training cohorts, not full official evaluation",
        "old_autosaddler_superiority_gate_not_redefined": True,
    }


def diagnostic_bounds(scores: dict, groups: dict[str, str], config: dict) -> dict:
    from phiagent.evaluation.chisel_component_quality import paired_finite_sample_lower_bound

    if set(scores) != set(config["arms"]) or any(set(rows) != set(groups) for rows in scores.values()):
        raise ProtocolError("Paired inference statistics require every frozen method and task")
    seed, plan, result = config["model"]["request_seed"], statistical_plan(config), {}
    for left, right, metric in plan["contrasts"]:
        if metric == "target_token_slack":
            scale = 1 - config["preservation"]["minimum_total_token_saving"][left]
            lvalues = {task: scale * row["tokens"] for task, row in scores[right].items()}
            rvalues = {task: row["tokens"] for task, row in scores[left].items()}
        else:
            lvalues = {task: row[metric] for task, row in scores[left].items()}
            rvalues = {task: row[metric] for task, row in scores[right].items()}
        result[f"{left}_vs_{right}_{metric}"] = paired_finite_sample_lower_bound(
            {seed: lvalues}, {seed: rvalues}, groups, seeds=(seed,),
            score_upper_bound=config["model"]["context_length"] if metric == "target_token_slack" else 1,
            family_size=plan["family_size"], alpha=plan["alpha"], binary=metric == "exact_match",
        )
    return result


def result_diagnostics(planned: list[dict], scores: dict, config: dict) -> dict:
    originals = scores[INCUMBENT]
    pairs = {}
    for arm in config["arms"]:
        if set(scores[arm]) != set(originals):
            raise ProtocolError("Pairwise preservation diagnostics cannot omit tasks")
        counts = Counter(
            (originals[task]["exact_match"], scores[arm][task]["exact_match"])
            for task in originals
        )
        pairs[arm] = {
            "both_correct": counts[1, 1], "both_wrong": counts[0, 0],
            "regressions": counts[1, 0], "gains": counts[0, 1],
            "f1_regressions": sum(scores[arm][task]["f1"] < originals[task]["f1"] for task in originals),
            "regression_task_ids": sorted(
                task for task in originals
                if originals[task]["exact_match"] == 1 and scores[arm][task]["exact_match"] == 0
            ),
        }
    direct_tasks = {
        cell["task"] for cell in planned if cell.get("execution_kind") == DIRECT_EXECUTION_KIND
    }
    branches = {}
    for name, tasks in (
        ("direct_source_field", direct_tasks),
        ("fresh_original_fallback", set(originals) - direct_tasks),
    ):
        branch_scores = scores[SOURCE_FIELD]
        branches[name] = {
            "tasks": len(tasks),
            "successes": sum(branch_scores[task]["exact_match"] for task in tasks),
            "official_f1_sum": math.fsum(branch_scores[task]["f1"] for task in tasks),
            "em_regressions_vs_original": sum(
                originals[task]["exact_match"] > branch_scores[task]["exact_match"] for task in tasks
            ),
            "em_gains_vs_original": sum(
                originals[task]["exact_match"] < branch_scores[task]["exact_match"] for task in tasks
            ),
            "f1_regressions_vs_original": sum(
                originals[task]["f1"] > branch_scores[task]["f1"] for task in tasks
            ),
            "provider_tokens": sum(branch_scores[task]["tokens"] for task in tasks),
        }
    direct = len(direct_tasks)
    return {
        "paired_original_comparisons": pairs,
        "source_field_branch_diagnostics": branches,
        "execution_accounting": {
            "logical_cells": len(planned), "real_provider_requests": len(planned) - direct,
            "deterministic_source_outputs": direct, "reused_model_responses": 0,
            "native_outputs_are_not_provider_completions": True,
        },
    }


def replay_matrix(tokenizer, public, planned, source, config):
    prompts = {
        arm: (source / f"prompts/{arm}.system.txt").read_text()
        for arm in prompt_artifact_names(config)
    }
    rebuilt, summary = prepare_cells(tokenizer, public, prompts, config, source)
    if rebuilt != planned:
        raise RuntimeError("Fresh preservation requests/native decisions differ from their source freeze")
    by_pair = {(cell["task"], cell["arm"]): cell for cell in rebuilt}
    expected = {(item["id"], arm) for item in public for arm in config["arms"]}
    if set(by_pair) != expected or len(by_pair) != len(rebuilt):
        raise RuntimeError("Every preservation and comparison arm must retain every complete task")
    for item in public:
        original = by_pair[item["id"], INCUMBENT]
        source_field = by_pair[item["id"], SOURCE_FIELD]
        if source_field.get("execution_kind", "model") == "model" and source_field["body"] != original["body"]:
            raise RuntimeError("The new source-field fallback changed the exact Original request")
        primary = by_pair[item["id"], "capability_routed"]["mixture_receipt"]
        random = by_pair[item["id"], "count_matched_random"]["mixture_receipt"]
        if primary["selected_optional_count"] != random["selected_optional_count"]:
            raise RuntimeError("An existing count-matched comparison was changed")
    return {
        "summary": summary, "complete_request_and_receipt_replay": True,
        "source_fields_recomputed_without_official_answers": True,
        "all_fallback_model_requests_identical_to_original": True,
        "original_head_and_full_context_retained": True,
        "all_model_output_caps_retained": True,
        "native_branch_has_no_model_output_cap_or_provider_charge": True,
        "answers_or_actor_outcomes_used_for_reconstruction": False,
    }
