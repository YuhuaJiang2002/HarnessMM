#!/usr/bin/env python3
"""Audit saved ALFWorld pilot provenance and costs; never run an evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_chisel_repair_alfworld as runner


def expected_cells(manifest: Mapping[str, Any], phase: str) -> set[tuple[str, str]]:
    arms = runner.BASELINES if phase == "discovery_baselines" else (
        runner.PATCH_ARMS if phase == "discovery_patches"
        else (*runner.BASELINES, *runner.PATCH_ARMS)
    )
    split = "evaluation" if phase == "evaluation" else "discovery"
    return {(task_id, arm) for task_id in manifest["split"][split] for arm in arms}


def validate_row(row: Mapping[str, Any], budget: Mapping[str, Any]) -> list[str]:
    errors = []
    if type(row.get("valid")) is not bool:
        errors.append("validity_mask_not_boolean")
    if row.get("valid"):
        if type(row.get("official_success")) is not int or row["official_success"] not in (0, 1):
            errors.append("valid_cell_lacks_native_binary_success")
        if not row.get("cost_complete"):
            errors.append("valid_cell_has_unknown_cost")
        if row.get("calls") != row.get("turns"):
            errors.append("actor_calls_differ_from_official_action_turns")
        if not 0 < row.get("turns", -1) <= budget["max_steps"]:
            errors.append("common_step_cap_violated")
    elif row.get("official_success") is not None:
        errors.append("infrastructure_invalid_cell_scored_instead_of_masked")
    usage = row.get("usage") or {}
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in runner.USAGE_KEYS):
        errors.append("usage_not_nonnegative_actual_integers")
    elif usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        errors.append("per_task_usage_arithmetic_differs")
    return errors


def validate_feedback_signals(step: Mapping[str, Any]) -> list[str]:
    errors = []
    if type(step.get("projection_format_valid")) is not bool:
        errors.append("missing_projection_format_diagnostic")
    if type(step.get("environment_command_error")) is not bool:
        errors.append("missing_public_command_error_signal")
    elif not isinstance(step.get("observation"), str) or (
        step["environment_command_error"] != runner.tool_problem(step["observation"])
    ):
        errors.append("command_error_signal_differs_from_public_feedback")
    return errors


def validate_repair_error_counts(conversation, traces) -> list[str]:
    if len(conversation) != len(traces):
        return ["repair_trace_count_differs_from_observed_turns"]
    consecutive_errors = 0
    for step, trace in zip(conversation, traces):
        if trace["consecutive_public_tool_errors"] != consecutive_errors:
            return ["recovery_error_count_differs_from_public_feedback"]
        consecutive_errors = consecutive_errors + 1 if step["environment_command_error"] else 0
    return []


def audit(study: Path, *, discovery_only: bool = False) -> dict[str, Any]:
    manifest, binding = runner.load_bound(study)
    budget = manifest["budget"]
    errors: list[dict[str, Any]] = []

    def require(condition: bool, name: str, **context: Any) -> None:
        if not condition:
            errors.append({"check": name, **context})

    rows_by_phase = {}
    all_rows = []
    actual_calls = 0
    actual_tokens = dict.fromkeys(runner.USAGE_KEYS, 0)
    invalid_cells = []
    inventory = {item["id"]: item for item in manifest["task_inventory"]}
    discovery_groups = {inventory[task_id]["group_id"] for task_id in manifest["split"]["discovery"]}
    evaluation_groups = {inventory[task_id]["group_id"] for task_id in manifest["split"]["evaluation"]}
    require(not discovery_groups & evaluation_groups, "discovery_evaluation_floorplan_overlap")
    require(len(inventory) == 60, "incomplete_previous60_inventory")
    require(len(manifest["split"]["evaluation"]) >= 36, "insufficient_evaluation_task_count")
    parent_selection = runner.read_json(study / "parent_selection.json")
    compiler = runner.read_json(study / "compiler.json")
    proposal = runner.read_json(study / "proposal/selection.json")
    patch = runner.read_json(study / "proposal/patch.json")
    public_packets = runner.read_json(study / "proposal/discovery_public_packets.json")
    parent = parent_selection["selected_parent"]
    require(proposal["calls"] == proposal["candidate_count"] == 1, "fixed_single_proposal_budget")
    require(set(public_packets["discovery_ids"]) <= set(manifest["split"]["discovery"]),
            "proposal_packet_outside_discovery")
    require(runner.file_hash(study / "proposal/patch.json") == compiler["patch_sha256"],
            "compiled_patch_changed")
    require(patch["parent_harness_sha256"] == binding["harness_hashes"][parent],
            "patch_bound_to_different_parent")

    phases = ("discovery_baselines", "discovery_patches") if discovery_only else (
        "discovery_baselines", "discovery_patches", "evaluation"
    )
    for phase in phases:
        phase_result = runner.read_json(study / phase / "result.json")
        rows = phase_result["rows"]
        rows_by_phase[phase] = rows
        all_rows.extend(rows)
        actual_pairs = [(row["id"], row["method"]) for row in rows]
        require(set(actual_pairs) == expected_cells(manifest, phase), "task_arm_coverage_differs", phase=phase)
        require(len(actual_pairs) == len(set(actual_pairs)), "duplicate_task_arm_cells", phase=phase)
        schedule = runner.read_json(study / phase / "frozen_schedule.json")
        require({(row["id"], row["arm"]) for row in schedule["cells"]} == set(actual_pairs),
                "executed_cells_differ_from_frozen_schedule", phase=phase)
        if phase == "evaluation":
            require(datetime.fromisoformat(compiler["compiled_at"]) <=
                    datetime.fromisoformat(schedule["created_at"]),
                    "evaluation_started_before_compiler_freeze")
        for row in rows:
            key = {"phase": phase, "method": row["method"], "id": row["id"]}
            for message in validate_row(row, budget):
                require(False, message, **key)
            require(row.get("model_id") == binding["model"]["id"], "row_actor_model_differs", **key)
            require(row.get("model_revision") == binding["model"]["revision"], "row_actor_revision_differs", **key)
            require(row.get("group_id") == inventory[row["id"]]["group_id"], "row_floorplan_group_differs", **key)
            if not row["valid"]:
                invalid_cells.append({**key, "failure_type": row.get("failure_type"),
                                      "cost_complete": row["cost_complete"]})
            root = Path(row.get("artifact_dir", ""))
            if not root.is_absolute() or not root.is_relative_to(study):
                require(False, "missing_or_outside_study_cell_artifact", **key)
                continue
            require(runner.read_json(root / "result.json") == row, "phase_cell_differs_from_saved_task_cell", **key)
            job = runner.read_json(root / "job.json")
            policy_arm = parent if row["method"] in runner.PATCH_ARMS else row["method"]
            expected_policy = {
                **manifest["baselines"][policy_arm],
                "harness_sha256": binding["harness_hashes"][policy_arm],
            }
            require(job["policy"] == expected_policy, "executed_policy_differs_from_frozen_parent_or_baseline", **key)
            require(job["item"] == inventory[row["id"]], "executed_task_metadata_differs_from_frozen_inventory", **key)
            require(row["parent_harness_sha256"] == expected_policy["harness_sha256"],
                    "row_policy_binding_differs", **key)
            conversation = runner.read_json(root / "conversation.json")
            require(row.get("tool_error_signal_version") == binding.get("tool_error_signal_version"),
                    "per_cell_error_signal_contract_differs", **key)
            corrected_error_signal = row.get("tool_error_signal_version") == runner.TOOL_ERROR_SIGNAL_VERSION
            if corrected_error_signal:
                for step in conversation:
                    for error in validate_feedback_signals(step):
                        require(False, error, **key)
            call_directories = sorted(path for path in (root / "calls").glob("*") if path.is_dir())
            require(len(call_directories) == row["attempted_calls"], "call_directory_count_differs", **key)
            accounted = dict.fromkeys(runner.USAGE_KEYS, 0)
            accounted_calls = 0
            for call_root in call_directories:
                if not (call_root / "request.json").is_file():
                    require(not row["valid"], "valid_cell_missing_raw_request", **key)
                    continue
                request = runner.read_json(call_root / "request.json")
                body = request["body"]
                require(request["endpoint"] in binding["model"]["endpoints"], "undeclared_actor_endpoint", **key)
                require(body["model"] == binding["model"]["id"], "request_actor_model_differs", **key)
                require(body["temperature"] == budget["temperature"], "request_temperature_differs", **key)
                require(body["max_tokens"] == budget["max_completion_tokens"], "request_completion_cap_differs", **key)
                require(body["seed"] == row["seed"], "request_seed_differs", **key)
                require(body["chat_template_kwargs"] == {"enable_thinking": False},
                        "request_thinking_contract_differs", **key)
                require(body["messages"][0] == {"role": "system", "content": runner.SYSTEM},
                        "common_system_instruction_modified", **key)
                require(len(body["messages"]) == 2 and body["messages"][1]["role"] == "user",
                        "repair_context_not_ephemeral_user_level", **key)
                require(body["messages"][1]["content"].count("[CHISEL recovery-memory component:") <= 1,
                        "repair_context_accumulated", **key)
                if not (call_root / "accounting.json").is_file():
                    require(not row["valid"], "valid_cell_lacks_actual_call_usage", **key)
                    continue
                response = runner.read_json(call_root / "response.json")
                usage = response["usage"]
                accounting = runner.read_json(call_root / "accounting.json")
                require(not row["valid"] or response["model"] == binding["model"]["id"],
                        "valid_cell_response_model_differs", **key)
                require({name: usage[name] for name in runner.USAGE_KEYS} == accounting["usage"],
                        "saved_call_usage_differs_from_provider_response", **key)
                for name in runner.USAGE_KEYS:
                    accounted[name] += usage[name]
                accounted_calls += 1
            require(accounted == row["usage"], "task_cost_differs_from_raw_provider_usage", **key)
            require(accounted_calls == row.get("accounted_calls"), "accounted_call_count_differs", **key)
            actual_calls += row["calls"]
            for name in runner.USAGE_KEYS:
                actual_tokens[name] += accounted[name]
            if row["method"] in runner.PATCH_ARMS:
                traces = runner.read_json(root / "repair_trace.json")
                if corrected_error_signal:
                    for error in validate_repair_error_counts(conversation, traces):
                        require(False, error, **key)
                require(sum(trace["activated_now"] for trace in traces) <= 1, "multiple_component_installs", **key)
                require(all(trace["router_model_calls"] == 0 for trace in traces), "unbudgeted_router_calls", **key)
                require(all(not trace["uses_gold_or_evaluator_signals"] for trace in traces),
                        "router_declares_outcome_signals", **key)
                require(all(trace["parent_harness_sha256"] == patch["parent_harness_sha256"] for trace in traces),
                        "repair_trace_parent_binding_differs", **key)
        for arm, summary in phase_result["summaries"].items():
            expected = runner.arm_summary([row for row in rows if row["method"] == arm], len(phase_result["task_ids"]))
            materialized = dict(summary)
            for field in ("elapsed_seconds_sum", "cpu_seconds_sum"):
                require(math.isclose(materialized.pop(field), expected.pop(field),
                                     rel_tol=1e-12, abs_tol=1e-9),
                        "elapsed_or_cpu_sum_differs", phase=phase, method=arm, field=field)
            require(materialized == expected, "summary_differs_from_complete_task_cells", phase=phase, method=arm)

    discovery = runner.read_json(study / "discovery_baselines/result.json")
    require(runner.file_hash(study / "discovery_baselines/result.json") ==
            parent_selection["discovery_result_sha256"], "selected_parent_discovery_result_changed")
    require(runner.choose_parent(discovery["summaries"]) == parent, "parent_selection_rule_differs")
    all_discovery_summaries = {
        **discovery["summaries"],
        **runner.read_json(study / "discovery_patches/result.json")["summaries"],
    }
    require(runner.compile_choice(all_discovery_summaries, parent) == compiler["mode"],
            "compiled_policy_differs_from_preregistered_rule")
    parent_rows = {row["id"]: row for row in rows_by_phase["discovery_baselines"] if row["method"] == parent}
    require(all(parent_rows[task_id]["valid"] and parent_rows[task_id]["official_success"] == 0
                for task_id in public_packets["discovery_ids"]), "proposal_packet_not_a_parent_discovery_failure")
    evaluation_rows = rows_by_phase.get("evaluation", [])
    eval_parent = {row["id"]: row for row in evaluation_rows if row["method"] == parent}
    paired = []
    for row in evaluation_rows:
        reference = eval_parent[row["id"]]
        paired.append({
            "id": row["id"], "group_id": row["group_id"], "method": row["method"],
            "parent": parent, "same_seed": row["seed"] == reference["seed"],
            "valid_pair": row["valid"] and reference["valid"],
            "official_success": row["official_success"], "parent_success": reference["official_success"],
            "success_difference": row["official_success"] - reference["official_success"]
            if row["valid"] and reference["valid"] else None,
            "total_tokens": row["usage"]["total_tokens"],
            "parent_total_tokens": reference["usage"]["total_tokens"],
        })
        require(row["seed"] == reference["seed"], "whole_task_pair_seed_differs",
                id=row["id"], method=row["method"])
    proposer_response = runner.read_json(study / "proposal/llm_call/response.json")
    require(proposer_response["model"] == binding["model"]["id"], "proposer_actor_model_differs")
    proposer_usage = proposer_response["usage"]
    require({name: proposer_usage[name] for name in runner.USAGE_KEYS} == proposal["usage"],
            "proposer_cost_differs_from_actual_response")
    return {
        "schema_version": "phiagent-chisel-repair-alfworld-integrity-audit/v1",
        "audited_at": runner.now(), "status": "PASS" if not errors else "FAIL",
        "scope_phases": list(phases),
        "all_planned_cells_valid": not invalid_cells,
        "expected_cells": sum(len(expected_cells(manifest, phase)) for phase in rows_by_phase),
        "observed_cells": len(all_rows), "invalid_cells": invalid_cells,
        "actor_calls": actual_calls, "actor_usage": actual_tokens, "proposer_calls": proposal["calls"],
        "proposer_usage": proposal["usage"], "errors": errors,
        "model": binding["model"], "binding_sha256": binding["binding_sha256"],
        "tool_error_signal_version": binding.get("tool_error_signal_version", "legacy-v1-format-conflation"),
        "manifest_sha256": manifest["manifest_sha256"],
        "parent": parent, "compiled_mode": compiler["mode"], "evaluation_pairs": paired,
        "uncertainty_unit": "Complete task, grouped by floorplan; pairs supplied for parent cluster uncertainty",
        "same_prefix_causal_evidence": False, "fresh_confirmation": False,
        "exposure": runner.EXPOSURE, "audit_source_sha256": runner.file_hash(Path(__file__)),
        "audit_scope": "Saved-result provenance and arithmetic only; no environment/evaluator/model invoked",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--discovery-only", action="store_true")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    output = study / ("discovery_integrity_audit.json" if args.discovery_only else "integrity_audit.json")
    if output.exists():
        raise FileExistsError(output)
    result = audit(study, discovery_only=args.discovery_only)
    runner.write_json(output, result)
    print(json.dumps({key: result[key] for key in (
        "status", "observed_cells", "expected_cells", "all_planned_cells_valid", "actor_calls",
    )}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
