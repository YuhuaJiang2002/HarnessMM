"""Cost-correct diagnostics for a paid QA draft followed by an optional strong answer."""

from __future__ import annotations

import math
from typing import Any, Mapping

from phiagent.learning.candidate_cost_frontier import at_budget, binary_cost_frontier


def _validate_cells(rows: list[dict[str, Any]], profiles: tuple[str, ...]) -> None:
    binary_cost_frontier(rows, profiles)
    for row in rows:
        for name in profiles:
            cell = row["cells"][name]
            if (cell["tokens"] <= 0
                    or type(cell["f1"]) not in (int, float)
                    or not math.isfinite(cell["f1"]) or not 0 <= cell["f1"] <= 1
                    or (cell["score"] == 1 and cell["f1"] != 1)):
                raise ValueError("positive measured costs and consistent official EM/F1 are required")


def paid_recheck_rows(
    rows: list[dict[str, Any]], *, draft: str, references: tuple[str, ...],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if (not isinstance(draft, str) or not draft or not references
            or any(not isinstance(name, str) or not name for name in references)
            or draft in references or len(set(references)) != len(references)):
        raise ValueError("a draft and distinct non-draft recheck profiles are required")
    profiles = (draft, *references)
    _validate_cells(rows, profiles)
    names = ("keep_draft", *(f"recheck:{name}" for name in references))
    transformed = []
    for row in rows:
        first = row["cells"][draft]
        cells = {
            "keep_draft": {
                "score": first["score"], "f1": first["f1"], "tokens": first["tokens"],
                "draft_tokens": first["tokens"], "recheck_tokens": 0, "calls": 1,
            }
        }
        for name in references:
            second = row["cells"][name]
            cells[f"recheck:{name}"] = {
                "score": second["score"], "f1": second["f1"],
                "tokens": first["tokens"] + second["tokens"],
                "draft_tokens": first["tokens"], "recheck_tokens": second["tokens"], "calls": 2,
            }
        transformed.append({"id": row["id"], "cells": cells})
    return transformed, names


def summarize_paid_policy(
    rows: list[dict[str, Any]], profiles: tuple[str, ...], choices: Mapping[str, str],
) -> dict[str, Any]:
    if (not rows or len({row["id"] for row in rows}) != len(rows)
            or set(choices) != {row["id"] for row in rows}
            or any(choice not in profiles for choice in choices.values())):
        raise ValueError("one declared decision for every unique task is required")
    _validate_cells(rows, profiles)
    selected = [row["cells"][choices[row["id"]]] for row in rows]
    for cell in selected:
        if (any(type(cell[key]) is not int or cell[key] < 0 for key in (
                "tokens", "draft_tokens", "recheck_tokens", "calls",
            )) or cell["tokens"] != cell["draft_tokens"] + cell["recheck_tokens"]
                or cell["draft_tokens"] <= 0
                or cell["calls"] != 1 + int(cell["recheck_tokens"] > 0)):
            raise ValueError("selected paths must charge the paid draft and every recheck")
    return {
        "tasks": len(rows),
        "successes": sum(cell["score"] for cell in selected),
        "f1": math.fsum(cell["f1"] for cell in selected) / len(rows),
        "draft_tokens": sum(cell["draft_tokens"] for cell in selected),
        "recheck_tokens": sum(cell["recheck_tokens"] for cell in selected),
        "total_tokens": sum(cell["tokens"] for cell in selected),
        "model_calls_if_executed": sum(cell["calls"] for cell in selected),
        "rechecked_tasks": sum(cell["calls"] == 2 for cell in selected),
    }


def describe_paid_headroom(
    rows: list[dict[str, Any]], *, draft: str, references: tuple[str, ...],
    comparators: tuple[str, ...], cost_reference: str,
) -> dict[str, Any]:
    if (not comparators or len(comparators) != len(set(comparators))
            or cost_reference not in comparators):
        raise ValueError("complete distinct fixed comparators and the cost reference are required")
    # Fixed methods are quality/cost references, never free alternatives after a paid draft.
    _validate_cells(rows, tuple(dict.fromkeys((draft, *references, *comparators))))
    fixed = {
        name: {
            "successes": sum(row["cells"][name]["score"] for row in rows),
            "f1": math.fsum(row["cells"][name]["f1"] for row in rows) / len(rows),
            "tokens": sum(row["cells"][name]["tokens"] for row in rows),
        } for name in comparators
    }
    paid, profiles = paid_recheck_rows(rows, draft=draft, references=references)
    budget = fixed[cost_reference]["tokens"] - 1
    frontier = binary_cost_frontier(paid, profiles)
    bound = at_budget(frontier, budget)
    strongest_em = max(row["successes"] for row in fixed.values())
    strongest_f1 = max(row["f1"] for row in fixed.values())
    choices = {
        row["id"]: min(profiles, key=lambda name: (
            -row["cells"][name]["score"], -row["cells"][name]["f1"],
            row["cells"][name]["tokens"], profiles.index(name),
        )) for row in paid
    }
    witness = summarize_paid_policy(paid, profiles, choices)
    witness.update({
        "choices": choices, "selection_uses_recorded_outcomes": True,
        "deployable": False,
        "strict_joint_gate": (
            witness["successes"] > strongest_em
            and witness["f1"] > strongest_f1 + 1e-12
            and witness["total_tokens"] <= budget
        ),
    })
    if not bound["feasible"] or bound["successes"] <= strongest_em:
        status = "PAID_RECHECK_INFEASIBLE_BY_EM_FOR_RECORDED_BANK"
    elif witness["strict_joint_gate"]:
        status = "PAID_RECHECK_ORACLE_HEADROOM_NOT_A_METHOD_RESULT"
    else:
        status = "PAID_RECHECK_EM_HEADROOM_JOINT_F1_COST_UNRESOLVED"
    return {
        "status": status, "draft": draft, "recheck_profiles": list(references),
        "cost_reference": cost_reference, "fixed_comparators": fixed,
        "tasks": len(rows), "strict_token_budget": budget,
        "strongest_fixed_successes": strongest_em, "strongest_fixed_f1": strongest_f1,
        "paid_binary_frontier": frontier, "maximum_em_under_budget": bound,
        "outcome_privileged_witness": witness,
        "draft_only": summarize_paid_policy(
            paid, profiles, {row["id"]: "keep_draft" for row in paid},
        ),
        "always_recheck": {
            name: summarize_paid_policy(
                paid, profiles, {row["id"]: f"recheck:{name}" for row in paid},
            ) for name in references
        },
        "first_pass_charged_on_every_task": True,
        "second_pass_contract": "Unmodified strong request, not conditioned on the draft",
        "actual_new_model_calls": 0, "new_verification_or_correction_quality_established": False,
        "oracle_only": True, "paper_admitted": False,
        "scope": "Recorded opened-development bank; not a new execution, learned policy or confirmation",
    }


def describe_paid_cost_envelope(
    records: list[dict[str, Any]], methods: dict[str, dict[str, Any]], *,
    draft: str, reference: str, cost_reference: str, max_output_tokens: int,
) -> dict[str, Any]:
    """Bound selected completion costs without pretending prompt partitions include them."""
    if (not records or len(methods) < 3 or len({draft, reference, cost_reference}) != 3
            or not {draft, reference, cost_reference} <= set(methods)
            or type(max_output_tokens) is not int or max_output_tokens <= 0):
        raise ValueError("complete profiles, distinct roles and a positive output cap are required")
    by_arm: dict[str, dict[str, dict]] = {arm: {} for arm in methods}
    groups: dict[str, str] = {}
    for record in records:
        task, arm, group = record["task"], record["arm"], record["group"]
        if (arm not in by_arm or not isinstance(task, str) or not task
                or not isinstance(group, str) or not group or task in by_arm[arm]
                or (task in groups and groups[task] != group)):
            raise ValueError("unique complete task/arm identities with consistent groups are required")
        score, f1, tokens = (
            record["official_exact_match"], record["official_f1"], record["prompt_tokens"],
        )
        if (type(score) is not int or score not in (0, 1)
                or type(f1) not in (int, float) or not math.isfinite(f1) or not 0 <= f1 <= 1
                or (score == 1 and f1 != 1) or type(tokens) is not int or tokens <= 0):
            raise ValueError("invalid official score or exact input-token count")
        groups[task] = group
        by_arm[arm][task] = record
    tasks = sorted(groups)
    for arm, rows in by_arm.items():
        aggregate = methods[arm]
        if set(rows) != set(tasks):
            raise ValueError("partial paired matrices cannot support a recheck diagnosis")
        if any(type(aggregate[key]) is not int or aggregate[key] < 0 for key in (
            "tasks", "successes", "prompt_tokens", "completion_tokens",
        )):
            raise ValueError("aggregate output usage must be known, not imputed or estimated")
        if (aggregate["tasks"] != len(tasks)
                or aggregate["completion_tokens"] > max_output_tokens * len(tasks)
                or aggregate["successes"] != sum(row["official_exact_match"] for row in rows.values())
                or aggregate["prompt_tokens"] != sum(row["prompt_tokens"] for row in rows.values())):
            raise ValueError("per-request data and frozen aggregate counts differ")
        for field, computed in (
            ("exact_match", 100 * aggregate["successes"] / len(tasks)),
            ("f1", 100 * math.fsum(row["official_f1"] for row in rows.values()) / len(tasks)),
        ):
            if (type(aggregate[field]) not in (int, float) or not math.isfinite(aggregate[field])
                    or abs(aggregate[field] - computed) > 1e-10):
                raise ValueError("official metric rollup differs beyond floating-point reduction roundoff")

    first, second = by_arm[draft], by_arm[reference]
    first_tokens = methods[draft]["prompt_tokens"] + methods[draft]["completion_tokens"]
    budget = methods[cost_reference]["prompt_tokens"] + methods[cost_reference]["completion_tokens"] - 1
    optimistic_rows = [
        {"id": task, "cells": {
            "keep": {"score": first[task]["official_exact_match"], "tokens": first[task]["prompt_tokens"]},
            "recheck": {"score": second[task]["official_exact_match"],
                        "tokens": first[task]["prompt_tokens"] + second[task]["prompt_tokens"]},
        }} for task in tasks
    ]
    lower_frontier = binary_cost_frontier(optimistic_rows, ("keep", "recheck"))
    # Every policy pays the same complete first pass, including all its output tokens.
    debit = methods[draft]["completion_tokens"]
    lower_frontier["minimum_possible_tokens"] += debit
    for point in lower_frontier["points"]:
        point["minimum_tokens"] += debit
    bound = at_budget(lower_frontier, budget)
    selected = [
        task for task in tasks if (
            second[task]["official_exact_match"], second[task]["official_f1"]
        ) > (first[task]["official_exact_match"], first[task]["official_f1"])
    ]
    selected_set = set(selected)
    chosen = [second[task] if task in selected_set else first[task] for task in tasks]
    count = len(selected)
    second_prompt = sum(second[task]["prompt_tokens"] for task in selected)
    all_second_completion = methods[reference]["completion_tokens"]
    completion_lower = max(0, all_second_completion - (len(tasks) - count) * max_output_tokens)
    completion_upper = min(all_second_completion, count * max_output_tokens)
    lower = first_tokens + second_prompt + completion_lower
    upper = first_tokens + second_prompt + completion_upper
    successes = sum(row["official_exact_match"] for row in chosen)
    f1_percent = 100 * math.fsum(row["official_f1"] for row in chosen) / len(tasks)
    strongest_em = max(method["successes"] for method in methods.values())
    strongest_f1 = max(method["f1"] for method in methods.values())
    passes_quality = successes > strongest_em and f1_percent > strongest_f1 + 1e-10
    if not bound["feasible"] or bound["successes"] <= strongest_em:
        status = "PAID_RECHECK_INFEASIBLE_BY_OPTIMISTIC_EM_BOUND"
    elif passes_quality and upper <= budget:
        status = "COST_BOUNDED_ORACLE_HEADROOM_NOT_A_METHOD_RESULT"
    else:
        status = "JOINT_QUALITY_COST_FEASIBILITY_UNRESOLVED"
    return {
        "status": status, "tasks": len(tasks), "question_groups": len(set(groups.values())),
        "profiles_checked": sorted(methods), "records_checked": len(records),
        "draft": draft, "reference": reference, "cost_reference": cost_reference,
        "strict_total_token_budget": budget, "draft_total_tokens_exact": first_tokens,
        "maximum_output_tokens_per_original_call": max_output_tokens,
        "strongest_fixed_successes": strongest_em, "strongest_fixed_f1_percent": strongest_f1,
        "optimistic_em_frontier_token_lower_bounds": lower_frontier,
        "optimistic_em_at_budget": bound,
        "witness": {
            "recheck_task_ids": selected, "recheck_task_count": count,
            "successes": successes, "f1_percent": f1_percent,
            "recheck_prompt_tokens_exact": second_prompt,
            "recheck_completion_tokens_lower_bound": completion_lower,
            "recheck_completion_tokens_upper_bound": completion_upper,
            "total_token_lower_bound": lower, "total_token_upper_bound": upper,
            "total_tokens_exact": lower if lower == upper else None,
            "strict_quality_gate": passes_quality,
            "strict_cost_upper_bound_gate": upper <= budget,
            "model_calls_if_executed": len(tasks) + count,
            "selection_uses_recorded_outcomes": True, "deployable": False,
        },
        "always_recheck_total_tokens_exact": (
            first_tokens + methods[reference]["prompt_tokens"] + all_second_completion
        ),
        "per_selected_response_completion_usage_available": False,
        "unknown_selected_output_cost_imputed_as_zero": False,
        "cost_bounds_are_measured_selected_path_tokens": False,
        "runtime_second_pass": "Original strong request without draft-conditioned revision",
        "actual_new_model_calls": 0, "paper_admitted": False, "oracle_only": True,
        "scope": "Opened training-source diagnosis; neither learned routing nor new execution",
    }
