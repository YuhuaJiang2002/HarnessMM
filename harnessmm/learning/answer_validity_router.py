"""Fail-closed response-validity routing for rejected skill candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CANONICAL_ANSWER_PATTERN = r"<answer>\s*([A-G])\s*</answer>"


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_answer_labels(
    response: str,
    *,
    pattern: str = CANONICAL_ANSWER_PATTERN,
) -> tuple[str, ...]:
    return tuple(match.upper() for match in re.findall(pattern, str(response), flags=re.IGNORECASE))


def has_admissible_answer(
    labels: Sequence[str],
    *,
    require_unique_label: bool,
) -> bool:
    if not labels:
        return False
    return not require_unique_label or len(set(labels)) == 1


def route_responses(
    incumbent_response: str,
    candidate_response: str,
    *,
    require_unique_label: bool = False,
) -> dict[str, Any]:
    incumbent_labels = canonical_answer_labels(incumbent_response)
    candidate_labels = canonical_answer_labels(candidate_response)
    incumbent_admissible = has_admissible_answer(
        incumbent_labels,
        require_unique_label=require_unique_label,
    )
    candidate_admissible = has_admissible_answer(
        candidate_labels,
        require_unique_label=require_unique_label,
    )
    triggered = not incumbent_admissible
    use_candidate = bool(triggered and candidate_admissible)
    return {
        "triggered": triggered,
        "trigger_reason": (
            "incumbent_missing_unique_canonical_answer"
            if triggered and require_unique_label
            else ("incumbent_missing_canonical_answer" if triggered else "incumbent_answer_valid")
        ),
        "candidate_admissible": candidate_admissible,
        "selected_arm": "candidate" if use_candidate else "incumbent",
        "incumbent_answer_labels": list(incumbent_labels),
        "candidate_answer_labels": list(candidate_labels),
        "require_unique_label": require_unique_label,
        "ground_truth_accessed": False,
    }


def route_nested_fallback_responses(
    base_response: str,
    incumbent_fallback_response: str,
    candidate_fallback_response: str,
    *,
    require_unique_label: bool = True,
) -> dict[str, Any]:
    """Escalate to a candidate fallback only after the incumbent route is invalid."""
    base_labels = canonical_answer_labels(base_response)
    incumbent_labels = canonical_answer_labels(incumbent_fallback_response)
    candidate_labels = canonical_answer_labels(candidate_fallback_response)
    base_admissible = has_admissible_answer(
        base_labels,
        require_unique_label=require_unique_label,
    )
    incumbent_admissible = has_admissible_answer(
        incumbent_labels,
        require_unique_label=require_unique_label,
    )
    candidate_admissible = has_admissible_answer(
        candidate_labels,
        require_unique_label=require_unique_label,
    )
    candidate_triggered = not base_admissible and not incumbent_admissible
    if base_admissible:
        selected_arm = "base"
    elif incumbent_admissible:
        selected_arm = "incumbent_fallback"
    elif candidate_admissible:
        selected_arm = "candidate_fallback"
    else:
        selected_arm = "base_fail_closed"
    return {
        "type": "nested_answer_validity_fallback_cascade",
        "selected_arm": selected_arm,
        "candidate_triggered": candidate_triggered,
        "trigger_reason": (
            "base_and_incumbent_fallback_missing_admissible_answer"
            if candidate_triggered
            else "candidate_not_needed"
        ),
        "base_answer_labels": list(base_labels),
        "incumbent_fallback_answer_labels": list(incumbent_labels),
        "candidate_fallback_answer_labels": list(candidate_labels),
        "base_admissible": base_admissible,
        "incumbent_fallback_admissible": incumbent_admissible,
        "candidate_fallback_admissible": candidate_admissible,
        "require_unique_label": require_unique_label,
        "ground_truth_accessed": False,
    }


def paired_outcome_audit(
    reference: Sequence[int | bool],
    candidate: Sequence[int | bool],
    unit_ids: Sequence[str],
) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate) or len(reference) != len(unit_ids):
        raise ValueError("paired outcomes must be aligned and non-empty")
    if len(set(unit_ids)) != len(unit_ids) or any(not str(unit_id) for unit_id in unit_ids):
        raise ValueError("paired outcome unit_ids must be unique and non-empty")
    beneficial = [
        unit_id
        for unit_id, before, after in zip(unit_ids, reference, candidate, strict=True)
        if not bool(before) and bool(after)
    ]
    harmful = [
        unit_id
        for unit_id, before, after in zip(unit_ids, reference, candidate, strict=True)
        if bool(before) and not bool(after)
    ]
    discordant = len(beneficial) + len(harmful)
    one_sided_sign_p = (
        sum(math.comb(discordant, value) for value in range(len(beneficial), discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "both_success": sum(
            bool(before) and bool(after) for before, after in zip(reference, candidate, strict=True)
        ),
        "both_failure": sum(
            not bool(before) and not bool(after)
            for before, after in zip(reference, candidate, strict=True)
        ),
        "beneficial_flip_count": len(beneficial),
        "beneficial_flip_ids": beneficial,
        "harmful_flip_count": len(harmful),
        "harmful_flip_ids": harmful,
        "delta_successes": len(beneficial) - len(harmful),
        "one_sided_exact_sign_p": one_sided_sign_p,
    }


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    unit_ids = tuple(str(row.get("id") or "") for row in rows)
    if not unit_ids or any(not unit_id for unit_id in unit_ids):
        raise ValueError(f"{arm} rows require non-empty ids")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError(f"{arm} row ids must be unique")
    return unit_ids, dict(zip(unit_ids, rows, strict=True))


@dataclass(frozen=True)
class RouterReplay:
    selected_rows: tuple[Mapping[str, Any], ...]
    per_task: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]


def replay_answer_validity_router(
    incumbent_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    require_unique_label: bool = True,
    minimum_beneficial_flips: int = 1,
    maximum_harmful_flips: int = 0,
    require_strict_score_improvement: bool = True,
) -> RouterReplay:
    if minimum_beneficial_flips < 0 or maximum_harmful_flips < 0:
        raise ValueError("router flip gates must be non-negative")
    unit_ids, incumbent_by_id = _index_rows(incumbent_rows, arm="incumbent")
    candidate_ids, candidate_by_id = _index_rows(candidate_rows, arm="candidate")
    if set(unit_ids) != set(candidate_ids):
        raise ValueError("router replay arms must contain the same unit ids")

    selected_rows = []
    per_task = []
    incumbent_outcomes = []
    candidate_outcomes = []
    router_outcomes = []
    for unit_id in unit_ids:
        incumbent = incumbent_by_id[unit_id]
        candidate = candidate_by_id[unit_id]
        decision = route_responses(
            str(incumbent.get("response") or ""),
            str(candidate.get("response") or ""),
            require_unique_label=require_unique_label,
        )
        selected = candidate if decision["selected_arm"] == "candidate" else incumbent
        incumbent_success = int(bool(incumbent.get("hard")))
        candidate_success = int(bool(candidate.get("hard")))
        router_success = int(bool(selected.get("hard")))
        selected_rows.append(selected)
        incumbent_outcomes.append(incumbent_success)
        candidate_outcomes.append(candidate_success)
        router_outcomes.append(router_success)
        per_task.append(
            {
                "task_id": unit_id,
                "incumbent_success": incumbent_success,
                "candidate_success": candidate_success,
                "router_success": router_success,
                "router": decision,
            }
        )

    paired = paired_outcome_audit(incumbent_outcomes, router_outcomes, unit_ids)
    incumbent_successes = sum(incumbent_outcomes)
    candidate_successes = sum(candidate_outcomes)
    router_successes = sum(router_outcomes)
    gates = {
        "minimum_beneficial_flips": (paired["beneficial_flip_count"] >= minimum_beneficial_flips),
        "maximum_harmful_flips": (paired["harmful_flip_count"] <= maximum_harmful_flips),
        "strict_score_improvement": (
            not require_strict_score_improvement or router_successes > incumbent_successes
        ),
        "runtime_ground_truth_absent": all(
            not row["router"]["ground_truth_accessed"] for row in per_task
        ),
    }
    audit = {
        "type": "answer_validity_fallback_cascade",
        "accepted": all(gates.values()),
        "n": len(unit_ids),
        "scores": {
            "incumbent": {
                "successes": incumbent_successes,
                "success_rate": incumbent_successes / len(unit_ids),
            },
            "candidate": {
                "successes": candidate_successes,
                "success_rate": candidate_successes / len(unit_ids),
            },
            "router": {
                "successes": router_successes,
                "success_rate": router_successes / len(unit_ids),
            },
        },
        "paired_vs_incumbent": paired,
        "trigger_count": sum(row["router"]["triggered"] for row in per_task),
        "candidate_admitted_count": sum(
            row["router"]["selected_arm"] == "candidate" for row in per_task
        ),
        "gates": gates,
        "require_unique_label": require_unique_label,
    }
    return RouterReplay(
        selected_rows=tuple(selected_rows),
        per_task=tuple(per_task),
        audit=audit,
    )


@dataclass(frozen=True)
class RouterCascade:
    selected_rows: tuple[Mapping[str, Any], ...]
    incumbent_rows: tuple[Mapping[str, Any], ...]
    candidate_rows: tuple[Mapping[str, Any], ...]
    per_task: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]


def rollout_answer_validity_cascade(
    *,
    items: Sequence[Mapping[str, Any]],
    incumbent_prompt: str,
    candidate_prompt: str,
    output_root: Path,
    rollout: Callable[
        [Sequence[Mapping[str, Any]], str, str],
        Sequence[Mapping[str, Any]],
    ],
    require_unique_label: bool = True,
    minimum_beneficial_flips: int = 1,
    maximum_harmful_flips: int = 0,
    require_strict_score_improvement: bool = True,
) -> RouterCascade:
    item_ids, _ = _index_rows(items, arm="router items")
    incumbent_rows = tuple(rollout(items, incumbent_prompt, str(output_root / "incumbent")))
    incumbent_ids, incumbent_by_id = _index_rows(
        incumbent_rows,
        arm="incumbent rollout",
    )
    if set(incumbent_ids) != set(item_ids):
        raise RuntimeError("incumbent router rollout did not preserve item coverage")

    triggered_ids = {
        unit_id
        for unit_id in item_ids
        if route_responses(
            str(incumbent_by_id[unit_id].get("response") or ""),
            "",
            require_unique_label=require_unique_label,
        )["triggered"]
    }
    triggered_items = [item for item in items if str(item.get("id") or "") in triggered_ids]
    candidate_rows = (
        tuple(rollout(triggered_items, candidate_prompt, str(output_root / "candidate")))
        if triggered_items
        else ()
    )
    if candidate_rows:
        candidate_ids, candidate_by_id = _index_rows(
            candidate_rows,
            arm="candidate rollout",
        )
        if set(candidate_ids) != triggered_ids:
            raise RuntimeError("candidate router rollout did not preserve trigger coverage")
    else:
        candidate_by_id = {}

    selected_rows = []
    per_task = []
    incumbent_outcomes = []
    router_outcomes = []
    for unit_id in item_ids:
        incumbent = incumbent_by_id[unit_id]
        candidate = candidate_by_id.get(unit_id)
        decision = route_responses(
            str(incumbent.get("response") or ""),
            str((candidate or {}).get("response") or ""),
            require_unique_label=require_unique_label,
        )
        selected = (
            candidate
            if candidate is not None and decision["selected_arm"] == "candidate"
            else incumbent
        )
        incumbent_success = int(bool(incumbent.get("hard")))
        router_success = int(bool(selected.get("hard")))
        selected_rows.append(selected)
        incumbent_outcomes.append(incumbent_success)
        router_outcomes.append(router_success)
        per_task.append(
            {
                "task_id": unit_id,
                "incumbent_success": incumbent_success,
                "candidate_success": (
                    int(bool(candidate.get("hard"))) if candidate is not None else None
                ),
                "router_success": router_success,
                "router": decision,
            }
        )

    paired = paired_outcome_audit(incumbent_outcomes, router_outcomes, item_ids)
    incumbent_successes = sum(incumbent_outcomes)
    router_successes = sum(router_outcomes)
    gates = {
        "minimum_beneficial_flips": (paired["beneficial_flip_count"] >= minimum_beneficial_flips),
        "maximum_harmful_flips": (paired["harmful_flip_count"] <= maximum_harmful_flips),
        "strict_score_improvement": (
            not require_strict_score_improvement or router_successes > incumbent_successes
        ),
        "runtime_ground_truth_absent": all(
            not row["router"]["ground_truth_accessed"] for row in per_task
        ),
        "candidate_called_only_after_trigger": len(candidate_rows) == len(triggered_ids),
    }
    audit = {
        "type": "answer_validity_fallback_cascade",
        "accepted": all(gates.values()),
        "n": len(item_ids),
        "scores": {
            "incumbent": {
                "successes": incumbent_successes,
                "success_rate": incumbent_successes / len(item_ids),
            },
            "router": {
                "successes": router_successes,
                "success_rate": router_successes / len(item_ids),
            },
        },
        "paired_vs_incumbent": paired,
        "trigger_count": len(triggered_ids),
        "candidate_admitted_count": sum(
            row["router"]["selected_arm"] == "candidate" for row in per_task
        ),
        "model_calls": len(item_ids) + len(triggered_ids),
        "mean_model_calls": (len(item_ids) + len(triggered_ids)) / len(item_ids),
        "gates": gates,
        "require_unique_label": require_unique_label,
    }
    return RouterCascade(
        selected_rows=tuple(selected_rows),
        incumbent_rows=incumbent_rows,
        candidate_rows=candidate_rows,
        per_task=tuple(per_task),
        audit=audit,
    )


def routed_artifact_sha256(
    *,
    incumbent_skill_tree_sha256: str,
    candidate_skill_tree_sha256: str,
    router: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "artifact_type": "answer_validity_routed_skill_cascade",
            "incumbent_skill_tree_sha256": incumbent_skill_tree_sha256,
            "candidate_skill_tree_sha256": candidate_skill_tree_sha256,
            "router": dict(router),
        }
    )


__all__ = [
    "CANONICAL_ANSWER_PATTERN",
    "RouterCascade",
    "RouterReplay",
    "canonical_answer_labels",
    "canonical_json_sha256",
    "has_admissible_answer",
    "paired_outcome_audit",
    "replay_answer_validity_router",
    "rollout_answer_validity_cascade",
    "route_nested_fallback_responses",
    "route_responses",
    "routed_artifact_sha256",
]
