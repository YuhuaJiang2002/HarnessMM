"""Ground-truth-free routing for fallback disagreements involving META options."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from phiagent.learning.answer_validity_router import (
    CANONICAL_ANSWER_PATTERN,
    canonical_answer_labels,
    has_admissible_answer,
)


META_OPTION_PATTERN = re.compile(
    r"^one of the remaining options is correct,? but a stronger result can be proven\.?$",
    flags=re.IGNORECASE,
)


def normalize_choice_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def meta_option_label(choices: Sequence[Mapping[str, Any]]) -> str | None:
    labels = [
        str(choice.get("label") or "").strip().upper()
        for choice in choices
        if META_OPTION_PATTERN.fullmatch(normalize_choice_text(str(choice.get("text") or "")))
    ]
    unique = tuple(dict.fromkeys(label for label in labels if label))
    return unique[0] if len(unique) == 1 else None


def _admissible_label(response: str) -> tuple[bool, str | None, tuple[str, ...]]:
    labels = canonical_answer_labels(response)
    admissible = has_admissible_answer(labels, require_unique_label=True)
    label = next(iter(set(labels))) if admissible else None
    return admissible, label, labels


def visible_reasoning_chars(response: str) -> int:
    without_answer = re.sub(
        CANONICAL_ANSWER_PATTERN,
        "",
        str(response),
        flags=re.IGNORECASE,
    )
    return len(" ".join(without_answer.split()))


def route_meta_disagreement_responses(
    *,
    choices: Sequence[Mapping[str, Any]],
    base_response: str,
    incumbent_fallback_response: str,
    candidate_fallback_response: str,
    candidate_override_min_length_ratio: float | None = None,
) -> dict[str, Any]:
    """Prefer a META-selecting candidate only in an otherwise-valid disagreement."""

    if (
        candidate_override_min_length_ratio is not None
        and candidate_override_min_length_ratio <= 1
    ):
        raise ValueError("candidate override length ratio must be greater than one")
    base_ok, base_label, base_labels = _admissible_label(base_response)
    incumbent_ok, incumbent_label, incumbent_labels = _admissible_label(incumbent_fallback_response)
    candidate_ok, candidate_label, candidate_labels = _admissible_label(candidate_fallback_response)
    meta_label = meta_option_label(choices)
    incumbent_reasoning_chars = visible_reasoning_chars(incumbent_fallback_response)
    candidate_reasoning_chars = visible_reasoning_chars(candidate_fallback_response)
    candidate_to_incumbent_length_ratio = (
        candidate_reasoning_chars / incumbent_reasoning_chars
        if incumbent_reasoning_chars
        else (float("inf") if candidate_reasoning_chars else 1.0)
    )
    length_override_supported = bool(
        candidate_override_min_length_ratio is not None
        and candidate_ok
        and incumbent_ok
        and candidate_label != incumbent_label
        and candidate_to_incumbent_length_ratio >= candidate_override_min_length_ratio
    )
    candidate_evaluated = bool(
        not base_ok and (not incumbent_ok or meta_label is None or incumbent_label != meta_label)
    )
    if base_ok:
        selected_arm = "base"
        reason = "base_answer_valid"
    elif not incumbent_ok:
        if candidate_ok:
            selected_arm = "candidate_fallback"
            reason = "incumbent_invalid_candidate_valid"
        else:
            selected_arm = "base_fail_closed"
            reason = "all_arms_invalid"
    elif (
        candidate_ok
        and candidate_label != incumbent_label
        and meta_label is not None
        and candidate_label == meta_label
        and incumbent_label != meta_label
    ):
        selected_arm = "candidate_fallback"
        reason = "candidate_selects_explicit_meta_option"
    elif length_override_supported:
        selected_arm = "candidate_fallback"
        reason = "candidate_has_dominant_visible_reasoning_length"
    else:
        selected_arm = "incumbent_fallback"
        reason = (
            "incumbent_and_candidate_agree"
            if candidate_ok and candidate_label == incumbent_label
            else "no_supported_candidate_override"
        )
    return {
        "type": "meta_option_disagreement_fallback_cascade",
        "selected_arm": selected_arm,
        "selection_reason": reason,
        "base_admissible": base_ok,
        "incumbent_fallback_admissible": incumbent_ok,
        "candidate_fallback_admissible": candidate_ok,
        "base_answer_labels": list(base_labels),
        "incumbent_fallback_answer_labels": list(incumbent_labels),
        "candidate_fallback_answer_labels": list(candidate_labels),
        "meta_option_label": meta_label,
        "incumbent_visible_reasoning_chars": incumbent_reasoning_chars,
        "candidate_visible_reasoning_chars": candidate_reasoning_chars,
        "candidate_to_incumbent_length_ratio": candidate_to_incumbent_length_ratio,
        "candidate_override_min_length_ratio": candidate_override_min_length_ratio,
        "length_override_supported": length_override_supported,
        "candidate_evaluated": candidate_evaluated,
        "ground_truth_accessed": False,
        "outcome_features_accessed": False,
    }


__all__ = [
    "META_OPTION_PATTERN",
    "meta_option_label",
    "normalize_choice_text",
    "route_meta_disagreement_responses",
    "visible_reasoning_chars",
]
