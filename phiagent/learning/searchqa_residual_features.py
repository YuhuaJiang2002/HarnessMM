"""Outcome-blind SearchQA features for residual Meta-Harness V2."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Mapping, Sequence


FEATURE_NAMES = (
    "question_token_count",
    "question_named_token_count",
    "question_has_quoted_span",
    "question_has_relation_cue",
    "question_context_coverage",
    "context_token_count",
    "document_count",
    "expected_answer_numeric",
)
_BOOLEAN_FEATURES = frozenset(
    {
        "question_has_quoted_span",
        "question_has_relation_cue",
        "expected_answer_numeric",
    }
)
_RELATION_CUE = re.compile(
    r"\b(?:after|before|born|capital|daughter|father|founded|husband|mother|"
    r"named after|preceded|sister|son|succeeded|wife)\b",
    re.IGNORECASE,
)


def _normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def extract_features(
    task: Mapping[str, Any],
) -> dict[str, bool | int | float]:
    """Extract the complete frozen feature vector without IDs or outcomes."""

    if not isinstance(task, Mapping) or not {"question", "context"} <= set(task):
        raise ValueError("SearchQA task lacks question or context")
    question = task["question"]
    context = task["context"]
    if not isinstance(question, str) or not isinstance(context, str):
        raise TypeError("SearchQA question and context must be strings")
    question_normalized = _normalized_text(question)
    context_normalized = _normalized_text(context)
    question_tokens = set(question_normalized.split())
    context_tokens = set(context_normalized.split())
    content_tokens = {
        token
        for token in question_tokens
        if len(token) > 2
        and token not in {"and", "for", "that", "the", "this", "with"}
    }
    coverage = (
        sum(token in context_tokens for token in content_tokens)
        / len(content_tokens)
        if content_tokens
        else 0.0
    )
    expected_numeric = bool(
        re.search(
            r"\b(?:how many|how old|what date|what percentage|what year|"
            r"which year)\b",
            question_normalized,
        )
    )
    features: dict[str, bool | int | float] = {
        "question_token_count": len(question_normalized.split()),
        "question_named_token_count": sum(
            token[:1].isupper()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9'.-]*", question)
        ),
        "question_has_quoted_span": bool(
            re.search(r"""["'][^"']+["']""", question)
        ),
        "question_has_relation_cue": bool(_RELATION_CUE.search(question)),
        "question_context_coverage": coverage,
        "context_token_count": len(context_normalized.split()),
        "document_count": max(1, context.count("[DOC]")) if context else 0,
        "expected_answer_numeric": expected_numeric,
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("SearchQA residual feature order changed")
    return features


def build_feature_schema(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | str]]:
    """Freeze numeric normalization from outcome-blind discovery inputs."""

    if not tasks:
        raise ValueError("cannot build a feature schema without tasks")
    rows = [extract_features(task) for task in tasks]
    schema: dict[str, dict[str, float | str]] = {}
    for name in FEATURE_NAMES:
        if name in _BOOLEAN_FEATURES:
            schema[name] = {"kind": "boolean"}
            continue
        values = [float(row[name]) for row in rows]
        mean = statistics.fmean(values)
        scale = statistics.pstdev(values)
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        schema[name] = {
            "kind": "numeric",
            "mean": mean,
            "scale": scale,
            "minimum": min(values),
            "maximum": max(values),
        }
    return schema


__all__ = ["FEATURE_NAMES", "build_feature_schema", "extract_features"]
