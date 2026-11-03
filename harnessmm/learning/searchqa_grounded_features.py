"""Case-insensitive lexical anchors, not named entities or factual support."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math
import re
from types import MappingProxyType


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_SOURCE_BOUNDARY = re.compile(r"\[(?:DOC|SEP)\]", re.IGNORECASE)
_CONTENT_STOPWORDS = frozenset({"the", "and", "for"})
NAMED_FEATURE_FIELDS = MappingProxyType({
    "capitalized": "question_named_token_count",
    "omit": None,
    "grounded_count": "question_grounded_anchor_count",
    "grounded_rarity": "question_grounded_anchor_mass",
})


def named_feature_field(mode: str) -> str | None:
    if not isinstance(mode, str) or mode not in NAMED_FEATURE_FIELDS:
        raise ValueError(f"unknown named-feature mode: {mode!r}")
    return NAMED_FEATURE_FIELDS[mode]


def content_terms(tokens: Sequence[str]) -> set[str]:
    return {
        token.casefold()
        for token in tokens
        if len(token) > 2 and token.casefold() not in _CONTENT_STOPWORDS
    }


@dataclass(frozen=True, slots=True)
class GroundedAnchors:
    query_terms: tuple[str, ...]
    source_group_count: int
    document_frequencies: tuple[tuple[str, int], ...]
    grounded_count: int
    rarity_mass: float


def grounded_anchors(question: str, context: str) -> GroundedAnchors:
    """Count distinct question terms supported by source-local word matches.

    A matched term has weight log((D + 1) / df) / log(D + 1).
    Publisher delimiter groups are not independent or verified web sources.
    """
    if not isinstance(question, str) or not isinstance(context, str):
        raise TypeError("question and context must be strings")
    query = tuple(sorted(content_terms(TOKEN_PATTERN.findall(question))))
    documents = [
        {token.casefold() for token in TOKEN_PATTERN.findall(piece)}
        for piece in _SOURCE_BOUNDARY.split(context)
        if piece.strip()
    ]
    documents = [terms for terms in documents if terms]
    frequency = Counter(term for terms in documents for term in terms)
    matched = tuple((term, frequency[term]) for term in query if frequency[term])
    count = len(documents)
    mass = (
        math.fsum(
            math.log((count + 1) / occurrences) / math.log(count + 1)
            for _, occurrences in matched
        )
        if count
        else 0.0
    )
    return GroundedAnchors(query, count, matched, len(matched), mass)
