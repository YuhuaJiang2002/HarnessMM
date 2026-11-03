"""Dependency-free lexical retrieval shared by CHISEL adapters."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import heapq
import math
import re
from typing import Any


WORD_PATTERN = re.compile(r"[a-z0-9]+")


def stable_feature(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


def text_features(question: str, *, mode: str) -> set[int]:
    tokens = WORD_PATTERN.findall(question.casefold())
    features = {stable_feature(f"w:{token}") for token in tokens}
    features.update(stable_feature(f"b:{left}_{right}") for left, right in zip(tokens, tokens[1:]))
    if tokens:
        features.add(stable_feature(f"first:{tokens[0]}"))
        features.add(stable_feature("prefix2:" + "_".join(tokens[:2])))
    features.add(stable_feature(f"length:{min(len(tokens) // 4, 8)}"))
    features.add(stable_feature(f"quote:{int(chr(34) in question or chr(39) in question)}"))
    features.add(
        stable_feature(f"number:{int(any(character.isdigit() for character in question))}")
    )
    if mode == "hybrid":
        normalized = "_".join(tokens)
        for size in (4, 5):
            features.update(
                stable_feature(f"c{size}:{normalized[index : index + size]}")
                for index in range(0, max(0, len(normalized) - size + 1), 2)
            )
    elif mode != "word":
        raise ValueError(f"unknown retrieval feature mode: {mode}")
    return features


class SparseTextIndex:
    """Fit the vocabulary and IDF on training text, never on query outcomes."""

    def __init__(self, texts: Sequence[str], *, mode: str = "word") -> None:
        self.mode = mode
        feature_sets = [text_features(text, mode=mode) for text in texts]
        document_frequency: Counter[int] = Counter()
        for features in feature_sets:
            document_frequency.update(features)
        count = len(feature_sets)
        self.idf = {
            feature: math.log((count + 1) / (frequency + 1)) + 1.0
            for feature, frequency in document_frequency.items()
            if frequency >= 2 and frequency <= max(8, int(0.25 * count))
        }
        self.postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for index, features in enumerate(feature_sets):
            weighted = {feature: self.idf[feature] for feature in features if feature in self.idf}
            norm = math.sqrt(sum(weight * weight for weight in weighted.values()))
            if not norm:
                continue
            for feature, weight in weighted.items():
                self.postings[feature].append((index, weight / norm))

    def neighbors(self, question: str, *, maximum: int = 64) -> list[tuple[float, int]]:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("maximum neighbors must be a positive integer")
        features = text_features(question, mode=self.mode)
        weighted = {feature: self.idf[feature] for feature in features if feature in self.idf}
        norm = math.sqrt(sum(weight * weight for weight in weighted.values()))
        if not norm:
            return []
        scores: dict[int, float] = defaultdict(float)
        for feature, weight in weighted.items():
            query_weight = weight / norm
            for index, document_weight in self.postings[feature]:
                scores[index] += query_weight * document_weight
        return heapq.nlargest(
            maximum, ((score, index) for index, score in scores.items() if score > 0)
        )


class SparseRetriever(SparseTextIndex):
    """Compatibility adapter for the original SearchQA discovery scripts."""

    def __init__(self, records: Sequence[Mapping[str, Any]], *, mode: str) -> None:
        self.records = list(records)
        super().__init__([str(record["question"]) for record in records], mode=mode)

    def neighbors(
        self, question: str, *, mode: str | None = None, maximum: int = 64
    ) -> list[tuple[float, int]]:
        if mode is not None and mode != self.mode:
            raise ValueError("query feature mode must match the fitted index")
        return super().neighbors(question, maximum=maximum)
