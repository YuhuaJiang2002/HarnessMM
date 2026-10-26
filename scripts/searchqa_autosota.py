"""TRAIN-fitted answer-support ranking with marginal-cost query coverage."""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Callable

from scripts import figure2_searchqa_ranker as ranker
from scripts import figure2_searchqa_recovery as recovery
from scripts.searchqa_span_union import union_spans


SCHEMA = "searchqa-autosota-answer-support/v1"
FEATURES = 20
STRATEGIES = ("trained_greedy", "coverage")
TokenCounter = Callable[[list[dict]], int]


def features(question: str, text: str, position: float,
             frequencies: Counter[str]) -> list[float]:
    query = set(ranker.words(question)) - ranker.STOP
    source = set(ranker.words(text))
    weights = {word: 1 / math.sqrt(max(frequencies[word], 1)) for word in query}
    rare_coverage = sum(weights[word] for word in query & source) / max(
        sum(weights.values()), 1e-12,
    )
    numeric = bool(re.search(r"\b(?:when|year|date|how many|how much)\b", question, re.I))
    person = bool(re.search(r"\b(?:who|author|writer|painter|composer)\b", question, re.I))
    relation = bool(re.search(
        r"\b(?:born|wrote|founded|located|married|played|named|called|capital)\b", text, re.I,
    ))
    return ranker.features(question, text, position) + [
        rare_coverage,
        float(numeric and bool(re.search(r"\d", text))),
        float(person and bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))),
        float(relation),
        float(bool(re.search(r"\b(?:answer|solution)\s*[:=]", text, re.I))),
        float(bool(re.search(r"\b(?:not|except|before|after)\b", question, re.I))),
        float(bool(query) and len(query & source) >= min(3, len(query))),
        float(bool(re.search(r"\b(?:also known|better known|full name)\b", text, re.I))),
    ]


def probability(weights: list[float], vector: list[float]) -> float:
    if len(weights) != FEATURES or len(vector) != FEATURES:
        raise ValueError("Answer-support feature dimensions differ")
    if not all(math.isfinite(value) for value in weights + vector):
        raise ValueError("Answer-support features and weights must be finite")
    score = sum(weight * value for weight, value in zip(weights, vector))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, score))))


def fit(items: list[dict], labels: dict[str, list[str]], *, epochs: int = 5) -> dict:
    identities = [item["id"] for item in items]
    if (len(set(identities)) != len(identities) or set(identities) != set(labels)
            or type(epochs) is not int or epochs < 1):
        raise ValueError("Fit requires unique, exactly matched TRAIN identities and positive epochs")
    pairs = []
    for item in sorted(items, key=lambda row: row["id"]):
        context = item["context"]
        frequencies = Counter(ranker.words(context))
        needles = [" ".join(ranker.words(answer)) for answer in labels[item["id"]]]
        needles = [needle for needle in needles if needle]
        positive, negative = [], []
        for start, end in ranker.units(context):
            text = context[start:end]
            vector = features(item["question"], text, start / max(len(context), 1), frequencies)
            normalized = " " + " ".join(ranker.words(text)) + " "
            target = positive if any(" " + needle + " " in normalized for needle in needles) else negative
            target.append(vector)
        for supported in positive[:4]:
            for distractor in sorted(negative, key=lambda vector: (-vector[12], -vector[1]))[:4]:
                pairs.append([a - b for a, b in zip(supported, distractor)])
    if not pairs:
        raise ValueError("TRAIN has no answer-support ranking pairs")
    weights = [0.0] * FEATURES
    for _ in range(epochs):
        for vector in pairs:
            error = 1 - probability(weights, vector)
            weights = [
                weight + 0.025 * (error * value - 0.0005 * weight)
                for weight, value in zip(weights, vector)
            ]
    return {
        "schema": SCHEMA, "weights": weights, "features": FEATURES,
        "training_ids": sorted(labels), "training_questions": len(items),
        "pairs": len(pairs), "epochs": epochs, "coverage_weight": 0.35,
        "strategy": "coverage", "optimizer_model_calls": 0,
        "labels_from_validation_or_test": False,
        "inference_uses_answers_or_task_ids": False,
        "identity": "AutoSOTA-inspired TRAIN answer-support and evidence-allocation adaptation",
    }


def request(item: dict, predictor: dict, count: TokenCounter, selection_cap: int = 640,
            extra_instruction: str = "") -> tuple[list[dict], dict]:
    strategy = predictor.get("strategy")
    coverage_weight = predictor.get("coverage_weight")
    if (predictor.get("schema") != SCHEMA or strategy not in STRATEGIES
            or predictor.get("labels_from_validation_or_test") is not False
            or predictor.get("inference_uses_answers_or_task_ids") is not False
            or type(selection_cap) is not int or selection_cap < 1
            or not isinstance(coverage_weight, (int, float))
            or not math.isfinite(coverage_weight) or coverage_weight < 0):
        raise ValueError("A frozen TRAIN-only answer-support allocation policy is required")
    context, question = item["context"], item["question"]
    frequencies = Counter(ranker.words(context))
    query = set(ranker.words(question)) - ranker.STOP
    query_weights = {word: 1 / math.sqrt(max(frequencies[word], 1)) for word in query}
    candidates = []
    for start, end in ranker.units(context):
        text = context[start:end]
        candidates.append((
            probability(predictor["weights"], features(
                question, text, start / max(len(context), 1), frequencies,
            )), start, end, set(ranker.words(text)) & query,
        ))

    def messages(spans: list[tuple[int, int]], system: str) -> list[dict]:
        source = "\n...\n".join(context[start:end] for start, end in spans)
        return ranker.messages(item, source, system + extra_instruction)

    selected: list[tuple[int, int]] = []
    admitted: list[tuple[int, int]] = []
    covered: set[str] = set()
    current_tokens = count(messages([], recovery.CONTRACT_SYSTEM))
    while candidates:
        options = []
        for support, start, end, clues in candidates:
            merged = union_spans(selected + [(start, end)])
            if merged == selected:
                continue
            tokens = count(messages(merged, recovery.CONTRACT_SYSTEM))
            if tokens > selection_cap:
                continue
            new_characters = (
                sum(b - a for a, b in merged) - sum(b - a for a, b in selected)
            )
            novelty = sum(query_weights[word] for word in clues - covered) / max(
                sum(query_weights.values()), 1e-12,
            )
            marginal = max(tokens - current_tokens, 1)
            score = support if strategy == "trained_greedy" else (
                support * math.sqrt(new_characters / (end - start))
                + coverage_weight * novelty
            ) / math.sqrt(marginal)
            options.append((score, -start, start, end, merged, tokens, clues))
        if not options:
            break
        _, _, start, end, selected, current_tokens, clues = max(options, key=lambda row: row[:2])
        admitted.append((start, end))
        covered.update(clues)
        candidates = [row for row in candidates if row[1:3] != (start, end)]
    if not selected:
        raise ValueError("Question and one intact source span exceed the frozen input cap")
    output = messages(selected, recovery.TARGET_ROLE_SYSTEM)
    return output, {
        "spans": selected, "context_sha256": ranker.digest(context), "source_only": True,
        "source_selection_input_cap": selection_cap,
        "actual_input_tokens": count(output), "source_selection_input_tokens": current_tokens,
        "selection_model_calls": 0, "admitted_windows": admitted,
        "overlap_characters_not_repeated": (
            sum(end - start for start, end in admitted)
            - sum(end - start for start, end in selected)
        ),
        "mechanism": "train_answer_support_exact_union_" + strategy,
        "covered_query_clues": sorted(covered), "query_clues": sorted(query),
        "extra_instruction_tokens_paid": bool(extra_instruction),
        "inference_uses_answers_or_task_ids": False,
    }
