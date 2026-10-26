"""Provisional input-only routing over frozen QA profiles; no quality certificate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
import unicodedata

from phiagent.learning.residual_meta_harness import canonical_hash


POLICY_SCHEMA = "phiagent-qa-quality-cost-router/v1"
RISK_POLICY_SCHEMA = "phiagent-qa-quality-cost-router/v2"
HASH_BUCKETS = 128
QUESTION_WORD_LIMIT = 256
CONTEXT_CHARACTER_LIMIT = 16384
NUMERIC_FEATURES = (
    "question_log_characters", "question_log_words", "context_log_characters",
    "context_log_words", "document_log_count", "query_word_coverage",
    "quoted_question", "numeric_question", "person_question", "group_question",
    "relation_question", "form_question", "fragment_question", "list_question",
    "question_exactly_in_context", "pipe_format_context", "answer_field_context",
    "question_has_year", "feature_context_clipped", "context_distinct_word_fraction",
)
WORD = re.compile(r"\w+", re.UNICODE)
TITLE = re.compile(r"\[tle\](.*?)\[par\]", re.DOTALL)
STOP_WORDS = frozenset(
    "a an the this that these those is are was were be been in on at of for to from "
    "and or as by with it its who what which where when how his her their".split()
)


def feature_spec(mode: str) -> dict:
    if mode not in {"numeric", "lexical"}:
        raise ValueError("feature mode must be numeric or lexical")
    return {
        "schema_version": "phiagent-qa-input-only-features/v1",
        "mode": mode, "numeric_features": list(NUMERIC_FEATURES),
        "hash_buckets": HASH_BUCKETS if mode == "lexical" else 0,
        "lexical_blocks": ["question_words", "question_bigrams", "source_title_words"]
        if mode == "lexical" else [],
        "normalization": "NFKC plus casefold; signed SHA256 hashing and within-block L2",
        "length_features": "original character lengths; normalized word statistics",
        "question_word_limit": QUESTION_WORD_LIMIT,
        "context_character_limit": CONTEXT_CHARACTER_LIMIT,
        "actor_context_is_modified": False,
        "outcomes_predictions_answers_task_ids_are_features": False,
    }


def feature_dimension(mode: str) -> int:
    feature_spec(mode)
    return len(NUMERIC_FEATURES) + (3 * HASH_BUCKETS if mode == "lexical" else 0)


def _hash_block(terms: Sequence[str], namespace: str) -> list[float]:
    values = [0.0] * HASH_BUCKETS
    for term, count in sorted(Counter(terms).items()):
        digest = hashlib.sha256(f"{namespace}\0{term}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % HASH_BUCKETS
        sign = 1 if digest[8] & 1 else -1
        values[bucket] += sign * (1 + math.log(count))
    norm = math.sqrt(math.fsum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def input_features(*, question: str, context: str, mode: str = "lexical") -> list[float]:
    feature_spec(mode)
    if any(not isinstance(text, str) or not text.strip() for text in (question, context)):
        raise ValueError("nonempty original question and context strings are required")
    query = unicodedata.normalize("NFKC", question).casefold()
    source = unicodedata.normalize("NFKC", context[:CONTEXT_CHARACTER_LIMIT]).casefold()
    query_words = WORD.findall(query)[:QUESTION_WORD_LIMIT]
    source_words = WORD.findall(source)
    query_set = set(query_words) - STOP_WORDS
    source_set = set(source_words)
    coverage = len(query_set & source_set) / len(query_set) if query_set else 0.0
    first = query_words[0] if query_words else ""
    values = [
        math.log1p(len(question)), math.log1p(len(query_words)),
        math.log1p(len(context)), math.log1p(len(source_words)),
        math.log1p(source.count("[doc]")),
        coverage,
        float(any(mark in query for mark in ('"', "\u201c", "\u201d"))),
        float(bool(re.search(r"\b(?:how many|how much|this many|number of|count|age)\b", query))),
        float(bool(re.search(r"\b(?:who|person|man|woman|actor|author|surname)\b", query))),
        float(bool(re.search(r"\b(?:group|court|organization|organisation|band|company|country)\b", query))),
        float(bool(re.search(r"\b(?:his|her|their|wife|husband|father|mother|son|daughter|brother)\b", query))),
        float(bool(re.search(r"\b(?:letter|letters|abbreviation|acronym|name|term|word|means)\b", query))),
        float(first not in {"who", "what", "when", "where", "why", "how", "which"}),
        float(query.count(",") >= 2 or "/" in query),
        float(" ".join(query.split()) in " ".join(source.split())),
        float("|" in source),
        float("right:" in source or "answer:" in source or "answer |" in source),
        float(bool(re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", query))),
        float(len(context) > CONTEXT_CHARACTER_LIMIT),
        len(source_set) / len(source_words) if source_words else 0.0,
    ]
    if mode == "lexical":
        bigrams = [" ".join(pair) for pair in zip(query_words, query_words[1:])]
        title_words = [word for title in TITLE.findall(source) for word in WORD.findall(title)]
        values.extend(_hash_block(query_words, "question"))
        values.extend(_hash_block(bigrams, "bigram"))
        values.extend(_hash_block(title_words, "title"))
    if len(values) != feature_dimension(mode) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("input-only feature construction returned invalid values")
    return values


def _finite_vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list) or len(value) != size
        or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{label} must contain exactly {size} finite numbers")
    return tuple(float(item) for item in value)


class QualityCostRouter:
    """A frozen CPU-only predictor; inference never accepts task outcomes."""

    def __init__(self, artifact: Mapping[str, object]) -> None:
        self._artifact = json.loads(json.dumps(dict(artifact), allow_nan=False))
        value = self._artifact
        if value.get("schema_version") not in {POLICY_SCHEMA, RISK_POLICY_SCHEMA}:
            raise ValueError("unsupported QA quality/cost policy schema")
        if value["schema_version"] == RISK_POLICY_SCHEMA and "decision_rule" not in value:
            raise ValueError("the risk-aware policy must declare its decision rule")
        self.decision_rule = (
            "weighted_sum" if value["schema_version"] == POLICY_SCHEMA else value["decision_rule"]
        )
        if self.decision_rule not in {"weighted_sum", "worst_quality_regret"}:
            raise ValueError("unknown frozen quality/cost decision rule")
        if value["schema_version"] == POLICY_SCHEMA and value.get("decision_rule", "weighted_sum") != "weighted_sum":
            raise ValueError("the original policy schema cannot silently change its decision rule")
        self.mode = value["feature_mode"]
        if value["feature_spec"] != feature_spec(self.mode):
            raise ValueError("the frozen feature implementation/schema changed")
        profiles = value["profiles"]
        if (
            not isinstance(profiles, list) or len(profiles) < 2
            or any(not isinstance(name, str) or not name for name in profiles)
            or len(profiles) != len(set(profiles)) or "answer_only" not in profiles
        ):
            raise ValueError("unique frozen profiles including answer_only are required")
        self.profiles = tuple(profiles)
        if self.decision_rule == "worst_quality_regret" and len(self.profiles) != 2:
            raise ValueError("worst-quality-regret routing requires one cheap and one full profile")
        dimension, arms = feature_dimension(self.mode), len(self.profiles)
        self.mean = _finite_vector(value["feature_mean"], dimension, "feature mean")
        self.scale = _finite_vector(value["feature_scale"], dimension, "feature scale")
        if any(item <= 0 for item in self.scale):
            raise ValueError("feature scales must be positive")
        self.em_intercepts = _finite_vector(value["em_intercepts"], arms, "EM intercepts")
        self.f1_intercepts = _finite_vector(value["f1_intercepts"], arms, "F1 intercepts")
        self.em_weights = self._weights(value["em_weights"], arms, dimension, "EM weights")
        self.f1_weights = self._weights(value["f1_weights"], arms, dimension, "F1 weights")
        self.completion_means = _finite_vector(
            value["completion_means"], arms, "training-only completion means",
        )
        if any(item < 0 for item in self.completion_means):
            raise ValueError("completion estimates cannot be negative")
        for key in ("cost_penalty", "f1_weight", "training_auto_mean"):
            if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
                raise ValueError(f"{key} must be a finite number")
        if value["cost_penalty"] < 0 or value["f1_weight"] < 0 or value["training_auto_mean"] <= 0:
            raise ValueError("invalid cost/quality objective scales")
        self.policy_sha256 = canonical_hash(value)

    @property
    def artifact(self) -> dict:
        return json.loads(json.dumps(self._artifact, allow_nan=False))

    @staticmethod
    def _weights(value: object, arms: int, dimension: int, label: str) -> tuple:
        if not isinstance(value, list) or len(value) != arms:
            raise ValueError(f"{label} has a wrong profile dimension")
        return tuple(_finite_vector(row, dimension, label) for row in value)

    def choose(
        self, *, question: str, context: str, prompt_tokens: Mapping[str, int],
    ) -> dict:
        if set(prompt_tokens) != set(self.profiles) or any(
            type(value) is not int or value <= 0 for value in prompt_tokens.values()
        ):
            raise ValueError("exact positive input-only prompt counts are required for every profile")
        features = input_features(question=question, context=context, mode=self.mode)
        vector = [(value - mean) / scale for value, mean, scale in
                  zip(features, self.mean, self.scale, strict=True)]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("feature normalization overflowed the frozen policy")
        estimates = {}
        for index, profile in enumerate(self.profiles):
            raw_em = self.em_intercepts[index] + math.fsum(
                weight * feature for weight, feature in zip(self.em_weights[index], vector, strict=True)
            )
            raw_f1 = self.f1_intercepts[index] + math.fsum(
                weight * feature for weight, feature in zip(self.f1_weights[index], vector, strict=True)
            )
            if not math.isfinite(raw_em) or not math.isfinite(raw_f1):
                raise ValueError("the frozen quality predictor returned a nonfinite estimate")
            delta_em, delta_f1 = max(-1.0, min(1.0, raw_em)), max(-1.0, min(1.0, raw_f1))
            estimated_tokens = prompt_tokens[profile] + self.completion_means[index]
            quality_term = (
                delta_em + self._artifact["f1_weight"] * delta_f1
                if self.decision_rule == "weighted_sum"
                else max(delta_em, self._artifact["f1_weight"] * delta_f1)
            )
            utility = quality_term - (
                self._artifact["cost_penalty"] * estimated_tokens / self._artifact["training_auto_mean"]
            )
            estimates[profile] = {
                "predicted_delta_em": delta_em, "predicted_delta_f1": delta_f1,
                "raw_predicted_delta_em": raw_em, "raw_predicted_delta_f1": raw_f1,
                "prompt_tokens": prompt_tokens[profile],
                "estimated_completion_tokens": self.completion_means[index],
                "estimated_total_tokens": estimated_tokens, "utility": utility,
            }
        selected = max(
            self.profiles,
            key=lambda profile: (estimates[profile]["utility"],
                                 -estimates[profile]["estimated_total_tokens"],
                                 -self.profiles.index(profile)),
        )
        return {
            "schema_version": (
                "phiagent-qa-quality-cost-route/v2"
                if self._artifact["schema_version"] == RISK_POLICY_SCHEMA
                else "phiagent-qa-quality-cost-route/v1"
            ),
            "selected_profile": selected, "estimates": estimates,
            "policy_sha256": self.policy_sha256, "feature_sha256": canonical_hash(features),
            **({"decision_rule": self.decision_rule}
               if self._artifact["schema_version"] == RISK_POLICY_SCHEMA else {}),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "feature_context_characters": min(len(context), CONTEXT_CHARACTER_LIMIT),
            "actor_context_modified": False, "runtime_outcomes_or_actor_predictions_used": False,
            "router_model_calls": 0, "router_provider_tokens": 0,
            "quality_prediction_is_certificate": False,
        }


def fit_router(
    examples: Sequence[Mapping], *, profiles: Sequence[str], feature_mode: str,
    ridge: float, cost_penalty: float, f1_weight: float, seed: int,
    decision_rule: str = "weighted_sum",
) -> dict:
    """Fit group-weighted residual EM/F1 heads using only the supplied training rows."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Router fitting requires the optional phiagent[qa-routing] dependencies") from error
    if (
        not examples or type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0
        or type(seed) is not int or len(profiles) != len(set(profiles))
        or "answer_only" not in profiles or len(profiles) < 2
        or decision_rule not in {"weighted_sum", "worst_quality_regret"}
        or (decision_rule == "worst_quality_regret" and len(profiles) != 2)
    ):
        raise ValueError("complete training rows, unique profiles and a positive ridge are required")
    names = tuple(profiles)
    identities, group_counts = set(), Counter()
    feature_rows, targets, weights, completion_rows, auto_costs = [], [], [], [], []
    for row in examples:
        if set(row) != {"id", "group", "question", "context", "scores"}:
            raise ValueError("training rows must explicitly separate original inputs and scores")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in identities:
            raise ValueError("training task identities must be unique and nonempty")
        if not isinstance(row["group"], str) or not row["group"]:
            raise ValueError("whole-question group identities are required")
        identities.add(row["id"])
        group_counts[row["group"]] += 1
        if not (set(names) | {"autosaddler"}) <= set(row["scores"]):
            raise ValueError("training outcomes are missing a profile or the fixed cost reference")
        for score in row["scores"].values():
            if (
                set(score) != {"exact_match", "f1", "prompt_tokens", "completion_tokens"}
                or type(score["exact_match"]) is not int or score["exact_match"] not in (0, 1)
                or type(score["f1"]) not in (int, float) or not math.isfinite(score["f1"])
                or not 0 <= score["f1"] <= 1
                or any(type(score[key]) is not int or score[key] < 0
                       for key in ("prompt_tokens", "completion_tokens"))
                or score["prompt_tokens"] == 0
            ):
                raise ValueError("training requires complete official binary EM, F1 and actual costs")
        feature_rows.append(input_features(
            question=row["question"], context=row["context"], mode=feature_mode,
        ))
        anchor = row["scores"]["answer_only"]
        targets.append(
            [row["scores"][name]["exact_match"] - anchor["exact_match"] for name in names]
            + [row["scores"][name]["f1"] - anchor["f1"] for name in names]
        )
        completion_rows.append([row["scores"][name]["completion_tokens"] for name in names])
        auto = row["scores"]["autosaddler"]
        auto_costs.append(auto["prompt_tokens"] + auto["completion_tokens"])
    weights = np.asarray([1.0 / group_counts[row["group"]] for row in examples], dtype=np.float64)
    matrix = np.asarray(feature_rows, dtype=np.float64)
    mean = np.average(matrix, axis=0, weights=weights)
    scale = np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(examples)), (matrix - mean) / scale])
    target = np.asarray(targets, dtype=np.float64)
    gram = design.T @ (design * weights[:, None])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(gram + penalty, design.T @ (target * weights[:, None]))
    arms = len(names)
    artifact = {
        "schema_version": POLICY_SCHEMA if decision_rule == "weighted_sum" else RISK_POLICY_SCHEMA,
        "feature_mode": feature_mode,
        "feature_spec": feature_spec(feature_mode), "profiles": list(names),
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "em_intercepts": coefficients[0, :arms].tolist(),
        "f1_intercepts": coefficients[0, arms:].tolist(),
        "em_weights": coefficients[1:, :arms].T.tolist(),
        "f1_weights": coefficients[1:, arms:].T.tolist(),
        "completion_means": np.average(np.asarray(completion_rows), axis=0, weights=weights).tolist(),
        "training_auto_mean": float(np.average(auto_costs, weights=weights)),
        "ridge": ridge, "cost_penalty": cost_penalty, "f1_weight": f1_weight,
        "seed": seed, "training_tasks": len(examples), "training_groups": len(group_counts),
        "training_inputs_sha256": canonical_hash([
            {"id": row["id"], "group": row["group"], "question": row["question"],
             "context": row["context"]} for row in examples
        ]),
        "training_targets_sha256": canonical_hash(targets),
        "numpy_version": np.__version__, "group_weighting": "each whole question group has weight one",
        "clipped_prediction_range": [-1, 1], "paper_admitted": False,
    }
    if decision_rule != "weighted_sum":
        artifact["decision_rule"] = decision_rule
    QualityCostRouter(artifact)
    return artifact
