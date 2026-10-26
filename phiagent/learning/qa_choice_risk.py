"""Training-only logistic ranking of pre-answer numeric choice features.

Scores are not calibrated probabilities, risk certificates, or evidence of a
scientific gain. Real-data fitting belongs on an authorized remote server.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Integral, Real

from phiagent.learning.qa_meta_rsi import digest, text_hash


SCHEMA = "phiagent-qa-choice-risk/v1"
METHOD_ID = "pre-answer-choice-risk-v1"
_SETTINGS = {
    "iterations": 1000, "learning_rate": 0.01, "l2": 0.1, "class_balanced": True,
    "optimizer": "phiagent.evaluation.learned_metric_evaluator.fit_logistic_metric_evaluator",
    "initialization": "zero weights and bias; seed is provenance only",
    "learning_rate_schedule": "learning_rate / sqrt(1 + iteration / 100)",
    "feature_normalization": "uniform source-group base mass, before class balancing",
    "loss_weighting": "equal positive/negative mass after source-group weighting",
    "active_scale_floor": 1e-6, "feature_clip": 100.0, "logit_clip": 30.0,
}
_FLAGS = {
    "score_is_calibrated_probability": False, "risk_certificate": False, "paper_admitted": False,
}
_TRAINING_KEYS = {
    "scope", "tasks", "groups", "group_hashes", "group_task_counts", "group_class_counts",
    "class_counts", "positive_rate", "normalized_base_group_weights",
    "normalized_base_class_mass", "normalized_loss_class_mass", "observations_sha256",
}
_ARTIFACT_KEYS = {
    "schema_version", "method_id", "feature_names", "dimension", "feature_mean",
    "feature_scale", "active_features", "weights", "bias", "settings", "training",
    "status", "reason", "fitting_performed", "constant_score", "model_sha256",
} | set(_FLAGS)


def _sequence(value: object, name: str) -> list:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not hasattr(value, "__len__") or not hasattr(value, "__getitem__")
    ):
        raise ValueError(f"{name} must be a sequence")
    try:
        return list(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a sequence") from error


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}, not a boolean")
    return int(value)


def _labels(value: object) -> list[int]:
    labels = [_integer(item, "binary label") for item in _sequence(value, "labels")]
    if any(item not in (0, 1) for item in labels):
        raise ValueError("labels must contain only integer zero or one")
    return labels


def _names(value: object) -> list[str]:
    names = _sequence(value, "feature names")
    if (
        not names or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature names must be nonempty, unique strings")
    return names


def _vector(value: object, dimension: int, name: str) -> list[float]:
    values = _sequence(value, name)
    if len(values) != dimension:
        raise ValueError(f"{name} requires exactly {dimension} values")
    return [_number(item, name) for item in values]


def _keys(value: object, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")


def _hash(value: object, name: str) -> None:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _payload_hash(value: Mapping) -> str:
    return digest({key: item for key, item in value.items() if key != "model_sha256"})


def _settings(seed, iterations, learning_rate, l2) -> dict:
    seed = _integer(seed, "seed")
    iterations = _integer(iterations, "iterations", 1)
    learning_rate, l2 = _number(learning_rate, "learning rate"), _number(l2, "L2")
    if seed > 2**64 - 1:
        raise ValueError("seed must fit an unsigned 64-bit provenance integer")
    if (iterations, learning_rate, l2) != (1000, 0.01, 0.1):
        raise ValueError("choice-risk settings are fixed at 1000 iterations, learning_rate=0.01, l2=0.1")
    return {**_SETTINGS, "seed": seed}


def _backend():
    try:
        import numpy as np
        from phiagent.evaluation.learned_metric_evaluator import (
            LogisticMetricEvaluator,
            fit_logistic_metric_evaluator,
        )
    except ImportError as error:
        raise RuntimeError(
            "Choice-risk fitting and fitted-model scoring require the optional NumPy dependency",
        ) from error
    return np, LogisticMetricEvaluator, fit_logistic_metric_evaluator


def _training_metadata(features, labels, group_ids) -> dict:
    group_classes = defaultdict(lambda: {"positives": 0, "negatives": 0})
    for group, label in zip(group_ids, labels, strict=True):
        group_classes[text_hash(group)]["positives" if label else "negatives"] += 1
    group_classes = dict(sorted(group_classes.items()))
    counts = {group: sum(classes.values()) for group, classes in group_classes.items()}
    positives, cases = sum(labels), len(labels)
    positive_mass = math.fsum(
        classes["positives"] / counts[group] / len(counts)
        for group, classes in group_classes.items()
    )
    return {
        "scope": "training_only", "tasks": cases, "groups": len(counts),
        "group_hashes": list(counts), "group_task_counts": counts,
        "group_class_counts": group_classes,
        "class_counts": {"positives": positives, "negatives": cases - positives},
        "positive_rate": positives / cases,
        "normalized_base_group_weights": {group: 1 / len(counts) for group in counts},
        "normalized_base_class_mass": {"positives": positive_mass, "negatives": 1 - positive_mass},
        "normalized_loss_class_mass": (
            {"positives": 0.5, "negatives": 0.5} if 0 < positives < cases else None
        ),
        "observations_sha256": digest({
            "features": features, "labels": labels, "group_ids": group_ids,
        }),
    }


def _status(positives: int, negatives: int) -> tuple[str, str]:
    if not positives or not negatives:
        return (
            "UNINFORMATIVE_SINGLE_CLASS",
            "Only one training class is present; the unfitted constant 0.5 score cannot rank risk.",
        )
    return (
        "FITTED_BINARY_RANKING",
        "Both training classes are present; scores are rankings, not calibrated probabilities.",
    )


def fit_choice_risk(
    features: Sequence[Sequence[float]], labels: Sequence[int], group_ids: Sequence[str], *,
    feature_names: Sequence[str], seed: int, iterations: int = 1000,
    learning_rate: float = 0.01, l2: float = 0.1,
) -> dict:
    """Fit supplied training rows only, with fixed settings and no parameter search.

    The reused optimizer initializes at zero; ``seed`` records provenance without
    introducing randomness. Single-class data produce an explicit unfitted model.
    """
    settings = _settings(seed, iterations, learning_rate, l2)
    names = _names(feature_names)
    rows = [_vector(row, len(names), "feature row") for row in _sequence(features, "features")]
    truth, groups = _labels(labels), _sequence(group_ids, "group IDs")
    if not rows or len(rows) != len(truth) or len(rows) != len(groups):
        raise ValueError("nonempty features, labels and group IDs must have matching row counts")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("group IDs must be nonempty strings")
    training = _training_metadata(rows, truth, groups)
    positives, negatives = sum(truth), len(truth) - sum(truth)
    status, reason = _status(positives, negatives)
    fitted = bool(positives and negatives)
    np, _, fit_evaluator = _backend()
    counts = Counter(groups)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x, y = np.asarray(rows, dtype=np.float64), np.asarray(truth, dtype=np.float64)
            base_mass = np.asarray([1 / counts[group] for group in groups], dtype=np.float64)
            if fitted:
                model = fit_evaluator(
                    x, y, feature_names=tuple(names), class_balanced=True,
                    iterations=iterations, learning_rate=learning_rate, l2=l2,
                    sample_weights=base_mass,
                )
                mean, scale, active, weights, bias = (
                    model.mean, model.scale, model.active_features, model.weights, model.bias,
                )
            else:
                mean = np.average(x, axis=0, weights=base_mass)
                raw_scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=base_mass))
                active = raw_scale >= _SETTINGS["active_scale_floor"]
                scale = np.where(active, raw_scale, 1.0)
                weights, bias = np.zeros(len(names), dtype=np.float64), 0.0
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("choice-risk normalization or optimization overflowed") from error
    artifact = {
        "schema_version": SCHEMA, "method_id": METHOD_ID,
        "feature_names": names, "dimension": len(names),
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "active_features": active.tolist(), "weights": weights.tolist(), "bias": float(bias),
        "settings": settings, "training": training, "status": status, "reason": reason,
        "fitting_performed": fitted, "constant_score": None if fitted else 0.5,
        **_FLAGS,
    }
    artifact["model_sha256"] = _payload_hash(artifact)
    return ChoiceRiskPredictor(artifact).artifact


class ChoiceRiskPredictor:
    """A frozen score over one pre-answer numeric vector; no outcomes or IDs accepted."""

    def __init__(self, artifact: Mapping) -> None:
        try:
            self._initialize(artifact)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("malformed choice-risk artifact") from error

    def _initialize(self, artifact: Mapping) -> None:
        if not isinstance(artifact, Mapping):
            raise ValueError("a choice-risk artifact mapping is required")
        value = deepcopy(dict(artifact))
        _keys(value, _ARTIFACT_KEYS, "choice-risk artifact")
        if value["schema_version"] != SCHEMA or value["method_id"] != METHOD_ID:
            raise ValueError("unsupported choice-risk schema or method identity")
        _hash(value["model_sha256"], "model hash")
        if _payload_hash(value) != value["model_sha256"]:
            raise ValueError("choice-risk model hash mismatch")
        if any(value[key] is not expected for key, expected in _FLAGS.items()):
            raise ValueError("choice-risk scores must remain noncalibrated and uncertified")
        dimension = _integer(value["dimension"], "feature dimension", 1)
        if not isinstance(value["feature_names"], list) or len(_names(value["feature_names"])) != dimension:
            raise ValueError("feature names and dimension disagree")
        settings = value["settings"]
        _keys(settings, set(_SETTINGS) | {"seed"}, "choice-risk settings")
        expected = _settings(
            settings["seed"], settings["iterations"], settings["learning_rate"], settings["l2"],
        )
        if any(
            type(settings[key]) is not type(item) or settings[key] != item
            for key, item in expected.items()
        ):
            raise ValueError("the frozen logistic implementation or settings changed")
        for name in ("feature_mean", "feature_scale", "weights"):
            if not isinstance(value[name], list):
                raise ValueError("frozen logistic parameters must be JSON vectors")
        self._mean = tuple(_vector(value["feature_mean"], dimension, "feature mean"))
        self._scale = tuple(_vector(value["feature_scale"], dimension, "feature scale"))
        self._weights = tuple(_vector(value["weights"], dimension, "weights"))
        active = value["active_features"]
        if (
            not isinstance(active, list) or len(active) != dimension
            or any(type(flag) is not bool for flag in active)
        ):
            raise ValueError("active feature flags must be a matched boolean vector")
        self._active = tuple(active)
        for scale, weight, flag in zip(self._scale, self._weights, active, strict=True):
            if scale <= 0 or (flag and scale < _SETTINGS["active_scale_floor"]):
                raise ValueError("active feature scales must satisfy the frozen normalization floor")
            if not flag and (scale != 1.0 or weight != 0.0):
                raise ValueError("inactive features require unit scale and zero weight")
        self._bias = _number(value["bias"], "bias")
        positives, negatives = self._validate_training(value["training"])
        status, reason = _status(positives, negatives)
        fitted = bool(positives and negatives)
        if (
            value["status"] != status or value["reason"] != reason
            or value["fitting_performed"] is not fitted
        ):
            raise ValueError("fitting status must expose single-class training without a success claim")
        if fitted:
            if value["constant_score"] is not None:
                raise ValueError("a fitted binary model cannot declare a constant-score fallback")
        elif (
            _number(value["constant_score"], "constant score") != 0.5
            or self._bias != 0.0 or any(weight != 0.0 for weight in self._weights)
        ):
            raise ValueError("single-class models must be unfitted constant 0.5 scorers")
        self._artifact = value
        self.model_sha256 = value["model_sha256"]
        self._evaluator = None

    @staticmethod
    def _validate_training(training: object) -> tuple[int, int]:
        _keys(training, _TRAINING_KEYS, "training metadata")
        if training["scope"] != "training_only":
            raise ValueError("choice-risk training scope must be training_only")
        cases = _integer(training["tasks"], "training task count", 1)
        groups = _integer(training["groups"], "training group count", 1)
        hashes = training["group_hashes"]
        if not isinstance(hashes, list) or len(hashes) != groups:
            raise ValueError("group hash inventory and count disagree")
        for group in hashes:
            _hash(group, "training group hash")
        if hashes != sorted(set(hashes)):
            raise ValueError("training group hashes must be unique and sorted")
        inventory = set(hashes)
        for name in ("group_task_counts", "group_class_counts", "normalized_base_group_weights"):
            _keys(training[name], inventory, name)
        positives = negatives = 0
        contributions = []
        for group in hashes:
            count = _integer(training["group_task_counts"][group], "source-group task count", 1)
            classes = training["group_class_counts"][group]
            _keys(classes, {"positives", "negatives"}, "source-group class counts")
            positive = _integer(classes["positives"], "source-group positives")
            negative = _integer(classes["negatives"], "source-group negatives")
            if positive + negative != count:
                raise ValueError("source-group class and task counts disagree")
            positives += positive
            negatives += negative
            contributions.append(positive / count / groups)
            weight = _number(training["normalized_base_group_weights"][group], "base group weight")
            if not 0 < weight <= 1 or not math.isclose(
                weight, 1 / groups, rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError("base normalization must give every source group equal mass")
        positive_mass = math.fsum(contributions)
        classes = training["class_counts"]
        _keys(classes, {"positives", "negatives"}, "training class counts")
        if (
            positives + negatives != cases
            or _integer(classes["positives"], "training positives") != positives
            or _integer(classes["negatives"], "training negatives") != negatives
            or _number(training["positive_rate"], "training positive rate") != positives / cases
        ):
            raise ValueError("training group totals, class counts or prevalence disagree")
        masses = training["normalized_base_class_mass"]
        _keys(masses, {"positives", "negatives"}, "base class masses")
        for label, expected in (("positives", positive_mass), ("negatives", 1 - positive_mass)):
            mass = _number(masses[label], "normalized base class mass")
            if not 0 <= mass <= 1 or not math.isclose(mass, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("base class masses must derive from source-group counts")
        loss_mass = training["normalized_loss_class_mass"]
        if positives and negatives:
            _keys(loss_mass, {"positives", "negatives"}, "balanced loss masses")
            if any(_number(mass, "loss class mass") != 0.5 for mass in loss_mass.values()):
                raise ValueError("binary logistic loss must give both classes equal mass")
        elif loss_mass is not None:
            raise ValueError("single-class training must not claim a balanced binary loss")
        _hash(training["observations_sha256"], "training observations hash")
        return positives, negatives

    @property
    def artifact(self) -> dict:
        return deepcopy(self._artifact)

    def score(self, features: Sequence[float]) -> float:
        values = _vector(features, self._artifact["dimension"], "pre-answer feature vector")
        if not self._artifact["fitting_performed"]:
            return float(self._artifact["constant_score"])
        np, evaluator_type, _ = _backend()
        if self._evaluator is None:
            self._evaluator = evaluator_type(
                feature_names=tuple(self._artifact["feature_names"]),
                mean=np.asarray(self._mean), scale=np.asarray(self._scale),
                active_features=np.asarray(self._active, dtype=bool),
                weights=np.asarray(self._weights), bias=self._bias,
                threshold=0.5, class_balanced=True,
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                score = float(self._evaluator.predict_proba(np.asarray([values], dtype=np.float64))[0])
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("choice-risk feature normalization or scoring overflowed") from error
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("choice-risk scorer returned an invalid ranking score")
        return score


def binary_ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict:
    """Tie-aware AUROC/AP; single-class results do not establish discrimination."""
    truth = _labels(labels)
    values = [_number(score, "ranking score") for score in _sequence(scores, "scores")]
    if len(truth) != len(values):
        raise ValueError("binary labels and ranking scores must have matching lengths")
    cases, positives = len(truth), sum(truth)
    negatives = cases - positives
    result = {
        "cases": cases, "positives": positives, "negatives": negatives,
        "positive_rate": positives / cases if cases else None,
        "auroc": None, "average_precision": None,
        "status": "DEFINED", "reasons": {},
        "tie_policy": "AUROC half credit; AP precision after each entire score-tie block",
        "paper_admitted": False,
    }
    if not positives or not negatives:
        if not cases:
            result.update(
                status="UNDEFINED_NO_CASES",
                reasons={
                    "auroc": "No paired cases were supplied.",
                    "average_precision": "No paired cases were supplied.",
                },
            )
        else:
            result.update(
                status="UNINFORMATIVE_SINGLE_CLASS",
                average_precision=1.0 if positives else None,
                reasons={
                    "auroc": "AUROC requires both positive and negative cases.",
                    "average_precision": (
                        "All cases are positive; AP is 1 by definition, not evidence of discrimination."
                        if positives else "No positive cases are present; average precision is undefined."
                    ),
                },
            )
        return result
    blocks = defaultdict(lambda: [0, 0])
    for label, score in zip(truth, values, strict=True):
        blocks[score][label] += 1
    twice_concordant = negatives_below = 0
    for score in sorted(blocks):
        negative, positive = blocks[score]
        twice_concordant += positive * (2 * negatives_below + negative)
        negatives_below += negative
    result["auroc"] = twice_concordant / (2 * positives * negatives)
    found = seen = 0
    precision_increments = []
    for score in sorted(blocks, reverse=True):
        negative, positive = blocks[score]
        found += positive
        seen += positive + negative
        precision_increments.append((positive / positives) * (found / seen))
    result["average_precision"] = math.fsum(precision_increments)
    return result
