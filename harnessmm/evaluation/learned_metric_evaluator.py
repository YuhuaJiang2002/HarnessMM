"""Small deterministic logistic evaluator for frozen metric features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticMetricEvaluator:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    active_features: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    class_balanced: bool

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        normalized = np.clip((features - self.mean) / self.scale, -100.0, 100.0)
        normalized = normalized * self.active_features
        logits = np.clip(
            np.sum(normalized * self.weights[None, :], axis=1) + self.bias,
            -30.0,
            30.0,
        )
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features) >= self.threshold


def fit_logistic_metric_evaluator(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    class_balanced: bool,
    iterations: int,
    learning_rate: float,
    l2: float,
    threshold: float = 0.5,
    sample_weights: np.ndarray | None = None,
) -> LogisticMetricEvaluator:
    """Fit a deterministic full-batch logistic evaluator.

    Optional positive finite sample weights determine feature statistics before
    class balancing. Their loss mass is normalized to preserve the average-loss
    objective and L2 strength. Omitting weights retains the original arithmetic.
    """

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("features and labels have incompatible shapes")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match the feature matrix")
    if len(x) == 0 or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("training data must contain binary labels")
    if len(np.unique(y)) != 2:
        raise ValueError("training data must contain both classes")
    if iterations <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("optimizer settings are invalid")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")

    base_weights = None
    if sample_weights is not None:
        raw_weights = np.asarray(sample_weights)
        if (
            raw_weights.shape != (len(y),) or raw_weights.dtype.kind not in "iuf"
            or any(isinstance(value, (bool, np.bool_)) for value in sample_weights)
        ):
            raise ValueError("sample_weights must be a matched vector of real numbers, not booleans")
        with np.errstate(over="ignore", invalid="ignore"):
            base_weights = raw_weights.astype(np.float64)
            total_mass = float(np.sum(base_weights))
        if (
            not np.all(np.isfinite(base_weights)) or not np.all(base_weights > 0)
            or not np.isfinite(total_mass) or total_mass <= 0
        ):
            raise ValueError("sample_weights must have positive finite entries and total mass")
    if base_weights is None:
        mean = np.mean(x, axis=0)
        raw_scale = np.std(x, axis=0)
    else:
        mean = np.average(x, axis=0, weights=base_weights)
        raw_scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=base_weights))
    active_features = raw_scale >= 1e-6
    scale = np.where(active_features, raw_scale, 1.0)
    normalized = np.clip((x - mean) / scale, -100.0, 100.0)
    normalized = normalized * active_features
    active_normalized = normalized[:, active_features]
    active_weights = np.zeros(int(np.sum(active_features)), dtype=np.float64)
    bias = 0.0
    sample_weights = np.ones(len(y), dtype=np.float64)
    if class_balanced:
        positives = int(np.sum(y == 1.0))
        negatives = int(np.sum(y == 0.0))
        sample_weights = np.where(
            y == 1.0,
            len(y) / (2.0 * positives),
            len(y) / (2.0 * negatives),
        )
    if base_weights is not None:
        sample_weights = base_weights.copy()
        if class_balanced:
            for label in (0.0, 1.0):
                selected = y == label
                sample_weights[selected] = (
                    base_weights[selected] / np.sum(base_weights[selected])
                ) * (len(y) / 2.0)
        else:
            sample_weights = (base_weights / total_mass) * len(y)

    for iteration in range(iterations):
        logits = np.clip(
            np.sum(
                active_normalized * active_weights[None, :],
                axis=1,
            )
            + bias,
            -30.0,
            30.0,
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - y) * sample_weights
        rate = learning_rate / np.sqrt(1.0 + iteration / 100.0)
        gradient = np.sum(
            active_normalized * error[:, None],
            axis=0,
        ) / len(y)
        active_weights -= rate * (gradient + l2 * active_weights)
        bias -= rate * float(np.mean(error))
        if not np.all(np.isfinite(active_weights)) or not np.isfinite(bias):
            raise RuntimeError("logistic evaluator optimization diverged")

    weights = np.zeros(x.shape[1], dtype=np.float64)
    weights[active_features] = active_weights
    return LogisticMetricEvaluator(
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        active_features=active_features,
        weights=weights,
        bias=bias,
        threshold=threshold,
        class_balanced=class_balanced,
    )


def binary_report(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("labels and predictions must be matched vectors")
    true_positive = int(np.sum(truth & predicted))
    true_negative = int(np.sum(~truth & ~predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    return {
        "cases": len(truth),
        "accuracy": float(np.mean(truth == predicted)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "positive_recall": (true_positive / int(np.sum(truth)) if np.any(truth) else 0.0),
        "negative_recall": (true_negative / int(np.sum(~truth)) if np.any(~truth) else 0.0),
    }


__all__ = [
    "LogisticMetricEvaluator",
    "binary_report",
    "fit_logistic_metric_evaluator",
]
