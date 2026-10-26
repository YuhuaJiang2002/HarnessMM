"""Capacity-matched context hashing and empirical training-group reweighting.

Only the final lexical block changes in ``full_context``. Shared legacy numeric
and question features retain their original definitions and limits. The new
block consumes the complete context; hashing is lossy, not a pretrained semantic
encoder. Actor inputs are unchanged. Real-data fitting belongs on a remote server.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import unicodedata

from phiagent.learning.qa_interaction_compiler import (
    ENDPOINTS,
    FULL,
    MASKS,
    METHOD_ID as COMPILER_METHOD_ID,
    SCHEMA as COMPILER_SCHEMA,
    InstructionBank,
    InteractionCompiler,
    InterventionTask,
    _hash,
    _mean_outcome,
    _number,
    _observations_digest,
    _tasks,
    interaction_terms,
    mask_basis,
)
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_quality_cost_router import (
    CONTEXT_CHARACTER_LIMIT,
    HASH_BUCKETS,
    WORD,
    _hash_block,
    feature_dimension,
    feature_spec,
    input_features,
)
from phiagent.learning.qa_subset_predictors import (
    PredictorSpec,
    SubsetPredictor,
    _SHARED_KEYS,
    _fit_mlp,
    _integer,
    _keys,
    _payload_hash,
)


SCHEMA = "phiagent-qa-context-sensitivity/v1"
METHOD_ID = "context-sensitivity-subsets-v1"
DIMENSION = 404
SENSITIVITY_ABSOLUTE_TOLERANCE = 1e-12
_CONTEXT_NAMESPACE = "full-context"
_LABEL_DEFINITION = (
    "A task is sensitive iff the maximum minus minimum of its arithmetic repeat-mean "
    "official exact_match or f1 across the 32 masks exceeds the absolute tolerance "
    "1e-12 (no relative tolerance). A source group "
    "is sensitive iff any supplied training task in that group is sensitive."
)
_WEIGHTING_RULES = {
    "scope": "training_groups_only",
    "label_definition": _LABEL_DEFINITION,
    "sensitivity_absolute_tolerance": SENSITIVITY_ABSOLUTE_TOLERANCE,
    "base_task_mass": "1 / supplied task count within source group",
    "supervised_mass": "multiply sensitive group total mass, then normalize all task masses",
    "feature_normalization": "uniform_source_groups",
    "aa_noise_floor": "uniform_source_groups",
    "target_normalization": "supervised_task_mass",
    "loss": "supervised_task_mass",
    "repeats_or_cells_are_independent": False,
    "causal_or_statistical_certificate": False,
}
_TRAINING_KEYS = {
    "task_hashes", "group_hashes", "input_hashes", "tasks", "groups", "repeat_blocks",
    "seed", "repeat_strategy", "aa_uses_all_acquired_blocks", "observations_sha256", "scope",
}


def context_feature_spec(representation: str) -> dict:
    """Describe the unchanged shared prefix and the bounded-width replacement."""
    if not isinstance(representation, str) or representation not in {"legacy", "full_context"}:
        raise ValueError("representation must be legacy or full_context")
    if feature_dimension("lexical") != DIMENSION:
        raise ValueError("the capacity-matched legacy lexical width must remain 404")
    full = representation == "full_context"
    return {
        "schema_version": "phiagent-qa-context-sensitivity-features/v1",
        "representation": representation,
        "dimension": DIMENSION,
        "legacy_feature_spec": feature_spec("lexical"),
        "shared_prefix_dimensions": DIMENSION - HASH_BUCKETS,
        "shared_prefix_definition": (
            "Identical first 20 numeric, 128 question-word and 128 question-bigram "
            "features; all legacy definitions and character/word limits remain unchanged."
        ),
        "final_block": {
            "dimension": HASH_BUCKETS,
            "source": "all context words and adjacent bigrams" if full else "legacy source title words",
            "namespace": _CONTEXT_NAMESPACE if full else "title",
            "term_tags": ["u:{word}", "b:{left} {right}"] if full else ["{title_word}"],
            "normalization": "NFKC then casefold; signed SHA256, 1+log(TF), joint block L2",
            "character_limit": None if full else CONTEXT_CHARACTER_LIMIT,
            "word_limit": None,
            "chunk_limit": None,
            "complete_context_consumed": full,
        },
        "hashing_is_lossy": True,
        "pretrained_semantic_encoder": False,
        "actor_context_is_modified": False,
        "outcomes_predictions_answers_task_ids_are_features": False,
    }


def context_features(*, question: str, context: str, representation: str) -> list[float]:
    """Use input text only; the full-context block never truncates words or characters."""
    context_feature_spec(representation)
    values = input_features(question=question, context=context, mode="lexical")
    if representation == "full_context":
        words = WORD.findall(unicodedata.normalize("NFKC", context).casefold())
        terms = [f"u:{word}" for word in words]
        terms.extend(f"b:{left} {right}" for left, right in zip(words, words[1:]))
        values = values[:-HASH_BUCKETS] + _hash_block(terms, _CONTEXT_NAMESPACE)
    if len(values) != DIMENSION or not all(math.isfinite(value) for value in values):
        raise ValueError("context features must contain exactly 404 finite values")
    return values


@dataclass(frozen=True)
class ContextSensitivitySpec:
    representation: str
    sensitive_group_weight: float = 1.0
    hidden_units: int = 32
    epochs: int = 200
    learning_rate: float = 0.01
    order: int = 2

    def __post_init__(self) -> None:
        context_feature_spec(self.representation)
        try:
            weight = _number(self.sensitive_group_weight, "sensitive group weight")
        except OverflowError as error:
            raise ValueError("sensitive group weight must be 1.0 or 8.0") from error
        if weight not in (1.0, 8.0):
            raise ValueError("sensitive group weight must be 1.0 or 8.0")
        mlp = _mlp_spec(self)
        object.__setattr__(self, "sensitive_group_weight", weight)
        object.__setattr__(self, "learning_rate", mlp.learning_rate)


def _mlp_spec(spec: ContextSensitivitySpec) -> PredictorSpec:
    return PredictorSpec(
        "mlp", feature_mode="lexical", order=spec.order, hidden_units=spec.hidden_units,
        epochs=spec.epochs, learning_rate=spec.learning_rate,
    )


def _weighting_report(tasks, group_labels, weights, spec: ContextSensitivitySpec) -> dict:
    indices = defaultdict(list)
    for index, task in enumerate(tasks):
        indices[text_hash(task.group_id)].append(index)
    labels = {text_hash(group): sensitive for group, sensitive in group_labels.items()}
    labels = dict(sorted(labels.items()))
    sensitive = sum(labels.values())
    active = 0 < sensitive < len(labels)
    return {
        **_WEIGHTING_RULES,
        "sensitive_group_weight": spec.sensitive_group_weight,
        "group_labels": labels,
        "group_label_hashes": {
            group: digest({"group_sha256": group, "sensitive": label})
            for group, label in labels.items()
        },
        "group_labels_sha256": digest(labels),
        "group_task_counts": {group: len(indices[group]) for group in labels},
        "normalized_group_weights": {
            group: math.fsum(float(weights[index]) for index in indices[group]) for group in labels
        },
        "sensitive_groups": sensitive,
        "non_sensitive_groups": len(labels) - sensitive,
        "supervision_active": active,
        "reweighting_active": active and spec.sensitive_group_weight != 1.0,
    }


def fit_context_predictor(
    tasks: Sequence[InterventionTask], *, bank: InstructionBank, spec: ContextSensitivitySpec,
    seed: int = 0,
) -> dict:
    """Fit one three-head MLP; all labels, weights and normalizers are training-only."""
    if not isinstance(bank, InstructionBank) or not isinstance(spec, ContextSensitivitySpec):
        raise ValueError("a frozen InstructionBank and typed ContextSensitivitySpec are required")
    spec = ContextSensitivitySpec(**asdict(spec))
    _integer(seed, "seed", 0, 2**64 - 1)
    tasks = _tasks(tasks, bank.sha256)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Context fitting requires the optional phiagent[qa-routing] NumPy dependency",
        ) from error
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            groups = Counter(task.group_id for task in tasks)
            uniform_mass = np.asarray([1 / groups[task.group_id] for task in tasks], dtype=float)
            raw_features = np.asarray([
                context_features(
                    question=task.question, context=task.context, representation=spec.representation,
                )
                for task in tasks
            ])
            mean = np.average(raw_features, axis=0, weights=uniform_mass)
            scale = np.sqrt(np.average((raw_features - mean) ** 2, axis=0, weights=uniform_mass))
            scale[scale < 1e-8] = 1
            features = (raw_features - mean) / scale
            observations = np.asarray([
                [[_mean_outcome(task, mask, endpoint) for endpoint in ENDPOINTS] for mask in MASKS]
                for task in tasks
            ])
            group_labels = dict.fromkeys(groups, False)
            for task, outcomes in zip(tasks, observations, strict=True):
                if bool(np.any(np.ptp(outcomes[:, :2], axis=0) > SENSITIVITY_ABSOLUTE_TOLERANCE)):
                    group_labels[task.group_id] = True
            weights = uniform_mass * np.asarray([
                spec.sensitive_group_weight if group_labels[task.group_id] else 1.0 for task in tasks
            ])
            weights /= weights.sum()
            basis = np.asarray([mask_basis(mask, spec.order) for mask in MASKS])
            targets = np.einsum("mt,nme->nte", basis, observations) / len(MASKS)
            noise = np.asarray([
                [
                    math.fsum(
                        abs(getattr(block.reference_aa, endpoint)
                            - getattr(block.outcomes[FULL], endpoint))
                        for block in task.blocks
                    ) / len(task.blocks)
                    for endpoint in ENDPOINTS[:2]
                ]
                for task in tasks
            ])
            aa_noise = np.average(noise, axis=0, weights=uniform_mass)
            if not all(np.isfinite(array).all() for array in (features, targets, weights, aa_noise)):
                raise ValueError("training features, targets, group weights and A/A noise must be finite")
            state = _fit_mlp(np, features, targets, weights, _mlp_spec(spec), seed)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("context predictor training normalization or learned state overflowed") from error
    artifact = {
        "schema_version": SCHEMA, "method_id": METHOD_ID,
        "bank": bank.artifact, "bank_sha256": bank.sha256,
        "order": spec.order, "terms": [list(term) for term in interaction_terms(spec.order)],
        "task_conditioned": True, "feature_mode": "lexical",
        "feature_spec": context_feature_spec(spec.representation),
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "aa_noise_floor": aa_noise.tolist(),
        "training": {
            "task_hashes": sorted(text_hash(task.task_id) for task in tasks),
            "group_hashes": sorted(text_hash(group) for group in groups),
            "input_hashes": sorted({task.input_sha256 for task in tasks}),
            "tasks": len(tasks), "groups": len(groups),
            "repeat_blocks": sum(len(task.blocks) for task in tasks),
            "seed": seed, "repeat_strategy": "mean", "aa_uses_all_acquired_blocks": True,
            "observations_sha256": _observations_digest(tasks), "scope": "training_only",
        },
        "weighting": _weighting_report(tasks, group_labels, weights, spec),
        "spec": asdict(spec), "state": state,
        "quality_certificate": False, "paper_admitted": False,
    }
    artifact["model_sha256"] = _payload_hash(artifact)
    return ContextSensitivityPredictor(artifact).artifact


class ContextSensitivityPredictor(SubsetPredictor):
    """Frozen input-only inference with the original compiler's regret/cost decoder."""

    def __init__(self, artifact: Mapping) -> None:
        try:
            self._initialize_context(artifact)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("malformed context-sensitivity predictor artifact") from error

    def _initialize_context(self, artifact: Mapping) -> None:
        if not isinstance(artifact, Mapping):
            raise ValueError("a context-sensitivity predictor artifact mapping is required")
        value = deepcopy(dict(artifact))
        _keys(
            value, _SHARED_KEYS | {"spec", "state", "weighting", "model_sha256"},
            "context-sensitivity artifact",
        )
        if value["schema_version"] != SCHEMA or value["method_id"] != METHOD_ID:
            raise ValueError("unsupported context-sensitivity schema or method identity")
        _hash(value["model_sha256"], "model hash")
        if _payload_hash(value) != value["model_sha256"]:
            raise ValueError("context-sensitivity model hash mismatch")
        _keys(value["spec"], set(ContextSensitivitySpec.__dataclass_fields__), "context spec")
        self.spec = ContextSensitivitySpec(**value["spec"])
        if (
            value["order"] != self.spec.order or value["task_conditioned"] is not True
            or value["feature_mode"] != "lexical"
            or value["feature_spec"] != context_feature_spec(self.spec.representation)
        ):
            raise ValueError("frozen context representation, capacity or interaction order changed")
        training = value["training"]
        _keys(training, _TRAINING_KEYS, "training metadata")
        if training["scope"] != "training_only" or training["repeat_strategy"] != "mean":
            raise ValueError("context supervision requires training-only repeat means")
        _integer(training["seed"], "training seed", 0, 2**64 - 1)
        terms = len(interaction_terms(self.spec.order))
        shared = {key: value[key] for key in _SHARED_KEYS}
        # Adapter-only placeholders reuse bank/basis/dimension/noise validation, not a ridge fit.
        shared.update(
            schema_version=COMPILER_SCHEMA, method_id=COMPILER_METHOD_ID,
            feature_spec=feature_spec("lexical"), training={**training, "ridge": 1.0},
            heads={endpoint: [[0.0] * (DIMENSION + 1) for _ in range(terms)] for endpoint in ENDPOINTS},
        )
        InteractionCompiler.__init__(self, shared)
        if len(training["input_hashes"]) > training["tasks"]:
            raise ValueError("training input count exceeds training task exposure")
        self._validate_weighting(value["weighting"], training)
        self._mlp_heads = self._validate_mlp(value["state"], DIMENSION, terms)
        self._artifact = value
        self.model_sha256 = value["model_sha256"]

    def _validate_weighting(self, weighting: object, training: Mapping) -> None:
        _keys(weighting, set(_WEIGHTING_RULES) | {
            "sensitive_group_weight", "group_labels", "group_label_hashes", "group_labels_sha256",
            "group_task_counts", "normalized_group_weights", "sensitive_groups",
            "non_sensitive_groups", "supervision_active", "reweighting_active",
        }, "group weighting diagnostics")
        if any(
            type(weighting[key]) is not type(expected) or weighting[key] != expected
            for key, expected in _WEIGHTING_RULES.items()
        ):
            raise ValueError("empirical sensitivity definition or uniform feature/A/A weighting changed")
        if (
            _number(weighting["sensitive_group_weight"], "sensitive group weight")
            != self.spec.sensitive_group_weight
        ):
            raise ValueError("sensitivity multiplier disagrees with the frozen spec")
        groups = set(training["group_hashes"])
        for name in (
            "group_labels", "group_label_hashes", "group_task_counts", "normalized_group_weights",
        ):
            _keys(weighting[name], groups, name)
        labels = weighting["group_labels"]
        if any(type(label) is not bool for label in labels.values()):
            raise ValueError("empirical source-group sensitivity labels must be booleans")
        if weighting["group_labels_sha256"] != digest(labels):
            raise ValueError("group label digest mismatch")
        counts = weighting["group_task_counts"]
        for group, label in labels.items():
            if weighting["group_label_hashes"][group] != digest({
                "group_sha256": group, "sensitive": label,
            }):
                raise ValueError("source-group label hash mismatch")
            _integer(counts[group], "group task count", 1, training["tasks"])
        if sum(counts.values()) != training["tasks"]:
            raise ValueError("group task counts must cover all supplied training tasks")
        sensitive = sum(labels.values())
        for name, expected in (
            ("sensitive_groups", sensitive), ("non_sensitive_groups", len(groups) - sensitive),
        ):
            if type(weighting[name]) is not int or weighting[name] != expected:
                raise ValueError("sensitive/non-sensitive source-group counts disagree")
        active = 0 < sensitive < len(groups)
        if (
            weighting["supervision_active"] is not active
            or weighting["reweighting_active"] is not (active and self.spec.sensitive_group_weight != 1.0)
        ):
            raise ValueError("empirical supervision activity flags disagree with the group labels")
        total = math.fsum(
            self.spec.sensitive_group_weight if label else 1.0 for label in labels.values()
        )
        for group, label in labels.items():
            weight = _number(weighting["normalized_group_weights"][group], "normalized group weight")
            expected = (self.spec.sensitive_group_weight if label else 1.0) / total
            if weight <= 0 or not math.isclose(weight, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("normalized group weights do not match empirical supervision")

    def _features(self, *, question: str, context: str) -> list[float]:
        features = context_features(
            question=question, context=context, representation=self.spec.representation,
        )
        vector = [(value - mean) / scale for value, mean, scale in
                  zip(features, self.mean, self.scale, strict=True)]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("context feature normalization overflowed")
        return vector

    def predict_effects(self, *, question: str, context: str) -> dict[str, list[float]]:
        try:
            effects = self._predict_mlp(self._features(question=question, context=context))
        except OverflowError as error:
            raise ValueError("context-sensitivity effect prediction overflowed") from error
        if not all(math.isfinite(item) for values in effects.values() for item in values):
            raise ValueError("context-sensitivity effect predictions must be finite")
        return effects

    def choose(
        self, *, question: str, context: str, prompt_tokens: Mapping[str, int],
        quality_margin: float = 0.0, repeat_guard: float = 1.0,
    ) -> dict:
        receipt = InteractionCompiler.choose(
            self, question=question, context=context, prompt_tokens=prompt_tokens,
            quality_margin=quality_margin, repeat_guard=repeat_guard,
        )
        receipt["method_id"] = METHOD_ID
        return receipt
