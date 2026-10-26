"""Provisional input-to-subset-effect learners over the frozen compiler features.

Real-data fitting belongs on an authorized server. Inference uses only the original
question/context and needs neither NumPy nor a model provider. Artifacts use a new
identity, including for the ridge family; they are not frozen v1 compiler fits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from phiagent.learning.qa_interaction_compiler import (
    ENDPOINTS,
    MASKS,
    METHOD_ID as COMPILER_METHOD_ID,
    SCHEMA as COMPILER_SCHEMA,
    InstructionBank,
    InteractionCompiler,
    InterventionTask,
    _hash,
    _mean_outcome,
    _number,
    _tasks,
    _vector,
    fit_compiler,
    interaction_terms,
    mask_basis,
)
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_quality_cost_router import feature_dimension, feature_spec, input_features


SCHEMA = "phiagent-qa-subset-predictors/v1"
METHOD_ID = "input-subset-predictors-v1"
_SHARED_KEYS = {
    "schema_version", "method_id", "bank", "bank_sha256", "order", "task_conditioned",
    "terms", "feature_mode", "feature_spec", "feature_mean", "feature_scale",
    "aa_noise_floor", "training", "quality_certificate", "paper_admitted",
}
_ADAM = {"name": "adam", "batch": "full", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
_KNN_RULES = {
    "distance": "squared-euclidean/train-standardized-features",
    "neighbors_unit": "source_groups",
    "weighting": "uniform-selected-groups",
    "within_group": "nearest-input; duplicate-group-input-targets-averaged",
    "tie_break": "distance-then-group-sha256-then-input-sha256",
    "unconditioned": "uniform-all-source-groups; task-mean-within-each-group",
}


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class PredictorSpec:
    """Frozen bounded hyperparameters, including fields unused by a given family.

    Ridge applies only to ridge heads. MLP heads independently train one tanh layer
    with 1..256 units, 1..10000 full-batch epochs and a learning rate in (0, 1].
    kNN uses 1..4096 nearest source groups, with one nearest input vote per group.
    Without task conditioning, kNN averages all source groups' task means instead.
    """

    family: str
    feature_mode: str = "numeric"
    order: int = 2
    ridge: float = 1.0
    hidden_units: int = 32
    epochs: int = 200
    learning_rate: float = 0.01
    neighbors: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or self.family not in {"ridge", "mlp", "knn"}:
            raise ValueError("predictor family must be ridge, mlp or knn")
        if not isinstance(self.feature_mode, str):
            raise ValueError("feature mode must be numeric or lexical")
        feature_spec(self.feature_mode)
        interaction_terms(self.order)
        try:
            ridge = _number(self.ridge, "ridge", minimum=0)
            rate = _number(self.learning_rate, "learning rate", minimum=0)
        except OverflowError as error:
            raise ValueError("hyperparameters must be finite numbers") from error
        if ridge == 0 or not 0 < rate <= 1:
            raise ValueError("ridge must be positive and learning rate must lie in (0, 1]")
        _integer(self.hidden_units, "hidden units", 1, 256)
        _integer(self.epochs, "epochs", 1, 10000)
        _integer(self.neighbors, "neighbors", 1, 4096)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "learning_rate", rate)


def _keys(value: object, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")


def _matrix(value: object, rows: int, columns: int, label: str) -> tuple:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} requires exactly {rows} rows")
    return tuple(_vector(row, columns, label) for row in value)


def _payload_hash(artifact: Mapping) -> str:
    return digest({key: value for key, value in artifact.items() if key != "model_sha256"})


def _supervision(np, tasks: Sequence[InterventionTask], artifact: Mapping):
    groups = Counter(task.group_id for task in tasks)
    weights = np.asarray([1 / groups[task.group_id] for task in tasks], dtype=float)
    weights /= weights.sum()
    features = np.asarray([
        input_features(
            question=task.question, context=task.context, mode=artifact["feature_mode"],
        )
        for task in tasks
    ], dtype=float)
    if not artifact["task_conditioned"]:
        features = np.zeros_like(features)
    features = (
        (features - np.asarray(artifact["feature_mean"])) / np.asarray(artifact["feature_scale"])
    )
    observations = np.asarray([
        [
            [
                _mean_outcome(task, mask, endpoint)
                if artifact["training"]["repeat_strategy"] == "mean"
                else getattr(task.blocks[0].outcomes[mask], endpoint)
                for endpoint in ENDPOINTS
            ]
            for mask in MASKS
        ]
        for task in tasks
    ], dtype=float)
    basis = np.asarray([mask_basis(mask, artifact["order"]) for mask in MASKS])
    targets = np.einsum("mt,nme->nte", basis, observations) / len(MASKS)
    if not all(np.isfinite(array).all() for array in (features, targets, weights)):
        raise ValueError("training features, Walsh targets and group weights must be finite")
    return features, targets, weights


def _fit_mlp(np, features, targets, weights, spec: PredictorSpec, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dimension, terms = features.shape[1], targets.shape[1]
    heads, diagnostics = {}, {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        raw = targets[:, :, endpoint_index]
        mean = np.average(raw, axis=0, weights=weights)
        # Large constant cost offsets must not create round-off-only training gradients.
        constant = np.all(raw == raw[0], axis=0)
        mean[constant] = raw[0, constant]
        scale = np.sqrt(np.average((raw - mean) ** 2, axis=0, weights=weights))
        scale[scale < 1e-8] = 1
        target = (raw - mean) / scale
        hidden_weights = rng.normal(
            0, math.sqrt(2 / (dimension + spec.hidden_units)), (dimension, spec.hidden_units),
        )
        hidden_bias = np.zeros(spec.hidden_units)
        output_weights = rng.normal(
            0, math.sqrt(2 / (spec.hidden_units + terms)), (spec.hidden_units, terms),
        )
        output_bias = np.zeros(terms)
        parameters = [hidden_weights, hidden_bias, output_weights, output_bias]
        initial_hidden = hidden_weights.copy()
        moment1 = [np.zeros_like(parameter) for parameter in parameters]
        moment2 = [np.zeros_like(parameter) for parameter in parameters]

        def forward():
            hidden = np.tanh(features @ hidden_weights + hidden_bias)
            return hidden, hidden @ output_weights + output_bias

        def loss(prediction):
            return float(np.sum(weights[:, None] * (prediction - target) ** 2) / terms)

        hidden, prediction = forward()
        losses = [loss(prediction)]
        for step in range(1, spec.epochs + 1):
            output_gradient = 2 * weights[:, None] * (prediction - target) / terms
            hidden_gradient = (output_gradient @ output_weights.T) * (1 - hidden ** 2)
            gradients = [
                features.T @ hidden_gradient, hidden_gradient.sum(axis=0),
                hidden.T @ output_gradient, output_gradient.sum(axis=0),
            ]
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                moment1[index] = _ADAM["beta1"] * moment1[index] + (1 - _ADAM["beta1"]) * gradient
                moment2[index] = (
                    _ADAM["beta2"] * moment2[index] + (1 - _ADAM["beta2"]) * gradient ** 2
                )
                parameter -= spec.learning_rate * (
                    moment1[index] / (1 - _ADAM["beta1"] ** step)
                ) / (
                    np.sqrt(moment2[index] / (1 - _ADAM["beta2"] ** step)) + _ADAM["epsilon"]
                )
            hidden, prediction = forward()
            losses.append(loss(prediction))
        heads[endpoint] = {
            "target_mean": mean.tolist(), "target_scale": scale.tolist(),
            "hidden_weights": hidden_weights.T.tolist(), "hidden_bias": hidden_bias.tolist(),
            "output_weights": output_weights.T.tolist(), "output_bias": output_bias.tolist(),
        }
        diagnostics[endpoint] = {
            "initial_loss": losses[0], "final_loss": losses[-1], "loss_history": losses,
            "initial_hidden_sha256": digest(initial_hidden.T.tolist()),
            "hidden_weight_change_l2": float(np.sqrt(np.sum((hidden_weights - initial_hidden) ** 2))),
        }
    return {
        "activation": "tanh", "optimizer": deepcopy(_ADAM), "heads": heads,
        "training_diagnostics": {
            "scope": "training_only", "benchmark_result": False,
            "loss": "group-weighted mean squared standardized Walsh effects",
            "heads": diagnostics,
        },
    }


def _fit_knn(tasks: Sequence[InterventionTask], features, targets) -> dict:
    identities = defaultdict(list)
    for index, task in enumerate(tasks):
        identities[(text_hash(task.group_id), task.input_sha256)].append(index)
    rows = []
    for (group_hash, input_hash), indices in sorted(identities.items()):
        rows.append({
            "group_sha256": group_hash, "input_sha256": input_hash,
            "tasks": len(indices),
            "features": features[indices[0]].tolist(),
            "effects": {
                endpoint: [
                    math.fsum(float(targets[index, term, endpoint_index]) for index in indices)
                    / len(indices)
                    for term in range(targets.shape[1])
                ]
                for endpoint_index, endpoint in enumerate(ENDPOINTS)
            },
        })
    return {"rules": deepcopy(_KNN_RULES), "rows": rows}


def fit_predictor(
    tasks: Sequence[InterventionTask], *, bank: InstructionBank, spec: PredictorSpec,
    seed: int = 0, task_conditioned: bool = True, repeat_strategy: str = "mean",
) -> dict:
    """Fit supplied training tasks only; no validation, query labels or model calls.

    Shared features, group normalization, exposure hashes and descriptive A/A
    noise are inherited from ``fit_compiler``. NumPy is optional and imported here
    only. The embedded model hash covers every other artifact field.
    """
    if not isinstance(spec, PredictorSpec) or not isinstance(bank, InstructionBank):
        raise ValueError("a typed PredictorSpec and frozen InstructionBank are required")
    spec = PredictorSpec(**asdict(spec))
    _integer(seed, "seed", 0, 2**64 - 1)
    if (
        type(task_conditioned) is not bool or not isinstance(repeat_strategy, str)
        or repeat_strategy not in {"mean", "first"}
    ):
        raise ValueError("declare input conditioning and mean/first repeat supervision")
    tasks = _tasks(tasks, bank.sha256)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Predictor fitting requires the optional phiagent[qa-routing] NumPy dependency",
        ) from error
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            artifact = fit_compiler(
                tasks, bank=bank, order=spec.order, feature_mode=spec.feature_mode,
                ridge=spec.ridge, seed=seed, task_conditioned=task_conditioned,
                repeat_strategy=repeat_strategy,
            )
            ridge_heads = artifact.pop("heads")
            if spec.family == "ridge":
                state = {"heads": ridge_heads}
            else:
                features, targets, weights = _supervision(np, tasks, artifact)
                state = (
                    _fit_mlp(np, features, targets, weights, spec, seed)
                    if spec.family == "mlp" else _fit_knn(tasks, features, targets)
                )
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("nonfinite training normalization, targets or learned state") from error
    artifact.update({
        "schema_version": SCHEMA, "method_id": METHOD_ID, "spec": asdict(spec), "state": state,
    })
    artifact["training"]["scope"] = "training_only"
    artifact["model_sha256"] = _payload_hash(artifact)
    return SubsetPredictor(artifact).artifact


class SubsetPredictor(InteractionCompiler):
    """Frozen input-only effects with the compiler's unchanged regret/cost rule."""

    def __init__(self, artifact: Mapping) -> None:
        try:
            self._initialize(artifact)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("malformed subset predictor artifact") from error

    def _initialize(self, artifact: Mapping) -> None:
        if not isinstance(artifact, Mapping):
            raise ValueError("a subset predictor artifact mapping is required")
        value = deepcopy(dict(artifact))
        _keys(value, _SHARED_KEYS | {"spec", "state", "model_sha256"}, "predictor artifact")
        if value["schema_version"] != SCHEMA or value["method_id"] != METHOD_ID:
            raise ValueError("unsupported subset predictor schema or method identity")
        _hash(value["model_sha256"], "model hash")
        if _payload_hash(value) != value["model_sha256"]:
            raise ValueError("subset predictor model hash mismatch")
        _keys(value["spec"], set(PredictorSpec.__dataclass_fields__), "predictor spec")
        self.spec = PredictorSpec(**value["spec"])
        if self.spec.order != value["order"] or self.spec.feature_mode != value["feature_mode"]:
            raise ValueError("predictor spec and frozen basis/features disagree")
        training = value["training"]
        if training["scope"] != "training_only" or training["ridge"] != self.spec.ridge:
            raise ValueError("predictor training scope or ridge metadata disagrees")
        _integer(training["seed"], "training seed", 0, 2**64 - 1)
        dimension = feature_dimension(self.spec.feature_mode)
        terms = len(interaction_terms(self.spec.order))
        state = value["state"]
        if self.spec.family == "ridge":
            _keys(state, {"heads"}, "ridge state")
            heads = state["heads"]
        else:
            # Reuse the base metadata validator once, without retaining unused ridge weights.
            heads = {endpoint: [[0.0] * (dimension + 1) for _ in range(terms)]
                     for endpoint in ENDPOINTS}
        shared = {key: value[key] for key in _SHARED_KEYS}
        shared.update(schema_version=COMPILER_SCHEMA, method_id=COMPILER_METHOD_ID, heads=heads)
        super().__init__(shared)
        if len(training["input_hashes"]) > training["tasks"]:
            raise ValueError("training input counts exceed task exposure")
        if self.spec.family == "mlp":
            self._mlp_heads = self._validate_mlp(state, dimension, terms)
        elif self.spec.family == "knn":
            self._knn_rows, self._knn_global_effects = self._validate_knn(
                state, dimension, terms, training,
            )
        self._artifact = value
        self.model_sha256 = value["model_sha256"]

    def _validate_mlp(self, state: object, dimension: int, terms: int) -> dict:
        _keys(state, {"activation", "optimizer", "heads", "training_diagnostics"}, "MLP state")
        if state["activation"] != "tanh" or state["optimizer"] != _ADAM:
            raise ValueError("unsupported MLP activation or full-batch Adam settings")
        _keys(state["heads"], set(ENDPOINTS), "MLP heads")
        diagnostics = state["training_diagnostics"]
        _keys(diagnostics, {"scope", "benchmark_result", "loss", "heads"}, "MLP diagnostics")
        if (
            diagnostics["scope"] != "training_only" or diagnostics["benchmark_result"] is not False
            or diagnostics["loss"] != "group-weighted mean squared standardized Walsh effects"
        ):
            raise ValueError("MLP diagnostics must remain training-only loss evidence")
        _keys(diagnostics["heads"], set(ENDPOINTS), "MLP loss diagnostics")
        heads = {}
        hidden = self.spec.hidden_units
        for endpoint in ENDPOINTS:
            head = state["heads"][endpoint]
            _keys(head, {
                "target_mean", "target_scale", "hidden_weights", "hidden_bias",
                "output_weights", "output_bias",
            }, f"{endpoint} MLP head")
            scales = _vector(head["target_scale"], terms, "target scales")
            if any(scale <= 0 for scale in scales):
                raise ValueError("MLP target scales must be positive")
            heads[endpoint] = {
                "target_mean": _vector(head["target_mean"], terms, "target means"),
                "target_scale": scales,
                "hidden_weights": _matrix(head["hidden_weights"], hidden, dimension, "hidden weights"),
                "hidden_bias": _vector(head["hidden_bias"], hidden, "hidden biases"),
                "output_weights": _matrix(head["output_weights"], terms, hidden, "output weights"),
                "output_bias": _vector(head["output_bias"], terms, "output biases"),
            }
            info = diagnostics["heads"][endpoint]
            _keys(info, {
                "initial_loss", "final_loss", "loss_history",
                "initial_hidden_sha256", "hidden_weight_change_l2",
            }, f"{endpoint} loss diagnostics")
            history = _vector(info["loss_history"], self.spec.epochs + 1, "training loss history")
            if (
                any(loss < 0 for loss in history)
                or _number(info["initial_loss"], "initial loss") != history[0]
                or _number(info["final_loss"], "final loss") != history[-1]
            ):
                raise ValueError("MLP training loss diagnostics are inconsistent")
            _number(info["hidden_weight_change_l2"], "hidden weight change", minimum=0)
            _hash(info["initial_hidden_sha256"], "initial hidden weight hash")
        return heads

    @staticmethod
    def _validate_knn(state: object, dimension: int, terms: int, training: Mapping) -> tuple:
        _keys(state, {"rules", "rows"}, "kNN state")
        if state["rules"] != _KNN_RULES:
            raise ValueError("unsupported kNN source-group weighting or tie rule")
        rows = state["rows"]
        if not isinstance(rows, list) or not 0 < len(rows) <= training["tasks"]:
            raise ValueError("kNN requires nonempty training-only representations")
        parsed, identities = [], []
        groups = defaultdict(list)
        for row in rows:
            _keys(row, {"group_sha256", "input_sha256", "tasks", "features", "effects"}, "kNN row")
            _hash(row["group_sha256"], "neighbor group hash")
            _hash(row["input_sha256"], "neighbor input hash")
            count = _integer(row["tasks"], "neighbor task count", 1, training["tasks"])
            _keys(row["effects"], set(ENDPOINTS), "neighbor effects")
            identities.append((row["group_sha256"], row["input_sha256"]))
            parsed.append((
                row["group_sha256"], row["input_sha256"],
                _vector(row["features"], dimension, "neighbor features"),
                {endpoint: _vector(row["effects"][endpoint], terms, "neighbor effects")
                 for endpoint in ENDPOINTS},
            ))
            groups[row["group_sha256"]].append((count, parsed[-1][3]))
        if (
            identities != sorted(set(identities))
            or {group for group, _ in identities} != set(training["group_hashes"])
            or {source for _, source in identities} != set(training["input_hashes"])
            or sum(count for members in groups.values() for count, _ in members) != training["tasks"]
        ):
            raise ValueError("kNN rows and task counts must canonically bind the training exposures")
        group_means = []
        for members in groups.values():
            size = sum(count for count, _ in members)
            group_means.append({
                endpoint: [
                    math.fsum((count / size) * effects[endpoint][term] for count, effects in members)
                    for term in range(terms)
                ]
                for endpoint in ENDPOINTS
            })
        global_effects = {
            endpoint: _vector([
                math.fsum(group[endpoint][term] / len(group_means) for group in group_means)
                for term in range(terms)
            ], terms, "global group-weighted effects")
            for endpoint in ENDPOINTS
        }
        return tuple(parsed), global_effects

    def _features(self, *, question: str, context: str) -> list[float]:
        features = input_features(question=question, context=context, mode=self.mode)
        if not self._artifact["task_conditioned"]:
            features = [0.0] * len(features)
        vector = [(value - mean) / scale for value, mean, scale in
                  zip(features, self.mean, self.scale, strict=True)]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("input feature normalization overflowed")
        return vector

    def _predict_mlp(self, vector: Sequence[float]) -> dict[str, list[float]]:
        effects = {}
        for endpoint, head in self._mlp_heads.items():
            hidden = []
            for row, bias in zip(head["hidden_weights"], head["hidden_bias"], strict=True):
                activation = math.fsum(w * x for w, x in zip(row, vector, strict=True)) + bias
                if not math.isfinite(activation):
                    raise ValueError("MLP hidden activation overflowed")
                hidden.append(math.tanh(activation))
            effects[endpoint] = [
                (math.fsum(w * x for w, x in zip(row, hidden, strict=True)) + bias) * scale + mean
                for row, bias, mean, scale in zip(
                    head["output_weights"], head["output_bias"],
                    head["target_mean"], head["target_scale"], strict=True,
                )
            ]
        return effects

    def _predict_knn(self, vector: Sequence[float]) -> dict[str, list[float]]:
        if not self._artifact["task_conditioned"]:
            return {endpoint: list(values) for endpoint, values in self._knn_global_effects.items()}
        nearest = {}
        for group, source, features, effects in self._knn_rows:
            distance = math.fsum((left - right) ** 2
                                 for left, right in zip(vector, features, strict=True))
            if not math.isfinite(distance):
                raise ValueError("kNN feature distance overflowed")
            if group not in nearest or (distance, source) < nearest[group][:2]:
                nearest[group] = (distance, source, effects)
        selected = sorted(
            ((distance, group, source, effects)
             for group, (distance, source, effects) in nearest.items()),
            key=lambda item: item[:3],
        )[:self.spec.neighbors]
        return {
            endpoint: [
                math.fsum(row[3][endpoint][term] for row in selected) / len(selected)
                for term in range(len(self._artifact["terms"]))
            ]
            for endpoint in ENDPOINTS
        }

    def predict_effects(self, *, question: str, context: str) -> dict[str, list[float]]:
        try:
            if self.spec.family == "ridge":
                effects = super().predict_effects(question=question, context=context)
            else:
                vector = self._features(question=question, context=context)
                effects = (
                    self._predict_mlp(vector) if self.spec.family == "mlp"
                    else self._predict_knn(vector)
                )
        except OverflowError as error:
            raise ValueError("input-only subset-effect prediction overflowed") from error
        if not all(math.isfinite(item) for values in effects.values() for item in values):
            raise ValueError("subset-effect predictions must be finite")
        return effects

    def choose(
        self, *, question: str, context: str, prompt_tokens: Mapping[str, int],
        quality_margin: float = 0.0, repeat_guard: float = 1.0,
    ) -> dict:
        receipt = super().choose(
            question=question, context=context, prompt_tokens=prompt_tokens,
            quality_margin=quality_margin, repeat_guard=repeat_guard,
        )
        receipt["method_id"] = METHOD_ID
        return receipt
