"""Safe residual routing over a frozen static harness policy.

The router never calls a model. It chooses a non-default operation only when a
task-disjoint calibration certificate and a low-capacity local score both pass.
Unknown, missing, or out-of-distribution features fall back to the static
incumbent and are recorded in the decision trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
import statistics
from typing import Any, Mapping, Sequence


POLICY_SCHEMA_VERSION = "phiagent-residual-meta-harness-policy/v1"
TRACE_SCHEMA_VERSION = "phiagent-residual-meta-harness-trace/v1"
EVIDENCE_SCHEMA_VERSION = "phiagent-residual-meta-harness-evidence/v1"
MAX_FEATURES = 8
MAX_OVERRIDE_HEADS = 1
CALIBRATION_BOUND_METHODS = frozenset({"hoeffding", "paired_exact_bonferroni"})
ALLOWED_FEATURE_NAMES = frozenset(
    {
        "action_fraction",
        "browser_failure_fraction",
        "context_character_count",
        "context_token_count",
        "document_count",
        "expected_answer_numeric",
        "expected_answer_open",
        "expected_answer_person",
        "expected_answer_yes_no",
        "family_debugging",
        "family_information_retrieval",
        "family_navigation",
        "family_security",
        "family_state_change",
        "no_progress_fraction",
        "question_character_count",
        "question_context_coverage",
        "question_grounded_anchor_count",
        "question_grounded_anchor_mass",
        "question_has_quoted_span",
        "question_has_relation_cue",
        "question_named_token_count",
        "question_token_count",
        "recovery_signal",
        "remaining_fraction",
        "repeated_action_fraction",
        "repository_history_task",
        "turn_fraction",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_FEATURE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_FORBIDDEN_RUNTIME_FIELD_TOKENS = frozenset(
    {
        "answer",
        "correct",
        "correctness",
        "evaluator",
        "gold",
        "grade",
        "label",
        "loss",
        "metric",
        "outcome",
        "reward",
        "score",
        "success",
        "target",
        "verdict",
        "winner",
    }
)


class PolicyValidationError(ValueError):
    """Raised when a residual policy or runtime input violates its contract."""


class Operation(str, Enum):
    KEEP = "KEEP"
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"


class FeatureKind(str, Enum):
    BOOLEAN = "boolean"
    NUMERIC = "numeric"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    kind: FeatureKind
    mean: float = 0.0
    scale: float = 1.0
    minimum: float = 0.0
    maximum: float = 1.0

    def normalize(self, value: object) -> float:
        if self.kind is FeatureKind.BOOLEAN:
            if type(value) is not bool:
                raise PolicyValidationError("boolean feature has a non-boolean value")
            return float(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyValidationError("numeric feature has a non-numeric value")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PolicyValidationError("numeric feature must be finite")
        if not self.minimum <= numeric <= self.maximum:
            raise PolicyValidationError(
                f"numeric feature {numeric} is outside [{self.minimum}, {self.maximum}]"
            )
        return (numeric - self.mean) / self.scale


@dataclass(frozen=True, slots=True)
class SafetyGate:
    minimum_training_tasks: int
    minimum_calibration_tasks: int
    minimum_effective_sample_size: float
    minimum_override_support: int
    maximum_harmful_rate_upper: float
    minimum_utility_lcb: float
    maximum_token_delta_upper: float
    maximum_extra_actor_calls: int
    maximum_router_model_calls: int


@dataclass(frozen=True, slots=True)
class SafetyCertificate:
    training_task_count: int
    calibration_task_count: int
    effective_sample_size: float
    override_support: int
    grouped_crossfit_folds: int
    task_disjoint_calibration: bool
    harmful_rate_upper: float
    utility_lcb: float
    token_delta_upper: float
    extra_actor_calls_upper: int
    router_model_calls: int
    evidence_sha256: str

    def passes(self, gate: SafetyGate) -> bool:
        return (
            self.training_task_count >= gate.minimum_training_tasks
            and self.calibration_task_count >= gate.minimum_calibration_tasks
            and self.effective_sample_size >= gate.minimum_effective_sample_size
            and self.override_support >= gate.minimum_override_support
            and self.grouped_crossfit_folds >= 2
            and self.task_disjoint_calibration
            and self.harmful_rate_upper <= gate.maximum_harmful_rate_upper
            and self.utility_lcb >= gate.minimum_utility_lcb
            and self.token_delta_upper <= gate.maximum_token_delta_upper
            and self.extra_actor_calls_upper <= gate.maximum_extra_actor_calls
            and self.router_model_calls <= gate.maximum_router_model_calls
        )


@dataclass(frozen=True, slots=True)
class OverrideHead:
    head_id: str
    stage: str
    operation: Operation
    intercept: float
    weights: Mapping[str, float]
    score_threshold: float
    certificate: SafetyCertificate


@dataclass(frozen=True, slots=True)
class Policy:
    policy_id: str
    benchmark: str
    feature_extractor_sha256: str
    max_overrides_per_task: int
    defaults_by_stage: Mapping[str, Operation]
    features: Mapping[str, FeatureSpec]
    safety_gate: SafetyGate
    override_heads: tuple[OverrideHead, ...]
    evidence_documents: Mapping[str, Mapping[str, Any]]
    policy_sha256: str


@dataclass(frozen=True, slots=True)
class RouteDecision:
    task_id: str
    stage: str
    default_operation: Operation
    operation: Operation
    selected_head_id: str | None
    used_default: bool
    reason: str
    trace: Mapping[str, Any]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise PolicyValidationError(
            f"{label} fields differ: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )


def _field_tokens(name: object) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(name).casefold()) if token}


def reject_outcome_fields(value: object, *, path: str = "$") -> None:
    """Reject evaluator-bearing runtime feature names recursively."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            forbidden = sorted(_field_tokens(key) & _FORBIDDEN_RUNTIME_FIELD_TOKENS)
            if forbidden:
                raise PolicyValidationError(f"forbidden runtime field at {path}.{key}: {forbidden}")
            reject_outcome_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_outcome_fields(nested, path=f"{path}[{index}]")


def deterministic_group_folds(
    group_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign complete task groups to balanced deterministic folds."""

    if folds < 2:
        raise ValueError("grouped cross-fitting requires at least two folds")
    if not group_ids or any(not isinstance(value, str) or not value for value in group_ids):
        raise ValueError("group_ids must be non-empty strings")
    unique = set(group_ids)
    if len(unique) != len(group_ids):
        raise ValueError("group_ids must be unique task-level groups")
    if len(unique) < folds:
        raise ValueError("group count must be at least the fold count")
    ordered = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
            value,
        ),
    )
    return {group_id: index % folds for index, group_id in enumerate(ordered)}


def effective_sample_size(weights: Sequence[float]) -> float:
    """Return Kish effective sample size for positive finite weights."""

    if not weights or any(
        not math.isfinite(float(value)) or float(value) <= 0 for value in weights
    ):
        raise ValueError("weights must be positive finite values")
    total = math.fsum(float(value) for value in weights)
    squared = math.fsum(float(value) ** 2 for value in weights)
    return total * total / squared


def _binomial_cdf(events: int, trials: int, probability: float) -> float:
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if events == trials else 0.0
    logs = [
        (
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            + index * math.log(probability)
            + (trials - index) * math.log1p(-probability)
        )
        for index in range(events + 1)
    ]
    maximum = max(logs)
    return math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in logs)


def clopper_pearson_upper(
    events: int,
    trials: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Return the one-sided exact upper confidence bound for a binomial rate."""

    if (
        isinstance(events, bool)
        or isinstance(trials, bool)
        or not isinstance(events, int)
        or not isinstance(trials, int)
        or trials <= 0
        or not 0 <= events <= trials
    ):
        raise ValueError("events/trials are invalid")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if events == trials:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if _binomial_cdf(events, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def clopper_pearson_lower(
    events: int,
    trials: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Return the one-sided exact lower confidence bound for a binomial rate."""

    if (
        isinstance(events, bool)
        or isinstance(trials, bool)
        or not isinstance(events, int)
        or not isinstance(trials, int)
        or trials <= 0
        or not 0 <= events <= trials
    ):
        raise ValueError("events/trials are invalid")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if events == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(
        trials - events,
        trials,
        alpha=alpha,
    )


def calibration_bounds(
    utilities: Sequence[float],
    harmful: Sequence[bool],
    token_deltas: Sequence[float],
    *,
    alpha: float = 0.05,
    bound_method: str = "hoeffding",
) -> dict[str, float | int]:
    """Build conservative task-level bounds for one frozen override set.

    Utilities and normalized token deltas must lie in [-1, 1]. The Hoeffding
    bounds are intentionally conservative and are recomputed from the embedded
    task-level calibration records when a policy is parsed.
    """

    if not utilities or len(utilities) != len(harmful) or len(utilities) != len(token_deltas):
        raise ValueError("calibration vectors must be non-empty and aligned")
    if any(type(value) is not bool for value in harmful):
        raise ValueError("harmful indicators must be booleans")
    for label, values in (
        ("utilities", utilities),
        ("token_deltas", token_deltas),
    ):
        if any(not math.isfinite(float(value)) or not -1 <= float(value) <= 1 for value in values):
            raise ValueError(f"{label} must contain finite values in [-1, 1]")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if bound_method not in CALIBRATION_BOUND_METHODS:
        raise ValueError("unsupported calibration bound method")
    count = len(utilities)
    radius = math.sqrt(2 * math.log(1 / alpha) / count)
    if bound_method == "paired_exact_bonferroni":
        if any(float(value) not in {-1.0, 0.0, 1.0} for value in utilities):
            raise ValueError("paired exact bounds require utilities in {-1, 0, 1}")
        wins = sum(float(value) == 1.0 for value in utilities)
        losses = sum(float(value) == -1.0 for value in utilities)
        utility_lcb = clopper_pearson_lower(
            wins,
            count,
            alpha=alpha / 2,
        ) - clopper_pearson_upper(
            losses,
            count,
            alpha=alpha / 2,
        )
    else:
        utility_lcb = statistics.fmean(utilities) - radius
    return {
        "override_support": count,
        "harmful_count": sum(harmful),
        "harmful_rate_upper": clopper_pearson_upper(
            sum(harmful),
            count,
            alpha=alpha,
        ),
        "utility_lcb": max(-1.0, utility_lcb),
        "token_delta_upper": min(
            1.0,
            statistics.fmean(token_deltas) + radius,
        ),
    }


def build_evidence_document(
    *,
    head_id: str,
    training_task_folds: Mapping[str, int],
    calibration_records: Sequence[Mapping[str, Any]],
    source_matrix_sha256: str,
    head_spec_sha256: str,
    training_result_sha256: str,
    alpha: float = 0.05,
    bound_method: str = "hoeffding",
    reference_policy_id: str | None = None,
    extra_actor_calls_upper: int = 0,
    router_model_calls: int = 0,
) -> dict[str, Any]:
    """Build a self-hashed task-level calibration document for one head."""

    if _IDENTIFIER.fullmatch(head_id) is None:
        raise ValueError("head_id is invalid")
    document: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "TASK_DISJOINT_CALIBRATION_COMPLETE",
        "head_id": head_id,
        "alpha": alpha,
        "bound_method": bound_method,
        "training_task_folds": [
            {"task_sha256": task_sha256, "fold": fold}
            for task_sha256, fold in sorted(training_task_folds.items())
        ],
        "calibration_records": [dict(row) for row in calibration_records],
        "extra_actor_calls_upper": extra_actor_calls_upper,
        "router_model_calls": router_model_calls,
        "source_matrix_sha256": source_matrix_sha256,
        "head_spec_sha256": head_spec_sha256,
        "training_result_sha256": training_result_sha256,
    }
    if reference_policy_id is not None:
        if _IDENTIFIER.fullmatch(reference_policy_id) is None:
            raise ValueError("reference_policy_id is invalid")
        document["reference_policy_id"] = reference_policy_id
    document["canonical_sha256"] = canonical_hash(document)
    _evidence_summary(document, expected_head_id=head_id)
    return document


def _evidence_summary(
    value: object,
    *,
    expected_head_id: str | None = None,
    expected_head_spec_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyValidationError("evidence document must be an object")
    keys = frozenset(
        {
            "schema_version",
            "status",
            "head_id",
            "alpha",
            "bound_method",
            "reference_policy_id",
            "training_task_folds",
            "calibration_records",
            "extra_actor_calls_upper",
            "router_model_calls",
            "source_matrix_sha256",
            "head_spec_sha256",
            "training_result_sha256",
            "canonical_sha256",
        }
    )
    _strict_keys(
        value,
        allowed=keys,
        required=keys - {"bound_method", "reference_policy_id"},
        label="evidence",
    )
    unsigned = dict(value)
    declared = unsigned.pop("canonical_sha256")
    if (
        value["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or value["status"] != "TASK_DISJOINT_CALIBRATION_COMPLETE"
        or declared != canonical_hash(unsigned)
    ):
        raise PolicyValidationError("evidence identity or self-hash is invalid")
    head_id = value["head_id"]
    if (
        not isinstance(head_id, str)
        or _IDENTIFIER.fullmatch(head_id) is None
        or (expected_head_id is not None and head_id != expected_head_id)
    ):
        raise PolicyValidationError("evidence head_id differs")
    alpha = value["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise PolicyValidationError("evidence alpha must be numeric")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise PolicyValidationError("evidence alpha is invalid")
    bound_method = value.get("bound_method", "hoeffding")
    if bound_method not in CALIBRATION_BOUND_METHODS:
        raise PolicyValidationError("evidence bound_method is invalid")
    reference_policy_id = value.get("reference_policy_id")
    if reference_policy_id is not None and (
        not isinstance(reference_policy_id, str)
        or _IDENTIFIER.fullmatch(reference_policy_id) is None
    ):
        raise PolicyValidationError("evidence reference_policy_id is invalid")
    if (
        type(value["extra_actor_calls_upper"]) is not int
        or value["extra_actor_calls_upper"] < 0
        or type(value["router_model_calls"]) is not int
        or value["router_model_calls"] < 0
        or _SHA256.fullmatch(str(value["source_matrix_sha256"])) is None
        or _SHA256.fullmatch(str(value["head_spec_sha256"])) is None
        or _SHA256.fullmatch(str(value["training_result_sha256"])) is None
    ):
        raise PolicyValidationError("evidence cost or source fields are invalid")
    if (
        expected_head_spec_sha256 is not None
        and value["head_spec_sha256"] != expected_head_spec_sha256
    ):
        raise PolicyValidationError("evidence does not bind the deployed head")
    training = value["training_task_folds"]
    calibration = value["calibration_records"]
    if (
        not isinstance(training, list)
        or not training
        or not isinstance(calibration, list)
        or not calibration
    ):
        raise PolicyValidationError("evidence task records must be non-empty lists")
    training_tasks = set()
    folds = set()
    for index, row in enumerate(training):
        if not isinstance(row, Mapping) or set(row) != {"task_sha256", "fold"}:
            raise PolicyValidationError(f"training_task_folds[{index}] fields differ")
        task_sha256 = row["task_sha256"]
        fold = row["fold"]
        if (
            not isinstance(task_sha256, str)
            or _SHA256.fullmatch(task_sha256) is None
            or task_sha256 in training_tasks
            or type(fold) is not int
            or fold < 0
        ):
            raise PolicyValidationError("training task fold record is invalid")
        training_tasks.add(task_sha256)
        folds.add(fold)
    if len(folds) < 2 or folds != set(range(len(folds))):
        raise PolicyValidationError("training folds must be contiguous from zero")
    calibration_tasks = set()
    utilities = []
    harmful = []
    token_deltas = []
    overridden = []
    calibration_record_fields: frozenset[str] | None = None
    for index, row in enumerate(calibration):
        if not isinstance(row, Mapping):
            raise PolicyValidationError(f"calibration_records[{index}] fields differ")
        fields = frozenset(row)
        allowed_fields = {
            frozenset(
                {
                    "task_sha256",
                    "utility",
                    "harmful",
                    "token_delta",
                }
            ),
            frozenset(
                {
                    "task_sha256",
                    "utility",
                    "harmful",
                    "token_delta",
                    "overridden",
                }
            ),
        }
        if fields not in allowed_fields or (
            calibration_record_fields is not None and fields != calibration_record_fields
        ):
            raise PolicyValidationError(f"calibration_records[{index}] fields differ")
        calibration_record_fields = fields
        task_sha256 = row["task_sha256"]
        if (
            not isinstance(task_sha256, str)
            or _SHA256.fullmatch(task_sha256) is None
            or task_sha256 in calibration_tasks
        ):
            raise PolicyValidationError("calibration task identity is invalid")
        calibration_tasks.add(task_sha256)
        if type(row["harmful"]) is not bool:
            raise PolicyValidationError("calibration harmful must be boolean")
        for name in ("utility", "token_delta"):
            item = row[name]
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not -1 <= float(item) <= 1
            ):
                raise PolicyValidationError(f"calibration {name} must be finite in [-1, 1]")
        utilities.append(float(row["utility"]))
        harmful.append(bool(row["harmful"]))
        token_deltas.append(float(row["token_delta"]))
        if "overridden" in row:
            if type(row["overridden"]) is not bool:
                raise PolicyValidationError("calibration overridden must be boolean")
            overridden.append(bool(row["overridden"]))
    if training_tasks & calibration_tasks:
        raise PolicyValidationError("training and calibration tasks overlap")
    bounds = calibration_bounds(
        utilities,
        harmful,
        token_deltas,
        alpha=alpha,
        bound_method=str(bound_method),
    )
    return {
        "training_task_count": len(training_tasks),
        "calibration_task_count": len(calibration_tasks),
        "effective_sample_size": float(len(calibration_tasks)),
        "override_support": (sum(overridden) if overridden else int(bounds["override_support"])),
        "grouped_crossfit_folds": len(folds),
        "task_disjoint_calibration": True,
        "harmful_rate_upper": float(bounds["harmful_rate_upper"]),
        "utility_lcb": float(bounds["utility_lcb"]),
        "token_delta_upper": float(bounds["token_delta_upper"]),
        "extra_actor_calls_upper": int(value["extra_actor_calls_upper"]),
        "router_model_calls": int(value["router_model_calls"]),
        "evidence_sha256": str(declared),
    }


def certificate_from_evidence(
    value: Mapping[str, Any],
    *,
    expected_head_id: str | None = None,
    expected_head_spec_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute the certificate summary from raw task-level evidence."""

    return _evidence_summary(
        value,
        expected_head_id=expected_head_id,
        expected_head_spec_sha256=expected_head_spec_sha256,
    )


def head_spec_hash(
    *,
    head_id: str,
    stage: str,
    operation: Operation | str,
    intercept: float,
    weights: Mapping[str, float],
    score_threshold: float,
    feature_schema: Mapping[str, FeatureSpec | Mapping[str, Any]],
    feature_extractor_sha256: str,
) -> str:
    """Hash every behavior-bearing field of one atomic residual head."""

    parsed_operation = Operation(operation)
    if _SHA256.fullmatch(feature_extractor_sha256) is None:
        raise ValueError("feature_extractor_sha256 is invalid")
    normalized_features = normalized_feature_schema(feature_schema)
    return canonical_hash(
        {
            "head_id": head_id,
            "stage": stage,
            "operation": parsed_operation.value,
            "intercept": float(intercept),
            "weights": {name: float(weight) for name, weight in sorted(weights.items())},
            "score_threshold": float(score_threshold),
            "feature_schema": normalized_features,
            "feature_extractor_sha256": feature_extractor_sha256,
        }
    )


def normalized_feature_schema(
    feature_schema: Mapping[str, FeatureSpec | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the canonical representation of a frozen feature contract."""

    normalized_features = {}
    for name, value in sorted(feature_schema.items()):
        spec = value if isinstance(value, FeatureSpec) else _parse_feature(str(name), value)
        normalized_features[str(name)] = (
            {"kind": spec.kind.value}
            if spec.kind is FeatureKind.BOOLEAN
            else {
                "kind": spec.kind.value,
                "mean": spec.mean,
                "scale": spec.scale,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
            }
        )
    return normalized_features


def feature_schema_hash(
    feature_schema: Mapping[str, FeatureSpec | Mapping[str, Any]],
) -> str:
    return canonical_hash(normalized_feature_schema(feature_schema))


def _parse_feature(name: str, value: object) -> FeatureSpec:
    if _FEATURE_NAME.fullmatch(name) is None:
        raise PolicyValidationError(f"invalid feature name: {name!r}")
    if _field_tokens(name) & _FORBIDDEN_RUNTIME_FIELD_TOKENS:
        raise PolicyValidationError(f"outcome-like feature name: {name!r}")
    if name not in ALLOWED_FEATURE_NAMES:
        raise PolicyValidationError(f"feature {name!r} is not in the frozen runtime allowlist")
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"feature {name!r} must be an object")
    kind_value = value.get("kind")
    try:
        kind = FeatureKind(kind_value)
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(f"feature {name!r} has an invalid kind") from error
    if kind is FeatureKind.BOOLEAN:
        _strict_keys(
            value,
            allowed=frozenset({"kind"}),
            required=frozenset({"kind"}),
            label=f"feature_schema.{name}",
        )
        return FeatureSpec(kind=kind)
    _strict_keys(
        value,
        allowed=frozenset({"kind", "mean", "scale", "minimum", "maximum"}),
        required=frozenset({"kind", "mean", "scale", "minimum", "maximum"}),
        label=f"feature_schema.{name}",
    )
    numeric = {}
    for key in ("mean", "scale", "minimum", "maximum"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PolicyValidationError(f"feature {name!r}.{key} must be numeric")
        numeric[key] = float(item)
        if not math.isfinite(numeric[key]):
            raise PolicyValidationError(f"feature {name!r}.{key} must be finite")
    if (
        numeric["scale"] <= 0
        or numeric["minimum"] > numeric["maximum"]
        or not numeric["minimum"] <= numeric["mean"] <= numeric["maximum"]
    ):
        raise PolicyValidationError(f"feature {name!r} normalization is invalid")
    return FeatureSpec(kind=kind, **numeric)


def _parse_gate(value: object) -> SafetyGate:
    if not isinstance(value, Mapping):
        raise PolicyValidationError("safety_gate must be an object")
    keys = frozenset(
        {
            "minimum_training_tasks",
            "minimum_calibration_tasks",
            "minimum_effective_sample_size",
            "minimum_override_support",
            "maximum_harmful_rate_upper",
            "minimum_utility_lcb",
            "maximum_token_delta_upper",
            "maximum_extra_actor_calls",
            "maximum_router_model_calls",
        }
    )
    _strict_keys(value, allowed=keys, required=keys, label="safety_gate")
    integer_keys = (
        "minimum_training_tasks",
        "minimum_calibration_tasks",
        "minimum_override_support",
        "maximum_extra_actor_calls",
        "maximum_router_model_calls",
    )
    if any(type(value[key]) is not int for key in integer_keys):
        raise PolicyValidationError("integer safety gates must be integers")
    if (
        value["minimum_training_tasks"] < 1
        or value["minimum_calibration_tasks"] < 1
        or value["minimum_override_support"] < 1
        or value["maximum_extra_actor_calls"] != 0
        or value["maximum_router_model_calls"] != 0
    ):
        raise PolicyValidationError("safety gates require positive support and zero extra calls")
    floats = {}
    for key in (
        "minimum_effective_sample_size",
        "maximum_harmful_rate_upper",
        "minimum_utility_lcb",
        "maximum_token_delta_upper",
    ):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PolicyValidationError(f"safety_gate.{key} must be numeric")
        floats[key] = float(item)
        if not math.isfinite(floats[key]):
            raise PolicyValidationError(f"safety_gate.{key} must be finite")
    if (
        floats["minimum_effective_sample_size"] <= 0
        or not 0 <= floats["maximum_harmful_rate_upper"] <= 1
    ):
        raise PolicyValidationError("safety probability/support gates are invalid")
    return SafetyGate(
        minimum_training_tasks=int(value["minimum_training_tasks"]),
        minimum_calibration_tasks=int(value["minimum_calibration_tasks"]),
        minimum_effective_sample_size=floats["minimum_effective_sample_size"],
        minimum_override_support=int(value["minimum_override_support"]),
        maximum_harmful_rate_upper=floats["maximum_harmful_rate_upper"],
        minimum_utility_lcb=floats["minimum_utility_lcb"],
        maximum_token_delta_upper=floats["maximum_token_delta_upper"],
        maximum_extra_actor_calls=0,
        maximum_router_model_calls=0,
    )


def _parse_certificate(value: object, *, label: str) -> SafetyCertificate:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{label} must be an object")
    keys = frozenset(
        {
            "training_task_count",
            "calibration_task_count",
            "effective_sample_size",
            "override_support",
            "grouped_crossfit_folds",
            "task_disjoint_calibration",
            "harmful_rate_upper",
            "utility_lcb",
            "token_delta_upper",
            "extra_actor_calls_upper",
            "router_model_calls",
            "evidence_sha256",
        }
    )
    _strict_keys(value, allowed=keys, required=keys, label=label)
    integer_keys = (
        "training_task_count",
        "calibration_task_count",
        "override_support",
        "grouped_crossfit_folds",
        "extra_actor_calls_upper",
        "router_model_calls",
    )
    if any(type(value[key]) is not int for key in integer_keys):
        raise PolicyValidationError(f"{label} count fields must be integers")
    if (
        value["training_task_count"] < 1
        or value["calibration_task_count"] < 1
        or value["override_support"] < 1
        or value["override_support"] > value["calibration_task_count"]
        or not 2 <= value["grouped_crossfit_folds"] <= 20
        or value["extra_actor_calls_upper"] < 0
        or value["router_model_calls"] < 0
        or type(value["task_disjoint_calibration"]) is not bool
    ):
        raise PolicyValidationError(f"{label} count/support fields are invalid")
    floats = {}
    for key in (
        "effective_sample_size",
        "harmful_rate_upper",
        "utility_lcb",
        "token_delta_upper",
    ):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PolicyValidationError(f"{label}.{key} must be numeric")
        floats[key] = float(item)
        if not math.isfinite(floats[key]):
            raise PolicyValidationError(f"{label}.{key} must be finite")
    if (
        floats["effective_sample_size"] <= 0
        or floats["effective_sample_size"] > value["calibration_task_count"]
        or not 0 <= floats["harmful_rate_upper"] <= 1
        or not -1 <= floats["utility_lcb"] <= 1
        or not -1 <= floats["token_delta_upper"] <= 1
        or _SHA256.fullmatch(str(value["evidence_sha256"])) is None
    ):
        raise PolicyValidationError(f"{label} bounds or evidence hash are invalid")
    return SafetyCertificate(
        training_task_count=int(value["training_task_count"]),
        calibration_task_count=int(value["calibration_task_count"]),
        effective_sample_size=floats["effective_sample_size"],
        override_support=int(value["override_support"]),
        grouped_crossfit_folds=int(value["grouped_crossfit_folds"]),
        task_disjoint_calibration=bool(value["task_disjoint_calibration"]),
        harmful_rate_upper=floats["harmful_rate_upper"],
        utility_lcb=floats["utility_lcb"],
        token_delta_upper=floats["token_delta_upper"],
        extra_actor_calls_upper=int(value["extra_actor_calls_upper"]),
        router_model_calls=int(value["router_model_calls"]),
        evidence_sha256=str(value["evidence_sha256"]),
    )


def parse_policy(value: Mapping[str, Any]) -> Policy:
    """Parse and validate a frozen residual routing policy."""

    if not isinstance(value, Mapping):
        raise PolicyValidationError("policy must be an object")
    keys = frozenset(
        {
            "schema_version",
            "policy_id",
            "benchmark",
            "feature_extractor_sha256",
            "max_overrides_per_task",
            "defaults_by_stage",
            "feature_schema",
            "safety_gate",
            "override_heads",
            "evidence_documents",
            "policy_sha256",
        }
    )
    _strict_keys(
        value,
        allowed=keys,
        required=keys - {"policy_sha256"},
        label="policy",
    )
    if value["schema_version"] != POLICY_SCHEMA_VERSION:
        raise PolicyValidationError("unsupported policy schema_version")
    policy_id = value["policy_id"]
    benchmark = value["benchmark"]
    feature_extractor_sha256 = value["feature_extractor_sha256"]
    if not isinstance(policy_id, str) or _IDENTIFIER.fullmatch(policy_id) is None:
        raise PolicyValidationError("policy_id is invalid")
    if not isinstance(benchmark, str) or not benchmark.strip() or len(benchmark) > 128:
        raise PolicyValidationError("benchmark is invalid")
    if (
        not isinstance(feature_extractor_sha256, str)
        or _SHA256.fullmatch(feature_extractor_sha256) is None
    ):
        raise PolicyValidationError("feature_extractor_sha256 is invalid")
    max_overrides = value["max_overrides_per_task"]
    if type(max_overrides) is not int or max_overrides != 1:
        raise PolicyValidationError("max_overrides_per_task must equal one")
    defaults_value = value["defaults_by_stage"]
    if not isinstance(defaults_value, Mapping) or not defaults_value:
        raise PolicyValidationError("defaults_by_stage must be a non-empty object")
    defaults = {}
    for stage, operation in defaults_value.items():
        if not isinstance(stage, str) or _IDENTIFIER.fullmatch(stage) is None:
            raise PolicyValidationError("default stage is invalid")
        try:
            defaults[stage] = Operation(operation)
        except (TypeError, ValueError) as error:
            raise PolicyValidationError(f"default operation for {stage!r} is invalid") from error
    feature_value = value["feature_schema"]
    if (
        not isinstance(feature_value, Mapping)
        or not feature_value
        or len(feature_value) > MAX_FEATURES
    ):
        raise PolicyValidationError(f"feature_schema must define 1..{MAX_FEATURES} features")
    features = {
        str(name): _parse_feature(str(name), spec) for name, spec in sorted(feature_value.items())
    }
    gate = _parse_gate(value["safety_gate"])
    evidence_value = value["evidence_documents"]
    if not isinstance(evidence_value, Mapping):
        raise PolicyValidationError("evidence_documents must be an object")
    evidence_documents = {
        str(head_id): dict(document)
        for head_id, document in evidence_value.items()
        if isinstance(document, Mapping)
    }
    if len(evidence_documents) != len(evidence_value):
        raise PolicyValidationError("evidence_documents contain a non-object")
    heads_value = value["override_heads"]
    if not isinstance(heads_value, list) or len(heads_value) > MAX_OVERRIDE_HEADS:
        raise PolicyValidationError(
            f"override_heads must contain at most {MAX_OVERRIDE_HEADS} heads"
        )
    heads = []
    seen_head_ids = set()
    for index, item in enumerate(heads_value):
        label = f"override_heads[{index}]"
        if not isinstance(item, Mapping):
            raise PolicyValidationError(f"{label} must be an object")
        head_keys = frozenset(
            {
                "head_id",
                "stage",
                "operation",
                "intercept",
                "weights",
                "score_threshold",
                "certificate",
            }
        )
        _strict_keys(item, allowed=head_keys, required=head_keys, label=label)
        head_id = item["head_id"]
        stage = item["stage"]
        if not isinstance(head_id, str) or _IDENTIFIER.fullmatch(head_id) is None:
            raise PolicyValidationError(f"{label}.head_id is invalid")
        if head_id in seen_head_ids:
            raise PolicyValidationError("override head IDs must be unique")
        seen_head_ids.add(head_id)
        if stage not in defaults:
            raise PolicyValidationError(f"{label}.stage lacks a default")
        try:
            operation = Operation(item["operation"])
        except (TypeError, ValueError) as error:
            raise PolicyValidationError(f"{label}.operation is invalid") from error
        if operation is defaults[stage]:
            raise PolicyValidationError(f"{label} does not override its static default")
        weights_value = item["weights"]
        if not isinstance(weights_value, Mapping) or not weights_value:
            raise PolicyValidationError(f"{label}.weights must be non-empty")
        if not set(weights_value) <= set(features):
            raise PolicyValidationError(f"{label}.weights name unknown features")
        weights = {}
        for name, raw_weight in weights_value.items():
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise PolicyValidationError(f"{label}.weights must be numeric")
            weight = float(raw_weight)
            if not math.isfinite(weight):
                raise PolicyValidationError(f"{label}.weights must be finite")
            weights[str(name)] = weight
        numeric_values = {}
        for name in ("intercept", "score_threshold"):
            raw = item[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise PolicyValidationError(f"{label}.{name} must be numeric")
            numeric_values[name] = float(raw)
            if not math.isfinite(numeric_values[name]):
                raise PolicyValidationError(f"{label}.{name} must be finite")
        certificate = _parse_certificate(
            item["certificate"],
            label=f"{label}.certificate",
        )
        evidence = evidence_documents.get(head_id)
        if evidence is None:
            raise PolicyValidationError(f"{label} lacks raw calibration evidence")
        deployed_head_sha256 = head_spec_hash(
            head_id=head_id,
            stage=str(stage),
            operation=operation,
            intercept=numeric_values["intercept"],
            weights=weights,
            score_threshold=numeric_values["score_threshold"],
            feature_schema=features,
            feature_extractor_sha256=feature_extractor_sha256,
        )
        recomputed = certificate_from_evidence(
            evidence,
            expected_head_id=head_id,
            expected_head_spec_sha256=deployed_head_sha256,
        )
        declared_certificate = {
            "training_task_count": certificate.training_task_count,
            "calibration_task_count": certificate.calibration_task_count,
            "effective_sample_size": certificate.effective_sample_size,
            "override_support": certificate.override_support,
            "grouped_crossfit_folds": certificate.grouped_crossfit_folds,
            "task_disjoint_calibration": certificate.task_disjoint_calibration,
            "harmful_rate_upper": certificate.harmful_rate_upper,
            "utility_lcb": certificate.utility_lcb,
            "token_delta_upper": certificate.token_delta_upper,
            "extra_actor_calls_upper": certificate.extra_actor_calls_upper,
            "router_model_calls": certificate.router_model_calls,
            "evidence_sha256": certificate.evidence_sha256,
        }
        for key, observed_value in recomputed.items():
            declared_value = declared_certificate[key]
            if isinstance(observed_value, float):
                if not math.isclose(
                    observed_value,
                    float(declared_value),
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise PolicyValidationError(f"{label} certificate differs from raw evidence")
            elif observed_value != declared_value:
                raise PolicyValidationError(f"{label} certificate differs from raw evidence")
        if not certificate.passes(gate):
            raise PolicyValidationError(f"{label} lacks a passing task-level safety certificate")
        heads.append(
            OverrideHead(
                head_id=head_id,
                stage=str(stage),
                operation=operation,
                intercept=numeric_values["intercept"],
                weights=weights,
                score_threshold=numeric_values["score_threshold"],
                certificate=certificate,
            )
        )
    if set(evidence_documents) != seen_head_ids:
        raise PolicyValidationError("evidence_documents must match override head IDs exactly")
    declared = value.get("policy_sha256")
    provisional = Policy(
        policy_id=policy_id,
        benchmark=benchmark.strip(),
        feature_extractor_sha256=feature_extractor_sha256,
        max_overrides_per_task=max_overrides,
        defaults_by_stage=defaults,
        features=features,
        safety_gate=gate,
        override_heads=tuple(heads),
        evidence_documents=evidence_documents,
        policy_sha256="",
    )
    observed = canonical_hash(policy_to_dict(provisional, include_hash=False))
    if declared is not None and declared != observed:
        raise PolicyValidationError("policy_sha256 mismatch")
    return Policy(
        policy_id=provisional.policy_id,
        benchmark=provisional.benchmark,
        feature_extractor_sha256=provisional.feature_extractor_sha256,
        max_overrides_per_task=provisional.max_overrides_per_task,
        defaults_by_stage=provisional.defaults_by_stage,
        features=provisional.features,
        safety_gate=provisional.safety_gate,
        override_heads=provisional.override_heads,
        evidence_documents=provisional.evidence_documents,
        policy_sha256=observed,
    )


def policy_to_dict(policy: Policy, *, include_hash: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "benchmark": policy.benchmark,
        "feature_extractor_sha256": policy.feature_extractor_sha256,
        "max_overrides_per_task": policy.max_overrides_per_task,
        "defaults_by_stage": {
            stage: operation.value for stage, operation in sorted(policy.defaults_by_stage.items())
        },
        "feature_schema": {
            name: (
                {"kind": spec.kind.value}
                if spec.kind is FeatureKind.BOOLEAN
                else {
                    "kind": spec.kind.value,
                    "mean": spec.mean,
                    "scale": spec.scale,
                    "minimum": spec.minimum,
                    "maximum": spec.maximum,
                }
            )
            for name, spec in sorted(policy.features.items())
        },
        "safety_gate": {
            "minimum_training_tasks": policy.safety_gate.minimum_training_tasks,
            "minimum_calibration_tasks": (policy.safety_gate.minimum_calibration_tasks),
            "minimum_effective_sample_size": (policy.safety_gate.minimum_effective_sample_size),
            "minimum_override_support": policy.safety_gate.minimum_override_support,
            "maximum_harmful_rate_upper": (policy.safety_gate.maximum_harmful_rate_upper),
            "minimum_utility_lcb": policy.safety_gate.minimum_utility_lcb,
            "maximum_token_delta_upper": (policy.safety_gate.maximum_token_delta_upper),
            "maximum_extra_actor_calls": 0,
            "maximum_router_model_calls": 0,
        },
        "override_heads": [
            {
                "head_id": head.head_id,
                "stage": head.stage,
                "operation": head.operation.value,
                "intercept": head.intercept,
                "weights": dict(sorted(head.weights.items())),
                "score_threshold": head.score_threshold,
                "certificate": {
                    "training_task_count": head.certificate.training_task_count,
                    "calibration_task_count": (head.certificate.calibration_task_count),
                    "effective_sample_size": (head.certificate.effective_sample_size),
                    "override_support": head.certificate.override_support,
                    "grouped_crossfit_folds": (head.certificate.grouped_crossfit_folds),
                    "task_disjoint_calibration": (head.certificate.task_disjoint_calibration),
                    "harmful_rate_upper": head.certificate.harmful_rate_upper,
                    "utility_lcb": head.certificate.utility_lcb,
                    "token_delta_upper": head.certificate.token_delta_upper,
                    "extra_actor_calls_upper": (head.certificate.extra_actor_calls_upper),
                    "router_model_calls": head.certificate.router_model_calls,
                    "evidence_sha256": head.certificate.evidence_sha256,
                },
            }
            for head in policy.override_heads
        ],
        "evidence_documents": {
            head_id: dict(document)
            for head_id, document in sorted(policy.evidence_documents.items())
        },
    }
    if include_hash:
        value["policy_sha256"] = policy.policy_sha256
    return value


class PolicySession:
    """Apply one frozen residual policy to one complete task episode."""

    def __init__(self, policy: Policy, *, task_id: str, benchmark: str) -> None:
        if not task_id:
            raise PolicyValidationError("task_id must be non-empty")
        if benchmark != policy.benchmark:
            raise PolicyValidationError("runtime benchmark differs from policy")
        self.policy = policy
        self.task_id = task_id
        self._override_count = 0
        self._evaluated_head_ids: set[str] = set()
        self._decisions: list[RouteDecision] = []

    @property
    def decisions(self) -> tuple[RouteDecision, ...]:
        return tuple(self._decisions)

    def _default(
        self,
        *,
        stage: str,
        features: Mapping[str, Any],
        reason: str,
        head_scores: Sequence[Mapping[str, Any]] = (),
    ) -> RouteDecision:
        default = self.policy.defaults_by_stage[stage]
        trace: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.policy_sha256,
            "task_id": self.task_id,
            "stage": stage,
            "feature_sha256": canonical_hash(features),
            "head_scores": list(head_scores),
            "default_operation": default.value,
            "selected_head_id": None,
            "effective_operation": default.value,
            "used_default": True,
            "reason": reason,
            "override_count_before": self._override_count,
            "router_model_calls": 0,
            "extra_actor_calls": 0,
            "uses_gold_or_evaluator_signals": False,
        }
        trace["trace_sha256"] = canonical_hash(trace)
        decision = RouteDecision(
            task_id=self.task_id,
            stage=stage,
            default_operation=default,
            operation=default,
            selected_head_id=None,
            used_default=True,
            reason=reason,
            trace=trace,
        )
        self._decisions.append(decision)
        return decision

    def choose(
        self,
        *,
        stage: str,
        features: Mapping[str, Any],
    ) -> RouteDecision:
        if stage not in self.policy.defaults_by_stage:
            raise PolicyValidationError(f"unknown residual stage: {stage!r}")
        if not isinstance(features, Mapping):
            raise PolicyValidationError("runtime features must be an object")
        reject_outcome_fields(features, path="$.features")
        matching_heads = [
            head
            for head in self.policy.override_heads
            if head.stage == stage and head.head_id not in self._evaluated_head_ids
        ]
        if not matching_heads and any(head.stage == stage for head in self.policy.override_heads):
            return self._default(
                stage=stage,
                features=features,
                reason="atomic residual head was already evaluated for this task",
            )
        self._evaluated_head_ids.update(head.head_id for head in matching_heads)
        missing = sorted(set(self.policy.features) - set(features))
        if missing:
            return self._default(
                stage=stage,
                features=features,
                reason=f"missing calibrated features: {missing}",
            )
        normalized = {}
        try:
            for name, spec in self.policy.features.items():
                normalized[name] = spec.normalize(features[name])
        except PolicyValidationError as error:
            return self._default(
                stage=stage,
                features=features,
                reason=f"feature contract fallback: {error}",
            )
        if self._override_count >= self.policy.max_overrides_per_task:
            return self._default(
                stage=stage,
                features=features,
                reason="per-task override budget exhausted",
            )
        head_scores = []
        eligible = []
        for head in matching_heads:
            score = head.intercept + math.fsum(
                weight * normalized[name] for name, weight in head.weights.items()
            )
            margin = score - head.score_threshold
            row = {
                "head_id": head.head_id,
                "operation": head.operation.value,
                "score": score,
                "score_threshold": head.score_threshold,
                "margin": margin,
                "certificate_evidence_sha256": head.certificate.evidence_sha256,
            }
            head_scores.append(row)
            if margin >= 0:
                eligible.append((margin, score, head.head_id, head))
        if not eligible:
            return self._default(
                stage=stage,
                features=features,
                reason="no certified residual head crossed its frozen threshold",
                head_scores=head_scores,
            )
        _, _, _, selected = min(
            eligible,
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        default = self.policy.defaults_by_stage[stage]
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.policy_sha256,
            "task_id": self.task_id,
            "stage": stage,
            "feature_sha256": canonical_hash(features),
            "head_scores": head_scores,
            "default_operation": default.value,
            "selected_head_id": selected.head_id,
            "effective_operation": selected.operation.value,
            "used_default": False,
            "reason": "certified residual utility threshold passed",
            "override_count_before": self._override_count,
            "router_model_calls": 0,
            "extra_actor_calls": 0,
            "uses_gold_or_evaluator_signals": False,
        }
        trace["trace_sha256"] = canonical_hash(trace)
        decision = RouteDecision(
            task_id=self.task_id,
            stage=stage,
            default_operation=default,
            operation=selected.operation,
            selected_head_id=selected.head_id,
            used_default=False,
            reason=str(trace["reason"]),
            trace=trace,
        )
        self._override_count += 1
        self._decisions.append(decision)
        return decision


__all__ = [
    "ALLOWED_FEATURE_NAMES",
    "EVIDENCE_SCHEMA_VERSION",
    "MAX_FEATURES",
    "MAX_OVERRIDE_HEADS",
    "POLICY_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "FeatureKind",
    "FeatureSpec",
    "Operation",
    "OverrideHead",
    "Policy",
    "PolicySession",
    "PolicyValidationError",
    "RouteDecision",
    "SafetyCertificate",
    "SafetyGate",
    "build_evidence_document",
    "calibration_bounds",
    "canonical_hash",
    "canonical_json",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "certificate_from_evidence",
    "deterministic_group_folds",
    "effective_sample_size",
    "feature_schema_hash",
    "head_spec_hash",
    "normalized_feature_schema",
    "parse_policy",
    "policy_to_dict",
    "reject_outcome_fields",
]
