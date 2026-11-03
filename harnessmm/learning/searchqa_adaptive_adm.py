"""Outcome-blind adaptive ADD/DELETE/MODIFY policy for SearchQA.

The policy language is deliberately small.  Policies can inspect only the
features produced here, select one cached response arm, and request at most one
repair.  Unknown fields, outcome-like fields, and ill-typed comparisons are
rejected before a policy can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


POLICY_SCHEMA_VERSION = "phiagent-searchqa-adaptive-adm-policy/v1"
TRACE_SCHEMA_VERSION = "phiagent-searchqa-adaptive-adm-trace/v1"


class PolicyValidationError(ValueError):
    """Raised when a policy or its outcome-blind inputs are unsafe."""


class Stage(str, Enum):
    INITIAL = "INITIAL"
    REPAIR = "REPAIR"


class Operation(str, Enum):
    KEEP = "KEEP"
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"


_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "answer",
        "answers",
        "accuracy",
        "correct",
        "correctness",
        "em",
        "evaluator",
        "f1",
        "gold",
        "grade",
        "hard",
        "label",
        "loss",
        "metric",
        "outcome",
        "outcomes",
        "reward",
        "result",
        "results",
        "score",
        "scores",
        "soft",
        "success",
        "target",
        "verdict",
        "win",
        "winner",
    }
)
_FORBIDDEN_FIELD_NAMES = frozenset(
    {"exact_match", "exactmatch", "ground_truth", "groundtruth"}
)
_TASK_FIELDS = frozenset({"id", "question", "context"})
_DRAFT_FIELDS = frozenset({"response", "usage", "response_sha256", "arm"})
_USAGE_FIELDS = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens"}
)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_REFUSAL = re.compile(
    r"\b(?:cannot|can't|unable to|insufficient information|not enough "
    r"information|no information|i do not know|i don't know|unknown)\b",
    re.IGNORECASE,
)


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically, rejecting non-finite numbers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    """Return a SHA-256 over canonical UTF-8 JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _field_tokens(name: object) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(name).casefold())
        if token
    }


def reject_forbidden_fields(value: object, *, path: str = "$") -> None:
    """Recursively reject evaluator/outcome-bearing field names."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            tokens = _field_tokens(key)
            forbidden = sorted(tokens & _FORBIDDEN_FIELD_TOKENS)
            normalized_name = re.sub(
                r"[^a-z0-9]+", "_", str(key).casefold()
            ).strip("_")
            if forbidden or normalized_name in _FORBIDDEN_FIELD_NAMES:
                raise PolicyValidationError(
                    f"forbidden field at {path}.{key}: {forbidden}"
                )
            reject_forbidden_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_forbidden_fields(nested, path=f"{path}[{index}]")


def _normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _expected_answer_type(question: str) -> str:
    value = _normalized_text(question)
    if re.search(
        r"\b(?:how many|what year|which year|what date|what percentage|how old)\b",
        value,
    ):
        return "numeric"
    if re.match(r"^(?:who|whose)\b", value):
        return "person"
    if re.match(
        r"^(?:is|are|was|were|do|does|did|can|could|has|have|had|will|would)\b",
        value,
    ):
        return "yes_no"
    letter_count = re.search(r"\b(\d+)[ -]letter\b", value)
    return f"letters:{letter_count.group(1)}" if letter_count else "open"


def _type_fit(expected: str, draft: str) -> bool:
    normalized = _normalized_text(draft)
    if not normalized:
        return False
    if expected == "numeric":
        return bool(re.search(r"\d", draft))
    if expected == "person":
        return bool(re.search(r"[A-Za-z]", draft)) and normalized not in {
            "yes",
            "no",
            "true",
            "false",
        }
    if expected == "yes_no":
        return normalized in {"yes", "no", "true", "false"}
    if expected.startswith("letters:"):
        expected_count = int(expected.partition(":")[2])
        return sum(character.isalpha() for character in draft) == expected_count
    return True


FEATURE_TYPES: Mapping[str, type] = {
    "stage": str,
    "question_character_count": int,
    "question_token_count": int,
    "question_named_token_count": int,
    "question_has_quoted_span": bool,
    "question_has_relation_cue": bool,
    "question_context_coverage": float,
    "question_context_coverage_bucket": str,
    "context_character_count": int,
    "context_token_count": int,
    "document_count": int,
    "document_count_bucket": str,
    "expected_answer_type": str,
    "draft_present": bool,
    "draft_valid_single_tag": bool,
    "draft_tag_count": int,
    "draft_nonempty": bool,
    "draft_refusal": bool,
    "draft_token_count": int,
    "draft_token_count_bucket": str,
    "draft_character_count": int,
    "draft_explanation_token_count": int,
    "draft_has_explanation": bool,
    "draft_context_occurrence_count": int,
    "draft_context_occurrence_bucket": str,
    "draft_exact_context_support": bool,
    "draft_lexical_coverage": float,
    "draft_question_overlap": float,
    "draft_type_fit": bool,
    "prior_operation_count": int,
    "previous_operation": str,
}
_INITIAL_FEATURES = frozenset(
    {
        "stage",
        "question_character_count",
        "question_token_count",
        "question_named_token_count",
        "question_has_quoted_span",
        "question_has_relation_cue",
        "question_context_coverage",
        "question_context_coverage_bucket",
        "context_character_count",
        "context_token_count",
        "document_count",
        "document_count_bucket",
        "expected_answer_type",
        "draft_present",
        "prior_operation_count",
        "previous_operation",
    }
)
_REPAIR_FEATURES = frozenset(FEATURE_TYPES)
STAGE_FEATURES: Mapping[Stage, frozenset[str]] = {
    Stage.INITIAL: _INITIAL_FEATURES,
    Stage.REPAIR: _REPAIR_FEATURES,
}
MAX_POLICY_RULES = 64
MAX_RULE_CONDITIONS = 8
_RELATION_CUE = re.compile(
    r"\b(?:after|before|born|capital|daughter|father|founded|husband|mother|"
    r"named after|preceded|sister|son|succeeded|wife)\b",
    re.IGNORECASE,
)


def _strict_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
    *,
    required: frozenset[str] = frozenset(),
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise PolicyValidationError(
            f"{label} fields differ: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )


def extract_features(
    task: Mapping[str, Any],
    *,
    stage: Stage | str,
    draft: str | Mapping[str, Any] | None = None,
    prior_operations: Sequence[Operation | str] = (),
) -> dict[str, bool | int | float | str]:
    """Extract the complete, outcome-blind feature vector used by the DSL."""
    reject_forbidden_fields(task, path="$.task")
    _strict_keys(task, _TASK_FIELDS, "task", required=_TASK_FIELDS)
    if not all(isinstance(task[key], str) for key in _TASK_FIELDS):
        raise PolicyValidationError("task id, question, and context must be strings")
    if not task["id"]:
        raise PolicyValidationError("task id must be non-empty")
    try:
        parsed_stage = Stage(stage)
        parsed_prior = tuple(Operation(value) for value in prior_operations)
    except ValueError as exc:
        raise PolicyValidationError(str(exc)) from exc
    if parsed_stage is Stage.INITIAL and parsed_prior:
        raise PolicyValidationError("INITIAL decisions cannot have prior operations")
    if parsed_stage is Stage.REPAIR and len(parsed_prior) != 1:
        raise PolicyValidationError("REPAIR requires exactly one prior operation")
    if parsed_stage is Stage.INITIAL and draft is not None:
        raise PolicyValidationError("INITIAL decisions cannot inspect a draft")
    if parsed_stage is Stage.REPAIR and draft is None:
        raise PolicyValidationError("REPAIR decisions require a draft")

    response = ""
    draft_present = draft is not None
    if isinstance(draft, Mapping):
        reject_forbidden_fields(draft, path="$.draft")
        _strict_keys(draft, _DRAFT_FIELDS, "draft")
        if not isinstance(draft.get("response"), str):
            raise PolicyValidationError("draft.response must be a string")
        usage = draft.get("usage")
        if usage is not None:
            if not isinstance(usage, Mapping):
                raise PolicyValidationError("draft.usage must be an object")
            _strict_keys(usage, _USAGE_FIELDS, "draft.usage")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in usage.values()
            ):
                raise PolicyValidationError(
                    "draft usage values must be non-negative integers"
                )
        response = draft["response"]
    elif isinstance(draft, str):
        response = draft
    elif draft is not None:
        raise PolicyValidationError("draft must be a string, object, or null")

    question = task["question"]
    context = task["context"]
    question_norm = _normalized_text(question)
    context_norm = _normalized_text(context)
    tags = [match.strip() for match in _ANSWER_TAG.findall(response)]
    answer = tags[-1] if tags else response.strip().splitlines()[-1] if response.strip() else ""
    answer_norm = _normalized_text(answer)
    answer_tokens = answer_norm.split()
    context_tokens = set(context_norm.split())
    question_tokens = set(question_norm.split())
    question_content_tokens = {
        token
        for token in question_tokens
        if len(token) > 2 and token not in {"the", "and", "for", "that", "this", "with"}
    }
    question_context_coverage = (
        sum(token in context_tokens for token in question_content_tokens)
        / len(question_content_tokens)
        if question_content_tokens
        else 0.0
    )
    question_context_coverage_bucket = (
        "low"
        if question_context_coverage < 0.4
        else "medium"
        if question_context_coverage < 0.75
        else "high"
    )
    lexical_coverage = (
        sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )
    question_overlap = (
        sum(token in question_tokens for token in answer_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )
    outside_answer = _ANSWER_TAG.sub(" ", response)
    explanation_tokens = _normalized_text(outside_answer).split()
    context_occurrences = (
        len(
            re.findall(
                rf"(?:^| ){re.escape(answer_norm)}(?: |$)",
                context_norm,
            )
        )
        if answer_norm
        else 0
    )
    expected_type = _expected_answer_type(question)
    document_count = max(1, context.count("[DOC]")) if context else 0
    document_count_bucket = (
        "low"
        if document_count <= 5
        else "medium"
        if document_count <= 10
        else "high"
    )
    draft_token_count = len(answer_tokens)
    draft_token_count_bucket = (
        "short"
        if draft_token_count <= 3
        else "medium"
        if draft_token_count <= 8
        else "long"
    )
    draft_context_occurrence_bucket = (
        "zero"
        if context_occurrences == 0
        else "one"
        if context_occurrences == 1
        else "many"
    )
    features: dict[str, bool | int | float | str] = {
        "stage": parsed_stage.value,
        "question_character_count": len(question),
        "question_token_count": len(question_norm.split()),
        "question_named_token_count": sum(
            token[:1].isupper()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9'.-]*", question)
        ),
        "question_has_quoted_span": bool(
            re.search(r"""["'][^"']+["']""", question)
        ),
        "question_has_relation_cue": bool(_RELATION_CUE.search(question)),
        "question_context_coverage": question_context_coverage,
        "question_context_coverage_bucket": question_context_coverage_bucket,
        "context_character_count": len(context),
        "context_token_count": len(context_norm.split()),
        "document_count": document_count,
        "document_count_bucket": document_count_bucket,
        "expected_answer_type": expected_type,
        "draft_present": draft_present,
        "draft_valid_single_tag": len(tags) == 1 and bool(tags[0]),
        "draft_tag_count": len(tags),
        "draft_nonempty": bool(answer_norm),
        "draft_refusal": bool(_REFUSAL.search(response)),
        "draft_token_count": draft_token_count,
        "draft_token_count_bucket": draft_token_count_bucket,
        "draft_character_count": len(answer),
        "draft_explanation_token_count": len(explanation_tokens),
        "draft_has_explanation": len(explanation_tokens) > 12,
        "draft_context_occurrence_count": context_occurrences,
        "draft_context_occurrence_bucket": draft_context_occurrence_bucket,
        "draft_exact_context_support": bool(
            context_occurrences
        ),
        "draft_lexical_coverage": lexical_coverage,
        "draft_question_overlap": question_overlap,
        "draft_type_fit": _type_fit(expected_type, answer),
        "prior_operation_count": len(parsed_prior),
        "previous_operation": parsed_prior[-1].value if parsed_prior else "",
    }
    return features


@dataclass(frozen=True)
class Condition:
    feature: str
    comparator: str
    value: bool | int | float | str | tuple[bool | int | float | str, ...]


@dataclass(frozen=True)
class Decision:
    operation: Operation
    reason: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    stage: Stage
    match: str
    when: tuple[Condition, ...]
    decision: Decision


@dataclass(frozen=True)
class Policy:
    policy_id: str
    rules: tuple[Rule, ...]
    defaults: Mapping[Stage, Decision]
    max_repairs: int
    policy_sha256: str


@dataclass(frozen=True)
class PolicyDecision:
    task_id: str
    stage: Stage
    operation: Operation
    proposed_operation: Operation
    reason: str
    matched_rule_id: str | None
    duplicate_suppressed: bool
    trace: Mapping[str, Any]


_COMPARATORS = frozenset(
    {"eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in", "contains"}
)


def _is_exact_type(value: object, expected: type) -> bool:
    if expected is float:
        return not isinstance(value, bool) and isinstance(value, (int, float))
    return type(value) is expected


def _parse_condition(value: object, label: str) -> Condition:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{label} must be an object")
    _strict_keys(
        value,
        frozenset({"feature", "comparator", "value"}),
        label,
        required=frozenset({"feature", "comparator", "value"}),
    )
    feature = value["feature"]
    comparator = value["comparator"]
    if not isinstance(feature, str) or feature not in FEATURE_TYPES:
        raise PolicyValidationError(f"{label} references unknown feature")
    if not isinstance(comparator, str) or comparator not in _COMPARATORS:
        raise PolicyValidationError(f"{label} has unknown comparator")
    expected = FEATURE_TYPES[feature]
    operand = value["value"]
    if comparator in {"in", "not_in"}:
        if not isinstance(operand, list) or not operand:
            raise PolicyValidationError(f"{label} membership value must be a nonempty list")
        if any(not _is_exact_type(item, expected) for item in operand):
            raise PolicyValidationError(f"{label} membership values have wrong type")
        parsed: Any = tuple(operand)
    else:
        if not _is_exact_type(operand, expected):
            raise PolicyValidationError(f"{label} value has wrong type")
        if isinstance(operand, float) and not math.isfinite(operand):
            raise PolicyValidationError(f"{label} value must be finite")
        if comparator in {"lt", "lte", "gt", "gte"} and expected not in {int, float}:
            raise PolicyValidationError(f"{label} ordering requires a numeric feature")
        if comparator == "contains" and expected is not str:
            raise PolicyValidationError(f"{label} contains requires a string feature")
        parsed = operand
    return Condition(feature=feature, comparator=str(comparator), value=parsed)


def _parse_decision(value: object, label: str) -> Decision:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{label} must be an object")
    _strict_keys(
        value,
        frozenset({"operation", "reason"}),
        label,
        required=frozenset({"operation", "reason"}),
    )
    try:
        operation = Operation(value["operation"])
    except (ValueError, TypeError) as exc:
        raise PolicyValidationError(f"{label} operation is invalid") from exc
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        raise PolicyValidationError(f"{label} reason must contain 1..240 characters")
    return Decision(operation=operation, reason=reason)


def parse_policy(value: Mapping[str, Any]) -> Policy:
    """Parse and type-check one policy, including its optional self-hash."""
    if not isinstance(value, Mapping):
        raise PolicyValidationError("policy must be an object")
    reject_forbidden_fields(value, path="$.policy")
    _strict_keys(
        value,
        frozenset(
            {
                "schema_version",
                "policy_id",
                "max_repairs",
                "defaults",
                "rules",
                "policy_sha256",
            }
        ),
        "policy",
        required=frozenset(
            {"schema_version", "policy_id", "max_repairs", "defaults", "rules"}
        ),
    )
    if value["schema_version"] != POLICY_SCHEMA_VERSION:
        raise PolicyValidationError("unsupported policy schema_version")
    policy_id = value["policy_id"]
    if (
        not isinstance(policy_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", policy_id)
    ):
        raise PolicyValidationError("policy_id is invalid")
    if type(value["max_repairs"]) is not int or value["max_repairs"] != 1:
        raise PolicyValidationError("max_repairs must equal one")
    defaults_value = value["defaults"]
    if not isinstance(defaults_value, Mapping):
        raise PolicyValidationError("defaults must be an object")
    if set(defaults_value) != {stage.value for stage in Stage}:
        raise PolicyValidationError("defaults must define exactly INITIAL and REPAIR")
    defaults = {
        stage: _parse_decision(defaults_value[stage.value], f"default.{stage.value}")
        for stage in Stage
    }
    rules_value = value["rules"]
    if not isinstance(rules_value, list):
        raise PolicyValidationError("rules must be a list")
    if len(rules_value) > MAX_POLICY_RULES:
        raise PolicyValidationError(
            f"rules cannot exceed {MAX_POLICY_RULES}"
        )
    rules: list[Rule] = []
    for index, raw in enumerate(rules_value):
        label = f"rule[{index}]"
        if not isinstance(raw, Mapping):
            raise PolicyValidationError(f"{label} must be an object")
        _strict_keys(
            raw,
            frozenset({"rule_id", "stage", "match", "when", "decision"}),
            label,
            required=frozenset(
                {"rule_id", "stage", "match", "when", "decision"}
            ),
        )
        rule_id = raw["rule_id"]
        if (
            not isinstance(rule_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", rule_id)
        ):
            raise PolicyValidationError(f"{label}.rule_id is invalid")
        try:
            stage = Stage(raw["stage"])
        except (ValueError, TypeError) as exc:
            raise PolicyValidationError(f"{label}.stage is invalid") from exc
        match = raw["match"]
        if not isinstance(match, str) or match not in {"all", "any"}:
            raise PolicyValidationError(f"{label}.match must be all or any")
        when = raw["when"]
        if not isinstance(when, list) or not when:
            raise PolicyValidationError(f"{label}.when must be a nonempty list")
        if len(when) > MAX_RULE_CONDITIONS:
            raise PolicyValidationError(
                f"{label}.when cannot exceed {MAX_RULE_CONDITIONS} conditions"
            )
        parsed_when = tuple(
            _parse_condition(condition, f"{label}.when[{condition_index}]")
            for condition_index, condition in enumerate(when)
        )
        unavailable = sorted(
            condition.feature
            for condition in parsed_when
            if condition.feature not in STAGE_FEATURES[stage]
        )
        if unavailable:
            raise PolicyValidationError(
                f"{label} uses features unavailable at {stage.value}: "
                f"{unavailable}"
            )
        rules.append(
            Rule(
                rule_id=rule_id,
                stage=stage,
                match=match,
                when=parsed_when,
                decision=_parse_decision(raw["decision"], f"{label}.decision"),
            )
        )
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise PolicyValidationError("rule_id values must be unique")

    unsigned = dict(value)
    declared_hash = unsigned.pop("policy_sha256", None)
    observed_hash = canonical_hash(unsigned)
    if declared_hash is not None and declared_hash != observed_hash:
        raise PolicyValidationError("policy hash mismatch")
    return Policy(
        policy_id=policy_id,
        rules=tuple(rules),
        defaults=defaults,
        max_repairs=1,
        policy_sha256=observed_hash,
    )


def load_policy(path: str | Path) -> Policy:
    """Load a UTF-8 JSON policy without permissive coercion."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyValidationError(f"could not load policy JSON: {exc}") from exc
    return parse_policy(value)


def _compare(actual: bool | int | float | str, condition: Condition) -> bool:
    expected = condition.value
    comparator = condition.comparator
    if comparator == "eq":
        return actual == expected
    if comparator == "ne":
        return actual != expected
    if comparator == "lt":
        return actual < expected  # type: ignore[operator]
    if comparator == "lte":
        return actual <= expected  # type: ignore[operator]
    if comparator == "gt":
        return actual > expected  # type: ignore[operator]
    if comparator == "gte":
        return actual >= expected  # type: ignore[operator]
    if comparator == "in":
        return actual in expected  # type: ignore[operator]
    if comparator == "not_in":
        return actual not in expected  # type: ignore[operator]
    if comparator == "contains":
        return expected in actual  # type: ignore[operator]
    raise AssertionError(comparator)


def decide(
    policy: Policy,
    task: Mapping[str, Any],
    *,
    stage: Stage | str,
    draft: str | Mapping[str, Any] | None = None,
    prior_operations: Sequence[Operation | str] = (),
) -> PolicyDecision:
    """Evaluate a policy and emit a hash-chained, outcome-free trace."""
    parsed_stage = Stage(stage)
    parsed_prior = tuple(Operation(value) for value in prior_operations)
    features = extract_features(
        task,
        stage=parsed_stage,
        draft=draft,
        prior_operations=parsed_prior,
    )
    evaluations: list[dict[str, Any]] = []
    selected_rule: Rule | None = None
    for rule in policy.rules:
        if rule.stage is not parsed_stage:
            continue
        predicates = [
            {
                "feature": condition.feature,
                "comparator": condition.comparator,
                "expected": list(condition.value)
                if isinstance(condition.value, tuple)
                else condition.value,
                "actual": features[condition.feature],
                "matched": _compare(features[condition.feature], condition),
            }
            for condition in rule.when
        ]
        matched = (
            all(row["matched"] for row in predicates)
            if rule.match == "all"
            else any(row["matched"] for row in predicates)
        )
        evaluations.append(
            {
                "rule_id": rule.rule_id,
                "match": rule.match,
                "predicates": predicates,
                "matched": matched,
            }
        )
        if matched:
            selected_rule = rule
            break
    chosen = selected_rule.decision if selected_rule else policy.defaults[parsed_stage]
    duplicate = (
        parsed_stage is Stage.REPAIR
        and chosen.operation is not Operation.KEEP
        and chosen.operation in parsed_prior
    )
    effective = Operation.KEEP if duplicate else chosen.operation
    trace: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_sha256,
        "task_id": task["id"],
        "task_input_sha256": canonical_hash(
            {key: task[key] for key in sorted(_TASK_FIELDS)}
        ),
        "stage": parsed_stage.value,
        "features": features,
        "feature_sha256": canonical_hash(features),
        "prior_operations": [operation.value for operation in parsed_prior],
        "rule_evaluations": evaluations,
        "matched_rule_id": selected_rule.rule_id if selected_rule else None,
        "used_default": selected_rule is None,
        "proposed_operation": chosen.operation.value,
        "effective_operation": effective.value,
        "duplicate_operation_suppressed": duplicate,
        "reason": (
            f"duplicate {chosen.operation.value} suppressed after initial call"
            if duplicate
            else chosen.reason
        ),
        "uses_gold_or_evaluator_signals": False,
    }
    trace["trace_sha256"] = canonical_hash(trace)
    return PolicyDecision(
        task_id=task["id"],
        stage=parsed_stage,
        operation=effective,
        proposed_operation=chosen.operation,
        reason=str(trace["reason"]),
        matched_rule_id=selected_rule.rule_id if selected_rule else None,
        duplicate_suppressed=duplicate,
        trace=trace,
    )


class PolicySession:
    """Stateful guard that permits one INITIAL decision and one REPAIR decision."""

    def __init__(self, policy: Policy, task: Mapping[str, Any]) -> None:
        self.policy = policy
        self.task = dict(task)
        self._operations: list[Operation] = []
        self._initial: PolicyDecision | None = None
        self._repair: PolicyDecision | None = None

    @property
    def decisions(self) -> tuple[PolicyDecision, ...]:
        return tuple(
            decision
            for decision in (self._initial, self._repair)
            if decision is not None
        )

    def decide_initial(self) -> PolicyDecision:
        if self._initial is not None:
            raise RuntimeError("INITIAL decision already made")
        decision = decide(self.policy, self.task, stage=Stage.INITIAL)
        self._initial = decision
        self._operations.append(decision.operation)
        return decision

    def decide_repair(
        self, draft: str | Mapping[str, Any]
    ) -> PolicyDecision:
        if self._initial is None:
            raise RuntimeError("INITIAL decision is required before REPAIR")
        if self._repair is not None:
            raise RuntimeError("repair budget exhausted")
        decision = decide(
            self.policy,
            self.task,
            stage=Stage.REPAIR,
            draft=draft,
            prior_operations=(self._initial.operation,),
        )
        self._repair = decision
        self._operations.append(decision.operation)
        return decision


def policy_to_dict(policy: Policy, *, include_hash: bool = True) -> dict[str, Any]:
    """Return the canonical external representation of a parsed policy."""
    value: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "max_repairs": policy.max_repairs,
        "defaults": {
            stage.value: {
                "operation": policy.defaults[stage].operation.value,
                "reason": policy.defaults[stage].reason,
            }
            for stage in Stage
        },
        "rules": [
            {
                "rule_id": rule.rule_id,
                "stage": rule.stage.value,
                "match": rule.match,
                "when": [
                    {
                        "feature": condition.feature,
                        "comparator": condition.comparator,
                        "value": list(condition.value)
                        if isinstance(condition.value, tuple)
                        else condition.value,
                    }
                    for condition in rule.when
                ],
                "decision": {
                    "operation": rule.decision.operation.value,
                    "reason": rule.decision.reason,
                },
            }
            for rule in policy.rules
        ],
    }
    if include_hash:
        value["policy_sha256"] = policy.policy_sha256
    return value


__all__ = [
    "FEATURE_TYPES",
    "MAX_POLICY_RULES",
    "MAX_RULE_CONDITIONS",
    "POLICY_SCHEMA_VERSION",
    "STAGE_FEATURES",
    "TRACE_SCHEMA_VERSION",
    "Condition",
    "Decision",
    "Operation",
    "Policy",
    "PolicyDecision",
    "PolicySession",
    "PolicyValidationError",
    "Rule",
    "Stage",
    "canonical_hash",
    "canonical_json",
    "decide",
    "extract_features",
    "load_policy",
    "parse_policy",
    "policy_to_dict",
    "reject_forbidden_fields",
]
