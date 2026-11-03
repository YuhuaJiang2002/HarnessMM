"""Outcome-blind, stage-aware ADD/DELETE/MODIFY harness controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


POLICY_SCHEMA_VERSION = "phiagent-stage-aware-adm-policy/v1"
TRACE_SCHEMA_VERSION = "phiagent-stage-aware-adm-trace/v1"
_DYNAMIC_MESSAGE_PREFIX = "<stage_aware_adm"
_OPTIONAL_ROUTE_FIELDS = frozenset(
    {
        "task_family_initial_operations",
        "task_signal_initial_operations",
        "task_family_route_stages",
        "task_signal_route_stages",
    }
)
_FORBIDDEN_FIELD_TOKENS = frozenset(
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
        "verdict",
    }
)


class PolicyValidationError(ValueError):
    """Raised when a dynamic ADM policy or input violates its contract."""


class Stage(str, Enum):
    INITIAL = "INITIAL"
    EXECUTE = "EXECUTE"
    RECOVER = "RECOVER"
    VERIFY = "VERIFY"


class Operation(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"


@dataclass(frozen=True)
class Policy:
    policy_id: str
    mode: str
    random_seed: int
    max_decisions: int
    fallback_by_stage: Mapping[Stage, Operation]
    directives: Mapping[Stage, Mapping[Operation, str]]
    task_family_initial_operations: Mapping[str, Operation]
    task_signal_initial_operations: Mapping[str, Operation]
    task_family_route_stages: tuple[Stage, ...]
    task_signal_route_stages: tuple[Stage, ...]
    delete_tool_message_characters: int
    delete_preserve_recent_tool_messages: int
    delete_history_actions: int
    serialized_optional_fields: frozenset[str]
    policy_sha256: str


@dataclass(frozen=True)
class Decision:
    task_id: str
    stage: Stage
    proposed_operation: Operation | None
    operation: Operation
    source: str
    directive: str
    compact_history: bool
    active_directives: tuple[str, ...]
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


def _field_tokens(name: object) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(name).casefold())
        if token
    }


def reject_outcome_fields(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            forbidden = sorted(_field_tokens(key) & _FORBIDDEN_FIELD_TOKENS)
            if forbidden:
                raise PolicyValidationError(
                    f"forbidden outcome field at {path}.{key}: {forbidden}"
                )
            reject_outcome_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_outcome_fields(nested, path=f"{path}[{index}]")


def _strict_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise PolicyValidationError(
            f"{label} fields differ: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )


def _parse_stage_operations(
    value: object,
    *,
    label: str,
) -> dict[Stage, Operation]:
    if not isinstance(value, Mapping) or set(value) != {stage.value for stage in Stage}:
        raise PolicyValidationError(f"{label} must define every stage exactly once")
    try:
        return {stage: Operation(value[stage.value]) for stage in Stage}
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(f"{label} contains an invalid operation") from error


def _parse_directives(
    value: object,
) -> dict[Stage, dict[Operation, str]]:
    if not isinstance(value, Mapping) or set(value) != {stage.value for stage in Stage}:
        raise PolicyValidationError("directives must define every stage exactly once")
    parsed: dict[Stage, dict[Operation, str]] = {}
    for stage in Stage:
        stage_value = value[stage.value]
        if not isinstance(stage_value, Mapping) or set(stage_value) != {
            operation.value for operation in Operation
        }:
            raise PolicyValidationError(
                f"directives.{stage.value} must define ADD, DELETE, and MODIFY"
            )
        parsed[stage] = {}
        for operation in Operation:
            directive = stage_value[operation.value]
            if (
                not isinstance(directive, str)
                or not directive.strip()
                or len(directive) > 600
            ):
                raise PolicyValidationError(
                    f"directives.{stage.value}.{operation.value} "
                    "must contain 1..600 characters"
                )
            parsed[stage][operation] = directive.strip()
    return parsed


def parse_policy(value: Mapping[str, Any]) -> Policy:
    """Parse a frozen policy and verify its optional canonical self-hash."""

    if not isinstance(value, Mapping):
        raise PolicyValidationError("policy must be an object")
    reject_outcome_fields(value, path="$.policy")
    _strict_keys(
        value,
        frozenset(
            {
                "schema_version",
                "policy_id",
                "mode",
                "random_seed",
                "max_decisions",
                "fallback_by_stage",
                "directives",
                "task_family_initial_operations",
                "task_signal_initial_operations",
                "task_family_route_stages",
                "task_signal_route_stages",
                "delete_policy",
                "policy_sha256",
            }
        ),
        frozenset(
            {
                "schema_version",
                "policy_id",
                "mode",
                "random_seed",
                "max_decisions",
                "fallback_by_stage",
                "directives",
                "delete_policy",
            }
        ),
        "policy",
    )
    if value["schema_version"] != POLICY_SCHEMA_VERSION:
        raise PolicyValidationError("unsupported policy schema_version")
    policy_id = value["policy_id"]
    if (
        not isinstance(policy_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", policy_id) is None
    ):
        raise PolicyValidationError("policy_id is invalid")
    mode = value["mode"]
    if mode not in {"model", "compiled", "fixed", "random", "residual"}:
        raise PolicyValidationError(
            "mode must be model, compiled, fixed, random, or residual"
        )
    if type(value["random_seed"]) is not int:
        raise PolicyValidationError("random_seed must be an integer")
    if (
        type(value["max_decisions"]) is not int
        or not 1 <= value["max_decisions"] <= 64
    ):
        raise PolicyValidationError("max_decisions must be in 1..64")
    fallback = _parse_stage_operations(
        value["fallback_by_stage"],
        label="fallback_by_stage",
    )
    directives = _parse_directives(value["directives"])
    task_family_value = value.get("task_family_initial_operations", {})
    if not isinstance(task_family_value, Mapping):
        raise PolicyValidationError(
            "task_family_initial_operations must be an object"
        )
    task_family_initial_operations = {}
    for family, operation in task_family_value.items():
        if (
            not isinstance(family, str)
            or not family
            or re.fullmatch(r"[a-z0-9_-]+", family) is None
        ):
            raise PolicyValidationError("task-family route key is invalid")
        try:
            task_family_initial_operations[family] = Operation(operation)
        except (TypeError, ValueError) as error:
            raise PolicyValidationError(
                f"task-family route {family!r} has an invalid operation"
            ) from error
    task_signal_value = value.get("task_signal_initial_operations", {})
    if not isinstance(task_signal_value, Mapping):
        raise PolicyValidationError(
            "task_signal_initial_operations must be an object"
        )
    task_signal_initial_operations = {}
    for signal, operation in task_signal_value.items():
        if (
            not isinstance(signal, str)
            or not signal
            or re.fullmatch(r"[a-z0-9_]+", signal) is None
        ):
            raise PolicyValidationError("task-signal route key is invalid")
        try:
            task_signal_initial_operations[signal] = Operation(operation)
        except (TypeError, ValueError) as error:
            raise PolicyValidationError(
                f"task-signal route {signal!r} has an invalid operation"
            ) from error
    route_stages_value = value.get("task_family_route_stages", ["INITIAL"])
    if (
        not isinstance(route_stages_value, list)
        or not route_stages_value
        or len(route_stages_value) != len(set(route_stages_value))
    ):
        raise PolicyValidationError(
            "task_family_route_stages must be a nonempty unique list"
        )
    try:
        task_family_route_stages = tuple(
            Stage(stage) for stage in route_stages_value
        )
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(
            "task_family_route_stages contains an invalid stage"
        ) from error
    signal_stages_value = value.get("task_signal_route_stages", ["INITIAL"])
    if (
        not isinstance(signal_stages_value, list)
        or not signal_stages_value
        or len(signal_stages_value) != len(set(signal_stages_value))
    ):
        raise PolicyValidationError(
            "task_signal_route_stages must be a nonempty unique list"
        )
    try:
        task_signal_route_stages = tuple(
            Stage(stage) for stage in signal_stages_value
        )
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(
            "task_signal_route_stages contains an invalid stage"
        ) from error
    delete = value["delete_policy"]
    if not isinstance(delete, Mapping):
        raise PolicyValidationError("delete_policy must be an object")
    _strict_keys(
        delete,
        frozenset(
            {
                "tool_message_characters",
                "preserve_recent_tool_messages",
                "history_actions",
            }
        ),
        frozenset(
            {
                "tool_message_characters",
                "preserve_recent_tool_messages",
                "history_actions",
            }
        ),
        "delete_policy",
    )
    for key, lower, upper in (
        ("tool_message_characters", 128, 8000),
        ("preserve_recent_tool_messages", 0, 8),
        ("history_actions", 1, 8),
    ):
        item = delete[key]
        if type(item) is not int or not lower <= item <= upper:
            raise PolicyValidationError(
                f"delete_policy.{key} must be an integer in {lower}..{upper}"
            )
    unsigned = dict(value)
    declared = unsigned.pop("policy_sha256", None)
    observed = canonical_hash(unsigned)
    if declared is not None and declared != observed:
        raise PolicyValidationError("policy_sha256 mismatch")
    return Policy(
        policy_id=policy_id,
        mode=str(mode),
        random_seed=int(value["random_seed"]),
        max_decisions=int(value["max_decisions"]),
        fallback_by_stage=fallback,
        directives=directives,
        task_family_initial_operations=task_family_initial_operations,
        task_signal_initial_operations=task_signal_initial_operations,
        task_family_route_stages=task_family_route_stages,
        task_signal_route_stages=task_signal_route_stages,
        delete_tool_message_characters=int(delete["tool_message_characters"]),
        delete_preserve_recent_tool_messages=int(
            delete["preserve_recent_tool_messages"]
        ),
        delete_history_actions=int(delete["history_actions"]),
        serialized_optional_fields=frozenset(set(value) & _OPTIONAL_ROUTE_FIELDS),
        policy_sha256=observed,
    )


def policy_to_dict(policy: Policy, *, include_hash: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "mode": policy.mode,
        "random_seed": policy.random_seed,
        "max_decisions": policy.max_decisions,
        "fallback_by_stage": {
            stage.value: policy.fallback_by_stage[stage].value for stage in Stage
        },
        "directives": {
            stage.value: {
                operation.value: policy.directives[stage][operation]
                for operation in Operation
            }
            for stage in Stage
        },
        "delete_policy": {
            "tool_message_characters": policy.delete_tool_message_characters,
            "preserve_recent_tool_messages": (
                policy.delete_preserve_recent_tool_messages
            ),
            "history_actions": policy.delete_history_actions,
        },
    }
    if "task_family_initial_operations" in policy.serialized_optional_fields:
        value["task_family_initial_operations"] = {
            family: operation.value
            for family, operation in sorted(
                policy.task_family_initial_operations.items()
            )
        }
    if "task_signal_initial_operations" in policy.serialized_optional_fields:
        value["task_signal_initial_operations"] = {
            signal: operation.value
            for signal, operation in sorted(
                policy.task_signal_initial_operations.items()
            )
        }
    if "task_family_route_stages" in policy.serialized_optional_fields:
        value["task_family_route_stages"] = [
            stage.value for stage in policy.task_family_route_stages
        ]
    if "task_signal_route_stages" in policy.serialized_optional_fields:
        value["task_signal_route_stages"] = [
            stage.value for stage in policy.task_signal_route_stages
        ]
    if include_hash:
        value["policy_sha256"] = policy.policy_sha256
    return value


def default_policy_value(domain: str, *, mode: str) -> dict[str, Any]:
    """Build the frozen seed policy for a Meta-Harness-style online ADM arm."""

    if domain == "terminal":
        directives = {
            Stage.INITIAL: {
                Operation.ADD: (
                    "Add a concise completion contract: enumerate required artifacts "
                    "and the strongest available local check before editing."
                ),
                Operation.DELETE: (
                    "PARENT: delete optional harness clauses and use the unmodified "
                    "base prompt for initial inspection."
                ),
                Operation.MODIFY: (
                    "PARENT: retain the exact-command-guard parent until live "
                    "execution evidence requires a change."
                ),
            },
            Stage.EXECUTE: {
                Operation.ADD: (
                    "Add one missing requirement or diagnostic supported by the latest "
                    "tool output before the next edit."
                ),
                Operation.DELETE: (
                    "Delete stale inspections and repeated probes; retain the latest "
                    "state and execute the next unmet step."
                ),
                Operation.MODIFY: (
                    "PARENT: retain the exact-command-guard parent while execution "
                    "continues without a concrete failure."
                ),
            },
            Stage.RECOVER: {
                Operation.ADD: (
                    "Add one bounded diagnostic that isolates the first concrete "
                    "failure before another repair."
                ),
                Operation.DELETE: (
                    "Delete the failed approach and stale context; do not retry an "
                    "unchanged command or installation."
                ),
                Operation.MODIFY: (
                    "Modify the command or artifact from the latest exit code and "
                    "stderr, repairing one diagnosed failure at a time. Re-read the "
                    "surrounding declaration before deleting or replacing definitions."
                ),
            },
            Stage.VERIFY: {
                Operation.ADD: (
                    "Add the strongest available local parser, compiler, executable, "
                    "or representative-input check before finishing."
                ),
                Operation.DELETE: (
                    "Delete further exploration once every requested artifact exists "
                    "and the strongest available check passes."
                ),
                Operation.MODIFY: (
                    "Modify only the diagnosed failing part, rerun the strongest local "
                    "check, and preserve all verified work."
                ),
            },
        }
        fallback = {
            Stage.INITIAL: (
                Operation.MODIFY
                if mode in {"compiled", "residual"}
                else Operation.ADD
            ),
            Stage.EXECUTE: (
                Operation.MODIFY
                if mode in {"compiled", "residual"}
                else Operation.DELETE
            ),
            Stage.RECOVER: Operation.MODIFY,
            Stage.VERIFY: Operation.ADD,
        }
        delete = {
            "tool_message_characters": 8000,
            "preserve_recent_tool_messages": 8,
            "history_actions": 8,
        }
        task_family_initial_operations = {
            "debugging": Operation.DELETE,
            "security": Operation.MODIFY,
        }
        task_signal_initial_operations = {
            "repository_history_task": Operation.ADD,
        }
        task_family_route_stages = (Stage.INITIAL,)
        task_signal_route_stages = (Stage.INITIAL,)
        max_decisions = 64
    elif domain == "webarena":
        directives = {
            Stage.INITIAL: {
                Operation.ADD: (
                    "Add a compact checklist of the requested entity, fields, and "
                    "visible completion evidence before navigating."
                ),
                Operation.DELETE: (
                    "Delete irrelevant navigation assumptions and act only on the "
                    "current page's visible controls."
                ),
                Operation.MODIFY: (
                    "Modify the initial route to match the task's information, "
                    "navigation, or state-change family. For a ranking, count, total, "
                    "or average, prefer the relevant report or sorted grid over an "
                    "exact-entity filter."
                ),
            },
            Stage.EXECUTE: {
                Operation.ADD: (
                    "Add one missing field or visible state check needed by the task."
                ),
                Operation.DELETE: (
                    "Delete stale page assumptions and repeated actions; keep only "
                    "the current visible state and unmet goal. If the requested answer "
                    "or state is visibly complete, exit instead of continuing."
                ),
                Operation.MODIFY: (
                    "Modify the route using the current URL and visible elements; "
                    "choose a different grounded action."
                ),
            },
            Stage.RECOVER: {
                Operation.ADD: (
                    "Add one bounded inspection of a new visible control or route."
                ),
                Operation.DELETE: (
                    "Delete the repeated action and stale page assumption before "
                    "choosing another grounded control."
                ),
                Operation.MODIFY: (
                    "Modify the action using the failure feedback. Never repeat the "
                    "same element/action pair: backtrack once, change the query, or "
                    "select another observed control. If all requested fields are "
                    "already filled, use the visible Save, Submit, or Confirm control."
                ),
            },
            Stage.VERIFY: {
                Operation.ADD: (
                    "Add one final visible-state check or confirmation action before "
                    "claiming completion."
                ),
                Operation.DELETE: (
                    "Delete further interaction when the requested answer or final "
                    "state is already visibly present, then exit."
                ),
                Operation.MODIFY: (
                    "Modify exactly one incomplete field or confirmation action before "
                    "exiting."
                ),
            },
        }
        fallback = {
            Stage.INITIAL: (
                Operation.MODIFY
                if mode in {"compiled", "residual"}
                else Operation.ADD
            ),
            Stage.EXECUTE: Operation.DELETE,
            Stage.RECOVER: Operation.DELETE,
            Stage.VERIFY: Operation.DELETE,
        }
        delete = {
            "tool_message_characters": 800,
            "preserve_recent_tool_messages": 2,
            "history_actions": 3,
        }
        task_family_initial_operations = {
            "information_retrieval": Operation.DELETE,
            "navigation": Operation.DELETE,
            "state_change": Operation.ADD,
        }
        task_signal_initial_operations = {
            "aggregate_request": Operation.MODIFY,
        }
        task_family_route_stages = (Stage.INITIAL,)
        task_signal_route_stages = (Stage.INITIAL,)
        max_decisions = 32
    else:
        raise PolicyValidationError(f"unsupported dynamic ADM domain: {domain}")
    value: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": f"{domain}-stage-aware-adm-{mode}-v1",
        "mode": mode,
        "random_seed": 20260908,
        "max_decisions": max_decisions,
        "fallback_by_stage": {
            stage.value: fallback[stage].value for stage in Stage
        },
        "directives": {
            stage.value: {
                operation.value: directives[stage][operation]
                for operation in Operation
            }
            for stage in Stage
        },
        "task_family_initial_operations": {
            family: operation.value
            for family, operation in task_family_initial_operations.items()
        },
        "task_signal_initial_operations": {
            signal: operation.value
            for signal, operation in task_signal_initial_operations.items()
        },
        "task_family_route_stages": [
            stage.value for stage in task_family_route_stages
        ],
        "task_signal_route_stages": [
            stage.value for stage in task_signal_route_stages
        ],
        "delete_policy": delete,
    }
    return policy_to_dict(parse_policy(value))


def infer_stage(
    *,
    turn_index: int,
    max_turns: int,
    action_count: int,
    recovery_signal: bool,
    completion_requested: bool = False,
    verification_reserve: int = 3,
) -> Stage:
    """Infer the current execution stage from outcome-blind runtime state."""

    if (
        turn_index < 0
        or max_turns < 1
        or action_count < 0
        or verification_reserve < 0
        or verification_reserve >= max_turns
    ):
        raise PolicyValidationError("invalid stage-inference counters")
    if completion_requested:
        return Stage.VERIFY
    if recovery_signal:
        return Stage.RECOVER
    if action_count == 0:
        return Stage.INITIAL
    if max_turns - turn_index <= verification_reserve:
        return Stage.VERIFY
    return Stage.EXECUTE


def parse_operation(value: object) -> Operation | None:
    if not isinstance(value, str):
        return None
    try:
        return Operation(value.strip().upper())
    except ValueError:
        return None


def parse_tagged_action(response: str) -> tuple[Operation | None, str]:
    """Parse an optional ``ADM: OP`` header and return the underlying action."""

    lines = [
        line.strip()
        for line in response.strip().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return None, ""
    match = re.fullmatch(r"ADM\s*:\s*(ADD|DELETE|MODIFY)", lines[0], re.IGNORECASE)
    if match is None:
        return None, "\n".join(lines)
    return Operation(match.group(1).upper()), "\n".join(lines[1:])


def operation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "stage_aware_adm_operation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["ADD", "DELETE", "MODIFY"],
                    }
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
        },
    }


def parse_operation_response(content: str) -> tuple[Operation | None, str | None]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, Mapping):
        return None, "controller response is not an object"
    operation = parse_operation(value.get("operation"))
    if operation is None:
        return None, "controller response has no valid operation"
    if set(value) != {"operation"}:
        return None, "controller response contains unexpected fields"
    return operation, None


def controller_messages(
    session: PolicySession,
    *,
    stage: Stage | str,
    task_text: str,
    trajectory_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the runtime controller of a Meta-Harness-style agent. "
                "Select one harness edit from live task evidence only."
            ),
        },
        {
            "role": "user",
            "content": (
                session.turn_instruction(stage)
                + "\nTask:\n"
                + task_text[:6000]
                + "\nMost recent trajectory evidence:\n"
                + trajectory_text[-4000:]
                + "\nReturn only the operation object."
            ),
        },
    ]


def _compact_tool_content(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[deleted stale tool context]...\n"
    remaining = limit - len(marker)
    return value[: remaining // 2] + marker + value[-(remaining - remaining // 2) :]


def compact_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_message_characters: int,
    preserve_recent_tool_messages: int,
) -> list[dict[str, Any]]:
    """Delete stale ADM directives and compact non-recent tool observations."""

    kept = [
        dict(message)
        for message in messages
        if not (
            message.get("role") == "user"
            and str(message.get("content", "")).startswith(_DYNAMIC_MESSAGE_PREFIX)
        )
    ]
    tool_indices = [
        index for index, message in enumerate(kept) if message.get("role") == "tool"
    ]
    preserved = (
        set(tool_indices[-preserve_recent_tool_messages:])
        if preserve_recent_tool_messages
        else set()
    )
    for index in tool_indices:
        if index not in preserved:
            kept[index]["content"] = _compact_tool_content(
                str(kept[index].get("content", "")),
                tool_message_characters,
            )
    return kept


class PolicySession:
    """Apply one frozen policy to a single task and emit hash-chained decisions."""

    def __init__(
        self,
        policy: Policy,
        *,
        task_id: str,
        task_context: Mapping[str, Any],
    ) -> None:
        if not task_id:
            raise PolicyValidationError("task_id must be non-empty")
        reject_outcome_fields(task_context, path="$.task_context")
        self.policy = policy
        self.task_id = task_id
        self.task_context = dict(task_context)
        self.task_context_sha256 = canonical_hash(task_context)
        self._active_directives: list[str] = []
        self._decisions: list[Decision] = []

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def active_directives(self) -> tuple[str, ...]:
        return tuple(self._active_directives)

    def choose(
        self,
        *,
        stage: Stage | str,
        proposed_operation: Operation | str | None,
        features: Mapping[str, Any],
    ) -> Decision:
        if len(self._decisions) >= self.policy.max_decisions:
            raise PolicyValidationError("dynamic ADM decision budget exhausted")
        reject_outcome_fields(features, path="$.features")
        parsed_stage = Stage(stage)
        proposed = parse_operation(proposed_operation)
        if self.policy.mode in {"model", "residual"} and proposed is not None:
            operation = proposed
            source = "model" if self.policy.mode == "model" else "residual"
        elif self.policy.mode == "compiled" and parsed_stage in (
            self.policy.task_signal_route_stages
        ) and any(
            self.task_context.get(signal) is True
            for signal in self.policy.task_signal_initial_operations
        ):
            signal = next(
                name
                for name in self.policy.task_signal_initial_operations
                if self.task_context.get(name) is True
            )
            operation = self.policy.task_signal_initial_operations[signal]
            source = "compiled_task_stage"
        elif (
            self.policy.mode == "compiled"
            and parsed_stage in self.policy.task_family_route_stages
            and str(self.task_context.get("family", ""))
            in self.policy.task_family_initial_operations
        ):
            operation = self.policy.task_family_initial_operations[
                str(self.task_context["family"])
            ]
            source = "compiled_task_stage"
        elif self.policy.mode == "compiled":
            operation = self.policy.fallback_by_stage[parsed_stage]
            source = "compiled_stage"
        elif self.policy.mode == "random":
            digest = hashlib.sha256(
                (
                    f"{self.policy.random_seed}\0{self.task_id}\0"
                    f"{parsed_stage.value}\0{len(self._decisions)}"
                ).encode()
            ).digest()
            operation = tuple(Operation)[int.from_bytes(digest[:8], "big") % 3]
            source = "matched_random"
        elif self.policy.mode == "residual":
            operation = self.policy.fallback_by_stage[parsed_stage]
            source = "residual_fallback"
        else:
            operation = self.policy.fallback_by_stage[parsed_stage]
            source = "fixed" if self.policy.mode == "fixed" else "model_fallback"

        directive = self.policy.directives[parsed_stage][operation]
        active_before = tuple(self._active_directives)
        compact_history = operation is Operation.DELETE
        if operation is Operation.ADD:
            if directive not in self._active_directives:
                self._active_directives.append(directive)
        elif operation is Operation.DELETE:
            if self._active_directives:
                self._active_directives.pop(0)
        else:
            if self._active_directives:
                self._active_directives[-1] = directive
            else:
                self._active_directives.append(directive)
        trace: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.policy_sha256,
            "task_id": self.task_id,
            "task_context_sha256": self.task_context_sha256,
            "decision_index": len(self._decisions),
            "stage": parsed_stage.value,
            "proposed_operation": proposed.value if proposed is not None else None,
            "effective_operation": operation.value,
            "decision_source": source,
            "features": dict(features),
            "feature_sha256": canonical_hash(features),
            "active_directives_before_sha256": canonical_hash(active_before),
            "active_directives_after_sha256": canonical_hash(
                tuple(self._active_directives)
            ),
            "compact_history": compact_history,
            "uses_gold_or_evaluator_signals": False,
        }
        trace["trace_sha256"] = canonical_hash(trace)
        decision = Decision(
            task_id=self.task_id,
            stage=parsed_stage,
            proposed_operation=proposed,
            operation=operation,
            source=source,
            directive=directive,
            compact_history=compact_history,
            active_directives=tuple(self._active_directives),
            trace=trace,
        )
        self._decisions.append(decision)
        return decision

    def turn_instruction(self, stage: Stage | str) -> str:
        parsed_stage = Stage(stage)
        active = (
            "\n".join(f"- {directive}" for directive in self._active_directives)
            or "- (none)"
        )
        choices = "\n".join(
            f"- {operation.value}: {self.policy.directives[parsed_stage][operation]}"
            for operation in Operation
        )
        return (
            f"{_DYNAMIC_MESSAGE_PREFIX} stage=\"{parsed_stage.value}\">\n"
            "Choose the harness edit that should govern this step: ADD appends a "
            "missing constraint, DELETE removes stale context, and MODIFY replaces "
            "the current plan after new evidence. Select exactly one of ADD, DELETE, "
            "or MODIFY using only the task and live trajectory; do not default to "
            "ADD. Never use hidden evaluation information.\n"
            f"Stage-specific choices:\n{choices}\n"
            f"Active directives:\n{active}\n"
            "</stage_aware_adm>"
        )

    def active_instruction(self, stage: Stage | str) -> str:
        parsed_stage = Stage(stage)
        if not self._decisions:
            return ""
        latest = self._decisions[-1]
        if latest.directive.startswith("PARENT:"):
            return ""
        active = (
            "\n".join(f"- {directive}" for directive in self._active_directives)
            or "- (none)"
        )
        return (
            f"{_DYNAMIC_MESSAGE_PREFIX} stage=\"{parsed_stage.value}\" "
            f"operation=\"{latest.operation.value}\">\n"
            f"Apply now: {latest.directive}\n"
            f"Persistent directives:\n{active}\n"
            "</stage_aware_adm>"
        )


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "Decision",
    "Operation",
    "Policy",
    "PolicySession",
    "PolicyValidationError",
    "Stage",
    "canonical_hash",
    "canonical_json",
    "compact_messages",
    "controller_messages",
    "default_policy_value",
    "infer_stage",
    "parse_operation",
    "parse_operation_response",
    "parse_policy",
    "parse_tagged_action",
    "operation_response_format",
    "policy_to_dict",
    "reject_outcome_fields",
]
