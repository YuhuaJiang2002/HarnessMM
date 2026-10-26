"""Outcome-blind, first-admissible response routing for ALFWorld.

The adapter deliberately has no ALFWorld, model, Torch, or CUDA dependency.
Callers provide frozen response layers and the public state available before an
environment step.  Only a selected environment-provided command may be passed
to ``env_step``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


_CANONICAL_ACTION = re.compile(r"<action>(.*?)</action>", flags=re.DOTALL)
_ACTION_LIKE_TAG = re.compile(r"<\s*/?\s*action\b[^>]*>", flags=re.IGNORECASE)
_CONTEXT_FIELDS = frozenset(
    {"step_index", "current_observation", "history", "task_type", "admissible_actions"}
)
_HISTORY_FIELDS = frozenset({"observation", "action"})
_PROHIBITED_CONTEXT_FIELDS = frozenset(
    {
        "reward",
        "rewards",
        "hard",
        "won",
        "success",
        "outcome",
        "outcomes",
        "future_observation",
        "future_observations",
        "next_observation",
        "gamefile",
        "game_file",
        "gamefile_identity",
        "done",
        "terminal",
    }
)
PRE_ENV_STEP_PHASE = "pre_env_step"


def normalize_action(action: str) -> str:
    """Normalize superficial case and whitespace without changing semantics."""

    if not isinstance(action, str):
        raise TypeError("action must be a string")
    return " ".join(action.split()).casefold()


def _validate_mapping_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    where: str,
) -> None:
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{where} field names must be strings")
    prohibited = keys.intersection(_PROHIBITED_CONTEXT_FIELDS)
    if prohibited:
        raise ValueError(f"{where} contains prohibited outcome fields: {sorted(prohibited)}")
    unknown = keys.difference(allowed)
    if unknown:
        raise ValueError(f"{where} contains unsupported fields: {sorted(unknown)}")


@dataclass(frozen=True)
class HistoryEntry:
    observation: str
    action: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, str) or not isinstance(self.action, str):
            raise TypeError("history observation and action must be strings")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HistoryEntry":
        _validate_mapping_fields(value, allowed=_HISTORY_FIELDS, where="history entry")
        if set(value) != _HISTORY_FIELDS:
            raise ValueError("history entry requires observation and action")
        return cls(observation=value["observation"], action=value["action"])


@dataclass(frozen=True)
class RouterContext:
    """Deeply immutable, pre-outcome inputs exposed to response layers."""

    step_index: int
    current_observation: str
    history: tuple[HistoryEntry, ...]
    task_type: str
    admissible_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not isinstance(self.current_observation, str):
            raise TypeError("current_observation must be a string")
        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ValueError("task_type must be a non-empty public label")
        if not isinstance(self.history, tuple) or any(
            not isinstance(entry, HistoryEntry) for entry in self.history
        ):
            raise TypeError("history must be a tuple of HistoryEntry values")
        if not isinstance(self.admissible_actions, tuple) or not self.admissible_actions:
            raise ValueError("admissible_actions must be a non-empty tuple")
        if any(not isinstance(action, str) or not normalize_action(action) for action in self.admissible_actions):
            raise ValueError("admissible actions must be non-empty strings")
        normalized = tuple(normalize_action(action) for action in self.admissible_actions)
        if len(set(normalized)) != len(normalized):
            raise ValueError("admissible actions must be unique after normalization")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RouterContext":
        """Build an allowlisted context, explicitly rejecting outcome leakage."""

        if not isinstance(value, Mapping):
            raise TypeError("router context must be a mapping")
        _validate_mapping_fields(value, allowed=_CONTEXT_FIELDS, where="router context")
        if set(value) != _CONTEXT_FIELDS:
            missing = sorted(_CONTEXT_FIELDS.difference(value))
            raise ValueError(f"router context is missing required fields: {missing}")

        raw_history = value["history"]
        if isinstance(raw_history, (str, bytes)) or not isinstance(raw_history, Sequence):
            raise TypeError("history must be a sequence")
        history = tuple(
            entry
            if isinstance(entry, HistoryEntry)
            else HistoryEntry.from_mapping(entry)
            if isinstance(entry, Mapping)
            else _raise_history_type()
            for entry in raw_history
        )
        raw_actions = value["admissible_actions"]
        if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
            raise TypeError("admissible_actions must be a sequence")
        return cls(
            step_index=value["step_index"],
            current_observation=value["current_observation"],
            history=history,
            task_type=value["task_type"],
            admissible_actions=tuple(raw_actions),
        )

    def fingerprint(self) -> str:
        payload = {
            "step_index": self.step_index,
            "current_observation": self.current_observation,
            "history": [asdict(entry) for entry in self.history],
            "task_type": self.task_type,
            "admissible_actions": self.admissible_actions,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _raise_history_type() -> HistoryEntry:
    raise TypeError("history entries must be HistoryEntry values or mappings")


@dataclass(frozen=True)
class ParsedAction:
    action: str | None
    normalized_action: str | None
    canonical_action_count: int
    valid: bool
    reason: str


def parse_canonical_action(response: str) -> ParsedAction:
    """Parse exactly one lower-case canonical ``<action>`` pair."""

    if not isinstance(response, str):
        return ParsedAction(None, None, 0, False, "response_not_string")

    opening_count = response.count("<action>")
    closing_count = response.count("</action>")
    canonical_tag_count = opening_count + closing_count
    action_like_tags = _ACTION_LIKE_TAG.findall(response)
    matches = _CANONICAL_ACTION.findall(response)

    if len(action_like_tags) != canonical_tag_count:
        return ParsedAction(None, None, len(matches), False, "noncanonical_action_syntax")
    if max(opening_count, closing_count) > 1:
        return ParsedAction(None, None, len(matches), False, "multiple_canonical_actions")
    if opening_count != closing_count:
        return ParsedAction(None, None, len(matches), False, "malformed_action_syntax")
    if not matches:
        return ParsedAction(None, None, 0, False, "missing_canonical_action")
    if len(matches) != 1:
        return ParsedAction(None, None, len(matches), False, "multiple_canonical_actions")

    action = matches[0]
    normalized = normalize_action(action)
    if not normalized:
        return ParsedAction(action, normalized, 1, False, "empty_canonical_action")
    if "<" in action or ">" in action:
        return ParsedAction(action, normalized, 1, False, "invalid_action_content")
    return ParsedAction(action, normalized, 1, True, "exactly_one_canonical_action")


@dataclass(frozen=True)
class RouterLayer:
    layer_id: str
    response_provider: Callable[[RouterContext], str]

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise ValueError("layer_id must be non-empty")
        if not callable(self.response_provider):
            raise TypeError("response_provider must be callable")


@dataclass(frozen=True)
class FrozenFallback:
    layer_id: str
    action: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise ValueError("fallback layer_id must be non-empty")
        if not isinstance(self.action, str) or not normalize_action(self.action):
            raise ValueError("fallback action must be a non-empty string")


@dataclass(frozen=True)
class LayerAttempt:
    layer_id: str
    source: str
    response: str | None
    canonical_action_count: int
    parsed_action: str | None
    normalized_action: str | None
    parse_valid: bool
    parse_reason: str
    admissible: bool
    admissibility_reason: str
    accepted: bool
    error: str | None = None


@dataclass(frozen=True)
class RouteRecord:
    step_index: int
    task_type: str
    context_sha256: str
    frozen_layer_order: tuple[str, ...]
    attempts: tuple[LayerAttempt, ...]
    selected_layer: str | None
    selected_action: str | None
    selected_normalized_action: str | None
    selection_reason: str
    used_frozen_fallback: bool
    executable: bool
    decision_phase: str = PRE_ENV_STEP_PHASE
    decision_precedes_env_step: bool = True
    outcome_fields_accessed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutedEnvironmentStep:
    route: RouteRecord
    environment_result: Any


@dataclass(frozen=True)
class FirstAdmissibleNestedRouter:
    """Evaluate immutable response layers in order, then fail closed."""

    layers: tuple[RouterLayer, ...]
    fallback: FrozenFallback
    propagate_layer_errors: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.layers, tuple) or not self.layers:
            raise ValueError("layers must be a non-empty tuple")
        if any(not isinstance(layer, RouterLayer) for layer in self.layers):
            raise TypeError("layers must contain RouterLayer values")
        ids = tuple(layer.layer_id for layer in self.layers)
        if len(set(ids)) != len(ids):
            raise ValueError("layer ids must be unique")
        if self.fallback.layer_id in ids:
            raise ValueError("fallback layer_id must be distinct from response layers")
        if not isinstance(self.propagate_layer_errors, bool):
            raise TypeError("propagate_layer_errors must be boolean")

    def decide(self, context: RouterContext | Mapping[str, Any]) -> RouteRecord:
        if isinstance(context, Mapping):
            context = RouterContext.from_mapping(context)
        if not isinstance(context, RouterContext):
            raise TypeError("context must be RouterContext or an allowlisted mapping")

        admissible_by_normalized = {
            normalize_action(action): action for action in context.admissible_actions
        }
        attempts: list[LayerAttempt] = []
        frozen_order = tuple(layer.layer_id for layer in self.layers) + (
            self.fallback.layer_id,
        )

        for layer in self.layers:
            try:
                response = layer.response_provider(context)
            except Exception as exc:
                if self.propagate_layer_errors:
                    raise
                attempts.append(
                    LayerAttempt(
                        layer_id=layer.layer_id,
                        source="response_layer",
                        response=None,
                        canonical_action_count=0,
                        parsed_action=None,
                        normalized_action=None,
                        parse_valid=False,
                        parse_reason="layer_error",
                        admissible=False,
                        admissibility_reason="not_checked_layer_error",
                        accepted=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            parsed = parse_canonical_action(response)
            admissible = bool(
                parsed.valid and parsed.normalized_action in admissible_by_normalized
            )
            attempt = LayerAttempt(
                layer_id=layer.layer_id,
                source="response_layer",
                response=response if isinstance(response, str) else None,
                canonical_action_count=parsed.canonical_action_count,
                parsed_action=parsed.action,
                normalized_action=parsed.normalized_action,
                parse_valid=parsed.valid,
                parse_reason=parsed.reason,
                admissible=admissible,
                admissibility_reason=(
                    "action_in_current_admissible_set"
                    if admissible
                    else (
                        "action_not_in_current_admissible_set"
                        if parsed.valid
                        else "not_checked_parse_invalid"
                    )
                ),
                accepted=admissible,
            )
            attempts.append(attempt)
            if admissible:
                selected = admissible_by_normalized[parsed.normalized_action]
                return self._record(
                    context=context,
                    frozen_order=frozen_order,
                    attempts=attempts,
                    selected_layer=layer.layer_id,
                    selected_action=selected,
                    selection_reason="first_admissible_layer_response",
                    used_fallback=False,
                )

        fallback_normalized = normalize_action(self.fallback.action)
        fallback_admissible = fallback_normalized in admissible_by_normalized
        fallback_attempt = LayerAttempt(
            layer_id=self.fallback.layer_id,
            source="frozen_fallback",
            response=None,
            canonical_action_count=0,
            parsed_action=self.fallback.action,
            normalized_action=fallback_normalized,
            parse_valid=True,
            parse_reason="explicit_frozen_fallback",
            admissible=fallback_admissible,
            admissibility_reason=(
                "action_in_current_admissible_set"
                if fallback_admissible
                else "fallback_not_in_current_admissible_set"
            ),
            accepted=fallback_admissible,
        )
        attempts.append(fallback_attempt)
        if fallback_admissible:
            return self._record(
                context=context,
                frozen_order=frozen_order,
                attempts=attempts,
                selected_layer=self.fallback.layer_id,
                selected_action=admissible_by_normalized[fallback_normalized],
                selection_reason="all_response_layers_invalid_frozen_fallback",
                used_fallback=True,
            )
        return self._record(
            context=context,
            frozen_order=frozen_order,
            attempts=attempts,
            selected_layer=None,
            selected_action=None,
            selection_reason="frozen_fallback_not_currently_admissible_refuse_to_step",
            used_fallback=True,
        )

    def _record(
        self,
        *,
        context: RouterContext,
        frozen_order: tuple[str, ...],
        attempts: Sequence[LayerAttempt],
        selected_layer: str | None,
        selected_action: str | None,
        selection_reason: str,
        used_fallback: bool,
    ) -> RouteRecord:
        return RouteRecord(
            step_index=context.step_index,
            task_type=context.task_type,
            context_sha256=context.fingerprint(),
            frozen_layer_order=frozen_order,
            attempts=tuple(attempts),
            selected_layer=selected_layer,
            selected_action=selected_action,
            selected_normalized_action=(
                normalize_action(selected_action) if selected_action is not None else None
            ),
            selection_reason=selection_reason,
            used_frozen_fallback=used_fallback,
            executable=selected_action is not None,
        )

    def route_and_step(
        self,
        context: RouterContext | Mapping[str, Any],
        env_step: Callable[[str], Any],
    ) -> RoutedEnvironmentStep:
        """Decide first and only then invoke the supplied environment step."""

        resolved_context = (
            RouterContext.from_mapping(context) if isinstance(context, Mapping) else context
        )
        route = self.decide(resolved_context)
        assert route.decision_phase == PRE_ENV_STEP_PHASE
        assert route.decision_precedes_env_step
        assert route.step_index == resolved_context.step_index
        if not route.executable or route.selected_action is None:
            raise RuntimeError("router failed closed; environment step was not called")
        result = env_step(route.selected_action)
        return RoutedEnvironmentStep(route=route, environment_result=result)


__all__ = [
    "FirstAdmissibleNestedRouter",
    "FrozenFallback",
    "HistoryEntry",
    "LayerAttempt",
    "PRE_ENV_STEP_PHASE",
    "ParsedAction",
    "RouteRecord",
    "RoutedEnvironmentStep",
    "RouterContext",
    "RouterLayer",
    "normalize_action",
    "parse_canonical_action",
]
