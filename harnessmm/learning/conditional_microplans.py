"""Actor-authored, one-continuation microplans; no model calls or action execution.

Typical lifecycle: decide -> actor -> parse_microplan -> install -> execute the
first action -> observe -> decide. A committed continuation requires one more
actual observe, then a new actor plan. install may also start from a fresh/empty
state without an earlier deferral. A second decide after a deferral requires a
new install; a decision never executes an action or advances an action counter.

Only reasoning/action/continuation are accepted in model JSON. Reasoning is
uninterpreted text, never guard evidence. Both plan modes require the same
well-formed guard with at least one positive predicate; unguarded_plan removes
only runtime predicate enforcement. Predicates use case-sensitive literal
substring matching after whitespace collapse, NOT regex, code or world-state
inference. Native actions are always exact strings and are never repaired.

The observation passed to a continuation decide must equal the actual feedback
recorded by observe, byte-for-byte as Python strings. Whitespace normalization
applies ONLY to predicate evaluation, not to substituting feedback or actions.
Guard/membership failures consume the plan; malformed calls raise ValueError
without consuming pending execution. Native-invalid actual feedback is logged,
consumes the plan, and adds no fictitious success. A new install cannot overwrite
an outstanding action or unevaluated continuation.

Parser/installation errors have no fallback here: the parent owns charged,
explicit actor-format fallback, native termination, actor prompts and execution.
No old trajectory-reuse controller, task rule, reward or evaluator is consulted.
Audits retain the last plan_sha256 after consumption until the next install;
phase/actor_required, not that provenance hash, indicate whether an action remains.
"""

from __future__ import annotations

import hashlib
import json
import math


STATE_SCHEMA = "chisel-conditional-microplan-state/v1"
MODES = ("structured_single", "guarded_plan", "unguarded_plan")
_PLAN_KEYS = {"reasoning", "action", "continuation"}
_CONTINUATION_KEYS = {"action", "observation_contains_all", "observation_contains_none"}


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _string(value: object, name: str, *, limit: int | None = None, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}string")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} contains an invalid Unicode surrogate") from error
    return value


def _actions(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("native actions must be a list of exact action strings")
    for action in value:
        _string(action, "native action")
    if len(set(value)) != len(value):
        raise ValueError("native actions must be unique")
    return value


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _mode(value: object) -> str:
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("Unknown microplan mode")
    return value


def response_schema(allow_continuation: bool) -> dict:
    """Return an owned standard JSON Schema; no provider-specific response wrapper."""
    _bool(allow_continuation, "allow_continuation")

    def string(limit: int) -> dict:
        return {"type": "string", "minLength": 1, "maxLength": limit, "pattern": r"\S"}

    continuation = {
        "type": "object", "additionalProperties": False,
        "description": "Check feedback from the first action BEFORE this continuation, never its result.",
        "required": ["action", "observation_contains_all", "observation_contains_none"],
        "properties": {
            "action": {
                **string(160),
                "description": "Complete next native command with all already bound entity arguments; no guessing.",
            },
            "observation_contains_all": {
                "type": "array", "minItems": 1, "maxItems": 4, "uniqueItems": True,
                "items": string(160),
                "description": "Literal fragments required in the intermediate observation returned by the first action.",
            },
            "observation_contains_none": {
                "type": "array", "minItems": 0, "maxItems": 4, "uniqueItems": True,
                "items": string(160),
                "description": "Literal fragments forbidden in that same intermediate observation, not future feedback.",
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": False,
        "description": "Execute action now; consider one conditional continuation after observing its feedback.",
        "required": ["reasoning", "action", "continuation"],
        "properties": {
            "reasoning": {**string(6000), "description": "Current decision rationale, not observation evidence."},
            "action": {
                **string(160),
                "description": "Execute NOW: exact complete current native command, including every bound argument.",
            },
            "continuation": {
                "anyOf": [{"type": "null"}, continuation],
                "description": "One conditional next action when an intermediate observation justifies it; otherwise null.",
            } if allow_continuation else {
                "type": "null", "description": "No continuation; execute only the current action.",
            },
        },
    }


def _validated_plan(plan: object, native_actions: list[str], allow_continuation: bool) -> dict:
    _bool(allow_continuation, "allow_continuation")
    _actions(native_actions)
    if not isinstance(plan, dict) or set(plan) != _PLAN_KEYS:
        raise ValueError("Plan must contain exactly reasoning, action and continuation")
    reasoning = _string(plan["reasoning"], "reasoning", limit=6000)
    action = _string(plan["action"], "action", limit=160)
    if action not in native_actions:
        raise ValueError("First action is not an exact member of the current native action set")
    continuation = plan["continuation"]
    if continuation is not None:
        if not allow_continuation:
            raise ValueError("structured_single requires continuation=null")
        if not isinstance(continuation, dict) or set(continuation) != _CONTINUATION_KEYS:
            raise ValueError("Continuation must be one object with exactly action and both predicate arrays")
        continuation = {"action": _string(continuation["action"], "continuation action", limit=160),
                        **{key: continuation[key] for key in _CONTINUATION_KEYS - {"action"}}}
        normalized = {}
        for key in ("observation_contains_all", "observation_contains_none"):
            predicates = continuation[key]
            minimum = 1 if key == "observation_contains_all" else 0
            if not isinstance(predicates, list) or not minimum <= len(predicates) <= 4:
                raise ValueError(f"{key} must contain {minimum} to 4 literal predicates")
            normalized[key] = [
                _normalize(_string(predicate, key + " predicate", limit=160)) for predicate in predicates
            ]
            if len(set(normalized[key])) != len(predicates):
                raise ValueError(f"{key} contains duplicate whitespace-normalized predicates")
            continuation[key] = list(predicates)
        if any(negative in positive for positive in normalized["observation_contains_all"]
               for negative in normalized["observation_contains_none"]):
            raise ValueError("Contradictory guard: a required predicate contains a forbidden predicate")
    return {"reasoning": reasoning, "action": action, "continuation": continuation}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON value is forbidden: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number is forbidden")
    return number


def parse_microplan(text: str, *, native_actions: list[str], allow_continuation: bool) -> dict:
    """Parse one strict JSON object; raise ValueError with no repair or fallback.

    Checks beyond JSON Schema include normalized predicate duplicates, substring
    contradictions and exact FIRST-action membership. The continuation may
    become native only after the first action; membership is checked at decide.
    Returned containers are owned copies, also when later installed into state.
    """
    _bool(allow_continuation, "allow_continuation")
    _string(text, "microplan JSON text")
    try:
        plan = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
                          parse_float=_finite_float)
    except json.JSONDecodeError as error:
        raise ValueError(f"Expected exactly one JSON object: {error.msg}") from error
    except RecursionError as error:
        raise ValueError("Microplan JSON nesting is too deep") from error
    return _validated_plan(plan, native_actions, allow_continuation)


def prompt_contract(mode: str) -> str:
    """Guarded and unguarded modes deliberately receive byte-identical instructions."""
    _mode(mode)
    common = (
        'Return one JSON object with required "reasoning", "action", "continuation" keys. '
        'JSON replaces earlier <think>/<action> output-tag instructions; all task, observation '
        'and native-action information still applies. No markdown, extra/duplicate keys or '
        'code/tool/replan calls.\n'
        '"reasoning": nonempty, <=6000 characters. "action" executes NOW: copy a complete current '
        'native command exactly, preserving every argument, case and whitespace (1..160 characters).\n'
    )
    if mode == "structured_single":
        return common + '"continuation" must be null. Choose only the current actor action.'
    return common + (
        'Provide ONE CONDITIONAL next action whenever a simple intermediate observation '
        'justifies it; use null if none can be specified or feedback requires new reasoning.\n'
        '"continuation": null or one object with "action", "observation_contains_all", '
        '"observation_contains_none". Check predicates BEFORE it executes, using the observation '
        'returned by the first "action": the INTERMEDIATE state, NEVER the continuation result.\n'
        'Continuation action: complete native syntax with already bound entity arguments, '
        '1..160 characters. Never shorten commands or guess unknown entity identities. '
        'contains_all: 1..4 predicates; contains_none: 0..4; each 1..160 nonempty characters. '
        'Use case-sensitive literal substrings with whitespace collapsed; no regex, normalized '
        'duplicates or contradictions. Guards and native membership still gate execution; '
        'then return to the actor.\n'
        'Timing example: invented commands, NOT valid task actions; DO NOT copy them:\n'
        '{"reasoning":"Zorp returns ZORP_READY before nuv runs.","action":"zorp arg7",'
        '"continuation":{"action":"nuv arg7 arg2","observation_contains_all":["ZORP_READY"],'
        '"observation_contains_none":[]}}'
    )


class MicroplanState:
    """Execute nothing; manage a single installed actor action and optional continuation."""

    def __init__(self, mode: str):
        self.mode = _mode(mode)
        self._phase = "empty"
        self._plan = None
        self._plan_hash = None
        self._feedback = None
        self._boundary_reason = None
        self._completed = 0

    def _audit(self) -> dict:
        return {
            "schema_version": STATE_SCHEMA, "mode": self.mode, "phase": self._phase,
            "plan_sha256": self._plan_hash, "completed_actions": self._completed,
            "actor_required": self._phase in ("empty", "awaiting_plan"),
            "boundary_reason": self._boundary_reason,
        }

    def _clear(self, reason: str, *, awaiting_plan: bool = False) -> None:
        self._plan = self._feedback = None
        self._phase = "awaiting_plan" if awaiting_plan else "empty"
        self._boundary_reason = reason

    def install(self, plan: dict, *, current_observation: str, current_actions: list[str]) -> dict:
        """Install an owned valid plan; current observation is not checked as future feedback."""
        if self._phase not in ("empty", "awaiting_plan"):
            raise ValueError("Cannot overwrite an outstanding microplan action or continuation")
        _string(current_observation, "current_observation", empty=True)
        validated = _validated_plan(plan, current_actions, self.mode != "structured_single")
        plan_hash = hashlib.sha256(json.dumps(
            validated, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest()
        self._plan, self._plan_hash = validated, plan_hash
        self._phase, self._boundary_reason, self._feedback = "awaiting_first_action", None, None
        return {
            **self._audit(), "reason": "actor_action_pending", "action": validated["action"],
            "continuation_queued": validated["continuation"] is not None,
        }

    def observe(self, action: str, observation: str, *, native_valid: bool) -> dict:
        """Record the exact pending actual action, never a proposed or substituted action."""
        if self._phase not in ("awaiting_first_action", "awaiting_continuation"):
            raise ValueError("No pending executed action to observe")
        _bool(native_valid, "native_valid")
        _string(action, "observed action", limit=160)
        _string(observation, "observed feedback", empty=True)
        first = self._phase == "awaiting_first_action"
        expected = self._plan["action"] if first else self._plan["continuation"]["action"]
        if action != expected:
            raise ValueError("Observed action does not match the pending microplan action")
        self._completed += 1
        if not native_valid:
            reason = "actor_action_not_native" if first else "continuation_action_not_native"
            self._clear(reason)
        elif not first:
            reason = "continuation_completed_actor_required"
            self._clear(reason)
        elif self._plan["continuation"] is None:
            reason = "actor_action_observed_no_continuation"
            self._clear(reason)
        else:
            reason = "continuation_ready"
            self._phase, self._feedback = "ready_continuation", observation
        return {
            **self._audit(), "reason": reason, "action": action, "observation": observation,
            "native_valid": native_valid, "action_role": "actor" if first else "continuation",
            "continuation_queued": self._plan is not None,
        }

    def decide(self, observation: str, native_actions: list[str]) -> dict:
        """Consider at most one local action after matching actual first-action feedback."""
        if self._phase not in ("empty", "ready_continuation"):
            raise ValueError("A pending decision/action needs install or observe before another decide")
        _string(observation, "decision observation", empty=True)
        _actions(native_actions)
        result = {
            "commit": False, "action": None, "candidate_action": None, "reason": "actor_required",
            "guard_checked": False, "guard_passed": None, "native_member": None,
            "missing_positive_predicates": [], "present_negative_predicates": [],
        }
        if self._phase == "empty":
            self._phase = "awaiting_plan"
            result["reason"] = "actor_boundary_required" if self._boundary_reason else "actor_required"
            return {**self._audit(), **result}
        if observation != self._feedback:
            raise ValueError("Decision observation does not match recorded first-action feedback")
        continuation = self._plan["continuation"]
        action = continuation["action"]
        result.update(candidate_action=action, native_member=action in native_actions)
        reason = None
        if action not in native_actions:
            reason = "continuation_not_native"
        elif self.mode == "guarded_plan":
            feedback = _normalize(self._feedback)
            missing = [term for term in continuation["observation_contains_all"]
                       if _normalize(term) not in feedback]
            forbidden = [term for term in continuation["observation_contains_none"]
                         if _normalize(term) in feedback]
            result.update(guard_checked=True, guard_passed=not missing and not forbidden,
                          missing_positive_predicates=missing, present_negative_predicates=forbidden)
            if missing or forbidden:
                reason = "observation_guard_failed"
        if reason is not None:
            result["reason"] = reason
            self._clear(reason, awaiting_plan=True)
        else:
            self._phase = "awaiting_continuation"
            result.update(commit=True, action=action, reason=(
                "observation_guard_passed" if self.mode == "guarded_plan" else "guard_enforcement_disabled"
            ))
        return {**self._audit(), **result}
