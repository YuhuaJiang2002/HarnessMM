"""One supported-candidate commit or one fallback, without model or task imports.

Adapters supply candidates derived from public evidence and retain native I/O,
scheduling, and usage accounting. Support and membership checks are execution
guards, not certificates of semantic correctness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Generic, Literal, TypeVar


SCHEMA = "phiagent-supported-local-decision/v1"
ResultT = TypeVar("ResultT")
DecisionReason = Literal[
    "supported_candidate_committed",
    "unsupported_state",
    "empty_candidate_set",
    "invalid_candidate_set",
    "fallback_requested",
    "invalid_selection",
    "selection_not_supported",
]


@dataclass(frozen=True)
class SupportedCandidates:
    """An ordered public-evidence projection, after the adapter's scheduling gate."""

    supported: bool
    actions: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise ValueError("Candidate support must be an explicit boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Candidate support requires an explicit reason")


@dataclass(frozen=True)
class DecisionTrace:
    route: Literal["commit", "fallback"]
    reason: DecisionReason
    supported: bool
    support_reason: str
    candidate_count: int | None
    selector_called: bool
    proposed_action: str | None
    proposed_type: str | None
    committed_action: str | None

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class DecisionResult(Generic[ResultT]):
    value: ResultT
    trace: DecisionTrace


def execute_supported_decision(
    support: SupportedCandidates,
    *,
    commit: Callable[[str], ResultT],
    fallback: Callable[[], ResultT],
    select: Callable[[tuple[str, ...]], str | None] | None = None,
    fallback_action: str | None = None,
) -> DecisionResult[ResultT]:
    """Select only from supported candidates; commit or fall back exactly once.

    With no selector, choose the first candidate in the supplied order. A selector
    may return ``fallback_action`` to defer explicitly. Unsupported states, empty
    or malformed candidate tuples, and disallowed selections bypass commitment.
    Callback exceptions propagate unchanged: neither retries nor a second route
    are attempted. Callback invocation counts do not imply model/token counts.
    """
    if fallback_action is not None and (
        not isinstance(fallback_action, str) or not fallback_action.strip()
    ):
        raise ValueError("An explicit fallback action must be a nonempty string")

    actions = support.actions
    reason: DecisionReason | None = None
    if not support.supported:
        reason = "unsupported_state"
    elif not isinstance(actions, tuple):
        reason = "invalid_candidate_set"
    elif not actions:
        reason = "empty_candidate_set"
    elif (
        any(not isinstance(action, str) or not action.strip() for action in actions)
        or len(set(actions)) != len(actions)
        or fallback_action in actions
    ):
        reason = "invalid_candidate_set"

    proposed: object = None
    selector_called = False
    selected = reason is None
    committed_action = None
    if selected:
        selector_called = select is not None
        proposed = actions[0] if select is None else select(actions)
        if proposed == fallback_action:
            reason = "fallback_requested"
        elif not isinstance(proposed, str) or not proposed.strip():
            reason = "invalid_selection"
        elif proposed not in actions:
            reason = "selection_not_supported"
        else:
            committed_action = proposed

    trace = DecisionTrace(
        route="commit" if committed_action is not None else "fallback",
        reason=reason or "supported_candidate_committed",
        supported=support.supported,
        support_reason=support.reason,
        candidate_count=len(actions) if isinstance(actions, tuple) else None,
        selector_called=selector_called,
        proposed_action=proposed if isinstance(proposed, str) else None,
        proposed_type=type(proposed).__name__ if selected else None,
        committed_action=committed_action,
    )
    value = commit(committed_action) if committed_action is not None else fallback()
    return DecisionResult(value=value, trace=trace)
