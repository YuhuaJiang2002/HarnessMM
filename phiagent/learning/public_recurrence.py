"""Shared recurrence predicate over adapter-supplied public observations."""

from __future__ import annotations

from collections.abc import Iterable


def has_no_progress_recurrence(
    history: Iterable[tuple[int, str]],
    *,
    observation_index: int,
    context: str,
    no_progress_actions: int,
    window: int = 16,
    minimum_action_separation: int = 2,
    recurrence_visits: int = 2,
    stall_actions: int = 8,
) -> bool:
    """Match the frozen ALF rule without inferring hidden environment state.

    Adapters determine public context and progress. Indices only establish
    ordering/window membership; they must never be treated as progress evidence.
    """
    for name, value, minimum in (
        ("observation_index", observation_index, 0),
        ("no_progress_actions", no_progress_actions, 0),
        ("window", window, 1),
        ("minimum_action_separation", minimum_action_separation, 1),
        ("recurrence_visits", recurrence_visits, 2),
        ("stall_actions", stall_actions, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(context, str) or not context:
        raise ValueError("a nonempty public context identifier is required")
    entries = []
    previous_index = -1
    for index, previous_context in history:
        if type(index) is not int or not previous_index < index < observation_index:
            raise ValueError(
                "history indices must be strictly increasing and precede the observation"
            )
        if not isinstance(previous_context, str) or not previous_context:
            raise ValueError("history requires nonempty public context identifiers")
        previous_index = index
        if observation_index - index <= window:
            entries.append((index, previous_context))
    qualifying = [
        index
        for index, previous_context in entries
        if previous_context == context
        and observation_index - index >= minimum_action_separation
        and any(later_context != context for later, later_context in entries if later > index)
    ]
    return no_progress_actions >= stall_actions and len(qualifying) + 1 >= recurrence_visits
