"""Synthetic equivalence tests; no environment, provider or benchmark inputs."""

from itertools import product

import pytest

from phiagent.learning.public_recurrence import has_no_progress_recurrence


def frozen_reference(history, index, context, stalled, window=16, separation=2, visits=2):
    within_window = [(step, value) for step, value in history if index - step <= window]
    qualifying = [
        step
        for step, value in within_window
        if value == context
        and index - step >= separation
        and any(later_value != context for later, later_value in within_window if later > step)
    ]
    return stalled >= 8 and len(qualifying) + 1 >= visits


def test_matches_frozen_rule_across_synthetic_histories():
    for length in range(7):
        for values in product(("context-a", "context-b"), repeat=length):
            history = [(index + 1, value) for index, value in enumerate(values)]
            for context in ("context-a", "context-b"):
                for stalled in (0, 7, 8, 12):
                    for window in (3, 16):
                        actual = has_no_progress_recurrence(
                            history,
                            observation_index=length + 1,
                            context=context,
                            no_progress_actions=stalled,
                            window=window,
                        )
                        assert actual == frozen_reference(
                            history,
                            length + 1,
                            context,
                            stalled,
                            window=window,
                        )


def test_consecutive_repeats_are_not_nonconsecutive_recurrence():
    assert not has_no_progress_recurrence(
        [(1, "a"), (2, "a"), (3, "a")],
        observation_index=4,
        context="a",
        no_progress_actions=8,
    )
    assert has_no_progress_recurrence(
        [(1, "a"), (2, "b"), (3, "b")],
        observation_index=4,
        context="a",
        no_progress_actions=8,
    )


def test_progress_reset_and_new_context_do_not_trigger():
    history = [(1, "a"), (2, "b")]
    assert not has_no_progress_recurrence(
        history,
        observation_index=3,
        context="a",
        no_progress_actions=0,
    )
    assert not has_no_progress_recurrence(
        history,
        observation_index=3,
        context="new-public-state",
        no_progress_actions=8,
    )


@pytest.mark.parametrize("history", [[(2, "a"), (1, "b")], [(1, "a"), (1, "b")], [(3, "a")]])
def test_invalid_order_cannot_establish_recurrence(history):
    with pytest.raises(ValueError, match="history indices"):
        has_no_progress_recurrence(
            history,
            observation_index=3,
            context="a",
            no_progress_actions=8,
        )


def test_unknown_context_and_boolean_threshold_are_rejected():
    with pytest.raises(ValueError, match="context"):
        has_no_progress_recurrence([], observation_index=0, context="", no_progress_actions=0)
    with pytest.raises(ValueError, match="stall_actions"):
        has_no_progress_recurrence(
            [],
            observation_index=0,
            context="a",
            no_progress_actions=0,
            stall_actions=True,
        )
