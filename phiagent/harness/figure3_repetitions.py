"""Prospective Figure 3 repetition policies; legacy freezes keep their seeds."""

from collections.abc import Mapping, Sequence

SINGLE_PASS_AUTHORIZATION = "user_speedup_20261004"


def test_seeds(settings: Mapping, registered: Sequence[int]) -> list[int]:
    policy = settings.get("test_repetition_policy")
    if policy is None:
        return list(registered)
    if (
        not isinstance(policy, Mapping)
        or policy.get("mode") != "single_pass"
        or policy.get("authorization") != SINGLE_PASS_AUTHORIZATION
        or policy.get("seed_rule") != "first_registered_seed_before_test"
        or policy.get("test_feedback_used") is not False
    ):
        raise ValueError("Single-pass TEST needs a prospective, outcome-independent user policy")
    return [registered[0]]


def single_pass_policy() -> dict:
    return {
        "mode": "single_pass", "authorization": SINGLE_PASS_AUTHORIZATION,
        "seed_rule": "first_registered_seed_before_test", "test_feedback_used": False,
    }
