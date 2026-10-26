"""Train-derived domain guards integrated into one full-capacity solver call."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from typing import Any

from phiagent.harness.aime_counterfactual import (
    MAXIMUM_OUTPUT_TOKENS,
    _DIRECT_VIEW,
    _checked_output,
    _solver_prompt,
    parse_final_answer,
)

POLICY_VERSION = "aime-domain-solver/20260928-v1"
DOMAIN_RULES = """Carry domain restrictions through the derivation, not just into a
final verbal check. Use the following construction discipline within this solve.

First write the variables and their legal domains. For any transformation that
removes objects, decreases a count, divides by an expression, introduces a root,
or constructs a geometric configuration, derive the condition that makes the
transformed object legal. Keep that condition attached to every resulting case.
An algebraically attractive alternative is not a competitor if it is infeasible.

Partition off boundary and degenerate cases before extending an interior rule.
Evaluate the smallest admissible parameter, zero counts, endpoints, and vanishing
factors directly from the original definitions when they occur. A rule obtained
by changing a count is not valid at a boundary where that change is impossible.
Write the boundary contribution explicitly rather than claiming it was checked.

Only then sum or count the cases. Before multiplying a repeating pattern across
blocks, identify the blocks on which all transformation guards hold; calculate
any exceptional prefix or suffix separately. Show that the final cases are
disjoint and cover the original domain. In geometry, retain the orientation,
interiority, and nondegeneracy conditions when selecting algebraic branches.
Use these steps to build the proof; do not append an independent second solution."""


def solver_prompt(skill: str) -> str:
    return _solver_prompt(skill, _DIRECT_VIEW + "\n\n" + DOMAIN_RULES)


def run_domain_solver(study, item: Mapping[str, Any], skill: str) -> dict[str, Any]:
    prompt = solver_prompt(skill)
    row = deepcopy(study.infer(item, prompt, "chisel_train_domain_solver",
                              maximum=MAXIMUM_OUTPUT_TOKENS))
    public = _checked_output(row)
    parsed = parse_final_answer(public)
    row["domain_solver"] = {
        "policy_version": POLICY_VERSION,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "calls": 1, "reused": False, "token_usage_mode": "unlimited",
        "maximum_output_tokens": MAXIMUM_OUTPUT_TOKENS,
        "final_declaration_status": parsed.reason,
        "training_feedback_available_at_inference": False,
    }
    return row


def inherited_calls(previous: Mapping[str, Any], prior_root: str) -> dict[int, dict[str, Any]]:
    """Preserve original evidence locations across multiple paid continuations."""
    if previous["unknown_usage_calls"] or previous["inflight_reserved_tokens"]:
        raise RuntimeError("Cannot inherit unresolved provider usage")
    calls: dict[int, dict[str, Any]] = {}
    for call in previous["calls"]:
        sequence, usage = call["sequence"], call["usage"]
        if type(sequence) is not int or sequence < 1 or sequence in calls:
            raise RuntimeError("Duplicate or invalid inherited sequence")
        if call["status"] != "SETTLED" or not isinstance(usage, Mapping):
            raise RuntimeError("Inherited provider call is not settled")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(usage.get(field)) is not int or usage[field] < 0:
                raise RuntimeError("Invalid inherited token usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise RuntimeError("Inconsistent inherited token usage")
        origin = call.get("origin", prior_root)
        if not isinstance(origin, str) or not origin:
            raise RuntimeError("Missing inherited provider origin")
        calls[sequence] = {**deepcopy(call), "origin": origin, "inherited": True}
    if sorted(calls) != list(range(1, len(calls) + 1)):
        raise RuntimeError("Inherited call sequences must be contiguous")
    if sum(call["usage"]["total_tokens"] for call in calls.values()) != previous["known_tokens"]:
        raise RuntimeError("Inherited native token sum differs")
    return calls
