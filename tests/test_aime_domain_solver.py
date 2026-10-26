"""Synthetic tests: no benchmark payloads, evaluator, model, or runner imports."""
from copy import deepcopy
import re

import pytest

from phiagent.harness.aime_domain_solver import (
    DOMAIN_RULES, inherited_calls, run_domain_solver, solver_prompt,
)


def row():
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return {"response": "A synthetic derivation.\nANSWER: 3", "hard": 1,
            "usage": usage, "raw_response": {"id": "synthetic-call",
                "usage": deepcopy(usage), "choices": [{"finish_reason": "stop",
                    "message": {"content": "A synthetic derivation.\nANSWER: 3"}}]}}


class FakeStudy:
    def __init__(self):
        self.calls = []

    def infer(self, item, skill, role, maximum):
        self.calls.append({"question": item["question"], "skill": skill,
                           "role": role, "maximum": maximum})
        return row()


def test_single_solver_request_with_matched_capacity_and_native_response():
    study = FakeStudy()
    actual = run_domain_solver(study, {"question": "Synthetic problem."}, "Exact arithmetic.")
    assert len(study.calls) == 1
    assert study.calls[0]["maximum"] == 32768
    assert actual["raw_response"] == row()["raw_response"]
    assert actual["usage"] == row()["usage"]
    assert actual["domain_solver"]["calls"] == 1
    assert not actual["domain_solver"]["reused"]


def test_skill_has_no_numeric_examples_task_ids_or_conditional_question_dispatch():
    assert not re.search(r"\d", DOMAIN_RULES)
    assert all(word not in DOMAIN_RULES.lower() for word in ("aime", "coin", "silas", "training answer"))
    assert "exceptional prefix or suffix" in DOMAIN_RULES
    assert "transformed object legal" in DOMAIN_RULES
    assert "boundary contribution explicitly" in DOMAIN_RULES


def test_training_labels_and_ids_do_not_enter_solver_prompt():
    calls = []
    for identity, answer in (("synthetic-left", "private-left"), ("synthetic-right", "private-right")):
        study = FakeStudy()
        run_domain_solver(study, {"id": identity, "question": "Same public input.", "answer": answer},
                          "Exact arithmetic.")
        calls.append(study.calls)
    assert calls[0] == calls[1]
    assert "private" not in calls[0][0]["skill"]


def test_changed_guidance_changes_frozen_request():
    assert solver_prompt("Exact arithmetic.") != solver_prompt("Use symbolic algebra.")
    assert solver_prompt("Exact arithmetic.") == solver_prompt("Exact arithmetic.")


def ledger():
    return {"unknown_usage_calls": 0, "inflight_reserved_tokens": 0, "known_tokens": 36,
            "calls": [
                {"sequence": 1, "status": "SETTLED", "role": "old", "origin": "/oldest",
                 "usage": deepcopy(row()["usage"])},
                {"sequence": 2, "status": "SETTLED", "role": "newer",
                 "usage": deepcopy(row()["usage"])},
            ]}


def test_multiple_generation_cost_lineage_preserves_origins_once():
    prior = ledger()
    frozen = deepcopy(prior)
    calls = inherited_calls(prior, "/prior")
    assert calls[1]["origin"] == "/oldest" and calls[2]["origin"] == "/prior"
    assert sum(call["usage"]["total_tokens"] for call in calls.values()) == 36
    assert prior == frozen
    replay = {**prior, "calls": list(calls.values())}
    again = inherited_calls(replay, "/latest")
    assert [call["origin"] for call in again.values()] == ["/oldest", "/prior"]


@pytest.mark.parametrize("change", ["duplicate", "gap", "unknown", "unsettled", "bad_sum", "bad_native"])
def test_invalid_inherited_ledger_fails_explicitly(change):
    prior = ledger()
    if change == "duplicate":
        prior["calls"][1]["sequence"] = 1
    elif change == "gap":
        prior["calls"][1]["sequence"] = 3
    elif change == "unknown":
        prior["unknown_usage_calls"] = 1
    elif change == "unsettled":
        prior["calls"][1]["status"] = "INFLIGHT"
    elif change == "bad_sum":
        prior["known_tokens"] += 1
    else:
        prior["calls"][1]["usage"]["completion_tokens"] += 1
    with pytest.raises(RuntimeError):
        inherited_calls(prior, "/prior")
