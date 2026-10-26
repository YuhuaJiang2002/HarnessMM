"""Synthetic residual-learning contracts; no benchmark data, SDK or model calls."""

import copy

import pytest

from phiagent.evaluation.edgebench_native import validate_native_outcome
from phiagent.harness.deepseek_chisel_policy import (
    fit_candidate, measured_pair, route, seal_stage, select_validation, validate_proposal,
    verify_stage,
)


def tasks():
    return [
        {"task_id": str(index), "unit_id": f"unit-{index // 2}",
         "public_prompt": f"Inspect the synthetic output at step {index}."}
        for index in range(8)
    ]


def rows(score=0.0, tokens=100):
    return {task["task_id"]: {"native_score": score, "total_tokens": tokens,
                            "usage_complete": True} for task in tasks()}


def admitted():
    proposal = validate_proposal("Check the current public receipt.", ["0", "2"], list(rows()))
    return fit_candidate(proposal, tasks(), rows(), rows(1.0, 80))


def test_training_groups_and_positive_head_preserve_method():
    candidate = admitted()
    assert candidate["enabled"]
    assert candidate["training_tasks"] == 8
    assert candidate["training_units"] == 4
    assert candidate["head"]["training_task_count"] == 4
    assert len(candidate["head"]["training_task_folds"]) == 4
    assert route(candidate, tasks()[0]["public_prompt"])["instruction"] == candidate["instruction"]


def test_no_gain_retains_baseline_and_out_of_support_abstains():
    proposal = validate_proposal("Check the public receipt.", ["0"], list(rows()))
    candidate = fit_candidate(proposal, tasks(), rows(1), rows(1))
    assert not candidate["enabled"]
    assert not route(candidate, tasks()[0]["public_prompt"])["instruction"]
    assert route(admitted(), "x" * 10000)["reason"] == "outside_training_support"
    assert route(None, "synthetic")["instruction"] == ""


@pytest.mark.parametrize("support", [["held-out"], ["0", "0"], "0"])
def test_proposals_cannot_cite_nontraining_or_duplicate_support(support):
    with pytest.raises(ValueError, match="TRAIN"):
        validate_proposal("A residual.", support, ["0"])


@pytest.mark.parametrize("instruction,support", [
    ("x" * 1201, ["0"]), ("x " * 181, ["0"]), ("x\0", ["0"]), ("", ["0"]), ("edit", []),
])
def test_sparse_proposal_and_abstention_contract(instruction, support):
    with pytest.raises(ValueError, match="sparse residual"):
        validate_proposal(instruction, support, ["0"])


@pytest.mark.parametrize("changes", [
    {"total_tokens": None}, {"total_tokens": True}, {"native_score": None},
    {"native_score": float("nan")}, {"native_score": True}, {"native_score": -1},
    {"usage_complete": False},
])
def test_unknown_cost_or_score_cannot_train_or_select(changes):
    with pytest.raises(ValueError, match="Complete native"):
        measured_pair({**rows()["0"], **changes})


def test_partial_training_and_tampered_candidate_fail():
    proposal = validate_proposal("Inspect the receipt.", ["0"], list(rows()))
    partial = rows()
    del partial["0"]
    with pytest.raises(ValueError, match="complete paired"):
        fit_candidate(proposal, tasks(), partial, rows(1))
    candidate = admitted()
    candidate["instruction"] = "A post-training change."
    with pytest.raises(ValueError, match="hash mismatch"):
        route(candidate, tasks()[0]["public_prompt"])


def test_validation_requires_complete_population_and_strict_gain():
    candidate = admitted()
    ident = candidate["id"]
    outcomes = {"baseline": rows(0.5), ident: rows(0.75, 80)}
    selection = select_validation([candidate], outcomes, list(rows()))
    assert selection["selected_candidate_id"] == ident
    assert selection["validation_used_for_optimization"] is False
    assert selection["test_opened"] is False
    outcomes[ident] = rows(0.5, 50)
    assert select_validation([candidate], outcomes, list(rows()))[
        "selected_candidate_id"] == "baseline"
    incomplete = copy.deepcopy(outcomes)
    del incomplete[ident]["0"]
    with pytest.raises(ValueError, match="Incomplete"):
        select_validation([candidate], incomplete, list(rows()))
    with pytest.raises(ValueError, match="every admitted"):
        select_validation([candidate], {"baseline": rows()}, list(rows()))


def test_held_out_releases_are_bound_to_complete_earlier_phases_and_sources():
    candidate = admitted()
    partition = "a" * 64
    document = {
        "partition_sha256": partition, "training_finalized": True,
        "frozen_at_unix": 100, "source_files": {"/frozen/controller.py": "b" * 64},
        "candidates": [candidate],
    }
    release = seal_stage(document, "validation")
    verify_stage(release, "validation", partition, 101)
    with pytest.raises(ValueError, match="freeze"):
        verify_stage(release, "validation", partition, 99)
    with pytest.raises(ValueError, match="freeze"):
        verify_stage(release, "test", partition, 101)
    changed = {**release, "training_finalized": False}
    with pytest.raises(ValueError, match="hash"):
        verify_stage(changed, "validation", partition, 101)
    selection = select_validation(
        [candidate], {"baseline": rows(), candidate["id"]: rows(1)}, list(rows()),
    )
    test_release = seal_stage({
        **document, "selection": selection, "frozen_before_test": True,
    }, "test")
    verify_stage(test_release, "test", partition, 101)


def native_rows(improvement=0.0):
    result = {}
    for task in tasks():
        index = int(task["task_id"])
        scale = 10000 if index % 2 else 1
        outcome = validate_native_outcome(
            {"score": (10 + improvement) * scale, "score_0_100": None,
             "valid": True, "pass_rate": 1, "errors": 0, "timed_out": False},
            task_id=task["task_id"], score_direction="maximize", selection_policy="score_first",
            normalization_declared=False, infrastructure_ok=True,
        )
        result[task["task_id"]] = {
            "native_outcome": outcome, "total_tokens": 100, "usage_complete": True,
        }
    return result


def test_mixed_native_units_are_used_only_for_internal_task_preferences():
    proposal = validate_proposal("Check the public receipt.", ["0", "2"], list(rows()))
    baseline, improved = native_rows(), native_rows(1)
    candidate = fit_candidate(proposal, tasks(), baseline, improved, native_preferences=True)
    assert candidate["enabled"]
    assert candidate["head"]["oof_mean_utility"] == 1
    selection = select_validation(
        [candidate], {"baseline": baseline, candidate["id"]: improved},
        list(baseline), native_preferences=True,
    )
    assert selection["selected_candidate_id"] == candidate["id"]
    assert selection["quality_aggregate"] is None
    improved["0"] = native_rows(-1)["0"]
    selection = select_validation(
        [candidate], {"baseline": baseline, candidate["id"]: improved},
        list(baseline), native_preferences=True,
    )
    assert selection["selected_candidate_id"] == "baseline"
