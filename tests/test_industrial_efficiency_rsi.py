"""Synthetic cost-aware selection checks; no benchmark inputs or runtime calls."""

from copy import deepcopy

import pytest

from phiagent.harness.industrial_efficiency_rsi import (
    IndustrialEfficiencyController, OBJECTIVE, actor_only_result, efficiency_gate,
)


def result(score=.5, tokens=1000, applications=8):
    return {
        "complete_native_scores": True, "mean_native_score": score,
        "measured_total_tokens": tokens, "residual_applications": applications,
    }


def test_equal_quality_lower_cost_is_an_improvement():
    gate = efficiency_gate(result(), result(tokens=750))
    assert gate["accepted"] and gate["token_reduction_fraction"] == .25


@pytest.mark.parametrize("candidate", [
    result(tokens=1001), result(tokens=1000), result(score=.49, tokens=500),
    result(score=.9, tokens=1500), result(tokens=500, applications=0),
])
def test_quality_or_cost_regression_or_noop_is_not_admitted(candidate):
    assert not efficiency_gate(result(), candidate)["accepted"]


def test_all_zero_scores_do_not_establish_quality_preservation():
    assert not efficiency_gate(result(score=0), result(score=0, tokens=50))["accepted"]


def test_unresolved_cost_is_not_zero():
    candidate = result()
    candidate["measured_total_tokens"] = None
    assert efficiency_gate(result(), candidate)["reason"] == "incomplete_native_quality_or_cost"


def test_missing_native_score_is_not_a_success():
    candidate = result()
    candidate["complete_native_scores"] = False
    assert not efficiency_gate(result(), candidate)["accepted"]


def test_objective_carries_both_successful_mechanisms_without_shortcuts():
    for phrase in ("complete-task", "ALFWorld", "SoL-Pi", "complete callable tool schemas",
                   "unconditional polling", "original observations", "validation and test"):
        assert phrase in OBJECTIVE


def test_old_test_cannot_be_opened_by_successor():
    controller = object.__new__(IndustrialEfficiencyController)
    controller._execution_identity = {"runtime": "official_codex_cli"}
    with pytest.raises(RuntimeError, match="previously opened TEST"):
        controller.evaluate_test(lambda: pytest.fail("TEST loader invoked"))


def test_optimizer_objective_rebound_before_request_seal(monkeypatch):
    from phiagent.harness import codex_chisel as core

    controller = object.__new__(IndustrialEfficiencyController)
    controller._execution_identity = {"runtime": "official_codex_cli"}
    incoming = {"objective": "old quality-only objective", "phase": "train"}
    original = deepcopy(incoming)
    captured = {}

    def invoke(self, callback, request, kind, **kwargs):
        captured.update(request)
        return {"observed": True}

    monkeypatch.setattr(core.CodexChiselController, "_invoke", invoke)
    assert controller._invoke(None, incoming, "optimizer")["observed"]
    assert incoming == original
    assert captured["objective"] == OBJECTIVE


def test_full_cost_aware_training_validation_and_freeze():
    from tests.test_codex_chisel import (
        EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SOURCE_HASHES, SyntheticCallbacks, population, usage,
    )
    from phiagent.harness.codex_chisel import ChiselConfig

    callbacks = SyntheticCallbacks(score=lambda request: 1.0)
    manifest, index, splits = population(units=50)

    def execute(request):
        value = callbacks.execute(request)
        value["usage"] = usage(10 if request["residual"] else 100, 5)
        return value

    controller = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=execute, optimize=callbacks.optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-efficiency-loop", config=ChiselConfig(),
    )
    report = controller.train(splits["train"])
    assert report["state"] == "trained"
    assert report["industrial_efficiency_successor"]["unchanged_industrial_runtime"]
    assert callbacks.optimizations[0]["objective"] == OBJECTIVE
    assert callbacks.optimizations[1]["previous_candidates"]
    selection = controller.select_validation(splits["validation"])
    assert selection["reason"] == "quality_preserving_token_saving"
    policy = controller.freeze(source_hashes=SOURCE_HASHES)
    assert policy.to_dict()["sources"]["external"]["industrial_efficiency_rsi"]
    assert policy.to_dict()["candidate"]["enabled"]
    assert "VALIDATION_SECRET_DIAGNOSTIC" not in str(callbacks.optimizations)
    assert all(row["phase"] != "test" for row in callbacks.executions)


def test_roundoff_tolerance_does_not_admit_a_real_below_threshold_score():
    from phiagent.harness import codex_chisel as core
    import math

    features = core.public_text_features("Synthetic count task.")
    candidate = {
        "id": "synthetic", "enabled": True,
        "feature_bounds": {name: [value, value] for name, value in features.items()},
        "head": {
            "intercept": math.nextafter(.5, 0), "score_threshold": .5,
            "weights": {name: 0.0 for name in core._feature_schema()},
        },
    }
    assert core._route(candidate, "Synthetic count task.")["residual_id"] == "synthetic"
    candidate["head"]["intercept"] = .5 - 1e-8
    assert core._route(candidate, "Synthetic count task.")["residual_id"] is None


def test_deepseek_proposer_receives_complete_actor_cost():
    from scripts.run_deepseek_gaia2_campaign import optimizer_payload

    observation = {"native_score": 1, "execution_status": "completed",
                   "usage": {"total_tokens": 876}, "feedback": ""}
    request = {
        "phase": "train", "objective": "synthetic", "limits": {"max_characters": 100, "max_words": 20},
        "previous_candidates": [],
        "training_evidence": [{"task": {"task_id": "synthetic"}, "seed": 1,
                               "baseline": observation, "candidates": {}}],
    }
    payload = optimizer_payload(request, {"synthetic"})
    assert payload["training_evidence"][0]["baseline"]["complete_task_actor_tokens"] == 876


def test_successor_output_roots_preserve_original_native_contracts():
    from pathlib import Path
    from scripts.run_industrial_efficiency_rsi import validate_output_root

    for harness, root in (
        ("codex", "/opt/phiagent/work/codex-chisel/20260929-v1/gaia2/synthetic-new"),
        ("deepseek", "/opt/phiagent/work/deepseek-harness/20260929-v1/runs/synthetic-new"),
    ):
        validate_output_root(Path(root), harness)
        with pytest.raises(ValueError, match="original native adapter"):
            validate_output_root(Path("/path/to/data/phiagent/work/synthetic"), harness)


def test_native_judge_is_preserved_but_not_used_as_actor_objective():
    from tests.test_codex_chisel import usage

    actor, judge = usage(100, 20), usage(10, 5)
    original = {
        "usage": usage(110, 25),
        "provenance": {"provider_usage": {
            "complete": True, "actor": {"usage": actor}, "judge": {"usage": judge},
        }},
    }
    result = actor_only_result(original)
    assert result["usage"]["total_tokens"] == 120
    assert original["usage"]["total_tokens"] == 135
    assert result["provenance"]["efficiency_cost_roles"]["judge_usage_separate"] == judge
    original["usage"] = usage(111, 25)
    with pytest.raises(ValueError, match="do not drop recovery costs"):
        actor_only_result(original)


@pytest.mark.parametrize("improvement", [True, False])
def test_quality_and_cost_extremes_stay_inside_trainer_utility_contract(improvement):
    from tests.test_codex_chisel import (
        EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SyntheticCallbacks, population, usage,
    )

    callbacks = SyntheticCallbacks(
        score=lambda request: float(bool(request["residual"]) == improvement),
    )
    manifest, index, splits = population(units=50)

    def execute(request):
        value = callbacks.execute(request)
        value["usage"] = usage(
            (10 if improvement else 200) if request["residual"] else 100, 5,
        )
        return value

    controller = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=execute, optimize=callbacks.optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-extreme-utility",
    )
    report = controller.train(splits["train"])
    assert report["state"] == "trained"
    examples = next(row["efficiency_training_examples"] for row in report["candidate_bank"]
                    if row["id"] != "noop")
    assert all(-1 <= row["utility"] <= 1 for row in examples)
    assert all(row["harmful"] is not improvement for row in examples)


def test_bad_model_support_is_rejected_and_paid_without_stopping_next_round():
    from tests.test_codex_chisel import (
        EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SyntheticCallbacks, population, usage,
    )

    callbacks = SyntheticCallbacks(score=lambda request: 1.0)
    manifest, index, splits = population(units=50)

    def execute(request):
        value = callbacks.execute(request)
        value["usage"] = usage(10 if request["residual"] else 100, 5)
        return value

    def optimize(request):
        result = callbacks.optimize(request)
        if request["iteration"] == 0:
            result["supporting_task_ids"] = ["not-supplied-to-optimizer"]
        return result

    controller = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=execute, optimize=optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-rejected-support",
    )
    report = controller.train(splits["train"])
    assert report["state"] == "trained"
    assert report["optimization_iterations"][0]["status"] == "optimizer_abstained_or_failed"
    assert report["optimization_iterations"][1]["status"] == "admitted_training_oof"
    assert report["accounting"]["by_stage"]["training_optimizer"]["usage"]["total_tokens"]["total"] == 240
    assert len(callbacks.executions) == 2 * len(splits["train"])
    failed = next(value for value in report["observations"].values()
                  if value["execution_status"] == "failed")
    assert failed["provenance"]["rejected_untrusted_proposal"]["actual_usage_preserved"]


def refit_fixture():
    import hashlib
    import json
    from tests.test_codex_chisel import (
        EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SyntheticCallbacks, population, usage,
    )

    callbacks = SyntheticCallbacks(score=lambda request: 1.0)
    manifest, index, splits = population(units=50)

    def execute(request):
        value = callbacks.execute(request)
        value["usage"] = usage(10 if request["residual"] else 100, 5)
        return value

    def optimize(request):
        value = callbacks.optimize(request)
        if request["iteration"] == 1:
            value.update(execution_status="failed", instruction="", supporting_task_ids=[])
            value["usage"] = usage(500, 50)
        return value

    original = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=execute, optimize=optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-source-training",
    )
    source = original.train(splits["train"])
    source["predecessor_accounting"] = {"failed_proposer_tokens": 41}

    def forbidden(request):
        pytest.fail("Refit invoked a model callback")

    fresh = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=forbidden, optimize=forbidden,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-new-refit",
    )
    options = {
        "source_report": json.dumps(source).encode(),
        "source_report_path": "synthetic/TRAIN_COMPLETE.json",
        "native_evidence_sha256": hashlib.sha256(b"synthetic-native-evidence").hexdigest(),
    }
    return fresh, source, splits, options, callbacks, execute


def test_native_quality_only_harm_allows_individual_cost_increases():
    from tests.test_codex_chisel import (
        EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SyntheticCallbacks, population, usage,
    )

    manifest, index, splits = population(units=50)
    expensive = splits["train"][0]["task_id"]
    callbacks = SyntheticCallbacks(score=lambda request: 1.0)

    def execute(request):
        value = callbacks.execute(request)
        cost = 100
        if request["residual"]:
            cost = 150 if request["task"]["task_id"] == expensive else 10
        value["usage"] = usage(cost, 5)
        return value

    controller = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=execute, optimize=callbacks.optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-population-cost",
    )
    report = controller.train(splits["train"])
    candidate = next(item for item in report["candidate_bank"] if item["id"] != "noop")
    examples = candidate["efficiency_training_examples"]
    assert any(item["token_delta"] > 0 for item in examples)
    assert not any(item["harmful"] for item in examples)
    assert candidate["head"]["oof_harmful_rate"] == 0
    assert candidate["head"]["oof_mean_utility"] > 0
    assert candidate["head"]["oof_mean_token_delta"] < 0
    assert candidate["enabled"]


def test_refit_imports_paid_history_without_replaying_actors_or_proposers():
    import hashlib

    fresh, source, splits, options, callbacks, _ = refit_fixture()
    count = len(callbacks.executions), len(callbacks.optimizations)
    report = fresh.refit_training_report(splits["train"], **options)
    assert fresh.state == "trained"
    assert (len(callbacks.executions), len(callbacks.optimizations)) == count
    assert report["ledger"] == source["ledger"]
    assert report["observations"] == source["observations"]
    assert report["optimizer_evidence"] == source["optimizer_evidence"]
    assert report["accounting"] == source["accounting"]
    lineage = report["training_refit"]
    assert lineage["source_report_sha256"] == hashlib.sha256(options["source_report"]).hexdigest()
    assert lineage["new_train_callbacks"] == lineage["new_train_model_calls"] == 0
    assert lineage["source_predecessor_accounting"]["failed_proposer_tokens"] == 41
    assert lineage["bank_frozen_before_validation"]
    before = {item["id"]: item for item in source["candidate_bank"]}
    for candidate in fresh.candidate_bank:
        for key in ("id", "instruction", "supporting_task_ids"):
            assert candidate[key] == before[candidate["id"]][key]
        if candidate["id"] != "noop":
            assert candidate["paired_evidence_sha256"] == before[candidate["id"]]["paired_evidence_sha256"]
    failed = [item for item in report["observations"].values()
              if item["execution_status"] == "failed"]
    assert len(failed) == 1 and failed[0]["usage"]["total_tokens"] == 550
    with pytest.raises(RuntimeError, match="requires state initialized"):
        fresh.refit_training_report(splits["train"], **options)


def test_refit_replaces_old_cost_harm_heads_not_original_instruction_evidence():
    import json
    from phiagent.harness import codex_chisel as core

    fresh, source, splits, options, _, _ = refit_fixture()
    candidate = source["candidate_bank"][1]
    candidate["head"]["oof_harmful_rate"] = .25
    candidate["enabled"] = False
    candidate["admission"] = "no_quality_preserving_oof_token_saving"
    source["candidate_bank"][1] = core._seal(
        {key: value for key, value in candidate.items() if key != "candidate_sha256"},
        "candidate_sha256",
    )
    options["source_report"] = json.dumps(source).encode()
    report = fresh.refit_training_report(splits["train"], **options)
    updated = fresh.candidate_bank[1]
    assert updated["enabled"] and updated["head"]["oof_harmful_rate"] == 0
    assert updated["candidate_sha256"] != source["candidate_bank"][1]["candidate_sha256"]
    assert report["training_refit"]["candidates"][0]["source_admission"] == candidate["admission"]
    assert report["observations"] == source["observations"]


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "validation", "usage", "support", "same_run", "seed", "router",
])
def test_refit_rejects_incomplete_changed_or_unsealed_training(mutation):
    import json
    from phiagent.harness import codex_chisel as core

    fresh, source, splits, options, _, _ = refit_fixture()
    actor = next(row for row in source["ledger"] if row["kind"] == "executor")
    if mutation == "missing":
        source["observations"].pop(actor["request_id"])
    elif mutation == "duplicate":
        source["ledger"].append(deepcopy(actor))
    elif mutation == "validation":
        actor["phase"] = "validation"
    elif mutation == "usage":
        source["observations"][actor["request_id"]]["usage"]["total_tokens"] += 1
    elif mutation == "support":
        source["candidate_bank"][1]["supporting_task_ids"] = ["not-training", "not-training-2"]
        source["candidate_bank"][1] = core._seal(source["candidate_bank"][1], "candidate_sha256")
    elif mutation == "same_run":
        source["run_id"] = fresh._run_id
    elif mutation == "seed":
        source["accounting"]["execution_seeds"] = [123]
    else:
        source["candidate_bank"][1]["head"]["ridge"] = 2
        source["candidate_bank"][1] = core._seal(source["candidate_bank"][1], "candidate_sha256")
    options["source_report"] = json.dumps(source).encode()
    with pytest.raises(ValueError):
        fresh.refit_training_report(splits["train"], **options)
    assert fresh.state == "initialized"


def test_refit_bank_cannot_change_before_validation():
    from phiagent.learning.residual_meta_harness import canonical_json

    fresh, _, splits, options, _, _ = refit_fixture()
    fresh.refit_training_report(splits["train"], **options)
    altered = fresh.candidate_bank
    altered[1]["instruction"] = "Changed after TRAIN freeze."
    fresh._bank_json = canonical_json(altered)
    with pytest.raises(RuntimeError, match="Frozen refit candidate bank changed"):
        fresh.select_validation(splits["validation"])


def test_refit_validation_uses_frozen_router_and_full_population_cost_gate():
    from tests.test_codex_chisel import SOURCE_HASHES, usage

    fresh, source, splits, options, callbacks, execute = refit_fixture()
    fresh.refit_training_report(splits["train"], **options)
    first_validation = sorted(task["task_id"] for task in splits["validation"])[0]

    def validation_only(request):
        assert request["phase"] == "validation"
        value = execute(request)
        if request["residual"] and request["task"]["task_id"] == first_validation:
            value["usage"] = usage(150, 5)
        return value

    fresh._execute_callback = validation_only
    selected = fresh.select_validation(splits["validation"])
    assert selected["reason"] == "quality_preserving_token_saving"
    assert selected["results"][0]["tasks"] == selected["results"][1]["tasks"] == 10
    assert selected["results"][1]["residual_applications"] == 10
    assert len([row for row in fresh.report()["ledger"] if row["phase"] == "train"]) == len(source["ledger"])
    assert not selected["validation_used_for_optimization"]
    assert all(row["phase"] != "test" for row in callbacks.executions)
    policy = fresh.freeze(source_hashes=SOURCE_HASHES)
    assert policy.to_dict()["candidate"]["enabled"]
    with pytest.raises(RuntimeError, match="previously opened TEST"):
        fresh.evaluate_test(lambda: pytest.fail("TEST loaded"))


def test_population_quality_loss_cannot_be_hidden_outside_oof_overrides(monkeypatch):
    from tests.test_codex_chisel import (
        EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SyntheticCallbacks, population, usage,
    )
    from phiagent.harness import industrial_efficiency_rsi as module

    manifest, index, splits = population(units=50)
    callbacks = SyntheticCallbacks(score=lambda request: 0.0 if request["residual"] else 1.0)

    def execute(request):
        value = callbacks.execute(request)
        value["usage"] = usage(1 if request["residual"] else 100, 1, cache=0)
        return value

    original_fit = module.training.fit_cross_fitted_linear_head

    def selective_head(*args, **kwargs):
        head = original_fit(*args, **kwargs)
        head.update(status="DISCOVERY_CANDIDATE", oof_mean_utility=.2,
                    oof_harmful_rate=0, oof_mean_token_delta=-.2)
        return head

    monkeypatch.setattr(module.training, "fit_cross_fitted_linear_head", selective_head)
    controller = IndustrialEfficiencyController(
        partition=manifest, task_index=index, execute=execute, optimize=callbacks.optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-quality-rejection",
    )
    candidate = controller.train(splits["train"])["candidate_bank"][1]
    assert not candidate["enabled"]
    assert candidate["admission"] == "training_native_quality_regression"
