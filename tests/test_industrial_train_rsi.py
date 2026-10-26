"""Synthetic/static TRAIN RSI contracts; no benchmark data or native requests."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phiagent.harness.codex_chisel import ChiselConfig
from phiagent.harness.industrial_train_rsi import (
    CODING_OBJECTIVE, IDENTITY, IndustrialTrainRSIController, quality_cost_gate,
)
from scripts.run_industrial_train_rsi import DurableTrainCallbacks, stage, verify_admission
from scripts.industrial_deepseek_train_adapter import (
    NativeTrainAdapter, admission_draft, procedural_job, sha,
)
from tests.test_codex_chisel import (
    EXECUTOR_IDENTITY, OPTIMIZER_IDENTITY, SyntheticCallbacks, population, usage,
)

ROOT = Path(__file__).resolve().parents[1]


def summary(successes=10, tokens=1000, count=18):
    return {
        "task_ids": [f"synthetic-{index}" for index in range(count)],
        "native_successes": successes, "native_scored": count,
        "all_attempt_actor_control_tokens": tokens, "all_attempt_usage_complete": True,
        "residual_applications": count,
    }


def test_quality_first_and_strict_all_attempt_cost():
    assert quality_cost_gate(summary(), summary(tokens=900))["accepted"]
    assert quality_cost_gate(summary(), summary(successes=11, tokens=1100))["status"] == "QUALITY_COST_TRADEOFF"
    assert quality_cost_gate(summary(), summary(successes=9, tokens=500))["status"] == "QUALITY_REGRESSION"
    assert not quality_cost_gate(summary(), summary(tokens=1000))["accepted"]
    assert not quality_cost_gate(summary(), summary(tokens=None))["accepted"]


def test_native_adapter_changes_only_the_supported_system_instruction_arm():
    job = {"agents": [{"kwargs": {
        "arm": "baseline", "chisel": None, "max_tokens": 32768,
        "runtime": {"model": "synthetic"}, "tools": ["native-shell"],
    }}], "verifier": {"timeout_multiplier": 1.0}}
    original = deepcopy(job)
    request = {
        "phase": "train", "task": {"public_prompt": "Synthetic coding request."},
        "codex_prompt": "Synthetic coding request.",
        "system_prompt_append": "Inspect dependencies before editing.",
        "residual": {"instruction": "Inspect dependencies before editing.",
                     "application": "system_section"},
    }
    descriptor = {"artifact_kind": "training-derived", "sha256": "a" * 64}
    value = procedural_job(job, request, descriptor)
    assert job == original
    assert value["agents"][0]["kwargs"]["arm"] == "chisel"
    assert value["agents"][0]["kwargs"]["max_tokens"] == 32768
    assert value["agents"][0]["kwargs"]["tools"] == ["native-shell"]
    assert value["verifier"] == original["verifier"]
    noop = {**request, "system_prompt_append": "", "residual": None}
    assert procedural_job(job, noop, None) == original
    with pytest.raises(ValueError, match="protected"):
        procedural_job(job, {**request, "phase": "test"}, descriptor)


def test_selected18_cost_is_not_the_all_attempt_comparator():
    gate = quality_cost_gate(
        summary(tokens=43824255), summary(tokens=43500000),
    )
    assert gate["accepted"]
    assert gate["all_attempt_token_delta"] == -324255


def test_all18_identities_are_required_and_noop_cannot_claim_savings():
    partial = summary(count=17)
    assert not quality_cost_gate(summary(), partial)["accepted"]
    changed = summary()
    changed["task_ids"][0] = "another-synthetic"
    assert not quality_cost_gate(summary(), changed)["accepted"]
    duplicate = summary()
    duplicate["task_ids"][0] = duplicate["task_ids"][1]
    with pytest.raises(ValueError, match="duplicate"):
        quality_cost_gate(summary(), duplicate)
    assert not quality_cost_gate(summary(), {**summary(tokens=800), "residual_applications": 0})["accepted"]


def synthetic_controller(*, optimizer=None):
    callbacks = SyntheticCallbacks(score=lambda request: 1.0)
    manifest, index, splits = population(units=89)

    def execute(request):
        result = callbacks.execute(request)
        result["usage"] = usage(10 if request["residual"] else 100, 5)
        return result

    controller = IndustrialTrainRSIController(
        partition=manifest, task_index=index, execute=execute,
        optimize=optimizer or callbacks.optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id="synthetic-industrial-train", config=ChiselConfig(optimization_iterations=1),
    )
    return controller, callbacks, splits


def test_real_engine_reuse_all18_train_and_freeze_without_protected_payloads():
    controller, callbacks, splits = synthetic_controller()
    report = controller.train(splits["train"])
    assert len(callbacks.optimizations[0]["training_evidence"]) == 18
    assert callbacks.optimizations[0]["objective"] == CODING_OBJECTIVE
    assert all(request["phase"] == "train" for request in callbacks.executions)
    assert "VALIDATION_SECRET" not in json.dumps(report)
    candidate = report["candidate_bank"][1]
    assert candidate["enabled"]
    assert candidate["industrial_coding_admission"]["thresholds_lowered"] is False
    frozen = controller.freeze_train_bank(sources={"synthetic": "a" * 64}, shared_contract={})
    assert frozen["status"] == "FROZEN_FOR_SEPARATE_PROTECTED_VALIDATION"
    assert not frozen["component_identity"]["full_cot_jev"]
    assert controller.freeze_train_bank(sources={"synthetic": "a" * 64}, shared_contract={}) == frozen
    with pytest.raises(RuntimeError, match="separate"):
        controller.select_validation(lambda: pytest.fail("Protected loader was called"))
    with pytest.raises(RuntimeError):
        controller.evaluate_test(lambda: pytest.fail("Protected loader was called"))
    routed = controller.evaluate_frozen_train(splits["train"], candidate["id"])
    assert routed["complete_native_scores"]
    assert routed["residual_applications"] == 18
    assert controller.candidate_bank == frozen["candidate_bank"]


@pytest.mark.parametrize("support", [["one"], ["one", "one"], ["one", "protected"]])
def test_invalid_proposal_support_is_rejected_with_paid_usage_retained(support):
    def optimize(request):
        ids = [row["task"]["task_id"] for row in request["training_evidence"]]
        supplied = [ids[0] if item == "one" else item for item in support]
        return {
            "request_id": request["request_id"], "identity": request["identity"],
            "execution_status": "completed", "instruction": "Inspect dependencies before editing.",
            "supporting_task_ids": supplied, "usage": usage(80, 20),
        }

    controller, _, splits = synthetic_controller(optimizer=optimize)
    report = controller.train(splits["train"])
    assert len(report["candidate_bank"]) == 1
    assert report["accounting"]["by_stage"]["training_optimizer"]["usage"]["total_tokens"]["total"] == 100
    assert report["optimization_iterations"][0]["status"] == "optimizer_abstained_or_failed"


def test_declared_runtime_is_not_a_model_selector():
    assert IDENTITY["runtime_selector_model_calls"] == 0
    assert IDENTITY["fast_activation"] is False
    assert "linear" in IDENTITY["runtime_decision_layer"]


def test_callback_guard_precedes_native_dispatch(tmp_path):
    class Adapter:
        def execute(self, request):
            pytest.fail("Protected request reached native runtime")

    callbacks = DurableTrainCallbacks(tmp_path, Adapter(), identity={})
    with pytest.raises(ValueError, match="protected"):
        callbacks.execute({"phase": "validation"})
    assert not (tmp_path / "callbacks").exists()


def test_parent_all_attempt_accounting_is_checked_before_paid_proposal(tmp_path):
    class Adapter:
        def optimize(self, request):
            pytest.fail("Under-accounted parent reached paid proposal")

    callbacks = DurableTrainCallbacks(tmp_path, Adapter(), identity={}, parent_summary=summary())
    callbacks.completed = [
        {"arm": "baseline", "task_id": f"synthetic-{index}",
         "success": index < 10, "usage": usage(8, 2)}
        for index in range(18)
    ]
    with pytest.raises(RuntimeError, match="before paid proposal"):
        callbacks.optimize({"phase": "train", "request_id": "synthetic"})


def test_failed_native_dispatch_is_durable_and_never_retried(tmp_path):
    class Adapter:
        calls = 0

        def execute(self, request):
            self.calls += 1
            raise RuntimeError("Synthetic native settlement blocker")

    adapter = Adapter()
    callbacks = DurableTrainCallbacks(tmp_path, adapter, identity={})
    request = {"phase": "train", "request_id": "synthetic", "identity": {}}
    with pytest.raises(RuntimeError, match="settlement"):
        callbacks.execute(request)
    with pytest.raises(RuntimeError, match="infrastructure owner"):
        callbacks.execute(request)
    assert adapter.calls == 1
    error = json.loads((tmp_path / "callbacks/executor/synthetic/ERROR.json").read_text())
    assert error["native_receipts_preserved"]


def test_owner_gate_blocks_before_any_training_payload_read():
    settings = json.loads((ROOT / "configs/industrial_terminal_train_rsi_v1.json").read_text())
    with pytest.raises(RuntimeError, match="native owner"):
        verify_admission({}, settings, "deepseek")


def test_paid_train_proposal_inherits_cost_without_another_provider_request(tmp_path):
    payload = {"objective": "Synthetic reusable coding procedure", "training_evidence": []}
    identity = {"model": "synthetic", "runtime": "synthetic-native-sdk"}
    prior = {
        "request_id": "previous", "identity": identity, "execution_status": "completed",
        "instruction": "Inspect dependencies before editing.",
        "supporting_task_ids": ["synthetic-a", "synthetic-b"],
        "usage": usage(80, 20), "provenance": {"synthetic": True},
    }
    result_path, payload_path = tmp_path / "prior-result.json", tmp_path / "prior-payload.json"
    result_path.write_text(json.dumps(prior))
    payload_path.write_text(json.dumps(payload))
    adapter = object.__new__(NativeTrainAdapter)
    adapter.run = tmp_path / "fresh-run"
    adapter.admission = {"inherited_train_proposer": {
        "result": str(result_path), "result_sha256": sha(result_path),
        "payload": str(payload_path), "payload_sha256": sha(payload_path),
    }}

    def prepare(operation, directory, request_path):
        assert operation == "prepare-optimizer"
        (directory / "PROVIDER_PROMPT.json").write_text(json.dumps(payload))

    adapter.bridge = prepare
    adapter.process = lambda *args: pytest.fail("Paid proposal was dispatched twice")
    result = adapter.optimize({"request_id": "new", "identity": identity, "phase": "train"})
    assert result["request_id"] == "new"
    assert result["usage"]["total_tokens"] == 100
    assert result["provenance"]["inherited_train_proposal"]["new_provider_requests"] == 0
    assert json.loads(result_path.read_text()) == prior


@pytest.mark.parametrize("harness", ["codex", "deepseek"])
def test_owner_released_contract_uses_same_backbone_and_complete_parent(harness, tmp_path):
    settings = json.loads((ROOT / "configs/industrial_terminal_train_rsi_v1.json").read_text())
    runtime = "official_codex_cli" if harness == "codex" else "official_deepseek_harness_sdk"
    wire = "responses" if harness == "codex" else "anthropic_messages"
    contract = {
        "model": "deepseek-flash", "runtime": runtime, "wire_api": wire,
        "shared_prompt_version": settings["shared_prompt_version"],
        "shared_prompt_sha256": settings["shared_prompt_sha256"],
        "native_output_time_tool_contracts_unchanged": True,
    }
    admission = {
        "schema": "industrial-terminal-train-admission/v1", "status": "READY",
        "harness": harness, "phase": "train", "validation_or_test_read": False,
        "exclusive_successor_owner": settings["owner"], "native_owner_handoff": True,
        "parent": settings["parents"][harness], "shared_contract": contract,
        "all_attempt_actor_control_usage": True, "official_native_verifier": True,
        "train_payload_count": 18, "callback_factory": "synthetic:factory",
        "source_sha256": {"synthetic.py": "a" * 64},
        "source_root": "/synthetic/source", "train_input_sha256": "b" * 64,
        "execution_identity": {key: contract[key] for key in (
            "model", "runtime", "wire_api", "shared_prompt_sha256",
        )},
        "optimizer_identity": {"model": "deepseek-flash"},
    }
    with pytest.raises(RuntimeError, match="Capability READY"):
        verify_admission(admission, settings, harness)
    release = {
        "schema": "industrial-native-callback-owner-approval/v1", "status": "READY",
        "scope": "exclusive_industrial_train_callback_factory",
        "issued_by": "synthetic-native-callback-owner",
        "exclusive_successor_owner": settings["owner"], "harness": harness, "phase": "train",
        **{field: admission[field] for field in (
            "callback_factory", "source_root", "source_sha256", "train_input_sha256",
            "parent", "shared_contract", "execution_identity", "optimizer_identity",
        )},
    }
    approval_path = tmp_path / "callback-owner-approval.json"
    approval_path.write_text(json.dumps(release))
    admission["callback_owner_approval"] = {"path": str(approval_path), "sha256": sha(approval_path)}
    assert verify_admission(admission, settings, harness) == contract
    invalid = deepcopy(admission)
    invalid["execution_identity"]["model"] = "another-model"
    with pytest.raises(ValueError, match="exact frozen"):
        verify_admission(invalid, settings, harness)
    invalid = deepcopy(admission)
    invalid["all_attempt_actor_control_usage"] = False
    with pytest.raises(RuntimeError, match="all-attempt"):
        verify_admission(invalid, settings, harness)
    invalid = deepcopy(admission)
    invalid["callback_factory"] = "another:factory"
    with pytest.raises(ValueError, match="exact sources"):
        verify_admission(invalid, settings, harness)
    release["issued_by"] = settings["owner"]
    approval_path.write_text(json.dumps(release))
    admission["callback_owner_approval"]["sha256"] = sha(approval_path)
    with pytest.raises(RuntimeError, match="not self-release"):
        verify_admission(admission, settings, harness)


def test_capability_preparation_can_only_create_a_non_releasing_draft():
    original = {"status": "READY", "native_owner_handoff": True, "parent": {"native_scored": 18}}
    draft = admission_draft(original)
    assert draft["status"] == "AWAITING_EXCLUSIVE_CALLBACK_OWNER_APPROVAL"
    assert draft["native_owner_handoff"] is False
    assert draft["callback_owner_approval"] is None
    assert original["status"] == "READY"


def test_committed_callback_reuse_validates_receipt_hash_and_retains_cost(tmp_path):
    class Adapter:
        calls = 0

        def execute(self, request):
            self.calls += 1
            return {
                "request_id": request["request_id"], "identity": request["identity"],
                "task_id": "synthetic", "execution_status": "completed",
                "native_score": 1, "success": True, "usage": usage(8, 2),
            }

    adapter = Adapter()
    request = {"phase": "train", "request_id": "synthetic", "identity": {},
               "task": {"task_id": "synthetic"}, "arm": "candidate"}
    callbacks = DurableTrainCallbacks(tmp_path, adapter, identity={})
    result = callbacks.execute(request)
    restarted = DurableTrainCallbacks(tmp_path, adapter, identity={})
    assert restarted.execute(request) == result
    assert adapter.calls == 1
    assert restarted.completed[0]["usage"]["total_tokens"] == 10
    result_path = tmp_path / "callbacks/executor/synthetic/RESULT.json"
    result_path.write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        DurableTrainCallbacks(tmp_path, adapter, identity={}).execute(request)


def test_staging_freezes_new_source_config_and_exact_blocker(tmp_path):
    config = ROOT / "configs/industrial_terminal_train_rsi_v1.json"
    run = tmp_path / "new-industrial-train"
    stage(run, config, harness="deepseek")
    state = json.loads((run / "STATE.json").read_text())
    assert state["pid"] is None
    assert state["new_native_requests"] == 0
    assert state["status"] == "STAGED_AWAITING_NATIVE_OWNER_ADMISSION"
    assert (run / "source/phiagent/harness/industrial_train_rsi.py").is_file()
    original = deepcopy(state)
    with pytest.raises(ValueError, match="fresh"):
        stage(run, config, harness="deepseek")
    assert json.loads((run / "STATE.json").read_text()) == original
