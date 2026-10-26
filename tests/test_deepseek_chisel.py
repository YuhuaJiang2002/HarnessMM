"""Synthetic lifecycle checks; no native data, external models or evaluators."""

import hashlib

import pytest

from phiagent.harness.benchmark_partition import build_partition, canonical_sha256
from phiagent.harness.deepseek_chisel import (
    ChiselConfig, DeepSeekChiselController, RUNTIME, adapt_request,
)


USAGE = {
    "input_tokens": 2, "output_tokens": 1, "cache_read_tokens": 0,
    "cache_write_tokens": 0, "total_tokens": 3,
}


def test_executor_translation_preserves_residual_and_original_request():
    source = {
        "schema": "shared-controller", "codex_prompt": "Public task\n\nlearned residual",
        "agents_md_append": "", "contract": {"runtime": "unmodified_official_codex_cli"},
        "residual": {"instruction": "learned residual"}, "request_id": "synthetic",
    }
    actual = adapt_request(source, optimizer=False)
    assert actual["prompt"] == source["codex_prompt"]
    assert actual["residual"] == source["residual"]
    assert actual["contract"]["runtime"] == "unmodified_official_deepseek_harness_sdk"
    assert "codex_prompt" not in actual
    assert "codex_prompt" in source
    assert source["contract"]["runtime"] == "unmodified_official_codex_cli"
    with pytest.raises(ValueError, match="filesystem"):
        adapt_request({**source, "agents_md_append": "must not inject"}, optimizer=False)


def test_optimizer_names_the_real_harness_without_changing_training_evidence():
    evidence = [{"task_id": "synthetic", "native_score": 0.6}]
    value = adapt_request({
        "schema": "shared", "training_evidence": evidence,
        "objective": "Compile one residual for the unchanged official Codex CLI. Preserve Codex core.",
    }, optimizer=True)
    assert "Codex" not in value["objective"]
    assert "official DeepSeek Harness SDK" in value["objective"]
    assert value["training_evidence"] == evidence
    with pytest.raises(ValueError, match="objective changed"):
        adapt_request({"objective": "unknown"}, optimizer=True)


def test_runtime_neutral_shared_objective_requires_explicit_deepseek_identity():
    objective = (
        "Compile ONE short, reusable residual instruction for the unchanged official "
        "industrial harness identified in this request. Preserve native tools."
    )
    request = {"objective": objective, "identity": {"runtime": RUNTIME}}
    assert "official DeepSeek Harness SDK" in adapt_request(request, optimizer=True)["objective"]
    assert "official DeepSeek Harness SDK" in adapt_request(
        {**request, "identity": {"model": "separate-optimizer"}},
        optimizer=True, execution_identity={"runtime": RUNTIME},
    )["objective"]
    with pytest.raises(ValueError, match="objective changed"):
        adapt_request({**request, "identity": {"runtime": "unregistered"}}, optimizer=True)


@pytest.mark.parametrize("proposal", ["", "Check the supplied constraints before submitting."])
@pytest.mark.parametrize("application", ["prompt_suffix", "system_section"])
def test_shared_train_selection_and_test_firewalls_remain_intact(tmp_path, proposal, application):
    tasks = [
        {"task_id": f"synthetic-{i}", "unit_id": f"synthetic-{i}",
         "public_prompt": f"Synthetic public instruction {i}"}
        for i in range(10)
    ]
    records = [
        {"id": task["unit_id"], "payload_sha256": canonical_sha256(task),
         "public_task_sha256": canonical_sha256(task["public_prompt"])}
        for task in tasks
    ]
    partition = build_partition(records, benchmark="synthetic", seed=20260928)
    index = [
        {"task_id": task["task_id"], "unit_id": task["unit_id"],
         "public_prompt_sha256": hashlib.sha256(task["public_prompt"].encode()).hexdigest()}
        for task in tasks
    ]
    seen = []

    def execute(request):
        seen.append(request)
        assert request["contract"]["runtime"] == "unmodified_official_deepseek_harness_sdk"
        assert request["contract"]["context_application"] == application
        assert "codex_prompt" not in request
        if application == "system_section":
            assert request["prompt"] == request["task"]["public_prompt"]
            assert request["system_prompt_append"] == (
                request["residual"]["instruction"] if request["residual"] else ""
            )
        return {
            "request_id": request["request_id"], "task_id": request["task"]["task_id"],
            "identity": request["identity"], "execution_status": "completed",
            "native_score": 0.9 if request["residual"] else 0.6,
            "success": None, "usage": USAGE,
        }

    def optimize(request):
        assert "Codex" not in request["objective"]
        assert {row["task"]["task_id"] for row in request["training_evidence"]} == set(
            partition["splits"]["train"]
        )
        return {
            "request_id": request["request_id"], "identity": request["identity"],
            "execution_status": "completed", "instruction": proposal,
            "supporting_task_ids": sorted(partition["splits"]["train"]) if proposal else [],
            "usage": USAGE,
        }

    controller = DeepSeekChiselController(
        partition=partition, task_index=index, execute=execute, optimize=optimize,
        execution_identity={"model": "synthetic-model", "runtime": RUNTIME,
                            "wire_api": "anthropic_messages"},
        optimizer_identity={"model": "synthetic-model"}, run_id="synthetic-run",
        config=ChiselConfig(optimization_iterations=1, application=application),
    )

    def split(name):
        return [task for task in tasks if task["unit_id"] in partition["splits"][name]]

    with pytest.raises(RuntimeError):
        controller.evaluate_test(lambda: pytest.fail("sealed test loader was called"),
                                 policy_sha256="0" * 64, source_sha256="0" * 64)
    controller.train(split("train"))
    controller.select_validation(split("validation"))
    source = tmp_path / "synthetic-source.txt"
    source.write_text("synthetic adapter source")
    policy = controller.freeze(source_hashes={
        name: hashlib.sha256(source.read_bytes()).hexdigest()
        for name in ("deepseek_harness", "executor", "optimizer", "scorer")
    })
    assert policy.to_dict()["method"] == "CHISEL+official-DeepSeek-Harness"
    assert policy.to_dict()["candidate"]["enabled"] is bool(proposal)
    controller.evaluate_test(
        lambda: split("test"),
        policy_sha256=policy.policy_sha256, source_sha256=policy.source_sha256,
    )
    assert controller.state == "complete"
    assert len([request for request in seen if request["phase"] == "test"]) == 12
    assert controller.report()["execution_identity"]["runtime"] == RUNTIME
