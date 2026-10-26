"""Synthetic callback contracts only: no provider, dataset, benchmark or evaluator."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import json
from threading import Barrier, Lock

import pytest

from phiagent.harness.benchmark_partition import build_partition
from phiagent.harness import codex_chisel as chisel
from phiagent.learning.residual_meta_harness import canonical_hash, deterministic_group_folds


EXECUTOR_IDENTITY = {"model": "synthetic-model", "runtime": "synthetic-codex-callback"}
OPTIMIZER_IDENTITY = {"model": "synthetic-optimizer", "transport": "in-memory-fixture"}
INSTRUCTION = "Verify the requested count against the listed objects before submitting."
OTHER_INSTRUCTION = "Check the listed objects once and verify the count before submitting."
SOURCE_HASHES = {
    key: hashlib.sha256(("synthetic-source-" + key).encode()).hexdigest()
    for key in ("codex_cli", "executor", "optimizer", "scorer")
}


def usage(input_tokens=10, output_tokens=5, *, cache=4, cost=0.015):
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cache_read_tokens": cache, "cache_write_tokens": 0,
        "total_tokens": input_tokens + output_tokens, "cost_usd": cost,
    }


def population(units=25, questions_per_unit=1, prompt=None):
    tasks, records = [], []
    for index in range(units):
        unit = f"synthetic-unit-{index:04d}"
        members = []
        for question in range(questions_per_unit):
            task = {
                "task_id": f"synthetic-task-{index:04d}-{question}",
                "unit_id": unit,
                "public_prompt": (
                    prompt(index, question) if prompt else "Synthetic task: count the blue objects."
                ),
            }
            members.append(task)
            tasks.append(task)
        records.append({
            "id": unit, "payload_sha256": canonical_hash({"synthetic": unit}),
            "public_task_sha256": canonical_hash(members),
        })
    manifest = build_partition(
        records, benchmark="synthetic-codex-contract", seed=chisel.SPLIT_SEED,
        independent_unit="image" if questions_per_unit > 1 else "task",
        rounding="exact" if units % 5 == 0 else "largest_remainder",
    )
    index = [{
        "task_id": task["task_id"], "unit_id": task["unit_id"],
        "public_prompt_sha256": hashlib.sha256(task["public_prompt"].encode()).hexdigest(),
    } for task in tasks]
    splits = {
        split: [task for task in tasks if task["unit_id"] in manifest["splits"][split]]
        for split in ("train", "validation", "test")
    }
    return manifest, index, splits


class SyntheticCallbacks:
    def __init__(self, *, proposals=None, score=None):
        self.proposals = proposals if proposals is not None else [INSTRUCTION, INSTRUCTION]
        self.score = score or (lambda request: 0.75 if request["residual"] else 0.25)
        self.executions = []
        self.optimizations = []

    def execute(self, request):
        self.executions.append(deepcopy(request))
        score = self.score(request)
        return {
            "request_id": request["request_id"], "task_id": request["task"]["task_id"],
            "identity": deepcopy(EXECUTOR_IDENTITY),
            "execution_status": "completed" if score is not None else "failed",
            "native_score": score, "success": score >= 0.5 if score is not None else None,
            "usage": usage(12 if request["residual"] else 10, 5),
            "feedback": {
                "train": "TRAIN_ONLY_DIAGNOSTIC",
                "validation": "VALIDATION_SECRET_DIAGNOSTIC",
                "test": "SEALED_TEST_DIAGNOSTIC",
            }[request["phase"]],
            "provenance": {"synthetic": True},
        }

    def optimize(self, request):
        self.optimizations.append(deepcopy(request))
        instruction = self.proposals[request["iteration"] % len(self.proposals)]
        return {
            "request_id": request["request_id"], "identity": deepcopy(OPTIMIZER_IDENTITY),
            "execution_status": "completed", "instruction": instruction,
            "supporting_task_ids": (
                sorted({row["task"]["task_id"] for row in request["training_evidence"]})
                if instruction else []
            ),
            "usage": usage(100, 20, cache=30, cost=0.12),
            "provenance": {"synthetic": True},
        }


def controller(
    callbacks=None, *, data=None, config=None, execute=None, optimize=None,
    run_id="synthetic-unit-test-run",
):
    callbacks = callbacks or SyntheticCallbacks()
    manifest, index, splits = data or population()
    instance = chisel.CodexChiselController(
        partition=manifest, task_index=index, execute=execute or callbacks.execute,
        optimize=optimize or callbacks.optimize,
        execution_identity=EXECUTOR_IDENTITY, optimizer_identity=OPTIMIZER_IDENTITY,
        run_id=run_id, config=config or chisel.ChiselConfig(),
    )
    return instance, callbacks, splits


def fitted(callbacks=None, **kwargs):
    instance, callbacks, splits = controller(callbacks, **kwargs)
    instance.train(splits["train"])
    instance.select_validation(splits["validation"])
    policy = instance.freeze(source_hashes=SOURCE_HASHES)
    return instance, callbacks, splits, policy


def test_completed_train_report_restores_without_replaying_paid_calls():
    trained, callbacks, splits = controller()
    report = trained.train(splits["train"])
    restored, restored_callbacks, _ = controller(
        data=population(), run_id=report["run_id"],
    )
    restored.restore_trained(report)
    assert restored.report() == report
    selection = restored.select_validation(splits["validation"])
    assert selection["test_opened"] is False
    assert restored_callbacks.optimizations == []
    eligible = sum(row["id"] == "noop" or row["enabled"] for row in report["candidate_bank"])
    assert len(restored_callbacks.executions) == eligible * len(splits["validation"])
    assert len(callbacks.executions) == report["accounting"]["by_stage"]["training_executor"]["calls"]


@pytest.mark.parametrize("field", [
    "candidate", "observation", "ledger", "accounting", "provenance", "run_id",
])
def test_completed_train_restore_rejects_changed_evidence(field):
    trained, _, splits = controller()
    report = trained.train(splits["train"])
    changed = deepcopy(report)
    if field == "candidate":
        changed["candidate_bank"][1]["instruction"] += " changed"
    elif field == "observation":
        next(iter(changed["observations"].values()))["native_score"] = 0.123
    elif field == "ledger":
        changed["ledger"][0]["sequence"] = 2
    elif field == "accounting":
        changed["accounting"]["all_calls"]["calls"] += 1
    elif field == "provenance":
        changed["implementation_provenance"]["implementation"] = "changed"
    else:
        changed["run_id"] = "changed-run"
    restored, _, _ = controller(data=population(), run_id=report["run_id"])
    with pytest.raises(ValueError):
        restored.restore_trained(changed)


def test_native_responses_identity_is_preserved_without_provider_translation():
    callbacks = SyntheticCallbacks()
    manifest, index, splits = population()
    native_identity = {
        "model": "deepseek-flash", "runtime": "official_codex_cli",
        "wire_api": "responses", "transport": "byte_preserving_metering_proxy",
        "upstream": "https://api.deepseek.com/responses",
        "runtime_revision": "synthetic-pinned-revision",
    }

    def synthetic_native_executor(request):
        result = callbacks.execute(request)
        result["identity"] = deepcopy(native_identity)
        return result

    instance = chisel.CodexChiselController(
        partition=manifest, task_index=index, execute=synthetic_native_executor,
        optimize=callbacks.optimize, execution_identity=native_identity,
        optimizer_identity=OPTIMIZER_IDENTITY, run_id="synthetic-native-identity-contract",
    )
    instance.train(splits["train"])
    instance.select_validation(splits["validation"])
    policy = instance.freeze(source_hashes={
        **SOURCE_HASHES, "metering_proxy": canonical_hash("synthetic-byte-preserving-proxy"),
    })
    assert policy.to_dict()["sources"]["execution_identity"] == native_identity
    for request in callbacks.executions:
        assert request["identity"] == native_identity
        assert len(request["request_id"]) == 64
        assert request["codex_prompt"].startswith(request["task"]["public_prompt"])
        assert "api.deepseek.com" not in request["codex_prompt"]
        assert "wire_api" not in request["codex_prompt"]
        assert "Authorization" not in request["codex_prompt"]
        if request["residual"] is None:
            assert request["codex_prompt"] == request["task"]["public_prompt"]
    assert all(result["provenance"]["synthetic"]
               for result in instance.report()["observations"].values())


def common_receipt(*, status="COMPLETED", usage_status="MEASURED"):
    return {
        "status": status, "final_answer": "Synthetic final answer.",
        "model": EXECUTOR_IDENTITY["model"], "provider_sampling_seed_supported": False,
        "provider_usage": {
            "usage_status": usage_status,
            "known_usage": {"input_tokens": 36, "output_tokens": 17, "total_tokens": 53},
            "cached_input_tokens": 12, "reasoning_output_tokens": 5,
            "unresolved_call_ids": [] if usage_status == "MEASURED" else ["synthetic-call-2"],
        },
    }


def receipt_request(phase="train"):
    return {
        "request_id": "synthetic-common-receipt-request", "identity": EXECUTOR_IDENTITY,
        "task": {"task_id": "synthetic-task"}, "phase": phase,
    }


@pytest.mark.parametrize("score", [0.0, 0.3, 1.0])
def test_common_receipt_preserves_native_fractional_scores_and_measured_usage(score):
    receipt = common_receipt()
    result = chisel.execution_result_from_receipt(
        receipt_request(), receipt, native_score=score,
    )
    assert result["execution_status"] == "completed"
    assert result["native_score"] == score
    assert result["success"] is None
    assert result["usage"] == {
        "input_tokens": 36, "output_tokens": 17, "total_tokens": 53,
        "cache_read_tokens": 12, "cache_write_tokens": None, "cost_usd": None,
    }
    assert result["feedback"] == receipt["final_answer"]
    assert result["provenance"]["provider_sampling_seed_supported"] is False
    assert result["provenance"]["provider_usage"]["reasoning_output_tokens"] == 5
    assert result["provenance"]["common_executor_receipt_sha256"] == canonical_hash(receipt)


@pytest.mark.parametrize("usage_status", ["MEASURED", "UNRESOLVED"])
def test_blocked_common_executor_never_becomes_zero_score_success_or_tokens(usage_status):
    result = chisel.execution_result_from_receipt(
        receipt_request(), common_receipt(status="BLOCKED", usage_status=usage_status),
        native_score=None,
    )
    assert result["execution_status"] == "blocked"
    assert result["native_score"] is None
    assert result["success"] is None
    if usage_status == "MEASURED":
        assert result["usage"]["total_tokens"] == 53
    else:
        assert all(value is None for value in result["usage"].values())
        assert result["provenance"]["provider_usage"]["known_usage"]["total_tokens"] == 53


def test_common_receipt_adapter_does_not_export_heldout_answers_as_optimizer_feedback():
    for phase in ("validation", "test"):
        result = chisel.execution_result_from_receipt(
            receipt_request(phase), common_receipt(), native_score=0.3,
            feedback="Held-out private diagnostics.",
        )
        assert "feedback" not in result
        assert "Synthetic final answer." not in json.dumps(result)
        assert "Held-out private diagnostics." not in json.dumps(result)
    with pytest.raises(ValueError, match="fractional"):
        chisel.execution_result_from_receipt(receipt_request(), common_receipt(), native_score=30)
    with pytest.raises(ValueError, match="explicit native score"):
        chisel.execution_result_from_receipt(
            receipt_request(), common_receipt(status="SCORED"), native_score=None,
        )
    with pytest.raises(ValueError, match="terminal status"):
        chisel.execution_result_from_receipt(
            receipt_request(), common_receipt(status="RUNNING"), native_score=None,
        )


def test_common_receipts_work_through_the_existing_controller_callback_contract():
    data = population(units=89)
    callbacks = SyntheticCallbacks(proposals=[""])

    def common_executor(request):
        return chisel.execution_result_from_receipt(
            request, common_receipt(), native_score=0.3,
        )

    instance, _, splits = controller(callbacks, data=data, execute=common_executor)
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [18, 18, 53]
    instance.train(splits["train"])
    assert instance.accounting()["available_training_tasks"] == 18
    assert instance.accounting()["optimizer_sample_tasks"] == 18
    assert instance.accounting()["by_stage"]["training_executor"]["usage"]["total_tokens"][
        "total"
    ] == 18 * 53


@pytest.mark.parametrize("repetitions", [1, 3, 5])
def test_distinct_test_seeds_keep_complete_pairs_inside_their_split_even_for_noop(repetitions):
    seeds = tuple(900 + index for index in range(repetitions))
    instance, callbacks, splits, policy = fitted(
        SyntheticCallbacks(proposals=[""]),
        config=chisel.ChiselConfig(test_execution_seeds=seeds, executor_workers=4),
    )
    frozen = policy.to_dict()
    assert frozen["config"]["execution_seeds"] == [chisel.SPLIT_SEED]
    assert frozen["config"]["test_execution_seeds"] == list(seeds)
    assert frozen["training_repetitions"] == frozen["validation_repetitions"] == 1
    assert frozen["test_repetitions"] == repetitions
    assert frozen["candidate"]["id"] == "noop"
    assert chisel.FrozenPolicy.from_dict(frozen) == policy
    result = instance.evaluate_test(
        lambda: splits["test"], policy_sha256=policy.policy_sha256,
        source_sha256=policy.source_sha256,
    )
    for phase in ("train", "validation", "test"):
        requests = [row for row in callbacks.executions if row["phase"] == phase]
        assert {row["task"]["unit_id"] for row in requests} == {
            row["unit_id"] for row in splits[phase]
        }
        assert {row["seed"] for row in requests} == (
            set(seeds) if phase == "test" else {chisel.SPLIT_SEED}
        )
        assert all(row["residual"] is None for row in requests)
    test_requests = [row for row in callbacks.executions if row["phase"] == "test"]
    assert len(test_requests) == 2 * len(splits["test"]) * repetitions
    assert len({row["request_id"] for row in test_requests}) == len(test_requests)
    for arm in ("baseline", "chisel"):
        assert result[arm]["observations"] == len(splits["test"]) * repetitions
        assert result[arm]["repetitions"] == repetitions
        assert result[arm]["execution_seeds"] == list(seeds)
        assert result[arm]["residual_applications"] == 0
    for pair in result["pair_orders"]:
        observed = [row["arm"] for row in test_requests if row["pair_id"] == pair["pair_id"]]
        digest = hashlib.sha256(json.dumps(
            ["synthetic-unit-test-run", pair["task_id"], pair["seed"]],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).digest()
        expected = ["baseline", "chisel"] if digest[0] % 2 == 0 else ["chisel", "baseline"]
        assert pair["planned_order"] == pair["actual_order"] == observed == expected
    assert {tuple(row["actual_order"]) for row in result["pair_orders"]} == {
        ("baseline", "chisel"), ("chisel", "baseline"),
    }
    accounting = instance.accounting()
    assert accounting["training_repetitions"] == accounting["validation_repetitions"] == 1
    assert accounting["test_repetitions"] == repetitions
    assert accounting["by_stage"]["final_baseline"]["calls"] == len(splits["test"]) * repetitions
    assert accounting["by_stage"]["final_chisel"]["calls"] == len(splits["test"]) * repetitions


def test_test_seed_default_resolves_and_repetition_metadata_is_validated():
    config = chisel.ChiselConfig(execution_seeds=(17, 29))
    assert config.test_execution_seeds == (17, 29)
    instance, _, splits, policy = fitted(SyntheticCallbacks(proposals=[""]), config=config)
    assert policy.to_dict()["config"]["test_execution_seeds"] == [17, 29]
    assert instance.accounting()["test_repetitions"] == 2
    result = instance.evaluate_test(
        lambda: splits["test"], policy_sha256=policy.policy_sha256,
        source_sha256=policy.source_sha256,
    )
    assert result["baseline"]["observations"] == len(splits["test"]) * 2
    modified = policy.to_dict()
    modified["test_repetitions"] = 99
    modified.pop("policy_sha256")
    modified["policy_sha256"] = canonical_hash(modified)
    with pytest.raises(ValueError, match="repetition counts"):
        chisel.FrozenPolicy.from_dict(modified)


@pytest.mark.parametrize("seeds", [(), (1, 1), (True,), [1], ("1",), 1])
def test_invalid_test_seeds_are_rejected(seeds):
    with pytest.raises(ValueError, match="test_execution_seeds"):
        chisel.ChiselConfig(test_execution_seeds=seeds)


@pytest.mark.parametrize("workers", [0, -1, True, 1.0, None, chisel.MAX_EXECUTOR_WORKERS + 1])
def test_executor_worker_bound_is_predeclared_and_validated(workers):
    with pytest.raises(ValueError, match="executor_workers"):
        chisel.ChiselConfig(executor_workers=workers)


class ConcurrentCallbacks(SyntheticCallbacks):
    def __init__(self, *, failing_task=None):
        super().__init__(proposals=[INSTRUCTION, OTHER_INSTRUCTION])
        self.lock = Lock()
        self.barrier = Barrier(4)
        self.active = 0
        self.peak_active = 0
        self.started = []
        self.baseline_starts = 0
        self.failing_task = failing_task
        self.owner = None

    def execute(self, request):
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.started.append(deepcopy(request))
            ticket = None
            if request["phase"] == "train" and request["arm"] == "baseline":
                ticket = self.baseline_starts
                self.baseline_starts += 1
        try:
            if ticket is not None and ticket < 4:
                self.barrier.wait(timeout=10)
            if self.owner is not None:
                snapshot = self.owner.report()
                assert len({row["request_id"] for row in snapshot["ledger"]}) == len(
                    snapshot["ledger"],
                )
                json.dumps(snapshot, allow_nan=False)
            if request["task"]["task_id"] == self.failing_task:
                raise OSError("synthetic concurrent executor failure")
            if request["phase"] != "train":
                assert len(self.optimizations) == 2
            return super().execute(request)
        finally:
            with self.lock:
                self.active -= 1

    def optimize(self, request):
        with self.lock:
            assert self.active == 0, "an optimizer iteration overlapped an executor batch"
        return super().optimize(request)


def test_workers_one_and_four_have_identical_learned_policy_selection_and_accounting():
    seeds = (71, 73, 79)
    serial, _, splits, serial_policy = fitted(
        SyntheticCallbacks(proposals=[INSTRUCTION, OTHER_INSTRUCTION]),
        config=chisel.ChiselConfig(test_execution_seeds=seeds, executor_workers=1),
    )
    callbacks = ConcurrentCallbacks()
    parallel, _, parallel_splits = controller(
        callbacks, config=chisel.ChiselConfig(test_execution_seeds=seeds, executor_workers=4),
    )
    callbacks.owner = parallel
    parallel.train(parallel_splits["train"])
    parallel.select_validation(parallel_splits["validation"])
    parallel_policy = parallel.freeze(source_hashes=SOURCE_HASHES)
    assert serial.candidate_bank == parallel.candidate_bank
    assert serial.report()["selection"] == parallel.report()["selection"]
    assert serial_policy.to_dict()["candidate"] == parallel_policy.to_dict()["candidate"]
    assert serial_policy.to_dict()["training_evidence_sha256"] == (
        parallel_policy.to_dict()["training_evidence_sha256"]
    )
    results = []
    for instance, policy in ((serial, serial_policy), (parallel, parallel_policy)):
        results.append(instance.evaluate_test(
            lambda: splits["test"], policy_sha256=policy.policy_sha256,
            source_sha256=policy.source_sha256,
        ))
    assert callbacks.peak_active == 4 and callbacks.active == 0
    assert serial.accounting() == parallel.accounting()
    assert results[0]["pair_orders"] == results[1]["pair_orders"]
    for arm in ("baseline", "chisel"):
        first, second = deepcopy(results[0][arm]), deepcopy(results[1][arm])
        for value in (first, second):
            for row in value["rows"]:
                row["result"].pop("request_id")
        assert first == second
    for instance in (serial, parallel):
        ledger = instance.report()["ledger"]
        assert [row["sequence"] for row in ledger] == list(range(len(ledger)))
        assert len({row["request_id"] for row in ledger}) == len(ledger)
    assert serial_policy.policy_sha256 != parallel_policy.policy_sha256
    assert parallel.report()["executor_workers"] == 4


def test_parallel_failure_drains_inflight_calls_and_preserves_all_incurred_accounting():
    data = population()
    failing_task = min(task["task_id"] for task in data[2]["train"])
    callbacks = ConcurrentCallbacks(failing_task=failing_task)
    instance, _, splits = controller(callbacks, data=data, config=chisel.ChiselConfig(executor_workers=4))
    callbacks.owner = instance
    with pytest.raises(OSError, match="concurrent executor failure"):
        instance.train(splits["train"])
    assert callbacks.peak_active == 4 and callbacks.active == 0
    assert not callbacks.optimizations
    assert instance.state == "blocked"
    report = instance.report()
    assert len(report["ledger"]) == len(callbacks.started)
    assert {row["request_id"] for row in report["ledger"]} == {
        row["request_id"] for row in callbacks.started
    }
    assert len(report["executor_failures"]) == 1
    assert report["executor_failures"][0]["task_id"] == failing_task
    assert sum(row["execution_status"] == "callback_error" for row in report["ledger"]) == 1
    tokens = report["accounting"]["all_calls"]["usage"]["total_tokens"]
    assert tokens["total"] is None and tokens["missing_calls"] == 1
    assert tokens["measured_subtotal"] == (len(callbacks.started) - 1) * 15
    assert len(report["observations"]) == len(callbacks.started) - 1
    assert all(row["phase"] == "train" for row in callbacks.started)
    with pytest.raises(RuntimeError):
        instance.select_validation(splits["validation"])


def test_concurrent_duplicate_request_registration_cannot_charge_twice():
    instance, callbacks, splits = controller()
    request = {
        "run_id": "synthetic-duplicate-request", "phase": "train", "arm": "baseline",
        "task": splits["train"][0], "seed": chisel.SPLIT_SEED,
        "identity": EXECUTOR_IDENTITY, "residual": None,
    }
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(instance._invoke, callbacks.execute, deepcopy(request), "executor")
            for _ in range(2)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except RuntimeError as error:
                errors.append(str(error))
    assert len(results) == len(errors) == 1
    assert "duplicate callback request" in errors[0]
    assert len(callbacks.executions) == 1
    assert instance.accounting()["all_calls"]["calls"] == 1
    assert instance.accounting()["all_calls"]["usage"]["total_tokens"]["total"] == 15


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2"])
def test_optimizer_unit_cap_requires_a_positive_integer_or_none(cap):
    with pytest.raises(ValueError, match="optimizer_max_units"):
        chisel.ChiselConfig(optimizer_max_units=cap)


def expected_optimizer_units(manifest, iteration, cap, run_id="synthetic-unit-test-run"):
    return sorted(manifest["splits"]["train"], key=lambda unit: (
        canonical_hash([run_id, chisel.SPLIT_SEED, iteration, unit]), unit,
    ))[:cap]


def test_optimizer_sampling_is_outcome_independent_stable_and_preserves_whole_units():
    data = population(units=50, questions_per_unit=3)
    config = chisel.ChiselConfig(
        optimizer_max_units=2, execution_seeds=(11, 19), executor_workers=4,
    )
    good = SyntheticCallbacks(proposals=[INSTRUCTION, OTHER_INSTRUCTION])
    first, _, splits = controller(good, data=data, config=config)
    first.train(splits["train"])
    shuffled = (data[0], list(reversed(data[1])), {
        phase: list(reversed(tasks)) for phase, tasks in splits.items()
    })
    bad = SyntheticCallbacks(
        proposals=[INSTRUCTION, OTHER_INSTRUCTION],
        score=lambda request: 0.1 if request["residual"] else 0.9,
    )
    second, _, second_splits = controller(bad, data=shuffled, config=config)
    second.train(second_splits["train"])
    seen_tasks, seen_units = set(), set()
    for iteration, (left, right) in enumerate(zip(good.optimizations, bad.optimizations, strict=True)):
        expected_units = set(expected_optimizer_units(data[0], iteration, 2))
        expected_tasks = {
            task["task_id"] for task in splits["train"] if task["unit_id"] in expected_units
        }
        expected_rows = {(task_id, seed) for task_id in expected_tasks for seed in (11, 19)}
        for request in (left, right):
            rows = request["training_evidence"]
            assert request["limits"]["optimizer_max_units"] == 2
            assert len(rows) == 12
            assert {(row["task"]["task_id"], row["seed"]) for row in rows} == expected_rows
            assert {row["task"]["unit_id"] for row in rows} == expected_units
            assert request["evidence_selection"]["previously_exposed_task_ids"] == sorted(seen_tasks)
            assert all("head" not in value and "feature_bounds" not in value
                       and "training_units" not in value and "admission" not in value
                       for value in request["previous_candidates"])
            assert "VALIDATION_SECRET" not in json.dumps(request)
            assert "SEALED_TEST" not in json.dumps(request)
        record = first.report()["optimizer_evidence"][iteration]
        assert record["unit_ids"] == sorted(expected_units)
        assert record["task_ids"] == sorted(expected_tasks)
        assert record["task_commitments_sha256"] == canonical_hash([
            row for row in sorted(data[1], key=lambda row: row["task_id"])
            if row["task_id"] in expected_tasks
        ])
        assert record["training_evidence_sha256"] == canonical_hash(left["training_evidence"])
        assert record["request_id"] == left["request_id"] and record["callback_invoked"]
        assert record["unit_count"] == 2 and record["task_count"] == 6
        assert record["observation_count"] == 12
        unsigned = {key: value for key, value in record.items() if key != "evidence_selection_sha256"}
        assert record["evidence_selection_sha256"] == canonical_hash(unsigned)
        seen_tasks.update(expected_tasks)
        seen_units.update(expected_units)
    counts = first.accounting()
    assert counts["optimizer_sample_tasks"] == len(seen_tasks) < 30
    assert counts["optimizer_sample_units"] == len(seen_units) < 10
    assert counts["optimizer_task_exposures"] == 12
    assert counts["optimizer_unit_exposures"] == 4
    assert counts["optimizer_observation_exposures"] == 24
    assert counts["available_training_tasks"] == 30
    assert counts["available_training_units"] == 10
    complete_matrix = {
        (task["task_id"], seed) for task in splits["train"] for seed in (11, 19)
    }
    assert len(good.executions) == 3 * len(complete_matrix)
    for arm in {request["arm"] for request in good.executions}:
        records = [request for request in good.executions if request["arm"] == arm]
        assert {(row["task"]["task_id"], row["seed"]) for row in records} == complete_matrix
        assert len(records) == len(complete_matrix)
    for candidate in first.candidate_bank[1:]:
        assert candidate["training_units"] == candidate["head"]["training_task_count"] == 10
        assert candidate["training_tasks"] == 30 and candidate["paired_training_observations"] == 60
        assert len(candidate["head"]["training_task_folds"]) == 10


def test_sampled_optimizer_provenance_and_actual_counts_survive_policy_json_freeze():
    instance, callbacks, splits, policy = fitted(
        SyntheticCallbacks(proposals=[INSTRUCTION, OTHER_INSTRUCTION]),
        data=population(units=50, questions_per_unit=2),
        config=chisel.ChiselConfig(optimizer_max_units=2),
    )
    document = json.loads(json.dumps(policy.to_dict()))
    assert document["config"]["optimizer_max_units"] == 2
    assert document["optimizer_evidence"] == instance.report()["optimizer_evidence"]
    assert document["optimizer_evidence_sha256"] == canonical_hash(document["optimizer_evidence"])
    assert chisel.FrozenPolicy.from_dict(document) == policy
    counts = instance.accounting()
    instance.evaluate_test(
        lambda: splits["test"], policy_sha256=policy.policy_sha256,
        source_sha256=policy.source_sha256,
    )
    assert instance.accounting()["optimizer_sample_units"] == counts["optimizer_sample_units"]
    assert instance.accounting()["optimizer_sample_tasks"] == counts["optimizer_sample_tasks"]
    assert len(callbacks.optimizations) == 2
    report_copy = instance.report()
    report_copy["optimizer_evidence"][0]["unit_ids"].clear()
    assert instance.report()["optimizer_evidence"][0]["unit_ids"]
    broken = deepcopy(document)
    record = broken["optimizer_evidence"][0]
    record["unit_count"] += 1
    record.pop("evidence_selection_sha256")
    record["evidence_selection_sha256"] = canonical_hash(record)
    broken["optimizer_evidence_sha256"] = canonical_hash(broken["optimizer_evidence"])
    broken.pop("policy_sha256")
    broken["policy_sha256"] = canonical_hash(broken)
    with pytest.raises(ValueError, match="sampling plan"):
        chisel.FrozenPolicy.from_dict(broken)


def test_optimizer_can_cite_explicitly_previously_exposed_training_evidence():
    data = population(units=100, questions_per_unit=2)
    callbacks = SyntheticCallbacks(
        proposals=[INSTRUCTION, OTHER_INSTRUCTION],
        score=lambda request: (
            0.9 if request["residual"] and request["residual"]["instruction"] == OTHER_INSTRUCTION
            else 0.6 if request["residual"] else 0.25
        ),
    )
    prior_support = []

    def previous_evidence(request):
        result = callbacks.optimize(request)
        if request["iteration"] == 1:
            prior = {
                row["task"]["task_id"] for row in callbacks.optimizations[0]["training_evidence"]
            }
            current = {row["task"]["task_id"] for row in request["training_evidence"]}
            assert prior - current
            prior_support.append(min(prior - current))
            result["supporting_task_ids"] = prior_support[:]
        return result

    instance, _, _, policy = fitted(
        callbacks, data=data, config=chisel.ChiselConfig(optimizer_max_units=2),
        optimize=previous_evidence, run_id="synthetic-prior-evidence",
    )
    candidate = next(row for row in instance.candidate_bank if row.get("iteration") == 1)
    assert candidate["supporting_current_task_ids"] == []
    assert candidate["supporting_previous_task_ids"] == prior_support
    assert prior_support[0] in instance.report()["optimizer_evidence"][1]["previously_exposed_task_ids"]
    assert policy.to_dict()["candidate"]["iteration"] == 1
    assert chisel.FrozenPolicy.from_dict(policy.to_dict()) == policy


@pytest.mark.parametrize("scope", ["unseen_train", "validation", "test", "unknown"])
def test_optimizer_support_cannot_name_unprovided_or_heldout_evidence(scope):
    data = population(questions_per_unit=2)
    callbacks = SyntheticCallbacks()

    def invalid_support(request):
        result = callbacks.optimize(request)
        provided = {row["task"]["task_id"] for row in request["training_evidence"]}
        if scope == "unseen_train":
            result["supporting_task_ids"] = [
                next(task["task_id"] for task in data[2]["train"] if task["task_id"] not in provided)
            ]
        elif scope == "unknown":
            result["supporting_task_ids"] = ["unknown-private-evaluator-id"]
        else:
            result["supporting_task_ids"] = [data[2][scope][0]["task_id"]]
        return result

    instance, _, splits = controller(
        callbacks, data=data, optimize=invalid_support,
        config=chisel.ChiselConfig(optimizer_max_units=1),
    )
    with pytest.raises(ValueError, match="supplied evidence"):
        instance.train(splits["train"])
    assert instance.state == "blocked"
    assert len(callbacks.executions) == len(splits["train"])
    assert instance.accounting()["optimizer_sample_units"] == 1
    assert instance.accounting()["optimizer_sample_tasks"] == 2
    assert len(instance.report()["optimizer_evidence"]) == 1


def test_default_optimizer_payload_preserves_full_training_and_legacy_callback_shape():
    default, callbacks, splits = controller()
    explicit, explicit_callbacks, explicit_splits = controller(
        config=chisel.ChiselConfig(optimizer_max_units=None),
    )
    default.train(splits["train"])
    explicit.train(explicit_splits["train"])
    assert callbacks.optimizations == explicit_callbacks.optimizations
    assert default.candidate_bank == explicit.candidate_bank
    expected_fields = {
        "schema", "run_id", "phase", "iteration", "identity", "objective", "native_metric",
        "limits", "training_evidence", "previous_candidates", "request_id",
    }
    for request in callbacks.optimizations:
        assert set(request) == expected_fields
        assert "optimizer_max_units" not in request["limits"]
        assert {row["task"]["task_id"] for row in request["training_evidence"]} == {
            task["task_id"] for task in splits["train"]
        }
    assert callbacks.optimizations[1]["previous_candidates"] == default.candidate_bank[1:]
    assert default.accounting()["optimizer_sample_units"] == 5
    assert default.accounting()["optimizer_max_units"] is None


def test_oversized_unit_cap_includes_the_complete_available_training_population():
    instance, callbacks, splits = controller(config=chisel.ChiselConfig(optimizer_max_units=999))
    instance.train(splits["train"])
    assert instance.accounting()["optimizer_sample_units"] == 5
    assert all(record["unit_count"] == 5 for record in instance.report()["optimizer_evidence"])
    assert all(len(request["training_evidence"]) == len(splits["train"])
               for request in callbacks.optimizations)


def test_sampling_counts_only_invoked_callbacks_and_preserves_failed_call_exposure(monkeypatch):
    callbacks = SyntheticCallbacks()

    def failing_optimizer(request):
        callbacks.optimizations.append(deepcopy(request))
        raise OSError("synthetic optimizer receipt unavailable")

    instance, _, splits = controller(
        callbacks, optimize=failing_optimizer, config=chisel.ChiselConfig(optimizer_max_units=1),
    )
    with pytest.raises(OSError):
        instance.train(splits["train"])
    assert instance.accounting()["optimizer_sample_units"] == 1
    assert instance.accounting()["optimizer_sample_tasks"] == 1
    assert instance.report()["optimizer_evidence"][0]["callback_invoked"]
    optimizer_usage = instance.accounting()["by_stage"]["training_optimizer"]["usage"]["total_tokens"]
    assert optimizer_usage["total"] is None and optimizer_usage["missing_calls"] == 1
    blocked, _, blocked_splits = controller(config=chisel.ChiselConfig(optimizer_max_units=1))
    original = blocked._invoke

    def before_callback(callback, request, kind, **kwargs):
        if kind == "optimizer":
            raise RuntimeError("synthetic pre-callback source gate")
        return original(callback, request, kind, **kwargs)

    monkeypatch.setattr(blocked, "_invoke", before_callback)
    with pytest.raises(RuntimeError, match="pre-callback"):
        blocked.train(blocked_splits["train"])
    assert blocked.accounting()["optimizer_sample_units"] == 0
    assert blocked.accounting()["optimizer_sample_tasks"] == 0
    assert blocked.report()["optimizer_evidence"] == []


def test_full_lifecycle_keeps_validation_and_test_out_of_optimization():
    instance, callbacks, splits = controller()
    opened = []

    def loader():
        opened.append(True)
        return splits["test"]

    with pytest.raises(RuntimeError, match="requires state frozen"):
        instance.evaluate_test(loader, policy_sha256="not-frozen", source_sha256="not-frozen")
    assert not opened
    training_report = instance.train(splits["train"])
    frozen_bank = instance.candidate_bank
    selection = instance.select_validation(splits["validation"])
    assert instance.candidate_bank == frozen_bank
    assert selection["selected_candidate_id"] != "noop"
    assert len(callbacks.optimizations) == 2
    train_ids = {task["task_id"] for task in splits["train"]}
    for request in callbacks.optimizations:
        assert request["phase"] == "train"
        assert {row["task"]["task_id"] for row in request["training_evidence"]} == train_ids
        assert "VALIDATION_SECRET" not in json.dumps(request)
        assert "SEALED_TEST" not in json.dumps(request)
        assert request["limits"]["cumulative_token_limit"] is None
        assert request["limits"]["max_residuals_per_task"] == 1
    assert all(row["phase"] != "test" for row in callbacks.executions)
    policy = instance.freeze(source_hashes=SOURCE_HASHES)
    for policy_hash, source_hash in (("wrong", policy.source_sha256),
                                     (policy.policy_sha256, "wrong")):
        with pytest.raises(ValueError, match="final policy and source seals"):
            instance.evaluate_test(loader, policy_sha256=policy_hash, source_sha256=source_hash)
        assert not opened
    result = instance.evaluate_test(
        loader, policy_sha256=policy.policy_sha256, source_sha256=policy.source_sha256,
    )
    assert opened == [True]
    assert instance.state == "complete"
    assert result["baseline"]["mean_native_score"] == 0.25
    assert result["chisel"]["mean_native_score"] == 0.75
    assert len(callbacks.optimizations) == 2
    assert len([row for row in callbacks.executions if row["phase"] == "test"]) == 30
    for request in (row for row in callbacks.executions if row["phase"] == "test"):
        assert request["final_policy_sha256"] == policy.policy_sha256
        assert request["source_sha256"] == policy.source_sha256
    assert instance.report()["selection"] == selection
    assert not result["selection_or_optimization_from_test"]
    assert "VALIDATION_SECRET" not in json.dumps(instance.report())
    assert "SEALED_TEST_DIAGNOSTIC" not in json.dumps(instance.report())
    assert training_report["benchmark_improvement_claim"] is None
    json.dumps(instance.report(), allow_nan=False)
    with pytest.raises(RuntimeError):
        instance.evaluate_test(loader, policy_sha256=policy.policy_sha256,
                               source_sha256=policy.source_sha256)
    assert opened == [True]
    for method, tasks in ((instance.train, splits["train"]),
                          (instance.select_validation, splits["validation"])):
        with pytest.raises(RuntimeError):
            method(tasks)


@pytest.mark.parametrize("invalid", ["held_out", "missing", "changed_prompt", "private_category"])
def test_training_rejects_leaked_or_uncommitted_inputs_before_any_callback(invalid):
    instance, callbacks, splits = controller()
    tasks = deepcopy(splits["train"])
    if invalid == "held_out":
        tasks[0] = splits["validation"][0]
    elif invalid == "missing":
        tasks.pop()
    elif invalid == "changed_prompt":
        tasks[0]["public_prompt"] += " uncommitted held-out content"
    else:
        tasks[0]["private_category"] = "evaluator label"
    with pytest.raises(ValueError):
        instance.train(tasks)
    assert not callbacks.executions and not callbacks.optimizations
    assert instance.state == "initialized"


def test_wrong_validation_or_partial_population_never_reaches_executor():
    instance, callbacks, splits = controller()
    with pytest.raises(RuntimeError):
        instance.select_validation(splits["validation"])
    instance.train(splits["train"])
    count = len(callbacks.executions)
    with pytest.raises(ValueError, match="non-validation"):
        instance.select_validation(splits["test"])
    with pytest.raises(ValueError, match="incomplete validation"):
        instance.select_validation(splits["validation"][:-1])
    assert len(callbacks.executions) == count


def test_partition_seed_membership_ratio_and_independent_population_are_enforced():
    for modification in ("seed", "membership", "ratio", "missing_unit"):
        manifest, index, splits = population()
        if modification == "seed":
            manifest["split_seed"] += 1
        elif modification == "membership":
            manifest["splits"]["train"], manifest["splits"]["validation"] = (
                manifest["splits"]["validation"], manifest["splits"]["train"],
            )
        elif modification == "ratio":
            manifest["ratios"] = {"train": 0.6, "validation": 0.2, "test": 0.2}
        else:
            index.pop()
        with pytest.raises(ValueError):
            controller(data=(manifest, index, splits))


def test_deterministic_outcome_independent_rounding_is_accepted():
    data = population(units=7)
    instance, _, splits = controller(data=data)
    assert [len(splits[part]) for part in ("train", "validation", "test")] == [2, 1, 4]
    instance.train(splits["train"])
    assert instance.accounting()["available_training_units"] == 2


def test_all_questions_and_repetitions_stay_out_of_their_held_unit_router_fit(monkeypatch):
    data = population(units=50, questions_per_unit=3)
    captured = []
    original = chisel.training._fit_ridge

    def record_fit(rows, *, ridge):
        captured.append([row[0] for row in rows])
        return original(rows, ridge=ridge)

    monkeypatch.setattr(chisel.training, "_fit_ridge", record_fit)
    config = chisel.ChiselConfig(execution_seeds=(31, 47))
    instance, callbacks, splits = controller(data=data, config=config)
    instance.train(splits["train"])
    candidate = instance.candidate_bank[1]
    units = data[0]["splits"]["train"]
    assignments = deterministic_group_folds(units, folds=2, seed=chisel.SPLIT_SEED)
    assert len(captured) == 3
    for fold in range(2):
        assert set(captured[fold]) == {unit for unit in units if assignments[unit] != fold}
        assert not set(captured[fold]) & {unit for unit in units if assignments[unit] == fold}
    assert set(captured[-1]) == set(units)
    assert candidate["head"]["training_task_count"] == 10
    assert candidate["training_tasks"] == 30
    assert candidate["paired_training_observations"] == 60
    assert len(candidate["head"]["training_task_folds"]) == 10
    assert instance.accounting()["optimizer_sample_tasks"] == 30
    assert instance.accounting()["optimizer_sample_units"] == 10
    baseline = [row for row in callbacks.executions if row["arm"] == "baseline"]
    assert len(baseline) == 60
    for candidate_request in (row for row in callbacks.executions if row["residual"]):
        paired = next(row for row in baseline if row["pair_id"] == candidate_request["pair_id"])
        assert paired["seed"] == candidate_request["seed"]
        assert paired["identity"] == candidate_request["identity"]
        assert paired["task"] == candidate_request["task"]
    instance.select_validation(splits["validation"])
    assert len(captured) == 3


def test_candidates_and_final_policy_are_deeply_immutable_and_round_trip_json():
    instance, _, splits, policy = fitted()
    original = policy.to_dict()
    document = policy.to_dict()
    document["candidate"]["instruction"] = "Changed after validation."
    document["candidate"]["head"]["weights"]["question_character_count"] = 1e6
    assert policy.to_dict() == original
    bank = instance.candidate_bank
    bank[1]["head"]["intercept"] = -999
    assert instance.candidate_bank[1]["head"]["intercept"] != -999
    with pytest.raises(FrozenInstanceError):
        policy._document = "{}"
    restored = chisel.FrozenPolicy.from_dict(json.loads(json.dumps(original)))
    assert restored == policy
    assert restored.apply(splits["test"][0]["public_prompt"]) == policy.apply(
        splits["test"][0]["public_prompt"],
    )
    with pytest.raises(ValueError, match="policy_sha256"):
        chisel.FrozenPolicy.from_dict(document)
    with pytest.raises(RuntimeError):
        instance.freeze(source_hashes=SOURCE_HASHES)


def test_rehashed_configuration_changes_cannot_override_the_source_freeze():
    _, _, _, policy = fitted()
    document = policy.to_dict()
    document["config"]["application"] = "agents_md"
    document.pop("policy_sha256")
    document["policy_sha256"] = canonical_hash(document)
    with pytest.raises(ValueError, match="configuration differs"):
        chisel.FrozenPolicy.from_dict(document)


def test_source_drift_prevents_routing_and_does_not_open_test_loader(monkeypatch):
    instance, _, splits, policy = fitted()
    changed = deepcopy(chisel.implementation_provenance())
    changed["executable_sources"]["phiagent/harness/codex_chisel.py"] = "0" * 64
    monkeypatch.setattr(chisel, "implementation_provenance", lambda: deepcopy(changed))
    with pytest.raises(RuntimeError, match="sources changed"):
        policy.route(splits["test"][0]["public_prompt"])
    opened = []
    with pytest.raises(RuntimeError, match="sources changed"):
        instance.evaluate_test(lambda: opened.append(True), policy_sha256=policy.policy_sha256,
                               source_sha256=policy.source_sha256)
    assert not opened


def test_source_drift_during_a_callback_preserves_usage_and_stops_the_next_call(monkeypatch):
    callbacks = SyntheticCallbacks()
    original = chisel.implementation_provenance()

    def changing_executor(request):
        result = callbacks.execute(request)
        changed = deepcopy(original)
        changed["executable_sources"]["phiagent/harness/codex_chisel.py"] = "1" * 64
        monkeypatch.setattr(chisel, "implementation_provenance", lambda: deepcopy(changed))
        return result

    instance, _, splits = controller(callbacks, execute=changing_executor)
    with pytest.raises(RuntimeError, match="sources changed"):
        instance.train(splits["train"])
    assert instance.state == "blocked"
    assert len(callbacks.executions) == 1
    assert not callbacks.optimizations
    assert instance.accounting()["all_calls"]["usage"]["total_tokens"]["total"] == 15


def test_missing_external_source_seals_do_not_release_test():
    instance, _, splits = controller()
    instance.train(splits["train"])
    instance.select_validation(splits["validation"])
    with pytest.raises(ValueError, match="source hashes"):
        instance.freeze(source_hashes={"executor": SOURCE_HASHES["executor"]})
    assert instance.state == "selected"
    invalid = {**SOURCE_HASHES, "codex_cli": "unhashed"}
    with pytest.raises(ValueError, match="SHA-256"):
        instance.freeze(source_hashes=invalid)


@pytest.mark.parametrize("application", ["prompt_suffix", "agents_md"])
def test_at_most_one_learned_residual_is_applied_without_replacing_codex(application):
    callbacks = SyntheticCallbacks(proposals=[INSTRUCTION, OTHER_INSTRUCTION])
    instance, callbacks, splits, policy = fitted(
        callbacks, config=chisel.ChiselConfig(application=application),
    )
    assert len(instance.candidate_bank) == 3
    assert all(row["origin"] == "optimizer_callback_train_only"
               for row in instance.candidate_bank[1:])
    for task in splits["test"]:
        applied = policy.apply(task["public_prompt"])
        assert applied["decision"]["residual_id"] is not None
        text = applied["codex_prompt"] + applied["agents_md_append"]
        assert text.count("CHISEL task-local residual") == 1
        assert (INSTRUCTION in text) != (OTHER_INSTRUCTION in text)
        if application == "prompt_suffix":
            assert applied["codex_prompt"].startswith(task["public_prompt"])
            assert applied["agents_md_append"] == ""
        else:
            assert applied["codex_prompt"] == task["public_prompt"]
            assert applied["agents_md_append"].startswith("\n\n## ")
    assert all(request["contract"]["runtime"] == "unmodified_official_codex_cli"
               for request in callbacks.executions)
    assert all(request["contract"]["max_residuals_per_task"] == 1
               for request in callbacks.executions)
    for request in callbacks.executions:
        if request["residual"] is None:
            assert request["codex_prompt"] == request["task"]["public_prompt"]
            assert request["agents_md_append"] == ""


@pytest.mark.parametrize("reason", ["empty_optimizer", "negative_train", "tied_validation",
                                   "worse_validation", "missing_native_score"])
def test_noop_is_retained_when_there_is_no_measured_validation_improvement(reason):
    def score(request):
        if reason == "missing_native_score" and request["phase"] == "train":
            return None
        if request["residual"]:
            if reason == "negative_train" or (
                reason == "worse_validation" and request["phase"] == "validation"
            ):
                return 0.0
            if reason == "tied_validation" and request["phase"] == "validation":
                return 0.25
            return 0.75
        return 0.25

    callbacks = SyntheticCallbacks(proposals=[""] if reason == "empty_optimizer" else None, score=score)
    instance, callbacks, splits, policy = fitted(callbacks)
    assert policy.to_dict()["candidate"]["id"] == "noop"
    assert instance.report()["selection"]["reason"] == "retain_empty_baseline"
    task = splits["test"][0]
    applied = policy.apply(task["public_prompt"])
    assert applied["codex_prompt"] == task["public_prompt"]
    assert applied["agents_md_append"] == ""
    assert applied["decision"]["residual_id"] is None
    result = instance.evaluate_test(
        lambda: splits["test"], policy_sha256=policy.policy_sha256,
        source_sha256=policy.source_sha256,
    )
    test_requests = [row for row in callbacks.executions if row["phase"] == "test"]
    assert len(test_requests) == len(splits["test"]) * 2
    assert all(row["residual"] is None for row in test_requests)
    assert result["baseline"]["mean_native_score"] == result["chisel"]["mean_native_score"]


def test_local_routing_is_learned_deterministic_and_abstains_outside_training_support():
    data = population(
        units=100, prompt=lambda unit, question: (
            "Count the blue objects before submission." if unit % 2 == 0
            else "Describe the blue objects in the scene."
        ),
    )

    def score(request):
        numeric = request["task"]["public_prompt"].startswith("Count")
        return float(numeric == bool(request["residual"]))

    instance, _, splits, policy = fitted(SyntheticCallbacks(score=score), data=data)
    numeric = "Count the blue objects before submission."
    descriptive = "Describe the blue objects in the scene."
    assert policy.route(numeric)["residual_id"] is not None
    assert policy.route(descriptive)["residual_id"] is None
    assert policy.route(numeric) == policy.route(numeric)
    assert policy.route("unseen " * 1000)["reason"] == "outside_training_feature_support"
    reversed_data = (data[0], list(reversed(data[1])), {
        name: list(reversed(tasks)) for name, tasks in splits.items()
    })
    other, _, _, other_policy = fitted(SyntheticCallbacks(score=score), data=reversed_data)
    assert other.candidate_bank == instance.candidate_bank
    assert other_policy.policy_sha256 == policy.policy_sha256
    assert chisel.public_text_features(numeric) == chisel.public_text_features(numeric)
    with pytest.raises(ValueError):
        policy.route({"public_prompt": numeric, "category": "private-label"})


def test_token_accounting_is_exact_once_cache_inclusive_and_separated_by_stage():
    instance, callbacks, splits, policy = fitted()
    instance.evaluate_test(lambda: splits["test"], policy_sha256=policy.policy_sha256,
                           source_sha256=policy.source_sha256)
    accounting = instance.accounting()
    assert accounting["cumulative_token_limit"] is None
    assert accounting["optimizer_sample_tasks"] == 5
    assert accounting["optimizer_sample_units"] == 5
    assert accounting["optimizer_observation_exposures"] == 10
    assert accounting["optimizer_task_exposures"] == 10
    assert accounting["optimizer_unit_exposures"] == 10
    stages = accounting["by_stage"]
    assert stages["training_executor"]["calls"] == 10
    assert stages["training_optimizer"]["calls"] == 2
    assert stages["validation_selection"]["calls"] == 10
    assert stages["final_baseline"]["calls"] == 15
    assert stages["final_chisel"]["calls"] == 15
    assert [stages[name]["usage"]["total_tokens"]["total"] for name in (
        "training_executor", "training_optimizer", "validation_selection",
        "final_baseline", "final_chisel",
    )] == [160, 240, 160, 225, 255]
    all_usage = accounting["all_calls"]["usage"]
    assert all_usage["total_tokens"]["total"] == 1040
    assert all_usage["total_tokens"]["total"] == (
        all_usage["input_tokens"]["total"] + all_usage["output_tokens"]["total"]
    )
    assert all_usage["cache_read_tokens"]["total"] > 0
    assert all_usage["cost_usd"]["total"] == pytest.approx(0.99)
    assert len({row["request_id"] for row in callbacks.executions}) == len(callbacks.executions)
    assert instance.report()["optimization_iterations"][1]["status"] == (
        "duplicate_reused_training_evidence"
    )
    assert callbacks.optimizations[1]["training_evidence"][0]["candidates"]


def test_unknown_usage_remains_unknown_and_cannot_manufacture_a_routing_advantage():
    callbacks = SyntheticCallbacks()

    def unknown_usage(request):
        result = callbacks.execute(request)
        result["usage"] = dict.fromkeys(chisel.USAGE_FIELDS)
        return result

    instance, _, splits = controller(callbacks, execute=unknown_usage)
    instance.train(splits["train"])
    assert instance.candidate_bank[1]["admission"] == "incomplete_measured_paired_evidence"
    assert not instance.candidate_bank[1]["enabled"]
    totals = instance.accounting()["by_stage"]["training_executor"]["usage"]
    assert totals["total_tokens"] == {"total": None, "measured_subtotal": 0, "missing_calls": 10}
    combined = instance.accounting()["all_calls"]["usage"]["total_tokens"]
    assert combined == {"total": None, "measured_subtotal": 240, "missing_calls": 10}


def test_a_single_training_unit_cannot_create_cross_fitted_support():
    instance, _, splits = controller(data=population(units=5, questions_per_unit=3))
    instance.train(splits["train"])
    candidate = instance.candidate_bank[1]
    assert candidate["training_tasks"] == 3
    assert candidate["training_units"] == 1
    assert candidate["admission"] == "insufficient_independent_support"
    assert candidate["head"] is None
    assert not candidate["enabled"]


def test_unlimited_accounting_does_not_stop_predeclared_optimizer_iterations():
    callbacks = SyntheticCallbacks(proposals=[""])

    def enormous_usage(request):
        result = callbacks.optimize(request)
        result["usage"] = usage(10**15, 10**15, cache=0, cost=None)
        return result

    instance, _, splits = controller(callbacks, optimize=enormous_usage,
                                     config=chisel.ChiselConfig(optimization_iterations=3))
    instance.train(splits["train"])
    assert len(callbacks.optimizations) == 3
    assert instance.state == "trained"
    assert instance.accounting()["by_stage"]["training_optimizer"]["usage"]["total_tokens"][
        "total"
    ] == 6 * 10**15
    with pytest.raises(ValueError, match="unlimited"):
        chisel.ChiselConfig(cumulative_token_limit=10**18)


def test_native_soft_scores_keep_their_predeclared_scale():
    callbacks = SyntheticCallbacks(score=lambda request: 40.0 if request["residual"] else 10.0)
    instance, _, _, policy = fitted(callbacks, config=chisel.ChiselConfig(
        score_maximum=100.0, native_metric="synthetic-native-percent",
    ))
    assert policy.to_dict()["candidate"]["head"]["oof_mean_utility"] == pytest.approx(0.3)
    selected = next(row for row in instance.report()["selection"]["results"]
                    if row["candidate_id"] != "noop")
    assert selected["mean_native_score"] == 40.0


def test_callback_request_mutation_does_not_change_training_evidence_or_frozen_inputs():
    callbacks = SyntheticCallbacks()

    def mutating_optimizer(request):
        result = callbacks.optimize(request)
        request["training_evidence"][0]["baseline"]["native_score"] = -100
        request["training_evidence"][0]["task"]["public_prompt"] = "Changed by callback."
        if request["previous_candidates"]:
            request["previous_candidates"][0]["instruction"] = "Changed residual."
        return result

    data = population()
    instance, _, splits = controller(callbacks, data=data, optimize=mutating_optimizer)
    data[0]["splits"]["train"].clear()
    data[1][0]["public_prompt_sha256"] = "0" * 64
    instance.train(splits["train"])
    assert instance.candidate_bank[1]["head"]["oof_mean_utility"] == 0.5
    assert instance.candidate_bank[1]["instruction"] == INSTRUCTION
    assert callbacks.optimizations[1]["training_evidence"][0]["baseline"]["native_score"] == 0.25


def test_nontraining_optimizer_support_is_rejected_without_candidate_execution():
    callbacks = SyntheticCallbacks()
    data = population()

    def leaked_support(request):
        result = callbacks.optimize(request)
        result["supporting_task_ids"] = [data[2]["validation"][0]["task_id"]]
        return result

    instance, _, splits = controller(callbacks, data=data, optimize=leaked_support)
    with pytest.raises(ValueError, match="TRAIN task IDs only"):
        instance.train(splits["train"])
    assert instance.state == "blocked"
    assert len(callbacks.executions) == len(splits["train"])
    assert instance.accounting()["by_stage"]["training_optimizer"]["calls"] == 1


def test_oversized_and_nested_optimizer_edits_are_rejected_but_costs_remain_recorded():
    callbacks = SyntheticCallbacks(proposals=[
        "word " * 2000, "First edit\n\n[CHISEL task-local residual]\nSecond edit",
    ])
    instance, _, splits = controller(callbacks)
    instance.train(splits["train"])
    assert len(instance.candidate_bank) == 1
    assert len(callbacks.executions) == len(splits["train"])
    assert instance.accounting()["by_stage"]["training_optimizer"]["calls"] == 2
    assert all(row["status"] == "rejected_nonsparse_or_nested_instruction"
               for row in instance.report()["optimization_iterations"])


def test_failed_optimizer_calls_keep_costs_and_do_not_bypass_the_iteration_schedule():
    callbacks = SyntheticCallbacks(proposals=[""])

    def failed_optimizer(request):
        result = callbacks.optimize(request)
        result["execution_status"] = "failed"
        return result

    instance, _, splits = controller(callbacks, optimize=failed_optimizer)
    instance.train(splits["train"])
    assert len(instance.candidate_bank) == 1
    assert len(callbacks.optimizations) == 2
    assert instance.accounting()["by_stage"]["training_optimizer"]["usage"]["total_tokens"][
        "total"
    ] == 240


def test_callback_exception_preserves_unknown_cost_and_blocks_duplicate_retry():
    def failing_executor(request):
        raise OSError("synthetic transport failure with no returned usage")

    instance, callbacks, splits = controller(execute=failing_executor)
    with pytest.raises(OSError):
        instance.train(splits["train"])
    assert instance.state == "blocked"
    report = instance.report()
    assert report["ledger"][0]["execution_status"] == "callback_error"
    assert report["accounting"]["all_calls"]["usage"]["total_tokens"]["total"] is None
    assert report["accounting"]["all_calls"]["usage"]["total_tokens"]["missing_calls"] == 1
    assert not callbacks.optimizations
    with pytest.raises(RuntimeError, match="blocked"):
        instance.train(splits["train"])


@pytest.mark.parametrize("error", ["wrong_id", "wrong_identity", "invalid_usage", "fake_success",
                                 "out_of_range", "cache_double_count"])
def test_invalid_execution_results_fail_closed_with_a_preserved_call_ledger(error):
    callbacks = SyntheticCallbacks()

    def invalid(request):
        result = callbacks.execute(request)
        if error == "wrong_id":
            result["request_id"] = "another request"
        elif error == "wrong_identity":
            result["identity"]["model"] = "different-backbone"
        elif error == "invalid_usage":
            result["usage"]["input_tokens"] = -1
        elif error == "fake_success":
            result["success"] = "true"
        elif error == "out_of_range":
            result["native_score"] = 101
        else:
            result["usage"]["total_tokens"] += result["usage"]["cache_read_tokens"]
        return result

    instance, _, splits = controller(callbacks, execute=invalid)
    with pytest.raises(ValueError):
        instance.train(splits["train"])
    assert instance.state == "blocked"
    assert len(instance.report()["ledger"]) == 1


def test_incomplete_final_native_scores_preserve_population_without_imputation():
    callbacks = SyntheticCallbacks(
        score=lambda request: None if request["phase"] == "test" else (
            0.75 if request["residual"] else 0.25
        ),
    )
    instance, _, splits, policy = fitted(callbacks)
    result = instance.evaluate_test(
        lambda: splits["test"], policy_sha256=policy.policy_sha256,
        source_sha256=policy.source_sha256,
    )
    for arm in ("baseline", "chisel"):
        assert not result[arm]["complete_native_scores"]
        assert result[arm]["mean_native_score"] is None
        assert result[arm]["tasks"] == len(splits["test"])
        assert all(row["result"]["native_score"] is None for row in result[arm]["rows"])


def test_implementation_provenance_identifies_actual_reused_chisel_sources():
    provenance = chisel.implementation_provenance()
    assert provenance["reused_trainer"].endswith("fit_cross_fitted_linear_head")
    assert provenance["cross_fit_scope"] == "routing head only; compiler uses TRAIN only"
    assert provenance["max_residuals_per_task"] == 1
    assert "phiagent/harness/mixture_router.py" in provenance["inspected_design_sources"]
    assert all(len(value) == 64 for value in provenance["executable_sources"].values())
    assert provenance["executable_sources"]["phiagent/harness/codex_chisel.py"] == (
        hashlib.sha256(chisel.Path(chisel.__file__).read_bytes()).hexdigest()
    )
