"""All fixtures are invented; tests never load benchmark records or scorers."""

import copy
import json

import pytest

from phiagent.harness.aime_202060 import (
    METHODS, grounded_repair, ledger_prompt, score_only, select_candidate, usage_sum, validate_data,
    validate_ledger, validate_manifest, validate_policy_freeze,
)
from phiagent.harness.benchmark_partition import build_partition, canonical_sha256


def fixture():
    items = [
        {"id": f"2025-synthetic-{i}", "question": f"Invented fixture {i}", "answer": str(i)}
        for i in range(30)
    ]
    records = [
        {"id": item["id"], "payload_sha256": canonical_sha256(item),
         "public_task_sha256": canonical_sha256(item["question"])}
        for item in items
    ]
    manifest = build_partition(records, benchmark="aime2025", seed=20260928)
    index = {item["id"]: item for item in items}
    data = {
        "train": [index[i] for i in manifest["splits"]["train"]],
        "development": [index[i] for i in manifest["splits"]["validation"]],
        "test": [],
    }
    return manifest, data, index


def test_exact_6_6_18_contract_and_no_test_in_optimization():
    manifest, data, items = fixture()
    validate_data(data, manifest)
    assert manifest["counts"] == {"train": 6, "validation": 6, "test": 18}
    exposed = {**data, "test": [items[i] for i in manifest["splits"]["test"]]}
    with pytest.raises(ValueError):
        validate_data(exposed, manifest)
    validate_data(exposed, manifest, final=True)
    with pytest.raises(ValueError):
        validate_data(data, manifest, final=True)


@pytest.mark.parametrize("change", ["seed", "size", "swap", "duplicate", "payload", "answer"])
def test_hard_gap_rejects_repartition_and_content_drift(change):
    manifest, data, _ = fixture()
    if change == "seed":
        manifest["split_seed"] += 1
    elif change == "size":
        data["train"].pop()
    elif change == "swap":
        a, b = manifest["splits"]["train"], manifest["splits"]["validation"]
        a[0], b[0] = b[0], a[0]
    elif change == "duplicate":
        data["development"][0] = data["train"][0]
    elif change == "payload":
        data["train"][0]["question"] += " mutated"
    else:
        data["train"][0]["answer"] = "999"
    with pytest.raises(ValueError):
        validate_data(data, manifest)


def test_old_cohort_is_not_a_new_result():
    manifest, _, _ = fixture()
    manifest["benchmark"] = "aime2024"
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_validation_projection_contains_no_question_trace_answer_or_failure():
    row = {
        "id": "synthetic", "hard": 0, "usage": {"total_tokens": 5},
        "question": "HIDDEN QUESTION", "response": "HIDDEN TRACE",
        "predicted_answer": "HIDDEN ANSWER", "fail_reason": "HIDDEN FAILURE",
        "raw_response": {"content": "HIDDEN NATIVE RESPONSE"}, "answer": "GOLD",
    }
    projected = score_only(row)
    assert projected["hard"] == 0
    assert "HIDDEN" not in str(projected) and "GOLD" not in str(projected)
    assert "raw_response" not in projected


def test_all_four_audited_before_test():
    ids = ["synthetic-train"]
    freeze = {
        "manifest_sha256": "bound", "test_opened": False, "training_ids": ids,
        "policies": {
            name: {"status": "PASS", "policy_sha256": "bytes", "official_rounds": 4,
                   "training_ids": ids}
            for name in METHODS
        },
    }
    validate_policy_freeze(freeze, "bound")
    for name in METHODS:
        invalid = copy.deepcopy(freeze)
        del invalid["policies"][name]
        with pytest.raises(ValueError):
            validate_policy_freeze(invalid, "bound")
    freeze["policies"]["skillopt"]["official_rounds"] = 3
    with pytest.raises(ValueError):
        validate_policy_freeze(freeze, "bound")


def test_constraint_ledger_grounded_only_in_training():
    ledger = {"diagnosis": "synthetic", "rules": [
        {"rule_id": "C1", "check": "Check each stated endpoint explicitly.",
         "evidence_ids": ["synthetic-train"]},
    ]}
    validate_ledger(ledger, ["synthetic-train"])
    ledger["rules"][0]["evidence_ids"] = ["synthetic-validation"]
    with pytest.raises(ValueError):
        validate_ledger(ledger, ["synthetic-train"])


def test_complete_validation_selection_ties_use_real_cost():
    scores = [
        {"candidate_id": "b", "tasks": 6, "correct": 4, "total_tokens": 40},
        {"candidate_id": "a", "tasks": 6, "correct": 4, "total_tokens": 30},
    ]
    assert select_candidate(scores)["candidate_id"] == "a"
    scores[0]["tasks"] = 5
    with pytest.raises(ValueError):
        select_candidate(scores)
    assert usage_sum([{"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}]) == {
        "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
    }


def test_runtime_returns_only_scores_to_validation_optimizers(monkeypatch):
    from pathlib import Path
    from scripts import run_aime_202060 as runner

    study = object.__new__(runner.ResplitStudy)
    study.root = Path("synthetic-no-files-written")
    study.train_ids = {"train-synthetic"}
    study.validation_ids = {"validation-synthetic"}
    study.records = []
    study.config = {"mode": "skillopt"}
    writes = {}
    row = {
        "id": "validation-synthetic", "hard": 0, "question": "HIDDEN QUESTION",
        "response": "HIDDEN TRACE", "predicted_answer": "HIDDEN ANSWER",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "raw_response": {"id": "synthetic-response"}, "fail_reason": "HIDDEN FAILURE",
    }
    monkeypatch.setattr(runner.aime.Study, "infer", lambda *a, **k: dict(row))
    monkeypatch.setattr(runner.base, "write", lambda path, value: writes.update({str(path): value}))
    answer = study.infer({"id": "validation-synthetic"}, "synthetic skill", "skillopt")
    assert "HIDDEN" not in str(answer)
    assert any("private_records/validation" in name for name in writes)
    with pytest.raises(RuntimeError, match="Test or unknown"):
        study.infer({"id": "test-synthetic"}, "synthetic skill", "skillopt")
    with pytest.raises(RuntimeError, match="32768"):
        study.infer({"id": "train-synthetic"}, "synthetic skill", "skillopt", maximum=2048)


def test_serial_transport_contract_uses_full_capacity_timeouts():
    from scripts.aime_client_admission import policy, serial_configuration

    config = {"resource": {}, "budget": {}, "model": {"max_tokens": 32768}}
    serial_configuration(config)
    assert policy(config) == {
        "limit": 1, "workers": 1, "analysts": 1,
        "provider_timeout": 7200, "client_timeout": 7500,
    }
    assert config["model"]["max_tokens"] == 32768


def synthetic_review():
    return {
        "verdict": "REPAIR", "draft_error_quote": "We omitted the endpoint.",
        "why_invalid": "The stated interval includes its right endpoint.",
        "replacement_derivation": "Including the right endpoint adds the single omitted term. "
                                  "Every interior term remains counted exactly once.",
        "answer": 42,
        "checks": [
            {"problem_quote": "inclusive endpoints", "status": "checked",
             "evidence": "Both endpoint substitutions satisfy the original condition."},
        ],
    }


def test_grounded_verifier_changes_answer_only_with_actual_evidence_quotes():
    problem = "Synthetic task with inclusive endpoints."
    draft = "We omitted the endpoint.\nANSWER: 41"
    review = synthetic_review()
    answer, decision = grounded_repair(problem, draft, review)
    assert decision == "apply_grounded_repair" and answer.endswith("ANSWER: 42")
    for field, value in (
        ("draft_error_quote", "Invented statement not in the draft"),
        ("why_invalid", ""), ("replacement_derivation", ""), ("answer", 1000),
        ("answer", True), ("checks", []),
    ):
        assert grounded_repair(problem, draft, {**review, field: value})[0] == draft
    review["checks"][0]["problem_quote"] = "An invented constraint"
    assert grounded_repair(problem, draft, review)[0] == draft


def test_verification_keeps_unresolved_or_valid_drafts_without_reference_access():
    review = synthetic_review()
    for verdict in ("UNRESOLVED", "KEEP", "invalid"):
        assert grounded_repair("synthetic", "draft", {**review, "verdict": verdict})[0] == "draft"
    ledger = {"rules": [{
        "rule_id": "C1", "check": "Check the included endpoints explicitly.",
        "evidence_ids": ["private-training-identity"],
    }]}
    assert "private-training-identity" not in ledger_prompt(ledger)


def test_sealed_file_never_read_before_all_four_valid_policy_bindings():
    from pathlib import Path
    from scripts.prepare_aime_202060 import open_frozen_test

    frozen = {
        "manifest_sha256": "bound", "test_opened": False, "training_ids": ["synthetic"],
        "policies": {
            name: {"status": "PASS", "policy_sha256": "hash-" + name, "path": name,
                   "official_rounds": 4, "training_ids": ["synthetic"]}
            for name in METHODS
        },
    }
    reads = []

    def reader(path):
        reads.append(path.name)
        if path.name == "POLICY_FREEZE.json":
            return frozen
        if path.name == "VERIFIED.json":
            return {"status": "PASS", "policy_freeze_sha256": "freeze-hash"}
        if path.name == "SEALED_TEST.json":
            return {"test": [{"id": "synthetic-only"}]}
        raise AssertionError("Unexpected input read")

    def digest(path):
        return "freeze-hash" if path.name == "POLICY_FREEZE.json" else "hash-" + path.name

    result = open_frozen_test(
        Path("synthetic-partition"), Path("synthetic-freeze"), "bound", read_json=reader, sha_file=digest,
    )
    assert result[1] == [{"id": "synthetic-only"}]
    assert reads[-1] == "SEALED_TEST.json"
    frozen["policies"]["autosaddler"]["official_rounds"] = 3
    reads.clear()
    with pytest.raises(ValueError):
        open_frozen_test(
            Path("synthetic-partition"), Path("synthetic-freeze"), "bound",
            read_json=reader, sha_file=digest,
        )
    assert "SEALED_TEST.json" not in reads


def test_every_preparer_receives_corrected_partition_path_explicitly(monkeypatch):
    from types import SimpleNamespace
    from scripts import continue_aime_202060 as continuation
    from scripts import queue_aime_202060 as initial

    calls = []
    monkeypatch.setattr(
        initial.subprocess, "run",
        lambda command, **kwargs: (
            calls.append(command) or SimpleNamespace(returncode=0, stdout="{}", stderr="")
        ),
    )
    initial.launch(initial.RUNS[0], "/synthetic/bundle", "synthetic-hash")
    assert "--partition-root /opt/phiagent/work/df-202060/20260928-partition-v2/aime2025" in calls[0][-1]
    commands = []
    monkeypatch.setattr(continuation, "execute", lambda command: commands.append(command))
    continuation.launch_stage("/synthetic/bundle", "synthetic-hash", "test", "final", 3, 100, [])
    index = commands[0].index("--partition-root")
    assert commands[0][index + 1] == initial.PARTITION + "/aime2025"


def test_metadata_resource_forks_are_filtered_before_decoding():
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.run_aime_202060 import package_versions

    class ResourceFork:
        _path = Path("synthetic/._broken.dist-info")

        @property
        def metadata(self):
            raise AssertionError("AppleDouble metadata must not be decoded")

    packages, ignored = package_versions([
        ResourceFork(),
        SimpleNamespace(_path=Path("synthetic/real.dist-info"),
                        metadata={"Name": "synthetic-package"}, version="1.0"),
    ])
    assert packages == {"synthetic-package": "1.0"}
    assert ignored == ["synthetic/._broken.dist-info"]


def test_fixed_no_skill_validation_is_independent_and_never_duplicated():
    from scripts.continue_aime_202060 import ready_stages

    name = "no-skill-val-v1"
    assert [row[0] for row in ready_stages(set(), {}, {})] == [name]
    assert ready_stages(set(), {}, {name: {"LAUNCH.json": {"pid": 123}}}) == []
    assert ready_stages(set(), {}, {name: {"FAILURE.json": {"error": "synthetic"}}}) == []
    assert ready_stages({name}, {}, {}) == []
    assert ready_stages(set(), {name: {}}, {}) == []


def unused_waiter():
    return {
        "alive": True, "matching_command": True, "waiting": True,
        "gpu_selected": False, "model_owner": False, "provider_calls": 0,
        "provider_ledger": False, "result": False,
    }


@pytest.mark.parametrize("field,value", [
    ("alive", False), ("matching_command", False), ("waiting", False),
    ("gpu_selected", True), ("model_owner", True), ("provider_calls", 1),
    ("provider_ledger", True), ("result", True),
])
def test_resource_successor_never_retires_allocated_or_paid_work(field, value):
    from scripts.relocate_aime_202060 import can_retire

    assert can_retire(unused_waiter())
    assert not can_retire({**unused_waiter(), field: value})
    with pytest.raises(ValueError):
        can_retire({})


def test_resource_successors_rewrite_learning_and_freeze_dependencies():
    from scripts.continue_aime_202060 import next_attempt, path, rewrite_dependencies
    from scripts.relocate_aime_202060 import mode_arguments

    replacements = {"train1-v2": "train1-v3", "chisel-learn-v1": "chisel-learn-v2"}
    command = ["--baseline-root", path("train1-v2"), "--library-root", path("chisel-learn-v1")]
    assert rewrite_dependencies(command, replacements) == [
        "--baseline-root", path("train1-v3"), "--library-root", path("chisel-learn-v2"),
    ]
    assert path("train1-v2", replacements) == path("train1-v3")
    assert next_attempt("train1-v2") == "train1-v3"
    with pytest.raises(ValueError):
        next_attempt("unversioned")
    assert mode_arguments({"mode": "train_baseline", "training_shards": {"index": 0}}, None) == [
        "--shard", "0",
    ]


def test_native_schema_tightening_preserves_bounds_and_rejects_control_bytes():
    from jsonschema import ValidationError
    from phiagent.harness.aime_202060 import (
        diagnosis_schema, parse_structured_response, strict_text_schema,
    )

    original = diagnosis_schema()
    schema = strict_text_schema(original)
    assert "pattern" not in original["properties"]["diagnosis"]
    assert schema["properties"]["rules"]["maxItems"] == 3
    assert schema["properties"]["rules"]["items"]["minLength"] == 20
    assert schema["additionalProperties"] is False
    assert strict_text_schema(schema) == schema
    good = {"diagnosis": "Synthetic diagnosis.", "rules": ["Check all stated boundary conditions."]}

    def response(text):
        return {"choices": [{"message": {"content": text}}]}

    assert parse_structured_response(response(json.dumps(good)), schema) == good
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_response(
            response('{"diagnosis":"literal\ttab","rules":["Check all stated boundary conditions."]}'),
            schema,
        )
    with pytest.raises(ValidationError):
        parse_structured_response(response(json.dumps({**good, "diagnosis": "escaped\ttab"})), schema)
    with pytest.raises(ValidationError):
        parse_structured_response(response(json.dumps({**good, "rules": ["short"]})), schema)
    with pytest.raises(ValidationError):
        parse_structured_response(response(json.dumps({**good, "fabricated": True})), schema)
    with pytest.raises(ValueError, match="overwrite"):
        strict_text_schema({"type": "string", "pattern": "existing"})


def test_repaired_pipeline_selects_on_full_validation_without_rerunning_train():
    from scripts.continue_aime_202060 import STAGES, ready_stages

    assert all(stage[1] != "chisel_train" for stage in STAGES)
    ready = ready_stages(
        {"chisel-learn-v2", "no-skill-val-v1", "train0-v3", "train1-v2"},
        {}, {},
    )
    assert {stage[1] for stage in ready} == {"chisel_validation"}
    assert {stage[0] for stage in ready} == {"chisel-val0-v1", "chisel-val1-v1"}
