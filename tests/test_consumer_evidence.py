"""Data-free consumer-contract tests; no environment, model or benchmark access."""

from copy import deepcopy

import pytest

from phiagent.learning.consumer_evidence import (
    HistoricalSelector, PublicAction, PublicReport, commit_branch, resolve_branch,
)
from phiagent.learning.evidence_contracts import canonical_json


def validate_action(value):
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        raise ValueError("synthetic registered tool requires explicit arguments")
    if value["tool"] != "synthetic_public_tool" or set(value["arguments"]) != {"operation"}:
        raise ValueError("only the registered synthetic action is supported")
    return PublicAction(value["tool"], canonical_json(value["arguments"]))


def raw_request():
    return {
        "report_sequence": 0,
        "selector": {"kind": "exact_span", "pointer": "/text", "prefix": "status=", "suffix": "\n", "occurrence": 0},
        "equals": "complete",
        "when_equal": {"tool": "synthetic_public_tool", "arguments": {"operation": "inspect-output"}},
        "when_different": {"tool": "synthetic_public_tool", "arguments": {"operation": "inspect-progress"}},
        "on_unknown": "REPLAN",
    }


def request():
    return commit_branch(raw_request(), validate_action=validate_action)


def report(text, sequence=0):
    return PublicReport(sequence, canonical_json({"text": text}))


def resolve(value=None, *, reports=None, visible=(), mode="bound_checked", budget=2304):
    return resolve_branch(
        value or request(), reports=reports if reports is not None else (report("status=complete\n"),),
        visible_reports=visible, budget_bytes=budget, mode=mode,
    )


def test_pending_plan_is_frozen_before_archive_access_and_retrieved_text_never_becomes_code():
    raw = raw_request()
    committed = commit_branch(raw, validate_action=validate_action)
    before = committed.sha256
    raw["when_equal"]["arguments"]["operation"] = "changed-after-commit"
    assert committed.sha256 == before
    resolved = resolve(committed, reports=(report("status=arbitrary executable-looking text\n"),))
    assert resolved.action == committed.when_different
    assert resolved.action.as_dict()["arguments"] == {"operation": "inspect-progress"}
    assert "arbitrary" not in resolved.action.arguments_json
    assert resolved.action_selection_is_not_semantic_necessity


def test_missing_visible_support_is_restored_before_bound_branch_selection():
    resolved = resolve()
    assert resolved.status == "COMMITTED_BRANCH_RESOLVED"
    assert resolved.action == request().when_equal
    assert resolved.consumer_bound
    assert [edit.kind.value for edit in resolved.repair.edits] == ["ADD"]
    assert "retention_or_representation_gap" in {gap.value for _, gap in resolved.repair.gaps}


def test_already_visible_value_does_not_count_as_a_retention_repair():
    item = report("status=complete\n")
    resolved = resolve(reports=(item,), visible=(item,))
    assert not resolved.repair.edits and not resolved.repair.entries
    assert resolved.action == request().when_equal


def test_unknown_is_not_the_false_branch_and_does_not_automatically_acquire():
    unresolved = resolve(reports=())
    assert unresolved.status == "UNKNOWN_NOT_OBSERVED"
    assert unresolved.action is None and not unresolved.answer.known
    assert not unresolved.requires_charged_observation
    uninterpretable = resolve(reports=(report("no declared field"),))
    assert uninterpretable.status == "UNKNOWN_INTERPRETATION_PREFIX"
    assert uninterpretable.action is None


def test_explicit_observation_branch_is_separate_from_a_retention_edit():
    raw = raw_request()
    raw["on_unknown"] = {"observe": {
        "tool": "synthetic_public_tool", "arguments": {"operation": "observe-status"},
    }}
    committed = commit_branch(raw, validate_action=validate_action)
    unresolved = resolve(committed, reports=())
    assert unresolved.requires_charged_observation
    assert not unresolved.consumer_bound
    assert unresolved.action == committed.on_unknown_observe
    assert unresolved.repair.edits == ()
    disabled = resolve_branch(
        committed, reports=(), visible_reports=(), budget_bytes=2304, mode="bound_checked",
        acquisition_enabled=False,
    )
    assert disabled.action is None and not disabled.requires_charged_observation


def test_literal_span_delimiters_and_occurrence_are_frozen_not_deletion_directed():
    selector = HistoricalSelector("exact_span", "/text", "id=", "\n", 1)
    assert selector.select({"text": "id=first\nid=second\n"}).value_json == '"second"'
    assert selector.select({"text": "id=first\n"}).value_json is None
    assert selector.select({"text": "id=first\nid=no-ending"}).value_json is None


def test_json_pointer_keeps_null_false_zero_and_escaped_keys_distinct():
    selector = HistoricalSelector("json_pointer", "/a~1b/~0key/1")
    assert selector.select({"a/b": {"~key": [0, None]}}).value_json == "null"
    assert HistoricalSelector("json_pointer", "/x").select({"x": False}).value_json == "false"
    assert HistoricalSelector("json_pointer", "/x").select({"x": 0}).value_json == "0"
    assert HistoricalSelector("json_pointer", "/missing").select({}).value_json is None
    with pytest.raises(ValueError, match="escapes"):
        HistoricalSelector("json_pointer", "/bad~2")
    with pytest.raises(ValueError, match="literal strings"):
        HistoricalSelector("json_pointer", "/x", prefix=None)


def test_false_report_is_known_not_unknown_and_does_not_equal_numeric_zero():
    raw = raw_request()
    raw["selector"] = {"kind": "json_pointer", "pointer": "/x", "prefix": "", "suffix": "", "occurrence": 0}
    raw["equals"] = 0
    committed = commit_branch(raw, validate_action=validate_action)
    resolved = resolve(committed, reports=(PublicReport(0, canonical_json({"x": False})),))
    assert resolved.answer.known
    assert resolved.action == committed.when_different


def test_budget_refusal_neither_truncates_evidence_nor_executes_the_unknown_branch():
    raw = raw_request()
    raw["on_unknown"] = {"observe": deepcopy(raw["when_different"])}
    committed = commit_branch(raw, validate_action=validate_action)
    large = report("status=" + "x" * 4000 + "\n")
    refused = resolve(committed, reports=(large,))
    assert refused.status == "BUDGET_REFUSED_REPLAN"
    assert refused.action is None and not refused.requires_charged_observation
    assert refused.answer_capsule == ""
    assert not refused.answer.known


def test_advisory_control_returns_identical_evidence_without_executing_a_consumer():
    bound = resolve()
    advisory = resolve(mode="advisory_lookup")
    assert advisory.answer_capsule == bound.answer_capsule
    assert advisory.answer == bound.answer
    assert advisory.action is None and not advisory.consumer_bound
    assert advisory.status == "ADVISORY_REPORT_ONLY"


@pytest.mark.parametrize("text", ["status=complete\n", "status=running\n", "no parsed status"])
def test_plain_lookup_and_checker_have_no_invented_quality_difference_on_equal_answers(text):
    item = report(text)
    checked = resolve(reports=(item,))
    ordinary = resolve(reports=(item,), mode="bound_lookup")
    assert checked.action == ordinary.action
    assert checked.answer == ordinary.answer
    assert checked.answer_capsule == ordinary.answer_capsule


def test_same_branch_or_unregistered_actions_do_not_establish_a_consumer():
    raw = raw_request()
    raw["when_different"] = deepcopy(raw["when_equal"])
    with pytest.raises(ValueError, match="different"):
        commit_branch(raw, validate_action=validate_action)
    raw = raw_request()
    raw["when_equal"]["tool"] = "unregistered"
    with pytest.raises(ValueError, match="registered"):
        commit_branch(raw, validate_action=validate_action)


def test_report_order_and_future_report_absence_are_not_silent_defaults():
    with pytest.raises(ValueError, match="unique and increasing"):
        resolve(reports=(report("a", 1), report("b", 0)))
    raw = raw_request()
    raw["report_sequence"] = 10
    assert resolve(commit_branch(raw, validate_action=validate_action)).status == "UNKNOWN_NOT_OBSERVED"


def test_fixed_recency_does_not_retrieve_old_missing_values_or_forget_visible_values():
    reports = (report("status=complete\n"), report("second", 1), report("third", 2))
    absent = resolve(reports=reports, mode="bound_typed_recency")
    assert absent.status == "UNKNOWN_OUTSIDE_FIXED_RECENCY"
    assert not absent.answer.known and absent.action is None
    assert "complete" not in absent.answer_capsule
    visible = resolve(reports=reports, visible=(reports[0],), mode="bound_typed_recency")
    assert visible.answer.known and visible.action == request().when_equal
    recent = resolve(reports=reports[:2], mode="bound_typed_recency")
    assert recent.answer.known and recent.action == request().when_equal
