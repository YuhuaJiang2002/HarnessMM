"""Data-free contract algebra tests; no benchmark/model/environment execution."""

from itertools import product

import pytest

from phiagent.learning.evidence_contracts import (
    Answer, CapsuleEntry, Edit, EditKind, Gap, Invalidation, Observation, Obligation,
    PublicHistory, QueryScope, apply_edit, apply_transaction, canonical_json, capsule_sha256, diagnose,
    plan_context_repair,
)


def observation(identity="e1", sequence=1, key="tool-report", value=True, **kwargs):
    return Observation(identity, sequence, key, canonical_json(value), "a" * 64, **kwargs)


def test_retention_is_not_acquisition_or_unverified_freshness():
    history = PublicHistory([observation()])
    query = Obligation("q", "tool-report", QueryScope.LATEST)
    assert diagnose(history, (), query) == Gap.RETENTION
    assert diagnose(history, (), Obligation("missing", "other", QueryScope.LATEST)) == Gap.ACQUISITION
    assert diagnose(history, (), Obligation("live", "tool-report", QueryScope.LIVE)) == Gap.FRESHNESS
    edited = apply_edit(history, (), (query,), Edit(EditKind.ADD, "q"), budget_bytes=2000)
    assert edited.admitted and diagnose(history, edited.entries, query) == Gap.PRESERVED


def test_unknown_is_not_false_and_add_cannot_acquire_an_observation_for_free():
    with pytest.raises(ValueError, match="UNKNOWN"):
        Answer(False, "false", (), "unknown")
    history = PublicHistory([])
    query = Obligation("q", "missing", QueryScope.LATEST)
    result = apply_edit(history, (), (query,), Edit(EditKind.ADD, "q"), budget_bytes=2000)
    assert not result.admitted
    assert result.reason == "acquisition_or_freshness_required_not_a_free_memory_add"
    assert result.entries == ()


def test_delete_never_drops_a_mandatory_known_witness():
    history = PublicHistory([observation()])
    query = Obligation("q", "tool-report", QueryScope.LATEST)
    entries = (CapsuleEntry("q", history.answer(query)),)
    result = apply_edit(history, entries, (query,), Edit(EditKind.DELETE, "q"), budget_bytes=2000)
    assert not result.admitted and result.entries == entries
    retired = apply_edit(history, entries, (), Edit(EditKind.DELETE, "q"), budget_bytes=2000)
    assert retired.admitted and retired.entries == ()


def test_atomic_adds_bootstrap_all_mandatory_queries_without_intermediate_publication():
    history = PublicHistory([observation(key="a"), observation("e2", 2, "b", False)])
    queries = (Obligation("a", "a", QueryScope.LATEST), Obligation("b", "b", QueryScope.LATEST))
    result = apply_transaction(
        history, (), queries, (Edit(EditKind.ADD, "a"), Edit(EditKind.ADD, "b")), budget_bytes=2000,
    )
    assert result.admitted and all(diagnose(history, result.entries, query) == Gap.PRESERVED for query in queries)
    blocked = apply_transaction(
        history, (), queries, (Edit(EditKind.ADD, "a"), Edit(EditKind.ADD, "b")), budget_bytes=1,
    )
    assert not blocked.admitted and blocked.entries == ()
    assert "budget_blocked" in blocked.reason


def test_live_invalidation_does_not_destroy_historical_evidence():
    event = observation(version_domain="snapshot", version="v1")
    history = PublicHistory([event])
    live = Obligation("live", event.key, QueryScope.LIVE, version_domain="snapshot", current_tool_version="v1")
    historical = Obligation("old", event.key, QueryScope.EVENT, event_id="e1")
    entries = (CapsuleEntry("live", history.answer(live)), CapsuleEntry("old", history.answer(historical)))
    changed = PublicHistory([event], [Invalidation(2, "snapshot", "unobserved write may affect the live claim")])
    assert diagnose(changed, entries, live) == Gap.FRESHNESS
    assert diagnose(changed, entries, historical) == Gap.PRESERVED
    result = apply_edit(changed, entries, (live, historical), Edit(EditKind.MODIFY, "live"), budget_bytes=2000)
    assert result.admitted
    assert not next(item.answer for item in result.entries if item.query_id == "live").known
    assert changed.answer(historical).known


def test_contradiction_is_preserved_not_rewritten_as_requirement_success():
    history = PublicHistory([observation(value=False)])
    query = Obligation("q", "tool-report", QueryScope.LATEST, required_value_json="true")
    result = apply_edit(history, (), (query,), Edit(EditKind.ADD, "q"), budget_bytes=2000)
    assert result.admitted and diagnose(history, result.entries, query) == Gap.CONTRADICTION
    assert "contradicted" in result.reason
    assert result.entries[0].answer.value_json == "false"


def test_admitted_capsules_separate_distinct_synthetic_query_vectors():
    encodings = set()
    queries = (Obligation("a", "a", QueryScope.LATEST), Obligation("b", "b", QueryScope.LATEST))
    for left, right in product((False, True), repeat=2):
        history = PublicHistory([observation(key="a", value=left), observation("e2", 2, "b", right)])
        result = apply_transaction(
            history, (), queries, (Edit(EditKind.ADD, "a"), Edit(EditKind.ADD, "b")), budget_bytes=2000,
        )
        assert result.admitted
        encodings.add(capsule_sha256(result.entries))
    assert len(encodings) == 4


def test_event_ids_and_obligation_definitions_cannot_be_duplicated():
    with pytest.raises(ValueError, match="not be reused"):
        PublicHistory([observation(), observation(sequence=2)])
    history = PublicHistory([observation()])
    query = Obligation("q", "tool-report", QueryScope.LATEST)
    with pytest.raises(ValueError, match="unique"):
        apply_edit(history, (), (query, query), Edit(EditKind.ADD, "q"), budget_bytes=2000)


def test_context_repair_adds_modifies_and_deletes_without_mutating_query_definitions():
    query = Obligation("q", "tool-report", QueryScope.LATEST)
    first = PublicHistory([observation(value="first")])
    empty = PublicHistory([])
    added = plan_context_repair(first, empty, (), (query,), budget_bytes=2000)
    assert added.admitted and added.edits == (Edit(EditKind.ADD, "q"),)
    second = PublicHistory([observation(value="first"), observation("e2", 2, value="second")])
    modified = plan_context_repair(second, empty, added.entries, (query,), budget_bytes=2000)
    assert modified.admitted and modified.edits == (Edit(EditKind.MODIFY, "q"),)
    restored = plan_context_repair(second, second, modified.entries, (query,), budget_bytes=2000)
    assert restored.admitted and restored.edits == (Edit(EditKind.DELETE, "q"),)
    assert restored.entries == () and restored.encoded_bytes == 0


def test_context_repair_distinguishes_missing_history_from_lost_visible_history():
    query = Obligation("q", "tool-report", QueryScope.LATEST)
    unknown = PublicHistory([])
    result = plan_context_repair(unknown, unknown, (), (query,), budget_bytes=2000)
    assert result.admitted and not result.entries
    assert result.gaps == (("q", Gap.ACQUISITION),)
    assert "unresolved" in result.reason


def test_context_repair_invalidates_a_stale_visible_live_answer():
    event = observation(version_domain="tool-snapshot", version="v1")
    visible = PublicHistory([event])
    history = PublicHistory([event], [Invalidation(2, "tool-snapshot", "explicit unknown mutation")])
    query = Obligation("q", "tool-report", QueryScope.LIVE,
                       version_domain="tool-snapshot", current_tool_version="v1")
    result = plan_context_repair(history, visible, (), (query,), budget_bytes=2000)
    assert result.admitted and result.gaps == (("q", Gap.FRESHNESS),)
    assert result.entries[0].answer.known is False and result.entries[0].answer.value_json is None


def test_context_repair_does_not_hide_budget_failure_by_deleting_an_obligation():
    history = PublicHistory([observation(value="x" * 200)])
    query = Obligation("q", "tool-report", QueryScope.LATEST)
    result = plan_context_repair(history, PublicHistory([]), (), (query,), budget_bytes=10)
    assert not result.admitted and result.edits == ()
    assert result.gaps == (("q", Gap.RETENTION),)
