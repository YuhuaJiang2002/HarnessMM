"""Trace-relative query preservation, not world-truth or action-optimality certification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from collections.abc import Iterable


VERSION = "public-evidence-contracts/v1"


class QueryScope(str, Enum):
    EVENT = "historical_event"
    LATEST = "latest_observed_report"
    LIVE = "tool_versioned_current_report"


class Gap(str, Enum):
    PRESERVED = "support_preserved"
    RETENTION = "retention_or_representation_gap"
    ACQUISITION = "acquisition_or_interpretation_gap"
    FRESHNESS = "freshness_or_invalidation_gap"
    CONTRADICTION = "observed_requirement_contradiction"


class EditKind(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Observation:
    event_id: str
    sequence: int
    key: str
    value_json: str | None
    public_source_sha256: str
    version_domain: str | None = None
    version: str | None = None

    def __post_init__(self):
        if not self.event_id or not self.key or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("public observations require a nonempty identity/key and nonnegative sequence")
        if re.fullmatch(r"[0-9a-f]{64}", self.public_source_sha256) is None:
            raise ValueError("public source content hash is required")
        if self.value_json is not None and canonical_json(json.loads(self.value_json)) != self.value_json:
            raise ValueError("observation values must use canonical JSON")
        if (self.version_domain is None) != (self.version is None):
            raise ValueError("version domain and tool-owned version must be supplied together")


@dataclass(frozen=True)
class Invalidation:
    sequence: int
    domain: str
    reason: str

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0 or not self.domain or not self.reason:
            raise ValueError("invalidation requires explicit sequence, domain and reason")


@dataclass(frozen=True)
class Obligation:
    query_id: str
    key: str
    scope: QueryScope
    event_id: str | None = None
    version_domain: str | None = None
    current_tool_version: str | None = None
    required_value_json: str | None = None

    def __post_init__(self):
        if not self.query_id or not self.key or not isinstance(self.scope, QueryScope):
            raise ValueError("obligations require a frozen query identity, key and supported scope")
        if (self.scope == QueryScope.EVENT) != (self.event_id is not None):
            raise ValueError("historical-event queries require exactly one event identity")
        if self.scope != QueryScope.LIVE and (self.version_domain is not None or self.current_tool_version is not None):
            raise ValueError("historical queries must not silently acquire live-state semantics")
        if self.required_value_json is not None and canonical_json(json.loads(self.required_value_json)) != self.required_value_json:
            raise ValueError("required values must use canonical JSON")


@dataclass(frozen=True)
class Answer:
    known: bool
    value_json: str | None
    witness_ids: tuple[str, ...]
    reason: str

    def __post_init__(self):
        if type(self.known) is not bool or not self.reason:
            raise ValueError("answers need an explicit knowledge state and reason")
        if self.known:
            if (self.value_json is None or not self.witness_ids
                    or canonical_json(json.loads(self.value_json)) != self.value_json):
                raise ValueError("known answers require a canonical value and public witnesses")
        elif self.value_json is not None or self.witness_ids:
            raise ValueError("UNKNOWN must not be represented as false or as an unsupported value")


@dataclass(frozen=True)
class CapsuleEntry:
    query_id: str
    answer: Answer

    def __post_init__(self):
        if not self.query_id:
            raise ValueError("capsule entries require a query identity")


@dataclass(frozen=True)
class Edit:
    kind: EditKind
    query_id: str


@dataclass(frozen=True)
class EditResult:
    admitted: bool
    entries: tuple[CapsuleEntry, ...]
    reason: str
    encoded_bytes: int
    unsupported_queries: tuple[str, ...]


@dataclass(frozen=True)
class RepairPlan:
    admitted: bool
    entries: tuple[CapsuleEntry, ...]
    edits: tuple[Edit, ...]
    gaps: tuple[tuple[str, Gap], ...]
    encoded_bytes: int
    reason: str


class PublicHistory:
    def __init__(self, observations: Iterable[Observation], invalidations: Iterable[Invalidation] = ()):
        self.observations = tuple(observations)
        self.invalidations = tuple(invalidations)
        ids = [item.event_id for item in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("public event identities must not be reused")
        sequences = [item.sequence for item in self.observations]
        if len(set(sequences)) != len(sequences) or sequences != sorted(sequences):
            raise ValueError("public observations must have unique increasing sequence numbers")

    def answer(self, query: Obligation) -> Answer:
        candidates = [item for item in self.observations if item.key == query.key]
        if query.scope == QueryScope.EVENT:
            candidates = [item for item in candidates if item.event_id == query.event_id]
        if not candidates:
            return Answer(False, None, (), "no_matching_public_observation")
        observed = candidates[-1]
        if observed.value_json is None:
            return Answer(False, None, (), "unsupported_public_report")
        if query.scope == QueryScope.LIVE:
            if (not query.version_domain or not query.current_tool_version
                    or observed.version_domain != query.version_domain or observed.version is None):
                return Answer(False, None, (), "no_tool_owned_freshness_contract")
            if observed.version != query.current_tool_version:
                return Answer(False, None, (), "tool_version_mismatch")
            if any(event.domain == query.version_domain and event.sequence > observed.sequence
                   for event in self.invalidations):
                return Answer(False, None, (), "later_public_invalidation")
        return Answer(True, observed.value_json, (observed.event_id,), "trace_relative_report")


def capsule_answer(entries: Iterable[CapsuleEntry], query_id: str) -> Answer:
    matching = [entry for entry in entries if entry.query_id == query_id]
    if len(matching) > 1:
        raise ValueError("capsule query identities must be unique")
    return matching[0].answer if matching else Answer(False, None, (), "no_capsule_entry")


def diagnose(history: PublicHistory, entries: Iterable[CapsuleEntry], query: Obligation) -> Gap:
    full = history.answer(query)
    retained = capsule_answer(entries, query.query_id)
    if full.known and query.required_value_json is not None and full.value_json != query.required_value_json:
        return Gap.CONTRADICTION
    if not full.known:
        if full.reason in ("no_tool_owned_freshness_contract", "tool_version_mismatch", "later_public_invalidation"):
            return Gap.FRESHNESS
        return Gap.ACQUISITION
    return Gap.PRESERVED if retained == full else Gap.RETENTION


def encoded_capsule(entries: Iterable[CapsuleEntry]) -> str:
    values = sorted(entries, key=lambda entry: entry.query_id)
    if len({entry.query_id for entry in values}) != len(values):
        raise ValueError("capsule query identities must be unique")
    return canonical_json({
        "schema_version": VERSION,
        "scope": "untrusted public reports, not current-world truth or optimal actions",
        "answers": [{
            "query_id": item.query_id, "known": item.answer.known, "value": (
                json.loads(item.answer.value_json) if item.answer.value_json is not None else None
            ), "witness_ids": list(item.answer.witness_ids), "reason": item.answer.reason,
        } for item in values],
    })


def apply_edit(
    history: PublicHistory,
    entries: Iterable[CapsuleEntry],
    obligations: Iterable[Obligation],
    edit: Edit,
    *,
    budget_bytes: int,
) -> EditResult:
    return apply_transaction(history, entries, obligations, (edit,), budget_bytes=budget_bytes)


def apply_transaction(
    history: PublicHistory,
    entries: Iterable[CapsuleEntry],
    obligations: Iterable[Obligation],
    edits: Iterable[Edit],
    *,
    budget_bytes: int,
) -> EditResult:
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("a positive explicit capsule byte budget is required")
    before = tuple(entries)
    queries = tuple(obligations)
    by_query = {query.query_id: query for query in queries}
    if len(by_query) != len(queries):
        raise ValueError("mandatory obligations must be unique and cannot be dropped by the edit")
    encoded_capsule(before)
    existing = {item.query_id: item for item in before}
    for edit in edits:
        if not isinstance(edit.kind, EditKind):
            raise ValueError("unsupported edit operation")
        if edit.kind == EditKind.ADD:
            if edit.query_id in existing or edit.query_id not in by_query:
                raise ValueError("ADD requires an absent declared obligation")
            answer = history.answer(by_query[edit.query_id])
            if not answer.known:
                return EditResult(False, before, "acquisition_or_freshness_required_not_a_free_memory_add",
                                  len(encoded_capsule(before).encode()), (edit.query_id,))
            existing[edit.query_id] = CapsuleEntry(edit.query_id, answer)
        elif edit.kind == EditKind.MODIFY:
            if edit.query_id not in existing or edit.query_id not in by_query:
                raise ValueError("MODIFY requires an existing declared obligation")
            existing[edit.query_id] = CapsuleEntry(edit.query_id, history.answer(by_query[edit.query_id]))
        else:
            if edit.query_id not in existing:
                raise ValueError("DELETE requires an existing capsule entry")
            del existing[edit.query_id]
    candidate = tuple(existing[key] for key in sorted(existing))
    mismatches = []
    for query in queries:
        expected = history.answer(query)
        observed = capsule_answer(candidate, query.query_id)
        if expected.known:
            if observed != expected:
                mismatches.append(query.query_id)
        elif observed.known:
            mismatches.append(query.query_id)
    cost = len(encoded_capsule(candidate).encode())
    if mismatches:
        return EditResult(False, before, "mandatory_query_support_not_preserved", cost, tuple(mismatches))
    if cost > budget_bytes:
        return EditResult(False, before, "explicit_budget_blocked_no_obligation_dropped", cost, ())
    contradictions = any(
        diagnose(history, candidate, query) == Gap.CONTRADICTION for query in queries
    )
    return EditResult(
        True, candidate,
        "query_reports_preserved_but_required_values_contradicted" if contradictions else "trace_relative_queries_preserved",
        cost, (),
    )


def capsule_sha256(entries: Iterable[CapsuleEntry]) -> str:
    return hashlib.sha256(encoded_capsule(entries).encode()).hexdigest()


def plan_context_repair(
    history: PublicHistory,
    visible_history: PublicHistory,
    previous_repair: Iterable[CapsuleEntry],
    obligations: Iterable[Obligation],
    *,
    budget_bytes: int,
) -> RepairPlan:
    """Compile a query override, charging its bytes separately from the given context/archive."""
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("a positive explicit additional-capsule budget is required")
    queries = tuple(obligations)
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("mandatory obligation identities must be unique")
    previous = tuple(previous_repair)
    encoded_capsule(previous)
    before = {item.query_id: item for item in previous}
    needed = {}
    gaps = []
    for query in queries:
        full, visible = history.answer(query), visible_history.answer(query)
        visible_entries = (CapsuleEntry(query.query_id, visible),)
        gaps.append((query.query_id, diagnose(history, visible_entries, query)))
        if full.known and visible != full:
            needed[query.query_id] = CapsuleEntry(query.query_id, full)
        elif not full.known and visible.known:
            needed[query.query_id] = CapsuleEntry(query.query_id, full)
    entries = tuple(needed[key] for key in sorted(needed))
    edits = []
    for key in sorted(set(before) | set(needed)):
        if key not in needed:
            edits.append(Edit(EditKind.DELETE, key))
        elif key not in before:
            edits.append(Edit(EditKind.ADD, key))
        elif before[key] != needed[key]:
            edits.append(Edit(EditKind.MODIFY, key))
    cost = len(encoded_capsule(entries).encode()) if entries else 0
    if cost > budget_bytes:
        return RepairPlan(False, previous, (), tuple(gaps), cost, "budget_blocked_no_obligation_dropped")
    for query in queries:
        expected = history.answer(query)
        observed = needed[query.query_id].answer if query.query_id in needed else visible_history.answer(query)
        if (expected.known and observed != expected) or (not expected.known and observed.known):
            raise AssertionError("compiled repair does not preserve the frozen query result")
    unresolved = any(gap in (Gap.ACQUISITION, Gap.FRESHNESS) for _, gap in gaps)
    return RepairPlan(
        True, entries, tuple(edits), tuple(gaps), cost,
        "reports_preserved_with_unresolved_acquisition_or_freshness" if unresolved else "trace_relative_query_reports_preserved",
    )
