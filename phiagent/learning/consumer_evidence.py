"""Committed consumers of historical public evidence, not semantic-necessity certificates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence

from phiagent.learning.evidence_contracts import (
    Answer, CapsuleEntry, Obligation, Observation, PublicHistory, QueryScope, RepairPlan,
    canonical_json, capsule_answer, encoded_capsule, plan_context_repair,
)


VERSION = "consumer-bound-historical-evidence/v1"
MODES = ("bound_checked", "bound_lookup", "advisory_lookup", "bound_typed_recency")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Selection:
    value_json: str | None
    reason: str


@dataclass(frozen=True)
class HistoricalSelector:
    kind: str
    pointer: str
    prefix: str = ""
    suffix: str = ""
    occurrence: int = 0

    def __post_init__(self):
        if self.kind not in ("json_pointer", "exact_span"):
            raise ValueError("only literal JSON pointers and exact historical spans are supported")
        if not isinstance(self.pointer, str) or (self.pointer and not self.pointer.startswith("/")):
            raise ValueError("a selector requires an explicit JSON pointer")
        if re.search(r"~(?![01])", self.pointer):
            raise ValueError("JSON pointer escapes must be ~0 or ~1")
        if type(self.occurrence) is not int or self.occurrence < 0:
            raise ValueError("span occurrences are zero-based nonnegative integers")
        if not isinstance(self.prefix, str) or not isinstance(self.suffix, str):
            raise ValueError("selector delimiters must be literal strings")
        if self.kind == "json_pointer":
            if self.prefix or self.suffix or self.occurrence:
                raise ValueError("JSON field selection must not silently apply text interpretation")
        elif (not self.prefix or not self.suffix
              or max(len(self.prefix.encode()), len(self.suffix.encode())) > 128):
            raise ValueError("exact spans require nonempty literal delimiters of at most 128 bytes")

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "pointer": self.pointer, "prefix": self.prefix,
            "suffix": self.suffix, "occurrence": self.occurrence,
        }

    def select(self, payload: object) -> Selection:
        value = payload
        parts = self.pointer.split("/")[1:] if self.pointer else ()
        for part in parts:
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif (isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key)
                  and int(key) < len(value)):
                value = value[int(key)]
            else:
                return Selection(None, "UNKNOWN_INTERPRETATION_POINTER")
        if self.kind == "json_pointer":
            return Selection(canonical_json(value), "historical_json_value")
        if not isinstance(value, str):
            return Selection(None, "UNKNOWN_INTERPRETATION_NOT_TEXT")
        start = 0
        for _ in range(self.occurrence + 1):
            found = value.find(self.prefix, start)
            if found < 0:
                return Selection(None, "UNKNOWN_INTERPRETATION_PREFIX")
            start = found + len(self.prefix)
        end = value.find(self.suffix, start)
        if end < 0:
            return Selection(None, "UNKNOWN_INTERPRETATION_SUFFIX")
        return Selection(canonical_json(value[start:end]), "literal_historical_span")


@dataclass(frozen=True)
class PublicReport:
    sequence: int
    payload_json: str

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("public report identities are nonnegative chronological integers")
        if canonical_json(json.loads(self.payload_json)) != self.payload_json:
            raise ValueError("public report payloads must be canonical JSON")

    @property
    def event_id(self) -> str:
        return f"public-report-{self.sequence:06d}"


@dataclass(frozen=True)
class PublicAction:
    tool: str
    arguments_json: str

    def __post_init__(self):
        if not self.tool or not isinstance(json.loads(self.arguments_json), dict):
            raise ValueError("a pending action needs a registered tool and argument object")
        if canonical_json(json.loads(self.arguments_json)) != self.arguments_json:
            raise ValueError("pending action arguments must be canonical JSON")

    def as_dict(self) -> dict:
        return {"tool": self.tool, "arguments": json.loads(self.arguments_json)}


@dataclass(frozen=True)
class CommittedBranch:
    report_sequence: int
    selector: HistoricalSelector
    equals_json: str
    when_equal: PublicAction
    when_different: PublicAction
    on_unknown_observe: PublicAction | None

    def __post_init__(self):
        if type(self.report_sequence) is not int or self.report_sequence < 0:
            raise ValueError("the consumer must name one historical public report")
        if canonical_json(json.loads(self.equals_json)) != self.equals_json:
            raise ValueError("the declared comparison must be canonical JSON")
        if self.when_equal == self.when_different:
            raise ValueError("an evidence consumer must declare two different pending actions")

    def as_dict(self) -> dict:
        return {
            "schema_version": VERSION, "report_sequence": self.report_sequence,
            "selector": self.selector.as_dict(), "equals": json.loads(self.equals_json),
            "when_equal": self.when_equal.as_dict(), "when_different": self.when_different.as_dict(),
            "on_unknown": (
                {"observe": self.on_unknown_observe.as_dict()}
                if self.on_unknown_observe is not None else "REPLAN"
            ),
            "scope": "historical_report_not_current_world_truth",
        }

    @property
    def sha256(self) -> str:
        return digest(self.as_dict())

    @property
    def query(self) -> Obligation:
        return Obligation(
            "consumer:" + self.sha256, "selector:" + digest(self.selector.as_dict()),
            QueryScope.EVENT, event_id=f"public-report-{self.report_sequence:06d}",
        )


def commit_branch(
    value: Mapping[str, object],
    *,
    validate_action: Callable[[Mapping[str, object]], PublicAction],
) -> CommittedBranch:
    """Validate and freeze the consumer before the resolver receives any archive."""
    fields = {"report_sequence", "selector", "equals", "when_equal", "when_different", "on_unknown"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("consumer requests require exactly the frozen branch-interface fields")
    selector = value["selector"]
    if not isinstance(selector, Mapping) or set(selector) != {"kind", "pointer", "prefix", "suffix", "occurrence"}:
        raise ValueError("the literal selector must be fully specified before archive lookup")
    unknown = value["on_unknown"]
    if unknown == "REPLAN":
        observe = None
    elif isinstance(unknown, Mapping) and set(unknown) == {"observe"}:
        observe = validate_action(unknown["observe"])
    else:
        raise ValueError("UNKNOWN requires REPLAN or one explicit, charged public observation")
    return CommittedBranch(
        value["report_sequence"], HistoricalSelector(**selector), canonical_json(value["equals"]),
        validate_action(value["when_equal"]), validate_action(value["when_different"]), observe,
    )


def project_reports(
    reports: Sequence[PublicReport], request: CommittedBranch,
) -> tuple[PublicHistory, Selection | None]:
    ids = [report.sequence for report in reports]
    if ids != sorted(set(ids)):
        raise ValueError("public report sequences must be unique and increasing")
    matching = [report for report in reports if report.sequence == request.report_sequence]
    if not matching:
        return PublicHistory(()), None
    report = matching[0]
    selection = request.selector.select(json.loads(report.payload_json))
    observed = Observation(
        report.event_id, report.sequence, request.query.key, selection.value_json,
        hashlib.sha256(report.payload_json.encode()).hexdigest(),
    )
    return PublicHistory((observed,)), selection


@dataclass(frozen=True)
class ConsumerResolution:
    status: str
    request_sha256: str
    answer: Answer
    action: PublicAction | None
    consumer_bound: bool
    requires_charged_observation: bool
    repair: RepairPlan
    answer_capsule: str
    action_selection_is_not_semantic_necessity: bool = True


def resolve_branch(
    request: CommittedBranch,
    *,
    reports: Sequence[PublicReport],
    visible_reports: Sequence[PublicReport],
    previous_repair: Sequence[CapsuleEntry] = (),
    budget_bytes: int,
    mode: str,
    acquisition_enabled: bool = True,
) -> ConsumerResolution:
    if mode not in MODES:
        raise ValueError("unsupported consumer-bound evidence mode")
    if type(acquisition_enabled) is not bool:
        raise ValueError("acquisition must be explicitly enabled or disabled")
    history, selected = project_reports(reports, request)
    visible, _ = project_reports(visible_reports, request)
    plan = plan_context_repair(
        history, visible, previous_repair, (request.query,), budget_bytes=budget_bytes,
    )
    full_answer = history.answer(request.query)
    answer = full_answer
    if (mode == "bound_typed_recency" and full_answer.known
            and visible.answer(request.query) != full_answer
            and request.report_sequence not in {report.sequence for report in reports[-2:]}):
        answer = Answer(False, None, (), "outside_frozen_two_report_recency")
    capsule = encoded_capsule((CapsuleEntry(request.query.query_id, answer),))
    if (mode == "bound_checked" and not plan.admitted) or len(capsule.encode()) > budget_bytes:
        return ConsumerResolution(
            "BUDGET_REFUSED_REPLAN", request.sha256, Answer(False, None, (), "explicit_budget_refusal"),
            None, False, False, plan, "",
        )
    if not answer.known:
        reason = (
            "UNKNOWN_OUTSIDE_FIXED_RECENCY" if full_answer.known
            else selected.reason if selected is not None else "UNKNOWN_NOT_OBSERVED"
        )
        acquire = request.on_unknown_observe if acquisition_enabled else None
        return ConsumerResolution(
            reason, request.sha256, answer, acquire, False,
            acquire is not None, plan, capsule,
        )
    if mode == "bound_checked":
        answer = (
            capsule_answer(plan.entries, request.query.query_id)
            if plan.entries else visible.answer(request.query)
        )
        if answer != full_answer:
            raise AssertionError("the committed consumer's historical value was not preserved")
    action = request.when_equal if answer.value_json == request.equals_json else request.when_different
    advisory = mode == "advisory_lookup"
    return ConsumerResolution(
        "ADVISORY_REPORT_ONLY" if advisory else "COMMITTED_BRANCH_RESOLVED",
        request.sha256, answer, None if advisory else action, not advisory, False, plan, capsule,
    )
