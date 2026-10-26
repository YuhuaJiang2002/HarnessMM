"""Training-case abstraction and actionable, case-free proposal feedback.

Observed answer/span relationships are diagnostic hypotheses, not causal
instruction labels. Structural admission does not establish task improvement.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re

from phiagent.learning import qa_joint_optimization as joint
from phiagent.learning import qa_meta_rsi as patches
from phiagent.learning.qa_quality_cost_router import NUMERIC_FEATURES, input_features


SCHEMA = "phiagent-qa-rule-abstraction/v1"
IDENTITY = "CHISEL case-abstracted rule proposal and structured rejection feedback"
PATTERN_SPECS = {
    "overextended_span": {
        "observation": "Incorrect predictions contain an annotated short answer span plus extra words.",
        "hypothesis": "Answer-boundary normalization may be relevant.",
        "runtime_operation": "Use the question and context to retain the required answer span and necessary disambiguators.",
        "sections": ("normalization", "answer_type"),
    },
    "incomplete_span": {
        "observation": "Incorrect predictions omit part of an annotated answer span.",
        "hypothesis": "Required specificity or entity disambiguation may be missing.",
        "runtime_operation": "Check which modifiers and identifiers the question requires before shortening an answer.",
        "sections": ("normalization", "answer_type"),
    },
    "competing_supported_span": {
        "observation": "The incorrect prediction and an annotated answer both occur literally in the supplied context.",
        "hypothesis": "Selecting a context mention without checking all clue constraints may explain the error.",
        "runtime_operation": "Compare candidate mentions against the question's relations, type and explicit constraints.",
        "sections": ("evidence", "answer_type", "traps"),
    },
    "unsupported_prediction": {
        "observation": "An annotated answer occurs literally in context but the incorrect prediction does not.",
        "hypothesis": "A context-grounding check may be relevant.",
        "runtime_operation": "Prefer a context-supported answer that satisfies the question; do not replace missing support with a guess.",
        "sections": ("evidence", "traps"),
    },
}
PUBLIC_HINTS = {
    "TRAINING_ANSWER_LITERAL": "Remove the instance-specific answer or example. Express the operation in terms of the question and supplied context.",
    "TRAINING_QUESTION_PHRASE": "Replace copied clue wording with the abstract relation, answer-type or format check. Do not paraphrase that individual clue.",
    "TASK_ID_LITERAL": "Do not encode task identities in reusable rules.",
    "EVIDENCE_ID_LITERAL": "Keep support identifiers only in the evidence field.",
    "PROTECTED_SURFACE": "Do not change the task interface, evaluator, routing machinery or execution environment.",
    "JSON_SYNTAX": "Return one complete JSON object with the declared patch/applicability schema.",
    "SCHEMA": "Use only the declared object fields and JSON value types.",
    "PARENT_HASH": "Copy the supplied parent binding exactly; do not bind to a previous rejected child.",
    "EVIDENCE": "Cite only supported abstract-pattern IDs from the current evidence packet.",
    "FEATURES": "Choose one to four distinct names from the supplied input-only feature allowlist.",
    "RULE_ID": "Use an existing delete.* rule ID exactly once.",
    "RULE_TEXT": "Use a single '- ' rule line, or an empty string for deletion, within the declared character limit.",
    "NO_CHANGE": "A proposed replacement must change the current parent rule.",
    "EDIT_COUNT": "Return a nonempty patch within the declared edit-count limit.",
    "PROFILE_GROWTH": "Shorten the proposed rules without removing required behavior; the original byte-growth limit still applies.",
    "NATIVE_GROWTH": "Shorten the proposal to fit the unchanged full native-input growth allowance.",
    "DUPLICATE": "Do not resubmit the same content and applicability specification.",
    "COMPILER_REJECTED": "Regenerate a bounded parent-relative edit; the authoritative compiler has not admitted this proposal.",
}


@dataclass(frozen=True)
class Literal:
    text: str
    kind: str
    evidence: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str) or not self.text
            or self.kind not in {
                "TASK_ID_LITERAL", "TRAINING_ANSWER_LITERAL", "TRAINING_QUESTION_PHRASE",
                "EVIDENCE_ID_LITERAL",
            }
            or not isinstance(self.evidence, str) or not self.evidence
        ):
            raise ValueError("guard literals require typed training provenance")


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    kind: str
    evidence: tuple[str, ...]
    groups: tuple[str, ...]
    mean_f1: float
    feature_means: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pattern_id, str) or re.fullmatch(r"A\d{3}", self.pattern_id) is None
            or self.kind not in PATTERN_SPECS
            or not all(isinstance(value, tuple) for value in (self.groups, self.evidence, self.feature_means))
            or len(self.groups) < 2 or len(set(self.groups)) != len(self.groups)
            or len(self.evidence) < len(self.groups) or len(set(self.evidence)) != len(self.evidence)
            or any(not isinstance(value, str) or not value for value in (*self.groups, *self.evidence))
            or type(self.mean_f1) not in (int, float) or not math.isfinite(self.mean_f1)
            or not 0 <= self.mean_f1 <= 1
            or tuple(name for name, _ in self.feature_means) != NUMERIC_FEATURES
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for _, value in self.feature_means)
        ):
            raise ValueError("public patterns require canonical types and independent multi-group support")

    def public(self) -> dict:
        spec = PATTERN_SPECS[self.kind]
        return {
            "partition": "train", "evidence": self.pattern_id,
            "observed_relationship": self.kind,
            "observation": spec["observation"], "hypothesis": spec["hypothesis"],
            "runtime_operation": spec["runtime_operation"],
            "relevant_sections": list(spec["sections"]),
            "support_tasks": len(self.evidence), "independent_source_groups": len(self.groups),
            "mean_recorded_training_f1": self.mean_f1,
            "diagnostic_weighting": "equal_source_groups",
            "observable_feature_means": dict(self.feature_means),
            "causal_defect_established": False,
        }


@dataclass(frozen=True)
class EvidencePacket:
    patterns: tuple[Pattern, ...]
    observations_json: str
    literals: tuple[Literal, ...]
    trace_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.patterns, tuple) or not isinstance(self.literals, tuple)
            or any(not isinstance(pattern, Pattern) for pattern in self.patterns)
            or len({pattern.pattern_id for pattern in self.patterns}) != len(self.patterns)
            or any(not isinstance(literal, Literal) for literal in self.literals)
        ):
            raise ValueError("evidence packets require typed patterns and guard literals")
        trace = json.loads(self.trace_json)
        if (
            trace["schema_version"] != SCHEMA or trace["partition"] != "train"
            or set(trace["patterns"]) != {pattern.pattern_id for pattern in self.patterns}
            or not isinstance(json.loads(self.observations_json), list)
        ):
            raise ValueError("an evidence packet must retain its checked training trace")

    @property
    def evidence_ids(self) -> set[str]:
        return {pattern.pattern_id for pattern in self.patterns}

    @property
    def forbidden(self) -> list[str]:
        return list(dict.fromkeys(literal.text for literal in self.literals))

    def public_patterns(self) -> list[dict]:
        return [pattern.public() for pattern in self.patterns]

    def raw_observations(self) -> list[dict]:
        return json.loads(self.observations_json)

    def trace(self) -> dict:
        return json.loads(self.trace_json)


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and " " + needle + " " in " " + haystack + " "


def observed_relationship(prediction: str, answers: Sequence[str], context: str) -> str:
    predicted = patches.normalized(prediction)
    references = tuple(patches.normalized(answer) for answer in answers)
    source = patches.normalized(context)
    if not predicted:
        return "empty_prediction"
    if predicted in references:
        return "normalized_match"
    if any(_contains(predicted, answer) for answer in references):
        return "overextended_span"
    if any(_contains(answer, predicted) for answer in references):
        return "incomplete_span"
    if not any(_contains(source, answer) for answer in references):
        return "no_literal_reference_anchor"
    return "competing_supported_span" if _contains(source, predicted) else "unsupported_prediction"


def build_packet(
    items: Sequence[Mapping], parent_blocks: Mapping, answers: Mapping, *,
    partition: str, minimum_groups: int = 2,
    task_identity_inventory: Sequence[str] | None = None,
) -> EvidencePacket:
    if partition != "train":
        raise ValueError("only training-source records may enter the abstraction boundary")
    if type(minimum_groups) is not int or minimum_groups < 2:
        raise ValueError("an abstract pattern needs at least two independent source groups")
    joint.validate_items(items)
    joint.validate_blocks(items, parent_blocks, len(parent_blocks))
    ids = {item["id"] for item in items}
    inventory = sorted(ids) if task_identity_inventory is None else list(task_identity_inventory)
    if (
        any(not isinstance(value, str) or not value for value in inventory)
        or len(set(inventory)) != len(inventory) or not ids <= set(inventory)
    ):
        raise ValueError("the complete original task-ID guard inventory must include every selected task")
    if set(answers) != ids or any(
        not isinstance(values, list) or not values
        or any(not isinstance(answer, str) or not answer.strip() for answer in values)
        for values in answers.values()
    ):
        raise ValueError("the complete selected training answer inventory is required")
    grouped, observations, literals, cases = defaultdict(list), [], [], []
    ordered = sorted(items, key=lambda item: item["id"])
    for index, item in enumerate(ordered, 1):
        task, evidence = item["id"], f"T{index:03d}"
        rows = [parent_blocks[str(repeat)][task] for repeat in range(len(parent_blocks))]
        if any(not isinstance(row.get("prediction"), str) for row in rows):
            raise ValueError("abstract relationships require recorded extracted predictions")
        features = dict(zip(NUMERIC_FEATURES, input_features(
            question=item["question"], context=item["context"], mode="numeric",
        ), strict=True))
        observed = {
            observed_relationship(row["prediction"], answers[task], item["context"]) for row in rows
        }
        if all(row["exact_match"] == 1 for row in rows):
            kind = "protected_success"
        elif any(row["exact_match"] == 1 for row in rows) or len(observed) != 1:
            kind = "unstable_error_signature"
        else:
            kind = next(iter(observed))
        mean_f1 = math.fsum(row["f1"] for row in rows) / len(rows)
        case = {"evidence": evidence, "task": task, "group": item["group"], "kind": kind}
        cases.append(case)
        if kind in PATTERN_SPECS:
            grouped[kind].append({**case, "features": features, "f1": mean_f1})
        observations.append({
            "partition": "train", "evidence": evidence, "question": item["question"],
            "context": item["context"], "accepted_training_answers": list(answers[task]),
            "parent_predictions": [row["prediction"] for row in rows],
            "mean_recorded_training_f1": mean_f1, "observable_features": features,
        })
        literals.append(Literal(task, "TASK_ID_LITERAL", evidence))
        literals.extend(Literal(answer, "TRAINING_ANSWER_LITERAL", evidence) for answer in answers[task])
        words = patches.normalized(item["question"]).split()
        literals.extend(
            Literal(" ".join(words[start:start + 5]), "TRAINING_QUESTION_PHRASE", evidence)
            for start in range(max(0, len(words) - 4))
        )
    patterns, unsupported = [], []
    for kind, members in sorted(grouped.items()):
        groups = tuple(sorted({member["group"] for member in members}))
        if len(groups) < minimum_groups:
            unsupported.append({"kind": kind, "groups": len(groups), "reason": "INSUFFICIENT_GROUP_SUPPORT"})
            continue
        multiplicities = Counter(member["group"] for member in members)
        denominator = len(groups)
        feature_means = tuple(
            (name, math.fsum(
                member["features"][name] / multiplicities[member["group"]] for member in members
            ) / denominator) for name in NUMERIC_FEATURES
        )
        patterns.append(Pattern(
            f"A{len(patterns) + 1:03d}", kind,
            tuple(sorted(member["evidence"] for member in members)), groups,
            math.fsum(member["f1"] / multiplicities[member["group"]] for member in members) / denominator,
            feature_means,
        ))
    literals.extend(Literal(pattern.pattern_id, "EVIDENCE_ID_LITERAL", pattern.pattern_id) for pattern in patterns)
    literals.extend(Literal(task, "TASK_ID_LITERAL", "source_inventory") for task in inventory if task not in ids)
    trace = {
        "schema_version": SCHEMA, "partition": "train", "minimum_independent_groups": minimum_groups,
        "cases": cases, "unsupported_patterns": unsupported,
        "case_categories": dict(sorted(Counter(case["kind"] for case in cases).items())),
        "patterns": {
            pattern.pattern_id: {"kind": pattern.kind, "evidence": list(pattern.evidence), "groups": list(pattern.groups)}
            for pattern in patterns
        },
        "repeat_count": len(parent_blocks),
        "raw_instance_fields_in_public_patterns": False,
        "hypotheses_are_not_causal_attribution": True,
    }
    return EvidencePacket(
        tuple(patterns), json.dumps(observations, ensure_ascii=False, sort_keys=True, allow_nan=False),
        tuple(literals), json.dumps(trace, ensure_ascii=False, sort_keys=True, allow_nan=False),
    )


def issue(
    code: str, field: str, *, rule_id: str | None = None,
    private: dict | None = None, observed: int | None = None, limit: int | None = None,
) -> dict:
    return {
        "code": code, "field": field, "rule_id": rule_id, "action": PUBLIC_HINTS[code],
        "observed": observed, "limit": limit, "private": private,
    }


def inspect_fields(parent: Mapping, value: object, packet: EvidencePacket, limits: Mapping) -> list[dict]:
    findings = []
    if not isinstance(value, dict) or set(value) != {"patch", "applicability"}:
        return [issue("SCHEMA", "$")]
    patch, applicability = value["patch"], value["applicability"]
    if not isinstance(patch, dict) or set(patch) != {
        "parent_skill_sha256", "hypothesis", "risk", "evidence", "changes",
    }:
        return [issue("SCHEMA", "$.patch")]
    if patch["parent_skill_sha256"] != patches.digest(parent):
        findings.append(issue("PARENT_HASH", "$.patch.parent_skill_sha256"))
    for field in ("hypothesis", "risk"):
        if not isinstance(patch[field], str) or len(patch[field]) > 1600:
            findings.append(issue("SCHEMA", "$.patch." + field))
    evidence = patch["evidence"]
    if (
        not isinstance(evidence, list) or not evidence
        or any(not isinstance(name, str) for name in evidence)
        or (isinstance(evidence, list) and all(isinstance(name, str) for name in evidence)
            and (len(set(evidence)) != len(evidence) or not set(evidence) <= packet.evidence_ids))
    ):
        findings.append(issue("EVIDENCE", "$.patch.evidence"))
    if not isinstance(applicability, dict) or set(applicability) != {"hypothesis", "features"}:
        findings.append(issue("SCHEMA", "$.applicability"))
    else:
        if (
            not isinstance(applicability["hypothesis"], str)
            or not 1 <= len(applicability["hypothesis"].strip()) <= 1600
        ):
            findings.append(issue("SCHEMA", "$.applicability.hypothesis"))
        try:
            joint.validate_features(applicability["features"])
        except ValueError:
            findings.append(issue("FEATURES", "$.applicability.features"))
    changes = patch["changes"]
    if not isinstance(changes, list):
        return findings + [issue("SCHEMA", "$.patch.changes")]
    if not 1 <= len(changes) <= limits["max_changes"]:
        findings.append(issue(
            "EDIT_COUNT", "$.patch.changes", observed=len(changes), limit=limits["max_changes"],
        ))
    known_literals = defaultdict(list)
    for literal in packet.literals:
        known_literals[literal.text].append(literal)
    seen = set()
    for index, change in enumerate(changes):
        field = f"$.patch.changes[{index}]"
        if not isinstance(change, dict) or set(change) != {"rule_id", "text"}:
            findings.append(issue("SCHEMA", field))
            continue
        name, text = change["rule_id"], change["text"]
        valid_name = (
            isinstance(name, str) and name in parent["rules"] and name.startswith("delete.") and name not in seen
        )
        if not valid_name:
            findings.append(issue("RULE_ID", field + ".rule_id"))
        if isinstance(name, str):
            seen.add(name)
        safe_name = name if valid_name else None
        if not isinstance(text, str):
            findings.append(issue("RULE_TEXT", field + ".text", rule_id=safe_name))
            continue
        if (
            len(text) > limits["max_rule_characters"]
            or any(character in text for character in ("\n", "\r", "\x00"))
            or (text and (not text.startswith("- ") or len(text) <= 2))
        ):
            findings.append(issue(
                "RULE_TEXT", field + ".text", rule_id=safe_name,
                observed=len(text), limit=limits["max_rule_characters"],
            ))
        if valid_name and text == parent["rules"][name]:
            findings.append(issue("NO_CHANGE", field + ".text", rule_id=safe_name))
        for match in patches.memorization_violations(text, packet.forbidden):
            if match["kind"] == "forbidden_literal":
                supports = known_literals[match["constraint"]]
                code = supports[0].kind
                evidence_ids = sorted({literal.evidence for literal in supports})
            else:
                code = "EVIDENCE_ID_LITERAL" if match["kind"] == "evidence_id" else "PROTECTED_SURFACE"
                evidence_ids = []
            findings.append(issue(
                code, field + ".text", rule_id=safe_name,
                private={**match, "evidence": evidence_ids, "span_basis": "rejected_proposed_text"},
            ))
    return findings


@dataclass(frozen=True)
class Assessment:
    status: str
    proposal_sha256: str
    findings_json: str
    child: dict | None = None
    receipt: dict | None = None

    def __post_init__(self) -> None:
        if (
            self.status not in {"ADMITTED", "REJECTED"}
            or re.fullmatch(r"[0-9a-f]{64}", self.proposal_sha256) is None
        ):
            raise ValueError("assessments require an explicit status and exact raw proposal hash")
        findings = json.loads(self.findings_json)
        if not isinstance(findings, list):
            raise ValueError("assessment findings must be a list")
        for finding in findings:
            if (
                not isinstance(finding, dict)
                or set(finding) != {"code", "field", "rule_id", "action", "observed", "limit", "private"}
                or finding["code"] not in PUBLIC_HINTS
                or finding["action"] != PUBLIC_HINTS[finding["code"]]
                or not isinstance(finding["field"], str)
                or re.fullmatch(
                    r"\$(?:\.patch(?:\.(?:parent_skill_sha256|hypothesis|risk|evidence|changes"
                    r"(?:\[\d+\](?:\.(?:rule_id|text))?)?))?"
                    r"|\.applicability(?:\.(?:hypothesis|features))?)?",
                    finding["field"],
                ) is None
                or (finding["rule_id"] is not None and (
                    not isinstance(finding["rule_id"], str)
                    or re.fullmatch(r"delete\.(?:normalization|evidence|answer_type|snippet_format|traps)\.r\d+", finding["rule_id"]) is None
                ))
                or any(finding[key] is not None and type(finding[key]) is not int for key in ("observed", "limit"))
            ):
                raise ValueError("public rejection metadata must use canonical case-free fields and messages")
            private = finding["private"]
            if private is not None and not isinstance(private, dict):
                raise ValueError("private diagnostics require an explicit mapping")
            if private is not None and "start" in private:
                if (
                    type(private["start"]) is not int or type(private.get("end")) is not int
                    or not 0 <= private["start"] <= private["end"]
                ):
                    raise ValueError("diagnostic spans require checked integer offsets")
            if private is not None and {"line", "column"} <= set(private):
                if any(type(private[key]) is not int or private[key] < 1 for key in ("line", "column")):
                    raise ValueError("JSON error locations require positive integers")
        if self.status == "ADMITTED":
            if findings or self.child is None or self.receipt is None:
                raise ValueError("admitted assessments require a compiled candidate and no violations")
        elif not findings or self.child is not None or self.receipt is not None:
            raise ValueError("a rejected assessment requires explicit findings and no deployed child")

    def private_findings(self) -> list[dict]:
        return json.loads(self.findings_json)

    def public_feedback(self, detail: str) -> dict:
        if detail not in {"generic", "specific"}:
            raise ValueError("feedback detail must be generic or specific")
        if self.status == "ADMITTED":
            return {
                "status": self.status, "proposal_sha256": self.proposal_sha256,
                "message": "Structural admission passed; quality and applicability benefit have not been evaluated.",
            }
        findings = self.private_findings()
        result = {
            "status": "REJECTED", "proposal_sha256": self.proposal_sha256,
            "message": "The proposal failed the unchanged rule/evidence/interface checks. Generate a new general rule.",
        }
        if detail == "specific":
            result["issue_count"] = len(findings)
            result["issues"] = []
            for finding in findings[:16]:
                safe = {key: finding[key] for key in ("code", "field", "rule_id", "action", "observed", "limit")}
                if finding["private"] is not None and "start" in finding["private"]:
                    safe["rejected_text_span"] = {
                        "start": finding["private"]["start"], "end": finding["private"]["end"],
                        "basis": "rejected_proposed_text", "literal_redacted": True,
                        "unit": "unicode_codepoints", "interval": "zero_based_half_open",
                    }
                elif finding["private"] is not None and {"line", "column"} <= set(finding["private"]):
                    safe["json_location"] = {
                        key: finding["private"][key] for key in ("line", "column")
                    }
                result["issues"].append(safe)
            result["omitted_issue_count"] = max(0, len(findings) - 16)
        return result


def assess_proposal(
    parent: dict, raw: str, packet: EvidencePacket, *, limits: Mapping,
    native_envelope: Callable[[dict], Mapping[str, int]] | None = None,
    prior_candidates: Sequence[str] = (),
) -> Assessment:
    if not packet.patterns:
        raise ValueError("no multi-group abstraction supports a proposal")
    digest = patches.text_hash(raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        findings = [issue("JSON_SYNTAX", "$", private={"line": error.lineno, "column": error.colno})]
        return Assessment("REJECTED", digest, json.dumps(findings, sort_keys=True))
    findings = inspect_fields(parent, value, packet, limits)
    child, receipt = None, None
    if not findings:
        try:
            child, receipt = joint.compile_proposal(
                parent, value, evidence_ids=packet.evidence_ids, forbidden=packet.forbidden,
                max_changes=limits["max_changes"], max_rule_characters=limits["max_rule_characters"],
                max_profile_growth_bytes=limits["max_profile_growth_bytes"],
            )
        except ValueError as error:
            code = "PROFILE_GROWTH" if str(error) == "per-step profile byte-growth bound exceeded" else "COMPILER_REJECTED"
            growth = sum(
                len(change["text"].encode()) - len(parent["rules"][change["rule_id"]].encode())
                for change in value["patch"]["changes"]
            ) if code == "PROFILE_GROWTH" else None
            findings.append(issue(
                code, "$.patch", private={"compiler_error": str(error)}, observed=growth,
                limit=limits["max_profile_growth_bytes"] if code == "PROFILE_GROWTH" else None,
            ))
    if child is not None and receipt is not None:
        candidate_hash = patches.digest({
            "profiles": patches.render_skill(child), "features": receipt["applicability"]["features"],
        })
        if candidate_hash in prior_candidates:
            findings.append(issue("DUPLICATE", "$"))
        if native_envelope is not None:
            measured = native_envelope(child)
            if set(measured) != {"maximum_growth", "allowed_growth"} or any(
                type(number) is not int for number in measured.values()
            ) or measured["allowed_growth"] < 0:
                raise ValueError("native-envelope adapter must return measured integer growth and its frozen limit")
            if measured["maximum_growth"] > measured["allowed_growth"]:
                findings.append(issue(
                    "NATIVE_GROWTH", "$.patch", observed=measured["maximum_growth"],
                    limit=measured["allowed_growth"],
                ))
        if not findings:
            receipt = {
                **receipt, "abstraction_schema": SCHEMA, "candidate_sha256": candidate_hash,
                "pattern_evidence": {
                    name: packet.trace()["patterns"][name] for name in value["patch"]["evidence"]
                },
                "exact_case_literal_guard_passed": True,
                "semantic_case_independence_certified": False,
                "quality_or_generalization_certified": False,
            }
    return Assessment(
        "REJECTED" if findings else "ADMITTED", digest,
        json.dumps(findings, ensure_ascii=False, sort_keys=True, allow_nan=False),
        None if findings else child, None if findings else receipt,
    )


def proposal_messages(
    parent: Mapping, packet: EvidencePacket, history: Sequence[Assessment], *,
    exposure: str, feedback_detail: str, branch: str, maximum_changes: int,
) -> list[dict]:
    if exposure not in {"with_cases", "abstract"} or feedback_detail not in {"generic", "specific"}:
        raise ValueError("the evidence-exposure and rejection-feedback conditions must be explicit")
    if not packet.patterns:
        raise ValueError("no supported abstract pattern; stop instead of inventing general rules")
    if any(not isinstance(value, Assessment) for value in history):
        raise ValueError("history must contain checked assessments, not raw proposal or case text")
    messages = joint.proposal_messages(
        parent, packet.public_patterns(), [], branch=branch, maximum_changes=maximum_changes,
    )
    messages[0]["content"] += (
        " The evidence entries are abstract error-pattern cards supported by at least two "
        "independent training groups. Cite their A-prefixed IDs in patch.evidence. "
        "They describe observed relationships, not proven instruction defects. Use their "
        "runtime operations to formulate a reusable conditional procedure; do not insert "
        "example answers, named instances, copied clue phrases or template slots for those facts. "
        "Runtime rules may use only the question and supplied context, never training labels "
        "or evaluation outcomes. Rejection feedback refers to previous proposed replacements, "
        "not character offsets in the parent. Regenerate the indicated rule from its abstract "
        "operation rather than replacing a forbidden entity with a different example. "
        "Only admitted candidates may later be tested for quality."
    )
    payload = json.loads(messages[1]["content"])
    payload["evidence_interface"] = SCHEMA
    payload["training_history"] = [value.public_feedback(feedback_detail) for value in history]
    payload["guard_limits"] = {
        "maximum_changed_rules": maximum_changes,
        "case_literals_and_protected_interfaces": "unchanged authoritative compiler",
    }
    if exposure == "with_cases":
        payload["case_material"] = packet.raw_observations()
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return deepcopy(messages)
