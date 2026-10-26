"""Bounded native TRAIN acceptance and analytical checks, without benchmark imports."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import re

from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition

CANDIDATE = "constraint_train_accepted"
MAX_UPDATES = 2
CERTIFICATES = {
    "finite_prefix_count": {
        "setup": "On the integers from one through eight, all values below three qualify; "
                 "from three onward exactly the even values qualify.",
        "naive_global_period_count": "4", "exact_prefix_plus_tail_count": "5",
        "lesson": "A periodic tail does not authorize applying its residue rule to the finite prefix.",
    },
    "nonnegative_exchange": {
        "setup": "An exchange consumes one existing resource. Stock is either zero or one.",
        "feasible_with_zero_stock": "0", "feasible_with_one_stock": "1",
        "lesson": "An improving exchange can refute optimality only where the exchange is feasible.",
    },
    "oriented_area": {
        "setup": "The unit square is traversed once in either orientation.",
        "forward_cross_sum": "2", "reverse_cross_sum": "-2", "unsigned_area": "1",
        "lesson": "Check vertex order, orientation and the half factor before using an area formula.",
    },
}
PROPOSAL_SYSTEM = (
    "You are proposing one small general mathematical reasoning correction using TRAIN evidence only. "
    "Explain the actual error in the incumbent derivation, contrast it with the supplied successful "
    "training view, and propose a short conditional rule. Preserve the behavior of already-correct "
    "tasks. Do not encode task IDs, benchmark names, answers, numeric literals, names or unique "
    "phrases from a task in the rule. Do not give a worked solution in the rule. The policy is a "
    "general instruction, not an answer lookup. Select exact source-passage IDs as evidence. "
    "Any arithmetic claim must be supported by one of the supplied exact analytical certificates; "
    "these are toy checks, not benchmark results or LLM theorem verification. "
    "Return exactly one JSON object with keys diagnosis, general_rule, evidence_refs, "
    "arithmetic_checks, nonregression_rationale. evidence_refs must be keys from "
    "training_evidence_catalog; arithmetic_checks must be keys from analytical_certificates. "
    "Do not emit code fences, notes, self-correction prose, or another JSON object."
)


def policy_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evidence_catalog(incumbent, contrasts, previous=None):
    needed = {row["id"] for row in incumbent if not row["hard"]}
    if previous:
        needed.update(row["id"] for row in previous if not row["hard"])
    sources = {
        row["id"] + ".incumbent": {"id": row["id"], "hard": row["hard"], "text": row["response"]}
        for row in incumbent if row["id"] in needed
    }
    sources.update(contrasts)
    if previous:
        for row in previous:
            if not row["hard"]:
                sources[row["id"] + ".rejected_candidate"] = {
                    "id": row["id"], "hard": row["hard"], "text": row["response"],
                }
    catalog = {}
    for name, source in sources.items():
        for index, text in enumerate(source["text"].split("\n\n")):
            if len(text.strip()) >= 12:
                catalog[f"{name}.{index:04d}"] = {
                    "training_id": source["id"], "native_correct": source["hard"], "passage": text,
                }
    return catalog


def training_only_gate(data, manifest, *, final=False):
    validate_partition(manifest)
    if final or manifest["benchmark"] != "aime2025" or manifest["counts"] != {"train": 6, "validation": 6, "test": 18}:
        raise ValueError("Training loop requires the canonical six-only training population")
    if set(data) != {"train", "development", "test"} or data["development"] or data["test"]:
        raise ValueError("Validation and test payloads must be absent from the training loop")
    if [item["id"] for item in data["train"]] != manifest["splits"]["train"]:
        raise ValueError("Training-loop population differs")
    bindings = {row["id"]: row["payload_sha256"] for row in manifest["records"]}
    if any(canonical_sha256(item) != bindings[item["id"]] for item in data["train"]):
        raise ValueError("Training payload drift")


def complete_scores(rows, expected_ids):
    if len(expected_ids) != 6 or len(set(expected_ids)) != 6:
        raise ValueError("The training population must contain six unique canonical IDs")
    if len(rows) != 6 or len({row["id"] for row in rows}) != 6:
        raise ValueError("Partial or duplicated candidate training population")
    indexed = {row["id"]: row for row in rows}
    if set(indexed) != set(expected_ids):
        raise ValueError("Candidate training membership differs")
    if any(type(row["hard"]) is not int or row["hard"] not in (0, 1) for row in rows):
        raise ValueError("Only native binary integer correctness may select a training policy")
    return indexed


def acceptance(incumbent, proposed, expected_ids):
    old = complete_scores(incumbent, expected_ids)
    new = complete_scores(proposed, expected_ids)
    regressions = [key for key in expected_ids if old[key]["hard"] == 1 and new[key]["hard"] != 1]
    improvements = [key for key in expected_ids if old[key]["hard"] == 0 and new[key]["hard"] == 1]
    old_correct = sum(row["hard"] for row in old.values())
    new_correct = sum(row["hard"] for row in new.values())
    accepted = not regressions and new_correct > old_correct
    return {
        "accepted": accepted, "incumbent_correct": old_correct, "candidate_correct": new_correct,
        "tasks": 6, "regressions": regressions, "newly_correct": improvements,
        "criterion": "Strict native accuracy improvement and zero loss of previously correct training cases",
        "reason": "ACCEPT" if accepted else "REJECT_REGRESSION" if regressions else "REJECT_NO_STRICT_IMPROVEMENT",
    }


def check_certificate(identifier):
    if identifier not in CERTIFICATES:
        raise ValueError("Unregistered arithmetic assertion")
    if identifier == "finite_prefix_count":
        values = range(1, 9)
        actual = sum(n < 3 or n % 2 == 0 for n in values)
        naive = sum(n % 2 == 0 for n in values)
        if (naive, actual) != (4, 5):
            raise AssertionError("Finite-prefix arithmetic failed")
    elif identifier == "nonnegative_exchange":
        if (int(0 <= 1 <= 0), int(0 <= 1 <= 1)) != (0, 1):
            raise AssertionError("Feasible-exchange arithmetic failed")
    else:
        vertices = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
                    (Fraction(1), Fraction(1)), (Fraction(0), Fraction(1))]
        def twice(points):
            return sum(
                points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
                for i in range(4)
            )
        if twice(vertices) != 2 or twice(vertices[::-1]) != -2 or abs(twice(vertices)) / 2 != 1:
            raise AssertionError("Oriented-area arithmetic failed")
    return {"certificate_id": identifier, "status": "EXACT_ANALYTICAL_CHECK_PASS",
            "example": CERTIFICATES[identifier], "benchmark_result": False, "llm_theorem_verified": False}


def validate_general_rule(rule, training_items):
    if not isinstance(rule, str) or not 40 <= len(rule) <= 1600:
        raise ValueError("A proposed rule must be one bounded general instruction")
    if re.search(r"\d", rule) or any(item["id"].lower() in rule.lower() for item in training_items):
        raise ValueError("Do not encode numerical literals or case identities in the rule")
    lowered = rule.lower()
    if any(term in lowered for term in ("aime", "benchmark", "answer:", "correct answer")):
        raise ValueError("Benchmark/answer-specific policy content is forbidden")
    rule_words = re.findall(r"[a-z]+", lowered)
    for item in training_items:
        words = re.findall(r"[a-z]+", item["question"].lower())
        segments = {" ".join(words[i:i + 8]) for i in range(max(0, len(words) - 7))}
        if any(" ".join(rule_words[i:i + 8]) in segments for i in range(max(0, len(rule_words) - 7))):
            raise ValueError("Proposed rule copied a task-specific passage")
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", item["question"])
        stop = {"Let", "Find", "From", "There", "The", "For", "However", "Graph", "Suppose", "Every"}
        if any(name not in stop and re.search(r"\b" + re.escape(name.lower()) + r"\b", lowered)
               for name in names):
            raise ValueError("Proposed rule contains a task-specific name")
    return rule


def proposal_schema(passage_ids):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["diagnosis", "general_rule", "evidence_refs", "arithmetic_checks", "nonregression_rationale"],
        "properties": {
            "diagnosis": {"type": "string", "minLength": 40, "maxLength": 4000},
            "general_rule": {"type": "string", "minLength": 40, "maxLength": 1600},
            "evidence_refs": {"type": "array", "minItems": 1, "maxItems": 8,
                              "items": {"type": "string", "enum": list(passage_ids)}},
            "arithmetic_checks": {"type": "array", "minItems": 1, "maxItems": 3,
                                  "items": {"type": "string", "enum": list(CERTIFICATES)}},
            "nonregression_rationale": {"type": "string", "minLength": 40, "maxLength": 2000},
        },
    }
