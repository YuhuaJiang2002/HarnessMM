"""Exact analytical checks for one prospective, training-grounded rule audit."""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import math
import re

CANDIDATE = "constraint_ledger_audited"
CORRECTIONS = {
    "R1": (
        "Partition intersections by the stated branches. Substitute the actual curve equation "
        "into each branch and solve the resulting equation; use a quadratic equation only when "
        "its degree is actually two. Filter every solution using the original branch domain, "
        "including strict endpoints."
    ),
    "R2": (
        "When Vieta gives the sum S of all algebraic roots, subtract the sum of every root "
        "excluded by the original domain: valid sum = S - sum(excluded roots). Equivalently, "
        "add only roots individually verified admissible. Never add an excluded root to S; "
        "count boundary roots once."
    ),
    "R6": (
        "For an LCM target, factor the target into prime powers and require the selected divisors "
        "to attain each prime's maximum exponent; different divisors may supply different primes. "
        "Count complements and their intersections by inclusion-exclusion. Do not replace these "
        "coordinatewise conditions with requiring the target divisor itself to be selected. "
        "For a scalar maximum, check whether an element must attain it before using a single "
        "preimage condition."
    ),
}
WITNESSES = {
    "R1": {
        "certificate_id": "nonquadratic-curve", "inputs": ["-1/2", "0", "1/2"],
        "reference": "0", "old_result": "2", "correct_result": "3",
    },
    "R2": {
        "certificate_id": "excluded-root-sum", "inputs": ["1", "2"],
        "reference": "3", "old_result": "5", "correct_result": "1",
    },
    "R6": {
        "certificate_id": "coordinatewise-lcm", "inputs": ["2", "3"],
        "reference": "6", "old_result": "0", "correct_result": "1",
    },
}
AUDIT_SYSTEM = (
    "Audit one existing mathematical rule using only its quoted rule text and supplied training "
    "evidence. The analytical witness is a toy check, not a benchmark result. For a documented "
    "defect, explain precisely why the old wording is unsafe and how the proposed minimal "
    "correction fixes it. Quote the actual rule and an actual supplied training passage. "
    "Use the registered correction and exact rational witness without inventing evidence. "
    "For a rule without a demonstrated defect, retain its exact text and explain its scope. "
    "Do not claim that your judgment proves a theorem. Return the required JSON."
)


def rational(value):
    if not isinstance(value, str) or not re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value):
        raise ValueError("Analytical certificates require exact rational strings")
    return Fraction(value)


def verify_witness(rule_id, witness):
    if rule_id not in WITNESSES:
        raise ValueError("No arithmetic correction is registered for this rule")
    expected = WITNESSES[rule_id]
    if set(witness) != set(expected) or witness["certificate_id"] != expected["certificate_id"]:
        raise ValueError("Wrong analytical certificate")
    values = [rational(value) for value in witness["inputs"]]
    if values != [rational(value) for value in expected["inputs"]]:
        raise ValueError("Do not substitute an unregistered analytical witness")
    reference = rational(witness["reference"])
    old, corrected = rational(witness["old_result"]), rational(witness["correct_result"])
    if rule_id == "R2":
        total = sum(values, Fraction())
        valid, excluded = values
        if reference != total or old != total + excluded or corrected != total - excluded:
            raise ValueError("Excluded-root arithmetic is inconsistent")
        if corrected != valid or corrected == old:
            raise ValueError("The witness does not distinguish valid from excluded-root addition")
        derivation = "S = sum(valid) + sum(excluded), hence sum(valid) = S - sum(excluded)."
    elif rule_id == "R1":
        # The curve x=y^3+3y/4 meets the branch y=x at three rational interior points.
        if len(set(values)) != 3 or any(y**3 - y / 4 != 0 or not -1 < y < 1 for y in values):
            raise ValueError("The cubic witness lacks three distinct admissible roots")
        if (reference, old, corrected) != (0, 2, 3):
            raise ValueError("Polynomial-degree certificate is inconsistent")
        derivation = "Substitution gives y^3-y/4=0, of degree3, with roots-1/2,0,1/2."
    else:
        if any(value.denominator != 1 or value <= 0 for value in values):
            raise ValueError("LCM witness must use positive integers")
        numbers = [int(value) for value in values]
        joined = math.lcm(*numbers)
        requires_target_member = any(value == reference for value in values)
        if joined != reference or old != int(requires_target_member) or corrected != 1 or old == corrected:
            raise ValueError("Coordinatewise LCM counterexample is inconsistent")
        derivation = "lcm(2,3)=6 although6 is not selected; different elements supply prime maxima."
    return {
        "status": "EXACT_RATIONAL_CHECK_PASS", "rule_id": rule_id,
        "witness": copy.deepcopy(witness), "derivation": derivation,
        "benchmark_result": False, "llm_theorem_verification": False,
    }


def expected_witness(rule_id):
    return copy.deepcopy(WITNESSES.get(rule_id, {
        "certificate_id": "none", "inputs": [], "reference": "0",
        "old_result": "0", "correct_result": "0",
    }))


def citation_catalog(evidence, allowed_ids):
    catalog = {}
    for case_id in allowed_ids:
        for field in ("question", "baseline_derivation", "teacher_diagnosis"):
            source = evidence[case_id][field]
            for index, quote in enumerate(source.split("\n\n")):
                if len(quote.strip()) >= 12:
                    catalog[f"{case_id}.{field}.{index:04d}"] = {
                        "case_id": case_id, "field": field, "quote": quote,
                    }
    return catalog


def resolve_citations(response, catalog):
    """Resolve selected immutable passages; never repair or paraphrase a quote."""
    references = response["citation_ids"]
    if not references or len(references) != len(set(references)):
        raise ValueError("Missing or duplicate source-passage references")
    if any(reference not in catalog for reference in references):
        raise ValueError("Audit selected a nonexistent training passage")
    result = copy.deepcopy(response)
    del result["citation_ids"]
    result["evidence"] = [copy.deepcopy(catalog[reference]) for reference in references]
    return result


def audit_schema(rule_id, train_ids):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["rule_id", "decision", "rule_quote", "evidence", "explanation", "revised_check", "witness"],
        "properties": {
            "rule_id": {"type": "string", "enum": [rule_id]},
            "decision": {"type": "string", "enum": ["retain", "correct"]},
            "rule_quote": {"type": "string", "minLength": 12},
            "evidence": {
                "type": "array", "minItems": 1, "maxItems": 6,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["case_id", "field", "quote"],
                    "properties": {
                        "case_id": {"type": "string", "enum": list(train_ids)},
                        "field": {"type": "string", "enum": ["question", "baseline_derivation", "teacher_diagnosis"]},
                        "quote": {"type": "string", "minLength": 12},
                    },
                },
            },
            "explanation": {"type": "string", "minLength": 60, "maxLength": 4000},
            "revised_check": {"type": "string", "minLength": 20, "maxLength": 800},
            "witness": {
                "type": "object", "additionalProperties": False,
                "required": ["certificate_id", "inputs", "reference", "old_result", "correct_result"],
                "properties": {
                    "certificate_id": {"type": "string", "enum": [
                        "none", "excluded-root-sum", "nonquadratic-curve", "coordinatewise-lcm",
                    ]},
                    "inputs": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
                    "reference": {"type": "string"}, "old_result": {"type": "string"},
                    "correct_result": {"type": "string"},
                },
            },
        },
    }


def apply_audit(rule, response, evidence):
    """Validate provenance and exact checks before applying a predeclared correction."""
    identifier = rule["rule_id"]
    if response["rule_id"] != identifier or response["rule_quote"] not in rule["check"]:
        raise ValueError("Audit does not quote the actual rule")
    if len(response["rule_quote"]) < 12 or len(response["explanation"]) < 60:
        raise ValueError("Audit explanation or rule quote is incomplete")
    if not response["evidence"]:
        raise ValueError("Audit lacks training evidence")
    for citation in response["evidence"]:
        if citation["case_id"] not in rule["evidence_ids"] or citation["case_id"] not in evidence:
            raise ValueError("Audit cites unavailable or non-training evidence")
        source = evidence[citation["case_id"]][citation["field"]]
        if len(citation["quote"]) < 12 or citation["quote"] not in source:
            raise ValueError("Audit fabricated or changed its training quote")
    desired = CORRECTIONS.get(identifier, rule["check"])
    action = "correct" if identifier in CORRECTIONS else "retain"
    if response["decision"] != action or response["revised_check"] != desired:
        raise ValueError("Audit did not justify the single registered correction or exact retention")
    if action == "correct":
        certificate = verify_witness(identifier, response["witness"])
    else:
        if response["witness"] != expected_witness(identifier):
            raise ValueError("Do not manufacture a correction for a retained rule")
        certificate = {"status": "RETAINED_UNCHANGED", "benchmark_result": False}
    revised = {**copy.deepcopy(rule), "check": desired}
    return revised, {
        "rule_id": identifier, "decision": action,
        "original_check_sha256": hashlib.sha256(rule["check"].encode()).hexdigest(),
        "revised_check_sha256": hashlib.sha256(desired.encode()).hexdigest(),
        "evidence": copy.deepcopy(response["evidence"]),
        "explanation": response["explanation"], "certificate": certificate,
        "model_proof_claim": False,
    }


def library_sha256(library):
    return hashlib.sha256(json.dumps(library, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
