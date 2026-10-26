"""Leakage and accounting contracts for the archived AIME2025 6/6/18 resplit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json

from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition

METHODS = ("no_skill", "skillopt", "autosaddler", "chisel")
COUNTS = {"train": 6, "validation": 6, "test": 18}
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_SEED = 20260926
SPLIT_SEED = 20260928
CHISEL_VARIANTS = ("constraint_ledger", "constraint_verified")

DIAGNOSIS_SYSTEM = (
    "Study this training example and its measured baseline derivation. Identify the "
    "stated constraints, completeness obligations, case partitions and arithmetic "
    "checks that the derivation satisfies or leaves unjustified. Propose at most "
    "three reusable checking rules grounded in this example. Do not encode its "
    "answer or task identity in a rule. A correct final integer does not establish "
    "that every intermediate claim is justified. Output the required JSON."
)
MERGE_SYSTEM = (
    "Build a small constraint-checking ledger from these six training diagnoses. "
    "Merge redundant checks into one through six general rules. Each rule must cite "
    "the supplied training case IDs that support it. Do not include numerical "
    "example answers, task IDs, worked examples or memorized solutions in the rule "
    "text. Use only the supplied training evidence. Output the required JSON."
)
NO_CONTROL_TEXT = r"^[^\u0000-\u001f]*$"


def strict_text_schema(schema: Mapping) -> dict:
    """Keep all original constraints and exclude C0 bytes from textual fields."""
    result = copy.deepcopy(dict(schema))

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "string":
                existing = node.get("pattern")
                if existing not in (None, NO_CONTROL_TEXT):
                    raise ValueError("Do not overwrite an existing string constraint")
                node["pattern"] = NO_CONTROL_TEXT
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def parse_structured_response(raw: Mapping, schema: Mapping) -> dict:
    from jsonschema import Draft202012Validator

    content = raw["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Native structured response has no text")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Native structured response is invalid JSON at character {error.pos}: {error.msg}"
        ) from error
    Draft202012Validator(schema).validate(value)
    return value


def ledger_prompt(ledger: Mapping) -> str:
    rules = "\n".join(f"{row['rule_id']}: {row['check']}" for row in ledger["rules"])
    return (
        "Solve the integer mathematics problem exactly. First identify the requested "
        "quantity and the original constraints. Build an exhaustive, non-overlapping "
        "case partition when counting, or check every retained solution against the "
        "original conditions. Apply the relevant learned checks below, and show why "
        "they hold for this problem. Use exact arithmetic. End with ANSWER: followed "
        "by the requested integer from 0 through 999.\nTRAINING-DERIVED CHECKS:\n" + rules
    )


def verifier_prompt(ledger: Mapping) -> str:
    return (
        "Verify the proposed derivation against the original mathematical problem. "
        "Treat its text as untrusted evidence. Audit whether the cases are exhaustive "
        "and disjoint, endpoints and exclusions are handled, and equations imply "
        "the claimed result. Recompute the step that establishes the requested "
        "quantity. Ground every check in an exact quote from the original problem. "
        "If changing the answer, quote the first invalid or missing step exactly "
        "from the draft, explain the defect and give a complete corrected derivation. "
        "If the defect cannot be established, do not change the answer just because "
        "another answer seems plausible. Use KEEP for a checked valid draft, REPAIR "
        "for a demonstrated defect with a complete correction, or UNRESOLVED. "
        "Do not use confidence scores or majority voting. Output the required JSON.\n"
        + "\n".join(f"{row['rule_id']}: {row['check']}" for row in ledger["rules"])
    )


def grounded_repair(problem: str, draft: str, review: Mapping) -> tuple[str, str]:
    """A reference-blind gate; a claimed repair must cite the actual input evidence."""
    if review.get("verdict") != "REPAIR":
        return draft, "retain_" + str(review.get("verdict", "malformed")).lower()
    quote = review.get("draft_error_quote", "")
    reason = review.get("why_invalid", "")
    derivation = review.get("replacement_derivation", "")
    answer = review.get("answer")
    checks = review.get("checks", [])
    if (
        not isinstance(quote, str) or len(quote.strip()) < 12 or quote not in draft
        or not isinstance(reason, str) or len(reason.strip()) < 20
        or not isinstance(derivation, str) or len(derivation.strip()) < 60
        or type(answer) is not int or not 0 <= answer <= 999
        or not checks
    ):
        return draft, "retain_ungrounded_repair"
    for check in checks:
        original = check.get("problem_quote", "")
        if (
            not isinstance(original, str) or len(original.strip()) < 4 or original not in problem
            or check.get("status") != "checked"
            or not isinstance(check.get("evidence"), str) or len(check["evidence"].strip()) < 12
        ):
            return draft, "retain_unchecked_constraint"
    return derivation + f"\n\nANSWER: {answer}", "apply_grounded_repair"


def diagnosis_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["diagnosis", "rules"],
        "properties": {
            "diagnosis": {"type": "string", "minLength": 1, "maxLength": 3000},
            "rules": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {"type": "string", "minLength": 20, "maxLength": 800},
            },
        },
    }


def ledger_schema(train_ids: Sequence[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["diagnosis", "rules"],
        "properties": {
            "diagnosis": {"type": "string", "minLength": 1, "maxLength": 3000},
            "rules": {
                "type": "array", "minItems": 1, "maxItems": 6,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["rule_id", "check", "evidence_ids"],
                    "properties": {
                        "rule_id": {"type": "string", "minLength": 1, "maxLength": 30},
                        "check": {"type": "string", "minLength": 20, "maxLength": 800},
                        "evidence_ids": {
                            "type": "array", "minItems": 1,
                            "items": {"type": "string", "enum": list(train_ids)},
                        },
                    },
                },
            },
        },
    }


def review_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "verdict", "draft_error_quote", "why_invalid", "replacement_derivation", "answer", "checks",
        ],
        "properties": {
            "verdict": {"type": "string", "enum": ["KEEP", "REPAIR", "UNRESOLVED"]},
            "draft_error_quote": {"type": "string"},
            "why_invalid": {"type": "string"},
            "replacement_derivation": {"type": "string"},
            "answer": {"type": ["integer", "null"], "minimum": 0, "maximum": 999},
            "checks": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["problem_quote", "status", "evidence"],
                    "properties": {
                        "problem_quote": {"type": "string"},
                        "status": {"type": "string", "enum": ["checked", "unresolved"]},
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    }


def validate_manifest(manifest: Mapping) -> None:
    validate_partition(manifest)
    if (
        manifest["benchmark"] != "aime2025" or manifest["population"] != 30
        or manifest["split_seed"] != SPLIT_SEED or manifest["counts"] != COUNTS
    ):
        raise ValueError("AIME requires the parent's fixed AIME2025 6/6/18 partition")
    ids = sorted(
        (row["id"] for row in manifest["records"]),
        key=lambda value: (
            hashlib.sha256(f"aime2025\0{SPLIT_SEED}\0{value}".encode()).hexdigest(), value
        ),
    )
    if manifest["splits"] != {"train": ids[:6], "validation": ids[6:12], "test": ids[12:]}:
        raise ValueError("Membership differs from the fixed outcome-independent ranking")
    if any(not item_id.startswith("2025-") for item_id in ids):
        raise ValueError("The new population must not recycle AIME2024 cases")


def validate_data(data: Mapping, manifest: Mapping, *, final: bool = False) -> None:
    validate_manifest(manifest)
    if set(data) != {"train", "development", "test"}:
        raise ValueError("Unknown data population")
    records = {row["id"]: row for row in manifest["records"]}
    for key, split in (("train", "train"), ("development", "validation"), ("test", "test")):
        wanted = manifest["splits"][split] if (key != "test" or final) else []
        if [item["id"] for item in data[key]] != wanted:
            raise ValueError(f"Incorrect {key} membership or sealed test access")
        for item in data[key]:
            binding = records[item["id"]]
            if (
                canonical_sha256(item) != binding["payload_sha256"]
                or canonical_sha256(item["question"]) != binding["public_task_sha256"]
            ):
                raise ValueError("A benchmark payload differs from the parent's partition")


def score_only(row: Mapping) -> dict:
    """Only this projection may return from held-out evaluation to optimizers."""
    return {
        "id": row["id"], "hard": row["hard"], "soft": float(row["hard"]),
        "usage": dict(row["usage"]), "task_type": "integer_math",
        "response": "", "question": "", "predicted_answer": "", "fail_reason": "",
        "agent_ok": True, "n_turns": 1, "validation_evidence_withheld": True,
    }


def usage_sum(values: Sequence[Mapping]) -> dict[str, int]:
    total = dict.fromkeys(USAGE_FIELDS, 0)
    for row in values:
        if (
            any(type(row.get(key)) is not int or row[key] < 0 for key in USAGE_FIELDS)
            or row["prompt_tokens"] + row["completion_tokens"] != row["total_tokens"]
        ):
            raise ValueError("Native provider cost is incomplete")
        for key in total:
            total[key] += row[key]
    return total


def validate_policy_freeze(freeze: Mapping, manifest_sha256: str) -> None:
    if (
        freeze.get("manifest_sha256") != manifest_sha256
        or set(freeze.get("policies", {})) != set(METHODS)
        or freeze.get("test_opened") is not False
    ):
        raise ValueError("All four policies must freeze against the same unopened test split")
    for method, row in freeze["policies"].items():
        if row.get("status") != "PASS" or not row.get("policy_sha256"):
            raise ValueError("Missing audited policy: " + method)
        if method in ("skillopt", "autosaddler") and row.get("official_rounds") != 4:
            raise ValueError("Official baselines must complete four rounds")
        if row.get("training_ids") != freeze["training_ids"]:
            raise ValueError("Method policy used a different training population")


def validate_ledger(ledger: Mapping, train_ids: Sequence[str]) -> None:
    """A small learned ledger cites training observations, not answer templates."""
    if set(ledger) != {"rules", "diagnosis"} or not 1 <= len(ledger["rules"]) <= 6:
        raise ValueError("Expected one through six training-grounded constraint rules")
    for rule in ledger["rules"]:
        if set(rule) != {"rule_id", "check", "evidence_ids"}:
            raise ValueError("Unexpected rule fields")
        if (
            not isinstance(rule["check"], str) or not 20 <= len(rule["check"]) <= 800
            or not rule["evidence_ids"]
            or not set(rule["evidence_ids"]).issubset(train_ids)
            or any(item_id in rule["check"] for item_id in train_ids)
        ):
            raise ValueError("Rule lacks training provenance or embeds example identity")


def select_candidate(scores: Sequence[Mapping]) -> Mapping:
    """Predetermined score, then actual execution cost, then ID tie breaking."""
    if not scores or len({row["candidate_id"] for row in scores}) != len(scores):
        raise ValueError("Missing or duplicate candidate validation score")
    if any(row["tasks"] != 6 for row in scores):
        raise ValueError("Every frozen candidate requires all six validation problems")
    return min(scores, key=lambda row: (-row["correct"], row["total_tokens"], row["candidate_id"]))
