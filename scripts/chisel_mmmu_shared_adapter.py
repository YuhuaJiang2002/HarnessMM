"""Native visual answer adapter for the common supported-decision controller."""
from __future__ import annotations

import re
from collections.abc import Callable

from phiagent.learning.supported_local_decision import (
    SupportedCandidates, execute_supported_decision,
)
from scripts.cvpr_mmmu_campaign import options_list

POLICIES = ("draft_only", "native_format_recovery")
RECOVERY_INSTRUCTION = (
    "The previous draft did not select exactly one of this question's public option letters. "
    "Reconsider the complete original question and every supplied original image. "
    "Return exactly one option letter. Correct formatting alone does not establish correctness."
)


def visual_choice_domain(row):
    choices = options_list(row["options"])
    proof = row["native_prompt"]
    native = (
        bool(row["images"]) and proof["image_tokens"] > 0
        and proof["pixel_tensor_elements"] > 0
    )
    if row["question_type"] != "multiple-choice":
        return None
    if not native:
        return None
    if not 2 <= len(choices) <= 26:
        return None
    return tuple(chr(65 + index) for index in range(len(choices)))


def exact_choice(response, choices):
    if not isinstance(response, str):
        return None
    match = re.fullmatch(r"\s*(?:([A-Z])|\(([A-Z])\))\s*[.]?\s*", response)
    selected = next((part for part in match.groups() if part), None) if match else None
    return selected if selected in choices else None


def visual_decision(row, *, policy, draft: Callable, recover: Callable):
    """Generate the draft outside the kernel, then accept it using format only."""
    if policy not in POLICIES:
        raise ValueError("Unknown frozen visual recovery policy")
    domain = visual_choice_domain(row)
    record = draft()
    calls = [record]
    support = SupportedCandidates(
        domain is not None and exact_choice(record["raw_content"], domain) is not None,
        ("keep",), "format_only",
    )

    def commit(_action):
        return calls[0]

    def fallback():
        if domain is None or policy == "draft_only":
            return calls[0]
        record = recover(calls[0])
        calls.append(record)
        return record

    result = execute_supported_decision(
        support, commit=commit, fallback=fallback,
    )
    return result.value, result.trace.to_dict(), calls
