"""CHISEL's supported-answer controller adapted to public VQA answer contracts.

The draft sees the same full image and question as the original DSH actor.
These local format/type checks do not establish visual correctness. Native VQA
accuracy must select the frozen candidate on validation before test release.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from phiagent.learning.supported_local_decision import (
    SupportedCandidates, execute_supported_decision,
)


ADAPTER_ID = "chisel-dsh-vqa-typed-answer-v1"
_BYPASS = re.compile(r"\b(?:not|except|least|most|different|between|rather than)\b", re.I)
_YES_NO = re.compile(r"^(?:is|are|does|do|has|have|can|was|were|will)\b", re.I)
_COUNT = re.compile(r"^how many\b", re.I)
_COLOR = re.compile(r"^what (?:is the |are the )?colou?rs?\b", re.I)
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
)
_COLOR_WORDS = frozenset(
    "black white red green blue yellow orange purple pink brown gray grey silver "
    "gold beige tan turquoise maroon violet multicolored multi-colored light dark and".split()
)


@dataclass(frozen=True)
class Answer:
    text: str
    complete: bool
    artifact_id: str

    def __post_init__(self):
        if (
            not isinstance(self.text, str) or type(self.complete) is not bool
            or not isinstance(self.artifact_id, str) or not self.artifact_id
        ):
            raise ValueError("An answer requires raw text, completion status and a recorded call ID")


def public_answer_type(question: str) -> str | None:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The public VQA question must be a nonempty string")
    question = question.strip()
    if _BYPASS.search(question):
        return None
    for expression, answer_type in ((_YES_NO, "yes_no"), (_COUNT, "count"), (_COLOR, "color")):
        if expression.search(question):
            return answer_type
    return None


def draft_supported(answer: Answer, answer_type: str) -> bool:
    if answer_type not in ("yes_no", "count", "color"):
        raise ValueError("Unsupported public answer type")
    if not answer.complete or "\n" in answer.text or "\r" in answer.text:
        return False
    text = answer.text.strip().casefold().rstrip(".")
    if answer_type == "yes_no":
        return text in ("yes", "no")
    if answer_type == "count":
        return bool(re.fullmatch(r"[0-9]+", text)) or text in _NUMBER_WORDS
    words = text.split()
    return (
        1 <= len(words) <= 4 and set(words) <= _COLOR_WORDS
        and bool(set(words) - {"light", "dark", "and"})
    )


def answer_with_chisel(
    question: str,
    *,
    draft: Callable[[], Answer],
    original_actor: Callable[[], Answer],
) -> tuple[Answer, dict]:
    answer_type = public_answer_type(question)
    calls = []
    proposed = None
    if answer_type is not None:
        proposed = draft()
        calls.append({"role": "draft", "artifact_id": proposed.artifact_id})
    supported = proposed is not None and draft_supported(proposed, answer_type)

    def fallback() -> Answer:
        value = original_actor()
        calls.append({"role": "original_actor", "artifact_id": value.artifact_id})
        return value

    def commit(selected: str) -> Answer:
        if proposed is None or selected != proposed.artifact_id:
            raise ValueError("Selected draft is not the recorded public candidate")
        return proposed

    decision = execute_supported_decision(
        SupportedCandidates(
            supported=supported,
            actions=(proposed.artifact_id,) if proposed is not None else (),
            reason=(
                "Complete draft satisfies the declared public answer format"
                if supported else "Unsupported question form or rejected/truncated draft"
            ),
        ),
        commit=commit, fallback=fallback,
    )
    return decision.value, {
        "adapter_id": ADAPTER_ID, "public_answer_type": answer_type,
        "decision": decision.trace.to_dict(), "model_calls": calls,
        "model_call_count": len(calls), "visual_correctness_checked_locally": False,
        "candidate_text_changed": False,
    }
