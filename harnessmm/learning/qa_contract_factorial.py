"""Prospective common-skeleton QA controls; separate from the frozen TRAIN64 prompts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace


PREFIX = "Use the supplied context to answer the question."
SUFFIX = "Return only a concise, context-supported answer inside <answer>...</answer>."
TARGET_RULE = (
    "Identify the exact entity, value or phrase requested by the question. "
    "Do not substitute an entity that is merely mentioned as an anchor."
)
CONSTRAINT_RULE = (
    "Check the proposed answer against every explicit qualifier in the question: "
    "time, place, role, count, comparison and negation."
)
NEUTRAL_HEADER = "Ignore this neutral filler:"
NEUTRAL_UNIT = " x"


@dataclass(frozen=True)
class FactorialContract:
    target: str = ""
    constraint: str = ""

    def render(self) -> str:
        return (
            f"{PREFIX}\n\nTarget rule:\n{self.target}\n\n"
            f"Constraint rule:\n{self.constraint}\n\n{SUFFIX}\n"
        )


def factorial_contracts() -> dict[str, FactorialContract]:
    return {
        "neither": FactorialContract(),
        "target_only": FactorialContract(target=TARGET_RULE),
        "constraint_only": FactorialContract(constraint=CONSTRAINT_RULE),
        "target_and_constraint": FactorialContract(TARGET_RULE, CONSTRAINT_RULE),
    }


def length_matched_neutral(
    contract: FactorialContract,
    *,
    remove: str,
    count_complete_prompt: Callable[[str], int],
) -> tuple[FactorialContract, dict]:
    """The caller must count the full served chat template with the same public task."""
    if remove not in {"target", "constraint"} or not getattr(contract, remove):
        raise ValueError("one present factor must be removed")
    target = count_complete_prompt(contract.render())
    if type(target) is not int or target <= 0:
        raise ValueError("the complete native prompt count must be a positive integer")
    low, high = 0, target
    while low <= high:
        units = (low + high) // 2
        controlled = replace(contract, **{remove: NEUTRAL_HEADER + NEUTRAL_UNIT * units})
        measured = count_complete_prompt(controlled.render())
        if type(measured) is not int or measured < 0:
            raise ValueError("native prompt count is invalid")
        if measured == target:
            return controlled, {
                "removed_factor": remove, "padding_units": units,
                "native_prompt_tokens": target, "exact_length_match": True,
                "neutrality_is_not_assumed": True,
            }
        if measured < target:
            low = units + 1
        else:
            high = units - 1
    raise ValueError("exact tokenizer-length matching is unavailable; no approximate padding")
