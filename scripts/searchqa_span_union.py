"""Pack exact source spans without repeatedly paying for overlapping text."""
from __future__ import annotations

from scripts import figure2_searchqa_ranker as ranker
from scripts import figure2_searchqa_recovery as recovery


def union_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not 0 <= start < end:
            raise ValueError("Source spans must be nonempty nonnegative intervals")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def request(item: dict, predictor: dict, count, selection_cap: int = 640,
            extra_instruction: str = "") -> tuple[list[dict], dict]:
    context = item["context"]
    scored = []
    for start, end in ranker.units(context):
        vector = ranker.features(
            item["question"], context[start:end], start / max(len(context), 1),
        )
        scored.append((sum(w * x for w, x in zip(predictor["weights"], vector)), start, end))

    def render(spans):
        return "\n...\n".join(context[start:end] for start, end in spans)

    def budget_messages(spans):
        return ranker.messages(
            item, render(spans), recovery.CONTRACT_SYSTEM + extra_instruction,
        )

    selected: list[tuple[int, int]] = []
    admitted: list[tuple[int, int]] = []
    for _, start, end in sorted(scored, key=lambda row: (-row[0], row[1])):
        merged = union_spans(selected + [(start, end)])
        if merged != selected and count(budget_messages(merged)) <= selection_cap:
            selected = merged
            admitted.append((start, end))
    if not selected:
        raise ValueError("Question and one intact source span exceed the frozen input cap")
    messages = ranker.messages(
        item, render(selected), recovery.TARGET_ROLE_SYSTEM + extra_instruction,
    )
    return messages, {
        "spans": selected, "context_sha256": ranker.digest(context), "source_only": True,
        "source_selection_input_cap": selection_cap, "actual_input_tokens": count(messages),
        "source_selection_input_tokens": count(budget_messages(selected)),
        "selection_model_calls": 0, "admitted_windows": admitted,
        "overlap_characters_not_repeated": (
            sum(end - start for start, end in admitted)
            - sum(end - start for start, end in selected)
        ),
        "mechanism": "exact_source_interval_union_then_budget_refill",
        "extra_instruction_tokens_paid": bool(extra_instruction),
    }
