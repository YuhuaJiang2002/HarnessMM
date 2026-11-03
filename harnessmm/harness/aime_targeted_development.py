"""Pure protocol helpers for a draft-visibility development experiment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import threading
from typing import Any
from urllib.parse import urlsplit


ARMS = ("draft_quarantine", "draft_visible_control")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
MISSING_DRAFT = "[No proposed derivation is supplied.]"
REVIEW_EXTENSION = (
    "\nIf no proposed derivation is supplied, independently solve and verify "
    "the original problem."
)
NATIVE_REQUEST_TIMEOUT_SECONDS = 7200
CAPTURE_RESPONSE_GRACE_SECONDS = 60


class SingleInFlight:
    """Reject concurrent admission before a request can enter the model queue."""

    def __init__(self):
        self._lock = threading.Lock()

    @contextmanager
    def admit(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Only one inference request may be in flight per replica")
        try:
            yield
        finally:
            self._lock.release()


def inference_timeout(url: str, resource: Mapping, budget: Mapping, default: int) -> int:
    """Extend only the owned inference endpoints, leaving readiness probes alone."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
        or not parsed.path.endswith("/v1/chat/completions")
    ):
        return default
    if parsed.port not in (resource["model_port"], resource["capture_port"]):
        return default
    seconds = budget["request_timeout_seconds"]
    if type(seconds) is not int or seconds < NATIVE_REQUEST_TIMEOUT_SECONDS:
        raise ValueError("Full-capacity serialized inference requires at least 7200 seconds")
    return seconds + (
        CAPTURE_RESPONSE_GRACE_SECONDS if parsed.port == resource["capture_port"] else 0
    )


def validate_population(data: Mapping[str, Any], expected_ids: Sequence[str]) -> None:
    """Reject any final data, hidden training population, or changed task order."""
    if set(data) != {"train", "development", "test"} or data["train"] or data["test"]:
        raise ValueError("This run permits the frozen development population only")
    ids = [item["id"] for item in data["development"]]
    if (
        len(ids) != 10
        or len(set(ids)) != 10
        or ids != list(expected_ids)
        or any(not value.startswith("2024-") for value in ids)
    ):
        raise ValueError("Expected the complete ordered AIME2024 development10")


def review_problem(question: str, draft: str, arm: str) -> str:
    """Change only whether the verifier can read the generated derivation."""
    if arm not in ARMS:
        raise ValueError("Unknown draft-visibility arm")
    evidence = MISSING_DRAFT if arm == "draft_quarantine" else draft
    return (
        "ORIGINAL PROBLEM:\n" + question
        + "\n\nPROPOSED DERIVATION (may contain errors):\n" + evidence
    )


def usage_sum(*usages: Mapping[str, Any]) -> dict[str, int]:
    result = dict.fromkeys(USAGE_FIELDS, 0)
    for usage in usages:
        if any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_FIELDS):
            raise ValueError("Invalid provider token accounting")
        if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
            raise ValueError("Provider token totals differ")
        for key in USAGE_FIELDS:
            result[key] += usage[key]
    return result


def paired_rows(
    drafts: Sequence[Mapping[str, Any]], reviews: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    draft_map = {row["id"]: row for row in drafts}
    review_map = {row["id"]: row for row in reviews}
    if len(draft_map) != len(drafts) or len(review_map) != len(reviews):
        raise ValueError("Duplicate paired task")
    if set(draft_map) != set(review_map):
        raise ValueError("Missing paired task")
    return [
        {
            "id": item_id,
            "draft_correct": draft["hard"],
            "review_correct": review_map[item_id]["hard"],
            "draft_answer": draft["predicted_answer"],
            "review_answer": review_map[item_id]["predicted_answer"],
            "transition": f'{draft["hard"]}->{review_map[item_id]["hard"]}',
            "draft_usage": usage_sum(draft["usage"]),
            "review_usage": usage_sum(
                review_map[item_id].get("review_usage", review_map[item_id]["usage"])
            ),
        }
        for item_id, draft in draft_map.items()
    ]


def paired_summary(candidate: Sequence[Mapping], control: Sequence[Mapping]) -> dict:
    pairs = paired_rows(candidate, control)
    transitions = Counter(row["transition"] for row in pairs)
    return {
        "tasks": len(pairs),
        "candidate_correct": sum(row["draft_correct"] for row in pairs),
        "control_correct": sum(row["review_correct"] for row in pairs),
        "candidate_only_correct": transitions["1->0"],
        "control_only_correct": transitions["0->1"],
        "both_correct": transitions["1->1"],
        "both_wrong": transitions["0->0"],
        "unit": "problem; paired complete frozen development population",
    }


def historical_cost_union(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count a physical provider call once across inherited ledger snapshots."""
    calls: dict[tuple[str, int], dict[str, int]] = {}
    ledgers = []
    for binding in bindings:
        root, ledger = binding["root"], binding["ledger"]
        if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
            raise ValueError("Unsettled historical provider usage")
        rows = ledger["calls"]
        if len({row["sequence"] for row in rows}) != len(rows):
            raise ValueError("Duplicate sequence within a historical ledger")
        subtotal = usage_sum(*(row["usage"] for row in rows))
        if subtotal["total_tokens"] != ledger["known_tokens"]:
            raise ValueError("Historical ledger total differs")
        for row in rows:
            if row["status"] != "SETTLED":
                raise ValueError("Unsettled historical provider call")
            identity = (row.get("origin", root), row["sequence"])
            usage = usage_sum(row["usage"])
            if identity in calls and calls[identity] != usage:
                raise ValueError("Conflicting inherited provider usage")
            calls[identity] = usage
        ledgers.append({"root": root, "recorded_usage": subtotal, "calls": len(rows)})
    return {
        "unique_calls": len(calls),
        "usage": usage_sum(*calls.values()),
        "ledgers": ledgers,
        "identity": "origin run directory and provider sequence",
        "new_execution_cost": False,
    }


def normalize_repeat_ledger(summary: Mapping, calls: Sequence[Mapping]) -> dict:
    """Normalize separately bound synthetic-repeat receipts without guessing costs."""
    if (
        type(summary["calls"]) is not int
        or len(calls) != summary["calls"]
        or summary["unknown_usage_calls"]
        or any(row["status"] != "SETTLED" for row in calls)
        or len({row["sequence"] for row in calls}) != len(calls)
    ):
        raise ValueError("Incomplete synthetic-repeat receipts")
    usage = usage_sum(*(row["usage"] for row in calls))
    if usage != usage_sum(summary["usage"]):
        raise ValueError("Repeat receipt total differs from preserved summary")
    return {
        "calls": list(calls), "known_tokens": usage["total_tokens"],
        "unknown_usage_calls": 0, "inflight_reserved_tokens": 0,
        "schema": "normalized complete synthetic-repeat receipts",
    }
