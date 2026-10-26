"""SoL-Pi-inspired exact recall after two successful full model exposures."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any, Mapping, Sequence

from phiagent.harness.observation_pack import ObservationPack


LOGGER = logging.getLogger(__name__)
RECALL_TOOL = "CHISELMemory__recall"
RECALL_PREFIX = "[Exact archived observation page]"
RECEIPT_PREFIX = "[Archived public observation]"


@dataclass(frozen=True)
class PreparedProjection:
    messages: list[dict[str, Any]]
    request_index: int
    full_events: tuple[str, ...]
    audit: dict[str, Any]


class DeferredObservationPack(ObservationPack):
    """Keep original history and native error/recall outputs; archive only pure text."""

    minimum_bytes = 10 * 1024
    full_sends = 2

    def __init__(self, directory):
        super().__init__(directory)
        self._exposures: dict[str, int] = {}
        self._pending: dict[int, tuple[str, ...]] = {}
        self._sequence = 0

    def prepare(self, messages: Sequence[Mapping[str, Any]]) -> PreparedProjection:
        projected = [dict(message) for message in messages]
        full, events = [], []
        for index, message in enumerate(messages):
            text = message.get("content")
            if (message.get("role") != "tool" or not isinstance(text, str)
                    or message.get("is_error") or message.get("name") == RECALL_TOOL
                    or text.startswith((RECALL_PREFIX, RECEIPT_PREFIX))
                    or "sol_pi_evidence_receipt_v1" in text.splitlines()
                    or len(text.encode("utf-8")) <= self.minimum_bytes):
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Packing requires a stable native observation event ID")
            event = hashlib.sha256(json.dumps(
                [call_id, message.get("name"), hashlib.sha256(text.encode()).hexdigest()],
                ensure_ascii=False,
            ).encode()).hexdigest()
            sent = self._exposures.get(event, 0)
            try:
                observation_id = self._archive(text)
            except (OSError, UnicodeError, ValueError) as error:
                self._stats["archive_failures"] += 1
                LOGGER.warning("Preserving the original observation after archive failure: %s", error)
                events.append({"message": index, "status": "ORIGINAL_PRESERVED_ARCHIVE_ERROR",
                               "error": repr(error)})
                continue
            if sent < self.full_sends:
                full.append(event)
                events.append({"message": index, "status": "FULL_EXPOSURE_REQUIRED",
                               "completed_full_exposures": sent, "observation_id": observation_id})
                continue
            receipt = (
                f"{RECEIPT_PREFIX}\n"
                f"id={observation_id}; characters={len(text)}; sha256={observation_id[4:]}\n"
                "The exact original was supplied in two earlier model requests. It remains "
                "available, including every omitted value, through " + RECALL_TOOL + ". "
                "This preview is incomplete; do not infer that unshown facts are absent.\n"
                "Recall arguments: " + json.dumps({"observation_id": observation_id,
                                                  "offset": 0, "limit": 8000}) + "\n"
                "Verbatim head:\n" + text[:240] + "\nVerbatim tail:\n" + text[-240:]
            )
            if len(receipt) >= len(text):
                full.append(event)
                events.append({"message": index, "status": "ORIGINAL_PRESERVED_NO_SAVING"})
                continue
            self._read(observation_id)
            projected[index]["content"] = receipt
            self._stats["projected_messages"] += 1
            self._stats["projected_characters_saved"] += len(text) - len(receipt)
            events.append({"message": index, "status": "EXACT_RECALL_RECEIPT",
                           "completed_full_exposures": sent, "observation_id": observation_id,
                           "source_characters": len(text), "receipt_characters": len(receipt)})
        self._sequence += 1
        identifiers = tuple(dict.fromkeys(full))
        self._pending[self._sequence] = identifiers
        return PreparedProjection(projected, self._sequence, identifiers, {
            "mechanism": "deferred_exact_observation_recall_v1",
            "minimum_full_model_exposures": self.full_sends, "events": events,
            "projected_observations": sum(row["status"] == "EXACT_RECALL_RECEIPT" for row in events),
            "first_use_evidence_removed": False, "native_history_modified": False,
            "new_model_calls_for_projection": 0,
        })

    def commit(self, projection: PreparedProjection, usage: Mapping[str, Any]) -> None:
        values = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if (any(type(value) is not int or value < 0 for value in values)
                or values[0] + values[1] != values[2] or values[2] <= 0):
            raise ValueError("A successful full model exposure needs exact positive provider usage")
        events = self._pending.pop(projection.request_index)
        if events != projection.full_events:
            raise ValueError("Prepared observation identities changed")
        for event in events:
            self._exposures[event] = self._exposures.get(event, 0) + 1

    def abort(self, projection: PreparedProjection) -> None:
        self._pending.pop(projection.request_index)

    def recall(self, observation_id: str, offset: int = 0, limit: int = 8000) -> str:
        return RECALL_PREFIX + "\n" + super().recall(observation_id, offset, limit)

    def report(self):
        return {**super().report(), "policy": "deferred_exact_observation_recall_v1",
                "full_sends": self.full_sends, "threshold_bytes": self.minimum_bytes,
                "keep_recent": None, "maximum_inline_characters": None,
                "event_exposures": dict(self._exposures), "pending_model_requests": len(self._pending),
                "character_savings_are_not_provider_token_savings": True}
