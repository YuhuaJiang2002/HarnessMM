"""Prospective, action-conditioned exact-recall routing; no model or benchmark imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from phiagent.harness.observation_pack import ObservationPack


LOGGER = logging.getLogger(__name__)
RECEIPT_PREFIX = "[Recovery-aware archived observation]"
RECALL_PREFIX = "[Exact archived observation page]"
_WORDS = re.compile(r"[\w./:@-]{3,}", re.UNICODE)
_COMMON = frozenset(
    "the and for from with this that then into are was were have has had not you your "
    "tool tools read write list search action action_input path file name value limit offset "
    "true false null observation result content text data input output".split()
)


def _terms(text: str) -> set[str]:
    return {word.casefold() for word in _WORDS.findall(text)} - _COMMON


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RecoveryRoutingPolicy:
    mode: str = "recovery_aware"
    minimum_tokens: int = 256
    full_exposures: int = 2
    keep_recent: int = 2
    future_exposures: int = 4
    base_need_probability: float = 0.10
    relevance_need_probability: float = 0.50
    recovery_completion_tokens: int = 128
    cost_margin: float = 1.25

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "relevance_only", "recovery_aware"):
            raise ValueError("Unknown recovery-routing mode")
        for name in ("minimum_tokens", "full_exposures", "keep_recent", "future_exposures"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.recovery_completion_tokens) is not int or self.recovery_completion_tokens < 0:
            raise ValueError("recovery_completion_tokens must be a nonnegative integer")
        for name in ("base_need_probability", "relevance_need_probability", "cost_margin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.base_need_probability <= self.relevance_need_probability <= 1:
            raise ValueError("Recovery-need priors must be ordered probabilities in [0, 1]")
        if self.cost_margin < 1:
            raise ValueError("The cost margin must be at least one")


@dataclass(frozen=True)
class RecoveryProjection:
    messages: list[dict[str, Any]]
    request_index: int
    full_events: tuple[str, ...]
    audit: dict[str, Any]


class RecoveryAwareObservationPack(ObservationPack):
    """Keep first exposures and exact archives; use declared, uncalibrated need priors."""

    def __init__(
        self, directory: Path, *, policy: RecoveryRoutingPolicy,
        count_tokens: Callable[[str], int], recall_tool: str = "ObsRecall",
    ) -> None:
        if not isinstance(policy, RecoveryRoutingPolicy) or not callable(count_tokens):
            raise TypeError("A checked routing policy and native tokenizer callback are required")
        if not isinstance(recall_tool, str) or not re.fullmatch(r"[A-Za-z]\w*", recall_tool):
            raise ValueError("A safe recall-tool name is required")
        super().__init__(directory)
        self._policy = policy
        self.recall_tool = recall_tool
        self._count_tokens = count_tokens
        self._token_cache: dict[str, int] = {}
        self._exposures: dict[str, int] = {}
        self._recalled: set[str] = set()
        self._pending: dict[int, tuple[RecoveryProjection, str]] = {}
        self._sequence = 0
        self._paid_calls = 0
        self._paid_tokens = 0
        self._committed_projections = 0

    @property
    def policy(self) -> RecoveryRoutingPolicy:
        return self._policy

    def _tokens(self, text: str) -> int:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self._token_cache:
            count = self._count_tokens(text)
            if type(count) is not int or count < 0:
                raise ValueError("The native tokenizer must return a nonnegative integer")
            self._token_cache[key] = count
        return self._token_cache[key]

    def _reference(self, identifier: str, text: str, message: Mapping[str, Any]) -> str:
        return (
            f"{RECEIPT_PREFIX}\n"
            f"id={identifier}; tool={message.get('name', '')}; "
            f"event={message['tool_call_id']}; characters={len(text)}\n"
            "This is an incomplete preview of previously observed evidence, not current state. "
            "Omitted facts are not absent. Read exact pages before using an omitted value; "
            "a read does not repeat the original environment action.\n"
            f"Call {self.recall_tool} with "
            + json.dumps({"observation_id": identifier, "offset": 0, "limit": 8000})
            + "; follow next_offset.\nVerbatim head:\n"
            + text[:self.preview_characters]
            + "\nVerbatim tail:\n" + text[-self.preview_characters:]
        )

    def prepare(
        self, messages: Sequence[Mapping[str, Any]], *, task_text: str = "",
        action_text: str = "", required_terms: Sequence[str] = (),
        request_overhead_tokens: int = 0,
    ) -> RecoveryProjection:
        if self._pending:
            raise RuntimeError("Settle or abort the preceding provider request before routing again")
        if not isinstance(task_text, str) or not isinstance(action_text, str):
            raise TypeError("Routing context must be public text")
        if (isinstance(required_terms, str)
                or any(not isinstance(term, str) or not term for term in required_terms)):
            raise ValueError("Required dependencies must be nonempty strings")
        if type(request_overhead_tokens) is not int or request_overhead_tokens < 0:
            raise ValueError("Request overhead must be a nonnegative native-token count")
        projected = [dict(message) for message in messages]
        tool_indices = [index for index, message in enumerate(messages)
                        if message.get("role") == "tool"]
        recent = set(tool_indices[-self.policy.keep_recent:])
        task_terms, action_terms = _terms(task_text), _terms(action_text)
        full: list[str] = []
        events: list[dict[str, Any]] = []
        removed = 0
        for index in tool_indices:
            message = messages[index]
            text = message.get("content")
            if not isinstance(text, str):
                continue
            if (message.get("is_error") or message.get("name") == self.recall_tool
                    or text.startswith((RECALL_PREFIX, RECEIPT_PREFIX))
                    or "sol_pi_evidence_receipt_v1" in text.splitlines()
                    or re.search(r"(?i)(?:error:|traceback \(|\"error\"\s*:)", text)):
                events.append({"message": index, "status": "PROTECTED_ERROR_OR_RECALL"})
                continue
            tokens = self._tokens(text)
            if tokens < self.policy.minimum_tokens:
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("A stable tool-call event identity is required")
            event = _digest([call_id, message.get("name"), hashlib.sha256(text.encode()).hexdigest()])
            row: dict[str, Any] = {"message": index, "event": event, "source_tokens": tokens}
            events.append(row)
            try:
                identifier = self._archive(text)
            except (OSError, UnicodeError, ValueError) as error:
                self._stats["archive_failures"] += 1
                LOGGER.warning("Keeping full recovery-routing input after archive failure: %s", error)
                row.update(status="ORIGINAL_PRESERVED_ARCHIVE_ERROR", error=repr(error))
                continue
            row["observation_id"] = identifier
            full.append(event)
            if self._exposures.get(event, 0) < self.policy.full_exposures:
                row["status"] = "FULL_EXPOSURE_REQUIRED"
                continue
            if index in recent or identifier in self._recalled:
                row["status"] = "PROTECTED_RECENT_OR_RECOVERED"
                continue
            content_terms = _terms(text)
            required = any(term.casefold() in text.casefold() for term in required_terms)
            action_overlap = sorted(content_terms & action_terms)
            task_overlap = len(content_terms & task_terms) / max(1, len(task_terms))
            row.update(action_dependencies=action_overlap, task_overlap_fraction=task_overlap)
            if required or (self.policy.mode != "fixed" and action_overlap):
                row["status"] = "PROTECTED_ACTION_DEPENDENCY"
                continue
            receipt = self._reference(identifier, text, message)
            receipt_tokens = self._tokens(receipt)
            saving = tokens - receipt_tokens
            need = (self.policy.base_need_probability
                    + (self.policy.relevance_need_probability - self.policy.base_need_probability)
                    * task_overlap)
            pages = math.ceil(len(text) / self.maximum_page_characters)
            paid_mean = self._paid_tokens / self._paid_calls if self._paid_calls else 0.0
            recovery = pages * (paid_mean + self.policy.recovery_completion_tokens) + tokens
            predicted_saved = self.policy.future_exposures * saving
            predicted_cost = (need * recovery * self.policy.cost_margin
                              + self.policy.future_exposures * request_overhead_tokens)
            row.update(
                receipt_tokens=receipt_tokens, content_tokens_removed=saving,
                declared_need_prior=need, recovery_pages_upper_bound=pages,
                observed_mean_paid_call_tokens=paid_mean,
                estimated_recovery_tokens=recovery,
                estimated_future_content_saving=predicted_saved,
                estimated_future_recovery_and_overhead=predicted_cost,
                priors_are_calibrated=False,
            )
            if saving <= 0:
                row["status"] = "ORIGINAL_PRESERVED_NO_SAVING"
                continue
            if self.policy.mode == "recovery_aware" and predicted_saved <= predicted_cost:
                row["status"] = "ORIGINAL_PRESERVED_RECOVERY_COST"
                continue
            projected[index]["content"] = receipt
            full.remove(event)
            row["status"] = "EXACT_RECALL_RECEIPT"
            removed += saving
        if removed <= request_overhead_tokens:
            for row in events:
                if row["status"] == "EXACT_RECALL_RECEIPT":
                    projected[row["message"]] = dict(messages[row["message"]])
                    full.append(row["event"])
                    row["status"] = "ORIGINAL_PRESERVED_REQUEST_OVERHEAD"
            removed = 0
        self._sequence += 1
        audit = {
            "mechanism": "action_conditioned_recovery_routing_v1",
            "policy": asdict(self.policy), "policy_sha256": _digest(asdict(self.policy)),
            "events": events, "projected_observations":
                sum(row["status"] == "EXACT_RECALL_RECEIPT" for row in events),
            "native_content_tokens_removed": removed,
            "request_overhead_tokens": request_overhead_tokens,
            "estimates_are_not_provider_savings": True,
            "priors_are_calibrated": False, "native_history_modified": False,
            "first_use_evidence_removed": False, "new_model_calls_for_routing": 0,
        }
        value = RecoveryProjection(projected, self._sequence, tuple(dict.fromkeys(full)), audit)
        self._pending[value.request_index] = (value, _digest(projected))
        return value

    def commit(self, projection: RecoveryProjection, usage: Mapping[str, Any]) -> None:
        counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if (any(type(count) is not int or count < 0 for count in counts)
                or counts[0] + counts[1] != counts[2] or counts[2] <= 0):
            raise ValueError("Exact positive paid-provider usage is required")
        pending, fingerprint = self._pending[projection.request_index]
        if pending is not projection or _digest(projection.messages) != fingerprint:
            raise ValueError("The prepared provider projection changed before accounting")
        for event in projection.full_events:
            self._exposures[event] = self._exposures.get(event, 0) + 1
        self._paid_calls += 1
        self._paid_tokens += counts[2]
        self._committed_projections += projection.audit["projected_observations"]
        self._pending.pop(projection.request_index)

    def abort(self, projection: RecoveryProjection) -> None:
        pending, _ = self._pending[projection.request_index]
        if pending is not projection:
            raise ValueError("The aborted projection does not match the pending request")
        self._pending.pop(projection.request_index)

    def recall(self, observation_id: str, offset: int = 0, limit: int = 8000) -> str:
        result = super().recall(observation_id, offset, limit)
        self._recalled.add(observation_id)
        return result

    def report(self) -> dict[str, Any]:
        archive = super().report()
        for name in ("minimum_characters", "keep_recent", "maximum_inline_characters",
                     "projected_messages", "projected_characters_saved"):
            archive.pop(name)
        return {
            **archive, "policy": asdict(self.policy),
            "policy_sha256": _digest(asdict(self.policy)),
            "report_schema": "recovery-aware-observation-pack/v1",
            "minimum_content_tokens": self.policy.minimum_tokens,
            "keep_recent_observations": self.policy.keep_recent,
            "paid_actor_calls": self._paid_calls, "paid_actor_tokens": self._paid_tokens,
            "paid_projected_observations": self._committed_projections,
            "pending_model_requests": len(self._pending),
            "recovered_observations": len(self._recalled),
            "priors_are_calibrated": False, "estimates_are_not_provider_savings": True,
        }
