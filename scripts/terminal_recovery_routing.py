"""TRAIN-only exact-recall routing through the existing Terminal observation hook."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from phiagent.harness.recovery_aware_context import (
    RecoveryAwareObservationPack, RecoveryRoutingPolicy,
)
from scripts.terminal_solpi import RECALL_TOOL, TerminalSoLPiSession


class TerminalRecoveryRoutingSession(TerminalSoLPiSession):
    def __init__(
        self, config: Mapping, trial_dir: Path, *, instruction: str,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        if set(config) != {"stage", "policy", "tokenizer_url", "model"} or config["stage"] != "train":
            raise ValueError("Recovery-routing requires a closed TRAIN-only configuration")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("The unchanged public task instruction is required")
        if not re.fullmatch(r"http://127\.0\.0\.1:\d+/tokenize", config["tokenizer_url"]):
            raise ValueError("Native tokenization must use the pinned loopback model service")
        if not isinstance(config["model"], str) or not config["model"]:
            raise ValueError("The pinned tokenizer model identity is required")
        super().__init__({"observation_pack": False, "action_fusion": False}, trial_dir)
        self.instruction = instruction
        self._pending_projection = None
        self._recall_declared = False
        self._routing_events: list[dict] = []
        if count_tokens is None:
            from scripts.run_terminal_bench_qwen35_4b_direct import post_json

            def count_tokens(text):
                response = post_json(
                    config["tokenizer_url"],
                    {"model": config["model"], "prompt": text, "add_special_tokens": False},
                    timeout=30,
                )
                return response["count"]

        self.pack = RecoveryAwareObservationPack(
            trial_dir / "recovery-routing" / "observations",
            policy=RecoveryRoutingPolicy(**config["policy"]), count_tokens=count_tokens,
        )
        self._overhead = self.pack._tokens(json.dumps(RECALL_TOOL, ensure_ascii=False))

    def project(self, messages: Sequence[Mapping]) -> list[dict]:
        compatible = []
        tool_names = {}
        last_action = ""
        for message in messages:
            item = dict(message)
            if item.get("role") == "assistant":
                calls = item.get("tool_calls", [])
                if calls:
                    last_action = json.dumps(calls, ensure_ascii=False)
                for call in calls:
                    tool_names[call["id"]] = call["function"]["name"]
            if item.get("role") == "tool":
                item["name"] = item.get("name", tool_names.get(item.get("tool_call_id"), "native"))
                text = item.get("content")
                if isinstance(text, str):
                    exit_code = re.search(r"(?m)^exit_code=(-?\d+)\s*$", text)
                    item["is_error"] = bool(
                        item.get("is_error") or "timed_out=true" in text
                        or text.startswith("Tool call rejected:")
                        or exit_code and int(exit_code[1]) != 0
                    )
            compatible.append(item)
        projection = self.pack.prepare(
            compatible, task_text=self.instruction, action_text=last_action,
            request_overhead_tokens=self._overhead,
        )
        self._pending_projection = projection
        self._recall_declared |= projection.audit["projected_observations"] > 0
        self._routing_events.append(deepcopy(projection.audit))
        return [
            {**original, "content": selected["content"]} if "content" in selected else dict(original)
            for original, selected in zip(messages, projection.messages)
        ]

    def tools(self, tools: Sequence[Mapping]) -> list[dict]:
        return super().tools(tools) if self._recall_declared else deepcopy(list(tools))

    def record_response(self, usage: object, *, role: str = "actor", complete: bool = True) -> None:
        super().record_response(usage, role=role, complete=complete)
        if self._pending_projection is None:
            raise RuntimeError("A paid Terminal response has no prepared recovery projection")
        if not complete or not isinstance(usage, Mapping):
            self.pack.abort(self._pending_projection)
            self._pending_projection = None
            raise ValueError("Missing Terminal provider usage; no success-shaped fallback")
        self.pack.commit(self._pending_projection, usage)
        self._pending_projection = None

    def report(self) -> dict:
        value = super().report()
        value.update(
            identity="PhiAgent TRAIN-only action-conditioned recovery routing",
            official_solpi_reproduction=False, recovery_routing=True,
            recovery_policy=self.pack.report(), routing_events=list(self._routing_events),
            new_model_calls_for_routing=0, full_callable_contracts_preserved=True,
        )
        return value
