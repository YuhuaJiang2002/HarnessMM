"""Reuse CHISEL's train-only residual learner with the official DeepSeek SDK.

The shared controller lives in the Codex adapter module. This adapter changes
only its runtime labels, prompt field name and optimizer's harness description.
Training, cross-fitted local routing, validation selection, sealed testing and
complete token accounting remain the same implementation. Neither callback may
replace the official DeepSeek agent loop or its native benchmark tools.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from phiagent.harness.codex_chisel import (
    ChiselConfig, CodexChiselController, FrozenPolicy, JSONCallback,
)


SCHEMA = "phiagent-deepseek-chisel/v1"
RUNTIME = "official_deepseek_harness_sdk"


def adapt_request(
    request: Mapping[str, Any], *, optimizer: bool,
    execution_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    value = json.loads(json.dumps(request, allow_nan=False))
    original = value.get("schema")
    value["schema"] = SCHEMA
    value["controller_origin_schema"] = original
    if optimizer:
        objective = value.get("objective")
        generic = (
            isinstance(objective, str)
            and objective.startswith(
                "Compile ONE short, reusable residual instruction for the unchanged official "
                "industrial harness identified in this request."
            )
            and (execution_identity or value.get("identity", {})).get("runtime") == RUNTIME
        )
        if not isinstance(objective, str) or ("Codex" not in objective and not generic):
            raise ValueError("Shared optimizer objective changed; review the runtime adaptation")
        value["objective"] = objective.replace(
            "official Codex CLI", "official DeepSeek Harness SDK",
        ).replace("Codex core", "DeepSeek Harness core").replace(
            "official industrial harness identified in this request", "official DeepSeek Harness SDK",
        )
        if "Codex" in value["objective"]:
            raise ValueError("Unmapped runtime name in shared optimizer objective")
    else:
        if value.pop("agents_md_append", None) != "":
            raise ValueError("DeepSeek SDK interventions use a prompt suffix, not filesystem injection")
        value["prompt"] = value.pop("codex_prompt")
        value["contract"]["runtime"] = "unmodified_official_deepseek_harness_sdk"
        value["contract"]["context_application"] = (
            "system_section" if "system_prompt_append" in value else "prompt_suffix"
        )
    return value


class DeepSeekChiselController:
    def __init__(
        self,
        *,
        partition: Mapping[str, Any],
        task_index: Sequence[Mapping[str, str]],
        execute: JSONCallback,
        optimize: JSONCallback,
        execution_identity: Mapping[str, str],
        optimizer_identity: Mapping[str, str],
        run_id: str,
        config: ChiselConfig = ChiselConfig(application="system_section"),
    ):
        if (
            execution_identity.get("runtime") != RUNTIME
            or execution_identity.get("wire_api") != "anthropic_messages"
            or config.application not in ("prompt_suffix", "system_section")
        ):
            raise ValueError("Official DeepSeek SDK, native Messages and one frozen text intervention are required")
        self._identity = dict(execution_identity)
        self._adapter_sha256: str | None = None
        self._core = CodexChiselController(
            partition=partition, task_index=task_index,
            execute=lambda request: execute(adapt_request(request, optimizer=False)),
            optimize=lambda request: optimize(adapt_request(
                request, optimizer=True, execution_identity=self._identity,
            )),
            execution_identity=execution_identity, optimizer_identity=optimizer_identity,
            run_id=run_id, config=config,
        )

    @property
    def state(self) -> str:
        return self._core.state

    @property
    def candidate_bank(self) -> list[dict[str, Any]]:
        return self._core.candidate_bank

    def train(self, tasks: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        return self._core.train(tasks)

    def select_validation(self, tasks: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        return self._core.select_validation(tasks)

    def freeze(self, *, source_hashes: Mapping[str, str]) -> FrozenPolicy:
        source = Path(__file__).resolve()
        hashes = dict(source_hashes)
        current = hashlib.sha256(source.read_bytes()).hexdigest()
        if "deepseek_adapter" in hashes and hashes["deepseek_adapter"] != current:
            raise ValueError("DeepSeek adapter source differs from the supplied freeze")
        hashes["deepseek_adapter"] = current
        policy = self._core.freeze(source_hashes=hashes)
        self._adapter_sha256 = current
        return policy

    def evaluate_test(self, load_tasks, *, policy_sha256: str, source_sha256: str) -> dict[str, Any]:
        if self._adapter_sha256 is None:
            raise RuntimeError("DeepSeek adapter must freeze before opening test")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._adapter_sha256:
            raise RuntimeError("DeepSeek adapter source changed after freeze")
        result = self._core.evaluate_test(
            load_tasks, policy_sha256=policy_sha256, source_sha256=source_sha256,
        )
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._adapter_sha256:
            raise RuntimeError("DeepSeek adapter source changed during test")
        return result

    def report(self) -> dict[str, Any]:
        source = Path(__file__).resolve()
        return {
            "schema": SCHEMA, "method": "CHISEL", "execution_identity": self._identity,
            "adapter_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "controller_origin": "phiagent.harness.codex_chisel.CodexChiselController",
            "runtime_adaptation": "Native DeepSeek SDK; unchanged shared learner and routing gates",
            "core_report": self._core.report(),
        }


__all__ = ["ChiselConfig", "DeepSeekChiselController", "RUNTIME", "adapt_request"]
