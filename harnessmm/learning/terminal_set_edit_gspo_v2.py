"""Successor Terminal-Bench harness frozen from prior failure analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


VALIDATION_REMINDER = (
    "Validation reserve is active. Stop broad exploration and repeated editing. "
    "Inspect the current artifact, run the task-named syntax/compiler/runtime check, "
    "repair one concrete failure if needed, and finish after a passing check."
)


@dataclass(frozen=True)
class TerminalHarnessV2:
    action_id: str
    operations: tuple[str, ...]
    max_turns: int
    max_output_tokens: int
    components: tuple[tuple[str, str], ...]
    duplicate_command_policy: str
    reserve_validation_turns: int

    @property
    def component_map(self) -> dict[str, str]:
        return dict(self.components)

    @property
    def tool_policy(self) -> dict[str, object]:
        return {
            "duplicate_command_policy": self.duplicate_command_policy,
            "reserve_validation_turns": self.reserve_validation_turns,
            "validation_reminder": VALIDATION_REMINDER,
        }

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "action_id": self.action_id,
            "operations": self.operations,
            "max_turns": self.max_turns,
            "max_output_tokens": self.max_output_tokens,
            "components": self.components,
            "tool_policy": self.tool_policy,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def render(self, base_prompt: str) -> str:
        sections = "\n\n".join(
            f"### {name.replace('_', ' ').title()}\n{text}"
            for name, text in self.components
        )
        return (
            base_prompt.rstrip()
            + "\n\n## Ours-v2: enforced action and validation policy\n"
            + sections
            + "\n\nThe task instruction remains authoritative.\n"
        )


def terminal_v2_candidate() -> TerminalHarnessV2:
    return TerminalHarnessV2(
        action_id="adm_action_guard_16",
        operations=(
            "ADD:exact_action_hash_guard",
            "ADD:reserved_final_validation",
            "DELETE:consecutive_exact_command_repeats",
            "MODIFY:max_turns_12_to_16",
            "MODIFY:max_output_tokens_4096_to_3072",
        ),
        max_turns=16,
        max_output_tokens=3072,
        components=(
            (
                "execution_component",
                "Inspect the working directory, inputs, and installed tools before editing. "
                "Prefer the smallest direct artifact change that satisfies the task.",
            ),
            (
                "action_ledger_component",
                "Treat an exact consecutive command as already completed. If the harness "
                "rejects a duplicate, inspect the prior result and choose a materially "
                "different action rather than rephrasing or retrying it.",
            ),
            (
                "efficiency_component",
                "Use installed tools and local files before network installation. Make at "
                "most one attempt for an unavailable dependency, then pivot to an offline "
                "or direct implementation.",
            ),
            (
                "validation_component",
                "Preserve the final validation reserve. Run the requested entry point, "
                "syntax checker, compiler, or format parser against the actual artifact; "
                "repair one diagnosed failure and stop after validation passes.",
            ),
        ),
        duplicate_command_policy="reject_consecutive_exact",
        reserve_validation_turns=3,
    )


__all__ = ["TerminalHarnessV2", "VALIDATION_REMINDER", "terminal_v2_candidate"]
