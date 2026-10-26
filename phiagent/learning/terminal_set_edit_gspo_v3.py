"""Duplicate-budget successor to the Terminal-Bench action-guard harness."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from phiagent.learning.terminal_set_edit_gspo_v2 import VALIDATION_REMINDER


@dataclass(frozen=True)
class TerminalHarnessV3:
    action_id: str
    operations: tuple[str, ...]
    max_turns: int
    max_output_tokens: int
    components: tuple[tuple[str, str], ...]
    duplicate_command_policy: str
    max_duplicate_rejections: int
    reserve_validation_turns: int

    @property
    def tool_policy(self) -> dict[str, object]:
        return {
            "duplicate_command_policy": self.duplicate_command_policy,
            "max_duplicate_rejections": self.max_duplicate_rejections,
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
            + "\n\n## Ours-v3: bounded action and early validation policy\n"
            + sections
            + "\n\nThe task instruction remains authoritative.\n"
        )


def terminal_v3_candidate() -> TerminalHarnessV3:
    return TerminalHarnessV3(
        action_id="adm_duplicate_budget_16",
        operations=(
            "ADD:exact_action_hash_guard",
            "ADD:duplicate_rejection_budget_2",
            "ADD:early_validation_reserve",
            "DELETE:post_budget_agent_turns",
            "MODIFY:validation_reminder_turn_14_to_10",
        ),
        max_turns=16,
        max_output_tokens=3072,
        components=(
            (
                "execution_component",
                "Inspect inputs and installed tools before editing. Make the smallest "
                "direct artifact change, then preserve completed work.",
            ),
            (
                "action_budget_component",
                "An exact duplicate rejection means the action is exhausted. Inspect the "
                "prior output and change approach immediately; after two such rejections "
                "the harness ends execution and preserves the current artifact for scoring.",
            ),
            (
                "efficiency_component",
                "Use local tools before network installation. Do not spend remaining turns "
                "on repeated commands, unchanged dependency attempts, or repeated file reads.",
            ),
            (
                "validation_component",
                "Start final validation when the reserve reminder appears. Run the requested "
                "entry point, parser, compiler, or syntax checker and repair one concrete "
                "failure before finishing.",
            ),
        ),
        duplicate_command_policy="reject_consecutive_exact",
        max_duplicate_rejections=2,
        reserve_validation_turns=7,
    )


__all__ = ["TerminalHarnessV3", "terminal_v3_candidate"]
