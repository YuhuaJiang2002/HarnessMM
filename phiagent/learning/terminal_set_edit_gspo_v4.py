"""Progress-aware Terminal-Bench successor and matched component ablations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from phiagent.learning.terminal_set_edit_gspo_v2 import VALIDATION_REMINDER


@dataclass(frozen=True)
class TerminalHarnessV4:
    action_id: str
    operations: tuple[str, ...]
    max_turns: int
    max_output_tokens: int
    components: tuple[tuple[str, str], ...]
    duplicate_command_policy: str
    reserve_validation_turns: int
    ablated_component: str | None = None

    @property
    def tool_policy(self) -> dict[str, object]:
        policy: dict[str, object] = {
            "duplicate_command_policy": self.duplicate_command_policy,
        }
        if self.reserve_validation_turns:
            policy.update(
                {
                    "reserve_validation_turns": self.reserve_validation_turns,
                    "validation_reminder": VALIDATION_REMINDER,
                }
            )
        return policy

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "action_id": self.action_id,
            "operations": self.operations,
            "max_turns": self.max_turns,
            "max_output_tokens": self.max_output_tokens,
            "components": self.components,
            "tool_policy": self.tool_policy,
            "ablated_component": self.ablated_component,
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
            + "\n\n## Ours-v4: progress-aware bounded execution policy\n"
            + sections
            + "\n\nThe task instruction remains authoritative.\n"
        )


def terminal_v4_candidate() -> TerminalHarnessV4:
    return TerminalHarnessV4(
        action_id="adm_progress_guard_20",
        operations=(
            "ADD:explicit_completion_contract",
            "ADD:batched_diagnostic_search",
            "ADD:nonterminal_exact_duplicate_guard",
            "ADD:late_validation_reserve",
            "DELETE:duplicate_rejection_termination",
            "MODIFY:max_turns_16_to_20",
            "MODIFY:validation_reminder_turn_10_to_17",
        ),
        max_turns=20,
        max_output_tokens=3072,
        components=(
            (
                "completion_contract_component",
                "Before acting, identify every required output, its exact path and format, "
                "and the strongest available local check. Keep these as the completion "
                "contract and do not finish while a named output is missing.",
            ),
            (
                "diagnostic_search_component",
                "Batch independent inspections and probes into one command. For unknown "
                "values or failure boundaries, prefer static inspection, structured search, "
                "or interval/batched tests over one-value-at-a-time exploration.",
            ),
            (
                "efficiency_component",
                "Use installed tools and local files before network installation. Treat a "
                "rejected exact duplicate as feedback only: inspect the prior result and "
                "change approach without discarding completed artifacts.",
            ),
            (
                "validation_component",
                "Once the first candidate artifact exists, spend remaining work on the "
                "task-named parser, compiler, executable, or representative input. Before "
                "finishing, verify every required output exists and repair one diagnosed "
                "failure at a time.",
            ),
        ),
        duplicate_command_policy="reject_consecutive_exact",
        reserve_validation_turns=4,
    )


def terminal_v4_ablations() -> tuple[TerminalHarnessV4, ...]:
    full = terminal_v4_candidate()
    definitions = [full]
    for component, suffix in (
        ("completion_contract_component", "no-contract"),
        ("diagnostic_search_component", "no-search"),
        ("efficiency_component", "no-efficiency"),
        ("validation_component", "no-validation"),
    ):
        candidate = replace(
            full,
            action_id=f"{full.action_id}-{suffix}",
            operations=full.operations + (f"ABLATE:{component}",),
            components=tuple(
                entry for entry in full.components if entry[0] != component
            ),
            duplicate_command_policy=(
                "allow"
                if component == "efficiency_component"
                else full.duplicate_command_policy
            ),
            reserve_validation_turns=(
                0
                if component == "validation_component"
                else full.reserve_validation_turns
            ),
            ablated_component=component,
        )
        definitions.append(candidate)
    return tuple(definitions)


__all__ = [
    "TerminalHarnessV4",
    "terminal_v4_ablations",
    "terminal_v4_candidate",
]
