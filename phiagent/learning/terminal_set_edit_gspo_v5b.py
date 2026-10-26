"""SQL-ledger-aligned Terminal-Bench V5b ablation arms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from phiagent.learning.terminal_set_edit_gspo_v5 import TerminalHarnessV5


COMPONENTS = (
    "exact_command_guard",
    "repeat_no_progress_termination",
    "validation_reserve",
    "task_family_budget_routing",
    "history_observation_compaction",
    "adm_prompt_clauses",
)


@dataclass(frozen=True)
class TerminalV5bArm:
    action_id: str
    components: tuple[str, ...]
    control: str | None = None

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "action_id": self.action_id,
                    "components": self.components,
                    "control": self.control,
                    "adm_definition": ["ADD", "MODIFY"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def harness(self) -> TerminalHarnessV5:
        values = set(self.components)
        return TerminalHarnessV5(
            action_id=self.action_id,
            exact_action_guard="exact_command_guard" in values,
            no_progress_termination=(
                "repeat_no_progress_termination" in values
            ),
            validation_reserve="validation_reserve" in values,
            task_family_routing="task_family_budget_routing" in values,
            history_compaction=(
                "history_observation_compaction" in values
            ),
            prompt_clauses=(
                ("ADD", "MODIFY") if "adm_prompt_clauses" in values else ()
            ),
            control=self.control,
        )


def terminal_v5b_arms() -> tuple[TerminalV5bArm, ...]:
    full = frozenset(COMPONENTS)
    arms = [TerminalV5bArm("v5b-full", COMPONENTS)]
    for component in COMPONENTS:
        arms.append(
            TerminalV5bArm(
                f"v5b-minus-{component}",
                tuple(value for value in COMPONENTS if value != component),
            )
        )
    arms.append(TerminalV5bArm("v5b-empty", ()))
    for component in COMPONENTS:
        arms.append(TerminalV5bArm(f"v5b-only-{component}", (component,)))
    arms.extend(
        (
            TerminalV5bArm(
                "v5b-fixed-routing-control",
                COMPONENTS,
                "fixed_routing_budget_permutation",
            ),
            TerminalV5bArm(
                "v5b-random-routing-control",
                COMPONENTS,
                "random_routing_budget_permutation_seed_20260907",
            ),
            TerminalV5bArm(
                "v5b-compute-matched-control",
                (),
                "empty_prompt_with_full_budget_envelope",
            ),
            TerminalV5bArm(
                "v5b-token-matched-control",
                ("history_observation_compaction",),
                "empty_prompt_with_full_budget_and_compaction",
            ),
            TerminalV5bArm(
                "v5b-interaction-guard-adm",
                ("exact_command_guard", "adm_prompt_clauses"),
            ),
            TerminalV5bArm(
                "v5b-interaction-validation-compaction",
                (
                    "validation_reserve",
                    "history_observation_compaction",
                ),
            ),
        )
    )
    result = tuple(arms)
    if len(result) != 20 or len({arm.action_id for arm in result}) != 20:
        raise RuntimeError("V5b requires 20 unique arms")
    if set(result[0].components) != full:
        raise RuntimeError("V5b Full components differ")
    return result


def ledger_component_map() -> Mapping[str, str]:
    return {
        component: component for component in COMPONENTS
    }


__all__ = [
    "COMPONENTS",
    "TerminalV5bArm",
    "ledger_component_map",
    "terminal_v5b_arms",
]
