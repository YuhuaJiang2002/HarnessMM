"""Prospective Terminal-Bench V5 component family and controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
from typing import Mapping, Sequence

from phiagent.learning.terminal_set_edit_gspo_v2 import VALIDATION_REMINDER


NO_PROGRESS_REMINDER = (
    "No-progress guard: recent commands have the same structure. Stop varying only "
    "literals or rewriting the same artifact. Inspect the latest result, choose a "
    "different diagnostic, or validate the current candidate."
)

PROMPT_CLAUSES = {
    "ADD": (
        "Before acting, list every required output path and format and the strongest "
        "available local check. Do not finish while a required output is missing."
    ),
    "DELETE": (
        "Do not repeat completed inspection, retry unchanged installation, perform "
        "one-value-at-a-time probing, or rewrite an unchanged full artifact."
    ),
    "MODIFY": (
        "After the first candidate artifact exists, pivot from exploration to the "
        "task-named parser, compiler, executable, or representative-input check; "
        "repair one diagnosed failure at a time."
    ),
}


@dataclass(frozen=True)
class TerminalHarnessV5:
    action_id: str
    exact_action_guard: bool
    no_progress_termination: bool
    validation_reserve: bool
    task_family_routing: bool
    history_compaction: bool
    prompt_clauses: tuple[str, ...]
    control: str | None = None
    ablated_component: str | None = None
    isolated_component: str | None = None

    @property
    def tool_policy(self) -> dict[str, object]:
        policy: dict[str, object] = {
            "duplicate_command_policy": (
                "reject_consecutive_exact" if self.exact_action_guard else "allow"
            )
        }
        if self.no_progress_termination:
            policy.update(
                {
                    "no_progress_policy": "remind_then_terminate",
                    "no_progress_patience": 3,
                    "no_progress_reminder": NO_PROGRESS_REMINDER,
                }
            )
        if self.validation_reserve:
            policy.update(
                {
                    "reserve_validation_turns": 4,
                    "validation_reminder": VALIDATION_REMINDER,
                }
            )
        return policy

    @property
    def history_policy(self) -> dict[str, int]:
        if not self.history_compaction:
            return {}
        return {
            "tool_message_characters": 1200,
            "preserve_recent_tool_messages": 2,
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "action_id": self.action_id,
                    "exact_action_guard": self.exact_action_guard,
                    "no_progress_termination": self.no_progress_termination,
                    "validation_reserve": self.validation_reserve,
                    "task_family_routing": self.task_family_routing,
                    "history_compaction": self.history_compaction,
                    "prompt_clauses": self.prompt_clauses,
                    "tool_policy": self.tool_policy,
                    "history_policy": self.history_policy,
                    "control": self.control,
                    "ablated_component": self.ablated_component,
                    "isolated_component": self.isolated_component,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def render(self, base_prompt: str) -> str:
        if not self.prompt_clauses:
            return base_prompt.rstrip() + "\n"
        clauses = "\n".join(
            f"- {operation}: {PROMPT_CLAUSES[operation]}"
            for operation in self.prompt_clauses
        )
        return (
            base_prompt.rstrip()
            + "\n\n## Ours-v5: constrained component policy\n"
            + clauses
            + "\n\nThe task instruction remains authoritative.\n"
        )


def terminal_v5_full() -> TerminalHarnessV5:
    return TerminalHarnessV5(
        action_id="v5-full",
        exact_action_guard=True,
        no_progress_termination=True,
        validation_reserve=True,
        task_family_routing=True,
        history_compaction=True,
        prompt_clauses=("ADD", "DELETE", "MODIFY"),
    )


def terminal_v5_arms() -> tuple[TerminalHarnessV5, ...]:
    full = terminal_v5_full()
    arms = [full]
    core_fields = (
        ("exact_action_guard", "action-guard"),
        ("no_progress_termination", "no-progress"),
        ("validation_reserve", "validation"),
        ("task_family_routing", "routing"),
        ("history_compaction", "compaction"),
    )
    for field, label in core_fields:
        arms.append(
            replace(
                full,
                action_id=f"v5-minus-{label}",
                **{field: False},
                ablated_component=field,
            )
        )
    for operation in ("ADD", "DELETE", "MODIFY"):
        arms.append(
            replace(
                full,
                action_id=f"v5-minus-prompt-{operation.lower()}",
                prompt_clauses=tuple(
                    clause for clause in full.prompt_clauses if clause != operation
                ),
                ablated_component=f"prompt_{operation.lower()}",
            )
        )

    empty = {
        "exact_action_guard": False,
        "no_progress_termination": False,
        "validation_reserve": False,
        "task_family_routing": False,
        "history_compaction": False,
        "prompt_clauses": (),
    }
    for field, label in core_fields:
        arms.append(
            TerminalHarnessV5(
                action_id=f"v5-only-{label}",
                **{**empty, field: True},
                isolated_component=field,
            )
        )
    for operation in ("ADD", "DELETE", "MODIFY"):
        arms.append(
            TerminalHarnessV5(
                action_id=f"v5-only-prompt-{operation.lower()}",
                **{**empty, "prompt_clauses": (operation,)},
                isolated_component=f"prompt_{operation.lower()}",
            )
        )
    arms.extend(
        (
            replace(
                full,
                action_id="v5-fixed-budget-control",
                task_family_routing=False,
                control="matched_fixed_budget",
            ),
            replace(
                full,
                action_id="v5-random-budget-control",
                task_family_routing=False,
                control="matched_random_budget",
            ),
            TerminalHarnessV5(
                action_id="v5-compute-matched-control",
                **empty,
                control="base_prompt_exact_full_task_compute_ceiling",
            ),
            TerminalHarnessV5(
                action_id="v5-token-ceiling-matched-control",
                **{**empty, "history_compaction": True},
                control="base_prompt_full_compute_and_observation_ceiling",
            ),
        )
    )
    if len(arms) != 21 or len({arm.action_id for arm in arms}) != len(arms):
        raise RuntimeError("Terminal V5 must define 21 unique arms")
    return tuple(arms)


def routed_runtime(category: str, difficulty: str) -> dict[str, int]:
    if category in {"data-processing", "file-operations", "optimization"}:
        return {
            "max_turns": 18,
            "max_output_tokens": 4096,
            "reserve_validation_turns": 3,
        }
    if difficulty == "hard" or category in {"security", "mathematics"}:
        return {
            "max_turns": 24,
            "max_output_tokens": 3072,
            "reserve_validation_turns": 4,
        }
    return {
        "max_turns": 20,
        "max_output_tokens": 4096,
        "reserve_validation_turns": 4,
    }


def matched_runtime_assignments(
    task_rows: Sequence[Mapping[str, str]],
    *,
    mode: str,
    seed: int,
) -> dict[str, dict[str, int]]:
    routed = {
        str(row["name"]): routed_runtime(
            str(row["category"]), str(row["difficulty"])
        )
        for row in task_rows
    }
    if mode in {"routed", "compute_matched"}:
        return routed
    names = sorted(routed)
    budgets = [routed[name] for name in names]
    if mode == "fixed":
        budgets = sorted(
            budgets,
            key=lambda value: (
                value["max_turns"],
                value["max_output_tokens"],
                value["reserve_validation_turns"],
            ),
        )
    elif mode == "random":
        random.Random(seed).shuffle(budgets)
    else:
        raise ValueError(f"unknown runtime assignment mode: {mode}")
    return {
        name: dict(budget) for name, budget in zip(names, budgets, strict=True)
    }


__all__ = [
    "NO_PROGRESS_REMINDER",
    "PROMPT_CLAUSES",
    "TerminalHarnessV5",
    "matched_runtime_assignments",
    "routed_runtime",
    "terminal_v5_arms",
    "terminal_v5_full",
]
