"""Frozen ALFWorld component-edit sequences for constrained GSPO search."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from phiagent.learning.alfworld_autosaddler_pfn import (
    TASK_FAMILIES,
    pfn_components,
    validate_nested_schema,
)


@dataclass(frozen=True)
class HarnessEditSequence:
    """One complete component mutation sequence."""

    action_id: str
    operations: tuple[str, ...]
    components: Mapping[str, str]

    @property
    def component_bytes(self) -> int:
        return sum(len(value.encode("utf-8")) for value in self.components.values())

    @property
    def component_sha256(self) -> str:
        payload = "\n".join(f"{key}\0{self.components[key]}" for key in sorted(self.components))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


COMPACT_ROUTES = {
    "pick_and_place_simple": (
        "Locate the exact target and destination. Take the target, go to the "
        "destination, move it there, and stop immediately on success."
    ),
    "look_at_obj_in_light": (
        "Locate and take the exact target, go to its named lamp, use the lamp, "
        "and stop immediately when success is reported."
    ),
    "pick_two_obj_and_place": (
        "Track both requested instances. Move each exact object to the requested "
        "destination in turn; verify both placements and stop on success."
    ),
    "pick_clean_then_place_in_recep": (
        "Take the exact target, clean it with a sinkbasin, move it to the requested "
        "destination, and stop on success."
    ),
    "pick_heat_then_place_in_recep": (
        "Take the exact target, heat it with a microwave, move it to the requested "
        "destination, and stop on success."
    ),
    "pick_cool_then_place_in_recep": (
        "Take the exact target, cool it with a fridge, move it to the requested "
        "destination, and stop on success."
    ),
}

MINIMAL_ROUTES = {
    family: text.replace("the exact ", "").replace("requested ", "")
    for family, text in COMPACT_ROUTES.items()
}

COMPACT_SHARED = (
    "Use only admissible actions. Track targets, inventory, and state changes. "
    "Do not repeat an ineffective action; stop immediately on success."
)
MINIMAL_SHARED = "Follow the routed plan with admissible actions and stop on success."


def _compact_components(*, minimal: bool, loop_guard: bool) -> dict[str, str]:
    routes = MINIMAL_ROUTES if minimal else COMPACT_ROUTES
    shared = MINIMAL_SHARED if minimal else COMPACT_SHARED
    if loop_guard:
        shared += " After one failed action, inspect once and choose a different admissible action."
    result = {"shared_skill": shared}
    result.update({f"route::{family}": routes[family] for family in TASK_FAMILIES})
    return result


def build_edit_sequences(
    autosaddler_components: Mapping[str, str],
) -> tuple[HarnessEditSequence, ...]:
    """Build all candidate sequences before any new evaluation outcome."""

    validate_nested_schema(autosaddler_components)
    full = pfn_components()
    compact = _compact_components(minimal=False, loop_guard=False)
    compact_guard = _compact_components(minimal=False, loop_guard=True)
    minimal = _compact_components(minimal=True, loop_guard=False)
    hybrid = dict(autosaddler_components)
    proposals = pfn_components()
    for family in (
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
    ):
        hybrid[f"route::{family}"] = proposals[f"route::{family}"]
    hybrid["shared_skill"] = COMPACT_SHARED
    shared_only = dict(autosaddler_components)
    shared_only["shared_skill"] = COMPACT_SHARED
    definitions = (
        (
            "proposal_full",
            (
                "modify:replace_four_routes_with_train_grounded_proposals",
                "modify:replace_shared_policy_with_proposal_first_policy",
                "add:explicit_terminal_stop",
            ),
            full,
        ),
        (
            "proposal_compact",
            (
                "modify:replace_all_routes_with_action_macros",
                "delete:remove_redundant_route_explanations",
                "add:explicit_terminal_stop",
            ),
            compact,
        ),
        (
            "proposal_compact_guard",
            (
                "modify:replace_all_routes_with_action_macros",
                "delete:remove_redundant_route_explanations",
                "add:one_failure_loop_guard",
                "add:explicit_terminal_stop",
            ),
            compact_guard,
        ),
        (
            "proposal_minimal",
            (
                "modify:replace_all_routes_with_minimal_macros",
                "delete:remove_nonessential_shared_checks",
            ),
            minimal,
        ),
        (
            "hybrid_four_experts",
            (
                "modify:replace_four_stateful_routes_with_proposals",
                "delete:compress_shared_policy",
            ),
            hybrid,
        ),
        (
            "shared_only_compact",
            (
                "delete:compress_shared_policy",
                "add:one_failure_loop_guard",
            ),
            shared_only,
        ),
    )
    sequences = tuple(
        HarnessEditSequence(
            action_id=action_id,
            operations=operations,
            components=components,
        )
        for action_id, operations, components in definitions
    )
    for sequence in sequences:
        validate_nested_schema(sequence.components)
    if len({sequence.action_id for sequence in sequences}) != len(sequences):
        raise RuntimeError("candidate action IDs are duplicated")
    if len({sequence.component_sha256 for sequence in sequences}) != len(sequences):
        raise RuntimeError("candidate component maps are duplicated")
    return sequences


__all__ = [
    "HarnessEditSequence",
    "build_edit_sequences",
]
