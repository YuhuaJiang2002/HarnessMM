"""Auditable add-delete-modify edits for the ALFWorld harness clause set."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from phiagent.learning.alfworld_autosaddler_pfn import (
    TASK_FAMILIES,
    validate_nested_schema,
)


ADDED_CLAUSES = (
    "blocked_action_memory",
    "explicit_transform_commit",
    "visited_location_set",
    "visible_target_priority",
)
DELETED_CLAUSES = (
    "immediate_failed_action_retry",
    "post_transform_reinspection",
    "revisit_inspected_locations",
    "unbounded_container_first_search",
)
MODIFIED_CLAUSES = (
    "search_order_visible_surfaces_before_closed_storage",
    "state_transition_from_explicit_environment_feedback",
    "two_object_recorded_source_return",
)


@dataclass(frozen=True)
class HarnessSetEdit:
    """One complete clause-set mutation with all three operation types."""

    action_id: str
    added: tuple[str, ...]
    deleted: tuple[str, ...]
    modified: tuple[str, ...]
    components: Mapping[str, str]

    def __post_init__(self) -> None:
        validate_nested_schema(self.components)
        if not self.action_id:
            raise ValueError("set-edit action ID must be non-empty")
        for name, values in (
            ("added", self.added),
            ("deleted", self.deleted),
            ("modified", self.modified),
        ):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} clauses must be non-empty, unique, and sorted")
        if set(self.added) & set(self.deleted):
            raise ValueError("a clause cannot be both added and deleted")

    @property
    def operations(self) -> tuple[str, ...]:
        return (
            *(f"add:set_clause:{value}" for value in self.added),
            *(f"delete:set_clause:{value}" for value in self.deleted),
            *(f"modify:set_clause:{value}" for value in self.modified),
        )

    @property
    def operation_types(self) -> tuple[str, ...]:
        return ("add", "delete", "modify")

    @property
    def component_bytes(self) -> int:
        return sum(len(value.encode("utf-8")) for value in self.components.values())

    @property
    def component_sha256(self) -> str:
        payload = "\n".join(
            f"{key}\0{self.components[key]}" for key in sorted(self.components)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SHARED_POLICY = (
    "Use only admissible actions. Keep a compact state set containing the exact "
    "target(s), destination, inventory, known source locations, inspected "
    "locations, and blocked actions. Act on a currently visible exact target "
    "before searching. Search visible open surfaces before closed storage, and "
    "never revisit an inspected location unless new feedback names the target "
    "there. If an action is rejected or has no effect, mark that exact action "
    "blocked and choose a different admissible action instead of repeating it. "
    "Treat explicit environment feedback that an object was cleaned, heated, or "
    "cooled as a committed state transition; do not examine or repeat that "
    "transform. Stop immediately when the environment reports success."
)

ROUTES = {
    "pick_and_place_simple": (
        "If the exact target is visible, take it immediately. Otherwise inspect "
        "unvisited visible countertops, tables, shelves, and other open surfaces "
        "before opening unvisited storage. Record every checked location. Once "
        "holding the target, go directly to the destination, open it if needed, "
        "place the target, and stop on success."
    ),
    "look_at_obj_in_light": (
        "Record the locations of the exact target and named lamp when either is "
        "seen. Search each visible surface or storage location at most once. Take "
        "the target, go directly to the recorded lamp, use the lamp on the target, "
        "and stop on success. A rejected action is blocked rather than repeated."
    ),
    "pick_two_obj_and_place": (
        "When either requested instance is seen, record its source and the common "
        "destination. Move the first instance to the destination, then return "
        "directly to the recorded source of the second instance before searching "
        "elsewhere. Place the second instance in the same destination and stop on "
        "success; never re-search an already inspected location."
    ),
    "pick_clean_then_place_in_recep": (
        "Find the exact target using the visited-location set, take it, and go to "
        "the sinkbasin. Execute the admissible clean action once. Feedback of the "
        "form 'You clean ... using ...' commits the clean state: do not examine or "
        "clean again. Go directly to the destination, open it if needed, place the "
        "object, and stop on success."
    ),
    "pick_heat_then_place_in_recep": (
        "Find the exact target using the visited-location set, take it, and go to "
        "the microwave. Execute the admissible heat action once. Feedback of the "
        "form 'You heat ... using ...' commits the hot state: do not examine or "
        "heat again. Go directly to the destination, open it if needed, place the "
        "object, and stop on success."
    ),
    "pick_cool_then_place_in_recep": (
        "Find the exact target using the visited-location set, prioritizing visible "
        "stoveburners and countertops before closed storage. Take it and go to the "
        "fridge. Execute the admissible cool action once. Feedback of the form "
        "'You cool ... using ...' commits the cool state: do not examine or cool "
        "again. Go directly to the destination, place the object, and stop on "
        "success."
    ),
}


def build_add_delete_modify_edit(
    autosaddler_components: Mapping[str, str],
) -> HarnessSetEdit:
    """Build the frozen clause-set edit before any new evaluation outcome."""

    validate_nested_schema(autosaddler_components)
    components = {"shared_skill": SHARED_POLICY}
    components.update(
        {f"route::{family}": ROUTES[family] for family in TASK_FAMILIES}
    )
    edit = HarnessSetEdit(
        action_id="adm_search_state_memory_v3_budget_tuned",
        added=tuple(sorted(ADDED_CLAUSES)),
        deleted=tuple(sorted(DELETED_CLAUSES)),
        modified=tuple(sorted(MODIFIED_CLAUSES)),
        components=components,
    )
    if edit.component_sha256 == _component_sha256(autosaddler_components):
        raise RuntimeError("set edit did not change the component map")
    return edit


def _component_sha256(components: Mapping[str, str]) -> str:
    payload = "\n".join(f"{key}\0{components[key]}" for key in sorted(components))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "HarnessSetEdit",
    "build_add_delete_modify_edit",
]
