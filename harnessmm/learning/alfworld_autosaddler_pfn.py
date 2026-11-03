"""AutoSaddler-compatible proposal-first nested surface for ALFWorld.

The module is deliberately dependency-free.  Routing is performed once from
the public ALFWorld task family, before the environment episode starts.  The
returned component map uses the same ``shared_skill``/``route::<family>``
schema as the public AutoSaddler V2 ALFWorld nested adapter, so the complete
outer loop can evolve it without a second online routing model.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json


TASK_FAMILIES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)

GENERIC_SHARED_SKILL = (
    "Solve the ALFWorld task using only admissible text actions. Inspect observations, "
    "track inventory and state changes, and stop only after the requested relation is verified."
)

PROPOSAL_FIRST_SHARED_SKILL = (
    "Use the task-family proposal below as the primary plan. Use only admissible "
    "ALFWorld text actions, track inventory and state changes, and stop immediately "
    "when the requested terminal relation is reported."
)

GENERIC_ROUTES = {
    family: (
        f"Solve the {family} task from current observations without game-specific "
        "assumptions. Inspect before acting and verify the requested relation."
    )
    for family in TASK_FAMILIES
}

# These four artifacts were proposed from TRAIN traces and frozen before the
# earlier fresh confirmation campaign.  The hashes below are the original
# artifact hashes, not hashes created for this hybrid.
FROZEN_PROPOSALS = {
    "look_at_obj_in_light": """---
name: look_at_obj_in_light_carry_then_illuminate
---

## Trigger
Apply only when the task type is exactly `look_at_obj_in_light` and the task names a target object and a light source.

## Grounded executable procedure
1. Use `go to <receptacle>`, `look`, or `examine <receptacle>` to locate the exact target and named lamp.
2. Use `take <target instance> from <receptacle>` so the exact target is held.
3. Navigate to the receptacle containing the named light source while retaining the target.
4. Use the admissible `use <desklamp instance>` action. Stop immediately when the environment reports completion.

## Guardrails
- Do not invent `move to`, `turn on`, or `look at` pseudo-actions.
- Do not substitute a different object instance.
- After one ineffective `look`/`examine`, re-check the target and lamp location instead of looping.
""",
    "pick_clean_then_place_in_recep": """---
name: pick_clean_direct_procedure
---

## Trigger
Apply only when the task type is exactly `pick_clean_then_place_in_recep`.

## Procedure
1. Locate and `take <exact target instance> from <source receptacle>`.
2. `go to sinkbasin <instance>` while holding that target.
3. Execute the admissible `clean <target instance> with sinkbasin <instance>` action.
4. `go to <requested destination instance>` and `move <target instance> to <destination instance>`.
5. Stop immediately on environment success.

Never manipulate a distractor object and never replace the requested destination with the source receptacle.
""",
    "pick_heat_then_place_in_recep": """---
name: pick_heat_direct_procedure
---

## Trigger
Apply only when the task type is exactly `pick_heat_then_place_in_recep`.

## Procedure
1. Locate and `take <exact target instance> from <source receptacle>`.
2. `go to microwave <instance>` while holding that target.
3. Execute the admissible `heat <target instance> with microwave <instance>` action.
4. `go to <requested destination instance>` and `move <target instance> to <destination instance>`.
5. Stop immediately on environment success.

Never manipulate a distractor object and never replace the requested destination with the source receptacle.
""",
    "pick_cool_then_place_in_recep": """---
name: pick_cool_direct_procedure
---

## Trigger
Apply only when the task type is exactly `pick_cool_then_place_in_recep`.

## Procedure
1. Locate and `take <exact target instance> from <source receptacle>`.
2. `go to fridge <instance>` while holding that target.
3. Execute the admissible `cool <target instance> with fridge <instance>` action.
4. `go to <requested destination instance>` and `move <target instance> to <destination instance>`.
5. Stop immediately on environment success.

Never manipulate a distractor object and never replace the requested destination with the source receptacle.
""",
}

FROZEN_PROPOSAL_SHA256 = {
    "look_at_obj_in_light": "33ca2982df272e5c569602cd9728b2a8e614741cca5e82fb35ea1e8aaf423c35",
    "pick_clean_then_place_in_recep": "9a6965bebcfe14deaa0c48805b8e82c4dff4882cdedad21d45ccb955d2d8ba52",
    "pick_heat_then_place_in_recep": "1344b1ecc7e7519cf67a0db023627c8b06b6628796d587e3fd9edb7a7c276627",
    "pick_cool_then_place_in_recep": "b64968fe3816ad112a58613f30461ea1a24e8f37217f4a4df857a1b4c32e38ac",
}


def canonical_sha256(value: object) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def proposal_hashes_match() -> bool:
    """Check that embedded proposal bytes still match their frozen artifacts."""

    return all(
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        == FROZEN_PROPOSAL_SHA256[family]
        for family, text in FROZEN_PROPOSALS.items()
    )


def nested_components() -> dict[str, str]:
    """Return the generic task-level nested-router baseline."""

    result = {"shared_skill": GENERIC_SHARED_SKILL}
    result.update(
        {f"route::{family}": GENERIC_ROUTES[family] for family in TASK_FAMILIES}
    )
    return result


def pfn_components() -> dict[str, str]:
    """Return the proposal-first nested baseline for AutoSaddler evolution."""

    if not proposal_hashes_match():
        raise RuntimeError("embedded proposal artifact hash drift")
    result = nested_components()
    result["shared_skill"] = PROPOSAL_FIRST_SHARED_SKILL
    for family, proposal in FROZEN_PROPOSALS.items():
        result[f"route::{family}"] = proposal
    return result


def validate_nested_schema(components: Mapping[str, str]) -> None:
    """Validate the component schema shared with the AutoSaddler V2 adapter."""

    expected = {"shared_skill", *(f"route::{family}" for family in TASK_FAMILIES)}
    if set(components) != expected:
        raise ValueError("nested component schema differs")
    if any(not isinstance(value, str) or not value.strip() for value in components.values()):
        raise TypeError("nested components must be non-empty strings")


def render_task_skill(
    components: Mapping[str, str],
    task_family: str,
) -> tuple[str, dict[str, object]]:
    """Render one outcome-blind task-level route and its audit record."""

    validate_nested_schema(components)
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"unknown ALFWorld task family: {task_family}")
    route_name = f"route::{task_family}"
    proposal_covered = task_family in FROZEN_PROPOSALS
    rendered = (
        components[route_name].rstrip()
        + "\n\n"
        + components["shared_skill"].lstrip()
    )
    record = {
        "decision_timing": "before_episode_from_public_task_family",
        "task_family": task_family,
        "selected_route": route_name,
        "proposal_first": proposal_covered,
        "fallback": not proposal_covered,
        "online_router_model_calls": 0,
        "rendered_skill_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    return rendered, record


def component_patch_taxonomy(component_name: str) -> str:
    """Map an AutoSaddler component update to a stable plot/report taxonomy."""

    if component_name == "shared_skill":
        return "shared_policy"
    if component_name.startswith("route::"):
        family = component_name.removeprefix("route::")
        return (
            "proposal_expert"
            if family in FROZEN_PROPOSALS
            else "fallback_route"
        )
    return "unknown"


__all__ = [
    "FROZEN_PROPOSALS",
    "FROZEN_PROPOSAL_SHA256",
    "GENERIC_ROUTES",
    "GENERIC_SHARED_SKILL",
    "PROPOSAL_FIRST_SHARED_SKILL",
    "TASK_FAMILIES",
    "canonical_sha256",
    "component_patch_taxonomy",
    "nested_components",
    "pfn_components",
    "proposal_hashes_match",
    "render_task_skill",
    "validate_nested_schema",
]
