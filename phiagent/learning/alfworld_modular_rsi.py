"""Budget-neutral, single-component evolution of frozen ALFWorld routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from phiagent.learning.alfworld_autosaddler_pfn import (
    TASK_FAMILIES,
    canonical_sha256,
    render_task_skill,
    validate_nested_schema,
)


IDENTITY = "CHISEL + ModularRSI-inspired single-module evolution"
MODULES = ("agent_loop", "tools", "observation", "context_mgmt", "verification")
EDITABLE_SPANS = {
    "agent_loop": (
        "Act on a currently visible exact target before searching. Search visible open surfaces "
        "before closed storage, and never revisit an inspected location unless new feedback names the target there."
    ),
    "tools": (
        "If an action is rejected or has no effect, mark that exact action blocked and choose "
        "a different admissible action instead of repeating it."
    ),
    "context_mgmt": (
        "Keep a compact state set containing the exact target(s), destination, inventory, "
        "known source locations, inspected locations, and blocked actions."
    ),
    "verification": (
        "Treat explicit environment feedback that an object was cleaned, heated, or cooled as "
        "a committed state transition; do not examine or repeat that transform. Stop immediately "
        "when the environment reports success."
    ),
}


def episode_settings(
    budget: Mapping[str, Any], settings: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    if set(settings) != {"max_steps", "episode_seed"}:
        raise ValueError("Only the frozen step cap and seed may override episode execution")
    if type(settings["max_steps"]) is not int or not 0 < settings["max_steps"] <= budget["max_steps"]:
        raise ValueError("Module evolution cannot increase the episode step cap")
    if type(settings["episode_seed"]) is not int or not 0 <= settings["episode_seed"] < 2**32:
        raise ValueError("Episode seed must be a nonnegative uint32")
    result = dict(budget)
    result["max_steps"] = settings["max_steps"]
    return result, settings["episode_seed"]


def replace_component(
    parent: Mapping[str, str],
    *,
    component: str,
    replacement: str,
    module: str,
    count_tokens: Callable[[str], int],
) -> tuple[dict[str, str], dict[str, Any]]:
    validate_nested_schema(parent)
    if module not in MODULES or component not in parent:
        raise ValueError("Unknown module or component")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("Replacement must be nonempty text")
    if parent[component] == replacement:
        raise ValueError("Candidate must change the selected component")
    candidate = dict(parent)
    candidate[component] = replacement
    counts = {}
    for family in TASK_FAMILIES:
        parent_tokens = count_tokens(render_task_skill(parent, family)[0])
        candidate_tokens = count_tokens(render_task_skill(candidate, family)[0])
        if any(type(value) is not int or value < 0 for value in (parent_tokens, candidate_tokens)):
            raise ValueError("Native token counts must be nonnegative integers")
        if candidate_tokens > parent_tokens:
            raise ValueError(f"Candidate increases rendered skill tokens for {family}")
        counts[family] = {"parent": parent_tokens, "candidate": candidate_tokens}
    return candidate, {
        "identity": IDENTITY,
        "locked_module": module,
        "changed_component": component,
        "parent_sha256": canonical_sha256(parent),
        "candidate_sha256": canonical_sha256(candidate),
        "rendered_skill_tokens": counts,
        "extra_online_model_calls": 0,
        "official_modular_rsi_reproduction": False,
    }


def module_candidate(
    parents: Mapping[str, Mapping[str, str]],
    proposal: Mapping[str, Any],
    *,
    support_groups: Mapping[str, str],
    count_tokens: Callable[[str], int],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if set(parents) != set(TASK_FAMILIES):
        raise ValueError("Frozen family routing must be complete")
    module = proposal.get("module")
    if module not in EDITABLE_SPANS:
        raise ValueError("Only a declared, non-overlapping instruction module may change")
    support = proposal.get("support_task_ids")
    if not isinstance(support, list) or not all(isinstance(task, str) for task in support):
        raise ValueError("Proposal must name its training support tasks")
    if not set(support) <= set(support_groups) or len({support_groups[task] for task in support}) < 2:
        raise ValueError("At least two distinct training groups must support the module")
    replacement = proposal.get("replacement")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("Replacement must be nonempty text")
    if any(task in replacement for task in support_groups):
        raise ValueError("Task IDs cannot appear in a reusable module")
    if any(marker in replacement for marker in ("json_2.1.1/", "trial_", "trial-", "FloorPlan")):
        raise ValueError("Instance-specific benchmark paths cannot appear in a module")
    old = EDITABLE_SPANS[module]
    candidate = deepcopy(dict(parents))
    audits = {}
    for family, components in parents.items():
        validate_nested_schema(components)
        shared = components["shared_skill"]
        if old not in shared:
            continue
        if shared.count(old) != 1:
            raise ValueError("Ambiguous parent module boundary")
        candidate[family], audits[family] = replace_component(
            components, component="shared_skill", replacement=shared.replace(old, replacement, 1),
            module=module, count_tokens=count_tokens,
        )
    if not audits:
        raise ValueError("No frozen parent contains the declared module")
    return candidate, {
        "module": module, "support_task_ids": sorted(set(support)),
        "support_groups": sorted({support_groups[task] for task in support}),
        "parent_sha256": canonical_sha256(parents),
        "candidate_sha256": canonical_sha256(candidate),
        "family_audits": audits, "unchanged_families": sorted(set(parents) - set(audits)),
    }


def public_memory_lines(initial: str, conversation: Sequence[Mapping[str, Any]]) -> list[str]:
    from scripts.alfworld_public_progress import DEFAULT_CONFIG, PublicProgressTracker

    task = initial.split("Your task is to: ", 1)[-1].strip()
    tracker = PublicProgressTracker(task, initial, DEFAULT_CONFIG)
    last_action = None
    for step in conversation:
        action = step.get("action")
        feedback = step.get("observation", step.get("env_feedback"))
        if not isinstance(action, str) or not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("Context memory requires executed actions and public feedback")
        tool_error = bool(re.search(
            r"nothing happens|cannot |can't |not possible|don't understand|invalid action|not a valid", feedback, re.I,
        ))
        tracker.observe(action, feedback, tool_error=tool_error)
        last_action = action
    lines = []
    if last_action:
        lines.append("Last action: " + last_action)
    if tracker.position:
        lines.append("Last confirmed location: " + tracker.position)
    if tracker.inventory_status != "unknown":
        lines.append("Observed inventory updates: " + (", ".join(sorted(tracker.held)) or "none held")
                     + "; " + tracker.inventory_status)
    for name, state in sorted(tracker.properties.items()):
        lines.append("Confirmed transform: " + name + " " + json_text(state))
    for name, destination in sorted(tracker.placements.items()):
        lines.append("Confirmed placement: " + name + " -> " + destination)
    sightings = [f"{name} at {location}" for name, location in sorted(tracker.sightings)
                 if name not in tracker.held and name not in tracker.placements]
    if sightings:
        lines.append("Target last seen: " + "; ".join(sightings))
    lamp_sightings = [f"{name} at {location}" for location, rec in sorted(tracker.receptacles.items())
                      for name in (rec.last_contents or ()) if name.rsplit(" ", 1)[0] == tracker.goal.light_type]
    if lamp_sightings:
        lines.append("Light last seen: " + "; ".join(lamp_sightings))
    inspected = [name for name, rec in sorted(tracker.receptacles.items()) if rec.last_contents is not None]
    if inspected:
        lines.append("Contents previously inspected: " + ", ".join(inspected))
    closed = [name for name, rec in sorted(tracker.receptacles.items())
              if rec.visited and rec.access == "closed" and rec.last_contents is None]
    if closed:
        lines.append("Visited but contents unknown (closed): " + ", ".join(closed))
    for feedback in tracker.unsupported.values():
        lines.append("Other earlier feedback: " + feedback)
    return lines


def json_text(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def replace_history(
    prompt: str, initial: str, conversation: Sequence[Mapping[str, Any]], *,
    variant: str, count_tokens: Callable[[str], int],
) -> tuple[str, dict[str, Any]]:
    if variant not in ("context_recent", "context_all"):
        raise ValueError("Unknown context-management candidate")
    boundary = re.search(
        r"\nPrior to this step, you have already taken \d+ step\(s\)\. "
        r"Below are the most recent \d+ observations and the corresponding actions you took:\s+.*?"
        r"(?=\nYou are now at step \d+ and your current observation is:)", prompt, re.S,
    )
    before_tokens = count_tokens(prompt)
    audit = {"variant": variant, "original_tokens": before_tokens, "candidate_tokens": before_tokens,
             "changed": False, "additional_model_calls": 0, "kept_lines": 0, "omitted_lines": 0}
    if boundary is None or not conversation:
        return prompt, {**audit, "reason": "no_native_history_block"}
    from scripts.alfworld_public_progress import UnsupportedPublicState

    try:
        selected_history = conversation if variant == "context_all" else conversation[-2:]
        lines = public_memory_lines(initial, selected_history)
    except UnsupportedPublicState:
        return prompt, {**audit, "reason": "unsupported_public_goal_or_capacity"}
    header = "\nPublic memory (partial; sightings may be stale; missing is unknown):\n"
    for kept in range(len(lines), 0, -1):
        replacement = header + "\n".join(lines[:kept])
        candidate = prompt[:boundary.start()] + replacement + prompt[boundary.end():]
        candidate_tokens = count_tokens(candidate)
        if candidate_tokens <= before_tokens:
            return candidate, {**audit, "changed": True, "candidate_tokens": candidate_tokens,
                               "kept_lines": kept, "omitted_lines": len(lines) - kept,
                               "history_start": boundary.start(), "history_end": boundary.end(),
                               "source_steps": len(selected_history), "reason": "within_native_prompt_budget"}
    return prompt, {**audit, "reason": "memory_cannot_fit_without_growth"}


def compact_boilerplate(
    prompt: str, *, system: str, variant: str, count_tokens: Callable[[str], int],
) -> tuple[str, dict[str, Any]]:
    if variant not in ("boilerplate_wrapper", "boilerplate_role", "boilerplate_both"):
        raise ValueError("Unknown static boilerplate candidate")
    header = "\n\n## Skill Knowledge\n"
    wrapper = ("Below is a skill document with learned strategies. "
               "Use these guidelines to inform your decisions:\n\n")
    role = "You are an expert agent operating in the ALFRED Embodied Environment."
    before_tokens = count_tokens(prompt)
    audit = {"variant": variant, "original_tokens": before_tokens, "candidate_tokens": before_tokens,
             "changed": False, "additional_model_calls": 0, "removed_spans": []}
    spans = []
    if variant in ("boilerplate_wrapper", "boilerplate_both") and prompt.startswith(header + wrapper):
        spans.append({"kind": "skill_wrapper", "start": len(header), "end": len(header + wrapper), "text": wrapper})
    if variant in ("boilerplate_role", "boilerplate_both") and system == role:
        matches = list(re.finditer(re.escape("\n" + role) + r"(?= Your task is to: |\nYour current observation is: )", prompt))
        if len(matches) == 1 and prompt.count(role) == 1:
            match = matches[0]
            spans.append({"kind": "duplicate_system_role", "start": match.start() + 1, "end": match.end(), "text": role})
    if not spans:
        return prompt, {**audit, "reason": "no_known_boilerplate"}
    candidate = prompt
    for span in sorted(spans, key=lambda value: value["start"], reverse=True):
        candidate = candidate[:span["start"]] + candidate[span["end"]:]
    after_tokens = count_tokens(candidate)
    if any(type(value) is not int or value < 0 for value in (before_tokens, after_tokens)):
        raise ValueError("Native prompt token counts must be nonnegative integers")
    if after_tokens >= before_tokens:
        return prompt, {**audit, "reason": "no_native_input_saving"}
    return candidate, {**audit, "candidate_tokens": after_tokens, "changed": True,
                       "removed_spans": spans, "reason": "exact_static_boilerplate_only"}


def compact_action_contract(prompt: str, *, count_tokens: Callable[[str], int],
                            reasoning_word_limit: int | None = None) -> tuple[str, dict[str, Any]]:
    if reasoning_word_limit is not None and (
        type(reasoning_word_limit) is not int or not 32 <= reasoning_word_limit <= 256
    ):
        raise ValueError("A prospective concise-reasoning contract needs 32--256 words")
    marker = "Now it's your turn to take an action."
    expected = (
        "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed "
        "within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible "
        "action for current step and present it within <action> </action> tags."
    )
    before_tokens = count_tokens(prompt)
    audit = {"variant": "brief_action", "original_tokens": before_tokens,
             "candidate_tokens": before_tokens, "changed": False, "additional_model_calls": 0}
    prefix, found, suffix = prompt.rpartition(marker)
    if not found or " ".join(suffix.split()) != expected:
        return prompt, {**audit, "reason": "unknown_output_contract"}
    reasoning = ("a state note of at most 12 words" if reasoning_word_limit is None
                 else f"concise reasoning of at most {reasoning_word_limit} words")
    candidate = (prefix + marker + f"\nReturn exactly <think>{reasoning}</think>"
                 "<action>one admissible action</action>. No other text.\n")
    candidate_tokens = count_tokens(candidate)
    if candidate_tokens > before_tokens:
        return prompt, {**audit, "reason": "output_contract_would_grow_input"}
    return candidate, {**audit, "candidate_tokens": candidate_tokens,
                       "changed": True, "reason": ("bounded_state_note_and_native_action"
                       if reasoning_word_limit is None else "concise_reasoning_and_native_action"),
                       **({"reasoning_word_limit": reasoning_word_limit} if reasoning_word_limit is not None else {})}


def paired_gate(
    parent_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids: Sequence[str],
    minimum_score: float | None = None,
) -> dict[str, Any]:
    expected = set(expected_ids)
    if not expected or len(expected) != len(expected_ids):
        raise ValueError("Expected task IDs must be nonempty and unique")
    summaries = {}
    records = {}
    for arm, rows in (("parent", parent_rows), ("candidate", candidate_rows)):
        indexed = {row["task_id"]: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != expected:
            raise ValueError("Gate requires exactly the complete paired task inventory")
        for row in rows:
            usage = row.get("usage", {})
            if row.get("valid") is not True or row.get("cost_complete") is not True:
                raise ValueError("Invalid episodes or incomplete usage cannot qualify")
            if type(row.get("official_success")) is not int or row["official_success"] not in (0, 1):
                raise ValueError("Gate requires official binary episode success")
            values = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Gate requires exact nonnegative provider usage")
            if values[0] + values[1] != values[2]:
                raise ValueError("Provider usage arithmetic differs")
        successes = sum(row["official_success"] for row in rows)
        summaries[arm] = {
            "tasks": len(rows), "successes": successes,
            "score": 100.0 * successes / len(rows),
            "total_tokens": sum(row["usage"]["total_tokens"] for row in rows),
        }
        records[arm] = indexed
    parent = summaries["parent"]
    candidate = summaries["candidate"]
    quality_pass = candidate["successes"] > parent["successes"]
    budget_pass = candidate["total_tokens"] <= parent["total_tokens"]
    target_pass = minimum_score is None or candidate["score"] > minimum_score
    return {
        "passed": quality_pass and budget_pass and target_pass,
        "strict_quality_gain": quality_pass,
        "nonincreasing_actual_tokens": budget_pass,
        "target_score_pass": target_pass,
        "minimum_score_exclusive": minimum_score,
        "arms": deepcopy(summaries),
        "beneficial_flips": sum(
            records["candidate"][task]["official_success"] > records["parent"][task]["official_success"]
            for task in expected
        ),
        "harmful_flips": sum(
            records["candidate"][task]["official_success"] < records["parent"][task]["official_success"]
            for task in expected
        ),
    }
