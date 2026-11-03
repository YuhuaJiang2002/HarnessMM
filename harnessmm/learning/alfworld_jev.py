"""Typed Jev decisions interleaved with the unchanged ALFWorld CoT actor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import string
from typing import Any


SCHEMA = "phiagent-alfworld-cot-jev/v1"
JEV_MODEL = "jev-1.13.0"
DEFER = "deliberate"
MAX_ACTIONS = 254
LOCAL_BACKEND = "qwen_single_token_discriminator"
LOCAL_LABELS = string.ascii_uppercase + string.ascii_lowercase + string.digits
LOCAL_DEFER = "?"
RULE_MEMORY_SCHEMA = "phiagent-alfworld-source-rule-memory/v1"
RULE_MEMORY_MODE = "stage_rule_memory_v1"
PROGRESS_RECOVERY_MODE = "typed_progress_recovery_v1"
RECOVERY_ABLATIONS = ("none", "no_stall_gate", "no_progress_filter", "no_periodic_refresh", "no_typed_selector")
RULE_STAGES = {
    "take": r"\b(take|pick|held|holding|carry|carrying)\b",
    "place": r"\b(move|put|place|destination|placement)\b",
    "clean": r"\b(clean|cleaned|sinkbasin)\b",
    "heat": r"\b(heat|heated|microwave)\b",
    "cool": r"\b(cool|cooled|fridge)\b",
    "use": r"\b(use|light|lamp|desklamp|floorlamp)\b",
}


class RuleMemory:
    def __init__(self, source: str):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Rule memory requires the unchanged nonempty source skill")
        self.source = source
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.entries = []
        for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", source):
            text = match.group().strip()
            if not text or text.startswith(("#", "---", "name:")):
                continue
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            stages = [stage for stage, pattern in RULE_STAGES.items() if re.search(pattern, text, re.I)]
            guard = re.search(r"\b(never|do not|don't|avoid|must not)\b", text, re.I) is not None
            self.entries.append({"id": len(self.entries), "start": start, "end": start + len(text),
                                 "text": text, "stages": stages, "guard": guard})

    def retrieve(self, stage: str) -> list[dict]:
        if stage not in RULE_STAGES:
            raise ValueError("Unknown public interaction stage")
        return [dict(entry) for entry in self.entries if stage in entry["stages"] or entry["guard"]]

    def record(self) -> dict:
        return {"schema_version": RULE_MEMORY_SCHEMA, "source_sha256": self.source_sha256,
                "source": self.source, "entries": self.entries, "model_calls": 0,
                "selection": "exact_source_spans_by_public_interaction_stage"}


@dataclass(frozen=True)
class LocalGate:
    minimum_probability: float = 0.90
    minimum_margin: float = 0.25
    maximum_fast_steps: int = 3

    def __post_init__(self) -> None:
        for value in (self.minimum_probability, self.minimum_margin):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Local discriminator thresholds must be finite probabilities")
        if type(self.maximum_fast_steps) is not int or self.maximum_fast_steps < 1:
            raise ValueError("Maximum fast steps must be a positive integer")


@dataclass(frozen=True)
class Gate:
    minimum_probability: float = 0.90
    minimum_confidence: float = 0.80
    minimum_margin: float = 0.25
    minimum_routine_probability: float = 0.90
    maximum_fast_steps: int = 3

    def __post_init__(self) -> None:
        for value in (self.minimum_probability, self.minimum_confidence,
                      self.minimum_margin, self.minimum_routine_probability):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Jev thresholds must be finite probabilities")
        if type(self.maximum_fast_steps) is not int or self.maximum_fast_steps < 1:
            raise ValueError("Maximum consecutive fast steps must be a positive integer")


def probability(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid Jev probability or confidence")
    return float(value)


def provider_usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("Jev response lacks actual provider usage")
    values = [usage.get(key) for key in ("input_tokens", "output_tokens")]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Jev usage must contain exact nonnegative input/output tokens")
    return {"prompt_tokens": values[0], "completion_tokens": values[1],
            "total_tokens": values[0] + values[1]}


def public_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    result = []
    for step in history:
        action = step.get("action")
        observation = step.get("observation")
        if not isinstance(action, str) or not isinstance(observation, str):
            raise ValueError("Only public action and observation text may enter Jev state")
        result.append({"action": action, "observation": observation})
    return result


def validate_actions(actions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(actions, (str, bytes)):
        raise ValueError("Native admissible actions must be a sequence")
    result = tuple(actions)
    if not result or len(result) != len(set(result)):
        raise ValueError("Native action inventory must be nonempty and unique")
    if any(not isinstance(action, str) or not action.strip()
           or any(character in action for character in ("\n", "\r", "<", ">")) for action in result):
        raise ValueError("Admissible actions must be intact single-line native commands")
    return result


def routing_reason(
    *, turn: int, actions: Sequence[str], history: Sequence[Mapping[str, Any]],
    last_cot_note: str, consecutive_fast_steps: int, gate: Gate,
) -> str:
    native_actions = validate_actions(actions)
    public = public_history(history)
    if type(turn) is not int or turn != len(public) or turn < 0:
        raise ValueError("Turn index must match the public conversation")
    if type(consecutive_fast_steps) is not int or not 0 <= consecutive_fast_steps <= turn:
        raise ValueError("Invalid consecutive Jev-step count")
    if not isinstance(last_cot_note, str):
        raise ValueError("CoT context must be text")
    if turn == 0 or not last_cot_note.strip():
        return "cot_initial_or_missing_context"
    if len(native_actions) > MAX_ACTIONS:
        return "cot_action_cardinality_exceeds_jev_limit"
    if consecutive_fast_steps >= gate.maximum_fast_steps:
        return "cot_periodic_refresh"
    latest = public[-1]
    if re.search(r"nothing happens|cannot |can't |not possible|don't understand|invalid action|not a valid",
                 latest["observation"], re.I):
        return "cot_public_tool_error"
    if latest in public[:-1]:
        return "cot_repeated_action_and_feedback"
    return "jev_candidate"


def grounded_interaction(*, task: str, observation: str, actions: Sequence[str],
                         history: Sequence[Mapping[str, Any]]) -> dict:
    from scripts.alfworld_public_progress import ENTITY, UnsupportedPublicState, parse_feedback, parse_goal

    commands = validate_actions(actions)
    public = public_history(history)
    try:
        goal = parse_goal(task)
    except UnsupportedPublicState:
        return {"eligible": False, "reason": "cot_memory_unsupported_goal"}
    held = set()
    properties = {}
    placements = {}
    lights = {}
    for step in public:
        error = re.search(r"nothing happens|cannot |can't |not possible|don't understand|invalid action|not a valid",
                          step["observation"], re.I) is not None
        event = parse_feedback(step["observation"], tool_error=error)
        if event.kind == "take":
            held.add(event.object_id)
            placements.pop(event.object_id, None)
        elif event.kind == "move":
            held.discard(event.object_id)
            placements[event.object_id] = event.destination
        elif event.kind == "inventory":
            held = set(event.objects or ())
        elif event.kind == "transform":
            states = properties.setdefault(event.object_id, set())
            if event.state in ("hot", "cool"):
                states.difference_update(("hot", "cool"))
            states.add(event.state)
        elif event.kind == "light":
            lights[event.object_id] = event.state
        elif event.kind == "unsupported" and step["action"].startswith(("take ", "move ", "put ", "clean ", "heat ", "cool ", "use ")):
            return {"eligible": False, "reason": "cot_memory_unknown_transition"}
    facts = {"held": sorted(held), "properties": {key: sorted(value) for key, value in sorted(properties.items())},
             "placements": dict(sorted(placements.items()))}
    placed = {obj for obj, destination in placements.items()
              if obj.rsplit(" ", 1)[0] == goal.object_type and destination.rsplit(" ", 1)[0] == goal.destination_type}
    targets_held = {obj for obj in held if obj.rsplit(" ", 1)[0] == goal.object_type}
    event = parse_feedback(observation, tool_error=False)
    candidates = []
    for command in commands:
        taken = re.fullmatch(rf"take ({ENTITY}) from ({ENTITY})", command)
        moved = re.fullmatch(rf"move ({ENTITY}) to ({ENTITY})", command)
        transformed = re.fullmatch(rf"(clean|heat|cool) ({ENTITY}) with ({ENTITY})", command)
        used = re.fullmatch(rf"use ({ENTITY})", command)
        if (taken and not held and taken[1].rsplit(" ", 1)[0] == goal.object_type
                and taken[1] not in placed and len(placed) < goal.count
                and event.objects is not None and taken[1] in event.objects and event.location == taken[2]):
            candidates.append((command, "take"))
        elif (moved and moved[1] in targets_held and moved[2].rsplit(" ", 1)[0] == goal.destination_type
                and (goal.condition is None or goal.condition in properties.get(moved[1], set()))):
            candidates.append((command, "place"))
        elif transformed and transformed[2] in targets_held:
            condition = {"clean": "clean", "heat": "hot", "cool": "cool"}[transformed[1]]
            appliance = {"clean": "sinkbasin", "heat": "microwave", "cool": "fridge"}[transformed[1]]
            if (goal.condition == condition and condition not in properties.get(transformed[2], set())
                    and transformed[3].rsplit(" ", 1)[0] == appliance):
                candidates.append((command, transformed[1]))
        elif (used and len(targets_held) == 1 and used[1].rsplit(" ", 1)[0] == goal.light_type
                and lights.get(used[1]) != "on"):
            candidates.append((command, "use"))
    if len(candidates) != 1:
        return {"eligible": False, "reason": "cot_memory_ambiguous_interaction" if candidates else
                "cot_memory_search_or_unmet_prerequisite", "public_facts": facts}
    command, stage = candidates[0]
    return {"eligible": True, "reason": "local_grounded_interaction", "stage": stage,
            "grounded_action": command, "public_facts": facts}


def progress_choices(*, task: str, initial_observation: str, actions: Sequence[str],
                     history: Sequence[Mapping[str, Any]]) -> dict:
    from scripts.alfworld_public_progress import (
        DEFAULT_CONFIG, ENTITY, Inspection, PublicProgressTracker, UnsupportedPublicState,
    )

    commands = validate_actions(actions)
    public = public_history(history)
    try:
        tracker = PublicProgressTracker(task, initial_observation, DEFAULT_CONFIG)
        for step in public:
            error = re.search(r"nothing happens|cannot |can't |not possible|don't understand|invalid action|not a valid",
                              step["observation"], re.I) is not None
            tracker.observe(step["action"], step["observation"], tool_error=error)
    except UnsupportedPublicState:
        return {"eligible": False, "reason": "cot_progress_unsupported_public_state"}
    if tracker.last_update.get("supported") is False:
        return {"eligible": False, "reason": "cot_progress_unsupported_public_state"}
    goal = tracker.goal
    held = sorted(obj for obj in tracker.held if obj.rsplit(" ", 1)[0] == goal.object_type)
    placed = {obj for obj, destination in tracker.placements.items()
              if obj.rsplit(" ", 1)[0] == goal.object_type and destination.rsplit(" ", 1)[0] == goal.destination_type}
    locations = {location for obj, location in tracker.sightings
                 if obj not in placed and obj not in tracker.held and location in tracker.receptacles}
    action_groups = []
    destination_type = None
    if held:
        conditions = [tracker.properties.get(obj, {}) for obj in held]
        transform_needed = (goal.condition == "clean" and not all(value.get("clean") is True for value in conditions)
                            or goal.condition in ("hot", "cool")
                            and not all(value.get("temperature") == goal.condition for value in conditions))
        if transform_needed:
            verb, destination_type = {"clean": ("clean", "sinkbasin"), "hot": ("heat", "microwave"),
                                      "cool": ("cool", "fridge")}[goal.condition]
            action_groups.append(("complete_required_transform", [command for command in commands
                if any(re.fullmatch(rf"{verb} {re.escape(obj)} with {destination_type} \d+", command) for obj in held)]))
        elif goal.light_type:
            action_groups.append(("illuminate_held_target", [command for command in commands
                if re.fullmatch(rf"use {re.escape(goal.light_type)} \d+", command)]))
            locations = {name for name, rec in tracker.receptacles.items()
                         if any(obj.rsplit(" ", 1)[0] == goal.light_type for obj in (rec.last_contents or ()))}
        else:
            destination_type = goal.destination_type
            action_groups.append(("place_held_target", [command for command in commands
                if any(re.fullmatch(rf"move {re.escape(obj)} to {re.escape(destination_type)} \d+", command) for obj in held)]))
        if destination_type:
            action_groups.append(("open_required_destination", [command for command in commands
                if re.fullmatch(rf"open {re.escape(destination_type)} \d+", command)]))
            action_groups.append(("navigate_required_destination", [command for command in commands
                if re.fullmatch(rf"go to {re.escape(destination_type)} \d+", command)]))
            locations = set()
    elif not tracker.held and len(placed) < goal.count:
        take = []
        for command in commands:
            match = re.fullmatch(rf"take ({ENTITY}) from ({ENTITY})", command)
            if match and match[1].rsplit(" ", 1)[0] == goal.object_type and match[1] not in placed:
                take.append(command)
        action_groups.append(("take_public_target", take))
    else:
        return {"eligible": False, "reason": "cot_progress_unresolved_inventory_or_completion"}
    action_groups.append(("open_known_source", [command for command in commands
        if command.startswith("open ") and command[5:] in locations]))
    action_groups.append(("navigate_known_source", [command for command in commands
        if command.startswith("go to ") and command[6:] in locations]))
    action_groups.append(("inspect_unopened_location", [command for command in commands
        if command.startswith("open ") and command[5:] in tracker.receptacles
        and tracker.receptacles[command[5:]].inspection is Inspection.UNKNOWN]))
    action_groups.append(("search_uninspected_location", [command for command in commands
        if command.startswith("go to ") and command[6:] in tracker.receptacles
        and tracker.receptacles[command[6:]].inspection is Inspection.UNKNOWN and command[6:] != tracker.position]))
    if public and tracker.last_update.get("kind") == "error":
        rejected = public[-1]["action"]
        action_groups = [(reason, [command for command in group if command != rejected]) for reason, group in action_groups]
    selected = next(((reason, group) for reason, group in action_groups if group), None)
    if selected is None:
        return {"eligible": False, "reason": "cot_progress_no_supported_choice"}
    reason, allowed = selected
    state = {"goal": asdict(goal), "position": tracker.position, **tracker.task_state(),
             "known_target_locations": sorted(locations),
             "inspected_locations": sorted(name for name, rec in tracker.receptacles.items()
                                            if rec.inspection is not Inspection.UNKNOWN)}
    return {"eligible": True, "reason": reason, "allowed_actions": allowed, "public_state": state,
            "complete_native_actions": list(commands), "no_progress_actions": tracker.no_progress_actions}


def progress_choice_messages(*, task: str, skill: str, observation: str, progress: Mapping) -> tuple[list[dict], dict]:
    if progress.get("eligible") is not True:
        raise ValueError("Typed recovery requires a public supported action set")
    commands = validate_actions(progress["allowed_actions"])
    if len(commands) > len(LOCAL_LABELS) or not set(commands) <= set(progress["complete_native_actions"]):
        raise ValueError("Recovery options exceed label capacity or the native inventory")
    choices = {LOCAL_LABELS[index]: action for index, action in enumerate(commands)}
    choices[LOCAL_DEFER] = None
    state = {"task": task, "skill": skill, "observation": observation, "public_state": progress["public_state"],
             "choice_type": progress["reason"], "actions": {label: action if action is not None else "abstain"
                                                             for label, action in choices.items()}}
    return [
        {"role": "system", "content": (
            "Choose the next action label to advance the ALFWorld task. The options are native legal actions "
            + ("from the complete native action inventory. " if progress["reason"] == "unfiltered_native_actions" else
               "filtered by public inventory, completed transformations and inspected locations. ") +
            "For search, choose an uninspected location; do not demand prior proof that it contains the target. "
            "Use observed facts, not invented object locations. Return one label, or ? if these options cannot help."
        )},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
    ], choices


def decision_request(
    *, task: str, skill: str, observation: str, actions: Sequence[str],
    history: Sequence[Mapping[str, Any]], last_cot_note: str, model: str,
) -> dict[str, Any]:
    if re.fullmatch(r"jev-\d+\.\d+\.\d+", model) is None:
        raise ValueError("Jev must use a pinned model identifier, not latest")
    native_actions = validate_actions(actions)
    if len(native_actions) > MAX_ACTIONS:
        raise ValueError("Do not truncate the legal action list to fit Jev")
    if any(not isinstance(value, str) or not value.strip() for value in (task, observation, last_cot_note)):
        raise ValueError("Task, public observation and last CoT note are required")
    if not isinstance(skill, str):
        raise ValueError("Original CHISEL skill must be text")
    recent = public_history(history)[-2:]
    criteria = {f"action_{index:03d}": action for index, action in enumerate(native_actions)}
    criteria[DEFER] = "Abstain: the next action needs further deliberation or missing information."
    return {
        "model": model,
        "state": {
            "task": task, "chisel_skill": skill, "current_observation": observation,
            "recent_public_steps": recent,
            "last_cot_note": last_cot_note,
            "note_semantics": "Earlier actor deliberation is a hypothesis, not an observed fact; current feedback takes precedence.",
        },
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "Choose one currently admissible action that advances the stated task and chisel_skill. "
                    "Use only public evidence. Abstain for an ambiguous "
                    "search strategy, unsupported prerequisite, repeated ineffective action, or uncertain plan."
                ),
                "criteria": criteria,
            },
            "routine_step": {
                "type": "noul",
                "instructions": (
                    "The current public observation and prior deliberation establish an unambiguous routine "
                    "next step, with no new search-strategy choice, unobserved prerequisite, or unresolved failure."
                ),
            },
        },
    }


def select_decision(response: Mapping[str, Any], request: Mapping[str, Any], *, gate: Gate) -> dict[str, Any]:
    usage = provider_usage(response)
    if response.get("model") != request["model"]:
        raise ValueError("Actual Jev response model differs from the pinned request")
    answers = response.get("answers")
    if not isinstance(answers, Mapping) or set(answers) != {"next_action", "routine_step"}:
        raise ValueError("Jev answers do not match the declared atomic questions")
    choice = answers["next_action"]
    routine = answers["routine_step"]
    if not isinstance(choice, Mapping) or choice.get("type") != "choice":
        raise ValueError("Jev next_action must be a Choice")
    if not isinstance(routine, Mapping) or routine.get("type") != "noul":
        raise ValueError("Jev routine_step must be a Noul")
    criteria = request["questions"]["next_action"]["criteria"]
    values = choice.get("probabilities")
    if not isinstance(values, Mapping) or set(values) != set(criteria):
        raise ValueError("Jev probability support differs from the complete action inventory")
    probabilities = {label: probability(value) for label, value in values.items()}
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Jev probabilities do not sum to one; never silently normalize")
    selected = choice.get("choice")
    if not isinstance(selected, str) or selected not in probabilities:
        raise ValueError("Jev selected an undeclared option")
    best = max(probabilities.values())
    if probabilities[selected] != best:
        raise ValueError("Jev choice is not a maximum-probability option")
    runner_up = max(value for label, value in probabilities.items() if label != selected)
    confidence = probability(choice.get("confidence"))
    routine_probability = probability(routine.get("noul"))
    margin = best - runner_up
    permitted = (selected != DEFER and best > runner_up and best >= gate.minimum_probability
                 and confidence >= gate.minimum_confidence and margin >= gate.minimum_margin
                 and routine_probability >= gate.minimum_routine_probability)
    return {
        "route": "jev" if permitted else "cot",
        "action": criteria[selected] if permitted else None,
        "reason": "typed_high_confidence_routine_action" if permitted else "jev_abstention_or_uncertainty",
        "choice": selected, "probability": best, "confidence": confidence, "margin": margin,
        "routine_probability": routine_probability, "usage": usage,
        "confidence_is_not_a_correctness_guarantee": True,
    }


def local_choice_messages(
    *, task: str, skill: str, observation: str, actions: Sequence[str],
    history: Sequence[Mapping[str, Any]], last_cot_note: str,
    rule_memory: RuleMemory | None = None, routine: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    commands = validate_actions(actions)
    if len(commands) > len(LOCAL_LABELS):
        raise ValueError("Local discriminator cannot truncate a large legal action inventory")
    if not all(isinstance(value, str) and value.strip() for value in (task, observation, last_cot_note)):
        raise ValueError("Local decisions require task, observation and prior deliberation")
    if not isinstance(skill, str):
        raise ValueError("CHISEL skill must be text")
    choices = {LOCAL_LABELS[index]: action for index, action in enumerate(commands)}
    choices[LOCAL_DEFER] = None
    state = {
        "task": task, "skill": skill, "observation": observation,
        "recent_steps": public_history(history)[-2:], "prior_deliberation": last_cot_note,
        "actions": {label: action if action is not None else "defer to full deliberation"
                    for label, action in choices.items()},
    }
    messages = [
        {"role": "system", "content": (
            "Select the next ALFWorld action from the labels. Return exactly one label, no explanation. "
            "Choose an action only when current public observations and the prior plan establish a routine "
            "next step with satisfied prerequisites. Follow the original skill. Prior deliberation is a "
            "hypothesis; observed feedback overrides it. Use ? for a new search strategy, ambiguous object "
            "or destination, uncertain state, or failed plan. Never invent facts."
        )},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
    ]
    if rule_memory is not None:
        if rule_memory.source != skill or not routine or routine.get("eligible") is not True:
            raise ValueError("Rule retrieval requires the bound source and a public grounded stage")
        state.pop("skill")
        state["rules"] = [entry["text"] for entry in rule_memory.retrieve(routine["stage"])]
        state["stage"] = routine["stage"]
        state["public_facts"] = routine["public_facts"]
        messages = [
            {"role": "system", "content": (
                "Select one action label for the task and current interaction stage using the retrieved rules. "
                "Current observations and public facts override the prior hypothesis. Return ? when uncertain; "
                "otherwise return one label."
            )},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
        ]
    return messages, choices


def select_local_decision(
    response: Mapping[str, Any], choices: Mapping[str, str | None], *,
    expected_model: str, gate: LocalGate,
) -> dict[str, Any]:
    if response.get("model") != expected_model:
        raise ValueError("Local discriminator model differs from the unchanged CoT model")
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("Local discriminator requires actual provider usage")
    values = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    if (any(type(value) is not int or value < 0 for value in values)
            or values[0] + values[1] != values[2] or values[1] != 1):
        raise ValueError("Local discriminator must charge exactly one generated token")
    if LOCAL_DEFER not in choices or choices[LOCAL_DEFER] is not None or len(choices) < 2:
        raise ValueError("A full action-label map and explicit abstention are required")
    outputs = response.get("choices")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Exactly one local discriminator completion is required")
    output = outputs[0]
    selected = (output.get("message") or {}).get("content")
    if selected not in choices:
        raise ValueError("Local discriminator emitted an undeclared action label")
    entries = (output.get("logprobs") or {}).get("content")
    if not isinstance(entries, list) or len(entries) != 1 or entries[0].get("token") != selected:
        raise ValueError("Missing selected-token provider log-probability")
    top = entries[0].get("top_logprobs")
    if not isinstance(top, list) or len(top) != len(choices):
        raise ValueError("Require the complete constrained label distribution, not a truncated top-k")
    if any(isinstance(entry, Mapping) and entry.get("token") not in choices for entry in top):
        return {
            "route": "cot", "action": None, "reason": "local_logprob_support_outside_allowed_labels",
            "selected_label": selected, "probability": None, "margin": None,
            "probabilities": None,
            "usage": dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), values)),
            "backend": LOCAL_BACKEND, "true_jev_used": False, "calibration_claimed": False,
            "probability_scope": "invalid_support_not_used_for_action_acceptance",
        }
    distribution = {}
    raw_logs = {}
    for entry in top:
        label = entry.get("token")
        logprob = entry.get("logprob")
        if (label not in choices or label in distribution or type(logprob) not in (float, int)
                or not math.isfinite(logprob) or logprob > 1e-6):
            raise ValueError("Invalid constrained provider log-probability")
        distribution[label] = math.exp(min(logprob, 0.0))
        raw_logs[label] = logprob
    if not math.isclose(sum(distribution.values()), 1.0, rel_tol=0, abs_tol=1e-4):
        return {
            "route": "cot", "action": None, "reason": "local_logprob_distribution_not_normalized",
            "selected_label": selected, "probability": None, "margin": None, "probabilities": None,
            "observed_probability_mass": sum(distribution.values()),
            "usage": dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), values)),
            "backend": LOCAL_BACKEND, "true_jev_used": False, "calibration_claimed": False,
            "probability_scope": "invalid_normalization_not_used_for_action_acceptance",
        }
    selected_log = entries[0].get("logprob")
    if (type(selected_log) not in (float, int) or not math.isfinite(selected_log)
            or not math.isclose(selected_log, raw_logs[selected], rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Selected token score differs from the returned distribution")
    best = distribution[selected]
    runner_up = max(value for label, value in distribution.items() if label != selected)
    if best < runner_up:
        raise ValueError("Greedy choice is not the highest-probability legal label")
    margin = best - runner_up
    accepted = (selected != LOCAL_DEFER and best > runner_up
                and best >= gate.minimum_probability and margin >= gate.minimum_margin)
    return {"route": "local" if accepted else "cot", "action": choices[selected] if accepted else None,
            "reason": "local_high_margin_action" if accepted else "local_abstention_or_uncertainty",
            "selected_label": selected, "probability": best, "margin": margin,
            "probabilities": distribution,
            "usage": dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), values)),
            "backend": LOCAL_BACKEND, "true_jev_used": False, "calibration_claimed": False,
            "probability_scope": "same_Qwen_next_token_distribution_conditioned_on_legal_labels"}


def native_action_response(action: str) -> str:
    validate_actions([action])
    return f"<think>Typed action from public state.</think><action>{action}</action>"
