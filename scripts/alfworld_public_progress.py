"""ALF-only typed public-evidence tracking and opt-in mechanism sessions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping

from phiagent.learning.counterfactual_repair import RepairSession, parse_patch
from phiagent.learning.public_recurrence import has_no_progress_recurrence


TRACKER_VERSION = "alfworld-public-progress/v1"
MODES = ("skillopt", "previous_plain", "recurrence_plain", "recurrence_ledger",
         "static_ledger", "random_ledger")
DEFAULT_CONFIG = {
    "injected_block_cap_characters": 2304, "no_progress_actions": 8, "recurrence_window": 16,
    "recurrence_visits": 2, "minimum_action_separation": 2, "max_receptacles": 128,
    "max_target_instances": 64, "unsupported_feedback_records": 4,
}
ENTITY = r"[a-z][a-z_-]*(?: [a-z][a-z_-]*)* \d+"


class UnsupportedPublicState(ValueError):
    pass


class Inspection(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "explicit_empty"
    CONTENTS = "explicit_contents"


@dataclass(frozen=True)
class PublicGoal:
    object_type: str
    count: int
    condition: str | None
    destination_type: str | None
    light_type: str | None = None


@dataclass
class Receptacle:
    navigable: bool = False
    visited: bool = False
    access: str = "unknown"
    inspection: Inspection = Inspection.UNKNOWN
    contents_currently_visible: bool = False
    last_contents: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PublicEvent:
    kind: str
    location: str | None = None
    objects: tuple[str, ...] | None = None
    object_id: str | None = None
    destination: str | None = None
    state: str | None = None
    navigable: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ProgressDecision:
    active: bool
    activated_now: bool
    message: str | None
    trace: Mapping[str, Any]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def parse_goal(task: str) -> PublicGoal:
    text = " ".join(task.lower().strip().rstrip(".").split())
    placement = re.fullmatch(
        r"put (two |a |an |some |the )?(?:(hot|cool|cold|clean) )?([a-z_-]+) (?:in|on) ([a-z_-]+)", text,
    )
    if placement:
        article, condition, obj, destination = placement.groups()
        return PublicGoal(obj, 2 if article == "two " else 1,
                          "cool" if condition == "cold" else condition, destination)
    two = re.fullmatch(r"find two ([a-z_-]+) and put them (?:in|on) ([a-z_-]+)", text)
    if two:
        return PublicGoal(two[1], 2, None, two[2])
    transform = re.fullmatch(
        r"(heat|cool|clean) (?:some |a |an |the )?([a-z_-]+) and put it (?:in|on) ([a-z_-]+)", text,
    )
    if transform:
        verb, obj, destination = transform.groups()
        return PublicGoal(obj, 1, {"heat": "hot", "cool": "cool", "clean": "clean"}[verb], destination)
    light = re.fullmatch(
        r"(?:look at|examine) (?:the |a |an )?([a-z_-]+) (?:under|with) (?:the |a |an )?([a-z_-]+)", text,
    )
    if light:
        return PublicGoal(light[1], 1, None, None, light[2])
    raise UnsupportedPublicState("Unsupported public task wording; no target facts inferred")


def entities(text: str) -> tuple[str, ...]:
    text = text.strip().rstrip(".").strip()
    if text.lower() in ("nothing", "nothing else", "empty"):
        return ()
    values = []
    for part in re.split(r"\s*,\s*|\s+and\s+", text.lower()):
        part = re.sub(r"^(?:and )?(?:a |an |the )?", "", part.strip())
        if not part:
            continue
        if re.fullmatch(ENTITY, part) is None:
            raise UnsupportedPublicState("Unparsed public entity list")
        values.append(part)
    if not values:
        raise UnsupportedPublicState("Empty unparsed entity list is not an empty receptacle")
    return tuple(sorted(set(values)))


def initial_navigation(observation: str) -> tuple[str, ...]:
    text = " ".join(observation.split())
    match = re.search(r"Looking quickly around you, you see (.*?)\.\s*Your task is to:", text, re.I)
    if match is None:
        raise UnsupportedPublicState("Initial public navigable-receptacle description is unsupported")
    return entities(match[1])


def parse_feedback(observation: str, *, tool_error: bool) -> PublicEvent:
    if not isinstance(observation, str) or type(tool_error) is not bool:
        raise TypeError("Only ordinary public feedback and its explicit error flag are accepted")
    text = " ".join(observation.lower().split()).strip()
    if tool_error:
        return PublicEvent("error")
    try:
        arrival = re.fullmatch(rf"you arrive at ({ENTITY})\.\s*(.*)", text)
        opened = re.fullmatch(rf"you open (?:the )?({ENTITY})\.\s*(.*)", text)
        inspected = re.fullmatch(rf"(?:on|in) (?:the )?({ENTITY}), you see (.*)", text)
        open_inspection = re.fullmatch(rf"(?:the )?({ENTITY}) is open\. in it, you see (.*)", text)
        if open_inspection:
            return PublicEvent("inspection", open_inspection[1], entities(open_inspection[2]), state="open")
        if arrival or opened:
            match = arrival or opened
            location, tail = match[1], match[2]
            state = "open" if opened else None
            if re.fullmatch(rf"(?:the )?{re.escape(location)} is closed\.?", tail):
                if opened:
                    return PublicEvent("unsupported", reason="Contradictory open/closed public feedback")
                return PublicEvent("arrival" if arrival else "open", location, state="closed")
            tail = re.sub(rf"^(?:the )?{re.escape(location)} is open\.\s*", "", tail)
            contents = re.fullmatch(
                rf"(?:(?:on|in) (?:the )?{re.escape(location)}|in it), you see (.*)", tail,
            )
            if contents:
                return PublicEvent("arrival" if arrival else "open", location,
                                   objects=entities(contents[1]), state=state)
            if not tail:
                return PublicEvent("arrival" if arrival else "open", location, state=state)
            return PublicEvent("unsupported", reason="Unparsed arrival/open feedback tail")
        if inspected:
            return PublicEvent("inspection", inspected[1], entities(inspected[2]))
        closed = re.fullmatch(rf"you close (?:the )?({ENTITY})\.?", text)
        if closed:
            return PublicEvent("close", closed[1], state="closed")
        taken = re.fullmatch(rf"you pick up (?:the )?({ENTITY}) from (?:the )?({ENTITY})\.?", text)
        if taken:
            return PublicEvent("take", object_id=taken[1], location=taken[2])
        moved = re.fullmatch(rf"you move (?:the )?({ENTITY}) to (?:the )?({ENTITY})\.?", text)
        if moved:
            return PublicEvent("move", object_id=moved[1], destination=moved[2])
        transformed = re.fullmatch(rf"you (heat|cool|clean) (?:the )?({ENTITY}) using (?:the )?({ENTITY})\.?", text)
        if transformed:
            return PublicEvent("transform", object_id=transformed[2], location=transformed[3],
                               state={"heat": "hot", "cool": "cool", "clean": "clean"}[transformed[1]])
        light = re.fullmatch(rf"you turn (on|off) (?:the )?({ENTITY})\.?", text)
        if light:
            return PublicEvent("light", object_id=light[2], state=light[1])
        carrying = re.fullmatch(r"you are carrying:\s*(.*)", text)
        if carrying:
            return PublicEvent("inventory", objects=entities(carrying[1]))
        if text in ("you are not carrying anything.", "you are carrying nothing.", "your inventory is empty."):
            return PublicEvent("inventory", objects=())
        facing = re.fullmatch(r"you are facing (.*?)\.(?: next to it, you see (.*))?", text)
        if facing:
            return PublicEvent("overview", objects=entities(facing[2]) if facing[2] is not None else None,
                               navigable=entities(facing[1]))
        overview = re.search(r"looking quickly around you, you see (.*?)\.", text)
        if text.startswith("you are in the middle of a room.") and overview:
            return PublicEvent("overview", navigable=entities(overview[1]))
    except UnsupportedPublicState as error:
        return PublicEvent("unsupported", reason=str(error))
    return PublicEvent("unsupported", reason="Unknown public feedback wording")


@dataclass
class PublicProgressTracker:
    task_text: str
    initial_observation: str
    config: Mapping[str, int]
    goal: PublicGoal = field(init=False)
    receptacles: dict[str, Receptacle] = field(default_factory=dict, init=False)
    position: str | None = field(default=None, init=False)
    held: set[str] = field(default_factory=set, init=False)
    inventory_status: str = field(default="unknown", init=False)
    properties: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    placements: dict[str, str] = field(default_factory=dict, init=False)
    sightings: set[tuple[str, str]] = field(default_factory=set, init=False)
    lights: dict[str, str] = field(default_factory=dict, init=False)
    unsupported: dict[str, str] = field(default_factory=dict, init=False)
    _facts: set[tuple[Any, ...]] = field(default_factory=set, init=False)
    _history: deque = field(default_factory=deque, init=False)
    observations: int = field(default=0, init=False)
    no_progress_actions: int = field(default=0, init=False)
    recurrent: bool = field(default=False, init=False)
    last_update: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.goal = parse_goal(self.task_text)
        for name in initial_navigation(self.initial_observation):
            self._receptacle(name).navigable = True
            self._facts.add(("navigable", name))

    def _receptacle(self, name):
        if name not in self.receptacles:
            if len(self.receptacles) >= self.config["max_receptacles"]:
                raise UnsupportedPublicState("Public receptacle capacity exceeded")
            self.receptacles[name] = Receptacle()
        return self.receptacles[name]

    def _target(self, instance):
        return instance.rsplit(" ", 1)[0] == self.goal.object_type

    def _fact(self, *value):
        self._facts.add(tuple(value))

    def progress_signature(self):
        return canonical_hash(sorted(self._facts, key=repr))

    def task_state(self):
        return {
            "held": sorted(instance for instance in self.held if self._target(instance)),
            "inventory_status": self.inventory_status,
            "properties": {key: value for key, value in sorted(self.properties.items()) if self._target(key)},
            "placements": {key: value for key, value in sorted(self.placements.items()) if self._target(key)},
            "lights": {key: value for key, value in sorted(self.lights.items())
                       if self.goal.light_type and key.rsplit(" ", 1)[0] == self.goal.light_type},
        }

    def observe(self, action: str, observation: str, *, tool_error: bool):
        if not isinstance(action, str):
            raise TypeError("Executed public action must be text")
        self.observations += 1
        before = self.progress_signature()
        event = parse_feedback(observation, tool_error=tool_error)
        if event.kind == "unsupported":
            key = hashlib.sha256(observation.encode()).hexdigest()
            self.unsupported[key] = observation[:160]
            while len(self.unsupported) > self.config["unsupported_feedback_records"]:
                self.unsupported.pop(next(iter(self.unsupported)))
            self._history.clear()
            self.no_progress_actions = 0
            self.recurrent = False
            self.last_update = {"supported": False, "kind": event.kind, "reason": event.reason,
                                "task_progress": False, "recurrence": False,
                                "progress_signature": before}
            return self.last_update
        if event.kind in ("arrival", "open", "close", "inspection"):
            rec = self._receptacle(event.location)
            if event.objects is None:
                rec.contents_currently_visible = False
            if event.kind == "arrival":
                rec.navigable = rec.visited = True
                self.position = event.location
                self._fact("visited", event.location)
            if event.state is not None:
                rec.access = event.state
                self._fact("access_seen", event.location, event.state)
                if event.state == "closed":
                    rec.contents_currently_visible = False
            if event.objects is not None:
                if event.state is None:
                    rec.access = "publicly_exposed"
                rec.inspection = Inspection.CONTENTS if event.objects else Inspection.EMPTY
                rec.contents_currently_visible = True
                rec.last_contents = event.objects
                self._fact("inspected_contents", event.location, event.objects)
                for obj in event.objects:
                    if self._target(obj):
                        self.sightings.add((obj, event.location))
                        self._fact("target_sighting", obj, event.location)
        elif event.kind == "take":
            self.held.add(event.object_id)
            self.inventory_status = "partial_public_updates"
            self.placements.pop(event.object_id, None)
            if event.location in self.receptacles:
                self.receptacles[event.location].contents_currently_visible = False
            if self._target(event.object_id):
                self._fact("target_held", event.object_id)
        elif event.kind == "move":
            self.held.discard(event.object_id)
            self.inventory_status = "partial_public_updates"
            self.placements[event.object_id] = event.destination
            if event.destination in self.receptacles:
                self.receptacles[event.destination].contents_currently_visible = False
            if self._target(event.object_id):
                self._fact("target_placement", event.object_id, event.destination)
        elif event.kind == "inventory":
            self.held = set(event.objects)
            self.inventory_status = "explicit_inventory_observation"
            self._fact("target_inventory_seen", tuple(sorted(obj for obj in self.held if self._target(obj))))
        elif event.kind == "transform":
            properties = self.properties.setdefault(event.object_id, {})
            properties["clean" if event.state == "clean" else "temperature"] = (
                True if event.state == "clean" else event.state
            )
            if self._target(event.object_id):
                self._fact("target_transformation", event.object_id, event.state)
        elif event.kind == "light":
            self.lights[event.object_id] = event.state
            if self.goal.light_type and event.object_id.rsplit(" ", 1)[0] == self.goal.light_type:
                self._fact("requested_light", event.object_id, event.state)
        elif event.kind == "overview":
            for name in event.navigable:
                self._receptacle(name).navigable = True
                self._fact("navigable", name)
            if event.objects is not None:
                self._fact("overview_objects", event.objects)
                for obj in event.objects:
                    if self._target(obj):
                        self.sightings.add((obj, "location_unknown"))
                        self._fact("target_sighting_unknown_location", obj)
        if len({obj for obj, _ in self.sightings}) > self.config["max_target_instances"]:
            raise UnsupportedPublicState("Public target-instance capacity exceeded")
        progress = self.progress_signature() != before
        self.recurrent = False
        if progress:
            self.no_progress_actions = 0
            self._history.clear()
        else:
            self.no_progress_actions += 1
            if event.kind in ("arrival", "inspection", "overview") and self.position:
                key = canonical_hash({"position": self.position, "task_state": self.task_state()})
                while self._history and self.observations - self._history[0][0] > self.config["recurrence_window"]:
                    self._history.popleft()
                self.recurrent = has_no_progress_recurrence(
                    self._history, observation_index=self.observations, context=key,
                    no_progress_actions=self.no_progress_actions, window=self.config["recurrence_window"],
                    minimum_action_separation=self.config["minimum_action_separation"],
                    recurrence_visits=self.config["recurrence_visits"],
                    stall_actions=self.config["no_progress_actions"],
                )
                self._history.append((self.observations, key))
        self.last_update = {
            "supported": True, "kind": event.kind, "task_progress": progress,
            "recurrence": self.recurrent, "no_progress_actions": self.no_progress_actions,
            "progress_signature": self.progress_signature(),
        }
        return self.last_update

    def compact(self):
        def priority(item):
            name, rec = item
            return (0 if rec.inspection is Inspection.UNKNOWN else
                    1 if not rec.contents_currently_visible else 2, name)
        nav = []
        for name, rec in sorted(self.receptacles.items(), key=priority):
            if not rec.navigable:
                continue
            status = ("closed_unknown" if rec.access == "closed"
                      else rec.inspection.value if rec.contents_currently_visible else "unknown")
            nav.append([name, rec.visited, status])
        return {
            "goal": [self.goal.object_type, self.goal.count, self.goal.condition,
                     self.goal.destination_type, self.goal.light_type],
            "position": self.position, "inventory_status": self.inventory_status,
            "held": sorted(self.held),
            "properties": [[key, value] for key, value in sorted(self.properties.items()) if self._target(key)],
            "placements": [[key, value] for key, value in sorted(self.placements.items()) if self._target(key)],
            "sightings": [list(value) for value in sorted(self.sightings)],
            "sightings_are_historical": True,
            "lights": [[key, value] for key, value in sorted(self.lights.items())],
            "nav": nav,
            "nav_columns": ["receptacle", "visited", "public_contents_state"],
            "unsupported": [[key[:12], value] for key, value in self.unsupported.items()],
            "omitted": {},
            "evidence_only_not_goal_success": True,
        }


def render_ledger(tracker: PublicProgressTracker, directive: str, cap: int) -> str:
    prefix = (
        "[ALF public progress ledger: active]\n" + directive + "\n"
        "Only explicit public evidence is summarized; unknown/closed is not empty. "
        "Omitted entries are not negative facts or completed goals. Keep the original task and tools.\n"
        "<public_progress_ledger>\n"
    )
    suffix = "\n</public_progress_ledger>"
    value = tracker.compact()
    while True:
        text = prefix + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + suffix
        if len(text) <= cap:
            return text
        for name in ("nav", "unsupported", "sightings", "lights", "properties", "placements", "held"):
            if value[name]:
                value[name].pop()
                value["omitted"][name] = value["omitted"].get(name, 0) + 1
                break
        else:
            raise UnsupportedPublicState("Essential ledger and exact directive do not fit the common block cap")


class PublicProgressSession:
    """One opt-in install; ordinary state updates are not new edit operations."""

    def __init__(self, patch, *, mode, task_id, expected_parent_sha256, seed,
                 task_text, initial_observation, config):
        if mode not in MODES:
            raise ValueError("Unknown fixed mechanism arm")
        self.patch = parse_patch(patch)
        self.mode = mode
        self.config = {key: config[key] for key in DEFAULT_CONFIG}
        if any(type(value) is not int or value <= 0 for value in self.config.values()):
            raise ValueError("Progress configuration bounds must be positive")
        self.tracker = None if mode in ("skillopt", "previous_plain") else PublicProgressTracker(
            task_text, initial_observation, self.config,
        )
        backend_mode = ("CONDITIONAL_PATCH" if mode == "previous_plain" else
                        "RANDOM_TIME" if mode == "random_ledger" else "STATIC_PATCH")
        self.backend = RepairSession(self.patch, mode=backend_mode, task_id=task_id,
                                     expected_parent_sha256=expected_parent_sha256, seed=seed, random_window=8)
        self.active = False
        self._last_turn = -1
        self._traces = []

    def observe(self, action, observation, *, tool_error):
        self.backend.observe(action, observation, tool_error=tool_error)
        if self.tracker is not None:
            self.tracker.observe(action, observation, tool_error=tool_error)

    def before_actor(self, *, turn_index, remaining_steps, stage="EXECUTE"):
        if turn_index <= self._last_turn:
            raise ValueError("Use one increasing logical actor turn")
        self._last_turn = turn_index
        recurrence_mode = self.mode in ("recurrence_plain", "recurrence_ledger")
        install = recurrence_mode and not self.active and self.tracker.recurrent and remaining_steps > 0
        self.active = self.active or install
        decision = None
        if self.mode != "skillopt" and (not recurrence_mode or self.active):
            decision = self.backend.before_actor(
                turn_index=turn_index, remaining_steps=remaining_steps, stage=stage,
            )
            self.active = decision.active
            install = decision.activated_now
        message = None
        if decision and decision.message:
            message = (decision.message if self.mode in ("previous_plain", "recurrence_plain")
                       else render_ledger(self.tracker, self.patch.directive,
                                          self.config["injected_block_cap_characters"]))
        if message is not None and len(message) > self.config["injected_block_cap_characters"]:
            raise UnsupportedPublicState("Exact reference exceeds the frozen common block cap; no truncation")
        trace = {
            "schema_version": TRACKER_VERSION, "mode": self.mode, "turn_index": turn_index,
            "active": self.active, "activated_now": install, "component_installs": int(install),
            "ledger_updates_are_edits": False,
            "public_progress": dict(self.tracker.last_update) if self.tracker else None,
            "message_sha256": hashlib.sha256(message.encode()).hexdigest() if message else None,
            "message_characters": len(message) if message else 0,
            "core_trace": dict(decision.trace) if decision else None,
            "uses_hidden_facts_ids_or_outcomes_for_progress": False,
        }
        self._traces.append(trace)
        return ProgressDecision(self.active, install, message, trace)

    @property
    def traces(self):
        return list(self._traces)

    def public_progress_snapshot(self):
        return self.tracker.compact() if self.tracker else None
