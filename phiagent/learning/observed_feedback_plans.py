"""Actor-authored one-step continuations using observed TRAIN feedback events.

Only source action text supplies the entity-type vocabulary. Entity references
are a word/type followed by a decimal instance, with or without separating
whitespace. Action shapes replace whole references by ordered distinct slots;
repeated references retain their slot identity. Nonentity text remains literal.
Matching normalizes whitespace, preserves case, and never executes predicates.

Feedback templates replace references bound in the first action. A first unbound
reference ends a prefix template; an empty/whitespace-only prefix is rejected.
Prefix trailing whitespace is retained as a matching boundary. Events require
independent source groups, not repeated records. Corpus IDs are provenance only.
An observed match is not a correctness claim or a guaranteed effect.

catalog_for_actions factors the menu by learned action-shape/feedback-template,
assigning request-local e0/e1/... aliases in support-ranked catalogue-ID order.
Each entry retains its long catalog_event_id and all source evidence. first_actions
and bindings retain the exact allowed-first-action -> ordered-argument map in
structured data. Only model text uses short aliases and one-line $0/$1 templates.
maximum_events limits templates, not first actions. Neither evidence nor hashes
are removed from stored artifacts. State audits bind aliases to menu_sha256 and
record both the local event_id and long catalog_event_id.
Runtime binds the chosen first action before matching; entity IDs are never guessed.
All first native actions remain in the response enum, shared by structured_single.
Event-plan generation is identical across the three runtime event modes.
All four modes in this event representation bound reasoning to 1800 characters.
This is a shared output-contract change, not a pure runtime-guard ablation or a
guarantee against provider truncation; incomplete JSON still raises ValueError.

Lifecycle: decide -> actor/parse -> install -> execute first action -> observe ->
decide -> at most one native continuation -> observe -> new actor boundary.
install can also start from an empty state. Invalid calls raise ValueError without
consuming state; failed runtime checks consume the plan and request the actor.
The next decide must use the EXACT recorded first-action feedback string.

event_guarded enforces the event and the new-goal-information barrier;
event_no_novelty removes only the barrier; event_unchecked removes only event
matching. Novel entities are casefolded TRAIN type + exact decimal-instance
strings present in first feedback but absent from pre-observation AND first-action
arguments. Exact task-word tokens select goal types; no stemming or task rules.
All modes require native continuation membership. The parent owns model calls,
explicit format fallback, task termination, source-split checks and native runs.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re


CATALOG_SCHEMA = "chisel-observed-feedback-catalog/v1"
MENU_SCHEMA = "chisel-grounded-feedback-event-menu/v3"
STATE_SCHEMA = "chisel-observed-feedback-event-plan-state/v3"
MODES = ("event_guarded", "event_no_novelty", "event_unchecked", "structured_single")
_ENTITY = re.compile(
    r"(?<!\w)(?P<type>[^\W\d]+(?:-[^\W\d]+)*)\s*(?P<instance>[0-9]+)(?!\w)"
)
_WORDS = re.compile(r"\w+(?:-\w+)*")
_RECORD_KEYS = {"source_id", "task_id", "group_id", "action", "post_observation"}
_EVENT_KEYS = {
    "action_shape", "argument_count", "feedback_template", "match_type",
    "sources", "source_count", "support_groups", "support_group_count",
}
_MENU_EVENT_KEYS = {
    "catalog_event_id", "action_shape", "argument_count", "feedback_template",
    "match_type", "description", "first_actions", "bindings",
    "source_count", "source_ids", "support_groups", "support_group_count",
}


def _text(value: object, name: str, *, empty: bool = False, limit: int | None = None) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}string")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} contains an invalid Unicode surrogate") from error
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _record(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Invalid {name} keys")
    return value


def _strings(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    for item in value:
        _text(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"Duplicate {name}")
    return value


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected finite JSON-serializable data") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _mode(value: object) -> str:
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("Unknown event-plan mode")
    return value


def _identity(match) -> tuple[str, str]:
    return match["type"].casefold(), match["instance"]


def _action_shape(action: str) -> tuple[list, list[str], dict]:
    text = _normalize(action)
    parts, arguments, slots, end = [], [], {}, 0
    for match in _ENTITY.finditer(text):
        entity = _identity(match)
        if entity not in slots:
            slots[entity] = len(arguments)
            arguments.append(match[0])
        slot = slots[entity]
        if arguments[slot] != match[0]:
            raise ValueError("One action entity has inconsistent literal spellings")
        parts.extend([text[end:match.start()], {"slot": slot}])
        end = match.end()
    parts.append(text[end:])
    return parts, arguments, slots


def _render(template: list, arguments: list[str]) -> str:
    return "".join(part if isinstance(part, str) else arguments[part["slot"]] for part in template)


def _compact_template(template: list) -> str:
    """Render slots without losing literal pipes, dollars, escapes or prefix boundaries."""
    escapes = {"\\": "\\\\", "|": "\\|", "$": "\\$", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    parts = []
    for index, part in enumerate(template):
        if isinstance(part, dict):
            parts.append(f'${part["slot"]}')
            continue
        for offset, char in enumerate(part):
            if char in escapes:
                parts.append(escapes[char])
            elif (ord(char) < 32 or char in "\x7f\x85\u2028\u2029"
                  or (index > 0 and offset == 0 and char in "0123456789")):
                parts.append(f"\\u{ord(char):04x}")
            else:
                parts.append(char)
    rendered = "".join(parts)
    leading = len(rendered) - len(rendered.lstrip(" "))
    trailing = len(rendered) - len(rendered.rstrip(" "))
    if leading == len(rendered):
        return "\\u0020" * leading
    return "\\u0020" * leading + rendered[leading:len(rendered) - trailing] + "\\u0020" * trailing


def _menu_entries(menu: dict):
    for index in range(len(menu["events"])):
        alias = f"e{index}"
        yield alias, menu["events"][alias]


def _feedback_template(observation: str, arguments: list[str], slots: dict) -> tuple[list, str, str | None]:
    text, parts, end = _normalize(observation), [], 0
    for match in _ENTITY.finditer(text):
        entity = _identity(match)
        if entity not in slots:
            parts.append(text[end:match.start()])
            return parts, "prefix_match", None if _render(parts, arguments).strip() else "empty_unbound_prefix"
        slot = slots[entity]
        if arguments[slot] != match[0]:
            return [], "exact_match", "bound_feedback_reference_spelling_mismatch"
        parts.extend([text[end:match.start()], {"slot": slot}])
        end = match.end()
    parts.append(text[end:])
    return parts, "exact_match", None


def _event_id(event: dict) -> str:
    return "ev_" + _hash({key: event[key] for key in ("action_shape", "feedback_template", "match_type")})


def fit_feedback_catalog(transitions: list[dict], *, excluded_groups=frozenset(),
                         minimum_groups: int = 2) -> dict:
    """Fit observed templates only; excluded groups never supply vocabulary or support."""
    _integer(minimum_groups, "minimum_groups", 2)
    if not isinstance(transitions, list) or not isinstance(excluded_groups, (set, frozenset)):
        raise ValueError("Expected a transition list and a set of excluded groups")
    for group in excluded_groups:
        _text(group, "excluded group")
    records, tasks, vocabulary = {}, {}, defaultdict(lambda: {"source_ids": set(), "source_groups": set()})
    counts = {
        "input_records": len(transitions), "excluded_records": 0, "duplicate_records": 0,
        "retained_records": 0, "prefix_records": 0, "rejected_records": 0,
        "candidate_events": 0, "rejected_low_support_events": 0, "admitted_events": 0,
    }
    for row in transitions:
        _record(row, _RECORD_KEYS, "TRAIN transition")
        for field in _RECORD_KEYS:
            _text(row[field], field, empty=field == "post_observation")
        if row["group_id"] in excluded_groups:
            counts["excluded_records"] += 1
            continue
        if tasks.setdefault(row["task_id"], row["group_id"]) != row["group_id"]:
            raise ValueError("One source task cannot belong to different source groups")
        identifier = row["source_id"]
        if identifier in records:
            if records[identifier] != row:
                raise ValueError("Conflicting duplicate source_id")
            counts["duplicate_records"] += 1
        else:
            records[identifier] = deepcopy(row)
    counts["retained_records"] = len(records)
    for row in records.values():
        for match in _ENTITY.finditer(_normalize(row["action"])):
            entry = vocabulary[match["type"].casefold()]
            entry["source_ids"].add(row["source_id"])
            entry["source_groups"].add(row["group_id"])
    candidates, rejected = {}, []
    for row in sorted(records.values(), key=lambda item: item["source_id"]):
        shape, arguments, slots = _action_shape(row["action"])
        template, match_type, reason = _feedback_template(row["post_observation"], arguments, slots)
        if reason is not None:
            counts["rejected_records"] += 1
            rejected.append({"reason": reason, "source_ids": [row["source_id"]],
                             "support_groups": [row["group_id"]]})
            continue
        counts["prefix_records"] += int(match_type == "prefix_match")
        pattern = {
            "action_shape": shape, "argument_count": len(arguments),
            "feedback_template": template, "match_type": match_type,
        }
        event = candidates.setdefault(_event_id(pattern), {**pattern, "sources": []})
        event["sources"].append({key: row[key] for key in ("source_id", "task_id", "group_id")})
    events = {}
    for identifier, event in sorted(candidates.items()):
        groups = sorted({row["group_id"] for row in event["sources"]})
        event.update(source_count=len(event["sources"]), support_groups=groups, support_group_count=len(groups))
        if len(groups) < minimum_groups:
            counts["rejected_low_support_events"] += 1
            rejected.append({"reason": "insufficient_independent_groups", "event_id": identifier, **event})
        else:
            events[identifier] = event
    counts["candidate_events"], counts["admitted_events"] = len(candidates), len(events)
    return {
        "schema_version": CATALOG_SCHEMA, "minimum_groups": minimum_groups,
        "excluded_groups": sorted(excluded_groups),
        "entity_vocabulary": {name: {key: sorted(value) for key, value in evidence.items()}
                              for name, evidence in sorted(vocabulary.items())},
        "events": events, "counts": counts, "rejections": sorted(rejected, key=_json),
        "semantics": "Observed TRAIN feedback patterns; no guaranteed effect, goal correctness or safety probability",
    }


def _template(template: object, argument_count: int, *, shape: bool = False) -> None:
    if not isinstance(template, list) or not template or len(template) % 2 != 1:
        raise ValueError("Templates must alternate literal text and argument slots")
    slots = []
    for index, part in enumerate(template):
        if index % 2 == 0:
            _text(part, "template literal", empty=True)
        else:
            _record(part, {"slot"}, "argument slot")
            slot = _integer(part["slot"], "slot")
            if slot >= argument_count:
                raise ValueError("Unbound event-template slot")
            slots.append(slot)
    if shape and list(dict.fromkeys(slots)) != list(range(argument_count)):
        raise ValueError("Action arguments must use ordered distinct slots")


def _validate_catalog(catalog: dict) -> None:
    _record(catalog, {
        "schema_version", "minimum_groups", "excluded_groups", "entity_vocabulary",
        "events", "counts", "rejections", "semantics",
    }, "feedback catalogue")
    if catalog["schema_version"] != CATALOG_SCHEMA:
        raise ValueError("Unknown feedback catalogue schema")
    minimum = _integer(catalog["minimum_groups"], "minimum_groups", 2)
    excluded = set(_strings(catalog["excluded_groups"], "excluded groups"))
    if not isinstance(catalog["entity_vocabulary"], dict) or not isinstance(catalog["events"], dict):
        raise ValueError("Catalogue vocabulary/events must be mappings")
    for kind, provenance in catalog["entity_vocabulary"].items():
        _text(kind, "entity type")
        if kind != kind.casefold() or re.fullmatch(r"[^\W\d]+(?:-[^\W\d]+)*", kind) is None:
            raise ValueError("Entity vocabulary requires normalized native type words")
        _record(provenance, {"source_ids", "source_groups"}, "type provenance")
        ids = _strings(provenance["source_ids"], "vocabulary source IDs")
        groups = _strings(provenance["source_groups"], "vocabulary groups")
        if not ids or not groups or set(groups) & excluded:
            raise ValueError("Vocabulary provenance is empty or contains excluded groups")
    for identifier, event in catalog["events"].items():
        _record(event, _EVENT_KEYS, "catalogue event")
        count = _integer(event["argument_count"], "argument_count")
        _template(event["action_shape"], count, shape=True)
        _template(event["feedback_template"], count)
        if event["match_type"] not in ("exact_match", "prefix_match") or identifier != _event_id(event):
            raise ValueError("Invalid event identity or match type")
        if event["match_type"] == "prefix_match" and not _render(
            event["feedback_template"], ["ARG"] * count,
        ).strip():
            raise ValueError("An empty prefix cannot be admitted")
        if not isinstance(event["sources"], list):
            raise ValueError("Event sources must be a list")
        ids, groups = set(), set()
        for source in event["sources"]:
            _record(source, {"source_id", "task_id", "group_id"}, "event provenance")
            for field in source:
                _text(source[field], field)
            if source["source_id"] in ids:
                raise ValueError("Duplicate event source identity")
            ids.add(source["source_id"])
            groups.add(source["group_id"])
        if (len(groups) < minimum or groups & excluded
                or event["support_groups"] != sorted(groups)
                or _integer(event["support_group_count"], "support_group_count") != len(groups)
                or _integer(event["source_count"], "source_count") != len(ids)):
            raise ValueError("Event lacks consistent independent non-excluded group support")
    if not isinstance(catalog["counts"], dict) or not isinstance(catalog["rejections"], list):
        raise ValueError("Invalid catalogue audit metadata")
    for value in catalog["counts"].values():
        _integer(value, "catalogue count")
    _text(catalog["semantics"], "catalogue semantics")
    _json(catalog)


def catalog_for_actions(catalog: dict, native_actions: list[str], *, maximum_events: int = 24) -> dict:
    """Select the same support-ranked templates, then assign local aliases with full provenance."""
    _validate_catalog(catalog)
    _strings(native_actions, "native actions")
    _integer(maximum_events, "maximum_events", 1)
    available = {}
    for action in sorted(native_actions):
        shape, arguments, _ = _action_shape(action)
        for identifier, event in catalog["events"].items():
            if event["action_shape"] != shape:
                continue
            if identifier not in available:
                available[identifier] = {
                    "catalog_event_id": identifier,
                    "action_shape": deepcopy(event["action_shape"]),
                    "argument_count": event["argument_count"],
                    "feedback_template": deepcopy(event["feedback_template"]),
                    "match_type": event["match_type"],
                    "description": f'{event["match_type"]}: {_json(event["feedback_template"])}',
                    "first_actions": [], "bindings": {}, "source_count": event["source_count"],
                    "source_ids": sorted(source["source_id"] for source in event["sources"]),
                    "support_groups": list(event["support_groups"]),
                    "support_group_count": event["support_group_count"],
                }
            available[identifier]["first_actions"].append(action)
            available[identifier]["bindings"][action] = list(arguments)
    ranked = sorted(available, key=lambda identifier: (-available[identifier]["support_group_count"], identifier))
    selected = {f"e{index}": available[identifier]
                for index, identifier in enumerate(ranked[:maximum_events])}
    candidate_bindings = sum(len(event["bindings"]) for event in available.values())
    selected_bindings = sum(len(event["bindings"]) for event in selected.values())
    return {
        "schema_version": MENU_SCHEMA, "catalog_sha256": _hash(catalog),
        "native_actions": sorted(native_actions), "maximum_events": maximum_events, "events": selected,
        "action_events": {action: sorted(identifier for identifier, event in selected.items()
                                        if action in event["first_actions"]) for action in sorted(native_actions)},
        "counts": {
            "candidate_events": len(available), "selected_events": len(selected),
            "truncated_events": len(available) - len(selected),
            "candidate_bindings": candidate_bindings, "selected_bindings": selected_bindings,
            "truncated_bindings": candidate_bindings - selected_bindings,
        },
    }


def _validate_menu(menu: dict, native_actions: list[str]) -> None:
    _strings(native_actions, "native actions")
    _record(menu, {"schema_version", "catalog_sha256", "native_actions", "maximum_events",
                   "events", "action_events", "counts"}, "event menu")
    if menu["schema_version"] != MENU_SCHEMA or menu["native_actions"] != sorted(native_actions):
        raise ValueError("Event menu does not describe this native action set")
    digest = _text(menu["catalog_sha256"], "catalogue hash")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Invalid catalogue hash")
    limit = _integer(menu["maximum_events"], "maximum_events", 1)
    if not isinstance(menu["events"], dict) or len(menu["events"]) > limit:
        raise ValueError("Invalid menu event set or event limit")
    if set(menu["events"]) != {f"e{index}" for index in range(len(menu["events"]))}:
        raise ValueError("Menu aliases must be consecutive request-local e0/e1/... IDs")
    catalogue_ids = set()
    for identifier, event in menu["events"].items():
        _record(event, _MENU_EVENT_KEYS, "factored menu event")
        _text(identifier, "event ID", limit=160)
        _text(event["catalog_event_id"], "catalogue event ID")
        count = _integer(event["argument_count"], "argument_count")
        _template(event["action_shape"], count, shape=True)
        _template(event["feedback_template"], count)
        if (event["match_type"] not in ("exact_match", "prefix_match")
                or event["catalog_event_id"] != _event_id(event)
                or event["catalog_event_id"] in catalogue_ids):
            raise ValueError("Invalid factored event identity")
        catalogue_ids.add(event["catalog_event_id"])
        if event["description"] != f'{event["match_type"]}: {_json(event["feedback_template"])}':
            raise ValueError("Event description disagrees with its feedback template")
        actions = _strings(event["first_actions"], "event first actions")
        ids = _strings(event["source_ids"], "event source IDs")
        groups = _strings(event["support_groups"], "event support groups")
        if (not actions or not set(actions) <= set(native_actions)
                or _integer(event["source_count"], "source_count") != len(ids)
                or _integer(event["support_group_count"], "support_group_count", 2) != len(groups)
                or len(ids) < len(groups)):
            raise ValueError("Inconsistent event action relation or source support")
        if not isinstance(event["bindings"], dict) or set(event["bindings"]) != set(actions):
            raise ValueError("Event bindings must cover exactly its allowed first actions")
        for action, arguments in event["bindings"].items():
            _strings(arguments, "event arguments")
            shape, expected_arguments, _ = _action_shape(action)
            if shape != event["action_shape"] or arguments != expected_arguments:
                raise ValueError("Event binding disagrees with the actual native action arguments")
            if event["match_type"] == "prefix_match" and not _render(
                event["feedback_template"], arguments,
            ).strip():
                raise ValueError("Empty grounded prefix")
    ranked_aliases = sorted(menu["events"], key=lambda alias: (
        -menu["events"][alias]["support_group_count"], menu["events"][alias]["catalog_event_id"],
    ))
    if ranked_aliases != [alias for alias, _ in _menu_entries(menu)]:
        raise ValueError("Menu aliases do not follow support-ranked catalogue-ID order")
    expected = {action: sorted(identifier for identifier, event in menu["events"].items()
                               if action in event["first_actions"]) for action in sorted(native_actions)}
    if menu["action_events"] != expected:
        raise ValueError("Event/first-action relation is inconsistent")
    _record(menu["counts"], {
        "candidate_events", "selected_events", "truncated_events",
        "candidate_bindings", "selected_bindings", "truncated_bindings",
    }, "menu counts")
    for value in menu["counts"].values():
        _integer(value, "menu count")
    if (menu["counts"]["selected_events"] != len(menu["events"])
            or menu["counts"]["candidate_events"] - menu["counts"]["selected_events"]
            != menu["counts"]["truncated_events"]
            or menu["counts"]["selected_bindings"]
            != sum(len(event["bindings"]) for event in menu["events"].values())
            or menu["counts"]["candidate_bindings"] - menu["counts"]["selected_bindings"]
            != menu["counts"]["truncated_bindings"]):
        raise ValueError("Menu counts disagree with event selection")


def response_schema(native_actions: list[str], menu: dict, *, allow_continuation: bool) -> dict:
    """Provider-compatible schema: native FIRST-action enum, no uniqueItems or regex pattern."""
    _bool(allow_continuation, "allow_continuation")
    _validate_menu(menu, native_actions)
    if not native_actions:
        raise ValueError("Generation requires at least one native first action")
    text = {"type": "string", "minLength": 1, "maxLength": 160}
    continuation = {
        "type": "object", "additionalProperties": False, "required": ["action", "after_event"],
        "description": "One next action, only after actual intermediate feedback from the first action.",
        "properties": {
            "action": {**text, "description": "Complete next native command with already bound arguments."},
            "after_event": {**text, "enum": [alias for alias, _ in _menu_entries(menu)],
                            "description": "Request-local event alias for the first action; bind its argument slots."},
        },
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["reasoning", "action", "continuation"],
        "properties": {
            "reasoning": {"type": "string", "minLength": 1, "maxLength": 1800},
            "action": {**text, "enum": sorted(native_actions),
                       "description": "Execute now: one complete current native command, exactly as listed."},
            "continuation": {"anyOf": [{"type": "null"}, continuation]}
            if allow_continuation and menu["events"] else {"type": "null"},
        },
    }


def _validated_plan(plan: dict, native_actions: list[str], menu: dict, allow: bool) -> dict:
    _bool(allow, "allow_continuation")
    _validate_menu(menu, native_actions)
    _record(plan, {"reasoning", "action", "continuation"}, "event plan")
    reasoning = _text(plan["reasoning"], "reasoning", limit=1800)
    action = _text(plan["action"], "first action", limit=160)
    if action not in native_actions:
        raise ValueError("First action is not an exact current native action")
    continuation = plan["continuation"]
    if continuation is not None:
        if not allow or not menu["events"]:
            raise ValueError("Continuation must be null for single mode or an empty event menu")
        _record(continuation, {"action", "after_event"}, "continuation")
        next_action = _text(continuation["action"], "continuation action", limit=160)
        event = _text(continuation["after_event"], "after_event", limit=160)
        if event not in menu["events"]:
            raise ValueError("Unknown after_event alias")
        if event not in menu["action_events"][action]:
            raise ValueError("after_event is not allowed for the chosen first action")
        continuation = {"action": next_action, "after_event": event}
    return {"reasoning": reasoning, "action": action, "continuation": continuation}


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str):
    raise ValueError(f"Non-finite JSON value is forbidden: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _nonfinite(value)
    return number


def parse_event_plan(text: str, *, native_actions: list[str], menu: dict,
                     allow_continuation: bool) -> dict:
    """Parse one strict JSON object. No fallback, action repair or guessed event IDs."""
    _bool(allow_continuation, "allow_continuation")
    _text(text, "event-plan JSON")
    try:
        plan = json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_nonfinite, parse_float=_finite_float)
    except json.JSONDecodeError as error:
        raise ValueError(f"Expected exactly one JSON object: {error.msg}") from error
    except RecursionError as error:
        raise ValueError("Event-plan JSON nesting is too deep") from error
    return _validated_plan(plan, native_actions, menu, allow_continuation)


def prompt_contract(mode: str, menu: dict) -> str:
    """Identical event-mode generation; single mode renders no unused event catalogue."""
    _mode(mode)
    _validate_menu(menu, menu.get("native_actions") if isinstance(menu, dict) else [])
    common = (
        'Return ONLY JSON with required "reasoning", "action", "continuation" keys. This replaces '
        'earlier <think>/<action> tags; task, observation and native-action information still applies. '
        'No extra/duplicate keys, code or tool calls. reasoning: nonempty, <=1800 characters. '
        'action: execute NOW, one exact current native command from the schema enum, <=160 characters, '
        'with every argument and case preserved.\n'
    )
    if mode == "structured_single":
        return common + '"continuation" must be null. Choose only the current action.'
    if not menu["events"]:
        return common + 'No supported events are available; "continuation" must be null.'
    lines = [
        f'{alias} | {_compact_template(event["action_shape"])} | '
        f'{"exact" if event["match_type"] == "exact_match" else "prefix"} | '
        f'{_compact_template(event["feedback_template"])}'
        for alias, event in _menu_entries(menu)
    ]
    return (
        'JSON only. reasoning: nonempty, <=1800 characters; action: exact current native enum command; '
        'continuation: null or object with string action and after_event. Commands <=160 characters, '
        'complete bound arguments. JSON replaces earlier output tags; task information still applies.\n'
        'You may propose ONE continuation for a PLAUSIBLE event branch without predicting that branch '
        'will occur. Runtime checks actual first-action feedback BEFORE the continuation; mismatch '
        'returns to actor. Use null if no suitable branch; after one continuation return to actor.\n'
        'Events are observed TRAIN feedback patterns, not guaranteed effects. $0/$1/... bind distinct '
        'first-action entities in appearance order; repeated slots reuse the same entity. Choose the '
        'matching action template. exact/prefix match case-sensitive, whitespace-normalized feedback.\n'
        r'Display escapes: \\ \| \$ \n \r \t \uXXXX. Do not copy display notation into commands.'
        '\nEvents: id | action | match | feedback\n' + "\n".join(lines)
    )


class EventPlanState:
    """One installed actor action plus at most one event-conditioned native continuation."""

    def __init__(self, catalog: dict, task: str, mode: str):
        _validate_catalog(catalog)
        _text(task, "task")
        self.mode = _mode(mode)
        self._catalog = deepcopy(catalog)
        self._catalog_hash = _hash(catalog)
        self._goal_types = set(catalog["entity_vocabulary"]) & set(_WORDS.findall(task.casefold()))
        self._phase, self._plan, self._event = "empty", None, None
        self._plan_hash, self._menu_hash, self._feedback, self._boundary = None, None, None, None
        self._pre_entities, self._completed = set(), 0

    def _entities(self, text: str) -> set[tuple[str, str]]:
        return {_identity(match) for match in _ENTITY.finditer(_normalize(text))
                if match["type"].casefold() in self._goal_types}

    def _audit(self) -> dict:
        return {
            "schema_version": STATE_SCHEMA, "mode": self.mode, "phase": self._phase,
            "catalog_sha256": self._catalog_hash, "plan_sha256": self._plan_hash,
            "menu_sha256": self._menu_hash,
            "completed_actions": self._completed, "actor_required": self._phase in ("empty", "awaiting_plan"),
            "boundary_reason": self._boundary, "goal_types": sorted(self._goal_types),
        }

    def _clear(self, reason: str, *, awaiting_plan: bool = False) -> None:
        self._plan = self._event = self._feedback = None
        self._pre_entities = set()
        self._phase = "awaiting_plan" if awaiting_plan else "empty"
        self._boundary = reason

    def install(self, plan: dict, *, current_observation: str,
                current_actions: list[str], menu: dict) -> dict:
        if self._phase not in ("empty", "awaiting_plan"):
            raise ValueError("Cannot overwrite an outstanding event plan")
        _text(current_observation, "current observation", empty=True)
        validated = _validated_plan(plan, current_actions, menu, self.mode != "structured_single")
        if menu != catalog_for_actions(self._catalog, current_actions, maximum_events=menu["maximum_events"]):
            raise ValueError("Menu is stale, altered or belongs to another catalogue")
        continuation = validated["continuation"]
        bound_event = None
        if continuation is not None:
            event = menu["events"][continuation["after_event"]]
            arguments = list(event["bindings"][validated["action"]])
            bound_event = {
                "catalog_event_id": event["catalog_event_id"], "match_type": event["match_type"],
                "arguments": arguments, "feedback": _render(event["feedback_template"], arguments),
            }
        self._plan, self._plan_hash = validated, _hash(validated)
        self._menu_hash = _hash(menu)
        self._event = bound_event
        self._pre_entities = self._entities(current_observation) | self._entities(validated["action"])
        self._phase, self._boundary, self._feedback = "awaiting_first_action", None, None
        return {
            **self._audit(), "reason": "actor_action_pending", "action": validated["action"],
            "event_id": None if continuation is None else continuation["after_event"],
            "catalog_event_id": None if bound_event is None else bound_event["catalog_event_id"],
            "event_arguments": [] if bound_event is None else list(bound_event["arguments"]),
            "continuation_queued": continuation is not None,
        }

    def observe(self, action: str, observation: str, *, native_valid: bool) -> dict:
        if self._phase not in ("awaiting_first_action", "awaiting_continuation"):
            raise ValueError("No pending actual action to observe")
        _bool(native_valid, "native_valid")
        _text(action, "actual action", limit=160)
        _text(observation, "actual observation", empty=True)
        first = self._phase == "awaiting_first_action"
        continuation = self._plan["continuation"]
        expected = self._plan["action"] if first else continuation["action"]
        if action != expected:
            raise ValueError("Actual action differs from the installed pending action")
        event_id = None if continuation is None else continuation["after_event"]
        catalog_event_id = None if self._event is None else self._event["catalog_event_id"]
        event_arguments = [] if self._event is None else list(self._event["arguments"])
        self._completed += 1
        if not native_valid:
            reason = "actor_action_not_native" if first else "continuation_action_not_native"
            self._clear(reason)
        elif not first:
            reason = "continuation_completed_actor_required"
            self._clear(reason)
        elif continuation is None:
            reason = "actor_action_observed_no_continuation"
            self._clear(reason)
        else:
            reason = "continuation_ready"
            self._phase, self._feedback = "ready_continuation", observation
        return {
            **self._audit(), "reason": reason, "action": action, "observation": observation,
            "native_valid": native_valid, "action_role": "actor" if first else "continuation",
            "event_id": event_id, "continuation_queued": self._plan is not None,
            "catalog_event_id": catalog_event_id,
            "event_arguments": event_arguments,
        }

    def decide(self, observation: str, native_actions: list[str]) -> dict:
        if self._phase not in ("empty", "ready_continuation"):
            raise ValueError("A pending plan/decision needs install or observe before another decide")
        _text(observation, "decision observation", empty=True)
        _strings(native_actions, "native actions")
        result = {
            "commit": False, "action": None, "candidate_action": None, "reason": "actor_required",
            "guard_checked": False, "guard_passed": None, "native_member": None,
            "event_id": None, "catalog_event_id": None, "match_type": None, "expected_feedback": None,
            "event_arguments": [],
            "event_checked": False, "event_matched": None, "novelty_checked": False,
            "novelty_passed": None, "novel_goal_entities": [],
        }
        if self._phase == "empty":
            self._phase = "awaiting_plan"
            result["reason"] = "actor_boundary_required" if self._boundary else "actor_required"
            return {**self._audit(), **result}
        if observation != self._feedback:
            raise ValueError("Decision observation differs from actual first-action feedback")
        continuation, event = self._plan["continuation"], self._event
        action = continuation["action"]
        novel = sorted(self._entities(observation) - self._pre_entities)
        result.update(
            candidate_action=action, native_member=action in native_actions,
            event_id=continuation["after_event"], catalog_event_id=event["catalog_event_id"],
            match_type=event["match_type"], expected_feedback=event["feedback"],
            event_arguments=list(event["arguments"]),
            novel_goal_entities=[{"type": kind, "instance": number} for kind, number in novel],
        )
        reason = None
        if not result["native_member"]:
            reason = "continuation_not_native"
        else:
            normalized = _normalize(observation)
            event_checked = self.mode != "event_unchecked"
            event_matched = (normalized == event["feedback"] if event["match_type"] == "exact_match"
                             else normalized.startswith(event["feedback"])) if event_checked else None
            novelty_checked = self.mode != "event_no_novelty"
            novelty_passed = not novel if novelty_checked else None
            result.update(
                guard_checked=True, guard_passed=event_matched is not False and novelty_passed is not False,
                event_checked=event_checked, event_matched=event_matched,
                novelty_checked=novelty_checked, novelty_passed=novelty_passed,
            )
            reason = ("feedback_event_mismatch" if event_matched is False else
                      "new_goal_information" if novelty_passed is False else None)
        if reason is not None:
            result["reason"] = reason
            self._clear(reason, awaiting_plan=True)
        else:
            self._phase = "awaiting_continuation"
            result.update(commit=True, action=action, reason="event_plan_ready")
        return {**self._audit(), **result}
