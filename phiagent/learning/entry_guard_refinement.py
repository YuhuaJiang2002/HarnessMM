"""Standalone TRAIN-only causal entry guards; no controller or runner wiring.

Public APIs:
* fit_entity_vocabulary(examples, *, excluded_groups=frozenset(), intervention_kind=None)
* extract_guard_features(vocabulary, public_state, program_actions,
  validated_action_history, *, task_family, representation="lifted")
* fit_entry_guards(examples, *, excluded_groups=frozenset(),
  minimum_harmful_groups=2, max_literals=2, max_rules=8, representation="lifted",
  intervention_kind=None)
* decide_entry_guard(model, public_state, program_actions, validated_action_history,
  *, task_family, group_id=None)
* decide_exclusion has the same arguments, but accepts only exclude_entry models.

Examples have exactly group_id, task_family, program_actions, public_state,
validated_action_history, evidence_label and intervention_kind. Labels are supplied by the parent
after complete repeated interventions, not inferred here from text changes.
Excluded groups and unresolved records supply neither vocabulary nor features,
candidates, conflicts or scores. Group IDs and labels remain audit metadata.

Intervention kinds are NOT interchangeable. exclude_entry removes one instance
and may leave other compiled alternatives; it does not establish a need for actor
reasoning. force_actor_once replaces the nominated entry by exactly one actor
call, then resumes the same frozen deployed continuation. Only evidence from
that intervention can produce a defer_to_actor_once directive. Mixed batch kinds
are rejected even for unresolved or excluded rows; this is a design/schema check,
not feature fitting. An optional fitting keyword pins
the intended kind, including for empty batches; it never relabels input evidence.
Kinds/directives are recorded on model, evidence and rule artifacts, not features.
For force_actor_once, defer_preferred_supported supplies intervention-support
evidence and compiled_route_preferred_supported supplies conflict evidence.
Original label strings are preserved. The legacy harmful/helpful label names
remain accepted with an explicit intervention kind; actor-specific labels are
never accepted as instance-exclusion evidence. One supportive group cannot
admit a rule, regardless of its number of repeated samples or observed gains.

The source-native vocabulary recognizes a word/type followed by a decimal
instance token in native commands. It infers no plurals, synonyms or ontology.
Lifted goal roles follow mention order, never semantic names. An unnumbered goal
type binds only to a unique program-selected instance of that type; repeated
unnumbered roles, unknown types and unbound program arguments abstain. Explicit
references preserve identity and repeated-role order. Non-entity goal/command
syntax and the role type-equality partition remain in the lifted shape.

Features include native affordance templates, selected/other-instance observation
references, program/goal identity relations and the LAST validated native action
mentioning each program argument. Public history/instructions never supply
validated-action history. Missing native-template keys mean absent affordances;
other features are explicit, including null when no mentioning action exists.

Literal representation is a separate, explicit fit, keyed by exact task text and
literal program actions. Lifted extraction never falls back to literal matching.
At most 32 source-group-ranked literals per scope supply conjunctions of arity
one or two. Rules require >=2 harmful groups and zero helpful-group conflicts;
max_rules is a prospective global cap. Scores/coverage count groups, not prefixes.
Every discarded source, literal, candidate or scope has an audit reason.
These are empirical candidate guards, NOT safety certificates or goal guarantees.
A directive=none result is abstention/non-match, not a safe-entry label. The
learner never invokes an actor, blacklists an instance, or enforces continuation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import re

from phiagent.learning.counterfactual_invocation import validate_public_state


VOCABULARY_SCHEMA = "chisel-source-native-entity-vocabulary/v2"
FEATURE_SCHEMA = "chisel-observable-entry-features/v1"
MODEL_SCHEMA = "chisel-causal-entry-exclusion-guards/v2"
INTERVENTION_KINDS = ("exclude_entry", "force_actor_once")
_DIRECTIVES = {"exclude_entry": "exclude_entry", "force_actor_once": "defer_to_actor_once"}
_SYNTAX = "word_type_followed_by_exact_decimal_instance_token"
_HARMFUL = "harmful_entry_supported"
_HELPFUL = "helpful_entry_supported"
_UNRESOLVED = "unresolved"
_INTERVENTION_PREFERRED = "intervention_preferred"
_COMPILED_PREFERRED = "compiled_preferred"
_LABEL_ROLES = {
    "exclude_entry": {
        _HARMFUL: _INTERVENTION_PREFERRED, _HELPFUL: _COMPILED_PREFERRED,
        _UNRESOLVED: _UNRESOLVED,
    },
    "force_actor_once": {
        "defer_preferred_supported": _INTERVENTION_PREFERRED,
        "compiled_route_preferred_supported": _COMPILED_PREFERRED,
        _HARMFUL: _INTERVENTION_PREFERRED, _HELPFUL: _COMPILED_PREFERRED,
        _UNRESOLVED: _UNRESOLVED,
    },
}
_REPRESENTATIONS = ("lifted", "literal")
_LITERAL_LIMIT = 32
_RULE_LIMIT = 32
_NATIVE_PREFIX = "native.template:"
_TOKENS = re.compile(r"\w+(?:-\w+)*|[^\w\s]")
_TYPE_WORD = re.compile(r"[^\W\d]\w*(?:-\w+)*")
_INSTANCE = re.compile(r"[0-9]+")
_OBJECTIVE = (
    "greedy new harmful-group coverage within family/shape; ties: total harmful groups, "
    "fewer literals, lexical family/shape/literals; no sample-count weighting"
)
_EXAMPLE_KEYS = {
    "group_id", "task_family", "program_actions", "public_state",
    "validated_action_history", "evidence_label", "intervention_kind",
}
Entity = tuple[str, str]


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else ''}string")
    return value


def _integer(value: object, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid integer {name}")
    return value


def _record(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Invalid {name} schema")
    return value


def _strings(value: object, name: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    for item in value:
        _text(item, name)
    if unique and len(set(value)) != len(value):
        raise ValueError(f"Duplicate {name}")
    return value


def _groups(value: object) -> frozenset[str]:
    if not isinstance(value, (set, frozenset)):
        raise ValueError("excluded_groups must be a set of group IDs")
    for item in value:
        _text(item, "excluded group")
    return frozenset(value)


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected finite JSON-serializable data") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _representation(value: object) -> str:
    if not isinstance(value, str) or value not in _REPRESENTATIONS:
        raise ValueError("representation must be lifted or literal")
    return value


def _intervention_kind(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or value not in INTERVENTION_KINDS:
        raise ValueError("intervention_kind must be exclude_entry or force_actor_once")
    return value


def _evidence_role(label: object, kind: str | None) -> str:
    if not isinstance(label, str) or kind not in _LABEL_ROLES or label not in _LABEL_ROLES[kind]:
        raise ValueError("Unknown or incompatible evidence_label for intervention_kind")
    return _LABEL_ROLES[kind][label]


def _role_groups(rows: list[dict], role: str) -> list[str]:
    return sorted({row["group_id"] for row in rows
                   if _evidence_role(row["evidence_label"], row["intervention_kind"]) == role})


def _query(public_state: dict, program_actions: list[str],
           validated_action_history: list[str], task_family: str) -> None:
    validate_public_state(public_state)
    _text(task_family, "task_family")
    _strings(program_actions, "program_actions")
    if not 2 <= len(program_actions) <= 4:
        raise ValueError("Entry programs require two to four literal native actions")
    _strings(validated_action_history, "validated_action_history")


def _filtered(examples: list[dict], excluded: frozenset[str],
              intervention_kind: str | None) -> tuple[list[dict], dict, list[dict], str | None]:
    if not isinstance(examples, list):
        raise ValueError("examples must be a list")
    kind = _intervention_kind(intervention_kind, optional=True)
    counts = {
        "input_examples": len(examples), "excluded_examples": 0,
        "unresolved_examples": 0, "duplicate_examples": 0, "retained_examples": 0,
    }
    kept, rejected = {}, []
    for example in examples:
        _record(example, _EXAMPLE_KEYS, "causal entry example")
        _text(example["group_id"], "group_id")
        _query(example["public_state"], example["program_actions"],
               example["validated_action_history"], example["task_family"])
        source_kind = _intervention_kind(example["intervention_kind"])
        if kind is None:
            kind = source_kind
        elif kind != source_kind:
            raise ValueError("Incompatible mixed intervention_kind values; fit each intervention separately")
        _evidence_role(example["evidence_label"], source_kind)
        canonical = {**example, "public_state": {
            **example["public_state"], "actions": sorted(example["public_state"]["actions"]),
        }}
        source_id = _digest(canonical)
        reason = None
        if example["group_id"] in excluded:
            counts["excluded_examples"] += 1
            reason = "excluded_group"
        elif example["evidence_label"] == _UNRESOLVED:
            counts["unresolved_examples"] += 1
            reason = "unresolved_not_used_as_evidence"
        elif source_id in kept:
            counts["duplicate_examples"] += 1
            reason = "duplicate_source_example"
        if reason is not None:
            rejected.append({"stage": "source", "source_id": source_id,
                             "group_id": example["group_id"], "reason": reason,
                             "intervention_kind": source_kind})
        else:
            kept[source_id] = {"source_id": source_id, **deepcopy(canonical)}
    counts["retained_examples"] = len(kept)
    rows = [kept[key] for key in sorted(kept)]
    counts["intervention_support_group_count"] = len(_role_groups(rows, _INTERVENTION_PREFERRED))
    counts["compiled_preferred_group_count"] = len(_role_groups(rows, _COMPILED_PREFERRED))
    return rows, counts, sorted(rejected, key=_json), kind


def _references(text: str) -> tuple[list[str], dict[int, Entity]]:
    tokens, refs = _TOKENS.findall(text), {}
    index = 0
    while index + 1 < len(tokens):
        if _TYPE_WORD.fullmatch(tokens[index]) and _INSTANCE.fullmatch(tokens[index + 1]):
            refs[index] = tokens[index].casefold(), tokens[index + 1]
            index += 2
        else:
            index += 1
    return tokens, refs


def _native_commands(row: dict) -> list[str]:
    return (row["program_actions"] + row["public_state"]["actions"]
            + row["validated_action_history"])


def _build_vocabulary(rows: list[dict], excluded: frozenset[str],
                      counts: dict, rejections: list[dict], intervention_kind: str | None) -> dict:
    types: dict[str, dict] = {}
    commands = set()
    for row in rows:
        for command in set(_native_commands(row)):
            commands.add((row["group_id"], command))
            for kind, _ in _references(command)[1].values():
                info = types.setdefault(kind, {"groups": set(), "sources": set(), "commands": set()})
                info["groups"].add(row["group_id"])
                info["sources"].add(row["source_id"])
                info["commands"].add((row["group_id"], command))
    return {
        "schema_version": VOCABULARY_SCHEMA, "syntax": _SYNTAX,
        "intervention_kind": intervention_kind,
        "types": [{
            "name": name, "source_groups": sorted(info["groups"]),
            "source_ids": sorted(info["sources"]), "native_command_count": len(info["commands"]),
        } for name, info in sorted(types.items())],
        "excluded_groups": sorted(excluded),
        "fit_groups": sorted({row["group_id"] for row in rows}),
        "counts": {**counts, "native_commands": len(commands), "entity_types": len(types)},
        "rejections": deepcopy(rejections),
    }


def fit_entity_vocabulary(examples: list[dict], *,
                          excluded_groups: set[str] | frozenset[str] = frozenset(),
                          intervention_kind: str | None = None) -> dict:
    """Record native vocabulary from one explicitly identified intervention design."""
    excluded = _groups(excluded_groups)
    rows, counts, rejected, kind = _filtered(examples, excluded, intervention_kind)
    vocabulary = _build_vocabulary(rows, excluded, counts, rejected, kind)
    _validate_vocabulary(vocabulary)
    return vocabulary


def _validate_counts(counts: object) -> None:
    if not isinstance(counts, dict):
        raise ValueError("counts must be a mapping")
    for name, value in counts.items():
        _text(name, "count name")
        _integer(value, name)


def _validate_vocabulary(vocabulary: dict) -> set[str]:
    _record(vocabulary, {
        "schema_version", "syntax", "types", "excluded_groups", "fit_groups", "counts", "rejections",
        "intervention_kind",
    }, "entity vocabulary")
    if vocabulary["schema_version"] != VOCABULARY_SCHEMA or vocabulary["syntax"] != _SYNTAX:
        raise ValueError("Unknown entity vocabulary schema/syntax")
    excluded = set(_strings(vocabulary["excluded_groups"], "excluded groups", unique=True))
    fit = set(_strings(vocabulary["fit_groups"], "vocabulary fit groups", unique=True))
    if fit & excluded:
        raise ValueError("Excluded groups entered the vocabulary")
    _validate_counts(vocabulary["counts"])
    kind = _intervention_kind(vocabulary["intervention_kind"], optional=True)
    input_count = _integer(vocabulary["counts"].get("input_examples"), "input_examples")
    if kind is None and (input_count or vocabulary["types"] or fit):
        raise ValueError("Nonempty vocabulary evidence requires an explicit intervention kind")
    if not isinstance(vocabulary["types"], list) or not isinstance(vocabulary["rejections"], list):
        raise ValueError("Vocabulary types/rejections must be lists")
    names = set()
    for row in vocabulary["types"]:
        _record(row, {"name", "source_groups", "source_ids", "native_command_count"}, "entity type")
        name = _text(row["name"], "entity type name")
        if name != name.casefold() or not _TYPE_WORD.fullmatch(name) or name in names:
            raise ValueError("Entity types must be unique normalized native words")
        groups = set(_strings(row["source_groups"], "type source groups", unique=True))
        sources = _strings(row["source_ids"], "type source IDs", unique=True)
        if not groups or not groups <= fit or not sources:
            raise ValueError("Entity vocabulary requires non-excluded native provenance")
        _integer(row["native_command_count"], "native_command_count", minimum=len(groups))
        names.add(name)
    _validate_rejections(vocabulary["rejections"], kind)
    _json(vocabulary)
    return names


def _validate_rejections(rejections: list[dict], kind: str | None) -> None:
    for rejection in rejections:
        if not isinstance(rejection, dict) or not {
            "stage", "reason", "intervention_kind",
        } <= rejection.keys():
            raise ValueError("Rejections must identify their intervention")
        _text(rejection["stage"], "rejection stage")
        _text(rejection["reason"], "rejection reason")
        if _intervention_kind(rejection["intervention_kind"]) != kind:
            raise ValueError("Rejection artifacts mix intervention meanings")


def _template(text: str, role_entities: list[Entity]) -> list:
    tokens, references = _references(text)
    roles = {entity: index for index, entity in enumerate(role_entities)}
    others, other_types, result = {}, {}, []
    index = 0
    while index < len(tokens):
        entity = references.get(index)
        if entity is None:
            result.append(tokens[index])
            index += 1
            continue
        if entity in roles:
            result.append({"role": roles[entity]})
        else:
            other_id = others.setdefault(entity, len(others))
            same_type = [role for role, bound in enumerate(role_entities) if bound[0] == entity[0]]
            result.append(
                {"other_of_roles": same_type, "entity": other_id} if same_type else
                {"other_type": other_types.setdefault(entity[0], len(other_types)), "entity": other_id}
            )
        index += 2
    return result


def _bind_goal(task: str, program_entities: list[Entity], native_entities: set[Entity],
               vocabulary: set[str]) -> tuple[list[Entity], list, str | None]:
    tokens = _TOKENS.findall(task)
    mentions, index = [], 0
    while index < len(tokens):
        kind = tokens[index].casefold()
        if kind in vocabulary:
            instance = (tokens[index + 1] if index + 1 < len(tokens)
                        and _INSTANCE.fullmatch(tokens[index + 1]) else None)
            mentions.append((index, kind, instance))
            index += 2 if instance is not None else 1
        else:
            index += 1
    if not mentions:
        return [], [], "unbound_goal_roles"
    frequency = Counter(kind for _, kind, _ in mentions)
    role_entities, positions = [], {}
    for position, kind, instance in mentions:
        if instance is None:
            if frequency[kind] != 1:
                return [], [], "ambiguous_unnumbered_goal_roles"
            choices = [entity for entity in program_entities if entity[0] == kind]
            if len(choices) != 1:
                return [], [], ("ambiguous_program_instances_for_goal_role" if choices
                                else "unbound_goal_role")
            entity = choices[0]
        else:
            entity = kind, instance
            if entity not in native_entities:
                return [], [], "goal_reference_not_native"
        if entity not in role_entities:
            role_entities.append(entity)
        positions[position] = role_entities.index(entity), 2 if instance is not None else 1
    if any(entity not in role_entities for entity in program_entities):
        return [], [], "unbound_program_argument"
    goal, index = [], 0
    while index < len(tokens):
        if index in positions:
            role, width = positions[index]
            goal.append({"role": role})
            index += width
        else:
            goal.append(tokens[index])
            index += 1
    return role_entities, goal, None


def extract_guard_features(vocabulary: dict, public_state: dict, program_actions: list[str],
                           validated_action_history: list[str], *, task_family: str,
                           representation: str = "lifted") -> dict:
    """Extract only observable features; ambiguous lifting returns status=abstain.

    Goal/program identity and type-equality relations are positional. Context
    templates mark other instances explicitly, never bind them to a selected
    role merely because they share its type. No label/group/task ID argument is
    accepted. Literal extraction is opt-in and never a fallback from a failed lift.
    """
    known = _validate_vocabulary(vocabulary)
    _query(public_state, program_actions, validated_action_history, task_family)
    _representation(representation)
    result = {
        "schema_version": FEATURE_SCHEMA, "representation": representation,
        "task_family": task_family, "status": "abstain", "reason": None,
        "shape": None, "shape_id": None, "features": {}, "bindings": [],
    }
    native = set(public_state["actions"])
    features = {f"program.native.{index}": action in native
                for index, action in enumerate(program_actions)}
    if representation == "literal":
        shape = {"representation": "literal", "goal": public_state["task"],
                 "program": list(program_actions)}
        features.update({_NATIVE_PREFIX + _json(action): True for action in native})
        features["observation.literal"] = public_state["observation"]
        features["history.last_native"] = (
            validated_action_history[-1] if validated_action_history else None
        )
        features.update({f"program.seen.{index}": action in validated_action_history
                         for index, action in enumerate(program_actions)})
    else:
        commands = program_actions + public_state["actions"] + validated_action_history
        native_entities = {entity for action in commands for entity in _references(action)[1].values()}
        if {kind for kind, _ in native_entities} - known:
            return {**result, "reason": "unknown_native_entity_type"}
        if {kind for kind, _ in _references(public_state["task"])[1].values()} - known:
            return {**result, "reason": "unknown_goal_entity_type"}
        arguments = list(dict.fromkeys(
            entity for action in program_actions for entity in _references(action)[1].values()
        ))
        if not arguments:
            return {**result, "reason": "no_program_entity_references"}
        roles, goal, reason = _bind_goal(public_state["task"], arguments, native_entities, known)
        if reason is not None:
            return {**result, "reason": reason}
        kinds = {}
        partition = [kinds.setdefault(entity[0], len(kinds)) for entity in roles]
        shape = {
            "representation": "lifted", "goal": goal, "role_type_partition": partition,
            "program": [_template(action, roles) for action in program_actions],
            "arguments": [roles.index(entity) for entity in arguments],
        }
        features.update({_NATIVE_PREFIX + _json(_template(action, roles)): True for action in native})
        observed = {entity for entity in _references(public_state["observation"])[1].values()
                    if entity[0] in known}
        parsed_history = [(action, set(_references(action)[1].values()))
                          for action in validated_action_history]
        for index, entity in enumerate(arguments):
            features[f"program.arg.{index}.goal_role"] = roles.index(entity)
            features[f"observation.arg.{index}.selected"] = entity in observed
            features[f"observation.arg.{index}.other_same_type"] = any(
                other[0] == entity[0] and other != entity for other in observed
            )
            last = next((action for action, refs in reversed(parsed_history) if entity in refs), None)
            features[f"history.arg.{index}.last"] = None if last is None else _json(_template(last, roles))
            for step, action in enumerate(program_actions):
                features[f"history.arg.{index}.last_is_step.{step}"] = last == action
            for other in range(index + 1, len(arguments)):
                features[f"program.arg.{index}.same_type.{other}"] = entity[0] == arguments[other][0]
        result["bindings"] = [
            {"role": index, "type": entity[0], "instance": entity[1],
             "program_argument": arguments.index(entity) if entity in arguments else None}
            for index, entity in enumerate(roles)
        ]
    _validate_shape(shape, representation)
    _validate_features(features, shape)
    return {
        **result, "status": "ok", "reason": "bound_native_roles" if representation == "lifted"
        else "explicit_literal_scope", "shape": shape, "shape_id": _digest(shape),
        "features": dict(sorted(features.items())),
    }


def _fixed_feature_types(shape: dict) -> dict[str, tuple[type, ...]]:
    types = {f"program.native.{index}": (bool,) for index in range(len(shape["program"]))}
    if shape["representation"] == "literal":
        types.update({f"program.seen.{index}": (bool,) for index in range(len(shape["program"]))})
        return {**types, "observation.literal": (str,), "history.last_native": (str, type(None))}
    for index in range(len(shape["arguments"])):
        types[f"program.arg.{index}.goal_role"] = (int,)
        types[f"observation.arg.{index}.selected"] = (bool,)
        types[f"observation.arg.{index}.other_same_type"] = (bool,)
        types[f"history.arg.{index}.last"] = (str, type(None))
        for step in range(len(shape["program"])):
            types[f"history.arg.{index}.last_is_step.{step}"] = (bool,)
        for other in range(index + 1, len(shape["arguments"])):
            types[f"program.arg.{index}.same_type.{other}"] = (bool,)
    return types


def _validate_template(template: object, role_count: int, *, background: bool = False) -> None:
    if not isinstance(template, list) or not template:
        raise ValueError("A native/goal template must be a nonempty token list")
    for token in template:
        if isinstance(token, str):
            _text(token, "literal syntax token")
        elif isinstance(token, dict) and set(token) == {"role"}:
            _integer(token["role"], "goal role", maximum=role_count - 1)
        elif background and isinstance(token, dict) and set(token) == {"other_of_roles", "entity"}:
            roles = token["other_of_roles"]
            if not isinstance(roles, list) or not roles:
                raise ValueError("Other-instance relations require explicit type-related roles")
            for role in roles:
                _integer(role, "related goal role", maximum=role_count - 1)
            if len(set(roles)) != len(roles):
                raise ValueError("Duplicate related goal role")
            _integer(token["entity"], "local other-instance identity")
        elif background and isinstance(token, dict) and set(token) == {"other_type", "entity"}:
            _integer(token["other_type"], "local other-type identity")
            _integer(token["entity"], "local other-instance identity")
        else:
            raise ValueError("Unknown template token or invented role")


def _validate_shape(shape: dict, representation: str) -> None:
    keys = {"representation", "goal", "program"}
    if representation == "lifted":
        keys |= {"role_type_partition", "arguments"}
    _record(shape, keys, "program shape")
    if shape["representation"] != representation:
        raise ValueError("Mixed literal/lifted shape")
    program = shape["program"]
    if not isinstance(program, list) or not 2 <= len(program) <= 4:
        raise ValueError("Invalid shape program length")
    if representation == "literal":
        _text(shape["goal"], "literal goal")
        _strings(program, "literal program")
        return
    partition = shape["role_type_partition"]
    if not isinstance(partition, list) or not partition:
        raise ValueError("Lifted shapes require bound goal roles")
    seen_types = set()
    for item in partition:
        _integer(item, "role type class")
        if item not in seen_types and item != len(seen_types):
            raise ValueError("Type partitions must preserve first-occurrence order")
        seen_types.add(item)
    _validate_template(shape["goal"], len(partition))
    goal_order = list(dict.fromkeys(token["role"] for token in shape["goal"] if isinstance(token, dict)))
    if goal_order != list(range(len(partition))):
        raise ValueError("Goal roles must preserve positional order")
    for template in program:
        _validate_template(template, len(partition))
    argument_order = list(dict.fromkeys(
        token["role"] for template in program for token in template if isinstance(token, dict)
    ))
    if not argument_order or not isinstance(shape["arguments"], list):
        raise ValueError("Lifted programs require bound arguments")
    for argument in shape["arguments"]:
        _integer(argument, "program argument role", maximum=len(partition) - 1)
    if argument_order != shape["arguments"]:
        raise ValueError("Program argument identity/order disagrees with its shape")


def _feature_type(key: str, shape: dict) -> tuple[type, ...]:
    fixed = _fixed_feature_types(shape)
    if key in fixed:
        return fixed[key]
    if not key.startswith(_NATIVE_PREFIX):
        raise ValueError("Unknown or non-observable feature")
    try:
        template = json.loads(key[len(_NATIVE_PREFIX):])
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid native-affordance template feature") from error
    if shape["representation"] == "literal":
        _text(template, "literal native affordance")
    else:
        _validate_template(template, len(shape["role_type_partition"]), background=True)
    return (bool,)


def _validate_feature_value(key: str, value: object, shape: dict) -> None:
    if type(value) not in _feature_type(key, shape):
        raise ValueError("Observable feature has an invalid scalar type")
    if key.startswith("program.arg.") and key.endswith(".goal_role"):
        _integer(value, "feature goal role", maximum=len(shape["role_type_partition"]) - 1)
        argument = int(key.split(".")[2])
        if value != shape["arguments"][argument]:
            raise ValueError("Program/goal identity feature contradicts its bound shape")
    if key.startswith("program.arg.") and ".same_type." in key:
        parts = key.split(".")
        first, second = int(parts[2]), int(parts[4])
        partition, arguments = shape["role_type_partition"], shape["arguments"]
        if value != (partition[arguments[first]] == partition[arguments[second]]):
            raise ValueError("Type-identity feature contradicts its bound shape")
    if key == "history.last_native" and value is not None:
        _text(value, "last native action")
    if key.startswith("history.arg.") and key.endswith(".last") and value is not None:
        try:
            template = json.loads(value)
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid last-native-action template") from error
        _validate_template(template, len(shape["role_type_partition"]), background=True)


def _validate_features(features: dict, shape: dict) -> None:
    if not isinstance(features, dict) or not _fixed_feature_types(shape).keys() <= features.keys():
        raise ValueError("Missing observable feature fields")
    for key, value in features.items():
        _text(key, "feature name")
        _validate_feature_value(key, value, shape)


def _feature_value(features: dict, key: str):
    if key in features:
        return features[key]
    if key.startswith(_NATIVE_PREFIX):
        return False
    raise ValueError("Missing required observable feature")


def _matches(features: dict, literals: list[dict]) -> bool:
    return all(type(_feature_value(features, item["feature"])) is type(item["value"])
               and _feature_value(features, item["feature"]) == item["value"] for item in literals)


def _candidate(rows: list[dict], literals: list[dict], family: str, shape_id: str,
               intervention_kind: str) -> dict:
    harmful = [row for row in rows
               if _evidence_role(row["evidence_label"], intervention_kind) == _INTERVENTION_PREFERRED
               and _matches(row["features"], literals)]
    helpful = [row for row in rows
               if _evidence_role(row["evidence_label"], intervention_kind) == _COMPILED_PREFERRED
               and _matches(row["features"], literals)]
    groups = sorted({row["group_id"] for row in harmful})
    conflicts = sorted({row["group_id"] for row in helpful})
    identity = {
        "task_family": family, "shape_id": shape_id, "literals": literals,
        "intervention_kind": intervention_kind, "directive": _DIRECTIVES[intervention_kind],
    }
    return {
        "rule_id": _digest(identity), **identity,
        "harmful_groups": groups, "helpful_conflict_groups": conflicts,
        "support_source_ids": sorted(row["source_id"] for row in harmful),
        "counts": {
            "harmful_group_count": len(groups), "harmful_example_count": len(harmful),
            "helpful_group_count": len(conflicts), "helpful_example_count": len(helpful),
        },
    }


def fit_entry_guards(examples: list[dict], *,
                     excluded_groups: set[str] | frozenset[str] = frozenset(),
                     minimum_harmful_groups: int = 2, max_literals: int = 2,
                     max_rules: int = 8, representation: str = "lifted",
                     intervention_kind: str | None = None) -> dict:
    """Fit one intervention meaning; insufficient independent support never forces a rule.

    harmful_groups / minimum_harmful_groups retain their API names and mean
    groups supporting the measured intervention. Under force_actor_once these
    are defer-preferred groups, not individual gains or repeated trials.
    """
    _integer(minimum_harmful_groups, "minimum_harmful_groups", 2)
    _integer(max_literals, "max_literals", 1, 2)
    _integer(max_rules, "max_rules", 1, _RULE_LIMIT)
    _representation(representation)
    excluded = _groups(excluded_groups)
    rows, counts, rejected, kind = _filtered(examples, excluded, intervention_kind)
    vocabulary = _build_vocabulary(rows, excluded, counts, rejected, kind)
    evidence, shapes = [], {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        extracted = extract_guard_features(
            vocabulary, row["public_state"], row["program_actions"], row["validated_action_history"],
            task_family=row["task_family"], representation=representation,
        )
        if extracted["status"] != "ok":
            rejected.append({
                "stage": "features", "reason": extracted["reason"],
                "source_id": row["source_id"], "group_id": row["group_id"], "intervention_kind": kind,
            })
            continue
        item = {key: row[key] for key in (
            "source_id", "group_id", "task_family", "evidence_label", "intervention_kind",
        )}
        item.update(shape_id=extracted["shape_id"], features=extracted["features"])
        evidence.append(item)
        scope = row["task_family"], extracted["shape_id"]
        grouped[scope].append(item)
        shapes[scope] = extracted["shape"]
    admitted_candidates, scope_results = [], []
    for (family, shape_id), members in sorted(grouped.items()):
        if kind is None:
            raise ValueError("Fitted evidence must identify its intervention")
        harmful_groups = _role_groups(members, _INTERVENTION_PREFERRED)
        helpful_groups = _role_groups(members, _COMPILED_PREFERRED)
        scope = {
            "task_family": family, "shape_id": shape_id, "shape": shapes[family, shape_id],
            "intervention_kind": kind,
            "harmful_groups": harmful_groups, "helpful_groups": helpful_groups,
            "status": "no_admitted_rule", "reason": None,
            "counts": {"examples": len(members), "candidate_literals": 0,
                       "incompatible_literal_pairs": 0, "tested_candidates": 0,
                       "admissible_candidates": 0, "admitted_rules": 0},
        }
        scope_results.append(scope)
        if len(harmful_groups) < minimum_harmful_groups:
            scope["reason"] = "insufficient_independent_harmful_groups"
            rejected.append({"stage": "scope", "reason": scope["reason"],
                             "task_family": family, "shape_id": shape_id,
                             "harmful_groups": harmful_groups, "intervention_kind": kind})
            continue
        feature_names = sorted({name for row in members for name in row["features"]})
        atoms = {
            _json({"feature": name, "value": _feature_value(row["features"], name)}):
            {"feature": name, "value": _feature_value(row["features"], name)}
            for row in members
            if _evidence_role(row["evidence_label"], kind) == _INTERVENTION_PREFERRED
            for name in feature_names
        }
        scope["counts"]["candidate_literals"] = len(atoms)
        eligible_atoms = []
        for atom in (atoms[key] for key in sorted(atoms)):
            candidate = _candidate(members, [atom], family, shape_id, kind)
            if len(candidate["harmful_groups"]) < minimum_harmful_groups:
                rejected.append({"stage": "literal", "reason": "insufficient_harmful_groups", **candidate})
            else:
                eligible_atoms.append(candidate)
        eligible_atoms.sort(key=lambda row: (
            -len(row["harmful_groups"]), len(row["helpful_conflict_groups"]), _json(row["literals"]),
        ))
        for candidate in eligible_atoms[_LITERAL_LIMIT:]:
            rejected.append({"stage": "literal", "reason": "candidate_literal_cap", **candidate})
        atoms = [row["literals"][0] for row in eligible_atoms[:_LITERAL_LIMIT]]
        clauses = [[atom] for atom in atoms]
        if max_literals == 2:
            for pair in combinations(atoms, 2):
                clause = sorted(pair, key=_json)
                if pair[0]["feature"] == pair[1]["feature"]:
                    scope["counts"]["incompatible_literal_pairs"] += 1
                    rejected.append({
                        "stage": "construction", "reason": "contradictory_feature_values",
                        "task_family": family, "shape_id": shape_id, "literals": clause,
                        "intervention_kind": kind,
                    })
                else:
                    clauses.append(clause)
        for clause in clauses:
            candidate = _candidate(members, sorted(clause, key=_json), family, shape_id, kind)
            scope["counts"]["tested_candidates"] += 1
            if len(candidate["harmful_groups"]) < minimum_harmful_groups:
                reason = "insufficient_harmful_groups"
            elif candidate["helpful_conflict_groups"]:
                reason = "helpful_group_conflict"
            else:
                admitted_candidates.append(candidate)
                scope["counts"]["admissible_candidates"] += 1
                continue
            rejected.append({"stage": "candidate", "reason": reason, **candidate})
    rules, covered = [], defaultdict(set)
    while admitted_candidates:
        def rank(row):
            scope = row["task_family"], row["shape_id"]
            return (
                -len(set(row["harmful_groups"]) - covered[scope]), -len(row["harmful_groups"]),
                len(row["literals"]), row["task_family"], row["shape_id"], _json(row["literals"]),
            )
        best = min(admitted_candidates, key=rank)
        admitted_candidates.remove(best)
        scope = best["task_family"], best["shape_id"]
        if not set(best["harmful_groups"]) - covered[scope]:
            rejected.append({"stage": "selection", "reason": "redundant_group_coverage", **best})
        elif len(rules) >= max_rules:
            rejected.append({"stage": "selection", "reason": "prospective_rule_cap", **best})
        else:
            rules.append(best)
            covered[scope].update(best["harmful_groups"])
    for scope in scope_results:
        scope["counts"]["admitted_rules"] = sum(
            row["task_family"] == scope["task_family"] and row["shape_id"] == scope["shape_id"]
            for row in rules
        )
        if scope["counts"]["admitted_rules"]:
            scope.update(status="admitted_rules", reason="group_supported_zero_helpful_conflicts")
        elif scope["reason"] is None:
            scope["reason"] = ("prospective_rule_cap" if scope["counts"]["admissible_candidates"]
                               else "no_conflict_free_supported_conjunction")
            rejected.append({"stage": "scope", "reason": scope["reason"],
                             "task_family": scope["task_family"], "shape_id": scope["shape_id"],
                             "intervention_kind": kind})
    all_unresolved = (
        counts["unresolved_examples"] > 0
        and counts["unresolved_examples"] == counts["input_examples"] - counts["excluded_examples"]
    )
    reason = ("group_supported_empirical_candidates" if rules else
              "all_evidence_unresolved" if all_unresolved else
              "no_resolved_training_evidence" if not rows else
              "insufficient_independent_intervention_groups"
              if counts["intervention_support_group_count"] < minimum_harmful_groups else
              "no_extractable_training_evidence" if not evidence else "no_admissible_candidate")
    model = {
        "schema_version": MODEL_SCHEMA, "feature_schema_version": FEATURE_SCHEMA,
        "intervention_kind": kind,
        "representation": representation, "vocabulary": vocabulary,
        "parameters": {"minimum_harmful_groups": minimum_harmful_groups, "max_literals": max_literals,
                       "max_rules": max_rules, "candidate_literal_limit": _LITERAL_LIMIT},
        "excluded_groups": sorted(excluded), "fit_groups": vocabulary["fit_groups"],
        "objective": _OBJECTIVE, "evidence": sorted(evidence, key=lambda row: row["source_id"]),
        "scopes": scope_results, "rules": sorted(rules, key=lambda row: row["rule_id"]),
        "counts": {**counts, "feature_examples": len(evidence), "scopes": len(scope_results),
                   "admitted_rules": len(rules)},
        "rejections": sorted(rejected, key=_json),
        "status": "admitted_rules" if rules else "no_admitted_rule", "reason": reason,
        "semantics": (
            "Empirical TRAIN intervention-specific candidates; no safety certificate, "
            "reasoning-necessity claim or goal guarantee"
        ),
    }
    _validate_model(model)
    return json.loads(_json(model))


def _validate_model(model: dict) -> dict[tuple[str, str], dict]:
    _record(model, {
        "schema_version", "feature_schema_version", "representation", "vocabulary", "parameters",
        "excluded_groups", "fit_groups", "objective", "evidence", "scopes", "rules", "counts",
        "rejections", "status", "reason", "semantics", "intervention_kind",
    }, "entry guard model")
    if model["schema_version"] != MODEL_SCHEMA or model["feature_schema_version"] != FEATURE_SCHEMA:
        raise ValueError("Unknown entry guard model schema")
    representation = _representation(model["representation"])
    _validate_vocabulary(model["vocabulary"])
    kind = _intervention_kind(model["intervention_kind"], optional=True)
    if model["vocabulary"]["intervention_kind"] != kind:
        raise ValueError("Vocabulary/model intervention meanings disagree")
    params = _record(model["parameters"], {
        "minimum_harmful_groups", "max_literals", "max_rules", "candidate_literal_limit",
    }, "guard parameters")
    minimum = _integer(params["minimum_harmful_groups"], "minimum_harmful_groups", 2)
    arity = _integer(params["max_literals"], "max_literals", 1, 2)
    budget = _integer(params["max_rules"], "max_rules", 1, _RULE_LIMIT)
    if _integer(params["candidate_literal_limit"], "candidate_literal_limit") != _LITERAL_LIMIT:
        raise ValueError("Unknown prospective candidate-literal cap")
    excluded = set(_strings(model["excluded_groups"], "excluded groups", unique=True))
    fit = set(_strings(model["fit_groups"], "fit groups", unique=True))
    if (fit & excluded or fit != set(model["vocabulary"]["fit_groups"])
            or excluded != set(model["vocabulary"]["excluded_groups"])):
        raise ValueError("Guard/vocabulary group exclusions disagree")
    for field in ("objective", "reason", "semantics"):
        _text(model[field], field)
    for field in ("scopes", "evidence", "rules", "rejections"):
        if not isinstance(model[field], list):
            raise ValueError(f"{field} must be a list")
    _validate_counts(model["counts"])
    if _integer(model["counts"].get("input_examples"), "input_examples") and kind is None:
        raise ValueError("Nonempty evidence must identify its intervention")
    _validate_rejections(model["rejections"], kind)
    shapes = {}
    for scope in model["scopes"]:
        _record(scope, {"task_family", "shape_id", "shape", "harmful_groups", "helpful_groups",
                        "status", "reason", "counts", "intervention_kind"}, "guard scope")
        if _intervention_kind(scope["intervention_kind"]) != kind:
            raise ValueError("Scope intervention meaning disagrees with model")
        family = _text(scope["task_family"], "scope family")
        _validate_shape(scope["shape"], representation)
        key = family, _text(scope["shape_id"], "shape ID")
        if key in shapes or key[1] != _digest(scope["shape"]):
            raise ValueError("Duplicate or inconsistent program shape")
        shapes[key] = scope["shape"]
        _validate_counts(scope["counts"])
        for field in ("harmful_groups", "helpful_groups"):
            if not set(_strings(scope[field], field, unique=True)) <= fit:
                raise ValueError("Scope contains unfitted source groups")
        if scope["status"] not in ("admitted_rules", "no_admitted_rule"):
            raise ValueError("Invalid scope status")
        _text(scope["reason"], "scope reason")
    grouped, seen = defaultdict(list), set()
    for source in model["evidence"]:
        _record(source, {
            "source_id", "group_id", "task_family", "evidence_label", "shape_id", "features",
            "intervention_kind",
        },
                "fitted evidence")
        if _intervention_kind(source["intervention_kind"]) != kind:
            raise ValueError("Fitted evidence mixes incompatible intervention meanings")
        identifier = _text(source["source_id"], "source ID")
        if identifier in seen or not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise ValueError("Invalid or duplicate fitted source ID")
        seen.add(identifier)
        group = _text(source["group_id"], "evidence group")
        if group not in fit or _evidence_role(source["evidence_label"], kind) == _UNRESOLVED:
            raise ValueError("Unresolved/excluded/unfitted labels cannot supply rule evidence")
        key = _text(source["task_family"], "evidence family"), _text(source["shape_id"], "shape ID")
        if key not in shapes:
            raise ValueError("Evidence has no validated scope")
        _validate_features(source["features"], shapes[key])
        grouped[key].append(source)
    for scope in model["scopes"]:
        members = grouped[scope["task_family"], scope["shape_id"]]
        for field, role in (
            ("harmful_groups", _INTERVENTION_PREFERRED), ("helpful_groups", _COMPILED_PREFERRED),
        ):
            expected = _role_groups(members, role)
            if scope[field] != expected:
                raise ValueError("Scope group accounting disagrees with fitted evidence")
    for field, role in (
        ("intervention_support_group_count", _INTERVENTION_PREFERRED),
        ("compiled_preferred_group_count", _COMPILED_PREFERRED),
    ):
        if field in model["counts"] and model["counts"][field] < len(_role_groups(model["evidence"], role)):
            raise ValueError("Reported independent support is smaller than fitted evidence")
    if len(model["rules"]) > budget:
        raise ValueError("Rule budget exceeded")
    seen_rules = set()
    for rule in model["rules"]:
        _record(rule, {"rule_id", "task_family", "shape_id", "literals", "harmful_groups",
                       "helpful_conflict_groups", "support_source_ids", "counts",
                       "intervention_kind", "directive"}, "entry guard rule")
        rule_kind = _intervention_kind(rule["intervention_kind"])
        if rule_kind is None or rule_kind != kind or rule["directive"] != _DIRECTIVES[rule_kind]:
            raise ValueError("Rule intervention/directive disagrees with the measured intervention")
        key = _text(rule["task_family"], "rule family"), _text(rule["shape_id"], "rule shape")
        if key not in shapes:
            raise ValueError("Rule has no validated scope")
        clauses = rule["literals"]
        if not isinstance(clauses, list) or not 1 <= len(clauses) <= arity:
            raise ValueError("Rule must conjoin one or two observable literals")
        fields = set()
        for literal in clauses:
            _record(literal, {"feature", "value"}, "observable literal")
            field = _text(literal["feature"], "literal feature")
            if field in fields:
                raise ValueError("Duplicate/contradictory feature in a conjunction")
            fields.add(field)
            _validate_feature_value(field, literal["value"], shapes[key])
        expected = _candidate(grouped[key], clauses, *key, rule_kind)
        _validate_counts(rule["counts"])
        for field in ("harmful_groups", "helpful_conflict_groups", "support_source_ids"):
            _strings(rule[field], field, unique=True)
        if (rule != expected or rule["rule_id"] in seen_rules
                or len(rule["harmful_groups"]) < minimum or rule["helpful_conflict_groups"]):
            raise ValueError("Rule lacks verified independent harmful support or conflicts with helpful evidence")
        seen_rules.add(rule["rule_id"])
    if model["status"] != ("admitted_rules" if model["rules"] else "no_admitted_rule"):
        raise ValueError("Model status disagrees with admitted rules")
    if model["counts"].get("admitted_rules") != len(model["rules"]):
        raise ValueError("Admitted-rule accounting disagrees with the model")
    for scope in model["scopes"]:
        count = sum(rule["task_family"] == scope["task_family"] and rule["shape_id"] == scope["shape_id"]
                    for rule in model["rules"])
        if (scope["counts"].get("admitted_rules") != count
                or scope["status"] != ("admitted_rules" if count else "no_admitted_rule")):
            raise ValueError("Scope rule accounting disagrees with the model")
    _json(model)
    return shapes


def decide_entry_guard(model: dict, public_state: dict, program_actions: list[str],
                       validated_action_history: list[str], *, task_family: str,
                       group_id: str | None = None) -> dict:
    """Return the measured intervention's control directive, without executing it.

    Optional group_id is an audit-only leakage check, never a predictive feature.
    Fresh full-policy evaluation and TRAIN split enforcement remain with the parent.
    force_actor_once rules return defer_to_actor=True, actor_calls=1 and
    directive=defer_to_actor_once, NEVER exclude=True. The caller must bypass
    compiled alternatives for that single decision and then resume the unchanged
    deployed continuation. An unmatched/empty model requests neither operation.
    actor_calls counts requested forced calls, not natural actor calls that may
    occur later under either continuation.
    """
    _validate_model(model)
    _query(public_state, program_actions, validated_action_history, task_family)
    if group_id is not None:
        _text(group_id, "query group_id")
        if group_id in model["fit_groups"]:
            raise ValueError("Target group supplied fitting data; refit with its exclusion")
    result = {
        "exclude": False, "reason": "no_admitted_rule", "representation": model["representation"],
        "task_family": task_family, "shape_id": None, "matched_rule_ids": [],
        "support_groups": [], "support_group_count": 0,
        "intervention_kind": model["intervention_kind"], "directive": "none",
        "defer_to_actor": False, "actor_calls": 0,
    }
    if not model["rules"]:
        return result
    extracted = extract_guard_features(
        model["vocabulary"], public_state, program_actions, validated_action_history,
        task_family=task_family, representation=model["representation"],
    )
    if extracted["status"] != "ok":
        return {**result, "reason": extracted["reason"]}
    result["shape_id"] = extracted["shape_id"]
    candidates = [rule for rule in model["rules"] if rule["task_family"] == task_family
                  and rule["shape_id"] == extracted["shape_id"]]
    if not candidates:
        return {**result, "reason": "no_rule_for_family_and_shape"}
    matched = [rule for rule in candidates if _matches(extracted["features"], rule["literals"])]
    if not matched:
        return {**result, "reason": "no_matching_observable_rule"}
    groups = sorted({group for rule in matched for group in rule["harmful_groups"]})
    kind = model["intervention_kind"]
    actor = kind == "force_actor_once"
    return {
        **result, "exclude": kind == "exclude_entry",
        "defer_to_actor": actor, "actor_calls": int(actor), "directive": _DIRECTIVES[kind],
        "reason": "group_supported_empirical_actor_deferral" if actor
        else "group_supported_empirical_exclusion",
        "matched_rule_ids": sorted(rule["rule_id"] for rule in matched),
        "support_groups": groups, "support_group_count": len(groups),
    }


def decide_exclusion(model: dict, public_state: dict, program_actions: list[str],
                     validated_action_history: list[str], *, task_family: str,
                     group_id: str | None = None) -> dict:
    """Instance-exclusion-only compatibility API; actor-deferral models are rejected."""
    if not isinstance(model, dict) or model.get("intervention_kind") != "exclude_entry":
        raise ValueError(
            "decide_exclusion requires exclude_entry evidence; use decide_entry_guard for actor deferral"
        )
    return decide_entry_guard(
        model, public_state, program_actions, validated_action_history,
        task_family=task_family, group_id=group_id,
    )
