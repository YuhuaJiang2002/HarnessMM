"""TRAIN proposal compiler; candidate construction never selects or executes an action.

Bindings use one public list scope, exact public/previous paths, native defaults,
or a supported positive integer advance of the same tool argument. No argument
literal is stored in a program. Support/provenance lives outside runtime features.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import re
from typing import Any, Iterable

from phiagent.harness.gaia2_cot_jev import (
    at_path, bind, fast_candidate_action_names, observation_data,
)
from phiagent.harness.gaia2_dual_rsi import well_typed


FORMAT = "gaia2-train-cot-programs-v1"


@dataclass(frozen=True)
class Limits:
    candidates: int = 8
    expansion: int = 128
    variants: int = 64
    nodes: int = 20000
    depth: int = 8
    observation_characters: int = 1000000
    integer_advance: int = 128


LIMITS = Limits()
MIN_SUPPORT = 2


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Rejected(ValueError):
    """A named, auditable compiler/runtime refusal."""


def _equal(left: Any, right: Any) -> bool:
    return canonical(left) == canonical(right)


def _inputs(tools: dict, name: str) -> dict:
    tool = tools[name]
    return tool if isinstance(tool, dict) else tool.inputs


def _valid(arguments: dict, inputs: dict) -> bool:
    canonical(arguments)
    return not (set(arguments) - set(inputs)) and all(
        (key not in arguments and "default" in spec)
        or (key in arguments and (
            ("default" in spec and _equal(arguments[key], spec["default"]))
            or spec.get("type") == "any" or well_typed(arguments[key], spec)))
        for key, spec in inputs.items()
    )


def public_state(text: str, limits: Limits = LIMITS) -> tuple[Any, str, list]:
    if not isinstance(text, str) or not text.strip():
        raise Rejected("missing_public_observation")
    if len(text) > limits.observation_characters:
        raise Rejected("observation_capacity")
    try:
        value = observation_data(text)
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        raise Rejected("unsupported_public_literal") from error
    paths = []

    def walk(node, path):
        if len(paths) >= limits.nodes or len(path) > limits.depth:
            raise Rejected("public_schema_capacity")
        paths.append((path, node))
        if isinstance(node, dict):
            # Dynamic map keys cannot become entity-specific program constants.
            if any(not re.fullmatch(r"[a-z_]{1,64}", key) for key in node):
                raise Rejected("unsupported_dynamic_field_name")
            return ["object", [[key, walk(node[key], path + [key])] for key in sorted(node)]]
        if isinstance(node, list):
            shapes = {canonical(walk(child, path + [index])) for index, child in enumerate(node)}
            return ["array", sorted(shapes)]
        try:
            canonical(node)
        except (ValueError, TypeError) as error:
            raise Rejected("unsupported_public_literal") from error
        return type(node).__name__

    shape = walk(value, [])
    return value, digest(shape), paths


def _options(value, key, paths, previous, inputs, name, limits):
    options = []
    for path, observed in paths:
        if not _equal(value, observed):
            continue
        indices = [i for i, part in enumerate(path) if type(part) is int]
        if not indices:
            options.append(({"$obs": path}, None, None))
        elif len(indices) == 1:
            i = indices[0]
            options.append(({"$item": path[i + 1:]}, path[:i], path[i]))
    if "default" in inputs[key] and _equal(value, inputs[key]["default"]):
        options.append(({"$default": [key]}, None, None))
    for old_key, old_value in previous["action_input"].items():
        if _equal(value, old_value):
            options.append(({"$prev": [old_key]}, None, None))
    old = previous["action_input"].get(key)
    if (previous["action"] == name and type(old) is int and type(value) is int
            and 0 < value - old <= limits.integer_advance):
        options.append(({"$add": [{"$prev": [key]}, value - old]}, None, None))
    return list({canonical(option): option for option in options}.values())


def compile_transition(row: dict, limits: Limits = LIMITS) -> list[dict]:
    """Compile a public TRAIN CoT proposal, irrespective of its episode outcome."""
    if row.get("split") != "train":
        raise Rejected("train_only")
    previous, action, tools = row.get("previous"), row.get("action"), row["tools"]
    if not previous:
        raise Rejected("missing_previous_issued_action")
    if (not isinstance(action, dict) or not isinstance(action.get("action_input"), dict)
            or action.get("action") not in fast_candidate_action_names(
                tools, allow_routine_writes=True)):
        raise Rejected("requires_slow_cot_or_invalid_action")
    name, arguments = action["action"], action["action_input"]
    inputs = _inputs(tools, name)
    try:
        valid = _valid(arguments, inputs)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise Rejected("native_argument_contract")
    _, schema, paths = public_state(row.get("observation"), limits)
    keys = sorted(arguments)
    alternatives = [_options(arguments[key], key, paths, previous, inputs, name, limits)
                    for key in keys]
    if any(not values for values in alternatives):
        raise Rejected("unresolved_argument")
    combinations = 1
    for values in alternatives:
        combinations *= len(values)
    if combinations > limits.variants:
        raise Rejected("template_variant_overflow")
    programs = {}
    for choices in product(*alternatives):
        scopes = {canonical([scope, index]) for _, scope, index in choices if scope is not None}
        if len(scopes) > 1:
            continue
        scope = next((scope for _, scope, _ in choices if scope is not None), None)
        program = {
            "previous_tool": previous["action"], "observation_schema": schema,
            "action": name, "contract_sha256": digest(inputs),
            "action_input": {key: choice[0] for key, choice in zip(keys, choices)},
        }
        if scope is not None:
            program["foreach"] = scope
        programs[digest(program)] = program
    if not programs:
        raise Rejected("incompatible_public_list_scopes")
    return list(programs.values())


def compile_bank(rows: Iterable[dict], *, min_support: int = MIN_SUPPORT) -> tuple[dict, dict]:
    """Keep templates with distinct-task support >= min_support (default two).

    Explicit support one is exploratory TRAIN resubstitution, not evidence of
    cross-task generalization. It changes no compiler or candidate-binding rule.
    """
    if type(min_support) is not int or min_support < 1:
        raise ValueError("A positive integer independent TRAIN task support is required")
    programs, support, rejected = {}, {}, Counter()
    proposals = 0
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Non-TRAIN input is forbidden, not silently filtered")
        task = row.get("task_hash")
        if not isinstance(task, str) or not re.fullmatch(r"[0-9a-f]{64}", task):
            raise ValueError("A public task hash is required solely for support accounting")
        proposals += 1
        try:
            variants = compile_transition(row)
        except Rejected as error:
            rejected[str(error)] += 1
            continue
        for program in variants:
            identity = digest(program)
            programs[identity] = program
            support.setdefault(identity, set()).add(task)
    admitted = sorted(key for key in programs if len(support[key]) >= min_support)
    bank = {
        "format": FORMAT, "min_support": min_support, "limits": asdict(LIMITS),
        "programs": [programs[key] for key in admitted],
    }
    if min_support == 1:
        bank["exploratory_single_task"] = True
    metadata = {
        "proposals": proposals, "proposed_templates": len(programs),
        "admitted_templates": len(admitted), "rejected_proposals": dict(sorted(rejected.items())),
        "templates_below_min_support": len(programs) - len(admitted),
        "support_unit": "distinct public task content SHA256; repetitions/calls do not add support",
        "support": {key: {"tasks": len(support[key]), "task_hashes": sorted(support[key])}
                    for key in sorted(programs)},
    }
    return bank, metadata


@dataclass(frozen=True)
class CandidateSet:
    status: str
    actions: tuple[dict, ...] = ()
    reasons: tuple[str, ...] = ()
    matched_templates: int = 0


def candidates(bank: dict, *, observation: str, previous: dict | None,
               tools: dict) -> CandidateSet:
    """Return the complete finite set, or nothing with explicit unknown/overflow.

    The caller must still use the existing finite-label discriminator and its
    confidence/unknown gates. Neither task identity nor training support is input.
    """
    if (bank.get("format") != FORMAT or bank.get("limits") != asdict(LIMITS)
            or type(bank.get("min_support")) is not int or bank["min_support"] < 1
            or (bank["min_support"] == 1 and bank.get("exploratory_single_task") is not True)):
        raise ValueError("Unknown or modified frozen program-bank contract")
    if previous is None:
        return CandidateSet("unknown", reasons=("missing_previous_issued_action",))
    try:
        state, schema, _ = public_state(observation)
    except Rejected as error:
        return CandidateSet("unknown", reasons=(str(error),))
    matching = [p for p in bank["programs"]
                if p["previous_tool"] == previous["action"] and p["observation_schema"] == schema]
    if not matching:
        return CandidateSet("unknown", reasons=("no_supported_template",))
    rows = {}
    for program in matching:
        try:
            name = program["action"]
            inputs = _inputs(tools, name)
            if digest(inputs) != program["contract_sha256"]:
                raise Rejected("native_contract_changed")
            members = at_path(state, program["foreach"]) if "foreach" in program else [None]
            if not isinstance(members, list):
                raise Rejected("unresolved_public_list")
            if len(members) > LIMITS.expansion:
                return CandidateSet("overflow", reasons=("public_expansion_overflow",),
                                    matched_templates=len(matching))
            for member in members:
                arguments = {}
                for key, expression in program["action_input"].items():
                    if set(expression) == {"$default"}:
                        if expression["$default"] != [key]:
                            raise Rejected("invalid_default_reference")
                        arguments[key] = deepcopy(inputs[key]["default"])
                    else:
                        if not (len(expression) == 1 and set(expression) <= {
                                "$obs", "$prev", "$item", "$add"}):
                            raise Rejected("literal_program_argument_forbidden")
                        if "$add" in expression:
                            terms = expression["$add"]
                            if (not isinstance(terms, list) or len(terms) != 2
                                    or terms[0] != {"$prev": [key]}
                                    or type(terms[1]) is not int
                                    or not 0 < terms[1] <= LIMITS.integer_advance
                                    or previous["action"] != name):
                                raise Rejected("invalid_learned_integer_advance")
                        arguments[key] = bind(expression, state, previous["action_input"], member)
                if not _valid(arguments, inputs):
                    raise Rejected("native_argument_contract")
                action = {"action": name, "action_input": arguments}
                rows[digest(action)] = action
                if len(rows) > LIMITS.candidates:
                    return CandidateSet("overflow", reasons=("complete_candidate_overflow",),
                                        matched_templates=len(matching))
        except (KeyError, IndexError, ValueError, TypeError) as error:
            return CandidateSet("unknown", reasons=("unresolved_template:" + str(error),),
                                matched_templates=len(matching))
    if not rows:
        return CandidateSet("unknown", reasons=("empty_candidate_set",),
                            matched_templates=len(matching))
    return CandidateSet("complete", tuple(rows[key] for key in sorted(rows)),
                        matched_templates=len(matching))
