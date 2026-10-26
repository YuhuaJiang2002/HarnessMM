"""Compile and execute literal fragments with observable-transition contracts.

The v5 JSON schema stores conditioned instances in ``programs`` and maps starting
demonstration IDs to ``instance_id`` values in ``entry_index``. ``program_id``
retains the original candidate ID. ``conditioning`` is ``task_family`` for an
opaque, caller-supplied actor-visible family label, or ``exact_task_text`` when
source metadata is absent. Exact fallback requires an unlabeled controller and
casefolded, whitespace-collapsed task equality; labeled instances require the
same supplied family. Every mode also checks the selected source's ``goal_words``
(source-task words intersected with literal action words) against the current task.

``history_prerequisite_words`` is the intersection of native-action word tokens
in EVERY retained source example's explicit ``pre_action_history``, without a
task-word filter. The caller must verify these actions in their original native
states. Neither task instructions nor observations supply history facts. Each step has
``positive_guard`` and ``refined_guard`` records containing up to three OR
clauses. STEP literals remain ``{"sign": "+"/"-", "tokens": [word, ...]}``.
NET observation literals add ``"field": "observation"``; history literals use
``{"field": "history_action", "sign": "+"/"-", "action": full_action_string}``.
"+" requires absence before and presence after; "-" reverses those
requirements. Only a clause whose precondition held can witness a postcondition.

Support, coverage and refinement penalties count independent source groups
within an instance, never across families. These are empirical preconditions,
not guaranteed world state or task correctness. Literal arguments are not
rewritten; goal paraphrases may pass only within a supplied family and with the
source word bindings. All controllers share the step-eligible instance pool and
entry prerequisites. ``refined_no_memory`` removes only the seen-entry-state
registry, not the counter or native-action history.
Each instance also has ``positive_net_guard`` and ``refined_net_guard``, learned
from its first to last typed observable working state, never by conjoining step
guards. Net literals explicitly tag ``observation`` ngrams or ``history_action``
facts (exact full native action strings). Positive history grows by the program's
verified native actions. An observation return with new native-action facts is
progress, not a no-progress training negative. Both net guards are null when no
group-supported change exists; step-only modes still retain those instances.
``composed_contract`` uses positive step/net guards and ``refined_composed``
uses refined step/net guards. Net-checking modes match at entry and check the
same pre-matched clauses at completion. Three positive-composed ablations keep
the same net-supported instance pool: ``composed_no_net_checks`` disables only
net enforcement while retaining its learned history projection;
``composed_raw_reentry`` replaces that projection with the original raw public
state hash but keeps instance identity; ``composed_no_reentry_memory`` disables
only the seen-entry registry. Native history, counters and actor boundaries stay.
Whole-fragment negatives require matching original public-state hashes from the
parent's actual execution trace plus unchanged observation and validated-action
set, not hashes reconstructed from observation text. Composed reentry uses the
instance, current task/observation/native actions, and that mode's selected
history-fact projection, replacing raw public history in the key. Unrelated
history does not reset the projected block. The raw-reentry ablation deliberately
retains raw-history sensitivity. Step-only modes retain their v3 state hash.
``reentry_projection`` stores positive/refined maps with ``history_actions``
(exact selected net action facts) and ``history_words`` (existing prerequisite
word facts). Only their presence is projected, not all actions sharing a word.
Construct a controller per episode, or call ``reset_episode`` between episodes.
``decide`` must be followed by actual-action ``observe`` feedback, including
when the decision defers. Actor feedback without a decision is also permitted.
``library_behavior_hash`` fingerprints the executable library for activation
audits, excluding fit diagnostics and source timestamps. Pin the controller
implementation, memory policy and constructor arguments separately.
``runtime_contract_hash`` additionally resolves a mode to its active guards and
omits logging labels/indices, so identical positive/refined action policies alias.
TRAIN entry interventions require ``allow_training_interventions=True``.
``exclude_entry`` records an episode-local instance/working-state exclusion,
not a learned guard or an undo operation. The caller enforces TRAIN membership;
task IDs, evaluator state and future labels are not intervention inputs.
Controller source, constructor flags and the ordered episode intervention log
must be pinned separately: neither library hash contains that external state.
Training exclusions always use relevant projected history, including in the
raw-reentry and no-memory ablations; no ablation disables that opt-in safeguard.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import hashlib
import json
import math
import re

from phiagent.learning import counterfactual_invocation as invocation


LIBRARY_SCHEMA = "chisel-observable-fragment-contracts/v5"
LIBRARY_BEHAVIOR_HASH_SCHEMA = "chisel-fragment-library-behavior/v3"
RUNTIME_CONTRACT_HASH_SCHEMA = "chisel-fragment-runtime-contract/v2"
MODES = (
    "independent_actions", "ordered_fragment", "positive_contract",
    "refined_contract", "refined_no_memory", "composed_contract", "refined_composed",
    "composed_no_net_checks", "composed_raw_reentry", "composed_no_reentry_memory",
)
_POSITIVE_COMPOSED_MODES = (
    "composed_contract", "composed_no_net_checks", "composed_raw_reentry",
    "composed_no_reentry_memory",
)
_MAX_CLAUSES = 3
_CONJUNCTION_ATOMS = 32
_OBJECTIVES = {
    "positive": (
        "greedy marginal positive-source-group coverage; ties: total positive groups, "
        "fewer literals, lexical literals; stop without new positive coverage"
    ),
    "refined": (
        "greedy marginal positive groups minus marginal negative-precondition groups "
        "(unit group penalty); ties: marginal positive groups, fewer total negative groups, "
        "total positive groups, fewer literals, lexical literals; after the first clause "
        "require positive marginal objective and new positive coverage"
    ),
}
_TASK_CONDITIONING = (
    "supplied_task_family_else_exact_casefold_whitespace_normalized_source_task"
)
_HISTORY_PREREQUISITES = "intersection_of_caller_verified_native_pre_action_history_tokens"
_NEGATIVE_CONDITIONING = "exact_action_and_matching_supplied_family_metadata"
_NEGATIVE_PROGRAM_KIND = "original_hash_cycle_with_unchanged_observation_and_native_action_set"
_NEGATIVE_PROGRAM_CONDITIONING = "exact_action_sequence_and_matching_supplied_family_metadata"
_NET_WORKING_STATE = "observation_ngrams_and_validated_full_native_action_set"
_COMPOSED_REENTRY = "instance_task_observation_actions_and_selected_native_history"
_PUBLIC_STATE_REENTRY = "v3_public_state_sha256"
_INSTANCE_RAW_REENTRY = "instance_raw_public_state_sha256"
_TRAINING_ENTRY_EXCLUDED = "training_entry_excluded"
_SEMANTICS = (
    "Supplied-family/exact-task conditioned observable changes and native-action-history "
    "correlations are empirical preconditions, not guaranteed world state or task correctness"
)
Atom = tuple[str, str, tuple[str, ...]]
WorkingState = dict[str, set[tuple[str, ...]]]


def _mode_settings(mode: str) -> tuple[str | None, str | None, bool, str, bool, str]:
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Unknown fragment controller mode")
    step_variant = ("positive" if mode == "positive_contract" or mode in _POSITIVE_COMPOSED_MODES else
                    "refined" if mode in (
                        "refined_contract", "refined_no_memory", "refined_composed",
                    ) else None)
    net_variant = ("positive" if mode in _POSITIVE_COMPOSED_MODES else
                   "refined" if mode == "refined_composed" else None)
    net_checks = net_variant is not None and mode != "composed_no_net_checks"
    reentry_scope = (_INSTANCE_RAW_REENTRY if mode == "composed_raw_reentry" else
                     _COMPOSED_REENTRY if net_variant is not None else _PUBLIC_STATE_REENTRY)
    return (
        step_variant, net_variant, mode not in ("refined_no_memory", "composed_no_reentry_memory"),
        "independent_policy" if mode == "independent_actions" else "stored_sequence",
        net_checks, reentry_scope,
    )


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else ''}string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid integer {name}")
    return value


def _record(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or not keys <= value.keys():
        raise ValueError(f"Invalid {name} schema")
    return value


def _texts(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    for item in value:
        _text(item, name)
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate {name}")
    return value


def _groups(value: object) -> frozenset[str]:
    if not isinstance(value, (set, frozenset)):
        raise ValueError("excluded_groups must be a set of group IDs")
    for group in value:
        _text(group, "excluded group")
    return frozenset(value)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _task_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _family(value: object) -> str | None:
    return None if value is None else _text(value, "task_family")


def _condition(task_family: str | None, task_text: str) -> tuple[str, str]:
    return ("task_family", task_family) if task_family is not None else ("exact_task_text", task_text)


def _instance_id(program_id: str, condition: tuple[str, str]) -> str:
    identity = json.dumps([program_id, *condition], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()


def _action_history(value: object, name: str = "pre_action_history") -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must list caller-verified native action strings")
    for action in value:
        _text(action, "native source action")
    return value


def _history_words(actions: list[str]) -> set[str]:
    return {word for action in actions for word in _words(action)}


def _history_prerequisites(examples: list[dict]) -> list[str]:
    if not examples:
        raise ValueError("History prerequisites require retained positive examples")
    required = _history_words(examples[0]["pre_action_history"])
    for example in examples[1:]:
        required.intersection_update(_history_words(example["pre_action_history"]))
    return sorted(required)


def _ngrams(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {(word,) for word in words} | set(zip(words, words[1:]))


def _change_atoms(before: str, after: str) -> set[Atom]:
    pre, post = _ngrams(before), _ngrams(after)
    return ({("observation", "+", term) for term in post - pre}
            | {("observation", "-", term) for term in pre - post})


def _working_state(observation: str, native_actions: set[str] | None = None) -> WorkingState:
    return {
        "observation": _ngrams(observation),
        "history_action": set() if native_actions is None else {(action,) for action in native_actions},
    }


def _net_changes(before: str, after: str, history: list[str], actions: list[str]) -> set[Atom]:
    return _change_atoms(before, after) | {
        ("history_action", "+", (action,)) for action in set(actions) - set(history)
    }


def _literal_atom(literal: dict, *, typed: bool) -> Atom:
    field = literal["field"] if typed else "observation"
    value = (literal["action"],) if field == "history_action" else tuple(literal["tokens"])
    return field, literal["sign"], value


def _atom_literal(atom: Atom, *, typed: bool) -> dict:
    field, sign, value = atom
    if not typed:
        return {"sign": sign, "tokens": list(value)}
    if field == "history_action":
        return {"field": field, "sign": sign, "action": value[0]}
    return {"field": field, "sign": sign, "tokens": list(value)}


def _atom_key(atom: Atom) -> tuple:
    field, sign, value = atom
    return (0 if field == "history_action" else 1), len(value), sign, value


def _holds(atoms: tuple[Atom, ...], terms: WorkingState, *, post: bool) -> bool:
    return all((value in terms[field]) == (sign == ("+" if post else "-"))
               for field, sign, value in atoms)


def _matching(guard: dict, observation: str, *, post: bool,
              allowed: list[int] | None = None, typed: bool = False,
              native_actions: set[str] | None = None) -> list[int]:
    terms = _working_state(observation, native_actions)
    indices = range(len(guard["clauses"])) if allowed is None else allowed
    return [
        index for index in indices
        if _holds(tuple(_literal_atom(literal, typed=typed)
                        for literal in guard["clauses"][index]["literals"]), terms, post=post)
    ]


def _guard_contract(guard: dict | None, *, preserve_clause_order: bool = False,
                    typed: bool = False) -> dict | None:
    if guard is None:
        return None
    clauses = [
        sorted((_literal_atom(literal, typed=typed) for literal in clause["literals"]),
               key=_atom_key)
        for clause in guard["clauses"]
    ]
    if not preserve_clause_order:
        clauses.sort(key=lambda atoms: tuple(_atom_key(atom) for atom in atoms))
    return {"clauses": [
        {"literals": [_atom_literal(atom, typed=typed) for atom in atoms]}
        for atoms in clauses
    ]}


def _reentry_projection(guard: dict | None, words: list[str]) -> dict:
    return {
        "history_actions": sorted({
            literal["action"] for clause in ([] if guard is None else guard["clauses"])
            for literal in clause["literals"] if literal["field"] == "history_action"
        }),
        "history_words": list(words),
    }


def _select(candidates: list[dict], variant: str, *, typed: bool = False) -> list[dict]:
    remaining = list(candidates)
    selected, covered, accepted = [], set(), set()

    def rank(candidate: dict) -> tuple:
        positives = set(candidate["positive_groups"])
        negatives = set(candidate["negative_precondition_groups"])
        new_positive, new_negative = len(positives - covered), len(negatives - accepted)
        lexical = tuple(_atom_key(_literal_atom(literal, typed=typed))
                        for literal in candidate["literals"])
        tail = (-len(positives), len(lexical), lexical)
        if variant == "positive":
            return (-new_positive, *tail)
        return (-(new_positive - new_negative), -new_positive, len(negatives), *tail)

    while remaining and len(selected) < _MAX_CLAUSES:
        best = min(remaining, key=rank)
        new_positive = set(best["positive_groups"]) - covered
        new_negative = set(best["negative_precondition_groups"]) - accepted
        gain = len(new_positive) - (len(new_negative) if variant == "refined" else 0)
        if selected and (not new_positive or gain <= 0):
            break
        selected.append({
            **deepcopy(best),
            "selection": {
                "new_positive_group_count": len(new_positive),
                "new_negative_precondition_group_count": len(new_negative),
                "objective_gain": gain,
            },
        })
        covered.update(new_positive)
        accepted.update(new_negative)
        remaining.remove(best)
    return selected


def _fit_step(samples: list[tuple[str, str, set[Atom]]],
              negatives: list[tuple[str, WorkingState]],
              minimum_groups: int, max_literals: int, *, typed: bool = False) -> dict | None:
    by_atom: dict[Atom, set[str]] = defaultdict(set)
    group_by_id = {identifier: group for identifier, group, _ in samples}
    for identifier, _, changes in samples:
        for atom in changes:
            by_atom[atom].add(identifier)

    def groups(ids: set[str]) -> set[str]:
        return {group_by_id[identifier] for identifier in ids}

    eligible = [atom for atom, ids in by_atom.items() if len(groups(ids)) >= minimum_groups]
    eligible.sort(key=lambda atom: (-len(groups(by_atom[atom])), _atom_key(atom)))
    if not eligible:
        return None
    clauses = [(atom,) for atom in eligible]
    top = eligible[:_CONJUNCTION_ATOMS] if max_literals == 2 else []
    clauses.extend(tuple(sorted(pair, key=_atom_key)) for pair in combinations(top, 2)
                   if len(groups(by_atom[pair[0]] & by_atom[pair[1]])) >= minimum_groups)
    negative_by_atom = {
        atom: {index for index, (_, terms) in enumerate(negatives)
               if _holds((atom,), terms, post=False)}
        for atom in eligible
    }
    candidates = []
    for clause in clauses:
        positive_ids = set.intersection(*(by_atom[atom] for atom in clause))
        negative_ids = set.intersection(*(negative_by_atom[atom] for atom in clause))
        positive_groups = sorted(groups(positive_ids))
        negative_groups = sorted({negatives[index][0] for index in negative_ids})
        candidates.append({
            "literals": [_atom_literal(atom, typed=typed) for atom in clause],
            "positive_groups": positive_groups,
            "positive_group_count": len(positive_groups),
            "positive_demonstration_ids": sorted(positive_ids),
            "positive_transition_count": len(positive_ids),
            "negative_precondition_groups": negative_groups,
            "negative_precondition_group_count": len(negative_groups),
            "negative_precondition_count": len(negative_ids),
        })
    result = {}
    for variant in _OBJECTIVES:
        selected = _select(candidates, variant, typed=typed)
        covered = sorted({group for clause in selected for group in clause["positive_groups"]})
        accepted = sorted({group for clause in selected
                           for group in clause["negative_precondition_groups"]})
        result[f"{variant}_guard"] = {
            "objective": _OBJECTIVES[variant] + (
                "; net field ties prefer exact validated history-action facts before observation ngrams"
                if typed else ""
            ),
            "clauses": selected,
            "positive_groups": sorted(set(group_by_id.values())),
            "negative_groups": sorted({group for group, _ in negatives}),
            "covered_positive_groups": covered,
            "accepted_negative_groups": accepted,
            "counts": {
                "positive_transition_count": len(samples),
                "positive_group_count": len(set(group_by_id.values())),
                "negative_transition_count": len(negatives),
                "negative_group_count": len({group for group, _ in negatives}),
                "observed_atom_count": len(by_atom),
                "eligible_atom_count": len(eligible),
                "conjunction_atom_count": len(top),
                "candidate_clause_count": len(candidates),
                "selected_clause_count": len(selected),
                "covered_positive_group_count": len(covered),
                "accepted_negative_group_count": len(accepted),
            },
        }
    return result


def compile_library(programs: list[dict], demonstrations: list[dict], transitions: dict,
                    negative_transitions: list[dict], *,
                    excluded_groups: frozenset[str] = frozenset(),
                    minimum_groups: int = 2, max_literals: int = 1,
                    negative_programs: list[dict] | None = None) -> dict:
    """Fit both guard variants after exclusion; incomplete windows are counted and omitted.

    Duplicate IDs/windows and conflicting source identities are errors. Repeated
    source prefixes do not increase group support. Only byte-identical observable
    stutters supply negatives, matched by exact action and supplied family.
    Positive examples are partitioned by original program ID and supplied
    ``task_family`` before support counting and fitting. A source example's
    optional family must agree with every referenced transition's family.
    Absent metadata uses exact normalized task text, never an inferred family.
    Negatives may also carry ``task_family``; unlabeled negatives are available
    only to exact-text instances. Unused negatives are counted explicitly.
    Every transition must provide full, caller-verified native
    ``pre_action_history``. Filtered history may be shorter than ``start_turn``
    because invalid actions are omitted; it cannot be longer, and the fragment
    suffix must agree with the demonstrated native action sequence.
    Arity-two candidates use at most 32 group-ranked atoms; singleton candidates
    remain available. Refinement never changes the eligible program pool.
    Counts distinguish original programs from conditioned instances;
    ``rejected_programs`` diagnoses rejected instances or originals with no
    complete source window (null instance ID/task text in the latter case).

    ``negative_programs`` also requires pre_action_history/post_action_history,
    containing only caller-validated full native action strings. History sets
    must grow monotonically, with new facts drawn from the actual action sequence.
    Observation returns that add history facts are counted and ignored, not
    treated as no-progress negatives. Remaining negatives require unchanged
    observation/history sets and parent-computed, equal original public-state
    hashes (pre_state_sha256/post_state_sha256). Missing or contradictory
    evidence is an error. Only exact action-sequence/family matches train net
    guards. Step and net negatives never cross-train. Positive net history is
    pre_action_history followed by the program's validated native actions.
    """
    excluded = _groups(excluded_groups)
    _integer(minimum_groups, "minimum_groups", minimum=1)
    _integer(max_literals, "max_literals", minimum=1, maximum=2)
    if not all(isinstance(value, list) for value in
               (programs, demonstrations, negative_transitions)) or not isinstance(transitions, dict):
        raise ValueError("Expected program/demonstration/negative lists and a transition mapping")
    if negative_programs is None:
        negative_programs = []
    if not isinstance(negative_programs, list):
        raise ValueError("negative_programs must be a list of observed cycle records")
    lookup, task_groups = {}, {}
    for row in demonstrations:
        _record(row, {"example_id", "task_id", "group_id", "action", "public_state"},
                "demonstration")
        for name in ("example_id", "task_id", "group_id", "action"):
            _text(row[name], name)
        if row["example_id"] in lookup:
            raise ValueError("Duplicate demonstration ID")
        if task_groups.setdefault(row["task_id"], row["group_id"]) != row["group_id"]:
            raise ValueError("A source task cannot belong to multiple groups")
        invocation.validate_public_state(row["public_state"])
        if row["action"] not in row["public_state"]["actions"]:
            raise ValueError("Demonstration action is not native")
        lookup[row["example_id"]] = row
    for identifier, transition in transitions.items():
        _text(identifier, "transition demonstration ID")
        _record(transition, {"post_observation", "pre_action_history"}, "transition")
        _text(transition["post_observation"], "post_observation", empty=True)
        _action_history(transition["pre_action_history"])
        _family(transition.get("task_family"))

    counts = {
        "input_programs": len(programs),
        "input_demonstrations": len(demonstrations),
        "input_transitions": len(transitions),
        "unknown_transition_ids": len(set(transitions) - lookup.keys()),
        "input_negative_transitions": len(negative_transitions),
        "excluded_demonstrations": sum(row["group_id"] in excluded for row in demonstrations),
        "excluded_negative_transitions": 0,
        "ignored_non_stutter_negatives": 0,
        "duplicate_negative_transitions": 0,
        "usable_negative_transitions": 0,
        "input_negative_programs": len(negative_programs),
        "excluded_negative_programs": 0,
        "ignored_non_cycle_negative_programs": 0,
        "ignored_information_gaining_observation_cycles": 0,
        "duplicate_negative_programs": 0,
        "usable_negative_programs": 0,
        "instances_without_net_guard": 0,
        "step_guard_pairs_equal": 0,
        "step_guard_pairs_different": 0,
        "net_guard_pairs_equal": 0,
        "net_guard_pairs_different": 0,
        "input_examples": 0,
        "excluded_examples": 0,
        "incomplete_examples": 0,
        "missing_demonstration_references": 0,
        "missing_transition_references": 0,
        "complete_examples": 0,
        "input_conditioned_instances": 0,
        "retained_instances": 0,
        "rejected_instances": 0,
        "retained_programs": 0,
        "rejected_programs": 0,
    }
    negative_by_action: dict[tuple[str | None, str], list] = defaultdict(list)
    seen_negatives = set()
    for row in negative_transitions:
        _record(row, {"group_id", "action", "observation", "post_observation"}, "negative")
        _text(row["group_id"], "negative group")
        _text(row["action"], "negative action")
        family = _family(row.get("task_family"))
        for field in ("observation", "post_observation"):
            _text(row[field], field, empty=True)
        if row["group_id"] in excluded:
            counts["excluded_negative_transitions"] += 1
            continue
        if row["observation"] != row["post_observation"]:
            counts["ignored_non_stutter_negatives"] += 1
            continue
        key = family, row["group_id"], row["action"], row["observation"]
        if key in seen_negatives:
            counts["duplicate_negative_transitions"] += 1
            continue
        seen_negatives.add(key)
        counts["usable_negative_transitions"] += 1
        negative_by_action[family, row["action"]].append(
            (row["group_id"], _working_state(row["observation"])),
        )

    negative_by_program: dict[tuple[str | None, tuple[str, ...]], list] = defaultdict(list)
    seen_cycles, cycle_observations = set(), {}
    for row in negative_programs:
        _record(row, {
            "group_id", "task_family", "actions", "observation", "post_observation",
            "pre_state_sha256", "post_state_sha256", "pre_action_history", "post_action_history",
        }, "negative program")
        _text(row["group_id"], "negative program group")
        family = _family(row["task_family"])
        actions = row["actions"]
        if not isinstance(actions, list) or not 2 <= len(actions) <= 4:
            raise ValueError("A negative program requires two to four actual actions")
        for action in actions:
            _text(action, "negative program action")
        for field in ("observation", "post_observation"):
            _text(row[field], field, empty=True)
        pre_actions = set(_action_history(row["pre_action_history"]))
        post_actions = set(_action_history(row["post_action_history"], "post_action_history"))
        for field in ("pre_state_sha256", "post_state_sha256"):
            digest = _text(row[field], field)
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Cycle negatives require original public-state SHA-256 hashes")
        if row["group_id"] in excluded:
            counts["excluded_negative_programs"] += 1
            continue
        if not pre_actions <= post_actions:
            raise ValueError("Validated native-action facts cannot disappear inside a negative program")
        if not post_actions - pre_actions <= set(actions):
            raise ValueError("New native-action facts must come from the negative program's actions")
        if row["observation"] == row["post_observation"] and post_actions != pre_actions:
            counts["ignored_information_gaining_observation_cycles"] += 1
            continue
        if row["pre_state_sha256"] != row["post_state_sha256"]:
            counts["ignored_non_cycle_negative_programs"] += 1
            continue
        if row["observation"] != row["post_observation"]:
            raise ValueError("Equal original public-state hashes contradict different observations")
        if cycle_observations.setdefault(row["pre_state_sha256"], row["observation"]) != row[
            "observation"
        ]:
            raise ValueError("One original cycle state hash cannot name different observations")
        key = (family, tuple(actions), row["group_id"], row["pre_state_sha256"],
               row["observation"], tuple(sorted(pre_actions)))
        if key in seen_cycles:
            counts["duplicate_negative_programs"] += 1
            continue
        seen_cycles.add(key)
        counts["usable_negative_programs"] += 1
        negative_by_program[family, tuple(actions)].append(
            (row["group_id"], _working_state(row["observation"], pre_actions)),
        )

    retained, rejected, seen_programs = [], [], set()
    changes_by_id, source_positions, demo_positions, task_families = {}, {}, {}, {}
    for program in programs:
        _record(program, {"program_id", "actions", "examples"}, "program")
        identifier = _text(program["program_id"], "program_id")
        if identifier in seen_programs:
            raise ValueError("Duplicate program ID")
        seen_programs.add(identifier)
        actions = program["actions"]
        if not isinstance(actions, list) or not 2 <= len(actions) <= 4:
            raise ValueError("A fragment must have two to four actions")
        for action in actions:
            _text(action, "fragment action")
        if not isinstance(program["examples"], list):
            raise ValueError("Program examples must be a list")
        by_condition: dict[tuple[str, str], list[dict]] = defaultdict(list)
        seen_starts, seen_windows = set(), set()
        local = {key: 0 for key in (
            "input_examples", "excluded_examples", "incomplete_examples",
            "missing_demonstration_references", "missing_transition_references", "complete_examples",
        )}
        for example in program["examples"]:
            _record(example, {"task_id", "group_id", "start_turn", "demonstration_ids"},
                    "fragment example")
            task = _text(example["task_id"], "source task")
            group = _text(example["group_id"], "source group")
            family = _family(example.get("task_family"))
            start = _integer(example["start_turn"], "start_turn")
            ids = _texts(example["demonstration_ids"], "source demonstration IDs")
            if len(ids) != len(actions):
                raise ValueError("Source demonstration count must match fragment length")
            if ids[0] in seen_starts or (task, start) in seen_windows:
                raise ValueError("Duplicate source fragment window")
            seen_starts.add(ids[0])
            seen_windows.add((task, start))
            local["input_examples"] += 1
            if group in excluded:
                local["excluded_examples"] += 1
                continue
            if task_groups.setdefault(task, group) != group:
                raise ValueError("Fragment source group disagrees with its task")
            if task_families.setdefault(task, family) != family:
                raise ValueError("A source task cannot belong to multiple conditioning families")
            for step, demo_id in enumerate(ids):
                row = lookup.get(demo_id)
                if row is not None and (row["group_id"] != group or row["task_id"] != task
                                        or row["action"] != actions[step]):
                    raise ValueError("Fragment crosses source identities or action sequence")
                position = task, start + step
                if (source_positions.setdefault(position, demo_id) != demo_id
                        or demo_positions.setdefault(demo_id, position) != position):
                    raise ValueError("Conflicting demonstration positions")
            missing_demos = sum(demo_id not in lookup for demo_id in ids)
            missing_transitions = sum(demo_id not in transitions for demo_id in ids)
            local["missing_demonstration_references"] += missing_demos
            local["missing_transition_references"] += missing_transitions
            if missing_demos or missing_transitions:
                local["incomplete_examples"] += 1
                continue
            if any(_family(transitions[demo_id].get("task_family")) != family for demo_id in ids):
                raise ValueError("Source example and transitions disagree on task_family metadata")
            if any(transitions[ids[step]]["post_observation"]
                   != lookup[ids[step + 1]]["public_state"]["observation"]
                   for step in range(len(ids) - 1)):
                raise ValueError("Source fragment observations are not consecutive")
            source = lookup[ids[0]]["public_state"]
            task_text = _task_text(source["task"])
            if any(_task_text(lookup[demo_id]["public_state"]["task"]) != task_text
                   for demo_id in ids):
                raise ValueError("Source task text changes inside a fragment")
            history = transitions[ids[0]]["pre_action_history"]
            if len(history) > start or any(
                transitions[demo_id]["pre_action_history"] != history + actions[:step]
                for step, demo_id in enumerate(ids)
            ):
                raise ValueError("Native source history disagrees with fragment turns or actions")
            by_condition[_condition(family, task_text)].append({
                "task_id": task, "group_id": group, "start_turn": start,
                "task_family": family, "task_text": task_text,
                "demonstration_ids": list(ids),
                "goal_words": sorted(set(_words(task_text)) & set(_words(" ".join(actions)))),
                "pre_action_history": list(history),
                "start_state_sha256": invocation.public_state_sha256(source),
            })
            local["complete_examples"] += 1
        for key, value in local.items():
            counts[key] += value
        counts["input_conditioned_instances"] += len(by_condition)
        if not by_condition:
            rejected.append({
                "program_id": identifier, "instance_id": None, "task_text": None,
                "conditioning": None, "task_family": None,
                "reason": "no_complete_source_examples", "counts": local,
                "first_unsupported_step": None,
            })
        for condition, examples in sorted(by_condition.items()):
            conditioning, condition_value = condition
            family = condition_value if conditioning == "task_family" else None
            task_text = condition_value if conditioning == "exact_task_text" else None
            source_groups = sorted({example["group_id"] for example in examples})
            instance_id = _instance_id(identifier, condition)
            identity = {
                "program_id": identifier, "instance_id": instance_id, "conditioning": conditioning,
                "task_family": family, "task_text": task_text,
            }
            instance_counts = {
                "complete_examples": len(examples), "source_group_count": len(source_groups),
            }
            reason, steps = None, []
            if len(source_groups) < minimum_groups:
                reason = "insufficient_source_groups"
            else:
                for step, action in enumerate(actions):
                    ids = sorted({example["demonstration_ids"][step] for example in examples})
                    samples = []
                    for demo_id in ids:
                        if demo_id not in changes_by_id:
                            changes_by_id[demo_id] = _change_atoms(
                                lookup[demo_id]["public_state"]["observation"],
                                transitions[demo_id]["post_observation"],
                            )
                        samples.append((demo_id, lookup[demo_id]["group_id"], changes_by_id[demo_id]))
                    guards = _fit_step(
                        samples, negative_by_action[family, action], minimum_groups, max_literals,
                    )
                    if guards is None:
                        reason = "step_without_supported_change_witness"
                        break
                    steps.append({"step": step, "action": action, **guards})
            if reason is not None:
                rejected.append({
                    **identity,
                    "reason": reason, "counts": instance_counts,
                    "first_unsupported_step": len(steps) if len(source_groups) >= minimum_groups
                    else None,
                })
                continue
            net_samples = [
                (example["demonstration_ids"][0], example["group_id"], _net_changes(
                    lookup[example["demonstration_ids"][0]]["public_state"]["observation"],
                    transitions[example["demonstration_ids"][-1]]["post_observation"],
                    example["pre_action_history"], actions,
                ))
                for example in examples
            ]
            net_guards = _fit_step(
                net_samples, negative_by_program[family, tuple(actions)], minimum_groups, max_literals,
                typed=True,
            )
            if net_guards is None:
                counts["instances_without_net_guard"] += 1
            history_words = _history_prerequisites(examples)
            positive_net = None if net_guards is None else net_guards["positive_guard"]
            refined_net = None if net_guards is None else net_guards["refined_guard"]
            retained.append({
                **identity,
                "history_prerequisite_words": history_words,
                "actions": list(actions), "source_groups": source_groups,
                "source_tasks": sorted({example["task_id"] for example in examples}),
                "examples": sorted(examples, key=lambda row: row["demonstration_ids"][0]),
                "steps": steps, "counts": instance_counts,
                "positive_net_guard": positive_net,
                "refined_net_guard": refined_net,
                "reentry_projection": {
                    "positive": _reentry_projection(positive_net, history_words),
                    "refined": _reentry_projection(refined_net, history_words),
                },
            })
    retained.sort(key=lambda row: (row["program_id"], row["instance_id"]))
    entry_index: dict[str, list[str]] = defaultdict(list)
    fit_groups, net_fit_groups, used_negative_keys, used_cycle_keys = set(), set(), set(), set()
    for program in retained:
        fit_groups.update(program["source_groups"])
        for example in program["examples"]:
            entry_index[example["demonstration_ids"][0]].append(program["instance_id"])
        for step in program["steps"]:
            fit_groups.update(step["refined_guard"]["negative_groups"])
            used_negative_keys.add((program["task_family"], step["action"]))
            equal = _guard_contract(step["positive_guard"]) == _guard_contract(step["refined_guard"])
            counts["step_guard_pairs_equal" if equal else "step_guard_pairs_different"] += 1
        if program["positive_net_guard"] is not None:
            net_fit_groups.update(program["source_groups"])
            net_fit_groups.update(program["refined_net_guard"]["negative_groups"])
            used_cycle_keys.add((program["task_family"], tuple(program["actions"])))
            equal = (_guard_contract(program["positive_net_guard"], typed=True)
                     == _guard_contract(program["refined_net_guard"], typed=True))
            counts["net_guard_pairs_equal" if equal else "net_guard_pairs_different"] += 1
    counts["negative_transitions_unused_by_retained_instances"] = (
        counts["usable_negative_transitions"]
        - sum(len(negative_by_action[key]) for key in used_negative_keys)
    )
    counts["negative_programs_unused_by_retained_net_guards"] = (
        counts["usable_negative_programs"]
        - sum(len(negative_by_program[key]) for key in used_cycle_keys)
    )
    counts["retained_instances"] = len(retained)
    counts["rejected_instances"] = sum(row["instance_id"] is not None for row in rejected)
    counts["retained_programs"] = len({row["program_id"] for row in retained})
    counts["rejected_programs"] = len(programs) - counts["retained_programs"]
    return {
        "schema_version": LIBRARY_SCHEMA,
        "semantics": _SEMANTICS,
        "parameters": {
            "minimum_groups": minimum_groups, "max_literals": max_literals,
            "max_clauses": _MAX_CLAUSES, "conjunction_atom_limit": _CONJUNCTION_ATOMS,
            "excluded_groups": sorted(excluded), "negative_kind": "exact_observation_stutter",
            "task_conditioning": _TASK_CONDITIONING,
            "history_prerequisites": _HISTORY_PREREQUISITES,
            "negative_conditioning": _NEGATIVE_CONDITIONING,
            "negative_program_kind": _NEGATIVE_PROGRAM_KIND,
            "negative_program_conditioning": _NEGATIVE_PROGRAM_CONDITIONING,
            "net_working_state": _NET_WORKING_STATE,
            "composed_reentry": _COMPOSED_REENTRY,
        },
        "objectives": dict(_OBJECTIVES),
        "programs": retained,
        "entry_index": {key: sorted(value) for key, value in sorted(entry_index.items())},
        "fit_groups": sorted(fit_groups),
        "net_fit_groups": sorted(net_fit_groups),
        "counts": counts,
        "rejected_programs": sorted(rejected, key=lambda row: (
            row["program_id"], row["instance_id"] or "",
        )),
    }


def _validate_counts(counts: object) -> None:
    if not isinstance(counts, dict):
        raise ValueError("Counts must be a mapping")
    for key, value in counts.items():
        _text(key, "count name")
        _integer(value, key)


def _validate_guard(guard: dict, *, groups: set[str], source_ids: dict[str, str],
                    excluded: set[str], minimum: int, arity: int, typed: bool = False,
                    source_histories: dict[str, set[str]] | None = None,
                    native_actions: set[str] | None = None) -> set[str]:
    _record(guard, {
        "objective", "clauses", "positive_groups", "negative_groups",
        "covered_positive_groups", "accepted_negative_groups", "counts",
    }, "guard")
    _text(guard["objective"], "guard objective summary")
    _validate_counts(guard["counts"])
    if set(_texts(guard["positive_groups"], "guard positive groups")) != groups:
        raise ValueError("Guard positive provenance disagrees with program")
    negative_groups = set(_texts(guard["negative_groups"], "guard negative groups"))
    if negative_groups & excluded:
        raise ValueError("Excluded negatives entered a guard")
    clauses = guard["clauses"]
    if not isinstance(clauses, list) or not 1 <= len(clauses) <= _MAX_CLAUSES:
        raise ValueError("A guard requires one to three supported OR clauses")
    covered, accepted, seen_clauses = set(), set(), set()
    for clause in clauses:
        _record(clause, {
            "literals", "positive_groups", "positive_group_count", "positive_demonstration_ids",
            "positive_transition_count", "negative_precondition_groups",
            "negative_precondition_group_count", "negative_precondition_count", "selection",
        }, "clause")
        literals = clause["literals"]
        if not isinstance(literals, list) or not 1 <= len(literals) <= arity:
            raise ValueError("Invalid clause arity")
        atoms = []
        for literal in literals:
            _record(literal, {"field", "sign"} if typed else {"sign", "tokens"}, "literal")
            if literal["sign"] not in ("+", "-"):
                raise ValueError("Invalid signed literal")
            field = literal["field"] if typed else "observation"
            if field == "history_action":
                _record(literal, {"action"}, "history-action literal")
                _text(literal["action"], "full native history action")
                if "tokens" in literal:
                    raise ValueError("History-action literals use exact action strings, not tokens")
            elif field == "observation":
                _record(literal, {"tokens"}, "observation literal")
                tokens = literal["tokens"]
                if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 2 or any(
                    not isinstance(token, str) or _words(token) != [token] for token in tokens
                )):
                    raise ValueError("Literal tokens must be normalized unigrams or bigrams")
                if typed and "action" in literal:
                    raise ValueError("Observation literals cannot contain history-action payloads")
            else:
                raise ValueError("Unknown net working-state field")
            atoms.append(_literal_atom(literal, typed=typed))
        signature = tuple(sorted(atoms, key=_atom_key))
        if len(set(atoms)) != len(atoms) or signature in seen_clauses:
            raise ValueError("Duplicate literal or clause")
        seen_clauses.add(signature)
        positive = set(_texts(clause["positive_groups"], "clause positive groups"))
        ids = _texts(clause["positive_demonstration_ids"], "clause demonstration IDs")
        if (not set(ids) <= source_ids.keys()
                or positive != {source_ids[demo_id] for demo_id in ids}
                or len(positive) < minimum):
            raise ValueError("Clause lacks independent positive source support")
        for field, sign, value in atoms:
            if field == "history_action":
                if source_histories is None or native_actions is None:
                    raise ValueError("History-action witnesses require native source history")
                for demo_id in ids:
                    before = source_histories[demo_id]
                    after = before | native_actions
                    if ((value[0] in before) != (sign == "-")
                            or (value[0] in after) != (sign == "+")):
                        raise ValueError("History-action literal is not supported by its source transition")
        negative = set(_texts(clause["negative_precondition_groups"], "clause negative groups"))
        if not negative <= negative_groups:
            raise ValueError("Clause negative provenance is inconsistent")
        for field, expected in (
            ("positive_group_count", len(positive)),
            ("positive_transition_count", len(ids)),
            ("negative_precondition_group_count", len(negative)),
        ):
            if _integer(clause[field], field) != expected:
                raise ValueError("Clause support counts disagree with provenance")
        _integer(clause["negative_precondition_count"], "negative precondition count",
                 minimum=len(negative))
        selection = _record(clause["selection"], {
            "new_positive_group_count", "new_negative_precondition_group_count", "objective_gain",
        }, "clause selection")
        _integer(selection["new_positive_group_count"], "new positive groups")
        _integer(selection["new_negative_precondition_group_count"], "new negative groups")
        if type(selection["objective_gain"]) is not int:
            raise ValueError("Selection objective must be an integer, not a probability")
        covered.update(positive)
        accepted.update(negative)
    if (covered != set(_texts(guard["covered_positive_groups"], "covered groups"))
            or accepted != set(_texts(guard["accepted_negative_groups"], "accepted negative groups"))):
        raise ValueError("Guard union provenance is inconsistent")
    return negative_groups


def _validate_library(library: dict) -> None:
    _record(library, {"schema_version", "parameters", "programs", "entry_index", "fit_groups",
                      "net_fit_groups", "objectives", "counts", "rejected_programs", "semantics"},
            "library")
    if library["schema_version"] != LIBRARY_SCHEMA:
        raise ValueError("Unknown fragment library schema")
    objectives = _record(library["objectives"], set(_OBJECTIVES), "objective summaries")
    for variant in _OBJECTIVES:
        _text(objectives[variant], f"{variant} objective summary")
    params = _record(library["parameters"], {
        "minimum_groups", "max_literals", "max_clauses", "conjunction_atom_limit",
        "excluded_groups", "negative_kind", "task_conditioning", "history_prerequisites",
        "negative_conditioning", "negative_program_kind", "negative_program_conditioning",
        "net_working_state", "composed_reentry",
    }, "library parameters")
    minimum = _integer(params["minimum_groups"], "minimum_groups", minimum=1)
    arity = _integer(params["max_literals"], "max_literals", minimum=1, maximum=2)
    if (_integer(params["max_clauses"], "max_clauses") != _MAX_CLAUSES
            or _integer(params["conjunction_atom_limit"], "conjunction_atom_limit")
            != _CONJUNCTION_ATOMS or params["negative_kind"] != "exact_observation_stutter"
            or params["task_conditioning"] != _TASK_CONDITIONING
            or params["history_prerequisites"] != _HISTORY_PREREQUISITES
            or params["negative_conditioning"] != _NEGATIVE_CONDITIONING
            or params["negative_program_kind"] != _NEGATIVE_PROGRAM_KIND
            or params["negative_program_conditioning"] != _NEGATIVE_PROGRAM_CONDITIONING
            or params["net_working_state"] != _NET_WORKING_STATE
            or params["composed_reentry"] != _COMPOSED_REENTRY):
        raise ValueError("Unknown clause selection, entry prerequisites or negative semantics")
    excluded = set(_texts(params["excluded_groups"], "excluded groups"))
    _validate_counts(library["counts"])
    if not isinstance(library["programs"], list) or not isinstance(library["rejected_programs"], list):
        raise ValueError("Library programs must be lists")
    expected_index: dict[str, list[str]] = defaultdict(list)
    seen, fit_groups, net_fit_groups, original_actions = set(), set(), set(), {}
    for program in library["programs"]:
        _record(program, {"program_id", "actions", "examples", "source_groups", "source_tasks",
                          "steps", "counts", "instance_id", "task_text",
                          "history_prerequisite_words", "conditioning", "task_family",
                          "positive_net_guard", "refined_net_guard", "reentry_projection"},
                "compiled program")
        original_id = _text(program["program_id"], "program_id")
        identifier = _text(program["instance_id"], "instance_id")
        family = _family(program["task_family"])
        task_text = program["task_text"]
        if program["conditioning"] == "task_family":
            if family is None or task_text is not None:
                raise ValueError("A family instance requires only its supplied family label")
            condition = "task_family", family
        elif program["conditioning"] == "exact_task_text":
            task_text = _text(task_text, "source task text")
            if family is not None or task_text != _task_text(task_text):
                raise ValueError("Exact-text fallback must be normalized and have no family metadata")
            condition = "exact_task_text", task_text
        else:
            raise ValueError("Unknown instance conditioning")
        if identifier != _instance_id(original_id, condition):
            raise ValueError("Instance identity must match its conditioning and original program")
        if identifier in seen:
            raise ValueError("Duplicate compiled instance")
        seen.add(identifier)
        actions = program["actions"]
        if not isinstance(actions, list) or not 2 <= len(actions) <= 4:
            raise ValueError("Invalid compiled action sequence")
        for action in actions:
            _text(action, "compiled action")
        if original_actions.setdefault(original_id, actions) != actions:
            raise ValueError("Instances of an original program disagree on its action sequence")
        source_groups = set(_texts(program["source_groups"], "source groups"))
        if len(source_groups) < minimum or source_groups & excluded:
            raise ValueError("Insufficient or excluded compiled source groups")
        source_tasks = set(_texts(program["source_tasks"], "source tasks"))
        _validate_counts(program["counts"])
        if not isinstance(program["examples"], list) or not program["examples"]:
            raise ValueError("Missing compiled source examples")
        by_step = [dict() for _ in actions]
        groups, tasks, starts = set(), set(), set()
        for example in program["examples"]:
            _record(example, {"task_id", "group_id", "start_turn", "demonstration_ids",
                              "goal_words", "start_state_sha256", "pre_action_history",
                              "task_text", "task_family"},
                    "compiled source")
            task = _text(example["task_id"], "source task")
            group = _text(example["group_id"], "source group")
            start = _integer(example["start_turn"], "start_turn")
            if len(_action_history(example["pre_action_history"])) > start:
                raise ValueError("Native source history cannot exceed its completed source turns")
            source_task = _text(example["task_text"], "source task text")
            if (source_task != _task_text(source_task)
                    or _condition(_family(example["task_family"]), source_task) != condition):
                raise ValueError("Source conditioning disagrees with its instance")
            ids = _texts(example["demonstration_ids"], "demonstration IDs")
            if len(ids) != len(actions) or ids[0] in starts:
                raise ValueError("Invalid or duplicate compiled source window")
            starts.add(ids[0])
            for step, demo_id in enumerate(ids):
                if demo_id in by_step[step]:
                    raise ValueError("Duplicate step demonstration")
                by_step[step][demo_id] = group
            words = _texts(example["goal_words"], "goal words")
            if words != sorted(set(_words(source_task)) & set(_words(" ".join(actions)))):
                raise ValueError("Goal requirements disagree with the instance task and actions")
            digest = _text(example["start_state_sha256"], "source state hash")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Invalid source state hash")
            groups.add(group)
            tasks.add(task)
            expected_index[ids[0]].append(identifier)
        if groups != source_groups or tasks != source_tasks:
            raise ValueError("Source provenance does not match compiled examples")
        prerequisites = _texts(program["history_prerequisite_words"], "history prerequisite words")
        if prerequisites != _history_prerequisites(program["examples"]):
            raise ValueError("History prerequisites must occur in every retained source history")
        fit_groups.update(groups)
        if not isinstance(program["steps"], list) or len(program["steps"]) != len(actions):
            raise ValueError("Invalid compiled steps")
        for index, step in enumerate(program["steps"]):
            _record(step, {"step", "action", "positive_guard", "refined_guard"}, "compiled step")
            if _integer(step["step"], "step") != index or step["action"] != actions[index]:
                raise ValueError("Compiled step disagrees with action sequence")
            for variant in _OBJECTIVES:
                fit_groups.update(_validate_guard(
                    step[f"{variant}_guard"], groups=groups, source_ids=by_step[index],
                    excluded=excluded, minimum=minimum, arity=arity,
                ))
        if (program["positive_net_guard"] is None) != (program["refined_net_guard"] is None):
            raise ValueError("Positive/refined net guards must share the supported instance pool")
        if program["positive_net_guard"] is not None:
            net_fit_groups.update(groups)
            source_histories = {
                source["demonstration_ids"][0]: set(source["pre_action_history"])
                for source in program["examples"]
            }
            for variant in _OBJECTIVES:
                net_fit_groups.update(_validate_guard(
                    program[f"{variant}_net_guard"], groups=groups, source_ids=by_step[0],
                    excluded=excluded, minimum=minimum, arity=arity,
                    typed=True, source_histories=source_histories, native_actions=set(actions),
                ))
        expected_projection = {
            variant: _reentry_projection(program[f"{variant}_net_guard"], prerequisites)
            for variant in _OBJECTIVES
        }
        if program["reentry_projection"] != expected_projection:
            raise ValueError("Reentry projection must match selected net action facts and prerequisites")
    expected = {key: sorted(value) for key, value in expected_index.items()}
    if not isinstance(library["entry_index"], dict) or library["entry_index"] != expected:
        raise ValueError("Entry index disagrees with source starting demonstrations")
    if set(_texts(library["fit_groups"], "fitted groups")) != fit_groups:
        raise ValueError("Library fitted-group provenance is inconsistent")
    if set(_texts(library["net_fit_groups"], "net fitted groups")) != net_fit_groups:
        raise ValueError("Library net fitted-group provenance is inconsistent")
    try:
        json.dumps(library, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Library must be JSON serializable with finite values") from error


def library_behavior_hash(library: dict) -> str:
    """Return a versioned SHA-256 fingerprint of a validated executable library.

    Includes all declared contract parameters, target-group exclusions, instance
    identities, ordered actions/steps/clauses, entry references, runtime source
    bindings, goal/native-history requirements and both step/net guard variants.
    Clause order is preserved because clause indices appear in runtime logs.
    Program/example ordering and conjunction-literal ordering are canonicalized.

    Fit counts, objective summaries, clause support/selection diagnostics,
    rejected candidates and timestamps are excluded. Raw source history and
    ``start_turn`` are also excluded: their effective prerequisite words and
    runtime source IDs/state hashes are included instead. Source-group output
    order is retained; fitted groups remain included because they reject target
    groups at construction. Diagnostic objective text does not select clauses
    at runtime and need not equal the compiler's current summary wording.

    This audits aliases only for this fixed controller, not general semantic
    equivalence. Memory policy, controller code/arguments and episode state are
    external inputs and must be pinned separately by the caller. This includes
    the training-intervention flag and episode entry-exclusion state/log.
    """
    _validate_library(library)
    parameters = {key: library["parameters"][key] for key in (
        "minimum_groups", "max_literals", "max_clauses", "conjunction_atom_limit",
        "excluded_groups", "negative_kind", "task_conditioning", "history_prerequisites",
        "negative_conditioning", "negative_program_kind", "negative_program_conditioning",
        "net_working_state", "composed_reentry",
    )}
    parameters["excluded_groups"] = sorted(parameters["excluded_groups"])
    programs = []
    for program in sorted(library["programs"], key=lambda row: row["instance_id"]):
        executable = {key: program[key] for key in (
            "program_id", "instance_id", "conditioning", "task_family", "task_text",
            "actions", "source_groups", "history_prerequisite_words",
            "reentry_projection",
        )}
        executable["examples"] = [
            {key: source[key] for key in (
                "task_id", "group_id", "task_family", "task_text", "demonstration_ids",
                "goal_words", "start_state_sha256",
            )}
            for source in sorted(program["examples"], key=lambda row: row["demonstration_ids"][0])
        ]
        steps = []
        for step in program["steps"]:
            executable_step = {"step": step["step"], "action": step["action"]}
            for variant in _OBJECTIVES:
                executable_step[f"{variant}_guard"] = _guard_contract(
                    step[f"{variant}_guard"], preserve_clause_order=True,
                )
            steps.append(executable_step)
        executable["steps"] = steps
        for variant in _OBJECTIVES:
            executable[f"{variant}_net_guard"] = _guard_contract(
                program[f"{variant}_net_guard"], preserve_clause_order=True, typed=True,
            )
        programs.append(executable)
    payload = {
        "hash_schema": LIBRARY_BEHAVIOR_HASH_SCHEMA,
        "library_schema": library["schema_version"],
        "parameters": parameters,
        "fit_groups": sorted(library["fit_groups"]),
        "net_fit_groups": sorted(library["net_fit_groups"]),
        "entry_index": library["entry_index"],
        "programs": programs,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def runtime_contract_hash(library: dict, mode: str) -> str:
    """Fingerprint effective action logic, not mode labels or diagnostic output.

    Resolve the mode to active step/net literal clauses and action selection.
    Include entry references/bindings, history prerequisites, retrieval exclusions,
    and memory/counter/actor-boundary behavior. Canonicalize OR-clause and
    conjunction-literal ordering; logged clause indices are deliberately omitted.
    Instances without a required net guard cannot execute and are omitted.
    The no-net-check ablation retains that same instance pool and fingerprints
    its net-derived reentry projection, but not unenforced net literal clauses.
    Raw-instance reentry and disabled seen memory are distinct policy settings.

    Ignore fit diagnostics/eligibility thresholds, inactive guards, source proof
    records and logging-only metadata. Both libraries must validate and construct
    successfully with the separately pinned memory policy, constructor inputs
    and controller code, with matching source bindings. Only valid action traces
    are compared, not validation-error diagnostics. This is structural action-alias
    auditing, not arbitrary Boolean/program equivalence or native task performance.
    Training entry interventions are external episode state, not library fields:
    pin their flag, ordered intervention log and controller source separately.
    """
    _validate_library(library)
    (step_variant, net_variant, memory_enabled, action_selection,
     net_checks, reentry_scope) = _mode_settings(mode)
    programs, active_ids = [], set()
    for program in sorted(library["programs"], key=lambda row: row["instance_id"]):
        net_guard = None if net_variant is None else program[f"{net_variant}_net_guard"]
        if net_variant is not None and net_guard is None:
            continue
        active_ids.add(program["instance_id"])
        executable = {key: program[key] for key in (
            "program_id", "instance_id", "conditioning", "task_family", "task_text",
            "actions", "history_prerequisite_words",
        )}
        executable["entries"] = [
            {
                "example_id": source["demonstration_ids"][0],
                **{key: source[key] for key in (
                    "task_id", "group_id", "task_family", "task_text", "goal_words",
                    "start_state_sha256",
                )},
            }
            for source in sorted(program["examples"], key=lambda row: row["demonstration_ids"][0])
        ]
        executable["step_guards"] = None if step_variant is None else [
            _guard_contract(step[f"{step_variant}_guard"]) for step in program["steps"]
        ]
        executable["net_guard"] = _guard_contract(net_guard, typed=True) if net_checks else None
        if net_variant is not None and memory_enabled and reentry_scope == _COMPOSED_REENTRY:
            executable["reentry_projection"] = program["reentry_projection"][net_variant]
        programs.append(executable)
    entry_index = {}
    for source, identifiers in library["entry_index"].items():
        active = [identifier for identifier in identifiers if identifier in active_ids]
        if active:
            entry_index[source] = active
    payload = {
        "hash_schema": RUNTIME_CONTRACT_HASH_SCHEMA,
        "library_schema": library["schema_version"],
        "policy": {
            "action_selection": action_selection,
            "step_guard_enabled": step_variant is not None,
            "net_guard_enabled": net_checks,
            "seen_entry_registry": memory_enabled,
            "reentry_scope": reentry_scope,
            "net_working_state": None if net_variant is None else _NET_WORKING_STATE,
            "native_membership_required": True,
            "native_history_only": True,
            "counter_advances_on_observe": True,
            "actor_after_completion_or_abort": True,
            "postconditions_use_same_pre_matched_clauses": True,
        },
        "excluded_groups": sorted(library["parameters"]["excluded_groups"]),
        "entry_index": entry_index,
        "programs": programs,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


class FragmentController:
    """Stateful local-action control; no model calls, hidden state, or observation repairs.

    ``step`` is zero-based and advances only after matching compiled feedback.
    Successful completion and failed checks require actor feedback before any
    further entry. A compiled postcondition may use only the clauses accepted at
    that action's precondition. Independent actions call the existing P0 proposal
    at every local step while retaining the selected fragment's planned length.
    Net-checking composed modes pre-match once at entry and check only those net
    clauses after the final actual local action. The no-net-check ablation retains
    the net-supported pool and positive projection, but reports net checks as
    unchecked with no pass/fail value. net_guard_variant still names the net/
    projection source in that ablation; the checked fields report enforcement.
    Failed checked effects
    abort without undoing actions or deleting their native-history evidence.
    Their reentry key uses instance identity and task/observation/native actions,
    replacing raw public history with the selected full-action/required-word
    projection, except the raw-reentry ablation, which keeps instance identity
    and uses the full raw public-state hash. The no-reentry-memory ablation
    disables only registry reads/writes. Step-only modes keep their v3 key.
    All modes share supplied-family/exact-task conditioning, source goal-word
    bindings and learned native-history prerequisites. Feed every executed
    action through ``observe``, including the initial actor action, and pass its
    actual membership check as ``native_valid``. Public task/observation/history
    text never seeds this registry. Non-native feedback is logged but supplies
    no history prerequisites. Actor boundaries preserve episode history; only
    ``reset_episode`` clears it. Family labels are never inferred or enumerated.
    Training entry exclusions are disabled by default and do not change fitted
    guards, history, counters or boundaries. The caller must restrict opt-in use
    to TRAIN; partition membership is deliberately not inferred from task IDs.
    """

    def __init__(self, library: dict, memory_policy: dict, *, group_id: str,
                 mode: str, entry_threshold: float = 0.55, task_family: str | None = None,
                 allow_training_interventions: bool = False):
        _validate_library(library)
        _text(group_id, "group_id")
        if type(allow_training_interventions) is not bool:
            raise ValueError("allow_training_interventions must be a bool")
        self._allow_training_interventions = allow_training_interventions
        self.task_family = _family(task_family)
        (self._variant, self._net_variant, self._memory_enabled, self._action_selection,
         self._net_checks_enabled, self._reentry_scope) = _mode_settings(mode)
        if (type(entry_threshold) not in (int, float) or not math.isfinite(entry_threshold)
                or not 0 <= entry_threshold <= 1):
            raise ValueError("entry_threshold must be a finite number in [0, 1]")
        _record(memory_policy, {"schema_version", "records", "index"}, "memory policy")
        if (memory_policy["schema_version"] != invocation.POLICY_SCHEMA
                or not isinstance(memory_policy["records"], list) or not memory_policy["records"]
                or not isinstance(memory_policy["index"], dict)
                or memory_policy["index"].get("with_action") is not False
                or not isinstance(memory_policy["index"].get("idf"), dict)
                or not isinstance(memory_policy["index"].get("postings"), dict)):
            raise ValueError("Invalid native-action memory policy")
        if group_id in library["fit_groups"] or (
            self._net_variant is not None and group_id in library["net_fit_groups"]
        ):
            raise ValueError("Target group supplied fitted evidence; recompile with its exclusion")
        self._library = deepcopy(library)
        self._policy = memory_policy
        self.group_id, self.mode, self.entry_threshold = group_id, mode, float(entry_threshold)
        self._excluded = frozenset(library["parameters"]["excluded_groups"]) | {group_id}
        self._programs = {row["instance_id"]: row for row in self._library["programs"]}
        self._sources = {
            (program["instance_id"], source["demonstration_ids"][0]): source
            for program in self._library["programs"] for source in program["examples"]
        }
        self._pending = None
        self.reset_episode()

    def reset_episode(self) -> None:
        """Clear only episode state; pending actual-action feedback must arrive first."""
        if self._pending is not None:
            raise ValueError("Observe the pending action before resetting the episode")
        self._active = None
        self._actor_required = None
        self._seen_entries: set[str] = set()
        self._completed_actions: list[str] = []
        self._native_actions: list[str] = []
        self._native_action_set: set[str] = set()
        self._native_action_words: set[str] = set()
        self._entry_exclusions: dict[str, set[str]] = {}

    def _record_completed_action(self, action: str, *, native_valid: bool) -> None:
        self._completed_actions.append(action)
        if native_valid:
            self._native_actions.append(action)
            self._native_action_set.add(action)
            self._native_action_words.update(_words(action))

    def _entry_memory_key(self, program: dict, public_state_hash: str) -> str:
        if self._reentry_scope == _PUBLIC_STATE_REENTRY:
            return public_state_hash
        if self._reentry_scope == _INSTANCE_RAW_REENTRY:
            return hashlib.sha256(json.dumps({
                "scope": _INSTANCE_RAW_REENTRY, "instance_id": program["instance_id"],
                "public_state": public_state_hash,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return self._working_entry_key(program, public_state_hash)

    def _working_entry_key(self, program: dict, public_state_hash: str) -> str:
        projection = (program["reentry_projection"][self._net_variant]
                      if self._net_variant is not None else {
                          "history_actions": [], "history_words": program["history_prerequisite_words"],
                      })
        projected_state = {
            "scope": _COMPOSED_REENTRY, "instance_id": program["instance_id"],
            "public_state": public_state_hash,
            "history_actions": [action for action in projection["history_actions"]
                                if action in self._native_action_set],
            "history_words": [word for word in projection["history_words"]
                              if word in self._native_action_words],
        }
        return hashlib.sha256(json.dumps(
            projected_state, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()

    def exclude_entry(self, public_state: dict, instance_id: str) -> dict:
        """Register an auditable TRAIN-only exclusion before a new entry decision.

        Requires opt-in, a known instance, no pending decision and no active
        fragment. The key uses task/observation/native actions (not raw public
        history), plus that instance's active net action facts and prerequisite
        word presence. Step-only modes project prerequisite words only.
        All composed ablations retain the positive net-derived projection for
        training exclusions, regardless of net enforcement or seen-memory policy.

        Registration is idempotent and does not require the instance to be the
        otherwise-selected candidate. Existing actor boundaries remain in force.
        Actual matching exclusions are logged as ``training_entry_excluded`` in
        entry_rejections with exclusion_key; other candidates remain eligible.
        Only reset_episode clears exclusions. No actions are undone, and no
        histories, counters, pending actions, guards or actor boundaries change.
        """
        if not self._allow_training_interventions:
            raise ValueError("Training entry interventions are disabled")
        if self._pending is not None:
            raise ValueError("Observe the pending action before excluding an entry")
        if self._active is not None:
            raise ValueError("Cannot exclude an entry mid-fragment")
        _text(instance_id, "instance_id")
        program = self._programs.get(instance_id)
        if program is None:
            raise ValueError("Unknown program instance_id")
        state_hash = invocation.public_state_sha256(public_state)
        working_hash = invocation.public_state_sha256({**public_state, "history": []})
        key = self._working_entry_key(program, working_hash)
        exclusions = self._entry_exclusions.setdefault(instance_id, set())
        already_excluded = key in exclusions
        exclusions.add(key)
        return {
            "reason": _TRAINING_ENTRY_EXCLUDED,
            "program_id": program["program_id"], "instance_id": instance_id,
            "state_sha256": state_hash, "exclusion_key": key,
            "already_excluded": already_excluded,
        }

    def _proposal(self, action: str | None = None) -> dict:
        result = {"action": action}
        if self._active is not None:
            active = self._active
            program, source, step = active["program"], active["source"], active["step"]
            result.update({
                "program_id": program["program_id"], "instance_id": program["instance_id"],
                "step": step, "conditioning": program["conditioning"],
                "task_family": program["task_family"],
                "planned_burst_length": len(program["actions"]),
                "stored_action": program["actions"][step],
                "step_source_demonstration_id": source["demonstration_ids"][step],
                "source_groups": list(program["source_groups"]),
                "entry_source": {
                    "example_id": source["demonstration_ids"][0],
                    "group_id": source["group_id"], "task_id": source["task_id"],
                    "similarity": active["similarity"], "goal_words": list(source["goal_words"]),
                    "task_text": source["task_text"], "task_family": source["task_family"],
                    "history_prerequisite_words": list(program["history_prerequisite_words"]),
                },
            })
        return result

    def _decision(self, state_hash: str, *, commit: bool, reason: str, **details) -> dict:
        return {
            "commit": commit, "reason": reason, "mode": self.mode,
            "guard_variant": self._variant, "precondition_checked": False,
            "matched_precondition_clauses": [], "memory_enabled": self._memory_enabled,
            "net_guard_variant": self._net_variant, "net_precondition_checked": False,
            "matched_net_precondition_clauses": [] if self._active is None else
            list(self._active["net_precondition_clauses"]),
            "task_conditioning": _TASK_CONDITIONING, "history_prerequisites_enabled": True,
            "entry_prerequisites_checked": False,
            "completed_action_count": len(self._completed_actions),
            "native_action_count": len(self._native_actions), "task_family": self.task_family,
            "state_sha256": state_hash,
            "action_selection": self._action_selection,
            **details,
        }

    def _defer(self, state_hash: str, reason: str, *, abort: bool = False, **details) -> dict:
        proposal = self._proposal()
        decision = self._decision(state_hash, commit=False, reason=reason, **details)
        if abort:
            self._active = None
            self._actor_required = reason
        self._pending = {"compiled": False}
        return {"proposal": proposal, "decision": decision}

    def _entry_condition_reason(self, program: dict, source: dict, task: str) -> str | None:
        if program["conditioning"] == "task_family":
            if self.task_family is None:
                return "task_family_required"
            if self.task_family != program["task_family"]:
                return "task_family_mismatch"
        else:
            if self.task_family is not None:
                return "source_task_family_missing"
            if _task_text(task) != program["task_text"]:
                return "source_task_text_mismatch"
        if not set(source["goal_words"]) <= set(_words(task)):
            return "source_goal_words_missing"
        return None

    def _entry(self, state: dict, state_hash: str) -> tuple[bool, list[dict], bool]:
        neighbors = invocation.retrieve_demonstrations(
            self._policy, state, excluded_groups=self._excluded, limit=128,
        )
        candidates = []
        for similarity, demo in neighbors:
            for identifier in self._library["entry_index"].get(demo["example_id"], []):
                program = self._programs[identifier]
                source = self._sources[identifier, demo["example_id"]]
                if (demo["group_id"] != source["group_id"] or demo["task_id"] != source["task_id"]
                        or demo["action"] != program["actions"][0]
                        or _task_text(demo["public_state"]["task"]) != source["task_text"]
                        or invocation.public_state_sha256(demo["public_state"])
                        != source["start_state_sha256"]):
                    raise ValueError("Retrieved source does not match the compiled demonstration")
                if demo["group_id"] not in self._excluded:
                    candidates.append((similarity, program, source))
        candidates.sort(key=lambda item: (
            -item[0], -len(item[1]["actions"]), item[1]["program_id"],
            item[1]["instance_id"], item[2]["demonstration_ids"][0],
        ))
        native = set(state["actions"])
        projected_state_hash = state_hash
        if self._net_variant is not None or self._entry_exclusions:
            projected_state_hash = invocation.public_state_sha256({**state, "history": []})
        memory_state_hash = (projected_state_hash if self._reentry_scope == _COMPOSED_REENTRY
                             else state_hash)
        rejections = []
        net_checked = False
        for similarity, program, source in candidates:
            reason = None
            net_matched = []
            entry_memory_key = self._entry_memory_key(program, memory_state_hash)
            exclusion_key = None
            exclusions = self._entry_exclusions.get(program["instance_id"])
            if self._allow_training_interventions and exclusions:
                key = self._working_entry_key(program, projected_state_hash)
                if key in exclusions:
                    exclusion_key = key
            missing_history = sorted(
                set(program["history_prerequisite_words"]) - self._native_action_words,
            )
            condition_reason = self._entry_condition_reason(program, source, state["task"])
            if exclusion_key is not None:
                reason = _TRAINING_ENTRY_EXCLUDED
            elif similarity < self.entry_threshold:
                reason = "below_entry_threshold"
            elif program["actions"][0] not in native:
                reason = "first_action_not_native"
            elif condition_reason is not None:
                reason = condition_reason
            elif missing_history:
                reason = "history_prerequisites_missing"
            elif self._variant is not None and not _matching(
                program["steps"][0][f"{self._variant}_guard"], state["observation"], post=False,
            ):
                reason = "entry_precondition_failed"
            elif self._net_variant is not None:
                net_guard = program[f"{self._net_variant}_net_guard"]
                if net_guard is None:
                    reason = "no_supported_net_guard"
                elif self._net_checks_enabled:
                    net_checked = True
                    net_matched = _matching(
                        net_guard, state["observation"], post=False, typed=True,
                        native_actions=self._native_action_set,
                    )
                    if not net_matched:
                        reason = "entry_net_precondition_failed"
            if (reason is None and self._memory_enabled and self._net_variant is not None
                    and entry_memory_key in self._seen_entries):
                reason = "repeated_fragment_entry_state"
            if reason:
                rejections.append({
                    "program_id": program["program_id"], "instance_id": program["instance_id"],
                    "task_family": program["task_family"],
                    "source_example_id": source["demonstration_ids"][0], "reason": reason,
                    "missing_history_prerequisite_words": missing_history,
                    **({"exclusion_key": exclusion_key} if reason == _TRAINING_ENTRY_EXCLUDED else {}),
                })
                continue
            self._active = {
                "program": program, "source": source, "step": 0, "similarity": similarity,
                "entry_state_sha256": state_hash,
                "entry_memory_key": entry_memory_key,
                "net_precondition_clauses": net_matched,
            }
            return True, rejections, net_checked
        return False, rejections, net_checked

    def decide(self, public_state: dict) -> dict:
        """Return a native proposal or a logged deferral; never execute or advance here."""
        if self._pending is not None:
            raise ValueError("observe must follow decide before another decision")
        state_hash = invocation.public_state_sha256(public_state)
        if self._actor_required is not None:
            return self._defer(state_hash, "actor_boundary_required",
                               boundary_reason=self._actor_required)
        entered, rejections, net_checked = False, [], False
        if self._active is None:
            if (self._memory_enabled and self._reentry_scope == _PUBLIC_STATE_REENTRY
                    and state_hash in self._seen_entries):
                return self._defer(state_hash, "repeated_fragment_entry_state")
            entered, rejections, net_checked = self._entry(public_state, state_hash)
            if not entered:
                repeated = bool(rejections) and all(
                    row["reason"] == "repeated_fragment_entry_state" for row in rejections
                )
                return self._defer(state_hash, "repeated_fragment_entry_state" if repeated
                                   else "no_eligible_fragment", entry_rejections=rejections,
                                   entry_guards_used=self._variant is not None,
                                   entry_prerequisites_checked=True,
                                   net_precondition_checked=net_checked)
        active = self._active
        program, step = active["program"], active["step"]
        condition_reason = self._entry_condition_reason(program, active["source"], public_state["task"])
        if condition_reason is not None:
            return self._defer(
                state_hash, "active_" + condition_reason.removeprefix("source_"), abort=True,
            )
        independent = None
        if self._action_selection == "independent_policy":
            independent = invocation.propose(self._policy, public_state, excluded_groups=self._excluded)
            action = independent["action"]
        else:
            action = program["actions"][step]
        if action is None or action not in public_state["actions"]:
            return self._defer(
                state_hash, "no_native_independent_action" if independent is not None
                else "fragment_action_not_native", abort=True, entry_rejections=rejections,
            )
        matched = []
        if self._variant is not None:
            matched = _matching(program["steps"][step][f"{self._variant}_guard"],
                                public_state["observation"], post=False)
            if not matched:
                return self._defer(state_hash, "precondition_failed", abort=True,
                                   precondition_checked=True)
        proposal = self._proposal(action)
        if independent is not None:
            proposal["independent_proposal"] = deepcopy(independent)
        self._pending = {
            "compiled": True, "action": action, "matched_precondition_clauses": matched,
        }
        return {
            "proposal": proposal,
            "decision": self._decision(
                state_hash, commit=True, reason="fragment_entry" if entered else "fragment_continuation",
                precondition_checked=self._variant is not None,
                matched_precondition_clauses=list(matched), entry_rejections=rejections,
                entry_guards_used=self._variant is not None,
                entry_prerequisites_checked=entered,
                net_precondition_checked=entered and net_checked,
            ),
        }

    def observe(self, action: str, observation: str, *, compiled: bool,
                native_valid: bool = True) -> dict:
        """Log actual feedback; only caller-validated native actions add history facts.

        Initial/consecutive actor feedback needs no preceding decision. A
        matching compiled action reported non-native aborts and requires actor
        feedback, even if its observation happens to match a learned witness.
        """
        if type(compiled) is not bool:
            raise ValueError("compiled must be a bool")
        if type(native_valid) is not bool:
            raise ValueError("native_valid must be a bool")
        _text(action, "executed action", empty=not compiled and not native_valid)
        _text(observation, "actual observation", empty=True)
        actual = {
            "action": action, "observation": observation, "native_valid": native_valid,
            "history_recorded": native_valid,
            "net_guard_variant": self._net_variant,
            "net_postcondition_checked": False, "net_postcondition_passed": None,
            "matched_net_precondition_clauses": [], "matched_net_postcondition_clauses": [],
        }
        if not compiled:
            self._record_completed_action(action, native_valid=native_valid)
            interrupted = self._active is not None
            boundary = self._actor_required
            self._active, self._pending, self._actor_required = None, None, None
            return {
                **actual,
                "compiled": False, "reason": "actor_action_not_native" if not native_valid
                else "actor_interrupted_fragment" if interrupted
                else "actor_feedback", "postcondition_checked": False,
                "postcondition_passed": None, "fragment_complete": False,
                "fragment_aborted": interrupted, "cleared_boundary": boundary,
                "actor_required": False, "seen_entry_count": len(self._seen_entries),
                "completed_action_count": len(self._completed_actions),
                "native_action_count": len(self._native_actions),
            }
        if self._pending is None or not self._pending["compiled"] or self._active is None:
            raise ValueError("Compiled observation requires a pending committed action")
        if action != self._pending["action"]:
            raise ValueError("Executed compiled action does not match the pending action")
        self._record_completed_action(action, native_valid=native_valid)
        active, pending = self._active, self._pending
        program, step = active["program"], active["step"]
        if self._memory_enabled and step == 0:
            self._seen_entries.add(active["entry_memory_key"])
        matched = []
        passed = None
        checked = self._variant is not None and native_valid
        if checked:
            matched = _matching(
                program["steps"][step][f"{self._variant}_guard"], observation, post=True,
                allowed=pending["matched_precondition_clauses"],
            )
            passed = bool(matched)
        final_step = step + 1 == len(program["actions"])
        net_checked = native_valid and final_step and self._net_checks_enabled
        net_matched, net_passed = [], None
        if net_checked:
            net_matched = _matching(
                program[f"{self._net_variant}_net_guard"], observation, post=True,
                allowed=active["net_precondition_clauses"],
                typed=True, native_actions=self._native_action_set,
            )
            net_passed = bool(net_matched)
        aborted = not native_valid or passed is False or net_passed is False
        complete = not aborted and final_step
        reason = ("compiled_action_not_native" if not native_valid else
                  "postcondition_failed" if passed is False else
                  "net_postcondition_failed" if net_passed is False else
                  "fragment_completed" if complete else "fragment_step_observed")
        self._pending = None
        if aborted or complete:
            self._active = None
            self._actor_required = reason
        else:
            active["step"] += 1
        return {
            **actual,
            "compiled": True, "program_id": program["program_id"],
            "instance_id": program["instance_id"], "step": step,
            "planned_burst_length": len(program["actions"]), "executed_steps": step + 1,
            "next_step": None if aborted or complete else active["step"],
            "reason": reason,
            "guard_variant": self._variant, "postcondition_checked": checked,
            "postcondition_passed": passed,
            "matched_precondition_clauses": list(pending["matched_precondition_clauses"]),
            "matched_postcondition_clauses": matched,
            "net_postcondition_checked": net_checked, "net_postcondition_passed": net_passed,
            "matched_net_precondition_clauses": list(active["net_precondition_clauses"]),
            "matched_net_postcondition_clauses": net_matched,
            "fragment_complete": complete, "fragment_aborted": aborted,
            "actor_required": aborted or complete, "seen_entry_count": len(self._seen_entries),
            "completed_action_count": len(self._completed_actions),
            "native_action_count": len(self._native_actions),
        }
