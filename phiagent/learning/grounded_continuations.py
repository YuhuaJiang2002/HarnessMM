"""Parameter-bound, actor-anchored ONE-step continuations; an alternative to R1.

Only two-step source examples and their immutable policy demonstrations are used.
Literal-program support counts, STEP/NET guards and history prerequisites are NOT
imported. Sources are deduplicated by starting demonstration ID and target/declared
excluded groups are removed before source-native vocabulary and index fitting.
The NEW grounded action still requires unanimous eligible sources and at least
two independent groups. No outcome, task ID or group ID is a predictive feature.

Inputs may be the original schema-v5 library or an explicit
chisel-two-step-program-evidence/v1 raw catalog with programs, sorted unique
source_training_ids and source_corpus_sha256. Raw catalogs have no literal-support
admission. Program/example keys are strict; identity conflicts are errors.
Raw chronology is verified against complete public histories or the original
policy's last-two histories with SHA-256([task_id, turn]) demonstration IDs.
The latter digest uses sorted-key, compact, UTF-8 JSON with ensure_ascii=False.
SHA-256-shaped IDs are always checked against that native task/turn convention;
opaque IDs require complete histories. This is provenance validation, not text
normalization or a predictive use of identifiers.
Unverifiable pairs are errors, never assumed consecutive. Source corpus content
verification remains with the parent; its declared digest/manifest are provenance,
not features. The original v5 extraction/checks and bank hashes remain unchanged.

Goal roles are ordered distinct native entity TYPES mentioned in the task.
Repeated mentions are accepted only when all explicitly name the same instance;
otherwise same-type role ambiguity abstains. Actor syntax, positional roles,
non-goal types and same/different entity identity form the retrieval index key.
The actor command supplies a bijection; remaining next-action entities require a
unique currently native entity of the corresponding role/type. Rendering must
produce an EXACT current native action. No verb, object or family is hardcoded.

Cosine TF-IDF uses canonical POST-actor task/observation unigrams and bigrams,
preserving non-entity words and anchor-relative identities. Unknown query features
remain in the norm. Indices are cached per family/anchor shape. At most 128
similarity-qualified, grounded, eligible sources participate in unanimity.
Full mode additionally requires equal goal-relative native-template sets.
The no-affordance mode removes ONLY this equality check, not binding or grounding.

context_partitioned_grounded_continuation first selects an observable-context
leaf within the same native family/actor shape. Its TRAIN predicates are canonical
observation unigram/bigram, goal-affordance and verified public-history template
presence. Canonical teacher NEXT templates are labels only, never features.
Each node gives each group total weight one; child impurity is weighted using
those parent weights, and weights are recomputed within a recursive child.
Feature selection uses root group-balanced presence variance, lexical ties,
and a fixed cap of 64. Splits require strictly positive Gini reduction and two
groups in BOTH children; depth is capped at three. Fractions make ties exact.
Leaves return source IDs, not a majority action. The original actor-shape TF-IDF
weights are retained and only leaf sources undergo the existing runtime checks.
Conflicting/unsupported leaves still defer. This is an empirical partition, not
a safety certificate. context_partition_artifacts() returns the fitted audit
trees, predicates, support counts and rejected splits as independent JSON data.

Source history facts require a matching native first-action/post-observation
pair at that past task/turn in the SAME filtered bank. Runtime history facts
require matching native-valid observe feedback at that past position. Missing,
unverified and malformed history never supplies a native-action predicate.
Task/turn/group identifiers verify provenance only; they are not predicates.

decide precedes every actual action. Native actor feedback saves an anchor;
one committed continuation consumes it and forces another actor boundary.
Empty/non-native actor feedback counts as an actual step but clears the anchor.
No method calls an actor or assumes that a demonstrated action is task-optimal.
Pin source, source_bank_hash, constructor settings and action/feedback logs for
replay. This module does not read datasets, fit outcome labels, or wire a runner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import re

from phiagent.learning.counterfactual_invocation import (
    POLICY_SCHEMA, public_state_sha256, validate_public_state,
)


CONTROLLER_SCHEMA = "chisel-grounded-one-step-continuations/v1"
PARTITIONED_CONTROLLER_SCHEMA = "chisel-grounded-one-step-continuations/v2"
CONTEXT_PARTITION_SCHEMA = "chisel-group-balanced-context-partition/v1"
SOURCE_BANK_SCHEMA = "chisel-two-step-native-source-bank/v1"
V5_SOURCE_SCHEMA = "chisel-observable-fragment-contracts/v5"
RAW_EVIDENCE_SCHEMA = "chisel-two-step-program-evidence/v1"
PARTITIONED_MODE = "context_partitioned_grounded_continuation"
MODES = ("grounded_continuation", "grounded_continuation_no_affordance_check", PARTITIONED_MODE)
MAX_SOURCES = 128
_PARTITION_DEPTH = 3
_PARTITION_FEATURES = 64
_PARTITION_GROUPS = 2
_ENTITY = re.compile(r"(?<!\w)(?P<type>[^\W\d]\w*(?:-\w+)*)\s+(?P<id>[0-9]+)(?!\w)")
_WORDS = re.compile(r"\w+(?:-\w+)*")
_TOKENS = re.compile(r"\w+(?:-\w+)*|[^\w\s]")


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else ''}string")
    return value


def _strings(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    for item in value:
        _text(item, name)
    return value


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _validate_raw_library(library: dict) -> dict:
    if set(library) != {"schema_version", "programs", "source_training_ids", "source_corpus_sha256"}:
        raise ValueError("Invalid raw evidence library keys")
    training_ids = _strings(library["source_training_ids"], "source_training_ids")
    if training_ids != sorted(set(training_ids)):
        raise ValueError("source_training_ids must be sorted and unique")
    corpus_hash = _text(library["source_corpus_sha256"], "source_corpus_sha256")
    if re.fullmatch(r"[0-9a-fA-F]{64}", corpus_hash) is None:
        raise ValueError("source_corpus_sha256 must contain 64 hexadecimal characters")
    declared, programs, tasks, positions, identities = set(training_ids), set(), {}, {}, {}
    for program in library["programs"]:
        if not isinstance(program, dict) or set(program) != {"program_id", "actions", "examples"}:
            raise ValueError("Invalid raw program keys")
        identifier = _text(program["program_id"], "raw program_id")
        if identifier in programs:
            raise ValueError("Duplicate raw program_id")
        programs.add(identifier)
        actions = _strings(program["actions"], "raw actions")
        if len(actions) != 2 or not isinstance(program["examples"], list) or not program["examples"]:
            raise ValueError("Raw programs require exactly two actions and nonempty source examples")
        for example in program["examples"]:
            if not isinstance(example, dict) or set(example) != {
                "task_id", "group_id", "start_turn", "demonstration_ids", "task_family",
            }:
                raise ValueError("Invalid raw example keys")
            task = _text(example["task_id"], "raw task_id")
            group = _text(example["group_id"], "raw group_id")
            family = _text(example["task_family"], "raw task_family")
            start = example["start_turn"]
            ids = _strings(example["demonstration_ids"], "raw demonstration_ids")
            if type(start) is not int or start < 0:
                raise ValueError("Raw start_turn must be a nonnegative integer")
            if task not in declared or len(ids) != 2 or ids[0] == ids[1]:
                raise ValueError("Raw examples require declared TRAIN tasks and two distinct demo IDs")
            if tasks.setdefault(task, (group, family)) != (group, family):
                raise ValueError("Raw task group/family identities conflict")
            for offset, demo_id in enumerate(ids):
                position = task, start + offset
                identity = task, group, family, start + offset, actions[offset]
                if (positions.setdefault(position, demo_id) != demo_id
                        or identities.setdefault(demo_id, identity) != identity):
                    raise ValueError("Raw demonstration identities/positions conflict")
    return {
        "source_training_ids_sha256": _hash(training_ids),
        "source_training_task_count": len(training_ids),
        "source_corpus_sha256": corpus_hash.lower(),
    }


def _validate_raw_pair(example: dict, actions: list[str], before: dict, after: dict) -> str:
    start = example["start_turn"]
    expected_ids = [_hash([example["task_id"], start]), _hash([example["task_id"], start + 1])]
    ids = example["demonstration_ids"]
    if any(re.fullmatch(r"[0-9a-fA-F]{64}", identifier) for identifier in ids) and ids != expected_ids:
        raise ValueError("Raw hashed demonstration IDs disagree with task/start_turn chronology")
    pre_history, post_history = before["history"], after["history"]
    expected = pre_history + [{"action": actions[0], "observation": after["observation"]}]
    if pre_history and pre_history[-1]["observation"] != before["observation"]:
        raise ValueError("Raw pre-state observation disagrees with its completed history")
    if len(pre_history) == start and len(post_history) == start + 1:
        if post_history != expected:
            raise ValueError("Raw source pair is not consecutive in its full public history")
        return "raw_full_history_examples"
    if (len(pre_history) == min(start, 2) and len(post_history) == min(start + 1, 2)
            and post_history == expected[-2:]
            and ids == expected_ids):
        return "raw_hashed_window_examples"
    raise ValueError("Raw pair lacks verifiable consecutive full-history or hashed last-two history")


def _parse(text: str) -> tuple[list, dict]:
    parts, surfaces, end = [], defaultdict(set), 0
    for match in _ENTITY.finditer(text):
        entity = match["type"].casefold(), match["id"]
        parts.extend([text[end:match.start()], entity])
        surfaces[entity].add(match[0])
        end = match.end()
    parts.append(text[end:])
    return parts, dict(surfaces)


def _roles(task: str, vocabulary: set[str]) -> tuple[list[str], str | None]:
    tokens, mentions, order = _TOKENS.findall(task.casefold()), defaultdict(list), []
    for index, token in enumerate(tokens):
        if token not in vocabulary:
            continue
        if token not in mentions:
            order.append(token)
        number = tokens[index + 1] if index + 1 < len(tokens) else ""
        mentions[token].append(number if re.fullmatch(r"[0-9]+", number) else None)
    if not order:
        return [], "unbound_goal_roles"
    if any(len(values) > 1 and (None in values or len(set(values)) != 1)
           for values in mentions.values()):
        return [], "ambiguous_same_type_goal_roles"
    return order, None


def _context(task: str, action: str, vocabulary: set[str]) -> tuple[dict | None, str | None]:
    roles, reason = _roles(task, vocabulary)
    if reason is not None:
        return None, reason
    parts, surfaces = _parse(action)
    if any(kind not in vocabulary and kind in set(_WORDS.findall(task.casefold()))
           for kind, _ in surfaces):
        return None, "unknown_goal_type"
    if any(len(spellings) != 1 for spellings in surfaces.values()):
        return None, "ambiguous_actor_entity_spelling"
    entities = list(surfaces)
    context = {
        "roles": roles, "parts": parts, "entities": entities,
        "surfaces": {entity: next(iter(values)) for entity, values in surfaces.items()},
        "anchors": {entity: index for index, entity in enumerate(entities)},
    }
    shape = {"role_count": len(roles), "actor": _canonical_action(parts, context)}
    return {**context, "shape": shape, "shape_id": _hash(shape)}, None


def _entity_token(entity: tuple[str, str], context: dict, others: dict) -> tuple:
    kind, _ = entity
    identity = (("anchor", context["anchors"][entity]) if entity in context["anchors"]
                else ("other", others.setdefault(entity, len(others))))
    relation = (("goal_role", context["roles"].index(kind)) if kind in context["roles"]
                else ("type", kind))
    return *identity, *relation


def _canonical_action(parts: list, context: dict) -> list:
    others = {}
    return [part if isinstance(part, str) else _entity_token(part, context, others) for part in parts]


def _canonical_text(text: str, context: dict, vocabulary: set[str], *, task: bool) -> list[tuple]:
    tokens, others = [], {}
    for part in _parse(text)[0]:
        if isinstance(part, tuple) and part[0] in vocabulary:
            if task and part[0] in context["roles"]:
                tokens.append(("goal_type", context["roles"].index(part[0])))
            else:
                tokens.append(_entity_token(part, context, others))
        else:
            words = _WORDS.findall(part.casefold()) if isinstance(part, str) else list(part)
            tokens.extend(
                ("goal_type", context["roles"].index(word)) if word in context["roles"]
                else ("word", word) for word in words
            )
    return tokens


def _features(state: dict, context: dict, vocabulary: set[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for field in ("task", "observation"):
        tokens = _canonical_text(state[field], context, vocabulary, task=field == "task")
        counts.update(_json([field, "one", token]) for token in tokens)
        counts.update(_json([field, "two", left, right]) for left, right in zip(tokens, tokens[1:]))
    return {key: 1.0 + math.log(count) for key, count in sorted(counts.items())}


def _affordances(state: dict, context: dict) -> set[str]:
    templates = set()
    for action in state["actions"]:
        parts, entities = _parse(action)
        if any(kind in context["roles"] for kind, _ in entities):
            templates.add(_json(_canonical_action(parts, context)))
    return templates


def _partition_features(state: dict, context: dict, vocabulary: set[str],
                        verified_history: list[bool]) -> set[str]:
    features = {
        key for key in _features(state, context, vocabulary) if key.startswith('["observation",')
    }
    features.update(_json(["affordance", json.loads(template)])
                    for template in _affordances(state, context))
    for index, (record, verified) in enumerate(zip(state["history"], verified_history)):
        if verified:
            field = "history.last" if index == len(state["history"]) - 1 else "history.prior"
            features.add(_json([field, _canonical_action(_parse(record["action"])[0], context)]))
    return features


def _weighted_labels(rows: list[dict], group_sizes: Counter) -> dict:
    weights = defaultdict(Fraction)
    for row in rows:
        weights[row["partition_label"]] += Fraction(1, group_sizes[row["group_id"]])
    return dict(weights)


def _gini(weights: dict) -> Fraction:
    total = sum(weights.values())
    return Fraction(0) if not total else 1 - sum((weight / total) ** 2 for weight in weights.values())


def _fit_context_partition(rows: list[dict], family: str, shape_id: str) -> dict:
    group_sizes = Counter(row["group_id"] for row in rows)
    feature_groups = defaultdict(Counter)
    for row in rows:
        for feature in row["partition_features"]:
            feature_groups[feature][row["group_id"]] += 1
    selectable, rejected_features = [], []
    for feature, presence in sorted(feature_groups.items()):
        true_groups = set(presence)
        false_groups = {group for group, count in group_sizes.items() if presence[group] < count}
        support = {"predicate": feature, "absent_groups": sorted(false_groups),
                   "present_groups": sorted(true_groups)}
        if min(len(true_groups), len(false_groups)) < _PARTITION_GROUPS:
            rejected_features.append({**support, "reason": "child_group_support"})
            continue
        probability = sum(Fraction(presence[group], count) for group, count in group_sizes.items())
        probability /= len(group_sizes)
        selectable.append((probability * (1 - probability), feature, support))
    selectable.sort(key=lambda item: (-item[0], item[1]))
    features = sorted(item[1] for item in selectable[:_PARTITION_FEATURES])
    rejected_features.extend({**support, "reason": "candidate_feature_cap"}
                             for _, _, support in selectable[_PARTITION_FEATURES:])

    def build(members: list[dict], depth: int, identifier: str) -> dict:
        sizes = Counter(row["group_id"] for row in members)
        labels = _weighted_labels(members, sizes)
        impurity = _gini(labels)
        node = {
            "node_id": identifier, "depth": depth, "source_count": len(members),
            "source_ids": sorted(row["source_id"] for row in members),
            "support_groups": sorted(sizes), "group_count": len(sizes),
            "impurity": float(impurity),
            "label_weights": [{"template": label, "weight": float(weight)}
                              for label, weight in sorted(labels.items())],
            "predicate": None, "gain": None, "gain_fraction": None,
            "rejected_splits": [], "absent": None, "present": None, "stop_reason": None,
        }
        if not impurity or depth >= _PARTITION_DEPTH:
            node["stop_reason"] = "pure_labels" if not impurity else "depth_cap"
            return node
        possible = []
        for feature in features:
            left = [row for row in members if feature not in row["partition_features"]]
            right = [row for row in members if feature in row["partition_features"]]
            left_groups = sorted({row["group_id"] for row in left})
            right_groups = sorted({row["group_id"] for row in right})
            audit = {
                "predicate": feature, "absent_groups": left_groups, "present_groups": right_groups,
                "absent_source_count": len(left), "present_source_count": len(right), "gain": None,
            }
            if min(len(left_groups), len(right_groups)) < _PARTITION_GROUPS:
                node["rejected_splits"].append({**audit, "reason": "child_group_support"})
                continue
            left_labels, right_labels = _weighted_labels(left, sizes), _weighted_labels(right, sizes)
            remaining = (
                sum(left_labels.values()) * _gini(left_labels)
                + sum(right_labels.values()) * _gini(right_labels)
            ) / len(sizes)
            gain = impurity - remaining
            audit["gain"] = float(gain)
            if gain <= 0:
                node["rejected_splits"].append({**audit, "reason": "no_strict_impurity_reduction"})
            else:
                possible.append((gain, feature, left, right, audit))
        if not possible:
            node["stop_reason"] = "no_supported_strict_improvement"
            return node
        possible.sort(key=lambda item: (-item[0], item[1]))
        gain, feature, left, right, _ = possible[0]
        node["rejected_splits"].extend(
            {**audit, "reason": "lower_gain_or_lexical_tie"} for _, _, _, _, audit in possible[1:]
        )
        node.update(predicate=feature, gain=float(gain), gain_fraction=[gain.numerator, gain.denominator],
                    absent=build(left, depth + 1, identifier + ".0"),
                    present=build(right, depth + 1, identifier + ".1"))
        return node

    artifact = {
        "schema_version": CONTEXT_PARTITION_SCHEMA, "task_family": family, "anchor_shape_id": shape_id,
        "parameters": {
            "max_depth": _PARTITION_DEPTH, "candidate_feature_limit": _PARTITION_FEATURES,
            "minimum_child_groups": _PARTITION_GROUPS,
            "impurity": "group_balanced_gini", "feature_selection": "balanced_presence_variance_then_lexical",
            "split_weighting": "parent_group_weights_then_child_rebalance",
        },
        "candidate_features": features,
        "rejected_features": sorted(rejected_features, key=lambda row: row["predicate"]),
        "tree": build(rows, 0, "root"),
    }
    return {**artifact, "artifact_hash": _hash(artifact)}


def _partition_leaf(artifact: dict, features: set[str]) -> tuple[dict, list[dict]]:
    node, path = artifact["tree"], []
    while node["predicate"] is not None:
        present = node["predicate"] in features
        path.append({"node_id": node["node_id"], "predicate": node["predicate"], "present": present})
        node = node["present" if present else "absent"]
    return node, path


def _available(state: dict) -> dict:
    entities = defaultdict(set)
    for action in state["actions"]:
        for entity, spellings in _parse(action)[1].items():
            entities[entity].update(spellings)
    return dict(entities)


def _ground(source: dict, current: dict, available: dict) -> tuple[str | None, list[dict], str | None]:
    original = source["context"]
    if original["shape_id"] != current["shape_id"]:
        return None, [], "incompatible_actor_binding"
    binding, reverse, log = {}, {}, []

    def bind(old, new, surface, origin):
        if (old in binding and binding[old][0] != new) or (new in reverse and reverse[new] != old):
            return False
        binding[old], reverse[new] = (new, surface), old
        log.append({
            "source_entity": list(old), "current_entity": list(new), "origin": origin,
            "goal_role": original["roles"].index(old[0]) if old[0] in original["roles"] else None,
        })
        return True

    for old, new in zip(original["entities"], current["entities"]):
        if not bind(old, new, current["surfaces"][new], "actor_argument"):
            return None, log, "inconsistent_actor_bijection"
    for entity in source["next_entities"]:
        if entity in binding:
            continue
        kind = (current["roles"][original["roles"].index(entity[0])]
                if entity[0] in original["roles"] else entity[0])
        choices = [candidate for candidate in available if candidate[0] == kind]
        if len(choices) != 1:
            return None, log, "unbound_next_entity" if not choices else "ambiguous_next_entity"
        target = choices[0]
        if len(available[target]) != 1:
            return None, log, "ambiguous_native_entity_spelling"
        if not bind(entity, target, next(iter(available[target])), "unique_current_native_entity"):
            return None, log, "non_bijective_next_argument"
    action = "".join(part if isinstance(part, str) else binding[part][1] for part in source["next_parts"])
    return action, log, None


class GroundedContinuationController:
    """Native actor anchor -> at most one grounded, source-supported local action."""

    def __init__(self, library: dict, memory_policy: dict, *, group_id: str, task_family: str,
                 mode: str = "grounded_continuation", entry_threshold: float = 0.55,
                 minimum_groups: int = 2):
        _text(group_id, "group_id")
        _text(task_family, "task_family")
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError("Unknown grounded continuation mode")
        if type(minimum_groups) is not int or minimum_groups < 2:
            raise ValueError("minimum_groups must be an integer of at least two")
        if (type(entry_threshold) not in (int, float) or not math.isfinite(entry_threshold)
                or not 0 <= entry_threshold <= 1):
            raise ValueError("entry_threshold must be finite and within [0, 1]")
        if (not isinstance(library, dict) or library.get("schema_version") not in (
            V5_SOURCE_SCHEMA, RAW_EVIDENCE_SCHEMA,
        ) or not isinstance(library.get("programs"), list)):
            raise ValueError("Expected a schema-v5 library or raw two-step evidence library")
        self._raw = library["schema_version"] == RAW_EVIDENCE_SCHEMA
        source_provenance = {"schema_version": library["schema_version"]}
        if self._raw:
            source_provenance.update(_validate_raw_library(library))
            excluded = []
        else:
            if not isinstance(library.get("parameters"), dict):
                raise ValueError("Expected schema-v5 library parameters")
            excluded = _strings(library["parameters"].get("excluded_groups"), "excluded_groups")
        if (not isinstance(memory_policy, dict) or memory_policy.get("schema_version") != POLICY_SCHEMA
                or not isinstance(memory_policy.get("records"), list)):
            raise ValueError("Expected native demonstration policy records")
        self.group_id, self.task_family, self.mode = group_id, task_family, mode
        self._partitioned = mode == PARTITIONED_MODE
        self._source_positions = defaultdict(set)
        self.entry_threshold, self.minimum_groups = float(entry_threshold), minimum_groups
        self._excluded = set(excluded) | {group_id}
        records = {}
        for record in memory_policy["records"]:
            if not isinstance(record, dict):
                raise ValueError("Invalid demonstration record")
            identifier = _text(record.get("example_id"), "example_id")
            if identifier in records:
                raise ValueError("Duplicate policy demonstration ID")
            records[identifier] = record
        self._bank, self._bank_counts = self._source_bank(library, records)
        bank_payload = {
            "schema_version": SOURCE_BANK_SCHEMA, "sources": self._bank,
            "excluded_groups": sorted(self._excluded), "task_family": task_family,
        }
        if self._raw:
            bank_payload["source_evidence"] = source_provenance
        self._bank_hash = _hash(bank_payload)
        self._source_evidence = {
            **source_provenance,
            "evidence_hash": _hash({**source_provenance, "source_bank_hash": self._bank_hash}),
            "counts": {
                "schema_v5_programs": 0 if self._raw else len(library["programs"]),
                "raw_two_step_programs": len(library["programs"]) if self._raw else 0,
            },
        }
        self._vocabulary = {
            entity[0] for row in self._bank for field in ("pre_state", "next_state")
            for command in row[field]["actions"] for entity in _parse(command)[1]
        }
        self._vocabulary_hash = _hash(sorted(self._vocabulary))
        self._index, self._cache, self._bank_reasons = defaultdict(list), {}, Counter()
        for row in self._bank:
            context, reason = _context(row["pre_state"]["task"], row["actor_action"], self._vocabulary)
            if reason is not None:
                self._bank_reasons[reason] += 1
                continue
            parts, entities = _parse(row["next_action"])
            self._index[task_family, context["shape_id"]].append({
                **row, "context": context, "next_parts": parts, "next_entities": list(entities),
                "features": _features(row["next_state"], context, self._vocabulary),
                "affordances": _affordances(row["next_state"], context),
            })
        self._bank_counts["indexed_sources"] = sum(len(rows) for rows in self._index.values())
        self._partitions = {}
        if self._partitioned:
            native_effects = defaultdict(set)
            for row in self._bank:
                positions = self._source_positions[row["source_id"]]
                if len(positions) == 1:
                    group, task, turn = next(iter(positions))
                    native_effects[group, task, turn - 1].add((
                        row["actor_action"], row["next_state"]["observation"],
                    ))
            for key, rows in sorted(self._index.items()):
                for row in rows:
                    history = row["next_state"]["history"]
                    verified = [False] * len(history)
                    positions = self._source_positions[row["source_id"]]
                    if len(positions) == 1:
                        group, task, turn = next(iter(positions))
                        for index, step in enumerate(history):
                            proof = native_effects.get((group, task, turn - len(history) + index), set())
                            verified[index] = len(proof) == 1 and (step["action"], step["observation"]) in proof
                    row["partition_features"] = _partition_features(
                        row["next_state"], row["context"], self._vocabulary, verified,
                    )
                    row["partition_label"] = _json(_canonical_action(row["next_parts"], row["context"]))
                self._partitions[key] = _fit_context_partition(rows, *key)
            self._partition_hash = _hash([
                artifact["artifact_hash"] for _, artifact in sorted(self._partitions.items())
            ])
        self._pending = None
        self.reset_episode()

    def _source_bank(self, library: dict, records: dict) -> tuple[list[dict], dict]:
        bank = {}
        counts = dict.fromkeys((
            "input_programs", "ignored_non_two_step_programs", "input_examples", "excluded_examples",
            "other_family_examples", "duplicate_examples", "bank_sources",
        ), 0)
        if self._raw:
            counts.update(raw_full_history_examples=0, raw_hashed_window_examples=0)
        for program in library["programs"]:
            if not isinstance(program, dict):
                raise ValueError("Invalid source program")
            program_id = _text(program.get("program_id"), "program_id")
            actions = _strings(program.get("actions"), "program actions")
            if not 2 <= len(actions) <= 4 or not isinstance(program.get("examples"), list):
                raise ValueError("Invalid literal source program")
            counts["input_programs"] += 1
            if len(actions) != 2:
                counts["ignored_non_two_step_programs"] += 1
                continue
            for example in program["examples"]:
                if not isinstance(example, dict):
                    raise ValueError("Invalid source example")
                group = _text(example.get("group_id"), "source group")
                counts["input_examples"] += 1
                if group in self._excluded:
                    counts["excluded_examples"] += 1
                    continue
                family = example.get("task_family")
                if family is not None:
                    _text(family, "source task_family")
                if family != self.task_family:
                    counts["other_family_examples"] += 1
                    continue
                if not self._raw and program.get("task_family") != family:
                    raise ValueError("Program/source native families disagree")
                ids = _strings(example.get("demonstration_ids"), "demonstration_ids")
                if len(ids) != 2 or len(set(ids)) != 2:
                    raise ValueError("Two distinct source demonstrations are required")
                task = _text(example.get("task_id"), "source task ID")
                for identifier, action in zip(ids, actions):
                    demo = records.get(identifier)
                    if (demo is None or demo.get("group_id") != group or demo.get("task_id") != task
                            or demo.get("action") != action):
                        raise ValueError("Missing or inconsistent source demonstrations")
                    if self._raw and "task_family" in demo and demo["task_family"] != family:
                        raise ValueError("Raw family disagrees with immutable demonstration metadata")
                    validate_public_state(demo.get("public_state"))
                    if action not in demo["public_state"]["actions"]:
                        raise ValueError("Source action is not native")
                before, after = (records[identifier]["public_state"] for identifier in ids)
                if " ".join(before["task"].casefold().split()) != " ".join(after["task"].casefold().split()):
                    raise ValueError("Source task changes inside a two-step demonstration")
                if "start_state_sha256" in example and example["start_state_sha256"] != public_state_sha256(before):
                    raise ValueError("Source starting state disagrees with the library")
                if self._raw:
                    counts[_validate_raw_pair(example, actions, before, after)] += 1
                start = example.get("start_turn")
                if self._partitioned and type(start) is int and start >= 0:
                    self._source_positions[ids[0]].add((group, task, start + 1))
                row = {
                    "source_id": ids[0], "next_demo_id": ids[1], "group_id": group,
                    "task_family": family, "actor_action": actions[0], "next_action": actions[1],
                    "pre_state": {**before, "actions": sorted(before["actions"])},
                    "next_state": {**after, "actions": sorted(after["actions"])},
                }
                if ids[0] in bank:
                    if {key: value for key, value in bank[ids[0]].items() if key != "program_ids"} != row:
                        raise ValueError("One source starting ID names conflicting continuations")
                    counts["duplicate_examples"] += 1
                    bank[ids[0]]["program_ids"] = sorted(set(bank[ids[0]]["program_ids"]) | {program_id})
                else:
                    bank[ids[0]] = deepcopy({**row, "program_ids": [program_id]})
        counts["bank_sources"] = len(bank)
        return [bank[key] for key in sorted(bank)], counts

    def reset_episode(self) -> None:
        if self._pending is not None:
            raise ValueError("Observe the pending decision before resetting")
        self._anchor = None
        self._actor_boundary = False
        self._turn = self._native_count = 0
        self._feedback: list[dict | None] = []

    def context_partition_artifacts(self) -> list[dict]:
        """Return independent JSON audit trees; old modes fit nothing and return []."""
        return deepcopy([artifact for _, artifact in sorted(self._partitions.items())])

    def _runtime_history(self, state: dict) -> list[bool]:
        start = self._turn - len(state["history"])
        return [
            0 <= start + index < len(self._feedback) and self._feedback[start + index] == step
            for index, step in enumerate(state["history"])
        ]

    def _audit(self) -> dict:
        return {
            "schema_version": PARTITIONED_CONTROLLER_SCHEMA if self._partitioned else CONTROLLER_SCHEMA,
            "mode": self.mode, "turn": self._turn,
            "group_id": self.group_id, "task_family": self.task_family,
            "source_bank_schema": SOURCE_BANK_SCHEMA,
            "source_evidence": self._source_evidence,
            "source_bank_hash": self._bank_hash, "vocabulary_hash": self._vocabulary_hash,
            "source_bank_counts": self._bank_counts, "source_bank_reasons": dict(sorted(self._bank_reasons.items())),
            "entry_threshold": self.entry_threshold, "minimum_groups": self.minimum_groups,
            "retrieval_limit": MAX_SOURCES,
            "affordance_check": self.mode != "grounded_continuation_no_affordance_check",
            "native_action_count": self._native_count,
            **({"context_partition_schema": CONTEXT_PARTITION_SCHEMA,
                "context_partition_model_hash": self._partition_hash} if self._partitioned else {}),
        }

    def _scores(self, key: tuple, query: dict, *, source_ids: set[str] | None = None) -> list[tuple[float, dict]]:
        sources = self._index.get(key, [])
        if not sources:
            return []
        if key not in self._cache:
            frequency = Counter(term for source in sources for term in source["features"])
            idf = {term: 1.0 + math.log((len(sources) + 1) / (count + 1))
                   for term, count in frequency.items()}
            vectors = []
            for source in sources:
                vector = {term: value * idf[term] for term, value in source["features"].items()}
                norm = math.sqrt(sum(value * value for value in vector.values()))
                vectors.append({term: value / norm for term, value in vector.items()})
            self._cache[key] = idf, vectors
        idf, vectors = self._cache[key]
        unknown_idf = 1.0 + math.log(len(sources) + 1)
        weighted = {term: value * idf.get(term, unknown_idf) for term, value in query.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if not norm:
            return []
        return [
            (min(1.0, max(0.0, sum(value * vector.get(term, 0.0)
                                   for term, value in weighted.items()) / norm)), source)
            for source, vector in zip(sources, vectors)
            if source_ids is None or source["source_id"] in source_ids
        ]

    def _candidates(self, state: dict) -> tuple[list[dict], list[dict], dict, str | None]:
        context, reason = _context(self._anchor["state"]["task"], self._anchor["action"], self._vocabulary)
        metadata = {"goal_roles": [], "anchor_shape_id": None, "indexed_source_count": 0}
        if self._partitioned:
            metadata["context_partition"] = None
        if reason is not None:
            return [], [], metadata, reason
        current_roles, reason = _roles(state["task"], self._vocabulary)
        if reason is not None or current_roles != context["roles"]:
            return [], [], metadata, reason or "current_task_roles_changed"
        key = self.task_family, context["shape_id"]
        metadata.update(goal_roles=context["roles"], anchor_shape_id=context["shape_id"],
                        indexed_source_count=len(self._index.get(key, [])))
        if not metadata["indexed_source_count"]:
            return [], [], metadata, "no_source_for_anchor_shape"
        native, available = set(state["actions"]), _available(state)
        signatures = _affordances(state, context)
        accepted, rejected = [], []
        source_ids = None
        if self._partitioned:
            verified = self._runtime_history(state)
            features = _partition_features(state, context, self._vocabulary, verified)
            artifact = self._partitions[key]
            leaf, path = _partition_leaf(artifact, features)
            source_ids = set(leaf["source_ids"])
            metadata["context_partition"] = {
                "artifact_hash": artifact["artifact_hash"], "leaf_id": leaf["node_id"], "path": path,
                "leaf_source_ids": leaf["source_ids"], "leaf_support_groups": leaf["support_groups"],
                "leaf_source_count": leaf["source_count"], "leaf_group_count": leaf["group_count"],
                "outside_leaf_count": metadata["indexed_source_count"] - leaf["source_count"],
                "runtime_feature_hash": _hash(sorted(features)), "verified_history_entries": sum(verified),
            }
        scores = (self._scores(key, _features(state, context, self._vocabulary), source_ids=source_ids)
                  if self._partitioned else self._scores(key, _features(state, context, self._vocabulary)))
        for score, source in scores:
            bindings, action = [], None
            if score < self.entry_threshold:
                reason = "below_entry_threshold"
            else:
                action, bindings, reason = _ground(source, context, available)
                if reason is None and action not in native:
                    reason = "grounded_action_not_native"
                if (reason is None and self.mode != "grounded_continuation_no_affordance_check"
                        and source["affordances"] != signatures):
                    reason = "goal_affordance_mismatch"
            entry = {
                "source_id": source["source_id"], "next_demo_id": source["next_demo_id"],
                "group_id": source["group_id"], "similarity": score,
                "grounded_action": action, "bindings": bindings,
            }
            if reason is not None:
                rejected.append({**entry, "reason": reason})
            else:
                accepted.append(entry)
        accepted.sort(key=lambda row: (-row["similarity"], row["source_id"]))
        rejected.extend({**row, "reason": "retrieval_limit"} for row in accepted[MAX_SOURCES:])
        return accepted[:MAX_SOURCES], sorted(rejected, key=lambda row: row["source_id"]), metadata, None

    def decide(self, public_state: dict) -> dict:
        if self._pending is not None:
            raise ValueError("observe must follow decide")
        state_hash = public_state_sha256(public_state)
        anchor = self._anchor
        candidates, rejections = [], []
        metadata = {"goal_roles": [], "anchor_shape_id": None, "indexed_source_count": 0}
        if self._partitioned:
            metadata["context_partition"] = None
        if anchor is None:
            reason = "actor_boundary_required" if self._actor_boundary else "actor_anchor_required"
        else:
            candidates, rejections, metadata, reason = self._candidates(public_state)
        groups = sorted({row["group_id"] for row in candidates})
        competing = defaultdict(list)
        for candidate in candidates:
            competing[candidate["grounded_action"]].append(candidate)
        if reason is None:
            reason = ("no_supported_grounded_continuation" if not candidates else
                      "ambiguous_grounded_actions" if len(competing) != 1 else
                      "insufficient_independent_groups" if len(groups) < self.minimum_groups else None)
        commit = reason is None
        action = candidates[0]["grounded_action"] if commit else None
        self._anchor = None
        self._pending = {
            "commit": commit, "action": action, "state": deepcopy(public_state),
            "sources": candidates if commit else [],
            **({"context_partition": deepcopy(metadata["context_partition"])} if self._partitioned else {}),
        }
        return deepcopy({
            "proposal": {"action": action},
            "decision": {
                **self._audit(), **metadata, "commit": commit,
                "reason": reason or "grounded_source_consensus", "state_sha256": state_hash,
                "anchor_action": None if anchor is None else anchor["action"],
                "anchor_turn": None if anchor is None else anchor["turn"],
                "candidates": candidates, "source_count": len(candidates), "support_groups": groups,
                "support_group_count": len(groups),
                "competing_actions": [{
                    "action": action, "support_groups": sorted({row["group_id"] for row in rows}),
                    "source_ids": sorted(row["source_id"] for row in rows),
                } for action, rows in sorted(competing.items())],
                "rejections": rejections,
                "reason_counts": dict(sorted(Counter(row["reason"] for row in rejections).items())),
            },
        })

    def observe(self, action: str, observation: str, *, compiled: bool,
                native_valid: bool = True) -> dict:
        if self._pending is None:
            raise ValueError("decide must precede every actual action")
        if type(compiled) is not bool or type(native_valid) is not bool:
            raise ValueError("compiled and native_valid must be bools")
        if compiled and not native_valid:
            raise ValueError("Compiled actions must be native-valid")
        _text(action, "actual action", empty=not compiled and not native_valid)
        _text(observation, "actual observation", empty=True)
        pending = self._pending
        if compiled and (not pending["commit"] or action != pending["action"]):
            raise ValueError("Compiled feedback must match the pending local proposal")
        if native_valid and action not in pending["state"]["actions"]:
            raise ValueError("Native-valid feedback escaped the pre-action native set")
        turn = self._turn
        self._turn += 1
        self._native_count += int(native_valid)
        if self._partitioned:
            self._feedback.append({"action": action, "observation": observation} if native_valid else None)
        self._pending = None
        self._actor_boundary = compiled
        self._anchor = ({
            "action": action, "observation": observation,
            "state": pending["state"], "turn": turn,
        } if not compiled and native_valid else None)
        sources = pending["sources"] if compiled else []
        groups = sorted({row["group_id"] for row in sources})
        return deepcopy({
            **self._audit(), "turn": turn, "action": action, "observation": observation,
            "compiled": compiled, "native_valid": native_valid,
            "reason": "continuation_executed_actor_required" if compiled else
            "actor_anchor_saved" if native_valid else "actor_action_not_native",
            "actor_required": compiled or not native_valid,
            "anchor_action": None if self._anchor is None else action,
            "anchor_turn": None if self._anchor is None else turn,
            "completed_action_count": self._turn,
            "source_ids": sorted(row["source_id"] for row in sources),
            "support_groups": groups, "support_group_count": len(groups),
            **({"context_partition": pending["context_partition"] if compiled else None}
               if self._partitioned else {}),
        })
