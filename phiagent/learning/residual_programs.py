"""Actor-anchored literal continuations with retrieved-source consensus.

Consumes a schema-v5 fragment-contract library and a build_policy TF-IDF policy.
Its sole fragment_contracts import is the public library_behavior_hash helper;
no FragmentController, runtime-contract hash, intervention or ablation helpers
are required. The caller pins the original compiler/hash implementation.
Guarded modes use ONLY positive STEP clauses; no mode enforces NET conditions.
The positive composed history projection is used solely for entry-cycle keys.
No goal progress, hidden state, or actor reasoning is inferred from text changes.

A hypothesis is one instance/counter with its actually retrieved, eligible source
examples. Multiple prefixes/instances from one group never increase independent
support. Any surviving completed hypothesis vetoes a longer continuation, even
when that short hypothesis has just one retrieved source. Disagreement also
vetoes: this is unanimity, not majority voting. anchored_single instead selects
the best actor-compatible instance with minimum_groups supporting source groups,
retaining that instance's sources rather than inventing single-source confidence.

Every actual action needs a preceding decide, including turn zero. Actor feedback
matches the action ACTUALLY executed against freshly prepared first steps, then
checks their pre-matched postconditions before advancing to step one. After a
fallback, reanchoring uses the current pre-action state, never hypothetical prior
actions. The caller retains the SAME actor/prompt and owns native termination.

The optional anchored_effect_* modes instead save actual native actor feedback
and retrieve at the NEXT decide, conditioned on the exact actor action. Cached
TF-IDF memories contain source NEXT-demonstration states, not source pre-action
states. Their policy action fields remain those demonstrations' native NEXT
actions; the exact first actor action selects the memory, not a fabricated
post-state action. Memories are built once per source first action and cached,
with excluded groups removed before both indexing and TF-IDF fitting.
Source entry bindings/prerequisites and first-step preconditions use
the saved PRE-actor state/native history; first postconditions use actual actor
feedback. Current positive STEP checks still apply before local continuation.

anchored_effect_consensus also requires equal goal-relative native-affordance
sets at each source step. Types are learned only from native command word/integer
syntax; roles are ordered exact task-word occurrences of those types. Empty,
repeated-type or incompatible roles abstain. Only entity instance integers are
normalized in action templates, not action syntax or entity types. The
anchored_effect_no_affordance_check ablation skips ONLY template-set equality;
role binding, effect retrieval and all remaining checks are retained. These are
observable support checks, not assumptions that any teacher action is optimal.

Entry keys contain instance identity, current task/observation/native actions,
and relevant validated-history facts, not raw-history churn. Fresh actor actions
can reauthorize a residual at a seen key. Only unanchored step-zero initiation is
blocked by that registry; no residual restarts itself or advances without feedback.
Construct per episode or call reset_episode. Pin source, constructor arguments,
library/policy hashes and decide/observe logs for replay; no models are called here.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re

from phiagent.learning import counterfactual_invocation as invocation
from phiagent.learning.fragment_contracts import library_behavior_hash


CONTROLLER_SCHEMA = "chisel-actor-anchored-residual-programs/v1"
EFFECT_CONTROLLER_SCHEMA = "chisel-actor-anchored-residual-programs/v2"
EFFECT_MEMORY_SCHEMA = "chisel-action-conditioned-source-effects/v1"
AFFORDANCE_SCHEMA = "chisel-goal-type-native-affordances/v1"
EFFECT_MODES = ("anchored_effect_consensus", "anchored_effect_no_affordance_check")
MODES = (
    "anchored_guarded_consensus", "anchored_consensus", "anchored_single",
    "unanchored_guarded_consensus", *EFFECT_MODES,
)
_ENTITY = re.compile(r"(?<!\w)(?P<type>[^\W\d]\w*(?:-\w+)*)\s+(?P<instance>[0-9]+)(?!\w)")
_TASK_WORD = re.compile(r"\w+(?:-\w+)*")


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else ''}string")
    return value


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Expected finite JSON-serializable data") from error
    return hashlib.sha256(encoded).hexdigest()


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _clauses(guard: dict, observation: str, *, post: bool,
             allowed: list[int] | None = None) -> list[int]:
    words = re.findall(r"\w+", observation.casefold())
    terms = {(word,) for word in words} | set(zip(words, words[1:]))
    indices = range(len(guard["clauses"])) if allowed is None else allowed
    return [
        index for index in indices
        if all((tuple(literal["tokens"]) in terms) == (literal["sign"] == ("+" if post else "-"))
               for literal in guard["clauses"][index]["literals"])
    ]


def _affordance_profile(state: dict) -> dict:
    by_type: dict[str, set[str]] = defaultdict(set)
    for action in state["actions"]:
        references = list(_ENTITY.finditer(action))
        if not references:
            continue
        template, start = [], 0
        for match in references:
            template.extend([action[start:match.start("instance")], {"instance": True}])
            start = match.end("instance")
        template.append(action[start:])
        encoded = json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        for match in references:
            by_type[match["type"].casefold()].add(encoded)
    return {"task_words": _TASK_WORD.findall(state["task"].casefold()), "by_type": dict(by_type)}


class ResidualProgramController:
    """Deterministic version-space control; no native execution or actor calls."""

    def __init__(self, library: dict, memory_policy: dict, *, group_id: str, task_family: str,
                 mode: str = "anchored_guarded_consensus", entry_threshold: float = 0.55,
                 minimum_groups: int = 2):
        _text(group_id, "group_id")
        _text(task_family, "task_family")
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError("Unknown residual controller mode")
        if type(minimum_groups) is not int or minimum_groups < 2:
            raise ValueError("minimum_groups must be an integer of at least two")
        if (type(entry_threshold) not in (int, float) or not math.isfinite(entry_threshold)
                or not 0 <= entry_threshold <= 1):
            raise ValueError("entry_threshold must be finite and within [0, 1]")
        self._library_hash = library_behavior_hash(library)
        if group_id in set(library["fit_groups"]) | set(library["net_fit_groups"]):
            raise ValueError("Target group supplied fitted evidence; recompile with its exclusion")
        if (not isinstance(memory_policy, dict)
                or memory_policy.get("schema_version") != invocation.POLICY_SCHEMA
                or not isinstance(memory_policy.get("records"), list)
                or not isinstance(memory_policy.get("index"), dict)):
            raise ValueError("Expected a native TF-IDF memory policy")
        for source in memory_policy["records"]:
            if not isinstance(source, dict) or not {
                "example_id", "task_id", "group_id", "action", "public_state",
            } <= source.keys():
                raise ValueError("Invalid policy demonstration schema")
        rebuilt = invocation.build_policy(memory_policy["records"])
        self._policy_hash = _digest({key: memory_policy[key] for key in ("records", "index")})
        if self._policy_hash != _digest({key: rebuilt[key] for key in ("records", "index")}):
            raise ValueError("Policy records/index must match the validated build_policy artifact")
        self._policy, self._library = rebuilt, deepcopy(library)
        self.group_id, self.task_family, self.mode = group_id, task_family, mode
        self.entry_threshold, self.minimum_groups = float(entry_threshold), minimum_groups
        self._guarded = mode != "anchored_consensus"
        self._anchored = mode != "unanchored_guarded_consensus"
        self._effect = mode in EFFECT_MODES
        self._affordance_equality = mode == "anchored_effect_consensus"
        self._excluded = frozenset(library["parameters"]["excluded_groups"]) | {group_id}
        self._programs = {program["instance_id"]: program for program in self._library["programs"]}
        self._sources = {
            (program["instance_id"], source["demonstration_ids"][0]): source
            for program in self._library["programs"] for source in program["examples"]
        }
        self._effect_cache: dict[str, dict] = {}
        if self._effect:
            self._index_effect_sources()
        self._pending = None
        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset episode memory, counters and residuals; reject unobserved decisions."""
        if self._pending is not None:
            raise ValueError("Observe the pending decision before resetting")
        self._active: list[dict] = []
        self._actor_required: str | None = None
        self._turn = self._native_count = 0
        self._native_actions: set[str] = set()
        self._native_words: set[str] = set()
        self._native_history: list[str] = []
        self._seen_entries: set[str] = set()
        self._effect_anchor = None
        self._effect_metadata = None

    def _index_effect_sources(self) -> None:
        self._demos = {row["example_id"]: row for row in self._policy["records"]}
        index: dict[str, dict[str, list[list[str]]]] = {}
        self._source_profiles, self._goal_types = {}, set()
        for program in self._programs.values():
            if program["conditioning"] != "task_family" or program["task_family"] != self.task_family:
                continue
            for source in program["examples"]:
                if source["group_id"] in self._excluded:
                    continue
                for step, identifier in enumerate(source["demonstration_ids"]):
                    demo = self._demos.get(identifier)
                    if (demo is None or demo["group_id"] != source["group_id"]
                            or demo["task_id"] != source["task_id"]
                            or demo["action"] != program["actions"][step]
                            or " ".join(demo["public_state"]["task"].casefold().split())
                            != source["task_text"]):
                        raise ValueError("Effect memory requires matching source-step demonstrations")
                    if step == 0 and invocation.public_state_sha256(demo["public_state"]) != source[
                        "start_state_sha256"
                    ]:
                        raise ValueError("Effect memory starting state disagrees with compiled source")
                    if identifier not in self._source_profiles:
                        profile = _affordance_profile(demo["public_state"])
                        self._source_profiles[identifier] = profile
                        self._goal_types.update(profile["by_type"])
                first, second = source["demonstration_ids"][:2]
                entries = index.setdefault(program["actions"][0], {}).setdefault(second, [])
                entries.append([program["instance_id"], first])
        self._effect_index = {
            action: {identifier: sorted(links) for identifier, links in sorted(entries.items())}
            for action, entries in sorted(index.items())
        }
        self._effect_index_hash = _digest({
            "schema_version": EFFECT_MEMORY_SCHEMA, "task_family": self.task_family,
            "excluded_groups": sorted(self._excluded), "goal_types": sorted(self._goal_types),
            "links_by_actor_action": self._effect_index,
        })

    def _effect_memory(self, actor_action: str) -> dict | None:
        links = self._effect_index.get(actor_action)
        if not links:
            return None
        if actor_action not in self._effect_cache:
            policy = invocation.build_policy([self._demos[identifier] for identifier in links])
            memory = {
                "schema_version": EFFECT_MEMORY_SCHEMA, "actor_action": actor_action,
                "task_family": self.task_family, "policy": policy, "links": links,
            }
            self._effect_cache[actor_action] = {**memory, "memory_hash": _digest(memory)}
        return self._effect_cache[actor_action]

    @staticmethod
    def _groups(hypotheses: list[dict]) -> list[str]:
        return sorted({source["group_id"] for h in hypotheses for source in h["sources"]})

    def _rank(self, h: dict) -> tuple:
        program = self._programs[h["instance_id"]]
        return (-max(source["similarity"] for source in h["sources"]), -len(program["actions"]),
                program["program_id"], program["instance_id"], h["sources"][0]["example_id"])

    def _summary(self, hypotheses: list[dict]) -> dict:
        snapshots, competing, completed = [], defaultdict(list), []
        for h in sorted(hypotheses, key=self._rank):
            program = self._programs[h["instance_id"]]
            ended = h["step"] == len(program["actions"])
            action = None if ended else program["actions"][h["step"]]
            if ended:
                completed.append(h["instance_id"])
            else:
                competing[action].append(h)
            snapshots.append({
                **h, "program_id": program["program_id"], "length": len(program["actions"]),
                "next_action": action, "support_groups": self._groups([h]),
            })
        groups = self._groups(hypotheses)
        return {
            "hypotheses": snapshots, "hypothesis_count": len(hypotheses),
            "source_count": sum(len(h["sources"]) for h in hypotheses),
            "support_groups": groups, "support_group_count": len(groups),
            "completed_instance_ids": sorted(completed),
            "competing_actions": [
                {"action": action, "support_groups": self._groups(rows),
                 "instance_ids": sorted(h["instance_id"] for h in rows),
                 "source_count": sum(len(h["sources"]) for h in rows)}
                for action, rows in sorted(competing.items())
            ],
        }

    def _audit(self) -> dict:
        return {
            "schema_version": EFFECT_CONTROLLER_SCHEMA if self._effect else CONTROLLER_SCHEMA,
            "mode": self.mode, "turn": self._turn,
            "library_behavior_hash": self._library_hash, "memory_policy_hash": self._policy_hash,
            "group_id": self.group_id, "task_family": self.task_family,
            "entry_threshold": self.entry_threshold, "minimum_groups": self.minimum_groups,
            "step_guard_variant": "positive" if self._guarded else None, "net_checks": False,
            "native_action_count": self._native_count, "seen_entry_count": len(self._seen_entries),
            **({
                "retrieval_basis": "exact_actor_action_then_next_demonstration_state",
                "effect_index_hash": self._effect_index_hash,
                "effect_memory": self._effect_metadata,
                "affordance_schema": AFFORDANCE_SCHEMA,
                "goal_role_checks": True, "affordance_equality_check": self._affordance_equality,
            } if self._effect else {}),
        }

    def _rejection(self, h: dict, stage: str, reason: str) -> dict:
        return {
            "stage": stage, "reason": reason, "instance_id": h["instance_id"], "step": h["step"],
            "source_example_ids": sorted(source["example_id"] for source in h["sources"]),
            "support_groups": self._groups([h]),
        }

    def _binding_reason(self, program: dict, source: dict, state: dict, *,
                        native_words: set[str] | None = None) -> str | None:
        if program["conditioning"] != "task_family":
            return "source_task_family_missing"
        if program["task_family"] != self.task_family:
            return "task_family_mismatch"
        if not set(source["goal_words"]) <= _words(state["task"]):
            return "source_goal_words_missing"
        if not set(program["history_prerequisite_words"]) <= (
            self._native_words if native_words is None else native_words
        ):
            return "history_prerequisites_missing"
        return None

    def _entry_key(self, program: dict, state: dict, *,
                   native_actions: set[str] | None = None,
                   native_words: set[str] | None = None) -> str:
        actions = self._native_actions if native_actions is None else native_actions
        words = self._native_words if native_words is None else native_words
        projection = program["reentry_projection"]["positive"]
        return _digest({
            "instance_id": program["instance_id"],
            "public_state": invocation.public_state_sha256({**state, "history": []}),
            "history_actions": [action for action in projection["history_actions"]
                                if action in actions],
            "history_words": [word for word in projection["history_words"]
                              if word in words],
        })

    def _prepare_effect(self, state: dict) -> tuple[list[dict], list[dict], list[dict]]:
        anchor = self._effect_anchor
        memory = self._effect_memory(anchor["action"])
        if memory is None:
            self._effect_metadata["status"] = "no_source_action"
            return [], [], []
        records = memory["policy"]["records"]
        self._effect_metadata.update({
            "status": "ready", "memory_hash": memory["memory_hash"], "record_count": len(records),
            "source_reference_count": sum(len(links) for links in memory["links"].values()),
            "source_groups": sorted({row["group_id"] for row in records}),
        })
        prior_actions = set(anchor["native_history"])
        prior_words = {word for action in prior_actions for word in _words(action)}
        hypotheses, rejected, checks, seen = {}, [], [], set()
        for score, demo in invocation.retrieve_demonstrations(
            memory["policy"], state, excluded_groups=self._excluded, limit=128,
        ):
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Retrieval similarity must be a finite number within [0, 1]")
            for identifier, first_id in memory["links"].get(demo["example_id"], []):
                source = self._sources[identifier, first_id]
                program = self._programs[identifier]
                h = {
                    "instance_id": identifier, "step": 1, "anchor_turn": anchor["turn"],
                    "entry_key": self._entry_key(program, anchor["state"], native_actions=prior_actions,
                                                 native_words=prior_words),
                    "precondition_clauses": [],
                    "sources": [{"example_id": first_id, "group_id": source["group_id"],
                                 "similarity": score, "effect_example_id": demo["example_id"]}],
                }
                pair = identifier, first_id
                reason = ("excluded_source_group" if demo["group_id"] in self._excluded else
                          "duplicate_source_reference" if pair in seen else
                          "below_entry_threshold" if score < self.entry_threshold else
                          self._binding_reason(program, source, anchor["state"],
                                               native_words=prior_words))
                seen.add(pair)
                if reason is None:
                    guard = program["steps"][0]["positive_guard"]
                    pre = _clauses(guard, anchor["state"]["observation"], post=False)
                    post = _clauses(guard, anchor["observation"], post=True, allowed=pre)
                    checks.append({
                        "instance_id": identifier, "source_example_id": first_id,
                        "group_id": source["group_id"], "precondition_clauses": pre,
                        "postcondition_clauses": post, "passed": bool(pre and post),
                    })
                    reason = ("actor_precondition_failed" if not pre else
                              "actor_postcondition_failed" if not post else None)
                if reason is not None:
                    rejected.append(self._rejection(h, "effect_entry", reason))
                elif identifier in hypotheses:
                    hypotheses[identifier]["sources"].extend(h["sources"])
                else:
                    hypotheses[identifier] = h
        for h in hypotheses.values():
            h["sources"].sort(key=lambda row: (-row["similarity"], row["example_id"]))
        return sorted(hypotheses.values(), key=self._rank), rejected, checks

    def _affordance_reason(self, source: dict, current: dict) -> tuple[str | None, dict]:
        vocabulary = self._goal_types | set(current["by_type"])
        source_roles = [word for word in source["task_words"] if word in vocabulary]
        current_roles = [word for word in current["task_words"] if word in vocabulary]
        source_templates = sorted({template for role in source_roles
                                   for template in source["by_type"].get(role, ())})
        current_templates = sorted({template for role in current_roles
                                    for template in current["by_type"].get(role, ())})
        reason = ("source_goal_types_unbound" if not source_roles else
                  "current_goal_types_unbound" if not current_roles else
                  "ambiguous_goal_type_roles" if len(set(source_roles)) != len(source_roles)
                  or len(set(current_roles)) != len(current_roles) else
                  "incompatible_goal_type_roles" if source_roles != current_roles else
                  "goal_affordance_mismatch" if self._affordance_equality
                  and source_templates != current_templates else None)
        return reason, {
            "source_goal_roles": source_roles, "current_goal_roles": current_roles,
            "source_affordance_hash": _digest(source_templates),
            "current_affordance_hash": _digest(current_templates),
            "source_affordance_count": len(source_templates),
            "current_affordance_count": len(current_templates),
        }

    def _prepare(self, state: dict) -> tuple[list[dict], list[dict]]:
        hypotheses, rejected, seen = {}, [], set()
        for score, demo in invocation.retrieve_demonstrations(
            self._policy, state, excluded_groups=self._excluded, limit=128,
        ):
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Retrieval similarity must be a finite number within [0, 1]")
            for identifier in self._library["entry_index"].get(demo["example_id"], []):
                program = self._programs[identifier]
                source = self._sources[identifier, demo["example_id"]]
                if (demo["group_id"] != source["group_id"] or demo["task_id"] != source["task_id"]
                        or demo["action"] != program["actions"][0]
                        or invocation.public_state_sha256(demo["public_state"])
                        != source["start_state_sha256"]):
                    raise ValueError("Retrieved demonstration disagrees with the compiled source")
                h = {
                    "instance_id": identifier, "step": 0, "anchor_turn": None,
                    "entry_key": self._entry_key(program, state), "precondition_clauses": [],
                    "sources": [{"example_id": demo["example_id"], "group_id": source["group_id"],
                                 "similarity": score}],
                }
                pair = identifier, demo["example_id"]
                reason = ("excluded_source_group" if demo["group_id"] in self._excluded else
                          "duplicate_source_reference" if pair in seen else
                          "below_entry_threshold" if score < self.entry_threshold else
                          "first_action_not_native" if program["actions"][0] not in state["actions"]
                          else self._binding_reason(program, source, state))
                seen.add(pair)
                if reason is None and self._guarded:
                    h["precondition_clauses"] = _clauses(
                        program["steps"][0]["positive_guard"], state["observation"], post=False,
                    )
                    if not h["precondition_clauses"]:
                        reason = "entry_precondition_failed"
                if reason is not None:
                    rejected.append(self._rejection(h, "entry", reason))
                elif identifier in hypotheses:
                    hypotheses[identifier]["sources"].extend(h["sources"])
                else:
                    hypotheses[identifier] = h
        for h in hypotheses.values():
            h["sources"].sort(key=lambda row: (-row["similarity"], row["example_id"]))
        return sorted(hypotheses.values(), key=self._rank), sorted(rejected, key=_digest)

    def _preconditions(self, state: dict) -> tuple[list[dict], list[dict]]:
        survivors, rejected = [], []
        current_profile = _affordance_profile(state) if self._effect else None
        for old in self._active:
            h = deepcopy(old)
            program = self._programs[h["instance_id"]]
            sources = []
            for source in h["sources"]:
                details = {}
                reason = self._binding_reason(
                    program, self._sources[h["instance_id"], source["example_id"]], state,
                )
                if reason is None and self._effect:
                    demo_id = self._sources[h["instance_id"], source["example_id"]][
                        "demonstration_ids"
                    ][h["step"]]
                    reason, details = self._affordance_reason(self._source_profiles[demo_id], current_profile)
                    details["source_step_demonstration_id"] = demo_id
                if reason is None:
                    sources.append(source)
                else:
                    rejected.append({
                        **self._rejection({**h, "sources": [source]}, "step", reason), **details,
                    })
            h["sources"] = sources
            if not sources:
                continue
            if program["actions"][h["step"]] not in state["actions"]:
                rejected.append(self._rejection(h, "step", "next_action_not_native"))
                continue
            h["precondition_clauses"] = []
            if self._guarded:
                h["precondition_clauses"] = _clauses(
                    program["steps"][h["step"]]["positive_guard"], state["observation"], post=False,
                )
                if not h["precondition_clauses"]:
                    rejected.append(self._rejection(h, "step", "step_precondition_failed"))
                    continue
            survivors.append(h)
        return survivors, rejected

    def _boundary(self, hypotheses: list[dict]) -> str | None:
        if not hypotheses:
            return "no_supported_residual"
        if any(h["step"] == len(self._programs[h["instance_id"]]["actions"]) for h in hypotheses):
            return "completed_hypothesis"
        if len(self._groups(hypotheses)) < self.minimum_groups:
            return "insufficient_source_groups"
        return None

    def decide(self, public_state: dict) -> dict:
        """Prepare an action or actor deferral; program counters never advance here."""
        if self._pending is not None:
            raise ValueError("observe must follow decide before another decision")
        state_hash = invocation.public_state_sha256(public_state)
        starts, rejected = None, []
        effect_checks = []
        boundary = self._actor_required
        if boundary is not None:
            hypotheses, reason = self._active, "actor_boundary_required"
        elif self._active:
            hypotheses, rejected = self._preconditions(public_state)
            reason = self._boundary(hypotheses)
        elif self._effect:
            starts = []
            if self._effect_anchor is None:
                hypotheses, reason = [], "actor_anchor_required"
            else:
                self._active, rejected, effect_checks = self._prepare_effect(public_state)
                hypotheses, step_rejections = self._preconditions(public_state)
                rejected.extend(step_rejections)
                self._effect_anchor = None
                reason = self._boundary(hypotheses)
        else:
            starts, rejected = self._prepare(public_state)
            hypotheses = starts
            reason = "actor_anchor_required" if self._anchored else None
            if not self._anchored:
                hypotheses = []
                for h in starts:
                    if h["entry_key"] in self._seen_entries:
                        rejected.append(self._rejection(h, "entry", "repeated_local_entry"))
                    else:
                        hypotheses.append(h)
                reason = self._boundary(hypotheses)
        summary = self._summary(hypotheses)
        if reason is None and len(summary["competing_actions"]) != 1:
            reason = "ambiguous_continuations"
        commit = reason is None
        action = summary["competing_actions"][0]["action"] if commit else None
        if not commit and starts is None:
            starts, entry_rejections = ([], []) if self._effect else self._prepare(public_state)
            rejected.extend(entry_rejections)
        self._active = hypotheses if commit else []
        self._pending = {
            "commit": commit, "action": action, "state": deepcopy(public_state),
            "hypotheses": hypotheses if commit else starts,
        }
        return deepcopy({
            "proposal": {"action": action},
            "decision": {
                **self._audit(), **summary, "commit": commit,
                "reason": reason or ("local_entry_consensus" if hypotheses[0]["step"] == 0
                                     else "single_hypothesis_continuation"
                                     if self.mode == "anchored_single"
                                     else "residual_consensus"),
                "state_sha256": state_hash, "boundary_reason": boundary,
                "anchor_candidates": [] if commit else self._summary(starts)["hypotheses"],
                "rejections": rejected,
                **({"actor_effect_checks": effect_checks} if self._effect else {}),
            },
        })

    def observe(self, action: str, observation: str, *, compiled: bool,
                native_valid: bool = True) -> dict:
        """Consume actual feedback; actor overrides use the saved pre-action state.

        Malformed/empty actor strings with native_valid=False are recorded as
        actual steps, but add no native-history facts and retain no anchor.
        Compiled feedback must be nonempty, native-valid and match the proposal.
        """
        if self._pending is None:
            raise ValueError("decide must precede every actual action, including actor actions")
        if type(compiled) is not bool or type(native_valid) is not bool:
            raise ValueError("compiled and native_valid must be bools")
        if compiled and not native_valid:
            raise ValueError("Compiled feedback must be native-valid")
        _text(action, "actual action", empty=not native_valid and not compiled)
        _text(observation, "actual observation", empty=True)
        pending = self._pending
        if compiled and (not pending["commit"] or action != pending["action"]):
            raise ValueError("Compiled feedback must match the pending committed action")
        if native_valid and action not in pending["state"]["actions"]:
            raise ValueError("native_valid action is not in the saved native action set")
        rejected, checks, survivors = [], [], []
        hypotheses = pending["hypotheses"]
        if not compiled and self._effect:
            hypotheses = []
            self._effect_anchor = ({
                "action": action, "observation": observation, "state": pending["state"],
                "native_history": list(self._native_history), "turn": self._turn,
            } if native_valid else None)
            self._effect_metadata = ({
                "schema_version": EFFECT_MEMORY_SCHEMA, "status": "retrieval_pending",
                "actor_action": action, "actor_turn": self._turn,
                "pre_actor_state_sha256": invocation.public_state_sha256(pending["state"]),
                "pre_actor_native_history_sha256": _digest(self._native_history),
                "memory_hash": None, "record_count": 0, "source_reference_count": 0,
                "source_groups": [],
            } if native_valid else None)
        elif not compiled and pending["commit"]:
            hypotheses, rejected = self._prepare(pending["state"])
        for h in hypotheses:
            program, step = self._programs[h["instance_id"]], h["step"]
            if not native_valid or program["actions"][step] != action:
                rejected.append(self._rejection(
                    h, "feedback", "action_not_native" if not native_valid else "actor_action_mismatch",
                ))
                continue
            if compiled:
                self._seen_entries.add(h["entry_key"])
            matched = _clauses(
                program["steps"][step]["positive_guard"], observation, post=True,
                allowed=h["precondition_clauses"],
            ) if self._guarded else []
            passed = bool(matched) if self._guarded else None
            checks.append({
                "instance_id": h["instance_id"], "step": step, "support_groups": self._groups([h]),
                "checked": self._guarded, "passed": passed,
                "precondition_clauses": list(h["precondition_clauses"]),
                "postcondition_clauses": matched,
            })
            if passed is False:
                rejected.append(self._rejection(h, "feedback", "step_postcondition_failed"))
            else:
                survivors.append({
                    **h, "step": step + 1, "precondition_clauses": [],
                    "anchor_turn": h["anchor_turn"] if compiled else self._turn,
                })
        if not compiled and self.mode == "anchored_single":
            chosen = None
            for h in sorted(survivors, key=self._rank):
                if len(self._groups([h])) < self.minimum_groups:
                    reason = "single_instance_insufficient_groups"
                elif chosen is not None:
                    reason = "single_hypothesis_not_selected"
                else:
                    chosen = h
                    continue
                rejected.append(self._rejection(h, "selection", reason))
            survivors = [] if chosen is None else [chosen]
        turn = self._turn
        self._turn += 1
        if native_valid:
            self._native_count += 1
            self._native_actions.add(action)
            self._native_words.update(_words(action))
            self._native_history.append(action)
        self._pending = None
        self._active = survivors
        self._actor_required = self._boundary(survivors)
        if not compiled and not survivors:
            self._actor_required = None
        reason = ("actor_action_not_native" if not native_valid else
                  "actor_effect_pending" if not compiled and self._effect else
                  self._actor_required if compiled and self._actor_required else
                  "local_action_observed" if compiled else
                  "actor_anchor_accepted" if survivors else "actor_has_no_supported_anchor")
        return deepcopy({
            **self._audit(), **self._summary(survivors), "turn": turn,
            "action": action, "observation": observation, "compiled": compiled,
            "native_valid": native_valid, "reason": reason, "checks": checks,
            "rejections": rejected,
            "actor_required": self._actor_required is not None or (
                self._anchored and not survivors and self._effect_anchor is None
            ),
            "boundary_reason": self._actor_required,
            "completed_action_count": self._turn,
        })
