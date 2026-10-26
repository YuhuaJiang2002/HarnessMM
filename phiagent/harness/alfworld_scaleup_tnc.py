"""Dependency-aware harness minimization primitives for ALFWorld.

The module is deliberately independent of ALFWorld, Torch, and model runtimes.
It defines the frozen actor-facing components, lossless action-schema
representations, the typed pair graph, and the deterministic minimum-retained-
token optimizer used by the remote experiment runner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


SYSTEM_ROLE = "You are an expert agent operating in the ALFRED Embodied Environment."
REASONING_INSTRUCTION = (
    "You should first reason step-by-step about the current situation. This "
    "reasoning process MUST be enclosed within <think> </think> tags."
)
ACTION_OUTPUT_INSTRUCTION = (
    "Once you've finished your reasoning, you should choose an admissible "
    "action for current step and present it within <action> </action> tags."
)
TURN_CUE = "Now it's your turn to take an action."

ACTION_PREFIXES = (
    "go to ",
    "take ",
    "put ",
    "open ",
    "close ",
    "toggle ",
    "examine ",
    "clean ",
    "heat ",
    "cool ",
)

SCHEMA_COMPONENT_TO_FAMILIES = {
    "schema_go_to_representation": frozenset({"go to"}),
    "schema_take_representation": frozenset({"take"}),
    "schema_exact_representation": frozenset({"exact"}),
    "schema_examine_representation": frozenset({"examine"}),
    "schema_tail_family_representation": frozenset(
        {"put", "open", "close", "toggle", "clean", "heat", "cool"}
    ),
}

COMPONENT_IDS = (
    *SCHEMA_COMPONENT_TO_FAMILIES,
    "trajectory_action_history",
    "reasoning_planning_instruction",
    "duplicate_user_role",
    "base_system_role",
    "turn_cue",
)

GRAPH_NODE_STATES = (
    "REMOVABLE",
    "NECESSARY",
    "UNRESOLVED",
    "REMOVABLE_GIVEN(X)",
)
GRAPH_EDGE_STATES = (
    "JOINTLY_REMOVABLE",
    "NOT_JOINTLY_REMOVABLE",
    "SUBSTITUTABLE_WITH",
    "UNRESOLVED",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_hash(value: Mapping[str, Any]) -> str:
    """Hash a JSON object while excluding its self-referential binding."""

    unbound = {key: item for key, item in value.items() if key != "hash_binding"}
    return sha256_bytes(canonical_json_bytes(unbound))


def bind_content(value: Mapping[str, Any]) -> dict[str, Any]:
    bound = dict(value)
    bound["hash_binding"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding hash_binding",
        "digest": content_hash(bound),
    }
    return bound


def verify_content_binding(value: Mapping[str, Any]) -> bool:
    binding = value.get("hash_binding")
    return (
        isinstance(binding, Mapping)
        and binding.get("algorithm") == "sha256"
        and binding.get("digest") == content_hash(value)
    )


def action_family(action: str) -> tuple[str, str]:
    for prefix in ACTION_PREFIXES:
        if action.startswith(prefix):
            return prefix.strip(), action[len(prefix) :]
    return "exact", action


def flat_action_schema(actions: Sequence[str]) -> str:
    return "\n ".join(f"'{action}'" for action in actions if action != "help")


def compact_family_line(family: str, values: Sequence[str]) -> str:
    """Return a self-contained, exactly invertible family representation."""

    if not values:
        raise ValueError("cannot encode an empty action family")
    if any(" | " in value or "}" in value for value in values):
        raise ValueError("an action contains a reserved compact-schema delimiter")
    if family == "exact":
        return (
            "exact actions = {"
            + " | ".join(values)
            + "}; each entry is one unchanged admissible command"
        )
    return (
        f"{family} + {{"
        + " | ".join(values)
        + f"}}; each command is the literal prefix '{family} ' plus one suffix"
    )


def parse_compact_family_line(line: str) -> list[str]:
    if line.startswith("exact actions = {"):
        payload, suffix = line[len("exact actions = {") :].rsplit("}", 1)
        if suffix != "; each entry is one unchanged admissible command":
            raise ValueError("invalid exact-family decoder contract")
        return payload.split(" | ")
    if " + {" not in line:
        raise ValueError("invalid family representation")
    family, remainder = line.split(" + {", 1)
    payload, suffix = remainder.rsplit("}", 1)
    expected = f"; each command is the literal prefix '{family} ' plus one suffix"
    if suffix != expected:
        raise ValueError("invalid prefixed-family decoder contract")
    return [f"{family} {value}" for value in payload.split(" | ")]


def mixed_action_schema(
    actions: Sequence[str], deleted_components: Iterable[str]
) -> str:
    """Apply disjoint lossless representation changes to selected families.

    Every environment-provided command remains reconstructable.  Component
    deletion never suppresses an action family or changes the environment-side
    admissible set.
    """

    original = [action for action in actions if action != "help"]
    deleted = set(deleted_components)
    compressed_families: set[str] = set()
    for component, families in SCHEMA_COMPONENT_TO_FAMILIES.items():
        if component in deleted:
            compressed_families.update(families)

    grouped: dict[str, list[str]] = defaultdict(list)
    for action in original:
        family, value = action_family(action)
        grouped[family].append(value)

    emitted_families: set[str] = set()
    lines: list[str] = []
    for action in original:
        family, _ = action_family(action)
        if family not in compressed_families:
            lines.append(f"'{action}'")
            continue
        if family in emitted_families:
            continue
        lines.append(compact_family_line(family, grouped[family]))
        emitted_families.add(family)

    reconstructed: list[str] = []
    for line in lines:
        if line.startswith("'") and line.endswith("'"):
            reconstructed.append(line[1:-1])
        else:
            reconstructed.extend(parse_compact_family_line(line))
    if Counter(reconstructed) != Counter(original):
        raise RuntimeError("mixed schema failed to preserve the action multiset")
    return "\n ".join(lines)


def build_actor_messages(
    *,
    task_description: str,
    current_observation: str,
    admissible_actions: Sequence[str],
    history_records: Sequence[Mapping[str, str]],
    total_previous_steps: int,
    deleted_components: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Build the real actor request with component-level interventions."""

    deleted = set(deleted_components)
    unknown = deleted.difference(COMPONENT_IDS)
    if unknown:
        raise ValueError(f"unknown harness components: {sorted(unknown)}")

    schema = mixed_action_schema(admissible_actions, deleted)
    lines: list[str] = []
    if "duplicate_user_role" not in deleted:
        if history_records:
            lines.append(f"{SYSTEM_ROLE} Your task is to: {task_description}")
        else:
            lines.append(SYSTEM_ROLE)

    if history_records:
        if "duplicate_user_role" in deleted:
            lines.append(f"Your task is to: {task_description}")
        if "trajectory_action_history" not in deleted:
            rendered = []
            start = total_previous_steps - len(history_records) + 1
            for offset, record in enumerate(history_records):
                step = start + offset
                rendered.append(
                    f"[Observation {step}: '{record['text_obs']}', "
                    f"Action {step}: '{record['action']}']"
                )
            lines.append(
                f"Prior to this step, you have already taken {total_previous_steps} "
                f"step(s). Below are the most recent {len(history_records)} "
                "observations and the corresponding actions you took: "
                + "\n".join(rendered)
            )
        else:
            lines.append(
                f"Prior to this step, you have already taken {total_previous_steps} "
                "step(s)."
            )
        lines.append(
            f"You are now at step {total_previous_steps + 1} and your current "
            f"observation is: {current_observation}"
        )
    else:
        lines.append(f"Your current observation is: {current_observation}")

    lines.append(
        "Your admissible actions of the current situation are: [" + schema + "]."
    )
    lines.append("")
    if "turn_cue" not in deleted:
        lines.append(TURN_CUE)
    if "reasoning_planning_instruction" not in deleted:
        lines.append(REASONING_INSTRUCTION)
    lines.append(ACTION_OUTPUT_INSTRUCTION)
    user = "\n".join(lines)

    messages: list[dict[str, str]] = []
    if "base_system_role" not in deleted:
        messages.append({"role": "system", "content": SYSTEM_ROLE})
    messages.append({"role": "user", "content": user})
    return messages


def mutate_real_actor_messages(
    *,
    full_user_prompt: str,
    admissible_actions: Sequence[str],
    deleted_components: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Mutate the real SkillOpt-rendered request without rebuilding fixed text.

    With an empty deletion set this returns the byte-identical user prompt
    emitted by the audited environment manager.  This property makes ``Full``
    an exact baseline rather than a reimplemented approximation.
    """

    deleted = set(deleted_components)
    unknown = deleted.difference(COMPONENT_IDS)
    if unknown:
        raise ValueError(f"unknown harness components: {sorted(unknown)}")
    user = full_user_prompt

    schema_components = deleted.intersection(SCHEMA_COMPONENT_TO_FAMILIES)
    if schema_components:
        prefix = "Your admissible actions of the current situation are: ["
        suffix = "]."
        start = user.find(prefix)
        if start < 0:
            raise ValueError("real prompt lacks the admissible-action schema prefix")
        value_start = start + len(prefix)
        end = user.find(suffix, value_start)
        if end < 0:
            raise ValueError("real prompt lacks the admissible-action schema suffix")
        replacement = mixed_action_schema(admissible_actions, schema_components)
        user = user[:value_start] + replacement + user[end:]

    if "trajectory_action_history" in deleted:
        history_pattern = re.compile(
            r"Prior to this step, you have already taken (?P<count>\d+) step\(s\)\. "
            r"Below are the most recent \d+ observations and the corresponding "
            r"actions you took: .*?(?=\nYou are now at step )",
            re.DOTALL,
        )
        user, substitutions = history_pattern.subn(
            lambda match: (
                "Prior to this step, you have already taken "
                + match.group("count")
                + " step(s)."
            ),
            user,
            count=1,
        )
        # Initial observations legitimately contain no history field.
        if substitutions == 0 and "Prior to this step" in user:
            raise ValueError("history field exists but did not match the frozen grammar")

    if "reasoning_planning_instruction" in deleted:
        user = user.replace(REASONING_INSTRUCTION + "\n", "", 1)
    if "turn_cue" in deleted:
        user = user.replace(TURN_CUE + "\n", "", 1)
    if "duplicate_user_role" in deleted:
        if (SYSTEM_ROLE + " ") in user:
            user = user.replace(SYSTEM_ROLE + " ", "", 1)
        else:
            user = user.replace(SYSTEM_ROLE + "\n", "", 1)

    messages: list[dict[str, str]] = []
    if "base_system_role" not in deleted:
        messages.append({"role": "system", "content": SYSTEM_ROLE})
    messages.append({"role": "user", "content": user})
    return messages


def manifest_components() -> list[dict[str, Any]]:
    """Return the outcome-blind 10-component bank and audited mass proxies."""

    family_source = {
        "schema_go_to_representation": 5045.1,
        "schema_take_representation": 855.2,
        "schema_exact_representation": 425.6,
        "schema_examine_representation": 416.3,
        "schema_tail_family_representation": 51.0,
    }
    family_total = sum(family_source.values())
    schema_marginal_mass = 8300.90625
    schema_saving = 1414.0
    rows: list[dict[str, Any]] = []
    for identifier, raw_mass in family_source.items():
        families = sorted(SCHEMA_COMPONENT_TO_FAMILIES[identifier])
        rows.append(
            {
                "id": identifier,
                "source_token_mass_proxy_per_episode": (
                    schema_marginal_mass * raw_mass / family_total
                ),
                "maximum_removable_token_proxy_per_episode": (
                    schema_saving * raw_mass / family_total
                ),
                "exact_source": (
                    "AlfWorldEnvironmentManager.build_text_obs: each command is "
                    "serialized as a quoted flat entry; selected family set="
                    + ",".join(families)
                ),
                "location": "user.admissible_actions",
                "addressability": "LOSSLESS_REPRESENTATION_ONLY",
                "semantic_role": "environment action-interface representation",
                "semantic_group": "admissible_action_schema",
                "pre_outcome_metadata": {
                    "dev_global_suppression_mass_tokens_per_episode": raw_mass,
                    "families": families,
                    "environment_action_set_mutated": False,
                    "parser_contract_mutated": False,
                    "selection_basis": "prior TRAIN+VALIDATION token mass and usage",
                },
            }
        )

    rows.extend(
        [
            {
                "id": "trajectory_action_history",
                "source_token_mass_proxy_per_episode": 3388.09375,
                "maximum_removable_token_proxy_per_episode": 3388.09375,
                "exact_source": (
                    "SimpleMemory.fetch(history_length=2) action_history field and "
                    "its history-introduction sentence"
                ),
                "location": "user.action_history",
                "addressability": "CONDITIONALLY_ADDRESSABLE",
                "semantic_role": "recent latent-state and action trace",
                "semantic_group": "trajectory_memory",
                "pre_outcome_metadata": {
                    "history_length": 2,
                    "risk": "HIGH",
                    "selection_basis": "prior audited actor-context mass",
                },
            },
            {
                "id": "reasoning_planning_instruction",
                "source_token_mass_proxy_per_episode": 964.4375,
                "maximum_removable_token_proxy_per_episode": 964.4375,
                "exact_source": REASONING_INSTRUCTION,
                "location": "user.reasoning_instruction",
                "addressability": "CONDITIONALLY_ADDRESSABLE",
                "semantic_role": "reasoning-mode instruction; think tags remain parser-required",
                "semantic_group": "response_protocol",
                "pre_outcome_metadata": {
                    "risk": "MEDIUM_HIGH",
                    "parser_contract_mutated": False,
                    "selection_basis": "prior audited actor-context mass",
                },
            },
            {
                "id": "duplicate_user_role",
                "source_token_mass_proxy_per_episode": 557.40625,
                "maximum_removable_token_proxy_per_episode": 557.40625,
                "exact_source": SYSTEM_ROLE,
                "location": "user.role_prefix",
                "addressability": "TNC_ADDRESSABLE",
                "semantic_role": "user-side duplicate actor identity",
                "semantic_group": "actor_identity_redundancy",
                "pre_outcome_metadata": {
                    "risk": "LOW_SINGLE_HIGH_JOINT",
                    "dependency_prior": "base_system_role",
                    "selection_basis": "exact duplicate string and prior audit",
                },
            },
            {
                "id": "base_system_role",
                "source_token_mass_proxy_per_episode": 556.40625,
                "maximum_removable_token_proxy_per_episode": 556.40625,
                "exact_source": SYSTEM_ROLE,
                "location": "messages[role=system]",
                "addressability": "CONDITIONALLY_ADDRESSABLE",
                "semantic_role": "system-side actor identity",
                "semantic_group": "actor_identity_redundancy",
                "pre_outcome_metadata": {
                    "risk": "LOW_SINGLE_HIGH_JOINT",
                    "dependency_prior": "duplicate_user_role",
                    "selection_basis": "exact duplicate string and prior audit",
                },
            },
            {
                "id": "turn_cue",
                "source_token_mass_proxy_per_episode": 408.03125,
                "maximum_removable_token_proxy_per_episode": 408.03125,
                "exact_source": TURN_CUE,
                "location": "user.turn_cue",
                "addressability": "TNC_ADDRESSABLE",
                "semantic_role": "turn-taking cue",
                "semantic_group": "response_protocol",
                "pre_outcome_metadata": {
                    "risk": "LOW",
                    "selection_basis": "prior audited actor-context mass",
                },
            },
        ]
    )
    if tuple(row["id"] for row in rows) != COMPONENT_IDS:
        raise RuntimeError("component bank order drifted")
    return rows


@dataclass(frozen=True)
class PairConstraint:
    left: str
    right: str
    edge_state: str


def minimum_retained_harness(
    *,
    components: Sequence[Mapping[str, Any]],
    node_states: Mapping[str, str],
    pair_constraints: Sequence[PairConstraint],
) -> dict[str, Any]:
    """Find a deterministic maximum-saving feasible deletion set.

    The branch-and-bound search operates only on already probed graph
    constraints; it does not execute or simulate any unprobed component subset.
    Lexicographically smaller retained-id tuples break equal-token ties.
    """

    weights = {
        str(row["id"]): float(row["maximum_removable_token_proxy_per_episode"])
        for row in components
    }
    removable = sorted(
        identifier
        for identifier, state in node_states.items()
        if state in {"REMOVABLE", "REMOVABLE_GIVEN(X)"}
    )
    conflicts = {
        frozenset((edge.left, edge.right))
        for edge in pair_constraints
        if edge.edge_state != "JOINTLY_REMOVABLE"
    }
    best_deleted: tuple[str, ...] = ()
    best_saving = -1.0

    suffix_upper = [0.0] * (len(removable) + 1)
    for index in range(len(removable) - 1, -1, -1):
        suffix_upper[index] = suffix_upper[index + 1] + max(
            0.0, weights[removable[index]]
        )

    def feasible_add(identifier: str, selected: Sequence[str]) -> bool:
        return all(
            frozenset((identifier, other)) not in conflicts for other in selected
        )

    def visit(index: int, selected: tuple[str, ...], saving: float) -> None:
        nonlocal best_deleted, best_saving
        if saving + suffix_upper[index] + 1e-12 < best_saving:
            return
        if index == len(removable):
            retained = tuple(sorted(set(weights) - set(selected)))
            best_retained = tuple(sorted(set(weights) - set(best_deleted)))
            if saving > best_saving + 1e-12 or (
                abs(saving - best_saving) <= 1e-12 and retained < best_retained
            ):
                best_saving = saving
                best_deleted = selected
            return
        identifier = removable[index]
        if feasible_add(identifier, selected):
            visit(
                index + 1,
                selected + (identifier,),
                saving + weights[identifier],
            )
        visit(index + 1, selected, saving)

    visit(0, (), 0.0)
    retained = sorted(set(weights) - set(best_deleted))
    return {
        "objective": "minimize retained controllable token proxy",
        "deleted_components": list(best_deleted),
        "retained_components": retained,
        "removed_token_proxy_per_episode": best_saving,
        "retained_token_proxy_per_episode": sum(weights.values()) - best_saving,
        "tie_break": "lexicographically smallest retained component-id tuple",
        "search": "deterministic branch-and-bound over certified graph only",
    }


def rank_pairs(
    removable_ids: Sequence[str], components: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Outcome-blind pair ranking from frozen mass and semantic metadata."""

    rows = {str(row["id"]): row for row in components}
    ranked: list[dict[str, Any]] = []
    ordered = sorted(set(removable_ids))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            left_row = rows[left]
            right_row = rows[right]
            same_group = left_row["semantic_group"] == right_row["semantic_group"]
            explicit_prior = (
                left_row["pre_outcome_metadata"].get("dependency_prior") == right
                or right_row["pre_outcome_metadata"].get("dependency_prior") == left
            )
            mass = float(left_row["maximum_removable_token_proxy_per_episode"]) + float(
                right_row["maximum_removable_token_proxy_per_episode"]
            )
            score = mass + (10000.0 if explicit_prior else 0.0) + (
                1000.0 if same_group else 0.0
            )
            ranked.append(
                {
                    "left": left,
                    "right": right,
                    "priority_score": score,
                    "combined_token_proxy_per_episode": mass,
                    "same_semantic_group": same_group,
                    "explicit_dependency_prior": explicit_prior,
                    "ranking_inputs": "frozen manifest metadata only; no pair outcome",
                }
            )
    return sorted(
        ranked,
        key=lambda row: (
            -float(row["priority_score"]),
            str(row["left"]),
            str(row["right"]),
        ),
    )


__all__ = [
    "ACTION_OUTPUT_INSTRUCTION",
    "COMPONENT_IDS",
    "GRAPH_EDGE_STATES",
    "GRAPH_NODE_STATES",
    "PairConstraint",
    "SYSTEM_ROLE",
    "action_family",
    "bind_content",
    "build_actor_messages",
    "content_hash",
    "flat_action_schema",
    "manifest_components",
    "minimum_retained_harness",
    "mixed_action_schema",
    "mutate_real_actor_messages",
    "parse_compact_family_line",
    "rank_pairs",
    "verify_content_binding",
]
