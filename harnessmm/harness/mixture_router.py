"""Opt-in, input-only mixtures of execution profiles and dependency-closed skills."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _names(values: tuple[str, ...], name: str) -> None:
    if (
        not isinstance(values, tuple)
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique nonempty strings")


def _fraction(value: float, name: str, *, allow_zero: bool = True) -> None:
    if (
        type(value) not in (int, float) or not math.isfinite(value)
        or not 0 <= value <= 1 or (not allow_zero and value == 0)
    ):
        raise ValueError(f"{name} must be a finite fraction")


def _random_key(seed: int, *parts: str) -> bytes:
    if type(seed) is not int:
        raise ValueError("the routing seed must be an integer")
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).digest()


@dataclass(frozen=True, slots=True)
class SkillSpec:
    skill_id: str
    text: str
    tags: tuple[str, ...]
    priority: int = 0
    dependencies: tuple[str, ...] = ()
    required: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.skill_id, str) or not self.skill_id.strip()
            or not isinstance(self.text, str) or not self.text.strip()
        ):
            raise ValueError("skills require a nonempty identity and exact text")
        _names(self.tags, "skill tags")
        _names(self.dependencies, "skill dependencies")
        if type(self.priority) is not int or type(self.required) is not bool:
            raise ValueError("skill priority and required flag have invalid types")


def select_skills(
    skills: Sequence[SkillSpec],
    *,
    active_tags: frozenset[str],
    top_k: int,
    mode: str = "routed",
    seed: int = 0,
    input_key: str = "",
) -> tuple[SkillSpec, ...]:
    """Select whole skills; required skills and their dependencies cannot be truncated."""
    _positive_integer(top_k, "skill top_k")
    if mode not in {"routed", "all", "none", "random"}:
        raise ValueError("unknown skill selection mode")
    if not isinstance(active_tags, frozenset):
        raise ValueError("active tags must be a frozen set")
    _names(tuple(active_tags), "active tags")
    _random_key(seed, input_key)
    registry = {skill.skill_id: skill for skill in skills}
    if not registry or len(registry) != len(skills):
        raise ValueError("a nonempty, uniquely named skill library is required")
    closures: dict[str, frozenset[str]] = {}

    def closure(identity: str, visiting: frozenset[str] = frozenset()) -> frozenset[str]:
        if identity not in registry:
            raise ValueError(f"unknown skill dependency: {identity}")
        if identity in visiting:
            raise ValueError(f"cyclic skill dependency: {identity}")
        if identity not in closures:
            members = {identity}
            for dependency in registry[identity].dependencies:
                members.update(closure(dependency, visiting | {identity}))
            closures[identity] = frozenset(members)
        return closures[identity]

    for identity in registry:
        closure(identity)
    selected: set[str] = set()
    for skill in skills:
        if skill.required:
            selected.update(closures[skill.skill_id])
    if mode != "all" and len(selected) > top_k:
        raise ValueError("mandatory skills and dependencies exceed skill top_k")
    if mode == "all":
        selected.update(registry)
    elif mode != "none":
        candidates = [
            skill for skill in skills
            if mode == "random" or active_tags.intersection(skill.tags)
        ]
        if mode == "random":
            candidates.sort(key=lambda skill: (_random_key(seed, input_key, skill.skill_id),
                                              skill.skill_id))
        else:
            candidates.sort(key=lambda skill: (
                -skill.priority, -len(active_tags.intersection(skill.tags)), skill.skill_id,
            ))
        for skill in candidates:
            proposal = selected | closures[skill.skill_id]
            if len(proposal) <= top_k:
                selected = proposal
    # Dependencies precede consumers, even when library registration order differs.
    result: list[SkillSpec] = []
    visited: set[str] = set()

    def emit(identity: str) -> None:
        if identity in visited:
            return
        for dependency in sorted(registry[identity].dependencies):
            emit(dependency)
        result.append(registry[identity])
        visited.add(identity)

    for identity in sorted(selected):
        emit(identity)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class HarnessProfile:
    harness_id: str
    context_token_budget: int | None
    max_output_tokens: int
    minimum_coverage: float

    def __post_init__(self) -> None:
        if not isinstance(self.harness_id, str) or not self.harness_id.strip():
            raise ValueError("harness profiles require a nonempty identity")
        if self.context_token_budget is not None:
            _positive_integer(self.context_token_budget, "context token budget")
        _positive_integer(self.max_output_tokens, "maximum output tokens")
        _fraction(self.minimum_coverage, "minimum coverage")


@dataclass(frozen=True, slots=True)
class HarnessCandidate:
    profile: HarnessProfile
    prompt_tokens: int
    evidence_coverage: float
    evidence_present: bool

    def __post_init__(self) -> None:
        _positive_integer(self.prompt_tokens, "full chat-template prompt tokens")
        _fraction(self.evidence_coverage, "observed lexical coverage")
        if type(self.evidence_present) is not bool:
            raise ValueError("evidence presence must be a boolean")

    @property
    def token_upper_bound(self) -> int:
        return self.prompt_tokens + self.profile.max_output_tokens


@dataclass(frozen=True, slots=True)
class HarnessDecision:
    selected_harness: str
    used_fallback: bool
    reason: str
    expected_prompt_tokens: int
    max_output_tokens: int
    token_upper_bound: int
    target_upper_bound_feasible: bool
    rejected_harnesses: tuple[str, ...]


def choose_harness(
    candidates: Sequence[HarnessCandidate],
    *,
    reference_id: str,
    enabled: bool = False,
    mode: str = "routed",
    fixed_harness_id: str | None = None,
    token_ratio_target: float = 0.1,
    seed: int = 0,
    input_key: str = "",
) -> HarnessDecision:
    """Lexical gates are routing heuristics, never performance certificates."""
    _fraction(token_ratio_target, "token ratio target", allow_zero=False)
    if type(enabled) is not bool or mode not in {"routed", "fixed", "random"}:
        raise ValueError("an explicit enable flag and supported harness mode are required")
    _random_key(seed, input_key)
    inventory = {candidate.profile.harness_id: candidate for candidate in candidates}
    if len(inventory) != len(candidates) or reference_id not in inventory:
        raise ValueError("unique profiles and an explicit incumbent are required")
    reference = inventory[reference_id]
    if reference.profile.context_token_budget is not None:
        raise ValueError("the incumbent must preserve its full original context")
    if any(
        candidate.profile.context_token_budget is None
        for identity, candidate in inventory.items() if identity != reference_id
    ):
        raise ValueError("only the explicit incumbent may have an unlimited context")
    if mode == "fixed":
        if fixed_harness_id not in inventory:
            raise ValueError("a fixed control must name a registered harness")
    elif fixed_harness_id is not None:
        raise ValueError("only fixed controls may specify a fixed harness")
    eligible = [
        candidate for identity, candidate in inventory.items()
        if identity != reference_id and candidate.evidence_present
        and candidate.evidence_coverage >= candidate.profile.minimum_coverage
    ]
    rejected = tuple(sorted(
        identity for identity, candidate in inventory.items()
        if identity != reference_id and candidate not in eligible
    ))
    if not enabled:
        selected, reason = reference, "router_disabled_preserve_incumbent"
    elif mode == "fixed":
        selected, reason = inventory[fixed_harness_id], "declared_fixed_profile_ablation"
    elif mode == "random":
        selected = min(candidates, key=lambda item: (
            _random_key(seed, input_key, item.profile.harness_id), item.profile.harness_id,
        ))
        reason = "declared_random_profile_ablation"
    elif eligible:
        selected = min([reference, *eligible], key=lambda item: (
            item.token_upper_bound, item.profile.harness_id,
        ))
        reason = (
            "incumbent_cheaper_than_eligible_profiles" if selected is reference
            else "cheapest_profile_passing_input_only_lexical_gate"
        )
    else:
        selected, reason = reference, "no_eligible_evidence_profile_preserve_incumbent"
    return HarnessDecision(
        selected_harness=selected.profile.harness_id,
        used_fallback=selected is reference,
        reason=reason,
        expected_prompt_tokens=selected.prompt_tokens,
        max_output_tokens=selected.profile.max_output_tokens,
        token_upper_bound=selected.token_upper_bound,
        # Incumbent completions can only increase the denominator.
        target_upper_bound_feasible=(
            selected.token_upper_bound <= token_ratio_target * reference.prompt_tokens
        ),
        rejected_harnesses=rejected,
    )


def count_matched_random_routes(
    routes: Mapping[str, str], *, strata: Mapping[str, str], seed: int,
) -> dict[str, str]:
    """Permute frozen profile assignments within input-only strata, not outcome strata."""
    if not routes or set(routes) != set(strata):
        raise ValueError("random controls require complete routes and matching strata")
    if any(
        not isinstance(key, str) or not key
        or not isinstance(routes[key], str) or not routes[key]
        or not isinstance(strata[key], str) or not strata[key]
        for key in routes
    ):
        raise ValueError("route identities, profile names and strata must be nonempty strings")
    _random_key(seed)
    result = {}
    for stratum in sorted(set(strata.values())):
        keys = sorted(key for key in routes if strata[key] == stratum)
        shuffled = sorted(keys, key=lambda key: (_random_key(seed, stratum, key), key))
        result.update({key: routes[donor] for key, donor in zip(keys, shuffled)})
        if Counter(result[key] for key in keys) != Counter(routes[key] for key in keys):
            raise RuntimeError("the random control changed a frozen profile count")
    return result
