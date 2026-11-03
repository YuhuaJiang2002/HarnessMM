"""Fail-closed routing for task-family-local textual skill edits.

This module is benchmark-agnostic.  A caller supplies a stable task-group
label, an incumbent skill tree, and one candidate tree whose mutation has been
audited as local.  The candidate prompt is activated only on its declared
route; all other and unknown groups receive the byte-identical incumbent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence


_LEXEME = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def skill_tree_snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def skill_tree_hash(root: Path) -> str:
    return _canonical_hash(skill_tree_snapshot(root))


def render_skill_tree(root: Path) -> str:
    snapshot = skill_tree_snapshot(root)
    sections = []
    for relative, content in snapshot.items():
        if relative.endswith("/SKILL.md"):
            sections.append(f"## {relative}\n\n{content.strip()}")
    return "\n\n".join(sections)


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    task_groups: tuple[str, ...]
    editable_skill_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.route_id.strip() or not self.task_groups:
            raise ValueError("route_id and task_groups must be non-empty")
        if any(not group.strip() for group in self.task_groups):
            raise ValueError("route task_groups must not contain empty values")
        if len(set(self.task_groups)) != len(self.task_groups):
            raise ValueError("route task_groups must be unique")
        if not self.editable_skill_names:
            raise ValueError("a route must declare editable skill names")
        if any("/" in name or not name.strip() for name in self.editable_skill_names):
            raise ValueError("editable skill names must be safe path components")


@dataclass(frozen=True)
class ConditionalRouteRule:
    candidate_id: str
    required_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.candidate_id, str) or not self.candidate_id.strip()
                or not isinstance(self.required_features, tuple) or not self.required_features
                or any(not isinstance(name, str) or not name.strip() for name in self.required_features)
                or len(set(self.required_features)) != len(self.required_features)):
            raise ValueError("A conditional rule requires a candidate and unique explicit features")


def resolve_conditional_candidate(
    parent_features: Mapping[str, bool | None], rules: Sequence[ConditionalRouteRule]
) -> str | None:
    if any(value is not None and type(value) is not bool for value in parent_features.values()):
        raise ValueError("Route inference accepts only boolean or unknown parent features")
    candidates = {
        rule.candidate_id for rule in rules
        if all(parent_features.get(name) is True for name in rule.required_features)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def fit_conditional_rule(
    observations: Sequence[Mapping[str, Any]],
    *,
    feature_names: tuple[str, ...],
    required_seeds: tuple[int, ...],
    eligible_candidates: frozenset[str],
    excluded_case_ids: frozenset[str] = frozenset(),
    minimum_source_cases: int = 2,
) -> dict[str, Any]:
    if (not observations or not feature_names or len(set(feature_names)) != len(feature_names)
            or len(required_seeds) < 3 or len(set(required_seeds)) != len(required_seeds)
            or any(type(seed) is not int for seed in required_seeds)
            or type(minimum_source_cases) is not int or minimum_source_cases < 2):
        raise ValueError("Freeze nonempty features, three or more seeds and at least two source cases")
    candidates = sorted({row["candidate_id"] for row in observations})
    cases = {row["case_id"] for row in observations}
    if cases & excluded_case_ids:
        raise ValueError("Held-out source cases must never enter router fitting")
    expected = {(candidate, case, seed) for candidate in candidates for case in cases for seed in required_seeds}
    identities = [(row["candidate_id"], row["case_id"], row["seed"]) for row in observations]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Every candidate and source case must retain all frozen seeds")
    metric_count = len(observations[0]["raw_delta"])
    features_by_input = {}
    for row in observations:
        if (row["split"] != "train" or not isinstance(row["case_id"], str) or not row["case_id"]
                or not isinstance(row["candidate_id"], str) or not row["candidate_id"]):
            raise ValueError("Router fitting accepts identified training records only")
        features = row["parent_features"]
        if set(features) != set(feature_names) or any(value is not None and type(value) is not bool for value in features.values()):
            raise ValueError("Require exactly the prospectively declared parent features")
        identity = (row["case_id"], row["seed"])
        if features_by_input.setdefault(identity, dict(features)) != features:
            raise ValueError("Parent route features cannot depend on the candidate outcome")
        for kind in ("raw_delta", "selected_delta"):
            vector = row[kind]
            if (not metric_count or len(vector) != metric_count
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
                raise ValueError("Require complete finite paired metric vectors")
        if any(type(row[field]) is not bool for field in ("repair_adopted", "preserves_pass_gates")):
            raise ValueError("Repair and preservation evidence must be explicit booleans")

    def mean_vector(records, kind):
        return tuple(sum(row[kind][index] for row in records) / len(records)
                     for index in range(metric_count)) if records else (0.0,) * metric_count

    diagnostics = []
    winner = None
    for candidate in candidates:
        for feature in feature_names:
            active = [row for row in observations if row["candidate_id"] == candidate
                      and row["parent_features"][feature] is True]
            raw = mean_vector(active, "raw_delta")
            selected = mean_vector(active, "selected_delta")
            per_seed = {str(seed): {kind: mean_vector([row for row in active if row["seed"] == seed], kind)
                                   for kind in ("raw_delta", "selected_delta")} for seed in required_seeds}
            checks = {
                "multiple_source_cases": len({row["case_id"] for row in active}) >= minimum_source_cases,
                "all_seeds_observed": {row["seed"] for row in active} == set(required_seeds),
                "adopted_repairs_in_multiple_seeds": len({row["seed"] for row in active if row["repair_adopted"]}) >= 2,
                "raw_metric_dominance": all(value >= 0 for value in raw) and any(value > 0 for value in raw),
                "selected_metric_dominance": all(value >= 0 for value in selected) and any(value > 0 for value in selected),
                "each_seed_nonregression": all(value >= 0 for kinds in per_seed.values()
                                               for vector in kinds.values() for value in vector),
                "preserves_parent_pass_gates": bool(active) and all(row["preserves_pass_gates"] for row in active),
            }
            item = {"candidate_id": candidate, "feature": feature, "active_records": len(active),
                    "supporting_cases": sorted({row["case_id"] for row in active}),
                    "raw_delta": raw, "selected_delta": selected, "per_seed_deltas": per_seed,
                    "effect_checks": checks, "conditional_effect_passed": all(checks.values()),
                    "input_consistency_admitted": candidate in eligible_candidates}
            diagnostics.append(item)
            if not item["conditional_effect_passed"] or not item["input_consistency_admitted"]:
                continue
            if winner is None:
                winner = item
                continue
            better = all(value >= reference for value, reference in zip(selected, winner["selected_delta"]))
            strict = any(value > reference for value, reference in zip(selected, winner["selected_delta"]))
            if better and (strict or len(active) < winner["active_records"]):
                winner = item
    rules = [] if winner is None else [asdict(ConditionalRouteRule(winner["candidate_id"], (winner["feature"],)))]
    return {"rules": rules, "region_diagnostics": diagnostics,
            "training_source_cases": len(cases), "training_seeds": list(required_seeds),
            "status": "CONDITIONAL_RULE_FROZEN" if rules else "ABSTAIN_NO_SUPPORTED_LOCAL_REPAIR",
            "out_of_region_action": "Byte-identical incumbent output",
            "fit_objective": "Conditional raw/selected Pareto benefit with seed and PASS preservation; incumbent on failure",
            "rule_selection": "Sorted candidate/feature order, Pareto-improving selected deltas, then fewer activations on ties",
            "independent_confirmation_established": False}


@dataclass(frozen=True)
class LocalEditAudit:
    valid: bool
    changed_paths: tuple[str, ...]
    changed_skill_names: tuple[str, ...]
    changed_lexemes: int
    maximum_changed_skills: int
    maximum_changed_lexemes: int
    gates: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutedPrompt:
    route_id: str | None
    task_group: str
    selected_tree: str
    selected_tree_sha256: str
    prompt: str
    abstained: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompt_sha256"] = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        del value["prompt"]
        return value


@dataclass(frozen=True)
class RouteLocalAcceptance:
    accepted: bool
    route_id: str
    route_reference_successes: int
    route_candidate_successes: int
    beneficial_flip_ids: tuple[str, ...]
    harmful_flip_ids: tuple[str, ...]
    group_gains: Mapping[str, float]
    gates: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_route_table(routes: Sequence[RouteSpec]) -> dict[str, RouteSpec]:
    table: dict[str, RouteSpec] = {}
    route_ids: set[str] = set()
    for route in routes:
        if route.route_id in route_ids:
            raise ValueError(f"duplicate route_id: {route.route_id}")
        route_ids.add(route.route_id)
        for group in route.task_groups:
            if group in table:
                raise ValueError(f"task group belongs to multiple routes: {group}")
            table[group] = route
    if not table:
        raise ValueError("at least one route is required")
    return table


def _changed_lexemes(before: str, after: str) -> int:
    left = _LEXEME.findall(before)
    right = _LEXEME.findall(after)
    total = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=left, b=right, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            total += (i2 - i1) + (j2 - j1)
    return total


def audit_local_edit(
    incumbent_root: Path,
    candidate_root: Path,
    route: RouteSpec,
    *,
    maximum_changed_skills: int = 1,
    maximum_changed_lexemes: int = 512,
) -> LocalEditAudit:
    if maximum_changed_skills <= 0 or maximum_changed_lexemes <= 0:
        raise ValueError("local edit budgets must be positive")
    before = skill_tree_snapshot(incumbent_root)
    after = skill_tree_snapshot(candidate_root)
    paths = tuple(
        sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    )
    skill_names = tuple(sorted({path.split("/", 1)[0] for path in paths}))
    lexemes = sum(_changed_lexemes(before.get(path, ""), after.get(path, "")) for path in paths)
    gates = {
        "artifact_changed": bool(paths),
        "skill_count_budget": len(skill_names) <= maximum_changed_skills,
        "route_scope": set(skill_names).issubset(route.editable_skill_names),
        "skill_markdown_only": all(
            len(path.split("/")) == 2 and path.endswith("/SKILL.md") for path in paths
        ),
        "lexeme_budget": lexemes <= maximum_changed_lexemes,
    }
    return LocalEditAudit(
        valid=all(gates.values()),
        changed_paths=paths,
        changed_skill_names=skill_names,
        changed_lexemes=lexemes,
        maximum_changed_skills=maximum_changed_skills,
        maximum_changed_lexemes=maximum_changed_lexemes,
        gates=gates,
    )


def resolve_routed_prompt(
    *,
    task_group: str,
    route_table: Mapping[str, RouteSpec],
    candidate_route_id: str,
    incumbent_root: Path,
    candidate_root: Path,
) -> RoutedPrompt:
    group = str(task_group).strip()
    route = route_table.get(group)
    activate = route is not None and route.route_id == candidate_route_id
    selected_root = candidate_root if activate else incumbent_root
    return RoutedPrompt(
        route_id=route.route_id if route is not None else None,
        task_group=group,
        selected_tree="candidate" if activate else "incumbent",
        selected_tree_sha256=skill_tree_hash(selected_root),
        prompt=render_skill_tree(selected_root),
        abstained=not activate,
    )


def rollout_with_local_router(
    *,
    items: Sequence[Mapping[str, Any]],
    route_table: Mapping[str, RouteSpec],
    candidate_route_id: str,
    incumbent_prompt: str,
    candidate_prompt: str,
    output_root: Path,
    rollout: Callable[[Sequence[Mapping[str, Any]], str, str], Sequence[Mapping[str, Any]]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if not items:
        raise ValueError("routed rollout requires at least one item")
    ids = [str(item.get("id") or "") for item in items]
    if any(not unit_id for unit_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("routed items require unique non-empty ids")
    candidate_items = []
    incumbent_items = []
    decisions = []
    for item in items:
        group = str(
            item.get("task_group") or item.get("task_type") or item.get("group") or ""
        ).strip()
        route = route_table.get(group)
        activate = route is not None and route.route_id == candidate_route_id
        (candidate_items if activate else incumbent_items).append(item)
        decisions.append(
            {
                "id": str(item["id"]),
                "task_group": group,
                "route_id": route.route_id if route else None,
                "selected_tree": "candidate" if activate else "incumbent",
                "abstained": not activate,
            }
        )
    rows: list[Mapping[str, Any]] = []
    if candidate_items:
        rows.extend(rollout(candidate_items, candidate_prompt, str(output_root / "candidate")))
    if incumbent_items:
        rows.extend(rollout(incumbent_items, incumbent_prompt, str(output_root / "incumbent")))
    by_id = {str(row.get("id") or ""): row for row in rows}
    if set(by_id) != set(ids) or len(rows) != len(ids):
        raise RuntimeError("routed rollout did not return one aligned row per item")
    return [by_id[unit_id] for unit_id in ids], {
        "candidate_route_id": candidate_route_id,
        "candidate_units": len(candidate_items),
        "incumbent_fallback_units": len(incumbent_items),
        "unknown_group_units": sum(row["route_id"] is None for row in decisions),
        "decisions": decisions,
    }


def route_local_acceptance(
    *,
    unit_ids: Sequence[str],
    group_ids: Sequence[str],
    reference_outcomes: Sequence[bool | int],
    candidate_outcomes: Sequence[bool | int],
    route: RouteSpec,
    local_edit: LocalEditAudit,
    nonroute_prompt_identity: bool,
) -> RouteLocalAcceptance:
    lengths = {
        len(unit_ids),
        len(group_ids),
        len(reference_outcomes),
        len(candidate_outcomes),
    }
    if len(lengths) != 1 or not unit_ids:
        raise ValueError("route-local acceptance inputs must align and be non-empty")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_ids must be unique")
    if any(not str(group).strip() for group in group_ids):
        raise ValueError("group_ids must not contain empty values")
    route_groups = set(route.task_groups)
    route_indices = [index for index, group in enumerate(group_ids) if group in route_groups]
    beneficial = tuple(
        unit_ids[index]
        for index in route_indices
        if not bool(reference_outcomes[index]) and bool(candidate_outcomes[index])
    )
    harmful = tuple(
        unit_ids[index]
        for index in route_indices
        if bool(reference_outcomes[index]) and not bool(candidate_outcomes[index])
    )
    group_gains = {
        group: sum(
            int(bool(candidate_outcomes[index])) - int(bool(reference_outcomes[index]))
            for index, observed in enumerate(group_ids)
            if observed == group
        )
        / sum(observed == group for observed in group_ids)
        for group in sorted(set(group_ids))
    }
    reference_successes = sum(bool(reference_outcomes[index]) for index in route_indices)
    candidate_successes = sum(bool(candidate_outcomes[index]) for index in route_indices)
    gates = {
        "local_edit_valid": local_edit.valid,
        "route_has_validation_units": bool(route_indices),
        "route_strict_gain": candidate_successes > reference_successes,
        "route_zero_harmful_flips": not harmful,
        "no_task_group_regression": min(group_gains.values()) >= 0.0,
        "nonroute_prompt_identity": bool(nonroute_prompt_identity),
    }
    return RouteLocalAcceptance(
        accepted=all(gates.values()),
        route_id=route.route_id,
        route_reference_successes=reference_successes,
        route_candidate_successes=candidate_successes,
        beneficial_flip_ids=beneficial,
        harmful_flip_ids=harmful,
        group_gains=group_gains,
        gates=gates,
    )


__all__ = [
    "ConditionalRouteRule",
    "LocalEditAudit",
    "RouteLocalAcceptance",
    "RouteSpec",
    "RoutedPrompt",
    "audit_local_edit",
    "build_route_table",
    "fit_conditional_rule",
    "render_skill_tree",
    "resolve_routed_prompt",
    "resolve_conditional_candidate",
    "rollout_with_local_router",
    "route_local_acceptance",
    "skill_tree_hash",
]
