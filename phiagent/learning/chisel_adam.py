"""SkillAdam-inspired, validation-gated edits; never mutate Original CHISEL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import statistics

from phiagent.learning.risk_coverage_skill_evolution import (
    protected_success_candidate_acceptance,
)


UPSTREAM_REPOSITORY = "https://github.com/ruc-datalab/SkillAdam"
UPSTREAM_REVISION = "72ef9ba48bbadc059c8a2d099f1b7d531fc8288f"
METHOD_ID = "chisel_adam"
IDENTITY = "SkillAdam-inspired CHISEL adapter; not an official SkillAdam reproduction"


def content_hash(value: object) -> str:
    wire = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True)
class EditBudget:
    base: int = 4
    minimum: int = 1
    beta: float = 0.9
    variance_scale: float = 0.25

    def __post_init__(self) -> None:
        _integer(self.base, "base", 1)
        _integer(self.minimum, "minimum", 1)
        _number(self.beta, "beta")
        _number(self.variance_scale, "variance_scale")
        if self.minimum > self.base or not 0 <= self.beta <= 1 or self.variance_scale <= 0:
            raise ValueError("invalid adaptive edit-budget bounds")

    def limit(self, variance_ema: float, *, adaptive: bool = True) -> int:
        _number(variance_ema, "variance_ema")
        if variance_ema < 0 or type(adaptive) is not bool:
            raise ValueError("variance must be nonnegative and adaptive must be boolean")
        ratio = min(variance_ema / self.variance_scale, 1.0) if adaptive else 0.0
        return max(self.minimum, round(self.base * (1.0 - ratio)))


@dataclass(frozen=True)
class SkillEdit:
    profile: str
    old: str
    new: str
    problem: str

    def __post_init__(self) -> None:
        for name in ("profile", "old", "problem"):
            _text(getattr(self, name), name)
        if not isinstance(self.new, str):
            raise ValueError("replacement must be text, or empty text for deletion")
        if any(character in self.old + self.new for character in ("\n", "\r")):
            raise ValueError("edits must replace one complete line")
        if self.old.lstrip().startswith("#") or self.new.lstrip().startswith("#"):
            raise ValueError("profile headings are protected")
        if self.old == self.new:
            raise ValueError("an unchanged line is not an edit; use an empty patch")


def apply_edits(
    profiles: Mapping[str, str],
    edits: Sequence[SkillEdit],
    *,
    limit: int,
    maximum_profile_bytes: int,
) -> dict[str, str]:
    """Apply simultaneous exact-line edits without changing untouched bytes."""
    _integer(limit, "limit", 1)
    _integer(maximum_profile_bytes, "maximum_profile_bytes", 1)
    if not profiles:
        raise ValueError("at least one complete skill profile is required")
    for name, text in profiles.items():
        _text(name, "profile name")
        _text(text, "profile text")
        if len(text.encode("utf-8")) > maximum_profile_bytes:
            raise ValueError("the original profile exceeds the declared byte ceiling")
    if len(edits) > limit:
        raise ValueError("patch exceeds this iteration's edit budget")
    replacements: dict[tuple[str, str], str] = {}
    for edit in edits:
        if not isinstance(edit, SkillEdit) or edit.profile not in profiles:
            raise ValueError("patch names an absent profile or invalid edit")
        key = (edit.profile, edit.old)
        lines = profiles[edit.profile].splitlines()
        if key in replacements or lines.count(edit.old) != 1:
            raise ValueError("every edited line must occur exactly once in the original profile")
        replacements[key] = edit.new
    result = {}
    for name, text in profiles.items():
        parts = []
        for line in text.splitlines(keepends=True):
            old = line.rstrip("\r\n")
            if (name, old) not in replacements:
                parts.append(line)
            elif replacements[name, old]:
                parts.append(replacements[name, old] + line[len(old):])
        candidate = "".join(parts)
        if not candidate.strip() or len(candidate.encode("utf-8")) > maximum_profile_bytes:
            raise ValueError("candidate is empty or exceeds the declared profile byte ceiling")
        result[name] = candidate
    return result


@dataclass(frozen=True)
class CaseFeedback:
    group: str
    success: int
    metrics: Mapping[str, float]
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        _text(self.group, "group")
        if type(self.success) is not int or self.success not in (0, 1):
            raise ValueError("success must be a measured binary official outcome")
        if not self.metrics:
            raise ValueError("official per-case metrics are required")
        for name, value in self.metrics.items():
            _text(name, "metric")
            _number(value, name)
        _integer(self.prompt_tokens, "prompt_tokens", 1)
        _integer(self.completion_tokens, "completion_tokens")


@dataclass(frozen=True)
class BatchFeedback:
    method: str
    benchmark: str
    split: str
    protocol_sha256: str
    primary_metric: str
    cases: Mapping[str, CaseFeedback]
    overhead_tokens: int = 0
    unknown_cost_requests: int = 0

    def __post_init__(self) -> None:
        for name in ("method", "benchmark", "primary_metric"):
            _text(getattr(self, name), name)
        if self.split not in {"train", "validation", "screen", "confirmation"}:
            raise ValueError("an explicit training/validation/confirmation boundary is required")
        if (
            not isinstance(self.protocol_sha256, str)
            or len(self.protocol_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.protocol_sha256)
        ):
            raise ValueError("a frozen matched protocol SHA-256 is required")
        if not self.cases:
            raise ValueError("a complete nonempty task cohort is required")
        names = None
        for case_id, case in self.cases.items():
            _text(case_id, "case ID")
            if not isinstance(case, CaseFeedback):
                raise ValueError("every case requires typed feedback")
            if names is None:
                names = set(case.metrics)
            if set(case.metrics) != names or case.metrics.get(self.primary_metric) != case.success:
                raise ValueError("metric inventory or official primary outcome differs between cases")
        _integer(self.overhead_tokens, "overhead_tokens")
        _integer(self.unknown_cost_requests, "unknown_cost_requests")

    @property
    def total_tokens(self) -> int:
        return self.overhead_tokens + sum(
            case.prompt_tokens + case.completion_tokens for case in self.cases.values()
        )

    @property
    def means(self) -> dict[str, float]:
        names = next(iter(self.cases.values())).metrics
        return {
            name: math.fsum(case.metrics[name] for case in self.cases.values()) / len(self.cases)
            for name in names
        }


def comparable(reference: BatchFeedback, candidate: BatchFeedback) -> None:
    if (
        reference.benchmark != candidate.benchmark
        or reference.split != candidate.split
        or reference.protocol_sha256 != candidate.protocol_sha256
        or reference.primary_metric != candidate.primary_metric
        or set(reference.cases) != set(candidate.cases)
        or set(reference.means) != set(candidate.means)
        or any(reference.cases[key].group != candidate.cases[key].group for key in reference.cases)
    ):
        raise ValueError("benchmark, split, task/group, official metric, or matched protocol differs")


def group_delta_variance(reference: BatchFeedback, candidate: BatchFeedback) -> float:
    comparable(reference, candidate)
    if reference.split != "validation":
        raise ValueError("edit budgets may only consume paired validation feedback")
    groups: dict[str, list[float]] = {}
    for key, before in reference.cases.items():
        groups.setdefault(before.group, []).append(candidate.cases[key].success - before.success)
    if len(groups) < 2:
        raise ValueError("variance adaptation requires at least two independent validation groups")
    return statistics.variance(math.fsum(values) / len(values) for values in groups.values())


def assess_candidate(
    candidate: BatchFeedback,
    references: Sequence[BatchFeedback],
    *,
    expected_references: Sequence[str],
) -> dict:
    """Apply the new non-maximum-token target, not the old AutoSaddler ceiling."""
    expected = tuple(expected_references)
    by_name = {row.method: row for row in references}
    if (
        not expected or len(set(expected)) != len(expected)
        or len(by_name) != len(references) or set(by_name) != set(expected)
        or candidate.method in by_name
    ):
        raise ValueError("every prospectively declared comparison method is required exactly once")
    for reference in references:
        comparable(reference, candidate)
    known = not candidate.unknown_cost_requests and all(
        not row.unknown_cost_requests for row in references
    )
    differences = {
        row.method: {key: candidate.means[key] - value for key, value in row.means.items()}
        for row in references
    }
    strict_quality = all(value > 0 for row in differences.values() for value in row.values())
    observed_best = all(row[candidate.primary_metric] >= 0 for row in differences.values())
    largest_reference = max(row.total_tokens for row in references)
    nonmaximum = candidate.total_tokens < largest_reference if known else None
    return {
        "quality_differences": differences,
        "observed_primary_best_or_tied": observed_best,
        "strictly_better_all_official_metrics": strict_quality,
        "complete_cost_known": bool(known),
        "tokens_not_highest": nonmaximum,
        "largest_reference_tokens": largest_reference if known else None,
        "observed_user_target": observed_best and nonmaximum is True,
        "strict_development_target": strict_quality and nonmaximum is True,
        "paper_admitted": False,
    }


def decide_update(
    parent: BatchFeedback,
    candidate: BatchFeedback,
    references: Sequence[BatchFeedback],
    *,
    expected_references: Sequence[str],
    changed: bool,
) -> dict:
    comparable(parent, candidate)
    if parent.split != "validation" or type(changed) is not bool:
        raise ValueError("updates require validation feedback and an explicit changed flag")
    target = assess_candidate(candidate, references, expected_references=expected_references)
    ids = sorted(parent.cases)
    protected = protected_success_candidate_acceptance(
        unit_ids=ids,
        reference_outcomes=[parent.cases[key].success for key in ids],
        candidate_outcomes=[candidate.cases[key].success for key in ids],
        changed=changed,
        minimum_beneficial_flips=0,
        maximum_harmful_flips=0,
        require_strict_score_improvement=False,
    )
    deltas = {key: candidate.means[key] - value for key, value in parent.means.items()}
    checks = {
        "changed": changed,
        "protected_successes": protected.accepted,
        "official_metrics_nonregressed": all(value >= 0 for value in deltas.values()),
        "strict_quality_or_cost_gain": (
            any(value > 0 for value in deltas.values()) or candidate.total_tokens < parent.total_tokens
        ),
        "complete_cost_known": (
            not parent.unknown_cost_requests and target["complete_cost_known"]
        ),
        "tokens_not_highest": target["tokens_not_highest"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": not failed,
        "reason": "paired validation gates passed" if not failed else "failed: " + ", ".join(failed),
        "checks": checks,
        "parent_metric_deltas": deltas,
        "paired_outcomes": protected.paired_outcomes.to_dict(),
        "target": target,
    }


@dataclass
class OptimizationMemory:
    variance_ema: float = 0.0
    problems: dict[str, dict] = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        _number(self.variance_ema, "variance_ema")
        if self.variance_ema < 0:
            raise ValueError("variance_ema must be nonnegative")

    def budget(self, config: EditBudget, *, adaptive: bool) -> int:
        return config.limit(self.variance_ema, adaptive=adaptive)

    def register(self, issues: Sequence[Mapping], *, training_evidence: set[str]) -> None:
        seen = set()
        for issue in issues:
            if set(issue) != {"label", "description", "evidence"}:
                raise ValueError("issues require exactly label, description and training evidence")
            label, description, evidence = (issue[key] for key in ("label", "description", "evidence"))
            _text(label, "problem label")
            _text(description, "problem description")
            if (
                label in seen or not isinstance(evidence, list) or not evidence
                or any(not isinstance(item, str) for item in evidence)
                or not set(evidence) <= training_evidence
                or len(set(evidence)) != len(evidence)
            ):
                raise ValueError("problem evidence must name unique training-only observations")
            seen.add(label)
        for issue in issues:
            label = issue["label"]
            previous = self.problems.get(label, {"observations": []})
            self.problems[label] = {
                "description": issue["description"],
                "observations": [
                    *previous["observations"],
                    {"iteration": len(self.attempts) + 1, "evidence": list(issue["evidence"])},
                ],
            }

    def repeated(self, parent_profiles: Mapping[str, str], edits: Sequence[SkillEdit]) -> bool:
        parent_hash = content_hash(parent_profiles)
        patch_hash = content_hash([asdict(edit) for edit in edits])
        return any(
            attempt["parent_sha256"] == parent_hash and attempt["patch_sha256"] == patch_hash
            for attempt in self.attempts
        )

    def record(
        self,
        *,
        parent_profiles: Mapping[str, str],
        edits: Sequence[SkillEdit],
        decision: Mapping,
        variance: float,
        config: EditBudget,
    ) -> None:
        _number(variance, "paired group variance")
        if variance < 0 or any(edit.problem not in self.problems for edit in edits):
            raise ValueError("variance must be nonnegative and every edit must name a recorded problem")
        if type(decision.get("accepted")) is not bool:
            raise ValueError("an explicit measured acceptance decision is required")
        _text(decision.get("reason"), "decision reason")
        self.variance_ema = config.beta * self.variance_ema + (1 - config.beta) * variance
        self.attempts.append({
            "iteration": len(self.attempts) + 1,
            "parent_sha256": content_hash(parent_profiles),
            "patch_sha256": content_hash([asdict(edit) for edit in edits]),
            "edits": [asdict(edit) for edit in edits],
            "decision": dict(decision),
            "paired_group_variance": variance,
            "variance_ema": self.variance_ema,
        })

    def prompt_memory(self, *, enabled: bool, maximum_attempts: int = 4) -> dict:
        _integer(maximum_attempts, "maximum_attempts", 1)
        if type(enabled) is not bool:
            raise ValueError("memory ablation must be explicitly boolean")
        if not enabled:
            return {"enabled": False}
        recent = self.attempts[-maximum_attempts:]
        labels = {edit["problem"] for attempt in recent for edit in attempt["edits"]}
        return {
            "enabled": True,
            "problems": {key: self.problems[key] for key in sorted(labels)},
            "attempts": recent,
            "resolution_claimed": False,
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": "phiagent-chisel-adam-memory/v1",
            "variance_ema": self.variance_ema,
            "problems": self.problems,
            "attempts": self.attempts,
        }
