"""Outcome-blind routing over explicitly named, frozen harness profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType

from phiagent.learning.sparse_retrieval import SparseTextIndex


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    group_id: str
    text: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.task_id, self.group_id, self.text)
        ):
            raise ValueError("task identity, group, and public input text are required")


@dataclass(frozen=True)
class Outcome:
    score: int
    tokens: float | None

    def __post_init__(self) -> None:
        if type(self.score) is not int or self.score not in (0, 1):
            raise ValueError("a valid binary outcome is required, not an imputed score")
        if self.tokens is not None and (
            isinstance(self.tokens, bool)
            or not isinstance(self.tokens, (int, float))
            or not math.isfinite(self.tokens)
            or self.tokens < 0
        ):
            raise ValueError("tokens must be nonnegative and finite, or explicitly unknown")


@dataclass(frozen=True)
class TrainingExample:
    task: TaskInput
    outcomes: Mapping[str, Outcome]

    def __post_init__(self) -> None:
        if not self.outcomes or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, Outcome)
            for key, value in self.outcomes.items()
        ):
            raise ValueError("named profile outcomes are required")
        object.__setattr__(self, "outcomes", MappingProxyType(dict(self.outcomes)))


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "word"
    k: int = 64
    shrinkage: float = 8.0
    threshold: float = 0.04
    similarity_power: float = 1.0
    cost_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("word", "hybrid"):
            raise ValueError("unknown feature mode")
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be a positive integer")
        for field in ("shrinkage", "threshold", "similarity_power", "cost_weight"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite nonnegative number")
        if self.similarity_power == 0:
            raise ValueError("similarity_power must be positive")


@dataclass(frozen=True)
class RouteDecision:
    profile: str
    default_profile: str
    reason: str
    neighbors: tuple[tuple[str, float], ...]
    estimated_advantages: tuple[tuple[str, float], ...]


def grouped_folds(tasks: Sequence[TaskInput], *, folds: int, seed: int) -> dict[str, int]:
    if type(folds) is not int or folds < 2:
        raise ValueError("at least two folds are required")
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("tasks must be nonempty with unique identities")
    groups = sorted(
        {task.group_id for task in tasks},
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    if len(groups) < 2:
        raise ValueError("at least two independent groups are required")
    by_group = {group: rank % min(folds, len(groups)) for rank, group in enumerate(groups)}
    return {task.task_id: by_group[task.group_id] for task in tasks}


class HarnessRetriever:
    """A query contains no response, score, gold, or trajectory observations."""

    def __init__(
        self,
        training: Sequence[TrainingExample],
        *,
        profiles: Sequence[str],
        config: RetrievalConfig = RetrievalConfig(),
        default_profile: str | None = None,
    ) -> None:
        self.profiles = tuple(profiles)
        if len(set(self.profiles)) != len(self.profiles) or len(self.profiles) < 2:
            raise ValueError("at least two unique profiles are required")
        self.training = tuple(sorted(training, key=lambda example: example.task.task_id))
        if not self.training:
            raise ValueError("training examples are required")
        self.training_ids = frozenset(example.task.task_id for example in self.training)
        self.training_groups = frozenset(example.task.group_id for example in self.training)
        if len(self.training_ids) != len(self.training):
            raise ValueError("duplicate training task identity")
        if any(set(example.outcomes) != set(self.profiles) for example in self.training):
            raise ValueError("training must contain a complete matched profile matrix")
        self.config = config
        complete_costs = all(
            outcome.tokens is not None
            for example in self.training
            for outcome in example.outcomes.values()
        )
        if config.cost_weight and not complete_costs:
            raise ValueError("a cost-aware router requires observed training costs")
        if default_profile is not None and default_profile not in self.profiles:
            raise ValueError("default profile is outside the frozen library")

        def rank(profile: str) -> tuple[float, float, int]:
            cost = (
                sum(
                    outcome.tokens
                    for example in self.training
                    if (outcome := example.outcomes[profile]).tokens is not None
                )
                if complete_costs
                else 0.0
            )
            return (
                -sum(example.outcomes[profile].score for example in self.training),
                cost,
                self.profiles.index(profile),
            )

        self.default_profile = default_profile or min(self.profiles, key=rank)
        self.index = SparseTextIndex(
            [example.task.text for example in self.training], mode=config.mode
        )

    def route(self, task: TaskInput) -> RouteDecision:
        if task.task_id in self.training_ids or task.group_id in self.training_groups:
            raise ValueError("query task/group overlaps training; rebuild the held-group index")
        neighbors = self.index.neighbors(task.text, maximum=self.config.k)
        estimates = {}
        for profile in self.profiles:
            if profile == self.default_profile:
                continue
            utility = 0.0
            mass = 0.0
            for similarity, index in neighbors:
                weight = similarity**self.config.similarity_power
                reference = self.training[index].outcomes[self.default_profile]
                candidate = self.training[index].outcomes[profile]
                cost = 0.0
                if self.config.cost_weight:
                    if candidate.tokens is None or reference.tokens is None:
                        raise RuntimeError("validated training token usage disappeared")
                    cost = max(
                        0.0, (candidate.tokens - reference.tokens) / max(reference.tokens, 1.0)
                    )
                utility += weight * (
                    candidate.score - reference.score - self.config.cost_weight * cost
                )
                mass += weight
            denominator = mass + self.config.shrinkage
            estimates[profile] = utility / denominator if denominator else 0.0
        best = max(estimates, key=lambda profile: estimates[profile])
        selected = best if estimates[best] > self.config.threshold else self.default_profile
        return RouteDecision(
            profile=selected,
            default_profile=self.default_profile,
            reason=(
                "local_advantage"
                if selected != self.default_profile
                else "no_neighbors"
                if not neighbors
                else "below_threshold"
            ),
            neighbors=tuple(
                (self.training[index].task.task_id, similarity) for similarity, index in neighbors
            ),
            estimated_advantages=tuple(estimates.items()),
        )
