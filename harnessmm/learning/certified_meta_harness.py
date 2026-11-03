"""Certificate-aware component evolution and fail-closed benchmark routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence


class ComponentRole(str, Enum):
    CAPABILITY = "capability"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    BUNDLE = "bundle"


class CertificateStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    DEV_PROMISING = "DEV_PROMISING"
    CERTIFIED_CAPABILITY = "CERTIFIED_CAPABILITY"
    CERTIFIED_EFFICIENCY = "CERTIFIED_EFFICIENCY"
    CERTIFIED_SAFETY = "CERTIFIED_SAFETY"
    CERTIFIED_BUNDLE = "CERTIFIED_BUNDLE"
    INTERACTION_ONLY = "INTERACTION_ONLY"
    UNRESOLVED = "UNRESOLVED"
    REJECTED = "REJECTED"


CERTIFIED_STATUSES = frozenset(
    {
        CertificateStatus.CERTIFIED_CAPABILITY,
        CertificateStatus.CERTIFIED_EFFICIENCY,
        CertificateStatus.CERTIFIED_SAFETY,
        CertificateStatus.CERTIFIED_BUNDLE,
    }
)


class EditKind(str, Enum):
    ADD = "add"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    benchmark: str
    slot: str
    role: ComponentRole
    artifact_sha256: str
    task_families: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("component_id", self.component_id),
            ("benchmark", self.benchmark),
            ("slot", self.slot),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lower-case SHA256")
        for values, label in (
            (self.task_families, "task_families"),
            (self.dependencies, "dependencies"),
            (self.conflicts, "conflicts"),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{label} must be unique and sorted")
        if self.component_id in self.dependencies:
            raise ValueError("a component cannot depend on itself")
        if self.component_id in self.conflicts:
            raise ValueError("a component cannot conflict with itself")

    def applies_to(self, benchmark: str, task_family: str | None) -> bool:
        if benchmark != self.benchmark:
            return False
        return not self.task_families or task_family in self.task_families


@dataclass(frozen=True)
class ComponentCertificate:
    component_id: str
    status: CertificateStatus
    artifact_sha256: str
    evidence_sha256: str
    capability_regret_upper: float | None
    token_saving_lower: float | None
    safety_violations_upper: int
    holm_adjusted_p: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("evidence_sha256", self.evidence_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef"
                for character in value
            ):
                raise ValueError(f"{label} must be a lower-case SHA256")
        for label, value in (
            ("capability_regret_upper", self.capability_regret_upper),
            ("token_saving_lower", self.token_saving_lower),
            ("holm_adjusted_p", self.holm_adjusted_p),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{label} must be finite when present")
        if self.safety_violations_upper < 0:
            raise ValueError("safety_violations_upper must be non-negative")
        if self.holm_adjusted_p is not None and not 0 <= self.holm_adjusted_p <= 1:
            raise ValueError("holm_adjusted_p must lie in [0, 1]")


@dataclass(frozen=True)
class ComponentGenome:
    benchmark: str
    component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.benchmark.strip():
            raise ValueError("benchmark must be non-empty")
        if tuple(sorted(set(self.component_ids))) != self.component_ids:
            raise ValueError("component_ids must be unique and sorted")


@dataclass(frozen=True)
class ComponentEdit:
    kind: EditKind
    component_id: str
    replacement_id: str | None = None

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if (self.kind is EditKind.REPLACE) != (self.replacement_id is not None):
            raise ValueError("only REPLACE edits name a replacement")


@dataclass(frozen=True)
class DevelopmentObservation:
    genome: ComponentGenome
    successes: int
    tasks: int
    total_tokens: int
    valid: bool = True

    def __post_init__(self) -> None:
        if self.tasks <= 0 or not 0 <= self.successes <= self.tasks:
            raise ValueError("development success counts are invalid")
        if self.total_tokens <= 0:
            raise ValueError("development token usage must be positive")

    @property
    def success_rate(self) -> float:
        return self.successes / self.tasks

    @property
    def tokens_per_task(self) -> float:
        return self.total_tokens / self.tasks


@dataclass(frozen=True)
class CertifiedPolicy:
    policy_id: str
    benchmark: str
    component_ids: tuple[str, ...]
    artifact_sha256: str
    evidence_sha256: str
    capability_regret_upper: float
    token_upper_per_task: float
    task_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.benchmark.strip():
            raise ValueError("policy identity must be non-empty")
        if tuple(sorted(set(self.component_ids))) != self.component_ids:
            raise ValueError("policy component_ids must be unique and sorted")
        if tuple(sorted(set(self.task_families))) != self.task_families:
            raise ValueError("policy task_families must be unique and sorted")
        for value in (
            self.capability_regret_upper,
            self.token_upper_per_task,
        ):
            if not math.isfinite(value):
                raise ValueError("policy bounds must be finite")
        if self.token_upper_per_task <= 0:
            raise ValueError("policy token upper bound must be positive")
        for value in (self.artifact_sha256, self.evidence_sha256):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("policy hashes must be lower-case SHA256")

    def applies_to(self, benchmark: str, task_family: str | None) -> bool:
        if benchmark != self.benchmark:
            return False
        return not self.task_families or task_family in self.task_families


@dataclass(frozen=True)
class RouteDecision:
    benchmark: str
    task_family: str | None
    selected_policy_id: str
    used_fallback: bool
    reason: str


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic Holm step-down adjusted p-values."""
    if not p_values:
        raise ValueError("Holm adjustment requires at least one hypothesis")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in p_values.values()
    ):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def validate_genome(
    genome: ComponentGenome,
    catalog: Mapping[str, ComponentSpec],
) -> None:
    """Reject invalid benchmark, dependency, conflict, or slot combinations."""
    selected = set(genome.component_ids)
    missing = selected - set(catalog)
    if missing:
        raise ValueError(f"unknown components: {sorted(missing)}")
    slots: dict[str, str] = {}
    for component_id in genome.component_ids:
        component = catalog[component_id]
        if component.benchmark != genome.benchmark:
            raise ValueError("genome mixes benchmark namespaces")
        if not set(component.dependencies) <= selected:
            raise ValueError(f"{component_id} has missing dependencies")
        if set(component.conflicts) & selected:
            raise ValueError(f"{component_id} has an active conflict")
        if component.slot in slots:
            raise ValueError(
                f"slot {component.slot} contains both "
                f"{slots[component.slot]} and {component_id}"
            )
        slots[component.slot] = component_id


def apply_edit(
    genome: ComponentGenome,
    edit: ComponentEdit,
    catalog: Mapping[str, ComponentSpec],
) -> ComponentGenome:
    """Apply one component-level edit and validate the resulting genome."""
    selected = set(genome.component_ids)
    if edit.kind is EditKind.ADD:
        if edit.component_id in selected:
            raise ValueError("ADD component is already active")
        selected.add(edit.component_id)
    elif edit.kind is EditKind.DELETE:
        if edit.component_id not in selected:
            raise ValueError("DELETE component is not active")
        selected.remove(edit.component_id)
    else:
        if edit.component_id not in selected:
            raise ValueError("REPLACE source component is not active")
        assert edit.replacement_id is not None
        selected.remove(edit.component_id)
        selected.add(edit.replacement_id)
    child = ComponentGenome(genome.benchmark, tuple(sorted(selected)))
    validate_genome(child, catalog)
    return child


def evolutionary_neighbors(
    genome: ComponentGenome,
    catalog: Mapping[str, ComponentSpec],
) -> tuple[tuple[ComponentEdit, ComponentGenome], ...]:
    """Enumerate every valid one-edit ADD, DELETE, and same-slot REPLACE child."""
    validate_genome(genome, catalog)
    selected = set(genome.component_ids)
    candidates: dict[tuple[str, ...], ComponentEdit] = {}
    for component_id, component in sorted(catalog.items()):
        if component.benchmark != genome.benchmark:
            continue
        if component_id not in selected:
            edit = ComponentEdit(EditKind.ADD, component_id)
            try:
                child = apply_edit(genome, edit, catalog)
            except ValueError:
                pass
            else:
                candidates.setdefault(child.component_ids, edit)
    for component_id in genome.component_ids:
        edit = ComponentEdit(EditKind.DELETE, component_id)
        try:
            child = apply_edit(genome, edit, catalog)
        except ValueError:
            pass
        else:
            candidates.setdefault(child.component_ids, edit)
        source = catalog[component_id]
        for replacement_id, replacement in sorted(catalog.items()):
            if (
                replacement_id in selected
                or replacement.benchmark != genome.benchmark
                or replacement.slot != source.slot
            ):
                continue
            edit = ComponentEdit(
                EditKind.REPLACE,
                component_id,
                replacement_id,
            )
            try:
                child = apply_edit(genome, edit, catalog)
            except ValueError:
                continue
            candidates.setdefault(child.component_ids, edit)
    return tuple(
        (candidates[key], ComponentGenome(genome.benchmark, key))
        for key in sorted(candidates)
    )


def development_pareto_frontier(
    observations: Iterable[DevelopmentObservation],
) -> tuple[DevelopmentObservation, ...]:
    """Keep valid points not dominated on success and measured token cost."""
    rows = tuple(observations)
    if not rows:
        return ()
    if len({row.genome for row in rows}) != len(rows):
        raise ValueError("development observations contain duplicate genomes")
    frontier = []
    for row in rows:
        if not row.valid:
            continue
        dominated = any(
            other.valid
            and other.genome.benchmark == row.genome.benchmark
            and other.success_rate >= row.success_rate
            and other.tokens_per_task <= row.tokens_per_task
            and (
                other.success_rate > row.success_rate
                or other.tokens_per_task < row.tokens_per_task
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return tuple(
        sorted(
            frontier,
            key=lambda row: (
                row.genome.benchmark,
                -row.success_rate,
                row.tokens_per_task,
                row.genome.component_ids,
            ),
        )
    )


class CertifiedComponentRegistry:
    """Immutable-by-ID component catalog with independent certificate binding."""

    def __init__(
        self,
        specs: Iterable[ComponentSpec],
        certificates: Iterable[ComponentCertificate],
    ) -> None:
        spec_rows = tuple(specs)
        certificate_rows = tuple(certificates)
        self._specs = {spec.component_id: spec for spec in spec_rows}
        self._certificates = {
            certificate.component_id: certificate
            for certificate in certificate_rows
        }
        if len(self._specs) != len(spec_rows):
            raise ValueError("component specifications contain duplicate IDs")
        if len(self._certificates) != len(certificate_rows):
            raise ValueError("component certificates contain duplicate IDs")
        if not set(self._certificates) <= set(self._specs):
            raise ValueError("certificate references an unknown component")
        for component_id, certificate in self._certificates.items():
            if (
                certificate.artifact_sha256
                != self._specs[component_id].artifact_sha256
            ):
                raise ValueError("certificate artifact hash differs")

    def certificate(self, component_id: str) -> ComponentCertificate | None:
        return self._certificates.get(component_id)

    def component_is_certified(
        self,
        component_id: str,
        *,
        capability_margin: float,
    ) -> bool:
        certificate = self._certificates.get(component_id)
        if certificate is None or certificate.status not in CERTIFIED_STATUSES:
            return False
        if certificate.safety_violations_upper != 0:
            return False
        if (
            certificate.capability_regret_upper is not None
            and certificate.capability_regret_upper > capability_margin
        ):
            return False
        if (
            certificate.status is CertificateStatus.CERTIFIED_EFFICIENCY
            and (
                certificate.token_saving_lower is None
                or certificate.token_saving_lower <= 0
            )
        ):
            return False
        if (
            certificate.status is CertificateStatus.CERTIFIED_CAPABILITY
            and (
                certificate.holm_adjusted_p is None
                or certificate.holm_adjusted_p > 0.05
            )
        ):
            return False
        return True

    def policy_is_certified(
        self,
        policy: CertifiedPolicy,
        *,
        capability_margin: float,
    ) -> bool:
        genome = ComponentGenome(policy.benchmark, policy.component_ids)
        try:
            validate_genome(genome, self._specs)
        except ValueError:
            return False
        return (
            policy.capability_regret_upper <= capability_margin
            and all(
                self.component_is_certified(
                    component_id,
                    capability_margin=capability_margin,
                )
                for component_id in policy.component_ids
            )
        )


class CertifiedMetaRouter:
    """Select the cheapest eligible certified policy or its frozen fallback."""

    def __init__(
        self,
        registry: CertifiedComponentRegistry,
        policies: Iterable[CertifiedPolicy],
        fallback_by_benchmark: Mapping[str, CertifiedPolicy],
        *,
        capability_margin: float,
    ) -> None:
        if not math.isfinite(capability_margin) or capability_margin < 0:
            raise ValueError("capability_margin must be finite and non-negative")
        self._registry = registry
        self._policies = tuple(policies)
        self._fallback = dict(fallback_by_benchmark)
        self._margin = capability_margin

    def route(
        self,
        *,
        benchmark: str,
        task_family: str | None = None,
        token_budget: float | None = None,
    ) -> RouteDecision:
        if benchmark not in self._fallback:
            raise KeyError(f"no frozen fallback for benchmark {benchmark}")
        candidates = [
            policy
            for policy in self._policies
            if policy.applies_to(benchmark, task_family)
            and self._registry.policy_is_certified(
                policy,
                capability_margin=self._margin,
            )
            and (
                token_budget is None
                or policy.token_upper_per_task <= token_budget
            )
        ]
        if not candidates:
            fallback = self._fallback[benchmark]
            return RouteDecision(
                benchmark=benchmark,
                task_family=task_family,
                selected_policy_id=fallback.policy_id,
                used_fallback=True,
                reason="no compatible certified component policy",
            )
        selected = min(
            candidates,
            key=lambda policy: (
                policy.token_upper_per_task,
                policy.capability_regret_upper,
                policy.policy_id,
            ),
        )
        return RouteDecision(
            benchmark=benchmark,
            task_family=task_family,
            selected_policy_id=selected.policy_id,
            used_fallback=False,
            reason="minimum certified token upper bound",
        )


__all__ = [
    "CERTIFIED_STATUSES",
    "CertificateStatus",
    "CertifiedComponentRegistry",
    "CertifiedMetaRouter",
    "CertifiedPolicy",
    "ComponentCertificate",
    "ComponentEdit",
    "ComponentGenome",
    "ComponentRole",
    "ComponentSpec",
    "DevelopmentObservation",
    "EditKind",
    "RouteDecision",
    "apply_edit",
    "development_pareto_frontier",
    "evolutionary_neighbors",
    "holm_adjust",
    "validate_genome",
]
