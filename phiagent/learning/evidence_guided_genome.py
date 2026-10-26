"""Evidence-guided front end for typed, certificate-bound harness evolution.

This module handles candidate discovery and safe materialization only. Historical
evidence never authorizes deployment; independent benchmark certificates remain
the authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from phiagent.learning.certified_meta_harness import (
    ComponentEdit,
    ComponentGenome,
    ComponentSpec,
    EditKind,
    apply_edit,
    validate_genome,
)


MAX_DISCOVERY_BYTES_PER_SOURCE = 65_536


class OverrideMode(str, Enum):
    INHERIT = "inherit"
    RESET = "reset"
    OVERRIDE = "override"


class EvidenceKind(str, Enum):
    KNOWLEDGE = "knowledge"
    POSITIVE_PATTERN = "positive_pattern"
    ANTI_PATTERN = "anti_pattern"


class UpdateStatus(str, Enum):
    CURRENT = "CURRENT"
    UPDATE_ALLOWED = "UPDATE_ALLOWED"
    MODIFIED_LOCAL_COPY = "MODIFIED_LOCAL_COPY"
    GENOME_ID_MISMATCH = "GENOME_ID_MISMATCH"


@dataclass(frozen=True)
class DiscoveryRecord:
    """Hash-only metadata for one bounded historical evidence source."""

    source_id: str
    source_sha256: str
    content_bytes: int
    relevance_score: float
    recency_rank: int

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        _validate_sha256(self.source_sha256, "source_sha256")
        if self.content_bytes <= 0:
            raise ValueError("content_bytes must be positive")
        if not math.isfinite(self.relevance_score):
            raise ValueError("relevance_score must be finite")
        if self.recency_rank < 0:
            raise ValueError("recency_rank must be non-negative")

    @property
    def inspected_bytes(self) -> int:
        return min(self.content_bytes, MAX_DISCOVERY_BYTES_PER_SOURCE)


@dataclass(frozen=True)
class DiscoveryWindow:
    records: tuple[DiscoveryRecord, ...]
    inspected_bytes: int
    truncated_source_ids: tuple[str, ...]
    selection_rule: str = (
        "descending relevance, ascending recency rank, lexical source ID; "
        "bounded metadata selection only, never fitness or certification"
    )


@dataclass(frozen=True)
class ProposalEvidence:
    """Structured attribution from a selected evidence source to one edit."""

    source_id: str
    source_sha256: str
    component_id: str
    kind: EvidenceKind
    support_count: int
    contradiction_count: int
    inspected_bytes: int

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.component_id.strip():
            raise ValueError("evidence identity must be non-empty")
        _validate_sha256(self.source_sha256, "source_sha256")
        if (
            self.support_count < 0
            or self.contradiction_count < 0
            or self.inspected_bytes <= 0
            or self.inspected_bytes > MAX_DISCOVERY_BYTES_PER_SOURCE
        ):
            raise ValueError("proposal evidence counts or byte bound are invalid")


@dataclass(frozen=True)
class ComponentFieldOverride:
    """One inherit/reset/override decision on a component-owned runtime path."""

    component_id: str
    path: str
    mode: OverrideMode
    value: Any = None

    def __post_init__(self) -> None:
        if not self.component_id.strip() or not _valid_path(self.path):
            raise ValueError("override component and path must be valid")
        if self.mode is OverrideMode.OVERRIDE and self.value is None:
            raise ValueError("OVERRIDE requires a non-null value")
        if self.mode is not OverrideMode.OVERRIDE and self.value is not None:
            raise ValueError("INHERIT and RESET cannot carry a value")


@dataclass(frozen=True)
class GenomeProposal:
    parent: ComponentGenome
    child: ComponentGenome
    edit: ComponentEdit
    evidence: tuple[ProposalEvidence, ...]
    intent: str
    proposal_sha256: str

    @property
    def support(self) -> int:
        return sum(row.support_count for row in self.evidence)

    @property
    def contradictions(self) -> int:
        return sum(row.contradiction_count for row in self.evidence)

    @property
    def evidence_sources(self) -> int:
        return len({row.source_id for row in self.evidence})


@dataclass(frozen=True)
class ProjectionUse:
    component_id: str
    path: str
    consumer_id: str
    applied: bool
    runtime_value_sha256: str | None

    def __post_init__(self) -> None:
        if (
            not self.component_id.strip()
            or not _valid_path(self.path)
            or not self.consumer_id.strip()
        ):
            raise ValueError("projection identity must be valid")
        if self.runtime_value_sha256 is not None:
            _validate_sha256(
                self.runtime_value_sha256,
                "runtime_value_sha256",
            )


@dataclass(frozen=True)
class UpdateDecision:
    status: UpdateStatus
    update_allowed: bool
    reason: str


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lower-case SHA256")


def _valid_path(path: str) -> bool:
    fields = path.split(".")
    return bool(fields) and all(
        field
        and field.replace("_", "").replace("-", "").isalnum()
        for field in fields
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def select_discovery_window(
    records: Iterable[DiscoveryRecord],
    *,
    max_sources: int,
    max_total_bytes: int,
) -> DiscoveryWindow:
    """Select a deterministic bounded history window for proposal discovery."""
    rows = tuple(records)
    if max_sources <= 0 or max_total_bytes <= 0:
        raise ValueError("discovery bounds must be positive")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError("discovery sources contain duplicate IDs")
    selected = []
    inspected = 0
    for row in sorted(
        rows,
        key=lambda item: (
            -item.relevance_score,
            item.recency_rank,
            item.source_id,
        ),
    ):
        if len(selected) >= max_sources:
            break
        bounded_bytes = row.inspected_bytes
        if inspected + bounded_bytes > max_total_bytes:
            continue
        selected.append(row)
        inspected += bounded_bytes
    return DiscoveryWindow(
        records=tuple(selected),
        inspected_bytes=inspected,
        truncated_source_ids=tuple(
            sorted(
                row.source_id
                for row in selected
                if row.content_bytes > MAX_DISCOVERY_BYTES_PER_SOURCE
            )
        ),
    )


def validate_component_ownership(
    catalog: Mapping[str, ComponentSpec],
    owned_paths: Mapping[str, Sequence[str]],
    allowed_edits: Mapping[str, Sequence[EditKind]],
) -> None:
    """Require exclusive path ownership and explicit mutation permissions."""
    if set(owned_paths) != set(catalog) or set(allowed_edits) != set(catalog):
        raise ValueError("ownership and edit schemas must cover the catalog")
    owner_by_path: dict[str, str] = {}
    for component_id in sorted(catalog):
        paths = tuple(owned_paths[component_id])
        edits = tuple(allowed_edits[component_id])
        if (
            not paths
            or len(set(paths)) != len(paths)
            or not all(_valid_path(path) for path in paths)
        ):
            raise ValueError(f"{component_id} has invalid owned paths")
        if not edits or len(set(edits)) != len(edits):
            raise ValueError(f"{component_id} has invalid allowed edits")
        if any(not isinstance(edit, EditKind) for edit in edits):
            raise TypeError(f"{component_id} has a non-EditKind permission")
        for path in paths:
            if path in owner_by_path:
                raise ValueError(
                    f"path {path} is owned by both "
                    f"{owner_by_path[path]} and {component_id}"
                )
            owner_by_path[path] = component_id


def compile_component_projection(
    *,
    base_state: Mapping[str, Any],
    genome: ComponentGenome,
    catalog: Mapping[str, ComponentSpec],
    owned_paths: Mapping[str, Sequence[str]],
    allowed_edits: Mapping[str, Sequence[EditKind]],
    overrides: Sequence[ComponentFieldOverride],
) -> dict[str, Any]:
    """Apply component-owned tri-state overrides to a flat runtime state."""
    validate_component_ownership(catalog, owned_paths, allowed_edits)
    validate_genome(genome, catalog)
    active = set(genome.component_ids)
    seen_paths: set[str] = set()
    result = dict(base_state)
    for override in overrides:
        if override.component_id not in active:
            raise ValueError("inactive component cannot project a field")
        if override.path not in set(owned_paths[override.component_id]):
            raise ValueError("component attempted to project an unowned path")
        if override.path in seen_paths:
            raise ValueError("runtime path has multiple override declarations")
        seen_paths.add(override.path)
        permissions = set(allowed_edits[override.component_id])
        if override.mode is OverrideMode.INHERIT:
            continue
        if override.mode is OverrideMode.RESET:
            if EditKind.DELETE not in permissions:
                raise ValueError("component is not permitted to reset fields")
            result.pop(override.path, None)
            continue
        if EditKind.REPLACE not in permissions:
            raise ValueError("component is not permitted to override fields")
        result[override.path] = override.value
    return result


def build_genome_proposal(
    *,
    parent: ComponentGenome,
    edit: ComponentEdit,
    catalog: Mapping[str, ComponentSpec],
    evidence: Sequence[ProposalEvidence],
    intent: str,
) -> GenomeProposal:
    """Create a content-addressed proposal with hash-bound discovery evidence."""
    if not intent.strip() or not evidence:
        raise ValueError("proposal requires intent and evidence")
    child = apply_edit(parent, edit, catalog)
    evidence_rows = tuple(
        sorted(
            evidence,
            key=lambda row: (
                row.source_id,
                row.component_id,
                row.kind.value,
            ),
        )
    )
    if any(
        row.component_id
        not in {edit.component_id, edit.replacement_id}
        for row in evidence_rows
    ):
        raise ValueError("proposal evidence names an unrelated component")
    unsigned = {
        "parent": {
            "benchmark": parent.benchmark,
            "component_ids": list(parent.component_ids),
        },
        "child": {
            "benchmark": child.benchmark,
            "component_ids": list(child.component_ids),
        },
        "edit": {
            "kind": edit.kind.value,
            "component_id": edit.component_id,
            "replacement_id": edit.replacement_id,
        },
        "evidence": [
            {
                "source_id": row.source_id,
                "source_sha256": row.source_sha256,
                "component_id": row.component_id,
                "kind": row.kind.value,
                "support_count": row.support_count,
                "contradiction_count": row.contradiction_count,
                "inspected_bytes": row.inspected_bytes,
            }
            for row in evidence_rows
        ],
        "intent": intent,
    }
    return GenomeProposal(
        parent=parent,
        child=child,
        edit=edit,
        evidence=evidence_rows,
        intent=intent,
        proposal_sha256=canonical_hash(unsigned),
    )


def rank_discovery_proposals(
    proposals: Iterable[GenomeProposal],
) -> tuple[GenomeProposal, ...]:
    """Rank candidates for DEV evaluation, never for deployment admission."""
    rows = tuple(proposals)
    if len({row.proposal_sha256 for row in rows}) != len(rows):
        raise ValueError("proposal list contains duplicate content identities")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -(row.support - row.contradictions),
                -row.evidence_sources,
                row.contradictions,
                row.proposal_sha256,
            ),
        )
    )


def audit_runtime_projection(
    overrides: Sequence[ComponentFieldOverride],
    uses: Sequence[ProjectionUse],
) -> dict[str, Any]:
    """Reject declared non-inherited fields that do not reach one consumer."""
    expected = {
        (row.component_id, row.path): row
        for row in overrides
        if row.mode is not OverrideMode.INHERIT
    }
    if len(expected) != sum(
        row.mode is not OverrideMode.INHERIT for row in overrides
    ):
        raise ValueError("projection overrides contain duplicate active paths")
    observed: dict[tuple[str, str], ProjectionUse] = {}
    for row in uses:
        key = (row.component_id, row.path)
        if key not in expected or key in observed:
            raise ValueError("projection use is undeclared or duplicated")
        observed[key] = row
    missing = sorted(set(expected) - set(observed))
    unapplied = []
    for key, use in observed.items():
        override = expected[key]
        expected_hash = (
            None
            if override.mode is OverrideMode.RESET
            else canonical_hash(override.value)
        )
        if (
            not use.applied
            or use.runtime_value_sha256 != expected_hash
        ):
            unapplied.append(key)
    unapplied.sort()
    passed = not missing and not unapplied
    return {
        "status": "PASS" if passed else "FAIL",
        "declared_active_fields": len(expected),
        "observed_consumers": len(observed),
        "missing": [list(key) for key in missing],
        "unapplied": [list(key) for key in unapplied],
        "projection_authorized": passed,
    }


def decide_seed_update(
    *,
    installed_genome_id: str,
    installed_content_sha256: str,
    seeded_content_sha256: str,
    incoming_genome_id: str,
    incoming_content_sha256: str,
) -> UpdateDecision:
    """Allow refresh only for an unmodified copy with the same stable identity."""
    if not installed_genome_id.strip() or not incoming_genome_id.strip():
        raise ValueError("genome IDs must be non-empty")
    for value in (
        installed_content_sha256,
        seeded_content_sha256,
        incoming_content_sha256,
    ):
        _validate_sha256(value, "genome content hash")
    if installed_genome_id != incoming_genome_id:
        return UpdateDecision(
            UpdateStatus.GENOME_ID_MISMATCH,
            False,
            "incoming genome has a different stable identity",
        )
    if installed_content_sha256 == incoming_content_sha256:
        return UpdateDecision(
            UpdateStatus.CURRENT,
            False,
            "installed genome already matches the incoming content",
        )
    if installed_content_sha256 != seeded_content_sha256:
        return UpdateDecision(
            UpdateStatus.MODIFIED_LOCAL_COPY,
            False,
            "installed genome differs from its recorded seed content",
        )
    return UpdateDecision(
        UpdateStatus.UPDATE_ALLOWED,
        True,
        "same stable identity and unmodified seeded copy",
    )


__all__ = [
    "MAX_DISCOVERY_BYTES_PER_SOURCE",
    "ComponentFieldOverride",
    "DiscoveryRecord",
    "DiscoveryWindow",
    "EvidenceKind",
    "GenomeProposal",
    "OverrideMode",
    "ProjectionUse",
    "ProposalEvidence",
    "UpdateDecision",
    "UpdateStatus",
    "audit_runtime_projection",
    "build_genome_proposal",
    "canonical_hash",
    "compile_component_projection",
    "decide_seed_update",
    "rank_discovery_proposals",
    "select_discovery_window",
    "validate_component_ownership",
]
