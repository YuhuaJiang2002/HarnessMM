"""Compile repair proposals from provenance-bound public-interface rule IDs.

This constrains what a proposer can emit. It does not certify the correctness
of a supplied rule catalog or the effectiveness of the resulting intervention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from phiagent.learning.counterfactual_repair import (
    PATCH_SCHEMA,
    RUNTIME_SEMANTICS,
    RepairPatch,
    RepairValidationError,
    parse_patch,
    patch_to_dict,
)
from phiagent.learning.stage_aware_adm import canonical_hash, reject_outcome_fields


CATALOG_SCHEMA = "phiagent-public-repair-rule-catalog/v1"
PROPOSAL_SCHEMA = "phiagent-grounded-repair-proposal/v1"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PATCH_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,95}")
_HASH = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True, slots=True)
class PublicRepairRule:
    rule_id: str
    instruction: str
    public_source: str


@dataclass(frozen=True, slots=True)
class RepairRuleCatalog:
    catalog_id: str
    interface_source_sha256: str
    rules: tuple[PublicRepairRule, ...]
    catalog_sha256: str


@dataclass(frozen=True, slots=True)
class CompiledGroundedRepair:
    patch: RepairPatch
    catalog_sha256: str
    proposal_sha256: str
    behavior_sha256: str
    rule_ids: tuple[str, ...]
    unexecuted_rationale: str


def catalog_to_dict(catalog: RepairRuleCatalog, *, include_hash: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": CATALOG_SCHEMA,
        "catalog_id": catalog.catalog_id,
        "interface_source_sha256": catalog.interface_source_sha256,
        "rules": [
            {
                "rule_id": rule.rule_id,
                "instruction": rule.instruction,
                "public_source": rule.public_source,
            }
            for rule in catalog.rules
        ],
    }
    if include_hash:
        result["catalog_sha256"] = catalog.catalog_sha256
    return result


def parse_catalog(value: Mapping[str, Any]) -> RepairRuleCatalog:
    if not isinstance(value, Mapping):
        raise RepairValidationError("rule catalog must be an object")
    required = {"schema_version", "catalog_id", "interface_source_sha256", "rules"}
    if required - set(value) or set(value) - (required | {"catalog_sha256"}):
        raise RepairValidationError("rule catalog fields differ from the closed schema")
    reject_outcome_fields(value, path="$.public_rule_catalog")
    if value["schema_version"] != CATALOG_SCHEMA:
        raise RepairValidationError("unsupported public rule catalog schema")
    if (
        not isinstance(value["catalog_id"], str)
        or _IDENTIFIER.fullmatch(value["catalog_id"]) is None
    ):
        raise RepairValidationError("catalog_id must be a bounded identifier")
    source_hash = value["interface_source_sha256"]
    if not isinstance(source_hash, str) or _HASH.fullmatch(source_hash) is None:
        raise RepairValidationError("public interface source must have an immutable SHA256 binding")
    raw_rules = value["rules"]
    if not isinstance(raw_rules, list) or not 1 <= len(raw_rules) <= 32:
        raise RepairValidationError("catalog must contain 1..32 explicit public rules")
    rules = []
    for row in raw_rules:
        if not isinstance(row, Mapping) or set(row) != {"rule_id", "instruction", "public_source"}:
            raise RepairValidationError("public rule must declare only id, instruction and source")
        if not isinstance(row["rule_id"], str) or _IDENTIFIER.fullmatch(row["rule_id"]) is None:
            raise RepairValidationError("rule_id must be a bounded identifier")
        if (
            not isinstance(row["instruction"], str)
            or not row["instruction"].strip()
            or len(row["instruction"]) > 450
        ):
            raise RepairValidationError("a public rule instruction must contain 1..450 characters")
        if not isinstance(row["public_source"], str) or not row["public_source"].strip():
            raise RepairValidationError("each rule needs an explicit public-interface source")
        rules.append(PublicRepairRule(row["rule_id"], row["instruction"], row["public_source"]))
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise RepairValidationError("duplicate public rule IDs")
    unsigned = RepairRuleCatalog(value["catalog_id"], source_hash, tuple(rules), "")
    digest = canonical_hash(catalog_to_dict(unsigned, include_hash=False))
    if "catalog_sha256" in value and value["catalog_sha256"] != digest:
        raise RepairValidationError("public rule catalog self-hash differs")
    return RepairRuleCatalog(unsigned.catalog_id, source_hash, unsigned.rules, digest)


def compile_grounded_repair(
    proposal: Mapping[str, Any],
    *,
    catalog: RepairRuleCatalog,
    expected_parent_sha256: str,
) -> CompiledGroundedRepair:
    """Never insert model-generated rationale or free-form instructions."""
    checked_catalog = parse_catalog(catalog_to_dict(catalog))
    if not isinstance(proposal, Mapping):
        raise RepairValidationError("grounded proposal must be an object")
    required = {
        "schema_version",
        "patch_id",
        "parent_harness_sha256",
        "catalog_sha256",
        "rule_ids",
        "trigger",
        "memory_characters",
        "max_entries",
    }
    if required - set(proposal) or set(proposal) - (required | {"rationale"}):
        raise RepairValidationError("grounded proposals cannot supply free-form directive fields")
    reject_outcome_fields(proposal, path="$.grounded_proposal")
    if proposal["schema_version"] != PROPOSAL_SCHEMA:
        raise RepairValidationError("unsupported grounded proposal schema")
    if (
        not isinstance(proposal["patch_id"], str)
        or _PATCH_IDENTIFIER.fullmatch(proposal["patch_id"]) is None
    ):
        raise RepairValidationError("patch_id must be a bounded nonexecuting identifier")
    if proposal["catalog_sha256"] != checked_catalog.catalog_sha256:
        raise RepairValidationError("proposal is bound to another public-interface catalog")
    if proposal["parent_harness_sha256"] != expected_parent_sha256:
        raise RepairValidationError("proposal is bound to another parent")
    rule_ids = proposal["rule_ids"]
    if (
        not isinstance(rule_ids, list)
        or not 1 <= len(rule_ids) <= 4
        or any(not isinstance(rule_id, str) for rule_id in rule_ids)
        or len(set(rule_ids)) != len(rule_ids)
    ):
        raise RepairValidationError("select 1..4 distinct approved rule IDs")
    rules = {rule.rule_id: rule for rule in checked_catalog.rules}
    if not set(rule_ids) <= set(rules):
        raise RepairValidationError(
            "proposal refers to a rule outside the public-interface catalog"
        )
    rationale = proposal.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > 4000:
        raise RepairValidationError("audit-only rationale must be bounded text")
    directive = "\n".join(
        f"{rank}. {rules[rule_id].instruction}" for rank, rule_id in enumerate(rule_ids, 1)
    )
    patch = parse_patch(
        {
            "schema_version": PATCH_SCHEMA,
            "patch_id": proposal["patch_id"],
            "parent_harness_sha256": proposal["parent_harness_sha256"],
            "component": "recovery_memory",
            "operation": "ADD",
            "directive": directive,
            "trigger": proposal["trigger"],
            "memory_characters": proposal["memory_characters"],
            "max_entries": proposal["max_entries"],
        }
    )
    behavior = patch_to_dict(patch, include_hash=False)
    behavior.pop("patch_id")
    behavior["runtime_semantics"] = RUNTIME_SEMANTICS
    return CompiledGroundedRepair(
        patch=patch,
        catalog_sha256=checked_catalog.catalog_sha256,
        proposal_sha256=canonical_hash(dict(proposal)),
        behavior_sha256=canonical_hash(behavior),
        rule_ids=tuple(rule_ids),
        unexecuted_rationale=rationale,
    )
