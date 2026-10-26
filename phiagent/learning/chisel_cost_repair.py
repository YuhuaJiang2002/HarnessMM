"""Source-mapped compression and paired quality/token gates for CHISEL."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

from phiagent.learning.chisel_adam import BatchFeedback, comparable
from phiagent.learning.risk_coverage_skill_evolution import paired_outcome_audit


SKILL_START = "\n\n## Skill\n"
SKILL_END = "\n\n## Answer Contract\n"
METHODS = ("compression_only", "repair_only", "compression_repair")
PRIMARY = "compression_repair"
IDENTITY = "CHISEL cost-constrained compression and repair; SkillAdam-inspired development adapter"


def system_token_count(tokenizer: Any, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("the native tokenizer must return positive raw-system token IDs")
    return len(tokens)


def inventory(system: str) -> dict[str, Any]:
    if system.count(SKILL_START) != 1 or system.count(SKILL_END) != 1:
        raise ValueError("the exact skill and answer-contract boundaries are required")
    prefix, rest = system.split(SKILL_START)
    skill, suffix = rest.split(SKILL_END)
    sections: list[dict] = []
    count = 0
    for line in skill.splitlines():
        if line.startswith("#"):
            sections.append({"heading": line, "rules": []})
        elif line.strip():
            if not sections or not line.startswith("- "):
                raise ValueError("only complete original rule bullets and headings are supported")
            count += 1
            sections[-1]["rules"].append({"id": f"R{count:03d}", "text": line})
    if not count:
        raise ValueError("the source skill has no rules")
    return {"prefix": prefix, "suffix": suffix, "sections": sections}


def compile_compression(system: str, response: str) -> tuple[str, dict]:
    """Require a complete source partition; lineage is not semantic equivalence."""
    source = inventory(system)
    proposal = json.loads(response)
    if not isinstance(proposal, dict) or set(proposal) != {"rules"} or not isinstance(proposal["rules"], list):
        raise ValueError("compression must return exactly a rules array")
    positions = {
        rule["id"]: (section_id, index)
        for section_id, section in enumerate(source["sections"])
        for index, rule in enumerate(section["rules"])
    }
    seen: set[str] = set()
    compiled: dict[int, list[tuple[int, str]]] = {}
    mappings = []
    for rule in proposal["rules"]:
        if not isinstance(rule, dict) or set(rule) != {"sources", "text"}:
            raise ValueError("each compressed rule requires sources and text")
        sources, text = rule["sources"], rule["text"]
        if (
            not isinstance(sources, list) or not sources
            or any(not isinstance(key, str) or key not in positions for key in sources)
            or len(set(sources)) != len(sources) or seen.intersection(sources)
        ):
            raise ValueError("original rules must be covered exactly once, without invented sources")
        section_ids = {positions[key][0] for key in sources}
        if len(section_ids) != 1:
            raise ValueError("merging rules across original sections is forbidden")
        if (
            not isinstance(text, str) or not text.startswith("- ") or not text[2:].strip()
            or any(character in text for character in ("\r", "\n"))
        ):
            raise ValueError("compressed rules must be complete single-line bullets")
        section_id = next(iter(section_ids))
        compiled.setdefault(section_id, []).append((min(positions[key][1] for key in sources), text))
        seen.update(sources)
        mappings.append({"sources": sorted(sources), "text": text})
    if seen != set(positions):
        raise ValueError("compression omitted original rules")
    blocks = []
    for section_id, section in enumerate(source["sections"]):
        lines = [text for _, text in sorted(compiled.get(section_id, []))]
        blocks.append("\n".join([section["heading"], *lines]))
    candidate = source["prefix"] + SKILL_START + "\n\n".join(blocks) + SKILL_END + source["suffix"]
    return candidate, {
        "source_rules": len(positions), "compressed_rules": len(mappings),
        "mappings": mappings, "all_rules_have_lineage": True,
        "semantic_equivalence_certified": False,
    }


def require_envelope(candidate: str, original: str) -> None:
    left, right = inventory(candidate), inventory(original)
    if (
        left["prefix"] != right["prefix"] or left["suffix"] != right["suffix"]
        or [s["heading"] for s in left["sections"]] != [s["heading"] for s in right["sections"]]
    ):
        raise ValueError("task instruction, section inventory or answer contract changed")


def profile_token_gate(
    profiles: Mapping[str, str], original: Mapping[str, str], count_tokens: Callable[[str], int],
    *, maximum_per_mille: int,
) -> dict:
    if (
        set(profiles) != {"delete", "modify"} or set(original) != set(profiles)
        or type(maximum_per_mille) is not int or not 0 < maximum_per_mille <= 1000
    ):
        raise ValueError("both routed profiles and a fixed token-ratio ceiling are required")
    rows = {}
    for name in sorted(profiles):
        require_envelope(profiles[name], original[name])
        before, after = count_tokens(original[name]), count_tokens(profiles[name])
        if any(type(value) is not int or value <= 0 for value in (before, after)):
            raise ValueError("positive exact tokenizer counts are required")
        rows[name] = {
            "before": before, "after": after,
            "passed": after * 1000 <= before * maximum_per_mille,
        }
    return {"passed": all(row["passed"] for row in rows.values()), "profiles": rows,
            "maximum_per_mille": maximum_per_mille, "quality_certified": False}


def joint_gate(candidate: BatchFeedback, original: BatchFeedback, *, minimum_saving_per_mille: int) -> dict:
    comparable(original, candidate)
    if type(minimum_saving_per_mille) is not int or not 0 < minimum_saving_per_mille < 1000:
        raise ValueError("a positive prospective token-saving threshold is required")
    known = candidate.unknown_cost_requests == original.unknown_cost_requests == 0
    deltas = {name: candidate.means[name] - value for name, value in original.means.items()}
    quality = all(value > 0 for value in deltas.values())
    cost = (
        candidate.total_tokens * 1000 <= original.total_tokens * (1000 - minimum_saving_per_mille)
        if known else None
    )
    ids = sorted(original.cases)
    paired = paired_outcome_audit(
        unit_ids=ids,
        reference_outcomes=[original.cases[key].success for key in ids],
        candidate_outcomes=[candidate.cases[key].success for key in ids],
    )
    return {
        "reference": original.method, "official_metric_deltas": deltas,
        "all_official_metrics_strictly_better": quality,
        "complete_cost_known": known, "token_target_passed": cost,
        "minimum_saving_per_mille": minimum_saving_per_mille,
        "actual_token_saving_fraction": 1 - candidate.total_tokens / original.total_tokens if known else None,
        "observed_joint_target_passed": quality and cost is True,
        "paired_outcomes": paired.to_dict(), "paper_admitted": False,
    }


def repair_decision(parent: BatchFeedback, candidate: BatchFeedback, *, changed: bool) -> dict:
    comparable(parent, candidate)
    if parent.split != "validation" or type(changed) is not bool:
        raise ValueError("repair decisions require validation-only feedback and an explicit changed flag")
    ids = sorted(parent.cases)
    paired = paired_outcome_audit(
        unit_ids=ids, reference_outcomes=[parent.cases[key].success for key in ids],
        candidate_outcomes=[candidate.cases[key].success for key in ids],
    )
    deltas = {name: candidate.means[name] - value for name, value in parent.means.items()}
    checks = {
        "changed": changed,
        "known_cost": candidate.unknown_cost_requests == parent.unknown_cost_requests == 0,
        "no_lost_solved_cases": not paired.harmful_flip_ids,
        "official_metrics_nonregressed": all(value >= 0 for value in deltas.values()),
        "strict_quality_or_cost_gain": (
            any(value > 0 for value in deltas.values()) or candidate.total_tokens < parent.total_tokens
        ),
        "complete_tokens_nonincreased": candidate.total_tokens <= parent.total_tokens,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": not failures, "reason": "paired repair passed" if not failures else "failed: " + ", ".join(failures),
        "checks": checks, "metric_deltas": deltas, "paired_outcomes": paired.to_dict(),
    }
