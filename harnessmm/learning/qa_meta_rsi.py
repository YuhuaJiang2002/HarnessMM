"""Bounded, data-free contracts for a MetaRSI-inspired QA harness adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import re


IDENTITY = "MetaRSI-inspired harness-only adaptation"
PRIMARY_SOURCE = "https://arxiv.org/html/2609.06396v2"
PARTITIONS = ("train", "inner_validation", "meta_validation", "final_screen")
BRANCHES = ("fixed_rsi", "meta_rsi")
FINAL_ARMS = (
    "original_chisel", "fixed_rsi", "meta_rsi", "no_skill", "skillopt",
    "autosaddler", "delete", "original_aa",
)
SECTION_IDS = {
    "Concise Answer Normalization": "normalization",
    "Context-Grounded Evidence Matching": "evidence",
    "Clue Interpretation and Answer Type": "answer_type",
    "Trivia / Jeopardy Snippet Formats": "snippet_format",
    "Common Clue Traps": "traps",
}
INITIAL_STRATEGY = (
    "Diagnose recurring training failures before editing. Preserve evidence grounding, "
    "answer-type discrimination and solved cases. Shorten or remove redundant, overly "
    "specific or conflicting rules, and rephrase a small number of harmful rules into "
    "general instructions. Allocate edits across both routed profiles when useful. "
    "Aim to improve exact-match and F1 while saving at least ten percent of complete "
    "inference tokens. Do not confuse brevity with correct answer normalization."
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def initial_skill(profiles: Mapping[str, str]) -> dict:
    """Assign persistent IDs once; subsequent patches never match natural-language lines."""
    if set(profiles) != {"delete", "modify"}:
        raise ValueError("both archived DELETE/MODIFY profiles are required")
    layout, rules = {}, {}
    for profile, text in profiles.items():
        section, seen, entries, counts = None, set(), [], {}
        contract = False
        for line in text.splitlines(keepends=True):
            content = line.rstrip("\r\n")
            ending = line[len(content):]
            if content == "## Answer Contract":
                contract, section = True, None
            elif content.startswith("## ") and content[3:] in SECTION_IDS:
                if contract:
                    raise ValueError("editable sections cannot follow the answer contract")
                section = SECTION_IDS[content[3:]]
                if section in seen:
                    raise ValueError("duplicate original section")
                seen.add(section)
            if section is not None and content.startswith("- "):
                counts[section] = counts.get(section, 0) + 1
                rule_id = f"{profile}.{section}.r{counts[section]:02d}"
                rules[rule_id] = content
                entries.append({"rule_id": rule_id, "ending": ending})
            else:
                entries.append({"literal": line})
        if seen != set(SECTION_IDS.values()) or not contract or "<answer>" not in text:
            raise ValueError("complete original sections and answer contract are required")
        layout[profile] = entries
    value = {
        "object_type": "qa_skill", "version": 0, "parent_sha256": None,
        "source_profiles_sha256": {name: text_hash(text) for name, text in profiles.items()},
        "layout": layout, "rules": rules,
    }
    if render_skill(value) != dict(profiles):
        raise ValueError("initial skill must reproduce every original source byte")
    return value


def render_skill(skill: Mapping) -> dict[str, str]:
    if skill["object_type"] != "qa_skill" or set(skill["layout"]) != {"delete", "modify"}:
        raise ValueError("unknown skill object")
    result, used = {}, []
    for profile, entries in skill["layout"].items():
        pieces = []
        for entry in entries:
            if set(entry) == {"literal"}:
                pieces.append(entry["literal"])
            elif set(entry) == {"rule_id", "ending"}:
                rule_id = entry["rule_id"]
                if not rule_id.startswith(profile + "."):
                    raise ValueError("rule/profile binding differs")
                pieces.append(skill["rules"][rule_id] + entry["ending"])
                used.append(rule_id)
            else:
                raise ValueError("only source literals and bound rules may be rendered")
        result[profile] = "".join(pieces)
    if len(used) != len(set(used)) or set(used) != set(skill["rules"]):
        raise ValueError("every stable rule ID must occur exactly once")
    return result


def skill_content_hash(skill: Mapping) -> str:
    return digest(render_skill(skill))


def initial_policy() -> dict:
    return {
        "object_type": "qa_improvement_policy", "policy_id": "training_only_rule_proposer",
        "version": 0, "parent_sha256": None, "strategy": INITIAL_STRATEGY,
    }


def normalized(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def memorization_violations(text: str, forbidden: list[str]) -> list[dict]:
    """Locate the existing guard's matches in the original, unnormalized text."""
    folded_parts, origins = [], []
    for index, character in enumerate(text):
        folded = character.casefold()
        folded_parts.append(folded)
        origins.extend([index] * len(folded))
    folded = "".join(folded_parts)
    tokens = list(re.finditer(r"\w+", folded))
    words = [token.group() for token in tokens]
    violations = []
    for entry in forbidden:
        needle = normalized(entry).split()
        if not needle:
            continue
        for offset in range(len(words) - len(needle) + 1):
            if words[offset:offset + len(needle)] != needle:
                continue
            start = origins[tokens[offset].start()]
            end = origins[tokens[offset + len(needle) - 1].end() - 1] + 1
            violations.append({
                "kind": "forbidden_literal", "constraint": entry,
                "start": start, "end": end, "matched_text": text[start:end],
                "span_unit": "unicode_codepoints",
            })
    for match in re.finditer(r"\bT\d{2,4}\b", text):
        violations.append({
            "kind": "evidence_id", "constraint": "training_evidence_id",
            "start": match.start(), "end": match.end(), "matched_text": match.group(),
            "span_unit": "unicode_codepoints",
        })
    for fragment in (
        "ignore previous", "ignore the answer contract", "change the evaluator",
        "change the router", "cuda_visible_devices", "```", "<script", "import os",
    ):
        offset = folded.find(fragment)
        while offset >= 0:
            start, end = origins[offset], origins[offset + len(fragment) - 1] + 1
            violations.append({
                "kind": "protected_surface", "constraint": fragment,
                "start": start, "end": end, "matched_text": text[start:end],
                "span_unit": "unicode_codepoints",
            })
            offset = folded.find(fragment, offset + 1)
    return violations


def prohibit_memorization(text: str, forbidden: list[str]) -> None:
    violations = memorization_violations(text, forbidden)
    if violations:
        messages = {
            "forbidden_literal": "case identity, answer or case-specific question phrase in mutable text",
            "evidence_id": "training evidence IDs belong in evidence fields, not reusable instructions",
            "protected_surface": "mutable text attempts to alter a protected surface or supply executable code",
        }
        raise ValueError(messages[violations[0]["kind"]])


def apply_skill_proposal(
    parent: dict, proposal: object, *, evidence_ids: set[str], forbidden: list[str],
    max_changes: int, max_rule_characters: int, max_profile_growth_bytes: int,
) -> tuple[dict, dict]:
    required = {"parent_skill_sha256", "hypothesis", "risk", "evidence", "changes"}
    if not isinstance(proposal, dict) or set(proposal) != required:
        raise ValueError("skill proposal fields differ from the protected schema")
    if proposal["parent_skill_sha256"] != digest(parent):
        raise ValueError("skill proposal is not bound to the exact current parent")
    if any(not isinstance(proposal[key], str) or len(proposal[key]) > 1600
           for key in ("hypothesis", "risk")):
        raise ValueError("hypothesis and risk require bounded text")
    evidence = proposal["evidence"]
    if (not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence))
            or any(not isinstance(item, str) for item in evidence) or not set(evidence) <= evidence_ids):
        raise ValueError("edits must cite only supplied training evidence")
    changes = proposal["changes"]
    if not isinstance(changes, list) or not 0 < len(changes) <= max_changes:
        raise ValueError("a candidate requires a nonempty bounded stable-ID patch")
    result, seen = deepcopy(parent), set()
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"rule_id", "text"}:
            raise ValueError("each change requires exactly rule_id and text")
        rule_id, text = change["rule_id"], change["text"]
        if (not isinstance(rule_id, str) or rule_id not in parent["rules"] or rule_id in seen
                or not isinstance(text, str) or len(text) > max_rule_characters
                or "\n" in text or "\r" in text or "\x00" in text
                or (text and (not text.startswith("- ") or len(text) <= 2))):
            raise ValueError("unknown/repeated rule ID or out-of-surface rule text")
        if text == parent["rules"][rule_id]:
            raise ValueError("a counted rule edit must actually change its value")
        prohibit_memorization(text, forbidden)
        result["rules"][rule_id] = text
        seen.add(rule_id)
    result["version"], result["parent_sha256"] = parent["version"] + 1, digest(parent)
    original, candidate = render_skill(parent), render_skill(result)
    if any(len(candidate[name].encode()) > len(original[name].encode()) + max_profile_growth_bytes
           for name in original):
        raise ValueError("per-step profile byte-growth bound exceeded")
    if skill_content_hash(parent) == skill_content_hash(result):
        raise ValueError("byte-identical harnesses cannot constitute an update")
    return result, {
        "parent_sha256": digest(parent), "child_sha256": digest(result),
        "parent_content_sha256": skill_content_hash(parent),
        "child_content_sha256": skill_content_hash(result),
        "changed_rule_ids": sorted(seen), "proposal": proposal,
        "layout_and_answer_contract_unchanged": result["layout"] == parent["layout"],
        "semantic_equivalence_certified": False,
    }


def apply_policy_proposal(
    parent: dict, proposal: object, *, evolve: bool, evidence_ids: set[str],
    forbidden: list[str], max_strategy_characters: int,
) -> tuple[dict, list[dict]]:
    if not isinstance(proposal, dict) or set(proposal) != {
        "parent_policy_sha256", "strategy", "diagnostics",
    }:
        raise ValueError("policy/reflection proposal schema differs")
    if proposal["parent_policy_sha256"] != digest(parent):
        raise ValueError("policy proposal is not bound to the exact parent")
    strategy, diagnostics = proposal["strategy"], proposal["diagnostics"]
    if not isinstance(strategy, str) or not strategy.strip() or len(strategy) > max_strategy_characters:
        raise ValueError("improvement strategy requires bounded nonempty text")
    if not evolve and strategy != parent["strategy"]:
        raise ValueError("the fixed-policy reflection may not change its strategy")
    if not isinstance(diagnostics, list) or len(diagnostics) > 4:
        raise ValueError("at most four training-only diagnostics are allowed")
    for entry in diagnostics:
        if (not isinstance(entry, dict) or set(entry) != {"evidence", "text"}
                or not isinstance(entry["text"], str) or not 0 < len(entry["text"]) <= 800
                or not isinstance(entry["evidence"], list) or not entry["evidence"]
                or any(not isinstance(item, str) for item in entry["evidence"])
                or not set(entry["evidence"]) <= evidence_ids):
            raise ValueError("reflection evidence must belong to the current training signal")
    result = deepcopy(parent)
    if evolve and strategy != parent["strategy"]:
        prohibit_memorization(strategy, forbidden)
        result.update({
            "version": parent["version"] + 1, "parent_sha256": digest(parent), "strategy": strategy,
        })
    return result, diagnostics


def select_partitions(
    identities: Mapping[str, str], excluded: set[str], *, counts: Mapping[str, int],
    salt: str, maximum_tasks: int,
) -> dict:
    if set(counts) != set(PARTITIONS) or any(type(counts[key]) is not int or counts[key] < 1 for key in counts):
        raise ValueError("four nonempty whole-question-group partitions are required")
    if any(not isinstance(key, str) or not key or not isinstance(group, str) or not group
           for key, group in identities.items()):
        raise ValueError("task and group identities must be nonempty strings")
    ordered = sorted(set(identities.values()) - excluded, key=lambda group: (digest([salt, group]), group))
    if len(ordered) < sum(counts.values()):
        raise ValueError("insufficient disjoint source groups")
    groups, tasks, offset = {}, {}, 0
    for split in PARTITIONS:
        groups[split] = sorted(ordered[offset:offset + counts[split]])
        chosen = set(groups[split])
        tasks[split] = sorted(task for task, group in identities.items() if group in chosen)
        offset += counts[split]
    if sum(map(len, tasks.values())) > maximum_tasks:
        raise ValueError("whole groups exceed the prospective task cap; no duplicate dropping")
    return {"groups": groups, "task_ids": tasks}


def summarize(rows: Mapping[str, Mapping]) -> dict:
    if not rows:
        raise ValueError("empty result matrices cannot be scored")
    for row in rows.values():
        if any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or not 0 <= row[key] <= 1
               for key in ("exact_match", "f1")):
            raise ValueError("official scores must be finite values in [0,1]")
        if (any(type(row[key]) is not int or row[key] < 0 for key in ("prompt_tokens", "completion_tokens"))
                or row["tokens"] != row["prompt_tokens"] + row["completion_tokens"]):
            raise ValueError("complete measured prompt+completion usage is required")
    return {
        "tasks": len(rows), "successes": sum(row["exact_match"] for row in rows.values()),
        "exact_match": 100 * sum(row["exact_match"] for row in rows.values()) / len(rows),
        "f1": 100 * math.fsum(row["f1"] for row in rows.values()) / len(rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in rows.values()),
        "total_inference_tokens": sum(row["tokens"] for row in rows.values()),
    }


def _paired(*matrices: Mapping) -> None:
    if not matrices[0] or any(set(matrix) != set(matrices[0]) for matrix in matrices[1:]):
        raise ValueError("trusted gates require complete identical task inventories")


def quality_nonregression(child: Mapping, parent: Mapping) -> bool:
    return all(child[key] + 1e-12 >= parent[key] for key in ("exact_match", "f1"))


def pareto_better(child: Mapping, parent: Mapping) -> bool:
    return (
        quality_nonregression(child, parent)
        and child["total_inference_tokens"] <= parent["total_inference_tokens"]
        and (
            any(child[key] > parent[key] + 1e-12 for key in ("exact_match", "f1"))
            or child["total_inference_tokens"] < parent["total_inference_tokens"]
        )
    )


def skill_gate(parent: Mapping, child: Mapping, original: Mapping, *, changed: bool) -> dict:
    _paired(parent, child, original)
    before, after, anchor = map(summarize, (parent, child, original))
    gates = {
        "actual_harness_change": bool(changed),
        "parent_em_f1_nonregression": quality_nonregression(after, before),
        "complete_token_target": after["total_inference_tokens"] * 10 <= anchor["total_inference_tokens"] * 9,
        "useful_quality_or_cost_change": (
            any(after[key] > before[key] + 1e-12 for key in ("exact_match", "f1"))
            or after["total_inference_tokens"] < before["total_inference_tokens"]
        ),
    }
    return {
        "accepted": all(gates.values()), "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "parent": before, "child": after, "original_anchor": anchor,
        "parent_token_monotonicity_required": False,
        "parent_em_regressions": sorted(task for task in parent
                                       if parent[task]["exact_match"] > child[task]["exact_match"]),
        "parent_f1_regressions": sorted(task for task in parent
                                       if parent[task]["f1"] > child[task]["f1"] + 1e-12),
    }


def decide_term(
    *, branch: str, parent_skill: dict, parent_policy: dict, proposed_policy: dict,
    children: list[dict], matrices: Mapping[str, Mapping], anchors: Mapping[str, Mapping],
) -> dict:
    if branch not in BRANCHES or len(children) != 2 or set(matrices) != set(PARTITIONS[1:3]):
        raise ValueError("each branch requires two children on inner and meta validation")
    gates = {}
    for index, child in enumerate(children):
        name = f"child_{index}"
        gates[name] = {
            split: skill_gate(
                matrices[split]["parent"], matrices[split][name], anchors[split],
                changed=skill_content_hash(child) != skill_content_hash(parent_skill),
            )
            for split in PARTITIONS[1:3]
        }
    selected, policy_changed, failed = None, False, []
    if branch == "fixed_rsi":
        if proposed_policy != parent_policy:
            raise ValueError("fixed-policy branch cannot receive a mutated policy")
        passing = [index for index in range(2) if all(
            gate["accepted"] for gate in gates[f"child_{index}"].values()
        )]
        if passing:
            selected = min(passing, key=lambda index: (
                -gates[f"child_{index}"]["meta_validation"]["child"]["exact_match"],
                -gates[f"child_{index}"]["meta_validation"]["child"]["f1"],
                gates[f"child_{index}"]["meta_validation"]["child"]["total_inference_tokens"],
                index,
            ))
        else:
            failed.append("neither fixed-policy child passes both protected gates")
    else:
        policy_gates = {
            "policy_strategy_changed": proposed_policy["strategy"] != parent_policy["strategy"],
            "new_policy_child_differs_from_control": skill_content_hash(children[1]) != skill_content_hash(children[0]),
            "new_policy_child_passes_inner_and_meta": all(
                gate["accepted"] for gate in gates["child_1"].values()
            ),
        }
        for split in PARTITIONS[1:3]:
            _paired(matrices[split]["child_0"], matrices[split]["child_1"])
            control = summarize(matrices[split]["child_0"])
            candidate = summarize(matrices[split]["child_1"])
            policy_gates[split + "_control_nonregression"] = (
                quality_nonregression(candidate, control)
                and candidate["total_inference_tokens"] <= control["total_inference_tokens"]
            )
        policy_gates["strict_meta_control_improvement"] = pareto_better(
            summarize(matrices["meta_validation"]["child_1"]),
            summarize(matrices["meta_validation"]["child_0"]),
        )
        if all(policy_gates.values()):
            selected, policy_changed = 1, True
        failed.extend(key for key, passed in policy_gates.items() if not passed)
        gates["policy"] = policy_gates
    return {
        "branch": branch, "selected_child": selected,
        "skill_accepted": selected is not None, "policy_accepted": policy_changed,
        "rollback_both": branch == "meta_rsi" and selected is None,
        "failed_gates": failed, "gates": gates,
        "parent_skill_sha256": digest(parent_skill), "parent_policy_sha256": digest(parent_policy),
        "after_skill_sha256": digest(parent_skill if selected is None else children[selected]),
        "after_policy_sha256": digest(proposed_policy if policy_changed else parent_policy),
    }


def primary_release_gates(
    methods: Mapping[str, Mapping], *, skill_changed: bool, policy_changed: bool,
) -> dict:
    if set(methods) != set(FINAL_ARMS):
        raise ValueError("the complete declared final-screen comparison is required")
    candidate = methods["meta_rsi"]
    original, fixed = methods["original_chisel"], methods["fixed_rsi"]
    gates = {
        "actual_skill_and_policy_updates": skill_changed and policy_changed,
        "strict_original_em": candidate["exact_match"] > original["exact_match"] + 1e-12,
        "strict_original_f1": candidate["f1"] > original["f1"] + 1e-12,
        "at_least_ten_percent_complete_token_saving":
            candidate["total_inference_tokens"] * 10 <= original["total_inference_tokens"] * 9,
        "strict_fixed_rsi_em": candidate["exact_match"] > fixed["exact_match"] + 1e-12,
        "strict_fixed_rsi_f1": candidate["f1"] > fixed["f1"] + 1e-12,
        "fixed_rsi_complete_token_nonregression":
            candidate["total_inference_tokens"] <= fixed["total_inference_tokens"],
    }
    return {
        "gates": gates, "joint_development_target_established": all(gates.values()),
        "better_em_and_f1_than_every_compared_arm": skill_changed and policy_changed and all(
            all(candidate[key] > row[key] + 1e-12 for key in ("exact_match", "f1"))
            for name, row in methods.items() if name != "meta_rsi"
        ),
        "paper_admitted": False, "full_official_protocol": False,
        "independent_confirmation": False,
    }
