"""Controlled keep/drop evidence for one prospective QA module rewrite."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy

from phiagent.learning import qa_meta_rsi as patches
from phiagent.learning import qa_single_module_rsi as rewrite
from phiagent.learning.qa_rule_module_interventions import module_mask_request


SCHEMA = "phiagent-qa-single-module-contrasts/v1"
IDENTITY = "Controlled single-module keep/drop QA contrast acquisition"
MODULES = rewrite.MODULES
ARMS = ("parent", "original_aa", *(f"drop_{module}" for module in MODULES))
CATEGORIES = ("drop_helpful", "drop_harmful", "quality_unchanged", "f1_only", "unstable", "aa_disagreement")


def intervention_profiles(original: dict) -> tuple[dict, dict]:
    if original["version"] != 0 or original["parent_sha256"] is not None:
        raise ValueError("controlled acquisition must start from the exact original skill")
    parent = patches.render_skill(original)
    profiles = {"parent": parent, "original_aa": deepcopy(parent)}
    receipts = {}
    for index, module in enumerate(MODULES):
        mask = "".join("0" if position == index else "1" for position in range(len(MODULES)))
        arm = f"drop_{module}"
        profiles[arm], receipts[arm] = {}, {}
        for profile in ("delete", "modify"):
            complete = module_mask_request(system_prompt=parent[profile], profile=profile, mask="11111")
            if complete.system_prompt != parent[profile]:
                raise ValueError("the full mask must reproduce the original source bytes")
            intervention = module_mask_request(system_prompt=parent[profile], profile=profile, mask=mask)
            profiles[arm][profile] = intervention.system_prompt
            receipts[arm][profile] = intervention.receipt
    return profiles, receipts


def _same_quality(left: Mapping, right: Mapping) -> bool:
    return (
        left["exact_match"] == right["exact_match"]
        and abs(left["f1"] - right["f1"]) <= 1e-12
    )


def _nonregression(left: Mapping, right: Mapping) -> bool:
    return (
        left["exact_match"] >= right["exact_match"]
        and left["f1"] + 1e-12 >= right["f1"]
    )


def _summary(blocks: Mapping) -> dict:
    rows = {f"{repeat}:{task}": row for repeat, current in blocks.items() for task, row in current.items()}
    value = patches.summarize(rows)
    value["execution_cells"] = value.pop("tasks")
    value["successful_executions"] = value.pop("successes")
    value["mean_tokens_per_request"] = value["total_inference_tokens"] / value["execution_cells"]
    return value


def analyze(
    items: Sequence[Mapping], scores: Mapping, native_inputs: Mapping, *,
    repeats: int, minimum_groups: int,
) -> dict:
    if type(minimum_groups) is not int or minimum_groups < 2:
        raise ValueError("at least two independent supporting groups are required")
    if set(scores) != set(ARMS):
        raise ValueError("the complete original, AA and five single-drop arms are required")
    tasks = {item["id"] for item in items}
    if set(native_inputs) != tasks or any(set(row) != set(ARMS) for row in native_inputs.values()):
        raise ValueError("complete prospectively counted native inputs are required")
    for arm, blocks in scores.items():
        rewrite.validate_blocks(items, blocks, repeats)
        for task in tasks:
            hashes = set()
            for rows in blocks.values():
                row = rows[task]
                digest = row["request_sha256"]
                if (
                    not isinstance(digest, str) or len(digest) != 64
                    or set(digest) - set("0123456789abcdef")
                    or row["prompt_tokens"] != native_inputs[task][arm]
                ):
                    raise ValueError("request identity or complete native input differs from its binding")
                hashes.add(digest)
            if len(hashes) != 1:
                raise ValueError("a fixed task/arm must retain its exact request across repeats")
    for task in tasks:
        parent_hash = scores["parent"]["0"][task]["request_sha256"]
        if scores["original_aa"]["0"][task]["request_sha256"] != parent_hash:
            raise ValueError("Original A/A must be a byte-identical parent request")
        if any(scores[arm]["0"][task]["request_sha256"] == parent_hash for arm in ARMS[2:]):
            raise ValueError("a single-drop request cannot be an unchanged parent alias")
    group_tasks = defaultdict(list)
    for item in items:
        group_tasks[item["group"]].append(item["id"])
    summaries = {arm: _summary(blocks) for arm, blocks in scores.items()}
    parent_summary = summaries["parent"]
    modules = {}
    for module in MODULES:
        arm = f"drop_{module}"
        pairs = {}
        for task in sorted(tasks):
            keep = [scores["parent"][str(repeat)][task] for repeat in range(repeats)]
            drop = [scores[arm][str(repeat)][task] for repeat in range(repeats)]
            aa = [scores["original_aa"][str(repeat)][task] for repeat in range(repeats)]
            aa_clean = all(_same_quality(left, right) for left, right in zip(keep, aa))
            helpful = all(
                right["exact_match"] > left["exact_match"] and _nonregression(right, left)
                for left, right in zip(keep, drop)
            )
            harmful = all(
                right["exact_match"] < left["exact_match"] and _nonregression(left, right)
                for left, right in zip(keep, drop)
            )
            unchanged = all(_same_quality(left, right) for left, right in zip(keep, drop))
            em_unchanged = all(left["exact_match"] == right["exact_match"] for left, right in zip(keep, drop))
            category = (
                "aa_disagreement" if not aa_clean else
                "drop_helpful" if helpful else
                "drop_harmful" if harmful else
                "quality_unchanged" if unchanged else
                "f1_only" if em_unchanged else "unstable"
            )
            pairs[task] = {
                "category": category, "aa_quality_agrees_in_every_block": aa_clean,
                "drop_nonregression_in_every_block": all(
                    _nonregression(right, left) for left, right in zip(keep, drop)
                ),
                "paired_em_deltas": [right["exact_match"] - left["exact_match"] for left, right in zip(keep, drop)],
                "paired_f1_deltas": [right["f1"] - left["f1"] for left, right in zip(keep, drop)],
            }
        eligible_groups = []
        for group, ids in sorted(group_tasks.items()):
            if (
                any(pairs[task]["category"] == "drop_helpful" for task in ids)
                and all(
                    pairs[task]["aa_quality_agrees_in_every_block"]
                    and pairs[task]["drop_nonregression_in_every_block"]
                    for task in ids
                )
            ):
                eligible_groups.append(group)
        summary = summaries[arm]
        modules[module] = {
            "arm": arm, "mask": "".join("0" if name == module else "1" for name in MODULES),
            "task_categories": {
                category: sum(value["category"] == category for value in pairs.values())
                for category in CATEGORIES
            },
            "eligible_support_groups": eligible_groups,
            "eligible_support_group_count": len(eligible_groups),
            "protected_function_groups": sorted({
                group for group, ids in group_tasks.items()
                if any(pairs[task]["category"] == "drop_harmful" for task in ids)
            }),
            "quality_deltas_percentage_points": {
                endpoint: summary[endpoint] - parent_summary[endpoint] for endpoint in ("exact_match", "f1")
            },
            "total_token_delta": summary["total_inference_tokens"] - parent_summary["total_inference_tokens"],
            "task_pairs": pairs,
        }
    eligible_modules = [
        module for module in MODULES
        if modules[module]["eligible_support_group_count"] >= minimum_groups
    ]
    selected = min(eligible_modules, key=lambda module: (
        -modules[module]["eligible_support_group_count"],
        len(modules[module]["protected_function_groups"]), MODULES.index(module),
    )) if eligible_modules else None
    return {
        "schema_version": SCHEMA, "method_identity": IDENTITY,
        "status": "REWRITE_EVIDENCE_SUPPORTED" if selected else "STOPPED_NO_REWRITE_SUPPORT",
        "selected_module": selected, "rewrite_evidence_supported": selected is not None,
        "tasks": len(tasks), "independent_groups": len(group_tasks), "repeat_blocks": repeats,
        "task_module_pairs": len(tasks) * len(MODULES),
        "actor_requests": len(tasks) * repeats * len(ARMS),
        "minimum_distinct_support_groups": minimum_groups,
        "all_repeat_blocks_required": True,
        "selection_rule": "Most eligible groups, then fewest protected-function groups, then frozen module order.",
        "modules": modules, "arm_summaries": summaries,
        "per_repeat": {
            repeat: {arm: patches.summarize(scores[arm][repeat]) for arm in ARMS}
            for repeat in map(str, range(repeats))
        },
        "known_total_inference_tokens": sum(value["total_inference_tokens"] for value in summaries.values()),
        "rewrites_performed": 0, "diagnosis_calls": 0, "proposal_calls": 0,
        "controller_and_router_frozen": True, "module_expansion_started": False,
        "combination_search_started": False, "joint_deletions": 0,
        "validated_module_improvement": False, "paper_admitted": False,
        "full_official_benchmark_comparison": False,
        "generalization_or_public_benchmark_improvement_established": False,
        "support_is_a_development_hypothesis_not_a_causal_certificate": True,
        "keep_drop_contrast_does_not_isolate_content_from_input_length": True,
        "next_step": (
            "FREEZE_ONE_MODULE_REWRITE_WITH_SELECTED_DROP_AND_LENGTH_MATCHED_CONTROLS"
            if selected else "RETAIN_ORIGINAL_AND_REVIEW_NEGATIVE_EVIDENCE"
        ),
    }
