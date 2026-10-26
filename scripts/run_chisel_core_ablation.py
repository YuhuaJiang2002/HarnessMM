#!/usr/bin/env python3
"""Run three source-frozen CHISEL RSI lineages on the native ALFWorld actor."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_chisel_interactive_shared226 as native  # noqa: E402
from scripts import run_chisel_instruction_only_validation as common  # noqa: E402
from scripts import run_alfworld_ablation_rerun as previous  # noqa: E402

SCHEMA = "chisel-core-rsi-ablation/v1"
METHODS = ("full", "without_sparse_edit", "without_cost_gate")
SELF = "scripts/run_chisel_core_ablation.py"
HOSTS = {"h200-1": "harnessmm-worker-1", "h200-2": common.HOST}
NORMALIZED_COMPILATION = "request-bound-effective-edits/v1"
SCIENTIFIC_KEYS = (
    "methods", "rounds", "sparse_maximum_slots", "dense_maximum_slots",
    "minimum_saving_fraction", "proposal_max_tokens", "proposal_seed",
    "feedback_tasks_per_family", "initial_policy", "episode_seed",
    "partition_counts", "partition_sha256", "membership_sha256", "initial_freeze_sha256",
)


def validate_spec(spec: dict) -> None:
    native.require(spec["schema_version"] == SCHEMA and tuple(spec["methods"]) == METHODS,
                   "Only the three core RSI conditions are authorized")
    native.require(spec["rounds"] == 2 and spec["sparse_maximum_slots"] == 2
                   and spec["dense_maximum_slots"] == 12
                   and spec["minimum_saving_fraction"] == .1
                   and spec["proposal_max_tokens"] == 4096
                   and spec["feedback_tasks_per_family"] == 4,
                   "The matched optimization protocol changed")
    native.require(spec["partition_counts"] == common.COUNTS
                   and type(spec["episode_seed"]) is int and spec["episode_seed"] in (9401, 9402, 9403)
                   and spec["initial_policy"] == "chisel_skills", "Wrong initialization or split")
    native.require(spec["cumulative_token_mode"] == "unlimited" and all(
        spec[key] is None for key in (
            "maximum_total_provider_tokens", "maximum_request_attempts", "maximum_run_seconds")),
        "Cumulative token/request/time termination is disabled")
    native.require(spec["workers"] == 8 and spec["host_alias"] in HOSTS
                   and len(set(spec["gpu_indices"])) == len(set(spec["ports"])) == 3,
                   "Three independent, non-overlapping worker resources are required")
    native.require(spec["design"]["handwritten_controller"] is False,
                   "Every action must remain model generated")
    native.require(spec.get("proposal_compilation", "strict/v1") in (
        "strict/v1", NORMALIZED_COMPILATION), "Unknown proposal compilation contract")


def policy_slots(policy: dict) -> dict[str, str]:
    native.require(set(policy) == {"skills", "harness_sha256"},
                   "Only skill text may enter the core ablation")
    native.verify_signed(policy, "harness_sha256")
    slots = {}
    native.require(len(policy["skills"]) == 6, "All six task families must remain present")
    for family, text in sorted(policy["skills"].items()):
        parts = text.split("\n\n")
        native.require(len(parts) == 2, "The frozen policy must retain its two named paragraphs")
        for index, part in enumerate(parts):
            slots[f"{family}/{index}"] = part
    return slots


def compile_proposal(parent: dict, proposal: dict, feedback: list[dict], maximum: int) -> tuple[dict, dict]:
    slots = policy_slots(parent)
    native.require(type(proposal) is dict and set(proposal) == {
        "parent_sha256", "abstain", "support_ids", "edits", "rationale"},
        "Unexpected proposal schema")
    native.require(proposal["parent_sha256"] == parent["harness_sha256"]
                   and type(proposal["abstain"]) is bool
                   and isinstance(proposal["rationale"], str), "Wrong proposal parent or types")
    edits, support = proposal["edits"], proposal["support_ids"]
    native.require(type(edits) is list and type(support) is list
                   and all(isinstance(key, str) for key in support)
                   and len(support) == len(set(support)), "Malformed edits or support")
    cases = {case["id"]: case for case in feedback}
    native.require(all(case["partition"] == "train" for case in feedback)
                   and set(support) <= cases.keys(), "Only supplied TRAIN evidence may support edits")
    if proposal["abstain"]:
        native.require(not edits, "An abstention cannot contain edits")
        return deepcopy(parent), {"status": "EXPLICIT_ABSTENTION", "changed_slots": 0}
    native.require(1 <= len(edits) <= maximum
                   and len({cases[key]["group"] for key in support}) >= 2,
                   "Edit cardinality or distinct TRAIN support constraint failed")
    changed = {}
    for edit in edits:
        native.require(type(edit) is dict and set(edit) == {"slot", "replacement"},
                       "Only a named slot and its replacement are editable")
        slot, replacement = edit["slot"], edit["replacement"]
        native.require(isinstance(slot, str) and slot in slots and slot not in changed,
                       "Unknown or duplicate edit slot")
        native.require(isinstance(replacement, str) and len(replacement) <= 5000
                       and "\n\n" not in replacement and replacement != slots[slot],
                       "Replacement changes the paragraph schema or is not an edit")
        native.require(not any(key in replacement for key in cases)
                       and not re.search(r"FloorPlan|trial[_-]|json_2\.1\.1/|[a-z]+[_ ]\d+\b",
                                         replacement),
                       "A reusable rule cannot contain benchmark-instance literals")
        changed[slot] = replacement
    revised = {**slots, **changed}
    child = {"skills": {
        family: "\n\n".join(revised[f"{family}/{i}"] for i in range(2))
        for family in sorted(parent["skills"])
    }}
    child["harness_sha256"] = native.digest(child)
    return child, {"status": "COMPILED", "changed_slots": len(changed),
                   "slots": sorted(changed), "parent_sha256": parent["harness_sha256"],
                   "child_sha256": child["harness_sha256"]}


def normalize_proposal(parent: dict, proposal: dict) -> tuple[dict, dict]:
    slots = policy_slots(parent)
    required = {"parent_sha256", "abstain", "support_ids", "edits", "rationale"}
    native.require(type(proposal) is dict and set(proposal) in (required, required - {"parent_sha256"}),
                   "Unexpected proposal schema")
    normalized = deepcopy(proposal)
    missing_parent = "parent_sha256" not in normalized
    if missing_parent:
        normalized["parent_sha256"] = parent["harness_sha256"]
    native.require(normalized["parent_sha256"] == parent["harness_sha256"],
                   "Explicit proposal parent differs from the signed request")
    native.require(type(normalized["abstain"]) is bool and type(normalized["edits"]) is list,
                   "Malformed proposal action fields")
    original = normalized["edits"]
    native.require(len(original) <= len(slots), "Too many named edit entries")
    seen, unchanged, effective = set(), [], []
    for edit in original:
        native.require(type(edit) is dict and set(edit) == {"slot", "replacement"},
                       "Only a named slot and its replacement are editable")
        key, replacement = edit["slot"], edit["replacement"]
        native.require(isinstance(key, str) and key in slots and key not in seen,
                       "Unknown or duplicate edit slot")
        native.require(isinstance(replacement, str), "Replacement must be text")
        seen.add(key)
        if replacement == slots[key]:
            unchanged.append(key)
        else:
            effective.append(edit)
    native.require(not normalized["abstain"] or not original,
                   "An explicit abstention cannot contain edits")
    noop_alias = bool(original and not effective)
    normalized["edits"] = effective
    if noop_alias:
        normalized["abstain"] = True
    return normalized, {
        "contract": NORMALIZED_COMPILATION, "parent_hash_bound_from_request": missing_parent,
        "unchanged_entries_removed": unchanged, "raw_entry_count": len(original),
        "effective_entry_count": len(effective), "noop_alias": noop_alias,
        "changed_replacement_text_rewritten": False,
    }


def compile_for_spec(parent: dict, proposal: dict, feedback: list[dict],
                     maximum: int, spec: dict) -> tuple[dict, dict]:
    if spec.get("proposal_compilation", "strict/v1") == "strict/v1":
        return compile_proposal(parent, proposal, feedback, maximum)
    native.require(spec["proposal_compilation"] == NORMALIZED_COMPILATION,
                   "Unknown prospective compiler contract")
    normalized, normalization = normalize_proposal(parent, proposal)
    child, receipt = compile_proposal(parent, normalized, feedback, maximum)
    if normalization["noop_alias"]:
        receipt["status"] = "NO_EFFECTIVE_EDIT_PARENT_RETAINED"
    receipt["normalization"] = normalization
    return child, receipt


def reused_provider_usage(response: dict, accounting: dict) -> dict:
    usage = {key: response["usage"].get(key) for key in common.USAGE_KEYS}
    native.require(all(type(value) is int and value >= 0 for value in usage.values())
                   and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
                   and accounting["usage"] == usage,
                   "Reused proposal differs from its accounted provider token totals")
    return usage


def selection_gate(parent: dict, child: dict, initial: dict, method: str, *, changed: bool) -> dict:
    native.require(method in METHODS, "Unknown ablation")
    for row in (parent, child, initial):
        native.require(row["tasks"] == 710 and type(row["successes"]) is int
                       and 0 <= row["successes"] <= 710
                       and type(row["total_tokens"]) is int and row["total_tokens"] > 0,
                       "Selection requires complete validation710 native aggregates")
    quality = child["successes"] >= parent["successes"]
    improvement = (child["successes"] > parent["successes"]
                   or child["total_tokens"] < parent["total_tokens"])
    savings = 10 * child["total_tokens"] <= 9 * initial["total_tokens"]
    retained = changed and quality and improvement and (savings or method == "without_cost_gate")
    return {
        "accepted": retained, "changed": changed, "quality_non_decrease": quality,
        "strict_quality_or_cost_improvement": improvement, "fixed_reference_saving_pass": savings,
        "cost_gate_enabled": method != "without_cost_gate",
        "decision_changes_without_cost_gate": bool(changed and quality and improvement and not savings),
    }


def validate_population(rows: list[dict], items: list[dict], *, seed: int = 9401) -> None:
    native.require(type(seed) is int and seed in (9401, 9402, 9403), "Invalid replication seed")
    expected = {item["id"]: item for item in items}
    native.require(len(expected) == len(items) > 0 and len(rows) == len(expected)
                   and {row["id"] for row in rows} == set(expected),
                   "Incomplete, duplicate or out-of-scope actor population")
    for row in rows:
        native.require(row["method"] == "actor" and row["gamefile"] == expected[row["id"]]["gamefile"]
                       and type(row["seed"]) is int and row["seed"] == seed
                       and row["valid"] is True and row["cost_complete"] is True,
                       "Invalid actor identity or incomplete execution")
        native.require(type(row["official_success"]) is int and row["official_success"] in (0, 1)
                       and all(type(row[key]) is int for key in ("turns", "calls", "attempted_calls"))
                       and 0 < row["turns"] <= 50
                       and row["calls"] == row["attempted_calls"] == row["turns"],
                       "Every native action requires one accounted actor call")
        usage = row["usage"]
        native.require(all(type(usage.get(key)) is int and usage[key] >= 0 for key in common.USAGE_KEYS)
                       and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                       and usage["total_tokens"] > 0, "Unknown or inconsistent actor usage")


def feedback_packet(rows: list[dict], items: list[dict], run: Path, policy: dict, spec: dict) -> list[dict]:
    validate_population(rows, items, seed=spec["episode_seed"])
    by_id = {row["id"]: row for row in rows}
    cases = []
    for family in sorted(policy["skills"]):
        members = [item for item in items if item["task_type"] == family]
        selected, used_groups = [], set()
        buckets = {
            success: sorted((item for item in members if by_id[item["id"]]["official_success"] == success),
                            key=lambda item: native.digest([spec["proposal_seed"], item["id"]]))
            for success in (0, 1)
        }
        while len(selected) < spec["feedback_tasks_per_family"]:
            added = False
            for success in (0, 1):
                bucket = buckets[success]
                while bucket and native.scenario(bucket[0]) in used_groups:
                    bucket.pop(0)
                if bucket and len(selected) < spec["feedback_tasks_per_family"]:
                    item = bucket.pop(0)
                    selected.append(item)
                    used_groups.add(native.scenario(item))
                    added = True
            if not added:
                break
        native.require(len(selected) == spec["feedback_tasks_per_family"],
                       "Insufficient distinct TRAIN groups for the frozen feedback packet")
        for item in selected:
            row = by_id[item["id"]]
            episode = Path(row["artifact_dir"])
            trace = native.read(episode / "conversation.json")
            cases.append({
                "id": item["id"], "group": native.scenario(item), "partition": "train",
                "family": family, "success": row["official_success"], "usage": row["usage"],
                "initial_observation": native.read(episode / "initial_observation.json")["observation"][:800],
                "trajectory": [
                    {"action": step["action"], "observation": step["observation"][:192],
                     "error": step["environment_command_error"]}
                    for step in (trace[:2] + trace[-2:] if len(trace) > 4 else trace)
                ],
            })
    native.write(run / "TRAIN_FEEDBACK.json", cases)
    return cases


def propose(run: Path, binding: dict, parent: dict, cases: list[dict],
            spec: dict, method: str, round_index: int, *,
            reuse_sources: tuple[Path, ...] = ()) -> tuple[dict, dict]:
    from scripts import run_alfworld_modular_rsi as core

    maximum = spec["dense_maximum_slots"] if method == "without_sparse_edit" else spec["sparse_maximum_slots"]
    packet = {
        "parent_sha256": parent["harness_sha256"], "slots": policy_slots(parent),
        "maximum_changed_slots": maximum, "training_feedback": cases,
        "output_schema": {"parent_sha256": parent["harness_sha256"], "abstain": False,
                          "support_ids": ["at least two supplied TRAIN IDs from distinct groups"],
                          "edits": [{"slot": "one of the supplied slot keys", "replacement": "reusable text"}],
                          "rationale": "brief evidence-linked explanation"},
    }
    messages = [
        {"role": "system", "content": (
            "Improve the frozen ALFWorld instruction policy using only the supplied TRAIN feedback. "
            "Preserve native task success while reducing complete-task input plus output tokens. "
            "Return only the requested JSON object. Edit reusable rule text in named slots; "
            "retained slots and native tool/action interfaces stay unchanged. "
            "You may replace up to maximum_changed_slots slots, including deleting their text. "
            "Do not introduce new tools, extra model calls, examples, task IDs, object-instance "
            "identifiers or benchmark solutions into the rule text. Preserve successful behavior. "
            "Each replacement is one paragraph with no blank lines. Cite at least two distinct "
            "training groups. If there is no supported edit, set abstain=true and edits=[]. "
            "Validation examples and results are unavailable."
        )},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    reused = None
    for source in reuse_sources:
        folder = source / f"round-{round_index + 1:02d}" / "proposal"
        request_path, response_path = folder / "request.json", folder / "response.json"
        if not request_path.exists() or not response_path.exists():
            continue
        request = native.read(request_path)
        body = request["body"]
        if (request["stage"] != "core_rsi_proposal" or body["messages"] != messages
                or body["seed"] != spec["proposal_seed"] + round_index
                or body["max_tokens"] != spec["proposal_max_tokens"]
                or body["model"] != binding["model"]["id"]
                or body["temperature"] != 0 or body["chat_template_kwargs"] != {"enable_thinking": False}):
            continue
        response = native.read(response_path)
        raw = response["choices"][0]["message"]["content"]
        usage = reused_provider_usage(response, native.read(folder / "accounting.json"))
        reused = {
            "request": str(request_path), "request_sha256": native.file_hash(request_path),
            "response": str(response_path), "response_sha256": native.file_hash(response_path),
            "new_provider_calls": 0, "original_usage_inherited_once": usage,
        }
        native.write(run / "PROPOSAL_REUSE.json", reused)
        break
    if reused is None:
        raw, usage = core.metered_call(
            binding, endpoint_index=0, messages=messages, max_tokens=spec["proposal_max_tokens"],
            seed=spec["proposal_seed"] + round_index, output=run / "proposal",
            timeout=180, stage="core_rsi_proposal")
    try:
        proposal = core.repair.extract_json_object(raw)
        child, receipt = compile_for_spec(parent, proposal, cases, maximum, spec)
    except (ValueError, TypeError, KeyError) as error:
        child, receipt = deepcopy(parent), {
            "status": "INVALID_PROPOSAL_PARENT_RETAINED", "error": str(error),
            "changed_slots": 0, "raw_preserved": True,
        }
    receipt.update(usage=usage, maximum_changed_slots=maximum, method=method,
                   new_provider_calls=0 if reused is not None else 1,
                   reused_proposal=reused,
                   sparse_constraint_exceeded=receipt["changed_slots"] > spec["sparse_maximum_slots"])
    native.write(run / "PROPOSAL_RECEIPT.json", receipt)
    native.write(run / "CANDIDATE.json", child)
    return child, receipt


def evaluate(run: Path, config: dict, protocol: dict, policy: dict, split: str,
             phase: str, items: list[dict]) -> tuple[list[dict], dict]:
    native.require(split in common.COUNTS
                   and {item["id"] for item in items} <= {item["id"] for item in protocol["splits"][split]},
                   "A phase crossed the frozen split boundary")
    previous.verify_assets(config, protocol, items)
    stage = {"stage": phase, "jobs": [{"method": "actor", "item": item} for item in items]}
    if config["episode_seed"] == 9401:
        native.run_stage(run, config, {"policies": {"actor": policy}}, stage)
    else:
        native.run_stage(run, config, {"policies": {"actor": policy}}, stage,
                         episode_seed=config["episode_seed"])
    rows = common.read_rows(run, phase)
    validate_population(rows, items, seed=config["episode_seed"])
    audit_config = {"data_root": config["data_root"], "skillopt_root": config["skillopt_root"],
                    "frozen_source": str(ROOT), "step_limit": 50}
    hashes = {}
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        pending = {
            pool.submit(previous.replay_one, audit_config, row,
                        str(run / f"{phase}-native-audit" / row["id"].replace(":", "_"))): row
            for row in rows
        }
        for future in as_completed(pending):
            row = pending[future]
            receipt = future.result()
            native.require(receipt["status"] == "PASS", "Independent native replay failed")
            name = f"{phase}-native-audit/{row['id'].replace(':', '_')}/RESULT.json"
            hashes[name] = native.file_hash(run / name)
    summary = previous.summarize_exact(rows)["actor"]
    audit = {
        "status": "PASS_NATIVE_REPLAY", "split": split, "phase": phase, "summary": summary,
        "policy_sha256": policy["harness_sha256"], "tasks": len(items),
        "rows_sha256": native.file_hash(run / f"{phase}-rows.jsonl"),
        "native_audit_hashes": hashes, "at_utc": native.now(),
    }
    native.write(run / f"{phase}-AUDIT.json", audit)
    return rows, summary


def cost_audit(run: Path) -> dict:
    from scripts.audit_alfworld_modular_rsi import audit_cost_tree

    record = audit_cost_tree(run, set())
    native.write(run / "COST_AUDIT.json", record)
    return record


def verify_sources(freeze: dict) -> None:
    native.require(all(native.file_hash(ROOT / name) == value for name, value in freeze["source_hashes"].items()),
                   "Frozen source changed")


def verify_inherited_selection(selected: dict, prior: dict, current: dict, method: str) -> None:
    native.verify_signed(selected, "selection_sha256")
    native.verify_signed(prior, "freeze_sha256")
    native.require(selected["status"] == "FROZEN_AFTER_COMPLETE_VALIDATION"
                   and selected["method"] == method and selected["test_opened"] is False
                   and selected["campaign_freeze_sha256"] == prior["freeze_sha256"]
                   and selected["validation"]["tasks"] == 710
                   and len(selected["rounds"]) == current["spec"]["rounds"],
                   "Only a complete pre-test selection can be inherited")
    native.require(prior["initial_policy"] == current["initial_policy"]
                   and all(prior["spec"][key] == current["spec"][key] for key in SCIENTIFIC_KEYS),
                   "Continuation cannot alter the optimizer or population")
    native.require(prior["spec"].get("proposal_compilation", "strict/v1")
                   == current["spec"].get("proposal_compilation", "strict/v1"),
                   "A compiler change requires recomputing selection from preserved development evidence")
    operational = {SELF, "scripts/prepare_chisel_core_compiler_recovery.py"}
    native.require({key: value for key, value in prior["source_hashes"].items() if key not in operational}
                   == {key: value for key, value in current["source_hashes"].items() if key not in operational},
                   "Continuation changed the native actor or its dependencies")
    policy_slots(selected["policy"])


def accounted_predecessors(root: Path) -> tuple[list[str], list[dict]]:
    pending, seen, accounted, empty = [root.resolve()], set(), [], []
    while pending:
        source = pending.pop(0)
        native.require(source not in seen, "Duplicate or cyclic cost predecessor")
        seen.add(source)
        if (source / "usage-ledger.json").exists():
            accounted.append(str(source))
            continue
        native.require((source / "RETIREMENT.json").exists()
                       and (source / "config.json").exists(),
                       "A missing ledger requires a retired, identified pre-start attempt")
        forbidden = ("request.json", "response.json", "accounting.json", "binding.json",
                     "FIRST_EPISODE.json", "GPU_SELECTION.json")
        native.require(not any(any(source.rglob(name)) for name in forbidden)
                       and not list(source.glob("*-PLAN.json")),
                       "Cannot infer zero usage when model or native execution artifacts exist")
        config = native.read(source / "config.json")
        empty.append({
            "source": str(source), "status": "VERIFIED_PRESTART_NO_REQUESTS",
            "own_provider_calls": 0, "own_provider_tokens": 0,
            "config_sha256": native.file_hash(source / "config.json"),
            "retirement_sha256": native.file_hash(source / "RETIREMENT.json"),
            "prior_costs_not_discarded": config.get("prior_run_directories", []),
        })
        pending.extend(Path(value).resolve() for value in config.get("prior_run_directories", []))
    return accounted, empty


def completed_source_lineage(root: Path) -> tuple[Path, ...]:
    pending, seen, result = [root.resolve()], set(), []
    while pending:
        source = pending.pop(0)
        if source in seen:
            continue
        seen.add(source)
        native.require((source / "RETIREMENT.json").exists(), f"Source worker has not retired: {source}")
        native.require(not (source / "test-PLAN.json").exists(),
                       "Development recovery cannot inherit an opened test run")
        result.append(source)
        config = native.read(source / "config.json")
        pending.extend(Path(value).resolve() for value in config.get("prior_run_directories", []))
    return tuple(result)


def validate_development_reuse(old_config: dict, old_freeze: dict,
                               config: dict, freeze: dict) -> None:
    native.verify_signed(old_freeze, "freeze_sha256")
    native.require(old_config["method"] == config["method"]
                   and old_config["episode_budget"] == config["episode_budget"]
                   and old_config["episode_seed"] == config["episode_seed"],
                   "Cannot reuse another method, actor budget or seed")
    native.require(all(old_freeze["spec"][key] == freeze["spec"][key] for key in SCIENTIFIC_KEYS)
                   and old_freeze["initial_policy"] == freeze["initial_policy"],
                   "Development recovery changed the scientific protocol or initial instructions")
    operational = {SELF, "scripts/prepare_chisel_core_compiler_recovery.py"}
    native.require({key: value for key, value in old_freeze["source_hashes"].items() if key not in operational}
                   == {key: value for key, value in freeze["source_hashes"].items() if key not in operational},
                   "Native actor dependencies changed across development recovery")
    for key in ("repository", "revision", "served_name", "max_model_len", "enable_thinking"):
        native.require(old_config["model"][key] == config["model"][key], f"Actor identity changed: {key}")


def validated_phase(run: Path, policy: dict, split: str, rows: list[dict],
                    audit: dict, items: list[dict], seed: int) -> dict:
    validate_population(rows, items, seed=seed)
    summary = previous.summarize_exact(rows)["actor"]
    native.require(audit["status"] == "PASS_NATIVE_REPLAY" and audit["split"] == split
                   and audit["policy_sha256"] == policy["harness_sha256"]
                   and audit["tasks"] == len(items) and audit["summary"] == summary
                   and len(audit["native_audit_hashes"]) == len(rows),
                   "Reused phase differs from its audited native population")
    observed = {}
    for name, expected in audit["native_audit_hashes"].items():
        path = (run / name).resolve()
        native.require(path.is_relative_to(run.resolve()) and native.file_hash(path) == expected,
                       "Native replay receipt changed or escaped its source")
        record = native.read(path)
        native.require(record["status"] == "PASS" and record["id"] not in observed
                       and record["seed"] == seed, "Invalid, duplicate or cross-seed native replay")
        observed[record["id"]] = record["native_success"]
    native.require(observed == {row["id"]: row["official_success"] for row in rows},
                   "Terminal outcomes differ from native replay")
    return summary


def load_reusable_development(run: Path, config: dict, freeze: dict, protocol: dict
                              ) -> tuple[dict, dict, tuple[Path, ...]]:
    sources = completed_source_lineage(Path(config["resume_development_from"]))
    train_cache, validation_cache, receipts = {}, {}, []
    for source in sources:
        old_config = native.read(source / "config.json")
        validate_development_reuse(old_config, native.read(source / "FROZEN.json"), config, freeze)
        native.require(native.read(source / "CHECKPOINT.json") == protocol["checkpoint"],
                       "Reused checkpoint receipt differs from the immutable partition")
        initial_train_parts = {}
        for audit_path in sorted(source.glob("*-AUDIT.json")):
            audit = native.read(audit_path)
            if audit.get("split") not in ("train", "validation"):
                continue
            phase, split = audit["phase"], audit["split"]
            native.require(audit["status"] == "PASS_NATIVE_REPLAY", "Unverified phase cannot be reused")
            plan_path = source / f"{phase}-PLAN.json"
            rows_path = source / f"{phase}-rows.jsonl"
            native.require(native.file_hash(rows_path) == audit["rows_sha256"],
                           "Reused episode rows differ from their native audit")
            jobs = native.read(plan_path)
            native.require(jobs and all(job["arm"] == "actor" and job["phase"] == phase for job in jobs),
                           "Reused native plan has a different execution identity")
            policy = jobs[0]["policy"]
            policy_slots(policy)
            key = policy["harness_sha256"]
            native.require(all(job["policy"] == policy for job in jobs)
                           and key == audit["policy_sha256"], "Reused phase mixes instruction policies")
            items = [job["item"] for job in jobs]
            allowed = {item["id"]: item for item in protocol["splits"][split]}
            native.require(all(item["id"] in allowed and item == allowed[item["id"]] for item in items),
                           "Reused phase crossed split membership or altered native task metadata")
            rows = common.read_rows(source, phase)
            validated_phase(source, policy, split, rows, audit, items, config["episode_seed"])
            receipts.append({
                "source": str(source), "phase": phase, "split": split, "policy_sha256": key,
                "tasks": len(items), "audit_sha256": native.file_hash(audit_path),
                "plan_sha256": native.file_hash(plan_path), "rows_sha256": audit["rows_sha256"],
                "new_actor_calls": 0, "original_usage_inherited_once": audit["summary"]["total_tokens"],
            })
            if split == "train" and phase in ("initial_train_gate", "initial_train"):
                initial_train_parts[phase] = (key, rows)
            if len(rows) == common.COUNTS[split] and {item["id"] for item in items} == set(allowed):
                if split == "train":
                    train_cache.setdefault(key, rows)
                else:
                    validation_cache.setdefault(key, audit["summary"])
        if set(initial_train_parts) == {"initial_train_gate", "initial_train"}:
            first_key, first = initial_train_parts["initial_train_gate"]
            rest_key, rest = initial_train_parts["initial_train"]
            native.require(first_key == rest_key, "The first-task gate and remaining TRAIN changed policy")
            validate_population(first + rest, protocol["splits"]["train"], seed=config["episode_seed"])
            train_cache.setdefault(first_key, first + rest)
    initial = freeze["initial_policy"]["harness_sha256"]
    native.require(initial in train_cache and initial in validation_cache,
                   "Recovery requires complete audited initial TRAIN710 and validation710")
    native.write(run / "REUSED_DEVELOPMENT_PHASES.json", {
        "status": "COMPLETE_NATIVE_PHASES_REUSED", "sources": list(map(str, sources)),
        "initial_train_tasks": len(train_cache[initial]),
        "initial_validation_tasks": validation_cache[initial]["tasks"],
        "receipts": receipts, "new_initial_actor_calls": 0,
        "selection_recomputed_under_new_compiler": True,
    })
    return train_cache, validation_cache, sources


def initialize_service(run: Path, config: dict, freeze: dict, manifest: dict, service_name: str):
    from scripts import run_alfworld_modular_rsi as core

    directory = run / service_name
    directory.mkdir(exist_ok=False)
    held = previous.reserve_device(config, directory)
    process, descriptors, logfile = core.start_service(directory, config, lease_descriptors=held)
    try:
        endpoint = f"http://127.0.0.1:{config['resource']['port']}/v1"
        core.await_service(process, endpoint, config["startup_timeout_seconds"])
        binding = {
            "hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
            "adapter_source_hashes": freeze["source_hashes"],
            "model": {"id": config["model"]["served_name"], "revision": native.REVISION, "endpoints": [endpoint]},
            "gpu": {"visible_devices": "", "service_selection": str(directory / "GPU_SELECTION.json")},
            "skillopt_root": config["skillopt_root"], "data_root": config["data_root"],
            "ledger": str(run / "usage-ledger.json"), "max_model_len": 32768,
        }
        binding["binding_sha256"] = native.digest(binding)
        native.write(run / "binding.json", binding)
        return process, descriptors, logfile, binding
    except BaseException:
        common.stop_owned_service(process, descriptors, logfile)
        raise


def worker(config_path: Path) -> None:
    config = native.read(config_path)
    spec = config["core_ablation"]
    validate_spec(spec)
    native.remote_only(config["authorized_hostname"])
    native.require(socket.gethostname() == config["authorized_hostname"]
                   and common.HOST_GPU_UUIDS[socket.gethostname()][config["resource"]["gpu_index"]]
                   == config["resource"]["gpu_uuid"], "Wrong physical GPU or host")
    native.require(config["episode_budget"] == {**common.BUDGET, "seed": spec["episode_seed"]}
                   and config["episode_seed"] == spec["episode_seed"]
                   and config["cumulative_token_mode"] == "unlimited"
                   and all(config[key] is None for key in (
                       "maximum_total_provider_tokens", "maximum_request_attempts", "maximum_run_seconds")),
                   "Common actor or cumulative accounting contract changed")
    native.require(Path(sys.prefix).resolve() == Path(config["actor_python"]).parent.parent.resolve(),
                   "Use the pinned native actor Python")
    freeze = native.read(Path(config["core_freeze"]))
    native.verify_signed(freeze, "freeze_sha256")
    native.require(freeze["spec"] == spec, "Worker settings differ from the campaign freeze")
    verify_sources(freeze)
    protocol = native.read(Path(config["partition_path"]))
    partition = native.verify_partition(protocol)
    native.require(partition["counts"] == common.COUNTS
                   and partition["partition_sha256"] == spec["partition_sha256"]
                   and partition["membership_sha256"] == spec["membership_sha256"],
                   "The exact 20/20/60 partition changed")
    run = Path(config["run_directory"])
    native.require(run.parent == Path(spec["run_root"]) and config["method"] in METHODS,
                   "Use a new method-specific core-ablation directory")
    run.mkdir(parents=True, exist_ok=False)
    native.write(run / "config.json", config)
    native.write(run / "FROZEN.json", freeze)
    process, descriptors, logfile = None, [], None
    try:
        from scripts import run_alfworld_modular_rsi as core

        native.initialize_worker()
        scratch = common.ipc_scratch(run)
        os.environ.update(CUDA_VISIBLE_DEVICES="", NO_PROXY="127.0.0.1,localhost",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
                          PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
        for directory in config["resource"]["lease_roots"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        native.write(run / "PID.json", {"pid": os.getpid(), "argv": sys.argv,
                                       "hostname": socket.gethostname(), "at_utc": native.now()})
        native.write(run / "PROVENANCE.json", {
            "source_hashes": freeze["source_hashes"], "git_state": config["source_git_state"],
            "config_sha256": native.file_hash(config_path), "seed": spec["episode_seed"],
            "actor_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
            "model_packages": core.repair.capture([config["model"]["python"], "-m", "pip", "freeze"]),
        })
        native.require(shutil.disk_usage(run).free >= spec["minimum_free_storage_bytes"],
                       "Insufficient persistent storage for complete traces")
        for name, value in protocol["checkpoint"]["files"].items():
            native.require(native.file_hash(Path(config["model"]["directory"]) / name) == value,
                           f"Pinned checkpoint changed: {name}")
        adapter = native.read(Path(config["native_adapter_path"]))
        native.require(adapter["files_count"] == len(adapter["files"]) == 182,
                       "Incomplete adapter receipt")
        for name, entry in adapter["files"].items():
            native.require(native.file_hash(Path(config["skillopt_root"]) / name) == entry["sha256"],
                           f"Native adapter changed: {name}")
        native.write(run / "CHECKPOINT.json", protocol["checkpoint"])
        native.write(run / "NATIVE_ADAPTER_SOURCE.json", adapter)
        ledger = common.new_ledger()
        for predecessor in config.get("prior_run_directories", []):
            prior = Path(predecessor)
            while not (prior / "RETIREMENT.json").exists():
                native.write(run / "HEARTBEAT.json", {
                    "status": "WAITING_PRESERVED_PREDECESSOR", "predecessor": predecessor,
                    "at_utc": native.now(), "gpu_leased": False,
                })
                time.sleep(15)
            from scripts.audit_alfworld_modular_rsi import audit_cost_tree

            inherited = audit_cost_tree(prior, set())
            ledger["used_tokens"] += inherited["all_in_total_tokens"]
            ledger["attempts"] += inherited["all_in_calls"]
            ledger["accounted_attempts"] += inherited["all_in_calls"]
            native.write(run / "INHERITED_COST.json", inherited)
        native.write(run / "usage-ledger.json", ledger)
        manifest = {
            "schema_version": core.repair.SCHEMA, "budget": config["episode_budget"],
            "alfworld_environment_mode": "eval_out_of_distribution", "seed": 20260928,
            "task_inventory": protocol["splits"]["train"] + protocol["splits"]["validation"] + protocol["splits"]["test"],
            "protocol_sha256": spec["partition_sha256"], "scope": "core-rsi-ablation",
        }
        manifest["manifest_sha256"] = native.digest(manifest)
        native.write(run / "manifest.json", manifest)
        native.write(run / "PRELAUNCH_RESOURCE_AUDIT.json", native.resource_audit())
        if config.get("resume_selection_from"):
            prior = Path(config["resume_selection_from"])
            selected = native.read(prior / "SELECTED.json")
            verify_inherited_selection(selected, native.read(prior / "FROZEN.json"), freeze, config["method"])
            native.require(not (prior / "test-PLAN.json").exists(), "This continuation must inherit development only")
            parent, rounds, validation = selected["policy"], selected["rounds"], selected["validation"]
            native.write(run / "INHERITED_SELECTION.json", {
                "source": str(prior / "SELECTED.json"), "sha256": native.file_hash(prior / "SELECTED.json"),
                "new_training_or_validation_calls": 0, "rounds_unchanged": True,
            })
        else:
            parent = deepcopy(freeze["initial_policy"])
            train_items = native.ranked_items(protocol, "train")
            validation_items = native.ranked_items(protocol, "validation")
            proposal_sources = ()
            if config.get("resume_development_from"):
                train_cache, validation_cache, proposal_sources = load_reusable_development(
                    run, config, freeze, protocol)
                initial_summary = validation_cache[parent["harness_sha256"]]
            process, descriptors, logfile, binding = initialize_service(run, config, freeze, manifest, "train-service")
            if not config.get("resume_development_from"):
                first, _ = evaluate(run, config, protocol, parent, "train", "initial_train_gate", train_items[:1])
                native.write(run / "FIRST_EPISODE_GATE.json", {
                    "status": "PASS_NATIVE_ACTOR_REPLAY_AND_ACCOUNTING", "row": first[0],
                    "cost": cost_audit(run), "counts_once_in_train710": True, "success_required": False,
                })
                remaining, _ = evaluate(run, config, protocol, parent, "train", "initial_train", train_items[1:])
                train_rows = first + remaining
                validate_population(train_rows, train_items, seed=spec["episode_seed"])
                _, initial_summary = evaluate(run, config, protocol, parent, "validation", "initial_validation", validation_items)
                train_cache = {parent["harness_sha256"]: train_rows}
                validation_cache = {parent["harness_sha256"]: initial_summary}
            rounds = []
            for index in range(spec["rounds"]):
                round_dir = run / f"round-{index + 1:02d}"
                round_dir.mkdir()
                parent_key = parent["harness_sha256"]
                if parent_key not in train_cache:
                    rows, _ = evaluate(run, config, protocol, parent, "train",
                                       f"round_{index + 1}_train", train_items)
                    train_cache[parent_key] = rows
                cases = feedback_packet(train_cache[parent_key], train_items, round_dir, parent, spec)
                child, proposal = propose(round_dir, binding, parent, cases, spec, config["method"], index,
                                          reuse_sources=proposal_sources)
                child_key = child["harness_sha256"]
                if child_key not in validation_cache:
                    _, summary = evaluate(run, config, protocol, child, "validation",
                                          f"round_{index + 1}_validation", validation_items)
                    validation_cache[child_key] = summary
                gate = selection_gate(validation_cache[parent_key], validation_cache[child_key],
                                      initial_summary, config["method"], changed=child_key != parent_key)
                record = {
                    "round": index + 1, "parent_sha256": parent_key, "child_sha256": child_key,
                    "gate": gate, "proposal": proposal, "train_feedback_policy_sha256": parent_key,
                    "validation_case_feedback_to_proposer": False, "cost": cost_audit(run),
                }
                if gate["accepted"]:
                    parent = child
                record["retained_sha256"] = parent["harness_sha256"]
                native.write(round_dir / "SELECTION.json", record)
                rounds.append(record)
            validation = validation_cache[parent["harness_sha256"]]
        verify_sources(freeze)
        selected = {
            "status": "FROZEN_AFTER_COMPLETE_VALIDATION", "method": config["method"],
            "policy": parent, "rounds": rounds, "validation": validation,
            "campaign_freeze_sha256": freeze["freeze_sha256"], "cost": cost_audit(run),
            "at_utc": native.now(), "test_opened": False,
        }
        selected["selection_sha256"] = native.digest(selected)
        native.write(run / "COMPONENT_ACTIVATION.json", {
            "method": config["method"],
            "compiled_proposals": sum(row["proposal"]["status"] == "COMPILED" for row in rounds),
            "changed_slots_by_round": [row["proposal"]["changed_slots"] for row in rounds],
            "accepted_children": sum(row["gate"]["accepted"] for row in rounds),
            "sparse_constraint_exceeded": any(row["proposal"]["sparse_constraint_exceeded"] for row in rounds),
            "cost_gate_decision_changes": sum(row["gate"]["decision_changes_without_cost_gate"] for row in rounds),
            "final_policy_differs_from_initial": parent != freeze["initial_policy"],
            "interpretation": "Report inactive or failed mechanisms explicitly; a retained parent is not an innovation gain.",
        })
        native.write(run / "SELECTED.json", selected)
        common.stop_owned_service(process, descriptors, logfile)
        process, descriptors, logfile = None, [], None
        release_path = run.parent / "TEST_RELEASE.json"
        while not release_path.exists():
            failed = [method for method in METHODS if (run.parent / method / "FAILURE.json").exists()]
            native.require(not failed, f"Test barrier blocked by failed lineages: {failed}")
            native.write(run / "HEARTBEAT.json", {
                "status": "WAITING_ALL_LINEAGES_FROZEN", "at_utc": native.now(),
                "gpu_released": True, "successor_task_id": "test2133",
            })
            time.sleep(15)
        release = native.read(release_path)
        native.verify_signed(release, "release_sha256")
        native.require(release["campaign_freeze_sha256"] == freeze["freeze_sha256"]
                       and release["selections"][config["method"]] == selected["selection_sha256"],
                       "Test release differs from the selected frozen lineage")
        owner = release["policy_owners"][parent["harness_sha256"]]
        if owner != config["method"]:
            owner_result = run.parent / owner / "RESULT.json"
            while not owner_result.exists():
                native.require(not (run.parent / owner / "FAILURE.json").exists(),
                               "The owner of the identical selected policy failed")
                native.write(run / "HEARTBEAT.json", {
                    "status": "WAITING_IDENTICAL_POLICY_OWNER", "owner": owner,
                    "at_utc": native.now(), "gpu_released": True,
                })
                time.sleep(15)
            shared = native.read(owner_result)
            native.require(shared["status"] == "COMPLETE_NATIVE_AUDITED_TEST"
                           and shared["policy_sha256"] == parent["harness_sha256"],
                           "An alias cannot reuse a different or incomplete test")
            test = shared["test"]
            test_reuse = {"owner": owner, "result_sha256": native.file_hash(owner_result),
                          "new_test_tokens": 0, "original_test_acquisition": str(owner_result)}
        else:
            process, descriptors, logfile, binding = initialize_service(run, config, freeze, manifest, "test-service")
            _, test = evaluate(run, config, protocol, parent, "test", "test", native.ranked_items(protocol, "test"))
            test_reuse = None
        native.write(run / "RESULT.json", {
            "status": "COMPLETE_NATIVE_AUDITED_TEST", "method": config["method"],
            "policy_sha256": parent["harness_sha256"], "selection_sha256": selected["selection_sha256"],
            "test": test, "test_reuse": test_reuse, "cost": cost_audit(run),
            "at_utc": native.now(), "new_optimizer_run": True,
            "historical_optimizer_reattributed": False, "paper_admitted": False,
        })
    except BaseException as error:
        native.write(run / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": native.now(),
            "failed_attempts_preserved": True, "unknown_usage_not_zero_imputed": True,
        })
        raise
    finally:
        common.stop_owned_service(process, descriptors, logfile)
        native.write(run / "RETIREMENT.json", {"at_utc": native.now(), "unrelated_processes_stopped": 0})


def prepare(spec_path: Path, destination: Path) -> dict:
    native.remote_only()
    spec = native.read(spec_path)
    validate_spec(spec)
    host = HOSTS[spec["host_alias"]]
    native.require(socket.gethostname() == host, "Prepare on the configured physical host")
    original = Path(spec["predecessor_prepared"])
    archived = native.read(original / "FROZEN.json")
    native.verify_signed(archived, "freeze_sha256")
    native.require(archived["freeze_sha256"] == spec["initial_freeze_sha256"],
                   "Wrong frozen initial policy source")
    initial = archived["policies"][spec["initial_policy"]]
    policy_slots(initial)
    protocol = native.read(original / "partition.json")
    partition = native.verify_partition(protocol)
    native.require(partition["partition_sha256"] == spec["partition_sha256"]
                   and partition["membership_sha256"] == spec["membership_sha256"],
                   "Wrong frozen split")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("partition.json", "native_adapter.json"):
        shutil.copyfile(original / name, destination / name)
    hashes = {str(path.relative_to(ROOT)): native.file_hash(path)
              for path in sorted(ROOT.rglob("*.py")) if "__pycache__" not in path.parts}
    native.require(SELF in hashes and len(hashes) >= 10, "A complete frozen source bundle is required")
    freeze = {
        "schema_version": SCHEMA, "spec": spec, "initial_policy": initial, "partition": partition,
        "source_hashes": hashes, "original_freeze_sha256": archived["freeze_sha256"], "at_utc": native.now(),
    }
    freeze["freeze_sha256"] = native.digest(freeze)
    native.write(destination / "FROZEN.json", freeze)
    base = native.read(original / ("worker-01.json" if spec["host_alias"] == "h200-1" else "worker-00.json"))
    native.require(base["authorized_hostname"] == host, "Wrong host-specific pinned runtime")
    configs = []
    for method, gpu, port in zip(METHODS, spec["gpu_indices"], spec["ports"], strict=True):
        config = deepcopy(base)
        config.update(schema_version=SCHEMA, scope="core_rsi_ablation", method=method,
                      core_ablation=spec, core_freeze=str(destination / "FROZEN.json"),
                      partition_path=str(destination / "partition.json"),
                      native_adapter_path=str(destination / "native_adapter.json"),
                      run_directory=str(Path(spec["run_root"]) / method),
                      workers=spec["workers"], cumulative_token_mode="unlimited",
                      episode_seed=spec["episode_seed"],
                      episode_budget={**common.BUDGET, "seed": spec["episode_seed"]},
                      maximum_total_provider_tokens=None, maximum_request_attempts=None, maximum_run_seconds=None)
        config["resource"].update(host_alias=spec["host_alias"], gpu_index=gpu,
                                  gpu_uuid=common.HOST_GPU_UUIDS[host][gpu], port=port)
        config["source_git_state"] = {"predecessor": base["source_git_state"],
                                     "new_source_hashes": hashes,
                                     "local_benchmark_execution": False}
        if spec.get("predecessor_campaign"):
            prior = str(Path(spec["predecessor_campaign"]) / method)
            if spec.get("resume_development") is True:
                config["prior_run_directories"], config["verified_prestart_attempts"] = accounted_predecessors(Path(prior))
            else:
                config["prior_run_directories"] = [prior]
            if method in spec.get("inherit_selected_methods", []):
                config["resume_selection_from"] = prior
            elif spec.get("resume_development") is True:
                config["resume_development_from"] = prior
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        path = destination / f"{method}.json"
        native.write(path, config)
        configs.append({"method": method, "path": str(path), "sha256": native.file_hash(path)})
    record = {"status": "PREPARED", "freeze_sha256": freeze["freeze_sha256"], "configs": configs,
              "spec_path": str(spec_path), "spec_sha256": native.file_hash(spec_path),
              "source_root": str(ROOT), "run_root": spec["run_root"], "at_utc": native.now()}
    native.write(destination / "PREPARED.json", record)
    return record


def coordinate(prepared: Path) -> None:
    record = native.read(prepared / "PREPARED.json")
    freeze = native.read(prepared / "FROZEN.json")
    spec = freeze["spec"]
    validate_spec(spec)
    native.remote_only()
    root = Path(spec["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes = {}
    for entry in record["configs"]:
        config = native.read(Path(entry["path"]))
        native.require(native.file_hash(Path(entry["path"])) == entry["sha256"], "Worker config changed")
        native.require(not (root / entry["method"]).exists(), "Existing runs cannot be restarted implicitly")
        log = (root / f"{entry['method']}.log").open("xb")
        command = [config["actor_python"], "-B", "-u", str(ROOT / SELF),
                   "worker", "--config", entry["path"]]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        processes[entry["method"]] = process
        native.append(root / "LAUNCHES.jsonl", {
            "method": entry["method"], "pid": process.pid, "command": command,
            "config_sha256": entry["sha256"], "at_utc": native.now(),
        })
    last_capacity = 0.0
    while True:
        state = {}
        for method, process in processes.items():
            run = root / method
            terminal = [run / name for name in ("FAILURE.json", "RESULT.json") if (run / name).exists()]
            progress = [run / name for name in ("HEARTBEAT.json", "PROGRESS.json", "PID.json")
                        if (run / name).exists()]
            selected_path = terminal[0] if terminal else max(progress, key=lambda path: path.stat().st_mtime) if progress else None
            data = native.read(selected_path) if selected_path else {}
            state[method] = {"pid": process.pid, "returncode": process.poll(), "progress": data}
            if process.poll() is not None and not (run / "RESULT.json").exists() and not (run / "FAILURE.json").exists():
                run.mkdir(exist_ok=True)
                native.write(run / "FAILURE.json", {
                    "error": "Worker exited without a result", "returncode": process.returncode,
                    "log": str(root / f"{method}.log"), "at_utc": native.now(),
                })
        if (all((root / method / "SELECTED.json").exists() for method in METHODS)
                and not any((root / method / "FAILURE.json").exists() for method in METHODS)
                and not (root / "TEST_RELEASE.json").exists()):
            selections = {method: native.read(root / method / "SELECTED.json") for method in METHODS}
            for selected in selections.values():
                native.verify_signed(selected, "selection_sha256")
                native.require(selected["status"] == "FROZEN_AFTER_COMPLETE_VALIDATION"
                               and selected["campaign_freeze_sha256"] == freeze["freeze_sha256"]
                               and selected["validation"]["tasks"] == 710, "Invalid pre-test freeze")
            owners = {}
            for method in METHODS:
                owners.setdefault(selections[method]["policy"]["harness_sha256"], method)
            release = {"status": "PASS_ALL_LINEAGES_FROZEN",
                       "campaign_freeze_sha256": freeze["freeze_sha256"],
                       "selections": {key: value["selection_sha256"] for key, value in selections.items()},
                       "policy_owners": owners, "at_utc": native.now()}
            release["release_sha256"] = native.digest(release)
            native.write(root / "TEST_RELEASE.json", release)
        native.write(root / "CAMPAIGN_STATUS.json", {"at_utc": native.now(), "methods": state,
                                                    "successor_task_id": spec["successor_task_id"]})
        if time.monotonic() - last_capacity >= spec["capacity_audit_seconds"]:
            native.append(root / "CAPACITY.jsonl", {
                "host": socket.gethostname(), "resource_audit": native.resource_audit(),
                "storage_free_bytes": shutil.disk_usage(root).free, "at_utc": native.now(),
                "other_node": "The two-node read-only capacity observer runs from the authorized SSH client.",
            })
            last_capacity = time.monotonic()
        if all(process.poll() is not None for process in processes.values()):
            native.write(root / "COMPLETE.json", {
                "status": "COMPLETE" if all((root / method / "RESULT.json").exists() for method in METHODS)
                else "BLOCKED_WITH_FAILURE_EVIDENCE", "methods": state, "at_utc": native.now(),
                "paper_admitted": False,
            })
            return
        time.sleep(spec["monitor_seconds"])


def observe(spec_path: Path, destination: Path) -> None:
    spec = native.read(spec_path)
    validate_spec(spec)
    destination.mkdir(parents=True, exist_ok=False)
    native.write(destination / "PID.json", {"pid": os.getpid(), "argv": sys.argv,
                                          "benchmark_execution": False, "at_utc": native.now()})
    while True:
        for host in ("h200-1", "h200-2"):
            remote = (
                "hostname; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader; "
                "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; "
                "df -h /opt/phiagent/runs; free -m; cat /proc/locks")
            if host == spec["host_alias"]:
                remote += f"; cat {spec['run_root']}/CAMPAIGN_STATUS.json; test ! -f {spec['run_root']}/COMPLETE.json"
                if spec.get("predecessor_campaign"):
                    remote = remote.replace(
                        "; test ! -f",
                        f"; cat {spec['predecessor_campaign']}/CAMPAIGN_STATUS.json; test ! -f")
            try:
                value = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                        host, remote], capture_output=True, text=True, timeout=60)
                status = {"returncode": value.returncode, "stdout": value.stdout, "stderr": value.stderr}
            except subprocess.TimeoutExpired as error:
                status = {"status": "SSH_OBSERVATION_TIMEOUT", "error": str(error)}
            native.append(destination / "OBSERVATIONS.jsonl", {
                "host": host, "at_utc": native.now(), **status})
            if host == spec["host_alias"] and status.get("returncode") == 1 and "CAMPAIGN_STATUS" not in status.get("stderr", ""):
                finished = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                     f"cat {spec['run_root']}/COMPLETE.json"], capture_output=True, text=True, timeout=30)
                if finished.returncode == 0:
                    native.write(destination / "REMOTE_COMPLETE.json", json.loads(finished.stdout))
                    return
        time.sleep(spec["capacity_audit_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    staging = sub.add_parser("prepare")
    staging.add_argument("--spec", type=Path, required=True)
    staging.add_argument("--output", type=Path, required=True)
    execution = sub.add_parser("worker")
    execution.add_argument("--config", type=Path, required=True)
    monitor = sub.add_parser("coordinate")
    monitor.add_argument("--prepared", type=Path, required=True)
    observer = sub.add_parser("observe")
    observer.add_argument("--spec", type=Path, required=True)
    observer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.spec, args.output), indent=2))
    elif args.mode == "worker":
        worker(args.config)
    elif args.mode == "coordinate":
        coordinate(args.prepared)
    else:
        observe(args.spec, args.output)


if __name__ == "__main__":
    main()
