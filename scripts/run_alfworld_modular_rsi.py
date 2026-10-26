#!/usr/bin/env python3
"""Remote-only, frozen-parent ALFWorld modular instruction evolution."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.learning.alfworld_autosaddler_pfn import TASK_FAMILIES, render_task_skill
from phiagent.learning.alfworld_modular_rsi import (
    EDITABLE_SPANS, IDENTITY, module_candidate, paired_gate, replace_history,
    compact_action_contract, compact_boilerplate,
)
from scripts import run_chisel_repair_alfworld as repair


SCHEMA = "phiagent-alfworld-modular-rsi/v1"
PROVIDER_CALL = repair.UNMETERED_PROVIDER_CALL
RECOVERY_ABLATION_ARMS = ("parent", "cot_local", "no_skill", "no_stall_gate", "no_progress_filter",
                          "no_periodic_refresh", "no_typed_selector")
SOURCE_FILES = (
    "phiagent/__init__.py", "phiagent/learning/alfworld_autosaddler_pfn.py",
    "phiagent/learning/alfworld_modular_rsi.py", "scripts/run_alfworld_modular_rsi.py",
    "scripts/run_chisel_repair_alfworld.py", "scripts/run_alfworld_autosaddler_worker.py",
    "scripts/alfworld_public_progress.py", "phiagent/learning/counterfactual_repair.py",
    "phiagent/learning/public_recurrence.py",
    "phiagent/learning/stage_aware_adm.py", "scripts/audit_alfworld_modular_rsi.py",
    "scripts/audit_chisel_repair_alfworld.py",
    "phiagent/learning/alfworld_jev.py", "scripts/alfworld_jev_runtime.py",
)


def hash_file(filename: Path) -> str:
    checksum = hashlib.sha256()
    with filename.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


@contextmanager
def locked_ledger(filename: Path):
    import fcntl

    with filename.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.load(handle)
        try:
            yield state
        finally:
            handle.seek(0)
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)


def ledger_snapshot(filename: Path) -> dict:
    with locked_ledger(filename) as state:
        return deepcopy(state)


def reserve_request(state: dict, amount: int, *, current_time: float) -> None:
    if type(amount) is not int or amount <= 0:
        raise ValueError("Request reservation must be a positive native token count")
    if state["cost_unknown"] or current_time >= state["deadline_unix"]:
        raise RuntimeError("Campaign is blocked by unknown usage or its wall-time limit")
    if state["attempts"] >= state["maximum_attempts"]:
        raise RuntimeError("Campaign request limit reached")
    maximum = state["maximum_tokens"]
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("Token limit must be a positive integer or explicit null (unlimited)")
    if maximum is not None and state["used_tokens"] + state["reserved_tokens"] + amount > maximum:
        raise RuntimeError("Campaign token limit would be exceeded")
    state["reserved_tokens"] += amount
    state["attempts"] += 1


def settle_request(state: dict, amount: int, usage: dict | None) -> None:
    if usage is None:
        state["cost_unknown"] = True
        return
    values = [usage.get(key) for key in repair.USAGE_KEYS]
    if any(type(value) is not int or value < 0 for value in values) or values[0] + values[1] != values[2]:
        state["cost_unknown"] = True
        raise ValueError("Cannot settle inconsistent provider usage")
    state["reserved_tokens"] -= amount
    state["used_tokens"] += usage["total_tokens"]
    state["accounted_attempts"] += 1
    if usage["total_tokens"] > amount:
        state["cost_unknown"] = True
        raise ValueError("Native request token reservation was not an upper bound")


def native_count(binding: dict, *, messages: list | None = None, text: str | None = None) -> int:
    body = {"model": binding["model"]["id"]}
    if messages is not None:
        body.update(messages=messages, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False})
    else:
        body.update(prompt=text, add_special_tokens=False)
    endpoint = binding["model"]["endpoints"][0].removesuffix("/v1") + "/tokenize"
    request = Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        error.close()
        raise
    count = result.get("count")
    if type(count) is not int or count < 0:
        raise ValueError("Service did not return a native token count")
    return count


def metered_call(binding: dict, **kwargs):
    if kwargs["stage"] == "actor":
        episode_root = kwargs["output"].parent.parent
        job = repair.read_json(episode_root / "job.json")
        variant = job["policy"].get("context_variant")
        response_variant = job["policy"].get("response_variant")
        boilerplate_variant = job["policy"].get("boilerplate_variant")
        if variant or response_variant or boilerplate_variant:
            original = deepcopy(kwargs["messages"])
            initial = repair.read_json(episode_root / "initial_observation.json")["observation"]
            history_file = episode_root / "conversation.json"
            history = repair.read_json(history_file) if history_file.exists() else []
            public_history = [{"action": step["action"], "observation": step["observation"]} for step in history]

            def count_prompt(text):
                messages = [original[0], {"role": "user", "content": text}]
                return native_count(binding, messages=messages)

            text = original[1]["content"]
            transformation = None
            if variant:
                text, transformation = replace_history(text, initial, public_history,
                                                        variant=variant, count_tokens=count_prompt)
            response_transformation = None
            if response_variant:
                if response_variant != "brief_action":
                    raise ValueError("Unknown response contract variant")
                text, response_transformation = compact_action_contract(text, count_tokens=count_prompt)
            boilerplate_transformation = None
            if boilerplate_variant:
                if variant or response_variant:
                    raise ValueError("Static boilerplate experiment cannot change other modules")
                text, boilerplate_transformation = compact_boilerplate(
                    text, system=original[0]["content"], variant=boilerplate_variant, count_tokens=count_prompt,
                )
            kwargs["messages"] = [original[0], {"role": "user", "content": text}]
            repair.write_json(episode_root / "context-inputs" / f"{kwargs['output'].name}.json", {
                "original_messages": original, "transformed_messages": kwargs["messages"],
                "transformation": transformation, "public_history_steps": len(public_history),
                "response_transformation": response_transformation,
                "boilerplate_transformation": boilerplate_transformation,
                "evaluator_fields_accessed": False,
            })
    input_tokens = native_count(binding, messages=list(kwargs["messages"]))
    amount = input_tokens + kwargs["max_tokens"]
    if amount > binding["max_model_len"]:
        raise ValueError("Request exceeds the unchanged model context limit")
    ledger = Path(binding["ledger"])
    with locked_ledger(ledger) as state:
        reserve_request(state, amount, current_time=time.time())
    usage = None
    try:
        content, usage = PROVIDER_CALL(binding, **kwargs)
        if usage["prompt_tokens"] != input_tokens:
            raise ValueError("Service tokenization differs from actual charged input tokens")
        return content, usage
    finally:
        accounting = kwargs["output"] / "accounting.json"
        if usage is None and accounting.exists():
            usage = repair.read_json(accounting).get("usage")
        with locked_ledger(ledger) as state:
            settle_request(state, amount, usage)


def initialize_worker() -> None:
    repair.provider_call = metered_call


def development_split(items: list[dict], seed: int, per_family: int) -> tuple[list[dict], list[dict]]:
    split = repair.split_tasks(items, seed)
    train_ids = set(split["discovery"])
    train = [item for item in items if item["id"] in train_ids]
    validation = []
    for family in TASK_FAMILIES:
        available = [item for item in items if item["task_type"] == family and item["id"] not in train_ids]
        ranked = sorted(available, key=lambda item: repair.digest([seed, "validation", item["id"]]))
        if len(ranked) < per_family:
            raise ValueError("Insufficient floorplan-disjoint development tasks")
        validation.extend(ranked[:per_family])
    if {item["group_id"] for item in train} & {item["group_id"] for item in validation}:
        raise ValueError("Training and validation floorplans overlap")
    return train, validation


def trajectory_excerpt(steps: list[dict]) -> list[dict]:
    indices = sorted(set(range(min(3, len(steps)))) | set(range(max(0, len(steps) - 5), len(steps))))
    excerpts = []
    for index in indices:
        action = str(steps[index].get("action") or "")
        feedback = steps[index].get("observation", steps[index].get("env_feedback"))
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("A training step lacks public observation/env_feedback evidence")
        observation = feedback
        excerpts.append({"index": index, "action": action[:256], "observation": observation[:384],
                         "action_truncated": len(action) > 256, "observation_truncated": len(observation) > 384})
    return excerpts


def training_packet(parent: Path, items: list[dict], maximum_cases: int) -> list[dict]:
    indexed = {}
    for arm in ("adm39", "adm49", "auto50", "current"):
        output = repair.read_json(parent / "evaluations/rl_dev" / arm / "output.json")
        indexed[arm] = {row["id"]: row for row in output["rows"]}
    cases = []
    for item in sorted(items, key=lambda row: repair.digest(["contrast", row["id"]])):
        family = item["task_type"]
        incumbent = "auto50" if "clean" in family else "adm49" if any(word in family for word in ("heat", "cool")) else "adm39"
        alternatives = [arm for arm in indexed if arm != incumbent]
        baseline = indexed[incumbent][item["id"]]
        other = next((arm for arm in alternatives if indexed[arm][item["id"]]["hard"] != baseline["hard"]), None)
        if other is None or not baseline.get("agent_ok") or not indexed[other][item["id"]].get("agent_ok"):
            continue
        trajectories = []
        for arm in (incumbent, other):
            filename = parent / "evaluations/rl_dev" / arm / "rollouts" / family / "predictions" / item["id"] / "conversation.json"
            steps = repair.read_json(filename)
            if any(step.get("model_response") == "<think>error</think><action>look</action>" for step in steps):
                trajectories = []
                break
            trajectories.append({
                "arm": arm, "official_success": indexed[arm][item["id"]]["hard"],
                "source_sha256": hash_file(filename), "turns": len(steps),
                "source_path": str(filename), "steps": trajectory_excerpt(steps),
            })
        if trajectories:
            cases.append({"task_id": item["id"], "group_id": item["group_id"], "family": family,
                          "task": baseline.get("task_description"), "parent_arm": incumbent,
                          "contrast_kind": "same_task_different_harness_or_step_cap_not_causal_attribution",
                          "trajectories": trajectories})
        if len(cases) == maximum_cases:
            break
    if len({case["group_id"] for case in cases}) < 2:
        raise ValueError("Not enough distinct training groups with valid contrastive evidence")
    return cases


def compile_policy(profiles: dict, context_variant: str | None = None, response_variant: str | None = None,
                   *, boilerplate_variant: str | None = None) -> dict:
    policy = {"skills": {family: render_task_skill(profiles[family], family)[0] for family in TASK_FAMILIES}}
    if context_variant is not None:
        if context_variant not in ("context_recent", "context_all"):
            raise ValueError("Unknown context policy")
        policy["context_variant"] = context_variant
    if response_variant is not None:
        if response_variant != "brief_action":
            raise ValueError("Unknown response contract")
        policy["response_variant"] = response_variant
    if boilerplate_variant is not None:
        if boilerplate_variant not in ("boilerplate_wrapper", "boilerplate_role", "boilerplate_both"):
            raise ValueError("Unknown boilerplate policy")
        if context_variant or response_variant:
            raise ValueError("Boilerplate policy must preserve all other modules")
        policy["boilerplate_variant"] = boilerplate_variant
    policy["harness_sha256"] = repair.digest(policy)
    return policy


def reusable_cells(run: Path, config: dict, phase: str, items: list[dict], policies: dict) -> dict:
    result = {arm: [] for arm in policies}
    source_name = config.get("preserve_completed_from")
    if source_name is None:
        return result
    source = Path(source_name).resolve()
    if source == run.resolve() or source_name not in config.get("prior_run_directories", []):
        raise ValueError("Recovery source must be a distinct charged prior run")
    original = repair.read_json(source / "config.json")
    fields = ("evaluation_scope", "module_variants", "seed", "episode_seed", "workers",
              "family_step_caps", "episode_budget", "model", "parent_manifest_sha256",
              "parent_request_sha256", "historical_reference", "local_discriminator", "recovery_ablation_study")
    if any(config.get(key) != original.get(key) for key in fields):
        raise ValueError("Recovery changed a scientific input or acceptance gate")
    freeze = repair.read_json(source / "development60-freeze.json")
    if (freeze["sha256"] != repair.digest({key: value for key, value in freeze.items() if key != "sha256"})
            or freeze["policies"] != policies or freeze["task_ids"] != [item["id"] for item in items]):
        raise ValueError("Recovery policies or full task inventory differ from the frozen prior run")
    inventory = {item["id"]: item for item in items}
    receipt = {"phase": phase, "source_run": str(source), "source_config_sha256": hash_file(source / "config.json"),
               "source_manifest_sha256": hash_file(source / "manifest.json"),
               "source_freeze_sha256": hash_file(source / "development60-freeze.json"),
               "selection_rule": "All valid complete-cost cells, including zero-success episodes; no outcome selection",
               "cells": [], "invalid_prior_cells": []}
    seen = set()
    for filename in sorted((source / phase).glob("*/*/result.json")):
        row = repair.read_json(filename)
        if row.get("valid") is not True:
            if row.get("official_success") is not None:
                raise ValueError("Prior infrastructure-invalid episode was incorrectly scored")
            receipt["invalid_prior_cells"].append({"result_path": str(filename), "sha256": hash_file(filename)})
            continue
        arm, task_id = row["method"], row["id"]
        if (arm not in policies or task_id not in inventory or (arm, task_id) in seen
                or row.get("cost_complete") is not True or row["seed"] != config["episode_seed"]
                or Path(row["artifact_dir"]).resolve() != filename.parent):
            raise ValueError("Invalid reusable episode identity or accounting")
        job = repair.read_json(filename.parent / "job.json")
        if (job["policy"] != policies[arm] or job["item"] != inventory[task_id]
                or job["phase"] != phase or job["arm"] != arm):
            raise ValueError("Reusable episode job differs from the frozen method/task")
        seen.add((arm, task_id))
        result[arm].append({**row, "task_id": task_id})
        receipt["cells"].append({"arm": arm, "id": task_id, "artifact_dir": str(filename.parent),
                                  "result_sha256": hash_file(filename), "job_sha256": hash_file(filename.parent / "job.json")})
    repair.write_json(run / "REUSED_CELLS.json", receipt)
    return result


def run_phase(run: Path, config: dict, phase: str, items: list[dict], policies: dict) -> dict:
    result = reusable_cells(run, config, phase, items, policies)
    reused = {(arm, row["id"]) for arm, rows in result.items() for row in rows}
    expected_count = len(items) * len(policies)
    jobs = [{
        "study": str(run), "phase": phase, "arm": arm, "item": item,
        "policy": policy, "endpoint_index": 0,
        "alfworld_modular_rsi": {"max_steps": config["family_step_caps"][item["task_type"]],
                                "episode_seed": config["episode_seed"]},
    } for arm, policy in policies.items() for item in items if (arm, item["id"]) not in reused]
    jobs.sort(key=lambda job: repair.digest([config["seed"], phase, job["arm"], job["item"]["id"]]))
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=initialize_worker) as pool:
        futures = [pool.submit(repair.run_episode, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            row["task_id"] = row["id"]
            result[row["method"]].append(row)
            progress = {"phase": phase, "completed": sum(len(rows) for rows in result.values()),
                        "expected": expected_count, "reused_cells": len(reused), "at_utc": repair.now(),
                        "ledger": ledger_snapshot(run / "usage-ledger.json")}
            repair.write_json(run / "PROGRESS.json", progress)
            print(json.dumps({key: progress[key] for key in ("phase", "completed", "expected")}), flush=True)
            if not row["valid"] or not row["cost_complete"]:
                for pending in futures:
                    pending.cancel()
                repair.write_json(run / f"{phase}-partial.json", result)
                raise RuntimeError(f"Invalid episode {row['id']}: {row.get('failure_message')}; no retry")
    for rows in result.values():
        rows.sort(key=lambda row: row["task_id"])
        if len(rows) != len(items) or {row["id"] for row in rows} != {item["id"] for item in items}:
            raise ValueError("Phase does not cover the complete paired inventory")
        if any(not row["valid"] or not row["cost_complete"] for row in rows):
            raise RuntimeError(f"Infrastructure-invalid or unaccounted episode in {phase}; no retry")
    repair.write_json(run / f"{phase}-rows.json", result)
    return result


def review_packet(module: str, proposal: dict) -> dict:
    return {"locked_module": module, "original_runtime_text": EDITABLE_SPANS[module],
            "replacement_runtime_text": proposal["replacement"]}


def propose(run: Path, config: dict, binding: dict, profiles: dict, cases: list[dict]) -> tuple[dict, dict]:
    supports = {case["task_id"]: case["group_id"] for case in cases}
    candidates = {}
    proposals = {}
    count = lambda text: native_count(binding, text=text)
    proposal_source = config.get("proposal_source_directory")
    if proposal_source:
        previous_cases = repair.read_json(Path(proposal_source) / "training-evidence.json")
        if repair.digest(previous_cases) != repair.digest(cases):
            raise ValueError("Reused proposals require byte-identical structured training evidence")
    for module, span in EDITABLE_SPANS.items():
        packet = {
            "locked_module": module, "only_editable_text": span,
            "maximum_replacement_native_tokens": count(span),
            "parent_shared_instruction": next(components["shared_skill"] for components in profiles.values() if span in components["shared_skill"]),
            "training_evidence": cases,
            "schema": {"module": module, "support_task_ids": ["two or more IDs from different supplied groups"],
                       "replacement": "short general replacement for only_editable_text", "rationale": "evidence-linked explanation"},
        }
        messages = [{"role": "system", "content": (
            "Improve one ALFWorld CHISEL instruction module from training evidence. Return JSON only. "
            "This is a ModularRSI-inspired adaptation, not an official reproduction. "
            "Compare successful and failed SAME-TASK traces, then identify a recurring mechanism across tasks. "
            "Cross-harness contrasts are leads, not causal proof. Change only the named module. "
            "Preserve useful behavior and the XML action interface. Use fewer words than the original span. "
            "Do not add model calls, examples, task-specific IDs, object-instance solutions, or new tools. "
            "Provide at least two supporting task IDs from different groups. If unsupported, return {\"abstain\":true}."
        )}, {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
        if proposal_source:
            source_file = Path(proposal_source) / "proposer" / module / "response.json"
            raw = repair.read_json(source_file)["choices"][0]["message"]["content"]
            repair.write_json(run / "proposer" / module / "reused.json", {
                "source": str(source_file), "sha256": hash_file(source_file), "additional_model_calls": 0,
            })
        else:
            raw, _ = metered_call(binding, endpoint_index=0, messages=messages, max_tokens=2048,
                                  seed=config["seed"], output=run / "proposer" / module,
                                  timeout=config["request_timeout_seconds"], stage="module_proposal")
        try:
            proposal = repair.extract_json_object(raw)
            if proposal.get("abstain"):
                raise ValueError("Proposer abstained")
            if proposal.get("module") != module:
                raise ValueError("Proposer changed the locked module")
            candidate, audit = module_candidate(profiles, proposal, support_groups=supports, count_tokens=count)
            if count(proposal["replacement"]) > count(span):
                raise ValueError("Replacement grows the standalone native-token budget")
            review, _ = metered_call(
                binding, endpoint_index=0, messages=[
                    {"role": "system", "content": (
                        "Review a reusable ALFWorld instruction diff without seeing any validation rewards. "
                        "Accept only if it stays inside the locked module, has a plausible mechanism, preserves "
                        "the unchanged harness interfaces and useful behavior, and contains no instance-specific "
                        "solution or extra model call. Only replacement_runtime_text is executed. "
                        "Review that text, not offline proposal metadata. "
                        "Return JSON: {\"accept\":true or false,\"reason\":\"...\"}."
                    )}, {"role": "user", "content": json.dumps(review_packet(module, proposal))}],
                max_tokens=512, seed=config["seed"], output=run / "review" / module,
                timeout=config["request_timeout_seconds"], stage="reward_blind_diff_review",
            )
            verdict = repair.extract_json_object(review)
            if verdict.get("accept") is not True:
                raise ValueError("Module failed the reward-blind diff review")
            candidates[module] = candidate
            proposals[module] = proposal
            repair.write_json(run / "candidates" / f"{module}.json", {"proposal": proposal, "audit": audit, "review": verdict, "profiles": candidate})
        except (ValueError, TypeError, KeyError) as error:
            repair.write_json(run / "rejected" / f"{module}.json", {"reason": str(error), "raw": raw})
    return candidates, proposals


def context_review_schema(support_ids: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["supported", "support_task_ids", "mechanism", "review_accept", "review_reason"],
        "properties": {
            "supported": {"type": "boolean"},
            "support_task_ids": {"type": "array", "items": {"type": "string", "enum": support_ids}},
            "mechanism": {"type": "string"},
            "review_accept": {"type": "boolean"},
            "review_reason": {"type": "string"},
        },
    }


def context_candidates(run: Path, config: dict, binding: dict, profiles: dict, cases: list[dict]) -> dict:
    import inspect
    from phiagent.learning.alfworld_modular_rsi import json_text, public_memory_lines

    variants = config.get("module_variants") or config["context_variants"]
    functions = (json_text, public_memory_lines, replace_history)
    if config.get("module_variants"):
        functions = (*functions, compact_action_contract)
    source = "\n".join(inspect.getsource(function) for function in functions)
    packet = {"locked_module": "agent_loop_output_and_context_separate_controls" if config.get("module_variants") else "context_mgmt", "training_evidence": cases,
              "hypothesis": ("Long verbal state analysis can dominate cost and distract from admissible action selection; "
                             "test a short state-note/action contract alone and with public memory.") if config.get("module_variants") else "Two-step history loses previously observed target locations and inspected places, causing repeated search.",
              "fixed_candidates": variants, "implementation": source,
              "authorship": "Assistant-authored code candidates; model-assisted training diagnosis, not autonomous code generation.",
              "schema": {"supported": True, "support_task_ids": ["at least two supplied IDs from different floorplans"],
                         "mechanism": "public-feedback evidence", "review_accept": True,
                         "review_reason": "scope, interfaces, and generality"}}
    raw, _ = metered_call(binding, endpoint_index=0, messages=[
        {"role": "system", "content": (
            "Read the training success/failure contrasts and the proposed scoped harness code. "
            "Return JSON only using the schema. Decide whether at least two distinct task groups support "
            "the stated hypothesis. Review whether the code changes only its declared "
            "history or output-contract scope, uses public observations, preserves task/current-observation/actions, "
            "and avoids task-specific solutions and extra online model calls. Supporting IDs are "
            "offline evidence metadata, never executable rules. Do not claim a performance improvement."
        )}, {"role": "user", "content": json.dumps(packet)}], max_tokens=1536, seed=config["seed"],
        output=run / "context-diagnosis", timeout=config["request_timeout_seconds"], stage="context_diagnosis_review",
        response_schema=context_review_schema([case["task_id"] for case in cases]))
    proposal = repair.extract_json_object(raw)
    groups = {case["task_id"]: case["group_id"] for case in cases}
    supports = proposal.get("support_task_ids", [])
    accepted = (proposal.get("supported") is True and proposal.get("review_accept") is True
                and isinstance(supports, list) and all(isinstance(task, str) and task in groups for task in supports)
                and len({groups[task] for task in supports}) >= 2)
    repair.write_json(run / "context-proposal.json", {"packet": packet, "proposal": proposal, "accepted": accepted,
                                                      "code_sha256": hashlib.sha256(source.encode()).hexdigest()})
    if not accepted:
        return {}
    if config.get("module_variants"):
        if variants != ["brief_action", "context_all_brief"]:
            raise ValueError("Only the frozen loop-only and loop/context comparison is supported")
        return {"brief_action": compile_policy(profiles, response_variant="brief_action"),
                "context_all_brief": compile_policy(profiles, "context_all", "brief_action")}
    return {name: compile_policy(profiles, name) for name in variants}


def acquire_service_leases(resource: dict) -> list[int]:
    import fcntl

    descriptors = []
    try:
        for root in resource["lease_roots"]:
            for name in (f"physical-gpu-{resource['gpu_index']}.lock", f"physical-{resource['gpu_uuid']}.lock", f"physical-GPU-{resource['gpu_uuid']}.lock"):
                descriptor = os.open(Path(root) / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptors
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def structured_output_arguments(config: dict) -> list[str]:
    enabled = config.get("compact_json_output", False)
    if type(enabled) is not bool:
        raise ValueError("compact_json_output must be an explicit boolean")
    if not enabled:
        return []
    return ["--structured-outputs-config",
            '{"backend":"xgrammar","disable_any_whitespace":true}']


def model_memory_argument(config: dict) -> str:
    if "model_service_gpu_memory_fraction" not in config:
        return "0.40"
    value = config["model_service_gpu_memory_fraction"]
    if type(value) not in (int, float) or not 0 < value < 1:
        raise ValueError("model_service_gpu_memory_fraction must be a finite fraction in (0,1)")
    return str(value)


def start_service(run: Path, config: dict, *, lease_descriptors: list[int] | None = None):
    """Consume optional inherited GPU leases, including on launch failure."""
    import fcntl

    resource = config["resource"]
    descriptors = [] if lease_descriptors is None else list(lease_descriptors)
    logfile = None
    try:
        if lease_descriptors is None:
            descriptors = acquire_service_leases(resource)
        else:
            expected_paths = [
                Path(root) / name for root in resource["lease_roots"]
                for name in (f"physical-gpu-{resource['gpu_index']}.lock",
                             f"physical-{resource['gpu_uuid']}.lock",
                             f"physical-GPU-{resource['gpu_uuid']}.lock")
            ]
            if len(descriptors) != len(expected_paths) or len(set(descriptors)) != len(descriptors):
                raise ValueError("Inherited GPU lease inventory differs from the configured resource")
            for descriptor, path in zip(descriptors, expected_paths):
                actual, expected = os.fstat(descriptor), path.lstat()
                if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                    raise ValueError("Inherited GPU lease is bound to another resource")
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = repair.capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
        apps = repair.capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
        row = next(line for line in inventory["output"].splitlines() if line.split(",")[0].strip() == str(resource["gpu_index"]))
        if inventory["returncode"] or apps["returncode"] or resource["gpu_uuid"] not in row or resource["gpu_uuid"] in apps["output"]:
            raise RuntimeError("Physical GPU identity, occupancy, or inspection failed")
        if int(row.split(",")[3].strip().split()[0]) > 100:
            raise RuntimeError("Selected GPU has unexpected memory allocations")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", resource["port"]))
        model = config["model"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(resource["gpu_index"]),
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
        environment["PATH"] = str(Path(model["python"]).parent) + os.pathsep + environment.get("PATH", "")
        command = [
            model["python"], "-m", "vllm.entrypoints.openai.api_server", "--model", model["directory"],
            "--served-model-name", model["served_name"], "--host", "127.0.0.1", "--port", str(resource["port"]),
            "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", str(model["max_model_len"]),
            "--gpu-memory-utilization", model_memory_argument(config),
            "--max-num-seqs", str(config["workers"]),
            "--enable-prefix-caching", "--attention-backend", "FLASH_ATTN", "--generation-config", "vllm",
            "--default-chat-template-kwargs", '{"enable_thinking":false}', "--trust-remote-code",
        ]
        command.extend(structured_output_arguments(config))
        if config.get("module_variants") == ["cot_local"]:
            command.extend(["--logprobs-mode", "processed_logprobs", "--max-logprobs", "64"])
        selection = {"at_utc": repair.now(), "hostname": socket.gethostname(), "inventory": inventory,
                     "compute_apps": apps, "selected_index": resource["gpu_index"], "selected_uuid": resource["gpu_uuid"],
                     "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"], "command": command,
                     "run": str(run), "owner_pid": os.getpid(), "released": False}
        repair.write_json(run / "GPU_SELECTION.json", selection)
        for descriptor in descriptors:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, json.dumps(selection).encode())
            os.ftruncate(descriptor, os.lseek(descriptor, 0, os.SEEK_CUR))
        logfile = (run / "service.log").open("x", encoding="utf-8")
        process = subprocess.Popen(command, cwd=run, env=environment, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        repair.write_json(run / "SERVICE.json", {"pid": process.pid, "command": command, "started_at": repair.now()})
        return process, descriptors, logfile
    except BaseException:
        for descriptor in set(descriptors):
            os.close(descriptor)
        if logfile is not None:
            logfile.close()
        raise


def await_service(process, endpoint: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Model service exited: {process.returncode}")
        try:
            repair.get_models(endpoint)
            return
        except HTTPError as error:
            error.close()
        except OSError:
            pass
        signal.sigtimedwait(set(), 2)
    raise TimeoutError("Model service startup limit exceeded")


def context_activation_gate(run: Path, phase: str, variants: list[str]) -> dict:
    counts = {}
    context_counts = {}
    response_counts = {}
    boilerplate_counts = {}
    for variant in variants:
        records = [repair.read_json(filename) for filename in (run / phase / variant).glob("*/context-inputs/*.json")]
        context_counts[variant] = sum((record.get("transformation") or {}).get("changed") is True for record in records)
        response_counts[variant] = sum((record.get("response_transformation") or {}).get("changed") is True for record in records)
        boilerplate_counts[variant] = sum((record.get("boilerplate_transformation") or {}).get("changed") is True for record in records)
        counts[variant] = response_counts[variant] if variant == "brief_action" else context_counts[variant]
        if variant == "context_all_brief":
            counts[variant] = min(context_counts[variant], response_counts[variant])
        if variant.startswith("boilerplate_"):
            counts[variant] = boilerplate_counts[variant]
        if variant in ("cot_jev", "cot_local"):
            decisions = [repair.read_json(filename) for filename in (run / phase / variant).glob("*/decisions/*.json")]
            receipt_path = run / "REUSED_CELLS.json"
            if receipt_path.exists():
                receipt = repair.read_json(receipt_path)
                if receipt["phase"] != phase:
                    raise ValueError("Reuse receipt belongs to a different phase")
                for cell in receipt["cells"]:
                    if cell["arm"] == variant:
                        decisions.extend(repair.read_json(filename) for filename in
                                         (Path(cell["artifact_dir"]) / "decisions").glob("*.json"))
            route = "local" if variant == "cot_local" else "jev"
            counts[variant] = sum(decision["route"] == route for decision in decisions)
    gate = {"phase": phase, "changed_requests": counts, "context_changed_requests": context_counts,
            "response_changed_requests": response_counts, "boilerplate_changed_requests": boilerplate_counts,
            "passed": all(count > 0 for count in counts.values())}
    repair.write_json(run / f"{phase}-activation.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Context intervention did not activate on real inputs; cannot interpret scores")
    return gate


def direct_development(config: dict) -> bool:
    scope = config.get("evaluation_scope", "development18_then_historical60")
    if scope not in ("development18_then_historical60", "opened_historical60_development"):
        raise ValueError("Unknown ALFWorld development scope")
    if scope == "opened_historical60_development":
        if config.get("allow_opened_historical60_development") is not True:
            raise ValueError("Direct development on historical60 must be explicitly authorized")
        return True
    return False


def development_cohorts(config: dict, items: list[dict], historical: list[dict]) -> tuple[list, list, list]:
    if not direct_development(config):
        train, validation = development_split(items, config["seed"], config["validation_per_family"])
        return train, validation, historical
    if (len(historical) != 60 or len({item["id"] for item in historical}) != 60
            or len({item["gamefile"] for item in historical}) != 60):
        raise ValueError("Direct development requires all 60 unique original tasks")
    return deepcopy(historical), deepcopy(historical), []


def direct_development_gate(parent_rows: list, candidate_rows: list, ids: list[str], reference: dict) -> dict:
    gate = paired_gate(parent_rows, candidate_rows, expected_ids=ids,
                       minimum_score=reference["minimum_score_exclusive"])
    mode = reference.get("acceptance_mode")
    if mode is not None:
        if mode != "score_and_mean_total_tokens":
            raise ValueError("Unknown direct60 acceptance mode")
        limit = reference.get("maximum_mean_total_tokens")
        if (type(limit) is not int or limit <= 0 or len(ids) != 60 or reference.get("tasks") != 60
                or reference.get("quality_nondecrease_and_strict_token_saving")
                or "minimum_total_token_saving_fraction" in reference):
            raise ValueError("Absolute token-cap gate requires complete60 and cannot mix legacy saving gates")
        parent = gate["arms"]["parent"]
        candidate = gate["arms"]["candidate"]
        ceiling = limit * len(ids)
        cost_pass = candidate["total_tokens"] <= ceiling
        quality_pass = candidate["successes"] >= parent["successes"]
        return {**gate, "acceptance_mode": mode, "maximum_mean_total_tokens": limit,
                "actual_mean_total_tokens": candidate["total_tokens"] / len(ids),
                "fixed_reference_token_ceiling": ceiling, "fixed_reference_token_ceiling_pass": cost_pass,
                "quality_nondecreasing": quality_pass,
                "passed": gate["target_score_pass"] and quality_pass and cost_pass}
    fixed_cost_pass = gate["arms"]["candidate"]["total_tokens"] <= reference["total_tokens"]
    result = {**gate, "fixed_reference_token_ceiling": reference["total_tokens"],
              "fixed_reference_token_ceiling_pass": fixed_cost_pass,
              "passed": gate["passed"] and fixed_cost_pass}
    if reference.get("quality_nondecrease_and_strict_token_saving") is True:
        parent = gate["arms"]["parent"]
        candidate = gate["arms"]["candidate"]
        quality_pass = candidate["successes"] >= parent["successes"]
        saving_pass = candidate["total_tokens"] < parent["total_tokens"]
        result.update(quality_nondecreasing=quality_pass, strict_token_saving=saving_pass,
                      passed=quality_pass and saving_pass and fixed_cost_pass and gate["target_score_pass"])
    if "minimum_total_token_saving_fraction" in reference:
        from decimal import Decimal

        required = Decimal(str(reference["minimum_total_token_saving_fraction"]))
        if not required.is_finite() or not 0 < required < 1:
            raise ValueError("Minimum total-token saving must be strictly between zero and one")
        parent_tokens = gate["arms"]["parent"]["total_tokens"]
        candidate_tokens = gate["arms"]["candidate"]["total_tokens"]
        substantial = parent_tokens > 0 and Decimal(candidate_tokens) <= Decimal(parent_tokens) * (1 - required)
        result.update(minimum_total_token_saving_fraction=float(required), substantial_token_saving=substantial,
                      passed=result["passed"] and substantial)
    return result


def recovery_ablation_full_arm(study: dict) -> str:
    selector = study.get("base_selector", "model")
    periodic = study.get("base_periodic_refresh", True)
    if selector not in ("model", "deterministic") or type(periodic) is not bool or not periodic and selector != "deterministic":
        raise ValueError("Invalid recovery ablation base configuration")
    full = "no_periodic_refresh" if not periodic else "no_typed_selector" if selector == "deterministic" else "cot_local"
    if study.get("full_arm", full) != full:
        raise ValueError("Declared full arm differs from the recovery ablation base")
    return full


def direct_development_policies(config: dict, profiles: dict) -> dict:
    variants = config.get("module_variants")
    if variants == ["brief_action", "context_all_brief"]:
        return {"parent": compile_policy(profiles),
                "brief_action": compile_policy(profiles, response_variant="brief_action"),
                "context_all_brief": compile_policy(profiles, "context_all", "brief_action")}
    if variants in (["boilerplate_both"], ["boilerplate_wrapper"], ["boilerplate_role"]):
        return {"parent": compile_policy(profiles),
                variants[0]: compile_policy(profiles, boilerplate_variant=variants[0])}
    if variants == ["cot_jev"]:
        from phiagent.learning.alfworld_jev import Gate, JEV_MODEL

        protocol = config["jev"]
        if protocol.get("model") != JEV_MODEL:
            raise ValueError("Jev model must be fixed before candidate execution")
        Gate(**protocol["gate"])
        candidate = compile_policy(profiles)
        candidate.pop("harness_sha256")
        candidate["jev"] = deepcopy(protocol)
        candidate["harness_sha256"] = repair.digest(candidate)
        return {"parent": compile_policy(profiles), "cot_jev": candidate}
    if variants == ["cot_local"]:
        from phiagent.learning.alfworld_jev import LOCAL_BACKEND
        from scripts.alfworld_jev_runtime import readiness

        protocol = config["local_discriminator"]
        if protocol.get("backend") != LOCAL_BACKEND or readiness(protocol)["blockers"]:
            raise ValueError("Invalid local discriminator contract")
        candidate = compile_policy(profiles)
        candidate.pop("harness_sha256")
        candidate["local_discriminator"] = deepcopy(protocol)
        candidate["harness_sha256"] = repair.digest(candidate)
        policies = {"parent": compile_policy(profiles), "cot_local": candidate}
        study = config.get("recovery_ablation_study")
        if study is not None:
            from phiagent.learning.alfworld_jev import PROGRESS_RECOVERY_MODE

            if (protocol.get("context_mode") != PROGRESS_RECOVERY_MODE
                    or protocol.get("recovery_ablation", "none") != "none"
                    or study.get("arms") != list(RECOVERY_ABLATION_ARMS)):
                raise ValueError("Ablation study must preserve the full typed-recovery configuration and declared arms")
            deterministic_base = study.get("base_selector", "model")
            recovery_ablation_full_arm(study)
            periodic_base = study.get("base_periodic_refresh", True)
            for name in RECOVERY_ABLATION_ARMS[2:]:
                ablated = deepcopy(candidate)
                ablated.pop("harness_sha256")
                if name == "no_skill":
                    ablated["skills"] = dict.fromkeys(TASK_FAMILIES, "")
                else:
                    ablated["local_discriminator"]["recovery_ablation"] = name
                if deterministic_base == "deterministic" and name != "no_typed_selector":
                    ablated["local_discriminator"]["deterministic_action_selection"] = True
                if not periodic_base and name in ("no_skill", "no_stall_gate", "no_progress_filter"):
                    ablated["local_discriminator"]["periodic_cot_refresh"] = False
                ablated["harness_sha256"] = repair.digest(ablated)
                policies[name] = ablated
            if not periodic_base:
                selected = deepcopy(candidate)
                selected.pop("harness_sha256")
                selected["local_discriminator"]["recovery_ablation"] = "no_periodic_refresh"
                selected["harness_sha256"] = repair.digest(selected)
                policies["cot_local"] = selected
        return policies
    raise ValueError("Direct60 requires a predeclared supported module comparison")


def recovery_ablation_summary(rows: dict, task_ids: list[str], *, comparison_arm: str = "cot_local") -> dict:
    if (set(rows) != set(RECOVERY_ABLATION_ARMS) or len(task_ids) != 60
            or len(set(task_ids)) != 60 or comparison_arm not in {"cot_local", "no_typed_selector", "no_periodic_refresh"}):
        raise ValueError("Recovery ablation requires all seven arms and the complete60 task inventory")
    methods = {}
    indexed = {}
    for arm in RECOVERY_ABLATION_ARMS:
        indexed[arm] = {row["task_id"]: row for row in rows[arm]}
        if len(rows[arm]) != 60 or set(indexed[arm]) != set(task_ids):
            raise ValueError("Ablation coverage differs between arms")
        for row in rows[arm]:
            usage = row["usage"]
            if (row.get("valid") is not True or row.get("cost_complete") is not True
                    or type(row["official_success"]) is not int or row["official_success"] not in (0, 1)
                    or any(type(usage.get(key)) is not int or usage[key] < 0 for key in repair.USAGE_KEYS)
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ValueError("Ablation contains an invalid outcome or incomplete provider cost")
        totals = {key: sum(row["usage"][key] for row in rows[arm]) for key in repair.USAGE_KEYS}
        successes = sum(row["official_success"] for row in rows[arm])
        methods[arm] = {"tasks": 60, "successes": successes, "score": 100 * successes / 60,
                        **totals, "tokens_per_task": totals["total_tokens"] / 60,
                        "tokens_per_task_k": totals["total_tokens"] / 60000}
    paired = {}
    for arm in RECOVERY_ABLATION_ARMS:
        if arm == comparison_arm:
            continue
        differences = [indexed[comparison_arm][task]["official_success"] - indexed[arm][task]["official_success"]
                       for task in task_ids]
        paired[arm] = {"full_gains": differences.count(1), "full_losses": differences.count(-1),
                       "score_difference_points": 100 * sum(differences) / 60,
                       "total_token_difference": methods[comparison_arm]["total_tokens"] - methods[arm]["total_tokens"]}
    return {"status": "DEVELOPMENT60_ABLATION_COMPLETE", "methods": methods, "paired_full_vs_ablation": paired,
            "arms": list(RECOVERY_ABLATION_ARMS), "tasks_per_arm": 60, "winner": None,
            "selection_uses_development60_outcomes": False, "fresh_confirmation": False,
            "paper_admitted": False, "production_defaults_changed": False,
            "evaluation_scope": "opened_historical60_development", "full_official_benchmark_comparison": False,
            "typesafe_api_used": False, "true_jev_used": False, "calibration_claimed": False}


def recovery_ablation_activation(run: Path) -> dict:
    counts = {}
    receipt_path = run / "REUSED_CELLS.json"
    reused = repair.read_json(receipt_path)["cells"] if receipt_path.exists() else []
    freeze_path = run / "development60-freeze.json"
    policies = repair.read_json(freeze_path)["policies"] if freeze_path.exists() else {}
    for arm in RECOVERY_ABLATION_ARMS[1:]:
        decisions = [repair.read_json(filename) for filename in
                     (run / "development60" / arm).glob("*/decisions/*.json")]
        for cell in reused:
            if cell["arm"] == arm:
                decisions.extend(repair.read_json(filename) for filename in
                                 (Path(cell["artifact_dir"]) / "decisions").glob("*.json"))
        deterministic = policies.get(arm, {}).get("local_discriminator", {}).get("deterministic_action_selection", False)
        route = "deterministic" if arm == "no_typed_selector" or deterministic else "local"
        counts[arm] = sum(record["route"] == route for record in decisions)
        if arm == "no_skill" and any(record["skill_sha256"] != repair.digest("") for record in decisions):
            raise ValueError("No-skill ablation retained task-family skill content")
        if arm == "no_progress_filter":
            for record in decisions:
                if record.get("local_calls"):
                    if record["typed_progress"]["public_constraint"]["allowed_actions"] != record["native_actions"]:
                        raise ValueError("Unfiltered ablation omitted native actions")
    return {"action_counts": counts, "passed": all(count > 0 for count in counts.values())}


def execute_direct_development(run: Path, config: dict, binding: dict, profiles: dict, items: list) -> dict:
    if not direct_development(config):
        raise ValueError("Direct60 execution was not authorized")
    development_cohorts(config, [], items)
    policies = direct_development_policies(config, profiles)
    freeze = {"frozen_at": repair.now(), "policies": policies, "task_ids": [item["id"] for item in items],
              "selection_data_role": "OPENED_DEVELOPMENT60", "development_feedback_permitted": True,
              "fresh_confirmation": False, "prior18_qualification_required": False,
              "candidate_origin": config.get("candidate_origin", "Previously implemented and execution-tested v8 modules; no new proposal call")}
    freeze["sha256"] = repair.digest(freeze)
    if config.get("recovery_ablation_study") is not None:
        study = config["recovery_ablation_study"]
        full_arm = recovery_ablation_full_arm(study)
        bound_arms = ("parent", "no_periodic_refresh", "no_typed_selector") if not study.get("base_periodic_refresh", True) else ("parent", "cot_local")
        reference = Path(study["reference_run_directory"])
        prior = repair.read_json(reference / "development60-freeze.json")
        if (prior["task_ids"] != freeze["task_ids"] or any(prior["policies"][arm] != policies[arm]
                                                          for arm in bound_arms)):
            raise ValueError("Ablation full method, parent or task order differs from Figure1(a)")
        repair.write_json(run / "FIGURE1A_BINDING.json", {"reference_run": str(reference),
                          "reference_freeze_sha256": hash_file(reference / "development60-freeze.json"),
                          "full_arm": full_arm, "reference_policy_arms": list(bound_arms),
                          "full_policy_sha256": policies[full_arm]["harness_sha256"],
                          "parent_policy_sha256": policies["parent"]["harness_sha256"],
                          "new_execution_for_every_arm": True, "historical_scores_pooled": False})
    repair.write_json(run / "development60-freeze.json", freeze)
    metered_call(binding, endpoint_index=0, messages=[{"role": "user", "content": "Reply with OK."}],
                 max_tokens=8, seed=config["seed"], output=run / "admission", timeout=180, stage="synthetic_service_admission")
    if config.get("module_variants") == ["cot_local"]:
        from phiagent.learning.alfworld_jev import LocalGate, local_choice_messages, select_local_decision
        from scripts.alfworld_jev_runtime import EpisodeDriver

        driver = EpisodeDriver(binding=binding, budget=config["episode_budget"], protocol=config["local_discriminator"],
                               output=run / "local-admission", seed=config["seed"])
        messages, choices = local_choice_messages(task="Inspect the visible table.", skill="Use the observed legal action.",
            observation="A table is visible.", actions=["look", "examine table 1"], history=[],
            last_cot_note="Examine the visible table next.")
        response = driver._local({"model": binding["model"]["id"], "messages": messages, "choices": choices},
                                 run / "local-admission")
        admission = select_local_decision(response, choices, expected_model=binding["model"]["id"],
                                          gate=LocalGate(**config["local_discriminator"]["gate"]))
        repair.write_json(run / "LOCAL_ADMISSION.json", {"status": "VALID_SINGLE_TOKEN_DISTRIBUTION",
                          "synthetic_only": True, "true_jev_used": False, "decision": admission})
    rows = run_phase(run, config, "development60", items, policies)
    if config.get("recovery_ablation_study") is not None:
        activation = recovery_ablation_activation(run)
        repair.write_json(run / "RECOVERY_ABLATION_ACTIVATION.json", activation)
        if not activation["passed"]:
            raise RuntimeError("A declared recovery ablation did not activate on real inputs")
        reused = repair.read_json(run / "REUSED_CELLS.json")["cells"] if (run / "REUSED_CELLS.json").exists() else []
        comparison_arm = recovery_ablation_full_arm(config["recovery_ablation_study"])
        return {"identity": config["method_identity"], **recovery_ablation_summary(rows, freeze["task_ids"], comparison_arm=comparison_arm),
            "reused_completed_cells": len(reused)}
    context_activation_gate(run, "development60", config["module_variants"])
    gates = {name: direct_development_gate(rows["parent"], rows[name], freeze["task_ids"], config["historical_reference"])
             for name in config["module_variants"]}
    qualified = [name for name, gate in gates.items() if gate["passed"]]
    winner = min(qualified, key=lambda name: (-gates[name]["arms"]["candidate"]["successes"],
                                            gates[name]["arms"]["candidate"]["total_tokens"], name)) if qualified else None
    result = {"identity": config["method_identity"], "development_gates": gates, "winner": winner,
              "status": "DEVELOPMENT60_TARGET_MET" if winner else "DEVELOPMENT60_TARGET_NOT_MET",
              "evaluation_scope": "opened_historical60_development", "selection_uses_development60_outcomes": True,
              "tasks_per_arm": 60, "arms": list(policies), "fresh_confirmation": False,
              "prior18_qualification_required": False, "official_valid_unseen_calls": 0,
              "paper_admitted": False, "production_defaults_changed": False,
              "full_official_benchmark_comparison": False}
    if config.get("module_variants") == ["cot_local"]:
        result.update(true_jev_used=False, typesafe_api_used=False,
                      discriminator_identity=config["local_discriminator"]["backend"],
                      discriminator_model=config["model"]["served_name"], calibration_claimed=False)
    repair.write_json(run / "development60-gates.json", gates)
    return result


def execute(run: Path, config: dict, binding: dict, profiles: dict, train: list, validation: list, historical: list) -> dict:
    if direct_development(config):
        return execute_direct_development(run, config, binding, profiles, validation)
    cases = training_packet(Path(config["parent_directory"]), train, config["maximum_support_cases"])
    repair.write_json(run / "training-evidence.json", cases)
    metered_call(binding, endpoint_index=0, messages=[{"role": "user", "content": "Reply with OK."}],
                 max_tokens=8, seed=config["seed"], output=run / "admission", timeout=180, stage="synthetic_service_admission")
    base_policy = compile_policy(profiles)
    variants = config.get("module_variants") or config.get("context_variants")
    if variants:
        policies = {"parent": base_policy, **context_candidates(run, config, binding, profiles, cases)}
        candidates = {name: profiles for name in policies if name != "parent"}
        proposals = {}
    else:
        candidates, proposals = propose(run, config, binding, profiles, cases)
        policies = {"parent": base_policy, **{name: compile_policy(candidate) for name, candidate in candidates.items()}}
    if not candidates:
        return {"status": "NO_VALID_MODULE_PROPOSAL", "paper_admitted": False, "production_defaults_changed": False}
    run_phase(run, config, "train_smoke", train[:2], policies)
    if variants:
        context_activation_gate(run, "train_smoke", variants)
    rows = run_phase(run, config, "validation", validation, policies)
    if variants:
        context_activation_gate(run, "validation", variants)
    ids = [item["id"] for item in validation]
    gates = {name: paired_gate(rows["parent"], rows[name], expected_ids=ids) for name in candidates}
    accepted = [name for name, gate in gates.items() if gate["passed"]]
    if len(accepted) > 1 and not variants:
        combined = deepcopy(profiles)
        integration = []
        for name in sorted(accepted):
            combined, audit = module_candidate(combined, proposals[name],
                support_groups={case["task_id"]: case["group_id"] for case in cases},
                count_tokens=lambda text: native_count(binding, text=text))
            integration.append(audit)
        repair.write_json(run / "integration.json", {"modules": accepted, "audits": integration, "profiles": combined})
        combined_rows = run_phase(run, config, "integration_validation", validation, {"integrated": compile_policy(combined)})
        rows["integrated"] = combined_rows["integrated"]
        gates["integrated"] = paired_gate(rows["parent"], rows["integrated"], expected_ids=ids)
        policies["integrated"] = compile_policy(combined)
        if gates["integrated"]["passed"]:
            accepted.append("integrated")
    repair.write_json(run / "validation-gates.json", gates)
    result = {"identity": IDENTITY, "validation_gates": gates, "paper_admitted": False,
              "official_modular_rsi_reproduction": False, "production_defaults_changed": False,
              "full_official_benchmark_comparison": False, "official_valid_unseen_calls": 0}
    if not accepted:
        return {**result, "status": "NO_BUDGET_NEUTRAL_VALIDATION_GAIN"}
    winner = min(accepted, key=lambda name: (-gates[name]["arms"]["candidate"]["successes"],
                                           gates[name]["arms"]["candidate"]["total_tokens"], name))
    release = {"winner": winner, "policy": policies[winner], "selection_uses_historical_outcomes": False,
               "frozen_at": repair.now(), "selection": gates[winner]}
    release["sha256"] = repair.digest(release)
    repair.write_json(run / "candidate-freeze.json", release)
    comparison = run_phase(run, config, "historical60", historical, {"parent": base_policy, "candidate": policies[winner]})
    target = paired_gate(comparison["parent"], comparison["candidate"], expected_ids=[item["id"] for item in historical], minimum_score=66.7)
    historical_cost_pass = target["arms"]["candidate"]["total_tokens"] <= config["historical_reference"]["total_tokens"]
    result.update(winner=winner, historical60=target, historical_token_ceiling_pass=historical_cost_pass,
                  scope="Opened TRAIN-derived historical60; no fresh confirmation or official-test claim",
                  status="HISTORICAL_TARGET_PASSED_OFFICIAL_EVALUATION_PENDING" if target["passed"] and historical_cost_pass else "HISTORICAL_TARGET_NOT_MET")
    return result


def validate_predecode_rejection(call_root: Path) -> dict:
    failure = repair.read_json(call_root / "failure.json")
    request = repair.read_json(call_root / "request.json")
    error = repair.read_json(call_root / "http_error_body.txt")
    expected = 'Grammar error: Unimplemented keys: ["uniqueItems"]'
    schema = request["body"]["response_format"]["json_schema"]["schema"]
    if (failure.get("http_status") != 400 or error.get("error", {}).get("message") != expected
            or schema["properties"]["support_task_ids"].get("uniqueItems") is not True
            or (call_root / "response.json").exists() or (call_root / "accounting.json").exists()):
        raise ValueError("Failure is not the verified pre-decoding schema rejection")
    log_path = call_root.parent / "service.log"
    log = log_path.read_text(encoding="utf-8")
    if expected not in log or "_validate_structured_outputs" not in log or "validate_guidance_grammar" not in log:
        raise ValueError("Server-side pre-decoding rejection evidence is missing")
    return {"call_root": str(call_root), "http_status": 400, "reason": expected,
            "inference_tokens": 0, "accounting_basis": "self_hosted_sampling_parameter_validation_before_decoding",
            "request_sha256": hash_file(call_root / "request.json"),
            "failure_sha256": hash_file(call_root / "failure.json"),
            "http_error_sha256": hash_file(call_root / "http_error_body.txt"),
            "service_log_sha256": hash_file(log_path)}


def run(config_path: Path, *, preflight_only: bool = False) -> int:
    config = repair.read_json(config_path)
    repair.remote_only(config["authorized_hostname"])
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected frozen protocol")
    if preflight_only and config.get("module_variants") not in (["cot_jev"], ["cot_local"]):
        raise ValueError("Only the Jev access check supports preflight-only execution")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    run_dir = Path(config["run_directory"])
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    repair.write_json(run_dir / "config.json", config)
    repair.write_json(run_dir / "PROVENANCE.json", {
        "hostname": socket.gethostname(), "argv": sys.argv, "started_at": repair.now(),
        "source_hashes": {name: hash_file(ROOT / name) for name in SOURCE_FILES},
        "source_provenance": repair.read_json(ROOT / "SOURCE_PROVENANCE.json"),
        "actor_packages": repair.capture([sys.executable, "-m", "pip", "freeze"]),
        "model_packages": repair.capture([config["model"]["python"], "-m", "pip", "freeze"]),
    })
    process = None
    descriptors = []
    logfile = None
    error = None
    try:
        if config.get("module_variants") in (["cot_jev"], ["cot_local"]):
            from scripts.alfworld_jev_runtime import access_preflight

            is_local = config["module_variants"] == ["cot_local"]
            protocol = config["local_discriminator"] if is_local else config["jev"]
            readiness = access_preflight(protocol, config["authorized_hostname"])
            repair.write_json(run_dir / ("LOCAL_PREFLIGHT.json" if is_local else "JEV_PREFLIGHT.json"), readiness)
            execution_authorized = config.get("allow_model_execution") is True
            if readiness["blockers"] or preflight_only or not execution_authorized:
                repair.write_json(run_dir / "RESULT.json", {
                    "status": ("BLOCKED_LOCAL_DISCRIMINATOR" if is_local else "BLOCKED_JEV_ACCESS") if readiness["blockers"] else
                        "PREFLIGHT_ONLY_COMPLETE" if preflight_only else "BLOCKED_MODEL_EXECUTION_AUTHORIZATION",
                    "readiness": readiness, "preflight_only": preflight_only,
                    "model_execution_authorized": execution_authorized,
                    "new_model_calls": 0, "new_provider_tokens": 0, "benchmark_episodes": 0,
                    "gpu_allocated": False, "production_defaults_changed": False,
                })
                print(json.dumps(readiness), flush=True)
                return 0
        parent = Path(config["parent_directory"])
        parent_manifest = repair.read_json(parent / "manifest.json")
        if parent_manifest["manifest_sha256"] != config["parent_manifest_sha256"]:
            raise ValueError("Frozen CHISEL parent identity differs")
        if parent_manifest["manifest_sha256"] != repair.digest({key: value for key, value in parent_manifest.items() if key != "manifest_sha256"}):
            raise ValueError("Parent manifest contents no longer match their hash")
        request = repair.read_json(parent / "evaluations/confirmation/ours_adm/request.json")
        if request["request_sha256"] != config["parent_request_sha256"] or request["request_sha256"] != repair.digest({key: value for key, value in request.items() if key != "request_sha256"}):
            raise ValueError("Frozen CHISEL family policies changed")
        profiles = {family: policy["components"] for family, policy in request["family_policies"].items()}
        for family, policy in request["family_policies"].items():
            runtime = repair.read_json(Path(policy["runtime_manifest_path"]))
            if runtime["execution"]["max_steps"] != config["family_step_caps"][family] or runtime["execution"]["seed"] != config["episode_seed"]:
                raise ValueError("Original family steps or environment seed changed")
        original_service = repair.read_json(Path(config["model"]["historical_service_config"]))
        if original_service["model_revision"] != config["model"]["revision"]:
            raise ValueError("Historical model revision differs")
        model_files = {}
        for name, expected in original_service["model_hashes"].items():
            actual = hash_file(Path(config["model"]["directory"]) / name)
            if actual != expected:
                raise ValueError(f"Pinned model file differs: {name}")
            model_files[name] = actual
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            model_files[name] = hash_file(Path(config["model"]["directory"]) / name)
        repair.write_json(run_dir / "CHECKPOINT.json", {"revision": config["model"]["revision"], "files": model_files})
        items = repair.read_json(parent / "splits/rl_dev/val/items.json")
        historical = repair.read_json(parent / "splits/confirmation/val/items.json")
        for item in (*items, *historical):
            item["group_id"] = repair.floorplan(item["gamefile"])
        if len(historical) != 60 or {item["gamefile"] for item in items} & {item["gamefile"] for item in historical}:
            raise ValueError("Historical cohort identity or path separation differs")
        if direct_development(config) and [item["id"] for item in historical] != request["task_ids"]:
            raise ValueError("Direct60 inventory differs from the original CHISEL task order")
        train, validation, historical = development_cohorts(config, items, historical)
        manifest = {"schema_version": SCHEMA, "budget": config["episode_budget"],
                    "alfworld_environment_mode": "train", "parent_manifest_sha256": config["parent_manifest_sha256"],
                    "splits": {"train": train, "validation": validation, "historical60": historical},
                    "exposure": "OPENED_HISTORICAL60_DEVELOPMENT" if direct_development(config) else "OPENED_TRAIN_DERIVED_DEVELOPMENT",
                    "evaluation_scope": config.get("evaluation_scope", "development18_then_historical60"),
                    "development_feedback_permitted": direct_development(config), "fresh_confirmation": False,
                    "baseline_profiles": profiles,
                    "seed": config["seed"], "config_sha256": repair.digest(config)}
        manifest["manifest_sha256"] = repair.digest(manifest)
        repair.write_json(run_dir / "manifest.json", manifest)
        prior_usage = []
        for prior_directory in config.get("prior_run_directories", []):
            prior = ledger_snapshot(Path(prior_directory) / "usage-ledger.json")
            rejection_roots = [Path(value) for value in config.get("verified_predecode_rejections", [])
                               if Path(value).parent == Path(prior_directory)]
            if rejection_roots:
                settlements = [validate_predecode_rejection(value) for value in rejection_roots]
                if prior["attempts"] - prior["accounted_attempts"] != len(settlements) or not prior["cost_unknown"]:
                    raise ValueError("Pre-decoding settlements do not cover exactly the unaccounted attempts")
                repair.write_json(run_dir / "PRIOR_REJECTION_SETTLEMENT.json", {
                    "original_ledger": prior, "settlements": settlements, "old_artifacts_modified": False,
                })
                prior = {**prior, "cost_unknown": False, "reserved_tokens": 0,
                         "accounted_attempts": prior["attempts"], "predecode_rejections": settlements}
            if prior["cost_unknown"] or prior["reserved_tokens"] or prior["attempts"] != prior["accounted_attempts"]:
                raise ValueError("Prior attempt has unknown or incomplete usage")
            prior_usage.append({"directory": prior_directory, "ledger": prior})
        repair.write_json(run_dir / "usage-ledger.json", {
            "used_tokens": sum(record["ledger"]["used_tokens"] for record in prior_usage),
            "reserved_tokens": 0,
            "attempts": sum(record["ledger"]["attempts"] for record in prior_usage),
            "accounted_attempts": sum(record["ledger"]["accounted_attempts"] for record in prior_usage),
            "prior_attempts": prior_usage,
            "cost_unknown": False, "maximum_tokens": config["maximum_total_provider_tokens"],
            "maximum_attempts": config["maximum_request_attempts"],
            "deadline_unix": min(started + config["maximum_run_seconds"], config.get("absolute_deadline_unix", float("inf"))),
        })
        process, descriptors, logfile = start_service(run_dir, config)
        endpoint = f"http://127.0.0.1:{config['resource']['port']}/v1"
        await_service(process, endpoint, config["startup_timeout_seconds"])
        binding = {
            "hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
            "adapter_source_hashes": {name: hash_file(ROOT / name) for name in SOURCE_FILES},
            "model": {"id": config["model"]["served_name"], "revision": config["model"]["revision"], "endpoints": [endpoint]},
            "gpu": {"visible_devices": "", "service_selection": str(run_dir / "GPU_SELECTION.json")},
            "skillopt_root": config["skillopt_root"], "data_root": config["data_root"],
            "ledger": str(run_dir / "usage-ledger.json"), "max_model_len": config["model"]["max_model_len"],
        }
        if config.get("module_variants") == ["cot_local"]:
            from scripts.alfworld_jev_runtime import bind_local_labels

            binding["local_label_token_ids"] = bind_local_labels(binding, run_dir / "LOCAL_TOKENIZER.json")
            binding["local_discriminator_logprobs_mode"] = "processed_logprobs"
            binding["typesafe_api_used"] = False
        binding["binding_sha256"] = repair.digest(binding)
        repair.write_json(run_dir / "binding.json", binding)
        result = execute(run_dir, config, binding, profiles, train, validation, historical)
        result["all_in_usage"] = ledger_snapshot(run_dir / "usage-ledger.json")
        result["finished_at"] = repair.now()
        repair.write_json(run_dir / "RESULT.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
        repair.write_json(run_dir / "FAILURE.json", {"error": error, "traceback": traceback.format_exc(), "at_utc": repair.now()})
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        retirement = {"released": True, "error": error, "run": str(run_dir), "finished_at": repair.now(),
                      "gpu_allocated": process is not None,
                      "seconds": time.time() - started, "service_exit_code": process.returncode if process else None,
                      "gpu_after": repair.capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])}
        retirement["allocation_cleared"] = (
            process is None or (retirement["gpu_after"]["returncode"] == 0
            and config["resource"]["gpu_uuid"] not in retirement["gpu_after"]["output"])
        )
        retirement["released"] = retirement["allocation_cleared"]
        for descriptor in descriptors:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, json.dumps(retirement).encode())
            os.ftruncate(descriptor, os.lseek(descriptor, 0, os.SEEK_CUR))
            os.close(descriptor)
        if logfile is not None:
            logfile.close()
        repair.write_json(run_dir / "RETIREMENT.json", retirement)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.config.resolve(), preflight_only=arguments.preflight_only))
