#!/usr/bin/env python3
"""Audit saved Modular RSI calls and episodes without rerunning the benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.learning.alfworld_modular_rsi import (
    EDITABLE_SPANS, compact_action_contract, compact_boilerplate, paired_gate, replace_history,
)
from scripts.audit_chisel_repair_alfworld import validate_feedback_signals, validate_row
from scripts import run_chisel_repair_alfworld as repair
from scripts.run_alfworld_modular_rsi import (
    compile_policy, direct_development, direct_development_gate, direct_development_policies,
    hash_file, validate_predecode_rejection, recovery_ablation_summary, recovery_ablation_activation, recovery_ablation_full_arm,
    RECOVERY_ABLATION_ARMS,
)
from phiagent.learning.alfworld_jev import LOCAL_BACKEND, RuleMemory, local_choice_messages, provider_usage as jev_usage
from scripts.alfworld_jev_runtime import (
    ENDPOINT as JEV_ENDPOINT, InterleavedActor, episode_ledger, require_local_binding,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def native_chat_tokens(tokenizer, messages: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False, return_dict=False)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        require(len(encoded) == 1, "Unexpected tokenizer batch size")
        encoded = encoded[0]
    require(isinstance(encoded, list) and all(type(token) is int for token in encoded), "Unexpected native token output")
    return len(encoded)


def validate_call(request: dict, response: dict, accounting: dict, expected_model: str) -> dict:
    usage = response.get("usage", {})
    values = [usage.get(key) for key in repair.USAGE_KEYS]
    require(all(type(value) is int and value >= 0 for value in values), "Nonintegral or missing provider usage")
    require(values[0] + values[1] == values[2], "Provider usage arithmetic differs")
    actual = {key: usage[key] for key in repair.USAGE_KEYS}
    require(accounting["usage"] == actual, "Accounting differs from raw provider response")
    require(request["body"]["model"] == response["model"] == expected_model, "Actor model identity differs")
    require(request["body"]["temperature"] == 0, "Temperature changed")
    require(request["body"]["chat_template_kwargs"] == {"enable_thinking": False}, "Thinking mode changed")
    require(actual["completion_tokens"] <= request["body"]["max_tokens"], "Actual completion exceeds its cap")
    return actual


def validate_terminal(row: dict, conversation: list[dict], step_cap: int) -> None:
    require(not validate_row(row, {"max_steps": step_cap}), "Episode validity, turns, usage, or budget failed")
    require(row["valid"] is True, "Infrastructure-invalid episode")
    require(len(conversation) == row["turns"], "Conversation length differs from episode turns")
    require([step["turn"] for step in conversation] == list(range(len(conversation))), "Conversation turn order differs")
    require(not any(step["done"] for step in conversation[:-1]), "Episode continues after environment termination")
    terminal = conversation[-1]
    require(type(terminal["official_won"]) is bool and type(terminal["done"]) is bool, "Terminal outcome is not native boolean feedback")
    require(row["official_success"] == int(terminal["official_won"] and terminal["done"]), "Episode success differs from terminal environment signal")
    require(terminal["done"] or row["turns"] == step_cap, "Episode ended early without environment termination")
    for step in conversation:
        require(not validate_feedback_signals(step), "Public feedback error signal differs")


def audit_cost_tree(run: Path, visited: set[Path], verified_rejections: set[str] | None = None) -> dict:
    run = run.resolve()
    require(run not in visited, "A prior run is charged more than once")
    visited.add(run)
    config = repair.read_json(run / "config.json")
    ledger = repair.read_json(run / "usage-ledger.json")
    verified_rejections = {
        str(Path(value).resolve())
        for value in set(verified_rejections or ()) | set(config.get("verified_predecode_rejections", []))
    }
    response_files = sorted(run.rglob("response.json"))
    request_files = sorted(run.rglob("request.json"))
    missing = [filename.parent for filename in request_files if not (filename.parent / "response.json").exists()]
    require(all(str(root) in verified_rejections for root in missing), "A request lacks a complete raw response or verified predecode rejection")
    rejections = [validate_predecode_rejection(root) for root in missing]
    own_usage = dict.fromkeys(repair.USAGE_KEYS, 0)
    for response_file in response_files:
        root = response_file.parent
        request = repair.read_json(root / "request.json")
        response = repair.read_json(response_file)
        accounting = repair.read_json(root / "accounting.json")
        if config.get("local_discriminator", {}).get("backend") == LOCAL_BACKEND:
            require(request["stage"] != "jev", "Local experiment must never call TypeSafe")
            require_local_binding({"model": {"id": config["model"]["served_name"],
                "revision": config["model"]["revision"], "endpoints": [request["endpoint"]]}})
        if request["stage"] == "jev":
            usage = jev_usage(response)
            require(request["endpoint"] == JEV_ENDPOINT and request["body"]["model"] == response["model"] == config["jev"]["model"],
                    "Jev provider/model changed")
            require(accounting["usage"] == usage, "Jev accounting differs from raw provider usage")
        else:
            usage = validate_call(request, response, accounting, config["model"]["served_name"])
            if request["stage"] == "local":
                require(config.get("local_discriminator", {}).get("backend") == LOCAL_BACKEND,
                        "Local logits call lacks declared local discriminator identity")
                require_local_binding({"model": {"id": config["model"]["served_name"],
                    "revision": config["model"]["revision"], "endpoints": [request["endpoint"]]}})
        for key in own_usage:
            own_usage[key] += usage[key]
    previous = [audit_cost_tree(Path(directory), visited, verified_rejections) for directory in config.get("prior_run_directories", [])]
    total_calls = len(request_files) + sum(record["all_in_calls"] for record in previous)
    total_tokens = own_usage["total_tokens"] + sum(record["all_in_total_tokens"] for record in previous)
    settled_count = len(rejections) if ledger["cost_unknown"] else 0
    require((ledger["cost_unknown"] is False and ledger["reserved_tokens"] == 0)
            or (settled_count > 0 and ledger["attempts"] - ledger["accounted_attempts"] == settled_count), "Unsettled or unknown usage")
    require(ledger["attempts"] == ledger["accounted_attempts"] + settled_count == total_calls, "Ledger calls differ from raw calls")
    require(ledger["used_tokens"] == total_tokens, "Ledger token total differs")
    require(ledger["maximum_tokens"] is None or total_tokens <= ledger["maximum_tokens"], "Ledger token ceiling differs")
    return {"run": str(run), "own_calls": len(response_files), "own_usage": own_usage,
            "own_predecode_rejections": rejections,
            "all_in_calls": total_calls, "all_in_total_tokens": total_tokens, "prior_runs": previous}


def validate_development60_freeze(config: dict, manifest: dict, freeze: dict, policies: dict) -> None:
    require(direct_development(config), "Direct60 development was not authorized")
    require(manifest["exposure"] == "OPENED_HISTORICAL60_DEVELOPMENT", "Wrong development60 exposure label")
    require(manifest["splits"]["historical60"] == [], "Direct development cannot retain the same60 as a holdout")
    items = manifest["splits"]["validation"]
    require(len(items) == len({item["id"] for item in items}) == len({item["gamefile"] for item in items}) == 60,
        "Direct development inventory is not complete60")
    require(manifest["splits"]["train"] == items, "Declared development-feedback pool differs from development60")
    require(freeze["sha256"] == repair.digest({key: value for key, value in freeze.items() if key != "sha256"}), "Development candidate freeze hash differs")
    require(freeze["task_ids"] == [item["id"] for item in items] and freeze["policies"] == policies, "Frozen tasks or methods differ")
    require(freeze["fresh_confirmation"] is False and freeze["development_feedback_permitted"] is True
        and freeze["prior18_qualification_required"] is False, "Direct60 data roles or qualification gate differ")


def audit_hybrid_episode(root: Path, row: dict, policy: dict, config: dict, tokenizer) -> dict:
    conversation = repair.read_json(root / "conversation.json")
    validate_terminal({**row, "calls": row["turns"]}, conversation, config["family_step_caps"][row["task_type"]])
    actual_ledger = episode_ledger(root)
    for key in ("usage", "calls", "attempted_calls", "accounted_calls", "cost_complete",
                "provider_usage", "router_model_calls", "actor_cot_calls"):
        require(actual_ledger[key] == row[key], f"Hybrid episode ledger differs: {key}")
    require(row["cost_complete"], "Unknown hybrid provider usage")
    used = set()

    def cot(prompt, output):
        used.add(output)
        request = repair.read_json(output / "request.json")
        response = repair.read_json(output / "response.json")
        accounting = repair.read_json(output / "accounting.json")
        body = request["body"]
        require(request["stage"] == "cot", "Unexpected slow-path provider stage")
        require(body["messages"] == [{"role": "system", "content": repair.SYSTEM}, {"role": "user", "content": prompt}],
                "Hybrid CoT prompt was modified")
        require(body["seed"] == row["seed"] and body["max_tokens"] == config["episode_budget"]["max_completion_tokens"],
                "Hybrid CoT seed or output cap differs")
        usage = validate_call(request, response, accounting, config["model"]["served_name"])
        require(native_chat_tokens(tokenizer, body["messages"]) == usage["prompt_tokens"], "CoT input count differs")
        return str(response["choices"][0]["message"].get("content") or "").strip(), usage

    def jev(payload, output):
        used.add(output)
        request = repair.read_json(output / "request.json")
        response = repair.read_json(output / "response.json")
        accounting = repair.read_json(output / "accounting.json")
        require(request["body"] == payload and request["endpoint"] == JEV_ENDPOINT and request["stage"] == "jev",
                "Jev public-state request differs from replay")
        require(jev_usage(response) == accounting["usage"], "Jev usage differs")
        return response

    def local(payload, output):
        used.add(output)
        request = repair.read_json(output / "request.json")
        response = repair.read_json(output / "response.json")
        body = request["body"]
        require(request["stage"] == "local" and body["messages"] == payload["messages"], "Local decision messages differ")
        token_ids = []
        for label in payload["choices"]:
            encoded = tokenizer.encode(label, add_special_tokens=False)
            require(len(encoded) == 1, "Local label is not a single native token")
            token_ids.append(encoded[0])
        require(body["allowed_token_ids"] == token_ids and body["top_logprobs"] == len(token_ids)
                and body["logprobs"] is True and body["max_tokens"] == 1 and body["seed"] == row["seed"],
                "Local decision output contract changed")
        require(repair.read_json(output / "choice-map.json") == {
            "choices": payload["choices"], "token_ids": token_ids, "backend": LOCAL_BACKEND,
            "true_jev_used": False, "calibration_claimed": False}, "Local native label mapping changed")
        usage = validate_call(request, response, repair.read_json(output / "accounting.json"), config["model"]["served_name"])
        require(native_chat_tokens(tokenizer, body["messages"]) == usage["prompt_tokens"], "Local decision input count differs")
        if actor.memory_mode:
            original_messages, original_choices = local_choice_messages(task=row["task_description"], skill=skill,
                observation=observation, actions=commands, history=histories, last_cot_note=actor.last_cot_note)
            require(original_choices == payload["choices"], "Rule memory narrowed the legal action inventory")
            original_tokens = native_chat_tokens(tokenizer, original_messages)
            memory_checks["full_rules_fixed_trace_input_tokens"] += original_tokens
            memory_checks["retrieved_input_tokens"] += usage["prompt_tokens"]
            memory_checks["input_growth_calls"] += int(usage["prompt_tokens"] > original_tokens)
        require_local_binding({"model": {"id": config["model"]["served_name"],
            "revision": config["model"]["revision"], "endpoints": [request["endpoint"]]}})
        return response

    protocol = policy.get("local_discriminator", policy.get("jev"))
    is_local = protocol.get("backend") == LOCAL_BACKEND
    actor = InterleavedActor(protocol, cot=cot, jev=local if is_local else jev)
    histories = []
    routes = {"cot_steps": 0, "jev_steps": 0, "jev_then_cot_steps": 0,
              "local_steps": 0, "local_then_cot_steps": 0, "deterministic_steps": 0}
    initial = repair.read_json(root / "initial_observation.json")["observation"]
    skill = policy["skills"][row["task_type"]]
    memory_checks = None
    progress_checks = None
    if actor.memory_mode:
        require(repair.read_json(root / "RULE_MEMORY.json") == RuleMemory(skill).record(),
                "Stored source rule memory differs from the unchanged CHISEL skill")
        memory_checks = {"prefilter_bypasses": 0, "retrieval_calls": 0,
                         "retrieved_entries": 0, "code_rejected_choices": 0,
                         "full_rules_fixed_trace_input_tokens": 0, "retrieved_input_tokens": 0,
                         "input_growth_calls": 0}
    if actor.progress_mode:
        progress_checks = {"recovery_steps": 0, "typed_calls": 0, "typed_actions": 0,
                           "native_options": 0, "supported_options": 0}
    for turn, step in enumerate(conversation):
        decision = repair.read_json(root / "decisions" / f"{turn:03d}.json")
        calls = root / "providers" / f"{turn:03d}"
        prompt = decision["native_prompt"]
        require(repair.digest(prompt) == decision["native_prompt_sha256"], "Native prompt binding differs")
        observation = initial if turn == 0 else conversation[turn - 1]["observation"]
        require(decision["public_observation"] == observation and decision["skill_sha256"] == repair.digest(skill),
                "Jev observation or original skill differs from public evidence")
        commands = decision["native_actions"]
        block = "\n ".join(f"'{command}'" for command in commands)
        marker = "Your admissible actions of the current situation are: ["
        require(marker in prompt and prompt.rsplit(marker, 1)[1].split("].", 1)[0] == block,
                "Jev option set differs from public native actions")
        replay = actor.step(turn=turn, task=row["task_description"], prompt=prompt, actions=commands,
                            skill=skill, observation=observation,
                            history=histories, output=calls)
        require(replay["text"] == step["model_response"] and replay["usage"] == step["usage"], "Hybrid step replay differs")
        require(replay["decision"] == step["hybrid_decision"], "CoT/Jev route replay differs")
        expected_decision = {**replay["decision"], "usage": replay["usage"], "native_actions": commands,
                             "native_prompt_sha256": repair.digest(prompt), "native_prompt": prompt,
                             "public_observation": observation, "skill_sha256": repair.digest(skill)}
        require(decision == expected_decision, "Decision sidecar differs")
        route = replay["decision"]["route"]
        routes[f"{route}_steps"] += 1
        routes["jev_then_cot_steps"] += int(route == "cot" and replay["decision"]["jev_calls"] == 1)
        routes["local_then_cot_steps"] += int(route == "cot" and replay["decision"].get("local_calls") == 1)
        if memory_checks is not None:
            memory_checks["prefilter_bypasses"] += int(replay["decision"]["pre_route"].startswith("cot_memory_"))
            if replay["decision"]["local_calls"]:
                memory_checks["retrieval_calls"] += 1
                memory_checks["retrieved_entries"] += len(replay["decision"]["rule_memory"]["retrieved_rule_ids"])
            memory_checks["code_rejected_choices"] += int(replay["decision"]["reason"] == "local_choice_failed_code_prerequisite")
        if progress_checks is not None:
            progress_checks["recovery_steps"] += int(replay["decision"]["typed_progress"]["recovery_active"])
            if replay["decision"]["local_calls"] or route == "deterministic":
                supported = replay["decision"]["typed_progress"]["public_constraint"]
                require(supported["complete_native_actions"] == commands
                        and set(supported["allowed_actions"]) <= set(commands), "Typed progress uses a nonnative action")
                progress_checks["typed_calls"] += replay["decision"]["local_calls"]
                progress_checks["typed_actions"] += int(route == "local")
                progress_checks["native_options"] += len(commands)
                progress_checks["supported_options"] += len(supported["allowed_actions"])
        histories.append({"action": step["action"], "observation": step["observation"]})
    require(used == {filename.parent for filename in (root / "providers").glob("*/*/request.json")},
            "Hybrid episode includes undeclared or missing provider calls")
    require({key: sum(step["usage"][key] for step in conversation) for key in repair.USAGE_KEYS} == row["usage"],
            "Hybrid per-step usage differs from actual providers")
    return {**routes, "provider_usage": actual_ledger["provider_usage"], "cost_complete": actual_ledger["cost_complete"],
            **({"rule_memory_checks": memory_checks} if memory_checks is not None else {}),
            **({"typed_progress_checks": progress_checks} if progress_checks is not None else {}),
            "typesafe_api_used": not is_local, "true_jev_used": not is_local}


def audit_reused_cells(run: Path, config: dict, freeze: dict) -> tuple[dict, dict | None]:
    source_name = config.get("preserve_completed_from")
    if source_name is None:
        return {}, None
    source = Path(source_name).resolve()
    receipt = repair.read_json(run / "REUSED_CELLS.json")
    require(Path(receipt["source_run"]).resolve() == source and source_name in config["prior_run_directories"],
            "Reuse source is not a charged declared prior run")
    for name, key in (("config.json", "source_config_sha256"), ("manifest.json", "source_manifest_sha256"),
                      ("development60-freeze.json", "source_freeze_sha256")):
        require(hash_file(source / name) == receipt[key], "Reused source evidence changed")
    prior_config = repair.read_json(source / "config.json")
    fields = ("evaluation_scope", "module_variants", "seed", "episode_seed", "workers",
              "family_step_caps", "episode_budget", "model", "parent_manifest_sha256",
              "parent_request_sha256", "historical_reference", "local_discriminator", "recovery_ablation_study")
    require(all(config.get(key) == prior_config.get(key) for key in fields), "Recovery scientific contract drift")
    prior_freeze = repair.read_json(source / "development60-freeze.json")
    require(prior_freeze["policies"] == freeze["policies"] and prior_freeze["task_ids"] == freeze["task_ids"],
            "Recovery methods or task inventory changed")
    provenance = repair.read_json(source / "PROVENANCE.json")
    original_source = Path(provenance["argv"][0]).resolve().parents[1]
    for name, expected in provenance["source_hashes"].items():
        require(hash_file(original_source / name) == expected, "Prior frozen source changed during recovery")
    reused = {}
    for cell in receipt["cells"]:
        root = Path(cell["artifact_dir"]).resolve()
        require(root.is_relative_to(source) and str(root) not in reused, "Invalid reused artifact location")
        require(hash_file(root / "result.json") == cell["result_sha256"]
                and hash_file(root / "job.json") == cell["job_sha256"], "Reused raw episode changed")
        row = repair.read_json(root / "result.json")
        require(row["valid"] is True and row["cost_complete"] is True
                and row["id"] == cell["id"] and row["method"] == cell["arm"], "Invalid reusable episode")
        reused[str(root)] = cell
    expected = {str(filename.parent.resolve()) for filename in (source / receipt["phase"]).glob("*/*/result.json")
                if repair.read_json(filename).get("valid") is True}
    require(set(reused) == expected, "Recovery omitted or repeated completed cells based on outcomes")
    return reused, prior_freeze


def ablation_all_attempt_costs(run: Path, result: dict, cost: dict) -> dict:
    completed = repair.read_json(run / "development60-rows.json")
    require(set(completed) == set(RECOVERY_ABLATION_ARMS), "All-attempt accounting lacks a declared ablation")
    expected_roots = {str(Path(row["artifact_dir"]).resolve()) for rows in completed.values() for row in rows}
    require(len(expected_roots) == 420, "All-attempt accounting requires420 distinct complete episodes")
    totals = {arm: {"usage": dict.fromkeys(repair.USAGE_KEYS, 0), "calls": 0,
                    "failed_prefix_usage": dict.fromkeys(repair.USAGE_KEYS, 0), "failed_prefix_calls": 0,
                    "failed_prefixes": 0, "completed_episodes": 0} for arm in RECOVERY_ABLATION_ARMS}
    pending = [run.resolve()]
    visited = set()
    actual_complete = set()
    raw_hashes = {}
    non_episode_hashes = {}
    non_episode_usage = dict.fromkeys(repair.USAGE_KEYS, 0)
    all_responses = set()
    while pending:
        directory = pending.pop()
        require(directory not in visited, "An ablation source run would be charged twice")
        visited.add(directory)
        config = repair.read_json(directory / "config.json")
        pending.extend(Path(value).resolve() for value in config.get("prior_run_directories", []))
        all_responses.update(str(filename.resolve()) for filename in directory.rglob("response.json"))
        for name in ("admission", "local-admission"):
            response_path = directory / name / "response.json"
            if not response_path.exists():
                continue
            charged = validate_call(repair.read_json(response_path.parent / "request.json"),
                repair.read_json(response_path), repair.read_json(response_path.parent / "accounting.json"),
                config["model"]["served_name"])
            non_episode_hashes[str(response_path)] = hash_file(response_path)
            for key in non_episode_usage:
                non_episode_usage[key] += charged[key]
        for filename in sorted((directory / "development60").glob("*/*/result.json")):
            row = repair.read_json(filename)
            arm = row["method"]
            require(arm in totals and row["cost_complete"] is True, "An online ablation attempt has unknown cost")
            root = filename.parent.resolve()
            responses = sorted((root / "providers").glob("*/*/response.json")) + sorted((root / "calls").glob("*/response.json"))
            usage = dict.fromkeys(repair.USAGE_KEYS, 0)
            for response_path in responses:
                response = repair.read_json(response_path)
                request = repair.read_json(response_path.parent / "request.json")
                accounting = repair.read_json(response_path.parent / "accounting.json")
                charged = validate_call(request, response, accounting, config["model"]["served_name"])
                for key in usage:
                    usage[key] += charged[key]
                require(str(response_path) not in raw_hashes, "An online provider call would be counted twice")
                raw_hashes[str(response_path)] = hash_file(response_path)
            require(usage == row["usage"] and len(responses) == row["accounted_calls"], "Online attempt cost differs from raw calls")
            if row["valid"] is True:
                require(str(root) in expected_roots, "A complete episode was excluded from the final comparison")
                actual_complete.add(str(root))
                totals[arm]["completed_episodes"] += 1
            else:
                require(row["official_success"] is None, "Interrupted attempt was incorrectly scored")
                totals[arm]["failed_prefixes"] += 1
                totals[arm]["failed_prefix_calls"] += len(responses)
                for key in usage:
                    totals[arm]["failed_prefix_usage"][key] += usage[key]
            totals[arm]["calls"] += len(responses)
            for key in usage:
                totals[arm]["usage"][key] += usage[key]
    require(actual_complete == expected_roots, "Final comparison contains an unaudited episode")
    methods = {}
    for arm, accounted in totals.items():
        measured = result["methods"][arm]
        require(accounted["completed_episodes"] == measured["tasks"] == 60, "All-attempt denominator changed")
        usage = accounted["usage"]
        for key in repair.USAGE_KEYS:
            require(usage[key] == measured[key] + accounted["failed_prefix_usage"][key], "Completed and interrupted costs do not add")
        methods[arm] = {**measured, **usage, "tokens_per_task": usage["total_tokens"] / 60,
                        "tokens_per_task_k": usage["total_tokens"] / 60000,
                        "completed_episode_usage": {key: measured[key] for key in repair.USAGE_KEYS},
                        "failed_prefix_usage": accounted["failed_prefix_usage"],
                        "failed_prefixes": accounted["failed_prefixes"], "online_calls": accounted["calls"]}
    online = sum(row["total_tokens"] for row in methods.values())
    calls = sum(row["online_calls"] for row in methods.values())
    require(set(raw_hashes) | set(non_episode_hashes) == all_responses, "An acquired response has no cost category")
    require(online + non_episode_usage["total_tokens"] == cost["all_in_total_tokens"]
            and calls + len(non_episode_hashes) == cost["all_in_calls"], "Online and admission costs differ from research ledger")
    return {"status": "PASS_ALL_ONLINE_ATTEMPTS", "result_sha256": hash_file(run / "RESULT.json"),
            "methods": methods, "online_total_tokens": online, "online_calls": calls,
            "research_total_tokens": cost["all_in_total_tokens"],
            "non_episode_tokens": non_episode_usage["total_tokens"],
            "non_episode_calls": len(non_episode_hashes), "non_episode_response_sha256": non_episode_hashes,
            "raw_response_sha256": raw_hashes, "completed_episodes": 420,
            "source_runs": sorted(str(directory) for directory in visited),
            "model_calls_during_audit": 0, "score_denominator_includes_all60": True,
            "interrupted_online_attempts_charged": True}


def audit(run: Path) -> dict:
    config = repair.read_json(run / "config.json")
    repair.remote_only(config["authorized_hostname"])
    result = repair.read_json(run / "RESULT.json")
    manifest = repair.read_json(run / "manifest.json")
    binding = repair.read_json(run / "binding.json")
    is_direct60 = direct_development(config)
    require(manifest["manifest_sha256"] == repair.digest({key: value for key, value in manifest.items() if key != "manifest_sha256"}), "Manifest hash changed")
    require(binding["binding_sha256"] == repair.digest({key: value for key, value in binding.items() if key != "binding_sha256"}), "Binding hash changed")
    require(manifest["config_sha256"] == repair.digest(config), "Configuration differs from frozen manifest")
    if config.get("module_variants") == ["cot_local"]:
        require_local_binding(binding)
        require(binding.get("local_discriminator_logprobs_mode") == "processed_logprobs", "Local probability mode not pinned")
        command = repair.read_json(run / "SERVICE.json")["command"]
        require(command[command.index("--logprobs-mode") + 1] == "processed_logprobs"
            and int(command[command.index("--max-logprobs") + 1]) >= 63, "Service logprob contract differs")
        require(result["typesafe_api_used"] is False and result["true_jev_used"] is False,
            "Local experiment is incorrectly labeled as real Jev")
    provenance = repair.read_json(run / "PROVENANCE.json")
    source_root = Path(provenance["argv"][0]).resolve().parents[1]
    for name, expected in provenance["source_hashes"].items():
        require(hash_file(source_root / name) == expected, f"Frozen source changed: {name}")
    parents = manifest["baseline_profiles"]
    policies = {"parent": compile_policy(parents)}
    tokenizer = None
    has_episodes = (run / "train_smoke-rows.json").exists() or (run / "development60-rows.json").exists()
    if (config.get("context_variants") or config.get("module_variants")) and has_episodes:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["model"]["directory"], local_files_only=True, trust_remote_code=False)
        if not is_direct60:
            context_proposal = repair.read_json(run / "context-proposal.json")
            require(context_proposal["accepted"] is True, "Context candidate lacks accepted training diagnosis")
        for variant in config.get("context_variants", []):
            policies[variant] = compile_policy(parents, variant)
        if config.get("module_variants"):
            if is_direct60:
                policies = direct_development_policies(config, parents)
            else:
                require(config["module_variants"] == ["brief_action", "context_all_brief"], "Unexpected module factorial")
                policies["brief_action"] = compile_policy(parents, response_variant="brief_action")
                policies["context_all_brief"] = compile_policy(parents, "context_all", "brief_action")
    support = {item["id"]: item["group_id"] for item in manifest["splits"]["train"]}
    train_groups = set(support.values())
    reused_cells = {}
    reused_freeze = None
    if is_direct60:
        development_freeze = repair.read_json(run / "development60-freeze.json")
        validate_development60_freeze(config, manifest, development_freeze, policies)
        reused_cells, reused_freeze = audit_reused_cells(run, config, development_freeze)
        original = repair.read_json(Path(config["parent_directory"]) / "evaluations/confirmation/ours_adm/request.json")
        require(development_freeze["task_ids"] == original["task_ids"], "Direct60 differs from original CHISEL cohort")
        require({filename.name for filename in run.glob("*-rows.json")} == {"development60-rows.json"},
                "Direct60 ran an undeclared selection or duplicate confirmation phase")
    else:
        require(not train_groups & {item["group_id"] for item in manifest["splits"]["validation"]}, "Training/validation floorplans overlap")
    for filename in sorted((run / "candidates").glob("*.json")):
        candidate = repair.read_json(filename)
        proposal = candidate["proposal"]
        module = proposal["module"]
        ids = proposal["support_task_ids"]
        require(set(ids) <= set(support) and len({support[task_id] for task_id in ids}) >= 2, "Candidate support is not training-only or lacks two groups")
        expected_profiles = json.loads(json.dumps(parents))
        for family, components in expected_profiles.items():
            components["shared_skill"] = components["shared_skill"].replace(EDITABLE_SPANS[module], proposal["replacement"], 1)
        require(expected_profiles == candidate["profiles"], "Candidate changed outside its declared module")
        require(candidate["review"]["accept"] is True, "Candidate was not accepted by diff review")
        policies[module] = compile_policy(expected_profiles)
    if (run / "integration.json").exists():
        integration = repair.read_json(run / "integration.json")
        policies["integrated"] = compile_policy(integration["profiles"])
    if (run / "candidate-freeze.json").exists():
        release = repair.read_json(run / "candidate-freeze.json")
        require(release["sha256"] == repair.digest({key: value for key, value in release.items() if key != "sha256"}), "Candidate freeze hash differs")
        require(release["policy"] == policies[release["winner"]], "Released candidate differs from validation policy")
        policies["candidate"] = release["policy"]
    rows_by_phase = {}
    tables = {}
    context_checks = {}
    hybrid_checks = {}
    for phase_file in sorted(run.glob("*-rows.json")):
        phase = phase_file.name.removesuffix("-rows.json")
        split = "train" if phase == "train_smoke" else "historical60" if phase == "historical60" else "validation"
        items = manifest["splits"][split][:2] if phase == "train_smoke" else manifest["splits"][split]
        inventory = {item["id"]: item for item in items}
        arms = repair.read_json(phase_file)
        if is_direct60:
            require(set(arms) == set(policies), "Direct development60 is missing a declared method")
        rows_by_phase[phase] = arms
        tables[phase] = {}
        for arm, rows in arms.items():
            context_stats = {"requests": 0, "changed_requests": 0, "fixed_trajectory_input_tokens_saved": 0,
                             "growing_requests": 0, "context_changed_requests": 0, "response_changed_requests": 0,
                             "boilerplate_changed_requests": 0}
            require(len(rows) == len(inventory) and {row["id"] for row in rows} == set(inventory), "Incomplete task coverage")
            for row in rows:
                root = Path(row["artifact_dir"])
                reused = str(root.resolve()) in reused_cells
                require(root.is_relative_to(run) or reused, "Episode artifact is outside declared new/reused runs")
                if reused:
                    cell = reused_cells[str(root.resolve())]
                    require(cell["id"] == row["id"] and cell["arm"] == arm, "Reused cell identity differs")
                require(repair.read_json(root / "result.json") == {key: value for key, value in row.items() if key != "task_id"}, "Summary row differs from raw episode")
                job = repair.read_json(root / "job.json")
                require(job["item"] == inventory[row["id"]] and job["policy"] == policies[arm], "Executed task or policy differs")
                require(row["seed"] == config["episode_seed"], "Episode seed differs")
                if "jev" in job["policy"] or "local_discriminator" in job["policy"]:
                    hybrid_checks[f"{phase}/{arm}/{row['id']}"] = audit_hybrid_episode(root, row, job["policy"], config, tokenizer)
                    for filename in (root / "providers").glob("*/*/request.json"):
                        cell_freeze = reused_freeze if reused else development_freeze
                        require(datetime.fromisoformat(cell_freeze["frozen_at"]).timestamp() <= filename.stat().st_mtime,
                                "Hybrid request precedes frozen method")
                    continue
                conversation = repair.read_json(root / "conversation.json")
                initial = repair.read_json(root / "initial_observation.json")["observation"]
                validate_terminal(row, conversation, config["family_step_caps"][row["task_type"]])
                usage_total = dict.fromkeys(repair.USAGE_KEYS, 0)
                calls = sorted((root / "calls").iterdir())
                require(len(calls) == row["turns"] == row["accounted_calls"], "Actor call count differs")
                for turn, call in enumerate(calls):
                    request = repair.read_json(call / "request.json")
                    body = request["body"]
                    require(request["stage"] == "actor" and body["max_tokens"] == config["episode_budget"]["max_completion_tokens"], "Actor output cap changed")
                    require(body["seed"] == row["seed"] and body["messages"][0] == {"role": "system", "content": repair.SYSTEM}, "Actor seed or system changed")
                    usage = validate_call(request, repair.read_json(call / "response.json"), repair.read_json(call / "accounting.json"), config["model"]["served_name"])
                    require(conversation[turn]["usage"] == usage, "Turn usage differs from provider")
                    if tokenizer is not None:
                        require(native_chat_tokens(tokenizer, body["messages"]) == usage["prompt_tokens"], "Native tokenization differs from provider input usage")
                    variant = job["policy"].get("context_variant")
                    response_variant = job["policy"].get("response_variant")
                    boilerplate_variant = job["policy"].get("boilerplate_variant")
                    if variant or response_variant or boilerplate_variant:
                        context = repair.read_json(root / "context-inputs" / f"{turn:03d}.json")
                        original = context["original_messages"]
                        require(body["messages"] == context["transformed_messages"], "Actor request differs from transformed context")
                        public_history = [{"action": step["action"], "observation": step["observation"]} for step in conversation[:turn]]
                        count_tokens = lambda text: native_chat_tokens(tokenizer, [original[0], {"role": "user", "content": text}])
                        expected_text = original[1]["content"]
                        transformation = None
                        if variant:
                            expected_text, transformation = replace_history(expected_text, initial, public_history,
                                                                             variant=variant, count_tokens=count_tokens)
                        require(transformation == context["transformation"], "Public-history context replay differs")
                        response_transformation = None
                        if response_variant:
                            require(response_variant == "brief_action", "Unknown response contract")
                            expected_text, response_transformation = compact_action_contract(expected_text, count_tokens=count_tokens)
                        require(response_transformation == context.get("response_transformation"), "Response-contract replay differs")
                        boilerplate_transformation = None
                        if boilerplate_variant:
                            require(not variant and not response_variant, "Static boilerplate candidate changed another module")
                            expected_text, boilerplate_transformation = compact_boilerplate(
                                expected_text, system=original[0]["content"], variant=boilerplate_variant,
                                count_tokens=count_tokens,
                            )
                        require(boilerplate_transformation == context.get("boilerplate_transformation"), "Static boilerplate replay differs")
                        require(expected_text == body["messages"][1]["content"], "Composed actor prompt replay differs")
                        before_tokens = native_chat_tokens(tokenizer, original)
                        after_tokens = usage["prompt_tokens"]
                        require(after_tokens <= before_tokens, "Composed modules grew prompt tokens")
                        context_changed = bool((transformation or {}).get("changed"))
                        response_changed = bool((response_transformation or {}).get("changed"))
                        boilerplate_changed = bool((boilerplate_transformation or {}).get("changed"))
                        context_stats["requests"] += 1
                        context_stats["changed_requests"] += int(context_changed or response_changed or boilerplate_changed)
                        context_stats["context_changed_requests"] += int(context_changed)
                        context_stats["response_changed_requests"] += int(response_changed)
                        context_stats["boilerplate_changed_requests"] += int(boilerplate_changed)
                        context_stats["fixed_trajectory_input_tokens_saved"] += before_tokens - after_tokens
                        context_stats["growing_requests"] += int(after_tokens > before_tokens)
                    for key in usage_total:
                        usage_total[key] += usage[key]
                    if phase == "historical60":
                        require(datetime.fromisoformat(release["frozen_at"]).timestamp() <= (call / "request.json").stat().st_mtime, "Historical outcome was opened before candidate freeze")
                    if phase == "development60":
                        cell_freeze = reused_freeze if reused else development_freeze
                        require(datetime.fromisoformat(cell_freeze["frozen_at"]).timestamp() <= (call / "request.json").stat().st_mtime,
                                "Development60 actor call precedes method freeze")
                require(usage_total == row["usage"], "Episode tokens differ from raw response totals")
            successes = sum(row["official_success"] for row in rows)
            tokens = sum(row["usage"]["total_tokens"] for row in rows)
            tables[phase][arm] = {"tasks": len(rows), "successes": successes, "score": 100 * successes / len(rows),
                                  "total_tokens": tokens, "tokens_per_task": tokens / len(rows)}
            if context_stats["requests"]:
                require(context_stats["changed_requests"] > 0, "Declared context intervention never activated")
                if policies[arm].get("context_variant"):
                    require(context_stats["context_changed_requests"] > 0, "Declared memory module never activated")
                if policies[arm].get("response_variant"):
                    require(context_stats["response_changed_requests"] > 0, "Declared loop output module never activated")
                if policies[arm].get("boilerplate_variant"):
                    require(context_stats["boilerplate_changed_requests"] > 0, "Declared boilerplate removal never activated")
                context_checks[f"{phase}/{arm}"] = context_stats
    if "validation" in rows_by_phase:
        rows = rows_by_phase["validation"]
        if "integration_validation" in rows_by_phase:
            rows.update(rows_by_phase["integration_validation"])
        ids = [item["id"] for item in manifest["splits"]["validation"]]
        for arm, gate in result["validation_gates"].items():
            require(paired_gate(rows["parent"], rows[arm], expected_ids=ids) == gate, "Validation gate differs from raw episode results")
    if "development60" in rows_by_phase and config.get("recovery_ablation_study") is not None:
        ids = [item["id"] for item in manifest["splits"]["validation"]]
        comparison_arm = recovery_ablation_full_arm(config["recovery_ablation_study"])
        summary = recovery_ablation_summary(rows_by_phase["development60"], ids, comparison_arm=comparison_arm)
        require(all(result.get(key) == value for key, value in summary.items()), "Ablation result differs from raw replay")
        require(result["reused_completed_cells"] == len(reused_cells), "Ablation continuation omitted completed cells")
        for arm, measured in tables["development60"].items():
            require(all(result["methods"][arm][key] == value for key, value in measured.items()),
                "Ablation table differs from independently summed episodes")
        activation = recovery_ablation_activation(run)
        require(activation["passed"] and activation == repair.read_json(run / "RECOVERY_ABLATION_ACTIVATION.json"),
            "Ablation mechanism activation differs")
        reference = Path(config["recovery_ablation_study"]["reference_run_directory"])
        prior = repair.read_json(reference / "development60-freeze.json")
        figure_binding = repair.read_json(run / "FIGURE1A_BINDING.json")
        bound_arms = ("parent", "no_periodic_refresh", "no_typed_selector") if not config["recovery_ablation_study"].get("base_periodic_refresh", True) else ("parent", "cot_local")
        require(figure_binding["reference_freeze_sha256"] == hash_file(reference / "development60-freeze.json")
            and prior["task_ids"] == ids
            and all(prior["policies"][arm] == policies[arm] for arm in bound_arms),
            "Figure1(a) full-policy or task binding differs")
        if not config["recovery_ablation_study"].get("base_periodic_refresh", True):
            require(figure_binding["full_arm"] == comparison_arm
                and figure_binding["full_policy_sha256"] == policies[comparison_arm]["harness_sha256"],
                "No-refresh CHISEL identity differs from the selected prior policy")
    elif "development60" in rows_by_phase:
        rows = rows_by_phase["development60"]
        ids = [item["id"] for item in manifest["splits"]["validation"]]
        gates = {arm: direct_development_gate(rows["parent"], rows[arm], ids, config["historical_reference"])
             for arm in config["module_variants"]}
        require(gates == result["development_gates"], "Development60 gate differs from raw paired results")
        qualified = [arm for arm, gate in gates.items() if gate["passed"]]
        winner = min(qualified, key=lambda arm: (-gates[arm]["arms"]["candidate"]["successes"],
                            gates[arm]["arms"]["candidate"]["total_tokens"], arm)) if qualified else None
        require(result["winner"] == winner and result["fresh_confirmation"] is False
            and result["selection_uses_development60_outcomes"] is True, "Development-only selection identity differs")
        require(result["status"] == ("DEVELOPMENT60_TARGET_MET" if winner else "DEVELOPMENT60_TARGET_NOT_MET"),
            "Development60 target status differs from audited gates")
        if "cot_jev" in rows:
            require(sum(record["jev_steps"] for record in hybrid_checks.values()) > 0,
                    "No Jev-selected actions executed; not an interleaving experiment")
        if "cot_local" in rows:
            require(sum(record["local_steps"] for record in hybrid_checks.values()) > 0,
                    "No local discriminator actions executed")
    if "historical60" in rows_by_phase:
        rows = rows_by_phase["historical60"]
        expected = paired_gate(rows["parent"], rows["candidate"], expected_ids=[item["id"] for item in manifest["splits"]["historical60"]], minimum_score=66.7)
        require(expected == result["historical60"], "Historical target gate differs")
        cost_pass = expected["arms"]["candidate"]["total_tokens"] <= config["historical_reference"]["total_tokens"]
        require(cost_pass == result["historical_token_ceiling_pass"], "Historical fixed token ceiling differs")
    cost = audit_cost_tree(run, set())
    retirement = repair.read_json(run / "RETIREMENT.json")
    require(retirement["released"] and retirement["allocation_cleared"], "GPU allocation was not retired")
    evidence = [] if is_direct60 else repair.read_json(run / "training-evidence.json")
    evidence_complete = None if is_direct60 else all(
        isinstance(step.get("observation"), str) and bool(step["observation"].strip())
        for case in evidence for trajectory in case["trajectories"] for step in trajectory["steps"]
    )
    return {"status": "PASS", "run": str(run), "result_status": result["status"], "tables": tables,
            "cost": cost, "official_valid_unseen_calls": 0, "paper_admitted": False,
            "context_checks": context_checks,
            "hybrid_checks": hybrid_checks,
            "typesafe_api_used": any(record["typesafe_api_used"] for record in hybrid_checks.values()),
            "reused_completed_cells": len(reused_cells),
            "diagnostic_public_feedback_complete": evidence_complete,
            "evaluation_scope": config.get("evaluation_scope", "development18_then_historical60"),
            "fresh_confirmation": False,
            "diagnosis_support_status": "PREVIOUSLY_TESTED_CANDIDATES_NO_NEW_DIAGNOSIS" if is_direct60 else
                "COMPLETE_PUBLIC_EXCERPTS" if evidence_complete else "INCOMPLETE_ACTION_ONLY_EVIDENCE",
            "hostname": socket.gethostname(), "audit_source_sha256": hash_file(Path(__file__)), "at_utc": repair.now()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run.resolve()
    config = repair.read_json(run_dir / "config.json")
    repair.remote_only(config["authorized_hostname"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.output.mkdir(parents=True, exist_ok=False)
    repair.write_json(args.output / "command.json", {"argv": sys.argv, "hostname": socket.gethostname(),
                      "python": sys.version, "config_sha256": hash_file(run_dir / "config.json"),
                      "seed": config["seed"], "source_provenance": repair.read_json(ROOT / "SOURCE_PROVENANCE.json"),
                      "packages": repair.capture([sys.executable, "-m", "pip", "freeze"])})
    try:
        verified = audit(run_dir)
        repair.write_json(args.output / "VERIFIED.json", verified)
        print(json.dumps(verified), flush=True)
    except BaseException:
        repair.write_json(args.output / "FAILURE.json", {"traceback": traceback.format_exc(), "at_utc": repair.now()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
