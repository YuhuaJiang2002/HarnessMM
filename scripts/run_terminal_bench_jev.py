#!/usr/bin/env python3
"""Freeze an independent matched Jev-inspired Terminal-Bench study."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import select
import shutil
import socket
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_terminal_bench_native as native
from scripts.terminal_jev import EXPLICIT_PLAN_INSTRUCTION, POSTCONDITION_PLAN_INSTRUCTION, TerminalJevSession


ARMS = ("no_skill", "chisel", "chisel_pack", "chisel_batch", "chisel_jev")


def validate_protocol(protocol):
    if (protocol.get("schema_version") != "terminal-jev-full89/v1"
            or protocol.get("benchmark") != "terminal-bench@2.0"
            or protocol.get("dataset_revision") != "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
            or protocol.get("model_revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            or protocol.get("arms") != list(ARMS) or protocol.get("task_count") != 89
            or protocol.get("repeat_seeds") != [0, 1, 2, 3, 4]
            or protocol.get("maximum_followups") != 2
            or protocol.get("typesafe_api_used") is not False
            or protocol.get("calibration_claimed") is not False
            or protocol.get("previous_trial_scores_reused") is not False):
        raise ValueError("Jev study must retain all 89 tasks, five repeats and the admitted matched arms")
    allocation = protocol.get("allocation", {})
    bounds = {"maximum_seconds": 86400, "maximum_requests": 50000,
              "maximum_provider_tokens": 200000000, "concurrency": 8}
    if set(allocation) != set(bounds) or any(type(allocation[key]) is not int or
            not 0 < allocation[key] <= maximum for key, maximum in bounds.items()):
        raise ValueError("study allocation exceeds the bounded resource reservation")
    if type(protocol.get("schedule_seed")) is not int:
        raise ValueError("study order requires a frozen integer seed")
    if protocol.get("interface_policy", "optional_v1") not in ("optional_v1", "explicit_plan_v2"):
        raise ValueError("unregistered prospective interface policy")
    if protocol.get("condition_target", "previous") not in ("previous", "self"):
        raise ValueError("unregistered plan condition semantics")


def compose_actors(parent_binding, protocol):
    validate_protocol(protocol)
    actors = {"no_skill": deepcopy(parent_binding["actors"]["no_skill"]),
              "chisel": deepcopy(parent_binding["actors"]["chisel"])}
    if actors["chisel"].get("jev") is not None:
        raise ValueError("the CHISEL parent must be the unchanged original method")
    for arm in ARMS[2:]:
        actor = deepcopy(actors["chisel"])
        actor["solpi"] = {"observation_pack": True, "action_fusion": False}
        if arm in ("chisel_batch", "chisel_jev"):
            actor["jev"] = {"mode": "guarded" if arm == "chisel_jev" else "success_only",
                            "maximum_followups": protocol["maximum_followups"]}
            if protocol.get("interface_policy") == "explicit_plan_v2":
                actor["jev"]["explicit_plan"] = True
                actor["jev"]["condition_target"] = protocol.get("condition_target", "previous")
                actor["system_prompt"] += (POSTCONDITION_PLAN_INSTRUCTION
                    if actor["jev"]["condition_target"] == "self" else EXPLICIT_PLAN_INSTRUCTION)
            TerminalJevSession(actor["jev"])
        actors[arm] = actor
    return actors


def build_plan(parent_plan, protocol):
    validate_protocol(protocol)
    cells = {}
    for row in parent_plan:
        task = row["task_id"]
        seed = row["repeat_seed"]
        if (not isinstance(task, str) or Path(task).name != task or task in (".", "..")
                or type(seed) is not int or seed not in protocol["repeat_seeds"]):
            raise ValueError("invalid source task or seed")
        cells.setdefault((task, seed), row)
    tasks = {task for task, _seed in cells}
    expected = {(task, seed) for task in tasks for seed in protocol["repeat_seeds"]}
    if len(tasks) != 89 or set(cells) != expected:
        raise ValueError("study cannot omit tasks or recalculate one rollout as five trials")
    order = sorted(cells)
    random.Random(protocol["schedule_seed"]).shuffle(order)
    result = []
    for position, (task, seed) in enumerate(order):
        offset = position % len(ARMS)
        methods = ARMS[offset:] + ARMS[:offset]
        for arm in methods:
            result.append({"trial_id": f"{arm}/{task}/repeat-{seed + 1:02d}-seed-{seed}",
                           "task_id": task, "repeat_seed": seed, "repeat": seed + 1,
                           "arm": arm, "source_config_path": cells[(task, seed)]["config_path"],
                           "source_config_sha256": cells[(task, seed)]["config_sha256"]})
    return result


def merge_stage_usage(previous, admission):
    native.UsageLedger(previous)
    native.UsageLedger(admission)
    fields = ("attempts", "responses_with_counts", "input_tokens", "output_tokens", "total_tokens",
              "preinference_rejections", "rejected_prompt_tokens_lower_bound", "rejection_budget_reservation_tokens")
    combined = {key: previous.get(key, 0) + admission.get(key, 0) for key in fields}
    if (combined["input_tokens"] + combined["output_tokens"] != combined["total_tokens"]
            or combined["responses_with_counts"] + combined["preinference_rejections"] != combined["attempts"]):
        raise ValueError("stage usage has unaccounted requests or inconsistent token totals")
    combined.update(at_utc=native.now(), prior_attempts=previous["attempts"], complete=True,
                    inflight=0, unknown_usage_attempts=0,
                    all_attempts_returned_native_usage=not combined["preinference_rejections"])
    return combined


def prior_stage_evidence(prior_root):
    prior_result = json.loads((prior_root / "RESULT.json").read_text())
    if prior_result["status"] != "BLOCKED":
        raise ValueError("only a closed prototype stage can precede an interface revision")
    usage = json.loads((prior_root / "PROVIDER_FINAL.json").read_text())
    native.UsageLedger(usage)
    execution = json.loads((prior_root / "EXECUTION.json").read_text())
    started = datetime.fromisoformat(json.loads((prior_root / "STARTED.json").read_text())["at_utc"])
    deadline = execution.get("study_deadline_utc") or (started + timedelta(seconds=execution["maximum_seconds"])).isoformat()
    return {"prior_study_root": str(prior_root),
            "prior_study_provider_final_sha256": native.sha256(prior_root / "PROVIDER_FINAL.json"),
            "prior_study_result_sha256": native.sha256(prior_root / "RESULT.json"),
            "prior_study_usage": usage, "study_deadline_utc": deadline, "prior_stage_scores_pooled": False}


def activation_check(args, root, protocol, service):
    from scripts.run_terminal_bench_qwen35_4b_direct import run_agent

    if protocol.get("interface_policy") != "explicit_plan_v2" or not args.parent_run:
        raise ValueError("activation check requires the explicit plan and original actor binding")
    binding = json.loads((args.parent_run / "actor-binding.json").read_text())
    actor = compose_actors(binding, protocol)["chisel_jev"]
    model = deepcopy(binding["model"])
    model.update(chat_completions_url=f"http://127.0.0.1:{args.port}/v1/chat/completions")
    if model["expected_response_model_id"] != service["model"]:
        raise ValueError("activation actor and admitted model differ")
    container = "terminal-jev-synthetic-" + native.sha256(args.service_receipt)[:12]
    command = ["docker", "run", "--detach", "--name", container, "--cpus", "1", "--memory", "2g",
               "--entrypoint", "tail", "alexgshaw/bn-fit-modify:20251031", "-f", "/dev/null"]
    native.write(root / "ACTIVATION_CONFIG.json", {"command": command, "actor": actor,
        "model": model, "official_task_read": False, "synthetic_task": True, "timeout_seconds": 300})
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    try:
        result = run_agent(config={"model": model, "agent": actor},
            instruction=("Synthetic terminal check: create /tmp/typed_probe.py that prints exactly 42 and a newline. "
                         "Then run python3 /tmp/typed_probe.py to check it. The two commands are known in advance. "
                         "Use one explicit typed plan for creation and validation. Do not inspect task assets or /app. "
                         "Finish after the validation returns 42."),
            container_name=container, trial_dir=root / "activation", agent_timeout_seconds=300)
        check = subprocess.run(["docker", "exec", container, "python3", "/tmp/typed_probe.py"],
                               capture_output=True, text=True, timeout=15)
        executed = sum(row["followup_actions_without_extra_model_calls"] for row in result["jev"]["traces"])
        if not executed or check.returncode or check.stdout != "42\n":
            raise ValueError("explicit typed continuation did not execute its synthetic public validation")
        receipt = {"status": "PASS", "identity": "synthetic behavior check, not benchmark performance",
                   "environment_actions": result["environment_tool_calls_completed"],
                   "executed_followups": executed, "model_calls": result["model_calls"],
                   "actor_usage": {key: result[key] for key in ("input_tokens", "output_tokens", "total_tokens")},
                   "public_check_stdout": check.stdout, "jev": result["jev"]}
        native.write(root / "ACTIVATION_PASS.json", receipt)
        return receipt
    finally:
        subprocess.run(["docker", "stop", "--time", "5", container], capture_output=True, text=True, timeout=20)
        subprocess.run(["docker", "rm", container], capture_output=True, text=True, timeout=20)


def admit(args):
    from scripts.run_terminal_bench_qwen35_4b_direct import EXEC_TOOL

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    if protocol is not None:
        validate_protocol(protocol)
        native.MAX_REQUESTS = protocol["allocation"]["maximum_requests"]
        native.MAX_TOTAL_TOKENS = protocol["allocation"]["maximum_provider_tokens"]
    prior_usage = None
    prior_evidence = {}
    if args.prior_admission:
        previous_admission = args.prior_admission.resolve()
        if not (previous_admission / "FAILURE.json").is_file():
            raise ValueError("admission recovery must retain an explicitly failed previous check")
        prior_usage = json.loads((previous_admission / "PROVIDER_FINAL.json").read_text())
        native.UsageLedger(prior_usage)
        prior_evidence = {"prior_admission_root": str(previous_admission),
            "prior_admission_usage": prior_usage,
            "prior_admission_final_sha256": native.sha256(previous_admission / "PROVIDER_FINAL.json"),
            "prior_admission_failure_sha256": native.sha256(previous_admission / "FAILURE.json")}
    if args.prior_study:
        prior_root = args.prior_study.resolve()
        stage = prior_stage_evidence(prior_root)
        prior_usage = merge_stage_usage(stage["prior_study_usage"], prior_usage) if prior_usage else stage["prior_study_usage"]
        prior_evidence.update(stage)
    service = json.loads(args.service_receipt.read_text())
    if service.get("status") != "READY" or service.get("preinference_validation_source_sha256") != native.PREINFERENCE_RUNTIME_SOURCES:
        raise ValueError("provider admission requires the verified model service")
    session = TerminalJevSession({"mode": "guarded", "maximum_followups": 2,
                                 "explicit_plan": bool(protocol and protocol.get("interface_policy") == "explicit_plan_v2"),
                                 "condition_target": protocol.get("condition_target", "previous") if protocol else "previous"})
    arguments = {"command": "printf ready", "followups": [
        {"command": "printf checked", "stdout_contains": ["checked" if session.condition_target == "self" else "ready"], "stderr_empty": True}]}
    payload = {"model": service["model"], "messages": [{"role": "user", "content":
        "Synthetic interface check only. Call exec_in_sandbox once with exactly these arguments: " + json.dumps(arguments)}],
        "tools": session.tools([EXEC_TOOL]), "tool_choice": "auto", "parallel_tool_calls": False,
        "max_tokens": 512, "temperature": 0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
    server, ledger = native.start_proxy(root, args.upstream, args.port, prior_usage, verified_preinference=True)
    activation = None
    try:
        request = Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=330) as response:
            value = json.load(response)
        calls = value["choices"][0]["message"].get("tool_calls", [])
        if (value.get("model") != service["model"] or len(calls) != 1
                or calls[0]["function"]["name"] != "exec_in_sandbox"
                or json.loads(calls[0]["function"]["arguments"]) != arguments
                or session.parse(calls[0]) != arguments["followups"]):
            raise ValueError("model does not preserve the prospective typed continuation interface")
        if args.activation_check:
            activation = activation_check(args, root, protocol, service)
    except Exception as error:
        native.write(root / "FAILURE.json", {"type": type(error).__name__, "error": str(error),
                     "at_utc": native.now(), "usage": ledger.snapshot()})
        raise
    finally:
        server.shutdown()
        server.server_close()
        native.write(root / "PROVIDER_FINAL.json", ledger.snapshot())
    usage = ledger.snapshot()
    if not usage["complete"] or usage["preinference_rejections"] != (prior_usage or {}).get("preinference_rejections", 0):
        raise ValueError("typed interface admission usage must be returned and complete")
    native.write(root / "PASS.json", {"status": "PASS", "at_utc": native.now(),
        "identity": "synthetic tool-interface operability only", "benchmark_calls": 0,
        "environment_actions": activation["environment_actions"] if activation else 0,
        "activation": activation, "provider_usage": usage, "service_receipt_sha256": native.sha256(args.service_receipt),
        **prior_evidence,
        "raw_artifacts": [{"path": str(target), "sha256": native.sha256(target)}
                          for target in sorted((root / "provider").glob("call-*/*.json"))]})
    print(json.dumps({"status": "PASS", "provider_usage": usage, "benchmark_calls": 0}), flush=True)


def prepare(args):
    from harbor.models.trial.config import TrialConfig

    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    parent = args.parent_run.resolve()
    root = args.root.resolve()
    endpoint = urlsplit(args.upstream)
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or endpoint.path or endpoint.query:
        raise ValueError("study inference must use its owned loopback transport")
    parent_binding = json.loads((parent / "actor-binding.json").read_text())
    parent_execution = json.loads((parent / "EXECUTION.json").read_text())
    if native.sha256(parent / "actor-binding.json") != parent_execution["actor_binding_sha256"]:
        raise ValueError("parent actor binding changed")
    if native.sha256(parent / "TRIAL_PLAN.json") != parent_execution["trial_plan_sha256"]:
        raise ValueError("parent trial inventory changed")
    if parent_binding["model"]["revision"] != protocol["model_revision"]:
        raise ValueError("study must use the exact parent model revision")
    manifest = json.loads((args.preparation / "frozen/manifest.json").read_text())
    if native.sha256(args.preparation / "frozen/manifest.json") != parent_execution["official_manifest_sha256"]:
        raise ValueError("official source manifest differs from parent")
    parent_plan = json.loads((parent / "TRIAL_PLAN.json").read_text())
    rows = build_plan(parent_plan, protocol)
    first = json.loads(Path(rows[0]["source_config_path"]).read_text())
    task_source = Path(first["task"]["path"]).parent
    for entry in manifest["files"]:
        if native.sha256(task_source / entry["path"]) != entry["sha256"]:
            raise ValueError("official task source changed: " + entry["path"])
    images = json.loads((parent / "IMAGE_VERIFICATION.json").read_text())
    if images.get("official_images_present") != 89 or images.get("full89_manifest_sha256") != parent_execution["official_manifest_sha256"]:
        raise ValueError("official image inventory differs from the full89 source")
    for entry in images["images"]:
        observed = json.loads(subprocess.check_output(["docker", "image", "inspect", entry["image"]], text=True))[0]
        if observed["Id"] != entry["id"]:
            raise ValueError("official installed image changed: " + entry["image"])
    service = json.loads(args.service_receipt.read_text())
    if (service.get("status") != "READY" or service.get("revision") != protocol["model_revision"]
            or service.get("preinference_validation_source_sha256") != native.PREINFERENCE_RUNTIME_SOURCES):
        raise ValueError("independent model service lacks the pinned identity or rejection contract")
    admission = json.loads(args.admission_receipt.read_text())
    if (admission.get("status") != "PASS" or admission.get("benchmark_calls") != 0
            or admission.get("service_receipt_sha256") != native.sha256(args.service_receipt)):
        raise ValueError("prospective interface admission is missing or uses a different service")
    if protocol.get("interface_policy") == "explicit_plan_v2" and not (admission.get("activation") or {}).get("executed_followups"):
        raise ValueError("explicit plan requires actual synthetic continuation execution")
    if args.prior_study and not admission.get("prior_study_root"):
        previous_evidence = prior_stage_evidence(args.prior_study.resolve())
        fresh_usage = admission["provider_usage"]
        admission = {**admission, **previous_evidence, "this_stage_admission_usage": fresh_usage,
                     "provider_usage": merge_stage_usage(previous_evidence["prior_study_usage"], fresh_usage)}
    if admission.get("prior_admission_root"):
        previous_check = Path(admission["prior_admission_root"])
        if (native.sha256(previous_check / "PROVIDER_FINAL.json") != admission["prior_admission_final_sha256"]
                or native.sha256(previous_check / "FAILURE.json") != admission["prior_admission_failure_sha256"]
                or json.loads((previous_check / "PROVIDER_FINAL.json").read_text()) != admission["prior_admission_usage"]):
            raise ValueError("failed admission evidence changed")
    native.UsageLedger(admission["provider_usage"])
    for entry in admission["raw_artifacts"]:
        if native.sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("provider admission evidence changed")
    if admission.get("prior_study_root"):
        previous = Path(admission["prior_study_root"])
        if (native.sha256(previous / "PROVIDER_FINAL.json") != admission["prior_study_provider_final_sha256"]
                or native.sha256(previous / "RESULT.json") != admission["prior_study_result_sha256"]):
            raise ValueError("earlier prototype evidence changed")
    native.validate_preinference_contract(parent / "PROVIDER_REJECTION_CONTRACT.json")
    native.verify_dependency_cache_receipt(parent / "DEPENDENCY_CACHE_READY.json")
    native.verify_apt_cache_receipt(parent / "APT_CACHE_READY.json")
    root.mkdir(parents=True, exist_ok=False)
    model = deepcopy(parent_binding["model"])
    model.update(chat_completions_url=f"http://127.0.0.1:{args.port}/v1/chat/completions",
                 models_url=args.upstream + "/v1/models")
    actors = compose_actors(parent_binding, protocol)
    binding = root / "actor-binding.json"
    native.write(binding, {"model": model, "actors": actors,
        "identity": "Jev-inspired typed continuation ablations; no Jev API, no RLCD reproduction",
        "parent_binding_sha256": native.sha256(parent / "actor-binding.json")})
    metadata = {entry["task_id"]: entry for entry in manifest["tasks"]}
    configs = []
    for row in rows:
        original = Path(row["source_config_path"])
        if native.sha256(original) != row["source_config_sha256"]:
            raise ValueError("source configuration changed")
        value = native.trial_config(row, root=root, source=task_source, binding=binding,
            binding_sha=native.sha256(binding), task_metadata=metadata[row["task_id"]],
            verifier_https_proxy=parent_execution["verifier_https_proxy"],
            verifier_artifact_cache=parent_execution["verifier_artifact_cache"],
            verifier_apt_cache=parent_execution["verifier_apt_cache"])
        config = TrialConfig.model_validate(value)
        target = root / "trial-configs" / row["arm"] / f"{row['task_id']}__seed-{row['repeat_seed']}.json"
        native.write(target, config.model_dump(mode="json"))
        configs.append({**row, "config_path": str(target), "config_sha256": native.sha256(target)})
    native.write(root / "TRIAL_PLAN.json", configs)
    native.write(root / "PROTOCOL.json", protocol)
    native.write(root / "PRESERVED_RESULTS.json", {"rows": [], "source_artifacts": []})
    for name in ("DEPENDENCY_CACHE_READY.json", "APT_CACHE_READY.json", "PROVIDER_REJECTION_CONTRACT.json",
                 "IMAGE_VERIFICATION.json", "PACKAGES.json"):
        shutil.copy2(parent / name, root / name)
    shutil.copy2(args.service_receipt, root / "SERVICE_READY.json")
    native.write(root / "INTERFACE_ADMISSION.json", admission)
    sources = ("scripts/run_terminal_bench_jev.py", "scripts/terminal_jev.py", "scripts/run_terminal_bench_native.py",
               "scripts/harbor_terminal_agent.py", "scripts/run_terminal_bench_qwen35_4b_direct.py",
               "scripts/terminal_solpi.py", "phiagent/harness/observation_pack.py")
    allocation = protocol["allocation"]
    maximum_seconds = allocation["maximum_seconds"]
    if admission.get("study_deadline_utc"):
        maximum_seconds = min(maximum_seconds, int((datetime.fromisoformat(admission["study_deadline_utc"])
                              - datetime.now(timezone.utc)).total_seconds()))
    if maximum_seconds <= 0:
        raise ValueError("original study time reservation is exhausted")
    native.write(root / "EXECUTION.json", {"status": "FROZEN_BEFORE_EXECUTION", "created_at_utc": native.now(),
        "hostname": socket.gethostname(), "argv": sys.argv, "seed_schedule": protocol["repeat_seeds"],
        "upstream": args.upstream, "proxy_port": args.port, "concurrency": allocation["concurrency"],
        "maximum_requests": allocation["maximum_requests"],
        "maximum_provider_tokens": allocation["maximum_provider_tokens"],
        "maximum_seconds": maximum_seconds, "planned_trials": len(configs), "per_arm": 445,
        "study_deadline_utc": admission.get("study_deadline_utc"),
        "prior_prototype_root": admission.get("prior_study_root"), "prior_prototype_scores_pooled": False,
        "preserved_trials": 0, "remaining_trials": len(configs), "prior_provider_usage": admission["provider_usage"],
        "protocol_sha256": native.sha256(root / "PROTOCOL.json"),
        "interface_admission_sha256": native.sha256(root / "INTERFACE_ADMISSION.json"),
        "trial_plan_sha256": native.sha256(root / "TRIAL_PLAN.json"), "actor_binding_sha256": native.sha256(binding),
        "preserved_results_sha256": native.sha256(root / "PRESERVED_RESULTS.json"),
        "dependency_cache_receipt_sha256": native.sha256(root / "DEPENDENCY_CACHE_READY.json"),
        "apt_cache_receipt_sha256": native.sha256(root / "APT_CACHE_READY.json"),
        "provider_rejection_contract_sha256": native.sha256(root / "PROVIDER_REJECTION_CONTRACT.json"),
        "service_receipt_sha256": native.sha256(root / "SERVICE_READY.json"),
        "official_manifest_sha256": parent_execution["official_manifest_sha256"],
        "source_files_sha256": {name: native.sha256(ROOT / name) for name in sources},
        "continuous": True, "verifier_apt_cache": parent_execution["verifier_apt_cache"],
        "verifier_source_or_metric_modified": False, "task_timeouts_and_resource_overrides": None,
        "retries": 0, "new_claim_admitted": False, "parent_scores_reused": False,
        "task_exposure": protocol["task_exposure"], "independent_task_count": 89,
        "native_evaluator": "Harbor Trial", "native_metric": "harbor.metrics.mean.Mean"})
    native.write(root / "PREPARATION.json", {"status": "FROZEN_INDEPENDENT_MATCHED_STUDY", "trials": len(configs),
        "per_arm": 445, "arms": list(ARMS), "parent_campaign_untouched": str(parent),
        "parent_scores_reused": False, "root": str(root), "protocol_sha256": native.sha256(root / "PROTOCOL.json")})
    print(json.dumps({"status": "FROZEN_INDEPENDENT_MATCHED_STUDY", "trials": len(configs), "arms": list(ARMS)}), flush=True)


def transition(args):
    import fcntl

    base = args.root.resolve().parent
    previous = args.prior_study.resolve()
    with (base / "TRANSITION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base / "CONTROLLER_OWNER.json").exists() or (args.root / "STARTED.json").exists():
            raise FileExistsError("successor controller already exists")
        owner = json.loads((previous.parent / "CONTROLLER_OWNER.json").read_text())
        descriptor = None
        if not (previous / "PROVIDER_FINAL.json").exists():
            try:
                descriptor = os.pidfd_open(owner["pid"])
            except ProcessLookupError:
                pass
        if descriptor is not None:
            try:
                observed = Path(f"/proc/{owner['pid']}/cmdline").read_bytes().split(b"\0")
                if [part.decode() for part in observed if part] != owner["command"]:
                    raise ValueError("predecessor PID identity changed")
                execution = json.loads((previous / "EXECUTION.json").read_text())
                started = datetime.fromisoformat(json.loads((previous / "STARTED.json").read_text())["at_utc"])
                deadline = datetime.fromisoformat(execution["study_deadline_utc"]) if execution.get("study_deadline_utc") else (
                    started + timedelta(seconds=execution["maximum_seconds"]))
                remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                native.write(base / "TRANSITION_WAITING.json", {"status": "WAITING_FOR_PRIOR_STAGE_FINAL",
                    "at_utc": native.now(), "pid": os.getpid(), "prior_controller_pid": owner["pid"],
                    "study_deadline_utc": deadline.isoformat(), "original_campaign_untouched": True})
                if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                    raise TimeoutError("original study allocation expired before stage transition")
            finally:
                os.close(descriptor)
        prepare(args)
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--root", str(args.root.resolve())]
        execution = json.loads((args.root / "EXECUTION.json").read_text())
        with (base / "controller.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        native.write(base / "CONTROLLER_OWNER.json", {"pid": process.pid, "command": command,
            "at_utc": native.now(), "planned_trials": execution["planned_trials"],
            "previous_costs_inherited": execution["prior_provider_usage"],
            "study_deadline_utc": execution["study_deadline_utc"], "previous_scores_pooled": False})
        native.write(base / "TRANSITION_COMPLETE.json", {"status": "SUCCESSOR_STARTED", "pid": process.pid,
            "at_utc": native.now(), "original_study_budget_preserved": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("admit", "prepare", "run", "transition"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--service-receipt", type=Path)
    parser.add_argument("--admission-receipt", type=Path)
    parser.add_argument("--prior-study", type=Path)
    parser.add_argument("--prior-admission", type=Path)
    parser.add_argument("--activation-check", action="store_true")
    parser.add_argument("--upstream", default="http://127.0.0.1:18998")
    parser.add_argument("--port", type=int, default=19000)
    args = parser.parse_args()
    if socket.gethostname() != "harnessmm-api-worker":
        parser.error("all benchmark preparation and evaluation must run on A800-4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.mode == "admit":
        admit(args)
    elif args.mode == "prepare":
        prepare(args)
    elif args.mode == "transition":
        try:
            transition(args)
        except Exception as error:
            native.write(args.root.resolve().parent / "TRANSITION_FAILURE.json",
                         {"at_utc": native.now(), "type": type(error).__name__, "error": str(error)})
            raise
    else:
        protocol = json.loads((args.root / "PROTOCOL.json").read_text())
        validate_protocol(protocol)
        execution = json.loads((args.root / "EXECUTION.json").read_text())
        if (native.sha256(args.root / "PROTOCOL.json") != execution["protocol_sha256"]
                or native.sha256(args.root / "INTERFACE_ADMISSION.json") != execution["interface_admission_sha256"]):
            raise ValueError("study protocol or admission changed after freeze")
        if not 0 < execution["maximum_seconds"] <= protocol["allocation"]["maximum_seconds"]:
            raise ValueError("study time allocation exceeds the original reservation")
        for key in ("maximum_requests", "maximum_provider_tokens", "concurrency"):
            if execution[key] != protocol["allocation"][key]:
                raise ValueError("study allocation changed after freeze")
        native.MAX_REQUESTS = execution["maximum_requests"]
        native.MAX_TOTAL_TOKENS = execution["maximum_provider_tokens"]
        asyncio.run(native.run_campaign(args.root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
