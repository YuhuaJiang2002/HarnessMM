#!/usr/bin/env python3
"""Freeze and supervise a bounded remote development-only compact-contract study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener
from urllib.error import HTTPError

from phiagent.learning.compact_qa_contracts import (
    COMPACT_CONTRACTS, COMPACT_REFINEMENTS, COMPACT_SLOT_CONTRACTS,
    cost_source_contracts, delete_compression_contracts,
)
from scripts.run_compact_harness_training_pilot import (
    atomic_json, canonical_hash, checked, http_transport,
    live_identity, mixture_helpers, now, prompt_length, read_json, sha, signed, strict_response, valid_usage, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/compact_harness_refinement_v2.json")
OVERLAY_FILES = (
    CONFIG.as_posix(), "phiagent/learning/compact_qa_contracts.py",
    "scripts/run_compact_harness_training_pilot.py",
    "scripts/run_compact_harness_refinement.py",
    "tests/test_compact_qa_contracts.py", "tests/test_compact_harness_training_pilot.py",
    "phiagent/__init__.py", "scripts/__init__.py",
    "scripts/prepare_compact_harness_training_pilot.py",
    "scripts/prepare_chisel_component_quality_data.py",
    "scripts/verify_compact_harness_training_pilot.py",
    "phiagent/evaluation/chisel_component_quality.py",
    "scripts/compact_h200_runtime.py",
    "phiagent/learning/qa_context_evidence.py",
    "phiagent/learning/public_evidence_retention.py",
    "phiagent/learning/sparse_retrieval.py",
    "tests/test_qa_context_evidence.py",
    "tests/test_qa_context_campaign_contract.py",
)


def require_remote(config: dict) -> None:
    if (socket.gethostname() != config["resource"]["selected_hostname"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("This study requires its declared remote host and an explicitly CPU-only controller")


def amended_resources(config: dict, amendment: dict) -> dict:
    if set(amendment["changes"]) != {"resource", "budget"}:
        raise ValueError("A resource amendment may change only resource and budget fields")
    result = dict(config)
    result.update(amendment["changes"])
    result["resource_amendment"] = {key: value for key, value in amendment.items() if key != "changes"}
    if (result["budget"]["maximum_physical_gpus"] > config["budget"]["maximum_physical_gpus"]
            or result["budget"]["maximum_allocation_seconds"] + amendment["prior_allocation_debit_seconds"] > config["budget"]["maximum_allocation_seconds"]
            or result["budget"]["maximum_real_request_attempts"] > config["budget"]["maximum_real_request_attempts"]
            or result["budget"]["transport_retries"] != 0):
        raise ValueError("Resource recovery must stay within the original GPU/time/real-request budget")
    return result


def freeze_local(output: Path, config_path: Path = ROOT / CONFIG, amendment_path: Path | None = None) -> None:
    config_relative = config_path.resolve().relative_to(ROOT).as_posix()
    files = (config_relative,) + OVERLAY_FILES[1:]
    config = read_json(config_path)
    mixture = mixture_helpers(config)
    if mixture is not None:
        files += mixture.OVERLAY_FILES
    amendment = None
    if amendment_path is not None:
        amendment = read_json(amendment_path)
        if sha(config_path) != amendment["base_protocol_sha256"]:
            raise ValueError("Resource amendment requires the original frozen protocol hash")
        config = amended_resources(config, amendment)
    if mixture is not None and config["resource"].get("binding_status") != "BOUND_AFTER_FRESH_AUDIT":
        raise RuntimeError("Mixture resources require a fresh physical GPU/service/port audit before freeze")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == config_relative and amendment is not None:
            write_json(destination, config)
        else:
            shutil.copyfile(ROOT / relative, destination)
        hashes[relative] = sha(destination)
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
        "relevant_diff": ["git", "diff", "--binary", "--", *files],
    }
    freeze = {
        "at_utc": now(), "hostname": socket.gethostname(), "command": sys.argv,
        "files_sha256": hashes,
        "git": {name: subprocess.check_output(command, cwd=ROOT, text=True) for name, command in commands.items()},
        "no_benchmark_reads_or_calls": True,
    }
    write_json(output / "LOCAL_FREEZE.json", freeze)
    if config["data"].get("component_data_manifest_sha256"):
        write_json(output / "DATA_ORIGIN.json", {
            "source_sha256": hashes, "local_freeze": freeze,
            "component_data_manifest_sha256": config["data"]["component_data_manifest_sha256"],
        })
    print(json.dumps({"status": "LOCAL_SOURCE_FROZEN", "output": str(output), "files": len(hashes)}))


def copy_training_cohort(config_path: Path, destination: Path) -> None:
    config = read_json(config_path)
    source = Path(config["data"]["source_data"])
    reference = Path(config["data"]["reference_result"])
    if sha(reference) != config["data"]["reference_result_sha256"]:
        raise RuntimeError("The verified prior result binding changed")
    result = checked(reference)
    if sha(source / "manifest.json") != result["data_manifest_sha256"]:
        raise RuntimeError("The prior result no longer binds this exact training cohort")
    manifest = checked(source / "manifest.json")
    if manifest["question_group_count"] != config["data"]["question_groups"]:
        raise RuntimeError("The declared training group count changed")
    if not 0 < manifest["task_count"] <= config["data"]["maximum_tasks"]:
        raise RuntimeError("The complete training cohort exceeds the declared task count")
    for filename, expected in manifest["files_sha256"].items():
        if Path(filename).name != filename or sha(source / filename) != expected:
            raise RuntimeError(f"Training input binding changed: {filename}")
    destination.mkdir(parents=True, exist_ok=False)
    for filename in manifest["files_sha256"]:
        shutil.copyfile(config_path if filename == "config.json" else source / filename, destination / filename)
    os.chmod(destination / "answers.json", 0o600)
    manifest.pop("canonical_sha256")
    manifest.update({
        "status": "EXPOSED_TRAIN64_REUSED_FOR_DEVELOPMENT_ONLY",
        "hostname": socket.gethostname(), "completed_at_utc": now(),
        "config_sha256": sha(config_path), "prior_manifest_sha256": sha(source / "manifest.json"),
        "claim_boundary": "Exact previously exposed training cohort. Not new confirmation or a paper result.",
    })
    manifest["files_sha256"]["config.json"] = sha(destination / "config.json")
    write_json(destination / "manifest.json", signed(manifest))


def copy_prepared_training(config_path: Path, source: Path, destination: Path) -> None:
    config = read_json(config_path)
    manifest = checked(source / "manifest.json")
    if (manifest["config_sha256"] != sha(config_path)
            or manifest["source_manifest_sha256"] != config["data"]["component_data_manifest_sha256"]
            or manifest["question_group_count"] != config["data"]["question_groups"]
            or not 0 < manifest["task_count"] <= config["data"]["maximum_tasks"]):
        raise RuntimeError("Prepared training cohort/configuration binding differs")
    required = (
        "group_disjoint_from_component_training", "group_disjoint_from_component_calibration",
        "group_disjoint_from_full_official_development", "group_disjoint_from_prior_training",
    )
    if any(manifest["independence"].get(key) is not True for key in required):
        raise RuntimeError("Prepared cohort lacks required disjoint-group evidence")
    if (manifest["independence"]["evaluation_answers_opened"]
            or manifest["independence"]["evaluation_responses_opened"]):
        raise RuntimeError("Evaluation answers or responses were exposed during preparation")
    for filename, expected in manifest["files_sha256"].items():
        if Path(filename).name != filename or (source / filename).is_symlink() or sha(source / filename) != expected:
            raise RuntimeError(f"Prepared input changed: {filename}")
    selection = read_json(source / "selection-freeze.json")
    if selection["prior_training_exclusions"] != config["data"]["exclude_prior_training"]:
        raise RuntimeError("Prepared prior-training exclusions differ from the frozen protocol")
    shutil.copytree(source, destination)
    os.chmod(destination / "answers.json", 0o600)


def seal_remote(campaign: Path, config_path: Path = ROOT / CONFIG) -> None:
    config = read_json(config_path)
    require_remote(config)
    mixture = mixture_helpers(config)
    if campaign.resolve() != ROOT.parent or "compact-harness-refinement" not in campaign.parts:
        raise RuntimeError("Seal only a new private refinement source snapshot")
    if (campaign / "SOURCE_FROZEN.json").exists():
        raise RuntimeError("A campaign can be sealed only once")
    overlay = read_json(ROOT / "LOCAL_FREEZE.json")
    for relative, expected in overlay["files_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Local/remote source overlay differs: {relative}")
    origin_path = ROOT / "origin.json"
    old_origin_hash, origin = sha(origin_path), read_json(origin_path)
    for relative, expected in origin["source_sha256"].items():
        if relative not in overlay["files_sha256"] and sha(ROOT / relative) != expected:
            raise RuntimeError(f"An unrelated frozen source changed: {relative}")
    fixed_prompts = config["prompt_sha256"] if mixture is None else mixture.fixed_prompt_hashes(config)
    for arm, expected in fixed_prompts.items():
        if sha(ROOT / "prompts" / f"{arm}.system.txt") != expected:
            raise RuntimeError(f"A fixed baseline prompt changed: {arm}")
    derived = {**COMPACT_CONTRACTS, **COMPACT_REFINEMENTS, **COMPACT_SLOT_CONTRACTS}
    mixture_files = []
    if mixture is not None:
        from phiagent.learning.qa_harness_mixture import mixture_arm_prompt

        mixture_files = mixture.stage_incumbent(ROOT, config)
        derived.update({arm: mixture_arm_prompt() for arm in config["candidate_arms"]})
    if config.get("context_selection"):
        settings = config["context_selection"]
        if (config["schema_version"] != "phiagent-qa-context-evidence/v1"
                or set(settings["methods"]) != set(config["arms"])
                or settings["candidate_system_source"] != "delete"):
            raise RuntimeError("Context candidates must share the unchanged DELETE system prompt")
        parent_prompt = (ROOT / "prompts/delete.system.txt").read_text()
        derived.update({arm: parent_prompt for arm in config["candidate_arms"]})
    compression_files = []
    if config.get("cost_source_design") or config.get("rule_compression"):
        derived.update(cost_source_contracts(
            (ROOT / "prompts/no_skill.system.txt").read_text(),
            (ROOT / "prompts/delete.system.txt").read_text(),
        ))
        if config.get("rule_compression"):
            compressed, mapping = delete_compression_contracts((ROOT / "prompts/delete.system.txt").read_text())
            if mapping["source_clause_count"] != config["rule_compression"]["source_clause_count"]:
                raise RuntimeError("Frozen source clause inventory differs from compression protocol")
            derived.update(compressed)
            write_json(ROOT / "DELETE_COMPRESSION_MAP.json", mapping)
            compression_files.append("DELETE_COMPRESSION_MAP.json")
        for arm, control in config["length_matched_controls"].items():
            derived[arm] = derived[control["base_arm"]]
    for arm, prompt in derived.items():
        if arm not in config["arms"]:
            continue
        destination = ROOT / "prompts" / f"{arm}.system.txt"
        if destination.exists():
            if destination.read_text() != prompt:
                raise RuntimeError(f"An old candidate prompt changed: {arm}")
        else:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(prompt)
    relatives = set(origin["source_sha256"]) | set(overlay["files_sha256"])
    relatives.update(compression_files)
    relatives.update(mixture_files)
    prompt_arms = config["arms"] if mixture is None else mixture.prompt_artifact_names(config)
    relatives.update(f"prompts/{arm}.system.txt" for arm in prompt_arms)
    if mixture is not None:
        relatives.update(f"prompts/{arm}.system.txt" for arm in fixed_prompts)
    origin.update({
        "base_origin_sha256": old_origin_hash, "refinement_local_freeze": overlay,
        "source_sha256": {relative: sha(ROOT / relative) for relative in sorted(relatives)},
        "scope": "New private source snapshot; old six-arm results and protocol unchanged",
    })
    atomic_json(origin_path, origin)
    if mixture is not None and hasattr(mixture, "copy_prepared_data"):
        mixture.copy_prepared_data(config_path, campaign / "data-input", campaign / "data")
    elif config["data"].get("mode") == "disjoint_prepared_training":
        copy_prepared_training(config_path, campaign / "data-input", campaign / "data")
    else:
        copy_training_cohort(config_path, campaign / "data")
    write_json(campaign / "SOURCE_FROZEN.json", {
        "at_utc": now(), "hostname": socket.gethostname(), "command": sys.argv,
        "config_sha256": sha(config_path), "origin_sha256": sha(origin_path),
        "data_manifest_sha256": sha(campaign / "data/manifest.json"),
        "models_called": 0, "new_confirmation_answers_opened": False,
    })
    print("SOURCE_AND_TRAINING_COHORT_FROZEN_NO_MODEL_CALLS", flush=True)


def runtime_helpers(config: dict):
    if config["resource"].get("native_runtime"):
        from scripts import compact_h200_runtime

        runtime = compact_h200_runtime.H200Runtime(config)
        return runtime, runtime, compact_h200_runtime
    adapter = Path(config["resource"]["runtime_adapter"])
    seal = adapter / "execution-seal.json"
    if sha(seal) != config["resource"]["runtime_adapter_seal_sha256"]:
        raise RuntimeError("Pinned A800 runtime adapter seal changed")
    for filename, expected in read_json(seal)["files_sha256"].items():
        if sha(adapter / filename) != expected:
            raise RuntimeError(f"Pinned runtime helper changed: {filename}")
    sys.path.insert(0, str(adapter))
    return tuple(importlib.import_module(name) for name in (
        "a800_environment", "a800_attestation", "a800_lifecycle_v4",
    ))


def logged(command: list[str], environment: dict, destination: Path, timeout: float) -> None:
    write_json(destination.with_suffix(".command.json"), {
        "argv": command, "cwd": str(ROOT), "at_utc": now(), "timeout_seconds": timeout,
        "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
    })
    with destination.open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)


def admit(campaign: Path, config_path: Path = ROOT / CONFIG) -> None:
    config = read_json(config_path)
    require_remote(config)
    binding = read_json(campaign / "binding.unadmitted.json")
    spec = read_json(campaign / "experiment/model-spec.json")
    live_identity(binding, spec)
    from transformers import AutoTokenizer
    from scripts.run_searchqa_residual_meta_campaign import extract_answer

    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    messages = [{"role": "user", "content": "Return only the number 42 inside <answer>...</answer>."}]
    expected = prompt_length(tokenizer, messages)
    body = json.dumps({
        "model": config["model"]["repository"], "messages": messages,
        "temperature": 0, "max_tokens": 32, "seed": config["model"]["request_seed"],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    (campaign / "admission-request.body").write_bytes(body)
    write_json(campaign / "ADMISSION_INTENT.json", {"at_utc": now(), "synthetic_requests": 1})
    try:
        status, headers, response = http_transport(binding["endpoint"], body, 120)
    except HTTPError as error:
        status, response = error.code, error.read()
        headers = dict(error.headers) if error.headers is not None else None
    (campaign / "admission-response.body").write_bytes(response)
    write_json(campaign / "admission-transport.json", {"status": status, "headers": headers, "at_utc": now()})
    raw = json.loads(response)
    write_json(campaign / "ADMISSION_USAGE.json", {"at_utc": now(), "requests": 1, "usage": valid_usage(raw),
                                                "unknown_cost_requests": int(valid_usage(raw) is None)})
    if status != 200:
        raise RuntimeError(f"Synthetic admission returned HTTP {status}; preserve usage/unknown cost and stop")
    validated = strict_response(raw, model=config["model"]["repository"], expected_prompt_tokens=expected)
    if status != 200 or extract_answer(raw["choices"][0]["message"]["content"]) != "42":
        raise RuntimeError("Fresh synthetic plain-QA admission failed; no benchmark generation allowed")
    binding.update({
        "status": "READY_LIVE", "synthetic_request_attempts": 1,
        "synthetic_total_tokens": validated["usage"]["total_tokens"],
        "fresh_plain_QA_admission_required": False,
    })
    write_json(campaign / "binding.admitted.json", binding)


def gpu_snapshot(config: dict, destination: Path, *, require_idle: bool) -> set[int]:
    resource = config["resource"]
    commands = [
        ["nvidia-smi", "-i", str(resource["physical_gpu_index"]),
         "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "-i", str(resource["physical_gpu_index"]),
         "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
    ]
    outputs = [subprocess.check_output(command, text=True, timeout=30) for command in commands]
    write_json(destination, {"at_utc": now(), "hostname": socket.gethostname(), "commands": commands, "outputs": outputs})
    rows = list(csv.reader(io.StringIO(outputs[0]), skipinitialspace=True))
    if (len(rows) != 1 or int(rows[0][0]) != resource["physical_gpu_index"]
            or rows[0][1] != resource["gpu_uuid"] or resource.get("hardware_name", "A800") not in rows[0][2]):
        raise RuntimeError("The selected physical GPU identity changed")
    processes = {int(row[0]) for row in csv.reader(io.StringIO(outputs[1]), skipinitialspace=True) if row}
    if require_idle and (processes or float(rows[0][3]) >= 1024 or float(rows[0][5]) > 5):
        raise RuntimeError("Selected GPU is occupied; never share or retire another study's service")
    return processes


def descendants(lifecycle) -> list[dict]:
    pending, found = [os.getpid()], []
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except FileNotFoundError:
            continue
        for child in children:
            identity = lifecycle.identity(int(child))
            if identity is not None:
                pending.append(identity["pid"])
                if identity["state"] != "Z":
                    found.append(identity)
    return found


def eligible_refinements(result: dict, config: dict) -> list[str]:
    if mixture_helpers(config) is not None:
        gate = result["target_gate"]
        return [config["primary_candidate"]] if (
            gate["primary_baseline_status"] == "BOUND"
            and gate["point_target_passed"] is True
            and gate["statistical_target_passed"] is True
        ) else []
    eligible = [
        arm for arm in config["selection_eligible_arms"]
        if result["advancement"][arm]["eligible_for_larger_training_study"]
    ]
    retained = config.get("retained_method")
    if retained is not None:
        reference = result["methods"][retained]
        eligible = [
            arm for arm in eligible
            if all(result["methods"][arm][metric] >= reference[metric] for metric in ("exact_match", "f1"))
            and any(result["methods"][arm][metric] > reference[metric] + 1e-12 for metric in ("exact_match", "f1"))
        ]
    quality = config.get("quality_matched_cost")
    if quality is not None:
        eligible = [
            arm for arm in eligible
            if all(result["methods"][arm][metric] >= result["methods"][reference][metric]
                   for reference in quality["strong_quality_references"] for metric in quality["metrics"])
        ]
    return sorted(eligible, key=lambda arm: (
        -result["methods"][arm]["exact_match"], -result["methods"][arm]["f1"],
        result["methods"][arm]["total_tokens"], arm,
    ))


def run_campaign(campaign: Path, config_path: Path = ROOT / CONFIG) -> None:
    import ctypes
    import fcntl

    config = read_json(config_path)
    require_remote(config)
    freeze = read_json(campaign / "SOURCE_FROZEN.json")
    if freeze["config_sha256"] != sha(config_path) or freeze["origin_sha256"] != sha(ROOT / "origin.json"):
        raise RuntimeError("Source/configuration changed after freezing")
    write_json(campaign / "STARTED.json", {
        "at_utc": now(), "pid": os.getpid(), "hostname": socket.gethostname(), "command": sys.argv,
        "packages": {entry.metadata["Name"]: entry.version for entry in importlib.metadata.distributions() if entry.metadata["Name"]},
        "seed": config["model"]["request_seed"], "CUDA_VISIBLE_DEVICES": "",
    })
    environment_helper, attestation, lifecycle = runtime_helpers(config)
    service = Path(config["resource"]["service_root"])
    environment_helper.rpc_check(service)
    service.mkdir(parents=True, exist_ok=False)
    (service / "scratch").mkdir()
    (service / "caches").mkdir()
    cpu_environment = environment_helper.environment(service, cuda="")
    cpu_environment["PYTHONPATH"] = str(ROOT)
    native = bool(config["resource"].get("native_runtime"))
    if native:
        python = environment_helper.python
        model_spec_path = environment_helper.prepare(campaign, write_json)
    else:
        python = str(environment_helper.ASSETS / "runtime/bin/python")
        model_spec_path = config["resource"]["reference_model_spec"]
        write_json(campaign / "opencv-loader.json", attestation.verify_opencv_loader())
        with (campaign / "checkpoint-verification.log").open("x") as stream:
            subprocess.run(["sha256sum", "-c", "model-files.sha256"], cwd=environment_helper.ASSETS,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
    logged([
        python, "-m", "scripts.run_compact_harness_training_pilot", "prepare",
        "--config", str(config_path), "--data", str(campaign / "data"),
        "--model-spec", str(model_spec_path), "--experiment", str(campaign / "experiment"),
    ], cpu_environment, campaign / "preparation.log", 600)
    prepared = checked(campaign / "experiment/PREPARED.json")
    spec = read_json(campaign / "experiment/model-spec.json")
    gpu_environment = environment_helper.environment(service, cuda=str(config["resource"]["physical_gpu_index"]))
    if not native:
        helper_path = environment_helper.PROFILE / "launch_bound_profile.py"
        runtime_ready = read_json(Path(config["resource"]["runtime_adapter"]).parent / "files-ready.json")
        if sha(helper_path) != runtime_ready["profile_helper_sha256"]:
            raise RuntimeError("Pinned JIT preparation helper changed")
        helper_spec = importlib.util.spec_from_file_location("frozen_refinement_jit", helper_path)
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        profile = read_json(environment_helper.PROFILE / "alf32768.json")
        jit = {key: gpu_environment[key] for key in ("NVCC_PREPEND_FLAGS", "FLASHINFER_WORKSPACE_BASE")}
        helper.prepare_jit_environment(profile, service, jit)
        helper.preflight_jit_paths(profile, service, jit)
    port = config["resource"].get("port", 19544)
    endpoint = f"http://127.0.0.1:{port}/v1"
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Owned-process subreaper setup failed")
    leases, server, allocation_start = [], None, None
    shutdown = threading.Event()
    service_log = None

    def stop_requested(number: int, _frame: object) -> None:
        raise RuntimeError(f"Controller signal {number}; retire only this study's owned service")

    previous_handlers = {
        number: signal.signal(number, stop_requested)
        for number in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        lease_root = Path(config["resource"]["lease_root"])
        if not lease_root.is_dir():
            raise RuntimeError("Shared physical GPU lease namespace is unavailable")
        for name in (f"physical-gpu-{config['resource']['physical_gpu_index']}.lock",
                     f"physical-{config['resource']['gpu_uuid']}.lock"):
            descriptor = os.open(lease_root / name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            leases.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_snapshot(config, campaign / "GPU_SELECTION.json", require_idle=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        allocation_start = time.monotonic()
        deadline = allocation_start + config["budget"]["maximum_allocation_seconds"] - 90
        write_json(service / "LEASE.json", {
            "at_utc": now(), "hostname": socket.gethostname(), "pid": os.getpid(),
            "physical_gpu_index": config["resource"]["physical_gpu_index"], "gpu_uuid": config["resource"]["gpu_uuid"],
            "CUDA_VISIBLE_DEVICES": gpu_environment["CUDA_VISIBLE_DEVICES"],
            "budget": config["budget"], "config_sha256": sha(config_path),
        })
        command = environment_helper.command(service)
        write_json(service / "FROZEN_LAUNCH.json", {"command": command, "environment": gpu_environment, "model_spec": spec})
        service_log = (service / "service.log").open("x")
        server = subprocess.Popen(command, cwd=service, env=gpu_environment, stdin=subprocess.DEVNULL,
                                  stdout=service_log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(service / "SERVER_IDENTITY.json", lifecycle.identity(server.pid))
        print(json.dumps({"status": "MODEL_STARTING", "pid": server.pid, "gpu": config["resource"]["physical_gpu_index"], "at_utc": now()}), flush=True)
        opener = build_opener(ProxyHandler({}))
        while True:
            if server.poll() is not None or time.monotonic() - allocation_start > 900:
                raise RuntimeError("Owned model service exited or startup deadline expired")
            try:
                with opener.open(endpoint + "/models", timeout=5) as response:
                    models = json.load(response)["data"]
            except OSError:
                shutdown.wait(5)
                continue
            if (len(models) != 1 or models[0]["id"] != spec["model_repository"]
                    or models[0]["max_model_len"] != spec["context_length"]
                    or Path(models[0]["root"]).resolve() != Path(spec["checkpoint_path"]).resolve()):
                raise RuntimeError("The new service does not advertise the frozen model/context")
            break
        owned = descendants(lifecycle)
        gpu_processes = gpu_snapshot(config, campaign / "GPU_READY.json", require_idle=False)
        if not gpu_processes or not gpu_processes <= {entry["pid"] for entry in owned}:
            raise RuntimeError("The new service does not exclusively own its selected GPU")
        for process_id in gpu_processes | {server.pid}:
            environment = dict(part.split(b"=", 1) for part in Path(f"/proc/{process_id}/environ").read_bytes().split(b"\0") if b"=" in part)
            observed = {key: environment.get(key.encode(), b"").decode() for key in attestation.ENVIRONMENT_KEYS}
            attestation.validate_environment(gpu_environment, observed, model_worker=process_id != server.pid)
            maps = Path(f"/proc/{process_id}/maps").read_text()
            libraries = sorted({line.split()[-1] for line in maps.splitlines() if any(
                name in line for name in ("libcuda.so", "libcudart.so", "libnvrtc.so", "libcublas.so", "libcudnn.so")
            )})
            write_json(service / f"PROCESS_ATTESTATION_{process_id}.json", {"environment": observed, "libraries": libraries})
            attestation.validate_library_maps(libraries, model_worker=process_id in gpu_processes)
        binding = {
            **spec, "hostname": socket.gethostname(), "endpoint": endpoint,
            "model_spec_sha256": canonical_hash(spec), "physical_gpu_index": config["resource"]["physical_gpu_index"],
            "gpu_uuid": config["resource"]["gpu_uuid"], "server_identity": lifecycle.identity(server.pid),
            "allocation_deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=deadline - time.monotonic())).isoformat(),
            "synthetic_request_attempts": 0, "synthetic_total_tokens": 0, "service_root": str(service),
        }
        write_json(campaign / "binding.unadmitted.json", binding)
        logged([python, "-m", "scripts.run_compact_harness_refinement", "admit", "--campaign", str(campaign),
            "--config", str(config_path)],
               cpu_environment, campaign / "admission.log", 180)
        print(json.dumps({"status": "RUNNING_TRAINING_SCREEN", "tasks": prepared["tasks"],
                  "requests": prepared["requests"], "workers": prepared["actual_maximum_inflight_requests"], "at_utc": now()}), flush=True)
        logged([
            python, "-m", "scripts.run_compact_harness_training_pilot", "run", "--experiment", str(campaign / "experiment"),
            "--service-binding", str(campaign / "binding.admitted.json"), "--authorize-protocol", prepared["config_sha256"],
        ], cpu_environment, campaign / "generation.log", max(1, deadline - time.monotonic()))
        result = checked(campaign / "experiment/RESULT.json")
        eligible = eligible_refinements(result, config)
        protocol_helpers = mixture_helpers(config)
        write_json(campaign / "DEVELOPMENT_DECISION.json", {
            "at_utc": now(), "eligible_candidates": eligible, "selected_candidate": eligible[0] if eligible else None,
            "status": getattr(protocol_helpers, "DEVELOPMENT_STATUS", (
                "REQUIRES_NEW_FROZEN_LARGER_STUDY" if eligible else "NEGATIVE_SCREEN_STOP_NO_CONFIRMATION"
            )),
            "result_sha256": sha(campaign / "experiment/RESULT.json"),
            "retained_method": config.get("retained_method"),
            "retained_method_quality_nonregression_required": config.get("retained_method") is not None,
            "confirmation_opened": False, "statistical_guarantee_established": False, "paper_admitted": False,
            **({
                "primary_reference": config["primary_reference"],
                "primary_baseline_status": config["incumbent"]["status"],
                "target_gate": result["target_gate"], "automatic_official_scale_up_permitted": False,
            } if mixture_helpers(config) is not None else {}),
        })
        print(json.dumps({"status": "DEVELOPMENT_COMPLETE", "methods": result["methods"], "eligible": eligible}), flush=True)
    finally:
        began = time.monotonic()
        owned = descendants(lifecycle)
        write_json(service / "OWNED_PROCESSES_AT_TEARDOWN.json", owned)
        while descendants(lifecycle) and time.monotonic() - began < 65:
            for identity in descendants(lifecycle):
                lifecycle.send(identity, signal.SIGTERM if time.monotonic() - began < 30 else signal.SIGKILL)
            if server is not None:
                server.poll()
            shutdown.wait(1)
        remaining = descendants(lifecycle)
        if service_log is not None:
            service_log.close()
        for descriptor in leases:
            os.close(descriptor)
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        write_json(campaign / "RETIREMENT.json", {
            "at_utc": now(), "owned_live_processes": remaining, "leases_released": True,
            "allocation_seconds": time.monotonic() - allocation_start if allocation_start else 0,
            "status": "RETIRED" if not remaining else "BLOCKED_OWNED_PROCESS_REMAINS",
        })
        if remaining:
            raise RuntimeError("Owned model processes remain; do not declare successful teardown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze-local", "seal", "run", "admit"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG)
    parser.add_argument("--resource-amendment", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze-local":
        freeze_local(args.campaign, args.config, args.resource_amendment)
        return
    if args.resource_amendment is not None:
        parser.error("resource amendments may only be applied during local source freezing")
    config = read_json(args.config)
    require_remote(config)
    try:
        {"seal": seal_remote, "run": run_campaign, "admit": admit}[args.mode](args.campaign, args.config.resolve())
    except BaseException as error:
        destination = args.campaign / f"{args.mode.upper()}_FAILURE.json"
        if args.campaign.is_dir() and not destination.exists():
            write_json(destination, {"at_utc": now(), "hostname": socket.gethostname(), "command": sys.argv, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
