#!/usr/bin/env python3
"""Prepare, run and score one bounded remote compact-harness training pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

from phiagent.learning.qa_context_evidence import select_context
from scripts.run_chisel_component_quality import (
    ProtocolError,
    atomic_json,
    build_messages,
    http_transport,
    neutral_padding,
    official_scorer,
    prompt_length,
    remote_guard,
    score_rows,
    strict_response,
    valid_usage,
)
from scripts.prepare_chisel_component_quality_data import canonical_hash, write_json


BASELINES = ("no_skill", "skillopt", "autosaddler", "delete")
CANDIDATES = ("compact_target", "compact_constraint")
EXPECTED_PACKAGES = {
    "vllm": "0.25.1+cu129", "torch": "2.11.0+cu129",
    "transformers": "5.16.1", "tokenizers": "0.23.1",
}
SCORER_RUNTIME_MODE = "cached-path-local-only-v1"
SCORER_RUNTIME_DECLARATION = "cached-path-local-only-v1; official metric functions unchanged"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_cpu_client() -> None:
    remote_guard()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("The remote HTTP/tokenizer/scorer client requires CUDA_VISIBLE_DEVICES=''")


def signed(value: dict) -> dict:
    return {**value, "canonical_sha256": canonical_hash(value)}


def scorer_runtime_mode(config: dict) -> str:
    if config["scoring"]["runtime_compatibility"] != SCORER_RUNTIME_DECLARATION:
        raise ProtocolError("Unsupported frozen scorer compatibility declaration")
    return SCORER_RUNTIME_MODE


def mixture_helpers(config: dict):
    if config.get("schema_version") in {
        "phiagent-qa-profile-compression/v1", "phiagent-qa-profile-compression/v2",
    }:
        from scripts import qa_profile_compression_protocol

        qa_profile_compression_protocol.validate_config(config)
        return qa_profile_compression_protocol
    if config.get("schema_version") == "phiagent-qa-lossless-format/v1":
        from scripts import qa_lossless_format_protocol

        qa_lossless_format_protocol.validate_config(config)
        return qa_lossless_format_protocol
    if (config.get("schema_version") == "phiagent-qa-lossless-routing/v1"
            or config.get("routing_control") is not None):
        from scripts import qa_lossless_routing_protocol

        qa_lossless_routing_protocol.validate_config(config)
        return qa_lossless_routing_protocol
    if config.get("schema_version") == "phiagent-qa-aa-calibration/v1":
        from scripts import qa_aa_calibration_protocol

        qa_aa_calibration_protocol.validate_config(config)
        return qa_aa_calibration_protocol
    if config.get("schema_version") == "phiagent-qa-chisel-preservation/v1":
        from scripts import qa_chisel_preservation_protocol

        qa_chisel_preservation_protocol.validate_config(config)
        return qa_chisel_preservation_protocol
    if (config.get("schema_version") in {
                "phiagent-qa-capability-cost/v2", "phiagent-qa-capability-core/v3",
            }
            or config.get("capability_routing") is not None):
        from scripts import qa_capability_cost_protocol

        qa_capability_cost_protocol.validate_config(config)
        return qa_capability_cost_protocol
    if ("mixture_router" not in config
            and config.get("schema_version") != "phiagent-qa-harness-mixture/v1"):
        return None
    from scripts import qa_harness_mixture_protocol

    qa_harness_mixture_protocol.validate_config(config)
    return qa_harness_mixture_protocol


def checked(path: Path) -> dict:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("canonical_sha256") != canonical_hash({
        key: item for key, item in value.items() if key != "canonical_sha256"
    }):
        raise ProtocolError(f"Self-hashed JSON differs: {path}")
    return value


def verify_sources(source: Path, origin: dict) -> None:
    for relative, expected in origin["source_sha256"].items():
        if sha(source / relative) != expected:
            raise ProtocolError(f"Frozen source/prompt changed: {relative}")


def validate_model(config: dict, spec: dict) -> None:
    model = config["model"]
    for key, expected in (
        ("model_repository", model["repository"]), ("revision", model["revision"]),
        ("dtype", model["dtype"]), ("context_length", model["context_length"]),
    ):
        if spec[key] != expected:
            raise ProtocolError(f"Model specification differs: {key}")
    for package, expected in EXPECTED_PACKAGES.items():
        if importlib.metadata.version(package) != expected:
            raise ProtocolError(f"Runtime package differs: {package}")
    if not spec["tokenizer_files_sha256"]:
        raise ProtocolError("Tokenizer file bindings are required")
    tokenizer_root = Path(spec["tokenizer_path"]).resolve()
    for relative, expected in spec["tokenizer_files_sha256"].items():
        path = tokenizer_root / relative
        if not path.resolve().is_relative_to(tokenizer_root) or sha(path) != expected:
            raise ProtocolError(f"Tokenizer file binding differs: {relative}")


def pilot_advancement(
    methods: dict[str, dict], *, candidate_arms: tuple[str, ...] = CANDIDATES,
    comparison_arms: tuple[str, ...] | None = None,
) -> dict:
    if (not candidate_arms or len(set(candidate_arms)) != len(candidate_arms)
            or set(candidate_arms) & set(BASELINES)):
        raise ValueError("distinct non-baseline candidate arms are required")
    if set(methods) != set(BASELINES + candidate_arms):
        raise ValueError("all declared baseline and candidate arms are required")
    references = BASELINES if comparison_arms is None else comparison_arms
    if (not references or len(set(references)) != len(references)
            or not set(BASELINES) <= set(references) <= set(methods)):
        raise ValueError("comparison references must include all fixed baselines and declared arms only")
    counts = {method["tasks"] for method in methods.values()}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("identical positive task counts are required")
    for method in methods.values():
        if type(method["tasks"]) is not int:
            raise ValueError("integer task counts are required")
        if type(method["total_tokens"]) is not int or method["total_tokens"] < 0:
            raise ValueError("complete exact token usage is required")
        if any(
            isinstance(method[metric], bool)
            or not isinstance(method[metric], (int, float))
            or not math.isfinite(method[metric])
            or not 0 <= method[metric] <= 100
            for metric in ("exact_match", "f1")
        ):
            raise ValueError("finite official metrics are required")
    candidates = {}
    for candidate in candidate_arms:
        comparators = tuple(arm for arm in references if arm != candidate)
        quality = all(
            methods[candidate][metric] > methods[baseline][metric] + 1e-12
            for baseline in comparators for metric in ("exact_match", "f1")
        )
        cheaper = all(
            methods[candidate]["total_tokens"] < methods[baseline]["total_tokens"]
            for baseline in comparators
        )
        candidates[candidate] = {
            "all_quality_metrics_strictly_above_every_baseline": quality,
            "tokens_strictly_below_every_baseline": cheaper,
            "eligible_for_larger_training_study": quality and cheaper,
            "paper_admitted": False,
            "retrieval_contribution_established": False,
        }
    return candidates


def controlled_context(
    tokenizer: Any, item: dict, config: dict, arm: str, cache: dict | None = None,
) -> tuple[dict, dict | None]:
    settings = config.get("context_selection")
    if settings is None:
        return item, None
    if (config["schema_version"] != "phiagent-qa-context-evidence/v1"
            or set(settings["methods"]) != set(config["arms"])
            or any(settings["methods"][baseline] != "full" for baseline in BASELINES)
            or config.get("rule_compression") or config.get("length_matched_controls")):
        raise ProtocolError("Context edits require their own protocol and unchanged full-context baselines")
    method = settings["methods"][arm]
    key = canonical_hash([item["question"], item["context"], settings, method])
    selection = cache.get(key) if cache is not None else None
    if selection is None:
        selection = select_context(
            item["question"], item["context"], method=method,
            count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            token_budget=settings["token_budget"],
            maximum_unit_tokens=settings["maximum_unit_tokens"],
            mmr_relevance_weight=settings["mmr_relevance_weight"],
            coverage_weight=settings["coverage_weight"],
            cooccurrence_weight=settings["cooccurrence_weight"],
        )
        if cache is not None:
            cache[key] = selection
    return {**item, "context": selection.context}, selection.receipt


def controlled_system_prompt(tokenizer: Any, item: dict, prompts: dict, config: dict, arm: str) -> tuple[str, dict | None]:
    controls = config.get("length_matched_controls", {})
    if arm not in controls:
        return prompts[arm], None
    control = controls[arm]
    base, reference = control["base_arm"], control["reference_arm"]
    if (base == reference or arm in (base, reference) or base not in prompts
            or reference not in prompts or prompts[arm] != prompts[base]):
        raise ProtocolError("Length-matched control must bind distinct frozen base and reference prompts")
    padding = neutral_padding(
        tokenizer, item, {"delete": prompts[base], "modify": prompts[reference]},
        {"neutral_control": config["neutral_padding"]},
    )
    return padding["system"], {
        "base_arm": base, "reference_arm": reference,
        "padding_units": padding["padding_units"], "prompt_tokens": padding["prompt_tokens"],
        "status": "EXACT_FROZEN_TOKENIZER_MATCH",
    }


def prepare(args: argparse.Namespace) -> None:
    require_cpu_client()
    root = args.experiment
    if root.exists():
        raise FileExistsError("A preparation must use a new experiment directory")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "PREPARATION_STARTED.json", {
        "at_utc": now(), "hostname": socket.gethostname(), "command": sys.argv,
        "CUDA_VISIBLE_DEVICES": "", "model_calls": 0,
    })
    source = Path(__file__).resolve().parents[1]
    origin = read_json(source / "origin.json")
    verify_sources(source, origin)
    config_path = getattr(args, "config", None) or source / "configs/compact_harness_training_pilot_v1.json"
    config = read_json(config_path)
    mixture = mixture_helpers(config)
    candidate_arms = tuple(config.get("candidate_arms", CANDIDATES))
    if mixture is None and (not candidate_arms or len(set(candidate_arms)) != len(candidate_arms)
            or set(candidate_arms) & set(BASELINES)
            or config["arms"] != list(BASELINES + candidate_arms)):
        raise ProtocolError("The fixed arm inventory changed")
    data = checked(args.data / "manifest.json")
    for name in ("public.jsonl", "selection-freeze.json", "mrqa_official_eval.py", "config.json"):
        if sha(args.data / name) != data["files_sha256"][name]:
            raise ProtocolError(f"Prepared input differs: {name}")
    if data["config_sha256"] != sha(config_path):
        raise ProtocolError("Data selection and execution configurations differ")
    items = [json.loads(line) for line in (args.data / "public.jsonl").read_text().splitlines()]
    ids = {item["id"] for item in items}
    if len(items) != data["task_count"] or len(ids) != len(items):
        raise ProtocolError("Duplicate or incomplete public task cohort")
    if any(set(item) != {"id", "group", "question", "context"} for item in items):
        raise ProtocolError("Public input schema differs")
    spec = read_json(args.model_spec)
    validate_model(config, spec)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_path"], local_files_only=True, trust_remote_code=False,
    )
    scorer_data = {"scorer": {
        "path": str((args.data / "mrqa_official_eval.py").resolve()),
        "revision": config["scoring"]["revision"],
        "sha256": config["scoring"]["source_sha256"],
    }}
    runtime_mode = scorer_runtime_mode(config)
    scorer = official_scorer(scorer_data, runtime=runtime_mode)
    for prediction, expected in (("answer", 100.0), ("wrong", 0.0)):
        evaluated = scorer.evaluate(
            {"synthetic": ["answer"]}, {"synthetic": prediction}, skip_no_answer=False
        )
        if evaluated != {"exact_match": expected, "f1": expected}:
            raise ProtocolError("Official scorer failed its remote synthetic interface check")
    prompt_arms = config["arms"] if mixture is None else mixture.prompt_artifact_names(config)
    prompts = {arm: (source / "prompts" / f"{arm}.system.txt").read_text() for arm in prompt_arms}
    planned = []
    context_started = time.monotonic()
    context_cache: dict = {}
    if mixture is not None:
        planned, summary = mixture.prepare_cells(tokenizer, items, prompts, config, source)
        write_json(root / "MIXTURE_PREPARATION.json", {
            **summary, "at_utc": now(), "preparation_seconds": time.monotonic() - context_started,
        })
    else:
        for item in items:
            for arm in config["arms"]:
                model_item, context_receipt = controlled_context(tokenizer, item, config, arm, context_cache)
                system, length_match = controlled_system_prompt(tokenizer, model_item, prompts, config, arm)
                messages = build_messages(system, model_item)
                count = prompt_length(tokenizer, messages)
                if length_match is not None and length_match["prompt_tokens"] != count:
                    raise ProtocolError("Neutral control token equality changed during request preparation")
                if count + config["model"]["max_output_tokens"] > config["model"]["context_length"]:
                    raise ProtocolError("Full untruncated input exceeds the attested model context")
                cell = canonical_hash([item["id"], arm, config["model"]["request_seed"]])
                planned.append({
                    "cell": cell, "task": item["id"], "group": item["group"], "arm": arm,
                    "expected_prompt_tokens": count,
                    **({"length_match": length_match} if length_match is not None else {}),
                    **({"context_evidence": context_receipt} if context_receipt is not None else {}),
                    "body": {
                        "model": config["model"]["repository"], "messages": messages,
                        "temperature": 0, "max_tokens": config["model"]["max_output_tokens"],
                        "seed": config["model"]["request_seed"],
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                })
    planned.sort(key=lambda item: item["cell"])
    if config.get("context_selection"):
        write_json(root / "CONTEXT_PREPARATION.json", {
            "status": "SOURCE_BOUND_CONTEXTS_FROZEN_BEFORE_MODEL_CALLS",
            "at_utc": now(), "preparation_seconds": time.monotonic() - context_started,
            "unique_context_transforms": len(context_cache),
            "methods": {
                arm: {
                    "changed_tasks": sum(not cell["context_evidence"]["identity_transform"]
                                         for cell in planned if cell["arm"] == arm),
                    "source_context_tokens": sum(cell["context_evidence"]["source_context_tokens"]
                                                 for cell in planned if cell["arm"] == arm),
                    "selected_context_tokens": sum(cell["context_evidence"]["rendered_context_tokens"]
                                                   for cell in planned if cell["arm"] == arm),
                }
                for arm in config["arms"]
            },
            "selection_uses_answers_or_model_responses": False,
            "provenance_is_not_factual_truth_or_answer_sufficiency": True,
        })
    if config.get("rule_compression"):
        prompt_counts = {(cell["task"], cell["arm"]): cell["expected_prompt_tokens"] for cell in planned}
        compression = {}
        for arm, minimum in config["rule_compression"]["minimum_prompt_saving_each_input"].items():
            savings = {task: prompt_counts[task, "delete"] - prompt_counts[task, arm] for task in ids}
            if min(savings.values()) < minimum:
                raise ProtocolError("A declared compression does not reduce the actual full-input prompt length")
            compression[arm] = {"minimum_prompt_tokens_saved": min(savings.values()),
                                "total_prompt_tokens_saved": sum(savings.values()), "tasks": len(savings)}
        write_json(root / "COMPRESSION_PREFLIGHT.json", {
            "status": "FULL_INPUT_PROMPTS_SHORTER_BEFORE_MODEL_CALLS", "at_utc": now(),
            "source_map_sha256": sha(source / "DELETE_COMPRESSION_MAP.json"), "candidates": compression,
            "no_semantic_or_behavioral_fidelity_claim": True,
        })
    worst_case = sum(
        item["expected_prompt_tokens"] + (
            mixture.output_cap(item, config) if mixture is not None else config["model"]["max_output_tokens"]
        ) for item in planned
    )
    synthetic_reserve = (
        config["budget"]["maximum_synthetic_request_attempts"] * config["model"]["context_length"]
    )
    if (
        len(planned) > config["budget"]["maximum_real_request_attempts"]
        or worst_case + synthetic_reserve > config["budget"]["maximum_total_provider_tokens"]
    ):
        raise ProtocolError("Full paired matrix exceeds the frozen request/token budget")
    workers = config["budget"].get("execution_workers", 1)
    if (type(workers) is not int
            or not 1 <= workers <= config["budget"]["maximum_inflight_requests"]):
        raise ProtocolError("Execution workers exceed the frozen inflight-request budget")
    write_json(root / "requests.json", planned)
    write_json(root / "config.json", config)
    write_json(root / "model-spec.json", spec)
    write_json(root / "origin.json", origin)
    write_json(root / "PREPARED.json", signed({
        "status": "PREPARED_NO_MODEL_CALLS", "at_utc": now(),
        "hostname": socket.gethostname(), "python": sys.version,
        "source": str(source), "data": str(args.data.resolve()),
        "data_manifest_sha256": sha(args.data / "manifest.json"),
        "requests_sha256": sha(root / "requests.json"),
        "config_sha256": sha(root / "config.json"),
        "model_spec_sha256": sha(root / "model-spec.json"),
        "tasks": len(items), "groups": len({item["group"] for item in items}),
        "requests": len(planned), "worst_case_real_tokens": worst_case,
        "reserved_synthetic_tokens": synthetic_reserve,
        "actual_maximum_inflight_requests": workers,
        "execution_order": (
            "Frozen protocol-specific client batches, independent of outcomes; see EXECUTION_SCHEDULE.json"
            if mixture is not None and hasattr(mixture, "execution_batches")
            else "Fixed cell-hash ordering, independent of outcomes"
        ),
        "scorer_import_passed": True, "synthetic_scorer_checks": 2,
        "scorer_runtime_mode": runtime_mode,
        "scorer_runtime_declaration": config["scoring"]["runtime_compatibility"],
        "benchmark_examples_scored_during_preparation": 0,
        "training_answers_parsed": False, "evaluation_answers_or_responses_opened": False,
        "model_calls": 0, "command": sys.argv,
    }))
    print((root / "PREPARED.json").read_text())


def live_identity(binding: dict, spec: dict) -> None:
    if binding["hostname"] != socket.gethostname():
        raise ProtocolError("Service binding belongs to another host")
    if binding["model_spec_sha256"] != canonical_hash(spec):
        raise ProtocolError("Live service is not bound to the frozen model specification")
    parsed = urlparse(binding["endpoint"])
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path.rstrip("/") != "/v1":
        raise ProtocolError("Only the attested loopback endpoint is allowed")
    identity = binding["server_identity"]
    proc = Path(f"/proc/{identity['pid']}")
    fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
    if (
        int(fields[19]) != identity["start_time_ticks"]
        or sha(proc / "cmdline") != identity["cmdline_sha256"]
        or proc.stat().st_uid != identity["uid"]
    ):
        raise ProtocolError("Service process identity changed")
    environment = dict(
        part.split(b"=", 1) for part in (proc / "environ").read_bytes().split(b"\0") if b"=" in part
    )
    if environment.get(b"CUDA_VISIBLE_DEVICES") != str(binding["physical_gpu_index"]).encode():
        raise ProtocolError("Service CUDA visibility differs from its physical GPU binding")


def perform_request(cell: dict, root: Path, config: dict, binding: dict, deadline: float) -> dict:
    outcome = {"usage": None, "record": None, "failure": None}
    directory = root / "calls" / cell["cell"]
    try:
        mixture = mixture_helpers(config)
        cap = mixture.output_cap(cell, config) if mixture is not None else config["model"]["max_output_tokens"]
        directory.mkdir(parents=True, exist_ok=False)
        if cell.get("execution_kind", "model") != "model":
            if (mixture is None or not hasattr(mixture, "perform_direct_cell")
                    or cell["execution_kind"] != mixture.DIRECT_EXECUTION_KIND):
                raise ProtocolError("Undeclared non-model execution cannot bypass the provider")
            outcome["usage"] = dict(mixture.ZERO_USAGE)
            return mixture.perform_direct_cell(cell, directory, config)
        body = json.dumps(cell["body"], ensure_ascii=False, separators=(",", ":")).encode()
        (directory / "request.body").write_bytes(body)
        write_json(directory / "intent.json", {
            "cell": cell["cell"], "task": cell["task"], "arm": cell["arm"],
            "at_utc": now(), "expected_prompt_tokens": cell["expected_prompt_tokens"],
            "body_sha256": hashlib.sha256(body).hexdigest(),
            **({"max_output_tokens": cap} if mixture is not None else {}),
        })
        started = time.monotonic()
        request_started_ns = time.monotonic_ns()
        request_started_utc = now()
        try:
            status, headers, response_body = http_transport(
                binding["endpoint"], body, min(120, deadline - time.monotonic()),
            )
        except HTTPError as error:
            status, response_body = error.code, error.read()
            headers = dict(error.headers) if error.headers is not None else None
        (directory / "response.body").write_bytes(response_body)
        write_json(directory / "transport.json", {
            "status": status, "headers": headers,
            "elapsed_seconds": time.monotonic() - started, "at_utc": now(),
        })
        if mixture is not None and getattr(mixture, "RECORD_CLIENT_TIMING", False):
            write_json(directory / "client-timing.json", {
                "started_at_utc": request_started_utc, "completed_at_utc": now(),
                "started_monotonic_ns": request_started_ns,
                "completed_monotonic_ns": time.monotonic_ns(),
                "scope": "Client HTTP envelope, not server microbatch or GPU kernel timing",
            })
        raw = json.loads(response_body)
        outcome["usage"] = valid_usage(raw)
        if status != 200:
            raise ProtocolError(f"HTTP {status}; no retry is allowed")
        response = strict_response(
            raw, model=config["model"]["repository"],
            expected_prompt_tokens=cell["expected_prompt_tokens"],
        )
        if response["usage"]["completion_tokens"] > cap:
            raise ProtocolError("Provider exceeded the frozen output-token cap")
        record = {
            "task": cell["task"], "group": cell["group"], "arm": cell["arm"],
            "raw": raw, "usage": response["usage"],
            "finish_reason": response["finish_reason"], "empty_answer": response["empty_answer"],
            "response_sha256": hashlib.sha256(response_body).hexdigest(),
        }
        write_json(directory / "record.json", record)
        outcome["record"] = record
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        outcome["failure"] = {"type": type(error).__name__, "message": str(error), "at_utc": now()}
        if directory.is_dir() and not (directory / "request-failure.json").exists():
            write_json(directory / "request-failure.json", outcome["failure"])
    return outcome


def run(args: argparse.Namespace) -> None:
    require_cpu_client()
    root = args.experiment
    prepared = checked(root / "PREPARED.json")
    config, spec = read_json(root / "config.json"), read_json(root / "model-spec.json")
    mixture = mixture_helpers(config)
    for name, expected in (
        ("config.json", prepared["config_sha256"]),
        ("model-spec.json", prepared["model_spec_sha256"]),
        ("requests.json", prepared["requests_sha256"]),
    ):
        if sha(root / name) != expected:
            raise ProtocolError(f"Frozen experiment input changed: {name}")
    if args.authorize_protocol != prepared["config_sha256"]:
        raise ProtocolError("Explicit frozen-protocol hash authorization is required")
    data_root = Path(prepared["data"])
    if sha(data_root / "manifest.json") != prepared["data_manifest_sha256"]:
        raise ProtocolError("Frozen data manifest changed before generation")
    validate_model(config, spec)
    verify_sources(Path(prepared["source"]), read_json(root / "origin.json"))
    binding = read_json(args.service_binding)
    live_identity(binding, spec)
    if mixture is not None and hasattr(mixture, "attest_runtime"):
        write_json(root / "RUN_RUNTIME_ATTESTATION.json", mixture.attest_runtime(binding, config))
    synthetic_calls, synthetic_tokens = binding["synthetic_request_attempts"], binding["synthetic_total_tokens"]
    maximum_synthetic = config["budget"]["maximum_synthetic_request_attempts"] if mixture is not None else 6
    if (
        type(synthetic_calls) is not int or not 0 <= synthetic_calls <= maximum_synthetic
        or type(synthetic_tokens) is not int
        or not 0 <= synthetic_tokens <= prepared["reserved_synthetic_tokens"]
    ):
        raise ProtocolError("Synthetic service-check usage is missing or outside its reservation")
    deadline_utc = datetime.fromisoformat(binding["allocation_deadline_utc"])
    remaining = (deadline_utc - datetime.now(timezone.utc)).total_seconds()
    if not 0 < remaining <= config["budget"]["maximum_allocation_seconds"] + 1:
        raise ProtocolError("Allocation deadline is invalid or exhausted")
    deadline = time.monotonic() + remaining
    write_json(root / "RUN_STARTED.json", {
        "at_utc": now(), "pid": os.getpid(), "command": sys.argv,
        "hostname": socket.gethostname(), "service_binding": binding,
        "source_preparation_sha256": sha(root / "PREPARED.json"),
        "client_CUDA_VISIBLE_DEVICES": "",
        "backend": "Separately leased remote GPU model service; not CPU-only model execution",
    })
    stopping = False

    def stop_requested(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    opener = build_opener(ProxyHandler({}))
    with opener.open(binding["endpoint"] + "/models", timeout=15) as response:
        advertised = json.load(response)["data"]
    if (
        len(advertised) != 1 or advertised[0]["id"] != config["model"]["repository"]
        or advertised[0]["max_model_len"] != config["model"]["context_length"]
    ):
        raise ProtocolError("Live advertised model/context differs")
    planned = read_json(root / "requests.json")
    progress = {
        "status": "RUNNING", "attempted": 0, "completed": 0, "pending_http": 0,
        "known_real_tokens": 0, "unknown_cost_requests": 0,
        "synthetic_requests": synthetic_calls, "synthetic_tokens": synthetic_tokens,
    }
    direct_kind = getattr(mixture, "DIRECT_EXECUTION_KIND", None)
    if direct_kind is not None:
        progress.update({
            "counter_unit": "logical_cells",
            "provider_requests_attempted": 0, "provider_requests_completed": 0,
            "deterministic_outputs": 0,
        })
    records = {arm: {} for arm in config["arms"]}
    failure = None
    workers = prepared.get("actual_maximum_inflight_requests", 1)
    if type(workers) is not int or workers < 1:
        raise ProtocolError("Invalid frozen request concurrency")
    scheduled = mixture is not None and hasattr(mixture, "execution_batches")
    batches = (
        mixture.execution_batches(planned, config) if scheduled else
        [planned[offset:offset + workers] for offset in range(0, len(planned), workers)]
    )
    if scheduled:
        write_json(root / "EXECUTION_SCHEDULE.json", {
            "schema_version": "phiagent-explicit-client-batches/v1",
            "batches": [[cell["cell"] for cell in batch] for batch in batches],
            "maximum_workers": workers, "barrier_after_every_batch": True,
            "server_microbatch_layout_controlled": False,
        })
        (root / "dispatches").mkdir(exist_ok=False)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch_index, batch in enumerate(batches):
                if stopping or time.monotonic() + 120 >= deadline:
                    raise ProtocolError("Stop requested or insufficient time to safely finish a request")
                live_identity(binding, spec)
                if scheduled:
                    progress["phase"] = batch[0]["arm"]
                    dispatch = {
                        "batch_index": batch_index, "phase": batch[0]["arm"],
                        "cells": [cell["cell"] for cell in batch],
                        "started_at_utc": now(), "started_monotonic_ns": time.monotonic_ns(),
                    }
                    write_json(root / "dispatches" / f"{batch_index:04d}.started.json", dispatch)
                model_batch = sum(cell.get("execution_kind", "model") == "model" for cell in batch)
                progress["attempted"] += len(batch)
                progress["pending_http"] = model_batch
                if direct_kind is not None:
                    progress["provider_requests_attempted"] += model_batch
                atomic_json(root / "HEARTBEAT.json", {**progress, "at_utc": now()})
                futures = [
                    executor.submit(perform_request, cell, root, config, binding, deadline)
                    for cell in batch
                ]
                batch_failure = None
                for cell, future in zip(batch, futures, strict=True):
                    outcome = future.result()
                    if outcome["usage"] is None:
                        progress["unknown_cost_requests"] += 1
                    else:
                        progress["known_real_tokens"] += outcome["usage"]["total_tokens"]
                    record = outcome["record"]
                    if record is not None:
                        records[record["arm"]][record["task"]] = record
                        progress["completed"] += 1
                        if direct_kind is not None:
                            counter = (
                                "deterministic_outputs"
                                if cell.get("execution_kind") == direct_kind
                                else "provider_requests_completed"
                            )
                            progress[counter] += 1
                    if outcome["failure"] is not None:
                        batch_failure = batch_failure or outcome["failure"]
                    if cell.get("execution_kind", "model") == "model":
                        progress["pending_http"] -= 1
                    atomic_json(root / "HEARTBEAT.json", {**progress, "at_utc": now()})
                print(json.dumps({**progress, "at_utc": now()}), flush=True)
                if scheduled:
                    write_json(root / "dispatches" / f"{batch_index:04d}.completed.json", {
                        **dispatch, "completed_at_utc": now(),
                        "completed_monotonic_ns": time.monotonic_ns(),
                        "batch_failure": batch_failure,
                    })
                if batch_failure:
                    raise ProtocolError(batch_failure["message"])
                if progress["known_real_tokens"] + synthetic_tokens > config["budget"]["maximum_total_provider_tokens"]:
                    raise ProtocolError("Measured provider usage exceeded the frozen budget")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        failure = {"type": type(error).__name__, "message": str(error), "at_utc": now()}
        write_json(root / "FAILURE.json", {**failure, "progress": progress})
        raise
    finally:
        complete = progress["completed"] == len(planned) and failure is None
        write_json(root / ("GENERATION_DRAINED.json" if complete else "CLIENT_STOPPED.json"), signed({
            **progress, "at_utc": now(), "pending_http": 0,
            "generation_complete": complete,
            "server_completion_verified": complete,
            "failure": failure,
            "meaning": (
                "All requests completed with valid usage; the owner may retire its verified PIDs."
                if complete else
                "Client stopped. Request server-side graceful cancellation; unknown costs remain unknown."
            ),
        }))
    if sha(data_root / "manifest.json") != prepared["data_manifest_sha256"]:
        raise ProtocolError("Frozen data manifest changed before scoring")
    data = checked(data_root / "manifest.json")
    if sha(data_root / "answers.json") != data["files_sha256"]["answers.json"]:
        raise ProtocolError("Training answer binding changed")
    answers = read_json(data_root / "answers.json")
    scorer = official_scorer({"scorer": {
        "path": str(data_root / "mrqa_official_eval.py"),
        "revision": config["scoring"]["revision"], "sha256": config["scoring"]["source_sha256"],
    }}, runtime=scorer_runtime_mode(config))
    methods = {}
    mixture_scores = {}
    for arm in config["arms"]:
        task_scores = (
            mixture.score_records(records[arm], answers, scorer)
            if mixture is not None and hasattr(mixture, "score_records")
            else score_rows(records[arm], answers, scorer)
        )
        predictions = {qid: item["prediction"] for qid, item in task_scores.items()}
        aggregate = scorer.evaluate(answers, predictions, skip_no_answer=False)
        methods[arm] = {
            "tasks": len(task_scores),
            "exact_match": float(aggregate["exact_match"]), "f1": float(aggregate["f1"]),
            "total_tokens": sum(item["tokens"] for item in task_scores.values()),
            "prompt_tokens": sum(item["prompt_tokens"] for item in task_scores.values()),
            "completion_tokens": sum(item["completion_tokens"] for item in task_scores.values()),
            "empty_answers": sum(item["empty_answer"] for item in task_scores.values()),
            "length_stops": sum(item["finish_reason"] == "length" for item in task_scores.values()),
        }
        write_json(root / f"{arm}.scores.json", task_scores)
        write_json(root / f"{arm}.predictions.json", predictions)
        if mixture is not None:
            mixture_scores[arm] = task_scores
    extra = {}
    if mixture is not None:
        groups = {cell["task"]: cell["group"] for cell in planned}
        bounds = mixture.diagnostic_bounds(mixture_scores, groups, config)
        gate = mixture.target_gate(methods, config, bounds)
        write_json(root / "MIXTURE_BOUNDS.json", signed({
            "bounds": bounds, "paper_admitted": False, "statistical_guarantee_established": False,
            "scope": "Prospective training-source numerical bounds, not complete public confirmation",
        }))
        advancement = {
            arm: {
                "eligible_for_larger_training_study": (
                    arm == config["primary_candidate"] and gate["point_target_passed"] is True
                    and gate["statistical_target_passed"] is True
                ),
                "paper_admitted": False,
            } for arm in config["candidate_arms"]
        }
        extra = {
            "target_gate": gate, "mixture_bounds_sha256": sha(root / "MIXTURE_BOUNDS.json"),
            "cost_accounting": {
                "selected_path_tokens": {arm: row["total_tokens"] for arm, row in methods.items()},
                "real_matrix_acquisition_tokens": progress["known_real_tokens"],
                "synthetic_acquisition_tokens": synthetic_tokens,
                "all_in_acquisition_tokens": progress["known_real_tokens"] + synthetic_tokens,
                "router_provider_tokens": 0, "selected_path_is_all_in_acquisition_cost": False,
            },
        }
        if hasattr(mixture, "result_diagnostics"):
            extra.update(mixture.result_diagnostics(planned, mixture_scores, config))
    else:
        advancement = pilot_advancement(
            methods, candidate_arms=tuple(config.get("candidate_arms", CANDIDATES)),
            comparison_arms=(tuple(config["comparison_arms"]) if "comparison_arms" in config else None),
        )
    result = signed({
        "schema_version": "phiagent-qa-harness-mixture-result/v1" if mixture is not None else "phiagent-compact-harness-training-result/v1",
        "status": getattr(mixture, "RESULT_STATUS", "COMPLETE_TRAINING_PILOT_NOT_PAPER_ADMITTED"),
        "completed_at_utc": now(),
        "methods": methods, "advancement": advancement, **extra,
        "calls": progress, "data_manifest_sha256": sha(data_root / "manifest.json"),
        "config_sha256": sha(root / "config.json"), "service_binding_sha256": sha(args.service_binding),
        "official_aggregate_evaluations": len(config["arms"]),
        "per_task_metric_evaluations": 2 * prepared["tasks"] * len(config["arms"]),
        "all_inputs_full_context": all(
            (cell.get("mixture_receipt", {}).get("identity_context_transform", True)
             if mixture is not None else cell.get("context_evidence", {}).get("identity_transform", True))
            for cell in planned
        ),
        "evaluation_answers_or_responses_opened": False,
        "paper_admitted": False, "retrieval_contribution_established": False,
        "scope": getattr(
            mixture, "RESULT_SCOPE",
            "Disjoint official MRQA training-source groups only; not full public evaluation or fresh confirmation",
        ),
    })
    write_json(root / "RESULT.json", result)
    atomic_json(root / "HEARTBEAT.json", {**progress, "status": result["status"], "at_utc": now()})
    print(json.dumps({"status": result["status"], "methods": methods, "advancement": result["advancement"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preparing = sub.add_parser("prepare")
    preparing.add_argument("--config", type=Path)
    preparing.add_argument("--data", type=Path, required=True)
    preparing.add_argument("--model-spec", type=Path, required=True)
    preparing.add_argument("--experiment", type=Path, required=True)
    running = sub.add_parser("run")
    running.add_argument("--experiment", type=Path, required=True)
    running.add_argument("--service-binding", type=Path, required=True)
    running.add_argument("--authorize-protocol", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.experiment.exists():
            parser.error("use a new experiment directory")
        try:
            prepare(args)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            if args.experiment.is_dir():
                if type(error).__name__ == "LightReadinessError":
                    from scripts.qa_chisel_preservation_protocol import LightReadinessError

                    if isinstance(error, LightReadinessError):
                        write_json(args.experiment / "LIGHT_READINESS.json", {
                            **error.readiness_receipt,
                            "evaluator_calls": 2, "synthetic_scorer_checks": 2,
                            "call_count_scope": "whole preparation; no benchmark outcomes",
                        })
                write_json(args.experiment / "PREPARATION_FAILURE.json", {
                    "type": type(error).__name__, "message": str(error), "at_utc": now(),
                    "model_calls": 0,
                })
            raise
    else:
        if not (args.experiment / "PREPARED.json").is_file():
            parser.error("a complete source/data/tokenizer preparation is required")
        if (args.experiment / "RUN_LOCK.json").exists():
            parser.error("an experiment can execute only once; use a new directory after failure")
        write_json(args.experiment / "RUN_LOCK.json", {"pid": os.getpid(), "at_utc": now()})
        try:
            run(args)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            if not (args.experiment / "FAILURE.json").exists():
                write_json(args.experiment / "FAILURE.json", {
                    "type": type(error).__name__, "message": str(error), "at_utc": now(),
                })
            if not any((args.experiment / name).exists() for name in (
                "GENERATION_DRAINED.json", "CLIENT_STOPPED.json"
            )):
                write_json(args.experiment / "CLIENT_STOPPED.json", signed({
                    "at_utc": now(), "pending_http": 0, "generation_complete": False,
                    "server_completion_verified": False,
                    "meaning": "Early client failure; gracefully stop the owned service, not a successful drain.",
                }))
            raise


if __name__ == "__main__":
    main()
