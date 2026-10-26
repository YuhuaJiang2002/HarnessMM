#!/usr/bin/env python3
"""Remote independent replay of AA selection, scheduling, outputs, and official metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import socket
import sys

from scripts import qa_aa_calibration_protocol as protocol
from scripts.run_compact_harness_training_pilot import (
    checked, now, official_scorer, read_json, require_cpu_client, scorer_runtime_mode,
    sha, signed, validate_model, verify_sources, write_json,
)
from scripts.verify_qa_harness_mixture import (
    audit_dependency_binding, configuration_binding, replay_raw_calls, rescore, verify_accounting,
)


def verify_client_schedule(root: Path, planned: list[dict], config: dict) -> dict:
    batches = protocol.execution_batches(planned, config)
    expected = {
        "schema_version": "phiagent-explicit-client-batches/v1",
        "batches": [[cell["cell"] for cell in batch] for batch in batches],
        "maximum_workers": 16, "barrier_after_every_batch": True,
        "server_microbatch_layout_controlled": False,
    }
    if read_json(root / "EXECUTION_SCHEDULE.json") != expected:
        raise RuntimeError("The actual client schedule differs from its frozen request plan")
    expected_files = {
        f"{index:04d}.{phase}.json" for index in range(len(batches)) for phase in ("started", "completed")
    }
    if {path.name for path in (root / "dispatches").iterdir()} != expected_files:
        raise RuntimeError("Client dispatch receipts are missing or extra")
    result = {arm: {"requests": 0, "batches": 0, "peak_client_http_overlap": 0, "latencies_seconds": []}
              for arm in protocol.ARMS}
    hashes, previous_end = {}, -1
    for index, batch in enumerate(batches):
        start_path = root / "dispatches" / f"{index:04d}.started.json"
        end_path = root / "dispatches" / f"{index:04d}.completed.json"
        start, end = read_json(start_path), read_json(end_path)
        if (
            start["batch_index"] != index or start["phase"] != batch[0]["arm"]
            or start["cells"] != [cell["cell"] for cell in batch]
            or any(end.get(key) != value for key, value in start.items())
            or end["batch_failure"] is not None
            or not previous_end <= start["started_monotonic_ns"] <= end["completed_monotonic_ns"]
        ):
            raise RuntimeError("Batch order, barrier, completion, or failure evidence differs")
        previous_end = end["completed_monotonic_ns"]
        events = []
        values = result[batch[0]["arm"]]
        values["batches"] += 1
        for cell in batch:
            path = root / "calls" / cell["cell"] / "client-timing.json"
            timing = read_json(path)
            begin, finish = timing["started_monotonic_ns"], timing["completed_monotonic_ns"]
            if (
                type(begin) is not int or type(finish) is not int
                or not start["started_monotonic_ns"] <= begin <= finish <= end["completed_monotonic_ns"]
                or timing["scope"] != "Client HTTP envelope, not server microbatch or GPU kernel timing"
            ):
                raise RuntimeError("Per-request timing escaped its actual dispatched client batch")
            values["latencies_seconds"].append((finish - begin) / 1e9)
            values["requests"] += 1
            events.extend(((begin, 1), (finish, -1)))
            hashes[str(path.relative_to(root))] = sha(path)
        overlap, peak = 0, 0
        for _, delta in sorted(events):
            overlap += delta
            peak = max(peak, overlap)
        if overlap != 0 or peak > len(batch):
            raise RuntimeError("The observed client overlap is impossible for this frozen batch")
        values["peak_client_http_overlap"] = max(values["peak_client_http_overlap"], peak)
        for path in (start_path, end_path):
            hashes[str(path.relative_to(root))] = sha(path)
    for values in result.values():
        latencies = sorted(values.pop("latencies_seconds"))
        values["latency_mean_seconds"] = math.fsum(latencies) / len(latencies)
        values["latency_p50_seconds"] = latencies[math.ceil(len(latencies) * .5) - 1]
        values["latency_p95_seconds"] = latencies[math.ceil(len(latencies) * .95) - 1]
    return {
        "status": "FROZEN_CLIENT_ORDER_BARRIERS_AND_HTTP_OVERLAP_VERIFIED",
        "phases": result,
        "parallel_http_observed_in_every_concurrent_phase": all(
            result[arm]["peak_client_http_overlap"] > 1
            for arm, _, concurrency, _ in protocol.PHASES if concurrency > 1
        ),
        "server_microbatch_layout_observed": False, "timing_files_sha256": hashes,
    }


def verify(root: Path, output: Path) -> dict:
    require_cpu_client()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(root / "config.json")
    protocol.validate_config(config)
    write_json(output / "EXECUTION.json", {
        "at_utc": now(), "hostname": socket.gethostname(), "python": sys.version,
        "command": sys.argv, "CUDA_VISIBLE_DEVICES": "", "physical_gpu_selected": None,
        "model_calls": 0, "seed": config["model"]["request_seed"],
        "packages": {entry.metadata["Name"]: entry.version for entry in importlib.metadata.distributions()
                     if entry.metadata["Name"]},
    })
    for path in (root / "FAILURE.json", root / "CLIENT_STOPPED.json", root.parent / "RUN_FAILURE.json"):
        if path.exists():
            raise RuntimeError("A partial or failed AA run cannot become a completed comparison")
    prepared = checked(root / "PREPARED.json")
    for name, key in (
        ("config.json", "config_sha256"), ("requests.json", "requests_sha256"),
        ("model-spec.json", "model_spec_sha256"),
    ):
        if sha(root / name) != prepared[key]:
            raise RuntimeError(f"AA frozen execution input changed: {name}")
    source, data_root = Path(prepared["source"]), Path(prepared["data"])
    origin = read_json(root / "origin.json")
    if read_json(source / "origin.json") != origin:
        raise RuntimeError("Source and execution origins differ")
    verify_sources(source, origin)
    frozen = read_json(root.parent / "SOURCE_FROZEN.json")
    if (
        frozen["origin_sha256"] != sha(source / "origin.json")
        or frozen["data_manifest_sha256"] != prepared["data_manifest_sha256"]
        or sha(data_root / "manifest.json") != prepared["data_manifest_sha256"]
        or frozen["models_called"] != 0 or frozen["new_confirmation_answers_opened"] is not False
    ):
        raise RuntimeError("AA pre-outcome source/data freeze differs")
    dependencies = set(protocol.OVERLAY_FILES) | {
        "scripts/run_compact_harness_training_pilot.py", "scripts/run_compact_harness_refinement.py",
        "scripts/compact_h200_runtime.py",
    }
    audit_source = Path(__file__).resolve().parents[1]
    dependency = audit_dependency_binding(root, audit_source, origin, dependencies, None)
    data = protocol.validate_prepared_data(config, data_root, frozen["config_sha256"])
    binding = configuration_binding(source, data_root, origin, frozen, prepared, data, config)
    parent = Path(config["data"]["parent_data"])
    if sha(parent / "manifest.json") != config["data"]["parent_manifest_sha256"]:
        raise RuntimeError("Opened parent manifest changed")
    parent_manifest = checked(parent / "manifest.json")
    if sha(parent / "public.jsonl") != parent_manifest["files_sha256"]["public.jsonl"]:
        raise RuntimeError("Opened parent public inputs changed")
    public_pool = [json.loads(line) for line in (parent / "public.jsonl").read_text().splitlines()]
    public = [json.loads(line) for line in (data_root / "public.jsonl").read_text().splitlines()]
    reconstructed = protocol.select_public_items(
        public_pool, count=config["data"]["question_groups"], salt=config["data"]["selection_salt"],
    )
    if public != reconstructed:
        raise RuntimeError("Label-blind whole-group selection does not replay")
    selection = read_json(data_root / "selection-freeze.json")
    groups = {item["id"]: item["group"] for item in public}
    if (
        len(groups) != len(public) or len(public) != prepared["tasks"]
        or set(groups) != set(selection["task_ids"])
        or set(groups.values()) != set(selection["groups"])
        or selection["accepted_answers_consulted_for_selection"] is not False
        or selection["prior_outcome_labels_consulted_for_selection"] is not False
        or selection["historical_outcomes_exposed"] is not True
        or selection["independent_quality_confirmation"] is not False
    ):
        raise RuntimeError("AA task/group identity or historical exposure was misreported")
    spec = read_json(root / "model-spec.json")
    validate_model(config, spec)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    planned = read_json(root / "requests.json")
    replay = protocol.replay_matrix(tokenizer, public, planned, source, config)
    preparation = read_json(root / "MIXTURE_PREPARATION.json")
    if any(preparation[key] != value for key, value in replay["summary"].items()):
        raise RuntimeError("AA preparation summary differs from independently reconstructed requests")
    schedule = verify_client_schedule(root, planned, config)
    runtime = read_json(root / "RUN_RUNTIME_ATTESTATION.json")
    if (
        runtime["batch_invariant_runtime_changed"] is not False
        or any(value["VLLM_BATCH_INVARIANT"] != "0" for value in runtime["processes"].values())
        or not runtime["worker_pids"] or str(runtime["server_pid"]) not in runtime["processes"]
    ):
        raise RuntimeError("AA model worker runtime flag was not independently attested")
    raw_records, raw_hashes = replay_raw_calls(root, planned, config)
    answers = read_json(data_root / "answers.json")
    if set(answers) != set(groups):
        raise RuntimeError("Official scoring must keep every paired task")
    scorer = official_scorer({"scorer": {
        "path": str(data_root / "mrqa_official_eval.py"), "revision": config["scoring"]["revision"],
        "sha256": config["scoring"]["source_sha256"],
    }}, runtime=scorer_runtime_mode(config))
    methods, scores, predictions = rescore(raw_records, answers, scorer, config)
    result = checked(root / "RESULT.json")
    if (
        result["methods"] != methods or result["status"] != protocol.RESULT_STATUS
        or result["scope"] != protocol.RESULT_SCOPE or result["advancement"] != {}
        or result["paper_admitted"] is not False or result["retrieval_contribution_established"] is not False
    ):
        raise RuntimeError("AA official metrics or diagnostic identity differ")
    for arm in protocol.ARMS:
        if read_json(root / f"{arm}.scores.json") != scores[arm] or read_json(root / f"{arm}.predictions.json") != predictions[arm]:
            raise RuntimeError("AA per-task output/official metric replay differs")
    bounds = protocol.diagnostic_bounds(scores, groups, config)
    gate = protocol.target_gate(methods, config, bounds)
    if (
        result["target_gate"] != gate or result["aa_repeatability"] != bounds["aa_repeatability"]
        or result["mixture_bounds_sha256"] != sha(root / "MIXTURE_BOUNDS.json")
        or checked(root / "MIXTURE_BOUNDS.json")["bounds"] != bounds
    ):
        raise RuntimeError("AA change counts or conclusion differ")
    decision = read_json(root.parent / "DEVELOPMENT_DECISION.json")
    if (
        decision["status"] != protocol.DEVELOPMENT_STATUS or decision["eligible_candidates"] != []
        or decision["selected_candidate"] is not None or decision["confirmation_opened"]
        or decision["target_gate"] != gate or decision["result_sha256"] != sha(root / "RESULT.json")
    ):
        raise RuntimeError("The AA diagnostic was promoted into a candidate experiment")
    accounting = verify_accounting(root, methods, planned, config, tokenizer)
    for name, value in (
        ("CONFIGURATION_BINDING.json", binding), ("AUDIT_DEPENDENCY_BINDING.json", dependency),
        ("REQUESTS_RECONSTRUCTED.json", replay), ("CLIENT_SCHEDULE.json", schedule),
        ("TOKEN_ACCOUNTING.json", accounting), ("REPEATABILITY.json", bounds["aa_repeatability"]),
    ):
        write_json(output / name, value)
    verified = signed({
        "schema_version": "phiagent-qa-aa-verification/v1",
        "status": "INDEPENDENT_AA_CALIBRATION_REPLAY_VERIFIED",
        "at_utc": now(), "hostname": socket.gethostname(),
        "result_sha256": sha(root / "RESULT.json"), "config_sha256": sha(root / "config.json"),
        "requests_sha256": sha(root / "requests.json"), "data_manifest_sha256": sha(data_root / "manifest.json"),
        "methods": methods, "repeatability": bounds["aa_repeatability"],
        "target_gate": gate, "accounting": accounting,
        "client_schedule_sha256": sha(output / "CLIENT_SCHEDULE.json"),
        "runtime_attestation_sha256": sha(root / "RUN_RUNTIME_ATTESTATION.json"),
        "raw_files_sha256": {**raw_hashes, **schedule["timing_files_sha256"]},
        "scores_sha256": {arm: sha(root / f"{arm}.scores.json") for arm in protocol.ARMS},
        "audit_dependencies": dependency, "configuration_binding": binding,
        "model_calls": 0, "paper_admitted": False, "automatic_followup_experiment": False,
    })
    write_json(output / "VERIFIED.json", verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("A new independent verification directory is required")
    try:
        value = verify(args.experiment, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if args.output.is_dir():
            write_json(args.output / "FAILURE.json", {"at_utc": now(), "type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps({"status": value["status"], "repeatability_status": value["repeatability"]["status"],
                      "requests": value["accounting"]["real_requests"]}))


if __name__ == "__main__":
    main()
