#!/usr/bin/env python3
"""Measure provider batch invariance on frozen opened prompts, never benchmark scores."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from phiagent.learning.stage_aware_adm import canonical_hash


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def remote_guard() -> None:
    if socket.gethostname() not in {"harnessmm-worker-1", "harnessmm-worker-2"}:
        raise RuntimeError(
            "real-prompt model fidelity experiments must run on an authorized server"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("the HTTP-only probe process requires explicit CUDA_VISIBLE_DEVICES=''")


def metadata() -> dict:
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "argv": sys.argv,
        "source_sha256": file_hash(Path(__file__)),
    }


def prepare(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary = load(args.source_study / "SUMMARY.json")
    selected = load(args.case_selection)["cases"]
    source_rows = [
        row
        for phase in summary["phases"].values()
        for row in phase["rows"]
        if row["method"] == "skillopt"
    ]
    rows = {row["id"]: row for row in source_rows}
    if len(rows) != len(source_rows):
        raise RuntimeError("source parent task identities are duplicated")
    requests = []
    for case in selected:
        source = rows[case["id"]]
        if not source["valid"] or not source["cost_complete"]:
            raise RuntimeError("source parent trajectory is not valid")
        for turn in sorted({0, min(3, case["prefix_steps"])}):
            path = Path(source["artifact_dir"]) / "calls" / f"{turn:03d}" / "request.json"
            request = load(path)
            body = request["body"]
            if (
                request["stage"] != "actor"
                or body["stream"] is not False
                or body["temperature"] != 0
                or "tools" in body
                or not body["messages"]
            ):
                raise RuntimeError("probe requires actual plain-chat parent request bodies")
            requests.append(
                {
                    "case_id": canonical_hash([case["id"], turn])[:20],
                    "task_id": case["id"],
                    "turn": turn,
                    "source_request": str(path),
                    "source_sha256": file_hash(path),
                    "body": body,
                }
            )
    if len({row["case_id"] for row in requests}) != len(requests):
        raise RuntimeError("duplicate fidelity input identities")
    write_new(root / "requests.json", requests)
    protocol = {
        "schema_version": "phiagent-model-fidelity-probe/v1",
        "status": "FROZEN_BEFORE_ANY_PROBE_GENERATION",
        "scope": "Opened real-prompt provider fidelity only; no tools executed, no benchmark/evaluator outcomes.",
        "source_study": str(args.source_study.resolve()),
        "source_summary_sha256": file_hash(args.source_study / "SUMMARY.json"),
        "actor": summary["model"],
        "common_context": 32768,
        "selection_sha256": file_hash(args.case_selection),
        "requests_sha256": file_hash(root / "requests.json"),
        "input_cases": len(requests),
        "conditions": [0, 1],
        "isolated_repeats": 2,
        "mixed_repeats": 2,
        "batch_width": 8,
        "original_sampling_and_output_caps_preserved": True,
        "gate": "All repeated outputs and finish reasons identical per input within a condition; finite test, not a universal determinism proof.",
        "benchmark_scores_computed": False,
        "sota_evidence": False,
        **metadata(),
    }
    protocol["canonical_sha256"] = canonical_hash(protocol)
    write_new(root / "protocol.json", protocol)
    print(json.dumps({"status": protocol["status"], "input_cases": len(requests)}))
    return 0


def response_identity(raw: dict, *, expected_model: str) -> dict:
    if raw.get("model") != expected_model:
        raise ValueError("model response identity mismatch")
    if not isinstance(raw.get("choices"), list) or len(raw["choices"]) != 1:
        raise ValueError("one actual model response is required")
    choice = raw["choices"][0]
    message = choice["message"]
    if not isinstance(message.get("content"), str):
        raise ValueError("response content is absent")
    usage = raw["usage"]
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in keys):
        raise ValueError("actual model token usage is required")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("model token usage is inconsistent")
    output = {"content": message["content"], "finish_reason": choice["finish_reason"]}
    return {"output_sha256": canonical_hash(output), "usage": {key: usage[key] for key in keys}}


def run(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    protocol = load(root / "protocol.json")
    unsigned = {key: value for key, value in protocol.items() if key != "canonical_sha256"}
    if protocol["canonical_sha256"] != canonical_hash(unsigned):
        raise RuntimeError("probe protocol changed")
    if file_hash(root / "requests.json") != protocol["requests_sha256"]:
        raise RuntimeError("frozen probe requests changed")
    if protocol["schema_version"] == "phiagent-model-fidelity-probe/v1":
        expected_batch, expected_sequences = args.condition, 16
    elif protocol["schema_version"] == "phiagent-model-fidelity-probe/v2":
        settings = protocol["condition_settings"][str(args.condition)]
        expected_batch = settings["batch_invariant"]
        expected_sequences = settings["max_num_seqs"]
        if args.condition == 0 and "reference_condition_0" in protocol:
            raise RuntimeError(
                "condition0 is already bound to immutable completed evidence; do not rerun"
            )
    else:
        raise RuntimeError("unknown provider fidelity protocol version")
    pool = load(args.pool)
    if pool["canonical_sha256"] != canonical_hash(
        {key: value for key, value in pool.items() if key != "canonical_sha256"}
    ):
        raise RuntimeError("model pool self-hash mismatch")
    if len(pool["endpoints"]) != 1:
        raise RuntimeError("each probe condition needs its own single verified replica")
    endpoint = pool["endpoints"][0]
    evidence_path = args.pool.with_suffix(".evidence.json")
    evidence = load(evidence_path)
    runtime = evidence["runtime_by_gpu"][str(endpoint["physical_gpu_index"])]
    if (
        runtime["batch_invariant_declared"] != str(expected_batch)
        or runtime["observed_environment"].get("VLLM_BATCH_INVARIANT") != str(expected_batch)
        or runtime["max_model_len"] != protocol["common_context"]
        or pool["repository"] != protocol["actor"]["repository"]
        or pool["revision"] != protocol["actor"]["revision"]
    ):
        raise RuntimeError("probe condition differs from captured live provider flag")
    if protocol["schema_version"] == "phiagent-model-fidelity-probe/v2":
        if runtime["observed_arguments"].get("--max-num-seqs") != str(expected_sequences):
            raise RuntimeError("probe concurrency condition differs from the live process")
    for path, digest in evidence["source_files_sha256"].items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError("captured provider provenance changed before the probe")
    output = root / f"condition-{args.condition}"
    output.mkdir(exist_ok=False)
    write_new(
        output / "run-config.json",
        {
            **metadata(),
            "pool": pool,
            "pool_file_sha256": file_hash(args.pool),
            "pool_evidence_sha256": file_hash(evidence_path),
            "condition": args.condition,
            "runtime": runtime,
        },
    )
    cases = load(root / "requests.json")

    def execute(case: dict, layout: str, repetition: int) -> dict:
        target = output / layout / str(repetition) / case["case_id"]
        target.mkdir(parents=True, exist_ok=False)
        payload = dict(case["body"])
        payload["model"] = pool["request_model_id"]
        write_new(target / "request.json", payload)
        started = time.monotonic()
        request = Request(
            endpoint["base_url"] + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                raw = json.load(response)
            write_new(target / "response.json", raw)
            checked = response_identity(raw, expected_model=pool["expected_response_model_id"])
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as error:
            detail = {
                "type": type(error).__name__,
                "message": str(error),
                "raw_response_saved": (target / "response.json").is_file(),
                "accounting_status": "UNVALIDATED_DO_NOT_ASSUME_ZERO",
            }
            if isinstance(error, HTTPError):
                detail["http_body"] = error.read().decode("utf-8", errors="replace")
            write_new(target / "failure.json", detail)
            raise
        result = {
            **checked,
            "case_id": case["case_id"],
            "layout": layout,
            "repetition": repetition,
            "elapsed_seconds": time.monotonic() - started,
            "request_sha256": canonical_hash(payload),
            "response_file_sha256": file_hash(target / "response.json"),
        }
        write_new(target / "result.json", result)
        return result

    results = []
    for repetition in range(protocol["isolated_repeats"]):
        for case in cases:
            results.append(execute(case, "isolated", repetition))
    # Concurrent HTTP arrivals are the experimental batch-composition variable.
    for repetition in range(protocol["mixed_repeats"]):
        ordered = sorted(cases, key=lambda row: canonical_hash([repetition, row["case_id"]]))
        for offset in range(0, len(ordered), protocol["batch_width"]):
            batch = ordered[offset : offset + protocol["batch_width"]]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(execute, case, "mixed", repetition) for case in batch]
                results.extend(future.result() for future in futures)
    summary = summarize_probe(results, expected_cases=[case["case_id"] for case in cases])
    summary.update(
        {
            "condition": args.condition,
            "protocol_sha256": protocol["canonical_sha256"],
            "model_pool_sha256": pool["canonical_sha256"],
            "no_tools_or_evaluator_invoked": True,
            "sota_evidence": False,
        }
    )
    summary["canonical_sha256"] = canonical_hash(summary)
    write_new(output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def summarize_probe(results: list[dict], *, expected_cases: list[str]) -> dict:
    if not expected_cases or len(set(expected_cases)) != len(expected_cases):
        raise ValueError("probe case identities must be nonempty and unique")
    grouped = {case: [] for case in expected_cases}
    for row in results:
        if row["case_id"] not in grouped:
            raise ValueError("probe output contains an unexpected case")
        grouped[row["case_id"]].append(row)
    expected_keys = {("isolated", 0), ("isolated", 1), ("mixed", 0), ("mixed", 1)}
    for rows in grouped.values():
        if len(rows) != 4 or {(row["layout"], row["repetition"]) for row in rows} != expected_keys:
            raise ValueError("probe is incomplete or duplicates a condition")
    inconsistent = [
        case for case, rows in grouped.items() if len({row["output_sha256"] for row in rows}) != 1
    ]
    layout_drift = {
        layout: [
            case
            for case, rows in grouped.items()
            if len({row["output_sha256"] for row in rows if row["layout"] == layout}) != 1
        ]
        for layout in ("isolated", "mixed")
    }
    return {
        "status": "COMPLETE",
        "input_cases": len(grouped),
        "generations": len(results),
        "inconsistent_case_count": len(inconsistent),
        "inconsistent_cases": inconsistent,
        "within_layout_inconsistent_cases": layout_drift,
        "finite_fidelity_gate_passed": not inconsistent,
        "total_tokens": sum(row["usage"]["total_tokens"] for row in results),
        "scope": "Finite provider-repeatability gate, not model correctness or benchmark superiority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-study", type=Path, required=True)
    prep.add_argument("--case-selection", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--experiment-dir", type=Path, required=True)
    execute.add_argument("--pool", type=Path, required=True)
    execute.add_argument("--condition", type=int, choices=(0, 1), required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--experiment-dir", type=Path, required=True)
    inspect.add_argument("--condition", type=int, choices=(0, 1), required=True)
    inspect.add_argument("--output", type=Path, required=True)
    successor = commands.add_parser("serial-successor")
    successor.add_argument("--predecessor", type=Path, required=True)
    successor.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--experiment-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    remote_guard()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "compare":
        root = args.experiment_dir.resolve()
        protocol = load(root / "protocol.json")
        if protocol["canonical_sha256"] != canonical_hash(
            {key: value for key, value in protocol.items() if key != "canonical_sha256"}
        ):
            raise RuntimeError("comparison protocol changed")
        reference_binding = protocol["reference_condition_0"]
        reference_path = Path(reference_binding["path"])
        if file_hash(reference_path) != reference_binding["file_sha256"]:
            raise RuntimeError("reused reference condition changed")
        reference = load(reference_path)
        current = load(root / "condition-1/SUMMARY.json")
        for value in (reference, current):
            if (
                value["status"] != "COMPLETE"
                or value["input_cases"] != protocol["input_cases"]
                or value["canonical_sha256"]
                != canonical_hash(
                    {key: item for key, item in value.items() if key != "canonical_sha256"}
                )
            ):
                raise RuntimeError("fidelity condition is incomplete or changed")
        if current["protocol_sha256"] != protocol["canonical_sha256"]:
            raise RuntimeError("serial condition belongs to a different frozen protocol")
        result = {
            **metadata(),
            "schema_version": "phiagent-provider-fidelity-comparison/v1",
            "status": "COMPLETE_FINITE_PROVIDER_FIDELITY_COMPARISON",
            "input_cases": protocol["input_cases"],
            "reference_inconsistent_cases": reference["inconsistent_case_count"],
            "serial_inconsistent_cases": current["inconsistent_case_count"],
            "reference_generations_reused": reference["generations"],
            "new_serial_generations": current["generations"],
            "total_known_probe_tokens": reference["total_tokens"] + current["total_tokens"],
            "finite_serial_gate_passed": current["finite_fidelity_gate_passed"],
            "source_summaries": {
                "reference": file_hash(reference_path),
                "serial": file_hash(root / "condition-1/SUMMARY.json"),
            },
            "scope": "Finite repeatability evidence only; not a benchmark score, correctness guarantee or SOTA claim.",
            "unsupported_batch_invariant_mode_was_not_bypassed": True,
            "sota_evidence": False,
            "new_generations_in_this_analysis": 0,
        }
        result["canonical_sha256"] = canonical_hash(result)
        write_new(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "serial-successor":
        previous = args.predecessor.resolve()
        protocol = load(previous / "protocol.json")
        if protocol["canonical_sha256"] != canonical_hash(
            {key: value for key, value in protocol.items() if key != "canonical_sha256"}
        ):
            raise RuntimeError("predecessor fidelity protocol changed")
        if file_hash(previous / "requests.json") != protocol["requests_sha256"]:
            raise RuntimeError("predecessor requests changed")
        reference_path = previous / "condition-0/SUMMARY.json"
        reference = load(reference_path)
        if (
            reference["status"] != "COMPLETE"
            or reference["protocol_sha256"] != protocol["canonical_sha256"]
            or reference["canonical_sha256"]
            != canonical_hash(
                {key: value for key, value in reference.items() if key != "canonical_sha256"}
            )
        ):
            raise RuntimeError("completed baseline condition is not immutable and protocol-bound")
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "requests.json", load(previous / "requests.json"))
        if file_hash(output / "requests.json") != protocol["requests_sha256"]:
            raise RuntimeError("successor changed the frozen request inputs")
        value = {
            **{key: item for key, item in protocol.items() if key != "canonical_sha256"},
            **metadata(),
            "schema_version": "phiagent-model-fidelity-probe/v2",
            "comparison": "ordinary max_num_seqs16 versus ordinary max_num_seqs1",
            "change_reason": "Pinned model uses GDN_ATTN, which explicitly does not support VLLM_BATCH_INVARIANT. Normal isolated repeats were stable; mixed batches drifted.",
            "condition_settings": {
                "0": {"batch_invariant": 0, "max_num_seqs": 16},
                "1": {"batch_invariant": 0, "max_num_seqs": 1},
            },
            "reference_condition_0": {
                "path": str(reference_path),
                "file_sha256": file_hash(reference_path),
                "canonical_sha256": reference["canonical_sha256"],
                "generations_reused_without_modification": reference["generations"],
            },
            "requests_sha256": file_hash(output / "requests.json"),
            "predecessor_protocol_sha256": protocol["canonical_sha256"],
        }
        value["canonical_sha256"] = canonical_hash(value)
        write_new(output / "protocol.json", value)
        print(json.dumps({"status": value["status"], "comparison": value["comparison"]}))
        return 0
    if args.command == "inspect":
        root = args.experiment_dir.resolve()
        protocol = load(root / "protocol.json")
        requests = load(root / "requests.json")
        results = [
            load(path)
            for path in sorted((root / f"condition-{args.condition}").glob("*/*/*/result.json"))
        ]
        result = {
            **metadata(),
            **summarize_probe(results, expected_cases=[row["case_id"] for row in requests]),
            "protocol_sha256": protocol["canonical_sha256"],
            "new_generations": 0,
            "inspection_only": True,
        }
        write_new(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
