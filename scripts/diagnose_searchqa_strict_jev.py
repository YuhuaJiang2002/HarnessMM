"""TRAIN-only candidate-coverage diagnosis for the isolated Jev successor."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys


HOSTS = ("harnessmm-worker-1", "harnessmm-worker-2")


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def train_records(archive, ids):
    items, labels = {}, {}
    decoded = 0
    with gzip.open(archive, "rb") as stream:
        for line in stream:
            present = {value.decode() for value in re.findall(rb'"qid"\s*:\s*"([^"\\]+)"', line)}
            if not present.intersection(ids):
                continue
            if not present <= ids:
                raise ValueError("A publisher context crosses the independent TRAIN boundary")
            record = json.loads(line)
            decoded += 1
            for qa in record["qas"]:
                if qa["qid"] in items:
                    raise ValueError("Duplicate TRAIN question")
                items[qa["qid"]] = {
                    "id": qa["qid"], "question": qa["question"], "context": record["context"],
                }
                labels[qa["qid"]] = qa["answers"]
    if set(items) != ids:
        raise ValueError("The complete canonical TRAIN population is required")
    return items, labels, decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    args = parser.parse_args()
    if (socket.gethostname() not in HOSTS or args.output.exists()
            or args.output.parent != Path("/opt/phiagent/work/figure2-token-efficiency")):
        raise RuntimeError("Use one new authorized remote TRAIN diagnosis directory")
    args.output.mkdir()
    config = read(args.parent / "pool-config.json")
    host = config["hosts"][socket.gethostname()]
    if digest(host["archive"]) != config["archive_sha256"]:
        raise ValueError("Publisher archive changed")
    partition = read(args.parent / "inputs/PARTITION.json")
    if digest(args.parent / "inputs/PARTITION.json") != config["partition_sha256"]:
        raise ValueError("Canonical independent-group partition changed")
    save(args.output / "PROVENANCE.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "argv": sys.argv, "source_sha256": digest(__file__), "git_head": args.git_head,
        "request_seed": config["request_seed"], "split_seed": partition["seed"],
        "parent_root": str(args.parent), "parent_result_sha256": digest(args.parent / "TRAIN_RESULT.json"),
        "partition_sha256": config["partition_sha256"], "scope": "TRAIN3395",
        "python": sys.version, "packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True),
        "model_calls": 0, "validation_test_records_decoded": 0,
        "maximum_cumulative_tokens": None, "token_usage_mode": "unlimited",
    })
    sys.path.insert(0, str(args.parent / "source"))
    from scripts.figure2_searchqa_train import official_scorer

    scorer = official_scorer({
        "scorer": str(args.parent / "inputs/mrqa_official_eval.py"),
        "scorer_sha256": config["scorer_sha256"],
    })
    ids = set(partition["train"]["ids"])
    _, labels, decoded = train_records(host["archive"], ids)
    rows = [read(path) for path in sorted((args.parent / "tasks").glob("*.json"))]
    if len(ids) != 3395 or len(rows) != len(ids) or {row["id"] for row in rows} != ids:
        raise ValueError("Every native TRAIN task must be diagnosed once")
    counts, oracle_em, oracle_f1 = Counter(), 0, 0.0
    calls_hashes = {}
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Protected task encountered")
        path = args.parent / "calls" / (row["calls"][0]["call_id"] + ".json")
        record = read(path)
        if record["id"] != row["id"] or record["status"] != "COMPLETE":
            raise ValueError("Missing native draft receipt")
        calls_hashes[str(path)] = digest(path)
        text = record["raw"]["choices"][0]["message"]["content"]
        match = re.search(r"<candidates>(.*?)</candidates>", text, re.S | re.I)
        candidates = re.findall(r"(?:^|;)\s*[AB]\s*:\s*([^;]+)", match[1]) if match else []
        counts["declared_bank_tasks"] += bool(candidates)
        counts["paid_recovery_tasks"] += bool(row["paid_recovery"])
        em = max((scorer.metric_max_over_ground_truths(
            scorer.exact_match_score, value.strip(), labels[row["id"]]) for value in candidates), default=0)
        f1 = max((scorer.metric_max_over_ground_truths(
            scorer.f1_score, value.strip(), labels[row["id"]]) for value in candidates), default=0.0)
        oracle_em += em
        oracle_f1 += f1
        counts["bank_contains_exact_answer"] += em
        counts["wrong_native_answer_but_bank_contains_exact_answer"] += bool(em and not row["em"])
        counts["native_correct_but_declared_bank_missing_exact_answer"] += bool(row["em"] and not em)
    result = {
        "schema": "searchqa-strict-jev-train-diagnosis/v1",
        "status": "COMPLETE_TRAIN_ONLY_CANDIDATE_COVERAGE_DIAGNOSIS",
        "tasks": len(rows), "publisher_train_records_decoded": decoded,
        "validation_test_records_decoded": 0, "new_model_calls": 0,
        "counts": dict(counts), "declared_bank_oracle_em_percent": 100 * oracle_em / len(rows),
        "declared_bank_oracle_f1_percent": 100 * oracle_f1 / len(rows),
        "oracle_is_diagnostic_not_a_method_result": True,
        "parent_native_result": read(args.parent / "TRAIN_RESULT.json")["summary"],
        "candidate_first_call_hashes_sha256": hashlib.sha256(
            json.dumps(calls_hashes, sort_keys=True).encode()).hexdigest(),
        "rationale": (
            "Replace ordinary slow answer selection with explicit CoT-authored alternative "
            "candidates and a same-backbone, probability-audited constrained label selector; "
            "retain paid native fallback on invalid candidate evidence or explicit abstention."
        ),
        "new_candidate_uses_gold_or_task_id_routes": False,
        "maximum_cumulative_tokens": None, "token_usage_mode": "unlimited",
    }
    save(args.output / "TRAIN_DIAGNOSIS.json", result)
    save(args.output / "TERMINAL.json", {
        "status": result["status"], "at_utc": datetime.now(timezone.utc).isoformat(),
        "model_calls": 0, "official_scorer_used_only_on_train": True,
    })
    print(json.dumps({key: result[key] for key in (
        "status", "tasks", "counts", "declared_bank_oracle_em_percent",
        "declared_bank_oracle_f1_percent")}))


if __name__ == "__main__":
    main()
