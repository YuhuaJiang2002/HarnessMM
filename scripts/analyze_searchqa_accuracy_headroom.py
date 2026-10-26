"""Diagnose completed TRAIN compaction errors without new model calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import figure2_searchqa_ranker as ranker  # noqa: E402
from scripts.figure2_searchqa_train import save, sha  # noqa: E402
from scripts.train_searchqa_confidence_router import collect_and_audit  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text())


def source_support(context: str, answers: list[str]) -> bool:
    text = " " + " ".join(ranker.words(context)) + " "
    return any(
        words and " " + " ".join(words) + " " in text
        for words in (ranker.words(answer) for answer in answers)
    )


def load_train(root: Path) -> tuple[dict, dict, dict, dict]:
    config, partition = read(root / "pool-config.json"), read(root / "inputs/PARTITION.json")
    ids = set(partition["train"]["ids"])
    host = config["hosts"][socket.gethostname()]
    if sha(host["archive"]) != config["archive_sha256"]:
        raise ValueError("Publisher archive differs from the frozen TRAIN source")
    items, labels = {}, {}
    with gzip.open(host["archive"], "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for question in record.get("qas", []):
                identity = question["qid"]
                if identity in ids:
                    if identity in items:
                        raise ValueError("Duplicate TRAIN question")
                    items[identity] = {
                        "id": identity, "question": question["question"],
                        "context": record["context"],
                    }
                    labels[identity] = question["answers"]
    rows = {arm: {} for arm in ("no_skill", "chisel_current", "target_role_guard640")}
    recorded = read(root / "inputs/REUSED_GATE.json")["rows"]
    recorded.extend(read(path) for path in sorted((root / "tasks").glob("*.json")))
    for row in recorded:
        if (row["arm"] not in rows or row["id"] not in ids or row["split"] != "train"
                or row["id"] in rows[row["arm"]]):
            raise ValueError("Duplicate or out-of-scope TRAIN outcome")
        rows[row["arm"]][row["id"]] = row
    if set(items) != ids or any(set(arm) != ids for arm in rows.values()):
        raise ValueError("Complete matched TRAIN populations are required")
    return items, labels, rows, partition


def diagnose(items: dict, labels: dict, rows: dict) -> dict:
    counts = Counter()
    for identity, item in items.items():
        candidate = rows["target_role_guard640"][identity]
        reference = rows["chisel_current"][identity]
        category = (
            "both_correct" if candidate["em"] and reference["em"]
            else "incumbent_only" if reference["em"]
            else "compact_only" if candidate["em"]
            else "both_wrong"
        )
        counts[category] += 1
        spans = candidate["source_receipt"]["spans"]
        if any(not 0 <= start < end <= len(item["context"]) for start, end in spans):
            raise ValueError("Recorded source span is invalid")
        selected = "\n".join(item["context"][start:end] for start, end in spans)
        supported = source_support(selected, labels[identity])
        counts[category + ("_answer_retained" if supported else "_answer_absent")] += 1
        counts["recovered"] += bool(candidate["paid_recovery"])
        counts["recovered_" + category] += bool(candidate["paid_recovery"])
        counts["full_context_answer_absent"] += not source_support(item["context"], labels[identity])
    return {
        "tasks": len(items), "paired_outcomes": dict(counts),
        "source_support_is_diagnostic_not_native_correctness": True,
        "candidate_correct": counts["both_correct"] + counts["compact_only"],
        "incumbent_correct": counts["both_correct"] + counts["incumbent_only"],
        "required_net_recoveries": counts["incumbent_only"] - counts["compact_only"],
        "validation_or_test_feedback": False, "new_model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git-head", required=True)
    args = parser.parse_args()
    if (socket.gethostname() not in ("harnessmm-worker-1", "harnessmm-worker-2")
            or args.output.exists()
            or not args.output.is_relative_to("/opt/phiagent/work/figure2-token-efficiency")):
        raise RuntimeError("Use a new authorized remote TRAIN diagnosis directory")
    args.output.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    save(args.output / "PROVENANCE.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "argv": sys.argv, "source_sha256": sha(Path(__file__)),
        "predecessor": str(args.root), "seed": 20260928,
        "source_complete_sha256": sha(args.root / "TRAIN_DATASET_COMPLETE.json"),
        "source_git_head": args.source_git_head,
        "predecessor_git_state": read(args.root / "SUCCESSOR_FREEZE.json")["git_state"],
        "packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True),
        "python": sys.version, "new_model_calls": 0, "scope": "train_only",
    })
    try:
        _, audit = collect_and_audit(args.root)
        save(args.output / "NATIVE_TRAIN_AUDIT.json", audit)
        items, labels, rows, _ = load_train(args.root)
        result = diagnose(items, labels, rows)
        save(args.output / "HEADROOM.json", result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        save(args.output / "FAILURE.json", {
            "error": repr(error), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
