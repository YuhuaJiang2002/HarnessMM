"""Reconstruct complete remote TRAIN packing results from raw native responses."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import figure2_searchqa_ranker as ranker  # noqa: E402
from scripts import figure2_searchqa_recovery as recovery  # noqa: E402
from scripts import figure2_searchqa_v4 as native  # noqa: E402
from scripts import searchqa_span_union as packing  # noqa: E402
from scripts.analyze_searchqa_accuracy_headroom import load_train, read  # noqa: E402
from scripts.figure2_searchqa_train import official_scorer, save, sha  # noqa: E402


def check_usage(call: dict, config: dict, count) -> dict:
    request, raw = call["request"], call["raw"]
    usage = raw["usage"]
    if (call["status"] != "COMPLETE"
            or request["model"] != config["model_repository"]
            or raw["model"] != request["model"]
            or request["seed"] != config["request_seed"]
            or request["temperature"] != 0 or request["max_tokens"] != 512
            or request["chat_template_kwargs"] != {"enable_thinking": False}
            or request.get("logprobs") is not True or request.get("top_logprobs") != 0
            or any(type(usage.get(key)) is not int or usage[key] < 0
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or usage != call["usage"]
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
            or usage["prompt_tokens"] != count(request["messages"])
            or call["expected_prompt_tokens"] != usage["prompt_tokens"]
            or call["confidence"] != recovery.response_confidence(raw)):
        raise ValueError("Raw native request, model identity, confidence or usage differs")
    return usage


def check_summary(observed: dict, reported: dict) -> None:
    if set(observed) != set(reported):
        raise ValueError("Complete native summary fields differ")
    for field, value in observed.items():
        matches = (
            math.isclose(value, reported[field], rel_tol=0, abs_tol=1e-10)
            if field == "f1_percent" else value == reported[field]
        )
        if not matches:
            raise ValueError(f"Complete native summary differs for {field}: {value} != {reported[field]}")


def audit(root: Path) -> dict:
    from transformers import AutoTokenizer
    from phiagent.learning import qa_meta_rsi
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel

    config, freeze, result = (
        read(root / name) for name in ("pool-config.json", "CANDIDATES_FROZEN.json", "TRAIN_RESULT.json")
    )
    source = read(root / "SUCCESSOR_FREEZE.json")
    if (sha(root / "pool-config.json") != source["config_sha256"]
            or result["status"] != "COMPLETE_3395_NATIVE_TRAIN"
            or result["validation_calls"] or result["test_calls"]
            or config["maximum_total_provider_tokens"] is not None
            or config["token_usage_mode"] != "unlimited"):
        raise ValueError("Expected a frozen complete TRAIN-only execution")
    for name, expected in source["source_sha256"].items():
        if sha(root / "source" / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in read(root / "INPUTS_FROZEN.json").items():
        if sha(root / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    items, labels, references, partition = load_train(Path(config["predecessor"]))
    host = config["hosts"][socket.gethostname()]
    tokenizer = AutoTokenizer.from_pretrained(host["model_directory"], local_files_only=True)

    def count(messages):
        return len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        ))

    reference = root / "inputs/reference"
    profiles = read(reference / "PROFILES.json")
    router = FrozenOriginalChisel(reference / "lineage", read(reference / "lineage/config.json"))
    original = qa_meta_rsi.initial_skill({k: profiles[k] for k in ("delete", "modify")})
    scorer = official_scorer({
        "scorer": str(root / "inputs/mrqa_official_eval.py"),
        "scorer_sha256": config["scorer_sha256"],
    })
    candidate = freeze["candidate"]
    rows, used, totals = {}, set(), {k: 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    raw_hashes = {}
    for path in sorted((root / "tasks").glob("*.json")):
        row = read(path)
        identity = row["id"]
        if identity in rows or identity not in items or row["arm"] != candidate or row["split"] != "train":
            raise ValueError("Duplicate or non-TRAIN candidate outcome")
        item, receipt = items[identity], row["source_receipt"]
        spans = [tuple(span) for span in receipt["spans"]]
        if (spans != packing.union_spans([tuple(span) for span in receipt["admitted_windows"]])
                or any(not 0 <= start < end <= len(item["context"]) for start, end in spans)
                or receipt["context_sha256"] != ranker.digest(item["context"])):
            raise ValueError("Source interval union does not reproduce exact public text")
        context = "\n...\n".join(item["context"][start:end] for start, end in spans)
        draft_messages = ranker.messages(item, context, freeze["system"])
        budget_messages = ranker.messages(item, context, recovery.CONTRACT_SYSTEM + freeze["extra_instruction"])
        if (receipt["actual_input_tokens"] != count(draft_messages)
                or receipt["source_selection_input_tokens"] != count(budget_messages)
                or count(budget_messages) > config["source_selection_input_cap"]):
            raise ValueError("Exact inherited instructions or source capacity is not fully charged")
        replay_calls = []

        def replay(messages, stage):
            key = ranker.digest(f"{identity}:{candidate}:{stage}")
            if key in used:
                raise ValueError("A paid native response is reused across outcomes")
            call_path = root / "calls" / (key + ".json")
            recorded = read(call_path)
            if (recorded["id"] != identity or recorded["arm"] != candidate
                    or recorded["stage"] != stage or recorded["request"]["messages"] != messages):
                raise ValueError("Native request differs from the frozen candidate or fallback")
            usage = check_usage(recorded, config, count)
            reference_call = row["calls"][len(replay_calls)]
            if (reference_call["call_id"] != key or reference_call["usage"] != usage
                    or reference_call["confidence"] != recorded["confidence"]):
                raise ValueError("Row call binding differs from native response")
            for field in totals:
                totals[field] += usage[field]
            replay_calls.append(recorded)
            raw_hashes[str(call_path.relative_to(root))] = sha(call_path)
            used.add(key)
            return recorded["raw"]["choices"][0]["message"]["content"]

        answer = replay(draft_messages, "compact")
        recovered = recovery.fallback_needed(item["question"], item["context"], answer)
        if recovered:
            answer = native.current_chisel(item, router, profiles, original, replay, "paid_recovery")
        prediction = router.campaign.extract_answer(answer)
        em = int(scorer.metric_max_over_ground_truths(scorer.exact_match_score, prediction, labels[identity]))
        f1 = scorer.metric_max_over_ground_truths(scorer.f1_score, prediction, labels[identity])
        usage = {field: sum(call["usage"][field] for call in replay_calls) for field in totals}
        if (len(row["calls"]) != len(replay_calls) or row["answer"] != answer
                or row["prediction"] != prediction or row["paid_recovery"] != recovered
                or row["em"] != em or abs(row["f1"] - f1) > 1e-12 or row["usage"] != usage):
            raise ValueError("Native score, paid fallback or episode usage differs")
        raw_hashes[str(path.relative_to(root))] = sha(path)
        rows[identity] = row
    if (set(rows) != set(items) or len(rows) != 3395
            or used != {path.stem for path in (root / "calls").glob("*.json")}):
        raise ValueError("Incomplete native population or orphaned provider call")
    ledger = read(root / "PROVIDER_ACCOUNTING.json")
    if (any(ledger[field] != totals[field] for field in totals)
            or ledger["calls"] != len(used) or ledger["failed_calls"] or ledger["unknown_usage_calls"]
            or result["provider_usage"] != ledger):
        raise ValueError("Complete raw costs disagree with the cumulative ledger")
    summary = ranker.summary(list(rows.values()))
    check_summary(summary, result["summary"])
    return {
        "status": "PASS_COMPLETE_NATIVE_TRAIN_RECONSTRUCTION",
        "tasks": len(rows), "native_calls_checked": len(used), "usage": totals, "summary": summary,
        "unknown_usage_calls": 0, "result_sha256": sha(root / "TRAIN_RESULT.json"),
        "policy_sha256": sha(root / "CANDIDATES_FROZEN.json"),
        "raw_file_sha256": raw_hashes,
        "paired_uncertainty": {
            arm: ranker.paired(list(rows.values()), list(reference_rows.values()), partition["groups"])
            for arm, reference_rows in references.items()
        },
        "new_model_calls": 0, "validation_calls": 0, "test_calls": 0,
        "paper_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (socket.gethostname() != "harnessmm-worker-2" or args.output.exists()
            or not args.output.is_relative_to("/opt/phiagent/work/figure2-token-efficiency")):
        raise RuntimeError("Use a new remote native-audit directory")
    args.output.mkdir()
    save(args.output / "STARTED.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "source_sha256": sha(Path(__file__)), "hostname": socket.gethostname(),
        "scope": "completed_train_only", "bootstrap_seed": 20260928, "bootstrap_draws": 1000,
    })
    try:
        result = audit(args.root)
        save(args.output / "AUDIT.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "raw_file_sha256"}, indent=2))
    except Exception as error:
        save(args.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
