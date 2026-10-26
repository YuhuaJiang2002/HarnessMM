"""Audit complete remote TRAIN trajectories, then cross-fit a paid-recovery policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import figure2_searchqa_ranker as ranker  # noqa: E402
from scripts import figure2_searchqa_recovery as recovery  # noqa: E402
from scripts.figure2_searchqa_train import official_scorer, sha  # noqa: E402


PENALTIES = (.000025, .00005, .000075, .0001, .00015)
FEATURES = (
    "question_words", "selected_clue_coverage", "best_source_clue_coverage",
    "question_has_wh_word", "quoted_clue", "year_clue",
    "answer_question_overlap", "answer_words", "minimum_logprob", "mean_logprob",
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def feature_vector(item, compact, predictor):
    first = compact["calls"][0]
    if compact["paid_recovery"] or "confidence" not in first:
        return None
    confidence = first["confidence"]
    if confidence["uses_gold_or_task_id"] or not confidence["native_provider_feature"]:
        raise ValueError("Only measured native model confidence is eligible")
    public = recovery.public_features(item, compact["source_receipt"], predictor)
    answer, question = set(ranker.words(compact["prediction"])), set(ranker.words(item["question"]))
    public.update(
        minimum_logprob=confidence["minimum_logprob"], mean_logprob=confidence["mean_logprob"],
        answer_question_overlap=len(answer & question) / max(len(answer), 1),
        answer_words=len(ranker.words(compact["prediction"])),
    )
    result = {name: float(public[name]) for name in FEATURES}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Confidence/public features must be finite")
    return result


def aggregate(examples, routes):
    if len(examples) != len(routes):
        raise ValueError("Every complete TRAIN question requires one route")
    return {
        "tasks": len(examples),
        "correct": sum(row[route + "_em"] for row, route in zip(examples, routes)),
        "actor_tokens": sum(row[route + "_tokens"] for row, route in zip(examples, routes)),
        "mean_actor_tokens": sum(row[route + "_tokens"] for row, route in zip(examples, routes))
        / len(examples),
        "new_full_recoveries": routes.count("original"),
    }


def fit_fold(arguments):
    examples, penalty, fold = arguments
    training = [row for row in examples if row["fold"] != fold and row["features"] is not None]
    if len(training) < 256:
        raise ValueError("Insufficient TRAIN-only confidence examples for a stable inner fit")
    policy = recovery.fit_tree(training, penalty, depth=2, minimum_leaf=64)
    routes = {
        row["id"]: recovery.route(policy["tree"], row["features"])
        if row["features"] is not None else "compact"
        for row in examples if row["fold"] == fold
    }
    return {"penalty": penalty, "fold": fold, "policy": policy, "routes": routes}


def collect_and_audit(root):
    config, frozen, completed = (read(root / name) for name in
                                ("pool-config.json", "SUCCESSOR_FREEZE.json", "TRAIN_DATASET_COMPLETE.json"))
    if (completed["status"] != "COMPLETE_3395_TRAIN_ROUTER_DATA"
            or completed["validation_calls"] or completed["test_calls"]
            or sha(root / "pool-config.json") != frozen["config_sha256"]):
        raise ValueError("A complete, frozen TRAIN collection is required")
    for name, expected in frozen["source_sha256"].items():
        if sha(root / "source" / name) != expected:
            raise ValueError("Collection source changed: " + name)
    for name, expected in read(root / "INPUTS_FROZEN.json").items():
        if sha(root / name) != expected:
            raise ValueError("Collection input changed: " + name)
    partition = read(root / "inputs/PARTITION.json")
    ids = set(partition["train"]["ids"])
    if len(ids) != 3395:
        raise ValueError("Full canonical TRAIN population required")
    arms = ("no_skill", "chisel_current", "target_role_guard640")
    results = {name: {} for name in arms}
    reused = read(root / "inputs/REUSED_GATE.json")
    for row in reused["rows"]:
        if (row["id"] not in ids or row["split"] != "train" or row["arm"] not in results
                or row["id"] in results[row["arm"]]):
            raise ValueError("Reused outcomes must be unique canonical TRAIN cells")
        results[row["arm"]][row["id"]] = row
    requests, tokens, confidence_checked = {}, 0, 0
    for path in (root / "calls").glob("*.json"):
        call = read(path)
        body, raw, usage = call["request"], call["raw"], call["usage"]
        if (call["status"] != "COMPLETE" or call["id"] not in ids
                or body["model"] != config["model_repository"] or raw["model"] != body["model"]
                or body["seed"] != config["request_seed"] or body["max_tokens"] != 512
                or body["temperature"] != 0 or not body["logprobs"] or body["top_logprobs"] != 0
                or raw["usage"] != usage
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                or usage["prompt_tokens"] != call["expected_prompt_tokens"]):
            raise ValueError("Incomplete or mismatched native request/usage evidence")
        if call["confidence"] != recovery.response_confidence(raw):
            raise ValueError("Native token-confidence replay differs")
        confidence_checked += 1
        requests[path.stem] = call
        tokens += usage["total_tokens"]
    used = set()
    for path in (root / "tasks").glob("*.json"):
        row = read(path)
        if row["id"] not in ids or row["split"] != "train" or row["id"] in results[row["arm"]]:
            raise ValueError("Duplicate or held-out task entered training")
        observed = {name: 0 for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
        for reference in row["calls"]:
            key = reference["call_id"]
            if key in used:
                raise ValueError("A provider call was charged to two outcomes")
            used.add(key)
            call = requests[key]
            if (call["id"] != row["id"] or call["arm"] != row["arm"]
                    or reference.get("confidence") != call["confidence"]):
                raise ValueError("Wrong task/method call binding")
            for field in observed:
                observed[field] += call["usage"][field]
        if observed != row["usage"]:
            raise ValueError("Task usage differs from raw native calls")
        results[row["arm"]][row["id"]] = row
    if used != requests.keys() or any(set(rows) != ids for rows in results.values()):
        raise ValueError("Every TRAIN outcome and paid call must appear exactly once")
    if tokens != completed["new_provider_usage"]["total_tokens"]:
        raise ValueError("New provider cost differs")
    total = sum(row["usage"]["total_tokens"] for rows in results.values() for row in rows.values())
    if total != tokens + reused["provider_tokens"] or total != completed["logical_execution_tokens"]:
        raise ValueError("New and reused physical costs do not reconcile")
    scorer = official_scorer({"scorer": str(root / "inputs/mrqa_official_eval.py"),
                             "scorer_sha256": config["scorer_sha256"]})
    public, labels = {}, {}
    archive = Path(config["hosts"][socket.gethostname()]["archive"])
    if sha(archive) != config["archive_sha256"]:
        raise ValueError("Canonical publisher archive changed")
    with gzip.open(archive, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for qa in record.get("qas", []):
                if qa["qid"] in ids:
                    public[qa["qid"]] = {"question": qa["question"], "context": record["context"]}
                    labels[qa["qid"]] = qa["answers"]
    if set(public) != ids:
        raise ValueError("Missing canonical TRAIN payloads")
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel

    reference = root / "inputs/reference"
    extractor = FrozenOriginalChisel(reference / "lineage", read(reference / "lineage/config.json"))
    for rows in results.values():
        for row in rows.values():
            if extractor.campaign.extract_answer(row["answer"]) != row["prediction"]:
                raise ValueError("Final answer extraction differs from the frozen native adapter")
            correct = int(scorer.metric_max_over_ground_truths(
                scorer.exact_match_score, row["prediction"], labels[row["id"]],
            ))
            f1 = scorer.metric_max_over_ground_truths(
                scorer.f1_score, row["prediction"], labels[row["id"]],
            )
            if correct != row["em"] or not math.isclose(f1, row["f1"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Native EM/F1 replay differs")
    for arm, rows in results.items():
        summary = ranker.summary(list(rows.values()))
        if any(not math.isclose(value, completed["methods"][arm][key], rel_tol=0, abs_tol=1e-9)
               for key, value in summary.items()):
            raise ValueError("Complete per-method summary replay differs")
    groups = {identity: group for group, members in partition["groups"].items() for identity in members}
    predictor = read(root / "inputs/RANKER_FROZEN.json")
    examples = []
    for identity in sorted(ids):
        compact, original = results["target_role_guard640"][identity], results["chisel_current"][identity]
        features = feature_vector(public[identity], compact, predictor)
        examples.append({
            "id": identity, "fold": int(ranker.digest("risk-inner-20260928:" + groups[identity]), 16) % 5,
            "features": features, "compact_em": compact["em"], "original_em": original["em"],
            "compact_tokens": compact["usage"]["total_tokens"],
            "original_tokens": compact["usage"]["total_tokens"] + original["usage"]["total_tokens"],
        })
    return examples, {
        "status": "PASS_COMPLETE_TRAIN_AUDIT", "native_rescored": 3395 * 3,
        "new_native_calls": len(requests), "confidence_records_verified": confidence_checked,
        "new_provider_tokens": tokens, "reused_tokens_once": reused["provider_tokens"],
        "logical_execution_tokens": total, "unknown_usage_calls": 0,
        "eligible_confidence_examples": sum(row["features"] is not None for row in examples),
        "missing_or_already_recovered_stays_base": True,
        "required_correct": max(sum(row["em"] for row in rows.values()) for rows in results.values()),
        "token_target": .8 * completed["methods"]["no_skill"]["mean_actor_tokens"],
        "source_complete_sha256": sha(root / "TRAIN_DATASET_COMPLETE.json"),
        "validation_calls": 0, "test_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (socket.gethostname() != "harnessmm-worker-2" or args.output.exists()
            or not args.output.is_relative_to("/opt/phiagent/work/figure2-token-efficiency")):
        raise RuntimeError("Use a new authorized remote TRAIN fitting directory")
    args.output.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    write(args.output / "STARTED.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "hostname": socket.gethostname(),
        "source_sha256": sha(Path(__file__)), "penalties": PENALTIES,
        "depth": 2, "minimum_leaf": 64, "inner_folds": 5, "features": FEATURES,
        "native_model_calls": 0, "new_validation_test_calls": 0, "pid": os.getpid(),
    })
    try:
        examples, audit = collect_and_audit(args.root)
        write(args.output / "NATIVE_TRAIN_AUDIT.json", audit)
        write(args.output / "TRAIN_FEATURES.json", examples)
        jobs = [(examples, penalty, fold) for penalty in PENALTIES for fold in range(5)]
        results = []
        with ProcessPoolExecutor(max_workers=5) as pool:
            for result in pool.map(fit_fold, jobs):
                results.append(result)
                write(args.output / "PROGRESS.json", {"completed_fits": len(results), "total_fits": len(jobs)})
        summaries = []
        for penalty in PENALTIES:
            folds = [row for row in results if row["penalty"] == penalty]
            predictions = {key: value for row in folds for key, value in row["routes"].items()}
            summary = aggregate(examples, [predictions[row["id"]] for row in examples])
            summaries.append({
                "penalty": penalty, **summary,
                "passes_inner_training_gate": summary["correct"] >= audit["required_correct"]
                and summary["mean_actor_tokens"] <= audit["token_target"],
            })
        feasible = [row for row in summaries if row["passes_inner_training_gate"]]
        selected = min(feasible, key=lambda row: row["mean_actor_tokens"]) if feasible else None
        if selected:
            policy = recovery.fit_tree(
                [row for row in examples if row["features"] is not None],
                selected["penalty"], depth=2, minimum_leaf=64,
            )
            write(args.output / "FROZEN_TRAIN_POLICY.json", {
                **policy, "features": FEATURES, "cost_mode": "paid_recovery_after_compact_draft",
                "already_recovered_never_recovered_again": True,
                "native_confidence_required_at_inference": True, "source_audit_sha256":
                sha(args.output / "NATIVE_TRAIN_AUDIT.json"), "validation_or_test_feedback": False,
            })
        write(args.output / "DECISION.json", {
            "status": "TRAIN_CROSSFIT_CANDIDATE_READY" if selected else "NO_FEASIBLE_TRAIN_CROSSFIT_POLICY",
            "training_questions": 3395, "inner_crossfit": summaries, "selected": selected,
            "required_correct": audit["required_correct"], "token_target": audit["token_target"],
            "elapsed_seconds": time.monotonic() - started, "new_model_calls": 0,
            "new_scorer_calls": 3395 * 3, "official_validation_or_test_executions": 0,
            "counterfactual_training_estimate_not_final_execution_result": True, "paper_admitted": False,
            "next": "A trained eligible policy needs a separately frozen native validation evaluation; "
                    "a negative fit remains TRAIN evidence and does not open held-out data.",
        })
        print(json.dumps(read(args.output / "DECISION.json"), indent=2))
    except Exception as error:
        write(args.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
