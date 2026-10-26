"""Run and independently replay the frozen, remote-only SearchQA Jev TRAIN gate."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.request import ProxyHandler, Request, build_opener


SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts import searchqa_strict_jev as method  # noqa: E402
from scripts.diagnose_searchqa_strict_jev import digest, read, train_records  # noqa: E402


BASE = Path("/opt/phiagent/work/figure2-token-efficiency")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def canonical_record(value):
    return json.loads(json.dumps(value, allow_nan=False))


def native_summary_view(full, reported):
    fields = {
        "tasks", "em_percent", "f1_percent", "mean_actor_tokens",
        "actor_input_tokens", "actor_output_tokens",
    }
    if set(reported) != fields or set(full) != fields | {"total_actor_tokens"}:
        raise ValueError("Unexpected native or official summary schema")
    if full["total_actor_tokens"] != full["actor_input_tokens"] + full["actor_output_tokens"]:
        raise ValueError("The independently reconstructed total actor cost is not additive")
    return {key: full[key] for key in reported}


def require_remote(config, root):
    if (socket.gethostname() != config["shared_endpoint"]["hostname"]
            or socket.gethostname() != "harnessmm-worker-2"
            or root.parent != BASE or str(root) != config["run_directory"]
            or config["schema"] != "searchqa-strict-jev/v1" or config["scope"] != "train"
            or config["training_tasks"] != 3395 or config["required_split"] != [20, 20, 60]
            or config["maximum_total_provider_tokens"] is not None
            or config["maximum_cumulative_tokens"] is not None
            or config["token_usage_mode"] != "unlimited"
            or config["validation_calls"] or config["test_calls"]):
        raise RuntimeError("Only the new frozen remote TRAIN3395 Jev run is authorized")
    resource = config["shared_endpoint"]
    os.environ.update(
        CUDA_VISIBLE_DEVICES=resource["gpu_uuid"], PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        TMPDIR=str(root / "runtime"), TMP=str(root / "runtime"), TEMP=str(root / "runtime"),
    )


def verify_freeze(root):
    frozen = read(root / "FROZEN.json")
    if digest(root / "config.json") != frozen["config_sha256"]:
        raise ValueError("Prospective Jev configuration changed")
    for name, expected in frozen["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen source/input changed: " + name)
    return frozen


def request_body(config, messages, maximum, extra=None):
    return {
        "model": config["shared_endpoint"]["served_model_name"], "messages": messages,
        "temperature": 0, "max_tokens": maximum, "seed": config["request_seed"],
        "chat_template_kwargs": {"enable_thinking": False}, **(extra or {}),
    }


class Provider:
    def __init__(self, root, config, tokenizer):
        self.root, self.config, self.tokenizer = root, config, tokenizer
        self.lock = threading.Lock()
        self.usage = method.complete_usage([])
        (root / "calls").mkdir()

    def count(self, messages):
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        ))

    def call(self, identity, stage, messages, maximum=512, extra=None):
        call_id = hashlib.sha256(
            f"{identity}:{self.config['candidate']}:{stage}:attempt0".encode()).hexdigest()
        path = self.root / "calls" / (call_id + ".json")
        if path.exists():
            raise RuntimeError("A paid native call cannot be repeated under the same identity")
        body = request_body(self.config, messages, maximum, extra)
        record = {
            "call_id": call_id, "id": identity, "stage": stage, "attempt": 0,
            "request": body, "expected_prompt_tokens": self.count(messages),
            "submitted_at_utc": now(), "status": "REQUEST_INTENT",
            "implementation_identity": self.config.get("implementation_identity", method.IDENTITY),
        }
        save(path, record)
        try:
            request = Request(
                self.config["shared_endpoint"]["url"] + "/v1/chat/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
            )
            with build_opener(ProxyHandler({})).open(request, timeout=300) as stream:
                raw = json.load(stream)
            record["raw"] = raw
            usage = raw["usage"]
            if (raw["model"] != body["model"]
                    or any(type(usage.get(key)) is not int or usage[key] < 0
                           for key in method.USAGE_FIELDS)
                    or usage["prompt_tokens"] != record["expected_prompt_tokens"]
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ValueError("Native model alias or complete input/output accounting differs")
            record.update(status="COMPLETE", usage=usage, completed_at_utc=now())
        except Exception as error:
            record.update(status="FAILED", error=repr(error), failed_at_utc=now())
            raise
        finally:
            save(path, record)
            with self.lock:
                observed = method.complete_usage([record])
                for key in (*method.USAGE_FIELDS, "calls", "failed_calls", "unknown_usage_calls"):
                    self.usage[key] += observed[key]
                save(self.root / "PROVIDER_ACCOUNTING.json", self.usage)
        return raw["choices"][0]["message"]["content"], record


def proposal_request(item, predictor, count, cap):
    from scripts import searchqa_autosota as selector

    def cot_count(messages):
        return count([{"role": "system", "content": method.COT_SYSTEM}, messages[1]])

    messages, receipt = selector.request(item, predictor, cot_count, cap)
    messages[0] = {"role": "system", "content": method.COT_SYSTEM}
    source = "\n...\n".join(item["context"][start:end] for start, end in receipt["spans"])
    if messages != method.candidate_messages(item["question"], source) or count(messages) > cap:
        raise ValueError("The full CoT request exceeded its frozen public-source capacity")
    return messages, {
        **receipt, "actual_input_tokens": count(messages),
        "system_identity": "explicit CoT candidate authoring; no final generated answer",
    }, source


def policy(item, predictor, router, profiles, original, provider, config):
    from scripts import figure2_searchqa_recovery as recovery
    from scripts import figure2_searchqa_v4 as native

    settings = config["method"]
    messages, receipt, source = proposal_request(
        item, predictor, provider.count, settings["slow_candidate_input_cap"],
    )
    records = []

    def call(request, stage, maximum=512, extra=None):
        text, record = provider.call(item["id"], stage, request, maximum, extra)
        records.append(record)
        return text, record

    slow, _ = call(messages, "cot_candidates", settings["slow_native_output_cap"], {
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "searchqa_cot_candidates", "strict": True, "schema": method.CANDIDATE_SCHEMA,
        }},
    })
    proposal, decision, reason, answer = None, None, None, None
    try:
        proposal = method.parse_candidates(slow, source, item["question"])
        fast_messages = method.decision_messages(item["question"], proposal)
        if provider.count(fast_messages) > settings["fast_complete_input_cap"]:
            reason = "DECLARED_DECISION_INPUT_CAPACITY_ABSTENTION"
        else:
            labels = [*method.LABELS[:len(proposal["candidates"])], "?"]
            _, fast = call(fast_messages, "jev_label", 1, {
                "structured_outputs": {"choice": labels},
                "logprobs": True, "top_logprobs": settings["fast_top_logprobs"],
            })
            decision = method.decide(
                fast["raw"], len(proposal["candidates"]), settings["confidence_floor"],
            )
            if decision["abstained"]:
                reason = "JEV_EXPLICIT_OR_PROBABILITY_ABSTENTION"
            else:
                index = method.LABELS.index(decision["selected_label"])
                answer = "<answer>" + proposal["candidates"][index]["text"].strip() + "</answer>"
                if recovery.fallback_needed(item["question"], item["context"], answer):
                    reason = "SELECTED_CANDIDATE_NATIVE_PUBLIC_GUARD"
    except (ValueError, TypeError, KeyError) as error:
        reason = "INVALID_DECLARED_COT_OR_PROBABILITY_DECISION: " + str(error)
    if reason is not None:
        answer = native.current_chisel(
            item, router, profiles, original,
            lambda request, stage: call(request, stage)[0], "native_fallback",
        )
    if not answer:
        raise ValueError("A complete task needs a candidate decision or native slow fallback")
    usage = method.complete_usage(records)
    return {
        "answer": answer, "proposal": proposal, "decision": decision,
        "paid_recovery": reason is not None, "fallback_reason": reason,
        "source_receipt": receipt,
        "calls": [
            {key: record[key] for key in ("call_id", "stage", "status", "usage")}
            for record in records
        ],
        "usage": {key: usage[key] for key in method.USAGE_FIELDS},
    }


def load_inputs(root, config):
    from phiagent.learning import qa_meta_rsi
    from scripts.figure2_searchqa_train import official_scorer
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel
    from scripts.searchqa_official_admission import population
    from transformers import AutoTokenizer

    partition = read(root / "inputs/PARTITION.json")
    ids = population(partition)["train"]
    if (partition["seed"] != config["split_seed"]
            or digest(root / "inputs/PARTITION.json") != config["partition_sha256"]
            or digest(config["archive"]) != config["archive_sha256"]):
        raise ValueError("Canonical split, seed or publisher archive changed")
    items, labels, decoded = train_records(config["archive"], ids)
    profiles = read(root / "inputs/reference/PROFILES.json")
    parent = read(root / "inputs/reference/lineage/config.json")
    router = FrozenOriginalChisel(root / "inputs/reference/lineage", parent)
    original = qa_meta_rsi.initial_skill({key: profiles[key] for key in ("delete", "modify")})
    tokenizer = AutoTokenizer.from_pretrained(
        config["shared_endpoint"]["model_directory"], local_files_only=True,
    )
    for label in method.LABELS:
        if len(tokenizer.encode(label, add_special_tokens=False)) != 1:
            raise ValueError("Every Jev control label must be one native backbone token")
    scorer = official_scorer({
        "scorer": str(root / "inputs/mrqa_official_eval.py"),
        "scorer_sha256": config["scorer_sha256"],
    })
    ordered = [
        identity for group in partition["train"]["groups"]
        for identity in sorted(partition["groups"][group])
    ]
    if len(ordered) != 3395 or len(set(ordered)) != 3395 or set(ordered) != set(items):
        raise ValueError("No missing, duplicated or selectively routed TRAIN task is allowed")
    return items, labels, partition, read(root / "inputs/RANKER_FROZEN.json"), profiles, (
        router, original, tokenizer, scorer, ordered, decoded,
    )


def score(answer, labels, router, scorer):
    prediction = router.campaign.extract_answer(answer)
    return {
        "prediction": prediction,
        "em": int(scorer.metric_max_over_ground_truths(scorer.exact_match_score, prediction, labels)),
        "f1": scorer.metric_max_over_ground_truths(scorer.f1_score, prediction, labels),
    }


def train(root, config):
    from scripts import figure2_searchqa_ranker as ranker

    verify_freeze(root)
    with (root / "owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "STARTED.json").exists():
            raise RuntimeError("Adopt an existing native owner; never duplicate its paid calls")
        state = {
            "pid": os.getpid(), "hostname": socket.gethostname(), "root": str(root),
            "phase": "LOADING_FROZEN_TRAIN", "completed_cells": 0, "target_cells": 3395,
            "at_utc": now(), "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        }
        save(root / "STARTED.json", {
            **state, "argv": sys.argv, "source_freeze_sha256": digest(root / "FROZEN.json"),
            "python": sys.version, "packages": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True),
            "request_seed": config["request_seed"], "split_seed": config["split_seed"],
            "implementation_identity": method.IDENTITY,
        })
        lock, stopped = threading.Lock(), threading.Event()

        def heartbeat():
            while not stopped.is_set():
                with lock:
                    save(root / "HEARTBEAT.json", {**state, "at_utc": now(), "at_unix": time.time()})
                stopped.wait(5)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            items, labels, partition, predictor, profiles, rest = load_inputs(root, config)
            router, original, tokenizer, scorer, ordered, decoded = rest
            provider = Provider(root, config, tokenizer)
            rows = {}
            (root / "tasks").mkdir()
            state["phase"] = "FIRST_NATIVE_COMPLETE_TASK_ACCOUNTING_GATE"

            def execute(identity):
                result = policy(items[identity], predictor, router, profiles, original, provider, config)
                row = {
                    "id": identity, "arm": config["candidate"], "split": "train", **result,
                    **score(result["answer"], labels[identity], router, scorer),
                }
                save(root / "tasks" / (ranker.digest(identity) + ".json"), row)
                with lock:
                    if identity in rows:
                        raise ValueError("Duplicate paid native outcome")
                    rows[identity] = row
                    state["completed_cells"] = len(rows)

            execute(ordered[0])
            first = rows[ordered[0]]
            save(root / "FIRST_NATIVE_RECEIPT.json", {
                "status": "PASS_REAL_NATIVE_COMPLETE_TASK_ACCOUNTING",
                "at_utc": now(), "id": first["id"], "calls": first["calls"],
                "usage": first["usage"], "em": first["em"], "f1": first["f1"],
                "jev_executed": first["decision"] is not None,
                "native_label": (first["decision"] or {}).get("native_label"),
                "first_task_included_exactly_once_in_full_train": True,
            })
            with lock:
                state["phase"] = "RUNNING_COMPLETE_3395_NATIVE_TRAIN"
            with ThreadPoolExecutor(
                max_workers=config["shared_endpoint"]["maximum_client_workers"],
            ) as pool:
                list(pool.map(execute, ordered[1:]))
            if set(rows) != set(ordered):
                raise ValueError("Incomplete canonical TRAIN population")
            summary = ranker.summary([rows[identity] for identity in ordered])
            accounting = read(root / "PROVIDER_ACCOUNTING.json")
            if summary["actor_input_tokens"] + summary["actor_output_tokens"] != accounting["total_tokens"]:
                raise ValueError("All provider attempts must belong to complete task costs")
            controls = read(config["control_receipt"])
            gate = method.training_gate(summary, controls["verified_controls"], accounting)
            result = {
                "status": "COMPLETE_3395_NATIVE_TRAIN", "candidate": config["candidate"],
                "implementation_identity": method.IDENTITY, "summary": summary,
                "provider_usage": accounting, "point_gate": gate, "independent_audit_pending": True,
                "stages": dict(Counter(
                    entry["stage"] for row in rows.values() for entry in row["calls"])),
                "paid_recovery_tasks": sum(row["paid_recovery"] for row in rows.values()),
                "explicit_jev_decision_tasks": sum(row["decision"] is not None for row in rows.values()),
                "publisher_train_records_decoded": decoded, "validation_test_records_decoded": 0,
                "partition_sha256": config["partition_sha256"],
                "new_development_model_calls": 0, "inherited_development": config["inherited_development"],
                "maximum_cumulative_tokens": None, "token_usage_mode": "unlimited",
                "validation_calls": 0, "test_calls": 0, "paper_admitted": False,
            }
            save(root / "TRAIN_RESULT.json", result)
            state["phase"] = "COMPLETE_3395_NATIVE_TRAIN"
            save(root / "TERMINAL.json", {
                "status": state["phase"], "pid": os.getpid(), "at_utc": now(),
                "complete_tasks": len(rows), "shared_service_owner_preserved": True,
            })
        except Exception as error:
            state.update(phase="FAILED", error=repr(error))
            failure = {**state, "at_utc": now(), "traceback": traceback.format_exc()}
            save(root / "FAILURE.json", failure)
            save(root / "TERMINAL.json", {**failure, "status": "FAILED"})
            raise
        finally:
            stopped.set()
            thread.join()
            save(root / "HEARTBEAT.json", {**state, "at_utc": now(), "at_unix": time.time()})


def audit(root, config, *, output=None):
    from scripts import audit_searchqa_contrast_native as controls_helper
    from scripts import figure2_searchqa_ranker as ranker
    from scripts.audit_searchqa_span_union import check_summary
    from scripts.searchqa_official_admission import identity_gate, summarize

    output = output or root
    verify_freeze(root)
    if (output / "AUDIT.json").exists():
        return read(output / "AUDIT.json")
    if read(root / "TERMINAL.json")["status"] != "COMPLETE_3395_NATIVE_TRAIN":
        raise ValueError("Independent audit requires the complete native TRAIN terminal")
    items, labels, partition, predictor, profiles, rest = load_inputs(root, config)
    router, original, tokenizer, scorer, ordered, decoded = rest
    rows = [read(path) for path in sorted((root / "tasks").glob("*.json"))]
    if len(rows) != len(ordered) or {row["id"] for row in rows} != set(ordered):
        raise ValueError("Independent native audit found a missing or duplicate task")
    seen, hashes = set(), {}
    counter = lambda messages: len(tokenizer.apply_chat_template(  # noqa: E731
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    ))

    class Replay:
        def __init__(self, row):
            self.row, self.index = row, 0

        count = staticmethod(counter)

        def call(self, identity, stage, messages, maximum=512, extra=None):
            entry = self.row["calls"][self.index]
            self.index += 1
            path = root / "calls" / (entry["call_id"] + ".json")
            if entry["call_id"] in seen:
                raise ValueError("A provider receipt was charged to more than one complete task")
            seen.add(entry["call_id"])
            record = read(path)
            body = request_body(config, messages, maximum, extra)
            if (record["id"] != identity or record["stage"] != stage or record["request"] != body
                    or record["status"] != "COMPLETE" or record["usage"] != entry["usage"]
                    or record["raw"]["model"] != body["model"]
                    or record["expected_prompt_tokens"] != counter(messages)
                    or record["usage"]["prompt_tokens"] != counter(messages)):
                raise ValueError("Native request/response/source/usage replay differs")
            hashes[str(path)] = digest(path)
            return record["raw"]["choices"][0]["message"]["content"], record

    for row in rows:
        if row["split"] != "train" or row["arm"] != config["candidate"]:
            raise ValueError("Wrong method identity or protected task in native audit")
        replay = Replay(row)
        reconstructed = canonical_record(policy(
            items[row["id"]], predictor, router, profiles, original, replay, config,
        ))
        if (replay.index != len(row["calls"])
                or any(reconstructed[key] != row[key] for key in reconstructed)
                or score(row["answer"], labels[row["id"]], router, scorer) != {
                    key: row[key] for key in ("prediction", "em", "f1")}):
            raise ValueError("CoT/Jev/fallback or official EM/F1 reconstruction differs")
    if {path.stem for path in (root / "calls").glob("*.json")} != seen:
        raise ValueError("Orphan, failed or uncharged native provider attempt")
    accounting = method.complete_usage(read(root / "calls" / (key + ".json")) for key in sorted(seen))
    if accounting != read(root / "PROVIDER_ACCOUNTING.json"):
        raise ValueError("Raw native provider ledger differs from online accounting")
    controls_helper.RUN = root
    controls_helper.ADMISSION = Path(config["control_receipt"])
    controls_helper.ADMISSION_SHA = config["control_receipt_sha256"]
    receipt, reference_rows = controls_helper.verified_controls(
        config, set(ordered), summarize, check_summary, hashes,
    )
    for name in method.CONTROLS:
        identity_gate(name, receipt["identities"].get(name, {}))
    summary = summarize(rows, set(ordered))
    reported = read(root / "TRAIN_RESULT.json")["summary"]
    check_summary(native_summary_view(summary, reported), reported)
    gate = method.training_gate(summary, receipt["verified_controls"], accounting)
    result = {
        "schema": "searchqa-strict-jev-native-audit/v1", "status": "PASS_COMPLETE_NATIVE_REPLAY_AUDIT",
        "at_utc": now(), "native_root": str(root), "candidate": config["candidate"],
        "implementation_identity": method.IDENTITY, "tasks": len(rows), "summary": summary,
        "formal_gate": gate, "native_usage": accounting,
        "explicit_jev_decision_tasks": sum(row["decision"] is not None for row in rows),
        "paid_recovery_tasks": sum(row["paid_recovery"] for row in rows),
        "stages": dict(Counter(entry["stage"] for row in rows for entry in row["calls"])),
        "matched_controls": receipt["verified_controls"], "control_identities": receipt["identities"],
        "control_costs_separate": receipt["costs"],
        "control_receipt_sha256": config["control_receipt_sha256"],
        "control_populations_reused_exactly_once": True,
        "paired_uncertainty": {
            name: ranker.paired(rows, values, partition["groups"], config["split_seed"], 1000)
            for name, values in reference_rows.items()
        },
        "native_evidence_hashes_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_freeze_sha256": digest(root / "FROZEN.json"),
        "model_manifest_sha256": digest(root / "inputs/MODEL_MANIFEST.json"),
        "gpu_selection_sha256": digest(root / "GPU_SELECTION.json"),
        "publisher_train_records_decoded": decoded, "validation_test_records_decoded": 0,
        "new_audit_model_calls": 0, "native_task_scores_recomputed": len(rows),
        "inherited_development": config["inherited_development"],
        "maximum_cumulative_tokens": None, "token_usage_mode": "unlimited",
        "validation_calls": 0, "test_calls": 0, "paper_admitted": False,
        "next_trigger": (
            "Parent alone decides matched protected selection after this complete TRAIN gate; "
            "do not train on validation trajectories or launch a width/threshold sweep."
        ),
    }
    save(output / "AUDIT.json", result)
    save(output / "DELIVERY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = read(root / "config.json")
    require_remote(config, root)
    if args.action == "train":
        train(root, config)
    else:
        try:
            result = audit(root, config)
            print(json.dumps({key: result[key] for key in ("status", "summary", "formal_gate")}))
        except Exception as error:
            save(root / "AUDIT_FAILURE.json", {
                "at_utc": now(), "error": repr(error), "traceback": traceback.format_exc(),
                "new_model_calls": 0, "native_evidence_preserved": True,
            })
            raise


if __name__ == "__main__":
    main()
