"""Read-only reconstruction of the pinned complete contrastive TRAIN run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback


BASE = Path("/opt/phiagent/work/figure2-token-efficiency")
RUN = BASE / "20261002-sq-as-ct3"
CONFIG_SHA = "37e4db9077a27a837f50d51979472f4679da8d2d75aeb49891c10f5191d99e9c"
FREEZE_SHA = "003a17f635511c429444fa5a742fd68fcc71c08832a1e104b292c3f6100d6183"
ADMISSION = BASE / "20261002-sq-ar1/admission/RECEIPT.json"
ADMISSION_SHA = "a8c32d2fb941f3e36c4621d296c2be0bcd5717f63d9e2b64867baba975d7e4fc"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
USAGE = ("prompt_tokens", "completion_tokens", "total_tokens")
CONTROLS = ("no_skill", "chisel_current", "skillopt", "autosaddler")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_train_record(line, ids):
    """Decode only records whose explicit qids belong to the canonical TRAIN set."""
    present = {value.decode("utf-8") for value in re.findall(
        rb'"qid"\s*:\s*"([^"\\]+)"', line,
    )}
    if not present.intersection(ids):
        return None
    require(present <= ids, "A source record crosses the canonical TRAIN boundary")
    record = json.loads(line)
    require({qa["qid"] for qa in record["qas"]} == present, "Unexpected qid encoding")
    return record


def terminal_decision(terminal, has_result, failure, owner_alive):
    if failure is not None or (terminal and terminal.get("phase") == "FAILED"):
        return "NATIVE_RUN_FAILED"
    if (
        terminal and terminal.get("completed") is True
        and terminal.get("phase") == "COMPLETE_3395_NATIVE_TRAIN" and has_result
    ):
        return "READY"
    return "WAITING_NATIVE_TERMINAL" if owner_alive else "OWNER_EXITED_WITHOUT_SUCCESS"


def owner_alive(pid):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        status = Path(f"/proc/{pid}/status").read_text()
        return str(RUN).encode() in command and "\nState:\tZ" not in status
    except FileNotFoundError:
        return False


def wait_for_native(output, pid):
    missing_owner_observations = 0
    while True:
        terminal = read(RUN / "TERMINAL.json") if (RUN / "TERMINAL.json").exists() else None
        failure = read(RUN / "FAILURE.json") if (RUN / "FAILURE.json").exists() else None
        alive = owner_alive(pid)
        state = terminal_decision(terminal, (RUN / "TRAIN_RESULT.json").exists(), failure, alive)
        heartbeat = read(RUN / "HEARTBEAT.json") if (RUN / "HEARTBEAT.json").exists() else {}
        save(output / "NATIVE_STATUS.json", {
            "at_utc": datetime.now(timezone.utc).isoformat(), "owner_pid": pid,
            "owner_alive_with_matching_run": alive, "state": state,
            "completed_cells": heartbeat.get("completed_cells"),
            "native_phase": heartbeat.get("phase"),
        })
        if state == "READY":
            return
        if state == "NATIVE_RUN_FAILED":
            save(output / "NATIVE_FAILURE.json", failure or terminal)
            raise RuntimeError("Native run failed; no complete result is admitted")
        missing_owner_observations = (
            missing_owner_observations + 1 if state == "OWNER_EXITED_WITHOUT_SUCCESS" else 0
        )
        if missing_owner_observations >= 2:
            raise RuntimeError("Native owner exited without terminal success; no audited result")
        time.sleep(30)


def certified_reference_bindings(proof):
    # The reference mirror carries certified rows; original raw-call blobs stay at their origin.
    return {
        **proof["reference_metadata_sha256"],
        **{name: value for name, value in proof["files_sha256"].items()
           if name.startswith("tasks/")},
    }


def verified_controls(config, ids, summarize, check_summary, hashes):
    require(sha(ADMISSION) == ADMISSION_SHA, "Certified four-method admission receipt changed")
    receipt = read(ADMISSION)
    pool = Path(receipt["locations"]["skillopt"]["root"])
    require(sha(pool / "FROZEN.json") == receipt["source_freeze_sha256"],
            "Certified official pool freeze changed")
    frozen = read(pool / "FROZEN.json")
    require(sha(pool / "REUSED_TRAIN.json") == frozen["reused_train_sha256"],
            "Certified reference reuse manifest changed")
    require(sha(pool / "config.json") == frozen["config_sha256"], "Official pool config changed")
    reference = Path(config["predecessor"])
    require(str(reference) == read(pool / "config.json")["reference_root"],
            "Original reference root differs")
    proof = read(pool / "REUSED_TRAIN.json")["reference_audit"]
    for relative, expected in certified_reference_bindings(proof).items():
        path = reference / relative
        require(sha(path) == expected, "Certified reference evidence changed: " + relative)
        hashes[str(path)] = expected
    rows = {name: [] for name in CONTROLS}
    reference_rows = list(read(reference / "inputs/REUSED_GATE.json")["rows"])
    reference_rows += [
        read(reference / name) for name in proof["files_sha256"] if name.startswith("tasks/")
    ]
    for row in reference_rows:
        require(row["split"] == "train" and row["id"] in ids, "Non-TRAIN reference outcome")
        if row["arm"] in ("no_skill", "chisel_current"):
            rows[row["arm"]].append(row)
    for method in ("skillopt", "autosaddler"):
        directory = ADMISSION.parent.parent / "certificates" / method
        certificate = read(directory / "CERTIFICATE.json")
        expected = receipt["locations"][method]["rows_sha256"]
        require(sha(directory / "ROWS.json") == expected == certificate["rows_sha256"],
                "Certified official TRAIN outcome rows changed")
        require(
            certificate["status"] == "PASS_INDEPENDENT_COMPLETE_OFFICIAL_TRAIN"
            and certificate["identity"] == receipt["identities"][method]
            and certificate["partition_sha256"] == config["partition_sha256"]
            and certificate["model_manifest_sha256"] == sha(RUN / "inputs/MODEL_MANIFEST.json")
            and not certificate["validation_calls"] and not certificate["test_calls"],
            "Official TRAIN certificate protocol differs",
        )
        rows[method] = read(directory / "ROWS.json")
        hashes[str(directory / "ROWS.json")] = expected
        hashes[str(directory / "CERTIFICATE.json")] = sha(directory / "CERTIFICATE.json")
    for method in CONTROLS:
        check_summary(summarize(rows[method], ids), receipt["verified_controls"][method])
    hashes[str(ADMISSION)] = ADMISSION_SHA
    hashes[str(pool / "FROZEN.json")] = receipt["source_freeze_sha256"]
    hashes[str(pool / "REUSED_TRAIN.json")] = frozen["reused_train_sha256"]
    return receipt, rows


def official_admission_helper():
    require(sha(ADMISSION) == ADMISSION_SHA, "Certified admission receipt changed")
    receipt = read(ADMISSION)
    pool = Path(receipt["locations"]["skillopt"]["root"])
    require(sha(pool / "FROZEN.json") == receipt["source_freeze_sha256"],
            "Official admission helper's source freeze changed")
    relative = "scripts/searchqa_official_admission.py"
    path = pool / "source" / relative
    require(sha(path) == read(pool / "FROZEN.json")["source_sha256"][relative],
            "Certified admission helper changed")
    spec = importlib.util.spec_from_file_location("contrast_audit_official_admission", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, path


def audit(output):
    require(sha(RUN / "pool-config.json") == CONFIG_SHA, "Pinned candidate configuration changed")
    require(sha(RUN / "SUCCESSOR_FREEZE.json") == FREEZE_SHA, "Pinned source freeze changed")
    config, frozen = read(RUN / "pool-config.json"), read(RUN / "SUCCESSOR_FREEZE.json")
    hashes = {}
    for relative, expected in frozen["source_sha256"].items():
        path = RUN / "source" / relative
        require(sha(path) == expected, "Frozen source changed: " + relative)
        hashes[str(path)] = expected
    for relative, expected in read(RUN / "INPUTS_FROZEN.json").items():
        path = RUN / relative
        require(sha(path) == expected, "Frozen input changed: " + relative)
        hashes[str(path)] = expected
    require(
        config["scope"] == "train" and config["training_tasks"] == 3395
        and config["model_repository"] == "Qwen/Qwen3.5-9B"
        and config["model_revision"] == REVISION and config["request_seed"] == 20260918
        and config["source_selection_input_cap"] == 480 and config["output_cap"] == 512
        and config["token_usage_mode"] == "unlimited"
        and config["maximum_total_provider_tokens"] is None,
        "Prospective native protocol changed",
    )
    sys.path.insert(0, str(RUN / "source"))
    from scripts import figure2_searchqa_ranker as ranker
    from scripts import figure2_searchqa_v4 as native
    from scripts import run_searchqa_autosota_contrast as contrast
    from scripts.audit_searchqa_span_union import check_summary, check_usage
    from scripts.figure2_searchqa_train import official_scorer
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel
    from scripts.run_searchqa_span_union import inherited_surface_rule
    from phiagent.learning import qa_meta_rsi
    from transformers import AutoTokenizer

    helper, helper_path = official_admission_helper()
    admission, summarize = helper.admission, helper.summarize
    hashes[str(helper_path)] = sha(helper_path)
    require(Path(contrast.__file__).is_relative_to(RUN / "source"), "Wrong contrast source import")
    partition = read(RUN / "inputs/PARTITION.json")
    ids = set(partition["train"]["ids"])
    require(len(ids) == 3395 and sha(RUN / "inputs/PARTITION.json") == config["partition_sha256"],
            "Canonical TRAIN membership changed")
    host = config["hosts"][socket.gethostname()]
    require(sha(host["archive"]) == config["archive_sha256"], "Publisher archive changed")
    items, labels = {}, {}
    with gzip.open(host["archive"], "rb") as stream:
        for line in stream:
            record = decode_train_record(line, ids)
            if record is None:
                continue
            for qa in record["qas"]:
                require(qa["qid"] not in items, "Duplicate canonical TRAIN question")
                items[qa["qid"]] = {
                    "id": qa["qid"], "question": qa["question"], "context": record["context"],
                }
                labels[qa["qid"]] = qa["answers"]
    require(set(items) == ids, "Missing canonical TRAIN source/label")
    manifest = read(RUN / "inputs/MODEL_MANIFEST.json")
    require(manifest == read(RUN / "MODEL_VERIFIED.json"), "Native model attestation changed")
    require((manifest["repository"], manifest["revision"]) == (config["model_repository"], REVISION),
            "Model repository/revision differs")
    for name, expected in manifest["files_sha256"].items():
        require(sha(Path(host["model_directory"]) / name) == expected, "Model asset changed: " + name)
    command = read(RUN / "SERVICE_COMMAND.json")
    for flag, value in (
        ("--model", host["model_directory"]), ("--served-model-name", config["model_repository"]),
        ("--seed", str(config["request_seed"])), ("--generation-config", "vllm"),
    ):
        require(command["argv"][command["argv"].index(flag) + 1] == value,
                "Native serving command differs: " + flag)
    require(command["CUDA_VISIBLE_DEVICES"] == read(RUN / "GPU_SELECTION.json")["gpu_uuid"],
            "Native physical GPU binding differs")
    tokenizer = AutoTokenizer.from_pretrained(host["model_directory"], local_files_only=True)

    def count(messages):
        return len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        ))

    freeze, result = read(RUN / "CANDIDATES_FROZEN.json"), read(RUN / "TRAIN_RESULT.json")
    require(result["status"] == "COMPLETE_3395_NATIVE_TRAIN"
            and not result["validation_calls"] and not result["test_calls"], "Incomplete TRAIN result")
    predictor = read(RUN / "inputs/RANKER_FROZEN.json")
    reference = RUN / "inputs/reference"
    profiles = read(reference / "PROFILES.json")
    lineage = read(reference / "lineage/config.json")
    require(all(ranker.digest(profiles[name]) == lineage["prompt_sha256"][name]
                for name in ("no_skill", "delete", "modify")), "Reference prompts changed")
    extra = inherited_surface_rule(profiles) if config["inherit_surface_rule"] else ""
    require(
        freeze["system"] == contrast.CONTRAST_SYSTEM + extra
        and freeze["extra_instruction"] == extra
        and freeze["source_selection_input_cap"] == 480
        and len(freeze["ids"]) == 3395 and set(freeze["ids"]) == ids
        and freeze["ranker_sha256"] == sha(RUN / "inputs/RANKER_FROZEN.json"),
        "Frozen contrast system, evidence allocator, or population differs",
    )
    router = FrozenOriginalChisel(reference / "lineage", lineage)
    original = qa_meta_rsi.initial_skill({name: profiles[name] for name in ("delete", "modify")})
    scorer = official_scorer({
        "scorer": str(RUN / "inputs/mrqa_official_eval.py"),
        "scorer_sha256": config["scorer_sha256"],
    })
    candidate = freeze["candidate"]
    require(candidate == config["family"][0] == result["candidate"], "Candidate identity changed")
    rows, used, totals, stages = {}, set(), dict.fromkeys(USAGE, 0), {}
    for path in sorted((RUN / "tasks").glob("*.json")):
        row = read(path)
        identity = row["id"]
        require(identity in ids and identity not in rows and row["split"] == "train"
                and row["arm"] == candidate, "Duplicate or non-TRAIN candidate outcome")
        item = items[identity]
        messages, receipt = contrast.contrast_request(item, predictor, count, 480, extra)
        require(json.loads(json.dumps(receipt)) == row["source_receipt"],
                "Exact contrast evidence allocation does not reproduce")
        calls = []

        def replay(request_messages, stage):
            key = ranker.digest(f"{identity}:{candidate}:{stage}")
            require(key not in used and len(calls) < len(row["calls"]), "Repeated or extra native call")
            call_path = RUN / "calls" / (key + ".json")
            call = read(call_path)
            expected_request = {
                "model": config["model_repository"], "messages": request_messages,
                "temperature": 0, "max_tokens": 512, "seed": config["request_seed"],
                "chat_template_kwargs": {"enable_thinking": False},
                "logprobs": True, "top_logprobs": 0,
            }
            require(call["id"] == identity and call["arm"] == candidate and call["stage"] == stage
                    and call["request"] == expected_request, "Exact frozen native request differs")
            usage = check_usage(call, config, count)
            binding = row["calls"][len(calls)]
            require(binding["call_id"] == key and binding["usage"] == usage
                    and binding["confidence"] == call["confidence"], "Native response binding differs")
            require(len(call["raw"]["choices"]) == 1
                    and isinstance(call["raw"]["choices"][0]["message"]["content"], str),
                    "Native answer is missing")
            for field in USAGE:
                totals[field] += usage[field]
            stages[stage] = stages.get(stage, 0) + 1
            hashes[str(call_path)] = sha(call_path)
            used.add(key)
            calls.append(call)
            return call["raw"]["choices"][0]["message"]["content"]

        answer = replay(messages, "contrastive_compact")
        recovered = contrast.contrast_contract_risk(item["question"], item["context"], answer)
        if recovered:
            answer = native.current_chisel(
                item, router, profiles, original, replay, "contract_recovery",
            )
        prediction = router.campaign.extract_answer(answer)
        em = int(scorer.metric_max_over_ground_truths(
            scorer.exact_match_score, prediction, labels[identity],
        ))
        f1 = scorer.metric_max_over_ground_truths(scorer.f1_score, prediction, labels[identity])
        usage = {field: sum(call["usage"][field] for call in calls) for field in USAGE}
        require(
            len(calls) == len(row["calls"]) and row["answer"] == answer
            and row["prediction"] == prediction and row["paid_recovery"] == recovered
            and row["em"] == em and abs(row["f1"] - f1) <= 1e-12 and row["usage"] == usage,
            "Native score, final answer, recovery flow or full task cost differs",
        )
        rows[identity] = row
        hashes[str(path)] = sha(path)
        if len(rows) % 250 == 0:
            save(output / "AUDIT_PROGRESS.json", {"phase": "NATIVE_RECONSTRUCTION", "tasks": len(rows)})
    require(set(rows) == ids and used == {p.stem for p in (RUN / "calls").glob("*.json")},
            "Incomplete population or uncharged orphan provider request")
    ledger = read(RUN / "PROVIDER_ACCOUNTING.json")
    require(ledger["calls"] == len(used) and not ledger["failed_calls"]
            and not ledger["unknown_usage_calls"] and result["provider_usage"] == ledger
            and all(ledger[field] == totals[field] for field in USAGE),
            "All failed/fallback/unknown provider costs must reconcile without omissions")
    check_summary(ranker.summary(list(rows.values())), result["summary"])
    controls, control_rows = verified_controls(config, ids, summarize, check_summary, hashes)
    summary = summarize(list(rows.values()), ids)
    gate = admission(summary, controls["verified_controls"], controls["identities"])
    paired = {
        name: ranker.paired(list(rows.values()), control_rows[name], partition["groups"])
        for name in CONTROLS
    }
    for name in (
        "pool-config.json", "SUCCESSOR_FREEZE.json", "INPUTS_FROZEN.json",
        "CANDIDATES_FROZEN.json", "TRAIN_RESULT.json", "TERMINAL.json",
        "PROVIDER_ACCOUNTING.json", "MODEL_VERIFIED.json", "GPU_SELECTION.json",
        "SERVICE_COMMAND.json", "STARTED.json",
    ):
        hashes[str(RUN / name)] = sha(RUN / name)
    save(output / "EVIDENCE_HASHES.json", hashes)
    return {
        "status": "PASS_COMPLETE_INDEPENDENT_NATIVE_TRAIN_AUDIT",
        "scope": "TRAIN_ONLY_3395", "native_root": str(RUN),
        "tasks": len(rows), "native_calls_checked": len(used), "native_usage": totals,
        "summary": summary, "stages": stages,
        "paid_recovery_tasks": sum(row["paid_recovery"] for row in rows.values()),
        "formal_named_control_gate": gate, "runner_prototype_gate": result["training_gate_passed"],
        "prototype_em_only_flag_is_not_formal_admission": True,
        "certified_controls": controls["verified_controls"],
        "certified_control_costs": controls["costs"],
        "certified_control_admission_sha256": ADMISSION_SHA,
        "control_populations_reused_exactly_once": list(CONTROLS),
        "control_task_rows_hash_verified": True,
        "control_raw_calls_replayed": 0,
        "control_original_raw_call_blobs_reaudited": False,
        "paired_uncertainty": paired, "bootstrap_draws": 1000, "bootstrap_seed": 20260928,
        "failed_calls": ledger["failed_calls"], "unknown_usage_calls": ledger["unknown_usage_calls"],
        "new_model_calls": 0, "new_provider_tokens": 0, "new_control_optimizer_tokens": 0,
        "candidate_historical_optimizer_tokens_reported_not_summed":
            result["historical_optimizer_provider_tokens"],
        "maximum_total_provider_tokens": None, "token_usage_mode": "unlimited",
        "only_train_records_json_decoded": True, "validation_test_records_decoded": False,
        "validation_calls": 0, "test_calls": 0, "paper_admitted": False,
        "evidence_hashes_sha256": sha(output / "EVIDENCE_HASHES.json"),
        "next_event": "STOP_WITH_PARENT_HANDOFF; no inference or held-out promotion",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, default=548284)
    parser.add_argument("--source-git-head", required=True)
    args = parser.parse_args()
    require(
        socket.gethostname() == "harnessmm-worker-1"
        and args.output.parent == BASE and args.output.name.startswith("20261002-sq-ct3-audit-")
        and not args.output.exists(), "Use a new authorized remote audit root",
    )
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    args.output.mkdir()
    (args.output / "runtime").mkdir()
    os.environ.update(TMPDIR=str(args.output / "runtime"), TMP=str(args.output / "runtime"),
                      TEMP=str(args.output / "runtime"))
    shutil.copy2(__file__, args.output / Path(__file__).name)
    started = time.monotonic()
    save(args.output / "STARTED.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "pid": os.getpid(), "argv": sys.argv, "python": sys.version,
        "source_sha256": sha(__file__), "native_config_sha256": CONFIG_SHA,
        "auditor_source_git_head": args.source_git_head, "auditor_source_committed": False,
        "native_source_freeze_sha256": FREEZE_SHA, "CUDA_VISIBLE_DEVICES": "",
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        "new_model_calls": 0, "validation_test_access": False,
    })
    try:
        wait_for_native(args.output, args.owner_pid)
        result = audit(args.output)
        result.update(at_utc=datetime.now(timezone.utc).isoformat(),
                      cpu_wall_seconds=time.monotonic() - started)
        save(args.output / "AUDIT.json", result)
        print(json.dumps({key: result[key] for key in (
            "status", "tasks", "native_calls_checked", "summary", "formal_named_control_gate",
        )}, indent=2, sort_keys=True))
    except Exception as error:
        save(args.output / "AUDIT_FAILURE.json", {
            "status": "NO_AUDITED_RESULT", "error": repr(error),
            "traceback": traceback.format_exc(), "new_model_calls": 0,
            "native_run_modified": False, "at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
