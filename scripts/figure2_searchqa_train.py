"""Remote-only supervised SearchQA training with a durable GPU owner/monitor."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import importlib.metadata
import importlib.util
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
import traceback
import types
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import figure2_searchqa_ranker as ranker


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def snapshot():
    return {"at_unix": time.time(), "hostname": socket.gethostname(),
            "gpus": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                                             "--format=csv,noheader,nounits"], text=True),
            "processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                                                  "--format=csv,noheader,nounits"], text=True)}


def acquire(config, run, state):
    roots = ["/opt/phiagent/gpu-leases", "/opt/phiagent/runs/video-skillopt/gpu-leases",
             "/opt/phiagent/runs/video-skillopt-gpu-leases",
             "/opt/phiagent/runs/video-three-track-minimax-rv2v/gpu-leases"]
    names = [f"physical-gpu-{config['gpu_index']}.lock",
             f"physical-{config['gpu_uuid']}.lock", f"physical-GPU-{config['gpu_uuid']}.lock"]
    deadline = time.monotonic() + 86400
    while time.monotonic() < deadline:
        descriptors = []
        try:
            for root in roots:
                for name in names:
                    path = Path(root) / name
                    if not path.parent.exists():
                        continue
                    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                    descriptors.append(fd)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            resources = snapshot()
            row = next(line.split(",") for line in resources["gpus"].splitlines()
                       if line.split(",")[0].strip() == str(config["gpu_index"]))
            if row[1].strip() != config["gpu_uuid"]:
                raise ValueError("physical GPU UUID changed")
            if int(row[2]) > 100 or config["gpu_uuid"] in resources["processes"]:
                raise BlockingIOError("authorized physical GPU busy")
            save(run / "GPU_SELECTION.json", {**resources, "CUDA_VISIBLE_DEVICES": str(config["gpu_index"]),
                                              "lease_paths": [str(Path(r)/n) for r in roots for n in names],
                                              "owner_pid": os.getpid()})
            return descriptors
        except BlockingIOError as error:
            for fd in descriptors:
                os.close(fd)
            state.update(phase="WAITING_AUTHORIZED_GPU2", blocker=str(error))
            save(run / "WAITING.json", {**state, "at_unix": time.time()})
            time.sleep(60)
        except Exception:
            for fd in descriptors:
                os.close(fd)
            raise
    raise TimeoutError("authorized GPU2 unavailable for 24 hours; no job preempted")


def official_scorer(config):
    if sha(config["scorer"]) != config["scorer_sha256"]:
        raise ValueError("official scorer changed")
    # The pinned native scorer only needs cached_path for CLI file acquisition.
    stub = types.ModuleType("allennlp.common.file_utils")
    stub.cached_path = lambda path: str(Path(path).resolve(strict=True))
    sys.modules["allennlp.common.file_utils"] = stub
    spec = importlib.util.spec_from_file_location("figure2_native_mrqa", config["scorer"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(config, run, state):
    if sha(config["archive"]) != config["archive_sha256"]:
        raise ValueError("publisher archive changed")
    items = []
    with gzip.open(config["archive"], "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for qa in record.get("qas", []):
                items.append({"id": qa["qid"], "question": qa["question"], "context": record["context"]})
    if len(items) != 16980:
        raise ValueError("complete official development population required")
    split = ranker.partition(items, config["split_seed"])
    save(run / "PARTITION.json", split)
    train_ids = set(split["train"]["ids"])
    train = [item for item in items if item["id"] in train_ids]
    del items
    labels = {}
    with gzip.open(config["archive"], "rt") as stream:
        for line in stream:
            for qa in json.loads(line).get("qas", []):
                if qa["qid"] in train_ids:
                    labels[qa["qid"]] = qa["answers"]
    ordered = split["train"]["groups"]
    if len(ordered) <= config["fit_groups"] + config["gate_groups"]:
        raise ValueError("independent-group inventory too small for frozen training gate")
    fit_ids = {identity for group in ordered[:config["fit_groups"]] for identity in split["groups"][group]}
    gate_ids = {identity for group in ordered[config["fit_groups"]:config["fit_groups"]+config["gate_groups"]]
                for identity in split["groups"][group]}
    state.update(phase="FITTING_TRAIN_RANKER", training_tasks=len(train))
    predictor = ranker.fit([item for item in train if item["id"] in fit_ids],
                           {identity: labels[identity] for identity in fit_ids})
    save(run / "RANKER_FROZEN.json", predictor)
    save(run / "CANDIDATES_FROZEN.json", {
        "system": ranker.SYSTEM, "input_caps": config["input_caps"], "output_cap": 512,
        "ranker_sha256": sha(run / "RANKER_FROZEN.json"), "gate_ids": sorted(gate_ids),
        "train_ids": sorted(train_ids), "validation_labels_opened": False, "test_labels_opened": False,
        "baseline_identity": "No Skill and exact archived routed CHISEL/no-echo; official optimizers pending parent",
        "selection": "TRAIN gate only; no validation/test execution is implemented",
        "source_sha256": sha(Path(__file__)), "ranker_source_sha256": sha(Path(ranker.__file__)),
        "prior_negative": "Full-source prompt compaction: 75.68 EM/1685.81 tokens; no repeated fixed BM25 or output truncation.",
    })
    return train, labels, split, predictor, gate_ids


def study(config, run, state, train, labels, split, predictor, gate_ids):
    from transformers import AutoTokenizer
    from phiagent.learning import qa_meta_rsi, qa_typed_recovery
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel
    reference = Path(config["reference_directory"])
    parent = json.loads((reference / "lineage/config.json").read_text())
    profiles = json.loads((reference / "PROFILES.json").read_text())
    for arm in ("no_skill", "delete", "modify"):
        if ranker.digest(profiles[arm]) != parent["prompt_sha256"][arm]:
            raise ValueError("archived control prompt differs")
    router = FrozenOriginalChisel(reference / "lineage", parent)
    original = qa_meta_rsi.initial_skill({arm: profiles[arm] for arm in ("delete", "modify")})
    save(run / "CONTROL_BINDINGS.json", {"profiles_sha256": parent["prompt_sha256"], "router": router.bindings,
                                         "recovery_sha256": sha(Path(qa_typed_recovery.__file__))})
    scorer = official_scorer(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_directory"], local_files_only=True)

    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False, return_dict=False))

    arms = ["no_skill", "chisel_current"] + [f"ranker_{cap}" for cap in config["input_caps"]]
    results = {arm: {} for arm in arms}
    lock = threading.Lock()
    ledger = {"calls": 0, "known_tokens": 0, "failed_calls": 0, "unknown_usage_calls": 0,
              "token_usage_mode": "unlimited", "maximum_total_provider_tokens": None}
    (run / "calls").mkdir(exist_ok=True)
    (run / "tasks").mkdir(exist_ok=True)

    def call(identity, arm, stage, messages):
        call_id = ranker.digest(f"{identity}:{arm}:{stage}")
        path = run / "calls" / f"{call_id}.json"
        if path.exists():
            raise ValueError("duplicate request identity; frozen runs cannot silently retry")
        body = {"model": config["model_repository"], "messages": messages, "temperature": 0,
                "max_tokens": 512, "seed": config["request_seed"],
                "chat_template_kwargs": {"enable_thinking": False}}
        expected = count(messages)
        record = {"id": identity, "arm": arm, "stage": stage, "request": body, "expected_prompt_tokens": expected}
        save(path, {**record, "status": "REQUEST_INTENT"})
        try:
            request = Request(f"http://127.0.0.1:{config['port']}/v1/chat/completions",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with build_opener(ProxyHandler({})).open(request, timeout=300) as response:
                raw = json.load(response)
            record["raw"] = raw
            usage = raw.get("usage", {})
            if (raw.get("model") != config["model_repository"]
                    or any(type(usage.get(k)) is not int or usage[k] < 0
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["prompt_tokens"] != expected
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ValueError("provider identity or complete native usage mismatch")
            record.update(status="COMPLETE", usage=usage)
            save(path, record)
            with lock:
                ledger["calls"] += 1
                ledger["known_tokens"] += usage["total_tokens"]
                save(run / "PROVIDER_ACCOUNTING.json", ledger)
            return raw["choices"][0]["message"]["content"], usage, call_id
        except Exception as error:
            save(path, {**record, "status": "FAILED", "error": repr(error)})
            with lock:
                ledger["calls"] += 1
                ledger["failed_calls"] += 1
                observed = record.get("raw", {}).get("usage", {})
                if isinstance(observed.get("total_tokens"), int):
                    ledger["known_tokens"] += observed["total_tokens"]
                else:
                    ledger["unknown_usage_calls"] += 1
                save(run / "PROVIDER_ACCOUNTING.json", ledger)
            raise

    def execute(item):
        for arm in arms:
            if item["id"] in results[arm]:
                continue
            records, receipt = [], None

            def request(messages, stage):
                text, usage, identity = call(item["id"], arm, stage, messages)
                records.append({"usage": usage, "call_id": identity})
                return text

            if arm == "no_skill":
                answer = request(ranker.messages(item, item["context"], profiles["no_skill"]), "actor")
            elif arm == "chisel_current":
                branch = router.route(item)["variant"]
                full = ranker.messages(item, item["context"], profiles[branch])
                fallback = lambda: request(full, "original_fallback")
                if qa_typed_recovery.preflight_route(item["question"]) != "compact_candidate":
                    answer = fallback()
                else:
                    system, _ = qa_typed_recovery.retrieve_source_rules(
                        original, profile=branch, question=item["question"], context=item["context"])
                    draft = request(ranker.messages(item, item["context"], system), "full_source_draft")
                    answer = qa_typed_recovery.recover_answer(
                        question=item["question"], context=item["context"], draft=draft,
                        decide=lambda s: qa_typed_recovery.deterministic_recovery(s)["action"],
                        original=fallback, reject_question_echo=False)["answer"]
            else:
                messages, receipt = ranker.select(item, predictor, int(arm.split("_")[-1]), count)
                answer = request(messages, "ranked_source_actor")
            prediction = router.campaign.extract_answer(answer)
            gold = labels[item["id"]]
            usage = {k: sum(record["usage"][k] for record in records)
                     for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            row = {"id": item["id"], "arm": arm, "answer": answer, "prediction": prediction,
                   "em": int(scorer.metric_max_over_ground_truths(scorer.exact_match_score, prediction, gold)),
                   "f1": scorer.metric_max_over_ground_truths(scorer.f1_score, prediction, gold),
                   "usage": usage, "calls": records, "source_receipt": receipt, "split": "train"}
            save(run / "tasks" / f"{ranker.digest(item['id'] + ':' + arm)}.json", row)
            with lock:
                results[arm][item["id"]] = row
                state["completed_cells"] = sum(map(len, results.values()))

    def evaluate(items):
        with ThreadPoolExecutor(max_workers=config["execution_workers"]) as pool:
            for _ in pool.map(execute, items):
                pass

    state.update(phase="TRAIN_GATE", gate_tasks=len(gate_ids))
    evaluate([item for item in train if item["id"] in gate_ids])
    gate = {arm: ranker.summary(list(rows.values())) for arm, rows in results.items()}
    viable = [arm for arm in arms[2:]
              if gate[arm]["em_percent"] >= max(gate["no_skill"]["em_percent"], gate["chisel_current"]["em_percent"])
              - config["gate_maximum_em_loss_pp"]
              and gate[arm]["mean_actor_tokens"] <= config["gate_maximum_cost_ratio"]*gate["no_skill"]["mean_actor_tokens"]]
    gate_uncertainty = {
        arm: {control: ranker.paired(list(results[arm].values()), list(results[control].values()), split["groups"])
              for control in arms[:2]} for arm in arms[2:]}
    save(run / "TRAIN_GATE.json", {"methods": gate, "viable": viable, "paired_uncertainty": gate_uncertainty,
                                   "criterion": "screen, not provisional acceptance",
                                   "validation_or_test_opened": False, "all_failed_calls_counted": ledger["failed_calls"]})
    if not viable:
        state.update(phase="NEGATIVE_TRAIN_GATE", completed=True)
        return
    state.update(phase="COMPLETE_TRAIN", viable=viable)
    evaluate(train)
    methods = {arm: ranker.summary(list(rows.values())) for arm, rows in results.items()}
    uncertainty = {arm: {control: ranker.paired(list(results[arm].values()), list(results[control].values()), split["groups"])
                        for control in arms[:2]} for arm in arms[2:]}
    if ledger["unknown_usage_calls"] or ledger["failed_calls"]:
        raise ValueError("failed or unaccounted calls prevent a complete training result")
    accounting = sum(row["usage"]["total_tokens"] for rows in results.values() for row in rows.values())
    if accounting != ledger["known_tokens"]:
        raise ValueError("provider ledger does not reconcile with complete tasks")
    save(run / "TRAIN_RESULT.json", {"status": "COMPLETE_TRAIN_ONLY", "methods": methods,
                                     "paired_uncertainty": uncertainty, "optimizer_cost": ledger,
                                     "final_test_execution_tokens": 0, "paper_admitted": False,
                                     "official_optimizer_baselines": "PENDING_PARENT",
                                     "validation_and_test_sealed": True})
    save(run / "SUCCESSOR.json", {"status": "AWAIT_PARENT_ALL_METHOD_TRAINING",
                                 "candidates_sha256": sha(run / "CANDIDATES_FROZEN.json"),
                                 "partition_sha256": sha(run / "PARTITION.json"),
                                 "validation": "scalar selection among frozen TRAIN candidates only",
                                 "test": "requires parent all-method policy/source freeze; not executable here"})
    state.update(phase="COMPLETE_TRAIN_ONLY", completed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if socket.gethostname() != config["hostname"]:
        raise RuntimeError("remote-only run requires the exact authorized H200 hostname")
    if (config["budget"]["maximum_total_provider_tokens"] is not None
            or config["budget"]["token_usage_mode"] != "unlimited" or config["max_output_tokens"] != 512):
        raise ValueError("unlimited cumulative accounting and native output allowance required")
    run = Path(config["run_directory"])
    run.mkdir(parents=True, exist_ok=True)
    ownership = (run / "owner.lock").open("a")
    fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "STARTED.json").exists():
        raise RuntimeError("frozen run already started; a repaired attempt requires a new run directory")
    state = {"pid": os.getpid(), "phase": "PREPARING", "completed": False, "completed_cells": 0}
    save(run / "STARTED.json", {**state, "at_unix": time.time(), "hostname": socket.gethostname(),
                               "argv": sys.argv, "python": sys.version, "config": config,
                               "packages": {k: importlib.metadata.version(k) for k in ("vllm", "torch", "transformers")},
                               "config_sha256": sha(args.config)})
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            save(run / "HEARTBEAT.json", {**state, "at_unix": time.time()})
            stop.wait(15)

    threading.Thread(target=heartbeat, daemon=True).start()
    descriptors, server = [], None
    try:
        if config.get("predecessor"):
            prior = Path(config["predecessor"])
            failure = json.loads((prior / "FAILURE.json").read_text())
            if failure["completed_cells"] != 0 or list((prior / "calls").glob("*.json")):
                raise ValueError("this continuation permits only a zero-call startup predecessor")
            save(run / "PRIOR_ATTEMPT.json", {
                "root": str(prior), "failure_sha256": sha(prior / "FAILURE.json"),
                "service_log_sha256": sha(prior / "service.log"),
                "model_calls": 0, "provider_tokens": 0,
                "repair": config.get("repair_note", "shorten project-local IPC cache path; preserve native answer extraction")})
        prepared = prepare(config, run, state)
        manifest = json.loads(Path(config["model_manifest"]).read_text())
        if (manifest["repository"], manifest["revision"]) != (config["model_repository"], config["model_revision"]):
            raise ValueError("checkpoint revision mismatch")
        state["phase"] = "VERIFYING_CHECKPOINT"
        for name, expected in manifest["files_sha256"].items():
            if sha(Path(config["model_directory"])/name) != expected:
                raise ValueError(f"checkpoint file changed: {name}")
        save(run / "MODEL_VERIFIED.json", manifest)
        descriptors = acquire(config, run, state)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(config["gpu_index"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
               "TMPDIR": str(run / "c"), "HF_HOME": str(run / "c"),
               "VLLM_CACHE_ROOT": str(run / "c/vllm"),
               "PATH": str(Path(config["runtime_python"]).parent) + os.pathsep + os.environ.get("PATH", "")}
        if len(env["TMPDIR"].encode()) + 37 > 107:
            raise ValueError("project-local IPC socket path exceeds the Unix path limit")
        if not shutil.which("ninja", path=env["PATH"]):
            raise RuntimeError("the native runtime must expose its existing ninja executable")
        Path(env["TMPDIR"]).mkdir(exist_ok=True)
        command = [config["runtime_python"], "-m", "vllm.entrypoints.openai.api_server",
                   "--model", config["model_directory"], "--served-model-name", config["model_repository"],
                   "--host", "127.0.0.1", "--port", str(config["port"]), "--dtype", "bfloat16",
                   "--tensor-parallel-size", "1", "--max-model-len", "32768", "--max-num-seqs", "16",
                   "--gpu-memory-utilization", "0.45", "--generation-config", "vllm",
                   "--max-num-batched-tokens", "4096", "--enforce-eager", "--language-model-only",
                   "--attention-config", '{"backend":"FLASH_ATTN","flash_attn_version":3}',
                   "--gdn-prefill-backend", "triton", "--seed", str(config["request_seed"]),
                   "--default-chat-template-kwargs", '{"enable_thinking":false}']
        save(run / "SERVICE_COMMAND.json", {"argv": command, "env": {k: env[k] for k in
             ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE", "TMPDIR", "PATH")}})
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", config["port"]))
        with (run / "service.log").open("x") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, pass_fds=tuple(descriptors))
        state.update(phase="SERVICE_STARTING", server_pid=server.pid)
        save(run / "SERVICE_PID.json", {"pid": server.pid})
        deadline = time.monotonic()+900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"service exited {server.returncode}; see service.log")
            try:
                with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{config['port']}/v1/models", timeout=3) as response:
                    models = json.load(response)
                if models["data"][0]["id"] != config["model_repository"]:
                    raise ValueError("served model identity mismatch")
                save(run / "SERVICE_READY.json", models)
                break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("service did not become ready within 900s")
        study(config, run, state, *prepared)
    except Exception as error:
        state.update(phase="FAILED", error=repr(error), completed=True)
        save(run / "FAILURE.json", {**state, "traceback": traceback.format_exc()})
        raise
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        for fd in descriptors:
            os.close(fd)
        stop.set()
        save(run / "HEARTBEAT.json", {**state, "at_unix": time.time(), "gpu_released": True})
        save(run / "TERMINAL.json", state)


if __name__ == "__main__":
    main()
