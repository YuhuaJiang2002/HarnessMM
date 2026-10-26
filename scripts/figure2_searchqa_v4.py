"""Frozen contract-preserving CHISEL recovery, with a disjoint TRAIN gate."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import figure2_searchqa_ranker as ranker
from scripts import figure2_searchqa_recovery as recovery
from scripts.figure2_searchqa_train import save, sha, snapshot, official_scorer
from scripts.serve_terminal_bench_model import acquire_gpu_leases


def bounded_execute(function, items, workers):
    remaining = iter(items)
    exhausted = object()
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = set()
    try:
        for _ in range(workers):
            item = next(remaining, exhausted)
            if item is not exhausted:
                pending.add(pool.submit(function, item))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
            for _ in completed:
                item = next(remaining, exhausted)
                if item is not exhausted:
                    pending.add(pool.submit(function, item))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def registry(root: Path, action: str, body: dict) -> dict:
    config = json.loads((root/"pool-config.json").read_text())
    secret = (root/"claim.key").read_text().strip()
    request = Request(config["registry_url"]+"/"+action, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer "+secret})
    with build_opener(ProxyHandler({})).open(request, timeout=10) as stream:
        return json.load(stream)


def acquire(root: Path, host: dict, state: dict):
    roots = [Path(path) for path in (
        "/opt/phiagent/gpu-leases", "/opt/phiagent/runs/video-skillopt-gpu-leases",
        "/opt/phiagent/runs/video-skillopt/gpu-leases",
        "/opt/phiagent/runs/video-three-track-minimax-rv2v/gpu-leases")]
    profile = {"leases": roots[0], "additional_leases": tuple(roots[1:])}
    while True:
        winner = registry(root, "status", {"pid": os.getpid()})["winner"]
        if winner is not None:
            state.update(phase="STANDBY_OTHER_HOST_WON", completed=True, winner=winner)
            return None, None
        inventory = snapshot()
        save(root/"RESOURCE_LATEST.json", inventory)
        rows = {line.split(",")[0].strip(): [x.strip() for x in line.split(",")]
                for line in inventory["gpus"].splitlines()}
        for index, uuid in host["eligible_gpus"].items():
            row = rows[index]
            if row[1] != uuid:
                raise ValueError("frozen physical GPU mapping changed")
            if int(row[2]) > 100 or uuid in inventory["processes"]:
                continue
            lease = None
            try:
                lease = acquire_gpu_leases(profile, int(index), uuid)
                current = snapshot()
                check = next(line.split(",") for line in current["gpus"].splitlines()
                             if line.split(",")[0].strip() == index)
                if int(check[2]) > 100 or uuid in current["processes"]:
                    lease.close()
                    continue
                body = {"hostname": socket.gethostname(), "gpu_index": int(index), "gpu_uuid": uuid,
                        "pid": os.getpid()}
                winner = registry(root, "claim", body)["winner"]
                if any(winner[k] != body[k] for k in body):
                    lease.close()
                    state.update(phase="STANDBY_OTHER_HOST_WON", completed=True, winner=winner)
                    return None, None
                save(root/"GPU_SELECTION.json", {**current, **body, "CUDA_VISIBLE_DEVICES": uuid,
                                                "lease_roots": list(map(str, roots))})
                return lease, body
            except BlockingIOError:
                if lease is not None:
                    lease.close()
            except Exception:
                if lease is not None:
                    lease.close()
                raise
        state.update(phase="WAITING_COMPATIBLE_GPU", eligible_gpus=host["eligible_gpus"],
                     blocker="No unowned, physically idle, eligible non-GPU0 slot",
                     completed=False)
        time.sleep(30)


def prepare(root: Path, config: dict, host: dict):
    if sha(host["archive"]) != config["archive_sha256"]:
        raise ValueError("publisher archive mismatch")
    partition = json.loads((root/"inputs/PARTITION.json").read_text())
    predictor = json.loads((root/"inputs/RANKER_FROZEN.json").read_text())
    previous = json.loads((root/"inputs/V3_CANDIDATES_FROZEN.json").read_text())
    if sha(root/"inputs/PARTITION.json") != config["partition_sha256"]:
        raise ValueError("canonical partition changed")
    train_ids = set(partition["train"]["ids"])
    start, stop = config["gate"]["training_group_indices"]
    gate_groups = partition["train"]["groups"][start:stop]
    gate_ids = {key for group in gate_groups for key in partition["groups"][group]}
    if len(gate_groups) != 64 or gate_ids & (set(previous["gate_ids"]) | set(predictor["training_ids"])):
        raise ValueError("the new TRAIN gate must be disjoint from earlier optimization feedback")
    items, labels = [], {}
    with gzip.open(host["archive"], "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for qa in record.get("qas", []):
                if qa["qid"] in train_ids:
                    items.append({"id": qa["qid"], "question": qa["question"], "context": record["context"]})
                    labels[qa["qid"]] = qa["answers"]
    if len(items) != 3395 or set(labels) != train_ids:
        raise ValueError("the complete canonical TRAIN inventory is required")
    family = config["family"]
    if len(family) != 1 or not isinstance(family[0], str):
        raise ValueError("One prospectively fixed candidate is required")
    policy = config.get("answer_policy", "single_answer_v1")
    save(root/"CANDIDATES_FROZEN.json", {
        "candidate": family[0], "system": recovery.answer_system(policy),
        "answer_policy": policy, "input_cap": config["input_cap"],
        "output_cap": 512, "gate_groups": gate_groups, "gate_ids": sorted(gate_ids),
        "train_ids": sorted(train_ids), "source_sha256": sha(Path(__file__)),
        "recovery_sha256": sha(Path(recovery.__file__)), "ranker_sha256": sha(root/"inputs/RANKER_FROZEN.json"),
        "partition_sha256": config["partition_sha256"], "frontier_sha256": sha(root/"frontier/FRONTIER.json"),
        "validation_or_test_outcomes_used": False, "candidate_family_size": 1,
        "pre_routing_tree_not_deployed": "None met both training cost and quality; avoid a known-negative sweep",
        "fallback": "complete fresh current CHISEL/no-echo; every draft and original recovery call charged",
        "gate": {"minimum_em_delta_vs_current_chisel_and_no_skill": 0, "maximum_tokens_vs_no_skill": .8},
    })
    return items, labels, partition, predictor, gate_ids


class Provider:
    def __init__(self, root, config, tokenizer):
        self.root, self.config, self.tokenizer = root, config, tokenizer
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                      "failed_calls": 0, "unknown_usage_calls": 0,
                      "maximum_total_provider_tokens": None, "token_usage_mode": "unlimited"}
        (root/"calls").mkdir()

    def count(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False, return_dict=False))

    def call(self, identity, arm, stage, messages):
        key = ranker.digest(f"{identity}:{arm}:{stage}")
        path = self.root/"calls"/(key+".json")
        if path.exists():
            raise RuntimeError("repeated actor call identity")
        body = {"model": self.config["model_repository"], "messages": messages, "temperature": 0,
                "max_tokens": 512, "seed": self.config["request_seed"],
                "chat_template_kwargs": {"enable_thinking": False}}
        if self.config.get("collect_logprobs"):
            body.update(logprobs=True, top_logprobs=0)
        record = {"id": identity, "arm": arm, "stage": stage, "request": body,
                  "expected_prompt_tokens": self.count(messages), "status": "REQUEST_INTENT"}
        save(path, record)
        try:
            request = Request(f"http://127.0.0.1:{self.config['port']}/v1/chat/completions",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with build_opener(ProxyHandler({})).open(request, timeout=300) as stream:
                raw = json.load(stream)
            record["raw"] = raw
            usage = raw["usage"]
            if (raw["model"] != self.config["model_repository"]
                    or any(type(usage.get(k)) is not int or usage[k] < 0
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["prompt_tokens"] != record["expected_prompt_tokens"]
                    or usage["total_tokens"] != usage["prompt_tokens"]+usage["completion_tokens"]):
                raise ValueError("model identity or native usage validation failed")
            if self.config.get("collect_logprobs"):
                from scripts.figure2_searchqa_recovery import response_confidence
                record["confidence"] = response_confidence(raw)
            record.update(status="COMPLETE", usage=usage)
        except Exception as error:
            record.update(status="FAILED", error=repr(error))
            raise
        finally:
            save(path, record)
            with self.lock:
                self.usage["calls"] += 1
                observed = record.get("raw", {}).get("usage", {})
                if all(type(observed.get(k)) is int and observed[k] >= 0
                       for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        self.usage[k] += observed[k]
                else:
                    self.usage["unknown_usage_calls"] += 1
                self.usage["failed_calls"] += record["status"] != "COMPLETE"
                save(self.root/"PROVIDER_ACCOUNTING.json", self.usage)
        return raw["choices"][0]["message"]["content"], {
            "call_id": key, "usage": usage, **({"confidence": record["confidence"]}
                                             if "confidence" in record else {}),
        }


def current_chisel(item, router, profiles, original, call, prefix):
    from phiagent.learning import qa_typed_recovery

    branch = router.route(item)["variant"]
    full = ranker.messages(item, item["context"], profiles[branch])
    def original_call():
        return call(full, prefix+"_original")
    if qa_typed_recovery.preflight_route(item["question"]) != "compact_candidate":
        return original_call()
    system, _ = qa_typed_recovery.retrieve_source_rules(
        original, profile=branch, question=item["question"], context=item["context"])
    draft = call(ranker.messages(item, item["context"], system), prefix+"_full_source")
    return qa_typed_recovery.recover_answer(
        question=item["question"], context=item["context"], draft=draft,
        decide=lambda value: qa_typed_recovery.deterministic_recovery(value)["action"],
        original=original_call, reject_question_echo=False)["answer"]


def study(root, config, prepared, state):
    from transformers import AutoTokenizer
    from phiagent.learning import qa_meta_rsi
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel
    items, labels, partition, predictor, gate_ids = prepared
    reference = root/"inputs/reference"
    parent = json.loads((reference/"lineage/config.json").read_text())
    profiles = json.loads((reference/"PROFILES.json").read_text())
    for arm in ("no_skill", "delete", "modify"):
        if ranker.digest(profiles[arm]) != parent["prompt_sha256"][arm]:
            raise ValueError("control profile binding failed")
    router = FrozenOriginalChisel(reference/"lineage", parent)
    original = qa_meta_rsi.initial_skill({k: profiles[k] for k in ("delete", "modify")})
    scorer = official_scorer({"scorer": str(root/"inputs/mrqa_official_eval.py"),
                             "scorer_sha256": config["scorer_sha256"]})
    provider = Provider(root, config, AutoTokenizer.from_pretrained(config["model_directory"], local_files_only=True))
    candidate_name = config["family"][0]
    system = recovery.answer_system(config.get("answer_policy", "single_answer_v1"))
    public_router = None
    if config.get("public_router_sha256"):
        router_path = root/"inputs/PUBLIC_ROUTER.json"
        if sha(router_path) != config["public_router_sha256"]:
            raise ValueError("frozen public-input router changed")
        public_router = json.loads(router_path.read_text())
        if public_router["inference_uses_gold_or_task_id"] is not False:
            raise ValueError("only public-input routes may precede the actor")
    arms = ("no_skill", "chisel_current", candidate_name)
    results = {arm: {} for arm in arms}
    lock = threading.Lock()
    (root/"tasks").mkdir()

    def execute(item):
        for arm in arms:
            if item["id"] in results[arm]:
                continue
            calls, recovered, receipt = [], False, None
            chosen_route = None

            def call(messages, stage):
                answer, record = provider.call(item["id"], arm, stage, messages)
                calls.append(record)
                return answer

            def original_chisel(prefix):
                return current_chisel(item, router, profiles, original, call, prefix)

            if arm == "no_skill":
                answer = call(ranker.messages(item, item["context"], profiles["no_skill"]), "actor")
            elif arm == "chisel_current":
                answer = original_chisel("control")
            else:
                if config.get("answer_policy", "single_answer_v1") == "target_role_v1":
                    messages, receipt = recovery.target_role_request(item, predictor, provider.count)
                else:
                    messages, receipt = recovery.compact_request(
                        item, predictor, provider.count, config["input_cap"], system,
                    )
                chosen_route = recovery.route(
                    public_router["tree"], recovery.public_features(item, receipt, predictor),
                ) if public_router else "compact"
                if chosen_route == "original":
                    answer = original_chisel("public_input_route")
                else:
                    answer = call(messages, "compact")
                    if recovery.fallback_needed(item["question"], item["context"], answer):
                        answer, recovered = original_chisel("paid_recovery"), True
            prediction = router.campaign.extract_answer(answer)
            row = {"id": item["id"], "arm": arm, "split": "train", "answer": answer, "prediction": prediction,
                   "em": int(scorer.metric_max_over_ground_truths(scorer.exact_match_score, prediction, labels[item["id"]])),
                   "f1": scorer.metric_max_over_ground_truths(scorer.f1_score, prediction, labels[item["id"]]),
                   "usage": {k: sum(c["usage"][k] for c in calls)
                             for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                   "calls": calls, "paid_recovery": recovered, "source_receipt": receipt,
                   "public_input_route": chosen_route}
            save(root/"tasks"/(ranker.digest(item["id"]+":"+arm)+".json"), row)
            with lock:
                results[arm][item["id"]] = row
                state["completed_cells"] = sum(map(len, results.values()))

    def evaluate(subset):
        bounded_execute(execute, subset, config["execution_workers"])

    if config.get("purpose") == "complete_train_router_acquisition":
        reused = json.loads((root/"inputs/REUSED_GATE.json").read_text())
        if (reused["source_candidate"] != candidate_name
                or set(reused["tasks_per_arm"]) != set(arms)):
            raise ValueError("Only fully audited compatible TRAIN cells may be reused")
        reused_tokens = 0
        for row in reused["rows"]:
            if row["id"] not in partition["train"]["ids"] or row["arm"] not in results:
                raise ValueError("Reused outcome is not a canonical TRAIN task")
            if row["id"] in results[row["arm"]]:
                raise ValueError("Reused task-arm outcome appears twice")
            results[row["arm"]][row["id"]] = row
            reused_tokens += row["usage"]["total_tokens"]
        if reused_tokens != reused["provider_tokens"]:
            raise ValueError("Reused gate costs differ from physical provider evidence")
        if {arm: len(rows) for arm, rows in results.items()} != reused["tasks_per_arm"]:
            raise ValueError("Reused per-method populations differ")
        state.update(phase="COMPLETE_TRAIN_ROUTER_ACQUISITION",
                     completed_cells=sum(map(len, results.values())), target_cells=3395*3)
        evaluate(items)
        if (provider.usage["failed_calls"] or provider.usage["unknown_usage_calls"]
                or any(set(rows) != set(partition["train"]["ids"]) for rows in results.values())):
            raise ValueError("Every paired TRAIN trajectory and native cost is required")
        total = sum(row["usage"]["total_tokens"] for rows in results.values() for row in rows.values())
        if total != provider.usage["total_tokens"] + reused_tokens:
            raise ValueError("Complete TRAIN accounting must include the reused gate exactly once")
        save(root/"TRAIN_DATASET_COMPLETE.json", {
            "status": "COMPLETE_3395_TRAIN_ROUTER_DATA", "tasks_per_arm": 3395,
            "methods": {arm: ranker.summary(list(rows.values())) for arm, rows in results.items()},
            "new_provider_usage": provider.usage, "reused_gate_tokens_once": reused_tokens,
            "logical_execution_tokens": total,
            "cumulative_optimizer_tokens": config["inherited_optimizer_provider_tokens"]
            + provider.usage["total_tokens"],
            "confidence_missing_for_reused_gate": True, "reused_tasks_per_arm": reused["tasks_per_arm"],
            "data_acquisition_not_candidate_admission": True,
            "validation_calls": 0, "test_calls": 0, "paper_admitted": False,
        })
        save(root/"NEXT_DEPENDENCY.json", {
            "status": "FIT_COST_SENSITIVE_ROUTER_ON_COMPLETE_TRAIN",
            "training_only": True, "confidence_missing_is_not_zero_confidence": True,
            "source_gates": reused["source_roots"], "official_metrics": "Native MRQA EM and F1",
            "acceptance_unchanged": "Quality at least strongest matched control and "
                                    "actor tokens at most80% cheapest control before promotion",
        })
        state.update(phase="COMPLETE_TRAIN_ROUTER_DATA", completed=True)
        return

    def report():
        methods = {arm: ranker.summary(list(rows.values())) for arm, rows in results.items()}
        intervals = {control: ranker.paired(list(results[candidate_name].values()),
                                           list(results[control].values()), partition["groups"])
                     for control in arms[:2]}
        return {"methods": methods, "paired_uncertainty": intervals,
                "recovery_tasks": sum(x["paid_recovery"] for x in results[candidate_name].values()),
                "optimizer_provider_usage": provider.usage,
                "inherited_v3_optimizer_tokens": config["inherited_optimizer_provider_tokens"],
                "cumulative_optimizer_tokens": provider.usage["total_tokens"]+config["inherited_optimizer_provider_tokens"],
                "validation_calls": 0, "test_calls": 0, "final_test_execution_tokens": 0, "paper_admitted": False}

    state.update(phase="DISJOINT_NATIVE_TRAIN_GATE", gate_tasks=len(gate_ids))
    evaluate([item for item in items if item["id"] in gate_ids])
    gate = report()
    candidate = gate["methods"][candidate_name]
    gate["passed"] = (
        candidate["em_percent"] >= max(gate["methods"][arm]["em_percent"] for arm in arms[:2])
        and candidate["mean_actor_tokens"] <= .8*gate["methods"]["no_skill"]["mean_actor_tokens"])
    save(root/"TRAIN_GATE.json", gate)
    if not gate["passed"]:
        save(root/"NEXT_DEPENDENCY.json", {
            "status": "TRAIN_ONLY_RECOVERY_DIAGNOSIS_REQUIRED",
            "input_result": str(root/"TRAIN_GATE.json"),
            "input_tasks": str(root/"tasks"), "input_calls": str(root/"calls"),
            "action": "Separate supported wrong answers, invalid contracts, and expensive correct recoveries on this TRAIN gate; compute the next quality/cost frontier before any new inference.",
            "validation_and_test_remain_sealed": True, "full_train_not_dispatched": True})
        state.update(phase="NEGATIVE_TRAIN_GATE", completed=True)
        return
    state["phase"] = "COMPLETE_TRAIN_AUTOMATIC_SUCCESSOR"
    evaluate(items)
    if provider.usage["failed_calls"] or provider.usage["unknown_usage_calls"]:
        raise RuntimeError("full training cannot pass with failed or unknown-usage calls")
    if sum(row["usage"]["total_tokens"] for rows in results.values() for row in rows.values()) != provider.usage["total_tokens"]:
        raise RuntimeError("raw provider and complete-task totals do not reconcile")
    save(root/"TRAIN_RESULT.json", {"status": "COMPLETE_3395_TRAIN", **report()})
    save(root/"NEXT_DEPENDENCY.json", {
        "status": "AWAIT_PARENT_MATCHED_OFFICIAL_OPTIMIZERS_AND_VALIDATION",
        "candidate_sha256": sha(root/"CANDIDATES_FROZEN.json"),
        "test": "sealed until parent-controlled all-method source/policy freeze"})
    state.update(phase="COMPLETE_3395_TRAIN", completed=True)


def run(root: Path, *, prepare_study=prepare, run_study=study):
    root = root.resolve()
    config = json.loads((root/"pool-config.json").read_text())
    hostname = socket.gethostname()
    if hostname not in config["hosts"]:
        raise RuntimeError("remote-only authorized H200 execution required")
    host = config["hosts"][hostname]
    config = {**config, "model_directory": host["model_directory"]}
    if config["token_usage_mode"] != "unlimited" or config["maximum_total_provider_tokens"] is not None:
        raise ValueError("cumulative token accounting must remain unlimited")
    successor = root/"SUCCESSOR_FREEZE.json"
    if successor.exists():
        frozen = json.loads(successor.read_text())
        if sha(root/"pool-config.json") != frozen["config_sha256"]:
            raise ValueError("prospectively frozen configuration changed")
        for name, expected in frozen["source_sha256"].items():
            if sha(root/"source"/name) != expected:
                raise ValueError("prospectively frozen source changed: "+name)
        for name, expected in json.loads((root/"INPUTS_FROZEN.json").read_text()).items():
            if sha(root/name) != expected:
                raise ValueError("prospectively frozen input changed: "+name)
    owner = (root/"owner.lock").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/"STARTED.json").exists():
        raise RuntimeError("no silent restart of a frozen run")
    state = {"pid": os.getpid(), "hostname": hostname, "phase": "PREPARING", "completed_cells": 0, "completed": False}
    save(root/"STARTED.json", {**state, "at_unix": time.time(), "argv": sys.argv,
                              "python": sys.version, "config_sha256": sha(root/"pool-config.json"),
                              "packages": {k: importlib.metadata.version(k) for k in ("vllm", "torch", "transformers")}})
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            save(root/"HEARTBEAT.json", {**state, "at_unix": time.time()})
            stop.wait(15)

    threading.Thread(target=heartbeat, daemon=True).start()
    lease, server, selection = None, None, None
    try:
        prepared = prepare_study(root, config, host)
        state["phase"] = "EXACT_CHECKPOINT_ATTESTATION"
        manifest = json.loads((root/"inputs/MODEL_MANIFEST.json").read_text())
        if (manifest["repository"], manifest["revision"]) != (config["model_repository"], config["model_revision"]):
            raise ValueError("checkpoint identity mismatch")
        for name, expected in manifest["files_sha256"].items():
            if sha(Path(host["model_directory"])/name) != expected:
                raise ValueError("checkpoint byte mismatch: "+name)
        save(root/"MODEL_VERIFIED.json", manifest)
        lease, selection = acquire(root, host, state)
        if lease is None:
            return
        cache = root/"c"
        cache.mkdir(exist_ok=True)
        python = host["runtime_python"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": selection["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PATH": str(Path(python).parent)+os.pathsep+os.environ.get("PATH", ""),
               "TMPDIR": str(cache), "TMP": str(cache), "TEMP": str(cache), "HF_HOME": str(cache),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
               "VLLM_CACHE_ROOT": str(cache/"vllm"), "VLLM_RPC_BASE_PATH": str(cache)}
        if len(str(cache).encode())+37 > 107:
            raise ValueError("IPC cache path too long")
        command = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", host["model_directory"],
                   "--served-model-name", config["model_repository"], "--host", "127.0.0.1",
                   "--port", str(config["port"]), "--dtype", "bfloat16", "--tensor-parallel-size", "1",
                   "--max-model-len", "32768", "--max-num-seqs", "16", "--gpu-memory-utilization", ".45",
                   "--generation-config", "vllm", "--max-num-batched-tokens", "4096", "--enforce-eager",
                   "--language-model-only", "--attention-config", '{"backend":"FLASH_ATTN","flash_attn_version":3}',
                   "--gdn-prefill-backend", "triton", "--seed", str(config["request_seed"]),
                   "--default-chat-template-kwargs", '{"enable_thinking":false}']
        save(root/"SERVICE_COMMAND.json", {"argv": command, "CUDA_VISIBLE_DEVICES": selection["gpu_uuid"]})
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", config["port"]))
        with (root/"service.log").open("x") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase="SERVICE_STARTING", service_pid=server.pid)
        deadline = time.monotonic()+900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("native model service exited: see service.log")
            try:
                with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{config['port']}/v1/models", timeout=3) as stream:
                    models = json.load(stream)
                if models["data"][0]["id"] != config["model_repository"]:
                    raise ValueError("served model identity mismatch")
                save(root/"SERVICE_READY.json", models)
                break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("native service startup exceeded900seconds")
        run_study(root, config, prepared, state)
    except Exception as error:
        state.update(phase="FAILED", completed=True, error=repr(error))
        save(root/"FAILURE.json", {**state, "traceback": traceback.format_exc()})
        raise
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if lease is not None:
            lease.close()
        if selection is not None:
            try:
                registry(root, "finish", {**selection, "phase": state["phase"]})
            except Exception as error:
                save(root/"REGISTRY_FINISH_FAILURE.json", {"error": repr(error)})
        stop.set()
        save(root/"HEARTBEAT.json", {**state, "at_unix": time.time(), "owned_gpu_released": True})
        save(root/"TERMINAL.json", state)


def main():
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
