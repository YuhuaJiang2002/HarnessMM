#!/usr/bin/env python3
"""Bind instruction RSI to the admitted DeepSeek Terminal parent, without recovery jobs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import write_json_atomic  # noqa: E402

PARENT = Path(
    "/path/to/data/phiagent/work/harness-terminal-recovery/"
    "industrial-capability-20261001-v4-adoption/sealed-deepseek"
)
BINDING_SHA = "a67b4792c0fd309f5cbabc761e6ecb0fa5513708513bcbe87d316dde0743ac78"
SEAL_SHA = "2d1aa44652fecc8f0426d526a99f72c77b958290f49a2287243937a9ddcd1198"


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_remote():
    if (sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("Use the admitted A800 CPU/API native runtime")


def procedural_job(job, request, descriptor):
    """Change only the inherited SDK instruction arm; preserve every native contract."""
    if request["phase"] != "train":
        raise ValueError("This adapter cannot open protected splits")
    instruction = request.get("system_prompt_append", "")
    if request["codex_prompt"] != request["task"]["public_prompt"]:
        raise ValueError("Preserve the original native task prompt")
    residual = request.get("residual")
    if not instruction and residual is None and descriptor is None:
        return deepcopy(job)
    if (not instruction or not residual or residual["instruction"] != instruction
            or residual["application"] != "system_section" or descriptor is None):
        raise ValueError("Require an actual compiled TRAIN system-section instruction")
    value = deepcopy(job)
    kwargs = value["agents"][0]["kwargs"]
    kwargs["arm"], kwargs["chisel"] = "chisel", descriptor
    return value


def native_bridge(operation, native, directory, request_path):
    require_remote()
    from scripts import run_industrial_capability_baselines as capability
    from scripts import run_deepseek_terminal_campaign as callbacks
    from integrations.deepseek_harness.harbor_agent import (
        normalize_runner_usage, recover_runner_accounting,
    )

    spec = importlib.util.spec_from_file_location(
        "industrial_strict_native_verifier", native / "guards/capability.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Strict official native verifier guard is absent")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if not callable(getattr(guard, "executed_native_verdict", None)):
        raise RuntimeError("Strict native executed-tests/reward API is not available")
    config = load(native / "CONFIG.json")
    request = load(request_path)
    train_ids = set(load(native / "SPLIT_MANIFEST.json")["splits"]["train"])
    if request["phase"] != "train":
        raise ValueError("Protected request rejected before native execution")
    if operation == "prepare-feedback":
        if {row["task_id"] for row in request["tasks"]} != train_ids:
            raise ValueError("Feedback is restricted to the complete TRAIN population")
        for row in request["parent_rows"]:
            feedback = load_json_feedback(row["public_feedback"])
            trace = feedback["public_trace_path"]
            if trace is not None:
                public = json.loads(callbacks.compact_public_trace(Path(trace)))
                public["official_task_reward"] = row["native_score"]
                row["public_feedback"] = json.dumps(public)
        write_json_atomic(request_path, request)
    elif operation == "prepare-trial":
        task_id = request["task"]["task_id"]
        if task_id not in train_ids:
            raise ValueError("Task is outside the admitted TRAIN identities")
        capacity = capability.storage_capacity(config, native)
        write_json_atomic(directory / "RESOURCE_ADMISSION.json", capacity)
        if not capacity["ready"]:
            raise RuntimeError("Native memory/output/private/cache/Docker reserves are not ready")
        command = capability.trial_command(config, "deepseek", native, directory, task_id)
        job = procedural_job(
            load(directory / "HARBOR_JOB.json"), request,
            load(directory / "INSTRUCTION_PROVENANCE.json"),
        )
        write_json_atomic(directory / "HARBOR_JOB.json", job)
        write_json_atomic(directory / "NATIVE_COMMAND.json", {"argv": command})
    elif operation == "prepare-optimizer":
        payload = callbacks.optimizer_payload(request, train_ids)
        for public, original in zip(payload["training_evidence"], request["training_evidence"]):
            public["baseline"]["all_attempt_actor_control_usage"] = original["baseline"]["usage"]
            for ident, value in public["candidates"].items():
                value["all_attempt_actor_control_usage"] = original["candidates"][ident]["usage"]
        private = native / "private" / "optimizer" / request["request_id"]
        job = {
            "schema_version": 1, "run_dir": str(private), "purpose": "benchmark",
            "arm": "baseline", "chisel": None,
            "session_id": "industrial-rsi-" + request["request_id"][:24],
            "seed": 20260928, "model": "deepseek-flash",
            "max_tokens": config["deepseek"]["max_output_tokens"],
            "prompt": json.dumps(payload, ensure_ascii=False), "tools": [], "bridge": None,
            "api_key_file": config["deepseek"]["api_key_file"],
            "runtime": config["deepseek"]["runtime"],
            "source_revision": config["deepseek"]["official_revision"],
        }
        write_json_atomic(directory / "PROVIDER_PROMPT.json", payload)
        write_json_atomic(directory / "SDK_JOB.json", job)
        write_json_atomic(directory / "NATIVE_COMMAND.json", {"argv": [
            config["deepseek"]["sdk_python"],
            str(native / "source/integrations/deepseek_harness/runner.py"),
            "--job", str(directory / "SDK_JOB.json"),
        ]})
    elif operation in ("harvest-trial", "harvest-optimizer"):
        if operation == "harvest-trial":
            row = capability.harvest(
                config, "deepseek", native, directory, request["task"]["task_id"],
            )
            try:
                verdict = guard.executed_native_verdict(
                    load(directory / "native_result.json"), directory / "harbor/native",
                )
                write_json_atomic(directory / "EXECUTED_NATIVE_VERDICT.json", verdict)
            except (OSError, RuntimeError, ValueError) as error:
                write_json_atomic(directory / "NATIVE_VERIFIER_ERROR.json", {
                    "type": type(error).__name__, "message": str(error),
                    "native_reward_not_substituted_for_executed_tests": True,
                })
                row.update(status="BLOCKED", native_score=None, error=str(error))
            write_json_atomic(directory / "NATIVE_VERDICT.json", row)
            if not row["cost"].get("sdk_worker_dir"):
                raise RuntimeError("Native trial has no identified SDK worker; usage remains UNKNOWN")
            worker = Path(row["cost"]["sdk_worker_dir"])
            trace = worker.parent / "native-backend/terminal-exec.jsonl"
            if not trace.is_file():
                trace = worker.parent / "terminal-exec.jsonl"
            feedback = callbacks.compact_public_trace(trace) if trace.is_file() else json.dumps({
                "public_actor_events": [], "trace_available": False,
                "diagnostic": "No public actor trace was emitted; native receipts remain retained",
            })
            result = {
                "request_id": request["request_id"], "identity": request["identity"],
                "task_id": request["task"]["task_id"],
                "execution_status": "completed" if row["status"] == "SCORED" else "blocked",
                "native_score": row["native_score"],
                "success": bool(row["native_score"]) if row["native_score"] is not None else None,
                "feedback": feedback,
            }
        else:
            job = load(directory / "SDK_JOB.json")
            worker = Path(job["run_dir"])
            response = load(worker / "result.json")
            proposal = {"instruction": "", "supporting_task_ids": []}
            failure = None
            try:
                if response.get("status") != "completed":
                    raise ValueError("Native SDK proposer did not complete")
                proposal = callbacks.parse_proposal(
                    response.get("final_response"), train_ids, request["limits"],
                )
                if proposal["instruction"].strip() and len(set(proposal["supporting_task_ids"])) < 2:
                    raise ValueError("Reusable instruction needs two distinct TRAIN scenarios")
            except (ValueError, TypeError) as error:
                failure = {"type": type(error).__name__, "message": str(error)}
            result = {
                "request_id": request["request_id"], "identity": request["identity"],
                "execution_status": "failed" if failure else "completed", **proposal,
            }
        accounting = recover_runner_accounting(worker)
        write_json_atomic(directory / "RAW_ALL_ATTEMPT_ACCOUNTING.json", accounting)
        result["usage"] = callbacks.canonical_usage(
            normalize_runner_usage(accounting.get("usage")),
            accounting.get("complete") is True,
        )
        result["provenance"] = {
            "raw_all_attempt_accounting": str(directory / "RAW_ALL_ATTEMPT_ACCOUNTING.json"),
            "raw_all_attempt_accounting_sha256": sha(directory / "RAW_ALL_ATTEMPT_ACCOUNTING.json"),
            "all_attempt_usage_complete": accounting.get("complete") is True,
            "unknown_is_zero": False, "sdk_worker_dir": str(worker),
            "failed_proposal": failure if operation == "harvest-optimizer" else None,
            "official_native_verifier": operation == "harvest-trial",
        }
        write_json_atomic(directory / "CALLBACK_RESULT.json", result)
    else:
        raise ValueError("Unknown native bridge operation")


def load_json_feedback(value):
    feedback = json.loads(value)
    if not isinstance(feedback, dict):
        raise ValueError("Parent feedback must be a public observation object")
    return feedback


class NativeTrainAdapter:
    def __init__(self, *, run, admission, train_input):
        self.run, self.admission, self.data = run, admission, train_input
        self.native = Path(load(run / "NATIVE_RUNTIME_ROOT.json")["path"])
        self.config = load(self.native / "CONFIG.json")
        self.rows = {row["task_id"]: row for row in train_input["parent_rows"]}
        if set(self.rows) != set(train_input["partition"]["splits"]["train"]):
            raise ValueError("Every admitted native TRAIN parent row is required")

    def process(self, directory, stage, command):
        folder = directory / stage
        folder.mkdir()
        cache = folder / "process-cache"
        cache.mkdir()
        write_json_atomic(folder / "COMMAND.json", {"argv": command, "at": now()})
        with (folder / "stdout.log").open("xb") as log:
            process = subprocess.Popen(
                command, cwd=self.native,
                env={**os.environ, "PYTHONPATH": str(self.native / "source"),
                     "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                     "TMPDIR": str(cache)},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json_atomic(folder / "PROCESS.json", {
            "pid": process.pid, "at": now(), "controller_pid": os.getpid(),
        })
        code = process.wait()
        write_json_atomic(folder / "EXIT.json", {"returncode": code, "at": now()})
        if code != 0:
            raise RuntimeError(f"Native {stage} failed with exit {code}; inspect {folder / 'stdout.log'}")

    def bridge(self, operation, directory, request_path):
        self.process(directory, operation, [
            self.config["deepseek"]["python"],
            str(self.native / "source/scripts/industrial_deepseek_train_adapter.py"),
            operation, "--native-root", str(self.native),
            "--directory", str(directory), "--request", str(request_path),
        ])

    def optimize(self, request):
        directory = self.run / "native-callbacks" / "optimizer" / request["request_id"]
        directory.mkdir(parents=True)
        request_path = directory / "REQUEST.json"
        write_json_atomic(request_path, request)
        self.bridge("prepare-optimizer", directory, request_path)
        inherited = self.admission.get("inherited_train_proposer")
        if inherited is not None:
            prior_result = Path(inherited["result"])
            prior_payload = Path(inherited["payload"])
            if (sha(prior_result) != inherited["result_sha256"]
                    or sha(prior_payload) != inherited["payload_sha256"]
                    or load(prior_payload) != load(directory / "PROVIDER_PROMPT.json")):
                raise ValueError("Infrastructure continuation cannot change the paid TRAIN proposal")
            result = load(prior_result)
            if result["identity"] != request["identity"] or result["execution_status"] != "completed":
                raise ValueError("Reused proposer must preserve the actual model/interface identity")
            result = deepcopy(result)
            result["request_id"] = request["request_id"]
            result["provenance"]["inherited_train_proposal"] = {
                **inherited, "new_provider_requests": 0, "charged_once_in_lineage": True,
            }
            write_json_atomic(directory / "CALLBACK_RESULT.json", result)
            return result
        self.process(directory, "native-sdk", load(directory / "NATIVE_COMMAND.json")["argv"])
        self.bridge("harvest-optimizer", directory, request_path)
        return load(directory / "CALLBACK_RESULT.json")

    def execute(self, request):
        if request["phase"] != "train":
            raise ValueError("This adapter executes only TRAIN")
        if request["arm"] == "baseline":
            row = self.rows[request["task"]["task_id"]]
            if request["residual"] is not None or request["seed"] != 20260928:
                raise ValueError("Only the identical frozen native parent may be reused")
            return {
                "request_id": request["request_id"], "identity": request["identity"],
                "task_id": row["task_id"], "execution_status": "completed",
                "native_score": row["native_score"], "success": bool(row["native_score"]),
                "usage": {**row["cost"]["usage"], "cache_read_tokens": None,
                          "cache_write_tokens": None, "cost_usd": None},
                "feedback": row["public_feedback"],
                "provenance": {
                    "native_parent_binding_sha256": BINDING_SHA,
                    "native_parent_seal_sha256": SEAL_SHA,
                    "native_record": row["native_result"],
                    "native_record_sha256": row["native_result_sha256"],
                    "new_actor_calls": 0, "all_attempt_usage_complete": True,
                },
            }
        directory = self.native / "tasks" / request["request_id"]
        directory.mkdir(parents=True)
        request_path = directory / "REQUEST.json"
        write_json_atomic(request_path, request)
        instruction = request.get("system_prompt_append", "")
        proof = []
        for path in (self.run / "callbacks/optimizer").glob("*/RESULT.json"):
            proposal = load(path)
            if (proposal["execution_status"] == "completed"
                    and proposal["instruction"].strip() == instruction.strip()):
                original = load(path.parent / "REQUEST.json")
                payload = (
                    self.run / "native-callbacks/optimizer" / original["request_id"]
                    / "PROVIDER_PROMPT.json"
                )
                proof.append({
                    "optimizer_request_id": original["request_id"],
                    "iteration": original["iteration"],
                    "optimizer_result_sha256": sha(path),
                    "optimizer_request_sha256": sha(path.parent / "REQUEST.json"),
                    "optimizer_payload_sha256": sha(payload),
                    "supporting_task_ids": proposal["supporting_task_ids"],
                    "optimizer_task_ids": sorted(row["task"]["task_id"]
                                                 for row in original["training_evidence"]),
                    "provider_usage": proposal["usage"],
                })
        if instruction and not proof:
            raise ValueError("Require a paid completed TRAIN proposal, not a handwritten instruction")
        artifact = directory / "instruction.txt"
        if instruction:
            artifact.write_text(instruction)
        descriptor = {
            "path": str(artifact), "sha256": sha(artifact), "artifact_kind": "training-derived",
            "provenance": {
                "method": "CHISEL", "application": "system_section",
                "partition_sha256": self.admission["partition_sha256"],
                "optimizer_receipts": proof, "candidate_sha256": request["candidate_sha256"],
                "final_policy_sha256": request.get("final_policy_sha256"),
                "source_sha256": request.get("source_sha256"),
            },
        } if instruction else None
        write_json_atomic(directory / "INSTRUCTION_PROVENANCE.json", descriptor)
        self.bridge("prepare-trial", directory, request_path)
        self.process(directory, "native-harbor", load(directory / "NATIVE_COMMAND.json")["argv"])
        self.bridge("harvest-trial", directory, request_path)
        return load(directory / "CALLBACK_RESULT.json")


def factory(*, run, admission, train_input):
    return NativeTrainAdapter(run=run, admission=admission, train_input=train_input)


def admission_draft(admission):
    value = deepcopy(admission)
    value.update(
        status="AWAITING_EXCLUSIVE_CALLBACK_OWNER_APPROVAL",
        native_owner_handoff=False,
        callback_owner_approval=None,
        authority="Capability/provenance binding only; not native callback-factory approval",
    )
    return value


def prepare_admission(run, continuation=None):
    require_remote()
    if (ROOT != run / "source" or (run / "ADMISSION.json").exists()
            or (run / "ADMISSION_DRAFT.json").exists()):
        raise ValueError("Prepare only a fresh frozen successor")
    binding_path = PARENT.parent / "BINDING-deepseek.json"
    if sha(binding_path) != BINDING_SHA or sha(PARENT / "SEAL.json") != SEAL_SHA:
        raise ValueError("Owner-admitted capable parent pins changed")
    binding, seal = load(binding_path), load(PARENT / "SEAL.json")
    names = (
        "CONFIG.json", "REPORT.json", "SOURCE_SHA256.json", "SPLIT_MANIFEST.json",
        "TASK_SOURCE_SHA256.json", "TASK_PROMPT_RECEIPTS.json", "NATIVE_RUNTIME.json",
    )
    for name in names:
        if sha(PARENT / name) != seal["files"][name]:
            raise ValueError(f"Owner-sealed input changed: {name}")
    if (binding["ready"] is not True or binding["native_scored"] != 18
            or binding["native_successes"] != 10):
        raise ValueError("Use the recovered capable18/10 parent, not its unrecovered predecessor")
    native = (
        Path("/path/to/data/phiagent/work/deepseek-harness/industrial-capability/20260930-v1")
        / ("rsi-" + run.name + "-native")
    )
    if native.exists():
        raise ValueError("The native SDK/Harbor namespace must also be fresh")
    native.mkdir()
    write_json_atomic(run / "NATIVE_RUNTIME_ROOT.json", {
        "path": str(native),
        "reason": "Preserve both inherited SDK and native Harbor output-root admission contracts",
        "parent_root_not_modified": True,
    })
    shutil.copytree(PARENT / "source", native / "source",
                    ignore=shutil.ignore_patterns("__pycache__", "._*"))
    for name in names:
        shutil.copyfile(PARENT / name, native / name)
    (native / "guards").mkdir()
    shutil.copyfile(
        ROOT / "scripts/run_industrial_capability_baselines.py",
        native / "guards/capability.py",
    )
    source_name = "scripts/industrial_deepseek_train_adapter.py"
    shutil.copyfile(ROOT / source_name, native / "source" / source_name)
    source_hashes = load(native / "SOURCE_SHA256.json")
    source_hashes[source_name] = sha(native / "source" / source_name)
    verifier_api = "scripts/run_codex_terminal.py"
    original_verifier_api_sha = source_hashes[verifier_api]
    shutil.copyfile(ROOT / verifier_api, native / "source" / verifier_api)
    source_hashes[verifier_api] = sha(native / "source" / verifier_api)
    for name, digest in source_hashes.items():
        if sha(native / "source" / name) != digest:
            raise ValueError("Owner-sealed native source changed during cloning")
    write_json_atomic(native / "SOURCE_SHA256.json", source_hashes)
    config = load(native / "CONFIG.json")
    config["require_executed_native_tests"] = True
    write_json_atomic(native / "CONFIG.json", config)
    write_json_atomic(native / "FROZEN_IDENTITY.json", {
        name: sha(native / name) for name in (*names, "SOURCE_SHA256.json", "guards/capability.py")
    })
    partition = load(native / "SPLIT_MANIFEST.json")
    report = load(native / "REPORT.json")
    rows = deepcopy(report["attempts"])
    if (len(rows) != 18 or {row["task_id"] for row in rows} != set(partition["splits"]["train"])
            or sum(row["native_score"] for row in rows) != 10
            or sum(row["cost"]["usage"]["total_tokens"] for row in rows) != 29907333):
        raise ValueError("The admitted TRAIN parent rows do not reconcile with its binding")
    records = {row["id"]: row for row in partition["records"]}
    tasks = []
    for row in rows:
        task_id = row["task_id"]
        prompt_path = Path(config["task_root"]) / task_id / "instruction.md"
        if sha(prompt_path) != records[task_id]["public_task_sha256"]:
            raise ValueError("Admitted public TRAIN instruction changed")
        tasks.append({"task_id": task_id, "unit_id": task_id,
                      "public_prompt": prompt_path.read_text()})
        worker = row["cost"].get("sdk_worker_dir")
        trace = Path(worker).parent / "native-backend/terminal-exec.jsonl" if worker else None
        if trace is not None and not trace.is_file():
            trace = Path(worker).parent / "terminal-exec.jsonl"
        row["public_feedback"] = json.dumps({
            "official_task_reward": row["native_score"], "public_actor_events": [],
            "public_trace_path": str(trace) if trace is not None and trace.is_file() else None,
            "trace_preparation": "Native-only compact trace prepared by admitted helper before release",
        })
    tasks.sort(key=lambda row: partition["splits"]["train"].index(row["task_id"]))
    data = {
        "phase": "train", "partition": partition, "tasks": tasks, "parent_rows": rows,
        "task_index": [{"task_id": row["id"], "unit_id": row["id"],
                        "public_prompt_sha256": row["public_task_sha256"]}
                       for row in partition["records"]],
        "parent_summary": {
            "task_ids": partition["splits"]["train"], "native_scored": 18,
            "native_successes": 10, "all_attempt_actor_control_tokens": 29907333,
            "all_attempt_usage_complete": True,
        },
    }
    write_json_atomic(run / "TRAIN_INPUT.json", data)
    subprocess.run([
        config["deepseek"]["python"],
        str(native / "source/scripts/industrial_deepseek_train_adapter.py"),
        "prepare-feedback", "--native-root", str(native),
        "--directory", str(run), "--request", str(run / "TRAIN_INPUT.json"),
    ], check=True, cwd=native, env={
        **os.environ, "PYTHONPATH": str(native / "source"), "PYTHONDONTWRITEBYTECODE": "1",
    })
    settings = load(run / "RSI_CONFIG.json")
    identity = {
        "model": "deepseek-flash", "runtime": "official_deepseek_harness_sdk",
        "wire_api": "anthropic_messages", "shared_prompt_sha256": config["shared_prompt_sha256"],
        "official_revision": config["deepseek"]["official_revision"],
        "sampling_seed": "unsupported_by_provider",
    }
    admission = {
        "schema": "industrial-terminal-train-admission/v1",
        "status": "AWAITING_EXCLUSIVE_CALLBACK_OWNER_APPROVAL",
        "harness": "deepseek", "phase": "train", "validation_or_test_read": False,
        "exclusive_successor_owner": settings["owner"], "native_owner_handoff": False,
        "parent": settings["parents"]["deepseek"],
        "shared_contract": {
            **identity, "shared_prompt_version": config["shared_prompt_version"],
            "native_output_time_tool_contracts_unchanged": True,
            "max_output_tokens": config["deepseek"]["max_output_tokens"],
            "native_timeout_multiplier": 1.0,
        },
        "partition_sha256": config["partition_sha256"],
        "all_attempt_actor_control_usage": True, "official_native_verifier": True,
        "train_payload_count": 18, "execution_identity": identity, "optimizer_identity": identity,
        "callback_factory": "scripts.industrial_deepseek_train_adapter:factory",
        "source_root": str(ROOT), "source_sha256": load(run / "SOURCE_SHA256.json"),
        "native_source_sha256": source_hashes, "train_input_sha256": sha(run / "TRAIN_INPUT.json"),
        "strict_verifier_guard_sha256": sha(native / "guards/capability.py"),
        "native_runtime_root": str(native),
        "native_verifier_api_upgrade": {
            "source": verifier_api, "original_sha256": original_verifier_api_sha,
            "sha256": source_hashes[verifier_api],
            "purpose": "Reuse existing executed official tests/reward guard; no actor changes",
            "official_sdk_loop_and_tools_changed": False,
        },
        "authority": "d32 capability/provenance binding only; callback owner approval absent",
        "binding_sha256": BINDING_SHA, "seal_sha256": SEAL_SHA,
        "validation_or_test_learning": False,
    }
    if continuation is not None:
        state = load(continuation / "STATE.json")
        if state["status"] != "BLOCKED":
            raise ValueError("Never take over a healthy or running TRAIN operation")
        committed = list((continuation / "callbacks/optimizer").glob("*/COMMITTED.json"))
        if len(committed) != 1:
            raise ValueError("Reuse exactly one complete paid TRAIN proposer, never an ambiguous call")
        folder = committed[0].parent
        previous = load(folder / "RESULT.json")
        receipt = load(committed[0])
        if (sha(folder / "REQUEST.json") != receipt["request_sha256"]
                or sha(folder / "RESULT.json") != receipt["result_sha256"]):
            raise ValueError("Committed predecessor proposer changed")
        optimizer_request = load(folder / "REQUEST.json")
        payload = (
            continuation / "native-callbacks/optimizer" / optimizer_request["request_id"]
            / "PROVIDER_PROMPT.json"
        )
        admission["inherited_train_proposer"] = {
            "predecessor": str(continuation), "result": str(folder / "RESULT.json"),
            "result_sha256": sha(folder / "RESULT.json"),
            "payload": str(payload), "payload_sha256": sha(payload),
            "usage": previous["usage"], "native_paid_proposer_requests": 1,
            "scope": "TRAIN-only instruction; unchanged payload and model; source-bound native path fix",
        }
    write_json_atomic(run / "ADMISSION_DRAFT.json", admission_draft(admission))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=(
        "prepare-admission", "prepare-feedback", "prepare-trial", "prepare-optimizer",
        "harvest-trial", "harvest-optimizer",
    ))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--continuation", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare-admission":
        if args.run_dir is None:
            parser.error("prepare-admission requires --run-dir")
        prepare_admission(args.run_dir, args.continuation)
    else:
        if args.native_root is None or args.directory is None or args.request is None:
            parser.error("native bridge requires --native-root, --directory and --request")
        native_bridge(args.operation, args.native_root, args.directory, args.request)


if __name__ == "__main__":
    main()
