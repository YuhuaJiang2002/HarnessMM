#!/usr/bin/env python3
"""Complete official optimizers with TRAIN-only ALFWorld feedback and a sealed test.

This is a new successor identity; the running native validation source is not
modified. The native actor is the already verified 50-step runner. Official
SkillOpt and AutoSaddler own their complete outer loops, while this adapter owns
native episode I/O, provider accounting, primary split boundaries and resources.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_chisel_interactive_shared226 as native  # noqa: E402

SCHEMA = "chisel-interactive-complete-optimizers/v1"
METHODS = ("skillopt", "autosaddler")
REVISIONS = {
    "skillopt": "79124b37e9a6371e13b753f8bcd7adb1e493ade1",
    "autosaddler": "30e20ce004486c58e7ee97c66182a8d0d41ec90e",
}
ACTOR_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
IMPLEMENTATION_LABEL = "AutoSaddler V2 ALFWorld scenario adapter (PhiAgent)"


def request_json(url: str, value: dict | None = None, *, timeout: int = 120):
    payload = None if value is None else json.dumps(value).encode()
    request = Request(url, payload, {"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def internal_feedback_roles(items: list[dict], size: int, seed: int) -> dict:
    """Assign optimizer feedback roles inside TRAIN without changing the primary split."""
    groups = {}
    for item in items:
        groups.setdefault(native.scenario(item), []).append(item)
    previous = {0: None}
    for name in sorted(groups, key=lambda key: native.digest([seed, "optimizer-feedback", key])):
        for count in sorted(previous, reverse=True):
            following = count + len(groups[name])
            if following <= size and following not in previous:
                previous[following] = (count, name)
        if size in previous:
            break
    native.require(size in previous, "Cannot form the internal TRAIN feedback cohort by whole groups")
    selected = set()
    remaining = size
    while remaining:
        remaining, name = previous[remaining]
        selected.add(name)
    roles = {
        "train": [item for item in items if native.scenario(item) not in selected],
        "development": [item for item in items if native.scenario(item) in selected],
    }
    native.require(all(roles.values()), "Optimizer feedback roles must both be nonempty")
    native.require(sum(map(len, roles.values())) == len(items), "Optimizer dropped TRAIN items")
    return roles


def global_policy(skill: str) -> dict:
    from phiagent.learning.alfworld_autosaddler_pfn import TASK_FAMILIES
    value = {"skills": dict.fromkeys(TASK_FAMILIES, skill)}
    value["harness_sha256"] = native.digest(value)
    return value


def assert_phase_ids(binding: dict, ids: list[str], phase: str) -> list[dict]:
    native.require(phase in ("optimizer", "validation", "test"), "Unknown primary phase")
    primary = "test" if phase == "test" else "validation" if phase == "validation" else "train"
    allowed = {item["id"]: item for item in binding["partition"]["splits"][primary]}
    native.require(bool(ids) and len(ids) == len(set(ids)), "Empty or duplicate native request")
    native.require(set(ids) <= set(allowed), f"{phase} request crossed the primary {primary} boundary")
    return [allowed[ident] for ident in ids]


def prepare(config_path: Path) -> None:
    native.remote_only()
    config = native.read(config_path)
    output = Path(config["prepared_directory"])
    output.mkdir(parents=True, exist_ok=False)
    partition = native.read(Path(config["native_prepared"]) / "PARTITION.json")
    native.verify_partition(partition)
    native.require(partition["protocol_sha256"] == config["partition_sha256"],
                   "Primary partition identity changed")
    old_freeze = native.read(Path(config["native_prepared"]) / "FROZEN.json")
    native.verify_signed(old_freeze, "freeze_sha256")
    native.require(old_freeze["freeze_sha256"] == config["native_freeze_sha256"],
                   "Different native predecessor")
    roles = internal_feedback_roles(partition["splits"]["train"],
                                    config["internal_feedback_cases"], partition["seed"])
    provenance = native.read(ROOT / "SOURCE_PROVENANCE.json")
    binding = {
        "schema_version": SCHEMA, "partition": partition, "optimizer_feedback": roles,
        "primary_counts": old_freeze["partition"]["counts"],
        "native_predecessor_freeze": old_freeze["freeze_sha256"],
        "native_policies": {key: old_freeze["policies"][key] for key in native.NATIVE_METHODS},
        "checkpoint": old_freeze["checkpoint"], "source_provenance": provenance,
        "source_hashes": provenance["source_hashes"],
        "optimizer_revisions": REVISIONS, "actor_seed": 9401,
        "optimizer_seed": 20260928, "primary_partition_changed": False,
        "feedback_boundary": "Both optimizer train and internal development are primary TRAIN only",
        "validation_role": "External validation of the single official TRAIN-selected candidate",
        "cost_mode": "All attempts recorded; unlimited cumulative tokens; no zero-imputed failures",
        "native_predecessor_sources_unchanged": True,
        "chisel_historical_optimizer_cost": "Unresolved lineage, separate from new execution",
    }
    binding["binding_sha256"] = native.digest(binding)
    native.write(output / "BINDING.json", binding)
    workers = []
    for index, node in enumerate(config["nodes"]):
        previous = native.read(Path(config["native_prepared"]) / f"worker-{index:02d}.json")
        run = f"{config['run_root']}/{node['method']}"
        worker = {
            **previous, "schema_version": SCHEMA, "method": node["method"],
            "prepared_directory": str(output), "run_directory": run,
            "predecessor_run": previous["run_directory"],
            "binding_sha256": binding["binding_sha256"], "source_root": str(ROOT),
            "optimizer_python": node["optimizer_python"],
            "model_port": config["model_port_base"] + index,
            "gateway_port": config["gateway_port_base"] + index,
            "gateway_address": node["address"], "peer": config["nodes"][1 - index]["address"],
            "peer_gateway_port": config["gateway_port_base"] + 1 - index,
            "optimizer_max_tokens": config["optimizer_max_tokens"],
            "optimizer_context_tokens": config["optimizer_context_tokens"],
            "skillopt_epochs": config["skillopt_epochs"],
            "autosaddler_iterations": config["autosaddler_iterations"],
            "optimizer_batch_size": config["optimizer_batch_size"],
            "optimizer_workers": config["optimizer_workers"],
            "enable_idle_peer_actor_help": True,
            "cumulative_token_mode": "unlimited",
        }
        worker["model"] = deepcopy(previous["model"])
        worker["model"]["max_model_len"] = config["optimizer_context_tokens"]
        worker["resource"] = deepcopy(previous["resource"])
        worker["resource"]["port"] = worker["model_port"]
        path = output / f"worker-{index:02d}.json"
        native.write(path, worker)
        workers.append({"index": index, "host": node["address"], "config": str(path),
                        "run": run, "method": node["method"], "python": node["optimizer_python"]})
    native.write(output / "PREPARED.json", {
        "binding_sha256": binding["binding_sha256"], "workers": workers,
        "role_counts": {key: len(value) for key, value in roles.items()},
        "primary_counts": binding["primary_counts"], "source_ready": True,
        "queue": ["wait_native_retirement", "complete_optimizer", "external_validation",
                  "help_slow_peer_if_ready", "four_method_seal", "full_test", "audit_and_retire"],
    })
    print(json.dumps({"status": "PREPARED", "binding_sha256": binding["binding_sha256"],
                      "roles": {key: len(value) for key, value in roles.items()}}), flush=True)


def actor_binding(config: dict, root: Path, manifest: dict, *, help_from_peer: bool) -> dict:
    endpoints = [f"http://127.0.0.1:{config['model_port']}/v1"]
    if help_from_peer:
        endpoints.append(
            f"http://{config['peer']}:{config['peer_gateway_port']}/{config['method']}_actor/v1")
    binding = {
        "hostname": socket.gethostname(), "manifest_sha256": manifest["manifest_sha256"],
        "adapter_source_hashes": native.read(Path(config["prepared_directory"]) /
                                             "BINDING.json")["source_hashes"],
        "model": {"id": config["model"]["served_name"], "revision": native.REVISION,
                  "endpoints": endpoints},
        "gpu": {"visible_devices": "", "service_selection": str(Path(config["run_directory"]) /
                                                               "GPU_SELECTION.json")},
        "skillopt_root": config["skillopt_root"], "data_root": config["data_root"],
        "ledger": str(root / "usage-ledger.json"), "max_model_len": 32768,
        "local_discriminator_logprobs_mode": "processed_logprobs", "typesafe_api_used": False,
    }
    return binding


def episode_export(row: dict) -> tuple[dict, list[dict]]:
    history = native.read(Path(row["artifact_dir"]) / "conversation.json")
    turns = [{
        "step": step["turn"] + 1, "action": step["action"],
        "reasoning": (re.search(r"<think>(.*?)</think>", step["model_response"], re.S).group(1)
                      if re.search(r"<think>(.*?)</think>", step["model_response"], re.S) else ""),
        "model_response": step["model_response"], "env_feedback": step["observation"],
        "done": step["done"], "won": step["official_won"],
    } for step in history]
    exported = {
        **row, "hard": row["official_success"], "soft": float(row["official_success"]),
        "n_turns": row["turns"], "agent_ok": row["valid"], "instruction_type": row["task_type"],
        "question": row["task_description"],
        "fail_reason": "" if row["official_success"] else
                       f"Native episode failed ({row['stop_reason']}, {row['turns']} steps)",
        "public_trace": [{"action": step["action"], "observation": step["observation"]}
                         for step in history],
    }
    return exported, turns


def evaluate_request(request_path: Path, output_path: Path) -> None:
    native.remote_only()
    request = native.read(request_path)
    native.verify_signed(request, "request_sha256")
    config = native.read(Path(request["runtime_manifest_path"]))
    full_binding = native.read(Path(config["prepared_directory"]) / "BINDING.json")
    native.verify_signed(full_binding, "binding_sha256")
    native.require(full_binding["binding_sha256"] == config["binding_sha256"], "Wrong source binding")
    phase = request.get("primary_phase", "optimizer")
    ids = request["task_ids"]
    items = assert_phase_ids(full_binding, ids, phase)
    if phase == "test":
        seal = native.read(Path(config["run_directory"]) / "TEST_SEAL.json")
        native.require(seal["binding_sha256"] == config["binding_sha256"], "Unsealed test")
    root = Path(request["rollout_root"])
    root.mkdir(parents=True, exist_ok=False)
    from scripts import run_alfworld_modular_rsi as core
    from scripts.run_alfworld_autosaddler_worker import candidate_skill
    from scripts.alfworld_jev_runtime import bind_local_labels
    native.write(root / "config.json", config)
    native.write(root / "command.json", {"argv": sys.argv, "at_utc": native.now(),
                                       "request_sha256": request["request_sha256"]})
    native.write(root / "NATIVE_BATCH.json", {"primary_phase": phase, "method": config["method"],
                                            "task_ids": ids, "request_sha256": request["request_sha256"]})
    asset_groups = {}
    for relative, checksum in full_binding["partition"]["data_hashes"].items():
        asset_groups.setdefault(str(Path(relative).parent), {})[relative] = checksum
    for item in items:
        assets = asset_groups.get(str(Path(item["gamefile"]).parent))
        native.require(bool(assets), "Frozen task assets are absent from the partition manifest")
        for relative, checksum in assets.items():
            native.require(native.file_hash(Path(config["data_root"]) / relative) == checksum,
                           f"Frozen benchmark asset changed: {relative}")
    manifest = {
        "schema_version": core.repair.SCHEMA, "budget": config["episode_budget"],
        "alfworld_environment_mode": "eval_out_of_distribution",
        "task_inventory": items, "seed": 20260928,
    }
    manifest["manifest_sha256"] = native.digest(manifest)
    native.write(root / "manifest.json", manifest)
    native.write(root / "usage-ledger.json", {
        "used_tokens": 0, "reserved_tokens": 0, "attempts": 0, "accounted_attempts": 0,
        "cost_unknown": False, "maximum_tokens": None, "maximum_attempts": None,
        "deadline_unix": None, "cumulative_token_mode": "unlimited",
    })
    policies = request.get("policies")
    if policies is None:
        from phiagent.learning.alfworld_autosaddler_pfn import TASK_FAMILIES
        policies = {
            config["method"]: {
                "skills": {family: candidate_skill(
                    surface=request["surface"], components=request["components"],
                    task_type=family) for family in TASK_FAMILIES},
            },
        }
        for policy in policies.values():
            policy["harness_sha256"] = native.digest(policy)
    has_shared = any("local_discriminator" in policy for policy in policies.values())
    helping = request.get("help_from_peer", False) and not has_shared
    binding = actor_binding(config, root, manifest, help_from_peer=helping)
    if has_shared:
        binding["local_label_token_ids"] = bind_local_labels(binding, root / "LOCAL_TOKENIZER.json")
    binding["binding_sha256"] = native.digest(binding)
    native.write(root / "binding.json", binding)
    jobs = [{
        "study": str(root), "phase": "episodes", "arm": method, "item": item,
        "policy": policy, "endpoint_index": (index % 2 if helping else 0),
        "verify_native_interface": True,
        "alfworld_modular_rsi": {"max_steps": 50, "episode_seed": 9401},
    } for index, item in enumerate(items) for method, policy in policies.items()]
    records = []
    try:
        cache = None
        fresh_jobs = jobs
        if config.get("native_cache_manifest"):
            from scripts import chisel_alfworld_native_cache as reuse
            native.require(phase == "optimizer" and set(policies) == {"autosaddler"} and not helping,
                           "Only exact local AutoSaddler TRAIN feedback can use this cache")
            cache = reuse.load_for_request(config, full_binding, ROOT)
            fresh_jobs = []
            for job in jobs:
                row = reuse.lookup(cache, job["item"], job["policy"])
                if row is None:
                    fresh_jobs.append(job)
                    continue
                native.append(root / "rows.jsonl", row)
                exported, turns = episode_export(row)
                records.append(exported)
                native.write(root / "predictions" / row["id"] / "conversation.json", turns)
            native.write(root / "CACHE_REUSE.json", {
                "cache_sha256": cache["cache_sha256"], "reused_cells": len(records),
                "fresh_cells": len(fresh_jobs), "new_calls_for_reused_cells": 0,
                "physical_cost_inherited_not_recharged": True,
            })
        workers = config["workers"] * (2 if helping else 1)
        with ProcessPoolExecutor(max_workers=workers, initializer=native.initialize_worker) as pool:
            pending = {pool.submit(core.repair.run_episode, job): job for job in fresh_jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    row = future.result()
                    native.append(root / "rows.jsonl", row)
                    native.validate_row(row, job["item"], job["arm"],
                                        shared_control="local_discriminator" in job["policy"])
                except BaseException:
                    for queued in pending:
                        queued.cancel()
                    raise
                exported, turns = episode_export(row)
                records.append(exported)
                if len(policies) == 1:
                    native.write(root / "predictions" / row["id"] / "conversation.json", turns)
                native.write(root / "PROGRESS.json", {
                    "at_utc": native.now(), "completed": len(records), "expected": len(jobs),
                    "primary_phase": phase, "method": config["method"],
                })
                if not (Path(config["run_directory"]) / "FIRST_NATIVE_EPISODE.json").exists():
                    native.write(Path(config["run_directory"]) / "FIRST_NATIVE_EPISODE.json", row)
                    native.write(Path(config["run_directory"]) / "STATE.json", {
                        "status": "COMPLETE_OPTIMIZER_RUNNING", "at_utc": native.now(),
                        "first_native_terminal_verified": True})
        expected = {(item["id"], method) for item in items for method in policies}
        native.require(len(records) == len(expected) and
                       {(row["id"], row["method"]) for row in records} == expected,
                       "Incomplete native population")
        records.sort(key=lambda row: (ids.index(row["id"]), row["method"]))
        usage = {key: sum(row["usage"][key] for row in records) for key in ACTOR_KEYS}
        new_usage = {key: sum(row["usage"][key] for row in records
                              if "execution_reuse" not in row) for key in ACTOR_KEYS}
        ledger = core.ledger_snapshot(root / "usage-ledger.json")
        native.require(not ledger["cost_unknown"] and ledger["reserved_tokens"] == 0
                       and ledger["used_tokens"] == new_usage["total_tokens"],
                       "Native batch accounting does not reconcile")
        result = {
            "schema_version": "phiagent-alfworld-candidate-eval-result/v1",
            "implementation_label": request["implementation_label"],
            "task_ids": ids, "phase": request["phase"], "surface": request["surface"],
            "rows": records, "usage": usage, "official_test_calls": int(phase == "test") * len(jobs),
            "request_sha256": request["request_sha256"], "primary_phase": phase,
            "help_from_peer": helping, "benchmark_model": config["model"]["served_name"],
            "new_physical_usage": new_usage,
            "inherited_native_usage": {key: usage[key] - new_usage[key] for key in ACTOR_KEYS},
            "reused_native_cells": sum("execution_reuse" in row for row in records),
        }
        result["canonical_sha256"] = native.digest(result)
        native.write(output_path, result)
    except BaseException as error:
        native.write(root / "FAILURE.json", {"error": str(error), "traceback": traceback.format_exc(),
                                            "at_utc": native.now()})
        raise


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config: dict, root: Path):
        self.config, self.root = config, root
        self.lock = threading.Lock()
        self.sequence = 0
        self.optimizer_open = True
        self.helper_available = False
        self.failure = None
        super().__init__(("0.0.0.0", config["gateway_port"]), GatewayHandler)

    def summary(self) -> dict:
        rows = [native.read(path) for path in (self.root / "provider").glob("*/accounting.json")]
        optimizer = [row for row in rows if row["role"] == "optimizer"]
        return {
            "calls": len(optimizer), "unknown_usage_calls": sum(
                row["usage"] is None for row in optimizer),
            "failed_calls": sum(row["status"] != "COMPLETE" for row in optimizer),
            **{key: sum(row["usage"][key] for row in optimizer if row["usage"] is not None)
               for key in ACTOR_KEYS},
            "helper_calls": len(rows) - len(optimizer), "cumulative_token_mode": "unlimited",
        }


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def allowed(self) -> bool:
        return self.client_address[0] in {
            "127.0.0.1", self.server.config["gateway_address"], self.server.config["peer"],
        }

    def send_json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed() or self.path != "/status":
            self.send_error(403)
            return
        names = ("STATE", "CANDIDATE", "VALIDATION_COMPLETE", "TEST_COMPLETE",
                 "FAILURE", "PREDECESSOR_NATIVE_COMPLETE")
        self.send_json({
            "binding_sha256": self.server.config["binding_sha256"],
            "method": self.server.config["method"],
            "helper_available": self.server.helper_available,
            "at_utc": native.now(),
            **{name: native.read(self.server.root / f"{name}.json")
               for name in names if (self.server.root / f"{name}.json").exists()},
        })

    def do_POST(self):
        config, root = self.server.config, self.server.root
        if not self.allowed():
            self.send_error(403)
            return
        role = "optimizer" if self.path == "/optimizer/v1/chat/completions" else "helper"
        allowed_helper = self.path in (
            "/skillopt_actor/v1/chat/completions", "/autosaddler_actor/v1/chat/completions")
        if role == "helper" and (not allowed_helper or not self.server.helper_available):
            self.send_error(409, "Peer actor capacity not released")
            return
        if role == "optimizer" and not self.server.optimizer_open:
            self.send_error(409, "Optimizer is sealed")
            return
        with self.server.lock:
            self.server.sequence += 1
            sequence = self.server.sequence
        output = root / "provider" / f"{sequence:07d}"
        output.mkdir(parents=True)
        usage, forwarded, status = None, False, "FAILED"
        reply, reply_status = {}, 200
        try:
            native.require(self.server.failure is None, "A provider failure requires diagnosis")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            native.require(body.get("model") == config["model"]["served_name"]
                           and not body.get("stream"), "Wrong model or streaming transport")
            requested = body.pop("max_completion_tokens", body.get("max_tokens", 16384))
            maximum = config["optimizer_max_tokens"] if role == "optimizer" else 2048
            native.require(type(requested) is int and requested > 0, "Invalid output capacity")
            body.update(max_tokens=min(requested, maximum), seed=9401,
                        chat_template_kwargs={"enable_thinking": False})
            if role == "helper":
                native.require(body.get("temperature") == 0, "Peer native actor temperature differs")
            native.write(output / "request.json", {"role": role, "path": self.path, "json": body})
            token_request = {key: body[key] for key in (
                "model", "messages", "tools", "tool_choice", "chat_template_kwargs") if key in body}
            token_request["add_generation_prompt"] = True
            endpoint = f"http://127.0.0.1:{config['model_port']}"
            tokenized = request_json(endpoint + "/tokenize", token_request)
            native.write(output / "tokenize.json", tokenized)
            limit = config["optimizer_context_tokens"] if role == "optimizer" else 32768
            if tokenized["count"] + body["max_tokens"] > limit:
                usage = dict.fromkeys(ACTOR_KEYS, 0)
                raise ValueError("Native request exceeds its per-call model context capacity")
            forwarded = True
            response = request_json(endpoint + "/v1/chat/completions", body, timeout=1800)
            native.write(output / "response.json", response)
            incoming = response.get("usage", {})
            native.require(all(type(incoming.get(key)) is int and incoming[key] >= 0
                               for key in ACTOR_KEYS)
                           and incoming["total_tokens"] == incoming["prompt_tokens"] +
                           incoming["completion_tokens"], "Provider usage is incomplete")
            usage = {key: incoming[key] for key in ACTOR_KEYS}
            native.require(bool(response.get("choices")), "Provider returned no choices")
            status = "COMPLETE"
            reply = response
        except Exception as error:
            self.server.failure = str(error)
            native.write(output / "error.json", {
                "error": str(error), "traceback": traceback.format_exc(), "forwarded": forwarded})
            reply = {"error": {"message": str(error), "type": "recorded_provider_failure"}}
            reply_status = 502
        finally:
            native.write(output / "accounting.json", {
                "role": role, "path": self.path, "status": status, "forwarded": forwarded,
                "usage": usage, "at_utc": native.now(),
                "cost_owner": config["method"] if role == "optimizer" else self.path.split("/")[1],
            })
        self.send_json(reply, reply_status)


class Study:
    def __init__(self, config: dict, binding: dict, root: Path, gateway: Gateway):
        self.config, self.binding, self.root, self.gateway = config, binding, root, gateway
        self.role_items = binding["optimizer_feedback"]
        self.native_sequence = 0
        self.peer_lock = threading.Lock()

    def peer_status(self) -> dict:
        try:
            status = request_json(
                f"http://{self.config['peer']}:{self.config['peer_gateway_port']}/status", timeout=30)
        except OSError:
            saved = self.root / "PEER_LAST.json"
            if saved.exists():
                status = native.read(saved)
                if "FAILURE" in status or "TEST_COMPLETE" in status:
                    return status
            raise
        with self.peer_lock:
            path = self.root / "PEER_LAST.json"
            if path.exists():
                previous = native.read(path)
                if ("FAILURE" in previous or "TEST_COMPLETE" in previous) and not (
                    "FAILURE" in status or "TEST_COMPLETE" in status
                ):
                    return previous
            native.write(path, status)
        return status

    def batch(self, items: list[dict], skill: str, output: Path, *,
              phase: str = "optimizer", policies: dict | None = None) -> dict:
        native.require(self.gateway.failure is None, "Recorded provider failure blocks further work")
        self.native_sequence += 1
        output.mkdir(parents=True, exist_ok=True)
        request_path, result_path = output / "NATIVE_REQUEST.json", output / "NATIVE_RESULT.json"
        ids = [item["id"] for item in items]
        help_from_peer = False
        if phase in ("optimizer", "validation"):
            try:
                peer = self.peer_status()
            except OSError as error:
                native.append(self.root / "HELPER_STATUS.jsonl",
                              {"at_utc": native.now(), "unavailable": str(error)})
            else:
                native.require(peer["binding_sha256"] == self.config["binding_sha256"],
                               "Peer uses a different complete campaign")
                help_from_peer = peer["helper_available"]
        request = {
            "schema_version": "phiagent-alfworld-candidate-eval/v1",
            "implementation_label": IMPLEMENTATION_LABEL, "phase": str(output),
            "primary_phase": phase, "surface": "global", "components": {"global_skill": skill},
            "task_ids": ids, "runtime_manifest_path": str(self.root / "config.json"),
            "rollout_root": str(output / "native"), "help_from_peer": help_from_peer,
        }
        if policies is not None:
            request["policies"] = policies
        request["request_sha256"] = native.digest(request)
        if request_path.exists():
            previous = native.read(request_path)
            # Capacity ownership is execution metadata, not a candidate change.
            request["help_from_peer"] = previous["help_from_peer"]
            request.pop("request_sha256")
            request["request_sha256"] = native.digest(request)
            native.require(previous == request and result_path.is_file(),
                           "An incomplete native batch requires a new explicit continuation")
        else:
            native.write(request_path, request)
            command = [self.config["actor_python"], "-u", str(Path(__file__).resolve()),
                       "evaluate", "--request", str(request_path), "--output", str(result_path)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
            with (output / "native.log").open("x") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               env=environment, check=True)
        result = native.read(result_path)
        native.verify_signed(result, "canonical_sha256")
        native.require(result["request_sha256"] == request["request_sha256"],
                       "Native adapter returned a different request")
        predictions = output / "predictions"
        if phase != "test":
            for row in result["rows"]:
                _, turns = episode_export(row)
                native.write(predictions / row["id"] / "conversation.json", turns)
        with (output / "results.jsonl").open("w") as stream:
            for row in result["rows"]:
                stream.write(json.dumps(row) + "\n")
        return result

    def skillopt(self) -> str:
        from skillopt.config import flatten_config, load_config
        from skillopt.engine.trainer import ReflACTTrainer
        from skillopt.envs.alfworld.adapter import ALFWorldAdapter
        from skillopt.model import qwen_backend
        from scripts.chisel_skillopt_usage_v2 import IncompleteRun
        study = self
        original = qwen_backend._chat_messages_impl
        signature = inspect.signature(original)

        @wraps(original)
        def strict_transport(*args, **kwargs):
            supplied = signature.bind(*args, **kwargs)
            supplied.arguments["retries"] = 1
            try:
                return original(*supplied.args, **supplied.kwargs)
            except Exception as error:
                native.write(study.root / "OPTIMIZER_PROVIDER_FAILURE.json", {
                    "error": str(error), "traceback": traceback.format_exc(),
                    "automatic_retries": 0, "at_utc": native.now()})
                raise IncompleteRun("Complete SkillOpt provider failed; no success fallback") from error

        qwen_backend._chat_messages_impl = strict_transport
        split = self.root / "optimizer-splits"
        for role, folder in (("train", "train"), ("development", "val")):
            rows = [{**item, "gamefile": str(Path(self.config["data_root"]) / item["gamefile"])}
                    for item in self.role_items[role]]
            native.write(split / folder / "items.json", rows)
        native.write(split / "test/items.json", [])

        class Adapter(ALFWorldAdapter):
            @property
            def _env_name(self):
                return "alfworld"

            def rollout(self, env_manager, skill_content, out_dir, **kwargs):
                items = list(env_manager.items or [])
                assert_phase_ids(study.binding, [item["id"] for item in items], "optimizer")
                try:
                    return study.batch(items, skill_content, Path(out_dir))["rows"]
                except Exception as error:
                    native.write(study.root / "OPTIMIZER_NATIVE_FAILURE.json", {
                        "error": str(error), "traceback": traceback.format_exc(),
                        "at_utc": native.now(), "output": out_dir})
                    raise IncompleteRun("Native feedback failed; no incomplete optimizer admission") from error

        source = ROOT / "external/SkillOpt"
        cfg = flatten_config(load_config(str(source / "configs/alfworld/default.yaml")))
        cfg.update(
            out_root=str(self.root / "skillopt"), env_name="alfworld",
            skill_init=str(source / "skillopt/envs/alfworld/skills/initial.md"),
            split_mode="split_dir", split_dir=str(split), train_size=len(self.role_items["train"]),
            batch_size=self.config["optimizer_batch_size"], accumulation=1,
            num_epochs=self.config["skillopt_epochs"], seed=20260928, eval_test=False,
            sel_env_num=len(self.role_items["development"]), test_env_num=0,
            optimizer_backend="qwen_chat", target_backend="qwen_chat",
            optimizer_model=self.config["model"]["served_name"],
            target_model=self.config["model"]["served_name"],
            qwen_chat_base_url=f"http://127.0.0.1:{self.config['gateway_port']}/optimizer/v1",
            qwen_chat_api_key="EMPTY", qwen_chat_thinking_mode="disabled",
            qwen_chat_temperature=0.7, qwen_chat_max_tokens=self.config["optimizer_max_tokens"],
            qwen_chat_timeout_seconds=1800, gate_metric="hard", use_semantic_density=False,
            use_gate=True, use_slow_update=True, slow_update_samples=20,
            slow_update_gate_with_selection=False, longitudinal_pair_policy="mixed",
            use_meta_skill=True, use_skill_aware_reflection=False, skill_update_mode="patch",
            learning_rate=4, min_learning_rate=2, lr_scheduler="cosine", lr_control_mode="fixed",
            minibatch_size=8, merge_batch_size=8, failure_only=False,
        )
        os.environ.update(OPTIMIZER_QWEN_CHAT_GENERATION_SEED="9401",
                          TARGET_QWEN_CHAT_GENERATION_SEED="9401")
        native.write(self.root / "SKILLOPT_CONFIG.json", cfg)
        adapter = Adapter(split_dir=str(split), seed=20260928, train_size=0, max_steps=50,
                          workers=self.config["workers"], max_api_workers=self.config["workers"],
                          analyst_workers=self.config["optimizer_workers"], max_completion_tokens=2048)
        result = ReflACTTrainer(cfg, adapter).train()
        native.write(self.root / "SKILLOPT_RETURN.json", result)
        native.require(self.gateway.failure is None, "SkillOpt completed after a provider failure")
        native.require({row["epoch"] for row in result["epoch_stats"]} ==
                       set(range(1, self.config["skillopt_epochs"] + 1))
                       and result["final_selection_hard"] is not None,
                       "SkillOpt did not complete every epoch and final candidate selection")
        selected = self.root / "skillopt/best_skill.md"
        native.require(selected.is_file(), "Complete SkillOpt did not export its selected skill")
        native.write(self.root / "OPTIMIZER_COMPLETE.json", {
            "method": "skillopt", "official_engine_returned": True,
            "epochs": self.config["skillopt_epochs"], "provider": self.gateway.summary(),
            "primary_train_population": 710, "internal_roles": {
                key: len(value) for key, value in self.role_items.items()}, "at_utc": native.now(),
        })
        return selected.read_text()

    def autosaddler(self) -> str:
        import yaml
        from autosaddler.v2.config.registry import ScenarioRegistration, build_runtime, default_registry
        from autosaddler.v2.plugins.api import SCENARIO_PLUGIN_API_VERSION, ScenarioPlugin
        from phiagent_autosaddler.alfworld_scenario import build_alfworld_components
        from phiagent_autosaddler.openai_function_provider import build_openai_function_provider
        study = self
        seed_skill = (ROOT / "external/SkillOpt/skillopt/envs/alfworld/skills/initial.md").read_text()
        split_dirs = {}
        for role, items in self.role_items.items():
            directory = self.root / "optimizer-splits" / role
            native.write(directory / "val/items.json", items)
            split_dirs[role] = str(directory)
        settings = {
            "surface": "global", "baseline": {"global_skill": seed_skill},
            "component_descriptions": {"global_skill": "Native ALFWorld task and action skill"},
            "train_cases": [{key: item[key] for key in ("id", "task_type")}
                            for item in self.role_items["train"]],
            "development_cases": [{key: item[key] for key in ("id", "task_type")}
                                  for item in self.role_items["development"]],
            "split_dirs": split_dirs, "runtime_manifest_path": str(self.root / "config.json"),
            "frozen_runner_path": str(Path(__file__).resolve()),
            "evaluator_command": [self.config["actor_python"], str(Path(__file__).resolve()),
                                  "evaluate", "--request", "{request}", "--output", "{output}"],
            "evaluator_timeout_seconds": 7200,
            "base_url": f"http://127.0.0.1:{self.config['model_port']}/v1",
            "autosaddler_revision": REVISIONS["autosaddler"],
            "source_sha256": self.binding["source_hashes"],
            "benchmark_identity": {
                "benchmark": "ALFWorld", "primary_partition": self.config["binding_sha256"],
                "metric": "Native won AND done", "train_population": 710,
                "internal_development_is_primary_train": True, "test": "SEALED",
            },
        }

        def factory(**kwargs):
            components = build_alfworld_components(**kwargs)
            store, run_dir = kwargs["store"], kwargs["run_dir"]

            class Evidence:
                def build(self, evaluation):
                    native.require(evaluation.split == "train", "Only TRAIN evidence may enter diagnosis")
                    cases = []
                    for observation in evaluation.observations:
                        payload = native.read(run_dir / observation.output.uri)
                        row = next(row for row in payload["rows"] if row["id"] == observation.case_id)
                        assert_phase_ids(study.binding, [row["id"]], "optimizer")
                        cases.append({
                            "case_id": row["id"], "task_type": row["task_type"],
                            "question": row["task_description"], "success": row["hard"],
                            "fail_reason": row["fail_reason"], "public_trace": row["public_trace"],
                        })
                    return store.write_json(
                        f"evidence/{native.digest(evaluation.evaluation_id)}/evidence.json",
                        {"schema_version": SCHEMA, "case_records": cases,
                         "success_rate": evaluation.aggregate_score,
                         "information_boundary": "Primary TRAIN public trajectories and native feedback only"},
                        kind="alfworld-training-evidence")

            return replace(components, evidence_builder=Evidence())

        registry = default_registry()
        registry.register_scenario(ScenarioRegistration(plugin=ScenarioPlugin(
            name="phiagent_alfworld", api_version=SCENARIO_PLUGIN_API_VERSION, factory=factory),
            source="builtin"))
        registry.providers["phiagent_openai_function"] = build_openai_function_provider
        cfg = {
            "schema_version": "autosaddler/v2",
            "scenario": {"type": "phiagent_alfworld", "settings": settings},
            "optimization": {
                "task_selection": {"type": "epoch_shuffled",
                                   "batch_size": self.config["optimizer_batch_size"], "seed": 20260928},
                "acceptance": {"type": "matched_valid_strict_improvement"},
                "development": {"type": "full_on_accept"},
                "ranking": {"type": "mean_development_score"},
                "budget": {"max_iterations": self.config["autosaddler_iterations"],
                           "max_rollouts": 100000},
                "diagnosis_patch_timeout_seconds": 1800, "selection_timeout_seconds": 1800,
                "reflection_timeout_seconds": 1800, "session_retries": 0,
                "session_retry_backoff_seconds": 0,
            },
            "provider": {
                "type": "phiagent_openai_function",
                "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
                "settings": {
                    "model": self.config["model"]["served_name"], "reasoning_effort": None,
                    "max_tool_steps": 48, "temperature": 0.7,
                    "max_completion_tokens": self.config["optimizer_max_tokens"],
                    "enable_thinking": False, "tool_choice_mode": "qwen_text",
                    "provider": {
                        "type": "openai", "base_url":
                        f"http://127.0.0.1:{self.config['gateway_port']}/optimizer/v1",
                        "wire_api": "chat_completions",
                        "model_id": self.config["model"]["served_name"],
                        "wire_model": self.config["model"]["served_name"],
                    },
                },
            },
            "storage": {"type": "local", "run_root": str(self.root / "autosaddler")},
        }
        path = self.root / "AUTOSADDLER_CONFIG.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        runtime = build_runtime(path, run_id="native-train-only", registry=registry)
        result = runtime.engine.run()
        native.require(self.gateway.failure is None, "AutoSaddler completed after a provider failure")
        native.require(result.iterations == self.config["autosaddler_iterations"],
                       "AutoSaddler did not complete all configured optimizer rounds")
        materialized = runtime.scenario.harness_space.materialize(
            runtime.engine._state().candidates[result.selected_candidate_id], "evaluate")
        try:
            manifest = native.read(materialized.root / "candidate.json")
            native.require(set(manifest) == {"global_skill"}, "Selected AutoSaddler schema differs")
            skill = manifest["global_skill"]
        finally:
            materialized.release()
        native.write(self.root / "OPTIMIZER_COMPLETE.json", {
            "method": "autosaddler", "official_engine_returned": True,
            "selected_candidate_id": result.selected_candidate_id,
            "iterations": result.iterations, "provider": self.gateway.summary(),
            "primary_train_population": 710, "internal_roles": {
                key: len(value) for key, value in self.role_items.items()}, "at_utc": native.now(),
        })
        return skill


def audit_result_costs(root: Path) -> dict:
    grouped = {}
    for path in root.rglob("NATIVE_RESULT.json"):
        result = native.read(path)
        native.verify_signed(result, "canonical_sha256")
        phase = result["primary_phase"]
        target = grouped.setdefault(phase, {key: 0 for key in ACTOR_KEYS})
        for key in ACTOR_KEYS:
            target[key] += result["usage"][key]
    # AutoSaddler's native bridge names result files output.json.
    for path in (root / "autosaddler").rglob("output.json"):
        result = native.read(path)
        if result.get("schema_version") != "phiagent-alfworld-candidate-eval-result/v1":
            continue
        native.verify_signed(result, "canonical_sha256")
        target = grouped.setdefault(result["primary_phase"], {key: 0 for key in ACTOR_KEYS})
        for key in ACTOR_KEYS:
            target[key] += result["usage"][key]
    return grouped


def aggregate_test(own: dict, peer: dict, binding: dict) -> dict:
    expected = {item["id"] for item in binding["partition"]["splits"]["test"]}
    own_ids, peer_ids = set(own["test_task_ids"]), set(peer["test_task_ids"])
    native.require(not own_ids & peer_ids and own_ids | peer_ids == expected,
                   "Incomplete or overlapping final test populations")
    native.require(own["policy_hashes"] == peer["policy_hashes"]
                   and own["binding_sha256"] == peer["binding_sha256"],
                   "Different policies or protocols entered the final comparison")
    methods = {}
    for method in native.MAIN_METHODS:
        rows = [result["summary"][method] for result in (own, peer)]
        combined = {key: sum(row[key] for row in rows) for key in rows[0] if key != "native_success_rate"}
        native.require(combined["tasks"] == len(expected), "A method has incomplete test coverage")
        combined["native_success_rate"] = combined["successes"] / len(expected)
        methods[method] = combined
    chisel = methods["chisel_shared_support_first"]["successes"]
    return {
        "at_utc": native.now(), "status": "COMPLETE_FOUR_METHOD_NATIVE_COMPARISON",
        "binding_sha256": own["binding_sha256"], "methods": methods,
        "chisel_strict_success_gate": all(
            chisel > methods[method]["successes"] for method in ("no_skill", "skillopt", "autosaddler")),
        "test_label": "Custom held-out test from the complete official TRAIN pool; prior exposure disclosed",
        "paper_admitted": False, "paper_admission_owner": "Parent evidence audit",
    }


def run(config_path: Path) -> None:
    config = native.read(config_path)
    native.remote_only(config["authorized_hostname"])
    root = Path(config["run_directory"])
    root.mkdir(parents=True, exist_ok=False)
    native.write(root / "config.json", config)
    native.write(root / "PID.json", {"pid": os.getpid(), "at_utc": native.now(),
                                    "hostname": socket.gethostname(), "argv": sys.argv})
    binding = native.read(Path(config["prepared_directory"]) / "BINDING.json")
    native.verify_signed(binding, "binding_sha256")
    native.require(binding["binding_sha256"] == config["binding_sha256"], "Wrong frozen successor")
    for name, checksum in binding["source_hashes"].items():
        native.require(native.file_hash(ROOT / name) == checksum, f"Changed successor source: {name}")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NO_PROXY="127.0.0.1,localhost,214.40.221.1,214.40.221.2")
    from scripts import run_alfworld_modular_rsi as core
    native.write(root / "PROVENANCE.json", {
        "source": binding["source_provenance"], "hostname": socket.gethostname(),
        "seed": 20260928, "argv": sys.argv,
        "optimizer_packages": core.repair.capture([sys.executable, "-m", "pip", "freeze"]),
        "actor_packages": core.repair.capture([config["actor_python"], "-m", "pip", "freeze"]),
        "model_packages": core.repair.capture([config["model"]["python"], "-m", "pip", "freeze"]),
    })
    process, descriptors, logfile, gateway = None, [], None, None
    stop = threading.Event()
    predecessor = Path(config["predecessor_run"])
    try:
        gateway = Gateway(config, root)
        gateway.optimizer_open = False
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        study = Study(config, binding, root, gateway)

        def heartbeat():
            audit_at = 0
            while not stop.is_set():
                native.write(root / "HEARTBEAT.json", {
                    "at_utc": native.now(), "provider": gateway.summary(),
                    "helper_available": gateway.helper_available,
                    "model_alive": process is not None and process.poll() is None,
                })
                try:
                    study.peer_status()
                except OSError as error:
                    native.write(root / "PEER_CONNECTION.json", {
                        "at_utc": native.now(), "error": str(error)})
                if time.monotonic() >= audit_at:
                    native.append(root / "CAPACITY_AUDIT.jsonl", native.resource_audit())
                    audit_at = time.monotonic() + 300
                stop.wait(min(120, max(1, audit_at - time.monotonic())))

        threading.Thread(target=heartbeat, daemon=True).start()
        while not (predecessor / "RETIREMENT.json").exists():
            native.require(not (predecessor / "FAILURE.json").exists(), "Native predecessor failed")
            native.write(root / "STATE.json", {"status": "WAITING_NATIVE_RETIREMENT",
                                               "at_utc": native.now(), "predecessor": str(predecessor)})
            time.sleep(120)
        previous = native.read(predecessor / "NATIVE_COMPLETE.json")
        native.require(previous["status"] == "NATIVE_TRAIN_VALIDATION_COMPLETE_TEST_BLOCKED"
                       and previous["freeze_sha256"] == binding["native_predecessor_freeze"],
                       "Native predecessor did not finish its frozen validation")
        native.write(root / "PREDECESSOR_NATIVE_COMPLETE.json", previous)
        for name, checksum in binding["checkpoint"]["files"].items():
            native.require(native.file_hash(Path(config["model"]["directory"]) / name) == checksum,
                           f"Checkpoint bytes changed: {name}")
        process, descriptors, logfile = core.start_service(root, config)
        core.await_service(process, f"http://127.0.0.1:{config['model_port']}/v1",
                           config["startup_timeout_seconds"])
        gateway.optimizer_open = True
        native.write(root / "STATE.json", {"status": "STARTING_COMPLETE_OPTIMIZER", "at_utc": native.now()})
        skill = getattr(study, config["method"])()
        gateway.optimizer_open = False
        policy = global_policy(skill)
        candidate = {"method": config["method"], "policy": policy,
                     "binding_sha256": config["binding_sha256"], "at_utc": native.now(),
                     "complete_optimizer": native.read(root / "OPTIMIZER_COMPLETE.json")}
        candidate["candidate_sha256"] = native.digest(candidate)
        native.write(root / "CANDIDATE.json", candidate)
        native.write(root / "STATE.json", {"status": "EXTERNAL_VALIDATION", "at_utc": native.now()})
        validation = study.batch(binding["partition"]["splits"]["validation"], skill,
                                 root / "external-validation", phase="validation")
        native.write(root / "VALIDATION_COMPLETE.json", {
            "binding_sha256": config["binding_sha256"], "candidate_sha256": candidate["candidate_sha256"],
            "summary": native.summarize(validation["rows"]), "at_utc": native.now(),
        })
        gateway.helper_available = True
        native.write(root / "STATE.json", {"status": "WAITING_PEER_READY_HELPER_AVAILABLE",
                                          "at_utc": native.now()})
        while True:
            native.require(process.poll() is None, "Owned model exited")
            try:
                peer = study.peer_status()
            except OSError as error:
                native.append(root / "PEER_STATUS.jsonl", {"at_utc": native.now(), "error": str(error)})
                time.sleep(120)
                continue
            native.require("FAILURE" not in peer, "Peer complete optimizer failed; test stays sealed")
            if "VALIDATION_COMPLETE" in peer:
                break
            time.sleep(120)
        native.require(peer["binding_sha256"] == config["binding_sha256"]
                       and "PREDECESSOR_NATIVE_COMPLETE" in peer, "Peer validation boundary differs")
        peer_candidate = peer["CANDIDATE"]
        native.verify_signed(peer_candidate, "candidate_sha256")
        policies = {**binding["native_policies"], config["method"]: policy,
                    peer["method"]: peer_candidate["policy"]}
        native.require(set(policies) == set(native.MAIN_METHODS), "A full comparator is missing")
        seal = {"binding_sha256": config["binding_sha256"], "at_utc": native.now(),
                "policies": policies, "own_candidate": candidate,
                "peer_candidate": peer_candidate, "all_validation_complete": True}
        seal["seal_sha256"] = native.digest(seal)
        native.write(root / "TEST_SEAL.json", seal)
        gateway.helper_available = False
        native.write(root / "STATE.json", {"status": "FULL_FOUR_METHOD_TEST", "at_utc": native.now()})
        items = native.ranked_items(binding["partition"], "test")[
            config["shard_index"]::config["shard_count"]]
        result = study.batch(items, "", root / "test", phase="test", policies=policies)
        test_complete = {
            "binding_sha256": config["binding_sha256"], "at_utc": native.now(),
            "summary": native.summarize(result["rows"]), "native_costs": audit_result_costs(root),
            "optimizer_cost": gateway.summary(), "paper_admitted": False,
            "test_task_ids": [item["id"] for item in items],
            "policy_hashes": {method: value["harness_sha256"] for method, value in policies.items()},
        }
        native.write(root / "TEST_COMPLETE.json", test_complete)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        for descriptor in descriptors:
            os.close(descriptor)
        descriptors.clear()
        native.write(root / "STATE.json", {
            "status": "WAITING_PEER_TEST_COMPLETE_GPU_RELEASED", "at_utc": native.now()})
        while True:
            try:
                peer = study.peer_status()
            except OSError as error:
                native.append(root / "PEER_STATUS.jsonl", {"at_utc": native.now(), "error": str(error)})
                time.sleep(120)
                continue
            native.require("FAILURE" not in peer, "Peer test failed; no complete comparison")
            if "TEST_COMPLETE" in peer:
                break
            time.sleep(120)
        native.write(root / "GLOBAL_RESULT.json", aggregate_test(test_complete, peer["TEST_COMPLETE"], binding))
        native.write(root / "STATE.json", {"status": "COMPLETE", "at_utc": native.now()})
        time.sleep(240)
        stop.set()
    except BaseException as error:
        native.write(root / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "at_utc": native.now(),
            "automatic_retries": 0, "test_admitted": False,
        })
        raise
    finally:
        if gateway is not None:
            gateway.optimizer_open = False
            gateway.helper_available = False
            native.write(root / "OPTIMIZER_USAGE.json", gateway.summary())
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        for descriptor in descriptors:
            os.close(descriptor)
        if logfile is not None:
            logfile.close()
        native.write(root / "RETIREMENT.json", {"at_utc": native.now(),
                                               "audit": native.resource_audit()})
        if gateway is not None:
            if (root / "FAILURE.json").exists():
                time.sleep(300)
            stop.set()
            gateway.shutdown()
            gateway.server_close()


def imports(config_path: Path) -> None:
    native.remote_only()
    config = native.read(config_path)
    from skillopt.engine.trainer import ReflACTTrainer
    from skillopt.envs.alfworld.adapter import ALFWorldAdapter
    from autosaddler.v2.config.registry import build_runtime
    from phiagent_autosaddler.alfworld_scenario import build_alfworld_components
    from phiagent_autosaddler.openai_function_provider import build_openai_function_provider
    native.require(all((ReflACTTrainer, ALFWorldAdapter, build_runtime,
                        build_alfworld_components, build_openai_function_provider)), "Missing engines")
    print(json.dumps({"status": "IMPORT_PASS", "python": sys.version, "method": config["method"],
                      "model_calls": 0, "benchmark_episodes": 0}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "evaluate", "imports"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "external/SkillOpt"), str(ROOT / "external/AutoSaddler/src"),
                    str(ROOT / "integrations/autosaddler_phiagent/src")]
    if args.command == "evaluate":
        native.require(args.request is not None and args.output is not None, "Missing worker paths")
        evaluate_request(args.request, args.output)
    else:
        native.require(args.config is not None, "--config is required")
        {"prepare": prepare, "run": run, "imports": imports}[args.command](args.config)


if __name__ == "__main__":
    main()
