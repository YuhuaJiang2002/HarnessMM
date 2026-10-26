#!/usr/bin/env python3
"""Remote-only GAIA2 native evaluation with public-feedback Jev-inspired controls."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import importlib.util
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
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.harness.gaia2_jev import Gaia2Engine, PublicContinuation, action_object
from scripts.serve_terminal_bench_model import acquire_gpu_leases, runtime_environment, stop_owned
from scripts.cumulative_token_policy import token_limit_allows

ARMS = ("native", "cot", "jev", "no_stall")
STATE_ARMS = ARMS + ("stall", "latest_only")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(target: Path, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(target)


def read(target: Path):
    return json.loads(target.read_text())


def digest(target: Path):
    value = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_remote(config):
    partition = config.get("dataset", {}).get("partition")
    population_valid = config.get("dataset", {}).get("tasks") == 160 if partition is None else (
        partition.get("count") == 4 and type(partition.get("index")) is int
        and 0 <= partition["index"] < 4 and config["dataset"]["tasks"] == 40)
    if (sys.platform != "linux" or socket.gethostname() != config["hostname"]
            or not Path(config["run_directory"]).is_relative_to("/opt/phiagent/work/gaia2-jev")
            or tuple(config["arms"]) != (STATE_ARMS if config["method_identity"].get("recovery_policy") == "native_enumeration_state_v2" else ARMS) or not population_valid
            or config["dataset"]["repetitions"] != 3):
        raise RuntimeError("Frozen GAIA2 comparison is restricted to its authorized remote host")


def native_usage(raw, model, context):
    usage = raw.get("usage", {})
    if (raw.get("model") != model
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_FIELDS)
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
            or not 0 < usage["total_tokens"] <= context):
        raise ValueError("Incomplete or inconsistent native provider usage")
    return {key: usage[key] for key in USAGE_FIELDS}


def native_response_text(text, stop_sequences, *, judge=False):
    if not isinstance(text, str):
        raise RuntimeError("Native model response lacks text")
    # Quoted tool arguments and case-sensitive judge verdicts are literal data.
    if not judge:
        text = re.sub(
            r'"(?:\\.|[^"\\])*"|\b(?:False|True)\b',
            lambda match: match[0] if match[0].startswith('"') else match[0].lower(),
            text,
            flags=re.DOTALL,
        )
    for stop in stop_sequences:
        text = text.split(stop)[0]
    return text


def get_json(url, payload=None, timeout=30):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        code = error.code
        error.close()
        raise RuntimeError(f"HTTP {code}: {detail[:1000]}") from None


class Ledger:
    def __init__(self, root, limits, context, deadline):
        self.root, self.limits, self.context, self.deadline = root, limits, context, deadline
        self.lock = threading.Lock()
        self.calls = {}
        self.known = self.reserved = self.unknown = 0

    def inherit(self, previous, prior_root):
        if self.calls or previous["unknown_usage_calls"] or previous["inflight_reserved_tokens"]:
            raise RuntimeError("Inherited study costs must be complete and loaded once")
        for row in previous["calls"]:
            if row["status"] != "SETTLED" or row["usage"] is None:
                raise RuntimeError("Prior attempt contains unresolved native costs")
            self.calls[row["sequence"]] = {**row, "origin": str(prior_root), "historical_infrastructure_attempt": True}
        self.known = sum(row["usage"]["total_tokens"] for row in self.calls.values())
        if self.known != previous["known_tokens"]:
            raise RuntimeError("Prior attempt token sum differs")

    def reserve(self, role):
        with self.lock:
            if (self.unknown or time.time() >= self.deadline
                    or (self.limits["maximum_requests"] is not None
                        and len(self.calls) >= self.limits["maximum_requests"])
                    or not token_limit_allows(self.limits, "maximum_tokens", self.known + self.reserved + self.context)):
                raise RuntimeError("Frozen all-attempt study budget exhausted or usage unresolved")
            sequence = len(self.calls) + 1
            self.calls[sequence] = {"sequence": sequence, "role": role, "status": "INFLIGHT", "at_utc": now()}
            self.reserved += self.context
            write(self.root / "PROVIDER_LEDGER.json", self.snapshot_unlocked(include_calls=False))
            return sequence

    def settle(self, sequence, usage, *, preinference_rejection=False):
        with self.lock:
            self.reserved -= self.context
            self.known += usage["total_tokens"] if usage is not None else 0
            self.unknown += int(usage is None and not preinference_rejection)
            self.calls[sequence].update(status="REJECTED_BEFORE_FORWARD" if preinference_rejection else
                                       "SETTLED" if usage is not None else "UNKNOWN", usage=usage, settled_at_utc=now())
            with (self.root / "PROVIDER_CALLS.jsonl").open("a") as output:
                output.write(json.dumps(self.calls[sequence]) + "\n")
            write(self.root / "PROVIDER_LEDGER.json", self.snapshot_unlocked(include_calls=False))

    def snapshot_unlocked(self, *, include_calls=True):
        result = {"attempts": len(self.calls), "known_tokens": self.known,
                "inflight_reserved_tokens": self.reserved, "unknown_usage_calls": self.unknown,
                "deadline_utc": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat()}
        if include_calls:
            result["calls"] = list(self.calls.values())
        return result

    def snapshot(self, *, include_calls=True):
        with self.lock:
            return self.snapshot_unlocked(include_calls=include_calls)

    def terminal_stop_reason(self):
        """Stop the controller only when settled usage makes progress impossible."""
        with self.lock:
            if self.unknown:
                return "unknown_provider_usage"
            if time.time() >= self.deadline:
                return "absolute_deadline"
            if any(row["status"] == "INFLIGHT" for row in self.calls.values()):
                return None
            if (self.limits["maximum_requests"] is not None
                    and len(self.calls) >= self.limits["maximum_requests"]):
                return "request_limit"
            if not token_limit_allows(self.limits, "maximum_tokens", self.known + self.context):
                return "insufficient_tokens_for_next_reservation"
            return None


class Gateway(ThreadingHTTPServer):
    def __init__(self, config, root, ledger):
        super().__init__(("127.0.0.1", config["resource"]["capture_port"]), GatewayHandler)
        self.config, self.root, self.ledger = config, root, ledger


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 5 or parts[2:] != ["v1", "chat", "completions"] or parts[0] not in (*STATE_ARMS, "admission") or parts[1] not in ("actor", "judge"):
            self.send_error(404)
            return
        role = "/".join(parts[:2])
        config, ledger = self.server.config, self.server.ledger
        model = config["model"]
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("model") != model["served_name"] or body.get("stream", False):
                raise ValueError("Frozen model identity or streaming policy differs")
            body.update(temperature=model["temperature"], seed=model["seed"], max_tokens=model["max_tokens"],
                        chat_template_kwargs={"enable_thinking": model["enable_thinking"]})
            sequence = ledger.reserve(role)
        except (ValueError, RuntimeError) as error:
            self.send_error(429, str(error))
            return
        call = self.server.root / "provider" / f"call-{sequence:06d}"
        call.mkdir(parents=True, exist_ok=False)
        write(call / "request.json", {"role": role, "json": body, "at_utc": now()})
        usage, reply, code, rejected_before_forward = None, None, 502, False
        try:
            tokenized = get_json(f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",
                {key: body[key] for key in ("model", "messages", "chat_template_kwargs")}, timeout=60)
            write(call / "tokenize.json", tokenized)
            count = tokenized.get("count")
            if type(count) is not int or count < 1:
                raise RuntimeError("Native input token count is unavailable")
            if count + body["max_tokens"] > model["max_model_len"]:
                rejected_before_forward = True
                write(call / "REJECTED_BEFORE_FORWARD.json", {"input_tokens": count,
                    "max_tokens": body["max_tokens"], "context": model["max_model_len"], "model_calls": 0})
                reply, code = json.dumps({"error": "Frozen native context limit exceeded before model inference"}).encode(), 413
            else:
                raw = get_json(f"http://127.0.0.1:{config['resource']['model_port']}/v1/chat/completions", body,
                               timeout=config["budget"]["request_timeout_seconds"])
                write(call / "response.json", raw)
                usage = native_usage(raw, model["served_name"], model["max_model_len"])
                if usage["prompt_tokens"] != count:
                    raise RuntimeError("Preflight token count differs from the native inference input")
                reply, code = json.dumps(raw).encode(), 200
        except Exception as error:
            write(call / "failure.json", {"at_utc": now(), "error": str(error), "traceback": traceback.format_exc()})
            reply = json.dumps({"error": "Provider request failed; raw failure preserved"}).encode()
        finally:
            ledger.settle(sequence, usage, preinference_rejection=rejected_before_forward)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        try:
            self.wfile.write(reply)
        except (BrokenPipeError, ConnectionResetError):
            write(call / "delivery_failure.json", {"at_utc": now(), "native_response_preserved": usage is not None})


def install_native_adapter(config, root, arm):
    from are.simulation.agents.default_agent.base_agent import BaseAgent
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine

    original_step = BaseAgent.step
    lock = threading.Lock()
    trace = root / "decisions.jsonl"

    def record(value):
        with lock, trace.open("a") as output:
            output.write(json.dumps({"at_utc": now(), **value}, default=str) + "\n")

    def completion(engine, messages, stop_sequences=(), **kwargs):
        converted = [engine._convert_message_to_litellm_format(message) for message in messages]
        if any(isinstance(message["content"], list) for message in converted):
            raise RuntimeError("Text model received image attachments; retain this unresolved outcome")
        endpoint = engine.model_config.endpoint
        if (config.get('public_context_policy') == 'task_anchor_v1' and arm in ('jev', 'no_stall', 'stall', 'latest_only')
                and endpoint and '/actor/' in endpoint):
            from phiagent.harness.gaia2_task_context import project
            converted, projection = project(converted, memory=arm != 'latest_only')
            record({'source': 'public_context_projection', 'projection': projection})
        if not endpoint or not endpoint.startswith(f"http://127.0.0.1:{config['resource']['capture_port']}/{arm}/"):
            raise RuntimeError("Every actor and judge call must use the frozen metered endpoint")
        started = time.monotonic()
        raw = get_json(endpoint.rstrip("/") + "/chat/completions",
                       {"model": config["model"]["served_name"], "messages": converted, "stop": list(stop_sequences)},
                       timeout=config["budget"]["request_timeout_seconds"] + 30)
        usage = native_usage(raw, config["model"]["served_name"], config["model"]["max_model_len"])
        text = native_response_text(raw["choices"][0]["message"]["content"], stop_sequences, judge="/judge/" in endpoint)
        return text, {**usage, "completion_duration": time.monotonic() - started}

    def step(agent):
        if not isinstance(agent.llm_engine, Gaia2Engine):
            agent.llm_engine = Gaia2Engine(agent, agent.llm_engine, mode=arm, skill=config["skill"], record=record,
                recovery_policy=config["method_identity"].get("recovery_policy", "declared_followups"))
        return original_step(agent)

    LiteLLMEngine.chat_completion = completion
    BaseAgent.step = step


def native_actor_route_check(config, root, arm):
    from unittest.mock import patch
    from are.simulation.agents.agent_config_builder import AgentConfigBuilder
    from are.simulation.agents.agent_log import TaskLog
    from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
    from are.simulation.agents.default_agent.agent_factory import are_simulation_react_json_agent
    from are.simulation.agents.default_agent.base_agent import BaseAgent
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine
    from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder

    target = root / "actor-route-check" / arm
    target.mkdir(parents=True, exist_ok=False)
    endpoint = f"http://127.0.0.1:{config['resource']['capture_port']}/{arm}/actor/v1"
    base_config = AgentConfigBuilder().build("default").get_base_agent_config()
    base_config.llm_engine_config = LLMEngineConfig(model_name=config["model"]["served_name"],
        provider=config["model"]["provider"], endpoint=endpoint)
    engine = LLMEngineBuilder().create_engine(base_config.llm_engine_config)
    if not isinstance(engine, LiteLLMEngine) or engine.model_config.endpoint != endpoint:
        raise RuntimeError("Default native actor does not use the metered local model route")
    agent = are_simulation_react_json_agent(engine, base_config)
    agent.tools = {"Synthetic__read": lambda name: "contents:" + name}
    agent.update_system_prompt_tools = lambda prompts, tools: prompts
    agent.initialize()
    agent.append_agent_log(TaskLog(content="Read synthetic report", timestamp=agent.make_timestamp(), agent_id=agent.agent_id))
    calls = []

    def synthetic_provider(url, payload=None, timeout=30):
        if url != endpoint + "/chat/completions":
            raise RuntimeError("Synthetic native actor escaped the declared endpoint")
        calls.append(payload)
        return {"model": config["model"]["served_name"], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                "choices": [{"message": {"content": 'Thought: Inspect.\nAction:\n{"action":"Synthetic__read","action_input":{"name":"report"}}'}}]}

    original_step, original_completion = BaseAgent.step, LiteLLMEngine.chat_completion
    try:
        install_native_adapter(config, target, arm)
        with patch(__name__ + ".get_json", synthetic_provider):
            agent.step()
        if len(calls) != 1 or agent.logs[-1].get_type() != "observation" or agent.logs[-1].content != "contents:report":
            raise RuntimeError("Native actor route did not produce exactly one synthetic tool result")
        decisions = [json.loads(line) for line in (target / "decisions.jsonl").read_text().splitlines()]
        actor_decisions = [row for row in decisions if row['source'] != 'public_context_projection']
        if len(actor_decisions) != 1 or actor_decisions[0]["source"] != "model":
            raise RuntimeError("Native actor decision wrapper was bypassed")
        write(target / "PASS.json", {"status": "PASS", "arm": arm, "engine": str(type(engine)),
              "endpoint": endpoint, "synthetic_transport_calls": 1, "actual_model_calls": 0, "benchmark_calls": 0})
    finally:
        BaseAgent.step, LiteLLMEngine.chat_completion = original_step, original_completion


def synthetic_activation(root):
    from are.simulation.agents.default_agent.tools.action_executor import AgentAction
    from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor

    controller = PublicContinuation(require_stall=False)
    reply = 'Thought: Inspect.\nAction:\n' + json.dumps({"action": "Synthetic__list", "action_input": {},
        "followups": [{"action": "Synthetic__read", "action_input": {"name": "report"}, "observation_contains": ["report"]}]})
    tools = {"Synthetic__list", "Synthetic__read"}
    controller.remember(reply, tools)
    selected, record = controller.decide(observation="report", error=False, intervened=False, legal_tools=tools)
    executor = JsonActionExecutor(tools={"Synthetic__read": lambda name: "contents:" + name}, use_custom_logger=False)
    parsed = executor.parse_action(AgentAction(rationale="public synthetic gate", action=json.dumps(action_object(selected))))
    logs = []
    executor.execute_parsed_action(parsed, logs.append, time.time, "synthetic")
    if logs[-1].content != "contents:report" or record["selection_model_calls"] != 0:
        raise RuntimeError("Native deterministic action activation failed")
    write(root / "NATIVE_SYNTHETIC_PASS.json", {"status": "PASS", "model_calls": 0, "benchmark_calls": 0,
          "native_executor": str(type(executor)), "public_observation": logs[-1].content, "decision": record})


def paid_activation(config, root):
    if config["method_identity"].get("recovery_policy") in ("native_public_progress_no_refresh", "native_enumeration_state_v2"):
        return paid_native_progress_activation(config, root)
    from are.simulation.agents.default_agent.tools.action_executor import AgentAction
    from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
    from phiagent.harness.gaia2_jev import PLAN_INSTRUCTION

    message = (
        "This is a synthetic tool-protocol check, not a benchmark task. "
        "Use Thought: followed by Action: and one JSON object. Available tools: "
        "Synthetic__list() returns the literal report.txt; Synthetic__read(name) reads that filename. "
        "The directory contains exactly report.txt. The previous public step failed, so recovery is active. "
        "First list, then use a declared followup to read report.txt when the preceding observation contains report.txt."
        + PLAN_INSTRUCTION
    )
    raw = get_json(f"http://127.0.0.1:{config['resource']['capture_port']}/admission/actor/v1/chat/completions",
        {"model": config["model"]["served_name"], "messages": [{"role": "user", "content": message}]}, timeout=630)
    response = raw["choices"][0]["message"]["content"]
    controller = PublicContinuation()
    controller.stalled = True
    tools = {"Synthetic__list": lambda: "report.txt", "Synthetic__read": lambda name: "contents:" + name}
    executed = controller.remember(response, set(tools))
    primary = action_object(executed)
    if primary != {"action": "Synthetic__list", "action_input": {}} or len(controller.pending) != 1:
        raise RuntimeError("Paid model did not produce the required bounded synthetic plan")
    executor = JsonActionExecutor(tools=tools, use_custom_logger=False)
    logs = []
    executor.execute_action(AgentAction(rationale="synthetic", action=json.dumps(primary)), logs.append, time.time, "synthetic")
    selected, decision = controller.decide(observation=logs[-1].content, error=False, intervened=False, legal_tools=set(tools))
    if selected is None or action_object(selected) != {"action": "Synthetic__read", "action_input": {"name": "report.txt"}}:
        raise RuntimeError("Paid plan does not pass the deterministic public-feedback gate")
    executor.execute_action(AgentAction(rationale="synthetic followup", action=json.dumps(action_object(selected))), logs.append, time.time, "synthetic")
    if logs[-1].content != "contents:report.txt":
        raise RuntimeError("Native synthetic followup did not execute")
    write(root / "PAID_ACTIVATION_PASS.json", {"at_utc": now(), "status": "PASS", "response": response,
          "decision": decision, "native_tool_executions": 2, "followup_selection_model_calls": 0,
          "usage": native_usage(raw, config["model"]["served_name"], config["model"]["max_model_len"]), "benchmark_calls": 0})


def paid_native_progress_activation(config, root):
    from phiagent.harness.gaia2_public_progress import NativePublicProgress
    from are.simulation.agents.default_agent.tools.action_executor import AgentAction
    from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
    prompt = ('Synthetic admission: enumerate all emails by first listing offset 0, limit 10. '
              'Use the native ReAct JSON serialization shown below, including the action and action_input keys. '
              'The action name belongs INSIDE the JSON object. Do not use an Action Input: label. '
              'Do not add a followups field. Required format:\n'
              'Thought: I will inspect the first public page.\nAction:\n'
              '{"action":"Mail__list_emails","action_input":{"offset":0,"limit":10}}')
    raw = get_json(f"http://127.0.0.1:{config['resource']['capture_port']}/admission/actor/v1/chat/completions",
        {"model": config["model"]["served_name"], "messages": [{"role": "user", "content": prompt}]}, timeout=630)
    response = raw["choices"][0]["message"]["content"]
    primary = action_object(response)
    if primary != {"action": "Mail__list_emails", "action_input": {"offset": 0, "limit": 10}}:
        raise RuntimeError("Paid native progress admission did not produce the exact initial native action")
    if config["method_identity"].get("recovery_policy") == "native_enumeration_state_v2":
        from phiagent.harness.gaia2_state_progress import StatePublicProgress
        controller = StatePublicProgress(task="Enumerate all emails")
    else:
        controller = NativePublicProgress(task="Enumerate all emails", stalled=True)
    tools = {"Mail__list_emails": lambda offset=0, limit=10:
        f"ReturnedEmails(emails=[], emails_range=({offset}, {min(offset+limit, 50)}), total_returned_emails=11, total_emails=50)"}
    executor = JsonActionExecutor(tools=tools, use_custom_logger=False)
    controller.remember(response, set(tools))
    logs, decisions = [], []
    executor.execute_action(AgentAction(rationale="synthetic", action=json.dumps(primary)), logs.append, time.time, "synthetic")
    for expected_offset in (10, 20, 30, 40):
        action, decision = controller.decide(observation=logs[-1].content, error=False, intervened=False, legal_tools=set(tools))
        if action is None or action_object(action)["action_input"]["offset"] != expected_offset:
            raise RuntimeError("Native public progress did not continue beyond the removed three-step refresh")
        decisions.append(decision)
        executor.execute_action(AgentAction(rationale="synthetic native progress", action=json.dumps(action_object(action))),
                                logs.append, time.time, "synthetic")
    action, _ = controller.decide(observation=logs[-1].content, error=False, intervened=False, legal_tools=set(tools))
    if action is not None:
        raise RuntimeError("Native progress failed to abstain at the final public page")
    write(root / "PAID_ACTIVATION_PASS.json", {"at_utc": now(), "status": "PASS", "response": response,
          "decisions": decisions, "native_tool_executions": 5, "followup_selection_model_calls": 0,
          "periodic_refresh": False, "candidate_origin": "native_public_progress",
          "usage": native_usage(raw, config["model"]["served_name"], config["model"]["max_model_len"]), "benchmark_calls": 0})



def native_state_activation(root):
    from are.simulation.apps.contacts import ContactsApp, Contact
    from are.simulation.agents.default_agent.tools.action_executor import AgentAction
    from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
    from phiagent.harness.gaia2_state_progress import StatePublicProgress
    app = ContactsApp(contacts={str(i): Contact(first_name="Fixture", last_name=str(i), contact_id=str(i)) for i in range(25)})
    tools = {"Contacts__get_contacts": app.get_contacts}
    executor = JsonActionExecutor(tools=tools, use_custom_logger=False)
    controller = StatePublicProgress(task="List all my contacts")
    first = {"action": "Contacts__get_contacts", "action_input": {"offset": 0}}
    controller.remember("Thought: Read.\nAction:\n" + json.dumps(first), set(tools))
    logs, decisions = [], []
    executor.execute_action(AgentAction(rationale="synthetic native contact", action=json.dumps(first)), logs.append, time.time, "synthetic")
    for expected in (10, 20):
        selected, record = controller.decide(observation=logs[-1].content, error=False, intervened=False, legal_tools=set(tools))
        if selected is None or action_object(selected)["action_input"] != {"offset": expected}:
            raise RuntimeError("Expanded controller failed the actual native contact schema")
        decisions.append(record)
        executor.execute_action(AgentAction(rationale="synthetic native contact", action=json.dumps(action_object(selected))), logs.append, time.time, "synthetic")
    selected, record = controller.decide(observation=logs[-1].content, error=False, intervened=False, legal_tools=set(tools))
    if selected is not None:
        raise RuntimeError("Native contact terminal page failed to return to the actor")
    write(root / "NATIVE_STATE_ACTIVATION.json", {"status": "PASS", "decisions": decisions + [record],
          "native_observations": [row.content for row in logs if row.get_type() == "observation"], "model_calls": 0, "benchmark_calls": 0})

def prepare(config, root):
    require_remote(config)
    if (root / "PREPARED.json").exists():
        raise FileExistsError("Preparation is immutable")
    freeze = read(root / "SOURCE_FREEZE.json")
    for relative, expected in freeze["files_sha256"].items():
        if digest(root / "source" / relative) != expected:
            raise RuntimeError("Frozen experiment implementation changed")
    assets = Path(config["dataset"]["assets"])
    inventory = read(assets / "VERIFIED.json")
    for entry in inventory["files"]:
        if digest(assets / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Official data asset changed")
    import pyarrow.parquet as parquet

    rows = parquet.read_table(assets / "gaia2/mini/validation-00000-of-00001.parquet").to_pylist()
    if len(rows) != 160 or len({row["scenario_id"] for row in rows}) != 160:
        raise RuntimeError("Official mini inventory differs from160 unique scenarios")
    partition = config["dataset"].get("partition")
    if partition is not None:
        if type(partition["count"]) is not int or type(partition["index"]) is not int or not 0 <= partition["index"] < partition["count"]:
            raise ValueError("Invalid outcome-independent official mini partition")
        rows = [row for index, row in enumerate(sorted(rows, key=lambda item: item["scenario_id"]))
                if index % partition["count"] == partition["index"]]
    if len(rows) != config["dataset"]["tasks"]:
        raise RuntimeError("Declared full partition inventory differs")
    target = root / "data"
    target.mkdir()
    identities = []
    for row in rows:
        identifier = row["scenario_id"]
        if Path(identifier).name != identifier or identifier in (".", ".."):
            raise RuntimeError("Unsafe scenario identifier")
        filename = target / f"{identifier}.json"
        filename.write_text(row["data"])
        identities.append({"scenario_id": identifier, "path": str(filename), "sha256": digest(filename)})
    synthetic_activation(root)
    if config["method_identity"].get("recovery_policy") == "native_enumeration_state_v2":
        native_state_activation(root)
    for arm in config["arms"]:
        native_actor_route_check(config, root, arm)
    native = Path(importlib.util.find_spec("are.simulation").submodule_search_locations[0])
    source_hashes = {str(filename.relative_to(native)): digest(filename) for filename in native.rglob("*.py")}
    write(root / "PREPARED.json", {"at_utc": now(), "status": "PREPARED_FULL_MINI160",
          "assets_manifest_sha256": digest(assets / "VERIFIED.json"), "scenarios": identities,
          "native_source_sha256": source_hashes, "native_package": importlib.metadata.version("meta-agents-research-environments"),
          "seed": config["model"]["seed"], "model_calls": 0, "benchmark_calls": 0})


def evaluate_arm(config, root, arm):
    require_remote(config)
    from are.simulation.benchmark.scenario_executor import run_dataset

    arm_root = root / "arms" / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    install_native_adapter(config, arm_root, arm)
    endpoint = f"http://127.0.0.1:{config['resource']['capture_port']}/{arm}"
    write(arm_root / "STARTED.json", {"at_utc": now(), "arm": arm, "label": config["labels"][arm],
          "argv": sys.argv, "hostname": socket.gethostname()})
    try:
        result = run_dataset(model=config["model"]["served_name"], model_provider=config["model"]["provider"],
            dataset_path=str(root / "data"), agent="default", endpoint=endpoint + "/actor/v1",
            output_dir=str(arm_root / "traces"), max_concurrent_scenarios=config["resource"]["native_workers_per_arm"],
            executor_type="thread", scenario_timeout=config["resource"]["scenario_timeout_seconds"],
            num_runs=config["dataset"]["repetitions"], simulated_generation_time_mode=config["resource"]["simulated_generation_time_mode"],
            judge_model=config["model"]["served_name"], judge_provider=config["model"]["provider"], judge_endpoint=endpoint + "/judge/v1")
        rows = result.to_polars().to_dicts()
        write(arm_root / "RESULT.json", {"at_utc": now(), "arm": arm, "rows": rows,
              "official_success_rate_percent": result.success_rate(), "native_run_config": result.run_config.model_dump(),
              "successful": result.successful_count, "failed": result.failed_count,
              "exceptions": result.exception_count, "no_validation": result.no_validation_count})
        if len(rows) != config["dataset"]["tasks"] * config["dataset"]["repetitions"]:
            raise RuntimeError("Native runner did not return every planned160x3 scenario")
    except Exception as error:
        write(arm_root / "FAILURE.json", {"at_utc": now(), "error": str(error), "traceback": traceback.format_exc()})
        raise


def gpu_snapshot():
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    return {"at_utc": now(), "devices": devices, "compute_apps": processes}


def select_gpu(config, root, stack, stopped):
    deadline = time.time() + config["budget"]["maximum_queue_seconds"]
    if config["budget"].get("absolute_deadline_utc"):
        deadline = min(deadline, datetime.fromisoformat(config["budget"]["absolute_deadline_utc"]).timestamp())
    resource = config["resource"]
    while time.time() < deadline and not stopped.is_set():
        snapshot = gpu_snapshot()
        devices = {int(row.split(",")[0]): [part.strip() for part in row.split(",")] for row in snapshot["devices"].splitlines()}
        for index in resource["preferred_gpu_indices"]:
            row = devices[index]
            if int(row[3]) >= 128 or row[1] in snapshot["compute_apps"]:
                continue
            try:
                lease = acquire_gpu_leases({"leases": Path(resource["lease_root"])}, index, row[1])
            except BlockingIOError:
                continue
            with lease:
                current = gpu_snapshot()
                matched = next(line.split(",") for line in current["devices"].splitlines() if int(line.split(",")[0]) == index)
                if int(matched[3]) >= 128 or row[1] in current["compute_apps"]:
                    continue
                stack.enter_context(lease.pop_all())
                selection = {**current, "index": index, "uuid": row[1], "CUDA_VISIBLE_DEVICES": row[1], "hostname": socket.gethostname()}
                write(root / "GPU_SELECTION.json", selection)
                return selection
        write(root / "WAITING_FOR_GPU.json", {"status": "WAITING_FOR_SHARED_GPU", **snapshot,
              "queue_deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat()})
        stopped.wait(min(60, max(0, deadline - time.time())))
    raise TimeoutError("No available authorized H200 before the frozen queue deadline")


def run(config, config_path, root):
    require_remote(config)
    if any((root / name).exists() for name in ("STARTED.json", "RESULT.json", "FAILURE.json")):
        raise FileExistsError("Every experiment attempt requires a new directory")
    prepared = read(root / "PREPARED.json")
    if prepared["native_package"] != config["native_package"].split("==")[1]:
        raise RuntimeError("Official package version differs")
    for relative, expected in read(root / "SOURCE_FREEZE.json")["files_sha256"].items():
        if digest(root / "source" / relative) != expected:
            raise RuntimeError("Frozen experiment implementation changed")
    for row in prepared["scenarios"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise RuntimeError("Frozen official scenario changed")
    stopped = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stopped.set())
    children, model_process, ledger, gateway = [], None, None, None
    with ExitStack() as stack:
        try:
            write(root / "STARTED.json", {"at_utc": now(), "pid": os.getpid(), "hostname": socket.gethostname(),
                  "config": config, "config_sha256": digest(config_path), "command": sys.argv,
                  "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}})
            assets = Path(config["model"]["assets"])
            manifest = assets / config["model"]["manifest"]
            if digest(manifest) != config["model"]["manifest_sha256"]:
                raise RuntimeError("Frozen model manifest changed")
            for entry in read(manifest)["files"]:
                if digest(assets / "model" / entry["path"]) != entry["sha256"]:
                    raise RuntimeError("Exact model files differ")
            selection = select_gpu(config, root, stack, stopped)
            deadline = time.time() + config["budget"]["maximum_wall_seconds"]
            if config["budget"].get("absolute_deadline_utc"):
                deadline = min(deadline, datetime.fromisoformat(config["budget"]["absolute_deadline_utc"]).timestamp())
            ledger = Ledger(root, config["budget"], config["model"]["max_model_len"], deadline)
            if config.get("prior_attempt"):
                prior = Path(config["prior_attempt"]["root"])
                for name, expected in config["prior_attempt"]["files_sha256"].items():
                    if digest(prior / name) != expected:
                        raise RuntimeError("Prior infrastructure failure evidence changed")
                closeout = read(prior / "INVALID_RUN_CLOSEOUT.json")
                if closeout["actor_benchmark_calls"] != 0 or closeout["known_tokens"] != config["prior_attempt"]["known_tokens"]:
                    raise RuntimeError("Infrastructure retry cannot discard actual actor executions")
                ledger.inherit(read(prior / "PROVIDER_FINAL.json"), prior)
                write(root / "PRIOR_USAGE.json", {"root": str(prior), "known_tokens": ledger.known,
                      "attempts": len(ledger.calls), "scores_pooled": False, "deadline_utc": ledger.snapshot()["deadline_utc"]})
            service = root / "service"
            service.mkdir()
            (service / "w").mkdir()
            python = assets / "runtime/bin/python"
            command = [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", str(assets / "model"),
                "--served-model-name", config["model"]["served_name"], "--host", "127.0.0.1", "--port", str(config["resource"]["model_port"]),
                "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", str(config["model"]["max_model_len"]),
                "--max-num-seqs", str(config["resource"]["max_num_seqs"]), "--gpu-memory-utilization", str(config["resource"]["gpu_memory_utilization"]),
                "--generation-config", "vllm", "--max-num-batched-tokens", "4096", "--gdn-prefill-backend", "triton",
                "--default-chat-template-kwargs", '{"enable_thinking":false}', "--attention-config", '{"backend":"FLASH_ATTN","flash_attn_version":3}',
                "--language-model-only", "--enforce-eager", "--seed", str(config["model"]["seed"])]
            write(service / "COMMAND.json", command)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", config["resource"]["model_port"]))
            log = stack.enter_context((service / "service.log").open("x"))
            environment = runtime_environment(service, python, gpu_uuid=selection["uuid"])
            model_process = subprocess.Popen(command, env=environment, cwd=service, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(service / "OWNER.json", {"pid": model_process.pid, "at_utc": now(), "command": command})
            startup = min(deadline, time.time() + config["budget"]["startup_timeout_seconds"])
            while True:
                if stopped.is_set() or model_process.poll() is not None or time.time() >= startup:
                    raise RuntimeError("Owned model startup did not complete within the frozen deadline")
                try:
                    models = get_json(f"http://127.0.0.1:{config['resource']['model_port']}/v1/models", timeout=5)
                    if config["model"]["served_name"] in {item["id"] for item in models["data"]}:
                        break
                except (OSError, RuntimeError, URLError):
                    pass
                stopped.wait(2)
            gateway = Gateway(config, root, ledger)
            threading.Thread(target=gateway.serve_forever, daemon=True).start()
            admitted = get_json(f"http://127.0.0.1:{config['resource']['capture_port']}/admission/actor/v1/chat/completions",
                {"model": config["model"]["served_name"], "messages": [{"role": "user", "content": "Return only the number 42."}]}, timeout=630)
            if "42" not in admitted["choices"][0]["message"]["content"]:
                raise RuntimeError("Paid native model admission failed")
            paid_activation(config, root)
            write(root / "ADMITTED.json", {"at_utc": now(), "usage": native_usage(admitted, config["model"]["served_name"], config["model"]["max_model_len"]),
                "gpu_selection_sha256": digest(root / "GPU_SELECTION.json"), "activation_sha256": digest(root / "PAID_ACTIVATION_PASS.json"),
                "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat()})
            for arm in config["arms"]:
                command = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(config_path), "--arm", arm]
                log = stack.enter_context((root / f"{arm}.log").open("x"))
                child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((arm, child))
            while any(child.poll() is None for _, child in children):
                write(root / "PROGRESS.json", {"at_utc": now(), "phase": "OFFICIAL_MINI160_EVALUATION",
                      "workers": {arm: {"pid": child.pid, "returncode": child.poll()} for arm, child in children},
                      "accounting": ledger.snapshot(include_calls=False), "gpu": gpu_snapshot()})
                terminal_reason = ledger.terminal_stop_reason()
                if terminal_reason:
                    write(root / "TERMINAL_BUDGET_STOP.json", {"at_utc": now(),
                          "reason": terminal_reason, "accounting": ledger.snapshot(include_calls=False),
                          "completed_artifacts_preserved": True})
                    raise RuntimeError(f"Frozen provider budget stopped evaluation: {terminal_reason}")
                if stopped.is_set() or time.time() >= deadline or ledger.unknown or any(child.poll() not in (None, 0) for _, child in children):
                    raise RuntimeError("Evaluation stopped by frozen deadline, failed worker or unresolved native usage")
                stopped.wait(30)
            if any(child.returncode != 0 for _, child in children):
                raise RuntimeError("A native evaluation worker failed")
        except Exception as error:
            write(root / "FAILURE.json", {"at_utc": now(), "error": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            for _, child in children:
                stop_owned(child)
            stop_owned(model_process)
            if gateway is not None:
                gateway.shutdown()
                gateway.server_close()
            if ledger is not None:
                write(root / "PROVIDER_FINAL.json", ledger.snapshot())
            write(root / "RETIREMENT.json", {"at_utc": now(), "owned_model_returncode": model_process.poll() if model_process else None,
                  "owned_children_returncodes": {arm: child.poll() for arm, child in children}, "gpu_state": gpu_snapshot()})
    audit(config, root)


def audit(config, root):
    require_remote(config)
    ledger = read(root / "PROVIDER_FINAL.json")
    if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
        raise RuntimeError("Incomplete native accounting cannot support exact costs")
    totals = {}
    for call in ledger["calls"]:
        directory = Path(call.get("origin", root)) / "provider" / f"call-{call['sequence']:06d}"
        if call["status"] == "REJECTED_BEFORE_FORWARD":
            rejection = read(directory / "REJECTED_BEFORE_FORWARD.json")
            if (rejection["model_calls"] != 0 or rejection["input_tokens"] + rejection["max_tokens"] <= rejection["context"]
                    or (directory / "response.json").exists()):
                raise RuntimeError("Pre-inference context rejection proof differs")
            continue
        raw = read(directory / "response.json")
        usage = native_usage(raw, config["model"]["served_name"], config["model"]["max_model_len"])
        if usage != call["usage"] or read(directory / "request.json")["role"] != call["role"]:
            raise RuntimeError("Raw provider accounting replay differs")
        if not call.get("historical_infrastructure_attempt"):
            totals[call["role"]] = totals.get(call["role"], 0) + usage["total_tokens"]
    expected = {(row["scenario_id"], repeat) for row in read(root / "PREPARED.json")["scenarios"]
                for repeat in range(1, config["dataset"]["repetitions"] + 1)}
    methods = {}
    for arm in config["arms"]:
        result = read(root / "arms" / arm / "RESULT.json")
        rows = result["rows"]
        if len(rows) != len(expected) or {(row["base_scenario_id"], row["run_number"]) for row in rows} != expected:
            raise RuntimeError("Native scenario identities or repetitions are incomplete")
        unresolved = sum(row["success_bool"] is None or row["has_exception"] for row in rows)
        methods[arm] = {"tasks": config["dataset"]["tasks"], "repetitions": config["dataset"]["repetitions"], "executions": len(rows),
            "official_success_rate_percent": result["official_success_rate_percent"],
            "unresolved": unresolved, "actor_tokens": totals.get(arm + "/actor", 0),
            "actor_tokens_per_execution": totals.get(arm + "/actor", 0) / len(expected),
            "judge_tokens": totals.get(arm + "/judge", 0), "result_sha256": digest(root / "arms" / arm / "RESULT.json")}
    write(root / "RESULT.json", {"at_utc": now(), "status": "COMPLETE_NATIVE_MINI_PARTITION_COMPARISON"
          if config["dataset"].get("partition") else "COMPLETE_NATIVE_MINI160_COMPARISON",
          "partition": config["dataset"].get("partition"),
          "methods": methods, "all_stage_tokens": ledger["known_tokens"], "native_usage_replay": "PASS",
          "paper_admitted": False, "full_gaia2_leaderboard_protocol": False, "unknown_usage_calls": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "audit", "native-check"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=STATE_ARMS)
    args = parser.parse_args()
    config = read(args.config)
    require_remote(config)
    root = Path(config["run_directory"])
    os.environ["DEMO_FS_PATH"] = str(Path(config["dataset"]["assets"]) / "gaia2_filesystem/demo_filesystem")
    os.environ["FS_PATH"] = str(root / "filesystem-cache")
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.action == "prepare":
        prepare(config, root)
    elif args.action == "native-check":
        synthetic_activation(root)
    elif args.action == "worker":
        evaluate_arm(config, root, args.arm)
    elif args.action == "run":
        run(config, args.config.resolve(), root)
    else:
        audit(config, root)


if __name__ == "__main__":
    main()
