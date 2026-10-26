"""Receipt-conditioned action graphs using only the native actor's public history.

One actor authors both the native action and optional bounded followups. The
executor never infers or rewrites entity names. Every subsequent prompt contains
the actions actually returned to the native executor and their public receipts.
The matched removal control keeps the instructions and audit records but hides
the dynamic progress state from the actor and disables compiled followups.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re
import threading
import time

from phiagent.harness.gaia2_jev import action_object
from phiagent.harness.gaia2_202060_ranges import coverage, range_receipt


INSTRUCTION = """Maintain one coherent public-state action graph for this request.
Use the action_receipts below as the record of actions actually issued. A receipt
is evidence, not automatic proof that an entire goal is satisfied. Never undo a
successful object merely because an earlier plan guessed another name. Keep the
user's exact names/wording; resolve missing IDs through native read tools.
For all/every/top-k requests, enumerate the complete candidate set, apply every
filter and exception, and record which members already satisfy the requested
effect. Count newly changed members, not total or previously satisfied members.
Do not replace a group operation by unrelated individual operations.
Keep timing/ordering and communication-channel constraints explicit. Re-evaluate
on incoming messages; no cached action may skip a requested wait or user decision.
For each next action, identify its still-open requested effect and public evidence.
Do not repeat an observed write as verification. Use a read when verification is
necessary. Before completion, reconcile every requested effect and set member.
After action and action_input you MAY add a progress_plan object containing:
{"request_quotes":["literal request clause"],
 "entity_sets":[{"request_quote":"literal request clause",
   "source_quote":"literal public observation excerpt","members":["literal observed ID"]}],
 "followups":[{"action":"native tool","action_input":{},
   "after_observation_contains":["literal expected receipt substring"],
   "argument_evidence":[{"source":"task or observation","quote":"literal supporting excerpt"}]}]}.
Use at most 3 followups. Include only fixed, already-known arguments and actions
that remain valid if the preceding receipt literally matches. Do not schedule
messages, destructive operations, completion, waits, or conditional actions here.
Use an empty followups array whenever discovery or interpretation is still needed.
The ledger will explicitly label declared sets as unproven exhaustive; do not
confuse a partial list, your own claims, or the plan itself with public completion."""


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def literal_leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from literal_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from literal_leaves(child)
    else:
        yield value


def safe_followup(step, *, legal, task, observations):
    if not isinstance(step, dict) or set(step) != {
        "action", "action_input", "after_observation_contains", "argument_evidence"
    }:
        return False
    name = step["action"]
    if not isinstance(name, str) or name not in legal or not isinstance(step["action_input"], dict):
        return False
    operation = name.split("__")[-1]
    if not operation.startswith(("get_", "list_", "search_", "find_", "read_", "save_", "add_")):
        return False
    if name.startswith("AgentUserInterface__") or any(
        word in operation for word in ("message", "email", "send", "delete", "remove", "buy", "order",
                                      "transfer", "wait", "sleep", "cancel", "payment")
    ):
        return False
    predicates = step["after_observation_contains"]
    if (not isinstance(predicates, list) or not 1 <= len(predicates) <= 3
            or any(not isinstance(text, str) or not 3 <= len(text.strip()) <= 256
                   for text in predicates)):
        return False
    evidence = step["argument_evidence"]
    if not isinstance(evidence, list) or not evidence:
        return False
    sources = {"task": task, "observation": observations}
    for entry in evidence:
        if (not isinstance(entry, dict) or entry.get("source") not in sources
                or not isinstance(entry.get("quote"), str) or not entry["quote"].strip()
                or entry["quote"] not in sources[entry["source"]]):
            return False
    public = task + "\n" + observations
    for value in literal_leaves(step["action_input"]):
        if isinstance(value, str) and value and value not in public:
            return False
    return True


@dataclass
class ReceiptGraph:
    execute_followups: bool = True
    task: str = ""
    observations: str = ""
    receipts: list = field(default_factory=list)
    entity_sets: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    previous: dict | None = None
    attempted: set = field(default_factory=set)
    pages: dict = field(default_factory=dict)
    public_next: dict | None = None

    def observe(self, observation, *, error=False, intervened=False):
        self.public_next = None
        if self.previous is not None:
            self.receipts.append({
                "receipt_id": len(self.receipts) + 1, "executed_action": deepcopy(self.previous),
                "status": "error" if error else "observed" if observation else "unconfirmed",
                "observation": observation, "observation_sha256": key(observation),
                "effect_completion": "Requires interpretation of this public receipt",
            })
            page = range_receipt(self.previous, observation, self.task) if not error and not intervened else None
            if page:
                current = self.pages.setdefault(page["scope"], [])
                if any(previous["total"] != page["total"] for previous in current):
                    current.clear()
                if not any(previous["start"] == page["start"] for previous in current):
                    current.append(page)
                self.public_next = page["next_action"]
            elif not self.previous["action"].split("__")[-1].startswith(
                ("get_", "list_", "search_", "read_", "find_", "lookup_")
            ):
                self.pages.clear()
            self.previous = None
        if observation:
            self.observations += "\n" + observation
        if error or intervened:
            self.pending.clear()
            self.pages.clear()
            self.public_next = None

    def remember(self, text, legal):
        action = action_object(text)
        self.pending.clear()
        if action is None:
            return text
        actual = {field: deepcopy(action[field]) for field in ("action", "action_input")}
        plan = action.get("progress_plan")
        accepted_sets = []
        if isinstance(plan, dict):
            quotes = plan.get("request_quotes")
            if isinstance(quotes, list) and quotes and all(
                isinstance(quote, str) and quote.strip() and quote in self.task for quote in quotes
            ):
                sets = plan.get("entity_sets", [])
                for entry in sets[:16] if isinstance(sets, list) else []:
                    if not isinstance(entry, dict):
                        continue
                    quote, source, members = (entry.get(field) for field in
                                               ("request_quote", "source_quote", "members"))
                    if (quote in quotes and isinstance(source, str) and source.strip()
                            and len(source) <= 4096
                            and source in self.observations and isinstance(members, list)
                            and 0 < len(members) <= 64 and all(
                                isinstance(member, str) and member and member in source
                                for member in members) and len(set(members)) == len(members)):
                        accepted_sets.append({**entry, "coverage_status": "Grounded members; "
                                              "exhaustiveness must still be established"})
                steps = plan.get("followups", [])
                if isinstance(steps, list) and len(steps) <= 3 and all(
                    safe_followup(step, legal=legal, task=self.task, observations=self.observations)
                    for step in steps
                ):
                    self.pending = deepcopy(steps)
        if accepted_sets:
            self.entity_sets = accepted_sets
        self.previous = actual
        self.attempted.add(key(actual))
        return text.split("Action:", 1)[0] + "Action:\n" + json.dumps(actual, ensure_ascii=False)

    def continuation(self, observation, *, error, intervened, legal):
        if (self.execute_followups and self.public_next is not None and not error
                and not intervened and self.public_next["action"] in legal
                and key(self.public_next) not in self.attempted):
            actual = deepcopy(self.public_next)
            self.public_next = None
            self.pending.clear()
            self.previous = actual
            self.attempted.add(key(actual))
            return ("Thought: Advance the required entity set using its public range receipt.\n"
                    "Action:\n" + json.dumps(actual, ensure_ascii=False)), {
                "public_range_compiler": True, "no_guessed_arguments": True,
                "next_offset_from_native_receipt": True}
        step = self.pending[0] if self.pending else None
        actual = {name: step[name] for name in ("action", "action_input")} if step else None
        checks = {
            "execution_enabled": self.execute_followups,
            "plan_available": step is not None, "no_error": not error,
            "no_intervention": not intervened, "public_receipt": bool(observation),
            "legal": step is not None and step["action"] in legal,
            "literal_receipt_matches": step is not None and bool(observation)
                and all(text in observation for text in step["after_observation_contains"]),
            "not_already_issued": actual is not None and key(actual) not in self.attempted,
        }
        if not all(checks.values()):
            self.pending.clear()
            return None, checks
        self.pending.pop(0)
        self.previous = deepcopy(actual)
        self.attempted.add(key(actual))
        return "Thought: Execute the actor's receipt-supported next graph edge.\nAction:\n" + json.dumps(
            actual, ensure_ascii=False), checks


class ReceiptEngine:
    def __init__(self, agent, provider, *, skill, record, execute_followups):
        self.agent, self.provider, self.skill, self.record = agent, provider, skill, record
        self.graph = ReceiptGraph(execute_followups=execute_followups)
        self.cursor = 0

    def __call__(self, prompt, **kwargs):
        started = time.monotonic()
        fresh = self.agent.logs[self.cursor:]
        tasks = [row.content for row in self.agent.logs if row.get_type() == "task"
                 and isinstance(getattr(row, "content", None), str)]
        task = "\n".join(tasks)
        if task != self.graph.task or self.cursor > len(self.agent.logs):
            self.graph = ReceiptGraph(execute_followups=self.graph.execute_followups, task=task)
        kinds = [row.get_type() for row in fresh]
        observations = [row.content for row in fresh if row.get_type() == "observation"
                        and isinstance(getattr(row, "content", None), str)]
        observation = "\n".join(observations)
        error = "error" in kinds or bool(re.search(
            r'(?i)(?:error:|traceback \(|"error"\s*:)', observation))
        intervened = any(kind in kinds for kind in ("task", "agent_user_interface",
                          "environment_notifications", "hint", "task_reminder"))
        self.graph.observe(observation, error=error, intervened=intervened)
        self.cursor = len(self.agent.logs)
        legal = set(self.agent.action_executor.tools)
        planned, checks = self.graph.continuation(
            observation, error=error, intervened=intervened, legal=legal)
        if planned is not None:
            self.record({"source": "receipt_graph_execution", "agent_id": self.agent.agent_id,
                         "executed_response": planned, "checks": checks,
                         "provider_response": False, "new_model_calls": 0})
            return planned, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                             "completion_duration": time.monotonic() - started}
        messages = deepcopy(prompt)
        if not messages or messages[0]["role"] != "system":
            raise RuntimeError("Native public system prompt is required")
        messages[0]["content"] += "\n" + self.skill + "\n" + INSTRUCTION
        active = self.graph.execute_followups
        compact_receipts = [
            {**{name: value for name, value in receipt.items() if name != "observation"},
             "observation_excerpt": receipt["observation"][:600],
             "excerpt_truncated": len(receipt["observation"]) > 600,
             "full_observation_in_native_history": True}
            for receipt in self.graph.receipts
        ]
        messages.append({"role": "user", "content": "[Public receipt-consistent action graph]\n" +
            json.dumps({"request": self.graph.task,
                        "action_receipts": compact_receipts if active else [],
                        "grounded_entity_sets": self.graph.entity_sets if active else [],
                        "enumeration_coverage": {scope: coverage(pages)
                            for scope, pages in self.graph.pages.items()} if active else {},
                        "controller_does_not_rewrite_native_arguments": True}, ensure_ascii=False)})
        response = self.provider(messages, **kwargs)
        text, metadata = response if isinstance(response, tuple) else (response, {})
        executed = self.graph.remember(text, legal)
        self.record({"source": "receipt_graph_model", "agent_id": self.agent.agent_id,
                     "original_response": text, "executed_response": executed,
                     "plan_edges_queued": len(self.graph.pending), "checks": checks,
                     "metadata": metadata})
        return executed, metadata


def install_metered_provider(config, directory, role):
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine
    from scripts import run_gaia2_jev as base
    lock = threading.Lock()

    def record(value):
        with lock, (directory / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(value, default=str) + "\n")

    def completion(engine, messages, stop_sequences=(), **kwargs):
        converted = [engine._convert_message_to_litellm_format(message) for message in messages]
        endpoint = engine.model_config.endpoint
        if not endpoint or not endpoint.startswith(
            f"http://127.0.0.1:{config['resource']['capture_port']}/{role}/"
        ):
            raise RuntimeError("Unmetered actor/judge route")
        started = time.monotonic()
        raw = base.get_json(endpoint.rstrip("/") + "/chat/completions",
                            {"model": config["model"]["served_name"], "messages": converted,
                             "stop": list(stop_sequences)}, timeout=1830)
        usage = base.native_usage(raw, config["model"]["served_name"], 131072)
        text = raw["choices"][0]["message"]["content"]
        if "/actor/" in endpoint:
            record({"source": "public_actor_messages", "messages": converted,
                    "response": text, "usage": usage})
        return base.native_response_text(text, stop_sequences, judge="/judge/" in endpoint), {
            **usage, "completion_duration": time.monotonic() - started}

    LiteLLMEngine.chat_completion = completion
    return record


def install_adapter(config, directory, role, mode, skill):
    """Replace only the CHISEL actor wrapper; leave native judge transport untouched."""
    from are.simulation.agents.default_agent.base_agent import BaseAgent

    old_step = BaseAgent.step
    record = install_metered_provider(config, directory, role)

    def step(agent):
        if not isinstance(agent.llm_engine, ReceiptEngine):
            agent.llm_engine = ReceiptEngine(
                agent, agent.llm_engine, skill=skill, record=record,
                execute_followups=config.get("receipt_execution", True))
        return old_step(agent)

    BaseAgent.step = step
