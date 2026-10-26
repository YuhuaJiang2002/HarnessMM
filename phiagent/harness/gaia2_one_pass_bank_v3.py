"""Native-parent CoT with in-band, public-receipt conditional candidate banks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import threading
import time

from phiagent.harness.gaia2_cot_jev import (
    LABELS,
    decision_messages,
    digest,
    native_fast_execution_receipt,
    parse_bank,
    public_receipt_history,
    public_state_candidates,
)
from phiagent.harness.gaia2_dual_rsi import combine_usage
from phiagent.harness.gaia2_jev import Gaia2Engine, action_object
from phiagent.learning.alfworld_jev import LocalGate, select_local_decision
from phiagent.harness.gaia2_declared_bank_contract_v4 import (
    BANK_OBLIGATION, CONTRACT as DECLARED_CONTRACT, declared_bank_prompt,
)


CONTRACT = "native_parent_in_band_conditional_bank_v3"
INSTRUCTION = """
Keep the native Thought/Action format, full reasoning, and all task/tool rules.
Choose the next native action first. In that SAME Action JSON you may additionally
author "jev_bank":{"note":"remaining plan, evidence and unresolved constraints",
"candidates":[{"when":"public condition warranting this later action",
"action":"exact native tool name","action_input":{}}]}.
This is an optional continuation annotation, not another action or model call.
Use an empty bank when discovery, timing, a user decision or interpretation is
still needed. Do not sacrifice the current action or reasoning to fill a bank.
Author only useful later operations in your actual plan, not vague alternatives
such as "if needed". Preserve every task filter, exception and ordering constraint.
Keep at most eight candidates; never include completion, waits or user messages.
Use actual native parameter names and types. Omit unneeded optional arguments.
Known arguments may be literals. Unknown IDs must use explicit public bindings:
{"$obs":["field",0,"id"]} reads the latest observation;
{"$prev":["offset"]} reads the last issued action's arguments;
{"$add":[term,term]} is a declared numeric increment;
{"$receipt":{"tool":"exact native tool","path":["field",0,"id"],
"arguments":{}}} reads that tool's latest matching original public receipt
(omit arguments to match any call of that tool).
For a public list, an optional "foreach":{"receipt":{"tool":"exact native tool"},
"path":[],"where":{"field":"literal value"}} binds {"$item":["actual_id_field"]}.
Without receipt, foreach uses the latest observation. Use the observed field
names; never guess id/user_id or treat a plan as evidence. Expansion above eight,
missing/ambiguous bindings, an opaque latest observation or an intervening event
requires full native reasoning. Historical receipts do not prove current effects.
Each subsequent operation needs its own same-backbone one-token selection or
abstention. No bank executes an automatic sequence. Keep the native end marker.
""".strip()


def validate_settings(settings):
    expected = {
        "implementation": "same_backbone_one_token_typed_decision",
        "enabled": True,
        "explicit_abstention": "?",
        "maximum_candidates": 8,
        "maximum_decision_input_tokens": 8192,
        "maximum_decision_output_tokens": 1,
        "minimum_probability": 0.0,
        "minimum_margin": 0.0,
        "periodic_cot_refresh": False,
        "routine_writes_enabled": True,
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("The native-parent bank, typed capacities or abstention gate changed")
    if settings.get("mechanism_revision") not in (CONTRACT, DECLARED_CONTRACT):
        raise ValueError("Require a registered source-frozen continuation-bank contract")
    if settings.get("maximum_fast_steps") != 6:
        raise ValueError("Keep the inherited discriminator contract")
    return settings


def split_native_bank(text, tools):
    """Remove only the annotation; retain native action values and reasoning."""
    action = action_object(text)
    if action is None or "jev_bank" not in action:
        return text, action, None, None
    native = {key: deepcopy(value) for key, value in action.items() if key != "jev_bank"}
    body = text.split("Action:", 1)[1].strip()
    prefix = ""
    if body.startswith("```json"):
        prefix, body = "```json\n", body[7:].lstrip()
    elif body.startswith("```"):
        prefix, body = "```\n", body[3:].lstrip()
    _, end = json.JSONDecoder().raw_decode(body)
    executed = (
        text.split("Action:", 1)[0] + "Action:\n" + prefix
        + json.dumps(native, ensure_ascii=False) + body[end:]
    )
    try:
        bank = parse_bank(action, tools, allow_routine_writes=True, continuation_bank=True)
        return executed, native, bank, None
    except (KeyError, TypeError, ValueError) as error:
        return executed, native, None, str(error)


class NativeBankProvider:
    """A provider decorator underneath the unchanged capable-parent actor."""

    def __init__(self, agent, provider, config, role, record, label_tokens, transport):
        self.agent, self.provider, self.config = agent, provider, config
        self.role, self.record = role, record
        self.label_tokens, self.transport = label_tokens, transport
        self.settings = validate_settings(config["cot_jev"])
        self.cursor = self.fast_steps = 0
        self.bank = None
        self.previous = {}
        self.seen = set()
        self.issued = []
        self.pending_fast = None
        self.pending_slow = None

    def __call__(self, parent_prompt, **kwargs):
        from scripts.run_gaia2_jev import native_response_text, native_usage

        self.pending_fast = None
        self.pending_slow = None
        fresh = self.agent.logs[self.cursor:]
        kinds = [row.get_type() for row in fresh]
        if "task" in kinds or self.cursor > len(self.agent.logs):
            self.bank, self.previous, self.fast_steps, self.seen = None, {}, 0, set()
            self.issued = []
        self.cursor = len(self.agent.logs)
        task = "\n".join(
            row.content for row in self.agent.logs
            if row.get_type() in ("task", "agent_user_interface")
            and isinstance(getattr(row, "content", None), str)
        )
        observations = [
            row.content for row in fresh if row.get_type() == "observation"
            and isinstance(getattr(row, "content", None), str)
            and not getattr(row, "attachments", ())
        ]
        observation = observations[0] if len(observations) == 1 else ""
        interrupted = any(kind in kinds for kind in (
            "task", "agent_user_interface", "environment_notifications",
            "hint", "task_reminder", "error",
        )) or bool(re.search(r'(?i)(?:error:|traceback \(|"error"\s*:)', observation))
        pending_usage = None
        tools = self.agent.action_executor.tools
        if self.bank and observation and not interrupted:
            history, unknown_history = public_receipt_history(self.agent.logs)
            history.extend({
                **row, "historical_receipt_not_current_state": True,
            } for row in unknown_history)
            candidates, failures, receipt = public_state_candidates(
                self.bank, observation, self.previous.get("action_input", {}),
                tools, self.seen, native_receipts=history,
            )
            self.record({
                "source": "cot_jev_public_observation_contract",
                "agent_id": self.agent.agent_id, **receipt,
                "previous_native_action": self.previous.get("action"),
                "native_observation_units": len(observations),
            })
            self.record({
                "source": "cot_jev_candidates", "candidates": candidates,
                "unresolved_bindings": failures, "bank_sha256": digest(self.bank),
            })
            if receipt["status"] == "UNKNOWN" or failures:
                candidates = []
                self.record({
                    "source": "cot_jev_binding_unavailable",
                    "agent_id": self.agent.agent_id, **receipt,
                    "unresolved_bindings": failures, "route": "full_cot",
                })
            if candidates:
                messages, choices = decision_messages(
                    task, observation, self.bank, candidates, tools,
                    self.previous, self.issued,
                    bank_binding_receipts=receipt.get("bank_binding_receipts"),
                )
                port = self.config["resource"]["model_port"]
                tokenized = self.transport(
                    f"http://127.0.0.1:{port}/tokenize",
                    {"model": self.config["model"]["served_name"], "messages": messages,
                     "chat_template_kwargs": {"enable_thinking": False}}, timeout=60,
                )
                if tokenized["count"] <= self.settings["maximum_decision_input_tokens"]:
                    started = time.monotonic()
                    raw = self.transport(
                        self.endpoint(),
                        {"model": self.config["model"]["served_name"], "messages": messages,
                         "chisel_jev_decision": True, "max_tokens": 1,
                         "logprobs": True, "top_logprobs": len(choices),
                         "allowed_token_ids": [self.label_tokens[label] for label in choices]},
                        timeout=7500,
                    )
                    usage = native_usage(raw, self.config["model"]["served_name"], 131072)
                    pending_usage = {**usage, "completion_duration": time.monotonic() - started}
                    decision = select_local_decision(
                        raw, choices, expected_model=self.config["model"]["served_name"],
                        gate=LocalGate(0.0, 0.0, self.settings["maximum_fast_steps"]),
                    )
                    self.record({
                        "source": "cot_jev_typed_decision", "agent_id": self.agent.agent_id,
                        "decision": decision, "choices": choices, "raw": raw,
                        "native_observation_sha256": receipt["native_observation_sha256"],
                        "native_observation_preserved": True,
                        "full_candidate_tool_contracts_preserved": True,
                        "one_control_token_no_generated_reasoning": True,
                    })
                    if decision["route"] == "local":
                        action = json.loads(decision["action"])
                        self.pending_fast = {
                            "action": deepcopy(action), "decision_sha256": digest(decision),
                            "decision_usage": usage,
                            "input_observation_sha256": receipt["native_observation_sha256"],
                        }
                        self.previous = action
                        self.seen.add(digest(action))
                        self.issued.append(deepcopy(action))
                        self.fast_steps += 1
                        return (
                            "Thought: Execute the CoT-authored candidate selected from public receipts.\n"
                            "Action:\n" + json.dumps(action, ensure_ascii=False),
                            pending_usage,
                        )
                else:
                    self.record({
                        "source": "cot_jev_input_overflow", "input_tokens": tokenized["count"],
                        "route": "full_cot", "candidates_truncated": False,
                    })
        messages = deepcopy(parent_prompt)
        if not messages or messages[0].get("role") != "system":
            raise ValueError("The complete capable-parent native prompt is required")
        parent_converted = [
            self.provider._convert_message_to_litellm_format(message) for message in messages
        ]
        messages[0]["content"] += "\n" + INSTRUCTION
        declared = self.settings["mechanism_revision"] == DECLARED_CONTRACT
        if declared:
            messages = declared_bank_prompt(messages)
        converted = [
            self.provider._convert_message_to_litellm_format(message) for message in messages
        ]
        started = time.monotonic()
        raw = self.transport(
            self.endpoint(),
            {"model": self.config["model"]["served_name"], "messages": converted,
             "stop": list(kwargs.get("stop_sequences", ()))}, timeout=1830,
        )
        original = raw["choices"][0]["message"]["content"]
        usage = native_usage(raw, self.config["model"]["served_name"], 131072)
        self.record({
            "source": "public_actor_messages", "messages": converted,
            "response": original, "usage": usage, "component": "native_in_band_cot",
        })
        text = native_response_text(original, kwargs.get("stop_sequences", ()))
        executed, action, self.bank, error = split_native_bank(text, tools)
        self.previous = (
            {key: deepcopy(action[key]) for key in ("action", "action_input")}
            if action is not None else {}
        )
        self.pending_slow = deepcopy(self.previous) if self.previous else None
        self.seen = {digest(self.previous)} if self.previous else set()
        self.issued = [deepcopy(self.previous)] if self.previous else []
        self.fast_steps = 0
        if error:
            self.record({"source": "cot_jev_invalid_bank", "reason": error})
        self.record({
            "source": "cot_jev_slow_reasoning", "agent_id": self.agent.agent_id,
            "bank": self.bank, "original_response": original, "executed_response": executed,
            "new_cot_calls": 1, "jev_inputs_also_charged": pending_usage is not None,
            "parent_messages_sha256": digest(parent_converted),
            "parent_system_prompt_sha256": hashlib.sha256(
                parent_converted[0]["content"].encode()
            ).hexdigest(),
            "policy_instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
            "mechanism_revision": self.settings["mechanism_revision"],
            "final_public_turn_bank_obligation": declared,
            "bank_obligation_sha256": (
                hashlib.sha256(BANK_OBLIGATION.encode()).hexdigest() if declared else None
            ),
            "native_action_prompt_parser_and_history_preserved": True,
            "structured_replacement_envelope": False,
            "standalone_bank_authoring_calls": 0,
        })
        usage = {**usage, "completion_duration": time.monotonic() - started}
        return executed, (
            combine_usage(pending_usage, usage) if pending_usage is not None else usage
        )

    def endpoint(self):
        return (
            f"http://127.0.0.1:{self.config['resource']['capture_port']}/"
            f"{self.role}/actor/v1/chat/completions"
        )


def install_adapter(config, directory, role, mode, skill):
    if (
        config["method"] != "chisel" or config["stage"] != "train-candidate"
        or config["evaluation_split"] != "train" or config.get("receipt_execution") is not False
        or config["native_parent"]["skill_sha256"] != hashlib.sha256(skill.encode()).hexdigest()
    ):
        raise ValueError("Use the exact capable native parent on canonical TRAIN only")
    validate_settings(config["cot_jev"])
    from are.simulation.agents.default_agent.base_agent import BaseAgent
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine
    from scripts import run_gaia2_jev as base

    original_step = BaseAgent.step
    lock = threading.Lock()

    def record(value):
        with lock, (directory / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(value, default=str) + "\n")

    def completion(engine, messages, stop_sequences=(), **kwargs):
        endpoint = engine.model_config.endpoint
        if not endpoint or not endpoint.startswith(
            f"http://127.0.0.1:{config['resource']['capture_port']}/{role}/"
        ) or "/judge/" not in endpoint:
            raise ValueError("The actor must use its decorated metered native provider")
        converted = [engine._convert_message_to_litellm_format(m) for m in messages]
        started = time.monotonic()
        raw = base.get_json(
            endpoint.rstrip("/") + "/chat/completions",
            {"model": config["model"]["served_name"], "messages": converted,
             "stop": list(stop_sequences)}, timeout=1830,
        )
        text = base.native_response_text(
            raw["choices"][0]["message"]["content"], stop_sequences, judge=True,
        )
        return text, {
            **base.native_usage(raw, config["model"]["served_name"], 131072),
            "completion_duration": time.monotonic() - started,
        }

    tokens = {}
    for label in LABELS + "?":
        encoded = base.get_json(
            f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",
            {"model": config["model"]["served_name"], "prompt": label,
             "add_special_tokens": False}, timeout=60,
        )
        if encoded.get("count") != 1 or len(encoded.get("tokens", ())) != 1:
            raise ValueError("Every typed label must be exactly one backbone token")
        tokens[label] = encoded["tokens"][0]
    if len(set(tokens.values())) != len(tokens):
        raise ValueError("Finite-choice labels must have distinct token identities")
    record({
        "source": "cot_jev_label_binding", "label_token_ids": tokens,
        "model": config["model"], "typesafe_api_used": False,
        "native_parent": config["native_parent"], "mechanism_revision": CONTRACT,
    })

    def step(agent):
        if not isinstance(agent.llm_engine, Gaia2Engine) or not isinstance(
            agent.llm_engine.provider, NativeBankProvider,
        ):
            provider = NativeBankProvider(
                agent, agent.llm_engine, config, role, record, tokens, base.get_json,
            )
            agent.llm_engine = Gaia2Engine(
                agent, provider, mode="cot", skill=skill, record=record,
                recovery_policy="native_enumeration_state_v2",
            )
        provider = agent.llm_engine.provider
        provider.pending_fast = None
        provider.pending_slow = None
        start = len(agent.logs)
        completed = False
        try:
            result = original_step(agent)
            completed = True
            return result
        finally:
            pending = provider.pending_fast
            action = pending["action"] if pending is not None else provider.pending_slow
            if action is not None:
                fresh = agent.logs[start:]
                calls = [
                    {"tool_name": row.tool_name, "tool_arguments": row.tool_arguments}
                    for row in fresh if row.get_type() == "tool_call"
                ]
                observations = [
                    row.content for row in fresh if row.get_type() == "observation"
                    and isinstance(getattr(row, "content", None), str)
                ]
                record({
                    "source": (
                        "cot_jev_native_fast_execution" if pending is not None
                        else "cot_jev_native_slow_execution"
                    ),
                    "agent_id": agent.agent_id,
                    **(pending if pending is not None else {"action": action}),
                    "native_tool_call_logs": calls, "native_observation_logs": observations,
                    **native_fast_execution_receipt(
                        action, calls, observations, step_completed=completed,
                    ),
                })

    LiteLLMEngine.chat_completion = completion
    BaseAgent.step = step
