"""Bounded deterministic continuations using only public GAIA2 agent feedback."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


PLAN_INSTRUCTION = (
    "\nWhen a next tool action is already determined, you may add a followups array "
    "to the same Action JSON, after action and action_input. Each entry has exactly "
    "action, action_input, and observation_contains (one to three nonempty literal "
    "strings that must appear in the PRECEDING action's public observation). "
    "Use at most three entries, with literal arguments known now. Use [] when the "
    "next action needs interpretation, new identifiers, or a user decision. Never "
    "plan a final answer or a message to the user. The controller may defer any "
    "followup back to you. All native tool rules and step limits still apply."
)


def action_object(text: str) -> dict[str, Any] | None:
    if text.count("Action:") != 1:
        return None
    body = text.split("Action:", 1)[1].strip()
    if body.startswith("```json"):
        body = body[7:].lstrip()
    elif body.startswith("```"):
        body = body[3:].lstrip()
    try:
        value, position = json.JSONDecoder().raw_decode(body)
    except (ValueError, TypeError):
        return None
    if body[position:].strip() not in ("", "```", "<end_action>", "```<end_action>"):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        return None
    if not isinstance(value.get("action_input"), dict):
        return None
    return value


def fingerprint(action: dict, observation: str) -> str:
    encoded = json.dumps([action, observation], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class PublicContinuation:
    require_stall: bool = True
    maximum_fast_steps: int = 3
    pending: list[dict] = field(default_factory=list)
    previous_action: dict | None = None
    previous_feedback: str | None = None
    stalled: bool = False
    fast_steps: int = 0
    traces: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if type(self.require_stall) is not bool or type(self.maximum_fast_steps) is not int:
            raise ValueError("Continuation controls must be typed")
        if not 1 <= self.maximum_fast_steps <= 3:
            raise ValueError("At most three deterministic steps may precede a CoT refresh")

    def remember(self, response: str, legal_tools: set[str]) -> str:
        self.pending = []
        self.fast_steps = 0
        action = action_object(response)
        self.previous_action = None
        if action is None:
            return response
        self.previous_action = {key: action[key] for key in ("action", "action_input")}
        planned = action.get("followups", [])
        valid = isinstance(planned, list) and len(planned) <= self.maximum_fast_steps
        for step in planned if valid else []:
            if (not isinstance(step, dict)
                    or set(step) != {"action", "action_input", "observation_contains"}
                    or not isinstance(step["action"], str)
                    or step["action"] not in legal_tools
                    or step["action"] == "final_answer"
                    or step["action"].startswith("AgentUserInterface__")
                    or not isinstance(step["action_input"], dict)
                    or not isinstance(step["observation_contains"], list)
                    or not 1 <= len(step["observation_contains"]) <= 3
                    or any(not isinstance(value, str) or not value.strip() or len(value) > 256
                           for value in step["observation_contains"])):
                valid = False
                break
        if valid:
            self.pending = deepcopy(planned)
        self.traces.append({"event": "plan", "accepted": valid,
                            "declared": len(planned) if isinstance(planned, list) else None,
                            "queued": len(self.pending)})
        if "followups" not in action:
            return response
        return response.split("Action:", 1)[0] + "Action:\n" + json.dumps(self.previous_action)

    def decide(self, *, observation: str | None, error: bool, intervened: bool,
               legal_tools: set[str]) -> tuple[str | None, dict]:
        previous = self.previous_action
        current = fingerprint(previous, observation) if previous is not None and isinstance(observation, str) else None
        if error or current is not None and current == self.previous_feedback:
            self.stalled = True
        self.previous_feedback = current
        checks = {
            "public_observation": isinstance(observation, str) and bool(observation.strip()),
            "no_public_error": not error,
            "no_intervening_event": not intervened,
            "stall_gate": self.stalled or not self.require_stall,
            "refresh_not_due": self.fast_steps < self.maximum_fast_steps,
            "plan_available": bool(self.pending),
        }
        step = self.pending[0] if self.pending else None
        checks["native_tool_available"] = step is not None and step["action"] in legal_tools
        checks["public_predicates"] = step is not None and isinstance(observation, str) and all(
            value in observation for value in step["observation_contains"])
        execute = all(checks.values())
        record = {"event": "decision", "execute": execute, "checks": checks,
                  "feedback_sha256": current, "selection_model_calls": 0}
        self.traces.append(record)
        if not execute:
            self.pending = []
            return None, record
        self.pending.pop(0)
        self.fast_steps += 1
        self.previous_action = {key: deepcopy(step[key]) for key in ("action", "action_input")}
        return "Thought: Execute the publicly supported planned action.\nAction:\n" + json.dumps(
            self.previous_action), record


class Gaia2Engine:
    def __init__(self, agent, provider, *, mode: str, skill: str, record, recovery_policy="declared_followups"):
        if mode not in ("native", "cot", "jev", "no_stall", "stall", "latest_only"):
            raise ValueError("Unknown frozen GAIA2 comparison arm")
        self.agent, self.provider, self.mode, self.skill, self.record = agent, provider, mode, skill, record
        if recovery_policy not in ("declared_followups", "native_public_progress_no_refresh", "native_enumeration_state_v2"):
            raise ValueError("Unknown GAIA2 recovery policy")
        self.recovery_policy = recovery_policy
        self.controller = self.new_controller()
        self.cursor = 0

    def new_controller(self):
        if self.recovery_policy == "native_enumeration_state_v2":
            from phiagent.harness.gaia2_state_progress import StatePublicProgress
            return StatePublicProgress(require_stall=self.mode == "stall", persistent_memory=self.mode != "latest_only",
                                       include_contacts=self.mode != "no_stall")
        if self.recovery_policy == "native_public_progress_no_refresh":
            from phiagent.harness.gaia2_public_progress import NativePublicProgress
            return NativePublicProgress(require_stall=self.mode != "no_stall")
        return PublicContinuation(require_stall=self.mode != "no_stall")

    def __call__(self, prompt, **kwargs):
        fresh = self.agent.logs[self.cursor:]
        types = [row.get_type() for row in fresh]
        if "task" in types or self.cursor > len(self.agent.logs):
            self.controller = self.new_controller()
        if self.recovery_policy in ("native_public_progress_no_refresh", "native_enumeration_state_v2"):
            instructions = [row.content for row in self.agent.logs
                if row.get_type() in ("task", "agent_user_interface") and isinstance(getattr(row, "content", None), str)]
            self.controller.task = instructions[-1] if instructions else ""
        observations = [row for row in fresh if row.get_type() == "observation"]
        observation = observations[0].content if len(observations) == 1 and not getattr(
            observations[0], "attachments", []) else None
        error = "error" in types or isinstance(observation, str) and any(
            marker in observation.lower() for marker in ("error:", "traceback (", '"error":'))
        intervened = any(kind in types for kind in (
            "task", "agent_user_interface", "environment_notifications", "hint", "task_reminder"))
        action, decision = self.controller.decide(
            observation=observation, error=error, intervened=intervened,
            legal_tools=set(self.agent.action_executor.tools))
        self.cursor = len(self.agent.logs)
        if self.mode in ("jev", "no_stall", "stall", "latest_only") and action is not None:
            self.record({"source": "deterministic_controller", "action": action, "decision": decision,
                         "agent_id": self.agent.agent_id, "usage": {"prompt_tokens": 0,
                         "completion_tokens": 0, "total_tokens": 0}, "provider_response": False})
            return action, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                            "completion_duration": 0, "decision_source": "deterministic_controller"}
        messages = deepcopy(prompt)
        plan_enabled = self.recovery_policy == "declared_followups" and (
            self.mode == "no_stall" or self.mode == "jev" and self.controller.stalled)
        addition = (self.skill if self.mode != "native" else "") + (PLAN_INSTRUCTION if plan_enabled else "")
        if addition:
            if not messages or messages[0]["role"] != "system" or not isinstance(messages[0]["content"], str):
                raise ValueError("Native GAIA2 system prompt is required")
            messages[0]["content"] += "\n" + addition
        response = self.provider(messages, **kwargs)
        text, metadata = response if isinstance(response, tuple) and len(response) == 2 else (response, {})
        executed = self.controller.remember(text, set(self.agent.action_executor.tools))
        if not plan_enabled:
            self.controller.pending = []
            executed = text
        self.record({"source": "model", "agent_id": self.agent.agent_id,
                     "original_response": text, "executed_response": executed,
                     "plan_enabled": plan_enabled, "queued": len(self.controller.pending),
                     "decision": decision, "metadata": metadata})
        return executed, metadata
