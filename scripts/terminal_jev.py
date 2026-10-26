"""Jev-inspired typed continuation gates over public terminal feedback."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping


SCHEMA = "terminal-jev-guarded-continuation/v1"
MODES = ("guarded", "success_only")
EXPLICIT_PLAN_INSTRUCTION = (
    "\nTyped execution plan protocol: every exec_in_sandbox call must include followups. "
    "Use an empty array to defer when the next action requires interpreting unseen output. "
    "When a short next command is already determined, place it in followups instead of spending "
    "another model turn or hiding independent checks in a long shell chain. Include at most two "
    "followups, each with command, stdout_contains (a list of literal required substrings), and "
    "stderr_empty (a boolean). Conditions refer to the immediately preceding command. "
    "For silent file creation followed by a check, set stdout_contains=[]; never put the "
    "check's future expected output in a condition on the preceding command. "
    "All commands share the existing tool timeout and start fresh shells. Do not invent an "
    "unnecessary action to fill a plan, and do not infer task completion from exit zero alone."
)


POSTCONDITION_PLAN_INSTRUCTION = (
    "\nTyped execution plan protocol: every exec_in_sandbox call includes followups, "
    "a list of at most two already-known next commands, or [] to defer to another model turn. "
    "A next command runs only after the preceding command exits successfully within the shared timeout. "
    "Each followup has command, stdout_contains, and stderr_empty. These are POSTCONDITIONS: "
    "after that followup runs, its own stdout must contain every listed literal, and its own stderr "
    "must be empty when requested. Any failure, truncation, or postcondition mismatch stops the plan "
    "and returns all outputs to you. Use this to combine a known edit with its public validation. "
    "Never invent extra actions to fill a plan or treat exit zero as task completion. Each command "
    "starts a fresh shell. The original task and CHISEL rules still apply."
)


class TerminalJevSession:
    def __init__(self, config: Mapping):
        if (not isinstance(config, Mapping) or not {"mode", "maximum_followups"} <= set(config)
                or set(config) - {"mode", "maximum_followups", "explicit_plan", "condition_target"}
                or config["mode"] not in MODES or type(config["maximum_followups"]) is not int
            or not 1 <= config["maximum_followups"] <= 2
            or type(config.get("explicit_plan", False)) is not bool
            or config.get("condition_target", "previous") not in ("previous", "self")):
            raise ValueError("jev requires a closed, bounded continuation configuration")
        self.mode = config["mode"]
        self.maximum_followups = config["maximum_followups"]
        self.explicit_plan = config.get("explicit_plan", False)
        self.condition_target = config.get("condition_target", "previous")
        self.plans_seen = 0
        self.empty_plans = 0
        self.traces = []

    def tools(self, tools):
        result = deepcopy(tools)
        functions = [tool["function"] for tool in result
                     if tool.get("function", {}).get("name") == "exec_in_sandbox"]
        if len(functions) != 1:
            raise ValueError("Jev continuation requires the original terminal tool")
        function = functions[0]
        if "validation_command" in function["parameters"]["properties"]:
            raise ValueError("Jev continuation must be ablated separately from Action Fusion")
        function["description"] += (
            " Followups contain already-known next commands, not commands that require "
            "interpreting unseen output. Each starts a fresh shell under the same shared timeout. "
            "Execution stops on failure, timeout or an unmet condition; every result returns to you. "
            "Exit zero is not evidence of task completion. Do not access hidden tests or grading files."
        )
        function["parameters"]["properties"]["followups"] = {
            "type": "array", "maxItems": self.maximum_followups,
            "items": {"type": "object", "additionalProperties": False,
                "properties": {
                    "command": {"type": "string", "minLength": 1, "maxLength": 16384},
                    "stdout_contains": {"type": "array", "maxItems": 3,
                                        "description": "Required in the PREVIOUS command's stdout BEFORE this command runs. Not this command's expected output. Use [] after a silent write.",
                                        "items": {"type": "string", "minLength": 1, "maxLength": 256}},
                    "stderr_empty": {"type": "boolean", "description": "Require the PREVIOUS command's stderr to be empty before this command runs."},
                }, "required": ["command", "stdout_contains", "stderr_empty"]},
        }
        if self.explicit_plan:
            function["parameters"]["required"] = ["command", "followups"]
            function["description"] += " Always include followups; [] explicitly defers to another model decision."
        else:
            function["description"] += " Followups are optional."
        if self.condition_target == "self":
            properties = function["parameters"]["properties"]["followups"]["items"]["properties"]
            properties["stdout_contains"]["description"] = "After THIS followup runs, its own stdout must contain these literal strings. Use [] when no stdout assertion is needed."
            properties["stderr_empty"]["description"] = "After THIS followup runs, require its own stderr to be empty."
            function["description"] += " Text and stderr conditions are postconditions on each followup's OWN result, checked after it runs."
        if self.mode == "success_only":
            function["description"] += (
                " In this control, followups depend only on successful exit, not text conditions."
            )
        return result

    def parse(self, tool_call):
        function = tool_call.get("function", {})
        if function.get("name") != "exec_in_sandbox":
            raise ValueError("Jev continuation applies only to terminal execution")
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("terminal arguments must be an object")
        if "validation_command" in arguments:
            raise ValueError("followups and validation_command must not overlap")
        if set(arguments) - {"command", "followups"}:
            raise ValueError("unknown typed terminal argument")
        if self.explicit_plan and "followups" not in arguments:
            raise ValueError("explicit plan requires followups; use [] to defer")
        followups = arguments.get("followups", [])
        if not isinstance(followups, list) or len(followups) > self.maximum_followups:
            raise ValueError("followups exceed the declared execution bound")
        for step in followups:
            if (not isinstance(step, dict) or set(step) != {"command", "stdout_contains", "stderr_empty"}
                    or not isinstance(step["command"], str) or not step["command"].strip()
                    or len(step["command"]) > 16384 or type(step["stderr_empty"]) is not bool
                    or not isinstance(step["stdout_contains"], list) or len(step["stdout_contains"]) > 3
                    or any(not isinstance(value, str) or not value or len(value) > 256
                           for value in step["stdout_contains"])):
                raise ValueError("invalid typed followup command or predicates")
        self.plans_seen += 1
        self.empty_plans += not bool(followups)
        return deepcopy(followups)

    def gate(self, step, previous, *, remaining_seconds):
        checks = {
            "exit_success": type(previous.get("exit_code")) is int and previous["exit_code"] == 0,
            "not_timed_out": previous.get("timed_out") is False,
            "deadline_remaining": remaining_seconds > 0,
            "public_stdout_available": isinstance(previous.get("stdout"), str),
            "public_stderr_available": isinstance(previous.get("stderr"), str),
        }
        if self.mode == "guarded" and self.condition_target == "previous":
            checks["stdout_predicates"] = checks["public_stdout_available"] and all(
                value in previous["stdout"] for value in step["stdout_contains"])
            checks["stderr_predicate"] = checks["public_stderr_available"] and (
                not step["stderr_empty"] or not previous["stderr"].strip())
        return {"execute": all(checks.values()), "checks": checks,
                "decision_source": "deterministic_public_feedback", "model_calls": 0}

    def postcondition(self, step, result):
        checks = {"exit_success": type(result.get("exit_code")) is int and result["exit_code"] == 0,
                  "not_timed_out": result.get("timed_out") is False,
                  "public_stdout_available": isinstance(result.get("stdout"), str),
                  "public_stderr_available": isinstance(result.get("stderr"), str)}
        if self.mode == "guarded":
            checks["stdout_predicates"] = checks["public_stdout_available"] and all(
                value in result["stdout"] for value in step["stdout_contains"])
            checks["stderr_predicate"] = checks["public_stderr_available"] and (
                not step["stderr_empty"] or not result["stderr"].strip())
        return {"execute": all(checks.values()), "checks": checks, "phase": "after_followup",
                "decision_source": "deterministic_public_feedback", "model_calls": 0}

    def record(self, *, tool_call_id, declared, executed, gates, stopped_reason):
        trace = {"tool_call_id": tool_call_id, "declared_followups": len(declared),
                 "followup_actions_without_extra_model_calls": len(executed),
                 "commands_sha256": [hashlib.sha256(step["command"].encode()).hexdigest() for step in declared],
                 "gates": deepcopy(gates), "stopped_reason": stopped_reason}
        self.traces.append(trace)
        return trace

    def report(self):
        return {"schema_version": SCHEMA, "mode": self.mode, "maximum_followups": self.maximum_followups,
            "explicit_plan": self.explicit_plan, "plans_seen": self.plans_seen, "empty_plans": self.empty_plans,
                "condition_target": self.condition_target,
                "identity": "Jev-inspired typed gates, not the Jev model or RLCD",
                "typesafe_api_used": False, "neural_gate_calls": 0, "calibration_claimed": False,
                "traces": deepcopy(self.traces)}
