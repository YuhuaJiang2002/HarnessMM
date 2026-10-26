"""Native GAIA2 public-progress actions, selected without a model or refresh timer.

The first supported routine enumerates email pages under an explicit all-mail
task. It uses the installed ReturnedEmails range semantics, never email content
as executable code. Unsupported states return to the original native actor.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass, field
import json
import re

from phiagent.harness.gaia2_jev import action_object, fingerprint


def email_page(observation):
    if not isinstance(observation, str):
        return None
    # The native executor serializes ReturnedEmails with its dataclass repr.
    start = observation.find("ReturnedEmails(")
    if start < 0:
        return None
    expression = observation[start:observation.rfind(")") + 1]
    try:
        tree = ast.parse(expression, mode="eval").body
        if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != "ReturnedEmails" or tree.args:
            return None
        fields = {entry.arg: entry.value for entry in tree.keywords}
        if len(fields) != len(tree.keywords) or set(fields) != {"emails", "emails_range", "total_returned_emails", "total_emails"}:
            return None
        interval = ast.literal_eval(fields["emails_range"])
        total = ast.literal_eval(fields["total_emails"])
        if not isinstance(interval, tuple) or len(interval) != 2 or any(type(x) is not int for x in (*interval, total)):
            return None
        if not 0 <= interval[0] < interval[1] <= total:
            return None
        return interval[0], interval[1], total
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


@dataclass
class NativePublicProgress:
    require_stall: bool = True
    task: str = ""
    previous_action: dict | None = None
    previous_feedback: str | None = None
    stalled: bool = False
    pending: list = field(default_factory=list)
    attempted: set = field(default_factory=set)
    fast_steps: int = 0

    def remember(self, response, legal_tools):
        action = action_object(response)
        self.previous_action = None if action is None else {key: deepcopy(action[key]) for key in ("action", "action_input")}
        if self.previous_action is not None:
            self.attempted.add(json.dumps(self.previous_action, sort_keys=True))
        return response

    def decide(self, *, observation, error, intervened, legal_tools):
        current = fingerprint(self.previous_action, observation) if self.previous_action is not None and isinstance(observation, str) else None
        if error or current is not None and current == self.previous_feedback:
            self.stalled = True
        self.previous_feedback = current
        prior = self.previous_action or {}
        args = prior.get("action_input", {})
        page = email_page(observation)
        # Preserve the actor's folder and page size; only advance its public offset.
        eligible = (page is not None and prior.get("action", "").endswith("__list_emails")
                    and set(args) <= {"folder_name", "offset", "limit"}
                    and type(args.get("offset", 0)) is int and type(args.get("limit", 10)) is int
                    and args.get("limit", 10) > 0 and page[0] == args.get("offset", 0)
                    and page[1] == min(page[0] + args.get("limit", 10), page[2])
                    and page[1] < page[2])
        task_supported = bool(re.search(r"\ball\b", self.task, re.I) and re.search(r"\b(emails?|mail|inbox)\b", self.task, re.I))
        action = {"action": prior.get("action"), "action_input": {**args, "offset": page[1]}} if eligible else None
        checks = {"public_observation": isinstance(observation, str) and bool(observation.strip()),
                  "no_public_error": not error, "no_intervening_event": not intervened,
                  "stall_gate": self.stalled or not self.require_stall,
                  "task_requires_all_mail": task_supported, "native_range_progress": bool(eligible),
                  "native_tool_available": prior.get("action") in legal_tools,
                  "unattempted_action": action is not None and json.dumps(action, sort_keys=True) not in self.attempted}
        record = {"event": "decision", "execute": all(checks.values()), "checks": checks,
                  "feedback_sha256": current, "selection_model_calls": 0, "periodic_refresh": False,
                  "compiler": "native_email_pagination_v1", "public_range": page,
                  "candidate_origin": "native_tool_and_public_range", "fast_steps": self.fast_steps}
        if not record["execute"]:
            return None, record
        self.fast_steps += 1
        self.previous_action = action
        self.attempted.add(json.dumps(action, sort_keys=True))
        return "Thought: Continue the requested enumeration using the next public range.\nAction:\n" + json.dumps(action), record
