"""Recover rejected native pagination reads from their public error and schema."""
from __future__ import annotations

from copy import deepcopy
import json
import re
import time

from phiagent.harness import gaia2_202060_progress as progress


_TOOLS = {
    "Messages__list_recent_conversations",
    "Messages__list_conversations_by_participant",
}
_LIMIT_ERROR = re.compile(
    r"(?:^|\n)(?:Exception: )?(?:Error in tool call execution: )?"
    r"Limit must be smaller than the view limit of ([1-9][0-9]*)"
    r" - Please use a smaller limit and use offset to navigate(?:\n|$)"
)
_INTERRUPTIONS = {
    "task", "agent_user_interface", "environment_notifications", "hint", "task_reminder",
}


def recover_read_limit(action, error, tools):
    """Recover only a read-limit rejection matching the tool's public default."""
    if not isinstance(action, dict) or not isinstance(error, str):
        return None
    name, arguments = action.get("action"), action.get("action_input")
    if (not isinstance(name, str) or name not in _TOOLS or name not in tools
            or not isinstance(arguments, dict)):
        return None
    matches = _LIMIT_ERROR.findall(error)
    if len(matches) != 1:
        return None
    bound = int(matches[0])
    inputs = getattr(tools[name], "inputs", None)
    if not isinstance(inputs, dict):
        return None
    schema = inputs.get("limit")
    if (not isinstance(schema, dict) or schema.get("type") != "integer"
            or type(schema.get("default")) is not int or schema["default"] != bound
            or type(arguments.get("limit")) is not int or arguments["limit"] <= bound):
        return None
    nested = arguments.get("limit_recent_messages_per_conversation")
    nested_schema = inputs.get("limit_recent_messages_per_conversation", {})
    if nested is not None and (
        not isinstance(nested_schema, dict) or type(nested) is not int or nested <= 0
        or type(nested_schema.get("default")) is not int
        or nested != nested_schema["default"]
    ):
        return None
    corrected = deepcopy(action)
    corrected["action_input"]["limit"] = bound
    return corrected


class ReadLimitRecoveryEngine(progress.ReceiptEngine):
    def __call__(self, prompt, **kwargs):
        started = time.monotonic()
        logs = self.agent.logs
        task = "\n".join(
            row.content for row in logs
            if row.get_type() == "task" and isinstance(getattr(row, "content", None), str)
        )
        fresh = logs[self.cursor:]
        kinds = [row.get_type() for row in fresh]
        errors = [row.get_content_for_llm() for row in fresh if row.get_type() == "error"]
        correction = None
        if (self.graph.execute_followups and task == self.graph.task
                and self.cursor <= len(logs) and len(errors) == 1
                and not _INTERRUPTIONS.intersection(kinds) and "observation" not in kinds):
            correction = recover_read_limit(
                self.graph.previous, errors[0], self.agent.action_executor.tools
            )
        if correction is None or progress.key(correction) in self.graph.attempted:
            return super().__call__(prompt, **kwargs)
        failed_action = deepcopy(self.graph.previous)
        self.graph.observe(errors[0], error=True)
        self.cursor = len(logs)
        text = self.graph.remember(
            "Thought: Retry the rejected read with its public pagination limit.\nAction:\n"
            + json.dumps(correction, ensure_ascii=False),
            set(self.agent.action_executor.tools),
        )
        self.record({
            "source": "receipt_graph_execution", "agent_id": self.agent.agent_id,
            "executed_response": text, "failed_action": failed_action,
            "native_error_sha256": progress.key(errors[0]),
            "checks": {
                "public_read_limit_recovery": True, "typed_native_error": True,
                "public_schema_default_matches_bound": True,
                "only_read_limit_changed": True,
            },
            "provider_response": False, "new_model_calls": 0,
        })
        return text, {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "completion_duration": time.monotonic() - started,
        }


def install_adapter(config, directory, role, mode, skill):
    progress.ReceiptEngine = ReadLimitRecoveryEngine
    progress.install_adapter(config, directory, role, mode, skill)
