"""TRAIN-only paid fast proposals with native full-history escalation."""
from __future__ import annotations

from copy import deepcopy
import json
import re

from phiagent.harness.gaia2_jev import action_object
from phiagent.harness.gaia2_lazy_tool_schemas import read_schema_can_be_deferred
from phiagent.harness.gaia2_state_ledger import PREFIX, project_ledger


FAST_REQUEST = (
    "[Fast read-only proposal]\n"
    "The complete task and native tool contracts remain above. Only recent tool "
    "history is visible in this proposal. Older evidence is available to the "
    "full-history reasoning pass, not to this view. Propose one read-only native "
    "action whose arguments are supported by the visible task, observations, or "
    "documented tool defaults. "
    "Do not declare completion, issue a write, or add a progress_plan. "
    "If a needed fact or decision depends on older history, reply DEFER. "
    "Use the ordinary native Thought/Action format for a proposed read."
)
OBSERVATION = re.compile(r"^\[OUTPUT OF STEP \d+\] Observation:")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def project_fast(messages, keep_pairs=2):
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Native system and full tool contracts are required")
    observations = [
        index for index, message in enumerate(messages)
        if isinstance(message.get("content"), str)
        and OBSERVATION.match(message["content"])
    ]
    if len(observations) <= keep_pairs:
        return None
    keep = {0}
    for index, message in enumerate(messages):
        text = message.get("content", "")
        if message.get("role") in ("system", "user") and isinstance(text, str):
            if not OBSERVATION.match(text) and not text.startswith(PREFIX):
                keep.add(index)
    for index in observations[-keep_pairs:]:
        keep.add(index)
        if index and messages[index - 1].get("role") == "assistant":
            keep.add(index - 1)
    if observations[-1] < len(messages) - 1:
        keep.update(range(observations[-1] + 1, len(messages)))
    projected = []
    for index in sorted(keep):
        message = deepcopy(messages[index])
        text = message.get("content", "")
        if isinstance(text, str) and text.startswith(PREFIX):
            ledger = json.loads(text[len(PREFIX):])
            ledger["action_receipts"] = ledger.get("action_receipts", [])[-keep_pairs:]
            ledger["older_evidence_requires_full_history_reasoning"] = True
            message["content"] = PREFIX + json.dumps(ledger, ensure_ascii=False)
        projected.append(message)
    projected.append({"role": "user", "content": FAST_REQUEST})
    if projected[0] != messages[0]:
        raise ValueError("Fast proposals cannot change the native tool catalogue")
    return projected


def supported_arguments(value, evidence):
    if isinstance(value, dict):
        return all(supported_arguments(child, evidence) for child in value.values())
    if isinstance(value, list):
        return all(supported_arguments(child, evidence) for child in value)
    if isinstance(value, str):
        return not value or value in evidence
    if value is None or isinstance(value, bool):
        return True
    if type(value) in (int, float):
        return str(value) in evidence
    return False


def well_typed(value, specification):
    kind = specification.get("type")
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float)
    if kind == "boolean":
        return type(value) is bool
    if kind == "array":
        return isinstance(value, list) and all(
            well_typed(item, specification.get("items", {})) for item in value
        )
    if kind == "object":
        return isinstance(value, dict)
    return False


def accept_fast(text, tools, read_only, evidence):
    action = action_object(text)
    if not action or action.get("progress_plan"):
        return False, "defer_or_nonlocal_plan"
    name, arguments = action["action"], action["action_input"]
    if name not in read_only or name not in tools or not isinstance(arguments, dict):
        return False, "not_an_authoritatively_read_only_action"
    inputs = tools[name].inputs
    if set(arguments) - set(inputs):
        return False, "undeclared_native_arguments"
    for key, specification in inputs.items():
        if key not in arguments:
            if "default" not in specification:
                return False, "missing_required_native_argument"
        elif not well_typed(arguments[key], specification):
            return False, "native_argument_type_mismatch"
    nondefault = {
        key: value for key, value in arguments.items()
        if "default" not in inputs[key] or value != inputs[key]["default"]
    }
    if not supported_arguments(nondefault, evidence):
        return False, "arguments_need_full_history"
    return True, "supported_native_read"


def public_evidence(messages):
    return "\n".join(
        message["content"] for message in messages
        if message.get("role") not in ("system", "assistant")
        and isinstance(message.get("content"), str)
        and message["content"] != FAST_REQUEST
        and not message["content"].startswith(PREFIX)
    )


def combine_usage(first, second):
    for usage in (first, second):
        if (any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_FIELDS)
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("Both paid proposals require complete native usage")
    result = {key: first[key] + second[key] for key in USAGE_FIELDS}
    result["completion_duration"] = first["completion_duration"] + second["completion_duration"]
    return result


def install_adapter(config, directory, role, mode, skill):
    if (config.get("method") != "chisel" or mode != "jev"
            or config.get("stage") != "train-candidate"
            or config.get("evaluation_split") != "train"):
        raise RuntimeError("The new dual-system policy is frozen for TRAIN only")
    policy = config["dual_rsi_policy"]
    if (policy["keep_observation_pairs"] not in (1, 2)
            or policy["minimum_full_calls"] != 2 or policy["maximum_fast_streak"] not in (2, 3)
            or policy["maximum_fast_input_ratio"] != 0.5):
        raise ValueError("Unexpected prospective fast/slow policy")
    from phiagent.harness import gaia2_202060_progress as progress
    from scripts import run_gaia2_jev as base

    parent = progress.ReceiptEngine

    class DualEngine(parent):
        def __init__(self, agent, provider, *, skill, record, execute_followups):
            first_log = agent.logs[0] if agent.logs else None
            last_cursor = full_calls = fast_streak = 0
            read_only = {
                name for name, tool in agent.action_executor.tools.items()
                if read_schema_can_be_deferred(
                    name, write_operation=getattr(getattr(tool, "app_tool", None), "write_operation", None),
                )
            }

            def count(messages):
                converted = [provider._convert_message_to_litellm_format(message) for message in messages]
                result = base.get_json(
                    f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",
                    {"model": config["model"]["served_name"], "messages": converted,
                     "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True},
                    timeout=60,
                )
                if type(result.get("count")) is not int or result["count"] < 1:
                    raise ValueError("Native tokenizer returned an invalid request length")
                return result["count"]

            def dual_provider(messages, **kwargs):
                nonlocal first_log, last_cursor, full_calls, fast_streak
                current_first = agent.logs[0] if agent.logs else None
                if current_first is not first_log or last_cursor > len(agent.logs):
                    first_log, last_cursor, full_calls, fast_streak = current_first, 0, 0, 0
                fresh = agent.logs[last_cursor:]
                last_cursor = len(agent.logs)
                interrupted = any(
                    row.get_type() in ("task", "agent_user_interface", "environment_notifications",
                                       "hint", "task_reminder", "error")
                    for row in fresh
                )
                errors = any(
                    re.search(r'(?i)(?:error:|traceback \(|"error"\s*:)', str(getattr(row, "content", "")))
                    for row in fresh
                )
                full, ledger_audit = project_ledger(messages)
                fast = None
                if (not interrupted and not errors and full_calls >= policy["minimum_full_calls"]
                        and fast_streak < policy["maximum_fast_streak"] and read_only):
                    fast = project_fast(full, policy["keep_observation_pairs"])
                lengths = {}
                if fast is not None:
                    lengths = {"full_input_tokens": count(full), "fast_input_tokens": count(fast)}
                    if lengths["fast_input_tokens"] > policy["maximum_fast_input_ratio"] * lengths["full_input_tokens"]:
                        fast = None
                if fast is None:
                    response = provider(full, **kwargs)
                    full_calls += 1
                    fast_streak = 0
                    record({"source": "dual_rsi_route", "route": "full_history",
                            "agent_id": agent.agent_id, "ledger": ledger_audit, **lengths})
                    return response
                proposal, proposal_usage = provider(fast, **kwargs)
                accepted, reason = accept_fast(
                    proposal, agent.action_executor.tools, read_only, public_evidence(fast),
                )
                record({
                    "source": "dual_rsi_route", "route": "fast_read" if accepted else "paid_full_escalation",
                    "agent_id": agent.agent_id, "reason": reason, "proposal_usage": proposal_usage,
                    "complete_tool_contracts_preserved": True, "native_history_modified": False, **lengths,
                })
                if accepted:
                    fast_streak += 1
                    return proposal, proposal_usage
                answer, slow_usage = provider(full, **kwargs)
                full_calls += 1
                fast_streak = 0
                return answer, combine_usage(proposal_usage, slow_usage)

            super().__init__(agent, dual_provider, skill=skill, record=record,
                             execute_followups=execute_followups)

    progress.ReceiptEngine = DualEngine
    progress.install_adapter(config, directory, role, mode, skill)
