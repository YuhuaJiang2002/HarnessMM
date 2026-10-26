"""Slow CoT bank authorship over native schemas and public task state."""
from copy import deepcopy
import json
from phiagent.harness.gaia2_cot_jev import cot_response_format, parse_bank

CONTRACT = "native_parent_separate_continuation_bank_v6"
AUTHOR_INTERVAL_SLOW_CALLS = 6
AUTHOR = """Author conditional continuations for the supplied public task and current
native action. The current action was already chosen by the native CoT actor;
do not replace it. Reason about the next evidence-producing reads and warranted
follow-up operations. Return only a continuation bank with note and candidates.
Use actual native tool names and parameter types from the supplied schemas.
Known public arguments may be literals. Future IDs must use $obs, $prev,
$receipt or foreach/$item bindings. Never guess a field or an identifier.
Use conditions that specify what the next public observation must establish
before a candidate is safe. Include distinct useful alternatives for discovery
and follow-up when their arguments can be bound. Empty candidates are correct
when no later action is justified. No candidate is executed here; every next
operation is separately selected or declined by the one-token discriminator.
Do not propose completion, waits, messages or unsolicited writes."""


def author_request(model, tools, task, observation, current_action, native_reasoning):
    envelope = cot_response_format(tools, allow_routine_writes=True, continuation_bank=True)
    schema = envelope["json_schema"]["schema"]
    bank_schema = deepcopy(schema["properties"]["jev_bank"])
    if "$defs" in schema:
        bank_schema["$defs"] = deepcopy(schema["$defs"])
    state = {"task": task, "latest_public_observation": observation,
             "current_native_action": current_action,
             "current_native_reasoning": native_reasoning,
             "native_tools": {name: {"description": str(getattr(tool, "description", "")),
                                     "inputs": deepcopy(tool.inputs)}
                              for name, tool in tools.items()}}
    return {"model": model, "messages": [
        {"role": "system", "content": AUTHOR},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False)}],
        "max_tokens": 4096,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "native_continuation_bank", "strict": True, "schema": bank_schema}}}


def parse_author(raw, tools):
    choice = raw["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("A continuation author must finish naturally")
    return parse_bank({"jev_bank": json.loads(choice["message"]["content"])}, tools,
                      allow_routine_writes=True, continuation_bank=True)
