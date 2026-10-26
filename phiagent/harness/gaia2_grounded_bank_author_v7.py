"""Paid CoT candidate authorship grounded in the completed native public receipt."""
from copy import deepcopy
import json
from phiagent.harness.gaia2_cot_jev import cot_response_format, parse_bank

CONTRACT = "native_parent_post_observation_continuation_bank_v7"
AUTHOR_INTERVAL_SLOW_CALLS = 6
AUTHOR = """The supplied native action has already executed. Its latest public observation
is the actual returned result, not a predicted result. Author useful next native
operations from the task, this result, the completed action and its prior reasoning.
Return only a continuation bank with note and candidates. Use the actual native
tool schemas. Preserve all task filters, exceptions, ordering and authorization.
Use exact observed identifiers and field names. Known public arguments may be
literals; $obs and $prev refer to the supplied actual result and completed action.
$receipt and foreach/$item may use the supplied original public tool receipts.
Do not invent future observations, identifiers or schema fields. Reason through
which discovered records satisfy the task before declaring a candidate. Conditions
must state the evidence that makes each next operation warranted. Include useful
alternatives only when their arguments can be grounded. An empty bank is correct
when interpretation or further native reasoning is required. Every action needs a
separate one-token selection or abstention; no automatic sequence executes here.
Do not propose completion, waits, messages or unsolicited writes."""


def author_request(model, tools, task, observation, completed_action, native_reasoning, receipts):
    envelope = cot_response_format(tools, allow_routine_writes=True, continuation_bank=True)
    schema = envelope["json_schema"]["schema"]
    bank_schema = deepcopy(schema["properties"]["jev_bank"])
    if "$defs" in schema:
        bank_schema["$defs"] = deepcopy(schema["$defs"])
    state = {"task": task, "latest_public_observation": observation,
             "completed_native_action": completed_action,
             "prior_native_reasoning": native_reasoning,
             "original_public_tool_receipts": receipts,
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
