"""Native-schema CoT banks with bindings to original public tool receipts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from phiagent.harness.gaia2_dual_rsi import well_typed


CONTRACT = "native_schema_public_receipts_v1"
INSTRUCTION = """
Author a continuation bank covering the evidence-producing reads and warranted
follow-up operations in your plan, not just one alternative to the issued action.
The bank schema binds every candidate to that native tool's actual parameter
names and types. Omit optional parameters that are not needed; never insert null
or a placeholder for an unresolved required argument. Use a declared binding.
Use up to eight distinct justified alternatives, including alternatives for
different possible public outcomes. An empty bank is correct when none is safe.
An argument may additionally use
{"$receipt":{"tool":"native tool name","path":["field",0,"id"]}}.
This reads the most recent original public observation of that tool in this task.
Optional "arguments":{...} restricts the originating tool call by exact arguments.
For iteration over an earlier public list, use
"foreach":{"receipt":{"tool":"native list tool"},"path":[],"where":{...}}
and bind fields with {"$item":["id"]}. This preserves the observed list across
intervening reads; it does not infer new records or automatically execute a loop.
These are historical receipts, not proof of current state or successful effects.
Conditions must explain what current evidence warrants each action. Keep timing,
direction, filters and exceptions in the note. Every next action is separately
selected by the one-token discriminator or deferred to full reasoning.
Missing receipts, unknown observations, unresolved fields, type mismatches and
more than eight instantiated alternatives still require full reasoning.
"""


def path_schema():
    return {
        "type": "array", "maxItems": 8,
        "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
    }


def receipt_schema(tool_names, *, with_path):
    properties = {
        "tool": {"type": "string", "enum": sorted(tool_names)},
        "arguments": {"type": "object"},
    }
    required = ["tool"]
    if with_path:
        properties["path"] = path_schema()
        required.append("path")
    return {
        "type": "object", "additionalProperties": False,
        "required": required, "properties": properties,
    }


def references(tool_names):
    result = [
        {"type": "object", "additionalProperties": False,
         "required": [operator], "properties": {operator: path_schema()}}
        for operator in ("$obs", "$prev", "$item")
    ]
    result.append({
        "type": "object", "additionalProperties": False, "required": ["$receipt"],
        "properties": {"$receipt": receipt_schema(tool_names, with_path=True)},
    })
    return result


def literal_schema(specification):
    kind = specification.get("type")
    if kind not in ("string", "integer", "number", "boolean", "array", "object"):
        raise ValueError("No fast literal contract for native type: " + str(kind))
    keys = (
        "type", "enum", "const", "properties", "required", "additionalProperties",
        "minLength", "maxLength", "pattern", "minimum", "maximum", "minItems", "maxItems",
    )
    result = {key: deepcopy(specification[key]) for key in keys if key in specification}
    if kind == "array":
        try:
            result["items"] = literal_schema(specification.get("items", {}))
        except ValueError:
            result["maxItems"] = 0
    return result


def argument_schema(specification):
    reference = {"$ref": "#/$defs/cot_jev_public_argument_reference"}
    alternatives = [literal_schema(specification), reference]
    if specification["type"] in ("integer", "number"):
        term = {"anyOf": [{"type": specification["type"]}, reference]}
        alternatives.append({
            "type": "object", "additionalProperties": False, "required": ["$add"],
            "properties": {"$add": {"type": "array", "minItems": 2, "maxItems": 2,
                                    "items": term}},
        })
    return {"anyOf": alternatives}


def response_format(base, tools, legal_names):
    result = deepcopy(base)
    schema = result["json_schema"]["schema"]
    schema.setdefault("$defs", {}).update({
        "cot_jev_public_argument_reference": {"anyOf": references(tools)},
        "cot_jev_public_receipt_source": receipt_schema(tools, with_path=False),
    })
    bank = schema["properties"]["jev_bank"]["properties"]
    bank["note"].update(minLength=1, maxLength=1200)
    candidates = bank["candidates"]
    alternatives = []
    for name in legal_names:
        properties, required = {}, []
        unsupported_required = False
        for key, specification in tools[name].inputs.items():
            try:
                properties[key] = argument_schema(specification)
            except ValueError:
                unsupported_required |= "default" not in specification
                continue
            if "default" not in specification:
                required.append(key)
        if unsupported_required:
            continue
        alternatives.append({
            "type": "object", "additionalProperties": False,
            "required": ["when", "action", "action_input"],
            "properties": {
                "when": {"type": "string", "minLength": 1, "maxLength": 500},
                "action": {"type": "string", "const": name},
                "action_input": {"type": "object", "additionalProperties": False,
                                 "required": required, "properties": properties},
                "foreach": {
                    "type": "object", "additionalProperties": False,
                    "required": ["path", "where"],
                    "properties": {
                        "path": path_schema(), "where": {"type": "object"},
                        "receipt": {"$ref": "#/$defs/cot_jev_public_receipt_source"},
                    },
                },
            },
        })
    candidates["maxItems"] = 8 if alternatives else 0
    if alternatives:
        candidates["items"] = {"anyOf": alternatives}
    return result


def validate_receipt_reference(reference, *, with_path=True):
    required = {"tool", "path"} if with_path else {"tool"}
    if (
        not isinstance(reference, dict) or not required <= set(reference)
        or set(reference) - (required | {"arguments"})
        or not isinstance(reference["tool"], str)
        or "arguments" in reference and not isinstance(reference["arguments"], dict)
    ):
        raise ValueError("Invalid native public receipt reference")
    if with_path:
        path = reference["path"]
        if (not isinstance(path, list) or len(path) > 8
                or any(not isinstance(key, str) and not (type(key) is int and key >= 0)
                       for key in path)):
            raise ValueError("A bounded native receipt field path is required")


def receipt_value(reference, receipts):
    from phiagent.harness.gaia2_cot_jev import at_path, observation_data

    validate_receipt_reference(reference)
    matches = []
    for receipt in receipts:
        call = receipt.get("native_tool_call")
        if not isinstance(call, dict) or call.get("tool_name") != reference["tool"]:
            continue
        content = receipt.get("observation")
        index = receipt.get("log_index")
        if (
            not isinstance(content, str) or type(index) is not int or index < 0
            or receipt.get("historical_receipt_not_current_state") is not True
            or hashlib.sha256(content.encode()).hexdigest()
            != receipt.get("native_observation_sha256")
        ):
            raise ValueError("Public receipt provenance does not match original observation bytes")
        arguments = call.get("tool_arguments")
        if isinstance(arguments, str):
            arguments = observation_data(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Native receipt arguments are unavailable")
        if "arguments" in reference and arguments != reference["arguments"]:
            continue
        matches.append(receipt)
    if not matches:
        raise ValueError("No matching original public tool receipt")
    latest = max(row["log_index"] for row in matches)
    selected = [row for row in matches if row["log_index"] == latest]
    if len(selected) != 1:
        raise ValueError("Ambiguous native public receipt")
    receipt = selected[0]
    return at_path(observation_data(receipt["observation"]), reference["path"]), receipt


def bind_arguments(value, observation, previous, item, receipts, used):
    from phiagent.harness.gaia2_cot_jev import bind

    def resolve(current):
        if isinstance(current, dict):
            operators = set(current) & {"$obs", "$prev", "$item", "$add", "$receipt"}
            if operators:
                if len(current) != 1:
                    raise ValueError("Binding operators cannot hide additional fields")
                if "$receipt" in current:
                    resolved, receipt = receipt_value(current["$receipt"], receipts)
                    used[receipt["log_index"]] = receipt
                    return resolved
                if "$add" in current:
                    terms = current["$add"]
                    if not isinstance(terms, list) or len(terms) != 2:
                        raise ValueError("Only explicit two-term numeric addition is supported")
                    return bind({"$add": [resolve(term) for term in terms]}, {}, {})
                return bind(current, observation, previous, item)
            return {key: resolve(child) for key, child in current.items()}
        if isinstance(current, list):
            return [resolve(child) for child in current]
        return bind(current, observation, previous, item)

    return resolve(value)


def instantiate(bank, observation, previous, tools, seen, receipts):
    from phiagent.harness.gaia2_cot_jev import at_path, digest
    from phiagent.harness.gaia2_public_literal_table import UnsupportedLiteral

    rows, failures, identities, used = [], [], set(), {}
    for index, template in enumerate(bank["candidates"]):
        try:
            local_used = {}
            members = [None]
            if "foreach" in template:
                scope = template["foreach"]
                if "receipt" in scope:
                    validate_receipt_reference(scope["receipt"], with_path=False)
                    source, receipt = receipt_value(
                        {**scope["receipt"], "path": scope["path"]}, receipts,
                    )
                    local_used[receipt["log_index"]] = receipt
                else:
                    source = at_path(observation, scope["path"])
                if not isinstance(source, list):
                    raise ValueError("foreach requires a public record list")
                members = [
                    item for item in source if isinstance(item, dict)
                    and all(key in item and item[key] == expected
                            for key, expected in scope["where"].items())
                ]
            for member in members:
                arguments = bind_arguments(
                    template["action_input"], observation, previous, member, receipts, local_used,
                )
                json.dumps(arguments, allow_nan=False)
                specifications = tools[template["action"]].inputs
                if set(arguments) - set(specifications) or any(
                    key not in arguments and "default" not in specification
                    or key in arguments and not well_typed(arguments[key], specification)
                    for key, specification in specifications.items()
                ):
                    raise ValueError("Candidate violates the declared native argument contract")
                action = {"action": template["action"], "action_input": arguments}
                fingerprint = digest(action)
                if fingerprint in identities or fingerprint in seen:
                    continue
                identities.add(fingerprint)
                used.update(local_used)
                rows.append({
                    "action": action, "when": template["when"], "template": index,
                    "binding_receipt_indices": sorted(local_used),
                })
        except (KeyError, IndexError, TypeError, ValueError, SyntaxError, UnsupportedLiteral) as error:
            failures.append({"template": index, "reason": str(error)})
    if len(rows) > 8:
        return [], failures + [{"reason": "candidate_capacity_overflow_no_truncation"}], []
    return rows, failures, [deepcopy(used[index]) for index in sorted(used)]
