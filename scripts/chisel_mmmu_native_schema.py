"""Equivalent finite-property-name schemas for native constrained generation."""
from __future__ import annotations

import copy
import json


def validated_json_object_transport(payload):
    """Use JSON-object decoding while retaining the exact schema for validation.

    This deliberately changes generation guidance, not the official acceptance
    schema. The caller must retain the official post-generation validator.
    """
    result = copy.deepcopy(payload)
    response_format = result.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return result, None
    contract = response_format.get("json_schema")
    if not isinstance(contract, dict) or not isinstance(contract.get("schema"), dict):
        raise ValueError("Structured finalization requires its complete original JSON schema")
    schema = contract["schema"]
    if schema.get("type") != "object":
        raise ValueError("JSON-object transport is valid only for object-valued official sessions")
    if not isinstance(result.get("messages"), list) or not result["messages"]:
        raise ValueError("Structured finalization lacks its original message context")
    result["messages"].append({
        "role": "user",
        "content": (
            "Return one JSON object satisfying this complete, unchanged output schema. "
            "The official validator will reject any schema violation, including duplicate "
            "array items, extra fields, invalid candidate IDs or missing required fields.\n"
            + json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))),
    })
    result["response_format"] = {"type": "json_object"}
    return result, {
        "mode": "json_object_generation_with_exact_official_postvalidation",
        "original_response_format": copy.deepcopy(response_format),
        "wire_response_format": {"type": "json_object"},
        "original_schema_added_to_model_context": True,
        "official_output_schema_modified": False,
        "native_generation_guidance_changed": True,
        "official_validation_remains_required": True,
    }


def native_json_schema(schema):
    """Expand finite propertyNames without weakening the official output schema."""
    changes = []

    def visit(node, path):
        if isinstance(node, list):
            return [visit(value, path + (str(index),)) for index, value in enumerate(node)]
        if not isinstance(node, dict):
            return copy.deepcopy(node)
        result = copy.deepcopy(node)
        for key in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            if isinstance(result.get(key), dict):
                result[key] = {name: visit(value, path + (key, name))
                               for name, value in result[key].items()}
        for key in ("items", "additionalProperties", "propertyNames", "contains", "not",
                    "if", "then", "else", "unevaluatedProperties", "unevaluatedItems",
                    "allOf", "anyOf", "oneOf", "prefixItems"):
            if key in result:
                result[key] = visit(result[key], path + (key,))
        if "propertyNames" not in result:
            return result
        names = result["propertyNames"]
        if (result.get("type") != "object" or not isinstance(names, dict)
                or set(names) != {"enum"} or not isinstance(names["enum"], list)
                or not names["enum"] or any(not isinstance(name, str) for name in names["enum"])
                or "patternProperties" in result or "unevaluatedProperties" in result):
            raise ValueError("Unsupported propertyNames form; do not weaken the official schema")
        properties = result.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("Object properties must be a schema mapping")
        value_schema = result.get("additionalProperties", True)
        allowed = set(names["enum"])
        if not set(result.get("required", [])).issubset(allowed):
            raise ValueError("Unsatisfiable required/propertyNames schema")
        result["properties"] = {
            name: properties[name] if name in properties else copy.deepcopy(value_schema)
            for name in sorted(allowed)}
        result["additionalProperties"] = False
        del result["propertyNames"]
        changes.append({"path": "/".join(path), "operation": "finite-propertyNames-to-explicit-properties",
                        "allowed_names": sorted(allowed), "semantic_constraints_preserved": True})
        return result

    return visit(schema, ()), changes
