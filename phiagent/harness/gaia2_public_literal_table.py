"""Reversible tables for repeated public record schemas; never evaluates tool text."""
from __future__ import annotations

import ast
from functools import lru_cache
import json
import math
import re

from phiagent.harness.gaia2_token_efficiency import (
    OBSERVATION, CompactCalendarReceiptEngine, Projection, encoded, fingerprint,
    project_public_messages,
)


POLICY = "public_literal_tables_native_wire_capacity_v3"
PREFIX = (
    "[Lossless public observation table. @ denotes a record in its indexed schema; "
    "@id also defines an immutable record ID, and = reuses that exact earlier record. "
    "l/d/t/s denote list/dictionary/tuple/set; u denotes a signed literal. "
    "All values and their order are retained; a changed record receives a new ID.]\n"
)


class UnsupportedLiteral(ValueError):
    pass


def encode_literal(text):
    parsed = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(parsed)) > 200000:
        raise UnsupportedLiteral("public_literal_too_large")
    schemas = []
    indices = {}

    def pack(node):
        if isinstance(node, ast.Constant):
            value = node.value
            if (type(value) not in (str, int, float, bool, type(None))
                    or isinstance(value, float) and not math.isfinite(value)):
                raise UnsupportedLiteral("unsupported_literal_type")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if not isinstance(node.operand, ast.Constant) or type(node.operand.value) not in (int, float):
                raise UnsupportedLiteral("unsupported_signed_literal")
            return ["u", "-" if isinstance(node.op, ast.USub) else "+", pack(node.operand)]
        for kind, tag in ((ast.List, "l"), (ast.Tuple, "t"), (ast.Set, "s")):
            if isinstance(node, kind):
                return [tag, [pack(value) for value in node.elts]]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise UnsupportedLiteral("dictionary_unpacking_forbidden")
            return ["d", [[pack(key), pack(value)] for key, value in zip(node.keys, node.values)]]
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name) or not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", node.func.id)
                    or node.args or any(keyword.arg is None for keyword in node.keywords)):
                raise UnsupportedLiteral("only_named_public_record_fields_are_supported")
            fields = tuple(keyword.arg for keyword in node.keywords)
            if len(fields) != len(set(fields)):
                raise UnsupportedLiteral("duplicate_record_fields")
            key = node.func.id, fields
            if key not in indices:
                indices[key] = len(schemas)
                schemas.append([node.func.id, list(fields)])
            return ["@", indices[key], [pack(keyword.value) for keyword in node.keywords]]
        raise UnsupportedLiteral("expression_is_not_public_literal_data")

    value = pack(parsed.body)
    payload = {"format": "public-literal-table-v1", "schemas": schemas, "value": value}
    if ast.dump(decode_literal(payload)) != ast.dump(parsed.body):
        raise ValueError("Public literal schema/value round trip changed the native data")
    return payload


def decode_literal(payload, records=None):
    schemas = payload["schemas"]
    records = {} if records is None else records

    def unpack(value):
        if not isinstance(value, list):
            return ast.Constant(value=value)
        tag = value[0]
        if tag == "=":
            return records[value[1]]
        if tag == "@id":
            record = unpack(["@", value[2], value[3]])
            if value[1] in records:
                raise ValueError("Public record IDs are immutable")
            records[value[1]] = record
            return record
        if tag == "@":
            name, fields = schemas[value[1]]
            if len(fields) != len(value[2]):
                raise ValueError("Record schema and values have different lengths")
            return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[],
                            keywords=[ast.keyword(arg=name, value=unpack(item))
                                      for name, item in zip(fields, value[2])])
        if tag == "u":
            return ast.UnaryOp(op=ast.USub() if value[1] == "-" else ast.UAdd(),
                               operand=unpack(value[2]))
        if tag == "l":
            return ast.List(elts=[unpack(item) for item in value[1]], ctx=ast.Load())
        if tag == "t":
            return ast.Tuple(elts=[unpack(item) for item in value[1]], ctx=ast.Load())
        if tag == "s":
            return ast.Set(elts=[unpack(item) for item in value[1]])
        if tag == "d":
            return ast.Dict(keys=[unpack(item[0]) for item in value[1]],
                            values=[unpack(item[1]) for item in value[1]])
        raise ValueError("Unknown public literal table tag")

    return unpack(payload["value"])


@lru_cache(maxsize=64)
def _literal_encoding(text):
    payload = encode_literal(text)
    return encoded(payload), len(payload["schemas"]), fingerprint(payload)


def intern_records(payload, registry):
    schemas = payload["schemas"]
    reused = 0

    def intern(node):
        nonlocal reused
        if not isinstance(node, list):
            return node
        tag = node[0]
        if tag == "@":
            canonical = ast.dump(decode_literal({"schemas": schemas, "value": node}))
            key = fingerprint(canonical)
            worth_interning = len(encoded(node)) >= 64
            if worth_interning and key in registry:
                reused += 1
                return ["=", registry[key]]
            values = [intern(value) for value in node[2]]
            if not worth_interning:
                return ["@", node[1], values]
            identifier = "R" + str(len(registry))
            registry[key] = identifier
            return ["@id", identifier, node[1], values]
        if tag in ("l", "t", "s"):
            return [tag, [intern(value) for value in node[1]]]
        if tag == "d":
            return ["d", [[intern(key), intern(value)] for key, value in node[1]]]
        if tag == "u":
            return ["u", node[1], intern(node[2])]
        raise ValueError("Unrecognized public record value")

    return {**payload, "value": intern(payload["value"])}, reused


def table_observations(messages, *, reuse_records=False):
    projected = [dict(message) for message in messages]
    records = []
    registry, decoded_records = {}, {}
    for index, message in enumerate(messages):
        text = message.get("content")
        if message.get("role") != "user" or not isinstance(text, str):
            continue
        match = OBSERVATION.fullmatch(text)
        if match is None or len(match["body"]) < 2048:
            continue
        body = match["body"]
        try:
            text_encoding, schema_count, encoded_hash = _literal_encoding(body)
        except (SyntaxError, UnsupportedLiteral, RecursionError) as error:
            records.append({"message": index, "operation": "literal_table_not_applicable",
                            "reason": type(error).__name__ + ": " + str(error)[:160]})
            continue
        payload = json.loads(text_encoding)
        prospective_registry = dict(registry)
        reused = 0
        if reuse_records:
            payload, reused = intern_records(payload, prospective_registry)
        prospective_decoded = dict(decoded_records)
        decoded = decode_literal(payload, prospective_decoded)
        if ast.dump(decoded) != ast.dump(ast.parse(body, mode="eval").body):
            raise ValueError("Interning changed the original public observation values")
        compact = PREFIX + encoded(payload)
        if len(compact) >= len(body) or not schema_count:
            records.append({"message": index, "operation": "literal_table_not_shorter"})
            continue
        registry, decoded_records = prospective_registry, prospective_decoded
        projected[index]["content"] = text[:match.start("body")] + compact + text[match.end("body"):]
        records.append({
            "message": index, "operation": "lossless_public_literal_table",
            "source_sha256": fingerprint(body), "encoded_sha256": encoded_hash,
            "source_characters": len(body), "table_characters": len(compact),
            "schema_count": schema_count, "ast_roundtrip": True,
            "exact_records_reused": reused, "interned_sha256": fingerprint(payload),
        })
    return projected, records


def fit_native_capacity(messages, original_messages, count, *, context=131072, output=4096):
    if not 0 < output < context or len(messages) != len(original_messages):
        raise ValueError("Native capacity or message identity differs")
    projected = [dict(message) for message in messages]
    before = current = count(projected)
    if current + output <= context:
        return messages, {"before_tokens": before, "after_tokens": current, "excerpts": []}
    observation_indices = [
        index for index, message in enumerate(original_messages)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
        and OBSERVATION.fullmatch(message["content"]) is not None
    ]
    latest_observation = max(observation_indices, default=-1)
    candidates = []
    for index, message in enumerate(original_messages[:-2]):
        if index == latest_observation:
            continue
        text = message.get("content")
        if message.get("role") != "user" or not isinstance(text, str):
            continue
        match = OBSERVATION.fullmatch(text)
        if match is not None and len(match["body"]) > 6000:
            candidates.append((len(match["body"]), index, match))
    changes = []
    for _, index, match in sorted(candidates, reverse=True):
        body = match["body"]
        excerpt = (
            "[NATIVE CONTEXT CAPACITY EXCERPT: the middle of this earlier tool result is "
            "unavailable in this model input. The full original remains in the audit log. "
            "Re-read its native source if required; do not infer omitted facts. "
            "Original sha256=" + fingerprint(body) + "]\n"
            + body[:1200] + "\n[Middle omitted for native context capacity]\n" + body[-1200:]
        )
        text = original_messages[index]["content"]
        projected[index]["content"] = text[:match.start("body")] + excerpt + text[match.end("body"):]
        current = count(projected)
        changes.append({"message": index, "original_characters": len(body),
                        "retained_characters": 2400, "source_sha256": fingerprint(body),
                        "information_unavailable_in_actor_input": True})
        if current + output <= context:
            return projected, {"before_tokens": before, "after_tokens": current,
                               "excerpts": changes}
    raise ValueError("Protected native task/latest observations cannot fit the model context")


def project_with_tables(messages, *, native_catalogue, records):
    prior = project_public_messages(messages, native_catalogue=native_catalogue, records=records)
    projected, changes = table_observations(prior.messages)
    applied = any(row["operation"] == "lossless_public_literal_table" for row in changes)
    return Projection(projected if applied else prior.messages, {
        **prior.audit, "policy": POLICY, "applied": prior.audit["applied"] or applied,
        "changes": prior.audit["changes"] + changes,
        "projected_messages_sha256": fingerprint(projected),
        "projected_characters": sum(len(str(row.get("content", ""))) for row in projected),
        "literal_table_applied": applied, "public_values_removed": 0,
        "new_model_calls": 0,
    })


def capacity_projector(config, convert_message, request):
    def capacity_projection(messages, *, native_catalogue, records):
        wire = [convert_message(message) for message in messages]
        if any(not isinstance(message.get("content"), str) for message in wire):
            raise ValueError("This text-only adapter cannot compact multimodal messages")
        projection = project_with_tables(wire, native_catalogue=native_catalogue, records=records)

        def count(payload):
            result = request(
                f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",
                {"model": config["model"]["served_name"], "messages": payload,
                 "chat_template_kwargs": {"enable_thinking": False}},
            )
            if type(result.get("count")) is not int or result["count"] <= 0:
                raise ValueError("Native tokenizer returned invalid context usage")
            return result["count"]

        fitted, capacity = fit_native_capacity(
            projection.messages, wire, count, context=config["model"]["max_model_len"],
            output=config["model"]["max_tokens"],
        )
        native = [{**message, "content": projected["content"]}
                  for message, projected in zip(messages, fitted, strict=True)]
        if [convert_message(message) for message in native] != fitted:
            raise ValueError("Tokenizer and actual provider wire messages differ")
        return Projection(native, {
            **projection.audit, "native_capacity": capacity,
            "applied": projection.audit["applied"] or bool(capacity["excerpts"]),
            "source_native_messages_sha256": fingerprint(messages),
            "source_wire_messages_sha256": fingerprint(wire),
            "projected_messages_sha256": fingerprint(native),
            "projected_wire_messages_sha256": fingerprint(fitted),
            "projected_characters": sum(len(str(row.get("content", ""))) for row in fitted),
            "native_roles_preserved": True, "tokenizer_uses_native_wire_conversion": True,
            "actor_input_information_excerpted": bool(capacity["excerpts"]),
            "public_values_removed": None if capacity["excerpts"] else 0,
            "public_values_removed_from_native_log": 0,
            "full_native_history_preserved": True,
        })

    return capacity_projection


def install_adapter(config, directory, role, mode, skill):
    if (config.get("stage") != "train-candidate" or config.get("evaluation_split") != "train"
            or mode != "jev" or not config.get("receipt_execution")):
        raise RuntimeError("Public literal tables are admitted only to this TRAIN candidate")
    from phiagent.harness import gaia2_202060_progress as progress
    from scripts import run_gaia2_jev as base

    class CapacityAwareCalendarEngine(CompactCalendarReceiptEngine):
        def __init__(self, agent, provider, *, skill, record, execute_followups):
            converter = getattr(provider, "_convert_message_to_litellm_format", None)
            if not callable(converter):
                raise ValueError("Native actor wire converter is required for tokenization")
            super().__init__(agent, provider, skill=skill, record=record,
                             execute_followups=execute_followups,
                             projector=capacity_projector(config, converter, base.get_json),
                             policy=POLICY)

    progress.ReceiptEngine = CapacityAwareCalendarEngine
    progress.install_adapter(config, directory, role, mode, skill)
