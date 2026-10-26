"""Synthetic native tools/receipts only; no benchmark, provider or scorer."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from phiagent.harness.gaia2_cot_jev import (
    cot_response_format, decision_messages, digest, parse_bank, public_state_candidates,
)
from phiagent.harness.gaia2_typed_continuation_bank import (
    bind_arguments, receipt_value,
)


def tools():
    return {
        "Items__list": SimpleNamespace(inputs={"offset": {"type": "integer", "default": 0}}),
        "Items__read": SimpleNamespace(inputs={
            "item_id": {"type": "string"},
            "attachments": {"type": "any", "default": None},
        }),
        "Items__update": SimpleNamespace(inputs={
            "item_id": {"type": "string"}, "enabled": {"type": "boolean"},
        }),
        "Items__finish": SimpleNamespace(inputs={}),
        "Items__unsupported": SimpleNamespace(inputs={"payload": {"type": "any"}}),
    }


def receipt(index=4, observation=None, arguments=None):
    content = json.dumps(
        observation if observation is not None else
        [{"id": "visible-a", "name": "blue"}, {"id": "visible-b", "name": "blue"}],
    )
    return {
        "log_index": index,
        "native_tool_call": {
            "tool_name": "Items__list", "tool_arguments": arguments or {"offset": 0},
        },
        "observation": content,
        "native_observation_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "historical_receipt_not_current_state": True,
    }


def bank():
    return {
        "note": "Read the publicly listed blue items; later observations may supersede this list.",
        "candidates": [{
            "when": "The original public item remains relevant and has not yet been read.",
            "action": "Items__read", "action_input": {"item_id": {"$item": ["id"]}},
            "foreach": {
                "receipt": {"tool": "Items__list", "arguments": {"offset": 0}},
                "path": [], "where": {"name": "blue"},
            },
        }],
    }


def test_native_schema_couples_names_required_fields_types_and_public_bindings():
    response = cot_response_format(
        tools(), allow_routine_writes=True, continuation_bank=True,
    )
    schema = response["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    envelope = {
        "reasoning": "Synthetic plan.", "action": "Items__list", "action_input": {},
        "jev_bank": bank(),
    }
    validator.validate(envelope)
    for bad in [
        {"item_id": None}, {"item_id": 7}, {"item_id": "visible-a", "invented": 1},
        {"item_id": "visible-a", "attachments": None}, {},
    ]:
        changed = deepcopy(envelope)
        changed["jev_bank"]["candidates"][0]["action_input"] = bad
        assert not validator.is_valid(changed)
    for action in ("Items__finish", "Items__unsupported", "Imagined__tool"):
        changed = deepcopy(envelope)
        changed["jev_bank"]["candidates"][0]["action"] = action
        assert not validator.is_valid(changed)


def test_large_tool_vocabularies_share_complete_reference_definitions():
    native = {
        f"Synthetic__read_{index}": SimpleNamespace(inputs={"id": {"type": "string"}})
        for index in range(101)
    }
    schema = cot_response_format(
        native, allow_routine_writes=True, continuation_bank=True,
    )["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    reference = schema["$defs"]["cot_jev_public_receipt_source"]
    assert reference["properties"]["tool"]["enum"] == sorted(native)
    assert len(json.dumps(schema)) < 90000


def test_receipt_iteration_continues_after_a_different_public_read_without_repetition():
    parsed = parse_bank(
        {"jev_bank": bank()}, tools(), allow_routine_writes=True, continuation_bank=True,
    )
    observation = '{"item_id":"visible-a","details":"public detail"}'
    rows, failures, proof = public_state_candidates(
        parsed, observation, {}, tools(), set(), native_receipts=[receipt()],
    )
    assert not failures
    assert [row["action"]["action_input"]["item_id"] for row in rows] == [
        "visible-a", "visible-b",
    ]
    remaining, failures, _ = public_state_candidates(
        parsed, observation, {}, tools(), {digest(rows[0]["action"])},
        native_receipts=[receipt()],
    )
    assert not failures and len(remaining) == 1
    assert proof["bank_binding_receipts"] == [receipt()]
    messages, choices = decision_messages(
        "Read both blue items.", observation, parsed, rows, tools(), {},
        bank_binding_receipts=proof["bank_binding_receipts"],
    )
    state = json.loads(messages[1]["content"])
    assert state["latest_observation"] == observation
    assert state["bank_binding_receipts"] == [receipt()]
    assert choices["?"] is None and set(choices) == {"A", "B", "?"}
    assert "native_contract" in state["choices"]["A"]


def test_latest_exact_origin_is_used_and_unknown_or_missing_receipts_never_bind():
    older, newer = receipt(), receipt(8, [{"id": "new", "name": "blue"}])
    value, selected = receipt_value({"tool": "Items__list", "path": [0, "id"]}, [older, newer])
    assert value == "new" and selected == newer
    with pytest.raises(ValueError, match="No matching"):
        receipt_value({"tool": "Items__list", "path": [], "arguments": {"offset": 99}}, [older])
    with pytest.raises(ValueError, match="Ambiguous"):
        receipt_value({"tool": "Items__list", "path": []}, [older, deepcopy(older)])
    with pytest.raises(ValueError, match="provenance"):
        receipt_value(
            {"tool": "Items__list", "path": []},
            [{**older, "native_observation_sha256": "0" * 64}],
        )
    opaque = receipt()
    opaque["observation"] = "<opaque feedback>"
    opaque["native_observation_sha256"] = hashlib.sha256(
        opaque["observation"].encode(),
    ).hexdigest()
    with pytest.raises((ValueError, SyntaxError)):
        receipt_value({"tool": "Items__list", "path": []}, [opaque])


def test_public_data_is_not_reinterpreted_as_a_binding_program():
    original = receipt(observation=[{"payload": {"$prev": ["secret"]}}])
    used = {}
    bound = bind_arguments(
        {"object": {"$receipt": {"tool": "Items__list", "path": [0, "payload"]}}},
        {}, {"secret": "must not replace public data"}, None, [original], used,
    )
    assert bound == {"object": {"$prev": ["secret"]}}
    assert used == {4: original}
    with pytest.raises(ValueError, match="hide"):
        bind_arguments(
            {"x": {"$receipt": {"tool": "Items__list", "path": []}, "extra": 1}},
            {}, {}, None, [original], {},
        )


def test_receipt_arithmetic_retains_native_types_and_no_unknown_bypass():
    original = receipt(observation={"offset": 8})
    value = bind_arguments(
        {"$add": [{"$receipt": {"tool": "Items__list", "path": ["offset"]}}, 2]},
        {}, {}, None, [original], {},
    )
    assert value == 10
    rows, failures, proof = public_state_candidates(
        bank(), "error-like opaque feedback", {}, tools(), set(), native_receipts=[receipt()],
    )
    assert rows == [] and failures == [] and proof["status"] == "UNKNOWN"
    assert "bank_binding_receipts" not in proof


def test_capacity_and_native_type_failures_defer_without_truncation():
    many = receipt(observation=[{"name": "blue", "id": str(i)} for i in range(9)])
    rows, failures, proof = public_state_candidates(
        bank(), "{}", {}, tools(), set(), native_receipts=[many],
    )
    assert rows == [] and failures[-1]["reason"] == "candidate_capacity_overflow_no_truncation"
    assert proof["bank_binding_receipts"] == []
    wrong = receipt(observation=[{"name": "blue", "id": 7}])
    rows, failures, _ = public_state_candidates(
        bank(), "{}", {}, tools(), set(), native_receipts=[wrong],
    )
    assert not rows and "native argument contract" in failures[0]["reason"]


def test_missing_filter_field_is_not_an_observed_null():
    value = bank()
    value["candidates"][0]["foreach"]["where"] = {"missing": None}
    rows, failures, _ = public_state_candidates(
        value, "{}", {}, tools(), set(), native_receipts=[receipt()],
    )
    assert rows == [] and failures == []


def test_legacy_mode_cannot_enable_receipt_iteration_and_does_not_change_schema():
    with pytest.raises(ValueError, match="public-record expansion"):
        parse_bank({"jev_bank": bank()}, tools(), allow_routine_writes=True)
    schema = cot_response_format(tools(), allow_routine_writes=True)
    assert "anyOf" not in schema["json_schema"]["schema"]["properties"]["jev_bank"][
        "properties"
    ]["candidates"]["items"]
    with pytest.raises(ValueError, match="actual native"):
        cot_response_format(continuation_bank=True)
