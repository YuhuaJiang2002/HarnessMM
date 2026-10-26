import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from phiagent.learning import conditional_microplans as micro


FIRST = "select alpha"
NEXT = "select beta"
FEEDBACK = "ready alpha"
PLAN_MODES = ("guarded_plan", "unguarded_plan")


def plan(*, continuation=True, positives=None, negatives=None):
    return {
        "reasoning": "Choose the native action; the optional next action has a literal feedback guard.",
        "action": FIRST,
        "continuation": {
            "action": NEXT,
            "observation_contains_all": [FEEDBACK] if positives is None else positives,
            "observation_contains_none": ["blocked"] if negatives is None else negatives,
        } if continuation else None,
    }


def parse(value=None, *, allow=True, actions=None):
    return micro.parse_microplan(
        json.dumps(plan() if value is None else value),
        native_actions=[FIRST] if actions is None else actions, allow_continuation=allow,
    )


def installed(mode="guarded_plan", value=None, observation="initial"):
    control = micro.MicroplanState(mode)
    control.install(
        plan(continuation=mode != "structured_single") if value is None else value,
        current_observation=observation, current_actions=[FIRST],
    )
    return control


def ready(mode="guarded_plan", value=None, feedback=FEEDBACK):
    control = installed(mode, value)
    control.observe(FIRST, feedback, native_valid=True)
    return control


@pytest.mark.parametrize("allow", [False, True])
def test_response_schema_is_standard_closed_bounded_and_owned(allow):
    schema = micro.response_schema(allow)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == {"reasoning", "action", "continuation"}
    assert schema["properties"]["reasoning"]["maxLength"] == 6000
    assert schema["properties"]["reasoning"]["minLength"] == 1
    assert schema["properties"]["action"]["maxLength"] == 160
    if allow:
        alternatives = schema["properties"]["continuation"]["anyOf"]
        assert alternatives[0] == {"type": "null"}
        continuation = alternatives[1]
        assert continuation["type"] == "object"
        assert continuation["additionalProperties"] is False
        assert set(continuation["required"]) == set(continuation["properties"]) == {
            "action", "observation_contains_all", "observation_contains_none",
        }
        assert continuation["properties"]["action"]["maxLength"] == 160
        for key, minimum in (("observation_contains_all", 1), ("observation_contains_none", 0)):
            predicates = continuation["properties"][key]
            assert predicates["minItems"] == minimum
            assert predicates["maxItems"] == 4
            assert predicates["uniqueItems"] is True
            assert predicates["items"]["type"] == "string"
            assert predicates["items"]["minLength"] == 1
            assert predicates["items"]["maxLength"] == 160
    else:
        assert schema["properties"]["continuation"]["type"] == "null"
        assert schema["properties"]["continuation"]["description"]
    assert schema == json.loads(json.dumps(schema, allow_nan=False))
    schema["properties"]["action"]["maxLength"] = 1
    assert micro.response_schema(allow)["properties"]["action"]["maxLength"] == 160


def test_plan_modes_have_byte_identical_prompt_and_schema_contract():
    guarded = micro.prompt_contract("guarded_plan")
    unguarded = micro.prompt_contract("unguarded_plan")
    assert guarded.encode() == unguarded.encode()
    assert "observation_contains_all" in guarded
    assert "observation_contains_none" in guarded
    assert "requires new reasoning" in guarded
    assert "literal" in guarded and "null" in guarded
    assert "6000" in guarded and "160" in guarded
    assert "continuation" in micro.prompt_contract("structured_single")
    assert '"continuation" must be null' in micro.prompt_contract("structured_single")
    assert micro.prompt_contract("structured_single") != guarded


def without_descriptions(value):
    if isinstance(value, dict):
        return {key: without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


@pytest.mark.parametrize("allow,expected", [
    (False, "390e51986179e64280694ff424c27eaa717b23a97b557151590ee72cd0d30fd0"),
    (True, "8bd93c8e5423f28b1b09ac8cfee142bb3d1f9a0171a9bb59c3b6bb868aeef09e"),
])
def test_generation_revision_changes_only_schema_descriptions_not_original_constraints(allow, expected):
    constraints = without_descriptions(micro.response_schema(allow))
    canonical = json.dumps(constraints, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected
    assert constraints["properties"]["action"]["pattern"] == r"\S"
    if allow:
        predicates = constraints["properties"]["continuation"]["anyOf"][1]["properties"]
        assert predicates["observation_contains_all"]["uniqueItems"] is True
        assert predicates["observation_contains_none"]["uniqueItems"] is True


def test_generation_revision_leaves_parser_state_helpers_and_constants_unchanged():
    tree = ast.parse(Path(micro.__file__).read_bytes())
    unchanged = ast.Module(body=[
        node for node in tree.body
        if not isinstance(node, ast.FunctionDef) or node.name not in {"response_schema", "prompt_contract"}
    ], type_ignores=[])
    fingerprint = hashlib.sha256(ast.dump(unchanged, include_attributes=False).encode()).hexdigest()
    assert fingerprint == "532da4d08089f6cee172122d1367d8398e4d3556cc3e47ed4fbb9a7abe053498"


@pytest.mark.parametrize("mode", micro.MODES)
def test_json_output_contract_explicitly_overrides_only_earlier_output_tags(mode):
    prompt = micro.prompt_contract(mode)
    assert "JSON replaces earlier <think>/<action> output-tag instructions" in prompt
    assert "all task, observation and native-action information still applies" in prompt
    assert "No markdown, extra/duplicate keys" in prompt
    assert '"action" executes NOW' in prompt
    assert "complete current native command exactly" in prompt
    assert "every argument, case and whitespace" in prompt


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_prompt_makes_intermediate_guard_timing_complete_arguments_and_optional_planning_explicit(mode):
    prompt = micro.prompt_contract(mode)
    assert "Provide ONE CONDITIONAL next action whenever a simple intermediate observation justifies it" in prompt
    assert "use null if none can be specified or feedback requires new reasoning" in prompt
    assert 'BEFORE it executes, using the observation returned by the first "action"' in prompt
    assert "the INTERMEDIATE state, NEVER the continuation result" in prompt
    assert "complete native syntax with already bound entity arguments" in prompt
    assert "Never shorten commands or guess unknown entity identities" in prompt
    assert "Guards and native membership still gate execution" in prompt
    assert len(prompt) < 1592
    assert prompt.count('{"reasoning":') == 1


def test_schema_metadata_distinguishes_the_actor_action_intermediate_guard_and_full_next_command():
    schema = micro.response_schema(True)
    properties = schema["properties"]
    assert "Execute NOW" in properties["action"]["description"]
    assert "every bound argument" in properties["action"]["description"]
    continuation = properties["continuation"]["anyOf"][1]
    assert "feedback from the first action BEFORE this continuation" in continuation["description"]
    assert "never its result" in continuation["description"]
    assert "Complete next native command" in continuation["properties"]["action"]["description"]
    assert "all already bound entity arguments" in continuation["properties"]["action"]["description"]
    positive = continuation["properties"]["observation_contains_all"]["description"]
    negative = continuation["properties"]["observation_contains_none"]["description"]
    assert "intermediate observation returned by the first action" in positive
    assert "same intermediate observation, not future feedback" in negative
    continuation["description"] = "mutated caller metadata"
    assert micro.response_schema(True)["properties"]["continuation"]["anyOf"][1]["description"] != (
        "mutated caller metadata"
    )


def schematic_example():
    prompt = micro.prompt_contract("guarded_plan")
    assert "invented commands, NOT valid task actions; DO NOT copy them" in prompt
    return json.loads(prompt[prompt.index('{"reasoning":'):])


def test_schematic_example_is_domain_neutral_and_demonstrates_first_action_feedback_not_second_result():
    value = schematic_example()
    assert value["action"] == "zorp arg7"
    assert value["continuation"]["action"] == "nuv arg7 arg2"
    assert value["continuation"]["observation_contains_all"] == ["ZORP_READY"]
    assert value["continuation"]["observation_contains_none"] == []
    for actual_observation, commit in (("ZORP_READY", True), ("NUV_DONE", False)):
        control = micro.MicroplanState("guarded_plan")
        parsed = micro.parse_microplan(
            json.dumps(value), native_actions=[value["action"]], allow_continuation=True,
        )
        control.install(parsed, current_observation="initial", current_actions=[value["action"]])
        control.observe(value["action"], actual_observation, native_valid=True)
        result = control.decide(actual_observation, [value["continuation"]["action"]])
        assert result["commit"] is commit
        assert result["guard_checked"] is True


def test_schematic_commands_are_not_special_native_rules_and_incomplete_next_commands_are_not_repaired():
    value = schematic_example()
    with pytest.raises(ValueError, match="First action"):
        parse(value, actions=[FIRST])
    value["action"] = FIRST
    full_next_command = value["continuation"]["action"]
    value["continuation"]["action"] = "nuv arg7"
    parsed = parse(value, actions=[FIRST])
    control = installed(value=parsed)
    control.observe(FIRST, "ZORP_READY", native_valid=True)
    result = control.decide("ZORP_READY", [full_next_command])
    assert not result["commit"]
    assert result["candidate_action"] == "nuv arg7"
    assert result["reason"] == "continuation_not_native"


def test_earlier_think_action_tag_format_is_not_implicitly_parsed_or_repaired():
    with pytest.raises(ValueError, match="Expected exactly one JSON object"):
        micro.parse_microplan(
            f"<think>why</think><action>{FIRST}</action>",
            native_actions=[FIRST], allow_continuation=True,
        )


def test_parser_preserves_action_and_predicate_text_and_allows_future_native_action():
    value = plan(positives=["ready\t alpha"], negatives=[])
    value["action"] = "  select alpha  "
    parsed = parse(value, actions=[value["action"]])
    assert parsed == value
    assert parsed["action"] == "  select alpha  "
    assert parsed["continuation"]["action"] == NEXT
    assert NEXT not in [value["action"]]
    assert parsed["continuation"]["observation_contains_all"] == ["ready\t alpha"]
    assert micro.parse_microplan(
        "\n " + json.dumps(value) + " \t\n",
        native_actions=[value["action"]], allow_continuation=True,
    ) == value


def test_string_limits_and_predicate_counts_accept_the_exact_upper_boundary():
    value = plan(positives=[str(index) * 160 for index in range(4)], negatives=[])
    value["reasoning"] = "r" * 6000
    value["action"] = "a" * 160
    value["continuation"]["action"] = "b" * 160
    assert parse(value, actions=[value["action"]]) == value
    value["continuation"]["observation_contains_none"] = [letter * 160 for letter in "wxyz"]
    assert parse(value, actions=[value["action"]]) == value


@pytest.mark.parametrize("text", [
    "", " \t\n", "not JSON", "null", "true", "[]", "[{}]", '"a string"',
    "```json\n{}\n```", '{"action":"x",}', '{} {}', '{}\ntrailing text',
    json.dumps(plan()) + "\n" + json.dumps(plan()),
])
def test_parser_rejects_empty_wrapped_malformed_or_multiple_model_objects(text):
    with pytest.raises(ValueError):
        micro.parse_microplan(text, native_actions=[FIRST], allow_continuation=True)


@pytest.mark.parametrize("text", [
    '{"reasoning":"why","action":"select alpha","action":"select beta","continuation":null}',
    '{"reasoning":"why","action":"select alpha","act\\u0069on":"select alpha","continuation":null}',
    '{"reasoning":"why","action":"select alpha","continuation":null,"continuation":null}',
    '{"reasoning":"why","action":"select alpha","continuation":{"action":"select beta",'
    '"observation_contains_all":["ready"],"observation_contains_none":[],'
    '"observation_contains_none":[]}}',
])
def test_duplicate_json_keys_are_rejected_at_every_level(text):
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        micro.parse_microplan(text, native_actions=[FIRST], allow_continuation=True)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e10000", "-1e10000"])
@pytest.mark.parametrize("nested", [False, True])
def test_nonfinite_json_is_rejected_even_when_nested_or_in_an_extra_field(constant, nested):
    text = (
        '{"reasoning":"why","action":"select alpha","continuation":null,'
        f'"extra":{constant}' + "}"
    )
    if nested:
        text = '{"reasoning":"why","action":"select alpha","continuation":{"extra":[' + constant + "]}}"
    with pytest.raises(ValueError, match="Non-finite"):
        micro.parse_microplan(text, native_actions=[FIRST], allow_continuation=True)


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("reasoning"),
    lambda value: value.pop("action"),
    lambda value: value.pop("continuation"),
    lambda value: value.update(reasoning=True),
    lambda value: value.update(reasoning=None),
    lambda value: value.update(reasoning=""),
    lambda value: value.update(reasoning=" \n\t"),
    lambda value: value.update(reasoning="x" * 6001),
    lambda value: value.update(action=True),
    lambda value: value.update(action=3),
    lambda value: value.update(action=""),
    lambda value: value.update(action="a" * 161),
    lambda value: value.update(continuation=False),
    lambda value: value.update(continuation=[]),
    lambda value: value.update(continuation=[deepcopy(value["continuation"])] * 2),
    lambda value: value.update(confidence=1.0),
    lambda value: value.update(tool_calls=[]),
    lambda value: value.update(code="executable content is not a plan field"),
    lambda value: value["continuation"].update(action=""),
    lambda value: value["continuation"].update(action="x" * 161),
    lambda value: value["continuation"].update(action=True),
    lambda value: value["continuation"].pop("observation_contains_none"),
    lambda value: value["continuation"].update(continuation=None),
    lambda value: value["continuation"].update(replan=True),
    lambda value: value["continuation"].update(observation_contains_all=[]),
    lambda value: value["continuation"].update(observation_contains_all="ready"),
    lambda value: value["continuation"].update(observation_contains_none=False),
    lambda value: value["continuation"].update(observation_contains_none=None),
    lambda value: value["continuation"].update(observation_contains_all=[True]),
    lambda value: value["continuation"].update(observation_contains_none=[3]),
    lambda value: value["continuation"].update(observation_contains_all=[""]),
    lambda value: value["continuation"].update(observation_contains_none=[" \n"]),
    lambda value: value["continuation"].update(observation_contains_all=["x" * 161]),
    lambda value: value["continuation"].update(observation_contains_none=["x" * 161]),
    lambda value: value["continuation"].update(observation_contains_all=list("abcde")),
    lambda value: value["continuation"].update(observation_contains_none=list("vwxyz")),
])
def test_invalid_fields_types_limits_and_ambiguous_plan_shapes_raise_explicitly(mutation):
    value = plan()
    mutation(value)
    with pytest.raises(ValueError):
        parse(value)
    with pytest.raises(ValueError):
        installed(value=value)


@pytest.mark.parametrize("positives,negatives,reason", [
    (["ready", "ready"], [], "duplicate"),
    (["ready alpha", "ready\t alpha"], [], "duplicate"),
    (["ready"], ["blocked", "blocked"], "duplicate"),
    (["ready"], ["blocked alpha", " blocked  alpha "], "duplicate"),
    (["ready"], ["ready"], "Contradictory"),
    (["ready alpha"], ["alpha"], "Contradictory"),
    (["ready\nalpha"], ["ready alpha"], "Contradictory"),
])
def test_duplicate_normalized_predicates_and_substring_contradictions_are_errors(
    positives, negatives, reason,
):
    with pytest.raises(ValueError, match=reason):
        parse(plan(positives=positives, negatives=negatives))


def test_predicates_remain_case_sensitive_even_for_contradiction_validation():
    value = plan(positives=["READY"], negatives=["ready"])
    assert parse(value) == value
    control = ready(value=value, feedback="READY")
    decision = control.decide("READY", [NEXT])
    assert decision["commit"] and decision["guard_passed"]


@pytest.mark.parametrize("actions", [[], ["Select alpha"], ["select alpha "], ["different"]])
def test_first_action_requires_exact_native_membership_without_repair(actions):
    with pytest.raises(ValueError, match="First action"):
        parse(actions=actions)


@pytest.mark.parametrize("actions", [None, True, FIRST, [True], [None], [""], [FIRST, FIRST]])
def test_native_action_container_is_strict(actions):
    with pytest.raises(ValueError, match="native action"):
        micro.parse_microplan(json.dumps(plan()), native_actions=actions, allow_continuation=True)


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_allow_continuation_is_a_bool_not_an_integer_or_truthy_value(value):
    with pytest.raises(ValueError, match="allow_continuation must be a bool"):
        micro.response_schema(value)
    with pytest.raises(ValueError, match="allow_continuation must be a bool"):
        micro.parse_microplan(json.dumps(plan()), native_actions=[FIRST], allow_continuation=value)


@pytest.mark.parametrize("text", [None, True, 1, b'{"action":"select alpha"}'])
def test_model_output_must_be_a_string(text):
    with pytest.raises(ValueError, match="microplan JSON text"):
        micro.parse_microplan(text, native_actions=[FIRST], allow_continuation=True)


def test_invalid_unicode_surrogates_are_not_a_success_shaped_plan():
    value = plan()
    value["reasoning"] = "\ud800"
    with pytest.raises(ValueError, match="surrogate"):
        parse(value)


@pytest.mark.parametrize("mode", [True, None, 1, [], "actor", "guarded", ""])
def test_modes_are_exact_and_actor_generation_stays_outside_this_module(mode):
    with pytest.raises(ValueError, match="mode"):
        micro.MicroplanState(mode)
    with pytest.raises(ValueError, match="mode"):
        micro.prompt_contract(mode)


@pytest.mark.parametrize("mode", micro.MODES)
def test_single_or_null_continuation_requires_another_actor_without_a_local_step(mode):
    value = plan(continuation=False)
    assert parse(value, allow=mode != "structured_single") == value
    control = installed(mode, value)
    observed = control.observe(FIRST, FEEDBACK, native_valid=True)
    assert observed["reason"] == "actor_action_observed_no_continuation"
    assert observed["actor_required"]
    assert observed["completed_actions"] == 1
    result = control.decide(FEEDBACK, [NEXT])
    assert not result["commit"] and result["action"] is None
    assert result["reason"] == "actor_boundary_required"
    assert not result["guard_checked"] and result["guard_passed"] is None


def test_structured_single_rejects_any_continuation_at_parse_and_install():
    with pytest.raises(ValueError, match="continuation=null"):
        parse(allow=False)
    with pytest.raises(ValueError, match="continuation=null"):
        installed("structured_single", plan())


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_continuation_cannot_be_decided_or_overwritten_before_first_action_feedback(mode):
    control = installed(mode, observation=FEEDBACK)
    with pytest.raises(ValueError, match="pending"):
        control.decide(FEEDBACK, [FIRST, NEXT])
    with pytest.raises(ValueError, match="overwrite"):
        control.install(plan(), current_observation=FEEDBACK, current_actions=[FIRST, NEXT])
    feedback = control.observe(FIRST, FEEDBACK, native_valid=True)
    assert feedback["phase"] == "ready_continuation"
    assert feedback["completed_actions"] == 1
    assert not feedback["actor_required"]
    with pytest.raises(ValueError, match="overwrite"):
        control.install(plan(), current_observation=FEEDBACK, current_actions=[FIRST])
    with pytest.raises(ValueError, match="No pending"):
        control.observe(FIRST, FEEDBACK, native_valid=True)
    assert control.decide(FEEDBACK, [NEXT])["commit"]


def test_guard_uses_actual_feedback_not_initial_observation_or_model_reasoning():
    value = plan()
    value["reasoning"] = FEEDBACK
    control = installed(value=value, observation=FEEDBACK)
    control.observe(FIRST, "nothing matching", native_valid=True)
    decision = control.decide("nothing matching", [NEXT])
    assert not decision["commit"]
    assert decision["guard_checked"]
    assert decision["guard_passed"] is False
    assert decision["reason"] == "observation_guard_failed"
    assert decision["missing_positive_predicates"] == [FEEDBACK]
    assert decision["completed_actions"] == 1


def test_whitespace_is_normalized_only_for_literal_contains_checks():
    value = plan(positives=["ready\talpha", "complete result"], negatives=["bad\n signal"])
    observed = "  ready \n alpha\tand  complete\nresult  "
    control = ready(value=value, feedback=observed)
    decision = control.decide(observed, [NEXT])
    assert decision["commit"]
    assert decision["guard_checked"] and decision["guard_passed"]
    assert decision["action"] == NEXT
    assert decision["missing_positive_predicates"] == decision["present_negative_predicates"] == []
    control = ready(value=value, feedback=observed + " bad \t signal")
    denied = control.decide(observed + " bad \t signal", [NEXT])
    assert not denied["commit"]
    assert denied["present_negative_predicates"] == ["bad\n signal"]


def test_substring_predicates_are_literal_not_regex_or_executable_expressions():
    value = plan(positives=["[a-z]+", "__call__()"], negatives=[".*"])
    feedback = "observed [a-z]+ __call__()"
    result = ready(value=value, feedback=feedback).decide(feedback, [NEXT])
    assert result["commit"]
    absent = ready(value=value, feedback="abc __call__()").decide("abc __call__()", [NEXT])
    assert not absent["commit"]
    assert absent["missing_positive_predicates"] == ["[a-z]+"]


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_stale_or_substituted_feedback_cannot_satisfy_a_guard(mode):
    control = ready(mode, feedback="no match")
    with pytest.raises(ValueError, match="recorded first-action feedback"):
        control.decide(FEEDBACK, [NEXT])
    with pytest.raises(ValueError, match="recorded first-action feedback"):
        control.decide(" no match ", [NEXT])
    decision = control.decide("no match", [NEXT])
    assert decision["commit"] is (mode == "unguarded_plan")
    assert decision["guard_checked"] is (mode == "guarded_plan")


@pytest.mark.parametrize("mode", PLAN_MODES)
@pytest.mark.parametrize("native", [[], ["SELECT BETA"], ["other native action"]])
def test_no_guard_ablation_still_requires_exact_native_membership_and_consumes_failure(mode, native):
    control = ready(mode)
    result = control.decide(FEEDBACK, native)
    assert result["reason"] == "continuation_not_native"
    assert result["native_member"] is False
    assert result["action"] is None
    assert result["candidate_action"] == NEXT
    assert result["guard_checked"] is False
    assert result["guard_passed"] is None
    assert result["actor_required"]
    assert result["phase"] == "awaiting_plan"
    assert result["completed_actions"] == 1
    with pytest.raises(ValueError, match="pending"):
        control.decide(FEEDBACK, [NEXT])
    with pytest.raises(ValueError, match="No pending"):
        control.observe(NEXT, "not executed", native_valid=True)
    control.install(plan(continuation=False), current_observation=FEEDBACK, current_actions=[FIRST])
    assert control.observe(FIRST, "new actual feedback", native_valid=True)["actor_required"]


def test_unguarded_mode_changes_only_runtime_predicate_enforcement():
    value = plan()
    feedback = "not ready, blocked"
    guarded = ready("guarded_plan", value, feedback)
    unguarded = ready("unguarded_plan", value, feedback)
    a = guarded.decide(feedback, [NEXT])
    b = unguarded.decide(feedback, [NEXT])
    assert a["plan_sha256"] == b["plan_sha256"]
    assert not a["commit"] and a["guard_checked"] and a["guard_passed"] is False
    assert a["present_negative_predicates"] == ["blocked"]
    assert b["commit"] and not b["guard_checked"] and b["guard_passed"] is None
    assert b["reason"] == "guard_enforcement_disabled"
    assert b["native_member"] is True
    assert unguarded.observe(NEXT, "any actual result", native_valid=True)["actor_required"]


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_both_plan_modes_require_the_same_nonempty_positive_guard_schema(mode):
    value = plan(positives=[])
    with pytest.raises(ValueError, match="1 to 4"):
        parse(value)
    with pytest.raises(ValueError, match="1 to 4"):
        installed(mode, value)


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_exactly_one_local_continuation_then_mandatory_actor_boundary(mode):
    control = micro.MicroplanState(mode)
    initial = control.decide("initial", [FIRST])
    assert initial["actor_required"]
    assert initial["reason"] == "actor_required"
    with pytest.raises(ValueError, match="pending"):
        control.decide("initial", [FIRST])
    control.install(plan(), current_observation="initial", current_actions=[FIRST])
    control.observe(FIRST, FEEDBACK, native_valid=True)
    decision = control.decide(FEEDBACK, [NEXT])
    assert decision["commit"]
    assert decision["completed_actions"] == 1
    with pytest.raises(ValueError, match="pending"):
        control.decide(FEEDBACK, [NEXT])
    with pytest.raises(ValueError, match="overwrite"):
        control.install(plan(), current_observation=FEEDBACK, current_actions=[FIRST])
    feedback = control.observe(NEXT, "final feedback", native_valid=True)
    assert feedback["completed_actions"] == 2
    assert feedback["actor_required"]
    assert feedback["reason"] == "continuation_completed_actor_required"
    assert not feedback["continuation_queued"]
    with pytest.raises(ValueError, match="No pending"):
        control.observe(NEXT, "duplicate feedback", native_valid=True)
    boundary = control.decide("final feedback", [NEXT, FIRST])
    assert not boundary["commit"]
    assert boundary["reason"] == "actor_boundary_required"
    assert boundary["boundary_reason"] == "continuation_completed_actor_required"
    control.install(plan(), current_observation="final feedback", current_actions=[FIRST])
    assert control.observe(FIRST, FEEDBACK, native_valid=True)["completed_actions"] == 3
    assert control.decide(FEEDBACK, [NEXT])["commit"]


@pytest.mark.parametrize("phase", ["first", "continuation"])
def test_actual_executed_action_must_match_pending_action_and_cannot_be_repaired(phase):
    control = installed()
    expected, feedback = FIRST, FEEDBACK
    if phase == "continuation":
        control.observe(FIRST, FEEDBACK, native_valid=True)
        control.decide(FEEDBACK, [NEXT])
        expected, feedback = NEXT, "actual final"
    for wrong in ("", expected.upper(), expected + " ", "actor substitute"):
        with pytest.raises(ValueError):
            control.observe(wrong, feedback, native_valid=True)
    assert control.observe(expected, feedback, native_valid=True)["action"] == expected


@pytest.mark.parametrize("native_valid", [0, 1, None, "true", [], {}])
def test_observe_rejects_nonbool_flags_without_consuming_actual_action(native_valid):
    control = installed()
    with pytest.raises(ValueError, match="native_valid must be a bool"):
        control.observe(FIRST, FEEDBACK, native_valid=native_valid)
    assert control.observe(FIRST, FEEDBACK, native_valid=True)["completed_actions"] == 1


@pytest.mark.parametrize("phase", ["first", "continuation"])
def test_non_native_actual_feedback_is_logged_and_clears_the_plan(phase):
    control = installed()
    action = FIRST
    if phase == "continuation":
        control.observe(FIRST, FEEDBACK, native_valid=True)
        control.decide(FEEDBACK, [NEXT])
        action = NEXT
    result = control.observe(action, "native rejection", native_valid=False)
    assert result["native_valid"] is False
    assert result["actor_required"] and not result["continuation_queued"]
    assert result["reason"] == ("actor_action_not_native" if phase == "first"
                                else "continuation_action_not_native")
    assert not control.decide("native rejection", [FIRST, NEXT])["commit"]


def test_parsed_plan_and_installed_plan_do_not_alias_caller_containers():
    value = plan()
    parsed = parse(value)
    value["continuation"]["observation_contains_all"].clear()
    assert parsed["continuation"]["observation_contains_all"] == [FEEDBACK]
    control = installed(value=parsed)
    plan_hash = control._plan_hash
    parsed["action"] = "changed caller action"
    parsed["reasoning"] = "changed caller reasoning"
    parsed["continuation"]["action"] = "changed caller continuation"
    parsed["continuation"]["observation_contains_all"][:] = ["unrelated guard"]
    assert control.observe(FIRST, FEEDBACK, native_valid=True)["reason"] == "continuation_ready"
    result = control.decide(FEEDBACK, [NEXT])
    assert result["commit"] and result["action"] == NEXT
    assert result["plan_sha256"] == plan_hash
    result["action"] = "mutated return"
    assert control.observe(NEXT, "actual feedback", native_valid=True)["actor_required"]


def test_invalid_install_does_not_replace_an_empty_state_or_its_actor_boundary():
    control = micro.MicroplanState("guarded_plan")
    control.decide("initial", [FIRST])
    value = plan()
    value["reasoning"] = ""
    with pytest.raises(ValueError, match="reasoning"):
        control.install(value, current_observation="initial", current_actions=[FIRST])
    assert control._phase == "awaiting_plan"
    assert control._plan is None and control._plan_hash is None
    result = control.install(plan(), current_observation="initial", current_actions=[FIRST])
    assert result["phase"] == "awaiting_first_action"
    assert result["completed_actions"] == 0


@pytest.mark.parametrize("mode", micro.MODES)
def test_uninstalled_or_double_observations_are_errors(mode):
    control = micro.MicroplanState(mode)
    with pytest.raises(ValueError, match="No pending"):
        control.observe(FIRST, FEEDBACK, native_valid=True)
    control.decide("initial", [FIRST])
    with pytest.raises(ValueError, match="No pending"):
        control.observe(FIRST, FEEDBACK, native_valid=True)


def test_empty_observations_are_real_feedback_not_repaired():
    control = installed(observation="")
    observed = control.observe(FIRST, "", native_valid=True)
    assert observed["observation"] == ""
    decision = control.decide("", [NEXT])
    assert not decision["commit"]
    assert decision["reason"] == "observation_guard_failed"
    assert decision["missing_positive_predicates"] == [FEEDBACK]


def test_guard_failure_is_consumed_and_a_new_actor_plan_does_not_inherit_the_old_guard():
    control = ready(feedback="blocked")
    failed = control.decide("blocked", [NEXT])
    assert failed["guard_checked"] and failed["guard_passed"] is False
    assert failed["actor_required"]
    assert failed["boundary_reason"] == "observation_guard_failed"
    assert control._plan is None
    with pytest.raises(ValueError, match="pending"):
        control.decide(FEEDBACK, [NEXT])
    with pytest.raises(ValueError, match="No pending"):
        control.observe(NEXT, FEEDBACK, native_valid=True)
    replacement = plan(positives=["new feedback"], negatives=[])
    result = control.install(replacement, current_observation="blocked", current_actions=[FIRST])
    assert result["plan_sha256"] != failed["plan_sha256"]
    assert result["boundary_reason"] is None
    control.observe(FIRST, "new feedback blocked", native_valid=True)
    assert control.decide("new feedback blocked", [NEXT])["commit"]


def test_install_revalidates_current_native_membership_even_for_an_already_parsed_plan():
    parsed = parse()
    control = micro.MicroplanState("guarded_plan")
    with pytest.raises(ValueError, match="First action"):
        control.install(parsed, current_observation="changed context", current_actions=["other"])
    assert control._phase == "empty"
    assert control._plan is None
    assert control.install(
        parsed, current_observation="actual context", current_actions=[FIRST],
    )["action"] == FIRST


@pytest.mark.parametrize("observation", [None, True, 1, [], {}])
def test_observation_types_are_strict_and_invalid_calls_leave_the_phase_unchanged(observation):
    control = micro.MicroplanState("guarded_plan")
    with pytest.raises(ValueError, match="current_observation"):
        control.install(plan(), current_observation=observation, current_actions=[FIRST])
    assert control._phase == "empty"
    control.install(plan(), current_observation="", current_actions=[FIRST])
    with pytest.raises(ValueError, match="observed feedback"):
        control.observe(FIRST, observation, native_valid=True)
    assert control._completed == 0
    control.observe(FIRST, FEEDBACK, native_valid=True)
    with pytest.raises(ValueError, match="decision observation"):
        control.decide(observation, [NEXT])
    assert control.decide(FEEDBACK, [NEXT])["commit"]


def test_action_lengths_are_python_character_counts_not_utf8_byte_counts():
    value = plan(continuation=False)
    value["action"] = "\u03b1" * 160
    value["reasoning"] = "\u03b2" * 6000
    assert parse(value, actions=[value["action"]], allow=False) == value
    value["action"] += "\u03b1"
    with pytest.raises(ValueError, match="160 characters"):
        parse(value, actions=[value["action"]], allow=False)


_AUDIT_KEYS = {
    "schema_version", "mode", "phase", "plan_sha256", "completed_actions",
    "actor_required", "boundary_reason",
}
_DECISION_KEYS = _AUDIT_KEYS | {
    "commit", "action", "candidate_action", "reason", "guard_checked", "guard_passed",
    "native_member", "missing_positive_predicates", "present_negative_predicates",
}


@pytest.mark.parametrize("mode", micro.MODES)
def test_exact_public_output_keys_serialization_and_stable_plan_hash(mode):
    control = micro.MicroplanState(mode)
    initial = control.decide("initial", [FIRST])
    assert set(initial) == _DECISION_KEYS
    value = plan(continuation=mode != "structured_single")
    output = control.install(value, current_observation="initial", current_actions=[FIRST])
    assert set(output) == _AUDIT_KEYS | {"reason", "action", "continuation_queued"}
    assert output["schema_version"] == micro.STATE_SCHEMA
    assert len(output["plan_sha256"]) == 64
    assert set(output["plan_sha256"]) <= set("0123456789abcdef")
    reordered = dict(reversed(list(deepcopy(value).items())))
    other = micro.MicroplanState(mode).install(
        reordered, current_observation="different initial text", current_actions=[FIRST],
    )
    assert output["plan_sha256"] == other["plan_sha256"]
    feedback = control.observe(FIRST, FEEDBACK, native_valid=True)
    assert set(feedback) == _AUDIT_KEYS | {
        "reason", "action", "observation", "native_valid", "action_role", "continuation_queued",
    }
    decision = control.decide(FEEDBACK, [NEXT])
    assert set(decision) == _DECISION_KEYS
    for result in (initial, output, feedback, decision):
        assert result == json.loads(json.dumps(result, allow_nan=False))
    assert decision["commit"] is (mode != "structured_single")
