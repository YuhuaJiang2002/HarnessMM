from copy import deepcopy
import hashlib
import json

import pytest

from phiagent.learning import grounded_continuations as grounded
from phiagent.learning.counterfactual_invocation import build_policy
from phiagent.learning.fragment_contracts import compile_library


def state(task, observation, actions, history=None):
    return {"task": task, "observation": observation, "actions": list(actions),
            "history": [] if history is None else history}


def source(name, group, *, kind="orb", number="1", task=None, actions=None,
           before=None, after=None, pre_actions=(), next_actions=(), family="opaque-family"):
    task = f"inspect {kind}" if task is None else task
    actions = list(actions or (f"touch {kind} {number}", f"inspect {kind} {number}"))
    path = [
        before if before is not None else f"closed {kind} {number}",
        after if after is not None else f"{kind} {number} visible",
    ] + [f"stage{index} {kind} {number}" for index in range(2, len(actions) + 1)]
    ids = [f"{name}-demo-{index}" for index in range(len(actions))]
    task_id = f"{name}-task"
    demos, transitions = [], {}
    for index, action in enumerate(actions):
        extras = pre_actions if index == 0 else next_actions if index == 1 else ()
        demos.append({
            "example_id": ids[index], "task_id": task_id, "group_id": group, "action": action,
            "public_state": state(task, path[index], list(dict.fromkeys([action, *extras]))),
        })
        transitions[ids[index]] = {
            "post_observation": path[index + 1], "pre_action_history": actions[:index],
            "task_family": family,
        }
    program = {
        "program_id": name, "actions": actions,
        "examples": [{"task_id": task_id, "group_id": group, "start_turn": 0,
                      "demonstration_ids": ids, "task_family": family}],
    }
    return program, demos, transitions


def prepared(*sources):
    programs, demos, transitions = [], [], {}
    for program, examples, feedback in sources or (
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
    ):
        programs.append(program)
        demos.extend(examples)
        transitions.update(feedback)
    library = compile_library(programs, demos, transitions, [], minimum_groups=1)
    return library, build_policy(demos)


def controller(library, policy, mode="grounded_continuation", **kwargs):
    return grounded.GroundedContinuationController(
        library, policy, mode=mode,
        **{"group_id": "target", "task_family": "opaque-family", **kwargs},
    )


def query(kind="tile", number="3", *, task=None, action=None, native=None, observation=None):
    task = f"inspect {kind}" if task is None else task
    action = f"touch {kind} {number}" if action is None else action
    before = state(task, f"closed {kind} {number}", [action])
    observation = f"{kind} {number} visible" if observation is None else observation
    after = state(task, observation, [f"inspect {kind} {number}"] if native is None else native,
                  [{"action": action, "observation": observation}])
    return before, action, after


def anchor(control, before, action, after):
    result = control.decide(before)
    assert not result["decision"]["commit"]
    assert result["proposal"]["action"] is None
    feedback = control.observe(action, after["observation"], compiled=False)
    assert feedback["reason"] == "actor_anchor_saved"
    return control.decide(after)


@pytest.mark.parametrize("mode", grounded.MODES)
def test_entity_and_instance_renaming_transfer_with_two_independent_groups(mode):
    library, policy = prepared()
    assert all(len(program["source_groups"]) == 1 for program in library["programs"])
    before, action, after = query()
    control = controller(library, policy, mode)
    result = anchor(control, before, action, after)
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == "inspect tile 3"
    assert result["decision"]["support_groups"] == ["g0", "g1"]
    assert result["decision"]["source_count"] == 2
    assert result["decision"]["goal_roles"] == ["tile"]
    assert all(row["similarity"] == pytest.approx(1.0) for row in result["decision"]["candidates"])
    bindings = [row["bindings"][0] for row in result["decision"]["candidates"]]
    assert {tuple(row["source_entity"]) for row in bindings} == {("orb", "1"), ("tile", "2")}
    assert all(row["current_entity"] == ["tile", "3"] for row in bindings)
    assert all(row["origin"] == "actor_argument" and row["goal_role"] == 0 for row in bindings)


@pytest.mark.parametrize("source_command,current_command,accepted", [
    ("compare orb 1 with orb 1", "compare tile 3 with tile 3", True),
    ("compare orb 1 with orb 2", "compare tile 3 with tile 4", True),
    ("compare orb 1 with orb 1", "compare tile 3 with tile 4", False),
    ("compare orb 1 with orb 2", "compare tile 3 with tile 3", False),
])
def test_actor_argument_bijection_preserves_same_and_different_instances(
    source_command, current_command, accepted,
):
    library, policy = prepared(*[
        source(f"s{index}", f"g{index}", actions=(source_command, "inspect orb 1"),
               pre_actions=("catalog tile 9",))
        for index in range(2)
    ])
    before, action, after = query(action=current_command)
    result = anchor(controller(library, policy), before, action, after)
    assert result["decision"]["commit"] is accepted
    if accepted:
        assert result["proposal"]["action"] == "inspect tile 3"
        for candidate in result["decision"]["candidates"]:
            pairs = candidate["bindings"]
            assert len({tuple(row["source_entity"]) for row in pairs}) == len(pairs)
            assert len({tuple(row["current_entity"]) for row in pairs}) == len(pairs)
    else:
        assert result["decision"]["reason"] == "no_source_for_anchor_shape"


def test_goal_role_order_swaps_are_not_conflated():
    library, policy = prepared(
        source("a", "g0", task="inspect orb beside rack",
               actions=("touch orb 1 near rack 1", "inspect orb 1")),
        source("b", "g1", kind="tile", task="inspect tile beside tray",
               actions=("touch tile 2 near tray 2", "inspect tile 2")),
    )
    before, action, after = query(task="inspect tile beside tray", action="touch tile 3 near tray 4")
    result = anchor(controller(library, policy), before, action, after)
    assert result["decision"]["commit"]
    assert result["decision"]["goal_roles"] == ["tile", "tray"]
    swapped_before, swapped_action, swapped_after = query(
        task="inspect tray beside tile", action="touch tile 3 near tray 4",
    )
    swapped = anchor(controller(library, policy), swapped_before, swapped_action, swapped_after)
    assert not swapped["decision"]["commit"]
    assert swapped["decision"]["goal_roles"] == ["tray", "tile"]
    assert swapped["decision"]["anchor_shape_id"] != result["decision"]["anchor_shape_id"]


@pytest.mark.parametrize("task", ["inspect orb beside orb", "inspect orb 1 beside orb 2"])
def test_ambiguous_repeated_same_type_goal_roles_abstain(task):
    library, policy = prepared(
        source("a", "g0", task=task), source("b", "g1", task=task),
    )
    control = controller(library, policy)
    before, action, after = query(kind="orb", task=task)
    result = anchor(control, before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "ambiguous_same_type_goal_roles"
    assert result["decision"]["source_bank_reasons"]["ambiguous_same_type_goal_roles"] == 2


def test_repeated_explicit_same_entity_is_one_role_not_two_invented_roles():
    library, policy = prepared(
        source("a", "g0", task="inspect orb 1 then inspect orb 1"),
        source("b", "g1", kind="tile", number="2", task="inspect tile 2 then inspect tile 2"),
    )
    before, action, after = query(task="inspect tile 3 then inspect tile 3")
    result = anchor(controller(library, policy), before, action, after)
    assert result["decision"]["commit"]
    assert result["decision"]["goal_roles"] == ["tile"]


@pytest.mark.parametrize("mode", grounded.MODES)
@pytest.mark.parametrize("count,reason", [(0, "unbound_next_entity"), (1, None), (2, "ambiguous_next_entity")])
def test_unbound_next_goal_argument_requires_exactly_one_native_instance(mode, count, reason):
    library, policy = prepared(
        source("a", "g0", actions=("look", "inspect orb 1")),
        source("b", "g1", kind="tile", number="2", actions=("look", "inspect tile 2")),
    )
    native = [f"inspect tile {index + 3}" for index in range(count)] or ["look"]
    before, action, after = query(action="look", native=native)
    result = anchor(controller(library, policy, mode), before, action, after)
    assert result["decision"]["commit"] is (count == 1)
    if reason:
        assert result["decision"]["reason_counts"][reason] == 2
    else:
        assert result["proposal"]["action"] == "inspect tile 3"
        assert all(row["bindings"][0]["origin"] == "unique_current_native_entity"
                   for row in result["decision"]["candidates"])


def test_goal_absent_from_current_actions_is_still_parsed_from_train_vocabulary():
    library, policy = prepared(
        source("a", "g0", actions=("open crate 1", "visit crate 2"),
               pre_actions=("probe orb 1",), after="empty corridor"),
        source("b", "g1", kind="tile", number="2", actions=("open crate 2", "visit crate 4"),
               pre_actions=("probe tile 2",), after="empty corridor"),
    )
    before, action, after = query(action="open crate 9", native=["visit crate 3"], observation="empty corridor")
    result = anchor(controller(library, policy), before, action, after)
    assert result["decision"]["commit"]
    assert result["decision"]["goal_roles"] == ["tile"]
    assert result["proposal"]["action"] == "visit crate 3"
    for candidate in result["decision"]["candidates"]:
        assert candidate["bindings"][-1]["origin"] == "unique_current_native_entity"
        assert candidate["bindings"][-1]["goal_role"] is None


def test_distinct_unbound_next_entity_cannot_collapse_onto_an_already_bound_actor_entity():
    library, policy = prepared(
        source("a", "g0", actions=("touch orb 1", "inspect orb 2")),
        source("b", "g1", kind="tile", number="2", actions=("touch tile 2", "inspect tile 4")),
    )
    before, action, after = query()
    result = anchor(controller(library, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason_counts"]["non_bijective_next_argument"] == 2


def test_nongoal_types_and_nonentity_actor_syntax_cannot_be_renamed():
    library, policy = prepared(
        source("a", "g0", actions=("open crate 1", "inspect orb 1")),
        source("b", "g1", kind="tile", number="2", actions=("open crate 2", "inspect tile 2")),
    )
    for actor in ("open cabinet 9", "unlock crate 9", "open  crate 9"):
        before, action, after = query(action=actor)
        result = anchor(controller(library, policy), before, action, after)
        assert not result["decision"]["commit"]
        assert result["decision"]["reason"] == "no_source_for_anchor_shape"


@pytest.mark.parametrize("mode", grounded.MODES)
def test_grounded_candidate_requires_exact_native_membership(mode):
    library, policy = prepared()
    before, action, after = query(native=["observe tile 3"])
    result = anchor(controller(library, policy, mode), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason_counts"]["grounded_action_not_native"] == 2
    assert all(row["grounded_action"] == "inspect tile 3" for row in result["decision"]["rejections"])


def test_new_goal_affordance_rejects_stale_search_source_and_only_equality_is_ablated():
    library, policy = prepared(
        source("a", "g0", actions=("open crate 1", "visit crate 1"),
               next_actions=("probe orb 1",), after="unlatched crate 1"),
        source("b", "g1", kind="tile", number="2", actions=("open crate 2", "visit crate 2"),
               next_actions=("probe tile 2",), after="unlatched crate 2"),
    )
    before, action, after = query(
        action="open crate 3", observation="unlatched crate 3",
        native=["visit crate 3", "probe tile 7", "collect tile 7"],
    )
    full = anchor(controller(library, policy), before, action, after)
    ablated = anchor(controller(library, policy, grounded.MODES[1]), before, action, after)
    assert not full["decision"]["commit"]
    assert full["decision"]["reason_counts"]["goal_affordance_mismatch"] == 2
    assert ablated["decision"]["commit"]
    assert ablated["proposal"]["action"] == "visit crate 3"
    assert ablated["decision"]["support_groups"] == ["g0", "g1"]
    assert [row["similarity"] for row in full["decision"]["rejections"]] == [
        row["similarity"] for row in ablated["decision"]["candidates"]
    ]


def test_goal_affordance_templates_preserve_same_different_entity_identity_patterns():
    library, policy = prepared(
        source("a", "g0", next_actions=("compare orb 7 with orb 7",)),
        source("b", "g1", kind="tile", number="2", next_actions=("compare tile 8 with tile 8",)),
    )
    before, action, after = query(native=["inspect tile 3", "compare tile 5 with tile 6"])
    result = anchor(controller(library, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason_counts"]["goal_affordance_mismatch"] == 2
    after["actions"] = ["inspect tile 3", "compare tile 5 with tile 5"]
    assert anchor(controller(library, policy), before, action, after)["decision"]["commit"]


def test_irrelevant_nongoal_affordance_changes_do_not_block_transfer():
    library, policy = prepared()
    before, action, after = query(native=["inspect tile 3", "different crate 99"])
    assert anchor(controller(library, policy), before, action, after)["decision"]["commit"]


def test_many_prefixes_and_alias_programs_from_one_group_are_not_independent():
    library, policy = prepared(*[
        source(f"repeat{index}", "one-group", kind="tile", number=str(index + 1))
        for index in range(8)
    ])
    duplicate = deepcopy(library["programs"][0])
    duplicate["program_id"] = "another-program-alias"
    library["programs"].append(duplicate)
    before, action, after = query()
    result = anchor(controller(library, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "insufficient_independent_groups"
    assert result["decision"]["source_count"] == 8
    assert result["decision"]["support_groups"] == ["one-group"]
    assert result["decision"]["source_bank_counts"]["duplicate_examples"] == 1


def test_group_exclusion_precedes_vocabulary_and_index_fitting():
    library, policy = prepared(
        source("a", "g0"),
        source("b", "g1", kind="tile", number="2"),
        source("held", "target", kind="privatekind"),
        source("excluded", "held", kind="otherprivate"),
    )
    library["parameters"]["excluded_groups"] = ["held"]
    control = controller(library, policy)
    assert control._vocabulary == {"orb", "tile"}
    assert control._bank_counts["excluded_examples"] == 2
    assert {row["group_id"] for row in control._bank} == {"g0", "g1"}
    before, action, after = query(kind="privatekind")
    result = anchor(control, before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "unbound_goal_roles"


def test_only_two_step_sources_are_used_and_deduplicated():
    library, policy = prepared(
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
        source("long", "g2", kind="longonly",
               actions=("touch longonly 1", "inspect longonly 1", "finish longonly 1")),
    )
    control = controller(library, policy)
    assert control._bank_counts["ignored_non_two_step_programs"] == 1
    assert control._bank_counts["bank_sources"] == control._bank_counts["indexed_sources"] == 2
    assert "longonly" not in control._vocabulary


def test_old_literal_group_support_and_guards_do_not_govern_the_new_model():
    library, policy = prepared()
    assert all(len(row["source_groups"]) == 1 for row in library["programs"])
    for program in library["programs"]:
        program["steps"] = []
        program["positive_net_guard"] = program["refined_net_guard"] = None
        program["history_prerequisite_words"] = ["unused_literal_requirement"]
    before, action, after = query()
    result = anchor(controller(library, policy), before, action, after)
    assert result["decision"]["commit"]
    assert result["decision"]["support_group_count"] == 2


def test_supported_sources_must_agree_not_majority_vote():
    library, policy = prepared(
        source("a", "g0", next_actions=("observe orb 1",)),
        source("b", "g1", kind="tile", number="2", next_actions=("observe tile 2",)),
        source("c", "g2", kind="orb", number="3", actions=("touch orb 3", "observe orb 3"),
               next_actions=("inspect orb 3",)),
    )
    before, action, after = query(native=["inspect tile 3", "observe tile 3"])
    result = anchor(controller(library, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "ambiguous_grounded_actions"
    assert {row["action"]: row["support_groups"] for row in result["decision"]["competing_actions"]} == {
        "inspect tile 3": ["g0", "g1"], "observe tile 3": ["g2"],
    }


@pytest.mark.parametrize("mode", grounded.MODES)
def test_actor_output_is_actual_and_one_actor_call_is_mandatory_between_local_steps(mode):
    library, policy = prepared()
    before, action, after = query()
    control = controller(library, policy, mode)
    initial = control.decide(before)
    assert initial["decision"]["reason"] == "actor_anchor_required"
    assert initial["proposal"]["action"] is None
    control.observe(action, after["observation"], compiled=False)
    proposal = control.decide(after)
    assert proposal["proposal"]["action"] == "inspect tile 3"
    feedback = control.observe("inspect tile 3", "any actual native feedback", compiled=True)
    assert feedback["actor_required"]
    assert feedback["completed_action_count"] == 2
    assert feedback["anchor_action"] is None
    boundary = control.decide(before)
    assert not boundary["decision"]["commit"]
    assert boundary["decision"]["reason"] == "actor_boundary_required"
    control.observe(action, after["observation"], compiled=False)
    assert control.decide(after)["decision"]["commit"]


def test_actual_actor_syntax_not_a_preselected_source_action_determines_the_index():
    library, policy = prepared()
    before, _, after = query(action="other native command")
    control = controller(library, policy)
    assert not control.decide(before)["decision"]["commit"]
    control.observe("other native command", after["observation"], compiled=False)
    result = control.decide(after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "no_source_for_anchor_shape"


@pytest.mark.parametrize("mode", grounded.MODES)
def test_empty_or_non_native_actor_feedback_is_retained_without_an_anchor(mode):
    library, policy = prepared()
    before, _, after = query()
    control = controller(library, policy, mode)
    control.decide(before)
    observed = control.observe("", "rejected actor output", compiled=False, native_valid=False)
    assert observed["action"] == ""
    assert observed["reason"] == "actor_action_not_native"
    assert observed["native_action_count"] == 0
    assert observed["completed_action_count"] == 1
    assert observed["anchor_action"] is None
    assert control.decide(after)["decision"]["reason"] == "actor_anchor_required"


def test_source_next_state_is_train_data_but_unknown_current_observation_words_remain_in_norm():
    library, policy = prepared()
    before, action, after = query(observation=" ".join(f"unknown{index}" for index in range(100)))
    result = anchor(controller(library, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason_counts"]["below_entry_threshold"] == 2
    assert result["decision"]["entry_threshold"] == 0.55


def test_family_and_actor_shape_are_selected_before_ranking_and_models_are_cached():
    library, policy = prepared(
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
        source("distractor", "other", actions=("unrelated orb 1", "inspect orb 1")),
    )
    before, action, after = query()
    control = controller(library, policy)
    assert control._cache == {}
    result = anchor(control, before, action, after)
    assert result["decision"]["indexed_source_count"] == 2
    assert len(control._cache) == 1
    cached = next(iter(control._cache.values()))
    control.observe("inspect tile 3", "observed", compiled=True)
    anchor(control, before, action, after)
    assert next(iter(control._cache.values())) is cached


def test_serialization_source_order_native_order_and_input_immutability():
    library, policy = prepared()
    saved = deepcopy((library, policy))
    first = controller(library, policy)
    reordered_library, reordered_policy = json.loads(json.dumps(library)), json.loads(json.dumps(policy))
    reordered_library["programs"].reverse()
    reordered_policy["records"].reverse()
    for record in reordered_policy["records"]:
        record["public_state"]["actions"].reverse()
    second = controller(reordered_library, reordered_policy)
    before, action, after = query(native=["inspect tile 3", "irrelevant crate 5"])
    other_before, other_after = deepcopy(before), deepcopy(after)
    other_before["actions"].reverse()
    other_after["actions"].reverse()
    assert first.decide(before) == second.decide(other_before)
    assert first.observe(action, after["observation"], compiled=False) == second.observe(
        action, other_after["observation"], compiled=False,
    )
    result = first.decide(after)
    assert result == second.decide(other_after)
    assert result == json.loads(json.dumps(result, allow_nan=False))
    assert (library, policy) == saved
    result["decision"]["candidates"].clear()
    after["actions"].clear()
    assert first.observe("inspect tile 3", "done", compiled=True)["actor_required"]


@pytest.mark.parametrize("kwargs", [
    {"mode": True}, {"mode": "old_mode"}, {"mode": []},
    {"group_id": True}, {"group_id": ""}, {"task_family": None},
    {"entry_threshold": True}, {"entry_threshold": float("nan")},
    {"entry_threshold": float("inf")}, {"entry_threshold": -0.1}, {"entry_threshold": 1.1},
    {"minimum_groups": True}, {"minimum_groups": 1}, {"minimum_groups": 2.0},
])
def test_constructor_types_and_gates_are_strict(kwargs):
    library, policy = prepared()
    with pytest.raises(ValueError):
        controller(library, policy, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"compiled": 1}, {"compiled": 0}, {"compiled": None}, {"native_valid": 1},
    {"native_valid": "true"}, {"action": True}, {"action": None},
    {"observation": None}, {"observation": False},
])
def test_feedback_type_errors_do_not_consume_pending_decisions(kwargs):
    library, policy = prepared()
    before, action, after = query()
    control = controller(library, policy)
    control.decide(before)
    with pytest.raises(ValueError):
        control.observe(**{"action": action, "observation": after["observation"],
                           "compiled": False, **kwargs})
    assert control.observe(action, after["observation"], compiled=False)["anchor_action"] == action


def test_strict_alternation_compiled_membership_and_episode_reset():
    library, policy = prepared()
    before, action, after = query()
    control = controller(library, policy)
    with pytest.raises(ValueError, match="decide must precede"):
        control.observe(action, after["observation"], compiled=False)
    control.decide(before)
    with pytest.raises(ValueError, match="observe must follow"):
        control.decide(before)
    with pytest.raises(ValueError, match="pending"):
        control.reset_episode()
    with pytest.raises(ValueError, match="native set"):
        control.observe("not native", "", compiled=False)
    control.observe(action, after["observation"], compiled=False)
    control.decide(after)
    for executed, valid in (("", True), ("other", True), ("inspect tile 3", False)):
        with pytest.raises(ValueError):
            control.observe(executed, "actual feedback", compiled=True, native_valid=valid)
    control.observe("inspect tile 3", "feedback", compiled=True)
    control.reset_episode()
    assert control.decide(before) == controller(library, policy).decide(before)


@pytest.mark.parametrize("extra", ["reward", "official_success", "task_id", "expected_next_observation"])
def test_evaluator_hidden_and_future_fields_are_rejected(extra):
    library, policy = prepared()
    before, _, _ = query()
    with pytest.raises(ValueError, match="Public state"):
        controller(library, policy).decide({**before, extra: "not public"})
    policy["records"][0]["public_state"][extra] = "not public"
    with pytest.raises(ValueError, match="Public state"):
        controller(library, policy)


@pytest.mark.parametrize("mutation", [
    lambda lib, pol: lib.update(schema_version="unknown"),
    lambda lib, pol: lib["parameters"].update(excluded_groups=[True]),
    lambda lib, pol: lib["programs"][0]["actions"].__setitem__(0, True),
    lambda lib, pol: lib["programs"][0]["examples"][0].update(demonstration_ids=["one"]),
    lambda lib, pol: pol.update(schema_version="unknown"),
    lambda lib, pol: pol["records"].append(deepcopy(pol["records"][0])),
    lambda lib, pol: pol["records"].pop(0),
    lambda lib, pol: pol["records"][0].update(group_id="wrong"),
    lambda lib, pol: pol["records"][0]["public_state"].update(actions=["wrong"]),
])
def test_consumed_source_schema_is_validated_without_inventing_data(mutation):
    library, policy = prepared()
    mutation(library, policy)
    with pytest.raises(ValueError):
        controller(library, policy)


@pytest.mark.parametrize("mode", grounded.MODES)
@pytest.mark.parametrize("score,commit", [(0.55, True), (0.549, False)])
def test_similarity_threshold_is_inclusive_and_support_uses_only_eligible_groups(monkeypatch, mode, score, commit):
    library, policy = prepared()
    control = controller(library, policy, mode)
    original = control._scores

    def scores(key, vector, **kwargs):
        return [(1.0 if row["group_id"] == "g0" else score, row)
                for _, row in original(key, vector, **kwargs)]

    monkeypatch.setattr(control, "_scores", scores)
    before, action, after = query()
    result = anchor(control, before, action, after)
    assert result["decision"]["commit"] is commit
    assert result["decision"]["entry_threshold"] == 0.55
    assert result["decision"]["minimum_groups"] == 2
    if not commit:
        assert result["decision"]["support_groups"] == ["g0"]
        assert result["decision"]["reason_counts"]["below_entry_threshold"] == 1


def test_top128_cap_is_deterministic_and_explicit():
    library, policy = prepared(*[
        source(f"s{index:03}", f"g{index % 2}", kind="tile", number=str(index + 1))
        for index in range(132)
    ])
    before, action, after = query()
    result = anchor(controller(library, policy), before, action, after)
    assert result["decision"]["commit"]
    assert result["decision"]["indexed_source_count"] == 132
    assert result["decision"]["source_count"] == result["decision"]["retrieval_limit"] == 128
    assert result["decision"]["reason_counts"]["retrieval_limit"] == 4
    assert result["decision"]["support_groups"] == ["g0", "g1"]
    assert [row["source_id"] for row in result["decision"]["candidates"]] == [
        f"s{index:03}-demo-0" for index in range(128)
    ]


def test_family_metadata_is_used_before_vocabulary_and_index_not_inferred_from_task_text():
    library, policy = prepared(
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
        source("c", "g2", kind="privatekind", family="other-native-family"),
    )
    control = controller(library, policy)
    assert control._vocabulary == {"orb", "tile"}
    assert control._bank_counts["other_family_examples"] == 1
    before, action, after = query(task="inspect privatekind opaque-family", kind="privatekind")
    result = anchor(control, before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "unbound_goal_roles"


def test_empty_source_bank_returns_explicit_actor_deferral_without_fake_action():
    library, policy = prepared()
    library["programs"] = []
    before, action, after = query()
    result = anchor(controller(library, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["proposal"]["action"] is None
    assert result["decision"]["source_count"] == 0
    assert result["decision"]["source_bank_counts"]["bank_sources"] == 0


def test_observation_words_do_not_bind_a_next_entity_that_is_absent_from_native_actions():
    library, policy = prepared(
        source("a", "g0", actions=("look", "inspect orb 1")),
        source("b", "g1", kind="tile", number="2", actions=("look", "inspect tile 2")),
    )
    before, action, after = query(action="look", native=["look"], observation="tile 3 visible")
    after["history"] = [{"action": "inspect tile 3", "observation": "instructions tile 3"}]
    result = anchor(controller(library, policy, grounded.MODES[1]), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["goal_roles"] == ["tile"]
    assert result["decision"]["reason_counts"]["unbound_next_entity"] == 2


def test_program_counter_is_not_extended_and_actor_override_creates_a_fresh_anchor():
    library, policy = prepared()
    before, action, after = query(native=["inspect tile 3", "touch tile 3"])
    control = controller(library, policy, grounded.MODES[1])
    proposed = anchor(control, before, action, after)
    assert proposed["decision"]["commit"]
    control.observe("touch tile 3", after["observation"], compiled=False)
    next_decision = control.decide(after)
    assert next_decision["decision"]["commit"]
    assert next_decision["decision"]["anchor_turn"] == 1
    assert next_decision["decision"]["anchor_action"] == "touch tile 3"
    control.observe("inspect tile 3", "arbitrary observed feedback", compiled=True)
    boundary = control.decide(after)
    assert not boundary["decision"]["commit"]
    assert boundary["decision"]["reason"] == "actor_boundary_required"


_AUDIT_KEYS = {
    "schema_version", "mode", "turn", "group_id", "task_family", "source_bank_schema",
    "source_bank_hash", "vocabulary_hash", "source_bank_counts", "source_bank_reasons",
    "entry_threshold", "minimum_groups", "retrieval_limit", "affordance_check", "native_action_count",
    "source_evidence",
}


@pytest.mark.parametrize("mode", grounded.MODES)
def test_exact_public_api_keys_are_stable_for_deferral_grounding_and_feedback(mode):
    library, policy = prepared()
    before, action, after = query()
    control = controller(library, policy, mode)
    audit_keys = _AUDIT_KEYS | (
        {"context_partition_schema", "context_partition_model_hash"}
        if mode == grounded.PARTITIONED_MODE else set()
    )
    partition_keys = {"context_partition"} if mode == grounded.PARTITIONED_MODE else set()
    decision_keys = audit_keys | partition_keys | {
        "goal_roles", "anchor_shape_id", "indexed_source_count", "commit", "reason",
        "state_sha256", "anchor_action", "anchor_turn", "candidates", "source_count",
        "support_groups", "support_group_count", "competing_actions", "rejections", "reason_counts",
    }
    initial = control.decide(before)
    assert set(initial) == {"proposal", "decision"}
    assert set(initial["proposal"]) == {"action"}
    assert set(initial["decision"]) == decision_keys
    observed = control.observe(action, after["observation"], compiled=False)
    feedback_keys = audit_keys | partition_keys | {
        "action", "observation", "compiled", "native_valid", "reason", "actor_required",
        "anchor_action", "anchor_turn", "completed_action_count", "source_ids",
        "support_groups", "support_group_count",
    }
    assert set(observed) == feedback_keys
    committed = control.decide(after)
    assert set(committed["decision"]) == decision_keys
    assert set(committed["decision"]["candidates"][0]) == {
        "source_id", "next_demo_id", "group_id", "similarity", "grounded_action", "bindings",
    }
    assert set(committed["decision"]["candidates"][0]["bindings"][0]) == {
        "source_entity", "current_entity", "origin", "goal_role",
    }
    assert set(committed["decision"]["competing_actions"][0]) == {
        "action", "support_groups", "source_ids",
    }
    assert set(control.observe("inspect tile 3", "visible", compiled=True)) == feedback_keys


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def raw_prepared(*sources, start_turn=0, full_history=False, hashed_ids=True):
    programs, demos = [], []
    for program, records, _ in sources or (
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
    ):
        assert len(program["actions"]) == 2
        example = program["examples"][0]
        example["start_turn"] = start_turn
        prefix = [{"action": f"prior-action-{index}", "observation": f"prior-state-{index}"}
                  for index in range(start_turn)]
        if prefix:
            prefix[-1]["observation"] = records[0]["public_state"]["observation"]
        history = prefix + [{"action": records[0]["action"],
                             "observation": records[1]["public_state"]["observation"]}]
        records[0]["public_state"]["history"] = deepcopy(prefix if full_history else prefix[-2:])
        records[1]["public_state"]["history"] = deepcopy(history if full_history else history[-2:])
        if hashed_ids:
            ids = [digest([example["task_id"], start_turn + offset]) for offset in range(2)]
            example["demonstration_ids"] = ids
            for record, identifier in zip(records, ids):
                record["example_id"] = identifier
        programs.append(program)
        demos.extend(records)
    return {
        "schema_version": grounded.RAW_EVIDENCE_SCHEMA, "programs": programs,
        "source_training_ids": sorted({row["task_id"] for row in demos}),
        "source_corpus_sha256": digest(demos),
    }, build_policy(demos)


@pytest.mark.parametrize("mode", grounded.MODES)
def test_raw_singleton_literal_programs_can_supply_two_groups_after_legitimate_binding(mode):
    raw, policy = raw_prepared()
    assert all(len(program["examples"]) == 1 for program in raw["programs"])
    transitions = {}
    records = {row["example_id"]: row for row in policy["records"]}
    for program in raw["programs"]:
        first, second = program["examples"][0]["demonstration_ids"]
        transitions[first] = {
            "post_observation": records[second]["public_state"]["observation"],
            "pre_action_history": [], "task_family": "opaque-family",
        }
        transitions[second] = {
            "post_observation": "later observed change",
            "pre_action_history": [program["actions"][0]], "task_family": "opaque-family",
        }
    old_filtered = compile_library(raw["programs"], policy["records"], transitions, [], minimum_groups=2)
    assert old_filtered["programs"] == []
    before, action, after = query()
    old_result = anchor(controller(old_filtered, policy, mode), before, action, after)
    assert not old_result["decision"]["commit"]
    raw_control = controller(raw, policy, mode)
    result = anchor(raw_control, before, action, after)
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == "inspect tile 3"
    assert result["decision"]["support_groups"] == ["g0", "g1"]
    assert result["decision"]["source_count"] == 2
    assert result["decision"]["entry_threshold"] == 0.55
    assert result["decision"]["retrieval_limit"] == 128
    assert result["decision"]["minimum_groups"] == 2
    evidence = result["decision"]["source_evidence"]
    assert evidence["schema_version"] == grounded.RAW_EVIDENCE_SCHEMA
    assert evidence["source_corpus_sha256"] == raw["source_corpus_sha256"]
    assert evidence["source_training_ids_sha256"] == digest(raw["source_training_ids"])
    assert evidence["counts"] == {"schema_v5_programs": 0, "raw_two_step_programs": 2}


@pytest.mark.parametrize("start,full,hashed,proof", [
    (0, False, True, "raw_full_history_examples"),
    (1, False, True, "raw_full_history_examples"),
    (5, False, True, "raw_hashed_window_examples"),
    (0, True, False, "raw_full_history_examples"),
    (5, True, False, "raw_full_history_examples"),
])
def test_raw_pairs_verify_full_or_original_native_rolling_history(start, full, hashed, proof):
    raw, policy = raw_prepared(start_turn=start, full_history=full, hashed_ids=hashed)
    before, action, after = query()
    result = anchor(controller(raw, policy), before, action, after)
    assert result["decision"]["commit"]
    assert result["decision"]["source_bank_counts"][proof] == 2


def test_raw_exclusion_precedes_vocabulary_and_pair_fitting():
    raw, policy = raw_prepared(
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
        source("excluded", "target", kind="privatekind"),
    )
    target = next(row for row in policy["records"] if row["group_id"] == "target")
    target["public_state"] = {"not": "eligible evidence"}
    control = controller(raw, policy)
    assert control._vocabulary == {"orb", "tile"}
    assert control._bank_counts["excluded_examples"] == 1
    assert control._bank_counts["raw_full_history_examples"] == 2
    assert {row["group_id"] for row in control._bank} == {"g0", "g1"}
    before, action, after = query()
    assert anchor(control, before, action, after)["decision"]["commit"]


def test_raw_identical_starting_examples_are_counted_once_not_independent_sources():
    raw, policy = raw_prepared(source("a", "g0", kind="tile"))
    duplicate = deepcopy(raw["programs"][0])
    duplicate["program_id"] = "alias"
    raw["programs"].append(duplicate)
    control = controller(raw, policy)
    assert control._bank_counts["duplicate_examples"] == 1
    assert control._bank_counts["bank_sources"] == 1
    before, action, after = query()
    result = anchor(control, before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["source_count"] == 1
    assert result["decision"]["support_groups"] == ["g0"]
    assert result["decision"]["reason"] == "insufficient_independent_groups"


def test_raw_other_family_is_explicitly_excluded_and_never_supplies_types_or_support():
    raw, policy = raw_prepared(
        source("a", "g0"), source("b", "g1", kind="tile", number="2"),
        source("other", "g2", kind="privatekind", family="other-family"),
    )
    control = controller(raw, policy)
    assert control._bank_counts["other_family_examples"] == 1
    assert control._vocabulary == {"orb", "tile"}
    before, action, after = query()
    assert anchor(control, before, action, after)["decision"]["support_groups"] == ["g0", "g1"]


@pytest.mark.parametrize("mutation", [
    lambda raw, pol: raw.pop("source_corpus_sha256"),
    lambda raw, pol: raw.update(source_corpus_sha256=True),
    lambda raw, pol: raw.update(source_corpus_sha256="g" * 64),
    lambda raw, pol: raw.update(source_corpus_sha256="0" * 63),
    lambda raw, pol: raw.update(source_training_ids=True),
    lambda raw, pol: raw.update(source_training_ids=[True]),
    lambda raw, pol: raw["source_training_ids"].reverse(),
    lambda raw, pol: raw["source_training_ids"].append(raw["source_training_ids"][-1]),
    lambda raw, pol: raw.update(source_training_ids=[]),
    lambda raw, pol: raw.update(parameters={"minimum_groups": 2}),
    lambda raw, pol: raw["programs"][0].update(source_groups=["g0", "g1"]),
    lambda raw, pol: raw["programs"][0].update(actions=["one"]),
    lambda raw, pol: raw["programs"][0].update(actions=["one", "two", "three"]),
    lambda raw, pol: raw["programs"][0].update(actions=[True, "two"]),
    lambda raw, pol: raw["programs"][0].update(examples=[]),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(start_turn=True),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(start_turn=-1),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(start_turn=0.0),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(task_family=None),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(task_family=True),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(demonstration_ids=["one", "one"]),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(demonstration_ids=[True, "two"]),
    lambda raw, pol: raw["programs"][0]["examples"][0].update(task_id="undeclared"),
    lambda raw, pol: raw["programs"].append(deepcopy(raw["programs"][0])),
    lambda raw, pol: pol["records"].append(deepcopy(pol["records"][0])),
    lambda raw, pol: pol["records"].pop(0),
    lambda raw, pol: pol["records"][0].update(action="not the demonstrated action"),
    lambda raw, pol: pol["records"][0].update(task_family="wrong-explicit-family"),
    lambda raw, pol: pol["records"][0].update(task_id="wrong-task"),
    lambda raw, pol: pol["records"][0].update(group_id="wrong-group"),
    lambda raw, pol: pol["records"][0]["public_state"].update(actions=["not native"]),
])
def test_malformed_raw_banks_cannot_enter_the_source_bank(mutation):
    raw, policy = raw_prepared()
    mutation(raw, policy)
    with pytest.raises(ValueError):
        controller(raw, policy)


@pytest.mark.parametrize("mode", ["missing_history", "wrong_action", "wrong_observation", "wrong_prefix"])
def test_nonconsecutive_raw_full_history_pairs_are_rejected(mode):
    raw, policy = raw_prepared(start_turn=3, full_history=True)
    first, second = raw["programs"][0]["examples"][0]["demonstration_ids"]
    states = {row["example_id"]: row["public_state"] for row in policy["records"]}
    if mode == "missing_history":
        states[second]["history"] = []
    elif mode == "wrong_action":
        states[second]["history"][-1]["action"] = "another actor action"
    elif mode == "wrong_observation":
        states[second]["history"][-1]["observation"] = "not the actual post-observation"
    else:
        states[first]["history"][0]["action"] = "not in the same prefix"
    with pytest.raises(ValueError, match="consecutive"):
        controller(raw, policy)


@pytest.mark.parametrize("start", [1, 5])
def test_repeated_history_text_cannot_make_a_nonconsecutive_hashed_pair_look_valid(start):
    raw, policy = raw_prepared(start_turn=start)
    example = raw["programs"][0]["examples"][0]
    old_next = example["demonstration_ids"][1]
    new_next = digest([example["task_id"], start + 3])
    example["demonstration_ids"][1] = new_next
    next(row for row in policy["records"] if row["example_id"] == old_next)["example_id"] = new_next
    with pytest.raises(ValueError, match="chronology"):
        controller(raw, policy)


def test_raw_opaque_ids_with_only_truncated_history_are_not_silently_trusted():
    raw, policy = raw_prepared(start_turn=5, hashed_ids=False)
    with pytest.raises(ValueError, match="verifiable consecutive"):
        controller(raw, policy)


@pytest.mark.parametrize("field,value", [
    ("group_id", "other-group"), ("task_family", "other-family"), ("start_turn", 2),
])
def test_conflicting_raw_identities_fail_even_if_one_copy_would_be_filtered(field, value):
    raw, policy = raw_prepared()
    alias = deepcopy(raw["programs"][0])
    alias["program_id"] = "alias"
    alias["examples"][0][field] = value
    raw["programs"].append(alias)
    with pytest.raises(ValueError, match="identit"):
        controller(raw, policy)


def test_raw_provenance_affects_audit_not_features_and_serializes_deterministically():
    raw, policy = raw_prepared()
    saved = deepcopy((raw, policy))
    first = controller(raw, policy)
    extended = deepcopy(raw)
    extended["source_training_ids"] = sorted(raw["source_training_ids"] + ["unused-training-task"])
    extended["source_corpus_sha256"] = "f" * 64
    second = controller(extended, policy)
    before, action, after = query()
    a, b = anchor(first, before, action, after), anchor(second, before, action, after)
    assert a["proposal"] == b["proposal"]
    assert a["decision"]["candidates"] == b["decision"]["candidates"]
    assert a["decision"]["source_bank_hash"] != b["decision"]["source_bank_hash"]
    assert a["decision"]["source_evidence"]["evidence_hash"] != b["decision"]["source_evidence"]["evidence_hash"]
    feature_payload = json.dumps([row["features"] for rows in first._index.values() for row in rows])
    assert all(task not in feature_payload for task in raw["source_training_ids"])
    assert raw["source_corpus_sha256"] not in feature_payload
    raw["programs"].reverse()
    policy["records"].reverse()
    for record in policy["records"]:
        record["public_state"]["actions"].reverse()
    reordered = controller(json.loads(json.dumps(raw)), json.loads(json.dumps(policy)))
    assert anchor(reordered, before, action, after) == a
    assert a == json.loads(json.dumps(a, allow_nan=False))
    assert first._bank_hash == controller(*saved)._bank_hash


@pytest.mark.parametrize("mode,expected_trace", [
    ("grounded_continuation", "eab27cc1708a66e0aea2b209684b821831e7b77a8ec790d1fce94f9dfeeb0e78"),
    ("grounded_continuation_no_affordance_check", "d953e439d1dfa7806193a45a5f8977a4e255c419abf44c508a605c769f1c85a1"),
])
def test_v5_existing_trace_fields_and_bank_hash_are_unchanged(mode, expected_trace):
    library, policy = prepared()
    before, action, after = query()
    control = controller(library, policy, mode)
    trace = []
    for _ in range(2):
        trace.append(control.decide(before))
        trace.append(control.observe(action, after["observation"], compiled=False))
        trace.append(control.decide(after))
        trace.append(control.observe("inspect tile 3", "observed", compiled=True))
    for item in trace:
        audit = item["decision"] if "decision" in item else item
        assert audit["source_bank_hash"] == "4608aa81e1107e791304848462ffb1b3427f45432cd12297428aa93375d23fea"
        provenance = audit.pop("source_evidence")
        assert provenance["schema_version"] == grounded.V5_SOURCE_SCHEMA
        assert provenance["counts"] == {"schema_v5_programs": 2, "raw_two_step_programs": 0}
    assert digest(trace) == expected_trace


def test_raw_rolling_history_preserves_earlier_malformed_actor_steps():
    raw, policy = raw_prepared(start_turn=4)
    by_id = {row["example_id"]: row for row in policy["records"]}
    for program in raw["programs"]:
        first, second = program["examples"][0]["demonstration_ids"]
        before, after = by_id[first]["public_state"], by_id[second]["public_state"]
        before["history"][-1]["action"] = ""
        after["history"][0] = deepcopy(before["history"][-1])
    control = controller(raw, policy)
    assert control._bank_counts["raw_hashed_window_examples"] == 2
    before, action, after = query()
    assert anchor(control, before, action, after)["decision"]["commit"]


def test_raw_corpus_digest_is_recorded_canonically_without_claiming_content_verification():
    raw, policy = raw_prepared()
    first = controller(raw, policy)
    raw["source_corpus_sha256"] = raw["source_corpus_sha256"].upper()
    second = controller(raw, policy)
    assert first._source_evidence == second._source_evidence
    assert first._bank_hash == second._bank_hash


def test_empty_raw_catalog_has_explicit_source_schema_and_no_candidate():
    _, policy = prepared()
    raw = {
        "schema_version": grounded.RAW_EVIDENCE_SCHEMA, "programs": [],
        "source_training_ids": [], "source_corpus_sha256": "0" * 64,
    }
    before, action, after = query()
    result = anchor(controller(raw, policy), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["source_bank_counts"]["bank_sources"] == 0
    assert result["decision"]["source_evidence"]["source_training_task_count"] == 0
    assert result["decision"]["source_evidence"]["schema_version"] == grounded.RAW_EVIDENCE_SCHEMA


def partitioned_sources(*, counts=(2, 2), include_excluded=False):
    rows = []
    for branch, count in zip(("red", "blue"), counts):
        for index in range(count):
            kind = "orb" if index % 2 == 0 else "tile"
            verb = "inspect" if branch == "red" else "observe"
            other = "observe" if branch == "red" else "inspect"
            rows.append(source(
                f"{branch}-{index}", f"{branch}-group-{index}", kind=kind, number=str(index + 1),
                actions=(f"touch {kind} {index + 1}", f"{verb} {kind} {index + 1}"),
                next_actions=(f"{other} {kind} {index + 1}",),
                after=f"{kind} {index + 1} visible on a shared broad platform nearby {branch}",
            ))
    if include_excluded:
        rows.append(source("held", "target", kind="privatekind", after="secretflag privatekind 1"))
    return raw_prepared(*rows)


def partitioned_query(branch):
    return query(
        native=["inspect tile 3", "observe tile 3"],
        observation=f"tile 3 visible on a shared broad platform nearby {branch}",
    )


@pytest.mark.parametrize("branch,expected", [("red", "inspect tile 3"), ("blue", "observe tile 3")])
def test_context_partition_separates_supported_branches_without_majority_actions(branch, expected):
    library, policy = partitioned_sources()
    before, action, after = partitioned_query(branch)
    baseline = anchor(controller(library, policy), before, action, after)
    assert not baseline["decision"]["commit"]
    assert baseline["decision"]["reason"] == "ambiguous_grounded_actions"
    assert baseline["decision"]["source_count"] == 4
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    result = anchor(control, before, action, after)
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == expected
    assert result["decision"]["support_groups"] == [f"{branch}-group-0", f"{branch}-group-1"]
    partition = result["decision"]["context_partition"]
    assert partition["leaf_group_count"] == partition["leaf_source_count"] == 2
    assert partition["outside_leaf_count"] == 2
    assert len(partition["path"]) == 1
    assert partition["leaf_source_ids"] == sorted(row["source_id"] for row in result["decision"]["candidates"])
    assert result["decision"]["entry_threshold"] == 0.55
    assert result["decision"]["minimum_groups"] == 2
    assert result["decision"]["retrieval_limit"] == 128
    feedback = control.observe(expected, "actual feedback", compiled=True)
    assert feedback["context_partition"] == partition
    assert feedback["actor_required"]
    assert control.decide(before)["decision"]["reason"] == "actor_boundary_required"


def tree_record(identifier, group, features, label):
    return {
        "source_id": identifier, "group_id": group,
        "partition_features": set(features), "partition_label": json.dumps([label]),
    }


def test_group_balanced_duplicate_prefixes_do_not_dominate_feature_selection_or_splits():
    rows = [
        tree_record("a0", "g0", ["a", "b"], "left"),
        tree_record("a1", "g1", ["a", "b"], "left"),
        tree_record("b0", "g2", ["c"], "right"),
        tree_record("b1", "g3", ["c"], "right"),
    ]
    duplicated = rows + [tree_record(f"duplicate-{i}", "g0", ["a", "b"], "left") for i in range(80)]
    first = grounded._fit_context_partition(rows, "family", "shape")
    second = grounded._fit_context_partition(duplicated, "family", "shape")
    assert first["candidate_features"] == second["candidate_features"]
    a, b = first["tree"], second["tree"]
    assert a["predicate"] == b["predicate"] == "a"
    assert a["gain_fraction"] == b["gain_fraction"] == [1, 2]
    assert a["impurity"] == b["impurity"] == 0.5
    assert a["label_weights"] == b["label_weights"]
    assert sum(row["weight"] for row in b["label_weights"]) == b["group_count"] == 4
    assert b["source_count"] == 84
    assert all(row["reason"] == "lower_gain_or_lexical_tie" for row in a["rejected_splits"])


def test_group_weight_is_one_even_when_one_group_has_multiple_labels_and_contexts():
    rows = [
        tree_record("a", "g0", ["p"], "left"),
        tree_record("b", "g0", [], "right"),
        tree_record("c", "g1", ["p"], "left"),
        tree_record("d", "g1", [], "right"),
    ]
    artifact = grounded._fit_context_partition(rows, "family", "shape")
    root = artifact["tree"]
    assert root["predicate"] == "p"
    assert root["group_count"] == 2
    assert root["gain_fraction"] == [1, 2]
    assert all(row["weight"] == 1.0 for row in root["label_weights"])
    for branch in ("absent", "present"):
        assert root[branch]["group_count"] == 2
        assert root[branch]["label_weights"][0]["weight"] == 2.0


def test_children_without_two_groups_are_not_split_and_conflicting_leaf_still_defers():
    library, policy = partitioned_sources(counts=(1, 1))
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    artifact = control.context_partition_artifacts()[0]
    assert artifact["tree"]["predicate"] is None
    assert artifact["tree"]["stop_reason"] == "no_supported_strict_improvement"
    assert any(row["reason"] == "child_group_support" for row in artifact["rejected_features"])
    before, action, after = query(
        kind="orb", native=["inspect orb 3", "observe orb 3"],
        observation="orb 3 visible on a shared broad platform nearby red",
    )
    result = anchor(control, before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "ambiguous_grounded_actions"
    assert result["decision"]["context_partition"]["leaf_group_count"] == 2


def test_no_strict_impurity_improvement_means_no_split_not_a_majority_prediction():
    rows = [
        tree_record(f"{group}-{label}", group, ["p"] if group in ("g0", "g1") else [], label)
        for group in ("g0", "g1", "g2", "g3") for label in ("a", "b")
    ]
    artifact = grounded._fit_context_partition(rows, "family", "shape")
    root = artifact["tree"]
    assert root["predicate"] is None
    assert root["stop_reason"] == "no_supported_strict_improvement"
    assert root["rejected_splits"][0]["reason"] == "no_strict_impurity_reduction"
    assert root["rejected_splits"][0]["gain"] == 0.0
    assert "action" not in root and "prediction" not in root


def walk_tree(node):
    yield node
    if node["predicate"] is not None:
        yield from walk_tree(node["absent"])
        yield from walk_tree(node["present"])


def test_tree_depth_and_train_feature_caps_are_prospective_and_audited():
    rows = [
        tree_record(f"{group}-{code}", group, [f"bit{bit}" for bit in range(4) if code & (1 << bit)],
                    f"choice{code}")
        for group in ("g0", "g1") for code in range(16)
    ]
    artifact = grounded._fit_context_partition(rows, "family", "shape")
    nodes = list(walk_tree(artifact["tree"]))
    assert max(node["depth"] for node in nodes) == 3
    assert len([node for node in nodes if node["predicate"] is None]) == 8
    assert all(node["stop_reason"] == "depth_cap" for node in nodes if node["depth"] == 3)
    assert all(node["group_count"] >= 2 for node in nodes)
    features = [f"feature-{i:03}" for i in range(100)]
    capped = grounded._fit_context_partition([
        tree_record("a", "g0", features, "a"), tree_record("b", "g1", features, "a"),
        tree_record("c", "g2", [], "b"), tree_record("d", "g3", [], "b"),
    ], "family", "shape")
    assert capped["candidate_features"] == features[:64]
    assert sum(row["reason"] == "candidate_feature_cap" for row in capped["rejected_features"]) == 36
    assert capped["tree"]["predicate"] == features[0]
    assert capped["parameters"]["max_depth"] == 3
    assert capped["parameters"]["minimum_child_groups"] == 2


def test_partition_fitting_and_predicates_exclude_target_group_before_any_features():
    library, policy = partitioned_sources(include_excluded=True)
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    assert "privatekind" not in control._vocabulary
    assert control._bank_counts["excluded_examples"] == 1
    artifacts = control.context_partition_artifacts()
    payload = json.dumps(artifacts)
    assert "secretflag" not in payload
    assert "privatekind" not in payload
    excluded_id = next(row["example_id"] for row in policy["records"] if row["group_id"] == "target")
    assert excluded_id not in payload
    assert all("target" not in node["support_groups"]
               for artifact in artifacts for node in walk_tree(artifact["tree"]))


def test_features_are_observables_not_teacher_labels_or_source_identifiers():
    library, policy = partitioned_sources()
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    row = next(iter(control._index.values()))[0]
    flags = [False] * len(row["next_state"]["history"])
    first = grounded._partition_features(row["next_state"], row["context"], control._vocabulary, flags)
    altered = deepcopy(row)
    altered["next_action"] = "unobservable future choice"
    altered["partition_label"] = "unobservable future choice"
    altered["source_id"] = "private-task-id"
    altered["group_id"] = "private-group-id"
    second = grounded._partition_features(
        altered["next_state"], altered["context"], control._vocabulary, flags,
    )
    assert first == second
    payload = json.dumps(sorted(first))
    for forbidden in ("private-task-id", "private-group-id", "unobservable future choice",
                      row["source_id"], row["group_id"], "partition_label"):
        assert forbidden not in payload
    assert all(json.loads(feature)[0] in ("observation", "affordance") for feature in first)


def history_partition_bank():
    programs, records = [], []
    for index in range(4):
        task_id, group = f"history-task-{index}", f"history-group-{index}"
        kind = "orb" if index % 2 == 0 else "tile"
        prior = "prime" if index < 2 else "reset"
        next_verb = "inspect" if index < 2 else "observe"
        actions = [f"{prior} {kind} 1", f"touch {kind} 1", f"{next_verb} {kind} 1"]
        ids = [digest([task_id, step]) for step in range(3)]
        history = []
        for step, action in enumerate(actions):
            public = state(
                f"inspect {kind}", f"{kind} 1 visible on a shared broad platform nearby",
                list(dict.fromkeys([action, f"touch {kind} 1", f"inspect {kind} 1", f"observe {kind} 1"])),
                deepcopy(history[-2:]),
            )
            records.append({"example_id": ids[step], "task_id": task_id, "group_id": group,
                            "action": action, "public_state": public})
            history.append({"action": action, "observation": public["observation"]})
        for start in (0, 1):
            programs.append({
                "program_id": f"{task_id}-{start}", "actions": actions[start:start + 2],
                "examples": [{"task_id": task_id, "group_id": group, "start_turn": start,
                              "task_family": "opaque-family", "demonstration_ids": ids[start:start + 2]}],
            })
    return {
        "schema_version": grounded.RAW_EVIDENCE_SCHEMA, "programs": programs,
        "source_training_ids": sorted({row["task_id"] for row in records}),
        "source_corpus_sha256": digest(records),
    }, build_policy(records)


@pytest.mark.parametrize("prior,expected", [("prime", "inspect tile 3"), ("reset", "observe tile 3")])
def test_verified_prior_native_history_can_partition_otherwise_identical_contexts(prior, expected):
    library, policy = history_partition_bank()
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    observation = "tile 3 visible on a shared broad platform nearby"
    initial = state("inspect tile", observation, [f"{prior} tile 3"])
    control.decide(initial)
    control.observe(f"{prior} tile 3", observation, compiled=False)
    actor_state = state(
        "inspect tile", observation, ["touch tile 3", "inspect tile 3", "observe tile 3"],
        [{"action": f"{prior} tile 3", "observation": observation}],
    )
    control.decide(actor_state)
    control.observe("touch tile 3", observation, compiled=False)
    after = deepcopy(actor_state)
    after["history"].append({"action": "touch tile 3", "observation": observation})
    result = control.decide(after)
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == expected
    partition = result["decision"]["context_partition"]
    assert partition["verified_history_entries"] == 2
    assert json.loads(partition["path"][0]["predicate"])[0] == "history.prior"


def test_history_features_require_actual_verified_history_not_instruction_text_or_guessed_facts():
    library, policy = history_partition_bank()
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    before, action, after = query(
        native=["inspect tile 3", "observe tile 3"],
        observation="tile 3 visible on a shared broad platform nearby prime",
    )
    before["history"] = [{"action": "prime tile 3", "observation": before["observation"]}]
    control.decide(before)
    control.observe(action, after["observation"], compiled=False)
    after["history"] = [
        {"action": "prime tile 3", "observation": before["observation"]},
        {"action": action, "observation": after["observation"]},
    ]
    flags = control._runtime_history(after)
    assert flags == [False, True]
    context, reason = grounded._context(before["task"], action, control._vocabulary)
    assert reason is None
    features = grounded._partition_features(after, context, control._vocabulary, flags)
    assert any(json.loads(feature)[0] == "history.last" for feature in features)
    assert not any(json.loads(feature)[0] == "history.prior" for feature in features)


def test_partition_artifacts_are_json_deterministic_and_cannot_mutate_the_model():
    library, policy = partitioned_sources()
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    artifact = control.context_partition_artifacts()[0]
    assert artifact["schema_version"] == grounded.CONTEXT_PARTITION_SCHEMA
    assert artifact == json.loads(json.dumps(artifact, allow_nan=False))
    frozen_hash = artifact["artifact_hash"]
    artifact["tree"]["source_ids"].clear()
    artifact["tree"]["predicate"] = "mutated"
    assert control.context_partition_artifacts()[0]["artifact_hash"] == frozen_hash
    assert control.context_partition_artifacts()[0]["tree"]["source_ids"]
    library["programs"].reverse()
    policy["records"].reverse()
    for row in policy["records"]:
        row["public_state"]["actions"].reverse()
    reordered = controller(json.loads(json.dumps(library)), json.loads(json.dumps(policy)),
                           grounded.PARTITIONED_MODE)
    assert reordered.context_partition_artifacts() == control.context_partition_artifacts()
    before, action, after = partitioned_query("red")
    assert anchor(control, before, action, after) == anchor(reordered, before, action, after)


@pytest.mark.parametrize("kind,mode,expected", [
    ("v5", "grounded_continuation", "f0c29d3d37da3f0e912aabd1512173e19986b9e46dc50018d1f52846e0ca515f"),
    ("v5", "grounded_continuation_no_affordance_check", "169fcbd82267cbe80adca02940a12497bcf3ff69505e51b4849c4dfc3e5286d1"),
    ("raw", "grounded_continuation", "9a10972d87fad11cfbf734bcd751ca05d416667067e0922905b08e82f2111cb6"),
    ("raw", "grounded_continuation_no_affordance_check", "0df2bde1fb8c6c345cc6e56a0e096c6614cd13183b6697283549c88882e8b0c1"),
])
def test_unpartitioned_modes_preserve_complete_pre_revision_v5_and_raw_traces(kind, mode, expected):
    library, policy = prepared() if kind == "v5" else raw_prepared(start_turn=5)
    before, action, after = query()
    control = controller(library, policy, mode)
    assert control.context_partition_artifacts() == []
    trace = []
    for _ in range(2):
        trace.extend([
            control.decide(before), control.observe(action, after["observation"], compiled=False),
            control.decide(after), control.observe("inspect tile 3", "observed", compiled=True),
        ])
    assert digest(trace) == expected


def test_partition_does_not_borrow_outside_leaf_groups_or_bypass_native_affordance_checks():
    library, policy = partitioned_sources()
    before, action, after = partitioned_query("red")
    control = controller(library, policy, grounded.PARTITIONED_MODE, minimum_groups=3)
    result = anchor(control, before, action, after)
    assert result["decision"]["indexed_source_count"] == 4
    assert result["decision"]["context_partition"]["leaf_group_count"] == 2
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "insufficient_independent_groups"
    after["actions"].append("collect tile 3")
    result = anchor(controller(library, policy, grounded.PARTITIONED_MODE), before, action, after)
    assert not result["decision"]["commit"]
    assert result["decision"]["reason_counts"]["goal_affordance_mismatch"] == 2


def test_partition_uses_the_control_tfidf_weights_but_scores_only_the_selected_leaf():
    library, policy = partitioned_sources()
    before, action, after = partitioned_query("red")
    baseline = anchor(controller(library, policy), before, action, after)
    result = anchor(controller(library, policy, grounded.PARTITIONED_MODE), before, action, after)
    scores = {row["source_id"]: row["similarity"] for row in baseline["decision"]["candidates"]}
    assert all(row["similarity"] == scores[row["source_id"]] for row in result["decision"]["candidates"])
    leaf_ids = set(result["decision"]["context_partition"]["leaf_source_ids"])
    assert all(row["source_id"] in leaf_ids for row in result["decision"]["candidates"])
    assert all(row["source_id"] in leaf_ids for row in result["decision"]["rejections"])


def test_partition_artifact_and_leaf_audit_key_sets():
    library, policy = partitioned_sources()
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    artifact = control.context_partition_artifacts()[0]
    assert set(artifact) == {
        "schema_version", "task_family", "anchor_shape_id", "parameters",
        "candidate_features", "rejected_features", "tree", "artifact_hash",
    }
    assert set(artifact["tree"]) == {
        "node_id", "depth", "source_count", "source_ids", "support_groups", "group_count",
        "impurity", "label_weights", "predicate", "gain", "gain_fraction",
        "rejected_splits", "absent", "present", "stop_reason",
    }
    before, action, after = partitioned_query("red")
    result = anchor(control, before, action, after)
    assert result["decision"]["schema_version"] == grounded.PARTITIONED_CONTROLLER_SCHEMA
    assert set(result["decision"]["context_partition"]) == {
        "artifact_hash", "leaf_id", "path", "leaf_source_ids", "leaf_support_groups",
        "leaf_source_count", "leaf_group_count", "outside_leaf_count",
        "runtime_feature_hash", "verified_history_entries",
    }
    assert set(result["decision"]["context_partition"]["path"][0]) == {
        "node_id", "predicate", "present",
    }
    assert result == json.loads(json.dumps(result, allow_nan=False))


def test_missing_or_nonnative_runtime_history_cannot_create_a_native_history_predicate():
    library, policy = history_partition_bank()
    control = controller(library, policy, grounded.PARTITIONED_MODE)
    observation = "tile 3 visible on a shared broad platform nearby"
    before = state("inspect tile", observation, ["touch tile 3"])
    control.decide(before)
    control.observe("prime tile 3", observation, compiled=False, native_valid=False)
    control.decide(before)
    control.observe("touch tile 3", observation, compiled=False)
    after = state("inspect tile", observation, ["inspect tile 3", "observe tile 3"], [
        {"action": "prime tile 3", "observation": observation},
        {"action": "touch tile 3", "observation": observation},
    ])
    assert control._runtime_history(after) == [False, True]
    context, _ = grounded._context(before["task"], "touch tile 3", control._vocabulary)
    features = grounded._partition_features(
        after, context, control._vocabulary, control._runtime_history(after),
    )
    assert not any(json.loads(feature)[0] == "history.prior" for feature in features)
