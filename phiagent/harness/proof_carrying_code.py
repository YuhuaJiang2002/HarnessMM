"""Fail-closed static contracts for code-as-policy manipulation programs.

The auditor deliberately checks only claims that are visible in source code.  It
does not turn program order into force, collision, calibration, or execution
evidence; simulator success remains the external authority for task completion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.9.0"
SUPPORTED_TASKS = (
    "cube_lift",
    "cube_stack",
    "nut_assembly",
    "two_arm_lift",
    "two_arm_handover",
)

_ROBOT_CALLS = {
    "close_gripper",
    "close_gripper_arm0",
    "close_gripper_arm1",
    "get_handle0_pos",
    "get_handle0_pose",
    "get_handle1_pos",
    "get_handle1_pose",
    "get_hammer_pose",
    "get_object_pose",
    "goto_home_joint_position",
    "goto_pose",
    "goto_pose_arm0",
    "goto_pose_arm1",
    "goto_pose_both",
    "open_gripper",
    "open_gripper_arm0",
    "open_gripper_arm1",
    "sample_grasp_pose",
}
_FORBIDDEN_CALLS = {"compile", "eval", "exec", "globals", "locals", "__import__"}
_ACTION_LIKE_CALL_TERMS = (
    "approach",
    "grasp",
    "gripper",
    "goto",
    "handover",
    "insert",
    "lift",
    "move",
    "pick",
    "place",
    "release",
    "retract",
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strip_code_fences(content: str) -> str:
    """Normalize the model response without changing executable statements."""

    text = str(content).strip()
    if text.startswith("```python"):
        text = text[len("```python") :].lstrip("\r\n")
    elif text.startswith("```"):
        text = text[3:].lstrip("\r\n")
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = [node.func.attr]
        value = node.func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return "<dynamic>"


def _literal(value: ast.AST, bindings: Mapping[str, Any] | None = None) -> Any:
    if isinstance(value, ast.Name) and bindings is not None:
        return bindings.get(value.id)
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class CallEvent:
    index: int
    line: int
    name: str
    string_arguments: tuple[str, ...]
    positional_literals: tuple[Any, ...]
    keyword_literals: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "line": self.line,
            "name": self.name,
            "string_arguments": list(self.string_arguments),
            "positional_literals": list(self.positional_literals),
            "keyword_literals": {key: value for key, value in self.keyword_literals},
        }


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    passed: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ProgramAudit:
    task_family: str
    coordinate_frame: str
    events: tuple[CallEvent, ...]
    gates: tuple[GateResult, ...]
    accepted: bool
    parse_error: str | None
    control_flow_ambiguities: tuple[int, ...]
    claim_boundary: str

    @property
    def failed_gate_ids(self) -> tuple[str, ...]:
        return tuple(gate.gate_id for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "task_family": self.task_family,
            "coordinate_frame": self.coordinate_frame,
            "events": [event.to_dict() for event in self.events],
            "gates": [gate.to_dict() for gate in self.gates],
            "accepted": self.accepted,
            "parse_error": self.parse_error,
            "control_flow_ambiguities": list(self.control_flow_ambiguities),
            "claim_boundary": self.claim_boundary,
        }
        payload["audit_sha256"] = _canonical_sha256(payload)
        return payload


def _contains_robot_call(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call) and _call_name(child).split(".")[-1] in _ROBOT_CALLS
        for child in ast.walk(node)
    )


def _execution_statements(tree: ast.Module) -> tuple[Sequence[ast.stmt], list[int]]:
    """Select one statically visible robot execution scope.

    CaP-X accepts both straight-line modules and programs wrapped in ``main``.
    Robot calls in any other function, or split across module and ``main``, are
    rejected as ambiguous instead of being silently ignored.
    """

    main_functions = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "main"
        and _contains_robot_call(statement)
    ]
    hidden_robot_functions = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "main"
        )
        and _contains_robot_call(statement)
    ]
    module_robot_statements = [
        statement
        for statement in tree.body
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and _contains_robot_call(statement)
    ]
    ambiguous_lines = [int(statement.lineno) for statement in hidden_robot_functions]
    if len(main_functions) > 1:
        ambiguous_lines.extend(int(statement.lineno) for statement in main_functions)
    if main_functions and module_robot_statements:
        ambiguous_lines.extend(int(statement.lineno) for statement in module_robot_statements)
    if len(main_functions) == 1:
        main = main_functions[0]
        if main.args.args or main.args.posonlyargs or main.args.kwonlyargs:
            ambiguous_lines.append(int(main.lineno))
        return main.body, sorted(set(ambiguous_lines))
    return tree.body, sorted(set(ambiguous_lines))


def _scope_calls(
    statements: Sequence[ast.stmt], initial_ambiguities: Sequence[int]
) -> tuple[list[ast.Call], list[int]]:
    calls: list[ast.Call] = []
    ambiguous_lines = list(initial_ambiguities)

    def collect(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            if _contains_robot_call(node):
                ambiguous_lines.append(int(getattr(node, "lineno", 0)))
            return
        if isinstance(node, ast.Call):
            calls.append(node)
        for child in ast.iter_child_nodes(node):
            collect(child)

    for statement in statements:
        collect(statement)
    calls.sort(key=lambda call: (int(call.lineno), int(call.col_offset)))
    return calls, sorted(set(ambiguous_lines))


def _constant_bindings(statements: Sequence[ast.stmt]) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _literal(statement.value, bindings)
        if value is not None:
            bindings[target.id] = value
    return bindings


def _events(
    calls: Sequence[ast.Call], bindings: Mapping[str, Any]
) -> tuple[CallEvent, ...]:
    result = []
    for index, call in enumerate(calls):
        strings = tuple(
            value
            for argument in call.args
            if isinstance((value := _literal(argument, bindings)), str)
        )
        keywords = tuple(
            sorted(
                (keyword.arg or "**", _literal(keyword.value, bindings))
                for keyword in call.keywords
            )
        )
        result.append(
            CallEvent(
                index=index,
                line=int(call.lineno),
                name=_call_name(call),
                string_arguments=strings,
                positional_literals=tuple(
                    _literal(argument, bindings) for argument in call.args
                ),
                keyword_literals=keywords,
            )
        )
    return tuple(result)


def _indices(events: Sequence[CallEvent], name: str) -> list[int]:
    return [event.index for event in events if event.name.split(".")[-1] == name]


def _first(indices: Sequence[int]) -> int | None:
    return indices[0] if indices else None


def _last(indices: Sequence[int]) -> int | None:
    return indices[-1] if indices else None


def _ordered(*indices: int | None) -> bool:
    return all(index is not None for index in indices) and list(indices) == sorted(indices)


def _event_has_text(event: CallEvent, terms: Iterable[str]) -> bool:
    text = " ".join(event.string_arguments).lower()
    return all(term.lower() in text for term in terms)


def _positive_approach(event: CallEvent) -> bool:
    value = _approach_value(event)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _approach_value(event: CallEvent) -> Any:
    keywords = dict(event.keyword_literals)
    value = keywords.get("z_approach")
    if value is None and len(event.positional_literals) >= 3:
        value = event.positional_literals[2]
    return value


def _common_gates(
    events: Sequence[CallEvent],
    *,
    parse_error: str | None,
    coordinate_frame: str,
    ambiguous_lines: Sequence[int],
) -> list[GateResult]:
    forbidden = [
        event.name
        for event in events
        if event.name.split(".")[-1] in _FORBIDDEN_CALLS
        or event.name.startswith("env.")
        or event.name.startswith("subprocess.")
        or event.name.startswith("os.system")
    ]
    undocumented_actions = sorted(
        {
            event.name
            for event in events
            if (leaf := event.name.split(".")[-1]) not in _ROBOT_CALLS
            and any(term in leaf.lower() for term in _ACTION_LIKE_CALL_TERMS)
        }
    )
    return [
        GateResult("python_syntax", parse_error is None, parse_error or "AST parse passed"),
        GateResult(
            "named_coordinate_frame",
            bool(coordinate_frame and ":" in coordinate_frame),
            coordinate_frame or "missing coordinate frame",
        ),
        GateResult(
            "api_surface_only",
            not forbidden and not undocumented_actions,
            "no direct environment, dynamic execution, or undocumented action calls"
            if not forbidden and not undocumented_actions
            else (
                f"forbidden calls={forbidden}, "
                f"undocumented action calls={undocumented_actions}"
            ),
        ),
        GateResult(
            "single_path_robot_sequence",
            not ambiguous_lines,
            "robot API order is statically visible"
            if not ambiguous_lines
            else f"robot calls under branch/loop at lines {list(ambiguous_lines)}",
        ),
    ]


def _sampled_grasp_execution_evidence(
    statements: Sequence[ast.stmt],
    calls: Sequence[ast.Call],
    *,
    object_terms: Sequence[str],
) -> tuple[bool, str]:
    """Require the first grasp motion to execute a two-value sampled pose."""

    sample_bindings: dict[int, tuple[str, str]] = {}

    def collect(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            value = node.value
            if (
                isinstance(value, ast.Call)
                and _call_name(value).split(".")[-1] == "sample_grasp_pose"
            ):
                names = _tuple_names(node.targets[0])
                if len(names) >= 2 and names[0] is not None and names[1] is not None:
                    sample_bindings[id(value)] = (str(names[0]), str(names[1]))
        for child in ast.iter_child_nodes(node):
            collect(child)

    for statement in statements:
        collect(statement)

    matching_sample: tuple[int, str, str] | None = None
    first_goto: tuple[int, ast.Call] | None = None
    first_close_index: int | None = None
    for index, call in enumerate(calls):
        leaf = _call_name(call).split(".")[-1]
        if leaf == "sample_grasp_pose" and matching_sample is None:
            query = _literal(call.args[0]) if call.args else None
            if (
                isinstance(query, str)
                and all(term.lower() in query.lower() for term in object_terms)
                and id(call) in sample_bindings
            ):
                pos_name, quat_name = sample_bindings[id(call)]
                matching_sample = (index, pos_name, quat_name)
        elif leaf == "goto_pose" and first_goto is None:
            first_goto = (index, call)
        elif leaf == "close_gripper" and first_close_index is None:
            first_close_index = index

    passed = False
    observed = None
    if matching_sample is not None and first_goto is not None:
        sample_index, pos_name, quat_name = matching_sample
        goto_index, goto_call = first_goto
        observed = tuple(
            argument.id if isinstance(argument, ast.Name) else ast.unparse(argument)
            for argument in goto_call.args[:2]
        )
        passed = (
            len(goto_call.args) >= 2
            and isinstance(goto_call.args[0], ast.Name)
            and isinstance(goto_call.args[1], ast.Name)
            and goto_call.args[0].id == pos_name
            and goto_call.args[1].id == quat_name
            and sample_index < goto_index
            and (first_close_index is None or goto_index < first_close_index)
        )
    return (
        passed,
        f"sample_binding={matching_sample}, first_goto_args={observed}, "
        f"first_close={first_close_index}",
    )


def _positive_z_vector(node: ast.AST, *, base_name: str | None = None) -> bool:
    vector = node
    if isinstance(node, ast.Call) and node.args:
        vector = node.args[0]
    if not isinstance(vector, (ast.List, ast.Tuple)) or len(vector.elts) < 3:
        return False
    z_value = vector.elts[2]
    value = _literal(z_value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return True
    if base_name is None or not isinstance(z_value, ast.BinOp) or not isinstance(z_value.op, ast.Add):
        return False

    def is_base_z(value_node: ast.AST) -> bool:
        return (
            isinstance(value_node, ast.Subscript)
            and isinstance(value_node.value, ast.Name)
            and value_node.value.id == base_name
            and _literal(value_node.slice) == 2
        )

    terms = ((z_value.left, z_value.right), (z_value.right, z_value.left))
    return any(
        is_base_z(base)
        and isinstance((offset := _literal(delta)), (int, float))
        and not isinstance(offset, bool)
        and offset > 0
        for base, delta in terms
    )


def _lift_transport_evidence(
    statements: Sequence[ast.stmt], calls: Sequence[ast.Call]
) -> tuple[bool, str]:
    assignments: dict[str, ast.AST] = {}
    positive_z_updates: set[str] = set()

    def collect(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = node.value
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Subscript)
            and isinstance(node.target.value, ast.Name)
            and _literal(node.target.slice) == 2
            and isinstance(node.op, ast.Add)
        ):
            value = _literal(node.value)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            ):
                positive_z_updates.add(node.target.value.id)
        for child in ast.iter_child_nodes(node):
            collect(child)

    for statement in statements:
        collect(statement)

    sample_position = None
    close_index = None
    transport_call = None
    for index, call in enumerate(calls):
        leaf = _call_name(call).split(".")[-1]
        if leaf == "sample_grasp_pose" and sample_position is None:
            query = _literal(call.args[0]) if call.args else None
            if isinstance(query, str) and "red" in query.lower() and "cube" in query.lower():
                for statement in statements:
                    for node in ast.walk(statement):
                        if (
                            isinstance(node, ast.Assign)
                            and node.value is call
                            and len(node.targets) == 1
                        ):
                            names = _tuple_names(node.targets[0])
                            if names and names[0] is not None:
                                sample_position = str(names[0])
        elif leaf == "close_gripper" and close_index is None:
            close_index = index
        elif (
            leaf == "goto_pose"
            and close_index is not None
            and index > close_index
            and transport_call is None
        ):
            transport_call = call

    def has_positive_offset(node: ast.AST, seen: set[str]) -> bool:
        if isinstance(node, ast.Name):
            if node.id in positive_z_updates:
                return True
            if node.id in seen or node.id not in assignments:
                return False
            return has_positive_offset(assignments[node.id], seen | {node.id})
        if _positive_z_vector(node, base_name=sample_position):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _positive_z_vector(node.left) or _positive_z_vector(node.right)
        return False

    target = transport_call.args[0] if transport_call is not None and transport_call.args else None
    origins = (
        _dependency_origins(target, {}) if target is not None else set()
    )
    if isinstance(target, ast.Name) and target.id in assignments:
        origins = _dependency_origins(assignments[target.id], {})
    sampled_position_used = (
        sample_position is not None
        and (
            (sample_position, None) in origins
            or {
                (sample_position, 0),
                (sample_position, 1),
                (sample_position, 2),
            }.issubset(origins)
        )
    )
    passed = (
        sample_position is not None
        and target is not None
        and sampled_position_used
        and has_positive_offset(target, set())
    )
    return (
        passed,
        f"sample_position={sample_position}, transport_target="
        f"{ast.unparse(target) if target is not None else None}, "
        f"origins={sorted(origins)}, positive_z_offset="
        f"{has_positive_offset(target, set()) if target is not None else False}",
    )


def _lift_gates(
    events: Sequence[CallEvent],
    statements: Sequence[ast.stmt],
    calls: Sequence[ast.Call],
) -> list[GateResult]:
    samples = _indices(events, "sample_grasp_pose")
    moves = _indices(events, "goto_pose")
    closes = _indices(events, "close_gripper")
    opens = _indices(events, "open_gripper")
    sample = _first(samples)
    close = _first(closes)
    pre_moves = [index for index in moves if close is not None and index < close]
    post_moves = [index for index in moves if close is not None and index > close]
    approach_events = [events[index] for index in pre_moves]
    sampled_pose_executed, sampled_pose_evidence = _sampled_grasp_execution_evidence(
        statements, calls, object_terms=("red", "cube")
    )
    upward_transport, upward_evidence = _lift_transport_evidence(statements, calls)
    return [
        GateResult(
            "target_grasp_observed",
            any(_event_has_text(event, ("red", "cube")) for event in events if event.name.split(".")[-1] == "sample_grasp_pose"),
            f"sample queries={[list(event.string_arguments) for event in events if event.name.split('.')[-1] == 'sample_grasp_pose']}",
        ),
        GateResult(
            "grasp_pose_observed_before_motion",
            sample is not None and bool(pre_moves) and sample < pre_moves[0],
            f"sample={sample}, precontact_moves={pre_moves}",
        ),
        GateResult(
            "sampled_grasp_pose_executed",
            sampled_pose_executed,
            sampled_pose_evidence,
        ),
        GateResult(
            "bounded_precontact_approach",
            any(_positive_approach(event) for event in approach_events),
            "positive z_approach before close" if approach_events else "no precontact move",
        ),
        GateResult(
            "contact_precedes_transport",
            bool(pre_moves and close is not None and post_moves and pre_moves[-1] < close < post_moves[0]),
            f"precontact_moves={pre_moves}, close={close}, transport_moves={post_moves}",
        ),
        GateResult(
            "single_metric_upward_transport",
            len(post_moves) == 1 and upward_transport,
            f"postcontact_moves={post_moves}; {upward_evidence}",
        ),
        GateResult(
            "contact_persists_through_lift",
            close is not None and not any(index > close for index in opens),
            f"close={close}, release_events={opens}",
        ),
    ]


def _tuple_names(target: ast.AST) -> list[str | None]:
    if not isinstance(target, (ast.Tuple, ast.List)):
        return []
    return [item.id if isinstance(item, ast.Name) else None for item in target.elts]


def _dependency_origins(
    node: ast.AST, dependencies: Mapping[str, set[tuple[str, int | None]]]
) -> set[tuple[str, int | None]]:
    if isinstance(node, ast.Name):
        return set(dependencies.get(node.id) or {(node.id, None)})
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        index = _literal(node.slice)
        if isinstance(index, int) and not isinstance(index, bool):
            return {(node.value.id, index)}
    origins: set[tuple[str, int | None]] = set()
    for child in ast.iter_child_nodes(node):
        origins.update(_dependency_origins(child, dependencies))
    return origins


def _stack_placement_evidence(
    statements: Sequence[ast.stmt], calls: Sequence[ast.Call]
) -> tuple[bool, str]:
    pose_names: dict[str, str] = {}
    extent_names: dict[str, str] = {}
    dependencies: dict[str, set[tuple[str, int | None]]] = {}
    placement_origins: set[tuple[str, int | None]] = set()
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            if isinstance(value, ast.Call) and _call_name(value).split(".")[-1] == "get_object_pose":
                names = _tuple_names(target)
                query = _literal(value.args[0]) if value.args else None
                if isinstance(query, str) and len(names) >= 1:
                    color = "red" if "red" in query.lower() else "green" if "green" in query.lower() else None
                    if color is not None and names[0] not in {None, "_"}:
                        pose_names[color] = str(names[0])
                    bbox = any(
                        keyword.arg == "return_bbox_extent"
                        and _literal(keyword.value) is True
                        for keyword in value.keywords
                    )
                    if color is not None and bbox and len(names) >= 3 and names[2] not in {None, "_"}:
                        extent_names[color] = str(names[2])
            origins = _dependency_origins(value, dependencies)
            if isinstance(target, ast.Name):
                dependencies[target.id] = origins
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                dependencies.setdefault(target.value.id, set()).update(origins)
        elif isinstance(statement, ast.AugAssign):
            target = statement.target
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                origins = _dependency_origins(statement.value, dependencies)
                dependencies.setdefault(target.value.id, set()).update(origins)

    close_index = next(
        (
            index
            for index, call in enumerate(calls)
            if _call_name(call).split(".")[-1] == "close_gripper"
        ),
        None,
    )
    release_index = next(
        (
            index
            for index, call in enumerate(calls)
            if close_index is not None
            and index > close_index
            and _call_name(call).split(".")[-1] == "open_gripper"
        ),
        None,
    )
    placement_calls = [
        call
        for index, call in enumerate(calls)
        if close_index is not None
        and index > close_index
        and (release_index is None or index < release_index)
        and _call_name(call).split(".")[-1] == "goto_pose"
        and call.args
    ]
    if placement_calls:
        placement_origins = _dependency_origins(
            placement_calls[-1].args[0], dependencies
        )

    green_pos = pose_names.get("green")
    red_extent = extent_names.get("red")
    green_extent = extent_names.get("green")
    required_extents = {(red_extent, 2), (green_extent, 2)}
    names_bound = all(value is not None for value in (green_pos, red_extent, green_extent))
    green_position_used = (green_pos, None) in placement_origins or {
        (green_pos, 0),
        (green_pos, 1),
        (green_pos, 2),
    }.issubset(placement_origins)
    passed = (
        names_bound
        and green_position_used
        and required_extents.issubset(placement_origins)
    )
    observed_text = sorted(f"{name}[{index}]" for name, index in placement_origins)
    return (
        passed,
        "bound green position / red extent / green extent are all used in placement; "
        f"bindings={(green_pos, red_extent, green_extent)}, observed={observed_text}",
    )


def _goto_position_names(calls: Sequence[ast.Call]) -> dict[int, str]:
    positions = {}
    for index, node in enumerate(calls):
        if (
            _call_name(node).split(".")[-1] == "goto_pose"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            positions[index] = node.args[0].id
    return positions


def _stack_gates(
    events: Sequence[CallEvent],
    statements: Sequence[ast.stmt],
    calls: Sequence[ast.Call],
) -> list[GateResult]:
    gets = [event for event in events if event.name.split(".")[-1] == "get_object_pose"]
    samples = [event for event in events if event.name.split(".")[-1] == "sample_grasp_pose"]
    moves = _indices(events, "goto_pose")
    close = _first(_indices(events, "close_gripper"))
    closes = _indices(events, "close_gripper")
    releases = [index for index in _indices(events, "open_gripper") if close is not None and index > close]
    release = _first(releases)
    pre_moves = [index for index in moves if close is not None and index < close]
    carried_moves = [
        index
        for index in moves
        if close is not None and index > close and (release is None or index < release)
    ]
    extent_queries = [
        event
        for event in gets
        if dict(event.keyword_literals).get("return_bbox_extent") is True
    ]
    semantic_placement, semantic_evidence = _stack_placement_evidence(statements, calls)
    sampled_pose_executed, sampled_pose_evidence = _sampled_grasp_execution_evidence(
        statements, calls, object_terms=("red", "cube")
    )
    goto_positions = _goto_position_names(calls)
    explicit_lift_moves = [
        index
        for index in carried_moves
        if goto_positions.get(index, "").startswith("lift")
    ]
    return [
        GateResult(
            "both_object_extents_observed",
            any(_event_has_text(event, ("red", "cube")) for event in extent_queries)
            and any(_event_has_text(event, ("green", "cube")) for event in extent_queries),
            f"extent query lines={[event.line for event in extent_queries]}",
        ),
        GateResult(
            "target_grasp_observed",
            any(_event_has_text(event, ("red", "cube")) for event in samples),
            f"sample queries={[list(event.string_arguments) for event in samples]}",
        ),
        GateResult(
            "sampled_grasp_pose_executed",
            sampled_pose_executed,
            sampled_pose_evidence,
        ),
        GateResult(
            "placement_uses_bound_metric_extents",
            semantic_placement,
            semantic_evidence,
        ),
        GateResult(
            "single_grasp_contact",
            len(closes) == 1,
            f"close_events={closes}",
        ),
        GateResult(
            "bounded_precontact_approach",
            any(_positive_approach(events[index]) for index in pre_moves),
            f"precontact_moves={pre_moves}",
        ),
        GateResult(
            "lift_and_place_after_contact",
            close is not None and len(carried_moves) >= 2,
            f"close={close}, carried_moves={carried_moves}",
        ),
        GateResult(
            "explicit_lift_pose_executed",
            bool(explicit_lift_moves),
            f"named goto targets={goto_positions}, lift_moves={explicit_lift_moves}",
        ),
        GateResult(
            "release_after_controlled_placement",
            close is not None and release is not None and len(carried_moves) >= 2,
            f"close={close}, carried_moves={carried_moves}, release={release}",
        ),
    ]


def _nut_semantic_evidence(
    statements: Sequence[ast.stmt], source: str
) -> tuple[bool, str, bool, str]:
    """Check grasp dataflow and executable rigid-pose conversions.

    The transform check is intentionally structural: a quaternion cannot be
    reshaped into a rotation matrix, and the sampled handle pose must be the
    first commanded grasp target.  This still remains source evidence, not an
    execution or geometry proof.
    """

    handle_position = None
    first_goto = None
    for statement in statements:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value).split(".")[-1] == "sample_grasp_pose"
        ):
            query = _literal(statement.value.args[0]) if statement.value.args else None
            names = _tuple_names(statement.targets[0])
            if (
                isinstance(query, str)
                and "handle" in query.lower()
                and "nut" in query.lower()
                and names
                and names[0] not in {None, "_"}
            ):
                handle_position = names[0]
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and _call_name(node).split(".")[-1] == "goto_pose"
                and first_goto is None
            ):
                first_goto = node

    first_target = (
        first_goto.args[0].id
        if first_goto is not None
        and first_goto.args
        and isinstance(first_goto.args[0], ast.Name)
        else None
    )
    grasp_dataflow = handle_position is not None and first_target == handle_position

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    call_names = [
        _call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    imports_scipy_rotation = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "scipy.spatial.transform"
        and any(alias.name == "Rotation" for alias in node.names)
        for node in ast.walk(tree)
    )
    invalid_reshape = any(name.split(".")[-1] == "reshape" for name in call_names)
    conversions = {
        name: sum(call_name.split(".")[-1] == name for call_name in call_names)
        for name in ("from_quat", "as_matrix", "from_matrix", "as_quat")
    }
    reorder_indices = {
        tuple(
            int(index.slice.value)
            for index in node.elts
            if isinstance(index, ast.Subscript)
            and isinstance(index.slice, ast.Constant)
            and isinstance(index.slice.value, int)
        )
        for node in ast.walk(tree)
        if isinstance(node, (ast.List, ast.Tuple))
    }
    pose_builders: set[str] = set()
    matrix_converters: set[str] = set()
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = [argument.arg for argument in function.args.args]
        if len(parameters) >= 2:
            pos_name, quat_name = parameters[:2]
            matrix_names = {
                node.targets[0].id
                for node in function.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value).split(".")[-1] == "eye"
                and node.value.args
                and _literal(node.value.args[0]) == 4
            }
            quat_reordered = any(
                isinstance(node, (ast.List, ast.Tuple))
                and len(node.elts) == 4
                and all(
                    isinstance(item, ast.Subscript)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == quat_name
                    and _literal(item.slice) == expected
                    for item, expected in zip(
                        node.elts, (1, 2, 3, 0), strict=True
                    )
                )
                for node in ast.walk(function)
            )
            rotation_written = any(
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in matrix_names
                and any(
                    isinstance(child, ast.Call)
                    and _call_name(child).split(".")[-1] == "as_matrix"
                    for child in ast.walk(node.value)
                )
                for node in function.body
            )
            translation_written = any(
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in matrix_names
                and isinstance(node.value, ast.Name)
                and node.value.id == pos_name
                for node in function.body
            )
            returned_matrix = any(
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id in matrix_names
                for node in function.body
            )
            if (
                quat_reordered
                and rotation_written
                and translation_written
                and returned_matrix
            ):
                pose_builders.add(function.name)

        if parameters:
            matrix_name = parameters[0]
            position_names = {
                node.targets[0].id
                for node in function.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == matrix_name
            }
            quaternion_names = {
                node.targets[0].id
                for node in function.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value).split(".")[-1] == "as_quat"
            }
            valid_return = any(
                isinstance(node, ast.Return)
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(node.value.elts) == 2
                and isinstance(node.value.elts[0], ast.Name)
                and node.value.elts[0].id in position_names
                and isinstance(node.value.elts[1], ast.Call)
                and _call_name(node.value.elts[1]).split(".")[-1] == "array"
                and node.value.elts[1].args
                and isinstance(node.value.elts[1].args[0], (ast.List, ast.Tuple))
                and len(node.value.elts[1].args[0].elts) == 4
                and any(
                    all(
                        isinstance(item, ast.Subscript)
                        and isinstance(item.value, ast.Name)
                        and item.value.id == quat_name
                        and _literal(item.slice) == expected
                        for item, expected in zip(
                            node.value.elts[1].args[0].elts,
                            (3, 0, 1, 2),
                            strict=True,
                        )
                    )
                    for quat_name in quaternion_names
                )
                for node in function.body
            )
            if position_names and quaternion_names and valid_return:
                matrix_converters.add(function.name)
    built_matrices: set[str] = set()
    desired_pose_converted = False
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        value = statement.value
        if not isinstance(value, ast.Call):
            continue
        function_name = _call_name(value).split(".")[-1]
        target = statement.targets[0]
        if isinstance(target, ast.Name) and function_name in pose_builders:
            built_matrices.add(target.id)
        if (
            function_name in matrix_converters
            and isinstance(target, (ast.Tuple, ast.List))
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == "T_desired_handle"
        ):
            desired_pose_converted = True
    valid_rotations = (
        imports_scipy_rotation
        and all(conversions[name] >= 1 for name in conversions)
        and (1, 2, 3, 0) in reorder_indices
        and (3, 0, 1, 2) in reorder_indices
        and {"T_handle", "T_nut", "T_peg"}.issubset(built_matrices)
        and desired_pose_converted
        and not invalid_reshape
    )
    return (
        grasp_dataflow,
        f"sampled_handle_position={handle_position}, first_goto_target={first_target}",
        valid_rotations,
        f"scipy_rotation_import={imports_scipy_rotation}, calls={conversions}, "
        f"quaternion_reorders={sorted(reorder_indices)}, pose_builders={sorted(pose_builders)}, "
        f"built_matrices={sorted(built_matrices)}, matrix_converters={sorted(matrix_converters)}, "
        f"desired_pose_converted={desired_pose_converted}, reshape_present={invalid_reshape}",
    )


def _exact_nut_transform(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def assignment(name: str) -> ast.AST | None:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name
            ):
                return node.value
        return None

    offset = assignment("T_handle_to_center")
    desired = assignment("T_desired_handle")
    offset_valid = (
        isinstance(offset, ast.BinOp)
        and isinstance(offset.op, ast.MatMult)
        and isinstance(offset.left, ast.Call)
        and _call_name(offset.left) == "np.linalg.inv"
        and len(offset.left.args) == 1
        and isinstance(offset.left.args[0], ast.Name)
        and offset.left.args[0].id == "T_nut"
        and isinstance(offset.right, ast.Name)
        and offset.right.id == "T_handle"
    )
    desired_valid = (
        isinstance(desired, ast.BinOp)
        and isinstance(desired.op, ast.MatMult)
        and isinstance(desired.left, ast.Name)
        and desired.left.id == "T_peg"
        and isinstance(desired.right, ast.Name)
        and desired.right.id == "T_handle_to_center"
    )
    return offset_valid and desired_valid


def _nut_gates(
    events: Sequence[CallEvent], source: str, statements: Sequence[ast.stmt]
) -> list[GateResult]:
    samples = [event for event in events if event.name.split(".")[-1] == "sample_grasp_pose"]
    gets = [event for event in events if event.name.split(".")[-1] == "get_object_pose"]
    moves = _indices(events, "goto_pose")
    closes = _indices(events, "close_gripper")
    close = _first(closes)
    release = _last(_indices(events, "open_gripper"))
    home = _first(_indices(events, "goto_home_joint_position"))
    post_moves = [index for index in moves if close is not None and index > close]
    rigid_transform = _exact_nut_transform(source)
    grasp_dataflow, grasp_evidence, valid_rotations, rotation_evidence = (
        _nut_semantic_evidence(statements, source)
    )
    return [
        GateResult(
            "handle_grasp_observed",
            any(_event_has_text(event, ("handle", "nut")) for event in samples),
            f"sample queries={[list(event.string_arguments) for event in samples]}",
        ),
        GateResult(
            "precontact_targets_sampled_handle",
            grasp_dataflow,
            grasp_evidence,
        ),
        GateResult(
            "bounded_precontact_approach",
            bool(moves) and _positive_approach(events[moves[0]]),
            f"first_goto={moves[0] if moves else None}",
        ),
        GateResult(
            "single_grasp_contact",
            len(closes) == 1,
            f"close_events={closes}",
        ),
        GateResult(
            "nut_and_peg_frames_observed",
            any(_event_has_text(event, ("nut",)) for event in gets)
            and any(
                _event_has_text(event, (term,))
                for event in gets
                for term in ("peg", "block")
            ),
            f"pose queries={[list(event.string_arguments) for event in gets]}",
        ),
        GateResult(
            "valid_rigid_rotation_conversion",
            valid_rotations,
            rotation_evidence,
        ),
        GateResult(
            "rigid_handle_to_center_transform",
            rigid_transform,
            "matrix inverse and composition are present"
            if rigid_transform
            else "missing explicit rigid transform",
        ),
        GateResult(
            "contact_precedes_insertion",
            close is not None and len(post_moves) >= 2,
            f"close={close}, postcontact_moves={post_moves}",
        ),
        GateResult(
            "home_reset_after_grasp",
            close is not None
            and home is not None
            and post_moves
            and close < home < post_moves[0],
            f"close={close}, home={home}, insertion_moves={post_moves}",
        ),
        GateResult(
            "release_after_insertion",
            close is not None and release is not None and post_moves and release > post_moves[-1],
            f"close={close}, insertion_moves={post_moves}, release={release}",
        ),
    ]


def _two_arm_lift_gates(
    events: Sequence[CallEvent], statements: Sequence[ast.stmt]
) -> list[GateResult]:
    get0 = _first(_indices(events, "get_handle0_pos") + _indices(events, "get_handle0_pose"))
    get1 = _first(_indices(events, "get_handle1_pos") + _indices(events, "get_handle1_pose"))
    open0 = _first(_indices(events, "open_gripper_arm0"))
    open1 = _first(_indices(events, "open_gripper_arm1"))
    goto0_calls = _indices(events, "goto_pose_arm0")
    goto1_calls = _indices(events, "goto_pose_arm1")
    goto0 = _first(goto0_calls)
    goto1 = _first(goto1_calls)
    close0_calls = _indices(events, "close_gripper_arm0")
    close1_calls = _indices(events, "close_gripper_arm1")
    close0 = _first(close0_calls)
    close1 = _first(close1_calls)
    together_calls = _indices(events, "goto_pose_both")
    together = _last(together_calls)
    first_together = _first(together_calls)
    frame_aware_approach = (
        _ordered(open0, goto0, close0)
        and _ordered(open1, goto1, close1)
        and goto0 is not None
        and goto1 is not None
        and isinstance(_approach_value(events[goto0]), (int, float))
        and isinstance(_approach_value(events[goto1]), (int, float))
        and abs(float(_approach_value(events[goto0])) + 0.02) <= 0.005
        and abs(float(_approach_value(events[goto1])) + 0.02) <= 0.005
    )
    combined_settle = (
        goto0 is not None
        and goto1 is not None
        and first_together is not None
        and close0 is not None
        and close1 is not None
        and max(goto0, goto1) < first_together < min(close0, close1)
    )
    return [
        GateResult(
            "both_handle_frames_observed",
            get0 is not None and get1 is not None,
            f"handle0={get0}, handle1={get1}",
        ),
        GateResult(
            "frame_aware_bilateral_approach",
            frame_aware_approach,
            f"arm0={(open0, goto0, close0)}, arm1={(open1, goto1, close1)}, "
            f"local_z_approach={(_approach_value(events[goto0]) if goto0 is not None else None, _approach_value(events[goto1]) if goto1 is not None else None)}, "
            f"goto_pose_both={together_calls}",
        ),
        GateResult(
            "simultaneous_precontact_settle",
            combined_settle,
            f"individual_approaches={(goto0, goto1)}, precontact_both={first_together}, "
            f"closes={(close0, close1)}",
        ),
        GateResult(
            "single_bilateral_contact",
            len(close0_calls) == 1 and len(close1_calls) == 1,
            f"arm0_closes={close0_calls}, arm1_closes={close1_calls}",
        ),
        GateResult(
            "simultaneous_transport_after_bilateral_contact",
            close0 is not None
            and close1 is not None
            and together is not None
            and together > max(close0, close1),
            f"close0={close0}, close1={close1}, goto_pose_both={together}",
        ),
    ] + _two_arm_lift_semantic_gates(statements)


def _numeric_vectors(
    statements: Sequence[ast.stmt], bindings: Mapping[str, Any]
) -> dict[str, tuple[float, ...]]:
    vectors = {}
    for statement in statements:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        value = statement.value
        if (
            isinstance(value, ast.Call)
            and _call_name(value).split(".")[-1] == "array"
            and value.args
        ):
            value = value.args[0]
        literal = _literal(value, bindings)
        if isinstance(literal, (list, tuple)) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in literal
        ):
            vectors[statement.targets[0].id] = tuple(float(item) for item in literal)
    return vectors


def _vector_argument(
    node: ast.AST, vectors: Mapping[str, tuple[float, ...]], bindings: Mapping[str, Any]
) -> tuple[float, ...] | None:
    if isinstance(node, ast.Name):
        return vectors.get(node.id)
    if (
        isinstance(node, ast.Call)
        and _call_name(node).split(".")[-1] == "array"
        and node.args
    ):
        literal = _literal(node.args[0], bindings)
        if isinstance(literal, (list, tuple)) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in literal
        ):
            return tuple(float(item) for item in literal)
    return None


def _vector_close(
    observed: tuple[float, ...] | None,
    expected: tuple[float, ...],
    *,
    tolerance: float = 0.01,
) -> bool:
    return observed is not None and len(observed) == len(expected) and all(
        abs(left - right) <= tolerance
        for left, right in zip(observed, expected, strict=True)
    )


def _two_arm_lift_semantic_gates(
    statements: Sequence[ast.stmt],
) -> list[GateResult]:
    bindings = _constant_bindings(statements)
    vectors = _numeric_vectors(statements, bindings)
    handles: dict[int, str] = {}
    dependencies: dict[str, set[tuple[str, int | None]]] = {}
    component_deltas: dict[tuple[str, int], float] = {}
    calls = []
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            if isinstance(target, ast.Name):
                dependencies[target.id] = _dependency_origins(value, dependencies)
                if isinstance(value, ast.Call):
                    function_name = _call_name(value).split(".")[-1]
                    if function_name in {"get_handle0_pos", "get_handle0_pose"}:
                        handles[0] = target.id
                        dependencies[target.id] = {(target.id, None)}
                    elif function_name in {"get_handle1_pos", "get_handle1_pose"}:
                        handles[1] = target.id
                        dependencies[target.id] = {(target.id, None)}
        elif (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Subscript)
            and isinstance(statement.target.value, ast.Name)
        ):
            index = _literal(statement.target.slice, bindings)
            delta = _literal(statement.value, bindings)
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and isinstance(delta, (int, float))
                and not isinstance(delta, bool)
            ):
                sign = -1.0 if isinstance(statement.op, ast.Sub) else 1.0
                key = (statement.target.value.id, index)
                component_deltas[key] = component_deltas.get(key, 0.0) + sign * float(
                    delta
                )
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                calls.append(node)
    calls.sort(key=lambda node: (int(node.lineno), int(node.col_offset)))
    arm_calls = {
        arm: [
            call
            for call in calls
            if _call_name(call).split(".")[-1] == f"goto_pose_arm{arm}"
        ]
        for arm in (0, 1)
    }
    both_calls = [
        call for call in calls if _call_name(call).split(".")[-1] == "goto_pose_both"
    ]

    def argument_name(call: ast.Call | None, index: int) -> str | None:
        if call is None or len(call.args) <= index or not isinstance(call.args[index], ast.Name):
            return None
        return call.args[index].id

    first0 = arm_calls[0][0] if arm_calls[0] else None
    first1 = arm_calls[1][0] if arm_calls[1] else None
    approach0 = argument_name(first0, 0)
    approach1 = argument_name(first1, 0)
    handle0 = handles.get(0)
    handle1 = handles.get(1)
    opposed_corridors = (
        approach0 is not None
        and approach1 is not None
        and handle0 is not None
        and handle1 is not None
        and (handle0, None) in dependencies.get(approach0, set())
        and (handle1, None) in dependencies.get(approach1, set())
        and 0.10 <= component_deltas.get((approach0, 1), 0.0) <= 0.20
        and -0.20 <= component_deltas.get((approach1, 1), 0.0) <= -0.10
    )
    quat0 = (
        _vector_argument(first0.args[1], vectors, bindings)
        if first0 is not None and len(first0.args) >= 2
        else None
    )
    quat1 = (
        _vector_argument(first1.args[1], vectors, bindings)
        if first1 is not None and len(first1.args) >= 2
        else None
    )
    mirrored_orientations = _vector_close(
        quat0, (0.7071, 0.7071, 0.0, 0.0)
    ) and _vector_close(quat1, (0.7071, -0.7071, 0.0, 0.0))

    settle = both_calls[0] if both_calls else None
    exact_settle = (
        argument_name(settle, 0) == handle0
        and argument_name(settle, 2) == handle1
    )
    lift = both_calls[-1] if len(both_calls) >= 2 else None
    lift0 = argument_name(lift, 0)
    lift1 = argument_name(lift, 2)
    matched_lift = (
        lift0 is not None
        and lift1 is not None
        and handle0 is not None
        and handle1 is not None
        and (handle0, None) in dependencies.get(lift0, set())
        and (handle1, None) in dependencies.get(lift1, set())
        and abs(component_deltas.get((lift0, 2), 0.0) - 0.20) <= 0.005
        and abs(component_deltas.get((lift1, 2), 0.0) - 0.20) <= 0.005
    )
    return [
        GateResult(
            "opposed_lateral_pregrasp_corridors",
            opposed_corridors,
            f"handles={(handle0, handle1)}, approaches={(approach0, approach1)}, "
            f"component_deltas={component_deltas}",
        ),
        GateResult(
            "mirrored_sideways_orientations",
            mirrored_orientations,
            f"observed_quaternions={(quat0, quat1)}",
        ),
        GateResult(
            "simultaneous_exact_handle_settle",
            exact_settle,
            f"settle_targets={(argument_name(settle, 0), argument_name(settle, 2))}, "
            f"handles={(handle0, handle1)}",
        ),
        GateResult(
            "matched_metric_bilateral_lift",
            matched_lift,
            f"lift_targets={(lift0, lift1)}, component_deltas={component_deltas}",
        ),
    ]


def _handover_semantic_gates(statements: Sequence[ast.stmt]) -> list[GateResult]:
    bindings = _constant_bindings(statements)
    vectors = _numeric_vectors(statements, bindings)
    handle_position = None
    dependencies: dict[str, set[tuple[str, int | None]]] = {}
    component_deltas: dict[tuple[str, int], float] = {}
    calls = []
    bounded_handover_height = False
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            if (
                isinstance(value, ast.Call)
                and _call_name(value).split(".")[-1] == "get_hammer_pose"
            ):
                names = _tuple_names(target)
                if names and names[0] not in {None, "_"}:
                    handle_position = names[0]
                    dependencies[str(handle_position)] = {(str(handle_position), None)}
            if isinstance(target, ast.Name):
                dependencies[target.id] = _dependency_origins(value, dependencies)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _literal(target.slice, bindings) in {0, 1, 2}
            ):
                assigned = _literal(value, bindings)
                if isinstance(assigned, (int, float)) and not isinstance(assigned, bool):
                    component_deltas[(target.value.id, int(_literal(target.slice)))] = float(
                        assigned
                    )
            if isinstance(target, ast.Name) and "handover" in target.id.lower():
                vector = _vector_argument(value, vectors, bindings)
                bounded_handover_height = bounded_handover_height or (
                    vector is not None and len(vector) >= 3 and 0.08 <= vector[2] <= 0.12
                )
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and "handover" in target.value.id.lower()
                and _literal(target.slice, bindings) == 2
            ):
                height = _literal(value, bindings)
                bounded_handover_height = bounded_handover_height or (
                    isinstance(height, (int, float))
                    and not isinstance(height, bool)
                    and 0.08 <= height <= 0.12
                )
        elif (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Subscript)
            and isinstance(statement.target.value, ast.Name)
            and _literal(statement.target.slice, bindings) in {0, 1, 2}
        ):
            index = int(_literal(statement.target.slice, bindings))
            delta = _literal(statement.value, bindings)
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                sign = -1.0 if isinstance(statement.op, ast.Sub) else 1.0
                key = (statement.target.value.id, index)
                component_deltas[key] = component_deltas.get(key, 0.0) + sign * float(
                    delta
                )
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                calls.append(node)
    calls.sort(key=lambda node: (int(node.lineno), int(node.col_offset)))
    arm0_calls = [
        call for call in calls if _call_name(call).split(".")[-1] == "goto_pose_arm0"
    ]
    arm1_calls = [
        call for call in calls if _call_name(call).split(".")[-1] == "goto_pose_arm1"
    ]
    pickup_position = (
        arm0_calls[0].args[0]
        if arm0_calls and len(arm0_calls[0].args) >= 2
        else None
    )
    pickup_origins = (
        _dependency_origins(pickup_position, dependencies)
        if pickup_position is not None
        else set()
    )
    pickup_name = pickup_position.id if isinstance(pickup_position, ast.Name) else None
    calibrated_pickup = (
        pickup_name is not None
        and abs(component_deltas.get((pickup_name, 0), 0.0)) <= 0.005
        and abs(component_deltas.get((pickup_name, 1), 0.0) + 0.035) <= 0.005
        and abs(component_deltas.get((pickup_name, 2), 0.0) + 0.025) <= 0.005
    )
    pickup_quaternion = (
        _vector_argument(arm0_calls[0].args[1], vectors, bindings)
        if arm0_calls and len(arm0_calls[0].args) >= 2
        else None
    )
    handover_calls = [
        call
        for call in arm0_calls[1:]
        if call.args
        and isinstance(call.args[0], ast.Name)
        and "handover" in call.args[0].id.lower()
    ]
    handover_quaternion = (
        _vector_argument(handover_calls[0].args[1], vectors, bindings)
        if handover_calls and len(handover_calls[0].args) >= 2
        else None
    )
    receiver_quaternion = (
        _vector_argument(arm1_calls[0].args[1], vectors, bindings)
        if arm1_calls and len(arm1_calls[0].args) >= 2
        else None
    )
    return [
        GateResult(
            "giver_grasp_targets_observed_handle",
            handle_position is not None
            and (str(handle_position), None) in pickup_origins
            and calibrated_pickup,
            f"handle_position={handle_position}, pickup={ast.unparse(pickup_position) if pickup_position else None}, "
            f"origins={sorted(pickup_origins)}, component_deltas={component_deltas}",
        ),
        GateResult(
            "task_aligned_handover_orientations",
            _vector_close(pickup_quaternion, (0.0, 0.707, 0.707, 0.0))
            and _vector_close(handover_quaternion, (0.0, 1.0, 0.0, 0.0))
            and _vector_close(receiver_quaternion, (0.0, 0.0, 1.0, 0.0)),
            "pickup / handover / receiver quaternions="
            f"{(pickup_quaternion, handover_quaternion, receiver_quaternion)}",
        ),
        GateResult(
            "bounded_absolute_handover_height",
            bounded_handover_height,
            "handover gripper z must be statically bounded in [0.08, 0.12] m",
        ),
    ]


def _handover_gates(
    events: Sequence[CallEvent], statements: Sequence[ast.stmt]
) -> list[GateResult]:
    observations = _indices(events, "get_hammer_pose") + _indices(events, "get_object_pose")
    open0 = _first(_indices(events, "open_gripper_arm0"))
    goto0 = _indices(events, "goto_pose_arm0")
    close0 = _first(_indices(events, "close_gripper_arm0"))
    open1 = _first(_indices(events, "open_gripper_arm1"))
    goto1 = _first(_indices(events, "goto_pose_arm1"))
    close1 = _first(_indices(events, "close_gripper_arm1"))
    release_candidates = [
        index
        for index in _indices(events, "open_gripper_arm0")
        if close0 is not None and index > close0
    ]
    release0 = _first(release_candidates)
    retract0 = [index for index in goto0 if release0 is not None and index > release0]
    first_giver_motion = _first(goto0)
    return _handover_semantic_gates(statements) + [
        GateResult(
            "hammer_frame_observed",
            bool(observations)
            and (first_giver_motion is None or min(observations) < first_giver_motion),
            f"observations={sorted(observations)}, first_arm0_motion={first_giver_motion}",
        ),
        GateResult(
            "giver_contact_precedes_transport",
            open0 is not None
            and goto0
            and close0 is not None
            and _ordered(open0, goto0[0], close0)
            and len([index for index in goto0 if index > close0]) >= 2,
            f"arm0_open={open0}, arm0_moves={goto0}, arm0_close={close0}",
        ),
        GateResult(
            "receiver_contact_precedes_giver_release",
            _ordered(open1, goto1, close1, release0),
            f"receiver={(open1, goto1, close1)}, giver_release={release0}",
        ),
        GateResult(
            "giver_retracts_after_transfer",
            len(retract0) == 1,
            f"giver_release={release0}, retract_moves={retract0}",
        ),
    ]


def audit_robot_program(
    source: str,
    *,
    task_family: str,
    coordinate_frame: str,
) -> ProgramAudit:
    """Audit a generated program against a frozen task-specific source contract."""

    if task_family not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task family: {task_family}")
    normalized = strip_code_fences(source)
    parse_error = None
    try:
        tree = ast.parse(normalized)
    except SyntaxError as exc:
        parse_error = f"{exc.msg} at line {exc.lineno}"
        tree = ast.Module(body=[], type_ignores=[])
    statements, initial_ambiguities = _execution_statements(tree)
    calls, ambiguous_lines = _scope_calls(statements, initial_ambiguities)
    events = _events(calls, _constant_bindings(statements))
    gates = _common_gates(
        events,
        parse_error=parse_error,
        coordinate_frame=coordinate_frame,
        ambiguous_lines=ambiguous_lines,
    )
    if task_family == "cube_lift":
        gates.extend(_lift_gates(events, statements, calls))
    elif task_family == "cube_stack":
        gates.extend(_stack_gates(events, statements, calls))
    elif task_family == "nut_assembly":
        gates.extend(_nut_gates(events, normalized, statements))
    elif task_family == "two_arm_lift":
        gates.extend(_two_arm_lift_gates(events, statements))
    else:
        gates.extend(_handover_gates(events, statements))
    accepted = all(gate.passed for gate in gates)
    return ProgramAudit(
        task_family=task_family,
        coordinate_frame=coordinate_frame,
        events=events,
        gates=tuple(gates),
        accepted=accepted,
        parse_error=parse_error,
        control_flow_ambiguities=tuple(ambiguous_lines),
        claim_boundary=(
            "Static source-order evidence only. Acceptance does not establish IK feasibility, "
            "collision safety, contact force, metric calibration, simulator success, or real-robot success."
        ),
    )


_CONTRACTS: Mapping[str, str] = {
    "cube_lift": (
        "Bind the grasp to the red cube; use a positive z_approach; close only after the "
        "approach; then move upward while keeping the gripper closed. Do not release during lift."
    ),
    "cube_stack": (
        "Begin with exactly `red_pos, red_quat, red_extent = get_object_pose(\"red cube\", "
        "return_bbox_extent=True)` and the analogous green_pos/green_quat/green_extent line; "
        "never one-value-unpack this three-value API. Compute place z from green_pos[2] + "
        "green_extent[2]/2 + red_extent[2]/2. Bind the grasp to the red cube; use a positive "
        "numeric z_approach; close exactly once; create a higher lift_pos and actually pass "
        "lift_pos to goto_pose before lateral placement; descend under control and release only "
        "at the computed height."
    ),
    "nut_assembly": (
        "Sample handle_pos/handle_quat and make the first goto_pose target exactly handle_pos with "
        "a positive z_approach; close exactly once, call goto_home_joint_position, and never close "
        "again. Bind `nut_pos, _ = get_object_pose(\"white hollow center of the brown square nut\")` "
        "and bind peg_pos/peg_quat from the square block. Define `pose_to_matrix(pos, quat)` to "
        "return a full np.eye(4) transform: reorder wxyz as [quat[1],quat[2],quat[3],quat[0]] for "
        "R.from_quat(...).as_matrix(), write rotation to [:3,:3], pos unchanged to [:3,3], and "
        "return the 4x4 matrix. Define `matrix_to_pose(matrix)` to return matrix[:3,3] unchanged "
        "plus R.from_matrix(matrix[:3,:3]).as_quat() reordered back as "
        "[q[3],q[0],q[1],q[2]]. Never reshape or permute XYZ. Set exactly "
        "`T_handle = pose_to_matrix(handle_pos, handle_quat)`, "
        "`T_nut = pose_to_matrix(nut_pos, handle_quat)`, and "
        "`T_peg = pose_to_matrix(peg_pos, peg_quat)`; "
        "the source must contain the exact compositions "
        "`T_handle_to_center = np.linalg.inv(T_nut) @ T_handle` and "
        "`T_desired_handle = T_peg @ T_handle_to_center`, then convert T_desired_handle back to "
        "a pose. Insert with one approach to desired_handle_pos, then set final_pos from a copy and "
        "subtract .02 m from final_pos[2], execute it, and release only after that move."
    ),
    "two_arm_lift": (
        "Observe handle0_pos and handle1_pos; use mirrored sideways quaternions q0=[.7071,.7071,"
        "0,0] and q1=[.7071,-.7071,0,0]. Copy approach0 from handle0_pos and add exactly +.15 m "
        "to approach0[1]; copy approach1 from handle1_pos and add exactly -.15 m to approach1[1]. "
        "Open both grippers. Move Arm 0 to approach0 and Arm 1 to approach1 with z_approach=-.02 "
        "because this sideways orientation makes the API approach axis local, not world +Z. Then "
        "call goto_pose_both on the original exact handle0_pos and handle1_pos to settle. Close each "
        "gripper exactly once only after that settle. Copy both original handle positions into "
        "lift0/lift1, add exactly +.20 m to both [2] components, and execute a second "
        "goto_pose_both only after both closes."
    ),
    "two_arm_handover": (
        "Bind exactly `handle_pos, _ = get_hammer_pose()`, then apply only the calibrated privileged "
        "handle corrections handle_pos[2] -= .025 and handle_pos[1] -= .035 (never change X). "
        "Arm 0's first grasp must use handle_pos with q_pick=[0,.707,.707,0] and z_approach=.15; "
        "never use the returned hammer quaternion. Lift first to handle_pos+[0,0,.10], then again "
        "to handle_pos+[0,0,.20] before translating. Use the arm midpoint x=.81, y=0 "
        "and gripper z=.10 for handover, rotate Arm 0 to [0, 1, 0, 0], then let Arm 1 approach "
        "at handover_pos+[.10,0,-.01] with [0,0,1,0] and z_approach=.12. Arm 1 must "
        "close before Arm 0 releases. The exact contact order after reaching handover is "
        "open_gripper_arm1 -> goto_pose_arm1 -> close_gripper_arm1 -> open_gripper_arm0; never "
        "open Arm 0 between its grasp close and Arm 1's close. Arm 0 must then make exactly one "
        "safe retract motion. Keep all poses in robot0 base frame."
    ),
}


def contract_prompt(
    task_family: str,
    coordinate_frame: str,
    *,
    max_nonblank_lines: int = 90,
    max_code_characters: int | None = None,
    representation: str = "code_string",
    transport: str = "json_schema",
) -> str:
    if task_family not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task family: {task_family}")
    if not coordinate_frame or ":" not in coordinate_frame:
        raise ValueError("coordinate_frame must be a named frame")
    if max_nonblank_lines <= 0:
        raise ValueError("max_nonblank_lines must be positive")
    if max_code_characters is not None and max_code_characters <= 0:
        raise ValueError("max_code_characters must be positive")
    if representation not in {
        "code_string",
        "phase_dataflow_object",
        "phase_statement_object",
        "statement_array",
    }:
        raise ValueError(f"unsupported contract representation: {representation}")
    if transport not in {"json_schema", "prompted_json"}:
        raise ValueError(f"unsupported contract transport: {transport}")
    character_clause = (
        ""
        if max_code_characters is None
        else f" and at most {max_code_characters} characters"
    )
    if representation == "code_string":
        output_clause = "Return executable Python only, without Markdown fences or prose."
    elif representation == "statement_array":
        output_clause = (
            "Return the complete Python program through the structured `statements` array. "
            "Put exactly one complete simple Python statement in each item; do not put comments, "
            "semicolons, embedded newlines, Markdown, or prose in any item."
        )
    elif representation == "phase_statement_object":
        output_clause = (
            "Return the complete Python program through the structured `phases` object. Put each "
            "model-authored simple Python statement in exactly one of the generic `observe`, "
            "`approach`, `contact`, `transport`, or `release` arrays. The arrays execute in that "
            "fixed causal order; use an empty array when a phase is not needed. Do not put "
            "comments, semicolons, embedded newlines, Markdown, or prose in any item."
        )
    else:
        output_clause = (
            "Return the complete Python program through the structured `phases` object. Put each "
            "operation in exactly one generic `observe`, `approach`, `contact`, `transport`, or "
            "`release` array. Each operation has a `bind` array of zero or more variable names and "
            "one Python `expression`: non-empty `bind` means assign the expression result to those "
            "names, while empty `bind` is allowed only for an action call. The compiler supplies "
            "only assignment syntax and fixed phase order; you must author every API call, argument, "
            "value, and data dependency. Generic format example only: `bind=[\"x\"]` with "
            "`expression=\"read(y)\"` compiles to `x = read(y)`, while `bind=[]` with "
            "`expression=\"act(x)\"` compiles to `act(x)`. Never put `=` inside `expression`, "
            "and never put bindings around an action. Do not put comments, semicolons, embedded "
            "newlines, Markdown, or prose inside `expression`."
        )
    if transport == "prompted_json":
        program_key = {
            "code_string": "code",
            "statement_array": "statements",
            "phase_statement_object": "phases",
            "phase_dataflow_object": "phases",
        }[representation]
        payload_description = {
            "code_string": (
                "The `code` value must be one JSON string containing the complete executable "
                "Python program, without Markdown fences or prose."
            ),
            "statement_array": (
                "The `statements` value must be the complete ordered array of simple Python "
                "statements described above."
            ),
            "phase_statement_object": (
                "The `phases` value must be the complete five-array phase object described above."
            ),
            "phase_dataflow_object": (
                "The `phases` value must be the complete five-array dataflow object described above."
            ),
        }[representation]
        output_clause = (
            "Return exactly one valid JSON object and nothing else. It must contain exactly "
            f"`task_family`, `coordinate_frame`, and `{program_key}`; set the first two values "
            f"exactly to `{task_family}` and `{coordinate_frame}`. {payload_description}"
        )
    return (
        "\n\nPhiAgent proof-carrying control contract (all clauses are hard gates):\n"
        f"- Named motion frame: {coordinate_frame}. Never mix it with camera pixels or another base.\n"
        f"- {_CONTRACTS[task_family]}\n"
        "- Use only the documented robot APIs; do not call env directly or use dynamic execution.\n"
        "- Keep the robot API sequence on one statically visible path: either a straight-line module "
        "or one no-argument main() body. Other helpers may perform math but must not call robot APIs.\n"
        "- Never repeat an action sequence. End the program immediately after the final "
        "required task motion.\n"
        f"- Keep the complete program concise (at most {max_nonblank_lines} nonblank lines"
        f"{character_clause}).\n"
        f"{output_clause}"
    )


def repair_prompt(source: str, audit: ProgramAudit) -> str:
    failures = [
        f"- {gate.gate_id}: {gate.evidence}" for gate in audit.gates if not gate.passed
    ]
    failure_text = "\n".join(failures) if failures else "- no failed gates; independently verify all gates"
    return (
        "Revise the candidate under the same API and task. Do not weaken or reinterpret any gate.\n"
        f"Task family: {audit.task_family}\nCoordinate frame: {audit.coordinate_frame}\n"
        f"Hard task contract (repeat exactly): {_CONTRACTS[audit.task_family]}\n"
        f"Audit findings:\n{failure_text}\n\nCandidate:\n{strip_code_fences(source)}\n\n"
        "Return the complete executable Python program in at most 90 lines, without Markdown fences "
        "or prose."
    )


__all__ = [
    "CallEvent",
    "GateResult",
    "ProgramAudit",
    "SCHEMA_VERSION",
    "SUPPORTED_TASKS",
    "audit_robot_program",
    "contract_prompt",
    "repair_prompt",
    "strip_code_fences",
]
