"""Dependency-free proof-carrying skill graphs for public robot suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
COORDINATE_FRAME = "robot_base:maniskill"
PHASES = ("observe", "approach", "contact", "transport", "release")
OPERATION_PHASES = {
    "approach": "approach",
    "bind": "observe",
    "close_gripper": "contact",
    "grasp": "contact",
    "lift": "transport",
    "move_above": "transport",
    "move_to_goal": "transport",
    "push": "transport",
    "release": "release",
}


@dataclass(frozen=True)
class StepContract:
    phase: str
    operation: str
    target: str
    minimum_value_m: float
    maximum_value_m: float


@dataclass(frozen=True)
class SkillGraphGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SkillGraphAudit:
    schema_version: str
    task_family: str
    coordinate_frame: str
    accepted: bool
    gates: tuple[SkillGraphGate, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gates"] = [asdict(gate) for gate in self.gates]
        return payload


TASK_CONTRACTS: dict[str, tuple[StepContract, ...]] = {
    "pick_cube": (
        StepContract("observe", "bind", "cube", 0.0, 0.0),
        StepContract("observe", "bind", "goal", 0.0, 0.0),
        StepContract("approach", "approach", "cube", 0.03, 0.08),
        StepContract("contact", "grasp", "cube", 0.015, 0.035),
        StepContract("transport", "move_to_goal", "goal", 0.0, 0.0),
    ),
    "stack_cube": (
        StepContract("observe", "bind", "cube_a", 0.0, 0.0),
        StepContract("observe", "bind", "cube_b", 0.0, 0.0),
        StepContract("approach", "approach", "cube_a", 0.03, 0.08),
        StepContract("contact", "grasp", "cube_a", 0.015, 0.035),
        StepContract("transport", "lift", "cube_a", 0.08, 0.15),
        StepContract("transport", "move_above", "cube_b", 0.0, 0.0),
        StepContract("release", "release", "cube_a", 0.0, 0.0),
    ),
    "push_cube": (
        StepContract("observe", "bind", "object", 0.0, 0.0),
        StepContract("observe", "bind", "goal", 0.0, 0.0),
        StepContract("approach", "approach", "object", 0.03, 0.08),
        StepContract("contact", "close_gripper", "object", 0.0, 0.0),
        StepContract("transport", "push", "goal", 0.08, 0.16),
    ),
}


TASK_INSTRUCTIONS = {
    "pick_cube": "Pick up the cube and move it to the marked 3-D goal pose.",
    "stack_cube": "Pick up cube A, stack it stably on cube B, and release it.",
    "push_cube": "Push the cube into the marked planar goal region.",
}


# These are the operations consumed by the public task adapter, not the tighter
# PCC proof contract. Baseline graphs may omit proof-only binds and may contain
# additional schema-valid operations, but the required actions must occur once
# in causal order and every proposed parameter remains subject to host safety.
TASK_RUNTIME_REQUIREMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "pick_cube": (
        ("approach", "cube"),
        ("grasp", "cube"),
        ("move_to_goal", "goal"),
    ),
    "stack_cube": (
        ("approach", "cube_a"),
        ("grasp", "cube_a"),
        ("lift", "cube_a"),
        ("move_above", "cube_b"),
        ("release", "cube_a"),
    ),
    "push_cube": (
        ("approach", "object"),
        ("close_gripper", "object"),
        ("push", "goal"),
    ),
}


def supported_task_families() -> tuple[str, ...]:
    return tuple(TASK_CONTRACTS)


def graph_json_schema(task_family: str) -> dict[str, Any]:
    if task_family not in TASK_CONTRACTS:
        raise ValueError(f"unsupported task family: {task_family}")
    step_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^s[1-9][0-9]*$"},
            "phase": {"type": "string", "enum": list(PHASES)},
            "operation": {
                "type": "string",
                "enum": sorted(
                    {step.operation for steps in TASK_CONTRACTS.values() for step in steps}
                ),
            },
            "target": {"type": "string"},
            "value_m": {"type": "number"},
        },
        "required": ["id", "phase", "operation", "target", "value_m"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "phiagent_proof_carrying_skill_graph",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "task_family": {"type": "string", "const": task_family},
                    "coordinate_frame": {"type": "string", "const": COORDINATE_FRAME},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "items": step_schema,
                    },
                    "proof": {
                        "type": "object",
                        "properties": {
                            "ordered_step_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 12,
                                "items": {"type": "string"},
                            },
                            "bound_targets": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {"type": "string"},
                            },
                            "terminal_condition": {
                                "type": "string",
                                "const": "official_env_success",
                            },
                            "coordinate_frame": {
                                "type": "string",
                                "const": COORDINATE_FRAME,
                            },
                        },
                        "required": [
                            "ordered_step_ids",
                            "bound_targets",
                            "terminal_condition",
                            "coordinate_frame",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["task_family", "coordinate_frame", "steps", "proof"],
                "additionalProperties": False,
            },
        },
    }


def generic_prompt(task_family: str, state_summary: Mapping[str, Any]) -> str:
    if task_family not in TASK_CONTRACTS:
        raise ValueError(f"unsupported task family: {task_family}")
    operations = sorted(
        {step.operation for steps in TASK_CONTRACTS.values() for step in steps}
    )
    value_docs = {
        "approach": "non-negative standoff distance from the named target",
        "bind": "0; establish a symbolic target binding",
        "close_gripper": "0; close without grasping the named object",
        "grasp": "non-negative grasp-depth offset relative to the object",
        "lift": "non-negative vertical displacement from the current pose",
        "move_above": "non-negative clearance above the named target",
        "move_to_goal": "non-negative terminal offset from the named goal",
        "push": "non-negative push distance toward the named goal",
        "release": "0; open the gripper to release the named object",
    }
    operation_api = {
        operation: {"phase": phase, "value_m": value_docs[operation]}
        for operation, phase in OPERATION_PHASES.items()
    }
    return (
        "Synthesize one robot skill graph for the public ManiSkill task below. "
        "Return only JSON matching the supplied response schema. Use symbolic targets "
        "from the state summary, an explicit metric value_m for every step, unique step "
        "IDs s1, s2, ..., and a proof that lists execution order and bound targets. "
        "value_m is always a relative metric action parameter, never an absolute target "
        "position copied from state. Bind every symbolic target referenced by later "
        "steps exactly once, order steps causally, and include only operations needed "
        "to complete this task; do not enumerate every available operation. "
        f"Available operations: {', '.join(operations)}.\n"
        "Shared operation API: "
        + json.dumps(operation_api, sort_keys=True, separators=(",", ":"))
        + "\n"
        f"Task family: {task_family}\n"
        f"Task: {TASK_INSTRUCTIONS[task_family]}\n"
        f"State ({COORDINATE_FRAME}): "
        + json.dumps(state_summary, sort_keys=True, separators=(",", ":"))
    )


def contract_prompt(task_family: str, state_summary: Mapping[str, Any]) -> str:
    base = generic_prompt(task_family, state_summary)
    contract = TASK_CONTRACTS[task_family]
    rows = [
        {
            "phase": step.phase,
            "operation": step.operation,
            "target": step.target,
            "minimum_value_m": step.minimum_value_m,
            "maximum_value_m": step.maximum_value_m,
        }
        for step in contract
    ]
    return (
        base
        + "\nProof-carrying contract: emit exactly these ordered semantic steps and keep "
        "each metric value inside its inclusive interval: "
        + json.dumps(rows, sort_keys=True, separators=(",", ":"))
        + ". The proof ordered_step_ids must exactly match step order; bound_targets "
        "must equal the distinct bind-operation targets; terminal_condition must be "
        "official_env_success. Do not add recovery branches or unbound targets."
    )


def reflection_prompt(candidate: str | None = None) -> str:
    del candidate  # The candidate is already the immediately preceding assistant message.
    return (
        "Independently review the candidate skill graph for task completion, correct "
        "symbolic target bindings, metric offsets, phase order, and consistency between "
        "steps and proof. In particular, remove unnecessary operations and ensure every "
        "value_m is a safe relative action distance rather than a state position. Return "
        "the complete corrected JSON only, using the same schema. The candidate to review "
        "is the immediately preceding assistant message."
    )


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> Sequence[Any] | None:
    return value if isinstance(value, list) else None


def audit_skill_graph(payload: object, task_family: str) -> SkillGraphAudit:
    if task_family not in TASK_CONTRACTS:
        raise ValueError(f"unsupported task family: {task_family}")
    graph = _mapping(payload)
    gates: list[SkillGraphGate] = []
    gates.append(
        SkillGraphGate(
            "json_object",
            graph is not None,
            "top-level JSON object required" if graph is None else "top-level object present",
        )
    )
    if graph is None:
        return SkillGraphAudit(
            SCHEMA_VERSION, task_family, "", False, tuple(gates)
        )

    exact_top_keys = set(graph) == {
        "task_family",
        "coordinate_frame",
        "steps",
        "proof",
    }
    gates.append(
        SkillGraphGate(
            "exact_top_level_keys",
            exact_top_keys,
            "exact graph keys present" if exact_top_keys else f"observed keys={sorted(graph)}",
        )
    )
    observed_task = str(graph.get("task_family", ""))
    gates.append(
        SkillGraphGate(
            "task_family_binding",
            observed_task == task_family,
            f"expected={task_family}, observed={observed_task}",
        )
    )
    coordinate_frame = str(graph.get("coordinate_frame", ""))
    gates.append(
        SkillGraphGate(
            "named_coordinate_frame",
            coordinate_frame == COORDINATE_FRAME,
            f"expected={COORDINATE_FRAME}, observed={coordinate_frame}",
        )
    )

    steps = _sequence(graph.get("steps"))
    expected = TASK_CONTRACTS[task_family]
    gates.append(
        SkillGraphGate(
            "step_count",
            steps is not None and len(steps) == len(expected),
            f"expected={len(expected)}, observed={len(steps) if steps is not None else 'non-list'}",
        )
    )
    normalized_steps: list[Mapping[str, Any]] = [
        step for step in (steps or ()) if isinstance(step, Mapping)
    ]
    step_objects = steps is not None and len(normalized_steps) == len(steps)
    exact_step_keys = step_objects and all(
        set(step) == {"id", "phase", "operation", "target", "value_m"}
        for step in normalized_steps
    )
    gates.append(
        SkillGraphGate(
            "exact_step_schema",
            exact_step_keys,
            "all steps use exact keys" if exact_step_keys else "step schema mismatch",
        )
    )
    ids = [str(step.get("id", "")) for step in normalized_steps]
    expected_ids = [f"s{index}" for index in range(1, len(expected) + 1)]
    gates.append(
        SkillGraphGate(
            "canonical_unique_step_ids",
            ids == expected_ids,
            f"expected={expected_ids}, observed={ids}",
        )
    )
    semantics_ok = len(normalized_steps) == len(expected) and all(
        str(step.get("phase", "")) == contract.phase
        and str(step.get("operation", "")) == contract.operation
        and str(step.get("target", "")) == contract.target
        for step, contract in zip(normalized_steps, expected, strict=False)
    )
    gates.append(
        SkillGraphGate(
            "ordered_task_semantics",
            semantics_ok,
            "task phase/operation/target sequence passed"
            if semantics_ok
            else "task phase/operation/target sequence mismatch",
        )
    )
    metric_values_ok = len(normalized_steps) == len(expected)
    metric_details = []
    for index, (step, contract) in enumerate(
        zip(normalized_steps, expected, strict=False), start=1
    ):
        value = step.get("value_m")
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        passed = numeric and contract.minimum_value_m <= float(value) <= contract.maximum_value_m
        metric_values_ok = metric_values_ok and passed
        metric_details.append(
            f"s{index}={value} in [{contract.minimum_value_m},{contract.maximum_value_m}]"
        )
    gates.append(
        SkillGraphGate(
            "metric_parameter_bounds",
            metric_values_ok,
            "; ".join(metric_details) if metric_details else "no metric values",
        )
    )

    proof = _mapping(graph.get("proof"))
    exact_proof_keys = proof is not None and set(proof) == {
        "ordered_step_ids",
        "bound_targets",
        "terminal_condition",
        "coordinate_frame",
    }
    gates.append(
        SkillGraphGate(
            "exact_proof_schema",
            exact_proof_keys,
            "proof keys passed" if exact_proof_keys else "proof schema mismatch",
        )
    )
    proof_ids = list(proof.get("ordered_step_ids", ())) if proof else []
    gates.append(
        SkillGraphGate(
            "proof_order_matches_steps",
            proof_ids == ids == expected_ids,
            f"proof={proof_ids}, steps={ids}",
        )
    )
    expected_bindings = sorted(
        {step.target for step in expected if step.operation == "bind"}
    )
    observed_bindings = proof.get("bound_targets") if proof else None
    binding_ok = (
        isinstance(observed_bindings, list)
        and len(observed_bindings) == len(set(observed_bindings))
        and sorted(str(value) for value in observed_bindings) == expected_bindings
    )
    gates.append(
        SkillGraphGate(
            "proof_bindings_complete",
            binding_ok,
            f"expected={expected_bindings}, observed={observed_bindings}",
        )
    )
    terminal_ok = bool(proof) and proof.get("terminal_condition") == "official_env_success"
    gates.append(
        SkillGraphGate(
            "official_terminal_condition",
            terminal_ok,
            str(proof.get("terminal_condition")) if proof else "missing proof",
        )
    )
    proof_frame_ok = bool(proof) and proof.get("coordinate_frame") == COORDINATE_FRAME
    gates.append(
        SkillGraphGate(
            "proof_frame_matches_graph",
            proof_frame_ok and coordinate_frame == COORDINATE_FRAME,
            str(proof.get("coordinate_frame")) if proof else "missing proof",
        )
    )
    return SkillGraphAudit(
        schema_version=SCHEMA_VERSION,
        task_family=task_family,
        coordinate_frame=coordinate_frame,
        accepted=all(gate.passed for gate in gates),
        gates=tuple(gates),
    )


def safe_runtime_parameters(payload: Mapping[str, Any]) -> bool:
    """Apply broad host-safety bounds to every arm before motion planning."""

    steps = payload.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            return False
        value = step.get("value_m")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 0.25
        ):
            return False
    return True


def skill_graph_executable(payload: object, task_family: str) -> tuple[bool, str]:
    """Compile semantic steps for every arm while ignoring proof-only fields.

    Raw and reflection baselines are allowed to execute a semantically complete
    graph even when proof-only binds are absent, extra safe operations are
    present, or their proof is malformed. Host-wide metric safety and causal
    task operations remain mandatory; the tighter task contract is a PCC
    promotion gate, not a hidden baseline filter.
    """

    if task_family not in TASK_CONTRACTS:
        raise ValueError(f"unsupported task family: {task_family}")
    if not isinstance(payload, Mapping):
        return False, "graph_not_object"
    if payload.get("task_family") != task_family:
        return False, "task_family_mismatch"
    if payload.get("coordinate_frame") != COORDINATE_FRAME:
        return False, "coordinate_frame_mismatch"
    steps = payload.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        return False, "invalid_step_count"
    allowed_targets = {step.target for step in TASK_CONTRACTS[task_family]}
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            return False, f"step_{index}_not_object"
        if step.get("id") != f"s{index}":
            return False, f"step_{index}_id_mismatch"
        operation = step.get("operation")
        if operation not in OPERATION_PHASES:
            return False, f"step_{index}_operation_unknown"
        if step.get("phase") != OPERATION_PHASES[operation]:
            return False, f"step_{index}_phase_mismatch"
        if step.get("target") not in allowed_targets:
            return False, f"step_{index}_target_unknown"
    if not safe_runtime_parameters(payload):
        return False, "runtime_parameter_safety_failed"
    required_indices = []
    for operation, target in TASK_RUNTIME_REQUIREMENTS[task_family]:
        matches = [
            index
            for index, step in enumerate(steps)
            if step.get("operation") == operation and step.get("target") == target
        ]
        if not matches:
            return False, f"required_step_missing:{operation}/{target}"
        if len(matches) != 1:
            return False, f"required_step_duplicated:{operation}/{target}"
        required_indices.append(matches[0])
    if required_indices != sorted(required_indices):
        return False, "required_steps_out_of_causal_order"
    return True, "runtime_task_adapter_compiled"


__all__ = [
    "COORDINATE_FRAME",
    "OPERATION_PHASES",
    "PHASES",
    "SCHEMA_VERSION",
    "TASK_CONTRACTS",
    "TASK_INSTRUCTIONS",
    "TASK_RUNTIME_REQUIREMENTS",
    "SkillGraphAudit",
    "SkillGraphGate",
    "StepContract",
    "audit_skill_graph",
    "contract_prompt",
    "generic_prompt",
    "graph_json_schema",
    "reflection_prompt",
    "safe_runtime_parameters",
    "skill_graph_executable",
    "supported_task_families",
]
