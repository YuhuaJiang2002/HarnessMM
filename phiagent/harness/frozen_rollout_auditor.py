"""Frozen state-derived auditor for camera-frame T-shirt rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phiagent.data_engine.tshirt_camera_mesh import CameraMesh, Rollout

GATE_IDS = (
    "single_connected_topology",
    "material_identity_count_constant",
    "topology_edge_count_constant",
    "bounded_projected_edge_expansion",
    "bounded_projected_triangle_area_expansion",
    "bounded_single_frame_motion",
    "shoulder_anchor_stability",
    "contact_precedes_material_motion",
    "terminal_compactness",
    "terminal_stability",
)


@dataclass(frozen=True)
class FrozenAuditContract:
    coordinate_frame: str
    initial_positions: np.ndarray
    expected_material_count: int
    expected_edge_count: int
    anchor_indices: tuple[int, int]

    @classmethod
    def from_mesh(cls, mesh: CameraMesh) -> FrozenAuditContract:
        return cls(
            coordinate_frame=mesh.coordinate_frame,
            initial_positions=mesh.initial_xy.copy(),
            expected_material_count=len(mesh.material),
            expected_edge_count=len(mesh.edges),
            anchor_indices=(mesh.left_anchor, mesh.right_anchor),
        )


@dataclass(frozen=True)
class AuditedRolloutState:
    coordinate_frame: str
    positions: np.ndarray
    edges: np.ndarray
    triangles: np.ndarray
    material: tuple[str, ...]
    contact_active: np.ndarray

    @classmethod
    def from_rollout(cls, rollout: Rollout) -> AuditedRolloutState:
        return cls(
            coordinate_frame=rollout.coordinate_frame,
            positions=rollout.positions.copy(),
            edges=rollout.mesh.edges.copy(),
            triangles=rollout.mesh.triangles.copy(),
            material=tuple(rollout.mesh.material),
            contact_active=rollout.contact_active.copy(),
        )


@dataclass(frozen=True)
class FrozenAuditResult:
    metrics: dict[str, float | int | bool]
    gates: dict[str, bool]

    @property
    def accepted(self) -> bool:
        return all(self.gates.values())

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(gate for gate in GATE_IDS if not self.gates[gate])


def _connected_components(vertex_count: int, edges: np.ndarray) -> int:
    parent = list(range(vertex_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for start, end in edges.tolist():
        left = find(int(start))
        right = find(int(end))
        if left != right:
            parent[right] = left
    return len({find(vertex) for vertex in range(vertex_count)})


def _triangle_areas(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = positions[:, triangles[:, 0]]
    b = positions[:, triangles[:, 1]]
    c = positions[:, triangles[:, 2]]
    cross = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
        c[..., 0] - a[..., 0]
    )
    return np.abs(cross) * 0.5


def _contact_lead_frames(positions: np.ndarray, contact_active: np.ndarray) -> int:
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    moving = np.flatnonzero(np.max(displacement, axis=1) > 1e-6)
    contacts = np.flatnonzero(np.any(contact_active, axis=1))
    if not len(moving):
        return positions.shape[0]
    if not len(contacts):
        return -positions.shape[0]
    first_motion_frame = int(moving[0]) + 1
    return first_motion_frame - int(contacts[0])


def _validate_state(contract: FrozenAuditContract, state: AuditedRolloutState) -> None:
    if state.coordinate_frame != contract.coordinate_frame:
        raise ValueError("rollout state frame does not match the frozen contract")
    if state.positions.ndim != 3 or state.positions.shape[2] != 2:
        raise ValueError("positions must have shape [frames, vertices, 2]")
    if state.positions.shape[0] < 169:
        raise ValueError("auditor requires at least 169 frames")
    if state.positions.shape[1:] != contract.initial_positions.shape:
        raise ValueError("rollout vertices do not match the frozen contract")
    if state.edges.ndim != 2 or state.edges.shape[1] != 2:
        raise ValueError("edges must have shape [edges, 2]")
    if state.triangles.ndim != 2 or state.triangles.shape[1] != 3:
        raise ValueError("triangles must have shape [triangles, 3]")
    if state.contact_active.shape[0] != state.positions.shape[0]:
        raise ValueError("contact trace must align with rollout frames")
    vertex_count = state.positions.shape[1]
    if (
        np.any(state.edges < 0)
        or np.any(state.edges >= vertex_count)
        or np.any(state.triangles < 0)
        or np.any(state.triangles >= vertex_count)
    ):
        raise ValueError("topology references vertices outside the frozen registry")


def audit_rollout_state(
    contract: FrozenAuditContract, state: AuditedRolloutState
) -> FrozenAuditResult:
    """Recompute every gate from state under a non-evolving contract."""

    _validate_state(contract, state)
    initial = contract.initial_positions
    edge_initial = np.linalg.norm(initial[state.edges[:, 1]] - initial[state.edges[:, 0]], axis=1)
    edge_all = np.linalg.norm(
        state.positions[:, state.edges[:, 1]] - state.positions[:, state.edges[:, 0]],
        axis=2,
    )
    edge_ratio = edge_all / np.maximum(edge_initial[None, :], 1e-6)
    initial_areas = _triangle_areas(initial[None, :, :], state.triangles)[0]
    areas = _triangle_areas(state.positions, state.triangles)
    area_ratio = areas / np.maximum(initial_areas[None, :], 1e-6)
    displacement = np.linalg.norm(np.diff(state.positions, axis=0), axis=2)
    terminal = state.positions[167]
    initial_bbox_area = float(np.ptp(initial[:, 0]) * np.ptp(initial[:, 1]))
    terminal_bbox_area = float(np.ptp(terminal[:, 0]) * np.ptp(terminal[:, 1]))
    anchor_slip = max(
        float(
            np.max(
                np.linalg.norm(
                    state.positions[:, anchor] - initial[anchor],
                    axis=1,
                )
            )
        )
        for anchor in contract.anchor_indices
    )
    terminal_drift = float(
        np.max(
            np.linalg.norm(
                state.positions[168:] - state.positions[168],
                axis=2,
            )
        )
    )
    metrics: dict[str, float | int | bool] = {
        "connected_components": _connected_components(state.positions.shape[1], state.edges),
        "material_vertex_count": len(state.material),
        "material_vertex_count_constant": (len(state.material) == contract.expected_material_count),
        "topology_edge_count": len(state.edges),
        "topology_edge_count_constant": (len(state.edges) == contract.expected_edge_count),
        "max_edge_ratio": float(np.max(edge_ratio)),
        "max_triangle_area_ratio": float(np.max(area_ratio)),
        "max_single_frame_vertex_displacement_px": float(np.max(displacement)),
        "max_anchor_slip_px": anchor_slip,
        "contact_lead_frames": _contact_lead_frames(state.positions, state.contact_active),
        "terminal_bbox_area_ratio": terminal_bbox_area / initial_bbox_area,
        "terminal_drift_px": terminal_drift,
    }
    gates = {
        "single_connected_topology": metrics["connected_components"] == 1,
        "material_identity_count_constant": bool(metrics["material_vertex_count_constant"]),
        "topology_edge_count_constant": bool(metrics["topology_edge_count_constant"]),
        "bounded_projected_edge_expansion": metrics["max_edge_ratio"] <= 5.0,
        "bounded_projected_triangle_area_expansion": (metrics["max_triangle_area_ratio"] <= 3.0),
        "bounded_single_frame_motion": (metrics["max_single_frame_vertex_displacement_px"] <= 12.0),
        "shoulder_anchor_stability": metrics["max_anchor_slip_px"] <= 8.0,
        "contact_precedes_material_motion": metrics["contact_lead_frames"] >= 8,
        "terminal_compactness": metrics["terminal_bbox_area_ratio"] <= 0.48,
        "terminal_stability": metrics["terminal_drift_px"] <= 0.1,
    }
    return FrozenAuditResult(metrics=metrics, gates=gates)


__all__ = [
    "AuditedRolloutState",
    "FrozenAuditContract",
    "FrozenAuditResult",
    "GATE_IDS",
    "audit_rollout_state",
]
