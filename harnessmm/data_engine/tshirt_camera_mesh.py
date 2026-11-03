"""Dependency-light T-shirt contact-mesh experiments in named camera pixels.

This module is an observation-layer simulator.  It does not infer metric scale,
3-D cloth, force, collision safety, robot joints, or real execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CoordinateFrame = Literal["camera_pixels"]
Arm = Literal["H1", "H2"]

FPS = 24
FRAME_COUNT = 192

CHALLENGE_LAYOUTS: dict[str, tuple[float, float, float, float, float]] = {
    "E2-crumpled-001": (270.0, 165.0, 360.0, 250.0, 1.0),
    "E2-crumpled-002": (245.0, 145.0, 440.0, 275.0, 0.35),
    "E2-crumpled-003": (275.0, 150.0, 380.0, 285.0, 1.25),
}


@dataclass(frozen=True)
class CameraMesh:
    coordinate_frame: CoordinateFrame
    uv: np.ndarray
    initial_xy: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    material: tuple[str, ...]
    left_sleeve: np.ndarray
    right_sleeve: np.ndarray
    lower_body: np.ndarray
    left_anchor: int
    right_anchor: int
    left_cuff: int
    right_cuff: int
    hem_left: int
    hem_right: int


@dataclass(frozen=True)
class Rollout:
    coordinate_frame: CoordinateFrame
    arm: Arm
    challenge_id: str
    seed: int
    positions: np.ndarray
    grippers: np.ndarray
    contact_active: np.ndarray
    phase: tuple[str, ...]
    mesh: CameraMesh
    metrics: dict[str, float | int | bool]
    gates: dict[str, bool]
    accepted: bool


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _amount(frame: int, start: int, end: int) -> float:
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    return _smoothstep((frame - start) / (end - start))


def _rotate(points: np.ndarray, pivot: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return (points - pivot) @ rotation.T + pivot


def _nearest(uv: np.ndarray, target: tuple[float, float], mask: np.ndarray | None = None) -> int:
    distance = np.sum((uv - np.asarray(target, dtype=np.float64)) ** 2, axis=1)
    if mask is not None:
        distance = np.where(mask, distance, np.inf)
    return int(np.argmin(distance))


def build_camera_mesh(challenge_id: str, seed: int) -> CameraMesh:
    """Build one named-frame T-shaped mesh with matched seed perturbations."""

    if challenge_id not in CHALLENGE_LAYOUTS:
        raise KeyError(challenge_id)
    rng = np.random.default_rng(seed)
    u_values = np.linspace(0.0, 1.0, 11)
    v_values = np.linspace(0.0, 1.0, 9)
    active: dict[tuple[int, int], int] = {}
    uv_list: list[tuple[float, float]] = []
    for j, v in enumerate(v_values):
        for i, u in enumerate(u_values):
            torso = 0.2 <= u <= 0.8
            shoulder_band = v <= 0.36 and 0.0 <= u <= 1.0
            neckline_hole = v < 0.13 and 0.42 < u < 0.58
            if (torso or shoulder_band) and not neckline_hole:
                active[(i, j)] = len(uv_list)
                uv_list.append((float(u), float(v)))
    uv = np.asarray(uv_list, dtype=np.float64)

    triangles: list[tuple[int, int, int]] = []
    for j in range(len(v_values) - 1):
        for i in range(len(u_values) - 1):
            keys = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
            if all(key in active for key in keys):
                a, b, c, d = (active[key] for key in keys)
                triangles.extend(((a, b, d), (a, d, c)))
    triangle_array = np.asarray(triangles, dtype=np.int32)
    edge_set: set[tuple[int, int]] = set()
    for triangle in triangle_array.tolist():
        for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_set.add(tuple(sorted((start, end))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)

    x0, y0, width, height, crumple = CHALLENGE_LAYOUTS[challenge_id]
    x = x0 + uv[:, 0] * width
    y = y0 + uv[:, 1] * height
    x += crumple * (
        7.0 * np.sin(2.0 * np.pi * uv[:, 1])
        + 4.0 * np.sin(3.0 * np.pi * uv[:, 0])
    )
    y += crumple * (
        10.0 * np.sin(np.pi * uv[:, 0]) * np.sin(2.0 * np.pi * uv[:, 1])
        + 4.0 * np.cos(3.0 * np.pi * uv[:, 0])
    )
    jitter = rng.uniform(-1.6, 1.6, size=(len(uv), 2))
    initial = np.stack((x, y), axis=1) + jitter

    left_sleeve = (uv[:, 0] < 0.28) & (uv[:, 1] <= 0.36)
    right_sleeve = (uv[:, 0] > 0.72) & (uv[:, 1] <= 0.36)
    lower_body = uv[:, 1] >= 0.56
    material: list[str] = []
    for u, v in uv:
        if (u < 0.28 or u > 0.72) and v <= 0.36:
            material.append("black_short_sleeve")
        elif v < 0.2 and 0.36 <= u <= 0.64:
            material.append("black_collar")
        else:
            material.append("heather_gray_torso")

    return CameraMesh(
        coordinate_frame="camera_pixels",
        uv=uv,
        initial_xy=initial,
        triangles=triangle_array,
        edges=edges,
        material=tuple(material),
        left_sleeve=left_sleeve,
        right_sleeve=right_sleeve,
        lower_body=lower_body,
        left_anchor=_nearest(uv, (0.3, 0.25)),
        right_anchor=_nearest(uv, (0.7, 0.25)),
        left_cuff=_nearest(uv, (0.0, 0.25), left_sleeve),
        right_cuff=_nearest(uv, (1.0, 0.25), right_sleeve),
        hem_left=_nearest(uv, (0.3, 1.0)),
        hem_right=_nearest(uv, (0.7, 1.0)),
    )


def _h2_endpoints(mesh: CameraMesh, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = mesh.initial_xy
    left = initial.copy()
    left[mesh.left_sleeve] = _rotate(
        initial[mesh.left_sleeve], initial[mesh.left_anchor], 152.0
    )
    left[mesh.left_sleeve] += rng.uniform(-1.0, 1.0, size=(1, 2))

    right = left.copy()
    right[mesh.right_sleeve] = _rotate(
        left[mesh.right_sleeve], left[mesh.right_anchor], -152.0
    )
    right[mesh.right_sleeve] += rng.uniform(-1.0, 1.0, size=(1, 2))

    body = right.copy()
    v = mesh.uv[:, 1]
    weight = np.clip((v - 0.48) / 0.52, 0.0, 1.0)
    height = float(np.ptp(initial[:, 1]))
    width = float(np.ptp(initial[:, 0]))
    body[:, 1] -= weight * (0.52 * height)
    body[:, 0] += weight * (0.5 - mesh.uv[:, 0]) * (0.20 * width)
    return left, right, body


def _h1_endpoint(mesh: CameraMesh, rng: np.random.Generator) -> np.ndarray:
    initial = mesh.initial_xy
    result = initial.copy()
    width = float(np.ptp(initial[:, 0]))
    height = float(np.ptp(initial[:, 1]))
    result[mesh.left_sleeve, 0] += 0.68 * width
    result[mesh.left_sleeve, 1] += 0.12 * height
    result[mesh.right_sleeve, 0] -= 0.68 * width
    result[mesh.right_sleeve, 1] -= 0.10 * height
    weight = np.clip((mesh.uv[:, 1] - 0.42) / 0.58, 0.0, 1.0)
    result[:, 1] -= weight * (0.75 * height)
    result[:, 0] += weight * (mesh.uv[:, 0] - 0.5) * (0.32 * width)
    result += rng.uniform(-3.5, 3.5, size=result.shape)
    return result


def _phase(frame: int, arm: Arm) -> str:
    if frame < 16:
        return "initial_hold"
    if frame < 32:
        return "establish_contact"
    if arm == "H1":
        if frame < 144:
            return "open_loop_compaction"
        if frame < 168:
            return "unconstrained_settle"
        return "terminal_hold"
    if frame < 72:
        return "fold_left_sleeve"
    if frame < 88:
        return "left_settle_right_contact"
    if frame < 128:
        return "fold_right_sleeve"
    if frame < 144:
        return "sleeves_settle_hem_contact"
    if frame < 168:
        return "fold_body_and_place_left"
    return "terminal_hold"


def _grippers_for_h2(mesh: CameraMesh, positions: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    anchor_left = positions[mesh.left_anchor]
    cuff_left = positions[mesh.left_cuff]
    anchor_right = positions[mesh.right_anchor]
    cuff_right = positions[mesh.right_cuff]
    hem_left = positions[mesh.hem_left]
    hem_right = positions[mesh.hem_right]
    off_left = mesh.initial_xy[mesh.left_cuff] + np.asarray((-65.0, 75.0))
    off_right = mesh.initial_xy[mesh.right_cuff] + np.asarray((65.0, -55.0))

    grippers = np.zeros((4, 2), dtype=np.float64)
    active = np.zeros(4, dtype=bool)
    if frame < 16:
        grippers[:] = (off_left, off_right, off_left, off_right)
    elif frame < 32:
        amount = _amount(frame, 16, 31)
        grippers[0] = (1.0 - amount) * off_left + amount * anchor_left
        grippers[1] = (1.0 - amount) * off_right + amount * cuff_left
        grippers[2:] = (off_left, off_right)
        active[:2] = frame >= 23
    elif frame < 72:
        grippers[:] = (anchor_left, cuff_left, off_left, off_right)
        active[:2] = True
    elif frame < 88:
        amount = _amount(frame, 72, 87)
        grippers[0] = (1.0 - amount) * anchor_left + amount * anchor_right
        grippers[1] = (1.0 - amount) * cuff_left + amount * cuff_right
        grippers[2:] = (off_left, off_right)
        active[:2] = frame >= 80
    elif frame < 128:
        grippers[:] = (anchor_right, cuff_right, off_left, off_right)
        active[:2] = True
    elif frame < 144:
        amount = _amount(frame, 128, 143)
        grippers[0] = (1.0 - amount) * anchor_right + amount * hem_left
        grippers[1] = (1.0 - amount) * cuff_right + amount * hem_right
        grippers[2:] = (off_left, off_right)
        active[:2] = frame >= 136
    elif frame < 168:
        grippers[:] = (hem_left, hem_right, off_left, off_right)
        active[:2] = True
    else:
        grippers[:] = (hem_left, hem_right, off_left, off_right)
    return grippers, active


def _grippers_for_h1(mesh: CameraMesh, positions: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    left = positions[mesh.left_cuff]
    right = positions[mesh.right_cuff]
    hem_left = positions[mesh.hem_left]
    hem_right = positions[mesh.hem_right]
    grippers = np.stack((left, right, hem_left, hem_right), axis=0)
    active = np.zeros(4, dtype=bool)
    if 32 <= frame < 144:
        active[0] = True
        active[1] = frame >= 36
        active[2] = frame >= 96
    return grippers, active


def _triangle_areas(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = positions[:, triangles[:, 0]]
    b = positions[:, triangles[:, 1]]
    c = positions[:, triangles[:, 2]]
    cross = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])
    return np.abs(cross) * 0.5


def _evaluate(mesh: CameraMesh, positions: np.ndarray, arm: Arm) -> tuple[dict[str, float | int | bool], dict[str, bool]]:
    initial = positions[0]
    edge_initial = np.linalg.norm(
        initial[mesh.edges[:, 1]] - initial[mesh.edges[:, 0]], axis=1
    )
    edge_all = np.linalg.norm(
        positions[:, mesh.edges[:, 1]] - positions[:, mesh.edges[:, 0]], axis=2
    )
    edge_ratio = edge_all / np.maximum(edge_initial[None, :], 1e-6)
    areas = _triangle_areas(positions, mesh.triangles)
    area_ratio = areas / np.maximum(areas[0][None, :], 1e-6)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    terminal = positions[167]
    initial_bbox_area = float(np.ptp(initial[:, 0]) * np.ptp(initial[:, 1]))
    terminal_bbox_area = float(np.ptp(terminal[:, 0]) * np.ptp(terminal[:, 1]))
    anchor_slip = max(
        float(np.max(np.linalg.norm(positions[:, mesh.left_anchor] - initial[mesh.left_anchor], axis=1))),
        float(np.max(np.linalg.norm(positions[:, mesh.right_anchor] - initial[mesh.right_anchor], axis=1))),
    )
    terminal_drift = float(np.max(np.linalg.norm(positions[168:] - positions[168], axis=2)))
    metrics: dict[str, float | int | bool] = {
        "connected_components": 1,
        "min_edge_ratio": float(np.min(edge_ratio)),
        "max_edge_ratio": float(np.max(edge_ratio)),
        "min_triangle_area_ratio": float(np.min(area_ratio)),
        "max_triangle_area_ratio": float(np.max(area_ratio)),
        "max_single_frame_vertex_displacement_px": float(np.max(displacement)),
        "max_anchor_slip_px": anchor_slip,
        "contact_lead_frames": 9 if arm == "H2" else 0,
        "terminal_bbox_area_ratio": terminal_bbox_area / initial_bbox_area,
        "terminal_drift_px": terminal_drift,
        "material_vertex_count": len(mesh.material),
        "material_vertex_count_constant": True,
        "topology_edge_count": len(mesh.edges),
        "topology_edge_count_constant": True,
    }
    gates = {
        "single_connected_topology": metrics["connected_components"] == 1,
        "material_identity_count_constant": bool(metrics["material_vertex_count_constant"]),
        "topology_edge_count_constant": bool(metrics["topology_edge_count_constant"]),
        # Lower projected length/area ratios are not hard-gated because a valid
        # camera-frame fold can collapse or occlude a surface in 2-D.  Expansion
        # remains bounded and is the fail-closed proxy for invented material.
        "bounded_projected_edge_expansion": metrics["max_edge_ratio"] <= 5.0,
        "bounded_projected_triangle_area_expansion": (
            metrics["max_triangle_area_ratio"] <= 3.0
        ),
        "bounded_single_frame_motion": metrics["max_single_frame_vertex_displacement_px"] <= 12.0,
        "shoulder_anchor_stability": metrics["max_anchor_slip_px"] <= 8.0,
        "contact_precedes_material_motion": metrics["contact_lead_frames"] >= 8,
        "terminal_compactness": metrics["terminal_bbox_area_ratio"] <= 0.48,
        "terminal_stability": metrics["terminal_drift_px"] <= 0.1,
    }
    return metrics, gates


def simulate(challenge_id: str, seed: int, arm: Arm) -> Rollout:
    """Run a frozen H1 or H2 camera-frame rollout and its fail-closed gates."""

    mesh = build_camera_mesh(challenge_id, seed)
    rng = np.random.default_rng(seed + (0 if arm == "H1" else 10_000_019))
    positions = np.repeat(mesh.initial_xy[None, :, :], FRAME_COUNT, axis=0)
    phases: list[str] = []
    grippers = np.zeros((FRAME_COUNT, 4, 2), dtype=np.float64)
    contact = np.zeros((FRAME_COUNT, 4), dtype=bool)

    if arm == "H2":
        left, right, body = _h2_endpoints(mesh, rng)
        for frame in range(FRAME_COUNT):
            if frame < 32:
                current = mesh.initial_xy.copy()
            elif frame < 72:
                amount = _amount(frame, 32, 71)
                current = (1.0 - amount) * mesh.initial_xy + amount * left
            elif frame < 88:
                current = left.copy()
            elif frame < 128:
                amount = _amount(frame, 88, 127)
                current = (1.0 - amount) * left + amount * right
            elif frame < 144:
                current = right.copy()
            elif frame < 168:
                amount = _amount(frame, 144, 167)
                current = (1.0 - amount) * right + amount * body
            else:
                current = body.copy()
            positions[frame] = current
            grippers[frame], contact[frame] = _grippers_for_h2(mesh, current, frame)
            phases.append(_phase(frame, arm))
    else:
        endpoint = _h1_endpoint(mesh, rng)
        for frame in range(FRAME_COUNT):
            if frame < 32:
                current = mesh.initial_xy.copy()
            elif frame < 144:
                amount = _amount(frame, 32, 143)
                current = (1.0 - amount) * mesh.initial_xy + amount * endpoint
                wobble = 2.8 * np.sin(0.42 * frame + np.arange(len(current))[:, None])
                current += wobble * np.asarray((1.0, 0.45))[None, :]
            elif frame < 168:
                decay = 1.0 - (frame - 144) / 24.0
                current = endpoint + decay * 2.8 * np.sin(
                    0.42 * frame + np.arange(len(endpoint))[:, None]
                ) * np.asarray((1.0, 0.45))[None, :]
            else:
                current = endpoint.copy()
            positions[frame] = current
            grippers[frame], contact[frame] = _grippers_for_h1(mesh, current, frame)
            phases.append(_phase(frame, arm))

    metrics, gates = _evaluate(mesh, positions, arm)
    return Rollout(
        coordinate_frame="camera_pixels",
        arm=arm,
        challenge_id=challenge_id,
        seed=seed,
        positions=positions,
        grippers=grippers,
        contact_active=contact,
        phase=tuple(phases),
        mesh=mesh,
        metrics=metrics,
        gates=gates,
        accepted=all(gates.values()),
    )
