#!/usr/bin/env python3
"""Evaluate one frozen ALFWorld prompt candidate for AutoSaddler.

This worker is intentionally benchmark-host only.  The caller supplies a
content-addressed request and a fresh output path.  All task selection happens
before this process starts; the worker only executes the exact ordered IDs.
"""

from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from phiagent.learning.alfworld_autosaddler_pfn import (
    TASK_FAMILIES,
    render_task_skill,
)

HANDCODED_EXPERT_GUARD_VERSION = "alfworld-unused-expert-index-guard/v1"


def _guarded_handcoded_act(original):
    """Keep ALFWorld's unused expert-info side channel from aborting episodes.

    ALFWorld calls the hand-coded expert on every environment step even though
    this evaluator consumes only admissible actions and ``won``.  Some valid
    states leave that expert with an empty ``objs_of_interest`` list, causing
    ``random.choice`` to raise ``IndexError``.  Returning ``look`` only affects
    ``extra.expert_plan``; the environment separately validates membership and
    the actor never reads that field.
    """

    @wraps(original)
    def guarded(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except IndexError:
            return "look"

    setattr(guarded, "_phiagent_guard_version", HANDCODED_EXPERT_GUARD_VERSION)
    return guarded


def install_alfworld_unused_expert_guard() -> dict[str, Any]:
    from alfworld.agents.expert import HandCodedTWAgent

    current = HandCodedTWAgent.act
    if getattr(current, "_phiagent_guard_version", None) != HANDCODED_EXPERT_GUARD_VERSION:
        HandCodedTWAgent.act = _guarded_handcoded_act(current)
    return {
        "version": HANDCODED_EXPERT_GUARD_VERSION,
        "target": "alfworld.agents.expert.HandCodedTWAgent.act",
        "caught_exception": "IndexError",
        "fallback_extra_expert_plan_action": "look",
        "actor_consumes_extra_expert_plan": False,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_alfworld_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_skill(
    *,
    surface: str,
    components: Mapping[str, str],
    task_type: str,
) -> str:
    if surface == "no_skill":
        if components:
            raise ValueError("no_skill candidate must have no components")
        return ""
    if surface == "global":
        if set(components) != {"global_skill"}:
            raise ValueError("global candidate schema differs")
        return components["global_skill"]
    if surface == "nested_router":
        rendered, _ = render_task_skill(components, task_type)
        return rendered
    raise ValueError(f"unknown candidate surface: {surface}")


def ordered_groups(
    task_ids: Sequence[str],
    items_by_id: Mapping[str, Mapping[str, Any]],
    surface: str,
) -> list[tuple[str, list[str]]]:
    if surface != "nested_router":
        return [("all", list(task_ids))]
    groups: dict[str, list[str]] = {family: [] for family in TASK_FAMILIES}
    for task_id in task_ids:
        task_type = str(items_by_id[task_id].get("task_type") or "")
        if task_type not in groups:
            raise ValueError(f"unknown ALFWorld task type: {task_type}")
        groups[task_type].append(task_id)
    return [(family, groups[family]) for family in TASK_FAMILIES if groups[family]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request_path = args.request.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    request = read_json(request_path)
    supplied_hash = request.pop("request_sha256", None)
    if supplied_hash != canonical_sha256(request):
        raise RuntimeError("request hash mismatch")
    if request.get("schema_version") != "phiagent-alfworld-candidate-eval/v1":
        raise RuntimeError("request schema differs")

    runner_path = Path(str(request["frozen_runner_path"])).resolve()
    runtime_manifest = Path(str(request["runtime_manifest_path"])).resolve()
    split_dir = Path(str(request["split_dir"])).resolve()
    rollout_root = Path(str(request["rollout_root"])).resolve()
    if rollout_root.exists():
        raise FileExistsError(rollout_root)
    rollout_root.mkdir(parents=True)

    task_ids = [str(value) for value in request["task_ids"]]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must be nonempty and unique")
    components = request["components"]
    if not isinstance(components, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in components.items()
    ):
        raise TypeError("components must be a string map")
    surface = str(request["surface"])

    items = read_json(split_dir / "val/items.json")
    items_by_id = {str(item["id"]): item for item in items}
    missing = [task_id for task_id in task_ids if task_id not in items_by_id]
    if missing:
        raise ValueError(f"task IDs absent from frozen split: {missing}")

    expert_guard = install_alfworld_unused_expert_guard()
    runner = load_runner(runner_path)
    runtime = read_json(runtime_manifest)
    adapter, tracker = runner.configure_runtime(
        runtime,
        base_url=str(request["base_url"]),
        split_dir=split_dir,
        out_root=rollout_root,
    )
    batch = adapter.build_eval_env(0, "valid_seen", int(runtime["execution"]["seed"]))
    batch = runner.filter_batch_ids(batch, task_ids)
    if list(batch.result_ids or []) != task_ids:
        raise RuntimeError("adapter changed frozen task order")

    started = time.monotonic()
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    route_records = []
    for route, route_ids in ordered_groups(task_ids, items_by_id, surface):
        routed_batch = runner.filter_batch_ids(batch, route_ids)
        task_type = (
            str(items_by_id[route_ids[0]]["task_type"])
            if route != "all"
            else str(items_by_id[route_ids[0]].get("task_type") or "")
        )
        skill = candidate_skill(
            surface=surface,
            components=components,
            task_type=task_type,
        )
        route_root = rollout_root / route.replace("::", "__")
        rows = adapter.rollout(routed_batch, skill, str(route_root))
        runner.validate_rollout(route_root, rows, route_ids)
        for row in rows:
            task_id = str(row.get("id") or "")
            if task_id in rows_by_id:
                raise RuntimeError(f"duplicate result ID: {task_id}")
            rows_by_id[task_id] = dict(row)
        route_records.append(
            {
                "route": route,
                "task_ids": route_ids,
                "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
            }
        )

    if set(rows_by_id) != set(task_ids):
        raise RuntimeError("worker result coverage differs")
    ordered_rows = []
    for task_id in task_ids:
        row = dict(rows_by_id[task_id])
        row["task_type"] = str(items_by_id[task_id].get("task_type") or "")
        ordered_rows.append(row)
    result = {
        "schema_version": "phiagent-alfworld-candidate-eval-result/v1",
        "implementation_label": request["implementation_label"],
        "phase": request["phase"],
        "surface": surface,
        "task_ids": task_ids,
        "rows": ordered_rows,
        "routes": route_records,
        "usage": tracker.summary(),
        "elapsed_seconds": time.monotonic() - started,
        "actor_model": runtime["model"]["request_model_id"],
        "base_url": str(request["base_url"]),
        "official_test_calls": int(request.get("official_test_calls", 0)),
        "infrastructure_guard": expert_guard,
    }
    result["canonical_sha256"] = canonical_sha256(result)
    write_json(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
