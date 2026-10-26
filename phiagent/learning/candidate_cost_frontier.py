"""Exact outcome-aware bounds for complete recorded candidate libraries, not routers."""

from __future__ import annotations

import math
from typing import Any


def binary_cost_frontier(rows: list[dict[str, Any]], profiles: tuple[str, ...]) -> dict:
    if not rows or len(profiles) < 2 or len(set(profiles)) != len(profiles):
        raise ValueError("nonempty tasks and distinct profiles are required")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate task identity")
    base_cost = 0
    base_successes = 0
    upgrades = []
    for row in rows:
        cells = {profile: row["cells"][profile] for profile in profiles}
        for cell in cells.values():
            if type(cell["score"]) is not int or cell["score"] not in (0, 1):
                raise ValueError("binary, observed scores are required")
            if type(cell["tokens"]) is not int or cell["tokens"] < 0:
                raise ValueError("exact nonnegative integer token usage is required")
        cheapest = min(
            profiles,
            key=lambda profile: (
                cells[profile]["tokens"], -cells[profile]["score"], profiles.index(profile)
            ),
        )
        baseline = cells[cheapest]
        base_cost += baseline["tokens"]
        base_successes += baseline["score"]
        winning = [profile for profile in profiles if cells[profile]["score"] == 1]
        if baseline["score"] == 0 and winning:
            winner = min(winning, key=lambda profile: (cells[profile]["tokens"], profiles.index(profile)))
            upgrades.append({
                "task_id": row["id"],
                "extra_tokens": cells[winner]["tokens"] - baseline["tokens"],
                "winning_profile": winner,
            })
    upgrades.sort(key=lambda item: (item["extra_tokens"], item["task_id"]))
    points = [{"successes": base_successes, "minimum_tokens": base_cost}]
    for upgrade in upgrades:
        points.append({
            "successes": points[-1]["successes"] + 1,
            "minimum_tokens": points[-1]["minimum_tokens"] + upgrade["extra_tokens"],
        })
    return {
        "tasks": len(rows),
        "minimum_possible_tokens": base_cost,
        "successes_at_minimum_tokens": base_successes,
        "maximum_possible_successes": points[-1]["successes"],
        "points": points,
        "oracle_only": True,
    }


def at_budget(frontier: dict, budget: int) -> dict:
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    affordable = [point for point in frontier["points"] if point["minimum_tokens"] <= budget]
    if not affordable:
        return {"feasible": False, "successes": None, "minimum_tokens": None, "budget": budget}
    return {"feasible": True, "budget": budget, **affordable[-1]}


def f1_integer_scale(rows: list[dict], profiles: tuple[str, ...]) -> int:
    denominators = []
    for row in rows:
        for profile in profiles:
            score = row["cells"][profile]["f1"]
            if (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1
            ):
                raise ValueError("finite observed per-task F1 in [0,1] is required")
            denominators.append(float(score).as_integer_ratio()[1])
    if not denominators:
        raise ValueError("a nonempty F1 matrix is required")
    # Binary floats have power-of-two denominators, so their largest is a common scale.
    return max(denominators)


def f1_units(value: float, scale: int) -> int:
    numerator, denominator = float(value).as_integer_ratio()
    if type(scale) is not int or scale <= 0 or scale % denominator:
        raise ValueError("the F1 scale must exactly represent every recorded score")
    return numerator * (scale // denominator)


def decode_policy(code: str, tasks: int, profiles: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not isinstance(code, str) or not code.isdecimal()
        or type(tasks) is not int or tasks <= 0
        or len(profiles) < 2 or len(set(profiles)) != len(profiles)
    ):
        raise ValueError("a nonnegative policy code, positive task count and distinct profiles are required")
    value = int(code)
    if value >= len(profiles) ** tasks:
        raise ValueError("policy code exceeds the complete task inventory")
    result = []
    for _ in range(tasks):
        value, index = divmod(value, len(profiles))
        result.append(profiles[index])
    return tuple(reversed(result))


def joint_cost_frontier(
    rows: list[dict], profiles: tuple[str, ...], budget: int, *, maximum_states: int = 250_000,
) -> dict:
    binary = binary_cost_frontier(rows, profiles)
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if type(maximum_states) is not int or maximum_states <= 0:
        raise ValueError("maximum_states must be a positive integer")
    scale = f1_integer_scale(rows, profiles)
    floor = binary["minimum_possible_tokens"]
    result = {
        "tasks": len(rows), "budget": budget, "minimum_possible_tokens": floor,
        "f1_scale": str(scale), "points": [], "states_per_task": [],
        "peak_unpruned_states": 0, "oracle_only": True,
        "f1_representation": "Exact integer sums of the binary floats in the recorded official per-task scores",
    }
    if budget < floor:
        return {**result, "status": "NO_POLICY_FITS_TOKEN_BUDGET", "feasible_budget": False}
    slack = budget - floor
    states = {(0, 0): (0, 0)}
    for row in rows:
        minimum = min(row["cells"][profile]["tokens"] for profile in profiles)
        options = [
            (row["cells"][profile]["score"],
             row["cells"][profile]["tokens"] - minimum,
             f1_units(row["cells"][profile]["f1"], scale))
            for profile in profiles
        ]
        successors: dict[tuple[int, int], tuple[int, int]] = {}
        for (em, extra), (score, code) in states.items():
            for index, (gain, cost, f1) in enumerate(options):
                if extra + cost > slack:
                    continue
                key = em + gain, extra + cost
                value = score + f1, code * len(profiles) + index
                previous = successors.get(key)
                if previous is None or value[0] > previous[0] or (
                    value[0] == previous[0] and value[1] < previous[1]
                ):
                    successors[key] = value
                if len(successors) > maximum_states:
                    raise RuntimeError(
                        f"Exact frontier exceeded {maximum_states} states at task {row['id']}; "
                        "no approximation or infeasibility conclusion is permitted"
                    )
        result["peak_unpruned_states"] = max(result["peak_unpruned_states"], len(successors))
        states = {}
        best_by_em: dict[int, int] = {}
        for key in sorted(successors):
            score, code = successors[key]
            if score > best_by_em.get(key[0], -1):
                states[key] = score, code
                best_by_em[key[0]] = score
        result["states_per_task"].append(len(states))
    result["points"] = [
        {"successes": em, "tokens": floor + extra, "f1_units": str(score), "policy_code": str(code)}
        for (em, extra), (score, code) in sorted(states.items())
    ]
    return {**result, "status": "EXACT_RECORDED_SCORE_FRONTIER_COMPLETE", "feasible_budget": True}
