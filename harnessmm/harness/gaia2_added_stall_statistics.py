"""Paired scenario-cluster summaries of already audited native outcomes."""
from __future__ import annotations

import math
import random


USAGE = ("prompt_tokens", "completion_tokens", "total_tokens")


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def compare(cases, expected_ids, *, resamples=10000, seed=20260928):
    identifiers = sorted(expected_ids)
    if (not identifiers or len(identifiers) != len(set(identifiers))
            or set(cases) != {"eager", "require_stall"}
            or any(set(rows) != set(identifiers) for rows in cases.values())):
        raise ValueError("Statistics require the complete paired scenario population")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("At least one prespecified cluster resample is required")
    totals = {}
    repetitions = {}
    for arm, rows in cases.items():
        totals[arm] = {
            "scenarios": len(identifiers), "executions": 3 * len(identifiers), "successes": 0,
            "usage": {role: dict.fromkeys(USAGE, 0) for role in ("actor", "judge")},
            "controller_executions": 0, "rejected_only_by_stall": 0,
        }
        for identifier in identifiers:
            case = rows[identifier]
            native = case["native_rows"]
            if (len(native) != 3
                    or any(type(row["run_number"]) is not int for row in native)
                    or {row["run_number"] for row in native} != {1, 2, 3}
                    or any(row["base_scenario_id"] != identifier or row["has_exception"]
                           or type(row["success_bool"]) is not bool for row in native)):
                raise ValueError("Unresolved or mismatched native repetitions")
            successes = sum(row["success_bool"] for row in native)
            if successes != case["successes"]:
                raise ValueError("Native success count differs")
            totals[arm]["successes"] += successes
            for role in ("actor", "judge"):
                usage = case["usage"][role]
                if (any(type(usage[key]) is not int or usage[key] < 0 for key in USAGE)
                        or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]):
                    raise ValueError("Provider input/output usage is incomplete")
                for key in USAGE:
                    totals[arm]["usage"][role][key] += usage[key]
            for key in ("controller_executions", "rejected_only_by_stall"):
                totals[arm][key] += case["controller"][key]
            for row in native:
                repetitions[(arm, identifier, row["run_number"])] = row["success_bool"]
        totals[arm]["success_percent"] = (
            100 * totals[arm]["successes"] / totals[arm]["executions"]
        )
        totals[arm]["all_provider_tokens"] = sum(
            totals[arm]["usage"][role]["total_tokens"] for role in ("actor", "judge")
        )
    differences = {"success_pp": [], "actor_tokens_per_execution": [],
                   "judge_tokens_per_execution": [], "all_provider_tokens_per_execution": []}
    for identifier in identifiers:
        eager, gated = (cases[arm][identifier] for arm in ("eager", "require_stall"))
        differences["success_pp"].append(100 * (eager["successes"] - gated["successes"]) / 3)
        for role in ("actor", "judge"):
            differences[role + "_tokens_per_execution"].append(
                (eager["usage"][role]["total_tokens"] - gated["usage"][role]["total_tokens"]) / 3
            )
        differences["all_provider_tokens_per_execution"].append(
            differences["actor_tokens_per_execution"][-1] + differences["judge_tokens_per_execution"][-1]
        )
    random_state = random.Random(seed)
    draws = {key: [] for key in differences}
    for _ in range(resamples):
        selected = [random_state.randrange(len(identifiers)) for _ in identifiers]
        for key, values in differences.items():
            draws[key].append(sum(values[index] for index in selected) / len(selected))
    deltas = {
        key: {"estimate": sum(values) / len(values),
              "percentile_95_ci": [percentile(draws[key], 0.025), percentile(draws[key], 0.975)]}
        for key, values in differences.items()
    }
    contingency = dict.fromkeys(("both_success", "eager_only", "gated_only", "both_failure"), 0)
    for identifier in identifiers:
        for run in (1, 2, 3):
            eager = repetitions[("eager", identifier, run)]
            gated = repetitions[("require_stall", identifier, run)]
            key = ("both_success" if eager and gated else "eager_only" if eager else
                   "gated_only" if gated else "both_failure")
            contingency[key] += 1
    success_delta = totals["eager"]["successes"] - totals["require_stall"]["successes"]
    token_delta = (totals["eager"]["usage"]["actor"]["total_tokens"]
                   - totals["require_stall"]["usage"]["actor"]["total_tokens"])
    if success_delta < 0 and token_delta < 0:
        interpretation = "Fewer actor tokens with lower native success: a quality/cost tradeoff."
    elif success_delta > 0 and token_delta > 0:
        interpretation = "Higher native success with more actor tokens: a quality/cost tradeoff."
    elif success_delta == 0 and token_delta < 0:
        interpretation = "Equal observed native success with fewer actor tokens."
    elif success_delta > 0 and token_delta <= 0:
        interpretation = "Higher observed native success with no increase in actor tokens."
    elif success_delta == 0 and token_delta == 0:
        interpretation = "No observed success or actor-token difference."
    else:
        interpretation = "Report the success and token differences separately."
    denominator = totals["require_stall"]["usage"]["actor"]["total_tokens"]
    return {
        "arms": totals, "contrast": "eager minus prospective require_stall",
        "successful_execution_difference": success_delta, "actor_token_difference": token_delta,
        "actor_token_saving_percent": -100 * token_delta / denominator if denominator else None,
        "paired_execution_contingency_descriptive_only": contingency,
        "scenario_cluster_bootstrap": {
            "clusters": len(identifiers), "repetitions_retained_per_cluster": 3,
            "resamples": resamples, "seed": seed, "differences": deltas,
        },
        "interpretation": interpretation,
        "optimization_tokens": 0, "all_failed_native_tasks_retained": True,
    }
