"""Data-free definitions and bounded group statistics for the TRAIN format/cap diagnostic."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math


BASES = ("skillopt", "original_chisel")
CONDITIONS = ("unchanged", "answer_first", "generic_matched")
CAPS = (512, 2048)
ANSWER_FIRST = (
    "\n\nAnswer-order instruction: Begin the response with the requested answer inside "
    "<answer>...</answer>, with nothing before the opening tag. Keep the existing concise, "
    "context-supported answer-only output contract; do not add explanation outside the answer tags."
)
GENERIC_HEADER = "\n\nAdditional general instruction: Follow the task instructions carefully."
PADDING_UNIT = " x"
METRICS = ("exact_match", "f1", "prompt_tokens", "completion_tokens", "tokens")


def arm_name(base: str, condition: str, cap: int) -> str:
    if base not in BASES or condition not in CONDITIONS or type(cap) is not int or cap not in CAPS:
        raise ValueError("undeclared base, instruction or output cap")
    return f"{base}__{condition}__cap{cap}"


ARMS = tuple(arm_name(base, condition, cap)
             for base in BASES for condition in CONDITIONS for cap in CAPS)
FACTORS = {arm_name(base, condition, cap): (base, condition, cap)
           for base in BASES for condition in CONDITIONS for cap in CAPS}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def contrasts() -> dict[str, dict[str, int]]:
    result = {}
    for base in BASES:
        for cap in CAPS:
            for reference in ("generic_matched", "unchanged"):
                result[f"{base}/answer_first-minus-{reference}/cap{cap}"] = {
                    arm_name(base, "answer_first", cap): 1, arm_name(base, reference, cap): -1,
                }
        for condition in CONDITIONS:
            result[f"{base}/{condition}/cap2048-minus-cap512"] = {
                arm_name(base, condition, 2048): 1, arm_name(base, condition, 512): -1,
            }
        for reference in ("generic_matched", "unchanged"):
            result[f"{base}/instruction-by-cap/answer_first-minus-{reference}"] = {
                arm_name(base, "answer_first", 2048): 1,
                arm_name(base, reference, 2048): -1,
                arm_name(base, "answer_first", 512): -1,
                arm_name(base, reference, 512): 1,
            }
    return result


def randomized_order(cells: list[dict], seed: int, run_id: str) -> list[dict]:
    """Outcome-blind complete permutation, not phase-by-phase arm or cap execution."""
    if type(seed) is not int or not run_id or not cells:
        raise ValueError("a frozen order seed, distinct run identity and complete cells are required")
    pairs = {(cell["task"], cell["arm"]) for cell in cells}
    tasks = {cell["task"] for cell in cells}
    if (len(pairs) != len(cells) or pairs != {(task, arm) for task in tasks for arm in ARMS}
            or len({cell["cell"] for cell in cells}) != len(cells)):
        raise ValueError("the entire twelve-arm fresh-request matrix must be present exactly once")
    return sorted(cells, key=lambda cell: (
        canonical_hash(["format-cap-dispatch/v1", seed, run_id, cell["cell"]]), cell["cell"],
    ))


def factorial_statistics(scores: dict, groups: dict[str, str], *, context_length: int) -> dict:
    if (set(scores) != set(ARMS) or not groups
            or any(not isinstance(group, str) or not group for group in groups.values())
            or any(set(rows) != set(groups) for rows in scores.values())
            or type(context_length) is not int or context_length <= max(CAPS)):
        raise ValueError("complete matched source-question groups and a frozen context bound required")
    for arm, rows in scores.items():
        cap = FACTORS[arm][2]
        for row in rows.values():
            if (type(row["exact_match"]) is not int or row["exact_match"] not in (0, 1)
                    or type(row["f1"]) not in (float, int) or not math.isfinite(row["f1"])
                    or not 0 <= row["f1"] <= 1
                    or any(type(row[key]) is not int for key in METRICS[2:])
                    or row["prompt_tokens"] <= 0 or not 0 <= row["completion_tokens"] <= cap
                    or row["tokens"] != row["prompt_tokens"] + row["completion_tokens"]
                    or row["prompt_tokens"] + cap > context_length
                    or row["finish_reason"] not in {"stop", "length"}):
                raise ValueError("official scores, known exact usage or recorded finish reason invalid")
    sizes = Counter(groups.values())
    weight_squared = math.fsum((size / len(groups)) ** 2 for size in sizes.values())
    family_size = len(contrasts()) * len(METRICS)
    effects = {}
    for name, terms in contrasts().items():
        effects[name] = {}
        mass = sum(coefficient for coefficient in terms.values() if coefficient > 0)
        for metric in METRICS:
            values = {
                task: math.fsum(coefficient * scores[arm][task][metric]
                                for arm, coefficient in terms.items())
                for task in sorted(groups)
            }
            mean = math.fsum(values.values()) / len(groups)
            scale = 100 if metric in METRICS[:2] else 1
            single_bound = 1 if metric in METRICS[:2] else (
                max(CAPS) if metric == "completion_tokens" else context_length
            )
            bound = mass * single_bound
            radius = bound * math.sqrt(2 * weight_squared * math.log(2 * family_size / 0.05))
            effects[name][metric] = {
                "contrast_coefficients": terms, "mean_difference": scale * mean,
                "units": "percentage_points" if metric in METRICS[:2] else "tokens_per_task",
                "total_token_change": math.fsum(values.values()) if metric in METRICS[2:] else None,
                "simultaneous_bounded_interval": [
                    scale * max(-bound, mean - radius), scale * min(bound, mean + radius),
                ],
                "positive_tasks": sum(value > 0 for value in values.values()),
                "negative_tasks": sum(value < 0 for value in values.values()),
                "zero_tasks": sum(value == 0 for value in values.values()),
                "equivalence_or_noninferiority_certified": False,
            }
    return {
        "effects": effects, "tasks": len(groups), "question_groups": len(sizes),
        "group_size_histogram": {str(size): count for size, count in Counter(sizes.values()).items()},
        "official_qid_weighting_preserved": True,
        "family_size": family_size, "familywise_alpha": 0.05,
        "multiplicity": "Two-sided weighted question-group Hoeffding; Bonferroni over all 90 declared endpoints.",
        "length_stops": {arm: sum(row["finish_reason"] == "length" for row in rows.values())
                         for arm, rows in scores.items()},
        "length_stops_inferred_from_token_count": False,
        "assumptions": "Independent source question groups and outcome-independent fixed task weights; these assumptions are not certified by exposed TRAIN data.",
        "causal_scope": "Only the declared appended-text interventions and output-cap intervention within each frozen base/model. Generic text is not assumed neutral; cap and instruction are distinct factors.",
        "independent_confirmation": False, "official_full_benchmark": False,
        "model_pooling": False, "paper_admitted": False, "zero_bootstrap_certificate": False,
    }
