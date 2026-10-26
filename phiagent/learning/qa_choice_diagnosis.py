"""Pre-answer choice-risk features, loss accounting and budgeted diagnostic routing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType

from phiagent.learning.qa_context_sensitivity import context_features
from phiagent.learning.qa_interaction_compiler import FULL, MASKS, MODULE_IDS, InterventionTask
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_subset_selection import summarize_assignments


MODES = ("input_only", "input_router")
TOLERANCE = 1e-12
ROUTER_FEATURES = (
    *(f"retained_{name}" for name in MODULE_IDS),
    "retained_fraction", "log_full_input", "log_selected_input", "input_fraction",
    "predicted_total_fraction", "log_predicted_selected_output", "log_predicted_full_output",
    "predicted_em_regret", "predicted_f1_regret", "guarded_em_regret", "guarded_f1_regret",
    "predicted_em_regret_span", "predicted_f1_regret_span", "eligible_mask_fraction",
    "base_standardized_feature_rms", "base_standardized_feature_max_abs",
)


def feature_names(mode: str) -> tuple[str, ...]:
    if mode not in MODES:
        raise ValueError("risk features must be input_only or input_router")
    prefix = tuple(f"full_context_feature_{index:03d}" for index in range(404))
    return prefix if mode == "input_only" else (*prefix, *ROUTER_FEATURES)


def preanswer_features(
    *, question: str, context: str, native_inputs: Mapping[str, int], route: Mapping,
    feature_mean: Sequence[float], feature_scale: Sequence[float], mode: str,
) -> tuple[float, ...]:
    """Never accept actual answers, output usage, scores, labels or task IDs."""
    names = feature_names(mode)
    if (
        set(native_inputs) != set(MASKS)
        or any(type(value) is not int or value <= 0 for value in native_inputs.values())
        or route["selected_mask"] not in MASKS
        or route["runtime_outcomes_used"] is not False
        or route["quality_margin"] != 0.0 or route["repeat_guard"] != 1.0
        or route["input_sha256"] != digest({"question": question, "context": context})
        or set(route["estimates"]) != set(MASKS)
        or any(route["estimates"][mask]["prompt_tokens"] != native_inputs[mask] for mask in MASKS)
    ):
        raise ValueError("risk features require the fixed pre-answer router and exact native inputs")
    raw = context_features(question=question, context=context, representation="full_context")
    if len(raw) != 404 or len(feature_mean) != 404 or len(feature_scale) != 404:
        raise ValueError("base feature normalization must have the fixed dimension")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (*raw, *feature_mean, *feature_scale)
    ) or any(value <= 0 for value in feature_scale):
        raise ValueError("base features and normalization must be finite with positive scales")
    if mode == "input_only":
        return tuple(raw)
    estimates = route["estimates"]
    selected = route["selected_mask"]
    current, full = estimates[selected], estimates[FULL]
    for row in estimates.values():
        numbers = (
            row["predicted_completion_tokens"], row["estimated_total_tokens"],
            *row["predicted_regret"].values(), *row["guarded_regret"].values(),
        )
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers)
            or row["predicted_completion_tokens"] < 0 or row["estimated_total_tokens"] <= 0
            or type(row["eligible"]) is not bool
        ):
            raise ValueError("pre-answer model estimates are malformed")
    standardized = [(value - mean) / scale for value, mean, scale in zip(
        raw, feature_mean, feature_scale, strict=True,
    )]
    extra = [
        *(float(bit) for bit in selected),
        selected.count("1") / 5,
        math.log1p(native_inputs[FULL]), math.log1p(native_inputs[selected]),
        native_inputs[selected] / native_inputs[FULL],
        current["estimated_total_tokens"] / full["estimated_total_tokens"],
        math.log1p(current["predicted_completion_tokens"]),
        math.log1p(full["predicted_completion_tokens"]),
        current["predicted_regret"]["exact_match"], current["predicted_regret"]["f1"],
        current["guarded_regret"]["exact_match"], current["guarded_regret"]["f1"],
        *(max(row["predicted_regret"][endpoint] for row in estimates.values())
          - min(row["predicted_regret"][endpoint] for row in estimates.values())
          for endpoint in ("exact_match", "f1")),
        sum(row["eligible"] for row in estimates.values()) / len(MASKS),
        math.sqrt(math.fsum(value * value for value in standardized) / len(standardized)),
        max(abs(value) for value in standardized),
    ]
    result = (*raw, *extra)
    if len(result) != len(names) or not all(math.isfinite(value) for value in result):
        raise ValueError("risk feature construction returned invalid values")
    return result


@dataclass(frozen=True)
class ChoiceInput:
    task_id: str
    group_id: str
    input_sha256: str
    base_mask: str
    native_inputs: Mapping[str, int]
    input_only: tuple[float, ...]
    input_router: tuple[float, ...]
    base_model_sha256: str
    guard_slack_score: float

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.task_id, self.group_id, self.input_sha256, self.base_model_sha256,
        )):
            raise ValueError("pre-answer choice identities must be nonempty strings")
        if self.base_mask not in MASKS or set(self.native_inputs) != set(MASKS):
            raise ValueError("pre-answer choices require the complete source mask inventory")
        if any(type(value) is not int or value <= 0 for value in self.native_inputs.values()):
            raise ValueError("native input counts must be positive integers")
        for mode in MODES:
            values = self.features(mode)
            if len(values) != len(feature_names(mode)) or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in values
            ):
                raise ValueError("risk vectors must be finite and match their input-only schema")
        if type(self.guard_slack_score) not in (int, float) or not math.isfinite(self.guard_slack_score):
            raise ValueError("the pre-answer guard score must be finite")
        object.__setattr__(self, "native_inputs", MappingProxyType(dict(self.native_inputs)))
        object.__setattr__(self, "input_only", tuple(self.input_only))
        object.__setattr__(self, "input_router", tuple(self.input_router))

    @property
    def eligible(self) -> bool:
        return self.base_mask != FULL

    @property
    def extra_input_tokens(self) -> int:
        value = self.native_inputs[FULL] - self.native_inputs[self.base_mask]
        if self.eligible and value <= 0:
            raise ValueError("a pruned choice needs a positive native Full add-back cost")
        return value

    def features(self, mode: str) -> tuple[float, ...]:
        feature_names(mode)
        return self.input_only if mode == "input_only" else self.input_router


def make_choice(task: InterventionTask, model) -> ChoiceInput:
    native = {mask: task.blocks[0].outcomes[mask].prompt_tokens for mask in MASKS}
    route = model.choose(
        question=task.question, context=task.context, prompt_tokens=native,
        quality_margin=0.0, repeat_guard=1.0,
    )
    if route["model_sha256"] != model.model_sha256:
        raise ValueError("the pre-answer receipt belongs to a different base model")
    values = {
        mode: preanswer_features(
            question=task.question, context=task.context, native_inputs=native, route=route,
            feature_mean=model.artifact["feature_mean"], feature_scale=model.artifact["feature_scale"],
            mode=mode,
        )
        for mode in MODES
    }
    return ChoiceInput(
        task.task_id, task.group_id, task.input_sha256, route["selected_mask"], native,
        values["input_only"], values["input_router"], model.model_sha256,
        max(route["estimates"][route["selected_mask"]]["guarded_regret"].values()),
    )


def contrast(task: InterventionTask, mask: str) -> dict:
    if mask not in MASKS:
        raise ValueError("a source-bound mask is required")
    by_endpoint = {}
    for endpoint in ("exact_match", "f1"):
        differences = [
            getattr(block.outcomes[FULL], endpoint) - getattr(block.outcomes[mask], endpoint)
            for block in task.blocks
        ]
        mean = math.fsum(differences) / len(differences)
        middle = len(differences) // 2
        first = math.fsum(differences[:middle]) / middle
        second = math.fsum(differences[middle:]) / (len(differences) - middle)
        by_endpoint[endpoint] = {
            "full_minus_base_mean": mean,
            "loss_mass": max(mean, 0.0), "gain_mass": max(-mean, 0.0),
            "mean_harm": mean > TOLERANCE,
            "strict_harm_all_repeats": all(value > TOLERANCE for value in differences),
            "harm_first_half": first > TOLERANCE, "harm_second_half": second > TOLERANCE,
            "paired_repeat_losses": sum(value > TOLERANCE for value in differences),
            "paired_repeat_gains": sum(value < -TOLERANCE for value in differences),
        }
    return {
        "task_sha256": text_hash(task.task_id), "group_sha256": text_hash(task.group_id),
        "base_mask": mask,
        "dropped_modules": [name for name, bit in zip(MODULE_IDS, mask, strict=True) if bit == "0"],
        "endpoints": by_endpoint,
        "harm": int(any(row["mean_harm"] for row in by_endpoint.values())),
    }


def decompose(tasks: Sequence[InterventionTask], choices: Mapping[str, ChoiceInput]) -> dict:
    if set(choices) != {task.task_id for task in tasks}:
        raise ValueError("decomposition requires the complete population")
    cases = [contrast(task, choices[task.task_id].base_mask) for task in tasks]
    summary = {}
    for endpoint in ("exact_match", "f1"):
        rows = [case["endpoints"][endpoint] for case in cases]
        loss = math.fsum(row["loss_mass"] for row in rows)
        gain = math.fsum(row["gain_mass"] for row in rows)
        summary[endpoint] = {
            "gross_task_mean_loss": loss, "gross_task_mean_gain": gain,
            "net_task_mean_loss": loss - gain,
            "net_percentage_point_gap": 100 * (loss - gain) / len(tasks),
            "mean_harm_tasks": sum(row["mean_harm"] for row in rows),
            "strict_harm_all_repeats_tasks": sum(row["strict_harm_all_repeats"] for row in rows),
            "harm_in_both_halves_tasks": sum(
                row["harm_first_half"] and row["harm_second_half"] for row in rows
            ),
            "paired_repeat_losses": sum(row["paired_repeat_losses"] for row in rows),
            "paired_repeat_gains": sum(row["paired_repeat_gains"] for row in rows),
        }
    patterns = Counter(case["base_mask"] for case in cases if case["harm"])
    return {
        "tasks": len(tasks), "groups": len({task.group_id for task in tasks}),
        "pruned_tasks": sum(choice.eligible for choice in choices.values()),
        "any_quality_harm_tasks": sum(case["harm"] for case in cases),
        "endpoints": summary, "harm_by_base_mask": dict(sorted(patterns.items())),
        "cases": cases, "repeats_are_not_independent_tasks": True,
        "outcome_based_diagnosis_not_prediction": True,
    }


def addback_diagnosis(tasks: Sequence[InterventionTask], choices: Mapping[str, ChoiceInput]) -> dict:
    """Describe observed repairs of selected errors, without giving a predictor their labels."""
    counts = {
        name: {"harmful_cases_omitting_module": 0, "single_addback_restores_full_means": 0}
        for name in MODULE_IDS
    }
    minimum_repairs = Counter()
    for task in tasks:
        mask = choices[task.task_id].base_mask
        if not contrast(task, mask)["harm"]:
            continue
        means = {
            candidate: {
                endpoint: math.fsum(
                    getattr(block.outcomes[candidate], endpoint) for block in task.blocks
                ) / len(task.blocks)
                for endpoint in ("exact_match", "f1", "total_tokens")
            }
            for candidate in MASKS
        }
        for index, name in enumerate(MODULE_IDS):
            if mask[index] == "1":
                continue
            counts[name]["harmful_cases_omitting_module"] += 1
            restored = mask[:index] + "1" + mask[index + 1:]
            if all(means[restored][key] >= means[FULL][key] for key in ("exact_match", "f1")):
                counts[name]["single_addback_restores_full_means"] += 1
        feasible = [
            candidate for candidate in MASKS
            if all(old != "1" or new == "1" for old, new in zip(mask, candidate, strict=True))
            and all(means[candidate][key] >= means[FULL][key] for key in ("exact_match", "f1"))
        ]
        selected = min(feasible, key=lambda candidate: (
            candidate.count("1") - mask.count("1"), means[candidate]["total_tokens"], candidate,
        ))
        minimum_repairs[str(selected.count("1") - mask.count("1"))] += 1
    return {
        "by_module": counts, "minimum_added_modules_to_restore_full_means": dict(minimum_repairs),
        "conditioned_on_observed_wrong_choices": True,
        "module_opportunities_overlap_and_must_not_be_summed": True,
        "outcomes_used": True, "population_causal_effect_or_deployed_policy": False,
    }


def allocate_fallbacks(
    choices: Sequence[ChoiceInput], scores: Mapping[str, float], *,
    extra_input_fraction: float, seed: int,
) -> tuple[dict[str, str], dict]:
    if (
        not choices or len({row.task_id for row in choices}) != len(choices)
        or set(scores) != {row.task_id for row in choices if row.eligible}
        or type(extra_input_fraction) not in (int, float)
        or not math.isfinite(extra_input_fraction) or not 0 <= extra_input_fraction <= 0.2
        or type(seed) is not int
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores.values())
    ):
        raise ValueError("complete pre-answer choices, finite scores and a bounded budget are required")
    base_input = sum(row.native_inputs[row.base_mask] for row in choices)
    budget = int(Fraction(str(extra_input_fraction)) * base_input)
    routes = {row.task_id: row.base_mask for row in choices}
    spent, changed = 0, 0
    eligible = sorted((row for row in choices if row.eligible), key=lambda row: (
        -scores[row.task_id], digest({"seed": seed, "task": row.task_id}),
    ))
    for row in eligible:
        extra = row.extra_input_tokens
        if spent + extra <= budget:
            routes[row.task_id] = FULL
            spent += extra
            changed += 1
    return routes, {
        "base_native_input_tokens": base_input,
        "maximum_extra_native_input_tokens": budget,
        "used_extra_native_input_tokens": spent,
        "fallback_tasks": changed,
        "extra_input_fraction": float(extra_input_fraction),
        "actual_answers_or_output_usage_used": False,
        "pre_answer_single_actor_call": True,
    }


def mask_priors(choices: Sequence[ChoiceInput], labels: Mapping[str, int]) -> dict:
    eligible = [row for row in choices if row.eligible]
    if set(labels) != {row.task_id for row in eligible} or any(
        type(value) is not int or value not in (0, 1) for value in labels.values()
    ):
        raise ValueError("mask priors need complete binary inner-training labels")
    if not eligible:
        return {
            "status": "UNINFORMATIVE_NO_PRUNED_TRAINING_CHOICES",
            "scores": {mask: 0.0 for mask in MASKS}, "support": {},
        }
    groups = Counter(row.group_id for row in eligible)
    total_mass = math.fsum(1 / groups[row.group_id] for row in eligible)
    positive_mass = math.fsum(labels[row.task_id] / groups[row.group_id] for row in eligible)
    fallback = (positive_mass + 1) / (total_mass + 2)
    scores, support = {}, {}
    for mask in MASKS:
        rows = [row for row in eligible if row.base_mask == mask]
        mass = math.fsum(1 / groups[row.group_id] for row in rows)
        positive = math.fsum(labels[row.task_id] / groups[row.group_id] for row in rows)
        scores[mask] = (positive + 1) / (mass + 2) if rows else fallback
        support[mask] = {"tasks": len(rows), "uses_global_prior": not rows}
    return {"status": "TRAINING_ONLY_MASK_PRIOR", "scores": scores, "support": support}


def policy_summary(tasks, routes, references) -> dict:
    from phiagent.learning.qa_subset_selection import deployment_gate
    summary = summarize_assignments(tasks, routes)
    return {**summary, **deployment_gate(summary, references)}
