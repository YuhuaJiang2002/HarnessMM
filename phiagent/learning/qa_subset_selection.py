"""Train-only model selection for provisional input-conditioned instruction subsets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import re

from phiagent.learning.qa_interaction_compiler import (
    FULL, MASKS, InstructionBank, InterventionTask,
)
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_subset_predictors import (
    PredictorSpec, SubsetPredictor, fit_predictor,
)


MARGINS = (0.0, 0.01, 0.02, 0.05)
QUALITY = ("exact_match", "f1")


@dataclass(frozen=True)
class PredictorCandidate:
    candidate_id: str
    spec: PredictorSpec

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.candidate_id) is None
            or not isinstance(self.spec, PredictorSpec)
        ):
            raise ValueError("candidate requires a safe unique identifier and typed predictor spec")


def _tasks(tasks: Sequence[InterventionTask], bank_sha256: str) -> tuple[InterventionTask, ...]:
    values = tuple(tasks)
    if (
        not values or any(not isinstance(task, InterventionTask) for task in values)
        or len({task.task_id for task in values}) != len(values)
        or any(task.bank_sha256 != bank_sha256 for task in values)
    ):
        raise ValueError("complete unique typed tasks from one frozen instruction bank are required")
    owners = {}
    for task in values:
        if task.input_sha256 in owners and owners[task.input_sha256] != task.group_id:
            raise ValueError("identical inputs cannot cross source groups")
        owners[task.input_sha256] = task.group_id
    return tuple(sorted(values, key=lambda task: task.task_id))


def group_folds(
    tasks: Sequence[InterventionTask], *, folds: int = 5, seed: int = 0,
) -> dict[str, int]:
    values = tuple(tasks)
    if not values or not isinstance(values[0], InterventionTask):
        raise ValueError("typed intervention tasks are required before assigning folds")
    values = _tasks(values, values[0].bank_sha256)
    groups = {task.group_id for task in values}
    if type(seed) is not int or type(folds) is not int or not 2 <= folds <= len(groups):
        raise ValueError("an integer seed and at least one whole source group per fold are required")
    ordered = sorted(groups, key=lambda group: (digest({"seed": seed, "group": group}), group))
    return {group: index % folds for index, group in enumerate(ordered)}


def summarize_assignments(
    tasks: Sequence[InterventionTask], assignments: Mapping[str, str],
) -> dict:
    values = tuple(tasks)
    if not values or not isinstance(values[0], InterventionTask):
        raise ValueError("typed intervention tasks are required before summarizing assignments")
    values = _tasks(values, values[0].bank_sha256)
    if (
        set(assignments) != {task.task_id for task in values}
        or any(mask not in MASKS for mask in assignments.values())
    ):
        raise ValueError("assignments must cover every task exactly once with a source-bound mask")
    token_sum = sum((
        Fraction(sum(block.outcomes[assignments[task.task_id]].total_tokens
                     for block in task.blocks), len(task.blocks))
        for task in values
    ), Fraction())
    return {
        "tasks": len(values),
        "groups": len({task.group_id for task in values}),
        "population_sha256": digest([
            [task.task_id, task.group_id, task.input_sha256] for task in values
        ]),
        **{
            endpoint: math.fsum(
                math.fsum(getattr(block.outcomes[assignments[task.task_id]], endpoint)
                          for block in task.blocks) / len(task.blocks)
                for task in values
            ) / len(values)
            for endpoint in QUALITY
        },
        "mean_total_tokens": float(token_sum / len(values)),
        "sum_task_mean_tokens_fraction": [token_sum.numerator, token_sum.denominator],
        "route_counts": dict(sorted(Counter(assignments.values()).items())),
        "changed_tasks": sum(mask != FULL for mask in assignments.values()),
        "assignments_sha256": digest(dict(sorted(assignments.items()))),
    }


def fixed_controls(tasks: Sequence[InterventionTask]) -> tuple[dict, str]:
    values = tuple(tasks)
    fixed = {
        mask: summarize_assignments(values, {task.task_id: mask for task in values})
        for mask in MASKS
    }
    feasible = [
        mask for mask in MASKS
        if all(fixed[mask][endpoint] >= fixed[FULL][endpoint] for endpoint in QUALITY)
    ]
    best = min(feasible, key=lambda mask: (
        Fraction(*fixed[mask]["sum_task_mean_tokens_fraction"]), mask,
    ))
    return fixed, best


def deployment_gate(summary: Mapping, references: Sequence[Mapping]) -> dict:
    if len(references) != 2 or any(
        reference["tasks"] != summary["tasks"]
        or reference["population_sha256"] != summary["population_sha256"]
        for reference in references
    ):
        raise ValueError("both matched fixed references are required for the point gate")
    quality = all(
        summary[endpoint] >= reference[endpoint] for reference in references for endpoint in QUALITY
    )
    tokens = all(
        10 * Fraction(*summary["sum_task_mean_tokens_fraction"])
        <= 9 * Fraction(*reference["sum_task_mean_tokens_fraction"])
        for reference in references
    )
    adaptive = summary["changed_tasks"] > 0 and len(summary["route_counts"]) > 1
    return {
        "quality_point_gate": quality,
        "ten_percent_cost_point_gate": tokens,
        "multiple_subsets_used": adaptive,
        "eligible": quality and tokens and adaptive,
        "statistical_noninferiority_established": False,
    }


def _assign(
    predictor: SubsetPredictor, tasks: Sequence[InterventionTask], margin: float,
) -> dict[str, str]:
    return {
        task.task_id: predictor.choose(
            question=task.question, context=task.context,
            prompt_tokens={mask: task.blocks[0].outcomes[mask].prompt_tokens for mask in MASKS},
            quality_margin=margin, repeat_guard=1.0,
        )["selected_mask"]
        for task in tasks
    }


def select_predictor(
    tasks: Sequence[InterventionTask], *, bank: InstructionBank,
    candidates: Sequence[PredictorCandidate], folds: int = 5, seed: int = 0,
    on_fold: Callable[[str, int, dict], None] | None = None,
) -> dict:
    """Select a family and margin only from out-of-fold training predictions."""
    values = _tasks(tasks, bank.sha256)
    candidates = tuple(candidates)
    if (
        not 1 <= len(candidates) <= 8
        or any(not isinstance(candidate, PredictorCandidate) for candidate in candidates)
        or len({candidate.candidate_id for candidate in candidates}) != len(candidates)
    ):
        raise ValueError("declare one to eight distinct typed candidates before training")
    fold_ids = group_folds(values, folds=folds, seed=seed)
    splits = [
        (
            tuple(task for task in values if fold_ids[task.group_id] != fold),
            tuple(task for task in values if fold_ids[task.group_id] == fold),
        )
        for fold in range(folds)
    ]
    fixed_assignments = {}
    fold_references = {}
    for fold, (training, heldout) in enumerate(splits):
        _, mask = fixed_controls(training)
        fixed_assignments.update({task.task_id: mask for task in heldout})
        fold_references[str(fold)] = mask
    references = (
        summarize_assignments(values, {task.task_id: FULL for task in values}),
        summarize_assignments(values, fixed_assignments),
    )
    family, model_hashes, oof_assignments = [], {}, {}
    for candidate in candidates:
        assignments = {margin: {} for margin in MARGINS}
        model_hashes[candidate.candidate_id] = {}
        for fold, (training, heldout) in enumerate(splits):
            artifact = fit_predictor(
                training, bank=bank, spec=candidate.spec, seed=seed + fold,
            )
            predictor = SubsetPredictor(artifact)
            for key, expected in (
                ("task_hashes", {text_hash(task.task_id) for task in training}),
                ("group_hashes", {text_hash(task.group_id) for task in training}),
                ("input_hashes", {task.input_sha256 for task in training}),
            ):
                if set(predictor.artifact["training"][key]) != expected:
                    raise ValueError(f"fitted predictor exposure differs from its training fold: {key}")
            for margin in MARGINS:
                assignments[margin].update(_assign(predictor, heldout, margin))
            model_hashes[candidate.candidate_id][str(fold)] = predictor.model_sha256
            if on_fold is not None:
                on_fold(candidate.candidate_id, fold, artifact)
        oof_assignments[candidate.candidate_id] = {}
        for margin in MARGINS:
            summary = summarize_assignments(values, assignments[margin])
            family.append({
                "candidate_id": candidate.candidate_id,
                "spec": asdict(candidate.spec), "margin": margin,
                **summary, **deployment_gate(summary, references),
            })
            oof_assignments[candidate.candidate_id][str(margin)] = assignments[margin]
    feasible = [row for row in family if row["eligible"]]
    selected = (
        min(feasible, key=lambda row: (
            -row["exact_match"], -row["f1"],
            Fraction(*row["sum_task_mean_tokens_fraction"]),
            row["candidate_id"], -row["margin"],
        )) if feasible else None
    )
    result = {
        "schema_version": "phiagent-qa-subset-training-selection/v1",
        "status": "TRAINING_POLICY_SELECTED_PENDING_CALIBRATION" if selected else "TRAINING_SELECTION_FAILED_STOP",
        "selected": selected,
        "candidate_family": family,
        "fold_model_sha256": model_hashes,
        "fold_reference_masks": fold_references,
        "references": {
            "full_DELETE": references[0], "training_fold_selected_fixed": references[1],
        },
        "seed": seed, "folds": folds,
        "group_folds": {text_hash(group): fold for group, fold in sorted(fold_ids.items())},
        "training_task_hashes": [text_hash(task.task_id) for task in values],
        "training_group_hashes": sorted(text_hash(group) for group in fold_ids),
        "training_input_hashes": sorted({task.input_sha256 for task in values}),
        "bank_sha256": bank.sha256,
        "calibration_used_for_selection": False,
        "selection_rule": "Among feasible OOF policies: highest EM, highest F1, lowest tokens, candidate ID, largest margin.",
        "benchmark_result": False, "paper_admitted": False,
    }
    result["selection_sha256"] = digest(result)
    return {"report": result, "oof_assignments": oof_assignments}


def evaluate_frozen_predictor(
    artifact: Mapping, tasks: Sequence[InterventionTask], *, margin: float,
) -> dict:
    """Evaluate one frozen margin; no calibration-driven family or threshold search."""
    if type(margin) not in (int, float) or margin not in MARGINS:
        raise ValueError("a margin frozen from the declared training family is required")
    predictor = SubsetPredictor(artifact)
    values = _tasks(tasks, predictor.artifact["bank_sha256"])
    for key, hashes in (
        ("task_hashes", {text_hash(task.task_id) for task in values}),
        ("group_hashes", {text_hash(task.group_id) for task in values}),
        ("input_hashes", {task.input_sha256 for task in values}),
    ):
        if hashes.intersection(predictor.artifact["training"][key]):
            raise ValueError(f"training and calibration overlap in {key}")
    assignments = _assign(predictor, values, float(margin))
    summary = summarize_assignments(values, assignments)
    fixed, best = fixed_controls(values)
    gate = deployment_gate(summary, (fixed[FULL], fixed[best]))
    return {
        "schema_version": "phiagent-qa-subset-frozen-calibration/v1",
        "status": "FROZEN_CALIBRATION_POINT_PASS_PENDING_OFFICIAL" if gate["eligible"]
        else "FROZEN_CALIBRATION_FAILED_STOP",
        "model_sha256": predictor.model_sha256,
        "policy_sha256": digest({
            "model_sha256": predictor.model_sha256, "quality_margin": float(margin),
            "repeat_guard": 1.0,
        }),
        "margin": float(margin), **summary, **gate,
        "fixed_controls": fixed, "best_feasible_fixed_mask": best,
        "assignments": assignments,
        "calibration_group_hashes": sorted({text_hash(task.group_id) for task in values}),
        "candidate_refitted_or_reselected": False,
        "benchmark_result": False, "paper_admitted": False,
    }
