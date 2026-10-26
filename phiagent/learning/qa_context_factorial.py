"""Controlled training-only factorial comparison of representation and supervision."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
import math

from phiagent.learning.qa_context_sensitivity import (
    ContextSensitivityPredictor, ContextSensitivitySpec, fit_context_predictor,
)
from phiagent.learning.qa_interaction_compiler import FULL, MASKS, InstructionBank, InterventionTask
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_subset_selection import (
    MARGINS, _tasks, deployment_gate, fixed_controls, group_folds, summarize_assignments,
)


CELLS = {
    "legacy_uniform": ("legacy", 1.0),
    "full_uniform": ("full_context", 1.0),
    "legacy_sensitive": ("legacy", 8.0),
    "full_sensitive": ("full_context", 8.0),
}
PRIMARY = "full_sensitive"


def factorial_effects(cells: Mapping[str, Mapping]) -> dict:
    if set(cells) != set(CELLS):
        raise ValueError("all four factorial cells are required")
    reference = cells["legacy_uniform"]
    if any(
        row["population_sha256"] != reference["population_sha256"]
        or row["tasks"] != reference["tasks"]
        for row in cells.values()
    ):
        raise ValueError("factorial cells must cover the same paired population")
    result = {}
    for endpoint in ("exact_match", "f1", "mean_total_tokens"):
        values = {name: row[endpoint] for name, row in cells.items()}
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in values.values()):
            raise ValueError("factorial endpoints must be finite measured numbers")
        a, b, c, d = (values[name] for name in CELLS)
        result[endpoint] = {
            "full_context_at_uniform": b - a,
            "full_context_at_sensitive": d - c,
            "sensitivity_at_legacy": c - a,
            "sensitivity_at_full_context": d - b,
            "interaction_difference_of_differences": d - b - c + a,
        }
    return {
        "contrasts": result, "descriptive_only": True,
        "statistical_significance_established": False,
        "positive_token_difference_means_more_cost": True,
    }


def choose_assignments(
    model, tasks: Sequence[InterventionTask], *, margin: float,
) -> dict[str, str]:
    if type(margin) not in (int, float) or margin not in MARGINS:
        raise ValueError("use one predeclared margin")
    return {
        task.task_id: model.choose(
            question=task.question, context=task.context,
            prompt_tokens={mask: task.blocks[0].outcomes[mask].prompt_tokens for mask in MASKS},
            quality_margin=float(margin), repeat_guard=1.0,
        )["selected_mask"]
        for task in tasks
    }


def validate_exposure(model, tasks: Sequence[InterventionTask], *, training: bool) -> None:
    expected = {
        "task_hashes": {text_hash(task.task_id) for task in tasks},
        "group_hashes": {text_hash(task.group_id) for task in tasks},
        "input_hashes": {task.input_sha256 for task in tasks},
    }
    if any(task.bank_sha256 != model.artifact["bank_sha256"] for task in tasks):
        raise ValueError("a task binds a different instruction bank")
    for name, hashes in expected.items():
        observed = set(model.artifact["training"][name])
        if (training and observed != hashes) or (not training and observed & hashes):
            raise ValueError(f"model exposure violates the declared role: {name}")


def fit_factorial(
    tasks: Sequence[InterventionTask], *, bank: InstructionBank, seed: int,
    on_fold: Callable[[str, int, dict], None] | None = None,
    before_fit: Callable[[str, int], None] | None = None,
    legacy_routes: Mapping[str, Mapping[str, str]] | None = None,
) -> dict:
    values = _tasks(tasks, bank.sha256)
    folds = group_folds(values, folds=5, seed=seed)
    splits = [
        (
            tuple(task for task in values if folds[task.group_id] != fold),
            tuple(task for task in values if folds[task.group_id] == fold),
        )
        for fold in range(5)
    ]
    if any(
        {text_hash(task.context) for task in training}
        & {text_hash(task.context) for task in heldout}
        for training, heldout in splits
    ):
        raise ValueError("exact contexts cross the frozen training folds; no automatic regrouping")
    fixed_routes, reference_masks = {}, {}
    for fold, (training, heldout) in enumerate(splits):
        _, mask = fixed_controls(training)
        reference_masks[str(fold)] = mask
        fixed_routes.update({task.task_id: mask for task in heldout})
    references = (
        summarize_assignments(values, {task.task_id: FULL for task in values}),
        summarize_assignments(values, fixed_routes),
    )
    assignments, models, family = {}, {}, []
    for name, (representation, weight) in CELLS.items():
        spec = ContextSensitivitySpec(representation, sensitive_group_weight=weight)
        assignments[name] = {str(margin): {} for margin in MARGINS}
        models[name] = {}
        for fold, (training, heldout) in enumerate(splits):
            if before_fit is not None:
                before_fit(name, fold)
            artifact = fit_context_predictor(
                training, bank=bank, spec=spec, seed=seed + fold,
            )
            model = ContextSensitivityPredictor(artifact)
            validate_exposure(model, training, training=True)
            validate_exposure(model, heldout, training=False)
            for margin in MARGINS:
                assignments[name][str(margin)].update(
                    choose_assignments(model, heldout, margin=margin),
                )
            models[name][str(fold)] = model.model_sha256
            if on_fold is not None:
                on_fold(name, fold, artifact)
        if (
            name == "legacy_uniform" and legacy_routes is not None
            and assignments[name] != dict(legacy_routes)
        ):
            raise ValueError("legacy/uniform routes drifted from the previous frozen lexical MLP")
        for margin in MARGINS:
            summary = summarize_assignments(values, assignments[name][str(margin)])
            family.append({
                "cell": name, "spec": asdict(spec), "margin": margin,
                **summary, **deployment_gate(summary, references),
            })
    candidates = [row for row in family if row["cell"] == PRIMARY and row["eligible"]]
    winner = min(candidates, key=lambda row: (
        -row["exact_match"], -row["f1"],
        Fraction(*row["sum_task_mean_tokens_fraction"]), -row["margin"],
    )) if candidates else None
    report = {
        "schema_version": "phiagent-qa-context-sensitivity-training/v1",
        "status": "PRIMARY_TRAINING_POINT_PASS_PENDING_CALIBRATION" if winner
        else "PRIMARY_TRAINING_GATE_FAILED_STOP",
        "primary": PRIMARY, "selected": winner, "cell_margin_rows": family,
        "factorial_by_margin": {
            str(margin): factorial_effects({
                row["cell"]: row for row in family if row["margin"] == margin
            })
            for margin in MARGINS
        },
        "references": {
            "full_DELETE": references[0], "training_fold_selected_fixed": references[1],
        },
        "fold_reference_masks": reference_masks, "fold_model_sha256": models,
        "group_folds": {text_hash(group): fold for group, fold in folds.items()},
        "seed": seed, "folds": 5, "fits": len(CELLS) * 5,
        "legacy_uniform_prior_routes_reproduced": legacy_routes is not None,
        "calibration_used_for_training_or_selection": False,
        "control_promoted_as_primary": False,
        "benchmark_result": False, "paper_admitted": False,
    }
    report["selection_sha256"] = digest(report)
    return {"report": report, "assignments": assignments}
