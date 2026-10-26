"""Nested input-only identification of a fixed router's harmful subset choices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
import math

from phiagent.learning.qa_choice_diagnosis import (
    MODES, addback_diagnosis, allocate_fallbacks, contrast, decompose, feature_names, make_choice, mask_priors,
    policy_summary,
)
from phiagent.learning.qa_choice_risk import (
    ChoiceRiskPredictor, binary_ranking_metrics, fit_choice_risk,
)
from phiagent.learning.qa_context_factorial import validate_exposure
from phiagent.learning.qa_context_sensitivity import (
    ContextSensitivityPredictor, ContextSensitivitySpec, fit_context_predictor,
)
from phiagent.learning.qa_interaction_compiler import (
    FULL, InstructionBank, InterventionTask, _observations_digest,
)
from phiagent.learning.qa_interaction_controls import input_matched_random_masks
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_subset_selection import (
    _tasks, fixed_controls, group_folds, summarize_assignments,
)


def assert_disjoint(left: Sequence[InterventionTask], right: Sequence[InterventionTask]) -> None:
    for name, extract in (
        ("task", lambda task: task.task_id),
        ("group", lambda task: task.group_id),
        ("input", lambda task: task.input_sha256),
        ("exact context", lambda task: text_hash(task.context)),
    ):
        if {extract(task) for task in left} & {extract(task) for task in right}:
            raise ValueError(f"nested training and holdout overlap by {name}")


def aggregate_ranking(folds: list[dict]) -> dict:
    pairs = sum(row["positives"] * row["negatives"] for row in folds)
    positives = sum(row["positives"] for row in folds)
    return {
        "folds": folds,
        "within_fold_pair_weighted_auroc": (
            math.fsum(
                row["auroc"] * row["positives"] * row["negatives"]
                for row in folds if row["auroc"] is not None
            ) / pairs if pairs else None
        ),
        "within_fold_positive_weighted_average_precision": (
            math.fsum(
                row["average_precision"] * row["positives"]
                for row in folds if row["average_precision"] is not None
            ) / positives if positives else None
        ),
        "comparable_positive_negative_pairs": pairs,
        "positive_cases": positives,
        "scope": "Risk ranking only on pruned outer-holdout inputs; all tasks remain in policy metrics.",
        "scores_are_calibrated_probabilities": False,
    }


def coverage_reports(tasks, choices, fold_ids, scores, config) -> dict:
    losses = {task.task_id: contrast(task, choices[task.task_id].base_mask) for task in tasks}
    eligible = [row for row in choices.values() if row.eligible]
    total_positive = sum(losses[row.task_id]["harm"] for row in eligible)
    reports = {}
    for method in (*MODES, "mask_prior", "guard_slack"):
        reports[method] = []
        for fraction in config["diagnostics"]["rank_coverages"]:
            flagged = set()
            for fold in range(config["target"]["outer_folds"]):
                current = [row for row in eligible if fold_ids[row.group_id] == fold]
                current.sort(key=lambda row: (
                    -scores[str(fold)][method][row.task_id],
                    digest({"seed": config["seed"] + fold, "task": row.task_id}),
                ))
                count = math.ceil(fraction * len(current))
                flagged.update(row.task_id for row in current[:count])
            true_positive = sum(losses[task]["harm"] for task in flagged)
            row = {
                "requested_per_fold_coverage": fraction,
                "eligible_tasks": len(eligible), "positive_tasks": total_positive,
                "flagged_tasks": len(flagged), "true_positive": true_positive,
                "false_positive": len(flagged) - true_positive,
                "precision": true_positive / len(flagged) if flagged else None,
                "recall": true_positive / total_positive if total_positive else None,
                "cost_budget_enforced_for_this_diagnostic": False,
            }
            for endpoint in ("exact_match", "f1"):
                total_loss = math.fsum(item["endpoints"][endpoint]["loss_mass"]
                                       for item in losses.values())
                captured = math.fsum(losses[task]["endpoints"][endpoint]["loss_mass"]
                                     for task in flagged)
                discarded_gain = math.fsum(losses[task]["endpoints"][endpoint]["gain_mass"]
                                           for task in flagged)
                row[endpoint] = {
                    "total_loss_mass": total_loss, "captured_loss_mass": captured,
                    "discarded_base_gain_mass": discarded_gain,
                    "loss_mass_recall": captured / total_loss if total_loss else None,
                }
            reports[method].append(row)
    return reports


def development_intervals(tasks, base, primary, random, fixed, config) -> dict:
    import numpy as np

    group_names = sorted({task.group_id for task in tasks})
    if len(group_names) < 2:
        return {"status": "INSUFFICIENT_GROUPS", "intervals": None}
    names = (
        "em_primary_minus_full", "f1_primary_minus_full",
        "em_primary_minus_matched_random", "f1_primary_minus_matched_random",
        "token_slack_vs_full", "token_slack_vs_fold_fixed",
        "em_primary_minus_base", "f1_primary_minus_base",
    )
    sums = np.zeros((len(group_names), len(names)))
    sizes = np.zeros(len(group_names))
    index = {group: i for i, group in enumerate(group_names)}
    for task in tasks:
        def mean(mask, endpoint):
            return math.fsum(getattr(block.outcomes[mask], endpoint)
                             for block in task.blocks) / len(task.blocks)
        primary_em, primary_f1, primary_tokens = (
            mean(primary[task.task_id], endpoint)
            for endpoint in ("exact_match", "f1", "total_tokens")
        )
        values = (
            primary_em - mean(FULL, "exact_match"),
            primary_f1 - mean(FULL, "f1"),
            primary_em - mean(random[task.task_id], "exact_match"),
            primary_f1 - mean(random[task.task_id], "f1"),
            0.9 * mean(FULL, "total_tokens") - primary_tokens,
            0.9 * mean(fixed[task.task_id], "total_tokens") - primary_tokens,
            primary_em - mean(base[task.task_id], "exact_match"),
            primary_f1 - mean(base[task.task_id], "f1"),
        )
        sums[index[task.group_id]] += values
        sizes[index[task.group_id]] += 1
    rng = np.random.default_rng(config["diagnostics"]["bootstrap_seed"])
    weights = rng.multinomial(
        len(group_names), [1 / len(group_names)] * len(group_names),
        size=config["diagnostics"]["bootstrap_draws"],
    )
    estimates = (weights @ sums) / (weights @ sizes)[:, None]
    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    points = sums.sum(axis=0) / sizes.sum()
    return {
        "status": "DESCRIPTIVE_CONDITIONAL_GROUP_INTERVALS",
        "groups": len(group_names), "draws": config["diagnostics"]["bootstrap_draws"],
        "seed": config["diagnostics"]["bootstrap_seed"],
        "intervals": {
            name: {"point": float(points[i]), "unadjusted_conditional_ci95": bounds[:, i].tolist()}
            for i, name in enumerate(names)
        },
        "retraining_uncertainty_included": False, "p_values_or_significance_claimed": False,
        "simultaneous_intervals": False, "degeneracy_is_not_noninferiority": True,
    }


def nested_study(
    tasks: Sequence[InterventionTask], *, bank: InstructionBank, config: dict,
    load_outer: Callable[[int], Mapping], prior_routes: Mapping[str, str],
    before_fit: Callable[[str, int, int | str], None] | None = None,
    save_model: Callable[[str, int, int | str, dict], None] | None = None,
    save_fold: Callable[[int, dict], None] | None = None,
    load_fitted: Callable[[str, int, int | str], Mapping] | None = None,
) -> dict:
    values = _tasks(tasks, bank.sha256)
    outer_ids = group_folds(
        values, folds=config["target"]["outer_folds"], seed=config["target"]["outer_fold_seed"],
    )
    if set(prior_routes) != {task.task_id for task in values}:
        raise ValueError("the complete fixed target route inventory is required")
    fractions = tuple(config["policy_replay"]["extra_native_input_fractions"])
    methods = (*MODES, "mask_prior", "guard_slack")
    routes = {method: {str(fraction): {} for fraction in fractions} for method in methods}
    budget_receipts = {method: {str(fraction): [] for fraction in fractions} for method in methods}
    ranking = {method: [] for method in methods}
    all_choices, all_scores, outer_records, model_bindings = {}, {}, {}, {}
    fold_fixed = {}
    fit_count = 0
    for outer in range(config["target"]["outer_folds"]):
        training = tuple(task for task in values if outer_ids[task.group_id] != outer)
        heldout = tuple(task for task in values if outer_ids[task.group_id] == outer)
        assert_disjoint(training, heldout)
        base = ContextSensitivityPredictor(load_outer(outer))
        validate_exposure(base, training, training=True)
        validate_exposure(base, heldout, training=False)
        if base.spec != ContextSensitivitySpec("full_context"):
            raise ValueError("the reused outer router is not the fixed full/uniform target")
        if (
            base.artifact["training"]["seed"] != config["target"]["outer_fold_seed"] + outer
            or base.artifact["training"]["observations_sha256"] != _observations_digest(training)
        ):
            raise ValueError("the fixed outer router was trained on different observations/settings")
        _, fixed_mask = fixed_controls(training)
        fold_fixed.update({task.task_id: fixed_mask for task in heldout})
        inner_ids = group_folds(
            training, folds=config["nesting"]["inner_folds"],
            seed=config["nesting"]["inner_fold_seed_base"] + outer,
        )
        inner_choices, inner_labels, inner_bindings = {}, {}, {}
        for inner in range(config["nesting"]["inner_folds"]):
            fitted = tuple(task for task in training if inner_ids[task.group_id] != inner)
            inner_holdout = tuple(task for task in training if inner_ids[task.group_id] == inner)
            assert_disjoint(fitted, inner_holdout)
            assert_disjoint(fitted, heldout)
            assert_disjoint(inner_holdout, heldout)
            if load_fitted is None:
                if before_fit is not None:
                    before_fit("inner_router", outer, inner)
                artifact = fit_context_predictor(
                    fitted, bank=bank, spec=ContextSensitivitySpec("full_context"),
                    seed=config["nesting"]["inner_fit_seed_base"] + 10 * outer + inner,
                )
                fit_count += 1
            else:
                artifact = load_fitted("inner_router", outer, inner)
            model = ContextSensitivityPredictor(artifact)
            if (
                model.spec != ContextSensitivitySpec("full_context")
                or model.artifact["training"]["seed"]
                != config["nesting"]["inner_fit_seed_base"] + 10 * outer + inner
                or model.artifact["training"]["observations_sha256"] != _observations_digest(fitted)
            ):
                raise ValueError("the inner router settings or training observations differ")
            validate_exposure(model, fitted, training=True)
            validate_exposure(model, inner_holdout, training=False)
            validate_exposure(model, heldout, training=False)
            if save_model is not None and load_fitted is None:
                save_model("inner_router", outer, inner, artifact)
            inner_bindings[str(inner)] = {
                "model_sha256": model.model_sha256,
                "fit_group_hashes": sorted({text_hash(task.group_id) for task in fitted}),
                "prediction_group_hashes": sorted({text_hash(task.group_id) for task in inner_holdout}),
            }
            for task in inner_holdout:
                choice = make_choice(task, model)
                inner_choices[task.task_id] = choice
                if choice.eligible:
                    inner_labels[task.task_id] = contrast(task, choice.base_mask)["harm"]
        if set(inner_choices) != {task.task_id for task in training}:
            raise ValueError("inner OOF target construction lost or duplicated an outer training task")
        eligible_training = [inner_choices[task.task_id] for task in training if inner_choices[task.task_id].eligible]
        prior = mask_priors(eligible_training, inner_labels)
        risk_models, risk_bindings = {}, {}
        for index, mode in enumerate(MODES):
            if not eligible_training:
                risk_models[mode] = None
                risk_bindings[mode] = {
                    "status": "UNINFORMATIVE_NO_PRUNED_TRAINING_CHOICES",
                    "fit_attempted": False,
                }
                continue
            features = [row.features(mode) for row in eligible_training]
            labels = [inner_labels[row.task_id] for row in eligible_training]
            groups = [row.group_id for row in eligible_training]
            if load_fitted is None:
                if before_fit is not None:
                    before_fit("risk_detector", outer, mode)
                artifact = fit_choice_risk(
                    features, labels, groups, feature_names=feature_names(mode),
                    seed=config["nesting"]["risk_fit_seed_base"] + 10 * outer + index,
                    iterations=config["detectors"]["iterations"],
                    learning_rate=config["detectors"]["learning_rate"],
                    l2=config["detectors"]["l2"],
                )
                fit_count += 1
            else:
                artifact = load_fitted("risk_detector", outer, mode)
            risk_models[mode] = ChoiceRiskPredictor(artifact)
            training_hash = digest({
                "features": features, "labels": labels, "group_ids": groups,
            })
            if (
                artifact["training"]["observations_sha256"] != training_hash
                or set(artifact["training"]["group_hashes"])
                != {text_hash(group) for group in groups}
                or artifact["settings"]["seed"]
                != config["nesting"]["risk_fit_seed_base"] + 10 * outer + index
                or tuple(artifact["feature_names"]) != feature_names(mode)
            ):
                raise ValueError("risk detector exposure, settings or input features differ")
            if save_model is not None and load_fitted is None:
                save_model("risk_detector", outer, mode, artifact)
            risk_bindings[mode] = {
                "status": artifact["status"], "fit_attempted": True,
                "fitting_performed": artifact["fitting_performed"],
                "model_sha256": risk_models[mode].model_sha256,
                "training_features_labels_groups_sha256": training_hash,
                "training_tasks": len(eligible_training), "positives": sum(labels),
                "training_group_hashes": sorted({text_hash(group) for group in groups}),
            }
        outer_choices = [make_choice(task, base) for task in heldout]
        for choice in outer_choices:
            if choice.base_mask != prior_routes[choice.task_id]:
                raise ValueError("the diagnosed outer router no longer reproduces its saved choice")
            all_choices[choice.task_id] = choice
        fold_scores = {method: {} for method in methods}
        for choice in outer_choices:
            if not choice.eligible:
                continue
            for mode in MODES:
                model = risk_models[mode]
                fold_scores[mode][choice.task_id] = (
                    model.score(choice.features(mode)) if model is not None else 0.0
                )
            fold_scores["mask_prior"][choice.task_id] = prior["scores"][choice.base_mask]
            fold_scores["guard_slack"][choice.task_id] = choice.guard_slack_score
        fold_routes = {}
        for method in methods:
            fold_routes[method] = {}
            for fraction in fractions:
                assigned, receipt = allocate_fallbacks(
                    outer_choices, fold_scores[method], extra_input_fraction=fraction,
                    seed=config["seed"] + outer,
                )
                routes[method][str(fraction)].update(assigned)
                budget_receipts[method][str(fraction)].append(receipt)
                fold_routes[method][str(fraction)] = assigned
        record = {
            "outer": outer, "base_model_sha256": base.model_sha256,
            "training_group_hashes": sorted({text_hash(task.group_id) for task in training}),
            "heldout_group_hashes": sorted({text_hash(task.group_id) for task in heldout}),
            "inner_models": inner_bindings, "detectors": risk_bindings,
            "mask_prior": prior, "preanswer_scores": fold_scores, "preanswer_routes": fold_routes,
            "outer_outcomes_used_to_fit_or_allocate": False,
        }
        if save_fold is not None:
            save_fold(outer, record)
        outer_records[str(outer)] = record
        all_scores[str(outer)] = fold_scores
        # Outer outcomes enter only after detector fits and input-only allocations are fixed.
        labels_by_task = {
            task.task_id: contrast(task, all_choices[task.task_id].base_mask)["harm"]
            for task in heldout
        }
        eligible_holdout = [row for row in outer_choices if row.eligible]
        for method in methods:
            metrics = binary_ranking_metrics(
                [labels_by_task[row.task_id] for row in eligible_holdout],
                [fold_scores[method][row.task_id] for row in eligible_holdout],
            )
            ranking[method].append({"outer_fold": outer, **metrics})
        model_bindings[str(outer)] = {
            "inner_models": inner_bindings, "detectors": risk_bindings,
        }
    base_routes = {task.task_id: all_choices[task.task_id].base_mask for task in values}
    reference_summaries = (
        summarize_assignments(values, {task.task_id: FULL for task in values}),
        summarize_assignments(values, fold_fixed),
    )
    native = {task.task_id: dict(all_choices[task.task_id].native_inputs) for task in values}
    strata = {
        task.task_id: f"{outer_ids[task.group_id]}:{base_routes[task.task_id]}" for task in values
    }
    random_receipts = {}
    for mode in MODES:
        name = f"{mode}_matched_random"
        routes[name], random_receipts[name] = {}, {}
        for fraction in fractions:
            assigned, receipt = input_matched_random_masks(
                routes[mode][str(fraction)], strata=strata, native_inputs=native,
                seed=config["policy_replay"]["matched_random_seed"],
                maximum_relative_difference=0.0,
            )
            routes[name][str(fraction)] = assigned
            random_receipts[name][str(fraction)] = receipt
    diagnostic = decompose(values, all_choices)
    oracle_routes, oracle_receipts = {}, {}
    by_id = {task.task_id: task for task in values}
    for fraction in fractions:
        assigned, receipts = {}, []
        for outer in range(config["target"]["outer_folds"]):
            selected = [row for row in all_choices.values() if outer_ids[row.group_id] == outer]
            scores = {}
            for row in selected:
                if row.eligible:
                    losses = contrast(by_id[row.task_id], row.base_mask)["endpoints"]
                    scores[row.task_id] = math.fsum(
                        value["loss_mass"] for value in losses.values()
                    ) / row.extra_input_tokens
            # Zero-benefit cases are excluded only in this explicitly outcome-informed reference.
            useful = [row for row in selected if row.eligible and scores[row.task_id] > 0]
            assigned.update({row.task_id: row.base_mask for row in selected})
            budget = int(Fraction(str(fraction)) * sum(
                row.native_inputs[row.base_mask] for row in selected
            ))
            spent = 0
            for row in sorted(useful, key=lambda row: (-scores[row.task_id], text_hash(row.task_id))):
                if spent + row.extra_input_tokens <= budget:
                    assigned[row.task_id] = FULL
                    spent += row.extra_input_tokens
            receipts.append({"outer": outer, "budget": budget, "spent": spent})
        oracle_routes[str(fraction)] = assigned
        oracle_receipts[str(fraction)] = receipts
    summaries = {
        name: {
            fraction: policy_summary(values, assigned, reference_summaries)
            for fraction, assigned in by_budget.items()
        }
        for name, by_budget in routes.items()
    }
    primary_mode = config["detectors"]["primary"]
    primary_budget = str(config["policy_replay"]["primary_fraction"])
    base_summary = summarize_assignments(values, base_routes)
    for by_budget in summaries.values():
        for summary in by_budget.values():
            summary["net_gap_recovery_fraction"] = {
                endpoint: (
                    (summary[endpoint] - base_summary[endpoint])
                    / (reference_summaries[0][endpoint] - base_summary[endpoint])
                    if reference_summaries[0][endpoint] > base_summary[endpoint] else None
                )
                for endpoint in ("exact_match", "f1")
            }
    report = {
        "schema_version": "phiagent-qa-choice-risk-results/v1",
        "status": "NESTED_PREANSWER_DIAGNOSIS_COMPLETE",
        "primary_detector": config["detectors"]["primary"],
        "primary_budget_fraction": config["policy_replay"]["primary_fraction"],
        "base": base_summary,
        "references": {
            "full_DELETE": reference_summaries[0],
            "training_fold_selected_fixed": reference_summaries[1],
        },
        "decomposition": diagnostic,
        "conditional_addback_diagnosis": addback_diagnosis(values, all_choices),
        "ranking": {name: aggregate_ranking(rows) for name, rows in ranking.items()},
        "coverage_diagnostics": coverage_reports(values, all_choices, outer_ids, all_scores, config),
        "primary_descriptive_intervals": development_intervals(
            values, base_routes, routes[primary_mode][primary_budget],
            routes[f"{primary_mode}_matched_random"][primary_budget],
            fold_fixed, config,
        ),
        "policy_summaries": summaries, "budget_receipts": budget_receipts,
        "matched_random_receipts": random_receipts,
        "oracle": {
            "summaries": {
                fraction: policy_summary(values, assigned, reference_summaries)
                for fraction, assigned in oracle_routes.items()
            },
            "budget_receipts": oracle_receipts,
            "uses_outcomes": True, "globally_optimal_or_feasibility_certificate": False,
        },
        "model_bindings": model_bindings,
        "parameterized_fit_attempts": fit_count,
        "outer_routers_reused": 5, "mask_prior_tabulations": 5,
        "calibration_opened": False, "new_actor_calls": 0, "new_evaluator_calls": 0,
        "complete_public_benchmark_result": False, "paper_admitted": False,
        "boundary": "Nested diagnosis of one already-selected development router, not fresh confirmation.",
    }
    return {
        "report": report, "policy_routes": routes, "base_routes": base_routes,
        "oracle_routes": oracle_routes, "outer_records": outer_records, "outer_scores": all_scores,
    }
