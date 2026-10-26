#!/usr/bin/env python3
"""Freeze CHISEL ablation routes before evaluation-label access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phiagent.learning.residual_meta_harness import canonical_hash
from phiagent.learning.residual_meta_training import (
    _fit_ridge,
    _predict,
    _validate_examples,
)
from scripts import run_searchqa_residual_meta_campaign as campaign


MINIMUM_SUPPORT = 400
MAXIMUM_HARMFUL_RATE = 0.05
MAXIMUM_MEAN_TOKEN_DELTA = -0.08


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def verify_training(value: Mapping[str, Any]) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("training_sha256", None)
    if value.get("status") != "DISCOVERY_CANDIDATE" or declared != canonical_hash(unsigned):
        raise RuntimeError("frozen CHISEL training result differs")


def policy_metrics(
    matrix: Sequence[Mapping[str, Any]],
    overrides: set[str],
) -> dict[str, Any]:
    utility = 0
    harmful = 0
    token_deltas = []
    for row in matrix:
        task_id = str(row["id"])
        deployed = row["arms"]["modify" if task_id in overrides else "delete"]
        reference = row["arms"]["skillopt"]
        utility += int(deployed["hard"]) - int(reference["hard"])
        harmful += int(not int(deployed["hard"]) and int(reference["hard"]))
        reference_tokens = max(int(reference["total_tokens"]), 1)
        token_deltas.append(
            max(
                -1.0,
                min(
                    1.0,
                    (int(deployed["total_tokens"]) - reference_tokens) / reference_tokens,
                ),
            )
        )
    return {
        "override_support": len(overrides),
        "mean_utility": utility / len(matrix),
        "utility_sum": utility,
        "harmful_rate": harmful / len(matrix),
        "mean_token_delta": sum(token_deltas) / len(token_deltas),
    }


def select_threshold(
    *,
    scores: Mapping[str, float],
    matrix: Sequence[Mapping[str, Any]],
    require_cost_gate: bool,
) -> dict[str, Any]:
    ordered = sorted(
        scores,
        key=lambda task_id: (
            -float(scores[task_id]),
            hashlib.sha256(task_id.encode()).hexdigest(),
        ),
    )
    candidates = []
    cursor = 0
    while cursor < len(ordered):
        threshold = float(scores[ordered[cursor]])
        end = cursor
        while end < len(ordered) and float(scores[ordered[end]]) == threshold:
            end += 1
        overrides = set(ordered[:end])
        if len(overrides) >= MINIMUM_SUPPORT:
            metrics = policy_metrics(matrix, overrides)
            if float(metrics["harmful_rate"]) <= MAXIMUM_HARMFUL_RATE and (
                not require_cost_gate
                or float(metrics["mean_token_delta"]) <= MAXIMUM_MEAN_TOKEN_DELTA
            ):
                candidates.append(
                    {
                        "score_threshold": threshold,
                        **metrics,
                    }
                )
        cursor = end
    if not candidates:
        raise RuntimeError("no ablation threshold satisfies its frozen gates")
    return min(
        candidates,
        key=lambda row: (
            -int(row["utility_sum"]),
            float(row["harmful_rate"]),
            float(row["mean_token_delta"]),
            -float(row["score_threshold"]),
        ),
    )


def normalized_eval_features(
    public_items: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    rows = _validate_examples(
        [
            {
                "task_id": str(item["id"]),
                "features": campaign.task_features(item),
                "utility": 0,
                "harmful": False,
                "token_delta": 0,
            }
            for item in public_items
        ],
        campaign.FEATURE_SCHEMA,
    )
    return {task_id: features for task_id, features, *_ in rows}


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    experiment = args.experiment_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if (experiment / "analysis" / args.split).exists():
        raise RuntimeError("evaluation labels were opened before route freeze")
    protocol = campaign.load_protocol(experiment)
    public_path = experiment / "public" / f"{args.split}.json"
    public_items = campaign.read_json(public_path)
    if not isinstance(public_items, list) or not public_items:
        raise RuntimeError("public evaluation tasks are unavailable")
    training = campaign.read_json(args.training_result.resolve())
    verify_training(training)
    source_matrix_path = args.source_matrix.resolve()
    source_matrix = campaign.read_matrix(source_matrix_path)
    if len(source_matrix) != int(training["training_task_count"]):
        raise RuntimeError("source matrix and training task counts differ")

    examples = campaign.training_examples(
        source_matrix,
        candidate_variant="modify",
        default_variant="delete",
    )
    rows = _validate_examples(examples, campaign.FEATURE_SCHEMA)
    by_task = {task_id: row for task_id, *row in rows}
    names = tuple(sorted(campaign.FEATURE_SCHEMA))
    final_intercept, final_weights = _fit_ridge(
        rows,
        ridge=float(training["ridge"]),
    )
    expected_weights = [float(training["weights"][name]) for name in names]
    if abs(final_intercept - float(training["intercept"])) > 1e-12 or any(
        abs(observed - expected) > 1e-12
        for observed, expected in zip(
            final_weights,
            expected_weights,
            strict=True,
        )
    ):
        raise RuntimeError("source matrix does not reproduce CHISEL head")

    fold_by_hash = training["training_task_folds"]
    out_of_fold_scores = {}
    for fold in range(int(training["grouped_crossfit_folds"])):
        train = [
            row
            for row in rows
            if int(fold_by_hash[hashlib.sha256(row[0].encode()).hexdigest()]) != fold
        ]
        held = [
            row
            for row in rows
            if int(fold_by_hash[hashlib.sha256(row[0].encode()).hexdigest()]) == fold
        ]
        intercept, weights = _fit_ridge(
            train,
            ridge=float(training["ridge"]),
        )
        for task_id, features, *_ in held:
            out_of_fold_scores[task_id] = _predict(
                intercept,
                weights,
                features,
            )
    in_sample_scores = {
        task_id: _predict(
            final_intercept,
            final_weights,
            row[0],
        )
        for task_id, row in by_task.items()
    }
    without_cross_fit = select_threshold(
        scores=in_sample_scores,
        matrix=source_matrix,
        require_cost_gate=True,
    )
    without_cost_gate = select_threshold(
        scores=out_of_fold_scores,
        matrix=source_matrix,
        require_cost_gate=False,
    )

    evaluation_features = normalized_eval_features(public_items)
    full_threshold = float(training["score_threshold"])
    thresholds = {
        "chisel_full": full_threshold,
        "without_cross_fitting": float(without_cross_fit["score_threshold"]),
        "without_cost_gate": float(without_cost_gate["score_threshold"]),
    }
    routed_ids = {
        policy: sorted(
            task_id
            for task_id, features in evaluation_features.items()
            if _predict(
                final_intercept,
                final_weights,
                features,
            )
            >= threshold
        )
        for policy, threshold in thresholds.items()
    }
    full_support = len(routed_ids["chisel_full"])
    random_ids = sorted(
        evaluation_features,
        key=lambda task_id: (
            hashlib.sha256(f"chisel-matched-random-v1\0{task_id}".encode()).digest(),
            task_id,
        ),
    )[:full_support]

    policy_specs = {
        "skillopt_parent": {
            "default_variant": "skillopt",
            "override_variant": None,
            "override_task_ids": [],
        },
        "add_only": {
            "default_variant": "add",
            "override_variant": None,
            "override_task_ids": [],
        },
        "delete_only": {
            "default_variant": "delete",
            "override_variant": None,
            "override_task_ids": [],
        },
        "without_abstention": {
            "default_variant": "modify",
            "override_variant": None,
            "override_task_ids": [],
        },
        "chisel_full": {
            "default_variant": "delete",
            "override_variant": "modify",
            "override_task_ids": routed_ids["chisel_full"],
        },
        "matched_random": {
            "default_variant": "delete",
            "override_variant": "modify",
            "override_task_ids": sorted(random_ids),
        },
        "without_cross_fitting": {
            "default_variant": "delete",
            "override_variant": "modify",
            "override_task_ids": routed_ids["without_cross_fitting"],
        },
        "without_cost_gate": {
            "default_variant": "delete",
            "override_variant": "modify",
            "override_task_ids": routed_ids["without_cost_gate"],
        },
    }
    value = {
        "schema_version": "phiagent-chisel-ablation-routes/v1",
        "status": "FROZEN_BEFORE_EVALUATION_LABEL_ACCESS",
        "frozen_at_utc": utc_now(),
        "method_name": "CHISEL",
        "method_expansion": (
            "Certified Harness Interventions via Sparse Editing and Local routing"
        ),
        "evaluation": {
            "protocol_sha256": protocol["canonical_sha256"],
            "split": args.split,
            "task_count": len(public_items),
            "public_items_sha256": campaign.file_sha256(public_path),
            "labels_read": False,
        },
        "training": {
            "training_result_sha256": training["training_sha256"],
            "training_file_sha256": campaign.file_sha256(args.training_result.resolve()),
            "source_matrix_sha256": campaign.file_sha256(source_matrix_path),
            "source_task_count": len(source_matrix),
        },
        "threshold_ablations": {
            "chisel_full": {
                "score_threshold": full_threshold,
                "selection": "task-grouped OOF full-policy cost gate",
            },
            "without_cross_fitting": without_cross_fit,
            "without_cost_gate": without_cost_gate,
        },
        "policies": policy_specs,
        "matched_random_support_equals_full": (len(random_ids) == full_support),
        "oracle_is_posthoc_only": True,
    }
    value["canonical_sha256"] = canonical_hash(value)
    campaign.write_json(output, value, new=True)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--split", default="discovery")
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    value = freeze(parse_args(argv))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
