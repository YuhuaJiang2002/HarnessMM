"""Prospective comparator releases; never weaken the primary candidate's gate."""

from __future__ import annotations

from collections.abc import Mapping
import math

from phiagent.learning.qa_interaction_compiler import FULL, MASKS, InteractionCompiler
from phiagent.learning.qa_meta_rsi import digest


MARGINS = (0.0, 0.01, 0.02, 0.05)
COMPARATORS = frozenset({
    "additive_order_one", "no_task_features", "first_repeat_supervision",
    "no_aa_guard", "independent32_profile_predictor",
})


def _summary(row: Mapping, name: str) -> None:
    for metric in ("exact_match", "f1", "mean_total_tokens"):
        value = row[metric]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} requires finite numeric {metric}")
        if metric == "mean_total_tokens":
            if value <= 0:
                raise ValueError("actual mean total tokens must be positive")
        elif not 0 <= value <= 1:
            raise ValueError("calibration quality must use the official normalized scale")


def select_comparator_release(
    method_id: str, model_artifact: Mapping, calibration: Mapping,
) -> dict:
    """Use observed calibration cost/quality, not unseen official outcomes.

    Comparator eligibility is not scientific admission. A fixed fallback is
    explicit because a global ablation cannot beat the best fixed policy by 10%.
    Real-data invocation must run on an authorized server.
    """
    if method_id not in COMPARATORS:
        raise ValueError("this comparator-only release rule cannot release the primary method")
    compiler = InteractionCompiler(model_artifact)
    expected_modes = {
        "additive_order_one": (1, True, "mean", 1.0),
        "no_task_features": (2, False, "mean", 1.0),
        "first_repeat_supervision": (2, True, "first", 1.0),
        "no_aa_guard": (2, True, "mean", 0.0),
        "independent32_profile_predictor": (5, True, "mean", 1.0),
    }
    actual_mode = (
        model_artifact["order"], model_artifact["task_conditioned"],
        model_artifact["training"]["repeat_strategy"], calibration["repeat_guard"],
    )
    if actual_mode != expected_modes[method_id]:
        raise ValueError("the fitted model/settings do not implement the declared comparator")
    if (
        calibration["model_sha256"] != compiler.model_sha256
        or calibration["bank_sha256"] != model_artifact["bank_sha256"]
        or calibration["benchmark_result"] is not False
        or calibration["paper_admitted"] is not False
        or calibration["statistical_noninferiority_established"] is not False
        or calibration["minimum_token_saving"] != 0.1
        or type(calibration["tasks"]) is not int or calibration["tasks"] <= 0
        or type(calibration["groups"]) is not int
        or not 0 < calibration["groups"] <= calibration["tasks"]
    ):
        raise ValueError("complete source-bound calibration evidence and the original gate are required")
    fixed = calibration["fixed_controls"]
    if set(fixed) != set(MASKS):
        raise ValueError("all 32 fixed controls must remain available")
    for mask, value in fixed.items():
        _summary(value, mask)
    eligible_fixed = [
        mask for mask in MASKS
        if all(fixed[mask][metric] >= fixed[FULL][metric] for metric in ("exact_match", "f1"))
    ]
    best = min(eligible_fixed, key=lambda mask: (fixed[mask]["mean_total_tokens"], mask))
    if calibration["best_feasible_fixed_mask"] != best:
        raise ValueError("the frozen best-fixed reference was reselected or changed")
    references = (fixed[FULL], fixed[best])
    rows = calibration["candidate_family"]
    if (
        not isinstance(rows, list) or len(rows) != len(MARGINS)
        or tuple(row["margin"] for row in rows) != MARGINS
        or type(calibration["repeat_guard"]) not in (int, float)
        or not math.isfinite(calibration["repeat_guard"])
        or calibration["repeat_guard"] < 0
    ):
        raise ValueError("the complete predeclared margin family and repeat guard are required")
    candidates = [{
        "deployment_mode": "fixed_calibration_fallback",
        "fixed_mask": best, "quality_margin": None,
        "standalone_policy_sha256": digest({
            "bank_sha256": model_artifact["bank_sha256"], "fixed_mask": best,
        }),
        **fixed[best],
    }]
    strict = []
    for row in rows:
        if type(row["margin"]) not in (int, float) or not math.isfinite(row["margin"]):
            raise ValueError("margin values must be finite numbers, not boolean flags")
        _summary(row, "margin policy")
        quality_pass = all(
            row[metric] >= reference[metric]
            for reference in references for metric in ("exact_match", "f1")
        )
        cost_pass = all(
            row["mean_total_tokens"] <= 0.9 * reference["mean_total_tokens"]
            for reference in references
        )
        if (
            type(row["changed_tasks"]) is not int
            or not 0 <= row["changed_tasks"] <= calibration["tasks"]
            or row["quality_point_gate"] is not quality_pass
            or row["cost_point_gate"] is not cost_pass
            or row["eligible"] is not (quality_pass and cost_pass and row["changed_tasks"] > 0)
        ):
            raise ValueError("original calibration gate flags disagree with measured outcomes")
        policy_hash = digest({
            "model_sha256": compiler.model_sha256,
            "quality_margin": float(row["margin"]),
            "repeat_guard": float(calibration["repeat_guard"]),
        })
        if row["policy_sha256"] != policy_hash:
            raise ValueError("margin policy identity differs from the calibrated model/settings")
        if row["eligible"]:
            strict.append(row)
        if quality_pass:
            candidates.append({
                "deployment_mode": "conditional",
                "fixed_mask": None, "quality_margin": row["margin"],
                "standalone_policy_sha256": policy_hash,
                **{key: row[key] for key in ("exact_match", "f1", "mean_total_tokens")},
            })
    strict_selected = (
        min(strict, key=lambda row: (row["mean_total_tokens"], -row["margin"])) if strict else None
    )
    expected_status = (
        "CALIBRATION_POINT_GATE_PASS_PENDING_CONFIRMATION"
        if strict_selected else "BLOCKED_NO_ADMISSIBLE_CALIBRATION_POLICY"
    )
    if (
        calibration["status"] != expected_status
        or calibration["selected_margin"] != (
            strict_selected["margin"] if strict_selected else None
        )
        or calibration["selected_policy_sha256"] != (
            strict_selected["policy_sha256"] if strict_selected else None
        )
    ):
        raise ValueError("the original strict primary-style gate result was changed")
    selected = min(candidates, key=lambda row: (
        row["mean_total_tokens"], -row["exact_match"], -row["f1"],
        row["deployment_mode"] != "fixed_calibration_fallback",
        -(row["quality_margin"] or 0.0),
    ))
    result = {
        "schema_version": "phiagent-qa-interaction-comparator-release/v1",
        "method_id": method_id,
        "model_sha256": compiler.model_sha256,
        "bank_sha256": model_artifact["bank_sha256"],
        "calibration_sha256": digest(dict(calibration)),
        "calibration_observations_sha256": calibration["calibration_observations_sha256"],
        "strict_primary_style_calibration_status": calibration["status"],
        "repeat_guard": calibration["repeat_guard"],
        **selected,
        "selection_rule": (
            "Minimum actual calibration tokens among the unchanged four margin policies "
            "passing both quality floors and the declared best-fixed fallback; ties prefer "
            "higher EM, higher F1, fixed deployment, then larger margin."
        ),
        "reason": (
            "The explicit fixed reference is the best quality-feasible comparator deployment."
            if selected["deployment_mode"] == "fixed_calibration_fallback" else
            "A predeclared conditional policy is the best quality-feasible comparator deployment."
        ),
        "primary_gate_relaxed": False,
        "adaptive_success_claimed": False,
        "official_outcomes_consulted": False,
        "paper_admitted": False,
    }
    result["release_sha256"] = digest(result)
    return result


def choose_released_comparator(
    release: Mapping, model_artifact: Mapping, *,
    question: str, context: str, native_inputs: Mapping[str, int],
) -> dict:
    """Resolve a released comparator using only immutable inputs and token counts."""
    unsigned = dict(release)
    expected = unsigned.pop("release_sha256")
    if digest(unsigned) != expected:
        raise ValueError("comparator release changed after freezing")
    compiler = InteractionCompiler(model_artifact)
    if (
        release["method_id"] not in COMPARATORS
        or release["model_sha256"] != compiler.model_sha256
        or release["bank_sha256"] != model_artifact["bank_sha256"]
        or release["primary_gate_relaxed"] is not False
        or release["paper_admitted"] is not False
        or not isinstance(question, str) or not question.strip()
        or not isinstance(context, str) or not context.strip()
        or set(native_inputs) != set(MASKS)
        or any(type(value) is not int or value <= 0 for value in native_inputs.values())
    ):
        raise ValueError("comparator model, source, immutable input or native counts differ")
    if release["deployment_mode"] == "fixed_calibration_fallback":
        mask = release["fixed_mask"]
        if (
            mask not in MASKS or release["quality_margin"] is not None
            or release["standalone_policy_sha256"] != digest({
                "bank_sha256": release["bank_sha256"], "fixed_mask": mask,
            })
        ):
            raise ValueError("fixed comparator deployment is not bound to its exact mask")
        result = {"selected_mask": mask, "status": "EXPLICIT_FIXED_CALIBRATION_FALLBACK"}
    elif release["deployment_mode"] == "conditional":
        if release["fixed_mask"] is not None:
            raise ValueError("a conditional release cannot silently carry a fixed fallback mask")
        result = compiler.choose(
            question=question, context=context, prompt_tokens=native_inputs,
            quality_margin=release["quality_margin"], repeat_guard=release["repeat_guard"],
        )
        if result["policy_sha256"] != release["standalone_policy_sha256"]:
            raise ValueError("conditional comparator parameters differ from its release")
    else:
        raise ValueError("unknown comparator deployment mode")
    return {
        **result, "comparator_release_sha256": expected,
        "deployment_mode": release["deployment_mode"],
        "runtime_outcomes_used": False, "paper_admitted": False,
    }
