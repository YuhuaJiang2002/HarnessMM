"""Pure per-task EdgeBench metrics, internal selection, and paired token accounting.

Selection semantics were inspected in ByteDance-Seed/EdgeBench at the pinned
revision below. ``pass_rate_first`` ignores scores below a tied pass rate of one;
``score_first`` ignores validity; only ``valid_then_score`` filters invalid
entries. ``max_score`` is a score fallback only for ``pass_rate_first``.

Unlike the official selector's empty-history/default-value bookkeeping, missing
required observations remain unresolved here. No raw score is manufactured from
a pass rate, and a missing rescale mapping never becomes an identity mapping.
Comparisons are internal train/validation utilities, not benchmark metrics.
Reporting retains every declared task/repetition pair and never averages quality
across tasks. This module imports no SForge, benchmark, model, or GPU dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math


SELECTION_SOURCE_REVISION = "6e4e5f4aa087a9121aa09a6059e5484e5102c6ff"
SELECTION_SOURCE_SHA256 = "abb55b7b0466f73e4fab248dfed31ad89b1caec19b1a8117edcbf2429571cf0a"
SELECTION_SOURCE_URL = (
    "https://github.com/ByteDance-Seed/EdgeBench/blob/"
    f"{SELECTION_SOURCE_REVISION}/sforge/harness/selection.py"
)
POLICIES = ("pass_rate_first", "score_first", "valid_then_score")
DIRECTIONS = ("minimize", "maximize")
PURPOSES = ("train_routing", "validation_selection")
OUTCOME_SCHEMA = "edgebench-native-outcome/v1"


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be a boolean or None")
    return value


def _optional_number(value: object, name: str) -> int | float | None:
    if value is None:
        return None
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number or None")
    return value


def _optional_count(value: object, name: str) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a measured nonnegative integer or None")
    return value


def _protocol(score_direction: str, selection_policy: str, normalization_declared: bool) -> dict:
    if score_direction not in DIRECTIONS or selection_policy not in POLICIES:
        raise ValueError("Explicit official score direction and selection policy are required")
    if type(normalization_declared) is not bool:
        raise ValueError("normalization_declared must say whether TaskSpec.judge.rescale exists")
    return {
        "score_direction": score_direction, "selection_policy": selection_policy,
        "normalization_declared": normalization_declared,
    }


def validate_native_outcome(
    report: Mapping | None,
    *,
    task_id: str,
    score_direction: str,
    selection_policy: str,
    normalization_declared: bool,
    infrastructure_ok: bool | None,
) -> dict:
    """Return a JSON-safe, whitelisted native observation without inventing scores.

    Pass ``normalization_declared=task.judge.rescale is not None``; this helper
    preserves the native ``score_0_100`` and does not implement/rescale a metric.
    An observed normalized value without a declared mapping is rejected.
    ``report=None`` means missing. Infrastructure failure/unknown state makes all
    quality fields None, including any placeholder zeros in the supplied report;
    errors and timeout diagnostics remain available. Native invalidity and native
    timeouts are otherwise retained, not confused with infrastructure failure.
    """
    _identity(task_id, "task_id")
    protocol = _protocol(score_direction, selection_policy, normalization_declared)
    _optional_bool(infrastructure_ok, "infrastructure_ok")
    if report is not None and not isinstance(report, Mapping):
        raise ValueError("Native report must be a mapping or None")
    present = report is not None
    report = report if present else {}
    if report.get("task_id", task_id) != task_id:
        raise ValueError("Native report task identity differs from its declared task")
    entry_type, status = report.get("type", "submission"), report.get("status")
    if any(value is not None and not isinstance(value, str) for value in (entry_type, status)):
        raise ValueError("Native type/status must be strings or None")
    errors = _optional_count(report.get("errors"), "errors")
    timed_out = _optional_bool(report.get("timed_out"), "timed_out")
    if infrastructure_ok is False:
        availability = "infrastructure_error"
    elif infrastructure_ok is None:
        availability = "infrastructure_unknown"
    elif not present:
        availability = "missing"
    elif entry_type not in ("submission", "game") or status not in (None, "completed"):
        availability = "not_completed_submission"
    else:
        availability = "observed"
    quality = {field: None for field in ("score", "max_score", "pass_rate", "score_0_100", "valid")}
    if infrastructure_ok is True and present:
        quality.update({
            field: _optional_number(report.get(field), field)
            for field in ("score", "max_score", "pass_rate", "score_0_100")
        })
        quality["valid"] = _optional_bool(report.get("valid"), "valid")
        if quality["pass_rate"] is not None and not 0 <= quality["pass_rate"] <= 1:
            raise ValueError("Native pass_rate must be in [0, 1]")
        if quality["score_0_100"] is not None:
            if not normalization_declared:
                raise ValueError("Missing rescale mapping requires score_0_100=None")
            if not 0 <= quality["score_0_100"] <= 100:
                raise ValueError("Native score_0_100 must be in [0, 100]")
    return {
        "schema": OUTCOME_SCHEMA, "task_id": task_id, **protocol, **quality,
        "outcome_present": present, "infrastructure_ok": infrastructure_ok,
        "availability": availability, "type": entry_type, "status": status,
        "errors": errors, "timed_out": timed_out, "binary_task_success": None,
    }


def _checked_outcome(value: Mapping | None) -> dict | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping) or value.get("schema") != OUTCOME_SCHEMA
        or type(value.get("outcome_present")) is not bool
    ):
        raise ValueError("Use validate_native_outcome before comparison/reporting")
    try:
        return validate_native_outcome(
            value if value["outcome_present"] else None,
            task_id=value["task_id"], score_direction=value["score_direction"],
            selection_policy=value["selection_policy"],
            normalization_declared=value["normalization_declared"],
            infrastructure_ok=value["infrastructure_ok"],
        )
    except KeyError as error:
        raise ValueError(f"Native outcome lacks {error.args[0]}") from None


def _provenance() -> dict:
    return {
        "url": SELECTION_SOURCE_URL, "revision": SELECTION_SOURCE_REVISION,
        "sha256": SELECTION_SOURCE_SHA256,
    }


def compare_native_outcomes(
    baseline: Mapping | None,
    candidate: Mapping | None,
    *,
    purpose: str,
) -> dict:
    """Return candidate-relative ordinal win/tie/loss, or None when unresolved.

    ``purpose`` must be train_routing or validation_selection, never test scoring.
    Inputs come from validate_native_outcome. Required metric values must be
    observed; irrelevant missing fields do not prevent a policy-decided result.
    Both explicitly invalid entries tie under valid_then_score. Equal subunit
    pass rates tie regardless of score. Numeric comparisons use exact native
    inequalities, without epsilon, normalized scores, or token tie-breakers.
    """
    if purpose not in PURPOSES:
        raise ValueError("Ordinal selection is internal TRAIN routing/validation selection only")
    baseline, candidate = _checked_outcome(baseline), _checked_outcome(candidate)
    if baseline is not None and candidate is not None:
        if any(baseline[key] != candidate[key] for key in (
            "task_id", "score_direction", "selection_policy", "normalization_declared",
        )):
            raise ValueError("Comparison requires the same task and frozen native metric protocol")
    reference = baseline if baseline is not None else candidate

    def result(ordinal: str | None, reason: str) -> dict:
        return {
            "task_id": reference["task_id"] if reference else None,
            "ordinal": ordinal, "resolved": ordinal is not None, "reason": reason,
            "purpose": purpose, "candidate_relative_to": "baseline",
            "official_benchmark_metric": False, "selection_source": _provenance(),
        }

    if baseline is None or candidate is None:
        return result(None, "missing_outcome")
    if any(outcome["availability"] != "observed" for outcome in (baseline, candidate)):
        return result(None, "unobserved_or_infrastructure_failed_outcome")
    policy = baseline["selection_policy"]
    if policy == "valid_then_score":
        if baseline["valid"] is None or candidate["valid"] is None:
            return result(None, "missing_validity")
        if not baseline["valid"] and not candidate["valid"]:
            return result("tie", "both_explicitly_invalid")
        if any(outcome["valid"] and outcome["score"] is None
               for outcome in (baseline, candidate)):
            return result(None, "missing_score_for_valid_entry")
        if baseline["valid"] != candidate["valid"]:
            return result("win" if candidate["valid"] else "loss", "official_validity_filter")
    if policy == "pass_rate_first":
        if baseline["pass_rate"] is None or candidate["pass_rate"] is None:
            return result(None, "missing_pass_rate")
        old, new = baseline["pass_rate"], candidate["pass_rate"]
        if old < 1 or new < 1:
            ordinal = "win" if new > old else "loss" if new < old else "tie"
            return result(ordinal, "official_pass_rate_order")
    old, new = (
        outcome["score"] if outcome["score"] is not None else (
            outcome["max_score"] if policy == "pass_rate_first" else None
        )
        for outcome in (baseline, candidate)
    )
    if old is None or new is None:
        return result(None, "missing_ordering_score")
    if old == new:
        return result("tie", "equal_native_score")
    improvement = new < old if baseline["score_direction"] == "minimize" else new > old
    return result("win" if improvement else "loss", "official_raw_score_order")


def _difference(baseline: int | float | None, candidate: int | float | None):
    if baseline is None or candidate is None:
        return None
    value = candidate - baseline
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _token_summary(pairs: Sequence[dict], baseline: str, candidate: str) -> dict:
    methods = {}
    for method in (baseline, candidate):
        values = [pair["methods"][method]["actor_tokens"] for pair in pairs]
        known = [value for value in values if value is not None]
        complete = len(known) == len(pairs)
        methods[method] = {
            "complete": complete, "measured_trials": len(known),
            "unknown_trials": len(pairs) - len(known),
            "known_subtotal": sum(known) if known else None,
            "total": sum(known) if complete else None,
        }
    old, new = methods[baseline]["total"], methods[candidate]["total"]
    delta = new - old if old is not None and new is not None else None
    saving_percent = None
    if delta is not None and old > 0:
        try:
            saving_percent = 100 * ((old - new) / old)
        except OverflowError:
            pass
        if saving_percent is not None and not math.isfinite(saving_percent):
            saving_percent = None
    return {
        "methods": methods, "delta_candidate_minus_baseline": delta,
        "saving_percent": saving_percent,
    }


def summarize_native_pairs(
    rows: Iterable[Mapping],
    *,
    task_protocols: Mapping[str, Mapping],
    repetitions: Sequence[int],
    baseline: str = "baseline",
    candidate: str = "candidate",
) -> dict:
    """Report the complete *declared* population, including all missing pairs.

    task_protocols maps task ID to score_direction, selection_policy, and the
    explicit normalization_declared boolean. Repetitions are unique nonnegative
    integers. Each row contains task_id, repetition, method, outcome (validated
    dict or None), and actor_tokens (complete measured input+output total or None).
    Actor costs remain countable even when native evaluation infrastructure fails.

    Duplicate/out-of-population rows and protocol drift are errors. All expected
    pairs are emitted; absent observations and incomplete accounting stay None.
    Native quality is reported per task/repetition, never pooled (even over just
    the mapped tasks). Token totals/deltas require complete paired accounting.
    Observation completeness is separate from per-metric coverage (in particular,
    absent raw scores and missing mappings). All completeness refers to
    task_protocols, not an inferred public51 claim.
    """
    _identity(baseline, "baseline")
    _identity(candidate, "candidate")
    if baseline == candidate:
        raise ValueError("Baseline and candidate must be distinct methods")
    if not isinstance(task_protocols, Mapping) or not task_protocols:
        raise ValueError("An explicit complete declared task population is required")
    protocols = {}
    for task_id, protocol in task_protocols.items():
        _identity(task_id, "task_id")
        if not isinstance(protocol, Mapping):
            raise ValueError("Each task requires its frozen native metric protocol")
        protocols[task_id] = _protocol(
            protocol.get("score_direction"), protocol.get("selection_policy"),
            protocol.get("normalization_declared"),
        )
    if (
        not isinstance(repetitions, Sequence) or isinstance(repetitions, (str, bytes))
        or not repetitions or any(type(value) is not int or value < 0 for value in repetitions)
        or len(set(repetitions)) != len(repetitions)
    ):
        raise ValueError("Explicit unique nonnegative repetition IDs are required")
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each native result row must be a mapping")
        task_id, repeat, method = row.get("task_id"), row.get("repetition"), row.get("method")
        if (
            not isinstance(task_id, str) or task_id not in protocols
            or type(repeat) is not int or repeat not in repetitions
            or method not in (baseline, candidate)
        ):
            raise ValueError("Result row is outside the declared paired population")
        key = task_id, repeat, method
        if key in indexed:
            raise ValueError("Duplicate task/repetition/method result")
        outcome = _checked_outcome(row.get("outcome"))
        if outcome is not None and (
            outcome["task_id"] != task_id
            or any(outcome[field] != value for field, value in protocols[task_id].items())
        ):
            raise ValueError("Result outcome differs from the frozen per-task protocol")
        indexed[key] = {
            "row_present": True, "outcome": outcome,
            "actor_tokens": _optional_count(row.get("actor_tokens"), "actor_tokens"),
        }
    per_task, all_pairs = [], []
    for task_id, protocol in protocols.items():
        pairs = []
        for repeat in repetitions:
            methods = {
                method: indexed.get((task_id, repeat, method), {
                    "row_present": False, "outcome": None, "actor_tokens": None,
                })
                for method in (baseline, candidate)
            }
            outcomes = [methods[method]["outcome"] for method in (baseline, candidate)]
            observed = all(
                outcome is not None and outcome["availability"] == "observed" for outcome in outcomes
            )
            deltas = {
                field: _difference(outcomes[0][field], outcomes[1][field]) if observed else None
                for field in ("score", "pass_rate", "score_0_100")
            }
            tokens = [methods[method]["actor_tokens"] for method in (baseline, candidate)]
            pair = {
                "task_id": task_id, "repetition": repeat, "methods": methods,
                "both_rows_present": all(value["row_present"] for value in methods.values()),
                "both_native_outcomes_observed": observed,
                "native_deltas_candidate_minus_baseline": deltas,
                "actor_token_delta_candidate_minus_baseline": (
                    tokens[1] - tokens[0] if all(value is not None for value in tokens) else None
                ),
            }
            pairs.append(pair)
        per_task.append({
            "task_id": task_id, **protocol, "pairs": pairs,
            "actor_tokens": _token_summary(pairs, baseline, candidate),
        })
        all_pairs.extend(pairs)
    rows_complete = len(indexed) == 2 * len(all_pairs)
    outcomes_complete = all(pair["both_native_outcomes_observed"] for pair in all_pairs)
    metric_coverage = {
        field: sum(
            pair["both_native_outcomes_observed"]
            and all(pair["methods"][method]["outcome"][field] is not None
                    for method in (baseline, candidate))
            for pair in all_pairs
        )
        for field in ("score", "pass_rate", "valid", "score_0_100")
    }
    tokens = _token_summary(all_pairs, baseline, candidate)
    return {
        "schema": "edgebench-native-paired-report/v1",
        "population_scope": "explicitly_declared_tasks_and_repetitions",
        "task_ids": list(protocols), "task_count": len(protocols),
        "repetitions": list(repetitions), "expected_pairs": len(all_pairs),
        "expected_rows": 2 * len(all_pairs), "received_rows": len(indexed),
        "rows_complete": rows_complete, "native_outcomes_complete": outcomes_complete,
        "paired_metric_coverage": metric_coverage,
        "raw_scores_complete": metric_coverage["score"] == len(all_pairs),
        "actor_tokens_complete": all(value["complete"] for value in tokens["methods"].values()),
        "normalization_declared_tasks": sum(p["normalization_declared"] for p in protocols.values()),
        "tasks_without_normalization": [
            task_id for task_id, protocol in protocols.items() if not protocol["normalization_declared"]
        ],
        "baseline": baseline, "candidate": candidate, "per_task": per_task,
        "actor_tokens": tokens, "quality_aggregate": None, "ordinal_aggregate": None,
        "selection_source": _provenance(),
    }
