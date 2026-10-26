"""Matched validation of a TRAIN-derived instruction with and without routing."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
import math
from typing import Any

from phiagent.harness import codex_chisel as core
from phiagent.harness.industrial_efficiency_rsi import efficiency_gate
from phiagent.learning.residual_meta_harness import canonical_hash

SCHEMA = "industrial-candidate-routing-comparison/v1"
MODES = ("baseline", "candidate_direct", "candidate_routed")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def training_summaries(
    report: dict[str, Any], tasks: list[dict[str, str]], seeds: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """Reconstruct complete TRAIN populations without invoking an evaluator."""
    _require(report["schema"] == core.SCHEMA and report["state"] == "trained",
             "The source must be a completed, unselected TRAIN search")
    _require(report["selection"] is None and report["test"] is None,
             "Validation or TEST outcomes cannot enter candidate preparation")
    ids = {task["task_id"] for task in tasks}
    _require(len(ids) == len(tasks) and bool(ids) and bool(seeds)
             and len(set(seeds)) == len(seeds), "Invalid frozen TRAIN task/seed population")
    expected = {(task, seed) for task in ids for seed in seeds}
    populations: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    request_ids: set[str] = set()
    for entry in report["ledger"]:
        _require(entry["phase"] == "train", "The source ledger contains non-TRAIN calls")
        _require(entry["kind"] in ("executor", "optimizer"), "Unknown source callback role")
        ident = entry["request_id"]
        _require(ident not in request_ids, "Duplicate source callback identity")
        request_ids.add(ident)
        if entry["kind"] != "executor":
            continue
        _require(entry["kind"] == "executor" and entry["task_id"] in ids
                 and type(entry["seed"]) is int and entry["seed"] in seeds,
                 "Source actor call escapes the frozen TRAIN population")
        observed = report["observations"].get(ident)
        _require(isinstance(observed, dict), "Missing committed TRAIN actor result")
        _require(observed["task_id"] == entry["task_id"]
                 and observed["request_id"] == ident, "Source callback/task identity changed")
        score = observed["native_score"]
        _require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                 "TRAIN requires a measured native quality value, including failures")
        usage = core._usage(observed["usage"])
        _require(type(usage["total_tokens"]) is int and usage["total_tokens"] > 0
                 and usage == core._usage(entry["usage"]), "TRAIN actor accounting is incomplete")
        pair = (entry["task_id"], entry["seed"])
        rows = populations.setdefault(entry["arm"], {})
        _require(pair not in rows, "A TRAIN task/seed was charged more than once")
        rows[pair] = observed
    _require("baseline" in populations, "Missing complete TRAIN baseline")
    summaries = {}
    for arm, rows in populations.items():
        _require(set(rows) == expected, f"Incomplete TRAIN population for {arm}")
        summaries[arm] = {
            "tasks": len(tasks), "observations": len(rows),
            "complete_native_scores": True,
            "mean_native_score": math.fsum(row["native_score"] for row in rows.values()) / len(rows),
            "measured_total_tokens": sum(row["usage"]["total_tokens"] for row in rows.values()),
            "residual_applications": len(rows) if arm != "baseline" else 0,
        }
    return summaries


class MatchedRoutingController(core.CodexChiselController):
    """Reuse native callbacks; vary only the application of one frozen instruction."""

    def prepare(
        self, report: dict[str, Any], train_tasks: list[dict[str, str]], *,
        parent_report_sha256: str,
    ) -> dict[str, Any]:
        self._require("initialized")
        _require(len(parent_report_sha256) == 64
                 and all(char in "0123456789abcdef" for char in parent_report_sha256),
                 "The original TRAIN report needs its byte-level SHA256 binding")
        tasks = self._tasks(train_tasks, "train")
        summaries = training_summaries(report, tasks, self._config.execution_seeds)
        _require(all(
            report["observations"][entry["request_id"]]["identity"] == self._execution_identity
            for entry in report["ledger"] if entry["kind"] == "executor"
        ), "The source TRAIN actor differs from the matched native runtime")
        bank = report["candidate_bank"]
        _require(isinstance(bank, list) and bool(bank), "The TRAIN candidate bank is not frozen")
        candidates, decisions = {}, {}
        for candidate in bank:
            _require(candidate["candidate_sha256"] == canonical_hash({
                key: value for key, value in candidate.items() if key != "candidate_sha256"
            }), "A frozen TRAIN candidate was modified")
            if candidate["id"] == "noop":
                continue
            ident = candidate["id"]
            _require(ident not in candidates and ident in summaries,
                     "Duplicate candidate or missing native TRAIN evaluation")
            _require(isinstance(candidate["instruction"], str) and bool(candidate["instruction"].strip())
                     and isinstance(candidate["head"], dict)
                     and isinstance(candidate["feature_bounds"], dict),
                     "A matched candidate requires its original instruction and fitted routing head")
            candidates[ident] = candidate
            decisions[ident] = efficiency_gate(summaries["baseline"], summaries[ident])
        _require(set(summaries) == {"baseline", *candidates},
                 "TRAIN evaluations and candidate identities do not reconcile")
        eligible = [key for key, decision in decisions.items() if decision["accepted"]]
        _require(bool(eligible), "No TRAIN candidate preserves quality while reducing total actor tokens")
        selected_id = min(eligible, key=lambda key: (
            summaries[key]["measured_total_tokens"], -summaries[key]["mean_native_score"], key,
        ))
        selected = deepcopy(candidates[selected_id])
        routed = deepcopy(selected)
        routed["enabled"] = True
        routed["admission"] = "prospective_complete_train_quality_cost_gate"
        routed["historical_candidate_sha256"] = selected["candidate_sha256"]
        routed["candidate_sha256"] = canonical_hash({
            key: value for key, value in routed.items() if key != "candidate_sha256"
        })
        _require(selected["instruction"] == routed["instruction"]
                 and selected["head"] == routed["head"]
                 and selected["feature_bounds"] == routed["feature_bounds"],
                 "The direct/routed comparison must preserve instruction and routing bytes")
        self._matched_candidates = {
            "baseline": deepcopy(self._noop),
            "candidate_direct": selected,
            "candidate_routed": routed,
        }
        self._matched_freeze = {
            "schema": SCHEMA, "run_id": self._run_id, "phase": "validation",
            "parent_report_sha256": parent_report_sha256,
            "source_candidate_id": selected_id,
            "instruction_sha256": hashlib.sha256(selected["instruction"].encode()).hexdigest(),
            "modes": MODES,
            "policies": deepcopy(self._matched_candidates),
            "training_summaries": summaries, "training_gates": decisions,
            "training_selection": "lowest complete actor tokens among quality-nondecreasing candidates",
            "historical_admission": selected["admission"],
            "historical_enabled": selected["enabled"],
            "routed_head_refitted": False,
            "routed_admission_changed_prospectively": True,
            "legacy_rejected_policy_alias": "baseline" if not selected["enabled"] else None,
            "partition_sha256": self._partition_sha256,
            "task_index_sha256": self._task_index_sha256,
            "execution_seeds": self._config.execution_seeds,
            "test_access": False, "optimization_from_validation": False,
            "cumulative_token_limit": None,
            "new_training_or_optimizer_calls": 0,
            "inherited_development_accounting": report["accounting"],
            "inherited_predecessor_accounting": report.get("predecessor_accounting"),
        }
        self._matched_freeze["freeze_sha256"] = canonical_hash(self._matched_freeze)
        self._state = "matched_prepared"
        return deepcopy(self._matched_freeze)

    def _verify_matched_freeze(self) -> None:
        freeze = self._matched_freeze
        _require(freeze["freeze_sha256"] == canonical_hash({
            key: value for key, value in freeze.items() if key != "freeze_sha256"
        }) and freeze["policies"] == self._matched_candidates,
            "The prospective comparison changed after freezing")

    def evaluate_matched_validation(
        self, tasks: list[dict[str, str]], *,
        on_gate: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._require("matched_prepared")
        self._verify_matched_freeze()
        tasks = self._tasks(tasks, "validation")
        self._begin("matched_prepared", "matched_validation")
        ranked = sorted(tasks, key=lambda task: canonical_hash([
            self._run_id, "validation-gate", task["task_id"],
        ]))
        groups = []
        for task in ranked:
            for seed in self._config.execution_seeds:
                order = sorted(MODES, key=lambda mode: canonical_hash([
                    self._run_id, task["task_id"], seed, mode,
                ]))
                groups.append([
                    core._ExecutionCall(
                        task, self._matched_candidates[mode], "validation", mode, seed,
                        force_residual=mode == "candidate_direct",
                    ) for mode in order
                ])
        rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        self._matched_rows = rows

        def retain(plans, observed):
            for plan, results in zip(plans, observed, strict=True):
                for call, row in zip(plan, results, strict=True):
                    _require(row["result"]["native_score"] is not None
                             and row["result"]["usage"]["total_tokens"] is not None,
                             "Native validation outcome or actor usage is unresolved")
                    if call.arm == "candidate_direct":
                        row["decision"]["reason"] = "prospectively_frozen_direct_application"
                    row["comparison_mode"] = call.arm
                    rows[call.arm].append(row)

        try:
            retain(groups[:1], self._execute_groups(groups[:1]))
            self._verify_matched_freeze()
            if on_gate is not None:
                on_gate({
                    "schema": SCHEMA, "status": "VERIFIED_FIRST_MATCHED_NATIVE_GROUP",
                    "task_id": groups[0][0].task["task_id"],
                    "rows": deepcopy(rows), "counts_in_complete_population": True,
                    "outcome_based_selection": False, "freeze_sha256": self._matched_freeze["freeze_sha256"],
                })
            retain(groups[1:], self._execute_groups(groups[1:]))
            self._verify_matched_freeze()
            summaries = {
                mode: self._summarize(values, self._matched_candidates[mode])
                for mode, values in rows.items()
            }
            expected = {(task["task_id"], seed) for task in tasks for seed in self._config.execution_seeds}
            for mode, values in rows.items():
                _require(len(values) == len(expected)
                         and {(row["task_id"], row["seed"]) for row in values} == expected,
                         f"Matched validation is incomplete or duplicated for {mode}")
            result = {
                "schema": SCHEMA, "status": "COMPLETE_MATCHED_VALIDATION_NOT_TEST",
                "freeze_sha256": self._matched_freeze["freeze_sha256"],
                "summaries": summaries,
                "common_quality_cost_gates": {
                    mode: efficiency_gate(summaries["baseline"], summaries[mode])
                    for mode in MODES if mode != "baseline"
                },
                "new_execution_accounting": self.accounting(),
                "inherited_development_accounting": self._matched_freeze["inherited_development_accounting"],
                "test_access": False, "optimization_from_validation": False,
                "paper_admitted": False,
            }
            result["result_sha256"] = canonical_hash(result)
            self._state = "matched_validation_complete"
            self._matched_result = result
            return deepcopy(result)
        except Exception:
            self._state = "blocked"
            raise

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["matched_comparison"] = {
            "schema": SCHEMA,
            "freeze": deepcopy(getattr(self, "_matched_freeze", None)),
            "rows": deepcopy(getattr(self, "_matched_rows", None)),
            "result": deepcopy(getattr(self, "_matched_result", None)),
            "test_access": False,
        }
        return report

    def train(self, *args, **kwargs):
        raise RuntimeError("This diagnostic reuses frozen TRAIN evidence; it cannot run another optimizer")

    def select_validation(self, *args, **kwargs):
        raise RuntimeError("Use the prospectively frozen matched validation comparison")

    def evaluate_test(self, *args, **kwargs):
        raise RuntimeError("The previously opened GAIA2 TEST is sealed to this development diagnostic")

    def freeze(self, *args, **kwargs):
        raise RuntimeError("The matched diagnostic is not an original-policy deployment release")


__all__ = ["MODES", "SCHEMA", "MatchedRoutingController", "training_summaries"]
