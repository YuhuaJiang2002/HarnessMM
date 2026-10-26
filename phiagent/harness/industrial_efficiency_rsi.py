"""Prospective cost-aware CHISEL instruction RSI on an unchanged industrial runtime.

The original residual controller remains frozen. This successor changes its
proposal objective, TRAIN routing utility and protected validation selection.
It uses the same native callbacks, public-only features, complete populations,
source seals and exact-once cost ledger.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from phiagent.harness import codex_chisel as core
from phiagent.learning import residual_meta_harness as residual
from phiagent.learning import residual_meta_training as training

OBJECTIVE = (
    "Evolve ONE short reusable procedural skill for the unchanged official Codex CLI. "
    "Both native task quality AND complete-task input-plus-output token cost matter. "
    "Use only supplied TRAIN executions; validation and test are unavailable. "
    "Learn from successful and failed trajectories and every previous candidate, including "
    "rejected edits. Refine an admitted efficient parent if one exists; otherwise repair the "
    "original harness. Return the COMPLETE replacement residual, not another appended layer. "
    "ALFWorld's transferable lesson is a compact task-family procedure that reduces wrong "
    "actions and unnecessary model turns; do not introduce its handwritten solver. "
    "SoL-Pi motivates avoiding repeated large reads and already-resolved decisions, and "
    "combining only independently determined follow-ups when existing native tools support it. "
    "Keep complete callable tool schemas, original observations, errors, user constraints and "
    "uncertain evidence available. Prefer targeted native reads and reuse of established "
    "facts from the visible history, not deleting tool outputs or inventing a summary. "
    "A valid saving must survive the complete task, including failures, recovery and final "
    "verification. A shorter prompt or cached-input discount is not sufficient. "
    "Avoid unconditional polling, repeated whole-state inspection, verbose status narration "
    "and rechecking already-confirmed subgoals. Wait only for an outstanding dependency and "
    "honor native event semantics; sending a user message is not necessarily episode termination. "
    "Preserve exact user requirements: do not substitute an unavailable option or guess a "
    "missing fact merely to finish. Do not hide necessary tools, change output/context limits, "
    "disable reasoning, call another model, bypass native scoring or stop unfinished tasks. "
    "No task-ID conditionals, memorized answers, task-specific facts or benchmark oracle data. "
    "Support the edit with at least two distinct supplied TRAIN task IDs. "
    "Use instruction='' and supporting_task_ids=[] if no supported edit is justified."
)


def efficiency_gate(baseline: dict, candidate: dict) -> dict:
    complete = (
        baseline["complete_native_scores"] and candidate["complete_native_scores"]
        and baseline["measured_total_tokens"] is not None
        and candidate["measured_total_tokens"] is not None
    )
    if not complete:
        return {"accepted": False, "reason": "incomplete_native_quality_or_cost"}
    quality = candidate["mean_native_score"] - baseline["mean_native_score"]
    total = baseline["measured_total_tokens"]
    saving = 1 - candidate["measured_total_tokens"] / total if total else None
    accepted = (
        baseline["mean_native_score"] > 0
        and candidate["residual_applications"] > 0
        and quality >= -1e-12 and saving is not None and saving > 0
    )
    return {
        "accepted": accepted, "quality_delta": quality, "token_reduction_fraction": saving,
        "reason": "native_nonregression_and_fewer_tokens" if accepted else "quality_or_cost_rejected",
        "scope": "complete observed population; not a population-level noninferiority certificate",
    }


def actor_only_result(result: dict) -> dict:
    """Separate the native DeepSeek actor from its already-recorded judge."""
    value = deepcopy(result)
    accounting = value["provenance"]["provider_usage"]
    if accounting["complete"] is not True:
        raise ValueError("Actor-only optimization requires complete native role accounting")
    actor = core._usage(accounting["actor"]["usage"])
    judge = core._usage(accounting["judge"]["usage"])
    if (actor["total_tokens"] is None or judge["total_tokens"] is None
            or value["usage"]["total_tokens"] != actor["total_tokens"] + judge["total_tokens"]):
        raise ValueError("Native actor/judge totals do not reconcile; do not drop recovery costs")
    value["provenance"]["efficiency_cost_roles"] = {
        "original_combined_usage": value["usage"],
        "actor_usage": actor, "judge_usage_separate": judge,
        "all_provider_traffic_retained": True,
    }
    value["usage"] = actor
    return value


class IndustrialEfficiencyController(core.CodexChiselController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._efficiency_source = Path(__file__).resolve()
        self._efficiency_sha256 = hashlib.sha256(self._efficiency_source.read_bytes()).hexdigest()
        self._efficiency_provenance = {
            "source_sha256": self._efficiency_sha256,
            "proposal_objective_sha256": hashlib.sha256(OBJECTIVE.encode()).hexdigest(),
            "unchanged_industrial_runtime": True,
            "selection": "native quality nonregression AND strictly fewer complete-task tokens",
            "training_utility": "half normalized quality delta plus half bounded token saving; "
                                "only native quality losses are harmful; aggregate cost protection",
            "mechanism": "TRAIN-derived procedural instructions; native tools and observations intact",
            "official_sol_pi_reproduction": False,
        }
        self._training_refit = None

    def _verify_sources(self):
        super()._verify_sources()
        if hasattr(self, "_efficiency_source"):
            if hashlib.sha256(self._efficiency_source.read_bytes()).hexdigest() != self._efficiency_sha256:
                raise RuntimeError("Frozen efficiency RSI source changed")
        if getattr(self, "_training_refit", None) is not None:
            core._verify_seal(self._training_refit, "refit_sha256")
            if residual.canonical_hash(self.candidate_bank) != self._training_refit["bank_sha256"]:
                raise RuntimeError("Frozen refit candidate bank changed")

    def _invoke(self, callback, request, kind, *, ledger_sequence=None, evidence_manifest=None):
        if kind == "optimizer":
            request = deepcopy(request)
            request["objective"] = OBJECTIVE
            if self._execution_identity["runtime"] == "official_deepseek_harness_sdk":
                request["objective"] = request["objective"].replace(
                    "official Codex CLI", "official DeepSeek Harness SDK",
                )
            original_callback = callback

            def checked_proposal(payload):
                result = original_callback(payload)
                if (isinstance(result, dict) and result.get("execution_status") == "completed"
                        and isinstance(result.get("instruction"), str) and result["instruction"].strip()
                        and isinstance(result.get("supporting_task_ids"), list)
                        and all(isinstance(item, str) for item in result["supporting_task_ids"])):
                    support = result["supporting_task_ids"]
                    if (len(set(support)) < 2 or len(set(support)) != len(support)
                            or not set(support) <= self._optimizer_exposed_tasks):
                        rejected = deepcopy(result)
                        rejected.setdefault("provenance", {})["rejected_untrusted_proposal"] = {
                            "reason": "Support must be unique and cite at least two exposed TRAIN tasks",
                            "instruction": result["instruction"], "supporting_task_ids": support,
                            "actual_usage_preserved": True,
                        }
                        rejected.update(execution_status="failed", instruction="", supporting_task_ids=[])
                        return rejected
                return result

            callback = checked_proposal
        return super()._invoke(
            callback, request, kind, ledger_sequence=ledger_sequence,
            evidence_manifest=evidence_manifest,
        )

    def _fit_candidate(self, proposal, tasks, baseline, observed):
        features = {task["task_id"]: core.public_text_features(task["public_prompt"]) for task in tasks}
        groups = defaultdict(list)
        for task in tasks:
            groups[task["unit_id"]].append(task)
        complete = all(
            row["result"]["native_score"] is not None
            and row["result"]["usage"]["total_tokens"] is not None
            for rows in (baseline, observed) for row in rows.values()
        )
        admission, head = "incomplete_measured_paired_evidence", None
        population_quality = None
        examples = []
        if complete:
            differences = [
                observed[key]["result"]["native_score"] - baseline[key]["result"]["native_score"]
                for key in sorted(baseline)
            ]
            population_quality = {
                "baseline_score_sum": math.fsum(
                    row["result"]["native_score"] for row in baseline.values()
                ),
                "candidate_score_sum": math.fsum(
                    row["result"]["native_score"] for row in observed.values()
                ),
                "quality_loss_pairs": sum(delta < 0 for delta in differences),
                "quality_gain_pairs": sum(delta > 0 for delta in differences),
                "nonregression": math.fsum(differences) >= -1e-12,
            }
            for unit, members in sorted(groups.items()):
                pairs = [
                    (baseline[task["task_id"], seed]["result"],
                     observed[task["task_id"], seed]["result"])
                    for task in members for seed in self._config.execution_seeds
                ]
                quality = [right["native_score"] - left["native_score"] for left, right in pairs]
                costs = [
                    (right["usage"]["total_tokens"] - left["usage"]["total_tokens"])
                    / max(1, left["usage"]["total_tokens"]) for left, right in pairs
                ]
                examples.append({
                    "task_id": unit,
                    "features": {
                        name: math.fsum(features[task["task_id"]][name] for task in members)
                        / len(members) for name in sorted(core._feature_schema())
                    },
                    "utility": math.fsum(
                        delta / (self._config.score_maximum - self._config.score_minimum)
                        - max(-1.0, min(1.0, cost))
                        for delta, cost in zip(quality, costs)
                    ) / (2 * len(pairs)),
                    "harmful": any(delta < 0 for delta in quality),
                    "token_delta": math.fsum(max(-1., min(1., cost)) for cost in costs) / len(pairs),
                })
            head = training.fit_cross_fitted_linear_head(
                examples, feature_schema=core._feature_schema(), folds=self._config.crossfit_folds,
                seed=core.SPLIT_SEED, ridge=self._config.ridge,
                minimum_override_support=self._config.minimum_override_support,
            )
            admission = (
                "admitted_training_oof"
                if head["status"] == "DISCOVERY_CANDIDATE"
                and head["oof_mean_utility"] > 0
                and head["oof_harmful_rate"] <= self._config.maximum_oof_harmful_rate
                and head["oof_mean_token_delta"] < 0
                and population_quality["nonregression"]
                else "no_quality_preserving_oof_token_saving"
            )
            if not population_quality["nonregression"]:
                admission = "training_native_quality_regression"
        return core._seal({
            **proposal, "head": head, "enabled": admission == "admitted_training_oof",
            "admission": admission, "training_units": len(groups), "training_tasks": len(tasks),
            "paired_training_observations": len(observed),
            "paired_evidence_sha256": residual.canonical_hash({
                "baseline": [baseline[key] for key in sorted(baseline)],
                "candidate": [observed[key] for key in sorted(observed)],
            }),
            "efficiency_training_examples": examples,
            "training_population_quality": population_quality,
            "feature_bounds": {
                name: [min(row[name] for row in features.values()),
                       max(row[name] for row in features.values())]
                for name in sorted(core._feature_schema())
            },
        }, "candidate_sha256")

    def refit_training_report(
        self, tasks, *, source_report: bytes, source_report_path: str,
        native_evidence_sha256: str,
    ):
        """Refit frozen TRAIN instructions without invoking either native callback.

        The original ledger, observations, failed proposals and hashes remain
        historical evidence. Only derived routing heads and admission change.
        The caller separately verifies native files and seals that evidence.
        """
        self._require("initialized")
        tasks = self._tasks(tasks, "train")
        if (self._config.maximum_oof_harmful_rate != 0
                or self._config.optimizer_max_units is not None):
            raise ValueError("Refit requires zero quality harm and complete TRAIN evidence")
        if (not isinstance(source_report, bytes) or not isinstance(source_report_path, str)
                or not source_report_path or not isinstance(native_evidence_sha256, str)
                or not core._SHA256.fullmatch(native_evidence_sha256)):
            raise ValueError("Refit requires original report bytes, a source path and evidence hash")
        original = json.loads(source_report)
        if (original.get("schema") != core.SCHEMA or original.get("state") != "trained"
                or original.get("selection") is not None or original.get("test") is not None
                or original.get("test_pair_orders")):
            raise ValueError("Only complete TRAIN reports with sealed validation and TEST can refit")
        source_run = original.get("run_id")
        if not isinstance(source_run, str) or not source_run or source_run == self._run_id:
            raise ValueError("Refit requires a new run identity")
        accounting = original["accounting"]
        if (accounting["execution_seeds"] != list(self._config.execution_seeds)
                or accounting["cumulative_token_limit"] is not None
                or accounting["optimizer_max_units"] is not None
                or accounting["optimization_iterations_predeclared"]
                != self._config.optimization_iterations):
            raise ValueError("Refit must retain seeds, complete TRAIN scope and paid proposal count")
        ledger, observations = original["ledger"], original["observations"]
        if not isinstance(ledger, list) or not isinstance(observations, dict) or not ledger:
            raise ValueError("Refit requires a complete ledger and observation mapping")
        ids = [row["request_id"] for row in ledger]
        sequences = [row["sequence"] for row in ledger]
        if (len(set(ids)) != len(ids) or set(ids) != set(observations)
                or len(set(sequences)) != len(sequences)
                or any(type(value) is not int or value < 0 for value in sequences)):
            raise ValueError("Missing or duplicated paid TRAIN records")
        by_arm = defaultdict(dict)
        task_map = {task["task_id"]: task for task in tasks}
        expected_pairs = {
            (task["task_id"], seed) for task in tasks for seed in self._config.execution_seeds
        }
        optimizer_count = 0
        for entry in ledger:
            if entry["phase"] != "train" or entry["kind"] not in {"executor", "optimizer"}:
                raise ValueError("Only TRAIN actor and proposer callbacks may enter refit")
            value = observations[entry["request_id"]]
            if (value["request_id"] != entry["request_id"]
                    or residual.canonical_hash(value) != entry.get("result_sha256")
                    or core._usage(value["usage"]) != entry["usage"]
                    or value["usage"]["total_tokens"] is None
                    or entry["execution_status"] != value["execution_status"]):
                raise ValueError("Original TRAIN hashes, measured usage or completion differ")
            if entry["kind"] == "optimizer":
                optimizer_count += 1
                if value["identity"] != self._optimizer_identity:
                    raise ValueError("Refit optimizer identity differs from its original evidence")
                continue
            key = (entry["task_id"], entry["seed"])
            if (key not in expected_pairs or key in by_arm[entry["arm"]]
                    or entry["unit_id"] != task_map[entry["task_id"]]["unit_id"]
                    or value["task_id"] != entry["task_id"]
                    or value["identity"] != self._execution_identity
                    or entry["pair_id"] != residual.canonical_hash([
                        source_run, "train", entry["task_id"], entry["seed"],
                    ])):
                raise ValueError("TRAIN task, group, seed, native identity or pairing changed")
            score = value["native_score"]
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or not self._config.score_minimum <= score <= self._config.score_maximum):
                raise ValueError("Refit requires every original native TRAIN score")
            by_arm[entry["arm"]][key] = {
                "task_id": entry["task_id"], "unit_id": entry["unit_id"], "seed": entry["seed"],
                "decision": (
                    core._route(self._noop, task_map[entry["task_id"]]["public_prompt"])
                    if entry["arm"] == "baseline" else {
                        "residual_id": entry["arm"], "reason": "paired_training_intervention",
                        "score": None,
                    }
                ),
                "result": deepcopy(value),
            }
        if optimizer_count != self._config.optimization_iterations:
            raise ValueError("Refit requires every completed or failed paid proposer callback")
        bank = original["candidate_bank"]
        if (not isinstance(bank, list) or not bank or bank[0] != self._noop
                or len({candidate["id"] for candidate in bank}) != len(bank)
                or set(by_arm) != {"baseline", *(candidate["id"] for candidate in bank[1:])}
                or any(set(rows) != expected_pairs for rows in by_arm.values())):
            raise ValueError("Refit requires the original noop and all complete matched TRAIN arms")
        optimizer_evidence = original["optimizer_evidence"]
        if (len(optimizer_evidence) != optimizer_count
                or residual.canonical_hash(optimizer_evidence)
                != original["optimizer_evidence_sha256"]):
            raise ValueError("Original optimizer exposure evidence is incomplete")
        exposed = set()
        for evidence in optimizer_evidence:
            core._verify_seal(evidence, "evidence_selection_sha256")
            if (evidence["split"] != "train" or evidence["run_id"] != source_run
                    or evidence["execution_seeds"] != list(self._config.execution_seeds)
                    or set(evidence["task_ids"]) != set(task_map)
                    or evidence["task_commitments_sha256"] != residual.canonical_hash([
                        self._index[key] for key in evidence["task_ids"]
                    ])):
                raise ValueError("Refit may use only the originally exposed TRAIN population")
            exposed.update(evidence["task_ids"])
        proposals = []
        for candidate in bank:
            core._verify_seal(candidate, "candidate_sha256")
            if candidate["id"] == "noop":
                intervention = self._noop
                arm = "baseline"
            else:
                support = candidate["supporting_task_ids"]
                response = observations.get(candidate["optimizer_request_id"])
                head = candidate.get("head")
                if (candidate["id"] != core._hash_text(candidate["instruction"])
                        or not isinstance(support, list) or len(set(support)) < 2
                        or len(set(support)) != len(support) or not set(support) <= exposed
                        or response is None or response["execution_status"] != "completed"
                        or candidate["instruction"] != response["instruction"].strip()
                        or sorted(support) != sorted(response["supporting_task_ids"])
                        or candidate["optimizer_result_sha256"] != residual.canonical_hash(response)):
                    raise ValueError("Original learned instruction, support or proposer hash changed")
                if (not isinstance(head, dict)
                        or head["grouped_crossfit_folds"] != self._config.crossfit_folds
                        or head["ridge"] != self._config.ridge):
                    raise ValueError("Refit must retain the same grouped linear router")
                proposal = {key: deepcopy(candidate[key]) for key in (
                    "id", "instruction", "supporting_task_ids", "origin", "iteration",
                    "optimizer_request_id", "optimizer_result_sha256",
                )}
                intervention = core._seal({**proposal, "enabled": False}, "candidate_sha256")
                arm = candidate["id"]
                paired_hash = residual.canonical_hash({
                    "baseline": [by_arm["baseline"][key] for key in sorted(expected_pairs)],
                    "candidate": [by_arm[arm][key] for key in sorted(expected_pairs)],
                })
                if paired_hash != candidate["paired_evidence_sha256"]:
                    raise ValueError("Reconstructed native TRAIN pairs differ from original evidence")
                proposals.append(proposal)
            if any(entry["candidate_sha256"] != intervention["candidate_sha256"]
                   for entry in ledger if entry["kind"] == "executor" and entry["arm"] == arm):
                raise ValueError("Native TRAIN intervention differs from its original instruction")
        self._begin("initialized", "refitting")
        try:
            refitted = [
                self._fit_candidate(proposal, tasks, by_arm["baseline"], by_arm[proposal["id"]])
                for proposal in proposals
            ]
            frozen_bank = [self._noop, *sorted(refitted, key=lambda item: item["id"])]
            source_candidates = {candidate["id"]: candidate for candidate in bank}
            lineage = core._seal({
                "schema": "industrial-efficiency-training-refit/v1",
                "source_run_id": source_run, "run_id": self._run_id,
                "source_report_path": source_report_path,
                "source_report_sha256": hashlib.sha256(source_report).hexdigest(),
                "native_evidence_sha256": native_evidence_sha256,
                "source_bank_sha256": residual.canonical_hash(bank),
                "bank_sha256": residual.canonical_hash(frozen_bank),
                "original_observations_sha256": residual.canonical_hash(observations),
                "source_predecessor_accounting": deepcopy(original.get("predecessor_accounting")),
                "imported_actor_episodes": len(ledger) - optimizer_count,
                "imported_proposer_callbacks": optimizer_count,
                "new_train_callbacks": 0, "new_train_model_calls": 0,
                "original_raw_hashes_and_failed_proposal_costs_retained": True,
                "harm_definition": "native quality loss only",
                "bank_frozen_before_validation": True,
                "candidates": [{
                    "id": candidate["id"],
                    "source_candidate_sha256": source_candidates[candidate["id"]]["candidate_sha256"],
                    "refit_candidate_sha256": candidate["candidate_sha256"],
                    "source_admission": source_candidates[candidate["id"]]["admission"],
                    "refit_admission": candidate["admission"],
                } for candidate in refitted],
            }, "refit_sha256")
        except Exception:
            with self._lock:
                self._state = "blocked"
            raise
        with self._lock:
            self._ledger = deepcopy(ledger)
            self._observations = deepcopy(observations)
            self._request_ids = set(ids)
            self._next_sequence = max(sequences) + 1
            self._iterations = deepcopy(original["optimization_iterations"])
            self._optimizer_evidence = deepcopy(optimizer_evidence)
            self._optimizer_exposed_tasks = exposed
            self._executor_failures = deepcopy(original["executor_failures"])
            self._train_evidence_sha256 = residual.canonical_hash(observations)
            self._bank_json = residual.canonical_json(frozen_bank)
            self._training_refit = lineage
            self._state = "trained"
        self._require("trained")
        return self.report()

    def select_validation(self, tasks):
        self._require("trained")
        tasks = self._tasks(tasks, "validation")
        self._begin("trained", "selecting")
        bank = self.candidate_bank
        results = [
            self._batch(tasks, candidate, "validation", candidate["id"])
            for candidate in bank if candidate["id"] == "noop" or candidate["enabled"]
        ]
        baseline = results[0]
        decisions = {
            row["candidate_id"]: efficiency_gate(baseline, row) for row in results[1:]
        }
        contenders = [row for row in results[1:] if decisions[row["candidate_id"]]["accepted"]]
        selected = min(contenders, key=lambda row: (
            -row["mean_native_score"], row["measured_total_tokens"], row["candidate_sha256"],
        )) if contenders else baseline
        self._selection = {
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_sha256": selected["candidate_sha256"],
            "bank_sha256": residual.canonical_hash(bank),
            "rule": "native quality nonregression AND strictly fewer all-in tokens; then quality/cost/hash",
            "reason": "quality_preserving_token_saving" if contenders else "retain_empty_baseline",
            "validation_used_for_optimization": False, "test_opened": False,
            "efficiency_gates": decisions, "results": results,
        }
        self._state = "selected"
        self._require("selected")
        return deepcopy(self._selection)

    def evaluate_test(self, *args, **kwargs):
        raise RuntimeError(
            "This successor is TRAIN/validation development only; the previously opened TEST "
            "cannot be reused as independent confirmation or drive further evolution"
        )

    def freeze(self, *, source_hashes):
        self._verify_sources()
        hashes = dict(source_hashes)
        if hashes.get("industrial_efficiency_rsi", self._efficiency_sha256) != self._efficiency_sha256:
            raise ValueError("Efficiency successor freeze differs from its actual source")
        hashes["industrial_efficiency_rsi"] = self._efficiency_sha256
        hashes["efficiency_objective"] = self._efficiency_provenance["proposal_objective_sha256"]
        return super().freeze(source_hashes=hashes)

    def report(self):
        result = super().report()
        result["industrial_efficiency_successor"] = deepcopy(self._efficiency_provenance)
        result["heldout_test_permitted"] = False
        if self._training_refit is not None:
            result["training_refit"] = deepcopy(self._training_refit)
            result["optimization_iterations_scope"] = "Imported historical TRAIN proposer callbacks"
        return result
