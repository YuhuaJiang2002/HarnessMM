"""TRAIN-only instruction RSI on a source-bound capable industrial coding parent.

This reuses the residual compiler and cross-fitted lexical routing head. It is
not the CoT/Jev action-bank selector: native hosted callbacks expose neither a
verified finite-label probability contract nor a zero-decoding readout.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

from phiagent.harness import codex_chisel as core
from phiagent.harness.industrial_efficiency_rsi import IndustrialEfficiencyController, OBJECTIVE
from phiagent.learning.residual_meta_harness import canonical_hash

SCHEMA = "industrial-terminal-train-instruction-rsi/v1"
IDENTITY = {
    "component": "TRAIN-grounded procedural instruction RSI",
    "slow_component": "same-backbone native proposal callback",
    "runtime_decision_layer": "cross-fitted lexical linear routing head with abstention",
    "runtime_selector_model_calls": 0,
    "full_cot_jev": False,
    "fast_activation": False,
    "jev_dependency": "verified same-backbone finite-choice probability/readout callback",
}
CODING_OBJECTIVE = (
    "The complete capable native Terminal-Bench TRAIN parent is the reference. "
    "Prioritize preserving or increasing official native task successes; compare "
    "complete all-attempt actor plus control tokens second. More successes at "
    "higher cost are a quality/cost tradeoff, not joint efficiency. "
    "Propose one reusable short coding procedure supported by at least two "
    "distinct supplied TRAIN scenarios. Do not weaken checks or completion criteria. "
    "The shared completion prompt, model alias, native interface, tool schemas, "
    "output and time limits are frozen for both arms. "
    + OBJECTIVE
)


def quality_cost_gate(baseline: dict, candidate: dict, *, required_tasks: int = 18) -> dict:
    """Admit only complete matched populations with all-attempt execution costs."""
    for label, row in (("baseline", baseline), ("candidate", candidate)):
        ids = row.get("task_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError(f"{label} requires explicit task identities")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{label} contains duplicate task identities")
        count = row.get("native_successes")
        if type(count) is not int or not 0 <= count <= len(ids):
            raise ValueError(f"{label} requires an actual native success count")
        tokens = row.get("all_attempt_actor_control_tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError(f"{label} requires measured nonnegative integer tokens or null")
    complete = (
        len(baseline["task_ids"]) == required_tasks
        and set(baseline["task_ids"]) == set(candidate["task_ids"])
        and all(row.get("native_scored") == required_tasks
                and row.get("all_attempt_usage_complete") is True
                and row["all_attempt_actor_control_tokens"] is not None
                for row in (baseline, candidate))
    )
    if not complete:
        return {"accepted": False, "status": "INCOMPLETE_NATIVE_QUALITY_OR_ALL_ATTEMPT_COST"}
    quality_delta = candidate["native_successes"] - baseline["native_successes"]
    token_delta = (
        candidate["all_attempt_actor_control_tokens"]
        - baseline["all_attempt_actor_control_tokens"]
    )
    applied = candidate.get("residual_applications", 0) > 0
    accepted = quality_delta >= 0 and token_delta < 0 and applied
    status = (
        "JOINT_EFFICIENCY" if accepted else
        "QUALITY_REGRESSION" if quality_delta < 0 else
        "QUALITY_COST_TRADEOFF" if quality_delta > 0 and token_delta >= 0 else
        "NO_JOINT_GAIN"
    )
    return {
        "accepted": accepted, "status": status,
        "native_success_delta": quality_delta, "all_attempt_token_delta": token_delta,
        "development_cost_included_in_execution": False,
        "scope": "complete matched TRAIN population; protected validation still required",
    }


class IndustrialTrainRSIController(IndustrialEfficiencyController):
    """Reuse unchanged admission thresholds; retain both quality and saving candidates."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._coding_source = Path(__file__).resolve()
        self._coding_sha256 = hashlib.sha256(self._coding_source.read_bytes()).hexdigest()
        self._train_freeze = None

    def _verify_sources(self):
        super()._verify_sources()
        if hasattr(self, "_coding_source"):
            if hashlib.sha256(self._coding_source.read_bytes()).hexdigest() != self._coding_sha256:
                raise RuntimeError("Frozen industrial coding RSI source changed")
        if getattr(self, "_train_freeze", None) is not None:
            if canonical_hash(self.candidate_bank) != self._train_freeze["candidate_bank_sha256"]:
                raise RuntimeError("The TRAIN-frozen candidate bank changed")

    def _invoke(self, callback, request, kind, **kwargs):
        if kind == "optimizer":
            request = deepcopy(request)
            request["objective"] = CODING_OBJECTIVE.replace(
                "official Codex CLI",
                ("official DeepSeek Harness SDK"
                 if self._execution_identity["runtime"] == "official_deepseek_harness_sdk"
                 else "official Codex CLI"),
            )
            original = callback

            def checked(payload):
                result = original(payload)
                if (isinstance(result, dict) and result.get("execution_status") == "completed"
                        and isinstance(result.get("instruction"), str)
                        and result["instruction"].strip()):
                    support = result.get("supporting_task_ids")
                    if (not isinstance(support, list)
                            or any(not isinstance(item, str) for item in support)
                            or len(set(support)) < 2 or len(set(support)) != len(support)
                            or not set(support) <= self._optimizer_exposed_tasks):
                        result = deepcopy(result)
                        result.setdefault("provenance", {})["rejected_training_support"] = {
                            "instruction": result["instruction"], "supporting_task_ids": support,
                            "reason": "Require two distinct exposed TRAIN scenarios; usage retained",
                        }
                        result.update(
                            execution_status="failed", instruction="", supporting_task_ids=[],
                        )
                return result

            callback = checked
        return core.CodexChiselController._invoke(self, callback, request, kind, **kwargs)

    def _fit_candidate(self, proposal, tasks, baseline, observed):
        quality = core.CodexChiselController._fit_candidate(
            self, proposal, tasks, baseline, observed,
        )
        efficiency = super()._fit_candidate(proposal, tasks, baseline, observed)
        population = efficiency["training_population_quality"]
        quality_ok = quality["enabled"] and population is not None and population["nonregression"]
        selected = deepcopy(quality if quality_ok else efficiency)
        selected.pop("candidate_sha256")
        selected["industrial_coding_admission"] = {
            "rule": "unchanged quality OOF gate first; unchanged efficiency OOF gate otherwise",
            "quality_admission": quality["admission"],
            "efficiency_admission": efficiency["admission"],
            "selected_head": "quality" if quality_ok else "efficiency",
            "training_population_quality": population,
            "thresholds_lowered": False,
        }
        return core._seal(selected, "candidate_sha256")

    def freeze_train_bank(self, *, sources: dict, shared_contract: dict) -> dict:
        self._require("trained")
        self._verify_sources()
        if self._train_freeze is None:
            document = {
                "schema": SCHEMA, "run_id": self._run_id,
                "status": "FROZEN_FOR_SEPARATE_PROTECTED_VALIDATION",
                "component_identity": IDENTITY,
                "candidate_bank_sha256": canonical_hash(self.candidate_bank),
                "candidate_bank": self.candidate_bank, "config": asdict(self._config),
                "sources": deepcopy(sources), "shared_contract": deepcopy(shared_contract),
                "coding_source_sha256": self._coding_sha256,
                "objective_sha256": hashlib.sha256(CODING_OBJECTIVE.encode()).hexdigest(),
                "validation_read": False, "test_read": False,
            }
            self._train_freeze = core._seal(document, "train_freeze_sha256")
        elif (self._train_freeze["sources"] != sources
              or self._train_freeze["shared_contract"] != shared_contract):
            raise ValueError("A frozen method cannot accept changed sources or native contracts")
        return deepcopy(self._train_freeze)

    def select_validation(self, *args, **kwargs):
        raise RuntimeError("Protected validation requires a separate owner-admitted frozen run")

    def evaluate_frozen_train(self, tasks, candidate_id):
        self._require("trained")
        if self._train_freeze is None:
            raise RuntimeError("Freeze the method/config before routed TRAIN execution")
        tasks = self._tasks(tasks, "train")
        candidates = [row for row in self.candidate_bank if row["id"] == candidate_id]
        if len(candidates) != 1 or not candidates[0]["enabled"]:
            raise ValueError("Routed execution requires an admitted frozen TRAIN candidate")
        return self._batch(tasks, candidates[0], "train", "routed-" + candidate_id)

    def report(self):
        report = super().report()
        report.pop("industrial_efficiency_successor", None)
        report["industrial_coding_train_rsi"] = {
            **IDENTITY, "schema": SCHEMA, "validation_read": False, "test_read": False,
            "coding_source_sha256": self._coding_sha256,
            "objective_sha256": hashlib.sha256(CODING_OBJECTIVE.encode()).hexdigest(),
            "train_freeze_sha256": (
                self._train_freeze["train_freeze_sha256"] if self._train_freeze else None
            ),
        }
        return report
