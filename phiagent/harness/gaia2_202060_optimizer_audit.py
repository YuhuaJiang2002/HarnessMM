"""Strict all-success evidence for legitimate official AutoSaddler no-op rounds."""
from __future__ import annotations

from phiagent.harness.aime_transport_recovery import autosaddler_completion
from phiagent.harness.gaia2_202060_protocol import native_population_gate


def completion(events, iterations, *, training_ids, native_rows_by_evaluation):
    proof = autosaddler_completion(events, iterations, expected_train_ids=training_ids)
    verified_skips = {}
    for iteration in proof["official_no_training_failure_iterations"]:
        completions = [event["payload"] for event in events
                       if event["event_type"] == "IterationCompleted"
                       and event["payload"].get("iteration") == iteration
                       and event["payload"].get("outcome") == "no_training_failures"]
        if len(completions) != 1:
            raise ValueError("Ambiguous no-training-failures completion")
        parent_ids = completions[0].get("evaluated_candidate_ids", [])
        evaluations = [event["payload"].get("evaluation", {}) for event in events
                       if event["event_type"] == "EvaluationCompleted"]
        matches = [value for value in evaluations
                   if value.get("iteration") == iteration and value.get("split") == "train"
                   and value.get("purpose") == "train_before"
                   and value.get("candidate_id") in parent_ids]
        accepted = []
        for evaluation in matches:
            observations = evaluation.get("observations", [])
            if (len(observations) != len(training_ids)
                    or {row.get("case_id") for row in observations} != set(training_ids)
                    or any(row.get("disposition") != "success" or row.get("score") != 1
                           for row in observations)):
                continue
            ident = evaluation["evaluation_id"]
            native = native_rows_by_evaluation.get(ident)
            if native is None:
                continue
            count = native_population_gate(native, training_ids)
            if count["successes"] != len(training_ids) * 3:
                continue
            accepted.append(ident)
        if not accepted:
            raise ValueError("No-op optimizer round lacks complete all-success32x3 native evidence")
        verified_skips[str(iteration)] = accepted
    return {**proof, "native_all_success_skip_evaluations": verified_skips,
            "expected_training_scenarios": len(training_ids),
            "native_repetitions_per_scenario": 3}
