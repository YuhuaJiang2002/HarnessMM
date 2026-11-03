"""Compare frozen GAIA2 study contracts without loading task payloads."""

from __future__ import annotations

from copy import deepcopy

PARTITION_SHA256 = "5549b4a72340ae1d4cc883e419dc9d947af17b7a3f69fd1d60950c17108bebec"
COMMON_PROTOCOL = {
    "benchmark": "gaia2-mini",
    "dataset_revision": "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99",
    "partition_sha256": PARTITION_SHA256,
    "split_seed": 20260928,
    "counts": {"train": 32, "validation": 32, "test": 96},
    "independent_unit": "scenario",
    "are_version": "1.1.0",
    "native_metric": "official_are_scenario_task_success",
    "clock": "measured",
    "scenario_turns": "native_scenario_nb_turns",
    "observation": "original_native_tool_text",
    "feedback": "train_only",
    "selection": "validation_scalars_of_train_frozen_candidates",
    "test": "source_and_policy_frozen_complete_population_all_outcomes",
    "actor_cost": "all_attempt_input_plus_output_including_authorship_fast_and_fallback",
    "cumulative_token_limit": None,
    "chisel_design": "slow_cot_authors_candidates_fast_selects_or_judges_and_may_abstain",
}


def requested_protocol() -> dict:
    return deepcopy(COMMON_PROTOCOL)


def audit_alignment(left: dict, right: dict, *, direct_accuracy_comparison: bool = False) -> dict:
    """Separate protocol agreement from actual method admission and model effects.

    Repetition policies and limits must be prospectively registered and identical
    in a claimed aligned comparison. A one-attempt parent capability diagnosis
    remains a separate phase; it cannot become a three-repeat method comparison.
    Native harness/model identity is retained in the supplied run descriptors.
    """
    differences = []
    for name, study in (("figure2", left), ("figure3", right)):
        protocol = study.get("protocol", {})
        for key, expected in COMMON_PROTOCOL.items():
            if key not in protocol or protocol[key] != expected:
                differences.append({"study": name, "field": key, "actual": protocol.get(key),
                                    "required": expected})
        if study.get("scope") != "test":
            differences.append({"study": name, "field": "scope", "actual": study.get("scope"),
                                "required": "test"})
        for key in ("complete_population", "complete_actor_cost", "source_policy_frozen",
                    "native_fast_execution_verified"):
            if study.get(key) is not True:
                differences.append({"study": name, "field": key,
                                    "actual": study.get(key), "required": True})
        identity = study.get("implementation_identity")
        if (not isinstance(identity, dict)
                or any(not isinstance(identity.get(key), str) or not identity[key].strip()
                       for key in ("slow_author", "fast_decision", "source_sha256"))
                or len(identity.get("source_sha256", "")) != 64
                or any(char not in "0123456789abcdef" for char in identity.get("source_sha256", ""))):
            differences.append({"study": name, "field": "implementation_identity",
                                "required": "named slow author, fast decision implementation and frozen SHA256"})
    for key in ("repetitions_by_phase", "native_limits", "shared_prompt"):
        if not left.get(key) or left.get(key) != right.get(key):
            differences.append({"study": "pair", "field": key,
                                "figure2": left.get(key), "figure3": right.get(key)})
    same_model = bool(left.get("model_identity")) and left.get("model_identity") == right.get("model_identity")
    if direct_accuracy_comparison and not same_model:
        differences.append({"study": "pair", "field": "model_identity",
                            "figure2": left.get("model_identity"), "figure3": right.get("model_identity")})
    return {"status": "ALIGNED_VERIFIED" if not differences else "NOT_ALIGNED",
            "differences": differences, "same_backbone": same_model,
            "direct_cross_figure_accuracy_comparison": not differences and same_model,
            "historical_measurements_relabelled": False,
            "benchmark_or_model_calls": 0}
