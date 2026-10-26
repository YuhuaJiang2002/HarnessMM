"""Deterministic component-wise and compositional TNC decisions.

The functions in this module do not call a model or simulator.  They turn
paired official-success summaries into auditable local certificates and typed
conditional-dependency edges.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def interaction_quantities(
    *,
    full: float,
    delete_a: float,
    delete_b: float,
    delete_a_b: float,
) -> dict[str, float]:
    """Return deletion regrets and two explicit interaction quantities.

    ``pair_interaction`` is the super-additive deletion regret requested for the
    typed graph: joint regret minus the two single regrets.  ``interaction_gap``
    is the stricter operational gap already used by the ManiSkill screen: the
    worse safe-single success rate minus joint-deletion success.
    """

    rates = {
        "full": float(full),
        "delete_a": float(delete_a),
        "delete_b": float(delete_b),
        "delete_a_b": float(delete_a_b),
    }
    if any(not 0.0 <= value <= 1.0 for value in rates.values()):
        raise ValueError("success rates must be in [0, 1]")
    single_a_drop = rates["full"] - rates["delete_a"]
    single_b_drop = rates["full"] - rates["delete_b"]
    joint_drop = rates["full"] - rates["delete_a_b"]
    pair_interaction = joint_drop - single_a_drop - single_b_drop
    return {
        "single_a_drop": single_a_drop,
        "single_b_drop": single_b_drop,
        "joint_drop": joint_drop,
        "interaction_gap": min(rates["delete_a"], rates["delete_b"])
        - rates["delete_a_b"],
        "pair_interaction": pair_interaction,
        "factorial_interaction": rates["delete_a"]
        + rates["delete_b"]
        - rates["full"]
        - rates["delete_a_b"],
    }


def independent_certificate(
    *,
    component_id: str,
    full_rate: float,
    deletion_rate: float,
    maximum_drop: float,
    full_trials: int,
    deletion_trials: int,
    deletion_arm: str = "C10",
) -> dict[str, Any]:
    """Certify one component without consulting any joint-deletion outcome."""

    if not component_id:
        raise ValueError("component_id is required")
    if full_trials <= 0 or deletion_trials <= 0:
        raise ValueError("trial counts must be positive")
    if not 0.0 <= maximum_drop <= 1.0:
        raise ValueError("maximum_drop must be in [0, 1]")
    for value in (full_rate, deletion_rate):
        if not 0.0 <= value <= 1.0:
            raise ValueError("success rates must be in [0, 1]")
    drop = float(full_rate) - float(deletion_rate)
    removable = drop <= maximum_drop
    return {
        "component_id": component_id,
        "context": "full_harness",
        "evidence_arms": ["C00", str(deletion_arm)],
        "full_rate": float(full_rate),
        "deletion_rate": float(deletion_rate),
        "full_trials": int(full_trials),
        "deletion_trials": int(deletion_trials),
        "drop": drop,
        "maximum_allowed_drop": float(maximum_drop),
        "decision": "REMOVABLE" if removable else "RETAIN",
        "removable": removable,
        "joint_outcome_consulted": False,
    }


def pair_candidate_manifest(
    *,
    inventory: Sequence[Mapping[str, Any]],
    ranked_pairs: Sequence[Mapping[str, Any]],
    independent_certificates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze outcome-blind pair order and mark certificate eligibility."""

    component_ids = {str(row["id"]) for row in inventory}
    removable = {
        str(row["component_id"])
        for row in independent_certificates
        if bool(row["removable"])
    }
    candidates = []
    for rank, pair in enumerate(ranked_pairs, start=1):
        component_a = str(pair["component_a"])
        component_b = str(pair["component_b"])
        if component_a not in component_ids or component_b not in component_ids:
            raise ValueError("ranked pair references a component outside inventory")
        candidates.append(
            {
                **dict(pair),
                "discovery_rank": rank,
                "both_independently_removable": (
                    component_a in removable and component_b in removable
                ),
                "pair_outcome_fields_present": False,
            }
        )
    return {
        "component_count": len(component_ids),
        "ranked_candidate_count": len(candidates),
        "independently_certified_component_count": len(
            {str(row["component_id"]) for row in independent_certificates}
        ),
        "independently_removable_component_count": len(removable),
        "eligible_pair_count": sum(
            bool(row["both_independently_removable"]) for row in candidates
        ),
        "ranking_uses_pair_outcomes": False,
        "candidates": candidates,
    }


def conditional_dependency(
    *,
    pair: Mapping[str, Any],
    rates: Mapping[str, float],
    maximum_single_drop: float,
    minimum_joint_drop: float,
) -> dict[str, Any]:
    """Create a typed dependency edge only when the registered gate passes."""

    quantities = interaction_quantities(
        full=float(rates["C00"]),
        delete_a=float(rates["C10"]),
        delete_b=float(rates["C01"]),
        delete_a_b=float(rates["C11"]),
    )
    safe_singles = max(
        quantities["single_a_drop"], quantities["single_b_drop"]
    ) <= maximum_single_drop
    harmful_joint = quantities["joint_drop"] >= minimum_joint_drop
    detected = safe_singles and harmful_joint
    component_a = str(pair["component_a"])
    component_b = str(pair["component_b"])
    return {
        "pair_id": str(pair["pair_id"]),
        "source": component_a,
        "target": component_b,
        "edge_type": "CONDITIONAL_DEPENDENCY" if detected else "NO_EDGE",
        "detected": detected,
        "predicates": {
            "safe_single_deletions": safe_singles,
            "harmful_joint_deletion": harmful_joint,
            "maximum_single_drop": maximum_single_drop,
            "minimum_joint_drop": minimum_joint_drop,
        },
        "semantics": {
            component_a: f"REMOVABLE | {component_b} present",
            component_b: f"REMOVABLE | {component_a} present",
            "joint": f"NOT jointly removable({component_a},{component_b})",
        },
        "quantities": quantities,
    }


def choose_safe_single_deletion(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    maximum_drop: float,
) -> str:
    """Choose the cheapest safe one-component deletion without a manual tie-break."""

    full_rate = float(arms["C00"]["success_rate"])
    eligible = []
    for arm_id in ("C10", "C01"):
        row = arms[arm_id]
        if full_rate - float(row["success_rate"]) <= maximum_drop:
            eligible.append(
                (
                    float(row["mean_prompt_tokens"]),
                    str(arm_id),
                )
            )
    if not eligible:
        return "C00"
    return min(eligible)[1]


__all__ = [
    "choose_safe_single_deletion",
    "conditional_dependency",
    "independent_certificate",
    "interaction_quantities",
    "pair_candidate_manifest",
]
