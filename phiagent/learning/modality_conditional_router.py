"""Static modality dispatch for answer and interactive routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from phiagent.learning.answer_validity_router import (
    route_nested_fallback_responses,
)


InterfaceType = Literal["answer_selection", "interactive_action"]
ANSWER_LAYER_ORDER = ("base", "incumbent_fallback", "candidate_fallback")
INTERACTIVE_LAYER_ORDER = (
    "proposal_first",
    "wikiskill_incumbent",
    "no_skill",
)
DEPLOYMENT_METHOD = {
    "answer_selection": "nested_router",
    "interactive_action": "proposal_first",
}


def layer_order(interface_type: InterfaceType) -> tuple[str, ...]:
    if interface_type == "answer_selection":
        return ANSWER_LAYER_ORDER
    if interface_type == "interactive_action":
        return INTERACTIVE_LAYER_ORDER
    raise ValueError(f"unsupported interface type: {interface_type!r}")


def deployment_method(interface_type: InterfaceType) -> str:
    try:
        return DEPLOYMENT_METHOD[interface_type]
    except KeyError as error:
        raise ValueError(
            f"unsupported interface type: {interface_type!r}"
        ) from error


def route_answer_selection(
    *,
    base_response: str,
    incumbent_fallback_response: str,
    candidate_fallback_response: str,
    require_unique_label: bool = True,
) -> dict[str, Any]:
    """Preserve the verified LiveMath Nested branch exactly."""

    result = route_nested_fallback_responses(
        base_response,
        incumbent_fallback_response,
        candidate_fallback_response,
        require_unique_label=require_unique_label,
    )
    return {
        **result,
        "interface_type": "answer_selection",
        "layer_order": list(ANSWER_LAYER_ORDER),
        "dispatcher_ground_truth_accessed": False,
    }


def dispatch_metadata(
    interface_type: InterfaceType,
) -> Mapping[str, Any]:
    return {
        "interface_type": interface_type,
        "selected_method": deployment_method(interface_type),
        "layer_order": list(layer_order(interface_type)),
        "decision_time": "before_model_or_environment_outcome",
        "allowed_dispatch_features": ["static benchmark interface type"],
        "prohibited_dispatch_features": [
            "ground truth",
            "reward",
            "success",
            "future observation",
            "model response",
        ],
    }


__all__ = [
    "ANSWER_LAYER_ORDER",
    "INTERACTIVE_LAYER_ORDER",
    "deployment_method",
    "dispatch_metadata",
    "layer_order",
    "route_answer_selection",
]
