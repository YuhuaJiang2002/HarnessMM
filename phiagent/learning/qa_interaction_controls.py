"""Outcome-free controls for a frozen instruction-subset policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

from phiagent.harness.mixture_router import count_matched_random_routes
from phiagent.learning.qa_interaction_compiler import MASKS
from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_rule_module_interventions import (
    FAMILY_TEXT, PADDING_UNIT, ModuleIntervention, _exact_padding, _parse_source,
    module_mask_request,
)


def length_matched_subset_control(
    *, source: str, profile: str, mask: str, family: str,
    count_complete_prompt: Callable[[str], int],
) -> ModuleIntervention:
    """Replace every retained body, keeping headings, fixed input and answer contract."""
    if family not in FAMILY_TEXT or not callable(count_complete_prompt):
        raise ValueError("a declared control family and complete native request counter are required")
    present = module_mask_request(system_prompt=source, profile=profile, mask=mask)
    modules = _parse_source(source, profile)
    retained = [module for bit, module in zip(mask, modules, strict=True) if bit == "1"]
    calls = 0

    def count(prompt: str) -> int:
        nonlocal calls
        calls += 1
        result = count_complete_prompt(prompt)
        if type(result) is not int or result <= 0:
            raise ValueError("complete native request counts must be positive integers")
        return result

    def render(units: int) -> str:
        pieces = [source[:modules[0].start]]
        for index, module in enumerate(retained):
            pieces.extend((
                source[module.start:module.body_start], FAMILY_TEXT[family],
                PADDING_UNIT * units if index == 0 else "",
                source[module.body_end:module.end],
            ))
        pieces.append(source[modules[-1].end:])
        return "".join(pieces)

    target = count(present.system_prompt)
    if retained:
        units, maximum = _exact_padding(target, lambda value: count(render(value)))
    else:
        units, maximum = 0, 0
    result = render(units)
    if count(present.system_prompt) != target or count(result) != target:
        raise ValueError("complete native input matching changed between construction and verification")
    return ModuleIntervention(result, {
        "mask": mask, "family": family,
        "status": "BODY_CONTROL" if retained else "EMPTY_SUBSET_IDENTICAL_REQUEST_CONTROL",
        "source_sha256": text_hash(source),
        "real_subset_sha256": text_hash(present.system_prompt),
        "control_sha256": text_hash(result),
        "controlled_modules": [module.module_id for module in retained],
        "retained_headings_and_separators_unchanged": True,
        "answer_contract_unchanged": True,
        "target_complete_input_tokens": target,
        "actual_complete_input_tokens": target,
        "padding_units": units, "maximum_padding_units": maximum,
        "count_calls": calls,
        "outcomes_used": False,
        "behavioral_neutrality_assumed": False,
        "counter_scope_is_callers_responsibility": True,
    })


def input_matched_random_masks(
    routes: Mapping[str, str], *, strata: Mapping[str, str],
    native_inputs: Mapping[str, Mapping[str, int]], seed: int,
    maximum_relative_difference: float = 0.01,
) -> tuple[dict[str, str], dict]:
    """Use one prespecified permutation, never an outcome-selected or best-of-N control."""
    if (
        not routes or set(routes) != set(native_inputs)
        or any(mask not in MASKS for mask in routes.values())
        or any(set(counts) != set(MASKS) for counts in native_inputs.values())
        or any(type(value) is not int or value <= 0
               for counts in native_inputs.values() for value in counts.values())
        or type(maximum_relative_difference) not in (int, float)
        or not math.isfinite(maximum_relative_difference)
        or not 0 <= maximum_relative_difference <= 0.01
    ):
        raise ValueError("complete mask/count inventories and a <=1 percent input tolerance are required")
    randomized = count_matched_random_routes(routes, strata=strata, seed=seed)
    original_tokens = sum(native_inputs[task][mask] for task, mask in routes.items())
    random_tokens = sum(native_inputs[task][mask] for task, mask in randomized.items())
    difference = abs(random_tokens - original_tokens) / original_tokens
    if difference > maximum_relative_difference:
        raise ValueError(
            f"prespecified random input budget differs by {difference:.6f}; "
            "do not retry seeds, drop tasks, or relabel a mismatched control"
        )
    return randomized, {
        "seed": seed, "tasks": len(routes),
        "original_routes_sha256": digest(dict(routes)),
        "random_routes_sha256": digest(randomized),
        "strata_sha256": digest(dict(strata)),
        "original_input_tokens": original_tokens,
        "random_input_tokens": random_tokens,
        "relative_input_difference": difference,
        "changed_assignments": sum(routes[task] != randomized[task] for task in routes),
        "mask_counts_preserved_within_each_stratum": True,
        "output_or_total_tokens_assumed_equal": False,
        "outcomes_used": False,
        "permutations_attempted": 1,
    }
