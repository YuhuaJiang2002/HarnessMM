"""Source-bound Original CHISEL module omissions and exact-length body controls.

Only the supplied system string is edited. The caller owns the fixed remaining
messages, native chat template, output cap, tokenizer provenance, and outcome-free
counting callback. A receipt certifies mechanical edits, not semantics or quality.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType


MODULE_IDS = ("normalization", "evidence", "answer_type", "snippet_format", "traps")
MODULE_HEADINGS = MappingProxyType({
    "normalization": "Concise Answer Normalization",
    "evidence": "Context-Grounded Evidence Matching",
    "answer_type": "Clue Interpretation and Answer Type",
    "snippet_format": "Trivia / Jeopardy Snippet Formats",
    "traps": "Common Clue Traps",
})
SOURCE_SHA256 = MappingProxyType({
    "delete": "19fdbb91f0051028d213dd52845dfe40c87be1fda60cb58ea89f270a56333e88",
    "modify": "08e74cd6f8d8bfac8881dfde151d6884eec3f9a2db7662eaec8b517ee3af90ee",
})
FAMILY_TEXT = MappingProxyType({
    "generic": "- Read the provided material carefully and check the answer.",
    "unrelated": "- A catalog lists plain tiles in orderly rows.",
})
PADDING_UNIT = " x"
MAX_PADDING_UNITS = 8192

_SOURCE_BULLET_COUNTS = MappingProxyType({
    "delete": (8, 6, 9, 2, 8),
    "modify": (13, 6, 9, 2, 8),
})
_SKILL_ENVELOPE = "## Skill\n# Question Answering Skill\n\n"
_ANSWER_CONTRACT = "## Answer Contract\n"
_SECTION_HEADING = re.compile(r"^## ([^\n]+)\n", re.MULTILINE)
_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ModuleIntervention:
    system_prompt: str
    receipt: dict[str, object]


@dataclass(frozen=True)
class _ModuleSpan:
    module_id: str
    start: int
    body_start: int
    body_end: int
    end: int
    bullets: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique_marker(text: str, marker: str) -> re.Match[str]:
    matches = list(re.finditer("^" + re.escape(marker), text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"source boundary requires exactly one {marker.splitlines()[0]!r}")
    return matches[0]


def _parse_source(system_prompt: str, profile: str) -> tuple[_ModuleSpan, ...]:
    if not isinstance(profile, str) or profile not in _SOURCE_BULLET_COUNTS:
        raise ValueError("profile must be exactly 'delete' or 'modify'")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be the original UTF-8 source decoded as str")
    try:
        actual_hash = _sha256(system_prompt)
    except UnicodeEncodeError as error:
        raise ValueError("system_prompt is not a valid UTF-8 source string") from error
    if actual_hash != SOURCE_SHA256[profile]:
        raise ValueError(
            f"original source SHA256 mismatch for {profile}: "
            f"expected {SOURCE_SHA256[profile]}, got {actual_hash}; no normalization is allowed"
        )

    skill = _unique_marker(system_prompt, _SKILL_ENVELOPE)
    contract = _unique_marker(system_prompt, _ANSWER_CONTRACT)
    all_headings = list(_SECTION_HEADING.finditer(system_prompt))
    headings = [
        heading for heading in all_headings if skill.end() <= heading.start() < contract.start()
    ]
    expected_headings = tuple(MODULE_HEADINGS[module_id] for module_id in MODULE_IDS)
    if tuple(heading.group(1) for heading in headings) != expected_headings:
        raise ValueError("source module boundaries/headings must follow the fixed MODULE_IDS order")
    if headings[0].start() != skill.end():
        raise ValueError("source contains unassigned text before the first module boundary")
    for name in expected_headings:
        if sum(heading.group(1) == name for heading in all_headings) != 1:
            raise ValueError("source module headings must occur exactly once, inside the skill")

    modules = []
    for index, (module_id, heading, expected_bullets) in enumerate(
        zip(MODULE_IDS, headings, _SOURCE_BULLET_COUNTS[profile])
    ):
        end = headings[index + 1].start() if index + 1 < len(headings) else contract.start()
        body_start = heading.end()
        if not system_prompt[body_start:end].endswith(_SEPARATOR):
            raise ValueError(f"source module {module_id} lacks its exact two-newline separator")
        body_end = end - len(_SEPARATOR)
        lines = system_prompt[body_start:body_end].split("\n")
        if (
            not lines[0].startswith("- ")
            or not lines[-1].startswith("- ")
            or any(line and (not line.startswith("- ") or len(line) <= 2) for line in lines)
        ):
            raise ValueError(f"source module {module_id} has an unsupported rule body boundary")
        bullets = sum(line.startswith("- ") for line in lines)
        if bullets != expected_bullets:
            raise ValueError(
                f"source module {module_id} requires {expected_bullets} bullets, got {bullets}"
            )
        modules.append(_ModuleSpan(module_id, heading.start(), body_start, body_end, end, bullets))
    return tuple(modules)


def _receipt(
    source: str, result: str, profile: str, modules: tuple[_ModuleSpan, ...],
) -> dict[str, object]:
    return {
        "schema": "original_chisel_module_intervention_v1",
        "profile": profile,
        "module_order": list(MODULE_IDS),
        "source_sha256": _sha256(source),
        "result_sha256": _sha256(result),
        "hash_encoding": "raw_utf8_without_normalization",
        "source_characters": len(source),
        "result_characters": len(result),
        "source_utf8_bytes": len(source.encode("utf-8")),
        "result_utf8_bytes": len(result.encode("utf-8")),
        "span_frame": "original_system_prompt_unicode_codepoints",
        "span_bounds": "zero_based_half_open",
        "source_module_spans": [
            {
                "module_id": module.module_id,
                "heading": MODULE_HEADINGS[module.module_id],
                "whole_module_span": [module.start, module.end],
                "heading_span": [module.start, module.body_start],
                "rule_body_span": [module.body_start, module.body_end],
                "separator_span": [module.body_end, module.end],
                "source_bullets": module.bullets,
                "whole_module_sha256": _sha256(source[module.start:module.end]),
                "rule_body_sha256": _sha256(source[module.body_start:module.body_end]),
            }
            for module in modules
        ],
        "source_bullet_counts": {module.module_id: module.bullets for module in modules},
        "source_total_bullets": sum(module.bullets for module in modules),
        "all_other_source_bytes_preserved": True,
        "head_preserved": True,
        "skill_envelope_preserved": True,
        "answer_contract_preserved": True,
        "edited_request_field": "system_prompt_only",
        "other_messages_context_template_and_output_cap": "unchanged_by_adapter; caller_must_fix",
        "outcomes_used": False,
        "callback_must_not_use_outcomes": True,
        "semantic_certificate": False,
        "quality_certificate": False,
        "behavioral_neutrality_assumed": False,
    }


def module_mask_request(
    *, system_prompt: str, profile: str, mask: str,
) -> ModuleIntervention:
    """Omit whole source blocks; bit zero is normalization, never an integer bit.

    Blocks run from the heading start to the next heading/Answer Contract start,
    including their final two-newline separator. The full mask is byte-identical
    to the bound source; the empty mask retains the exact outer skill envelope.
    """
    if not isinstance(mask, str) or len(mask) != len(MODULE_IDS) or set(mask) - {"0", "1"}:
        raise ValueError("mask must be a five-character binary string in MODULE_IDS order")
    modules = _parse_source(system_prompt, profile)
    result = (
        system_prompt[:modules[0].start]
        + "".join(
            system_prompt[module.start:module.end]
            for bit, module in zip(mask, modules) if bit == "1"
        )
        + system_prompt[modules[-1].end:]
    )
    omitted = [module for bit, module in zip(mask, modules) if bit == "0"]
    receipt = _receipt(system_prompt, result, profile, modules)
    receipt.update({
        "intervention": "whole_module_omission",
        "mask": mask,
        "retained_module_ids": [module.module_id for bit, module in zip(mask, modules) if bit == "1"],
        "omitted_module_ids": [module.module_id for module in omitted],
        "changed_original_spans": [[module.start, module.end] for module in omitted],
        "omitted_headings_removed": True,
        "count_source": "not_requested_for_whole_module_omission",
        "count_scope": None,
        "target_tokens": None,
        "actual_tokens": None,
        "exact_length_match": None,
        "padding": None,
    })
    return ModuleIntervention(result, receipt)


def _exact_padding(
    target: int, count_units: Callable[[int], int],
) -> tuple[int, int]:
    maximum = min(MAX_PADDING_UNITS, target)
    observed: dict[int, int] = {}

    def count(units: int) -> int:
        if units not in observed:
            observed[units] = count_units(units)
        return observed[units]

    base = count(0)
    if base == target:
        return 0, maximum
    guess = min(maximum, max(0, target - base))
    if count(guess) == target:
        return guess, maximum

    low, high = 1, maximum
    while low <= high:
        middle = (low + high) // 2
        measured = count(middle)
        if measured == target:
            return middle, maximum
        if measured < target:
            low = middle + 1
        else:
            high = middle - 1

    # Tokenization need not be monotone. Exhaust the bounded, unvisited candidates
    # before declaring failure; the deficit/binary probes keep the usual path cheap.
    for units in range(1, maximum + 1):
        if count(units) == target:
            return units, maximum
    raise ValueError(
        f"exact complete-prompt length match unavailable within {maximum} padding units; "
        "no approximation or raw-system-token fallback"
    )


def length_matched_module_request(
    *,
    system_prompt: str,
    profile: str,
    module_id: str,
    family: str,
    count_complete_prompt: Callable[[str], int],
) -> ModuleIntervention:
    """Replace one rule body, retaining its heading and original separator.

    The callback must count the complete native chat template, including every
    fixed message and the generation prefix, using the served tokenizer/options.
    Its scope/provenance cannot be inferred from an integer and remain the caller's
    responsibility. Counts must be positive Python ints and repeat consistently.
    No source token count, approximation, outcomes, or alternative text is used.
    """
    if not isinstance(module_id, str) or module_id not in MODULE_IDS:
        raise ValueError("module_id must be one of the fixed MODULE_IDS")
    if not isinstance(family, str) or family not in FAMILY_TEXT:
        raise ValueError("family must be exactly 'generic' or 'unrelated'")
    if not callable(count_complete_prompt):
        raise ValueError("count_complete_prompt must be a complete-native-prompt counting callback")
    modules = _parse_source(system_prompt, profile)
    module = modules[MODULE_IDS.index(module_id)]
    core = FAMILY_TEXT[family]
    count_calls = 0

    def count(prompt: str, stage: str) -> int:
        nonlocal count_calls
        count_calls += 1
        try:
            value = count_complete_prompt(prompt)
        except Exception as error:
            raise ValueError(f"complete native chat-template count unavailable ({stage})") from error
        if type(value) is not int or value <= 0:
            raise ValueError(f"complete native chat-template count must be a positive integer ({stage})")
        return value

    def render(units: int) -> str:
        return (
            system_prompt[:module.body_start] + core + PADDING_UNIT * units
            + system_prompt[module.body_end:]
        )

    target = count(system_prompt, "original")
    units, maximum = _exact_padding(
        target, lambda amount: count(render(amount), f"replacement, {amount} padding units"),
    )
    result = render(units)
    verified_target = count(system_prompt, "original verification")
    actual = count(result, "replacement verification")
    if verified_target != target or actual != target:
        raise ValueError("inconsistent complete native chat-template counts; exact match rejected")

    receipt = _receipt(system_prompt, result, profile, modules)
    receipt.update({
        "intervention": "body_replacement_retaining_heading",
        "module_id": module_id,
        "family": family,
        "changed_original_spans": [[module.body_start, module.body_end]],
        "heading_retained": True,
        "original_separator_retained": True,
        "other_module_ids_preserved": [name for name in MODULE_IDS if name != module_id],
        "family_core_text": core,
        "family_core_sha256": _sha256(core),
        "replacement_body_sha256": _sha256(core + PADDING_UNIT * units),
        "count_source": "caller_supplied_count_complete_prompt",
        "count_scope": "complete_native_chat_template",
        "count_scope_is_caller_responsibility": True,
        "raw_system_token_count_used": False,
        "target_tokens": target,
        "actual_tokens": actual,
        "exact_length_match": True,
        "padding": {
            "unit_text": PADDING_UNIT,
            "units": units,
            "characters": len(PADDING_UNIT) * units,
            "sha256": _sha256(PADDING_UNIT * units),
            "maximum_units": maximum,
            "absolute_maximum_units": MAX_PADDING_UNITS,
            "search": "zero_then_deficit_then_binary_then_unvisited_exact_scan_v1",
            "complete_prompt_count_calls": count_calls,
        },
    })
    return ModuleIntervention(result, receipt)
