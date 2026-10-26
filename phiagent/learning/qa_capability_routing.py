"""Source-preserving capability skeletons and sparse, question-scoped QA skills."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json

from phiagent.harness.mixture_router import SkillSpec, select_skills
from phiagent.learning.compact_qa_contracts import DELETE_COMPACT_RULES, cost_source_contracts
from phiagent.learning.public_evidence_retention import text_sha256
from phiagent.learning.qa_harness_mixture import QARequestPlan, QA_SKILLS, qa_input_tags


CORE_BINDINGS = (
    ("core_requested_type", 2, 1),
    ("core_joint_evidence", 1, 1),
    ("core_minimal_unambiguous", 0, 0),
    ("core_source_wording", 0, 2),
)
CORE_SKILLS = tuple(
    SkillSpec(name, DELETE_COMPACT_RULES[section][1][clause], (), required=True)
    for name, section, clause in CORE_BINDINGS
)
OPTIONAL_SKILLS = tuple(skill for skill in QA_SKILLS if skill.skill_id != "answer_type")
CAPABILITY_MODES = frozenset({
    "capability_routed", "core_only", "routed_without_core", "answer_only",
    "all_capabilities", "count_matched_random", "legacy_nine_all",
})


def _hash(value: object) -> str:
    return text_sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def capability_request(
    *,
    question: str,
    context: str,
    no_skill_system: str,
    delete_system: str,
    mode: str,
    seed: int,
    max_output_tokens: int,
    optional_top_k: int,
    count_prompt: Callable[[str, str, str], int],
    random_optional_count: int | None = None,
) -> QARequestPlan:
    """Never remove evidence or infer sufficiency from source-word coverage."""
    if mode not in CAPABILITY_MODES:
        raise ValueError("an explicitly registered capability mode is required")
    if any(not isinstance(value, str) or not value.strip()
           for value in (question, context, no_skill_system, delete_system)):
        raise ValueError("all original input and prompt strings must be nonempty")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("the unchanged output cap must be positive")
    if type(optional_top_k) is not int or not 0 <= optional_top_k <= len(OPTIONAL_SKILLS):
        raise ValueError("optional skill budget exceeds the registered library")
    if type(seed) is not int:
        raise ValueError("an integer seed is required")
    if mode == "count_matched_random":
        if type(random_optional_count) is not int or not 0 <= random_optional_count <= optional_top_k:
            raise ValueError("random controls require the primary's exact optional skill count")
    elif random_optional_count is not None:
        raise ValueError("only random controls accept a frozen optional skill count")
    tags = qa_input_tags(question)
    include_core = mode in {"capability_routed", "core_only", "all_capabilities", "count_matched_random"}
    selected = list(CORE_SKILLS) if include_core else []
    if mode == "legacy_nine_all":
        selected = list(QA_SKILLS)
    elif mode == "all_capabilities":
        selected.extend(select_skills(
            OPTIONAL_SKILLS, active_tags=tags, top_k=len(OPTIONAL_SKILLS), mode="all",
        ))
    else:
        count = random_optional_count if mode == "count_matched_random" else optional_top_k
        if count and mode not in {"core_only", "answer_only"}:
            selected.extend(select_skills(
                OPTIONAL_SKILLS, active_tags=tags, top_k=count,
                mode="random" if mode == "count_matched_random" else "routed",
                seed=seed, input_key=_hash([question, context]),
            ))
    base = cost_source_contracts(no_skill_system, delete_system)["no_skill_answer_only"]
    system = base
    if selected:
        system += "\n## Applicable execution rules\n" + "".join(f"- {skill.text}\n" for skill in selected)
    prompt_tokens = count_prompt(system, question, context)
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("the exact full chat-template counter must return a positive integer")
    selected_ids = [skill.skill_id for skill in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("a capability clause was injected more than once")
    receipt = {
        "schema_version": "phiagent-source-preserving-qa-capabilities/v1",
        "mode": mode, "seed": seed, "optional_top_k": optional_top_k,
        "active_tags": sorted(tags),
        "selected_harness": "full_original_evidence",
        "selected_skills": selected_ids,
        "core_included": include_core,
        "selected_optional_count": sum(skill in OPTIONAL_SKILLS for skill in selected),
        "random_optional_count": random_optional_count,
        "library_sha256": _hash([asdict(skill) for skill in (*CORE_SKILLS, *OPTIONAL_SKILLS)]),
        "source_context_sha256": text_sha256(context),
        "rendered_context_sha256": text_sha256(context),
        "question_sha256": text_sha256(question),
        "system_sha256": text_sha256(system),
        "expected_prompt_tokens": prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "identity_context_transform": True,
        "identity_incumbent_request": False,
        "router_model_calls": 0, "router_provider_tokens": 0,
        "answers_or_actor_responses_used": False,
        "performance_nonregression_established": False,
        "request_content_sha256": _hash([system, question, context, max_output_tokens]),
        "selected_skill_sha256": {skill.skill_id: text_sha256(skill.text) for skill in selected},
    }
    receipt["canonical_sha256"] = _hash(receipt)
    return QARequestPlan(system, context, max_output_tokens, receipt)
