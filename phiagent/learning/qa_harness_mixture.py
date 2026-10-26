"""Provisional SearchQA adapter for sparse skills and source-bound harness routing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import re

from phiagent.harness.mixture_router import (
    HarnessCandidate, HarnessProfile, SkillSpec, choose_harness, select_skills,
)
from phiagent.learning.compact_qa_contracts import DELETE_COMPACT_RULES
from phiagent.learning.public_evidence_retention import text_sha256
from phiagent.learning.qa_context_evidence import ContextSelection, select_context
from phiagent.learning.sparse_retrieval import WORD_PATTERN


_BASE_CONTRACT = (
    "Answer the question from the context. Return only the concise answer "
    "inside <answer>...</answer>.\n"
)
_INCUMBENT = "incumbent"
_TAG_PATTERNS = {
    "person": r"\b(?:who|whose|person|man|woman|name|surname)\b",
    "relation": r"\b(?:wife|husband|father|mother|son|daughter|brother|sister|parent)\b",
    "work_role": r"\b(?:film|movie|novel|book|author|song|poem|lyric|actor|character)\b",
    "qualifier": (
        r"\b(?:when|year|century|before|after|first|last|oldest|largest|smallest|"
        r"letters|count|number|many|much)\b|\d"
    ),
    "form": r"\b(?:first name|last name|surname|acronym|abbreviation|initials)\b",
    "etymology": r"\b(?:etymology|derived|means|meaning|named after)\b",
    "fill": r"_{2,}|\b(?:blank|missing word|complete the)\b",
    "quote": r"""["\u201c\u201d]|\b(?:quote|quotation|lyric)\b""",
}
_RULE_BINDINGS = (
    ("answer_type", 2, 1, ("target",), 0),
    ("person_name", 0, 3, ("person",), 1),
    ("relation_direction", 4, 1, ("relation",), 4),
    ("work_role", 2, 3, ("work_role",), 3),
    ("qualifier", 2, 2, ("qualifier",), 3),
    ("requested_form", 4, 5, ("form",), 6),
    ("etymology", 0, 1, ("etymology",), 5),
    ("fill_slot", 2, 4, ("fill",), 5),
    ("quote_target", 4, 7, ("quote",), 2),
)
QA_SKILLS = tuple(
    SkillSpec(identity, DELETE_COMPACT_RULES[section][1][clause], tags, priority)
    for identity, section, clause, tags, priority in _RULE_BINDINGS
)


@dataclass(frozen=True, slots=True)
class QARequestPlan:
    system_prompt: str
    context: str
    max_output_tokens: int
    receipt: dict[str, object]


def mixture_arm_prompt() -> str:
    return _BASE_CONTRACT


def qa_input_tags(question: str) -> frozenset[str]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    folded = question.casefold()
    return frozenset({"target"} | {
        tag for tag, pattern in _TAG_PATTERNS.items() if re.search(pattern, folded)
    })


def _hash_json(value: object) -> str:
    return text_sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _source_words(text: str) -> set[str]:
    return set(WORD_PATTERN.findall(re.sub(r"\[(?:DOC|SEP)\]", " ", text).casefold()))


def build_mixture_request(
    *,
    question: str,
    context: str,
    reference_system: str,
    reference_max_output_tokens: int,
    settings: dict,
    arm: str,
    count_tokens: Callable[[str], int],
    count_prompt: Callable[[str, str, str], int],
    seed: int,
    context_cache: dict[str, ContextSelection] | None = None,
) -> QARequestPlan:
    """Build and attest one request without receiving answers, scores or actor outputs."""
    tags = qa_input_tags(question)
    if any(not isinstance(text, str) or not text.strip() for text in (context, reference_system)):
        raise ValueError("original source context and incumbent prompt must be nonempty")
    if not isinstance(settings, dict) or arm not in settings["arms"]:
        raise ValueError("the candidate arm must be explicitly registered")
    enabled = settings["enabled"]
    if type(enabled) is not bool:
        raise ValueError("mixture routing requires an explicit boolean enable flag")
    policy = settings["arms"][arm]
    if (
        not isinstance(policy, dict)
        or set(policy) - {"harness_mode", "skill_mode", "fixed_harness_id"}
        or not {"harness_mode", "skill_mode"} <= set(policy)
    ):
        raise ValueError("an arm must declare only its harness and skill interventions")
    profiles = tuple(HarnessProfile(**value) for value in settings["profiles"])
    if not profiles or any(
        profile.harness_id == _INCUMBENT or profile.context_token_budget is None
        for profile in profiles
    ):
        raise ValueError("nonempty candidate profiles cannot redefine the incumbent")
    reference = HarnessCandidate(
        HarnessProfile(_INCUMBENT, None, reference_max_output_tokens, 0),
        count_prompt(reference_system, question, context), 1.0, True,
    )
    input_key = _hash_json([question, context])
    skills = select_skills(
        QA_SKILLS, active_tags=tags, top_k=settings["skill_top_k"],
        mode=policy["skill_mode"], seed=seed, input_key=input_key,
    )
    system = _BASE_CONTRACT + "".join(f"- {skill.text}\n" for skill in skills)
    candidates = [reference]
    selections = {}
    coverages = {}
    if enabled:
        original_words = _source_words(context)
        for profile in profiles:
            cache_key = _hash_json([question, context, profile.context_token_budget])
            selection = context_cache.get(cache_key) if context_cache is not None else None
            if selection is None:
                selection = select_context(
                    question, context, method="bm25", count_tokens=count_tokens,
                    token_budget=profile.context_token_budget,
                    maximum_unit_tokens=min(48, profile.context_token_budget),
                )
                if context_cache is not None:
                    context_cache[cache_key] = selection
            elif (
                selection.receipt["source_context_sha256"] != text_sha256(context)
                or selection.receipt["question_sha256"] != text_sha256(question)
                or selection.receipt["token_budget"] != profile.context_token_budget
                or count_tokens(selection.context) != selection.receipt["rendered_context_tokens"]
                or count_tokens(context) != selection.receipt["source_context_tokens"]
            ):
                raise ValueError("cached source selection differs from the current input/tokenizer")
            selections[profile.harness_id] = selection
            available = set(selection.receipt["lexical_query_terms"]) & original_words
            retained = available & _source_words(selection.context)
            coverage = len(retained) / len(available) if available else 0.0
            coverages[profile.harness_id] = {
                "available_query_anchors": sorted(available),
                "retained_query_anchors": sorted(retained),
                "all_query_weighted_coverage": selection.receipt["lexical_query_coverage"],
            }
            candidates.append(HarnessCandidate(
                profile, count_prompt(system, question, selection.context), coverage,
                bool(available),
            ))
    else:
        # Validation must not make disabled routing transform or tokenize the source.
        candidates.extend(HarnessCandidate(profile, reference.prompt_tokens, 0.0, False)
                          for profile in profiles)
    decision = choose_harness(
        candidates, reference_id=_INCUMBENT, enabled=enabled,
        mode=policy["harness_mode"], fixed_harness_id=policy.get("fixed_harness_id"),
        token_ratio_target=settings["token_ratio_target"], seed=seed, input_key=input_key,
    )
    selection = None if decision.used_fallback else selections[decision.selected_harness]
    selected_context = context if selection is None else selection.context
    selected_system = reference_system if decision.used_fallback else system
    selected_skills = () if decision.used_fallback else skills
    actual_count = count_prompt(selected_system, question, selected_context)
    if actual_count != decision.expected_prompt_tokens:
        raise RuntimeError("the routed full chat-template token count changed")
    receipt: dict[str, object] = {
        "schema_version": "phiagent-qa-harness-mixture-request/v1",
        **asdict(decision),
        "arm": arm,
        "enabled": enabled,
        "harness_mode": policy["harness_mode"],
        "skill_mode": policy["skill_mode"],
        "active_tags": sorted(tags),
        "selected_skills": [skill.skill_id for skill in selected_skills],
        "selected_skill_sha256": {
            skill.skill_id: text_sha256(skill.text) for skill in selected_skills
        },
        "library_sha256": _hash_json([asdict(skill) for skill in QA_SKILLS]),
        "policy_sha256": _hash_json(settings),
        "seed": seed,
        "input_sha256": input_key,
        "question_sha256": text_sha256(question),
        "source_context_sha256": text_sha256(context),
        "reference_system_sha256": text_sha256(reference_system),
        "reference_prompt_tokens": reference.prompt_tokens,
        "reference_max_output_tokens": reference_max_output_tokens,
        "system_sha256": text_sha256(selected_system),
        "rendered_context_sha256": text_sha256(selected_context),
        "request_content_sha256": _hash_json([
            selected_system, question, selected_context, decision.max_output_tokens,
        ]),
        "token_ratio_target": settings["token_ratio_target"],
        "upper_bound_denominator": "incumbent full chat-template prompt tokens only",
        "context_evidence": None if selection is None else selection.receipt,
        "identity_context_transform": selected_context == context,
        "identity_incumbent_request": decision.used_fallback,
        "router_model_calls": 0,
        "router_provider_tokens": 0,
        "answers_or_actor_responses_used": False,
        "lexical_gate_is_performance_certificate": False,
        "coverage_definition": "fraction of source-present query lexemes retained, excluding DOC/SEP",
        "absent_query_words_are_not_claimed_satisfied": True,
        "performance_nonregression_established": False,
        "candidates": [
            {
                "harness_id": candidate.profile.harness_id,
                "prompt_tokens": candidate.prompt_tokens,
                "max_output_tokens": candidate.profile.max_output_tokens,
                "token_upper_bound": candidate.token_upper_bound,
                "lexical_coverage": candidate.evidence_coverage,
                "minimum_coverage": candidate.profile.minimum_coverage,
                **coverages.get(candidate.profile.harness_id, {}),
            }
            for candidate in candidates
        ],
    }
    receipt["canonical_sha256"] = _hash_json(receipt)
    return QARequestPlan(selected_system, selected_context, decision.max_output_tokens, receipt)
