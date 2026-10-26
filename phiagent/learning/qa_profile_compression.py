"""Source-bound compression of QA normalization and shared instruction blocks."""

from __future__ import annotations

import hashlib


SOURCE_HASHES = {
    "delete": "19fdbb91f0051028d213dd52845dfe40c87be1fda60cb58ea89f270a56333e88",
    "modify": "08e74cd6f8d8bfac8881dfde151d6884eec3f9a2db7662eaec8b517ee3af90ee",
}
SECTIONS = (
    "Concise Answer Normalization", "Context-Grounded Evidence Matching",
    "Clue Interpretation and Answer Type", "Trivia / Jeopardy Snippet Formats", "Common Clue Traps",
)
COUNTS = {"delete": (8, 6, 9, 2, 8), "modify": (13, 6, 9, 2, 8)}
VARIANTS = ("normalization", "common", "both", "layout")
ANSWER_CONTRACT = (
    "Immediately output exactly one concise answer inside <answer>...</answer> and stop. "
    "Do not print reasoning, evidence, alternatives, or a preamble.\n"
)
PREFIX = "Answer the QUESTION from the CONTEXT.\n\n"
NORMALIZATION = (
    ("Return the shortest unambiguous name/headword; omit generic labels, legal suffixes and "
     "formal expansions unless requested or needed for identity, including etymology answers.", (1, 2)),
    ("Copy the strongest matching title/snippet/answer-field spelling, case, punctuation and order. "
     "Use supported conventional full personal names unless the clue/source expects a short name.", (3, 4)),
    ("Use expected inflection: singular dictionary headwords/shared given names despite plural "
     "clue grammar, except inherently plural terms or explicit plural source answers.", (5, 6)),
    ("For replacements return the broad headword unless narrowed by clue/answer field. "
     "Fill-ins need standalone noun phrases, not source pronouns/possessives; retain necessary articles.", (7, 8)),
)
MODIFY_DELTA = (
    "Strip descriptors already stated in the clue unless the full official name is requested. "
    "Keep conventional geographic designators needed by the proper name/type; prefer distinctive "
    "company names. Copy supported ASCII punctuation, nickname/title styling and abbreviations "
    "without expanding or normalizing them."
)
MODIFY_SHARED_INDICES = (1, 3, 6, 9, 10, 11, 12, 13)
MODIFY_EXTRA_INDICES = (2, 4, 5, 7, 8)
COMMON = (
    (
        "Match distinctive clue facts jointly: names, dates, roles, relations, titles and quotes. "
        "Prefer explicit relations/answer fields over generic overlaps. Titles and corroborating "
        "snippets are evidence, not necessarily the target. Use the directly named canonical term, "
        "not a derivative or near-synonym.",
        ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 9)),
    ),
    (
        "Infer target type and relation direction from verbs/pronouns; enforce every qualifier "
        "(date, rank, length, version, brand, role). Distinguish work, creator, cast, character, "
        "quotation source and setting. Enforce crossword length, dual senses and required inflection.",
        ((2, 1), (2, 2), (2, 3), (2, 4), (2, 7)),
    ),
    (
        "Substitute candidates into blanks. Examples/lists imply their shared parent category "
        "or synonym, not a listed example. Inverse relations request the target, not named evidence.",
        ((2, 5), (2, 6), (4, 1), (4, 2), (4, 3)),
    ),
    (
        "Recognize CATEGORY|clue|answer, clue.ANSWER and right: formats; extract matching answer "
        "fields, not category/clue. Quotes/lyrics/wordplay identify exact missing continuations "
        "or associated targets; return an anchor only if requested. Honor first-name, acronym, "
        "abbreviation and lyric-word forms, with supported spelling/punctuation.",
        ((3, 1), (3, 2), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)),
    ),
    ("When an image/link is unavailable, use the textual clue and context.", ((2, 8),)),
)
PRIORITY_RULE = (
    "Resolve the requested entity/type and all clue constraints before normalizing. "
    "Exact supported source forms override brevity and default inflection."
)
PRIORITY_NORMALIZATION = (
    (
        "Return a concise, fully identified target, not the shortest possible string. Omit "
        "descriptors/legal suffixes/formal expansions only when not requested or needed for identity; "
        "etymology answers use the distinctive name/word.",
        (1, 2),
    ),
    (
        "Copy the strongest supported answer form exactly: spelling, capitalization, punctuation "
        "and word order. Keep conventional full personal names unless the clue/source explicitly "
        "expects a short form.",
        (3, 4),
    ),
    (
        "Explicit source answers and required grammatical forms override default singularization. "
        "Otherwise use singular dictionary headwords/shared given names despite plural clue grammar, "
        "except inherently plural terms.",
        (5, 6),
    ),
    NORMALIZATION[3],
)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_sections(profile: str, text: str) -> tuple[str, dict[str, list[str]], str]:
    if profile not in SOURCE_HASHES or text_sha(text) != SOURCE_HASHES[profile]:
        raise ValueError("Compression requires the exact archived DELETE/MODIFY source bytes")
    if text.count("## Answer Contract\n") != 1:
        raise ValueError("The unchanged answer contract must occur exactly once")
    body, answer = text.split("## Answer Contract\n")
    if answer != ANSWER_CONTRACT:
        raise ValueError("The output contract cannot be compressed or rewritten")
    prefix = body.split("## " + SECTIONS[0], 1)[0]
    sections: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        if line.startswith("## ") and line != "## Skill":
            current = line[3:]
            sections[current] = []
        elif line.startswith("- "):
            if current is None:
                raise ValueError("A source rule lacks its section")
            sections[current].append(line[2:])
    if tuple(sections) != SECTIONS or tuple(map(len, sections.values())) != COUNTS[profile]:
        raise ValueError("The exact archived clause/section inventory changed")
    return prefix, sections, answer


def clause_mapping(
    profile: str, sections: dict[str, list[str]], *, normalization=NORMALIZATION,
) -> list[dict]:
    normalized = normalization
    if profile == "modify":
        normalized = tuple(
            (text, tuple(MODIFY_SHARED_INDICES[index - 1] for index in indices))
            for text, indices in normalization
        ) + ((MODIFY_DELTA, MODIFY_EXTRA_INDICES),)
    groups = [
        (text, tuple((0, index) for index in indices)) for text, indices in normalized
    ] + list(COMMON)
    seen = set()
    mapping = []
    for number, (target, references) in enumerate(groups):
        for section_index, index in references:
            key = section_index, index
            if key in seen:
                raise ValueError("A source clause is assigned twice")
            original = sections[SECTIONS[section_index]][index - 1]
            seen.add(key)
            mapping.append({
                "section": SECTIONS[section_index], "clause": index,
                "source_text": original, "source_sha256": text_sha(original),
                "compressed_group": number, "compressed_text": target,
                "compressed_sha256": text_sha(target),
            })
    expected = {(s, i + 1) for s, name in enumerate(SECTIONS) for i in range(len(sections[name]))}
    if seen != expected:
        raise ValueError("Every archived condition and exception needs a source mapping")
    return mapping


def compile_profiles(originals: dict[str, str]) -> tuple[dict[str, dict[str, str]], dict]:
    if set(originals) != set(SOURCE_HASHES):
        raise ValueError("Both exact source profiles are required")
    rendered: dict[str, dict[str, str]] = {variant: {} for variant in VARIANTS}
    maps = {}
    parsed = {name: source_sections(name, text) for name, text in originals.items()}
    for section in SECTIONS[1:]:
        if parsed["delete"][1][section] != parsed["modify"][1][section]:
            raise ValueError("Shared rules differ between archived profiles")
    for profile, (_prefix, sections, _answer) in parsed.items():
        common_start = originals[profile].index("## " + SECTIONS[1])
        normal_start = originals[profile].index("## " + SECTIONS[0])
        answer_start = originals[profile].index("## Answer Contract\n")
        normal_original = originals[profile][normal_start:common_start]
        common_original = originals[profile][common_start:answer_start]
        normal_compact = "\n".join(text for text, _indices in NORMALIZATION)
        if profile == "modify":
            normal_compact += "\n" + MODIFY_DELTA
        common_compact = "\n".join(text for text, _refs in COMMON)
        ending = "## Answer Contract\n" + ANSWER_CONTRACT
        rendered["normalization"][profile] = (
            _prefix + normal_compact + "\n\n" + common_original + ending
        )
        rendered["common"][profile] = PREFIX + normal_original + common_compact + "\n\n" + ending
        rendered["both"][profile] = PREFIX + normal_compact + "\n\n" + common_compact + "\n\n" + ending
        rendered["layout"][profile] = (
            PREFIX + "\n".join(rule for rules in sections.values() for rule in rules) + "\n\n" + ending
        )
        maps[profile] = clause_mapping(profile, sections)
    mapping = {
        "schema_version": "phiagent-qa-profile-compression-map/v1",
        "source_sha256": SOURCE_HASHES, "clauses": maps,
        "candidate_sha256": {
            variant: {name: text_sha(text) for name, text in prompts.items()}
            for variant, prompts in rendered.items()
        },
        "answer_contract_sha256": text_sha(ANSWER_CONTRACT),
        "every_source_clause_mapped": True,
        "source_and_shared_clause_coverage_is_not_semantic_or_quality_equivalence": True,
        "new_router_or_threshold": False, "task_answers_used": False,
        "common_block_compression_includes_redundant_role_and_section_preambles": True,
    }
    return rendered, mapping


def answer_only_prompt() -> str:
    return PREFIX + "## Answer Contract\n" + ANSWER_CONTRACT


def compile_priority_profiles(originals: dict[str, str]) -> tuple[dict[str, dict[str, str]], dict]:
    _, legacy = compile_profiles(originals)
    output = {"priority": {}, "without_priority": {}}
    clauses = {}
    for profile, text in originals.items():
        _, sections, _ = source_sections(profile, text)
        normal = "\n".join(rule for rule, _refs in PRIORITY_NORMALIZATION)
        if profile == "modify":
            normal += "\n" + MODIFY_DELTA
        body = (
            "## Evidence and target\n" + "\n".join(rule for rule, _refs in COMMON)
            + "\n\n## Answer form\n" + normal
            + "\n\n## Answer Contract\n" + ANSWER_CONTRACT
        )
        output["without_priority"][profile] = PREFIX + body
        output["priority"][profile] = PREFIX + PRIORITY_RULE + "\n\n" + body
        clauses[profile] = clause_mapping(profile, sections, normalization=PRIORITY_NORMALIZATION)
    mapping = {
        "schema_version": "phiagent-qa-priority-compression-map/v2",
        "source_sha256": SOURCE_HASHES, "clauses": clauses,
        "priority_rule": PRIORITY_RULE,
        "priority_rule_source": "Original requested-type hard filters, strongest-evidence surface form and explicit answer-field/plural exceptions take precedence over shortest-answer defaults.",
        "candidate_sha256": {
            variant: {profile: text_sha(text) for profile, text in members.items()}
            for variant, members in output.items()
        },
        "answer_contract_sha256": legacy["answer_contract_sha256"],
        "source_coverage_is_not_semantic_or_quality_equivalence": True,
        "adaptation_scope": "Prospective successor after opened development regressions; requires a new disjoint cohort. No task IDs, questions or answers enter these templates.",
    }
    return output, mapping
