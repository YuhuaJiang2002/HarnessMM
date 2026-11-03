"""Provisional compact QA execution contracts, not validated retrieval innovations."""

from __future__ import annotations

import hashlib
from types import MappingProxyType


COMPACT_CONTRACTS = MappingProxyType({
    "compact_target": (
        "Answer using the supplied context. Identify the exact entity, value or phrase "
        "requested, not an entity merely mentioned in the question. Return only the "
        "shortest supported answer inside <answer>...</answer>. Preserve the wording "
        "needed to identify it. Do not explain.\n"
    ),
    "compact_constraint": (
        "Answer from the supplied context. Match the requested relation and every "
        "qualifier: time, place, role, count or comparison. Check the answer against "
        "the question. Return only the shortest supported canonical answer inside "
        "<answer>...</answer>. Do not explain.\n"
    ),
})


COMPACT_REFINEMENTS = MappingProxyType({
    "compact_complete": (
        "Answer using the supplied context. Identify exactly the entity, value or phrase "
        "requested. Check the relation and every time, place, role, count or comparison "
        "qualifier. Return the complete supported answer, preserving necessary modifiers, "
        "units and names, inside <answer>...</answer>. Do not explain.\n"
    ),
    "compact_complete_no_target": (
        "Answer using the supplied context. "
        "Check the relation and every time, place, role, count or comparison "
        "qualifier. Return the complete supported answer, preserving necessary modifiers, "
        "units and names, inside <answer>...</answer>. Do not explain.\n"
    ),
    "compact_complete_no_qualifiers": (
        "Answer using the supplied context. Identify exactly the entity, value or phrase "
        "requested. Return the complete supported answer, preserving necessary modifiers, "
        "units and names, inside <answer>...</answer>. Do not explain.\n"
    ),
    "compact_complete_shortest": (
        "Answer using the supplied context. Identify exactly the entity, value or phrase "
        "requested. Check the relation and every time, place, role, count or comparison "
        "qualifier. Return the shortest supported answer "
        "inside <answer>...</answer>. Do not explain.\n"
    ),
})


_IMPLICIT_TARGET_RULE = (
    "Treat a declarative clue as an implicit question. Identify the missing subject, "
    "object or category that satisfies the whole clue; return that answer, not the clue.\n"
)
_SPEAKER_ROLE_RULE = (
    "Distinguish the requested entity from speakers, relatives and other entities in "
    "quoted snippets. Resolve pronouns to the requested named entity.\n"
)
COMPACT_SLOT_CONTRACTS = MappingProxyType({
    "compact_implicit_target": COMPACT_CONTRACTS["compact_target"] + _IMPLICIT_TARGET_RULE,
    "compact_speaker_roles": COMPACT_CONTRACTS["compact_target"] + _SPEAKER_ROLE_RULE,
    "compact_target_roles": COMPACT_CONTRACTS["compact_target"] + _IMPLICIT_TARGET_RULE + _SPEAKER_ROLE_RULE,
})


COST_SOURCE_OUTPUTS = MappingProxyType({
    "direct": (
        "Provide exactly one concise final answer inside <answer>...</answer> tags. "
        "Output only these tags and their content. Do not provide reasoning or a preamble.\n"
    ),
    "explain": (
        "Explain how the context supports your answer, then provide exactly one concise "
        "final answer inside <answer>...</answer> tags. Keep all reasoning outside the answer tags.\n"
    ),
})


DELETE_SOURCE_SHA256 = "19fdbb91f0051028d213dd52845dfe40c87be1fda60cb58ea89f270a56333e88"
DELETE_COMPACT_RULES = (
    ("Concise Answer Normalization", (
        "Use the shortest unambiguous answer satisfying the question. Omit generic descriptors, legal suffixes and expanded formal names unless requested or needed to identify the entity.",
        "For name/word etymology, return the distinctive name or word, not a phrase with a generic type label.",
        "Preserve the strongest evidence's spelling, capitalization, punctuation and word order. A directly supplied title/snippet/answer form outranks an equivalent alternate spelling or reordered official/common name.",
        "For person trivia/crossword answers, use the supported conventional name. Use a surname, given name or saint/regnal name alone only when the clue/source expects it; otherwise use the strongest evidence's canonical full name, especially if a given name is ambiguous.",
        "Use the clue's expected grammatical base form. A pluralized crossword/shared-name category does not require pluralizing the answer; people sharing a given name take that singular name.",
        "For common-noun trivia/crossword categories, default to the singular dictionary headword despite plural clue grammar, except inherently plural terms or an explicit plural answer field/source.",
        "For things being replaced or substituted, return the broad headword. Add narrowing modifiers only when required by the clue or answer field, not merely present in context.",
        "For fill-in/definition clues, return a standalone noun phrase, not a source-dependent pronoun or possessive; use a natural article when needed.",
    )),
    ("Context-Grounded Evidence Matching", (
        "Identify distinctive names, dates, titles, quotations, unusual terms, roles, relations and categories in the question.",
        "Prioritize passages/titles jointly matching several distinctive clue facts, especially direct repeats or close paraphrases.",
        "Titles can name the answer while snippets confirm clue facts.",
        "A title is not automatically the answer: if its entity has the wrong requested type, extract the correctly typed entity from the snippet or relation.",
        "For known-as/called/definition/category clues, use the canonical term explicitly supplied by the strongest title/snippet/answer field, not a derivative or near-synonym. Prefer direct relation evidence matching the most distinctive facts.",
        "Ignore generic-word matches; prefer evidence directly linking clue facts to a specific entity.",
    )),
    ("Clue Interpretation and Answer Type", (
        "Infer the requested answer type from Jeopardy wording, including demonstratives, pronouns and family relations.",
        "Validate candidates against that type; return the concise requested person, place, title, organization, object, term or phrase.",
        "Every requested-type modifier is a hard filter: dates, size/rank, name length, remake/version, brand and relational roles must all fit.",
        "For creative works, distinguish the work, creator, performer/cast, character, quotation source and setting. Use verbs and pronouns to identify the requested target.",
        "Substitute candidates into fill-in placeholders; choose the concise answer completing the full phrase, title or fact correctly.",
        "For terse example/name lists separated by commas, slashes or 'or', infer and return their shared category, class or synonym.",
        "Crossword parenthetical/stated letter counts are hard constraints; omit labels violating them. Dual definitions need one word satisfying both senses with the required inflection.",
        "For unavailable image/link or visual hints, infer from the textual clue and context; do not require the missing image.",
        "Use multiple snippets corroborating one entity to choose its canonical/common answer form.",
    )),
    ("Trivia / Jeopardy Snippet Formats", (
        "Recognize scraped clue/answer formats: CATEGORY | clue | answer, clue. ANSWER, and right: labels.",
        "When the question matches a scraped clue, extract its answer field or adjacent answer name, not the category or whole clue.",
    )),
    ("Common Clue Traps", (
        "Watch inverse relationships: a clue identifying 'his wife' may request the husband, not the named wife.",
        "Preserve relation direction: return the requested relation target, not an entity already named as evidence.",
        "For examples/models/breeds/members/items introduced by include/like/such as, return the requested encompassing brand, category, place, entity or term, not a listed example.",
        "For the start of a quotation or phrase, return its exact missing continuation from context.",
        "For songs/poems/rhymes/quotations, distinguish a missing word/phrase from the associated creator, performer or work using pronouns and type signals.",
        "Honor requested first-name, abbreviation, acronym or lyric-word forms instead of fuller identities or explanations; preserve required conventional punctuation and spelling.",
        "Treat wordplay, quotation marks and puns as hints; answer with the real entity supported by evidence.",
        "When a title, quotation, narration, lyric, event, slogan or distinctive phrase identifies an associated 'this' entity, return that requested entity, not the quoted anchor unless explicitly requested.",
    )),
)


def delete_compression_contracts(delete: str) -> tuple[dict[str, str], dict]:
    source_hash = hashlib.sha256(delete.encode("utf-8")).hexdigest()
    if source_hash != DELETE_SOURCE_SHA256 or delete.count("## Answer Contract\n") != 1:
        raise ValueError("DELETE compression requires the exact frozen source and answer contract")
    body, answer_contract = delete.split("## Answer Contract\n")
    prefix = body.split("## Skill\n", 1)[0]
    source_sections: dict[str, list[str]] = {}
    section = None
    for line in body.splitlines():
        if line.startswith("## ") and line != "## Skill":
            section = line[3:]
            source_sections[section] = []
        elif line.startswith("- "):
            if section is None:
                raise ValueError("A source rule lacks a named section")
            source_sections[section].append(line[2:])
    if tuple(source_sections) != tuple(name for name, _rules in DELETE_COMPACT_RULES):
        raise ValueError("The frozen DELETE rule-section inventory changed")
    mapped_rules, compressed_sections, original_rules = [], [], []
    for name, compressed_rules in DELETE_COMPACT_RULES:
        originals = source_sections[name]
        if len(originals) != len(compressed_rules):
            raise ValueError("Every source clause must have exactly one compressed counterpart")
        compressed_sections.append(f"## {name}\n" + "\n".join(f"- {rule}" for rule in compressed_rules))
        original_rules.extend(originals)
        for index, (original, compressed) in enumerate(zip(originals, compressed_rules), start=1):
            mapped_rules.append({
                "section": name, "source_clause": index, "source_text": original,
                "source_clause_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                "compressed_text": compressed,
                "compressed_clause_sha256": hashlib.sha256(compressed.encode("utf-8")).hexdigest(),
            })
    ending = "\n\n## Answer Contract\n" + answer_contract
    contracts = {
        "delete_layout_compact": prefix + "## Rules\n" + "\n".join(f"- {rule}" for rule in original_rules) + ending,
        "delete_rule_compact": prefix + "\n\n".join(compressed_sections) + ending,
    }
    provenance = {
        "schema_version": "phiagent-delete-rule-compression-map/v1",
        "source_sha256": source_hash, "source_clause_count": len(mapped_rules),
        "source_section_count": len(source_sections), "rule_order_preserved": True,
        "answer_contract_preserved_exactly": True,
        "answer_contract_sha256": hashlib.sha256(answer_contract.encode("utf-8")).hexdigest(),
        "clauses": mapped_rules,
        "candidate_sha256": {arm: hashlib.sha256(prompt.encode("utf-8")).hexdigest() for arm, prompt in contracts.items()},
        "semantic_fidelity_status": "INTENDED_CLAUSE_AND_EXCEPTION_PRESERVATION_REQUIRES_EMPIRICAL_VALIDATION",
        "individual_rule_effectiveness_established": False,
        "scope": "Fixed source-only compression; no task answers, learned routing or confirmed behavioral equivalence.",
    }
    return contracts, provenance


MODIFY_SOURCE_SHA256 = "08e74cd6f8d8bfac8881dfde151d6884eec3f9a2db7662eaec8b517ee3af90ee"
MODIFY_EXTRA_COMPACT_RULES = (
    "Strip redundant requested-type labels and role, product or facility descriptors unless the full name is requested or needed.",
    "Keep geographic feature designators when part of the proper name or requested feature type; avoid ambiguous base names.",
    "For organizations and brands, use the distinctive common name; omit corporate suffixes unless required.",
    "Preserve the evidence's ordinary ASCII apostrophes and punctuation unless it explicitly supplies a stylized answer.",
    "Copy supported nicknames, epithets, saint names and quoted titles with exact spacing, case and conventional abbreviations.",
)
SOLPI_SECTION_CAPSULES = (
    "Use the shortest unambiguous supported name or headword; keep necessary modifiers and exact spelling/punctuation. Honor requested short names, singular/plural form and standalone phrases.",
    "Use snippets/titles jointly matching distinctive clue facts. Prefer explicit relation evidence over generic matches or near-synonyms; titles need not be the requested entity.",
    "Infer the requested entity/type. Enforce dates, roles, length and every qualifier. Resolve implicit clues, examples, fills and wordplay; use textual evidence when images are absent.",
    "Extract the matching scraped answer field, not its category or clue.",
    "Keep relation direction; distinguish speakers, quoted anchors, examples, creators and works from the target. Return missing continuations, specified forms or encompassing categories when asked.",
)
SOLPI_MODIFY_CAPSULE = (
    "Omit redundant corporate/type labels, but preserve geographic designators. "
    "Copy exact title, nickname and saint-name forms, including ASCII apostrophes."
)


def chisel_compression_contracts(delete: str, modify: str) -> tuple[dict[str, str], dict]:
    prompts, delete_mapping = delete_compression_contracts(delete)
    if hashlib.sha256(modify.encode("utf-8")).hexdigest() != MODIFY_SOURCE_SHA256:
        raise ValueError("MODIFY compression requires its exact frozen source")
    if modify.count("## Answer Contract\n") != 1:
        raise ValueError("MODIFY requires one answer contract")
    body, answer_contract = modify.split("## Answer Contract\n")
    if answer_contract != delete.split("## Answer Contract\n")[1]:
        raise ValueError("DELETE and MODIFY answer contracts must match")
    common = {row["source_text"]: row["compressed_text"] for row in delete_mapping["clauses"]}
    sections: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        if line.startswith("## ") and line != "## Skill":
            current = line[3:]
            sections[current] = []
        elif line.startswith("- "):
            if current is None:
                raise ValueError("Every MODIFY rule must have a source section")
            sections[current].append(line[2:])
    if tuple(sections) != tuple(name for name, _rules in DELETE_COMPACT_RULES):
        raise ValueError("MODIFY section inventory changed")
    extras, common_seen = [], set()
    mapping = []
    compact_sections = []
    for section, clauses in sections.items():
        compressed_clauses = []
        for original in clauses:
            if original in common:
                compressed = common[original]
                common_seen.add(original)
            else:
                if section != DELETE_COMPACT_RULES[0][0] or len(extras) >= 5:
                    raise ValueError("Unexpected MODIFY-only rule")
                compressed = MODIFY_EXTRA_COMPACT_RULES[len(extras)]
                extras.append(original)
            compressed_clauses.append(compressed)
            mapping.append({
                "section": section, "source_text": original, "compressed_text": compressed,
                "source_clause_sha256": hashlib.sha256(original.encode()).hexdigest(),
            })
        compact_sections.append(f"## {section}\n" + "\n".join(f"- {text}" for text in compressed_clauses))
    if len(extras) != 5 or common_seen != set(common):
        raise ValueError("MODIFY must retain all 33 DELETE clauses and five extra rules")
    prefix = body.split("## Skill\n", 1)[0]
    ending = "\n\n## Answer Contract\n" + answer_contract
    capsule = "Answer from the supplied context.\n" + "\n".join(SOLPI_SECTION_CAPSULES)
    candidates = {
        "delete_clause": prompts["delete_rule_compact"],
        "modify_clause": prefix + "\n\n".join(compact_sections) + ending,
        "delete_capsule": capsule + ending,
        "modify_capsule": capsule + "\n" + SOLPI_MODIFY_CAPSULE + ending,
    }
    return candidates, {
        "schema_version": "phiagent-chisel-solpi-compression/v1",
        "source_delete_sha256": DELETE_SOURCE_SHA256,
        "source_modify_sha256": MODIFY_SOURCE_SHA256,
        "delete_clause_map": delete_mapping,
        "modify_clauses": mapping,
        "modify_only_source_clauses": extras,
        "section_capsules": [
            {"section": section, "summary": summary}
            for (section, _rules), summary in zip(DELETE_COMPACT_RULES, SOLPI_SECTION_CAPSULES)
        ],
        "modify_only_capsule": SOLPI_MODIFY_CAPSULE,
        "answer_contract_preserved_exactly": True,
        "semantic_equivalence_established": False,
        "candidate_sha256": {
            key: hashlib.sha256(text.encode("utf-8")).hexdigest() for key, text in candidates.items()
        },
    }


def cost_source_contracts(no_skill: str, delete: str) -> dict[str, str]:
    reasoning = "Think step by step, then provide your final answer inside <answer>...</answer> tags."
    if (no_skill.count("## Answer Format\n") != 1 or delete.count("## Answer Contract\n") != 1
            or no_skill.count(reasoning) != 1):
        raise ValueError("the frozen baseline answer-format boundaries must be unique")
    short_body = no_skill.split("## Answer Format\n", 1)[0]
    long_body = delete.split("## Answer Contract\n", 1)[0]
    if not short_body.strip() or not long_body.strip():
        raise ValueError("both fixed prompt bodies must be nonempty")
    contracts = {
        "no_skill_answer_only": no_skill.replace(
            reasoning,
            "Immediately provide your final answer inside <answer>...</answer> tags. "
            "Do not provide reasoning or a preamble.",
        ),
    }
    for content, body in (("short", short_body), ("long", long_body)):
        for output, contract in COST_SOURCE_OUTPUTS.items():
            contracts[f"cost_{content}_{output}"] = body + "## Answer Contract\n" + contract
    return contracts


def select_training_groups(
    groups: set[str], excluded: set[str], *, count: int, salt: str
) -> tuple[str, ...]:
    import hashlib

    if type(count) is not int or count <= 0 or not salt:
        raise ValueError("positive group count and explicit selection salt are required")
    eligible = groups - excluded
    if len(eligible) < count:
        raise ValueError("not enough nonoverlapping training groups")
    return tuple(sorted(
        eligible,
        key=lambda group: (hashlib.sha256(f"{salt}\0{group}".encode()).digest(), group),
    )[:count])
