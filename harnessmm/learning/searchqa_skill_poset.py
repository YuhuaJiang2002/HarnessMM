"""Deterministic SearchQA skill-compression variants for poset evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SkillSection:
    """One independently removable SearchQA skill section."""

    name: str
    text: str


@dataclass(frozen=True)
class SearchQASkillVariant:
    """One immutable node in the SearchQA skill-compression poset."""

    variant_id: str
    parent_id: str
    mutation: str
    section_names: tuple[str, ...]
    text: str
    text_sha256: str


SECTIONS: tuple[SkillSection, ...] = (
    SkillSection(
        "evidence_and_target",
        """## Evidence and target
- Infer the requested answer type before choosing an entity. Treat every modifier, date, superlative, letter count, role, and relationship as a hard constraint.
- Find the passage or title where the most distinctive clue terms occur together. Prefer direct statements, close paraphrases, and explicit scraped answer fields; ignore results matching only generic words. A document title is evidence, not automatically the answer.
- Preserve relationship direction: return the requested target, not an entity already named as an anchor. If examples are introduced by "include," "like," or "such as," answer the requested parent class or entity.
- Check each candidate against all clue facts and substitute it back into fill-in wording. When several passages agree, use that corroboration to select the supported canonical form.""",
    ),
    SkillSection(
        "answer_surface",
        """## Answer surface
- Return the shortest unambiguous answer supported by the strongest evidence. Remove repeated type labels, role titles, legal suffixes, and decorative modifiers unless they identify the entity or the question explicitly requests the full form.
- Preserve source-supported spelling, capitalization, punctuation, word order, nicknames, abbreviations, and quoted-title form. Do not replace direct evidence with an equivalent alternate name.
- Keep conventional geographic designators such as Lake, River, Bay, Mount, Gorge, or Island when they are part of the proper name. For people, use the conventional supported full name unless the clue specifically asks for a surname, first name, saint or regnal name, abbreviation, or acronym.
- For common nouns, normally use the singular dictionary headword. For a blank or definition, return a standalone noun phrase rather than a source-dependent pronoun. Obey explicit grammatical form and length constraints.""",
    ),
    SkillSection(
        "clue_formats",
        """## Clue formats
- In Jeopardy-style clues using "this," "he," "his," or "her," infer whether the target is a person, place, work, organization, object, term, or category before extracting it.
- For books, films, songs, poems, plays, quotations, and other works, distinguish the work from its creator, performer, character, source, setting, or missing quoted words by using the governing verb and pronouns.
- A terse list of examples usually asks for their shared category or synonym. For dual-definition or wordplay clues, choose the single form satisfying both senses.
- Scraped trivia may appear as `CATEGORY | clue | answer`, `clue. ANSWER`, or with labels such as `right:`. When the clue matches, extract the adjacent answer field, not the category or full sentence.
- If an image or link is unavailable, rely on the textual clues. If a quotation is incomplete, return the exact missing continuation supported by context.""",
    ),
    SkillSection(
        "edge_cases",
        """## Edge cases
- For "known as," "called," or "defined as" clues, copy the canonical term explicitly linked to the requested relation rather than inventing a derivative or near-synonym.
- In inverse clues, return the relation source: for example, "His third wife was Jiang Qing" asks for the husband, not Jiang Qing.
- For things replaced or substituted by another system, use the broad headword unless the clue requires a narrower modifier.
- Preserve straight ASCII apostrophes when that is the supported surface form. Do not pluralize a shared first name or category merely because the clue's grammar is plural.""",
    ),
)

VARIANT_SECTIONS: Mapping[str, tuple[str, ...]] = {
    "semantic_compression_v1": (
        "evidence_and_target",
        "answer_surface",
        "clue_formats",
        "edge_cases",
    ),
    "compact_v1": (
        "evidence_and_target",
        "answer_surface",
        "clue_formats",
    ),
    "minimal_v1": (
        "evidence_and_target",
        "answer_surface",
    ),
}

PARENTS: Mapping[str, str] = {
    "semantic_compression_v1": "gpt55_full_checkpoint",
    "compact_v1": "semantic_compression_v1",
    "minimal_v1": "compact_v1",
}

MUTATIONS: Mapping[str, str] = {
    "semantic_compression_v1": "section_compression",
    "compact_v1": "strict_section_deletion:edge_cases",
    "minimal_v1": "strict_section_deletion:clue_formats",
}


def build_targeted_conservative_skill(parent_text: str) -> str:
    """Compress only low-risk context guidance while preserving exact answer rules."""
    start = parent_text.find("## Context-Grounded Evidence Matching")
    end = parent_text.find("## Clue Interpretation and Answer Type")
    if start < 0 or end <= start:
        raise ValueError("parent skill section boundaries changed")
    replacement = (
        "## Context-Grounded Evidence Matching\n"
        "- Identify distinctive names, dates, titles, quoted phrases, roles, "
        "relationships, and category descriptors. Prefer passages or titles "
        "where several such clues occur together and directly state the "
        "requested relationship.\n"
        "- A document title is evidence, not automatically the answer. Extract "
        "the entity matching the requested type, ignore generic-only matches, "
        "and prefer explicit canonical terms or scraped answer fields over "
        "invented derivatives and near-synonyms.\n"
        "- Use corroborating passages to select the supported canonical form.\n\n"
    )
    value = parent_text[:start] + replacement + parent_text[end:]
    value = value.replace(
        "\n(No learned rules yet. Rules will be added through the reflection process.)\n",
        "\n",
    )
    value = value.replace(
        "\n<!-- SLOW_UPDATE_START -->\n<!-- SLOW_UPDATE_END -->\n",
        "\n",
    )
    if len(value.encode("utf-8")) >= len(parent_text.encode("utf-8")):
        raise ValueError("targeted compression did not reduce parent bytes")
    preserved_sections = (
        "## Concise Answer Normalization",
        "## Clue Interpretation and Answer Type",
        "## Trivia / Jeopardy Snippet Formats",
        "## Common Clue Traps",
    )
    for heading in preserved_sections:
        parent_start = parent_text.index(heading)
        next_heading = parent_text.find("\n## ", parent_start + len(heading))
        if next_heading < 0:
            next_heading = parent_text.find(
                "\n<!-- SLOW_UPDATE_START -->", parent_start
            )
        parent_section = parent_text[
            parent_start : None if next_heading < 0 else next_heading
        ]
        if parent_section not in value:
            raise ValueError(f"targeted compression changed {heading}")
    return value


def build_metadata_pruned_skill(parent_text: str) -> str:
    """Delete only non-rule placeholder and empty update metadata."""
    value = parent_text.replace(
        "\n(No learned rules yet. Rules will be added through the reflection process.)\n",
        "\n",
    )
    value = value.replace(
        "\n<!-- SLOW_UPDATE_START -->\n<!-- SLOW_UPDATE_END -->\n",
        "\n",
    )
    if value == parent_text:
        raise ValueError("parent skill metadata markers changed")
    if len(value.encode("utf-8")) >= len(parent_text.encode("utf-8")):
        raise ValueError("metadata pruning did not reduce parent bytes")
    for section in (
        "## Concise Answer Normalization",
        "## Context-Grounded Evidence Matching",
        "## Clue Interpretation and Answer Type",
        "## Trivia / Jeopardy Snippet Formats",
        "## Common Clue Traps",
    ):
        if section not in value:
            raise ValueError(f"metadata pruning removed {section}")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_skill(section_names: Sequence[str]) -> str:
    """Render a canonical skill from an ordered subset of known sections."""
    indexed = {section.name: section.text.strip() for section in SECTIONS}
    names = tuple(section_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("skill sections must be a nonempty unique sequence")
    if any(name not in indexed for name in names):
        raise ValueError("skill references an unknown section")
    canonical_order = tuple(section.name for section in SECTIONS if section.name in names)
    if names != canonical_order:
        raise ValueError("skill sections do not preserve canonical order")
    return "# SearchQA Skill\n\n" + "\n\n".join(indexed[name] for name in names) + "\n"


def build_searchqa_skill_variants() -> tuple[SearchQASkillVariant, ...]:
    """Build and validate every preregistered skill-compression node."""
    variants = tuple(
        SearchQASkillVariant(
            variant_id=variant_id,
            parent_id=PARENTS[variant_id],
            mutation=MUTATIONS[variant_id],
            section_names=section_names,
            text=(text := render_skill(section_names)),
            text_sha256=_sha256(text),
        )
        for variant_id, section_names in VARIANT_SECTIONS.items()
    )
    validate_searchqa_skill_variants(variants)
    return variants


def validate_searchqa_skill_variants(
    variants: Sequence[SearchQASkillVariant],
) -> None:
    """Fail closed if the registered compression order is not monotone."""
    by_id = {variant.variant_id: variant for variant in variants}
    if len(by_id) != len(VARIANT_SECTIONS) or set(by_id) != set(VARIANT_SECTIONS):
        raise ValueError("skill variant registry is incomplete or duplicated")
    for variant in variants:
        if variant.text != render_skill(variant.section_names):
            raise ValueError(f"{variant.variant_id} text is not canonical")
        if variant.text_sha256 != _sha256(variant.text):
            raise ValueError(f"{variant.variant_id} hash mismatch")
        forbidden = ("gold answer", "exact match result", "candidate wins")
        if any(fragment in variant.text.lower() for fragment in forbidden):
            raise ValueError(f"{variant.variant_id} contains outcome information")
        if variant.parent_id in by_id:
            parent = by_id[variant.parent_id]
            child_sections = variant.section_names
            parent_sections = parent.section_names
            if len(child_sections) >= len(parent_sections):
                raise ValueError(f"{variant.variant_id} did not delete a section")
            iterator = iter(parent_sections)
            if not all(any(name == candidate for candidate in iterator) for name in child_sections):
                raise ValueError(
                    f"{variant.variant_id} is not an ordered section subsequence"
                )
            if len(variant.text.encode("utf-8")) >= len(parent.text.encode("utf-8")):
                raise ValueError(f"{variant.variant_id} did not reduce bytes")


def variant_manifest() -> dict[str, object]:
    """Return the deterministic candidate registry used by experiment freezes."""
    variants = build_searchqa_skill_variants()
    value: dict[str, object] = {
        "schema_version": "phiagent-searchqa-skill-poset/v1",
        "root": {
            "variant_id": "gpt55_full_checkpoint",
            "relationship": "external_frozen_parent",
        },
        "variants": [
            {
                "variant_id": variant.variant_id,
                "parent_id": variant.parent_id,
                "mutation": variant.mutation,
                "section_names": list(variant.section_names),
                "text_sha256": variant.text_sha256,
                "utf8_bytes": len(variant.text.encode("utf-8")),
                "word_count": len(variant.text.split()),
            }
            for variant in variants
        ],
    }
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return value
