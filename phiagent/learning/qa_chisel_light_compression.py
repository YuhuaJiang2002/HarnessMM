"""One experimental, source-bound wording adapter for Original CHISEL.

Only skill-rule sentences are edited. Callers must retain the original trained
head, DELETE/MODIFY routing, complete context, and generation/output settings.
Literal condition checks and prompt hashes are not a semantic-equivalence or
success-rate certificate. No task text, labels, model, or tokenizer is consumed.

Pinned CPU-only static counts save 155/187 raw system tokens (DELETE/MODIFY),
below the requested 300-450 average input-token reduction. Chat-template,
completion-token, and success-rate effects remain unvalidated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType


CANDIDATE_ID = "original-chisel-light-compression-v1"
SOURCE_SNAPSHOT = (
    "/opt/phiagent/work/compact-harness-refinement/"
    "20260916-core-confirmation-v3/source"
)
SKILL_START = "\n\n## Skill\n"
SKILL_END = "\n\n## Answer Contract\n"

_NORMALIZATION = "## Concise Answer Normalization"
_EVIDENCE = "## Context-Grounded Evidence Matching"
_INTERPRETATION = "## Clue Interpretation and Answer Type"
_TRAPS = "## Common Clue Traps"
_BOTH = ("delete", "modify")


@dataclass(frozen=True, slots=True)
class PromptBinding:
    source_relative_path: str
    source_sha256: str
    output_sha256: str


@dataclass(frozen=True, slots=True)
class SkillRuleEdit:
    rule_id: str
    section: str
    source: str
    replacement: str
    preserved_fragments: tuple[str, ...]
    profiles: tuple[str, ...] = _BOTH


@dataclass(frozen=True, slots=True)
class LightCompressionRequest:
    system_prompt: str
    receipt: dict[str, object]


@dataclass(frozen=True, slots=True)
class StaticTokenCounts:
    source_system_tokens: int
    output_system_tokens: int
    source_skill_tokens: int
    output_skill_tokens: int


SOURCE_BINDINGS: Mapping[str, PromptBinding] = MappingProxyType({
    "delete": PromptBinding(
        "prompts/delete.system.txt",
        "19fdbb91f0051028d213dd52845dfe40c87be1fda60cb58ea89f270a56333e88",
        "72b0422938a15e313429b17a1f0f2c58014f49c7a57d23aa4bf7ecb63d321a85",
    ),
    "modify": PromptBinding(
        "prompts/modify.system.txt",
        "08e74cd6f8d8bfac8881dfde151d6884eec3f9a2db7662eaec8b517ee3af90ee",
        "84c9c84ac5f947624778c9bdd3c2856b0c2f0429a19144e33f71bfd02fb446a0",
    ),
})

STATIC_TOKENIZER_BINDING: Mapping[str, str] = MappingProxyType({
    "server_alias": "h200-1",
    "model_path": (
        "/opt/phiagent/work/same-model-services/"
        "20260831T151445Z-h2001-qwen38-27b-cu129-v14-textonly-torchcodecfix/model"
    ),
    "python_path": (
        "/opt/phiagent/work/same-model-services/"
        "20260831T151445Z-h2001-qwen38-27b-cu129-v14-textonly-torchcodecfix/runtime/bin/python"
    ),
    "tokenizers_version": "0.23.1",
    "tokenizer_json_sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config_sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "method": "tokenizers.Tokenizer.from_file; encode(text, add_special_tokens=False)",
    "system_scope": "raw system_prompt only; no chat template or task text",
    "skill_scope": "SKILL_START plus skill body; excludes SKILL_END",
    "CUDA_VISIBLE_DEVICES": "",
    "status": "STATIC_CONFIGURATION_ONLY_NOT_AN_EXPERIMENT",
})

STATIC_TOKEN_COUNTS: Mapping[str, StaticTokenCounts] = MappingProxyType({
    "delete": StaticTokenCounts(1726, 1571, 1683, 1528),
    "modify": StaticTokenCounts(2051, 1864, 2008, 1821),
})


RULE_EDITS = (
    SkillRuleEdit(
        "normalization.shortest-unambiguous",
        _NORMALIZATION,
        "- Prefer the shortest unambiguous answer that directly satisfies the question. "
        "Do not include generic descriptors, legal suffixes, or expanded formal names unless "
        "the question specifically asks for the full official name or the descriptor is "
        "necessary to identify the entity.",
        "- Prefer the shortest unambiguous direct answer. Omit "
        "generic descriptors, legal suffixes, or expanded formal names unless the full "
        "official name is specifically requested or the descriptor is necessary to identify "
        "the entity.",
        (
            "shortest unambiguous", "generic descriptors, legal suffixes",
            "expanded formal names unless", "full official name",
            "descriptor is necessary to identify the entity",
        ),
    ),
    SkillRuleEdit(
        "normalization.descriptive-phrase",
        _NORMALIZATION,
        "- If the answer appears inside a longer descriptive phrase, strip words that merely "
        "repeat the clue's requested type or modifiers already stated in the clue. For "
        "short-answer trivia, return the distinctive core entity or headword rather than role "
        "titles, product flavor adjectives, or place/facility designators, even when those "
        "words are part of a fuller official phrase, unless the full official name is "
        "explicitly requested.",
        "- In longer descriptive phrases, strip words merely repeating the clue's requested "
        "type or stated modifiers. For short-answer trivia, return the distinctive core "
        "entity or headword, not role titles, product flavor adjectives, or place/facility "
        "designators, even within fuller official phrases, unless the full official name is "
        "explicitly requested.",
        (
            "clue's requested type", "modifiers", "short-answer trivia",
            "distinctive core entity or headword",
            "role titles, product flavor adjectives, or place/facility designators",
            "unless the full official name is explicitly requested",
        ),
        ("modify",),
    ),
    SkillRuleEdit(
        "normalization.etymology",
        _NORMALIZATION,
        "- For place/name-etymology questions asking for “the name” or “the word” that means "
        "something, answer the distinctive name/word itself rather than a larger phrase with "
        "a generic type label.",
        "- For place/name-etymology clues asking for “the name” or “the word” meaning "
        "something, give the distinctive name/word without a generic type label.",
        (
            "place/name-etymology", "“the name” or “the word”",
            "distinctive name/word", "generic type label",
        ),
    ),
    SkillRuleEdit(
        "normalization.geographic-designators",
        _NORMALIZATION,
        "- For natural geographic features, preserve conventional feature designators such "
        "as “Lake,” “River,” “Bay,” “Gorge,” “Mount,” or “Island” when they are part of the "
        "proper name or match the requested feature type. Do not shorten “Lake Okeechobee,” "
        "“Tampa Bay,” or “Olduvai Gorge” to an ambiguous base name merely to be concise.",
        "- For natural geographic features, keep conventional designators such as “Lake,” "
        "“River,” “Bay,” “Gorge,” “Mount,” or “Island” when part of the proper name or matching "
        "the requested feature type. Do not shorten “Lake Okeechobee,” “Tampa Bay,” or "
        "“Olduvai Gorge” to an ambiguous base name for concision.",
        (
            "natural geographic features",
            "such as “Lake,” “River,” “Bay,” “Gorge,” “Mount,” or “Island”",
            "proper name or", "requested feature type",
            "Do not shorten “Lake Okeechobee,” “Tampa Bay,” or “Olduvai Gorge”",
            "ambiguous base name",
        ),
        ("modify",),
    ),
    SkillRuleEdit(
        "normalization.organization-suffixes",
        _NORMALIZATION,
        "- For companies, brands, and organizations, answer the common distinctive name when "
        "sufficient; omit additions such as “Company,” “Corporation,” “Inc.,” etc. unless "
        "explicitly required.",
        "- For companies, brands, and organizations, use the common distinctive name when "
        "sufficient; omit “Company,” “Corporation,” “Inc.,” etc. unless explicitly required.",
        (
            "companies, brands, and organizations", "common distinctive name when sufficient",
            "“Company,” “Corporation,” “Inc.,” etc. unless explicitly required",
        ),
        ("modify",),
    ),
    SkillRuleEdit(
        "normalization.evidence-surface",
        _NORMALIZATION,
        "- Preserve the answer surface form supported by the strongest evidence when exact "
        "variants differ: spelling, capitalization, punctuation, and word order can matter. "
        "Do not substitute an equivalent official/common variant such as an alternate "
        "spelling or inverted institution name if a direct title/snippet/answer field gives "
        "the expected form.",
        "- When exact variants differ, preserve the strongest evidence's answer "
        "spelling, capitalization, punctuation, and word order. If a direct title/snippet/"
        "answer field gives the expected form, do not substitute an equivalent official/common "
        "variant (e.g., an alternate spelling or inverted institution name).",
        (
            "exact variants differ", "strongest evidence",
            "spelling, capitalization, punctuation, and word order",
            "a direct title/snippet/answer field gives the expected form",
            "equivalent official/common", "an alternate spelling or inverted institution name",
        ),
    ),
    SkillRuleEdit(
        "normalization.ascii-punctuation",
        _NORMALIZATION,
        "- When copying titles or quoted names, preserve ordinary ASCII punctuation from the "
        "evidence, especially straight apostrophes (`'`). Do not replace them with typographic "
        "curly quotes/apostrophes unless that exact stylized form is explicitly shown as the "
        "supported answer.",
        "- For titles or quoted names, copy evidence's ordinary ASCII punctuation, especially "
        "straight apostrophes (`'`); use typographic curly quotes/apostrophes only when that "
        "exact stylized form is explicitly shown as the supported answer.",
        (
            "titles or quoted names", "ordinary ASCII punctuation",
            "especially straight apostrophes (`'`)", "typographic curly quotes/apostrophes",
            "that exact stylized form is explicitly shown as the supported answer",
        ),
        ("modify",),
    ),
    SkillRuleEdit(
        "normalization.stylized-names",
        _NORMALIZATION,
        "- For nicknames, epithets, saints, and quoted titles, copy the supported surface form "
        "exactly, including spacing, capitalization, and conventional abbreviations such as "
        "“St.” Do not normalize a stylized or quoted form into a lowercase dictionary word or "
        "an expanded spelling when the clue/evidence points to the stylized answer.",
        "- Copy supported nicknames, epithets, saints, and quoted titles exactly: spacing, "
        "capitalization, and conventional abbreviations such as “St.” When the clue/evidence "
        "points to the stylized answer, do not change a stylized or quoted form to a lowercase "
        "dictionary word or an expanded spelling.",
        (
            "nicknames, epithets, saints, and quoted titles",
            "supported", "exactly",
            "spacing, capitalization, and conventional abbreviations such as “St.”",
            "a stylized or quoted form", "a lowercase dictionary word or an expanded spelling",
            "the clue/evidence points to the stylized answer",
        ),
        ("modify",),
    ),
    SkillRuleEdit(
        "normalization.person-name",
        _NORMALIZATION,
        "- For person answers in trivia or crossword-style clues, prefer the conventional "
        "supported name. Use just a surname, first name, or saint/regnal name only when the "
        "clue/source clearly expects that short form; otherwise use the canonical full personal "
        "name from the strongest evidence or answer field, especially when a lone given name "
        "would be ambiguous.",
        "- For trivia/crossword person answers, prefer the conventional supported name: "
        "a surname, first name, or saint/regnal name only when the clue/source clearly "
        "expects that short form; otherwise the canonical full personal name from the "
        "strongest evidence or answer field, especially if a lone given name is ambiguous.",
        (
            "conventional supported name", "a surname, first name, or saint/regnal name",
            "only when the clue/source clearly expects that short form",
            "canonical full personal name",
            "strongest evidence or answer field", "a lone given name", "ambiguous",
        ),
    ),
    SkillRuleEdit(
        "normalization.grammatical-base",
        _NORMALIZATION,
        "- Return the grammatical base form expected by the clue. Do not add a plural `s` "
        "merely because a crossword source pluralizes a shared name or category; if the clue "
        "lists people sharing a first name, answer the singular given name.",
        "- Return the clue's grammatical base form. Do not add a plural `s` just "
        "because a crossword source pluralizes a shared name or category; for listed "
        "people sharing a first name, answer the singular given name.",
        (
            "grammatical base form", "Do not add a plural `s`",
            "a crossword source pluralizes a shared name or category",
            "people sharing a first name", "answer the singular given name",
        ),
    ),
    SkillRuleEdit(
        "normalization.common-noun-number",
        _NORMALIZATION,
        "- For common-noun category answers, default to the singular dictionary headword in "
        "trivia/crossword-style clues, even if the clue uses plural words like “these,” “those,” "
        "“places,” or “items” for grammar. Use a plural only when the term is inherently plural "
        "or an answer field/source clearly gives a plural phrase.",
        "- For trivia/crossword common-noun categories, default to the singular dictionary "
        "headword despite grammatical plurals like “these,” “those,” “places,” or “items.” "
        "Use a plural only when the term is inherently plural or an answer field/source "
        "clearly gives a plural phrase.",
        (
            "default to the singular dictionary headword", "“these,” “those,” “places,”",
            "Use a plural only when the term is inherently plural or an answer field/source "
            "clearly gives a plural phrase",
        ),
    ),
    SkillRuleEdit(
        "normalization.replaced-headword",
        _NORMALIZATION,
        "- For common-noun clues about things being replaced, used in place of, or substituted "
        "by another system/item, answer the broad headword for the thing replaced unless a "
        "narrowing modifier is required by the clue or answer field. Do not add adjectives "
        "such as “letter,” “regular,” or “standard” merely because they appear in explanatory "
        "context.",
        "- For common-noun clues about things replaced, used in place of, or substituted by "
        "another system/item, give the replaced thing's broad headword unless the "
        "clue or answer field requires a narrowing modifier. Do not add adjectives such as "
        "“letter,” “regular,” or “standard” merely from explanatory context.",
        (
            "common-noun clues", "used in place of, or substituted by another system/item",
            "broad headword", "unless", "clue or answer field",
            "narrowing modifier",
            "Do not add adjectives such as “letter,” “regular,” or “standard”",
            "explanatory context",
        ),
    ),
    SkillRuleEdit(
        "normalization.standalone-phrase",
        _NORMALIZATION,
        "- For fill-in-the-blank or definitional clues using words like “this” or “that,” "
        "provide a standalone noun phrase. Avoid context-dependent pronouns or possessives "
        "from the source text; use a natural article such as “the” when needed (e.g., answer "
        "“the highest point,” not “its highest point”).",
        "- For fill-in-the-blank or definitional clues with “this” or “that,” give a standalone "
        "noun phrase, not context-dependent pronouns or possessives from the source. Use a "
        "natural article such as “the” when needed (e.g., “the highest point,” not "
        "“its highest point”).",
        (
            "fill-in-the-blank or definitional clues", "“this” or “that,”",
            "standalone noun phrase", "context-dependent pronouns or possessives",
            "natural article such as “the” when needed",
            "“the highest point,” not “its highest point”",
        ),
    ),
    SkillRuleEdit(
        "evidence.distinctive-terms",
        _EVIDENCE,
        "- Start by identifying the most distinctive terms in the question: proper names, "
        "dates, titles, quoted phrases, unusual words, roles, relationships, and category "
        "descriptors.",
        "- First identify the question's most distinctive terms: proper names, dates, titles, "
        "quoted phrases, unusual words, roles, relationships, and category descriptors.",
        (
            "most distinctive terms", "proper names, dates, titles, quoted phrases",
            "unusual words, roles, relationships, and category descriptors",
        ),
    ),
    SkillRuleEdit(
        "evidence.co-occurring-terms",
        _EVIDENCE,
        "- Prioritize passages or document titles where several distinctive clue terms occur "
        "together, especially if the wording directly repeats or closely paraphrases the "
        "question.",
        "- Prioritize passages or document titles combining several distinctive clue terms, "
        "especially wording that directly repeats or closely paraphrases the question.",
        (
            "Prioritize passages or document titles", "several distinctive clue terms",
            "especially", "directly repeats or closely paraphrases the question",
        ),
    ),
    SkillRuleEdit(
        "evidence.title-plus-snippet",
        _EVIDENCE,
        "- Treat document titles as useful evidence: the answer is often named in a title "
        "while the snippet confirms the clue facts.",
        "- Use document titles as evidence: they often name the answer while the snippet "
        "confirms the clue facts.",
        ("document titles", "evidence", "often", "while the snippet confirms the clue facts"),
    ),
    SkillRuleEdit(
        "evidence.title-type-mismatch",
        _EVIDENCE,
        "- Do not assume the document title itself is the answer. If the requested type "
        "differs from the title entity, use the title as context and extract the matching "
        "typed entity from the snippet or clue relationship.",
        "- Do not assume the title is the answer. If the requested type differs from its "
        "entity, use the title as context; extract the matching typed entity from the snippet "
        "or clue relationship.",
        (
            "Do not assume", "If the requested type differs",
            "use the title as context",
            "matching typed entity from the snippet or clue relationship",
        ),
    ),
    SkillRuleEdit(
        "evidence.canonical-term-and-candidates",
        _EVIDENCE,
        "- For “known as,” “called,” “defined as,” or category/type clues, choose the canonical "
        "term explicitly used in the strongest matching title/snippet or scraped answer field "
        "rather than inventing a related derivative or near-synonym from the clue wording. "
        "When multiple plausible candidates appear, prefer the candidate whose evidence "
        "directly states the requested relationship and repeats the most distinctive clue "
        "facts.",
        "- For “known as,” “called,” “defined as,” or category/type clues, choose the canonical "
        "term explicitly used in the strongest matching title/snippet or scraped answer field, "
        "not a related derivative or near-synonym invented from clue wording. Among plausible "
        "candidates, prefer one whose evidence directly states the requested relationship and "
        "repeats the most distinctive clue facts.",
        (
            "“known as,” “called,” “defined as,” or category/type clues",
            "canonical term explicitly used in the strongest matching title/snippet",
            "scraped answer field", "related derivative or near-synonym", "clue wording",
            "plausible candidates", "evidence directly states the requested relationship",
            "repeats the most distinctive clue facts",
        ),
    ),
    SkillRuleEdit(
        "interpretation.validate-type",
        _INTERPRETATION,
        "- Use that expected type to validate candidates: answer with the concise person, "
        "place, title, organization, object, term, or phrase requested by the clue.",
        "- Validate candidates against that expected type: give the concise person, place, "
        "title, organization, object, term, or phrase requested by the clue.",
        (
            "that expected type", "concise person, place, title, organization, object, term, "
            "or phrase requested by the clue",
        ),
    ),
    SkillRuleEdit(
        "interpretation.creative-work-target",
        _INTERPRETATION,
        "- For clues centered on creative works such as books, films, plays, songs, poems, or "
        "other media, first determine whether the clue asks for the work itself, its creator, "
        "a performer or cast member, a character, a quotation source, or a setting. Verbs such "
        "as “wrote,” “directed,” “stars,” “played,” and “set in,” plus pronouns like “he” or "
        "“her,” usually determine the target.",
        "- For creative-work clues (books, films, plays, songs, poems, or other media), first "
        "identify the requested target: work, creator, performer or cast member, character, "
        "quotation source, or setting. Verbs such as “wrote,” “directed,” “stars,” “played,” "
        "and “set in,” plus pronouns like “he” or “her,” usually determine it.",
        (
            "books, films, plays, songs, poems, or other media", "first",
            "work", "creator", "performer or cast member", "character", "quotation source",
            "setting", "Verbs such as “wrote,” “directed,” “stars,” “played,” and “set in,”",
            "plus pronouns like “he” or “her,” usually determine",
        ),
    ),
    SkillRuleEdit(
        "interpretation.fill-in-substitution",
        _INTERPRETATION,
        "- For fill-in-style clues with placeholders such as “this,” “these,” or “one of "
        "these,” substitute each candidate back into the clue and choose the concise answer "
        "that makes the full phrase, title, or fact read correctly.",
        "- For fill-in-style placeholders such as “this,” “these,” or “one of these,” "
        "substitute each candidate back into the clue; choose the concise answer making the "
        "full phrase, title, or fact read correctly.",
        (
            "fill-in-style", "placeholders such as “this,” “these,” or “one of these,”",
            "substitute each candidate back into the clue", "concise answer",
            "full phrase, title, or fact read correctly",
        ),
    ),
    SkillRuleEdit(
        "interpretation.terse-common-term",
        _INTERPRETATION,
        "- For terse clues that are just examples or names separated by commas, slashes, or "
        "“or,” infer the shared category, class, or synonym that links them, then answer "
        "with that concise common term.",
        "- For terse example/name lists separated by commas, slashes, or “or,” infer their "
        "shared category, class, or synonym; answer with that concise common term.",
        (
            "terse", "separated by commas, slashes, or “or,”",
            "shared category, class, or synonym", "answer with that concise common term",
        ),
    ),
    SkillRuleEdit(
        "interpretation.crossword-constraints",
        _INTERPRETATION,
        "- For crossword-style clues, treat parenthetical numbers or stated letter counts "
        "as hard constraints on the answer length, and omit generic labels that would violate "
        "them. In dual-definition clues using wording like “X, or what Y does,” choose the "
        "single word that satisfies both senses and preserve the required inflected form.",
        "- For crossword-style clues, parenthetical numbers or stated letter counts are hard "
        "answer-length constraints; omit violating generic labels. For dual-definition clues "
        "like “X, or what Y does,” choose the single word satisfying both senses in the "
        "required inflected form.",
        (
            "crossword-style clues", "parenthetical numbers or stated letter counts",
            "hard", "constraints", "generic labels", "dual-definition clues",
            "“X, or what Y does,”", "single word", "both senses", "required inflected form",
        ),
    ),
    SkillRuleEdit(
        "interpretation.unavailable-visual",
        _INTERPRETATION,
        "- If the clue references an unavailable image or link with wording like “seen here,” "
        "“pictured,” or parenthetical visual hints, rely on the textual clues and context to "
        "infer the answer; do not treat the missing image as necessary evidence.",
        "- For an unavailable image or link (“seen here,” “pictured,” or parenthetical visual "
        "hints), infer the answer from textual clues and context; the missing image is not "
        "necessary evidence.",
        (
            "unavailable image or link", "“seen here,” “pictured,” or parenthetical visual hints",
            "infer the answer", "textual clues and context", "missing image", "necessary evidence",
        ),
    ),
    SkillRuleEdit(
        "interpretation.corroboration",
        _INTERPRETATION,
        "- If multiple snippets support the same entity, use that corroboration to choose the "
        "canonical/common form of the answer.",
        "- If multiple snippets support the same entity, use their corroboration to choose "
        "the canonical/common answer form.",
        (
            "If multiple snippets support the same entity",
            "corroboration", "canonical/common",
        ),
    ),
    SkillRuleEdit(
        "traps.relation-direction",
        _TRAPS,
        "- More generally, preserve relation direction in clues: “A is evidence of this B,” "
        "“A is related to this language,” or “home to these characters” asks for the target "
        "of the relationship, not the entity already named in the clue.",
        "- Preserve relation direction: “A is evidence of this B,” “A is related to this "
        "language,” or “home to these characters” asks for the relationship's target, not "
        "the already named entity.",
        (
            "relation direction", "“A is evidence of this B,”",
            "“A is related to this language,” or “home to these characters”",
            "target", "not", "entity",
        ),
    ),
    SkillRuleEdit(
        "traps.parent-not-example",
        _TRAPS,
        "- When a clue says examples, models, breeds, members, or items “include,” “like,” or "
        "“such as” named entities, treat those names as evidence for the requested parent class "
        "or entity. Answer the encompassing brand, animal, category, place, or term requested "
        "by “this,” not one of the examples already given.",
        "- In “include,” “like,” or “such as” clues, named examples, models, breeds, members, "
        "or items are evidence for the requested parent class or entity. Answer the encompassing "
        "brand, animal, category, place, or term requested by “this,” not an example already "
        "given.",
        (
            "examples, models, breeds, members, or items", "“include,” “like,” or “such as”",
            "evidence for the requested parent class or entity",
            "Answer the encompassing brand, animal, category, place, or term requested by “this,”",
            "not", "already given",
        ),
    ),
    SkillRuleEdit(
        "traps.missing-continuation",
        _TRAPS,
        "- If the question gives the start of a quotation or phrase, answer with the exact "
        "missing continuation from the context.",
        "- If the question gives a quotation or phrase's start, return the exact missing "
        "continuation from the context.",
        (
            "If the question gives", "quotation or phrase",
            "exact missing continuation from the context",
        ),
    ),
    SkillRuleEdit(
        "traps.quote-versus-creator",
        _TRAPS,
        "- For song, poem, nursery-rhyme, or quotation clues, first decide whether the question "
        "asks for a missing word or phrase from the quote or for the associated creator, "
        "performer, or work; use pronouns and answer-type signals to choose the right target.",
        "- For song, poem, nursery-rhyme, or quotation clues, first distinguish a missing "
        "quote word or phrase from an associated creator, performer, or work; choose the target "
        "using pronouns and answer-type signals.",
        (
            "song, poem, nursery-rhyme, or quotation clues", "first", "missing",
            "word or phrase", "associated creator, performer, or work",
            "pronouns and answer-type signals",
        ),
    ),
    SkillRuleEdit(
        "traps.constrained-form",
        _TRAPS,
        "- When a clue asks for a constrained form such as a first name, abbreviation, acronym, "
        "or lyric word, return that exact form rather than the fuller person, title, or "
        "explanation; preserve conventional punctuation or spelling when it is part of the "
        "requested form.",
        "- Return the exact constrained form requested (e.g., first name, abbreviation, acronym, "
        "or lyric word), not the fuller person, title, or explanation; preserve conventional "
        "punctuation or spelling belonging to that form.",
        (
            "constrained form", "exact", "first name, abbreviation, acronym, or lyric word",
            "fuller person, title, or explanation", "preserve conventional punctuation or spelling",
        ),
    ),
    SkillRuleEdit(
        "traps.associated-entity-not-anchor",
        _TRAPS,
        "- If a clue includes a quoted title, quoted narration or lyric, named event, slogan, "
        "or other distinctive phrase but asks for an associated “this” entity, treat the quote "
        "or name as evidence to identify the requested person, work, place, group, category, "
        "source, or term; do not return the quoted anchor unless the clue explicitly asks for it.",
        "- If a clue gives a quoted title, quoted narration or lyric, named event, slogan, "
        "or other distinctive phrase but asks for an associated “this” entity, use the quote "
        "or name as evidence for the requested person, work, place, group, category, source, "
        "or term. Return the quoted anchor only if the clue explicitly asks for it.",
        (
            "quoted title, quoted narration or lyric, named event, slogan, or other distinctive "
            "phrase", "but asks for an associated “this” entity", "quote or name as evidence",
            "requested person, work, place, group, category, source, or term",
            "quoted anchor", "the clue explicitly asks for it",
        ),
    ),
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EDIT_PLAN_SHA256 = _sha256(json.dumps(
    [asdict(edit) for edit in RULE_EDITS],
    ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
))


def _split_skill(system_prompt: str) -> tuple[str, str, str]:
    if system_prompt.count(SKILL_START) != 1 or system_prompt.count(SKILL_END) != 1:
        raise ValueError("Expected exactly one original skill and output-contract envelope")
    before, _, remainder = system_prompt.partition(SKILL_START)
    skill, _, after = remainder.partition(SKILL_END)
    if not skill or SKILL_END in before:
        raise ValueError("Original skill must precede the output contract")
    return before + SKILL_START, skill, SKILL_END + after


def _sections(skill: str) -> dict[str, tuple[int, int]]:
    headings = []
    offset = 0
    for line in skill.splitlines(keepends=True):
        if line.startswith("## "):
            headings.append((line.rstrip("\r\n"), offset))
        offset += len(line)
    if len({heading for heading, _ in headings}) != len(headings):
        raise ValueError("Duplicate skill section heading")
    return {
        heading: (start, headings[index + 1][1] if index + 1 < len(headings) else len(skill))
        for index, (heading, start) in enumerate(headings)
    }


def _apply_skill_edits(
    skill: str, profile: str, *, prefix_length: int, source_sha256: str,
) -> tuple[str, list[dict[str, object]]]:
    sections = _sections(skill)
    changes: list[tuple[int, int, SkillRuleEdit, str]] = []
    for edit in RULE_EDITS:
        if profile not in edit.profiles:
            continue
        if (
            not edit.source.startswith("- ") or not edit.replacement.startswith("- ")
            or "\n" in edit.source or "\n" in edit.replacement
            or "\r" in edit.source or "\r" in edit.replacement
            or len(edit.replacement) >= len(edit.source)
        ):
            raise ValueError(f"Edit must shorten one complete rule sentence: {edit.rule_id}")
        if not all(
            fragment in edit.source and fragment in edit.replacement
            for fragment in edit.preserved_fragments
        ):
            raise ValueError(f"Preserved condition fragment differs: {edit.rule_id}")
        if edit.section not in sections:
            raise ValueError(f"Missing source section: {edit.section}")
        section_start, section_end = sections[edit.section]
        section_text = skill[section_start:section_end]
        matches = []
        offset = section_start
        for line in section_text.splitlines(keepends=True):
            if line.rstrip("\r\n") == edit.source:
                matches.append(offset)
            offset += len(line)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source rule: {edit.rule_id}")
        start = matches[0]
        changes.append((start, start + len(edit.source), edit, _sha256(section_text)))
    if not changes:
        raise ValueError(f"No frozen rule edits for profile: {profile}")

    chunks = []
    receipts: list[dict[str, object]] = []
    cursor = 0
    for start, end, edit, section_sha256 in sorted(changes, key=lambda change: change[0]):
        if start < cursor:
            raise ValueError(f"Overlapping source rule edits: {edit.rule_id}")
        chunks.extend((skill[cursor:start], edit.replacement))
        receipts.append({
            "rule_id": edit.rule_id,
            "section": edit.section,
            "source_prompt_sha256": source_sha256,
            "source_section_sha256": section_sha256,
            "source_rule_sha256": _sha256(edit.source),
            "output_rule_sha256": _sha256(edit.replacement),
            "source_skill_span": [start, end],
            "source_prompt_span": [prefix_length + start, prefix_length + end],
            "span_unit": "unicode_codepoints_half_open",
            "source": edit.source,
            "replacement": edit.replacement,
            "preserved_fragments": list(edit.preserved_fragments),
        })
        cursor = end
    chunks.append(skill[cursor:])
    output = "".join(chunks)
    if tuple(_sections(output)) != tuple(sections):
        raise ValueError("Skill section headings changed")
    if sum(line.startswith("- ") for line in output.splitlines()) != sum(
        line.startswith("- ") for line in skill.splitlines()
    ):
        raise ValueError("A skill rule was removed or inserted")
    return output, receipts


def light_compression_request(
    *, system_prompt: str, profile: str,
) -> LightCompressionRequest:
    """Apply the one frozen candidate, or fail closed on any changed input.

    ``profile`` is the caller's original route, not a new routing decision.
    Receipts describe adapter-local preservation only; they cannot attest to
    the caller's context, trained-head binding, or generation settings.
    """
    if not isinstance(profile, str) or profile not in _BOTH:
        raise ValueError("Unsupported profile: expected 'delete' or 'modify'")
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")
    binding = SOURCE_BINDINGS[profile]
    source_sha256 = _sha256(system_prompt)
    if source_sha256 != binding.source_sha256:
        raise ValueError(
            f"Source prompt SHA256 mismatch for {profile}: "
            f"expected {binding.source_sha256}, got {source_sha256}"
        )
    prefix, skill, suffix = _split_skill(system_prompt)
    output_skill, edits = _apply_skill_edits(
        skill, profile, prefix_length=len(prefix), source_sha256=source_sha256,
    )
    output_prompt = prefix + output_skill + suffix
    output_sha256 = _sha256(output_prompt)
    if output_sha256 != binding.output_sha256:
        raise ValueError(
            f"Frozen output SHA256 mismatch for {profile}: "
            f"expected {binding.output_sha256}, got {output_sha256}"
        )
    source_sections, output_sections = _sections(skill), _sections(output_skill)
    changed_sections = list(dict.fromkeys(str(edit["section"]) for edit in edits))
    section_bindings = [
        {
            "heading": heading,
            "changed": heading in changed_sections,
            "source_sha256": _sha256(skill[start:end]),
            "output_sha256": _sha256(output_skill[slice(*output_sections[heading])]),
        }
        for heading, (start, end) in source_sections.items()
    ]
    return LightCompressionRequest(output_prompt, {
        "schema_version": "phiagent-qa-chisel-light-compression/v1",
        "candidate_id": CANDIDATE_ID,
        "status": "EXPERIMENTAL_NOT_VALIDATED",
        "profile": profile,
        "source_snapshot": SOURCE_SNAPSHOT,
        "source_relative_path": binding.source_relative_path,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "edit_plan_sha256": EDIT_PLAN_SHA256,
        "source_skill_sha256": _sha256(skill),
        "output_skill_sha256": _sha256(output_skill),
        "unchanged_prefix_sha256": _sha256(prefix),
        "unchanged_suffix_sha256": _sha256(suffix),
        "changed_sections": changed_sections,
        "section_bindings": section_bindings,
        "rule_edits": edits,
        "changed_rule_count": len(edits),
        "source_skill_rule_count": sum(line.startswith("- ") for line in skill.splitlines()),
        "output_skill_rule_count": sum(line.startswith("- ") for line in output_skill.splitlines()),
        "skill_rules_removed": 0,
        "section_headings_modified": False,
        "non_skill_envelope_modified": False,
        "trained_head_modified": False,
        "route_modified": False,
        "context_received": False,
        "context_modified": False,
        "output_contract_modified": False,
        "generation_settings_modified": False,
        "preservation_scope": (
            "adapter_only; caller must retain original head, route and full context"
        ),
        "model_used": False,
        "model_calls": 0,
        "tokenizer_used": False,
        "benchmark_data_used": False,
        "labels_used": False,
        "answers_or_actor_responses_used": False,
        "outcome_based_selection": False,
        "semantic_equivalence_established": False,
        "performance_nonregression_established": False,
        "total_inference_token_reduction_established": False,
    })
