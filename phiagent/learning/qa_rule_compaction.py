"""Prospective label-blind rule compaction; canonical task evidence is untouched."""
from copy import deepcopy
import re

ARMS = ("chisel_layout_compact", "chisel_normalization_compact", "chisel_rules_compact")

NORMALIZATION = """## Concise Answer Normalization
- Give the shortest unambiguous answer of the requested type. Omit generic descriptors, legal suffixes and expanded names unless requested or needed for identity.
- For name/word etymology, return the distinctive name or word without its generic type label.
- Preserve spelling, capitalization, punctuation and word order from the strongest title, snippet or answer field; avoid equivalent name variants.
- Prefer the supported conventional full personal name. Use a surname, given name or saint/regnal name alone only when clearly expected and unambiguous.
- Use the grammatical base form: a shared given name is singular. Common-noun categories use the singular headword unless inherently plural or explicitly given as a plural answer.
- For things replaced or substituted, give the broad headword; add modifiers only when required by the clue or answer field.
- Fill-in/definitional answers must be standalone noun phrases, not source-dependent pronouns or possessives; retain an article when needed.

"""

REASONING = """## Evidence and Clue Matching
- Match distinctive names, dates, titles, quotations, roles and relations together. Prefer direct clue matches and explicit answer fields over generic word overlap.
- Titles are evidence, not automatically answers. Extract the entity of the requested type and preserve relation direction.
- Treat every modifier, date, size, letter count and requested name/abbreviation/lyric form as a constraint. Check candidates by substitution into the complete clue.
- Distinguish a creative work from its creator, performer, character, quotation or setting using verbs and pronouns. A named or quoted anchor may identify another requested entity.
- In scraped trivia formats such as CATEGORY | clue | answer, clue. ANSWER, or right:, extract the matching answer field rather than the category or clue.
- For definitions choose the directly supported canonical term, not a derivative or near-synonym. Examples often identify their parent category, brand or entity.
- Respect inverse relationships, negation, wordplay, missing quotation continuations and dual definitions. Use text evidence for unavailable images. Corroborating snippets help select the canonical answer form.

"""


def compact_messages(messages: list[dict], arm: str) -> list[dict]:
    if arm not in ARMS or len(messages) != 2 or messages[0]["role"] != "system":
        raise ValueError("Compaction requires a declared arm and the canonical two-message request")
    result = deepcopy(messages)
    text = messages[0]["content"]
    if arm != ARMS[0]:
        text, count = re.subn(r"## Concise Answer Normalization\n.*?(?=## Context-Grounded Evidence Matching)",
                             lambda _: NORMALIZATION, text, flags=re.S)
        if count != 1:
            raise ValueError("Original normalization section is required")
    if arm == ARMS[2]:
        text, count = re.subn(r"## Context-Grounded Evidence Matching\n.*?(?=## Answer Contract)",
                             lambda _: REASONING, text, flags=re.S)
        if count != 1:
            raise ValueError("Original rule sections and answer contract are required")
    # Remove empty lines left by rule selection, retaining section names and all
    # nonempty line content in the layout-only ablation.
    result[0]["content"] = "\n".join(line for line in text.splitlines() if line.strip()) + "\n"
    assert result[1] == messages[1]
    return result
