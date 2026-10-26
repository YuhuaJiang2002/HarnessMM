"""Provisional, source-bound QA context selection; no factual-support certification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import math
import re

from phiagent.learning.public_evidence_retention import text_sha256
from phiagent.learning.sparse_retrieval import WORD_PATTERN


METHODS = frozenset({
    "full", "head", "bm25", "mmr", "coverage", "bm25_closure", "coverage_closure",
})
GAP = "\n[...]\n"
_SOURCE_BOUNDARY = re.compile(r"\[(?:DOC|SEP)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_MARKERS = re.compile(r"^(?:\s*\[[A-Z]+\])*\s*")
_ANAPHOR = re.compile(
    r"""^(?:["'\u201c\u2018]|(?:he|she|it|they|his|her|their|its|this|these|those|such)\b)""",
    re.IGNORECASE,
)
_STOP = frozenset(
    "a an the and or but of to in on at for from by with as is are was were be been "
    "being do does did have has had who whom whose what which when where why how "
    "this that these those it its he she they his her their you your we our i me "
    "my not no so than then there here into about would could should can will may "
    "might must one also".split()
)


@dataclass(frozen=True, slots=True)
class ContextUnit:
    start_char: int
    end_char: int
    source_group: int
    continuation: bool


@dataclass(frozen=True, slots=True)
class ContextSelection:
    context: str
    receipt: dict[str, object]


def answer_candidates(question: str, context: str, *, maximum_candidates: int = 48) -> dict:
    if not isinstance(question, str) or not question.strip() or not isinstance(context, str) or not context.strip():
        raise ValueError("Answer candidates require a question and its nonempty original source")
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 52:
        raise ValueError("The declared choice vocabulary supports 1 to 52 source candidates")
    words = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*")
    query = set(words.findall(question.casefold())) - _STOP
    boundaries = sorted({0, *(match.start() for match in re.finditer(r"\[DOC\]", context))})
    documents = list(zip(boundaries, boundaries[1:] + [len(context)]))
    document_terms = [set(words.findall(context[start:end].casefold())) for start, end in documents]
    frequency = Counter(term for terms in document_terms for term in sorted(terms))
    relevance = [math.fsum(math.log1p(len(documents) / (1 + frequency[term])) for term in sorted(query & terms))
                 for terms in document_terms]
    scale = max(relevance, default=0.0) or 1.0
    candidates: dict[str, dict] = {}

    def add(start: int, end: int, group: int, kind: str, bonus: float) -> None:
        while start < end and context[start] in " \t\r\n\"'.,;:":
            start += 1
        while end > start and context[end - 1] in " \t\r\n\"'.,;:":
            end -= 1
        text = context[start:end]
        terms = words.findall(text.casefold())
        if (not terms or len(terms) > 10 or len(text) > 100 or "[" in text or "]" in text
                or all(term in _STOP for term in terms)):
            return
        key = " ".join(text.casefold().split())
        query_overlap = sum(term in query for term in terms) / len(terms)
        score = 3 * relevance[group] / scale + bonus - 2 * query_overlap
        existing = candidates.get(key)
        occurrence = {"start_char": start, "end_char": end, "source_group": group}
        if existing is None:
            candidates[key] = {"text": text, "kind": kind, "score": score, "occurrences": [occurrence],
                               "start_char": start, "end_char": end}
        else:
            if occurrence not in existing["occurrences"]:
                existing["occurrences"].append(occurrence)
            if score > existing["score"]:
                existing.update({"text": text, "kind": kind, "score": score,
                                 "start_char": start, "end_char": end})

    for group, (start, end) in enumerate(documents):
        document = context[start:end]
        title = re.search(r"\[TLE\]\s*(.*?)(?=\s*\[PAR\]|$)", document, re.DOTALL)
        if title:
            title_start, title_end = start + title.start(1), start + title.end(1)
            add(title_start, title_end, group, "title", 1.1)
            delimiter = re.search(r"\s+[|\u2013\u2014-]\s+|\s*\([^)]*\)|\.{3}", title.group(1))
            if delimiter:
                add(title_start, title_start + delimiter.start(), group, "title_prefix", 1.6)
        for match in re.finditer(r"\b(?:answer|correct response|solution)\s*(?::|is\b)\s*([^\n.;!?\[\]]{1,100})", document, re.IGNORECASE):
            add(start + match.start(1), start + match.end(1), group, "answer_field", 2.4)
        for match in re.finditer(r'"([^"\n]{2,100})"', document):
            add(start + match.start(1), start + match.end(1), group, "quotation", 0.7)
        for match in re.finditer(r"\b[A-Z][\w'-]*(?:\s+(?:(?:of|the|de|van|von|and)\s+)?[A-Z][\w'-]*){0,5}", document):
            add(start + match.start(), start + match.end(), group, "capitalized_phrase", 0.8)
        tokens = list(words.finditer(document))
        for token_index, first in enumerate(tokens):
            if first.group().casefold() in _STOP:
                continue
            for width in range(1, min(4, len(tokens) - token_index) + 1):
                last = tokens[token_index + width - 1]
                if last.group().casefold() in _STOP:
                    continue
                if any(not re.fullmatch(r"[\s-]*", document[left.end():right.start()])
                       for left, right in zip(tokens[token_index:token_index + width - 1],
                                              tokens[token_index + 1:token_index + width])):
                    continue
                if first.start() > 0 and document[first.start() - 1] == "[":
                    continue
                add(start + first.start(), start + last.end(), group, "word_span", 0.0)
    for row in candidates.values():
        groups = {occurrence["source_group"] for occurrence in row["occurrences"]}
        row["score"] += math.log1p(len(groups))
        row["source_text_sha256"] = text_sha256(row["text"])
        row["id"] = f"span:{row['start_char']}:{row['end_char']}:{row['source_text_sha256'][:12]}"
    ranked = sorted(candidates.values(), key=lambda row: (-row["score"], row["start_char"], row["end_char"]))
    return {
        "schema_version": "phiagent-source-answer-choices/v1",
        "question_sha256": text_sha256(question), "source_context_sha256": text_sha256(context),
        "candidates": ranked[:maximum_candidates], "unbounded_candidates": len(ranked),
        "maximum_candidates": maximum_candidates, "answer_or_model_response_used": False,
        "quality_equivalence_assumed": False, "neural_extractor_calls": 0,
    }


def pack_duplicate_documents(context: str) -> ContextSelection:
    if not isinstance(context, str):
        raise TypeError("context must be text")
    starts = sorted({0, *(match.start() for match in re.finditer(r"\[DOC\]", context))})
    spans = []
    pieces = []
    first_occurrences: dict[str, int] = {}
    for index, (start, end) in enumerate(zip(starts, starts[1:] + [len(context)])):
        text = context[start:end]
        reference = first_occurrences.get(text) if text.startswith("[DOC]") else None
        replacement = f"[DOC][duplicate of document {reference + 1}]" if reference is not None else text
        if len(replacement) >= len(text):
            reference, replacement = None, text
        if text.startswith("[DOC]"):
            first_occurrences.setdefault(text, index)
        pieces.append(replacement)
        spans.append({
            "document_index": index + 1,
            "start_char": start,
            "end_char": end,
            "source_text_sha256": text_sha256(text),
            "repeat_of": reference + 1 if reference is not None else None,
        })
    rendered = "".join(pieces)
    receipt = {
        "schema_version": "phiagent-exact-document-pack/v1",
        "source_context_sha256": text_sha256(context),
        "rendered_context_sha256": text_sha256(rendered),
        "spans": spans,
        "duplicate_documents": sum(span["repeat_of"] is not None for span in spans),
        "unique_evidence_removed": False,
    }
    return ContextSelection(rendered, receipt)


def verify_duplicate_documents(context: str, packed: ContextSelection) -> None:
    rebuilt = pack_duplicate_documents(context)
    if packed != rebuilt:
        raise ValueError("duplicate-document projection differs from exact source evidence")


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _counter(counter: Callable[[str], int]) -> Callable[[str], int]:
    @lru_cache(maxsize=4096)
    def count(text: str) -> int:
        value = counter(text)
        if type(value) is not int or value < 0:
            raise ValueError("the exact tokenizer counter must return a nonnegative integer")
        return value
    return count


def context_units(
    context: str, *, count_tokens: Callable[[str], int], maximum_unit_tokens: int,
) -> tuple[ContextUnit, ...]:
    """Partition exact source characters without treating separators as verified URLs."""
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a nonempty source string")
    _positive_integer(maximum_unit_tokens, "maximum_unit_tokens")
    count = _counter(count_tokens)
    source_starts = sorted({0, *(match.start() for match in _SOURCE_BOUNDARY.finditer(context))})
    units = []
    for group, (left, right) in enumerate(zip(source_starts, source_starts[1:] + [len(context)])):
        endings = [
            left + match.end()
            for match in _SENTENCE_BOUNDARY.finditer(context[left:right])
        ]
        endings = sorted({*endings, right})
        sentence_start = left
        for sentence_end in endings:
            position, continuation = sentence_start, False
            while position < sentence_end:
                end = sentence_end
                if count(context[position:end]) > maximum_unit_tokens:
                    boundaries = [
                        position + match.end()
                        for match in re.finditer(r"\S+\s*", context[position:sentence_end])
                    ]
                    low, high, best = 0, len(boundaries) - 1, None
                    while low <= high:
                        middle = (low + high) // 2
                        if count(context[position:boundaries[middle]]) <= maximum_unit_tokens:
                            best, low = boundaries[middle], middle + 1
                        else:
                            high = middle - 1
                    if best is None:
                        low, high, best = position + 1, sentence_end, position
                        while low <= high:
                            middle = (low + high) // 2
                            if count(context[position:middle]) <= maximum_unit_tokens:
                                best, low = middle, middle + 1
                            else:
                                high = middle - 1
                    end = best
                if end <= position or count(context[position:end]) > maximum_unit_tokens:
                    raise ValueError("a nonempty source unit cannot fit the declared token budget")
                units.append(ContextUnit(position, end, group, continuation))
                position, continuation = end, True
            sentence_start = sentence_end
    if (
        not units or units[0].start_char != 0 or units[-1].end_char != len(context)
        or any(left.end_char != right.start_char for left, right in zip(units, units[1:]))
    ):
        raise RuntimeError("source partition is not exact and complete")
    return tuple(units)


def _ranges(units: tuple[ContextUnit, ...], selected: set[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for index in sorted(selected):
        unit = units[index]
        if result and result[-1][1] == unit.start_char:
            result[-1] = (result[-1][0], unit.end_char)
        else:
            result.append((unit.start_char, unit.end_char))
    return result


def reconstruct_context(context: str, receipt: dict[str, object]) -> str:
    """Independently check source offsets/hashes and reconstruct only copied text."""
    if receipt.get("source_context_sha256") != text_sha256(context):
        raise ValueError("context receipt is bound to another source")
    spans = receipt.get("spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("a receipt must contain nonempty source spans")
    pieces, previous_end = [], -1
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("source spans must be objects")
        start, end = span.get("start_char"), span.get("end_char")
        if (
            type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(context) or start <= previous_end
        ):
            raise ValueError("source spans must be disjoint, ordered and maximally merged")
        piece = context[start:end]
        if text_sha256(piece) != span.get("source_text_sha256"):
            raise ValueError("source span content hash differs")
        pieces.append(piece)
        previous_end = end
    result = GAP.join(pieces)
    if receipt.get("rendered_context_sha256") != text_sha256(result):
        raise ValueError("rendered context contains an unbound edit")
    return result


def select_context_fraction(
    question: str, context: str, *, method: str, count_tokens: Callable[[str], int], fraction: float,
) -> ContextSelection:
    if method not in METHODS - {"full"} or isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("A source fraction in (0, 1] and an explicit selection method are required")
    source_tokens = count_tokens(context)
    _positive_integer(source_tokens, "source_context_tokens")
    budget = max(1, math.floor(source_tokens * fraction))
    fallback = None
    try:
        selected = select_context(question, context, method=method, count_tokens=count_tokens,
                                  token_budget=budget, maximum_unit_tokens=min(96, budget))
    except ValueError as error:
        if str(error) != "no source unit fits the declared budget; do not silently emit empty evidence":
            raise
        selected = select_context(question, context, method="full", count_tokens=count_tokens,
                                  token_budget=max(96, source_tokens))
        fallback = "no_nonempty_source_unit_fits_fraction"
    return ContextSelection(selected.context, {
        **selected.receipt, "source_fraction": fraction, "requested_method": method,
        "fraction_token_budget": budget, "fraction_fallback": fallback,
        "depends_on_system_prompt": False,
    })


def select_context_for_request_budget(
    question: str, context: str, *, method: str, count_tokens: Callable[[str], int],
    count_request: Callable[[str], int], maximum_request_tokens: int,
) -> ContextSelection:
    if method not in METHODS - {"full"}:
        raise ValueError("An explicit evidence selection method is required")
    _positive_integer(maximum_request_tokens, "maximum_request_tokens")
    original_tokens, empty_tokens = count_request(context), count_request("")
    for value in (original_tokens, empty_tokens):
        _positive_integer(value, "complete_request_tokens")
    source_tokens = count_tokens(context)
    _positive_integer(source_tokens, "source_context_tokens")
    fallback = None
    selection = select_context(
        question, context, method="full", count_tokens=count_tokens,
        token_budget=max(source_tokens, 96),
    )
    if original_tokens > maximum_request_tokens:
        budget = maximum_request_tokens - empty_tokens - 8
        if budget < 1:
            fallback = "system_and_question_exceed_request_budget"
        else:
            for _attempt in range(3):
                try:
                    proposal = select_context(
                        question, context, method=method, count_tokens=count_tokens,
                        token_budget=budget, maximum_unit_tokens=min(96, budget),
                    )
                except ValueError:
                    fallback = "no_verifiable_nonempty_evidence_fits"
                    break
                observed = count_request(proposal.context)
                _positive_integer(observed, "selected_request_tokens")
                if observed <= maximum_request_tokens and observed < original_tokens:
                    selection = proposal
                    break
                budget -= max(8, observed - maximum_request_tokens)
                if budget < 1:
                    fallback = "native_request_budget_not_reached"
                    break
            else:
                fallback = "native_request_budget_not_reached"
    receipt = {
        **selection.receipt,
        "request_budget": {
            "maximum_tokens": maximum_request_tokens,
            "original_tokens": original_tokens,
            "selected_tokens": count_request(selection.context),
            "fallback": fallback,
            "unique_evidence_may_be_omitted": selection.context != context,
            "quality_equivalence_assumed": False,
        },
    }
    if reconstruct_context(context, receipt) != selection.context:
        raise ValueError("Budgeted evidence cannot be reconstructed from its source")
    return ContextSelection(selection.context, receipt)


def select_context(
    question: str,
    context: str,
    *,
    method: str,
    count_tokens: Callable[[str], int],
    token_budget: int,
    maximum_unit_tokens: int = 96,
    mmr_relevance_weight: float = 0.7,
    coverage_weight: float = 1.0,
    cooccurrence_weight: float = 0.5,
) -> ContextSelection:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if not isinstance(context, str) or not context.strip() or method not in METHODS:
        raise ValueError("nonempty original context and a declared method are required")
    _positive_integer(token_budget, "token_budget")
    _positive_integer(maximum_unit_tokens, "maximum_unit_tokens")
    if maximum_unit_tokens > token_budget:
        raise ValueError("unit budget cannot exceed the context budget")
    for value in (mmr_relevance_weight, coverage_weight, cooccurrence_weight):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("selection weights must be finite and nonnegative")
    if mmr_relevance_weight > 1:
        raise ValueError("MMR relevance weight must be at most one")
    count = _counter(count_tokens)
    before = count(context)
    units = context_units(context, count_tokens=count, maximum_unit_tokens=maximum_unit_tokens)
    texts = [context[unit.start_char:unit.end_char] for unit in units]
    token_lists = [WORD_PATTERN.findall(text.casefold()) for text in texts]
    term_sets = [set(tokens) for tokens in token_lists]
    query = set(WORD_PATTERN.findall(question.casefold())) - _STOP
    frequency = Counter(term for terms in term_sets for term in terms)
    idf = {
        term: math.log1p((len(units) - frequency[term] + 0.5) / (frequency[term] + 0.5))
        for term in sorted(query)
    }
    query_mass = math.fsum(idf.values())
    average_length = max(1.0, sum(map(len, token_lists)) / len(units))
    relevance = []
    for tokens in token_lists:
        counts = Counter(tokens)
        normalization = 1.2 * (0.25 + 0.75 * len(tokens) / average_length)
        relevance.append(math.fsum(
            idf[term] * counts[term] * 2.2 / (counts[term] + normalization)
            for term in sorted(query) if counts[term]
        ))
    scale = max(max(relevance), 1e-12)
    normalized = [value / scale for value in relevance]
    closure_enabled = method.endswith("_closure")
    coverage_enabled = method.startswith("coverage")

    def closure(index: int) -> set[int]:
        result = {index}
        while closure_enabled and index > 0:
            current, previous = units[index], units[index - 1]
            leading = _LEADING_MARKERS.sub("", texts[index])
            # This is a source-local heuristic, not a semantic coreference verifier.
            needs_previous = current.continuation or _ANAPHOR.search(leading) is not None
            if not needs_previous or current.source_group != previous.source_group:
                break
            index -= 1
            result.add(index)
        return result

    def render(selected: set[int]) -> str:
        return GAP.join(context[start:end] for start, end in _ranges(units, selected))

    selected: set[int] = set()
    closures: list[dict[str, object]] = []
    if method == "full" or before <= token_budget:
        selected.update(range(len(units)))
    else:
        while len(selected) < len(units):
            covered = set().union(*(term_sets[index] for index in selected)) if selected else set()
            priorities = []
            for index in range(len(units)):
                if index in selected:
                    continue
                additions = closure(index) - selected
                gain = normalized[index]
                if method == "head":
                    gain = -float(index)
                elif method == "mmr":
                    similarity = max((
                        len(term_sets[index] & term_sets[other])
                        / max(1, len(term_sets[index] | term_sets[other]))
                        for other in selected
                    ), default=0.0)
                    gain = mmr_relevance_weight * gain - (1 - mmr_relevance_weight) * similarity
                elif coverage_enabled:
                    terms = set().union(*(term_sets[other] for other in additions))
                    new_mass = math.fsum(idf[term] for term in sorted((terms & query) - covered))
                    joint_mass = math.fsum(idf[term] for term in sorted(term_sets[index] & query))
                    denominator = max(query_mass, 1e-12)
                    gain += coverage_weight * new_mass / denominator
                    gain += cooccurrence_weight * (joint_mass / denominator) ** 2
                priorities.append((-gain, index, additions))
            accepted = False
            for _negative_gain, index, additions in sorted(priorities, key=lambda row: row[:2]):
                proposal = selected | additions
                if count(render(proposal)) <= token_budget:
                    selected = proposal
                    if len(additions) > 1:
                        closures.append({"center_unit": index, "added_unit_ids": sorted(additions)})
                    accepted = True
                    break
                if method == "head":
                    break
            if not accepted:
                break
    if not selected:
        raise ValueError("no source unit fits the declared budget; do not silently emit empty evidence")
    result = render(selected)
    after = count(result)
    if method != "full" and after > token_budget:
        raise RuntimeError("rendered context exceeds the exact tokenizer budget")
    spans = [
        {"start_char": start, "end_char": end, "source_text_sha256": text_sha256(context[start:end])}
        for start, end in _ranges(units, selected)
    ]
    covered = set().union(*(term_sets[index] for index in selected))
    receipt: dict[str, object] = {
        "schema_version": "phiagent-qa-context-selection/v1",
        "method": method,
        "question_sha256": text_sha256(question),
        "source_context_sha256": text_sha256(context),
        "rendered_context_sha256": text_sha256(result),
        "offset_unit": "Python Unicode character index; end exclusive",
        "spans": spans,
        "source_unit_count": len(units),
        "selected_unit_ids": sorted(selected),
        "source_group_semantics": "Publisher DOC/SEP delimiter groups, not verified web-source identities",
        "source_context_tokens": before,
        "rendered_context_tokens": after,
        "token_budget": token_budget,
        "identity_transform": result == context,
        "closure_enabled": closure_enabled,
        "closure_additions": closures,
        "lexical_query_terms": sorted(query),
        "lexical_query_coverage": math.fsum(
            idf[term] for term in sorted(covered & query)
        ) / max(query_mass, 1e-12),
        "numeric_reduction": "sorted_terms_math_fsum",
        "ranking_reason": "query_lexical_signals" if any(relevance) else "zero_overlap_source_order_ties",
        "factual_truth_or_answer_sufficiency_certified": False,
        "answer_or_model_response_used": False,
    }
    if reconstruct_context(context, receipt) != result:
        raise RuntimeError("source reconstruction differs from the selected context")
    return ContextSelection(result, receipt)
