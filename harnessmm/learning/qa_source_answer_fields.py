"""Conservative extraction from explicit question/answer records in the supplied source."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
import unicodedata


SCHEMA = "phiagent-qa-source-answer-field/v1"
DOCUMENT = re.compile(r"\[(?:DOC|SEP)\]")
OUTCOME = re.compile(r"\b(?:right|wrong)\s*:", re.IGNORECASE)
MAX_ANSWER_CHARACTERS = 128
MAX_ANSWER_WORDS = 16


def _key(text: str) -> str:
    value = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return re.sub(r"\s*([,;:!?()\[\]\"'])\s*", r"\1", value).strip()


@dataclass(frozen=True, slots=True)
class SourceAnswerSpan:
    frame: str
    start_char: int
    end_char: int
    document_start_char: int
    document_end_char: int
    support_kind: str
    text: str


def source_answer_field(*, question: str, context: str) -> dict:
    if any(not isinstance(text, str) or not text.strip() for text in (question, context)):
        raise ValueError("nonempty original question and context strings are required")
    candidates = []
    rejected = []
    if "|" not in question and not DOCUMENT.search(question):
        markers = list(DOCUMENT.finditer(context))
        bounds = [0, *[match.start() for match in markers], len(context)]
        bounds = sorted(set(bounds))
        for doc_start, doc_end in zip(bounds, bounds[1:]):
            document = context[doc_start:doc_end]
            paragraph = document.find("[PAR]")
            body_start = doc_start + paragraph + len("[PAR]") if paragraph >= 0 else doc_start
            title = document[:paragraph] if paragraph >= 0 else ""
            quiz_title = bool(re.search(r"jeopardy|j-archive", title, re.IGNORECASE))
            pipes = [match.start() for match in re.finditer(r"\|", context[body_start:doc_end])]
            pipes = [body_start + position for position in pipes]
            for left, right in zip(pipes, pipes[1:]):
                clue = context[left + 1:right]
                if _key(clue) != _key(question):
                    continue
                start = right + 1
                following_pipes = [position for position in pipes if position > right]
                pipe_end = following_pipes[0] if following_pipes else doc_end
                outcome = OUTCOME.search(context, start, pipe_end)
                header = context[max(body_start, left - 220):left]
                scored_header = bool(
                    re.search(r"\bvalue\s*:", header, re.IGNORECASE)
                    and re.search(r"\bnumber\s*:", header, re.IGNORECASE)
                )
                if outcome is not None:
                    end, support = outcome.start(), "exact_clue_with_outcome_marker"
                elif quiz_title or scored_header:
                    end = pipe_end
                    support = "exact_clue_in_named_quiz_record" if quiz_title else "exact_clue_with_record_header"
                    if end == doc_end and not context[start:end].rstrip().endswith("."):
                        rejected.append("unclosed_source_answer_field")
                        continue
                else:
                    rejected.append("unrecognized_source_record")
                    continue
                while start < end and context[start].isspace():
                    start += 1
                while end > start and context[end - 1].isspace():
                    end -= 1
                if "..." in context[start:end] or "\u2026" in context[start:end]:
                    rejected.append("invalid_or_truncated_source_answer_field")
                    continue
                if end > start and context[end - 1] == ".":
                    end -= 1
                while end > start and context[end - 1].isspace():
                    end -= 1
                answer = context[start:end]
                if (
                    not answer or len(answer) > MAX_ANSWER_CHARACTERS
                    or len(answer.split()) > MAX_ANSWER_WORDS
                    or any(marker in answer for marker in ("...", "\u2026", "<", ">", "[DOC]", "[TLE]", "[PAR]"))
                    or answer.split()[-1].casefold() in {"of", "the", "and", "or", "in", "at", "to", "by", "for", "with"}
                ):
                    rejected.append("invalid_or_truncated_source_answer_field")
                    continue
                candidates.append(SourceAnswerSpan(
                    "source_context", start, end, doc_start, doc_end, support, answer,
                ))
    keys = {_key(candidate.text) for candidate in candidates}
    if len(keys) == 1:
        selected = min(candidates, key=lambda candidate: candidate.start_char)
        answer, status, reason = selected.text, "UNIQUE_SUPPORTED_SOURCE_FIELD", "all_recognized_fields_agree"
    else:
        answer, status = None, "ABSTAIN"
        reason = "conflicting_source_answer_fields" if keys else "no_complete_unique_source_record"
    return {
        "schema_version": SCHEMA, "status": status, "reason": reason, "answer": answer,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "source_spans": [asdict(candidate) for candidate in candidates],
        "rejected_record_reasons": rejected,
        "actor_context_modified": False,
        "evaluation_answers_or_predictions_used": False,
        "source_field_is_factual_certificate": False,
        "router_model_calls": 0, "router_provider_tokens": 0,
    }
