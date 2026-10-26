"""Opt-in public-observation recheck ranking with a fully paid batch token budget."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


_STOP = frozenset("a an the of to in on at for from by with and or is was were this that these those it its his her their who what which when where".split())
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class DraftObservation:
    id: str
    group: str
    question: str
    context: str
    draft_answer: str
    draft_prompt_tokens: int
    draft_completion_tokens: int
    draft_total_tokens: int
    draft_finish_reason: str
    reference_prompt_tokens: int
    auto_prompt_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.id, self.group, self.question, self.context)):
            raise ValueError("complete public task identity, question and source are required")
        if not isinstance(self.draft_answer, str) or self.draft_finish_reason not in {"stop", "length"}:
            raise ValueError("the actual first answer and finish reason are required")
        if (any(type(value) is not int or value <= 0 for value in (
                self.draft_prompt_tokens, self.draft_total_tokens,
                self.reference_prompt_tokens, self.auto_prompt_tokens, self.max_output_tokens,
            )) or type(self.draft_completion_tokens) is not int
                or not 0 <= self.draft_completion_tokens <= self.max_output_tokens
                or self.draft_total_tokens != self.draft_prompt_tokens + self.draft_completion_tokens):
            raise ValueError("exact additive draft usage and pre-generation request budgets are required")

    @classmethod
    def from_dict(cls, value: dict) -> DraftObservation:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("runtime observations must not contain targets or undeclared fields")
        return cls(**value)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            [self.question, self.context, self.draft_answer], ensure_ascii=False,
            separators=(",", ":"),
        ).encode()).hexdigest()


def _words(text: str) -> list[str]:
    return _WORDS.findall(re.sub(r"\[(?:DOC|SEP)\]", " ", text).casefold())


def features(observation: DraftObservation, *, mode: str = "postdraft") -> dict[str, float]:
    if mode not in {"postdraft", "input_only"}:
        raise ValueError("unknown frozen feature mode")
    question = _words(observation.question)
    context = _words(observation.context)
    folded_question = observation.question.casefold()
    result = {
        "input_question_length": min(len(question) / 64, 1.0),
        "input_context_length": min(math.log1p(len(context)) / math.log1p(32768), 1.0),
        "input_documents": min(observation.context.count("[DOC]") / 32, 1.0),
        "input_numeric_clue": float(any(character.isdigit() for character in folded_question)),
        "input_relation_clue": float(bool(re.search(
            r"\b(?:wife|husband|father|mother|son|daughter|brother|sister|parent)\b", folded_question,
        ))),
        "input_work_role": float(bool(re.search(
            r"\b(?:author|writer|actor|character|film|novel|book|song|poem)\b", folded_question,
        ))),
        "input_quoted_clue": float(any(character in observation.question for character in '"\u201c\u201d')),
        "input_reference_length": min(math.log1p(observation.reference_prompt_tokens) / math.log1p(32768), 1.0),
    }
    if mode == "input_only":
        return result
    answer = _words(observation.draft_answer)
    answer_words, context_words = set(answer), set(context)
    clue = set(question) - _STOP
    source = " " + " ".join(context) + " "
    needle = " " + " ".join(answer) + " "
    occurrences = source.count(needle) if answer else 0
    support, start = 0.0, 0
    for _ in range(min(occurrences, 32)):
        position = source.find(needle, start)
        if position < 0:
            break
        nearby = set(_words(source[max(0, position - 320):position + len(needle) + 320]))
        support = max(support, len(clue & nearby) / len(clue) if clue else 0.0)
        start = position + len(needle) - 1
    result.update({
        "draft_empty": float(not answer),
        "draft_words": min(len(answer) / 12, 1.0),
        "draft_in_context": float(occurrences > 0),
        "draft_in_question": float(bool(answer) and needle in " " + " ".join(question) + " "),
        "draft_source_coverage": len(answer_words & context_words) / len(answer_words) if answer_words else 0.0,
        "draft_question_overlap": len(answer_words & set(question)) / len(answer_words) if answer_words else 0.0,
        "draft_corroboration": min(math.log1p(occurrences) / math.log1p(32), 1.0),
        "draft_local_clue_support": support,
        "draft_numeric": float(bool(answer) and all(word.isdecimal() for word in answer)),
        "draft_output_fraction": observation.draft_completion_tokens / observation.max_output_tokens,
        "draft_length_stop": float(observation.draft_finish_reason == "length"),
    })
    return result


def validate_outcome(record: dict) -> None:
    if (type(record["exact_match"]) is not int or record["exact_match"] not in (0, 1)
            or type(record["f1"]) not in (int, float) or not math.isfinite(record["f1"])
            or not 0 <= record["f1"] <= 1
            or (record["exact_match"] == 1 and record["f1"] != 1)
            or any(type(record[key]) is not int or record[key] < 0
                   for key in ("tokens", "prompt_tokens", "completion_tokens"))
            or record["tokens"] != record["prompt_tokens"] + record["completion_tokens"]):
        raise ValueError("official outcomes and exact additive provider token receipts are required")


def fit_gain_head(
    observations: Sequence[DraftObservation], targets: Mapping[str, dict], *,
    mode: str, draft: str, reference: str, ridge: float, f1_weight: float,
) -> dict[str, Any]:
    if (not observations or len({row.id for row in observations}) != len(observations)
            or set(targets) != {row.id for row in observations}
            or type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0
            or type(f1_weight) not in (int, float) or not math.isfinite(f1_weight)
            or not 0 <= f1_weight <= 1):
        raise ValueError("a complete fit cohort and finite frozen regression settings are required")
    names = sorted(features(observations[0], mode=mode))
    examples = []
    for observation in observations:
        target = targets[observation.id]
        if target["group"] != observation.group:
            raise ValueError("fit task and question-group identities differ")
        first, second = target["methods"][draft], target["methods"][reference]
        for record in (first, second):
            validate_outcome(record)
        gain = second["exact_match"] - first["exact_match"] + f1_weight * (second["f1"] - first["f1"])
        vector = features(observation, mode=mode)
        examples.append((observation.id, [vector[name] for name in names], gain, gain < 0, 0.0))
    from phiagent.learning.residual_meta_training import _fit_ridge

    intercept, weights = _fit_ridge(examples, ridge=ridge)
    return {
        "schema_version": "phiagent-qa-postdraft-gain-head/v1", "mode": mode,
        "feature_names": names, "intercept": intercept, "weights": weights,
        "ridge": ridge, "f1_gain_weight": f1_weight,
        "fit_tasks": len(observations), "fit_groups": len({row.group for row in observations}),
        "target": "incremental recheck EM + declared weighted incremental F1, not task difficulty",
        "ids_groups_or_target_fields_used_as_features": False,
    }


def predict_gain(observation: DraftObservation, head: dict) -> float:
    if head["schema_version"] == "phiagent-qa-postdraft-tree-head/v1":
        from phiagent.learning.qa_postdraft_tree import predict_tree

        return predict_tree(observation, head)
    values = features(observation, mode=head["mode"])
    if (head["schema_version"] != "phiagent-qa-postdraft-gain-head/v1"
            or head["feature_names"] != sorted(values)
            or len(head["weights"]) != len(values)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in [head["intercept"], *head["weights"]])):
        raise ValueError("the frozen head must match the public feature schema exactly")
    return head["intercept"] + math.fsum(
        weight * values[name] for weight, name in zip(head["weights"], head["feature_names"], strict=True)
    )


def plan_batch(
    observations: Sequence[DraftObservation], *, head: dict | None = None,
    threshold: float = 0.0, mode: str = "learned", seed: int = 0,
) -> dict[str, Any]:
    if (not observations or len({row.id for row in observations}) != len(observations)
            or mode not in {"learned", "random", "source_support"}
            or (mode == "learned" and head is None)
            or type(seed) is not int or type(threshold) not in (int, float)
            or not math.isfinite(threshold)):
        raise ValueError("a complete public batch and explicit finite routing settings are required")
    first_cost = sum(row.draft_total_tokens for row in observations)
    budget = sum(row.auto_prompt_tokens for row in observations) - 1
    priorities = {}
    for row in observations:
        if mode == "learned":
            value = predict_gain(row, head)
        elif mode == "source_support":
            signals = features(row)
            value = signals["draft_empty"] + signals["draft_in_question"] + 1 - signals["draft_local_clue_support"]
        else:
            value = int(hashlib.sha256(f"{seed}:{row.fingerprint()}".encode()).hexdigest()[:13], 16) / 16**13
        priorities[row.id] = value
    ordered = sorted(observations, key=lambda row: (
        -priorities[row.id] / (row.reference_prompt_tokens + row.max_output_tokens)
        if mode != "random" else -priorities[row.id],
        row.fingerprint(), row.id,
    ))
    routes, spent = {row.id: "keep_draft" for row in observations}, first_cost
    for row in ordered:
        reservation = row.reference_prompt_tokens + row.max_output_tokens
        if priorities[row.id] > threshold and spent + reservation <= budget:
            routes[row.id] = "recheck"
            spent += reservation
    return {
        "schema_version": "phiagent-qa-postdraft-batch-plan/v1", "mode": mode, "seed": seed,
        "threshold": threshold, "routes": routes, "predicted_gains_or_priorities": priorities,
        "tasks": len(observations), "question_groups": len({row.group for row in observations}),
        "first_pass_tokens": first_cost, "auto_prompt_budget": budget,
        "reserved_total_tokens": spent, "recheck_count": sum(value == "recheck" for value in routes.values()),
        "first_pass_fits_budget": first_cost <= budget,
        "status": "BOUND_PLAN_READY" if first_cost <= budget else "FIRST_PASS_EXCEEDS_BOUND_NO_RECHECK",
        "budget_source": "Known AutoSaddler input tokens only; no future Auto completion usage",
        "runtime_targets_or_strong_answers_used": False,
        "lexical_signals_are_quality_certificates": False,
        "batch_inference_not_a_per_question_cost_guarantee": True,
    }


def replay_plan(
    observations: Sequence[DraftObservation], targets: Mapping[str, dict], plan: dict, *,
    draft: str, reference: str,
) -> dict[str, Any]:
    ids = {row.id for row in observations}
    if len(ids) != len(observations) or set(targets) != ids or set(plan["routes"]) != ids:
        raise ValueError("replay requires every task and one frozen decision per task")
    successes, f1_values, tokens, calls = 0, [], 0, 0
    prompt_tokens = completion_tokens = reservations = first_cost = 0
    for row in observations:
        first, second = targets[row.id]["methods"][draft], targets[row.id]["methods"][reference]
        validate_outcome(first)
        validate_outcome(second)
        route = plan["routes"][row.id]
        if (route not in {"keep_draft", "recheck"} or targets[row.id]["group"] != row.group
                or first["tokens"] != row.draft_total_tokens
                or second["prompt_tokens"] != row.reference_prompt_tokens
                or second["completion_tokens"] > row.max_output_tokens):
            raise ValueError("frozen route, public observation and private score/cost receipt differ")
        chosen = second if route == "recheck" else first
        successes += chosen["exact_match"]
        f1_values.append(chosen["f1"])
        tokens += first["tokens"] + (second["tokens"] if route == "recheck" else 0)
        prompt_tokens += first["prompt_tokens"] + (second["prompt_tokens"] if route == "recheck" else 0)
        completion_tokens += first["completion_tokens"] + (second["completion_tokens"] if route == "recheck" else 0)
        first_cost += row.draft_total_tokens
        reservations += row.reference_prompt_tokens + row.max_output_tokens if route == "recheck" else 0
        calls += 1 + int(route == "recheck")
    expected_budget = sum(row.auto_prompt_tokens for row in observations) - 1
    if (plan["first_pass_tokens"] != first_cost
            or plan["reserved_total_tokens"] != first_cost + reservations
            or plan["recheck_count"] != calls - len(observations)
            or plan["auto_prompt_budget"] != expected_budget
            or plan["first_pass_fits_budget"] is not (first_cost <= expected_budget)
            or tokens > plan["reserved_total_tokens"]):
        raise ValueError("actual paid replay costs exceed the pre-response reservation")
    return {
        "tasks": len(observations), "successes": successes,
        "exact_match": 100 * successes / len(observations),
        "f1": 100 * math.fsum(f1_values) / len(observations),
        "total_tokens": tokens, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "model_calls_if_executed": calls,
        "rechecks": plan["recheck_count"], "first_pass_tokens": plan["first_pass_tokens"],
        "reserved_total_tokens": plan["reserved_total_tokens"],
        "budget_feasible": plan["first_pass_fits_budget"],
        "new_model_calls": 0, "new_evaluator_calls": 0,
        "scope": "Learned-policy replay of audited archived answers; not a new actor run",
    }


def public_dict(observation: DraftObservation) -> dict:
    return asdict(observation)
