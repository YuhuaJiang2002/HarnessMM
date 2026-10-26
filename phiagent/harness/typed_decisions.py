"""Typed probability readouts with optional bounded reasoning and no model imports."""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence


UNKNOWN = "unknown"


def alias_choice_logits(option_token_groups: Mapping[str, Sequence[int]], token_logits: Mapping[str, float]) -> dict[str, float]:
    if len(option_token_groups) < 2 or UNKNOWN not in option_token_groups:
        raise ValueError("Grouped choices require alternatives and unknown")
    flattened = []
    for name, tokens in option_token_groups.items():
        if (not isinstance(name, str) or not name or not isinstance(tokens, (list, tuple)) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Every label needs explicit native one-token spellings")
        flattened.extend(tokens)
    if len(set(flattened)) != len(flattened) or set(token_logits) != {str(token) for token in flattened}:
        raise ValueError("Native spellings must be disjoint and every logit must be observed")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in token_logits.values()):
        raise ValueError("Token logits must be finite observed numbers")
    combined = {}
    for name, tokens in option_token_groups.items():
        values = [token_logits[str(token)] for token in tokens]
        maximum = max(values)
        combined[name] = maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
    return combined


def choice_probabilities(logits: Mapping[str, float]) -> dict:
    if (len(logits) < 2 or UNKNOWN not in logits
            or any(not isinstance(name, str) or not name for name in logits)):
        raise ValueError("A choice requires named alternatives and an explicit unknown option")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in logits.values()):
        raise ValueError("Choice logits must be finite numbers, never generated text")
    maximum = max(logits.values())
    weights = {name: math.exp(value - maximum) for name, value in logits.items()}
    denominator = math.fsum(weights.values())
    probabilities = {name: weight / denominator for name, weight in weights.items()}
    selected = max(logits, key=lambda name: (logits[name], name == UNKNOWN))
    return {
        "type": "choice", "probabilities": probabilities,
        "selected": selected, "abstained": selected == UNKNOWN,
        "raw_logits": dict(logits), "probability_semantics": "softmax_over_declared_choices",
        "calibrated": False,
    }


def forward_choices(
    model: Callable, *, input_ids, attention_mask,
    option_token_ids: Sequence[Mapping[str, int]],
) -> list[dict]:
    masks = attention_mask.tolist() if hasattr(attention_mask, "tolist") else attention_mask
    if len(input_ids) != len(option_token_ids) or len(masks) != len(option_token_ids) or not masks:
        raise ValueError("Every input requires exactly one typed choice")
    prompt_counts = []
    for input_row, mask, options in zip(input_ids, masks, option_token_ids):
        if (not mask or len(input_row) != len(mask) or mask[-1] != 1
                or any(value not in (0, 1) for value in mask) or list(mask) != sorted(mask)):
            raise ValueError("Decision readout requires valid left-padded input masks")
        if (len(options) < 2 or UNKNOWN not in options
                or any(not isinstance(name, str) or not name for name in options)
                or any(type(token) is not int or token < 0 for token in options.values())
                or len(set(options.values())) != len(options)):
            raise ValueError("Each choice must use distinct native token IDs, including unknown")
        prompt_counts.append(sum(mask))
    response = model(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=False, logits_to_keep=1, return_dict=True)
    results = []
    for batch_index, options in enumerate(option_token_ids):
        selected_logits = response.logits[batch_index, -1, list(options.values())]
        values = selected_logits.detach().float().cpu().tolist()
        if len(values) != len(options):
            raise ValueError("The forward result omitted a declared choice")
        decision = choice_probabilities(dict(zip(options, values)))
        prompt_tokens = prompt_counts[batch_index]
        results.append({
            **decision,
            "option_token_ids": dict(options),
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0,
                      "total_tokens": prompt_tokens, "source": "native_input_ids"},
            "padded_input_positions": len(masks[batch_index]),
            "autoregressive_steps": 0, "generated_text": False,
        })
    return results


def reasoned_choices(
    *, input_token_ids: Sequence[Sequence[int]], option_token_ids: Sequence[Mapping[str, int]],
    advance: Callable, maximum_reasoning_tokens: int, handoff_token_ids: Sequence[int],
    stop_token_ids: Sequence[int], option_token_groups: Sequence[Mapping[str, Sequence[int]]] | None = None,
    stop_reasoning: Callable[[Sequence[int]], str | None] | None = None,
) -> list[dict]:
    if type(maximum_reasoning_tokens) is not int or not 0 <= maximum_reasoning_tokens <= 96:
        raise ValueError("Reasoning must have a fixed cap between zero and 96 tokens")
    if not input_token_ids or len(input_token_ids) != len(option_token_ids):
        raise ValueError("Every reasoning stream requires one declared choice")
    for tokens in (*input_token_ids, handoff_token_ids, stop_token_ids):
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Prompt, handoff and stop sequences require native token IDs")
    for options in option_token_ids:
        if (len(options) < 2 or UNKNOWN not in options
                or any(not isinstance(name, str) or not name for name in options)
                or any(type(token) is not int or token < 0 for token in options.values())
                or len(set(options.values())) != len(options)):
            raise ValueError("Every reasoning choice requires distinct native tokens and unknown")
    if option_token_groups is not None:
        if len(option_token_groups) != len(option_token_ids):
            raise ValueError("Each stream needs the complete grouped label vocabulary")
        for groups, options in zip(option_token_groups, option_token_ids):
            if set(groups) != set(options) or any(options[name] not in tokens for name, tokens in groups.items()):
                raise ValueError("Grouped spellings must retain all original candidate codes")
            alias_choice_logits(groups, {str(token): 0.0 for tokens in groups.values() for token in tokens})
    outputs = [[] for _ in input_token_ids]
    stopped = [False for _ in input_token_ids]
    stop_reasons = [None for _ in input_token_ids]
    submitted = [len(tokens) for tokens in input_token_ids]
    handoffs = [list(handoff_token_ids) for _ in input_token_ids]
    events = []

    def step(tokens, *, final):
        readouts = advance(tokens, final=final, use_cache=True)
        if len(readouts) != len(input_token_ids):
            raise ValueError("A cached forward omitted a reasoning stream")
        for readout in readouts:
            if readout.get("reused_prefix") is not bool(events):
                raise ValueError("Every continuation must reuse the existing prefix cache")
        events.append({"tokens_per_stream": [len(row) for row in tokens], "final_choice": final,
                       "reused_prefix": bool(events)})
        return readouts

    if maximum_reasoning_tokens == 0:
        readouts = step([list(tokens) + list(handoff_token_ids) for tokens in input_token_ids], final=True)
    else:
        readouts = step([list(tokens) for tokens in input_token_ids], final=False)
        for reasoning_index in range(maximum_reasoning_tokens):
            pending = []
            for stream_index, readout in enumerate(readouts):
                if stopped[stream_index]:
                    pending.append([])
                    continue
                token = readout.get("next_token_id")
                if type(token) is not int or token < 0:
                    raise ValueError("Reasoning must be decoded from an observed native token ID")
                outputs[stream_index].append(token)
                pending.append([token])
                reason = "stop_token" if token in stop_token_ids else stop_reasoning(outputs[stream_index]) if stop_reasoning else None
                if reason is not None and (not isinstance(reason, str) or not reason):
                    raise ValueError("A sentence stop rule must return a named reason or None")
                stop_reasons[stream_index] = reason
                stopped[stream_index] = reason is not None
            if all(stopped) or reasoning_index + 1 == maximum_reasoning_tokens:
                handoffs = [list(handoff_token_ids[1:] if output and output[-1] == handoff_token_ids[0]
                                 else handoff_token_ids) for output in outputs]
                readouts = step([tokens + handoff for tokens, handoff in zip(pending, handoffs)], final=True)
                break
            readouts = step(pending, final=False)
    results = []
    for stream_index, (readout, options) in enumerate(zip(readouts, option_token_ids)):
        logits = readout.get("choice_logits")
        groups = option_token_groups[stream_index] if option_token_groups is not None else None
        if groups is not None:
            combined = alias_choice_logits(groups, readout.get("token_logits", {}))
            if logits != combined:
                raise ValueError("The choice probabilities omit an observed native label spelling")
            logits = combined
        if not isinstance(logits, Mapping) or set(logits) != set(options):
            raise ValueError("The final probability readout must cover the original choices")
        prompt_tokens = submitted[stream_index] + len(handoffs[stream_index])
        completion_tokens = len(outputs[stream_index])
        processed = sum(event["tokens_per_stream"][stream_index] for event in events)
        if processed != prompt_tokens + completion_tokens:
            raise ValueError("The cache trace omitted or replayed input or reasoning tokens")
        results.append({
            **choice_probabilities(logits), "option_token_ids": dict(options),
            "reasoning_token_ids": outputs[stream_index], "maximum_reasoning_tokens": maximum_reasoning_tokens,
            "reasoning_stop": stop_reasons[stream_index] or ("budget" if maximum_reasoning_tokens else "disabled"),
            "handoff_token_ids": handoffs[stream_index], "cache_trace": events,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens, "source": "native_cached_token_stream"},
            "autoregressive_steps": completion_tokens, "generated_text": completion_tokens > 0,
            "replayed_input_tokens": 0, "final_answer_generated": False,
        })
        if groups is not None:
            results[-1].update({"option_token_groups": dict(groups), "raw_token_logits": dict(readout["token_logits"]),
                               "alias_normalization": "sum_probability_mass_over_disjoint_native_spellings"})
    return results
