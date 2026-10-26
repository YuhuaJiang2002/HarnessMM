"""Train-only, source-span evidence ranking; no model or dataset imports."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import random
import re


STOP = set("a an the of in on at to for and or is was are were this that it its with by from what which who how".split())
SYSTEM = (
    "Answer the question using the supplied source excerpts. Match every clue, "
    "including the requested entity, relation, date and location. Prefer explicit "
    "answer fields and directly matching evidence over incidental mentions. "
    "Use the shortest unambiguous supported name or dictionary headword, without "
    "extra descriptors. Return <answer>the answer</answer>."
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def words(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())


def partition(items: list[dict], seed: int) -> dict:
    """Union identical questions or contexts before an outcome-blind group split."""
    parents = {item["id"]: item["id"] for item in items}
    if len(parents) != len(items):
        raise ValueError("duplicate task IDs")

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen = {}
    for item in sorted(items, key=lambda row: row["id"]):
        for key in ("q:" + " ".join(words(item["question"])), "c:" + digest(item["context"])):
            if key in seen:
                a, b = root(item["id"]), root(seen[key])
                parents[max(a, b)] = min(a, b)
            seen[key] = item["id"]
    groups = {}
    for identity in parents:
        groups.setdefault(root(identity), []).append(identity)
    ordered = sorted(groups, key=lambda group: digest(f"{seed}:{group}"))
    n = len(ordered) // 5
    if n < 1:
        raise ValueError("at least five independent groups required")
    result = {"seed": seed, "ratio": [20, 20, 60],
              "rounding": "floor(groups/5) to train and validation; remainder to test",
              "prior_exposure": "All 16980 public dev questions were previously evaluated; not unseen.",
              "groups": {key: sorted(value) for key, value in groups.items()}}
    for name, selected in (("train", ordered[:n]), ("validation", ordered[n:2*n]), ("test", ordered[2*n:])):
        result[name] = {"groups": selected, "ids": sorted(identity for key in selected for identity in groups[key])}
    return result


def units(context: str) -> list[tuple[int, int]]:
    """Overlapping whole-word windows preserve exact source characters."""
    spans = list(re.finditer(r"\S+", context))
    result = []
    for begin in range(0, len(spans), 56):
        result.append((spans[begin].start(), spans[min(begin + 80, len(spans)) - 1].end()))
        if begin + 80 >= len(spans):
            break
    return result


def features(question: str, text: str, position: float) -> list[float]:
    q, s = words(question), words(text)
    query, source = set(q) - STOP, set(s)
    overlap = query & source
    pairs = set(zip(s, s[1:]))
    q_pairs = set(zip(q, q[1:]))
    nums = set(re.findall(r"\d+", question))
    return [
        1., len(overlap) / max(len(query), 1),
        len(overlap) / math.sqrt(max(len(source), 1)),
        len(q_pairs & pairs) / max(len(q_pairs), 1),
        float(bool(nums) and nums <= set(re.findall(r"\d+", text))),
        float(bool(re.search(r"\b(?:answer|solution)\b", text, re.I))),
        float("[TLE]" in text), float("[PAR]" in text),
        1. - position, float("year" in query and bool(re.search(r"\b\d{4}\b", text))),
        min(len(s), 100) / 100.,
        float(len(overlap) == len(query) and bool(query)),
    ]


def fit(train: list[dict], labels: dict[str, list[str]], epochs: int = 5) -> dict:
    """Learn pairwise relevance from TRAIN answer-bearing source windows only."""
    if set(labels) != {item["id"] for item in train}:
        raise ValueError("fit labels must be exactly the supplied training IDs")
    pairs = []
    for item in train:
        positive, negative = [], []
        needles = [" ".join(words(answer)) for answer in labels[item["id"]] if words(answer)]
        for begin, end in units(item["context"]):
            text = item["context"][begin:end]
            normalized = " " + " ".join(words(text)) + " "
            target = positive if any(" " + answer + " " in normalized for answer in needles) else negative
            target.append(features(item["question"], text, begin / max(len(item["context"]), 1)))
        for p in positive[:4]:
            for n in sorted(negative, key=lambda vector: -vector[1])[:4]:
                pairs.append([a-b for a, b in zip(p, n)])
    if not pairs:
        raise ValueError("no train supervision pairs")
    weights = [0.] * 12
    for _ in range(epochs):
        for vector in pairs:
            score = sum(a*b for a, b in zip(weights, vector))
            error = 1. / (1. + math.exp(max(-30., min(30., score))))
            weights = [w + .025 * (error*x - .0005*w) for w, x in zip(weights, vector)]
    return {"weights": weights, "pairs": len(pairs), "epochs": epochs,
            "training_ids": sorted(labels), "labels_from_validation_or_test": False,
            "mechanism": "train_answer_supervised_pairwise_source_window_ranking",
            "optimizer_model_calls": 0}


def messages(item: dict, context: str, system: str = SYSTEM) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"## Context\n{context}\n\n## Question\n{item['question']}"}]


def select(item: dict, model: dict, input_cap: int, count) -> tuple[list[dict], dict]:
    context = item["context"]
    scored = []
    for start, end in units(context):
        vector = features(item["question"], context[start:end], start / max(len(context), 1))
        scored.append((sum(w*x for w, x in zip(model["weights"], vector)), start, end))
    kept = []

    def render(spans):
        return "\n...\n".join(context[a:b] for a, b in sorted(spans))

    for _, start, end in sorted(scored, key=lambda row: (-row[0], row[1])):
        # Avoid paying twice for strongly overlapping neighboring windows.
        if any(max(0, min(end, b)-max(start, a)) > .35*(end-start) for a, b in kept):
            continue
        if count(messages(item, render(kept + [(start, end)]))) <= input_cap:
            kept.append((start, end))
    if not kept:
        raise ValueError("question plus one intact source window exceeds frozen input cap")
    request = messages(item, render(kept))
    return request, {"spans": sorted(kept), "context_sha256": digest(context),
                     "source_only": True, "input_cap": input_cap,
                     "actual_input_tokens": count(request), "selection_model_calls": 0}


def summary(rows: list[dict]) -> dict:
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique complete tasks required")
    n = len(rows)
    return {"tasks": n, "em_percent": 100*sum(row["em"] for row in rows)/n,
            "f1_percent": 100*sum(row["f1"] for row in rows)/n,
            "mean_actor_tokens": sum(row["usage"]["total_tokens"] for row in rows)/n,
            "actor_input_tokens": sum(row["usage"]["prompt_tokens"] for row in rows),
            "actor_output_tokens": sum(row["usage"]["completion_tokens"] for row in rows)}


def paired(candidate: list[dict], reference: list[dict], groups: dict, seed=20260928, draws=1000) -> dict:
    a, b = {r["id"]: r for r in candidate}, {r["id"]: r for r in reference}
    if set(a) != set(b) or not a:
        raise ValueError("paired comparison requires identical complete task IDs")
    blocks = [[key for key in identities if key in a] for identities in groups.values()]
    blocks = [block for block in blocks if block]
    if sum(map(len, blocks)) != len(a):
        raise ValueError("group inventory must cover each task exactly once")
    rng, estimates = random.Random(seed), []
    for _ in range(draws):
        ids = [key for _ in blocks for key in blocks[rng.randrange(len(blocks))]]
        estimates.append((
            100*sum(a[key]["em"]-b[key]["em"] for key in ids)/len(ids),
            100*sum(a[key]["f1"]-b[key]["f1"] for key in ids)/len(ids),
            sum(a[key]["usage"]["total_tokens"] for key in ids)
            /sum(b[key]["usage"]["total_tokens"] for key in ids)))
    return {"unit": "independent_question_or_identical_context_group",
            "groups": len(blocks), "draws": draws,
            "em_delta_pp_ci95": [sorted(x[0] for x in estimates)[int(draws*p)] for p in (.025, .975)],
            "f1_delta_pp_ci95": [sorted(x[1] for x in estimates)[int(draws*p)] for p in (.025, .975)],
            "actor_token_ratio_ci95": [sorted(x[2] for x in estimates)[int(draws*p)] for p in (.025, .975)]}
