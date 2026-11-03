"""Learn native decisions and invocation values without task-specific action rules.

The value gate uses terminal outcomes of paired one-step interventions followed
by the same actor. It is an empirical estimator, not a safety certificate or an
estimate of the value of repeatedly deploying the learned policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re


POLICY_SCHEMA = "chisel-learned-native-decisions/v1"
GATE_SCHEMA = "chisel-counterfactual-invocation-value/v2"
MIN_SIMILARITY = 0.20
MAX_NEIGHBORS = 128


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else ''}string")
    return value


def validate_public_state(state: dict) -> None:
    if not isinstance(state, dict) or set(state) != {"task", "observation", "actions", "history"}:
        raise ValueError("Public state permits only task, observation, actions and completed history")
    _text(state["task"], "task")
    _text(state["observation"], "observation", empty=True)
    actions = state["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError("Public state requires native actions")
    for action in actions:
        _text(action, "native action")
    if len(actions) != len(set(actions)):
        raise ValueError("Native actions must be unique")
    if not isinstance(state["history"], list):
        raise ValueError("History must be a list")
    for step in state["history"]:
        if not isinstance(step, dict) or set(step) != {"action", "observation"}:
            raise ValueError("History permits only completed actions and public observations")
        # A native trajectory may retain an empty, malformed past action.
        _text(step["action"], "history action", empty=True)
        _text(step["observation"], "history observation", empty=True)


def public_state_sha256(public_state: dict) -> str:
    validate_public_state(public_state)
    canonical = {**public_state, "actions": sorted(public_state["actions"])}
    return hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _features(state: dict, action: str | None = None) -> dict[str, float]:
    validate_public_state(state)
    fields = [
        ("task", state["task"]),
        ("observation", state["observation"]),
    ]
    for offset, step in enumerate(reversed(state["history"][-2:])):
        fields.extend([
            (f"history{offset}.action", step["action"]),
            (f"history{offset}.observation", step["observation"]),
        ])
    if action is not None:
        fields.append(("candidate", action))
    counts: Counter[str] = Counter()
    for field, text in fields:
        words = re.findall(r"\w+", text.casefold())
        counts.update(f"{field}:{word}" for word in words)
        counts.update(f"{field}:{left}/{right}" for left, right in zip(words, words[1:]))
    return {key: 1.0 + math.log(count) for key, count in counts.items()}


def _build_index(records: list[dict], *, with_action: bool) -> dict:
    vectors = [
        _features(row["public_state"], row["action"] if with_action else None)
        for row in records
    ]
    frequency: Counter[str] = Counter(key for vector in vectors for key in vector)
    idf = {key: 1.0 + math.log((len(records) + 1) / (count + 1))
           for key, count in frequency.items()}
    postings: dict[str, list[list[float | int]]] = defaultdict(list)
    for index, vector in enumerate(vectors):
        weighted = {key: value * idf[key] for key, value in vector.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        for key, value in weighted.items():
            postings[key].append([index, value / norm])
    return {"idf": idf, "postings": dict(postings), "with_action": with_action}


def _neighbors(model: dict, state: dict, excluded_groups: set[str] | frozenset[str],
               *, action: str | None = None) -> list[tuple[float, dict]]:
    if not isinstance(excluded_groups, (set, frozenset)) or any(
        not isinstance(group, str) for group in excluded_groups
    ):
        raise ValueError("Excluded groups must be a set of group IDs")
    features = _features(state, action if model["index"]["with_action"] else None)
    idf = model["index"]["idf"]
    weighted = {key: value * idf[key] for key, value in features.items() if key in idf}
    # Unknown features remain in the query norm; unfamiliar states must not
    # appear identical merely because only one common word was indexed.
    unknown_idf = 1.0 + math.log(len(model["records"]) + 1)
    norm = math.sqrt(sum(value * value for value in weighted.values())
                     + sum((value * unknown_idf) ** 2
                           for key, value in features.items() if key not in idf))
    if not norm:
        return []
    scores: dict[int, float] = defaultdict(float)
    records = model["records"]
    for feature, value in weighted.items():
        for index, coefficient in model["index"]["postings"].get(feature, []):
            row = records[index]
            if row["group_id"] in excluded_groups or row.get(
                "label_group_id", row["group_id"]
            ) in excluded_groups:
                continue
            if action is not None and row["action"] != action:
                continue
            scores[index] += value / norm * coefficient
    ranked = sorted(
        ((min(1.0, max(0.0, score)), records[index]) for index, score in scores.items()
         if score >= MIN_SIMILARITY),
        key=lambda item: (-item[0], item[1]["example_id"]),
    )
    return ranked[:MAX_NEIGHBORS]


def build_policy(examples: list[dict]) -> dict:
    if not isinstance(examples, list) or not examples:
        raise ValueError("A learned policy requires training demonstrations")
    records = []
    seen = set()
    for example in examples:
        for name in ("example_id", "task_id", "group_id", "action"):
            _text(example.get(name), name)
        if example["example_id"] in seen:
            raise ValueError("Duplicate demonstration ID")
        seen.add(example["example_id"])
        validate_public_state(example["public_state"])
        if example["action"] not in example["public_state"]["actions"]:
            raise ValueError("Demonstration action is not native")
        records.append({key: example[key] for key in (
            "example_id", "task_id", "group_id", "public_state", "action",
        )})
    records = json.loads(json.dumps(sorted(records, key=lambda row: row["example_id"])))
    return {
        "schema_version": POLICY_SCHEMA, "records": records,
        "index": _build_index(records, with_action=False),
        "semantics": "TF-IDF retrieval of demonstrated native actions; no task-specific rules",
    }


def retrieve_demonstrations(policy: dict, public_state: dict, *,
                            excluded_groups: set[str] | frozenset[str] = frozenset(),
                            limit: int = MAX_NEIGHBORS) -> list[tuple[float, dict]]:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("Unknown learned-decision policy schema")
    if type(limit) is not int or not 1 <= limit <= MAX_NEIGHBORS:
        raise ValueError("Retrieval limit must be within the frozen neighborhood size")
    validate_public_state(public_state)
    return _neighbors(policy, public_state, excluded_groups)[:limit]


def propose(policy: dict, public_state: dict, *,
            excluded_groups: set[str] | frozenset[str] = frozenset()) -> dict:
    grouped: dict[str, dict[str, tuple[float, dict]]] = defaultdict(dict)
    for score, row in retrieve_demonstrations(
        policy, public_state, excluded_groups=excluded_groups,
    ):
        action, group = row["action"], row["group_id"]
        if action in public_state["actions"] and group not in grouped[action]:
            grouped[action][group] = (score, row)
    if not grouped:
        return {
            "action": None, "confidence": 0.0, "margin": 0.0, "support_groups": [],
            "source_example_ids": [], "reason": "no_supported_native_demonstration",
        }
    candidates = []
    for action, groups in grouped.items():
        support = sorted(groups.values(), key=lambda item: (-item[0], item[1]["example_id"]))[:3]
        score = sum(item[0] for item in support) / len(support)
        candidates.append((score, action, support))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, action, support = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
    return {
        "action": action, "confidence": score, "margin": max(0.0, score - runner_up),
        "support_groups": [row["group_id"] for _, row in support],
        "source_example_ids": [row["example_id"] for _, row in support],
        "reason": "demonstration_similarity_not_calibrated_probability",
    }


def fit_value_gate(pairs: list[dict]) -> dict:
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Invocation learning requires measured counterfactual pairs")
    records, seen = [], set()
    for pair in pairs:
        for name in ("pair_id", "task_id", "group_id", "candidate_action"):
            _text(pair.get(name), name)
        if pair["pair_id"] in seen:
            raise ValueError("Duplicate counterfactual pair ID")
        seen.add(pair["pair_id"])
        prefix = pair.get("prefix_sha256")
        if not isinstance(prefix, str) or re.fullmatch(r"[0-9a-f]{64}", prefix) is None:
            raise ValueError("A verified replay-prefix hash is required")
        validate_public_state(pair["public_state"])
        if pair["candidate_action"] not in pair["public_state"]["actions"]:
            raise ValueError("Counterfactual candidate is not a native action")
        for arm in ("actor", "compiled"):
            if pair.get(f"{arm}_valid") is not True:
                raise ValueError("Invalid or incomplete branches cannot supply labels")
            success, tokens = pair.get(f"{arm}_success"), pair.get(f"{arm}_tokens")
            if type(success) is not int or success not in (0, 1):
                raise ValueError("Native terminal success must be integer zero or one")
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Complete suffix token cost must be a nonnegative integer")
        records.append({
            "example_id": pair["pair_id"], "task_id": pair["task_id"],
            "group_id": pair["group_id"], "public_state": pair["public_state"],
            "action": pair["candidate_action"], "prefix_sha256": prefix,
            "label_group_id": _text(pair.get("label_group_id", pair["group_id"]),
                                    "label origin group"),
            "regret": pair["actor_success"] - pair["compiled_success"],
            "saved_tokens": pair["actor_tokens"] - pair["compiled_tokens"],
        })
    records = json.loads(json.dumps(sorted(records, key=lambda row: row["example_id"])))
    return {
        "schema_version": GATE_SCHEMA, "records": records,
        "index": _build_index(records, with_action=True),
        "target": "paired terminal regret and complete suffix token difference",
        "guarantee": "none; repeated-policy effects require separate native evaluation",
    }


def _permuted_labels(gate: dict, excluded_groups: set[str] | frozenset[str]) -> dict:
    seed = gate["label_permutation_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Label permutation requires a nonnegative integer seed")
    grouped = defaultdict(list)
    for row in gate["records"]:
        if row["group_id"] != row["label_group_id"]:
            raise ValueError("Cannot permute already reassigned counterfactual labels")
        if row["group_id"] not in excluded_groups:
            grouped[row["group_id"]].append(row)
    if len(grouped) < 2:
        return {}
    groups = sorted(grouped, key=lambda group: hashlib.sha256(
        json.dumps([seed, group], separators=(",", ":")).encode(),
    ).hexdigest())
    reassigned = {}
    for index, group in enumerate(groups):
        source_group = groups[(index + 1) % len(groups)]
        recipients = sorted(grouped[group], key=lambda row: row["example_id"])
        donors = sorted(grouped[source_group], key=lambda row: row["example_id"])
        if len(recipients) != len(donors):
            raise ValueError("Matched group permutation requires equal prefix counts per group")
        for recipient, donor in zip(recipients, donors):
            reassigned[recipient["example_id"]] = {
                **recipient, "regret": donor["regret"], "saved_tokens": donor["saved_tokens"],
                "label_group_id": source_group,
            }
    return reassigned


def route(gate: dict, public_state: dict, proposal: dict, *,
          excluded_groups: set[str] | frozenset[str] = frozenset(),
          min_groups: int = 3, max_harm: float = 0.0) -> dict:
    if gate.get("schema_version") != GATE_SCHEMA:
        raise ValueError("Unknown invocation-value schema")
    if type(min_groups) is not int or min_groups < 1:
        raise ValueError("min_groups must be a positive integer")
    if type(max_harm) not in (int, float) or not math.isfinite(max_harm) or not -1 <= max_harm <= 1:
        raise ValueError("max_harm must be a finite terminal-regret threshold")
    validate_public_state(public_state)
    action = proposal.get("action")
    if action is not None and action not in public_state["actions"]:
        raise ValueError("Proposal escaped the current native action set")
    result = {
        "commit": False, "estimated_regret": None, "estimated_saved_tokens": None,
        "support_groups": [], "reason": "no_native_candidate",
    }
    if action is None:
        return result
    permuted = None
    if "label_permutation_seed" in gate:
        permuted = _permuted_labels(gate, excluded_groups)
        if not permuted:
            return {**result, "reason": "insufficient_groups_for_label_permutation"}
    independent = {}
    label_groups = set()
    for score, row in _neighbors(gate, public_state, excluded_groups, action=action):
        if permuted is not None:
            row = permuted[row["example_id"]]
        if row["group_id"] not in independent and row["label_group_id"] not in label_groups:
            independent[row["group_id"]] = (score, row)
            label_groups.add(row["label_group_id"])
    selected = sorted(independent.values(), key=lambda item: (-item[0], item[1]["example_id"]))[:8]
    result["support_groups"] = [row["label_group_id"] for _, row in selected]
    if len(selected) < min_groups:
        return {**result, "reason": "insufficient_independent_counterfactual_groups"}
    regret = sum(row["regret"] for _, row in selected) / len(selected)
    saving = sum(row["saved_tokens"] for _, row in selected) / len(selected)
    commit = regret <= max_harm and saving > 0
    return {
        **result, "commit": commit, "estimated_regret": regret,
        "estimated_saved_tokens": saving,
        "reason": ("supported_empirical_value" if commit
                   else "positive_regret" if regret > max_harm else "no_positive_suffix_saving"),
    }
