"""Dream-RSI-inspired training-controller adaptation, not an official reproduction.

Replay consumes recorded continuations only. Discovery and policy-development
callbacks must use separately authorized, bounded remote clients. Deployment
still requires the existing QA gates and fresh official evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import re

from phiagent.learning import qa_meta_rsi as qa


IDENTITY = "Dream-RSI-inspired training-controller adaptation"
PRIMARY_SOURCE = "https://arxiv.org/html/2609.14858v1"
SOURCE_REVISION = "4149ea9181ab1db80f85717ffda2c9f0f130e85b"
WORLD_SCHEMA = "phiagent-qa-dream-world/v1"


def _integer(value: object, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value: object, name: str, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, not boolean or missing")
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be finite, nonnegative and within its declared range")


def _hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires a SHA256 binding")


@dataclass(frozen=True)
class Usage:
    actor_calls: int = 0
    evaluator_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _integer(value, name)
        if not self.actor_calls and self.prompt_tokens + self.completion_tokens:
            raise ValueError("nonzero model usage requires actual actor calls")

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{key: value + asdict(other)[key] for key, value in asdict(self).items()})


@dataclass(frozen=True)
class Outcome:
    exact_match: float | None
    f1: float | None
    deployment_tokens: int | None
    eligible: bool
    error: str | None
    usage: Usage

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool or not isinstance(self.usage, Usage):
            raise ValueError("outcomes require explicit eligibility and complete measured usage")
        if self.exact_match is None:
            if self.f1 is not None or self.deployment_tokens is not None or self.eligible:
                raise ValueError("unmeasured failures cannot carry scores or be eligible")
        else:
            _finite(self.exact_match, "exact_match", 1)
            _finite(self.f1, "f1", 1)
            _integer(self.deployment_tokens, "deployment_tokens")
            _, _, measured_tokens = self.metrics()
            if (self.usage.actor_calls < 1 or self.usage.evaluator_calls < 1
                    or self.usage.prompt_tokens + self.usage.completion_tokens
                    < measured_tokens):
                raise ValueError("measured QA outcomes require complete actor, scorer and token costs")
        if self.eligible:
            if self.error is not None:
                raise ValueError("eligible outcomes cannot hide an error")
        elif not isinstance(self.error, str) or not self.error.strip():
            raise ValueError("failed or rejected outcomes require an explicit diagnosis")

    def metrics(self) -> tuple[float, float, int]:
        if self.exact_match is None or self.f1 is None or self.deployment_tokens is None:
            raise ValueError("an unmeasured outcome has no quality or deployment-cost metrics")
        return self.exact_match, self.f1, self.deployment_tokens


@dataclass(frozen=True)
class Node:
    node_id: str
    parent_id: str | None
    artifact_sha256: str
    outcome: Outcome

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("discovery nodes require nonempty identities")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id
        ):
            raise ValueError("parent identities must be nonempty or None")
        _hash(self.artifact_sha256, "artifact")
        if not isinstance(self.outcome, Outcome):
            raise ValueError("discovery nodes require a measured outcome or explicit failure")


@dataclass(frozen=True)
class World:
    world_id: str
    partition: str
    group_ids: tuple[str, ...]
    provenance_sha256: str
    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.world_id, str) or not self.world_id:
            raise ValueError("world identity is required")
        if self.partition not in {"train", "controller_validation"}:
            raise ValueError("replay accepts training-source worlds, never public dev/test outcomes")
        if (not isinstance(self.group_ids, tuple) or not self.group_ids
                or any(not isinstance(group, str) or not group
                                      for group in self.group_ids)
                or len(set(self.group_ids)) != len(self.group_ids)):
            raise ValueError("worlds require unique whole-question-group identities")
        _hash(self.provenance_sha256, "model/protocol provenance")
        if (not isinstance(self.nodes, tuple) or not self.nodes
                or any(not isinstance(node, Node) for node in self.nodes)
                or self.nodes[0].parent_id is not None):
            raise ValueError("a world must start with its measured root")
        if not self.nodes[0].outcome.eligible:
            raise ValueError("the root must be an eligible measured reference")
        seen, child_counts = set(), {}
        root = self.nodes[0].node_id
        for index, node in enumerate(self.nodes):
            if node.node_id in seen:
                raise ValueError("duplicate discovery-node identity")
            if index:
                if node.parent_id not in seen:
                    raise ValueError("recorded parents must precede their children")
                child_counts[node.parent_id] = child_counts.get(node.parent_id, 0) + 1
                if node.parent_id != root and child_counts[node.parent_id] > 1:
                    raise ValueError("a non-root node has only one recorded continuation")
            seen.add(node.node_id)

    def to_dict(self) -> dict:
        return {"schema_version": WORLD_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict) -> World:
        if set(value) != {
            "schema_version", "world_id", "partition", "group_ids", "provenance_sha256", "nodes",
        } or value["schema_version"] != WORLD_SCHEMA:
            raise ValueError("world schema or fields differ")
        nodes = []
        for raw in value["nodes"]:
            outcome = dict(raw["outcome"])
            outcome["usage"] = Usage(**outcome["usage"])
            nodes.append(Node(**{**raw, "outcome": Outcome(**outcome)}))
        return cls(
            value["world_id"], value["partition"], tuple(value["group_ids"]),
            value["provenance_sha256"], tuple(nodes),
        )


@dataclass(frozen=True)
class Limits:
    max_rounds: int = 8
    max_nodes: int = 16
    max_parallel: int = 2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _integer(value, name, 1)


@dataclass(frozen=True)
class Objective:
    node_penalty: float = 0.001
    batch_reward: float = 0.001

    def __post_init__(self) -> None:
        _finite(self.node_penalty, "node_penalty")
        _finite(self.batch_reward, "batch_reward")


@dataclass(frozen=True)
class Policy:
    priority: str = "quality"
    root_first: bool = False
    batch_size: int = 2
    minimum_rounds: int = 1
    patience: int | None = 2
    repair_limit: int = 1

    def __post_init__(self) -> None:
        if self.priority not in {"quality", "improvement", "cost"}:
            raise ValueError("unknown bounded controller priority")
        if type(self.root_first) is not bool:
            raise ValueError("root_first must be boolean")
        for name in ("batch_size", "minimum_rounds"):
            _integer(getattr(self, name), name, 1)
        _integer(self.repair_limit, "repair_limit")
        if self.patience is not None:
            _integer(self.patience, "patience", 1)

    def validate_limits(self, limits: Limits) -> None:
        if self.batch_size > limits.max_parallel or self.minimum_rounds > limits.max_rounds:
            raise ValueError("controller exceeds the frozen worker/round limits")


@dataclass(frozen=True)
class View:
    observed: tuple[Node, ...]
    legal_actions: tuple[str, ...]
    rounds: int
    stalled_rounds: int
    remaining_nodes: int


def _quality(node: Node) -> tuple[float, float, int]:
    outcome = node.outcome
    if outcome.exact_match is None:
        return -1.0, -1.0, 0
    em, f1, tokens = outcome.metrics()
    return em, f1, -tokens


def _best(nodes: Sequence[Node]) -> Node:
    return max((node for node in nodes if node.outcome.eligible), key=_quality)


def choose_actions(policy: Policy, view: View) -> tuple[str, ...]:
    """Use only the observed prefix and structural action availability."""
    if (policy.patience is not None and view.rounds >= policy.minimum_rounds
            and view.stalled_rounds >= policy.patience):
        return ()
    by_id = {node.node_id: node for node in view.observed}
    root = view.observed[0].node_id
    if not set(view.legal_actions) <= set(by_id):
        raise ValueError("legal actions must refer to already observed parents")
    leaves = []
    for action in view.legal_actions:
        if action == root:
            continue
        node, failures = by_id[action], 0
        if node.parent_id is None:
            raise ValueError("a non-root action must have an observed parent")
        current = node
        while current.parent_id is not None and current.outcome.exact_match is None:
            failures += 1
            current = by_id[current.parent_id]
        if failures > policy.repair_limit:
            continue
        score = _quality(node)
        if policy.priority == "improvement":
            score = (score[0] - _quality(by_id[node.parent_id])[0], score[1], score[2])
        elif policy.priority == "cost":
            score = (score[2], score[0], score[1])
        leaves.append((action, score))
    ordered = [action for action, _ in sorted(leaves, key=lambda item: item[1], reverse=True)]
    if root in view.legal_actions:
        ordered.insert(0 if policy.root_first else len(ordered), root)
    return tuple(ordered[:min(policy.batch_size, view.remaining_nodes)])


@dataclass(frozen=True)
class Replay:
    world_id: str
    controller_sha256: str
    decisions: tuple[dict, ...]
    best_node: Node
    represented_usage: Usage
    reward: float
    actual_actor_calls: int = 0
    actual_evaluator_calls: int = 0


def replay(world: World, policy: Policy, limits: Limits, objective: Objective) -> Replay:
    policy.validate_limits(limits)
    children: dict[str, list[Node]] = {}
    for node in world.nodes[1:]:
        if node.parent_id is None:
            raise ValueError("a recorded continuation requires a parent")
        children.setdefault(node.parent_id, []).append(node)
    observed, seen = [world.nodes[0]], {world.nodes[0].node_id}
    decisions, usage, stalled = [], Usage(), 0
    while len(decisions) < limits.max_rounds and len(observed) - 1 < limits.max_nodes:
        available = {
            node.node_id: next(
                (child for child in children.get(node.node_id, []) if child.node_id not in seen), None,
            ) for node in observed
        }
        legal = tuple(parent for parent, child in available.items() if child is not None)
        view = View(
            tuple(observed), legal, len(decisions), stalled, limits.max_nodes - len(observed) + 1,
        )
        actions = choose_actions(policy, view)
        if not actions:
            break
        before = _quality(_best(observed))
        revealed = []
        for parent in actions:
            child = available[parent]
            if child is None:
                raise ValueError("unsupported replay action; no outcome may be invented")
            observed.append(child)
            seen.add(child.node_id)
            revealed.append(child.node_id)
            usage += child.outcome.usage
        stalled = 0 if _quality(_best(observed)) > before else stalled + 1
        decisions.append({"parents": actions, "revealed": tuple(revealed)})
    count, rounds = len(observed) - 1, len(decisions)
    best = _best(observed)
    reward = (
        best.outcome.metrics()[0] - objective.node_penalty * count
        + objective.batch_reward * count / max(1, rounds)
    )
    return Replay(world.world_id, qa.digest(asdict(policy)), tuple(decisions), best, usage, reward)


def collect_world(
    *, world_id: str, group_ids: tuple[str, ...], provenance_sha256: str, root: Node,
    policy: Policy, limits: Limits, discover: Callable[[Node, tuple[Node, ...]], Node],
) -> World:
    """Dispatch real training discovery; callbacks own transport, guards and logging."""
    policy.validate_limits(limits)
    nodes, rounds, stalled = [root], 0, 0
    World(world_id, "train", group_ids, provenance_sha256, tuple(nodes))
    with ThreadPoolExecutor(max_workers=limits.max_parallel) as pool:
        while rounds < limits.max_rounds and len(nodes) - 1 < limits.max_nodes:
            parents = {node.parent_id for node in nodes[1:]}
            legal = tuple(node.node_id for node in nodes
                          if node == root or node.node_id not in parents)
            prefix = tuple(nodes)
            actions = choose_actions(policy, View(
                prefix, legal, rounds, stalled, limits.max_nodes - len(nodes) + 1,
            ))
            if not actions:
                break
            by_id = {node.node_id: node for node in nodes}
            before = _quality(_best(nodes))
            batch = list(pool.map(lambda action: discover(by_id[action], prefix), actions))
            for parent, child in zip(actions, batch):
                if child.parent_id != parent:
                    raise ValueError("live discovery returned a child of the wrong parent")
                nodes.append(child)
            World(world_id, "train", group_ids, provenance_sha256, tuple(nodes))
            rounds += 1
            stalled = 0 if _quality(_best(nodes)) > before else stalled + 1
    return World(world_id, "train", group_ids, provenance_sha256, tuple(nodes))


def qa_outcome(
    *, parent_skill: dict, child_skill: dict, parent_rows: Mapping, child_rows: Mapping,
    original_rows: Mapping, usage: Usage,
) -> Outcome:
    """Retain the existing paired EM/F1 and ten-percent deployment-token gate."""
    gate = qa.skill_gate(
        parent_rows, child_rows, original_rows,
        changed=qa.skill_content_hash(parent_skill) != qa.skill_content_hash(child_skill),
    )
    scores = gate["child"]
    if (usage.actor_calls < scores["tasks"] or usage.evaluator_calls < scores["tasks"]
            or usage.prompt_tokens < scores["prompt_tokens"]
            or usage.completion_tokens < scores["completion_tokens"]):
        raise ValueError("discovery usage must include every measured child request and score")
    return Outcome(
        scores["exact_match"] / 100, scores["f1"] / 100, scores["total_inference_tokens"],
        gate["accepted"], None if gate["accepted"] else "; ".join(gate["failed_gates"]), usage,
    )


def _validate_pools(training: Sequence[World], validation: Sequence[World]) -> None:
    if not training or not validation:
        raise ValueError("both training and independent controller-validation worlds are required")
    worlds = [*training, *validation]
    if len({world.world_id for world in worlds}) != len(worlds):
        raise ValueError("repeated world identities cannot count as independent evidence")
    if (any(world.partition != "train" for world in training)
            or any(world.partition != "controller_validation" for world in validation)):
        raise ValueError("worlds entered the wrong protected partition")
    if len({world.provenance_sha256 for world in worlds}) != 1:
        raise ValueError("all worlds must share frozen actor, protocol and metric provenance")
    train_groups = {group for world in training for group in world.group_ids}
    validation_groups = {group for world in validation for group in world.group_ids}
    if train_groups & validation_groups:
        raise ValueError("controller-validation question groups overlap training")


def select_controller(
    incumbent: Policy, candidates: Sequence[Policy], training: Sequence[World],
    validation: Sequence[World], limits: Limits, objective: Objective,
) -> dict:
    _validate_pools(training, validation)
    policies = [incumbent, *candidates]
    training_results = [
        [replay(world, policy, limits, objective) for world in training] for policy in policies
    ]
    means = [math.fsum(row.reward for row in rows) / len(rows) for rows in training_results]
    winner = max(range(len(policies)), key=lambda index: means[index])
    before = [replay(world, incumbent, limits, objective) for world in validation]
    after = [replay(world, policies[winner], limits, objective) for world in validation]
    gates = {
        "controller_changed": policies[winner] != incumbent,
        "strict_training_reward_improvement": means[winner] > means[0] + 1e-12,
        "validation_em_f1_nonregression": all(
            new.best_node.outcome.metrics()[0] >= old.best_node.outcome.metrics()[0]
            and new.best_node.outcome.metrics()[1] >= old.best_node.outcome.metrics()[1]
            for old, new in zip(before, after)
        ),
        "validation_deployment_token_nonregression": all(
            new.best_node.outcome.metrics()[2] <= old.best_node.outcome.metrics()[2]
            for old, new in zip(before, after)
        ),
        "validation_replay_reward_nonregression":
            math.fsum(row.reward for row in after) >= math.fsum(row.reward for row in before),
    }
    accepted = all(gates.values())
    return {
        "method_identity": IDENTITY, "accepted": accepted, "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "controller": asdict(policies[winner] if accepted else incumbent),
        "training_winner": asdict(policies[winner]), "training_mean_rewards": means,
        "training_replays": [[asdict(row) for row in rows] for rows in training_results],
        "validation_incumbent": [asdict(row) for row in before],
        "validation_candidate": [asdict(row) for row in after],
        "paper_admitted": False, "official_benchmark_result": False,
        "fresh_harness_validation_required": True,
    }


@dataclass(frozen=True)
class ProposalAttempt:
    raw_json: str
    usage: Usage

    def __post_init__(self) -> None:
        if not isinstance(self.raw_json, str) or not isinstance(self.usage, Usage):
            raise ValueError("controller proposals require raw responses and measured usage")
        if self.usage.actor_calls < 1:
            raise ValueError("policy-development model calls cannot be treated as free")


def proposal_messages(
    parent: Policy, feedback: list[dict], limits: Limits, objective: Objective,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "Revise a bounded TRAINING exploration controller, not the deployed QA harness. "
            "Use only supplied training replay feedback. Return JSON with exactly "
            "parent_controller_sha256, hypothesis, evidence_world_ids, policy. "
            "The policy has exactly priority (quality/improvement/cost), root_first (boolean), "
            "batch_size, minimum_rounds, patience (positive integer or null), repair_limit. "
            "Do not generate executable source, task-specific conditions or hidden node IDs. "
            "The actor, task interfaces, metrics, limits and deployment gates are immutable. "
            "Replay supports only recorded continuations; it cannot predict missing outcomes."
        )},
        {"role": "user", "content": json.dumps({
            "parent_controller_sha256": qa.digest(asdict(parent)), "policy": asdict(parent),
            "training_replay_feedback": feedback, "limits": asdict(limits),
            "objective": {"identity": "section-3-equation-1-with-eligible-training-EM", **asdict(objective)},
        }, sort_keys=True, allow_nan=False)},
    ]


def compile_proposal(
    attempt: ProposalAttempt, parent: Policy, training_world_ids: set[str], limits: Limits,
) -> Policy:
    value = json.loads(attempt.raw_json)
    if not isinstance(value, dict) or set(value) != {
        "parent_controller_sha256", "hypothesis", "evidence_world_ids", "policy",
    }:
        raise ValueError("controller proposal schema differs")
    if value["parent_controller_sha256"] != qa.digest(asdict(parent)):
        raise ValueError("controller proposal is not bound to its exact parent")
    if (not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip()
            or len(value["hypothesis"]) > 1600):
        raise ValueError("a bounded nonempty hypothesis is required")
    evidence = value["evidence_world_ids"]
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, str) for item in evidence)
            or len(set(evidence)) != len(evidence) or not set(evidence) <= training_world_ids):
        raise ValueError("proposal evidence must identify supplied training worlds only")
    if not isinstance(value["policy"], dict) or set(value["policy"]) != set(asdict(parent)):
        raise ValueError("only the bounded controller fields are writable")
    candidate = Policy(**value["policy"])
    candidate.validate_limits(limits)
    if candidate == parent:
        raise ValueError("a byte-identical controller is not an update")
    return candidate


def improve_controller(
    *, incumbent: Policy, training: Sequence[World], validation: Sequence[World],
    limits: Limits, objective: Objective, versions: int,
    develop: Callable[[list[dict[str, str]]], ProposalAttempt],
) -> dict:
    _validate_pools(training, validation)
    _integer(versions, "versions including incumbent", 1)
    incumbent.validate_limits(limits)
    parent, candidates, attempts, feedback, usage = incumbent, [], [], [], Usage()
    for _ in range(versions - 1):
        feedback.append({
            "controller": asdict(parent),
            "replays": [asdict(replay(world, parent, limits, objective)) for world in training],
            "previous_proposal_error": attempts[-1]["error"] if attempts else None,
        })
        messages = proposal_messages(parent, feedback, limits, objective)
        attempt = develop(messages)
        usage += attempt.usage
        error = None
        try:
            candidate = compile_proposal(
                attempt, parent, {world.world_id for world in training}, limits,
            )
        except (ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            candidates.append(candidate)
            parent = candidate
        attempts.append({
            "messages": messages, "raw_json": attempt.raw_json, "usage": asdict(attempt.usage),
            "error": error, "accepted_for_replay": error is None,
        })
    result = select_controller(incumbent, candidates, training, validation, limits, objective)
    result["proposal_attempts"] = attempts
    result["policy_development_usage"] = asdict(usage)
    acquisition = Usage()
    for world in (*training, *validation):
        for node in world.nodes:
            acquisition += node.outcome.usage
    result["archive_acquisition_usage"] = asdict(acquisition)
    result["replay_actor_calls"] = 0
    result["replay_evaluator_calls"] = 0
    return result


def run_training_cycle(
    *, incumbent: Policy, training: Sequence[World], validation: Sequence[World],
    world_id: str, group_ids: tuple[str, ...], provenance_sha256: str, root: Node,
    limits: Limits, objective: Objective, versions: int,
    discover: Callable[[Node, tuple[Node, ...]], Node],
    persist_world: Callable[[World], None],
    develop: Callable[[list[dict[str, str]]], ProposalAttempt],
) -> tuple[World, dict]:
    prospective = World(world_id, "train", group_ids, provenance_sha256, (root,))
    _validate_pools([*training, prospective], validation)
    _integer(versions, "versions including incumbent", 1)
    world = collect_world(
        world_id=world_id, group_ids=group_ids, provenance_sha256=provenance_sha256,
        root=root, policy=incumbent, limits=limits, discover=discover,
    )
    persist_world(world)
    result = improve_controller(
        incumbent=incumbent, training=[*training, world], validation=validation,
        limits=limits, objective=objective, versions=versions, develop=develop,
    )
    result["new_world_sha256"] = qa.digest(world.to_dict())
    result["training_world_ids"] = [item.world_id for item in (*training, world)]
    return world, result
