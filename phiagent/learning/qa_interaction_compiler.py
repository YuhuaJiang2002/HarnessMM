"""Provisional input-conditioned selection of source-bound instruction subsets.

Repeated full-factorial observations supervise a conditional second-order set
function. Its predictions and A/A noise guard are not quality certificates.
Real-data fitting and calibration belong on an authorized remote server.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations, product
import math
from types import MappingProxyType

from phiagent.learning.qa_meta_rsi import digest, text_hash
from phiagent.learning.qa_quality_cost_router import (
    feature_dimension,
    feature_spec,
    input_features,
)
from phiagent.learning.qa_rule_module_interventions import MODULE_IDS, module_mask_request


SCHEMA = "phiagent-qa-interaction-compiler/v1"
METHOD_ID = "conditional-rule-subsets-v1"
MASKS = tuple("".join(bits) for bits in product("01", repeat=len(MODULE_IDS)))
FULL = "1" * len(MODULE_IDS)
ENDPOINTS = ("exact_match", "f1", "completion_tokens")


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _hash(value: str, name: str) -> None:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def interaction_terms(order: int) -> tuple[tuple[int, ...], ...]:
    if type(order) is not int or order not in (1, 2, 5):
        raise ValueError("order must be 1 (additive), 2 (pair), or 5 (saturated-profile control)")
    return tuple(
        term for degree in range(order + 1)
        for term in combinations(range(len(MODULE_IDS)), degree)
    )


def mask_basis(mask: str, order: int) -> tuple[int, ...]:
    if mask not in MASKS:
        raise ValueError("a complete five-bit mask in the frozen module order is required")
    return tuple(
        math.prod(1 if mask[index] == "1" else -1 for index in term)
        for term in interaction_terms(order)
    )


@dataclass(frozen=True)
class Outcome:
    exact_match: int
    f1: float
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if type(self.exact_match) is not int or self.exact_match not in (0, 1):
            raise ValueError("exact_match must be a binary official per-question score")
        if not 0 <= _number(self.f1, "f1") <= 1:
            raise ValueError("f1 must use the normalized official [0, 1] scale")
        for name, minimum in (("prompt_tokens", 1), ("completion_tokens", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class RepeatBlock:
    block_id: str
    outcomes: Mapping[str, Outcome]
    reference_aa: Outcome

    def __post_init__(self) -> None:
        _identity(self.block_id, "block ID")
        if (
            set(self.outcomes) != set(MASKS)
            or any(not isinstance(value, Outcome) for value in self.outcomes.values())
            or not isinstance(self.reference_aa, Outcome)
        ):
            raise ValueError("every repeat needs all 32 masks and a separate full-reference A/A")
        if self.reference_aa.prompt_tokens != self.outcomes[FULL].prompt_tokens:
            raise ValueError("reference A/A must retain the full reference's native input length")
        object.__setattr__(self, "outcomes", MappingProxyType(dict(self.outcomes)))


@dataclass(frozen=True)
class InterventionTask:
    task_id: str
    group_id: str
    question: str
    context: str
    bank_sha256: str
    blocks: tuple[RepeatBlock, ...]

    def __post_init__(self) -> None:
        for name in ("task_id", "group_id", "question", "context"):
            _identity(getattr(self, name), name)
        _hash(self.bank_sha256, "prompt bank hash")
        blocks = tuple(self.blocks)
        if (
            len(blocks) < 2 or any(not isinstance(block, RepeatBlock) for block in blocks)
            or len({block.block_id for block in blocks}) != len(blocks)
        ):
            raise ValueError("at least two uniquely identified repeat blocks are required")
        for mask in MASKS:
            if len({block.outcomes[mask].prompt_tokens for block in blocks}) != 1:
                raise ValueError("native input counts must be unchanged within a task/mask cell")
        object.__setattr__(self, "blocks", blocks)

    @property
    def input_sha256(self) -> str:
        return digest({"question": self.question, "context": self.context})


@dataclass(frozen=True, init=False)
class InstructionBank:
    """Reuse exact source splicing; neither rule text nor the answer contract is rewritten."""

    prompts: Mapping[str, str]
    profile: str
    source_sha256: str
    sha256: str

    def __init__(self, *, system_prompt: str, profile: str = "delete") -> None:
        prompts = {
            mask: module_mask_request(
                system_prompt=system_prompt, profile=profile, mask=mask,
            ).system_prompt
            for mask in MASKS
        }
        object.__setattr__(self, "prompts", MappingProxyType(prompts))
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "source_sha256", text_hash(system_prompt))
        object.__setattr__(self, "sha256", digest(self.artifact))

    @property
    def artifact(self) -> dict:
        return {
            "profile": self.profile,
            "source_sha256": self.source_sha256,
            "module_order": list(MODULE_IDS),
            "prompt_sha256": {
                mask: text_hash(prompt) for mask, prompt in self.prompts.items()
            },
        }


def _tasks(
    tasks: Sequence[InterventionTask], bank_sha256: str,
) -> tuple[InterventionTask, ...]:
    result = tuple(tasks)
    if not result or any(not isinstance(task, InterventionTask) for task in result):
        raise ValueError("nonempty typed intervention tasks are required")
    if len({task.task_id for task in result}) != len(result):
        raise ValueError("task IDs must be unique")
    if any(task.bank_sha256 != bank_sha256 for task in result):
        raise ValueError("all observations must bind the exact frozen prompt bank")
    return result


def _mean_outcome(task: InterventionTask, mask: str, endpoint: str) -> float:
    return math.fsum(
        getattr(block.outcomes[mask], endpoint) for block in task.blocks
    ) / len(task.blocks)


def _observations_digest(tasks: Sequence[InterventionTask]) -> str:
    return digest([
        {
            "id": task.task_id, "group": task.group_id, "input_sha256": task.input_sha256,
            "blocks": [
                {
                    "id": block.block_id,
                    "outcomes": {mask: vars(block.outcomes[mask]) for mask in MASKS},
                    "reference_aa": vars(block.reference_aa),
                }
                for block in task.blocks
            ],
        }
        for task in tasks
    ])


def fit_compiler(
    tasks: Sequence[InterventionTask], *, bank: InstructionBank,
    order: int = 2, feature_mode: str = "numeric", ridge: float = 1.0, seed: int = 0,
    task_conditioned: bool = True, repeat_strategy: str = "mean",
) -> dict:
    """Fit Walsh main/interaction effects, averaging repeats inside source groups."""
    terms = interaction_terms(order)
    ridge = _number(ridge, "ridge", minimum=0)
    if ridge == 0 or type(seed) is not int:
        raise ValueError("a positive ridge and integer provenance seed are required")
    if type(task_conditioned) is not bool or repeat_strategy not in {"mean", "first"}:
        raise ValueError("declare input conditioning and mean/first repeat supervision")
    tasks = _tasks(tasks, bank.sha256)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Fitting requires the optional phiagent[qa-routing] dependency") from error

    groups = Counter(task.group_id for task in tasks)
    weights = np.asarray([1 / groups[task.group_id] for task in tasks])
    features = np.asarray([
        input_features(question=task.question, context=task.context, mode=feature_mode)
        for task in tasks
    ])
    if not task_conditioned:
        features = np.zeros_like(features)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1
    design = np.column_stack([np.ones(len(tasks)), (features - mean) / scale])
    basis = np.asarray([mask_basis(mask, order) for mask in MASKS])
    observations = np.asarray([
        [
            [
                _mean_outcome(task, mask, endpoint) if repeat_strategy == "mean"
                else getattr(task.blocks[0].outcomes[mask], endpoint)
                for endpoint in ENDPOINTS
            ]
            for mask in MASKS
        ]
        for task in tasks
    ])
    # Full-factorial orthogonality keeps the pair terms distinct from main effects.
    targets = np.einsum("mt,nme->nte", basis, observations) / len(MASKS)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        design.T @ (design * weights[:, None]) + penalty,
        design.T @ (targets.reshape(len(tasks), -1) * weights[:, None]),
    ).reshape(design.shape[1], len(terms), len(ENDPOINTS))
    noise = [
        [
            math.fsum(
                abs(getattr(block.reference_aa, endpoint)
                    - getattr(block.outcomes[FULL], endpoint))
                for block in task.blocks
            ) / len(task.blocks)
            for endpoint in ENDPOINTS[:2]
        ]
        for task in tasks
    ]
    artifact = {
        "schema_version": SCHEMA,
        "method_id": METHOD_ID,
        "bank": bank.artifact,
        "bank_sha256": bank.sha256,
        "order": order,
        "task_conditioned": task_conditioned,
        "terms": [list(term) for term in terms],
        "feature_mode": feature_mode,
        "feature_spec": feature_spec(feature_mode),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "heads": {
            endpoint: coefficients[:, :, index].T.tolist()
            for index, endpoint in enumerate(ENDPOINTS)
        },
        "aa_noise_floor": np.average(np.asarray(noise), axis=0, weights=weights).tolist(),
        "training": {
            "task_hashes": sorted(text_hash(task.task_id) for task in tasks),
            "group_hashes": sorted(text_hash(group) for group in groups),
            "input_hashes": sorted({task.input_sha256 for task in tasks}),
            "tasks": len(tasks),
            "groups": len(groups),
            "repeat_blocks": sum(len(task.blocks) for task in tasks),
            "ridge": ridge,
            "seed": seed,
            "repeat_strategy": repeat_strategy,
            "aa_uses_all_acquired_blocks": True,
            "observations_sha256": _observations_digest(tasks),
        },
        "quality_certificate": False,
        "paper_admitted": False,
    }
    return InteractionCompiler(artifact).artifact


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} requires exactly {length} entries")
    return tuple(_number(item, name) for item in value)


class InteractionCompiler:
    def __init__(self, artifact: Mapping) -> None:
        value = deepcopy(dict(artifact))
        if (
            value.get("schema_version") != SCHEMA or value.get("method_id") != METHOD_ID
            or value.get("quality_certificate") is not False
            or value.get("paper_admitted") is not False
        ):
            raise ValueError("an explicitly provisional instruction-compiler artifact is required")
        self.order = value["order"]
        if type(value["task_conditioned"]) is not bool:
            raise ValueError("input-conditioning mode must be an explicit boolean")
        terms = interaction_terms(self.order)
        if value["terms"] != [list(term) for term in terms]:
            raise ValueError("the frozen interaction basis changed")
        bank = value["bank"]
        if (
            set(bank) != {"profile", "source_sha256", "module_order", "prompt_sha256"}
            or bank["profile"] not in {"delete", "modify"}
            or bank["module_order"] != list(MODULE_IDS)
            or set(bank["prompt_sha256"]) != set(MASKS)
            or bank["source_sha256"] != bank["prompt_sha256"][FULL]
            or digest(bank) != value["bank_sha256"]
        ):
            raise ValueError("source, mask inventory, or prompt-bank binding changed")
        for entry in bank["prompt_sha256"].values():
            _hash(entry, "prompt hash")
        self.mode = value["feature_mode"]
        if value["feature_spec"] != feature_spec(self.mode):
            raise ValueError("frozen input-only feature schema changed")
        dimension = feature_dimension(self.mode)
        self.mean = _vector(value["feature_mean"], dimension, "feature means")
        self.scale = _vector(value["feature_scale"], dimension, "feature scales")
        if any(item <= 0 for item in self.scale):
            raise ValueError("feature scales must be positive")
        if set(value["heads"]) != set(ENDPOINTS):
            raise ValueError("EM, F1 and output cost must remain separate prediction heads")
        self.heads = {}
        for endpoint, rows in value["heads"].items():
            if not isinstance(rows, list) or len(rows) != len(terms):
                raise ValueError("each endpoint must contain every frozen interaction term")
            self.heads[endpoint] = tuple(
                _vector(row, dimension + 1, endpoint) for row in rows
            )
        self.noise = _vector(value["aa_noise_floor"], 2, "A/A absolute-difference means")
        if any(not 0 <= item <= 1 for item in self.noise):
            raise ValueError("A/A quality variation must lie in [0, 1]")
        training = value["training"]
        for name in ("task_hashes", "group_hashes", "input_hashes"):
            hashes = training[name]
            if (
                not isinstance(hashes, list) or not hashes
                or len(hashes) != len(set(hashes))
            ):
                raise ValueError("training exposure hashes must be nonempty and unique")
            for entry in hashes:
                _hash(entry, name)
        if (
            type(training["tasks"]) is not int
            or training["tasks"] != len(training["task_hashes"])
            or type(training["groups"]) is not int
            or training["groups"] != len(training["group_hashes"])
            or not 0 < training["groups"] <= training["tasks"]
            or type(training["repeat_blocks"]) is not int
            or training["repeat_blocks"] < 2 * training["tasks"]
            or type(training["seed"]) is not int
            or _number(training["ridge"], "training ridge", minimum=0) == 0
            or training["repeat_strategy"] not in {"mean", "first"}
            or training["aa_uses_all_acquired_blocks"] is not True
        ):
            raise ValueError("training counts or settings are inconsistent")
        _hash(training["observations_sha256"], "training observations hash")
        self._artifact = value
        self.model_sha256 = digest(value)

    @property
    def artifact(self) -> dict:
        return deepcopy(self._artifact)

    def predict_effects(self, *, question: str, context: str) -> dict[str, list[float]]:
        """Predict each endpoint's Walsh coefficients from the original input only."""
        features = input_features(question=question, context=context, mode=self.mode)
        if not self._artifact["task_conditioned"]:
            features = [0.0] * len(features)
        vector = [1.0, *(
            (value - mean) / scale
            for value, mean, scale in zip(features, self.mean, self.scale, strict=True)
        )]
        effects = {
            endpoint: [
                math.fsum(w * x for w, x in zip(row, vector, strict=True)) for row in rows
            ]
            for endpoint, rows in self.heads.items()
        }
        if not all(math.isfinite(item) for values in effects.values() for item in values):
            raise ValueError("input feature normalization or set-function prediction overflowed")
        return effects

    def choose(
        self, *, question: str, context: str, prompt_tokens: Mapping[str, int],
        quality_margin: float = 0.0, repeat_guard: float = 1.0,
    ) -> dict:
        margin = _number(quality_margin, "quality margin", minimum=0)
        guard = _number(repeat_guard, "repeat guard", minimum=0)
        if (
            set(prompt_tokens) != set(MASKS)
            or any(type(value) is not int or value <= 0 for value in prompt_tokens.values())
        ):
            raise ValueError("complete positive native input counts for all masks are required")
        effects = self.predict_effects(question=question, context=context)
        estimates = {}
        for mask in MASKS:
            basis = mask_basis(mask, self.order)
            regrets = [
                math.fsum((1 - bit) * effect for bit, effect in zip(
                    basis, effects[endpoint], strict=True,
                ))
                for endpoint in ENDPOINTS[:2]
            ]
            guarded = [
                regret + margin + guard * noise
                for regret, noise in zip(regrets, self.noise, strict=True)
            ]
            raw_output = math.fsum(
                bit * effect
                for bit, effect in zip(basis, effects["completion_tokens"], strict=True)
            )
            output = max(0.0, raw_output)
            estimates[mask] = {
                "predicted_regret": dict(zip(ENDPOINTS[:2], regrets, strict=True)),
                "guarded_regret": dict(zip(ENDPOINTS[:2], guarded, strict=True)),
                "eligible": mask == FULL or all(value <= 0 for value in guarded),
                "prompt_tokens": prompt_tokens[mask],
                "raw_predicted_completion_tokens": raw_output,
                "predicted_completion_tokens": output,
                "estimated_total_tokens": prompt_tokens[mask] + output,
            }
        selected = min(
            (mask for mask in MASKS if estimates[mask]["eligible"]),
            key=lambda mask: (
                estimates[mask]["estimated_total_tokens"], prompt_tokens[mask], mask,
            ),
        )
        return {
            "method_id": METHOD_ID,
            "selected_mask": selected,
            "retained_modules": [
                name for name, bit in zip(MODULE_IDS, selected, strict=True) if bit == "1"
            ],
            "status": "REFERENCE_RETAINED" if selected == FULL else "SUBSET_SELECTED",
            "reason": (
                "Full reference is the cheapest eligible choice; no improvement is asserted."
                if selected == FULL else
                "Lowest predicted total cost among subsets passing each predicted regret guard."
            ),
            "estimates": estimates,
            "model_sha256": self.model_sha256,
            "policy_sha256": digest({
                "model_sha256": self.model_sha256,
                "quality_margin": margin, "repeat_guard": guard,
            }),
            "bank_sha256": self._artifact["bank_sha256"],
            "input_sha256": digest({"question": question, "context": context}),
            "quality_margin": margin,
            "repeat_guard": guard,
            "aa_noise_guard_is_confidence_bound": False,
            "quality_certificate": False,
            "runtime_outcomes_used": False,
            "router_model_calls": 0,
            "paper_admitted": False,
        }

    def compile(
        self, *, bank: InstructionBank, question: str, context: str,
        count_complete_request: Callable[[str, str, str], int],
        quality_margin: float = 0.0, repeat_guard: float = 1.0,
    ) -> dict:
        _identity(question, "question")
        _identity(context, "context")
        if not callable(count_complete_request):
            raise ValueError("a complete native request token counter is required")
        if bank.sha256 != self._artifact["bank_sha256"]:
            raise ValueError("compiler and renderer bind different source prompts")
        counts = {
            mask: count_complete_request(bank.prompts[mask], question, context)
            for mask in MASKS
        }
        receipt = self.choose(
            question=question, context=context, prompt_tokens=counts,
            quality_margin=quality_margin, repeat_guard=repeat_guard,
        )
        prompt = bank.prompts[receipt["selected_mask"]]
        return {
            "system_prompt": prompt, "question": question, "context": context,
            "receipt": {**receipt, "system_prompt_sha256": text_hash(prompt)},
        }


def calibrate_margin(
    compiler: InteractionCompiler, tasks: Sequence[InterventionTask], *,
    margins: Sequence[float], repeat_guard: float = 1.0, minimum_token_saving: float = 0.1,
) -> dict:
    """Select on disjoint calibration outcomes; this is not a deployment certificate."""
    artifact = compiler.artifact
    tasks = _tasks(tasks, artifact["bank_sha256"])
    if not margins:
        raise ValueError("declare the finite margin family before calibration")
    margins = tuple(_number(item, "calibration margin", minimum=0) for item in margins)
    if len(set(margins)) != len(margins):
        raise ValueError("calibration margins must be unique")
    guard = _number(repeat_guard, "repeat guard", minimum=0)
    saving = _number(minimum_token_saving, "minimum saving", minimum=0)
    if not 0 < saving < 1:
        raise ValueError("minimum saving must be strictly between zero and one")
    for key, identities in (
        ("task_hashes", {text_hash(task.task_id) for task in tasks}),
        ("group_hashes", {text_hash(task.group_id) for task in tasks}),
        ("input_hashes", {task.input_sha256 for task in tasks}),
    ):
        if identities & set(artifact["training"][key]):
            raise ValueError(f"training/calibration exposure overlaps in {key}")

    def summarize(assignments: Sequence[str]) -> dict:
        return {
            **{
                endpoint: math.fsum(
                    _mean_outcome(task, mask, endpoint)
                    for task, mask in zip(tasks, assignments, strict=True)
                ) / len(tasks)
                for endpoint in ENDPOINTS[:2]
            },
            "mean_total_tokens": math.fsum(
                _mean_outcome(task, mask, "total_tokens")
                for task, mask in zip(tasks, assignments, strict=True)
            ) / len(tasks),
        }

    fixed = {mask: summarize([mask] * len(tasks)) for mask in MASKS}
    reference = fixed[FULL]
    eligible_fixed = [
        mask for mask in MASKS
        if all(fixed[mask][key] >= reference[key] for key in ENDPOINTS[:2])
    ]
    best_fixed = min(eligible_fixed, key=lambda mask: (fixed[mask]["mean_total_tokens"], mask))
    references = (reference, fixed[best_fixed])
    rows = []
    for margin in margins:
        assignments = [
            compiler.choose(
                question=task.question, context=task.context,
                prompt_tokens={
                    mask: task.blocks[0].outcomes[mask].prompt_tokens for mask in MASKS
                },
                quality_margin=margin, repeat_guard=guard,
            )["selected_mask"]
            for task in tasks
        ]
        summary = summarize(assignments)
        quality_pass = all(
            summary[key] >= comparison[key]
            for comparison in references for key in ENDPOINTS[:2]
        )
        cost_pass = all(
            summary["mean_total_tokens"] <= (1 - saving) * comparison["mean_total_tokens"]
            for comparison in references
        )
        rows.append({
            "margin": margin, **summary,
            "policy_sha256": digest({
                "model_sha256": compiler.model_sha256,
                "quality_margin": margin, "repeat_guard": guard,
            }),
            "changed_tasks": sum(mask != FULL for mask in assignments),
            "quality_point_gate": quality_pass,
            "cost_point_gate": cost_pass,
            "eligible": quality_pass and cost_pass and any(mask != FULL for mask in assignments),
            "assignments_sha256": digest(dict(zip(
                (task.task_id for task in tasks), assignments, strict=True,
            ))),
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = (
        min(eligible, key=lambda row: (row["mean_total_tokens"], -row["margin"]))
        if eligible else None
    )
    return {
        "status": (
            "CALIBRATION_POINT_GATE_PASS_PENDING_CONFIRMATION"
            if selected else "BLOCKED_NO_ADMISSIBLE_CALIBRATION_POLICY"
        ),
        "reason": (
            "Selected from calibration points only; a frozen fresh-call evaluation is still required."
            if selected else
            "No changed policy meets both endpoint floors and the saving gate versus both references."
        ),
        "selected_margin": selected["margin"] if selected else None,
        "repeat_guard": guard,
        "minimum_token_saving": saving,
        "candidate_family": rows,
        "fixed_controls": fixed,
        "best_feasible_fixed_mask": best_fixed,
        "tasks": len(tasks),
        "groups": len({task.group_id for task in tasks}),
        "model_sha256": compiler.model_sha256,
        "selected_policy_sha256": selected["policy_sha256"] if selected else None,
        "bank_sha256": artifact["bank_sha256"],
        "calibration_group_hashes": sorted({text_hash(task.group_id) for task in tasks}),
        "calibration_observations_sha256": _observations_digest(tasks),
        "benchmark_result": False,
        "statistical_noninferiority_established": False,
        "paper_admitted": False,
    }
