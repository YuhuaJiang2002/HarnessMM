"""Train-only CHISEL residual learning around an unchanged official harness.

Codex retains the original callback and freeze contracts. The DeepSeek SDK
adapter reuses this learner and selects runtime-specific freeze labels from
the declared execution identity.

Integration
-----------
Construct ``CodexChiselController`` with a frozen ``benchmark_partition`` manifest
(20/20/60, seed 20260928), a hash-only ``task_index``, and two synchronous
callbacks. Call ``train(train_tasks)``, ``select_validation(validation_tasks)``,
then ``freeze(source_hashes=...)``. Persist the returned policy's ``to_dict()``
in the new remote run directory. Only then call ``evaluate_test(test_loader,
policy_sha256=..., source_sha256=...)``. The loader is not invoked before both
seals exist and match. Re-training, re-selection and repeated testing on the
same controller are forbidden. The parent runner owns remote placement, native
scoring, durable callback logs, isolated workspaces, and the official Codex CLI.

``ChiselConfig.execution_seeds`` applies to training and validation (one
repetition by default). ``test_execution_seeds=None`` resolves to that tuple;
set a distinct tuple for the final paired test, e.g. five seeds for TB2, three
for GAIA2, or one for VQA. Both seed lists and their repetition counts are frozen.
``executor_workers`` bounds concurrent executor callbacks (default 1, maximum
64). Callbacks must support independent workspaces and thread-safe durable
logging keyed by request_id. Each optimizer iteration and every phase waits for
its preceding executor batch. On a callback exception, no further jobs are
submitted; active jobs are drained and all incurred calls/failures are retained.
Reports use planned order, independent of callback completion order.

``optimizer_max_units=None`` preserves full-TRAIN optimizer input. A positive
integer caps only the independent units supplied to each optimizer callback.
At zero-based iteration ``i``, units are ranked by SHA-256 of canonical JSON
``[run_id, split_seed, i, unit_id]``; the first units are selected without using
scores. Every question and execution repetition in each selected unit is
included. Baseline/candidate execution and routing fits still use ALL TRAIN.
With sampling enabled, ``previous_candidates`` contains instructions and their
provenance, not full-population routing statistics. Optimizer support IDs must
have appeared in the current or an earlier optimizer callback's evidence.
Reports and sampled-policy freezes retain per-iteration identities, commitments,
evidence hashes and actual callback exposures. These counts describe evidence
supplied to callbacks; the optimizer adapter owns matching provider-side logs.
The default callback payload is unchanged. This is a declared optimizer data
size, not a token ceiling, trace truncation, or experiment stopping condition.

For testing, SHA-256 of the canonical JSON ``[run_id, task_id, seed]`` chooses the
first arm: an even first digest byte means baseline first, otherwise CHISEL.
Both arms run sequentially within a pair, even for a no-op policy; independent
pairs run concurrently. Planned and actual callback-start arm orders are saved.

For the DeepSeek campaign, bind ``model='deepseek-flash'``,
``runtime='official_codex_cli'``, ``wire_api='responses'``, and
``transport='byte_preserving_metering_proxy'`` in ``execution_identity``, along
with the pinned Codex revision. The upstream is the native
``https://api.deepseek.com/responses`` endpoint. The parent runner's
``--provider-url`` points to its local metering proxy; the callback uses
``request_id`` for per-trial URL/header attribution. Preserve native Responses
request/response bodies and tool schemas unchanged. Do not add a Responses-to-
Chat adapter. The proxy alone holds the API key; credentials must not enter
identity metadata, task prompts, feedback, Codex context, or evaluator mounts.
Include the proxy/config and native-protocol receipt hashes in the parent's
freeze/provenance artifacts. This controller performs no provider translation.

All callback requests/results and public reports are JSON-serializable dicts.
The task index contains ``{task_id, unit_id, public_prompt_sha256}`` records;
``unit_id`` is the manifest's independent task/scenario/image ID. Hash the UTF-8
public prompt with SHA-256. Actual tasks contain exactly ``{task_id, unit_id,
public_prompt}``. Multiple questions on an image, and every repetition, must
share its image unit. No answer, category, evaluator metadata or custom routing
feature belongs in either interface. Every phase requires its complete indexed
task population. Preparing hashes does not authorize opening test data here.

``execute(request)`` receives:
  * ``request_id, run_id, phase, arm, pair_id, seed, task``;
  * ``identity``: the frozen model/runtime/transport identity supplied by the
    parent (at least ``model`` and ``runtime``);
  * ``codex_prompt`` and ``agents_md_append``: the ONLY text to apply;
  * ``residual``: null, or one provenance record ``{id, instruction,
    application}``; never inject this metadata a second time;
  * ``policy_sha256``/``candidate_sha256``: the immutable candidate hash;
    ``final_policy_sha256`` and ``source_sha256`` additionally bind test calls
    to their final freeze (null before test), plus ``contract``.

The executor must reset the task workspace for each request, use identical
Codex core/tools/backbone/scoring and the paired seed, and return measured
usage including its retries. For ``prompt_suffix``, submit ``codex_prompt``.
For ``agents_md``, submit the unchanged prompt and append ``agents_md_append``
to task-local AGENTS.md, preserving existing instructions. Never modify global
AGENTS.md or replace Codex's system instructions. An empty residual leaves
both the prompt and the existing AGENTS.md byte-identical.
Execution seeds identify paired repetitions; a provider without sampling-seed
support must report that explicitly rather than claim seeded model sampling.

Execution results contain exactly the required fields ``request_id, task_id,
identity, execution_status, native_score, success, usage``, with optional
``feedback`` (string) and ``provenance`` (JSON object). Status is one of
``completed, failed, timeout, blocked``. ``native_score`` is an actual native
per-task score in the configured range, or null; ``success`` is bool or null.
A failed execution may carry a score only if the native scorer actually
assigned it. Missing scores are never converted into zero. Training feedback
may include traces and native diagnostics. Validation/test feedback is discarded
by this controller, and cannot reach the optimizer.

``execution_result_from_receipt(request, receipt, native_score=score,
success=native_success)`` adapts the parent's durable executor result
(``status='COMPLETED'/'BLOCKED'``, ``final_answer``, and
``provider_usage={usage_status, known_usage, cached_input_tokens, ...}``) to
this callback contract. It also accepts an adapter's ``status='SCORED'`` when
an explicit native score is supplied. Pass fractional native scores in [0, 1],
including VQA's fractional accuracy; success remains null unless explicitly
provided by the native scorer. ``measured_usage_from_receipt`` supplies the same
accounting conversion for optimizer callbacks. An unresolved trial keeps its
complete-usage counters null, while its known partial counters remain in
provenance. A blocked execution with fully measured usage retains that usage.
Reused baseline receipts are charged once by the controller; do not add their
acquisition ledger again to the controller's cumulative accounting.

``optimize(request)`` is a real optimizer-model call. It receives ``request_id,
run_id, phase='train', iteration, identity, objective, limits, native_metric,
training_evidence, previous_candidates``. Each evidence row contains only a
training task, paired seed, measured baseline, and previous candidate results.
Return ``request_id, identity, execution_status, instruction,
supporting_task_ids, usage`` and optional ``provenance``. ``instruction=''``
abstains. Nonempty text must cite training task IDs and satisfy the frozen
character/word limits. There is no supplied task-solving residual. Duplicate
proposals reuse existing training evidence and are not charged as new executions.

``usage`` requires ``input_tokens, output_tokens, cache_read_tokens,
cache_write_tokens, total_tokens`` (nonnegative measured integers or null);
``cost_usd`` is optional (measured nonnegative number or null). Input includes
cached input; total is input + output, not input + output + cache. Unknown
fields remain unknown. Accounting reports measured subtotals and missing-call
counts separately. Optimization, validation selection and final execution have
separate ledgers. Baseline training evidence is charged once, although every
candidate pairs against it. Final baseline and CHISEL executions are both run,
even when the selected policy abstains. There is no cumulative token stop.

Method and provenance
---------------------
Each distinct optimizer residual is executed on the complete TRAIN population
against the shared plain-Codex baseline. Public-text features and paired score
deltas are averaged within independent units before invoking the existing
``residual_meta_training.fit_cross_fitted_linear_head``. Thus no question or
repetition from a held unit enters that fold's router fit. Its OOF threshold,
support and harm gate decide admission; all-source refitting supplies inference
weights. Cross-fitting applies to the routing head, not to residual generation.
It is discovery evidence, not an independent safety certificate. Insufficient
scores/tokens/support, nonpositive OOF advantage, out-of-support public features,
or a low local score all abstain. Validation selects among these already frozen
policies using mean native score; ties with baseline retain the empty policy.
No validation trace, example, threshold update or refit reaches training.

``implementation_provenance()`` records exact inspected design-source hashes and
current executable-source hashes. This is a PhiAgent CHISEL extension, not a
modification of Codex core or an official Codex optimization algorithm. Freeze
requires caller-computed SHA-256 entries ``codex_cli, executor, optimizer,
scorer`` (additional entries welcome). Bind source/config/model revisions and
verify those external hashes in the parent before opening test; Python cannot
attest opaque callbacks or enforce their network/filesystem access. This module
imports no model, benchmark dataset, evaluator, CUDA or third-party package.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock
from typing import Any

from phiagent.harness import benchmark_partition
from phiagent.learning import residual_meta_harness as residual
from phiagent.learning import residual_meta_training as training


SPLIT_SEED = 20260928
SCHEMA = "phiagent-codex-chisel/v1"
MAX_EXECUTOR_WORKERS = 64
TEST_PAIR_ORDER_RULE = "sha256(canonical_json([run_id,task_id,seed])):even_first_byte_baseline_first"
OPTIMIZER_UNIT_SELECTION_RULE = "sha256(canonical_json([run_id,split_seed,iteration,unit_id]))"
TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens",
)
USAGE_FIELDS = (*TOKEN_FIELDS, "cost_usd")
_STATUSES = frozenset({"completed", "failed", "timeout", "blocked"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUFFIX = "\n\n[CHISEL task-local residual]\n"
_AGENTS = "\n\n## CHISEL task-local residual\n\n"
_DESIGN_SOURCES = {
    "configs/chisel_method_identity.json":
        "44e9d1637632a2648c876a0e0c3a37175c250bff5eeb651b81d7653950410cc3",
    "phiagent/harness/mixture_router.py":
        "1811a672ef8f5d9752bcfbc0d6bdb5696e3f448b70dc25d82e0c826d6ee040be",
    "phiagent/learning/residual_meta_harness.py":
        "100636f874b9dccfbe443cd500a3b94e592268bc9f8b3f34282279aab6730bf2",
    "phiagent/learning/residual_meta_training.py":
        "ec90ef7ce88cf00ddb1b2cb3f12cfcaea74fc0afc9023bffcb6d4bcd1419e201",
    "scripts/freeze_chisel_ablation_routes.py":
        "f5b44110bf95fec191575c7a2fa96408bb5c6644fc6b5d4822b9582db66247da",
    "scripts/run_chisel_mmmu_shared.py":
        "3e57ba619241d0fde069e59e12550d68e1a9dde6e04104d3b8d455d309547ab7",
    "scripts/run_aime_chisel_202060.py":
        "41bc51f0f935f53634a6b60e05219a08fd606c2d67c997046618dbd4ef2c0aeb",
}
_COMPILER_OBJECTIVE = (
    "Compile ONE short, reusable residual instruction for the unchanged official industrial "
    "harness identified in this request. The PRIMARY objective is higher native task success. "
    "Do not treat cheaper failed tasks, shorter narration, or an unchanged low success rate "
    "as a success improvement. Within the unchanged native limits, prefer work that completes "
    "previously failed user requirements over stopping early to save tokens. Use complete-task "
    "token cost only as a secondary criterion among candidates with comparable native quality, "
    "using only the supplied TRAIN evidence. Identify a baseline failure mechanism and propose "
    "a sparse instruction edit that addresses it. Preserve the user's task, native agent loop, tools, "
    "and native scoring. Do not supply a replacement system prompt, task-specific answers, "
    "task-ID rules, evaluator-private categories, or changes to the benchmark. Cite supporting "
    "training task IDs. Use instruction='' and supporting_task_ids=[] when no edit is justified. "
    "Previous candidate outcomes are training feedback; validation and test are unavailable."
)

JSONCallback = Callable[[dict[str, Any]], Mapping[str, Any]]


def _copy(value: Any) -> Any:
    return json.loads(residual.canonical_json(value))


def _fields(value: Any, required: set[str], label: str, optional: set[str] | None = None):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if not required <= set(value) or set(value) - required - (optional or set()):
        raise ValueError(f"{label} fields differ from the callback contract")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = _copy(value)
    result[field] = residual.canonical_hash(result)
    return result


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != residual.canonical_hash(unsigned):
        raise ValueError(f"{field} mismatch")


def _identity(value: Mapping[str, str], *, executor: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or not {"model"} <= set(value):
        raise ValueError("a measured model identity is required")
    if executor and "runtime" not in value:
        raise ValueError("the execution identity must name the runtime")
    for key, item in value.items():
        _text(key, "identity key")
        _text(item, "identity value")
    return _copy(value)


def implementation_provenance() -> dict[str, Any]:
    """Return content-addressed provenance, including untracked working-tree sources."""
    sources = {
        "phiagent/harness/codex_chisel.py": __file__,
        "phiagent/harness/benchmark_partition.py": benchmark_partition.__file__,
        "phiagent/learning/residual_meta_harness.py": residual.__file__,
        "phiagent/learning/residual_meta_training.py": training.__file__,
    }
    return {
        "method": "CHISEL",
        "implementation": "PhiAgent train-only Codex residual-routing extension",
        "method_identity_canonical_sha256":
            "91a0200836d6e6548699bc581dace31000c0cec6fcfe775fcee1623d90c72521",
        "inspected_checkout_base_revision": "a468ee05b5c842d0fcd6d8af241f7b312c2e42f3",
        "working_tree_file_hashes_are_authoritative": True,
        "inspected_design_sources": dict(_DESIGN_SOURCES),
        "executable_sources": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in sources.items()
        },
        "reused_trainer": "phiagent.learning.residual_meta_training.fit_cross_fitted_linear_head",
        "cross_fit_unit": "independent unit; questions/repetitions averaged within unit",
        "cross_fit_scope": "routing head only; compiler uses TRAIN only",
        "inference_router_model_calls": 0,
        "max_residuals_per_task": 1,
    }


@dataclass(frozen=True, slots=True)
class ChiselConfig:
    optimization_iterations: int = 2
    crossfit_folds: int = 2
    minimum_override_support: int = 2
    ridge: float = 1.0
    maximum_oof_harmful_rate: float = 0.0
    max_residual_characters: int = 1200
    max_residual_words: int = 180
    application: str = "prompt_suffix"
    split_seed: int = SPLIT_SEED
    execution_seeds: tuple[int, ...] = (SPLIT_SEED,)
    test_execution_seeds: tuple[int, ...] | None = None
    executor_workers: int = 1
    native_metric: str = "native_score"
    score_minimum: float = 0.0
    score_maximum: float = 1.0
    cumulative_token_limit: None = None
    optimizer_max_units: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "optimization_iterations", "crossfit_folds", "minimum_override_support",
            "max_residual_characters", "max_residual_words",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.crossfit_folds < 2 or self.minimum_override_support < 2:
            raise ValueError("cross-fitting and override support require at least two units")
        if self.optimizer_max_units is not None and (
            type(self.optimizer_max_units) is not int or self.optimizer_max_units < 1
        ):
            raise ValueError("optimizer_max_units must be a positive integer or None")
        if type(self.split_seed) is not int or self.split_seed != SPLIT_SEED:
            raise ValueError("the frozen split seed must be 20260928")
        if self.test_execution_seeds is None:
            object.__setattr__(self, "test_execution_seeds", self.execution_seeds)
        for name in ("execution_seeds", "test_execution_seeds"):
            seeds = getattr(self, name)
            if (
                not isinstance(seeds, tuple) or not seeds
                or any(type(seed) is not int for seed in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValueError(f"{name} must be a nonempty unique integer tuple")
        if type(self.executor_workers) is not int or not 1 <= self.executor_workers <= MAX_EXECUTOR_WORKERS:
            raise ValueError(f"executor_workers must be an integer in [1, {MAX_EXECUTOR_WORKERS}]")
        for name in ("ridge", "maximum_oof_harmful_rate", "score_minimum", "score_maximum"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.ridge <= 0 or not 0 <= self.maximum_oof_harmful_rate <= 1:
            raise ValueError("invalid ridge or harmful-rate gate")
        if self.score_maximum <= self.score_minimum:
            raise ValueError("native score bounds must be ordered")
        _text(self.native_metric, "native metric")
        if self.application not in {"prompt_suffix", "agents_md", "system_section"}:
            raise ValueError("only prompt_suffix, task-local agents_md or SDK system_section is supported")
        if self.cumulative_token_limit is not None:
            raise ValueError("cumulative token usage is unlimited")


def _config(document: Mapping[str, Any]) -> ChiselConfig:
    value = dict(document)
    value["execution_seeds"] = tuple(value["execution_seeds"])
    if value.get("test_execution_seeds") is not None:
        value["test_execution_seeds"] = tuple(value["test_execution_seeds"])
    return ChiselConfig(**value)


def _validate_optimizer_evidence(
    records: Any, config: ChiselConfig, run_id: str,
) -> None:
    if not isinstance(records, list) or len(records) != config.optimization_iterations:
        raise ValueError("sampled optimizer provenance requires every declared iteration")
    exposed: set[str] = set()
    for iteration, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("optimizer evidence provenance must contain JSON objects")
        _verify_seal(record, "evidence_selection_sha256")
        units, tasks = record["unit_ids"], record["task_ids"]
        for values in (units, tasks):
            if (
                not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value for value in values)
                or sorted(set(values)) != values
            ):
                raise ValueError("optimizer evidence identities must be sorted and unique")
        if (
            record["iteration"] != iteration or record["run_id"] != run_id
            or record["split"] != "train" or record["split_seed"] != config.split_seed
            or record["optimizer_max_units"] != config.optimizer_max_units
            or record["selection_rule"] != OPTIMIZER_UNIT_SELECTION_RULE
            or record["callback_invoked"] is not True
            or record["execution_seeds"] != list(config.execution_seeds)
            or record["previously_exposed_task_ids"] != sorted(exposed)
            or len(units) > config.optimizer_max_units or len(tasks) < len(units)
            or record["unit_count"] != len(units) or record["task_count"] != len(tasks)
            or record["observation_count"] != len(tasks) * len(config.execution_seeds)
            or record["unit_ids_sha256"] != residual.canonical_hash(units)
            or record["task_ids_sha256"] != residual.canonical_hash(tasks)
        ):
            raise ValueError("optimizer evidence provenance disagrees with the frozen sampling plan")
        for field in ("request_id", "task_commitments_sha256", "training_evidence_sha256"):
            if not isinstance(record[field], str) or not _SHA256.fullmatch(record[field]):
                raise ValueError("optimizer evidence provenance requires SHA-256 commitments")
        exposed.update(tasks)


def _feature_schema() -> dict[str, residual.FeatureSpec]:
    numeric = residual.FeatureKind.NUMERIC
    return {
        "question_character_count": residual.FeatureSpec(
            numeric, scale=4096.0, maximum=1_000_000_000,
        ),
        "question_token_count": residual.FeatureSpec(
            numeric, scale=512.0, maximum=1_000_000_000,
        ),
        "question_has_quoted_span": residual.FeatureSpec(numeric),
        "question_has_relation_cue": residual.FeatureSpec(numeric),
        "expected_answer_numeric": residual.FeatureSpec(numeric),
    }


def public_text_features(public_prompt: str) -> dict[str, float]:
    """Extract only predeclared lexical features; IDs and evaluator fields are not inputs."""
    _text(public_prompt, "public prompt")
    return {
        "question_character_count": float(len(public_prompt)),
        "question_token_count": float(len(re.findall(r"\w+|[^\w\s]", public_prompt))),
        "question_has_quoted_span": float(bool(re.search(r"""["'“”]""", public_prompt))),
        "question_has_relation_cue": float(bool(re.search(
            r"\b(before|after|between|within|left|right|above|below)\b", public_prompt, re.I,
        ))),
        "expected_answer_numeric": float(bool(re.search(
            r"\b(how many|how much|count|calculate|number|total|sum)\b", public_prompt, re.I,
        ))),
    }


def _usage(value: Any) -> dict[str, int | float | None]:
    _fields(value, set(TOKEN_FIELDS), "usage", {"cost_usd"})
    result = {field: value.get(field) for field in USAGE_FIELDS}
    for field in TOKEN_FIELDS:
        if result[field] is not None and (
            type(result[field]) is not int or result[field] < 0
        ):
            raise ValueError(f"{field} must be a measured nonnegative integer or null")
    cost = result["cost_usd"]
    if cost is not None and (
        type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
    ):
        raise ValueError("cost_usd must be a measured nonnegative number or null")
    incoming, outgoing, total = (result[name] for name in (
        "input_tokens", "output_tokens", "total_tokens",
    ))
    if all(item is not None for item in (incoming, outgoing, total)):
        if total != incoming + outgoing:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
    for field in ("cache_read_tokens", "cache_write_tokens"):
        if incoming is not None and result[field] is not None and result[field] > incoming:
            raise ValueError("cached tokens are included in input, not additional tokens")
    return result


def measured_usage_from_receipt(provider_usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize native metering counters without estimating missing or partial totals."""
    if provider_usage is None:
        return dict.fromkeys(USAGE_FIELDS)
    if not isinstance(provider_usage, Mapping):
        raise ValueError("provider usage must be an object or null")
    if provider_usage.get("usage_status") != "MEASURED":
        return dict.fromkeys(USAGE_FIELDS)
    known = provider_usage.get("known_usage")
    if not isinstance(known, Mapping) or provider_usage.get("unresolved_call_ids"):
        raise ValueError("MEASURED provider usage requires resolved native counters")
    return _usage({
        "input_tokens": known.get("input_tokens"),
        "output_tokens": known.get("output_tokens"),
        "total_tokens": known.get("total_tokens"),
        "cache_read_tokens": provider_usage.get("cached_input_tokens"),
        "cache_write_tokens": provider_usage.get("cache_write_tokens"),
        "cost_usd": provider_usage.get("cost_usd"),
    })


def execution_result_from_receipt(
    request: Mapping[str, Any], receipt: Mapping[str, Any], *,
    native_score: float | None, success: bool | None = None, feedback: str | None = None,
) -> dict[str, Any]:
    """Adapt a durable common-executor receipt; this does not invoke an executor or scorer."""
    if not isinstance(request, Mapping) or not {
        "request_id", "identity", "task", "phase",
    } <= set(request):
        raise ValueError("an identified controller execution request is required")
    if not isinstance(receipt, Mapping):
        raise ValueError("the common executor receipt must be an object")
    statuses = {
        "COMPLETED": "completed", "BLOCKED": "blocked", "SCORED": "completed",
        "FAILED": "failed", "TIMEOUT": "timeout",
    }
    status = receipt.get("status")
    if status not in statuses:
        raise ValueError("the executor must return a terminal status")
    if native_score is not None and (
        type(native_score) not in (int, float) or not math.isfinite(native_score)
        or not 0 <= native_score <= 1
    ):
        raise ValueError("supply the fractional native score in [0, 1]")
    if status == "SCORED" and native_score is None:
        raise ValueError("SCORED receipts require an explicit native score")
    if success is not None and type(success) is not bool:
        raise ValueError("success must come explicitly from the native scorer")
    identity = _identity(request["identity"], executor=True)
    observed_model = receipt.get("model")
    if observed_model is not None:
        _text(observed_model, "executor model")
        identity["model"] = observed_model
    provider = receipt.get("provider_usage")
    normalized_usage = measured_usage_from_receipt(provider)
    answer = receipt.get("final_answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("final_answer must be a string or null")
    if feedback is not None and not isinstance(feedback, str):
        raise ValueError("training feedback must be a string or null")
    result = {
        "request_id": request["request_id"], "task_id": request["task"]["task_id"],
        "identity": identity, "execution_status": statuses[status],
        "native_score": native_score, "success": success, "usage": normalized_usage,
        "provenance": {
            "common_executor_receipt_sha256": residual.canonical_hash(receipt),
            "common_executor_status": status,
            "provider_usage": _copy(provider),
            "final_answer_sha256": _hash_text(answer) if answer is not None else None,
            "provider_sampling_seed_supported": receipt.get("provider_sampling_seed_supported"),
        },
    }
    if request["phase"] == "train":
        result["feedback"] = feedback if feedback is not None else (answer or "")
    return result


def _usage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {}
    for field in USAGE_FIELDS:
        measured = [row["usage"][field] for row in records if row["usage"][field] is not None]
        subtotal = math.fsum(measured) if field == "cost_usd" else sum(measured)
        missing = len(records) - len(measured)
        totals[field] = {
            "total": None if missing else subtotal,
            "measured_subtotal": subtotal,
            "missing_calls": missing,
        }
    return {
        "calls": len(records),
        "distinct_tasks": len({row["task_id"] for row in records if row.get("task_id")}),
        "distinct_units": len({row["unit_id"] for row in records if row.get("unit_id")}),
        "usage": totals,
    }


def _route(candidate: Mapping[str, Any], public_prompt: str) -> dict[str, Any]:
    if not candidate["enabled"]:
        return {"residual_id": None, "reason": "empty_or_unadmitted_policy", "score": None}
    features = public_text_features(public_prompt)
    if any(
        not bounds[0] <= features[name] <= bounds[1]
        for name, bounds in candidate["feature_bounds"].items()
    ):
        return {"residual_id": None, "reason": "outside_training_feature_support", "score": None}
    head = candidate["head"]
    schema = _feature_schema()
    score = head["intercept"] + math.fsum(
        head["weights"][name] * schema[name].normalize(features[name])
        for name in sorted(schema)
    )
    threshold = head["score_threshold"]
    # Refitting the same boundary can differ from an OOF score by a few ULPs.
    active = score >= threshold or math.isclose(
        score, threshold, rel_tol=0.0, abs_tol=8 * max(math.ulp(score), math.ulp(threshold)),
    )
    return {
        "residual_id": candidate["id"] if active else None,
        "reason": "cross_fitted_local_advantage" if active else "below_cross_fitted_threshold",
        "score": score,
        "threshold": head["score_threshold"],
    }


def _render(public_prompt: str, instruction: str, application: str) -> dict[str, str]:
    rendered = {
        "codex_prompt": public_prompt + (
            _SUFFIX + instruction if instruction and application == "prompt_suffix" else ""
        ),
        "agents_md_append": (
            _AGENTS + instruction + "\n" if instruction and application == "agents_md" else ""
        ),
    }
    if application == "system_section":
        rendered["system_prompt_append"] = instruction
    return rendered


@dataclass(frozen=True, slots=True)
class FrozenPolicy:
    """An immutable JSON value; every exported dict is an independent copy."""

    _document: str

    def __post_init__(self) -> None:
        value = json.loads(self._document)
        if (
            value.get("schema") != SCHEMA or type(value.get("max_residuals_per_task")) is not int
            or value["max_residuals_per_task"] != 1
        ):
            raise ValueError("unsupported frozen CHISEL policy")
        _verify_seal(value, "policy_sha256")
        _verify_seal(value["candidate"], "candidate_sha256")
        if value["source_sha256"] != residual.canonical_hash(value["sources"]):
            raise ValueError("source_sha256 mismatch")
        config = _config(value["config"])
        if value["sources"]["config"] != value["config"]:
            raise ValueError("policy configuration differs from the source freeze")
        repetition_counts = {
            "training_repetitions": len(config.execution_seeds),
            "validation_repetitions": len(config.execution_seeds),
            "test_repetitions": len(config.test_execution_seeds),
        }
        if (
            value["config"].get("test_execution_seeds") != list(config.test_execution_seeds)
            or any(type(value.get(name)) is not int or value[name] != count
                   for name, count in repetition_counts.items())
            or value.get("test_pair_order_rule") != TEST_PAIR_ORDER_RULE
        ):
            raise ValueError("the frozen seed roles, repetition counts or test arm order differ")
        optimizer_evidence = value.get("optimizer_evidence")
        if config.optimizer_max_units is not None:
            _validate_optimizer_evidence(optimizer_evidence, config, value["run_id"])
            if value.get("optimizer_evidence_sha256") != residual.canonical_hash(optimizer_evidence):
                raise ValueError("optimizer_evidence_sha256 mismatch")
        candidate = value["candidate"]
        if type(candidate["enabled"]) is not bool:
            raise ValueError("the residual admission flag must be boolean")
        if candidate["id"] == "noop":
            if candidate["instruction"] or candidate["enabled"] or candidate["head"] is not None:
                raise ValueError("the empty policy cannot contain a residual")
        else:
            if (
                not candidate["enabled"] or not candidate["instruction"]
                or len(candidate["instruction"]) > config.max_residual_characters
                or len(candidate["instruction"].split()) > config.max_residual_words
                or candidate["id"] != _hash_text(candidate["instruction"])
            ):
                raise ValueError("frozen residual does not satisfy the sparse policy contract")
            head = candidate["head"]
            _verify_seal(head, "training_sha256")
            if (
                set(head["weights"]) != set(_feature_schema())
                or set(candidate["feature_bounds"]) != set(_feature_schema())
                or head["feature_schema_sha256"] != residual.feature_schema_hash(_feature_schema())
            ):
                raise ValueError("frozen routing features differ")
            if (
                candidate["origin"] != "optimizer_callback_train_only"
                or not candidate["supporting_task_ids"]
                or candidate["admission"] != "admitted_training_oof"
                or head["status"] != "DISCOVERY_CANDIDATE"
                or head["training_task_count"] != candidate["training_units"]
                or head["grouped_crossfit_folds"] != config.crossfit_folds
                or head["oof_override_support"] < config.minimum_override_support
                or head["oof_mean_utility"] <= 0
                or head["oof_harmful_rate"] > config.maximum_oof_harmful_rate
            ):
                raise ValueError("frozen residual lacks its declared training admission evidence")
            if config.optimizer_max_units is not None:
                iteration = candidate["iteration"]
                if type(iteration) is not int or not 0 <= iteration < len(optimizer_evidence):
                    raise ValueError("residual iteration is outside its optimizer evidence history")
                evidence = optimizer_evidence[iteration]
                current = set(evidence["task_ids"])
                previous = set(evidence["previously_exposed_task_ids"])
                support = set(candidate["supporting_task_ids"])
                if (
                    candidate["optimizer_evidence_selection_sha256"]
                    != evidence["evidence_selection_sha256"]
                    or not support <= current | previous
                    or candidate["supporting_current_task_ids"] != sorted(support & current)
                    or candidate["supporting_previous_task_ids"] != sorted(support - current)
                ):
                    raise ValueError("residual support was not supplied to its optimizer")
        self.verify_sources()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrozenPolicy:
        return cls(residual.canonical_json(value))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._document)

    @property
    def policy_sha256(self) -> str:
        return self.to_dict()["policy_sha256"]

    @property
    def source_sha256(self) -> str:
        return self.to_dict()["source_sha256"]

    def verify_sources(self) -> None:
        recorded = self.to_dict()["sources"]["implementation"]
        if recorded != implementation_provenance():
            raise RuntimeError("CHISEL executable sources changed after policy freeze")

    def route(self, public_prompt: str) -> dict[str, Any]:
        self.verify_sources()
        _text(public_prompt, "public prompt")
        return _route(self.to_dict()["candidate"], public_prompt)

    def apply(self, public_prompt: str) -> dict[str, Any]:
        """Return one suffix/AGENTS append, or the byte-identical public prompt."""
        document = self.to_dict()
        decision = self.route(public_prompt)
        instruction = document["candidate"]["instruction"] if decision["residual_id"] else ""
        return {
            **_render(public_prompt, instruction, document["config"]["application"]),
            "decision": decision,
        }


@dataclass(frozen=True, slots=True)
class _ExecutionCall:
    task: Mapping[str, str]
    candidate: Mapping[str, Any]
    phase: str
    arm: str
    seed: int
    force_residual: bool = False


class CodexChiselController:
    """Single-use train -> validation selection -> freeze -> paired sealed test lifecycle."""

    def __init__(
        self, *, partition: Mapping[str, Any], task_index: Sequence[Mapping[str, str]],
        execute: JSONCallback, optimize: JSONCallback, execution_identity: Mapping[str, str],
        optimizer_identity: Mapping[str, str], run_id: str, config: ChiselConfig = ChiselConfig(),
    ) -> None:
        _text(run_id, "new run identity")
        if not isinstance(config, ChiselConfig) or not callable(execute) or not callable(optimize):
            raise ValueError("a frozen configuration and both callbacks are required")
        self._partition = _copy(partition)
        benchmark_partition.validate_partition(self._partition)
        if (
            self._partition["split_seed"] != SPLIT_SEED
            or self._partition["ratios"] != {"train": 0.2, "validation": 0.2, "test": 0.6}
        ):
            raise ValueError("the required partition is 20/20/60 with seed 20260928")
        rebuilt = benchmark_partition.build_partition(
            self._partition["records"], benchmark=self._partition["benchmark"],
            seed=SPLIT_SEED, independent_unit=self._partition["independent_unit"],
            rounding=self._partition.get("rounding_policy", "exact"),
        )
        if rebuilt["splits"] != self._partition["splits"]:
            raise ValueError("partition membership is not the outcome-independent frozen assignment")
        self._index = {}
        for entry in task_index:
            _fields(entry, {"task_id", "unit_id", "public_prompt_sha256"}, "task index")
            _text(entry["task_id"], "task ID")
            _text(entry["unit_id"], "independent unit ID")
            if (
                not isinstance(entry["public_prompt_sha256"], str)
                or not _SHA256.fullmatch(entry["public_prompt_sha256"])
                or entry["task_id"] in self._index
            ):
                raise ValueError("task index requires unique tasks and valid prompt hashes")
            self._index[entry["task_id"]] = dict(entry)
        if {row["unit_id"] for row in self._index.values()} != {
            row["id"] for row in self._partition["records"]
        }:
            raise ValueError("task index must cover the complete independent-unit population")
        self._partition_sha256 = residual.canonical_hash(self._partition)
        self._task_index_sha256 = residual.canonical_hash(self._index)
        self._execute_callback, self._optimizer_callback = execute, optimize
        self._execution_identity = _identity(execution_identity, executor=True)
        if (config.application == "system_section"
                and self._execution_identity["runtime"] != "official_deepseek_harness_sdk"):
            raise ValueError("system_section requires the official DeepSeek SDK adapter")
        self._optimizer_identity = _identity(optimizer_identity, executor=False)
        self._run_id, self._config = run_id, config
        self._provenance = implementation_provenance()
        self._lock = RLock()
        self._next_sequence = 0
        self._state = "initialized"
        self._ledger: list[dict[str, Any]] = []
        self._request_ids: set[str] = set()
        self._observations: dict[str, dict[str, Any]] = {}
        self._iterations: list[dict[str, Any]] = []
        self._bank_json: str | None = None
        self._selection: dict[str, Any] | None = None
        self._policy: FrozenPolicy | None = None
        self._test_report: dict[str, Any] | None = None
        self._test_pair_orders: dict[str, dict[str, Any]] = {}
        self._executor_failures: list[dict[str, Any]] = []
        self._train_evidence_sha256: str | None = None
        self._optimizer_exposed_tasks: set[str] = set()
        self._optimizer_evidence: list[dict[str, Any]] = []
        self._noop = _seal({
            "id": "noop", "instruction": "", "enabled": False, "head": None,
            "feature_bounds": {}, "supporting_task_ids": [],
            "origin": (
                "plain_deepseek_harness"
                if self._execution_identity["runtime"] == "official_deepseek_harness_sdk"
                else "plain_codex"
            ),
            "admission": "always_admissible",
        }, "candidate_sha256")

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def candidate_bank(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._bank_json is None:
                raise RuntimeError("training has not frozen the candidate bank")
            return json.loads(self._bank_json)

    def restore_trained(self, report: Mapping[str, Any]) -> None:
        """Restore an immutable completed-TRAIN report for protected validation."""
        self._require("initialized")
        value = _copy(report)
        training_provenance = value.get("implementation_provenance")
        current_provenance = _copy(self._provenance)
        comparable_training = _copy(training_provenance)
        comparable_current = _copy(current_provenance)
        controller_source = "phiagent/harness/codex_chisel.py"
        for provenance in (comparable_training, comparable_current):
            if isinstance(provenance, dict):
                provenance.get("executable_sources", {}).pop(controller_source, None)
        if (
            value.get("schema") != SCHEMA
            or value.get("state") != "trained"
            or value.get("run_id") != self._run_id
            or not isinstance(training_provenance, dict)
            or comparable_training != comparable_current
            or value.get("selection") is not None
            or value.get("test") is not None
            or value.get("test_pair_orders") != []
            or value.get("benchmark_improvement_claim") is not None
        ):
            raise ValueError("restore requires the matching completed TRAIN report")
        bank = value.get("candidate_bank")
        ledger = value.get("ledger")
        observations = value.get("observations")
        iterations = value.get("optimization_iterations")
        optimizer_evidence = value.get("optimizer_evidence")
        failures = value.get("executor_failures")
        if (
            not isinstance(bank, list) or not bank or bank[0] != self._noop
            or not isinstance(ledger, list)
            or not isinstance(observations, dict)
            or not isinstance(iterations, list)
            or not isinstance(optimizer_evidence, list)
            or not isinstance(failures, list)
        ):
            raise ValueError("completed TRAIN report has invalid state containers")
        for candidate in bank:
            _verify_seal(candidate, "candidate_sha256")
        request_ids = [row.get("request_id") for row in ledger]
        ledger_by_request = {row.get("request_id"): row for row in ledger}
        if (
            any(not isinstance(ident, str) or not _SHA256.fullmatch(ident)
                for ident in request_ids)
            or len(request_ids) != len(set(request_ids))
            or [row.get("sequence") for row in ledger] != list(range(len(ledger)))
            or set(request_ids) != set(observations)
            or any(row.get("request_id") != ident for ident, row in observations.items())
            or any(
                ledger_by_request[ident].get("result_sha256") != residual.canonical_hash(row)
                for ident, row in observations.items()
            )
            or value.get("optimizer_evidence_sha256")
            != residual.canonical_hash(optimizer_evidence)
        ):
            raise ValueError("completed TRAIN request ledger is not exact-once")
        with self._lock:
            self._bank_json = residual.canonical_json(bank)
            self._ledger = ledger
            self._request_ids = set(request_ids)
            self._observations = observations
            self._iterations = iterations
            self._optimizer_evidence = optimizer_evidence
            self._optimizer_exposed_tasks = {
                task_id for row in optimizer_evidence for task_id in row["task_ids"]
            }
            self._executor_failures = failures
            self._train_evidence_sha256 = residual.canonical_hash(observations)
            self._next_sequence = len(ledger)
            self._state = "trained"
        if self.accounting() != value.get("accounting"):
            raise ValueError("completed TRAIN accounting does not match its ledger")
        restored = self.report()
        value["implementation_provenance"] = current_provenance
        if restored != value:
            raise ValueError("completed TRAIN report cannot be restored exactly")

    def _require(self, state: str) -> None:
        with self._lock:
            if self._state != state:
                raise RuntimeError(f"requires state {state}; current state is {self._state}")
        self._verify_sources()

    def _begin(self, expected: str, state: str) -> None:
        with self._lock:
            self._require(expected)
            self._state = state

    def _verify_sources(self) -> None:
        if self._provenance != implementation_provenance():
            raise RuntimeError("CHISEL sources changed during this run")

    def _tasks(self, tasks: Sequence[Mapping[str, str]], phase: str) -> list[dict[str, str]]:
        units = set(self._partition["splits"][phase])
        expected = {key for key, row in self._index.items() if row["unit_id"] in units}
        seen: dict[str, dict[str, str]] = {}
        for task in tasks:
            _fields(task, {"task_id", "unit_id", "public_prompt"}, "public task")
            key = task["task_id"]
            if not isinstance(key, str) or key not in expected or key in seen:
                raise ValueError(f"duplicate or non-{phase} task")
            _text(task["public_prompt"], "public prompt")
            entry = self._index[key]
            if (
                task["unit_id"] != entry["unit_id"]
                or _hash_text(task["public_prompt"]) != entry["public_prompt_sha256"]
            ):
                raise ValueError("task unit or public prompt differs from its frozen commitment")
            seen[key] = dict(task)
        if set(seen) != expected:
            raise ValueError(f"incomplete {phase} task population")
        return [seen[key] for key in sorted(seen)]

    def _invoke(
        self, callback: JSONCallback, request: dict[str, Any], kind: str, *,
        ledger_sequence: int | None = None,
        evidence_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._verify_sources()
        request["request_id"] = residual.canonical_hash(request)
        task = request.get("task", {})
        record = {
            "request_id": request["request_id"], "kind": kind, "phase": request["phase"],
            "arm": request.get("arm", "optimizer"), "task_id": task.get("task_id"),
            "unit_id": task.get("unit_id"), "seed": request.get("seed"),
            "execution_status": "pending", "usage": dict.fromkeys(USAGE_FIELDS),
            "pair_id": request.get("pair_id"),
            "candidate_sha256": request.get("candidate_sha256"),
            "final_policy_sha256": request.get("final_policy_sha256"),
            "source_sha256": request.get("source_sha256"),
        }
        if kind == "optimizer":
            evidence = request["training_evidence"]
            record.update(
                evidence_observations=len(evidence),
                evidence_tasks=len({row["task"]["task_id"] for row in evidence}),
                evidence_units=len({row["task"]["unit_id"] for row in evidence}),
            )
        with self._lock:
            if request["request_id"] in self._request_ids:
                raise RuntimeError("duplicate callback request would double-charge evidence")
            self._request_ids.add(request["request_id"])
            if ledger_sequence is None:
                ledger_sequence = self._next_sequence
                self._next_sequence += 1
            record["sequence"] = ledger_sequence
            self._ledger.append(record)
            if kind == "optimizer":
                self._optimizer_exposed_tasks.update(row["task"]["task_id"] for row in evidence)
                if evidence_manifest is not None:
                    manifest = _seal({
                        **evidence_manifest, "request_id": request["request_id"],
                        "callback_invoked": True,
                    }, "evidence_selection_sha256")
                    self._optimizer_evidence.append(manifest)
                    record["evidence_selection_sha256"] = manifest["evidence_selection_sha256"]
            if request["phase"] == "test":
                self._test_pair_orders[request["pair_id"]]["actual_order"].append(request["arm"])
        try:
            result = callback(_copy(request))
            if isinstance(result, Mapping) and "usage" in result:
                measured = _usage(result["usage"])
                with self._lock:
                    record["usage"] = measured
            common = {"request_id", "identity", "execution_status", "usage"}
            required = (
                {"task_id", "native_score", "success"} if kind == "executor"
                else {"instruction", "supporting_task_ids"}
            )
            _fields(result, common | required, f"{kind} result", (
                {"feedback", "provenance"} if kind == "executor" else {"provenance"}
            ))
            result = _copy(result)
            if result["request_id"] != request["request_id"] or result["identity"] != request["identity"]:
                raise ValueError("callback request or actual model/runtime identity changed")
            if result["execution_status"] not in _STATUSES:
                raise ValueError("unsupported execution status")
            if "provenance" in result and not isinstance(result["provenance"], dict):
                raise ValueError("callback provenance must be a JSON object")
            if kind == "executor":
                if result["task_id"] != task["task_id"]:
                    raise ValueError("execution result belongs to another task")
                score = result["native_score"]
                if score is not None and (
                    type(score) not in (int, float) or not math.isfinite(score)
                    or not self._config.score_minimum <= score <= self._config.score_maximum
                ):
                    raise ValueError("native score is outside the predeclared metric range")
                if result["success"] is not None and type(result["success"]) is not bool:
                    raise ValueError("success must be a native boolean or null")
                if "feedback" in result and not isinstance(result["feedback"], str):
                    raise ValueError("feedback must be a string")
            elif (
                not isinstance(result["instruction"], str)
                or not isinstance(result["supporting_task_ids"], list)
                or any(not isinstance(key, str) for key in result["supporting_task_ids"])
            ):
                raise ValueError("optimizer must return one instruction and a list of training IDs")
            result_hash = residual.canonical_hash(result)
            if request["phase"] != "train":
                result.pop("feedback", None)
            with self._lock:
                record["execution_status"] = result["execution_status"]
                record["result_sha256"] = result_hash
                result["usage"] = dict(record["usage"])
                self._observations[request["request_id"]] = _copy(result)
            self._verify_sources()
            return result
        except Exception as error:
            with self._lock:
                record["execution_status"] = "callback_error"
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                self._state = "blocked"
            raise

    def _execute(
        self, task: Mapping[str, str], candidate: Mapping[str, Any], *, phase: str,
        arm: str, seed: int, force_residual: bool = False,
        ledger_sequence: int | None = None,
    ) -> dict[str, Any]:
        decision = (
            {"residual_id": candidate["id"], "reason": "paired_training_intervention", "score": None}
            if force_residual else _route(candidate, task["public_prompt"])
        )
        instruction = candidate["instruction"] if decision["residual_id"] else ""
        request = {
            "schema": SCHEMA, "run_id": self._run_id, "phase": phase, "arm": arm,
            "task": dict(task), "seed": seed, "identity": self._execution_identity,
            "pair_id": residual.canonical_hash([self._run_id, phase, task["task_id"], seed]),
            "policy_sha256": candidate["candidate_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "partition_sha256": self._partition_sha256,
            "task_index_sha256": self._task_index_sha256,
            "final_policy_sha256": self._policy.policy_sha256 if phase == "test" else None,
            "source_sha256": self._policy.source_sha256 if phase == "test" else None,
            **_render(task["public_prompt"], instruction, self._config.application),
            "residual": ({
                "id": candidate["id"], "instruction": instruction,
                "application": self._config.application,
            } if instruction else None),
            "contract": {
                "runtime": (
                    "unmodified_official_deepseek_harness_sdk"
                    if self._execution_identity["runtime"] == "official_deepseek_harness_sdk"
                    else "unmodified_official_codex_cli"
                ),
                "fresh_task_workspace": True,
                "max_residuals_per_task": 1, "extra_inference_model_calls": 0,
                "cumulative_token_limit": None, "native_metric": self._config.native_metric,
            },
        }
        result = self._invoke(
            self._execute_callback, request, "executor", ledger_sequence=ledger_sequence,
        )
        return {"task_id": task["task_id"], "unit_id": task["unit_id"], "seed": seed,
                "decision": decision, "result": result}

    def _execute_groups(
        self, groups: Sequence[Sequence[_ExecutionCall]],
    ) -> list[list[dict[str, Any]]]:
        """Bound in-flight groups; run each group's calls in order and drain on failure."""
        results: list[list[dict[str, Any]]] = [[] for _ in groups]
        sequences = []
        with self._lock:
            for group in groups:
                sequences.append(list(range(self._next_sequence, self._next_sequence + len(group))))
                self._next_sequence += len(group)
        failures: dict[int, Exception] = {}

        def run_group(index: int) -> None:
            for call, sequence in zip(groups[index], sequences[index], strict=True):
                results[index].append(self._execute(
                    call.task, call.candidate, phase=call.phase, arm=call.arm, seed=call.seed,
                    force_residual=call.force_residual, ledger_sequence=sequence,
                ))

        def failed(index: int, error: Exception) -> None:
            failures[index] = error
            position = len(results[index])
            call = groups[index][position]
            with self._lock:
                self._state = "blocked"
                self._executor_failures.append({
                    "sequence": sequences[index][position], "phase": call.phase,
                    "arm": call.arm, "task_id": call.task["task_id"], "unit_id": call.task["unit_id"],
                    "seed": call.seed, "error": {"type": type(error).__name__, "message": str(error)},
                })

        if self._config.executor_workers == 1:
            for index in range(len(groups)):
                try:
                    run_group(index)
                except Exception as error:
                    failed(index, error)
                    raise
            return results
        with ThreadPoolExecutor(max_workers=self._config.executor_workers) as pool:
            cursor = 0
            active = {}
            while cursor < len(groups) and len(active) < self._config.executor_workers:
                active[pool.submit(run_group, cursor)] = cursor
                cursor += 1
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: active[item]):
                    index = active.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        failed(index, error)
                if not failures and self.state != "blocked":
                    while cursor < len(groups) and len(active) < self._config.executor_workers:
                        active[pool.submit(run_group, cursor)] = cursor
                        cursor += 1
        if failures:
            raise failures[min(failures)]
        return results

    def _execution_rows(
        self, tasks: Sequence[Mapping[str, str]], candidate: Mapping[str, Any], *,
        phase: str, arm: str, force_residual: bool = False,
    ) -> list[dict[str, Any]]:
        seeds = self._config.test_execution_seeds if phase == "test" else self._config.execution_seeds
        groups = [
            [_ExecutionCall(task, candidate, phase, arm, seed, force_residual)]
            for task in tasks for seed in seeds
        ]
        return [row for group in self._execute_groups(groups) for row in group]

    def _fit_candidate(
        self, proposal: dict[str, Any], tasks: list[dict[str, str]],
        baseline: dict[tuple[str, int], dict[str, Any]],
        observed: dict[tuple[str, int], dict[str, Any]],
    ) -> dict[str, Any]:
        features = {task["task_id"]: public_text_features(task["public_prompt"]) for task in tasks}
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for task in tasks:
            groups[task["unit_id"]].append(task)
        admission = "insufficient_independent_support"
        head = None
        complete = all(
            row["result"]["native_score"] is not None
            and row["result"]["usage"]["total_tokens"] is not None
            for rows in (baseline, observed) for row in rows.values()
        )
        if not complete:
            admission = "incomplete_measured_paired_evidence"
        elif len(groups) >= max(self._config.crossfit_folds, self._config.minimum_override_support):
            examples = []
            for unit, members in sorted(groups.items()):
                pairs = [
                    (baseline[task["task_id"], seed]["result"],
                     observed[task["task_id"], seed]["result"])
                    for task in members for seed in self._config.execution_seeds
                ]
                differences = [right["native_score"] - left["native_score"] for left, right in pairs]
                examples.append({
                    "task_id": unit,
                    "features": {
                        name: math.fsum(features[task["task_id"]][name] for task in members) / len(members)
                        for name in sorted(_feature_schema())
                    },
                    "utility": math.fsum(differences) / len(pairs) / (
                        self._config.score_maximum - self._config.score_minimum
                    ),
                    "harmful": any(value < 0 for value in differences),
                    "token_delta": math.fsum(
                        max(-1.0, min(1.0, (
                            right["usage"]["total_tokens"] - left["usage"]["total_tokens"]
                        ) / max(1, left["usage"]["total_tokens"])))
                        for left, right in pairs
                    ) / len(pairs),
                })
            head = training.fit_cross_fitted_linear_head(
                examples, feature_schema=_feature_schema(), folds=self._config.crossfit_folds,
                seed=SPLIT_SEED, ridge=self._config.ridge,
                minimum_override_support=self._config.minimum_override_support,
            )
            admission = (
                "admitted_training_oof" if head["status"] == "DISCOVERY_CANDIDATE"
                and head["oof_harmful_rate"] <= self._config.maximum_oof_harmful_rate
                else "no_admissible_positive_oof_advantage"
            )
        return _seal({
            **proposal, "head": head, "enabled": admission == "admitted_training_oof",
            "admission": admission, "training_units": len(groups), "training_tasks": len(tasks),
            "paired_training_observations": len(observed),
            "paired_evidence_sha256": residual.canonical_hash({
                "baseline": [baseline[key] for key in sorted(baseline)],
                "candidate": [observed[key] for key in sorted(observed)],
            }),
            "feature_bounds": {
                name: [min(row[name] for row in features.values()),
                       max(row[name] for row in features.values())]
                for name in sorted(_feature_schema())
            },
        }, "candidate_sha256")

    def _optimizer_tasks(
        self, tasks: list[dict[str, str]], iteration: int,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        units = {task["unit_id"] for task in tasks}
        cap = self._config.optimizer_max_units
        if cap is not None:
            units = set(sorted(units, key=lambda unit: (
                residual.canonical_hash([self._run_id, self._config.split_seed, iteration, unit]),
                unit,
            ))[:cap])
        selected = [task for task in tasks if task["unit_id"] in units]
        unit_ids = sorted(units)
        task_ids = [task["task_id"] for task in selected]
        with self._lock:
            previous = sorted(self._optimizer_exposed_tasks)
        return selected, {
            "iteration": iteration, "split": "train", "run_id": self._run_id,
            "split_seed": self._config.split_seed, "optimizer_max_units": cap,
            "selection_rule": OPTIMIZER_UNIT_SELECTION_RULE if cap is not None else "all_train_units",
            "unit_ids": unit_ids, "task_ids": task_ids,
            "unit_ids_sha256": residual.canonical_hash(unit_ids),
            "task_ids_sha256": residual.canonical_hash(task_ids),
            "task_commitments_sha256": residual.canonical_hash([self._index[key] for key in task_ids]),
            "unit_count": len(unit_ids), "task_count": len(task_ids),
            "observation_count": len(task_ids) * len(self._config.execution_seeds),
            "execution_seeds": list(self._config.execution_seeds),
            "previously_exposed_task_ids": previous,
        }

    def train(self, tasks: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        self._require("initialized")
        tasks = self._tasks(tasks, "train")
        self._begin("initialized", "training")
        baseline = {
            (row["task_id"], row["seed"]): row
            for row in self._execution_rows(tasks, self._noop, phase="train", arm="baseline")
        }
        candidates: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        train_ids = {task["task_id"] for task in tasks}
        for iteration in range(self._config.optimization_iterations):
            optimizer_tasks, evidence_manifest = self._optimizer_tasks(tasks, iteration)
            request = {
                "schema": SCHEMA, "run_id": self._run_id, "phase": "train",
                "iteration": iteration, "identity": self._optimizer_identity,
                "objective": _COMPILER_OBJECTIVE, "native_metric": self._config.native_metric,
                "limits": {
                    "max_residuals_per_task": 1,
                    "max_characters": self._config.max_residual_characters,
                    "max_words": self._config.max_residual_words,
                    "optimization_iterations": self._config.optimization_iterations,
                    "cumulative_token_limit": None,
                },
                "training_evidence": [
                    {"task": task, "seed": seed, "baseline": baseline[task["task_id"], seed]["result"],
                     "candidates": {
                         key: rows[task["task_id"], seed]["result"]
                         for key, rows in sorted(evidence.items())
                     }}
                    for task in optimizer_tasks for seed in self._config.execution_seeds
                ],
                "previous_candidates": [candidates[key] for key in sorted(candidates)],
            }
            evidence_manifest["training_evidence_sha256"] = residual.canonical_hash(
                request["training_evidence"],
            )
            if self._config.optimizer_max_units is not None:
                request["limits"]["optimizer_max_units"] = self._config.optimizer_max_units
                request["evidence_selection"] = {
                    key: evidence_manifest[key] for key in (
                        "selection_rule", "unit_ids", "task_ids", "unit_count", "task_count",
                        "observation_count", "previously_exposed_task_ids",
                    )
                }
                request["previous_candidates"] = [
                    {field: candidate[field] for field in (
                        "id", "instruction", "supporting_task_ids", "origin", "iteration",
                        "optimizer_request_id", "candidate_sha256",
                    )}
                    for candidate in request["previous_candidates"]
                ]
            try:
                response = self._invoke(
                    self._optimizer_callback, request, "optimizer", evidence_manifest=evidence_manifest,
                )
            except Exception:
                with self._lock:
                    self._state = "blocked"
                raise
            instruction = response["instruction"].strip()
            support = response["supporting_task_ids"]
            evidence_record = self._optimizer_evidence[-1]
            entry = {
                "iteration": iteration, "request_id": response["request_id"],
                "evidence_selection_sha256": evidence_record["evidence_selection_sha256"],
            }
            self._iterations.append(entry)
            if (
                len(set(support)) != len(support) or not set(support) <= train_ids
                or not set(support) <= self._optimizer_exposed_tasks
            ):
                self._state = "blocked"
                raise ValueError("optimizer residual support must contain TRAIN task IDs only, "
                                 "from current or previously supplied evidence")
            if response["execution_status"] != "completed" or not instruction:
                entry["status"] = "optimizer_abstained_or_failed"
                continue
            if not support:
                self._state = "blocked"
                raise ValueError("optimizer residual support must contain TRAIN task IDs only")
            if (
                len(instruction) > self._config.max_residual_characters
                or len(instruction.split()) > self._config.max_residual_words
                or "\0" in instruction or _SUFFIX.strip() in instruction or _AGENTS.strip() in instruction
            ):
                entry["status"] = "rejected_nonsparse_or_nested_instruction"
                continue
            key = _hash_text(instruction)
            entry["candidate_id"] = key
            if key in candidates:
                entry["status"] = "duplicate_reused_training_evidence"
                continue
            proposal = {
                "id": key, "instruction": instruction, "supporting_task_ids": sorted(support),
                "origin": "optimizer_callback_train_only",
                "optimizer_result_sha256": residual.canonical_hash(response),
                "optimizer_request_id": response["request_id"], "iteration": iteration,
            }
            if self._config.optimizer_max_units is not None:
                current_ids = set(evidence_record["task_ids"])
                proposal.update(
                    optimizer_evidence_selection_sha256=evidence_record["evidence_selection_sha256"],
                    supporting_current_task_ids=sorted(set(support) & current_ids),
                    supporting_previous_task_ids=sorted(set(support) - current_ids),
                )
            intervention = _seal({**proposal, "enabled": False}, "candidate_sha256")
            observed = {
                (row["task_id"], row["seed"]): row
                for row in self._execution_rows(
                    tasks, intervention, phase="train", arm=key, force_residual=True,
                )
            }
            evidence[key] = observed
            candidates[key] = self._fit_candidate(proposal, tasks, baseline, observed)
            entry["status"] = candidates[key]["admission"]
        with self._lock:
            self._bank_json = residual.canonical_json([
                self._noop, *(candidates[key] for key in sorted(candidates)),
            ])
            self._train_evidence_sha256 = residual.canonical_hash(self._observations)
            self._state = "trained"
        self._require("trained")
        return self.report()

    def _batch(
        self, tasks: list[dict[str, str]], candidate: Mapping[str, Any], phase: str, arm: str,
    ) -> dict[str, Any]:
        rows = self._execution_rows(tasks, candidate, phase=phase, arm=arm)
        return self._summarize(rows, candidate)

    def _summarize(
        self, rows: list[dict[str, Any]], candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        complete = all(row["result"]["native_score"] is not None for row in rows)
        totals = [row["result"]["usage"]["total_tokens"] for row in rows]
        return {
            "candidate_id": candidate["id"], "candidate_sha256": candidate["candidate_sha256"],
            "complete_native_scores": complete,
            "native_metric": self._config.native_metric,
            "mean_native_score": (
                math.fsum(row["result"]["native_score"] for row in rows) / len(rows)
                if complete else None
            ),
            "tasks": len({row["task_id"] for row in rows}),
            "independent_units": len({row["unit_id"] for row in rows}),
            "execution_seeds": list(dict.fromkeys(row["seed"] for row in rows)),
            "repetitions": len({row["seed"] for row in rows}),
            "observations": len(rows),
            "residual_applications": sum(row["decision"]["residual_id"] is not None for row in rows),
            "measured_total_tokens": sum(totals) if all(item is not None for item in totals) else None,
            "rows": rows,
        }

    def select_validation(self, tasks: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        self._require("trained")
        tasks = self._tasks(tasks, "validation")
        self._begin("trained", "selecting")
        bank = self.candidate_bank
        eligible = [candidate for candidate in bank if candidate["id"] == "noop" or candidate["enabled"]]
        results = [
            self._batch(tasks, candidate, "validation", candidate["id"])
            for candidate in eligible
        ]
        baseline = results[0]
        contenders = [
            row for row in results[1:]
            if baseline["complete_native_scores"] and row["complete_native_scores"]
            and row["residual_applications"] > 0
            and row["mean_native_score"] > baseline["mean_native_score"] + 1e-12
        ]
        selected = min(contenders, key=lambda row: (
            -row["mean_native_score"],
            row["measured_total_tokens"] if row["measured_total_tokens"] is not None else math.inf,
            row["candidate_sha256"],
        )) if contenders else baseline
        self._selection = {
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_sha256": selected["candidate_sha256"],
            "bank_sha256": residual.canonical_hash(bank),
            "rule": "complete native mean; strict improvement over noop; then tokens and hash",
            "reason": "validation_native_improvement" if contenders else "retain_empty_baseline",
            "validation_used_for_optimization": False,
            "test_opened": False, "results": results,
        }
        self._state = "selected"
        self._require("selected")
        return _copy(self._selection)

    def freeze(self, *, source_hashes: Mapping[str, str]) -> FrozenPolicy:
        self._require("selected")
        is_deepseek = self._execution_identity["runtime"] == "official_deepseek_harness_sdk"
        runtime_source = "deepseek_harness" if is_deepseek else "codex_cli"
        if not isinstance(source_hashes, Mapping) or not {
            runtime_source, "executor", "optimizer", "scorer",
        } <= set(source_hashes):
            raise ValueError(
                f"freeze requires {runtime_source}, executor, optimizer and scorer source hashes"
            )
        for name, digest in source_hashes.items():
            _text(name, "source identity")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError("every frozen external source requires a SHA-256")
        sources = {
            "implementation": self._provenance, "external": dict(source_hashes),
            "execution_identity": self._execution_identity,
            "optimizer_identity": self._optimizer_identity,
            "config": asdict(self._config),
        }
        candidate = next(row for row in self.candidate_bank
                         if row["id"] == self._selection["selected_candidate_id"])
        selection = {
            key: value for key, value in self._selection.items() if key != "results"
        }
        selection["validation_results_sha256"] = residual.canonical_hash(self._selection["results"])
        sampled_provenance = ({
            "optimizer_evidence": self._optimizer_evidence,
            "optimizer_evidence_sha256": residual.canonical_hash(self._optimizer_evidence),
        } if self._config.optimizer_max_units is not None else {})
        self._policy = FrozenPolicy.from_dict(_seal({
            "schema": SCHEMA, "run_id": self._run_id,
            "method": "CHISEL+official-DeepSeek-Harness" if is_deepseek else "CHISEL+official-Codex",
            "config": asdict(self._config), "max_residuals_per_task": 1,
            "training_repetitions": len(self._config.execution_seeds),
            "validation_repetitions": len(self._config.execution_seeds),
            "test_repetitions": len(self._config.test_execution_seeds),
            "test_pair_order_rule": TEST_PAIR_ORDER_RULE,
            "partition_sha256": self._partition_sha256,
            "task_index_sha256": self._task_index_sha256,
            "baseline_policy_sha256": self._noop["candidate_sha256"], "candidate": candidate,
            "training_evidence_sha256": self._train_evidence_sha256, "selection": selection,
            **sampled_provenance,
            "sources": sources, "source_sha256": residual.canonical_hash(sources),
            "test_opened": False,
        }, "policy_sha256"))
        self._state = "frozen"
        return self._policy

    def evaluate_test(
        self, load_tasks: Callable[[], Sequence[Mapping[str, str]]], *,
        policy_sha256: str, source_sha256: str,
    ) -> dict[str, Any]:
        self._require("frozen")
        if (
            policy_sha256 != self._policy.policy_sha256
            or source_sha256 != self._policy.source_sha256
        ):
            raise ValueError("test release requires the final policy and source seals")
        self._policy.verify_sources()
        self._begin("frozen", "testing")
        try:
            tasks = self._tasks(load_tasks(), "test")
            candidate = self._policy.to_dict()["candidate"]
            rows: dict[str, list[dict[str, Any]]] = {"baseline": [], "chisel": []}
            policies = {"baseline": self._noop, "chisel": candidate}
            groups = []
            for task in tasks:
                for seed in self._config.test_execution_seeds:
                    order = ["baseline", "chisel"]
                    digest = hashlib.sha256(residual.canonical_json([
                        self._run_id, task["task_id"], seed,
                    ]).encode("utf-8")).digest()
                    if digest[0] % 2:
                        order.reverse()
                    pair_id = residual.canonical_hash([self._run_id, "test", task["task_id"], seed])
                    with self._lock:
                        self._test_pair_orders[pair_id] = {
                            "pair_id": pair_id, "task_id": task["task_id"],
                            "unit_id": task["unit_id"], "seed": seed,
                            "planned_order": order, "actual_order": [],
                        }
                    groups.append([
                        _ExecutionCall(task, policies[arm], "test", arm, seed) for arm in order
                    ])
            for plan, observed in zip(groups, self._execute_groups(groups), strict=True):
                for call, row in zip(plan, observed, strict=True):
                    rows[call.arm].append(row)
            self._test_report = {
                "policy_sha256": policy_sha256, "source_sha256": source_sha256,
                "baseline": self._summarize(rows["baseline"], self._noop),
                "chisel": self._summarize(rows["chisel"], candidate),
                "selection_or_optimization_from_test": False,
                "pair_order": TEST_PAIR_ORDER_RULE,
                "pair_orders": list(self._test_pair_orders.values()),
                "test_execution_seeds": list(self._config.test_execution_seeds),
                "test_repetitions": len(self._config.test_execution_seeds),
            }
            self._policy.verify_sources()
            with self._lock:
                self._state = "complete"
        except Exception:
            with self._lock:
                self._state = "blocked"
            raise
        return _copy(self._test_report)

    def accounting(self) -> dict[str, Any]:
        with self._lock:
            ledger = _copy(sorted(self._ledger, key=lambda row: row["sequence"]))
            exposed_tasks = set(self._optimizer_exposed_tasks)
        buckets = {
            "training_executor": [row for row in ledger
                                  if row["phase"] == "train" and row["kind"] == "executor"],
            "training_optimizer": [row for row in ledger if row["kind"] == "optimizer"],
            "validation_selection": [row for row in ledger if row["phase"] == "validation"],
            "final_baseline": [row for row in ledger
                               if row["phase"] == "test" and row["arm"] == "baseline"],
            "final_chisel": [row for row in ledger
                             if row["phase"] == "test" and row["arm"] == "chisel"],
        }
        train_units = set(self._partition["splits"]["train"])
        train_tasks = [row for row in self._index.values() if row["unit_id"] in train_units]
        return {
            "cumulative_token_limit": None,
            "optimization_iterations_predeclared": self._config.optimization_iterations,
            "optimizer_max_units": self._config.optimizer_max_units,
            "optimizer_exposure_scope": "evidence supplied to invoked optimizer callbacks",
            "optimizer_sample_tasks": len(exposed_tasks),
            "optimizer_sample_units": len({
                self._index[key]["unit_id"] for key in exposed_tasks
            }),
            "optimizer_observation_exposures": sum(
                row["evidence_observations"] for row in buckets["training_optimizer"]
            ),
            "optimizer_task_exposures": sum(
                row["evidence_tasks"] for row in buckets["training_optimizer"]
            ),
            "optimizer_unit_exposures": sum(
                row["evidence_units"] for row in buckets["training_optimizer"]
            ),
            "available_training_tasks": len(train_tasks), "available_training_units": len(train_units),
            "execution_repetitions": len(self._config.execution_seeds),
            "training_repetitions": len(self._config.execution_seeds),
            "validation_repetitions": len(self._config.execution_seeds),
            "test_repetitions": len(self._config.test_execution_seeds),
            "execution_seeds": list(self._config.execution_seeds),
            "test_execution_seeds": list(self._config.test_execution_seeds),
            "baseline_training_reuse": "charged once; referenced by every residual's paired evidence",
            "by_stage": {key: _usage_summary(rows) for key, rows in buckets.items()},
            "all_calls": _usage_summary(ledger),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            return _copy({
                "schema": SCHEMA, "state": self._state, "run_id": self._run_id,
                "implementation_provenance": self._provenance,
                "candidate_bank": self.candidate_bank if self._bank_json is not None else None,
                "optimization_iterations": self._iterations,
                "optimizer_evidence": self._optimizer_evidence,
                "optimizer_evidence_sha256": residual.canonical_hash(self._optimizer_evidence),
                "selection": self._selection, "test": self._test_report,
                "executor_workers": self._config.executor_workers,
                "accounting": self.accounting(),
                "ledger": sorted(self._ledger, key=lambda row: row["sequence"]),
                "executor_failures": sorted(self._executor_failures, key=lambda row: row["sequence"]),
                "test_pair_orders": list(self._test_pair_orders.values()),
                "observations": self._observations,
                "benchmark_improvement_claim": None,
            })


__all__ = [
    "ChiselConfig", "CodexChiselController", "FrozenPolicy", "JSONCallback",
    "SCHEMA", "SPLIT_SEED", "TOKEN_FIELDS", "USAGE_FIELDS",
    "MAX_EXECUTOR_WORKERS", "TEST_PAIR_ORDER_RULE",
    "OPTIMIZER_UNIT_SELECTION_RULE",
    "execution_result_from_receipt", "implementation_provenance",
    "measured_usage_from_receipt", "public_text_features",
]
