#!/usr/bin/env python3
"""Run official AutoSaddler V2 with a PhiAgent Terminal-Bench DEV adapter."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib.request import Request, urlopen


BASELINE_STRATEGY = "No additional strategy. Follow the task instruction using the base terminal agent."


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one object")
    return value


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if (
            ".git" in relative.parts
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
            or any(part.startswith("._") for part in relative.parts)
        ):
            continue
        rows.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def render_strategy(base_prompt: str, strategy: str) -> str:
    if strategy == BASELINE_STRATEGY:
        return base_prompt
    return (
        base_prompt.rstrip()
        + "\n\n## AutoSaddler strategy\n"
        + strategy.strip()
        + "\n\nApply this strategy when relevant; the task instruction remains authoritative.\n"
    )


def compact_trace(
    messages: Sequence[Mapping[str, Any]],
    *,
    head_events: int = 6,
    tail_events: int = 6,
    command_characters: int = 800,
    observation_characters: int = 1_600,
) -> list[dict[str, str]]:
    limits = (head_events, tail_events, command_characters, observation_characters)
    if any(value < 0 for value in limits) or head_events + tail_events < 1:
        raise ValueError("AutoSaddler trace limits must be non-negative")
    events = []
    pending: dict[str, str] = {}
    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, Mapping):
                        continue
                    call_id = call.get("id")
                    function = call.get("function")
                    if not isinstance(call_id, str) or not isinstance(function, Mapping):
                        continue
                    command = ""
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                        except json.JSONDecodeError:
                            parsed = {}
                        if isinstance(parsed, Mapping):
                            command = str(parsed.get("command") or "")
                    pending[call_id] = command
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            events.append(
                {
                    "command": pending.pop(call_id, "")[:command_characters],
                    "observation": str(message.get("content") or "")[
                        :observation_characters
                    ],
                }
            )
    if len(events) <= head_events + tail_events:
        return events
    return events[:head_events] + (events[-tail_events:] if tail_events else [])


class StructuredQwenProvider:
    def __init__(self, *, endpoint: str, model: str, max_tokens: int, api: Any) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.api = api
        self.usage_rows: list[dict[str, Any]] = []

    async def run(self, request: Any) -> Any:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, session_request: Any) -> Any:
        spec = session_request.spec
        workspace_files = dict(spec.workspace_files)
        candidate_path = session_request.workspace / "candidate.json"
        if candidate_path.is_file():
            workspace_files["candidate.json"] = candidate_path.read_text()
        prompt = {
            "system_context": spec.system_context,
            "task_prompt": spec.task_prompt,
            "skills": dict(spec.skills),
            "workspace_files": workspace_files,
            "output_schema": dict(spec.output_schema),
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the optimizer inside AutoSaddler V2. Use only the "
                        "provided training evidence. Return exactly one JSON object "
                        "matching the output schema, with no Markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, indent=2, sort_keys=True),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        trace_dir = session_request.trace_dir
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            write_json(trace_dir / "request.json", payload)
        started = time.monotonic()
        http_request = Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(http_request, timeout=session_request.timeout_seconds) as response:  # noqa: S310
            raw = json.loads(response.read().decode())
        if trace_dir is not None:
            write_json(trace_dir / "response.json", raw)
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("AutoSaddler optimizer response has no unique choice")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("AutoSaddler optimizer response has no text content")
        structured = json.loads(content)
        if not isinstance(structured, dict):
            raise TypeError("AutoSaddler optimizer output must be an object")
        usage = raw.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        duration = time.monotonic() - started
        usage_row = {
            "operation_id": session_request.operation_id,
            "kind": spec.kind,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": duration,
        }
        self.usage_rows.append(usage_row)
        return self.api.SessionResult(
            status="completed",
            structured_output=structured,
            raw_response=content,
            tool_calls=(),
            usage=(
                self.api.Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                    role="optimizer",
                    duration_seconds=duration,
                ),
            ),
            cost=self.api.Cost(
                sessions=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                wall_seconds=duration,
            ),
        )


class TerminalPromptPack:
    def __init__(self, *, store: Any, api: Any) -> None:
        self.store = store
        self.api = api

    def session(self, kind: str, context: Mapping[str, Any]) -> Any:
        if kind == "evolve":
            candidate_ids = list(context["candidate_ids"])
            output_schema = {
                "type": "object",
                "required": [
                    "schema_version",
                    "parent_ids",
                    "component_sources",
                    "rationale",
                ],
                "properties": {
                    "schema_version": {"const": "autosaddler-evolution/v1"},
                    "parent_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "component_sources": {"type": "object"},
                    "rationale": {"type": "string"},
                },
                "additionalProperties": False,
            }
            task_prompt = (
                "Select exactly one parent from these IDs and use no component "
                f"composition: {candidate_ids}. Return schema_version "
                "autosaddler-evolution/v1, parent_ids, empty component_sources, and rationale."
            )
            workspace_files = {
                "evolution_context.json": json.dumps(context, indent=2, sort_keys=True)
            }
            mutation_label = None
        elif kind == "diagnose_patch":
            evidence = context["evidence"]
            evidence_path = self.store.run_dir / str(evidence["uri"])
            output_schema = {
                "type": "object",
                "required": ["schema_version", "updates", "diagnosis"],
                "properties": {
                    "schema_version": {"const": "autosaddler-diagnosis-patch/v1"},
                    "updates": {
                        "type": "object",
                        "required": ["strategy"],
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "minLength": 20,
                                "maxLength": 4_000,
                            }
                        },
                        "additionalProperties": False,
                    },
                    "diagnosis": {"type": "string"},
                },
                "additionalProperties": False,
            }
            task_prompt = (
                "Diagnose the failed and successful terminal traces. Replace the single "
                "strategy component with a generic, compact terminal-solving strategy. "
                "Do not include task answers, filenames unique to one case, or hidden-test "
                "claims. Prefer inspect-first execution, bounded dependency retries, direct "
                "artifact construction, and explicit validation."
            )
            workspace_files = {
                "training_evidence.json": evidence_path.read_text(),
                "diagnosis_context.json": json.dumps(context, indent=2, sort_keys=True),
            }
            mutation_label = "terminal-strategy-diagnosis"
        elif kind == "reflect":
            output_schema = {
                "type": "object",
                "required": ["schema_version", "lessons"],
                "properties": {
                    "schema_version": {"const": "autosaddler-reflection/v1"},
                    "lessons": {"type": "array"},
                },
                "additionalProperties": False,
            }
            task_prompt = (
                "Summarize reusable lessons from the matched training and development "
                "aggregates. Do not propose another update."
            )
            workspace_files = {
                "reflection_context.json": json.dumps(context, indent=2, sort_keys=True)
            }
            mutation_label = None
        else:
            raise ValueError(f"unknown AutoSaddler session kind: {kind}")
        return self.api.SessionSpec(
            kind=kind,
            system_context=(
                "Follow the official AutoSaddler structured phase contract and use "
                "training evidence only; development details are quarantined."
            ),
            task_prompt=task_prompt,
            skills={},
            output_schema=output_schema,
            workspace_files=workspace_files,
            capabilities=frozenset({"read_workspace"}),
            mutation_label=mutation_label,
        )


class TerminalEvidenceBuilder:
    def __init__(
        self,
        *,
        store: Any,
        trace_budget: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.trace_budget = (
            {str(key): int(value) for key, value in trace_budget.items()}
            if trace_budget is not None
            else {}
        )

    def build(self, evaluation: Any) -> Any:
        if evaluation.split != "train":
            raise ValueError("AutoSaddler evidence must come from training")
        rows = []
        for observation in evaluation.observations:
            metadata = dict(observation.metadata)
            transcript_path = Path(str(metadata["transcript_path"]))
            messages = json.loads(transcript_path.read_text())
            rows.append(
                {
                    "case_id": observation.case_id,
                    "disposition": observation.disposition,
                    "score": observation.score,
                    "instruction": metadata["instruction"],
                    "trace": compact_trace(
                        [row for row in messages if isinstance(row, Mapping)],
                        **self.trace_budget,
                    ),
                    "deployment_tokens": metadata["total_tokens"],
                }
            )
        return self.store.write_json(
            f"evidence/{evaluation.evaluation_id.removeprefix('sha256:')}/evidence.json",
            {
                "schema_version": "phiagent-terminal-autosaddler-evidence/v1",
                "evaluation_id": evaluation.evaluation_id,
                "rows": rows,
            },
            kind="training-evidence",
        )


class TerminalEvaluator:
    def __init__(
        self,
        *,
        harness: Any,
        store: Any,
        config: Mapping[str, Any],
        runner: Any,
        api: Any,
    ) -> None:
        self.harness = harness
        self.store = store
        self.config = config
        self.runner = runner
        self.api = api
        self.fingerprint = str(config["evaluator_fingerprint"])
        self.task_specs = {
            str(row["name"]): dict(row) for row in config["task_specs"]
        }
        self.ledger_path = store.run_dir / "terminal-evaluations.jsonl"

    async def evaluate(self, candidate: Any, cases: Sequence[Any], context: Any) -> Any:
        if type(context.repetitions) is not int or context.repetitions != 1:
            raise ValueError(
                "The legacy TerminalEvaluator supports one independent rollout per task; "
                "repetitions require separately frozen trial directories, not reused rewards."
            )
        materialized = self.harness.materialize(candidate, "evaluate")
        try:
            components = read_object(materialized.root / "candidate.json")
        finally:
            materialized.release()
        strategy = str(components["strategy"])
        task_ids = tuple(case.case_id for case in cases)
        split = str(context.split)
        if strategy == BASELINE_STRATEGY:
            paths = self.config["blank_result_paths"][split]
            result_pairs = [
                (Path(str(paths[task_id])), task_id) for task_id in task_ids
            ]
            cached = True
        else:
            rollout_dir = (
                self.store.run_dir
                / "adapter-rollouts"
                / hashlib.sha256(context.operation_id.encode()).hexdigest()[:16]
            )
            result_path = self._run_live(
                task_ids=task_ids,
                strategy=strategy,
                split=split,
                experiment=rollout_dir,
            )
            result_pairs = [(result_path, task_id) for task_id in task_ids]
            cached = False
        trial_rows = self._trials(result_pairs, task_ids)
        observations = []
        for case in cases:
            for repetition in range(context.repetitions):
                completed = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if completed is not None:
                    observations.append(completed)
                    continue
                attempt_id, attempt_number = context.attempt_sink.start(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                trial, result_path = trial_rows[case.case_id]
                agent = trial["agent_result"]
                verifier = trial["verifier_result"]
                if (
                    trial.get("exception") is not None
                    or trial.get("valid_binary_reward") is not True
                    or not isinstance(agent, Mapping)
                    or not isinstance(verifier, Mapping)
                    or verifier.get("reward") not in {0, 1}
                ):
                    context.attempt_sink.fail(
                        attempt_id,
                        "invalid_terminal_trial",
                        self.api.Cost(rollouts=1),
                    )
                    raise RuntimeError(f"invalid AutoSaddler trial: {case.case_id}")
                reward = int(verifier["reward"])
                cost = self.api.Cost(
                    rollouts=1,
                    input_tokens=int(agent["input_tokens"]),
                    output_tokens=int(agent["output_tokens"]),
                )
                context.attempt_sink.observe_usage(
                    attempt_id,
                    self.api.Usage(
                        input_tokens=int(agent["input_tokens"]),
                        output_tokens=int(agent["output_tokens"]),
                        model=str(self.config["model_id"]),
                        role="target",
                    ),
                )
                transcript_path = (
                    result_path.parent
                    / "trials"
                    / case.case_id
                    / "agent"
                    / "transcript.json"
                )
                observation = self.api.Observation.create(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    split=context.split,
                    repetition=repetition,
                    disposition="success" if reward else "task_failure",
                    score=float(reward),
                    evaluator_fingerprint=self.fingerprint,
                    attempts=attempt_number,
                    cost=cost,
                    metadata={
                        "cached_blank_rollout": cached,
                        "direct_result_path": str(result_path),
                        "direct_result_sha256": file_sha256(result_path),
                        "transcript_path": str(transcript_path),
                        "instruction": (
                            Path(str(self.config["source_root"]))
                            / case.case_id
                            / "instruction.md"
                        ).read_text(),
                        "total_tokens": int(agent["total_tokens"]),
                    },
                )
                context.attempt_sink.complete(attempt_id, observation, cost)
                observations.append(observation)
        evaluation_id = self.api.sha256_digest(context.operation_id)
        artifact_prefix = (
            "evaluations" if context.split == "train" else "quarantine/dev"
        )
        artifact = self.store.write_json(
            (
                f"{artifact_prefix}/{evaluation_id.removeprefix('sha256:')}"
                "/terminal-summary.json"
            ),
            {
                "candidate_id": candidate.candidate_id,
                "split": split,
                "purpose": context.purpose,
                "strategy_sha256": hashlib.sha256(strategy.encode()).hexdigest(),
                "cached_blank_rollout": cached,
                "task_ids": list(task_ids),
                "successes": sum(item.score == 1.0 for item in observations),
                "deployment_tokens": sum(item.cost.input_tokens + item.cost.output_tokens for item in observations),
            },
            kind="evaluation-summary",
        )
        ledger = {
            "candidate_id": candidate.candidate_id,
            "split": split,
            "purpose": context.purpose,
            "task_ids": list(task_ids),
            "strategy_sha256": hashlib.sha256(strategy.encode()).hexdigest(),
            "cached_blank_rollout": cached,
            "successes": sum(item.score == 1.0 for item in observations),
            "deployment_tokens": sum(
                item.cost.input_tokens + item.cost.output_tokens for item in observations
            ),
            "artifact_uri": artifact.uri,
        }
        with self.ledger_path.open("a") as stream:
            stream.write(json.dumps(ledger, sort_keys=True) + "\n")
        return self.api.Evaluation(
            evaluation_id=evaluation_id,
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=task_ids,
            observations=tuple(observations),
            artifact_dir=self.api.ArtifactRef(
                uri=str(Path(artifact.uri).parent),
                kind="evaluation-directory",
            ),
        )

    def selected_deployment(self, candidate_id: str) -> dict[str, Any]:
        entries = [
            json.loads(line)
            for line in self.ledger_path.read_text().splitlines()
            if line.strip()
        ]
        rows = [
            row
            for row in entries
            if row["candidate_id"] == candidate_id and row["split"] == "development"
        ]
        if not rows:
            raise RuntimeError("selected AutoSaddler candidate has no DEV evaluation")
        return rows[-1]

    def _run_live(
        self,
        *,
        task_ids: tuple[str, ...],
        strategy: str,
        split: str,
        experiment: Path,
    ) -> Path:
        if experiment.exists():
            raise FileExistsError(experiment)
        experiment.mkdir(parents=True)
        direct = deepcopy(self.config["direct_runner_template"])
        strategy_sha = hashlib.sha256(strategy.encode()).hexdigest()
        direct["experiment_id"] = f"autosaddler-{split}-{strategy_sha[:12]}"
        direct["method"] = "AutoSaddler candidate"
        direct["tasks"] = [self.task_specs[task_id] for task_id in task_ids]
        direct["agent"]["system_prompt"] = render_strategy(
            str(self.config["base_system_prompt"]), strategy
        )
        direct["max_concurrent_trials"] = len(task_ids)
        direct["acceptance"] = {"minimum_successes": 0}
        write_json(experiment / "config.json", direct)
        write_json(
            experiment / "reservation.json",
            {
                "status": "RESERVED_BEFORE_RUN",
                "reserved_at_utc": utc_now(),
                "task_ids": list(task_ids),
                "strategy_sha256": strategy_sha,
            },
        )
        write_json(experiment / "git-state.json", self.config["git_state"])
        result, exit_code = self.runner.run_experiment(experiment)
        if exit_code != 0 or result["valid_binary_reward_cells"] != len(task_ids):
            raise RuntimeError(f"invalid AutoSaddler rollout: {experiment}")
        return experiment / "results.json"

    @staticmethod
    def _trials(
        results: Sequence[tuple[Path, str]],
        task_ids: tuple[str, ...],
    ) -> dict[str, tuple[dict[str, Any], Path]]:
        if len(set(task_ids)) != len(task_ids):
            raise RuntimeError("AutoSaddler task IDs must be unique; repeats are independent trials")
        rows = {}
        for result_path, requested in results:
            result = read_object(result_path)
            for trial in result["trials"]:
                if trial["task"] == requested:
                    if requested in rows:
                        raise RuntimeError("AutoSaddler result contains duplicate trials for one task")
                    rows[requested] = (trial, result_path)
        if set(rows) != set(task_ids):
            raise RuntimeError("AutoSaddler result coverage differs")
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    config_path = experiment / "config.json"
    reservation_path = experiment / "reservation.json"
    config = read_object(config_path)
    if (
        config.get("status") != "FROZEN_BEFORE_FIRST_RUN"
        or not reservation_path.is_file()
    ):
        raise RuntimeError("AutoSaddler experiment is not frozen and reserved")
    script_path = Path(__file__).resolve()
    if file_sha256(script_path) != config["adapter_sha256"]:
        raise RuntimeError("AutoSaddler adapter differs from frozen SHA-256")
    source_root = Path(str(config["autosaddler_source_root"]))
    source_manifest = read_object(Path(str(config["autosaddler_source_manifest"])))
    observed_source = normalized_manifest(source_root)
    expected_source = sorted(
        source_manifest["files"], key=lambda row: str(row["path"])
    )
    if (
        observed_source != expected_source
        or canonical_sha256(observed_source)
        != config["autosaddler_source_tree_sha256"]
    ):
        raise RuntimeError("AutoSaddler source differs from frozen manifest")
    runtime_root = Path(str(config["runtime_overlay_root"]))
    runtime_manifest = read_object(Path(str(config["runtime_overlay_manifest"])))
    observed_runtime = normalized_manifest(runtime_root)
    if (
        observed_runtime != runtime_manifest["files"]
        or canonical_sha256(observed_runtime)
        != config["runtime_overlay_tree_sha256"]
    ):
        raise RuntimeError("AutoSaddler runtime overlay differs from frozen manifest")
    runner_path = Path(str(config["direct_runner_path"]))
    if file_sha256(runner_path) != config["direct_runner_sha256"]:
        raise RuntimeError("Terminal direct runner differs from frozen SHA-256")
    for path_text, expected in config["input_file_sha256"].items():
        if file_sha256(Path(path_text)) != expected:
            raise RuntimeError(f"AutoSaddler frozen input changed: {path_text}")

    snapshot = experiment / "source_snapshot"
    snapshot.mkdir()
    shutil.copy2(script_path, snapshot / script_path.name)
    shutil.copy2(runner_path, snapshot / runner_path.name)
    runner = load_module("phiagent_terminal_direct_runner_autosaddler", runner_path)
    sys.path.insert(0, str(source_root / "src"))

    from autosaddler.v2.core.domain import (
        ArtifactRef,
        Case,
        Cost,
        Evaluation,
        Observation,
        sha256_digest,
    )
    from autosaddler.v2.core.engine import AutoSaddlerEngine
    from autosaddler.v2.core.policies import (
        BudgetPolicy,
        FixedTaskSelectionPolicy,
        FullOnAcceptDevelopment,
        MatchedValidStrictImprovement,
        MeanDevelopmentRanking,
        PolicyBundle,
    )
    from autosaddler.v2.core.ports import ScenarioComponents
    from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace
    from autosaddler.v2.prompting.models import SessionResult, SessionSpec, Usage
    from autosaddler.v2.storage.local import LocalRunStore

    class Api:
        pass

    api = Api()
    for name, value in {
        "ArtifactRef": ArtifactRef,
        "Case": Case,
        "Cost": Cost,
        "Evaluation": Evaluation,
        "Observation": Observation,
        "SessionResult": SessionResult,
        "SessionSpec": SessionSpec,
        "Usage": Usage,
        "sha256_digest": sha256_digest,
    }.items():
        setattr(api, name, value)

    run_dir = experiment / "engine"
    store = LocalRunStore(run_dir=run_dir, run_id=str(config["run_id"]))
    store.initialize(
        resolved_config=config["resolved_engine_config"],
        resolved_entities={
            "resolved/sources/terminal-bench.json": config["resolved_terminal_source"],
            "resolved/prompts/evolve.md": "Select one accepted parent without composition.\n",
            "resolved/prompts/diagnosis_patch.md": (
                "Diagnose training traces and replace the generic strategy component.\n"
            ),
            "resolved/prompts/reflect.md": "Record reusable aggregate lessons.\n",
        },
    )
    harness = ComponentMapHarnessSpace(
        baseline={"strategy": BASELINE_STRATEGY},
        store_root=run_dir / "candidates",
        materialization_root=run_dir / "materialized",
        validator=lambda value: _validate_components(value),
    )
    evaluator = TerminalEvaluator(
        harness=harness,
        store=store,
        config=config["adapter_config"],
        runner=runner,
        api=api,
    )
    provider = StructuredQwenProvider(
        endpoint=str(config["optimizer_endpoint"]),
        model=str(config["optimizer_model"]),
        max_tokens=int(config["optimizer_max_tokens"]),
        api=api,
    )
    scenario = ScenarioComponents(
        name="phiagent_terminal_bench",
        version="1",
        harness_space=harness,
        evaluator=evaluator,
        evidence_builder=TerminalEvidenceBuilder(
            store=store,
            trace_budget=config["adapter_config"].get("optimizer_trace_budget"),
        ),
        prompt_pack=TerminalPromptPack(store=store, api=api),
        train_cases=tuple(
            Case(case_id=task_id, split="train", payload={})
            for task_id in config["train_task_ids"]
        ),
        development_cases=tuple(
            Case(case_id=task_id, split="development", payload={})
            for task_id in config["dev_task_ids"]
        ),
        required_capabilities=frozenset({"read_workspace"}),
        resolved_entities={},
    )
    policies = PolicyBundle(
        task_selection=FixedTaskSelectionPolicy(
            batch_size=len(config["train_task_ids"])
        ),
        acceptance=MatchedValidStrictImprovement(),
        development=FullOnAcceptDevelopment(),
        ranking=MeanDevelopmentRanking(),
        budget=BudgetPolicy(max_rollouts=20, max_iterations=1),
    )
    started_at = utc_now()
    optimization = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=policies,
        diagnosis_patch_timeout_seconds=float(config["optimizer_timeout_seconds"]),
        selection_timeout_seconds=float(config["optimizer_timeout_seconds"]),
        reflection_timeout_seconds=float(config["optimizer_timeout_seconds"]),
        session_retries=1,
    ).run()
    candidate_path = (
        run_dir
        / "candidates"
        / optimization.selected_candidate_id.removeprefix("sha256:")
        / "candidate.json"
    )
    selected_components = read_object(candidate_path)
    deployment = evaluator.selected_deployment(optimization.selected_candidate_id)
    optimizer_tokens = {
        "input_tokens": sum(row["input_tokens"] for row in provider.usage_rows),
        "output_tokens": sum(row["output_tokens"] for row in provider.usage_rows),
        "total_tokens": sum(
            row["input_tokens"] + row["output_tokens"] for row in provider.usage_rows
        ),
        "calls": provider.usage_rows,
    }
    result = {
        "schema_version": "phiagent-terminalbench-autosaddler-dev/v1",
        "status": "WORKING",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "benchmark": "Terminal-Bench 2.0",
        "method": "AutoSaddler",
        "identity": config["method_identity"],
        "official_autosaddler_revision": config["official_autosaddler_revision"],
        "optimization": {
            "run_id": optimization.run_id,
            "selected_candidate_id": optimization.selected_candidate_id,
            "development_score": optimization.development_score,
            "iterations": optimization.iterations,
            "selected_components": selected_components,
        },
        "development": deployment,
        "optimizer_tokens": optimizer_tokens,
        "claim_boundary": config["claim_boundary"],
        "evidence": {
            "config_sha256": file_sha256(config_path),
            "reservation_sha256": file_sha256(reservation_path),
            "adapter_sha256": file_sha256(script_path),
            "direct_runner_sha256": file_sha256(runner_path),
            "event_log_sha256": file_sha256(store.events_path),
            "evaluation_ledger_sha256": file_sha256(evaluator.ledger_path),
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json(experiment / "optimizer-usage.json", optimizer_tokens)
    write_json(experiment / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_components(value: Mapping[str, str]) -> None:
    if set(value) != {"strategy"}:
        raise ValueError("AutoSaddler Terminal component schema changed")
    strategy = value["strategy"]
    if len(strategy.encode()) > 8_000:
        raise ValueError("AutoSaddler Terminal strategy exceeds 8,000 bytes")
    forbidden = {
        "79586",
        "count-dataset-tokens",
        "financial-document-processor",
        "pytorch-model-cli",
    }
    if any(token.lower() in strategy.lower() for token in forbidden):
        raise ValueError("AutoSaddler strategy contains task-specific training text")


if __name__ == "__main__":
    raise SystemExit(main())
