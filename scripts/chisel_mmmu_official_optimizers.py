"""Official SkillOpt/AutoSaddler loops with train-only native MMMU adapters."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from scripts.run_chisel_mmmu_shared import digest, image_urls, load, write
from scripts.chisel_mmmu_continuation import record_resumed_rollout

INITIAL_SKILL = (
    "# Visual reasoning\n"
    "Use the complete question, public options, and every supplied original image. "
    "Check the relevant visual details and reason about the requested quantity. "
    "Follow the question's requested final-answer form."
)


def _bind_sources(study):
    source = study.root / "source"
    sys.path[:0] = [
        str(source / "official/SkillOpt"),
        str(source / "official/AutoSaddler/src"),
        str(source / "integrations/autosaddler_phiagent/src"),
        str(source / "dependencies"),
    ]


def train_skillopt(study, *, prepare_only=False):
    _bind_sources(study)
    from skillopt.config import flatten_config, load_config
    from skillopt.datasets.base import SplitDataLoader
    from skillopt.engine.trainer import ReflACTTrainer
    from skillopt.envs.base import EnvAdapter
    from scripts.cvpr_skillopt_native_bridge import NativeBridge, IMAGE_STAGES, TEXT_STAGES

    root = study.client.directory
    profile = study.config["methods"]["skillopt"]
    allowed = set(study.by_id)
    feedback = load(study.root / "frozen/OPTIMIZER_FEEDBACK_SPLIT.json")
    update_rows = [study.by_id[identity] for identity in feedback["ids"]["update"]]
    gate_rows = [study.by_id[identity] for identity in feedback["ids"]["gate"]]
    checkpoint_root = study.config.get("optimizer_resume_from")
    if checkpoint_root:
        previous = Path(checkpoint_root) / "methods/skillopt/train/optimizer"
        if not (previous / "runtime_state.json").is_file():
            raise RuntimeError("Official SkillOpt continuation lacks a native checkpoint")
        shutil.copytree(previous, root / "optimizer")
        state_path = root / "optimizer/runtime_state.json"
        state = load(state_path)
        for field in ("current_skill_path", "best_skill_path"):
            relative = Path(state[field]).relative_to(previous)
            state[field] = str(root / "optimizer" / relative)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        write(root / "CHECKPOINT_RESUME.json", {
            "prior_root": str(previous), "last_completed_step": state["last_completed_step"],
            "native_checkpoint_preserved": True, "only_absolute_runtime_paths_rebound": True,
            "completed_wire_requests_reused_by_exact_hash_and_physical_gpu": True,
        })
    shared = SimpleNamespace(write=write, image_urls=image_urls)

    class Bridge(NativeBridge):
        def __init__(self):
            super().__init__(shared, study.client, lambda identity: study.client.chat(study.by_id[identity]))

        def require_integrity(self):
            if study.client.failed or (self.root / "skillopt-bridge-failure.json").exists():
                raise RuntimeError("Preserved native provider failure blocks the official trainer")

        def fail(self, stage, error):
            study.client.failed = True
            super().fail(stage, error)

        def optimizer(self, system, user, max_completion_tokens=16384, retries=5,
                      stage="optimizer", reasoning_effort=None, timeout=None):
            self.require_integrity()
            if stage not in IMAGE_STAGES | TEXT_STAGES:
                raise ValueError("Unbound official SkillOpt optimization stage")
            scope = self.scope.get()
            if stage in IMAGE_STAGES:
                if scope is None or scope[0] != stage or not scope[1]:
                    raise ValueError("Native reflection lacks an exact training-only image scope")
                identities = list(scope[1])
                if not set(identities).issubset(allowed):
                    raise RuntimeError("External validation/test entered optimizer reflection")
            else:
                if scope is not None:
                    raise RuntimeError("Text edit operator inherited an unrelated image scope")
                identities = []
            rows = [study.by_id[identity] for identity in identities]
            chat = [
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": user}]
                 + study.client.original_parts(rows)},
            ]
            with self.lock:
                record, _ = study.client.request(chat, role="optimizer", rows=rows)
                usage = record["usage"]
                if self.tracker is not None:
                    self.tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
                self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
                dispatches = self.scope_dispatches.get()
                if dispatches is not None:
                    dispatches.append(record)
            return record["raw_content"], usage

    bridge = Bridge()
    data = root / "optimizer-data"
    for split, rows in (("train", update_rows), ("val", gate_rows), ("test", [])):
        write(data / split / "items.json", rows)

    class TrainingOnlyLoader(SplitDataLoader):
        def get_split_items(self, split):
            if split not in {"train", "val", "valid_seen", "selection"}:
                raise RuntimeError("The official loop cannot open external validation/test")
            return super().get_split_items(split)

    loader = TrainingOnlyLoader(split_dir=str(data), split_mode="split_dir", seed=study.config["model"]["seed"])

    class Adapter(EnvAdapter):
        def setup(self, cfg):
            super().setup(cfg)
            loader.setup(cfg)
            self.analyst_workers = 1
            self.failure_only = cfg["failure_only"]
            self.minibatch_size = cfg["minibatch_size"]
            self.edit_budget = cfg["edit_budget"]

        def get_dataloader(self):
            return loader

        def build_env_from_batch(self, batch, **kwargs):
            rows = list(batch.payload)
            if not rows or not {row["id"] for row in rows}.issubset(allowed):
                raise RuntimeError("A SkillOpt batch crossed the fixed training population")
            return rows

        def build_train_env(self, batch_size, seed, **kwargs):
            return self.build_env_from_batch(loader.build_train_batch(batch_size, seed, **kwargs))

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return self.build_env_from_batch(loader.build_eval_batch(env_num, split, seed, **kwargs))

        def get_task_types(self):
            return sorted({row["subject"] for row in study.rows})

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            if kwargs.get("diagnostic_mode") or kwargs.get("diagnostic_instruction"):
                raise RuntimeError("Undeclared diagnostic prompt intervention")
            rows = list(env_manager)
            if not {row["id"] for row in rows}.issubset(allowed):
                raise RuntimeError("Only training can supply optimizer feedback")
            with ThreadPoolExecutor(max_workers=study.config["execution"]["actor_workers"]) as pool:
                outputs = list(pool.map(lambda row: study.client.actor(row, skill_content), rows))
            study.score(rows, outputs, label="official-skillopt-training")
            results = []
            directory = Path(out_dir)
            for row, output in zip(rows, outputs):
                result = {
                    "id": row["id"], "task_type": row["subject"], "task_description": row["question"],
                    "question": row["question"], "hard": output["hard"], "soft": float(output["hard"]),
                    "agent_ok": True, "n_turns": 1, "response": output["raw_content"],
                    "predicted_answer": output["raw_content"], "gold_answer": study.labels[row["id"]],
                    "reference_text": json.dumps({"official_answer": study.labels[row["id"]]}),
                    "fail_reason": "" if output["hard"] else "Incorrect under the unmodified official MMMU scorer",
                }
                conversation = []
                for message in study.client.chat(row, skill_content):
                    text = "\n".join(part["text"] if part["type"] == "text" else
                                     "[Original image; pixels are bound separately for reflection]"
                                     for part in message["content"])
                    conversation.append({"role": message["role"], "content": text})
                conversation.append({"role": "assistant", "content": output["raw_content"]})
                for name, value in (("conversation.json", conversation), ("result.json", result)):
                    destination = directory / "predictions" / row["id"] / name
                    record_resumed_rollout(
                        destination, value, optimizer_root=root / "optimizer",
                        prior_optimizer_root=(Path(checkpoint_root) / "methods/skillopt/train/optimizer"
                                              if checkpoint_root else None))
                if bridge.tracker is not None:
                    bridge.tracker.record("actor", output["usage"]["prompt_tokens"], output["usage"]["completion_tokens"])
                results.append(result)
            return results

        def reflect(self, *args, **kwargs):
            bridge.require_integrity()
            value = super().reflect(*args, **kwargs)
            bridge.require_integrity()
            return value

    official = study.root / "source/official/SkillOpt"
    cfg = flatten_config(load_config(str(official / "configs/docvqa/default.yaml")))
    initial = root / "initial_skill.md"
    initial.write_text(INITIAL_SKILL)
    cfg.update(
        out_root=str(root / "optimizer"), skill_init=str(initial), model_backend="openai_compatible",
        optimizer_backend="openai_compatible", target_backend="openai_compatible",
        optimizer_model=study.config["model"]["served_name"], target_model=study.config["model"]["served_name"],
        reasoning_effort="", seed=study.config["model"]["seed"], analyst_workers=1, workers=4,
        eval_test=False, sel_env_num=0, test_env_num=0, train_size=len(update_rows),
        max_completion_tokens=study.config["model"]["max_tokens"], env_name="mmmu",
        split_mode="split_dir", split_dir=str(data), num_epochs=profile["epochs"],
        batch_size=profile["batch_size"], minibatch_size=profile["minibatch_size"],
    )
    write(root / "OFFICIAL_TRAINER_CONFIG.json", cfg)
    if prepare_only:
        adapter = Adapter()
        adapter.setup(cfg)
        update = adapter.build_train_env(min(profile["batch_size"], len(update_rows)),
                                        study.config["model"]["seed"])
        gate = adapter.build_eval_env(0, "valid_seen", study.config["model"]["seed"])
        if (not {row["id"] for row in update}.issubset(set(feedback["ids"]["update"]))
                or {row["id"] for row in gate} != set(feedback["ids"]["gate"])):
            raise RuntimeError("Official SkillOpt loader changed the frozen feedback partition")
        with bridge.installed():
            pass
        write(root / "FACTORY_GATE.json", {
            "status": "OFFICIAL_LOADER_AND_NATIVE_BINDING_PREPARED",
            "update_count": len(update_rows), "gate_count": len(gate_rows),
            "model_calls": 0, "scorer_calls": 0, "optimizer_executed": False})
        return []
    with bridge.installed():
        summary = ReflACTTrainer(cfg, Adapter()).train()
    bridge.require_integrity()
    expected = profile["epochs"] * math.ceil(len(update_rows) / profile["batch_size"])
    actual_steps = [step for epoch in summary["epoch_stats"] for step in epoch["steps"]]
    if (summary["total_steps"] != expected or sorted(actual_steps) != list(range(1, expected + 1))
            or (not checkpoint_root and bridge.stage_calls.get("slow_update") != profile["epochs"] - 1)
            or (not checkpoint_root and bridge.stage_calls.get("meta_skill") != profile["epochs"] - 1)
            or any(summary[key] is not None for key in ("baseline_test_hard", "test_hard", "final_test_hard"))):
        raise RuntimeError("Incomplete official SkillOpt loop cannot release a candidate")
    for epoch in range(2, profile["epochs"] + 1):
        for folder, filename, field in (
                ("slow_update", "slow_result.json", "slow_update_content"),
                ("meta_skill", "meta_skill_result.json", "meta_skill_content")):
            path = root / "optimizer" / folder / f"epoch_{epoch:02d}" / filename
            if not path.is_file() or not load(path).get(field):
                raise RuntimeError("Official epoch-end operator did not produce its required content")
    write(root / "OPTIMIZER_COMPLETE.json", {
        "identity": profile["identity"], "revision": profile["revision"],
        "summary": summary, "stage_calls": bridge.stage_calls,
        "external_validation_or_test_used": False, "training_question_count": len(study.rows),
        "update_question_count": len(update_rows), "internal_gate_question_count": len(gate_rows),
        "official_internal_selection_scope": "Training partition only; external validation happens after this return",
    })
    best = (root / "optimizer/best_skill.md").read_text()
    return [{"instruction": INITIAL_SKILL}, {"instruction": best}]


def train_autosaddler(study, *, prepare_only=False):
    _bind_sources(study)
    from autosaddler.v2.config.registry import ScenarioRegistration, build_runtime, default_registry
    from autosaddler.v2.plugins.api import ScenarioPlugin, SCENARIO_PLUGIN_API_VERSION
    from autosaddler.v2.core.domain import ArtifactRef, Case, Cost, Evaluation, Observation, sha256_digest
    from autosaddler.v2.core.ports import ScenarioComponents
    from autosaddler.v2.prompting.models import Usage
    from phiagent_autosaddler.text_skill import TextSkillHarnessSpace, TextSkillPromptPack
    from phiagent_autosaddler import openai_function_provider as provider
    import yaml

    profile = study.config["methods"]["autosaddler"]
    directory = study.client.directory
    allowed = set(study.by_id)
    feedback = load(study.root / "frozen/OPTIMIZER_FEEDBACK_SPLIT.json")
    update_ids, gate_ids = set(feedback["ids"]["update"]), set(feedback["ids"]["gate"])
    native_scope = ContextVar("autosaddler_mmmu_training_image_scope", default=None)
    original_post, original_run = provider._post_json, provider.OpenAIFunctionTransport.run

    async def native_run(transport, session, timeout_seconds):
        scope_path = Path(session.workspace) / "native_training_scope.json"
        identities = load(scope_path)["question_ids"] if scope_path.exists() else []
        if not set(identities).issubset(allowed):
            raise RuntimeError("AutoSaddler optimizer image scope crossed training partition")
        token = native_scope.set(tuple(identities))
        try:
            return await original_run(transport, session, timeout_seconds)
        finally:
            native_scope.reset(token)

    def native_post(url, payload, timeout_seconds):
        identities = native_scope.get()
        if identities is None:
            raise RuntimeError("AutoSaddler provider request has no bound session scope")
        adaptation = None
        if profile.get("structured_finalization_transport") == "json_object_exact_postvalidation":
            from scripts.chisel_mmmu_native_schema import validated_json_object_transport
            payload, adaptation = validated_json_object_transport(payload)
        elif profile.get("structured_finalization_transport") not in (None, "native_json_schema"):
            raise ValueError("Unknown declared structured-finalization transport")
        rows = [study.by_id[identity] for identity in identities]
        chat = list(payload["messages"])
        if rows:
            chat = [chat[0], {"role": "user", "content": study.client.original_parts(rows)}, *chat[1:]]
        record, raw = study.client.request(chat, role="optimizer", rows=rows, payload=payload)
        if adaptation is not None:
            write(study.root / record["request_directory"] / "original-schema-postvalidation-contract.json",
                  adaptation)
        return raw

    def factory(*, settings, base_dir, run_dir, store, ledger):
        harness = TextSkillHarnessSpace(
            baseline_skill=INITIAL_SKILL, store_root=run_dir / "candidates",
            materialization_root=run_dir / "materialized")

        class NativeEvidence:
            def build(self, evaluation):
                if evaluation.split != "train":
                    raise RuntimeError("Only training observations enter optimizer evidence")
                ids = [observation.case_id for observation in evaluation.observations]
                if not set(ids).issubset(allowed):
                    raise RuntimeError("Reference labels crossed the fixed training boundary")
                evidence_id = sha256_digest(evaluation.evaluation_id).removeprefix("sha256:")
                return store.write_json(
                    f"evidence/{evidence_id}/evidence.json",
                    {"benchmark": "MMMU", "evaluation_id": evaluation.evaluation_id,
                     "native_official_accuracy": evaluation.aggregate_score,
                     "case_records": [{
                         "case_id": observation.case_id, "official_correct": observation.score,
                         "response": observation.metadata.get("response"),
                         "training_reference": study.labels[observation.case_id],
                     } for observation in evaluation.observations],
                     "information_boundary": "Training-only native annotations and original pixels; external validation/test unavailable"},
                    kind="mmmu-native-training-evidence")

        class NativePromptPack(TextSkillPromptPack):
            def session(self, kind, context):
                spec = super().session(kind, context)
                ids = list(context.get("train_case_ids", [])) if kind in {"diagnose_patch", "reflect"} else []
                if kind in {"diagnose_patch", "reflect"} and (
                        not ids or len(set(ids)) != len(ids) or not set(ids).issubset(update_ids)):
                    raise RuntimeError("Official image-bearing session lacks exact TRAIN-update identities")
                files = dict(spec.workspace_files)
                files["native_training_scope.json"] = json.dumps({"question_ids": ids}) + "\n"
                if ids:
                    files["training_references.json"] = json.dumps(
                        {identity: study.labels[identity] for identity in ids}, indent=2) + "\n"
                return replace(
                    spec, workspace_files=files,
                    system_context=spec.system_context.replace("gold, ", ""),
                    task_prompt=spec.task_prompt + (
                        "\nThis native MMMU adapter also provides complete original question/image inputs "
                        "and authoritative training references. These training references are permitted "
                        "optimizer feedback, exactly as in the SkillOpt adapter; never encode labels or case IDs "
                        "into a skill. External validation and test are inaccessible."),
                )

        class Evaluator:
            fingerprint = "official-mmmu-hf-98e6-native-shared-202060"

            async def evaluate(self, candidate, cases, context):
                if context.split not in {"train", "development"} or not {case.case_id for case in cases}.issubset(allowed):
                    raise RuntimeError("Official optimizer evaluation escaped the training partition")
                permitted = update_ids if context.split == "train" else gate_ids
                if not {case.case_id for case in cases}.issubset(permitted):
                    raise RuntimeError("Official optimizer update/gate membership changed")
                materialized = harness.materialize(candidate, "evaluate")
                try:
                    skill = load(materialized.root / "candidate.json")["skill"]
                finally:
                    materialized.release()
                semaphore = asyncio.Semaphore(study.config["execution"]["actor_workers"])

                async def one(case):
                    cached = context.attempt_sink.completed(candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=0)
                    if cached is not None:
                        return cached
                    attempt, number = context.attempt_sink.start(candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=0)
                    started = time.monotonic()
                    row = study.by_id[case.case_id]
                    try:
                        async with semaphore:
                            output = await asyncio.to_thread(study.client.actor, row, skill)
                            await asyncio.to_thread(study.score, [row], [output], label="official-autosaddler-training")
                    except BaseException:
                        context.attempt_sink.fail(attempt, "native_provider_or_scorer_failure", Cost(rollouts=1))
                        raise
                    usage = output["usage"]
                    cost = Cost(rollouts=1, input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                                wall_seconds=time.monotonic() - started)
                    context.attempt_sink.observe_usage(attempt, Usage(
                        input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                        total_tokens=usage["total_tokens"], model=study.config["model"]["served_name"], role="actor"))
                    target = context.artifact_dir / (digest(case.case_id) + ".json")
                    write(target, output)
                    ref = ArtifactRef(uri=str(target.relative_to(run_dir)), kind="native-visual-response")
                    observation = Observation.create(
                        candidate_id=candidate.candidate_id, case_id=case.case_id, split=context.split,
                        repetition=0, disposition="success" if output["hard"] else "task_failure",
                        score=float(output["hard"]), evaluator_fingerprint=self.fingerprint,
                        output=ref, attempts=number, cost=cost,
                        metadata={"response": output["raw_content"], "usage": usage})
                    context.attempt_sink.complete(attempt, observation, cost)
                    return observation

                observations = await asyncio.gather(*(one(case) for case in cases))
                return Evaluation(
                    evaluation_id=sha256_digest(context.operation_id), candidate_id=candidate.candidate_id,
                    split=context.split, purpose=context.purpose, iteration=context.iteration,
                    requested_case_ids=tuple(case.case_id for case in cases), observations=tuple(observations),
                    artifact_dir=ArtifactRef(uri=str(context.artifact_dir.relative_to(run_dir)), kind="native-evaluation"))

        def cases(split):
            identities = feedback["ids"]["update" if split == "train" else "gate"]
            return tuple(Case(case_id=identity, split=split, payload={"question": study.by_id[identity]["question"]})
                         for identity in identities)
        return ScenarioComponents(
            name="phiagent_native_mmmu", version="1", harness_space=harness, evaluator=Evaluator(),
            evidence_builder=NativeEvidence(), prompt_pack=NativePromptPack(
                store, benchmark="MMMU native vision", skill_document=INITIAL_SKILL),
            train_cases=cases("train"), development_cases=cases("development"),
            required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
            resolved_entities={"external_validation_used": False, "external_test_used": False})

    registry = default_registry()
    plugin = ScenarioPlugin(name="phiagent_native_mmmu", api_version=SCENARIO_PLUGIN_API_VERSION, factory=factory)
    registry.register_scenario(ScenarioRegistration(plugin=plugin, source="builtin"))
    registry.providers["phiagent_openai_function"] = provider.build_openai_function_provider
    model = study.config["model"]["served_name"]
    cfg = {
        "schema_version": "autosaddler/v2", "scenario": {"type": "phiagent_native_mmmu", "settings": {}},
        "optimization": {
            "task_selection": {"type": profile["task_selection"], "batch_size": profile["batch_size"],
                               "seed": study.config["model"]["seed"]},
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"}, "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 100000, "max_iterations": profile["iterations"]},
            "diagnosis_patch_timeout_seconds": 1800, "selection_timeout_seconds": 1800,
            "reflection_timeout_seconds": 1800,
            "session_retries": profile.get("session_retries", 0), "session_retry_backoff_seconds": 0,
        },
        "provider": {
            "type": "phiagent_openai_function", "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
            "settings": {
                "model": model, "reasoning_effort": None, "max_tool_steps": profile["max_tool_steps"],
                "temperature": 0, "max_completion_tokens": study.config["model"]["optimizer_max_tokens"],
                "enable_thinking": False, "finish_on_stalled_reads": True,
                "provider": {"type": "openai", "base_url": "http://127.0.0.1/native-bound/v1",
                             "wire_api": "chat_completions", "model_id": model, "wire_model": model},
            },
        },
        "storage": {"type": "local", "run_root": str(directory / "optimizer")},
    }
    path = directory / "OFFICIAL_AUTOSADDLER_CONFIG.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    runtime = build_runtime(path, run_id="native-mmmu-training", registry=registry)
    checkpoint_root = study.config.get("autosaddler_resume_from", study.config.get("optimizer_resume_from"))
    if checkpoint_root:
        from autosaddler.v2.storage.local import LocalRunStore
        previous = Path(checkpoint_root) / "methods/autosaddler/train/optimizer/native-mmmu-training"
        source = LocalRunStore(run_dir=previous, run_id="native-mmmu-training")
        source.validate_integrity()
        if runtime.store.events():
            raise RuntimeError("A new AutoSaddler continuation already has events")
        for item in previous.iterdir():
            if item.name in {"resolved", "resolved_config.yaml", "manifest.json", "snapshot.json",
                             "metrics-summary.json", "evolution_dag.json"}:
                continue
            target = runtime.store.run_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copyfile(item, target)
        runtime.store.refresh_projections()
        runtime.store.validate_integrity()
        write(directory / "CHECKPOINT_RESUME.json", {
            "prior_root": str(previous), "official_events_and_candidates_copied_exactly": True,
            "official_engine_unmodified": True, "session_retries": profile.get("session_retries", 0),
            "schema_transport_repair_only": True,
        })
    if prepare_only:
        write(directory / "FACTORY_GATE.json", {
            "status": "OFFICIAL_SCENARIO_PROVIDER_AND_DISJOINT_CASES_PREPARED",
            "update_count": len(update_ids), "gate_count": len(gate_ids),
            "prior_checkpoint_integrity": "VERIFIED" if checkpoint_root else "NOT_APPLICABLE",
            "model_calls": 0, "scorer_calls": 0, "optimizer_executed": False})
        return []
    provider._post_json = native_post
    provider.OpenAIFunctionTransport.run = native_run
    try:
        result = runtime.engine.run()
    finally:
        provider._post_json, provider.OpenAIFunctionTransport.run = original_post, original_run
    if study.client.failed or result.iterations != profile["iterations"]:
        raise RuntimeError("Incomplete official AutoSaddler loop cannot release a candidate")
    from phiagent.harness.aime_transport_recovery import autosaddler_completion
    events = [json.loads(line) for line in runtime.store.events_path.read_text().splitlines() if line.strip()]
    completion = autosaddler_completion(events, profile["iterations"])
    write(directory / "OFFICIAL_COMPLETION_AUDIT.json", completion)
    if completion["status"] != "PASS" or completion["failed_patch_sessions"]:
        raise RuntimeError("Official AutoSaddler lacks complete matched iteration/patch evidence")
    state = runtime.engine._state()
    materialized = runtime.scenario.harness_space.materialize(state.candidates[result.selected_candidate_id], "evaluate")
    try:
        selected = load(materialized.root / "candidate.json")["skill"]
    finally:
        materialized.release()
    write(directory / "OPTIMIZER_COMPLETE.json", {
        "identity": profile["identity"], "revision": profile["revision"], "iterations": result.iterations,
        "selected_candidate_id": result.selected_candidate_id, "training_question_count": len(study.rows),
        "update_question_count": len(update_ids), "internal_gate_question_count": len(gate_ids),
        "official_internal_development_scope": "Training partition only",
        "external_validation_or_test_used": False, "cost": study.client.cost(),
        "official_abandoned_deferred_work": [
            event["payload"] for event in events if event["event_type"] == "DeferredWorkAbandoned"],
    })
    return [{"instruction": INITIAL_SKILL}, {"instruction": selected}]


def main():
    import argparse
    from scripts.run_chisel_mmmu_shared import assert_remote, read_rows
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load(root / "frozen/config.json")
    assert_remote(config, root)
    rows = read_rows(config, "train")
    for method, function in (("skillopt", train_skillopt), ("autosaddler", train_autosaddler)):
        directory = root / "provenance/optimizer-factory" / method
        directory.mkdir(parents=True)
        client = SimpleNamespace(root=root, config=config, directory=directory, failed=False)
        study = SimpleNamespace(root=root, config=config, rows=rows, by_id={row["id"]: row for row in rows},
                                labels={}, client=client)
        function(study, prepare_only=True)
    print("OFFICIAL_OPTIMIZER_FACTORIES_PREPARED; no labels, model calls, scoring or optimization")


if __name__ == "__main__":
    main()
