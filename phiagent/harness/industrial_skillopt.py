"""Pinned SkillOpt with native industrial rollouts and an explicit cost-aware gate."""

from __future__ import annotations

from collections.abc import Callable
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import threading
from typing import Any

REVISION = "79124b37e9a6371e13b753f8bcd7adb1e493ade1"
SCHEMA = "official-skillopt-common-cost-gate/industrial-adapter/v1"
LABEL = "Official SkillOpt engine + common cost gate / native industrial adapter"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def summarize_native(rows: list[dict[str, Any]], expected: set[str]) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    if not expected or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Native rollout must cover each declared task exactly once")
    for row in rows:
        score, tokens = row["native_score"], row["actor_tokens"]
        if (type(score) not in (int, float) or not math.isfinite(score) or score not in (0, 1)
                or type(tokens) is not int or tokens <= 0):
            raise ValueError("The GAIA2 SkillOpt adapter needs measured binary scores and actor tokens")
    return {
        "tasks": len(rows), "quality": math.fsum(row["native_score"] for row in rows) / len(rows),
        "actor_tokens": sum(row["actor_tokens"] for row in rows),
    }


def common_cost_acceptance(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if reference["tasks"] != candidate["tasks"] or reference["tasks"] <= 0:
        raise ValueError("The common gate requires matched complete populations")
    return (
        reference["quality"] > 0
        and candidate["quality"] >= reference["quality"]
        and candidate["actor_tokens"] < reference["actor_tokens"]
    )


def select_cost_evidence(
    skill: str,
    gate_cache: dict[str, dict[str, Any]],
    complete_train_cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    key = text_hash(skill)
    return gate_cache.get(key) or complete_train_cache.get(key)


def restore_patch_order(observed, checkpoint):
    def key(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if Counter(map(key, observed)) != Counter(map(key, checkpoint)):
        raise ValueError("Replayed official reflection differs from its completed checkpoint")
    return deepcopy(checkpoint)


def verify_engine(source: Path, manifest_sha256: str | None = None) -> dict[str, Any]:
    critical = (
        "skillopt/engine/trainer.py", "skillopt/evaluation/gate.py", "skillopt/envs/base.py",
        "skillopt/model/qwen_backend.py", "configs/_base_/default.yaml",
    )
    if (source / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
        ).strip()
        if revision != REVISION:
            raise ValueError("Only the pinned official SkillOpt engine is permitted")
        changes = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", "skillopt", "configs"], text=True,
        )
        if changes.strip():
            raise ValueError("The official engine checkout contains changed or untracked code/configuration")
        expected_files = None
    else:
        path = source / "OFFICIAL_SOURCE.json"
        if manifest_sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() != manifest_sha256:
            raise ValueError("A pinned Git archive requires its independently frozen source manifest hash")
        manifest = json.loads(path.read_text())
        if manifest["revision"] != REVISION or manifest["repository"] != "microsoft/SkillOpt":
            raise ValueError("The archive does not identify the pinned official engine")
        expected_files = manifest["files_sha256"]
        if not set(critical) <= set(expected_files):
            raise ValueError("The archive omits a required official engine component")
        revision = REVISION
    hashes = {}
    for name in expected_files if expected_files is not None else critical:
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Official source path escapes the frozen engine")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_files is not None and hashes[name] != expected_files[name]:
            raise ValueError(f"The pinned engine archive changed: {name}")
    if "skillopt" in sys.modules:
        loaded = Path(sys.modules["skillopt"].__file__).resolve()
        if not loaded.is_relative_to(source.resolve()):
            raise RuntimeError("A different SkillOpt installation is already imported")
    return {"repository": "microsoft/SkillOpt", "revision": revision, "files_sha256": hashes}


@contextmanager
def patched_attributes(changes):
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in changes]
    try:
        for owner, name, value in changes:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


def optimize(
    *, engine_source: Path, output: Path, training_tasks: list[dict[str, Any]],
    native_rollout: Callable[[list[dict[str, Any]], str, Path], list[dict[str, Any]]],
    optimizer_call: Callable[[list[dict[str, Any]], int, str], dict[str, Any]],
    epochs: int = 4, seed: int = 20260928,
    source_manifest_sha256: str | None = None,
    reflection_checkpoint: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep optimizer feedback/gates in TRAIN; protected VAL is never supplied here."""
    manifest = verify_engine(engine_source, source_manifest_sha256)
    expected = {row["id"] for row in training_tasks}
    if len(expected) != len(training_tasks) or len(expected) < 2:
        raise ValueError("The official optimizer needs a unique, nonempty TRAIN population")
    if any(row.get("partition") != "train" for row in training_tasks):
        raise ValueError("Protected validation and TEST cannot enter SkillOpt optimization")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("The official epoch count must be declared before execution")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(engine_source))
    from skillopt.config import flatten_config, load_config
    from skillopt.engine import trainer
    from skillopt.envs.base import EnvAdapter
    from skillopt.evaluation import gate as official_gate
    from skillopt.model import qwen_backend

    gate_cache: dict[str, dict[str, Any]] = {}
    complete_train_cache: dict[str, dict[str, Any]] = {}
    gate_records, model_records, rollout_records, failures = [], [], [], []
    records_lock = threading.RLock()
    checkpoint_values = (
        [] if reflection_checkpoint is None
        else reflection_checkpoint if isinstance(reflection_checkpoint, list)
        else [reflection_checkpoint]
    )
    checkpoints = []
    for checkpoint in checkpoint_values:
        checkpoint = deepcopy(checkpoint)
        if "raw_patches_json" in checkpoint:
            checkpoint["raw_patches"] = [
                json.loads(value) for value in checkpoint.pop("raw_patches_json")
            ]
        checkpoints.append(checkpoint)
    restored_reflections = 0
    restored_candidates = 0

    def save(name, value):
        from phiagent.data_engine.provenance import write_json_atomic
        with records_lock:
            write_json_atomic(output / name, value)

    def failed(stage, error):
        with records_lock:
            failures.append({"stage": stage, "type": type(error).__name__, "error": str(error)})
            save("FAILURES.json", failures)

    class Batch:
        def __init__(self, items, gate):
            self.items, self.gate = items, gate

        def __len__(self):
            return len(self.items)

    class Adapter(EnvAdapter):
        analyst_workers = 2
        failure_only = False
        minibatch_size = 8
        edit_budget = 4

        @property
        def _env_name(self):
            return ""

        def build_train_env(self, batch_size, seed, **kwargs):
            ordered = list(training_tasks)
            random.Random(seed).shuffle(ordered)
            return Batch(ordered[:min(batch_size, len(ordered))], False)

        def build_eval_env(self, env_num, split, seed, **kwargs):
            if split not in ("optimizer_gate", "valid_seen"):
                raise ValueError(f"No protected or undeclared evaluation split is available: {split}")
            if env_num != len(training_tasks):
                raise ValueError("The common optimizer gate requires complete primary TRAIN")
            return Batch(list(training_tasks), True)

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            nonlocal restored_candidates
            if failures:
                raise RuntimeError("A recorded native/provider failure blocks further optimizer traffic")
            if env_manager.gate and skill_content and restored_candidates < len(checkpoints):
                checkpoint = checkpoints[restored_candidates]
                if text_hash(skill_content) != checkpoint["candidate_sha256"]:
                    raise RuntimeError("The official continuation did not restore its exact pre-gate candidate")
                restored_candidates += 1
                save("CANDIDATE_CHECKPOINT_REUSED.json", {
                    "candidate_sha256": text_hash(skill_content),
                    "reused_candidates": restored_candidates,
                    "new_optimizer_calls_for_replay": 0,
                })
            folder = Path(out_dir)
            folder.mkdir(parents=True, exist_ok=True)
            try:
                rows = native_rollout(env_manager.items, skill_content, folder)
                summary = summarize_native(rows, {row["id"] for row in env_manager.items})
            except Exception as error:
                failed("native_rollout", error)
                raise
            results = []
            for row in rows:
                destination = folder / "predictions" / row["id"]
                destination.mkdir(parents=True, exist_ok=True)
                if not isinstance(row.get("public_feedback"), str) or not row["public_feedback"]:
                    raise ValueError("SkillOpt reflection needs real public execution feedback")
                (destination / "conversation.json").write_text(json.dumps([
                    {"role": "user", "content": row["public_prompt"]},
                    {"role": "assistant", "content":
                     "Recorded native execution feedback (not a new model response):\n"
                     + row["public_feedback"]
                     + f"\nComplete actor input-plus-output tokens: {row['actor_tokens']}"},
                ], ensure_ascii=False) + "\n")
                results.append({
                    **row, "hard": row["native_score"], "soft": row["native_score"],
                    "task_type": "native_industrial",
                })
            with records_lock:
                rollout_records.append({
                    "directory": str(folder), "skill_sha256": text_hash(skill_content),
                    "primary_split": "train", "optimizer_gate": env_manager.gate, **summary,
                })
                complete_train_cache[text_hash(skill_content)] = summary
                if env_manager.gate:
                    gate_cache[text_hash(skill_content)] = summary
                save("NATIVE_ROLLOUTS.json", rollout_records)
            return results

        def get_task_types(self):
            return ["native_industrial"]

        def reflect(self, results, skill_content, out_dir, **kwargs):
            nonlocal restored_reflections
            values = super().reflect(results, skill_content, out_dir, **kwargs)
            if restored_reflections < len(checkpoints):
                checkpoint = checkpoints[restored_reflections]
                values = restore_patch_order(values, checkpoint["raw_patches"])
                restored_reflections += 1
                save("REFLECTION_CHECKPOINT_REUSED.json", {
                    "source": checkpoint["source"], "raw_patch_count": len(values),
                    "reused_reflections": restored_reflections,
                    "official_patch_order_preserved": True,
                })
            return values

    def cost_gate(
        candidate_skill, cand_hard, current_skill, current_score, best_skill,
        best_score, best_step, global_step, **kwargs,
    ):
        current = select_cost_evidence(current_skill, gate_cache, complete_train_cache)
        candidate = select_cost_evidence(candidate_skill, gate_cache, complete_train_cache)
        best = select_cost_evidence(best_skill, gate_cache, complete_train_cache)
        if current is None or candidate is None or best is None:
            raise RuntimeError("A SkillOpt gate lacks complete native cost evidence")
        if not math.isclose(candidate["quality"], cand_hard, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Official gate quality differs from native outcomes")
        accepted = candidate_skill != current_skill and common_cost_acceptance(current, candidate)
        new_best = accepted and (
            candidate["quality"] > best["quality"]
            or candidate["quality"] == best["quality"] and candidate["actor_tokens"] < best["actor_tokens"]
        )
        result = official_gate.GateResult(
            action="accept_new_best" if new_best else "accept" if accepted else "reject",
            current_skill=candidate_skill if accepted else current_skill,
            current_score=cand_hard if accepted else current_score,
            best_skill=candidate_skill if new_best else best_skill,
            best_score=cand_hard if new_best else best_score,
            best_step=global_step if new_best else best_step,
        )
        gate_records.append({
            "global_step": global_step, "action": result.action,
            "candidate_sha256": text_hash(candidate_skill),
            "current_sha256": text_hash(current_skill),
            "candidate": candidate, "current": current,
            "metric": "native quality nondecrease AND strictly lower complete actor tokens",
        })
        save("COMMON_GATE.json", gate_records)
        return result

    signature = inspect.signature(qwen_backend._chat_messages_impl)

    def native_transport(*args, **kwargs):
        if failures:
            raise RuntimeError("A recorded native/provider failure blocks further optimizer traffic")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values["role"] != "optimizer" or values.get("tools"):
            error = ValueError("Only optimizer text calls use this transport; actors use their native harness")
            failed("unsupported_model_route", error)
            raise error
        try:
            reply = optimizer_call(values["messages"], values["max_completion_tokens"], values["stage"])
        except Exception as error:
            failed(values["stage"], error)
            raise
        text, usage = reply["text"], reply["usage"]
        if (not isinstance(text, str) or not text.strip()
                or any(type(usage.get(key)) is not int or usage[key] < 0
                       for key in ("input_tokens", "output_tokens", "total_tokens"))
                or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]):
            error = ValueError("The optimizer returned no valid text or complete actual usage")
            failed(values["stage"], error)
            raise error
        qwen_backend.tracker.record(values["stage"], usage["input_tokens"], usage["output_tokens"])
        with records_lock:
            model_records.append({"stage": values["stage"], "usage": usage, "receipt": reply["receipt"]})
            save("OPTIMIZER_CALLS.json", model_records)
        result = (
            qwen_backend._compat_message_from_payload({"content": text}, {"finish_reason": "stop"})
            if values["return_message"] else text
        )
        return result, {
            "prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }

    seed_file = output / "initial_skill.md"
    seed_file.write_text("")
    cfg = flatten_config(load_config(str(engine_source / "configs/_base_/default.yaml")))
    cfg.update(
        env_name="native_industrial", out_root=str(output / "engine"),
        skill_init=str(seed_file), train_size=len(training_tasks), batch_size=len(training_tasks),
        accumulation=1, num_epochs=epochs, seed=seed, eval_test=False,
        sel_env_num=len(training_tasks), test_env_num=0, sel_split="optimizer_gate",
        backend="qwen_chat", optimizer_backend="qwen_chat", target_backend="qwen_chat",
        optimizer_model="deepseek-flash", target_model="deepseek-flash",
        use_gate=True, gate_metric="hard", use_semantic_density=False,
        use_slow_update=True, slow_update_gate_with_selection=True, use_meta_skill=True,
        analyst_workers=2,
    )
    save("CONFIG.json", {
        "schema": SCHEMA, "implementation_label": LABEL, "official_revision": REVISION,
        "official_source": manifest, "engine_config": cfg,
        "optimizer_gate_population": "complete primary TRAIN; not reported as held-out validation",
        "protected_validation_supplied": False, "test_access": False,
        "cumulative_token_limit": None,
        "gate_extension_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    changes = [
        (trainer, "evaluate_gate", cost_gate),
        (official_gate, "evaluate_gate", cost_gate),
        (qwen_backend, "_chat_messages_impl", native_transport),
    ]
    with patched_attributes(changes):
        result = trainer.ReflACTTrainer(cfg, Adapter()).train()
    if (failures or not model_records
            or {row["epoch"] for row in result["epoch_stats"]} != set(range(1, epochs + 1))
            or result["best_selection_hard"] is None):
        raise RuntimeError("The official SkillOpt outer loop did not complete")
    if any(result.get(key) is not None for key in ("baseline_test_hard", "test_hard", "final_test_hard")):
        raise RuntimeError("The optimizer must not execute or report a TEST phase")
    selected = output / "engine/best_skill.md"
    if not selected.is_file():
        raise RuntimeError("The official engine did not export its selected skill")
    chosen = gate_cache.get(text_hash(selected.read_text()))
    if chosen is None or not math.isclose(
        chosen["quality"], result["best_selection_hard"], rel_tol=0, abs_tol=1e-12,
    ):
        raise RuntimeError("The selected official skill lacks matching native gate evidence")
    verify_engine(engine_source, source_manifest_sha256)
    summary = {
        "schema": SCHEMA, "implementation_label": LABEL, "official_revision": REVISION,
        "official_engine_completed": True, "selected_skill": selected.read_text(),
        "selected_skill_sha256": text_hash(selected.read_text()), "engine_return": result,
        "native_actor_tokens": sum(row["actor_tokens"] for row in rollout_records),
        "optimizer_tokens": sum(row["usage"]["total_tokens"] for row in model_records),
        "optimizer_calls": len(model_records), "native_rollout_batches": len(rollout_records),
        "protected_validation_supplied": False, "test_access": False, "paper_admitted": False,
    }
    save("COMPLETE.json", summary)
    return deepcopy(summary)
