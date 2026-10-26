#!/usr/bin/env python3
"""Optional native transport and task adapter for the pinned complete SkillOpt core."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
from pathlib import Path
import threading
import time
import traceback


IMAGE_STAGES = frozenset({"analyst", "slow_update", "meta_skill"})
TEXT_STAGES = frozenset({"merge", "ranking"})


class NativeBridge:
    def __init__(self, core, client, original_chat):
        self.core = core
        self.client = client
        self.root = client.root
        self.original_chat = original_chat
        self.lock = threading.RLock()
        self.scope = ContextVar("skillopt_native_image_scope", default=None)
        self.scope_dispatches = ContextVar("skillopt_native_scope_dispatches", default=None)
        self.tracker = None
        self.stage_calls = {}

    def fail(self, stage, error):
        path = self.root / "skillopt-bridge-failure.json"
        if not path.exists():
            self.core.write(path, {
                "stage": stage, "error": repr(error), "traceback": traceback.format_exc(),
                "at_unix": time.time(), "retry_permitted": False,
            })

    def require_integrity(self):
        if (self.root / "skillopt-bridge-failure.json").exists():
            raise RuntimeError("Preserved SkillOpt bridge failure blocks further native dispatch")
        self.core.require_request_integrity(self.root, self.client.ledger)

    def image_parts(self, example_ids):
        if (not isinstance(example_ids, (list, tuple)) or
                any(not isinstance(value, str) or not value for value in example_ids) or
                len(set(example_ids)) != len(example_ids)):
            raise ValueError("Native image binding requires distinct explicit example identities")
        parts = []
        for example_id in example_ids:
            chat = self.original_chat(example_id)
            urls = self.core.image_urls(chat)
            parts.append({"type": "text", "text": f"Original images for trajectory id={example_id}:"})
            parts.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
        return parts

    def dispatch(self, chat, role, metadata):
        with self.lock:
            self.require_integrity()
            stage = metadata["stage"]
            try:
                expected_chat = [{"role": "user", "content": self.image_parts(metadata["example_ids"])}]
                expected = self.core.image_urls(expected_chat)
                if self.core.image_urls(chat) != expected:
                    raise ValueError("Native request dropped, changed or reordered original images")
                if role == "reflection":
                    self.client.expected_reflection_urls = expected
                record = self.client.request(chat, role, metadata)
                self.require_integrity()
                usage = record["usage"]
                if self.tracker is not None:
                    self.tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
                self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
                return record
            except BaseException as error:
                self.fail(stage, error)
                raise
            finally:
                if role == "reflection":
                    self.client.expected_reflection_urls = None

    def optimizer(self, system, user, max_completion_tokens=16384, retries=5,
                  stage="optimizer", reasoning_effort=None, timeout=None):
        self.require_integrity()
        try:
            if stage not in IMAGE_STAGES | TEXT_STAGES:
                raise ValueError("Unbound optimizer stage: " + str(stage))
            if max_completion_tokens != self.client.config["model"]["reflection"]["max_tokens"]:
                raise ValueError("Requested optimizer cap differs from the frozen native profile")
            if reasoning_effort not in (None, ""):
                raise ValueError("The frozen native profile does not enable a reasoning-effort override")
            scope = self.scope.get()
            if stage in IMAGE_STAGES:
                if scope is None or scope[0] != stage or not scope[1]:
                    raise ValueError("Image-bearing optimizer stage lacks its exact example scope")
                example_ids = list(scope[1])
            else:
                if scope is not None:
                    raise ValueError("Text-only edit operator inherited an unrelated image scope")
                example_ids = []
            if any(not example_id.startswith("dev_") for example_id in example_ids):
                raise ValueError("Held-out examples cannot enter the SkillOpt optimizer")
            content = [{"type": "text", "text": user}] + self.image_parts(example_ids)
            chat = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
            record = self.dispatch(chat, "reflection", {
                "stage": stage, "example_ids": example_ids,
                "official_requested_retries": retries, "effective_native_retries": 0,
                "official_requested_timeout": timeout,
                "official_requested_reasoning_effort": reasoning_effort,
                "image_scope_bound_before_dispatch": True,
            })
            dispatches = self.scope_dispatches.get()
            if dispatches is not None:
                dispatches.append(record)
            return record["raw_content"], record["usage"]
        except BaseException as error:
            with self.lock:
                self.fail(stage, error)
            raise

    def scoped(self, function, stage, argument):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            self.require_integrity()
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            items = bound.arguments[argument]
            if not isinstance(items, list) or not items:
                error = ValueError("Optimizer image scope requires a nonempty explicit example list")
                self.fail(stage, error)
                raise error
            example_ids = tuple(item["id"] for item in items)
            token = self.scope.set((stage, example_ids))
            dispatches = []
            dispatch_token = self.scope_dispatches.set(dispatches)
            try:
                result = function(*args, **kwargs)
                self.require_integrity()
                if len(dispatches) != 1:
                    raise RuntimeError("Official image-bearing stage returned without exactly one native request")
                return result
            except BaseException as error:
                with self.lock:
                    self.fail(stage, error)
                raise
            finally:
                self.scope_dispatches.reset(dispatch_token)
                self.scope.reset(token)

        return wrapped

    @contextmanager
    def installed(self):
        from skillopt.engine import trainer
        from skillopt.gradient import reflect
        from skillopt.model import openai_compatible_backend as backend

        if tuple(inspect.signature(self.optimizer).parameters) != tuple(
                inspect.signature(backend.chat_optimizer).parameters):
            raise RuntimeError("Native optimizer signature differs from the pinned backend protocol")
        bindings = [
            (backend, "chat_optimizer", self.optimizer),
            (reflect, "run_error_analyst_minibatch",
             self.scoped(reflect.run_error_analyst_minibatch, "analyst", "items")),
            (reflect, "run_success_analyst_minibatch",
             self.scoped(reflect.run_success_analyst_minibatch, "analyst", "items")),
            (trainer, "run_slow_update",
             self.scoped(trainer.run_slow_update, "slow_update", "comparison_pairs")),
            (trainer, "run_meta_skill",
             self.scoped(trainer.run_meta_skill, "meta_skill", "comparison_pairs")),
        ]
        originals = [(module, name, getattr(module, name)) for module, name, _ in bindings]
        self.tracker = backend.tracker
        try:
            for module, name, replacement in bindings:
                setattr(module, name, replacement)
            yield
            self.require_integrity()
        finally:
            for module, name, original in reversed(originals):
                setattr(module, name, original)
            self.tracker = None


def make_adapter(core, root, config, rows, train, selection, inputs, scorer, answers, bridge):
    from skillopt.datasets.base import SplitDataLoader
    from skillopt.envs.base import EnvAdapter

    allowed = {row["id"] for row in rows}
    if (len(allowed) != len(rows) or set(answers) != allowed or
            {row["id"] for row in train} | {row["id"] for row in selection} != allowed or
            {row["id"] for row in train} & {row["id"] for row in selection}):
        raise ValueError("SkillOpt requires a disjoint complete development partition")

    class SealedLoader(SplitDataLoader):
        def get_split_items(self, split):
            if split not in {"train", "val", "valid_seen", "selection"}:
                raise ValueError("Official held-out validation is sealed until candidate freeze")
            return super().get_split_items(split)

    split_dir = root / "optimizer-data"
    for split, items in (("train", train), ("val", selection), ("test", [])):
        core.write(split_dir / split / "items.json", items)
    loader = SealedLoader(split_dir=str(split_dir), split_mode="split_dir",
                          seed=config["official_skillopt"]["seed"])

    class MMMUAdapter(EnvAdapter):
        def setup(self, cfg):
            super().setup(cfg)
            self.analyst_workers = cfg["analyst_workers"]
            self.failure_only = cfg["failure_only"]
            self.minibatch_size = cfg["minibatch_size"]
            self.edit_budget = cfg["edit_budget"]
            loader.setup(cfg)

        def get_dataloader(self):
            return loader

        def build_env_from_batch(self, batch, **kwargs):
            items = list(batch.payload)
            if len(items) != batch.batch_size or any(item["id"] not in allowed for item in items):
                raise ValueError("Trainer batch escaped its sealed development identities")
            return items

        def build_train_env(self, batch_size, seed, **kwargs):
            return self.build_env_from_batch(loader.build_train_batch(batch_size, seed, **kwargs))

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return self.build_env_from_batch(loader.build_eval_batch(env_num, split, seed, **kwargs))

        def get_task_types(self):
            return sorted({row["subject"] for row in rows})

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            bridge.require_integrity()
            try:
                if kwargs.get("diagnostic_mode") or kwargs.get("diagnostic_instruction"):
                    raise ValueError("No undeclared diagnostic instruction may change the actor prompt")
                items = list(env_manager)
                if not items or any(item["id"] not in allowed for item in items):
                    raise ValueError("Only sealed development rows may enter training rollout")
                directory = Path(out_dir).resolve()
                stage = str(directory.relative_to(root / "optimizer"))
                outputs, conversations = [], []
                for row in items:
                    chat = inputs.chat(row, {"instruction": skill_content})
                    output = bridge.dispatch(chat, "actor", {
                        "stage": "actor:" + stage, "example_ids": [row["id"]],
                        "skill_sha256": core.digest({"instruction": skill_content}),
                    })
                    outputs.append(output)
                    conversation = []
                    for message in chat:
                        parts = []
                        for part in message["content"]:
                            if part["type"] == "text":
                                parts.append(part["text"])
                            elif part["type"] == "image_url":
                                parts.append("[Original image; pixels are bound separately for reflection]")
                            else:
                                raise ValueError("Unknown actor content cannot be omitted from its trace")
                        conversation.append({"role": message["role"], "content": "\n".join(parts)})
                    conversation.append({"role": "assistant", "content": output["raw_content"]})
                    conversations.append(conversation)
                scores, summary = scorer.score(items, outputs, answers, "dev")
                if len(scores) != len(items) or any(score not in (0.0, 1.0) for score in scores):
                    raise ValueError("MMMU official scorer must return one binary judgment per row")
                results = []
                for row, output, conversation, score in zip(items, outputs, conversations, scores):
                    result = {
                        "id": row["id"], "task_type": row["subject"],
                        "task_description": row["question"], "question": row["question"],
                        "hard": int(score), "soft": float(score), "agent_ok": True, "n_turns": 1,
                        "response": output["raw_content"], "predicted_answer": output["raw_content"],
                        "gold_answer": answers[row["id"]],
                        "reference_text": core.canonical({"official_answer": answers[row["id"]]}).decode(),
                        "fail_reason": "" if score else "Incorrect under the unmodified MMMU official scorer",
                    }
                    core.write(directory / "predictions" / row["id"] / "conversation.json", conversation)
                    core.write(directory / "predictions" / row["id"] / "result.json", result)
                    results.append(result)
                core.write(directory / "native-official-metric.json", summary)
                bridge.require_integrity()
                return results
            except BaseException as error:
                bridge.fail("rollout", error)
                raise

        def reflect(self, *args, **kwargs):
            bridge.require_integrity()
            try:
                result = super().reflect(*args, **kwargs)
                bridge.require_integrity()
                return result
            except BaseException as error:
                bridge.fail("reflect", error)
                raise

    return MMMUAdapter()
