"""Remote AIME2025 archived-public resplit: isolated training and selection."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.harness.aime_202060 import (  # noqa: E402
    MODEL_REVISION, MODEL_SEED, score_only, usage_sum, validate_data, validate_policy_freeze,
)
from scripts import run_aime_four_method as aime  # noqa: E402
from scripts import run_gaia2_jev as base  # noqa: E402
from scripts.aime_client_admission import policy  # noqa: E402
from scripts.aime_reproducible_runtime import (  # noqa: E402
    apply_reproducible_runtime, validate_reproducible_runtime, validate_server_info,
)


class ResplitStudy(aime.Study):
    def __init__(self, config, root):
        super().__init__(config, root)
        self.manifest = base.read(root / "SPLIT_MANIFEST.json")
        validate_data(self.data, self.manifest)
        self.train_ids = set(self.manifest["splits"]["train"])
        self.validation_ids = set(self.manifest["splits"]["validation"])
        self.records = []

    def infer(self, item, skill, role, maximum=32768):
        if maximum != 32768:
            raise RuntimeError("Task-agent output allowance must remain 32768")
        if item["id"] not in self.train_ids | self.validation_ids:
            raise RuntimeError("Test or unknown case reached a training/selection runner")
        split = "train" if item["id"] in self.train_ids else "validation"
        row = super().infer(item, skill, role + "_" + split, maximum=maximum)
        name = hashlib.sha256(row["raw_response"]["id"].encode()).hexdigest() + ".json"
        destination = self.root / "private_records" / split / name
        row["split"] = split
        row["system_prompt_sha256"] = hashlib.sha256(skill.encode()).hexdigest()
        base.write(destination, row)
        self.records.append(str(destination.relative_to(self.root)))
        base.write(self.root / "RECORD_INDEX.json", self.records)
        base.write(self.root / "PROGRESS.json", {
            "at_utc": base.now(), "phase": self.config["mode"], "last_split": split,
            "last_id": row["id"], "complete_native_responses": len(self.records),
            "last_usage": {key: row["usage"][key] for key in base.USAGE_FIELDS},
        })
        return score_only(row) if split == "validation" else row

    def batch(self, items, skill, role, directory, maximum=32768):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in items:
            row = self.infer(item, skill, role, maximum)
            destination = directory / "predictions" / item["id"]
            base.write(destination / "result.json", row)
            if item["id"] in self.train_ids:
                base.write(destination / "conversation.json", [
                    {"role": "user", "content": item["question"]},
                    {"type": "message", "turn": 1, "content": row["response"]},
                    {"role": "system", "content": f"Official integer accuracy: {row['hard']}"},
                ])
                (destination / "target_system_prompt.txt").write_text(skill)
                (destination / "target_user_prompt.txt").write_text(
                    self.module.USER_PROMPT_TEMPLATE.format(prompt=item["question"])
                )
                (destination / "trajectory.txt").write_text(
                    "Question:\n" + item["question"] + "\nResponse:\n" + row["response"]
                )
                (destination / "feedback.txt").write_text(row["fail_reason"])
            rows.append(row)
        base.write(directory / "RESULT.json", rows)
        return rows

    def execute(self):
        mode = self.config["mode"]
        if mode == "train_baseline":
            assigned = self.config["assigned_train_ids"]
            rows = self.batch(
                [item for item in self.data["train"] if item["id"] in assigned],
                "", "no_skill", self.root / "training",
            )
            if [row["id"] for row in rows] != assigned:
                raise RuntimeError("Training shard differs from its frozen assignment")
            result = {
                "mode": mode, "method": "no_skill", "assigned_ids": assigned,
                "correct": sum(row["hard"] for row in rows), "tasks": len(rows),
                "training_rows_sha256": base.digest(self.root / "training/RESULT.json"),
                "validation_executed": False,
            }
        elif mode in ("skillopt", "autosaddler"):
            skill = self.skillopt() if mode == "skillopt" else self.autosaddler()
            destination = self.root / "selected_skill.md"
            destination.write_text(skill)
            result = {
                "mode": mode, "method": mode, "policy_sha256": base.digest(destination),
                "official_rounds": 4, "selected_seed_unchanged": skill == aime.SEED_SKILL,
                "optimizer_data_size": {"train": 6, "validation": 6},
                "official_engine_modified": False,
            }
        elif mode == "no_skill_validation":
            rows = self.batch(self.data["development"], "", "no_skill", self.root / "selection")
            (self.root / "selected_skill.md").write_text("")
            result = {
                "mode": mode, "method": "no_skill",
                "policy_sha256": base.digest(self.root / "selected_skill.md"),
                "tasks": 6, "correct": sum(row["hard"] for row in rows),
            }
        else:
            raise ValueError("Unsupported initial campaign mode: " + mode)
        result.update(
            status="COMPLETE_PENDING_AUDIT", manifest_sha256=self.config["manifest_sha256"],
            training_ids=self.manifest["splits"]["train"], test_access=False,
            prior_exposure="Archived public resplit; not previously unseen data",
            paper_admitted=False,
        )
        base.write(self.root / "RESULT.json", result)

    def autosaddler(self):
        from autosaddler.v2.config.registry import ScenarioRegistration, build_runtime, default_registry
        from autosaddler.v2.core.domain import (
            ArtifactRef, Case, Cost, Evaluation, Observation, sha256_digest,
        )
        from autosaddler.v2.core.ports import ScenarioComponents
        from autosaddler.v2.plugins.api import ScenarioPlugin, SCENARIO_PLUGIN_API_VERSION
        from autosaddler.v2.prompting.models import Usage
        from phiagent_autosaddler.openai_function_provider import build_openai_function_provider
        from phiagent_autosaddler.text_skill import (
            TextSkillEvidenceBuilder, TextSkillHarnessSpace, TextSkillPromptPack,
        )
        from phiagent.harness.aime_transport_recovery import (
            autosaddler_completion, readable_training_evidence,
        )
        import yaml

        class PromptPack(TextSkillPromptPack):
            def _evidence(self, value):
                return json.dumps(json.loads(super()._evidence(value)), indent=2) + "\n"

            def session(self, kind, context):
                spec = super().session(kind, context)
                if kind != "diagnose_patch":
                    return spec
                files = dict(spec.workspace_files)
                files.update(readable_training_evidence(json.loads(files["training_evidence.json"])))
                return replace(
                    spec, workspace_files=files, mutation_label="aime-training-failure-patch",
                    task_prompt=spec.task_prompt
                    + "\nUse training_evidence_index.json and training_cases/ for complete readable "
                      "training-only evidence. Validation supplies selection scores, never traces.",
                )

        study = self

        def factory(*, settings, base_dir, run_dir, store, ledger):
            harness = TextSkillHarnessSpace(
                baseline_skill=aime.SEED_SKILL, store_root=run_dir / "candidates",
                materialization_root=run_dir / "materialized",
            )

            class Evaluator:
                fingerprint = "inspect-aime2025-resplit-" + study.config["scorer_sha256"]

                async def evaluate(self, candidate, cases, context):
                    materialized = harness.materialize(candidate, "evaluate")
                    try:
                        skill = base.read(materialized.root / "candidate.json")["skill"]
                    finally:
                        materialized.release()
                    private = {item["id"]: item for item in study.data[context.split]}
                    observations = []
                    for case in cases:
                        cached = context.attempt_sink.completed(
                            candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=0,
                        )
                        if cached is not None:
                            observations.append(cached)
                            continue
                        attempt, number = context.attempt_sink.start(
                            candidate_id=candidate.candidate_id, case_id=case.case_id, repetition=0,
                        )
                        started = time.monotonic()
                        try:
                            row = await asyncio.to_thread(
                                study.infer, private[case.case_id], skill, "autosaddler",
                            )
                        except BaseException:
                            context.attempt_sink.fail(attempt, "provider_failure", Cost(rollouts=1))
                            raise
                        usage = row["usage"]
                        cost = Cost(
                            rollouts=1, input_tokens=usage["prompt_tokens"],
                            output_tokens=usage["completion_tokens"],
                            wall_seconds=time.monotonic() - started,
                        )
                        context.attempt_sink.observe_usage(attempt, Usage(
                            input_tokens=usage["prompt_tokens"],
                            output_tokens=usage["completion_tokens"], total_tokens=usage["total_tokens"],
                            model=study.config["model"]["served_name"], role="actor",
                        ))
                        destination = context.artifact_dir / (
                            hashlib.sha256(case.case_id.encode()).hexdigest() + ".json"
                        )
                        base.write(destination, row)
                        metadata = {"usage": usage}
                        if context.split == "train":
                            metadata.update(
                                response="Problem:\n" + private[case.case_id]["question"]
                                + "\nResponse:\n" + row["response"],
                                predicted_answer=row["predicted_answer"], fail_reason=row["fail_reason"],
                            )
                        observation = Observation.create(
                            candidate_id=candidate.candidate_id, case_id=case.case_id,
                            split=context.split, repetition=0,
                            disposition="success" if row["hard"] else "task_failure",
                            score=float(row["hard"]), evaluator_fingerprint=self.fingerprint,
                            output=ArtifactRef(
                                uri=str(destination.relative_to(run_dir)), kind="score-only"
                                if context.split == "development" else "training-response",
                            ),
                            attempts=number, cost=cost, metadata=metadata,
                        )
                        context.attempt_sink.complete(attempt, observation, cost)
                        observations.append(observation)
                    return Evaluation(
                        evaluation_id=sha256_digest(context.operation_id),
                        candidate_id=candidate.candidate_id, split=context.split,
                        purpose=context.purpose, iteration=context.iteration,
                        requested_case_ids=tuple(case.case_id for case in cases),
                        observations=tuple(observations),
                        artifact_dir=ArtifactRef(
                            uri=str(context.artifact_dir.relative_to(run_dir)),
                            kind="evaluation-directory",
                        ),
                    )

            return ScenarioComponents(
                name="phiagent_aime_202060", version="1", harness_space=harness,
                evaluator=Evaluator(), evidence_builder=TextSkillEvidenceBuilder(store, "aime"),
                prompt_pack=PromptPack(store, benchmark="AIME 2025 train6", skill_document=aime.SEED_SKILL),
                train_cases=tuple(
                    Case(case_id=item["id"], split="train", payload={"question": item["question"]})
                    for item in study.data["train"]
                ),
                development_cases=tuple(
                    Case(case_id=item["id"], split="development", payload={})
                    for item in study.data["development"]
                ),
                required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
                resolved_entities={},
            )

        registry = default_registry()
        registry.register_scenario(ScenarioRegistration(
            plugin=ScenarioPlugin(
                name="phiagent_aime_202060", api_version=SCENARIO_PLUGIN_API_VERSION, factory=factory,
            ), source="builtin",
        ))
        registry.providers["phiagent_openai_function"] = build_openai_function_provider
        model = self.config["model"]["served_name"]
        session_timeout = 48 * 7500
        config = {
            "schema_version": "autosaddler/v2",
            "scenario": {"type": "phiagent_aime_202060", "settings": {}},
            "optimization": {
                "task_selection": {"type": "fixed", "batch_size": 6, "seed": MODEL_SEED},
                "acceptance": {"type": "matched_valid_strict_improvement"},
                "development": {"type": "full_on_accept"},
                "ranking": {"type": "mean_development_score"},
                "budget": {"max_rollouts": 100000, "max_iterations": 4},
                "diagnosis_patch_timeout_seconds": session_timeout,
                "selection_timeout_seconds": session_timeout,
                "reflection_timeout_seconds": session_timeout,
                "session_retries": 1, "session_retry_backoff_seconds": 0,
            },
            "provider": {
                "type": "phiagent_openai_function",
                "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
                "settings": {
                    "model": model, "reasoning_effort": None, "max_tool_steps": 48,
                    "temperature": 0, "max_completion_tokens": 32768, "enable_thinking": False,
                    "finish_on_stalled_reads": True,
                    "provider": {
                        "type": "openai", "base_url": self.endpoint("autosaddler_optimizer"),
                        "wire_api": "chat_completions", "model_id": model, "wire_model": model,
                    },
                },
            },
            "storage": {"type": "local", "run_root": str(self.root / "autosaddler")},
        }
        path = self.root / "AUTOSADDLER_CONFIG.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        runtime = build_runtime(path, run_id="aime2025-train6-validation6", registry=registry)
        result = runtime.engine.run()
        events = [
            json.loads(line) for line in runtime.store.events_path.read_text().splitlines() if line.strip()
        ]
        proof = autosaddler_completion(events, 4)
        proof["events_sha256"] = base.digest(runtime.store.events_path)
        base.write(self.root / "AUTOSADDLER_OPTIMIZER_AUDIT.json", proof)
        if proof["status"] != "PASS":
            raise RuntimeError("Official AutoSaddler four-round completion failed")
        base.write(self.root / "AUTOSADDLER_SELECTED.json", {
            "id": result.selected_candidate_id, "development_score": result.development_score,
            "iterations": result.iterations,
        })
        state = runtime.engine._state()
        materialized = runtime.scenario.harness_space.materialize(
            state.candidates[result.selected_candidate_id], "evaluate",
        )
        try:
            return base.read(materialized.root / "candidate.json")["skill"]
        finally:
            materialized.release()


def audit(config, root):
    ledger = base.read(root / "PROVIDER_FINAL.json")
    if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
        raise RuntimeError("Unresolved native usage; preserve evidence and block selection")
    calls = {}
    for call in ledger["calls"]:
        raw = base.read(root / "provider" / f"call-{call['sequence']:06d}" / "response.json")
        usage = base.native_usage(raw, config["model"]["served_name"], 131072)
        if usage != call["usage"] or call["status"] != "SETTLED" or raw["id"] in calls:
            raise RuntimeError("Native cost or response identity differs")
        calls[raw["id"]] = raw
    total = usage_sum([row["usage"] for row in ledger["calls"]])
    if total["total_tokens"] != ledger["known_tokens"]:
        raise RuntimeError("Provider total differs")
    data, manifest = base.read(root / "DATA.json"), base.read(root / "SPLIT_MANIFEST.json")
    validate_data(data, manifest)
    items = {item["id"]: item for split in ("train", "development") for item in data[split]}
    module = aime.official(config)
    rescored = 0
    for relative in base.read(root / "RECORD_INDEX.json"):
        row = base.read(root / relative)
        if calls.get(row["raw_response"]["id"]) != row["raw_response"]:
            raise RuntimeError("Task result differs from its paid provider response")
        hard, answer = aime.score_response(module, row["response"], items[row["id"]]["answer"])
        if hard != row["hard"] or answer != row["predicted_answer"]:
            raise RuntimeError("Official integer score differs")
        rescored += 1
    for name, digest in base.read(root / "SOURCE_FREEZE.json")["files_sha256"].items():
        if base.digest(ROOT / name) != digest:
            raise RuntimeError("Frozen source changed: " + name)
    result = base.read(root / "RESULT.json")
    result.update(
        status="COMPLETE_VERIFIED", new_usage=total, new_calls=len(calls),
        historical_costs_inherited=False, official_rescored=rescored,
    )
    base.write(root / "RESULT.json", result)
    base.write(root / "VERIFIED.json", {
        "status": "PASS", "result_sha256": base.digest(root / "RESULT.json"),
        "manifest_sha256": config["manifest_sha256"], "new_calls": len(calls),
        "official_rescored": rescored, "unknown_usage_calls": 0, "test_access": False,
    })


def package_versions(distributions):
    packages, ignored = {}, []
    for distribution in distributions:
        path = Path(str(getattr(distribution, "_path", "")))
        if path.name.startswith("._"):
            ignored.append(str(path))
            continue
        name = distribution.metadata["Name"]
        if name:
            packages[name] = distribution.version
    return packages, ignored


def run(config, root, study_type=ResplitStudy, audit_function=audit, *, final=False, data_validator=validate_data):
    if (root / "STARTED.json").exists():
        raise FileExistsError("Every attempt requires a new directory")
    if final:
        freeze = base.read(root / "POLICY_FREEZE.json")
        validate_policy_freeze(freeze, config["manifest_sha256"])
        if base.digest(root / "POLICY_FREEZE.json") != config["policy_freeze_sha256"]:
            raise RuntimeError("All-method policy freeze changed")
    data_validator(
        base.read(root / "DATA.json"), base.read(root / "SPLIT_MANIFEST.json"), final=final,
    )
    if base.digest(root / "SPLIT_MANIFEST.json") != config["manifest_sha256"]:
        raise RuntimeError("Parent split binding changed")
    transport = policy(config)
    if (
        transport != {"limit": 1, "workers": 1, "analysts": 1,
                      "provider_timeout": 7200, "client_timeout": 7500}
        or config["model"]["revision"] != MODEL_REVISION or config["seed"] != MODEL_SEED
        or config["model"]["max_tokens"] != 32768
    ):
        raise RuntimeError("Model identity, full output capacity, or serial transport changed")
    for name, digest in base.read(root / "SOURCE_FREEZE.json")["files_sha256"].items():
        if base.digest(ROOT / name) != digest:
            raise RuntimeError("Frozen source changed: " + name)
    packages, ignored = package_versions(importlib.metadata.distributions())
    base.write(root / "STARTED.json", {
        "at_utc": base.now(), "command": sys.argv, "hostname": socket.gethostname(),
        "config": config, "packages": packages, "transport": transport,
        "ignored_apple_double_resource_forks": ignored,
    })
    stopped = threading.Event()
    process = gateway = ledger = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with ExitStack() as stack:
        try:
            gpu = base.select_gpu(config, root, stack, stopped)
            assets = Path(config["model"]["assets"])
            manifest = assets / config["model"]["manifest"]
            if base.digest(manifest) != config["model"]["manifest_sha256"]:
                raise RuntimeError("Pinned model manifest changed")
            for row in base.read(manifest)["files"]:
                if base.digest(assets / "model" / row["path"]) != row["sha256"]:
                    raise RuntimeError("Pinned bf16 model shard changed")
            service = root / "service"
            (service / "w").mkdir(parents=True)
            (service / "ipc").mkdir()
            python = assets / "runtime/bin/python"
            command = [
                str(python), "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(assets / "model"), "--served-model-name", config["model"]["served_name"],
                "--host", "127.0.0.1", "--port", str(config["resource"]["model_port"]),
                "--dtype", "bfloat16", "--tensor-parallel-size", "1",
                "--max-model-len", "131072", "--max-num-seqs", "1",
                "--gpu-memory-utilization", "0.8", "--generation-config", "vllm",
                "--max-num-batched-tokens", "4096", "--gdn-prefill-backend", "triton",
                "--default-chat-template-kwargs", '{"enable_thinking":false}',
                "--attention-config", '{"backend":"FLASH_ATTN","flash_attn_version":3}',
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
                "--language-model-only", "--enforce-eager", "--seed", str(MODEL_SEED),
            ]
            if config.get("structured_output_backend"):
                if config["structured_output_backend"] != "xgrammar":
                    raise RuntimeError("Unvalidated structured decoding backend")
                structured_config = {"backend": "xgrammar"}
                if config.get("strict_structured_transport"):
                    structured_config.update(disable_any_whitespace=True, enable_in_reasoning=True)
                command.extend(("--structured-outputs-config", json.dumps(structured_config)))
            environment = base.runtime_environment(service, python, gpu_uuid=gpu["uuid"])
            environment["VLLM_RPC_BASE_PATH"] = "ipc"
            command, environment = apply_reproducible_runtime(command, environment)
            environment["VLLM_SERVER_DEV_MODE"] = "1"
            base.write(root / "REPRODUCIBLE_RUNTIME.json", validate_reproducible_runtime(command, environment))
            base.write(service / "COMMAND.json", command)
            log = stack.enter_context((service / "service.log").open("x"))
            process = subprocess.Popen(
                command, cwd=service, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            base.write(service / "OWNER.json", {"pid": process.pid, "command": command})
            deadline = time.time() + 1800
            endpoint = f"http://127.0.0.1:{config['resource']['model_port']}"
            while True:
                if process.poll() is not None or time.time() > deadline:
                    raise RuntimeError("Owned model startup failed")
                try:
                    if base.get_json(endpoint + "/v1/models")["data"]:
                        break
                except Exception:
                    time.sleep(2)
            observed = base.get_json(endpoint + "/server_info?config_format=json")
            base.write(root / "RUNTIME_OBSERVED.json", validate_server_info(
                observed, model_path=str(assets / "model"),
            ))
            if config.get("structured_output_backend"):
                structured = observed["vllm_config"]["structured_outputs_config"]
                if structured["backend"] != "xgrammar":
                    raise RuntimeError("Live native structured decoder differs from the frozen setting")
                if config.get("strict_structured_transport") and (
                    not structured["disable_any_whitespace"] or not structured["enable_in_reasoning"]
                ):
                    raise RuntimeError("Strict JSON transport was not activated in the live engine")
                base.write(root / "STRUCTURED_DECODER_OBSERVED.json", structured)
            ledger = base.Ledger(
                root, config["budget"], 131072, time.time() + config["budget"]["maximum_wall_seconds"],
            )
            gateway = aime.Gateway(config, root, ledger)
            threading.Thread(target=gateway.serve_forever, daemon=True).start()
            study_type(config, root).execute()
        except BaseException as error:
            base.write(root / "FAILURE.json", {
                "at_utc": base.now(), "error": str(error), "traceback": traceback.format_exc(),
            })
            raise
        finally:
            if gateway:
                gateway.shutdown()
                gateway.server_close()
            base.stop_owned(process)
            if ledger:
                base.write(root / "PROVIDER_FINAL.json", ledger.snapshot())
            base.write(root / "RETIREMENT.json", {"at_utc": base.now(), "gpu_state": base.gpu_snapshot()})
    audit_function(config, root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = base.read(args.config)
    root = aime.remote(config)
    try:
        run(config, root)
    except BaseException as error:
        if not (root / "FAILURE.json").exists():
            base.write(root / "FAILURE.json", {
                "at_utc": base.now(), "error": str(error), "traceback": traceback.format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
