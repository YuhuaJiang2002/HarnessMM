"""Remote-only 32/32/96 GAIA2 stages; no held-out feedback enters optimizers."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gaia2_four_method as four  # noqa: E402
from scripts import run_gaia2_jev as base  # noqa: E402
from scripts import run_aime_four_method as lifecycle  # noqa: E402
from phiagent.harness.gaia2_202060_protocol import (  # noqa: E402
    digest, items_gate, native_population_gate, partition_gate, test_gate,
)


class Study(four.Study):
    def infer(self, item, skill, role, maximum=None):
        validation = item["id"] in self.config.get("validation_ids", [])
        if validation and self.config["method"] != "autosaddler":
            raise RuntimeError("Held-out scoring is not enabled for this optimizer stage")
        if not validation and item["id"] not in self.config["training_ids"]:
            raise RuntimeError("Optimizer feedback may only contain the new training scenarios")
        if validation:
            candidate_hash = hashlib.sha256(skill.encode()).hexdigest()
            freeze = self.root / "VALIDATION_CANDIDATE_FREEZES" / (candidate_hash + ".json")
            freeze.parent.mkdir(exist_ok=True)
            if not freeze.exists():
                base.write(freeze, {"skill": skill, "skill_sha256": candidate_hash,
                           "at_utc": base.now(), "frozen_before_native_validation": True})
        directory = self.root / ("validation_quarantine" if validation else "rollouts") / __import__("uuid").uuid4().hex
        rows, usage = self.rollout([item], skill, directory, repetitions=3)
        summary = native_population_gate(rows, [item["id"]])
        if validation:
            return {"id": item["id"], "question": "Held-out native scalar selection only",
                    "response": "", "hard": summary["successes"] / 3,
                    "soft": summary["successes"] / 3, "task_type": "gaia2", "agent_ok": True,
                    "n_turns": 0, "fail_reason": "", "usage": usage, "public_messages": [],
                    "candidate_freeze_sha256": base.digest(freeze), "validation_trace_feedback": False}
        public = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()
                  if line.strip()]
        public = [row for row in public if row["source"] == "public_actor_messages"]
        response = "\n\n".join(row["response"] for row in public)
        question = "\n".join(str(row["content"]) for row in public[0]["messages"]
                             if row["role"] == "user") if public else item["question"]
        score = summary["successes"] / 3
        return {"id": item["id"], "question": question, "response": response,
                "hard": score, "soft": score, "task_type": "gaia2", "agent_ok": True,
                "n_turns": len(public), "fail_reason": "" if score == 1 else
                f"Native training successes: {summary['successes']}/3 repetitions.",
                "usage": usage, "trajectory": str(directory),
                "public_messages": public[-1]["messages"] if public else [],
                "native_repetition_scores": [row["success_bool"] for row in rows]}

    def execute(self):
        stage = self.config["stage"]
        method = self.config["method"]
        manifest = base.read(self.root / "SPLIT_MANIFEST.json")
        partition_gate(manifest)
        if self.data.get("test") and stage != "test":
            raise RuntimeError("Test payload must not be loaded into a training/validation stage")
        if stage == "test":
            proof = base.read(self.root / "ALL_METHOD_FREEZE.json")
            test_gate(manifest, proof["selections"])
            if base.digest(self.root / "ALL_METHOD_FREEZE.json") != self.config["all_method_freeze_sha256"]:
                raise RuntimeError("All-method freeze changed")
        if stage == "optimize":
            items_gate(self.data["train"], manifest, "train")
            if method == "autosaddler":
                items_gate(self.data["development"], manifest, "validation")
                install_feedback_firewall(self.config, self.root)
            else:
                if self.data["development"]:
                    raise RuntimeError("Held-out validation is forbidden inside this optimizer loop")
                self.data["development"] = deepcopy(self.data["train"])
            base.write(self.root / "TRAINING_ACCESS.json", {
                "train_ids": self.config["training_ids"],
                "optimizer_internal_selection": "quarantined validation scalars" if method == "autosaddler"
                    else "train only",
                "true_validation_not_loaded": method != "autosaddler", "native_repetitions": 3,
                "autosaddler_validation": "Disjoint32; only scalar ranking of content-frozen candidates; "
                    "raw traces quarantined and development fields removed from every optimizer workspace"})
            base.write(self.root / "PROGRESS.json", {
                "phase": "OFFICIAL_FOUR_ROUND_TRAIN32", "method": method, "at_utc": base.now()})
            skill = getattr(self, method)()
            candidates = {"selected": skill, "seed": four.SEED_SKILL}
            if method == "skillopt":
                for path in sorted((self.root / "skillopt").rglob("skill.md")):
                    candidates[str(path.relative_to(self.root / "skillopt"))] = path.read_text()
                for path in sorted((self.root / "skillopt").rglob("skill_*.md")):
                    candidates[str(path.relative_to(self.root / "skillopt"))] = path.read_text()
            else:
                for path in sorted((self.root / "autosaddler").rglob("candidate.json")):
                    value = base.read(path)
                    if isinstance(value.get("skill"), str):
                        candidates[str(path.relative_to(self.root / "autosaddler"))] = value["skill"]
            frozen = {}
            for origin, content in candidates.items():
                key = hashlib.sha256(content.encode()).hexdigest()
                if key not in frozen:
                    frozen[key] = {"skill": content, "skill_sha256": key, "origin": origin}
            base.write(self.root / "TRAINING_CANDIDATES.json", {
                "method": method, "partition_sha256": digest(manifest), "candidates": frozen,
                "training_complete": True, "validation_accessed": method == "autosaddler",
                "validation_trace_feedback": False, "test_accessed": False})
            base.write(self.root / "RESULT.json", {
                "stage": stage, "method": method, "training_complete": True,
                "candidate_count": len(frozen), "candidates_sha256":
                base.digest(self.root / "TRAINING_CANDIDATES.json"),
                "test_executions": 0, "validation_executions": sum(
                    len(base.read(path)["rows"]) for path in
                    (self.root / "validation_quarantine").glob("*/NATIVE_RESULT.json"))})
            return
        split = self.config["evaluation_split"]
        items = self.data[{"train": "train", "validation": "development", "test": "test"}[split]]
        items_gate(items, manifest, split, complete=False)
        skill = self.config.get("frozen_skill", "")
        mode = "native" if method == "native" else "jev" if method == "chisel" else "cot"
        base.write(self.root / "PROGRESS.json", {
            "phase": stage.upper(), "split": split, "scenarios": len(items), "at_utc": base.now()})
        rows, usage = self.rollout(items, skill, self.root / "evaluation", mode=mode, repetitions=3)
        result = native_population_gate(rows, [item["id"] for item in items])
        base.write(self.root / "RESULT.json", {
            **result, "stage": stage, "method": method, "split": split, "actor_usage": usage,
            "native_result_sha256": base.digest(self.root / "evaluation/NATIVE_RESULT.json"),
            "candidate_id": self.config.get("candidate_id"),
            "partition_sha256": digest(manifest),
            "test_executions": len(rows) if split == "test" else 0, "paper_admitted": False})


def install_feedback_firewall(config, root):
    import phiagent_autosaddler.text_skill as adapter
    from phiagent.harness.gaia2_202060_firewall import optimizer_workspace
    original = adapter.TextSkillPromptPack

    class TrainingOnlyPromptPack(original):
        def session(self, kind, context):
            spec = super().session(kind, context)
            files, proof = optimizer_workspace(spec.workspace_files, config["validation_ids"])
            with (root / "OPTIMIZER_FIREWALL.jsonl").open("a") as stream:
                stream.write(json.dumps({"kind": kind, **proof, "at_utc": base.now(),
                    "workspace_sha256": digest(files)}) + "\n")
            return replace(spec, workspace_files=files)

    adapter.TextSkillPromptPack = TrainingOnlyPromptPack


def audit(config, root):
    ledger = base.read(root / "PROVIDER_FINAL.json")
    if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
        raise RuntimeError("Unsettled provider usage")
    total = 0
    roles = {}
    for call in ledger["calls"]:
        folder = root / "provider" / f"call-{call['sequence']:06d}"
        if call["status"] == "REJECTED_BEFORE_FORWARD":
            if base.read(folder / "REJECTED_BEFORE_FORWARD.json")["model_calls"] != 0:
                raise RuntimeError("Unproven pre-inference rejection")
            continue
        if call["status"] != "SETTLED":
            raise RuntimeError("Unsettled native call")
        usage = base.native_usage(base.read(folder / "response.json"),
                                  config["model"]["served_name"], 131072)
        request = base.read(folder / "request.json")
        if call["role"].endswith("_optimizer") and any(
            ident in json.dumps(request["json"]) for ident in config.get("validation_ids", [])
        ):
            raise RuntimeError("Validation identities entered an optimizer model request")
        if usage != call["usage"] or request["role"] != call["role"]:
            raise RuntimeError("Provider evidence binding changed")
        if config.get("cot_jev"):
            from phiagent.harness.gaia2_cot_jev import audit_native_output_contract
            audit_native_output_contract(
                request, base.read(folder / "response.json"), config,
            )
        elif call["role"].endswith(("/actor", "/judge")) and request["json"]["max_tokens"] != 4096:
            raise RuntimeError("Native actor/judge output capacity changed")
        total += usage["total_tokens"]
        kind = call["role"].rsplit("/", 1)[-1]
        roles[kind] = roles.get(kind, 0) + usage["total_tokens"]
    if total != ledger["known_tokens"]:
        raise RuntimeError("All paid native costs must be counted")
    result = base.read(root / "RESULT.json")
    for name, expected in base.read(root / "SOURCE_FREEZE.json")["files_sha256"].items():
        if base.digest(ROOT / name) != expected:
            raise RuntimeError("Frozen source drift: " + name)
    if config["stage"] != "optimize":
        items = base.read(root / "DATA.json")[
            {"train": "train", "validation": "development", "test": "test"}[config["evaluation_split"]]]
        rows = base.read(root / "evaluation/NATIVE_RESULT.json")["rows"]
        native_population_gate(rows, [item["id"] for item in items])
        for row in rows:
            verdict = base.read(Path(row["export_path"]))["metadata"]["annotation"]["validation_decision"]
            if verdict not in ("Valid", "Invalid") or (verdict == "Valid") != row["success_bool"]:
                raise RuntimeError("Native scorer/export disagreement")
    base.write(root / "VERIFIED.json", {
        "status": "PASS", "result_sha256": base.digest(root / "RESULT.json"),
        "partition_sha256": result.get("partition_sha256"), "known_tokens": total,
        "token_roles": roles, "unknown_usage_calls": 0,
        "test_executions": result.get("test_executions", 0)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "worker", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    config = base.read(args.config)
    four.require_remote(config)
    root = Path(config["run_directory"])
    partition_gate(base.read(root / "SPLIT_MANIFEST.json"))
    os.environ.update(DEMO_FS_PATH=config["filesystem"], FS_PATH=str(root / "filesystem-cache"),
                      OPENAI_API_KEY="EMPTY", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    if args.action == "worker":
        if config["method"] == "chisel":
            from phiagent.harness.gaia2_202060_progress import install_adapter
            four.install_adapter = install_adapter
        four.worker(config, base.read(args.job))
        return
    if args.action == "audit":
        audit(config, root)
        return
    original = base.runtime_environment

    def environment(service, python, **kwargs):
        env = original(service, python, **kwargs)
        rpc = root.parent / ("rpc202060-" + hashlib.sha256(str(root).encode()).hexdigest()[:10])
        rpc.mkdir(mode=0o700, exist_ok=False)
        env["VLLM_RPC_BASE_PATH"] = str(rpc)
        base.write(root / "RPC_PATH.json", {"path": str(rpc), "owner": str(root)})
        return env

    base.runtime_environment = environment
    lifecycle.Study, lifecycle.Gateway, lifecycle.audit = Study, four.Gateway, audit
    try:
        lifecycle.run(config, root)
    except BaseException as error:
        if not (root / "FAILURE.json").exists():
            base.write(root / "FAILURE.json", {"at_utc": base.now(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
