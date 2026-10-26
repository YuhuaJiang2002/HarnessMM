"""Learn training-only constraint checks and evaluate two frozen CHISEL variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.harness.aime_202060 import (  # noqa: E402
    CHISEL_VARIANTS, DIAGNOSIS_SYSTEM, MERGE_SYSTEM, diagnosis_schema, grounded_repair,
    ledger_prompt, ledger_schema, parse_structured_response, review_schema,
    strict_text_schema, usage_sum, validate_data, validate_ledger, verifier_prompt,
)
from scripts import run_aime_202060 as campaign  # noqa: E402
from scripts import run_aime_four_method as aime  # noqa: E402
from scripts import run_gaia2_jev as base  # noqa: E402


class ChiselStudy(campaign.ResplitStudy):
    def structured(self, system, text, schema, role):
        schema = strict_text_schema(schema)
        raw = base.get_json(
            self.endpoint(role) + "/chat/completions", {
                "model": self.config["model"]["served_name"], "max_tokens": 32768,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "constraint_protocol", "strict": True, "schema": schema},
                },
            }, timeout=7500,
        )
        return parse_structured_response(raw, schema), raw

    def execute(self):
        mode = self.config["mode"]
        if mode == "chisel_learn":
            self.learn()
        elif mode in ("chisel_train", "chisel_validation"):
            self.evaluate_candidates()
        else:
            raise ValueError("Unknown CHISEL development stage")

    def learn(self):
        baselines = {}
        bindings = []
        for binding in self.config["baseline_roots"]:
            location = Path(binding["root"])
            if (
                base.digest(location / "RESULT.json") != binding["result_sha256"]
                or base.digest(location / "VERIFIED.json") != binding["verified_sha256"]
            ):
                raise RuntimeError("Training baseline evidence changed")
            result = base.read(location / "RESULT.json")
            if result["manifest_sha256"] != self.config["manifest_sha256"]:
                raise RuntimeError("Training baseline uses another public partition")
            for row in base.read(location / "training/RESULT.json"):
                if row["id"] in baselines:
                    raise RuntimeError("Training baseline was charged or assigned twice")
                baselines[row["id"]] = row
            bindings.append(binding)
        if set(baselines) != self.train_ids:
            raise RuntimeError("Constraint learning requires complete new train6 baseline evidence")
        diagnoses = []
        recovered = []
        recovery = self.config.get("learning_recovery")
        prior = Path(recovery["root"]) if recovery else None
        if recovery and base.digest(prior / "PROVIDER_FINAL.json") != recovery["ledger_sha256"]:
            raise RuntimeError("Preserved failed-learner accounting changed")
        for item in self.data["train"]:
            row = baselines[item["id"]]
            text = json.dumps({
                "training_problem": item["question"], "measured_baseline_correct": row["hard"],
                "baseline_derivation": row["response"],
            }, ensure_ascii=False)
            previous = prior / "diagnoses" / (item["id"] + ".json") if prior else None
            if previous is not None and previous.exists():
                saved = base.read(previous)
                raw = saved["raw_response"]
                diagnosis = parse_structured_response(raw, strict_text_schema(diagnosis_schema()))
                if saved["id"] != item["id"] or saved["diagnosis"] != diagnosis:
                    raise RuntimeError("Preserved diagnosis identity changed")
                matches = []
                for call in base.read(prior / "PROVIDER_FINAL.json")["calls"]:
                    folder = prior / "provider" / f"call-{call['sequence']:06d}"
                    original = base.read(folder / "response.json")
                    if original["id"] != raw["id"]:
                        continue
                    request = base.read(folder / "request.json")
                    if (
                        original != raw or request["role"] != "chisel_optimizer"
                        or request["json"]["model"] != self.config["model"]["served_name"]
                        or request["json"]["max_tokens"] != 32768
                        or request["json"]["seed"] != self.config["seed"]
                        or request["json"]["temperature"] != 0
                        or request["json"]["messages"] != [
                            {"role": "system", "content": DIAGNOSIS_SYSTEM},
                            {"role": "user", "content": text},
                        ]
                    ):
                        raise RuntimeError("Recovered diagnosis is not the same training-only request")
                    matches.append({
                        "id": item["id"], "source": str(previous), "source_sha256": base.digest(previous),
                        "provider_sequence": call["sequence"], "response_id": raw["id"],
                        "new_model_calls": 0,
                    })
                if len(matches) != 1:
                    raise RuntimeError("Ambiguous recovered diagnosis provider binding")
                recovered.extend(matches)
                base.write(self.root / "RECOVERED_DIAGNOSES.json", recovered)
            else:
                diagnosis, raw = self.structured(
                    DIAGNOSIS_SYSTEM, text, diagnosis_schema(), "chisel_optimizer",
                )
            value = {"id": item["id"], "diagnosis": diagnosis, "raw_response": raw}
            base.write(self.root / "diagnoses" / (item["id"] + ".json"), value)
            diagnoses.append({"id": item["id"], **diagnosis})
            base.write(self.root / "PROGRESS.json", {
                "at_utc": base.now(), "phase": "TRAIN_ONLY_CONSTRAINT_DIAGNOSIS",
                "completed_training_diagnoses": len(diagnoses), "population": 6,
            })
        ledger, raw = self.structured(
            MERGE_SYSTEM, json.dumps(diagnoses, ensure_ascii=False),
            ledger_schema(self.manifest["splits"]["train"]), "chisel_optimizer",
        )
        validate_ledger(ledger, self.manifest["splits"]["train"])
        base.write(self.root / "LEDGER_PROVIDER_RESPONSE.json", raw)
        base.write(self.root / "CONSTRAINT_LEDGER.json", ledger)
        library = {
            "schema": "aime-training-constraint-library/v1",
            "manifest_sha256": self.config["manifest_sha256"],
            "training_ids": self.manifest["splits"]["train"],
            "training_baseline_bindings": bindings,
            "ledger": ledger, "draft_system": ledger_prompt(ledger),
            "verifier_system": verifier_prompt(ledger),
            "verifier_schema": strict_text_schema(review_schema()),
            "native_structured_backend": "xgrammar",
            "variants": list(CHISEL_VARIANTS),
            "repair_gate_source_sha256": base.digest(ROOT / "phiagent/harness/aime_202060.py"),
            "validation_used_to_generate_rules": False, "test_access": False,
            "selection_rule": "Full validation6 native accuracy, then actual tokens, then candidate ID",
            "new_task_capabilities": [], "max_output_tokens_every_call": 32768,
        }
        base.write(self.root / "POLICY_LIBRARY.json", library)
        base.write(self.root / "RESULT.json", {
            "status": "COMPLETE_PENDING_AUDIT", "mode": "chisel_learn",
            "manifest_sha256": self.config["manifest_sha256"],
            "training_ids": self.manifest["splits"]["train"], "diagnosed_tasks": 6,
            "library_sha256": base.digest(self.root / "POLICY_LIBRARY.json"),
            "candidates": list(CHISEL_VARIANTS), "validation_access": False, "test_access": False,
            "paper_admitted": False,
            "recovered_diagnoses": len(recovered),
            "historical_cost_bindings": self.config.get("historical_cost_bindings", []),
        })

    def evaluate_candidates(self):
        binding = self.config["library"]
        path = Path(binding["path"])
        if base.digest(path) != binding["sha256"]:
            raise RuntimeError("Training-derived candidate library changed")
        library = base.read(path)
        if (
            library["manifest_sha256"] != self.config["manifest_sha256"]
            or library["training_ids"] != self.manifest["splits"]["train"]
            or library["repair_gate_source_sha256"] != base.digest(ROOT / "phiagent/harness/aime_202060.py")
        ):
            raise RuntimeError("Candidate training provenance or reference-blind gate changed")
        validate_ledger(library["ledger"], self.manifest["splits"]["train"])
        split = "train" if self.config["mode"] == "chisel_train" else "development"
        assigned = self.config["assigned_ids"]
        methods = {variant: [] for variant in CHISEL_VARIANTS}
        for item in (row for row in self.data[split] if row["id"] in assigned):
            self.infer(item, library["draft_system"], "chisel_draft")
            draft = base.read(self.root / self.records[-1])
            review, raw = self.structured(
                library["verifier_system"],
                "ORIGINAL PROBLEM:\n" + item["question"] + "\n\nPROPOSED DERIVATION:\n" + draft["response"],
                library["verifier_schema"], "chisel_verifier_" + split,
            )
            response, decision = grounded_repair(item["question"], draft["response"], review)
            hard, answer = aime.score_response(self.module, response, item["answer"])
            row = {
                "id": item["id"], "response": response, "hard": hard, "predicted_answer": answer,
                "decision": decision, "review": review, "review_raw_response": raw,
                "draft_raw_response_id": draft["raw_response"]["id"],
                "draft_response": draft["response"],
                "draft_usage": {key: draft["usage"][key] for key in base.USAGE_FIELDS},
                "review_usage": base.native_usage(raw, self.config["model"]["served_name"], 131072),
            }
            row["usage"] = usage_sum([row["draft_usage"], row["review_usage"]])
            methods["constraint_ledger"].append(draft)
            methods["constraint_verified"].append(row)
            base.write(self.root / "private_methods" / item["id"] / "RESULT.json", {
                "constraint_ledger": draft, "constraint_verified": row,
            })
            base.write(self.root / "PROGRESS.json", {
                "at_utc": base.now(), "phase": self.config["mode"],
                "completed": len(methods["constraint_ledger"]), "assigned": len(assigned),
                "last_id": item["id"], "reference_used_by_repair_gate": False,
            })
        if [row["id"] for row in methods["constraint_ledger"]] != assigned:
            raise RuntimeError("Incomplete frozen candidate shard")
        scores = []
        for name, rows in methods.items():
            costs = usage_sum([row["usage"] for row in rows])
            scores.append({
                "candidate_id": name, "tasks": len(rows), "correct": sum(row["hard"] for row in rows),
                "total_tokens": costs["total_tokens"], "usage": costs, "ids": assigned,
            })
        base.write(self.root / "SCORES.json", scores)
        base.write(self.root / "RESULT.json", {
            "status": "COMPLETE_PENDING_AUDIT", "mode": self.config["mode"],
            "manifest_sha256": self.config["manifest_sha256"], "library_sha256": binding["sha256"],
            "training_ids": self.manifest["splits"]["train"], "scores": scores,
            "test_access": False, "winner_selected": False, "paper_admitted": False,
            "shared_draft_cost": "One physical call; attributed once to each logical candidate",
        })


def audit(config, root):
    ledger = base.read(root / "PROVIDER_FINAL.json")
    if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
        raise RuntimeError("Unknown CHISEL usage blocks selection")
    calls = {}
    for call in ledger["calls"]:
        directory = root / "provider" / f"call-{call['sequence']:06d}"
        raw = base.read(directory / "response.json")
        usage = base.native_usage(raw, config["model"]["served_name"], 131072)
        request = base.read(directory / "request.json")
        body = request["json"]
        if (
            call["status"] != "SETTLED" or usage != call["usage"] or raw["id"] in calls
            or body["max_tokens"] != 32768 or body["temperature"] != 0
            or body["seed"] != 20260926 or body.get("tools")
        ):
            raise RuntimeError("CHISEL native identity, full capacity or cost changed")
        calls[raw["id"]] = raw
    total = usage_sum([row["usage"] for row in ledger["calls"]])
    if total["total_tokens"] != ledger["known_tokens"]:
        raise RuntimeError("CHISEL total usage differs")
    data = base.read(root / "DATA.json")
    manifest = base.read(root / "SPLIT_MANIFEST.json")
    validate_data(data, manifest)
    result = base.read(root / "RESULT.json")
    rescored = 0
    if config["mode"] == "chisel_learn":
        recovered = base.read(root / "RECOVERED_DIAGNOSES.json") if (root / "RECOVERED_DIAGNOSES.json").exists() else []
        if len(calls) + len(recovered) != 7 or len({row["id"] for row in recovered}) != len(recovered):
            raise RuntimeError("Expected six unique training diagnoses and one ledger synthesis")
        prior_calls = {}
        for binding in config.get("historical_cost_bindings", []):
            ledger_path = Path(binding["ledger_path"])
            if base.digest(ledger_path) != binding["sha256"]:
                raise RuntimeError("Failed learner cost binding changed")
            old = base.read(ledger_path)
            if old["unknown_usage_calls"] or old["inflight_reserved_tokens"]:
                raise RuntimeError("Failed learner has unresolved costs")
            total_prior = 0
            for entry in old["calls"]:
                raw = base.read(ledger_path.parent / "provider" / f"call-{entry['sequence']:06d}" / "response.json")
                native = base.native_usage(raw, config["model"]["served_name"], 131072)
                if native != entry["usage"] or entry["status"] != "SETTLED":
                    raise RuntimeError("Historical native usage differs")
                total_prior += native["total_tokens"]
                prior_calls[raw["id"]] = raw
            if total_prior != old["known_tokens"]:
                raise RuntimeError("Historical learner cost does not reconcile")
        diagnosis_ids = []
        for item in data["train"]:
            row = base.read(root / "diagnoses" / (item["id"] + ".json"))
            raw = row["raw_response"]
            if (calls | prior_calls).get(raw["id"]) != raw:
                raise RuntimeError("Diagnosis is not bound to a paid native response")
            if parse_structured_response(raw, strict_text_schema(diagnosis_schema())) != row["diagnosis"]:
                raise RuntimeError("Diagnosis schema replay differs")
            diagnosis_ids.append(row["id"])
        if diagnosis_ids != manifest["splits"]["train"]:
            raise RuntimeError("Incomplete canonical training evidence")
        library = base.read(root / "POLICY_LIBRARY.json")
        validate_ledger(library["ledger"], manifest["splits"]["train"])
        generated = base.read(root / "LEDGER_PROVIDER_RESPONSE.json")
        if calls.get(generated["id"]) != generated:
            raise RuntimeError("Rule library lacks a new native provider binding")
        if parse_structured_response(
            generated, strict_text_schema(ledger_schema(manifest["splits"]["train"])),
        ) != library["ledger"]:
            raise RuntimeError("Native rule-library schema replay differs")
        if base.digest(root / "POLICY_LIBRARY.json") != result["library_sha256"]:
            raise RuntimeError("Learned constraint library changed")
    else:
        items = {item["id"]: item for key in ("train", "development") for item in data[key]}
        module = aime.official(config)
        method_scores = {name: [] for name in CHISEL_VARIANTS}
        for item_id in config["assigned_ids"]:
            pair = base.read(root / "private_methods" / item_id / "RESULT.json")
            draft, verified = pair["constraint_ledger"], pair["constraint_verified"]
            if calls.get(draft["raw_response"]["id"]) != draft["raw_response"]:
                raise RuntimeError("Draft response has no exact paid provider binding")
            if calls.get(verified["review_raw_response"]["id"]) != verified["review_raw_response"]:
                raise RuntimeError("Verifier response has no exact paid provider binding")
            parsed = parse_structured_response(
                verified["review_raw_response"], strict_text_schema(review_schema()),
            )
            response, decision = grounded_repair(items[item_id]["question"], draft["response"], parsed)
            if response != verified["response"] or decision != verified["decision"]:
                raise RuntimeError("Reference-blind selection was not replayable")
            for name, row in pair.items():
                hard, answer = aime.score_response(module, row["response"], items[item_id]["answer"])
                if hard != row["hard"] or answer != row["predicted_answer"]:
                    raise RuntimeError("Native official correctness differs")
                rescored += 1
                method_scores[name].append(row)
        if len(calls) != 2 * len(config["assigned_ids"]):
            raise RuntimeError("Expected one draft plus one verifier per assigned problem")
        for summary in result["scores"]:
            rows = method_scores[summary["candidate_id"]]
            if (
                sum(row["hard"] for row in rows) != summary["correct"]
                or usage_sum([row["usage"] for row in rows]) != summary["usage"]
            ):
                raise RuntimeError("Candidate score or logical execution cost differs")
    for name, digest in base.read(root / "SOURCE_FREEZE.json")["files_sha256"].items():
        if base.digest(ROOT / name) != digest:
            raise RuntimeError("Frozen source drift")
    result.update(
        status="COMPLETE_VERIFIED", new_usage=total, new_calls=len(calls),
        official_rescored=rescored, historical_costs_inherited=False,
    )
    base.write(root / "RESULT.json", result)
    base.write(root / "VERIFIED.json", {
        "status": "PASS", "result_sha256": base.digest(root / "RESULT.json"),
        "manifest_sha256": config["manifest_sha256"], "new_calls": len(calls),
        "unknown_usage_calls": 0, "official_rescored": rescored, "test_access": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = base.read(args.config)
    root = aime.remote(config)
    try:
        campaign.run(config, root, study_type=ChiselStudy, audit_function=audit)
    except BaseException as error:
        if not (root / "FAILURE.json").exists():
            base.write(root / "FAILURE.json", {
                "at_utc": base.now(), "error": str(error), "traceback": traceback.format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
