"""Immutable component admission and registry adoption for the common runtime."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from phiagent.harness.gaia2_202060_protocol import digest, partition_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adopt_registry(registry, *, prefix):
    if not registry:
        raise ValueError("An existing registry is required")
    for name, job in registry.items():
        if name != job["name"] or not name.startswith(prefix):
            raise ValueError("An incompatible/old-runtime job entered adoption")
        if job["method"] not in ("native", "skillopt", "autosaddler", "chisel"):
            raise ValueError("Unknown method")
        if not set(job["dependencies"]) <= registry.keys():
            raise ValueError("Missing adopted dependency")
    return deepcopy(registry)


def verify_bundle(bundle, expected_sha, *, runtime_contract_sha256):
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "BUNDLE.json"
    if sha(manifest_path) != expected_sha:
        raise ValueError("Component bundle manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("runtime_contract_sha256") != runtime_contract_sha256:
        raise ValueError("Component changes the common runtime contract")
    for name, expected in manifest["files_sha256"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha(path) != expected:
            raise ValueError("Frozen component source changed: " + name)
    for required in ("phiagent/harness/gaia2_202060_progress.py",
                     "phiagent/harness/gaia2_202060_ranges.py"):
        if required not in manifest["files_sha256"]:
            raise ValueError("Public-range implementation is missing")
    return manifest


def candidate_pair(manifest, *, bundle, bundle_sha256, runtime_contract_sha256, train_input_sha256):
    partition_gate(manifest)
    result = {}
    for variant, enabled in (("receipt_graph", True), ("remove_progress", False)):
        candidate = {
            "skill": "", "skill_sha256": hashlib.sha256(b"").hexdigest(),
            "variant": variant, "receipt_execution": enabled,
            "component_bundle": str(bundle), "component_bundle_sha256": bundle_sha256,
            "runtime_contract_sha256": runtime_contract_sha256,
            "generation_seed": 20260926, "split_seed": 20260928,
            "partition_sha256": digest(manifest), "training_input_sha256": train_input_sha256,
            "training_ids": manifest["splits"]["train"],
            "training_state": "Fresh per-episode public receipts and range coverage; no inherited state",
            "old_runtime_scores_reused": False, "old_runtime_candidates_reused": False,
            "validation_accessed": False, "test_accessed": False,
            "selection_gate": "Both new-runtime variants must complete all32x3 training "
                              "and all32x3 validation before selection",
        }
        candidate["candidate_id"] = digest(candidate)
        result[variant] = candidate
    return result


def verify_candidate(candidate, manifest, *, runtime_contract_sha256):
    partition_gate(manifest)
    if (candidate["generation_seed"] != 20260926 or candidate["split_seed"] != 20260928
            or candidate["runtime_contract_sha256"] != runtime_contract_sha256
            or candidate["partition_sha256"] != digest(manifest)
            or candidate["training_ids"] != manifest["splits"]["train"]
            or any(candidate[key] is not False for key in
                   ("old_runtime_scores_reused", "old_runtime_candidates_reused",
                    "validation_accessed", "test_accessed"))):
        raise ValueError("Candidate training/runtime boundary differs")
    if candidate["skill_sha256"] != hashlib.sha256(candidate["skill"].encode()).hexdigest():
        raise ValueError("Candidate skill changed")
    unsigned = {key: value for key, value in candidate.items() if key != "candidate_id"}
    if candidate["candidate_id"] != digest(unsigned):
        raise ValueError("Candidate identity changed")
    expected = {"receipt_graph": True, "remove_progress": False}
    if candidate["variant"] not in expected or candidate["receipt_execution"] is not expected[candidate["variant"]]:
        raise ValueError("Incorrect paired removal control")
    return verify_bundle(candidate["component_bundle"], candidate["component_bundle_sha256"],
                         runtime_contract_sha256=runtime_contract_sha256)


def coverage_gate(records, *, enabled):
    compiled = 0
    certificates = 0
    for record in records:
        if record.get("source") == "receipt_graph_execution":
            compiled += bool(record.get("checks", {}).get("public_range_compiler"))
        if record.get("source") == "public_actor_messages":
            for message in record.get("messages", []):
                text = message.get("content")
                if isinstance(text, str) and text.startswith("[Public receipt-consistent action graph]\n"):
                    checkpoint = json.loads(text.split("\n", 1)[1])
                    certificates += sum(value.get("complete") is True
                                        for value in checkpoint.get("enumeration_coverage", {}).values())
    if enabled and (compiled < 1 or certificates < 1):
        raise ValueError("The trained full component did not demonstrate executable edges and coverage")
    if not enabled and (compiled or certificates):
        raise ValueError("The removal control exposed or executed the removed progress component")
    return {"status": "PASS", "public_range_executions": compiled,
            "complete_coverage_snapshots": certificates, "quality_gain_claimed": False}
