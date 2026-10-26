"""Pure admission and accounting checks for the prospective two-arm stall study."""
from __future__ import annotations

from collections import Counter
import hashlib
import json


POLICY = "receipt_graph_added_public_stall_v1"
ARMS = {"eager": False, "require_stall": True}
PARTITION_SHA = "5549b4a72340ae1d4cc883e419dc9d947af17b7a3f69fd1d60950c17108bebec"
PARTITION_CONTENT_SHA = "e3e34e7c60c78e1b0398966a9c1a4c781badaa46ba881124481afab4f232cda5"
PRIMARY_CANDIDATE = "0eff1aed24a725cceb4870e4077563a7e209abea8391ef807b59fe19505be729"
PRIMARY_BUNDLE_SHA = "38af6ab292255ec102d9f1ea18a923426c7df8c87be535a395453149d57a87b2"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()


def scientific_contract(config):
    selected = {
        name: config.get(name) for name in (
            "model", "seed", "native_package", "dataset_revision", "filesystem",
            "judge_transport", "require_reproducible_inference", "budget",
        )
    }
    selected["resource"] = {
        name: value for name, value in config["resource"].items()
        if name not in ("preferred_gpu_indices", "expected_gpu_uuid", "model_port", "capture_port")
    }
    return selected


def validate_config(config, protocol):
    arm = config["prospective_stall"]["arm"]
    if (arm not in ARMS or config["prospective_stall"]["policy"] != POLICY
            or config["prospective_stall"]["require_stall"] is not ARMS[arm]
            or config["prospective_stall"]["protocol_sha256"] != digest(protocol)
            or config["receipt_execution"] is not True or config["frozen_skill"] != ""
            or config["method"] != "chisel" or config["seed"] != 20260926
            or config["model"]["seed"] != 20260926
            or config["model"]["revision"] != MODEL_REVISION
            or config["model"]["max_tokens"] != 4096
            or config["model"]["max_model_len"] != 131072
            or config["model"]["temperature"] != 0
            or config["model"]["enable_thinking"] is not False
            or config["judge_transport"] != "verbatim_case_preserved"
            or config["require_reproducible_inference"] is not True
            or config["resource"]["max_num_seqs"] != 1
            or config["resource"]["native_workers_per_arm"] != 1
            or config["budget"]["maximum_tokens"] is not None
            or config["budget"]["token_usage_mode"] != "unlimited"
            or config.get("continuation") or config.get("inherit_provider_ledger")
            or scientific_contract(config) != protocol["actor_judge_contract"]):
        raise ValueError("Frozen paired stall protocol differs")
    if config["stage"] not in ("train-candidate", "validation", "test"):
        raise ValueError("This prospective study has no optimizer stage")
    return arm


def decision_counts(records, *, required):
    counts = Counter()
    for row in records:
        source = row.get("source")
        if source not in ("receipt_graph_model", "receipt_graph_execution"):
            continue
        gate = row["checks"]["prospective_stall_gate"]
        if gate["policy"] != POLICY or gate["require_stall"] is not required:
            raise ValueError("A decision has a different gate policy")
        counts["decisions"] += 1
        counts["eligible_without_stall"] += gate["eligible_without_stall"] is True
        counts["latched_decisions"] += gate["latched"] is True
        if gate["rejected_only_by_stall"]:
            if (not required or gate["latched"] or not gate["eligible_without_stall"]
                    or source != "receipt_graph_model"
                    or not isinstance(gate.get("blocked_native_response"), str)):
                raise ValueError("Invalid stall-only rejection")
            counts["rejected_only_by_stall"] += 1
        if source == "receipt_graph_execution":
            if (row.get("new_model_calls") != 0 or row.get("provider_response") is not False
                    or not gate["eligible_without_stall"] or (required and not gate["latched"])):
                raise ValueError("An unsupported or unlatched action was executed")
            counts["controller_executions"] += 1
            counts["public_range_executions"] += row["checks"].get("public_range_compiler") is True
        else:
            counts["actor_responses"] += 1
    return {key: counts[key] for key in (
        "decisions", "eligible_without_stall", "latched_decisions",
        "rejected_only_by_stall", "controller_executions", "public_range_executions",
        "actor_responses",
    )}


def test_release_gate(manifest, proof, *, owner_unsealed, owner_jobs):
    from phiagent.harness.gaia2_202060_protocol import test_gate

    if not owner_unsealed:
        raise ValueError("The existing owner's test boundary remains sealed")
    if digest(manifest) != PARTITION_CONTENT_SHA:
        raise ValueError("Test partition differs")
    selections = proof["selections"]
    result = test_gate(manifest, selections)
    required = {"native", "skillopt", "autosaddler", "chisel"}
    released = {
        row["method"] for row in owner_jobs.values()
        if row.get("stage") == "test" and row.get("method") in required
    }
    if released != required:
        raise ValueError("Existing owner has not released all four frozen test populations")
    return result
