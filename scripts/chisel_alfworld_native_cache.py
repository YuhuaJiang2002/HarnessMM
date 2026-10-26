"""Exact native-episode reuse for a separately identified optimizer continuation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

from scripts import run_chisel_interactive_shared226 as n


def contract(config, binding, source_root):
    from scripts import run_alfworld_modular_rsi as core
    package_text = subprocess.check_output([config["actor_python"], "-m", "pip", "freeze"], text=True)
    native_root = Path(config["skillopt_root"])
    return {
        "model": {"id": config["model"]["served_name"], "revision": config["model"]["revision"]},
        "checkpoint": binding["checkpoint"], "physical_uuid": config["resource"]["gpu_uuid"],
        "hostname": config["authorized_hostname"], "episode_seed": 9401,
        "episode_budget": config["episode_budget"], "actor_context_tokens": 32768,
        "environment": "eval_out_of_distribution_with_exact_frozen_train_gamefiles",
        "actor_packages_sha256": hashlib.sha256(package_text.encode()).hexdigest(),
        "native_sources": {name: n.file_hash(source_root / name) for name in core.SOURCE_FILES},
        "native_vendor": {str(path.relative_to(native_root)): n.file_hash(path)
                          for path in sorted((native_root / "skillopt").rglob("*.py"))},
        "native_data_sha256": n.digest(binding["partition"]["data_hashes"]),
        "workers": config["workers"],
    }


def key(item, policy, scientific_contract):
    return n.digest({"item": item, "policy": policy, "contract": scientific_contract})


def prepare(parent, new_config, new_binding, source_root, output):
    parent_config = n.read(parent / "config.json")
    parent_binding = n.read(Path(parent_config["prepared_directory"]) / "BINDING.json")
    previous_contract = contract(parent_config, parent_binding,
                                 Path(parent_config["source_root"]))
    current_contract = contract(new_config, new_binding, source_root)
    n.require(previous_contract == current_contract,
              "Native reuse changes a payload, runtime, physical assignment or execution contract")
    allowed = {item["id"]: item for item in new_binding["partition"]["splits"]["train"]}
    store = parent / "autosaddler/native-train-only"
    events = [__import__("json").loads(line) for line in (store / "events.jsonl").read_text().splitlines()]
    outputs = set()
    for event in events:
        if event["event_type"] == "EvaluationCompleted":
            evaluation = event["payload"]["evaluation"]
            n.require(evaluation["split"] in ("train", "development"), "Unexpected official split")
            outputs.update(row["output"]["uri"] for row in evaluation["observations"] if row.get("output"))
    entries = {}
    for uri in sorted(outputs):
        path = store / uri
        n.require(path.resolve().is_relative_to(store.resolve()), "External cache artifact")
        value = n.read(path)
        n.verify_signed(value, "canonical_sha256")
        n.require(value["primary_phase"] == "optimizer", "External validation must not enter TRAIN cache")
        for row in value["rows"]:
            n.require(row["id"] in allowed, "Cached case is outside primary TRAIN")
            origin = Path(row["artifact_dir"])
            job = n.read(origin / "job.json")
            n.require(job["item"] == allowed[row["id"]] and job["endpoint_index"] == 0,
                      "Cached input or service assignment differs")
            n.validate_row(row, job["item"], "autosaddler", shared_control=False)
            indexed = key(job["item"], job["policy"], current_contract)
            entry = {
                "result": str(origin / "result.json"), "result_sha256": n.file_hash(origin / "result.json"),
                "job": str(origin / "job.json"), "job_sha256": n.file_hash(origin / "job.json"),
                "conversation_sha256": n.file_hash(origin / "conversation.json"),
                "provider_origins": {str(p): n.file_hash(p) for p in sorted(origin.rglob("accounting.json"))},
            }
            if indexed in entries:
                n.require(entries[indexed] == entry, "Ambiguous cache identity; never select by outcome")
            entries[indexed] = entry
    cache = {"scientific_contract": current_contract, "entries": entries,
             "parent": str(parent), "external_validation_included": False}
    cache["cache_sha256"] = n.digest(cache)
    n.write(output, cache)
    return cache


def load_for_request(config, binding, source_root):
    value = n.read(Path(config["native_cache_manifest"]))
    n.verify_signed(value, "cache_sha256")
    n.require(value["scientific_contract"] == contract(config, binding, source_root),
              "Current native configuration differs from the immutable cache contract")
    return value


def lookup(cache, item, policy):
    entry = cache["entries"].get(key(item, policy, cache["scientific_contract"]))
    if entry is None:
        return None
    path = Path(entry["result"])
    n.require(n.file_hash(path) == entry["result_sha256"], "Cached result changed")
    n.require(n.file_hash(Path(entry["job"])) == entry["job_sha256"], "Cached input changed")
    row = n.read(path)
    n.require(n.file_hash(Path(row["artifact_dir"]) / "conversation.json") ==
              entry["conversation_sha256"], "Cached native trace changed")
    for filename, expected in entry["provider_origins"].items():
        n.require(n.file_hash(Path(filename)) == expected, "Cached paid-attempt evidence changed")
    n.validate_row(row, item, "autosaddler", shared_control=False)
    row = deepcopy(row)
    row["execution_reuse"] = {
        "cache_sha256": cache["cache_sha256"], "original_result": str(path),
        "original_result_sha256": entry["result_sha256"],
        "physical_provider_origins": list(entry["provider_origins"]),
        "new_physical_model_calls": 0,
    }
    return row
