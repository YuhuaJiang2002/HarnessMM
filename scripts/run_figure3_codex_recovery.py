#!/usr/bin/env python3
"""Reacquire complete native Codex TRAIN, freeze an instruction, then validation/TEST.

No historical success is reused after the storage incident. A new slow proposal
receives only the complete capable TRAIN parent's public actor traces. The four
instruction/context arms keep the official CLI, scorer, task limits and common
brief. This component study does not attribute a CoT/Jev core improvement.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.harness.figure3_repetitions import test_seeds  # noqa: E402

SCHEMA = "figure3-codex-native-reacquisition/v1"
KEYS = ("input_tokens", "output_tokens", "total_tokens")
ARMS = ("native", "native_compaction", "instruction", "instruction_compaction")
TRAIN_METHODS = {
    "native": "codex",
    "native_compaction": "codex_native_compaction",
    "instruction": "chisel_codex",
    "instruction_compaction": "chisel_codex_native_compaction",
}
PARTITION_SHA = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
PROMPT_SHA = "f2755a7b715e692b6fe794cdf7b316dfb58b301759dade0bf3f67c2ae02e57e5"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".writing")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def valid_usage(value):
    return (
        isinstance(value, dict)
        and all(type(value.get(k)) is int and value[k] >= 0 for k in KEYS)
        and value["input_tokens"] + value["output_tokens"] == value["total_tokens"]
    )


def total_usage(rows):
    if any(not valid_usage(row["usage"]) for row in rows):
        raise ValueError("Missing provider accounting cannot become zero")
    return {key: sum(row["usage"][key] for row in rows) for key in KEYS}


def choose_validation(summaries):
    if set(summaries) != set(ARMS) or any(
        row.get("complete_native_scores_and_cost") is not True
        or row.get("independent_tasks") != 18 or row.get("repetitions") != 1
        for row in summaries.values()
    ):
        raise ValueError("All four complete validation18 populations are required")
    candidates = ("instruction", "instruction_compaction")
    selected = min(candidates, key=lambda arm: (
        -summaries[arm]["mean_native_score"],
        summaries[arm]["actor_usage"]["total_tokens"], arm,
    ))
    controls = ("native", "native_compaction")
    eligible = [
        arm for arm in candidates
        if all(
            summaries[arm]["mean_native_score"] > summaries[control]["mean_native_score"]
            and summaries[arm]["actor_usage"]["total_tokens"]
            < summaries[control]["actor_usage"]["total_tokens"]
            for control in controls
        )
    ]
    return {
        "status": "FROZEN_FOR_TEST", "selected_arm": selected, "eligible_arms": eligible,
        "selection_rule": "Complete validation: highest native quality, lowest complete actor "
                          "tokens, lexical tie break; negative selections also receive full TEST",
        "positive_validation_gate": selected in eligible,
        "training_instruction_modified": False,
        "validation_feedback_to_optimizer": False, "test_outcomes_used": False,
    }


def plan(run, config, partition, phase, arms):
    seeds = test_seeds(config, range(20260928, 20260933)) if phase == "test" else [20260928]
    retained_path = run / "RETAINED_TRAIN_SLOTS.json"
    retained = read(retained_path)["slots"] if retained_path.exists() else {}
    validation_path = run / "RETAINED_VALIDATION_SLOTS.json"
    validation_retained = read(validation_path)["slots"] if validation_path.exists() else {}
    if any(not slot.startswith("validation/") for slot in validation_retained):
        raise ValueError("Validation retention cannot contain TRAIN or TEST slots")
    jobs = []
    for seed in seeds:
        for index, task in enumerate(partition["splits"][phase]):
            order = list(arms[index % len(arms):]) + list(arms[:index % len(arms)])
            for arm in order:
                identity = [str(run), phase, arm, seed, task]
                job = {
                    "phase": phase, "arm": arm, "seed": seed, "task_id": task,
                    "task_path": str(Path(config["task_root"]) / task),
                    "trial_id": "fc-" + canonical(identity)[:32],
                    "directory": str(run / "trials" / phase / arm / str(seed) / task),
                }
                slot = f"{phase}/{arm}/{seed}/{task}"
                if slot in retained:
                    if phase != "train" or (arm != "native" and not config.get("retain_all_matched_train")):
                        raise ValueError("Only declared source-bound TRAIN actors can be retained")
                    job["trial_id"] = retained[slot]["original_trial_id"]
                if slot in validation_retained:
                    if phase != "validation" or not config.get("retain_matched_validation"):
                        raise ValueError("Only declared source-bound validation actors can be retained")
                    job["trial_id"] = validation_retained[slot]["original_trial_id"]
                jobs.append(job)
    return jobs


def modules():
    from phiagent.harness import industrial_coding_confirmation as protocol
    from scripts import run_industrial_capability_baselines as native
    from scripts import run_codex_terminal as trial
    from scripts import serve_codex_deepseek as meter
    return protocol, native, trial, meter


def require_remote():
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("All benchmark/model/scorer execution is remote-only on A800-4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU/API campaign requires CUDA_VISIBLE_DEVICES=''")


def check_sources(run):
    for name, checksum in read(run / "SOURCE_SHA256.json").items():
        if sha(run / "source" / name) != checksum:
            raise ValueError("Frozen source changed: " + name)
    for name, checksum in read(run / "NATIVE_RUNTIME.json")["files"].items():
        if sha(name) != checksum:
            raise ValueError("Official native runtime changed: " + name)
    for name, checksum in read(run / "FROZEN_INPUTS.json").items():
        if sha(run / name) != checksum:
            raise ValueError("Frozen acquisition input changed: " + name)
    freeze = run / "FROZEN_IDENTITY.json"
    if freeze.exists():
        for name, checksum in read(freeze).items():
            if sha(run / name) != checksum:
                raise ValueError("Frozen method changed: " + name)


def validate_actor_binding(record, job, expected_method, expected_residual, expected_threshold):
    if (
        any(record.get(key) != job[key] for key in ("task_id", "seed", "phase", "trial_id"))
        or record["method"] != expected_method
        or record["partition_sha256"] != PARTITION_SHA
        or record["shared_prompt"]["sha256"] != PROMPT_SHA
        or record["residual_sha256"] != expected_residual
        or record["context_control"]["active_context_trigger_tokens"] != expected_threshold
        or record["model"] != "deepseek-flash"
    ):
        raise ValueError("Native actor identity or source-frozen policy differs")


def install_retained_train(run, predecessor):
    protocol, _, trial, _ = modules()
    previous = Path(predecessor["run"])
    check_sources(previous)
    if read(previous / "STATE.json").get("active"):
        raise ValueError("The predecessor still declares active native actors")
    partition = read(run / "SPLIT_MANIFEST.json")
    originals = read(previous / "TRAIN_NATIVE_ROWS.json")
    slots = {}
    for original in originals:
        if (original["phase"] != "train" or original["arm"] != "native"
                or original["seed"] != 20260928
                or original["task_id"] not in partition["splits"]["train"]):
            raise ValueError("A predecessor contains an undeclared native TRAIN identity")
        attempt = Path(original["directory"])
        actor_path = attempt / "native-run/RUN.json"
        actor = read(actor_path)
        validate_actor_binding(
            actor, original, TRAIN_METHODS["native"], hashlib.sha256(b"").hexdigest(),
            protocol.ARMS["native"]["context_compaction_threshold"],
        )
        if (original["usage_complete"] is not True
                or original["usage"] != actor["provider_usage"]["known_usage"]):
            raise ValueError("Retained native actor usage is incomplete or changed")
        row = deepcopy(original)
        proof = {
            "original_outcome": {"path": str(attempt / "OUTCOME.json"),
                                 "sha256": sha(attempt / "OUTCOME.json")},
            "original_actor": {"path": str(actor_path), "sha256": sha(actor_path)},
            "original_source": {"path": str(previous / "SOURCE_SHA256.json"),
                                "sha256": sha(previous / "SOURCE_SHA256.json")},
            "original_trial_id": original["trial_id"], "new_actor_calls": 0,
            "new_PID_fabricated": False,
        }
        inherited_recovery = original.get("retention_proof", {}).get("verifier_recovery")
        if original["status"] != "SCORED" or inherited_recovery is not None:
            entry = predecessor["verifier_recoveries"].get(original["task_id"]) or inherited_recovery
            if not entry or sha(entry["path"]) != entry["sha256"]:
                raise ValueError("Every paid predecessor failure needs its exact verifier recovery")
            recovered = read(entry["path"])
            binding = recovered["source_binding"]
            if (
                recovered["status"] != "NATIVE_VERIFIER_RECOVERED"
                or recovered["additional_actor_calls"] != 0
                or recovered["original_actor_usage"] != original["usage"]
                or binding["original_actor"] != proof["original_actor"]
                or binding["original_outcome"] != proof["original_outcome"]
                or recovered["native_result_sha256"] != sha(recovered["native_result"])
            ):
                raise ValueError("The recovered native verdict is not bound to the retained paid actor")
            native_path = Path(recovered["native_result"])
            verdict = trial.executed_native_verdict(read(native_path), native_path.parent / "native")
            if verdict["score"] != recovered["score"]:
                raise ValueError("Recovered native reward differs from the executed verifier")
            row.update(
                status="SCORED", score=verdict["score"], executed_native_tests_verified=True,
                native_verifier=verdict,
                actor_receipt=proof["original_actor"],
                native_receipt={"path": str(native_path), "sha256": sha(native_path)},
            )
            row.pop("error", None)
            proof["verifier_recovery"] = entry
        else:
            native_path = Path(original["native_receipt"]["path"])
            verdict = trial.executed_native_verdict(
                read(native_path), attempt / "native-run/native",
            )
            if (actor["status"] != "SCORED" or actor["score"] != verdict["score"]
                    or original["score"] != verdict["score"]):
                raise ValueError("A retained original score differs from its actual native verifier")
        verify_retained_row(row)
        row["retention_proof"] = proof
        slot = f"train/native/{original['seed']}/{original['task_id']}"
        if slot in slots:
            raise ValueError("A paid native TRAIN actor cannot be retained twice")
        slots[slot] = {"original_trial_id": original["trial_id"], "retention_proof": proof}
        save(run / "trials/train/native" / str(original["seed"])
             / original["task_id"] / "OUTCOME.json", row)
    save(run / "RETAINED_TRAIN_SLOTS.json", {
        "predecessor": str(previous), "slots": slots, "new_actor_calls": 0,
        "historical_failure_artifacts_modified": False,
    })
    inputs = read(run / "FROZEN_INPUTS.json")
    inputs["RETAINED_TRAIN_SLOTS.json"] = sha(run / "RETAINED_TRAIN_SLOTS.json")
    save(run / "FROZEN_INPUTS.json", inputs)


def recovered_train_row(row, entry, *, phase="train"):
    if phase not in ("train", "validation"):
        raise ValueError("Only source-bound TRAIN/validation scorer overlays are allowed")
    """Overlay a same-actor verifier receipt without changing the failed record."""
    _, _, trial, _ = modules()
    if row.get("phase") != phase or row.get("status") != "BLOCKED":
        raise ValueError("Verifier retention overlays require an unscored TRAIN actor")
    original = Path(row["directory"])
    actor = original / "native-run/RUN.json"
    outcome = original / "OUTCOME.json"
    if sha(entry["path"]) != entry["sha256"]:
        raise ValueError("Retained verifier recovery changed")
    receipt = read(entry["path"])
    binding = receipt.get("source_binding", {})
    result = Path(receipt["native_result"])
    if (
        receipt.get("status") != "NATIVE_VERIFIER_RECOVERED"
        or receipt.get("additional_actor_calls") != 0
        or receipt.get("phase") != phase
        or receipt.get("task_id") != row["task_id"]
        or receipt.get("original_actor_usage") != row["usage"]
        or row.get("usage_complete") is not True or not valid_usage(row["usage"])
        or binding.get("original_actor") != {"path": str(actor), "sha256": sha(actor)}
        or binding.get("original_outcome") != {"path": str(outcome), "sha256": sha(outcome)}
        or receipt.get("callback") != str(original / "native-run")
        or receipt.get("original_native_result_sha256") != sha(original / "native-run/native_result.json")
        or receipt.get("native_result_sha256") != sha(result)
    ):
        raise ValueError("Verifier retention must bind the identical paid TRAIN actor and cost")
    verdict = trial.executed_native_verdict(read(result), result.parent / "native")
    if verdict["score"] != receipt["score"]:
        raise ValueError("Recovered TRAIN reward differs from executed native tests")
    value = deepcopy(row)
    value.update(status="SCORED", score=verdict["score"], executed_native_tests_verified=True,
                 actor_receipt={"path": str(actor), "sha256": sha(actor)},
                 native_receipt={"path": str(result), "sha256": sha(result)},
                 native_verifier=verdict,
                 retention_proof={"verifier_recovery": entry,
                                  "original_outcome": binding["original_outcome"],
                                  "new_actor_calls": 0})
    value.pop("error", None)
    verify_retained_row(value)
    return value


def install_retained_matched_train(run, previous, verifier_recoveries=None, *, allow_validation_handoff=False):
    """Adopt settled paid TRAIN slots after a drained repetition-policy handoff."""
    protocol, native, trial, _ = modules()
    previous = Path(previous)
    check_sources(previous)
    state = read(previous / "STATE.json")
    if (state.get("phase") not in (("train", "validation") if allow_validation_handoff else ("train",))
            or state.get("test_opened") or state.get("active")):
        raise ValueError("Drain the TRAIN owner before the prospective single-pass handoff")
    if native.process_live(read(previous / "OWNER.json")):
        raise ValueError("A healthy TRAIN owner must release its controller before adoption")
    instruction = read(previous / "FROZEN_INSTRUCTION.json")
    if sha(previous / "INSTRUCTION.txt") != instruction["applied_residual_sha256"]:
        raise ValueError("The TRAIN-derived instruction changed")
    partition = read(run / "SPLIT_MANIFEST.json")
    slots, references = {}, []
    for path in sorted((previous / "trials/train").rglob("OUTCOME.json")):
        row = read(path)
        slot = f"train/{row['arm']}/{row['seed']}/{row['task_id']}"
        if row.get("status") == "BLOCKED" and slot in (verifier_recoveries or {}):
            row = recovered_train_row(row, verifier_recoveries[slot])
        if (row["phase"] != "train" or row["arm"] not in ARMS or row["seed"] != 20260928
                or row["task_id"] not in partition["splits"]["train"] or row["status"] != "SCORED"):
            raise ValueError("Resolve retained TRAIN verdicts/usage before adoption; never replay them")
        verify_retained_row(row)
        actor = read(Path(row["actor_receipt"]["path"]))
        arm = protocol.ARMS[row["arm"]]
        residual = instruction["applied_residual_sha256"] if arm["instruction"] else hashlib.sha256(b"").hexdigest()
        validate_actor_binding(actor, row, TRAIN_METHODS[row["arm"]], residual,
                               arm["context_compaction_threshold"])
        if actor["provider_usage"]["known_usage"] != row["usage"]:
            raise ValueError("Retained complete provider cost differs from the original actor")
        original = Path(row["directory"])
        receipt = Path(row["native_receipt"]["path"])
        recovery = row.get("retention_proof", {}).get("verifier_recovery")
        native_root = receipt.parent / "native" if recovery else original / "native-run/native"
        verdict = trial.executed_native_verdict(read(receipt), native_root)
        if verdict["score"] != row["score"]:
            raise ValueError("Retained TRAIN reward differs from the executed native verifier")
        slot = f"train/{row['arm']}/{row['seed']}/{row['task_id']}"
        if slot in slots:
            raise ValueError("A paid TRAIN task/method was counted twice")
        slots[slot] = {"original_trial_id": row["trial_id"], "original_outcome": {
            "path": str(path), "sha256": sha(path)}, "new_actor_calls": 0}
        references.append(slots[slot]["original_outcome"])
        save(run / "trials/train" / row["arm"] / str(row["seed"]) / row["task_id"]
             / "OUTCOME.json", row)
    for name in ("INSTRUCTION.txt", "FROZEN_INSTRUCTION.json"):
        shutil.copyfile(previous / name, run / name)
    save(run / "RETAINED_TRAIN_SLOTS.json", {"predecessor": str(previous), "slots": slots,
         "source_references": references, "new_actor_calls": 0,
         "retention_scope": "Every settled source-bound matched TRAIN slot; no TEST reused"})
    inputs = read(run / "FROZEN_INPUTS.json")
    for name in ("RETAINED_TRAIN_SLOTS.json", "INSTRUCTION.txt", "FROZEN_INSTRUCTION.json"):
        inputs[name] = sha(run / name)
    save(run / "FROZEN_INPUTS.json", inputs)


def install_retained_validation(run, previous, verifier_recoveries):
    """Retain frozen validation actors and scorers without changing the TRAIN method."""
    protocol, native, trial, _ = modules()
    previous = Path(previous)
    check_sources(previous)
    state = read(previous / "STATE.json")
    if (state.get("status") != "BLOCKED" or state.get("phase") != "validation"
            or state.get("test_opened") or state.get("active")
            or native.process_live(read(previous / "OWNER.json"))):
        raise ValueError("Drain the source-frozen validation owner before same-actor retention")
    partition = read(run / "SPLIT_MANIFEST.json")
    instruction = read(run / "FROZEN_INSTRUCTION.json")
    if sha(run / "INSTRUCTION.txt") != sha(previous / "INSTRUCTION.txt"):
        raise ValueError("Validation continuation cannot revise the TRAIN instruction")
    old_contract_sha = sha(previous / "PROTOCOL.json")
    slots, references = {}, []
    for path in sorted((previous / "trials/validation").rglob("OUTCOME.json")):
        row = read(path)
        slot = f"validation/{row['arm']}/{row['seed']}/{row['task_id']}"
        if row.get("status") == "BLOCKED" and slot in verifier_recoveries:
            row = recovered_train_row(row, verifier_recoveries[slot], phase="validation")
        if (row.get("phase") != "validation" or row.get("arm") not in ARMS
                or row.get("seed") != 20260928 or row.get("task_id") not in partition["splits"]["validation"]
                or row.get("status") != "SCORED" or slot in slots):
            raise ValueError("Retain every settled validation slot with its native verdict and complete cost")
        verify_retained_row(row)
        actor = read(row["actor_receipt"]["path"])
        arm = protocol.ARMS[row["arm"]]
        residual = instruction["applied_residual_sha256"] if arm["instruction"] else hashlib.sha256(b"").hexdigest()
        validate_actor_binding(actor, row, arm["method"], residual, arm["context_compaction_threshold"])
        if (actor["provider_usage"]["known_usage"] != row["usage"]
                or actor["confirmation"]["contract_sha256"] != old_contract_sha
                or actor["confirmation"]["arm"] != row["arm"]):
            raise ValueError("The original validation actor's cost or frozen contract changed")
        result = Path(row["native_receipt"]["path"])
        recovery = row.get("retention_proof", {}).get("verifier_recovery")
        native_root = result.parent / "native" if recovery else Path(row["directory"]) / "native-run/native"
        if trial.executed_native_verdict(read(result), native_root)["score"] != row["score"]:
            raise ValueError("Retained validation reward differs from the actual official verifier")
        slots[slot] = {"original_trial_id": row["trial_id"], "original_outcome": {
            "path": str(path), "sha256": sha(path)}, "new_actor_calls": 0}
        references.append(slots[slot]["original_outcome"])
        save(run / "trials/validation" / row["arm"] / str(row["seed"]) / row["task_id"] / "OUTCOME.json", row)
    save(run / "RETAINED_VALIDATION_SLOTS.json", {
        "predecessor": str(previous), "slots": slots, "source_references": references,
        "previous_contract": {"path": str(previous / "PROTOCOL.json"), "sha256": old_contract_sha},
        "new_actor_calls": 0, "validation_feedback_to_training": False, "test_opened": False,
    })
    inputs = read(run / "FROZEN_INPUTS.json")
    inputs["RETAINED_VALIDATION_SLOTS.json"] = sha(run / "RETAINED_VALIDATION_SLOTS.json")
    save(run / "FROZEN_INPUTS.json", inputs)


def prepare(config_path):
    require_remote()
    protocol, native, _, _ = modules()
    config = read(config_path)
    run = Path(config["run_directory"])
    parent = Path("/path/to/data/phiagent/work/industrial-coding-confirmation")
    if (
        config["schema"] != SCHEMA or config["cumulative_token_limit"] is not None
        or config["model"] != "deepseek-flash" or not run.is_relative_to(parent)
        or run == parent or ".." in run.parts or run.exists()
        or any(p.is_symlink() for p in (run, *run.parents))
    ):
        raise ValueError("Use a new, unlimited, matched native source-frozen run")
    native_config = config["native"]
    native.validate_config(native_config)
    bundle = read(ROOT / "SOURCE_BUNDLE.json")
    if "scripts/figure3_codex_native_trial.py" not in bundle["files"]:
        raise ValueError("Freeze the dispatched native Codex trial entry point before preparation")
    for name, checksum in bundle["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(ROOT / name) != checksum:
            raise ValueError("Staging source binding changed")
    partition_path = Path(native_config["partition_dir"]) / "SPLIT_MANIFEST.json"
    if sha(partition_path) != PARTITION_SHA:
        raise ValueError("Canonical original18/18/53 partition changed")
    partition = read(partition_path)
    protocol.validate_partition(partition)
    records = {row["id"]: row for row in partition["records"]}
    task_binding = []
    import tomllib
    for task_id in partition["splits"]["train"]:
        task = Path(native_config["task_root"]) / task_id
        files = {p.relative_to(task).as_posix(): sha(p)
                 for p in sorted(task.rglob("*")) if p.is_file()}
        if canonical(files) != records[task_id]["payload_sha256"]:
            raise ValueError("Restored original TRAIN source mismatch: " + task_id)
        task_config = tomllib.loads((task / "task.toml").read_text())
        image = task_config["environment"]["docker_image"]
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True,
        ).strip()
        task_binding.append({"task_id": task_id, "payload_sha256": canonical(files),
                             "image": image, "image_id": image_id})
    if not native.storage_capacity(native_config, run)["ready"]:
        raise RuntimeError("Native artifact storage/memory reserve is unavailable")
    if importlib.metadata.version("harbor") != "0.18.0":
        raise ValueError("Pinned Harbor0.18.0 is required")
    import harbor
    harbor_root = Path(harbor.__file__).parent
    runtime = {
        str(harbor_root / name): sha(harbor_root / name)
        for name in ("verifier/verifier.py", "trial/trial.py", "environments/docker/docker.py",
                     "agents/installed/codex.py", "models/task/task.py")
    }
    binary = native_config["codex"]["binary"]
    runtime[binary] = sha(binary)
    if runtime[binary] != native_config["codex"]["binary_sha256"]:
        raise ValueError("Official Codex0.142.5 binary differs")
    if runtime[str(harbor_root / "agents/installed/codex.py")] != native_config["codex"][
        "harbor_adapter_sha256"
    ]:
        raise ValueError("Official Harbor Codex adapter differs")
    run.mkdir(parents=True, mode=0o700)
    for name in bundle["files"]:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name, value in (
        ("CONFIG.json", config), ("SOURCE_SHA256.json", bundle["files"]),
        ("STAGING_PROVENANCE.json", bundle), ("HISTORY.json", config["historical_accounting"]),
        ("NATIVE_RUNTIME.json", {"files": runtime, "harbor_version": "0.18.0",
                                "codex_version": "0.142.5"}),
        ("TASK_SOURCE_BINDING.json", task_binding),
        ("RESOURCE_PRELAUNCH.json", native.resource_inventory()),
        ("NATIVE_CONFIG.json", native_config),
        ("VERIFIER_ENVIRONMENT.json", native_config["verifier_environment"]),
    ):
        save(run / name, value)
    shutil.copyfile(partition_path, run / "SPLIT_MANIFEST.json")
    save(run / "FROZEN_INPUTS.json", {
        name: sha(run / name) for name in (
            "CONFIG.json", "SOURCE_SHA256.json", "STAGING_PROVENANCE.json",
            "HISTORY.json", "NATIVE_RUNTIME.json", "SPLIT_MANIFEST.json", "TASK_SOURCE_BINDING.json",
        )
    })
    from phiagent.data_engine.provenance import capture_provenance
    save(run / "PROVENANCE.json", capture_provenance(ROOT, sys.argv, 20260928))
    save(run / "PACKAGES.json", {
        "python": sys.version,
        "installed_distributions": sorted(
            (distribution.metadata.get("Name", "unknown"), distribution.version)
            for distribution in importlib.metadata.distributions()
        ),
        "inventory_source": "importlib.metadata; no pip dependency or package installation",
    })
    if config.get("predecessor_native_train") is not None:
        install_retained_train(run, config["predecessor_native_train"])
    if config.get("predecessor_matched_train") is not None:
        if config.get("predecessor_native_train") is not None or not native_config.get("retain_all_matched_train"):
            raise ValueError("Use exactly one matched TRAIN retention manifest")
        test_seeds(native_config, range(20260928, 20260933))
        install_retained_matched_train(run, config["predecessor_matched_train"],
                                      config.get("train_verifier_recoveries"),
                                      allow_validation_handoff=config.get("predecessor_matched_validation") is not None)
    if config.get("predecessor_matched_validation") is not None:
        if (config["predecessor_matched_validation"] != config.get("predecessor_matched_train")
                or not native_config.get("retain_matched_validation")):
            raise ValueError("Validation continuation must preserve the same fully retained TRAIN parent")
        install_retained_validation(run, config["predecessor_matched_validation"],
                                    config.get("validation_verifier_recoveries", {}))
    save(run / "STATE.json", {"at": now(), "status": "PREPARED_NEW_TRAIN",
                             "test_opened": False, "cumulative_token_limit": None})
    return run


def start_provider(run):
    config = read(run / "CONFIG.json")
    service = config["provider"]
    command = [
        sys.executable, "-B", "-u", str(run / "source/scripts/serve_codex_deepseek.py"),
        "--root", str(run / "provider"), "--credential", service["credential_file"],
        "--relay-secret", service["relay_secret_file"], "--bind", "172.17.0.1",
        "--port", str(service["port"]),
    ]
    with (run / "provider.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, stdin=subprocess.DEVNULL)
    save(run / "PROVIDER_PROCESS.json", {"at": now(), "pid": process.pid, "argv": command})
    opener = build_opener(ProxyHandler({}))
    url = f"http://172.17.0.1:{service['port']}"
    for _ in range(30):
        if process.poll() is not None:
            raise RuntimeError("Dedicated native Responses metering service exited")
        try:
            health = json.loads(opener.open(url + "/health", timeout=5).read())
            if health["model"] != "deepseek-flash" or health["status"] != "READY":
                raise ValueError("Unexpected provider identity")
            if not Path(health["root"]).is_relative_to(run / "provider"):
                raise ValueError("Provider port is owned by another campaign")
            native_config = read(run / "NATIVE_CONFIG.json")
            native_config["codex"].update(provider_url=url, provider_ledger=health["root"])
            save(run / "NATIVE_CONFIG.json", native_config)
            save(run / "PROVIDER_HEALTH.json", health)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Dedicated native provider service is not responsive")


def command(run, job):
    protocol, _, _, _ = modules()
    native_config = read(run / "NATIVE_CONFIG.json")
    arm = protocol.ARMS[job["arm"]]
    actor = native_config["codex"]
    argv = [
        actor["python"], "-B", "-u", str(run / "source/scripts/figure3_codex_native_trial.py"),
        "--task", job["task_path"], "--partition", str(run / "SPLIT_MANIFEST.json"),
        "--phase", job["phase"], "--output", str(Path(job["directory"]) / "native-run"),
        "--trial-id", job["trial_id"], "--provider-url", actor["provider_url"],
        "--provider-ledger", actor["provider_ledger"], "--relay-secret", actor["relay_secret_file"],
        "--codex-binary", actor["binary"], "--method",
        TRAIN_METHODS[job["arm"]] if job["phase"] == "train" else arm["method"],
        "--context-compaction-threshold", str(arm["context_compaction_threshold"]),
        "--repetition-seed", str(job["seed"]), "--common-prompt-version",
        native_config["shared_prompt_version"], "--common-prompt-sha256", PROMPT_SHA,
        "--verifier-environment", str(run / "VERIFIER_ENVIRONMENT.json"),
        "--provider-settlement-timeout", str(actor["provider_settlement_timeout_seconds"]),
    ]
    if arm["instruction"]:
        argv += ["--residual-file", str(run / "INSTRUCTION.txt")]
    if job["phase"] != "train":
        argv += ["--confirmation-contract", str(run / "PROTOCOL.json"),
                 "--confirmation-contract-sha256", sha(run / "PROTOCOL.json"),
                 "--confirmation-arm", job["arm"]]
    if job["phase"] == "test":
        argv += ["--freeze", str(run / "TEST_RELEASE.json")]
    if job["task_id"] == "sam-cell-seg" and native_config.get("sam_verifier_recovery"):
        argv += ["--sam-verifier-recovery", native_config["sam_verifier_recovery"]]
    cache = native_config.get("native_verifier_caches", {}).get(job["task_id"])
    if cache:
        argv += ["--native-verifier-cache", cache]
    return argv


def harvest(run, job):
    protocol, _, trial, meter = modules()
    directory = Path(job["directory"])
    output = directory / "native-run"
    config = read(run / "NATIVE_CONFIG.json")
    usage = meter.usage_summary(Path(config["codex"]["provider_ledger"]), job["trial_id"])
    row = {
        **job, "status": "BLOCKED", "score": None,
        "usage": usage.get("known_usage", dict.fromkeys(KEYS, None)),
        "usage_complete": usage.get("usage_status") == "MEASURED",
        "provider_call_count": usage.get("calls"),
        "provider_receipts": usage.get("receipts", []),
        "unresolved_call_ids": usage.get("unresolved_call_ids", []),
        "executed_native_tests_verified": False, "actor_replayed": False,
    }
    try:
        record = read(output / "RUN.json")
        expected_method = TRAIN_METHODS[job["arm"]] if job["phase"] == "train" else (
            protocol.ARMS[job["arm"]]["method"]
        )
        expected_residual = sha(run / "INSTRUCTION.txt") if "instruction" in job["arm"] else (
            hashlib.sha256(b"").hexdigest()
        )
        validate_actor_binding(
            record, job, expected_method, expected_residual,
            protocol.ARMS[job["arm"]]["context_compaction_threshold"],
        )
        if record["status"] != "SCORED":
            raise ValueError(
                "Native execution or verifier is blocked: "
                + str(record.get("error", record["status"]))
            )
        if job["phase"] != "train" and (
            record["confirmation"]["contract_sha256"] != sha(run / "PROTOCOL.json")
            or record["confirmation"]["arm"] != job["arm"]
        ):
            raise ValueError("Held-out native actor contract differs")
        result = read(output / "native_result.json")
        verdict = trial.executed_native_verdict(result, output / "native")
        if (
            usage["usage_status"] != "MEASURED" or usage["provider_models"] != ["deepseek-flash"]
            or usage["known_usage"] != record["provider_usage"]["known_usage"]
            or verdict["score"] != record["score"]
        ):
            raise ValueError("Native verifier or complete provider accounting is missing")
        row.update(
            status="SCORED", score=verdict["score"], executed_native_tests_verified=True,
            native_verifier=verdict,
            native_receipt={"path": str(output / "native_result.json"),
                            "sha256": sha(output / "native_result.json")},
            actor_receipt={"path": str(output / "RUN.json"), "sha256": sha(output / "RUN.json")},
        )
        protocol.verify_measured_row(row)
    except (OSError, ValueError, KeyError) as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    save(directory / "OUTCOME.json", row)
    return row


def verify_retained_row(row):
    if row["status"] != "SCORED":
        return
    if not valid_usage(row.get("usage")) or row.get("usage_complete") is not True:
        raise ValueError("Retained actor accounting is incomplete")
    for key in ("actor_receipt", "native_receipt"):
        receipt = row[key]
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Retained native evidence changed or disappeared")


def run_stage(run, phase, arms):
    protocol, native, _, _ = modules()
    partition = read(run / "SPLIT_MANIFEST.json")
    config = read(run / "NATIVE_CONFIG.json")
    jobs = plan(run, config, partition, phase, arms)
    processes = {}
    last_resource = 0.0
    while True:
        check_sources(run)
        rows, active, pending = [], [], []
        for job in jobs:
            directory = Path(job["directory"])
            if (directory / "OUTCOME.json").exists():
                row = read(directory / "OUTCOME.json")
                if any(row.get(key) != job[key] for key in ("phase", "arm", "seed", "task_id", "trial_id")):
                    raise ValueError("Retained native task/repetition identity differs")
                verify_retained_row(row)
                rows.append(row)
            elif directory.exists():
                if not (directory / "PROCESS.json").exists():
                    raise RuntimeError("Ambiguous prior launch; no automatic actor replay")
                owner = read(directory / "PROCESS.json")
                process = processes.get(owner["pid"])
                live = process.poll() is None if process is not None else native.process_live(owner)
                if live:
                    active.append(job)
                else:
                    rows.append(harvest(run, job))
            else:
                pending.append(job)
        failures = [row for row in rows if row["status"] != "SCORED"]
        if time.monotonic() - last_resource >= 300:
            save(run / "RESOURCE_CURRENT.json", native.resource_inventory())
            last_resource = time.monotonic()
        label = phase.upper() + "_" + ("NATIVE" if arms == ("native",) else "ALL_ARMS")
        save(run / (label + "_ROWS.json"), rows)
        save(run / "STATE.json", {
            "schema": SCHEMA, "at": now(), "pid": os.getpid(), "phase": phase,
            "status": "DRAINING_NATIVE_FAILURE" if failures else "RUNNING_" + label,
            "active": active, "pending": len(pending), "settled": len(rows), "expected": len(jobs),
            "scored": sum(row["status"] == "SCORED" for row in rows),
            "native_failures": failures, "test_opened": phase == "test",
            "cumulative_token_limit": None, "token_usage_mode": "unlimited",
            "next_dependency": "Complete native TRAIN -> TRAIN-only frozen proposal -> "
                               "complete four-arm TRAIN -> validation72 -> TEST1060",
        })
        if failures and not active:
            raise RuntimeError("Native verdict/accounting failure; actors retained, no replay")
        if (run / "STOP").exists() and not active:
            raise RuntimeError("Explicit stop drained; no actor replay")
        if not active and not pending:
            seeds = test_seeds(config, range(20260928, 20260933)) if phase == "test" else [20260928]
            return {
                "phase": phase, "complete": True, "at": now(), "rows": rows,
                "summaries": {
                    arm: protocol.summarize_arm(
                        [row for row in rows if row["arm"] == arm],
                        partition["splits"][phase], seeds, arm,
                    ) for arm in arms
                },
            }
        if not failures and not (run / "STOP").exists():
            admitted = list(active)
            for job in pending[:config["maximum_workers"] - len(active)]:
                capacity = native.task_capacity(
                    config, run, [Path(item["task_path"]) for item in [*admitted, job]],
                )
                save(run / "NEXT_TASK_ADMISSION.json", capacity)
                if not capacity["ready"]:
                    break
                directory = Path(job["directory"])
                directory.mkdir(parents=True, exist_ok=False)
                cache = directory / "process-cache"
                cache.mkdir()
                argv = command(run, job)
                save(directory / "INTENT.json", {
                    **job, "at": now(), "argv": argv, "actor_replay_permitted": False,
                })
                with (directory / "process.log").open("xb") as log:
                    process = subprocess.Popen(
                        argv, cwd=directory, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                             "PYTHONPATH": str(run / "source"), "TMPDIR": str(cache)},
                    )
                owner = {"pid": process.pid, "start_ticks": native.process_identity(process.pid),
                         "started_at": now(), "controller_pid": os.getpid()}
                save(directory / "PROCESS.json", owner)
                processes[process.pid] = process
                admitted.append(job)
        time.sleep(15)


def propose(run, baseline):
    protocol, _, _, meter = modules()
    if baseline["summaries"]["native"]["successes"] < 9:
        raise RuntimeError("Complete native TRAIN parent misses the predeclared9/18 capability gate")
    partition = read(run / "SPLIT_MANIFEST.json")
    train = set(partition["splits"]["train"])
    observations = []
    for row in baseline["rows"]:
        if row["phase"] != "train" or row["task_id"] not in train:
            raise ValueError("Only genuine complete TRAIN evidence can enter proposals")
        output = Path(row["directory"]) / "native-run"
        traces = list(output.glob("native/**/agent/codex.txt"))
        observations.append({
            "task_id": row["task_id"], "native_score": row["score"], "actor_usage": row["usage"],
            "public_instruction": (Path(row["task_path"]) / "instruction.md").read_text(),
            "public_actor_trace_tail": traces[0].read_text(errors="replace")[-4500:] if traces else "",
        })
    payload = {
        "model": "deepseek-flash", "stream": False, "store": False,
        "reasoning": {"effort": "high"}, "max_output_tokens": 8192,
        "input": [
            {"role": "system", "content": [{
                "type": "input_text",
                "text": "Propose one reusable coding completion procedure, grounded only in "
                        "the supplied full TRAIN population. Prioritize native success, then "
                        "complete-task actor tokens. Preserve the identical completion brief, "
                        "tools, limits, scorer and task requirements. Do not inspect or mention "
                        "validation or TEST. Return only a JSON object with instruction "
                        "(nonempty, at most900characters and140words) and supporting_task_ids "
                        "(at least two distinct supplied TRAIN task IDs). No oracle/scorer "
                        "details, task-specific answers, arbitrary token ceilings or early stops.",
            }]},
            {"role": "user", "content": [{
                "type": "input_text", "text": json.dumps(observations, ensure_ascii=False),
            }]},
        ],
    }
    directory = run / "optimizer"
    directory.mkdir(exist_ok=False)
    save(directory / "REQUEST.json", payload)
    config = read(run / "NATIVE_CONFIG.json")["codex"]
    identity = "fc-opt-" + canonical([str(run), sha(directory / "REQUEST.json")])[:24]
    save(directory / "INTENT.json", {"at": now(), "train_ids": sorted(train),
                                   "optimizer_trial_id": identity,
                                   "validation_access": False, "test_access": False,
                                   "automatic_provider_replay_permitted": False})
    token = meter.trial_token(meter.read_credential(Path(config["relay_secret_file"])), identity)
    request = Request(
        config["provider_url"] + "/" + identity + "/responses",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        raw = build_opener(ProxyHandler({})).open(request, timeout=1800).read()
    finally:
        save(directory / "ACCOUNTING.json", meter.usage_summary(
            Path(config["provider_ledger"]), identity,
        ))
    (directory / "RESPONSE.json").write_bytes(raw)
    response = json.loads(raw)
    texts = [
        part["text"] for item in response.get("output", [])
        for part in item.get("content", []) if part.get("type") == "output_text"
    ]
    text = "".join(texts).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    candidate = json.loads(text)
    instruction = candidate["instruction"]
    support = candidate["supporting_task_ids"]
    usage = read(directory / "ACCOUNTING.json")
    if (
        not isinstance(instruction, str) or not instruction.strip()
        or len(instruction) > 900 or len(instruction.split()) > 140
        or not isinstance(support, list) or len(set(support)) < 2
        or len(set(support)) != len(support) or not set(support) <= train
        or usage["usage_status"] != "MEASURED" or usage["provider_models"] != ["deepseek-flash"]
    ):
        raise ValueError("TRAIN-only proposal/support/provider accounting gate failed; cost retained")
    residual = protocol.instruction_suffix(instruction)
    (run / "INSTRUCTION.txt").write_text(residual)
    save(run / "FROZEN_INSTRUCTION.json", {
        "candidate": candidate, "origin": "native_responses_optimizer_complete_train_only",
        "candidate_id": canonical(candidate), "instruction_sha256": protocol.text_sha256(instruction),
        "applied_residual_sha256": protocol.text_sha256(residual),
        "train_parent": {"path": str(run / "TRAIN_NATIVE_COMPLETE.json"),
                         "sha256": sha(run / "TRAIN_NATIVE_COMPLETE.json")},
        "request": {"path": str(directory / "REQUEST.json"), "sha256": sha(directory / "REQUEST.json")},
        "response": {"path": str(directory / "RESPONSE.json"), "sha256": sha(directory / "RESPONSE.json")},
        "optimizer_trial_id": identity, "optimizer_usage": usage,
        "historical14_of18_reused": False, "validation_or_test_feedback_used": False,
        "cot_jev_core_claim": False,
    })


def freeze_method(run):
    protocol, _, _, _ = modules()
    partition = read(run / "SPLIT_MANIFEST.json")
    instruction = read(run / "FROZEN_INSTRUCTION.json")
    config = read(run / "NATIVE_CONFIG.json")
    contract = {
        "schema": protocol.SCHEMA, "benchmark": "terminal-bench-2.0", "model": "deepseek-flash",
        "arms": protocol.ARMS, "partition": partition, "partition_sha256": PARTITION_SHA,
        "shared_prompt_sha256": PROMPT_SHA, "shared_prompt_version": config["shared_prompt_version"],
        "instruction_sha256": instruction["instruction_sha256"],
        "applied_residual_sha256": instruction["applied_residual_sha256"],
        "candidate_id": instruction["candidate_id"], "validation_seeds": [20260928],
        "test_seeds": test_seeds(config, range(20260928, 20260933)), "maximum_workers": config["maximum_workers"],
        "cumulative_token_limit": None, "token_usage_mode": "unlimited",
        "test_selection_from_outcomes": False, "optimization_from_validation": False,
        "method_identity": "TRAIN-derived industrial instruction/context study",
        "cot_jev_core_claim": False, "native_timeouts_and_resources": "Unmodified task.toml",
        "model_sampling_seed_supported": False,
        "validation_selection": "Quality first, complete cost second; complete negative TEST retained",
        "primary_test_contrast": "Validation-selected fixed candidate versus both native controls",
        "all_mechanism_arms_retained_for_test": True, "prior_exposure": partition["prior_exposure"],
        "installation_adapter": "Task-local Codex home; official run/tool loop inherited unchanged",
    }
    if config.get("test_repetition_policy") is not None:
        contract["test_repetition_policy"] = config["test_repetition_policy"]
    protocol.validate_contract(contract)
    save(run / "PROTOCOL.json", contract)
    save(run / "FROZEN_IDENTITY.json", {
        name: sha(run / name) for name in (
            "PROTOCOL.json", "INSTRUCTION.txt", "FROZEN_INSTRUCTION.json", "NATIVE_CONFIG.json",
            "SOURCE_SHA256.json", "NATIVE_RUNTIME.json", "SPLIT_MANIFEST.json",
            "VERIFIER_ENVIRONMENT.json", "FROZEN_INPUTS.json",
        )
    })


def worker(run):
    require_remote()
    protocol, native, _, _ = modules()
    if ROOT != run / "source":
        raise ValueError("Execute only the run's own frozen source capsule")
    with (run / "OWNER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(run / "OWNER.json", {"at": now(), "pid": os.getpid(), "hostname": socket.gethostname(),
                                 "start_ticks": native.process_identity(os.getpid()), "run": str(run),
                                 "selected_gpus": [], "CUDA_VISIBLE_DEVICES": ""})
        try:
            check_sources(run)
            if not (run / "PROVIDER_HEALTH.json").exists():
                start_provider(run)
            baseline = run_stage(run, "train", ("native",))
            save(run / "TRAIN_NATIVE_COMPLETE.json", baseline)
            if read(run / "CONFIG.json").get("native_train_only") is True:
                successes = baseline["summaries"]["native"]["successes"]
                save(run / "STATE.json", {
                    "schema": SCHEMA, "at": now(), "pid": os.getpid(), "phase": "train",
                    "status": "COMPLETE_CAPABLE_NATIVE_TRAIN" if successes >= 9
                    else "COMPLETE_NATIVE_TRAIN_INSUFFICIENT_CAPABILITY",
                    "scored": 18, "successes": successes, "test_opened": False,
                    "cumulative_token_limit": None, "token_usage_mode": "unlimited",
                    "next_dependency": "Use this complete source-frozen common parent for TRAIN-only "
                                       "method refinement and matched native efficiency conditions",
                })
                return
            if not (run / "FROZEN_INSTRUCTION.json").exists():
                propose(run, baseline)
            if not (run / "FROZEN_IDENTITY.json").exists():
                freeze_method(run)
            train = run_stage(run, "train", ARMS)
            save(run / "TRAIN_COMPLETE.json", train)
            validation = run_stage(run, "validation", ARMS)
            save(run / "VALIDATION_COMPLETE.json", validation)
            selection = choose_validation(validation["summaries"])
            save(run / "VALIDATION_SELECTION.json", selection)
            files = [
                {"path": str(run / name), "sha256": sha(run / name)}
                for name in ("PROTOCOL.json", "INSTRUCTION.txt", "FROZEN_INSTRUCTION.json",
                             "VALIDATION_SELECTION.json", "FROZEN_IDENTITY.json")
            ]
            files += [{"path": str(run / "source" / name), "sha256": checksum}
                      for name, checksum in read(run / "SOURCE_SHA256.json").items()]
            save(run / "TEST_RELEASE.json", {
                **selection, "at": now(), "partition_sha256": PARTITION_SHA, "files": files,
                "all_four_method_policies_unchanged": True,
            })
            test = run_stage(run, "test", ARMS)
            save(run / "TEST_COMPLETE.json", test)
            partition = read(run / "SPLIT_MANIFEST.json")
            comparison = protocol.paired_comparisons(
                {arm: [row for row in test["rows"] if row["arm"] == arm] for arm in ARMS},
                partition["splits"]["test"], read(run / "PROTOCOL.json")["test_seeds"],
                selection["selected_arm"],
            )
            instruction = read(run / "FROZEN_INSTRUCTION.json")
            comparison.update(
                native_success_unit="percentage",
                native_success_percentages={
                    arm: 100 * value["mean_native_score"]
                    for arm, value in comparison["summaries"].items()
                },
                development_accounting={
                    "new_train_actor_usage": total_usage(train["rows"]),
                    "new_optimizer_usage": instruction["optimizer_usage"],
                    "historical_setup_acquisition": read(run / "HISTORY.json"),
                    "final_test_actor_cost_excludes_development": True,
                },
                provider_model="deepseek-flash", native_codex_version="0.142.5",
                common_prompt_sha256=PROMPT_SHA, partition_sha256=PARTITION_SHA,
                method_identity="TRAIN-derived industrial instruction/context component study",
                all_failed_calls_retries_compactions_counted=True,
                cached_input_already_in_input_tokens=True,
            )
            save(run / "COMPARISON.json", comparison)
            save(run / "STATE.json", {
                "schema": SCHEMA, "at": now(), "status": "COMPLETE_PAIRED_TEST",
                "test_opened": True, "positive_result_qualified": comparison[
                    "positive_complete_comparison"
                ], "cot_jev_core_claim": False,
            })
        except BaseException as error:
            save(run / "FAILURE.json", {
                "at": now(), "type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "actor_replayed": False,
                "token_ceiling_triggered": False,
                "next_valid_work": "Recover retained verifier/provider evidence without new actor "
                                   "calls; never zero-impute or replay the frozen population",
            })
            previous = read(run / "STATE.json")
            save(run / "STATE.json", {**previous, "at": now(), "status": "BLOCKED",
                                     "failure": str(run / "FAILURE.json")})
            raise


SNAPSHOT_PROGRAM = r"""
from pathlib import Path
import json,hashlib
root=Path(RUN)
def load(name):
 p=root/name
 return json.loads(p.read_text())
def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
refs={}
for name in ('SOURCE_SHA256.json','FROZEN_INPUTS.json','FROZEN_IDENTITY.json'):
 if not (root/name).exists(): continue
 document=load(name)
 for file,checksum in document.items():
  path=root/'source'/file if name=='SOURCE_SHA256.json' else root/file
  if sha(path)!=checksum: raise ValueError('Frozen source/input changed: '+file)
for file,checksum in load('NATIVE_RUNTIME.json')['files'].items():
 if sha(Path(file))!=checksum: raise ValueError('Official native runtime changed')
out={'schema':'figure3-codex-native-observer/v1','run':RUN,'source_verified':True,
     'new_observer_actor_model_scorer_calls':0}
for name in ('STATE.json','OWNER.json','PROVIDER_PROCESS.json','PROVIDER_HEALTH.json',
             'FAILURE.json','VALIDATION_SELECTION.json','COMPARISON.json','HISTORY.json'):
 if (root/name).exists():
  out[name]=load(name)
  refs[name]={'path':str(root/name),'sha256':sha(root/name)}
owner=out.get('OWNER.json',{})
try:
 fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')',1)[1].split()
 owner['alive']=fields[0] not in ('Z','X','x') and fields[19]==owner['start_ticks']
except (OSError,KeyError): owner['alive']=False
state=out.get('STATE.json',{})
phase=state.get('phase','train')
rows=[]
for p in (root/'trials'/phase).rglob('OUTCOME.json'):
 rows.append(json.loads(p.read_text()))
out['progress_by_arm']={}
for arm in ('native','native_compaction','instruction','instruction_compaction'):
 values=[r for r in rows if r['arm']==arm]
 scored=[r for r in values if r['status']=='SCORED']
 complete=bool(values) and all(r.get('usage_complete') is True for r in values)
 known={k:sum(r.get('usage',{}).get(k) or 0 for r in values)
        for k in ('input_tokens','output_tokens','total_tokens')}
 out['progress_by_arm'][arm]={'phase':phase,'settled':len(values),'scored':len(scored),
  'successes':sum(r['score'] for r in scored),'known_usage_subtotal':known,
  'complete_usage':known if complete else None,'blocked':len(values)-len(scored)}
ledger=out.get('PROVIDER_HEALTH.json',{}).get('root')
receipts=[]
unresolved=[]
if ledger:
 for p in (Path(ledger)/'calls').glob('*/*/RECEIPT.json'):
  r=json.loads(p.read_text())
  if r.get('usage_status') != 'MEASURED':
   unresolved.append(str(p))
  if r.get('usage') and r.get('model')=='deepseek-flash':
   receipts.append({'path':str(p),'sha256':sha(p),'model':r['model'],'usage':r['usage'],
                    'response_id':r.get('response_id'),'trial_id':r.get('trial_id')})
out['paid_response_count']=len(receipts)
out['first_paid_response']=receipts[0] if receipts else None
out['paid_actor_usage_known_subtotal']={
 k:sum(r['usage'][k] for r in receipts if not r['trial_id'].startswith('fc-opt-'))
 for k in ('input_tokens','output_tokens','total_tokens')}
out['unresolved_provider_receipts']=unresolved
out['complete_campaign_actor_cost']=(
 out['paid_actor_usage_known_subtotal']
 if state.get('status')=='COMPLETE_PAIRED_TEST' and not unresolved else None)
out['source_references']=refs
print(json.dumps(out,sort_keys=True))
"""


def observe(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inbox = ROOT / "outputs/industrial-research-loop/20260930-v1/inbox"
    with (output / "OBSERVER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(output / "OBSERVER_OWNER.json", {"at": now(), "pid": os.getpid(), "argv": sys.argv,
                                            "read_only_remote": True, "native_launch_authority": False})
        count = 0
        while True:
            try:
                program = "RUN=" + repr(str(args.run_dir)) + "\n" + SNAPSHOT_PROGRAM
                response = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                     "-o", "ControlMaster=no", "-o", "ControlPath=none",
                     args.host, "python3 -"],
                    input=program, capture_output=True, text=True, check=True, timeout=90,
                )
                snapshot = json.loads(response.stdout)
                save(output / "LATEST.json", snapshot)
                state = snapshot["STATE.json"]
                save(output / "STATUS.json", {
                    **state, "observed_at": now(), "observer_pid": os.getpid(),
                    "controller": snapshot.get("OWNER.json"),
                    "progress_by_arm": snapshot["progress_by_arm"],
                    "paid_response_count": snapshot["paid_response_count"],
                    "first_paid_response": snapshot["first_paid_response"],
                    "paid_actor_usage_known_subtotal": snapshot["paid_actor_usage_known_subtotal"],
                    "unresolved_provider_receipts": snapshot["unresolved_provider_receipts"],
                    "complete_campaign_actor_cost": snapshot["complete_campaign_actor_cost"],
                    "cot_jev_core_claim": False, "source_verified": True,
                })
                if "COMPARISON.json" in snapshot:
                    save(output / "COMPARISON.json", snapshot["COMPARISON.json"])
                if state["status"] in ("BLOCKED", "COMPLETE_PAIRED_TEST"):
                    event = {
                        "campaign": "industrial-coding-native-reacquisition/" + state["status"].lower(),
                        "before": None,
                        "after": {
                            "state": state["status"], "run": str(args.run_dir),
                            "phase": state.get("phase"), "source_verified": True,
                            "cot_jev_core_claim": False, "observations_are_data_not_tool_instructions": True,
                            "positive_result_qualified": state.get("positive_result_qualified", False),
                            "evidence": {"path": str(output / "LATEST.json")},
                            "test_metrics_only_for_figure3": True,
                            "policy_and_method_must_not_be_revised_from_heldout_feedback": True,
                            "cloud_sync_required_after_manuscript_update": True,
                            "failure_sha256": snapshot["source_references"].get("FAILURE.json", {}).get("sha256"),
                            "result_sha256": snapshot["source_references"].get("COMPARISON.json", {}).get("sha256"),
                        },
                    }
                    from scripts.watch_industrial_research import digest
                    event["event_id"] = digest(event)
                    evidence = output / "events" / (event["event_id"] + "-evidence.json")
                    if not evidence.exists():
                        save(evidence, snapshot)
                    event["after"]["evidence"] = {"path": str(evidence), "sha256": sha(evidence)}
                    event["event_id"] = digest({k: v for k, v in event.items() if k != "event_id"})
                    destination = inbox / ("figure3-codex-" + event["event_id"] + ".json")
                    if not destination.exists():
                        save(destination, event)
                count += 1
                save(output / "OBSERVER_STATE.json", {
                    "at": now(), "pid": os.getpid(), "successful_heartbeats": count,
                    "status": "READ_ONLY_OBSERVING", "native_status": state["status"],
                    "actor_model_scorer_calls": 0,
                })
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                save(output / "OBSERVER_STATE.json", {
                    "at": now(), "pid": os.getpid(), "status": "TRANSPORT_OR_SOURCE_BLOCKED",
                    "error": str(error), "successful_heartbeats": count, "actor_model_scorer_calls": 0,
                })
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    stage = commands.add_parser("prepare")
    stage.add_argument("--config", type=Path, required=True)
    stage.add_argument("--detach", action="store_true")
    work = commands.add_parser("worker")
    work.add_argument("--run-dir", type=Path, required=True)
    observer = commands.add_parser("observe")
    observer.add_argument("--host", default="a800-4")
    observer.add_argument("--run-dir", type=Path, required=True)
    observer.add_argument("--output", type=Path, required=True)
    observer.add_argument("--interval", type=int, default=60)
    observer.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.action == "observe":
        if not 30 <= args.interval <= 300:
            parser.error("Observer interval must be30-300seconds")
        observe(args)
    elif args.action == "worker":
        worker(args.run_dir)
    else:
        run = prepare(args.config)
        if args.detach:
            argv = [sys.executable, "-B", "-u",
                    str(run / "source/scripts/run_figure3_codex_recovery.py"),
                    "worker", "--run-dir", str(run)]
            with (run / "controller.log").open("xb") as log:
                process = subprocess.Popen(
                    argv, cwd=run, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, "PYTHONPATH": str(run / "source"),
                                                "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
                )
            save(run / "LAUNCH.json", {"at": now(), "pid": process.pid, "argv": argv})
            print(json.dumps({"run": str(run), "controller_pid": process.pid}))


if __name__ == "__main__":
    main()
