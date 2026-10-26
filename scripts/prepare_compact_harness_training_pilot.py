#!/usr/bin/env python3
"""Prepare disjoint MRQA training groups remotely, without model or scorer calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from phiagent.learning.compact_qa_contracts import select_training_groups
from scripts.prepare_chisel_component_quality_data import (
    canonical_hash,
    file_hashes,
    iter_examples,
    question_group,
    require_remote,
    scan_public_archive,
    write_json as write_new,
)


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_hash(path: Path, expected: str) -> None:
    if file_hashes(path)["sha256"] != expected:
        raise ValueError(f"input hash mismatch: {path}")


def prior_training_groups(specifications: list[dict]) -> set[str]:
    groups: set[str] = set()
    for specification in specifications:
        path = Path(specification["selection_freeze_path"])
        require_hash(path, specification["sha256"])
        selected = read_object(path)["groups"]
        if (not isinstance(selected, list) or len(selected) != len(set(selected))
                or any(not isinstance(group, str) or not group for group in selected)):
            raise ValueError("prior training exclusions require unique nonempty question groups")
        groups.update(selected)
    return groups


def prior_exclusion_replica_bindings(
    source: Path, specifications: list[dict], *, required: bool = False,
) -> list[dict]:
    folder = source / "prior-exclusions"
    if not folder.exists() and not required:
        return []
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Cross-host preparation requires its complete private exclusion replicas")
    bindings = []
    for specification in specifications:
        expected = specification["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected
        ):
            raise ValueError("Exclusion replicas require resolved immutable SHA-256 bindings")
        path = folder / f"{expected}.json"
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Exclusion replica escapes its private source directory")
        require_hash(path, expected)
        bindings.append({
            "original_path": specification["selection_freeze_path"],
            "relative_path": str(path.relative_to(source)), "sha256": expected,
        })
    return bindings


def resolve_prior_training_bindings(specifications: list[dict]) -> list[dict]:
    resolved = []
    for specification in specifications:
        if specification.get("sha256") is not None:
            resolved.append(dict(specification))
            continue
        manifest_path = Path(specification["manifest_path"])
        require_hash(manifest_path, specification["manifest_sha256"])
        manifest = read_object(manifest_path)
        if (manifest.get("canonical_sha256") != canonical_hash({
                key: item for key, item in manifest.items() if key != "canonical_sha256"
            }) or Path(specification["selection_freeze_path"]).resolve()
                != (manifest_path.parent / "selection-freeze.json").resolve()):
            raise ValueError("prior-training selection requires its exact self-hashed parent manifest")
        expected = manifest["files_sha256"]["selection-freeze.json"]
        require_hash(Path(specification["selection_freeze_path"]), expected)
        resolved.append({**specification, "sha256": expected})
    return resolved


def protocol_prior_training_bindings(config: dict) -> tuple[list[dict], bool]:
    from scripts.run_compact_harness_training_pilot import mixture_helpers

    specifications = config["data"].get("exclude_prior_training", [])
    registered = mixture_helpers(config) is not None
    return (
        resolve_prior_training_bindings(specifications) if registered else specifications,
        registered,
    )


def prepare(config_path: Path, source_manifest: Path, origin: Path, output: Path) -> None:
    require_remote()
    output.mkdir(parents=True, exist_ok=False)
    config, source = read_object(config_path), read_object(source_manifest)
    origin_value = read_object(origin)
    source_root = Path(__file__).resolve().parents[1]
    for relative, expected in origin_value["source_sha256"].items():
        require_hash(source_root / relative, expected)
    require_hash(source_manifest, origin_value["component_data_manifest_sha256"])
    unsigned = {key: value for key, value in source.items() if key != "canonical_sha256"}
    if (
        source["canonical_sha256"] != canonical_hash(unsigned)
        or source["schema_version"] != "phiagent-chisel-component-data/v1"
        or source["dataset_id"] != "MRQA2019/SearchQA-v2"
    ):
        raise ValueError("source data manifest self-hash differs")
    if config["schema_version"] not in {
        "phiagent-compact-harness-training-pilot/v1", "phiagent-compact-harness-refinement/v3",
        "phiagent-compact-harness-cost-source/v4",
        "phiagent-delete-fidelity-compression/v5",
        "phiagent-qa-context-evidence/v1",
        "phiagent-qa-harness-mixture/v1",
        "phiagent-qa-capability-cost/v2",
        "phiagent-qa-capability-core/v3",
        "phiagent-qa-chisel-preservation/v1",
        "phiagent-qa-lossless-routing/v1",
        "phiagent-qa-lossless-format/v1",
        "phiagent-qa-profile-compression/v1",
        "phiagent-qa-profile-compression/v2",
    }:
        raise ValueError("unexpected pilot protocol")
    write_new(output / "config.json", config)
    write_new(output / "origin.json", origin_value)
    write_new(output / "execution.json", {
        "hostname": socket.gethostname(), "python": sys.version, "command": sys.argv,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "CUDA_VISIBLE_DEVICES": "", "selected_physical_gpu": None,
        "selection_reason": "CPU-only source preparation; no model or scorer use",
        "seed": config["data"]["selection_salt"],
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
    })
    (output / "gpu-observation.csv").write_text(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ], text=True, timeout=30), encoding="utf-8")
    archive = Path(source["raw_archives"]["train"]["path"])
    archive_hashes = file_hashes(archive)
    if (
        archive_hashes["sha256"] != config["data"]["train_archive_sha256"]
        or archive_hashes["md5"] != config["data"]["train_archive_publisher_md5"]
    ):
        raise ValueError("official training archive binding differs")
    freeze_path = Path(source["selection_freeze_path"])
    require_hash(freeze_path, source["selection_freeze_sha256"])
    sibling_groups = read_object(freeze_path)["selected_groups"]
    excluded = set(sibling_groups["train"]) | set(sibling_groups["calibration"])
    prior_specifications, registered_protocol = protocol_prior_training_bindings(config)
    prior_groups = prior_training_groups(prior_specifications)
    excluded.update(prior_groups)
    evaluation = source["splits"]["evaluation"]
    evaluation_public = Path(evaluation["public_path"])
    require_hash(evaluation_public, evaluation["public_sha256"])
    evaluation_groups = set()
    evaluation_count = 0
    with evaluation_public.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if set(record) != {"id", "group", "question", "context"}:
                raise ValueError("evaluation public metadata schema differs")
            evaluation_groups.add(record["group"])
            evaluation_count += 1
    if (
        evaluation_count != evaluation["task_count"]
        or canonical_hash(sorted(evaluation_groups)) != evaluation["question_groups_canonical_sha256"]
    ):
        raise ValueError("complete evaluation group-exclusion metadata differs")
    excluded.update(evaluation_groups)
    identities = scan_public_archive(
        archive, expected_count=config["data"]["official_train_task_count"]
    )
    chosen = set(select_training_groups(
        set(identities.values()), excluded,
        count=config["data"]["question_groups"], salt=config["data"]["selection_salt"],
    ))
    selected_ids = {qid for qid, group in identities.items() if group in chosen}
    if len(selected_ids) > config["data"]["maximum_tasks"]:
        raise ValueError("whole-group task count exceeds the frozen budget; do not drop tasks")
    write_new(output / "selection-freeze.json", {
        "groups": sorted(chosen), "task_ids": sorted(selected_ids),
        "excluded_groups_canonical_sha256": canonical_hash(sorted(excluded)),
        "no_component_train_calibration_or_dev_group_overlap": not bool(chosen & excluded),
        "prior_training_exclusions": config["data"].get("exclude_prior_training", []),
        "prior_training_group_count": len(prior_groups),
        "no_prior_training_group_overlap": not bool(chosen & prior_groups),
        "accepted_answers_consulted_for_selection": False,
        "selection_salt": config["data"]["selection_salt"],
        "model_or_scorer_calls_before_freeze": 0,
        **({"resolved_prior_training_exclusions": prior_specifications}
           if registered_protocol else {}),
    })
    answers, visited = {}, set()
    with (output / "public.jsonl").open("x", encoding="utf-8") as stream:
        for context, qa in iter_examples(archive):
            qid = qa["qid"]
            if qid not in selected_ids:
                continue
            if qid in visited:
                raise ValueError("selected task repeats")
            group = question_group(qa["question"])
            if group not in chosen or identities[qid] != group:
                raise ValueError("selected task/group identity changed")
            accepted = qa.get("answers")
            if not isinstance(accepted, list) or not accepted or any(
                not isinstance(answer, str) for answer in accepted
            ):
                raise ValueError("invalid official training answers")
            visited.add(qid)
            answers[qid] = accepted
            stream.write(json.dumps({
                "id": qid, "group": group, "question": qa["question"], "context": context,
            }, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    if visited != selected_ids:
        raise ValueError("whole selected cohort was not materialized")
    write_new(output / "answers.json", answers)
    os.chmod(output / "answers.json", 0o600)
    scorer_source = Path(source["scorer"]["path"])
    require_hash(scorer_source, config["scoring"]["source_sha256"])
    shutil.copy2(scorer_source, output / "mrqa_official_eval.py")
    require_hash(archive, archive_hashes["sha256"])
    manifest = {
        "schema_version": "phiagent-compact-harness-training-data/v1",
        "status": "TRAINING_PILOT_DATA_READY_NOT_A_QUALITY_RESULT",
        "hostname": socket.gethostname(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(visited), "question_group_count": len(chosen),
        "source_manifest_sha256": file_hashes(source_manifest)["sha256"],
        "config_sha256": file_hashes(config_path)["sha256"],
        "source_archive": {"path": str(archive), **archive_hashes},
        "files_sha256": {
            name: file_hashes(output / name)["sha256"]
            for name in ("public.jsonl", "answers.json", "selection-freeze.json",
                         "mrqa_official_eval.py", "config.json")
        },
        "scorer": {
            "repository": config["scoring"]["repository"],
            "revision": config["scoring"]["revision"],
            "filename": "mrqa_official_eval.py",
            "sha256": config["scoring"]["source_sha256"],
        },
        "independence": {
            "group_disjoint_from_component_training": not bool(chosen & set(sibling_groups["train"])),
            "group_disjoint_from_component_calibration": not bool(chosen & set(sibling_groups["calibration"])),
            "group_disjoint_from_full_official_development": not bool(chosen & evaluation_groups),
            "group_disjoint_from_prior_training": not bool(chosen & prior_groups),
            "historical_freshness_certified": False,
            "evaluation_public_group_metadata_read": True,
            "evaluation_answers_opened": False,
            "evaluation_responses_opened": False,
        },
        "new_model_or_scorer_calls": 0,
        "claim_boundary": "Official training-source subset only. No benchmark quality or paper claim.",
    }
    manifest["canonical_sha256"] = canonical_hash(manifest)
    write_new(output / "manifest.json", manifest)
    printed = json.dumps({
        "status": manifest["status"], "task_count": len(visited),
        "groups": len(chosen), "independence": manifest["independence"],
    }, indent=2)
    (output / "stdout.log").write_text(printed + "\n", encoding="utf-8")
    print(printed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a new output directory; no prior attempt may be reused")
    try:
        prepare(args.config, args.source_manifest, args.origin, args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        if args.output_dir.is_dir():
            write_new(args.output_dir / "FAILURE.json", {
                "error_type": type(error).__name__, "error": str(error),
                "at_utc": datetime.now(timezone.utc).isoformat(),
            })
        raise


if __name__ == "__main__":
    main()
