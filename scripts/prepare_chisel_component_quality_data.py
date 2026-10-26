#!/usr/bin/env python3
"""Prepare MRQA2019 SearchQA-v2 remotely; never generate predictions or run a scorer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import stat
import sys
import unicodedata
from typing import Any, Iterable, Iterator, Mapping
import urllib.request


SCHEMA = "phiagent-chisel-component-data/v1"
DATASET_ID = "MRQA2019/SearchQA-v2"
REVISION = "b86bbcc63352786f0ca246b76191f062dce4b52b"
REPOSITORY = "https://github.com/mrqa/MRQA-Shared-Task-2019"
SOURCE_BASE = f"https://raw.githubusercontent.com/mrqa/MRQA-Shared-Task-2019/{REVISION}"
SCORER_SHA256 = "02a7855398e7133952ee374582fcc90a8b6dcf6216e0759347f29487074ee992"
README_SHA256 = "75072b8886ce6a69c8f243923887463f57da21e937c3c0fc6d00bba5bd0a7f0e"
SALT = "chisel-component-quality-mrqa-v1"
REMOTE_HOSTS = {"harnessmm-worker-1", "harnessmm-worker-2"}
ARCHIVES = {
    "train": {
        "url": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz",
        "publisher_md5": "612245315e6e7c4d8446e5fcc3dc1086",
        "official_task_count": 117384,
    },
    "evaluation": {
        "url": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz",
        "publisher_md5": "9217ad3f6925c384702f2a4e6d520c38",
        "official_task_count": 16980,
    },
}
GROUP_NORMALIZATION = "Unicode NFKC, casefold, split/join whitespace; UTF-8 SHA256"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def file_hashes(path: Path) -> dict[str, Any]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest(), "bytes": size}


def write_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    path.chmod(mode)


def require_remote() -> None:
    if socket.gethostname() not in REMOTE_HOSTS:
        raise RuntimeError("MRQA benchmark preparation is restricted to authorized H200 servers")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only data preparation requires CUDA_VISIBLE_DEVICES=''")


def question_group(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if not normalized:
        raise ValueError("question must not be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def iter_examples(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                raise ValueError(f"invalid MRQA JSON record at line {number}") from None
            if number == 1:
                if not isinstance(row, dict) or not isinstance(row.get("header"), dict):
                    raise ValueError("official MRQA header is missing")
                if row["header"].get("dataset") != "SearchQA":
                    raise ValueError("MRQA header does not identify SearchQA")
                continue
            if not isinstance(row, dict) or not isinstance(row.get("context"), str):
                raise ValueError(f"invalid MRQA context at line {number}")
            if not isinstance(row.get("qas"), list) or not row["qas"]:
                raise ValueError(f"invalid MRQA question list at line {number}")
            for qa in row["qas"]:
                if not isinstance(qa, dict) or not isinstance(qa.get("qid"), str) or not qa["qid"]:
                    raise ValueError(f"missing original MRQA qid at line {number}")
                question_group(qa.get("question"))
                yield row["context"], qa


def scan_public_archive(path: Path, *, expected_count: int) -> dict[str, str]:
    """Inspect identity/question fields only; answers never participate in selection."""
    identities: dict[str, str] = {}
    for _context, qa in iter_examples(path):
        qid = qa["qid"]
        if qid in identities:
            raise ValueError("duplicate original MRQA qid")
        identities[qid] = question_group(qa["question"])
    if len(identities) != expected_count:
        raise ValueError(
            f"official MRQA task count mismatch: expected {expected_count}, observed {len(identities)}"
        )
    return identities


def select_question_groups(
    train_groups: Iterable[str],
    evaluation_groups: Iterable[str],
    *,
    train_limit: int = 8000,
    calibration_limit: int = 2000,
) -> dict[str, set[str]]:
    if train_limit < 0 or calibration_limit < 0:
        raise ValueError("group limits must be nonnegative")
    eligible = set(train_groups) - set(evaluation_groups)
    ordered = sorted(
        eligible,
        key=lambda group: (hashlib.sha256(f"{SALT}\0{group}".encode("utf-8")).digest(), group),
    )
    return {
        "train": set(ordered[:train_limit]),
        "calibration": set(ordered[train_limit : train_limit + calibration_limit]),
    }


def materialize_splits(
    paths: Mapping[str, Path],
    identities: Mapping[str, Mapping[str, str]],
    selections: Mapping[str, set[str]],
    output: Path,
) -> dict[str, dict[str, Any]]:
    evaluation_groups = set(identities["evaluation"].values())
    if (
        selections["train"] & selections["calibration"]
        or selections["train"] & evaluation_groups
        or selections["calibration"] & evaluation_groups
    ):
        raise ValueError("normalized-question group leakage")
    if not (selections["train"] | selections["calibration"]) <= set(identities["train"].values()):
        raise ValueError("selected group is absent from official train")
    public = output / "public"
    sealed = output / "sealed"
    public.mkdir(mode=0o755)
    sealed.mkdir(mode=0o700)
    streams = {}
    labels: dict[str, dict[str, list[str]]] = {split: {} for split in ("train", "calibration", "evaluation")}
    groups: dict[str, set[str]] = {split: set() for split in labels}
    visited: dict[str, dict[str, str]] = {"train": {}, "evaluation": {}}
    try:
        for split in labels:
            streams[split] = (public / f"{split}.jsonl").open("x", encoding="utf-8")
        for source_split in ("train", "evaluation"):
            for context, qa in iter_examples(paths[source_split]):
                qid, question = qa["qid"], qa["question"]
                group = question_group(question)
                if qid in visited[source_split]:
                    raise ValueError("duplicate qid during materialization")
                visited[source_split][qid] = group
                if source_split == "evaluation":
                    split = "evaluation"
                elif group in selections["train"]:
                    split = "train"
                elif group in selections["calibration"]:
                    split = "calibration"
                else:
                    continue
                answers = qa.get("answers")
                if not isinstance(answers, list) or not answers or any(
                    not isinstance(answer, str) for answer in answers
                ):
                    raise ValueError("MRQA accepted answers must be a nonempty string list")
                labels[split][qid] = answers
                groups[split].add(group)
                streams[split].write(
                    json.dumps(
                        {"id": qid, "question": question, "context": context, "group": group},
                        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                    ) + "\n"
                )
    finally:
        for stream in streams.values():
            stream.close()
    if visited != identities:
        raise ValueError("archive public identities changed after the selection freeze")
    metadata = {}
    for split, answers in labels.items():
        expected = {
            qid for qid, group in identities["evaluation" if split == "evaluation" else "train"].items()
            if split == "evaluation" or group in selections[split]
        }
        if set(answers) != expected:
            raise ValueError("whole-group or complete-development materialization failed")
        answers_path = sealed / f"{split}-answers.json"
        write_json(answers_path, answers, mode=0o400 if split == "evaluation" else 0o600)
        public_path = public / f"{split}.jsonl"
        metadata[split] = {
            "public_path": str(public_path.resolve()),
            "answers_path": str(answers_path.resolve()),
            "task_count": len(answers),
            "question_group_count": len(groups[split]),
            "public_sha256": file_hashes(public_path)["sha256"],
            "answers_sha256": file_hashes(answers_path)["sha256"],
            "task_ids_canonical_sha256": canonical_hash(sorted(answers)),
            "question_groups_canonical_sha256": canonical_hash(sorted(groups[split])),
            "source_split": "official in-domain dev" if split == "evaluation" else "official train",
            "official_split_complete": split == "evaluation",
            "whole_question_groups_complete": True,
            "answers_file_mode": oct(stat.S_IMODE(answers_path.stat().st_mode)),
        }
    return metadata


def task_id(row: Mapping[str, Any]) -> str | None:
    for key in ("id", "qid", "task_id", "sample_id", "key"):
        value = row.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
            return str(value)
    return None


def records(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from records(item)
    elif isinstance(value, (str, int)) and not isinstance(value, bool):
        yield {"id": str(value)}
    elif isinstance(value, dict):
        if task_id(value) is not None:
            yield value
        for key in ("items", "rows", "tasks", "data", "qas", "results", "records", "calls", "events"):
            if isinstance(value.get(key), (list, dict)):
                yield from records(value[key])


def file_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            for line in stream:
                if line.strip():
                    yield from records(json.loads(line))
        else:
            yield from records(json.load(stream))


def exposure_kind(row: Mapping[str, Any]) -> str:
    usage = row.get("usage")
    tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
    success = row.get("ok") is True or str(row.get("status", "")).lower() in {
        "ok", "success", "succeeded", "completed",
    }
    if (
        success and isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and tokens > 0
        and isinstance(row.get("response"), (str, dict, list))
        and any(row.get(key) for key in ("endpoint", "request_sha256", "response_sha256", "model"))
    ):
        return "verified_actual_call"
    if row.get("labels_accessed") is True and any(
        row.get(key) for key in ("timestamp", "opened_at", "scored_at")
    ):
        return "verified_actual_label_access"
    return "conservative_possible_exposure"


def source_role(path: Path) -> str | None:
    name = path.name.lower()
    if path.suffix not in {".json", ".jsonl"}:
        return None
    if {"public", "splits"} & set(path.parts) or name == "items.json":
        return "public_identity_manifest"
    if (
        {"responses", "collect", "shards"} & set(path.parts)
        or any(word in name for word in ("call", "ledger", "response", "result", "exposure", "matrix"))
    ):
        return "call_label_or_possible_exposure_ledger"
    return None


def audit_exposure(
    roots: Iterable[Path],
    target_groups: Mapping[str, set[str]],
    target_ids: Mapping[str, str],
    output: Path,
    *,
    max_file_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Join public question identities to call-bearing records; allocations are not calls."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, values in target_groups.items():
        for group in values:
            group_splits[group].add(split)
    inventory: list[dict[str, Any]] = []
    candidates: list[tuple[Path, str]] = []
    roots = list(roots)
    seen_paths: set[Path] = set()
    unresolved = []
    for root in roots:
        if not root.is_dir():
            unresolved.append({"path": str(root), "reason": "root unavailable"})
            continue
        walk_errors = []
        for directory, subdirectories, filenames in os.walk(
            root, followlinks=False, onerror=lambda error: walk_errors.append(str(error.filename))
        ):
            subdirectories[:] = sorted(
                name for name in subdirectories
                if name not in {
                    ".git", "__pycache__", "site-packages", "node_modules", "sealed", "raw", "labels",
                }
                and not (Path(directory) / name).is_symlink()
                and not (Path(directory) / name).resolve().is_relative_to(output.resolve())
            )
            for name in sorted(filenames):
                path = Path(directory) / name
                role = source_role(path)
                if role is None or path.is_symlink() or path in seen_paths:
                    continue
                seen_paths.add(path)
                try:
                    size = path.stat().st_size
                except OSError:
                    unresolved.append({"path": str(path), "reason": "stat failed"})
                    continue
                if size > max_file_bytes:
                    unresolved.append({"path": str(path), "bytes": size, "reason": "file size limit"})
                    continue
                candidates.append((path, role))
        unresolved.extend({"path": path, "reason": "directory unreadable"} for path in walk_errors)
    identity_groups: dict[str, set[str]] = defaultdict(set)
    identity_sources: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    identity_count = 0
    for path, role in candidates:
        if role != "public_identity_manifest":
            continue
        entry = {"path": str(path), "role": role}
        try:
            entry.update(file_hashes(path))
            count = 0
            for index, row in enumerate(file_records(path), 1):
                qid = task_id(row)
                if qid is None or not isinstance(row.get("question"), str):
                    continue
                group = question_group(row["question"])
                identity_groups[qid].add(group)
                if group in group_splits:
                    identity_sources[(qid, group)].append({
                        "path": str(path), "sha256": entry["sha256"], "record": index,
                    })
                count += 1
            identity_count += count
            entry["public_question_records"] = count
            entry["status"] = "READ_PUBLIC_IDENTITY_FIELDS"
        except (OSError, ValueError, TypeError):
            entry["status"] = "UNRESOLVED_PARSE_OR_READ"
            unresolved.append({"path": str(path), "reason": entry["status"]})
        inventory.append(entry)
    evidence_path = output / "exposure-evidence.jsonl"
    sets: dict[str, dict[str, set[str]]] = {
        kind: {split: set() for split in target_groups}
        for kind in (
            "verified_actual_call", "verified_actual_label_access", "conservative_possible_exposure",
        )
    }
    totals: Counter[str] = Counter()
    matched_ids: dict[str, set[str]] = {kind: set() for kind in sets}
    with evidence_path.open("x", encoding="utf-8") as evidence:
        def emit(value: dict[str, Any], kind: str, qid: str, group: str) -> None:
            totals[kind] += 1
            matched_ids[kind].add(qid)
            for split in group_splits[group]:
                sets[kind][split].add(group)
            evidence.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")

        for (qid, group), sources in sorted(identity_sources.items()):
            emit({
                "kind": "conservative_possible_exposure",
                "basis": "public manifest inclusion only; not evidence of an actual call",
                "historical_task_id": qid, "group": group,
                "target_splits": sorted(group_splits[group]), "public_identity_sources": sources,
            }, "conservative_possible_exposure", qid, group)
        for path, role in candidates:
            if role == "public_identity_manifest":
                continue
            entry = {"path": str(path), "role": role}
            try:
                entry.update(file_hashes(path))
                counts: Counter[str] = Counter()
                for index, row in enumerate(file_records(path), 1):
                    qid = task_id(row)
                    if qid is None:
                        continue
                    kind = exposure_kind(row)
                    counts[kind] += 1
                    mapped = set(identity_groups.get(qid, set()))
                    if isinstance(row.get("question"), str):
                        mapped.add(question_group(row["question"]))
                    if qid in target_ids:
                        mapped.add(target_ids[qid])
                    if not mapped:
                        counts["unmapped_task_records"] += 1
                    for group in sorted(mapped & group_splits.keys()):
                        resolved_kind = kind if len(mapped) == 1 else "conservative_possible_exposure"
                        public_sources = identity_sources.get((qid, group), [])
                        local_source_link = any(
                            path.is_relative_to(root)
                            and any(Path(source["path"]).is_relative_to(root) for source in public_sources)
                            for root in roots
                        )
                        direct_question_link = (
                            isinstance(row.get("question"), str)
                            and question_group(row["question"]) == group
                        )
                        if not (local_source_link or direct_question_link or target_ids.get(qid) == group):
                            resolved_kind = "conservative_possible_exposure"
                            counts["unverified_cross_root_identity_matches"] += 1
                        if len(mapped) > 1:
                            counts["ambiguous_identity_matches"] += 1
                        emit({
                            "kind": resolved_kind, "ledger_kind": kind,
                            "historical_task_id": qid, "group": group,
                            "target_splits": sorted(group_splits[group]),
                            "source": {
                                "path": str(path), "sha256": entry["sha256"], "record": index,
                                "record_canonical_sha256": canonical_hash(row),
                            },
                            "public_identity_sources": public_sources,
                            "identity_mapping": "unique" if len(mapped) == 1 else "ambiguous",
                            "within_audited_root_public_identity_link": local_source_link,
                        }, resolved_kind, qid, group)
                entry["record_counts"] = dict(counts)
                entry["status"] = "AUDITED_WITHOUT_USING_LABEL_VALUES"
                totals.update({f"all_source_{key}": count for key, count in counts.items()})
            except (OSError, ValueError, TypeError):
                entry["status"] = "UNRESOLVED_PARSE_OR_READ"
                unresolved.append({"path": str(path), "reason": entry["status"]})
            inventory.append(entry)
    inventory_path = output / "exposure-source-inventory.json"
    write_json(inventory_path, {"files": inventory, "unresolved": unresolved})
    split_counts = {}
    for split, groups in target_groups.items():
        actual = sets["verified_actual_call"][split] | sets["verified_actual_label_access"][split]
        possible = sets["conservative_possible_exposure"][split]
        split_counts[split] = {
            "target_question_groups": len(groups),
            "verified_actual_call_question_groups": len(sets["verified_actual_call"][split]),
            "verified_actual_label_access_question_groups": len(sets["verified_actual_label_access"][split]),
            "verified_actual_exposed_question_groups": len(actual),
            "conservative_possible_question_groups": len(possible),
            "possible_only_question_groups": len(possible - actual),
            "unmatched_question_groups_not_proven_fresh": len(groups - actual - possible),
        }
    receipt = {
        "schema_version": "phiagent-chisel-component-exposure/v1",
        "hostname": socket.gethostname(),
        "audited_roots": [str(root) for root in roots],
        "complete_global_coverage": False,
        "fresh_confirmation_admitted": False,
        "opened_prospectively_frozen_descriptive_study_admitted": True,
        "history_exclusions_from_evaluation": 0,
        "coverage_boundary": [
            "Only explicitly enumerated existing readable sources on this host were audited.",
            "Other hosts, deleted runs, unenumerated layouts, unavailable archives and pretraining exposure are not covered.",
            "Exact normalized-question matching only; semantic/paraphrase exposure is unverified.",
            "Public allocations and prior exposure-union declarations are possible exposure, never actual-call proof.",
            "Actual-call proof requires successful response-bearing records with positive recorded usage.",
            "Unknown, failed and unmapped ledger records are not evidence of freshness.",
        ],
        "answer_values_used_for_audit_or_selection": False,
        "legacy_public_manifests_may_contain_ignored_label_fields": True,
        "candidate_source_files": len(candidates),
        "audited_source_files": len(inventory),
        "unresolved_sources": len(unresolved),
        "public_identity_records": identity_count,
        "ambiguous_historical_task_ids": sum(len(values) > 1 for values in identity_groups.values()),
        "record_counts": dict(totals),
        "matched_historical_task_ids": {kind: len(ids) for kind, ids in matched_ids.items()},
        "splits": split_counts,
        "source_inventory_path": str(inventory_path),
        "source_inventory_sha256": file_hashes(inventory_path)["sha256"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": file_hashes(evidence_path)["sha256"],
    }
    write_json(output / "exposure-audit.json", receipt)
    return receipt


def acquire(path: Path, url: str, supplied: Path | None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if supplied is None:
        with urllib.request.urlopen(url, timeout=90) as response, path.open("xb") as destination:
            shutil.copyfileobj(response, destination, length=1024 * 1024)
        transport = "direct public HTTPS download on authorized server"
    elif supplied.resolve() == path.resolve():
        if not path.is_file():
            raise ValueError("supplied artifact does not exist")
        transport = "pre-staged unmodified public artifact; see acquisition provenance"
    else:
        with supplied.open("rb") as source, path.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        transport = "copied unmodified pre-staged public artifact"
    path.chmod(0o400)
    return {"path": str(path.resolve()), "url": url, "transport": transport, **file_hashes(path)}


def validate_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for spec in ARCHIVES.values():
        if any(
            token not in text for token in (
                spec["url"], spec["publisher_md5"], f'{spec["official_task_count"]:,}',
            )
        ):
            raise ValueError("pinned official README does not support the frozen release contract")
    if "In-Domain" not in text:
        raise ValueError("pinned official development-track contract missing")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    require_remote()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA, "dataset_id": DATASET_ID, "source_revision": REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "argv": sys.argv, "command": shlex.join([sys.executable, *sys.argv]),
        "selection_salt": SALT, "seed": "deterministic SHA256 salt; no PRNG",
        "requested_train_groups": 8000, "requested_calibration_groups": 2000,
        "question_group_normalization": GROUP_NORMALIZATION, "cuda_visible_devices": "",
        "gpu_allocation": None, "expected_official_archives": ARCHIVES,
        "exposure_roots": [str(path.resolve()) for path in args.exposure_root],
        "source_git_provenance": str(args.git_provenance.resolve()),
        "script_sha256": file_hashes(Path(__file__).resolve())["sha256"],
    }
    write_json(output / "run-config.json", config)
    try:
        if shutil.disk_usage(output).free < 5 * 1024**3:
            raise RuntimeError("insufficient free space: require at least 5 GiB for data preparation")
        if not args.git_provenance.is_dir() or not (
            args.git_provenance / "git-head.txt"
        ).is_file() or not (args.git_provenance / "git-status.txt").is_file():
            raise ValueError("frozen source Git HEAD and dirty status provenance are required")
        provenance = {
            "git_head": (args.git_provenance / "git-head.txt").read_text().strip(),
            "git_status_sha256": file_hashes(args.git_provenance / "git-status.txt")["sha256"],
            "packages": sorted(
                (
                    {"name": distribution.metadata["Name"], "version": distribution.version}
                    for distribution in importlib.metadata.distributions()
                ),
                key=lambda row: str(row["name"]).casefold(),
            ),
            "data_preparation_dependencies": "Python standard library only",
        }
        write_json(output / "runtime-provenance.json", provenance)
        source = acquire(
            output / "scorer" / "mrqa_official_eval.py",
            f"{SOURCE_BASE}/mrqa_official_eval.py", args.scorer_source,
        )
        readme = acquire(output / "scorer" / "README.md", f"{SOURCE_BASE}/README.md", args.readme_source)
        if source["sha256"] != SCORER_SHA256 or readme["sha256"] != README_SHA256:
            raise ValueError("pinned official scorer or README SHA256 mismatch")
        validate_readme(Path(readme["path"]))
        archives = {}
        paths = {}
        for split, supplied in (("train", args.train_archive), ("evaluation", args.dev_archive)):
            path = output / "raw" / f"SearchQA-{'dev' if split == 'evaluation' else 'train'}.jsonl.gz"
            archives[split] = {**ARCHIVES[split], **acquire(path, ARCHIVES[split]["url"], supplied)}
            if archives[split]["md5"] != ARCHIVES[split]["publisher_md5"]:
                raise ValueError(f"publisher MD5 mismatch for official {split} archive")
            paths[split] = path
        write_json(output / "artifact-hashes.json", {"archives": archives, "scorer": source, "readme": readme})
        print("Pinned archive hashes verified; scanning public question identities.", flush=True)
        identities = {
            split: scan_public_archive(paths[split], expected_count=ARCHIVES[split]["official_task_count"])
            for split in ("evaluation", "train")
        }
        if set(identities["train"]) & set(identities["evaluation"]):
            raise ValueError("official train and development qids overlap")
        selections = select_question_groups(identities["train"].values(), identities["evaluation"].values())
        selection_receipt = {
            "salt": SALT, "group_normalization": GROUP_NORMALIZATION,
            "ordering": "ascending SHA256(salt + NUL + question_group), group as tie-break",
            "official_train_task_count": len(identities["train"]),
            "official_evaluation_task_count": len(identities["evaluation"]),
            "official_train_question_group_count": len(set(identities["train"].values())),
            "official_evaluation_question_group_count": len(set(identities["evaluation"].values())),
            "train_groups_excluded_for_development_overlap": len(
                set(identities["train"].values()) & set(identities["evaluation"].values())
            ),
            "selection_fields": ["original qid", "public question"],
            "accepted_answers_consulted_for_selection": False,
            "selected_groups": {split: sorted(values) for split, values in selections.items()},
        }
        write_json(output / "selection-freeze.json", selection_receipt)
        splits = materialize_splits(paths, identities, selections, output)
        print(json.dumps({"materialized_task_counts": {key: value["task_count"] for key, value in splits.items()}}), flush=True)
        targets = {**selections, "evaluation": set(identities["evaluation"].values())}
        selected_group_union = set().union(*targets.values())
        target_ids = {
            qid: group for source_ids in identities.values() for qid, group in source_ids.items()
            if group in selected_group_union
        }
        exposure = audit_exposure(args.exposure_root, targets, target_ids, output)
        manifest = {
            "schema_version": SCHEMA, "dataset_id": DATASET_ID,
            "track": "complete official MRQA2019 SearchQA-v2 in-domain development",
            "status": "READY_FOR_OPENED_PROSPECTIVELY_FROZEN_DESCRIPTIVE_STUDY",
            "identity_boundary": "Not original SearchQA; not final out-of-domain MRQA testing.",
            "created_at_utc": config["created_at_utc"], "hostname": socket.gethostname(),
            "scorer": {
                **source, "repository": REPOSITORY, "revision": REVISION,
                "unmodified_official_source": True, "scorer_executed_during_preparation": False,
                "cli": [sys.executable, source["path"], archives["evaluation"]["path"], "predictions.json"],
                "api": "evaluate(answers, predictions, skip_no_answer=False)",
                "prediction_format": "JSON object: original qid -> answer string",
                "metrics": ["exact_match", "f1"], "scale": "percentage",
                "aggregation": "arithmetic mean over all official qids; per-qid max over accepted answers",
                "missing_predictions": "zero credit; skip_no_answer must remain false",
                "normalization": "lowercase; delete ASCII punctuation; remove a/an/the; collapse whitespace",
                "empty_token_overlap_f1": 0,
                "import_dependency": "allennlp.common.file_utils.cached_path",
                "runtime_status": "SOURCE_READY; native CLI dependency must be validated by scorer runner",
                "readme": readme,
            },
            "raw_archives": archives, "splits": splits,
            "question_group_normalization": GROUP_NORMALIZATION, "selection_salt": SALT,
            "selection_freeze_path": str(output / "selection-freeze.json"),
            "selection_freeze_sha256": file_hashes(output / "selection-freeze.json")["sha256"],
            "official_evaluation_complete": True, "evaluation_history_exclusions": 0,
            "train_dev_overlap_groups_excluded": selection_receipt["train_groups_excluded_for_development_overlap"],
            "exposure_audit_path": str(output / "exposure-audit.json"),
            "exposure_audit_sha256": file_hashes(output / "exposure-audit.json")["sha256"],
            "complete_global_exposure_coverage": exposure["complete_global_coverage"],
            "fresh_confirmation_admitted": False,
            "label_policy": {
                "evaluation_answers_path": splits["evaluation"]["answers_path"],
                "evaluation_answers_mode": "0400; sealed directory 0700",
                "answers_used_for_selection_or_routing": False,
                "raw_archives_contain_protected_labels": True,
                "model_or_route_process_must_not_open": [
                    splits["evaluation"]["answers_path"], archives["evaluation"]["path"],
                ],
                "same_uid_boundary": "Permissions are not isolation from the same UID; runner must enforce label gate.",
            },
            "runtime_provenance_path": str(output / "runtime-provenance.json"),
            "runtime_provenance_sha256": file_hashes(output / "runtime-provenance.json")["sha256"],
            "run_config_path": str(output / "run-config.json"),
            "run_config_sha256": file_hashes(output / "run-config.json")["sha256"],
            "benchmark_calls": 0, "evaluator_calls": 0, "model_calls": 0,
        }
        manifest["canonical_sha256"] = canonical_hash(manifest)
        write_json(output / "data-manifest.json", manifest)
        return manifest
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "FAILED", "exception_type": type(error).__name__,
            "message": str(error) if isinstance(error, (ValueError, RuntimeError)) else "See acquisition/runtime provenance.",
            "labels_printed": False, "hostname": socket.gethostname(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-provenance", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path)
    parser.add_argument("--dev-archive", type=Path)
    parser.add_argument("--scorer-source", type=Path)
    parser.add_argument("--readme-source", type=Path)
    parser.add_argument("--exposure-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps({
        "status": manifest["status"], "dataset_id": DATASET_ID,
        "data_manifest": str(args.output_dir.resolve() / "data-manifest.json"),
        "canonical_sha256": manifest["canonical_sha256"],
        "splits": {
            split: {key: value[key] for key in ("task_count", "question_group_count")}
            for split, value in manifest["splits"].items()
        },
        "fresh_confirmation_admitted": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
