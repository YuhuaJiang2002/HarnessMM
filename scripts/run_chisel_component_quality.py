#!/usr/bin/env python3
"""Remote-only, fail-closed MRQA component controller.

``init`` freezes the inputs; ``run --authorize-protocol SHA256`` explicitly
authorizes that frozen protocol. A stopped experiment is never replayed.
``verify`` checks a completed release without contacting a model.

The service receipt binds compact checkpoint-evidence pointers and a self-hashed
tokenizer contract. Their signed files are verified before use. Missing receipt
fields or native scorer dependencies are blockers, never inferred successes.
``GENERATION_DRAINED.json`` is the parent's resource-handoff event, not permission
to terminate a process. CPU-only continuation never requires a retired model PID.
An explicit ``init --scorer-runtime cached-path-local-only-v1`` freezes a
local-file-only cached_path import shim; it never modifies official metric code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.evaluation.chisel_component_quality import (  # noqa: E402
    adjust_primary_family,
    paired_quality_effect,
)
from phiagent.learning.chisel_component_quality import (  # noqa: E402
    fit_quality_heads,
    freeze_quality_routes,
)
from phiagent.learning.residual_meta_harness import canonical_hash, canonical_json  # noqa: E402
from scripts.probe_chisel_model_fidelity import response_identity  # noqa: E402
from scripts.run_searchqa_residual_meta_campaign import (  # noqa: E402
    FEATURE_SCHEMA,
    extract_answer,
    file_sha256,
    task_features,
    verify_self_hash,
)

HOSTS = frozenset({"harnessmm-worker-1", "harnessmm-worker-2"})
TRAIN_ARMS = ("delete", "modify", "skillopt")
PRIMITIVES = ("no_skill", "skillopt", "autosaddler", "add", "delete", "modify", "neutral_modify")
MODELS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B")
SCHEMA = "phiagent-chisel-component-controller/v1"
HTTP_TIMEOUT = 120
SCORER_RUNTIMES = ("native", "cached-path-local-only-v1")
CHAT_TEMPLATE_KWARGS = {
    "tokenize": True, "add_generation_prompt": True, "enable_thinking": False, "return_dict": False,
}


class ProtocolError(RuntimeError):
    """A durable, non-retryable protocol blocker."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_guard() -> None:
    if socket.gethostname() not in HOSTS:
        raise ProtocolError("benchmark reads and calls require an authorized remote hostname")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_lines(path: Path) -> list:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def encoded(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".pending")
    with staged.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProtocolError(f"immutable artifact changed: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def freeze_json(path: Path, value: Any) -> None:
    freeze_bytes(path, encoded(value))


def signed(value: Mapping[str, Any]) -> dict:
    result = dict(value)
    result["canonical_sha256"] = canonical_hash(result)
    return result


def self_checked(path: Path, schema: str) -> dict:
    value = read_json(path)
    verify_self_hash(value, label=str(path))
    if value["schema_version"] != schema:
        raise ProtocolError(f"wrong schema: {path}")
    return value


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=60)


def package_versions() -> dict:
    return {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }


def git_provenance(root: Path, source_state: Path | None) -> dict:
    if source_state is None:
        return {
            "git_state_source": "current_working_tree",
            "git_head": capture(["git", "rev-parse", "HEAD"]).strip(),
            "git_status": capture(["git", "--no-pager", "status", "--porcelain"]),
            "git_diff": capture(["git", "--no-pager", "diff", "--binary", "HEAD"]),
        }
    state = read_json(source_state)
    expected = {"captured_at_utc", "repository", "head", "branch", "tracked_status_sha256"}
    if (not isinstance(state, dict) or set(state) != expected
            or state["repository"] != "YuhuaJiang2002/PhiAgent"
            or not isinstance(state["branch"], str) or not state["branch"]):
        raise ProtocolError("external source Git state has an invalid identity")
    for field, length in (("head", 40), ("tracked_status_sha256", 64)):
        value = state[field]
        if (not isinstance(value, str) or len(value) != length
                or any(character not in "0123456789abcdef" for character in value)):
            raise ProtocolError(f"external source Git state has an invalid {field}")
    captured = datetime.fromisoformat(state["captured_at_utc"].replace("Z", "+00:00"))
    if captured.tzinfo is None:
        raise ProtocolError("external source Git state requires an explicit timezone")
    freeze_bytes(root / "source-git-state.json", source_state.read_bytes())
    return {
        "git_state_source": "explicit_external_source_snapshot",
        "git_head": state["head"], "git_branch": state["branch"],
        "source_git_state": state,
        "source_git_state_sha256": file_sha256(root / "source-git-state.json"),
        "git_boundary": "Original HEAD and status digest; frozen source_sha256 records deployed dirty/untracked file bytes. The deployment directory is not relabeled as a Git checkout.",
    }


def source_paths() -> list[Path]:
    return sorted(set(ROOT.joinpath("phiagent").rglob("*.py")) | {
        Path(__file__).resolve(),
        ROOT / "scripts/run_searchqa_residual_meta_campaign.py",
        ROOT / "scripts/probe_chisel_model_fidelity.py",
        ROOT / "scripts/prepare_chisel_component_quality_data.py",
    })


def check_config(config: dict) -> None:
    if (
        config["schema_version"] != "phiagent-chisel-component-quality-design/v1"
        or config["track"]["revision"] != "b86bbcc63352786f0ca246b76191f062dce4b52b"
        or config["track"]["evaluation_tasks"] != 16980
        or config["track"]["metrics"] != ["exact_match", "f1"]
        or config["models"] != list(MODELS)
        or config["primitive_arms"] != list(PRIMITIVES)
        or config["sampling"]["evaluation_seeds"] != [20260913, 20260914, 20260915]
        or config["sampling"]["training_seed"] != 20260913
        or config["routing"]["fold_seeds"] != config["sampling"]["evaluation_seeds"]
        or config["sampling"]["temperature"] != 0
        or config["sampling"]["enable_thinking"] is not False
        or config["execution"]["transport_retries"] != 0
        or not 1 <= config["execution"]["workers_per_model"] <= 16
        or not 0 < config["execution"]["maximum_seconds"] <= 172800
        or config["execution"]["capacity_audit_seconds"] != 300
        or config["execution"]["quality_blind_workflow_gate_training_tasks"] != 12
        or config["execution"]["quality_blind_workflow_gate_arms"]
        != [*TRAIN_ARMS, "neutral_modify"]
        or config["execution"]["stop_if_all_primary_routes_are_structural_aliases"] is not True
        or config["primary_contrasts"]["residual_editing"] != ["chisel_oof", "residual_placebo_oof"]
        or config["primary_contrasts"]["residual_editing_vs_delete"] != ["chisel_oof", "delete"]
        or config["statistics"]["primary_holm_family_size"]
        != len(config["primary_contrasts"]) * len(MODELS) * 2
    ):
        raise ProtocolError("not the frozen complete MRQA component design")


def manifest_path(manifest: dict, name: str) -> Path:
    path = Path(manifest[name])
    if not path.is_absolute():
        raise ProtocolError(f"{name} must be an absolute artifact path")
    return path


def load_public(data: dict, config: dict) -> dict[str, list[dict]]:
    remote_guard()
    from scripts.prepare_chisel_component_quality_data import question_group

    if data["dataset_id"] != "MRQA2019/SearchQA-v2":
        raise ProtocolError("original SearchQA and MRQA SearchQA-v2 are not interchangeable")
    if (data["official_evaluation_complete"] is not True
            or data["fresh_confirmation_admitted"] is not False
            or data["evaluation_history_exclusions"] != 0):
        raise ProtocolError("complete development data and non-fresh identity must be explicit")
    result, groups = {}, {}
    for split in ("train", "calibration", "evaluation"):
        record = data["splits"][split]
        path = manifest_path(record, "public_path")
        if file_sha256(path) != record["public_sha256"]:
            raise ProtocolError(f"{split} public file changed")
        rows = read_json_lines(path)
        if len(rows) != record["task_count"] or len({row["id"] for row in rows}) != len(rows):
            raise ProtocolError(f"{split} task coverage differs")
        for row in rows:
            if set(row) != {"id", "question", "context", "group"} or any(
                not isinstance(row[key], str) for key in row
            ) or not row["id"] or not row["group"]:
                raise ProtocolError("public inputs must contain only the four declared fields")
            if row["group"] != question_group(row["question"]):
                raise ProtocolError("public normalized-question group identity differs")
        groups[split] = {row["group"] for row in rows}
        if len(groups[split]) != record["question_group_count"]:
            raise ProtocolError("question group counts differ")
        result[split] = rows
    if any(groups[a] & groups[b] for a, b in (
        ("train", "calibration"), ("train", "evaluation"), ("calibration", "evaluation"),
    )):
        raise ProtocolError("question-group split leakage")
    if (
        len(groups["train"]) != config["data"]["training_question_groups"]
        or len(groups["calibration"]) != config["data"]["calibration_question_groups"]
        or len(result["evaluation"]) != config["track"]["evaluation_tasks"]
    ):
        raise ProtocolError("full preregistered groups/cohort are required")
    return result


def gpu_inventory() -> list[dict]:
    fields = ("index", "uuid", "name", "memory.total", "memory.used", "utilization.gpu")
    text = capture([
        "nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits",
    ])
    return [dict(zip(fields, [entry.strip() for entry in row]))
            for row in csv.reader(io.StringIO(text))]


def service_models(services: dict) -> dict[str, dict]:
    records = services["models"]
    if not isinstance(records, list) or len(records) != len(MODELS):
        raise ProtocolError("service manifest requires the two explicit model records as a list")
    if any(not isinstance(record, dict) or not isinstance(record.get("model_repository"), str)
           for record in records):
        raise ProtocolError("each service record requires an explicit model repository")
    models = {record["model_repository"]: record for record in records}
    if len(models) != len(records) or set(models) != set(MODELS):
        raise ProtocolError("service model identities are duplicated, missing, or unexpected")
    return models


def validate_tokenizer_contract(record: dict) -> dict:
    contract = record["tokenizer_contract"]
    verify_self_hash(contract, label="served tokenizer contract")
    if (contract["status"] != "VERIFIED"
            or contract["all_saved_synthetic_fixture_counts_match"] is not True
            or contract["apply_chat_template_kwargs"] != CHAT_TEMPLATE_KWARGS
            or contract["local_files_only"] is not True
            or contract["trust_remote_code"] is not False
            or contract["use_fast"] is not True
            or contract["tokenizer_path"] != record["tokenizer_path"]
            or contract["checkpoint_path"] != record["checkpoint_path"]
            or contract["effective_chat_template_sha256"] != record["chat_template_sha256"]):
        raise ProtocolError("served tokenizer contract differs from the exact request tokenizer")
    return contract


def proc_start_time_ticks(stat_text: str) -> int:
    parts = stat_text.rsplit(") ", 1)
    if len(parts) != 2 or len(parts[1].split()) <= 19:
        raise ProtocolError("invalid /proc process stat identity")
    try:
        return int(parts[1].split()[19])
    except ValueError as error:
        raise ProtocolError("invalid /proc process start time") from error


def observed_process_identity(pid: int) -> dict:
    directory = Path(f"/proc/{pid}")
    first_start = proc_start_time_ticks((directory / "stat").read_text())
    raw_command = (directory / "cmdline").read_bytes()
    result = {
        "pid": pid, "start_time_ticks": first_start,
        "cmdline_sha256": hashlib.sha256(raw_command).hexdigest(),
        "uid": directory.stat().st_uid, "cwd": str((directory / "cwd").resolve(strict=True)),
    }
    if proc_start_time_ticks((directory / "stat").read_text()) != first_start:
        raise ProtocolError("process identity changed while reading its attestation")
    return result


def validate_server_identity(record: dict, observed: dict) -> None:
    expected = record["server_identity"]
    if (expected["pid"] != record["server_pid"]
            or any(observed[key] != expected[key]
                   for key in ("pid", "start_time_ticks", "cmdline_sha256", "uid"))
            or Path(observed["cwd"]) != Path(record["service_dir"])):
        raise ProtocolError("live API process identity or cwd differs from the attested lease owner")


def validate_gpu_bindings(services: dict, inventory: list[dict]) -> None:
    remote_guard()
    if services["hostname"] != socket.gethostname():
        raise ProtocolError("service hostname differs from controller")
    available = {int(row["index"]): row for row in inventory}
    selected = set()
    models = service_models(services)
    for model in MODELS:
        record = models[model]
        index, uuid = record["physical_gpu_index"], record["gpu_uuid"]
        if type(index) is not int or index in selected or index not in available:
            raise ProtocolError("each model requires its own validated physical GPU")
        selected.add(index)
        if available[index]["uuid"] != uuid:
            raise ProtocolError("physical GPU/UUID binding changed")
        pid = record["server_pid"]
        if type(pid) is not int or pid <= 0:
            raise ProtocolError("invalid service PID")
        validate_server_identity(record, observed_process_identity(pid))
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        visibility = [entry.split(b"=", 1)[1].decode() for entry in environment
                      if entry.startswith(b"CUDA_VISIBLE_DEVICES=")]
        if visibility not in ([str(index)], [uuid]):
            raise ProtocolError("live server PID GPU visibility disagrees with attestation")
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if hashlib.sha256(raw_command).hexdigest() != record["server_identity"]["cmdline_sha256"]:
            raise ProtocolError("live API command changed after identity verification")
        raw_arguments = raw_command.decode().rstrip("\0").split("\0")
        arguments = [part for argument in raw_arguments for part in (
            argument.split("=", 1) if argument.startswith("--") and "=" in argument else [argument]
        )]
        if record["served_model"] not in arguments:
            raise ProtocolError("live service command does not bind the served model")
        checkpoint = manifest_path(record, "checkpoint_path").resolve()
        model_flag = "--model" if "--model" in arguments else "serve"
        offset = arguments.index(model_flag) + 1 if model_flag in arguments else len(arguments)
        if (offset >= len(arguments) or not Path(arguments[offset]).is_absolute()
                or Path(arguments[offset]).resolve() != checkpoint):
            raise ProtocolError("served alias is insufficient: live process must bind attested checkpoint")
        tokenizer = manifest_path(record, "tokenizer_path").resolve()
        if "--tokenizer" in arguments:
            offset = arguments.index("--tokenizer") + 1
            if offset >= len(arguments) or Path(arguments[offset]).resolve() != tokenizer:
                raise ProtocolError("live process tokenizer differs from the attested tokenizer")
        elif tokenizer != checkpoint:
            raise ProtocolError("separate attested tokenizer is not bound in the live server command")


def verify_file_records(records: Mapping[str, str], directory: Path) -> None:
    if not records:
        raise ProtocolError("empty checkpoint/tokenizer file attestation")
    for relative, expected in records.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProtocolError("attested filename must be relative to its artifact directory")
        path = directory / relative
        if file_sha256(path) != expected:
            raise ProtocolError(f"attested model/tokenizer file changed: {path}")


def checkpoint_file_records(record: dict, evidence: dict) -> dict[str, str]:
    pointer = record["checkpoint_evidence"]
    verify_self_hash(evidence, label="complete checkpoint verification file")
    if (evidence["canonical_sha256"] != pointer["canonical_sha256"]
            or evidence["status"] != "VERIFIED"
            or evidence["repository"] != record["model_repository"]
            or evidence["revision"] != record["revision"]
            or evidence["model_root"] != record["checkpoint_path"]
            or pointer["model_root"] != record["checkpoint_path"]
            or evidence["metadata_sha256"] != pointer["upstream_metadata_sha256"]
            or evidence["verified_bytes"] != pointer["verified_bytes"]):
        raise ProtocolError("checkpoint verification file does not bind the declared model")
    root = manifest_path(record, "checkpoint_path")
    files = {}
    declared_bytes = 0
    for item in evidence["files"]:
        path = Path(item["path"])
        if path.is_absolute() and not path.is_relative_to(root):
            raise ProtocolError("checkpoint evidence file lies outside its declared model directory")
        relative = str(path.relative_to(root) if path.is_absolute() else path)
        if (".." in Path(relative).parts or relative in files
                or type(item["bytes"]) is not int or item["bytes"] < 0
                or not isinstance(item["proof"]["kind"], str) or not item["proof"]["kind"]
                or not isinstance(item["proof"]["expected"], str) or not item["proof"]["expected"]):
            raise ProtocolError("checkpoint evidence has invalid or duplicate file proofs")
        files[relative] = item["sha256"]
        declared_bytes += item["bytes"]
    if not files or declared_bytes != evidence["verified_bytes"]:
        raise ProtocolError("checkpoint evidence byte coverage differs")
    for name, expected in pointer["config_tokenizer_template_hashes"].items():
        if files[name] != expected:
            raise ProtocolError("checkpoint pointer hash differs from its complete evidence file")
    return files


def tokenizer_file_records(record: dict) -> dict[str, str]:
    contract = validate_tokenizer_contract(record)
    root = manifest_path(record, "tokenizer_path")
    files = {}
    for name, item in contract["files"].items():
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or Path(item["path"]) != root / name):
            raise ProtocolError("tokenizer contract file path differs")
        files[name] = item["sha256"]
    if not files:
        raise ProtocolError("tokenizer contract has no file attestations")
    return files


def service_normalization_receipt(record: dict, checkpoint_files: dict, tokenizer_files: dict,
                                  *, source_manifest_canonical_sha256: str) -> dict:
    return signed({
        "schema_version": "phiagent-chisel-service-normalization/v1",
        "source_manifest_canonical_sha256": source_manifest_canonical_sha256,
        "model_repository": record["model_repository"], "revision": record["revision"],
        "checkpoint_evidence_canonical_sha256": record["checkpoint_evidence"]["canonical_sha256"],
        "tokenizer_contract_canonical_sha256": record["tokenizer_contract"]["canonical_sha256"],
        "checkpoint": {"path": record["checkpoint_path"], "files_sha256": dict(checkpoint_files)},
        "tokenizer": {"path": record["tokenizer_path"], "files_sha256": dict(tokenizer_files)},
        "source_manifest_modified": False,
    })


def verify_service_evidence(root: Path, record: dict, *, source_manifest_canonical_sha256: str) -> dict:
    remote_guard()
    evidence_path = manifest_path(record["checkpoint_evidence"], "path")
    evidence = read_json(evidence_path)
    checkpoint_files = checkpoint_file_records(record, evidence)
    upstream = evidence_path.with_name(f"{record['key']}-upstream-metadata.json")
    if file_sha256(upstream) != record["checkpoint_evidence"]["upstream_metadata_sha256"]:
        raise ProtocolError("upstream model/revision metadata file changed")
    contract = validate_tokenizer_contract(record)
    contract_path = manifest_path(contract, "evidence_path")
    if read_json(contract_path) != contract:
        raise ProtocolError("inline tokenizer contract differs from its signed evidence file")
    gate = record["synthetic_gate"]
    summary_path = Path(gate["artifact_dir"]) / "summary.json"
    summary = read_json(summary_path)
    verify_self_hash(summary, label="complete synthetic service gate summary")
    accounting = gate["tokenizer_prompt_token_accounting"]
    if (summary["canonical_sha256"] != gate["canonical_sha256"]
            or summary["status"] != "PASS"
            or summary["benchmark_data_used"] is not False
            or not 0 < summary["generation_calls"] <= summary["maximum_authorized_generation_calls"]
            or summary["generation_calls"] != gate["generation_calls"]
            or len(summary["results"]) != gate["generation_calls"]
            or contract["fixture_count"] != gate["generation_calls"]
            or gate["quality_evidence"] is not False
            or accounting["status"] != "PASS"
            or accounting["new_generation_calls"] != 0
            or accounting["canonical_sha256"] != contract["canonical_sha256"]
            or accounting["verified_saved_fixture_count"] != contract["fixture_count"]):
        raise ProtocolError("synthetic gate pointer does not bind its complete signed evidence")
    if any(valid_usage(item) is None for item in summary["results"]):
        raise ProtocolError("synthetic preparation usage is unknown")
    checkpoint_root = manifest_path(record, "checkpoint_path")
    for item in evidence["files"]:
        path = Path(item["path"])
        path = path if path.is_absolute() else checkpoint_root / path
        if path.stat().st_size != item["bytes"]:
            raise ProtocolError("checkpoint file size differs from its signed evidence")
    tokenizer_files = tokenizer_file_records(record)
    verify_file_records(checkpoint_files, manifest_path(record, "checkpoint_path"))
    verify_file_records(tokenizer_files, manifest_path(record, "tokenizer_path"))
    destination = root / "service-evidence" / model_key(record["model_repository"])
    for source, name in (
        (evidence_path, "checkpoint-verification.json"),
        (upstream, "upstream-metadata.json"),
        (contract_path, "tokenizer-contract.json"),
        (summary_path, "synthetic-gate-summary.json"),
    ):
        freeze_bytes(destination / name, source.read_bytes())
    normalized = service_normalization_receipt(
        record, checkpoint_files, tokenizer_files,
        source_manifest_canonical_sha256=source_manifest_canonical_sha256,
    )
    freeze_json(destination / "normalized-artifacts.json", normalized)
    return {
        "normalization_canonical_sha256": normalized["canonical_sha256"],
        "checkpoint_evidence_sha256": evidence["canonical_sha256"],
        "tokenizer_contract_sha256": contract["canonical_sha256"],
        "synthetic_gate_sha256": summary["canonical_sha256"],
        "preparation_calls": summary["generation_calls"],
        "preparation_tokens": sum(item["usage"]["total_tokens"] for item in summary["results"]),
        "preparation_cost_separate_from_controller_dispatches": True,
    }


def service_preflight(root: Path, services: dict) -> dict:
    remote_guard()
    if services["status"] != "READY":
        raise ProtocolError("service receipt has not reached the explicit READY gate")
    inventory = gpu_inventory()
    validate_gpu_bindings(services, inventory)
    # HTTP clients deliberately see no accelerator; validation precedes this binding.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    models = service_models(services)
    evidence_receipts = {}
    for model in MODELS:
        record = models[model]
        if (
            record["model_repository"] != model
            or not isinstance(record["revision"], str) or len(record["revision"]) != 40
            or any(character not in "0123456789abcdef" for character in record["revision"])
            or not record["package_versions"]
            or record["synthetic_gate"]["status"] != "PASS"
        ):
            raise ProtocolError("checkpoint provenance or synthetic service gate is incomplete")
        validate_tokenizer_contract(record)
        url = urlparse(record["base_url"])
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ProtocolError("only the attested local remote-server endpoint is authorized")
        evidence_receipts[model] = verify_service_evidence(
            root, record, source_manifest_canonical_sha256=services["canonical_sha256"],
        )
    receipt = signed({
        "at": now(), "hostname": socket.gethostname(), "gpu_inventory": inventory,
        "services_sha256": services["canonical_sha256"], "CUDA_VISIBLE_DEVICES": "",
        "verified_service_evidence": evidence_receipts,
    })
    freeze_json(root / "attempts" / f"preflight-{time.time_ns()}.json", receipt)
    return receipt


def init(args: argparse.Namespace) -> int:
    remote_guard()
    root = args.experiment_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    success = False
    try:
        freeze_json(root / "initialization.json", {
            "at": now(), "hostname": socket.gethostname(), "argv": sys.argv, "python": sys.version,
        })
        config = read_json(args.config)
        check_config(config)
        data = self_checked(args.data_manifest, "phiagent-chisel-component-data/v1")
        services = self_checked(args.service_manifest, "phiagent-chisel-component-services/v1")
        for record in service_models(services).values():
            validate_tokenizer_contract(record)
        public = load_public(data, config)
        freeze_bytes(root / "config.json", args.config.read_bytes())
        freeze_bytes(root / "data-manifest.json", args.data_manifest.read_bytes())
        freeze_bytes(root / "service-manifest.json", args.service_manifest.read_bytes())
        sources = {}
        for path in source_paths():
            relative = str(path.relative_to(ROOT))
            freeze_bytes(root / "sources" / relative, path.read_bytes())
            sources[relative] = file_sha256(path)
        prompts = {}
        for arm, expected in config["prompt_sha256"].items():
            path = ROOT / config["prompt_source"] / f"{arm}.system.txt"
            if file_sha256(path) != expected:
                raise ProtocolError(f"frozen {arm} prompt differs")
            freeze_bytes(root / "prompts" / path.name, path.read_bytes())
            prompts[arm] = expected
        metadata = {
            "at": now(), "hostname": socket.gethostname(), "argv": sys.argv,
            "python": sys.version, "package_versions": package_versions(),
            **git_provenance(root, args.source_git_state),
        }
        freeze_json(root / "provenance.json", metadata)
        protocol = signed({
            "schema_version": SCHEMA, "hostname": socket.gethostname(),
            "campaign_id": config["campaign_id"], "config_sha256": file_sha256(root / "config.json"),
            "data_manifest_sha256": file_sha256(root / "data-manifest.json"),
            "service_manifest_sha256": file_sha256(root / "service-manifest.json"),
            "source_sha256": sources, "prompt_sha256": prompts,
            "gate_task_ids": [item["id"] for item in public["train"][
                :config["execution"]["quality_blind_workflow_gate_training_tasks"]
            ]],
            "request_encoding": "canonical_json UTF-8, no trailing newline",
            "official_scorer_runtime": {
                "mode": args.scorer_runtime,
                "official_source_unmodified": True,
                "automatic_dependency_fallback": False,
                "compatibility_boundary": (
                    "Only allennlp.common.file_utils.cached_path may be replaced, and only in the "
                    "explicit local-only mode; official normalization, metrics and CLI remain unchanged."
                ),
            },
            "padding_search": "bounded exact chat-template token equality; no generation",
            "exposure_boundary": "NOT_FRESH; incomplete historical inventory never establishes freshness",
        })
        freeze_json(root / "protocol.json", protocol)
        atomic_json(root / "state.json", {"status": "INITIALIZED", "phase": "gate"})
        Ledger(root / "ledger.sqlite3").close()
        success = True
        print(canonical_json({"status": "INITIALIZED", "protocol_sha256":
                              protocol["canonical_sha256"], "experiment_dir": str(root)}))
        return 0
    finally:
        if not success:
            atomic_json(root / "FAILURE.json", {
                "status": "BLOCKED", "at": now(), "phase": "init",
                "error": str(sys.exc_info()[1]), "no_model_calls": True,
            })


def load_frozen(root: Path) -> tuple[dict, dict, dict, dict]:
    remote_guard()
    protocol = self_checked(root / "protocol.json", SCHEMA)
    if protocol["official_scorer_runtime"]["mode"] not in SCORER_RUNTIMES:
        raise ProtocolError("unrecognized frozen official scorer runtime")
    if protocol["hostname"] != socket.gethostname():
        raise ProtocolError("frozen controller hostname differs")
    for name, field in (
        ("config.json", "config_sha256"),
        ("data-manifest.json", "data_manifest_sha256"),
        ("service-manifest.json", "service_manifest_sha256"),
    ):
        if file_sha256(root / name) != protocol[field]:
            raise ProtocolError(f"frozen {name} changed")
    for relative, expected in protocol["source_sha256"].items():
        if file_sha256(root / "sources" / relative) != expected:
            raise ProtocolError("source snapshot changed")
        if file_sha256(ROOT / relative) != expected:
            raise ProtocolError("running source differs from immutable source snapshot")
    for arm, expected in protocol["prompt_sha256"].items():
        if file_sha256(root / "prompts" / f"{arm}.system.txt") != expected:
            raise ProtocolError("frozen prompt changed")
    return (
        protocol, read_json(root / "config.json"),
        self_checked(root / "data-manifest.json", "phiagent-chisel-component-data/v1"),
        self_checked(root / "service-manifest.json", "phiagent-chisel-component-services/v1"),
    )


class Ledger:
    """FULL-synchronous intents and raw result BLOBs; no replay of ambiguous calls."""

    def __init__(self, path: Path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS intents(
                cell TEXT PRIMARY KEY, model TEXT NOT NULL, split TEXT NOT NULL,
                task TEXT NOT NULL, seed INTEGER NOT NULL, arm TEXT NOT NULL,
                request_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
                padding_units INTEGER, started REAL NOT NULL, endpoint TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS results(
                cell TEXT PRIMARY KEY REFERENCES intents(cell), ok INTEGER NOT NULL,
                raw BLOB NOT NULL, raw_sha256 TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS padding(
                model TEXT NOT NULL, task TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY(model, task));
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()

    def recovery_check(self) -> None:
        with self.lock:
            unmatched = self.db.execute(
                "SELECT COUNT(*) FROM intents LEFT JOIN results USING(cell) WHERE results.cell IS NULL"
            ).fetchone()[0]
            failed = self.db.execute("SELECT COUNT(*) FROM results WHERE ok=0").fetchone()[0]
        if unmatched or failed:
            raise ProtocolError(
                f"replay blocked: {unmatched} unmatched intents (unknown cost), {failed} failed calls"
            )

    def has_result(self, cell: str) -> bool:
        with self.lock:
            return self.db.execute(
                "SELECT 1 FROM results WHERE cell=? AND ok=1", (cell,),
            ).fetchone() is not None

    def intent(self, cell: dict, request: bytes, prompt_tokens: int, padding: int | None) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO intents VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                cell["cell"], cell["model"], cell["split"], cell["task"], cell["seed"],
                cell["arm"], hashlib.sha256(request).hexdigest(), prompt_tokens, padding,
                time.time(), cell["endpoint"],
            ))

    def result(self, cell: str, *, ok: bool, raw: bytes, metadata: dict) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO results VALUES(?,?,?,?,?)", (
                cell, int(ok), raw, hashlib.sha256(raw).hexdigest(), canonical_json(metadata),
            ))

    def rows(self, model: str, split: str, seed: int, arm: str) -> dict[str, dict]:
        with self.lock:
            rows = self.db.execute("""
                SELECT task, raw, metadata FROM intents JOIN results USING(cell)
                WHERE model=? AND split=? AND seed=? AND arm=? AND ok=1 ORDER BY intents.rowid
            """, (model, split, seed, arm)).fetchall()
        return {task: {"raw": json.loads(raw), **json.loads(metadata)} for task, raw, metadata in rows}

    def costs(self) -> dict:
        known_tokens = calls = unknown = failed = 0
        by_phase: dict[str, dict] = {}
        with self.lock:
            rows = self.db.execute("""
                SELECT split, arm, ok, metadata FROM intents LEFT JOIN results USING(cell)
            """).fetchall()
        for split, arm, ok, text in rows:
            phase = "gate_neutral_only" if split == "train" and arm == "neutral_modify" else split
            bucket = by_phase.setdefault(phase, {"actual_calls": 0, "known_tokens": 0,
                                                "unknown_usage_calls": 0})
            calls += 1
            bucket["actual_calls"] += 1
            usage = json.loads(text).get("usage") if text else None
            if usage is None:
                unknown += 1
                bucket["unknown_usage_calls"] += 1
            else:
                known_tokens += usage["total_tokens"]
                bucket["known_tokens"] += usage["total_tokens"]
            failed += int(ok != 1)
        return {
            "actual_calls": calls, "known_tokens": known_tokens, "unknown_usage_calls": unknown,
            "unsuccessful_or_unmatched_calls": failed, "cost_complete": unknown == 0,
            "by_phase": by_phase, "selected_policy_cost_is_not_actual_campaign_cost": True,
        }

    def progress(self) -> dict:
        with self.lock:
            intents = self.db.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
            results = self.db.execute("SELECT COUNT(*), COALESCE(SUM(ok),0) FROM results").fetchone()
        return {"intents": intents, "results": results[0], "successful_cells": results[1],
                "inflight_or_unmatched": intents - results[0]}

    def dispatch_digest(self) -> str:
        digest = hashlib.sha256()
        with self.lock:
            for row in self.db.execute("""
                SELECT cell, request_sha256, raw_sha256 FROM intents JOIN results USING(cell)
                ORDER BY intents.rowid
            """):
                digest.update(encoded(list(row)))
        return digest.hexdigest()


def cell_spec(model: str, split: str, item: dict, seed: int, arm: str, endpoint: str) -> dict:
    return {
        "cell": canonical_hash([model, split, item["id"], seed, arm]),
        "model": model, "split": split, "task": item["id"], "seed": seed,
        "arm": arm, "endpoint": endpoint,
    }


def build_messages(system: str, item: Mapping[str, str]) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"## Context\n{item['context']}\n\n## Question\n{item['question']}"},
    ]


def prompt_length(tokenizer: Any, messages: list[dict]) -> int:
    tokens = tokenizer.apply_chat_template(messages, **CHAT_TEMPLATE_KWARGS)
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise ProtocolError("tokenizer did not return exact chat-template token IDs")
    return len(tokens)


def neutral_padding(tokenizer: Any, item: dict, prompts: dict, config: dict) -> dict:
    control = config["neutral_control"]
    target = prompt_length(tokenizer, build_messages(prompts["modify"], item))
    base = prompts["delete"] + control["padding_header"]

    def count(units: int) -> int:
        return prompt_length(tokenizer, build_messages(base + control["padding_unit"] * units, item))

    low, high = 0, max(1, target)
    while low <= high:
        middle = (low + high) // 2
        measured = count(middle)
        if measured == target:
            return {"padding_units": middle, "prompt_tokens": target,
                    "system": base + control["padding_unit"] * middle}
        if measured < target:
            low = middle + 1
        else:
            high = middle - 1
    raise ProtocolError("neutral MODIFY cannot achieve exact served chat-template token equality")


def valid_usage(raw: Any) -> dict | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("usage"), dict):
        return None
    usage = raw["usage"]
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in keys):
        return None
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        return None
    return {key: usage[key] for key in keys}


def strict_response(raw: dict, *, model: str, expected_prompt_tokens: int) -> dict:
    if (not isinstance(raw, dict) or not isinstance(raw.get("choices"), list)
            or len(raw["choices"]) != 1 or not isinstance(raw["choices"][0], dict)
            or not isinstance(raw["choices"][0].get("message"), dict)
            or not isinstance(raw.get("usage"), dict)):
        raise ProtocolError("malformed response structure")
    if raw["choices"][0]["message"].get("tool_calls"):
        raise ProtocolError("plain actor response unexpectedly contains tool calls")
    identity = response_identity(raw, expected_model=model)
    choice = raw["choices"][0]
    if choice["finish_reason"] not in {"stop", "length"}:
        raise ProtocolError("unexpected finish reason")
    if identity["usage"]["prompt_tokens"] != expected_prompt_tokens:
        raise ProtocolError("actual prompt tokens differ from exact served tokenizer")
    if identity["usage"]["total_tokens"] <= 0:
        raise ProtocolError("positive measured token usage is required")
    return {
        **identity, "model": raw["model"], "finish_reason": choice["finish_reason"],
        "empty_answer": extract_answer(choice["message"]["content"]) == "",
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProtocolError("redirect away from the attested model endpoint is forbidden")


def http_transport(endpoint: str, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
    remote_guard()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("HTTP client requires explicit CUDA_VISIBLE_DEVICES='' after GPU binding")
    request = Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return response.status, dict(response.headers), response.read()


class Controller:
    def __init__(self, root: Path, config: dict, services: dict, ledger: Ledger, deadline: float,
                 *, scorer_runtime: str = "native"):
        self.root, self.config, self.services, self.ledger = root, config, services, ledger
        self.models = service_models(services)
        self.deadline = deadline
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.phase = "preflight"
        self.tokenizers: dict[str, Any] = {}
        self.prompts = {arm: (root / "prompts" / f"{arm}.system.txt").read_text(encoding="utf-8")
                        for arm in PRIMITIVES if arm != "neutral_modify"}
        self.transport = http_transport
        self.started_at = time.time()
        self.last_audit = 0.0
        self.last_heartbeat = 0.0
        self.generation_closed = False
        self.matrix_active = False
        self.scorer_runtime = scorer_runtime

    def stop(self, reason: str, **evidence: Any) -> None:
        with self.lock:
            self.stopped.set()
            record = {"status": "BLOCKED", "at": now(), "phase": self.phase,
                      "reason": reason, **evidence}
            if not (self.root / "STOP.json").exists():
                atomic_json(self.root / "STOP.json", record)
                atomic_json(self.root / "FAILURE.json", record)

    def check(self) -> None:
        with self.lock:
            if time.time() >= self.deadline:
                self.stop("48-hour lifetime budget exhausted")
            if self.stopped.is_set() or (self.root / "STOP.json").exists():
                raise ProtocolError("global stop barrier is closed; failed attempts cannot be replayed")

    def heartbeat(self) -> None:
        with self.lock:
            atomic_json(self.root / "heartbeat.json", {
                "at": now(), "hostname": socket.gethostname(), "pid": os.getpid(),
                "phase": self.phase, "deadline": self.deadline, "stopped": self.stopped.is_set(),
                "progress": self.ledger.progress(),
                "generation_closed": self.generation_closed,
            })
            self.last_heartbeat = time.time()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.check()
            self.phase = phase
            state = read_json(self.root / "state.json") if (self.root / "state.json").exists() else {}
            state.update(status="RUNNING", phase=phase, deadline=self.deadline,
                         progress=self.ledger.progress())
            atomic_json(self.root / "state.json", state)
            freeze_json(self.root / "logs" / f"phase-{time.time_ns()}.json", {
                "phase": phase, "at": now(), "progress": self.ledger.progress(),
            })
            self.heartbeat()

    def audit(self) -> None:
        with self.lock:
            inventory = gpu_inventory()
            if not self.generation_closed:
                validate_gpu_bindings(self.services, inventory)
            freeze_json(self.root / "gpu-audits" / f"{time.time_ns()}.json", {
                "at": now(), "hostname": socket.gethostname(), "gpu_inventory": inventory,
                "phase": self.phase, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "service_binding_required": not self.generation_closed,
                "generation_closed": self.generation_closed,
            })
            self.last_audit = time.time()

    def close_generation(self, protocol: dict, *, reason: str) -> dict:
        with self.lock:
            self.check()
            if self.matrix_active:
                raise ProtocolError("generation handoff requires every executor to drain first")
            self.ledger.recovery_check()
            if reason not in {"FULL_EVALUATION_MATRIX_COLLECTED", "NEGATIVE_ACTIVATION_GATE"}:
                raise ProtocolError("unknown generation handoff boundary")
            progress = self.ledger.progress()
            if progress["inflight_or_unmatched"]:
                raise ProtocolError("unmatched requests forbid generation handoff")
            receipt = signed({
                "schema_version": "phiagent-chisel-generation-drained/v1",
                "status": "GENERATION_DRAINED_NO_FURTHER_MODEL_CALLS",
                "drained_at": now(),
                "reason": reason, "protocol_sha256": protocol["canonical_sha256"],
                "service_manifest_sha256": protocol["service_manifest_sha256"],
                "hostname": socket.gethostname(), "controller_pid": os.getpid(),
                "progress": progress, "dispatch_sha256": self.ledger.dispatch_digest(),
                "actual_campaign_cost": self.ledger.costs(),
                "frozen_service_attestations": self.services["models"],
                "retirement_authorized_by_controller": False,
                "retirement_requires_explicit_parent_authorization": True,
                "retirement_requires_attested_pid_starttime_cwd_and_lease": True,
                "parent_action": (
                    "Only after explicit parent authorization, the service owner must freshly verify "
                    "supervisor/API PID, starttime, cmdline, cwd, lease and GPU UUID before retiring "
                    "only its leased services. This event alone never authorizes termination."
                ),
                "official_quality_result": False,
                "analysis_and_release_verification_may_still_be_pending": True,
            })
            self.generation_closed = True
            freeze_json(self.root / "GENERATION_DRAINED.json", receipt)
            self.set_phase("generation-drained-cpu-analysis-only")
            return receipt

    def restore_generation_handoff(self, protocol: dict) -> dict | None:
        path = self.root / "GENERATION_DRAINED.json"
        if not path.exists():
            return None
        with self.lock:
            receipt = self_checked(path, "phiagent-chisel-generation-drained/v1")
            self.ledger.recovery_check()
            if (receipt["protocol_sha256"] != protocol["canonical_sha256"]
                    or receipt["service_manifest_sha256"] != protocol["service_manifest_sha256"]
                    or receipt["hostname"] != socket.gethostname()
                    or receipt["progress"] != self.ledger.progress()
                    or receipt["dispatch_sha256"] != self.ledger.dispatch_digest()
                    or receipt["frozen_service_attestations"] != self.services["models"]
                    or receipt["reason"] not in {
                        "FULL_EVALUATION_MATRIX_COLLECTED", "NEGATIVE_ACTIVATION_GATE",
                    }):
                raise ProtocolError("persisted generation handoff no longer binds the frozen ledger")
            self.generation_closed = True
            return receipt

    def monitor(self) -> None:
        while not self.finished.wait(1):
            try:
                self.check()
                if time.time() - self.last_audit >= self.config["execution"]["capacity_audit_seconds"]:
                    self.audit()
                if time.time() - self.last_heartbeat >= self.config["execution"]["heartbeat_seconds"]:
                    self.heartbeat()
            except (ProtocolError, OSError, ValueError, subprocess.SubprocessError) as error:
                self.stop("monitor failure", error=repr(error))
                return

    def tokenizer(self, model: str) -> Any:
        remote_guard()
        if model not in self.tokenizers:
            from transformers import AutoTokenizer

            record = self.models[model]
            contract = validate_tokenizer_contract(record)
            for package in ("transformers", "tokenizers"):
                if importlib.metadata.version(package) != record["package_versions"][package]:
                    raise ProtocolError(f"client {package} differs from the attested tokenizer runtime")
            tokenizer = AutoTokenizer.from_pretrained(
                record["tokenizer_path"],
                local_files_only=True, trust_remote_code=False, use_fast=True,
            )
            identity = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
            template_hash = hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest()
            if (identity != contract["tokenizer_class"]
                    or template_hash != contract["effective_chat_template_sha256"]):
                raise ProtocolError("actual tokenizer class or effective chat template differs")
            self.tokenizers[model] = tokenizer
        return self.tokenizers[model]

    def request(self, cell: dict, item: dict) -> tuple[bytes, int, int | None]:
        model, arm = cell["model"], cell["arm"]
        tokenizer = self.tokenizer(model)
        units = None
        if arm == "neutral_modify":
            with self.ledger.lock:
                existing = self.ledger.db.execute(
                    "SELECT value FROM padding WHERE model=? AND task=?", (model, item["id"]),
                ).fetchone()
            if existing:
                plan = json.loads(existing[0])
            else:
                plan = neutral_padding(tokenizer, item, self.prompts, self.config)
                with self.ledger.lock, self.ledger.db:
                    self.ledger.db.execute("INSERT OR IGNORE INTO padding VALUES(?,?,?)",
                                           (model, item["id"], canonical_json(plan)))
                    frozen = self.ledger.db.execute(
                        "SELECT value FROM padding WHERE model=? AND task=?", (model, item["id"]),
                    ).fetchone()[0]
                    if frozen != canonical_json(plan):
                        raise ProtocolError("concurrent neutral plans disagree")
            system, units = plan["system"], plan["padding_units"]
            if (type(units) is not int or units < 0
                    or system != self.prompts["delete"] + self.config["neutral_control"]["padding_header"]
                    + self.config["neutral_control"]["padding_unit"] * units
                    or plan["prompt_tokens"] != prompt_length(
                        tokenizer, build_messages(self.prompts["modify"], item))):
                raise ProtocolError("frozen neutral padding no longer reconstructs exactly")
        else:
            system = self.prompts[arm]
        messages = build_messages(system, item)
        count = prompt_length(tokenizer, messages)
        service = self.models[model]
        sampling = self.config["sampling"]
        if count + sampling["max_output_tokens"] > service["context_length"]:
            raise ProtocolError("full unchanged context exceeds attested service window; no truncation")
        body = {
            "model": service["served_model"], "messages": messages, "stream": False,
            "temperature": sampling["temperature"], "max_tokens": sampling["max_output_tokens"],
            "seed": cell["seed"], "chat_template_kwargs": {"enable_thinking": False},
        }
        return canonical_json(body).encode("utf-8"), count, units

    def one(self, cell: dict, item: dict) -> None:
        started, recorded = False, False
        raw, headers, status, usage = b"", {}, None, None
        began = time.time()
        try:
            self.check()
            if self.generation_closed:
                raise ProtocolError("generation is permanently closed after the drained handoff")
            body, count, units = self.request(cell, item)
            with self.lock:
                self.check()
                if self.generation_closed:
                    raise ProtocolError("generation is permanently closed after the drained handoff")
                self.ledger.intent(cell, body, count, units)
                started = True
            status, headers, raw = self.transport(
                cell["endpoint"], body, min(HTTP_TIMEOUT, max(1, self.deadline - time.time())),
            )
            value = json.loads(raw)
            usage = valid_usage(value)
            if status != 200:
                raise ProtocolError(f"unexpected HTTP status {status}")
            identity = strict_response(value, model=self.models[cell["model"]]["served_model"],
                                       expected_prompt_tokens=count)
            self.ledger.result(cell["cell"], ok=True, raw=raw, metadata={
                **identity, "http_status": status, "response_headers": headers,
                "elapsed_seconds": time.time() - began, "finished_at": now(),
            })
            recorded = True
        except HTTPError as error:
            self.stop("HTTP transport failure", cell=cell, error=repr(error), usage_known=False)
            raw = error.read()
            self.ledger.result(cell["cell"], ok=False, raw=raw, metadata={
                "error": repr(error), "http_status": error.code,
                "response_headers": dict(error.headers), "usage": None, "finished_at": now(),
            })
            recorded = True
        except (ProtocolError, URLError, OSError, ValueError, KeyError, TypeError) as error:
            self.stop("protocol or transport failure", cell=cell, error=repr(error),
                      usage_known=usage is not None, usage=usage)
            if started:
                self.ledger.result(cell["cell"], ok=False, raw=raw, metadata={
                    "error": repr(error), "http_status": status, "response_headers": headers,
                    "usage": usage, "finished_at": now(),
                })
                recorded = True
        finally:
            if started and not recorded:
                self.stop("unmatched intent: cost unknown; replay forbidden", cell=cell)
            elif not started and sys.exc_info()[0] is not None:
                self.stop("request preparation failed", cell=cell, error=str(sys.exc_info()[1]))

    def matrix(self, public: dict, split: str, arms: tuple | list, seeds: list[int],
               task_ids: set[str] | None = None) -> None:
        self.set_phase(f"collect-{split}")
        with self.lock:
            self.check()
            if self.generation_closed or self.matrix_active:
                raise ProtocolError("generation is closed or another matrix is active")
            self.matrix_active = True
        items = public[split]
        workers = self.config["execution"]["workers_per_model"]

        def cells(model: str):
            endpoint = self.models[model]["base_url"]
            for item in items:
                if task_ids is not None and item["id"] not in task_ids:
                    continue
                for seed in seeds:
                    for arm in sorted(arms, key=lambda name: canonical_hash([item["id"], seed, name])):
                        cell = cell_spec(model, split, item, seed, arm, endpoint)
                        if not self.ledger.has_result(cell["cell"]):
                            yield cell, item

        pools = {model: ThreadPoolExecutor(max_workers=workers) for model in MODELS}
        streams = {model: iter(cells(model)) for model in MODELS}
        pending = {}
        exhausted = set()
        try:
            while True:
                self.check()
                for model in MODELS:
                    while model not in exhausted and sum(value == model for value in pending.values()) < workers:
                        self.check()
                        item = next(streams[model], None)
                        if item is None:
                            exhausted.add(model)
                            break
                        pending[pools[model].submit(self.one, *item)] = model
                if not pending:
                    break
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    future.result()
        finally:
            if sys.exc_info()[0] is not None:
                self.stop("matrix interrupted", error=str(sys.exc_info()[1]))
            for future in pending:
                future.cancel()
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
            with self.lock:
                self.matrix_active = False
        self.check()
        self.ledger.recovery_check()
        expected = {item["id"] for item in items if task_ids is None or item["id"] in task_ids}
        for model in MODELS:
            for seed in seeds:
                for arm in arms:
                    if set(self.ledger.rows(model, split, seed, arm)) != expected and task_ids is None:
                        raise ProtocolError("complete matrix coverage failed")
                    if not expected <= self.ledger.rows(model, split, seed, arm).keys():
                        raise ProtocolError("gate response coverage failed")
        self.heartbeat()


def model_key(model: str) -> str:
    return model.split("/")[-1]


def load_answers(data: dict, split: str, public: list[dict]) -> dict:
    remote_guard()
    record = data["splits"][split]
    path = manifest_path(record, "answers_path")
    if file_sha256(path) != record["answers_sha256"]:
        raise ProtocolError(f"{split} sealed answers changed")
    answers = read_json(path)
    if set(answers) != {row["id"] for row in public} or any(
        not isinstance(values, list) or not values or any(not isinstance(x, str) for x in values)
        for values in answers.values()
    ):
        raise ProtocolError("sealed official answer coverage differs")
    return answers


@contextmanager
def scorer_import_runtime(mode: str, *, allowed_paths: tuple[Path, ...] = ()):
    if mode not in SCORER_RUNTIMES:
        raise ProtocolError("unknown official scorer runtime")
    receipt = {
        "mode": mode, "official_metric_functions_modified": False,
        "cached_path_shim_installed": mode != "native",
        "cached_path_accesses": [], "network_access_allowed_by_shim": False,
    }
    if mode == "native":
        yield receipt
        return
    allowed = {path.resolve() for path in allowed_paths}

    def cached_path(filename, *args, **kwargs):
        if args or kwargs:
            raise ProtocolError("local-only cached_path does not support cache/download options")
        text = os.fspath(filename)
        if not isinstance(text, str) or urlparse(text).scheme or not Path(text).is_absolute():
            raise ProtocolError("local-only cached_path requires an absolute local file, never a URL")
        path = Path(text).resolve()
        if path not in allowed or not path.is_file():
            raise ProtocolError("cached_path file is outside the explicit scorer input allowlist")
        receipt["cached_path_accesses"].append({
            "requested_path": text, "resolved_path": str(path), "sha256": file_sha256(path),
        })
        return text

    names = ("allennlp", "allennlp.common", "allennlp.common.file_utils")
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in names}
    modules = {name: ModuleType(name) for name in names}
    modules["allennlp"].__path__ = []
    modules["allennlp.common"].__path__ = []
    modules["allennlp"].common = modules["allennlp.common"]
    modules["allennlp.common"].file_utils = modules["allennlp.common.file_utils"]
    modules["allennlp.common.file_utils"].cached_path = cached_path
    receipt["compatibility_boundary"] = "Only local cached_path; this is not an allennlp runtime."
    try:
        sys.modules.update(modules)
        yield receipt
    finally:
        for name in names:
            if previous[name] is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def official_scorer(data: dict, *, runtime: str = "native"):
    remote_guard()
    artifact = data["scorer"]
    if artifact["revision"] != "b86bbcc63352786f0ca246b76191f062dce4b52b":
        raise ProtocolError("official scorer revision differs")
    path = manifest_path(artifact, "path")
    if file_sha256(path) != artifact["sha256"]:
        raise ProtocolError("official scorer hash differs")
    spec = importlib.util.spec_from_file_location("frozen_mrqa_official_eval", path)
    if spec is None or spec.loader is None:
        raise ProtocolError("official scorer import failed")
    module = importlib.util.module_from_spec(spec)
    try:
        with scorer_import_runtime(runtime):
            spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        raise ProtocolError(
            f"pinned official scorer dependency missing under {runtime}: {error.name}; "
            "provision the dependency on the authorized server before any generation"
        ) from error
    for name in ("metric_max_over_ground_truths", "exact_match_score", "f1_score", "evaluate"):
        if not callable(getattr(module, name, None)):
            raise ProtocolError("official scorer interface differs")
    return module


def score_rows(rows: dict, answers: dict, scorer: Any) -> dict:
    if set(rows) != set(answers):
        raise ProtocolError("scoring a partial cohort is forbidden")
    result = {}
    for task, row in rows.items():
        prediction = extract_answer(row["raw"]["choices"][0]["message"]["content"])
        result[task] = {
            "prediction": prediction,
            "exact_match": int(scorer.metric_max_over_ground_truths(
                scorer.exact_match_score, prediction, answers[task],
            )),
            "f1": float(scorer.metric_max_over_ground_truths(
                scorer.f1_score, prediction, answers[task],
            )),
            "tokens": row["usage"]["total_tokens"],
            "prompt_tokens": row["usage"]["prompt_tokens"],
            "completion_tokens": row["usage"]["completion_tokens"],
            "finish_reason": row["finish_reason"], "empty_answer": row["empty_answer"],
        }
    return result


def routes_path(root: Path, model: str, split: str, seed: int) -> Path:
    return root / "routes" / model_key(model) / split / f"{seed}.json"


def build_training_matrix(public_items: list[dict], scores: dict) -> list[dict]:
    task_ids = {item["id"] for item in public_items}
    if set(scores) != set(TRAIN_ARMS):
        raise ProtocolError("complete training matrices contain exactly DELETE/MODIFY/SkillOpt")
    if (not task_ids or len(task_ids) != len(public_items)
            or any(set(outcomes) != task_ids for outcomes in scores.values())):
        raise ProtocolError("complete three-arm training coverage is required")
    return [{
        "id": item["id"], "group": item["group"], "features": task_features(item),
        "arms": {arm: scores[arm][item["id"]] for arm in TRAIN_ARMS},
    } for item in public_items]


def fit_and_freeze(controller: Controller, public: dict, data: dict) -> dict:
    controller.set_phase("official-train-score-and-fit")
    root, config = controller.root, controller.config
    scorer = official_scorer(data, runtime=controller.scorer_runtime)
    answers = load_answers(data, "train", public["train"])
    routing = config["routing"]
    all_routes = {}
    for model in MODELS:
        controller.check()
        scores = {arm: score_rows(controller.ledger.rows(
            model, "train", config["sampling"]["training_seed"], arm,
        ), answers, scorer) for arm in TRAIN_ARMS}
        matrix = build_training_matrix(public["train"], scores)
        freeze_json(root / "training" / model_key(model) / "matrix.json", matrix)
        final_head = None
        for seed in routing["fold_seeds"]:
            controller.check()
            head = fit_quality_heads(
                matrix, feature_schema=FEATURE_SCHEMA, folds=routing["folds"], seed=seed,
                ridge=routing["ridge"], minimum_support=routing["minimum_override_support"],
                maximum_harmful_rate=routing["maximum_harmful_rate_vs_skillopt"],
                maximum_token_delta=routing["maximum_mean_normalized_token_delta_vs_skillopt"],
            )
            shared = canonical_hash([head["intercept"], head["weights"]])
            if final_head is not None and final_head != shared:
                raise ProtocolError("fold seeds changed the required shared all-training-data head")
            final_head = shared
            freeze_json(root / "heads" / model_key(model) / f"{seed}.json", head)
            for split in ("calibration", "evaluation"):
                routes = freeze_quality_routes(
                    head, public[split], {row["id"]: task_features(row) for row in public[split]},
                    feature_schema=FEATURE_SCHEMA, seed=seed,
                )
                freeze_json(routes_path(root, model, split, seed), routes)
                all_routes[f"{model}/{split}/{seed}"] = routes
    freeze_json(root / "routes-frozen.json", signed({
        "status": "ALL_ROUTES_FROZEN_BEFORE_CALIBRATION_OR_EVALUATION_LABELS",
        "route_hashes": {key: value["canonical_sha256"] for key, value in all_routes.items()},
    }))
    return all_routes


def load_saved_routes(root: Path, config: dict) -> dict:
    receipt = read_json(root / "routes-frozen.json")
    verify_self_hash(receipt, label="all frozen routes")
    expected = {
        f"{model}/{split}/{seed}"
        for model in config["models"]
        for split in ("calibration", "evaluation")
        for seed in config["routing"]["fold_seeds"]
    }
    if set(receipt["route_hashes"]) != expected:
        raise ProtocolError("persisted route family is incomplete")
    routes = {}
    for model in config["models"]:
        for split in ("calibration", "evaluation"):
            for seed in config["routing"]["fold_seeds"]:
                key = f"{model}/{split}/{seed}"
                value = self_checked(routes_path(root, model, split, seed),
                                     "phiagent-chisel-component-routes/v1")
                if value["canonical_sha256"] != receipt["route_hashes"][key]:
                    raise ProtocolError("persisted frozen route identity differs")
                routes[key] = value
    return routes


def activation_gate(all_routes: dict, config: dict) -> dict:
    expected = {
        f"{model}/evaluation/{seed}"
        for model in config["models"]
        for seed in config["routing"]["fold_seeds"]
    }
    observed = {key for key in all_routes if "/evaluation/" in key}
    if not expected or observed != expected:
        raise ProtocolError("activation gate requires every predeclared model and fold seed")
    coverage = {}
    for key in sorted(expected):
        routes = all_routes[key]
        verify_self_hash(routes, label=key)
        if routes["seed"] != int(key.rsplit("/", 1)[1]):
            raise ProtocolError("activation route seed differs from its declared identity")
        policies = routes["override_task_ids"]
        coverage[key] = {
            "route_sha256": routes["canonical_sha256"],
            "oof_overrides": len(policies["chisel_oof"]),
            "in_sample_overrides": len(policies["chisel_in_sample"]),
        }
    all_aliases = all(
        row["oof_overrides"] == 0 and row["in_sample_overrides"] == 0 for row in coverage.values()
    )
    stop = all_aliases and config["execution"]["stop_if_all_primary_routes_are_structural_aliases"]
    return signed({
        "schema_version": "phiagent-chisel-activation-gate/v1",
        "status": "NEGATIVE_ACTIVATION_GATE" if stop else "CONTINUE_ALL_PREDECLARED_CELLS",
        "activation_cohort": "evaluation",
        "all_models_and_seeds_accounted_for": True, "route_coverage": coverage,
        "all_oof_and_in_sample_routes_alias_delete": all_aliases,
        "stop_before_calibration_and_evaluation_generation": stop,
        "official_benchmark_quality_result": False,
        "evaluation_labels_read": False,
        "boundary": (
            "Only an outcome-blind structural activation gate. Nonzero activation requires all "
            "predeclared arms/tasks/seeds/models, regardless of anticipated quality."
        ),
    })


def write_negative_activation_summary(controller: Controller, all_routes: dict, gate: dict) -> None:
    if not gate["stop_before_calibration_and_evaluation_generation"]:
        raise ProtocolError("a nonzero route family cannot be reported as a negative activation gate")
    freeze_json(controller.root / "SUMMARY.json", signed({
        "schema_version": "phiagent-chisel-component-summary/v1", "status": "PARTIAL",
        "reason": "NEGATIVE_ACTIVATION_GATE_ALL_PRIMARY_ROUTES_ALIAS_DELETE",
        "benchmark_quality_result": False, "primary_family": None,
        "activation_gate_sha256": gate["canonical_sha256"],
        "actual_campaign_cost": controller.ledger.costs(), "route_diagnostics": all_routes,
    }))


def selected_arm(policy: str, task: str, routes: dict) -> str:
    if policy in PRIMITIVES:
        return policy
    if policy == "crossfit_equal_count":
        policy = "chisel_oof"
    if policy not in routes["override_task_ids"]:
        raise ProtocolError("unknown frozen policy")
    if task not in routes["override_task_ids"][policy]:
        return "delete"
    return "neutral_modify" if policy.startswith("residual_placebo_") else "modify"


def check_prompt_pairs(ledger: Ledger, model: str, split: str, seed: int,
                       tasks: set[str]) -> None:
    modify, neutral = (ledger.rows(model, split, seed, arm)
                       for arm in ("modify", "neutral_modify"))
    if not tasks <= modify.keys() or not tasks <= neutral.keys():
        raise ProtocolError("paired neutral/modify coverage differs")
    for task in tasks:
        if modify[task]["usage"]["prompt_tokens"] != neutral[task]["usage"]["prompt_tokens"]:
            raise ProtocolError("actual MODIFY/neutral prompt tokens are not exactly matched")


def run_gate(controller: Controller, public: dict, protocol: dict) -> None:
    path = controller.root / "gate.json"
    config = controller.config
    arms = config["execution"]["quality_blind_workflow_gate_arms"]
    gate_tasks = protocol["gate_task_ids"]
    if (arms != [*TRAIN_ARMS, "neutral_modify"]
            or config["execution"]["quality_blind_workflow_gate_training_tasks"] != 12
            or len(gate_tasks) != 12 or len(set(gate_tasks)) != 12
            or gate_tasks != [item["id"] for item in public["train"][:12]]):
        raise ProtocolError("gate requires the frozen first 12 tasks and all four declared arms")
    seed = config["sampling"]["training_seed"]

    def validate_gate_coverage() -> None:
        for model in MODELS:
            for arm in arms:
                observed = set(controller.ledger.rows(model, "train", seed, arm))
                if not set(gate_tasks) <= observed:
                    raise ProtocolError("complete four-arm gate coverage is required before escalation")
                if arm == "neutral_modify" and observed != set(gate_tasks):
                    raise ProtocolError("neutral training calls must be confined to the first-12 gate")
            check_prompt_pairs(controller.ledger, model, "train", seed, set(gate_tasks))

    if path.exists():
        receipt = self_checked(path, "phiagent-chisel-quality-blind-gate/v1")
        if (receipt["status"] != "PASS" or receipt["task_ids"] != gate_tasks
                or receipt["gate_arms"] != arms):
            raise ProtocolError("persisted gate did not pass for the frozen training tasks")
        validate_gate_coverage()
        return
    controller.matrix(public, "train", arms, [seed], set(gate_tasks))
    validate_gate_coverage()
    runtimes = {}
    for model in MODELS:
        elapsed = []
        for arm in arms:
            rows = controller.ledger.rows(model, "train", seed, arm)
            elapsed.extend(rows[task]["elapsed_seconds"] for task in protocol["gate_task_ids"])
        planned = (3 * (len(public["train"]) + len(public["calibration"]))
                   + len(PRIMITIVES) * len(public["evaluation"])
                   * len(config["sampling"]["evaluation_seeds"]))
        estimate = sum(elapsed) / len(elapsed) * planned / config["execution"]["workers_per_model"]
        runtimes[model] = {"mean_call_seconds": sum(elapsed) / len(elapsed),
                           "projected_total_seconds_at_declared_concurrency": estimate,
                           "planned_matrix_calls": planned}
    receipt = signed({
        "schema_version": "phiagent-chisel-quality-blind-gate/v1",
        "status": "PASS", "task_ids": gate_tasks, "models": runtimes,
        "gate_arms": arms, "training_matrix_arms": list(TRAIN_ARMS),
        "gate_actual_calls": len(MODELS) * len(gate_tasks) * len(arms),
        "gate_only_neutral_calls": len(MODELS) * len(gate_tasks),
        "reusable_training_calls": len(MODELS) * len(gate_tasks) * len(TRAIN_ARMS),
        "actual_paired_prompt_token_equality_verified": True,
        "quality_outcomes_read": False, "neutral_cells_charged_separately": True,
        "prediction_is_not_a_throughput_guarantee": True,
    })
    if max(row["projected_total_seconds_at_declared_concurrency"] for row in runtimes.values()) > (
        controller.deadline - time.time()
    ):
        receipt = signed({**{key: value for key, value in receipt.items()
                            if key != "canonical_sha256"}, "status": "BLOCKED_RUNTIME_BUDGET"})
        freeze_json(path, receipt)
        raise ProtocolError("quality-blind gate projects beyond the frozen lifetime budget")
    freeze_json(path, receipt)


def calibration_audit(controller: Controller, public: dict, data: dict) -> None:
    controller.set_phase("calibration-audit-no-retuning")
    answers = load_answers(data, "calibration", public["calibration"])
    scorer = official_scorer(data, runtime=controller.scorer_runtime)
    audits = {}
    for model in MODELS:
        scores = {arm: score_rows(controller.ledger.rows(
            model, "calibration", controller.config["sampling"]["training_seed"], arm,
        ), answers, scorer) for arm in TRAIN_ARMS}
        for seed in controller.config["routing"]["fold_seeds"]:
            routes = read_json(routes_path(controller.root, model, "calibration", seed))
            for policy in (*TRAIN_ARMS, "chisel_oof", "chisel_in_sample",
                           "matched_random_oof", "matched_random_in_sample"):
                rows = [scores[selected_arm(policy, item["id"], routes)][item["id"]]
                        for item in public["calibration"]]
                parents = [scores["skillopt"][item["id"]] for item in public["calibration"]]
                audits[f"{model}/{seed}/{policy}"] = {
                    "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
                    "f1": sum(row["f1"] for row in rows) / len(rows),
                    "selected_tokens": sum(row["tokens"] for row in rows),
                    "harmful_rate_vs_skillopt": sum(
                        not row["exact_match"] and parent["exact_match"]
                        for row, parent in zip(rows, parents)
                    ) / len(rows),
                    "mean_normalized_token_delta_vs_skillopt": sum(
                        max(-1.0, min(1.0, (row["tokens"] - parent["tokens"]) / parent["tokens"]))
                        for row, parent in zip(rows, parents)
                    ) / len(rows),
                }
    freeze_json(controller.root / "calibration-audit.json", {
        "status": "AUDIT_ONLY_NO_RETUNING_OR_QUALITY_GATE", "audits": audits,
        "neutral_placebo_not_generated_in_calibration": True,
    })


def official_source_command(data: dict, predictions: Path) -> list[str]:
    scorer = data["scorer"]
    declared = scorer["cli"]
    if (len(declared) != 4 or declared[1:] != [
        scorer["path"], data["raw_archives"]["evaluation"]["path"], "predictions.json",
    ] or scorer["scale"] != "percentage"):
        raise ProtocolError("official scorer CLI contract differs from the worker receipt")
    return [*declared[:3], str(predictions)]


def official_score_cli(args: argparse.Namespace) -> int:
    """Run the unchanged official __main__, with only the frozen path-loader shim."""
    remote_guard()
    root = args.experiment_dir.resolve()
    protocol, config, data, _services = load_frozen(root)
    mode = protocol["official_scorer_runtime"]["mode"]
    if mode != "cached-path-local-only-v1":
        raise ProtocolError("compatibility CLI requires explicit prospective runtime selection")
    if (root / "STOP.json").exists() or not (root / "GENERATION_DRAINED.json").exists():
        raise ProtocolError("official CLI requires a complete, non-stopped generation handoff")
    handoff = self_checked(root / "GENERATION_DRAINED.json", "phiagent-chisel-generation-drained/v1")
    if (handoff["reason"] != "FULL_EVALUATION_MATRIX_COLLECTED"
            or handoff["protocol_sha256"] != protocol["canonical_sha256"]
            or handoff["service_manifest_sha256"] != protocol["service_manifest_sha256"]
            or handoff["progress"]["inflight_or_unmatched"] != 0
            or handoff["progress"]["successful_cells"] != handoff["progress"]["intents"]):
        raise ProtocolError("official CLI requires the complete evaluation generation boundary")
    complete = read_json(root / "evaluation-responses-complete.json")
    if (complete["tasks"] != config["track"]["evaluation_tasks"]
            or complete["seeds"] != config["sampling"]["evaluation_seeds"]
            or complete["arms"] != list(PRIMITIVES)
            or complete["status"] !=
            "ALL_PRIMITIVES_MODELS_SEEDS_COMPLETE_BEFORE_EVALUATION_LABELS"):
        raise ProtocolError("evaluation responses are not complete for the frozen protocol")
    load_saved_routes(root, config)
    predictions = args.predictions.resolve()
    if not predictions.is_relative_to((root / "predictions").resolve()):
        raise ProtocolError("official predictions must remain inside this frozen experiment")
    predicted = read_json(predictions)
    evaluation = data["splits"]["evaluation"]
    if (not isinstance(predicted, dict) or len(predicted) != evaluation["task_count"]
            or any(not isinstance(value, str) for value in predicted.values())
            or canonical_hash(sorted(predicted)) != evaluation["task_ids_canonical_sha256"]):
        raise ProtocolError("official CLI requires predictions for every original development qid")
    source = Path(data["scorer"]["path"])
    dataset = Path(data["raw_archives"]["evaluation"]["path"])
    if (file_sha256(source) != data["scorer"]["sha256"]
            or file_sha256(dataset) != data["raw_archives"]["evaluation"]["sha256"]):
        raise ProtocolError("official CLI source or complete development archive changed")
    command = official_source_command(data, predictions)
    previous_argv = sys.argv
    successful = False
    with scorer_import_runtime(mode, allowed_paths=(dataset,)) as runtime_receipt:
        try:
            sys.argv = command[1:]
            try:
                runpy.run_path(str(source), run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
            successful = True
        finally:
            sys.argv = previous_argv
            evidence = {
                "status": "OFFICIAL_CLI_COMPLETED" if successful else "OFFICIAL_CLI_FAILED",
                "runtime": runtime_receipt, "official_source_command": command,
                "executed_python": sys.executable, "python_version": sys.version,
                "official_source_sha256": file_sha256(source),
                "wrapper_source_sha256": file_sha256(Path(__file__).resolve()),
                "predictions_sha256": file_sha256(predictions),
                "protocol_sha256": protocol["canonical_sha256"],
                "official_source_unmodified": True,
                "error": None if successful else str(sys.exc_info()[1]),
            }
            freeze_json(predictions.with_suffix(".scorer-runtime.json"), signed(evidence))
    return 0


def official_cli_check(root: Path, data: dict, predictions: Path, expected: dict,
                       *, runtime: str = "native") -> None:
    remote_guard()
    scorer = data["scorer"]
    if file_sha256(Path(scorer["path"])) != scorer["sha256"]:
        raise ProtocolError("official scorer changed before independent CLI verification")
    dataset = data["raw_archives"]["evaluation"]
    if file_sha256(Path(dataset["path"])) != dataset["sha256"]:
        raise ProtocolError("complete official dev archive changed")
    source_command = official_source_command(data, predictions)
    if runtime == "native":
        command = source_command
    elif runtime == "cached-path-local-only-v1":
        command = [
            source_command[0], str(Path(__file__).resolve()), "official-score-cli",
            "--experiment-dir", str(root), "--predictions", str(predictions),
        ]
    else:
        raise ProtocolError("unknown official CLI runtime; no automatic fallback is allowed")
    process = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    output = {"command": command, "returncode": process.returncode,
              "stdout": process.stdout, "stderr": process.stderr,
              "runtime": runtime, "official_source_command": source_command,
              "official_source_sha256": scorer["sha256"]}
    freeze_json(predictions.with_suffix(".official-cli.json"), output)
    if process.returncode:
        raise ProtocolError("independent official scorer CLI failed")
    if runtime == "cached-path-local-only-v1":
        receipt = read_json(predictions.with_suffix(".scorer-runtime.json"))
        verify_self_hash(receipt, label="official scorer compatibility runtime")
        if (receipt["status"] != "OFFICIAL_CLI_COMPLETED"
                or receipt["runtime"]["mode"] != runtime
                or receipt["runtime"]["official_metric_functions_modified"] is not False
                or receipt["official_source_sha256"] != scorer["sha256"]
                or receipt["official_source_command"] != source_command
                or receipt["predictions_sha256"] != file_sha256(predictions)
                or receipt["official_source_unmodified"] is not True
                or any(access["resolved_path"] != str(Path(dataset["path"]).resolve())
                       or access["sha256"] != dataset["sha256"]
                       for access in receipt["runtime"]["cached_path_accesses"])):
            raise ProtocolError("official scorer compatibility provenance differs")
    observed = json.loads(process.stdout)
    if any(observed[metric] != expected[metric] for metric in ("exact_match", "f1")):
        raise ProtocolError("independent official scorer aggregates do not agree exactly")


def component_quality_decisions(contrasts: dict, config: dict) -> dict:
    metrics = config["track"]["metrics"]
    expected = {
        f"{model}/{contrast}/{metric}"
        for model in config["models"]
        for contrast in config["primary_contrasts"]
        for metric in metrics
    }
    if (set(contrasts) != expected
            or len(expected) != config["statistics"]["primary_holm_family_size"]):
        raise ProtocolError("component decisions require the complete preregistered primary family")
    if any("holm_adjusted_p" not in row
           or type(row.get("positive_quality_contribution_supported")) is not bool
           for row in contrasts.values()):
        raise ProtocolError("component decisions require completed family-wide Holm adjustment")
    required_contrasts = {
        "residual_editing": ("residual_editing", "residual_editing_vs_delete"),
        "local_routing": ("local_routing",),
        "cross_fitting": ("cross_fitting",),
    }
    per_model = {}
    for model in config["models"]:
        per_model[model] = {}
        for component, names in required_contrasts.items():
            required = [f"{model}/{name}/{metric}" for name in names for metric in metrics]
            if not set(required) <= expected:
                raise ProtocolError("required component control was removed from the primary family")
            failed = [key for key in required
                      if not contrasts[key]["positive_quality_contribution_supported"]]
            per_model[model][component] = {
                "supported": not failed,
                "required_primary_tests": required,
                "failed_primary_tests": failed,
                "both_official_metrics_required": True,
            }
    return {
        "per_model": per_model,
        "supported_on_all_declared_models": {
            component: all(per_model[model][component]["supported"] for model in config["models"])
            for component in required_contrasts
        },
        "residual_editing_rule": (
            "Must beat BOTH length-matched placebo and unmodified DELETE on BOTH official metrics, "
            "with positive confidence bounds and complete-family Holm adjustment."
        ),
        "boundary": "Frozen research component evidence only; no innovation or SOTA claim is admitted.",
    }


def official_aggregate_scores(scorer: Any, public_items: list[dict],
                             answers: dict, selected: dict) -> dict:
    task_ids = [item["id"] for item in public_items]
    if (not task_ids or len(set(task_ids)) != len(task_ids)
            or set(task_ids) != set(answers) or set(task_ids) != set(selected)):
        raise ProtocolError("official aggregate scoring requires the complete paired cohort")
    # Match raw-archive answer order and canonical prediction-file order exactly.
    ordered_answers = {task: answers[task] for task in task_ids}
    predictions = {task: selected[task]["prediction"] for task in sorted(task_ids)}
    official = scorer.evaluate(ordered_answers, predictions, skip_no_answer=False)
    if any(type(official[metric]) not in (int, float) or not 0 <= official[metric] <= 100
           for metric in ("exact_match", "f1")):
        raise ProtocolError("official aggregate metrics must be finite percentages")
    return {metric: official[metric] for metric in ("exact_match", "f1")}


def analyze(controller: Controller, public: dict, data: dict, all_routes: dict) -> dict:
    controller.set_phase("complete-evaluation-official-analysis")
    config, root = controller.config, controller.root
    answers = load_answers(data, "evaluation", public["evaluation"])
    scorer = official_scorer(data, runtime=controller.scorer_runtime)
    seeds = config["sampling"]["evaluation_seeds"]
    groups = {row["id"]: row["group"] for row in public["evaluation"]}
    contrasts, table = {}, []
    policies = (*PRIMITIVES, "chisel_oof", "chisel_in_sample", "residual_placebo_oof",
                "residual_placebo_in_sample", "matched_random_oof", "matched_random_in_sample",
                "crossfit_equal_count")
    for model in MODELS:
        metrics = {policy: {metric: {} for metric in ("exact_match", "f1")} for policy in policies}
        for seed in seeds:
            controller.check()
            check_prompt_pairs(controller.ledger, model, "evaluation", seed, set(groups))
            scores = {arm: score_rows(controller.ledger.rows(model, "evaluation", seed, arm),
                                      answers, scorer) for arm in PRIMITIVES}
            freeze_json(root / "scores" / model_key(model) / f"{seed}.json", scores)
            routes = all_routes[f"{model}/evaluation/{seed}"]
            # Sets keep policy reconstruction linear in full-cohort size.
            route_sets = {**routes, "override_task_ids": {
                key: set(value) for key, value in routes["override_task_ids"].items()
            }}
            for policy in policies:
                selected = {task: scores[selected_arm(policy, task, route_sets)][task] for task in groups}
                for metric in ("exact_match", "f1"):
                    metrics[policy][metric][seed] = {task: row[metric] for task, row in selected.items()}
                aggregates = official_aggregate_scores(scorer, public["evaluation"], answers, selected)
                path = root / "predictions" / model_key(model) / str(seed) / f"{policy}.json"
                freeze_json(path, {task: row["prediction"] for task, row in selected.items()})
                controller.check()
                official_cli_check(root, data, path, aggregates, runtime=controller.scorer_runtime)
                table.append({
                    "model": model, "seed": seed, "policy": policy, **aggregates,
                    "tasks": len(groups), "selected_tokens": sum(row["tokens"] for row in selected.values()),
                    "selected_prompt_tokens": sum(row["prompt_tokens"] for row in selected.values()),
                    "selected_completion_tokens": sum(row["completion_tokens"] for row in selected.values()),
                    "empty_answers": sum(row["empty_answer"] for row in selected.values()),
                    "length_finishes": sum(row["finish_reason"] == "length" for row in selected.values()),
                })
        for contrast, (left, right) in config["primary_contrasts"].items():
            for metric in ("exact_match", "f1"):
                controller.check()
                result = paired_quality_effect(
                    metrics[left][metric], metrics[right][metric], groups, seeds=seeds,
                    bootstrap_samples=config["statistics"]["paired_group_bootstrap_samples"],
                    randomization_samples=config["statistics"]["paired_group_sign_randomization_samples"],
                    statistics_seed=config["statistics"]["statistics_seed"],
                )
                contrasts[f"{model}/{contrast}/{metric}"] = {
                    **result, "model": model, "contrast": contrast, "metric": metric,
                    "candidate": left, "reference": right,
                }
    adjust_primary_family(contrasts, expected_size=config["statistics"]["primary_holm_family_size"],
                          alpha=config["statistics"]["alpha"])
    component_decisions = component_quality_decisions(contrasts, config)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    freeze_bytes(root / "quality.csv", output.getvalue().encode("utf-8"))
    summary = signed({
        "schema_version": "phiagent-chisel-component-summary/v1", "status": "COMPLETE",
        "dataset_id": data["dataset_id"], "official_metrics": ["exact_match", "f1"],
        "full_evaluation_tasks": len(groups), "all_fixed_seeds": seeds,
        "claim_boundary": config["claim_boundary"], "fresh_confirmation": False,
        "actual_campaign_cost": controller.ledger.costs(), "per_seed_policies": table,
        "primary_family": contrasts, "component_quality_decisions": component_decisions,
        "route_diagnostics": {
            key: {"diagnostics": routes["diagnostics"],
                  "oof_in_sample_route_disagreements": routes["oof_in_sample_route_disagreements"],
                  "C_equal_count_structural_null": routes["crossfit_equal_override_count_control"]}
            for key, routes in all_routes.items()
        },
        "calibration_is_not_deployment_certification": True,
        "official_cli_exact_aggregate_agreement": True,
        "official_scorer_runtime": controller.scorer_runtime,
    })
    freeze_json(root / "SUMMARY.json", summary)
    return summary


def verify_ledger(controller: Controller, public: dict) -> None:
    """Reconstruct request bytes and recheck every raw model identity and token receipt."""
    controller.ledger.recovery_check()
    by_id = {split: {item["id"]: item for item in rows} for split, rows in public.items()}
    with controller.ledger.lock:
        cursor = controller.ledger.db.execute("""
            SELECT cell, model, split, task, seed, arm, request_sha256, prompt_tokens,
                   padding_units, endpoint, raw, raw_sha256, metadata
            FROM intents JOIN results USING(cell) ORDER BY intents.rowid
        """)
    for row in cursor:
        cell_id, model, split, task, seed, arm, digest, count, padding, endpoint, raw, raw_hash, text = row
        controller.check()
        cell = cell_spec(model, split, by_id[split][task], seed, arm,
                         controller.models[model]["base_url"])
        if cell["cell"] != cell_id or cell["endpoint"] != endpoint:
            raise ProtocolError("intent identity changed")
        request, actual_count, actual_padding = controller.request(cell, by_id[split][task])
        if (hashlib.sha256(request).hexdigest() != digest or actual_count != count
                or actual_padding != padding or hashlib.sha256(raw).hexdigest() != raw_hash):
            raise ProtocolError("raw response or reconstructed request hash differs")
        identity = strict_response(json.loads(raw),
                                   model=controller.models[model]["served_model"],
                                   expected_prompt_tokens=count)
        metadata = json.loads(text)
        if any(metadata[key] != value for key, value in identity.items()):
            raise ProtocolError("result identity or usage metadata changed")


def verify_planned_coverage(controller: Controller, public: dict, protocol: dict) -> None:
    """No extra seeds, tasks, arms, or generation attempts can enter a release."""
    training_seed = controller.config["sampling"]["training_seed"]
    seeds = controller.config["sampling"]["evaluation_seeds"]
    expected = {}
    for model in MODELS:
        for split in ("train", "calibration", "evaluation"):
            for seed in seeds if split == "evaluation" else [training_seed]:
                for arm in PRIMITIVES if split == "evaluation" else TRAIN_ARMS:
                    expected[(model, split, seed, arm)] = len(public[split])
        expected[(model, "train", training_seed, "neutral_modify")] = len(protocol["gate_task_ids"])
    with controller.ledger.lock:
        observed = {tuple(row[:4]): row[4] for row in controller.ledger.db.execute(
            "SELECT model, split, seed, arm, COUNT(*) FROM intents GROUP BY model, split, seed, arm",
        )}
    if observed != expected:
        raise ProtocolError("campaign has missing or unplanned generation cells")
    for model in MODELS:
        if set(controller.ledger.rows(model, "train", training_seed, "neutral_modify")) != set(
            protocol["gate_task_ids"]
        ):
            raise ProtocolError("neutral training calls were not confined to the predeclared gate")


@contextmanager
def lease(root: Path):
    with (root / "lease.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProtocolError("another controller holds the experiment lease") from error
        record = {"hostname": socket.gethostname(), "pid": os.getpid(), "acquired_at": now()}
        freeze_json(root / "attempts" / f"lease-{time.time_ns()}.json", record)
        atomic_json(root / "lease.json", record)
        try:
            yield
        finally:
            atomic_json(root / "lease.json", {**record, "released_at": now()})
            fcntl.flock(stream, fcntl.LOCK_UN)


def release(root: Path) -> None:
    excluded = {"RELEASE.json", "state.json", "heartbeat.json", "lease.json", "lease.lock"}
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*"))
              if path.is_file() and path.name not in excluded
              and not path.name.endswith(("-wal", "-shm", ".pending"))}
    data = read_json(root / "data-manifest.json")
    external = {data["scorer"]["path"]: data["scorer"]["sha256"]}
    for record in data["raw_archives"].values():
        external[record["path"]] = record["sha256"]
    for record in data["splits"].values():
        external[record["public_path"]] = record["public_sha256"]
        external[record["answers_path"]] = record["answers_sha256"]
    for path, expected in external.items():
        if file_sha256(Path(path)) != expected:
            raise ProtocolError(f"external immutable data/scorer artifact changed: {path}")
    freeze_json(root / "RELEASE.json", signed({
        "schema_version": "phiagent-chisel-component-release/v1",
        "status": "COMPLETE_VERIFIED", "artifacts": hashes, "external_artifacts": external,
    }))


def verify(args: argparse.Namespace) -> int:
    remote_guard()
    root = args.experiment_dir.resolve()
    load_frozen(root)
    if not (root / "RELEASE.json").exists():
        raise ProtocolError("PARTIAL/BLOCKED: no verified complete-family release exists")
    receipt = self_checked(root / "RELEASE.json", "phiagent-chisel-component-release/v1")
    if (root / "STOP.json").exists():
        raise ProtocolError("stopped campaign cannot be a complete release")
    for relative, expected in receipt["artifacts"].items():
        if file_sha256(root / relative) != expected:
            raise ProtocolError(f"release artifact changed: {relative}")
    for path, expected in receipt["external_artifacts"].items():
        if file_sha256(Path(path)) != expected:
            raise ProtocolError(f"external release artifact changed: {path}")
    summary = self_checked(root / "SUMMARY.json", "phiagent-chisel-component-summary/v1")
    if summary["status"] != "COMPLETE":
        raise ProtocolError("incomplete primary family is not a release")
    print(canonical_json({"status": "COMPLETE_VERIFIED", "release_sha256": receipt["canonical_sha256"]}))
    return 0


def run(args: argparse.Namespace) -> int:
    remote_guard()
    root = args.experiment_dir.resolve()
    protocol, config, data, services = load_frozen(root)
    if args.authorize_protocol != protocol["canonical_sha256"]:
        raise ProtocolError("explicit parent authorization of this exact protocol is required")
    if (root / "RELEASE.json").exists():
        return verify(args)
    if (root / "STOP.json").exists() or (root / "FAILURE.json").exists():
        raise ProtocolError("failed experiment is immutable; no in-place retries")
    with lease(root):
        freeze_json(root / "attempts" / f"command-{time.time_ns()}.json", {
            "at": now(), "hostname": socket.gethostname(), "argv": sys.argv,
            "authorized_protocol": args.authorize_protocol,
            "package_versions": package_versions(),
        })
        ledger = Ledger(root / "ledger.sqlite3")
        state = read_json(root / "state.json")
        if "deadline" not in state:
            state["started_at"] = time.time()
            state["deadline"] = state["started_at"] + config["execution"]["maximum_seconds"]
            atomic_json(root / "state.json", state)
        controller = Controller(
            root, config, services, ledger, state["deadline"],
            scorer_runtime=protocol["official_scorer_runtime"]["mode"],
        )
        monitor = None
        completed = False

        def interrupted(signum, frame):
            controller.stop("controller signal", signal=signum)
            raise ProtocolError(f"controller interrupted by signal {signum}")

        old_handlers = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        previous_alarm = signal.setitimer(signal.ITIMER_REAL, max(0.001, state["deadline"] - time.time()))
        try:
            controller.check()
            ledger.recovery_check()
            handoff = controller.restore_generation_handoff(protocol)
            if handoff is None:
                service_preflight(root, services)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                for model in MODELS:
                    record = controller.models[model]
                    verify_file_records(tokenizer_file_records(record), manifest_path(record, "tokenizer_path"))
            official_scorer(data, runtime=controller.scorer_runtime)
            freeze_json(root / "attempts" / f"scorer-import-{time.time_ns()}.json", signed({
                "status": "PINNED_SOURCE_IMPORT_ONLY_NO_EVALUATION",
                "at": now(), "hostname": socket.gethostname(), "python": sys.executable,
                "runtime": controller.scorer_runtime, "protocol_sha256": protocol["canonical_sha256"],
                "official_source_sha256": data["scorer"]["sha256"],
                "official_metric_functions_modified": False,
                "benchmark_labels_read": False, "metric_functions_called": False,
            }))
            for model in MODELS:
                controller.tokenizer(model)
            public = load_public(data, config)
            controller.audit()
            monitor = threading.Thread(target=controller.monitor, name="gpu-heartbeat", daemon=True)
            monitor.start()
            if handoff is None:
                run_gate(controller, public, protocol)
                controller.matrix(public, "train", TRAIN_ARMS, [config["sampling"]["training_seed"]])
                freeze_json(root / "training-responses-complete.json", {
                    "status": "BOTH_MODELS_COMPLETE_BEFORE_OFFICIAL_TRAIN_SCORING",
                    "training_tasks": len(public["train"]), "arms": list(TRAIN_ARMS),
                })
                all_routes = fit_and_freeze(controller, public, data)
            else:
                all_routes = load_saved_routes(root, config)
            activation = activation_gate(all_routes, config)
            freeze_json(root / "activation-gate.json", activation)
            if activation["stop_before_calibration_and_evaluation_generation"]:
                if handoff is None:
                    controller.close_generation(protocol, reason="NEGATIVE_ACTIVATION_GATE")
                elif handoff["reason"] != "NEGATIVE_ACTIVATION_GATE":
                    raise ProtocolError("persisted generation boundary disagrees with activation gate")
                write_negative_activation_summary(controller, all_routes, activation)
                state.update(status="PARTIAL", phase="negative-activation-gate")
                atomic_json(root / "state.json", state)
                completed = True
                print(canonical_json({"status": "PARTIAL", "reason": "NEGATIVE_ACTIVATION_GATE",
                                      "benchmark_quality_result": False}))
                return 0
            if handoff is None:
                controller.matrix(public, "calibration", TRAIN_ARMS, [config["sampling"]["training_seed"]])
                calibration_audit(controller, public, data)
                controller.matrix(public, "evaluation", PRIMITIVES, config["sampling"]["evaluation_seeds"])
                freeze_json(root / "evaluation-responses-complete.json", {
                    "status": "ALL_PRIMITIVES_MODELS_SEEDS_COMPLETE_BEFORE_EVALUATION_LABELS",
                    "tasks": len(public["evaluation"]), "seeds": config["sampling"]["evaluation_seeds"],
                    "arms": list(PRIMITIVES),
                })
            elif handoff["reason"] != "FULL_EVALUATION_MATRIX_COLLECTED":
                raise ProtocolError("persisted generation boundary disagrees with activation gate")
            verify_planned_coverage(controller, public, protocol)
            if handoff is None:
                controller.close_generation(protocol, reason="FULL_EVALUATION_MATRIX_COLLECTED")
            verify_ledger(controller, public)
            analyze(controller, public, data, all_routes)
            controller.check()
            completed = True
            state.update(status="COMPLETE", phase="verified-release")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            controller.finished.set()
            if monitor is not None:
                monitor.join(timeout=HTTP_TIMEOUT)
            if not completed:
                controller.stop("controller failed", error=str(sys.exc_info()[1]))
                state.update(status="BLOCKED", phase=controller.phase, cost=ledger.costs())
            atomic_json(root / "state.json", state)
            ledger.close()
            signal.setitimer(signal.ITIMER_REAL, *previous_alarm)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        try:
            controller.check()
            load_frozen(root)
            release(root)
            return verify(args)
        except (ProtocolError, OSError, ValueError, KeyError, TypeError) as error:
            controller.stop("release verification failed", error=repr(error))
            state.update(status="BLOCKED", phase="release-verification")
            atomic_json(root / "state.json", state)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("init", help="create a new immutable remote experiment")
    prepare.add_argument("--experiment-dir", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/chisel_component_quality_mrqa_v1.json")
    prepare.add_argument("--data-manifest", type=Path, required=True)
    prepare.add_argument("--service-manifest", type=Path, required=True)
    prepare.add_argument("--source-git-state", type=Path,
                         help="Explicit original Git-state receipt for a source-only remote deployment")
    prepare.add_argument("--scorer-runtime", choices=SCORER_RUNTIMES, default="native",
                         help="prospectively bind native dependency or audited local-file loader shim")
    execute = subparsers.add_parser("run", help="bounded automatic phase advancement; never retry failures")
    execute.add_argument("--experiment-dir", type=Path, required=True)
    execute.add_argument("--authorize-protocol", required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--experiment-dir", type=Path, required=True)
    official = subparsers.add_parser("official-score-cli", help="internal sealed-cohort scorer wrapper")
    official.add_argument("--experiment-dir", type=Path, required=True)
    official.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"init": init, "run": run, "verify": verify,
            "official-score-cli": official_score_cli}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
