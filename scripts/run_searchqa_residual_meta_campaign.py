#!/usr/bin/env python3
"""Run a task-disjoint SearchQA residual Meta-Harness campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phiagent.learning.residual_meta_harness import (  # noqa: E402
    FeatureKind,
    FeatureSpec,
    Operation,
    PolicySession,
    SafetyGate,
    build_evidence_document,
    canonical_hash,
    certificate_from_evidence,
    feature_schema_hash,
    head_spec_hash,
    parse_policy,
    policy_to_dict,
)
from phiagent.learning.residual_meta_training import (  # noqa: E402
    fit_cross_fitted_linear_head,
    select_calibration_records,
)
from phiagent.learning.searchqa_grounded_features import (  # noqa: E402
    TOKEN_PATTERN,
    content_terms,
    grounded_anchors,
    named_feature_field,
)


PROTOCOL_SCHEMA = "phiagent-searchqa-residual-meta-campaign/v1"
SERVICE_SCHEMA = "phiagent-searchqa-model-service-pool/v1"
VARIANTS = (
    "no_skill",
    "skillopt",
    "autosaddler",
    "add",
    "delete",
    "modify",
)
ROUTER_SPECS = {
    "delete": {
        "default_operation": Operation.DELETE,
        "candidates": {
            "no_skill": Operation.KEEP,
            "add": Operation.ADD,
            "modify": Operation.MODIFY,
        },
    },
    "skillopt": {
        "default_operation": Operation.KEEP,
        "candidates": {
            "add": Operation.ADD,
            "delete": Operation.DELETE,
            "modify": Operation.MODIFY,
        },
    },
}
SPLIT_SIZES = {
    "discovery": 8_000,
    "calibration": 2_000,
    "confirmation": 5_000,
    "reserve": 1_000,
}
FEATURE_SCHEMA = {
    "context_token_count": FeatureSpec(
        kind=FeatureKind.NUMERIC,
        mean=900.0,
        scale=600.0,
        minimum=0.0,
        maximum=8192.0,
    ),
    "document_count": FeatureSpec(
        kind=FeatureKind.NUMERIC,
        mean=20.0,
        scale=20.0,
        minimum=0.0,
        maximum=8192.0,
    ),
    "question_character_count": FeatureSpec(
        kind=FeatureKind.NUMERIC,
        mean=70.0,
        scale=50.0,
        minimum=0.0,
        maximum=16384.0,
    ),
    "question_context_coverage": FeatureSpec(
        kind=FeatureKind.NUMERIC,
        mean=0.5,
        scale=0.25,
        minimum=0.0,
        maximum=1.0,
    ),
    "question_has_quoted_span": FeatureSpec(kind=FeatureKind.BOOLEAN),
    "question_has_relation_cue": FeatureSpec(kind=FeatureKind.BOOLEAN),
    "question_named_token_count": FeatureSpec(
        kind=FeatureKind.NUMERIC,
        mean=2.0,
        scale=2.0,
        minimum=0.0,
        maximum=4096.0,
    ),
    "question_token_count": FeatureSpec(
        kind=FeatureKind.NUMERIC,
        mean=12.0,
        scale=8.0,
        minimum=0.0,
        maximum=4096.0,
    ),
}
RELATION_CUES = re.compile(
    r"\b(who|whose|where|when|which|what|how|born|died|founded|"
    r"directed|written|wrote|author|capital|located|member|played|"
    r"married|published|released|won)\b",
    re.IGNORECASE,
)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, new: bool = False) -> None:
    if new and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("canonical_sha256", None)
    if declared != canonical_hash(unsigned):
        raise RuntimeError(f"{label} self-hash mismatch")


def truncate_context(
    context: str, max_chars: int = 6000, *, include_truncation_marker: bool = True,
) -> str:
    if len(context) <= max_chars:
        return context
    documents = context.split("[DOC]")
    result = ""
    for document in documents:
        candidate = result + "[DOC]" + document if result else document
        if len(candidate) > max_chars:
            break
        result = candidate
    marker = "\n...[truncated]" if include_truncation_marker else ""
    return result or context[:max_chars] + marker


def build_user(item: Mapping[str, Any]) -> str:
    return (
        f"## Context\n{truncate_context(str(item['context']))}\n\n## Question\n{item['question']}"
    )


def task_features(
    item: Mapping[str, Any], *, named_feature: str = "capitalized",
) -> dict[str, Any]:
    replacement = named_feature_field(named_feature)
    question = str(item["question"])
    context = truncate_context(str(item["context"]))
    question_tokens = TOKEN_PATTERN.findall(question)
    context_tokens = TOKEN_PATTERN.findall(context)
    lowered_context = {token.casefold() for token in context_tokens}
    content_tokens = content_terms(question_tokens)
    coverage = (
        len(content_tokens & lowered_context) / len(content_tokens) if content_tokens else 0.0
    )
    features = {
        "context_token_count": len(context_tokens),
        "document_count": context.count("[DOC]"),
        "question_character_count": len(question),
        "question_context_coverage": coverage,
        "question_has_quoted_span": any(character in question for character in "\"'"),
        "question_has_relation_cue": bool(RELATION_CUES.search(question)),
        "question_named_token_count": sum(token[:1].isupper() for token in question_tokens[1:]),
        "question_token_count": len(question_tokens),
    }
    if named_feature != "capitalized":
        del features["question_named_token_count"]
        if replacement is not None:
            source_context = truncate_context(
                str(item["context"]), include_truncation_marker=False,
            )
            anchors = grounded_anchors(question, source_context)
            features[replacement] = (
                anchors.grounded_count
                if named_feature == "grounded_count"
                else anchors.rarity_mass
            )
    return features


def task_feature_schema(named_feature: str = "capitalized") -> dict[str, FeatureSpec]:
    replacement = named_feature_field(named_feature)
    schema = dict(FEATURE_SCHEMA)
    if named_feature != "capitalized":
        specification = schema.pop("question_named_token_count")
        if replacement is not None:
            schema[replacement] = specification
    return schema


def feature_schema_value() -> dict[str, Any]:
    result = {}
    for name, spec in sorted(FEATURE_SCHEMA.items()):
        if spec.kind is FeatureKind.BOOLEAN:
            result[name] = {"kind": spec.kind.value}
        else:
            result[name] = {
                "kind": spec.kind.value,
                "mean": spec.mean,
                "scale": spec.scale,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
            }
    return result


def feature_extractor_sha256() -> str:
    return canonical_hash(
        {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "feature_schema_sha256": feature_schema_hash(FEATURE_SCHEMA),
            "feature_extractor": "task_features/v1",
        }
    )


def split_sizes_from_args(args: argparse.Namespace) -> dict[str, int]:
    sizes = {
        "discovery": int(args.discovery_tasks),
        "calibration": int(args.calibration_tasks),
        "confirmation": int(args.confirmation_tasks),
        "reserve": int(args.reserve_tasks),
    }
    if any(size < 1 for size in sizes.values()):
        raise ValueError("every SearchQA split must contain at least one task")
    return sizes


def router_spec(
    protocol: Mapping[str, Any],
) -> tuple[str, Operation, dict[str, Operation]]:
    value = protocol["residual_router"]
    default_variant = str(value["static_incumbent"])
    if default_variant not in ROUTER_SPECS:
        raise RuntimeError("protocol has an unsupported static incumbent")
    expected = ROUTER_SPECS[default_variant]
    default_operation = Operation(str(value["static_default_operation"]))
    candidates = {
        str(variant): Operation(str(operation))
        for variant, operation in value["candidate_variants"].items()
    }
    if (
        default_operation is not expected["default_operation"]
        or candidates != expected["candidates"]
    ):
        raise RuntimeError("protocol residual router specification differs")
    return default_variant, default_operation, candidates


def normalize_answer(text: str) -> str:
    lowered = text.casefold()
    without_punctuation = PUNCTUATION_PATTERN.sub(" ", lowered)
    without_articles = ARTICLE_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str:
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def score_response(text: str, answers: Sequence[str]) -> dict[str, Any]:
    prediction = extract_answer(text)
    normalized_prediction = normalize_answer(prediction)
    normalized_answers = [normalize_answer(answer) for answer in answers]
    exact = int(normalized_prediction in normalized_answers)
    prediction_tokens = normalized_prediction.split()
    best_f1 = 0.0
    for answer in normalized_answers:
        answer_tokens = answer.split()
        if not prediction_tokens or not answer_tokens:
            best_f1 = max(best_f1, float(prediction_tokens == answer_tokens))
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
        if not overlap:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return {
        "hard": exact,
        "soft": best_f1,
        "predicted_answer": prediction,
    }


def _git_state() -> dict[str, Any]:
    def capture(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "captured_at_utc": utc_now(),
        "repository": "YuhuaJiang2002/PhiAgent",
        "head": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "tracked_status_sha256": hashlib.sha256(capture("status", "--short").encode()).hexdigest(),
    }


def _load_service_manifest(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise TypeError("service manifest must be an object")
    verify_self_hash(value, label="service manifest")
    if value.get("schema_version") != SERVICE_SCHEMA:
        raise RuntimeError("unsupported service manifest schema")
    required_top_level = {
        "schema_version",
        "status",
        "hostname",
        "request_model_id",
        "expected_response_model_id",
        "repository",
        "revision",
        "endpoints",
        "canonical_sha256",
    }
    if set(value) != required_top_level or value.get("status") != "READY":
        raise RuntimeError("service manifest fields or status differ")
    endpoints = value.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise RuntimeError("service manifest has no endpoints")
    required = {
        "base_url",
        "physical_gpu_index",
        "gpu_uuid",
        "port",
        "service_dir",
        "supervisor_pid",
        "server_pid",
    }
    if any(not isinstance(row, Mapping) or set(row) != required for row in endpoints):
        raise RuntimeError("service endpoint record differs")
    if len({row["base_url"] for row in endpoints}) != len(endpoints):
        raise RuntimeError("service endpoint URLs are duplicated")
    if len({row["gpu_uuid"] for row in endpoints}) != len(endpoints):
        raise RuntimeError("service endpoint GPU UUIDs are duplicated")
    return dict(value)


def freeze(args: argparse.Namespace) -> int:
    output = args.experiment_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    service = _load_service_manifest(args.service_manifest.resolve())
    selected_router_spec = ROUTER_SPECS[args.static_incumbent]
    split_sizes = split_sizes_from_args(args)
    remainder = read_json(args.remainder_ids.resolve())
    if (
        not isinstance(remainder, list)
        or any(not isinstance(item, str) or not item for item in remainder)
        or len(remainder) != len(set(remainder))
    ):
        raise RuntimeError("remainder IDs must be unique non-empty strings")
    required = sum(split_sizes.values())
    if len(remainder) < required:
        raise RuntimeError("remainder partition is too small")
    ordered = sorted(
        remainder,
        key=lambda task_id: (
            hashlib.sha256(f"{args.salt}\0{task_id}".encode()).digest(),
            task_id,
        ),
    )[:required]
    output.mkdir(parents=True)
    cursor = 0
    split_records = {}
    selected_sets = []
    for split, size in split_sizes.items():
        ids = ordered[cursor : cursor + size]
        cursor += size
        selected_sets.append(set(ids))
        path = output / "partitions" / f"{split}-ids.json"
        write_json(path, ids, new=True)
        split_records[split] = {
            "count": size,
            "ids_sha256": file_sha256(path),
            "canonical_id_sha256": canonical_hash(ids),
        }
    if sum(map(len, selected_sets)) != len(set().union(*selected_sets)):
        raise RuntimeError("task partitions overlap")

    prompt_records = {}
    for variant in VARIANTS:
        source = args.prompt_source_dir.resolve() / f"{variant}.system.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / "prompts" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        prompt_records[variant] = {
            "bytes": target.stat().st_size,
            "sha256": file_sha256(target),
        }
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    shutil.copy2(Path(__file__).resolve(), snapshot / Path(__file__).name)
    service_target = snapshot / args.service_manifest.name
    shutil.copy2(args.service_manifest.resolve(), service_target)
    remainder_target = snapshot / args.remainder_ids.name
    shutil.copy2(args.remainder_ids.resolve(), remainder_target)
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output / "packages.freeze.txt").write_text(packages, encoding="utf-8")
    source_state = (
        read_json(args.source_git_state.resolve())
        if args.source_git_state is not None
        else _git_state()
    )
    if not isinstance(source_state, Mapping):
        raise TypeError("source Git state must be an object")
    write_json(output / "git-state.json", source_state, new=True)

    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "FROZEN_BEFORE_LABEL_MATERIALIZATION_OR_MODEL_CALL",
        "frozen_at_utc": utc_now(),
        "benchmark": {
            "name": "SearchQA",
            "source_partition": "public train-source holdout",
            "official_metric": "SearchQA/SQuAD normalized exact match",
            "train_parquet": str(args.train_parquet.resolve()),
            "train_parquet_sha256": file_sha256(args.train_parquet.resolve()),
            "remainder_ids_sha256": file_sha256(args.remainder_ids.resolve()),
            "claim_boundary": (
                "Public train-source task-disjoint holdout; not an official "
                "validation, test, leaderboard, or hidden-test result."
            ),
        },
        "partition": {
            "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
            "selection": "SHA256(salt || NUL || task_id), ascending",
            "splits": split_records,
            "pairwise_task_disjoint": True,
            "independent_unit": "complete SearchQA query",
        },
        "response_matrix": {
            "variants": list(VARIANTS),
            "same_query_same_endpoint_and_seed": True,
            "temperature": 0.0,
            "max_output_tokens": args.max_output_tokens,
            "transport_retries": args.transport_retries,
            "model_id": service["request_model_id"],
            "expected_response_model_id": service["expected_response_model_id"],
            "prompts": prompt_records,
        },
        "residual_router": {
            "candidate_variants": {
                variant: operation.value
                for variant, operation in sorted(selected_router_spec["candidates"].items())
            },
            "static_incumbent": args.static_incumbent,
            "static_default_operation": selected_router_spec["default_operation"].value,
            "folds": 5,
            "minimum_discovery_override_support": 400,
            "minimum_calibration_override_support": 100,
            "maximum_features": 8,
            "maximum_override_heads": 1,
            "router_model_calls": 0,
            "extra_actor_calls": 0,
            "threshold_selection": "out-of-fold discovery predictions only",
            "confirmation_open_rule": "passing task-disjoint safety certificate",
            "calibration_bound_method": args.calibration_bound_method,
            "certificate_reference_variant": (
                args.certificate_reference_variant or args.static_incumbent
            ),
            "full_policy_calibration_evidence": (args.full_policy_calibration_evidence),
            "safety_gate": {
                "minimum_training_tasks": 8_000,
                "minimum_calibration_tasks": (
                    split_sizes["calibration"] if args.full_policy_calibration_evidence else 100
                ),
                "minimum_effective_sample_size": (
                    float(split_sizes["calibration"])
                    if args.full_policy_calibration_evidence
                    else 100.0
                ),
                "minimum_override_support": 100,
                "maximum_harmful_rate_upper": 0.05,
                "minimum_utility_lcb": args.minimum_utility_lcb,
                "maximum_token_delta_upper": 0.0,
                "maximum_extra_actor_calls": 0,
                "maximum_router_model_calls": 0,
            },
        },
        "feature_schema": feature_schema_value(),
        "feature_schema_sha256": feature_schema_hash(FEATURE_SCHEMA),
        "feature_extractor_sha256": feature_extractor_sha256(),
        "service_manifest": service,
        "service_manifest_file_sha256": file_sha256(service_target),
        "execution": {
            "hostname": socket.gethostname(),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "workers": args.workers,
            "seed": args.seed,
            "remote_only": True,
        },
        "source_state": source_state,
    }
    protocol["canonical_sha256"] = canonical_hash(protocol)
    write_json(output / "protocol.json", protocol, new=True)
    (output / "status.txt").write_text("FROZEN\n", encoding="utf-8")
    print(
        json.dumps(
            {"experiment_dir": str(output), "protocol_sha256": protocol["canonical_sha256"]},
            sort_keys=True,
        )
    )
    return 0


def load_protocol(root: Path) -> dict[str, Any]:
    value = read_json(root / "protocol.json")
    if not isinstance(value, Mapping):
        raise TypeError("protocol must be an object")
    verify_self_hash(value, label="protocol")
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise RuntimeError("unsupported protocol schema")
    if file_sha256(Path(__file__).resolve()) != value["execution"]["runner_sha256"]:
        raise RuntimeError("runner differs from frozen source SHA-256")
    for variant, record in value["response_matrix"]["prompts"].items():
        if file_sha256(root / "prompts" / f"{variant}.system.txt") != record["sha256"]:
            raise RuntimeError(f"frozen prompt differs for {variant}")
    return dict(value)


def materialize(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    protocol = load_protocol(root)
    train = args.train_parquet.resolve()
    if file_sha256(train) != protocol["benchmark"]["train_parquet_sha256"]:
        raise RuntimeError("SearchQA train Parquet differs from frozen source")
    if (root / "materialization.json").exists():
        raise FileExistsError(root / "materialization.json")
    import pyarrow.parquet as pq

    ids_by_split = {
        split: read_json(root / "partitions" / f"{split}-ids.json")
        for split in protocol["partition"]["splits"]
    }
    wanted = set().union(*(set(ids) for ids in ids_by_split.values()))
    selected: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(train)
    for batch in parquet.iter_batches(
        batch_size=2048,
        columns=["key", "question", "context", "answers"],
    ):
        columns = batch.to_pydict()
        for index, raw_id in enumerate(columns["key"]):
            task_id = str(raw_id)
            if task_id not in wanted:
                continue
            answers = columns["answers"][index]
            if isinstance(answers, Mapping):
                answers = answers.get("text")
            selected[task_id] = {
                "id": task_id,
                "question": str(columns["question"][index]),
                "context": str(columns["context"][index]),
                "answers": [str(answer) for answer in answers],
            }
    if set(selected) != wanted:
        missing = sorted(wanted - set(selected))
        raise RuntimeError(f"materialization lacks {len(missing)} frozen tasks")
    split_records = {}
    for split, ids in ids_by_split.items():
        public_path = root / "public" / f"{split}.json"
        label_path = root / "sealed_labels" / f"{split}.json"
        write_json(
            public_path,
            [
                {
                    "id": task_id,
                    "question": selected[task_id]["question"],
                    "context": selected[task_id]["context"],
                }
                for task_id in ids
            ],
            new=True,
        )
        write_json(
            label_path,
            [{"id": task_id, "answers": selected[task_id]["answers"]} for task_id in ids],
            new=True,
        )
        split_records[split] = {
            "count": len(ids),
            "public_sha256": file_sha256(public_path),
            "sealed_labels_sha256": file_sha256(label_path),
        }
    materialization = {
        "schema_version": "phiagent-searchqa-residual-materialization/v1",
        "status": "MATERIALIZED_AFTER_TASK_ID_FREEZE",
        "materialized_at_utc": utc_now(),
        "protocol_sha256": protocol["canonical_sha256"],
        "train_parquet_sha256": file_sha256(train),
        "splits": split_records,
    }
    materialization["canonical_sha256"] = canonical_hash(materialization)
    write_json(root / "materialization.json", materialization, new=True)
    (root / "status.txt").write_text("MATERIALIZED\n", encoding="utf-8")
    print(json.dumps(materialization, sort_keys=True))
    return 0


def get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310
        value = json.loads(response.read().decode())
    if not isinstance(value, Mapping):
        raise RuntimeError("GET response is not an object")
    return dict(value)


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError("unexpected nvidia-smi inventory row")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
            }
        )
    return rows


def service_preflight(root: Path, protocol: Mapping[str, Any], split: str) -> None:
    service = protocol["service_manifest"]
    if service["hostname"] != socket.gethostname():
        raise RuntimeError("runner host differs from frozen service host")
    endpoints = service["endpoints"]
    inventory = gpu_inventory()
    by_index = {row["index"]: row for row in inventory}
    checks = []
    for endpoint in endpoints:
        index = int(endpoint["physical_gpu_index"])
        if index not in by_index or by_index[index]["uuid"] != endpoint["gpu_uuid"]:
            raise RuntimeError("live physical GPU identity differs from service manifest")
        try:
            os.kill(int(endpoint["supervisor_pid"]), 0)
            os.kill(int(endpoint["server_pid"]), 0)
        except ProcessLookupError as error:
            raise RuntimeError("frozen model-service process is absent") from error
        models = get_json(endpoint["base_url"].rstrip("/") + "/models", timeout=20)
        model_ids = {row.get("id") for row in models.get("data", []) if isinstance(row, Mapping)}
        if service["request_model_id"] not in model_ids:
            raise RuntimeError("live endpoint lacks frozen model ID")
        checks.append(
            {
                "endpoint": endpoint,
                "live_gpu": by_index[index],
                "model_ids": sorted(str(item) for item in model_ids),
            }
        )
    visible = ",".join(str(row["physical_gpu_index"]) for row in endpoints)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    value = {
        "schema_version": "phiagent-searchqa-gpu-preflight/v1",
        "captured_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible,
        "inventory": inventory,
        "checks": checks,
    }
    value["canonical_sha256"] = canonical_hash(value)
    path = root / "runtime" / split / f"gpu-preflight-{int(time.time())}.json"
    write_json(path, value, new=True)


def request_one(
    *,
    endpoint: str,
    model_id: str,
    expected_model_id: str,
    prompt: str,
    item: Mapping[str, Any],
    variant: str,
    endpoint_slot: int,
    seed: int,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": build_user(item)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts = []
    for attempt in range(retries + 1):
        started = time.monotonic()
        request = Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:  # noqa: S310
                raw = response.read().decode()
            value = json.loads(raw)
            response_model = value.get("model")
            if response_model != expected_model_id:
                raise RuntimeError(f"unexpected response model identity: {response_model!r}")
            choice = value["choices"][0]
            content = str(choice["message"]["content"])
            usage = value.get("usage") or {}
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "ok": True,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            return {
                "id": str(item["id"]),
                "variant": variant,
                "ok": True,
                "endpoint": endpoint,
                "endpoint_slot": endpoint_slot,
                "seed": seed,
                "request_sha256": canonical_hash(payload),
                "response_model": response_model,
                "response": content,
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
                "attempts": attempts,
            }
        except HTTPError as error:
            retryable = error.code in {408, 429} or error.code >= 500
            detail = f"HTTPError {error.code}: {error.read().decode(errors='replace')[:1000]}"
        except (
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            RuntimeError,
        ) as error:
            retryable = isinstance(
                error,
                (URLError, TimeoutError, OSError, json.JSONDecodeError),
            )
            detail = f"{type(error).__name__}: {error}"
        attempts.append(
            {
                "attempt": attempt + 1,
                "ok": False,
                "elapsed_seconds": time.monotonic() - started,
                "error": detail,
            }
        )
        if not retryable or attempt == retries:
            return {
                "id": str(item["id"]),
                "variant": variant,
                "ok": False,
                "endpoint": endpoint,
                "endpoint_slot": endpoint_slot,
                "seed": seed,
                "request_sha256": canonical_hash(payload),
                "response": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "attempts": attempts,
                "error": detail,
            }
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def load_latest_responses(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        task_id = row.get("id")
        if not isinstance(task_id, str):
            raise RuntimeError(f"invalid response at {path}:{line_number}")
        rows[task_id] = row
    return rows


def run_matrix(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    protocol = load_protocol(root)
    split = args.split
    if not (root / "materialization.json").is_file():
        raise RuntimeError("task materialization is missing")
    if split == "calibration" and not (root / "selected-candidate.json").is_file():
        raise RuntimeError("calibration is sealed until discovery candidate freeze")
    if split == "confirmation" and not (root / "confirmation-open.json").is_file():
        raise RuntimeError("confirmation is sealed until safety certification passes")
    variants = tuple(args.variants or VARIANTS)
    if not variants or len(variants) != len(set(variants)) or not set(variants) <= set(VARIANTS):
        raise ValueError("run variants are empty, duplicated, or unknown")
    service_preflight(root, protocol, split)
    items = read_json(root / "public" / f"{split}.json")
    endpoints = [row["base_url"] for row in protocol["service_manifest"]["endpoints"]]
    prompts = {
        variant: (root / "prompts" / f"{variant}.system.txt").read_text(encoding="utf-8")
        for variant in variants
    }
    output_paths = {
        variant: root / "responses" / split / f"{variant}.jsonl" for variant in variants
    }
    completed = {variant: load_latest_responses(path) for variant, path in output_paths.items()}
    jobs = []
    for item_index, item in enumerate(items):
        task_id = str(item["id"])
        endpoint_slot = item_index % len(endpoints)
        seed = int(protocol["execution"]["seed"]) + item_index
        rotation = int(
            hashlib.sha256(f"arm-order\0{task_id}".encode()).hexdigest(),
            16,
        ) % len(variants)
        order = (*variants[rotation:], *variants[:rotation])
        for variant in order:
            existing = completed[variant].get(task_id)
            if existing is not None and bool(existing.get("ok")):
                continue
            jobs.append(
                (
                    item,
                    variant,
                    endpoint_slot,
                    seed,
                )
            )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(protocol["execution"]["workers"])
    started = time.monotonic()
    streams = {variant: path.open("a", encoding="utf-8") for variant, path in output_paths.items()}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    request_one,
                    endpoint=endpoints[endpoint_slot],
                    model_id=protocol["service_manifest"]["request_model_id"],
                    expected_model_id=protocol["service_manifest"]["expected_response_model_id"],
                    prompt=prompts[variant],
                    item=item,
                    variant=variant,
                    endpoint_slot=endpoint_slot,
                    seed=seed,
                    max_tokens=int(protocol["response_matrix"]["max_output_tokens"]),
                    retries=int(protocol["response_matrix"]["transport_retries"]),
                ): (variant, str(item["id"]))
                for item, variant, endpoint_slot, seed in jobs
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                variant, task_id = futures[future]
                row = future.result()
                if row["variant"] != variant or row["id"] != task_id:
                    raise RuntimeError("response identity differs from scheduled cell")
                streams[variant].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                streams[variant].flush()
                completed[variant][task_id] = row
                if completed_count % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "completed_this_invocation": completed_count,
                                "scheduled_this_invocation": len(jobs),
                                "elapsed_seconds": time.monotonic() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        for stream in streams.values():
            stream.close()

    summaries = {}
    expected_ids = [str(item["id"]) for item in items]
    for variant in variants:
        rows = [completed[variant].get(task_id) for task_id in expected_ids]
        valid = sum(row is not None and bool(row.get("ok")) for row in rows)
        failed = len(rows) - valid
        usage = {
            key: sum(
                int(row["usage"][key]) for row in rows if row is not None and bool(row.get("ok"))
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        summary = {
            "schema_version": "phiagent-searchqa-residual-response-run/v1",
            "status": "COMPLETE" if not failed else "FAILED_CELLS",
            "split": split,
            "variant": variant,
            "expected_rows": len(rows),
            "valid_rows": valid,
            "failed_rows": failed,
            "usage": usage,
            "response_log_sha256": file_sha256(output_paths[variant]),
            "completed_at_utc": utc_now(),
        }
        summary["canonical_sha256"] = canonical_hash(summary)
        write_json(
            root / "responses" / split / f"{variant}.summary.json",
            summary,
        )
        summaries[variant] = summary
    overall = {
        "split": split,
        "variants": summaries,
        "scheduled_this_invocation": len(jobs),
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(overall, sort_keys=True))
    return int(any(row["status"] != "COMPLETE" for row in summaries.values()))


def _response_map(root: Path, split: str, variant: str) -> dict[str, Any]:
    summary = read_json(root / "responses" / split / f"{variant}.summary.json")
    verify_self_hash(summary, label=f"{split}/{variant} response summary")
    if summary["status"] != "COMPLETE":
        raise RuntimeError(f"{split}/{variant} response matrix is incomplete")
    return load_latest_responses(root / "responses" / split / f"{variant}.jsonl")


def score_matrix(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    load_protocol(root)
    split = args.split
    if split == "calibration" and not (root / "selected-candidate.json").is_file():
        raise RuntimeError("calibration outcomes are sealed before candidate freeze")
    if split == "confirmation" and not (root / "confirmation-open.json").is_file():
        raise RuntimeError("confirmation outcomes are sealed before gate passage")
    output = root / "analysis" / split / "matrix.jsonl"
    if output.exists():
        raise FileExistsError(output)
    public = read_json(root / "public" / f"{split}.json")
    labels = {
        row["id"]: row["answers"] for row in read_json(root / "sealed_labels" / f"{split}.json")
    }
    responses = {variant: _response_map(root, split, variant) for variant in VARIANTS}
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = {variant: {"hard": 0, "soft": 0.0, "tokens": 0} for variant in VARIANTS}
    with output.open("x", encoding="utf-8") as stream:
        for item in public:
            task_id = str(item["id"])
            arms = {}
            for variant in VARIANTS:
                response = responses[variant][task_id]
                if not response["ok"]:
                    raise RuntimeError(f"invalid matrix cell: {task_id}/{variant}")
                metric = score_response(response["response"], labels[task_id])
                total_tokens = int(response["usage"]["total_tokens"])
                arms[variant] = {
                    **metric,
                    "total_tokens": total_tokens,
                    "endpoint_slot": int(response["endpoint_slot"]),
                    "seed": int(response["seed"]),
                    "request_sha256": response["request_sha256"],
                }
                totals[variant]["hard"] += metric["hard"]
                totals[variant]["soft"] += metric["soft"]
                totals[variant]["tokens"] += total_tokens
            row = {
                "id": task_id,
                "features": task_features(item),
                "arms": arms,
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    count = len(public)
    summary = {
        "schema_version": "phiagent-searchqa-residual-score-matrix/v1",
        "status": "COMPLETE",
        "split": split,
        "task_count": count,
        "official_metric": "SearchQA/SQuAD normalized exact match",
        "arms": {
            variant: {
                "successes": totals[variant]["hard"],
                "success_rate": totals[variant]["hard"] / count,
                "mean_f1": totals[variant]["soft"] / count,
                "total_tokens": totals[variant]["tokens"],
                "mean_tokens": totals[variant]["tokens"] / count,
            }
            for variant in VARIANTS
        },
        "matrix_sha256": file_sha256(output),
        "completed_at_utc": utc_now(),
    }
    summary["canonical_sha256"] = canonical_hash(summary)
    write_json(root / "analysis" / split / "summary.json", summary, new=True)
    print(json.dumps(summary, sort_keys=True))
    return 0


def read_matrix(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len({row["id"] for row in rows}):
        raise RuntimeError("score matrix contains duplicate task IDs")
    return rows


def training_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_variant: str,
    default_variant: str,
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        candidate = row["arms"][candidate_variant]
        default = row["arms"][default_variant]
        default_tokens = int(default["total_tokens"])
        token_delta = (int(candidate["total_tokens"]) - default_tokens) / max(default_tokens, 1)
        examples.append(
            {
                "task_id": str(row["id"]),
                "features": dict(row["features"]),
                "utility": int(candidate["hard"]) - int(default["hard"]),
                "harmful": bool(not int(candidate["hard"]) and int(default["hard"])),
                "token_delta": max(-1.0, min(1.0, token_delta)),
            }
        )
    return examples


def fit_candidate(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    protocol = load_protocol(root)
    if (root / "selected-candidate.json").exists():
        raise FileExistsError(root / "selected-candidate.json")
    if (root / "responses" / "calibration").exists():
        raise RuntimeError("calibration responses exist before candidate freeze")
    matrix_path = root / "analysis" / "discovery" / "matrix.jsonl"
    rows = read_matrix(matrix_path)
    if len(rows) != SPLIT_SIZES["discovery"]:
        raise RuntimeError("discovery matrix has the wrong task count")
    default_variant, _, candidate_variants = router_spec(protocol)
    results = {}
    for variant, operation in candidate_variants.items():
        result = fit_cross_fitted_linear_head(
            training_examples(
                rows,
                candidate_variant=variant,
                default_variant=default_variant,
            ),
            feature_schema=FEATURE_SCHEMA,
            folds=5,
            seed=int(protocol["execution"]["seed"]),
            ridge=args.ridge,
            minimum_override_support=400,
        )
        write_json(
            root / "training" / f"{variant}.json",
            result,
            new=True,
        )
        results[variant] = {
            "variant": variant,
            "operation": operation.value,
            "training": result,
        }
    eligible = [
        record
        for record in results.values()
        if record["training"]["status"] == "DISCOVERY_CANDIDATE"
    ]
    selection = {
        "schema_version": "phiagent-searchqa-residual-candidate/v1",
        "status": ("DISCOVERY_CANDIDATE_FROZEN" if eligible else "NO_POSITIVE_DISCOVERY_CANDIDATE"),
        "selected_at_utc": utc_now(),
        "source_matrix_sha256": file_sha256(matrix_path),
        "protocol_sha256": protocol["canonical_sha256"],
        "selection_rule": (
            "maximize OOF mean utility, then minimize harmful rate, token delta, and variant name"
        ),
        "static_incumbent": default_variant,
        "candidate_summaries": {
            variant: {
                "operation": record["operation"],
                **{
                    key: record["training"][key]
                    for key in (
                        "status",
                        "oof_override_support",
                        "oof_mean_utility",
                        "oof_harmful_rate",
                        "oof_mean_token_delta",
                        "training_sha256",
                    )
                },
            }
            for variant, record in sorted(results.items())
        },
        "selected_variant": None,
        "selected_operation": None,
        "training_result_sha256": None,
    }
    if eligible:
        selected = min(
            eligible,
            key=lambda record: (
                -float(record["training"]["oof_mean_utility"]),
                float(record["training"]["oof_harmful_rate"]),
                float(record["training"]["oof_mean_token_delta"]),
                str(record["variant"]),
            ),
        )
        selection.update(
            {
                "selected_variant": selected["variant"],
                "selected_operation": selected["operation"],
                "training_result_sha256": selected["training"]["training_sha256"],
            }
        )
    selection["canonical_sha256"] = canonical_hash(selection)
    write_json(root / "selected-candidate.json", selection, new=True)
    print(json.dumps(selection, sort_keys=True))
    return int(not eligible)


def _gate_value(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(protocol["residual_router"]["safety_gate"])
    SafetyGate(**value)
    return value


def certify_candidate(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    protocol = load_protocol(root)
    if (root / "certification.json").exists():
        raise FileExistsError(root / "certification.json")
    selection = read_json(root / "selected-candidate.json")
    verify_self_hash(selection, label="selected discovery candidate")
    if selection["status"] != "DISCOVERY_CANDIDATE_FROZEN":
        raise RuntimeError("no positive discovery candidate can be certified")
    variant = str(selection["selected_variant"])
    operation = Operation(str(selection["selected_operation"]))
    default_variant, default_operation, candidate_variants = router_spec(protocol)
    if variant not in candidate_variants or candidate_variants[variant] is not operation:
        raise RuntimeError("selected candidate differs from frozen router specification")
    if selection.get("static_incumbent") != default_variant:
        raise RuntimeError("selected candidate static incumbent differs")
    training = read_json(root / "training" / f"{variant}.json")
    if training["training_sha256"] != selection["training_result_sha256"]:
        raise RuntimeError("selected training result identity differs")
    matrix_path = root / "analysis" / "calibration" / "matrix.jsonl"
    rows = read_matrix(matrix_path)
    if len(rows) != SPLIT_SIZES["calibration"]:
        raise RuntimeError("calibration matrix has the wrong task count")
    examples = training_examples(
        rows,
        candidate_variant=variant,
        default_variant=default_variant,
    )
    selected_records = select_calibration_records(
        training,
        examples,
        feature_schema=FEATURE_SCHEMA,
    )
    if not selected_records:
        result = {
            "schema_version": "phiagent-searchqa-residual-certification/v1",
            "status": "FAIL",
            "certified_at_utc": utc_now(),
            "selected_variant": variant,
            "selected_operation": operation.value,
            "failure": "frozen head selected zero task-disjoint calibration tasks",
            "override_coverage": 0.0,
            "gate": _gate_value(protocol),
        }
        result["canonical_sha256"] = canonical_hash(result)
        write_json(root / "certification.json", result, new=True)
        print(json.dumps(result, sort_keys=True))
        return 1
    override_task_hashes = {str(row["task_sha256"]) for row in selected_records}
    full_policy_evidence = bool(protocol["residual_router"]["full_policy_calibration_evidence"])
    reference_variant = str(protocol["residual_router"]["certificate_reference_variant"])
    if reference_variant not in VARIANTS:
        raise RuntimeError("certificate reference variant is unknown")
    if full_policy_evidence:
        calibration_records = []
        for row in rows:
            task_hash = hashlib.sha256(str(row["id"]).encode()).hexdigest()
            overridden = task_hash in override_task_hashes
            deployed = row["arms"][variant if overridden else default_variant]
            reference = row["arms"][reference_variant]
            reference_tokens = int(reference["total_tokens"])
            token_delta = (int(deployed["total_tokens"]) - reference_tokens) / max(
                reference_tokens, 1
            )
            calibration_records.append(
                {
                    "task_sha256": task_hash,
                    "utility": int(deployed["hard"]) - int(reference["hard"]),
                    "harmful": bool(not int(deployed["hard"]) and int(reference["hard"])),
                    "token_delta": max(-1.0, min(1.0, token_delta)),
                    "overridden": overridden,
                }
            )
    else:
        calibration_records = selected_records
    head_id = f"searchqa-initial-{operation.value.casefold()}-v1"
    extractor_hash = str(protocol["feature_extractor_sha256"])
    spec_hash = head_spec_hash(
        head_id=head_id,
        stage="INITIAL",
        operation=operation,
        intercept=float(training["intercept"]),
        weights=training["weights"],
        score_threshold=float(training["score_threshold"]),
        feature_schema=FEATURE_SCHEMA,
        feature_extractor_sha256=extractor_hash,
    )
    evidence = build_evidence_document(
        head_id=head_id,
        training_task_folds=training["training_task_folds"],
        calibration_records=calibration_records,
        source_matrix_sha256=file_sha256(matrix_path),
        head_spec_sha256=spec_hash,
        training_result_sha256=training["training_sha256"],
        alpha=args.alpha,
        bound_method=protocol["residual_router"]["calibration_bound_method"],
        reference_policy_id=f"searchqa-{reference_variant}-v1",
        extra_actor_calls_upper=0,
        router_model_calls=0,
    )
    certificate = certificate_from_evidence(
        evidence,
        expected_head_id=head_id,
        expected_head_spec_sha256=spec_hash,
    )
    gate_value = _gate_value(protocol)
    gate = SafetyGate(**gate_value)
    passes = (
        certificate["training_task_count"] >= gate.minimum_training_tasks
        and certificate["calibration_task_count"] >= gate.minimum_calibration_tasks
        and certificate["effective_sample_size"] >= gate.minimum_effective_sample_size
        and certificate["override_support"] >= gate.minimum_override_support
        and certificate["grouped_crossfit_folds"] >= 2
        and certificate["task_disjoint_calibration"]
        and certificate["harmful_rate_upper"] <= gate.maximum_harmful_rate_upper
        and certificate["utility_lcb"] >= gate.minimum_utility_lcb
        and certificate["token_delta_upper"] <= gate.maximum_token_delta_upper
        and certificate["extra_actor_calls_upper"] <= gate.maximum_extra_actor_calls
        and certificate["router_model_calls"] <= gate.maximum_router_model_calls
    )
    write_json(root / "calibration-evidence.json", evidence, new=True)
    result = {
        "schema_version": "phiagent-searchqa-residual-certification/v1",
        "status": "PASS" if passes else "FAIL",
        "certified_at_utc": utc_now(),
        "selected_variant": variant,
        "selected_operation": operation.value,
        "head_id": head_id,
        "head_spec_sha256": spec_hash,
        "override_coverage": len(override_task_hashes) / len(rows),
        "full_policy_calibration_evidence": full_policy_evidence,
        "certificate_reference_variant": reference_variant,
        "gate": gate_value,
        "certificate": certificate,
        "evidence_sha256": evidence["canonical_sha256"],
    }
    result["canonical_sha256"] = canonical_hash(result)
    write_json(root / "certification.json", result, new=True)
    if not passes:
        print(json.dumps(result, sort_keys=True))
        return 1

    raw_policy = {
        "schema_version": "phiagent-residual-meta-harness-policy/v1",
        "policy_id": f"searchqa-residual-{operation.value.casefold()}-v1",
        "benchmark": "SearchQA",
        "feature_extractor_sha256": extractor_hash,
        "max_overrides_per_task": 1,
        "defaults_by_stage": {"INITIAL": default_operation.value},
        "feature_schema": feature_schema_value(),
        "safety_gate": gate_value,
        "override_heads": [
            {
                "head_id": head_id,
                "stage": "INITIAL",
                "operation": operation.value,
                "intercept": training["intercept"],
                "weights": training["weights"],
                "score_threshold": training["score_threshold"],
                "certificate": certificate,
            }
        ],
        "evidence_documents": {head_id: evidence},
    }
    policy = parse_policy(raw_policy)
    policy_value = policy_to_dict(policy)
    write_json(root / "policy.json", policy_value, new=True)
    opened = {
        "schema_version": "phiagent-searchqa-confirmation-open/v1",
        "status": "OPENED_AFTER_TASK_DISJOINT_CERTIFICATE_PASS",
        "opened_at_utc": utc_now(),
        "protocol_sha256": protocol["canonical_sha256"],
        "candidate_sha256": selection["canonical_sha256"],
        "certification_sha256": result["canonical_sha256"],
        "policy_sha256": policy.policy_sha256,
        "confirmation_ids_sha256": file_sha256(root / "partitions" / "confirmation-ids.json"),
    }
    opened["canonical_sha256"] = canonical_hash(opened)
    write_json(root / "confirmation-open.json", opened, new=True)
    print(json.dumps({"certification": result, "policy": policy_value}, sort_keys=True))
    return 0


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    observed = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        - trials * math.log(2)
    )
    return min(
        1.0,
        sum(
            math.exp(
                math.lgamma(trials + 1)
                - math.lgamma(value + 1)
                - math.lgamma(trials - value + 1)
                - trials * math.log(2)
            )
            for value in range(trials + 1)
            if (
                math.lgamma(trials + 1)
                - math.lgamma(value + 1)
                - math.lgamma(trials - value + 1)
                - trials * math.log(2)
            )
            <= observed + 1e-12
        ),
    )


def paired_bootstrap_interval(
    differences: Sequence[int],
    *,
    seed: int,
    samples: int = 5000,
) -> list[float]:
    generator = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(differences[generator.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def summarize_confirmation(args: argparse.Namespace) -> int:
    root = args.experiment_dir.resolve()
    protocol = load_protocol(root)
    policy = parse_policy(read_json(root / "policy.json"))
    matrix = read_matrix(root / "analysis" / "confirmation" / "matrix.jsonl")
    public = {row["id"]: row for row in read_json(root / "public" / "confirmation.json")}
    method_variants = {
        "No Skill": "no_skill",
        "SkillOpt": "skillopt",
        "AutoSaddler": "autosaddler",
    }
    default_variant, default_operation, candidate_variants = router_spec(protocol)
    operation_variants = {
        default_operation: default_variant,
        **{operation: variant for variant, operation in candidate_variants.items()},
    }
    ours = []
    traces = []
    for row in matrix:
        task_id = str(row["id"])
        session = PolicySession(policy, task_id=task_id, benchmark="SearchQA")
        decision = session.choose(
            stage="INITIAL",
            features=task_features(public[task_id]),
        )
        variant = operation_variants[decision.operation]
        selected = row["arms"][variant]
        ours.append(
            {
                "id": task_id,
                "variant": variant,
                "operation": decision.operation.value,
                "hard": int(selected["hard"]),
                "soft": float(selected["soft"]),
                "total_tokens": int(selected["total_tokens"]),
            }
        )
        traces.append(decision.trace)
    methods = {
        "Ours": {
            "hard": [row["hard"] for row in ours],
            "soft": [row["soft"] for row in ours],
            "tokens": [row["total_tokens"] for row in ours],
        }
    }
    for method, variant in method_variants.items():
        methods[method] = {
            "hard": [int(row["arms"][variant]["hard"]) for row in matrix],
            "soft": [float(row["arms"][variant]["soft"]) for row in matrix],
            "tokens": [int(row["arms"][variant]["total_tokens"]) for row in matrix],
        }
    count = len(matrix)
    raw_p = {}
    comparisons = {}
    for method in method_variants:
        differences = [
            ours_value - baseline_value
            for ours_value, baseline_value in zip(
                methods["Ours"]["hard"],
                methods[method]["hard"],
                strict=True,
            )
        ]
        wins = sum(value == 1 for value in differences)
        losses = sum(value == -1 for value in differences)
        p_value = exact_binomial_two_sided(wins, wins + losses)
        raw_p[method] = p_value
        comparisons[method] = {
            "ours_only_success": wins,
            "baseline_only_success": losses,
            "paired_success_delta": sum(differences) / count,
            "paired_success_delta_95pct_bootstrap": paired_bootstrap_interval(
                differences,
                seed=int(protocol["execution"]["seed"]) + sum(map(ord, method)),
            ),
            "mcnemar_exact_two_sided_p": p_value,
        }
    adjusted = holm_adjust(raw_p)
    for method, value in adjusted.items():
        comparisons[method]["holm_adjusted_p"] = value
    summaries = {
        method: {
            "successes": sum(values["hard"]),
            "success_rate": statistics.fmean(values["hard"]),
            "mean_f1": statistics.fmean(values["soft"]),
            "total_tokens": sum(values["tokens"]),
            "mean_tokens": statistics.fmean(values["tokens"]),
            "model_calls": count,
        }
        for method, values in methods.items()
    }
    ablations = {
        variant: {
            "successes": sum(int(row["arms"][variant]["hard"]) for row in matrix),
            "success_rate": statistics.fmean(int(row["arms"][variant]["hard"]) for row in matrix),
            "total_tokens": sum(int(row["arms"][variant]["total_tokens"]) for row in matrix),
        }
        for variant in ("add", "delete", "modify")
    }
    ours_successes = summaries["Ours"]["successes"]
    result = {
        "schema_version": "phiagent-searchqa-residual-confirmation-summary/v1",
        "status": "COMPLETE",
        "claim_boundary": protocol["benchmark"]["claim_boundary"],
        "task_count": count,
        "official_metric": protocol["benchmark"]["official_metric"],
        "methods": summaries,
        "comparisons_vs_ours": comparisons,
        "operation_ablations": ablations,
        "route_counts": dict(Counter(row["operation"] for row in ours)),
        "trace_sha256": canonical_hash(traces),
        "claim_gate": {
            "ours_better_than_every_compared_method": all(
                ours_successes > summaries[method]["successes"] for method in method_variants
            ),
            "all_model_calls_equal": all(
                values["model_calls"] == count for values in summaries.values()
            ),
            "router_model_calls": 0,
            "extra_actor_calls": 0,
        },
        "completed_at_utc": utc_now(),
    }
    result["canonical_sha256"] = canonical_hash(result)
    write_json(root / "confirmation-summary.json", result, new=True)
    write_json(root / "confirmation-routes.json", ours, new=True)
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--experiment-dir", type=Path, required=True)
    freeze_parser.add_argument("--train-parquet", type=Path, required=True)
    freeze_parser.add_argument("--remainder-ids", type=Path, required=True)
    freeze_parser.add_argument("--prompt-source-dir", type=Path, required=True)
    freeze_parser.add_argument("--service-manifest", type=Path, required=True)
    freeze_parser.add_argument("--source-git-state", type=Path)
    freeze_parser.add_argument("--salt", required=True)
    freeze_parser.add_argument("--seed", type=int, default=20260908)
    freeze_parser.add_argument("--workers", type=int, default=256)
    freeze_parser.add_argument("--max-output-tokens", type=int, default=512)
    freeze_parser.add_argument("--transport-retries", type=int, default=2)
    freeze_parser.add_argument(
        "--discovery-tasks",
        type=int,
        default=SPLIT_SIZES["discovery"],
    )
    freeze_parser.add_argument(
        "--calibration-tasks",
        type=int,
        default=SPLIT_SIZES["calibration"],
    )
    freeze_parser.add_argument(
        "--confirmation-tasks",
        type=int,
        default=SPLIT_SIZES["confirmation"],
    )
    freeze_parser.add_argument(
        "--reserve-tasks",
        type=int,
        default=SPLIT_SIZES["reserve"],
    )
    freeze_parser.add_argument(
        "--static-incumbent",
        choices=tuple(ROUTER_SPECS),
        default="delete",
    )
    freeze_parser.add_argument(
        "--calibration-bound-method",
        choices=("hoeffding", "paired_exact_bonferroni"),
        default="hoeffding",
    )
    freeze_parser.add_argument(
        "--minimum-utility-lcb",
        type=float,
        default=0.0,
    )
    freeze_parser.add_argument(
        "--certificate-reference-variant",
        choices=VARIANTS,
    )
    freeze_parser.add_argument(
        "--full-policy-calibration-evidence",
        action="store_true",
    )
    freeze_parser.set_defaults(handler=freeze)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--experiment-dir", type=Path, required=True)
    materialize_parser.add_argument("--train-parquet", type=Path, required=True)
    materialize_parser.set_defaults(handler=materialize)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment-dir", type=Path, required=True)
    run_parser.add_argument(
        "--split",
        choices=("discovery", "calibration", "confirmation", "reserve"),
        required=True,
    )
    run_parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    run_parser.add_argument("--workers", type=int)
    run_parser.set_defaults(handler=run_matrix)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--experiment-dir", type=Path, required=True)
    score_parser.add_argument(
        "--split",
        choices=("discovery", "calibration", "confirmation", "reserve"),
        required=True,
    )
    score_parser.set_defaults(handler=score_matrix)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--experiment-dir", type=Path, required=True)
    fit_parser.add_argument("--ridge", type=float, default=1.0)
    fit_parser.set_defaults(handler=fit_candidate)

    certify_parser = subparsers.add_parser("certify")
    certify_parser.add_argument("--experiment-dir", type=Path, required=True)
    certify_parser.add_argument("--alpha", type=float, default=0.05)
    certify_parser.set_defaults(handler=certify_candidate)

    summarize_parser = subparsers.add_parser("summarize-confirmation")
    summarize_parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
    )
    summarize_parser.set_defaults(handler=summarize_confirmation)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
