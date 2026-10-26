#!/usr/bin/env python3
"""Stage the complete cached MMMU population without reading any answer column.

This is an input/provenance gate, not an optimizer or a benchmark result. Native
processor work and all dataset access are restricted to the authorized H200.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

SPLITS = ("train", "validation", "test")
FORBIDDEN_INPUT_FIELDS = {"answer", "explanation", "gold", "target", "label"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def grouped_partition(records, *, seed):
    """Union shared-image/question components before an outcome-blind split."""
    if type(seed) is not int or not records:
        raise ValueError("A nonempty population and immutable integer seed are required")
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records) or any(not isinstance(key, str) or not key for key in by_id):
        raise ValueError("Every question must have one distinct nonempty identity")
    parents = {key: key for key in by_id}

    def find(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    anchors = {}
    for identity in sorted(by_id):
        row = by_id[identity]
        if FORBIDDEN_INPUT_FIELDS.intersection(row):
            raise ValueError("Answers/outcomes cannot enter the partitioner")
        values = [("question", row["public_question_sha256"])]
        values += [("image", value) for value in row["image_sha256"]]
        values += [("rgb", value) for value in row["rgb_sha256"]]
        for kind, value in values:
            if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
                raise ValueError("Group identities require SHA256 values")
            anchor = (kind, value)
            if anchor in anchors:
                union(identity, anchors[anchor])
            else:
                anchors[anchor] = identity
    groups = {}
    for identity in sorted(by_id):
        groups.setdefault(find(identity), []).append(identity)
    components = sorted(groups.values(), key=lambda ids: (digest([seed, "mmmu-group", ids]), ids))
    total = len(components)
    if total < 3:
        raise ValueError("At least three independent groups are needed for three splits")
    weights = (2, 2, 6)
    counts = [total * weight // 10 for weight in weights]
    remainders = [total * weight % 10 for weight in weights]
    for index in sorted(range(3), key=lambda i: (-remainders[i], i))[:total - sum(counts)]:
        counts[index] += 1
    if not all(counts):
        raise ValueError("Deterministic rounding leaves an empty split")
    assigned, offset = {}, 0
    for name, count in zip(SPLITS, counts):
        assigned[name] = components[offset:offset + count]
        offset += count
    split_ids = {name: [identity for group in assigned[name] for identity in group] for name in SPLITS}
    result = {
        "schema": "chisel-mmmu-grouped-202060/v1", "split_seed": seed,
        "population": len(records), "independent_groups": total,
        "group_counts": dict(zip(SPLITS, counts)),
        "question_counts": {name: len(ids) for name, ids in split_ids.items()},
        "ratios": [2, 2, 6], "groups": assigned, "splits": split_ids,
        "rounding": "Hamilton largest remainder on group counts; ties train, validation, test",
        "ordering": "SHA256([seed, 'mmmu-group', sorted_component_question_ids])",
        "grouping": "Transitive closure of identical normalized questions/options, original image bytes, or decoded RGB pixels",
        "records": [by_id[key] for key in sorted(by_id)],
        "outcome_or_answer_access": False,
        "prior_public_exposure": True,
        "official_split": False,
    }
    result["membership_sha256"] = digest({name: split_ids[name] for name in SPLITS})
    validate_partition(result)
    return result


def validate_partition(value):
    if value["schema"] != "chisel-mmmu-grouped-202060/v1" or value["outcome_or_answer_access"]:
        raise ValueError("Not an outcome-independent grouped partition")
    all_ids = [identity for name in SPLITS for identity in value["splits"][name]]
    if len(all_ids) != value["population"] or len(set(all_ids)) != len(all_ids):
        raise ValueError("Partition omits or repeats a question")
    records = value["records"]
    if {row["id"] for row in records} != set(all_ids) or len(records) != len(all_ids):
        raise ValueError("Question population and partition disagree")
    split_for = {identity: name for name in SPLITS for identity in value["splits"][name]}
    seen = {}
    for row in records:
        name = split_for[row["id"]]
        features = [("question", row["public_question_sha256"])]
        features += [("image", item) for item in row["image_sha256"]]
        features += [("rgb", item) for item in row["rgb_sha256"]]
        for feature in features:
            if feature in seen and seen[feature] != name:
                raise ValueError("Shared-image/question leakage across partitions")
            seen[feature] = name
    for name in SPLITS:
        flattened = [identity for group in value["groups"][name] for identity in group]
        if flattened != value["splits"][name]:
            raise ValueError("Group assignment differs from question membership")
        if len(value["groups"][name]) != value["group_counts"][name]:
            raise ValueError("Incorrect independent group count")
        if len(flattened) != value["question_counts"][name]:
            raise ValueError("Incorrect question count")
    if sum(value["group_counts"].values()) != value["independent_groups"]:
        raise ValueError("Incorrect total group count")
    if value["membership_sha256"] != digest(value["splits"]):
        raise ValueError("Immutable membership hash changed")


def inference_admission(config, evidence):
    reasons = []
    authority = config["benchmark"]["label_authority"]
    if not authority["selected"] or not authority["resolution_evidence"]:
        reasons.append("Unresolved official HF/publisher label authority; no outcome-selected source choice")
    if evidence.get("native_processor_status") != "ALL_INPUTS_VERIFIED":
        reasons.append("Complete native-image/context evidence is not ready")
    if evidence.get("shared_controller_status") != "FROZEN_AND_TESTED":
        reasons.append("The common controller and native visual adapter are not frozen/tested")
    if evidence.get("official_adapters_status") != "FROZEN_AND_TESTED":
        reasons.append("Both complete official optimizers need frozen/tested native adapters")
    limits = config["execution"]
    if limits["token_usage_mode"] != "unlimited" or limits["maximum_aggregate_tokens"] is not None:
        reasons.append("Cumulative-token termination must be explicitly unlimited")
    return {"admitted": not reasons, "blockers": reasons}


def optimizer_feedback_partition(partition):
    """Disjoint update/gate groups entirely inside the authorized outer TRAIN."""
    validate_partition(partition)
    groups = sorted(partition["groups"]["train"],
                    key=lambda ids: (digest([partition["split_seed"], "optimizer-feedback", ids]), ids))
    update_count = (4 * len(groups) + 2) // 5
    if not 0 < update_count < len(groups):
        raise ValueError("Training needs enough independent groups for a disjoint optimizer gate")
    assigned = {"update": groups[:update_count], "gate": groups[update_count:]}
    ids = {name: [identity for group in values for identity in group] for name, values in assigned.items()}
    if set(ids["update"]) & set(ids["gate"]) or set(ids["update"]) | set(ids["gate"]) != set(partition["splits"]["train"]):
        raise ValueError("Optimizer feedback subsets do not partition outer training")
    return {
        "schema": "chisel-mmmu-training-feedback/v1",
        "outer_membership_sha256": partition["membership_sha256"],
        "split_seed": partition["split_seed"], "ratios": [4, 1],
        "scope": "Training feedback only; neither subset is the external validation or test partition",
        "rounding": "Largest-remainder 4:1 allocation of training groups",
        "groups": assigned, "ids": ids,
        "group_counts": {name: len(values) for name, values in assigned.items()},
        "question_counts": {name: len(values) for name, values in ids.items()},
        "outcome_or_answer_access": False,
    }


def require_remote(config, root):
    if (sys.platform != "linux" or socket.gethostname() != config["hostname"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or root.parent != Path(config["run_namespace"])):
        raise RuntimeError("Dataset/processor preparation requires authorized H200 CPU and a new run")


def inspect_capacity():
    commands = {
        "gpus": ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader"],
        "compute": ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                    "--format=csv,noheader"],
        "processes": ["ps", "-eo", "user,pid,ppid,etimes,args"],
        "locks": ["lslocks", "-o", "COMMAND,PID,PATH"],
        "memory": ["free", "-m"],
        "disk": ["df", "-B1", "/opt/phiagent"],
    }
    return {"at_unix": time.time(), **{
        name: subprocess.check_output(command, text=True) for name, command in commands.items()}}


def verify_model(config):
    model = Path(config["model"]["model_path"])
    prior = load(config["model"]["weight_manifest"])
    if prior["status"] != "VERIFIED" or prior["model_revision"] != config["model"]["revision"]:
        raise RuntimeError("Exact-revision native model weight manifest is absent")
    checked = {}
    for item in prior["verified_files"]:
        path = model / item["path"]
        if path.stat().st_size != item["bytes"] or sha_file(path) != item["sha256"]:
            raise RuntimeError("Cached native model bytes changed: " + item["path"])
        checked[item["path"]] = item
    index = load(model / "model.safetensors.index.json")
    if not set(index["weight_map"].values()).issubset(checked):
        raise RuntimeError("Not every indexed native model shard is pinned")
    visual = [name for name in index["weight_map"] if "visual." in name]
    if not visual:
        raise RuntimeError("The model cache has no native vision tensors")
    processor_files = ("chat_template.jinja", "preprocessor_config.json", "tokenizer_config.json",
                       "tokenizer.json", "video_preprocessor_config.json", "merges.txt", "vocab.json")
    return {
        "model_path": str(model), "repository": config["model"]["repository"],
        "revision": prior["model_revision"], "verified_weights": checked,
        "vision_tensor_count": len(visual),
        "processor_source_sha256": {name: sha_file(model / name) for name in processor_files},
        "weight_manifest_sha256": sha_file(config["model"]["weight_manifest"]),
        "processor_publisher_pin_status": "BYTE_FROZEN; publisher blob identities require separate evidence",
    }


def prepare(root, config):
    from PIL import Image
    from transformers import AutoProcessor
    from cvpr_mmmu_campaign import messages, options_list

    owner = Path(config["benchmark"]["owner_root"])
    raw = Path(config["benchmark"]["raw_root"])
    if sha_file(owner / "frozen/manifest.json") != config["benchmark"]["owner_manifest_sha256"]:
        raise RuntimeError("Immutable complete cached population manifest changed")
    if sha_file(raw / "SOURCE_MANIFEST.json") != config["benchmark"]["raw_manifest_sha256"]:
        raise RuntimeError("Immutable complete official raw shard manifest changed")
    gate = load(owner / "source-gate.json")
    if (gate["github_revision"] != config["benchmark"]["revision"]
            or gate["dataset_revision"] != config["benchmark"]["dataset_revision"]):
        raise RuntimeError("Official data/scorer revision differs")
    manifest = load(owner / "frozen/manifest.json")
    verified_raw = {}
    raw_manifest = load(raw / "SOURCE_MANIFEST.json")
    if len(raw_manifest["files"]) != 60:
        raise RuntimeError("All sixty cached public official annotated shards are required")
    for entry in raw_manifest["files"]:
        path = raw / "public-data" / entry["path"]
        if path.stat().st_size != entry["size"] or sha_file(path) != entry["sha256"]:
            raise RuntimeError("Pinned original dataset shard changed")
        verified_raw[entry["path"]] = entry["sha256"]
    official = root / "official/MMMU"
    for relative, expected in gate["official_source_sha256"].items():
        path = owner / "official/MMMU" / relative
        if sha_file(path) != expected:
            raise RuntimeError("Pinned official scorer/documentation changed: " + relative)
        target = official / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write(root / "provenance/official-sources.json", {
        "source_gate_sha256": sha_file(owner / "source-gate.json"),
        "raw_shards": verified_raw, "official_source_sha256": gate["official_source_sha256"],
        "historical_host": gate["hostname"], "current_host": socket.gethostname(),
        "migration_does_not_change_file_identity": True, "answers_parsed": False,
    })
    model_proof = verify_model(config)
    write(root / "provenance/model-bytes.json", model_proof)
    processor = AutoProcessor.from_pretrained(config["model"]["model_path"], local_files_only=True)
    template_path = owner / "frozen/reference-template.json"
    if sha_file(template_path) != manifest["frozen/reference-template.json"]:
        raise RuntimeError("The reference prompt template changed")
    template = load(template_path)
    write(root / "frozen/reference-template.json", template)
    rows = []
    for split, expected in config["benchmark"]["declared_population"].items():
        relative = f"frozen/{split}.json"
        if sha_file(owner / relative) != manifest[relative]:
            raise RuntimeError("Cached answer-free input projection changed")
        source_rows = load(owner / relative)
        if len(source_rows) != expected:
            raise RuntimeError("Incomplete declared official source split")
        expected_per_subject = expected // 30
        expected_ids = {f"{split}_{subject}_{i}" for subject in gate["subjects"]
                        for i in range(1, expected_per_subject + 1)}
        if {row["id"] for row in source_rows} != expected_ids:
            raise RuntimeError("Full original population identities differ")
        for row in source_rows:
            if FORBIDDEN_INPUT_FIELDS.intersection(row):
                raise RuntimeError("Cached public inputs unexpectedly contain targets")
            row = {key: value for key, value in row.items() if key != "prompts"}
            row["source_split"] = split
            rows.append(row)
    if len({row["id"] for row in rows}) != 1050 or len(gate["subjects"]) != 30:
        raise RuntimeError("Complete cached official1050/30-subject population is required")
    image_cache, records, proofs = {}, [], []
    for position, row in enumerate(sorted(rows, key=lambda item: item["id"])):
        source_images = row["images"]
        images, raw_hashes, rgb_hashes = {}, [], []
        for index, info in sorted(source_images.items(), key=lambda entry: int(entry[0])):
            path = owner / info["path"]
            if str(path.resolve()).startswith(str((owner / "frozen/images").resolve()) + "/") is False:
                raise RuntimeError("Original image escapes the frozen source image root")
            if info["path"] not in manifest or manifest[info["path"]] != info["sha256"]:
                raise RuntimeError("Image does not belong to the pinned full population")
            if info["sha256"] not in image_cache:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise RuntimeError("Original image bytes changed")
                with Image.open(io.BytesIO(data)) as decoded:
                    rgb = decoded.convert("RGB")
                    rgb_sha = hashlib.sha256(canonical(list(rgb.size)) + rgb.tobytes()).hexdigest()
                    size = list(decoded.size)
                destination = root / "frozen/images" / (info["sha256"] + ".bin")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                image_cache[info["sha256"]] = {
                    "path": str(destination), "sha256": info["sha256"],
                    "rgb_sha256": rgb_sha, "original_size": size, "original_bytes": len(data)}
            bound = image_cache[info["sha256"]]
            images[int(index)] = bound["path"]
            raw_hashes.append(bound["sha256"])
            rgb_hashes.append(bound["rgb_sha256"])
        chat = messages(row, template, "", images)
        native = processor.apply_chat_template(
            chat, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=config["model"]["enable_thinking"])
        token_ids = native["input_ids"][0].tolist()
        image_token_id = processor.image_token_id
        image_tokens = token_ids.count(image_token_id)
        if bool(images) != bool(image_tokens):
            raise RuntimeError("Native processor lost images or invented image-token evidence")
        if images and ("pixel_values" not in native or "image_grid_thw" not in native):
            raise RuntimeError("No actual native image tensor/grid was produced")
        if len(token_ids) + config["model"]["max_tokens"] > config["model"]["max_model_len"]:
            raise RuntimeError("An original complete context exceeds the frozen actor context; no clipping allowed")
        row["images"] = {str(index): image_cache[info["sha256"]]
                         for index, info in source_images.items()}
        row["native_prompt"] = {
            "prompt_tokens": len(token_ids), "token_ids_sha256": digest(token_ids),
            "image_tokens": image_tokens,
            "image_grid_thw": native["image_grid_thw"].tolist() if images else [],
            "pixel_tensor_elements": native["pixel_values"].numel() if images else 0,
            "native_processor": type(processor).__name__,
        }
        def normalized(text):
            return " ".join(text.split())
        records.append({
            "id": row["id"], "subject": row["subject"], "source_split": row["source_split"],
            "public_question_sha256": digest([
                normalized(row["question"]), [normalized(option) for option in options_list(row["options"])]]),
            "image_sha256": sorted(set(raw_hashes)), "rgb_sha256": sorted(set(rgb_hashes)),
        })
        proofs.append({"id": row["id"], **row["native_prompt"]})
        if (position + 1) % 50 == 0:
            print(json.dumps({"event": "native_input_proof", "completed": position + 1,
                              "population": len(rows)}), flush=True)
    partition = grouped_partition(records, seed=config["benchmark"]["split_seed"])
    write(root / "frozen/SPLIT_MANIFEST.json", partition)
    by_id = {row["id"]: row for row in rows}
    for name in SPLITS:
        write(root / "frozen" / (name + "-inputs.json"),
              [by_id[identity] for identity in partition["splits"][name]])
    write(root / "frozen/native-input-proofs.json", proofs)
    native_status = {
        "native_processor_status": "ALL_INPUTS_VERIFIED",
        "population": len(rows), "images": len(image_cache),
        "image_bearing_questions": sum(bool(row["images"]) for row in rows),
        "zero_image_questions_retained": sum(not row["images"] for row in rows),
        "prompt_tokens": sum(item["prompt_tokens"] for item in proofs),
        "image_tokens": sum(item["image_tokens"] for item in proofs),
        "max_prompt_tokens": max(item["prompt_tokens"] for item in proofs),
        "processor": type(processor).__name__,
        "native_image_processor": type(processor.image_processor).__name__,
        "image_token_id": processor.image_token_id,
        "group_counts": partition["group_counts"], "question_counts": partition["question_counts"],
        "membership_sha256": partition["membership_sha256"], "source_labels_read": False,
        "model_calls": 0, "scorer_calls": 0, "gpu_allocations": 0,
        "shared_controller_status": "WAITING_SHARED_CORE_API",
        "official_adapters_status": "NOT_IMPLEMENTED_FOR_NEW_PARTITION",
    }
    write(root / "NATIVE_INPUT_GATE.json", native_status)
    return native_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config, root = load(args.config), args.root.resolve()
    require_remote(config, root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "work").mkdir()
    os.environ.update(TMPDIR=str(root / "work"), TMP=str(root / "work"),
                      TEMP=str(root / "work"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    write(root / "frozen/config.json", config)
    write(root / "STARTED.json", {
        "hostname": socket.gethostname(), "pid": os.getpid(), "at_unix": time.time(),
        "command": sys.argv, "python": sys.version, "source_sha256": sha_file(__file__),
        "config_sha256": sha_file(args.config), "CUDA_VISIBLE_DEVICES": "",
        "scope": "Answer-free source/input/native-processor staging only; no inference or scoring",
    })
    write(root / "CAPACITY_BEFORE.json", inspect_capacity())
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "vllm", "Pillow", "numpy")}
    write(root / "provenance/packages.json", packages)
    try:
        evidence = prepare(root, config)
        admission = inference_admission(config, evidence)
        final = {
            "status": "INPUTS_STAGED_INFERENCE_BLOCKED" if not admission["admitted"] else "INPUTS_READY",
            "at_unix": time.time(), **evidence, **admission,
            "benchmark_outcome": False, "innovation_admitted": False,
        }
        write(root / "READINESS.json", final)
        write(root / "frozen/FILES_SHA256.json", {
            str(path.relative_to(root)): sha_file(path)
            for path in sorted((root / "frozen").rglob("*")) if path.is_file()})
        write(root / "CAPACITY_AFTER.json", inspect_capacity())
        print(json.dumps(final, sort_keys=True), flush=True)
    except BaseException as error:
        write(root / "FAILURE.json", {
            "at_unix": time.time(), "error": repr(error), "traceback": traceback.format_exc(),
            "model_calls": 0, "scorer_calls": 0, "gpu_allocations": 0})
        raise


if __name__ == "__main__":
    main()
