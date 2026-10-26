#!/usr/bin/env python3
"""Remote native MMMU stages with isolated labels and unlimited token accounting."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_chisel_mmmu_shared_202060 import (  # noqa: E402
    canonical, digest, load, sha_file, validate_partition, write,
)
from scripts.cvpr_mmmu_campaign import messages, options_list  # noqa: E402
from scripts.chisel_mmmu_shared_adapter import RECOVERY_INSTRUCTION, visual_decision  # noqa: E402

METHODS = ("no_skill", "skillopt", "autosaddler", "chisel")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def checkpoint(path, value):
    path = Path(path)
    staged = path.with_suffix(path.suffix + ".new")
    staged.write_bytes(canonical(value) + b"\n")
    staged.replace(path)


def load_config(path):
    path = Path(path)
    value = load(path)
    if "base_config" not in value:
        return value
    result = load_config(path.parent / value["base_config"])

    def merge(left, right):
        for key, item in right.items():
            if isinstance(item, dict) and isinstance(left.get(key), dict):
                merge(left[key], item)
            else:
                left[key] = copy.deepcopy(item)

    merge(result, value["overrides"])
    if "ports" in result.get("model", {}):
        result["model"]["ports"] = {
            str(slot["index"]): result["model"]["ports"][str(slot["index"])]
            for slot in result["resources"]["slots"]}
    return result


def assert_remote(config, root):
    if (sys.platform != "linux" or socket.gethostname() != config["hostname"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or root.parent != Path(config["run_namespace"])
            or (config.get("run_directory") and root != Path(config["run_directory"]))):
        raise RuntimeError("All model calls, benchmark/scorer work and optimizer execution are remote-only")
    execution = config["execution"]
    if execution["token_usage_mode"] != "unlimited" or execution["maximum_aggregate_tokens"] is not None:
        raise RuntimeError("Discretionary cumulative-token termination is forbidden")
    authority = config["benchmark"]["label_authority"]
    if authority["selected"] != "hf-raw" or not authority["resolution_evidence"]:
        raise RuntimeError("Unresolved official annotation authority blocks model/scorer execution")


def verify_frozen(root):
    for relative, expected in load(root / "SOURCE_SHA256.json").items():
        if sha_file(root / relative) != expected:
            raise RuntimeError("Frozen source/config changed: " + relative)


def read_rows(config, stage):
    if stage not in {"train", "validation", "test"}:
        raise ValueError("Unknown partition")
    inputs = Path(config["input_root"])
    partition = load(inputs / "frozen/SPLIT_MANIFEST.json")
    validate_partition(partition)
    if partition["membership_sha256"] != config["input_membership_sha256"]:
        raise RuntimeError("Immutable common split membership changed")
    manifest = load(inputs / "frozen/FILES_SHA256.json")
    relative = f"frozen/{stage}-inputs.json"
    if sha_file(inputs / relative) != manifest[relative]:
        raise RuntimeError("Frozen answer-free native inputs changed")
    rows = load(inputs / relative)
    if [row["id"] for row in rows] != partition["splits"][stage]:
        raise RuntimeError("Method population differs from common membership")
    if any({"answer", "gold", "label", "explanation"}.intersection(row) for row in rows):
        raise RuntimeError("An inference input contains a gold target")
    return rows


def native_admission_row(rows, config):
    indices = sorted(int(index) for index in config["model"]["ports"])
    for row in rows:
        if indices[int(digest(row["id"])[:12], 16) % len(indices)] == indices[0]:
            return row
    raise RuntimeError("No training question maps to the first physical GPU")


def image_urls(chat):
    return [part["image_url"]["url"] for message in chat if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") == "image_url"]


def image_mime(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    raise ValueError("Unknown original image encoding; do not relabel or transcode it")


def pack_request(body, asset_root):
    """Lossless content-addressed images avoid duplicating pixels in every log."""
    packed = copy.deepcopy(body)
    for message in packed.get("messages", []):
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            header, encoded = url.split(",", 1)
            data = base64.b64decode(encoded, validate=True)
            image_hash = hashlib.sha256(data).hexdigest()
            path = asset_root / (image_hash + ".bin")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as stream:
                    stream.write(data)
            except FileExistsError:
                if sha_file(path) != image_hash:
                    raise RuntimeError("Captured image asset collision")
            part["image_url"]["url"] = {
                "lossless_image_asset_sha256": image_hash, "original_data_url_header": header}
    return packed


def unpack_request(packed, asset_root):
    body = copy.deepcopy(packed)
    for message in body.get("messages", []):
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") != "image_url":
                continue
            entry = part["image_url"]["url"]
            data = (asset_root / (entry["lossless_image_asset_sha256"] + ".bin")).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["lossless_image_asset_sha256"]:
                raise RuntimeError("Captured original pixels changed")
            part["image_url"]["url"] = (
                entry["original_data_url_header"] + "," + base64.b64encode(data).decode())
    return body


class NativeClient:
    def __init__(self, root, config, method, stage):
        self.root, self.config, self.method, self.stage = root, config, method, stage
        self.directory = root / "methods" / method / stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sequence = 0
        self.records = []
        self.pending = {}
        self.failed = False
        self.proof = load(Path(config["input_root"]) / "NATIVE_INPUT_GATE.json")
        self.image_cache = {}
        cache_path = root / "frozen/RESUMABLE_PROVIDER_RECORDS.json"
        self.reusable = [item for item in load(cache_path)
                         if item["method"] == method and item["stage"] == stage] if cache_path.exists() else []

    def image_url(self, image):
        image_hash = image["sha256"]
        with self.lock:
            if image_hash not in self.image_cache:
                data = Path(image["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != image_hash:
                    raise RuntimeError("Original image bytes changed")
                self.image_cache[image_hash] = (
                    "data:" + image_mime(data) + ";base64," + base64.b64encode(data).decode())
            return self.image_cache[image_hash]

    def chat(self, row, instruction=""):
        template = load(Path(self.config["input_root"]) / "frozen/reference-template.json")
        urls = {int(index): self.image_url(image) for index, image in row["images"].items()}
        chat = messages(row, template, instruction, urls)
        for message in chat:
            for part in message["content"]:
                if part["type"] == "image":
                    part["type"] = "image_url"
                    part["image_url"] = {"url": part.pop("image")}
        return chat

    def original_parts(self, rows):
        result = []
        for row in rows:
            result.append({"type": "text", "text": "Original input for training question " + row["id"]})
            result.extend(self.chat(row)[0]["content"])
        return result

    def _archive(self, path, value):
        with self.lock:
            packed = pack_request(value, self.root / "request-image-assets" / self.method)
        with path.open("xb") as stream:
            stream.write(gzip.compress(canonical(packed), mtime=0))

    def _post(self, endpoint, body, path):
        self._archive(path.with_suffix(".request.json.gz"), body)
        request = Request(endpoint, canonical(body), {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.config["execution"]["request_timeout_seconds"]) as response:
                raw = response.read()
        except HTTPError as error:
            path.with_suffix(".http-error.bin").write_bytes(error.read())
            raise
        with path.with_suffix(".response.json.gz").open("xb") as stream:
            stream.write(gzip.compress(raw, mtime=0))
        return json.loads(raw)

    def request(self, chat, *, role, rows=(), payload=None):
        if (self.root / "DISPATCH_FROZEN.json").exists():
            raise RuntimeError("Campaign failure froze new provider dispatches")
        verify_frozen(self.root)
        with self.lock:
            if self.failed:
                raise RuntimeError("A preserved provider failure blocks new requests")
            sequence = self.sequence
            self.sequence += 1
            self.pending[sequence] = {"status": "TOKENIZING", "model_dispatch_started": False}
        directory = self.directory / "requests" / f"{sequence:06d}"
        directory.mkdir(parents=True)
        expected_urls = [url for row in rows for url in image_urls(self.chat(row))]
        urls = image_urls(chat)
        if urls != expected_urls:
            raise RuntimeError("A native request changed, dropped or reordered its exact source images")
        maximum_images = self.config["model"]["max_optimizer_images"] if role == "optimizer" else 7
        if len(urls) > maximum_images:
            raise RuntimeError("Complete optimizer image scope exceeds frozen capacity; no clipping")
        indices = sorted(int(index) for index in self.config["model"]["ports"])
        key = rows[0]["id"] if role != "optimizer" and rows else [self.method, role]
        gpu = indices[int(digest(key)[:12], 16) % len(indices)]
        gpu_uuid = next(slot["uuid"] for slot in self.config["resources"]["slots"] if slot["index"] == gpu)
        endpoint = f"http://127.0.0.1:{self.config['model']['ports'][str(gpu)]}"
        model = self.config["model"]
        cap = model["optimizer_max_tokens"] if role == "optimizer" else model["max_tokens"]
        body = copy.deepcopy(payload) if payload is not None else {}
        body.update(model=model["served_name"], messages=chat, temperature=0, top_p=1,
                    max_tokens=cap, stream=False, seed=model["seed"],
                    chat_template_kwargs={"enable_thinking": False})
        body.pop("max_completion_tokens", None)
        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            from scripts.chisel_mmmu_native_schema import native_json_schema
            original_schema = response_format["json_schema"]["schema"]
            compiled_schema, adaptations = native_json_schema(original_schema)
            if adaptations:
                response_format["json_schema"]["schema"] = compiled_schema
                write(directory / "native-schema-adaptation.json", {
                    "original_schema": original_schema, "wire_schema": compiled_schema,
                    "adaptations": adaptations, "official_post_generation_validation_unchanged": True})
        metadata = {
            "sequence": sequence, "method": self.method, "stage": self.stage, "role": role,
            "question_ids": [row["id"] for row in rows], "physical_gpu": gpu,
            "hostname": self.config["hostname"], "physical_uuid": gpu_uuid,
            "started_at_unix": time.time(), "request_sha256": digest(body),
            "image_sha256": [hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
                             for url in urls], "transport_retries": 0,
        }
        write(directory / "metadata.json", metadata)
        started = False
        try:
            token_body = {key: body[key] for key in ("model", "messages", "chat_template_kwargs", "tools", "tool_choice")
                          if key in body}
            token_body.update(add_generation_prompt=True, add_special_tokens=False, return_token_strs=False)
            tokens = self._post(endpoint + "/tokenize", token_body, directory / "tokenize")
            count, ids = tokens["count"], tokens["tokens"]
            expected_visual = sum(row["native_prompt"]["image_tokens"] for row in rows)
            actual_visual = ids.count(self.proof["image_token_id"])
            if (type(count) is not int or count != len(ids) or actual_visual != expected_visual
                    or (urls and actual_visual <= 0) or count + cap > model["max_model_len"]):
                raise RuntimeError("Native image-token/context proof failed; no truncation or repair")
            write(directory / "native-proof.json", {
                "prompt_tokens": count, "actual_image_tokens": actual_visual,
                "expected_image_tokens": expected_visual, "token_ids_sha256": digest(ids),
                "original_images": len(urls), "maximum_output_tokens": cap,
            })
            if (self.root / "DISPATCH_FROZEN.json").exists():
                raise RuntimeError("Campaign failure froze dispatch after tokenization")
            with self.lock:
                reuse = next((item for item in self.reusable
                              if item["request_sha256"] == metadata["request_sha256"]
                              and item["role"] == role and item["physical_gpu"] == gpu
                              and item["physical_uuid"] == gpu_uuid
                              and item["hostname"] == self.config["hostname"]), None)
                if reuse is not None:
                    self.reusable.remove(reuse)
                self.pending[sequence].update(
                    native_prompt_tokens=count, possible_token_upper_bound=count + cap)
            if reuse is not None:
                if (sha_file(reuse["prior_record_path"]) != reuse["prior_record_sha256"]
                        or sha_file(reuse["prior_response_path"]) != reuse["prior_response_sha256"]):
                    raise RuntimeError("A resumable native response changed")
                self._archive(directory / "model.request.json.gz", body)
                raw_bytes = Path(reuse["prior_response_path"]).read_bytes()
                (directory / "model.response.json.gz").write_bytes(raw_bytes)
                raw = json.loads(gzip.decompress(raw_bytes))
            else:
                started = True
                with self.lock:
                    self.pending[sequence].update(status="INFLIGHT", model_dispatch_started=True)
                    checkpoint(self.directory / "COST.json", self.cost())
                raw = self._post(endpoint + "/v1/chat/completions", body, directory / "model")
            usage = raw.get("usage", {})
            if (raw.get("model") != model["served_name"]
                    or any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_FIELDS)
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                    or usage["prompt_tokens"] != count or usage["completion_tokens"] > cap):
                raise RuntimeError("Native provider usage or model identity mismatch")
            if len(raw["choices"]) != 1:
                raise RuntimeError("Exactly one native completion is required")
            content = raw["choices"][0]["message"].get("content")
            if content is None and raw["choices"][0]["message"].get("tool_calls"):
                content = ""
            if not isinstance(content, str):
                raise RuntimeError("Native response has no usable unmodified content")
            record = {
                **metadata, "raw_content": content, "usage": {key: usage[key] for key in USAGE_FIELDS},
                "finished_at_unix": time.time(), "native_image_tokens": actual_visual,
                "finish_reason": raw["choices"][0].get("finish_reason"),
                "request_directory": str(directory.relative_to(self.root)),
                "provider_reused": reuse is not None,
                "provider_origin": reuse["provider_origin"] if reuse is not None else str(directory),
            }
            if reuse is not None:
                record["prior_record_sha256"] = reuse["prior_record_sha256"]
                record["prior_source_manifest_sha256"] = reuse["prior_source_manifest_sha256"]
            write(directory / "record.json", record)
            with self.lock:
                self.records.append(record)
                del self.pending[sequence]
                checkpoint(self.directory / "COST.json", self.cost())
            first = self.root / "FIRST_MODEL_OUTPUT.json"
            try:
                write(first, record)
            except FileExistsError:
                pass
            if reuse is None:
                try:
                    write(self.root / "FIRST_FRESH_MODEL_OUTPUT.json", record)
                except FileExistsError:
                    pass
            return record, raw
        except BaseException as error:
            with self.lock:
                self.failed = True
                self.pending[sequence].update(
                    status="UNKNOWN_USAGE" if started else "NOT_DISPATCHED",
                    model_dispatch_started=started, error=repr(error))
                checkpoint(self.directory / "COST.json", self.cost())
            write(directory / "FAILURE.json", {
                **metadata, "error": repr(error), "traceback": traceback.format_exc(),
                "model_dispatch_started": started, "unknown_usage": started,
            })
            raise

    def actor(self, row, instruction=""):
        return self.request(self.chat(row, instruction), role="actor", rows=[row])[0]

    def cost(self):
        roles = sorted({record["role"] for record in self.records})
        return {
            "token_usage_mode": "unlimited", "maximum_aggregate_tokens": None,
            "requests": len(self.records), "failed": self.failed,
            "new_completed_provider_calls": sum(not record.get("provider_reused", False) for record in self.records),
            "reused_completed_provider_calls": sum(record.get("provider_reused", False) for record in self.records),
            "pending_or_failed_attempts": dict(self.pending),
            "unknown_usage_attempts": sum(item["status"] == "UNKNOWN_USAGE" for item in self.pending.values()),
            "by_role": {role: {
                "requests": sum(record["role"] == role for record in self.records),
                **{key: sum(record["usage"][key] for record in self.records if record["role"] == role)
                   for key in USAGE_FIELDS}} for role in roles},
            "distinct_question_ids": sorted({identity for record in self.records for identity in record["question_ids"]}),
        }


class Study:
    def __init__(self, root, config, method, stage):
        self.root, self.config, self.method, self.stage = root, config, method, stage
        if stage in {"validation", "test"}:
            if (root / "GLOBAL_RELEASE_BLOCKED.json").exists():
                raise RuntimeError("A method failure sealed all new external validation/test access")
            gate = load(root / "ALL_TRAIN_COMPLETE.json")
            if (not gate["every_training_worker_complete"] or set(gate["methods"]) != set(METHODS)
                    or gate["source_manifest_sha256"] != sha_file(root / "SOURCE_SHA256.json")
                    or gate["membership_sha256"] != config["input_membership_sha256"]):
                raise RuntimeError("All four complete training banks are required before external validation")
            for name in METHODS:
                if sha_file(root / "methods" / name / "CANDIDATES.json") != gate["methods"][name]:
                    raise RuntimeError("A training-derived candidate bank changed after global release")
        if stage == "test":
            final = load(root / "FINAL_POLICY_FREEZE.json")
            if set(final["methods"]) != set(METHODS):
                raise RuntimeError("Not every matched method was frozen before test")
            verify_frozen(root)
            for name in METHODS:
                if sha_file(root / "methods" / name / "SELECTED.json") != final["methods"][name]:
                    raise RuntimeError("Selected method changed after final freeze")
        self.rows = read_rows(config, "train" if stage == "admission" else stage)
        if stage == "admission":
            if method != "no_skill":
                raise RuntimeError("The fixed native operability gate uses only No Skill")
            self.rows = [native_admission_row(self.rows, config)]
        self.by_id = {row["id"]: row for row in self.rows}
        self.client = NativeClient(root, config, method, stage)
        self.lock = threading.RLock()
        self.score_sequence = 0
        self.labels = self.read_labels()

    def read_labels(self):
        import pyarrow.parquet as pq
        raw = Path(self.config["benchmark"]["raw_root"])
        wanted = set(self.by_id)
        labels = {}
        for entry in load(raw / "SOURCE_MANIFEST.json")["files"]:
            subject, original_split = entry["path"].split("/", 1)
            source_split = original_split.split("-", 1)[0]
            ids = [row["id"] for row in self.rows
                   if row["subject"] == subject and row["source_split"] == source_split]
            if not ids:
                continue
            path = raw / "public-data" / entry["path"]
            if sha_file(path) != entry["sha256"]:
                raise RuntimeError("Official raw labels changed after source freeze")
            # Arrow's filtered projection exposes only the authorized phase IDs.
            for row in pq.read_table(path, columns=["id", "answer"],
                                     filters=[("id", "in", ids)]).to_pylist():
                if row["id"] not in wanted or row["id"] in labels:
                    raise RuntimeError("Label projection crossed a partition boundary")
                labels[row["id"]] = row["answer"]
        if set(labels) != wanted:
            raise RuntimeError("Incomplete authoritative annotation population")
        write(self.client.directory / "LABEL_ACCESS.json", {
            "phase": self.stage, "ids": sorted(labels), "authority": "hf-raw",
            "resolution_evidence": self.config["benchmark"]["label_authority"]["resolution_evidence"],
            "test_policy_freeze_exists": (self.root / "FINAL_POLICY_FREEZE.json").is_file(),
        })
        return labels

    def score(self, rows, outputs, *, label):
        with self.lock:
            sequence = self.score_sequence
            self.score_sequence += 1
        directory = self.client.directory / "scores" / f"{sequence:05d}-{label}"
        subjects = {}
        if len(rows) != len(outputs) or len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError("Native scoring denominator mismatch")
        for row, output in zip(rows, outputs):
            options = options_list(row["options"])
            subjects.setdefault(row["subject"], []).append({
                "id": row["id"], "question_type": row["question_type"], "answer": self.labels[row["id"]],
                "response": output["raw_content"],
                "all_choices": [chr(65 + index) for index in range(len(options))],
                "index2ans": {chr(65 + index): option for index, option in enumerate(options)},
            })
        for subject, items in subjects.items():
            write(directory / subject / "output.json", items)
        official = self.root / "official/MMMU/mmmu"
        command = [sys.executable, str(official / "main_parse_and_eval.py"),
                   "--path", str(directory), "--subject", "ALL"]
        write(directory / "COMMAND.json", {"argv": command, "cwd": str(official)})
        with (directory / "official.log").open("xb") as log:
            subprocess.run(command, cwd=official, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"})
        judged, results = {}, {}
        for subject in subjects:
            results[subject] = load(directory / subject / "result.json")
            for item in load(directory / subject / "parsed_output.json"):
                judged[item["id"]] = int(item["judge"] == "Correct")
        if set(judged) != {row["id"] for row in rows}:
            raise RuntimeError("Unmodified official parser changed the scored population")
        spec = importlib.util.spec_from_file_location("chisel_pinned_mmmu_metric", official / "utils/eval_utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metric = float(module.calculate_ins_level_acc(results))
        summary = {
            "count": len(rows), "correct": sum(judged.values()), "official_accuracy": metric,
            "official_metric": "mmmu/utils/eval_utils.py:calculate_ins_level_acc",
            "scorer_unmodified": True, "authority": "hf-raw", "phase": self.stage,
        }
        if abs(metric - sum(judged.values()) / len(rows)) > 1e-12:
            raise RuntimeError("Official instance-weighted metric disagrees")
        write(directory / "METRIC.json", summary)
        for row, output in zip(rows, outputs):
            output["hard"] = judged[row["id"]]
        first = self.root / "FIRST_NATIVE_METRIC.json"
        try:
            write(first, {**summary, "method": self.method,
                          "request_ids": [output.get("request_directory") for output in outputs]})
        except FileExistsError:
            pass
        return summary

    def infer(self, row, candidate):
        if self.method != "chisel":
            record = self.client.actor(row, candidate.get("instruction", ""))
            result = {**record, "id": row["id"], "inference_usage": record["usage"]}
        else:
            output, trace, calls = visual_decision(
                row, policy=candidate["policy"], draft=lambda: self.client.actor(row),
                recover=lambda draft: self.client.actor(
                    row, RECOVERY_INSTRUCTION + "\nPrevious draft:\n" + draft["raw_content"]),
            )
            result = {
                **output, "id": row["id"], "decision_trace": trace,
                "inference_requests": [call["request_directory"] for call in calls],
                "inference_usage": {key: sum(call["usage"][key] for call in calls) for key in USAGE_FIELDS},
                "draft_raw_content": calls[0]["raw_content"],
            }
        write(self.client.directory / "predictions" / digest(candidate) / (row["id"] + ".json"), result)
        return result

    def batch(self, rows, candidate, *, label):
        with ThreadPoolExecutor(max_workers=self.config["execution"]["actor_workers"]) as pool:
            outputs = list(pool.map(lambda row: self.infer(row, candidate), rows))
        score = self.score(rows, outputs, label=label)
        report = {
            "candidate": candidate, "candidate_sha256": digest(candidate), "metric": score,
            "outputs": outputs,
            "inference_usage": {key: sum(item["inference_usage"][key] for item in outputs)
                                for key in USAGE_FIELDS},
            "membership_sha256": self.config["input_membership_sha256"],
        }
        write(self.client.directory / (label + ".json"), report)
        return report

    def train(self):
        if self.method == "no_skill":
            candidates = [{"instruction": ""}]
            self.batch(self.rows, candidates[0], label="train-population")
        elif self.method == "chisel":
            candidate = {"policy": "native_format_recovery"}
            report = self.batch(self.rows, candidate, label="train-recovery")
            control = [{**row, "raw_content": row["draft_raw_content"]} for row in report["outputs"]]
            control_metric = self.score(self.rows, control, label="train-draft-control")
            write(self.client.directory / "TRAIN_DERIVATION.json", {
                "recovery_metric": report["metric"], "draft_control_metric": control_metric,
                "local_format_fallbacks": sum(
                    row["decision_trace"]["route"] == "fallback" and len(row["inference_requests"]) > 1
                    for row in report["outputs"]),
                "only_training_supplies_feedback": True,
                "candidate_rules_fixed_before_validation": True,
            })
            candidates = [{"policy": "draft_only"}]
            if report["metric"]["correct"] > control_metric["correct"]:
                candidates.append({"policy": "native_format_recovery"})
        else:
            from scripts.chisel_mmmu_official_optimizers import train_autosaddler, train_skillopt
            candidates = (train_skillopt if self.method == "skillopt" else train_autosaddler)(self)
        if not candidates:
            raise RuntimeError("A complete optimizer must produce a frozen training-derived candidate bank")
        unique = {digest(candidate): candidate for candidate in candidates}
        write(self.root / "methods" / self.method / "CANDIDATES.json", {
            "candidates": [unique[key] for key in sorted(unique)], "training_complete": True,
            "available_training_question_count": len(self.rows),
            "optimizer_sample_count": len(self.client.cost()["distinct_question_ids"]) if self.method != "no_skill" else 0,
            "cost": self.client.cost(), "source_sha256": sha_file(self.root / "SOURCE_SHA256.json"),
            "at_unix": time.time(), "validation_seen": False, "test_seen": False,
        })

    def admission(self):
        self.batch(self.rows, {"instruction": ""}, label="native-admission")

    def validation(self):
        bank = load(self.root / "methods" / self.method / "CANDIDATES.json")
        if not bank["training_complete"]:
            raise RuntimeError("Incomplete official optimization cannot enter validation")
        results = []
        for index, candidate in enumerate(bank["candidates"]):
            result = self.batch(self.rows, candidate, label=f"candidate-{index:03d}")
            results.append(result)
        selected = min(results, key=lambda row: (
            -row["metric"]["correct"], row["inference_usage"]["total_tokens"], row["candidate_sha256"]))
        write(self.root / "methods" / self.method / "SELECTED.json", {
            "candidate": selected["candidate"], "candidate_sha256": selected["candidate_sha256"],
            "validation_metric": selected["metric"], "source_sha256": sha_file(self.root / "SOURCE_SHA256.json"),
            "bank_sha256": sha_file(self.root / "methods" / self.method / "CANDIDATES.json"),
            "training_source_sha256": bank["source_sha256"],
            "at_unix": time.time(), "test_seen": False,
            "validation_not_fed_to_optimizer": True, "cost": self.client.cost(),
        })

    def test(self):
        candidate = load(self.root / "methods" / self.method / "SELECTED.json")["candidate"]
        result = self.batch(self.rows, candidate, label="complete-test-population")
        write(self.root / "methods" / self.method / "TEST_COMPLETE.json", {
            "metric": result["metric"], "cost": self.client.cost(),
            "membership_sha256": self.config["input_membership_sha256"],
            "candidate_sha256": digest(candidate),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=("admission", "train", "validation", "test"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load(root / "frozen/config.json")
    assert_remote(config, root)
    verify_frozen(root)
    directory = root / "methods" / args.method / args.stage
    try:
        study = Study(root, config, args.method, args.stage)
        getattr(study, args.stage)()
        write(directory / "COMPLETE.json", {"at_unix": time.time(), "cost": study.client.cost()})
    except BaseException as error:
        write(directory / "FAILED.json", {
            "at_unix": time.time(), "error": repr(error), "traceback": traceback.format_exc(),
            "retry_permitted_without_new_diagnosis": False})
        raise


if __name__ == "__main__":
    main()
