"""Exact-once native response continuation and explicit interrupted-call costs."""
import gzip
import ast
import json
from pathlib import Path
import shutil

from scripts.prepare_chisel_mmmu_shared_202060 import canonical, digest, load, sha_file, write

IDENTITY_FIELDS = (
    "repository", "revision", "served_name", "model_path", "max_model_len",
    "max_tokens", "optimizer_max_tokens", "seed", "enable_thinking", "dtype",
)


def record_resumed_rollout(path, value, *, optimizer_root, prior_optimizer_root):
    path = Path(path)
    if not path.exists():
        write(path, value)
        return "NEW"
    if load(path) == value:
        return "IDENTICAL"
    if prior_optimizer_root is None:
        raise RuntimeError("A current-run optimizer trajectory cannot be overwritten")
    relative = path.resolve().relative_to(Path(optimizer_root).resolve())
    prior = Path(prior_optimizer_root) / relative
    if not prior.is_file() or sha_file(path) != sha_file(prior):
        raise RuntimeError("Only an unchanged copy of an identified predecessor trajectory may be regenerated")
    evidence = Path(optimizer_root).parent / "regenerated-predecessor-trajectories" / relative
    write(evidence, {
        "original_source": str(prior), "original_sha256": sha_file(prior), "original_value": load(prior),
        "new_value_sha256": digest(value),
        "reason": "Fresh current-run native output after changed resource/payload identity; not reuse of the old observation",
    })
    path.write_bytes(canonical(value) + b"\n")
    return "REGENERATED_WITH_ORIGINAL_PRESERVED"


def code_components(path, names):
    tree = ast.parse(Path(path).read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                found[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    key = node.name + "." + getattr(member, "name", "")
                    if key in names:
                        found[key] = ast.dump(member, include_attributes=False)
    if set(found) != set(names):
        raise RuntimeError("A required actor/scorer code component is absent")
    return found


def inherit_completed_training(root, config):
    source = Path(config["completed_training_source"])
    methods = config["inherit_completed_training_methods"]
    if set(methods) != {"no_skill", "skillopt", "chisel"} or len(methods) != 3:
        raise RuntimeError("This diagnosed continuation inherits exactly the three unaffected methods")
    terminal = load(source / "PARTIAL_TRAIN_COMPLETION.json")
    if set(terminal["method_failures"]) != {"autosaddler"} or not (source / "RETIREMENT.json").is_file():
        raise RuntimeError("A different predecessor failure requires its own diagnosis; do not retry it silently")
    old = load(source / "frozen/config.json")
    if old["input_membership_sha256"] != config["input_membership_sha256"]:
        raise RuntimeError("Inherited method data membership changed")
    model_fields = (
        "repository", "revision", "model_path", "runtime", "served_name", "dtype",
        "max_tokens", "max_model_len", "temperature", "top_p", "seed", "enable_thinking")
    if any(old["model"][key] != config["model"][key] for key in model_fields):
        raise RuntimeError("Inherited actor configuration changed")
    for relative in (
            "source/phiagent/learning/supported_local_decision.py",
            "source/scripts/chisel_mmmu_shared_adapter.py", "source/scripts/cvpr_mmmu_campaign.py"):
        if sha_file(source / relative) != sha_file(root / relative):
            raise RuntimeError("Inherited method actor/guard/prompt code changed: " + relative)
    names = ("NativeClient", "read_rows", "Study.read_labels", "Study.score",
             "Study.infer", "Study.batch", "Study.test")
    relative = "source/scripts/run_chisel_mmmu_shared.py"
    before, after = code_components(source / relative, names), code_components(root / relative, names)
    if before != after:
        raise RuntimeError("Inherited method inference/scorer execution differs")
    scorers = list((source / "official/MMMU").rglob("*.py"))
    if not scorers:
        raise RuntimeError("Inherited training lacks a bound official scorer")
    for path in scorers:
        relative = path.relative_to(source)
        if sha_file(path) != sha_file(root / relative):
            raise RuntimeError("Inherited official scorer changed")
    if sha_file(source / "frozen/OPTIMIZER_FEEDBACK_SPLIT.json") != sha_file(root / "frozen/OPTIMIZER_FEEDBACK_SPLIT.json"):
        raise RuntimeError("Inherited optimizer fit/internal-feedback groups changed")
    origins = {}
    for method in methods:
        if old["methods"][method] != config["methods"][method] or terminal["phases"][method] != "train_complete":
            raise RuntimeError("An inherited method was changed or did not finish TRAIN")
        base = source / "methods" / method
        bank = load(base / "CANDIDATES.json")
        complete = load(base / "train/COMPLETE.json")
        if (not bank["training_complete"] or bank["validation_seen"] or bank["test_seen"]
                or bank["source_sha256"] != sha_file(source / "SOURCE_SHA256.json")
                or complete["cost"]["failed"] or complete["cost"]["unknown_usage_attempts"]
                or complete["cost"]["pending_or_failed_attempts"]):
            raise RuntimeError("Inherited TRAIN bank lacks complete source/cost/boundary evidence")
        target = root / "methods" / method
        target.mkdir(parents=True)
        shutil.copyfile(base / "CANDIDATES.json", target / "CANDIDATES.json")
        receipt = {
            "source_run": str(source), "source_manifest_sha256": sha_file(source / "SOURCE_SHA256.json"),
            "candidate_bank_sha256": sha_file(base / "CANDIDATES.json"),
            "train_complete_sha256": sha_file(base / "train/COMPLETE.json"),
            "actor_and_scorer_components_sha256": digest(before),
            "native_method_configuration_equal": True, "training_bank_copied_without_relabeling": True,
            "new_model_calls": 0, "new_scorer_calls": 0,
        }
        origins[method] = receipt
        write(target / "train/COMPLETE.json", {
            "inherited_completed_training": True, "origin": receipt, "cost": complete["cost"],
            "new_model_calls": 0, "new_scorer_calls": 0})
        write(target / "train/COST.json", {
            "token_usage_mode": "unlimited", "maximum_aggregate_tokens": None,
            "requests": 0, "new_completed_provider_calls": 0, "reused_completed_provider_calls": 0,
            "by_role": {}, "distinct_question_ids": [], "pending_or_failed_attempts": {},
            "unknown_usage_attempts": 0, "failed": False,
            "inherited_complete_training": True, "original_cost": complete["cost"],
            "original_cost_source": str(base / "train/COMPLETE.json"),
        })
    write(root / "frozen/INHERITED_TRAIN_BANKS.json", origins)


def prepare_continuation(root, config):
    from scripts.run_chisel_mmmu_shared import unpack_request
    expected = load(Path(config["input_root"]) / "frozen/SPLIT_MANIFEST.json")
    train_ids = set(expected["splits"]["train"])
    known, interrupted, cache, resources = [], [], [], []
    admission_source = None
    seen_origins = set()
    for value in config.get("continuation_roots", []):
        prior = Path(value).resolve()
        if prior.parent != Path(config["run_namespace"]):
            raise RuntimeError("Continuation root escapes the authorized campaign namespace")
        old = load(prior / "frozen/config.json")
        if (old["input_membership_sha256"] != config["input_membership_sha256"]
                or any(old["model"][key] != config["model"][key] for key in IDENTITY_FIELDS)
                or old["benchmark"]["label_authority"]["selected"] != "hf-raw"):
            raise RuntimeError("Prior response identity differs from the matched native protocol")
        retirement = load(prior / "RETIREMENT.json")
        for selection in prior.glob("services/*/GPU_SELECTION.json"):
            lease = load(selection)
            resources.append({
                "root": str(prior), "physical_index": lease["physical_index"],
                "gpu_seconds": retirement["at_unix"] - lease["at_unix"],
                "gpu_selection_sha256": sha_file(selection),
                "retirement_sha256": sha_file(prior / "RETIREMENT.json"),
            })
        for metadata_path in sorted(prior.glob("methods/*/*/requests/*/metadata.json")):
            directory = metadata_path.parent
            metadata = load(metadata_path)
            method, stage = metadata["method"], metadata["stage"]
            if stage not in {"train", "admission"}:
                raise RuntimeError("This continuation cannot consume prior validation/test outcomes")
            if not set(metadata["question_ids"]).issubset(train_ids):
                raise RuntimeError("Continuation inference crossed the immutable training boundary")
            record_path = directory / "record.json"
            saved_record = load(record_path) if record_path.exists() else None
            origin = saved_record.get("provider_origin", str(directory)) if saved_record else str(directory)
            if origin in seen_origins:
                if saved_record is None or not saved_record.get("provider_reused"):
                    raise RuntimeError("A prior physical provider call was counted twice without a reuse receipt")
                continue
            seen_origins.add(origin)
            response = directory / "model.response.json.gz"
            request = directory / "model.request.json.gz"
            if response.exists() and request.exists():
                raw = json.loads(gzip.decompress(response.read_bytes()))
                packed = json.loads(gzip.decompress(request.read_bytes()))
                wire = unpack_request(packed, prior / "request-image-assets" / method)
                proof = load(directory / "native-proof.json")
                usage = raw.get("usage", {})
                if (digest(wire) != metadata["request_sha256"]
                        or raw.get("model") != config["model"]["served_name"]
                        or usage.get("prompt_tokens") != proof["prompt_tokens"]
                        or any(type(usage.get(key)) is not int or usage[key] < 0
                               for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                        or usage["completion_tokens"] > proof["maximum_output_tokens"]):
                    raise RuntimeError("Historical native wire response has inconsistent identity/usage")
                content = raw["choices"][0]["message"].get("content")
                if content is None and raw["choices"][0]["message"].get("tool_calls"):
                    content = ""
                if not isinstance(content, str):
                    raise RuntimeError("Historical response has neither content nor native tool calls")
                if not record_path.exists():
                    recovered = {
                        **metadata, "raw_content": content,
                        "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                        "native_image_tokens": proof["actual_image_tokens"],
                        "finish_reason": raw["choices"][0].get("finish_reason"),
                        "finished_at_unix": None, "recovered_terminal_wire_response": True,
                        "request_directory": str(directory.relative_to(prior)),
                    }
                    record_path = root / "provenance/recovered-records" / prior.name / method / stage / (directory.name + ".json")
                    write(record_path, recovered)
                record = load(record_path)
                if record["raw_content"] != content:
                    raise RuntimeError("Saved response differs from the unmodified native wire response")
                item = {
                    "method": method, "stage": stage, "role": metadata["role"],
                    "physical_gpu": metadata["physical_gpu"],
                    "hostname": old["hostname"],
                    "physical_uuid": next(
                        slot["uuid"] for slot in old["resources"]["slots"]
                        if slot["index"] == metadata["physical_gpu"]),
                    "request_sha256": metadata["request_sha256"], "provider_origin": origin,
                    "prior_record_path": str(record_path), "prior_record_sha256": sha_file(record_path),
                    "prior_response_path": str(response), "prior_response_sha256": sha_file(response),
                    "usage": record["usage"], "prior_source_manifest_sha256": sha_file(prior / "SOURCE_SHA256.json"),
                }
                known.append(item)
                if stage == "train":
                    cache.append(item)
            elif request.exists():
                proof = load(directory / "native-proof.json")
                error_path = directory / "model.http-error.bin"
                interrupted.append({
                    "method": method, "stage": stage, "provider_origin": origin,
                    "native_reported_tokens": None, "possible_token_lower_bound": 0,
                    "possible_token_upper_bound": proof["prompt_tokens"] + proof["maximum_output_tokens"],
                    "native_prompt_tokens": proof["prompt_tokens"],
                    "original_metadata_sha256": sha_file(metadata_path),
                    "status": "INTERRUPTED_NO_PROVIDER_USAGE; never count as zero or a completed result",
                    "http_error_body": error_path.read_text() if error_path.exists() else None,
                    "http_error_sha256": sha_file(error_path) if error_path.exists() else None,
                })
        if ((prior / "NATIVE_ADMISSION_COMPLETE.json").exists()
                and not (prior / "INHERITED_NATIVE_ADMISSION.json").exists()):
            if admission_source is not None:
                raise RuntimeError("More than one admission source would duplicate the native gate")
            admission_source = prior
    for supplement in config.get("additional_setup_cost_receipts", []):
        receipt_path = Path(supplement["path"])
        if sha_file(receipt_path) != supplement["sha256"]:
            raise RuntimeError("An additional native setup-cost receipt changed")
        receipt = load(receipt_path)
        synthetic = receipt["status"] == "NATIVE_JSON_OBJECT_TRANSPORT_WITH_OFFICIAL_VALIDATION_PASS"
        real_evolve = receipt["status"] == "REAL_TRAIN_OFFICIAL_EVOLVE_AND_TOOL_SESSION_PASS"
        if (not (synthetic or real_evolve) or receipt["cost"]["unknown_usage_attempts"]
                or receipt["cost"]["failed"]
                or (synthetic and (receipt["model_calls"] != 1 or receipt["benchmark_data_loaded"]))
                or (real_evolve and (receipt["external_validation_or_test_opened"]
                                    or not receipt["original_official_schema_and_semantic_validator_passed"]))):
            raise RuntimeError("Setup-cost receipt is incomplete or misclassified")
        records = [receipt["record"]] if synthetic else [
            load(path) for path in sorted(receipt_path.parent.glob("methods/autosaddler/train/requests/*/record.json"))]
        if len(records) != receipt["cost"]["new_completed_provider_calls"]:
            raise RuntimeError("Native capability receipt does not cover every original provider call")
        for record in records:
            directory = receipt_path.parent / record["request_directory"]
            raw = json.loads(gzip.decompress((directory / "model.response.json.gz").read_bytes()))
            if any(raw["usage"][field] != record["usage"][field]
                   for field in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise RuntimeError("Capability receipt disagrees with its actual provider response")
            if record["provider_origin"] in seen_origins:
                raise RuntimeError("Capability provider call was counted twice")
            seen_origins.add(record["provider_origin"])
            known.append({
                "method": record["method"], "stage": record["stage"], "role": record["role"],
                "provider_origin": record["provider_origin"], "usage": record["usage"],
                "setup_capability_only": synthetic, "real_train_evolve_checkpoint": real_evolve,
                "receipt_sha256": supplement["sha256"],
                "prior_response_path": str(directory / "model.response.json.gz"),
                "prior_response_sha256": sha_file(directory / "model.response.json.gz"),
            })
    write(root / "frozen/RESUMABLE_PROVIDER_RECORDS.json", cache)
    write(root / "provenance/INHERITED_ACCOUNTING.json", {
        "known_physical_calls": known, "interrupted_calls": interrupted,
        "reported_tokens": sum(item["usage"]["total_tokens"] for item in known),
        "interrupted_token_upper_bound": sum(item["possible_token_upper_bound"] for item in interrupted),
        "interrupted_usage_is_unknown_not_zero": True, "resource_allocations": resources,
        "exact_once_origin_keys": True,
    })
    if admission_source is not None:
        phase = Path("methods/no_skill/admission")
        shutil.copytree(admission_source / phase, root / phase)
        for name in ("FIRST_MODEL_OUTPUT.json", "FIRST_NATIVE_METRIC.json", "NATIVE_ADMISSION_COMPLETE.json"):
            shutil.copyfile(admission_source / name, root / name)
        for path in (root / phase).glob("requests/*/record.json"):
            record = load(path)
            previous = admission_source / path.relative_to(root)
            record.update(provider_reused=True, provider_origin=str(previous.parent),
                          prior_record_sha256=sha_file(previous),
                          prior_source_manifest_sha256=sha_file(admission_source / "SOURCE_SHA256.json"))
            path.write_bytes(canonical(record) + b"\n")
        first_path = root / "FIRST_MODEL_OUTPUT.json"
        first = load(first_path)
        first.update(provider_reused=True, provider_origin=str(admission_source / first["request_directory"]))
        first_path.write_bytes(canonical(first) + b"\n")
        cost_path = root / phase / "COST.json"
        cost = load(cost_path)
        cost.update(new_completed_provider_calls=0, reused_completed_provider_calls=cost["requests"])
        cost_path.write_bytes(canonical(cost) + b"\n")
        shutil.copytree(admission_source / "request-image-assets/no_skill",
                        root / "request-image-assets/no_skill")
        write(root / "INHERITED_NATIVE_ADMISSION.json", {
            "prior_root": str(admission_source),
            "prior_admission_sha256": sha_file(admission_source / "NATIVE_ADMISSION_COMPLETE.json"),
            "prior_source_manifest_sha256": sha_file(admission_source / "SOURCE_SHA256.json"),
            "new_model_calls": 0, "already_counted_in_inherited_accounting": True,
        })
