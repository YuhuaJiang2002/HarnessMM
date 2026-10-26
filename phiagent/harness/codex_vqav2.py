"""Question/image-only VQAv2 adapter for the official Codex CLI, not a model loop.

The trusted parent runtime implements ``runner(CodexVQARequest) -> Mapping``.
``make_common_runner`` delegates to ``codex_runtime.run_codex_trial`` with a new
output directory, the question prompt, one staged image and at most one residual.
The common runtime owns Docker, Codex 0.142.5, native Responses configuration and
the ephemeral CODEX_PROVIDER_TOKEN. Only the host-side wrapper receives the relay
secret path; no hosted-provider credential or evaluator asset enters Codex.
The metering proxy preserves request/response bodies and tool schemas byte for
byte. Attribution uses ``<provider_url>/<trial_id>``, outside the prompt. The
common ``codex_config`` is authoritative; this adapter has no TOML renderer,
direct completion client, subprocess loop or discretionary token ceiling.
``common_runtime_contract`` builds the matched public runtime metadata, using a
normalized trial URL for configuration hashing and an explicit timeout_seconds.

The callback returns ``final_answer``, ``exit_code``, ``cli_version``,
``runtime_sha256``, ``trial_id``, and ``provider_usage``. Native usage has source
``provider_native`` and input_tokens, output_tokens, cache_read_input_tokens,
total_tokens. Native Responses ``input_tokens_details.cached_tokens`` is also
accepted directly. cache_miss_input_tokens is optional and is never inferred.
Missing required accounting remains unresolved; Codex estimates are not native.
The common wrapper also returns ``common_receipt_path`` for controller APIs that
consume the parent's durable COMPLETED/BLOCKED receipt, rather than this adapter's
normalized envelope. Pass official_vqa_accuracy_01 as their native score; no
threshold-derived success flag is supplied. Provider sampling support and the
executor's seed metadata remain explicit provenance, not a claim of seeded sampling.

The preparer's ``HANDOFF.json`` (``phiagent-vqav2-assets/v1``) is accepted
directly; its pinned partition, image inventory and scorer contract are read as
metadata, without opening any split questions, answers, or image bytes.
Alternatively, asset manifest schema ``codex-vqav2-assets-v1`` contains benchmark, split_seed,
independent_unit="image", split_ratio=[20,20,60], and splits keyed by train,
validation, test. Each split has image_units, question_ids, questions and
annotations; the last two are {"path": relative_path, "sha256": digest}.
``images`` maps image unit IDs to the same file descriptors. ``official_scorer``
is the descriptor for the pinned upstream vqaEval.py. Annotations must be under
evaluator_only/. Questions use the preparer's split paths, or questions/<split>.json;
annotations use evaluator_only/<split>[_annotations].json. Image filenames equal
their image unit IDs plus .jpg. The manifest contains IDs and hashes, never
questions or labels, and resides at the asset root or outside the assets.
The partition is the preparer's SHA256 ordering with largest-remainder rounding,
not an official VQA split. Every question in the selected split is evaluated.

Test access requires a separately frozen, SHA256-pinned JSON gate containing
schema="codex-vqav2-policy-freeze-v1", frozen=true, training_split="train",
selection_split="validation", benchmark, split_seed, manifest_sha256,
runtime_sha256, source_sha256 (source paths to digests), and policies
{"no_skill": SHA256(empty bytes), "chisel": SHA256(selected residual bytes)}.
The gate must exist outside the assets before any test question, annotation, or
image is opened. Source hashes include this adapter and the callback's defining
source, plus caller-supplied controller/runtime sources. No automatic freezing
or selection from test results is provided.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit


CODEX_VERSION = "0.142.5"
MODEL = "deepseek-flash"
PROVIDER_TRANSPORT = "byte-preserving-metering-proxy"
NORMALIZED_PROVIDER_BASE = "http://127.0.0.1:1/TRIAL_ID"
BENCHMARK = "vqav2-val2014"
SPLIT_SEED = 20260928
SPLITS = ("train", "validation", "test")
ARMS = ("no_skill", "chisel")
REMOTE_HOSTS = frozenset({"a800-4", "harnessmm-api-worker"})
DEFAULT_ASSETS = Path("/opt/phiagent/work/deepseek-harness/20260929-v1/assets/vqav2")
PILOT_ROOT = Path("/opt/phiagent/work/codex-chisel/20260929-v1/vqav2/train/baseline/r0")
OFFICIAL_REPOSITORY = "https://github.com/GT-Vision-Lab/VQA"
OFFICIAL_REVISION = "a013f0043c1e2cdc995922dfe257f7149aa9af06"
OFFICIAL_SOURCE_PATH = "PythonEvaluationTools/vqaEvaluation/vqaEval.py"
OFFICIAL_SOURCE_SHA256 = "f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8"
PORTED_SOURCE_SHA256 = "ba95df786a47077b046d9e5715dcfdaa32159c15fb6606b242f13e808e3b6f93"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PROMPT_TEMPLATE = (
    "Answer the question using the attached image.\n"
    "Use only this image and question; do not retrieve reference answers or dataset annotations.\n"
    "Return only the exact short answer, with no explanation or surrounding formatting.\n\n"
    "Question: {question}\n"
)
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_miss_input_tokens",
    "total_tokens",
)
REQUIRED_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "total_tokens")
PREPARED_QUESTION_PATHS = {
    "train": "optimizer/train.jsonl",
    "validation": "runner_only/validation.jsonl",
    "test": "sealed/test.jsonl",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _positive_id(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer {'>= 0' if allow_zero else '> 0'}")
    return value


def image_unit_id(image_id: int) -> str:
    return f"COCO_val2014_{_positive_id(image_id, 'image_id'):012d}"


def require_remote() -> None:
    if platform.node() not in REMOTE_HOSTS:
        raise RuntimeError("VQAv2/Codex execution and official scoring are remote-only on a800-4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for this CPU/provider-only adapter")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _read_pinned_json(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(expected_sha256, str(path)):
        raise ValueError(f"SHA256 mismatch: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class VQAQuestion:
    question_id: int
    image_id: int
    question: str

    def __post_init__(self) -> None:
        _positive_id(self.question_id, "question_id", allow_zero=True)
        _positive_id(self.image_id, "image_id")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("Question text must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> VQAQuestion:
        if set(record) != {"question_id", "image_id", "question"}:
            raise ValueError("Only question_id, image_id and question are agent-visible")
        return cls(**record)

    @property
    def independent_group(self) -> str:
        return image_unit_id(self.image_id)


def validate_runtime(runtime: Mapping[str, Any]) -> dict:
    """Whitelist reproducibility metadata; never copy runtime secrets into requests."""
    fields = {
        "engine", "cli_version", "model", "binary_sha256", "configuration_sha256",
        "container_image", "isolation", "credential_free", "cumulative_token_limit",
        "wire_api", "transport",
        "timeout_seconds",
    }
    if set(runtime) != fields:
        raise ValueError(f"Runtime contract must contain exactly {sorted(fields)}")
    if runtime["engine"] != "official-codex-cli" or runtime["cli_version"] != CODEX_VERSION:
        raise ValueError(f"Official Codex CLI {CODEX_VERSION} is required")
    if runtime["isolation"] != "docker" or runtime["credential_free"] is not True:
        raise ValueError("Docker isolation without hosted-provider credentials is required")
    if runtime["cumulative_token_limit"] is not None:
        raise ValueError("Cumulative token accounting must be unlimited (null)")
    if runtime["model"] != MODEL:
        raise ValueError(f"The matched native Responses model must be {MODEL}")
    if runtime["wire_api"] != "responses" or runtime["transport"] != PROVIDER_TRANSPORT:
        raise ValueError("Use native Responses with byte-preserving metering, without schema conversion")
    timeout = runtime["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Freeze a positive, finite per-trial timeout_seconds")
    if not isinstance(runtime["container_image"], str) or re.search(
        r"(?:^|@)sha256:[0-9a-f]{64}$", runtime["container_image"]
    ) is None:
        raise ValueError("Pin the Docker image by digest, not a mutable tag")
    _digest(runtime["binary_sha256"], "binary_sha256")
    _digest(runtime["configuration_sha256"], "configuration_sha256")
    return dict(runtime)


def validate_provider_url(value: str) -> str:
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise ValueError("Provide a credential-free local metering endpoint URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid metering endpoint URL") from error
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment or port == 0
    ):
        raise ValueError("Metering URLs must not contain credentials, query parameters or fragments")
    host = parsed.hostname
    try:
        address = ipaddress.ip_address(host)
        local = address.is_loopback or address.is_private
    except ValueError:
        local = "." not in host or host.endswith((".internal", ".localhost", ".local"))
    if not local:
        raise ValueError("Use the local metering proxy, not a public provider endpoint")
    return value


def validate_trial_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", value) is None:
        raise ValueError("Trial attribution must be a nonempty, header-safe identifier")
    return value


def validate_policy(arm: str, residual: str) -> str:
    if arm not in ARMS or not isinstance(residual, str):
        raise ValueError("Choose no_skill or chisel and a single residual string")
    if arm == "no_skill" and residual:
        raise ValueError("The no_skill arm must have an empty residual")
    return hashlib.sha256(residual.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CodexVQARequest:
    """No dataset root, annotations, credentials, or other questions are passed here."""

    question_id: int
    image_id: int
    independent_group: str
    workspace: Path
    output_dir: Path
    prompt: str
    runtime: Mapping[str, Any]
    provider_url: str
    trial_id: str
    image_sha256: str
    question_sha256: str
    policy_sha256: str
    has_residual: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", MappingProxyType(validate_runtime(self.runtime)))
        validate_provider_url(self.provider_url)
        validate_trial_id(self.trial_id)

    @property
    def runtime_sha256(self) -> str:
        return canonical_sha256(dict(self.runtime))

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return (self.workspace / "image.jpg",)

    @property
    def container_image_paths(self) -> tuple[str, ...]:
        return ("/workspace/image.jpg",)

    @property
    def provider_base_url(self) -> str:
        return f"{self.provider_url.rstrip('/')}/{self.trial_id}"

    @property
    def codex_args(self) -> tuple[str, ...]:
        """Optional custom-runner arguments; the common runtime builds its own command."""
        return (
            "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-rules",
            "--config", 'approval_policy="never"', "--sandbox", "read-only", "--cd", "/workspace",
            "--model", self.runtime["model"], "-i", "/workspace/image.jpg",
            "--output-last-message", "/output/final-answer.txt", "-",
        )

    @property
    def data_mounts(self) -> tuple[tuple[str, str, str], ...]:
        return (
            (str(self.workspace), "/workspace", "ro"),
            (str(self.output_dir), "/output", "rw"),
        )


def prepare_request(
    question: VQAQuestion,
    image_path: Path,
    image_sha256: str,
    task_dir: Path,
    *,
    runtime: Mapping[str, Any],
    provider_url: str,
    trial_id: str,
    arm: str,
    residual: str = "",
) -> CodexVQARequest:
    """Pure file staging helper, also usable with synthetic unit-test images."""
    runtime = validate_runtime(runtime)
    validate_provider_url(provider_url)
    validate_trial_id(trial_id)
    policy_sha256 = validate_policy(arm, residual)
    image_path = Path(image_path)
    if image_path.is_symlink() or file_sha256(image_path) != _digest(image_sha256, "image"):
        raise ValueError("Image is a symlink or its SHA256 does not match the manifest")
    task_dir = Path(task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=False)
    workspace, output_dir = task_dir / "workspace", task_dir / "output"
    workspace.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    shutil.copyfile(image_path, workspace / "image.jpg")
    if file_sha256(workspace / "image.jpg") != image_sha256:
        raise ValueError("Image changed while staging the isolated workspace")
    question_bytes = question.question.encode("utf-8")
    (workspace / "question.txt").write_bytes(question_bytes)
    if residual:
        (workspace / "AGENTS.md").write_text(residual, encoding="utf-8")
    request = CodexVQARequest(
        question.question_id, question.image_id, question.independent_group, workspace, output_dir,
        PROMPT_TEMPLATE.format(question=question.question), runtime, provider_url, trial_id, image_sha256,
        hashlib.sha256(question_bytes).hexdigest(), policy_sha256, bool(residual),
    )
    validate_workspace(request)
    return request


def validate_workspace(request: CodexVQARequest) -> None:
    expected = {"question.txt", "image.jpg"} | ({"AGENTS.md"} if request.has_residual else set())
    files = list(request.workspace.iterdir())
    if {path.name for path in files} != expected or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise ValueError("The agent workspace must contain only this question, image and residual")
    for name, expected_digest in (
        ("question.txt", request.question_sha256), ("image.jpg", request.image_sha256),
    ):
        if file_sha256(request.workspace / name) != expected_digest:
            raise ValueError(f"Agent-visible input changed: {name}")
    if request.has_residual and file_sha256(request.workspace / "AGENTS.md") != request.policy_sha256:
        raise ValueError("Agent-visible residual changed")


def provider_usage(record: Mapping[str, Any] | None) -> dict:
    """Preserve missing native counters as unknown, not zeros or token estimates."""
    result = {field: None for field in USAGE_FIELDS}
    result.update(status="unresolved", source="provider_native")
    if not isinstance(record, Mapping) or record.get("source") != "provider_native":
        result["reason"] = "Provider-native usage was not supplied; Codex estimates are not native"
        return result
    for field in USAGE_FIELDS:
        value = record.get(field)
        if field == "cache_read_input_tokens":
            details = record.get("input_tokens_details")
            if details is not None:
                if not isinstance(details, Mapping):
                    raise ValueError("Invalid native input_tokens_details")
                cached = details.get("cached_tokens")
                if cached is not None and (type(cached) is not int or cached < 0):
                    raise ValueError("Invalid native cached_tokens counter")
                if value is not None and cached is not None and value != cached:
                    raise ValueError("Conflicting native cache counters")
                value = cached if value is None else value
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Invalid native provider usage counter: {field}")
        result[field] = value
    inputs, outputs, total = (result[key] for key in (
        "input_tokens", "output_tokens", "total_tokens"
    ))
    if None not in (inputs, outputs, total) and inputs + outputs != total:
        raise ValueError("Native total_tokens must equal input_tokens + output_tokens")
    hits, misses = result["cache_read_input_tokens"], result["cache_miss_input_tokens"]
    if inputs is not None and any(value is not None and value > inputs for value in (hits, misses)):
        raise ValueError("Native cache counters cannot exceed native input tokens")
    if None not in (inputs, hits, misses) and hits + misses != inputs:
        raise ValueError("DeepSeek cache hit/miss tokens must partition input tokens")
    if (
        all(result[field] is not None for field in REQUIRED_USAGE_FIELDS)
        and record.get("status") == "resolved"
    ):
        result["status"] = "resolved"
    else:
        result["reason"] = str(record.get("reason") or "Incomplete provider-native usage accounting")
    return result


def aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict:
    if not records:
        return provider_usage(None)
    values = [provider_usage(record) for record in records]
    result = {
        field: (sum(record[field] for record in values)
                if all(record[field] is not None for record in values) else None)
        for field in USAGE_FIELDS
    }
    result.update(
        source="provider_native",
        status="resolved" if all(record["status"] == "resolved" for record in values) else "unresolved",
    )
    if result["status"] == "unresolved":
        result["reason"] = "At least one task has unresolved provider-native usage"
    return result


def attributed_usage(request: CodexVQARequest, response: Mapping[str, Any]) -> dict:
    if response.get("trial_id") != request.trial_id:
        return provider_usage({
            "source": "provider_native", "status": "unresolved",
            "reason": "The callback did not attribute usage to this trial",
        })
    return provider_usage(response.get("provider_usage"))


def common_provider_usage(summary: Mapping[str, Any] | None, trial_id: str) -> dict:
    """Adapt the metering ledger, never native_cli_usage or an incomplete lower bound."""
    reason = "The common runtime did not return complete, trial-matched provider accounting"
    if (
        not isinstance(summary, Mapping) or summary.get("trial_id") != trial_id
        or summary.get("usage_status") != "MEASURED"
        or type(summary.get("calls")) is not int or summary["calls"] <= 0
        or summary.get("unresolved_call_ids") != []
        or not isinstance(summary.get("known_usage"), Mapping)
    ):
        return provider_usage({"source": "provider_native", "status": "unresolved", "reason": reason})
    return provider_usage({
        "source": "provider_native", "status": "resolved",
        **{field: summary["known_usage"].get(field) for field in (
            "input_tokens", "output_tokens", "total_tokens",
        )},
        "cache_read_input_tokens": summary.get("cached_input_tokens"),
    })


def common_runtime_contract(
    *, container_image: str | None = None, timeout_seconds: float = 1200,
) -> dict:
    """Configuration-only helper: imports the common runtime but executes no trial."""
    common = importlib.import_module("phiagent.harness.codex_runtime")
    if common.CODEX_VERSION != f"codex-cli {CODEX_VERSION}":
        raise ValueError("The common runtime must pin official Codex 0.142.5")
    return validate_runtime({
        "engine": "official-codex-cli", "cli_version": CODEX_VERSION, "model": MODEL,
        "binary_sha256": common.CODEX_SHA256,
        "configuration_sha256": hashlib.sha256(
            common.codex_config(NORMALIZED_PROVIDER_BASE).encode()
        ).hexdigest(),
        "container_image": container_image or common.BASE_IMAGE,
        "isolation": "docker", "credential_free": True, "cumulative_token_limit": None,
        "wire_api": "responses", "transport": PROVIDER_TRANSPORT,
        "timeout_seconds": timeout_seconds,
    })


def make_common_runner(
    *, relay_secret_file: Path, codex_binary: Path,
    trial_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[CodexVQARequest], Mapping[str, Any]]:
    """Bind host-only inputs; trial_runner injection is for pure synthetic helper tests."""
    common = None
    dependencies = []
    if trial_runner is None:
        common = importlib.import_module("phiagent.harness.codex_runtime")
        trial_runner = common.run_codex_trial
        dependencies.append(inspect.getsourcefile(common.read_credential))
    callback_source = inspect.getsourcefile(trial_runner)
    dependencies.append(callback_source)
    if any(path is None for path in dependencies):
        raise ValueError("Common runtime and metering dependencies must have hashable Python sources")
    relay_secret_file, codex_binary = Path(relay_secret_file).resolve(), Path(codex_binary).resolve()

    def run(request: CodexVQARequest) -> Mapping[str, Any]:
        validate_workspace(request)
        if common is not None:
            expected = common_runtime_contract(
                container_image=request.runtime["container_image"],
                timeout_seconds=request.runtime["timeout_seconds"],
            )
            if dict(request.runtime) != expected:
                raise ValueError("Generate the runtime contract with common_runtime_contract()")
        residual = ((request.workspace / "AGENTS.md").read_bytes().decode("utf-8")
                    if request.has_residual else "")
        output = request.output_dir / "codex"
        if output.exists():
            raise FileExistsError("Every common Codex trial requires a new output directory")
        result = trial_runner(
            output=output, prompt=request.prompt, trial_id=request.trial_id,
            provider_url=request.provider_url, relay_secret_file=relay_secret_file,
            codex_binary=codex_binary, images=request.image_paths, residual=residual,
            timeout_seconds=request.runtime["timeout_seconds"],
            container_image=request.runtime["container_image"],
        )
        if not isinstance(result, Mapping):
            raise TypeError("run_codex_trial must return a result mapping")
        expected_fields = {
            "trial_id": request.trial_id, "codex_version": f"codex-cli {CODEX_VERSION}",
            "codex_binary_sha256": request.runtime["binary_sha256"],
            "container_image": request.runtime["container_image"], "model": MODEL,
            "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
            "residual_sha256": request.policy_sha256,
            "maximum_tokens": None, "token_usage_mode": "unlimited",
        }
        mismatches = [key for key, value in expected_fields.items()
                      if key not in result or result[key] != value]
        images = result.get("images")
        if not isinstance(images, list) or len(images) != 1 or (
            not isinstance(images[0], Mapping) or images[0].get("sha256") != request.image_sha256
        ):
            mismatches.append("images")
        if common is not None:
            expected_config = hashlib.sha256(common.codex_config(request.provider_base_url).encode())
            if result.get("config_sha256") != expected_config.hexdigest():
                mismatches.append("config_sha256")
        provider = result.get("provider_usage")
        usage = common_provider_usage(provider, request.trial_id)
        execution = {key: result.get(key) for key in (
            "status", "returncode", "codex_version", "codex_revision", "codex_binary_sha256", "model",
            "container_image", "config_sha256", "runtime_sha256", "workspace",
            "run_directory", "events_path", "error_type", "error", "seed",
            "provider_sampling_seed_supported",
        )}
        execution.update(
            contract_mismatches=mismatches, runtime_record=str(output / "RUN.json"),
            seed_semantics="paired execution metadata; not an upstream model sampling seed",
        )
        completed = (
            result.get("status") == "COMPLETED" and type(result.get("returncode")) is int
            and result["returncode"] == 0 and not mismatches
        )
        accounting = {key: provider.get(key) for key in (
            "trial_id", "usage_status", "calls", "known_usage", "cached_input_tokens",
            "reasoning_output_tokens", "unresolved_call_ids", "provider_models",
            "model_response_ids", "receipts",
        )} if isinstance(provider, Mapping) else None
        return {
            "exit_code": 0 if completed else 1, "final_answer": result.get("final_answer"),
            "cli_version": CODEX_VERSION if result.get("codex_version") == f"codex-cli {CODEX_VERSION}"
            else result.get("codex_version"),
            "runtime_sha256": request.runtime_sha256, "trial_id": result.get("trial_id"),
            "provider_usage": usage, "provider_accounting": accounting,
            "native_cli_usage": result.get("native_cli_usage"), "execution": execution,
            "common_receipt_path": str(output / "RUN.json"),
            "provider_sampling_seed_supported": result.get("provider_sampling_seed_supported"),
        }

    setattr(run, "source_files", tuple(Path(path).resolve() for path in dependencies))
    setattr(run, "host_only_paths", (relay_secret_file, codex_binary))
    return run


def prediction_from_response(request: CodexVQARequest, response: Mapping[str, Any]) -> dict:
    if response.get("exit_code") != 0 or type(response.get("exit_code")) is not int:
        raise RuntimeError("Codex did not exit successfully; preserve its runtime logs")
    if response.get("cli_version") != CODEX_VERSION:
        raise ValueError("Runtime did not attest the pinned official Codex CLI version")
    if response.get("runtime_sha256") != request.runtime_sha256:
        raise ValueError("Runtime identity differs from the frozen matched runtime")
    if response.get("trial_id") != request.trial_id:
        raise ValueError("Runtime usage/final answer is attributed to a different trial")
    answer = response.get("final_answer")
    if not isinstance(answer, str):
        raise ValueError("Runtime must return the actual Codex final answer string")
    return {
        "question_id": request.question_id,
        "image_id": request.image_id,
        "trial_id": request.trial_id,
        "independent_group": request.independent_group,
        "predicted_answer": answer.strip(),
        "image_sha256": request.image_sha256,
        "provider_usage": attributed_usage(request, response),
        "native_cli_usage": response.get("native_cli_usage"),
        "common_receipt_path": response.get("common_receipt_path"),
        "provider_sampling_seed_supported": response.get("provider_sampling_seed_supported"),
    }


def _descriptor(value: Any, name: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must have exactly path and sha256")
    path = value["path"]
    if not isinstance(path, str) or "\\" in path:
        raise ValueError(f"Invalid relative asset path: {name}")
    parts = PurePosixPath(path)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError(f"Asset path must stay inside the asset root: {name}")
    _digest(value["sha256"], f"{name}.sha256")
    return dict(value)


def validate_manifest(manifest: Mapping[str, Any]) -> dict:
    if manifest.get("schema") != "codex-vqav2-assets-v1" or manifest.get("benchmark") != BENCHMARK:
        raise ValueError("Expected a codex-vqav2-assets-v1 manifest for provisional VQAv2 val2014")
    if type(manifest.get("split_seed")) is not int or manifest["split_seed"] != SPLIT_SEED:
        raise ValueError(f"Freeze split seed {SPLIT_SEED}")
    if manifest.get("independent_unit") != "image" or manifest.get("split_ratio") != [20, 20, 60]:
        raise ValueError("The partition must be 20/20/60 by independent image")
    allowed = {
        "schema", "benchmark", "split_seed", "independent_unit", "split_ratio",
        "splits", "images", "official_scorer",
    }
    if set(manifest) != allowed or not isinstance(manifest["splits"], dict):
        raise ValueError("The asset manifest is metadata-only; unsupported fields are not allowed")
    if set(manifest["splits"]) != set(SPLITS):
        raise ValueError("The manifest must include train, validation and test")
    units, qids = [], []
    for split in SPLITS:
        part = manifest["splits"][split]
        if not isinstance(part, dict) or set(part) != {
            "image_units", "question_ids", "questions", "annotations",
        }:
            raise ValueError(f"Invalid split metadata: {split}")
        if not isinstance(part["image_units"], list) or not part["image_units"]:
            raise ValueError("Every split needs independent image IDs")
        if not isinstance(part["question_ids"], list) or not part["question_ids"]:
            raise ValueError("Every split needs its complete question ID population")
        for unit in part["image_units"]:
            if not isinstance(unit, str) or re.fullmatch(r"COCO_val2014_\d{12}", unit) is None:
                raise ValueError("Invalid independent image unit ID")
            _positive_id(int(unit.rsplit("_", 1)[1]), "image_id")
        for qid in part["question_ids"]:
            _positive_id(qid, "question_id", allow_zero=True)
        units.extend(part["image_units"])
        qids.extend(part["question_ids"])
        question_file = _descriptor(part["questions"], f"{split}.questions")
        annotation_file = _descriptor(part["annotations"], f"{split}.annotations")
        if annotation_file["path"] not in {
            f"evaluator_only/{split}.json", f"evaluator_only/{split}_annotations.json",
        }:
            raise ValueError("Annotations must stay in their own evaluator_only split file")
        if question_file["path"] not in {PREPARED_QUESTION_PATHS[split], f"questions/{split}.json"}:
            raise ValueError("Questions must use their own split's public question path")
    if len(units) != len(set(units)) or len(qids) != len(set(qids)):
        raise ValueError("Image units and question IDs must be disjoint across splits")
    if len(units) < 5:
        raise ValueError("At least five image groups are required for 20/20/60")
    ranked = sorted(units, key=lambda unit: (
        hashlib.sha256(f"{BENCHMARK}\0{SPLIT_SEED}\0{unit}".encode()).hexdigest(), unit,
    ))
    numerators = (len(units), len(units), 3 * len(units))
    counts = [value // 5 for value in numerators]
    order = sorted(range(3), key=lambda index: (-(numerators[index] % 5), index))
    for index in order[:len(units) - sum(counts)]:
        counts[index] += 1
    offset = 0
    for split, count in zip(SPLITS, counts):
        if set(manifest["splits"][split]["image_units"]) != set(ranked[offset:offset + count]):
            raise ValueError("Split differs from the preparer's frozen SHA256 image partition")
        offset += count
    if not isinstance(manifest["images"], dict) or set(manifest["images"]) != set(units):
        raise ValueError("Every independent image needs exactly one pinned asset")
    for unit, descriptor in manifest["images"].items():
        image_file = _descriptor(descriptor, unit)
        image_path = PurePosixPath(image_file["path"])
        if image_path.parts[0] != "images" or image_path.name != unit + ".jpg":
            raise ValueError("Image assets must name their own independent image group under images/")
    scorer = _descriptor(manifest["official_scorer"], "official_scorer")
    if scorer["sha256"] != OFFICIAL_SOURCE_SHA256:
        raise ValueError("The scorer must be the pinned official VQA source, not a substitute")
    return deepcopy(dict(manifest))


def asset_path(root: Path, descriptor: Mapping[str, str]) -> Path:
    """Reject escaping paths and symlinks without opening the asset."""
    descriptor = _descriptor(dict(descriptor), "asset")
    root = root.resolve()
    candidate = root
    for part in PurePosixPath(descriptor["path"]).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Symlinked dataset assets are not accepted")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Dataset asset escapes its root")
    return candidate


def _prepared_descriptor(root: Path, descriptor: Mapping[str, Any]) -> dict:
    """Translate the preparer's absolute, byte-counted descriptors without changing assets."""
    path = Path(descriptor["path"])
    if path.is_absolute():
        try:
            path = path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("Prepared asset is outside the declared assets root") from error
    result = _descriptor({"path": path.as_posix(), "sha256": descriptor["sha256"]}, "prepared asset")
    asset_path(root, result)
    return result


def _read_pinned_jsonl(path: Path, expected_sha256: str) -> list[dict]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(expected_sha256, str(path)):
        raise ValueError(f"SHA256 mismatch: {path}")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSONL objects: {path}")
    return rows


def load_manifest(root: Path, path: Path, expected_sha256: str) -> tuple[dict, dict]:
    """Read only metadata; in particular never open sealed/test.jsonl or metadata-all.jsonl."""
    path = Path(path).resolve()
    if path.is_relative_to(root.resolve()) and len(path.relative_to(root.resolve()).parts) != 1:
        raise PermissionError("A metadata manifest inside assets must be at the root, not a split file")
    value = _read_pinned_json(path, expected_sha256)
    provenance = {"manifest_sha256": expected_sha256, "source_schema": value.get("schema")}
    if value.get("schema") == "codex-vqav2-assets-v1":
        return validate_manifest(value), provenance
    if value.get("schema") != "phiagent-vqav2-assets/v1":
        raise ValueError("Unsupported VQAv2 asset handoff schema")
    if value.get("status") != "WORKING" or value.get("benchmark") != BENCHMARK:
        raise ValueError("Prepared VQAv2 assets are not ready")
    population = value["population"]
    if population.get("status") != "WORKING" or population["images"].get("status") != "WORKING":
        raise ValueError("The complete VQAv2 image population must be ready")
    partition_file = _prepared_descriptor(root, population["partition"])
    if partition_file["path"] != "partition.json":
        raise ValueError("Prepared partition metadata must be partition.json")
    partition = _read_pinned_json(asset_path(root, partition_file), partition_file["sha256"])
    if (
        partition.get("schema") != "phiagent-20-20-60/v2"
        or partition.get("ratios") != {"train": 0.2, "validation": 0.2, "test": 0.6}
        or partition.get("score_or_outcome_used") is not False
        or partition.get("rounding_policy") != "largest_remainder"
        or partition.get("rounding_tie_order") != list(SPLITS)
        or partition.get("custom_partition_not_official_split") is not True
    ):
        raise ValueError("Prepared partition does not follow the frozen image-level protocol")
    inventory_file = _prepared_descriptor(root, population["images"]["inventory"])
    if inventory_file["path"] != "image-inventory.jsonl":
        raise ValueError("Prepared image metadata must be image-inventory.jsonl")
    inventory = _read_pinned_jsonl(asset_path(root, inventory_file), inventory_file["sha256"])
    images = {}
    for row in inventory:
        unit = image_unit_id(row["image_id"])
        if unit in images:
            raise ValueError("Duplicate image in prepared inventory")
        images[unit] = _prepared_descriptor(root, {
            "path": row["image_path"], "sha256": row["sha256"],
        })
    contract_file = _prepared_descriptor(root, value["native_scorer"]["contract"])
    if contract_file["path"] != "scorer/contract.json":
        raise ValueError("Prepared scorer metadata must be scorer/contract.json")
    contract = _read_pinned_json(asset_path(root, contract_file), contract_file["sha256"])
    if (
        value["native_scorer"].get("status") != "WORKING"
        or contract.get("upstream_commit") != OFFICIAL_REVISION
        or contract.get("upstream_source_sha256") != OFFICIAL_SOURCE_SHA256
        or contract.get("ported_source", {}).get("sha256") != PORTED_SOURCE_SHA256
    ):
        raise ValueError("Prepared scorer is not the pinned official syntax-only port")
    splits = {
        name: {
            "image_units": partition["splits"][name],
            "question_ids": partition["question_ids"][name],
            "questions": _prepared_descriptor(root, population["metadata"][name]),
            "annotations": _prepared_descriptor(
                root, population["split_annotations_evaluator_only"][name],
            ),
        }
        for name in SPLITS
    }
    units = {unit for part in splits.values() for unit in part["image_units"]}
    if not units.issubset(images):
        raise ValueError("Prepared inventory is missing VQAv2 image groups")
    normalized = validate_manifest({
        "schema": "codex-vqav2-assets-v1", "benchmark": partition["benchmark"],
        "split_seed": partition["split_seed"], "independent_unit": partition["independent_unit"],
        "split_ratio": [20, 20, 60], "splits": splits,
        "images": {unit: images[unit] for unit in sorted(units)},
        "official_scorer": _prepared_descriptor(root, {
            "path": contract["upstream_source_path"], "sha256": OFFICIAL_SOURCE_SHA256,
        }),
    })
    for split in SPLITS:
        if (
            partition["counts"][split] != len(splits[split]["image_units"])
            or partition["question_counts"][split] != len(splits[split]["question_ids"])
        ):
            raise ValueError("Prepared partition counts disagree with its ID population")
    if (
        partition["population"] != len(units)
        or partition["question_population"] != sum(len(part["question_ids"]) for part in splits.values())
    ):
        raise ValueError("Prepared partition does not account for its complete population")
    provenance.update(
        partition_sha256=partition_file["sha256"], image_inventory_sha256=inventory_file["sha256"],
        scorer_contract_sha256=contract_file["sha256"],
        normalized_manifest_sha256=canonical_sha256(normalized),
    )
    return normalized, provenance


def require_external_config(path: Path, assets_root: Path) -> None:
    """Prevent a mistaken --residual/--runtime-contract argument from opening sealed data."""
    if Path(path).resolve().is_relative_to(Path(assets_root).resolve()):
        raise PermissionError("Runtime configuration, residuals and freeze gates must be outside assets")


def source_hashes(files: Sequence[Path | str]) -> dict[str, str]:
    paths = {Path(__file__).resolve(), *(Path(path).resolve() for path in files)}
    if any(path.suffix != ".py" for path in paths):
        raise ValueError("Source hash inputs must be Python source files, never dataset files")
    return {str(path): file_sha256(path) for path in sorted(paths)}


def authorize_split(
    split: str,
    *,
    assets_root: Path,
    manifest_sha256: str,
    runtime: Mapping[str, Any],
    arm: str,
    residual: str,
    sources: Mapping[str, str],
    freeze_path: Path | None = None,
    freeze_sha256: str | None = None,
) -> dict:
    if split not in SPLITS:
        raise ValueError("Unknown split")
    policy_hash = validate_policy(arm, residual)
    runtime_hash = canonical_sha256(validate_runtime(runtime))
    if split != "test":
        return {"split": split, "test_access": False, "policy_sha256": policy_hash}
    if freeze_path is None or freeze_sha256 is None:
        raise PermissionError("Test files remain sealed until a policy/source freeze is pinned")
    freeze_path = Path(freeze_path).resolve()
    if freeze_path.is_relative_to(Path(assets_root).resolve()):
        raise PermissionError("The frozen policy gate must be outside dataset assets")
    gate = _read_pinned_json(freeze_path, freeze_sha256)
    expected = {
        "schema": "codex-vqav2-policy-freeze-v1", "frozen": True,
        "training_split": "train", "selection_split": "validation",
        "benchmark": BENCHMARK, "split_seed": SPLIT_SEED,
        "manifest_sha256": _digest(manifest_sha256, "manifest_sha256"),
        "runtime_sha256": runtime_hash, "source_sha256": dict(sources),
    }
    if any(gate.get(key) != value for key, value in expected.items()) or gate.get("frozen") is not True:
        raise PermissionError("Frozen policy/source/runtime/partition hashes do not match")
    policies = gate.get("policies")
    if not isinstance(policies, dict) or set(policies) != set(ARMS):
        raise PermissionError("Freeze both no_skill and chisel policies before opening test")
    for value in policies.values():
        _digest(value, "frozen policy")
    if policies["no_skill"] != EMPTY_SHA256 or policies[arm] != policy_hash:
        raise PermissionError("Selected residual differs from the pre-test frozen policy")
    return {
        "split": split, "test_access": True, "policy_sha256": policy_hash,
        "freeze_sha256": freeze_sha256,
    }


def load_questions(root: Path, manifest: Mapping[str, Any], access: Mapping[str, Any]) -> list[VQAQuestion]:
    split = access["split"]
    if split == "test" and access.get("test_access") is not True:
        raise PermissionError("Test questions remain sealed")
    part = manifest["splits"][split]
    descriptor = part["questions"]
    path = asset_path(root, descriptor)
    if path.suffix == ".jsonl":
        rows = _read_pinned_jsonl(path, descriptor["sha256"])
        questions = []
        for row in rows:
            if set(row) != {"question_id", "image_id", "question", "id", "group_id", "split", "image_path"}:
                raise ValueError("Prepared task metadata must not include labels or unsupported fields")
            question = VQAQuestion.from_record({key: row[key] for key in (
                "question_id", "image_id", "question",
            )})
            expected_image = asset_path(root, manifest["images"][question.independent_group])
            if (
                row["id"] != str(question.question_id) or row["group_id"] != question.independent_group
                or row["split"] != split or Path(row["image_path"]) != expected_image
            ):
                raise ValueError("Prepared task metadata disagrees with the frozen question/image IDs")
            questions.append(question)
    else:
        value = _read_pinned_json(path, descriptor["sha256"])
        if not isinstance(value.get("questions"), list):
            raise ValueError("Expected official VQA question JSON with a questions list")
        questions = [VQAQuestion.from_record(row) for row in value["questions"]]
    qids = [question.question_id for question in questions]
    if len(qids) != len(set(qids)) or set(qids) != set(part["question_ids"]):
        raise ValueError("Question population differs from the complete frozen split")
    if {question.independent_group for question in questions} != set(part["image_units"]):
        raise ValueError("Question image groups differ from the frozen split")
    return sorted(questions, key=lambda question: question.question_id)


def python3_scorer_source(source: bytes) -> bytes:
    """Only port the two Python-2 print statements; all metric logic stays upstream."""
    if hashlib.sha256(source).hexdigest() != OFFICIAL_SOURCE_SHA256:
        raise ValueError("Official VQA scorer source checksum mismatch")
    for text in (b"computing accuracy", b"Done computing accuracy"):
        old = b'print "' + text + b'"'
        if source.count(old) != 1:
            raise ValueError("Unexpected upstream scorer print statement")
        source = source.replace(old, b'print("' + text + b'")')
    return source


def first_training_question(manifest: Mapping[str, Any], questions: Sequence[VQAQuestion]) -> VQAQuestion:
    part = manifest["splits"]["train"]
    if {question.question_id for question in questions} != set(part["question_ids"]):
        raise ValueError("Pilot selection requires the complete frozen training question index")
    first_group = part["image_units"][0]
    candidates = [question for question in questions if question.independent_group == first_group]
    if not candidates:
        raise ValueError("The first frozen training image has no questions")
    return min(candidates, key=lambda question: question.question_id)


def validate_annotations(value: Mapping[str, Any], questions: Sequence[VQAQuestion]) -> list[dict]:
    rows = value.get("annotations")
    if not isinstance(rows, list):
        raise ValueError("Expected official VQA annotations JSON")
    expected = {question.question_id: question.image_id for question in questions}
    found = set()
    for row in rows:
        qid = _positive_id(row.get("question_id"), "annotation question_id", allow_zero=True)
        if (qid in found or qid not in expected or type(row.get("image_id")) is not int
                or row.get("image_id") != expected[qid]):
            raise ValueError("Annotation IDs/images differ from the selected question population")
        found.add(qid)
        answers = row.get("answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError("The official metric requires all ten human answer records")
        if any(
            not isinstance(answer, dict) or type(answer.get("answer_id")) is not int
            or not isinstance(answer.get("answer"), str)
            for answer in answers
        ) or {answer["answer_id"] for answer in answers} != set(range(1, 11)):
            raise ValueError("Malformed official human answer records")
        if not isinstance(row.get("question_type"), str) or not isinstance(row.get("answer_type"), str):
            raise ValueError("Missing official question/answer type")
    if found != set(expected):
        raise ValueError("Annotations must cover every question, without exclusions")
    return deepcopy(rows)


class _VQAIndex:
    """Data-only facade for the upstream evaluator's qa/getQuesIds interface."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.qa = {row["question_id"]: deepcopy(dict(row)) for row in rows}

    def getQuesIds(self) -> list[int]:
        return list(self.qa)


class OfficialVQAScorer:
    def __init__(self, source_path: Path) -> None:
        if not Path(source_path).is_file():
            raise FileNotFoundError("Pinned official VQA scorer is unavailable; no approximate fallback")
        original = Path(source_path).read_bytes()
        self._source = python3_scorer_source(original)
        self._code = compile(self._source, f"{OFFICIAL_REPOSITORY}@{OFFICIAL_REVISION}", "exec")
        self.provenance = {
            "repository": OFFICIAL_REPOSITORY, "revision": OFFICIAL_REVISION,
            "source_path": OFFICIAL_SOURCE_PATH, "source_sha256": OFFICIAL_SOURCE_SHA256,
            "python3_compatibility": "parenthesize two print statements only; metric unchanged",
            "executed_source_sha256": hashlib.sha256(self._source).hexdigest(),
            "metric": "official VQAEval.accuracy.overall", "native_scale": "0..100",
            "normalized_scale": "0..1 (native / 100)", "rounding_digits": 2,
        }

    def score(self, questions: Sequence[VQAQuestion], annotations: Mapping[str, Any],
              predictions: Sequence[Mapping[str, Any]], log_path: Path) -> dict:
        require_remote()
        rows = validate_annotations(annotations, questions)
        qids = [question.question_id for question in questions]
        predicted_ids = [row["question_id"] for row in predictions]
        if len(predicted_ids) != len(set(predicted_ids)) or set(predicted_ids) != set(qids):
            raise ValueError("Predictions must cover the complete question population exactly once")
        results = [{"question_id": row["question_id"], "answer": row["predicted_answer"]}
                   for row in predictions]
        namespace: dict[str, Any] = {"__name__": "pinned_official_vqa_eval"}
        exec(self._code, namespace)
        evaluator = namespace["VQAEval"](_VQAIndex(rows), _VQAIndex(results), n=2)
        with Path(log_path).open("x") as stream, redirect_stdout(stream):
            evaluator.evaluate(quesIds=qids)
        percent = _accuracy(evaluator.accuracy["overall"])
        per_question = {qid: _accuracy(evaluator.evalQA[qid]) for qid in qids}
        return {
            "official_vqa_accuracy_percent": percent,
            "official_vqa_accuracy_01": percent / 100,
            "per_question_percent": per_question,
            "official_accuracy_breakdown": evaluator.accuracy,
            "scoring_provenance": self.provenance,
        }


def _accuracy(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Official VQA metric must return a numeric percentage")
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("Official VQA percentage is outside 0..100")
    return float(value)


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
    )
    return {"commit": result.stdout.strip() if result.returncode == 0 else None,
            "exact_source_state": "source_sha256 records include uncommitted source bytes"}


def run_split(
    *,
    assets_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    split: str,
    arm: str,
    residual: str,
    runtime: Mapping[str, Any],
    provider_url: str,
    runner: Callable[[CodexVQARequest], Mapping[str, Any]],
    run_dir: Path,
    training_pilot: bool = False,
    provider_ledger_root: Path | None = None,
    extra_sources: Sequence[Path | str] = (),
    freeze_path: Path | None = None,
    freeze_sha256: str | None = None,
) -> dict:
    """Execute the complete split remotely; never resume/overwrite an existing run."""
    require_remote()
    if training_pilot and (split != "train" or arm != "no_skill" or residual):
        raise ValueError("The first retained pilot is training-only, baseline, with an empty residual")
    if training_pilot and provider_ledger_root is None:
        raise ValueError("Pilot execution requires the provider ledger root for duplicate protection")
    runtime = validate_runtime(runtime)
    provider_url = validate_provider_url(provider_url)
    assets_root = Path(assets_root).resolve()
    for path in getattr(runner, "host_only_paths", ()):
        require_external_config(path, assets_root)
    manifest, asset_provenance = load_manifest(assets_root, Path(manifest_path), manifest_sha256)
    callback_source = inspect.getsourcefile(runner)
    if callback_source is None:
        raise ValueError("The runtime callback must have a hashable Python source file")
    sources = source_hashes([
        callback_source, *getattr(runner, "source_files", ()), *extra_sources,
    ])
    access = authorize_split(
        split, assets_root=assets_root, manifest_sha256=manifest_sha256, runtime=runtime,
        arm=arm, residual=residual, sources=sources,
        freeze_path=freeze_path, freeze_sha256=freeze_sha256,
    )
    scorer = OfficialVQAScorer(asset_path(assets_root, manifest["official_scorer"]))
    run_dir = Path(run_dir).resolve()
    if training_pilot and not run_dir.is_relative_to(PILOT_ROOT):
        raise ValueError(f"The authorized pilot must stay under {PILOT_ROOT}")
    if run_dir.is_relative_to(assets_root) or assets_root.is_relative_to(run_dir):
        raise ValueError("New run directories must be separate from the shared dataset assets")
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(run_dir, 0o700)
    trial_prefix = "vqav2-" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:20]
    metadata = {
        "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": BENCHMARK, "interpretation": "provisional VQAv2 selection",
        "partition": "custom 20/20/60 by image within official val2014",
        "split_seed": SPLIT_SEED, "split": split, "arm": arm,
        "optimizer_feedback_allowed": split == "train",
        "cost_scope": "task execution only; controller optimization cost is separate",
        "runtime": runtime, "runtime_sha256": canonical_sha256(runtime),
        "provider_url": provider_url, "trial_attribution": "per-trial URL",
        "trial_prefix": trial_prefix,
        "policy_sha256": access["policy_sha256"], "source_sha256": sources,
        "manifest_sha256": manifest_sha256, "access_gate": access,
        "asset_provenance": asset_provenance,
        "scoring_provenance": scorer.provenance, "hostname": platform.node(),
        "python": sys.version, "platform": platform.platform(), "argv": sys.argv,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "git": _git_state(),
        "dependencies": "stdlib adapter; pinned external Codex and Docker image in runtime",
        "training_pilot": training_pilot,
        "result_scope": ("first retained training observation; not a final benchmark score"
                         if training_pilot else "complete frozen split"),
    }
    _write_json(run_dir / "run.json", metadata)
    _write_json(run_dir / "assets-manifest.json", manifest)
    (run_dir / "selected-residual.txt").write_text(residual, encoding="utf-8")
    (run_dir / "sources").mkdir()
    predictions = []
    attempted_usage = []
    try:
        if access["test_access"]:
            gate_copy = run_dir / "frozen-policy-gate.json"
            shutil.copyfile(freeze_path, gate_copy)
            if file_sha256(gate_copy) != freeze_sha256:
                raise ValueError("Frozen policy gate changed before opening test files")
        for index, source in enumerate(sources):
            destination = run_dir / "sources" / f"{index:03d}-{Path(source).name}"
            shutil.copyfile(source, destination)
            if file_sha256(destination) != sources[source]:
                raise ValueError("Source changed after policy authorization")
        questions = load_questions(assets_root, manifest, access)
        if training_pilot:
            questions = [first_training_question(manifest, questions)]
            if questions[0].question_id != 98590000:
                raise ValueError("The authorized first training question must remain 98590000")
        annotation_descriptor = manifest["splits"][split]["annotations"]
        annotation_path = asset_path(assets_root, annotation_descriptor)
        if not annotation_path.is_file():
            raise FileNotFoundError("Evaluator-only annotations are unavailable")
        if file_sha256(annotation_path) != annotation_descriptor["sha256"]:
            raise ValueError("Evaluator-only annotation asset checksum mismatch")
        pilot_annotations = None
        if training_pilot:
            all_annotations = _read_pinned_json(annotation_path, annotation_descriptor["sha256"])
            pilot_annotations = {"annotations": [
                row for row in all_annotations["annotations"]
                if row["question_id"] == questions[0].question_id
            ]}
            validate_annotations(pilot_annotations, questions)
        for question in questions:
            image = manifest["images"][question.independent_group]
            trial_id = (
                f"vqa-train-baseline-r0-{question.question_id}" if training_pilot
                else f"{trial_prefix}-{split}-{arm}-{question.question_id}"
            )
            if training_pilot:
                ledger = Path(provider_ledger_root) / "calls" / trial_id
                if ledger.exists() and any(ledger.iterdir()):
                    raise FileExistsError("Canonical pilot already has provider calls; do not repeat it")
            request = prepare_request(
                question, asset_path(assets_root, image), image["sha256"],
                run_dir / "tasks" / str(question.question_id),
                runtime=runtime, arm=arm, residual=residual, provider_url=provider_url,
                trial_id=trial_id,
            )
            _write_json(request.workspace.parent / "request.json", {
                "question_id": request.question_id, "image_id": request.image_id,
                "independent_group": request.independent_group,
                "custom_runner_args": list(request.codex_args), "staging_data_mounts": request.data_mounts,
                "runtime_sha256": request.runtime_sha256, "policy_sha256": request.policy_sha256,
                "image_sha256": request.image_sha256,
                "trial_id": request.trial_id, "provider_base_url": request.provider_base_url,
            })
            if training_pilot:
                _write_json(PILOT_ROOT / f"{trial_id}.CLAIM.json", {
                    "trial_id": trial_id, "question_id": question.question_id,
                    "independent_group": question.independent_group,
                    "manifest_sha256": manifest_sha256, "runtime_sha256": request.runtime_sha256,
                    "policy_sha256": request.policy_sha256, "image_sha256": request.image_sha256,
                    "source_sha256": sources, "run_directory": str(run_dir),
                    "result_path": str(run_dir / "result.json"),
                    "rule": "Retain this observation; never issue this canonical pilot twice",
                })
            attempted_usage.append(provider_usage(None))
            response = runner(request)
            if not isinstance(response, Mapping):
                raise TypeError("The Codex runtime callback must return a mapping")
            attempted_usage[-1] = attributed_usage(request, response)
            _write_json(request.workspace.parent / "runtime-outcome.json", {
                "exit_code": response.get("exit_code"), "cli_version": response.get("cli_version"),
                "final_answer": response.get("final_answer"),
                "trial_id": response.get("trial_id"),
                "provider_usage": attempted_usage[-1],
                "reported_provider_usage": provider_usage(response.get("provider_usage")),
                "native_cli_usage": response.get("native_cli_usage"),
                "provider_accounting": response.get("provider_accounting"),
                "execution": response.get("execution"),
                "common_receipt_path": response.get("common_receipt_path"),
                "provider_sampling_seed_supported": response.get("provider_sampling_seed_supported"),
            })
            validate_workspace(request)
            prediction = prediction_from_response(request, response)
            _write_json(request.workspace.parent / "prediction.json", prediction)
            predictions.append(prediction)
        _write_json(run_dir / "predictions-vqa.json", [
            {"question_id": row["question_id"], "answer": row["predicted_answer"]}
            for row in predictions
        ])
        annotations = (pilot_annotations if training_pilot else
                       _read_pinned_json(annotation_path, annotation_descriptor["sha256"]))
        scores = scorer.score(questions, annotations, predictions, run_dir / "official-scorer.log")
        if source_hashes(list(sources)) != sources:
            raise ValueError("Frozen runtime/policy source changed during execution")
        for row in predictions:
            percent = scores["per_question_percent"][row["question_id"]]
            row["official_vqa_accuracy_percent"] = percent
            row["official_vqa_accuracy_01"] = percent / 100
        result = {
            **metadata, "status": "COMPLETED", "finished_at": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "independent_image_count": len({question.image_id for question in questions}),
            **scores, "predictions": predictions,
            "provider_usage": aggregate_usage(attempted_usage),
            "canonical_trial_ids": [row["trial_id"] for row in predictions],
        }
        _write_json(run_dir / "result.json", result)
        return result
    except BaseException as error:
        _write_json(run_dir / "failure.json", {
            "status": "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
            "error_type": type(error).__name__,
            "error": str(error), "completed_prediction_count": len(predictions),
            "attempted_task_count": len(attempted_usage),
            "provider_usage": aggregate_usage(attempted_usage),
            "partial_results_are_not_split_metrics": True,
        })
        raise
