"""Prospective, single-sequence inference for the pinned AIME model.

Integration (before starting a NEW service, never a frozen running service)::

    command, environment = apply_reproducible_runtime(command, environment)
    receipt = validate_reproducible_runtime(command, environment)

Freeze both returned values and the receipt with the run. Keep the caller's
physical-GPU lease for the service lifetime. This module neither starts a model
nor caches responses. The caller still verifies the model manifest and records
every request, response, token usage, and failure.

vLLM 0.25.1 rejects VLLM_BATCH_INVARIANT=1 for this model's GDN_ATTN backend.
Instead, one active sequence, no prefix reuse, fixed prefill chunks, and a
synchronous eager scheduler remove cross-request batch/cache dependence.
Synthetic repeated-output acceptance is required before claiming that a given
installation passes; these settings alone are not an acceptance result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any


MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_MANIFEST_SHA256 = "f2b4dcccf9d0204a393caa7903f768fd5043441f74497707250f7442d1fd7fbd"
SYSTEM_FINGERPRINT = "vllm-0.25.1-06bea00b"
SEED = 20260926
MODE = "single-sequence-eager-no-prefix-v1"
CONDITIONS = ("alone", "mixed_concurrent")
ENVIRONMENT = {
    "VLLM_BATCH_INVARIANT": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": str(SEED),
}


def _option(command: Sequence[str], name: str) -> str | None:
    values = []
    for index, arg in enumerate(command):
        if arg == name:
            if index + 1 == len(command) or command[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {name}")
            values.append(command[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"Duplicate option: {name}")
    return values[0] if values else None


def _set_option(command: list[str], name: str, value: str) -> None:
    previous = _option(command, name)
    if previous is None:
        command.extend((name, value))
    else:
        for index, arg in enumerate(command):
            if arg == name:
                command[index + 1] = value
                break
            if arg.startswith(name + "="):
                command[index] = name + "=" + value
                break


def _set_boolean(command: list[str], name: str, enabled: bool) -> None:
    positive, negative = "--" + name, "--no-" + name
    for index, arg in enumerate(command):
        if arg.startswith((positive + "=", negative + "=")) or (
            arg in (positive, negative)
            and index + 1 < len(command)
            and command[index + 1].lower() in ("true", "false")
        ):
            raise ValueError(f"Use an explicit boolean switch for {name}")
    command[:] = [arg for arg in command if arg not in (positive, negative)]
    command.append(positive if enabled else negative)


def apply_reproducible_runtime(
    command: Sequence[str], environment: Mapping[str, str]
) -> tuple[list[str], dict[str, str]]:
    """Return copies with explicit serial execution; reject identity changes."""
    result, env = list(command), dict(environment)
    _set_option(result, "--max-num-seqs", "1")
    _set_option(result, "--max-num-batched-tokens", "4096")
    for name, enabled in (
        ("enable-prefix-caching", False),
        ("async-scheduling", False),
        ("enable-chunked-prefill", True),
        ("enforce-eager", True),
    ):
        _set_boolean(result, name, enabled)
    env.update(ENVIRONMENT)
    validate_reproducible_runtime(result, env)
    return result, env


def validate_reproducible_runtime(
    command: Sequence[str], environment: Mapping[str, str]
) -> dict[str, Any]:
    """Fail closed on changed precision/capacity, scheduling, seed, or GPU binding."""
    expected = {
        "--dtype": "bfloat16",
        "--tensor-parallel-size": "1",
        "--max-model-len": "131072",
        "--max-num-seqs": "1",
        "--max-num-batched-tokens": "4096",
        "--seed": str(SEED),
        "--generation-config": "vllm",
        "--gdn-prefill-backend": "triton",
    }
    for name, value in expected.items():
        if _option(command, name) != value:
            raise ValueError(f"Reproducible runtime requires {name}={value}")
    for name, enabled in (
        ("enable-prefix-caching", False),
        ("async-scheduling", False),
        ("enable-chunked-prefill", True),
        ("enforce-eager", True),
    ):
        wanted = "--" + ("" if enabled else "no-") + name
        opposite = "--" + ("no-" if enabled else "") + name
        if command.count(wanted) != 1 or any(
            arg == opposite or arg.startswith((wanted + "=", opposite + "="))
            for arg in command
        ):
            raise ValueError(f"Reproducible runtime requires {wanted}")
    for name, value in ENVIRONMENT.items():
        if environment.get(name) != value:
            raise ValueError(f"Reproducible runtime requires {name}={value}")
    gpu_uuid = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid):
        raise ValueError("Bind exactly one physical GPU UUID after acquiring its lease")
    rpc_path = environment.get("VLLM_RPC_BASE_PATH")
    if rpc_path and len((rpc_path + "/" + "0" * 36).encode()) >= 108:
        raise ValueError("VLLM_RPC_BASE_PATH is too long for a Linux Unix-domain socket")
    attention = json.loads(_option(command, "--attention-config") or "{}")
    template = json.loads(_option(command, "--default-chat-template-kwargs") or "{}")
    if attention != {"backend": "FLASH_ATTN", "flash_attn_version": 3}:
        raise ValueError("Preserve the pinned FLASH_ATTN v3 attention backend")
    if template != {"enable_thinking": False}:
        raise ValueError("Preserve the pinned non-thinking chat template")
    if _option(command, "--quantization") not in (None, "None"):
        raise ValueError("Do not quantize the pinned bf16 model")
    for name in ("--speculative-config", "--kv-sharing-fast-prefill"):
        if any(arg == name or arg.startswith(name + "=") for arg in command):
            raise ValueError(f"Unvalidated execution override: {name}")
    if _option(command, "--fingerprint-mode") not in (None, "full") or _option(
        command, "--fingerprint-value"
    ) is not None:
        raise ValueError("Preserve vLLM's genuine configuration-derived fingerprint")
    return {
        "mode": MODE,
        "environment": {key: environment[key] for key in (*ENVIRONMENT, "CUDA_VISIBLE_DEVICES")},
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "predecessor_system_fingerprint": SYSTEM_FINGERPRINT,
        "system_fingerprint_policy": "Native vLLM 0.25.1 configuration hash; equal across all calls",
        "maximum_cumulative_tokens": None,
        "token_usage_mode": "unlimited",
        "response_cache": False,
        "batch_invariant_backend_supported": False,
        "acceptance": "REQUIRES_REMOTE_SYNTHETIC_REPEATS",
    }


def validate_server_info(info: Mapping[str, Any], *, model_path: str) -> dict[str, Any]:
    """Check the live engine configuration, not just the submitted CLI flags."""
    config = info["vllm_config"]
    required = {
        "model_config": {
            "model": model_path, "max_model_len": 131072, "seed": SEED, "enforce_eager": True,
            "quantization": None,
        },
        "scheduler_config": {
            "max_num_seqs": 1, "max_num_batched_tokens": 4096,
            "enable_chunked_prefill": True, "async_scheduling": False,
        },
        "cache_config": {"enable_prefix_caching": False, "mamba_cache_mode": "none"},
        "parallel_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
    }
    for section, fields in required.items():
        for key, value in fields.items():
            if key not in config[section] or config[section][key] != value:
                raise ValueError(f"Live engine differs: {section}.{key}")
    if config["model_config"]["dtype"] not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("Live engine precision differs")
    attention = config["attention_config"]
    if attention["backend"] not in (
        "FLASH_ATTN", "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ) or attention["flash_attn_version"] != 3:
        raise ValueError("Live attention backend differs")
    if info["vllm_env"]["VLLM_BATCH_INVARIANT"] is not False:
        raise ValueError("Unsupported batch-invariant backend mode is enabled")
    return {"status": "PASS", "mode": MODE, "checked_fields": required}


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def response_signature(response: Mapping[str, Any]) -> dict[str, Any]:
    """Compare real output tokens, full message, usage, and runtime identity."""
    if not re.fullmatch(r"vllm-0\.25\.1-[0-9a-f]{8}", response.get("system_fingerprint") or ""):
        raise ValueError("Response fingerprint differs from the pinned runtime")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one generated sequence")
    choice = choices[0]
    message, tokens = choice.get("message"), choice.get("token_ids")
    prompt_tokens, usage = response.get("prompt_token_ids"), response.get("usage")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Missing complete generated message")
    for name, value in (("token_ids", tokens), ("prompt_token_ids", prompt_tokens)):
        if not isinstance(value, list) or not value or any(type(x) is not int for x in value):
            raise ValueError(f"Missing native {name}; retokenizing text is not sufficient")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        raise ValueError("Missing measured token usage")
    if (
        usage["prompt_tokens"] != len(prompt_tokens)
        or usage["completion_tokens"] != len(tokens)
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise ValueError("Native token IDs and usage disagree")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) != 0:
        raise ValueError("Prefix-cache reuse is forbidden in the diagnostic")
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("Incomplete or unexpected synthetic response")
    return {
        "model": response.get("model"),
        "system_fingerprint": response["system_fingerprint"],
        "message": message,
        "token_ids": tokens,
        "prompt_token_ids": prompt_tokens,
        "usage": usage,
        "finish_reason": choice["finish_reason"],
        "stop_reason": choice.get("stop_reason"),
    }


def verify_repeated_outputs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require >=12 target calls and genuine concurrent distractor submissions."""
    if len({row["sequence"] for row in records}) != len(records):
        raise ValueError("Duplicate call accounting")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    total_usage = Counter()
    fingerprints = set()
    for row in records:
        signature = response_signature(row["response"])
        fingerprints.add(signature["system_fingerprint"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[key] += signature["usage"][key]
        request = row["request"]
        if (
            request.get("seed") != SEED
            or request.get("temperature") != 0
            or request.get("n", 1) != 1
            or request.get("stream", False)
            or request.get("return_token_ids") is not True
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
        ):
            raise ValueError("Synthetic request differs from the fixed inference policy")
        if row["target"]:
            groups[row["prompt_name"]].append(row)
    if sum(map(len, groups.values())) < 12:
        raise ValueError("At least twelve target repeats are required")
    if len(fingerprints) != 1:
        raise ValueError("Runtime fingerprint changed during the diagnostic")
    results = {}
    for name, rows in groups.items():
        counts = Counter(row["condition"] for row in rows)
        if set(counts) != set(CONDITIONS) or min(counts.values()) < 3:
            raise ValueError("Each target needs at least three repeats in both conditions")
        if len({canonical_digest(row["request"]) for row in rows}) != 1:
            raise ValueError("Repeated request bodies differ")
        for row in rows:
            if row["condition"] == "mixed_concurrent" and not any(
                not other["target"]
                and other["group"] == row["group"]
                and max(other["started_ns"], row["started_ns"])
                < min(other["finished_ns"], row["finished_ns"])
                for other in records
            ):
                raise ValueError("Mixed condition lacks overlapping distractor requests")
        signatures = [response_signature(row["response"]) for row in rows]
        variants = len({canonical_digest(value) for value in signatures})
        results[name] = {
            "repeats": dict(counts),
            "request_sha256": canonical_digest(rows[0]["request"]),
            "output_signature_sha256": canonical_digest(signatures[0]),
            "distinct_output_signatures": variants,
        }
        if variants != 1:
            raise ValueError(f"Generated text/token IDs/usage differ for {name}: {variants} variants")
    return {
        "status": "PASS",
        "scope": "Synthetic repeated-output equality within this frozen installation",
        "mode": MODE,
        "system_fingerprint": next(iter(fingerprints)),
        "predecessor_system_fingerprint": SYSTEM_FINGERPRINT,
        "targets": results,
        "calls": len(records),
        "usage": dict(total_usage),
        "unknown_usage_calls": 0,
        "maximum_cumulative_tokens": None,
        "token_usage_mode": "unlimited",
        "response_cache": False,
    }
