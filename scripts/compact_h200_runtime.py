"""Pinned native H200 environment for the supervised compact-contract study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys


def digest(filename: Path) -> str:
    hasher = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def identity(process_id: int) -> dict | None:
    try:
        root = Path(f"/proc/{process_id}")
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": process_id, "uid": root.stat().st_uid, "start_time_ticks": int(fields[19]),
                "start_ticks": int(fields[19]), "cmdline_sha256": digest(root / "cmdline"), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def send(saved: dict, number: int) -> None:
    try:
        descriptor = os.pidfd_open(saved["pid"])
        try:
            current = identity(saved["pid"])
            if current and all(saved[key] == current[key] for key in ("pid", "uid", "start_ticks", "cmdline_sha256")):
                signal.pidfd_send_signal(descriptor, number)
        finally:
            os.close(descriptor)
    except ProcessLookupError:
        return


class H200Runtime:
    ENVIRONMENT_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_RPC_BASE_PATH", "HF_HUB_OFFLINE")

    def __init__(self, config: dict):
        self.config = config
        self.resource = config["resource"]
        self.native = self.resource["native_runtime"]
        self.model = Path(self.native["model_path"])
        self.python = self.native["python_executable"]

    def environment(self, run: Path, cuda: str = "") -> dict[str, str]:
        environment = {key: value for key, value in os.environ.items() if key in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL")}
        runtime = Path(self.python).parent
        environment.update({
            "PATH": f"{runtime}:/usr/local/cuda/bin:/usr/bin:/bin", "CUDA_HOME": "/usr/local/cuda",
            "CUDA_VISIBLE_DEVICES": cuda, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "SPT_NOENV": "1",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(run / "caches/huggingface"), "TOKENIZERS_PARALLELISM": "false",
            "VLLM_RPC_BASE_PATH": str(run / "scratch"), "VLLM_CACHE_ROOT": str(run / "caches/vllm"),
            "VLLM_CONFIG_ROOT": str(run / "caches/vllm-config"), "VLLM_BATCH_INVARIANT": "0",
            "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "PYTHONNOUSERSITE": "1",
            "TRITON_CACHE_DIR": str(run / "caches/triton"), "TORCH_EXTENSIONS_DIR": str(run / "caches/torch"),
            "XDG_CACHE_HOME": str(run / "caches/xdg"), "CUDA_CACHE_PATH": str(run / "caches/cuda"),
            "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(run / "scratch"),
        })
        return environment

    def rpc_check(self, run: Path) -> None:
        if len(os.fsencode(run / "scratch/00000000-0000-0000-0000-000000000000")) > 107:
            raise ValueError("The native service path exceeds the Unix socket length limit")

    def command(self, _run: Path) -> list[str]:
        model = self.config["model"]
        attention = self.native.get("attention_config", {"backend": "FLASH_ATTN", "flash_attn_version": 2})
        if (not isinstance(attention, dict) or set(attention) != {"backend", "flash_attn_version"}
                or attention["backend"] != "FLASH_ATTN"
                or type(attention["flash_attn_version"]) is not int
                or attention["flash_attn_version"] not in (2, 3)):
            raise ValueError("Native H200 attention must prospectively pin FlashAttention version 2 or 3")
        return [
            self.python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(self.model),
            "--served-model-name", model["repository"], "--host", "127.0.0.1", "--port", str(self.resource["port"]),
            "--dtype", model["dtype"], "--tensor-parallel-size", "1", "--max-model-len", str(model["context_length"]),
            "--max-num-seqs", "32", "--gpu-memory-utilization", "0.8", "--max-num-batched-tokens", "4096",
            "--seed", "0", "--enforce-eager", "--language-model-only", "--generation-config", "vllm",
            "--attention-config", json.dumps(attention, separators=(",", ":")),
            "--gdn-prefill-backend", "triton", "--default-chat-template-kwargs", '{"enable_thinking":false}',
        ]

    def prepare(self, campaign: Path, write_json) -> Path:
        manifest_path = Path(self.native["manifest_path"])
        if digest(manifest_path) != self.native["manifest_sha256"]:
            raise RuntimeError("The pinned H200 model manifest changed")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["canonical_repository"] != self.config["model"]["repository"]
                or manifest["canonical_huggingface_revision"] != self.config["model"]["revision"]):
            raise RuntimeError("The H200 checkpoint repository/revision differs")
        verified = {}
        for entry in manifest["files"]:
            filename = self.model / entry["path"]
            if (not filename.resolve().is_relative_to(self.model.resolve()) or filename.stat().st_size != entry["size"]
                    or digest(filename) != entry["sha256"]):
                raise RuntimeError(f"H200 model file binding changed: {entry['path']}")
            verified[entry["path"]] = entry["sha256"]
        tokenizer_files = ("chat_template.jinja", "config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
        spec = {
            "model_repository": self.config["model"]["repository"], "revision": self.config["model"]["revision"],
            "checkpoint_path": str(self.model), "tokenizer_path": str(self.model), "runtime_python": self.python,
            "context_length": self.config["model"]["context_length"], "dtype": self.config["model"]["dtype"],
            "tokenizer_files_sha256": {name: verified[name] for name in tokenizer_files},
        }
        write_json(campaign / "H200_CHECKPOINT_VERIFIED.json", {
            "manifest_sha256": digest(manifest_path), "files_sha256": verified, "model_spec": spec,
            "source_sha256": digest(Path(__file__)), "python": sys.version, "model_calls": 0,
        })
        destination = campaign / "h200-model-spec.json"
        write_json(destination, spec)
        return destination

    def validate_environment(self, expected: dict, observed: dict, *, model_worker: bool) -> None:
        if (observed["CUDA_DEVICE_ORDER"] != expected["CUDA_DEVICE_ORDER"]
                or observed["VLLM_RPC_BASE_PATH"] != expected["VLLM_RPC_BASE_PATH"]
                or observed["HF_HUB_OFFLINE"] != "1"):
            raise RuntimeError("Native H200 process environment differs from the frozen launch")
        visible = observed["CUDA_VISIBLE_DEVICES"]
        allowed = {expected["CUDA_VISIBLE_DEVICES"], "0"} if model_worker else {expected["CUDA_VISIBLE_DEVICES"]}
        if visible not in allowed:
            raise RuntimeError("Native H200 CUDA visibility differs from the selected device")

    def validate_library_maps(self, libraries: list[str], *, model_worker: bool) -> None:
        if any("cuda-compat" in library or "/path/to/data/" in library for library in libraries):
            raise RuntimeError("A800 compatibility paths must not enter the native H200 runtime")
        if model_worker and not any("libcuda.so" in library for library in libraries):
            raise RuntimeError("The H200 GPU worker lacks a recorded native CUDA driver mapping")
