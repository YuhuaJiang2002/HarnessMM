#!/usr/bin/env python3
"""Verify official installer bytes and their transport in a native TRAIN image.

This A800-only dependency check runs no actor, benchmark test, or scoring logic.
Its fresh directory freezes the configuration, commands, sources and receipts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deepseek_terminal_campaign import (  # noqa: E402
    AUTHORIZED_HOST, CAMPAIGN, CampaignBlocked, file_hash, load, persist, utc_now,
    validate_config,
)
from scripts.run_deepseek_terminal import resource_inventory  # noqa: E402

PUBLIC_CACHE = Path("/path/to/data/phiagent/work/terminal-dependency-cache/20260922-v1")
ASSETS = (
    {
        "filename": "uv-x86_64-unknown-linux-gnu.tar.gz",
        "route": "/github/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz",
        "size": 21370871,
        "sha256": "2cf10babba653310606f8b49876cfb679928669e7ddaa1fb41fb00ce73e64f66",
    },
    {
        "filename": (
            "cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        ),
        "route": (
            "/github/astral-sh/python-build-standalone/releases/download/20251014/"
            "cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        ),
        "size": 33517986,
        "sha256": "c0ccda275948c79996e46993c2c5476ff5cca606dee530f1dea712179131b348",
    },
)

CHECK = """set -eu
mkdir -p /logs/verifier
if ! command -v curl >/dev/null; then
  apt-get update
  apt-get install -y curl
fi
curl -LsSf --max-time 90 https://astral.sh/uv/0.9.5/install.sh -o /logs/verifier/uv-installer.sh
sh /logs/verifier/uv-installer.sh
export PATH=/root/.local/bin:$PATH
uv --version
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version
"""


def prepare(config_path: Path) -> dict:
    config = load(config_path)
    validate_config(config)
    if socket.gethostname() != AUTHORIZED_HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CampaignBlocked("Dependency acceptance requires a800-4 and CUDA_VISIBLE_DEVICES=''")
    manifest_path = Path(config["dependency_manifest"])
    root = manifest_path.parent
    if not root.is_relative_to(CAMPAIGN / "setup") or root == CAMPAIGN / "setup":
        raise CampaignBlocked("Use a fresh setup directory in the authorized campaign")
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    persist(root / "CONFIG.json", config, immutable=True)
    persist(root / "RESOURCE_INVENTORY.json", resource_inventory(), immutable=True)
    persist(root / "OWNER.json", {
        "pid": os.getpid(), "started_at": utc_now(), "hostname": socket.gethostname(),
        "purpose": "dependency_transport_only", "benchmark_or_provider_calls": 0,
    }, immutable=True)
    persist(root / "PACKAGES.json", {
        "python": sys.version, "executable": sys.executable,
        "packages": sorted((item.metadata["Name"], item.version)
                           for item in importlib.metadata.distributions()),
    }, immutable=True)
    persist(root / "SOURCE.json", {
        "preparer": str(Path(__file__).resolve()), "sha256": file_hash(Path(__file__)),
        "controller": str(ROOT / "scripts/run_deepseek_terminal_campaign.py"),
        "controller_sha256": file_hash(ROOT / "scripts/run_deepseek_terminal_campaign.py"),
        "upload_git_state": load(ROOT / "CAMPAIGN_UPLOAD_GIT_STATE.json"),
        "source_config_sha256": file_hash(config_path), "seed": config["split_seed"],
    }, immutable=True)
    ready_path = PUBLIC_CACHE / "READY.json"
    ready = load(ready_path)
    assets = []
    (root / "assets").mkdir()
    for asset in ASSETS:
        source = PUBLIC_CACHE / "artifacts" / asset["filename"]
        historical = ready["artifacts"].get(asset["route"], {})
        if (
            source.is_symlink() or source.stat().st_size != asset["size"]
            or file_hash(source) != asset["sha256"]
            or any(historical.get(key) != asset[key] for key in ("size", "sha256"))
        ):
            raise CampaignBlocked("Pinned official dependency cache differs from its receipt")
        target = root / "assets" / asset["filename"]
        shutil.copyfile(source, target)
        assets.append({
            **asset, "path": str(target), "source_path": str(source),
            "official_url": "https://github.com" + asset["route"].removeprefix("/github"),
        })
    persist(manifest_path, {
        "assets": assets, "cache_receipt": str(ready_path),
        "cache_receipt_sha256": file_hash(ready_path), "created_at": utc_now(),
        "model_or_benchmark_contents": False,
    }, immutable=True)
    task = Path(config["task_root"]) / "count-dataset-tokens"
    task_config = tomllib.loads((task / "task.toml").read_text())
    image = task_config["environment"]["docker_image"]
    image_receipt = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .}}", image],
        capture_output=True, text=True, check=True, timeout=30,
    )
    image_info = json.loads(image_receipt.stdout)
    persist(root / "NATIVE_IMAGE.json", {
        "image": image, "image_id": image_info["Id"], "repo_digests": image_info["RepoDigests"],
        "task_toml_sha256": file_hash(task / "task.toml"),
        "verifier_script_sha256": file_hash(task / "tests/test.sh"),
        "task_or_verifier_modified": False,
    }, immutable=True)
    port = int(config["dependency_base_url"].rsplit(":", 1)[1])
    with socket.socket() as probe:
        if probe.connect_ex(("172.17.0.1", port)) == 0:
            raise CampaignBlocked("Dependency port is already owned; do not replace its server")
    server_command = [
        sys.executable, str(ROOT / "scripts/run_deepseek_terminal_campaign.py"),
        "--serve-dependencies", str(manifest_path), "--dependency-port", str(port),
    ]
    with (root / "dependency-server.log").open("xb") as stream:
        server = subprocess.Popen(
            server_command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    persist(root / "DEPENDENCY_LAUNCH.json", {
        "argv": server_command, "pid": server.pid, "started_at": utc_now(),
        "start_new_session": True, "stdin": "DEVNULL",
    }, immutable=True)
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 15
    while not (root / "DEPENDENCY_SERVER.json").is_file():
        if server.poll() is not None or time.monotonic() > deadline:
            raise CampaignBlocked("Dependency server failed startup; preserve its log")
        time.sleep(0.2)
    for asset in assets:
        with opener.open(Request(
            config["dependency_base_url"] + asset["route"], method="HEAD",
        ), timeout=15) as response:
            if response.status != 200 or int(response.headers["Content-Length"]) != asset["size"]:
                raise CampaignBlocked("Pinned dependency archive is not served correctly")
    logs = root / "native-image-logs"
    logs.mkdir()
    name = "dsh-dependency-acceptance-" + root.name.removeprefix("terminal-recovery-").lower()
    command = [
        "docker", "run", "--name", name, "--cpus", "1", "--memory", "2g",
        "--pids-limit", "256", "--label", "phiagent.purpose=terminal-dependency-acceptance",
        "--mount", f"type=bind,src={logs},dst=/logs/verifier",
        "--env", "CUDA_VISIBLE_DEVICES=",
    ]
    for key, value in config["verifier_environment"].items():
        command.extend(["--env", f"{key}={value}"])
    command.extend(["--entrypoint", "/bin/bash", image, "-lc", CHECK])
    persist(root / "COMMANDS.json", {
        "preparer_argv": sys.argv, "cwd": str(ROOT), "server": server_command,
        "native_image_check": command, "CUDA_VISIBLE_DEVICES": "",
        "original_agent_timeout_seconds": task_config["agent"]["timeout_sec"],
        "original_verifier_timeout_seconds": task_config["verifier"]["timeout_sec"],
        "benchmark_test_executed": False, "provider_calls": 0, "cumulative_token_limit": None,
    }, immutable=True)
    with (root / "native-image-check.log").open("xb") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1000)
    log = (root / "native-image-check.log").read_text(errors="replace")
    success = completed.returncode == 0 and "uv 0.9.5" in log and "pytest 8.4.1" in log
    receipt = {
        "status": "passed" if success else "failed", "checked_at": utc_now(),
        "manifest_sha256": file_hash(manifest_path),
        "verifier_environment": config["verifier_environment"],
        "native_image": image, "native_image_id": image_info["Id"],
        "returncode": completed.returncode, "container_name": name,
        "log": str(root / "native-image-check.log"),
        "log_sha256": file_hash(root / "native-image-check.log"),
        "official_installer_sha256": (
            file_hash(logs / "uv-installer.sh") if (logs / "uv-installer.sh").is_file() else None
        ),
        "native_task_or_scorer_modified": False, "benchmark_test_executed": False,
        "provider_calls": 0, "selected_gpus": [], "cumulative_token_limit": None,
    }
    persist(Path(config["dependency_acceptance"]), receipt, immutable=True)
    if not success:
        raise CampaignBlocked("Native-image dependency transport failed; container/logs preserved")
    cleanup = subprocess.run(
        ["docker", "rm", name], capture_output=True, text=True, timeout=30,
    )
    persist(root / "CLEANUP.json", {
        "owned_acceptance_container": name, "returncode": cleanup.returncode,
        "dependency_server_retained_for_campaign": True,
    }, immutable=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.config.resolve())
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
