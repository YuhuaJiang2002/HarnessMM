#!/usr/bin/env python3
"""Prepare or reuse native dependencies and score a retained Terminal workspace."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import traceback
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic  # noqa: E402
from scripts.terminal_network_lifecycle import network_owner, retire_networks  # noqa: E402

HOST = "harnessmm-api-worker"
DATA = Path("/path/to/data/phiagent/work/harness-terminal-recovery")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_executed_native_tests(script: str, receipt: Path, reward: float, *, required: bool = False):
    if required or re.search(r"\bpytest\s+--ctrf\b", script):
        from scripts.audit_industrial_capability_recovery import executed_tests

        return executed_tests(load(receipt), reward)
    return None


def container_name(prefix: str, output: Path) -> str:
    return prefix + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:20]


def dependency_command(script: str) -> tuple[str, list[str]]:
    matches = list(re.finditer(r"(?m)^uvx\s", script))
    if len(matches) != 1:
        raise ValueError("Expected one explicit native uvx verifier command")
    start = matches[0].start()
    prelude = script[:start]
    installer_source = list(re.finditer(
        r"(?m)^source\s+(?:\"\$HOME\"|\$HOME)/\.local/bin/env\s*$", prelude,
    ))
    if installer_source:
        prelude = prelude[:installer_source[-1].end()] + "\n"
    command = script[start:].replace("\\\n", " ")
    match = re.search(r"\bpytest\s+--ctrf\b", command)
    if match is None:
        raise ValueError("The native pytest command cannot be isolated safely")
    arguments = shlex.split(command[:match.start()])
    if arguments[0] != "uvx" or len(arguments) % 2 != 1:
        raise ValueError("Unexpected native dependency argument structure")
    for index in range(1, len(arguments), 2):
        flag, value = arguments[index:index + 2]
        if flag in ("-p", "-w") and re.fullmatch(r"[\w.+=,<>\[\]-]+", value):
            continue
        if flag == "--index" and value == "https://download.pytorch.org/whl/cpu":
            continue
        if flag == "--index-strategy" and value == "unsafe-best-match":
            continue
        if flag == "-w" and value == (
            "git+https://github.com/ChaoningZhang/MobileSAM.git@${COMMIT_HASH}"
        ):
            commits = re.findall(r"(?m)^COMMIT_HASH=['\"]([a-f0-9]{40})['\"]\s*$", script)
            if len(commits) != 1:
                raise ValueError("The native MobileSAM dependency requires one literal commit")
            arguments[index + 1] = value.replace("${COMMIT_HASH}", commits[0])
            continue
        raise ValueError("Only literal native Python, pinned packages and declared native indexes are accepted")
    return 'mkdir -p "$TMPDIR"\n' + prelude, [*arguments, "python", "-c", "pass"]


def plain_pip_setup(script: str) -> str:
    installs = list(re.finditer(r"(?m)^pip install ([^\n]+)$", script))
    tests = list(re.finditer(r"(?m)^pytest\s+--ctrf\b", script))
    if len(installs) != 1 or len(tests) != 1 or installs[0].end() >= tests[0].start():
        raise ValueError("Expected one pinned pip install before the unchanged native pytest command")
    requirements = shlex.split(installs[0].group(1))
    if not requirements or any(
        re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", value) is None
        for value in requirements
    ):
        raise ValueError("Only literal version-pinned native pip dependencies may be prepared")
    setup = script[:installs[0].end()] + "\n"
    if "/tests/" in setup or "/app/" in setup or re.search(r"(?m)^\s*(cp|mv|rm|pytest|python)\s", setup):
        raise ValueError("Dependency-only setup must not access task-workspace or verifier code")
    return 'mkdir -p "${TMPDIR:-/tmp}"\n' + setup


def native_system_package_setup(script: str) -> list[str]:
    """Extract literal public APT setup, including lines after the uv installer.

    The working-directory guard and test commands stay in the native verifier.
    Dependencies are installed before its original deadline starts.
    """
    prefix = re.split(r"(?m)^uvx\s", script, maxsplit=1)[0]
    commands = []
    for line in prefix.splitlines():
        if not re.match(r"^apt-get\s", line):
            continue
        for part in line.split("&&"):
            words = shlex.split(part, comments=True)
            if words and words[0] == 'DEBIAN_FRONTEND=noninteractive':
                words = words[1:]
            if words[:2] == ["apt-get", "update"] and all(x in ("-q", "-qq") for x in words[2:]):
                continue
            if (words[:3] == ["apt-get", "install", "-y"] and len(words) > 3
                    and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+:~=\-]*", x) for x in words[3:])):
                continue
            raise ValueError("Only literal native public APT dependency commands may be prepared")
        commands.append(line)
    return commands


def uncached_native_setup(script: str) -> tuple[str, str]:
    if re.search(r"(?m)^uvx\s", script):
        setup, install = dependency_command(script)
        return "native_uv_setup", setup + "\n" + shlex.join(install) + "\n"
    return "native_pinned_pip_setup", plain_pip_setup(script)


def dependency_environment(environment: dict[str, str]) -> dict[str, str]:
    value = {"TMPDIR": "/tmp", **environment}
    if not isinstance(value["TMPDIR"], str) or not value["TMPDIR"].startswith("/"):
        raise ValueError("Native dependency setup needs a nonempty absolute temporary directory")
    return value


def native_cpu_thread_environment(environment: dict[str, str], cpus: float) -> dict[str, str]:
    """Match numerical-library threads to the unchanged container CPU quota."""
    if type(cpus) not in (int, float) or not math.isfinite(cpus) or not 1 <= cpus <= 256:
        raise ValueError("Native CPU thread matching requires a declared positive CPU quota")
    threads = str(int(cpus))
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    if any(key in environment and environment[key] != threads for key in keys):
        raise ValueError("A previously declared numerical thread policy cannot be silently replaced")
    return {**environment, **dict.fromkeys(keys, threads)}


def dependency_transport_environment(
    original: dict[str, str], overrides: dict[str, str] | None,
) -> dict[str, str]:
    if overrides is None:
        return dict(original)
    proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"}
    if set(overrides) - proxy_keys:
        raise ValueError("Only public dependency proxy transport may be overridden")
    for value in overrides.values():
        if not isinstance(value, str):
            raise ValueError("Dependency proxy must be an explicit private Docker-bridge URL")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http" or parsed.hostname != "172.17.0.1"
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.port is None
        ):
            raise ValueError("Dependency proxy must be an explicit private Docker-bridge URL")
    return {**original, **overrides}


def validate_retained_container(info: dict, expected_id: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_id) or info.get("Id") != expected_id:
        raise ValueError("Retained container identity differs")
    if info["State"]["Status"] != "exited" or info["State"].get("Running"):
        raise ValueError("Only a stopped owned workspace may be snapshotted")
    if info["Config"]["Cmd"] != ["sh", "-c", "sleep infinity"]:
        raise ValueError("Unexpected container startup command; never restart an actor")
    if any(
        mount["Destination"] not in ("/logs/agent", "/logs/verifier", "/logs/artifacts")
        for mount in info.get("Mounts", [])
    ):
        raise ValueError("Additional mounted task state requires explicit recovery")
    if info["HostConfig"].get("DeviceRequests"):
        raise ValueError("This recovery does not allocate or remap GPUs")


def verified_cache(manifest: Path, task: Path) -> dict:
    value = load(manifest)
    if (
        value.get("status") != "OFFLINE_DEPENDENCIES_VERIFIED"
        or value.get("task_id") != task.name
        or value.get("test_script_sha256") != digest(task / "tests/test.sh")
        or value.get("task_config_sha256") != digest(task / "task.toml")
        or value.get("additional_model_calls") != 0
        or value.get("native_verifier_calls") != 0
    ):
        raise ValueError("Verifier cache does not match the unchanged native task")
    archive = Path(value["archive"])
    if archive.is_symlink() or digest(archive) != value["archive_sha256"]:
        raise ValueError("Verified dependency archive changed")
    return value


def require_sam_cache_mode(
    *, capability: bool, task_id: str, phase: str, cache: Path | None,
    figure3_codex_train: bool = False,
) -> None:
    if not (capability or figure3_codex_train) or task_id != "sam-cell-seg" or phase != "train" or cache is not None:
        raise ValueError("Official SAM cache recovery is restricted to the same unscored TRAIN actor")


def command(argv: list[str], output: Path, name: str, *, timeout: int = 1800) -> str:
    with (output / (name + ".log")).open("x") as log:
        result = subprocess.run(
            argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {output / (name + '.log')}")
    return (output / (name + ".log")).read_text()


def reserve(output: Path) -> None:
    if socket.gethostname() != HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use authorized A800-4 CPU execution with CUDA_VISIBLE_DEVICES=''")
    if not output.resolve().is_relative_to(DATA) or output.resolve() == DATA:
        raise ValueError("Use a new campaign-owned data-volume recovery directory")
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "provenance.json", capture_provenance(ROOT, sys.argv, 20260928))
    write_json_atomic(output / "RESOURCES.json", {
        "hostname": socket.gethostname(), "pid": os.getpid(), "at": now(),
        "CUDA_VISIBLE_DEVICES": "", "selected_gpu": None,
        "gpu_inventory": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ], text=True, timeout=30),
        "available_memory": Path("/proc/meminfo").read_text(),
        "data_free_bytes": os.statvfs(output).f_bavail * os.statvfs(output).f_frsize,
        "cumulative_token_limit": None,
    })
    write_json_atomic(output / "SOURCE.json", {
        "script": str(Path(__file__).resolve()), "sha256": digest(Path(__file__)),
        "python": sys.version, "hostname": socket.gethostname(), "seed": 20260928,
        "packages": sorted(
            (item.metadata["Name"], item.version) for item in importlib.metadata.distributions()
        ),
        "uploaded_git_state": (
            load(ROOT / "UPLOAD_GIT_STATE.json") if (ROOT / "UPLOAD_GIT_STATE.json").is_file()
            else None
        ),
    })


def seed_dependency_metadata(source: Path, home: Path) -> dict:
    records = {}
    for pattern in ("simple-v*/pypi/*.rkyv", "wheels-v*/pypi/*/*.msgpack",
                    "simple-v*/index/*/*.rkyv", "wheels-v*/index/*/*/*.msgpack"):
        for path in sorted(source.glob(pattern)):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only closed regular public dependency metadata may be seeded")
            data = path.read_bytes()
            relative = path.relative_to(source)
            destination = home / ".cache/uv" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
            records[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    if not records:
        raise ValueError("The requested public metadata seed is empty")
    return {"source": str(source), "files_sha256": records, "partial_downloads_copied": False}


def seed_verified_public_wheels(source: Path, home: Path) -> dict:
    """Copy closed, official-hash public packages without changing native requirements."""
    proof = load(source / "COMPLETE.json")
    if proof.get("status") != "OFFICIAL_PYPI_HASH_VERIFIED_WHEELS" or not proof.get("packages"):
        raise ValueError("Public wheels require a complete official-hash verification receipt")
    records = []
    seen = set()
    for record in proof["packages"]:
        filename = record["filename"]
        path = source / "wheels" / filename
        if (
            record.get("name") not in {"numpy", "opencv-python", "opencv-contrib-python", "pillow"}
            or Path(filename).name != filename or not filename.endswith(".whl")
            or filename in seen or path.is_symlink() or not path.is_file()
            or urlsplit(record["official_url"]).hostname != "files.pythonhosted.org"
            or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]
        ):
            raise ValueError("Reject incomplete, altered or non-native public dependency wheels")
        seen.add(filename)
        records.append(record)
    destination = home / ".cache/uv/phiagent-wheels"
    destination.mkdir(parents=True, exist_ok=False)
    for record in records:
        shutil.copyfile(source / "wheels" / record["filename"], destination / record["filename"])
    return {"source": str(source), "manifest_sha256": digest(source / "COMPLETE.json"),
            "packages": records, "partial_downloads_copied": False,
            "native_dependency_command_changed": False, "additional_model_calls": 0}


def prepare(
    task: Path, output: Path, environment: dict[str, str], seed_metadata: Path | None = None,
    *, install_public_ca_certificates: bool = False, public_wheel_cache: Path | None = None,
) -> None:
    import tomllib

    reserve(output)
    environment = dependency_environment(environment)
    task_config = tomllib.loads((task / "task.toml").read_text())
    image = task_config["environment"]["docker_image"]
    prelude, install = dependency_command((task / "tests/test.sh").read_text())
    name = container_name("phiagent-verifier-deps-", output)
    home = output / "home"
    home.mkdir()
    if seed_metadata is not None:
        write_json_atomic(
            output / "PUBLIC_METADATA_SEED.json", seed_dependency_metadata(seed_metadata, home),
        )
    if public_wheel_cache is not None:
        if "UV_FIND_LINKS" in environment:
            raise ValueError("A public wheel seed cannot replace an existing dependency source")
        write_json_atomic(output / "PUBLIC_WHEEL_SEED.json", seed_verified_public_wheels(public_wheel_cache, home))
        environment = {**environment, "UV_FIND_LINKS": "/root/.cache/uv/phiagent-wheels"}
    config = {
        "task_id": task.name, "task": str(task), "image": image,
        "test_script_sha256": digest(task / "tests/test.sh"),
        "task_config_sha256": digest(task / "task.toml"),
        "dependency_command": install, "verifier_environment": environment,
        "additional_model_calls": 0, "native_verifier_calls": 0,
        "setup_outside_native_verifier_deadline": True,
        "install_public_ca_certificates": install_public_ca_certificates,
    }
    write_json_atomic(output / "CONFIG.json", config)
    create = [
        "docker", "run", "-d", "--name", name, "--cpus", "1", "--memory", "8g",
        "--label", "phiagent.purpose=native-verifier-public-dependencies",
        "--mount", f"type=bind,src={home},dst=/root",
        "--mount", f"type=bind,src={output},dst=/dependency-output",
        "--env", "CUDA_VISIBLE_DEVICES=",
    ]
    for key, value in environment.items():
        create.extend(("--env", f"{key}={value}"))
    create.extend(("--entrypoint", "sh", image, "-c", "sleep infinity"))
    command(create, output, "create", timeout=120)
    write_json_atomic(output / "STATE.json", {
        "status": "PREPARING", "pid": os.getpid(), "container": name, "updated_at": now(),
    })
    try:
        if install_public_ca_certificates:
            command(["docker", "exec", name, "bash", "-lc",
                     'set -e\nmkdir -p "${TMPDIR:-/tmp}"\napt-get update\napt-get install -y ca-certificates'],
                    output, "public-ca-certificates", timeout=1800)
        command(
            ["docker", "exec", name, "bash", "-lc", "set -e\n" + prelude + "\n" + shlex.join(install)],
            output, "online-dependencies", timeout=7200,
        )
        command(
            ["docker", "exec", "--env", "UV_OFFLINE=1", name, "bash", "-lc",
             "export PATH=/root/.local/bin:$PATH; " + shlex.join(install)],
            output, "offline-dependencies", timeout=300,
        )
        command(
            ["docker", "exec", name, "tar", "-C", "/root", "-czf",
             "/dependency-output/dependencies.tar.gz", ".cache/uv", ".local"],
            output, "archive", timeout=900,
        )
        archive = output / "dependencies.tar.gz"
        write_json_atomic(output / "PREPARED.json", {
            **config, "status": "OFFLINE_DEPENDENCIES_VERIFIED", "at": now(),
            "archive": str(archive), "archive_sha256": digest(archive),
            "archive_bytes": archive.stat().st_size,
        })
        write_json_atomic(output / "STATE.json", {
            "status": "COMPLETE", "pid": os.getpid(), "updated_at": now(),
        })
    except Exception as error:
        write_json_atomic(output / "STATE.json", {
            "status": "BLOCKED", "pid": os.getpid(), "updated_at": now(),
            "error_type": type(error).__name__, "error": str(error),
        })
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        command(["docker", "stop", "--time", "5", name], output, "stop", timeout=60)


async def install_cache(environment, manifest: Path, task: Path, output: Path) -> None:
    prepared = verified_cache(manifest, task)
    result = await environment.exec("mkdir -p /opt/phiagent-verifier-cache", user="root")
    if result.return_code:
        raise RuntimeError("Cannot prepare the owned verifier dependency directory")
    await environment.upload_file(
        Path(prepared["archive"]), "/opt/phiagent-verifier-cache/dependencies.tar.gz",
    )
    result = await environment.exec(
        "tar -C /root -xzf /opt/phiagent-verifier-cache/dependencies.tar.gz", user="root",
        timeout_sec=900,
    )
    (output / "cache-install.log").write_text((result.stdout or "") + (result.stderr or ""))
    if result.return_code:
        raise RuntimeError("Verified dependency archive could not be installed")
    write_json_atomic(output / "VERIFIER_CACHE.json", {
        "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "archive_sha256": prepared["archive_sha256"], "actor_execution_finished": True,
    })


async def record_owned_container(environment, output: Path) -> dict:
    result = await environment._run_docker_compose_command(["ps", "-q", "main"])
    identities = (result.stdout or "").split()
    if len(identities) != 1 or not re.fullmatch(r"[a-f0-9]{12,64}", identities[0]):
        raise RuntimeError("Harbor did not identify exactly one owned main container")
    inspected = await asyncio.to_thread(
        subprocess.run, ["docker", "inspect", "--format", "{{json .}}", identities[0]],
        capture_output=True, text=True, timeout=30, check=True,
    )
    value = json.loads(inspected.stdout)
    if not value["Id"].startswith(identities[0]) or value["State"]["Status"] != "running":
        raise RuntimeError("Owned native container identity or startup state differs")
    receipt = {
        "container_id": value["Id"], "image_id": value["Image"], "name": value["Name"],
        "captured_before_actor": True, "at": now(),
        "network_owner": network_owner(value),
    }
    write_json_atomic(output / "NATIVE_CONTAINER.json", receipt)
    return receipt


async def remove_scored_container(output: Path) -> None:
    receipt = load(output / "NATIVE_CONTAINER.json")
    identity = receipt["container_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", identity):
        raise ValueError("Invalid recorded native container identity")
    inspected = await asyncio.to_thread(
        subprocess.run, ["docker", "inspect", "--format", "{{json .}}", identity],
        capture_output=True, text=True, timeout=30, check=True,
    )
    value = json.loads(inspected.stdout)
    if value["Id"] != identity or value["State"]["Status"] != "exited":
        raise RuntimeError("Only the stopped scored native workspace may be removed")
    removed = await asyncio.to_thread(
        subprocess.run, ["docker", "rm", identity],
        capture_output=True, text=True, timeout=60, check=True,
    )
    write_json_atomic(output / "SCORED_WORKSPACE_CLEANUP.json", {
        "container_id": identity, "removed": True, "at": now(), "stdout": removed.stdout,
    })
    network_receipt = output / "NETWORK_CLEANUP.json"
    existing = load(network_receipt) if network_receipt.exists() else None
    if existing is None or existing.get("status") != "COMPLETE" or existing.get("retained"):
        await asyncio.to_thread(retire_networks, network_owner(value), network_receipt)


async def release_stopped_trial_network(output: Path) -> None:
    path = output / "NATIVE_CONTAINER.json"
    if not path.exists():
        write_json_atomic(output / "NETWORK_CLEANUP.json", {
            "status": "NO_NATIVE_CONTAINER_REGISTERED", "at": now(),
            "reason": "Environment failed before actor start; no guessed network ownership",
        })
        return
    receipt = load(path)
    owner = receipt.get("network_owner")
    if owner is None:
        identity = receipt["container_id"]
        inspected = await asyncio.to_thread(
            subprocess.run, ["docker", "inspect", "--format", "{{json .}}", identity],
            capture_output=True, text=True, timeout=30, check=True,
        )
        owner = network_owner(json.loads(inspected.stdout))
    await asyncio.to_thread(retire_networks, owner, output / "NETWORK_CLEANUP.json")


def cached_trial_config(job: dict, trial_name: str) -> dict:
    if (
        len(job.get("tasks", [])) != 1 or len(job.get("agents", [])) != 1
        or job.get("n_attempts") != 1 or job.get("retry", {}).get("max_retries") != 0
        or job.get("timeout_multiplier") != 1.0 or job.get("job_name") != "native"
    ):
        raise ValueError("The cache adapter requires one unchanged native trial without retries")
    return {
        "task": job["tasks"][0], "trial_name": trial_name,
        "trials_dir": str(Path(job["jobs_dir"]) / job["job_name"]),
        "agent": job["agents"][0], "environment": job["environment"],
        "verifier": job["verifier"], "timeout_multiplier": 1.0,
    }


async def cached_trial(job_path: Path, cache: Path, output: Path) -> None:
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial, TrialEvent

    if socket.gethostname() != HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Native cached trials require the authorized A800 CPU host")
    job = load(job_path)
    task = Path(job["tasks"][0]["path"])
    prepared = verified_cache(cache, task)
    config = cached_trial_config(job, task.name + "__" + output.name[:12])
    config["verifier"] = {
        **config["verifier"], "env": {
            **config["verifier"].get("env", {}),
            **prepared["verifier_environment"], "UV_OFFLINE": "1",
        },
    }
    trial = await Trial.create(TrialConfig.model_validate(config))

    async def prepare_native_verifier(event):
        if event.task_name != task.name:
            raise ValueError("Native verifier-cache hook task identity differs")
        await install_cache(trial.agent_environment, cache, task, output)

    trial.add_hook(TrialEvent.VERIFICATION_START, prepare_native_verifier)
    write_json_atomic(output / "CACHED_TRIAL_CONFIG.json", config)
    await trial.run()


def validate_frozen_test_binding(
    binding_path: Path, callback: Path, request: dict, old: dict, container: str,
) -> dict:
    binding = load(binding_path)
    predecessor = Path(binding.get("predecessor", ""))
    record_path = Path(binding.get("native_record", ""))
    original_path = Path(old["provenance"]["native_trial_result"])
    state_path = predecessor / "STATE.json"
    policy_path = predecessor / "POLICY.json"
    release_path = predecessor / "TEST_RELEASE.json"
    config_path = predecessor / "CONFIG.json"
    required = {
        state_path: binding.get("predecessor_state_sha256"),
        policy_path: binding.get("policy_file_sha256"),
        release_path: binding.get("test_release_sha256"),
        config_path: binding.get("predecessor_config_sha256"),
        record_path: binding.get("native_record_sha256"),
        original_path: binding.get("native_result_sha256"),
        callback / "REQUEST.json": binding.get("request_sha256"),
        callback / "RESULT.json": binding.get("result_sha256"),
    }
    if (
        binding.get("schema") != "frozen-codex-test-verifier-recovery/v1"
        or request.get("phase") != "test"
        or binding.get("phase") != "test"
        or binding.get("additional_actor_calls") != 0
        or binding.get("container_id") != container
        or any(not expected or not path.is_file() or digest(path) != expected
               for path, expected in required.items())
    ):
        raise ValueError("Frozen TEST verifier recovery binding is incomplete or changed")
    state = load(state_path)
    policy = load(policy_path)
    record = load(record_path)
    original = load(original_path)
    proc = Path("/proc") / str(state.get("pid")) / "stat"
    if proc.exists() and proc.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        raise ValueError("Frozen TEST predecessor controller is still live")
    if (
        state.get("status") != "BLOCKED"
        or state.get("test_opened") is not True
        or binding.get("policy_sha256") != policy.get("policy_sha256")
        or binding.get("source_sha256") != policy.get("source_sha256")
        or record.get("status") != "BLOCKED"
        or record.get("phase") != "test"
        or record.get("task_id") != old.get("task_id")
        or record.get("trial_id") != binding.get("trial_id")
        or record.get("provider_usage", {}).get("usage_status") != "MEASURED"
        or original.get("verifier_result") is not None
    ):
        raise ValueError("Only an unscored, frozen, measured TEST attempt may be recovered")
    return binding


def drained_active_report_proof(callback: Path, report: dict, task_id: str) -> dict | None:
    """A stale report cannot own a task after both pinned processes have exited."""
    if task_id not in report.get("active", {}):
        return None
    from scripts import run_industrial_capability_baselines as native
    owner = callback.parent.parent
    actor_receipt = load(callback / "PROCESS.json")
    owner_receipt = load(owner / "OWNER.json")
    reported = report["active"][task_id]
    if (native.process_live(actor_receipt) or native.process_live(owner_receipt)
            or native.process_live(reported)):
        raise ValueError("The original actor still has a live native owner")
    return {"original_report_sha256": digest(owner / "REPORT.json"),
            "historical_active_entry": reported,
            "original_actor_process_sha256": digest(callback / "PROCESS.json"),
            "original_owner_process_sha256": digest(owner / "OWNER.json"),
            "actor_and_owner_live": False, "additional_actor_calls": 0}


def capability_recovery_input(
    callback: Path, *, unexecuted_reward: bool = False,
) -> tuple[dict, dict, dict]:
    row = load(callback / "RESULT.json")
    if (row.get("phase") != "train" or row.get("status") != "BLOCKED"
            or row.get("native_score") is not None
            or Path(row["attempt_id"]).resolve() != callback.resolve()
            or row.get("track") not in ("codex", "deepseek")):
        raise ValueError("Recover only an identified, unscored capability TRAIN attempt")
    native_path = Path(row["native_result"])
    if (not native_path.resolve().is_relative_to(callback.resolve())
            or digest(native_path) != row["native_result_sha256"]):
        raise ValueError("Original capability native result changed or escaped its trial")
    native = load(native_path)
    rewards = (native.get("verifier_result") or {}).get("rewards")
    if native.get("task_name") != row["task_id"]:
        raise ValueError("Do not recover a different native task")
    if unexecuted_reward:
        if (row.get("error") != "NO_EXECUTED_TEST_RECEIPT"
                or rewards != {"reward": 0.0}
                or type(rewards["reward"]) not in (int, float)
                or list(callback.glob("**/verifier/ctrf.json"))
                or (native.get("exception_info") or {}).get("exception_type")
                not in (None, "AgentTimeoutError")):
            raise ValueError("Only a blocked wrapper zero with no executed native test receipt is eligible")
        from scripts import run_industrial_capability_baselines as capability

        owner = callback.parent.parent
        capability.check_sources(owner)
        config = load(owner / "CONFIG.json")
        records = capability.train_records(load(owner / "SPLIT_MANIFEST.json"), config)
        if row["task_id"] not in {record["id"] for record in records}:
            raise ValueError("Wrapper-reward recovery is restricted to frozen TRAIN membership")
        for path in (owner / "OWNER.json", callback / "PROCESS.json"):
            if capability.process_live(load(path)):
                raise ValueError("The original actor/controller is still live")
    elif rewards is not None:
        raise ValueError("Do not rerun an already supplied native reward or another task")
    owner = callback.parent.parent
    report = load(owner / "REPORT.json")
    drained_owner = drained_active_report_proof(callback, report, row["task_id"])
    expected = load(owner / "TASK_SOURCE_SHA256.json")[row["task_id"]]
    from scripts.run_deepseek_terminal import task_file_hashes

    if task_file_hashes(callback / "task-source" / row["task_id"]) != expected:
        raise ValueError("Frozen capability task bytes changed")
    container_path = callback / (
        "native-run/NATIVE_CONTAINER.json" if row["track"] == "codex" else "NATIVE_CONTAINER.json"
    )
    container = load(container_path)
    if container.get("captured_before_actor") is not True:
        raise ValueError("Native workspace identity was not captured before the actor")
    source = {
        "source_kind": "industrial_capability_native_trial",
        "capability_result": str(callback / "RESULT.json"),
        "capability_result_sha256": digest(callback / "RESULT.json"),
        "container_receipt": str(container_path),
        "container_receipt_sha256": digest(container_path),
        "track": row["track"], "shared_prompt_sha256": row["shared_prompt_sha256"],
        "partition_sha256": row["partition_sha256"],
        "original_actor_cost_record": row["cost"],
        "unexecuted_wrapper_reward": rewards if unexecuted_reward else None,
        "executed_native_tests_required": unexecuted_reward,
        "drained_historical_active_owner": drained_owner,
    }
    old = {
        "task_id": row["task_id"], "native_score": None, "usage": row["cost"],
        "provenance": {"native_trial_result": str(native_path)},
    }
    return {"phase": "train"}, old, {**container, "source_binding": source}


def validation_request_matches(request: dict, run: dict, config: dict) -> bool:
    from scripts.run_codex_chisel_terminal import task_local_suffix

    if (
        request.get("phase") != "validation"
        or request.get("task", {}).get("task_id") != run.get("task_id")
        or request.get("seed") != run.get("seed")
    ):
        return False
    if not isinstance(request.get("arm"), str) or type(request.get("seed")) is not int:
        raise ValueError("The matching validation request lacks its frozen arm or seed")
    baseline = request["arm"] in ("baseline", "noop")
    tag = "baseline" if baseline else "c-" + request["arm"][:16]
    repetition = request["seed"] - config["partition"]["seed"]
    expected_trial = (
        f"tb-validation-{tag}-r{repetition}-{run['task_id']}"
        + config.get("execution_suffix", "")
    )
    return (
        repetition >= 0 and run.get("trial_id") == expected_trial
        and run.get("method") == ("codex" if baseline else "chisel_codex")
        and run.get("residual_sha256")
        == hashlib.sha256(task_local_suffix(request).encode()).hexdigest()
    )


def validation_recovery_input(
    callback: Path,
) -> tuple[dict, dict, dict, Path, Path]:
    run_path = callback / "RUN.json"
    run = load(run_path)
    native_path = callback / "native_result.json"
    native = load(native_path)
    container_path = callback / "NATIVE_CONTAINER.json"
    container = load(container_path)
    controller = callback.parents[6]
    state_path = controller / "STATE.json"
    state = load(state_path)
    pid = state.get("pid")
    process = Path("/proc") / str(pid) / "stat"
    if process.exists() and process.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        raise ValueError("The validation predecessor controller is still live")
    controller_config = load(controller / "CONFIG.json")
    requests = [
        path for path in (controller / "terminal/validation/requests").glob("*.json")
        if validation_request_matches(load(path), run, controller_config)
    ]
    if len(requests) != 1:
        raise ValueError("Validation recovery requires one durable matching request")
    request_path = requests[0]
    request = load(request_path)
    config = load(callback / "CONFIG.json")
    task_path = Path(config.get("task", {}).get("path", ""))
    usage = run.get("provider_usage", {})
    if (
        state.get("status") != "BLOCKED"
        or state.get("test_opened") is not False
        or state.get("error", {}).get("message")
        != "Native execution is unscored; preserve it for explicit recovery"
        or run.get("status") != "BLOCKED"
        or run.get("phase") != "validation"
        or run.get("score") is not None
        or run.get("method") not in ("codex", "chisel_codex")
        or usage.get("usage_status") != "MEASURED"
        or usage.get("unresolved_call_ids")
        or native.get("task_name") != run.get("task_id")
        or native.get("verifier_result") is not None
        or request.get("phase") != "validation"
        or request.get("task", {}).get("task_id") != run.get("task_id")
        or request.get("seed") != run.get("seed")
        or task_path.name != run.get("task_id")
        or not task_path.is_dir()
        or container.get("captured_before_actor") is not True
    ):
        raise ValueError("Only the blocked, measured, sealed validation attempt may be recovered")
    source = {
        "source_kind": "industrial_success_validation_native_trial",
        "controller_state": str(state_path),
        "controller_state_sha256": digest(state_path),
        "validation_request": str(request_path),
        "validation_request_sha256": digest(request_path),
        "native_record": str(run_path),
        "native_record_sha256": digest(run_path),
        "native_result_sha256": digest(native_path),
        "container_receipt": str(container_path),
        "container_receipt_sha256": digest(container_path),
        "partition_sha256": run["partition_sha256"],
        "shared_prompt_sha256": run["shared_prompt"]["sha256"],
        "original_actor_cost_record": usage,
        "test_opened": False,
    }
    old = {
        "task_id": run["task_id"], "native_score": None, "usage": usage,
        "provenance": {"native_trial_result": str(native_path)},
    }
    return request, old, {**container, "source_binding": source}, task_path, run_path


def coding_confirmation_recovery_input(
    callback: Path,
) -> tuple[dict, dict, dict, Path, Path]:
    from scripts import run_industrial_coding_confirmation as confirmation

    controller = callback.parents[5]
    confirmation.check_run(controller)
    state_path = controller / "STATE.json"
    owner_path = controller / "OWNER.json"
    failure_path = controller / "FAILURE.json"
    state = load(state_path)
    owner = load(owner_path)
    failure = load(failure_path)
    process = Path("/proc") / str(owner.get("pid")) / "stat"
    owner_alive = False
    if process.exists():
        fields = process.read_text().rsplit(")", 1)[1].split()
        owner_alive = fields[0] not in ("Z", "X", "x") and (
            owner.get("start_ticks") is None or str(owner["start_ticks"]) == fields[19]
        )

    outcome_path = callback.parent / "OUTCOME.json"
    run_path = callback / "RUN.json"
    native_path = callback / "native_result.json"
    container_path = callback / "NATIVE_CONTAINER.json"
    outcome = load(outcome_path)
    run = load(run_path)
    native = load(native_path)
    container = load(container_path)
    config = load(callback / "CONFIG.json")
    task_path = Path(config.get("task", {}).get("path", ""))
    rewards = (native.get("verifier_result") or {}).get("rewards")
    usage = outcome.get("usage")
    if (
        owner_alive
        or state.get("status") != "BLOCKED"
        or state.get("test_opened") is not False
        or failure.get("error")
        != "A new native verdict/accounting failed; preserved all active work and stopped new submissions"
        or outcome.get("status") != "BLOCKED"
        or outcome.get("score") is not None
        or outcome.get("executed_native_tests_verified") is not False
        or outcome.get("usage_complete") is not True
        or not isinstance(usage, dict)
        or any(type(usage.get(key)) is not int or usage[key] < 0
               for key in ("input_tokens", "output_tokens", "total_tokens"))
        or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
        or run.get("status") != "BLOCKED"
        or run.get("error") != "NO_EXECUTED_TEST_RECEIPT"
        or run.get("phase") != "validation"
        or run.get("score") is not None
        or run.get("trial_id") != outcome.get("trial_id")
        or run.get("task_id") != outcome.get("task_id")
        or rewards != {"reward": 0.0}
        or list(callback.glob("native/**/verifier/ctrf.json"))
        or native.get("task_name") != run.get("task_id")
        or not task_path.is_dir()
        or task_path.name != run.get("task_id")
        or container.get("captured_before_actor") is not True
    ):
        raise ValueError(
            "Only the drained, measured coding-confirmation validation wrapper failure may recover"
        )
    source = {
        "source_kind": "industrial_coding_confirmation_validation_native_trial",
        "controller_state": str(state_path),
        "controller_state_sha256": digest(state_path),
        "controller_owner": str(owner_path),
        "controller_owner_sha256": digest(owner_path),
        "controller_failure": str(failure_path),
        "controller_failure_sha256": digest(failure_path),
        "outcome": str(outcome_path),
        "outcome_sha256": digest(outcome_path),
        "native_record": str(run_path),
        "native_record_sha256": digest(run_path),
        "native_result_sha256": digest(native_path),
        "container_receipt": str(container_path),
        "container_receipt_sha256": digest(container_path),
        "partition_sha256": run["partition_sha256"],
        "shared_prompt_sha256": run["shared_prompt"]["sha256"],
        "source_sha256": run["source_sha256"],
        "task_source_sha256": run["task_source_sha256"],
        "original_actor_cost_record": usage,
        "test_opened": False,
        "additional_actor_calls": 0,
    }
    request = {
        "phase": "validation", "arm": outcome["arm"], "seed": outcome["seed"],
        "task": {"task_id": outcome["task_id"]},
    }
    old = {
        "task_id": run["task_id"], "native_score": None, "usage": usage,
        "provenance": {"native_trial_result": str(native_path)},
    }
    return request, old, {**container, "source_binding": source}, task_path, outcome_path


def figure3_codex_input(callback: Path, *, phase: str) -> tuple[dict, dict, dict, Path, Path]:
    if phase not in ("train", "validation"):
        raise ValueError("Only source-bound TRAIN/validation actors can be recovered")
    from scripts import run_figure3_codex_recovery as owner_module
    from scripts import run_industrial_capability_baselines as native

    controller = callback.parents[5]
    owner_module.check_sources(controller)
    state, owner = load(controller / "STATE.json"), load(controller / "OWNER.json")
    outcome_path = callback.parent / "OUTCOME.json"
    outcome, run = load(outcome_path), load(callback / "RUN.json")
    config, original = load(callback / "CONFIG.json"), load(callback / "native_result.json")
    container = load(callback / "NATIVE_CONTAINER.json")
    usage = outcome.get("usage", {})
    partition = load(controller / "SPLIT_MANIFEST.json")
    task = Path(config["task"]["path"])
    if (
        native.process_live(owner) or state.get("status") != "BLOCKED"
        or state.get("phase") != phase or state.get("test_opened") is not False
        or outcome.get("phase") != phase
        or outcome.get("arm") not in owner_module.ARMS
        or callback.parents[2].name != outcome.get("arm")
        or outcome.get("status") != "BLOCKED" or outcome.get("score") is not None
        or outcome.get("usage_complete") is not True
        or any(type(usage.get(key)) is not int or usage[key] < 0
               for key in ("input_tokens", "output_tokens", "total_tokens"))
        or usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
        or run.get("phase") != phase or run.get("status") != "BLOCKED"
        or run.get("score") is not None or run.get("task_id") != outcome.get("task_id")
        or run.get("trial_id") != outcome.get("trial_id")
        or outcome["task_id"] not in partition["splits"][phase]
        or task.name != outcome["task_id"] or not task.is_dir()
        or container.get("captured_before_actor") is not True
        or original.get("task_name") != task.name
        or list(callback.glob("native/**/verifier/ctrf.json"))
        or (original.get("verifier_result") or {}).get("rewards") not in (None, {"reward": 0.0})
    ):
        raise ValueError("Recover only the drained fully metered Figure3 matched source-bound Codex actor")
    if phase == "validation":
        instruction = load(controller / "FROZEN_INSTRUCTION.json")
        if digest(controller / "INSTRUCTION.txt") != instruction["applied_residual_sha256"]:
            raise ValueError("The TRAIN-frozen instruction changed before validation scoring")
    request = {"phase": phase, "task": {"task_id": task.name}}
    old = {
        "task_id": task.name, "native_score": None, "usage": usage,
        "provenance": {"native_trial_result": str(callback / "native_result.json")},
    }
    binding = {
        "source_kind": f"figure3_codex_native_{phase}_same_actor",
        "arm": outcome["arm"],
        "outcome": str(outcome_path), "outcome_sha256": digest(outcome_path),
        "controller_owner": str(controller / "OWNER.json"),
        "controller_owner_sha256": digest(controller / "OWNER.json"),
        "original_actor_cost_record": usage, "additional_actor_calls": 0,
        "original_actor": {"path": str(callback / "RUN.json"),
                           "sha256": digest(callback / "RUN.json")},
        "original_outcome": {"path": str(outcome_path), "sha256": digest(outcome_path)},
        "test_opened": False,
    }
    return request, old, {**container, "source_binding": binding}, task, outcome_path


def figure3_codex_bank_train_input(callback: Path) -> tuple[dict, dict, dict, Path, Path]:
    """Bind a drained core-bank TRAIN actor to a separate scorer, without replay."""
    from scripts import run_industrial_capability_baselines as native
    from scripts.run_codex_terminal_bank_campaign import audit_physical_usage
    from phiagent.harness.codex_terminal_bank_contract import CORE_METHODS, PARTITION_SHA256
    from phiagent.harness.industrial_common_prompt import PROMPT_SHA256

    controller = callback.parents[5]
    hashes = load(controller / "SOURCE_SHA256.json")
    for name, expected in hashes.items():
        path = controller / "source" / name
        if not path.resolve().is_relative_to((controller / "source").resolve()) or digest(path) != expected:
            raise ValueError("Frozen native core-bank source changed")
    state, owner = load(controller / "STATE.json"), load(controller / "OWNER.json")
    outcome_path = callback.parent / "OUTCOME.json"
    outcome, run = load(outcome_path), load(callback / "RUN.json")
    config, original = load(callback / "CONFIG.json"), load(callback / "native_result.json")
    container = load(callback / "NATIVE_CONTAINER.json")
    task = Path(config["task"]["path"])
    campaign = load(controller / "CONFIG.json")
    native_config = load(Path(campaign["parent_root"]) / "NATIVE_CONFIG.json")
    partition_path = Path(native_config["partition_dir"]) / "SPLIT_MANIFEST.json"
    partition = load(partition_path)
    usage = run.get("provider_usage", {})
    stale_active = []
    for entry in state.get("active", []):
        if (entry.get("phase") != "train" or entry.get("arm") not in CORE_METHODS
                or entry.get("task_id") not in partition["splits"]["train"]):
            raise ValueError("Unbound historical active core-bank task")
        folder = controller / "trials/train" / entry["arm"] / "20260928" / entry["task_id"]
        process = load(folder / "PROCESS.json")
        if native.process_live(process) or not (folder / "OUTCOME.json").is_file():
            raise ValueError("Historical core-bank task still has a live actor")
        stale_active.append({"entry":entry,"process": {"path":str(folder/"PROCESS.json"),
            "sha256":digest(folder/"PROCESS.json")},"outcome": {"path":str(folder/"OUTCOME.json"),
            "sha256":digest(folder/"OUTCOME.json")},"actor_live":False})
    unexecuted_zero = (run.get("error") == "NO_EXECUTED_TEST_RECEIPT"
        and original.get("verifier_result") == {"rewards":{"reward":0.0}})
    if (
        native.process_live(owner) or not (controller / "BLOCKED.json").is_file()
        or state.get("phase") != "train"
        or state.get("test_opened") is not False
        or digest(partition_path) != PARTITION_SHA256
        or partition.get("counts") != {"train": 18, "validation": 18, "test": 53}
        or outcome.get("phase") != "train" or outcome.get("arm") not in CORE_METHODS
        or callback.parents[2].name != outcome.get("arm")
        or outcome.get("valid") is not False or outcome.get("native_score") is not None
        or outcome.get("native_run") != str(callback / "RUN.json")
        or outcome.get("seed") != 20260928
        or outcome.get("task_id") not in partition["splits"]["train"]
        or run.get("phase") != "train" or run.get("status") != "BLOCKED"
        or run.get("model") != "deepseek-flash" or run.get("score") is not None
        or run.get("task_id") != outcome.get("task_id")
        or run.get("method") != CORE_METHODS[outcome["arm"]]
        or run.get("partition_sha256") != PARTITION_SHA256
        or run.get("shared_prompt", {}).get("sha256") != PROMPT_SHA256
        or task.name != outcome["task_id"] or not task.is_dir()
        or container.get("captured_before_actor") is not True
        or original.get("task_name") != task.name
        or original.get("trial_name") != run.get("trial_id")
        or list(callback.glob("native/**/verifier/ctrf.json"))
        or (original.get("verifier_result") is not None and not unexecuted_zero)
        or usage.get("usage_status") != "MEASURED"
    ):
        raise ValueError("Recover only a drained fully metered native core-bank TRAIN actor")
    total, calls = audit_physical_usage(usage)
    if total["total_tokens"] != outcome.get("actor_tokens"):
        raise ValueError("Original core-bank paid actor costs changed")
    contract = run["terminal_bank"]
    if contract["arm"] != outcome["arm"] or digest(Path(contract["contract"])) != contract["contract_sha256"]:
        raise ValueError("TRAIN-frozen native action bank changed")
    request = {"phase": "train", "task": {"task_id": task.name}}
    old = {"task_id": task.name, "native_score": None, "usage": total,
           "provenance": {"native_trial_result": str(callback / "native_result.json")}}
    binding = {
        "source_kind": "figure3_codex_native_core_bank_train_same_actor",
        "arm": outcome["arm"], "controller": str(controller),
        "original_actor": {"path": str(callback / "RUN.json"), "sha256": digest(callback / "RUN.json")},
        "original_outcome": {"path": str(outcome_path), "sha256": digest(outcome_path)},
        "source_manifest_sha256": digest(controller / "SOURCE_SHA256.json"),
        "original_actor_cost_record": total, "physical_actor_calls": len(calls),
        "additional_actor_calls": 0, "test_opened": False,
        "drained_historical_active": stale_active,
        "unexecuted_zero_reward": unexecuted_zero,
    }
    return request, old, {**container, "source_binding": binding}, task, outcome_path


def figure3_codex_train_input(callback: Path) -> tuple[dict, dict, dict, Path, Path]:
    return figure3_codex_input(callback, phase="train")

def figure3_codex_validation_input(callback: Path) -> tuple[dict, dict, dict, Path, Path]:
    return figure3_codex_input(callback, phase="validation")


def protected_validation_failure(row: dict, original: dict) -> bool:
    """Admit missing verdicts from settled native actor/verifier timeouts."""
    if row.get('error') == 'NO_EXECUTED_TEST_RECEIPT':
        return True
    info = original.get('exception_info') or {}
    return (row.get('error') == 'Official verifier did not supply its unique native reward'
            and info.get('exception_type') in ('AgentTimeoutError', 'VerifierTimeoutError')
            and row.get('native_exception') == info)


def drained_validation_source_fault(state, stage_state, handoff, plan):
    if state.get("status") != "BLOCKED_ORCHESTRATOR_SOURCE_FAULT":
        return False
    from scripts import run_figure3_deepseek_protected as protected
    fault = protected.verified(state["source_fault"])
    return (
        fault.get("test_opened") is False
        and fault.get("actor_signals_sent") == 0
        and fault.get("resume_in_new_source_frozen_run") is True
        and handoff.get("include_cot_only_control") is True
        and stage_state.get("status") == "DRAINED_ORCHESTRATOR_SOURCE_FAULT"
        and not stage_state.get("active")
        and len(plan) == 36
        and state.get("desired_matched_validation_slots") == 54
    )


def figure3_deepseek_validation_input(callback: Path) -> tuple[dict, dict, dict, Path, Path]:
    """Bind a drained protected VAL actor without treating adapter TRAIN metadata as feedback."""
    from scripts import run_industrial_capability_baselines as native
    from scripts import run_figure3_deepseek_protected as protected
    from scripts.run_deepseek_terminal import task_file_hashes
    arm_root = callback.parent.parent
    stage = arm_root.parent
    controller = stage.parent
    if stage.name != "validation" or not controller.is_relative_to(protected.RUN_ROOT):
        raise ValueError("Only the registered protected validation root may supply a scorer repair")
    native.check_sources(arm_root)
    handoff = load(controller / "OWNER_HANDOFF.json")
    lease = load(controller / "LEASE.json")
    state = load(controller / "STATUS.json")
    stage_state = load(stage / "STATUS.json")
    row = load(callback / "PROTECTED_RESULT.json")
    original = load(callback / "native_result.json")
    config = load(arm_root / "CONFIG.json")
    partition = load(arm_root / "SPLIT_MANIFEST.json")
    plan = load(stage / "PLAN.json")
    matches = [x for x in plan if x["slot"] == row.get("slot")]
    drained_source_fault = drained_validation_source_fault(state, stage_state, handoff, plan)
    if (protected.live_process(lease) or stage_state.get("active")
        or state.get("test_opened") is not False
        or (state.get("status") != "VALIDATION_NO_ADMISSIBLE_TRAIN_FROZEN_CANDIDATE" and not drained_source_fault)
        or row.get("phase") != "validation" or row.get("track") != "deepseek"
        or row.get("native_score") is not None or row.get("status") != "BLOCKED"
        or row.get("cost", {}).get("usage_status") != "MEASURED"
        or not protected.usage_valid(row["cost"].get("usage"))
        or not protected_validation_failure(row, original)
        or row.get("partition_sha256") != protected.PARTITION_SHA
        or config.get("protected_phase") != "validation"
        or row.get("task_id") not in partition["splits"]["validation"]
        or len(matches) != 1 or protected.case_root(stage, matches[0]) != callback
        or original.get("task_name") != row["task_id"]
        or digest(callback / "native_result.json") != row["native_result_sha256"]
        or (original.get("verifier_result") or {}).get("rewards") not in (None, {"reward": 0.0})
        or list(callback.glob("harbor/**/verifier/ctrf.json"))
        or native.process_live(load(callback / "PROCESS.json"))):
        raise ValueError("Require the drained, source-bound, metered protected VAL actor without an executed verdict")
    item = matches[0]
    if item["arm"] != "native":
        candidate = next(x for x in handoff["train_candidates"] if x["sha256"] == item["candidate_sha256"])
        protected.verified(candidate["artifact"])
        if config["frozen_train_instruction"]["sha256"] != item["candidate_sha256"]:
            raise ValueError("The validation actor must retain its TRAIN-frozen skill")
    task = callback / "task-source" / row["task_id"]
    if task_file_hashes(task) != load(arm_root / "TASK_SOURCE_SHA256.json")[row["task_id"]]:
        raise ValueError("Original validation task bytes changed")
    ledger = load(callback / "PROVIDER_CALLS.json")
    totals = dict.fromkeys(("input_tokens", "output_tokens", "total_tokens"), 0)
    seen = set()
    for call in ledger["calls"]:
        if (call["call_id"] in seen or not call["ended"] or call["model"] != "deepseek-flash"
            or not protected.usage_valid(call.get("usage"))):
            raise ValueError("The original validation actor's physical fees remain unresolved")
        seen.add(call["call_id"])
        for key in totals: totals[key] += call["usage"][key]
    if totals != row["cost"]["usage"]:
        raise ValueError("Validation actor costs differ from its physical calls")
    container = load(callback / "NATIVE_CONTAINER.json")
    if container.get("captured_before_actor") is not True:
        raise ValueError("A same-actor scorer requires the original captured native container")
    record = callback / "PROTECTED_RESULT.json"
    binding = {"source_kind": "figure3_deepseek_protected_validation_same_actor",
        "arm": row["arm"], "slot": row["slot"], "additional_actor_calls": 0,
        "outcome": str(record), "outcome_sha256": digest(record),
        "original_actor_cost_record": row["cost"], "test_opened": False,
        "original_adapter_result": {"path": str(callback / "RESULT.json"), "sha256": digest(callback / "RESULT.json")},
        "protected_owner_handoff": {"path": str(controller / "OWNER_HANDOFF.json"), "sha256": digest(controller / "OWNER_HANDOFF.json")}}
    return ({"phase": "validation", "task": {"task_id": row["task_id"]}},
        {"task_id": row["task_id"], "native_score": None, "usage": row["cost"],
         "provenance": {"native_trial_result": str(callback / "native_result.json")}},
        {**container, "source_binding": binding}, task, record)


def snapshot_timeout(value: int) -> int:
    if type(value) is not int or not 300 <= value <= 3600:
        raise ValueError("Snapshot infrastructure timeout must be 300--3600 seconds")
    return value


async def recover(
    callback: Path,
    cache: Path | None,
    output: Path,
    *,
    capability: bool = False,
    unexecuted_reward: bool = False,
    validation: bool = False,
    coding_confirmation: bool = False,
    native_dependency_setup: bool = False,
    native_dependency_environment: dict[str, str] | None = None,
    frozen_test_binding: Path | None = None,
    sam_verifier_cache: Path | None = None,
    figure3_codex_train: bool = False,
    figure3_codex_validation: bool = False,
    figure3_codex_bank_train: bool = False,
    figure3_deepseek_validation: bool = False,
    snapshot_timeout_seconds: int = 300,
    match_native_cpu_threads: bool = False,
    install_public_ca_certificates: bool = False,
) -> None:
    snapshot_timeout_seconds = snapshot_timeout(snapshot_timeout_seconds)
    from harbor.environments.base import ExecResult
    from harbor.models.task.config import TaskOS
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from harbor.verifier.verifier import Verifier
    from types import SimpleNamespace

    reserve(output)
    if sum((capability, validation, coding_confirmation, figure3_codex_train, figure3_codex_validation, figure3_codex_bank_train, figure3_deepseek_validation)) > 1 or (
        unexecuted_reward and not (capability or figure3_codex_train or figure3_codex_validation or figure3_codex_bank_train or figure3_deepseek_validation)
    ):
        raise ValueError("Choose one recovery input identity")
    if figure3_codex_bank_train:
        request, old, environment_receipt, task_path, callback_record = figure3_codex_bank_train_input(callback)
        figure3_codex_train = True  # Dependency plumbing only; source kind stays core-bank TRAIN.
    elif figure3_deepseek_validation:
        request, old, environment_receipt, task_path, callback_record = figure3_deepseek_validation_input(callback)
        figure3_codex_train = True  # Reuse only dependency/receipt plumbing; actual identity remains DeepSeek VAL.
    elif figure3_codex_train or figure3_codex_validation:
        request, old, environment_receipt, task_path, callback_record = (
            figure3_codex_input(callback, phase="validation" if figure3_codex_validation else "train")
        )
        figure3_codex_train = True  # Shared native dependency/receipt plumbing; phase remains immutable.
    elif capability:
        request, old, environment_receipt = capability_recovery_input(
            callback, unexecuted_reward=unexecuted_reward,
        )
        task_path = callback / "task-source" / old["task_id"]
        callback_record = callback / "RESULT.json"
    elif validation:
        request, old, environment_receipt, task_path, callback_record = (
            validation_recovery_input(callback)
        )
    elif coding_confirmation:
        request, old, environment_receipt, task_path, callback_record = (
            coding_confirmation_recovery_input(callback)
        )
    else:
        request, old = load(callback / "REQUEST.json"), load(callback / "RESULT.json")
        private = Path(old["provenance"]["private_artifacts"])
        environment_receipt = load(private / "HARBOR_ENVIRONMENT.json")
        task_path = callback / "task-source" / old["task_id"]
        callback_record = callback / "RESULT.json"
    provenance = old["provenance"]
    original_path = Path(provenance["native_trial_result"])
    original = load(original_path)
    container = environment_receipt["container_id"]
    test_binding = None
    if old.get("native_score") is not None:
        raise ValueError("Only explicitly unscored verifier failures are eligible")
    if request["phase"] == "test":
        if capability or frozen_test_binding is None:
            raise ValueError("TEST recovery requires an explicit frozen-test binding")
        test_binding = validate_frozen_test_binding(
            frozen_test_binding, callback, request, old, container,
        )
    elif request["phase"] not in ("train", "validation") or frozen_test_binding is not None:
        raise ValueError("Only unscored TRAIN/validation or frozen TEST failures are eligible")
    info = json.loads(subprocess.check_output(
        ["docker", "inspect", "--format", "{{json .}}", container], text=True, timeout=30,
    ))
    validate_retained_container(info, container)
    native_setup = None
    dependency_mode = "verified_uv_cache"
    if native_dependency_setup and (cache is not None or capability or validation or coding_confirmation):
        raise ValueError("Explicit native dependency setup is only for uncached legacy recovery")
    if native_dependency_environment is not None and not (native_dependency_setup or figure3_codex_train or capability):
        raise ValueError("Dependency transport override requires explicit native dependency setup or a bound capability actor")
    if sam_verifier_cache is not None:
        require_sam_cache_mode(
            capability=capability, task_id=old["task_id"], phase=request["phase"], cache=cache,
            figure3_codex_train=figure3_codex_train,
        )
        from scripts import recover_codex_sam_verifier as sam

        sam.verified_cache(sam_verifier_cache)
        dependency_mode = "hash_verified_official_sam_cache"
        original_environment = (
            load(callback / "CONFIG.json").get("verifier", {}).get("env", {})
            if figure3_codex_train
            else load(callback.parent.parent / "CONFIG.json")["verifier_environment"]
        )
        prepared = {"verifier_environment": {
            **original_environment,
            "BASH_ENV": "/opt/sam-recovery/cache.env", "UV_OFFLINE": "1",
        }}
        prepared["verifier_environment"] = dependency_transport_environment(
            prepared["verifier_environment"], native_dependency_environment,
        )
    elif cache is None:
        if not (capability or coding_confirmation or figure3_codex_train or native_dependency_setup):
            raise ValueError("Legacy native recovery requires its verified dependency cache")
        dependency_mode, native_setup = uncached_native_setup((task_path / "tests/test.sh").read_text())
        if figure3_codex_train:
            prepared = {
                "verifier_environment": (load(callback.parent.parent / "CONFIG.json").get("verifier_environment", {}) if figure3_deepseek_validation else load(callback / "CONFIG.json").get("verifier", {}).get("env", {}))
            }
        elif coding_confirmation or native_dependency_setup:
            prepared = {
                "verifier_environment": (
                    (original.get("config") or {}).get("verifier") or {}
                ).get("env", {})
            }
        else:
            original_config = load(callback.parent.parent / "CONFIG.json")
            prepared = {"verifier_environment": original_config.get("verifier_environment", {})}
        prepared["verifier_environment"] = dependency_transport_environment(
            prepared["verifier_environment"], native_dependency_environment,
        )
    else:
        prepared = verified_cache(cache, task_path)
    task = Task(task_path)
    limit = task.config.verifier.timeout_sec
    cpu_thread_policy = None
    if match_native_cpu_threads:
        cpus = info["HostConfig"].get("NanoCpus", 0) / 1_000_000_000
        if cpus != task.config.environment.cpus:
            raise ValueError("The retained CPU quota must equal the original native task quota")
        prepared["verifier_environment"] = native_cpu_thread_environment(prepared["verifier_environment"], cpus)
        cpu_thread_policy = {"container_cpu_quota": cpus, "library_threads": int(cpus),
                             "scope": "verifier_only", "native_cpu_and_deadline_changed": False}
    name = container_name("phiagent-retained-verifier-", output)
    image = name + ":frozen"
    config = {
        "callback": str(callback), "callback_sha256": digest(callback_record),
        "original_native_result": str(original_path),
        "original_native_result_sha256": digest(original_path),
        "original_container_id": container, "container_state": info["State"],
        "task_id": old["task_id"], "phase": request["phase"],
        "dependency_manifest": str(cache) if cache is not None else None,
        "dependency_manifest_sha256": digest(cache) if cache is not None else None,
        "sam_cache_receipt": str(sam_verifier_cache / "PREPARED.json") if sam_verifier_cache else None,
        "sam_cache_sha256": (
            digest(sam_verifier_cache / "PREPARED.json") if sam_verifier_cache else None
        ),
        "dependency_mode": dependency_mode,
        "install_public_ca_certificates": install_public_ca_certificates,
        "dependency_transport_overrides": native_dependency_environment,
        "native_cpu_thread_policy": cpu_thread_policy,
        "official_verifier_timeout_seconds": limit,
        "additional_actor_calls": 0, "cumulative_token_limit": None,
        "source_binding": environment_receipt.get("source_binding"),
        "original_actor_exception": original.get("exception_info"),
        "unexecuted_wrapper_reward": unexecuted_reward,
        "frozen_test_binding": str(frozen_test_binding) if frozen_test_binding else None,
        "frozen_test_binding_sha256": (
            digest(frozen_test_binding) if frozen_test_binding else None
        ),
        "frozen_test_policy_sha256": (
            test_binding["policy_sha256"] if test_binding else None
        ),
    }
    write_json_atomic(output / "CONFIG.json", config)
    write_json_atomic(output / "SNAPSHOT_INFRASTRUCTURE.json", {
        "timeout_seconds": snapshot_timeout_seconds,
        "native_verifier_timeout_seconds": limit,
        "additional_actor_calls": 0,
        "official_task_limits_changed": False,
    })
    snapshot = command(["docker", "commit", container, image], output, "snapshot",
                       timeout=snapshot_timeout_seconds).strip()
    limits = info["HostConfig"]
    create = ["docker", "run", "-d", "--name", name,
              "--label", "phiagent.purpose=retained-native-verifier"]
    if limits.get("NanoCpus"):
        create.extend(("--cpus", str(limits["NanoCpus"] / 1_000_000_000)))
    if limits.get("Memory"):
        create.extend(("--memory", str(limits["Memory"])))
    create.extend(("--env", "CUDA_VISIBLE_DEVICES=", image))
    command(create, output, "create", timeout=120)
    write_json_atomic(output / "STATE.json", {
        "status": "PREPARING_VERIFIER", "pid": os.getpid(), "updated_at": now(),
        "snapshot_image_id": snapshot, "container": name, "task_id": old["task_id"],
    })

    class RetainedEnvironment:
        os = TaskOS.LINUX
        capabilities = SimpleNamespace(mounted=False)
        default_user = "root"

        async def call(self, argv: list[str], timeout: float | None = None):
            process = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except BaseException:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=15)
                raise
            return ExecResult(
                stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"),
                return_code=process.returncode,
            )

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            argv = ["docker", "exec", "--user", str(user or self.default_user)]
            if cwd:
                argv.extend(("--workdir", cwd))
            for key, value in (env or {}).items():
                argv.extend(("--env", f"{key}={value}"))
            return await self.call([*argv, name, "bash", "-lc", command], timeout_sec)

        async def upload_file(self, source_path, target_path):
            result = await self.call(["docker", "cp", str(source_path), f"{name}:{target_path}"], 900)
            if result.return_code:
                raise RuntimeError(result.stderr)

        async def upload_dir(self, source_dir, target_dir):
            made = await self.exec("mkdir -p " + shlex.quote(target_dir))
            if made.return_code:
                raise RuntimeError(made.stderr)
            await self.upload_file(str(source_dir) + "/.", target_dir)

        async def download_dir(self, source_dir, target_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            result = await self.call(
                ["docker", "cp", f"{name}:{source_dir}/.", str(target_dir)], 120,
            )
            if result.return_code:
                raise RuntimeError(result.stderr)

    environment = RetainedEnvironment()
    try:
        if install_public_ca_certificates:
            installed = await environment.exec(
                'set -e\nmkdir -p "${TMPDIR:-/tmp}"\n'
                'if test -n "$(dpkg --audit)"; then DEBIAN_FRONTEND=noninteractive dpkg --configure -a; fi\n'
                'apt-get update\nDEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates',
                env=prepared["verifier_environment"], timeout_sec=1800,
            )
            write_json_atomic(output / "PUBLIC_CA_CERTIFICATES.json", {
                "return_code": installed.return_code, "stdout": installed.stdout,
                "stderr": installed.stderr, "additional_actor_calls": 0,
                "task_and_test_sources_changed": False, "native_verifier_timer_started": False,
                "pending_native_package_configuration_repaired_before_install": True,
            })
            if installed.return_code:
                raise RuntimeError("Public certificate dependency setup failed before native scoring")
        if sam_verifier_cache is not None:
            await sam.prepare_verifier_cache(
                environment, sam_verifier_cache, output,
                transport_environment=prepared["verifier_environment"],
            )
        if cache is not None:
            # The Python cache does not contain the image's system packages.
            # Prepare the unchanged public APT prefix before starting native
            # scoring, just as the uncached recovery path does.
            system_setup = native_system_package_setup((task_path / "tests/test.sh").read_text())
            if system_setup:
                write_json_atomic(output / "STATE.json", {
                    "status": "PREPARING_RETAINED_VERIFIER_DEPENDENCIES",
                    "pid": os.getpid(), "updated_at": now(), "task_id": old["task_id"],
                    "additional_actor_calls": 0, "native_verifier_timer_started": False,
                })
                system_result = await environment.exec(
                    "set -e\n" + "\n".join(system_setup),
                    env=dependency_environment(prepared["verifier_environment"]),
                    timeout_sec=7200,
                )
                write_json_atomic(output / "CACHED_NATIVE_SYSTEM_SETUP.json", {
                    "stdout": system_result.stdout, "stderr": system_result.stderr,
                    "return_code": system_result.return_code,
                    "original_test_script_sha256": digest(task_path / "tests/test.sh"),
                    "native_system_package_commands": system_setup,
                    "additional_actor_calls": 0, "native_verifier_timer_started": False,
                })
                if system_result.return_code:
                    raise RuntimeError("Cached verifier system dependencies failed before native scoring")
            await install_cache(environment, cache, task_path, output)
        prepared_env = dependency_environment(prepared["verifier_environment"])
        if sam_verifier_cache is not None:
            # The provisioned wheels have already passed the cache hash audit.
            # UV_OFFLINE needs their local path at verifier execution as well.
            prepared_env["UV_FIND_LINKS"] = "/opt/sam-recovery/wheels"
        if cache is not None:
            prepared_env["UV_OFFLINE"] = "1"
        if (cache is None and sam_verifier_cache is None
                and (capability or coding_confirmation or figure3_codex_train or native_dependency_setup)):
            setup = native_setup
            if setup is None:
                setup, _ = dependency_command((task_path / "tests/test.sh").read_text())
            if "/tests/" in setup or "/app/" in setup or re.search(r"(?m)^\s*cp\s", setup):
                raise ValueError("Dependency-only recovery must not execute task-workspace commands")
            system_setup = native_system_package_setup((task_path / "tests/test.sh").read_text())
            setup = "\n".join(system_setup) + "\n" + setup
            write_json_atomic(output / "STATE.json", {
                "status": "PREPARING_RETAINED_VERIFIER_DEPENDENCIES",
                "pid": os.getpid(), "updated_at": now(), "task_id": old["task_id"],
                "additional_actor_calls": 0, "native_verifier_timer_started": False,
            })
            setup_result = await environment.exec(
                "set -e\n" + setup, env=prepared_env, timeout_sec=7200,
            )
            write_json_atomic(output / "DEPENDENCY_SETUP.json", {
                "stdout": setup_result.stdout, "stderr": setup_result.stderr,
                "return_code": setup_result.return_code,
                "original_test_script_sha256": digest(task_path / "tests/test.sh"),
                "workspace_commands_excluded": True, "additional_actor_calls": 0,
                "native_system_package_commands": system_setup,
            })
            if setup_result.return_code:
                raise RuntimeError("Retained verifier dependency setup failed; native timer not started")
            if dependency_mode == "native_pinned_pip_setup":
                prepared_env["PIP_NO_INDEX"] = "1"
                prepared_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            if dependency_mode == "native_uv_setup":
                prepared_env["UV_OFFLINE"] = "1"
        paths = TrialPaths(output / "native")
        paths.verifier_dir.mkdir(parents=True)
        cleared = await environment.exec(
            "mkdir -p /logs/verifier; "
            "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/ctrf.json"
        )
        if cleared.return_code:
            raise RuntimeError("Cannot clear old verifier receipts in the new snapshot clone")
        verifier = Verifier(task=task, trial_paths=paths, environment=environment,
                            override_env=prepared_env)
        started = now()
        write_json_atomic(output / "STATE.json", {
            "status": "NATIVE_VERIFIER_RUNNING", "pid": os.getpid(), "updated_at": started,
            "task_id": old["task_id"], "additional_actor_calls": 0,
        })
        verdict = await asyncio.wait_for(verifier.verify(), timeout=limit)
        rewards = verdict.model_dump(mode="json")["rewards"]
        if set(rewards) != {"reward"} or type(rewards["reward"]) not in (int, float) or (
            rewards["reward"] not in (0, 1)
        ):
            raise ValueError("The unchanged official verifier did not produce a binary reward")
        verify_executed_native_tests(
            (task_path / "tests/test.sh").read_text(), paths.verifier_dir / "ctrf.json",
            rewards["reward"], required=unexecuted_reward)
        actor_exception = original.get("exception_info")
        if actor_exception and actor_exception.get("exception_type") != "AgentTimeoutError":
            actor_exception = None
        result = {
            **original, "verifier_result": verdict.model_dump(mode="json"), "exception_info": actor_exception,
            "verifier": {"started_at": started, "finished_at": now()},
            "recovery": config,
        }
        write_json_atomic(output / "native_result.json", result)
        write_json_atomic(output / "RECOVERY.json", {
            **config, "status": "NATIVE_VERIFIER_RECOVERED", "score": rewards["reward"],
            "native_result": str(output / "native_result.json"),
            "native_result_sha256": digest(output / "native_result.json"),
            "original_actor_usage": old["usage"], "additional_actor_calls": 0,
            "official_verifier_sha256": digest(Path(sys.modules[Verifier.__module__].__file__)),
        })
        write_json_atomic(output / "STATE.json", {
            "status": "COMPLETE", "pid": os.getpid(), "updated_at": now(),
            "task_id": old["task_id"], "score": rewards["reward"], "additional_actor_calls": 0,
        })
    except Exception as error:
        write_json_atomic(output / "STATE.json", {
            "status": "BLOCKED", "pid": os.getpid(), "updated_at": now(),
            "task_id": old["task_id"], "error_type": type(error).__name__, "error": str(error),
        })
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        command(["docker", "stop", "--time", "5", name], output, "stop", timeout=60)


def campaign(plan_path: Path, output: Path) -> None:
    plan = load(plan_path)
    jobs = plan["jobs"]
    if (
        not jobs or len({job["task_id"] for job in jobs}) != len(jobs)
        or plan["maximum_workers"] not in (1, 2)
        or plan.get("additional_actor_calls") != 0
        or any(job.get("native_dependency_setup") and (
            job.get("cache") is not None or job.get("callback") is None
        ) for job in jobs)
        or any(job.get("dependency_environment") is not None
               and not job.get("native_dependency_setup") for job in jobs)
    ):
        raise ValueError("A recovery plan needs unique tasks, valid dependency modes and one or two CPU workers")
    reserve(output)
    write_json_atomic(output / "PLAN.json", plan)
    environment = load(Path(plan["verifier_environment"]))
    state = {
        "status": "RUNNING", "pid": os.getpid(), "started_at": now(),
        "completed": {}, "failed": {}, "active": [], "pending": [j["task_id"] for j in jobs],
        "additional_actor_calls": 0, "cumulative_token_limit": None,
    }

    def work(job):
        task = Path(job["task"])
        if task.name != job["task_id"]:
            raise ValueError("Task identity differs from its declared source")
        native_setup = job.get("native_dependency_setup", False)
        if job.get("cache") is not None:
            cache = Path(job["cache"])
            verified_cache(cache, task)
        elif native_setup:
            cache = None
        else:
            dependencies = output / "dependencies" / job["task_id"]
            prepare(task, dependencies, environment)
            cache = dependencies / "PREPARED.json"
        if job.get("callback") is not None:
            recovered = output / "verifiers" / job["task_id"]
            asyncio.run(recover(
                Path(job["callback"]), cache, recovered,
                native_dependency_setup=native_setup,
                native_dependency_environment=job.get("dependency_environment"),
            ))
            return {"recovery": str(recovered / "RECOVERY.json"),
                    "cache": str(cache) if cache is not None else None}
        return {"cache": str(cache), "purpose": "dependencies_only"}

    with ThreadPoolExecutor(max_workers=plan["maximum_workers"]) as pool:
        pending = iter(jobs)
        active = {}

        def refill():
            while len(active) < plan["maximum_workers"]:
                job = next(pending, None)
                if job is None:
                    break
                state["pending"].remove(job["task_id"])
                active[pool.submit(work, job)] = job["task_id"]

        refill()
        while active:
            state.update(active=sorted(active.values()), updated_at=now())
            write_json_atomic(output / "STATE.json", state)
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = active.pop(future)
                try:
                    state["completed"][task_id] = future.result()
                except Exception as error:
                    state["failed"][task_id] = {
                        "error_type": type(error).__name__, "error": str(error),
                    }
                    print(f"Recovery blocked for {task_id}: {error}", file=sys.stderr, flush=True)
            refill()
    state.update(
        status="BLOCKED" if state["failed"] else "COMPLETE",
        active=[], updated_at=now(),
    )
    write_json_atomic(output / "STATE.json", state)
    if state["failed"]:
        raise RuntimeError("Native recovery has explicit unresolved tasks; see STATE.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=(
            "prepare", "recover", "capability", "capability-unexecuted", "validation",
            "coding-confirmation", "figure3-codex-train", "figure3-codex-bank-train", "figure3-codex-validation", "figure3-deepseek-validation", "campaign", "trial",
        ),
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--native-job", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--callback", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--frozen-test-binding", type=Path)
    parser.add_argument("--native-dependency-setup", action="store_true")
    parser.add_argument("--verifier-environment", type=Path)
    parser.add_argument("--seed-metadata", type=Path)
    parser.add_argument("--sam-verifier-cache", type=Path)
    parser.add_argument("--snapshot-timeout-seconds", type=int, default=300)
    parser.add_argument("--match-native-cpu-threads", action="store_true")
    parser.add_argument("--install-public-ca-certificates", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sam_verifier_cache and (
        args.mode not in ("capability", "capability-unexecuted", "figure3-codex-train", "figure3-codex-bank-train") or args.cache is not None
    ):
        parser.error("--sam-verifier-cache requires an unscored capability actor and no generic cache")
    if args.native_dependency_setup and (args.mode != "recover" or args.cache is not None):
        parser.error("--native-dependency-setup requires recover without --cache")
    if args.mode == "recover" and args.verifier_environment is not None and not args.native_dependency_setup:
        parser.error("Recovery transport override requires --native-dependency-setup")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.umask(0o077)
    if args.mode == "trial":
        if args.native_job is None or args.cache is None:
            parser.error("trial requires --native-job and --cache")
        asyncio.run(cached_trial(args.native_job, args.cache, args.output))
    elif args.mode == "campaign":
        if args.plan is None:
            parser.error("campaign requires --plan")
        campaign(args.plan, args.output)
    elif args.mode == "prepare":
        if args.task is None or args.verifier_environment is None:
            parser.error("prepare requires --task and --verifier-environment")
        prepare(args.task, args.output, load(args.verifier_environment), args.seed_metadata,
                install_public_ca_certificates=args.install_public_ca_certificates)
    else:
        if args.callback is None or (
            args.mode not in ("capability", "capability-unexecuted", "coding-confirmation", "figure3-codex-train", "figure3-codex-bank-train", "figure3-codex-validation", "figure3-deepseek-validation")
            and args.cache is None
            and not args.native_dependency_setup
        ):
            parser.error("recover requires --callback and --cache")
        asyncio.run(recover(
            args.callback,
            args.cache,
            args.output,
            capability=args.mode in ("capability", "capability-unexecuted"),
            unexecuted_reward=args.mode in ("capability-unexecuted", "figure3-codex-train", "figure3-codex-bank-train", "figure3-codex-validation", "figure3-deepseek-validation"),
            validation=args.mode == "validation",
            coding_confirmation=args.mode == "coding-confirmation",
            native_dependency_setup=args.native_dependency_setup,
            native_dependency_environment=(
                load(args.verifier_environment)
                if (args.native_dependency_setup or args.mode in ("capability", "capability-unexecuted", "figure3-codex-train", "figure3-codex-bank-train", "figure3-codex-validation", "figure3-deepseek-validation"))
                and args.verifier_environment is not None else None
            ),
            frozen_test_binding=args.frozen_test_binding,
            sam_verifier_cache=args.sam_verifier_cache,
            figure3_codex_train=args.mode == "figure3-codex-train",
            figure3_codex_validation=args.mode == "figure3-codex-validation",
            figure3_codex_bank_train=args.mode == "figure3-codex-bank-train",
            figure3_deepseek_validation=args.mode == "figure3-deepseek-validation",
            snapshot_timeout_seconds=args.snapshot_timeout_seconds,
            match_native_cpu_threads=args.match_native_cpu_threads,
            install_public_ca_certificates=args.install_public_ca_certificates,
        ))


if __name__ == "__main__":
    main()
