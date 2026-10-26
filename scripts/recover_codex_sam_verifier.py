#!/usr/bin/env python3
"""Recover only the recorded SAM verifier, with no model calls.

Run ``launch`` on A800-4 with the pinned Harbor interpreter. A detached, bounded
dependency job automatically hands off to one native Trial after preparing an
offline uv cache. Only the recorded convert_masks.py is restored; native Harbor
alone introduces test inputs and executes the unchanged test.sh. The two curl
cache hits serve hash-verified official bytes, not replacement implementations.

An isolated successor can inherit verified artifacts with --predecessor and
--data. It never edits the predecessor. The Git TLS repair reconstructs the
identical Git objects from the official pinned archive and signed GitHub metadata.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback
from urllib.parse import urlsplit
import zlib
from datetime import datetime, timezone

HOST = "harnessmm-api-worker"
CAMPAIGN = Path("/opt/phiagent/work/codex-chisel/20260929-v1")
ORIGINAL = CAMPAIGN / "terminal/train/baseline/r0/sam-cell-seg"
RETENTION = CAMPAIGN / "terminal/sam-program-retention-v1"
RECEIPTS = CAMPAIGN / "terminal/sam-verifier-recovery-v1"
RECEIPTS_ROOT = RECEIPTS
DATA = Path("/path/to/data/phiagent/work/codex-chisel-sam-verifier/20260929-v1")
DATA_ROOT = DATA.parent
PREDECESSOR: Path | None = None
EXTRA_WHEELS: tuple[Path, ...] = ()
DEFAULT_DEPENDENCY_CACHE_URL = "http://172.17.0.1:28162"
DEPENDENCY_CACHE_URL = DEFAULT_DEPENDENCY_CACHE_URL
SOURCE_EVIDENCE = RECEIPTS_ROOT / "git-transport-diagnosis-v1"
SETUP_DEADLINE = 9600
PROGRAM_SHA = "76a13bfc50e09d8986a13c8d47426506e30baebaa43b067841d4acfce09fd005"
COMMIT = "34bbbfdface3c18e5221aa7de6032d7220c6c6a1"
REPOSITORY = "https://github.com/ChaoningZhang/MobileSAM.git"
API = "https://api.github.com/repos/ChaoningZhang/MobileSAM"
WEIGHTS_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
UV_URL = "https://astral.sh/uv/0.9.5/install.sh"
IMAGE = "alexgshaw/sam-cell-seg:20251031"
PACKAGES = (
    "torch==2.5.1", "torchvision==0.20.1", "pandas==2.3.2",
    "timm==1.0.19", "opencv-python==4.12.0.88", "shapely==2.1.1",
    "pytest==8.4.1", "pytest-json-ctrf==0.3.5", f"git+{REPOSITORY}@{COMMIT}",
)
UVX = ["uvx", "--index", "https://download.pytorch.org/whl/cpu",
       "--index-strategy", "unsafe-best-match", "-p", "3.11"]
for _package in PACKAGES:
    UVX.extend(("-w", _package))
FORBIDDEN = {"correct_output.csv", "test_metadata.csv", "test_img.png", "mobile_sam.pt"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, data: dict) -> None:
    stage = path.with_name(path.name + ".writing")
    stage.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    stage.replace(path)


def canonical_hash(mapping: dict) -> str:
    return hashlib.sha256(json.dumps(
        mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def verified_extra_wheel(path: Path) -> dict:
    metadata = path.parent / "PYPI.json"
    release = json.loads(metadata.read_text())
    if (
        release["info"]["name"] != "opencv-python"
        or release["info"]["version"] != "4.12.0.88"
        or path.is_symlink() or not path.is_file()
    ):
        raise ValueError("Only the exact official OpenCV verifier wheel is admitted")
    matches = [row for row in release["urls"] if row["filename"] == path.name]
    if len(matches) != 1:
        raise ValueError("Wheel is absent or duplicated in official release metadata")
    wheel = matches[0]
    if path.stat().st_size != wheel["size"] or sha256(path) != wheel["digests"]["sha256"]:
        raise ValueError("Cached OpenCV bytes differ from the official PyPI wheel")
    return {
        "source": str(path), "filename": path.name, "sha256": wheel["digests"]["sha256"],
        "size": wheel["size"], "official_url": wheel["url"],
        "metadata": str(metadata), "metadata_sha256": sha256(metadata),
    }


def inherit_dependency_home(predecessor: Path, home: Path) -> dict | None:
    state_record = json.loads((predecessor / "STATE.json").read_text())
    protocol = json.loads((predecessor / "PROTOCOL.json").read_text())
    if state_record["status"] != "BLOCKED" or protocol["native_image"] != IMAGE:
        raise ValueError("Dependency reuse requires a blocked predecessor with the same native image")
    prior_home = Path(protocol["data_directory"]) / "home"
    if not prior_home.is_dir():
        return None
    try:
        os.kill(state_record["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("A live predecessor dependency cache cannot be copied")
    shutil.copytree(
        prior_home, home, symlinks=True,
        ignore=shutil.ignore_patterns(".tmp*", ".lock", "*.lock"),
    )
    return {
        "source": str(prior_home), "predecessor": str(predecessor),
        "excluded": [".tmp*", ".lock", "*.lock"],
        "acceptance": "Untrusted partial cache; full pinned online and offline dependency gates required",
        "additional_model_calls": 0,
    }


def verify_program(program: Path, retention: dict, expected: str = PROGRAM_SHA) -> dict:
    if program.is_symlink() or not program.is_file():
        raise ValueError("Retained solution must be a regular file")
    digest = sha256(program)
    if digest != expected or retention["files"].get("app/convert_masks.py") != digest:
        raise ValueError("Retained solution SHA256 mismatch")
    if retention.get("status") != "ACTOR_PROGRAM_RETAINED_READ_ONLY":
        raise ValueError("Missing read-only actor retention evidence")
    tree = ast.parse(program.read_text())
    allowed = {"argparse", "os", "sys", "cv2", "numpy", "pandas", "torch", "tqdm", "mobile_sam"}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Unretained relative solution import")
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval", "__import__"}:
                raise ValueError("Dynamic code prevents exact solution-closure verification")
    if imports - allowed:
        raise ValueError(f"Unretained solution imports: {sorted(imports - allowed)}")
    return {
        "sha256": digest, "imports": sorted(imports),
        "restored_actor_files": ["convert_masks.py"],
        "excluded_verifier_created_files": sorted(FORBIDDEN),
        "closure": "One recorded Python program; imports supplied by native pinned dependencies",
    }


def reconcile_usage(original: dict) -> dict:
    paths = original["provider_usage"]["receipts"]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Missing or duplicate actor receipts")
    root = Path(paths[0]).parents[3]
    directory = root / "calls" / original["trial_id"]
    actual = sorted(p / "RECEIPT.json" for p in directory.iterdir() if p.is_dir())
    if set(map(str, actual)) != set(paths):
        raise ValueError("Provider ledger changed or includes an unresolved actor call")
    totals = dict.fromkeys(("input_tokens", "output_tokens", "total_tokens"), 0)
    cached = reasoning = 0
    missing = set()
    ids, hashes = [], {}
    for path in actual:
        receipt = json.loads(path.read_text())
        if receipt.get("usage_status") != "MEASURED" or receipt.get("model") != "deepseek-flash":
            raise ValueError("Unresolved or wrong-model actor receipt")
        usage = receipt["usage"]
        for key in totals:
            value = usage.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("Malformed actor usage")
            totals[key] += value
        if usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]:
            raise ValueError("Inconsistent actor usage")
        for key, value in (("cached_tokens", usage.get("input_tokens_details") or {}),
                           ("reasoning_tokens", usage.get("output_tokens_details") or {})):
            if key not in value:
                missing.add(key)
        cached += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        if receipt.get("response_id"):
            ids.append(receipt["response_id"])
        hashes[str(path)] = sha256(path)
    if totals["total_tokens"] <= 0:
        raise ValueError("Actor usage cannot be zeroed in a verifier continuation")
    return {
        "trial_id": original["trial_id"], "usage_status": "MEASURED", "calls": len(actual),
        "known_usage": totals, "cached_input_tokens": None if "cached_tokens" in missing else cached,
        "reasoning_output_tokens": None if "reasoning_tokens" in missing else reasoning,
        "unresolved_call_ids": [], "provider_models": ["deepseek-flash"],
        "model_response_ids": ids, "receipts": list(map(str, actual)), "receipt_sha256": hashes,
    }


def verify_weights(path: Path, pinned: dict, master: dict) -> dict:
    if pinned.get("sha") != master.get("sha") or pinned.get("size") != master.get("size"):
        raise ValueError("Official master and pinned-release weights differ")
    size = path.stat().st_size
    if size != pinned["size"] or size < 1024 * 1024:
        raise ValueError("Incomplete official weights")
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            blob.update(chunk)
    if blob.hexdigest() != pinned["sha"]:
        raise ValueError("Weights do not match the official Git blob")
    return {"sha256": sha256(path), "git_blob_sha1": blob.hexdigest(), "size": size,
            "official_url": WEIGHTS_URL, "package_commit": COMMIT,
            "master_matches_pinned_release": True}


def binary_reward(result: dict) -> float:
    rewards = (result.get("verifier_result") or {}).get("rewards")
    if not isinstance(rewards, dict) or set(rewards) != {"reward"}:
        raise ValueError("Native verifier supplied no unique reward")
    reward = rewards["reward"]
    if type(reward) not in (int, float) or reward not in (0, 1):
        raise ValueError("Native verifier reward must be binary")
    if result.get("exception_info"):
        raise ValueError("Infrastructure failure is not a scored solution")
    return float(reward)


def executed_tests(ctrf: dict, expected: set[str], reward: float) -> dict:
    results = ctrf.get("results") or {}
    tests = results.get("tests") or []
    if not tests or not expected:
        raise ValueError("Native tests did not execute; dependency failure is not a zero score")
    names = [row["name"].rsplit("::", 1)[-1] for row in tests]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Native CTRF does not contain the complete unchanged test population")
    statuses = [row.get("status") for row in tests]
    if any(value not in {"passed", "failed"} for value in statuses):
        raise ValueError("Native CTRF contains unexecuted tests")
    if results.get("summary", {}).get("tests") != len(tests):
        raise ValueError("Native CTRF test count is inconsistent")
    if (reward == 1) != all(value == "passed" for value in statuses):
        raise ValueError("Native reward and executed CTRF outcomes disagree")
    return {"executed": len(tests), "passed": statuses.count("passed"),
            "failed": statuses.count("failed"), "names": sorted(names)}


def git_hash(kind: str, payload: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(payload)}\0".encode() + payload).hexdigest()


def signed_commit_bytes(metadata: dict) -> bytes:
    verification = metadata["verification"]
    if not verification.get("verified") or not verification.get("payload"):
        raise ValueError("Official signed commit evidence is missing")
    headers, message = verification["payload"].split("\n\n", 1)
    signature = verification["signature"]
    for value in (signature, signature.rstrip("\n")):
        raw = (headers + "\ngpgsig " + value.replace("\n", "\n ") + "\n\n" + message).encode()
        if git_hash("commit", raw) == metadata["sha"]:
            return raw
    raise ValueError("Signed commit reconstruction does not match its official Git identity")


def tree_bytes(entries: list[dict]) -> bytes:
    def key(row):
        return (row["name"] + ("/" if row["type"] == "tree" else "")).encode()

    output = bytearray()
    for row in sorted(entries, key=key):
        name = row["name"]
        if not name or name in {".", ".."} or "/" in name or "\0" in name:
            raise ValueError("Unsafe Git tree entry")
        mode = row["mode"].lstrip("0")
        if mode not in {"40000", "100644", "100755"}:
            raise ValueError("Unexpected Git object mode")
        output.extend(f"{mode} {name}\0".encode() + bytes.fromhex(row["sha"]))
    return bytes(output)


def put_git_object(repository: Path, kind: str, payload: bytes, expected: str) -> None:
    if git_hash(kind, payload) != expected:
        raise ValueError(f"Official {kind} object hash mismatch: {expected}")
    target = repository / "objects" / expected[:2] / expected[2:]
    target.parent.mkdir(parents=True, exist_ok=True)
    content = f"{kind} {len(payload)}\0".encode() + payload
    if target.exists():
        if zlib.decompress(target.read_bytes()) != content:
            raise ValueError("Existing Git object differs from its official identity")
    else:
        target.write_bytes(zlib.compress(content))


def official_tree_root(commit: dict, tree: dict) -> str:
    expected = commit["tree"]["sha"]
    if tree.get("truncated") or tree["sha"] not in {expected, commit["sha"]}:
        raise ValueError("Incomplete or incorrect pinned official Git tree")
    return expected


def archive_git_mirror(archive: Path, commit: dict, tree: dict, repository: Path) -> dict:
    if commit["sha"] != COMMIT:
        raise ValueError("Incorrect pinned official Git commit")
    root_sha = official_tree_root(commit, tree)
    signed = signed_commit_bytes(commit)
    if signed.splitlines()[0] != b"tree " + root_sha.encode():
        raise ValueError("Official metadata differs from the signed commit's root tree")
    entries = {row["path"]: row for row in tree["tree"]}
    blobs = {name: row for name, row in entries.items() if row["type"] == "blob"}
    seen = set()
    prefix = f"MobileSAM-{COMMIT}/"
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.isdir():
                continue
            if not member.name.startswith(prefix) or not member.isfile():
                raise ValueError("Unexpected official archive member")
            name = member.name[len(prefix):]
            if name not in blobs or name in seen:
                raise ValueError("Archive and official Git tree have different file populations")
            row = blobs[name]
            if member.size != row["size"]:
                raise ValueError("Official archive member has the wrong size")
            with source.extractfile(member) as stream:
                payload = stream.read()
            put_git_object(repository, "blob", payload, row["sha"])
            seen.add(name)
    if seen != set(blobs):
        raise ValueError("Official archive is missing Git blobs")
    directories = {"": root_sha, **{
        name: row["sha"] for name, row in entries.items() if row["type"] == "tree"
    }}
    for directory, expected in directories.items():
        children = [
            {**row, "name": Path(name).name}
            for name, row in entries.items()
            if Path(name).parent == Path(directory or ".")
        ]
        put_git_object(repository, "tree", tree_bytes(children), expected)
    put_git_object(repository, "commit", signed, COMMIT)
    (repository / "shallow").write_text(COMMIT + "\n")
    return {
        "commit": COMMIT, "tree": root_sha, "blob_count": len(blobs),
        "api_requested_reference": tree["sha"],
        "source_archive_sha256": sha256(archive),
        "transport": "Official codeload archive plus hash-identical signed Git objects",
        "source_modified": False, "weights_reused": True,
    }


def native_config(original: dict, receipts: Path, env: dict) -> dict:
    config = json.loads(json.dumps(original))
    config["trial_name"] = "tb-train-baseline-r0-sam-cell-seg-verifier-only-" + receipts.name
    config["trials_dir"] = str(receipts / "native")
    config["agent"] = {
        "import_path": "recover_codex_sam_verifier:RetainedSolutionAgent",
        "model_name": "deepseek-flash",
        "kwargs": {"receipts": str(receipts)},
    }
    config["verifier"] = {**config.get("verifier", {}), "env": env}
    return config


def command(args: list[str], log: str, timeout: int, *, env: dict | None = None) -> None:
    with (RECEIPTS / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": now(), "argv": args, "timeout_sec": timeout}) + "\n")
    with (RECEIPTS / log).open("ab") as stream:
        process = subprocess.Popen(
            args, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        try:
            started = time.monotonic()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                artifacts = {}
                for name in ("mobile_sam.pt", "MobileSAM.tar.gz", "home.tar"):
                    path = DATA / "artifacts" / name
                    if path.exists():
                        artifacts[name] = path.stat().st_size
                save(RECEIPTS / "HEARTBEAT.json", {
                    "at": now(), "pid": os.getpid(), "child_pid": process.pid,
                    "elapsed_sec": time.monotonic() - started, "command_timeout_sec": timeout,
                    "log": str(RECEIPTS / log), "log_bytes": (RECEIPTS / log).stat().st_size,
                    "artifact_bytes": artifacts, "additional_model_calls": 0,
                })
                try:
                    returncode = process.wait(timeout=min(15, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, args)


def state(status: str, **details) -> None:
    save(RECEIPTS / "STATE.json", {
        "status": status, "at": now(), "pid": os.getpid(), "CUDA_VISIBLE_DEVICES": "",
        "additional_model_calls": 0, "continuation_type": "verifier-only", **details,
    })
    print(json.dumps({"at": now(), "status": status, **details}), flush=True)


def fetch(
    url: str, target: Path, timeout: int = 180, *, raw: bool = False, resume: bool = False,
) -> None:
    if target.exists() and not resume:
        raise ValueError(f"Refusing to overwrite artifact: {target}")
    args = ["curl", "--noproxy", "*", "--fail", "--location", "--show-error",
            "--connect-timeout", "15", "--max-time", str(timeout),
            "--speed-time", "180", "--speed-limit", "1024",
            "--dump-header", str(target.with_suffix(target.suffix + ".headers"))]
    if resume:
        args += ["--continue-at", "-"]
    if raw:
        args += ["-H", "Accept: application/vnd.github.raw+json"]
    command(args + [url, "--output", str(target)], "downloads.log", timeout + 10)


def freeze() -> dict:
    original = json.loads((ORIGINAL / "RUN.json").read_text())
    config = json.loads((ORIGINAL / "CONFIG.json").read_text())
    retained = json.loads((RETENTION / "RETENTION.json").read_text())
    closure = verify_program(RETENTION / "app/convert_masks.py", retained)
    usage = reconcile_usage(original)
    task = Path(config["task"]["path"])
    files = {p.relative_to(task).as_posix(): sha256(p)
             for p in sorted(task.rglob("*")) if p.is_file()}
    if canonical_hash(files) != original["task_source_sha256"]:
        raise ValueError("Official task differs from the original frozen training population")
    if original["phase"] != "train" or original["task_id"] != "sam-cell-seg":
        raise ValueError("Recovery is authorized for this training task only")
    if config["timeout_multiplier"] != 1.0:
        raise ValueError("Original native timeout multiplier changed")
    native = json.loads((ORIGINAL / "native_result.json").read_text())
    if not native.get("agent_execution", {}).get("finished_at"):
        raise ValueError("Original actor did not finish")
    if (native.get("exception_info") or {}).get("exception_type") != "VerifierTimeoutError":
        raise ValueError("Predecessor is not the expected verifier-only timeout")
    for field, key in (("n_input_tokens", "input_tokens"), ("n_output_tokens", "output_tokens")):
        if native["agent_result"][field] != usage["known_usage"][key]:
            raise ValueError("Provider receipts disagree with the recorded native actor usage")
    (RECEIPTS / "original-evidence").mkdir()
    for name in ("RUN.json", "CONFIG.json", "native_result.json", "failure.txt", "provenance.json"):
        if (ORIGINAL / name).is_file():
            shutil.copyfile(ORIGINAL / name, RECEIPTS / "original-evidence" / name)
    shutil.copyfile(RETENTION / "RETENTION.json", RECEIPTS / "original-evidence/RETENTION.json")
    shutil.copyfile(RETENTION / "app/convert_masks.py", RECEIPTS / "convert_masks.py")
    trajectory = ORIGINAL / "native" / original["trial_id"] / "agent/trajectory.json"
    record = {
        "created_at": now(), "hostname": socket.gethostname(), "seed": original["seed"],
        "split_seed": 20260928, "phase": "train", "protocol_label": original["protocol_label"],
        "original_run": str(ORIGINAL / "RUN.json"),
        "original_run_sha256": sha256(ORIGINAL / "RUN.json"),
        "original_native_sha256": sha256(ORIGINAL / "native_result.json"),
        "trajectory": str(trajectory), "trajectory_sha256": sha256(trajectory),
        "retention_sha256": sha256(RETENTION / "RETENTION.json"),
        "actor_program": closure, "provider_usage": usage, "task_files": files,
        "task_source_sha256": original["task_source_sha256"],
        "partition_sha256": original["partition_sha256"], "additional_model_calls": 0,
        "continuation_type": "verifier-only; not a new model or harness result",
        "native_image": IMAGE, "native_resources": {"cpus": 1, "memory": "4G", "storage": "10G"},
        "native_verifier_timeout_sec": 7200, "native_agent_timeout_sec": 7200,
        "native_agent_setup_timeout_sec": 360,
        "setup_deadline_sec": SETUP_DEADLINE, "token_usage_mode": "unlimited",
        "maximum_tokens": None, "data_directory": str(DATA),
        "source_sha256": sha256(Path(__file__)),
        "extra_wheels": [verified_extra_wheel(path) for path in EXTRA_WHEELS],
        "dependency_cache_url": DEPENDENCY_CACHE_URL,
        "acquisition_limits": {
            "archive_sec": 3000, "cpu_dependency_closure_sec": 5400,
            "curl_stall_sec": 180, "curl_minimum_bytes_per_sec": 1024,
            "basis": "Prior official transfer measured 63,359 B/s; pinned Git tree is "
                     "88,295,460 bytes. Archive bound exceeds twice that acquisition time. "
                     "CPU dependency closure allows larger torch and transitive wheels.",
        },
    }
    if PREDECESSOR:
        prior_state = json.loads((PREDECESSOR / "STATE.json").read_text())
        if prior_state["status"] != "BLOCKED" or (PREDECESSOR / "RUN.json").exists():
            raise ValueError("Only an unscored, blocked predecessor can be continued")
        try:
            os.kill(prior_state["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Predecessor worker is still live; refusing duplicate work")
        record["predecessor"] = {
            "receipts": str(PREDECESSOR), "state_sha256": sha256(PREDECESSOR / "STATE.json"),
            "protocol_sha256": sha256(PREDECESSOR / "PROTOCOL.json"),
            "failure": prior_state,
            "repair": (
                "Retain completed dependency downloads and prefetch the missing hash-identical "
                "OpenCV wheel; original actor and native task limits are unchanged"
                if EXTRA_WHEELS else
                "Resolve GitHub's requested-commit alias using the signed commit's root tree; "
                "inherit the completed archive without another download"
                if prior_state.get("error") == "Incomplete or incorrect pinned official Git tree"
                else "Replace failed GitHub Git TLS transport with official codeload archive; "
                     "reconstruct every original Git object and verify the exact pinned commit"
            ),
        }
    save(RECEIPTS / "PROTOCOL.json", record)
    command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
             "--format=csv"], "resources.log", 30)
    command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv"], "resources.log", 30)
    command(["sh", "-c", "hostname; free -h; uptime; df -h /path/to/data /opt; nproc"],
            "resources.log", 30)
    save(RECEIPTS / "runtime-packages.json", {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    })
    command(["docker", "image", "inspect", IMAGE], "original-image.json", 30)
    return record


def prepare() -> None:
    state("PREFETCHING_OFFICIAL_WEIGHTS", setup_deadline_sec=SETUP_DEADLINE)
    inherited_archive_complete = False
    if PREDECESSOR:
        prior = json.loads((PREDECESSOR / "PROTOCOL.json").read_text())
        old_data = Path(prior.get(
            "data_directory", str(DATA_ROOT / "20260929-v1"),
        ))
        for name in ("weight-pinned.json", "weight-master.json"):
            shutil.copy2(old_data / "metadata" / name, DATA / "metadata" / name)
        old_weights = old_data / "artifacts/mobile_sam.pt"
        expected = json.loads((PREDECESSOR / "WEIGHTS.json").read_text())
        if sha256(old_weights) != expected["sha256"]:
            raise ValueError("Previously verified official weights changed")
        shutil.copy2(old_weights, DATA / "artifacts/mobile_sam.pt")
        old_archive = old_data / "artifacts/MobileSAM.tar.gz"
        if old_archive.is_file():
            prior_state = json.loads((PREDECESSOR / "STATE.json").read_text())
            inherited_archive_complete = (
                (PREDECESSOR / "SOURCE_ARCHIVE.json").exists()
                or prior_state.get("error") == "Incomplete or incorrect pinned official Git tree"
            )
            shutil.copy2(old_archive, DATA / "artifacts/MobileSAM.tar.gz")
            save(RECEIPTS / "INHERITED_ARCHIVE.json", {
                "source": str(old_archive), "sha256": sha256(old_archive),
                "size": old_archive.stat().st_size,
                "completed_http_transfer": inherited_archive_complete,
                "predecessor_failure": str(PREDECESSOR / "failure.txt"),
                "native_source_gate": "Every archive blob and tree must match the signed commit",
            })
    for label, ref in (("pinned", COMMIT), ("master", "master")):
        target = DATA / "metadata" / f"weight-{label}.json"
        if not target.exists():
            fetch(f"{API}/contents/weights/mobile_sam.pt?ref={ref}", target)
    pinned = json.loads((DATA / "metadata/weight-pinned.json").read_text())
    master = json.loads((DATA / "metadata/weight-master.json").read_text())
    target = DATA / "artifacts/mobile_sam.pt"
    if not target.exists():
        fetch(pinned["git_url"], target, 1800, raw=True)
    if target.stat().st_size != pinned["size"]:
        payload = json.loads(target.read_bytes())
        if payload.get("encoding") != "base64":
            raise ValueError("GitHub returned neither raw nor base64 official weights")
        decoded = target.with_suffix(".decoded")
        decoded.write_bytes(base64.b64decode(payload["content"], validate=False))
        target.rename(target.with_suffix(".api-response"))
        decoded.rename(target)
    save(RECEIPTS / "WEIGHTS.json", verify_weights(target, pinned, master))
    state("PREFETCHING_PINNED_GIT_ARCHIVE", commit=COMMIT, weights_verified=True)
    for name in ("official-commit.json", "official-tree.json"):
        shutil.copy2(SOURCE_EVIDENCE / name, DATA / "metadata" / name)
    archive = DATA / "artifacts/MobileSAM.tar.gz"
    if not inherited_archive_complete:
        fetch(f"https://codeload.github.com/ChaoningZhang/MobileSAM/tar.gz/{COMMIT}",
              archive, 3000, resume=True)
    save(RECEIPTS / "SOURCE_ARCHIVE.json", {
        "sha256": sha256(archive), "size": archive.stat().st_size,
        "completed_http_transfer": True,
        "url": f"https://codeload.github.com/ChaoningZhang/MobileSAM/tar.gz/{COMMIT}",
        "inherited_complete_transfer": inherited_archive_complete,
    })
    repository = DATA / "artifacts/MobileSAM.git"
    command(["git", "init", "--bare", str(repository)], "git.log", 30)
    mirror = archive_git_mirror(
        archive, json.loads((DATA / "metadata/official-commit.json").read_text()),
        json.loads((DATA / "metadata/official-tree.json").read_text()), repository,
    )
    command(["git", "-C", str(repository), "update-ref", "refs/heads/pinned", COMMIT],
            "git.log", 30)
    command(["git", "-C", str(repository), "symbolic-ref", "HEAD", "refs/heads/pinned"],
            "git.log", 30)
    command(["git", "-C", str(repository), "fsck", "--full"], "git.log", 60)
    save(RECEIPTS / "GIT_SOURCE.json", mirror)
    fetch(UV_URL, DATA / "artifacts/uv-install.sh", 120)
    wheels = DATA / "wheels"
    wheels.mkdir()
    extras = json.loads((RECEIPTS / "PROTOCOL.json").read_text()).get("extra_wheels", [])
    for row in extras:
        source = Path(row["source"])
        if verified_extra_wheel(source) != row:
            raise ValueError("Frozen public-wheel provenance changed before dependency setup")
        shutil.copy2(source, wheels / row["filename"])
    state("PREFETCHING_EXACT_CPU_DEPENDENCIES", packages=list(PACKAGES))
    home = DATA / "home"
    inherited = inherit_dependency_home(PREDECESSOR, home) if PREDECESSOR else None
    if inherited:
        save(RECEIPTS / "INHERITED_DEPENDENCY_CACHE.json", inherited)
    else:
        home.mkdir()
    env = dependency_env(DEPENDENCY_CACHE_URL)
    docker = ["docker", "run", "--rm", "--name", "sam-verifier-dependencies-" + RECEIPTS.name,
              "--cpus", "1", "--memory", "4g", "--pids-limit", "256",
              "-v", f"{DATA}:/opt/sam-recovery", "-v", f"{home}:/root",
              "-w", "/opt/sam-recovery/work"]
    for key, value in env.items():
        docker += ["-e", f"{key}={value}"]
    warm = shlex.join(UVX + ["pytest", "--version"])
    offline = shlex.join(UVX + ["--offline", "pytest", "--version"])
    setup = (
        "set -e; apt-get update; apt-get install -y curl git libgl1; "
        "sh /opt/sam-recovery/artifacts/uv-install.sh; "
        "export PATH=/root/.local/bin:$PATH; uv --version; "
        "git config --global protocol.file.allow always; "
        "git config --global url.file:///opt/sam-recovery/artifacts/MobileSAM.git.insteadOf "
        + shlex.quote(REPOSITORY) + "; " + warm + "; " + offline
    )
    command(docker + [IMAGE, "bash", "-c", setup], "dependency-setup.log", 5400)
    if (home / ".gitconfig").exists():
        (home / ".gitconfig").unlink()
    command(["tar", "-C", str(home), "-cf", str(DATA / "artifacts/home.tar"),
             ".cache", ".local"], "archive.log", 300)
    prepared = {
        "status": "READY_FOR_NATIVE_VERIFIER", "at": now(),
        "weights": json.loads((RECEIPTS / "WEIGHTS.json").read_text()),
        "installer_sha256": sha256(DATA / "artifacts/uv-install.sh"),
        "home_archive_sha256": sha256(DATA / "artifacts/home.tar"),
        "home_archive_size": (DATA / "artifacts/home.tar").stat().st_size,
        "git_commit": COMMIT, "packages": list(PACKAGES),
        "data_directory": str(DATA), "git_source": mirror,
        "extra_wheels": {row["filename"]: row["sha256"] for row in extras},
        "dependency_cache_url": DEPENDENCY_CACHE_URL,
    }
    save(RECEIPTS / "PREPARED.json", prepared)


def dependency_env(cache_url: str = DEFAULT_DEPENDENCY_CACHE_URL) -> dict:
    parsed = urlsplit(cache_url)
    if (
        parsed.scheme != "http" or parsed.hostname != "172.17.0.1"
        or parsed.port is None or not 1024 <= parsed.port <= 65535
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Dependency cache must be an explicit private A800 bridge endpoint")
    cache_url = cache_url.rstrip("/")
    no_proxy = (
        "localhost,127.0.0.1,::1,172.17.0.1,pypi.org,files.pythonhosted.org,"
        "download.pytorch.org,download-r2.pytorch.org,astral.sh,releases.astral.sh"
    )
    return {
        "CUDA_VISIBLE_DEVICES": "", "HTTP_PROXY": "http://172.17.0.1:18997",
        "http_proxy": "http://172.17.0.1:18997", "HTTPS_PROXY": "http://172.17.0.1:18994",
        "https_proxy": "http://172.17.0.1:18994",
        "NO_PROXY": no_proxy, "no_proxy": no_proxy,
        "UV_FIND_LINKS": cache_url + "/wheels/",
        "UV_INSTALLER_GITHUB_BASE_URL": cache_url + "/github",
        "UV_PYTHON_INSTALL_MIRROR":
            cache_url + "/github/astral-sh/python-build-standalone/releases/download",
        "TMPDIR": "/opt/sam-recovery/work", "GIT_TERMINAL_PROMPT": "0",
    }


def verified_cache(receipts: Path) -> tuple[Path, dict]:
    prepared = json.loads((receipts / "PREPARED.json").read_text())
    protocol = json.loads((receipts / "PROTOCOL.json").read_text())
    if (
        prepared["status"] != "READY_FOR_NATIVE_VERIFIER"
        or prepared["git_commit"] != COMMIT
        or prepared["packages"] != list(PACKAGES)
        or prepared["data_directory"] != protocol["data_directory"]
        or prepared.get("dependency_cache_url", DEFAULT_DEPENDENCY_CACHE_URL)
        != protocol.get("dependency_cache_url", DEFAULT_DEPENDENCY_CACHE_URL)
    ):
        raise ValueError("SAM dependency cache identity differs from the frozen verifier")
    data = Path(prepared["data_directory"])
    extras = {row["filename"]: row["sha256"] for row in protocol.get("extra_wheels", [])}
    if prepared.get("extra_wheels", {}) != extras:
        raise ValueError("Prepared extra wheels differ from the frozen dependency inputs")
    for name, digest in extras.items():
        if Path(name).name != name or sha256(data / "wheels" / name) != digest:
            raise ValueError("Prepared public wheel changed")
    expected = {
        "home.tar": prepared["home_archive_sha256"],
        "mobile_sam.pt": prepared["weights"]["sha256"],
        "uv-install.sh": prepared["installer_sha256"],
    }
    for name, digest in expected.items():
        path = data / "artifacts" / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Prefetched SAM dependency changed: {name}")
    return data, prepared


async def prepare_verifier_cache(
    environment, receipts: Path, output: Path, *, transport_environment: dict | None = None,
) -> dict:
    data, prepared = verified_cache(receipts)
    shell_path = output / "sam-verifier-cache.env"
    shell_path.write_text(cache_shell(prepared))

    async def checked(text: str) -> None:
        result = await environment.exec(
            command=text,
            env=(
                dependency_env(prepared.get("dependency_cache_url", DEFAULT_DEPENDENCY_CACHE_URL))
                if transport_environment is None else transport_environment
            ),
        )
        with (output / "sam-verifier-cache.log").open("a") as stream:
            stream.write((result.stdout or "") + (result.stderr or ""))
        if result.return_code != 0:
            raise RuntimeError(f"SAM verifier dependency provisioning failed: {text}")

    await checked("mkdir -p /opt/sam-recovery/work /opt/sam-recovery/wheels")
    for name in ("home.tar", "mobile_sam.pt", "uv-install.sh"):
        await environment.upload_file(data / "artifacts" / name, "/opt/sam-recovery/" + name)
    await environment.upload_file(shell_path, "/opt/sam-recovery/cache.env")
    for name in prepared.get("extra_wheels", {}):
        await environment.upload_file(data / "wheels" / name, "/opt/sam-recovery/wheels/" + name)
    await checked(
        "tar -xf /opt/sam-recovery/home.tar -C /root && "
        "rm /opt/sam-recovery/home.tar && "
        "apt-get update && apt-get install -y curl git libgl1"
    )
    receipt = {
        "status": "VERIFIED_SAM_CACHE_PROVISIONED", "at": now(),
        "prepared_receipt": str(receipts / "PREPARED.json"),
        "prepared_sha256": sha256(receipts / "PREPARED.json"),
        "home_archive_sha256": prepared["home_archive_sha256"],
        "weights_sha256": prepared["weights"]["sha256"],
        "additional_model_calls": 0,
    }
    save(output / "SAM_VERIFIER_CACHE.json", receipt)
    return receipt


def cache_shell(prepared: dict) -> str:
    cases = []
    for url, filename, digest in (
        (WEIGHTS_URL, "mobile_sam.pt", prepared["weights"]["sha256"]),
        (UV_URL, "uv-install.sh", prepared["installer_sha256"]),
    ):
        cases.append(
            f"    {shlex.quote(url)}) cached=/opt/sam-recovery/{filename}; "
            f"digest={shlex.quote(digest)} ;;"
        )
    return "\n".join([
        "curl() {", "  local arg url='' output='' pending='' cached='' digest=''",
        '  for arg in "$@"; do',
        '    if [[ "$pending" == output ]]; then output="$arg"; pending=""; continue; fi',
        '    case "$arg" in -o|--output) pending=output ;; https://*) url="$arg" ;; esac',
        "  done", '  case "$url" in', *cases,
        '    *) command curl "$@"; return ;;', "  esac",
        '  printf "%s  %s\\n" "$digest" "$cached" | sha256sum -c - >/dev/null || return 97',
        '  if [[ -n "$output" ]]; then cp -- "$cached" "$output"; else cat "$cached"; fi',
        "}", "export -f curl", "",
    ])


def __getattr__(name: str):
    if name != "RetainedSolutionAgent":
        raise AttributeError(name)
    from harbor.agents.base import BaseAgent

    class RetainedSolutionAgent(BaseAgent):
        def __init__(self, *args, receipts: str, **kwargs):
            self.receipts = Path(receipts)
            self.protocol = json.loads((self.receipts / "PROTOCOL.json").read_text())
            self.data = Path(self.protocol["data_directory"])
            super().__init__(*args, **kwargs)

        @staticmethod
        def name():
            return "recorded-solution-verifier-continuation"

        def version(self):
            return "verifier-only-v1"

        async def checked(self, environment, text, **kwargs):
            result = await environment.exec(command=text, **kwargs)
            with (self.receipts / "native-setup.log").open("a") as stream:
                stream.write((result.stdout or "") + (result.stderr or ""))
            if result.return_code != 0:
                raise RuntimeError(f"Native continuation setup failed: {text}")
            return result

        async def setup(self, environment):
            await prepare_verifier_cache(environment, self.receipts, self.receipts)

        async def run(self, instruction, environment, context):
            program = self.receipts / "convert_masks.py"
            if sha256(program) != PROGRAM_SHA:
                raise ValueError("Recorded program changed before native restoration")
            result = await self.checked(
                environment, "find /app -maxdepth 1 -type f -printf '%f\\n' | sort",
            )
            before = set(result.stdout.splitlines())
            if before != {"demo_metadata.csv", "demo_rgb.png"}:
                raise ValueError(f"Native image app inventory changed: {sorted(before)}")
            await environment.upload_file(program, "/app/convert_masks.py")
            result = await self.checked(
                environment, "sha256sum /app/convert_masks.py; "
                "find /app -maxdepth 1 -type f -printf '%f\\n' | sort",
            )
            if result.stdout.split()[0] != PROGRAM_SHA:
                raise ValueError("Native restored program hash differs")
            usage = self.protocol["provider_usage"]
            context.n_input_tokens = usage["known_usage"]["input_tokens"]
            context.n_output_tokens = usage["known_usage"]["output_tokens"]
            context.n_cache_tokens = usage["cached_input_tokens"]
            context.metadata = {
                "verifier_only_continuation": True, "additional_model_calls": 0,
                "inherited_actor_run": self.protocol["original_run"],
            }
            save(self.receipts / "RESTORED.json", {
                "at": now(), "source_sha256": PROGRAM_SHA,
                "native_sha256": result.stdout.split()[0],
                "before_files": sorted(before),
                "restored_actor_files": ["convert_masks.py"],
                "verifier_created_files_restored": [], "additional_model_calls": 0,
            })

    RetainedSolutionAgent.__module__ = __name__
    RetainedSolutionAgent.__qualname__ = "RetainedSolutionAgent"
    return RetainedSolutionAgent


async def score() -> None:
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    if importlib.metadata.version("harbor") != "0.18.0":
        raise ValueError("Only the pinned Harbor 0.18.0 runtime is authorized")
    prepared = json.loads((RECEIPTS / "PREPARED.json").read_text())
    (RECEIPTS / "cache.env").write_text(cache_shell(prepared))
    env = {
        **dependency_env(prepared.get("dependency_cache_url", DEFAULT_DEPENDENCY_CACHE_URL)),
        "BASH_ENV": "/opt/sam-recovery/cache.env", "UV_OFFLINE": "1",
    }
    config = native_config(json.loads((ORIGINAL / "CONFIG.json").read_text()), RECEIPTS, env)
    save(RECEIPTS / "NATIVE_CONFIG.json", config)
    state("NATIVE_VERIFIER_STARTING", timeout_sec=7200)
    trial = await Trial.create(TrialConfig.model_validate(config))
    result = (await trial.run()).model_dump(mode="json")
    save(RECEIPTS / "native_result.json", result)
    score_value = binary_reward(result)
    protocol = json.loads((RECEIPTS / "PROTOCOL.json").read_text())
    task = Path(config["task"]["path"])
    actual_files = {p.relative_to(task).as_posix(): sha256(p)
                    for p in task.rglob("*") if p.is_file()}
    if actual_files != protocol["task_files"]:
        raise ValueError("Official task source changed during verification")
    ctrf_path = RECEIPTS / "native" / config["trial_name"] / "verifier/ctrf.json"
    expected_tests = {
        node.name for node in ast.walk(ast.parse((task / "tests/test_outputs.py").read_text()))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    ctrf_summary = executed_tests(json.loads(ctrf_path.read_text()), expected_tests, score_value)
    original = json.loads((ORIGINAL / "RUN.json").read_text())
    usage = reconcile_usage(original)
    if usage != protocol["provider_usage"]:
        raise ValueError("Actor cost ledger changed during verifier continuation")
    receipt = {
        **original, "status": "SCORED", "score": score_value, "provider_usage": usage,
        "native_result": str(RECEIPTS / "native_result.json"),
        "native_artifact_directory": str(RECEIPTS / "native" / config["trial_name"]),
        "original_actor_trial_id": original["trial_id"],
        "verifier_continuation_trial_id": config["trial_name"],
        "verifier_only_continuation": True, "additional_model_calls": 0,
        "inherited_actor_result": json.loads(
            (ORIGINAL / "native_result.json").read_text(),
        )["agent_result"],
        "recovered_from": str(ORIGINAL / "RUN.json"),
        "recovered_from_sha256": protocol["original_run_sha256"],
        "solution_sha256": PROGRAM_SHA, "continuation_protocol": str(RECEIPTS / "PROTOCOL.json"),
        "failure_evidence": str(RECEIPTS / "original-evidence"),
        "predecessor": protocol.get("predecessor"),
        "native_ctrf": str(ctrf_path), "native_ctrf_sha256": sha256(ctrf_path),
        "native_executed_tests": ctrf_summary,
    }
    receipt.pop("error", None)
    receipt.pop("error_type", None)
    save(RECEIPTS / "RUN.json", receipt)
    state("SCORED_VERIFIER_ONLY_CONTINUATION", score=score_value,
          result=str(RECEIPTS / "RUN.json"), sha256=sha256(RECEIPTS / "RUN.json"))


def main() -> None:
    global RECEIPTS, DATA, PREDECESSOR, EXTRA_WHEELS, DEPENDENCY_CACHE_URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    parser.add_argument("--receipts", type=Path, default=RECEIPTS_ROOT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--extra-wheel", type=Path, action="append", default=[])
    parser.add_argument("--dependency-cache-url", default=DEFAULT_DEPENDENCY_CACHE_URL)
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != HOST:
        raise RuntimeError("Dependency setup and native scoring are restricted to A800-4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.umask(0o077)
    RECEIPTS = args.receipts.resolve()
    DATA = args.data.resolve()
    PREDECESSOR = args.predecessor.resolve() if args.predecessor else None
    EXTRA_WHEELS = tuple(path.resolve() for path in args.extra_wheel)
    DEPENDENCY_CACHE_URL = args.dependency_cache_url.rstrip("/")
    dependency_env(DEPENDENCY_CACHE_URL)
    if any(not path.is_relative_to(DATA_ROOT) for path in EXTRA_WHEELS):
        raise ValueError("Public wheel inputs must stay within the authorized dependency cache")
    if not RECEIPTS.is_relative_to(RECEIPTS_ROOT):
        raise ValueError("Receipts must stay within the authorized new recovery directory")
    if not DATA.is_relative_to(DATA_ROOT) or DATA == DATA_ROOT:
        raise ValueError("Data must stay within a new authorized large-cache directory")
    if PREDECESSOR and not PREDECESSOR.is_relative_to(RECEIPTS_ROOT):
        raise ValueError("Predecessor is outside this recovery")
    os.environ["TMPDIR"] = str(DATA / "work")
    if args.mode == "launch":
        if (RECEIPTS / "LAUNCH.json").exists():
            raise ValueError("Refusing duplicate recovery launch")
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=False)
        for name in ("artifacts", "metadata", "work"):
            (DATA / name).mkdir()
        freeze()
        worker = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                  "--receipts", str(RECEIPTS), "--data", str(DATA),
                  "--dependency-cache-url", DEPENDENCY_CACHE_URL]
        if PREDECESSOR:
            worker += ["--predecessor", str(PREDECESSOR)]
        for wheel in EXTRA_WHEELS:
            worker += ["--extra-wheel", str(wheel)]
        with (RECEIPTS / "worker.log").open("xb") as log:
            process = subprocess.Popen(worker, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=log, start_new_session=True)
        save(RECEIPTS / "LAUNCH.json", {
            "at": now(), "pid": process.pid, "command": worker,
            "successor": "Same worker starts exactly one native Harbor Trial after dependency gate",
            "setup_deadline_sec": SETUP_DEADLINE, "native_timeout_sec": 7200,
        })
        print(json.dumps({"pid": process.pid, "receipts": str(RECEIPTS)}), flush=True)
        return
    try:
        if not (RECEIPTS / "PROTOCOL.json").exists() or (RECEIPTS / "STATE.json").exists():
            raise ValueError("Worker needs a fresh frozen continuation")
        def setup_deadline(_signum, _frame):
            raise TimeoutError(f"Bounded dependency setup exceeded {SETUP_DEADLINE} seconds")

        signal.signal(signal.SIGALRM, setup_deadline)
        signal.alarm(SETUP_DEADLINE)
        prepare()
        signal.alarm(0)
        asyncio.run(score())
    except Exception as error:
        signal.alarm(0)
        (RECEIPTS / "failure.txt").write_text(traceback.format_exc())
        state("BLOCKED", error_type=type(error).__name__, error=str(error))
        subprocess.run(
            ["docker", "rm", "-f", "sam-verifier-dependencies-" + RECEIPTS.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        raise


if __name__ == "__main__":
    main()
