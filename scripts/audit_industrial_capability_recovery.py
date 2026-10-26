#!/usr/bin/env python3
"""Read-only TRAIN admission sidecars for the frozen industrial capability runs.

This audit does not repair original receipts, invoke actors/verifiers, or consume
held-out outcomes. It distinguishes raw instruction files from Harbor's delivered
instruction, and rejects dependency-error reward files without executed tests.
The optional watcher owns only its new sidecar directory, never a native trial.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REMOTE = Path("/path/to/data/phiagent/work/deepseek-harness/industrial-capability/20260930-v1")
RECOVERIES = (
    "/path/to/data/phiagent/work/harness-terminal-recovery/20260930-capability-verifier-v1",
    "/path/to/data/phiagent/work/harness-terminal-recovery/20260930-capability-headless-verifier-v1",
)
PROMPT_SHA = "f2755a7b715e692b6fe794cdf7b316dfb58b301759dade0bf3f67c2ae02e57e5"
PARTITION_SHA = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
FIELDS = ("input_tokens", "output_tokens", "total_tokens")
GIB = 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def record(path: Path, evidence: dict) -> dict:
    value = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    evidence[str(path)] = value
    return value


def load(path: Path, evidence: dict) -> dict:
    raw = path.read_bytes()
    evidence[str(path)] = {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
    }
    return json.loads(raw)


def within(path: Path, root: Path) -> Path:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Evidence escaped its declared owner: {path}")
    return path


def check(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def owner_state(owner: dict) -> dict:
    pid = owner.get("pid")
    result = {**owner, "alive": False}
    if type(pid) is not int:
        return result
    try:
        process = Path("/proc") / str(pid)
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        result.update(observed_start_ticks=fields[19], command=command)
        result["alive"] = fields[0] != "Z" and (
            owner.get("start_ticks") is None or str(owner["start_ticks"]) == fields[19]
        )
    except (OSError, IndexError):
        pass
    return result


def prompt_boundary(raw_sha: str, expected: dict, actual: dict) -> dict:
    check(actual == expected, "Delivered shared prompt receipt differs from native rendering")
    check(actual.get("sha256") == PROMPT_SHA, "Different shared industrial parent")
    check(actual.get("shared_block_count") == 1, "Shared prompt not applied exactly once")
    check(actual.get("original_task_preserved") is True, "Native instruction not preserved")
    return {
        "status": "VERIFIED_NATIVE_RENDERING",
        "raw_instruction_sha256": raw_sha,
        "native_instruction_sha256": actual["original_task_sha256"],
        "delivered_instruction_sha256": actual["rendered_task_sha256"],
        "raw_and_native_instruction_differ": raw_sha != actual["original_task_sha256"],
        "original_receipt_modified": False,
        "boundary": "Raw task-file hash and Harbor strip_canary instruction hash are distinct",
    }


def executed_tests(ctrf: dict | None, score: float) -> dict:
    check(isinstance(ctrf, dict), "NO_EXECUTED_TEST_RECEIPT")
    results = ctrf.get("results") or {}
    tests = results.get("tests") or []
    summary = results.get("summary") or {}
    check(bool(tests), "NO_EXECUTED_TEST_RECEIPT")
    names = [test.get("name") for test in tests]
    statuses = [test.get("status") for test in tests]
    check(all(isinstance(name, str) and name for name in names), "Missing native test identity")
    check(len(set(names)) == len(names), "Duplicate native test receipt")
    check(all(status in ("passed", "failed") for status in statuses), "Unexecuted native tests")
    check(summary.get("tests") == len(tests), "Native test count mismatch")
    passed, failed = statuses.count("passed"), statuses.count("failed")
    check(summary.get("passed") == passed and summary.get("failed") == failed,
          "Native test summary mismatch")
    check((score == 1) == (failed == 0), "Reward does not match executed native tests")
    return {"tests": len(tests), "passed": passed, "failed": failed, "names": names}


def call_accounting(calls: list[dict], original: dict, *, settled: bool,
                    problems: list[str] | None = None) -> dict:
    issues = list(problems or [])
    identities = [call.get("call_id") for call in calls]
    unique = bool(all(isinstance(key, str) and key for key in identities)) or not calls
    unique = unique and len(set(identities)) == len(identities)
    if not unique:
        issues.append("Duplicate or missing actual provider-call identity")
    unknown = [
        call.get("call_id") for call in calls
        if call.get("ended") is False or any(
            type((call.get("usage") or {}).get(field)) is not int
            or (call.get("usage") or {}).get(field, -1) < 0 for field in FIELDS
        )
    ]
    if not settled:
        issues.append("Provider accounting is not settled")
    for call in calls:
        values = [(call.get("usage") or {}).get(field) for field in FIELDS]
        if all(type(value) is int and value >= 0 for value in values):
            if values[0] + values[1] != values[2]:
                issues.append(f"Provider token buckets disagree: {call.get('call_id')}")
    complete = unique and settled and not unknown and not issues
    totals = {}
    for field in FIELDS:
        values = [(call.get("usage") or {}).get(field) for call in calls]
        known = sum(value for value in values if type(value) is int and value >= 0)
        totals[field] = {
            "known_subtotal": known if unique else None,
            "complete_total": known if complete else None,
        }
    measured = {field: totals[field]["complete_total"] for field in FIELDS}
    matches = (
        complete and original.get("usage_status") == "MEASURED"
        and original.get("provider_call_count") == len(calls)
        and original.get("usage") == measured
    )
    return {
        "status": "MEASURED" if complete else "INCOMPLETE",
        "tokens": totals, "provider_call_count": len(calls),
        "unresolved_call_ids": unknown, "problems": issues,
        "matches_original_aggregate": matches,
        "original_cost_record": original,
        "original_record_rewritten": False,
        "cost_usd": None, "cost_usd_status": "provider_did_not_report_audited_usd",
        "calls": calls, "accounting_unit": "Actual actor call identity, exactly once",
    }


def unavailable_cost(original: dict | None, reason: str) -> dict:
    return {
        "status": "INCOMPLETE", "original_cost_record": original, "problems": [reason],
        "tokens": {field: {"known_subtotal": 0, "complete_total": None} for field in FIELDS},
        "cost_usd": None, "calls": [],
    }


def aggregate_costs(rows: list[dict], required_attempts: int = 18) -> dict:
    calls = [
        call.get("call_id") for row in rows for call in (row.get("cost") or {}).get("calls", [])
    ]
    duplicates = sorted((key for key, count in Counter(calls).items() if count > 1), key=str)
    totals = {}
    for field in FIELDS:
        costs = [((row.get("cost") or {}).get("tokens") or {}).get(field, {}) for row in rows]
        known = [cost.get("known_subtotal") for cost in costs]
        complete = [cost.get("complete_total") for cost in costs]
        totals[field] = {
            "known_subtotal": sum(value for value in known if type(value) is int)
            if not duplicates else None,
            "complete_total": sum(complete) if len(rows) == required_attempts and not duplicates
            and all(type(value) is int for value in complete) else None,
        }
    return {"tokens": totals, "duplicate_cross_attempt_call_ids": duplicates}


def merged_gate(track: str, train_ids: list[str], rows: list[dict],
                identity_valid: bool = True) -> dict:
    check(len(train_ids) == 18 and len(set(train_ids)) == 18, "Expected all18 TRAIN identities")
    identities = [row["task_id"] for row in rows]
    check(len(set(identities)) == len(identities), "Duplicate task in admission sidecar")
    check(set(identities).issubset(train_ids), "Held-out or foreign task in TRAIN admission")
    scored = [row for row in rows if row.get("admitted") is True]
    check(all(type(row.get("native_score")) in (int, float)
              and row["native_score"] in (0, 1) for row in scored), "Nonbinary native score")
    successes = sum(row["native_score"] == 1 for row in scored)
    full = len(scored) == 18 and identity_valid
    passed = full and successes >= 9
    return {
        "status": ("PASSED" if passed else "INSUFFICIENT_CAPABILITY") if full
        else ("WAITING_FULL_TRAIN" if identity_valid else "BLOCKED_IDENTITY"),
        "phase": "train", "track": track, "native_scored": len(scored),
        "native_successes": successes, "required": 18, "minimum_successes": 9,
        "success_rate": successes / 18 if full else None,
        "rsi_ready": passed, "automatic_rsi_execution": False,
        "shared_prompt_sha256": PROMPT_SHA, "partition_sha256": PARTITION_SHA,
        "complete_valid_capable_parent": passed,
        "scope": "TRAIN parent capability only; no held-out or token-efficiency claim",
    }


NATIVE_FACTS_PROGRAM = r"""
import hashlib, inspect, json, sys, warnings
from pathlib import Path
from harbor.models.task.task import Task, strip_canary
warnings.filterwarnings("ignore", category=DeprecationWarning)
request = json.load(sys.stdin)
run = Path(request["run"])
sys.path.insert(0, str(run / "source"))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
reader = Path(inspect.getfile(Task))
out = {"instruction_reader": {"path": str(reader), "sha256": sha(reader),
       "strip_canary_source_sha256": hashlib.sha256(inspect.getsource(strip_canary).encode()).hexdigest()},
       "tasks": {}, "costs": {}}
for row in request["attempts"]:
    attempt = Path(row["attempt_id"])
    task = Task(task_dir=attempt / "task-source" / row["task_id"])
    original = task.instruction
    rendered = original + "\n\n" + request["prompt"]["text"]
    receipt = {**request["prompt"], "original_task_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "original_task_bytes": len(original.encode()),
        "rendered_task_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "shared_block_count": 1, "original_task_preserved": True}
    out["tasks"][str(attempt)] = {"expected_prompt": receipt, "checksum": task.checksum,
        "verifier_timeout_seconds": task.config.verifier.timeout_sec}
    worker = (row.get("cost") or {}).get("sdk_worker_dir")
    if request["track"] == "deepseek" and worker:
        worker = Path(worker)
        if not worker.resolve().is_relative_to((run / "private").resolve()):
            raise ValueError("SDK worker escaped frozen run")
        from integrations.deepseek_harness.harbor_agent import recover_runner_accounting, normalize_runner_usage
        accounting = recover_runner_accounting(worker)
        calls = []
        for item in accounting.get("attempts", []):
            usage = normalize_runner_usage(item.get("usage"))
            calls.append({"call_id": str(worker) + "#" + item["attempt_id"],
                "usage": {k: (usage["prompt_tokens_including_cache"] if k == "input_tokens" else usage[k])
                          for k in ("input_tokens", "output_tokens", "total_tokens")},
                "ended": item.get("ended"), "model": item.get("start", {}).get("model"),
                "purpose": item.get("start", {}).get("purpose")})
        out["costs"][str(attempt)] = {"calls": calls, "complete": accounting.get("complete"),
            "unknown_reasons": accounting.get("unknown_reasons", []),
            "snapshot_reconciliation": accounting.get("snapshot_reconciliation"),
            "stream_path": str(worker / "llm-streams.jsonl"),
            "stream_sha256": sha(worker / "llm-streams.jsonl")}
print(json.dumps(out))
"""


def runtime_facts(run: Path, track: str, config: dict, prompt: dict, rows: list[dict]) -> dict:
    result = subprocess.run(
        [config[track]["python"], "-B", "-c", NATIVE_FACTS_PROGRAM],
        input=json.dumps({"run": str(run), "track": track, "prompt": prompt, "attempts": rows}),
        text=True, capture_output=True, timeout=180, check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return json.loads(result.stdout)


def native_validation(native: dict, row: dict, facts: dict, task: Path) -> float:
    rewards = (native.get("verifier_result") or {}).get("rewards")
    check(isinstance(rewards, dict) and set(rewards) == {"reward"}, "MISSING_NATIVE_REWARD")
    score = rewards["reward"]
    check(type(score) in (int, float) and score in (0, 1), "Nonbinary native reward")
    check(native.get("task_name") == row["task_id"], "Native task name differs")
    check(native.get("task_checksum") == facts["checksum"], "Native task checksum differs")
    config = native.get("config") or {}
    check(Path(config["task"]["path"]).resolve() == task.resolve(), "Native task path differs")
    check(Path(native["task_id"]["path"]).resolve() == task.resolve(), "Native task ID differs")
    check(config.get("timeout_multiplier") == 1, "Native timeout multiplier changed")
    check(config.get("verifier_timeout_multiplier") in (None, 1), "Verifier deadline changed")
    check(not config.get("extra_instruction_paths"), "Undeclared extra task instructions")
    verifier = config.get("verifier") or {}
    check(verifier.get("disable") is False, "Native verifier disabled")
    check(verifier.get("override_timeout_sec") is None
          and verifier.get("max_timeout_sec") is None, "Native verifier deadline overridden")
    agent = config.get("agent") or {}
    check(agent.get("model_name") == "deepseek-flash", "Different actor backbone")
    kwargs = agent.get("kwargs") or {}
    check(kwargs.get("shared_prompt_sha256") == PROMPT_SHA, "Different actor common prompt")
    check(kwargs.get("residual", "") == "", "Parent has an undeclared residual")
    check((native.get("exception_info") or {}).get("exception_type")
          in (None, "AgentTimeoutError"), "Unresolved native infrastructure exception")
    return float(score)


def scorer_receipts(base: Path, native: dict, score: float, evidence: dict) -> dict:
    paths = list(base.glob("**/verifier/ctrf.json"))
    check(len(paths) == 1, "NO_EXECUTED_TEST_RECEIPT" if not paths else "Ambiguous native tests")
    receipt = paths[0]
    tests = executed_tests(load(receipt, evidence), score)
    reward = receipt.parent / "reward.txt"
    check(reward.is_file() and float(reward.read_text().strip()) == score,
          "Native reward file differs or is missing")
    record(reward, evidence)
    for output in receipt.parent.glob("*"):
        if output.is_file():
            record(output, evidence)
    timing = native.get("verifier") or {}
    check(timing.get("started_at") and timing.get("finished_at"), "Missing native verifier timer")
    start = datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))
    finish = datetime.fromisoformat(timing["finished_at"].replace("Z", "+00:00"))
    check(finish >= start, "Negative native verifier duration")
    return {
        **tests, "ctrf": evidence[str(receipt)], "reward": evidence[str(reward)],
        "timer": timing, "phase_wall_seconds": (finish - start).total_seconds(),
        "phase_wall_includes_native_setup": "recovery" not in native,
    }


def actor_binding(attempt: Path, track: str, row: dict, prompt_sha: str, evidence: dict) -> dict:
    if track == "codex":
        paths = list(attempt.glob("**/agent/trajectory.json"))
        check(len(paths) == 1, "Missing or ambiguous original actor trajectory")
        trajectory = load(paths[0], evidence)
        matches = [
            step for step in trajectory.get("steps", [])
            if step.get("source") == "user" and isinstance(step.get("message"), str)
            and text_sha(step["message"]) == prompt_sha
        ]
        check(len(matches) == 1, "Actual actor trajectory does not match delivered prompt")
        return {"kind": "native_codex_trajectory", **evidence[str(paths[0])],
                "matching_user_step": matches[0].get("step_id")}
    private = (row.get("cost") or {}).get("private_artifacts")
    check(bool(private), "Missing original SDK private artifact binding")
    path = within(Path(private), attempt.parent.parent / "private") / "worker-job.json"
    job = load(path, evidence)
    check(text_sha(job.get("prompt", "")) == prompt_sha, "SDK actor job received another prompt")
    check(job.get("model") == "deepseek-flash" and job.get("arm") == "baseline",
          "SDK actor is not the original backbone/parent")
    for name in ("COMMAND.json", "HARBOR_ENVIRONMENT.json"):
        q = path.parent / name
        if q.is_file():
            record(q, evidence)
    return {"kind": "official_sdk_worker_job", **evidence[str(path)],
            "source_revision": job.get("source_revision")}


def retained_workspace(attempt: Path, track: str, evidence: dict) -> dict:
    receipt = attempt / (
        "native-run/NATIVE_CONTAINER.json" if track == "codex" else "NATIVE_CONTAINER.json"
    )
    if not receipt.is_file():
        return {"status": "NO_ORIGINAL_CONTAINER_RECEIPT"}
    data = load(receipt, evidence)
    container = data.get("container_id", "")
    check(len(container) == 64 and all(char in "0123456789abcdef" for char in container),
          "Invalid original container identity")
    command = ["docker", "inspect", "--format", "{{json .State}}", container]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return {
        "at": now(), "container_receipt": evidence[str(receipt)], "container_id": container,
        "command": command, "returncode": result.returncode,
        "status": "PRESENT" if result.returncode == 0 else "ORIGINAL_WORKSPACE_UNAVAILABLE",
        "state": json.loads(result.stdout) if result.returncode == 0 else None,
        "error": result.stderr if result.returncode else None,
        "native_container_mutated": False,
    }


def original_cost(attempt: Path, row: dict, facts: dict, config: dict, evidence: dict) -> dict:
    path = within(Path(row["provider_call_ledger"]), attempt)
    ledger = load(path, evidence)
    check(ledger.get("native_actor_attempt_id") == str(attempt), "Cost ledger belongs to another actor")
    calls = ledger.get("calls", [])
    problems = []
    if row["track"] == "deepseek":
        raw = facts.get("costs", {}).get(str(attempt))
        if raw is None:
            return call_accounting(calls, row["cost"], settled=False,
                                   problems=["No source-bound SDK stream accounting"])
        stream = Path(raw["stream_path"])
        record(stream, evidence)
        check(evidence[str(stream)]["sha256"] == raw["stream_sha256"],
              "Provider stream changed while snapshot was collected")
        cost = call_accounting(raw["calls"], row["cost"], settled=raw["complete"] is True,
                               problems=raw["unknown_reasons"])
        cost["raw_stream_reconciliation"] = raw.get("snapshot_reconciliation")
        cost["original_ledger_sha256"] = evidence[str(path)]["sha256"]
        return cost
    trial = load(attempt / "TRIAL_ID.json", evidence)["trial_id"]
    call_root = Path(config["codex"]["provider_ledger"]) / "calls" / trial
    actual = {str(p) for p in call_root.iterdir() if p.is_dir()} if call_root.is_dir() else set()
    check(actual == {call.get("call_id") for call in calls},
          "Provider attempt directories and original call ledger differ")
    for call in calls:
        receipt = within(Path(call["receipt"]), call_root)
        raw = load(receipt, evidence)
        if evidence[str(receipt)]["sha256"] != call["receipt_sha256"]:
            problems.append(f"Original provider receipt changed: {receipt}")
        if {field: (raw.get("usage") or {}).get(field) for field in FIELDS} != call.get("usage"):
            problems.append(f"Provider usage differs from the original per-call receipt: {receipt}")
        if raw.get("model") != "deepseek-flash":
            problems.append(f"Unexpected provider model: {receipt}")
    return call_accounting(calls, row["cost"], settled=True, problems=problems)


def recovery_binding(recovery: dict, row: dict, native: dict, recovered: dict,
                     row_sha: str, container: dict, container_sha: str,
                     verifier_sha: str, deadline: float) -> None:
    source = recovery.get("source_binding") or {}
    check(recovery.get("status") == "NATIVE_VERIFIER_RECOVERED", "Recovery is not complete")
    check(recovery.get("callback") == row["attempt_id"] and recovery.get("callback_sha256") == row_sha,
          "Recovery belongs to a different original attempt")
    check(recovery.get("original_native_result") == row["native_result"]
          and recovery.get("original_native_result_sha256") == row["native_result_sha256"],
          "Recovery original native result is not source-bound")
    check(source.get("capability_result_sha256") == row_sha
          and source.get("container_receipt_sha256") == container_sha,
          "Recovery source binding differs")
    check(source.get("track") == row["track"] and source.get("shared_prompt_sha256") == PROMPT_SHA
          and source.get("partition_sha256") == PARTITION_SHA, "Recovery parent/partition differs")
    check(container.get("captured_before_actor") is True
          and container.get("container_id") == recovery.get("original_container_id"),
          "Recovery workspace was not the retained original actor container")
    check(recovery.get("additional_actor_calls") == 0, "Recovery re-executed an actor")
    check(recovery.get("official_verifier_sha256") == verifier_sha
          and recovery.get("official_verifier_timeout_seconds") == deadline,
          "Recovery changed native scorer or deadline")
    check(recovery.get("original_actor_usage") == row.get("cost"), "Original actor costs were changed")
    mutable = {"verifier_result", "verifier", "exception_info", "recovery"}
    check({k: v for k, v in native.items() if k not in mutable}
          == {k: v for k, v in recovered.items() if k not in mutable},
          "Recovery modified original actor artifacts or task configuration")


def audit_recovery(path: Path, row: dict, native: dict, facts: dict, evidence: dict,
                   verifier_sha: str) -> tuple[dict, dict]:
    recovery = load(path / "RECOVERY.json", evidence)
    native_path = within(Path(recovery["native_result"]), path)
    recovered = load(native_path, evidence)
    check(evidence[str(native_path)]["sha256"] == recovery["native_result_sha256"],
          "Recovered native result changed")
    attempt = Path(row["attempt_id"])
    container_path = attempt / (
        "native-run/NATIVE_CONTAINER.json" if row["track"] == "codex" else "NATIVE_CONTAINER.json"
    )
    container = load(container_path, evidence)
    recovery_binding(
        recovery, row, native, recovered, evidence[str(attempt / "RESULT.json")]["sha256"],
        container, evidence[str(container_path)]["sha256"], verifier_sha,
        facts["verifier_timeout_seconds"],
    )
    config = load(path / "CONFIG.json", evidence)
    check(recovered.get("recovery") == config, "Recovered native configuration differs")
    setup = load(path / "DEPENDENCY_SETUP.json", evidence)
    check(setup.get("return_code") == 0 and setup.get("workspace_commands_excluded") is True
          and setup.get("additional_actor_calls") == 0, "Dependency-only recovery not established")
    task = attempt / "task-source" / row["task_id"]
    check(setup.get("original_test_script_sha256") == digest(task / "tests/test.sh"),
          "Recovery installed dependencies for a different scorer")
    source = load(path / "SOURCE.json", evidence)
    check(record(Path(source["script"]), evidence)["sha256"] == source.get("sha256"),
          "Recovery implementation source changed")
    if recovery.get("dependency_manifest"):
        cache_path = Path(recovery["dependency_manifest"])
        cache = load(cache_path, evidence)
        check(evidence[str(cache_path)]["sha256"] == recovery["dependency_manifest_sha256"],
              "Recovery dependency receipt changed")
        check(cache.get("status") == "OFFLINE_DEPENDENCIES_VERIFIED"
              and cache.get("test_script_sha256") == digest(task / "tests/test.sh")
              and cache.get("task_config_sha256") == digest(task / "task.toml")
              and cache.get("additional_model_calls") == 0
              and cache.get("native_verifier_calls") == 0, "Dependency cache is not source-bound")
        archive = Path(cache["archive"])
        check(record(archive, evidence)["sha256"] == cache["archive_sha256"],
              "Recovery dependency archive changed")
    score = native_validation(recovered, row, facts, task)
    check(recovery.get("score") == score, "Recovery score receipt differs")
    tests = scorer_receipts(path / "native", recovered, score, evidence)
    check(tests["phase_wall_seconds"] <= facts["verifier_timeout_seconds"],
          "Recovered scorer exceeded the original native deadline")
    return recovered, {
        "native_result": evidence[str(native_path)], "recovery": evidence[str(path / "RECOVERY.json")],
        "score": score, "tests": tests, "additional_actor_calls": 0,
    }


def recovery_inventory(roots: list[str], evidence: dict) -> list[dict]:
    output = []
    for root in roots:
        path = Path(root)
        check(path.resolve().is_relative_to(
            Path("/path/to/data/phiagent/work/harness-terminal-recovery")), "Foreign recovery directory")
        for config_path in sorted(path.glob("*/CONFIG.json")):
            config = load(config_path, evidence)
            callback = config.get("callback", "")
            if not str(callback).startswith(str(REMOTE) + "/train-"):
                continue
            entry = {"directory": str(config_path.parent), "callback": callback,
                     "task_id": config.get("task_id"), "config": evidence[str(config_path)]}
            for name in ("STATE.json", "RECOVERY.json", "SOURCE.json", "DEPENDENCY_SETUP.json"):
                item = config_path.parent / name
                if item.is_file():
                    data = load(item, evidence)
                    entry[name] = evidence[str(item)]
                    if name == "STATE.json":
                        entry["state"] = data
                        entry["owner"] = owner_state(data)
                        if str(config_path.parent) not in entry["owner"].get("command", ""):
                            entry["owner"]["alive"] = False
            entry["complete"] = (config_path.parent / "RECOVERY.json").is_file()
            output.append(entry)
        queued = path / "QUEUED_RECOVERY.json"
        if queued.is_file():
            queue = load(queued, evidence)
            if not any(entry["directory"] == queue["output"] for entry in output):
                output.append({
                    "directory": queue["output"], "callback": queue["callback"],
                    "task_id": queue["task_id"], "complete": False,
                    "state": {"status": "WAITING_VERIFIED_DEPENDENCIES"},
                    "owner": owner_state(queue["owner"]), "queue": evidence[str(queued)],
                })
    return output


def audit_track(track: str, recoveries: list[dict], evidence: dict, *,
                observations: list[dict] | None = None, attempt_root: Path | None = None) -> dict:
    """Audit original TRAIN or declared replacement rows against the same frozen context."""
    run = REMOTE / f"train-{track}-v1"
    config = load(run / "CONFIG.json", evidence)
    prompt = load(run / "SHARED_PROMPT.json", evidence)
    partition = load(run / "SPLIT_MANIFEST.json", evidence)
    report = load(run / "REPORT.json", evidence)
    frozen = load(run / "FROZEN_IDENTITY.json", evidence)
    identity_errors = []
    for name, expected in frozen.items():
        path = within(run / name, run)
        if record(path, evidence)["sha256"] != expected:
            identity_errors.append(f"Frozen identity changed: {name}")
    check(prompt.get("sha256") == PROMPT_SHA and text_sha(prompt.get("text", "")) == PROMPT_SHA,
          "Shared prompt does not match the declared parent")
    check(evidence[str(run / "SPLIT_MANIFEST.json")]["sha256"] == PARTITION_SHA,
          "Different partition manifest")
    check(config.get("phase") == "train" and config.get("partition_sha256") == PARTITION_SHA
          and config.get("shared_prompt_sha256") == PROMPT_SHA, "Different frozen parent identity")
    check(partition.get("counts") == {"train": 18, "validation": 18, "test": 53},
          "Different population split")
    ids = partition["splits"]["train"]
    sources = load(run / "SOURCE_SHA256.json", evidence)
    for name, expected in sources.items():
        path = within(run / "source" / name, run / "source")
        if record(path, evidence)["sha256"] != expected:
            identity_errors.append(f"Frozen implementation changed: {name}")
    runtime = load(run / "NATIVE_RUNTIME.json", evidence)
    for filename, expected in runtime["files"].items():
        if record(Path(filename), evidence)["sha256"] != expected:
            identity_errors.append(f"Frozen runtime changed: {filename}")
    verifier_files = [
        sha for filename, sha in runtime["files"].items() if filename.endswith("/verifier/verifier.py")
    ]
    check(len(verifier_files) == 1, "Missing unique pinned native verifier")
    task_sources = load(run / "TASK_SOURCE_SHA256.json", evidence)
    originals = report.get("attempts", []) if observations is None else observations
    check(len({row["attempt_id"] for row in originals}) == len(originals), "Duplicate original attempt")
    facts = runtime_facts(run, track, config, prompt, originals)
    reader = record(Path(facts["instruction_reader"]["path"]), evidence)
    check(reader["sha256"] == facts["instruction_reader"]["sha256"],
          "Instruction reader source changed during the snapshot")
    rows = []
    for observation in originals:
        attempt = within(Path(observation["attempt_id"]), attempt_root or run / "tasks")
        row = {
            "task_id": observation["task_id"], "attempt_id": str(attempt), "admitted": False,
            "native_score": None, "original_status": observation["status"],
            "original_native_score": observation.get("native_score"), "blockers": [],
            "recovery_attempts": [r for r in recoveries if r["callback"] == str(attempt)],
        }
        try:
            original = load(attempt / "RESULT.json", evidence)
            check(original == observation, "Report and original attempt receipt differ")
            check(original["phase"] == "train" and original["track"] == track
                  and original["task_id"] in ids and original["shared_prompt_sha256"] == PROMPT_SHA
                  and original["partition_sha256"] == PARTITION_SHA, "Foreign original attempt")
            row["original_result"] = evidence[str(attempt / "RESULT.json")]
            row["original_actor_exception"] = original.get("native_exception")
            try:
                row["cost"] = original_cost(attempt, original, facts, config, evidence)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["cost"] = unavailable_cost(original.get("cost"), str(error))
            task = attempt / "task-source" / original["task_id"]
            hashes = {
                p.relative_to(task).as_posix(): record(p, evidence)["sha256"]
                for p in sorted(task.rglob("*")) if p.is_file()
            }
            check(hashes == task_sources[original["task_id"]], "Original task bytes changed")
            row["task_source_sha256"] = hashes
            task_facts = facts["tasks"][str(attempt)]
            receipts = original["shared_prompt_receipts"]
            check(len(receipts) == 1, "Missing or duplicate shared prompt receipt")
            receipt_path = within(Path(receipts[0]), attempt)
            row["prompt_boundary"] = prompt_boundary(
                hashes["instruction.md"], task_facts["expected_prompt"], load(receipt_path, evidence),
            )
            row["actor_artifact"] = actor_binding(
                attempt, track, original, task_facts["expected_prompt"]["rendered_task_sha256"], evidence,
            )
            native_path = within(Path(original["native_result"]), attempt)
            native = load(native_path, evidence)
            check(evidence[str(native_path)]["sha256"] == original["native_result_sha256"],
                  "Original native result changed")
            row["original_native_result"] = evidence[str(native_path)]
            row["original_raw_reward"] = (native.get("verifier_result") or {}).get("rewards")
            row["official_verifier_sha256"] = verifier_files[0]
            row["native_verifier_timeout_seconds"] = task_facts["verifier_timeout_seconds"]
            selected = None
            try:
                score = native_validation(native, original, task_facts, task)
                tests = scorer_receipts(attempt, native, score, evidence)
                selected = {"native_result": evidence[str(native_path)], "score": score, "tests": tests}
            except ValueError as error:
                row["original_score_blocker"] = str(error)
            verified_recoveries = []
            for recovery in row["recovery_attempts"]:
                if recovery["complete"]:
                    try:
                        _, verified = audit_recovery(
                            Path(recovery["directory"]), original, native, task_facts,
                            evidence, verifier_files[0],
                        )
                        verified_recoveries.append(verified)
                    except (OSError, ValueError, KeyError) as error:
                        row["blockers"].append(f"Recovery audit: {error}")
            candidates = ([selected] if selected else []) + verified_recoveries
            check(len({candidate["score"] for candidate in candidates}) <= 1,
                  "Conflicting genuine verifier outcomes; no success-based selection")
            if candidates and not row["blockers"]:
                chosen = candidates[0]
                row.update(admitted=True, native_score=chosen["score"], selected_native=chosen)
                row["verified_recoveries"] = verified_recoveries
            elif not row["blockers"]:
                row["blockers"].append(row.get("original_score_blocker", "No native evidence"))
            if not row["admitted"]:
                row["retained_workspace"] = retained_workspace(attempt, track, evidence)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["blockers"].append(str(error))
        rows.append(row)
    observed = {row["task_id"] for row in rows}
    for task_id in ids if observations is None else []:
        if task_id not in observed:
            active = report.get("active", {}).get(task_id)
            rows.append({
                "task_id": task_id, "attempt_id": (active or {}).get("directory"),
                "admitted": False, "native_score": None,
                "blockers": ["LIVE_NATIVE_OWNER" if active else "NOT_YET_REPORTED"],
                "active_owner": owner_state(active or {}),
                "cost": unavailable_cost(None, "Original actor cost is not yet settled"),
            })
    latest = json.loads((run / "REPORT.json").read_text())
    latest_attempts = {item["attempt_id"]: item for item in latest.get("attempts", [])}
    if observations is None and any(latest_attempts.get(item["attempt_id"]) != item for item in originals):
        identity_errors.append("A published original attempt changed during this snapshot")
    if identity_errors:
        for row in rows:
            if row["admitted"]:
                row["score_before_identity_check"] = row["native_score"]
                row.update(admitted=False, native_score=None)
                row["blockers"].append("GLOBAL_FROZEN_IDENTITY_CHECK_FAILED")
    gate = merged_gate(track, ids, rows, not identity_errors)
    owner = owner_state(load(run / "OWNER.json", evidence))
    for filename in ("LEASE.json", "QUEUE.json", "CAPABILITY_GATE.json"):
        if (run / filename).is_file():
            record(run / filename, evidence)
    return {
        "run": str(run), "controller": owner, "native_status": report.get("status"),
        "identity_errors": identity_errors, "instruction_reader": facts["instruction_reader"],
        "original_report_at": report.get("at"), "report_progressed_during_read": latest != report,
        "gate": gate, "rows": rows, "accounting": {
            **aggregate_costs(rows), "cost_usd": None, "new_actor_calls": 0, "new_verifier_calls": 0,
            "unique_reported_actor_attempts": len(originals),
            "incomplete_task_costs": [row["task_id"] for row in rows
                                    if (row.get("cost") or {}).get("status") != "MEASURED"],
            "scope": "All original actor attempts; live/unreported costs remain incomplete",
        },
    }


def select_prospective_rows(originals: list[dict], replacements: dict[str, dict],
                            declarations: dict) -> tuple[list[dict], list[dict]]:
    """Selection is fixed by the prospective declaration, never by reward."""
    check(set(replacements) == set(declarations), "Every declared replacement must have a status row")
    original_by_task = {row["task_id"]: row for row in originals}
    check(len(original_by_task) == len(originals), "Duplicate original task")
    selected = []
    for row in originals:
        task_id = row["task_id"]
        if task_id not in declarations:
            selected.append(row)
            continue
        declaration, replacement = declarations[task_id], replacements[task_id]
        check(row.get("admitted") is False, "Already-valid original cannot be replaced")
        check(row["attempt_id"] == declaration["original_attempt"], "Wrong original attempt")
        check(replacement["task_id"] == task_id
              and replacement["attempt_id"] == declaration["replacement_attempt"],
              "Wrong replacement task or attempt")
        selected.append({
            **replacement, "admission_kind": "prospective_infrastructure_continuation",
            "original_attempt_id": row["attempt_id"],
            "original_attempt_retained_in_accounting": True,
            "restored_original_workspace": False,
        })
    check(set(declarations).issubset(original_by_task), "Foreign replacement task")
    attempts = [*originals, *replacements.values()]
    identities = [row.get("attempt_id") for row in attempts if row.get("attempt_id")]
    check(len(identities) == len(set(identities)), "Duplicate actual actor attempt")
    return selected, attempts


def live_replacement_cost(attempt: Path, config: dict, evidence: dict) -> dict:
    identity = attempt / "TRIAL_ID.json"
    if not identity.is_file():
        result = unavailable_cost(None, "Replacement has not dispatched an actor")
        result["provider_call_count"] = 0
        return result
    trial_id = load(identity, evidence)["trial_id"]
    directory = Path(config["codex"]["provider_ledger"]) / "calls" / trial_id
    calls = []
    for call in sorted(directory.iterdir()) if directory.is_dir() else []:
        receipt = call / "RECEIPT.json"
        raw = load(receipt, evidence) if receipt.is_file() else {}
        calls.append({
            "call_id": str(call), "usage": {key: (raw.get("usage") or {}).get(key) for key in FIELDS},
            "ended": bool(raw.get("finished_at")),
            "receipt": evidence.get(str(receipt)), "model": raw.get("model"),
        })
    return call_accounting(calls, {}, settled=False,
                           problems=["Fresh native actor or final accounting is still owned"])


def apply_prospective_manifest(base: dict, manifest_path: Path, expected_sha: str,
                               evidence: dict) -> dict:
    from scripts.continue_codex_capability_infrastructure import (
        original_eligible, replacement_config, validate_plan,
    )

    run = within(manifest_path.parent, Path("/path/to/data/phiagent/work/harness-terminal-recovery"))
    plan = load(manifest_path, evidence)
    check(evidence[str(manifest_path)]["sha256"] == expected_sha, "Prospective manifest changed")
    validate_plan(plan, run)
    check(record(Path(plan["source"]["path"]), evidence)["sha256"] == plan["source"]["sha256"],
          "Prospective native runner changed")
    check(digest(Path(__file__).with_name("continue_codex_capability_infrastructure.py"))
          == plan["source"]["sha256"], "Auditor and prospective runner contract versions differ")
    original_run = Path(plan["original_parent_run"])
    config = load(original_run / "CONFIG.json", evidence)
    observations, replacements = [], {}
    parents = {row["task_id"]: row for row in base["rows"]}
    for task_id, item in plan["tasks"].items():
        parent = parents[task_id]
        proof_path = within(Path(item["original_proof"]["path"]), run)
        proof = load(proof_path, evidence)
        check(evidence[str(proof_path)]["sha256"] == item["original_proof"]["sha256"],
              "Original failure/settlement proof changed")
        original_eligible(proof)
        check(parent.get("original_score_blocker") == "NO_EXECUTED_TEST_RECEIPT",
              "Original trial has genuine or unresolved native test evidence")
        for reference in (
            proof["original_result"], proof["original_native_result"], proof["native_template"],
            *proof["provider_receipts"],
        ):
            check(record(Path(reference["path"]), evidence)["sha256"] == reference["sha256"],
                  "Original paid attempt or settlement evidence changed")
        attempt = within(Path(item["replacement_attempt"]), run / "tasks")
        owner_path, state_path = attempt / "OWNER.json", attempt / "STATE.json"
        owner = owner_state(load(owner_path, evidence)) if owner_path.is_file() else {"alive": False}
        state = load(state_path, evidence) if state_path.is_file() else {"status": "NOT_STARTED"}
        entry = {
            "task_id": task_id, "attempt_id": str(attempt), "admitted": False, "native_score": None,
            "blockers": [state["status"]], "active_owner": owner, "recovery_status": state,
            "cost": live_replacement_cost(attempt, config, evidence),
        }
        start_path = attempt / "ACTOR_ATTEMPT_START.json"
        if start_path.is_file():
            start = load(start_path, evidence)
            check(start["prospective_manifest"]["sha256"] == expected_sha
                  and start["original_proof"] == item["original_proof"],
                  "New actor was not dispatched from this prospective declaration")
            check(datetime.fromisoformat(start["at"]) >= datetime.fromisoformat(plan["created_at"]),
                  "Replacement was chosen after its actor outcome")
            check(start.get("maximum_actor_attempts") == 1 and start.get("fresh_actor_attempt") is True
                  and start.get("restored_original_workspace") is False,
                  "Replacement is not a single fresh actor attempt")
            cache_path = Path(item["dependency_cache"])
            cache = load(cache_path, evidence)
            check(evidence[str(cache_path)]["sha256"] == start["dependency_cache"]["sha256"]
                  and cache.get("status") == "OFFLINE_DEPENDENCIES_VERIFIED"
                  and cache.get("additional_model_calls") == 0 and cache.get("native_verifier_calls") == 0,
                  "Native dependencies were not verified before the new actor")
            task = attempt / "task-source" / task_id
            check(cache["test_script_sha256"] == digest(task / "tests/test.sh")
                  and cache["task_config_sha256"] == digest(task / "task.toml")
                  and record(Path(cache["archive"]), evidence)["sha256"] == cache["archive_sha256"],
                  "Replacement dependency cache differs from the exact native task")
            trial_id = load(attempt / "TRIAL_ID.json", evidence)["trial_id"]
            template = load(Path(proof["native_template"]["path"]), evidence)
            expected = replacement_config(template, task, attempt, trial_id, cache["verifier_environment"])
            new_config_path = attempt / "native-run/CONFIG.json"
            new_config = load(new_config_path, evidence)
            check(new_config == expected
                  and evidence[str(new_config_path)]["sha256"] == start["native_config"]["sha256"],
                  "Replacement changed the native model, prompt, context, scorer, or limits")
            entry["prospective_actor_start"] = evidence[str(start_path)]
            result_path = attempt / "RESULT.json"
            if result_path.is_file():
                row = load(result_path, evidence)
                native = load(Path(row["native_result"]), evidence)
                original_native = load(Path(proof["original_native_result"]["path"]), evidence)
                check(native["config"] == replacement_config(
                    original_native["config"], task, attempt, trial_id, cache["verifier_environment"],
                ), "Executed replacement native configuration differs from the original protocol")
                observations.append(row)
        replacements[task_id] = entry
    audited = audit_track(
        "codex", [], evidence, observations=observations, attempt_root=run / "tasks",
    )
    for row in audited["rows"]:
        entry = replacements[row["task_id"]]
        replacements[row["task_id"]] = {
            **entry, **row, "recovery_status": {
                **entry["recovery_status"],
                "admission_status": "ADMITTED" if row["admitted"] else "UNSCORED_OR_INVALID",
            },
        }
    selected, attempts = select_prospective_rows(base["rows"], replacements, plan["tasks"])
    errors = [*base["identity_errors"], *audited["identity_errors"]]
    new_call_counts = [row["cost"].get("provider_call_count") for row in replacements.values()]
    gate = merged_gate("codex", [row["task_id"] for row in base["rows"]], selected, not errors)
    gate.update(parent_run=str(run), original_parent_run=str(original_run),
                prospective_manifest=evidence[str(manifest_path)])
    return {
        **base, "parent_run": str(run), "original_parent_run": str(original_run),
        "original_parent_gate": base["gate"], "prospective_manifest": evidence[str(manifest_path)],
        "gate": gate, "rows": selected, "admitted_rows": [row for row in selected if row["admitted"]],
        "all_actor_attempts": attempts, "identity_errors": errors,
        "prospective_replacement_status": replacements,
        "accounting": {
            **aggregate_costs(attempts, required_attempts=20),
            "cost_usd": None, "expected_final_actor_attempts": 20,
            "inherited_original_attempts": 18, "declared_new_attempts": 2,
            "new_actor_provider_calls_so_far": sum(new_call_counts)
            if all(type(value) is int for value in new_call_counts) else None,
            "new_actor_provider_calls_known_subtotal": sum(
                value for value in new_call_counts if type(value) is int
            ),
            "new_attempt_tokens": aggregate_costs(list(replacements.values()), required_attempts=2)["tokens"],
            "inherited_original_tokens": base["accounting"]["tokens"],
            "scope": "All18 original plus both prospectively declared new actor attempts, exactly once",
            "incomplete_task_costs": [row["attempt_id"] for row in attempts
                                    if (row.get("cost") or {}).get("status") != "MEASURED"],
        },
    }


def remote_snapshot(recovery_roots: list[str], replacement_manifest: Path | None = None,
                    replacement_sha256: str | None = None) -> dict:
    check(socket.gethostname() == "harnessmm-api-worker", "Native evidence reads require authorized A800-4")
    evidence = {}
    recoveries = recovery_inventory(recovery_roots, evidence)
    tracks = {track: audit_track(track, recoveries, evidence) for track in ("codex", "deepseek")}
    if replacement_manifest is not None:
        check(bool(replacement_sha256), "Pin the prospective declaration before collection")
        tracks["codex"] = apply_prospective_manifest(
            tracks["codex"], replacement_manifest, replacement_sha256, evidence,
        )
    resources = {}
    for path in ("/path/to/data", "/path/to/data/docker", "/root"):
        stats = os.statvfs(path)
        resources[path] = {"available_bytes": stats.f_bavail * stats.f_frsize,
                           "meets_8gib_reserve": stats.f_bavail * stats.f_frsize >= 8 * GIB}
    return {
        "schema": "industrial-capability-source-bound-admission/v1", "at": now(),
        "audit_source_sha256": digest(Path(__file__)),
        "hostname": socket.gethostname(), "read_only_remote": True,
        "original_receipts_modified": False, "additional_actor_calls": 0,
        "additional_native_verifier_calls": 0, "shared_prompt_sha256": PROMPT_SHA,
        "partition_sha256": PARTITION_SHA, "tracks": tracks, "recoveries": recoveries,
        "complete_valid_capable_parents": [
            track for track, data in tracks.items() if data["gate"]["complete_valid_capable_parent"]
        ],
        "storage": resources,
        "waiting_comparator": owner_state({"pid": 3133835}),
        "evidence": evidence,
    }


def atomic(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".writing")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def concise(snapshot: dict) -> dict:
    return {
        "at": snapshot["at"],
        "complete_valid_capable_parents": snapshot["complete_valid_capable_parents"],
        "tracks": {
            track: {
                "gate": data["gate"], "owner_alive": data["controller"]["alive"],
                "parent_run": data.get("parent_run", data["run"]),
                "accounting": data["accounting"],
                "blocked_tasks": [
                    {"task_id": row["task_id"], "blockers": row["blockers"]}
                    for row in data["rows"] if not row["admitted"]
                ],
                "identity_errors": data["identity_errors"],
            } for track, data in snapshot["tracks"].items()
        },
        "next_trigger": {
            "condition": "Native owner/recovery state or source-bound receipt changes",
            "action": "Recollect only; never restart a native owner, actor, verifier, or comparator",
            "write_target": "This independent admission sidecar only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-snapshot", action="store_true")
    parser.add_argument("--recovery-root", action="append")
    parser.add_argument("--replacement-manifest", type=Path)
    parser.add_argument("--replacement-sha256")
    parser.add_argument("--remote-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    recovery_roots = args.recovery_root or list(RECOVERIES)
    check(bool(args.replacement_manifest) == bool(args.replacement_sha256),
          "Supply both the prospective manifest and its frozen SHA256")
    if args.remote_snapshot:
        print(json.dumps(remote_snapshot(
            recovery_roots, args.replacement_manifest, args.replacement_sha256,
        ), sort_keys=True))
        return 0
    check(args.output is not None and args.remote_source is not None,
          "Specify a new sidecar directory and its frozen remote audit source")
    check(30 <= args.interval <= 600, "Read-only watcher interval must be 30–600 seconds")
    output = within(args.output.absolute(), ROOT / "artifacts")
    remote_source = within(args.remote_source, Path("/path/to/data/phiagent/work/harness-terminal-recovery"))
    output.mkdir(parents=True, exist_ok=True)
    check(os.statvfs(output).f_bavail * os.statvfs(output).f_frsize >= 8 * GIB,
          "Local sidecar path is below the 8GiB writable reserve")
    lock = (output / "OBSERVER.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote_args = [
        "python3", "-B", str(remote_source), "--remote-snapshot",
        *[part for root in recovery_roots for part in ("--recovery-root", root)],
    ]
    if args.replacement_manifest:
        remote_args += ["--replacement-manifest", str(args.replacement_manifest),
                        "--replacement-sha256", args.replacement_sha256]
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ControlMaster=no", "-o", "ControlPath=none", "a800-4",
               shlex.join(remote_args)]
    manifest = {
        "schema": "industrial-capability-admission-observer/v1",
        "source": str(Path(__file__).resolve()), "source_sha256": digest(Path(__file__)),
        "remote_source": str(remote_source), "remote_command": command,
        "recovery_roots": recovery_roots, "shared_prompt_sha256": PROMPT_SHA,
        "partition_sha256": PARTITION_SHA, "read_only_remote": True,
        "native_launch_authority": False, "cumulative_token_limit": None,
        "native_facts_program_sha256": text_sha(NATIVE_FACTS_PROGRAM),
        "prospective_replacement_manifest": str(args.replacement_manifest)
        if args.replacement_manifest else None,
        "prospective_replacement_manifest_sha256": args.replacement_sha256,
    }
    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        check(json.loads(manifest_path.read_text()) == manifest, "Observer configuration changed")
    else:
        atomic(manifest_path, manifest)
    atomic(output / "OWNER.json", {
        "pid": os.getpid(), "started_at": now(), "argv": sys.argv,
        "watch": args.watch, "read_only_remote": True, "native_launch_authority": False,
    })
    last_event, settled_reads = None, 0
    while True:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=480, check=True)
            snapshot = json.loads(result.stdout)
            check(snapshot.get("audit_source_sha256") == manifest["source_sha256"],
                  "Remote admission implementation differs from frozen local audit source")
            summary = concise(snapshot)
            atomic(output / "SOURCE_BOUND_TRAIN.json", snapshot)
            atomic(output / "LATEST.json", summary)
            material = {key: value for key, value in summary.items() if key != "at"}
            fingerprint = text_sha(json.dumps(material, sort_keys=True))
            if fingerprint != last_event:
                with (output / "EVENTS.jsonl").open("a") as stream:
                    stream.write(json.dumps(summary, sort_keys=True) + "\n")
                last_event = fingerprint
                print(json.dumps({"at": summary["at"], "gates": {
                    track: data["gate"] for track, data in summary["tracks"].items()
                }}), flush=True)
            alive = any(data["controller"]["alive"] for data in snapshot["tracks"].values())
            alive |= any(r.get("owner", {}).get("alive") for r in snapshot["recoveries"])
            alive |= any(row.get("active_owner", {}).get("alive")
                         for data in snapshot["tracks"].values() for row in data["rows"])
            settled_reads = 0 if alive else settled_reads + 1
            state = {
                "status": "OBSERVING_LIVE_OWNERS" if alive else "SOURCE_BOUND_OWNERS_SETTLED",
                "at": now(), "pid": os.getpid(), "next_check_seconds": args.interval,
                "latest_snapshot_sha256": digest(output / "SOURCE_BOUND_TRAIN.json"),
                "complete_valid_capable_parents": snapshot["complete_valid_capable_parents"],
                "next_trigger": summary["next_trigger"],
                "additional_actor_calls": 0, "additional_native_verifier_calls": 0,
            }
            atomic(output / "STATE.json", state)
            if snapshot["complete_valid_capable_parents"]:
                atomic(output / "CAPABLE_PARENT_TRIGGER.json", {
                    "at": now(), "parents": snapshot["complete_valid_capable_parents"],
                    "source_bound_train": str(output / "SOURCE_BOUND_TRAIN.json"),
                    "sha256": state["latest_snapshot_sha256"], "automatic_rsi_execution": False,
                })
            if not args.watch or settled_reads >= 2:
                return 0
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failure = {
                "at": now(), "type": type(error).__name__, "error": str(error),
                "stderr": getattr(error, "stderr", None), "remote_actions_taken": 0,
            }
            atomic(output / "COLLECTION_FAILURE.json", failure)
            with (output / "FAILURES.jsonl").open("a") as stream:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
            atomic(output / "STATE.json", {
                "status": "READ_ONLY_COLLECTION_BLOCKED", "at": now(), "pid": os.getpid(),
                "next_check_seconds": args.interval, "failure": failure,
            })
            if not args.watch:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
