#!/usr/bin/env python3
"""One partition-bound Terminal-Bench 2 trial using the official Harbor Codex loop."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.harness.benchmark_partition import (  # noqa: E402
    canonical_sha256, validate_partition,
)
from phiagent.harness.codex_runtime import provider_usage  # noqa: E402
from phiagent.harness.industrial_common_prompt import prompt_manifest  # noqa: E402
from scripts.serve_codex_deepseek import read_credential, trial_token, usage_summary  # noqa: E402

HARBOR_VERSION = "0.18.0"
HARBOR_REVISION = "527d50deb63a5d279e8c20593c18a2cbc7f61f9e"
TERMINAL_REVISION = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
COMPACTED_CHISEL_METHOD = "chisel_codex_native_compaction"
CONFIRMATION_METHODS = ("codex_confirmation_compaction", "chisel_codex_confirmation_compaction")
AUTOSOTA_METHOD = "chisel_codex_autosota_compaction"
TERMINAL_BANK_METHODS = ("codex_terminal_cot_bank", "codex_terminal_cot_jev_bank")
CODEX_METHODS = (
    "codex", "chisel_codex", "codex_native_compaction", COMPACTED_CHISEL_METHOD,
    *CONFIRMATION_METHODS, AUTOSOTA_METHOD, *TERMINAL_BANK_METHODS,
)


def native_reward(result: dict) -> float:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if not isinstance(rewards, dict) or set(rewards) != {"reward"}:
        raise ValueError("Official verifier did not supply its unique native reward")
    score = rewards["reward"]
    if type(score) not in (int, float) or not math.isfinite(score) or score not in (0, 1):
        raise ValueError("Terminal-Bench native reward must be binary")
    exception = result.get("exception_info")
    if exception and exception.get("exception_type") not in (
        "AgentTimeoutError", "NonZeroAgentExitCodeError",
    ):
        raise ValueError("Infrastructure exception is not a scored agent failure")
    return float(score)


def validate_test_freeze(path: Path | None, partition_sha256: str):
    if path is None:
        raise ValueError("Test execution requires the immutable policy/source freeze record")
    freeze = json.loads(path.read_text())
    if (
        freeze.get("status") != "FROZEN_FOR_TEST"
        or freeze.get("partition_sha256") != partition_sha256
        or not freeze.get("files")
    ):
        raise ValueError("Missing or mismatched policy/source freeze")
    for record in freeze["files"]:
        actual = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError("Frozen policy or source changed before test execution")


async def settle_provider_usage(
    ledger: Path, trial_id: str, output: Path, timeout_seconds: float,
) -> dict:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Provider settlement timeout must be positive")
    started = time.monotonic()
    while True:
        usage = usage_summary(ledger, trial_id)
        elapsed = time.monotonic() - started
        write_json_atomic(output / "ACCOUNTING.json", {
            "status": "MEASURED" if usage["usage_status"] == "MEASURED" else "WAITING",
            "elapsed_seconds": elapsed, "timeout_seconds": timeout_seconds,
            "additional_model_calls": 0, "provider_usage": usage,
        })
        if usage["usage_status"] == "MEASURED":
            if usage["provider_models"] != ["deepseek-flash"]:
                raise ValueError("Settled provider model differs from the frozen backbone")
            return usage
        if elapsed >= timeout_seconds:
            raise TimeoutError("Already-dispatched provider calls remain unresolved; no actor replay")
        await asyncio.sleep(min(15, timeout_seconds - elapsed))


def context_control(
    method: str, threshold: int, residual: str, phase: str, *,
    confirmation: dict | None = None, autosota: dict | None = None,
) -> dict:
    if type(threshold) is not int or not 1024 <= threshold < 1048576:
        raise ValueError("Native active-context trigger is outside the unchanged model capacity")
    if method not in CODEX_METHODS:
        raise ValueError("Unknown Codex comparison method")
    if method in TERMINAL_BANK_METHODS:
        if threshold != 996147 or not residual or confirmation is not None or autosota is not None:
            raise ValueError("Native action banks preserve the original compaction and require their own contract")
        return {
            "mechanism": "slow-authored literal native terminal bank; paid one-token selector or matched CoT-only",
            "implementation_label": method, "active_context_trigger_tokens": threshold,
            "model_context_window": 1048576, "cumulative_token_limit": None,
            "compaction_and_recovery_costs_included": True,
        }
    if method == AUTOSOTA_METHOD:
        from phiagent.harness.industrial_autosota import validate_trial_binding

        if autosota is None or confirmation is not None:
            raise ValueError("AutoSOTA compaction requires its separate frozen TRAIN contract")
        validate_trial_binding(
            autosota["contract"], arm=autosota["arm"], method=method,
            threshold=threshold, residual=residual, phase=phase,
            task_id=autosota["task_id"], seed=autosota["seed"],
            partition_sha256=autosota["partition_sha256"],
            shared_prompt_sha256=autosota["shared_prompt_sha256"],
        )
        return {
            "mechanism": "official_codex_automatic_compaction",
            "implementation_label": "AutoSOTA-workflow-adapted TRAIN instruction/context search",
            "active_context_trigger_tokens": threshold, "model_context_window": 1048576,
            "cumulative_token_limit": None, "compaction_and_recovery_costs_included": True,
        }
    if autosota is not None:
        raise ValueError("An AutoSOTA contract cannot relabel an existing comparison method")
    if method in CONFIRMATION_METHODS:
        from phiagent.harness.industrial_coding_confirmation import validate_trial_binding

        if confirmation is None:
            raise ValueError("Prospective compaction confirmation requires its frozen contract")
        validate_trial_binding(
            confirmation["contract"], arm=confirmation["arm"], method=method,
            threshold=threshold, residual=residual, phase=phase,
            task_id=confirmation["task_id"], seed=confirmation["seed"],
            partition_sha256=confirmation["partition_sha256"],
            shared_prompt_sha256=confirmation["shared_prompt_sha256"],
        )
        return {
            "mechanism": "official_codex_automatic_compaction",
            "implementation_label": "source-frozen industrial confirmation context arm",
            "active_context_trigger_tokens": threshold, "model_context_window": 1048576,
            "cumulative_token_limit": None, "compaction_and_recovery_costs_included": True,
        }
    if method in ("codex", "chisel_codex") and threshold != 996147:
        raise ValueError("Changed compaction triggers require an explicit native-compaction control")
    if method == "codex_native_compaction" and (residual or phase != "train"):
        raise ValueError("The native-compaction control is TRAIN-only and has no learned residual")
    if method == COMPACTED_CHISEL_METHOD and (
        threshold != 65536 or not isinstance(residual, str) or not residual or phase != "train"
    ):
        raise ValueError("The instruction/compaction study requires a frozen TRAIN instruction")
    label = "unchanged native compaction setting"
    if method == "codex_native_compaction":
        label = "native compaction control, not CHISEL"
    elif method == COMPACTED_CHISEL_METHOD:
        label = "frozen CHISEL instruction + official Codex compaction; TRAIN component study"
    return {
        "mechanism": "official_codex_automatic_compaction",
        "implementation_label": label,
        "active_context_trigger_tokens": threshold,
        "model_context_window": 1048576,
        "cumulative_token_limit": None,
        "compaction_and_recovery_costs_included": True,
    }


def executed_native_verdict(result: dict, directory: Path) -> dict:
    from scripts.audit_industrial_capability_recovery import scorer_receipts

    score = native_reward(result)
    evidence = {}
    tests = scorer_receipts(directory, result, score, evidence)
    return {"score": score, "executed_native_tests": tests, "source_evidence": evidence}


async def execute(args) -> dict:
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial, TrialEvent

    if importlib.metadata.version("harbor") != HARBOR_VERSION:
        raise ValueError("The pinned official Harbor runtime is required")
    partition_bytes = args.partition.read_bytes()
    partition_hash = hashlib.sha256(partition_bytes).hexdigest()
    manifest = json.loads(partition_bytes)
    validate_partition(manifest)
    if manifest["benchmark"] != "terminal-bench-2.0":
        raise ValueError("Wrong benchmark partition")
    task_id = args.task.name
    if task_id not in manifest["splits"][args.phase]:
        raise ValueError("Task is outside the frozen phase population")
    if args.phase == "test":
        validate_test_freeze(args.freeze, partition_hash)
    expected = next(row["payload_sha256"] for row in manifest["records"] if row["id"] == task_id)
    files = {
        path.relative_to(args.task).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.task.rglob("*")) if path.is_file()
    }
    if canonical_sha256(files) != expected:
        raise ValueError("Official task source differs from the frozen population")
    residual = args.residual_file.read_bytes().decode("utf-8") if args.residual_file else ""
    if residual and args.method == "codex":
        raise ValueError("A residual cannot be attributed to the plain Codex baseline")
    shared_version = getattr(args, "common_prompt_version", None)
    shared_hash = getattr(args, "common_prompt_sha256", None)
    shared = None
    if shared_version is not None or shared_hash is not None:
        shared = prompt_manifest(manifest["benchmark"], shared_version, shared_hash)
    confirmation = None
    contract_path = getattr(args, "confirmation_contract", None)
    if contract_path is not None:
        from phiagent.harness.industrial_coding_confirmation import validate_trial_binding

        checksum = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        if checksum != args.confirmation_contract_sha256 or shared is None:
            raise ValueError("Confirmation source contract or shared parent changed")
        confirmation = {
            "contract": json.loads(contract_path.read_text()), "arm": args.confirmation_arm,
            "task_id": task_id, "seed": args.repetition_seed,
            "partition_sha256": partition_hash, "shared_prompt_sha256": shared["sha256"],
        }
        validate_trial_binding(
            confirmation["contract"], arm=args.confirmation_arm, method=args.method,
            threshold=args.context_compaction_threshold, residual=residual, phase=args.phase,
            task_id=task_id, seed=args.repetition_seed, partition_sha256=partition_hash,
            shared_prompt_sha256=shared["sha256"],
        )
    autosota = None
    autosota_path = getattr(args, "autosota_contract", None)
    if autosota_path is not None:
        if contract_path is not None or shared is None:
            raise ValueError("AutoSOTA search must retain its own contract and shared native prompt")
        checksum = hashlib.sha256(autosota_path.read_bytes()).hexdigest()
        if checksum != args.autosota_contract_sha256:
            raise ValueError("Frozen AutoSOTA source contract changed")
        autosota = {
            "contract": json.loads(autosota_path.read_text()), "arm": args.autosota_arm,
            "task_id": task_id, "seed": args.repetition_seed,
            "partition_sha256": partition_hash, "shared_prompt_sha256": shared["sha256"],
        }
    compaction = context_control(
        args.method, getattr(args, "context_compaction_threshold", 996147), residual, args.phase,
        confirmation=confirmation, autosota=autosota,
    )
    bank_path = getattr(args, "terminal_bank_contract", None)
    bank_binding = None
    if args.method in TERMINAL_BANK_METHODS:
        from phiagent.harness.codex_terminal_bank_contract import validate_trial_contract

        if bank_path is None or shared is None or hashlib.sha256(bank_path.read_bytes()).hexdigest() != args.terminal_bank_contract_sha256:
            raise ValueError("Native terminal bank requires its source-bound matched contract")
        bank_binding = validate_trial_contract(
            json.loads(bank_path.read_text()), arm=args.terminal_bank_arm, method=args.method,
            residual=residual, phase=args.phase, task_id=task_id, seed=args.repetition_seed,
            partition_sha256=partition_hash, shared_prompt_sha256=shared["sha256"],
        )
    elif bank_path is not None:
        raise ValueError("A terminal bank cannot relabel another frozen method")
    require_executed_tests = shared is not None or args.method == "codex_native_compaction"
    verifier_environment = (
        json.loads(args.verifier_environment.read_text()) if args.verifier_environment else {}
    )
    if not isinstance(verifier_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in verifier_environment.items()
    ):
        raise ValueError("Verifier environment must be an explicit string mapping")
    sam_recovery = args.sam_verifier_recovery if task_id == "sam-cell-seg" else None
    if sam_recovery:
        from scripts.recover_codex_sam_verifier import (
            DEFAULT_DEPENDENCY_CACHE_URL, dependency_env, verified_cache,
        )

        _, prepared = verified_cache(sam_recovery)
        verifier_environment = {
            **verifier_environment,
            **dependency_env(prepared.get("dependency_cache_url", DEFAULT_DEPENDENCY_CACHE_URL)),
            "BASH_ENV": "/opt/sam-recovery/cache.env", "UV_OFFLINE": "1",
        }
    native_cache = getattr(args, "native_verifier_cache", None)
    if native_cache:
        from scripts.recover_terminal_verifier import verified_cache

        prepared = verified_cache(native_cache, args.task)
        verifier_environment = {
            **verifier_environment, **prepared["verifier_environment"], "UV_OFFLINE": "1",
        }
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "task": {"path": str(args.task), "source": "terminal-bench@2.0"},
        "trial_name": args.trial_id,
        "trials_dir": str(args.output / "native"),
        "timeout_multiplier": 1.0,
        "agent": {
            "import_path": "scripts.codex_terminal_agent:MeteredCodex",
            "model_name": "deepseek-flash",
            "kwargs": {
                "version": "0.142.5", "reasoning_effort": "high", "web_search": "disabled",
                "provider_url": args.provider_url, "trial_id": args.trial_id,
                "relay_secret_file": str(args.relay_secret),
                "codex_binary": str(args.codex_binary), "residual": residual,
                "context_compaction_threshold": compaction["active_context_trigger_tokens"],
            },
        },
        "environment": {
            "type": "docker", "force_build": False, "delete": False,
            "cpu_enforcement_policy": "limit", "memory_enforcement_policy": "limit",
            "kwargs": {"keep_containers": True},
        },
        "verifier": {"env": verifier_environment} if verifier_environment else {},
    }
    if shared:
        config["agent"]["import_path"] = "scripts.industrial_terminal_agents:SharedPromptCodex"
        config["agent"]["kwargs"].update(
            shared_prompt_version=shared["version"], shared_prompt_sha256=shared["sha256"],
            benchmark=manifest["benchmark"],
        )
    record = {
        "status": "RUNNING", "benchmark": "terminal-bench-2.0", "task_id": task_id,
        "trial_id": args.trial_id, "phase": args.phase, "seed": args.repetition_seed,
        "split_seed": 20260928, "provider_sampling_seed_supported": False,
        "model": "deepseek-flash", "method": args.method,
        "residual_applied": bool(residual),
        "partition_sha256": partition_hash, "task_source_sha256": expected,
        "residual_sha256": hashlib.sha256(residual.encode()).hexdigest(),
        "shared_prompt": shared,
        "context_control": compaction,
        "terminal_bank": {
            **bank_binding, "contract": str(bank_path),
            "contract_sha256": args.terminal_bank_contract_sha256,
        } if bank_binding is not None else None,
        "confirmation": {
            "contract": str(contract_path), "contract_sha256": args.confirmation_contract_sha256,
            "arm": args.confirmation_arm,
        } if confirmation is not None else None,
        "autosota_search": {
            "contract": str(autosota_path), "contract_sha256": args.autosota_contract_sha256,
            "arm": args.autosota_arm, "official_autosota_cli_executed": False,
        } if autosota is not None else None,
        "require_executed_native_tests": require_executed_tests,
        "harbor_version": HARBOR_VERSION, "harbor_revision": HARBOR_REVISION,
        "dataset_revision": TERMINAL_REVISION,
        "native_metric": "Official Harbor task verifier binary reward",
        "protocol_label": "Custom20/20/60; native scoring; not a leaderboard submission",
        "maximum_tokens": None, "token_usage_mode": "unlimited",
        "provider_settlement_timeout_seconds": args.provider_settlement_timeout,
        "sam_verifier_recovery": str(sam_recovery) if sam_recovery else None,
        "native_verifier_cache": str(native_cache) if native_cache else None,
        "unscored_workspace_retained": True,
        "CUDA_VISIBLE_DEVICES": "", "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    write_json_atomic(args.output / "CONFIG.json", config)
    write_json_atomic(args.output / "RUN.json", record)
    write_json_atomic(args.output / "provenance.json", capture_provenance(
        ROOT, sys.argv, args.repetition_seed
    ))
    source_files = (
        "scripts/run_codex_terminal.py", "scripts/codex_terminal_agent.py",
        "phiagent/harness/codex_runtime.py", "phiagent/harness/benchmark_partition.py",
        "scripts/serve_codex_deepseek.py",
        "scripts/recover_codex_sam_verifier.py",
        "phiagent/harness/industrial_common_prompt.py",
        "scripts/industrial_terminal_agents.py",
    )
    if require_executed_tests:
        source_files += ("scripts/audit_industrial_capability_recovery.py",)
    if confirmation is not None:
        source_files += ("phiagent/harness/industrial_coding_confirmation.py",)
    if autosota is not None:
        source_files += (
            "phiagent/harness/industrial_autosota.py",
            "phiagent/harness/industrial_coding_confirmation.py",
        )
    if bank_binding is not None:
        source_files += (
            "phiagent/harness/codex_terminal_bank.py",
            "phiagent/harness/codex_terminal_bank_contract.py",
            "phiagent/harness/codex_native_app_bank.py",
            "integrations/deepseek_harness/cot_jev.mjs",
            "integrations/deepseek_harness/codex_bank_contract.mjs",
        )
    source_manifest = {}
    for name in source_files:
        target = args.output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        source_manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json_atomic(args.output / "source-snapshot" / "SHA256.json", source_manifest)
    try:
        trial = await Trial.create(TrialConfig.model_validate(config))
        from scripts.recover_terminal_verifier import record_owned_container, remove_scored_container

        async def retain_native_identity(event):
            if event.task_name != task_id:
                raise ValueError("Native container hook received a different task")
            await record_owned_container(trial.agent_environment, args.output)

        trial.add_hook(TrialEvent.AGENT_START, retain_native_identity)
        if sam_recovery:
            from scripts.recover_codex_sam_verifier import prepare_verifier_cache

            async def prepare_sam_verifier(event):
                if event.task_name != "sam-cell-seg":
                    raise ValueError("SAM cache hook received a different native task")
                await prepare_verifier_cache(trial.agent_environment, sam_recovery, args.output)

            trial.add_hook(TrialEvent.VERIFICATION_START, prepare_sam_verifier)
        if native_cache:
            from scripts.recover_terminal_verifier import install_cache

            async def prepare_native_verifier(event):
                if event.task_name != task_id:
                    raise ValueError("Verifier cache hook received a different native task")
                await install_cache(
                    trial.agent_environment, native_cache, args.task, args.output,
                )

            trial.add_hook(TrialEvent.VERIFICATION_START, prepare_native_verifier)
        result = (await trial.run()).model_dump(mode="json")
        write_json_atomic(args.output / "native_result.json", result)
        from scripts.recover_terminal_verifier import release_stopped_trial_network

        try:
            await release_stopped_trial_network(args.output)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            record["network_cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
            warnings.warn(f"Native network cleanup failed: {error}", RuntimeWarning, stacklevel=2)
        usage = (
            usage_summary(args.provider_ledger, args.trial_id) if args.provider_ledger else
            provider_usage(
                args.provider_url, args.trial_id,
                trial_token(read_credential(args.relay_secret), args.trial_id),
            )
        )
        record["provider_usage"] = usage
        score = native_reward(result)
        if require_executed_tests:
            verdict = executed_native_verdict(result, args.output / "native")
            write_json_atomic(args.output / "VERIFIER_ACCEPTANCE.json", verdict)
            record["executed_native_tests_verified"] = True
        record["score"] = score
        if usage["usage_status"] != "MEASURED":
            if args.provider_ledger is None:
                raise ValueError("Complete native provider usage is required")
            record["provider_usage"] = await settle_provider_usage(
                args.provider_ledger, args.trial_id, args.output, args.provider_settlement_timeout,
            )
        record.update(status="SCORED", native_result=str(args.output / "native_result.json"))
        try:
            await remove_scored_container(args.output)
        except (OSError, RuntimeError, TimeoutError, ValueError, subprocess.SubprocessError) as error:
            record["workspace_cleanup_error"] = {
                "type": type(error).__name__, "message": str(error),
            }
            warnings.warn(f"Scored workspace cleanup failed: {error}", RuntimeWarning, stacklevel=2)
        else:
            record["unscored_workspace_retained"] = False
    except Exception as error:
        record.update(status="BLOCKED", error_type=type(error).__name__, error=str(error))
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        write_json_atomic(args.output / "RUN.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--provider-ledger", type=Path)
    parser.add_argument("--relay-secret", type=Path, required=True)
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--residual-file", type=Path)
    parser.add_argument("--common-prompt-version")
    parser.add_argument("--common-prompt-sha256")
    parser.add_argument("--verifier-environment", type=Path)
    parser.add_argument("--sam-verifier-recovery", type=Path)
    parser.add_argument("--native-verifier-cache", type=Path)
    parser.add_argument("--provider-settlement-timeout", type=float, default=600)
    parser.add_argument(
        "--method", choices=CODEX_METHODS, default="codex",
    )
    parser.add_argument("--context-compaction-threshold", type=int, default=996147)
    parser.add_argument("--confirmation-contract", type=Path)
    parser.add_argument("--confirmation-contract-sha256")
    parser.add_argument("--confirmation-arm")
    parser.add_argument("--autosota-contract", type=Path)
    parser.add_argument("--autosota-contract-sha256")
    parser.add_argument("--autosota-arm")
    parser.add_argument("--terminal-bank-contract", type=Path)
    parser.add_argument("--terminal-bank-contract-sha256")
    parser.add_argument("--terminal-bank-arm", choices=("cot_only", "cot_jev"))
    parser.add_argument("--repetition-seed", type=int, default=20260928)
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != "harnessmm-api-worker":
        raise RuntimeError("Terminal-Bench execution is authorized only on A800-4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(asyncio.run(execute(args)), sort_keys=True))


if __name__ == "__main__":
    main()
