"""Quality-guarded TRAIN instruction search after rejected context compaction."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phiagent.harness import industrial_autosota as compaction
from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition
from phiagent.harness.industrial_capacity import validate_workers
from phiagent.harness.industrial_coding_confirmation import text_sha256, verify_measured_row

SCHEMA = "industrial-autosota-instruction-search/v1"
METHOD_IDENTITY = (
    "AutoSOTA-workflow-adapted TRAIN instruction refinement; official CLI not executed"
)
SETTINGS = {
    "phase": "train",
    "model": "deepseek-flash",
    "method": "chisel_codex",
    "context_compaction_threshold": 996147,
    "optimization_iterations": 2,
    "maximum_workers": 2,
    "maximum_instruction_characters": 900,
    "maximum_instruction_words": 140,
    "execution_seeds": [20260928],
    "cumulative_token_limit": None,
    "token_usage_mode": "unlimited",
    "validation_access": False,
    "test_access": False,
    "official_autosota_cli_executed": False,
    "cot_jev_core_claim": False,
    "percentage_target_is_not_an_early_stop": True,
}
OBJECTIVE = (
    "Refine the supplied reusable coding instruction on the complete TRAIN population. "
    "Preserve or improve the strongest TRAIN instruction's native task success first; "
    "minimize complete-task input-plus-output actor tokens second. "
    "Keep the native Codex context capacity and compaction trigger unchanged. "
    "Reduce unnecessary exploration, repeated large tool output and repeated work, "
    "while retaining every requested artifact and every required correctness check. "
    "Do not shorten a task by skipping requirements, weakening verification, changing "
    "the model, tools, reasoning effort, time limits or output limits. "
    "Use evidence from at least two distinct supplied TRAIN tasks. "
    "Do not hardcode task IDs, answers, hidden tests or task-specific solutions. "
    "Return one complete replacement instruction, not an appended second instruction. "
    "Validation and TEST are unavailable; their outcomes must never enter proposals."
)


def verified_reference_capsule(path: Path, checksum: str) -> tuple[dict, dict]:
    def digest(target):
        value = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    if digest(path) != checksum:
        raise ValueError("The frozen native reference capsule changed")
    value = json.loads(path.read_bytes())
    evidence = value.get("source_evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("A frozen reference requires its original source evidence hashes")
    aliases, verified_bytes = 0, 0
    for name, entry in evidence.items():
        source = Path(name)
        if (
            not source.is_absolute() or ".." in source.parts
            or entry.get("path") != name or digest(source) != entry.get("sha256")
        ):
            raise ValueError(f"Frozen native source bytes changed: {name}")
        aliases += int(source.resolve() != source)
        verified_bytes += source.stat().st_size
    return value, {
        "capsule_sha256": checksum, "verified_source_references": len(evidence),
        "verified_bytes": verified_bytes, "archived_alias_sources": aliases,
        "identity_policy": "Original receipt SHA256, including approved byte-preserving archival aliases",
        "unknown_usage_replaced_with_zero": False, "new_actor_or_scorer_calls": 0,
    }


def validate_contract(contract: Mapping[str, Any]) -> None:
    fixed = {
        "schema": SCHEMA,
        "method_identity": METHOD_IDENTITY,
        "reference": compaction.REFERENCE,
        "rubric": compaction.RUBRIC,
        **{key: value for key, value in SETTINGS.items() if key != "maximum_workers"},
    }
    if any(key not in contract or canonical_sha256(contract[key]) != canonical_sha256(value)
           for key, value in fixed.items()):
        raise ValueError("Instruction-search identity, quality guard or TRAIN boundary changed")
    validate_workers(contract.get("maximum_workers"))
    validate_partition(contract["partition"])
    if (
        contract["partition"]["benchmark"] != "terminal-bench-2.0"
        or contract["partition"]["counts"] != {"train": 18, "validation": 18, "test": 53}
        or contract["partition"]["independent_unit"] != "task"
    ):
        raise ValueError("Require the complete task-grouped 18/18/53 partition")
    for key in ("partition_sha256", "shared_prompt_sha256", "incumbent_instruction_sha256"):
        value = contract.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"A frozen SHA256 is required: {key}")
    references = contract["reference_summaries"]
    for label in ("native_parent", "train_incumbent", "rejected_compact_32k"):
        compaction.validate_summary(references[label])
        if set(references[label]["task_scores"]) != set(contract["partition"]["splits"]["train"]):
            raise ValueError("References must cover the identical complete TRAIN population")
    if (
        references["native_parent"]["successes"] < 9
        or references["train_incumbent"]["successes"] < references["native_parent"]["successes"]
        or compaction.evaluate_rubric(
            references["native_parent"], references["train_incumbent"],
            references["rejected_compact_32k"],
        )["quality_preserved"] is not False
    ):
        raise ValueError("This successor requires a capable parent and a rejected quality trade-off")


def validate_proposal(contract: Mapping[str, Any], proposal: Mapping[str, Any]) -> str:
    validate_contract(contract)
    instruction, support = proposal.get("instruction"), proposal.get("supporting_task_ids")
    if (
        proposal.get("execution_status") != "completed"
        or not isinstance(instruction, str) or not instruction.strip()
        or len(instruction.strip()) > contract["maximum_instruction_characters"]
        or len(instruction.strip().split()) > contract["maximum_instruction_words"]
        or "\0" in instruction or "[CHISEL task-local residual]" in instruction
        or not isinstance(support, list) or any(not isinstance(task, str) for task in support)
        or len(set(support)) < 2 or len(set(support)) != len(support)
        or not set(support) <= set(contract["partition"]["splits"]["train"])
    ):
        raise ValueError("Require one sparse replacement supported by two distinct TRAIN tasks")
    return instruction.strip()


def optimizer_request(
    contract: Mapping[str, Any], *, run_id: str, tasks: Sequence[Mapping[str, Any]],
    parent_results: Mapping[str, Any], incumbent: Mapping[str, Any],
    incumbent_results: Mapping[str, Any], completed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_contract(contract)
    expected = set(contract["partition"]["splits"]["train"])
    prompt_hashes = {
        row["id"]: row["public_task_sha256"] for row in contract["partition"]["records"]
    }
    if (
        not isinstance(run_id, str) or not run_id
        or len(tasks) != 18 or {task["task_id"] for task in tasks} != expected
        or any(task.get("unit_id") != task["task_id"]
               or not isinstance(task.get("public_prompt"), str)
               or text_sha256(task["public_prompt"]) != prompt_hashes[task["task_id"]]
               for task in tasks)
        or set(parent_results) != expected or set(incumbent_results) != expected
        or text_sha256(incumbent["instruction"]) != contract["incumbent_instruction_sha256"]
        or len(completed) >= contract["optimization_iterations"]
    ):
        raise ValueError("Optimizer input must retain the frozen complete TRAIN references")
    def verify_results(summary, results):
        for task in expected:
            verify_measured_row({
                "status": "SCORED", "score": results[task]["native_score"],
                "usage": results[task]["usage"], "executed_native_tests_verified": True,
            })
        if (
            any(results[task]["execution_status"] != "completed"
                or results[task]["native_score"] != summary["task_scores"][task]
                for task in expected)
            or sum(results[task]["usage"]["total_tokens"] for task in expected)
            != summary["actor_usage"]["total_tokens"]
        ):
            raise ValueError("Optimizer reference outcomes or costs differ from native receipts")

    for label, values in (
        ("native_parent", parent_results), ("train_incumbent", incumbent_results),
    ):
        verify_results(contract["reference_summaries"][label], values)
    candidates = {incumbent["id"]: deepcopy(dict(incumbent))}
    results = {incumbent["id"]: incumbent_results}
    for round_result in completed:
        candidate = round_result["candidate"]
        compaction.validate_summary(round_result["summary"])
        if set(round_result["results"]) != expected:
            raise ValueError("Only complete TRAIN rounds may supply recursive feedback")
        verify_results(round_result["summary"], round_result["results"])
        if candidate["id"] != text_sha256(candidate["instruction"]):
            raise ValueError("A previous instruction changed after its native execution")
        candidates[candidate["id"]] = deepcopy(candidate)
        results[candidate["id"]] = round_result["results"]
    target = {
        "rubric": contract["rubric"],
        "reference_summaries": {
            label: {key: summary[key] for key in ("successes", "actor_usage")}
            for label, summary in contract["reference_summaries"].items()
        },
        "rejected_compaction_diagnosis": (
            "The completed 32Ki round reduced tokens but failed the quality guard. "
            "This revision preserves native context management and refines the instruction."
        ),
    }
    request = {
        "schema": SCHEMA, "run_id": run_id, "phase": "train", "iteration": len(completed),
        "identity": {"model": "deepseek-flash", "runtime": "native_responses"},
        "objective": OBJECTIVE + "\nFrozen objective and complete TRAIN diagnosis:\n"
        + json.dumps(target, sort_keys=True),
        "native_metric": "official_terminal_bench_binary_reward",
        "limits": {
            "max_residuals_per_task": 1,
            "max_characters": contract["maximum_instruction_characters"],
            "max_words": contract["maximum_instruction_words"],
            "optimization_iterations": contract["optimization_iterations"],
            "cumulative_token_limit": None,
        },
        "training_evidence": [
            {"task": deepcopy(dict(task)), "seed": 20260928,
             "baseline": deepcopy(parent_results[task["task_id"]]),
             "candidates": {key: deepcopy(values[task["task_id"]])
                            for key, values in sorted(results.items())}}
            for task in tasks
        ],
        "previous_candidates": [candidates[key] for key in sorted(candidates)],
    }
    request["request_id"] = canonical_sha256(request)
    return request


def instruction_command(command: list[str], instruction_file: str) -> list[str]:
    if (
        command.count("--method") != 1 or command.count("--phase") != 1
        or any(flag in command for flag in (
            "--residual-file", "--context-compaction-threshold",
            "--confirmation-contract", "--autosota-contract", "--freeze",
        ))
        or command.index("--method") + 1 >= len(command)
        or command.index("--phase") + 1 >= len(command)
        or command[command.index("--method") + 1] != "codex"
        or command[command.index("--phase") + 1] != "train"
        or not isinstance(instruction_file, str) or not instruction_file.startswith("/")
        or ".." in instruction_file.split("/") or "\0" in instruction_file
    ):
        raise ValueError("Refine only a clean native TRAIN command; preserve context and held-out gates")
    result = list(command)
    result[result.index("--method") + 1] = "chisel_codex"
    return [*result, "--residual-file", instruction_file]


def rejected_round_history(evidence: Mapping[str, Any]) -> dict[str, Any]:
    summary, rows = evidence["summary"], evidence["rows"]
    compaction.validate_summary(summary)
    if (
        evidence.get("phase") != "train" or evidence.get("test_access") is not False
        or evidence.get("additional_actor_calls") != 0
        or len(rows) != 18 or {row["task_id"] for row in rows} != set(summary["task_scores"])
        or len({row["original_attempt"] for row in rows}) != 18
    ):
        raise ValueError("Inherited compaction costs require the complete18 distinct paid TRAIN attempts")
    ledger, observations = [], {}
    for row in rows:
        verify_measured_row(row)
        if (
            row["score"] != summary["task_scores"][row["task_id"]]
            or row["seed"] != 20260928 or not isinstance(row["original_attempt"], str)
            or not row["original_attempt"].startswith("/")
        ):
            raise ValueError("Inherited TRAIN attempt identity or score changed")
        key = canonical_sha256({
            "kind": "inherited_compact_32k", "attempt": row["original_attempt"],
            "task_id": row["task_id"], "seed": row["seed"],
        })
        ledger.append({
            "request_id": key, "kind": "executor", "phase": "train",
            "arm": "inherited_compact_32k", "task_id": row["task_id"], "seed": row["seed"],
            "historical_outcome_reused": True, "new_actor_calls": 0,
        })
        observations[key] = {
            "execution_status": "completed", "native_score": row["score"],
            "usage": deepcopy(row["usage"]),
            "provenance": {"original_attempt": row["original_attempt"], "cost_only_history": True},
        }
    for token in ("input_tokens", "output_tokens", "total_tokens"):
        if sum(row["usage"][token] for row in rows) != summary["actor_usage"][token]:
            raise ValueError("Inherited compaction task costs and native summary disagree")
    return {"ledger": ledger, "observations": observations}
