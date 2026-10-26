#!/usr/bin/env python3
"""Verify complete collected Terminal TEST receipts; never run an evaluator or model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phiagent.harness.figure3_repetitions import test_seeds

ARMS = ("native", "cot_jev")
FIELDS = ("input_tokens", "output_tokens", "total_tokens")
PARTITION_SHA256 = "3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40"
PROMPT_SHA256 = "f2755a7b715e692b6fe794cdf7b316dfb58b301759dade0bf3f67c2ae02e57e5"
TEST_SEEDS = (2026092801, 2026092802, 2026092803, 2026092804, 2026092805)
HARNESS_REVISION = "4878cdabd87d4041bdaff61d04c966883b9fd07a"


def verified_bytes(reference: dict) -> bytes:
    path = Path(reference["path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"Evidence receipt changed: {path}")
    return payload


def verified_json(reference: dict) -> dict:
    return json.loads(verified_bytes(reference))


def counters(value: dict) -> bool:
    return (
        isinstance(value, dict)
        and all(type(value.get(key)) is int and value[key] >= 0 for key in FIELDS)
        and value["input_tokens"] + value["output_tokens"] == value["total_tokens"]
    )


def matched_arms(data: dict) -> tuple[str, ...]:
    flag = data.get("include_cot_only_control", False)
    if type(flag) is not bool:
        raise ValueError("Declare the prospective matched CoT-only control")
    return ("native", "cot_only", "cot_jev") if flag else ARMS


def verified_bank_author_usage(data: dict, read_reference) -> dict | None:
    reference = data.get("bank_author_receipt")
    if reference is None:
        return None
    value = read_reference(reference)
    usage = value.get("usage", {})
    if (value.get("usage_complete") is not True or value.get("closed") is not True
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in (
                "inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens"))):
        raise ValueError("Bank authorship requires its closed actual provider usage")
    result = {"input_tokens": usage["inputTokens"] + usage["cacheReadTokens"] + usage["cacheWriteTokens"],
              "output_tokens": usage["outputTokens"], "total_tokens": usage["totalTokens"]}
    if not counters(result):
        raise ValueError("Bank-author cached and uncached provider buckets do not reconcile")
    return result


def population(
    data: dict, partition: dict, read_reference=verified_json, verify_source=verified_bytes,
) -> dict:
    phase = data["phase"]
    seeds = [20260928] if phase == "validation" else test_seeds(data, TEST_SEEDS)
    count, repeats = (18, 1) if phase == "validation" else (53, len(seeds))
    task_ids = partition["splits"][phase]
    if (
        len(task_ids) != count or len(set(task_ids)) != count
        or data.get("schema") != f"figure3-deepseek-terminal-{phase}/v1"
        or data.get("benchmark") != "terminal-bench-2.0"
        or data.get("model") != "deepseek-flash"
        or data.get("split_seed") != 20260928
        or data.get("shared_prompt_sha256") != PROMPT_SHA256
        or data.get("cumulative_token_limit") is not None
        or data.get("token_usage_mode") != "unlimited"
        or data.get("native_limits_unchanged") is not True
        or set(data.get("arms", {})) != set(matched_arms(data))
        or data.get("execution_seeds") != (
            seeds
        )
        or data.get("provider_sampling_seed_supported") is not False
    ):
        raise ValueError("Preserve the matched backbone, native limits and complete protected partition")
    expected = {(task, repeat) for task in task_ids for repeat in range(repeats)}
    summaries, call_ids, attempt_ids = {}, [], []
    bank_author_usage = verified_bank_author_usage(data, read_reference)
    decision_ids = []
    fast_proofs = 0
    for arm in matched_arms(data):
        rows = data["arms"][arm]
        observed = [(item["task_id"], item["repetition"]) for item in rows]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError(f"{arm}: missing, duplicate or out-of-population protected result")
        total = {field: 0 for field in FIELDS}
        successes = 0
        for item in rows:
            row = read_reference(item["result"])
            cost = row.get("cost", {})
            usage = cost.get("usage")
            if (
                row.get("phase") != phase or row.get("task_id") != item["task_id"]
                or row.get("arm") != arm
                or row.get("status") != "SCORED"
                or type(row.get("native_score")) not in (int, float)
                or row["native_score"] not in (0, 1)
                or row.get("shared_prompt_sha256") != PROMPT_SHA256
                or row.get("partition_sha256") != PARTITION_SHA256
                or cost.get("usage_status") != "MEASURED" or not counters(usage)
                or item.get("seed") != data["execution_seeds"][item["repetition"]]
            ):
                raise ValueError(f"{arm}: incomplete native score or complete-task actor usage")
            attempt_ids.append(row["attempt_id"])
            native_record = read_reference(item["native_record"])
            if (
                native_record.get("task_name") != item["task_id"]
                or (native_record.get("verifier_result") or {}).get(
                    "rewards", {}).get("reward") != row["native_score"]
                or row.get("native_result") != item["native_record"]["path"]
                or row.get("native_result_sha256") != item["native_record"]["sha256"]
            ):
                raise ValueError("The native Harbor record does not bind this scored actor attempt")
            ledger = read_reference(item["provider_calls"])
            subtotal = {field: 0 for field in FIELDS}
            if not ledger.get("calls") or ledger.get("native_actor_attempt_id") != row["attempt_id"]:
                raise ValueError("A native actor result has no actual paid-provider receipts")
            for call in ledger["calls"]:
                if (
                    call.get("model") != "deepseek-flash" or call.get("ended") is not True
                    or not counters(call.get("usage"))
                ):
                    raise ValueError("Unsettled or mismatched provider call")
                call_ids.append(call["call_id"])
                for field in FIELDS:
                    subtotal[field] += call["usage"][field]
            if subtotal != usage:
                raise ValueError("Physical provider costs must reconcile exactly once to actor totals")
            verifier = read_reference(item["verifier_acceptance"])
            tests = verifier.get("executed_native_tests", {})
            if (
                verifier.get("score") != row["native_score"]
                or type(tests.get("tests")) is not int or tests["tests"] <= 0
                or not verifier.get("source_evidence")
            ):
                raise ValueError("An actual native binary scorer execution is required")
            for reference in verifier["source_evidence"].values():
                verify_source(reference)
            reward = read_reference(tests["reward"])
            if type(reward) not in (int, float) or reward != row["native_score"]:
                raise ValueError("The actual hashed native binary reward differs from its summary")
            if arm in ("cot_only", "cot_jev"):
                audit = read_reference(item["jev_receipt_audit"])
                if (
                    audit.get("all_physical_provider_costs_reconcile") is not True
                    or audit.get("total_tokens") != usage["total_tokens"]
                ):
                    raise ValueError("Every fast decision, slow call and fallback must be charged")
                if arm == "cot_only" and (audit.get("fast_provider_calls", 0) != 0
                        or audit.get("native_fast_execution_proofs")):
                    raise ValueError("The CoT-only control must retain its disabled selector")
                proofs = audit.get("native_fast_execution_proofs", [])
                decision_ids.extend(proof["decision_id"] for proof in proofs)
                fast_proofs += len(proofs)
            successes += int(row["native_score"])
            for field in FIELDS:
                total[field] += usage[field]
        task_execution_tokens = dict(total)
        if arm != "native" and bank_author_usage is not None:
            for field in FIELDS:
                total[field] += bank_author_usage[field]
        observations = count * repeats
        summaries[arm] = {
            "observations": observations, "successes": successes,
            "native_success_percent": 100 * successes / observations,
            "actor_tokens": total,
            "task_execution_tokens": task_execution_tokens,
            "bank_author_tokens_charged_once": bank_author_usage if arm != "native" else None,
            "actor_input_plus_output_tokens_per_task": total["total_tokens"] / observations,
            "complete_failed_tasks_included": True,
        }
    if len(call_ids) != len(set(call_ids)) or len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("One actual actor attempt or physical provider call was counted twice")
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("A native fast execution proof was counted twice")
    if fast_proofs == 0:
        raise ValueError("Instruction-only labels are not native CoT/Jev execution evidence")
    baseline, candidate = summaries["native"], summaries["cot_jev"]
    controls = [summaries[arm] for arm in matched_arms(data) if arm != "cot_jev"]
    quality = all(candidate["successes"] >= control["successes"] for control in controls)
    lower = all(candidate["actor_tokens"]["total_tokens"] < control["actor_tokens"]["total_tokens"]
                for control in controls)
    return {
        "phase": phase, "complete": True, "arms": summaries,
        "native_fast_execution_proofs": fast_proofs,
        "quality_preserved": quality, "strictly_fewer_complete_actor_tokens": lower,
        "joint_quality_and_strict_cost_gate": quality and lower,
        "bank_author_receipt": data.get("bank_author_receipt"),
        "complete_cot_only_control": "cot_only" in summaries,
        "admission_30_percent_against_both_controls": ("cot_only" in summaries and quality and all(
            candidate["actor_tokens"]["total_tokens"] * 100 <= control["actor_tokens"]["total_tokens"] * 70
            for control in controls)),
        "registered_40_percent_goal_against_both_controls": ("cot_only" in summaries and quality and all(
            candidate["actor_tokens"]["total_tokens"] * 100 <= control["actor_tokens"]["total_tokens"] * 60
            for control in controls)),
    }


def validate(data: dict, read_reference=verified_json, verify_source=verified_bytes) -> dict:
    from scripts.run_figure3_deepseek_protected import reporting_policy

    policy = reporting_policy(data)
    if data.get("phase") != "test" or data.get("partition", {}).get("sha256") != PARTITION_SHA256:
        raise ValueError("Final Figure3 export requires the canonical complete TEST partition")
    partition = read_reference(data["partition"])
    if (
        partition.get("seed", partition.get("split_seed")) != 20260928
        or any(len(partition["splits"][name]) != count
               for name, count in (("train", 18), ("validation", 18), ("test", 53)))
        or len(set().union(*map(set, partition["splits"].values()))) != 89
    ):
        raise ValueError("Canonical18/18/53 groups must be disjoint and unchanged")
    train = read_reference(data["train_admission"])
    if reporting_policy(train) != policy:
        raise ValueError("Final TEST and TRAIN admission must retain the same prospective reporting policy")
    if (
        train.get("phase") != "train" or train.get("complete") is not True
        or train.get("eligible_for_protected_validation") is not True
        or train.get("native_fast_execution_proofs", 0) < 1
        or train.get("arms", {}).get("native", {}).get("successes", 0) < 9
        or any(
            train.get("arms", {}).get(arm, {}).get("observations") != 18
            or train["arms"][arm].get("complete_verified") is not True
            for arm in ("native", "cot_only", "cot_jev")
        )
    ):
        raise ValueError("No protected access before the full native quality/cost TRAIN gate")
    validation = read_reference(data["protected_validation"])
    selected = read_reference(data["candidate_freeze"])
    if reporting_policy(selected) != policy:
        raise ValueError("The pre-TEST freeze cannot change the declared outcome reporting policy")
    if (
        selected.get("frozen_before_test") is not True
        or selected.get("test_feedback_used") is not False
        or selected.get("selected_candidate_sha256") not in selected.get("train_candidate_sha256", [])
        or validation.get("selected_candidate_sha256") != selected.get("selected_candidate_sha256")
        or data.get("selected_candidate_sha256") != selected.get("selected_candidate_sha256")
    ):
        raise ValueError("Validation may select only TRAIN-frozen candidates before any TEST access")
    if matched_arms(validation) != matched_arms(data) or validation.get("bank_author_receipt") != data.get("bank_author_receipt"):
        raise ValueError("The protected CoT-only control and bank cost scope changed after validation")
    validation_result = population(validation, partition, read_reference, verify_source)
    if policy == "qualified_only" and not validation_result["joint_quality_and_strict_cost_gate"]:
        raise ValueError("The protected validation quality/cost gate did not authorize this TEST")
    result = population(data, partition, read_reference, verify_source)
    source_identity = {}
    for arm in matched_arms(data):
        references = data["arm_identity"][arm]
        config = read_reference(references["config"])
        source_hashes = read_reference(references["source_manifest"])
        if (
            config.get("model") != "deepseek-flash"
            or config.get("shared_prompt_sha256") != PROMPT_SHA256
            or config.get("partition_sha256") != PARTITION_SHA256
            or config.get("cumulative_token_limit") is not None
            or config.get("token_usage_mode") != "unlimited"
            or config.get("deepseek", {}).get("official_revision") != HARNESS_REVISION
            or not source_hashes
        ):
            raise ValueError("Final native/control configurations or pinned source identity differ")
        source_root = Path(references["config"]["path"]).parent / "source"
        for relative, checksum in source_hashes.items():
            name = Path(relative)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("A frozen source receipt escaped its own native/control source root")
            verify_source({"path": str(source_root / name), "sha256": checksum})
        source_identity[arm] = {
            "config": references["config"], "source_manifest": references["source_manifest"],
            "source_sha256": source_hashes,
            "source_availability": "available_verified",
        }
    compact_rows = []
    for arm in matched_arms(data):
        for item in data["arms"][arm]:
            row = read_reference(item["result"])
            compact_rows.append({
                "method": "baseline" if arm == "native" else "cot_only_control" if arm == "cot_only" else "selected",
                "component_identity": "native" if arm == "native" else "native_cot_only" if arm == "cot_only" else "native_cot_jev",
                "task_id": item["task_id"], "seed": item["seed"], "repetition": item["repetition"],
                "native_score": row["native_score"], "attempt_id": row["attempt_id"],
                "native_record": item["native_record"],
                "actor_usage": row["cost"]["usage"], "actor_usage_source": item["result"],
                "actor_provider_receipts": item["provider_calls"],
                "native_verifier_source": item["verifier_acceptance"],
                "source_availability": {"native_record": True, "actor_usage": True},
            })
    return {
        "schema": "figure3-deepseek-terminal-publication/v1",
        "benchmark": "terminal-bench-2.0", "phase": "test", "model": "deepseek-flash",
        "split_seed": 20260928, "test_tasks": 53, "repetitions": len(test_seeds(data, TEST_SEEDS)),
        "harness_identity": {
            "name": "DeepSeek Harness", "official_tag": "dsh-v0.2.0-rc.1",
            "official_revision": HARNESS_REVISION,
        },
        "method_identity": {
            "baseline": "DeepSeek Harness", "selected": "DeepSeek Harness+CHISEL",
            "selected_component": "native CoT-authored actions plus one-token Jev selection",
            "selected_candidate_sha256": data["selected_candidate_sha256"],
            "instruction_only_or_heuristic": False,
        },
        "model_identity": {
            "model": "deepseek-flash", "revision": None,
            "revision_status": "provider_did_not_expose_immutable_revision",
            "slow_wire_api": "anthropic_messages",
            "fast_wire_api": "openai_chat_completions",
        },
        "partition": {
            "source": data["partition"], "test_ids": partition["splits"]["test"],
            "counts": {"train": 18, "validation": 18, "test": 53},
        },
        "execution_seeds": test_seeds(data, TEST_SEEDS), "provider_sampling_seed_supported": False,
        "seed_semantics": "Fixed paired repetition identity, not seeded model sampling",
        "native_control_identity": source_identity,
        "rows": compact_rows,
        "source_availability": {
            "native_test_records": "available_verified",
            "complete_actor_usage": "available_verified",
            "frozen_config_and_source": "available_verified",
            "optimizer_setup": "separately_declared; unknown historical usage remains unknown",
        },
        "publication_rows": {
            "DeepSeek Harness": result["arms"]["native"],
            "DeepSeek Harness+CHISEL": result["arms"]["cot_jev"],
        },
        **result,
        "candidate_benefit_claim_admitted": result["joint_quality_and_strict_cost_gate"],
        "null_and_negative_complete_outcomes_retained": True,
        "heldout_outcome_policy": policy,
        "positive_validation_gate": validation_result["joint_quality_and_strict_cost_gate"],
        "historical_setup_tokens_separate": 41069,
        "additional_acquisition_and_optimizer_costs": data["separate_development_accounting"],
        "cumulative_token_limit": None,
        "fast_implementation": "same-backbone one generated control token with native logprobs",
        "zero_decoding_readout": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "VERIFIED_COMPLETE_TEST", "output": str(args.output),
        "candidate_benefit_claim_admitted": result["candidate_benefit_claim_admitted"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
