#!/usr/bin/env python3
"""Collect saved native measurements and campaign state; never execute a benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def load_receipt(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Receipt must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def controller_running(
    pid: int | None, script: str | tuple[str, ...], proc: Path = Path("/proc"),
) -> bool | None:
    if pid is None or sys.platform != "linux":
        return None
    if type(pid) is not int or pid <= 0:
        raise ValueError("A controller PID must be a positive integer")
    try:
        command = (proc / str(pid) / "cmdline").read_bytes().split(b"\0")
        stat = (proc / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return False
    names = (script,) if isinstance(script, str) else script
    return (
        stat.rsplit(")", 1)[1].split()[0] not in {"Z", "T", "t", "X", "x"}
        and any(Path(part.decode()).name in names for part in command if part)
    )


def load_vqa_final(campaign: Path, summary: dict) -> dict:
    comparison, comparison_hash = load_receipt(campaign / "final-comparison.json")
    policy, policy_hash = load_receipt(campaign / "policy.json")
    release, release_hash = load_receipt(campaign / "TEST_RELEASE.json")
    methods = {"deepseek_harness", "chisel_deepseek_harness"}
    coverage = {
        row["method"]: row for row in summary.get("coverage", []) if row.get("phase") == "test"
    }
    if (
        summary.get("state") != "COMPLETE"
        or comparison.get("schema") != "deepseek-harness-paired-comparison/v1"
        or comparison.get("benchmark") != "vqav2-val2014"
        or comparison.get("metric") != "VQAv2_official_leave_one_out_consensus_accuracy"
        or comparison.get("scored_test_cases") != 128164
        or comparison.get("independent_test_units") != 24302
        or comparison.get("repetitions") != [1]
        or comparison.get("policy_hashes_supplied") is not True
        or set(comparison.get("methods", {})) != methods
        or set(coverage) != methods
        or release.get("frozen_before_test") is not True
        or release.get("test_questions") != 128164
        or release.get("test_images") != 24302
        or release.get("partition_sha256")
        != "a8fb545b9b17a12398a1d5d572bb985c35528b5395ec04ec1709451660d4e8ba"
        or any(
            not isinstance(policy.get(name), str) or len(policy[name]) != 64
            or any(character not in "0123456789abcdef" for character in policy[name])
            for name in ("policy_sha256", "source_sha256", "baseline_policy_sha256")
        )
        or release.get("policy_sha256") != policy.get("policy_sha256")
        or release.get("source_sha256") != policy.get("source_sha256")
        or release.get("baseline_policy_sha256") != policy.get("baseline_policy_sha256")
    ):
        raise ValueError("VQA final evidence must cover the complete frozen image-grouped TEST")
    for method in methods:
        row, counts = comparison["methods"][method], coverage[method]
        if (
            row.get("expected_trials") != 128164 or row.get("received_trials") != 128164
            or row.get("valid_trials") != 128164 or row.get("score_complete") is not True
            or counts.get("questions") != 128164 or counts.get("native_scored") != 128164
            or counts.get("images") != 24302
            or type(row.get("native_score_percent")) not in (int, float)
            or not math.isfinite(row["native_score_percent"])
            or not 0 <= row["native_score_percent"] <= 100
            or row.get("unknown_usage_trials") != counts.get("unknown_usage")
            or row.get("execution_usage_complete") is not (counts.get("unknown_usage") == 0)
            or (row.get("execution_usage_complete") is not True
                and row.get("mean_execution_tokens") is not None)
        ):
            raise ValueError("VQA final coverage or unknown-cost accounting is inconsistent")
    return {
        **comparison, "source": str(campaign / "final-comparison.json"),
        "source_sha256": comparison_hash, "frozen_policy_file_sha256": policy_hash,
        "test_release_file_sha256": release_hash, "unknown_usage_is_zero": False,
        "automatic_paper_admission": False,
    }


def load_gaia_test_audit(path: Path, campaign: Path, summary: dict) -> dict:
    audit, file_digest = load_receipt(path)
    sealed = {key: value for key, value in audit.items() if key != "audit_sha256"}
    digest = hashlib.sha256(json.dumps(
        sealed, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if (
        audit.get("schema") != "deepseek-gaia2-frozen-test-audit/v1"
        or audit.get("audit_sha256") != digest
        or audit.get("status") not in ("VERIFIED_NATIVE_TEST", "VERIFIED_SCORES_COSTS_PARTIAL")
        or summary.get("final_test_complete") is not True
        or summary.get("coverage", {}).get("test") != 576
    ):
        raise ValueError("A complete, sealed native GAIA2 audit is required")
    for name in ("TEST_COMPLETE.json", "FROZEN_POLICY.json"):
        source = campaign / name
        if audit["sources"].get(str(source)) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("GAIA2 audit belongs to a different completion or policy")
    for arm in ("baseline", "chisel"):
        row = audit["methods"][arm]
        if (
            row["observations"] != 288 or row["independent_scenarios"] != 96
            or row["native_mean_score"] != summary["final_native_means"][arm]
        ):
            raise ValueError("GAIA2 audited scores disagree with the complete native summary")
    return {
        **{key: audit[key] for key in (
            "status", "methods", "score_delta", "paired_scenario_bootstrap_ci95",
            "selected_policy", "residual_applications", "actor_costs_complete",
        )},
        "source": str(path), "source_sha256": file_digest,
        "unknown_usage_is_zero": False, "automatic_paper_admission": False,
    }


def build_report(
    root: Path,
    *,
    terminal_campaign: Path | None = None,
    vqa_campaign: Path | None = None,
    gaia_campaign: Path | None = None,
    gaia_test_audit: Path | None = None,
) -> dict:
    if gaia_test_audit is not None and gaia_campaign is None:
        raise ValueError("A final GAIA2 audit requires its explicit campaign")
    report = {
        "schema": "deepseek-harness-progress/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_kind": "Native progress; final paired comparisons require a supplied verified audit",
        "model": "deepseek-flash",
        "official_harness_commit": "4878cdabd87d4041bdaff61d04c966883b9fd07a",
        "primary_chisel": "TRAIN-only learned sparse residual with cross-fitted local routing",
        "token_usage_mode": "unlimited",
        "cached_input_included_in_execution_tokens": True,
        "automatic_paper_admission": False,
        "campaigns": {},
        "training_pilots": [],
        "data_readiness": {},
    }
    for benchmark, campaign in (
        ("terminal-bench-2.0", terminal_campaign),
        ("vqav2-val2014", vqa_campaign),
        ("gaia2-mini", gaia_campaign),
    ):
        if campaign is None:
            report["campaigns"][benchmark] = {
                "status": "NO_PRIMARY_CAMPAIGN_RECEIPT_SUPPLIED",
                "final_comparison_available": False,
            }
            continue
        path = next(
            (campaign / name for name in ("SUMMARY.json", "summary.json", "heartbeat.json", "HEARTBEAT.json")
             if (campaign / name).is_file()),
            None,
        )
        if path is None:
            report["campaigns"][benchmark] = {
                "status": "WAITING_FOR_SUMMARY", "root": str(campaign),
                "final_comparison_available": False,
            }
            continue
        summary, digest = load_receipt(path)
        report["campaigns"][benchmark] = {
            "root": str(campaign), "summary": summary, "summary_sha256": digest,
            "receipt_type": path.name,
            "status": summary.get("status", summary.get("state", summary.get("phase"))),
            "controller_running": controller_running(summary.get("pid"), {
                "terminal-bench-2.0": "run_deepseek_terminal_campaign.py",
                "vqav2-val2014": (
                    "run_deepseek_vqav2_campaign.py", "continue_deepseek_vqav2_stream_recovery.py",
                ),
                "gaia2-mini": "run_deepseek_gaia2_campaign.py",
            }[benchmark]),
        }
        policy_path = campaign / "FROZEN_POLICY.json"
        if policy_path.is_file():
            policy, policy_hash = load_receipt(policy_path)
            candidate = policy["candidate"]
            report["campaigns"][benchmark]["frozen_policy"] = {
                "method": policy["method"], "policy_sha256": policy["policy_sha256"],
                "candidate_id": candidate["id"], "enabled": candidate["enabled"],
                "selected_noop": candidate["id"] == "noop",
                "source": str(policy_path), "source_sha256": policy_hash,
            }
        if benchmark == "gaia2-mini" and gaia_test_audit is not None:
            report["campaigns"][benchmark]["final_comparison"] = load_gaia_test_audit(
                gaia_test_audit, campaign, summary,
            )
            report["campaigns"][benchmark]["final_comparison_available"] = True
        if benchmark == "vqav2-val2014":
            report["campaigns"][benchmark]["final_comparison_available"] = False
            if summary.get("state") == "COMPLETE" and (campaign / "final-comparison.json").exists():
                report["campaigns"][benchmark]["final_comparison"] = load_vqa_final(campaign, summary)
                report["campaigns"][benchmark]["final_comparison_available"] = True
    if (gaia_test_audit is not None
            and "final_comparison" not in report["campaigns"]["gaia2-mini"]):
        raise ValueError("A supplied GAIA2 audit has no current complete campaign summary")
    terminal = root / "terminal-training-preflight/headless-terminal-20260929T081333Z-v2/SUMMARY.json"
    if terminal.is_file():
        value, digest = load_receipt(terminal)
        report["training_pilots"].append({
            "benchmark": "terminal-bench-2.0", "method": "deepseek_harness",
            "scope": "One frozen TRAIN task, not the complete benchmark success rate",
            "task_id": value["task_id"], "native_rewards": value["native_rewards"],
            "native_exception_type": (
                value["native_exception"].get("exception_type")
                if value.get("native_exception") else None
            ),
            "native_usage": value["cumulative_usage"],
            "source": str(terminal), "source_sha256": digest,
        })
    vqa = root / "vqav2-training-preflight/20260929T080741.722897Z-reservation/training-pilot-summary.json"
    if vqa.is_file():
        value, digest = load_receipt(vqa)
        report["training_pilots"].append({
            "benchmark": "vqav2-val2014", "method": "deepseek_harness",
            "scope": "One frozen TRAIN question, not the complete benchmark accuracy",
            "row": value["row"], "source": str(vqa), "source_sha256": digest,
        })
    gaia = root / "runs/gaia2-first-training-20260929T111945Z-v4/trial/SUMMARY.json"
    if gaia.is_file():
        value, digest = load_receipt(gaia)
        report["training_pilots"].append({
            "benchmark": "gaia2-mini", "method": "deepseek_harness",
            "scope": "One frozen TRAIN scenario, not the complete benchmark success rate",
            "task_id": value["task_id"], "native_score": value["native_score"],
            "success": value["success"], "native_exception": value["native_exception"],
            "provider_usage": value["provider_usage"],
            "source": str(gaia), "source_sha256": digest,
        })
    development = root / "vqav2-training-preflight/20260929T082754.691617Z-development/paired-training-summary.json"
    if development.is_file():
        value, digest = load_receipt(development)
        report["development_evidence"] = {
            "scope": "Typed-answer controller development; excluded from the primary residual comparison",
            "trained_skill": False, "certified": False,
            "baseline_row": value["baseline_row"], "candidate_row": value["candidate_row"],
            "new_usage": value["new_usage"], "source": str(development), "source_sha256": digest,
        }
    for label, path in (
        ("gaia2-mini", root / "GAIA_PARTITION_READY.json"),
        ("vqav2-val2014", root / "assets/vqav2/RUNTIME_READINESS.json"),
        ("terminal-bench-2.0", root / "TERMINAL_NATIVE_SOURCE_VERIFIED.json"),
    ):
        if path.is_file():
            value, digest = load_receipt(path)
            report["data_readiness"][label] = {
                "receipt": value, "source": str(path), "source_sha256": digest,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-campaign", type=Path)
    parser.add_argument("--vqa-campaign", type=Path)
    parser.add_argument("--gaia-campaign", type=Path)
    parser.add_argument("--gaia-test-audit", type=Path)
    args = parser.parse_args()
    value = build_report(
        args.root, terminal_campaign=args.terminal_campaign,
        vqa_campaign=args.vqa_campaign, gaia_campaign=args.gaia_campaign,
        gaia_test_audit=args.gaia_test_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "campaigns": {
            key: item.get("status")
            for key, item in value["campaigns"].items()
        },
        "training_pilots": len(value["training_pilots"]),
    }))


if __name__ == "__main__":
    main()
