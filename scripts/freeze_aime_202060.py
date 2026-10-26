"""Freeze all four policies from completed training and validation; never read test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.harness.aime_202060 import (  # noqa: E402
    CHISEL_VARIANTS, select_candidate, usage_sum, validate_manifest, validate_policy_freeze,
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verified(root, manifest_sha256, mode):
    proof, result = read(root / "VERIFIED.json"), read(root / "RESULT.json")
    if (
        proof["status"] != "PASS" or proof["result_sha256"] != sha(root / "RESULT.json")
        or result["manifest_sha256"] != manifest_sha256 or result["mode"] != mode
        or result.get("test_access") is not False or proof["unknown_usage_calls"]
    ):
        raise RuntimeError("Unverified or wrong-population policy stage: " + str(root))
    return result


def freeze(
    output, partition, manifest_sha256, no_skill, skillopt, autosaddler, library_root,
    train_roots, validation_roots,
):
    if output.exists():
        raise FileExistsError("Policy freeze is immutable")
    manifest_path = partition / "SPLIT_MANIFEST.json"
    if sha(manifest_path) != manifest_sha256:
        raise RuntimeError("Parent split binding changed")
    manifest = read(manifest_path)
    validate_manifest(manifest)
    learned = verified(library_root, manifest_sha256, "chisel_learn")
    library_path = library_root / "POLICY_LIBRARY.json"
    if sha(library_path) != learned["library_sha256"]:
        raise RuntimeError("Training library changed")
    library = read(library_path)
    aggregate = {
        candidate: {"candidate_id": candidate, "tasks": 0, "correct": 0, "total_tokens": 0}
        for candidate in CHISEL_VARIANTS
    }
    stage_roots = [no_skill, skillopt, autosaddler, library_root, *validation_roots]
    if train_roots:
        raise ValueError("The repaired continuation reuses audited train6; do not rerun training cases")
    for roots, mode, split in ((validation_roots, "chisel_validation", "validation"),):
        coverage = {candidate: [] for candidate in CHISEL_VARIANTS}
        if len(roots) != 2 or len(set(roots)) != 2:
            raise RuntimeError("Expected the two predetermined complete candidate shards")
        for location in roots:
            result = verified(location, manifest_sha256, mode)
            if result["library_sha256"] != learned["library_sha256"]:
                raise RuntimeError("Different candidate libraries reached selection")
            for score in result["scores"]:
                name = score["candidate_id"]
                coverage[name].extend(score["ids"])
                if split == "validation":
                    for field in ("tasks", "correct", "total_tokens"):
                        aggregate[name][field] += score[field]
        for ids in coverage.values():
            if len(ids) != 6 or set(ids) != set(manifest["splits"][split]):
                raise RuntimeError("Missing, duplicated or wrong-split candidate evaluations")
    selected = select_candidate(list(aggregate.values()))
    policies = {}
    output.mkdir(parents=True)
    for method, location, mode in (
        ("no_skill", no_skill, "no_skill_validation"),
        ("skillopt", skillopt, "skillopt"), ("autosaddler", autosaddler, "autosaddler"),
    ):
        result = verified(location, manifest_sha256, mode)
        source = location / "selected_skill.md"
        if sha(source) != result["policy_sha256"]:
            raise RuntimeError("Selected official skill bytes changed")
        if method != "no_skill" and result.get("official_rounds") != 4:
            raise RuntimeError("Incomplete official optimization rounds")
        target = output / (method + "_policy.md")
        target.write_bytes(source.read_bytes())
        policies[method] = {
            "status": "PASS", "policy_sha256": sha(target), "path": str(target),
            "official_rounds": 4 if method != "no_skill" else 0,
            "training_ids": manifest["splits"]["train"],
            "selected_seed_unchanged": result.get("selected_seed_unchanged", True),
            "source_result_sha256": sha(location / "RESULT.json"),
            "source_verified_sha256": sha(location / "VERIFIED.json"),
            "optimizer": "Pinned official engine" if method != "no_skill" else "No optimizer",
        }
    chisel = {
        "selected_variant": selected["candidate_id"], "library": library,
        "library_sha256": learned["library_sha256"],
        "validation_scores": list(aggregate.values()), "selection_rule": library["selection_rule"],
    }
    chisel_path = output / "chisel_policy.json"
    write(chisel_path, chisel)
    policies["chisel"] = {
        "status": "PASS", "policy_sha256": sha(chisel_path), "path": str(chisel_path),
        "training_ids": manifest["splits"]["train"], "selected_variant": selected["candidate_id"],
        "training_only_rule_generation": True, "validation_trace_feedback": False,
    }
    for binding in library["training_baseline_bindings"]:
        location = Path(binding["root"])
        verified(location, manifest_sha256, "train_baseline")
        stage_roots.append(location)
    baseline_ids = [
        item_id for binding in library["training_baseline_bindings"]
        for item_id in read(Path(binding["root"]) / "RESULT.json")["assigned_ids"]
    ]
    if len(baseline_ids) != 6 or set(baseline_ids) != set(manifest["splits"]["train"]):
        raise RuntimeError("Learned rules did not use the complete, disjoint audited train6")
    if len(set(stage_roots)) != len(stage_roots):
        raise RuntimeError("Duplicate paid stage would inflate optimization cost")
    costs = []
    for location in stage_roots:
        ledger_path = location / "PROVIDER_FINAL.json"
        ledger = read(ledger_path)
        if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
            raise RuntimeError("Unsettled campaign costs block complete comparison")
        total = usage_sum([row["usage"] for row in ledger["calls"]])
        if total["total_tokens"] != ledger["known_tokens"]:
            raise RuntimeError("Development cost ledger mismatch")
        costs.append({
            "root": str(location), "ledger_sha256": sha(ledger_path),
            "new_calls": len(ledger["calls"]), "usage": total,
        })
    seen_ledgers = {str(location / "PROVIDER_FINAL.json") for location in stage_roots}
    for binding in learned.get("historical_cost_bindings", []):
        ledger_path = Path(binding["ledger_path"])
        if str(ledger_path) in seen_ledgers or sha(ledger_path) != binding["sha256"]:
            raise RuntimeError("Historical learner costs are duplicated or changed")
        seen_ledgers.add(str(ledger_path))
        ledger = read(ledger_path)
        if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
            raise RuntimeError("Historical learner cost is unsettled")
        total = usage_sum([row["usage"] for row in ledger["calls"]])
        if total["total_tokens"] != ledger["known_tokens"]:
            raise RuntimeError("Historical learner native total differs")
        costs.append({
            "root": str(ledger_path.parent), "ledger_sha256": binding["sha256"],
            "new_calls": 0, "historical_calls": len(ledger["calls"]), "usage": total,
            "purpose": binding["purpose"],
        })
    value = {
        "schema": "aime-202060-all-method-freeze/v1", "manifest_sha256": manifest_sha256,
        "training_ids": manifest["splits"]["train"], "validation_ids": manifest["splits"]["validation"],
        "policies": policies, "test_opened": False, "test_tasks": 18,
        "prior_exposure": "Archived public resplit; not previously unseen data",
        "development_costs": costs, "development_usage": usage_sum([row["usage"] for row in costs]),
        "old_campaign_costs_inherited": False, "paper_admitted": False,
    }
    validate_policy_freeze(value, manifest_sha256)
    write(output / "POLICY_FREEZE.json", value)
    write(output / "VERIFIED.json", {
        "status": "PASS", "policy_freeze_sha256": sha(output / "POLICY_FREEZE.json"),
        "manifest_sha256": manifest_sha256, "sealed_test_read": False,
    })
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    for name in ("no-skill", "skillopt", "autosaddler", "library-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train-root", type=Path, action="append", default=[])
    parser.add_argument("--validation-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    if socket.gethostname() != "harnessmm-worker-2":
        raise RuntimeError("Freeze campaign policies on the authorized server")
    result = freeze(
        args.output, args.partition_root, args.manifest_sha256, args.no_skill, args.skillopt,
        args.autosaddler, args.library_root, args.train_root, args.validation_root,
    )
    print(json.dumps({
        "status": "FOUR_POLICIES_FROZEN_TEST_UNOPENED", "root": str(args.output),
        "selected_chisel": result["policies"]["chisel"]["selected_variant"],
        "development_usage": result["development_usage"],
    }))


if __name__ == "__main__":
    main()
