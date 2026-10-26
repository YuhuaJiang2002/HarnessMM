"""Pure frozen final-request and five-candidate selection contracts."""
from __future__ import annotations

import hashlib
import json

METHODS = ("no_skill", "chisel", "skillopt", "autosaddler")
MANIFEST_SHA = "e9982117d29cb83fc4d5a1699374c337c1e2a8976ba69a54a13f1ccc291e48c6"
CANDIDATE_SHA = "821257bedbb76c2890f97b208e85ee25c072b26de4149e54a0c5a57c023ad039"
PROMPT_SHA = "75fcced1215100d28f365cba032d8cf2e4e8f4542883d5e2764a0611c4af0277"
SEED = 20260926


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def final_request(model, template, question, policy):
    messages = [{"role": "user", "content": template.format(prompt=question)}]
    if policy:
        messages.insert(0, {"role": "system", "content": policy})
    return {"model": model, "messages": messages, "max_tokens": 32768,
            "temperature": 0, "seed": SEED, "chat_template_kwargs": {"enable_thinking": False}}


def select_five(rows):
    expected = {"constraint_ledger", "constraint_verified", "constraint_ledger_audited",
                "structure_gated_counting", "domain_solver_821257be"}
    if len(rows) != 5 or {row["candidate_id"] for row in rows} != expected:
        raise RuntimeError("All five registered candidate outcomes are required")
    for row in rows:
        if (row["tasks"] != 6 or type(row["correct"]) is not int or not 0 <= row["correct"] <= 6
                or type(row["total_tokens"]) is not int or row["total_tokens"] < 0):
            raise RuntimeError("Invalid complete validation scalar")
    return min(rows, key=lambda row: (-row["correct"], row["total_tokens"], row["candidate_id"]))


def test_data_gate(data, manifest):
    from phiagent.harness.benchmark_partition import canonical_sha256, validate_partition
    validate_partition(manifest)
    if manifest["benchmark"] != "aime2025" or manifest["counts"] != {
        "train": 6, "validation": 6, "test": 18,
    }:
        raise RuntimeError("Wrong canonical final partition")
    if set(data) != {"train", "development", "test"} or data["train"] or data["development"]:
        raise RuntimeError("Final execution accepts sealed test payloads only")
    if [item["id"] for item in data["test"]] != manifest["splits"]["test"]:
        raise RuntimeError("Final population must contain the exact eighteen cases")
    hashes = {item["id"]: item["payload_sha256"] for item in manifest["records"]}
    if any(canonical_sha256(item) != hashes[item["id"]] for item in data["test"]):
        raise RuntimeError("Final canonical payload changed")
