"""Offline reproduction case; no provider or benchmark calls."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

import hashlib
from types import SimpleNamespace
from harnessmm.harness.gaia2_cot_jev import digest, parse_bank, public_state_candidates

data = json.loads((HERE / "fixture.json").read_text())
tools = {name: SimpleNamespace(inputs=inputs) for name, inputs in data["tools"].items()}
observation = json.dumps(data["public_items"])
receipt = {"log_index": 4, "native_tool_call": {
    "tool_name": "Items__list", "tool_arguments": {"offset": 0}},
    "observation": observation,
    "native_observation_sha256": hashlib.sha256(observation.encode()).hexdigest(),
    "historical_receipt_not_current_state": True}
bank = parse_bank({"jev_bank": data["bank"]}, tools,
    allow_routine_writes=True, continuation_bank=True)
latest = '{"item_id":"visible-a","details":"public detail"}'
rows, failures, proof = public_state_candidates(bank, latest, {}, tools, set(),
    native_receipts=[receipt])
assert not failures and proof["bank_binding_receipts"] == [receipt]
remaining, failures, _ = public_state_candidates(bank, latest, {}, tools,
    {digest(rows[0]["action"])}, native_receipts=[receipt])
assert not failures
# Opaque feedback cannot establish a candidate's receipt guard.
unknown, failures, unknown_proof = public_state_candidates(bank, "opaque feedback",
    {}, tools, set(), native_receipts=[receipt])
assert not unknown and not failures
summary = {"scope": "synthetic public-receipt binding",
    "candidate_item_ids": [r["action"]["action_input"]["item_id"] for r in rows],
    "remaining_item_ids": [r["action"]["action_input"]["item_id"] for r in remaining],
    "opaque_feedback_status": unknown_proof["status"]}
assert summary == json.loads((HERE / "expected.json").read_text())
print(json.dumps(summary, indent=2))
