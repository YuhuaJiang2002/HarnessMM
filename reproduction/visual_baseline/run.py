"""Offline reproduction case; no provider or benchmark calls."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

import base64
from integrations.baseline_harness import VisualBaseline

# Complete 1x1 PNG fixture, also used in the integration contract tests.
png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
fixture = json.loads((HERE / "fixture.json").read_text())
replies = iter(fixture["actor_replies"])
calls, events = [], []

def actor(messages, purpose, callback_id):
    supplied = messages[1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(supplied.split(",")[1]) == png
    calls.append({"purpose": purpose, "callback_id": callback_id})
    return next(replies)

baseline = VisualBaseline(actor=actor, audit=events.append)
actions = [baseline.decide(png=png, image_size=(1, 1), history=[],
    callback_id="step-0")]
actions.append(baseline.decide(png=png, image_size=(1, 1), history=actions,
    callback_id="step-1"))
summary = {"scope": "synthetic screenshot boundary", "actions": actions,
    "calls": calls, "invalid_responses": sum(e["kind"] == "baseline/invalid-action"
    for e in events), "executed_substitutions": 0}
assert summary == json.loads((HERE / "expected.json").read_text())
print(json.dumps(summary, indent=2))
