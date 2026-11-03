"""Offline reproduction case; no provider or benchmark calls."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from contextlib import nullcontext
from integrations.codex_harness import NativeBankBoundary

fixture = json.loads((HERE / "fixture.json").read_text())
events, calls, packets = [], [], []
labels = iter(["A", "?"])

def execute(name, arguments):
    calls.append({"tool": name, "arguments": arguments})
    return {"content": [{"type": "text", "text": "READY"}], "isError": False}

def judge(packet):
    packets.append(packet)
    label = next(labels)
    return {"model": "deepseek-flash", "usage": {
        "prompt_tokens": 17, "completion_tokens": 1, "total_tokens": 18,
        "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 9},
        "choices": [{"message": {"content": label},
                     "logprobs": {"content": [{"token": label, "logprob": -0.1}]}}]}

boundary = NativeBankBoundary(fixture["tools"], enabled=True, judge=judge,
    execute=execute, generation=nullcontext, finished=lambda: False, audit=events.append)
boundary.register(fixture["response"], public_task=["Read only the declared paths."])
result = boundary.call("Files__cat", {"path": "receipt.txt"})
assert "CHISEL_NATIVE_ACTION_RECEIPT" in result["content"][-1]["text"]
assert "private reasoning" not in json.dumps(packets)
# After abstention the next call belongs to the original slow actor.
boundary.call("Files__cat", {"path": "next-slow.txt"})
summary = {"scope": "synthetic native-boundary replay", "calls": calls,
    "selector_calls": len(packets),
    "reused_actions": sum(e["kind"] == "codex-bank/native-action" for e in events),
    "abstained": events[-1]["decision"]["label"] == "?"}
assert summary == json.loads((HERE / "expected.json").read_text())
print(json.dumps(summary, indent=2))
