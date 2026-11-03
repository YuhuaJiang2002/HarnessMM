"""Offline reproduction case; no provider or benchmark calls."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from decimal import Decimal, ROUND_HALF_UP

def two(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

data = json.loads((HERE / "aggregate.json").read_text())
assert data["complete"] and data["expected"] == data["settled"] == 257
native_total = data["arms"]["native"]["totals"]["total_tokens"]
rows = []
for name in ("native", "cot-only", "chisel"):
    arm = data["arms"][name]
    assert arm["expected"] == arm["settled"] == data["expected"]
    assert arm["unknown_requests"] == 0
    assert 0 <= arm["successes"] <= arm["settled"]
    total = arm["totals"]["input_tokens"] + arm["totals"]["output_tokens"]
    assert total == arm["totals"]["total_tokens"]
    rows.append({"arm": name,
        "Success (%)": two(Decimal(100) * arm["successes"] / arm["settled"]),
        "Tokens/task (k)": two(Decimal(total) / arm["settled"] / 1000),
        "Token saving vs native (%)": two(Decimal(100) * (1 - Decimal(total) / native_total))})
summary = {"scope": "archived aggregate recomputation; execution costs only",
    "benchmark": data["benchmark"], "tasks_per_arm": data["expected"], "rows": rows}
assert summary == json.loads((HERE / "expected.json").read_text())
print(json.dumps(summary, indent=2))
