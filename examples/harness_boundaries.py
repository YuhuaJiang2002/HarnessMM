"""Synthetic baseline and Codex import check; no provider or benchmark calls."""

import base64
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.baseline_harness import VisualBaseline
from integrations.codex_harness import CODEX_VERSION, NativeBankBoundary


def main():
    events, calls = [], []

    def synthetic_actor(messages, purpose, callback_id):
        calls.append((purpose, callback_id))
        return '{"reasoning":"Fixture center","action":{"kind":"click","x":0.5,"y":0.5}}'

    # A complete 1x1 PNG. The supplied size belongs to this fixture only.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    baseline = VisualBaseline(actor=synthetic_actor, audit=events.append)
    action = baseline.decide(png=png, image_size=(1, 1), history=[], callback_id="fixture-1")
    assert action == {"kind": "click", "x": 0, "y": 0}
    assert calls == [("native", "fixture-1")]
    print({"baseline_action": action, "fixture_calls": len(calls),
           "codex_version": CODEX_VERSION, "codex_boundary": NativeBankBoundary.__name__})


if __name__ == "__main__":
    main()
