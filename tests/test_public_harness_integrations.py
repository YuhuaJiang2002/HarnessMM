"""Public integration behavior with synthetic actors; no native benchmark calls."""

import base64
from copy import deepcopy
import json

import pytest

from integrations.baseline_harness import NATIVE_INSTRUCTION, VisualBaseline
from integrations.codex_harness import NativeBankBoundary, TerminalBankProvider
from phiagent.harness.codex_native_app_bank import NativeBankBoundary as OriginalBoundary
from phiagent.harness.codex_terminal_bank import TerminalBankProvider as OriginalProvider

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
ACTION = '{"reasoning":"visible center","action":{"kind":"click","x":0.5,"y":0.5}}'


def test_codex_exports_keep_original_types_and_bank_contract():
    assert NativeBankBoundary is OriginalBoundary
    assert TerminalBankProvider is OriginalProvider
    from integrations.codex_harness import ChiselConfig, CodexChiselController
    from phiagent.harness.codex_chisel import ChiselConfig as OriginalConfig
    from phiagent.harness.codex_chisel import CodexChiselController as OriginalController
    assert ChiselConfig is OriginalConfig
    assert CodexChiselController is OriginalController
    from phiagent.harness.codex_native_app_bank import POLICY, instruction
    assert instruction({**POLICY, "enabled": True}) == instruction({**POLICY, "enabled": False})


def test_baseline_sends_full_png_history_and_makes_a_new_call_for_each_boundary():
    requests, events = [], []
    def actor(messages, purpose, callback_id):
        requests.append((deepcopy(messages), purpose, callback_id))
        return ACTION
    baseline = VisualBaseline(actor=actor, audit=events.append)
    history = [{"kind": "key", "key": "Tab"}]
    for callback in ("native-0", "native-1"):
        action = baseline.decide(png=PNG, image_size=(1, 1), history=history, callback_id=callback)
        assert action == {"kind": "click", "x": 0, "y": 0}
    assert len(requests) == 2
    assert requests[0][0][0]["content"] == NATIVE_INSTRUCTION
    content = requests[0][0][1]["content"]
    assert json.loads(content[0]["text"].removeprefix("Public executed actions: ")) == history
    assert base64.b64decode(content[1]["image_url"]["url"].split(",")[1]) == PNG
    assert all(e["executed_substitutions"] == 0 for e in events)


def test_baseline_preserves_both_metered_calls_on_malformed_output():
    paid, events = [], []
    replies = iter(["malformed model response", ACTION])
    def actor(messages, purpose, callback_id):
        paid.append({"purpose": purpose, "callback_id": callback_id,
                     "input_tokens": 10, "output_tokens": 5})
        return next(replies)
    baseline = VisualBaseline(actor=actor, audit=events.append)
    baseline.decide(png=PNG, image_size=(1, 1), history=[], callback_id="step-0")
    assert [r["purpose"] for r in paid] == ["native", "paid_parse_fallback"]
    assert sum(r["input_tokens"] + r["output_tokens"] for r in paid) == 30
    assert events[0]["kind"] == "baseline/invalid-action"
    assert events[-1]["callback_id"] == "step-0-fallback"


def test_baseline_propagates_unsettled_provider_error_without_another_request():
    paid = []
    def actor(messages, purpose, callback_id):
        paid.append(callback_id)
        raise TimeoutError("unknown provider usage")
    baseline = VisualBaseline(actor=actor, audit=lambda event: None)
    with pytest.raises(TimeoutError):
        baseline.decide(png=PNG, image_size=(1, 1), history=[], callback_id="step-0")
    assert paid == ["step-0"]


def test_baseline_stops_after_one_invalid_paid_fallback():
    paid, events = [], []
    def actor(messages, purpose, callback_id):
        paid.append(callback_id)
        return '{"action":{"kind":"click","x":2.0,"y":0.5}}'
    baseline = VisualBaseline(actor=actor, audit=events.append)
    with pytest.raises(ValueError):
        baseline.decide(png=PNG, image_size=(1, 1), history=[], callback_id="step-0")
    assert paid == ["step-0", "step-0-fallback"]
    assert len(events) == 2
