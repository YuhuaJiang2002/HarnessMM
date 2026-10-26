"""Synthetic regression checks; no benchmark data, ARE or provider imports."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phiagent.harness.gaia2_execution_contract import (
    aligned_contract, judge_response_text, user_turn_limit,
)
from phiagent.harness.codex_gaia2 import NativeSurface
from scripts import advance_deepseek_gaia2_common_parent as advance
from scripts import run_deepseek_gaia2_campaign as campaign


@pytest.mark.parametrize("text", ["Evaluation: [[True]]", "Evaluation: [[False]]",
                                  "[[Success]]", "[[Failure]]", "no verdict"])
def test_judge_markers_and_literal_content_survive(text):
    assert judge_response_text(text + "<stop>ignored", ("<stop>",)) == text


def test_missing_judge_text_is_not_a_negative_score():
    with pytest.raises(ValueError, match="textual"):
        judge_response_text(None)


def test_native_declared_turns_win_and_missing_turns_can_wait_for_environment():
    assert user_turn_limit(2, 1) == 2
    assert user_turn_limit(1, None) == 1
    assert user_turn_limit(None, None) is None
    assert user_turn_limit(None, 1) == 1
    for value in (0, -1, True, "2"):
        with pytest.raises(ValueError):
            user_turn_limit(value, None)


def test_codex_surface_with_unspecified_turns_does_not_stop_after_first_reply(tmp_path):
    env = SimpleNamespace(state="running")
    surface = NativeSurface(env, None, [], {}, {}, None, tmp_path / "synthetic.jsonl")
    surface.turns = 1
    assert not surface.finished
    surface.turns = 2
    assert not surface.finished
    env.state = "stopped"
    assert surface.finished


def test_alignment_is_prospective_frozen_and_legacy_configuration_is_preserved():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/deepseek_gaia2_campaign_v1.json").read_text())
    old_fields = campaign.native_config(original)
    successor = advance.successor_config(
        original, {"http_status": 200, "provider_account_available": True},
        "synthetic-source", execution_alignment=True,
    )
    assert original["scenario_timeout_seconds"] == 1860
    assert original["native_max_turns"] == 1
    assert successor["scenario_timeout_seconds"] == 7260
    assert successor["native_max_turns"] is None
    assert successor["native_execution_contract"] == aligned_contract()
    assert successor["counts"] == {"train": 32, "validation": 32, "test": 96}
    assert successor["cumulative_token_limit"] is None
    assert campaign.native_config(successor) != old_fields
    campaign.validate_config(successor)
    successor["native_max_turns"] = 1
    with pytest.raises(campaign.GateError, match="native_max_turns"):
        campaign.validate_config(successor)
