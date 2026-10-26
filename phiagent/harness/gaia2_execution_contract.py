"""GAIA2 judge text and supervisor limits shared by prospective native arms.

ARE's actor-oriented True/False normalization breaks its case-sensitive judge
markers. Figure 2 preserves judge text; these helpers give Figure 3 the same
transport contract without changing judge prompts, votes, tools or thresholds.
"""

from __future__ import annotations

VERSION = "figure2-native-execution-alignment/v1"
WALL_TIMEOUT_SECONDS = 7260
TRANSPORT_GRACE_SECONDS = 240


def aligned_contract() -> dict:
    return {
        "version": VERSION,
        "unspecified_user_turn_limit": None,
        "outer_wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "judge_response_text": "verbatim_provider_text",
        "native_scenario_duration_unchanged": True,
        "native_judge_prompts_and_parser_unchanged": True,
    }


def validate_contract(value: object) -> dict:
    if value != aligned_contract():
        raise ValueError("GAIA2 execution contract differs from the registered Figure 2 alignment")
    return aligned_contract()


def user_turn_limit(scenario_turns: int | None, fallback: int | None) -> int | None:
    value = scenario_turns if scenario_turns is not None else fallback
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("Native user-turn limit must be a positive integer or None")
    return value


def judge_response_text(text: str, stop_sequences=()) -> str:
    if not isinstance(text, str):
        raise ValueError("Native judge response lacks textual content")
    for stop in stop_sequences:
        text = text.split(stop)[0]
    return text


def native_judge_completion(engine, messages, stop_sequences=(), **kwargs):
    """Use ARE's exact LiteLLM call, retaining text for its native judge parser.

    Imports are remote/runtime-only. The raw metering proxy still stores every
    request, response and usage receipt. No verdict is invented or case-fixed.
    """
    from are.simulation.agents.llm.litellm import litellm_engine as native

    try:
        converted = [engine._convert_message_to_litellm_format(item) for item in messages]
        config = engine.model_config
        response = native.completion(
            model=config.model_name,
            custom_llm_provider=config.provider if config.provider != "local" else None,
            messages=converted,
            api_base=config.endpoint,
            api_key=config.api_key,
            mock_response=engine.mock_response,
        )
        assert type(response) is native.ModelResponse
        assert len(response.choices) >= 1
        assert type(response.choices[0]) is native.Choices
        return judge_response_text(response.choices[0].message.content, stop_sequences), None
    except (native.AuthenticationError, native.APIError) as error:
        raise native.LLMEngineException("Auth error in litellm.") from error
