from __future__ import annotations

import json
from pathlib import Path

from phiagent.evaluation.deployment_token_accounting import (
    ALFWORLD_SYSTEM_PROMPT,
    ActorCallRecorder,
    RecordedActorCall,
    account_actor_episode,
    build_skill_prompt,
    count_skill_artifact,
)


class CharacterOffsetTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return list(range(len(text)))

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ):
        del add_special_tokens, return_offsets_mapping
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def _call(
    tmp_path: Path,
    index: int,
    user: str,
    response: str,
    prompt_tokens: int,
) -> RecordedActorCall:
    request = {
        "messages": [
            {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
    }
    result = {
        "choices": [{"message": {"content": response}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(response),
            "total_tokens": prompt_tokens + len(response),
        },
    }
    request_path = tmp_path / f"request-{index}.json"
    response_path = tmp_path / f"response-{index}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    response_path.write_text(json.dumps(result), encoding="utf-8")
    return RecordedActorCall(
        index=index,
        request=request,
        response=result,
        request_path=request_path,
        response_path=response_path,
        elapsed_seconds=1.0,
    )


def test_episode_accounting_partitions_real_api_input_without_char_proxies(
    tmp_path: Path,
) -> None:
    tokenizer = CharacterOffsetTokenizer()
    skill_root = tmp_path / "skills"
    skill = skill_root / "navigation" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("use look", encoding="utf-8")
    skill_content = "use look"
    prompt = build_skill_prompt(skill_content)
    task = "put apple on table"
    first_user = prompt + task + "\nroom"
    first_content_tokens = len(ALFWORLD_SYSTEM_PROMPT) + len(first_user)
    response = "<action>look</action>"
    second_user = prompt + task + "\n" + response + "\nnext room"
    second_content_tokens = len(ALFWORLD_SYSTEM_PROMPT) + len(second_user)
    calls = [
        _call(tmp_path, 1, first_user, response, first_content_tokens + 3),
        _call(tmp_path, 2, second_user, response, second_content_tokens + 3),
    ]
    conversation = [
        {
            "model_response": response,
            "env_feedback": "next room",
        },
        {
            "model_response": response,
            "env_feedback": "done",
        },
    ]
    artifact = count_skill_artifact(
        skill_root,
        tokenizer,
        active_skill_content=skill_content,
    )
    result = account_actor_episode(
        tokenizer=tokenizer,
        calls=calls,
        conversation=conversation,
        skill_content=skill_content,
        task_text=task,
        static_artifact=artifact,
        actor_identity={"model": "qwen"},
    )
    assert result["actor_call_count"] == 2
    assert result["active_skill_tokens"] == 2 * len(skill_content)
    assert result["task_tokens"] == 2 * len(task)
    assert result["trajectory_history_tokens"] >= len(response)
    assert result["unknown_tokens"] == 6
    additive = (
        result["base_system_tokens"]
        + result["task_tokens"]
        + result["active_skill_tokens"]
        + result["other_harness_tokens"]
        + result["wiki_or_memory_tokens"]
        + result["tool_schema_tokens"]
        + result["trajectory_history_tokens"]
        + result["other_tokens"]
        + result["unknown_tokens"]
    )
    assert additive == result["actor_input_tokens"]
    assert result["active_harness_tokens"] == (
        result["base_system_tokens"]
        + result["active_skill_tokens"]
        + result["other_harness_tokens"]
    )
    assert result["artifact_bytes_are_deployment_cost"] is False


def test_actor_recorder_preserves_payload_and_restores_backend(tmp_path: Path) -> None:
    class Backend:
        @staticmethod
        def _post_chat_completion(payload, timeout, config):
            assert timeout == 5
            assert config == "config"
            return {
                "model": "qwen",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }

    backend = Backend()
    original = backend._post_chat_completion
    with ActorCallRecorder(backend, tmp_path) as recorder:
        recorder.start_episode("task:1", tmp_path)
        response = backend._post_chat_completion(
            {"messages": [{"role": "user", "content": "hi"}]},
            5,
            "config",
        )
        calls = recorder.finish_episode()
    assert response["usage"]["total_tokens"] == 4
    assert len(calls) == 1
    assert calls[0].request["messages"][0]["content"] == "hi"
    assert calls[0].response["model"] == "qwen"
    assert backend._post_chat_completion == original
