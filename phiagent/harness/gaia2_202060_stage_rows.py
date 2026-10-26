"""Typed optimizer rows: selection never carries actor text; TRAIN is fail-closed."""
from __future__ import annotations
from enum import Enum
import json

from phiagent.harness.gaia2_202060_protocol import native_population_gate


class Stage(str, Enum):
    TRAIN = "train_feedback"
    SELECTION = "selection_score_only"
    VALIDATION = "validation_score_only"


def stage_row(item, rows, usage, *, stage, public=None, trajectory=None):
    stage = Stage(stage)
    summary = native_population_gate(rows, [item["id"]])
    score = summary["successes"] / 3
    common = {"id": item["id"], "hard": score, "soft": score, "task_type": "gaia2",
              "agent_ok": True, "usage": usage, "native_repetition_scores":
              [row["success_bool"] for row in rows], "evidence_stage": stage.value}
    if stage != Stage.TRAIN:
        return {**common, "question": "Native score-only selection",
                "response": None, "response_visibility": "withheld_selection",
                "public_messages": [], "n_turns": 0, "fail_reason": "",
                "validation_trace_feedback": False}
    if not isinstance(public, list) or not public:
        raise ValueError("TRAIN requires explicit public actor evidence")
    texts, errors = [], []
    for event in public:
        if event.get("source") == "public_actor_error":
            proof = event.get("provider_error_proof", {})
            if (event.get("error_kind") != "empty_provider_completion"
                    or proof.get("verified_raw_content_is_none") is not True
                    or not proof.get("request_sha256") or not proof.get("response_sha256")):
                raise ValueError("TRAIN provider error lacks bound raw evidence")
            errors.append(event)
            texts.append("[NATIVE MODEL ERROR, NOT GENERATED TEXT]\n" + json.dumps({
                "error_kind": event["error_kind"], "provider_error_proof": proof}, sort_keys=True))
        elif event.get("source") == "public_actor_messages":
            if not isinstance(event.get("response"), str):
                raise ValueError("Unexpected None/non-text in TRAIN actor evidence")
            texts.append(event["response"])
        else:
            raise ValueError("Unknown TRAIN evidence type")
    first = next((event for event in public if event.get("source") == "public_actor_messages"), None)
    if first is None:
        raise ValueError("TRAIN has no generated actor text")
    question = "\n".join(message["content"] for message in first["messages"]
                         if message["role"] == "user" and isinstance(message["content"], str))
    return {**common, "question": question, "response": "\n\n".join(texts),
            "response_visibility": "public_training", "n_turns": len(public),
            "fail_reason": "" if score == 1 else
                f"Native training successes: {summary['successes']}/3 repetitions.",
            "trajectory": trajectory, "public_messages": public[-1].get("messages", []),
            "typed_provider_errors": errors}


def prediction_artifacts(row):
    if row["response_visibility"] == "withheld_selection":
        if row["response"] is not None or row["public_messages"]:
            raise ValueError("Selection attempted to expose actor text")
        return {"conversation": [], "trajectory": "WITHHELD_SELECTION_ONLY\n",
                "feedback": "WITHHELD_SELECTION_ONLY\n"}
    if not isinstance(row["response"], str):
        raise ValueError("TRAIN prediction must contain typed generated/error evidence")
    return {"conversation": [{"role": "user", "content": row["question"]},
                             {"type": "message", "turn": 1, "content": row["response"]}],
            "trajectory": json.dumps(row["public_messages"], ensure_ascii=False) + "\n" + row["response"],
            "feedback": row["fail_reason"]}
