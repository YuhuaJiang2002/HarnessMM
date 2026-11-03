"""Exact public-text retention diagnostics, without latent-state or relevance claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def retention_census(
    public_history: Sequence[Mapping[str, str]],
    outgoing_messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected_ids = [row["id"] for row in public_history]
    if (len(set(expected_ids)) != len(expected_ids)
            or any(not isinstance(identity, str) or not identity for identity in expected_ids)):
        raise ValueError("public evidence identities must be unique")
    visible = {}
    contents = []
    for message in outgoing_messages:
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)
        if message.get("role") == "tool":
            identity = message.get("tool_call_id")
            if not isinstance(identity, str) or identity in visible or not isinstance(content, str):
                raise ValueError("outgoing public tool messages require unique linkage IDs and textual contents")
            visible[identity] = content
    rows = []
    for evidence in public_history:
        original = evidence["content"]
        if not isinstance(original, str):
            raise TypeError("only exact previously public text is permitted")
        current = visible.get(evidence["id"])
        status = "same_tool_message" if current == original else (
            "changed_tool_message" if current is not None else "not_a_tool_message_in_context"
        )
        rows.append({
            "id": evidence["id"], "source_sha256": text_sha256(original),
            "source_characters": len(original), "visible_tool_characters": len(current) if current is not None else 0,
            "retention": status,
            "verbatim_somewhere_in_context": not original or any(original in content for content in contents),
            "semantic_information_loss_claimed": False,
            "decision_relevance_or_task_progress_claimed": False,
        })
    return rows
