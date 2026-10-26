"""Evidence-preserving projection of historical coding-tool observations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class ObservationPack:
    minimum_characters = 8000
    keep_recent = 2
    preview_characters = 240
    maximum_inline_characters = 60000
    maximum_page_characters = 8000

    def __init__(self, directory: Path) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self._records: dict[str, int] = {}
        self._stats = {
            "archived_observations": 0,
            "archived_characters": 0,
            "projected_messages": 0,
            "projected_characters_saved": 0,
            "archive_failures": 0,
            "recall_calls": 0,
        }

    def _read(self, observation_id: str) -> str:
        if (
            not isinstance(observation_id, str)
            or not re.fullmatch(r"obs_[0-9a-f]{64}", observation_id)
            or observation_id not in self._records
        ):
            raise ValueError("unknown observation_id in this session")
        target = self.directory / f"{observation_id}.txt"
        with os.fdopen(os.open(target, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != observation_id[4:]:
            raise ValueError("observation archive checksum mismatch")
        return payload.decode("utf-8")

    def _archive(self, text: str) -> str:
        payload = text.encode("utf-8")
        observation_id = "obs_" + hashlib.sha256(payload).hexdigest()
        if observation_id in self._records:
            self._read(observation_id)
            return observation_id
        target = self.directory / f"{observation_id}.txt"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        self._records[observation_id] = len(text)
        self._stats["archived_observations"] += 1
        self._stats["archived_characters"] += len(text)
        return observation_id

    def _receipt(self, text: str) -> str:
        observation_id = self._archive(text)
        return json.dumps({
            "observation_id": observation_id,
            "sha256": observation_id[4:],
            "characters": len(text),
            "head": text[:self.preview_characters],
            "tail": text[-self.preview_characters:],
            "recall": {
                "tool": "ObsRecall",
                "arguments": {
                    "observation_id": observation_id,
                    "offset": 0,
                    "limit": self.maximum_page_characters,
                },
            },
        }, ensure_ascii=False)

    def project(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        projected = [dict(message) for message in messages]
        tool_indices = [
            index for index, message in enumerate(messages) if message.get("role") == "tool"
        ]
        recent = set(tool_indices[-self.keep_recent:])
        for index in tool_indices:
            message = messages[index]
            text = message.get("content")
            if (
                not isinstance(text, str)
                or message.get("name") == "ObsRecall"
                or len(text) < self.minimum_characters
                or (index in recent and len(text) <= self.maximum_inline_characters)
            ):
                continue
            try:
                receipt = self._receipt(text)
            except (OSError, UnicodeError, ValueError):
                self._stats["archive_failures"] += 1
                continue
            if len(receipt) >= len(text):
                continue
            projected[index]["content"] = receipt
            self._stats["projected_messages"] += 1
            self._stats["projected_characters_saved"] += len(text) - len(receipt)
        return projected

    def recall(self, observation_id: str, offset: int = 0, limit: int = 8000) -> str:
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative character index")
        if type(limit) is not int or not 1 <= limit <= self.maximum_page_characters:
            raise ValueError(f"limit must be between 1 and {self.maximum_page_characters}")
        text = self._read(observation_id)
        if offset > len(text):
            raise ValueError("offset exceeds observation length")
        end = min(len(text), offset + limit)
        self._stats["recall_calls"] += 1
        return json.dumps({
            "observation_id": observation_id,
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "total_characters": len(text),
            "text": text[offset:end],
        }, ensure_ascii=False)

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "archive_directory": str(self.directory),
            "minimum_characters": self.minimum_characters,
            "keep_recent": self.keep_recent,
            "maximum_inline_characters": self.maximum_inline_characters,
            "maximum_page_characters": self.maximum_page_characters,
            "counters_unit": "characters_not_model_tokens",
            **self._stats,
        }
