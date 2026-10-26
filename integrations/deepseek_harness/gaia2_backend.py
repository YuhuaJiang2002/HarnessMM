"""Public tools, notifications and clock adapter; the official DSH loop reasons."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any

from phiagent.data_engine.provenance import write_json_atomic


HIDDEN_AUI_TOOLS = frozenset({
    "AgentUserInterface__get_last_message_from_user",
    "AgentUserInterface__get_last_message_from_agent",
    "AgentUserInterface__get_last_unread_messages",
    "AgentUserInterface__get_all_messages",
})
ANSWER_TOOL = "AgentUserInterface__send_message_to_user"
WAIT_TOOL = "SystemApp__wait_for_notification"


def public_tool_schema(tool) -> dict:
    """Encode bound native annotations/defaults, not ARE 1.1's lossy type strings."""
    from jsonschema import Draft202012Validator

    public = tool.to_open_ai()["function"]
    if hasattr(tool, "args"):
        from pydantic import ConfigDict, Field, create_model

        fields = {}
        for argument in tool.args:
            annotation = argument.type_obj
            if annotation == "Any":
                annotation = Any
            if annotation is None:
                raise ValueError(f"Native argument lacks an encodable annotation: {argument.name}")
            default = argument.default if argument.has_default else ...
            fields[argument.name] = (
                annotation, Field(default=default, description=argument.description),
            )
        model = create_model(
            "NativeToolArguments", __config__=ConfigDict(extra="forbid"), **fields,
        )
        parameters = model.model_json_schema()
        parameters.pop("title", None)
        public = {**public, "parameters": parameters}
    Draft202012Validator.check_schema(public["parameters"])
    return public


class Gaia2Backend:
    def __init__(self, environment, root: Path):
        self.environment = environment
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stopped = False
        self.turn_state = "running"
        self.answered_turns = 0
        self.tools = {}
        self.catalog = []
        self.generating = False
        for tool in environment.get_tools():
            if tool.name in HIDDEN_AUI_TOOLS:
                continue
            if tool.name == ANSWER_TOOL:
                tool.class_instance.wait_for_user_response = False
            schema = public_tool_schema(tool)
            public_name = schema["name"]
            if public_name in self.tools:
                raise ValueError("Duplicate native public tool name")
            self.tools[public_name] = tool
            self.catalog.append({
                "name": public_name, "description": schema["description"],
                "parameters": schema["parameters"],
            })
        if not self.tools:
            raise ValueError("Native environment supplied no public agent tools")
        write_json_atomic(root / "NATIVE_CATALOG.json", self.catalog)

    @property
    def stop_requested(self) -> bool:
        return self.stopped

    def record(self, value: dict) -> None:
        value = {
            "at_unix": time.time(),
            "native_timestamp": self.environment.time_manager.time(),
            **value,
        }
        with self.lock, (self.root / "PUBLIC_BRIDGE.jsonl").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def call(self, name: str, arguments: dict) -> dict:
        from jsonschema import Draft202012Validator

        if name not in self.tools:
            raise ValueError("Tool is not in the native agent-visible catalog")
        schema = next(tool["parameters"] for tool in self.catalog if tool["name"] == name)
        validation = list(Draft202012Validator(schema).iter_errors(arguments))
        if validation:
            errors = [error.message for error in validation]
            self.record({"kind": "native_argument_error", "tool": name, "errors": errors})
            return {
                "observation": json.dumps({"tool_error": "Invalid native arguments", "errors": errors}),
                "conclude_turn": False, "is_error": True,
            }
        tool = self.tools[name]
        try:
            value = tool(**arguments)
        except Exception as error:
            # Native application errors are observations, not infrastructure successes.
            self.record({
                "kind": "native_tool_error", "tool": name, "arguments": arguments,
                "error_type": type(error).__name__, "error": str(error),
            })
            return {
                "observation": json.dumps({
                    "tool_error": type(error).__name__, "message": str(error),
                }),
                "conclude_turn": False, "is_error": True,
            }
        observation = value if isinstance(value, str) else str(value)
        native_name = tool.name
        conclude = native_name in (ANSWER_TOOL, WAIT_TOOL)
        if native_name == ANSWER_TOOL:
            self.answered_turns += 1
            self.turn_state = "answered"
        elif native_name == WAIT_TOOL:
            self.turn_state = "waiting"
        self.record({
            "kind": "native_tool_result", "tool": name, "arguments": arguments,
            "observation": observation, "conclude_turn": conclude,
        })
        return {"observation": observation, "conclude_turn": conclude, "is_error": False}

    def notifications(self) -> list[str | list[dict]]:
        timestamp = datetime.fromtimestamp(
            self.environment.time_manager.time(), tz=timezone.utc,
        )
        messages = self.environment.notification_system.message_queue.get_by_timestamp(timestamp)
        visible = []
        for message in messages:
            kind = message.message_type.value
            if kind == "ENVIRONMENT_STOP":
                self.stopped = True
                self.record({"kind": "native_environment_stop", "message": message.message})
                continue
            if kind not in ("USER_MESSAGE", "ENVIRONMENT_NOTIFICATION"):
                raise ValueError("Unrecognized native notification kind")
            text = f"[{kind} at {message.timestamp.isoformat()}]\n{message.message}"
            parts = [{"type": "text", "text": text}]
            for attachment in message.attachments:
                if not attachment.mime.startswith("image/"):
                    raise ValueError("Non-image native attachment requires an explicit public adapter")
                parts.append({
                    "type": "image", "data": attachment.base64_data.decode("ascii"),
                    "mimeType": attachment.mime,
                })
            visible.append(parts if message.attachments else text)
            self.record({
                "kind": "native_notification", "notification_type": kind,
                "message": message.message, "timestamp": message.timestamp.isoformat(),
                "attachment_count": len(message.attachments),
            })
        return visible

    def generation_start(self) -> None:
        if self.generating:
            raise RuntimeError("Native generation clock is already paused")
        self.environment.pause()
        self.generating = True
        self.record({"kind": "generation_start", "timing_mode": "per_attempt_measured"})

    def generation_end(self, elapsed_seconds: float) -> None:
        if not self.generating:
            raise RuntimeError("Native generation clock was not paused")
        self.environment.resume_with_offset(elapsed_seconds)
        self.generating = False
        self.record({"kind": "generation_end", "elapsed_seconds": elapsed_seconds})

    def finish_response(self, response: str | None) -> None:
        """Map an ordinary DSH final reply to the native user-message action once."""
        if not response or self.turn_state in ("answered", "waiting"):
            return
        candidates = [
            item for item in self.catalog if self.tools[item["name"]].name == ANSWER_TOOL
        ]
        if len(candidates) != 1:
            raise ValueError("Native user-message tool is missing or ambiguous")
        tool = candidates[0]
        properties = tool["parameters"]["properties"]
        if len(properties) != 1 or next(iter(properties.values())).get("type") != "string":
            raise ValueError("Native final-response action schema needs an explicit adapter")
        argument = next(iter(properties))
        result = self.call(tool["name"], {argument: response})
        if result["is_error"]:
            raise RuntimeError("Native final-response delivery failed")
        self.record({"kind": "harness_final_response_delivered", "tool": tool["name"]})
