"""Opt-in, evidence-preserving terminal actor request projection."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phiagent.harness.observation_pack import ObservationPack


UPSTREAM_REVISION = "bd005888b9b8a3fcdb511feb91fc27d3dfa8f2b1"
RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "ObsRecall",
        "description": "Read an exact page from an archived terminal observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "observation_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8000},
            },
            "required": ["observation_id"],
            "additionalProperties": False,
        },
    },
}


class TerminalSoLPiSession:
    def __init__(self, config: Mapping[str, Any], trial_dir: Path) -> None:
        if (not isinstance(config, Mapping) or "observation_pack" not in config
                or set(config) - {"observation_pack", "action_fusion"}):
            raise ValueError("solpi requires an explicit observation_pack switch")
        if any(type(value) is not bool for value in config.values()):
            raise ValueError("solpi switches must be boolean")
        self.action_fusion = config.get("action_fusion", False)
        self._fusion_requests = 0
        self._validation_executed = 0
        self._validation_failed = 0
        self._model_requests = 0
        self._usage: list[dict[str, Any]] = []
        self.pack = (
            ObservationPack(trial_dir / "solpi" / "observations")
            if config["observation_pack"] else None
        )

    def tools(self, tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = deepcopy(list(tools))
        if self.action_fusion:
            matches = [tool["function"] for tool in result
                       if tool.get("function", {}).get("name") == "exec_in_sandbox"]
            if len(matches) != 1:
                raise ValueError("Action Fusion requires the existing exec_in_sandbox tool")
            function = matches[0]
            function["description"] += (
                " When the next validation command is already known, include it as "
                "validation_command. It runs only after command succeeds, in the same "
                "container with the remaining shared timeout; both results are returned."
            )
            function["parameters"]["properties"]["validation_command"] = {
                "type": "string", "minLength": 1,
                "description": "An already-known validation command; omit if it depends on the first output.",
            }
        if self.pack is not None:
            if any(tool.get("function", {}).get("name") == "ObsRecall" for tool in result):
                raise ValueError("ObsRecall already exists in the actor tool interface")
            result.append(deepcopy(RECALL_TOOL))
        return result

    def project(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.pack is None:
            return [dict(message) for message in messages]
        return self.pack.project(messages)

    def recall(self, tool_call: Mapping[str, Any]) -> str:
        if self.pack is None:
            raise ValueError("ObservationPack is disabled")
        function = tool_call.get("function")
        if not isinstance(function, Mapping) or function.get("name") != "ObsRecall":
            raise ValueError("expected an ObsRecall tool call")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("ObsRecall arguments must be JSON text")
        value = json.loads(arguments)
        if (
            not isinstance(value, dict)
            or "observation_id" not in value
            or set(value) - {"observation_id", "offset", "limit"}
        ):
            raise ValueError("invalid ObsRecall arguments")
        return self.pack.recall(**value)

    def validation_command(self, tool_call: Mapping[str, Any]) -> str | None:
        arguments = json.loads(tool_call["function"]["arguments"])
        if "validation_command" not in arguments:
            return None
        if not self.action_fusion:
            raise ValueError("Action Fusion is disabled")
        command = arguments["validation_command"]
        if not isinstance(command, str) or not command.strip():
            raise ValueError("validation_command must be a nonempty string")
        return command

    def combine_results(
        self, primary: Mapping[str, Any], validation: Mapping[str, Any] | None,
        validation_command: str,
    ) -> dict[str, Any]:
        self._fusion_requests += 1
        result = dict(primary)
        result["primary_result"] = dict(primary)
        result["validation_result"] = dict(validation) if validation is not None else None
        result["validation_command"] = validation_command
        if validation is not None:
            self._validation_executed += 1
            result["validation_status"] = "executed"
            result["exit_code"] = validation["exit_code"]
            result["timed_out"] = bool(validation["timed_out"])
            result["stdout"] = str(primary["stdout"]) + "\n[validation stdout]\n" + str(validation["stdout"])
            result["stderr"] = str(primary["stderr"]) + "\n[validation stderr]\n" + str(validation["stderr"])
            result["duration_seconds"] = float(primary.get("duration_seconds", 0)) + float(validation.get("duration_seconds", 0))
            result["finished_at_utc"] = validation.get("finished_at_utc")
        elif primary["exit_code"] != 0 or primary["timed_out"]:
            result["validation_status"] = "skipped_primary_failed"
        else:
            result["validation_status"] = "shared_deadline_exhausted"
            result["exit_code"] = 124
            result["timed_out"] = True
            result["stderr"] = str(primary["stderr"]) + "\nValidation not executed: shared timeout exhausted."
        result["validation_failed"] = result["exit_code"] != 0 or result["timed_out"]
        self._validation_failed += int(result["validation_failed"])
        return result

    def record_request(self) -> None:
        self._model_requests += 1

    def record_response(self, usage: object, *, role: str = "actor", complete: bool = True) -> None:
        usage = usage if isinstance(usage, Mapping) else {}
        counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        valid = all(type(value) is int and value >= 0 for value in counts)
        if "total_tokens" in usage:
            valid = valid and type(usage["total_tokens"]) is int and usage["total_tokens"] == sum(counts)
        self._usage.append({
            "role": role,
            "complete": bool(valid and complete),
            "input_tokens": counts[0] if type(counts[0]) is int and counts[0] >= 0 else None,
            "output_tokens": counts[1] if type(counts[1]) is int and counts[1] >= 0 else None,
        })

    def report(self) -> dict[str, Any]:
        return {
            "upstream_revision": UPSTREAM_REVISION,
            "identity": "PhiAgent SoL-Pi-inspired terminal adapter",
            "observation_pack": (
                self.pack.report() if self.pack is not None else {"enabled": False}
            ),
            "action_fusion": {
                "enabled": self.action_fusion,
                "requests": self._fusion_requests,
                "validation_executed": self._validation_executed,
                "validation_failed": self._validation_failed,
            },
            "usage_accounting": {
                "model_requests_attempted": self._model_requests,
                "responses_returned": len(self._usage),
                "complete": self._model_requests == len(self._usage) and all(row["complete"] for row in self._usage),
                "known_input_tokens": sum(row["input_tokens"] or 0 for row in self._usage),
                "known_output_tokens": sum(row["output_tokens"] or 0 for row in self._usage),
                "responses": list(self._usage),
            },
        }
