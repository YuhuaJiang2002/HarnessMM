"""Native Harbor lifecycle for the frozen PhiAgent terminal actor."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shlex
import threading
import time

from harbor.agents.base import BaseAgent

from scripts.run_terminal_bench_qwen35_4b_direct import run_agent, utc_now, write_json


def trajectory_snapshot(directory: Path, *, model: str, session_id: str) -> tuple[dict, dict]:
    raw_root = directory / "agent"
    requests = sorted(raw_root.glob("turn-*/request.json"))
    responses = sorted(raw_root.glob("turn-*/response.json"))
    transcript_path = raw_root / "transcript.json"
    tools = []
    messages = []
    if requests:
        first = json.loads(requests[0].read_text())
        messages = deepcopy(first["messages"][:2])
        tools = first.get("tools", [])
    usage_rows = []
    for response_path in responses:
        response = json.loads(response_path.read_text())
        usage = response.get("usage") or {}
        valid = all(type(usage.get(key)) is int and usage[key] >= 0
                    for key in ("prompt_tokens", "completion_tokens"))
        if "total_tokens" in usage:
            valid = valid and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        usage_rows.append({"usage": usage, "complete": valid})
        choices = response.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            continue
        message = {"role": "assistant", **choices[0]["message"]}
        messages.append(message)
        for tool_root in sorted(response_path.parent.glob("tool-*")):
            result_path = tool_root / "result.json"
            if not result_path.is_file():
                continue
            call = json.loads((tool_root / "call.json").read_text())
            result = json.loads(result_path.read_text())
            messages.append({"role": "tool", "tool_call_id": call.get("id", "unknown"),
                             "content": result.get("content", json.dumps(result, sort_keys=True))})
    if transcript_path.is_file():
        messages = json.loads(transcript_path.read_text())
    steps = []
    agent_index = 0
    for message in messages:
        role = message.get("role")
        if role == "tool":
            if not steps or steps[-1]["source"] != "agent":
                raise ValueError("tool observation has no paired agent step")
            steps[-1].setdefault("observation", {"results": []})["results"].append({
                "source_call_id": message["tool_call_id"], "content": str(message.get("content") or ""),
            })
            continue
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported canonical transcript role")
        step = {"step_id": len(steps) + 1, "source": "agent" if role == "assistant" else role,
                "message": str(message.get("content") or "")}
        if role == "assistant":
            step["model_name"] = model
            calls = []
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                try:
                    parsed = json.loads(arguments)
                    if not isinstance(parsed, dict):
                        raise ValueError("tool arguments are not an object")
                except (ValueError, TypeError):
                    parsed = {"invalid_raw_arguments": arguments}
                calls.append({"tool_call_id": call["id"], "function_name": call["function"]["name"],
                              "arguments": parsed})
            if calls:
                step["tool_calls"] = calls
            if agent_index < len(usage_rows):
                row = usage_rows[agent_index]
                usage = row["usage"]
                step["llm_call_count"] = 1
                if row["complete"]:
                    step["metrics"] = {"prompt_tokens": usage["prompt_tokens"],
                                       "completion_tokens": usage["completion_tokens"],
                                       "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
            agent_index += 1
        steps.append(step)
    if not steps:
        steps = [{"step_id": 1, "source": "system", "message": "Actor did not return a model response."}]
    accounting = {
        "requests": len(requests), "responses": len(responses),
        "complete": len(requests) == len(responses) and all(row["complete"] for row in usage_rows),
        "input_tokens": sum(row["usage"].get("prompt_tokens", 0) or 0 for row in usage_rows),
        "output_tokens": sum(row["usage"].get("completion_tokens", 0) or 0 for row in usage_rows),
        "cached_tokens": sum((row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for row in usage_rows),
    }
    return {
        "schema_version": "ATIF-v1.7", "session_id": session_id,
        "agent": {"name": "phiagent-terminal", "version": "solpi-harbor-v1", "model_name": model,
                  "tool_definitions": tools}, "steps": steps,
        "extra": {"usage_accounting": accounting, "canonical_transcript_available": transcript_path.is_file()},
    }, accounting


class TerminalAgent(BaseAgent):
    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "phiagent-terminal"

    def version(self) -> str:
        return "solpi-harbor-v1"

    def __init__(self, *args, binding_path: str, binding_sha256: str, arm: str,
                 repeat_seed: int, task_id: str, official_agent_timeout: float, **kwargs):
        super().__init__(*args, **kwargs)
        payload = Path(binding_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding_sha256:
            raise ValueError("native actor binding changed after freeze")
        binding = json.loads(payload)
        self.config = {"model": deepcopy(binding["model"]), "agent": deepcopy(binding["actors"][arm])}
        self.config["model"]["seed"] = repeat_seed
        if self.config["agent"].get("context_overflow_policy", {}).get("max_retries", 0):
            raise ValueError("native matched campaign forbids lossy overflow retries")
        self.arm = arm
        self.task_id = task_id
        self.official_timeout = float(official_agent_timeout)
        if self.official_timeout <= 0:
            raise ValueError("official task timeout must be positive")
        self.raw_directory = self.logs_dir / "phiagent"
        self.cancelled = threading.Event()
        self.pending = set()
        self.pending_lock = threading.Lock()

    async def setup(self, environment) -> None:
        self.raw_directory.mkdir(parents=True, exist_ok=False)
        write_json(self.logs_dir / "actor-binding.json", {
            "arm": self.arm, "task_id": self.task_id, "model": self.config["model"],
            "official_agent_timeout": self.official_timeout, "agent": self.config["agent"],
            "environment_owner": "native Harbor", "verifier_modified": False,
        })

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise InterruptedError("native Harbor trial cancelled")

    async def run(self, instruction: str, environment, context) -> None:
        loop = asyncio.get_running_loop()

        def execute(_container: str, command: str, timeout_seconds: float) -> dict:
            self.check_cancelled()
            started = time.monotonic()
            started_at = utc_now()
            timeout = max(1, math.ceil(timeout_seconds))
            wrapped = f"timeout --signal=TERM --kill-after=5s {timeout}s bash -lc {shlex.quote(command)}"
            future = asyncio.run_coroutine_threadsafe(environment.exec(wrapped, timeout_sec=timeout + 15), loop)
            with self.pending_lock:
                self.pending.add(future)
            try:
                self.check_cancelled()
                result = future.result(timeout=timeout + 20)
                return {"command": command, "native_command": wrapped, "exit_code": result.return_code,
                        "stdout": result.stdout or "", "stderr": result.stderr or "",
                    "timed_out": result.return_code in {124, 137}, "started_at_utc": started_at,
                        "finished_at_utc": utc_now(), "duration_seconds": time.monotonic() - started,
                        "timeout_enforced_inside_container": True, "executor": "Harbor BaseEnvironment.exec"}
            except CancelledError as error:
                raise InterruptedError("native environment command cancelled") from error
            finally:
                with self.pending_lock:
                    self.pending.discard(future)
                if not future.done():
                    future.cancel()

        worker = asyncio.create_task(asyncio.to_thread(
            run_agent, config=self.config, instruction=instruction, container_name="native-harbor",
            trial_dir=self.raw_directory, agent_timeout_seconds=self.official_timeout,
            task_spec={"name": self.task_id}, tool_executor=execute, cancellation_check=self.check_cancelled,
        ))
        try:
            result = await asyncio.shield(worker)
            write_json(self.logs_dir / "actor-result.json", result)
        finally:
            self.cancelled.set()
            with self.pending_lock:
                for future in self.pending:
                    future.cancel()
            self.populate_context_post_run(context)

    def populate_context_post_run(self, context) -> None:
        from harbor.models.trajectories.trajectory import Trajectory

        trajectory, accounting = trajectory_snapshot(
            self.raw_directory, model=self.config["model"]["expected_response_model_id"],
            session_id=self.session_id or self.logs_dir.parent.name,
        )
        validated = Trajectory.model_validate(trajectory)
        write_json(self.logs_dir / "trajectory.json", validated.model_dump(mode="json", exclude_none=True))
        write_json(self.logs_dir / "usage.json", accounting)
        context.n_input_tokens = accounting["input_tokens"]
        context.n_output_tokens = accounting["output_tokens"]
        context.n_cache_tokens = accounting["cached_tokens"]
        context.metadata = {"arm": self.arm, "task_id": self.task_id,
                            "repeat_seed": self.config["model"]["seed"], "usage_accounting": accounting}
