"""Native ARE adapter for pinned Pi and the unchanged official SoL-Pi extensions."""
from __future__ import annotations

import atexit
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import subprocess
import time


class SettlingProviderServer(ThreadingHTTPServer):
    # Close admission first; accepted calls must settle before the model server exits.
    daemon_threads = False
    block_on_close = True


def finalize_provider_accounting(config, root):
    if not config.get("solpi"):
        raise ValueError("Official Pi configuration is required for final accounting")
    root = Path(root)
    ledger = json.loads((root / "PROVIDER_FINAL.json").read_text())
    if ledger["unknown_usage_calls"] or ledger["inflight_reserved_tokens"]:
        raise ValueError("Cannot finalize unsettled official Pi provider usage")
    calls = ledger["calls"]
    if any(call["status"] not in ("SETTLED", "REJECTED_BEFORE_FORWARD") for call in calls):
        raise ValueError("Cannot finalize an incomplete provider call")
    if len({call["sequence"] for call in calls}) != len(calls):
        raise ValueError("Duplicate provider call sequence")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    for call in calls:
        usage = call.get("usage")
        if call["status"] == "SETTLED" and (
            not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in fields)
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
        ):
            raise ValueError("Incomplete settled provider usage")
    if sum((call.get("usage") or {}).get("total_tokens", 0) for call in calls) != ledger["known_tokens"]:
        raise ValueError("Final provider total differs from its complete calls")
    result_path = root / "RESULT.json"
    result = json.loads(result_path.read_text())
    actor_calls = [call for call in calls if call["role"].endswith("/actor")]
    actor = {key: sum((call.get("usage") or {}).get(key, 0) for call in actor_calls) for key in fields}
    before = result["actor_usage"]
    if any(actor[key] < before[key] for key in fields):
        raise ValueError("Provider drain cannot remove previously recorded actor usage")
    receipt = {
        "result_before_drain_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "provider_final_sha256": hashlib.sha256((root / "PROVIDER_FINAL.json").read_bytes()).hexdigest(),
        "actor_usage_before": before, "actor_usage_final": actor,
        "all_accepted_requests_settled": True,
        "scored_outcomes_changed": False,
    }
    from scripts.run_gaia2_jev import write

    cost_path = root / "evaluation/COST.json"
    cost = json.loads(cost_path.read_text())
    role = cost["role"]
    scoped_actor = [call for call in calls if call["role"] == role + "/actor"]
    scoped_judge = [call for call in calls if call["role"] == role + "/judge"]
    final_cost = {
        **cost,
        "actor": {key: sum((call.get("usage") or {}).get(key, 0) for call in scoped_actor) for key in fields},
        "judge_tokens": sum((call.get("usage") or {}).get("total_tokens", 0) for call in scoped_judge),
        "actor_calls": len(scoped_actor), "judge_calls": len(scoped_judge), "unresolved_calls": 0,
    }
    if final_cost["actor"] != actor:
        raise ValueError("The final actor ledger contains calls outside the native evaluation")
    if final_cost != cost:
        with (root / "evaluation/COST_BEFORE_PROVIDER_DRAIN.json").open("x") as stream:
            json.dump(cost, stream, indent=2)
            stream.write("\n")
        write(cost_path, final_cost)
    if before != actor:
        archive = root / "RESULT_BEFORE_PROVIDER_DRAIN.json"
        with archive.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        result["actor_usage"] = actor
        write(result_path, result)
    receipt["final_result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    write(root / "PROVIDER_DRAIN_ACCOUNTING.json", receipt)


def validate_output_capacity(request, config):
    role, maximum = request["role"], request["json"].get("max_tokens")
    if role.endswith("/actor"):
        if (not config.get("solpi") or type(maximum) is not int
                or not 0 < maximum <= config["model"]["max_tokens"]):
            raise ValueError("Official Pi actor request exceeds native output capacity")
    elif role.endswith("/judge") and maximum != config["model"]["max_tokens"]:
        raise ValueError("Native judge output capacity changed")


def validate_runtime(settings):
    runtime = Path(settings["runtime"])
    receipt = runtime.parent / "READY.json"
    if not receipt.is_file():
        raise ValueError("The pinned official runtime has not passed its H200 portability check")
    ready = json.loads(receipt.read_text())
    if (ready.get("status") != "PASS_OFFICIAL_RUNTIME_ON_H200"
            or ready.get("revision") != settings["revision"]
            or ready.get("archive_sha256") != settings["installed_runtime_archive_sha256"]):
        raise ValueError("Official runtime provenance differs from the frozen adapter")
    lock = hashlib.sha256((runtime / "package-lock.json").read_bytes()).hexdigest()
    if lock != "d7bbfa580faf15501cc71610317563e72fdb2d8e4a0c00aa7038891dcd1e2d70":
        raise ValueError("The official SoL-Pi dependency lockfile changed")
    package = json.loads((runtime / "node_modules/@earendil-works/pi-coding-agent/package.json").read_text())
    if package.get("version") != "0.85.1":
        raise ValueError("The official Pi version changed")
    version = subprocess.check_output([settings["node"], "--version"], text=True).strip()
    if version != "v22.19.0":
        raise ValueError("The checksum-bound Node runtime changed")
    return {"status": "PASS", "runtime_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}


def request_transport(body, config, native_actor):
    if not config.get("solpi"):
        raise ValueError("Pi transport requires an explicit official-runtime configuration")
    value = dict(body)
    streaming = value.pop("stream", False)
    if type(streaming) is not bool:
        raise ValueError("Streaming must be explicit boolean metadata")
    value.pop("stream_options", None)
    requested = value.pop("max_completion_tokens", value.get("max_tokens", 4096))
    if type(requested) is not int or requested < 1:
        raise ValueError("A positive output capacity is required")
    maximum = min(requested, 4096 if native_actor else 32768)
    value["stream"] = False
    return value, streaming, maximum


def streaming_payload(raw):
    choice = raw["choices"][0]
    common = {key: raw[key] for key in ("id", "created", "model")}
    common["object"] = "chat.completion.chunk"
    message = choice["message"]
    delta = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": index, **call} for index, call in enumerate(message["tool_calls"])
        ]
    chunks = [
        {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
        {**common, "choices": [], "usage": raw["usage"]},
    ]
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"


def actor_usage(root, role):
    totals = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    path = Path(root) / "PROVIDER_CALLS.jsonl"
    if not path.exists():
        return totals
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        row = json.loads(line)
        if row["role"] != role + "/actor":
            continue
        if row["status"] == "UNKNOWN":
            raise ValueError("Official Pi inference has unresolved provider usage")
        if row.get("usage"):
            for key in totals:
                totals[key] += row["usage"][key]
    return totals


def close_child(process, logfile):
    if process.poll() is None:
        try:
            process.stdin.write((json.dumps({"type": "close"}) + "\n").encode())
            process.stdin.flush()
            process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    logfile.close()


def install_adapter(config, directory, role, mode, skill):
    if (mode != "native" or config["method"] != "native" or skill
            or config["stage"] != "train-candidate"
            or config["solpi"]["variant"] not in ("pi_no_solpi", "solpi_official")):
        raise RuntimeError("Only explicitly identified native Pi/SoL-Pi TRAIN comparisons are supported")
    validate_runtime(config["solpi"])
    from are.simulation.agents.default_agent.base_agent import BaseAgent
    from phiagent.harness.gaia2_202060_progress import install_metered_provider

    original_step = BaseAgent.step
    children = []
    record = install_metered_provider(config, directory, role)

    def cleanup():
        for process, logfile in children:
            close_child(process, logfile)

    atexit.register(cleanup)

    class OfficialPiEngine:
        def __init__(self, agent):
            self.agent = agent
            identity = hashlib.sha256(str(agent.agent_id).encode()).hexdigest()[:16]
            self.base = Path(directory) / "official-pi" / identity
            self.episode = -1
            self.first_log = None
            self.process = None
            self.logfile = None
            self.begin_episode()

        def begin_episode(self):
            if self.process is not None:
                close_child(self.process, self.logfile)
            self.episode += 1
            self.root = self.base / f"episode-{self.episode}"
            self.root.mkdir(parents=True, exist_ok=False)
            self.first_log = self.agent.logs[0] if self.agent.logs else None
            self.pending = None
            self.cursor = 0
            self.started = False
            self.buffer = b""
            self.usage = actor_usage(config["run_directory"], role)
            self.logfile = (self.root / "bridge.stderr.log").open("x")
            bridge = Path(config["worker_script"]).parent / "solpi_native_bridge.mjs"
            env = {
                "PATH": str(Path(config["solpi"]["node"]).parent) + ":/usr/bin:/bin",
                "HOME": str(self.root / "home"), "LANG": "C.UTF-8",
                "CUDA_VISIBLE_DEVICES": "", "PI_TELEMETRY": "0",
                "PHI_SOLPI_RUNTIME": config["solpi"]["runtime"],
                "PI_CODING_AGENT_DIR": str(self.root / "pi-agent"),
            }
            self.process = subprocess.Popen(
                [config["solpi"]["node"], str(bridge)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.logfile, bufsize=0, env=env,
            )
            children.append((self.process, self.logfile))
            record({"source": "official_pi_process", "pid": self.process.pid, "root": str(self.root),
                    "variant": config["solpi"]["variant"], "source_revision": config["solpi"]["revision"]})

        def send(self, value):
            self.process.stdin.write((json.dumps(value) + "\n").encode())
            self.process.stdin.flush()

        def event(self):
            while b"\n" not in self.buffer:
                if not select.select([self.process.stdout], [], [], 7500)[0]:
                    raise TimeoutError("Official Pi did not return a native action within the shared timeout")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Official Pi exited; see bridge.stderr.log")
                self.buffer += chunk
            line, self.buffer = self.buffer.split(b"\n", 1)
            return json.loads(line)

        def __call__(self, prompt, **kwargs):
            if not prompt or prompt[0].get("role") != "system" or not isinstance(prompt[0].get("content"), str):
                raise ValueError("Native Pi requires the complete native textual system prompt")
            start = time.monotonic()
            current_first = self.agent.logs[0] if self.agent.logs else None
            if self.started and (current_first is not self.first_log or self.cursor > len(self.agent.logs)):
                self.begin_episode()
            fresh = self.agent.logs[self.cursor:]
            self.cursor = len(self.agent.logs)
            observations = [
                row.content for row in fresh
                if row.get_type() in ("observation", "error") and isinstance(getattr(row, "content", None), str)
            ]
            events = [
                row.content for row in fresh
                if row.get_type() in ("task", "agent_user_interface", "environment_notifications", "hint", "task_reminder")
                and isinstance(getattr(row, "content", None), str)
            ]
            if not self.started:
                self.send({
                    "type": "start", "directory": str(self.root), "variant": config["solpi"]["variant"],
                    "model": config["model"]["served_name"],
                    "endpoint": f"http://127.0.0.1:{config['resource']['capture_port']}/{role}/actor/v1",
                    "system": prompt[0]["content"], "task": "\n".join(events),
                    "tools": list(self.agent.action_executor.tools),
                })
                self.started = True
            elif self.pending is not None:
                self.send({
                    "type": "result", "id": self.pending, "observation": "\n".join(observations),
                    "events": "\n".join(events), "is_error": any(row.get_type() == "error" for row in fresh),
                    "system": prompt[0]["content"],
                })
                self.pending = None
            else:
                self.send({"type": "next", "text": "\n".join(events + observations),
                           "system": prompt[0]["content"]})
            while True:
                event = self.event()
                if event["type"] == "ready":
                    record({"source": "official_solpi_ready", **event})
                    continue
                if event["type"] == "error":
                    raise RuntimeError("Official Pi bridge failed: " + event["message"])
                if event["type"] not in ("action", "final"):
                    raise ValueError("Unexpected official Pi protocol event")
                current = actor_usage(config["run_directory"], role)
                if current["total_tokens"] <= 0:
                    raise ValueError("An official Pi action has no accounted model evidence")
                usage = {key: current[key] - self.usage[key] for key in current}
                if any(value < 0 for value in usage.values()):
                    raise ValueError("Official Pi actor usage moved backwards")
                self.usage = current
                usage["completion_duration"] = time.monotonic() - start
                if event["type"] == "action":
                    if event["action"] not in self.agent.action_executor.tools or not isinstance(event["action_input"], dict):
                        raise ValueError("Official Pi attempted a non-native action")
                    self.pending = event["id"]
                    action = {key: event[key] for key in ("action", "action_input")}
                    response = "Thought: Execute the action selected by the official Pi session.\nAction:\n" + json.dumps(action)
                else:
                    response = event["text"]
                record({"source": "official_pi_native_action", "event": event, "usage": usage,
                        "main_reducer_and_compaction_calls_included": True})
                return response, usage

    def step(agent):
        if not isinstance(agent.llm_engine, OfficialPiEngine):
            agent.llm_engine = OfficialPiEngine(agent)
        return original_step(agent)

    BaseAgent.step = step
