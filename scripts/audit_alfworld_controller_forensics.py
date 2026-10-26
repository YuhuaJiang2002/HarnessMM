#!/usr/bin/env python3
"""Remote native replay and a validation-only, model-free controller diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_saved_usage(root: Path, row: dict, conversation: list[dict]) -> dict:
    totals = Counter({key: 0 for key in USAGE_KEYS})
    requests = sorted(root.rglob("request.json"))
    responses = {path.parent for path in root.rglob("response.json")}
    accounts = {path.parent for path in root.rglob("accounting.json")}
    require({path.parent for path in requests} == responses == accounts,
            f"Unmatched request/response/accounting artifacts: {root}")
    for path in requests:
        request = read_json(path)["body"]
        response = read_json(path.parent / "response.json")
        accounting = read_json(path.parent / "accounting.json")
        usage = response.get("usage", {})
        require(all(type(usage.get(key)) is int and usage[key] >= 0 for key in USAGE_KEYS),
                f"Invalid raw provider usage: {path}")
        measured = {key: usage[key] for key in USAGE_KEYS}
        require(measured["total_tokens"] == measured["prompt_tokens"] + measured["completion_tokens"],
                f"Provider arithmetic mismatch: {path}")
        require(accounting["usage"] == measured and accounting["calls"] == 1,
                f"Provider/accounting mismatch: {path}")
        require(request["model"] == response["model"] == row["model_id"],
                f"Provider model mismatch: {path}")
        totals.update(measured)
    require(dict(totals) == row["usage"], f"Episode/provider usage mismatch: {root}")
    require(len(requests) == row["calls"] == row["attempted_calls"],
            f"Episode/provider call-count mismatch: {root}")
    require({key: sum(step["usage"][key] for step in conversation) for key in USAGE_KEYS}
            == dict(totals), f"Per-step/provider usage mismatch: {root}")
    routes = Counter(step.get("hybrid_decision", {}).get("route", "actor") for step in conversation)
    return {"calls": len(requests), "usage": dict(totals), "routes": dict(routes)}


def verify_terminal(row: dict, conversation: list[dict], *, step_limit: int) -> None:
    require(row["valid"] is True and row["cost_complete"] is True, "Invalid or unaccounted episode")
    require(0 < len(conversation) == row["turns"] <= step_limit, "Invalid episode length")
    require([step["turn"] for step in conversation] == list(range(len(conversation))),
            "Missing or reordered native steps")
    require(not any(step["done"] for step in conversation[:-1]), "Steps after native termination")
    last = conversation[-1]
    require(type(last["done"]) is bool and type(last["official_won"]) is bool,
            "Terminal flags must be native booleans")
    require(row["official_success"] == int(last["done"] and last["official_won"]),
            "Recorded score differs from the native terminal flags")
    require(last["done"] or len(conversation) == step_limit, "Unexplained early termination")


def validation_items(protocol: dict) -> list[dict]:
    require(protocol["split_ratio"] == [0.2, 0.2, 0.6]
            or protocol["split_ratio"] == [20, 20, 60]
            or protocol["split_ratio"] == [2, 2, 6], "Expected the frozen 20/20/60 protocol")
    splits = protocol["splits"]
    identities = [{item["gamefile"] for item in splits[key]}
                  for key in ("train", "validation", "test")]
    require(all(len(values) == len(splits[key]) for values, key
                in zip(identities, ("train", "validation", "test"))), "Duplicate split identities")
    require(not identities[0] & identities[1] and not identities[0] & identities[2]
            and not identities[1] & identities[2], "Overlapping partitions")
    require([len(values) for values in identities] == [710, 710, 2133],
            "This diagnostic requires the complete frozen 3553-task partition")
    return sorted(splits["validation"], key=lambda item: item["id"])


def model_free_action(progress: dict, commands: list[str]) -> str | None:
    if not progress["eligible"]:
        return None
    candidates = progress["allowed_actions"]
    require(isinstance(candidates, list) and bool(candidates), "Supported state has no candidates")
    require(all(isinstance(action, str) and action in commands for action in candidates),
            "Controller proposed a nonnative action")
    return candidates[0]


def native_runtime(config: dict):
    os.environ.update(CUDA_VISIBLE_DEVICES="", ALFWORLD_DATA=config["data_root"],
                      ALFWORLD_WORKER_START_METHOD="fork", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    sys.path.insert(0, config["frozen_source"])
    sys.path.insert(0, config["skillopt_root"])
    from scripts.run_alfworld_autosaddler_worker import install_alfworld_unused_expert_guard
    from scripts.run_chisel_repair_alfworld import runtime_files
    from skillopt.envs.alfworld.rollout import _extract_action, build_alfworld_env

    install_alfworld_unused_expert_guard()
    runtime_files(Path.cwd())
    return build_alfworld_env, _extract_action


def replay_episode(config: dict, job: dict, output: Path) -> dict:
    root = Path(job["episode"])
    row = read_json(root / "result.json")
    conversation = read_json(root / "conversation.json")
    verify_terminal(row, conversation, step_limit=config["step_limit"])
    accounting = audit_saved_usage(root, row, conversation)
    build_env, extract_action = native_runtime(config)
    mode = row["alfworld_environment_mode"]
    env = build_env(1, mode, row["seed"], mode == "train", [row["gamefile"]])
    trace = []
    try:
        obs, infos = env.reset({})
        require(str(infos[0]["extra.gamefile"]).endswith(row["gamefile"]), "Native reset changed task")
        require(obs["anchor"][0] == read_json(root / "initial_observation.json")["observation"],
                "Native initial observation differs from the saved episode")
        for step in conversation:
            response = step["model_response"]
            if not response:
                response = "<think>empty model response</think><action>look</action>"
            elif extract_action(response) is None:
                response = "<think>missing action tag</think><action>look</action>"
            require((extract_action(response) or "").lower() == step["action"],
                    f"Saved action differs from the executed response at turn {step['turn']}")
            if "hybrid_decision" in step:
                decision = read_json(root / "decisions" / f"{step['turn']:03d}.json")
                commands = [value for value in env.envs.get_admissible_commands[0] if value != "help"]
                require(commands == decision["native_actions"], "Saved native-action inventory differs")
            obs, _, dones, infos = env.step([response])
            actual = {"turn": step["turn"], "action": step["action"],
                      "observation": str(obs["anchor"][0]), "done": bool(dones[0]),
                      "official_won": bool(infos[0].get("won", False)) if dones[0] else False}
            require(all(actual[key] == step[key] for key in actual),
                    f"Native environment replay mismatch at turn {step['turn']}")
            trace.append(actual)
        write_json(output / "native_replay.json", trace)
    finally:
        env.envs.close()
    return {"id": row["id"], "method": row["method"], "seed": row["seed"],
            "gamefile": row["gamefile"], "native_success": int(trace[-1]["done"] and trace[-1]["official_won"]),
            "steps": len(trace), "source_episode": str(root), "result_sha256": sha(root / "result.json"),
            "conversation_sha256": sha(root / "conversation.json"), **accounting}


def model_free_episode(config: dict, job: dict, output: Path) -> dict:
    item = job["item"]
    build_env, _ = native_runtime(config)
    from phiagent.learning.alfworld_jev import native_action_response, progress_choices

    env = build_env(1, "train", config["seed"], True, [item["gamefile"]])
    history = []
    trace = []
    done = won = False
    stop_reason = "native_step_limit"
    try:
        obs, infos = env.reset({})
        require(str(infos[0]["extra.gamefile"]).endswith(item["gamefile"]), "Native reset changed task")
        initial = str(obs["anchor"][0])
        task = initial.split("Your task is to: ", 1)[-1].strip()
        write_json(output / "initial_observation.json", {"observation": initial})
        for turn in range(config["step_limit"]):
            commands = [value for value in env.envs.get_admissible_commands[0] if value != "help"]
            block = "\n ".join(f"'{value}'" for value in commands)
            marker = "Your admissible actions of the current situation are: ["
            require(marker in obs["text"][0]
                    and obs["text"][0].rsplit(marker, 1)[1].split("].", 1)[0] == block,
                    "Controller action inventory is not actor-visible")
            progress = progress_choices(task=task, initial_observation=initial,
                                        actions=commands, history=history)
            action = model_free_action(progress, commands)
            if action is None:
                stop_reason = "controller_abstained_without_model_fallback"
                trace.append({"turn": turn, "abstained": True, "progress": progress})
                break
            obs, _, dones, infos = env.step([native_action_response(action)])
            done = bool(dones[0])
            won = bool(infos[0].get("won", False)) if done else False
            record = {"turn": turn, "action": action, "observation": str(obs["anchor"][0]),
                      "done": done, "official_won": won, "progress": progress}
            trace.append(record)
            history.append({"action": action, "observation": record["observation"]})
            if done:
                stop_reason = "native_environment_done"
                break
        write_json(output / "conversation.json", trace)
    finally:
        env.envs.close()
    return {"id": item["id"], "gamefile": item["gamefile"], "task_type": item["task_type"],
            "method": "handwritten_controller_only", "seed": config["seed"],
            "native_success": int(done and won), "steps": len(history), "stop_reason": stop_reason,
            "calls": 0, "usage": dict.fromkeys(USAGE_KEYS, 0),
            "routes": {"deterministic": len(history)}, "learned_skill_used": False,
            "model_output_used": False, "private_environment_facts_used": False}


def execute_job(config: dict, job: dict) -> dict:
    identity = job.get("episode") or job["item"]["id"]
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    output = Path(config["output"]) / "episodes" / key
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    write_json(output / "job.json", job)
    started = time.monotonic()
    try:
        result = (replay_episode(config, job, output) if config["mode"] == "native-replay"
                  else model_free_episode(config, job, output))
        result.update(status="PASS", elapsed_seconds=time.monotonic() - started)
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "type": type(error).__name__,
                                            "message": str(error), "traceback": traceback.format_exc()})
        raise


def run(config_path: Path) -> None:
    config = read_json(config_path)
    require(sys.platform != "darwin" and socket.gethostname() == config["authorized_hostname"],
            "Native benchmark replay and validation must run on the authorized remote server")
    require(config["mode"] in ("native-replay", "model-free-validation"), "Unknown diagnostic mode")
    require(config["step_limit"] == 50, "The native 50-step protocol must be preserved")
    require(config["cumulative_token_mode"] == "unlimited", "No cumulative token ceiling is allowed")
    output = Path(config["output"])
    require(output.is_absolute(), "A new absolute experiment directory is required")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    try:
        for relative, expected in config["source_hashes"].items():
            require(sha(Path(config["frozen_source"]) / relative) == expected,
                    f"Frozen controller source changed: {relative}")
        packages = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                  capture_output=True, text=True, check=True).stdout
        write_json(output / "PROVENANCE.json", {
            "argv": sys.argv, "hostname": socket.gethostname(), "python": sys.executable,
            "packages": packages.splitlines(), "script_sha256": sha(Path(__file__)),
            "source_hashes": config["source_hashes"], "config_sha256": sha(config_path),
            "source_git_state": config["source_git_state"], "seed": config["seed"],
        })
        write_json(output / "GPU_SELECTION.json", {
            "cuda_visible_devices": "", "physical_gpu_ids": [],
            "reason": "CPU-only native replay / model-free validation; no inference provider is invoked",
            "pool_snapshot": config["resource_snapshot"],
        })
        if config["mode"] == "native-replay":
            jobs = [{"episode": str(path.parent)} for root in config["source_runs"]
                    for arm in config["arms"]
                    for path in sorted((Path(root) / "official134" / arm).glob("*/result.json"))]
            require(len(jobs) == config["expected_episodes"], "Incomplete historical episode inventory")
        else:
            protocol_path = Path(config["protocol_path"])
            require(sha(protocol_path) == config["protocol_file_sha256"], "Partition file changed")
            protocol = read_json(protocol_path)
            jobs = [{"item": item} for item in validation_items(protocol)]
            require(config["seed"] == protocol["validation_seed"], "Validation seed changed")
            validation_directories = {str(Path(job["item"]["gamefile"]).parent) for job in jobs}
            data_hashes = {name: expected for name, expected in protocol["data_hashes"].items()
                           if str(Path(name).parent) in validation_directories}
            require({job["item"]["gamefile"] for job in jobs} <= set(data_hashes),
                    "Validation tasks lack frozen native data hashes")
            for name, expected in data_hashes.items():
                require(sha(Path(config["data_root"]) / name) == expected,
                        f"Frozen validation data changed: {name}")
            write_json(output / "DATA_VERIFIED.json", {
                "files": len(data_hashes), "files_sha256": data_hashes,
                "test_data_opened": False,
            })
        require(len({json.dumps(job, sort_keys=True) for job in jobs}) == len(jobs), "Duplicate jobs")
        if config.get("preflight_only"):
            jobs = jobs[:1]
        write_json(output / "JOBS.json", jobs)
        records = []
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            records.append(pool.submit(execute_job, config, jobs[0]).result())
            write_json(output / "FIRST_NATIVE_EPISODE_VERIFIED.json", records[0])
            futures = [pool.submit(execute_job, config, job) for job in jobs[1:]]
            for future in as_completed(futures):
                records.append(future.result())
                write_json(output / "HEARTBEAT.json", {
                    "completed": len(records), "expected": len(jobs), "updated_at_unix": time.time(),
                    "status": "RUNNING",
                })
        by_method = {}
        for method in sorted({row["method"] for row in records}):
            selected = [row for row in records if row["method"] == method]
            by_method[method] = {
                "episodes": len(selected), "successes": sum(row["native_success"] for row in selected),
                "calls": sum(row["calls"] for row in selected),
                "usage": {key: sum(row["usage"][key] for row in selected) for key in USAGE_KEYS},
                "steps": sum(row["steps"] for row in selected),
                "routes": dict(sum((Counter(row["routes"]) for row in selected), Counter())),
            }
        write_json(output / "RESULT.json", {
            "status": "PREFLIGHT_PASS" if config.get("preflight_only") else "COMPLETE",
            "mode": config["mode"], "episodes": len(records), "methods": by_method,
            "new_provider_calls": 0, "new_provider_tokens": 0,
            "population": "historical native replay" if config["mode"] == "native-replay" else "validation710",
            "final_test_result": False, "paper_method_improvement_admitted": False,
            "config_sha256": sha(output / "config.json"),
            "episode_results": {str(path.relative_to(output)): sha(path)
                                for path in sorted((output / "episodes").glob("*/result.json"))},
        })
        write_json(output / "HEARTBEAT.json", {
            "status": "COMPLETE", "completed": len(records), "expected": len(jobs),
        })
    except Exception as error:
        write_json(output / "FAILURE.json", {"status": "FAILED", "type": type(error).__name__,
                                            "message": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
