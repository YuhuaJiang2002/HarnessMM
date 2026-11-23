"""Run the frozen Adroit training pilot on a remote CPU."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import socket
from time import perf_counter

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from PIL import Image

from responses_actor import Actor

PROTOCOL = Path(__file__).with_name("protocol.json")


class Motor:
    """Execute the deterministic evaluation action from a public DAPG policy."""

    def __init__(self, path, source_sha256):
        with np.load(path, allow_pickle=False) as archive:
            if archive["source_sha256"].item() != source_sha256:
                raise ValueError(f"{path}: unexpected source checkpoint")
            self.in_shift, self.in_scale = archive["in_shift"], archive["in_scale"]
            self.out_shift, self.out_scale = archive["out_shift"], archive["out_scale"]
            self.layers = [
                (archive[f"weight{i}"], archive[f"bias{i}"])
                for i in range(int(archive["layers"]))
            ]
            self.activation = archive["activation"].item()
            if self.activation not in ("tanh", "relu"):
                raise ValueError(f"{path}: unsupported activation {self.activation}")
            np.testing.assert_allclose(
                self.evaluate(archive["sample"]), archive["reference"], rtol=1e-5, atol=1e-5
            )

    def evaluate(self, observation):
        x = (np.asarray(observation, dtype=np.float32) - self.in_shift) / (self.in_scale + 1e-8)
        for weight, bias in self.layers[:-1]:
            x = x @ weight.T + bias
            x = np.maximum(x, 0) if self.activation == "relu" else np.tanh(x)
        weight, bias = self.layers[-1]
        return (x @ weight.T + bias) * self.out_scale + self.out_shift

    def action(self, observation, space):
        action = self.evaluate(observation)
        if action.shape != space.shape or not np.isfinite(action).all():
            raise ValueError("Invalid DAPG action")
        return np.clip(action, space.low, space.high).astype(space.dtype)


def select_skill(actor, task, tasks, image):
    skills = [item["skill"] for item in tasks]
    prompt = (
        "You control a simulated dexterous hand. Choose one motor skill for the "
        "instruction and camera image. The motor runs the chosen skill closed loop. "
        "Available skills: pen (reorient a pen), door (unlatch and open a door), "
        "relocate (grasp and move a ball), or abstain. "
        "Preserve task performance while minimizing total input and output tokens. "
        f"Instruction: {task['instruction']}"
    )
    schema = {
        "type": "object",
        "properties": {"skill": {"type": "string", "enum": [*skills, "abstain"]}},
        "required": ["skill"],
        "additionalProperties": False,
    }
    selection = actor.call(prompt, [image], schema, "adroit_skill_select")["skill"]
    if selection not in [*skills, "abstain"]:
        raise ValueError("Provider returned an unregistered skill")
    return selection


def episode(task, tasks, seed, motor, actor, root):
    env = gym.make(
        task["environment"], render_mode="rgb_array" if actor else None,
        width=480, height=360,
    )
    try:
        observation, _ = env.reset(seed=seed)
        started = perf_counter()
        selection = task["skill"]
        if actor:
            image = root / "initial.png"
            Image.fromarray(env.render()).save(image)
            selection = select_skill(actor, task, tasks, image)
        row = {
            "task": task["skill"], "environment": task["environment"], "seed": seed,
            "skill": selection, "initial_observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
            "native_step_limit": env.spec.max_episode_steps, "native_dt_s": env.unwrapped.dt,
            "steps": 0, "success_steps": 0, "first_success_step": None,
            "final_success": False, "terminated": False, "truncated": False,
        }
        if selection != task["skill"]:
            # An incompatible skill cannot write to a different actuator layout.
            row["stop"] = "abstained" if selection == "abstain" else "incompatible_skill"
        else:
            with (root / "ticks.jsonl").open("x") as trace:
                for step in range(1, env.spec.max_episode_steps + 1):
                    action = motor.action(observation, env.action_space)
                    observation, reward, terminated, truncated, info = env.step(action)
                    success = bool(info["success"])
                    trace.write(json.dumps({
                        "step": step, "action": action.tolist(), "reward": float(reward),
                        "success": success, "terminated": terminated, "truncated": truncated,
                    }) + "\n")
                    row.update(steps=step, final_success=success, terminated=terminated, truncated=truncated)
                    row["success_steps"] += int(success)
                    if success and row["first_success_step"] is None:
                        row["first_success_step"] = step
                    if terminated or truncated:
                        row["stop"] = "terminated" if terminated else "truncated"
                        break
                else:
                    raise RuntimeError("Native environment did not stop at its registered limit")
        row["success"] = row["success_steps"] > 0
        row["elapsed_s"] = perf_counter() - started
        return row
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("motor", "agent"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    versions = {name: version(name) for name in protocol["runtime"]}
    if versions != protocol["runtime"]:
        raise SystemExit(f"Unexpected runtime: {versions}")
    gym.register_envs(gymnasium_robotics)
    tasks = protocol["tasks"]
    paths = {task["skill"]: args.weights / f"{task['skill']}.npz" for task in tasks}
    motors = {
        task["skill"]: Motor(paths[task["skill"]], task["checkpoint_sha256"]) for task in tasks
    }
    args.out.mkdir()
    (args.out / "protocol.json").write_bytes(protocol_bytes)
    run = {
        "mode": args.mode, "phase": "train", "host": socket.gethostname(), "versions": versions,
        "numpy": np.__version__,
        "weights_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    actor = None
    if args.mode == "agent":
        pilot = protocol["pilot"]
        actor = Actor(args.out / "actor", pilot["actor"], pilot["reasoning_effort"])
    rows = []
    for task in tasks:
        skill = task["skill"]
        for seed in protocol["seeds"]["train"]:
            root = args.out / f"{skill}-{seed}"
            root.mkdir()
            try:
                row = episode(task, tasks, seed, motors[skill], actor, root)
            except Exception as error:
                (root / "error.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
                raise
            (root / "episode.json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            rows.append(row)
    calls = actor.calls if actor else []
    usage = {key: sum(x["usage"][key] for x in calls) for key in ("input_tokens", "output_tokens")}
    summary = {
        "completed": len(rows), "successes": sum(row["success"] for row in rows),
        "actor_requests": len(calls), "actor_usage": usage, "episodes": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
