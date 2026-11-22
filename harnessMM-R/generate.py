"""Generate instruction-conditioned tabletop episodes in a fresh output directory."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from policies import can_continue, expert, parse_plan, qwen
from tabletop import BLOCKS, Tabletop


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def population(seed, count):
    # Largest-remainder allocation at 20/20/60. Split whole scenes before execution.
    weights = [1, 1, 3]
    names = ["train", "validation", "test"]
    counts = [count*w // 5 for w in weights]
    ranked = sorted(range(3), key=lambda i: -(count * weights[i] % 5))
    for i in ranked[:count - sum(counts)]:
        counts[i] += 1
    return [{"id": f"{split}_{ordinal:06d}", "scene_seed": [seed, split_index, ordinal], "split": split}
            for split_index, (split, n) in enumerate(zip(names, counts))
            for ordinal in range(n)]


def generate_episode(env, case, args):
    folder = args.out / case["id"]
    folder.mkdir()
    env.reset(case["scene_seed"])
    write_json(folder / "task.json", {**case, "instruction": env.instruction, "goals": env.goals})
    pending = []
    previous = ""
    feedback = "reset"
    done = False
    status = "step_limit"
    decisions = steps = ticks = 0
    image = folder / "000.png"
    Image.fromarray(env.rgb()).save(image)
    with (folder / "steps.jsonl").open("x") as steps_file, (folder / "control.jsonl").open("x") as control:
        for i in range(args.max_steps):
            obs = env.observe()
            reused = bool(pending and feedback == "executed" and can_continue(pending[0], previous, obs))
            source = "guarded_reuse" if reused else args.policy
            if not reused:
                decisions += 1
                if args.policy == "expert":
                    pending = expert(env.goals, obs)
                else:
                    try:
                        reply = qwen(env.instruction, obs, image, feedback, args.model, args.port)
                        write_json(folder / f"reply_{decisions:03}.json", reply)
                        pending = parse_plan(reply, args.model)
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        write_json(folder / "policy_error.json", {"decision": decisions, "error": str(exc)})
                        status = "policy_error"
                        break
                plan = list(pending)
            else:
                plan = None
            action = pending.pop(0)
            feedback, trace = env.step(action)
            for tick in trace:
                control.write(json.dumps({"step": i, **tick}, allow_nan=False) + "\n")
            after = env.observe()
            done = all(after["resting_on"][b] == p for b, p in env.goals.items()) and env.held is None
            next_image = folder / f"{i+1:03}.png"
            Image.fromarray(env.rgb()).save(next_image)
            record = {"step": i, "decision": decisions, "source": source, "authored_plan": plan,
                      "observation": obs, "rgb": image.name, "action": action, "feedback": feedback,
                      "control_start": ticks, "control_count": len(trace), "next_observation": after,
                      "next_rgb": next_image.name, "reward": int(done), "terminated": done,
                      "truncated": i + 1 == args.max_steps and not done}
            steps_file.write(json.dumps(record, allow_nan=False) + "\n")
            ticks += len(trace)
            steps += 1
            image, previous = next_image, action
            if done:
                status = "success"
                break
    result = {**case, "status": status, "success": done, "steps": steps,
              "decisions": decisions, "control_ticks": ticks, "final_rgb": image.name}
    write_json(folder / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10, help="size of the complete scene population")
    p.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--policy", choices=["expert", "qwen"], default="expert")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="Qwen3.8-27B-Q4_K_M")
    args = p.parse_args()
    if args.seed < 0 or min(args.episodes, args.max_steps) < 1 or not 1 <= args.port <= 65535:
        p.error("expected a nonnegative seed, positive counts, and a valid TCP port")
    cases = population(args.seed, args.episodes)
    selected = [c for c in cases if args.split in ("all", c["split"])]
    if not selected:
        p.error("this population contains no scenes in the requested split")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    sources = {f: hashlib.sha256((root / f).read_bytes()).hexdigest()
               for f in ["tabletop.xml", "tabletop.py", "generate.py", "policies.py"]}
    write_json(args.out / "manifest.json", {"schema_version": 1, "seed": args.seed, "population": cases,
               "split": args.split, "policy": args.policy, "model": args.model if args.policy == "qwen" else None,
               "max_steps": args.max_steps, "control_hz": 50, "block_order": BLOCKS, "sha256": sources,
               "mujoco": mujoco.__version__, "numpy": np.__version__})
    env = Tabletop()
    counts = Counter()
    try:
        for case in selected:
            result = generate_episode(env, case, args)
            counts[result["status"]] += 1
            print(f"{case['id']} {result['status']} ({result['steps']} steps)", flush=True)
            if result["status"] == "policy_error":
                break
    finally:
        env.renderer.close()
    summary = {"scheduled": len(selected), "recorded": sum(counts.values()), "outcomes": dict(counts)}
    write_json(args.out / "summary.json", summary)
    return int(counts["policy_error"] > 0)


if __name__ == "__main__":
    sys.exit(main())
