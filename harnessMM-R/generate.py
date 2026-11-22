"""Generate instruction-conditioned tabletop episodes in a fresh output directory."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
import tempfile
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from policies import can_continue, expert, parse_plan, qwen
from tabletop import Tabletop
from video import Recorder


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def split_counts(count):
    # Largest-remainder allocation at 20/20/60. Split whole scenes before execution.
    weights = [1, 1, 3]
    names = ["train", "validation", "test"]
    counts = [count*w // 5 for w in weights]
    ranked = sorted(range(3), key=lambda i: -(count * weights[i] % 5))
    for i in ranked[:count - sum(counts)]:
        counts[i] += 1
    return dict(zip(names, counts))


def population(seed, count, scene_names, shard_index, num_shards):
    offset = 0
    for split_index, (split, n) in enumerate(split_counts(count).items()):
        first = (shard_index - offset) % num_shards
        for ordinal in range(first, n, num_shards):
            yield {"id": f"{split}_{ordinal:06d}", "scene_seed": [seed, split_index, ordinal],
                   "split": split, "scene": scene_names[ordinal % len(scene_names)]}
        offset += n


def generate_episode(env, case, args, out):
    folder = out / case["id"]
    folder.mkdir()
    env.reset(case["scene_seed"])
    write_json(folder / "task.json", {**case, "instruction": env.instruction, "goals": env.goals,
                                    "block_order": env.blocks, "actions": env.actions})
    pending = []
    previous = ""
    feedback = "reset"
    done = False
    status = "step_limit"
    decisions = steps = ticks = 0
    image = folder / "000.png"
    Image.fromarray(env.rgb()).save(image)
    with ExitStack() as stack:
        record_video = case["id"] == args.video_episode
        if record_video:
            video = Recorder(env, folder)
            stack.callback(video.close)
            video.capture(None, "reset", None)
        steps_file = stack.enter_context((folder / "steps.jsonl").open("x"))
        control = stack.enter_context((folder / "control.jsonl").open("x"))
        for i in range(args.max_steps):
            obs = env.observe()
            reused = bool(pending and feedback == "executed" and can_continue(pending[0], previous, obs))
            source = "guarded_reuse" if reused else args.policy
            transfers = None
            if not reused:
                decisions += 1
                if args.policy == "expert":
                    pending = expert(env.goals, obs)
                else:
                    try:
                        if env.spec.get("occupied_pad") is not None:
                            transfers = env.transfers(obs)
                        reply = qwen(env.instruction, obs, image, feedback, args.model, args.port,
                                     env.actions, transfers)
                        reply.pop("__verbose", None)  # Server debug dump duplicates request settings.
                        write_json(folder / f"reply_{decisions:03}.json", reply)
                        pending = parse_plan(reply, args.model, env.actions)
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        write_json(folder / "policy_error.json", {"decision": decisions, "error": str(exc)})
                        status = "policy_error"
                        break
                plan = list(pending)
            else:
                plan = None
            action = pending.pop(0)
            callback = (lambda tick: video.after_tick(i, action, source)) if record_video else None
            feedback, trace = env.step(action, callback)
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
            if transfers is not None:
                record["planning_transfers"] = transfers
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
    p.add_argument("--scene-pack", type=Path, default=Path(__file__).with_name("scenes.json"))
    p.add_argument("--scenes", nargs="+", default=["sort-two"], help="cycle these recipes within each split")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--video-episode", help="record this episode during its original execution")
    p.add_argument("--resume", action="store_true", help="continue the same shard, preserving recorded episodes")
    args = p.parse_args()
    if args.seed < 0 or min(args.episodes, args.max_steps) < 1 or not 1 <= args.port <= 65535:
        p.error("expected a nonnegative seed, positive counts, and a valid TCP port")
    if not 0 <= args.shard_index < args.num_shards <= args.episodes:
        p.error("expected 0 <= shard-index < num-shards <= episodes")
    catalog = json.loads(args.scene_pack.read_text())
    if len(set(args.scenes)) != len(args.scenes) or set(args.scenes) - catalog.keys():
        p.error(f"choose distinct scenes from {list(catalog)}")
    cases = list(population(args.seed, args.episodes, args.scenes, args.shard_index, args.num_shards))
    selected = [c for c in cases if args.split in ("all", c["split"])]
    if args.video_episode:
        split, _, suffix = args.video_episode.rpartition("_")
        if (not suffix.isdigit() or int(suffix) >= split_counts(args.episodes).get(split, 0)
                or args.video_episode != f"{split}_{int(suffix):06d}" or args.split not in ("all", split)):
            p.error("video episode ID is outside the selected population")
    root = Path(__file__).parent
    hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest()
              for f in ["tabletop.xml", "tabletop.py", "generate.py", "policies.py", "scenes.py", "video.py"]}
    manifest = {"schema_version": 2, "seed": args.seed, "episode_count": args.episodes, "population": cases,
                "split": args.split, "shard_index": args.shard_index, "num_shards": args.num_shards,
                "scenes": args.scenes, "recipes": {name: catalog[name] for name in args.scenes},
                "policy": args.policy, "model": args.model if args.policy == "qwen" else None,
                "max_steps": args.max_steps, "video_episode": args.video_episode, "control_hz": 50,
                "sha256": hashes, "mujoco": mujoco.__version__, "numpy": np.__version__}
    out = args.out if args.num_shards == 1 else args.out / f"shard-{args.shard_index:05d}"
    if args.num_shards > 1:
        args.out.mkdir(parents=True, exist_ok=True)
        common = {k: v for k, v in manifest.items() if k not in ("population", "shard_index")}
        batch = args.out / "batch.json"
        # A complete immutable definition becomes visible to every concurrent worker at once.
        with tempfile.NamedTemporaryFile(dir=args.out, mode="w", suffix=".tmp") as tmp:
            json.dump(common, tmp, indent=2)
            tmp.flush()
            try:
                os.link(tmp.name, batch)
            except FileExistsError:
                if json.loads(batch.read_text()) != common:
                    p.error("output root belongs to another batch configuration")
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                p.error("resume requires the original shard configuration and source")
        else:
            write_json(path, manifest)
        counts = Counter()
        incomplete = []
        with ExitStack() as stack:
            environments = {}
            for case in selected:
                folder = out / case["id"]
                if (folder / "result.json").exists():
                    result = json.loads((folder / "result.json").read_text())
                elif folder.exists():
                    incomplete.append(case["id"])
                    continue
                else:
                    scene = case["scene"]
                    if scene not in environments:
                        environments[scene] = Tabletop(catalog[scene])
                        stack.callback(environments[scene].renderer.close)
                    result = generate_episode(environments[scene], case, args, out)
                    print(f"{case['id']} {result['status']} ({result['steps']} steps)", flush=True)
                counts[result["status"]] += 1
        write_json(out / "summary.json", {"scheduled": len(selected), "recorded": sum(counts.values()),
                   "outcomes": dict(counts), "incomplete": incomplete})
    return int(bool(incomplete or counts["policy_error"]))


if __name__ == "__main__":
    sys.exit(main())
