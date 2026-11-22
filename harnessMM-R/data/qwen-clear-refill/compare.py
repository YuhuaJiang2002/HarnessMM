"""Verify paired episodes and report full-population and split-level outcomes."""
import json
from collections import Counter
from pathlib import Path
import sys
import zipfile


def read(z, path):
    return json.loads(z.read(path))


def rows(z, path):
    return [json.loads(line) for line in z.read(path).splitlines()]


def measure(z, folder):
    task = read(z, folder + "task.json")
    result = read(z, folder + "result.json")
    steps = rows(z, folder + "steps.jsonl")
    control = rows(z, folder + "control.jsonl")
    assert len(steps) == result["steps"] <= 12 and len(control) == result["control_ticks"]
    success = result["success"]
    assert steps[-1]["terminated"] == success and steps[-1]["truncated"] != success
    final = steps[-1]["next_observation"]
    at_goal = final["holding"] is None and all(final["resting_on"][b] == p for b, p in task["goals"].items())
    assert at_goal == success
    ticks = decisions = tokens = rejected = 0
    for i, step in enumerate(steps):
        assert step["step"] == i and step["control_start"] == ticks
        if i:
            assert step["observation"] == steps[i-1]["next_observation"] and step["rgb"] == steps[i-1]["next_rgb"]
        for key in ("rgb", "next_rgb"):
            assert z.read(folder + step[key]).startswith(b"\x89PNG\r\n\x1a\n")
        if step["source"] == "qwen":
            decisions += 1
            reply = read(z, folder + f"reply_{decisions:03}.json")
            assert reply["model"] == "Qwen3.8-27B-Q4_K_M"
            plan = json.loads(reply["choices"][0]["message"]["content"])["actions"]
            assert plan == step["authored_plan"] and plan[0] == step["action"]
            tokens += reply["usage"]["total_tokens"]
        else:
            assert step["source"] == "guarded_reuse" and step["action"] == steps[i-1]["authored_plan"][1]
        end = ticks + step["control_count"]
        for j in range(ticks, end):
            assert control[j]["step"] == i and abs(control[j]["time_s"] - (.2+.02*(j+1))) < 1e-6
        assert abs(step["next_observation"]["time_s"] - (.2+.02*end)) < 1e-6
        rejected += int(step["control_count"] == 0)
        ticks = end
    assert ticks == len(control) and decisions == result["decisions"]
    counts = dict(episodes=1, successes=int(success), attempts=len(steps), zero_motion_attempts=rejected,
                  decisions=decisions, control_ticks=ticks, actor_tokens=tokens)
    return task, steps[0]["observation"], z.read(folder + "000.png"), counts


def compare(archive):
    pairs, totals = [], {}
    with zipfile.ZipFile(archive) as z:
        for cohort, expected, seed in (("original", 10, 20261122), ("fresh", 30, 20261123)):
            totals[cohort] = {}
            arms = {}
            for arm in ("before", "after"):
                prefix = f"{cohort}/{arm}/"
                arms[arm] = {p.split("/")[-2]: p[:-len("task.json")] for p in z.namelist()
                             if p.startswith(prefix) and p.endswith("/task.json")}
            expected_ids = {f"{split}_{ordinal:06d}" for split,n in (("train",6),("validation",6),("test",18))
                            for ordinal in range(n) if cohort == "fresh" or ordinal % 3 == 2}
            assert arms["before"].keys() == arms["after"].keys() == expected_ids and len(expected_ids) == expected
            for name in sorted(expected_ids):
                measurements = [measure(z, arms[arm][name]) for arm in ("before", "after")]
                before, after = measurements
                assert before[:3] == after[:3], f"initial conditions differ: {cohort}/{name}"
                task = before[0]
                assert task["scene_seed"][0] == seed and task["scene"] == "clear-and-refill"
                pair = {"cohort": cohort, "id": name, "split": task["split"]}
                for arm, measure_result in zip(("before", "after"), measurements):
                    pair[arm] = dict(path=arms[arm][name], **measure_result[-1])
                    for group in ("all", task["split"]):
                        totals[cohort].setdefault(group, {}).setdefault(arm, Counter()).update(measure_result[-1])
                pairs.append(pair)
    return pairs, totals


if __name__ == "__main__":
    root = Path(__file__).parent
    pairs, totals = compare(Path(sys.argv[1]) if len(sys.argv) > 1 else root / "trajectories.zip")
    (root / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pairs))
    (root / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    for cohort, groups in totals.items():
        for group, arms in groups.items():
            print(cohort, group, " ".join(f"{k} {v['successes']}/{v['episodes']}" for k,v in arms.items()))
