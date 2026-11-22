"""Index all completed shards for training. Each index row points to one episode."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile


def read(path):
    return json.loads(path.read_text())


def index_batch(root):
    common = read(root / "batch.json")
    outcomes = Counter()
    scenes = Counter()
    splits = Counter()
    seen = set()
    expected_population = 0
    output = tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False)
    try:
        with output:
            for i in range(common["num_shards"]):
                shard = root / f"shard-{i:05d}"
                manifest = read(shard / "manifest.json")
                if manifest["shard_index"] != i or any(manifest[k] != v for k, v in common.items()):
                    raise ValueError(f"incompatible manifest in {shard.name}")
                summary = read(shard / "summary.json")
                population = manifest["population"]
                expected_population += len(population)
                counts = Counter()
                for case in population:
                    if case["id"] in seen:
                        raise ValueError(f"duplicate episode {case['id']}")
                    seen.add(case["id"])
                    if common["split"] not in ("all", case["split"]):
                        continue
                    directory = shard / case["id"]
                    result = read(directory / "result.json")
                    if any(result[k] != v for k, v in case.items()):
                        raise ValueError(f"episode identity mismatch in {directory}")
                    row = {**result, "path": directory.relative_to(root).as_posix()}
                    output.write(json.dumps(row) + "\n")
                    counts[result["status"]] += 1
                    scenes[case["scene"]] += 1
                    splits[case["split"]] += 1
                if (summary["incomplete"] or dict(counts) != summary["outcomes"] or
                        summary["scheduled"] != summary["recorded"] or summary["recorded"] != sum(counts.values())):
                    raise ValueError(f"incomplete or inconsistent shard {shard.name}")
                outcomes.update(counts)
        if expected_population != common["episode_count"]:
            raise ValueError("shards do not cover the requested scene population")
        os.replace(output.name, root / "index.jsonl")
    finally:
        Path(output.name).unlink(missing_ok=True)
    return {"episodes": sum(outcomes.values()), "outcomes": dict(outcomes), "scenes": dict(scenes),
            "splits": dict(splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(index_batch(parser.parse_args().root), indent=2))
