# Clearing and refilling an occupied pad

[Successful refill video](train_000000/episode.mp4) · [Video observations and actions](train_000000/) · [All paired trajectories](trajectories.zip) · [Episode pairs](pairs.jsonl)

## What happened in training

In both original training episodes, Qwen picked the incoming red block while the back pad was occupied. The occupant was blue in `train_000002` and green in `train_000005`. Once the tool held red, it could no longer lift the occupant. The executor correctly refused all 11 subsequent attempts to place red on the back pad in each episode. Those 22 refusals consumed action attempts and produced zero control ticks. The original reuse guard checked the pad after the pick, too late to prevent this sequence.

After the candidate was frozen, a readback of all eight original failures confirmed the same first pick: the incoming block was lifted while its destination was occupied. In every case it was the episode’s only action that moved the tool.

The revised planner input names the observed occupant of each requested destination and explains the one-block carrying constraint. Qwen still chooses the actions. Before accepting a pick, the executor checks that the block needs moving and its assigned destination is free. A rejected pick returns the blocking block and pad to the next planning call while the tool is still empty. Placement checks the held block’s assigned destination, and release is followed by a measured settled-contact check. Only occupied-pad recipes use these changes. The same action vocabulary, motion primitives and contact-based success criterion apply to both versions.

## Paired results

| Population | Original | Revised | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Original 10 layouts | 2/10 (20.0%) | 10/10 (100.0%) | +80.0 |
| 30 fresh layouts | 10/30 (33.3%) | 30/30 (100.0%) | +66.7 |
| Held-out subset of the fresh layouts | 3/18 (16.7%) | 18/18 (100.0%) | +83.3 |

The original 10 include the two training episodes used for diagnosis. The separate 30-scene population has six training, six validation and 18 held-out scenes. Each pair starts with identical block positions, goal assignments, contact observations and overhead image bytes. Both versions ran once per layout with Qwen3.8-27B Q4_K_M on a shared NVIDIA H200, using up to 12 action attempts and 96 output tokens per model reply. Both the program and population were frozen before collection. [Protocol](protocol.json) · [Counts by split and version](summary.json)

Across the fresh batch, zero-motion attempts fell from 220 to 13. Qwen authored 262 plans before the change and 73 afterward. Complete model input and output per episode, including rejected attempts, measured 4.72 and 1.47 Tokens/task (k), respectively.

## Read the video and data

The designated training episode begins with blue on the back pad. Qwen issued two plans: move blue to the front pad, then move green to the back pad. The four actions generated 232 control ticks at 50 Hz. The video includes reset and 116 subsequent frames at 25 fps. Each [frame entry](train_000000/frames.jsonl) identifies its action, control row and simulation time. [High-level steps](train_000000/steps.jsonl) · [Control trajectory](train_000000/control.jsonl)

The archive contains 80 episode records. `original/before` contains the ten occupied-pad trajectories from the original release; `original/after` contains ten new executions with the same scene IDs. `fresh/before` and `fresh/after` contain 30 new trajectories each. All recorded outcomes, observations, actions, Qwen replies, usage and simulation control traces are included. Each arm contains its generator manifests; the two original-population folders contain the occupied-pad shard of the parent three-scene batch. Paths in `pairs.jsonl` are relative to the archive root.

## Reproduce the comparison

Run `python compare.py` in this directory to check the initial observations, action and control alignment, model responses and terminal rewards, and regenerate the pair index and summary. Run the generation commands below in `harnessMM-R` with the [local Qwen server and simulator](../../README.md#use-qwen-on-an-h200) active. For the original policy use the code at `d870241`, and for the revised policy use this release. Give each arm a fresh output directory.

```bash
set -e
: "${OUT:?Set OUT to a fresh output directory for this arm}"
pids=()
for shard in 0 1 2; do
  python generate.py --policy qwen --episodes 30 --seed 20261123 --max-steps 12 \
    --scenes clear-and-refill --num-shards 3 --shard-index "$shard" \
    --video-episode train_000000 --out "$OUT" &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 ))
python index_batch.py "$OUT"
```

For the original ten layouts use seed `20261122`, the scene list `sort-three sort-two clear-and-refill`, and only shard index 2 of 3. Both arms share the same source scene recipe and model weights. GPU, runtime and model identities are recorded in the archive’s `runtime.json` and `protocol.json`.

Archive SHA-256: `1a80f722da13b02169b02e4be811646e4de1f09747d9d759b0c24f83e2a75a9f`.
