# HarnessMM-R

Generate instruction-conditioned pick-and-place demonstrations in MuJoCo. The default task has three colored blocks and two landing pads. Two additional recipes introduce a third destination and a pad that must be cleared before it can receive the requested block. The scene, camera and tool geometry are included.

| Initial observation | After the two placements |
| --- | --- |
| ![Overhead camera before sorting](example/train_000000/000.png) | ![Red block on the left pad and yellow block on the right pad](example/train_000000/004.png) |

The [example episode](example/train_000000/) contains five RGB frames, four high-level actions and 232 control ticks. The default expert runs entirely on a CPU. A second author uses a locally served Qwen3.8-27B vision model. Both use simulator-provided object poses and contact labels. In new runs Qwen receives synchronized overhead and tool-mounted wrist images. Earlier releases used the overhead image.

## Run the offline expert

From the repository root, on Linux with Python 3.11 and FFmpeg with libx264:

```sh
cd harnessMM-R
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
# For headless CPU rendering on Debian or Ubuntu:
sudo apt-get install libosmesa6-dev ffmpeg
export MUJOCO_GL=osmesa
python generate.py --episodes 10 --seed 17 --out outputs/expert
```

The expert reads the current state, finds an unfilled goal and proposes a pick followed by a placement. After the pick, the harness checks that the tool holds the intended block and the destination pad is free. If the check passes, it executes the queued placement. Otherwise the author sees a fresh observation. The CPU path runs offline after installation and requires no model weights or API key.

## Use Qwen on an H200

In a GPU allocation, start a CUDA build of llama.cpp with vision support. The verified setup used build `b12534`, the `Qwen3.8-27B-UD-Q4_K_M.gguf` weights and matching `mmproj-F16.gguf` from revision `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` of `unsloth/Qwen3.8-27B-GGUF`. Set `MODEL_DIR` to the directory containing those two files.

```sh
llama-server -m "$MODEL_DIR/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --mmproj "$MODEL_DIR/mmproj-F16.gguf" -ngl 99 --ctx-size 4096 \
  --parallel 1 --reasoning off --alias Qwen3.8-27B-Q4_K_M \
  --host 127.0.0.1 --port 8080
```

In another shell on the same host, with the Python environment active and `MUJOCO_GL=osmesa`:

```sh
python generate.py --policy qwen --port 8080 --episodes 10 --seed 17 \
  --split train --out outputs/qwen-train
```

Qwen receives the instruction, both current camera images, the simulator state and feedback from the last action. It returns up to two actions from the pick/place vocabulary. Each model reply is saved alongside the episode. The generator uses a loopback connection to the model server. [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

In a smoke test on the two training scenes for seed 17, the CPU expert and Qwen on a shared NVIDIA H200 each completed both scenes. Each episode used two authored plans and four actions.

## Generate a multi-scene batch

The [H200 release](data/qwen-tabletop-30/) includes 30 Qwen trajectories and a frame-indexed video of a complete three-block sorting episode.

Pass a list to `--scenes` to cycle through recipes within each split. The supplied [scene pack](scenes.json) covers:

| Recipe | Task |
| --- | --- |
| `sort-two` | Select two of three blocks and fill the left and right pads. |
| `sort-three` | Select three of four blocks and fill three pads. |
| `clear-and-refill` | Move a block from an occupied pad to its destination, then refill the vacated pad. |

For occupied-pad tasks, Qwen sees the current occupant of each requested destination. Before a pick, the executor checks that the selected block can be delivered to its assigned pad. After release it checks settled contact. See the [clear-and-refill comparison, paired trajectories and video](data/qwen-clear-refill/).

For another tabletop, copy `scenes.json` and add a named entry. Give each block a name and RGB triple in `blocks`, each pad a name and center in meters in `pads`, and each block a separate starting lane in `source_x_m`. For an occupied destination, add `"occupied_pad": "back"` using one of the pad names. Pass the file with `--scene-pack my-scenes.json --scenes my-layout`. The generator builds the MuJoCo bodies, action vocabulary, instructions, guard inputs and recorder from the recipe. Keep pads on the table, separate their centers by more than 86 mm and leave room for the randomized source blocks. These recipes share the Cartesian pick-and-place tool and contact-based placement reward.

Scale generation by assigning independent episode IDs to workers. This example starts three simulator processes against one local model server. For concurrent inference, the release used `--parallel 3 --ctx-size 12288` when starting llama.cpp.

```bash
set -e
pids=()
for shard in 0 1 2; do
  python generate.py --out outputs/sort-batch --episodes 30 --seed 20261122 \
    --scenes sort-three sort-two clear-and-refill --max-steps 12 \
    --policy qwen --video-episode train_000000 --num-shards 3 --shard-index "$shard" &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 ))
python index_batch.py outputs/sort-batch
```

Each worker writes one shard on a shared filesystem supporting hard links and advisory locks. To distribute inference across GPUs or hosts, run one worker command per shard with its local model-server port. Use the same model, scene list, seed, population and shard count. For independent storage, gather the shard directories and their identical `batch.json` under one directory before indexing. The index has one relative path per episode, including unsuccessful outcomes.

Add `--resume` to the original worker command after interruption. The worker checks the manifest, skips finished episodes and generates untouched IDs. An interrupted episode directory stays available for inspection and is listed as incomplete. Archiving that directory outside the shard allows a deliberate retry of just that ID. Use a new batch directory when changing the population or configuration.

With `--video-episode ID`, install FFmpeg with the `libx264` encoder on the recording worker. This option captures the live execution at 25 fps in `episode.mp4`. The left view shows the lift and release from the side, and the right view uses the agent's overhead camera. `frames.jsonl` maps frame numbers and video times to simulator time, control row, step and author. Frame zero is the reset observation, followed by one frame per two control ticks.

## Dual-view observations and Blender rendering

The [dual-view release](data/qwen-dual-view/) has synchronized overhead and wrist videos, tool calls, phase labels, and Cycles-rendered training images. Every new episode writes `overhead.mp4` and `wrist.mp4` at 25 fps. Qwen receives the lossless PNG pair at each planning call. Both cameras render after the same physics update; the wrist camera is rigidly mounted on the Cartesian end effector.

- `tool_calls.jsonl` records each simulated tool name, arguments, originating model decision, new or reused action, execution feedback, before/after holdings and pad contacts, and start/end timestamps. Control intervals are zero-based and half-open. A rejected action can begin and end on the same frame.
- `control.jsonl` labels the motion phase at each 50 Hz tick. `poses.jsonl` links each paired video frame to a tool call, control row, active grasp, object and tool poses, and the two camera-to-world transforms. The reset frame precedes control row zero. Body poses use meters and scalar-first quaternions. Camera-local axes are +X right, +Y up and −Z forward.
- `cameras.json` stores the pinhole intrinsics at the native resolution. `steps.jsonl` links each call’s before and after PNG pair, authored plan, reward and outcome. `task.json` includes the scene recipe for rendering.

Blender is an optional offline rendering stage. It uses the recorded rigid poses and camera transforms, a workshop HDRI, scanned wood maps, and a detailed model of the vacuum tool and Cartesian stage. Instructions cover full videos, individual frames and images at every action boundary. [Rendering guide](rendering/README.md)

## Read a trajectory

Each episode directory contains:

| File | Contents |
| --- | --- |
| `task.json` | Language instruction, block-to-pad assignments, scene seed and split. |
| `steps.jsonl` | One row per attempted high-level action, with before/after observations, RGB paths, the authored plan or reused action, execution feedback, reward and termination flags. |
| `control.jsonl` | One row per executed 20 ms tick. Each row gives the Cartesian tool target in meters, active suction attachment and measured positions of the blocks in the order listed in the episode’s `task.json` (`manifest.json` for the original example). |
| `000.png`, `001.png`, … | 320 × 240 overhead RGB at reset and after each high-level action. |
| `result.json` | Outcome, decision count, action count, control ticks and final frame. |

Read `control_start` and `control_count` in a step row to find its zero-based slice in `control.jsonl`. Lossless observation PNGs are saved at action boundaries. New runs also record the two continuous RGB streams at 25 fps. Control runs at 50 Hz and physics at 500 Hz. The tool is a Cartesian suction end effector with a switchable rigid attachment. After release, MuJoCo integrates gravity and contacts. A goal earns a terminal reward of one when all requested blocks are released, settled, touching the correct pads and within 24 mm of their centers on both horizontal axes.

The run-level `manifest.json` records the worker’s scene population, selected split, source hashes and simulator version. `--episodes` sets the population size before `--split` filters it. Scenes are assigned 20% to training, 20% to validation and 60% to testing, using largest-remainder rounding with ties in that order. Scene seeds combine the run seed, split index and within-split index, so increasing the population preserves existing scenes in their original splits.

Use a fresh output directory to start a run. Each shard’s `summary.json` counts every recorded outcome. A connection or reply error is saved in the affected episode’s `policy_error.json`. The worker continues with the remaining episodes and returns a nonzero exit code if any such error occurred.
