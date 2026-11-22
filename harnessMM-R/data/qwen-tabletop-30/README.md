# 30 Qwen tabletop trajectories

[Watch the three-block episode](train_000000/episode.mp4) · [Browse its observations and actions](train_000000/) · [Download all 30 episodes](trajectories.zip) · [Browse the episode index](index.jsonl)

Qwen3.8-27B used the overhead RGB image and simulator state to plan pick-and-place actions on a shared NVIDIA H200. Three MuJoCo workers collected 30 episodes with 164 model replies, 224 high-level action attempts and 7,482 control ticks. The archive includes the outcome of every scheduled episode.

| Scene | Episodes | Completed the placement goals | Reached the 12-action limit |
| --- | ---: | ---: | ---: |
| Two-pad sorting | 10 | 9 | 1 |
| Three-pad sorting | 10 | 9 | 1 |
| Clear and refill an occupied pad | 10 | 2 | 8 |
| **Total** | **30** | **20** | **10** |

The population has 6 training, 6 validation and 18 test scenes, with all three recipes represented equally in each split. These counts describe the collected data. For successful demonstrations, filter the index by `success`. The `path` field is relative to the extracted archive root.

## Video and alignment

The first training episode was designated for recording before generation. The tool moves the red block to the left pad, the blue block to the center pad and the yellow block to the right pad. The green block stays on the table. Qwen authored three plans. The harness checked the observed grasp and destination before executing each queued placement.

The 7.00-second H.264 video shows the live simulator from the side and overhead at 25 fps. All 175 frames are mapped to simulator time in [frames.jsonl](train_000000/frames.jsonl). Frame zero is the reset observation. Frame 1 corresponds to zero-based control row 1, and frame 174 to row 347. Each frame entry also identifies the action step. The [six action records](train_000000/steps.jsonl) link the before/after RGB images, Qwen decisions and offsets into the [50 Hz control trace](train_000000/control.jsonl). `reply_001.json` through `reply_003.json` contain the model answers and token usage.

## Generation

Run the [three-worker command](../../README.md#generate-a-multi-scene-batch) to generate the same scene population. This batch used seed `20261122`, MuJoCo 3.17.0, NumPy 1.26.4, Pillow 11.3.0, OSMesa, and llama.cpp build `b12534-ecafc6714`. The server used three slots, a 12,288-token context across the slots and GPU offload for all 66 model layers. Collection took 97.48 seconds from worker launch through the last worker exit, including video recording. One other GPU process occupied 34,378 MiB on the allocated H200 before loading Qwen.

| Artifact | SHA-256 |
| --- | --- |
| Qwen3.8-27B-UD-Q4_K_M.gguf | `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529` |
| mmproj-F16.gguf | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` |
| trajectories.zip | `d54dbfd0d5ab23bd4a4c5e4d8a192d78e751a2c0c60e6324d6675f13f007d690` |
