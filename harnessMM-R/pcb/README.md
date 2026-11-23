# ARX X5 PCB assembly dataset

[Watch episode 1 from four synchronized cameras](demo.mp4) or [return to the preview](../#arx-x5-pcb-assembly).

The generator rebuilds two scripted demonstrations from the supplied `arx_x5_pcb_assembly_code_and_videos.zip`. Each episode assembles an enclosure and a PCB, then inserts and tightens four M3 screws. The workcell uses inverse kinematics, rigid grasp attachments and a screw pitch model. The trajectories contain joint targets and geometry and have no measured force or torque.

## Generate episodes

Use Python 3.12. The requirements below use the versions pinned in the original bundle. The generator loads the simulation code and workcell assets from your copy of the archive.

```sh
cd harnessMM-R
python3.12 -m venv .venv-pcb
. .venv-pcb/bin/activate
python -m pip install -r pcb/requirements.txt
```

For video export, install FFmpeg with `libx264` and make both `ffmpeg` and `ffprobe` available on `PATH`. On a headless Linux host with OSMesa installed, use `MUJOCO_GL=osmesa`.

```sh
python pcb/generate.py --source /path/to/arx_x5_pcb_assembly_code_and_videos.zip \
  --out outputs/pcb-assembly
```

`--source` also accepts the extracted `arx_x5_pcb_assembly` directory. Add `--states-only` to generate the trajectories and run the geometry checks without encoding videos. Use a fresh output path for each run.

## Read the dataset

| File | Contents |
| --- | --- |
| `dataset.json` | Scene path and checksum, episode lengths, action units and the decoded frame count for each exported video. |
| `episode_*/trajectory.npz` | Timestamps, full MuJoCo state, joint and gripper observations and targets, tool and object poses, screw state and turns, and grasp attachments. |
| `episode_*/metadata.json` | Stage and assembly event indices, object and joint ordering, and the mapping from cameras to video files. |
| `geometry_validation.json`, `scene/fixture_bores.json` | Bore clearances, driver alignment and screw pickup geometry. |
| `episode_*/overhead.mp4`, `left_wrist.mp4`, `right_wrist.mp4` | Three synchronized 640 × 480 RGB streams per episode. |
| `episode_01/mounted_top_camera.mp4` | An additional camera fixed to the robot mount. |
| `episode_01_table_front.mp4`, `episode_01_synchronized_mounted_camera_views.mp4` | The front camera and four-view review for episode 1. The review combines front, mounted top and both wrist feeds. |

Frame `k` in a 15 fps camera stream matches trajectory row `k`. Frame `2k` in either 30 fps review matches the same row. Each action is the next sampled absolute target for both arms. The final row repeats the final target.

To inspect the generated scene in the MuJoCo viewer, run `python outputs/pcb-assembly/replay.py --episode 1`. Use `mjpython` in place of `python` on macOS.
