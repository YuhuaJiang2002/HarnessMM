"""Check alignment of a collected dual-view batch and optional Cycles keyframes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

VIEWS = ("overhead", "wrist")


def load_rows(file):
    return [json.loads(line) for line in file.read_text().splitlines()]


def stream_info(file):
    raw = subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
                                   "-show_entries","stream=width,height,nb_read_frames,r_frame_rate",
                                   "-of","json",str(file)])
    return json.loads(raw)["streams"][0]


def verify(root, frames_dir=None):
    totals = Counter()
    for item in load_rows(root/"index.jsonl"):
        d = root/item["path"]
        poses = load_rows(d/"poses.jsonl")
        calls = load_rows(d/"tool_calls.jsonl")
        control = load_rows(d/"control.jsonl")
        steps = load_rows(d/"steps.jsonl")
        calibration = json.loads((d/"cameras.json").read_text())
        task = json.loads((d/"task.json").read_text())
        assert calibration["views"]["wrist"]["camera_parent"] == "tool"
        assert len(poses) == 1 + len(control)//2 and len(steps) == len(calls) == item["steps"]
        assert len(control) == item["control_ticks"]
        for view in VIEWS:
            meta = stream_info(d/(view+".mp4"))
            assert int(meta["nb_read_frames"]) == len(poses) and meta["r_frame_rate"] == "25/1"
            assert [meta["width"],meta["height"]] == calibration["resolution"]
        top = np.array(poses[0]["camera_to_world"]["overhead"])
        wrist_offset = (np.array(poses[0]["camera_to_world"]["wrist"])[:3,3]
                        - poses[0]["body_xyz_quat_wxyz"]["tool"][:3])
        for i,row in enumerate(poses):
            assert row["frame"] == i and abs(row["video_s"] - i/25) < 1e-7
            assert np.allclose(row["camera_to_world"]["overhead"],top)
            # The Cartesian tool in these recipes translates at fixed orientation.
            wrist = np.array(row["camera_to_world"]["wrist"])
            tool = np.array(row["body_xyz_quat_wxyz"]["tool"][:3])
            assert np.allclose(wrist[:3,3] - tool,wrist_offset)
            assert np.allclose(wrist[:3,:3].T @ wrist[:3,:3],np.eye(3))
            if i == 0:
                assert row["control_row"] is None and row["phase"] == "reset"
                continue
            tick = control[2*i-1]
            assert row["control_row"] == 2*i-1 and row["time_s"] == tick["time_s"]
            assert row["phase"] == tick["phase"] and row["tool_call"] == tick["step"]
            assert row["action"] == steps[tick["step"]]["action"] and row["suction_block"] == tick["suction_block"]
            blocks = [row["body_xyz_quat_wxyz"][b][:3] for b in task["block_order"]]
            assert np.allclose(blocks,tick["block_xyz_m"])
        for step,call in zip(steps,calls):
            start,n = step["control_start"],step["control_count"]
            assert call["control_range"] == [start,start+n] and call["tool_call"] == step["step"]
            assert call["decision"] == step["decision"] and call["source"] == step["source"]
            assert call["feedback"] == step["feedback"]
            assert step["action"] == call["tool"]+" "+next(iter(call["arguments"].values()))
            assert call["time_s"] == [poses[call["before_frame"]]["time_s"],poses[call["after_frame"]]["time_s"]]
            for view in VIEWS:
                for key, boundary in (("views","before_frame"),("next_views","after_frame")):
                    with Image.open(d/step[key][view]) as img:
                        assert list(img.size) == calibration["resolution"]
                    if frames_dir:
                        jpg=frames_dir/item["path"]/view/f"{call[boundary]:05d}.jpg"
                        with Image.open(jpg) as img:
                            assert img.size == (480,360)
        totals.update(episodes=1,successes=int(item["success"]),tool_calls=len(calls),
                      control_ticks=len(control),synchronized_pairs=len(poses),
                      unique_action_boundaries=len({0} | {c["after_frame"] for c in calls}))
    return totals


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--keyframes",type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.root,a.keyframes),indent=2))
