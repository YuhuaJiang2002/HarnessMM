"""Synchronized camera observations, poses and tool-use annotations."""
from contextlib import ExitStack
import json
import math
import subprocess

import numpy as np
from PIL import Image

VIEWS = ("overhead", "wrist")


def png_pair(env, folder, step):
    paths = {}
    for name in VIEWS:
        filename = f"{step:03}.png" if name == "overhead" else f"wrist_{step:03}.png"
        Image.fromarray(env.rgb(name)).save(folder / filename)
        paths[name] = filename
    return paths


def finish_encoder(process):
    try:
        process.stdin.close()
    finally:
        if process.wait():
            raise RuntimeError("FFmpeg failed while encoding synchronized camera data")


class Observations:
    def __init__(self, env, folder, stack: ExitStack):
        self.env = env
        self.frame = self.tick = 0
        self.states = stack.enter_context((folder / "poses.jsonl").open("x"))
        self.calls = stack.enter_context((folder / "tool_calls.jsonl").open("x"))
        self.encoders = {}
        calibration = {"resolution": [320, 240], "fps": 25, "units": "meters", "pixel_centers": "integer",
                       "camera_axes": "+X right, +Y up, -Z forward", "views": {}}
        for name in VIEWS:
            cam = env.model.camera(name)
            fy = 120 / math.tan(math.radians(float(cam.fovy[0])) / 2)
            calibration["views"][name] = {"K": [[fy, 0, 159.5], [0, fy, 119.5], [0, 0, 1]],
                                          "camera_parent": env.model.body(int(cam.bodyid[0])).name}
            proc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "320x240", "-framerate", "25", "-i", "-", "-an",
                "-c:v", "libx264", "-threads", "1", "-crf", "16", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(folder / f"{name}.mp4")], stdin=subprocess.PIPE)
            stack.callback(finish_encoder, proc)
            self.encoders[name] = proc
        (folder / "cameras.json").write_text(json.dumps(calibration, indent=2) + "\n")
        self.capture(None, "reset", None)

    def capture(self, step, phase, action):
        env = self.env
        transforms = {}
        for name in VIEWS:
            cam = env.data.camera(name)
            transform = np.eye(4)
            transform[:3,:3] = cam.xmat.reshape(3,3)
            transform[:3,3] = cam.xpos
            transforms[name] = transform.tolist()
            self.encoders[name].stdin.write(env.rgb(name).tobytes())
        poses = {b: [*env.data.body(b).xpos.tolist(), *env.data.body(b).xquat.tolist()]
                 for b in ("tool", *env.blocks)}
        row = {"frame": self.frame, "time_s": round(env.data.time, 6), "video_s": self.frame/25,
               "control_row": self.tick-1 if self.tick else None, "tool_call": step, "action": action,
               "phase": phase, "suction_block": env.held, "body_xyz_quat_wxyz": poses,
               "camera_to_world": transforms}
        self.states.write(json.dumps(row, allow_nan=False) + "\n")
        self.frame += 1

    def after_tick(self, tick, step, action):
        self.tick += 1
        if self.tick % 2 == 0:
            self.capture(step, tick["phase"], action)

    def label_call(self, step, action, source, decision, feedback, start_tick, before, after):
        name, argument = action.split()
        row = {"tool_call": step, "tool": name, "arguments": {"block" if name=="pick" else "pad": argument},
               "source": source, "decision": decision, "feedback": feedback,
               "time_s": [before["time_s"], after["time_s"]], "control_range": [start_tick, self.tick],
               "before_frame": start_tick//2, "after_frame": self.tick//2, "holding_before": before["holding"],
               "holding_after": after["holding"], "resting_before": before["resting_on"],
               "resting_after": after["resting_on"]}
        self.calls.write(json.dumps(row) + "\n")
