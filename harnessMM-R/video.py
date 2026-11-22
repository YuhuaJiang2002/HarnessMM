"""Record the live simulator at 25 fps and index each frame against control.jsonl."""
import json
import subprocess
from pathlib import Path

import mujoco as mj
from PIL import Image, ImageDraw, ImageFont


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


class Recorder:
    def __init__(self, env, folder):
        self.env = env
        self.tick = 0
        self.frame = 0
        self.camera = mj.MjvCamera()
        self.camera.lookat[:] = [0, .005, .042]
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 92, -29, .68
        self.large, self.small = font(23), font(16)
        self.encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x480", "-r", "25",
            "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "17",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(Path(folder) / "episode.mp4")],
            stdin=subprocess.PIPE)
        self.index = (Path(folder) / "frames.jsonl").open("x")

    def capture(self, step, action, source):
        env = self.env
        view = Image.fromarray(env.rgb(self.camera)).resize((576, 432), Image.Resampling.LANCZOS)
        overhead = Image.fromarray(env.rgb())
        canvas = Image.new("RGB", (960, 480), "#18242a")
        canvas.paste(view, (0, 0))
        canvas.paste(overhead, (606, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((606, 7), "Agent camera", font=self.small, fill="white")
        draw.text((606, 286), "Placement goals", font=self.large, fill="white")
        for i, (block, pad) in enumerate(env.goals.items()):
            draw.text((608, 324 + i*23), f"{block.title()}  to  {pad} pad", font=self.small, fill="#d9e7e5")
        draw.text((16, 441), action.title(), font=self.large, fill="white")
        draw.text((345, 448), f"Simulation  {env.data.time:.2f} s", font=self.small, fill="#d9e7e5")
        self.encoder.stdin.write(canvas.tobytes())
        self.index.write(json.dumps({"frame": self.frame, "video_s": self.frame / 25,
                         "time_s": round(env.data.time, 6), "control_row": self.tick - 1 if self.tick else None,
                         "step": step, "action": action, "source": source}) + "\n")
        self.frame += 1

    def after_tick(self, step, action, source):
        self.tick += 1
        if self.tick % 2 == 0:
            self.capture(step, action, source)

    def close(self):
        try:
            self.encoder.stdin.close()
        finally:
            self.index.close()
            if self.encoder.wait():
                raise RuntimeError("FFmpeg could not finish the episode video")
