"""Render the GitHub teaser and reduced-motion poster with Pillow."""
from functools import lru_cache
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SIZE = (1120, 548)
PAPER, INK, MUTED, RULE = "#F8F7F1", "#182D37", "#52646A", "#CED4CC"
NAVY, PANEL, PALE = "#142E37", "#25434A", "#DDE7DE"
MINT, ORANGE, WHITE = "#B2EAC8", "#FFAD75", "#FFFFFF"
SANS = ("Arial.ttf", "DejaVuSans.ttf")
BOLD = ("Arial Bold.ttf", "DejaVuSans-Bold.ttf")
MONO = ("Menlo.ttc", "DejaVuSansMono.ttf")


@lru_cache(None)
def font(family, size):
    for name in family:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Install Arial and Menlo, or DejaVu Sans and DejaVu Sans Mono")


def label(draw, xy, value, size, fill=INK, family=SANS, anchor="lm"):
    draw.text(xy, value, font=font(family, size), fill=fill, anchor=anchor)


def mark(draw, x, y):
    for offset in (0, 13):
        draw.polygon([(x+offset, y+12), (x+offset+7, y),
                      (x+offset+18, y), (x+offset+11, y+12)], fill=INK)


def shell(section):
    image = Image.new("RGB", SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    mark(draw, 42, 32)
    label(draw, (84, 39), "HarnessMM", 24, family=BOLD)
    label(draw, (1078, 39), section, 16, MUTED, MONO, "rm")
    draw.line((42, 69, 1078, 69), fill=RULE, width=2)
    return image, draw


def check(draw, x, y, fill):
    draw.line([(x-6, y), (x-2, y+4), (x+6, y-5)], fill=fill, width=3)


def execution(stage, progress=1):
    image, draw = shell("GUARDED DECISION REUSE")
    label(draw, (42, 119), "One plan, three callbacks skipped.", 54, family=BOLD)
    label(draw, (44, 170), "The model plans the steps. Each tool result decides whether the next can run.", 24, MUTED)
    draw.rounded_rectangle((42, 210, 1078, 452), 20, fill=NAVY)
    label(draw, (70, 242), "example.py", 19, WHITE, MONO)
    label(draw, (1048, 242), "EXECUTION TRACE", 16, PALE, MONO, "rm")
    draw.line((70, 266, 1048, 266), fill=PANEL, width=2)
    steps = ("Prepare", "Compile", "Test", "Report")
    centers = (174, 428, 682, 936)
    for i in range(3):
        start, end = centers[i]+78, centers[i+1]-78
        draw.line((start, 327, end, 327), fill="#627C80", width=3)
        ran = 1 if stage > i+1 else progress if stage == i+1 else 0
        if ran:
            head = start+(end-start)*ran
            draw.line((start, 327, head, 327), fill=MINT, width=4)
            draw.ellipse((head-4, 323, head+4, 331), fill=MINT)
    for i, (x, name) in enumerate(zip(centers, steps)):
        active = i < stage or (i == stage and progress >= .95)
        color = (ORANGE if i == 0 else MINT) if active else PANEL
        draw.rounded_rectangle((x-78, 300, x+78, 354), 12, fill=color)
        label(draw, (x, 327), name, 25, NAVY if active else PALE, BOLD, "mm")
        if i == 0:
            label(draw, (x, 388), "root call", 18, PALE, MONO, "mm")
        elif active:
            check(draw, x-54, 388, MINT)
            label(draw, (x-39, 387), "guard met", 17, MINT, MONO)
        else:
            label(draw, (x, 388), "planned", 17, PALE, MONO, "mm")
    notes = [
        "The model calls Prepare and stores the next three actions.",
        "Prepare returned exit code 0. The harness runs Compile.",
        "Compile returned exit code 0. The harness runs Test.",
        "Test returned exit code 0. The harness runs Report.",
        "The plan is exhausted. Control returns to the model.",
    ]
    label(draw, (44, 497), notes[stage], 22, INK)
    count = min(3, stage) if progress >= .95 else max(0, stage-1)
    label(draw, (1077, 498), f"{count} / 3 callbacks skipped", 18, MUTED, MONO, "rm")
    return image


def outcome():
    row = next(row for row in json.loads((HERE / "results.json").read_text())["main"]
               if (row["benchmark"], row["harness"]) == ("HumanEval", "Codex"))
    image, draw = shell("HUMANEVAL  /  CODEX")
    label(draw, (42, 122), f'{row["saving"]:.2f}% fewer execution tokens', 52, family=BOLD)
    label(draw, (44, 178), f'Success held at {row["success"][0]:.2f}% across the same {row["tasks"]} test tasks.', 25, MUTED)
    draw.rounded_rectangle((42, 210, 1078, 452), 20, fill=NAVY)
    label(draw, (74, 247), "Tokens/task (k)", 20, PALE, MONO)
    baseline, reduced = row["tokens"]
    for i, (name, value, color) in enumerate(zip(("Codex", "+ HarnessMM"), (baseline, reduced), (PALE, MINT))):
        y = 313+i*77
        label(draw, (74, y), name, 24, WHITE, BOLD)
        length = round(638*value/baseline)
        draw.rounded_rectangle((296, y-14, 296+length, y+14), 5, fill=color)
        label(draw, (314+length, y), f"{value:.2f}", 23, WHITE, MONO)
    label(draw, (44, 497), "Complete-task input and output tokens. Offline optimization is separate.", 22, MUTED)
    return image


def render():
    frames, durations = [], []
    def add(image, ms):
        frames.append(image)
        durations.append(ms)
    add(execution(0), 2100)
    for stage in range(1, 4):
        for index in range(1, 8):
            add(execution(stage, index/8), 80)
        add(execution(stage), 1650)
    add(execution(4), 2500)
    result = outcome()
    add(result, 4300)
    execution(4).save(HERE / "guarded-reuse.png", optimize=True)
    frames[0].save(HERE / "guarded-reuse.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)


if __name__ == "__main__":
    render()
