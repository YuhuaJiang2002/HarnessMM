"""Render the README token savings from the published coding comparisons."""
from functools import lru_cache
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SIZE = (1120, 440)
PAPER, INK, MUTED = "#F8F7F1", "#182D37", "#475D63"
TEAL, BASELINE, RULE = "#096C61", "#647980", "#CCD4D0"


@lru_cache(None)
def font(size, bold):
    names = ("Arial Bold.ttf", "DejaVuSans-Bold.ttf") if bold else ("Arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Install Arial or DejaVu Sans to render the figure")


def label(draw, x, y, value, size, color=INK, *, bold=False, anchor="lm"):
    draw.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)


def canvas():
    image = Image.new("RGB", SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    label(draw, 42, 39, "HarnessMM", 24, bold=True)
    draw.line((42, 69, 1078, 69), fill=RULE, width=2)
    return image, draw


def comparison(row, maximum):
    image, draw = canvas()
    label(draw, 1078, 39, f'{row["benchmark"]}  /  {row["tasks"]} tasks', 23, MUTED, anchor="rm")
    label(draw, 42, 134, f'{row["saving"]:.2f}%', 76, TEAL, bold=True)
    label(draw, 330, 140, "fewer execution tokens", 43, bold=True)
    label(draw, 42, 218, "Tokens/task (k)", 23, MUTED)
    for i, (name, value, color) in enumerate(zip(
        (row["harness"], "+ HarnessMM"), row["tokens"], (BASELINE, TEAL)
    )):
        y = 283 + 83 * i
        label(draw, 42, y, name, 22)
        end = 338 + round(630 * value / maximum)
        draw.rectangle((338, y - 13, end, y + 13), fill=color)
        label(draw, end + 14, y, f"{value:.2f}", 25, bold=True)
    return image


def overview(rows):
    image, draw = canvas()
    label(draw, 42, 123, "Execution tokens saved", 48, bold=True)
    label(draw, 42, 195, "Benchmark / harness", 21, MUTED)
    label(draw, 852, 195, "Tokens/task (k), before → after", 21, MUTED, anchor="rm")
    label(draw, 1078, 195, "Saving", 21, MUTED, anchor="rm")
    for i, row in enumerate(rows):
        y = 246 + 72 * i
        draw.line((42, y - 29, 1078, y - 29), fill=RULE)
        label(draw, 42, y, f'{row["benchmark"]} / {row["harness"]}', 24)
        before, after = row["tokens"]
        label(draw, 852, y, f"{before:.2f} → {after:.2f}", 27, anchor="rm")
        label(draw, 1078, y, f'{row["saving"]:.2f}%', 34, TEAL, bold=True, anchor="rm")
    return image


def render():
    rows = json.loads((HERE / "results.json").read_text())["main"]
    maximum = max(value for row in rows for value in row["tokens"])
    frames = [comparison(row, maximum) for row in rows]
    frames[0].save(HERE / "guarded-reuse.gif", save_all=True, append_images=frames[1:],
                   duration=3200, loop=0, optimize=True)
    overview(rows).save(HERE / "guarded-reuse.png", optimize=True)


if __name__ == "__main__":
    render()
