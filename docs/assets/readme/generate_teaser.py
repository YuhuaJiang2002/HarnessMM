"""Render the README demonstration and its reduced-motion summary."""

from functools import lru_cache
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
SIZE = (1200, 600)
LEFT, RIGHT = 48, 1152
PAPER, WHITE, INK, MUTED = "#F7F8F5", "#FFFFFF", "#18282D", "#486064"
TEAL, MINT = "#075F54", "#DDF1E9"
BLUE, ICE = "#344E7A", "#E7EDF7"
RULE, RAIL = "#D3DEDA", "#879B96"
LANE_X = (48, 332, 616, 900)

SIMPLE_STEPS = (
    ("Model call", "Plans the tool work"),
    ("Compile", "Tool returns exit code 0"),
    ("Model callback", "Decides what to do next"),
    ("Test", "Native terminal executes"),
)
REUSE_STEPS = (
    ("Authored graph", "Compile → test → report"),
    ("Compile receipt", "return_code = 0"),
    ("GDR checks guard", "Next action is eligible"),
    ("CCS dispatches test", "test example.py"),
)


@lru_cache(None)
def font(size, bold):
    names = ("Arial Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("Arial.ttf", "arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Install Arial or DejaVu Sans to render the figure")


def label(draw, x, y, value, size, color=INK, *, bold=False, anchor="lm"):
    draw.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)


def canvas(section, title, subtitle):
    image = Image.new("RGB", SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    label(draw, LEFT, 37, "HarnessMM", 28, bold=True)
    label(draw, RIGHT, 37, section, 21, MUTED, anchor="rm")
    draw.line((LEFT, 68, RIGHT, 68), fill=RULE, width=2)
    label(draw, LEFT, 119, title, 49, bold=True)
    label(draw, LEFT, 174, subtitle, 24, MUTED)
    return image, draw


def card(draw, x, y, width, height, title, detail, *, tone=INK, fill=WHITE, edge=RULE):
    draw.rounded_rectangle((x, y, x + width, y + height), radius=13,
                           fill=fill, outline=edge, width=2)
    label(draw, x + 16, y + height // 2 - 15, title, 22, tone, bold=True)
    label(draw, x + 16, y + height // 2 + 19, detail, 20, MUTED)


def arrow(draw, x0, x1, y, color=RAIL):
    draw.line((x0, y, x1 - 8, y), fill=color, width=3)
    draw.polygon(((x1 - 8, y - 6), (x1, y), (x1 - 8, y + 6)), fill=color)


def lane(draw, y, steps, *, active, tone, fill):
    for index, (x, (title, detail)) in enumerate(zip(LANE_X, steps)):
        focused = index == active
        if index:
            arrow(draw, x - 26, x - 9, y + 44, tone if focused else RAIL)
        card(draw, x, y, 252, 88, title, detail,
             tone=tone if focused else INK, fill=fill if focused else WHITE,
             edge=tone if focused else RULE)


def execution(stage, active):
    titles = {
        "baseline": "The model decides again after compilation",
        "author": "Write the next actions in one call",
        "guard": "A tool receipt unlocks the next action",
        "execute": "Run the tool and skip a model call",
    }
    image, draw = canvas("Controller example · synthetic", titles[stage],
                         "One task: compile, test and report example.py.")
    label(draw, LEFT, 232, "Simple harness", 22, BLUE, bold=True)
    lane(draw, 255, SIMPLE_STEPS, active=active if stage == "baseline" else None, tone=BLUE, fill=ICE)

    if stage == "baseline":
        draw.rounded_rectangle((LEFT, 386, RIGHT, 479), radius=13, fill=ICE)
        label(draw, LEFT + 20, 415, "After the compile receipt", 20, BLUE, bold=True)
        label(draw, LEFT + 20, 451, "The harness asks the model which action to run next.", 28, INK)
        return image

    label(draw, LEFT, 379, "Same task with HarnessMM", 22, TEAL, bold=True)
    lane(draw, 402, REUSE_STEPS, active=active, tone=TEAL, fill=MINT)
    if stage == "author":
        label(draw, LEFT, 543, "An Executable Decision Graph stores exact actions, dependencies and guards.", 24, MUTED)
    elif stage == "guard":
        label(draw, LEFT, 543, "Guarded Decision Reuse checks the compile receipt. The model handles unsupported results.", 23, MUTED)
    else:
        label(draw, LEFT, 525, "Certified Callback Substitution", 22, TEAL, bold=True)
        label(draw, LEFT, 560, "Test receipt recorded  ·  Bypassed model callback: callback-1", 24, INK)
    return image


def improvement(active=None):
    image, draw = canvas("Harness optimization", "Improve the harness with training feedback",
                         "Recursive self-improvement (RSI) iterates on the current configuration.")
    steps = (
        (LEFT, "Run training tasks", "Measure success and token use"),
        (432, "Edit graph and rules", "Revise nodes, edges or prompts"),
        (816, "Evaluate the candidate", "Apply the acceptance criteria"),
    )
    for i, (x, title, detail) in enumerate(steps):
        focused = active is None or active == i
        card(draw, x, 252, 336, 105, title, detail,
             tone=TEAL if focused else INK, fill=MINT if focused else WHITE)
        if i:
            arrow(draw, x - 38, x - 10, 304, TEAL if focused else RAIL)
    loop_color = TEAL if active is None else RAIL
    draw.line(((984, 357), (984, 427), (216, 427), (216, 366)), fill=loop_color, width=3)
    draw.polygon(((210, 373), (216, 364), (222, 373)), fill=loop_color)
    draw.rounded_rectangle((347, 408, 853, 447), radius=10, fill=PAPER)
    label(draw, 600, 427, "An accepted configuration starts the next round", 22, TEAL, bold=True, anchor="mm")
    label(draw, LEFT, 527, "Validation selects the deployment configuration. Model weights stay fixed.", 24, MUTED)
    return image


def results_table(draw, top, rows):
    label(draw, LEFT, top, "Benchmark and harness", 22, MUTED, bold=True)
    label(draw, 580, top, "Tokens/task (k)", 22, MUTED, bold=True, anchor="mm")
    label(draw, 833, top, "Success (%)", 22, MUTED, bold=True, anchor="mm")
    label(draw, RIGHT - 12, top, "Tokens saved", 22, TEAL, bold=True, anchor="rm")

    for index, row in enumerate(rows):
        y = top + 50 + 68 * index
        draw.line((LEFT, y - 26, RIGHT, y - 26), fill=RULE, width=2)
        label(draw, LEFT, y - 4, row["benchmark"], 24, bold=True)
        label(draw, LEFT, y + 21, f'{row["harness"]} · {row["tasks"]} tasks', 20, MUTED)
        before, after = row["tokens"]
        label(draw, 580, y + 4, f"{before:.2f} → {after:.2f}", 28, bold=True, anchor="mm")
        before, after = row["success"]
        label(draw, 833, y + 4, f"{before:.2f} → {after:.2f}", 25, anchor="mm")
        draw.rounded_rectangle((976, y - 18, RIGHT, y + 27), radius=12, fill=MINT)
        label(draw, RIGHT - 12, y + 4, f'{row["saving"]:.2f}%', 31, TEAL, bold=True, anchor="rm")


def results(rows, actor, *, reduced_motion=False):
    image, draw = canvas("Held-out coding results", "Fewer tokens at equal or higher success",
                         f"{actor} · Original harness → harness with HarnessMM")
    if reduced_motion:
        steps = (
            (LEFT, "Author", "Store actions in a graph"),
            (432, "Guard", "Check tool receipts with GDR"),
            (816, "Execute", "Run with CCS and log the bypass"),
        )
        for i, (x, title, detail) in enumerate(steps):
            card(draw, x, 207, 336, 65, title, detail, tone=TEAL, fill=MINT)
            if i:
                arrow(draw, x - 38, x - 10, 240, TEAL)
        label(draw, LEFT, 298, "RSI uses training feedback to refine the graph and execution rules.", 22, MUTED)
        top = 337
    else:
        top = 252

    results_table(draw, top, rows)
    label(draw, LEFT, 574, "Test execution includes actor input and output. Offline optimization and validation are separate.",
          20, MUTED)
    return image


def render():
    data = json.loads((HERE / "results.json").read_text())
    rows, actor = data["main"], data["actor"]
    timeline = (
        (execution("baseline", 0), 500),
        (execution("baseline", 1), 650),
        (execution("baseline", 2), 2350),
        (execution("author", 0), 2850),
        (execution("guard", 1), 650),
        (execution("guard", 2), 2850),
        (execution("execute", 3), 4050),
        (improvement(0), 550),
        (improvement(1), 550),
        (improvement(2), 550),
        (improvement(), 3200),
        (results(rows, actor), 6600),
    )
    frames, durations = zip(*timeline)
    frames[0].save(HERE / "guarded-reuse.gif", save_all=True, append_images=frames[1:],
                   duration=list(durations), loop=0, optimize=True)
    results(rows, actor, reduced_motion=True).save(HERE / "guarded-reuse.png", optimize=True)


def title_card():
    image = Image.new("RGB", SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    cx = SIZE[0] // 2
    face = font(88, True)
    first, second = "Harness", "MinusMinus"
    x = cx - (draw.textlength(first, font=face) + draw.textlength(second, font=face)) / 2
    draw.text((x, 191), first, font=face, fill=INK, anchor="lm")
    draw.text((x + draw.textlength(first, font=face), 191), second,
              font=face, fill=TEAL, anchor="lm")
    draw.line((cx - 36, 263, cx + 36, 263), fill=TEAL, width=3)
    label(draw, cx, 320, "Recursive Self-Improvement for Harness Token Efficiency",
          33, INK, anchor="mm")
    label(draw, cx, 393, "Minus Overhead, Minus Cost", 26, MUTED, anchor="mm")
    left, right = "Tsinghua University", "Microsoft Research Asia"
    face = font(24, False)
    left_width = 58 + 12 + draw.textlength(left, font=face)
    right_width = 42 + 12 + draw.textlength(right, font=face)
    x = round((SIZE[0] - left_width - 65 - right_width) / 2)
    with Image.open(HERE / "tsinghua-emblem.png") as source:
        emblem = source.convert("RGBA")
        emblem.thumbnail((58, 58), Image.Resampling.LANCZOS)
        image.paste(emblem, (x, 495 - emblem.height // 2), emblem)
    label(draw, x + 70, 495, left, 24, INK)
    x += round(left_width) + 65
    for dx, dy, color in ((0, 0, "#F25022"), (22, 0, "#7FBA00"),
                          (0, 22, "#00A4EF"), (22, 22, "#FFB900")):
        draw.rectangle((x + dx, 474 + dy, x + dx + 19, 493 + dy), fill=color)
    label(draw, x + 54, 495, right, 24, INK)
    return image


def split_readme_animations():
    with Image.open(HERE / "guarded-reuse.gif") as original:
        frames, durations = [], []
        for i in range(original.n_frames):
            original.seek(i)
            frames.append(original.convert("RGB"))
            durations.append(original.info["duration"])

    def save(name, scenes, delays):
        scenes[0].save(HERE / name, save_all=True, append_images=scenes[1:],
                       duration=delays, loop=0, optimize=True)

    with Image.open(HERE / "harnessmm-six-benchmark-results.png") as chart:
        benchmarks = ImageOps.pad(chart.convert("RGB"), SIZE,
                                  method=Image.Resampling.LANCZOS, color=WHITE)
    save("rsi-results.gif", [title_card(), *frames[7:10], frames[11], benchmarks],
         [1200, 1000, 1000, 1000, 2600, 3200])
    with Image.open(HERE / "overview.png") as overview:
        final = ImageOps.pad(overview.convert("RGB"), SIZE,
                             method=Image.Resampling.LANCZOS, color=WHITE)
    save("guarded-reuse-overview.gif", [*frames[:7], final], [*durations[:7], 6500])


if __name__ == "__main__":
    render()
    split_readme_animations()
