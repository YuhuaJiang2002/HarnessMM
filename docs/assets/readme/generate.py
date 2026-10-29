"""Rebuild README figures: python -m pip install matplotlib==3.11.2 pillow==13.0.0

Run this file from any directory. Data are published, rounded README values;
no raw experiments, new estimates, smoothing or uncertainty are introduced.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results.json").read_text())
NAVY = "#172B4D"
MUTED = "#526277"
TEAL = "#087F8C"
BLUE = "#65748B"
PALE = "#EAF6F6"
GRID = "#E6EBF0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": NAVY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": NAVY,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "path", "savefig.facecolor": "white",
})


def export(fig, name):
    fig.savefig(ROOT / f"{name}.svg")
    fig.savefig(ROOT / f"{name}.png", dpi=150)
    plt.close(fig)


def text(ax, x, y, value, size=12, color=NAVY, weight="normal", **kw):
    ax.text(x, y, value, fontsize=size, color=color, weight=weight,
            va="center", **kw)


def box(ax, x, y, w, h, fill="white", edge=GRID):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fill, edgecolor=edge, linewidth=1.2))


def arrow(ax, start, end, color=TEAL, style="arc3,rad=0", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=15, color=color, linewidth=1.7,
        connectionstyle=style, linestyle="--" if dashed else "-"))


def overview():
    fig, ax = plt.subplots(figsize=(14, 7.3))
    fig.subplots_adjust(left=.035, right=.965, top=.96, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    text(ax, .02, .93, "Reason once. Reuse when the receipt supports it.", 23, weight="bold")
    text(ax, .02, .865, "HarnessMM  /  guarded decision reuse through the native tool interface", 12, MUTED)
    xs = [.025, .36, .695]
    titles = ["Author the graph", "Check the receipt", "Execute the action"]
    tags = ["EDG", "GDR", "CCS"]
    bodies = ["The slow actor writes exact\nfuture tool actions and guards.",
              "A tool receipt supports\nan eligible stored action.",
              "The native tool runs the action;\nits callback substitution is recorded."]
    for x, title, tag, body in zip(xs, titles, tags, bodies):
        box(ax, x, .48, .28, .275, PALE if tag == "GDR" else "#F6F8FB")
        text(ax, x+.02, .71, tag, 11, TEAL, "bold")
        text(ax, x+.02, .65, title, 16, weight="bold")
        text(ax, x+.02, .555, body, 11, MUTED, linespacing=1.6)
    arrow(ax, (.31, .625), (.35, .625))
    arrow(ax, (.645, .625), (.685, .625))
    arrow(ax, (.835, .475), (.50, .475), style="arc3,rad=-.34")
    text(ax, .67, .35, "New receipt → check again", 11, TEAL, ha="center")
    arrow(ax, (.50, .78), (.165, .78), color=BLUE, dashed=True)
    text(ax, .335, .815, "New reasoning needed → resume the slow actor", 10, MUTED, ha="center")
    box(ax, .025, .035, .95, .205, "#F6F8FB")
    text(ax, .045, .195, "RECURSIVE SELF-IMPROVEMENT", 10, TEAL, "bold")
    for x, title, desc in [(.045, "Train", "Refine prompts and adapter rules"),
                            (.375, "Validate", "Select a frozen candidate"),
                            (.705, "Test", "Evaluate the frozen configuration")]:
        text(ax, x, .135, title, 15, weight="bold")
        text(ax, x, .08, desc, 10, MUTED)
    arrow(ax, (.30, .13), (.355, .13), color=BLUE)
    arrow(ax, (.63, .13), (.685, .13), color=BLUE)
    export(fig, "overview")


def style_axis(ax, xmax, xlabel):
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, labelpad=12, fontsize=12)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=9)


def main_results():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.255, right=.95, top=.73, bottom=.18, wspace=.19)
    fig.text(.045, .92, "Fewer tokens. Preserved or improved success.", fontsize=24, weight="bold")
    fig.text(.045, .855, "Three published comparisons · identical test tasks within each pair", fontsize=12, color=MUTED)
    for ax in axes:
        ax.set_ylim(-.7, 5.4)
    groups = [4.3, 2.3, .3]
    for row, y in zip(DATA["main"], groups):
        for i, (color, offset) in enumerate([(BLUE, .22), (TEAL, -.22)]):
            for ax, field in zip(axes, ["tokens", "success"]):
                value = row[field][i]
                ax.barh(y+offset, value, height=.33, color=color)
                ax.text(value+1.4, y+offset, f"{value:.2f}", va="center", fontsize=10, color=color, weight="bold")
    axes[0].set_yticks(groups, [f'{r["benchmark"]}\n{r["harness"]}\n{r["tasks"]} tasks' for r in DATA["main"]], fontsize=11)
    axes[1].set_yticks([])
    style_axis(axes[0], 112, "Tokens/task (k)")
    style_axis(axes[1], 112, "Success (%)")
    for ax in axes:
        ax.set_xticks([0, 25, 50, 75, 100])
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=TEAL)],
               ["Original harness", "HarnessMM"], loc="upper left", bbox_to_anchor=(.25, .82),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.255, .065, "Complete-task actor costs; offline optimization and validation reported separately.", fontsize=10, color=MUTED)
    export(fig, "benchmark-results")


def ablation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    fig.subplots_adjust(left=.18, right=.95, top=.68, bottom=.23, wspace=.22)
    fig.text(.045, .90, "Decision reuse and callback substitution both contribute.", fontsize=22, weight="bold")
    fig.text(.045, .81, "MBPP-sanitized · 257 test tasks · separate ablation comparison", fontsize=12, color=MUTED)
    for y, row in zip([2, 1, 0], DATA["ablation"]):
        color = TEAL if row["configuration"] == "HarnessMM" else BLUE
        for ax, field in zip(axes, ["tokens", "success"]):
            value = row[field]
            ax.barh(y, value, height=.50, color=color)
            ax.text(value+(.3 if field == "tokens" else 1.5), y, f"{value:.2f}",
                    va="center", fontsize=11, color=color, weight="bold")
    axes[0].set_yticks([2, 1, 0], [r["configuration"] for r in DATA["ablation"]])
    axes[1].set_yticks([])
    style_axis(axes[0], 21, "Tokens/task (k)")
    style_axis(axes[1], 112, "Success (%)")
    axes[1].set_xticks([0, 25, 50, 75, 100])
    fig.text(.18, .085, "GDR: Guarded Decision Reuse     ·     CCS: Certified Callback Substitution", fontsize=11, color=MUTED)
    export(fig, "ablation")


if __name__ == "__main__":
    overview()
    main_results()
    ablation()
