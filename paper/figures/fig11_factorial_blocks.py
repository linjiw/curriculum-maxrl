#!/usr/bin/env python
"""Independent seed-block view of the registered maze confirmation."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(
    HERE, "..", "..", "curriculum_maxrl", "maze_gpu_factorial",
    "block_reanalysis.json")

BLUE = "#3b76af"
ORANGE = "#e67e3d"
BLACK = "#222222"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

with open(SRC, encoding="utf-8") as handle:
    data = json.load(handle)
rows = data["waves"]["wave2"]["blocks"]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45))
panels = [
    ("cov_auc_maxrl_minus_grpo", "cov_auc_sampler_average",
     "registered primary: coverage AUC", "MaxRL − GRPO coverage AUC"),
    ("easy_band_maxrl_minus_grpo", "easy_band_sampler_average",
     "registered secondary: easy band", "MaxRL − GRPO endpoint change"),
]

for panel_index, (values_key, average_key, title, ylabel) in enumerate(panels):
    ax = axes[panel_index]
    for position, row in enumerate(rows):
        uniform = row[values_key]["uniform"]
        frontier = row[values_key]["frontier_un"]
        average = row[average_key]
        ax.plot([position - 0.10, position + 0.10], [uniform, frontier],
                color="#b9b9b9", lw=0.8, zorder=1)
        ax.scatter(position - 0.10, uniform, s=23, color=BLUE,
                   edgecolor="white", linewidth=0.35, zorder=2,
                   label="uniform" if position == 0 else None)
        ax.scatter(position + 0.10, frontier, s=23, color=ORANGE,
                   edgecolor="white", linewidth=0.35, zorder=2,
                   label="FrontierMax" if position == 0 else None)
        ax.scatter(position, average, marker="D", s=17, color=BLACK,
                   zorder=3, label="within-block average"
                   if position == 0 else None)
    ax.axhline(0, color="#777777", lw=0.75, ls="--", zorder=0)
    ax.set_xticks(range(len(rows)), [str(row["seed"]) for row in rows])
    ax.set_xlabel("independent seed/warmstart block")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    block_summary = data["waves"]["wave2"]["block_level"][
        "cov_auc" if panel_index == 0 else "easy_band"]
    low, high = block_summary["ci95_t"]
    ax.text(0.02, 0.04,
            f"block mean {block_summary['mean']:+.3f}\n"
            f"95% t CI [{low:+.3f}, {high:+.3f}]",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                  "pad": 1.5})

axes[0].legend(frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(1.02, -0.24), handletextpad=0.35,
               columnspacing=1.0)
fig.subplots_adjust(left=0.09, right=0.99, top=0.89, bottom=0.30, wspace=0.31)
fig.savefig(os.path.join(HERE, "fig11_factorial_blocks.pdf"))
fig.savefig(os.path.join(HERE, "fig11_factorial_blocks.png"), dpi=180)
print("wrote fig11_factorial_blocks.pdf / .png")
