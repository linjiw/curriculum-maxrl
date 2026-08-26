#!/usr/bin/env python
"""Frozen descriptive P0 secondary: per-level calibration gap vs gain.

One point is one MAZE-SCORE goal-distance level. The x-axis pools observed
group counts across both P0 arms and all 48 blocks, exactly as frozen. The
y-axis is the block-paired count-law-minus-plug-in cov-AUC difference at that
level. Neither levels nor pooled groups are independent replicates; the panel
is descriptive and must not be read as a mediation analysis.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ANALYSIS = os.path.join(
    ROOT,
    "curriculum_maxrl",
    "group_law_flip",
    "GROUP_LAW_FLIP_ANALYSIS.json",
)

BLUE = "#2a78d6"
GRAY = "#666666"
LIGHT_GRAY = "#b4b4b4"
RED = "#c1272d"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

with open(ANALYSIS, encoding="utf-8") as handle:
    secondary = json.load(handle)["descriptive_secondary"]

rows = secondary["per_level"]
levels = np.arange(13)
gaps = np.array([
    rows[str(level)]["measured_plugin_minus_group_law_gap"]
    for level in levels
])
gains = np.array([
    rows[str(level)]["mean_cov_auc_difference_grouplaw_minus_plugin"]
    for level in levels
])

fig, ax = plt.subplots(figsize=(3.65, 2.35))
ax.axhline(0, color="black", lw=0.7, alpha=0.7)
ax.axvline(0, color="black", lw=0.7, alpha=0.7)

for level, gap, gain in zip(levels, gaps, gains):
    if level in (2, 3, 4):
        color, marker, face = BLUE, "o", BLUE
    elif level == 5:
        color, marker, face = RED, "o", "white"
    elif level >= 8:
        color, marker, face = LIGHT_GRAY, "x", LIGHT_GRAY
    else:
        color, marker, face = GRAY, "o", "white"
    ax.plot(
        gap,
        gain,
        marker=marker,
        ms=5.2,
        color=color,
        mfc=face,
        mew=1.2,
        linestyle="none",
        zorder=4,
    )
    if level <= 5:
        dx = 4 if level not in (4, 5) else -8
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            str(level),
            (gap, gain),
            xytext=(dx, 2.5),
            textcoords="offset points",
            fontsize=7,
            color=color,
            ha=ha,
        )

rho = secondary["spearman_gap_vs_coverage_difference"]
ax.text(
    0.98,
    0.96,
    rf"descriptive $\rho={rho:.3f}$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=7.3,
    color=GRAY,
)
ax.text(
    0.03,
    0.06,
    "levels 8–12\nnear zero contrast",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=6.8,
    color=LIGHT_GRAY,
    linespacing=1.1,
)
ax.set_xlim(-0.035, 1.03)
ax.set_ylim(-0.029, 0.047)
ax.set_xlabel("measured plug-in minus count-law activity gap")
ax.set_ylabel("count law minus plug-in cov-AUC")
ax.set_title("P0 per-level descriptive secondary", loc="left", color=BLUE)

fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(HERE, "fig_grouplaw_secondary.pdf"))
fig.savefig(os.path.join(HERE, "fig_grouplaw_secondary.png"), dpi=150)
print("wrote fig_grouplaw_secondary.pdf / .png")
