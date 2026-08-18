#!/usr/bin/env python
"""MAZE-SCORE: the neural-scale boundary and the mechanism behind it.

Left   the frozen primary and secondary, paired over 48 blocks.
Right  why: predicted vs realized coefficient mass, leave-one-out pass rate.
       The identity is exact under conditionally i.i.d. rollouts; real groups
       share a level but not a maze, so unanimity is far more common than
       Binomial and realized mass falls short -- most where p is lowest, which
       is exactly where the deployed-N score puts its mass.

Numbers from hopper/MAZE_SCORE_ANALYSIS.json (frozen analyzer, run once) and
hopper/MAZE_SCORE_CALIBRATION.json (post-hoc, descriptive).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

BLUE = "#2a78d6"
GREEN = "#008300"
GRAY = "#555555"
RED = "#c1272d"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

cal = json.load(open(os.path.join(ROOT, "hopper/MAZE_SCORE_CALIBRATION.json")))
ana = json.load(open(os.path.join(ROOT, "hopper/MAZE_SCORE_ANALYSIS.json")))
_ = ana  # frozen primary is reported in the text and in the claim map

fig, axB = plt.subplots(1, 1, figsize=(3.7, 2.45))

# ------------------------------------------------------------------ Panel B
p = np.array([b["mean_pass_rate_loo"] for b in cal["bins"]])
pred = np.array([b["predicted_mass_A32"] for b in cal["bins"]])
obs = np.array([b["observed_mass_mean"] for b in cal["bins"]])
sem = np.array([b["observed_mass_sem"] for b in cal["bins"]])

grid = np.linspace(1e-4, 1, 600)
axB.plot(grid, 2 * (1 - grid - (1 - grid) ** 32), color=GRAY, lw=1.3, ls="--")
axB.errorbar(p, obs, yerr=sem, fmt="o", ms=4.2, color=BLUE, lw=1.0,
             capsize=1.8, zorder=5)
axB.plot(p, obs, color=BLUE, lw=1.1, alpha=0.55, zorder=4)

axB.text(0.61, 1.30, "predicted $A_{32}(p)$\n(i.i.d. rollouts)", fontsize=7.2,
         color=GRAY, ha="left", va="center", linespacing=1.15)
axB.text(0.70, 0.72, "realized", fontsize=7.5, color=BLUE, ha="left")

# where each score puts its mass
ps_un = 1 - 32 ** (-1 / 31)
axB.annotate("", xy=(ps_un, 0.02), xytext=(ps_un, 1.86),
             arrowprops=dict(arrowstyle="-", lw=0.8, color=RED, alpha=0.5,
                             ls=":"))
axB.annotate("", xy=(0.5, 0.02), xytext=(0.5, 1.86),
             arrowprops=dict(arrowstyle="-", lw=0.8, color=GREEN, alpha=0.5,
                             ls=":"))
axB.text(ps_un + .015, 0.60, "$u_{32}$ peak\n51% silent\n(2% predicted)",
         fontsize=6.7, color=RED, ha="left", va="top", linespacing=1.15)
axB.text(0.52, 0.42, "$p(1{-}p)$ peak\n12% silent",
         fontsize=6.7, color=GREEN, ha="left", va="top", linespacing=1.15)

axB.set_xlim(0, 1)
axB.set_ylim(0, 1.95)
axB.set_xlabel("leave-one-out pass rate $\\hat p$ of the level")
axB.set_ylabel("group coefficient mass")


fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(HERE, "fig_mazescore.pdf"))
fig.savefig(os.path.join(HERE, "fig_mazescore.png"), dpi=150)
print("wrote fig_mazescore.pdf / .png")
