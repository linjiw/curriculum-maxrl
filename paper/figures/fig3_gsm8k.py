#!/usr/bin/env python
"""Figure 3 — GSM8K 2x2 divergence (single column).

Val mean@4 trajectories come from data/fig3_gsm8k_data.json — a
versioned result table; this script contains no result literals.
Review fix (2026-08-05 draft review, "the LLM figure visually outruns
the evidence"): BOTH GRPO seeds are plotted as paired trajectories
(registered seed solid-weight, replication seed thin), and the
repeated-eval noise band is drawn once as a floor annotation rather
than as per-point error bars that read as method-level uncertainty.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"      # MaxRL / ours
GREEN = "#008300"     # reference / uniform
MAGENTA = "#e87ba4"   # GRPO
GRAY = "#555555"      # oracle / bounds

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.5, 2.8))

# palette freeze (paper-wide): hue = estimator (GRPO magenta, MaxRL
# blue); line style = intervention (uniform solid, +teacher dashed)
DATA = json.load(open(os.path.join(HERE, "data", "fig3_gsm8k_data.json")))
steps = DATA["steps"]
STYLE = {
    "grpo":          (MAGENTA, "-"),
    "grpo+teacher":  (MAGENTA, "--"),
    "maxrl":         (BLUE,    "-"),
    "maxrl+teacher": (BLUE,    "--"),
}

# repeated-eval noise floor of one fixed checkpoint — drawn as a scale
# bar, NOT per-point bars (those read as method-level uncertainty)
NOISE_SD = DATA["eval_noise_sd"]

# registered seed: full-weight lines
for name, (color, ls) in STYLE.items():
    ax.plot(steps, DATA["series"][name], color=color, ls=ls, lw=1.6,
            marker="o", ms=3.5, zorder=3, solid_capstyle="round",
            alpha=0.95)

# replication seed (GRPO cells only): thin paired trajectories
for name, ys in DATA.get("series_seed2", {}).items():
    color, ls = STYLE[name]
    ax.plot(steps, ys, color=color, ls=ls, lw=0.9, marker="o", ms=2.2,
            zorder=2, alpha=0.55)

# eval-noise scale bar (±1 SD) in the lower right
ax.errorbar([62], [0.072], yerr=NOISE_SD, color=GRAY, elinewidth=1.0,
            capsize=2.5, marker="", zorder=2)
ax.text(63.5, 0.072, "eval\nnoise\n±SD", fontsize=6.5, color=GRAY,
        va="center")

# direct labels at line ends (registered seed)
ax.text(51.5, 0.121, "grpo", color=MAGENTA, fontsize=8, va="center")
ax.text(51.5, 0.091, "grpo+teacher ↓", color=MAGENTA, fontsize=8,
        va="center")
ax.text(51.5, 0.100, "maxrl+teacher", color=BLUE, fontsize=8, va="center")
ax.text(51.5, 0.109, "maxrl", color=BLUE, fontsize=8, va="center")
ax.text(51.5, 0.116, "seed 2 (thin)", color=MAGENTA, fontsize=6.5,
        va="center", alpha=0.7)

# window annotation: what the two seeds do and do not share
ax.text(0, 0.134, "registered seed (thick): grpo+teacher\n"
        "regresses; replication seed (thin) climbs —\n"
        "shape is 1 of 2 seeds; endpoint teacher-\n"
        "deficit sign shared by both",
        fontsize=6.8, color=GRAY, ha="left", va="top",
        style="italic")

ax.set_xlim(-2, 70)
ax.set_ylim(0.060, 0.140)
ax.set_xticks(steps)
ax.set_xlabel("training step")
ax.set_ylabel("val mean@4")

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig3_gsm8k.pdf"))
fig.savefig(os.path.join(HERE, "fig3_gsm8k.png"), dpi=150)
print("wrote fig3_gsm8k.pdf / .png")
