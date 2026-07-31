#!/usr/bin/env python
"""Figure 3 — GSM8K 2x2 divergence (single column).

Val mean@4 trajectories at steps 0/25/50, from GSM8K_ANALYSIS.md:
  grpo (uniform)     .078 / .105 / .120
  grpo + teacher     .072 / .096 / .093   <- the only regressing cell
  maxrl + teacher    .066 / .099 / .102
  maxrl (uniform)    .091 / .097 / .108
Step-25-to-50 window shaded: the identical teacher helps MaxRL, hurts GRPO.
"""
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
steps = [0, 25, 50]
series = [
    # (label, steps, values, color, linestyle)
    ("grpo",           steps,    [0.078, 0.105, 0.120], MAGENTA, "-"),
    ("grpo+teacher",   steps,    [0.072, 0.096, 0.093], MAGENTA, "--"),
    ("maxrl",          steps,    [0.091, 0.097, 0.108], BLUE,    "-"),
    ("maxrl+teacher",  steps,    [0.066, 0.099, 0.102], BLUE,    "--"),
]

# divergence window shading (behind everything)
ax.axvspan(25, 50, color="#000000", alpha=0.05, zorder=0)

for label, xs, ys, color, ls in series:
    ax.plot(xs, ys, color=color, ls=ls, lw=1.6, marker="o", ms=3.5,
            zorder=3, solid_capstyle="round")

# direct labels at line ends
ax.text(51.5, 0.121, "grpo", color=MAGENTA, fontsize=8, va="center")
ax.text(51.5, 0.091, "grpo+teacher ↓", color=MAGENTA, fontsize=8,
        va="center")
ax.text(51.5, 0.100, "maxrl+teacher", color=BLUE, fontsize=8, va="center")
ax.text(51.5, 0.109, "maxrl", color=BLUE, fontsize=8, va="center")

# divergence-window annotation
ax.text(37.5, 0.0655, "divergence window: only\ngrpo+teacher regresses\n"
        "(P-G2 ✓); maxrl arms both climb", fontsize=8, color=GRAY,
        ha="center", va="bottom", style="italic")

ax.set_xlim(-2, 68)
ax.set_ylim(0.060, 0.128)
ax.set_xticks(steps)
ax.set_xlabel("training step")
ax.set_ylabel("val mean@4")

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig3_gsm8k.pdf"))
fig.savefig(os.path.join(HERE, "fig3_gsm8k.png"), dpi=150)
print("wrote fig3_gsm8k.pdf / .png")
