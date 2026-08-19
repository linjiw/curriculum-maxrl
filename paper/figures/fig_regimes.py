#!/usr/bin/env python
"""Figure 1 - the three-regime map.

The teacher allocates, hindsight creates, the estimator decides whether either
is safe.

Panel A  where a sampler can act at all.  Coefficient activity has exact zeros
         at p=0 and p=1, so the pass-rate axis splits into a region where no
         priority rule can help (nothing to sample toward), a band where a
         sampler redistributes existing signal, and a mastered tail.  Only
         relabeling adds signal outside the band; a sampler is capped by the
         allocation ceiling inside it.
Panel B  which estimator you deployed decides the shape of that band -- and
         GRPO's geometry leans on near-mastered prompts, which is why a
         frontier curriculum is not automatically safe under it.

All curves are exact expected coefficient mass / 2 for the deployed
estimators under conditionally i.i.d. rollouts (Prop. 1 and App. A).
"""
import os
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
GRAY = "#555555"
SAND = "#f0e2c8"
SKY = "#dce9f7"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 2.12))
p = np.linspace(0, 1, 2000)
N = 16
u = (1.0 - (1.0 - p) ** N) - p

# Regime edges are budget-dependent: with B groups per step a task is
# operationally dead when it yields under one live group, B*(1-(1-p)^N) < 1.
# Drawn at B=8, N=16 -> p_lo; the mastered edge is its mirror in all-pass.
B = 8
p_lo = 1.0 - (1.0 - 1.0 / B) ** (1.0 / N)
p_hi = (1.0 / B) ** (1.0 / N)

# ------------------------------------------------------------------ Panel A
axA.axvspan(0, p_lo, color=SAND, lw=0)
axA.axvspan(p_hi, 1, color=SAND, lw=0)
axA.axvspan(p_lo, p_hi, color=SKY, lw=0)
axA.plot(p, u, color=BLUE, lw=2.0, zorder=4)
axA.set_xlim(0, 1)
axA.set_ylim(0, 1.30)

# labelled bands along the top, with the numeric edges called out
axA.annotate("", xy=(0, 1.045), xytext=(p_lo, 1.045),
             arrowprops=dict(arrowstyle="|-|,widthA=.3,widthB=.3", lw=0.9,
                             color="#8a6d3b"))
axA.annotate("", xy=(p_lo, 1.045), xytext=(p_hi, 1.045),
             arrowprops=dict(arrowstyle="|-|,widthA=.3,widthB=.3", lw=0.9,
                             color=BLUE))
axA.annotate("", xy=(p_hi, 1.045), xytext=(1, 1.045),
             arrowprops=dict(arrowstyle="|-|,widthA=.3,widthB=.3", lw=0.9,
                             color="#8a6d3b"))
axA.text(0.005, 1.17, "create", fontsize=8.2, color="#8a6d3b", ha="left",
         weight="bold")
axA.text((p_lo + p_hi) / 2, 1.17, "allocate", fontsize=8.2, color=BLUE,
         ha="center", weight="bold")
axA.text(0.995, 1.17, "mastered", fontsize=8.2, color="#8a6d3b", ha="right",
         weight="bold")
axA.text(p_lo, 0.06, f"  $p{{=}}{p_lo:.3f}$", fontsize=6.6, color=GRAY,
         ha="left", va="bottom")
axA.text(p_hi, 0.06, f"$p{{=}}{p_hi:.3f}$  ", fontsize=6.6, color=GRAY,
         ha="right", va="bottom")
axA.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axA.set_xlabel("pass rate $p$")
axA.set_ylabel(r"coefficient activity $u_N$")
axA.set_title("A   where a sampler can act ($N{=}16$, $B{=}8$)", loc="left",
              pad=6)

# ------------------------------------------------------------------ Panel B
rloo = p * (1 - p)
grpo = np.zeros_like(p)
for k in range(1, N):
    grpo += comb(N, k) * p**k * (1 - p)**(N - k) * np.sqrt(k * (N - k)) / N
grpo *= np.sqrt((N - 1) / N)

axB.plot(p, u, color=BLUE, lw=1.9)
axB.plot(p, grpo, color=MAGENTA, lw=1.5, ls=":")
axB.plot(p, rloo, color=GREEN, lw=1.5, ls="--")
axB.text(0.145, 0.815, "MaxRL", fontsize=8, color=BLUE, ha="left")
axB.text(0.50, 0.335, "GRPO", fontsize=8, color=MAGENTA, ha="center", va="top")
axB.text(0.52, 0.135, "RLOO $=p(1{-}p)$", fontsize=8, color=GREEN,
         ha="center", va="top")
axB.annotate("GRPO keeps mass on\nnear-mastered prompts",
             xy=(0.90, float(np.interp(0.90, p, grpo))), xytext=(0.40, 0.66),
             fontsize=7.0, color=GRAY, ha="left", va="center",
             linespacing=1.15,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA=2, shrinkB=2))
axB.set_xlim(0, 1)
axB.set_ylim(0, 1.0)
axB.set_xlabel("pass rate $p$")
axB.set_ylabel(r"coefficient activity")
axB.set_title("B   which estimator decides the band", loc="left", pad=6)

fig.tight_layout(pad=0.35, w_pad=1.4)
fig.savefig(os.path.join(HERE, "fig_regimes.pdf"))
fig.savefig(os.path.join(HERE, "fig_regimes.png"), dpi=150)
print("wrote fig_regimes.pdf / .png")
