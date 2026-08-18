#!/usr/bin/env python
"""Figure 1 - what each estimator's activity geometry rewards at N=16.

The u_N(p) family and the identity u_N = p(1-p) w_{N-1} are carried by the
claim-map figure; this panel is the cross-estimator comparison only.

All three curves are EXACT finite-N expected coefficient mass / 2 for the
DEPLOYED estimators, same convention (sum of |per-rollout coefficients|;
RLOO/GRPO carry the 1/N prefactor):
  MaxRL  u_N = A_N/2                                   (Prop. 1)
  RLOO   p(1-p)                                        (exact)
  GRPO   sample-SD (ddof=1) normalization, matching estimators.py and verl
         core_algos.py:  sqrt((N-1)/N) * (1/N) E[sqrt(K(N-K))], K~Bin(N,p)
         (MC-verified against the code) -- not the population w(p) curves,
         which diverge at the tails.
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

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
p = np.linspace(0, 1, 2000)
N = 16

u16 = (1.0 - (1.0 - p) ** N) - p          # MaxRL mass / 2 (Prop. 1)
rloo = p * (1 - p)                        # RLOO mass / 2 (exact)
grpo = np.zeros_like(p)                   # GRPO mass / 2 (exact, sample SD)
for k in range(1, N):
    grpo += comb(N, k) * p**k * (1 - p)**(N - k) * np.sqrt(k * (N - k)) / N
grpo *= np.sqrt((N - 1) / N)

ax.plot(p, u16, color=BLUE, lw=1.9)
ax.plot(p, grpo, color=MAGENTA, lw=1.5, ls=":")
ax.plot(p, rloo, color=GREEN, lw=1.5, ls="--")

ax.text(0.145, 0.815, "MaxRL", fontsize=8, color=BLUE, ha="left")
ax.text(0.50, 0.335, "GRPO", fontsize=8, color=MAGENTA, ha="center", va="top")
ax.text(0.52, 0.135, "RLOO $=p(1{-}p)$", fontsize=8, color=GREEN,
        ha="center", va="top")

# frontier asymmetry: MaxRL mass -> (N-1) x RLOO as p -> 0
p0 = 0.05
ax.annotate("", xy=(p0, np.interp(p0, p, u16) - 0.02),
            xytext=(p0, p0 * (1 - p0) + 0.02),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color=GRAY))
ax.text(0.085, 0.455, "$(N{-}1)\\times$ RLOO\nas $p\\to0$", fontsize=7.2,
        color=GRAY, ha="left", va="center", linespacing=1.15)

# the two dead zones, where no priority rule can help
ax.text(0.115, 0.025, "unreachable", fontsize=7.0, color=GRAY,
        ha="center", va="center", style="italic")
ax.text(0.885, 0.025, "mastered", fontsize=7.0, color=GRAY,
        ha="center", va="center", style="italic")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.0)
ax.set_xlabel("pass rate $p$")
ax.set_ylabel(r"expected coefficient mass $/\,2$")

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(HERE, "fig1_utility.pdf"))
fig.savefig(os.path.join(HERE, "fig1_utility.png"), dpi=150)
print("wrote fig1_utility.pdf / .png")
