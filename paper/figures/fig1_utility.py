#!/usr/bin/env python
"""Figure 1 — "The estimator is the curriculum" hero math figure.

Panel A: u_N(p) = (1-(1-p)^N) - p for N in {4, 8, 16, 32}; peaks at
         p* = 1 - N^(-1/(N-1)) ~ ln N / N; dead zones annotated.
Panel B: at N=16, all three curves are EXACT finite-N expected coefficient
         mass for the DEPLOYED estimators (same convention, mass/2):
         MaxRL u_N = A_N/2 (Prop. 1), RLOO p(1-p) (exact), GRPO with
         sample-SD normalization (ddof=1, matching estimators.py and
         verl core_algos.py): sqrt((N-1)/N) * (1/N) E[sqrt(K(N-K))],
         K~Bin(N,p) (MC-verified against the code) — not the population
         w(p) curves, which diverge at the tails.

Numbers/forms from paper/main.tex Section 3 (Prop. 1-3, Remark scope).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def u(p, N):
    """Expected advantage mass / 2 under practical MaxRL weights (Prop. 1)."""
    return (1.0 - (1.0 - p) ** N) - p


def p_star(N):
    return 1.0 - N ** (-1.0 / (N - 1))


def tint(hex_color, f):
    """Blend hex color toward white by fraction f in [0, 1]."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c + (1 - c) * f for c in (r, g, b))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.6))
p = np.linspace(0, 1, 2000)

# ---------------------------------------------------------------- Panel A
Ns = [4, 8, 16, 32]
tints = [0.62, 0.42, 0.20, 0.0]  # sequential blues, light -> dark
for N, f in zip(Ns, tints):
    c = tint(BLUE, f)
    axA.plot(p, u(p, N), color=c, lw=1.6, solid_capstyle="round",
             label=f"$N$={N}")
    ps = p_star(N)
    axA.plot([ps], [u(ps, N)], "o", color=c, ms=4, zorder=5)

axA.legend(loc="upper right", frameon=False, handlelength=1.5,
           borderaxespad=0.2, labelspacing=0.25)

# N=16 peak annotation
ps16 = p_star(16)
axA.annotate(r"$p^{*}\approx \ln N/N$", xy=(ps16, u(ps16, 16)),
             xytext=(0.40, 0.93), fontsize=8, color=GRAY,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=-0.12",
                             shrinkA=2, shrinkB=5))

# dead zone: mastered (p -> 1)
axA.annotate("mastered:\nnothing to learn", xy=(0.975, 0.015),
             xytext=(0.97, 0.16), fontsize=8, color=GRAY,
             ha="right", va="bottom", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             shrinkA=2, shrinkB=2))

# dead zone: unreachable (p -> 0); recycling channel acts here
axA.annotate("unreachable: nothing\nto sample toward\n(recycling acts here)",
             xy=(0.008, 0.04), xytext=(0.24, 0.09), fontsize=8,
             color=GRAY, ha="left", va="bottom", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=-0.1",
                             shrinkA=2, shrinkB=2))

axA.set_xlim(0, 1)
axA.set_ylim(0, 1.0)
axA.set_xlabel("pass rate $p$")
axA.set_ylabel(r"expected coefficient mass $/\,2$")
axA.set_title(r"A   $u_N(p) = \mathrm{pass@}N - \mathrm{pass@}1$",
              loc="left")

# ---------------------------------------------------------------- Panel B
# All three curves are exact expected coefficient mass / 2, same
# convention: sum of |per-rollout coefficients| in each estimator's
# gradient (MaxRL w_i sums; RLOO/GRPO carry the 1/N prefactor).
# GRPO exact for the deployed sample-SD (ddof=1) normalization:
# sqrt((N-1)/N) * (1/N) E[ sqrt(K(N-K)) ], K ~ Bin(N, p), because
# s_K = sqrt(K(N-K)/(N(N-1))).  MC-verified against estimators.py.
from math import comb

N = 16
u16 = u(p, N)                            # MaxRL mass / 2 (Prop. 1)
rloo = p * (1 - p)                       # RLOO mass / 2 (exact)
grpo = np.zeros_like(p)                  # GRPO mass / 2 (exact, sample SD)
for k in range(1, N):
    grpo += (comb(N, k) * p**k * (1 - p)**(N - k)
             * np.sqrt(k * (N - k)) / N)
grpo *= np.sqrt((N - 1) / N)

axB.plot(p, u16, color=BLUE, lw=1.8, ls="-")
axB.plot(p, rloo, color=GREEN, lw=1.6, ls="--")
axB.plot(p, grpo, color=MAGENTA, lw=1.6, ls=":")

# direct labels
axB.text(0.135, 0.835, "MaxRL ($N$=16)", fontsize=8, color=BLUE,
         ha="left", va="bottom")
axB.text(0.42, 0.335, "GRPO (exact, $N$=16)", fontsize=8,
         color=MAGENTA, ha="center", va="top")
axB.text(0.50, 0.135, "RLOO = learnability ($N$=2 slice)", fontsize=8,
         color=GREEN, ha="center", va="top")

# frontier asymmetry: MaxRL >> GRPO at p->0
p0 = 0.05
axB.annotate("", xy=(p0, u(p0, N) - 0.015), xytext=(p0, p0 * (1 - p0) + 0.015),
             arrowprops=dict(arrowstyle="<->", lw=0.8, color=GRAY))
axB.text(0.078, 0.205, "$\\to (N{-}1)\\times$ RLOO\nas $p \\to 0$",
         fontsize=7.5, color=GRAY, ha="left", va="center",
         linespacing=1.15)
# mastered asymmetry: GRPO > MaxRL at p->1 (sample-SD tail ratio
# (N-1)/sqrt(N); the symmetric sqrt(N-1) holds only for population SD)
p1 = 0.93
g1 = float(np.interp(p1, p, grpo))
axB.annotate("$\\frac{N-1}{\\sqrt{N}}\\times$ MaxRL's mass\non mastered prompts",
             xy=(p1, g1), xytext=(0.80, 0.68), fontsize=7.5,
             color=GRAY, ha="center", va="center", style="italic",
             linespacing=1.15,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=-0.15",
                             shrinkA=2, shrinkB=2))

axB.set_xlim(0, 1)
axB.set_ylim(0, 1.0)
axB.set_xlabel("pass rate $p$")
axB.set_ylabel(r"expected coefficient mass $/\,2$")
axB.set_title("B   what each estimator rewards ($N$=16)", loc="left")

fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(os.path.join(HERE, "fig1_utility.pdf"))
fig.savefig(os.path.join(HERE, "fig1_utility.png"), dpi=150)
print("wrote fig1_utility.pdf / .png")
