#!/usr/bin/env python
"""Figure 1 — "The estimator is the curriculum" hero math figure.

Panel A: u_N(p) = (1-(1-p)^N) - p for N in {4, 8, 16, 32}; peaks at
         p* = 1 - N^(-1/(N-1)) ~ ln N / N; dead zones annotated.
Panel B: at N=16, MaxRL utility vs RLOO's p(1-p) (the N=2 slice) vs a
         GRPO-shaped sqrt(p(1-p)) profile scaled to the same max.

Numbers/forms from PAPER.md Sections 3 (Prop. 1, 4, 5).
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
    axA.plot(p, u(p, N), color=c, lw=1.6, solid_capstyle="round")
    ps = p_star(N)
    axA.plot([ps], [u(ps, N)], "o", color=c, ms=4, zorder=5)
    # direct label near each peak
    dx, dy = (0.012, 0.028)
    axA.text(ps + dx, u(ps, N) + dy, f"N={N}", fontsize=8, color=c,
             ha="left", va="bottom")

# N=16 peak annotation
ps16 = p_star(16)
axA.annotate(r"$p^{*}\approx \ln N/N$", xy=(ps16, u(ps16, 16)),
             xytext=(0.335, 0.955), fontsize=8, color=GRAY,
             ha="left", va="top",
             arrowprops=dict(arrowstyle="-", lw=0.7, color=GRAY,
                             shrinkA=2, shrinkB=4))

# dead zone: mastered (p -> 1)
axA.annotate("mastered:\nnothing to learn", xy=(0.975, 0.015),
             xytext=(0.60, 0.175), fontsize=8, color=GRAY,
             ha="left", va="center", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             shrinkA=2, shrinkB=2))

# dead zone: unreachable (p -> 0); recycling channel acts here
axA.annotate("unreachable: nothing to\nsample toward\n"
             "(recycling creates signal here)",
             xy=(0.008, 0.04), xytext=(0.055, 0.96), fontsize=8,
             color=GRAY, ha="left", va="top", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=0.15",
                             shrinkA=2, shrinkB=2))

axA.set_xlim(0, 1)
axA.set_ylim(0, 1.0)
axA.set_xlabel("pass rate $p$")
axA.set_ylabel(r"expected advantage mass $/\,2$")
axA.set_title(r"A   $u_N(p) = \mathrm{pass@}N - \mathrm{pass@}1$",
              loc="left")

# ---------------------------------------------------------------- Panel B
N = 16
u16 = u(p, N)
rloo = p * (1 - p)                       # RLOO mass / 2 (Prop. 4)
grpo_shape = np.sqrt(p * (1 - p))        # GRPO empirical profile shape
grpo = grpo_shape * (u16.max() / 0.5)    # scaled to same max

axB.plot(p, u16, color=BLUE, lw=1.8, ls="-")
axB.plot(p, rloo, color=GREEN, lw=1.6, ls="--")
axB.plot(p, grpo, color=MAGENTA, lw=1.6, ls=":")

# direct labels
axB.text(0.155, 0.845, "MaxRL ($N$=16)", fontsize=8, color=BLUE,
         ha="left", va="bottom")
axB.text(0.56, 0.855, "GRPO (inverted at extremes)", fontsize=8,
         color=MAGENTA, ha="center", va="bottom")
axB.text(0.52, 0.155, "RLOO = learnability ($N$=2 slice)", fontsize=8,
         color=GREEN, ha="center", va="top")

# ratio annotation near p = 0.05 between blue and green
p0 = 0.05
axB.annotate("", xy=(p0, u(p0, N) - 0.015), xytext=(p0, p0 * (1 - p0) + 0.015),
             arrowprops=dict(arrowstyle="<->", lw=0.8, color=GRAY))
axB.text(0.085, 0.24, r"$\to (N{-}1)\times$ as $p \to 0$",
         fontsize=8, color=GRAY, ha="left", va="center")

axB.set_xlim(0, 1)
axB.set_ylim(0, 1.0)
axB.set_xlabel("pass rate $p$")
axB.set_ylabel(r"expected signal $/\,2$")
axB.set_title("B   what each estimator rewards ($N$=16)", loc="left")

fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(os.path.join(HERE, "fig1_utility.pdf"))
fig.savefig(os.path.join(HERE, "fig1_utility.png"), dpi=150)
print("wrote fig1_utility.pdf / .png")
