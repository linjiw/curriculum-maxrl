#!/usr/bin/env python
"""Headline figure - one identity and one claim-boundary map.

Panel A  the identity  u_N(p) = p(1-p) * w_{N-1}(p):  the canonical
         learnability score reweighted by MaxRL's own objective weight at
         the truncation the deployed estimator targets (Lemma 1, Cor. 1).
         The peak trajectory p*_N = 1 - N^{-1/(N-1)} moves toward harder
         tasks as the rollout budget grows.
Panel B  what survives: every frozen and supporting shape contrast,
         u_N vs its N=2 slice p(1-p), in native effect units.
Panel C  where it stops: the two preregistered boundaries.

Every number is transcribed from body_iclr.tex and traceable to a study
section; see CLAIM_TRACE_ICLR.md.  Panels B and C share the convention that
a filled marker is a frozen confirmatory primary and an open marker is a
supporting or development read.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

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


def u(p, N):
    return (1.0 - (1.0 - p) ** N) - p


def w(p, T):
    """MaxRL's weight function w_T(p) = sum_{k=1..T} (1-p)^{k-1}."""
    return sum((1.0 - p) ** (k - 1) for k in range(1, T + 1))


def p_star(N):
    return 1.0 - N ** (-1.0 / (N - 1))


fig, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(7.5, 2.34), gridspec_kw={"width_ratios": [0.95, 1.26, 1.18]})

# ------------------------------------------------------------------ Panel A
p = np.linspace(1e-4, 1, 2000)
N = 16
axA.plot(p, p * (1 - p), color=GREEN, lw=1.5, ls="--")
axA.plot(p, u(p, N), color=BLUE, lw=2.0)
axA.set_xlim(0, 1)
axA.set_ylim(0, 0.80)
axA.set_xlabel("pass rate $p$")
axA.set_ylabel("score")

axAt = axA.twinx()
axAt.plot(p, w(p, N - 1), color=GRAY, lw=1.2, ls=":")
axAt.set_ylim(0, 20.6)
axAt.set_ylabel("$w_{N-1}(p)$", color=GRAY, labelpad=1)
axAt.tick_params(axis="y", colors=GRAY, labelsize=7)
axAt.spines["top"].set_visible(False)
axAt.spines["right"].set_color(GRAY)

axA.text(0.55, 0.285, "$p(1{-}p)$", color=GREEN, fontsize=8, ha="center")
axA.text(0.36, 0.52, "$u_{16}$", color=BLUE, fontsize=9, ha="center")
axA.text(0.30, 0.115, "$w_{15}$", color=GRAY, fontsize=8, ha="left")

# peak trajectory: p*_N marches left as the rollout budget grows
for Nn in (2, 4, 8, 16, 32, 64):
    ps = p_star(Nn)
    axA.plot([ps], [0.715], "v", color=BLUE, ms=3.2,
             alpha=0.35 + 0.65 * (np.log2(Nn) - 1) / 5, clip_on=False)
axA.annotate("", xy=(0.055, 0.715), xytext=(0.525, 0.715),
             arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE, alpha=0.6))
axA.text(0.545, 0.715, "$p^{*}_{N}$ as $N{\\uparrow}$", color=BLUE,
         fontsize=7.5, va="center", ha="left")
axA.set_title("A   $u_N(p)=p(1{-}p)\\cdot w_{N-1}(p)$", loc="left", pad=8)

# ------------------------------------------------------------------ Panel B
# u_N vs its N=2 slice p(1-p).  (effect, lo, hi, label, support, primary)
surv = [
    (0.0480, 0.0209, 0.0738, "Acrobot V2  $u_{16}$", "15/20", True),
    (0.0322, 0.0198, 0.0449, "replication A", "17/20", False),
    (0.0307, 0.0166, 0.0453, "replication B", "16/20", False),
    (0.20842, 0.16791, 0.24744, "Digits, MaxRL  $u_{8}$", "23/24", False),
    (0.17665, 0.13593, 0.22042, "Digits, RLOO  $u_{8}$", "24/24", False),
]
ys = np.arange(len(surv))[::-1]
for y, (m, lo, hi, lab, sup, prim) in zip(ys, surv):
    axB.plot([lo, hi], [y, y], color=BLUE, lw=1.5, solid_capstyle="round")
    axB.plot([m], [y], "o", ms=5.5 if prim else 4.5, color=BLUE,
             mfc=BLUE if prim else "white", mew=1.3, zorder=5)
    axB.text(0.300, y, sup, fontsize=6.8, color=GRAY, va="center", ha="right")
axB.axvline(0, color="black", lw=0.8)
axB.set_yticks(ys)
axB.set_yticklabels([s[3] for s in surv], fontsize=7.3)
axB.tick_params(axis="y", length=0)
axB.set_ylim(-0.7, len(surv) - 0.3)
axB.set_xlim(-0.012, 0.302)
axB.set_xticks([0, 0.1, 0.2])
axB.set_xlabel("effect vs $p(1{-}p)$")
axB.set_title("B   shape supported (exact-gradient scale)", loc="left",
              color=BLUE, pad=8)

# ------------------------------------------------------------------ Panel C
stop = [
    (-0.0113, -0.0297, 0.0057, "$u_{16}-u_{64}$  (A)", "10/20", True),
    (-0.0128, -0.0284, 0.0024, "$u_{16}-u_{64}$  (B)", "7/20", True),
    (-0.128, None, None, "AMaze: replace MaxMC", "0/5", False),
    (-0.039, None, None, "AMaze: gate MaxMC", "1/5", False),
    (-0.00324, -0.00543, -0.00111, "MAZE-SCORE 1.26M", "15/48", True),
]
ys = np.array([5.0, 4.3, 2.6, 1.9, 0.2])
for y, (m, lo, hi, lab, sup, hasci) in zip(ys, stop):
    if hasci:
        axC.plot([lo, hi], [y, y], color=RED, lw=1.5, solid_capstyle="round")
    axC.plot([m], [y], "o", ms=4.5, color=RED, mfc="white", mew=1.3, zorder=5)
    axC.text(0.070, y, sup, fontsize=6.8, color=GRAY, va="center", ha="right")
axC.axvline(0, color="black", lw=0.8)
axC.set_yticks(ys)
axC.set_yticklabels([s[3] for s in stop], fontsize=7.3)
axC.tick_params(axis="y", length=0)
axC.set_ylim(-0.35, 6.15)
axC.set_xlim(-0.158, 0.074)
axC.set_xticks([-0.12, -0.06, 0])
axC.set_xlabel("effect")
axC.set_title("C   where it stops", loc="left", color=RED, pad=8)
axC.text(0.070, 5.75, "peak location rejected", fontsize=6.9, color=RED,
         ha="right", va="center", style="italic")
axC.text(0.070, 3.35, "standalone signal rejected", fontsize=6.9, color=RED,
         ha="right", va="center", style="italic")
axC.text(0.070, 0.95, "neural scale rejected", fontsize=6.9, color=RED,
         ha="right", va="center", style="italic")

fig.tight_layout(pad=0.4, w_pad=1.5)
fig.savefig(os.path.join(HERE, "fig_claimmap.pdf"))
fig.savefig(os.path.join(HERE, "fig_claimmap.png"), dpi=150)
print("wrote fig_claimmap.pdf / .png")
