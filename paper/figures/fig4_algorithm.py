#!/usr/bin/env python
"""Figure 4 — one FrontierMax step (method diagram, Sections 4-6).

Left to right: task pool (posterior p-hat) --Thompson sample--> N rollouts
per prompt --> branch: 0<K<N (blue, practical MaxRL weights) / K=0
(orange, common-destination rewrite and reverification), both into the
policy update. Posterior evidence comes only from the requested task.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"    # teacher / allocation / live branch
ORANGE = "#eb6834"  # recycling / creation / dead branch
GREEN = "#008300"   # verification / implementation contract
GRAY = "#555555"    # neutral

plt.rcParams.update({
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def tint(hex_color, f):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c + (1 - c) * f for c in (r, g, b))


def shade(hex_color, f):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c * (1 - f) for c in (r, g, b))


fig = plt.figure(figsize=(7.0, 3.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.4)
ax.axis("off")


def box(cx, cy, w, h, text, color, fs=7.0, fill_tint=0.88, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=tint(color, fill_tint) if fill_tint is not None else "white",
        edgecolor=color, lw=lw, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, ha="center", va="center",
            color="black", linespacing=1.25, zorder=4)


def arrow(p0, p1, color, rad=0.0, lw=1.1, ls="-", zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, lw=lw, ls=ls,
        color=color, shrinkA=1, shrinkB=1, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}"))


# ============================================================ task pool
PL, PR, PB, PT = 0.35, 2.75, 1.9, 4.9
ax.add_patch(FancyBboxPatch((PL, PB), PR - PL, PT - PB,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor="white", edgecolor=GRAY, lw=1.1,
                            zorder=2))
ax.text((PL + PR) / 2, PB - 0.32, r"task pool (posterior $\hat{p}$)",
        fontsize=7.5, ha="center", va="center", color="black")

# 6 rows x 4 cols of squares: dark = mastered, mid = frontier, light = dead
row_colors = [shade(BLUE, 0.25)] * 2 + [tint(BLUE, 0.30)] * 2 + \
             [tint(BLUE, 0.82)] * 2
s, pitch = 0.30, 0.44
row_bottoms = [4.35 - i * pitch for i in range(6)]
for rb, rc in zip(row_bottoms, row_colors):
    for c in range(4):
        ax.add_patch(Rectangle((0.55 + c * 0.38, rb), s, s,
                               facecolor=rc, edgecolor="white", lw=0.5,
                               zorder=3))
for label, rows, col in (("mastered", (0, 1), GRAY),
                         ("frontier", (2, 3), BLUE),
                         ("dead", (4, 5), GRAY)):
    yc = (row_bottoms[rows[0]] + row_bottoms[rows[1]] + s) / 2
    ax.text(2.10, yc, label, fontsize=6, color=col, ha="left",
            va="center", style="italic", zorder=4)

# ==================================================== Thompson sampling
Y_MID = 3.40
arrow((PR + 0.08, Y_MID), (4.24, Y_MID), BLUE, lw=1.3)
ax.text(3.56, 3.58,
        "Thompson sample\n$\\nu_N(\\tilde{p})^\\gamma$ + uniform floor",
        fontsize=6.5, color=BLUE, ha="center", va="bottom",
        linespacing=1.2)

# ======================================================= rollout fan-out
FX0, FX1 = 4.35, 5.50
fan_ys = [2.45 + i * 0.38 for i in range(6)]
ax.add_patch(Circle((FX0, Y_MID), 0.09, facecolor=GRAY, edgecolor="none",
                    zorder=4))
for fy in fan_ys:
    ax.plot([FX0, FX1], [Y_MID, fy], color=GRAY, lw=0.8, zorder=2,
            solid_capstyle="round")
    ax.add_patch(Circle((FX1, fy), 0.065, facecolor="white",
                        edgecolor=GRAY, lw=0.9, zorder=4))
ax.text(4.92, 2.02, "$N$ rollouts per prompt", fontsize=6.5, color=GRAY,
        ha="center", va="top")

# ================================================== branch 1: live contrast
box(7.90, 4.95, 2.60, 1.00,
    "$0 < K < N$: practical weights\n$w_i = r_i/K - 1/N$", BLUE)
arrow((5.68, 3.85), (6.50, 4.68), BLUE, rad=-0.15, lw=1.3)
arrow((9.28, 4.95), (12.30, 3.92), BLUE, rad=-0.15, lw=1.3)

# All-pass groups update the requested-task posterior but have no practical
# policy update because their centered coefficients are constant zero.
box(7.82, 3.35, 2.70, 0.68, "$K=N$: posterior only\nno policy update",
    GRAY, fs=6.2, fill_tint=0.93, lw=0.8)
arrow((5.68, 3.40), (6.43, 3.38), GRAY, lw=0.9)

# ================================================== branch 2: K = 0
box(7.17, 1.77, 1.70, 0.85, "$K = 0$:\ndead group", ORANGE)
arrow((5.68, 2.95), (6.25, 2.10), ORANGE, rad=0.15, lw=1.3)

box(10.40, 1.77, 2.80, 0.96,
    "rewrite + reverify at $g'$\nadmit if $0<K'<N$ and $\\hat p_{g'}\\leq\\tau$",
    ORANGE, fs=6.5)
arrow((8.08, 1.77), (8.94, 1.77), ORANGE, lw=1.3)
ax.text(8.57, 2.42, "choose one common\ndestination $g'$",
        fontsize=6.2, color=ORANGE, ha="center", va="bottom",
        linespacing=1.2)

arrow((11.86, 2.04), (12.55, 3.10), ORANGE, rad=0.18, lw=1.3)
ax.text(11.92, 2.78, "destination\nscores", fontsize=6.2, color=ORANGE,
        ha="right", va="center", linespacing=1.15)

# ==================================================== policy update
box(12.95, 3.60, 1.66, 0.90, "policy\nupdate", "#222222", fs=7.5,
    fill_tint=None, lw=1.3)

# ============================================ posterior observe loop
arrow((5.62, 4.52), (1.55, 4.97), BLUE, rad=0.40, lw=1.0, ls=(0, (4, 2)))
ax.text(3.42, 6.08, "observe (requested task only — never relabels)",
        fontsize=6.5, color=BLUE, ha="center", va="center")

# ======================================= verification / proposal caveat
BX0, BX1, BY = 0.40, 13.70, 0.80
ax.plot([BX0, BX0, BX1, BX1], [BY + 0.14, BY, BY, BY + 0.14],
        color=GREEN, lw=1.3, solid_capstyle="round", zorder=2)
ax.text((BX0 + BX1) / 2, BY - 0.20,
        "exact verifier labels; source-induced destination proposal may be off-policy",
        fontsize=7.5, color=GREEN, ha="center", va="top")

fig.savefig(os.path.join(HERE, "fig4_algorithm.pdf"))
fig.savefig(os.path.join(HERE, "fig4_algorithm.png"), dpi=150)
print("wrote fig4_algorithm.pdf / .png")
