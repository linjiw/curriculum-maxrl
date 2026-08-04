#!/usr/bin/env python
"""Figure 5 — the three channels of Section 6, on one difficulty axis.

x: task difficulty as pass rate p (log scale, 1e-5 .. 1).
Bands: channel 1 (teacher, blue) on the frontier around p* ~ ln N / N;
channel 2 (recycling, orange) on the dead zone p < 1e-2; hatched gray
"mastered: skip" as p -> 1.  u_16(p) overlaid faintly for reference.
Channel 3 (the objective) is the safety rule underneath, annotated in
green at the bottom.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"    # channel 1: teacher / allocation
ORANGE = "#eb6834"  # channel 2: recycling / creation
GREEN = "#008300"   # channel 3: objective / safety
GRAY = "#555555"    # neutral

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def tint(hex_color, f):
    """Blend hex color toward white by fraction f in [0, 1]."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c + (1 - c) * f for c in (r, g, b))


def shade(hex_color, f):
    """Blend hex color toward black by fraction f in [0, 1]."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c * (1 - f) for c in (r, g, b))


N = 16
p = np.logspace(-5, 0, 1500)
u16 = (1.0 - (1.0 - p) ** N) - p
p_star = 1.0 - N ** (-1.0 / (N - 1))          # ~ ln N / N ~ 0.169
u_star = (1.0 - (1.0 - p_star) ** N) - p_star  # ~ 0.77

# band edges (p units)
FRONTIER = (0.025, 0.65)    # ~half-max support of u_16
DEAD = (1e-5, 1e-2)         # channel 2: no sampler reaches here
MASTERED = (0.65, 1.0)      # skip

fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.set_xscale("log")
ax.set_xlim(1e-5, 1.0)
ax.set_ylim(0, 1.02)

# ------------------------------------------------- mastered hatch (back)
ax.axvspan(*MASTERED, facecolor="0.93", edgecolor="0.60", lw=0.0,
           hatch="////", zorder=0)
ax.axvline(MASTERED[0], color="0.60", lw=0.8, zorder=1)
ax.text(0.81, 0.40, "mastered: skip", fontsize=7.5, color=GRAY,
        ha="center", va="center", rotation=90, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        zorder=3)

# ------------------------------------------------- u_16 reference curve
ax.fill_between(p, 0, u16, color=BLUE, alpha=0.07, lw=0, zorder=1)
ax.plot(p, u16, color=GRAY, lw=1.0, alpha=0.5, zorder=2)
ax.text(0.030, 0.60, r"$u_{16}(p)$", fontsize=7.5, color=GRAY,
        alpha=0.9, ha="right", va="center")

# p* marker: dotted drop line + label just above the peak
ax.plot([p_star, p_star], [0, u_star], color=GRAY, lw=0.7, ls=":",
        alpha=0.8, zorder=2)
ax.text(p_star, 0.815, r"$p^{*}\!\approx\!\ln N/N$", fontsize=7,
        color=GRAY, ha="center", va="center", zorder=3)

# ------------------------------------------------- channel bands (lanes)
def band(x0, x1, y0, y1, color, num, num_x, label):
    ax.axvspan(x0, x1, ymin=y0 / 1.02, ymax=y1 / 1.02,
               facecolor=tint(color, 0.78), edgecolor=color, lw=1.2,
               zorder=4)
    cx = np.sqrt(x0 * x1)
    cy = 0.5 * (y0 + y1)
    ax.text(cx, cy, label, fontsize=7, color=shade(color, 0.25),
            ha="center", va="center", linespacing=1.2, zorder=5)
    ax.text(num_x, cy, num, fontsize=8.5, color=color,
            ha="center", va="center", fontweight="bold", zorder=5)


# channel 1: teacher (blue), frontier lane (top); "1" sits left of band
band(*FRONTIER, 0.875, 1.015, BLUE, "1", 0.014,
     "teacher: allocate\ncompute here")

# channel 2: recycling (orange), dead-zone lane; "2" sits right of band
band(*DEAD, 0.665, 0.805, ORANGE, "2", 0.016,
     "recycling: create signal\nhere (no sampler can)")

# ------------------------------------------------- channel 3 annotation
ax.text(1.3e-5, 0.055,
        "channel 3 = the objective underneath\n"
        "decides whether 1 + 2 are safe",
        fontsize=7.5, color=GREEN, ha="left", va="bottom",
        linespacing=1.3, zorder=5)

ax.set_xlabel("task difficulty (pass rate $p$)")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig5_channels.pdf"))
fig.savefig(os.path.join(HERE, "fig5_channels.png"), dpi=150)
print("wrote fig5_channels.pdf / .png")
