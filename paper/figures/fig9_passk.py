#!/usr/bin/env python
"""Figure 9 — pass@k vs k for Countdown B1/B2/B3 (tier 1, seed 1).

The reviewer-requested one-panel version of the sharpening claim: it
quantifies over k, so show it over k. Ungated recycling (B2) is ABOVE
baseline at k=1 (mean up) and crosses BELOW by k=8-16 (coverage down);
the moderate gate (B3) tracks baseline's low-k gain with a smaller
high-k deficit. Curves cross — that crossing IS recycling-induced
sharpening.

Data: step-60 val, unbiased best@k estimates (verl), n=128 tier-1
problems x 16 samples, seed 1 (the seed with full per-k telemetry in
the ray logs; endpoints cross-check b_scoreboard_3seed.json).
Palette: intervention accents on the shared MaxRL-blue base is
overkill at 3 arms; use gray=baseline, magenta=ungated, orange=gated
(matching fig 7/8's arm colors).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

GRAY = "#555555"      # B1 no recycling
MAGENTA = "#e87ba4"   # B2 ungated recycling
ORANGE = "#eb6834"    # B3 gated

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

K = [1, 2, 4, 8, 16]
# step-60 tier-1 unbiased best@k, seed 1 (ray sessions 07-29_05 B1,
# 07-28_22 B2, 07-29_02 B3, 07-30_06 full-strength gate;
# mean@16 as the k=1 point)
B1 = [0.310, 0.384, 0.451, 0.509, 0.559]
B2 = [0.334, 0.388, 0.428, 0.459, 0.485]
B3 = [0.309, 0.374, 0.425, 0.459, 0.475]
FG = [0.220, 0.316, 0.412, 0.499, 0.564]

fig, ax = plt.subplots(figsize=(3.6, 2.8))

ax.plot(K, B1, color=GRAY, marker="o", ms=4, lw=1.6, label="no recycling")
ax.plot(K, B2, color=MAGENTA, marker="s", ms=4, lw=1.6, ls="--",
        label="recycling, ungated")
ax.plot(K, B3, color=ORANGE, marker="^", ms=4.5, lw=1.6, ls="-.",
        label="recycling + gate")
ax.plot(K, FG, color=ORANGE, marker="^", ms=4.5, lw=1.3, ls=":",
        mfc="white", label="full-strength gate")

ax.set_xscale("log", base=2)
ax.set_xticks(K, [str(k) for k in K])
ax.set_xlabel("$k$ (samples)")
ax.set_ylabel(r"pass@$k$ ($\uparrow$)")
ax.legend(frameon=False, loc="upper left", handlelength=1.6)

# annotate the crossing
ax.annotate("mean up", xy=(1.05, 0.336), xytext=(1.55, 0.305),
            fontsize=7.5, color=MAGENTA, style="italic",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=MAGENTA,
                            shrinkA=2, shrinkB=2))
ax.annotate("coverage down:\nthe curves cross", xy=(14.2, 0.492),
            xytext=(9.5, 0.415), fontsize=7.5, color=MAGENTA,
            ha="center", style="italic", linespacing=1.1,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=MAGENTA,
                            connectionstyle="arc3,rad=0.25",
                            shrinkA=2, shrinkB=3))

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig9_passk.pdf"))
fig.savefig(os.path.join(HERE, "fig9_passk.png"), dpi=150)
print("wrote fig9_passk.pdf / .png")
