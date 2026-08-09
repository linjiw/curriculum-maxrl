#!/usr/bin/env python
"""Figure 9 — VERL bootstrap best@k proxy for Countdown (tier 1, seed 1).

The reviewer-requested one-panel version of the sharpening claim: it
quantifies over k, so show it over k. Ungated recycling (B2) is above
baseline at k=1, crosses below by k=4, and remains below through k=16 in the
logged proxy;
the moderate gate (B3) tracks baseline's low-k gain with a smaller
high-k deficit. The crossing is consistent with recycling-package
concentration. This metric uses with-replacement resampling and is not
standard unbiased pass@k.

Data: step-60 val, VERL bootstrap best@k proxy, n=128 tier-1
problems x 16 samples, seed 1 (the seed with full per-k telemetry in
the ray logs; endpoints cross-check b_scoreboard_3seed.json).
Palette: intervention accents on the shared MaxRL-blue base is
overkill at 3 arms; use gray=baseline, magenta=ungated, orange=gated
(matching fig 7/8's arm colors).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "fig9_bestk_proxy.json")

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

with open(DATA) as f:
    logged = json.load(f)
K = logged["k"]
B1 = logged["series"]["no_recycling"]
B2 = logged["series"]["recycling"]
B3 = logged["series"]["recycling_faulty_decay_gate"]
FG = logged["series"]["full_strength_refuted"]

fig, ax = plt.subplots(figsize=(3.6, 2.8))

ax.plot(K, B1, color=GRAY, marker="o", ms=4, lw=1.6, label="no recycling")
ax.plot(K, B2, color=MAGENTA, marker="s", ms=4, lw=1.6, ls="--",
        label="recycling, ungated")
ax.plot(K, B3, color=ORANGE, marker="^", ms=4.5, lw=1.6, ls="-.",
        label="recycling + gate")
# superseded: the strength-dial reading this 1-seed curve suggested was
# refuted at 3 seeds (P-R1, Sec 6.9) — kept faded for the record
ax.plot(K, FG, color=ORANGE, marker="^", ms=4.5, lw=1.1, ls=":",
        mfc="white", alpha=0.45,
        label="full-strength (1 seed; P-R1 refuted)")

ax.set_xscale("log", base=2)
ax.set_xticks(K, [str(k) for k in K])
ax.set_xlabel("$k$ (samples)")
ax.set_ylabel(r"VERL bootstrap best@$k$ proxy ($\uparrow$)")
ax.legend(frameon=False, loc="upper left", handlelength=1.6)

# annotate the crossing
ax.annotate("mean up", xy=(1.05, 0.336), xytext=(1.55, 0.305),
            fontsize=7.5, color=MAGENTA, style="italic",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=MAGENTA,
                            shrinkA=2, shrinkB=2))
ax.annotate("proxy down:\nthe curves cross", xy=(14.2, 0.492),
            xytext=(9.5, 0.415), fontsize=7.5, color=MAGENTA,
            ha="center", style="italic", linespacing=1.1,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=MAGENTA,
                            connectionstyle="arc3,rad=0.25",
                            shrinkA=2, shrinkB=3))

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig9_passk.pdf"))
fig.savefig(os.path.join(HERE, "fig9_passk.png"), dpi=150)
print("wrote fig9_passk.pdf / .png")
