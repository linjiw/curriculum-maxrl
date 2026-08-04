#!/usr/bin/env python
"""Figure 7 — recycling-induced sharpening and the utility gate.

Three panels on the Countdown E-LLM-2b arms (B1 baseline, B2 hindsight
ungated, B3 hindsight + utility gate):

(a) "The trade" — 3-seed tier-1 endpoint scatter, x = pass@16
    (coverage), y = mean@16.  B1 (gray) -> B2 (magenta): recycling lifts
    the mean but costs coverage (sharpening).  The gate slides the arm
    back down the dose axis: B3 under-gated (orange, operating point,
    3-seed) restores the mean to baseline, and the post-fix full-strength
    gate (hollow orange, single seed, corrected-decay code — setting
    confounded with code version) pushes coverage above baseline while
    giving the mean gain back.  The settings are CONSISTENT WITH a
    mean-vs-coverage trade but do not establish a monotone dial (see
    Sec 6.9).  Data: b_scoreboard_3seed.json + b_strong_gate_1seed.json.

(b) "The mechanism" — B3 seed-1 gate rejection rate over training
    (orange), the monotone rise .12 -> .85 that IS the saturation story;
    B2's relabeled rollouts/step normalized to its own cap (magenta,
    dashed) shows the ungated dose riding the cap the whole run.

(c) "The cost signature" — entropy trajectories (seed 1) with final-
    window means as horizontal references.  B2 collapses below baseline;
    B3 sits ABOVE baseline: gated recycling preserves more exploration
    than running with no recycling at all.

Data: curriculum_maxrl/countdown/b_cells_dynamics.json (seed-1 dynamics)
and b_scoreboard_3seed.json (3-seed endpoint aggregates).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")



GRAY = "#555555"       # B1 baseline (no recycling)
MAGENTA = "#e87ba4"    # B2 hindsight (ungated recycling)
ORANGE = "#eb6834"     # B3 hindsight + utility gate
INK = "#333333"

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

dyn = json.load(open(os.path.join(DATA, "b_cells_dynamics.json")))
sb = json.load(open(os.path.join(DATA, "b_scoreboard_3seed.json")))


def clean(a):
    """Interpolate over None entries (first-step telemetry gaps)."""
    x = np.array([np.nan if v is None else v for v in a], float)
    m = np.isnan(x)
    if m.any():
        x[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), x[~m])
    return x


def roll(a, w=5):
    """Centered moving average, edge-padded (keeps length)."""
    pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, np.ones(w) / w, mode="valid")[:len(a)]


fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.0, 2.8))

# ===================================================== (a) the trade
# scoreboard rows: [mean@16_mean, mean@16_sd, pass@16_mean, pass@16_sd]
b1 = sb["B1_t1"]
b2 = sb["B2_t1"]
b3 = sb["B3_t1"]
# post-fix full-strength gate, single seed, corrected-decay code —
# vendored raw endpoint (b_strong_gate_1seed.json), not a literal
pf = json.load(open(os.path.join(DATA, "b_strong_gate_1seed.json")))
pf_pass, pf_mean = pf["tier1"]["pass_at_16"], pf["tier1"]["mean_at_16"]

B1x, B1y = b1[2], b1[0]
B2x, B2y = b2[2], b2[0]
B3x, B3y = b3[2], b3[0]

# baseline reference lines (grayscale-safe: dotted, muted)
axa.axhline(B1y, color=GRAY, lw=0.7, ls=":", alpha=0.55, zorder=1)
axa.axvline(B1x, color=GRAY, lw=0.7, ls=":", alpha=0.55, zorder=1)

EB = dict(capsize=2.5, elinewidth=0.9, capthick=0.9, zorder=4)
# B1 baseline — gray circle
axa.errorbar(B1x, B1y, xerr=b1[3], yerr=b1[1], fmt="o", ms=6,
             color=GRAY, mfc=GRAY, mec="white", mew=0.6,
             ecolor=GRAY, **EB)
# B2 hindsight — magenta square
axa.errorbar(B2x, B2y, xerr=b2[3], yerr=b2[1], fmt="s", ms=6,
             color=MAGENTA, mfc=MAGENTA, mec="white", mew=0.6,
             ecolor=MAGENTA, **EB)
# B3 gated (operating point) — orange triangle
axa.errorbar(B3x, B3y, xerr=b3[3], yerr=b3[1], fmt="^", ms=7,
             color=ORANGE, mfc=ORANGE, mec="white", mew=0.6,
             ecolor=ORANGE, **EB)
# post-fix full-strength gate — hollow orange triangle, no error bars
axa.plot(pf_pass, pf_mean, marker="^", ms=7, mfc="white",
         mec=ORANGE, mew=1.4, ls="none", zorder=4)

LBL_BG = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6)

# arrow B1 -> B2 : sharpening (mean up, coverage down)
axa.annotate("", xy=(B2x, B2y), xytext=(B1x, B1y),
             arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK,
                             shrinkA=7, shrinkB=7,
                             connectionstyle="arc3,rad=0.20"), zorder=3)
axa.text(0.513, 0.362, "recycling:\nmean ↑, coverage ↓",
         fontsize=7, color=INK, ha="center", va="top", style="italic",
         linespacing=1.1, zorder=5)

# dose arrow B2 -> B3 -> post-fix : the gate slides down the frontier and
# restores coverage (fully at full strength)
axa.annotate("", xy=(B3x, B3y), xytext=(B2x, B2y),
             arrowprops=dict(arrowstyle="-|>", lw=1.1, color=ORANGE,
                             shrinkA=7, shrinkB=8,
                             connectionstyle="arc3,rad=0.0"), zorder=3)
axa.annotate("", xy=(pf_pass, pf_mean), xytext=(B3x, B3y),
             arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ORANGE,
                             alpha=0.7, ls="--", shrinkA=7, shrinkB=7,
                             connectionstyle="arc3,rad=-0.28"), zorder=3)
axa.text(0.512, 0.301, "gate strength:\nsettings trace the\ntrade (strong pt:\n1 seed, new code)",
         fontsize=7, color=ORANGE, ha="left", va="center", style="italic",
         linespacing=1.1, bbox=LBL_BG, zorder=5)

# point labels — left column (B2, B3) label left; B1/full-gate label right
axa.text(0.478, 0.331, "B2\nhindsight", fontsize=7, color=MAGENTA,
         ha="right", va="center", linespacing=1.0, zorder=5)
axa.text(0.479, 0.278, "B3 gated\n(op. pt)", fontsize=7,
         color=ORANGE, ha="right", va="center", linespacing=1.0, zorder=5)
axa.text(0.550, 0.264, "B1 baseline", fontsize=7, color=GRAY,
         ha="left", va="center", zorder=5)
axa.text(0.573, 0.219, "full-strength\ngate (1 seed)", fontsize=7,
         color=ORANGE, ha="left", va="center", linespacing=1.0, zorder=5)

axa.set_xlim(0.440, 0.625)
axa.set_ylim(0.175, 0.365)
axa.set_xticks([0.48, 0.52, 0.56, 0.60])
axa.set_yticks([0.20, 0.25, 0.30, 0.35])
axa.set_xlabel("pass@16  (coverage)")
axa.set_ylabel("mean@16")
axa.set_title("(a) The trade", loc="left", fontsize=9)

# ===================================================== (b) the mechanism
steps = np.array(dyn["B3"]["steps"], float)
gate = clean(dyn["B3"]["gate_rejection_rate"])
gate_s = roll(gate, 5)

rel2 = np.array(dyn["B2"]["relabeled_rollouts"], float)
rel2n = rel2 / rel2.max()
rel2n_s = roll(rel2n, 5)

# B2 ungated dose (secondary, faint, dashed) — rides its cap
axb.plot(steps, rel2n_s, color=MAGENTA, lw=1.2, ls="--", alpha=0.65,
         zorder=2, solid_capstyle="round")
# B3 gate rejection rate (primary) — raw faint + smoothed bold
axb.plot(steps, gate, color=ORANGE, lw=0.7, alpha=0.28, zorder=2)
axb.plot(steps, gate_s, color=ORANGE, lw=1.8, zorder=3,
         solid_capstyle="round")

# annotate the rise .12 -> .85
early = gate[:12].mean()
late = gate[-8:].mean()
axb.annotate(f"{early:.2f}", xy=(9, gate[:12].mean()), xytext=(12, 0.30),
             fontsize=7.5, color=ORANGE, ha="left", va="center",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=ORANGE,
                             connectionstyle="arc3,rad=-0.25",
                             shrinkA=2, shrinkB=3), zorder=5)
axb.text(59, 0.905, f"{late:.2f}", fontsize=7.5, color=ORANGE,
         ha="right", va="bottom", fontweight="bold", zorder=5)

# direct labels — magenta at top-left (empty), orange below its curve
axb.text(6, 1.02, "ungated dose\nrides its cap", fontsize=7,
         color=MAGENTA, ha="left", va="top", style="italic",
         linespacing=1.1, zorder=5)
axb.text(58, 0.44, "fraction of relabels\nrejected as saturated",
         fontsize=7, color=ORANGE, ha="right", va="center",
         linespacing=1.1, zorder=5)

axb.set_xlim(1, 60)
axb.set_ylim(0, 1.06)
axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axb.set_xlabel("training step")
axb.set_ylabel("gate rejection rate")
axb.set_title("(b) The mechanism", loc="left", fontsize=9)

# ===================================================== (c) cost signature
W = 15  # final-window length for the reference means
arms = [
    ("B1", GRAY, "-", "B1 baseline"),
    ("B2", MAGENTA, "--", "B2 hindsight"),
    ("B3", ORANGE, "-", "B3 gated"),
]
finals = {}
for key, color, ls, _ in arms:
    e = np.array(dyn[key]["entropy"], float)
    s = np.array(dyn[key]["steps"], float)
    axc.plot(s, e, color=color, lw=1.5, ls=ls, zorder=3,
             solid_capstyle="round", dash_capstyle="round")
    fm = e[-W:].mean()
    finals[key] = fm
    # final-window mean as a faint horizontal reference in the arm color
    axc.plot([44, 60], [fm, fm], color=color, lw=0.8, ls=":",
             alpha=0.8, zorder=2)

# direct end labels at the reference heights, de-collided
axc.text(60.5, finals["B3"] + 0.006, f"B3 gated  {finals['B3']:.2f}",
         fontsize=7, color=ORANGE, ha="left", va="center", zorder=5)
axc.text(60.5, finals["B1"], f"B1 base  {finals['B1']:.2f}",
         fontsize=7, color=GRAY, ha="left", va="center", zorder=5)
axc.text(60.5, finals["B2"] - 0.004, f"B2 hind  {finals['B2']:.2f}",
         fontsize=7, color=MAGENTA, ha="left", va="center", zorder=5)

axc.text(31, 0.55,
         "gated recycling preserves\nMORE entropy than\nno recycling",
         fontsize=7, color=INK, ha="left", va="top", style="italic",
         linespacing=1.2, zorder=5)

axc.set_xlim(1, 60)
axc.set_ylim(0, 0.58)
axc.set_yticks([0, 0.2, 0.4])
axc.set_xlabel("training step")
axc.set_ylabel("policy entropy")
axc.set_title("(c) The cost signature", loc="left", fontsize=9)

fig.tight_layout(pad=0.4, w_pad=1.6)
fig.savefig(os.path.join(HERE, "fig7_sharpening.pdf"))
fig.savefig(os.path.join(HERE, "fig7_sharpening.png"), dpi=150)

# ---------------------------------------------------------- report
print("wrote fig7_sharpening.pdf / .png")
print("\n(a) tier-1 endpoints [pass@16, mean@16]:")
print(f"  B1 baseline  ({B1x:.3f}±{b1[3]:.3f}, {B1y:.3f}±{b1[1]:.3f})")
print(f"  B2 hindsight ({B2x:.3f}±{b2[3]:.3f}, {B2y:.3f}±{b2[1]:.3f})")
print(f"  B3 gated     ({B3x:.3f}±{b3[3]:.3f}, {B3y:.3f}±{b3[1]:.3f})")
print(f"  full-gate    ({pf_pass:.3f}, {pf_mean:.3f})  [single seed]")
print("\n(b) gate rejection rate (B3 seed-1):")
print(f"  early (steps 1-12) mean {early:.3f}  ->  late (last 8) mean {late:.3f}")
print(f"  raw final {gate[-1]:.3f}   B2 relabel cap {rel2.max():.0f} rollouts/step")
print(f"  B2 normalized dose, late (steps 20-60) mean {rel2n[19:].mean():.3f}")
print(f"\n(c) entropy final-window means (last {W} steps):")
for k in ("B1", "B2", "B3"):
    e = np.array(dyn[k]["entropy"], float)
    print(f"  {k}: {finals[k]:.3f}   (endpoint {e[-1]:.3f})")
