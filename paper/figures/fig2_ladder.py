#!/usr/bin/env python
"""Figure 2 — the experiment-ladder results summary (4 mini bar panels).

Data sources (verified):
  (a) frontier_rl/examples/v7_oracle_result.json:
      uniform 0.650, teacher 0.728, oracle(gamma1) 0.851, full stack 0.890
  (b) REPORT.md F-section: frontier-heavy pool (max p = 1e-5): uniform,
      DAPO, plain teacher all 0.00; teacher+recycling 0.93 AUC
  (c) REPORT.md #4 / PAPER.md 7.3: maze AUC uniform 0.211+-0.011 vs
      champion 0.229+-0.009, 6/6 paired wins
  (d) GSM8K_ANALYSIS.md results table (final val mean@4):
      grpo .120, grpo+teacher .093, maxrl+teacher .102, maxrl running
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"      # MaxRL / ours
GREEN = "#008300"     # reference / uniform
MAGENTA = "#e87ba4"   # GRPO
ORANGE = "#eb6834"    # hindsight / full stack
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
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.8))
axa, axb, axc, axd = axes
BAR_KW = dict(width=0.62, zorder=3)
TICK_FS = 7.5
VAL_FS = 7.5


def style(ax):
    ax.tick_params(axis="x", length=0, labelsize=TICK_FS)
    ax.set_axisbelow(True)


def slanted_ticks(ax, labels):
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right",
                  rotation_mode="anchor")


# ------------------------------------------------- (a) skill chain
vals_a = [0.650, 0.728, 0.851, 0.890]
labels_a = ["uniform", "teacher", "oracle", "full stack"]
colors_a = [GREEN, BLUE, "white", ORANGE]
bars = axa.bar(range(4), vals_a, color=colors_a,
               edgecolor=[GREEN, BLUE, GRAY, ORANGE], linewidth=0.8, **BAR_KW)
bars[2].set_hatch("///")
axa.axhline(0.851, color=GRAY, ls="--", lw=0.8, zorder=2)
axa.text(-0.42, 0.905, "oracle\nceiling", fontsize=8, color=GRAY,
         ha="left", va="bottom", style="italic")
for i, v in enumerate(vals_a):
    axa.text(i, v + 0.012, f"{v:.3f}".lstrip("0"), fontsize=VAL_FS,
             ha="center", va="bottom", color="#333333")
slanted_ticks(axa, labels_a)
axa.set_ylim(0, 1.05)
axa.set_ylabel("AUC")
axa.set_title("(a) Skill chain (5 seeds)", loc="left", fontsize=9)

# ------------------------------------------------- (b) frontier-heavy
vals_b = [0.0, 0.0, 0.0, 0.93]
labels_b = ["uniform", "DAPO", "teacher", "+recycling"]
colors_b = [GREEN, MAGENTA, BLUE, ORANGE]
# 0.006-tall stubs so the zero bars still show their category color
axb.bar(range(4), [max(v, 0.006) for v in vals_b], color=colors_b, **BAR_KW)
for i, v in enumerate(vals_b):
    txt = "0" if v == 0 else f"{v:.2f}".lstrip("0")
    axb.text(i, max(v, 0.006) + 0.014, txt, fontsize=VAL_FS, ha="center",
             va="bottom", color="#333333")
axb.text(0.04, 0.80, "signal creation\nvs allocation:\ncategorical",
         transform=axb.transAxes, fontsize=8, color=GRAY,
         ha="left", va="top", style="italic")
slanted_ticks(axb, labels_b)
axb.set_ylim(0, 1.05)
axb.set_ylabel("AUC")
axb.set_title(r"(b) Frontier ($p\leq 10^{-5}$)", loc="left", fontsize=9)

# ------------------------------------------------- (c) maze
vals_c = [0.211, 0.229]
errs_c = [0.011, 0.009]
labels_c = ["unif.", "champion"]
colors_c = [GREEN, ORANGE]
axc.bar(range(2), vals_c, color=colors_c, yerr=errs_c, capsize=3,
        error_kw=dict(lw=0.8, ecolor="#333333", zorder=4), **BAR_KW)
for i, (v, e) in enumerate(zip(vals_c, errs_c)):
    axc.text(i, v + e + 0.005, f"{v:.3f}".lstrip("0"), fontsize=VAL_FS,
             ha="center", va="bottom", color="#333333")
axc.text(0.5, 0.965, "6/6 paired wins", transform=axc.transAxes, fontsize=8,
         color=GRAY, ha="center", va="top", style="italic")
axc.set_xticks(range(2), labels_c)
axc.set_xlim(-0.75, 1.75)
axc.set_ylim(0, 0.29)
axc.set_ylabel("AUC")
axc.set_title("(c) Maze (3 seeds)", loc="left", fontsize=9)

# ------------------------------------------------- (d) GSM8K 2x2
vals_d = [0.120, 0.093, 0.102]
labels_d = ["grpo", "grpo+tea.", "maxrl+tea.", "maxrl"]
colors_d = [MAGENTA, MAGENTA, BLUE]
axd.bar(range(3), vals_d, color=colors_d, **BAR_KW)
# hollow/hatched bar for maxrl uniform, still running (last observed 0.097)
axd.bar([3], [0.097], width=0.62, facecolor="white", edgecolor=BLUE,
        linewidth=0.8, hatch="///", zorder=3)
for i, v in enumerate(vals_d):
    axd.text(i, v + 0.002, f"{v:.3f}".lstrip("0"), fontsize=VAL_FS,
             ha="center", va="bottom", color="#333333")
axd.text(3, 0.048, "running", fontsize=7.5, ha="center", va="center",
         color=BLUE, style="italic", rotation=90)
axd.annotate("only cell that\nregresses (P-G2 ✓)", xy=(1.28, 0.088),
             xytext=(2.35, 0.150), fontsize=8, color=GRAY,
             ha="center", va="center", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=0.2",
                             shrinkA=2, shrinkB=3))
slanted_ticks(axd, labels_d)
axd.set_ylim(0, 0.168)
axd.set_ylabel("final val mean@4")
axd.set_title("(d) GSM8K 2×2\n(pre-registered)", loc="left", fontsize=9)

for ax in axes:
    style(ax)

fig.tight_layout(pad=0.4, w_pad=1.3)
fig.savefig(os.path.join(HERE, "fig2_ladder.pdf"))
fig.savefig(os.path.join(HERE, "fig2_ladder.png"), dpi=150)
print("wrote fig2_ladder.pdf / .png")
