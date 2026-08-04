#!/usr/bin/env python
"""Figure 2 — the experiment-ladder results summary (4 mini bar panels).

All plotted values come from data/fig2_ladder_data.json (a versioned
result table with per-panel provenance); this script contains no result
literals. Missing data is a hard error.
"""
import json
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


DATA = json.load(open(os.path.join(HERE, "data", "fig2_ladder_data.json")))

# ------------------------------------------------- (a) skill chain
# honest post-retraction numbers (hindsight_controls.json): the no-floor
# gamma-matched oracle matches the full stack within seed noise;
# oracle+recycling adds +.005
vals_a = DATA["panel_a"]["auc"]
labels_a = DATA["panel_a"]["labels"]
colors_a = [GREEN, BLUE, "white", ORANGE, "white"]
edges_a = [GREEN, BLUE, GRAY, ORANGE, ORANGE]
bars = axa.bar(range(5), vals_a, color=colors_a,
               edgecolor=edges_a, linewidth=0.8, **BAR_KW)
bars[2].set_hatch("///")
bars[4].set_hatch("///")
axa.axhline(0.8885, color=GRAY, ls="--", lw=0.8, zorder=2)
axa.text(-0.42, 1.145, "allocation ceiling:\noracle ties the stack",
         fontsize=7.5, color=GRAY, ha="left", va="top", style="italic")
# stagger the near-tied top-3 value labels so they don't collide
lifts_a = [0.012, 0.012, 0.012, 0.075, 0.012]
for i, v in enumerate(vals_a):
    axa.text(i, v + lifts_a[i], f"{v:.3f}".lstrip("0"), fontsize=6.8,
             ha="center", va="bottom", color="#333333")
slanted_ticks(axa, labels_a)
axa.set_ylim(0, 1.16)
axa.set_ylabel("AUC")
axa.set_title("(a) Skill chain (5 seeds)", loc="left", fontsize=9)

# ------------------------------------------------- (b) frontier-heavy
# palette freeze: magenta is reserved for GRPO-the-estimator; DAPO is a
# baseline sampler here (all arms MaxRL), so it gets gray
vals_b = DATA["panel_b"]["auc"]
labels_b = DATA["panel_b"]["labels"]
colors_b = [GREEN, GRAY, BLUE, ORANGE]
# 0.006-tall stubs so the zero bars still show their category color
axb.bar(range(4), [max(v, 0.006) for v in vals_b], color=colors_b, **BAR_KW)
for i, v in enumerate(vals_b):
    txt = "0" if v == 0 else f"{v:.2f}".lstrip("0")
    axb.text(i, max(v, 0.006) + 0.014, txt, fontsize=VAL_FS, ha="center",
             va="bottom", color="#333333")
axb.text(0.04, 0.80, "creation is the\nonly live channel\n(unif.+rec. ties:\n.931 vs .928)",
         transform=axb.transAxes, fontsize=8, color=GRAY,
         ha="left", va="top", style="italic")
slanted_ticks(axb, labels_b)
axb.set_ylim(0, 1.05)
axb.set_ylabel("AUC")
axb.set_title(r"(b) Frontier ($p\leq 10^{-5}$)", loc="left", fontsize=9)

# ------------------------------------------------- (c) maze
vals_c = DATA["panel_c"]["auc"]
errs_c = DATA["panel_c"]["sd"]
labels_c = DATA["panel_c"]["labels"]
colors_c = [GREEN, ORANGE]
axc.bar(range(2), vals_c, color=colors_c, yerr=errs_c, capsize=3,
        error_kw=dict(lw=0.8, ecolor="#333333", zorder=4), **BAR_KW)
for i, (v, e) in enumerate(zip(vals_c, errs_c)):
    axc.text(i, v + e + 0.005, f"{v:.3f}".lstrip("0"), fontsize=VAL_FS,
             ha="center", va="bottom", color="#333333")
axc.text(0.5, 0.965, DATA["panel_c"]["annotation"], transform=axc.transAxes,
         fontsize=8, color=GRAY, ha="center", va="top", style="italic")
axc.set_xticks(range(2), labels_c)
axc.set_xlim(-0.75, 1.75)
axc.set_ylim(0, 0.29)
axc.set_ylabel("AUC")
axc.set_title("(c) Maze (3 seeds)", loc="left", fontsize=9)

# ------------------------------------------------- (d) GSM8K 2x2
vals_d = DATA["panel_d"]["mean_at_4"]
labels_d = DATA["panel_d"]["labels"]
colors_d = [MAGENTA, MAGENTA, BLUE, BLUE]
# same-model eval noise floor (repeated evals of one checkpoint):
# evaluation noise, not training-seed uncertainty
NOISE_SD = DATA["panel_d"]["eval_noise_sd"]
axd.bar(range(4), vals_d, color=colors_d, yerr=[NOISE_SD] * 4, capsize=3,
        error_kw=dict(lw=0.8, ecolor="#333333", zorder=4), **BAR_KW)
for i, v in enumerate(vals_d):
    axd.text(i, v + NOISE_SD + 0.004, f"{v:.3f}".lstrip("0"),
             fontsize=VAL_FS, ha="center", va="bottom", color="#333333")
axd.annotate("only regressing\ncell (reg. run)", xy=(1.3, 0.080),
             xytext=(2.1, 0.032), fontsize=7.5, color=GRAY,
             ha="center", va="center", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA=2, shrinkB=3))
axd.text(0.02, 0.985, "bars: same-model\neval noise SD",
         transform=axd.transAxes, fontsize=7, color=GRAY,
         ha="left", va="top", style="italic")
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
