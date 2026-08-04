#!/usr/bin/env python
"""Figure 2: artifact-backed estimator controls and sensitivity.

Panels (a-b) load the 20-seed matched-budget estimator study.
Panel (c) loads the post-hoc full-CV learning-rate sensitivity.
Panel (d) loads three-seed Countdown endpoint aggregates.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CHAIN = json.load(open(os.path.join(
    ROOT, "curriculum_maxrl", "results_estimator_variants.json")))
SENS = json.load(open(os.path.join(
    ROOT, "curriculum_maxrl", "results_fullcv_lr_sensitivity.json")))
COUNTDOWN = json.load(open(os.path.join(
    HERE, "data", "b_scoreboard_3seed.json")))

BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
GRAY = "#555555"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.75))
arms = ["raw", "full_cv", "practical", "practical+hindsight"]
labels = ["raw", "full CV", "practical", "+ hindsight"]
colors = [GRAY, MAGENTA, BLUE, ORANGE]


def chain_panel(ax, regime, title):
    data = CHAIN["regimes"][regime]
    vals = [data[a]["auc_mean"] for a in arms]
    errs = [data[a]["auc_sd"] for a in arms]
    shown = [max(v, 0.006) for v in vals]
    ax.bar(range(4), shown, yerr=errs, color=colors, width=0.68,
           capsize=2, error_kw={"lw": 0.7, "ecolor": "#333333"})
    for i, v in enumerate(vals):
        label = f"{v:.3f}" if v >= 0.0005 else "≈0"
        ax.text(i, max(shown[i] + errs[i] + 0.015, 0.02), label,
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(4), labels, rotation=31, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("AUC")
    ax.set_title(title, loc="left")


chain_panel(axes[0], "balanced", "(a) Balanced chain")
chain_panel(axes[1], "frontier_heavy", "(b) Frontier-heavy")
axes[1].text(0.03, 0.91, "full CV: 3,200\nK=0 updates/run",
             transform=axes[1].transAxes, color=MAGENTA, fontsize=6.7,
             va="top")

# Post-hoc learning-rate sensitivity. Mean and median separate the rare
# lottery escapes from a reliable bootstrap regime.
lr_rows = sorted((float(k), v) for k, v in SENS["learning_rates"].items())
lrs = np.array([x for x, _ in lr_rows])
means = np.array([v["auc_mean"] for _, v in lr_rows])
sds = np.array([v["auc_sd"] for _, v in lr_rows])
medians = np.array([v["auc_median"] for _, v in lr_rows])
axes[2].fill_between(lrs, np.maximum(means - sds, 0), means + sds,
                     color=MAGENTA, alpha=0.16, lw=0)
axes[2].plot(lrs, means, "o-", color=MAGENTA, ms=3, lw=1.2, label="mean")
axes[2].plot(lrs, medians, "--", color=GRAY, lw=1.0, label="median")
axes[2].set_xscale("log")
axes[2].set_ylim(0, 0.13)
axes[2].set_xlabel("full-CV learning rate")
axes[2].set_ylabel("AUC")
axes[2].set_title("(c) Post-hoc LR check", loc="left")
axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.78))
axes[2].text(0.98, 0.96, "rare escapes;\nmedian ≈ 0",
             transform=axes[2].transAxes, ha="right", va="top",
             color=GRAY, fontsize=6.7, style="italic")

# Countdown endpoints: each row is [mean success, seed SD, pass@16, seed SD].
cd_arms = ["B1_t1", "B2_t1", "B3_t1"]
cd_labels = ["B1 base", "B2 relabel", "B3 gate"]
x = np.arange(3)
width = 0.34
mean_vals = [COUNTDOWN[a][0] for a in cd_arms]
mean_errs = [COUNTDOWN[a][1] for a in cd_arms]
cov_vals = [COUNTDOWN[a][2] for a in cd_arms]
cov_errs = [COUNTDOWN[a][3] for a in cd_arms]
axes[3].bar(x - width / 2, mean_vals, width, yerr=mean_errs, color=BLUE,
            capsize=2, label="mean@16",
            error_kw={"lw": 0.7, "ecolor": "#333333"})
axes[3].bar(x + width / 2, cov_vals, width, yerr=cov_errs, color=ORANGE,
            hatch="//", capsize=2, label="pass@16",
            error_kw={"lw": 0.7, "ecolor": "#333333"})
axes[3].set_xticks(x, cd_labels, rotation=28, ha="right")
axes[3].set_ylim(0, 0.63)
axes[3].set_ylabel("endpoint")
axes[3].set_title("(d) Countdown T1", loc="left")
axes[3].legend(frameon=False, loc="upper left")

fig.tight_layout(pad=0.35, w_pad=1.0)
fig.savefig(os.path.join(HERE, "fig2_ladder.pdf"))
fig.savefig(os.path.join(HERE, "fig2_ladder.png"), dpi=180)
print("wrote fig2_ladder.pdf / .png")
