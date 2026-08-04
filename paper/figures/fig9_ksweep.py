#!/usr/bin/env python
"""Figure 9 - maze summed-coverage vs k: the crossing at k~4.

Data: curriculum_maxrl/maze_gpu/efficiency.json (final checkpoints,
uniform-teacher pair + champion), summed pass@k over levels 0-6.
Currently prose in 6.3 / App B; N1 on the submission gap list.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#2a78d6"
MAGENTA = "#e87ba4"
GRAY = "#555555"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

src = os.path.join(HERE, "..", "..", "curriculum_maxrl", "maze_gpu",
                   "efficiency.json")
d = json.load(open(src))

def summed(name):
    curves = d[name]["curves"]
    ks = sorted(int(k) for k in curves["0"])
    return ks, [sum(curves[lv][str(k)] for lv in curves if int(lv) <= 6)
                for k in ks]

ks, maxrl = summed("ck_uniform_maxrl")
_, grpo = summed("ck_uniform_grpo")

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(ks, maxrl, color=BLUE, lw=1.7, marker="o", ms=3.5, label="MaxRL")
ax.plot(ks, grpo, color=MAGENTA, lw=1.7, marker="s", ms=3.5, label="GRPO")
ax.set_xscale("log", base=2)
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("inference samples $k$")
ax.set_ylabel("summed pass@$k$ (levels 0–6)")

# crossing annotation
cross_k = None
for i in range(len(ks) - 1):
    if (maxrl[i] - grpo[i]) * (maxrl[i + 1] - grpo[i + 1]) < 0:
        cross_k = ks[i + 1]
        break
ax.axvline(4, color=GRAY, ls=":", lw=0.8)
ax.text(4.4, 3.15, "curves cross\nat $k\\approx4$", fontsize=7.5,
        color=GRAY, va="bottom")
ax.annotate("GRPO ahead\nat pass@1", xy=(1, grpo[0]), xytext=(1.15, 3.9),
            fontsize=7.5, color=MAGENTA,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=MAGENTA))
ax.annotate("MaxRL ahead by\nhalf a level at $k$=64",
            xy=(64, maxrl[-1]), xytext=(11, 4.75), fontsize=7.5,
            color=BLUE,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE))
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig9_ksweep.pdf"))
fig.savefig(os.path.join(HERE, "fig9_ksweep.png"), dpi=150)
print("wrote fig9_ksweep")
