#!/usr/bin/env python
"""Figure (appendix) — Jugs pool-conditionality: entropy collapse in
every arm, recycling accelerating it.

One panel: actor-entropy trajectories for the three Jugs arms x 3 seeds
(B1 no recycling, B2 ungated recycling, B3 gated). On this pool —
learnable band a single stratum of 1-2-move template solutions — plain
MaxRL collapses too (1.36 -> 0.01-0.21), and ungated recycling ends
LOWEST: the failure is pool-conditional, not intervention-specific.

Data: jugs_llm/results/entropy_trajectories.json (vendored with
PROVENANCE.md; prereg PREREG_E_LLM3.md).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "jugs_llm", "results",
                   "entropy_trajectories.json")

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

d = json.load(open(SRC))
ARMS = [
    ("jugsv1_maxrl_curfalse_hsfalse", GRAY, "-", "no recycling"),
    ("jugsv1_maxrl_curfalse_hstrue", MAGENTA, "--", "recycling, ungated"),
    ("jugsv1_maxrl_curfalse_hstrue_gated", ORANGE, "-.", "recycling + gate"),
]

fig, ax = plt.subplots(figsize=(3.9, 2.8))
finals = {}
for prefix, color, ls, label in ARMS:
    fs = []
    for s in (1, 2, 3):
        key = f"{prefix}_s{s}_16r"
        e = np.array(d[key]["entropy_by_step"], float)
        ax.plot(np.arange(1, len(e) + 1), e, color=color, ls=ls, lw=1.1,
                alpha=0.85, label=label if s == 1 else None,
                solid_capstyle="round", dash_capstyle="round")
        fs.append(d[key]["entropy_final"])
    finals[label] = fs

ax.set_xlim(1, 60)
ax.set_ylim(0, 1.45)
ax.set_xlabel("training step")
ax.set_ylabel("policy entropy")
# dedupe legend (one entry per arm)
handles, labels = ax.get_legend_handles_labels()
seen, hl = set(), []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        hl.append((h, l))
ax.legend([h for h, _ in hl], [l for _, l in hl], frameon=False,
          loc="upper right", handlelength=1.8)
ax.text(37, 0.72,
        "every arm collapses on this pool —\nthe failure is the pool's,\n"
        "not the intervention's",
        fontsize=7.5, color="#333333", style="italic", ha="center",
        linespacing=1.2)

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig10_jugs_entropy.pdf"))
fig.savefig(os.path.join(HERE, "fig10_jugs_entropy.png"), dpi=150)
print("wrote fig10_jugs_entropy.pdf / .png")
for label, fs in finals.items():
    print(f"  {label}: finals {[round(f,3) for f in fs]}")
