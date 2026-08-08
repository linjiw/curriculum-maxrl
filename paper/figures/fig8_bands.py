#!/usr/bin/env python
"""Figure 8 — where and when GRPO's coverage goes (maze, level-resolved).

(a) Band-resolved d(pass@8) (final - warmstart): GRPO loses coverage
    almost entirely on the easy/mid band L1-3 (all 4 runs) while its
    L1-3 pass@1 RISES — it funds reliability by liquidating easy-band
    coverage; MaxRL holds L1-3 flat and concentrates its gain on the
    frontier band L4-6.

    NOTE (statistics): the MaxRL pool is a heterogeneous run INVENTORY
    (teacher variants, hindsight doses, a wide model, repeated seeds),
    not 18 independent replicates; the figure is descriptive and
    intentionally carries no p-value (see the review of 2026-08-04).
(b) Diversity premium G = mean_levels(pass@8 - pass@1) vs wall-clock:
    GRPO's erosion starts within the first ~25-50 steps, long before
    any level saturates; MaxRL's premium stabilizes after warmup.

Data: the exact run manifest in data/fig8_run_manifest.json (passk
field: 16 mazes x 8 samples per level; band values average 3 levels x
runs).  The manifest freezes the cohort: adding a new matched_*.jsonl
to maze_gpu/ does NOT silently change this figure; missing manifest
entries are a hard error.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE = os.path.join(HERE, "..", "..", "curriculum_maxrl", "maze_gpu")

BLUE = "#2a78d6"      # MaxRL
MAGENTA = "#e87ba4"   # GRPO
GRAY = "#555555"

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


def series(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "passk" in d:
                rows.append(d)
    return rows


def band_delta(rows, lo, hi, k):
    a = np.mean([rows[0]["passk"][str(i)][k] for i in range(lo, hi + 1)])
    b = np.mean([rows[-1]["passk"][str(i)][k] for i in range(lo, hi + 1)])
    return b - a


def gap_curve(rows, n_levels=13):
    ts, gs = [], []
    for r in rows:
        pk = r["passk"]
        gap = np.mean([pk[str(i)]["8"] - pk[str(i)]["1"]
                       for i in range(n_levels)])
        ts.append(r.get("elapsed", r.get("step", 0)))
        gs.append(gap)
    return np.asarray(ts, float), np.asarray(gs, float)


MANIFEST = os.path.join(HERE, "data", "fig8_run_manifest.json")
with open(MANIFEST) as f:
    manifest = json.load(f)


def load_cohort(names):
    runs = []
    for name in names:
        path = os.path.join(MAZE, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"fig8 manifest names {name} but it is missing from "
                f"{MAZE}; refusing to silently redefine the cohort")
        rows = series(path)
        if not rows:
            raise ValueError(
                f"{name} has no passk rows; if it should be excluded, "
                f"move it to the manifest's 'excluded' list with a reason")
        runs.append(rows)
    return runs


grpo_runs = load_cohort(manifest["grpo_runs"])
maxrl_runs = load_cohort(manifest["maxrl_runs"])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7))

# ------------------------------------------------ (a) band deltas
bands = [("easy/mid\nL1–3", 1, 3), ("frontier\nL4–6", 4, 6)]
x = np.arange(len(bands))
W = 0.32
for off, runs, color, label in ((-W / 2, grpo_runs, MAGENTA, "GRPO"),
                                (W / 2, maxrl_runs, BLUE, "MaxRL")):
    means, sds, pts = [], [], []
    for _, lo, hi in bands:
        v = [band_delta(r, lo, hi, "8") for r in runs]
        means.append(np.mean(v))
        sds.append(np.std(v))
        pts.append(v)
    axa.bar(x + off, means, width=W, color=color, alpha=0.85, zorder=3,
            label=f"{label} (n={len(runs)})")
    for xi, v in zip(x + off, pts):
        axa.scatter([xi] * len(v), v, s=7, color="#333333", alpha=0.55,
                    zorder=4, linewidths=0)

axa.axhline(0, color=GRAY, lw=0.8, zorder=2)
axa.set_xticks(x, [b[0] for b in bands])
axa.set_ylabel(r"$\Delta$ pass@8 (final $-$ warmstart)")
axa.set_title("(a) Where the coverage goes", loc="left", fontsize=9)
axa.legend(frameon=False, loc="upper left", handlelength=1.2)
axa.text(0.44, -0.19, "GRPO also raises L1–3\npass@1 (+.02..+.14):\n"
         "reliability bought\nwith coverage", fontsize=7, color=MAGENTA,
         ha="left", va="center", style="italic", linespacing=1.15)
axa.set_ylim(-0.33, 0.33)

# ------------------------------------------------ (b) gap vs time
for runs, color in ((maxrl_runs, BLUE), (grpo_runs, MAGENTA)):
    curves = []
    for r in runs:
        t, g = gap_curve(r)
        curves.append((t, g))
    tmax = max(t[-1] for t, _ in curves)
    grid = np.linspace(0, tmax, 60)
    interped = np.array([np.interp(grid, t, g) for t, g in curves])
    mu = interped.mean(axis=0)
    lo, hi = interped.min(axis=0), interped.max(axis=0)
    axb.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
    axb.plot(grid, mu, color=color, lw=1.7, solid_capstyle="round")

axb.text(1550, 0.115, "MaxRL", color=BLUE, fontsize=8.5, va="bottom")
axb.text(1550, 0.038, "GRPO", color=MAGENTA, fontsize=8.5, va="bottom")
axb.annotate("erosion starts in the\nfirst ~25–50 steps,\nno level saturated",
             xy=(280, 0.085), xytext=(820, 0.155), fontsize=7.5, color=GRAY,
             ha="left", va="center", style="italic", linespacing=1.15,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY,
                             connectionstyle="arc3,rad=0.2",
                             shrinkA=2, shrinkB=2))
axb.set_xlabel("wall-clock (s), matched budget")
axb.set_ylabel(r"coverage--reliability gap $\overline{p@8 - p@1}$")
axb.set_title("(b) When it goes", loc="left", fontsize=9)
axb.set_xlim(0, None)
axb.set_ylim(0, None)

fig.tight_layout(pad=0.4, w_pad=1.4)
fig.savefig(os.path.join(HERE, "fig8_bands.pdf"))
fig.savefig(os.path.join(HERE, "fig8_bands.png"), dpi=150)
print("wrote fig8_bands.pdf / .png")
