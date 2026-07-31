#!/usr/bin/env python
"""Figure 6 — gym convergence: does the official task get solved?

Two panels (MountainCar left, CartPole right), from
frontier_rl/examples/gym_convergence.json (shared-policy runs, 3 seeds,
trained to plateau).  Per arm (standard RL target-only, gray;
FrontierMax gated, orange) two curves: mean-across-bins pass rate
(solid) and the hard-bin / official-task pass rate (dashed).  Band =
seed min/max.  The claim is the dashed pair: the official task reaches
a nonzero plateau under FrontierMax while staying at zero under
standard RL.  Endpoint multiplier on the mean curves is annotated.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "..", "frontier_rl", "examples")

ORANGE = "#eb6834"  # FrontierMax (gated)
GRAY = "#555555"    # standard RL (target only)

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

conv = json.load(open(os.path.join(EXAMPLES, "gym_convergence.json")))

# 5-seed endpoint tables, printed for cross-reference
for name in ("gym_performance.json", "gym_performance_3x.json"):
    path = os.path.join(EXAMPLES, name)
    if os.path.exists(path):
        tab = json.load(open(path))
        print(f"-- {name} (5-seed endpoint table) --")
        for k, v in tab.items():
            print(f"  {k:22s} target {v['target_task']:.3f} "
                  f"mean_bins {v['mean_bins']:.3f}")


def stack(runs, metric):
    """Align 3 seed curves that plateaued at different steps.

    Returns (steps, values[n_points, n_seeds]); shorter runs are
    extended by holding their final (plateau) value.
    """
    steps = max((r["steps"] for r in runs), key=len)
    n = len(steps)
    cols = []
    for r in runs:
        v = list(r[metric])
        v += [v[-1]] * (n - len(v))
        cols.append(v)
    return np.asarray(steps, float), np.asarray(cols, float).T


ARMS = [
    ("target_only", GRAY, "standard RL (target only)"),
    ("full_gated", ORANGE, "FrontierMax (gated)"),
]
GREEN = "#008300"  # uniform-over-bins control (official-task curve only)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, env, title in zip(axes, ("mc", "cp"), ("MountainCar", "CartPole")):
    ends = {}      # (arm, metric) -> final mean value
    ymax, S = 0.0, 0.0
    curves = {}
    for arm, color, label in ARMS:
        runs = conv[f"{env}_{arm}"]
        for metric, ls in (("mean", "-"), ("hard", "--")):
            steps, m = stack(runs, metric)
            mu, lo, hi = m.mean(axis=1), m.min(axis=1), m.max(axis=1)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15,
                            lw=0, zorder=2)
            ax.plot(steps, mu, color=color, ls=ls, lw=1.6, zorder=3,
                    solid_capstyle="round")
            curves[(arm, metric)] = (color, ls, float(steps[-1]),
                                     float(mu[-1]))
            ends[(arm, metric)] = float(mu[-1])
            ymax = max(ymax, float(hi.max()))
            S = max(S, float(steps[-1]))

    # uniform-over-bins control: official-task curve only (attribution:
    # spread vs teacher — it also cracks the task, slower and off-ceiling)
    if f"{env}_uniform" in conv:
        runs = conv[f"{env}_uniform"]
        steps, m = stack(runs, "hard")
        mu = m.mean(axis=1)
        ax.plot(steps, mu, color=GREEN, ls="--", lw=1.3, zorder=2.5,
                alpha=0.9)
        curves[("uniform", "hard")] = (GREEN, "--", float(steps[-1]),
                                       float(mu[-1]))
        ends[("uniform", "hard")] = float(mu[-1])
        S = max(S, float(steps[-1]))

    # arms early-stop at their own plateau; extend the flat tail faintly
    # to the common right edge so endpoints are compared at matched steps
    for (arm, metric), (color, ls, s_end, v_end) in curves.items():
        if s_end < S:
            ax.plot([s_end, S], [v_end, v_end], color=color, ls=ls,
                    lw=1.6, alpha=0.35, zorder=3)

    top = ymax * 1.14
    ax.set_xlim(0, S * 1.56)
    ax.set_ylim(-0.012, top)
    ax.set_xticks(np.linspace(0, S, 4).round(-1).astype(int))
    ax.set_xlabel("training step")
    ax.set_title(title)

    # endpoint multiplier on the mean curves
    to_end = ends[("target_only", "mean")]
    fg_end = ends[("full_gated", "mean")]
    mult = fg_end / max(to_end, 1e-9)
    ax.annotate("", xy=(S * 1.05, fg_end), xytext=(S * 1.05, to_end),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333",
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text(S * 1.08, 0.5 * (to_end + fg_end), f"{mult:.0f}×",
            fontsize=10, fontweight="bold", color="#333333",
            ha="left", va="center", zorder=5)

    # direct labels at the right ends, de-collided within each arm group:
    # the gray pair stacks up from a floor, the orange pair stacks down
    # from a ceiling (both pairs sit at their curves' endpoint heights)
    entries = [
        (("full_gated", "hard"), ORANGE, "official task", 7.5, "italic"),
        (("full_gated", "mean"), ORANGE, "FrontierMax\n(gated)", 8, None),
        (("uniform", "hard"), GREEN, "uniform bins\n(official)", 7.5, "italic"),
        (("target_only", "mean"), GRAY, "standard RL\n(target only)", 8, None),
        (("target_only", "hard"), GRAY, "official task: 0", 7.5, "italic"),
    ]
    entries = [e for e in entries if e[0] in ends]
    ys = {k: ends[k] for k, *_ in entries}
    min_gap = 0.105 * top
    # gray group: floor at 0.035*top, stack upward (hard below mean)
    g_lo, g_hi = ("target_only", "hard"), ("target_only", "mean")
    ys[g_lo] = max(ys[g_lo], 0.035 * top)
    ys[g_hi] = max(ys[g_hi], ys[g_lo] + min_gap)
    # orange group: ceiling at 0.95*top, stack downward (hard above mean)
    o_lo, o_hi = ("full_gated", "mean"), ("full_gated", "hard")
    ys[o_hi] = min(max(ys[o_hi], ys[o_lo] + min_gap), 0.95 * top)
    ys[o_lo] = min(ys[o_lo], ys[o_hi] - min_gap)
    # uniform control label: slot it below the orange pair
    u_k = ("uniform", "hard")
    if u_k in ys:
        ys[u_k] = min(ys[u_k], ys[o_lo] - min_gap)
        ys[g_hi] = min(ys[g_hi], ys[u_k] - min_gap)
    for key, color, label, fs, style in entries:
        ax.text(S * 1.17, ys[key], label, color=color, fontsize=fs,
                ha="left", va="center", linespacing=1.1, style=style,
                zorder=5)

    print(f"{env}: mean  target_only {to_end:.3f}  full_gated {fg_end:.3f}"
          f"  -> {mult:.1f}x")
    print(f"{env}: hard  target_only {ends[('target_only', 'hard')]:.3f}"
          f"  full_gated {ends[('full_gated', 'hard')]:.3f}")

axes[0].set_ylabel("pass rate")

fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(os.path.join(HERE, "fig6_gym.pdf"))
fig.savefig(os.path.join(HERE, "fig6_gym.png"), dpi=150)
print("wrote fig6_gym.pdf / .png")
