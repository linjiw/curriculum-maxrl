#!/usr/bin/env python
"""Figure 7 — Countdown sharpening and preliminary gate evidence.

Panels (a) and (b) show the three-seed endpoints for tiers 1 and 2.
The corrected strong-gate endpoint is a hollow marker without an error bar
because only one seed is available. Panel (c) shows one B3 run's rejection
rate together with B1/B3 entropy telemetry; it is mechanistic evidence, not
an independent replication or a gate-strength sweep.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

GRAY = "#555555"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

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

with open(os.path.join(DATA, "b_cells_dynamics.json"), encoding="utf-8") as f:
    dyn = json.load(f)
with open(os.path.join(DATA, "b_scoreboard_3seed.json"), encoding="utf-8") as f:
    sb = json.load(f)
with open(os.path.join(DATA, "corrected_gate_single_seed.json"),
          encoding="utf-8") as f:
    corrected_gate = json.load(f)


def as_nan(values):
    """Keep missing telemetry missing rather than inventing an endpoint."""
    return np.asarray([np.nan if v is None else v for v in values], float)


def smooth(values, width=5):
    """NaN-aware centered moving average."""
    kernel = np.ones(width)
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def endpoint_panel(ax, tier, corrected, title):
    """Plot coverage versus mean for B1/B2/B3 and one corrected endpoint."""
    arms = [
        ("B1", "baseline", GRAY, "o"),
        ("B2", "ungated", MAGENTA, "s"),
        ("B3", "moderate gate (old decay)", ORANGE, "^"),
    ]
    points = {}
    for key, label, color, marker in arms:
        row = sb[f"{key}_{tier}"]
        mean, mean_sd, coverage, coverage_sd = row
        points[key] = (coverage, mean)
        ax.errorbar(
            coverage, mean, xerr=coverage_sd, yerr=mean_sd,
            fmt=marker, ms=6, color=color, mfc=color, mec="white", mew=0.6,
            ecolor=color, capsize=2.2, elinewidth=0.9, capthick=0.9,
            label=label, zorder=4)

    # This line highlights only the replicated B1-to-B2 sharpening contrast.
    ax.annotate(
        "", xy=points["B2"], xytext=points["B1"],
        arrowprops=dict(arrowstyle="-|>", color=MAGENTA, lw=1.0,
                        shrinkA=7, shrinkB=7,
                        connectionstyle="arc3,rad=0.15"), zorder=3)
    ax.plot(
        corrected[1], corrected[0], marker="^", ms=7, mfc="white",
        mec=ORANGE, mew=1.4, ls="none", label="corrected gate (1 seed)",
        zorder=5)
    ax.axhline(points["B1"][1], color=GRAY, lw=0.6, ls=":", alpha=0.45)
    ax.axvline(points["B1"][0], color=GRAY, lw=0.6, ls=":", alpha=0.45)
    ax.set_xlabel("pass@16 (coverage)")
    ax.set_ylabel("mean@16")
    ax.set_title(title, loc="left")


fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.0, 2.7))

# Corrected strong-gate endpoints are (mean, pass@16), one seed each.
t1_corrected = corrected_gate["tiers"]["t1"]
t2_corrected = corrected_gate["tiers"]["t2"]
endpoint_panel(
    axa, "t1",
    corrected=(t1_corrected["mean_at_16"], t1_corrected["pass_at_16"]),
    title="(a) Saturated tier 1")
endpoint_panel(
    axb, "t2",
    corrected=(t2_corrected["mean_at_16"], t2_corrected["pass_at_16"]),
    title="(b) Frontier tier 2")

axa.set_xlim(0.45, 0.59)
axa.set_ylim(0.17, 0.36)
axb.set_xlim(0.16, 0.34)
axb.set_ylim(0.045, 0.17)
axa.text(0.455, 0.350, "B1→B2: mean ↑, coverage ↓", color=MAGENTA,
         fontsize=6.7, ha="left", va="top")
axb.text(0.165, 0.164, "B1→B2 point estimates", color=MAGENTA,
         fontsize=6.7, ha="left", va="top")

# One-run telemetry. The second y-axis is explicit to avoid equating entropy
# units with rejection fractions.
steps = np.asarray(dyn["B3"]["steps"], float)
gate = as_nan(dyn["B3"]["gate_rejection_rate"])
axc.plot(steps, gate, color=ORANGE, alpha=0.25, lw=0.7)
axc.plot(steps, smooth(gate), color=ORANGE, lw=1.7,
         label="B3 rejection rate")
axc.set_xlabel("training step")
axc.set_ylabel("rejection fraction", color=ORANGE)
axc.tick_params(axis="y", colors=ORANGE)
axc.set_ylim(0, 1.0)
axc.set_title("(c) One-seed telemetry", loc="left")

axe = axc.twinx()
axe.spines["top"].set_visible(False)
for key, color, ls, label in (
        ("B1", GRAY, "--", "B1 entropy"),
        ("B3", ORANGE, ":", "B3 entropy")):
    entropy = np.asarray(dyn[key]["entropy"], float)
    axe.plot(np.asarray(dyn[key]["steps"], float), smooth(entropy),
             color=color, ls=ls, lw=1.2, label=label)
axe.set_ylabel("policy entropy", color=GRAY)
axe.tick_params(axis="y", colors=GRAY)
axe.set_ylim(0, 0.6)

handles1, labels1 = axc.get_legend_handles_labels()
handles2, labels2 = axe.get_legend_handles_labels()
axc.legend(handles1 + handles2, labels1 + labels2, frameon=False,
           loc="upper right", handlelength=1.5)

# A shared legend defines endpoint markers once; no line joins the gate
# settings because these observations do not establish a dose response.
handles, labels = axa.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, -0.015), handletextpad=0.35, columnspacing=0.9)

fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.45, w_pad=1.4)
fig.savefig(os.path.join(HERE, "fig7_sharpening.pdf"))
fig.savefig(os.path.join(HERE, "fig7_sharpening.png"), dpi=180)

print("wrote fig7_sharpening.pdf / .png")
for tier in ("t1", "t2"):
    print(f"{tier} endpoints [mean, sd, pass@16, sd]")
    for key in ("B1", "B2", "B3"):
        print(f"  {key}: {sb[f'{key}_{tier}']}")
print("corrected gate (one seed): "
      f"t1 mean/pass={t1_corrected['mean_at_16']:.3f}/"
      f"{t1_corrected['pass_at_16']:.3f}; "
      f"t2 mean/pass={t2_corrected['mean_at_16']:.3f}/"
      f"{t2_corrected['pass_at_16']:.3f}")
