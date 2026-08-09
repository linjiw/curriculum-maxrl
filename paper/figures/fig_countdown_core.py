#!/usr/bin/env python3
"""Core Countdown result for the compact ICLR paper.

The figure deliberately separates the primary recycling comparison from the
higher-dose live-group control.  B1/B2 are three-seed aggregates from the
frozen scoreboard.  ARM B contributes all three recorded seed endpoints.
The control uses a higher update dose and is not a dose-matched estimate of a
relabel-specific effect. The horizontal metric is VERL's with-replacement
bootstrap best@16 proxy, not standard unbiased pass@16.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SCOREBOARD = HERE / "data" / "b_scoreboard_3seed.json"
REVIEWER = (
    HERE.parent.parent
    / "curriculum_maxrl"
    / "countdown_reviewer_arms"
    / "reviewer_arms_verdicts.json"
)

GRAY = "#555555"
MAGENTA = "#d95f8d"
BLUE = "#2678b2"
INK = "#222222"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> None:
    scoreboard = json.loads(SCOREBOARD.read_text())
    reviewer = json.loads(REVIEWER.read_text())

    # Frozen rows: [mean@16 mean, population SD, logged bootstrap-best@16
    # mean, population SD].
    # Convert the stored three-seed population spreads to sample SD so all
    # displayed error bars use the same convention as the replay endpoints.
    b1 = scoreboard["B1_t1"]
    b2 = scoreboard["B2_t1"]
    sample_sd_factor = np.sqrt(3.0 / 2.0)
    replay_mean = np.asarray(reviewer["P_R2"]["t1_mean16"], dtype=float)
    replay_pass = np.asarray(reviewer["P_R2"]["t1_pass16"], dtype=float)

    points = [
        (
            "no recycling",
            b1[2],
            b1[0],
            b1[3] * sample_sd_factor,
            b1[1] * sample_sd_factor,
            GRAY,
            "o",
        ),
        (
            "recycling",
            b2[2],
            b2[0],
            b2[3] * sample_sd_factor,
            b2[1] * sample_sd_factor,
            MAGENTA,
            "s",
        ),
        (
            "higher-dose replay",
            float(replay_pass.mean()),
            float(replay_mean.mean()),
            float(replay_pass.std(ddof=1)),
            float(replay_mean.std(ddof=1)),
            BLUE,
            "D",
        ),
    ]

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.axhline(b1[0], color=GRAY, lw=0.7, ls=":", alpha=0.5)
    ax.axvline(b1[2], color=GRAY, lw=0.7, ls=":", alpha=0.5)

    for label, x, y, xerr, yerr, color, marker in points:
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=6.5,
            color=color,
            mfc=color,
            mec="white",
            mew=0.7,
            ecolor=color,
            capsize=2.5,
            elinewidth=0.9,
            capthick=0.9,
            label=label,
            zorder=4,
        )

    # Show the replay seeds without implying a paired path.
    ax.scatter(
        replay_pass,
        replay_mean,
        marker="D",
        s=22,
        facecolors="none",
        edgecolors=BLUE,
        linewidths=0.8,
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(b2[2], b2[0]),
        xytext=(b1[2], b1[0]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=INK,
            lw=1.0,
            connectionstyle="arc3,rad=0.18",
            shrinkA=7,
            shrinkB=7,
        ),
    )
    ax.text(
        0.505,
        0.307,
        "recycling:\nmean up,\nproxy down",
        ha="center",
        va="top",
        fontsize=7.3,
        color=INK,
        style="italic",
    )
    ax.text(
        0.620,
        0.505,
        "2 epochs on all\nlive groups\n(higher dose)",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=BLUE,
    )

    ax.set_xlim(0.455, 0.700)
    ax.set_ylim(0.20, 0.54)
    ax.set_xlabel("VERL bootstrap best@16\n(coverage proxy)")
    ax.set_ylabel("mean@16  (accuracy)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4)
    ax.grid(color="#dddddd", lw=0.45, alpha=0.55)

    fig.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"fig_countdown_core.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
