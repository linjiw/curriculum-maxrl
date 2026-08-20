#!/usr/bin/env python3
"""Render the independent-block maze factorial summary used in the compact paper.

The plotted values come only from ``block_reanalysis.json``.  Uniform and
frontier contrasts are repeated observations inside a seed/warmstart block;
the open diamond is their within-block average and is the independent unit for
intervals and cross-wave counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent.parent / "curriculum_maxrl" / "maze_gpu_factorial" / "block_reanalysis.json"

BLUE = "#2a78d6"
ORANGE = "#df7f3b"
GRAY = "#8d8d8d"
LIGHT_GRAY = "#e7e7e7"

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _counts(values: np.ndarray, tol: float = 1e-12) -> tuple[int, int, int]:
    return (
        int(np.sum(values > tol)),
        int(np.sum(np.abs(values) <= tol)),
        int(np.sum(values < -tol)),
    )


def main() -> None:
    data = json.loads(INPUT.read_text())
    wave2 = data["waves"]["wave2"]
    blocks = wave2["blocks"]
    assert [row["seed"] for row in blocks] == list(range(6, 12))
    assert data["independent_unit"].startswith("seed/warmstart block")

    seeds = np.asarray([row["seed"] for row in blocks])
    cov_uniform = np.asarray(
        [row["cov_auc_maxrl_minus_grpo"]["uniform"] for row in blocks]
    )
    cov_frontier = np.asarray(
        [row["cov_auc_maxrl_minus_grpo"]["frontier_un"] for row in blocks]
    )
    cov_average = np.asarray([row["cov_auc_sampler_average"] for row in blocks])
    easy_uniform = np.asarray(
        [row["easy_band_maxrl_minus_grpo"]["uniform"] for row in blocks]
    )
    easy_frontier = np.asarray(
        [row["easy_band_maxrl_minus_grpo"]["frontier_un"] for row in blocks]
    )
    easy_average = np.asarray([row["easy_band_sampler_average"] for row in blocks])

    registered = data["registered_wave2_readout"]
    assert registered["P-F2"]["uniform_positive"] == 6
    assert registered["P-F2"]["frontier_un_positive"] == 6
    assert np.allclose(cov_average, (cov_uniform + cov_frontier) / 2)
    assert np.allclose(easy_average, (easy_uniform + easy_frontier) / 2)
    assert _counts(easy_average) == (4, 1, 1)

    all_cov = np.asarray(data["cross_wave_exploratory"]["cov_auc"]["values"])
    assert len(all_cov) == 12 and _counts(all_cov) == (12, 0, 0)
    all_mean = data["cross_wave_exploratory"]["cov_auc"]["mean"]
    all_ci = data["cross_wave_exploratory"]["cov_auc"]["ci95_t"]
    easy_ci = registered["P-F3"]["block_level_easy_band"]["ci95_t"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))

    ax = axes[0]
    for x, a, b in zip(seeds, cov_uniform, cov_frontier):
        ax.plot([x, x], [a, b], color="#b7b7b7", lw=1, zorder=1)
    ax.scatter(seeds - 0.08, cov_uniform, s=26, color=BLUE, label="uniform", zorder=3)
    ax.scatter(
        seeds + 0.08,
        cov_frontier,
        s=28,
        color=ORANGE,
        marker="s",
        label="frontier",
        zorder=3,
    )
    ax.scatter(
        seeds,
        cov_average,
        s=38,
        facecolor="white",
        edgecolor="#333333",
        marker="D",
        linewidth=1,
        label="within-block average",
        zorder=4,
    )
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_title("(a) Fresh wave-2 result", loc="left")
    ax.set_xlabel("fresh seed block")
    ax.set_ylabel("MaxRL − GRPO, coverage AUC")
    ax.set_xticks(seeds)
    ax.set_ylim(-0.002, 0.052)
    ax.text(
        0.02,
        0.97,
        "6/6 under each sampler\nexact $p=.031$ each",
        transform=ax.transAxes,
        va="top",
    )

    ax = axes[1]
    xs = np.r_[np.arange(6), np.arange(7, 13)]
    ax.axhspan(all_ci[0], all_ci[1], color=LIGHT_GRAY, zorder=0)
    ax.axhline(all_mean, color="#333333", lw=1.1, zorder=1)
    ax.axvline(6.5, color="#aaaaaa", lw=0.7, ls="--")
    ax.scatter(xs[:6], all_cov[:6], s=29, color=GRAY, zorder=3)
    ax.scatter(xs[6:], all_cov[6:], s=29, color="#367fa5", zorder=3)
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_title("(b) Independent blocks", loc="left")
    ax.set_ylabel("sampler-averaged contrast")
    ax.set_xticks([2.5, 9.5], ["wave 1\nseeds 0–5", "wave 2\nseeds 6–11"])
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.002, 0.058)
    ax.text(
        0.02,
        0.97,
        f"12/12 positive (descriptive)\nmean +{all_mean:.4f}\n"
        f"95% t CI [{all_ci[0]:+.4f}, {all_ci[1]:+.4f}]",
        transform=ax.transAxes,
        va="top",
    )

    ax = axes[2]
    for x, a, b in zip(seeds, easy_uniform, easy_frontier):
        ax.plot([x, x], [a, b], color="#b7b7b7", lw=1, zorder=1)
    ax.scatter(seeds - 0.08, easy_uniform, s=26, color=BLUE, alpha=0.75, zorder=3)
    ax.scatter(
        seeds + 0.08,
        easy_frontier,
        s=28,
        color=ORANGE,
        alpha=0.75,
        marker="s",
        zorder=3,
    )
    ax.scatter(
        seeds,
        easy_average,
        s=38,
        facecolor="white",
        edgecolor="#333333",
        marker="D",
        linewidth=1,
        zorder=4,
    )
    ax.axhspan(easy_ci[0], easy_ci[1], color=LIGHT_GRAY, zorder=0)
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_title("(c) Easy band (descriptive)", loc="left", fontsize=8.4)
    ax.set_xlabel("fresh seed block")
    ax.set_ylabel("MaxRL − GRPO, easy band")
    ax.set_xticks(seeds)
    ax.set_ylim(-0.08, 0.34)
    ax.text(
        0.02,
        0.97,
        f"4 positive, 1 tie, 1 negative\n"
        f"95% t CI [{easy_ci[0]:+.4f}, {easy_ci[1]:+.4f}]",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.2)
    fig.savefig(HERE / "fig_maze_block_contrasts.pdf")
    fig.savefig(HERE / "fig_maze_block_contrasts.png", dpi=150)
    print("wrote fig_maze_block_contrasts.pdf / .png")


if __name__ == "__main__":
    main()
