#!/usr/bin/env python3
"""Rebuild the balanced-maze evidence at the independent seed-block level.

The two sampler contrasts inside a seed/warm-start block are repeated
observations, not independent replicates.  This script reads the frozen wave-1
and wave-2 factorial result artifacts, recomputes MaxRL-minus-GRPO contrasts
from the cell records, averages the two samplers within each independent
block, and writes both a machine-readable analysis and a seed-block figure.

Run from anywhere:

    python paper/figures/fig_maze_block_analysis.py

Use ``--check`` to verify that committed outputs exactly match a fresh rebuild
without rewriting them.  The JSON output is deterministic; PDF date metadata
is disabled so the figure is deterministic for a fixed plotting stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FACTORIAL = REPO / "curriculum_maxrl" / "maze_gpu_factorial"
RESULTS = REPO / "paper" / "results"

SOURCES = {
    "wave1": FACTORIAL / "results_factorial_wave1.json",
    "wave2": FACTORIAL / "results_factorial_wave2.json",
}
EXPECTED_SEEDS = {"wave1": tuple(range(0, 6)), "wave2": tuple(range(6, 12))}
SAMPLERS = ("uniform", "frontier_un")
METRICS = {
    "coverage_auc": "cov_auc_delta",
    "easy_band_endpoint": "easy_band",
}

JSON_OUT = RESULTS / "maze_factorial_block_analysis.json"
PDF_OUT = HERE / "fig_maze_block_contrasts.pdf"
PNG_OUT = HERE / "fig_maze_block_contrasts.png"

# Fixed two-sided 95% Student-t critical values for the only degrees of
# freedom in this frozen analysis.  Pinning them avoids last-bit drift across
# SciPy versions and keeps ``--check --no-figure`` standard-library-only.
T_CRITICAL_975 = {
    4: 2.7764451051977987,
    5: 2.570581835636314,
    10: 2.2281388519649385,
    11: 2.200985160082949,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign(value: float, tolerance: float = 1e-12) -> int:
    """Classify signs without turning floating cancellation into a result."""
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def _clean_zero(value: float, tolerance: float = 1e-12) -> float:
    return 0.0 if abs(value) <= tolerance else value


def _canonicalize_numbers(value: Any) -> Any:
    """Pin serialized precision across supported Python minor versions."""
    if isinstance(value, float):
        return _clean_zero(round(value, 15))
    if isinstance(value, list):
        return [_canonicalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize_numbers(item) for key, item in value.items()}
    return value


def _t_summary(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(value) for value in values]
    if len(xs) < 2:
        raise ValueError("a t interval requires at least two blocks")
    mean = statistics.mean(xs)
    sample_sd = statistics.stdev(xs)
    standard_error = sample_sd / math.sqrt(len(xs))
    degrees_of_freedom = len(xs) - 1
    try:
        critical = T_CRITICAL_975[degrees_of_freedom]
    except KeyError as exc:
        raise ValueError(
            f"no pinned t critical value for df={degrees_of_freedom}"
        ) from exc
    return {
        "n_independent_blocks": len(xs),
        "n_positive": sum(_sign(value) > 0 for value in xs),
        "n_zero": sum(_sign(value) == 0 for value in xs),
        "n_negative": sum(_sign(value) < 0 for value in xs),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "t_critical_0.975": critical,
        "t_ci_95": [mean - critical * standard_error,
                    mean + critical * standard_error],
    }


def _sign_test_two_sided(n_positive: int, n_nonzero: int) -> float:
    """Exact two-sided fair-coin sign test after excluding exact ties."""
    extreme = max(n_positive, n_nonzero - n_positive)
    upper_tail = sum(
        math.comb(n_nonzero, k) for k in range(extreme, n_nonzero + 1)
    ) / (2 ** n_nonzero)
    return min(1.0, 2.0 * upper_tail)


def _leave_one_out(blocks: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    rows = []
    for omitted in blocks:
        kept = [
            block["sampler_average"][metric]
            for block in blocks
            if block["block_id"] != omitted["block_id"]
        ]
        summary = _t_summary(kept)
        rows.append({
            "omitted_block": omitted["block_id"],
            "remaining_mean": summary["mean"],
            "remaining_t_ci_95": summary["t_ci_95"],
            "all_remaining_block_contrasts_positive": all(value > 0 for value in kept),
        })
    return {
        "analyses": rows,
        "mean_range": [
            min(row["remaining_mean"] for row in rows),
            max(row["remaining_mean"] for row in rows),
        ],
        "ci_lower_bound_range": [
            min(row["remaining_t_ci_95"][0] for row in rows),
            max(row["remaining_t_ci_95"][0] for row in rows),
        ],
        "all_leave_one_out_t_intervals_exclude_zero": all(
            row["remaining_t_ci_95"][0] > 0 for row in rows
        ),
    }


def _cell_contrast(cells: dict[str, Any], sampler: str, seed: int,
                   metric: str) -> float:
    maxrl_key = f"{sampler}/maxrl/s{seed}"
    grpo_key = f"{sampler}/grpo/s{seed}"
    try:
        return float(cells[maxrl_key][metric]) - float(cells[grpo_key][metric])
    except KeyError as exc:
        raise KeyError(
            f"missing {metric} for required block {sampler}/s{seed}"
        ) from exc


def _load_wave(wave: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = json.loads(path.read_text())
    cells = source["cells"]
    blocks = []
    for seed in EXPECTED_SEEDS[wave]:
        sampler_contrasts: dict[str, dict[str, float]] = {}
        for sampler in SAMPLERS:
            sampler_contrasts[sampler] = {
                public_name: _cell_contrast(cells, sampler, seed, source_name)
                for public_name, source_name in METRICS.items()
            }
        sampler_average = {
            metric: _clean_zero(statistics.mean(
                sampler_contrasts[sampler][metric] for sampler in SAMPLERS
            ))
            for metric in METRICS
        }
        blocks.append({
            "block_id": f"{wave}-seed{seed}",
            "wave": int(wave[-1]),
            "seed": seed,
            "sampler_contrasts": sampler_contrasts,
            "sampler_average": sampler_average,
        })

    # The old analyzer stores the coverage contrasts separately.  Comparing
    # against it catches accidental changes to either extraction path.
    for sampler in SAMPLERS:
        key = f"expl-AUC {sampler}: maxrl-grpo cov_auc_delta"
        stored = [float(value) for value in source["contrasts"][key]["per_seed"]]
        rebuilt = [
            block["sampler_contrasts"][sampler]["coverage_auc"]
            for block in blocks
        ]
        if len(stored) != len(rebuilt) or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
            for a, b in zip(stored, rebuilt)
        ):
            raise AssertionError(f"{wave}/{sampler} contrast extraction drifted")
    return source, blocks


def build_analysis() -> dict[str, Any]:
    wave_blocks: dict[str, list[dict[str, Any]]] = {}
    source_records = []
    for wave, path in SOURCES.items():
        _, wave_blocks[wave] = _load_wave(wave, path)
        source_records.append({
            "wave": int(wave[-1]),
            "path": str(path.relative_to(REPO)),
            "sha256": _sha256(path),
        })

    wave2 = wave_blocks["wave2"]
    per_sampler = {}
    for sampler in SAMPLERS:
        values = [
            block["sampler_contrasts"][sampler]["coverage_auc"]
            for block in wave2
        ]
        n_positive = sum(_sign(value) > 0 for value in values)
        n_nonzero = sum(_sign(value) != 0 for value in values)
        per_sampler[sampler] = {
            "n_positive": n_positive,
            "n_blocks": len(values),
            "mean": statistics.mean(values),
            "exact_two_sided_sign_test_p": _sign_test_two_sided(
                n_positive, n_nonzero
            ),
            "per_block": values,
        }

    wave2_coverage_values = [
        block["sampler_average"]["coverage_auc"] for block in wave2
    ]
    wave2_easy_values = [
        block["sampler_average"]["easy_band_endpoint"] for block in wave2
    ]
    easy_repeated_values = [
        block["sampler_contrasts"][sampler]["easy_band_endpoint"]
        for block in wave2
        for sampler in SAMPLERS
    ]
    all_blocks = wave_blocks["wave1"] + wave2
    all_coverage_values = [
        block["sampler_average"]["coverage_auc"] for block in all_blocks
    ]

    analysis = {
        "schema_version": 1,
        "analysis_id": "maze-factorial-independent-seed-blocks-v1",
        "generated_by": "paper/figures/fig_maze_block_analysis.py",
        "sources": source_records,
        "design": {
            "independent_unit": "independently trained seed/warm-start block",
            "within_block_repeated_factor": "sampler (uniform or frontier_un)",
            "contrast": "MaxRL minus GRPO",
            "coverage_metric": (
                "mean in-training pass@8 coverage minus post-SFT initial "
                "coverage, then contrasted between estimators"
            ),
            "easy_band_metric": (
                "mean endpoint pass@8 change over levels 1-3, then contrasted "
                "between estimators"
            ),
            "registered_confirmation": (
                "wave 2 coverage-AUC direction under each sampler separately"
            ),
            "interval_note": (
                "All reported t intervals are post-hoc descriptive intervals "
                "over sampler-averaged independent seed blocks."
            ),
        },
        "waves": {
            wave: {"blocks": blocks}
            for wave, blocks in wave_blocks.items()
        },
        "wave2_registered_confirmation": {
            "per_sampler": per_sampler,
            "criterion_result": (
                "positive in 6/6 independent seed blocks under uniform and "
                "6/6 under frontier_un"
            ),
            "sampler_averaged_block_summary": _t_summary(wave2_coverage_values),
            "leave_one_block_out": _leave_one_out(wave2, "coverage_auc"),
        },
        "cross_wave_descriptive": {
            "scope": (
                "exploratory wave 1 plus registered confirmation wave 2; no "
                "pooled confirmatory p-value"
            ),
            "sampler_averaged_block_summary": _t_summary(all_coverage_values),
            "leave_one_block_out": _leave_one_out(all_blocks, "coverage_auc"),
        },
        "wave2_easy_band_descriptive": {
            "within_block_sampler_observations": {
                "n_positive": sum(_sign(value) > 0 for value in easy_repeated_values),
                "n_zero": sum(_sign(value) == 0 for value in easy_repeated_values),
                "n_negative": sum(_sign(value) < 0 for value in easy_repeated_values),
                "n_repeated_observations": len(easy_repeated_values),
                "warning": (
                    "These 12 observations are correlated within six blocks "
                    "and are not 12 independent replicates."
                ),
            },
            "sampler_averaged_block_summary": _t_summary(wave2_easy_values),
            "interpretation": (
                "Four positive blocks, one exact tie, and one negative block; "
                "the block-level 95% t interval includes zero, so easy-band "
                "localization is descriptive rather than established."
            ),
        },
    }
    return _canonicalize_numbers(analysis)


def _write_json(analysis: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")


def _render_figure(analysis: dict[str, Any], pdf_out: Path, png_out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {
        "uniform": "#2A78D6",
        "frontier_un": "#E07A31",
        "average": "#222222",
        "wave1": "#8A8A8A",
        "wave2": "#3B7EA1",
    }

    wave1 = analysis["waves"]["wave1"]["blocks"]
    wave2 = analysis["waves"]["wave2"]["blocks"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))

    # (a) The two fresh-wave sampler checks are repeated measurements nested
    # inside each of the six fresh independent blocks.
    ax = axes[0]
    for x, block in enumerate(wave2):
        uniform = block["sampler_contrasts"]["uniform"]["coverage_auc"]
        frontier = block["sampler_contrasts"]["frontier_un"]["coverage_auc"]
        average = block["sampler_average"]["coverage_auc"]
        ax.plot([x - 0.10, x + 0.10], [uniform, frontier], color="#B8B8B8",
                linewidth=0.8, zorder=1)
        ax.scatter(x - 0.10, uniform, color=colors["uniform"], s=20,
                   marker="o", zorder=3)
        ax.scatter(x + 0.10, frontier, color=colors["frontier_un"], s=22,
                   marker="s", zorder=3)
        ax.scatter(x, average, facecolor="white", edgecolor=colors["average"],
                   linewidth=0.8, s=25, marker="D", zorder=4)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(range(6), [str(block["seed"]) for block in wave2])
    ax.set_xlabel("fresh seed block")
    ax.set_ylabel(r"MaxRL $-$ GRPO, coverage AUC")
    ax.set_title("(a) Fresh wave-2 result", loc="left")
    ax.set_ylim(-0.002, 0.052)
    ax.text(0.02, 0.97, "6/6 under each sampler\nexact $p=.031$ each",
            transform=ax.transAxes, va="top", ha="left", fontsize=7)

    # (b) One dot per independent block across both waves.  The pooled summary
    # is descriptive because only wave 2 was the fresh confirmation.
    ax = axes[1]
    all_blocks = wave1 + wave2
    for x, block in enumerate(all_blocks):
        color = colors[f"wave{block['wave']}"]
        ax.scatter(x, block["sampler_average"]["coverage_auc"], color=color,
                   s=24, zorder=3)
    summary = analysis["cross_wave_descriptive"]["sampler_averaged_block_summary"]
    lo, hi = summary["t_ci_95"]
    ax.axhspan(lo, hi, color="#8A8A8A", alpha=0.15, linewidth=0)
    ax.axhline(summary["mean"], color="#333333", linewidth=1.0)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.axvline(5.5, color="#BBBBBB", linewidth=0.7, linestyle="--")
    ax.set_xticks([2.5, 8.5], ["wave 1\nseeds 0–5", "wave 2\nseeds 6–11"])
    ax.set_ylabel("sampler-averaged contrast")
    ax.set_title("(b) Independent blocks", loc="left")
    ax.set_ylim(-0.002, 0.058)
    ax.text(0.02, 0.97,
            f"12/12 positive (descriptive)\nmean {summary['mean']:+.4f}\n"
            f"95% t CI [{lo:+.4f}, {hi:+.4f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                  "pad": 1.0})

    # (c) The easy-band signal weakens after averaging repeated sampler
    # observations within each independent block.
    ax = axes[2]
    for x, block in enumerate(wave2):
        uniform = block["sampler_contrasts"]["uniform"]["easy_band_endpoint"]
        frontier = block["sampler_contrasts"]["frontier_un"]["easy_band_endpoint"]
        average = block["sampler_average"]["easy_band_endpoint"]
        ax.plot([x - 0.10, x + 0.10], [uniform, frontier], color="#B8B8B8",
                linewidth=0.8, zorder=1)
        ax.scatter(x - 0.10, uniform, color=colors["uniform"], s=17,
                   marker="o", zorder=3, alpha=0.75)
        ax.scatter(x + 0.10, frontier, color=colors["frontier_un"], s=19,
                   marker="s", zorder=3, alpha=0.75)
        ax.scatter(x, average, facecolor="white", edgecolor=colors["average"],
                   linewidth=0.8, s=25, marker="D", zorder=4)
    easy = analysis["wave2_easy_band_descriptive"]["sampler_averaged_block_summary"]
    lo, hi = easy["t_ci_95"]
    ax.axhspan(lo, hi, color="#8A8A8A", alpha=0.15, linewidth=0)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(range(6), [str(block["seed"]) for block in wave2])
    ax.set_xlabel("fresh seed block")
    ax.set_ylabel(r"MaxRL $-$ GRPO, easy band")
    ax.set_title("(c) Easy band (descriptive)", loc="left")
    ax.set_ylim(-0.08, 0.34)
    ax.text(0.02, 0.97,
            f"4 positive, 1 tie, 1 negative\n95% t CI [{lo:+.4f}, {hi:+.4f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                  "pad": 1.0})

    fig.legend(handles=[
        Line2D([], [], color=colors["uniform"], marker="o", linestyle="None",
               label="uniform"),
        Line2D([], [], color=colors["frontier_un"], marker="s", linestyle="None",
               label="frontier"),
        Line2D([], [], color=colors["average"], marker="D", markerfacecolor="white",
               linestyle="None", label="within-block average"),
    ], loc="lower center", ncol=3, frameon=False, handletextpad=0.25,
       columnspacing=0.9, borderpad=0.1, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.94), pad=0.45, w_pad=1.25)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_out, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png_out, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="verify committed JSON matches a fresh rebuild without rewriting",
    )
    parser.add_argument(
        "--no-figure", action="store_true",
        help="rebuild or check the JSON without importing matplotlib",
    )
    args = parser.parse_args()
    analysis = build_analysis()

    if args.check:
        expected = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
        if not JSON_OUT.exists() or JSON_OUT.read_text() != expected:
            raise SystemExit(f"stale or missing analysis: {JSON_OUT}")
        if not args.no_figure:
            with tempfile.TemporaryDirectory(prefix="maze-block-check-") as tmp:
                tmp_path = Path(tmp)
                fresh_pdf = tmp_path / PDF_OUT.name
                fresh_png = tmp_path / PNG_OUT.name
                _render_figure(analysis, fresh_pdf, fresh_png)
                for committed, fresh in ((PDF_OUT, fresh_pdf), (PNG_OUT, fresh_png)):
                    if not committed.exists() or _sha256(committed) != _sha256(fresh):
                        raise SystemExit(f"stale or missing figure: {committed}")
        print("maze block analysis is current")
        return

    _write_json(analysis, JSON_OUT)
    print(f"wrote {JSON_OUT.relative_to(REPO)}")
    if not args.no_figure:
        _render_figure(analysis, PDF_OUT, PNG_OUT)
        print(f"wrote {PDF_OUT.relative_to(REPO)}")
        print(f"wrote {PNG_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
