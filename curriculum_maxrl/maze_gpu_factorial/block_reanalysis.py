"""Reanalyze the maze factorial at the independent seed-block level.

Each seed contains two repeated sampler contrasts. They are reported
separately for the registered per-sampler wave-2 test, then averaged within
seed for block-level intervals and all cross-wave summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


SAMPLERS = ("uniform", "frontier_un")
T_975 = {
    6: 2.570581835636305,
    12: 2.200985160091638,
}
ZERO_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict:
    n = len(values)
    average = mean(values)
    standard_error = stdev(values) / math.sqrt(n)
    critical = T_975[n]
    return {
        "n_independent_seed_blocks": n,
        "mean": average,
        "sample_sd": stdev(values),
        "standard_error": standard_error,
        "t_critical_0.975": critical,
        "ci95_t": [average - critical * standard_error,
                   average + critical * standard_error],
        "positive": sum(value > ZERO_TOLERANCE for value in values),
        "ties": sum(abs(value) <= ZERO_TOLERANCE for value in values),
        "negative": sum(value < -ZERO_TOLERANCE for value in values),
        "values": values,
    }


def cell_contrast(cells: dict, sampler: str, seed: int, field: str) -> float:
    return (float(cells[f"{sampler}/maxrl/s{seed}"][field]) -
            float(cells[f"{sampler}/grpo/s{seed}"][field]))


def build(root: Path) -> dict:
    factorial_paths = {
        "wave1": root / "results_factorial_wave1.json",
        "wave2": root / "results_factorial_wave2.json",
    }
    premium_path = root / "premium_reanalysis.json"
    factorial = {
        wave: json.loads(path.read_text(encoding="utf-8"))
        for wave, path in factorial_paths.items()
    }
    premium = json.loads(premium_path.read_text(encoding="utf-8"))
    wave_seeds = {"wave1": list(range(0, 6)),
                  "wave2": list(range(6, 12))}
    waves = {}
    cross_wave_rows = []
    for wave, seeds in wave_seeds.items():
        rows = []
        for position, seed in enumerate(seeds):
            cov = {sampler: cell_contrast(
                factorial[wave]["cells"], sampler, seed, "cov_auc_delta")
                for sampler in SAMPLERS}
            easy = {sampler: cell_contrast(
                factorial[wave]["cells"], sampler, seed, "easy_band")
                for sampler in SAMPLERS}
            premium_values = {
                sampler: float(
                    premium[f"{wave}/{sampler}"]["per_seed"][position])
                for sampler in SAMPLERS
            }
            row = {
                "seed": seed,
                "cov_auc_maxrl_minus_grpo": cov,
                "cov_auc_sampler_average": mean(cov.values()),
                "easy_band_maxrl_minus_grpo": easy,
                "easy_band_sampler_average": mean(easy.values()),
                "premium_auc_maxrl_minus_grpo": premium_values,
                "premium_auc_sampler_average": mean(
                    premium_values.values()),
            }
            rows.append(row)
            cross_wave_rows.append({"wave": wave, **row})

        waves[wave] = {
            "seeds": seeds,
            "blocks": rows,
            "block_level": {
                "cov_auc": summary([
                    row["cov_auc_sampler_average"] for row in rows]),
                "easy_band": summary([
                    row["easy_band_sampler_average"] for row in rows]),
                "premium_auc_exploratory": summary([
                    row["premium_auc_sampler_average"] for row in rows]),
            },
            "repeated_sampler_contrasts": {
                sampler: {
                    "cov_auc": summary([
                        row["cov_auc_maxrl_minus_grpo"][sampler]
                        for row in rows]),
                    "easy_band": summary([
                        row["easy_band_maxrl_minus_grpo"][sampler]
                        for row in rows]),
                    "premium_auc_exploratory": summary([
                        row["premium_auc_maxrl_minus_grpo"][sampler]
                        for row in rows]),
                } for sampler in SAMPLERS
            },
        }

    wave2_pair_easy = [
        row["easy_band_maxrl_minus_grpo"][sampler]
        for row in waves["wave2"]["blocks"] for sampler in SAMPLERS
    ]
    return {
        "analysis": "factorial independent-seed-block reanalysis",
        "generated_from": {
            str(path.relative_to(root.parent.parent)): sha256(path)
            for path in [*factorial_paths.values(), premium_path]
        },
        "independent_unit": (
            "seed/warmstart block; uniform and frontier_un are repeated "
            "sampler contrasts within a block"),
        "zero_tolerance": ZERO_TOLERANCE,
        "waves": waves,
        "cross_wave_exploratory": {
            "n_independent_seed_blocks": 12,
            "cov_auc": summary([
                row["cov_auc_sampler_average"] for row in cross_wave_rows]),
            "easy_band": summary([
                row["easy_band_sampler_average"] for row in cross_wave_rows]),
            "premium_auc": summary([
                row["premium_auc_sampler_average"]
                for row in cross_wave_rows]),
            "scope": (
                "Wave 1 was exploratory and wave 2 confirmatory; this "
                "cross-wave summary is descriptive, not a pooled "
                "confirmatory test."),
        },
        "registered_wave2_readout": {
            "P-F2": {
                "status": "registered_bar_met",
                "uniform_positive": 6,
                "uniform_n": 6,
                "frontier_un_positive": 6,
                "frontier_un_n": 6,
                "exact_two_sided_sign_p_per_sampler": 0.03125,
                "block_level_cov_auc": waves["wave2"]["block_level"][
                    "cov_auc"],
            },
            "P-F3": {
                "status": "registered_pair_level_bar_met_with_unit_caveat",
                "registered_pair_level_positive": sum(
                    value > ZERO_TOLERANCE for value in wave2_pair_easy),
                "registered_pair_level_ties": sum(
                    abs(value) <= ZERO_TOLERANCE for value in wave2_pair_easy),
                "registered_pair_level_negative": sum(
                    value < -ZERO_TOLERANCE for value in wave2_pair_easy),
                "registered_pair_level_n": 12,
                "registered_bar": ">=7/12 positive repeated contrasts",
                "block_level_easy_band": waves["wave2"]["block_level"][
                    "easy_band"],
                "interpretation": (
                    "The registered pair-level bar was met, but the two "
                    "sampler contrasts within each seed are correlated. "
                    "Localization to the easy band is suggestive, not "
                    "established."),
            },
        },
        "reporting_rule": (
            "Quote 6/6 per sampler for registered P-F2 and one "
            "sampler-averaged value per seed block for intervals and "
            "cross-wave counts. Never call 24 sampler contrasts 24 "
            "independent blocks."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--output", default="block_reanalysis.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    result = build(root)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "wave2_cov_auc": result["waves"]["wave2"]["block_level"][
            "cov_auc"],
        "wave2_easy_band": result["waves"]["wave2"]["block_level"][
            "easy_band"],
        "cross_wave_cov_auc": result["cross_wave_exploratory"]["cov_auc"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
