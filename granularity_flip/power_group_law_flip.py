#!/usr/bin/env python3
"""Outcome-blind power proxy for the P0 conjunctive decision rule.

The registered analysis uses an exact sign-flip test and a percentile-bootstrap
interval.  Repeating those two resampling procedures inside 100,000 simulated
campaigns would obscure the sample-size decision with unnecessary Monte Carlo
cost.  As in the frozen MAZE-SCORE power memo, this script uses the paired
t-test and t interval as the symmetric-normal proxy while implementing the
observed-mean SESOI clause exactly.  The proxy is for design only; it is never
used to analyze evidence.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import t


DEFAULT_NS = (20, 30, 40, 48)
DEFAULT_EFFECTS = (0.005, 0.0075, 0.0100, 0.0125)
DEFAULT_SDS = (0.0077, 0.0135)
SESOI = 0.005


def simulate_cell(rng: np.random.Generator, n: int, effect: float, sd: float,
                  replications: int) -> tuple[float, float, float]:
    x = rng.normal(effect, sd, size=(replications, n))
    mean = x.mean(axis=1)
    se = x.std(axis=1, ddof=1) / np.sqrt(n)
    critical = t.ppf(0.975, n - 1)
    lower = mean - critical * se
    upper = mean + critical * se
    p_value = 2.0 * t.sf(np.abs(mean / se), n - 1)
    supported = (mean >= SESOI) & (lower > 0.0) & (p_value <= 0.05)
    ruled_out = upper < SESOI
    return (float(supported.mean()), float(ruled_out.mean()),
            float((~supported & ~ruled_out).mean()))


def simulate(replications: int = 100_000, seed: int = 20260820):
    rng = np.random.default_rng(seed)
    rows = []
    for sd in DEFAULT_SDS:
        for n in DEFAULT_NS:
            for effect in DEFAULT_EFFECTS:
                support, ruled, inconclusive = simulate_cell(
                    rng, n, effect, sd, replications)
                rows.append({
                    "sd": sd,
                    "n": n,
                    "effect": effect,
                    "support": support,
                    "practically_ruled_out": ruled,
                    "inconclusive": inconclusive,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    print("sd\tn\teffect\tsupport\truled_out\tinconclusive")
    for row in simulate(args.replications, args.seed):
        print(
            f"{row['sd']:.4f}\t{row['n']}\t{row['effect']:.4f}\t"
            f"{row['support']:.4f}\t{row['practically_ruled_out']:.4f}\t"
            f"{row['inconclusive']:.4f}")


if __name__ == "__main__":
    main()
