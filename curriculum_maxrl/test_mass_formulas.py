"""Enumeration tests for the paper's coefficient-mass formulas (review P2).

For each estimator, the exact expected absolute coefficient mass is
computed by enumerating K ~ Bin(N, p) and compared against the analytic
formula quoted in the paper, for several (N, p):

  MaxRL (drop K=0):        A_N(p) = 2 * (1 - (1-p)^N - p)      (Prop. 1)
  RLOO (1/N-normalized):   A_N(p) = 2 p (1-p)
  GRPO, sample SD (ddof=1, deployed):
      A_N(p) = 2 sqrt((N-1)/N) * (1/N) E[sqrt(K(N-K))]
  GRPO, population SD (ddof=0, NOT deployed):
      A_N(p) = 2 * (1/N) E[sqrt(K(N-K))]

Also checks the deployed-code path in estimators.py agrees with the
sample-SD formula (up to the EPS regularizer), and the tail ratios
quoted in the paper: MaxRL/GRPO -> sqrt(N) as p->0 and
GRPO/MaxRL -> (N-1)/sqrt(N) as p->1 for sample SD.

Run: python -m curriculum_maxrl.test_mass_formulas
"""
from __future__ import annotations

from math import comb, sqrt

import numpy as np

from .estimators import weights_grpo, weights_maxrl, weights_rloo


def enum_mass(weight_fn, N: int, p: float) -> float:
    """Exact E[sum_i |w_i|] by enumerating K (weights depend on r only
    through K for all these estimators; use a canonical r per K)."""
    total = 0.0
    for K in range(N + 1):
        prob = comb(N, K) * p**K * (1 - p) ** (N - K)
        r = np.array([1.0] * K + [0.0] * (N - K))
        total += prob * float(np.abs(weight_fn(r)).sum())
    return total


def grpo_mass_analytic(N: int, p: float, ddof: int) -> float:
    e = sum(comb(N, k) * p**k * (1 - p) ** (N - k) * sqrt(k * (N - k))
            for k in range(1, N))
    scale = sqrt((N - 1) / N) if ddof == 1 else 1.0
    return 2 * scale * e / N


def main() -> None:
    grid = [(4, 0.05), (4, 0.5), (8, 0.2), (16, 0.1), (16, 0.7),
            (32, 0.03), (32, 0.95)]
    tol = 1e-9
    eps_tol = 1e-4  # deployed GRPO carries an EPS in the denominator

    for N, p in grid:
        maxrl = enum_mass(weights_maxrl, N, p)
        want = 2 * (1 - (1 - p) ** N - p)
        assert abs(maxrl - want) < tol, ("maxrl", N, p, maxrl, want)

        rloo = enum_mass(weights_rloo, N, p)
        want = 2 * p * (1 - p)
        assert abs(rloo - want) < tol, ("rloo", N, p, rloo, want)

        grpo = enum_mass(weights_grpo, N, p)
        want = grpo_mass_analytic(N, p, ddof=1)
        assert abs(grpo - want) < eps_tol, ("grpo", N, p, grpo, want)

        pop = grpo_mass_analytic(N, p, ddof=0)
        assert abs(want - sqrt((N - 1) / N) * pop) < tol

    # tail ratios for the deployed (sample-SD) convention
    N = 16
    for p in (1e-6, 1e-7):
        ratio = 2 * (1 - (1 - p) ** N - p) / grpo_mass_analytic(N, p, 1)
        assert abs(ratio - sqrt(N)) < 1e-2, ("p->0 tail", p, ratio)
    for q in (1e-6, 1e-7):
        p = 1 - q
        ratio = grpo_mass_analytic(N, p, 1) / (2 * (1 - (1 - p) ** N - p))
        assert abs(ratio - (N - 1) / sqrt(N)) < 1e-2, ("p->1 tail", q, ratio)

    print("all mass-formula enumeration tests passed "
          f"({len(grid)} grid points, both GRPO conventions, tail ratios)")


if __name__ == "__main__":
    main()
