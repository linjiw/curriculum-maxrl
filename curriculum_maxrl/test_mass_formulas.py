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

Finally, it independently quadrature-checks the deterministic Beta-posterior
priority quoted in the paper,
E[u_N(p)] = 1 - (b)_N/(a+b)_N - a/(a+b), and its Jensen ordering against the
posterior-mean plug-in score. It also independently enumerates the first two
moments of the realized half-mass
m_N(K)=1{0<K<N}(N-K)/N, including the closed-form posterior-predictive noise
floor proposed for future calibration diagnostics.

Run: python -m curriculum_maxrl.test_mass_formulas
"""
from __future__ import annotations

from math import comb, gamma, prod, sqrt

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


def beta_expected_activity(a: float, b: float, N: int) -> float:
    """Closed-form E[1-(1-p)^N-p] for p ~ Beta(a,b)."""
    survival_moment = prod((b + i) / (a + b + i) for i in range(N))
    return 1.0 - survival_moment - a / (a + b)


def beta_expected_activity_quadrature(a: int, b: int, N: int) -> float:
    """Independent numerical integral for positive integer Beta shapes."""
    nodes, weights = np.polynomial.legendre.leggauss(256)
    p = (nodes + 1.0) / 2.0
    utility = 1.0 - (1.0 - p) ** N - p
    beta_norm = gamma(a) * gamma(b) / gamma(a + b)
    density = p ** (a - 1) * (1.0 - p) ** (b - 1) / beta_norm
    return float(np.dot(weights, utility * density) / 2.0)


def activity_second_moment(p: float, N: int) -> float:
    """Closed-form E[m_N(K)^2 | p] for K ~ Binomial(N,p)."""
    return (1.0 - p) ** 2 + p * (1.0 - p) / N - (1.0 - p) ** N


def beta_activity_second_moment(a: float, b: float, N: int) -> float:
    """Closed-form posterior-predictive E[m_N(K)^2]."""
    s = a + b
    survival_moment = prod((b + i) / (s + i) for i in range(N))
    return (b * (b + 1.0) + a * b / N) / (s * (s + 1.0)) \
        - survival_moment


def beta_binomial_activity_moments(a: float, b: float, N: int) -> tuple[float, float]:
    """Independent enumeration under the Beta-binomial predictive law."""
    beta_norm = gamma(a) * gamma(b) / gamma(a + b)
    mean = 0.0
    second = 0.0
    for k in range(N + 1):
        predictive = comb(N, k) \
            * gamma(a + k) * gamma(b + N - k) / gamma(a + b + N) \
            / beta_norm
        realized = (N - k) / N if 0 < k < N else 0.0
        mean += predictive * realized
        second += predictive * realized**2
    return mean, second


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

    posterior_grid = [
        (1, 1, 2), (1, 1, 8), (3, 2, 4),
        (2, 7, 8), (11, 3, 16), (5, 5, 32),
    ]
    for a, b, N in posterior_grid:
        expected = beta_expected_activity(a, b, N)
        integrated = beta_expected_activity_quadrature(a, b, N)
        assert abs(expected - integrated) < 1e-12, (
            "beta posterior activity", a, b, N, expected, integrated)
        posterior_mean = a / (a + b)
        plug_in = 1 - (1 - posterior_mean) ** N - posterior_mean
        assert expected <= plug_in + 1e-14, (
            "Jensen ordering", a, b, N, expected, plug_in)
        stated_gap = prod((b + i) / (a + b + i) for i in range(N)) \
            - (b / (a + b)) ** N
        assert abs((plug_in - expected) - stated_gap) < 1e-14, (
            "closed-form Jensen gap", a, b, N, plug_in - expected, stated_gap)

        enumerated_mean, enumerated_second = beta_binomial_activity_moments(a, b, N)
        stated_second = beta_activity_second_moment(a, b, N)
        assert abs(enumerated_mean - expected) < 1e-13, (
            "beta-binomial activity mean", a, b, N, enumerated_mean, expected)
        assert abs(enumerated_second - stated_second) < 1e-13, (
            "beta-binomial activity second moment", a, b, N,
            enumerated_second, stated_second)
        assert stated_second - expected**2 >= -1e-14, (
            "posterior-predictive activity variance", a, b, N,
            stated_second - expected**2)

    moment_grid = [(2, 0.0), (2, 0.5), (3, 0.1), (4, 0.25),
                   (8, 0.5), (16, 0.9), (32, 1.0)]
    for N, p in moment_grid:
        enumerated_second = 0.0
        for k in range(N + 1):
            probability = comb(N, k) * p**k * (1 - p) ** (N - k)
            realized = (N - k) / N if 0 < k < N else 0.0
            enumerated_second += probability * realized**2
        stated_second = activity_second_moment(p, N)
        assert abs(enumerated_second - stated_second) < 1e-13, (
            "conditional activity second moment", N, p,
            enumerated_second, stated_second)

    print("all mass-formula enumeration tests passed "
          f"({len(grid)} estimator grid points, both GRPO conventions, "
          f"tail ratios, {len(posterior_grid)} Beta posterior checks, "
          f"{len(moment_grid)} realized-activity moment checks)")


if __name__ == "__main__":
    main()
