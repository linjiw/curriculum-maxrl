"""Exact CPU stress test for practical-MaxRL mass under correlated rollouts.

The paper's closed form ``2 * (1 - (1-p)**N - p)`` uses conditionally
independent Bernoulli rollouts.  The more general distribution-free identity is

    E[A] = 2 * (P(K >= 1) - E[K]/N),

where ``A`` is the practical estimator's absolute coefficient mass.  With a
common marginal success probability, ``E[K]/N=p``.  This script evaluates the
identity exactly for beta-binomial groups, whose intra-class correlation is
``rho``, and quantifies the error from substituting the i.i.d. hit probability.

The experiment is analytic except for a small, fixed-seed Monte Carlo audit.
It requires only NumPy and runs on a laptop CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("results_correlated_rollout_stress.json")
GROUP_SIZES = (2, 4, 8, 16, 32)
CORRELATIONS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 0.80)
GRID_POINTS = 200_001
MONTE_CARLO_GROUPS = 200_000
MONTE_CARLO_SEED = 20_260_808


@dataclass(frozen=True)
class StressConfig:
    group_sizes: tuple[int, ...] = GROUP_SIZES
    correlations: tuple[float, ...] = CORRELATIONS
    grid_points: int = GRID_POINTS
    monte_carlo_groups: int = MONTE_CARLO_GROUPS
    monte_carlo_seed: int = MONTE_CARLO_SEED


def validate_config(config: StressConfig) -> None:
    if not config.group_sizes or any(type(n) is not int or n < 2 for n in config.group_sizes):
        raise ValueError("group_sizes must contain integers >= 2")
    if len(set(config.group_sizes)) != len(config.group_sizes):
        raise ValueError("group_sizes must be unique")
    if not config.correlations:
        raise ValueError("correlations cannot be empty")
    if any(not 0.0 <= float(rho) < 1.0 for rho in config.correlations):
        raise ValueError("correlations must lie in [0, 1)")
    if len(set(float(rho) for rho in config.correlations)) != len(config.correlations):
        raise ValueError("correlations must be unique")
    if (
        type(config.grid_points) is not int
        or config.grid_points < 10_001
        or config.grid_points % 2 == 0
    ):
        raise ValueError("grid_points must be an odd integer >= 10001")
    if type(config.monte_carlo_groups) is not int or config.monte_carlo_groups < 10_000:
        raise ValueError("monte_carlo_groups must be >= 10000")
    if type(config.monte_carlo_seed) is not int or config.monte_carlo_seed < 0:
        raise ValueError("monte_carlo_seed must be a non-negative integer")


def iid_half_mass(p: np.ndarray | float, n: int) -> np.ndarray:
    """Return ``P_iid(K>=1)-p`` for Bernoulli groups of size ``n``."""
    p_array = np.asarray(p, dtype=np.float64)
    return np.maximum(1.0 - np.power(1.0 - p_array, n) - p_array, 0.0)


def beta_binomial_zero_probability(
    p: np.ndarray | float, n: int, rho: float
) -> np.ndarray:
    """Exact ``P(K=0)`` for an exchangeable beta-binomial group.

    A latent probability ``Q ~ Beta(p*c, (1-p)*c)`` with
    ``c=1/rho-1`` yields marginal success ``p`` and, for ``0<p<1``, pairwise
    correlation ``rho``. Correlation is undefined at degenerate endpoint
    marginals. ``rho=0`` is evaluated by its i.i.d. limit.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0.0 <= float(rho) < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    p_array = np.asarray(p, dtype=np.float64)
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("p must lie in [0, 1]")
    if rho == 0.0:
        return np.power(1.0 - p_array, n)
    concentration = 1.0 / float(rho) - 1.0
    beta = (1.0 - p_array) * concentration
    probability = np.ones_like(p_array, dtype=np.float64)
    for j in range(n):
        probability *= (beta + j) / (concentration + j)
    return probability


def beta_binomial_half_mass(
    p: np.ndarray | float, n: int, rho: float
) -> np.ndarray:
    """Return expected half-mass in the registered beta-binomial family."""
    p_array = np.asarray(p, dtype=np.float64)
    hit = 1.0 - beta_binomial_zero_probability(p_array, n, rho)
    return np.maximum(hit - p_array, 0.0)


def beta_binomial_pmf(k: int, n: int, p: float, rho: float) -> float:
    """Exact beta-binomial mass, with the binomial limit at ``rho=0``."""
    if not 0 <= k <= n:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if rho == 0.0:
        return math.comb(n, k) * p**k * (1.0 - p) ** (n - k)
    if p in (0.0, 1.0):
        return float(k == int(n * p))
    concentration = 1.0 / rho - 1.0
    alpha = p * concentration
    beta = (1.0 - p) * concentration
    log_mass = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + math.lgamma(alpha + k)
        + math.lgamma(beta + n - k)
        - math.lgamma(alpha + beta + n)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(alpha + beta)
    )
    return math.exp(log_mass)


def expected_mass_from_count_distribution(n: int, p: float, rho: float) -> float:
    """Enumerate ``K`` and average the deployed per-group absolute mass."""
    expected = 0.0
    for k in range(n + 1):
        realized = 0.0 if k in (0, n) else 2.0 * (n - k) / n
        expected += beta_binomial_pmf(k, n, p, rho) * realized
    return expected


def iid_peak(n: int) -> float:
    return 1.0 - n ** (-1.0 / (n - 1.0))


def _grid_peak(n: int, rho: float, grid_points: int) -> tuple[float, float]:
    if rho == 0.0:
        p_peak = iid_peak(n)
        return p_peak, float(iid_half_mass(p_peak, n))
    grid = np.linspace(1e-6, 1.0 - 1e-6, grid_points, dtype=np.float64)
    values = beta_binomial_half_mass(grid, n, rho)
    index = int(np.argmax(values))
    return float(grid[index]), float(values[index])


def _identity_audit(config: StressConfig) -> dict:
    p_values = (0.01, 0.05, 0.15, 0.35, 0.65, 0.90)
    rows = []
    max_error = 0.0
    max_probability_error = 0.0
    for n in config.group_sizes:
        for rho in config.correlations:
            for p in p_values:
                pmf = np.asarray(
                    [beta_binomial_pmf(k, n, p, rho) for k in range(n + 1)]
                )
                probability_error = abs(float(pmf.sum()) - 1.0)
                enumerated = expected_mass_from_count_distribution(n, p, rho)
                identity = 2.0 * float(beta_binomial_half_mass(p, n, rho))
                error = abs(enumerated - identity)
                max_error = max(max_error, error)
                max_probability_error = max(max_probability_error, probability_error)
                rows.append(
                    {
                        "n": n,
                        "rho": rho,
                        "p": p,
                        "enumerated_expected_mass": enumerated,
                        "identity_expected_mass": identity,
                        "absolute_error": error,
                        "pmf_sum_error": probability_error,
                    }
                )
    return {
        "cases": len(rows),
        "max_absolute_identity_error": max_error,
        "max_pmf_sum_error": max_probability_error,
        "tolerance": 1e-11,
        "passed": max_error < 1e-11 and max_probability_error < 1e-11,
        "rows": rows,
    }


def _monte_carlo_audit(config: StressConfig) -> dict:
    rng = np.random.default_rng(config.monte_carlo_seed)
    rows = []
    for n in (4, 16, 32):
        if n not in config.group_sizes:
            continue
        p = iid_peak(n)
        for rho in (0.0, 0.10, 0.50):
            if rho not in config.correlations:
                continue
            if rho == 0.0:
                counts = rng.binomial(n, p, size=config.monte_carlo_groups)
            else:
                concentration = 1.0 / rho - 1.0
                latent = rng.beta(
                    p * concentration,
                    (1.0 - p) * concentration,
                    size=config.monte_carlo_groups,
                )
                counts = rng.binomial(n, latent)
            masses = np.where(
                (counts > 0) & (counts < n),
                2.0 * (n - counts) / n,
                0.0,
            )
            estimate = float(masses.mean())
            standard_error = float(masses.std(ddof=1) / math.sqrt(len(masses)))
            exact = 2.0 * float(beta_binomial_half_mass(p, n, rho))
            rows.append(
                {
                    "n": n,
                    "rho": rho,
                    "p_iid_peak": p,
                    "groups": config.monte_carlo_groups,
                    "estimated_expected_mass": estimate,
                    "exact_expected_mass": exact,
                    "absolute_error": abs(estimate - exact),
                    "standard_error": standard_error,
                    "absolute_z": (
                        abs(estimate - exact) / standard_error
                        if standard_error > 0.0
                        else 0.0
                    ),
                }
            )
    max_z = max(row["absolute_z"] for row in rows) if rows else 0.0
    return {
        "seed": config.monte_carlo_seed,
        "rows": rows,
        "max_absolute_z": max_z,
        "diagnostic_only": (
            "The exact enumeration is the verification target; fixed-seed Monte "
            "Carlo is retained only as an implementation sanity check."
        ),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        "curriculum_maxrl/run_correlated_rollout_stress.py",
        "curriculum_maxrl/estimators.py",
    )
    return {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in paths
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None


def build_results(config: StressConfig = StressConfig()) -> dict:
    validate_config(config)
    peak_rows = []
    iid_dominance_violations = []
    for n in config.group_sizes:
        p_iid = iid_peak(n)
        iid_peak_half_mass = float(iid_half_mass(p_iid, n))
        for rho in config.correlations:
            p_peak, correlated_peak_half_mass = _grid_peak(
                n, rho, config.grid_points
            )
            at_iid_peak = float(beta_binomial_half_mass(p_iid, n, rho))
            absolute_overstatement = iid_peak_half_mass - at_iid_peak
            relative_overstatement = (
                absolute_overstatement / at_iid_peak if at_iid_peak > 0.0 else None
            )
            if at_iid_peak > iid_peak_half_mass + 1e-12:
                iid_dominance_violations.append(
                    {"n": n, "rho": rho, "difference": at_iid_peak - iid_peak_half_mass}
                )
            peak_rows.append(
                {
                    "n": n,
                    "rho": rho,
                    "iid_peak_p": p_iid,
                    "correlated_peak_p": p_peak,
                    "peak_p_shift": p_peak - p_iid,
                    "iid_peak_half_mass": iid_peak_half_mass,
                    "correlated_half_mass_at_iid_peak": at_iid_peak,
                    "iid_absolute_overstatement_at_iid_peak": absolute_overstatement,
                    "iid_relative_overstatement_at_iid_peak": relative_overstatement,
                    "correlated_peak_half_mass": correlated_peak_half_mass,
                }
            )

    identity = _identity_audit(config)
    monte_carlo = _monte_carlo_audit(config)
    return {
        "schema_version": 1,
        "experiment": "exchangeable_beta_binomial_maxrl_mass_stress",
        "status": (
            "Exact post-guidance CPU scope analysis; not an independent empirical "
            "confirmation and not a model-training experiment."
        ),
        "question": (
            "Within an exchangeable beta-binomial family, how does increasing "
            "positive rollout correlation change practical-MaxRL coefficient "
            "activity relative to the i.i.d. closed form?"
        ),
        "general_identity": "E[A]=2*(P(K>=1)-E[K]/N)",
        "iid_substitution": "P(K>=1)=1-(1-p)^N",
        "dependence_model": (
            "Beta-binomial exchangeability with common marginal p and pairwise "
            "intra-class correlation rho."
        ),
        "config": {
            "group_sizes": list(config.group_sizes),
            "correlations": list(config.correlations),
            "peak_grid_points": config.grid_points,
            "peak_grid_resolution": (1.0 - 2e-6) / (config.grid_points - 1),
            "monte_carlo_groups": config.monte_carlo_groups,
            "monte_carlo_seed": config.monte_carlo_seed,
        },
        "checks": {
            "enumerated_count_distribution_matches_general_identity": identity[
                "passed"
            ],
            "positive_correlation_never_exceeds_iid_mass_at_iid_peak": not iid_dominance_violations,
            "all_peak_values_nonnegative": all(
                row["correlated_peak_half_mass"] >= 0.0 for row in peak_rows
            ),
        },
        "identity_audit": identity,
        "monte_carlo_audit": monte_carlo,
        "peak_and_misspecification_rows": peak_rows,
        "iid_dominance_violations": iid_dominance_violations,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
            "source_sha256": _source_hashes(),
        },
        "claim_boundary": (
            "The coefficient identity is distribution-free, while the numerical "
            "correlation sweep covers one exchangeable positive-dependence family. "
            "It does not show gradient norm, signal-to-noise, or learning improvement."
        ),
    }


def _parse_csv(values: str, cast) -> tuple:
    parsed = tuple(cast(piece.strip()) for piece in values.split(",") if piece.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return parsed


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def print_summary(results: dict) -> None:
    print("Correlated-rollout MaxRL mass stress test")
    print("  identity check:", results["checks"]["enumerated_count_distribution_matches_general_identity"])
    print("  iid dominance check:", results["checks"]["positive_correlation_never_exceeds_iid_mass_at_iid_peak"])
    print("  N   rho   iid peak p   corr peak p   iid overstatement at iid peak")
    selected = {0.0, 0.1, 0.5, 0.8}
    for row in results["peak_and_misspecification_rows"]:
        if row["rho"] in selected:
            print(
                f" {row['n']:2d}  {row['rho']:.2f}    {row['iid_peak_p']:.5f}      "
                f"{row['correlated_peak_p']:.5f}             "
                f"{row['iid_absolute_overstatement_at_iid_peak']:.5f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-sizes",
        type=lambda value: _parse_csv(value, int),
        default=GROUP_SIZES,
    )
    parser.add_argument(
        "--correlations",
        type=lambda value: _parse_csv(value, float),
        default=CORRELATIONS,
    )
    parser.add_argument("--grid-points", type=int, default=GRID_POINTS)
    parser.add_argument("--monte-carlo-groups", type=int, default=MONTE_CARLO_GROUPS)
    parser.add_argument("--monte-carlo-seed", type=int, default=MONTE_CARLO_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = StressConfig(
        group_sizes=tuple(args.group_sizes),
        correlations=tuple(args.correlations),
        grid_points=args.grid_points,
        monte_carlo_groups=args.monte_carlo_groups,
        monte_carlo_seed=args.monte_carlo_seed,
    )
    results = build_results(config)
    _write_json(args.out, results)
    print_summary(results)


if __name__ == "__main__":
    main()
