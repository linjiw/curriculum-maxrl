"""Tests for the exchangeable-rollout practical-MaxRL mass audit."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from curriculum_maxrl.run_correlated_rollout_stress import (
    StressConfig,
    _source_hashes,
    beta_binomial_zero_probability,
    beta_binomial_half_mass,
    build_results,
    expected_mass_from_count_distribution,
    iid_half_mass,
    iid_peak,
)


class CorrelatedRolloutStressTests(unittest.TestCase):
    def test_iid_limit_matches_closed_form(self) -> None:
        p = np.linspace(0.0, 1.0, 101)
        for n in (2, 4, 8, 16, 32):
            zero = beta_binomial_zero_probability(p, n, 0.0)
            np.testing.assert_allclose(zero, (1.0 - p) ** n, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(
                beta_binomial_half_mass(p, n, 0.0), iid_half_mass(p, n), atol=1e-15
            )

    def test_count_enumeration_matches_general_identity(self) -> None:
        for n in (2, 7, 16):
            for p in (0.03, 0.17, 0.51, 0.88):
                for rho in (0.0, 0.05, 0.2, 0.7):
                    enumerated = expected_mass_from_count_distribution(n, p, rho)
                    identity = 2.0 * float(beta_binomial_half_mass(p, n, rho))
                    self.assertAlmostEqual(enumerated, identity, places=12)

    def test_positive_correlation_reduces_activity(self) -> None:
        p = np.linspace(0.001, 0.999, 999)
        for n in (2, 4, 16, 32):
            iid = iid_half_mass(p, n)
            for rho in (0.01, 0.1, 0.5, 0.8):
                correlated = beta_binomial_half_mass(p, n, rho)
                self.assertTrue(np.all(correlated <= iid + 1e-14))

    def test_peak_formula_and_compact_build(self) -> None:
        self.assertAlmostEqual(iid_peak(2), 0.5)
        self.assertAlmostEqual(iid_peak(16), 1.0 - 16 ** (-1.0 / 15.0))
        result = build_results(
            StressConfig(
                group_sizes=(2, 4),
                correlations=(0.0, 0.1),
                grid_points=10_001,
                monte_carlo_groups=10_000,
                monte_carlo_seed=123,
            )
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["peak_and_misspecification_rows"]), 4)

    def test_saved_artifact_matches_source_and_registered_config(self) -> None:
        path = Path(__file__).with_name("results_correlated_rollout_stress.json")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["config"]["group_sizes"], [2, 4, 8, 16, 32])
        self.assertEqual(
            artifact["config"]["correlations"],
            [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8],
        )
        self.assertEqual(artifact["config"]["peak_grid_points"], 200_001)
        self.assertEqual(artifact["config"]["monte_carlo_groups"], 200_000)
        self.assertEqual(artifact["config"]["monte_carlo_seed"], 20_260_808)
        self.assertTrue(all(artifact["checks"].values()))
        self.assertEqual(artifact["software"]["source_sha256"], _source_hashes())


if __name__ == "__main__":
    unittest.main()
