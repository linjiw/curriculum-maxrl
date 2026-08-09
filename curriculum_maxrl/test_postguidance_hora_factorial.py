"""Accounting and mathematical tests for the post-guidance HORA factorial."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from curriculum_maxrl.run_postguidance_hora_factorial import (
    ALLOCATORS,
    FROZEN_HYPOTHESES,
    FactorialConfig,
    allocate_group_sizes,
    allocation_marginal,
    beta_bernoulli_marginal,
    run_cell,
    u_n,
    validate_design,
)


class PostGuidanceHoraFactorialTests(unittest.TestCase):
    def test_uniform_beta_marginal_has_closed_form(self) -> None:
        for exponent in range(20):
            expected = 1.0 / ((exponent + 1) * (exponent + 2))
            self.assertAlmostEqual(
                beta_bernoulli_marginal(1.0, 1.0, exponent), expected, places=15
            )

    def test_published_hora_recurrence(self) -> None:
        alpha, beta = 3.0, 4.0
        marginal = alpha / (alpha + beta)
        for ell in range(15):
            observed = allocation_marginal("hora_hit", alpha, beta, ell, probe_g0=4)
            self.assertAlmostEqual(observed, marginal, places=15)
            marginal *= (beta + ell) / (alpha + beta + ell + 1.0)

    def test_mass_aware_marginal_counts_spent_probes(self) -> None:
        alpha, beta, g0 = 2.0, 5.0, 4
        for ell in range(12):
            self.assertAlmostEqual(
                allocation_marginal("mass_aware", alpha, beta, ell, g0),
                beta_bernoulli_marginal(alpha, beta, g0 + ell),
                places=15,
            )
        self.assertLess(
            allocation_marginal("mass_aware", alpha, beta, 0, g0),
            allocation_marginal("hora_hit", alpha, beta, 0, g0),
        )

    def test_mass_marginal_is_exact_u_n_increment(self) -> None:
        p = np.linspace(0.0, 1.0, 101)
        for n in (1, 4, 16, 63):
            np.testing.assert_allclose(
                u_n(p, n + 1) - u_n(p, n),
                p * (1.0 - p) ** n,
                atol=2e-15,
                rtol=2e-13,
            )

    def test_all_allocators_preserve_exact_budget_and_probes(self) -> None:
        config = FactorialConfig(total_completions=512, checkpoint_every=256)
        counts = np.asarray([0, 1, 2, 3, 4, 0, 2, 4])
        allocations = {
            name: allocate_group_sizes(counts, name, config) for name in ALLOCATORS
        }
        for sizes in allocations.values():
            self.assertEqual(int(sizes.sum()), config.tasks_per_step * config.average_n)
            self.assertTrue(np.all(sizes >= config.probe_g0))
        np.testing.assert_array_equal(
            allocations["fixed"], np.full(config.tasks_per_step, config.average_n)
        )
        self.assertFalse(np.array_equal(allocations["hora_hit"], allocations["fixed"]))
        self.assertFalse(
            np.array_equal(allocations["mass_aware"], allocations["hora_hit"])
        )

    def test_design_rejects_misaligned_completion_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer number of steps"):
            validate_design(
                FactorialConfig(total_completions=513, checkpoint_every=256)
            )

    def test_tiny_cells_are_deterministic_and_exactly_accounted(self) -> None:
        config = FactorialConfig(total_completions=512, checkpoint_every=256)
        for allocator in ALLOCATORS:
            first = run_cell("u_n", allocator, 7, config)
            second = run_cell("u_n", allocator, 7, config)
            self.assertEqual(first, second)
            self.assertEqual(first["completions"], 512)
            self.assertEqual(first["teacher_observed_completions"], 512)
            self.assertEqual(first["mean_group_size"], 16)
            self.assertEqual(
                first["dead_groups"]
                + first["mixed_groups"]
                + first["all_pass_groups"],
                first["groups"],
            )
            self.assertEqual(
                [row["completions"] for row in first["checkpoints"]],
                [0, 256, 512],
            )

    def test_hypotheses_are_embedded_and_explicitly_post_guidance(self) -> None:
        self.assertEqual(FROZEN_HYPOTHESES["frozen_on"], "2026-08-07")
        self.assertIn("post-guidance", FROZEN_HYPOTHESES["status"])
        self.assertIn("No directional", FROZEN_HYPOTHESES["mean_pass_direction"])

    def test_saved_full_result_is_complete_and_source_matched(self) -> None:
        directory = Path(__file__).resolve().parent
        result = json.loads(
            (directory / "results_postguidance_hora_factorial.json").read_text()
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["cells"]), 6)
        self.assertEqual(result["config"]["seeds"], list(range(16)))
        source = directory / "run_postguidance_hora_factorial.py"
        recorded = result["software"]["source_sha256"][
            "curriculum_maxrl/run_postguidance_hora_factorial.py"
        ]
        self.assertEqual(recorded, hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
