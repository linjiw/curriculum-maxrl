"""Unit and budget checks for run_fixed_budget_n_sweep.py."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from curriculum_maxrl.run_fixed_budget_n_sweep import (
    FixedBudgetUtilityTeacher,
    SweepConfig,
    exact_two_sided_sign_p,
    run_cell,
    utility_values,
    validate_design,
)


class FixedBudgetNSweepTests(unittest.TestCase):
    def test_saved_artifact_matches_manuscript_table(self) -> None:
        artifact = Path(__file__).with_name("results_fixed_budget_n_sweep.json")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(payload["config"]["total_completions"], 51_200)
        self.assertEqual(payload["config"]["seeds"], list(range(8)))
        self.assertTrue(payload["checks"]["all_cells_exact_completion_budget"])
        self.assertTrue(payload["checks"]["all_cells_share_completion_checkpoints"])
        self.assertTrue(payload["checks"]["u_2_equals_p_times_one_minus_p_pairwise"])
        self.assertEqual(set(payload["by_n"]), {"2", "4", "8", "16", "32"})
        observed_cells = 0
        for n_result in payload["by_n"].values():
            self.assertEqual(set(n_result["arms"]), {"uniform", "learnability", "u_n"})
            for arm in n_result["arms"].values():
                self.assertEqual(len(arm["seed_runs"]), 8)
                self.assertEqual(
                    {run["seed"] for run in arm["seed_runs"]}, set(range(8))
                )
                observed_cells += len(arm["seed_runs"])
        self.assertEqual(observed_cells, 5 * 3 * 8)

        project_root = Path(__file__).resolve().parents[1]
        for relative, expected_sha in payload["software"]["source_sha256"].items():
            digest = hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected_sha, relative)

        expected_u_n_minus_learnability = {
            2: 0.0,
            4: 0.0307,
            8: 0.0920,
            16: 0.1526,
            32: 0.1909,
        }
        expected_u_n_minus_uniform = {
            2: -0.0106,
            4: -0.0032,
            8: 0.0309,
            16: 0.0453,
            32: 0.0836,
        }
        for n_rollouts in (2, 4, 8, 16, 32):
            contrasts = payload["by_n"][str(n_rollouts)]["paired_contrasts"]
            u_n_minus_learnability = contrasts["u_n_minus_learnability"]["metrics"][
                "normalized_auc_mean_pass"
            ]
            u_n_minus_uniform = contrasts["u_n_minus_uniform"]["metrics"][
                "normalized_auc_mean_pass"
            ]
            self.assertAlmostEqual(
                u_n_minus_learnability["mean"],
                expected_u_n_minus_learnability[n_rollouts],
                places=4,
            )
            self.assertAlmostEqual(
                u_n_minus_uniform["mean"],
                expected_u_n_minus_uniform[n_rollouts],
                places=4,
            )
            if n_rollouts == 2:
                self.assertEqual(u_n_minus_learnability["zero_seeds"], 8)
            else:
                self.assertEqual(u_n_minus_learnability["positive_seeds"], 8)

    def test_u2_is_exactly_learnability_in_implementation(self) -> None:
        p = np.linspace(0.0, 1.0, 101)
        np.testing.assert_array_equal(
            utility_values("u_n", p, 2),
            utility_values("learnability", p, 2),
        )

    def test_completion_normalized_decay_has_common_per_completion_rate(self) -> None:
        per_completion = []
        for n_rollouts in (2, 4, 8, 16, 32):
            teacher = FixedBudgetUtilityTeacher(
                3,
                sampler="u_n",
                n_rollouts=n_rollouts,
                seed=0,
                uniform_floor=0.1,
                reference_group_decay=0.9,
                reference_n=16,
            )
            per_completion.append(teacher.group_decay ** (1.0 / n_rollouts))
        np.testing.assert_allclose(per_completion, per_completion[0], rtol=1e-14)

    def test_design_rejects_inexact_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "not divisible"):
            validate_design(
                SweepConfig(total_completions=513, checkpoint_every=256),
                (2, 4, 8, 16, 32),
            )

    def test_exact_sign_test(self) -> None:
        self.assertEqual(exact_two_sided_sign_p(8, 0), 0.0078125)
        self.assertEqual(exact_two_sided_sign_p(4, 4), 1.0)
        self.assertIsNone(exact_two_sided_sign_p(0, 0))

    def test_cell_is_deterministic_and_exactly_budgeted(self) -> None:
        config = SweepConfig(total_completions=512, checkpoint_every=256)
        first = run_cell("u_n", 8, 3, config)
        second = run_cell("u_n", 8, 3, config)
        self.assertEqual(first, second)
        self.assertEqual(first["completions"], 512)
        self.assertEqual(first["groups"], 64)
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

    def test_n2_paired_arms_produce_identical_trajectories(self) -> None:
        config = SweepConfig(total_completions=512, checkpoint_every=256)
        learnability = run_cell("learnability", 2, 7, config)
        u_n = run_cell("u_n", 2, 7, config)
        self.assertEqual(learnability["checkpoints"], u_n["checkpoints"])
        self.assertEqual(
            learnability["requested_task_counts"], u_n["requested_task_counts"]
        )
        self.assertEqual(
            learnability["coefficient_l1_mass"], u_n["coefficient_l1_mass"]
        )


if __name__ == "__main__":
    unittest.main()
