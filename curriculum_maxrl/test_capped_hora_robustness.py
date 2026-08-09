"""Math, accounting, lock, and overlap tests for capped-HORA robustness."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from curriculum_maxrl import analyze_capped_hora_robustness as analyzer
from curriculum_maxrl import run_capped_hora_robustness as runner


class CappedHoraRobustnessTests(unittest.TestCase):
    def test_frozen_matrix_has_exactly_fifty_unique_cells(self) -> None:
        cells = runner.full_cell_specs()
        self.assertEqual(len(cells), 50)
        self.assertEqual(len({cell.cell_id for cell in cells}), 50)
        self.assertEqual(sum(cell.is_fixed for cell in cells), 2)
        self.assertEqual(sum(not cell.is_fixed for cell in cells), 48)
        self.assertNotIn("mass_aware", {cell.allocator for cell in cells})
        self.assertIn(
            "fresh_group_mass_proxy", {cell.allocator for cell in cells}
        )

    def test_exact_beta_moment_uniform_closed_form(self) -> None:
        for exponent in range(100):
            expected = 1.0 / ((exponent + 1) * (exponent + 2))
            self.assertAlmostEqual(
                runner.beta_bernoulli_moment(1.0, 1.0, exponent),
                expected,
                places=15,
            )

    def test_exact_beta_moment_not_plugin_posterior_mean(self) -> None:
        alpha, beta, exponent = 3.0, 4.0, 7
        exact = runner.beta_bernoulli_moment(alpha, beta, exponent)
        mean = alpha / (alpha + beta)
        plugin = mean * (1.0 - mean) ** exponent
        self.assertNotAlmostEqual(exact, plugin, places=8)

    def test_registered_score_exponents_and_claim_boundary(self) -> None:
        for ell in range(97):
            self.assertEqual(runner.score_exponent("hora_hit", ell, 4), ell)
            self.assertEqual(
                runner.score_exponent("fresh_group_mass_proxy", ell, 4),
                4 + ell,
            )
        protocol = runner.PROTOCOL_PATH.read_text(encoding="utf-8")
        self.assertIn("**not** the\nexact conditional marginal", protocol)
        self.assertIn("fresh_group_mass_proxy", protocol)

    def test_information_sources_use_exact_registered_point_predictions(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        counts = np.asarray([0, 1, 2, 3, 4, 0, 2, 4])
        history_alpha = np.asarray([7.0] * 8)
        history_beta = np.asarray([3.0] * 8)
        exact = np.linspace(0.0, 1.0, 8)

        alpha, beta, prediction = runner.information_state(
            "same_step", counts, history_alpha, history_beta, exact, config
        )
        np.testing.assert_array_equal(alpha, 1.0 + counts)
        np.testing.assert_array_equal(beta, 1.0 + 4 - counts)
        np.testing.assert_allclose(prediction, (1.0 + counts) / 6.0)

        alpha, beta, prediction = runner.information_state(
            "history_plus_probe",
            counts,
            history_alpha,
            history_beta,
            exact,
            config,
        )
        np.testing.assert_array_equal(alpha, 7.0 + counts)
        np.testing.assert_array_equal(beta, 3.0 + 4 - counts)
        np.testing.assert_allclose(prediction, (7.0 + counts) / 14.0)

        alpha, beta, prediction = runner.information_state(
            "oracle_preupdate", counts, history_alpha, history_beta, exact, config
        )
        self.assertIsNone(alpha)
        self.assertIsNone(beta)
        np.testing.assert_array_equal(prediction, exact)

    def test_duplicate_positions_share_history_but_add_own_probes(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        counts = np.asarray([0, 4, 1, 1, 1, 1, 1, 1])
        # The first two positions represent duplicate samples of one task and
        # therefore receive the exact same pre-batch snapshot.
        history_alpha = np.asarray([9.5, 9.5, 2, 2, 2, 2, 2, 2], dtype=float)
        history_beta = np.asarray([4.5, 4.5, 8, 8, 8, 8, 8, 8], dtype=float)
        exact = np.full(8, 0.25)
        alpha, beta, prediction = runner.information_state(
            "history_plus_probe",
            counts,
            history_alpha,
            history_beta,
            exact,
            config,
        )
        self.assertEqual(alpha[0], 9.5)
        self.assertEqual(beta[0], 8.5)
        self.assertEqual(alpha[1], 13.5)
        self.assertEqual(beta[1], 4.5)
        self.assertNotEqual(prediction[0], prediction[1])

    def test_zero_oracle_scores_expose_first_index_tie_and_cap_semantics(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        counts = np.zeros(8, dtype=int)
        history_alpha = np.ones(8)
        history_beta = np.ones(8)
        exact = np.zeros(8)
        sizes, diagnostics = runner.allocate_group_sizes(
            counts,
            allocator="hora_hit",
            information_source="oracle_preupdate",
            cap=24,
            history_alpha=history_alpha,
            history_beta=history_beta,
            exact_probabilities=exact,
            config=config,
        )
        np.testing.assert_array_equal(sizes, [24, 24, 24, 24, 20, 4, 4, 4])
        self.assertEqual(int(sizes.sum()), 128)
        self.assertEqual(diagnostics["allocation_decision_count"], 96)
        self.assertEqual(diagnostics["chosen_oracle_regret_sum"], 0.0)

    def test_every_registered_cap_preserves_budget_and_bounds(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        counts = np.asarray([0, 1, 2, 3, 4, 0, 2, 4])
        history_alpha = np.linspace(1.0, 8.0, 8)
        history_beta = np.linspace(8.0, 1.0, 8)
        exact = np.linspace(0.01, 0.99, 8)
        for allocator in runner.ALLOCATORS:
            for information in runner.INFORMATION_SOURCES:
                for cap in runner.CAPS:
                    sizes, diagnostics = runner.allocate_group_sizes(
                        counts,
                        allocator=allocator,
                        information_source=information,
                        cap=cap,
                        history_alpha=history_alpha,
                        history_beta=history_beta,
                        exact_probabilities=exact,
                        config=config,
                    )
                    maximum = 100 if cap is None else cap
                    self.assertEqual(int(sizes.sum()), 128)
                    self.assertTrue(np.all(sizes >= 4))
                    self.assertTrue(np.all(sizes <= maximum))
                    self.assertEqual(diagnostics["probability_error_count"], 8)
                    self.assertEqual(diagnostics["allocation_decision_count"], 96)

    def test_nearest_rank_p95_and_gini_follow_frozen_definitions(self) -> None:
        histogram = runner.Counter({4: 2, 16: 1, 24: 1})
        expanded = np.asarray([4.0, 4.0, 16.0, 24.0])
        expected_pair_sum = sum(abs(x - y) for x in expanded for y in expanded)
        expected_gini = expected_pair_sum / (
            2.0 * len(expanded) ** 2 * float(expanded.mean())
        )
        self.assertEqual(runner.nearest_rank_p95(histogram), 24)
        self.assertEqual(runner.group_size_gini(histogram), expected_gini)
        self.assertEqual(analyzer._nearest_rank_p95({"4": 2, "16": 1, "24": 1}), 24)
        self.assertEqual(
            analyzer._gini({"4": 2, "16": 1, "24": 1}), expected_gini
        )

    def test_tiny_cells_are_deterministic_and_exactly_accounted(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        specs = (
            runner.fixed_cell("uniform"),
            runner.adaptive_cell(
                "u_16", "fresh_group_mass_proxy", 24, "history_plus_probe"
            ),
            runner.adaptive_cell(
                "uniform", "hora_hit", 32, "oracle_preupdate"
            ),
        )
        for spec in specs:
            first = runner.run_cell(spec, 90, config)
            second = runner.run_cell(spec, 90, config)
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

    def test_oracle_diagnostics_are_exactly_zero(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        spec = runner.adaptive_cell(
            "uniform", "fresh_group_mass_proxy", 48, "oracle_preupdate"
        )
        result = runner.run_cell(spec, 91, config)
        self.assertEqual(result["mean_absolute_probability_error"], 0.0)
        self.assertEqual(result["mean_squared_probability_error"], 0.0)
        self.assertEqual(result["marginal_score_mae"], 0.0)
        self.assertEqual(result["chosen_oracle_regret_mean"], 0.0)

    def test_fixed_diagnostics_are_present_but_not_applicable(self) -> None:
        config = runner.RobustnessConfig(total_completions=512, checkpoint_every=256)
        result = runner.run_cell(runner.fixed_cell("u_16"), 91, config)
        self.assertFalse(result["allocation_diagnostics_applicable"])
        self.assertEqual(result["probability_error_position_count"], 0)
        self.assertEqual(result["allocation_decision_count"], 0)
        self.assertIsNone(result["mean_absolute_probability_error"])
        self.assertIsNone(result["marginal_score_mae"])

    def test_student_t_paired_summary_has_no_p_value(self) -> None:
        lhs = [{"seed": seed, "x": value} for seed, value in enumerate((2.0, 4.0))]
        rhs = [{"seed": seed, "x": value} for seed, value in enumerate((1.0, 1.0))]
        summary = analyzer.paired_summary(lhs, rhs, "x")
        self.assertEqual(summary["positive_seeds"], 2)
        self.assertEqual(summary["zero_seeds"], 0)
        self.assertEqual(summary["negative_seeds"], 0)
        self.assertIn("student_t_95_ci", summary)
        self.assertFalse(any("p_value" in key for key in summary))

    def test_seed_zero_overlap_fields_are_exact(self) -> None:
        old = json.loads(runner.OLD_RESULT_PATH.read_text(encoding="utf-8"))
        old_lookup = {}
        for cell in old["cells"].values():
            for old_run in cell["seed_runs"]:
                old_lookup[
                    (old_run["sampler"], old_run["allocator"], old_run["seed"])
                ] = old_run
        for spec in runner.overlap_cell_specs():
            observed = runner.run_cell(spec, 0, runner.RobustnessConfig())
            old_sampler = "u_n" if spec.sampler == "u_16" else "uniform"
            old_allocator = (
                "mass_aware"
                if spec.allocator == "fresh_group_mass_proxy"
                else spec.allocator
            )
            expected = old_lookup[(old_sampler, old_allocator, 0)]
            for field in runner.OVERLAP_FIELDS:
                self.assertEqual(
                    observed[field],
                    expected[field],
                    msg=f"{spec.cell_id} differs on {field}",
                )

    def test_runner_and_analyzer_independently_accept_canonical_lock(self) -> None:
        runner_lock, runner_hash = runner.load_and_verify_lock()
        analyzer_lock, analyzer_hash = analyzer.independently_verify_lock()
        self.assertEqual(runner_lock, analyzer_lock)
        self.assertEqual(runner_hash, analyzer_hash)
        self.assertEqual(runner_lock["runtime"], runner.PINNED_RUNTIME)
        self.assertEqual(runner_lock["matrix"]["cell_count"], 50)
        self.assertEqual(
            set(runner_lock["source_sha256"]), set(runner.SOURCE_RELATIVE_PATHS)
        )


if __name__ == "__main__":
    unittest.main()
