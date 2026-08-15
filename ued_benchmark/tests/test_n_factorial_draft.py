"""Outcome-blind contract tests for the DRAFT N={2,4,8} package."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "ued_benchmark/analysis"
sys.path.insert(0, str(ANALYSIS))

import validate_n_factorial_draft as validator  # noqa: E402


SOURCE = Path(os.environ.get("TIE_AWARE_MINIMAX_FRESH_SOURCE", "/nonexistent"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NFactorialDraftStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_static_package()
        cls.protocol = cls.result["protocol"]

    def test_hash_closure_binds_protocol_and_six_configs(self) -> None:
        manifest = validator.load_json(validator.MANIFEST_PATH, "test manifest")
        self.assertEqual(_sha256(validator.PROTOCOL_PATH), validator.PROTOCOL_SHA256)
        self.assertEqual(_sha256(validator.MANIFEST_PATH), validator.MANIFEST_SHA256)
        self.assertEqual(manifest["protocol"]["sha256"], validator.PROTOCOL_SHA256)
        self.assertEqual(
            {item["path"]: item["sha256"] for item in manifest["configs"]},
            validator.CONFIG_HASHES,
        )
        self.assertEqual(len(manifest["configs"]), 6)

    def test_prior_protocols_and_official_reference_are_byte_frozen(self) -> None:
        protected = (
            "ued_benchmark/analysis/development_protocol_v1.json",
            "ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json",
            "ued_benchmark/configs/maze_maxmc_upstream_official_reference_32x1_b4000.json",
            "ued_benchmark/configs/maze_frontier_posterior_bridge_n8_neval1.json",
        )
        for relative in protected:
            with self.subTest(path=relative):
                self.assertEqual(
                    _sha256(ROOT / relative), validator.PROTECTED_HASHES[relative]
                )

    def test_each_N_pair_differs_only_in_score_and_frontier_fields(self) -> None:
        for N, arms in self.result["configs"].items():
            frontier = arms["frontier"][0]
            maxmc = arms["maxmc"][0]
            differing = {
                key for key in set(frontier) | set(maxmc)
                if frontier.get(key, object()) != maxmc.get(key, object())
            }
            self.assertEqual(differing, validator.WITHIN_N_ALLOWED_DIFFERENCES)
            self.assertEqual(frontier["n_eval"], N)
            self.assertEqual(maxmc["n_eval"], N)
            self.assertEqual(frontier["n_parallel"] * N, 32)
            self.assertEqual(maxmc["n_parallel"] * N, 32)
            self.assertTrue(frontier["plr_tie_aware_score_ranks"])
            self.assertTrue(maxmc["plr_tie_aware_score_ranks"])
            self.assertEqual(maxmc["ued_score"], "max_mc")
            self.assertFalse(validator.FRONTIER_ONLY_FIELDS & set(maxmc))

    def test_nominal_63_cycle_identity_is_explicitly_conditional(self) -> None:
        warm = self.protocol["nominal_warm_fill"]
        self.assertEqual(
            warm["classification"],
            "exact_only_conditioned_on_distinct_accepted_new_groups",
        )
        self.assertIn(
            "zero duplicate-new groups",
            warm["receipt_gate"]["exact_63_cycle_label_requires"],
        )
        for layout in validator.LAYOUTS.values():
            threshold = math.ceil(layout["buffer_size"] * 0.5)
            cycles = math.ceil(threshold / layout["n_parallel"])
            self.assertEqual(cycles, 63)
            self.assertEqual(cycles * 32 * 256, 516096)

    def test_fixed_update_and_transition_views_are_both_frozen(self) -> None:
        reporting = self.protocol["reporting"]
        self.assertEqual(
            reporting["fixed_update_primary"]["target_student_ppo_updates"],
            30000,
        )
        self.assertEqual(
            reporting["fixed_update_primary"]["target_upstream_n_grad_updates"],
            30000,
        )
        cycles = reporting["fixed_transition_secondary"]["outer_cycle_grid"]
        transitions = reporting["fixed_transition_secondary"][
            "exact_training_transition_grid"
        ]
        self.assertEqual(transitions, [cycle * 8192 for cycle in cycles])
        gate = reporting["fixed_update_primary"]["within_N_seed_pair_validity_gate"]
        self.assertIn("outer-cycle count", gate["must_equal_exactly"])
        self.assertIn(
            "terminal upstream n_grad_updates = 30000",
            gate["must_equal_exactly"],
        )
        self.assertIn("training-transition count", gate["must_equal_exactly"])
        self.assertIn(
            "the complete set of aligned fixed-transition cycle observations",
            gate["must_equal_exactly"],
        )
        self.assertIn("invalidate", gate["failure_policy"])

    def test_direct_parser_identity_and_terminal_checkpoint_fail_closed(self) -> None:
        identity = self.protocol["matched_training_contract"][
            "execution_identity_and_path_contract"
        ]
        terminal = self.protocol["terminal_checkpoint_contract"]
        self.assertEqual(identity["direct_parser_default_xpid"], "latest")
        self.assertFalse(identity["direct_parser_launch_is_safe"])
        self.assertIn("xpid equal", " ".join(identity["future_hashed_driver_must_set_and_verify"]))
        self.assertFalse(terminal["resume_allowed"])
        self.assertFalse(terminal["periodic_checkpoint_is_admissible_endpoint"])
        self.assertTrue(terminal["round_trip_required_before_evaluation"])
        bindings = " ".join(terminal["round_trip_must_bind_exactly"])
        for token in (
            "manifest", "protocol", "config", "arm", "N", "n_eval",
            "n_grad_updates",
        ):
            self.assertIn(token, bindings)
        self.assertIn(
            "cannot be used for recovery or resume",
            terminal["periodic_checkpoint_role"],
        )
        prerequisites = " ".join(self.protocol["prerequisites_for_any_future_run"])
        for component in (
            "trainer", "evaluator", "assembler or finalizer", "analyzer",
            "scheduler script", "source bundle", "complete environment provenance",
        ):
            self.assertIn(component, prerequisites)

    def test_cross_N_estimands_are_joint_layout_not_estimator_N_effects(self) -> None:
        self.assertIn(
            "cannot identify an estimator-N-only effect",
            self.protocol["estimands"]["identifiability"],
        )
        reporting = self.protocol["reporting"]
        self.assertFalse(
            reporting["fixed_update_primary"][
                "across_N_terminal_cycle_or_transition_equality_required"
            ]
        )
        self.assertIn(
            "all six cells",
            reporting["fixed_transition_secondary"]["across_N_alignment"],
        )

    def test_five_seed_design_is_descriptive_and_confirmatory_held(self) -> None:
        design = self.protocol["development_design"]
        hold = self.protocol["multiplicity_and_confirmatory_hold"]
        self.assertEqual(design["paired_training_seeds"], [101, 102, 103, 104, 105])
        self.assertTrue(design["descriptive_only"])
        self.assertEqual(design["minimum_two_sided_exact_sign_flip_p_value"], 0.0625)
        self.assertTrue(hold["current_package_authorizes_none_of_these"])
        self.assertIn("Holm", hold["primary_family"])

    def test_package_cannot_authorize_endpoint_or_production_access(self) -> None:
        self.assertFalse(self.protocol["production_driver_authorized"])
        self.assertFalse(self.protocol["endpoint_access_authorized"])
        self.assertFalse(self.protocol["paper_evidence"])


@unittest.skipUnless(
    SOURCE.is_dir(),
    "set TIE_AWARE_MINIMAX_FRESH_SOURCE to a fresh exact-v4 applied clone",
)
class NFactorialDraftPinnedCpuRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = validator.validate(SOURCE)

    def test_all_six_configs_parse_and_grid_generate_unique_xpids_on_fresh_v4(self) -> None:
        runtime = self.receipt["runtime"]
        self.assertEqual(runtime["jax_backend"], "cpu")
        self.assertEqual(runtime["parsed_config_count"], 6)
        self.assertEqual(runtime["unique_xpid_count"], 6)
        self.assertEqual(runtime["packages"], validator.EXPECTED_VERSIONS)
        self.assertEqual(runtime["direct_parser_default_xpid"], "latest")
        self.assertFalse(runtime["direct_parser_launch_safe"])

    def test_checkpoint_static_N_mismatch_fails_closed(self) -> None:
        runtime = self.receipt["runtime"]
        self.assertTrue(runtime["checkpoint_static_N_mismatch_rejected"])
        self.assertTrue(runtime["strict_frontier_N_n_eval_mismatch_rejected"])

    def test_runtime_receipt_is_engineering_only(self) -> None:
        self.assertEqual(self.receipt["status"], "passed")
        self.assertFalse(self.receipt["paper_evidence"])
        self.assertFalse(self.receipt["endpoint_accessed"])
        self.assertFalse(self.receipt["production_authorized"])


if __name__ == "__main__":
    unittest.main()
