"""Adversarial unit and parsed-config tests for the opt-in v4 rank policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax

from minimax.arguments import parser
from minimax.config.xpid_maker import get_runner_info
from minimax.util.dotdict import DefaultDotDict
from minimax.util.rl.frontier_activity import expected_coefficient_activity
from minimax.util.rl.plr import PLRManager
from minimax.util.rl.tie_aware_rank import (
	effective_support,
	exact_score_tie_diagnostics,
	tie_aware_rank_mass,
)
from minimax.util.rl.training import VmapTrainState
from minimax.util.rl.ued_scores import UEDScore


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT/"ued_benchmark/configs"
V3_CONTRACT = "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000"
V3_APPLICATOR = "ddd3569b86adb703c8c7141fe7f2dae7a49c2c6b08e326edd61c3e3da7a345f7"
V4_CONTRACT = "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
UPSTREAM_PLR_CONFIG = "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b"
V1_PROTOCOL = "9d0ccbeaf83564958c5374e6e68793aa644013b1e9f6b889a91da69c99a720ba"
DISTINCT_NORMALIZED_ABS_TOLERANCE = 5e-7


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _manager(*, buffer_size=5, tie_aware=True, staleness_coef=0.0, temp=0.3):
	return PLRManager(
		example_level={"id": jnp.array([0], dtype=jnp.int32)},
		ued_score=UEDScore.RETURN,
		buffer_size=buffer_size,
		staleness_coef=staleness_coef,
		temp=temp,
		use_score_ranks=True,
		tie_aware_score_ranks=tie_aware,
	)


def _parse_config(name: str):
	document = json.loads((CONFIG_DIR/name).read_text(encoding="utf-8"))["args"]
	argv = ["tie-aware-config-test"] + [
		f"--{key}={values[0]}" for key, values in document.items()
	]
	with mock.patch.object(sys, "argv", argv):
		parsed = parser.parse_args()
	flat = DefaultDotDict({key: values[0] for key, values in document.items()})
	return document, parsed, get_runner_info(flat)


def _plain(value):
	if isinstance(value, dict):
		return {key: _plain(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_plain(item) for item in value]
	return value


class TieAwareRankMassTest(unittest.TestCase):
	def test_exact_temperature_transformed_block_mass_is_preserved(self):
		scores = jnp.asarray([9.0, 9.0, 9.0, 4.0, 2.0], dtype=jnp.float32)
		filled = jnp.ones((5,), dtype=jnp.bool_)
		mass = np.asarray(jax.jit(tie_aware_rank_mass)(scores, filled, 0.3))
		ranks = np.arange(1, 6, dtype=np.float32)
		stable_mass = (np.float32(1.0)/ranks)**np.float32(1.0/0.3)
		np.testing.assert_allclose(mass[:3], mass[0], rtol=0, atol=0)
		self.assertAlmostEqual(float(mass[:3].sum()), float(stable_mass[:3].sum()), places=6)
		self.assertAlmostEqual(float(mass[3]), float(stable_mass[3]), places=7)
		self.assertAlmostEqual(float(mass[4]), float(stable_mass[4]), places=7)

	def test_distribution_is_permutation_equivariant_with_mixed_ages(self):
		manager = _manager(buffer_size=5, tie_aware=True, staleness_coef=0.3)
		# All slots are filled and the mixed tie blocks/ages make a slot-order
		# reduction differ by one ulp unless normalization uses canonical order.
		scores = jnp.asarray([4.0, 4.0, 2.0, 1.0, 1.0])
		ages = jnp.asarray([0, 7, 2, 9, 3], dtype=jnp.uint32)
		filled = jnp.ones((5,), dtype=jnp.bool_)
		base_score = np.asarray(manager._get_score_dist(scores, filled))
		base_replay = np.asarray(manager._get_replay_dist(scores, ages, filled))
		base_buffer = manager.reset().replace(
			scores=scores,
			ages=ages,
			filled=filled,
			filled_count=jnp.asarray([5], dtype=jnp.int32),
		)
		base_metrics = manager.get_metrics(base_buffer)
		for permutation in (
			np.asarray([1, 0, 3, 4, 2]),
			np.asarray([4, 2, 0, 3, 1]),
			np.asarray([2, 3, 1, 0, 4]),
		):
			permuted_score = np.asarray(manager._get_score_dist(
				scores[permutation], filled[permutation]))
			permuted_replay = np.asarray(manager._get_replay_dist(
				scores[permutation], ages[permutation], filled[permutation]))
			np.testing.assert_array_equal(
				permuted_score, base_score[permutation])
			np.testing.assert_array_equal(
				permuted_replay, base_replay[permutation])
			permuted_buffer = manager.reset().replace(
				scores=scores[permutation],
				ages=ages[permutation],
				filled=filled[permutation],
				filled_count=jnp.asarray([5], dtype=jnp.int32),
			)
			permuted_metrics = manager.get_metrics(permuted_buffer)
			self.assertEqual(
				float(permuted_metrics["score_distribution_effective_support"]),
				float(base_metrics["score_distribution_effective_support"]),
			)
			self.assertEqual(
				float(permuted_metrics["replay_distribution_effective_support"]),
				float(base_metrics["replay_distribution_effective_support"]),
			)

	def test_distinct_scores_match_stable_in_canonical_source_order(self):
		scores = jnp.asarray([8.0, 5.0, 3.0, 1.0, -jnp.inf])
		filled = jnp.asarray([True, True, True, True, False])
		stable = np.asarray(_manager(tie_aware=False)._get_score_dist(scores, filled))
		tie_aware = np.asarray(_manager(tie_aware=True)._get_score_dist(scores, filled))
		np.testing.assert_array_equal(tie_aware, stable)

	def test_random_distinct_orders_preserve_raw_bits_and_normalized_bound(self):
		buffer_size = 500
		manager_stable = _manager(buffer_size=buffer_size, tie_aware=False)
		manager_tie = _manager(buffer_size=buffer_size, tie_aware=True)
		base_scores = jnp.arange(buffer_size, dtype=jnp.float32)[::-1]
		filled = jnp.ones((buffer_size,), dtype=jnp.bool_)
		base_raw = np.asarray(tie_aware_rank_mass(base_scores, filled, 0.3))
		base_tie = np.asarray(manager_tie._get_score_dist(base_scores, filled))
		rng = np.random.default_rng(20260814)
		for _ in range(64):
			permutation = rng.permutation(buffer_size)
			scores = base_scores[permutation]
			raw = np.asarray(tie_aware_rank_mass(scores, filled, 0.3))
			stable = np.asarray(manager_stable._get_score_dist(scores, filled))
			tie = np.asarray(manager_tie._get_score_dist(scores, filled))
			np.testing.assert_array_equal(raw, base_raw[permutation])
			np.testing.assert_array_equal(tie, base_tie[permutation])
			np.testing.assert_allclose(
				tie, stable, rtol=0, atol=DISTINCT_NORMALIZED_ABS_TOLERANCE)

	def test_deterministic_scan_edges_preserve_tail_mass_and_singletons(self):
		module_source = (ROOT/
			"ued_benchmark/overlay/minimax/util/rl/tie_aware_rank.py"
		).read_text(encoding="utf-8")
		self.assertNotIn(".at[", module_source)
		self.assertNotIn(".add(", module_source)
		jaxpr = str(jax.make_jaxpr(tie_aware_rank_mass)(
			jnp.arange(17, dtype=jnp.float32),
			jnp.ones((17,), dtype=jnp.bool_),
			0.3,
		))
		self.assertNotIn("scatter[", jaxpr)
		# Exercise the maximum authored buffer size with every filled score
		# distinct. Every singleton must retain the exact upstream float32 bits,
		# including masses near 1e-9 at the tail.
		buffer_size = 500
		scores = jnp.arange(buffer_size, dtype=jnp.float32)[::-1]
		filled = jnp.ones((buffer_size,), dtype=jnp.bool_)
		actual = np.asarray(jax.jit(tie_aware_rank_mass)(scores, filled, 0.3))
		ranks = 1.0 + jnp.arange(buffer_size, dtype=jnp.float32)
		expected = np.asarray((1.0/ranks)**(
			1.0/jnp.asarray(0.3, dtype=jnp.float32)))
		np.testing.assert_array_equal(actual, expected)
		stable_distribution = np.asarray(_manager(
			buffer_size=buffer_size, tie_aware=False)._get_score_dist(scores, filled))
		tie_distribution = np.asarray(_manager(
			buffer_size=buffer_size, tie_aware=True)._get_score_dist(scores, filled))
		np.testing.assert_array_equal(tie_distribution, stable_distribution)

		# A lowest-ranked tie must remain nonzero and preserve its local block
		# mass; a global float32 prefix subtraction would erase this fixture.
		tail_scores = scores.at[-3:].set(-7.0)
		tail_mass = np.asarray(jax.jit(tie_aware_rank_mass)(
			tail_scores, filled, 0.3))
		self.assertGreater(float(tail_mass[-1]), 0.0)
		np.testing.assert_array_equal(tail_mass[-3:], np.repeat(tail_mass[-1], 3))
		self.assertAlmostEqual(
			float(tail_mass[-3:].sum()), float(expected[-3:].sum()), places=14)

		np.testing.assert_array_equal(
			np.asarray(tie_aware_rank_mass(
				jnp.asarray([5.0]), jnp.asarray([True]), 0.3)),
			np.asarray([1.0], dtype=np.float32),
		)
		unfilled = jnp.zeros((7,), dtype=jnp.bool_)
		np.testing.assert_array_equal(
			np.asarray(tie_aware_rank_mass(
				jnp.arange(7, dtype=jnp.float32), unfilled, 0.3)),
			np.zeros((7,), dtype=np.float32),
		)
		diagnostics = exact_score_tie_diagnostics(
			jnp.arange(7, dtype=jnp.float32), unfilled)
		self.assertTrue(all(int(value) == 0 for value in diagnostics.values()))

	def test_all_equal_and_unfilled_slots(self):
		# The unfilled score is deliberately larger and must consume no rank.
		scores = jnp.asarray([1.0, 99.0, 1.0, -5.0, 1.0])
		filled = jnp.asarray([True, False, True, False, True])
		dist = np.asarray(_manager(tie_aware=True)._get_score_dist(scores, filled))
		np.testing.assert_allclose(dist, [1/3, 0, 1/3, 0, 1/3], rtol=0, atol=1e-7)
		diagnostics = exact_score_tie_diagnostics(scores, filled)
		self.assertEqual(int(diagnostics["distinct_filled_score_count"]), 1)
		self.assertEqual(int(diagnostics["score_tie_block_count"]), 1)
		self.assertEqual(int(diagnostics["score_max_tie_block_size"]), 3)

	def test_nonfinite_filled_scores_fail_closed(self):
		manager = _manager(tie_aware=True)
		for invalid in (jnp.nan, jnp.inf, -jnp.inf):
			dist = np.asarray(manager._get_replay_dist(
				jnp.asarray([2.0, invalid, 0.0, 0.0, 0.0]),
				jnp.arange(5, dtype=jnp.uint32),
				jnp.asarray([True, True, False, False, False])))
			self.assertTrue(np.isnan(dist).all())

	def test_stress_fixture_support_and_block_mass(self):
		# Outcome-blind deterministic fixture: 28 slots for each K=0..8.
		one_each = np.asarray([
			float(expected_coefficient_activity(k, 8, 8, 1.0, 1.0))
			for k in range(9)
		], dtype=np.float32)
		scores = jnp.asarray(np.repeat(one_each, 28))
		filled = jnp.ones((252,), dtype=jnp.bool_)
		stable = _manager(buffer_size=252, tie_aware=False)._get_score_dist(scores, filled)
		tied = _manager(buffer_size=252, tie_aware=True)._get_score_dist(scores, filled)
		top = np.repeat(np.arange(9) == 2, 28)
		self.assertAlmostEqual(float(stable[top].sum()), 0.9998504347, places=6)
		self.assertAlmostEqual(float(tied[top].sum()), 0.9998504347, places=6)
		self.assertAlmostEqual(float(stable.max()), 0.8715696963, places=6)
		self.assertAlmostEqual(float(tied.max()), 0.0357089441, places=6)
		self.assertAlmostEqual(float(effective_support(stable)), 1.30257482, places=5)
		self.assertAlmostEqual(float(effective_support(tied)), 28.0083771, places=4)


class SamplingAndResumeTest(unittest.TestCase):
	def test_sampling_is_with_replacement_and_reports_actual_duplicates(self):
		manager = _manager(buffer_size=1, tie_aware=True)
		buffer = manager.reset().replace(
			levels={"id": jnp.asarray([[7]], dtype=jnp.int32)},
			scores=jnp.asarray([1.0]),
			filled=jnp.asarray([True]),
			filled_count=jnp.asarray([1], dtype=jnp.int32),
		)
		_levels, idxs, buffer = manager._sample_replay_levels(
			jax.random.PRNGKey(11), buffer, 4)
		np.testing.assert_array_equal(np.asarray(idxs), [0, 0, 0, 0])
		self.assertEqual(int(buffer.last_replay_group_count[0]), 4)
		self.assertEqual(int(buffer.last_replay_distinct_group_count[0]), 1)
		self.assertEqual(int(buffer.last_replay_duplicate_group_count[0]), 3)
		self.assertEqual(int(buffer.replay_group_draw_count[0]), 4)
		self.assertEqual(int(buffer.replay_distinct_group_count[0]), 1)
		self.assertEqual(int(buffer.replay_duplicate_group_count[0]), 3)

		# These counters describe slot draws. force_unique acts later during
		# buffer update and therefore cannot resample this batch.
		_levels, _idxs, buffer = manager._sample_replay_levels(
			jax.random.PRNGKey(12), buffer, 2)
		self.assertEqual(int(buffer.last_replay_group_count[0]), 2)
		self.assertEqual(int(buffer.last_replay_distinct_group_count[0]), 1)
		self.assertEqual(int(buffer.last_replay_duplicate_group_count[0]), 1)
		self.assertEqual(int(buffer.replay_group_draw_count[0]), 6)
		self.assertEqual(int(buffer.replay_distinct_group_count[0]), 2)
		self.assertEqual(int(buffer.replay_duplicate_group_count[0]), 4)
		metrics = manager.get_metrics(buffer)
		self.assertEqual(float(metrics["tie_aware_score_ranks"]), 1.0)
		self.assertEqual(int(metrics["distinct_filled_score_count"]), 1)
		self.assertEqual(int(metrics["score_max_tie_block_size"]), 1)
		self.assertEqual(int(metrics["last_replay_group_count"]), 2)
		self.assertEqual(int(metrics["last_replay_distinct_group_count"]), 1)
		self.assertEqual(int(metrics["last_replay_duplicate_group_count"]), 1)
		self.assertEqual(int(metrics["replay_group_draw_count"]), 6)
		self.assertEqual(int(metrics["replay_distinct_group_count"]), 2)
		self.assertEqual(int(metrics["replay_duplicate_group_count"]), 4)
		self.assertAlmostEqual(
			float(metrics["score_distribution_effective_support"]), 1.0)
		self.assertAlmostEqual(
			float(metrics["replay_distribution_effective_support"]), 1.0)

	def test_nonfinite_candidates_do_not_partially_mutate_buffer(self):
		manager = _manager(buffer_size=2, tie_aware=True)
		levels = {"id": jnp.asarray([[10]], dtype=jnp.int32)}
		buffer = manager.update(
			manager.reset(), levels, jnp.asarray([-1]), jnp.asarray([1.0]))
		before_scores = np.asarray(buffer.scores).copy()
		before_successes = np.asarray(buffer.success_counts).copy()
		before_trials = np.asarray(buffer.trial_counts).copy()
		buffer = manager.update(
			buffer, levels, jnp.asarray([0]), jnp.asarray([jnp.nan]))
		np.testing.assert_array_equal(np.asarray(buffer.scores), before_scores)
		np.testing.assert_array_equal(np.asarray(buffer.success_counts), before_successes)
		np.testing.assert_array_equal(np.asarray(buffer.trial_counts), before_trials)
		self.assertEqual(int(buffer.filled_count[0]), 1)
		self.assertEqual(int(buffer.nonfinite_score_rejection_count[0]), 1)

		new_levels = {"id": jnp.asarray([[99]], dtype=jnp.int32)}
		buffer = manager.update(
			buffer, new_levels, jnp.asarray([-1]), jnp.asarray([jnp.inf]))
		self.assertEqual(int(buffer.filled_count[0]), 1)
		self.assertEqual(int(buffer.nonfinite_score_rejection_count[0]), 2)

	@staticmethod
	def _state(manager):
		tx = optax.sgd(learning_rate=0.01)
		return VmapTrainState.create(
			apply_fn=lambda *args: None,
			params={"w": jnp.zeros((1, 1), dtype=jnp.float32)},
			tx=tx,
			plr_buffer=manager.reset(),
		)

	def test_checkpoint_rejects_rank_mode_drift_both_directions(self):
		stable = self._state(_manager(tie_aware=False))
		tied = self._state(_manager(tie_aware=True))
		stable_checkpoint = pickle.loads(pickle.dumps(stable.state_dict))
		tied_checkpoint = pickle.loads(pickle.dumps(tied.state_dict))
		with self.assertRaisesRegex(ValueError, "tie_aware_score_ranks"):
			tied.load_state_dict(stable_checkpoint)
		with self.assertRaisesRegex(ValueError, "tie_aware_score_ranks"):
			stable.load_state_dict(tied_checkpoint)

		# A legacy-style object that genuinely lacks the new static field must
		# fail before any state can be restored.
		current = tied.plr_buffer
		legacy_missing_field = SimpleNamespace(
			buffer_size=current.buffer_size,
			ued_score=current.ued_score,
			replay_prob=current.replay_prob,
			staleness_coef=current.staleness_coef,
			temp=current.temp,
			use_score_ranks=current.use_score_ranks,
		)
		missing_field_checkpoint = dict(tied_checkpoint)
		missing_field_checkpoint["plr_buffer"] = legacy_missing_field
		with self.assertRaisesRegex(ValueError, "tie_aware_score_ranks"):
			tied.load_state_dict(missing_field_checkpoint)

		legacy = dict(tied_checkpoint)
		legacy["plr_buffer"] = legacy["plr_buffer"].replace(
			frontier_overlay_version="frontier-activity-v3",
			frontier_overlay_contract_sha256=V3_CONTRACT,
		)
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			tied.load_state_dict(legacy)

	def test_checkpoint_rejects_corrupted_nonfinite_filled_score(self):
		tied = self._state(_manager(tie_aware=True))
		checkpoint = dict(pickle.loads(pickle.dumps(tied.state_dict)))
		checkpoint["plr_buffer"] = checkpoint["plr_buffer"].replace(
			filled=jnp.asarray([True, False, False, False, False]),
			scores=jnp.asarray([jnp.nan, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf]),
		)
		with self.assertRaisesRegex(ValueError, "nonfinite filled score"):
			tied.load_state_dict(checkpoint)


class ConfigAndLineageTest(unittest.TestCase):
	def test_parsed_frontier_and_group_matched_common_runtime_is_exact(self):
		_frontier_doc, frontier, _frontier_xpid = _parse_config(
			"maze_frontier_exact_grouped_n8_tie_aware_v4.json")
		_maxmc_doc, maxmc, _maxmc_xpid = _parse_config(
			"maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json")
		frontier_plain = _plain(frontier)
		maxmc_plain = _plain(maxmc)
		allowed_runner_differences = {
			"ued_score",
			"frontier_n_rollouts",
			"frontier_require_n_eval_match",
			"frontier_prior_alpha",
			"frontier_prior_beta",
			"frontier_success_threshold",
			"frontier_posterior_mode",
		}
		frontier_runner = frontier_plain["train_runner_args"]
		maxmc_runner = maxmc_plain["train_runner_args"]
		for field in allowed_runner_differences:
			frontier_runner.pop(field)
			maxmc_runner.pop(field)
		self.assertEqual(frontier_runner, maxmc_runner)
		frontier_plain.pop("train_runner_args")
		maxmc_plain.pop("train_runner_args")
		self.assertEqual(frontier_plain, maxmc_plain)

		# Explicitly pin the high-risk common geometry and optimizer fields so
		# a future parser-default change produces a local, readable failure.
		common = frontier.train_runner_args
		self.assertEqual((common.n_parallel, common.n_eval, common.buffer_size), (4, 8, 500))
		self.assertEqual((common.temp, common.staleness_coef), (0.3, 0.3))
		self.assertEqual((common.replay_prob, common.min_fill_ratio), (0.5, 0.5))
		self.assertTrue(common.use_robust_plr)
		self.assertTrue(common.force_unique)
		self.assertTrue(common.tie_aware_score_ranks)
		self.assertEqual(frontier.student_rl_args, maxmc.student_rl_args)
		self.assertEqual(frontier.student_model_args, maxmc.student_model_args)
		self.assertEqual(frontier.env_args, maxmc.env_args)
		self.assertEqual(frontier.eval_args, maxmc.eval_args)
		self.assertEqual(frontier.eval_env_args, maxmc.eval_env_args)

	def test_exact_and_group_matched_configs_opt_in_official_reference_does_not(self):
		for name in (
			"maze_frontier_exact_grouped_n8_tie_aware_v4.json",
			"maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json",
		):
			document, parsed, runner_info = _parse_config(name)
			self.assertTrue(parsed.train_runner_args.tie_aware_score_ranks)
			self.assertEqual(
				parsed.train_runner_args.frontier_overlay_contract_sha256, V4_CONTRACT)
			self.assertIn("ovtie-aware-v4ch3d5f3827", runner_info)
			self.assertIn("rt_", runner_info)
			self.assertEqual(document["plr_tie_aware_score_ranks"], [True])

		document, parsed, runner_info = _parse_config(
			"maze_maxmc_v4_stable_rank_compat_32x1_b4000.json")
		self.assertFalse(parsed.train_runner_args.tie_aware_score_ranks)
		self.assertIn("ovtie-aware-v4ch3d5f3827", runner_info)
		self.assertIn("r_", runner_info)
		self.assertNotIn("rt_", runner_info)
		self.assertEqual(document["n_parallel"], [32])
		self.assertEqual(document["n_eval"], [1])
		self.assertEqual(document["plr_buffer_size"], [4000])

		upstream = json.loads((
			CONFIG_DIR/"maze_maxmc_upstream_official_reference_32x1_b4000.json"
		).read_text(encoding="utf-8"))["args"]
		self.assertEqual(
			_sha256(CONFIG_DIR/"maze_maxmc_upstream_official_reference_32x1_b4000.json"),
			UPSTREAM_PLR_CONFIG,
		)
		self.assertFalse(any(
			key.startswith("plr_frontier_") or key == "plr_tie_aware_score_ranks"
			for key in upstream))
		self.assertEqual(upstream["n_parallel"], [32])
		self.assertEqual(upstream["n_eval"], [1])
		self.assertEqual(upstream["plr_buffer_size"], [4000])

	def test_v3_reconstruction_artifacts_remain_byte_exact(self):
		self.assertEqual(_sha256(ROOT/"ued_benchmark/OVERLAY_CONTRACT.json"), V3_CONTRACT)
		self.assertEqual(
			_sha256(ROOT/"ued_benchmark/scripts/apply_minimax_overlay.py"),
			V3_APPLICATOR,
		)
		self.assertEqual(_sha256(ROOT/"ued_benchmark/OVERLAY_CONTRACT_V4.json"), V4_CONTRACT)
		v4_contract = json.loads((
			ROOT/"ued_benchmark/OVERLAY_CONTRACT_V4.json").read_text())
		self.assertIn("canonical ascending value order", v4_contract[
			"rank_normalization_policy"])
		self.assertIn("5e-7", v4_contract["rank_normalization_policy"])
		lineage = json.loads((ROOT/"ued_benchmark/OVERLAY_LINEAGE.json").read_text())
		for version in lineage["versions"].values():
			for stem in ("applicator", "contract"):
				path = ROOT/version[f"{stem}_path"]
				self.assertEqual(_sha256(path), version[f"{stem}_sha256"])
		v3 = lineage["versions"]["frontier-activity-v3"]
		self.assertEqual(v3["status"], "immutable_engineering_history")
		self.assertEqual(v3["frozen_development_protocol_sha256"], V1_PROTOCOL)
		self.assertEqual(
			_sha256(ROOT/v3["frozen_development_protocol_path"]), V1_PROTOCOL)
		for records in (v3["frozen_matched_configs"], v3["frozen_engineering_drivers"]):
			for record in records.values():
				self.assertEqual(_sha256(ROOT/record["path"]), record["sha256"])
		analyzer = v3["frozen_preregistered_analyzer"]
		self.assertEqual(_sha256(ROOT/analyzer["path"]), analyzer["sha256"])
		v4 = lineage["versions"]["frontier-activity-tie-aware-v4"]
		for records in (v4["matched_configs"], v4["modules"], v4["engineering_drivers"]):
			for record in records.values():
				self.assertEqual(_sha256(ROOT/record["path"]), record["sha256"])
		for name in ("source_faithful_reference", "stable_rank_compatibility"):
			record = v4[name]
			self.assertEqual(_sha256(ROOT/record["config_path"]), record["config_sha256"])
		protocol = v4["draft_development_protocol"]
		self.assertFalse(protocol["production_authorized"])
		self.assertEqual(_sha256(ROOT/protocol["path"]), protocol["sha256"])

	def test_invalid_rank_activation_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "requires use_score_ranks"):
			PLRManager(
				example_level={"id": jnp.array([0])},
				ued_score=UEDScore.RETURN,
				use_score_ranks=False,
				tie_aware_score_ranks=True,
			)
		with self.assertRaisesRegex(ValueError, "requires temp > 0"):
			_manager(tie_aware=True, temp=0.0)


if __name__ == "__main__":
	unittest.main()
