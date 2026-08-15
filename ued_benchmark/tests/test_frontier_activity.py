"""Focused tests for the pinned minimax FrontierRL overlay.

Run with the patched clone's ``src`` directory first on ``PYTHONPATH``; see the
top-level README in this directory for the exact command.
"""

from collections import namedtuple
import pickle
import unittest

import jax.numpy as jnp
import numpy as np
import optax

from minimax.util.rl.frontier_activity import (
	beta_posterior_mean,
	coefficient_activity_score,
	expected_coefficient_activity,
	sparse_goal_stream_counts,
	validate_success_trial_counts,
)
from minimax.util.rl.plr import PLRManager
from minimax.util.rl.training import VmapTrainState
from minimax.util.rl.ued_scores import UEDScore
from minimax.runners.plr_runner import frontier_group_is_valid


class FrontierFormulaTest(unittest.TestCase):
	def test_exact_n2_formula_and_boundaries(self):
		# With a deliberately tiny prior, posterior means approach the endpoints.
		p = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
		score = 1.0 - (1.0 - p)**2 - p
		np.testing.assert_allclose(score, p*(1.0 - p), rtol=0, atol=1e-7)
		self.assertEqual(float(score[0]), 0.0)
		self.assertEqual(float(score[-1]), 0.0)

	def test_beta_posterior_and_exact_activity(self):
		p = beta_posterior_mean(2, 3, prior_alpha=1.0, prior_beta=1.0)
		self.assertAlmostEqual(float(p), 3.0/5.0, places=6)
		expected = 1.0 - (1.0 - 3.0/5.0)**4 - 3.0/5.0
		actual = coefficient_activity_score(
			2, 3, 4, 1.0, 1.0, posterior_mode="mean_plugin")
		self.assertAlmostEqual(float(actual), expected, places=6)

	def test_exact_beta_expected_activity(self):
		# Uniform Beta(1,1), N=2: E[p(1-p)] = 1/6, while plugging in
		# E[p]=1/2 gives 1/4.
		expected = expected_coefficient_activity(0, 0, 2, 1.0, 1.0)
		plugin = coefficient_activity_score(
			0, 0, 2, 1.0, 1.0, posterior_mode="mean_plugin")
		self.assertAlmostEqual(float(expected), 1.0/6.0, places=6)
		self.assertAlmostEqual(float(plugin), 1.0/4.0, places=6)
		self.assertLessEqual(float(expected), float(plugin))

	def test_expected_activity_matches_rising_factorial(self):
		# successes=2,trials=3 with Beta(1,1) gives posterior Beta(3,2).
		actual = expected_coefficient_activity(2, 3, 4, 1.0, 1.0)
		failure_moment = (2*3*4*5)/(5*6*7*8)
		expected = 1.0 - failure_moment - 3.0/5.0
		self.assertAlmostEqual(float(actual), expected, places=6)
		plugin = coefficient_activity_score(
			2, 3, 4, 1.0, 1.0, posterior_mode="mean_plugin")
		self.assertLessEqual(float(actual), float(plugin) + 1e-7)

	def test_activity_targets_an_interior_probability(self):
		n = 8
		p_star = 1.0 - n**(-1.0/(n - 1))
		grid = np.linspace(0.0, 1.0, 10001)
		grid_argmax = grid[np.argmax(1.0 - (1.0 - grid)**n - grid)]
		self.assertLess(abs(grid_argmax - p_star), 2e-4)

	def test_invalid_bernoulli_counts_are_rejected(self):
		with self.assertRaisesRegex(ValueError, "cannot exceed"):
			validate_success_trial_counts([2], [1])
		with self.assertRaisesRegex(ValueError, "nonnegative"):
			validate_success_trial_counts([-1], [1])


class SparseGoalCountTest(unittest.TestCase):
	def test_strict_provisional_score_requires_all_eval_streams(self):
		trials = jnp.asarray([0, 1, 7, 8, 9], dtype=jnp.uint32)
		np.testing.assert_array_equal(
			np.asarray(frontier_group_is_valid(trials, 8, True)),
			[False, False, False, True, False])
		np.testing.assert_array_equal(
			np.asarray(frontier_group_is_valid(trials, 8, False)),
			[False, True, True, True, True])

	def test_counts_only_completed_positive_terminal_episodes(self):
		Batch = namedtuple("Batch", "rewards dones")
		# Shape is students x time x (levels * n_eval). The first two flat
		# environments are two evaluations of level 0; the next two are level 1.
		rewards = jnp.array([[
			[1.0, 0.0, 2.0, 0.0],  # positive nonterminal at index 2 is ignored
			[0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0],
		]], dtype=jnp.float32)
		dones = jnp.array([[
			[1, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 0],  # partial evaluation at rollout end is ignored
		]], dtype=jnp.uint8)
		successes, trials = sparse_goal_stream_counts(
			Batch(rewards=rewards, dones=dones), n_eval=2)
		np.testing.assert_array_equal(np.asarray(successes), [[1, 0]])
		np.testing.assert_array_equal(np.asarray(trials), [[2, 1]])

	def test_multiple_terminals_in_one_stream_are_one_observation(self):
		Batch = namedtuple("Batch", "rewards dones")
		rewards = jnp.array([[[1.0], [0.0], [1.0], [0.0]]])
		dones = jnp.array([[[1], [0], [1], [0]]], dtype=jnp.uint8)
		successes, trials = sparse_goal_stream_counts(
			Batch(rewards=rewards, dones=dones), n_eval=1)
		np.testing.assert_array_equal(np.asarray(successes), [[1]])
		np.testing.assert_array_equal(np.asarray(trials), [[1]])


class PLRBufferIntegrationTest(unittest.TestCase):
	@staticmethod
	def manager(buffer_size=2, n=4):
		return PLRManager(
			example_level={"id": jnp.array([0], dtype=jnp.int32)},
			ued_score=UEDScore.RETURN,
			buffer_size=buffer_size,
			staleness_coef=0.25,
			use_frontier_activity=True,
			frontier_n_rollouts=n,
			frontier_n_eval=n,
			frontier_require_n_eval_match=True,
			frontier_prior_alpha=1.0,
			frontier_prior_beta=1.0,
			frontier_posterior_mode="expected_activity",
		)

	@staticmethod
	def update(manager, buffer, ids, idxs, successes, trials, dupe_mask=None):
		n = len(ids)
		return manager.update(
			buffer,
			levels={"id": jnp.asarray(ids, dtype=jnp.int32).reshape(n, 1)},
			level_idxs=jnp.asarray(idxs, dtype=jnp.int32),
			# Deliberately wrong input scores prove the manager computes exact
			# posterior activity instead of trusting a provisional runner score.
			ued_scores=jnp.full((n,), -123.0, dtype=jnp.float32),
			info={
				"frontier_successes": jnp.asarray(successes, dtype=jnp.uint32),
				"frontier_trials": jnp.asarray(trials, dtype=jnp.uint32),
			},
			dupe_mask=(
				None if dupe_mask is None
				else jnp.asarray(dupe_mask, dtype=jnp.bool_)),
		)

	def test_insert_and_existing_update_accumulate_counts(self):
		manager = self.manager()
		buffer = self.update(
			manager, manager.reset(), [10, 20], [-1, -1], [0, 4], [4, 4])
		np.testing.assert_array_equal(np.asarray(buffer.success_counts), [0, 4])
		np.testing.assert_array_equal(np.asarray(buffer.trial_counts), [4, 4])
		self.assertAlmostEqual(
			float(buffer.scores[0]),
			float(expected_coefficient_activity(0, 4, 4)), places=6)

		buffer = self.update(manager, buffer, [10], [0], [2], [4])
		np.testing.assert_array_equal(np.asarray(buffer.success_counts), [2, 4])
		np.testing.assert_array_equal(np.asarray(buffer.trial_counts), [8, 4])
		self.assertAlmostEqual(
			float(buffer.scores[0]),
			float(expected_coefficient_activity(2, 8, 4)), places=6)

	def test_strict_incomplete_group_is_rejected_and_logged(self):
		manager = self.manager(n=4)
		buffer = self.update(manager, manager.reset(), [10], [-1], [0], [1])
		self.assertFalse(bool(buffer.filled[0]))
		self.assertEqual(int(buffer.trial_counts[0]), 0)
		self.assertEqual(int(buffer.incomplete_group_count[0]), 1)

	def test_existing_replay_duplicate_accumulates_all_evidence(self):
		manager = self.manager(buffer_size=1, n=4)
		buffer = self.update(manager, manager.reset(), [10], [-1], [0], [4])
		buffer = self.update(
			manager, buffer, [10, 10], [0, 0], [4, 0], [4, 4],
			dupe_mask=[False, True])
		self.assertEqual(int(buffer.success_counts[0]), 4)
		self.assertEqual(int(buffer.trial_counts[0]), 12)

	def test_duplicate_new_group_is_rejected_with_telemetry(self):
		manager = self.manager(buffer_size=2, n=4)
		buffer = self.update(
			manager, manager.reset(), [10, 10], [-1, -1], [0, 4], [4, 4],
			dupe_mask=[False, True])
		self.assertEqual(int(buffer.filled_count[0]), 1)
		self.assertEqual(int(buffer.trial_counts.sum()), 4)
		self.assertEqual(int(buffer.duplicate_new_group_count[0]), 1)

	def test_full_buffer_mixed_new_existing_preserves_level_identity(self):
		manager = self.manager(buffer_size=2, n=4)
		buffer = self.update(
			manager, manager.reset(), [10, 20], [-1, -1], [0, 2], [4, 4])
		buffer = self.update(
			manager, buffer, [99, 10], [-1, 0], [2, 0], [4, 4],
			dupe_mask=[False, False])

		# Existing level 10 is updated against its own 0/4 posterior before
		# new level 99 can evict its slot. The final replacement must contain
		# 99's clean 2/4 counts, never the contaminated 2/8 reproducer.
		ids = np.asarray(buffer.levels["id"]).reshape(-1).tolist()
		counts = {
			level_id: (
				int(np.asarray(buffer.success_counts)[idx]),
				int(np.asarray(buffer.trial_counts)[idx]))
			for idx, level_id in enumerate(ids)
		}
		self.assertEqual(counts, {99: (2, 4), 20: (2, 4)})

	def test_strict_group_size_mismatch_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "frontier_n_eval must equal"):
			PLRManager(
				example_level={"id": jnp.array([0], dtype=jnp.int32)},
				ued_score=UEDScore.RETURN,
				use_frontier_activity=True,
				frontier_n_rollouts=8,
				frontier_n_eval=1,
				frontier_require_n_eval_match=True,
			)
		bridge = PLRManager(
			example_level={"id": jnp.array([0], dtype=jnp.int32)},
			ued_score=UEDScore.RETURN,
			use_frontier_activity=True,
			frontier_n_rollouts=8,
			frontier_n_eval=1,
			frontier_require_n_eval_match=False,
		)
		metrics = bridge.get_metrics(bridge.reset())
		self.assertEqual(float(metrics["frontier_n_rollouts"]), 8.0)
		self.assertEqual(float(metrics["frontier_n_eval"]), 1.0)
		self.assertEqual(float(metrics["frontier_group_size_match"]), 0.0)

	def test_replacement_resets_evicted_posterior(self):
		manager = self.manager(buffer_size=1)
		buffer = self.update(manager, manager.reset(), [10], [-1], [4], [4])
		buffer = self.update(manager, buffer, [99], [-1], [0], [4])
		self.assertEqual(int(buffer.levels["id"][0, 0]), 99)
		self.assertEqual(int(buffer.success_counts[0]), 0)
		self.assertEqual(int(buffer.trial_counts[0]), 4)

	def test_upstream_rank_staleness_mixture_is_unchanged(self):
		manager = self.manager()
		dist = manager._get_replay_dist(
			jnp.array([2.0, 1.0]),
			jnp.array([0, 3], dtype=jnp.uint32),
			jnp.array([True, True]),
		)
		# Rank distribution [2/3,1/3], stale distribution [0,1], coefficient .25.
		np.testing.assert_allclose(np.asarray(dist), [0.5, 0.5], rtol=0, atol=1e-6)

	def test_plr_counts_round_trip_in_train_state_checkpoint(self):
		manager = self.manager()
		buffer = self.update(manager, manager.reset(), [10], [-1], [1], [4])
		tx = optax.sgd(learning_rate=0.01)
		state = VmapTrainState.create(
			apply_fn=lambda *args: None,
			params={"w": jnp.zeros((1, 1), dtype=jnp.float32)},
			tx=tx,
			plr_buffer=buffer,
		)
		checkpoint = pickle.loads(pickle.dumps(state.state_dict))
		blank = state.replace(plr_buffer=manager.reset())
		restored = blank.load_state_dict(checkpoint)
		np.testing.assert_array_equal(
			np.asarray(restored.plr_buffer.success_counts),
			np.asarray(buffer.success_counts))
		np.testing.assert_array_equal(
			np.asarray(restored.plr_buffer.trial_counts),
			np.asarray(buffer.trial_counts))

		# A PLR evidence resume must never silently reset a missing buffer.
		old_checkpoint = dict(checkpoint)
		old_checkpoint.pop("plr_buffer")
		with self.assertRaisesRegex(ValueError, "missing its buffer"):
			blank.load_state_dict(old_checkpoint)

		mismatched_checkpoint = dict(checkpoint)
		mismatched_checkpoint["plr_buffer"] = buffer.replace(
			frontier_prior_alpha=2.0)
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			blank.load_state_dict(mismatched_checkpoint)

		foreign_checkpoint = dict(checkpoint)
		foreign_checkpoint["plr_buffer"] = buffer.replace(
			use_frontier_activity=False)
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			blank.load_state_dict(foreign_checkpoint)

		old_overlay_checkpoint = dict(checkpoint)
		old_overlay_checkpoint["plr_buffer"] = buffer.replace(
			frontier_overlay_version="frontier-activity-v2")
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			blank.load_state_dict(old_overlay_checkpoint)

		plr_static_checkpoint = dict(checkpoint)
		plr_static_checkpoint["plr_buffer"] = buffer.replace(
			replay_prob=0.25, staleness_coef=0.75)
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			blank.load_state_dict(plr_static_checkpoint)

		# The grouped MaxMC control must reject a Frontier or wrong-buffer
		# checkpoint even though it does not consume posterior counts.
		maxmc_manager = PLRManager(
			example_level={"id": jnp.array([0], dtype=jnp.int32)},
			ued_score=UEDScore.MAX_MC,
			buffer_size=2,
			replay_prob=0.5,
			staleness_coef=0.25,
			use_frontier_activity=False,
			frontier_n_eval=4,
		)
		maxmc_state = blank.replace(plr_buffer=maxmc_manager.reset())
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			maxmc_state.load_state_dict(checkpoint)

		maxmc_checkpoint = dict(maxmc_state.state_dict)
		maxmc_checkpoint["plr_buffer"] = maxmc_state.plr_buffer.replace(
			buffer_size=3)
		with self.assertRaisesRegex(ValueError, "configuration mismatch"):
			maxmc_state.load_state_dict(maxmc_checkpoint)


if __name__ == "__main__":
	unittest.main()
