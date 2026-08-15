"""Parsed-config integration smokes for the grouped 4x8 benchmark lanes.

These tests retain the authored 4-level x 8-evaluation LSTM layout while
shrinking only rollout/update/model sizes so a reset and one real runner step
remain suitable for CPU validation.
"""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import jax
import numpy as np

from minimax.arguments import parser
from minimax.runners import ExperimentRunner


CONFIG_DIR = Path(__file__).resolve().parents[1]/"configs"


def _parse_authored_config(name):
	config = json.loads((CONFIG_DIR/f"{name}.json").read_text())["args"]
	argv = ["grouped-runner-smoke"]
	for key, values in config.items():
		value = values[0] if isinstance(values, list) else values
		argv.append(f"--{key}={value}")
	with mock.patch.object(sys, "argv", argv):
		return parser.parse_args()


def _make_bounded_runner(name):
	args = _parse_authored_config(name)
	# Preserve the recurrent 4x8 batch that this integration test targets.
	assert args.train_runner_args.n_parallel == 4
	assert args.train_runner_args.n_eval == 8
	assert args.student_model_args.recurrent_arch == "lstm"

	# Bound compilation and execution without changing grouped carry geometry.
	# A two-step horizon makes every evaluation stream produce exactly one
	# terminal Bernoulli observation during the two-step rollout.
	args.train_runner_args.n_rollout_steps = 2
	args.train_runner_args.n_unroll_rollout = 1
	args.train_runner_args.buffer_size = 8
	args.train_runner_args.min_fill_ratio = 0.5
	args.train_runner_args.replay_prob = 1.0
	args.env_args.max_episode_steps = 2
	args.student_rl_args.n_unroll_update = 1
	args.student_rl_args.n_epochs = 1
	args.student_model_args.hidden_dim = 16
	args.student_model_args.recurrent_hidden_dim = 16
	args.student_model_args.n_conv_filters = 4
	args.eval_args.env_names = None

	p = copy.deepcopy(args)
	runner = ExperimentRunner(
		train_runner=p.train_runner,
		env_name=p.env_name,
		agent_rl_algo=p.agent_rl_algo,
		student_model_name=p.student_model_name,
		teacher_model_name=p.teacher_model_name,
		train_runner_kwargs=p.train_runner_args,
		env_kwargs=p.env_args,
		ued_env_kwargs=p.ued_env_args,
		student_rl_kwargs=p.student_rl_args,
		teacher_rl_kwargs=p.teacher_rl_args,
		student_model_kwargs=p.student_model_args,
		teacher_model_kwargs=p.teacher_model_args,
		eval_kwargs=p.eval_args,
		eval_env_kwargs=p.eval_env_args,
		n_devices=p.n_devices,
	)
	return args, runner


class GroupedRunnerSmokeTest(unittest.TestCase):
	def _reset_and_two_steps(self, config_name, expect_frontier):
		args, experiment = _make_bounded_runner(config_name)
		state = experiment.runner.reset(jax.random.PRNGKey(args.seed))
		carry_shapes = [x.shape for x in jax.tree_util.tree_leaves(
			experiment.runner.zero_carry)]
		self.assertTrue(carry_shapes)
		self.assertTrue(all(shape[:2] == (1, 32) for shape in carry_shapes))
		self.assertEqual(
			bool(state[1].plr_buffer.use_frontier_activity), expect_frontier)

		stats, _eval_stats, *next_state = experiment.step(state, False)
		jax.tree_util.tree_map(
			lambda x: x.block_until_ready()
			if hasattr(x, "block_until_ready") else x,
			(stats, next_state),
		)
		self.assertGreaterEqual(
			int(np.asarray(next_state[1].n_iters).reshape(-1)[0]), 1)
		buffer = next_state[1].plr_buffer
		self.assertEqual(
			int(np.asarray(buffer.filled_count).reshape(-1)[0]), 4)
		if expect_frontier:
			self.assertEqual(int(np.asarray(buffer.trial_counts).sum()), 32)
			self.assertEqual(
				int(np.asarray(buffer.incomplete_group_count).reshape(-1)[0]), 0)
			self.assertEqual(
				int(np.asarray(buffer.duplicate_new_group_count).reshape(-1)[0]), 0)

		# The first outer cycle warms exactly half of the eight-slot buffer. A
		# forced replay on cycle two must execute a real PPO update.
		stats, _eval_stats, *final_state = experiment.step(next_state, False)
		jax.tree_util.tree_map(
			lambda x: x.block_until_ready()
			if hasattr(x, "block_until_ready") else x,
			(stats, final_state),
		)
		self.assertEqual(
			int(np.asarray(final_state[1].n_updates).reshape(-1)[0]), 1)
		self.assertGreater(
			int(np.asarray(final_state[1].n_grad_updates).reshape(-1)[0]), 0)

	def test_frontier_exact_grouped_insert_and_ppo_step(self):
		self._reset_and_two_steps("maze_frontier_exact_grouped_n8", True)

	def test_maxmc_group_matched_insert_and_ppo_step(self):
		self._reset_and_two_steps("maze_maxmc_group_matched_4x8_b500", False)


if __name__ == "__main__":
	unittest.main()
