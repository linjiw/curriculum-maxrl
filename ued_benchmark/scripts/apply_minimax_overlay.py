#!/usr/bin/env python3
"""Apply the FrontierRL PLR overlay to a separate pinned minimax clone.

The source-faithful clone is evidence and must remain unchanged.  Create a
second clone/worktree at the pinned commit, run this script with ``--check``,
then opt in with ``--apply``.  Re-running after application is idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PINNED_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
OVERLAY_VERSION = "frontier-activity-v3"
OVERLAY_CONTRACT_SHA256 = "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000"


@dataclass(frozen=True)
class Replacement:
	path: str
	marker: str
	before: str
	after: str


REPLACEMENTS = (
	Replacement(
		"src/minimax/arguments.py",
		"'coefficient_activity'",
		"""        'max_mc',
        'value_disagreement'
""",
		"""        'max_mc',
        'value_disagreement',
        'coefficient_activity'
""",
	),
	Replacement(
		"src/minimax/arguments.py",
		"--frontier_n_rollouts",
		"""plr_subparser.add_argument(
    '--force_unique',
    type=str2bool,
    default=False,
    help='Force level buffer members to be unique.'
)
""",
		"""plr_subparser.add_argument(
    '--force_unique',
    type=str2bool,
    default=False,
    help='Force level buffer members to be unique.'
)
plr_subparser.add_argument(
    '--frontier_n_rollouts',
    type=int,
    default=8,
    help='Declared N in the coefficient-activity score 1-(1-p)^N-p.'
)
plr_subparser.add_argument(
    '--frontier_require_n_eval_match',
    type=str2bool,
    default=True,
    help='Require n_eval to equal frontier_n_rollouts.'
)
plr_subparser.add_argument(
    '--frontier_prior_alpha',
    type=float,
    default=1.0,
    help='Alpha parameter of the per-level Beta success prior.'
)
plr_subparser.add_argument(
    '--frontier_prior_beta',
    type=float,
    default=1.0,
    help='Beta parameter of the per-level Beta success prior.'
)
plr_subparser.add_argument(
    '--frontier_success_threshold',
    type=float,
    default=0.0,
    help='Terminal reward must exceed this value to count as success.'
)
plr_subparser.add_argument(
    '--frontier_posterior_mode',
    type=str,
    choices=['expected_activity', 'mean_plugin'],
    default='expected_activity',
    help='Integrate u_N over the Beta posterior or plug in its mean.'
)
plr_subparser.add_argument(
    '--frontier_overlay_version',
    type=str,
    default='frontier-activity-v3',
    help='Content-contract version persisted in run metadata.'
)
plr_subparser.add_argument(
    '--frontier_overlay_contract_sha256',
    type=str,
    default='5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000',
    help='Content-contract hash persisted in run metadata.'
)
""",
	),
	Replacement(
		"src/minimax/config/xpid_maker.py",
		"elif p.ued_score == 'coefficient_activity'",
		"""\telif p.ued_score == 'value_disagreement':
\t\tued_score = 'vd'
\telse:
""",
		"""\telif p.ued_score == 'value_disagreement':
\t\tued_score = 'vd'
\telif p.ued_score == 'coefficient_activity':
\t\tued_score = 'ca'
\telse:
""",
	),
	Replacement(
		"src/minimax/config/xpid_maker.py",
		"frontier_n_rollouts",
		"""\tplr_info = f'p{p.plr_replay_prob}b{p.plr_buffer_size}t{p.plr_temp}s{p.plr_staleness_coef}m{p.plr_min_fill_ratio}'
\tif p.plr_use_score_ranks:
""",
		"""\toverlay_version = (
\t\tp.plr_frontier_overlay_version or 'frontier-activity-v3')
\toverlay_contract = (
\t\tp.plr_frontier_overlay_contract_sha256
\t\tor '5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000')
\tplr_info = (
\t\tf'p{p.plr_replay_prob}b{p.plr_buffer_size}t{p.plr_temp}'
\t\tf's{p.plr_staleness_coef}m{p.plr_min_fill_ratio}'
\t\tf'_ov{overlay_version.replace("frontier-activity-", "")}'
\t\tf'ch{overlay_contract[:8]}')
\tif p.ued_score == 'coefficient_activity':
\t\tposterior_mode = 'ea' if p.plr_frontier_posterior_mode == 'expected_activity' else 'mp'
\t\tgroup_mode = 'strict' if p.plr_frontier_require_n_eval_match else 'bridge'
\t\tplr_info = (
\t\t\tf'{plr_info}_N{p.plr_frontier_n_rollouts}ne{p.n_eval}'
\t\t\tf'a{p.plr_frontier_prior_alpha}b{p.plr_frontier_prior_beta}'
\t\t\tf'th{p.plr_frontier_success_threshold}'
\t\t\tf'{posterior_mode}{group_mode}')
\tif p.plr_use_score_ranks:
""",
	),
	Replacement(
		"src/minimax/runners/dr_runner.py",
		"reset carry aligned with the flattened",
		"""\t\t\tself.zero_carry = jax.tree_map(lambda x: x.at[:,:self.n_parallel].get(), carry)
""",
		"""\t\t\t# PLR evaluates n_eval streams per level. Keep the recurrent
\t\t\t# reset carry aligned with the flattened (level, eval) batch.
\t\t\tself.zero_carry = jax.tree_map(
\t\t\t\tlambda x: x.at[:,:self.n_parallel*self.n_eval].get(), carry)
""",
	),
	Replacement(
		"src/minimax/runners/plr_runner.py",
		"from minimax.util.rl.frontier_activity import",
		"""from minimax.util.rl import (
	AgentPop,
	VmapTrainState,
	RolloutStorage,
	RollingStats,
	UEDScore,
	compute_ued_scores,
	PopPLRManager
)
""",
		"""from minimax.util.rl import (
	AgentPop,
	VmapTrainState,
	RolloutStorage,
	RollingStats,
	UEDScore,
	compute_ued_scores,
	PopPLRManager
)
from minimax.util.rl.frontier_activity import (
	coefficient_activity_score,
	sparse_goal_stream_counts,
)


def frontier_group_is_valid(trials, n_eval, require_n_eval_match):
	'''Return whether each level cell has usable Bernoulli evidence.'''
	if require_n_eval_match:
		return trials == n_eval
	return trials > 0
""",
	),
	Replacement(
		"src/minimax/runners/plr_runner.py",
		"frontier_success_threshold=0.0",
		"""		use_robust_plr=False,
		use_parallel_eval=False,
		ued_score='l1_value_loss',
""",
		"""		use_robust_plr=False,
		use_parallel_eval=False,
		frontier_n_rollouts=8,
		frontier_require_n_eval_match=True,
		frontier_prior_alpha=1.0,
		frontier_prior_beta=1.0,
		frontier_success_threshold=0.0,
		frontier_posterior_mode='expected_activity',
		frontier_overlay_version='frontier-activity-v3',
		frontier_overlay_contract_sha256='5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000',
		ued_score='l1_value_loss',
""",
	),
	Replacement(
		"src/minimax/runners/plr_runner.py",
		"self.use_frontier_activity = ued_score == 'coefficient_activity'",
		"""		self.use_robust_plr = use_robust_plr
		self.use_parallel_eval = use_parallel_eval
		self.ued_score = UEDScore[ued_score.upper()]
""",
		"""		self.use_robust_plr = use_robust_plr
		self.use_parallel_eval = use_parallel_eval
		self.use_frontier_activity = ued_score == 'coefficient_activity'
		self.frontier_n_rollouts = frontier_n_rollouts
		self.frontier_require_n_eval_match = frontier_require_n_eval_match
		self.frontier_prior_alpha = frontier_prior_alpha
		self.frontier_prior_beta = frontier_prior_beta
		self.frontier_success_threshold = frontier_success_threshold
		self.frontier_posterior_mode = frontier_posterior_mode
		self.frontier_overlay_version = frontier_overlay_version
		self.frontier_overlay_contract_sha256 = frontier_overlay_contract_sha256
		if self.use_frontier_activity:
			if frontier_n_rollouts < 2:
				raise ValueError('frontier_n_rollouts must be at least 2.')
			if frontier_require_n_eval_match and self.n_eval != frontier_n_rollouts:
				raise ValueError(
					'Exact Frontier mode requires n_eval == frontier_n_rollouts; '
					'set frontier_require_n_eval_match=False only for a labeled posterior bridge ablation.')
			if frontier_prior_alpha <= 0 or frontier_prior_beta <= 0:
				raise ValueError('Frontier Beta prior parameters must be positive.')
			if frontier_posterior_mode not in ('expected_activity', 'mean_plugin'):
				raise ValueError('Unknown frontier_posterior_mode.')
			if self.n_devices != 1:
				raise ValueError(
					'Frontier coefficient activity currently requires n_devices=1 for fail-closed checkpoint resume.')
			if frontier_overlay_version != 'frontier-activity-v3':
				raise ValueError('Unexpected Frontier overlay version.')
			if frontier_overlay_contract_sha256 != '5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000':
				raise ValueError('Unexpected Frontier overlay contract hash.')
			# RETURN is a harmless enum placeholder. PLRManager replaces this score
			# before insertion; all PPO and replay logic remains upstream-identical.
			self.ued_score = UEDScore.RETURN
		else:
			self.ued_score = UEDScore[ued_score.upper()]
""",
	),
	Replacement(
		"src/minimax/runners/plr_runner.py",
		"use_frontier_activity=self.use_frontier_activity",
		"""			use_robust_plr=self.use_robust_plr,
			use_parallel_eval=self.use_parallel_eval,
			comparator_fn=self.comparator_fn,
""",
		"""			use_robust_plr=self.use_robust_plr,
			use_parallel_eval=self.use_parallel_eval,
			use_frontier_activity=self.use_frontier_activity,
			frontier_n_rollouts=self.frontier_n_rollouts,
			frontier_n_eval=self.n_eval,
			frontier_require_n_eval_match=self.frontier_require_n_eval_match,
			frontier_prior_alpha=self.frontier_prior_alpha,
			frontier_prior_beta=self.frontier_prior_beta,
			frontier_success_threshold=self.frontier_success_threshold,
			frontier_posterior_mode=self.frontier_posterior_mode,
			frontier_overlay_version=self.frontier_overlay_version,
			frontier_overlay_contract_sha256=self.frontier_overlay_contract_sha256,
			comparator_fn=self.comparator_fn,
""",
	),
	Replacement(
		"src/minimax/runners/plr_runner.py",
		"frontier_successes",
		"""		# Update PLR buffer
		if self.ued_score == UEDScore.MAX_MC:
			max_returns = jax.vmap(lambda x,y: x.at[y].get())(train_state.plr_buffer.max_returns, level_idxs)
			max_returns = jnp.where(
				jnp.greater_equal(level_idxs, 0),
				max_returns,
				jnp.full_like(max_returns, -jnp.inf)
			)
			ued_info = {'max_returns': max_returns}
		else:
			ued_info = None
		ued_scores, ued_score_info = compute_ued_scores(
			self.ued_score, train_batch, self.n_eval, info=ued_info, ignore_val=-jnp.inf, per_agent=True)
""",
		"""		# Update PLR buffer. Frontier activity uses only completed episodes in
		# this rollout and the stored posterior counts; it adds no evaluations.
		if self.use_frontier_activity:
			frontier_successes, frontier_trials = sparse_goal_stream_counts(
				train_batch, self.n_eval, self.frontier_success_threshold)
			safe_level_idxs = jnp.clip(level_idxs, 0, self.buffer_size - 1)
			previous_successes = jax.vmap(
				lambda counts, idxs: counts.take(idxs))(train_state.plr_buffer.success_counts, safe_level_idxs)
			previous_trials = jax.vmap(
				lambda counts, idxs: counts.take(idxs))(train_state.plr_buffer.trial_counts, safe_level_idxs)
			is_existing = jnp.greater_equal(level_idxs, 0)
			previous_successes = jnp.where(is_existing, previous_successes, 0)
			previous_trials = jnp.where(is_existing, previous_trials, 0)
			ued_scores = coefficient_activity_score(
				previous_successes + frontier_successes,
				previous_trials + frontier_trials,
				self.frontier_n_rollouts,
				self.frontier_prior_alpha,
				self.frontier_prior_beta,
				self.frontier_posterior_mode)
			frontier_valid = frontier_group_is_valid(
				frontier_trials, self.n_eval,
				self.frontier_require_n_eval_match)
			ued_scores = jnp.where(frontier_valid, ued_scores, -jnp.inf)
			ued_score_info = {
				'frontier_successes': frontier_successes,
				'frontier_trials': frontier_trials,
			}
		else:
			if self.ued_score == UEDScore.MAX_MC:
				max_returns = jax.vmap(lambda x,y: x.at[y].get())(train_state.plr_buffer.max_returns, level_idxs)
				max_returns = jnp.where(
					jnp.greater_equal(level_idxs, 0),
					max_returns,
					jnp.full_like(max_returns, -jnp.inf)
				)
				ued_info = {'max_returns': max_returns}
			else:
				ued_info = None
			ued_scores, ued_score_info = compute_ued_scores(
				self.ued_score, train_batch, self.n_eval, info=ued_info, ignore_val=-jnp.inf, per_agent=True)
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"from .frontier_activity import coefficient_activity_score",
		"""from .ued_scores import UEDScore
""",
		"""from .ued_scores import UEDScore
from .frontier_activity import coefficient_activity_score
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"success_counts: chex.Array",
		"""	n_mutations: chex.Array

	ued_score: int = struct.field(pytree_node=False, default=UEDScore.L1_VALUE_LOSS.value)
""",
		"""	n_mutations: chex.Array
	success_counts: chex.Array
	trial_counts: chex.Array
	incomplete_group_count: chex.Array
	duplicate_new_group_count: chex.Array

	ued_score: int = struct.field(pytree_node=False, default=UEDScore.L1_VALUE_LOSS.value)
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"use_frontier_activity: bool = struct.field",
		"""	use_parallel_eval: bool = struct.field(pytree_node=False, default=False)


class PLRManager:
""",
		"""	use_parallel_eval: bool = struct.field(pytree_node=False, default=False)
	use_frontier_activity: bool = struct.field(pytree_node=False, default=False)
	frontier_n_rollouts: int = struct.field(pytree_node=False, default=8)
	frontier_n_eval: int = struct.field(pytree_node=False, default=1)
	frontier_require_n_eval_match: bool = struct.field(pytree_node=False, default=True)
	frontier_prior_alpha: float = struct.field(pytree_node=False, default=1.0)
	frontier_prior_beta: float = struct.field(pytree_node=False, default=1.0)
	frontier_success_threshold: float = struct.field(pytree_node=False, default=0.0)
	frontier_posterior_mode: str = struct.field(pytree_node=False, default='expected_activity')
	frontier_overlay_version: str = struct.field(pytree_node=False, default='frontier-activity-v3')
	frontier_overlay_contract_sha256: str = struct.field(pytree_node=False, default='5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000')


class PLRManager:
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"use_frontier_activity=False",
		"""		use_robust_plr=False,
		use_parallel_eval=False,
		comparator_fn=None,
""",
		"""		use_robust_plr=False,
		use_parallel_eval=False,
		use_frontier_activity=False,
		frontier_n_rollouts=8,
		frontier_n_eval=1,
		frontier_require_n_eval_match=True,
		frontier_prior_alpha=1.0,
		frontier_prior_beta=1.0,
		frontier_success_threshold=0.0,
		frontier_posterior_mode='expected_activity',
		frontier_overlay_version='frontier-activity-v3',
		frontier_overlay_contract_sha256='5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000',
		comparator_fn=None,
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"self.frontier_n_rollouts = frontier_n_rollouts",
		"""		self.use_robust_plr = use_robust_plr
		self.use_parallel_eval = use_parallel_eval
		self.comparator_fn = comparator_fn
""",
		"""		self.use_robust_plr = use_robust_plr
		self.use_parallel_eval = use_parallel_eval
		self.use_frontier_activity = use_frontier_activity
		self.frontier_n_rollouts = frontier_n_rollouts
		self.frontier_n_eval = frontier_n_eval
		self.frontier_require_n_eval_match = frontier_require_n_eval_match
		self.frontier_prior_alpha = frontier_prior_alpha
		self.frontier_prior_beta = frontier_prior_beta
		self.frontier_success_threshold = frontier_success_threshold
		self.frontier_posterior_mode = frontier_posterior_mode
		self.frontier_overlay_version = frontier_overlay_version
		self.frontier_overlay_contract_sha256 = frontier_overlay_contract_sha256
		if use_frontier_activity:
			if frontier_n_rollouts < 2:
				raise ValueError('frontier_n_rollouts must be at least 2.')
			if frontier_require_n_eval_match and frontier_n_eval != frontier_n_rollouts:
				raise ValueError(
					'frontier_n_eval must equal frontier_n_rollouts in strict mode.')
			if frontier_prior_alpha <= 0 or frontier_prior_beta <= 0:
				raise ValueError('Frontier Beta prior parameters must be positive.')
			if frontier_posterior_mode not in ('expected_activity', 'mean_plugin'):
				raise ValueError('Unknown frontier_posterior_mode.')
		self.comparator_fn = comparator_fn
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"self.success_counts = jnp.zeros",
		"""		self.n_mutations = jnp.zeros(buffer_size, dtype=jnp.uint32)


	partial(jax.jit, static_argnums=(0,))
""",
		"""		self.n_mutations = jnp.zeros(buffer_size, dtype=jnp.uint32)
		self.success_counts = jnp.zeros(buffer_size, dtype=jnp.uint32)
		self.trial_counts = jnp.zeros(buffer_size, dtype=jnp.uint32)
		self.incomplete_group_count = jnp.zeros((1,), dtype=jnp.uint32)
		self.duplicate_new_group_count = jnp.zeros((1,), dtype=jnp.uint32)


	partial(jax.jit, static_argnums=(0,))
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"success_counts=self.success_counts",
		"""			use_robust_plr=self.use_robust_plr,
			use_parallel_eval=self.use_parallel_eval,
			levels=self.levels,
""",
		"""			use_robust_plr=self.use_robust_plr,
			use_parallel_eval=self.use_parallel_eval,
			use_frontier_activity=self.use_frontier_activity,
			frontier_n_rollouts=self.frontier_n_rollouts,
			frontier_n_eval=self.frontier_n_eval,
			frontier_require_n_eval_match=self.frontier_require_n_eval_match,
			frontier_prior_alpha=self.frontier_prior_alpha,
			frontier_prior_beta=self.frontier_prior_beta,
			frontier_success_threshold=self.frontier_success_threshold,
			frontier_posterior_mode=self.frontier_posterior_mode,
			frontier_overlay_version=self.frontier_overlay_version,
			frontier_overlay_contract_sha256=self.frontier_overlay_contract_sha256,
			levels=self.levels,
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"trial_counts=self.trial_counts",
		"""			filled_count=self.filled_count,
			n_mutations=self.n_mutations)
""",
		"""			filled_count=self.filled_count,
			n_mutations=self.n_mutations,
			success_counts=self.success_counts,
			trial_counts=self.trial_counts,
			incomplete_group_count=self.incomplete_group_count,
			duplicate_new_group_count=self.duplicate_new_group_count)
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"incomplete_group_count=plr_buffer.incomplete_group_count",
		"""		# Note: parent_idxs are only used for mutated levels
		done_masks = (ued_scores != ignore_val)
""",
		"""		# Note: parent_idxs are only used for mutated levels
		done_masks = (ued_scores != ignore_val)
		if self.use_frontier_activity:
			if self.frontier_require_n_eval_match:
				done_masks = info['frontier_trials'] == self.frontier_n_eval
				plr_buffer = plr_buffer.replace(
					incomplete_group_count=plr_buffer.incomplete_group_count
					+ jnp.array([(~done_masks).sum()], dtype=jnp.uint32))
			else:
				done_masks = info['frontier_trials'] > 0
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"frontier_duplicate_suppress_mask",
		"""\t\tif dupe_mask is not None:
\t\t\tdone_masks = jnp.logical_and(done_masks, ~dupe_mask) # Ignore duplicate levels in batch by treating them as not done
""",
		"""\t\tif dupe_mask is not None:
\t\t\tif self.use_frontier_activity:
\t\t\t\t# Existing replay duplicates are processed sequentially so every
\t\t\t\t# evaluation group updates the shared posterior. Preserve upstream
\t\t\t\t# suppression only for duplicate not-yet-buffered levels.
\t\t\t\tfrontier_duplicate_suppress_mask = jnp.logical_and(
\t\t\t\t\tdupe_mask, jnp.less(level_idxs, 0))
\t\t\t\tplr_buffer = plr_buffer.replace(
\t\t\t\t\tduplicate_new_group_count=plr_buffer.duplicate_new_group_count
\t\t\t\t\t+ jnp.array([frontier_duplicate_suppress_mask.sum()], dtype=jnp.uint32))
\t\t\t\tdone_masks = jnp.logical_and(done_masks, ~frontier_duplicate_suppress_mask)
\t\t\telse:
\t\t\t\tdone_masks = jnp.logical_and(done_masks, ~dupe_mask)
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"observed_successes, observed_trials = step",
		"""			score, level, level_idx, done_mask, parent_idx, max_return = step

			next_insert_idx = self._get_next_insert_idx(plr_buffer)
			is_new_level = jnp.greater(0, level_idx)
			insert_idx = jnp.where(
				is_new_level,
				next_insert_idx, # new level
				level_idx,
			)
""",
		"""			score, level, level_idx, done_mask, parent_idx, max_return, observed_successes, observed_trials = step

			next_insert_idx = self._get_next_insert_idx(plr_buffer)
			is_new_level = jnp.greater(0, level_idx)
			insert_idx = jnp.where(
				is_new_level,
				next_insert_idx, # new level
				level_idx,
			)

			base_successes = jnp.where(
				is_new_level, 0, plr_buffer.success_counts.at[insert_idx].get())
			base_trials = jnp.where(
				is_new_level, 0, plr_buffer.trial_counts.at[insert_idx].get())
			candidate_successes = base_successes + observed_successes
			candidate_trials = base_trials + observed_trials
			if self.use_frontier_activity:
				score = coefficient_activity_score(
					candidate_successes,
					candidate_trials,
					self.frontier_n_rollouts,
					self.frontier_prior_alpha,
					self.frontier_prior_beta,
					self.frontier_posterior_mode)
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"next_success_counts = jnp.where",
		"""			next_max_returns = jnp.where(
				should_insert_or_update,
				plr_buffer.max_returns.at[insert_idx].set(max_return),
				plr_buffer.max_returns
			)

			updated_level = jax.tree_map(
""",
		"""			next_max_returns = jnp.where(
				should_insert_or_update,
				plr_buffer.max_returns.at[insert_idx].set(max_return),
				plr_buffer.max_returns
			)
			next_success_counts = jnp.where(
				should_insert_or_update,
				plr_buffer.success_counts.at[insert_idx].set(candidate_successes),
				plr_buffer.success_counts
			)
			next_trial_counts = jnp.where(
				should_insert_or_update,
				plr_buffer.trial_counts.at[insert_idx].set(candidate_trials),
				plr_buffer.trial_counts
			)

			updated_level = jax.tree_map(
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"success_counts=next_success_counts",
		"""				filled_count=jnp.array([next_filled.sum()]),
				max_returns=next_max_returns
""",
		"""				filled_count=jnp.array([next_filled.sum()]),
				max_returns=next_max_returns,
				success_counts=next_success_counts,
				trial_counts=next_trial_counts
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"frontier_successes = info['frontier_successes']",
		"""		if plr_buffer.ued_score == UEDScore.MAX_MC.value:
			max_returns = info['max_returns']
		else:
			max_returns = jnp.full_like(level_idxs, -1)
		carry = (ued_scores, levels, level_idxs, done_masks, parent_idxs, max_returns)
""",
		"""		if plr_buffer.ued_score == UEDScore.MAX_MC.value:
			max_returns = info['max_returns']
		else:
			max_returns = jnp.full_like(level_idxs, -1)
		if self.use_frontier_activity:
			frontier_successes = info['frontier_successes']
			frontier_trials = info['frontier_trials']
		else:
			frontier_successes = jnp.zeros_like(level_idxs, dtype=jnp.uint32)
			frontier_trials = jnp.zeros_like(level_idxs, dtype=jnp.uint32)
		carry = (
			ued_scores, levels, level_idxs, done_masks, parent_idxs, max_returns,
			frontier_successes, frontier_trials)
		if self.use_frontier_activity:
			# A new insertion may evict a slot referenced by another element in
			# this batch. Resolve all existing identities first, preserving the
			# original order among duplicate observations of the same slot.
			existing_first_order = jnp.argsort(
				jnp.less(level_idxs, 0), stable=True)
			carry = jax.tree_util.tree_map(
				lambda x: x[existing_first_order], carry)
""",
	),
	Replacement(
		"src/minimax/util/rl/plr.py",
		"weighted_frontier_probability",
		"""		return dict(
			weighted_n_mutations=weighted_n_mutations,
			weighted_ued_score=weighted_ued_score,
			weighted_age=weighted_age
		)
""",
		"""		metrics = dict(
			weighted_n_mutations=weighted_n_mutations,
			weighted_ued_score=weighted_ued_score,
			weighted_age=weighted_age
		)
		if self.use_frontier_activity:
			posterior_p = (
				plr_buffer.success_counts + self.frontier_prior_alpha
			)/(plr_buffer.trial_counts + self.frontier_prior_alpha + self.frontier_prior_beta)
			metrics.update(dict(
				frontier_n_rollouts=jnp.asarray(self.frontier_n_rollouts, dtype=jnp.float32),
				frontier_n_eval=jnp.asarray(self.frontier_n_eval, dtype=jnp.float32),
				frontier_group_size_match=jnp.asarray(
					self.frontier_n_rollouts == self.frontier_n_eval, dtype=jnp.float32),
				weighted_frontier_probability=(posterior_p*replay_dist).sum(),
				weighted_frontier_trials=(plr_buffer.trial_counts*replay_dist).sum(),
				frontier_total_trials=plr_buffer.trial_counts.sum(),
				frontier_total_successes=plr_buffer.success_counts.sum(),
				frontier_incomplete_group_count=plr_buffer.incomplete_group_count[0],
				frontier_duplicate_new_group_count=plr_buffer.duplicate_new_group_count[0],
			))
		return metrics
""",
	),
	Replacement(
		"src/minimax/util/rl/training.py",
		"plr_buffer=self.plr_buffer",
		"""      params=self.params,
      opt_state=self.opt_state
    )
""",
		"""      params=self.params,
      opt_state=self.opt_state,
      plr_buffer=self.plr_buffer
    )
""",
	),
	Replacement(
		"src/minimax/util/rl/training.py",
		"Frontier checkpoint is missing its PLR posterior",
		"""  def load_state_dict(self, state):
    return self.replace(
      n_iters=state['n_iters'],
      n_updates=state['n_updates'],
      n_grad_updates=state['n_grad_updates'],
      params=state['params'],
      opt_state=state['opt_state']
    )
""",
		"""  def load_state_dict(self, state):
    saved_plr_buffer = state.get('plr_buffer')
    if self.plr_buffer is not None:
      if saved_plr_buffer is None:
        raise ValueError('PLR checkpoint is missing its buffer; refusing a curriculum-changing resume.')
      signature_fields = (
	        'buffer_size', 'ued_score', 'replay_prob', 'staleness_coef',
	        'temp', 'use_score_ranks', 'min_fill_ratio',
	        'use_robust_plr', 'use_parallel_eval',
	        'use_frontier_activity', 'frontier_n_rollouts', 'frontier_n_eval',
        'frontier_require_n_eval_match', 'frontier_prior_alpha',
        'frontier_prior_beta', 'frontier_success_threshold',
	        'frontier_posterior_mode', 'frontier_overlay_version',
	        'frontier_overlay_contract_sha256')
      for field in signature_fields:
        if getattr(saved_plr_buffer, field, None) != getattr(self.plr_buffer, field):
          raise ValueError(f'PLR checkpoint configuration mismatch: {field}.')
    elif saved_plr_buffer is not None:
      raise ValueError('Checkpoint contains a PLR buffer but the current runner does not.')

    return self.replace(
      n_iters=state['n_iters'],
      n_updates=state['n_updates'],
      n_grad_updates=state['n_grad_updates'],
      params=state['params'],
      opt_state=state['opt_state'],
      plr_buffer=saved_plr_buffer
    )
""",
	),
)


def git_head(target: Path) -> str:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"], cwd=target, check=True,
		text=True, capture_output=True)
	return result.stdout.strip()


def replacement_status(target: Path, spec: Replacement) -> tuple[str, str]:
	path = target/spec.path
	if not path.is_file():
		raise RuntimeError(f"missing pinned source file: {path}")
	text = path.read_text()
	after_count = text.count(spec.after)
	if after_count == 1:
		return "applied", text
	if after_count > 1:
		raise RuntimeError(
			f"applied overlay block is duplicated in {spec.path}")
	if spec.marker in text:
		raise RuntimeError(
			f"overlay marker is present but exact applied content differs in "
			f"{spec.path}; use a fresh pinned clone")
	count = text.count(spec.before)
	if count != 1:
		raise RuntimeError(
			f"expected exactly one source anchor in {spec.path}; found {count}")
	return "pending", text


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=Path, required=True)
	mode = parser.add_mutually_exclusive_group(required=True)
	mode.add_argument("--check", action="store_true")
	mode.add_argument("--apply", action="store_true")
	args = parser.parse_args()

	contract_path = Path(__file__).resolve().parents[1]/"OVERLAY_CONTRACT.json"
	contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
	if contract_sha256 != OVERLAY_CONTRACT_SHA256:
		raise RuntimeError(
			f"overlay contract hash mismatch: {contract_sha256}")
	contract = json.loads(contract_path.read_text())
	scoring_module = (
		Path(__file__).resolve().parents[1]
		/"overlay/minimax/util/rl/frontier_activity.py")
	scoring_sha256 = hashlib.sha256(scoring_module.read_bytes()).hexdigest()
	if scoring_sha256 != contract["scoring_module_sha256"]:
		raise RuntimeError(
			f"overlay scoring-module hash mismatch: {scoring_sha256}")

	target = args.target.expanduser().resolve()
	if git_head(target) != PINNED_COMMIT:
		raise RuntimeError(
			f"target HEAD must be {PINNED_COMMIT}; got {git_head(target)}")

	overlay_files = {spec.path for spec in REPLACEMENTS}
	overlay_files.add("src/minimax/util/rl/frontier_activity.py")
	manifest_path = target/".frontierrl_overlay.json"
	if manifest_path.exists():
		try:
			previous_manifest = json.loads(manifest_path.read_text())
		except json.JSONDecodeError as exc:
			raise RuntimeError("existing overlay manifest is invalid JSON") from exc
		expected_identity = {
			"base_commit": PINNED_COMMIT,
			"overlay": OVERLAY_VERSION,
			"overlay_contract_sha256": OVERLAY_CONTRACT_SHA256,
		}
		for key, value in expected_identity.items():
			if previous_manifest.get(key) != value:
				raise RuntimeError(
					f"existing overlay manifest has incompatible {key}; "
					"use a fresh pinned clone")
		if previous_manifest.get("overlay_files") != sorted(overlay_files):
			raise RuntimeError("existing overlay manifest file list differs")
		manifest_hashes = previous_manifest.get("overlay_file_sha256")
		if not isinstance(manifest_hashes, dict) \
			or set(manifest_hashes) != overlay_files:
			raise RuntimeError("existing overlay manifest lacks exact file hashes")
		for relative_path in sorted(overlay_files):
			applied_path = target/relative_path
			if not applied_path.is_file():
				raise RuntimeError(
					f"manifest overlay file is missing: {relative_path}")
			actual_sha256 = hashlib.sha256(applied_path.read_bytes()).hexdigest()
			if actual_sha256 != manifest_hashes[relative_path]:
				raise RuntimeError(
					f"manifest hash mismatch for overlay file: {relative_path}")

	statuses: list[tuple[Replacement, str]] = []
	for spec in REPLACEMENTS:
		status, _ = replacement_status(target, spec)
		statuses.append((spec, status))

	overlay_source = (
		Path(__file__).resolve().parents[1]
		/"overlay/minimax/util/rl/frontier_activity.py")
	overlay_target = target/"src/minimax/util/rl/frontier_activity.py"
	if overlay_target.exists():
		if overlay_target.read_bytes() != overlay_source.read_bytes():
			raise RuntimeError(f"different overlay module already exists: {overlay_target}")
		module_status = "applied"
	else:
		module_status = "pending"

	pending = sum(status == "pending" for _, status in statuses)
	pending += module_status == "pending"
	if args.check:
		if pending == 0 and not manifest_path.exists():
			raise RuntimeError(
				"applied overlay is missing .frontierrl_overlay.json")
		print(json.dumps({
			"commit": PINNED_COMMIT,
			"overlay": OVERLAY_VERSION,
			"overlay_contract_sha256": OVERLAY_CONTRACT_SHA256,
			"pending_changes": pending,
			"status": "already_applied" if pending == 0 else "applicable",
		}, sort_keys=True))
		return 0

	changed_files: set[str] = set()
	for spec, _ in statuses:
		# Re-read after every edit: several replacements intentionally target
		# the same source file.
		status, source = replacement_status(target, spec)
		if status == "pending":
			path = target/spec.path
			path.write_text(source.replace(spec.before, spec.after, 1))
			changed_files.add(spec.path)
	if module_status == "pending":
		shutil.copyfile(overlay_source, overlay_target)
		changed_files.add("src/minimax/util/rl/frontier_activity.py")

	overlay_file_sha256 = {
		path: hashlib.sha256((target/path).read_bytes()).hexdigest()
		for path in sorted(overlay_files)
	}
	manifest = {
		"base_commit": PINNED_COMMIT,
		"overlay": OVERLAY_VERSION,
		"overlay_contract_sha256": OVERLAY_CONTRACT_SHA256,
		"overlay_files": sorted(overlay_files),
		"overlay_file_sha256": overlay_file_sha256,
	}
	manifest_path.write_text(
		json.dumps(manifest, indent=2, sort_keys=True) + "\n")
	print(json.dumps({**manifest, "status": "applied"}, sort_keys=True))
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except (RuntimeError, subprocess.CalledProcessError) as exc:
		print(f"overlay error: {exc}", file=sys.stderr)
		raise SystemExit(2)
