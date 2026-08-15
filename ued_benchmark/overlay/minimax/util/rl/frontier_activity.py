"""FrontierRL coefficient-activity scoring for sparse-goal PLR.

This module is an overlay for facebookresearch/minimax at commit
``d053054c5290a04c1c4cd8b55704d999cad73e30``.  It deliberately depends only
on rollout rewards and terminal flags, so updating the curriculum requires no
additional environment interactions when ``n_eval=1``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def validate_success_trial_counts(successes, trials):
	"""Validate concrete (non-traced) Bernoulli sufficient statistics."""
	successes = np.asarray(successes)
	trials = np.asarray(trials)
	if np.any(successes < 0) or np.any(trials < 0):
		raise ValueError("Frontier success and trial counts must be nonnegative.")
	if np.any(successes > trials):
		raise ValueError("Frontier success counts cannot exceed trial counts.")


def beta_posterior_mean(successes, trials, prior_alpha=1.0, prior_beta=1.0):
	"""Return E[p | successes, trials] for a Beta-Bernoulli model."""
	successes = jnp.asarray(successes, dtype=jnp.float32)
	trials = jnp.asarray(trials, dtype=jnp.float32)
	return (successes + prior_alpha)/(trials + prior_alpha + prior_beta)


def expected_coefficient_activity(
		successes,
		trials,
		frontier_n_rollouts,
		prior_alpha=1.0,
		prior_beta=1.0):
	"""Return the exact Beta-posterior expectation of coefficient activity.

	For posterior ``Beta(a, b)``, this evaluates
	``1 - (b)_N/(a+b)_N - a/(a+b)`` using a finite product. This integrates
	posterior uncertainty and, by concavity, is no larger than evaluating
	activity at the posterior mean for ``N >= 2``.
	"""
	successes = jnp.asarray(successes, dtype=jnp.float32)
	trials = jnp.asarray(trials, dtype=jnp.float32)
	a = successes + prior_alpha
	b = trials - successes + prior_beta
	offsets = jnp.arange(frontier_n_rollouts, dtype=jnp.float32)
	failure_moment = jnp.prod(
		(b[..., jnp.newaxis] + offsets)
		/(a[..., jnp.newaxis] + b[..., jnp.newaxis] + offsets),
		axis=-1)
	return jnp.clip(1.0 - failure_moment - a/(a + b), 0.0, 1.0)


def coefficient_activity_score(
		successes,
		trials,
		frontier_n_rollouts,
		prior_alpha=1.0,
		prior_beta=1.0,
		posterior_mode='expected_activity'):
	"""Score a Beta posterior by one of two explicit activity modes.

	``p`` is the Beta-posterior mean success probability for a replay-buffer
	level. ``frontier_n_rollouts`` is the explicitly declared estimator group
	size N; it is not silently inferred from the environment batch shape.
	``expected_activity`` computes ``E[u_N(p)]`` under the full posterior, while
	``mean_plugin`` computes ``u_N(E[p])`` and is retained as an ablation.
	"""
	if posterior_mode == 'expected_activity':
		return expected_coefficient_activity(
			successes,
			trials,
			frontier_n_rollouts,
			prior_alpha,
			prior_beta)
	if posterior_mode != 'mean_plugin':
		raise ValueError(f"Unknown frontier posterior mode: {posterior_mode}")
	p = beta_posterior_mean(successes, trials, prior_alpha, prior_beta)
	return jnp.clip(1.0 - (1.0 - p)**frontier_n_rollouts - p, 0.0, 1.0)


@partial(jax.jit, static_argnums=(1,))
def sparse_goal_stream_counts(batch, n_eval, success_threshold=0.0):
	"""Count observed and successful evaluation streams per level.

	Each of the ``n_eval`` streams contributes at most one Bernoulli observation,
	regardless of how many episodes it completes inside the rollout. A stream is
	observed if any ``done`` occurs and successful if any terminal reward is
	strictly above ``success_threshold``. This prevents early auto-resets from
	quietly making the effective group size larger than ``n_eval``.

	Returns arrays with shape ``(n_students, n_levels)``.  Partial episodes at
	the end of a rollout contribute neither a success nor a trial.
	"""
	n_students, n_steps, flat_batch_size = batch.dones.shape
	if flat_batch_size % n_eval != 0:
		raise ValueError(
			f"flat batch size {flat_batch_size} is not divisible by n_eval={n_eval}")
	n_levels = flat_batch_size//n_eval
	shape = (n_students, n_steps, n_levels, n_eval)
	dones = batch.dones.reshape(shape).astype(jnp.bool_)
	rewards = batch.rewards.reshape(shape)

	observed_streams = dones.any(axis=1)
	successful_streams = jnp.logical_and(
		dones, rewards > success_threshold).any(axis=1)
	trials = observed_streams.sum(axis=-1, dtype=jnp.uint32)
	successes = successful_streams.sum(axis=-1, dtype=jnp.uint32)
	return successes, trials
