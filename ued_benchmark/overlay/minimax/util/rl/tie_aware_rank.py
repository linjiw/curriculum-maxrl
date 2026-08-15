"""Tie-aware score-rank masses for Prioritized Level Replay.

The source PLR implementation assigns the temperature-transformed mass
``rank**(-1 / temperature)`` after a stable score sort.  Stable sorting makes
the replay probability of exactly tied scores depend on buffer-slot order.
This module preserves the total mass of every exact-score rank block while
sharing that mass equally within the block.

Only filled slots participate.  Equality is exact (so ``+0.0`` and ``-0.0``
tie), and no score jitter is introduced.  A non-finite filled score is a
contract violation: callers receive an all-NaN mass vector so a corrupted
buffer cannot silently become a valid replay distribution.
"""

from jax import lax
import jax.numpy as jnp


def _sorted_tie_blocks(scores, filled):
	"""Return stable order and deterministic contiguous tie-block bounds."""
	scores = jnp.asarray(scores)
	filled = jnp.asarray(filled, dtype=jnp.bool_)
	masked_scores = jnp.where(filled, scores, -jnp.inf)
	order = jnp.argsort(-masked_scores, stable=True)
	sorted_scores = scores[order]
	sorted_filled = filled[order]
	positions = jnp.arange(scores.size, dtype=jnp.int32)
	previous_filled = jnp.concatenate(
		(jnp.zeros((1,), dtype=jnp.bool_), sorted_filled[:-1]))
	previous_scores = jnp.concatenate((sorted_scores[:1], sorted_scores[:-1]))
	starts_group = jnp.logical_and(
		sorted_filled,
		jnp.logical_or(
			positions == 0,
			jnp.logical_or(
				~previous_filled,
				sorted_scores != previous_scores)))
	next_filled = jnp.concatenate(
		(sorted_filled[1:], jnp.zeros((1,), dtype=jnp.bool_)))
	next_scores = jnp.concatenate((sorted_scores[1:], sorted_scores[-1:]))
	ends_group = jnp.logical_and(
		sorted_filled,
		jnp.logical_or(
			positions == scores.size - 1,
			jnp.logical_or(
				~next_filled,
				sorted_scores != next_scores)))
	block_start = lax.associative_scan(
		jnp.maximum, jnp.where(starts_group, positions, 0))
	block_end = jnp.flip(lax.associative_scan(
		jnp.minimum,
		jnp.flip(jnp.where(ends_group, positions, scores.size - 1))))
	block_size = 1 + block_end - block_start
	return (
		order,
		sorted_filled,
		starts_group,
		ends_group,
		block_end,
		block_size,
	)


def _segmented_prefix_sum(values, starts_group):
	"""Prefix-sum each contiguous block without repeated-index updates."""
	def step(carry, inputs):
		value, starts = inputs
		next_value = jnp.where(starts, value, carry + value)
		return next_value, next_value

	_, prefix = lax.scan(
		step,
		jnp.asarray(0.0, dtype=values.dtype),
		(values, starts_group),
	)
	return prefix


def _broadcast_block_totals(block_prefix, ends_group):
	"""Propagate each block-end local total backward through that block."""
	def step(carry, inputs):
		value, ends = inputs
		next_value = jnp.where(ends, value, carry)
		return next_value, next_value

	_, totals = lax.scan(
		step,
		jnp.asarray(0.0, dtype=block_prefix.dtype),
		(block_prefix, ends_group),
		reverse=True,
	)
	return totals


def tie_aware_rank_mass(scores, filled, temperature):
	"""Return final unnormalized rank mass with exact ties shared equally.

	For a tie occupying one-indexed ranks ``l..r``, every member receives
	``mean(j**(-1 / temperature) for j in l..r)``.  Consequently the block's
	unnormalized score-component mass is unchanged mathematically (up to the
	fixed float32 block reduction), and normalization preserves that equality
	within the tie-aware distribution.
	"""
	scores = jnp.asarray(scores)
	filled = jnp.asarray(filled, dtype=jnp.bool_)
	(
		order,
		sorted_filled,
		starts_group,
		ends_group,
		_block_end,
		block_size,
	) = _sorted_tie_blocks(scores, filled)
	ranks = 1.0 + jnp.arange(scores.size, dtype=jnp.float32)
	# Match upstream float32 arithmetic for singleton blocks exactly: upstream
	# first materializes 1/rank, then applies the temperature power.
	rank_mass = (1.0/ranks)**(
		1.0/jnp.asarray(temperature, dtype=jnp.float32))
	rank_mass = jnp.where(sorted_filled, rank_mass, 0.0)
	block_prefix = _segmented_prefix_sum(rank_mass, starts_group)
	block_mass = _broadcast_block_totals(block_prefix, ends_group)
	block_mean = block_mass/block_size.astype(jnp.float32)
	# Avoid a prefix/subtraction or division round-trip for singleton blocks so
	# every raw singleton mass remains bit-identical to upstream when all filled
	# scores are distinct.  Normalized tie-mode probabilities use a separate
	# canonical denominator because upstream's slot-order sum is permutation
	# sensitive; those probabilities are mathematically, not bitwise, equivalent.
	block_mean = jnp.where(block_size == 1, rank_mass, block_mean)
	sorted_mass = jnp.where(sorted_filled, block_mean, 0.0)
	# ``order`` is a permutation, so a second stable sort gives its inverse and
	# restores buffer order without any scatter operation.
	mass = sorted_mass[jnp.argsort(order, stable=True)]
	invalid_filled_score = jnp.logical_and(filled, ~jnp.isfinite(scores)).any()
	return jnp.where(invalid_filled_score, jnp.full_like(mass, jnp.nan), mass)


def exact_score_tie_diagnostics(scores, filled):
	"""Return scalar exact-score tie diagnostics for filled buffer slots."""
	scores = jnp.asarray(scores)
	filled = jnp.asarray(filled, dtype=jnp.bool_)
	(
		_,
		sorted_filled,
		starts_group,
		_ends_group,
		_block_end,
		block_size,
	) = _sorted_tie_blocks(scores, filled)
	block_size = block_size.astype(jnp.uint32)
	distinct = starts_group.sum(dtype=jnp.uint32)
	tied_starts = jnp.logical_and(starts_group, block_size > 1)
	tied_groups = tied_starts.sum(dtype=jnp.uint32)
	tied_levels = jnp.where(tied_starts, block_size, 0).sum(dtype=jnp.uint32)
	max_tie = jnp.where(sorted_filled, block_size, 0).max(initial=0)
	invalid = jnp.logical_and(filled, ~jnp.isfinite(scores)).sum(dtype=jnp.uint32)
	return {
		"distinct_filled_score_count": distinct,
		"score_tie_block_count": tied_groups,
		"score_tied_level_count": tied_levels,
		"score_max_tie_block_size": max_tie,
		"nonfinite_filled_score_count": invalid,
	}


def effective_support(probabilities):
	"""Return inverse Simpson concentration, or zero for zero/invalid mass."""
	probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
	denominator = jnp.square(probabilities).sum()
	valid = jnp.logical_and(jnp.isfinite(denominator), denominator > 0.0)
	return jnp.where(valid, 1.0/denominator, 0.0)
