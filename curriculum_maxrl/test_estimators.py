"""Exact regression tests for the corrected estimator and allocation proofs."""

from __future__ import annotations

from itertools import product
from math import comb

import numpy as np

from curriculum_maxrl.estimators import weights_maxrl_t, weights_maxrl_t_unbiased
from curriculum_maxrl.teachers import allocate_rollouts_greedy


def test_subset_estimators_have_the_claimed_population_weights():
    for n in (2, 4, 8):
        for truncation in range(1, n + 1):
            for p in (0.03, 0.2, 0.7):
                unbiased = 0.0
                dropped = 0.0
                for k in range(n + 1):
                    probability = comb(n, k) * p ** k * (1.0 - p) ** (n - k)
                    rewards = np.array([1.0] * k + [0.0] * (n - k))
                    score = rewards - p  # Bernoulli-logit score
                    unbiased += probability * float(
                        weights_maxrl_t_unbiased(rewards, truncation) @ score
                    )
                    dropped += probability * float(
                        weights_maxrl_t(rewards, truncation) @ score
                    )

                grad_p = p * (1.0 - p)
                population_weight = (
                    1.0 - (1.0 - p) ** truncation
                ) / p
                assert np.isclose(unbiased, population_weight * grad_p)
                assert np.isclose(
                    dropped,
                    (population_weight - (1.0 - p) ** (n - 1)) * grad_p,
                )


def test_greedy_allocation_matches_exhaustive_small_problem():
    p = np.array([0.04, 0.2, 0.7])
    lower, upper, budget = 1, 5, 9
    got = allocate_rollouts_greedy(p, budget, lower, upper)

    def objective(allocation):
        allocation = np.asarray(allocation, dtype=int)
        return float(np.sum(1.0 - (1.0 - p) ** allocation - p))

    feasible = [
        candidate for candidate in product(range(lower, upper + 1), repeat=3)
        if sum(candidate) == budget
    ]
    optimum = max(objective(candidate) for candidate in feasible)
    assert got.sum() == budget
    assert np.isclose(objective(got), optimum)


def test_greedy_allocation_preserves_tiny_supplied_probabilities():
    # Clipping both entries to 1e-4 used to create a false tie and allocate the
    # extra rollout to the wrong task. The theorem applies to the supplied p.
    p = np.array([1e-8, 5e-5])
    got = allocate_rollouts_greedy(p, total_budget=3, n_min=1, n_max=2)
    assert np.array_equal(got, np.array([1, 2]))
