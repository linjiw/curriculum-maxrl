"""Finite-N checks for the algebra used by the Frontier RL curriculum."""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from frontier_rl.estimators import (  # noqa: E402
    EPS,
    grpo_weights,
    maxrl_eq10_weights,
    maxrl_weights,
    rloo_weights,
)
from estimators import weights_maxrl_t, weights_maxrl_t_eq10  # noqa: E402
from teachers import (  # noqa: E402
    UniformTeacher,
    allocate_rollouts_adaptive,
    allocate_rollouts_greedy,
)


def expected_mass(weight_fn, n: int, p: float) -> float:
    total = 0.0
    for k in range(n + 1):
        rewards = np.r_[np.ones(k), np.zeros(n - k)]
        probability = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        total += probability * np.abs(weight_fn(rewards)).sum()
    return total


def expected_bernoulli_gradient(weight_fn, n: int, p: float) -> float:
    """Enumerate E[sum_i w_i grad_log Bernoulli(r_i; p)] / grad(p)."""
    total = 0.0
    for outcomes in itertools.product((0.0, 1.0), repeat=n):
        rewards = np.asarray(outcomes)
        probability = p ** rewards.sum() * (1 - p) ** (n - rewards.sum())
        scores = rewards / p - (1 - rewards) / (1 - p)
        total += probability * float(np.dot(weight_fn(rewards), scores))
    return total


@pytest.mark.parametrize("n", [2, 4, 16, 32])
@pytest.mark.parametrize("p", [0.005, 0.05, 0.2, 0.6, 0.95])
def test_maxrl_expected_mass_identity(n, p):
    exact = 2 * ((1 - (1 - p) ** n) - p)
    assert expected_mass(maxrl_weights, n, p) == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("n", [2, 4, 16, 32])
@pytest.mark.parametrize("p", [0.005, 0.05, 0.2, 0.6, 0.95])
def test_rloo_expected_mass_is_exactly_learnability(n, p):
    # With the implementation's final /N normalization there is no N/(N-1)
    # factor: E sum|w| = 2p(1-p) for every N >= 2.
    assert expected_mass(rloo_weights, n, p) == pytest.approx(2 * p * (1 - p), abs=1e-12)


@pytest.mark.parametrize("n", [2, 4, 16])
def test_grpo_conditional_mass_matches_sample_std_normalization(n):
    for k in range(1, n):
        rewards = np.r_[np.ones(k), np.zeros(n - k)]
        sample_std = math.sqrt(k * (n - k) / (n * (n - 1)))
        exact = 2 * k * (n - k) / (n * n * (sample_std + EPS))
        assert np.abs(grpo_weights(rewards)).sum() == pytest.approx(exact, abs=1e-12)


def test_advmass_peak_and_learnability_member():
    for n in [2, 4, 16, 32]:
        p_star = 1 - n ** (-1 / (n - 1))
        grid = np.linspace(0.0, 1.0, 200_001)
        utility = (1 - (1 - grid) ** n) - grid
        assert grid[np.argmax(utility)] == pytest.approx(p_star, abs=1e-5)

    p = np.linspace(0.0, 1.0, 101)
    # The exact advantage-mass family equals learnability at N=2. At N=1
    # it is identically zero; N=1 applies only to the older pass@N*(1-p)
    # heuristic.
    assert np.allclose((1 - (1 - p) ** 2) - p, p * (1 - p))
    assert np.allclose((1 - (1 - p)) - p, 0.0)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("p", [0.03, 0.2, 0.65])
def test_practical_algorithm_one_has_order_n_minus_one_expectation(n, p):
    expected_weight = (1 - (1 - p) ** (n - 1)) / p
    assert expected_bernoulli_gradient(maxrl_weights, n, p) == pytest.approx(
        expected_weight, abs=1e-12
    )


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("p", [0.03, 0.2, 0.65])
def test_full_eq10_control_has_order_n_expectation(n, p):
    expected_weight = (1 - (1 - p) ** n) / p
    assert expected_bernoulli_gradient(maxrl_eq10_weights, n, p) == pytest.approx(
        expected_weight, abs=1e-12
    )


@pytest.mark.parametrize("n,t", [(4, 1), (4, 3), (6, 2), (6, 6)])
@pytest.mark.parametrize("p", [0.05, 0.4])
def test_subset_control_distinguishes_exact_and_dropped_variants(n, t, p):
    exact_weight = (1 - (1 - p) ** t) / p
    dropped_weight = exact_weight - (1 - p) ** (n - 1)
    exact_fn = lambda rewards: weights_maxrl_t_eq10(rewards, t)
    dropped_fn = lambda rewards: weights_maxrl_t(rewards, t)
    assert expected_bernoulli_gradient(exact_fn, n, p) == pytest.approx(
        exact_weight, abs=1e-12
    )
    assert expected_bernoulli_gradient(dropped_fn, n, p) == pytest.approx(
        dropped_weight, abs=1e-12
    )


def test_greedy_rollout_allocation_matches_brute_force():
    p = np.array([0.08, 0.25, 0.7])
    budget, n_min, n_max = 9, 1, 5
    greedy = allocate_rollouts_greedy(p, budget, n_min=n_min, n_max=n_max)

    def objective(allocation):
        allocation = np.asarray(allocation)
        return float(np.sum((1 - (1 - p) ** allocation) - p))

    feasible = [
        allocation
        for allocation in itertools.product(range(n_min, n_max + 1), repeat=len(p))
        if sum(allocation) == budget
    ]
    optimum = max(objective(allocation) for allocation in feasible)
    assert objective(greedy) == pytest.approx(optimum, abs=1e-12)
    assert greedy.sum() == budget


@pytest.mark.parametrize("seed", range(12))
def test_greedy_rollout_allocation_matches_random_brute_force(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=3)
    n_min, n_max = 1, 4
    budget = int(rng.integers(len(p) * n_min, len(p) * n_max + 1))
    greedy = allocate_rollouts_greedy(
        p, budget, n_min=n_min, n_max=n_max
    )

    def objective(allocation):
        allocation = np.asarray(allocation)
        return float(np.sum((1 - (1 - p) ** allocation) - p))

    feasible = (
        allocation
        for allocation in itertools.product(
            range(n_min, n_max + 1), repeat=len(p)
        )
        if sum(allocation) == budget
    )
    optimum = max(objective(allocation) for allocation in feasible)
    assert objective(greedy) == pytest.approx(optimum, abs=1e-12)


def test_greedy_rollout_allocation_rejects_infeasible_budget():
    with pytest.raises(ValueError, match="infeasible"):
        allocate_rollouts_greedy(np.array([0.1, 0.2]), 3, n_min=2, n_max=4)
    with pytest.raises(ValueError, match="infeasible"):
        allocate_rollouts_greedy(np.array([0.1, 0.2]), 9, n_min=2, n_max=4)


def test_greedy_rollout_allocation_uses_boundary_probabilities_exactly():
    p = np.array([0.0, 0.2, 1.0])
    allocation = allocate_rollouts_greedy(p, 5, n_min=1, n_max=3)
    assert np.array_equal(allocation, np.array([1, 3, 1]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        allocate_rollouts_greedy(np.array([-0.1, 0.5]), 4, n_min=1, n_max=3)


def test_adaptive_rollout_allocation_preserves_exact_feasible_budget():
    teacher = UniformTeacher(4)
    for stats, p in zip(teacher.stats, [0.9, 0.5, 0.05, 0.0]):
        stats.ema_initialized = True
        stats.ema_pass = p

    allocation = allocate_rollouts_adaptive(
        teacher, np.arange(4), 64, n_min=4, n_max=32
    )
    assert allocation.sum() == 64
    assert allocation.min() >= 4 and allocation.max() <= 32
    assert allocation[3] >= allocation[2] >= allocation[1] >= allocation[0]


@pytest.mark.parametrize("budget", [3, 9])
def test_adaptive_rollout_allocation_rejects_infeasible_budget(budget):
    teacher = UniformTeacher(2)
    with pytest.raises(ValueError, match="infeasible"):
        allocate_rollouts_adaptive(
            teacher, np.arange(2), budget, n_min=2, n_max=4
        )


def test_adaptive_rollout_allocation_validates_empty_and_task_ids():
    teacher = UniformTeacher(2)
    assert allocate_rollouts_adaptive(
        teacher, np.array([], dtype=int), 0
    ).size == 0
    with pytest.raises(ValueError, match="empty"):
        allocate_rollouts_adaptive(
            teacher, np.array([], dtype=int), 1
        )
    with pytest.raises(ValueError, match="task_ids"):
        allocate_rollouts_adaptive(teacher, np.array([2]), 4)
