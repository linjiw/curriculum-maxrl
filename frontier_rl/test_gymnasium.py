"""Optional Gymnasium regression tests.

Run with an environment that has ``gymnasium[classic-control]`` installed:

    python3 frontier_rl/test_gymnasium.py
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np

try:
    import gymnasium  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        print("SKIPPED: install requirements-gym.txt (Python >=3.10)")
        raise SystemExit(0)
    import pytest
    pytest.skip("optional Gymnasium dependency", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from frontier_rl.adapters.gym_classic import MountainCarSpace
from frontier_rl.interfaces import GroupResult


def test_shared_policy_really_shares_tiles():
    shared = MountainCarSpace(seed=0, share_policy_across_tasks=True)
    separate = MountainCarSpace(seed=0, share_policy_across_tasks=False)
    obs = np.array([-0.5, 0.0])
    try:
        assert shared.policy.tile(obs, 0) == shared.policy.tile(obs, 9)
        assert separate.policy.tile(obs, 0) != separate.policy.tile(obs, 9)
    finally:
        shared.close()
        separate.close()


def test_evaluation_preserves_training_randomness():
    env = MountainCarSpace(seed=7, share_policy_across_tasks=True)
    env_state = copy.deepcopy(env.rng.bit_generator.state)
    policy_state = copy.deepcopy(env.policy.rng.bit_generator.state)
    try:
        rates_1 = env.eval_pass_rates(n=2, seed=123)
        assert env.rng.bit_generator.state == env_state
        assert env.policy.rng.bit_generator.state == policy_state
        rates_2 = env.eval_pass_rates(n=2, seed=123)
        assert np.array_equal(rates_1, rates_2)
    finally:
        env.close()


def test_group_update_is_permutation_invariant():
    env_a = MountainCarSpace(seed=3, share_policy_across_tasks=True)
    env_b = MountainCarSpace(seed=3, share_policy_across_tasks=True)
    trajectories = [
        [(np.array([-0.50, 0.00]), 0), (np.array([-0.49, 0.01]), 2)],
        [(np.array([-0.50, 0.00]), 2), (np.array([-0.51, -0.01]), 0)],
        [(np.array([-0.48, 0.01]), 1)],
    ]
    weights = np.array([0.5, -0.25, -0.25])
    try:
        env_a.update(0, trajectories, weights)
        order = np.array([2, 0, 1])
        env_b.update(0, [trajectories[i] for i in order], weights[order])
        assert np.allclose(env_a.policy.theta, env_b.policy.theta)
    finally:
        env_a.close()
        env_b.close()


def test_mountaincar_relabel_truncates_at_first_credited_crossing():
    env = MountainCarSpace(seed=0, share_policy_across_tasks=True)
    observations = [np.array([-0.5 + 0.01 * i, 0.0]) for i in range(5)]
    trajectories = [[(obs, 2) for obs in observations] for _ in range(2)]
    # best=4 (target ~=0.028); rollout 0 first crosses it on step 2 and then
    # continues, while rollout 1 never reaches it and remains a failure.
    infos = [
        {"max_x": 0.08, "x_after": [-0.4, 0.05, 0.02, 0.08, 0.07],
         "n_steps": 5},
        {"max_x": 0.02, "x_after": [-0.4, -0.2, -0.1, 0.01, 0.02],
         "n_steps": 5},
    ]
    group = GroupResult(9, np.zeros(2), trajectories, infos)
    try:
        task, rewards, relabeled = env.relabel(group)
        assert task == 4
        assert np.array_equal(rewards, np.array([1.0, 0.0]))
        assert len(relabeled[0]) == 2 and len(relabeled[1]) == 5
    finally:
        env.close()


if __name__ == "__main__":
    test_shared_policy_really_shares_tiles()
    test_evaluation_preserves_training_randomness()
    test_group_update_is_permutation_invariant()
    test_mountaincar_relabel_truncates_at_first_credited_crossing()
    print("Gymnasium adapter tests passed")
