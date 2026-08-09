"""Contract tests for the neural Acrobot transfer-control adapter."""

from __future__ import annotations

import copy

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from frontier_rl.adapters.acrobot_neural import (
    AcrobotNeuralSpace,
    AcrobotTransition,
    TanhCategoricalActor,
    normalize_observation,
    tip_height,
)
from frontier_rl.interfaces import GroupResult


def _post_observation_for_height(height: float) -> np.ndarray:
    # With cos(theta2)=1 and sin(theta2)=0, h=-2*cos(theta1).
    cosine = -float(height) / 2.0
    sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    return np.array([cosine, sine, 1.0, 0.0, 0.0, 0.0], dtype=float)


def _transition(height: float, action: int = 1) -> AcrobotTransition:
    return AcrobotTransition(
        obs_before=np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        action=action,
        obs_after=_post_observation_for_height(height),
        native_reward=-1.0,
        native_terminated=height > 1.0,
        truncated=False,
        height_after=float(height),
    )


def _actor(mode: str, seed: int = 4, learning_rate: float = 0.03):
    return TanhCategoricalActor(
        n_tasks=8,
        learning_rate=learning_rate,
        seed=seed,
        mode=mode,
    )


def test_height_known_states_and_matches_native_terminal_expression():
    assert tip_height([1, 0, 1, 0, 0, 0]) == pytest.approx(-2.0)
    assert tip_height([-1, 0, 1, 0, 0, 0]) == pytest.approx(2.0)
    assert tip_height([0, 1, 1, 0, 0, 0]) == pytest.approx(0.0)

    space = AcrobotNeuralSpace(seed=0)
    native = gym.make("Acrobot-v1")
    try:
        for state in (
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([np.pi, 0.0, 0.0, 0.0]),
            np.array([2.4, -0.3, 0.0, 0.0]),
        ):
            native.unwrapped.state = state.copy()
            observation = native.unwrapped._get_ob()
            assert space.native_success(observation) == native.unwrapped._terminal()

        # Both the curriculum threshold and native predicate are strict.
        assert not space.verify_height(7, 1.0)
        assert space.verify_height(7, np.nextafter(1.0, np.inf))
    finally:
        native.close()
        space.close()


def test_parameter_counts_and_disjoint_slot_isolation():
    expected = {
        "shared": (640, 640),
        "disjoint_total_budget": (640, 80),
        "disjoint_active_capacity": (5120, 640),
    }
    trajectory = [[(np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2]), 2)]]
    for mode, (total, active) in expected.items():
        actor = _actor(mode)
        assert actor.parameter_count == total
        assert actor.active_parameter_count == active
        assert not hasattr(actor, "b_out")  # the count excludes output bias

        before = [actor.slot_parameter_vector(task) for task in range(8)]
        actor.update(3, trajectory, np.array([1.0]))
        after = [actor.slot_parameter_vector(task) for task in range(8)]
        if mode == "shared":
            assert not np.array_equal(before[0], after[0])
            assert all(np.array_equal(after[0], after[task]) for task in range(8))
        else:
            assert not np.array_equal(before[3], after[3])
            for task in range(8):
                if task != 3:
                    assert np.array_equal(before[task], after[task])


def test_zero_frozen_summed_and_permutation_invariant_group_update():
    x0 = np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2])
    x1 = np.array([-0.4, 0.2, 0.1, 0.3, -0.2, 0.1])
    trajectories = [
        [(x0, 2), (x1, 0)],
        [(x1, 1)],
        [(x0, 0), (x0, 2), (x1, 1)],
    ]
    weights = np.array([0.7, -0.2, -0.5])

    zero = _actor("shared")
    zero_before = zero.parameter_vector()
    action_state = copy.deepcopy(zero.action_rng.bit_generator.state)
    last_stats = copy.deepcopy(zero.last_update_stats)
    diagnostics = zero.gradient_diagnostics(2, trajectories, weights)
    assert np.array_equal(zero.parameter_vector(), zero_before)
    assert zero.action_rng.bit_generator.state == action_state
    assert zero.last_update_stats == last_stats
    assert diagnostics["mutated"] is False

    zero.update(2, trajectories, np.zeros(3))
    assert np.array_equal(zero.parameter_vector(), zero_before)
    assert zero.last_update_stats["gradient_norm"] == 0.0
    assert zero.last_update_stats["update_norm"] == 0.0

    frozen = _actor("shared")
    before_in = frozen.W_in[0].copy()
    before_hidden = frozen.b_hidden[0].copy()
    before_out = frozen.W_out[0].copy()
    gradient = frozen.group_gradient(2, trajectories, weights)
    frozen.update(2, trajectories, weights)
    assert np.allclose(
        frozen.W_in[0], before_in + frozen.learning_rate * gradient["W_in"]
    )
    assert np.allclose(
        frozen.b_hidden[0],
        before_hidden + frozen.learning_rate * gradient["b_hidden"],
    )
    assert np.allclose(
        frozen.W_out[0], before_out + frozen.learning_rate * gradient["W_out"]
    )
    assert frozen.last_update_stats["n_score_terms"] == 6
    assert frozen.last_update_stats["frozen_group_parameters"] is True

    # A repeated timestep contributes twice exactly: there is no trajectory-
    # length normalization hidden inside the actor.
    once = _actor("shared")
    twice = _actor("shared")
    once_before = once.parameter_vector()
    twice_before = twice.parameter_vector()
    once.update(0, [[(x0, 2)]], [1.0])
    twice.update(0, [[(x0, 2), (x0, 2)]], [1.0])
    assert np.allclose(
        twice.parameter_vector() - twice_before,
        2.0 * (once.parameter_vector() - once_before),
    )

    ordered = _actor("shared")
    permuted = _actor("shared")
    ordered.update(1, trajectories, weights)
    order = np.array([2, 0, 1])
    permuted.update(
        1, [trajectories[index] for index in order], weights[order]
    )
    assert np.allclose(ordered.parameter_vector(), permuted.parameter_vector())


def test_relabel_is_verifier_valid_mixed_hardest_and_first_hit():
    actor = _actor("shared")
    space = AcrobotNeuralSpace(actor=actor, seed=3)
    trajectories = [
        [_transition(0.5), _transition(0.7), _transition(0.71), _transition(0.9)],
        [_transition(0.2), _transition(0.65)],
        [_transition(-0.4), _transition(0.69), _transition(0.1)],
    ]
    group = GroupResult(
        task_id=7,
        rewards=np.zeros(3),
        trajectories=trajectories,
        infos=[{} for _ in trajectories],
    )
    try:
        relabeled = space.relabel(group)
        assert relabeled is not None
        task, rewards, rewritten = relabeled
        assert task == 6  # hardest lower threshold, h > 0.7
        assert np.array_equal(rewards, np.array([1.0, 0.0, 0.0]))
        assert 0 < rewards.sum() < len(rewards)
        assert len(rewritten[0]) == 3  # equality at 0.7 is not a hit
        assert len(rewritten[1]) == len(trajectories[1])
        assert len(rewritten[2]) == len(trajectories[2])
        for reward, trajectory in zip(rewards, rewritten):
            heights = [tip_height(step.obs_after) for step in trajectory]
            assert bool(reward) == any(height > space.thresholds[task]
                                       for height in heights)
            if reward:
                assert heights[-1] > space.thresholds[task]
                assert all(height <= space.thresholds[task]
                           for height in heights[:-1])

        disjoint_space = AcrobotNeuralSpace(
            actor=_actor("disjoint_active_capacity"), seed=3
        )
        try:
            assert disjoint_space.relabel(group) is None
        finally:
            disjoint_space.close()
    finally:
        space.close()


def test_evaluation_uses_fresh_fixed_rng_and_preserves_training_sequence():
    evaluated = AcrobotNeuralSpace(actor=_actor("shared", seed=11), seed=11)
    control = AcrobotNeuralSpace(actor=_actor("shared", seed=11), seed=11)
    try:
        train_rng_state = copy.deepcopy(evaluated.rng.bit_generator.state)
        action_rng_state = copy.deepcopy(evaluated.actor.action_rng.bit_generator.state)
        parameters = evaluated.actor.parameter_vector()
        diagnostics = copy.deepcopy(evaluated.actor.last_update_stats)

        first = evaluated.evaluate(n=2, seed=9123)
        second = evaluated.evaluate(n=2, seed=9123)
        assert first == second
        assert len(first["pass_rates"]) == 8
        assert first["native_success_rate"] == first["pass_rates"][-1]
        assert first["time_to_goal_censoring"] == "native failures assigned 500 steps"
        assert evaluated.rng.bit_generator.state == train_rng_state
        assert evaluated.actor.action_rng.bit_generator.state == action_rng_state
        assert np.array_equal(evaluated.actor.parameter_vector(), parameters)
        assert evaluated.actor.last_update_stats == diagnostics

        group_after_eval = evaluated.rollout_group(2, 2)
        group_control = control.rollout_group(2, 2)
        assert np.array_equal(group_after_eval.rewards, group_control.rewards)
        for trajectory_a, trajectory_b in zip(
            group_after_eval.trajectories, group_control.trajectories
        ):
            assert [step.action for step in trajectory_a] == [
                step.action for step in trajectory_b
            ]
            assert np.allclose(
                [step.height_after for step in trajectory_a],
                [step.height_after for step in trajectory_b],
            )
    finally:
        evaluated.close()
        control.close()


def test_evaluation_action_rng_is_aligned_per_episode():
    """A longer prior episode must not shift the next episode's action RNG."""

    class ConstantActor:
        def probabilities(self, observation, task_id):
            return np.array([0.2, 0.3, 0.5], dtype=float)

    class LengthControlledEnv:
        def __init__(self, episode_lengths):
            self.episode_lengths = list(episode_lengths)
            self.episode = -1
            self.step_in_episode = 0
            self.actions = []

        def reset(self, seed):
            self.episode += 1
            self.step_in_episode = 0
            self.actions.append([])
            return np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), {}

        def step(self, action):
            self.actions[self.episode].append(int(action))
            self.step_in_episode += 1
            terminated = self.step_in_episode >= self.episode_lengths[self.episode]
            observation = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            return observation, -1.0, terminated, False, {}

    space = object.__new__(AcrobotNeuralSpace)
    space.actor = ConstantActor()
    episode_seeds = np.array([101, 202], dtype=np.int64)

    # Several streams make this reliably distinguish episode-aligned random
    # numbers from one batch-global stream.
    for action_seed in range(20):
        short_first = LengthControlledEnv([1, 1])
        long_first = LengthControlledEnv([2, 1])
        space._evaluate_slot(short_first, 0, episode_seeds, action_seed)
        space._evaluate_slot(long_first, 0, episode_seeds, action_seed)
        assert short_first.actions[1][0] == long_first.actions[1][0]
