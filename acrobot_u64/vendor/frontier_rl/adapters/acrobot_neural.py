"""Neural Gymnasium Acrobot adapter for the transfer-control experiment.

The curriculum consists of eight nested, binary post-transition predicates

    tip_height(observation) > threshold

on the unmodified ``Acrobot-v1`` dynamics and 500-step time limit.  The
height expression is exactly the one used by Gymnasium's native terminal
condition at threshold 1.0.  Lower thresholds only change when an episode is
stopped and scored; they do not change the simulator dynamics or reward.

Three policy layouts isolate transfer from parameter count and per-task
capacity:

``shared``
    One task-agnostic H=64 actor (640 parameters total and active).
``disjoint_total_budget``
    Eight disjoint H=8 actors (640 parameters total, 80 active per task).
``disjoint_active_capacity``
    Eight disjoint H=64 actors (5,120 total, 640 active per task).

The actor implements the exact frozen-policy group score gradient.  Score
terms are summed over timesteps and rollouts; there is deliberately no
trajectory-length normalization, gradient clipping, optimizer state, or
output bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from frontier_rl.interfaces import GroupResult

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError("pip install gymnasium[classic-control]") from e


THRESHOLDS = (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0)
MAX_EPISODE_STEPS = 500
OBSERVATION_SCALE = np.array(
    [1.0, 1.0, 1.0, 1.0, 4.0 * np.pi, 9.0 * np.pi], dtype=np.float64
)


def tip_height(observation: Sequence[float]) -> float:
    """Return the Acrobot endpoint height used by the native termination.

    Gymnasium observations are ``[cos(t1), sin(t1), cos(t2), sin(t2),
    t1_dot, t2_dot]``.  Consequently the endpoint height is
    ``-cos(t1) - cos(t1+t2)`` and can be computed without simulator state.
    """

    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (6,):
        raise ValueError(f"Acrobot observation must have shape (6,), got {obs.shape}")
    return float(-obs[0] - (obs[0] * obs[2] - obs[1] * obs[3]))


def normalize_observation(observation: Sequence[float]) -> np.ndarray:
    """Normalize the six official observations by their declared bounds."""

    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (6,):
        raise ValueError(f"Acrobot observation must have shape (6,), got {obs.shape}")
    return obs / OBSERVATION_SCALE


@dataclass(frozen=True)
class AcrobotTransition:
    """One complete simulator transition retained for gradient auditing."""

    obs_before: np.ndarray
    action: int
    obs_after: np.ndarray
    native_reward: float
    native_terminated: bool
    truncated: bool
    height_after: float


class TanhCategoricalActor:
    """One-hidden-layer NumPy categorical actor with exact group updates.

    Parameters are ``W_in`` (6 by H), a hidden bias (H), and ``W_out``
    (H by 3).  There is no output bias, so each slot has exactly ``10*H``
    parameters.  Disjoint modes index a separate slot by curriculum task;
    shared mode always uses slot zero and never observes the task id.
    """

    SHARED = "shared"
    DISJOINT_TOTAL_BUDGET = "disjoint_total_budget"
    DISJOINT_ACTIVE_CAPACITY = "disjoint_active_capacity"

    _ALIASES = {
        "shared": SHARED,
        "shared_h64": SHARED,
        "disjoint_total_budget": DISJOINT_TOTAL_BUDGET,
        "disjoint_h8": DISJOINT_TOTAL_BUDGET,
        "total_budget": DISJOINT_TOTAL_BUDGET,
        "disjoint_active_capacity": DISJOINT_ACTIVE_CAPACITY,
        "disjoint_h64": DISJOINT_ACTIVE_CAPACITY,
        "active_capacity": DISJOINT_ACTIVE_CAPACITY,
    }
    _EXPECTED_HIDDEN = {
        SHARED: 64,
        DISJOINT_TOTAL_BUDGET: 8,
        DISJOINT_ACTIVE_CAPACITY: 64,
    }

    def __init__(
        self,
        n_tasks: int = 8,
        hidden_size: Optional[int] = None,
        learning_rate: float = 0.01,
        seed: int = 0,
        mode: str = SHARED,
        action_seed: Optional[int] = None,
    ):
        try:
            canonical_mode = self._ALIASES[mode]
        except KeyError as e:
            choices = ", ".join(sorted(self._EXPECTED_HIDDEN))
            raise ValueError(f"mode must be one of {choices}, got {mode!r}") from e
        if int(n_tasks) != len(THRESHOLDS):
            raise ValueError("the fixed Acrobot study requires exactly eight tasks")

        expected_hidden = self._EXPECTED_HIDDEN[canonical_mode]
        if hidden_size is None:
            hidden_size = expected_hidden
        if int(hidden_size) != expected_hidden:
            raise ValueError(
                f"mode {canonical_mode!r} requires hidden_size={expected_hidden} "
                "for the registered capacity control"
            )
        if not np.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")

        self.n_tasks = int(n_tasks)
        self.mode = canonical_mode
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.n_slots = 1 if self.mode == self.SHARED else self.n_tasks

        parameter_rng = np.random.default_rng(seed)
        # Independent RNG streams ensure parameter construction never consumes
        # action randomness and evaluation can use its own non-mutating stream.
        self.action_rng = np.random.default_rng(
            seed + 1 if action_seed is None else action_seed
        )
        self.rng = self.action_rng  # compatibility with the classic adapters

        in_scale = 1.0 / np.sqrt(6.0)
        self.W_in = parameter_rng.normal(
            0.0, in_scale, size=(self.n_slots, 6, self.hidden_size)
        )
        self.b_hidden = np.zeros((self.n_slots, self.hidden_size), dtype=float)
        # Zero output weights make the initial action distribution exactly
        # uniform in every architecture while nonzero hidden features let the
        # first group update W_out.  There is still no output bias.
        self.W_out = np.zeros(
            (self.n_slots, self.hidden_size, 3), dtype=float
        )

        self.update_calls = 0
        self.applied_updates = 0
        self.slot_update_calls = np.zeros(self.n_slots, dtype=np.int64)
        self.last_update_stats = {
            "task_id": None,
            "slot": None,
            "gradient_norm": 0.0,
            "update_norm": 0.0,
            "entropy": None,
            "mean_policy_entropy": None,
            "n_trajectories": 0,
            "n_score_terms": 0,
            "n_weighted_score_terms": 0,
            "weight_l1": 0.0,
            "applied": False,
            "frozen_group_parameters": True,
        }

    @property
    def is_shared(self) -> bool:
        return self.mode == self.SHARED

    @property
    def parameter_count(self) -> int:
        return int(self.W_in.size + self.b_hidden.size + self.W_out.size)

    @property
    def active_parameter_count(self) -> int:
        return int(10 * self.hidden_size)

    @property
    def update_count(self) -> int:
        """Number of nonzero parameter updates (diagnostic compatibility)."""

        return self.applied_updates

    def _slot(self, task_id: int) -> int:
        task_id = int(task_id)
        if task_id < 0 or task_id >= self.n_tasks:
            raise IndexError(f"task_id {task_id} outside [0, {self.n_tasks})")
        return 0 if self.is_shared else task_id

    def parameter_vector(self) -> np.ndarray:
        """Return a copy of all trainable parameters in a stable order."""

        return np.concatenate(
            [self.W_in.ravel(), self.b_hidden.ravel(), self.W_out.ravel()]
        ).copy()

    def slot_parameter_vector(self, task_id: int) -> np.ndarray:
        """Return a copy of the active slot's parameters in a stable order."""

        slot = self._slot(task_id)
        return np.concatenate(
            [
                self.W_in[slot].ravel(),
                self.b_hidden[slot].ravel(),
                self.W_out[slot].ravel(),
            ]
        ).copy()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def _forward_with_parameters(
        self,
        normalized_observation: Sequence[float],
        W_in: np.ndarray,
        b_hidden: np.ndarray,
        W_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(normalized_observation, dtype=np.float64)
        if x.shape != (6,):
            raise ValueError(f"actor input must have shape (6,), got {x.shape}")
        hidden = np.tanh(x @ W_in + b_hidden)
        return hidden, self._softmax(hidden @ W_out)

    def probabilities(
        self, normalized_observation: Sequence[float], task_id: int = 0
    ) -> np.ndarray:
        slot = self._slot(task_id)
        _, probabilities = self._forward_with_parameters(
            normalized_observation,
            self.W_in[slot],
            self.b_hidden[slot],
            self.W_out[slot],
        )
        return probabilities

    def act(
        self,
        normalized_observation: Sequence[float],
        task_id: int = 0,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Sample an action, optionally from an external non-mutating RNG."""

        probabilities = self.probabilities(normalized_observation, task_id)
        generator = self.action_rng if rng is None else rng
        return int(generator.choice(3, p=probabilities))

    @staticmethod
    def _trajectory_step(step) -> tuple[np.ndarray, int]:
        if isinstance(step, AcrobotTransition):
            return normalize_observation(step.obs_before), int(step.action)
        if isinstance(step, Mapping):
            if "normalized_observation" in step:
                obs = step["normalized_observation"]
            elif "obs_before" in step:
                obs = normalize_observation(step["obs_before"])
            else:
                raise ValueError("mapping transition lacks an observation")
            return np.asarray(obs, dtype=float), int(step["action"])
        if len(step) != 2:
            raise ValueError("actor tuple steps must be (normalized_observation, action)")
        return np.asarray(step[0], dtype=float), int(step[1])

    def _group_gradient(
        self, task_id: int, trajectories: Sequence, weights: Sequence[float]
    ) -> tuple[dict[str, np.ndarray], dict[str, float | int | None]]:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (len(trajectories),):
            raise ValueError(
                "weights must be a one-dimensional array matching trajectories"
            )
        if not np.all(np.isfinite(weights_array)):
            raise ValueError("weights must be finite")

        slot = self._slot(task_id)
        # Explicit copies make the frozen-group contract mechanically clear.
        W_in = self.W_in[slot].copy()
        b_hidden = self.b_hidden[slot].copy()
        W_out = self.W_out[slot].copy()
        grad_in = np.zeros_like(W_in)
        grad_hidden = np.zeros_like(b_hidden)
        grad_out = np.zeros_like(W_out)
        entropy_sum = 0.0
        score_terms = 0
        weighted_score_terms = 0

        for trajectory, weight in zip(trajectories, weights_array):
            for step in trajectory:
                x, action = self._trajectory_step(step)
                if x.shape != (6,):
                    raise ValueError(f"actor input must have shape (6,), got {x.shape}")
                if action < 0 or action >= 3:
                    raise ValueError(f"Acrobot action must be 0, 1, or 2; got {action}")
                hidden, probabilities = self._forward_with_parameters(
                    x, W_in, b_hidden, W_out
                )
                entropy_sum -= float(
                    np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
                )
                score_terms += 1
                if weight == 0.0:
                    continue
                output_score = -probabilities
                output_score = output_score.copy()
                output_score[action] += 1.0
                hidden_score = (W_out @ output_score) * (1.0 - hidden * hidden)
                # Sum every score term exactly; do not divide by trajectory
                # length, rollout count, or total weight.
                grad_out += weight * np.outer(hidden, output_score)
                grad_in += weight * np.outer(x, hidden_score)
                grad_hidden += weight * hidden_score
                weighted_score_terms += 1

        mean_entropy = entropy_sum / score_terms if score_terms else None
        stats = {
            "slot": slot,
            "n_trajectories": len(trajectories),
            "n_score_terms": score_terms,
            "n_weighted_score_terms": weighted_score_terms,
            "weight_l1": float(np.sum(np.abs(weights_array))),
            "entropy": mean_entropy,
            "mean_policy_entropy": mean_entropy,
        }
        return {
            "W_in": grad_in,
            "b_hidden": grad_hidden,
            "W_out": grad_out,
        }, stats

    def group_gradient(
        self, task_id: int, trajectories: Sequence, weights: Sequence[float]
    ) -> dict[str, np.ndarray]:
        """Return the exact frozen-policy summed score gradient without mutation."""

        gradient, _ = self._group_gradient(task_id, trajectories, weights)
        return {name: value.copy() for name, value in gradient.items()}

    def gradient_diagnostics(
        self, task_id: int, trajectories: Sequence, weights: Sequence[float]
    ) -> dict:
        """Inspect an unscaled auxiliary gradient without changing any state.

        This is used by the scale-zero control: eligible hindsight groups are
        still constructed and measured, but neither parameters nor update
        counters (nor the action RNG) are touched.
        """

        gradient, stats = self._group_gradient(task_id, trajectories, weights)
        gradient_norm = float(
            np.sqrt(sum(np.sum(value * value) for value in gradient.values()))
        )
        return {
            "task_id": int(task_id),
            **stats,
            "gradient_norm": gradient_norm,
            "hypothetical_update_norm": self.learning_rate * gradient_norm,
            "frozen_group_parameters": True,
            "mutated": False,
        }

    def update(
        self, task_id: int, trajectories: Sequence, weights: Sequence[float]
    ) -> None:
        """Apply one plain-SGD-ascent update from a frozen rollout group."""

        gradient, stats = self._group_gradient(task_id, trajectories, weights)
        slot = int(stats["slot"])
        gradient_norm = float(
            np.sqrt(sum(np.sum(value * value) for value in gradient.values()))
        )
        update_norm = self.learning_rate * gradient_norm

        self.W_in[slot] += self.learning_rate * gradient["W_in"]
        self.b_hidden[slot] += self.learning_rate * gradient["b_hidden"]
        self.W_out[slot] += self.learning_rate * gradient["W_out"]

        self.update_calls += 1
        self.slot_update_calls[slot] += 1
        applied = bool(update_norm != 0.0)
        if applied:
            self.applied_updates += 1
        self.last_update_stats = {
            "task_id": int(task_id),
            **stats,
            "gradient_norm": gradient_norm,
            "update_norm": update_norm,
            "applied": applied,
            "frozen_group_parameters": True,
        }

    def diagnostics(self) -> dict:
        """Return JSON-friendly parameter, capacity, RNG, and update facts."""

        parameters = self.parameter_vector()
        return {
            "mode": self.mode,
            "hidden_size": self.hidden_size,
            "n_slots": self.n_slots,
            "parameter_count": self.parameter_count,
            "active_parameter_count": self.active_parameter_count,
            "parameter_norm": float(np.linalg.norm(parameters)),
            "update_calls": int(self.update_calls),
            "applied_updates": int(self.applied_updates),
            "slot_update_calls": self.slot_update_calls.astype(int).tolist(),
            "last_update_stats": dict(self.last_update_stats),
        }


class AcrobotNeuralSpace:
    """Eight nested height tasks on the official ``Acrobot-v1`` dynamics."""

    def __init__(
        self,
        actor: Optional[TanhCategoricalActor] = None,
        thresholds: Sequence[float] = THRESHOLDS,
        seed: int = 0,
        mode: str = TanhCategoricalActor.SHARED,
        learning_rate: float = 0.01,
    ):
        supplied_thresholds = np.asarray(thresholds, dtype=np.float64)
        registered_thresholds = np.asarray(THRESHOLDS, dtype=np.float64)
        if supplied_thresholds.shape != registered_thresholds.shape or not np.array_equal(
            supplied_thresholds, registered_thresholds
        ):
            raise ValueError(
                "this registered experiment uses the fixed thresholds "
                f"{list(THRESHOLDS)}"
            )

        self.thresholds = supplied_thresholds.copy()
        self.thresholds.flags.writeable = False
        self._n_tasks = len(self.thresholds)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 10_003)
        self.actor = actor or TanhCategoricalActor(
            n_tasks=self._n_tasks,
            learning_rate=learning_rate,
            seed=self.seed,
            mode=mode,
        )
        if self.actor.n_tasks != self._n_tasks:
            raise ValueError("actor and threshold task counts must match")
        self.policy = self.actor

        self.env = gym.make("Acrobot-v1")
        if self.env.spec is None or self.env.spec.max_episode_steps != MAX_EPISODE_STEPS:
            self.env.close()
            raise RuntimeError("Acrobot-v1 must use Gymnasium's 500-step time limit")

    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @staticmethod
    def height(observation: Sequence[float]) -> float:
        return tip_height(observation)

    def verify_height(self, task_id: int, height: float) -> bool:
        task_id = int(task_id)
        if task_id < 0 or task_id >= self._n_tasks:
            raise IndexError(f"task_id {task_id} outside [0, {self._n_tasks})")
        return bool(float(height) > self.thresholds[task_id])

    def verify_observation(self, task_id: int, observation: Sequence[float]) -> bool:
        return self.verify_height(task_id, tip_height(observation))

    @staticmethod
    def native_success(observation: Sequence[float]) -> bool:
        """The exact strict Gymnasium terminal predicate."""

        return bool(tip_height(observation) > 1.0)

    def _episode(self, task_id: int) -> tuple[list[AcrobotTransition], dict]:
        task_id = int(task_id)
        if task_id < 0 or task_id >= self._n_tasks:
            raise IndexError(f"task_id {task_id} outside [0, {self._n_tasks})")
        reset_seed = int(self.rng.integers(0, 2**31 - 1))
        observation, _ = self.env.reset(seed=reset_seed)
        trajectory: list[AcrobotTransition] = []
        heights_after: list[float] = []
        native_return = 0.0
        threshold_success = False
        native_terminated = False
        truncated = False
        time_to_goal = None

        for step_index in range(1, MAX_EPISODE_STEPS + 1):
            action = self.actor.act(normalize_observation(observation), task_id)
            next_observation, reward, terminated, was_truncated, _ = self.env.step(action)
            height_after = tip_height(next_observation)
            transition = AcrobotTransition(
                obs_before=np.asarray(observation, dtype=np.float64).copy(),
                action=action,
                obs_after=np.asarray(next_observation, dtype=np.float64).copy(),
                native_reward=float(reward),
                native_terminated=bool(terminated),
                truncated=bool(was_truncated),
                height_after=height_after,
            )
            trajectory.append(transition)
            heights_after.append(height_after)
            native_return += float(reward)
            observation = next_observation
            native_terminated = bool(terminated)
            truncated = bool(was_truncated)

            # Success is checked strictly on the post-transition observation,
            # before considering the time-limit flag, and the trace stops at
            # its first verified crossing.
            if self.verify_height(task_id, height_after):
                threshold_success = True
                time_to_goal = step_index
                break
            if terminated or was_truncated:
                break

        info = {
            "n_steps": len(trajectory),
            "reset_seed": reset_seed,
            "threshold": float(self.thresholds[task_id]),
            "threshold_success": threshold_success,
            "time_to_goal": time_to_goal,
            "max_height": float(max(heights_after)),
            "heights_after": heights_after,
            "native_terminated": native_terminated,
            "truncated": truncated,
            # For lower thresholds this is the native reward accumulated only
            # until the registered first-hit stopping rule.
            "native_return": float(native_return),
        }
        return trajectory, info

    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        if int(n_rollouts) <= 0:
            raise ValueError("n_rollouts must be positive")
        trajectories, rewards, infos = [], [], []
        for _ in range(int(n_rollouts)):
            trajectory, info = self._episode(task_id)
            trajectories.append(trajectory)
            rewards.append(float(info["threshold_success"]))
            infos.append(info)
        return GroupResult(
            task_id=int(task_id),
            rewards=np.asarray(rewards, dtype=np.float64),
            trajectories=trajectories,
            infos=infos,
        )

    @staticmethod
    def _verified_heights(trajectory: Sequence) -> Optional[list[float]]:
        """Recompute heights from stored post-transition observations."""

        heights = []
        for transition in trajectory:
            if isinstance(transition, AcrobotTransition):
                observation = transition.obs_after
            elif isinstance(transition, Mapping) and "obs_after" in transition:
                observation = transition["obs_after"]
            else:
                return None
            heights.append(tip_height(observation))
        return heights

    def relabel(self, group: GroupResult):
        """Relabel a dead shared-policy group to its hardest mixed lower task.

        Centered MaxRL is nonzero only when ``0 < K < N``.  Starting just
        below the requested threshold, this method selects the hardest lower
        predicate with such a mixed outcome.  Every reward is recomputed from
        the stored post-transition observation.  Successful traces are cut at
        their first strict crossing, exactly matching the live-task stopping
        rule.  Disjoint actors fail closed because their task conditioning
        cannot be rewritten into shared-policy trajectories.
        """

        if not self.actor.is_shared:
            return None
        original_rewards = np.asarray(group.rewards, dtype=float)
        if original_rewards.ndim != 1 or len(original_rewards) != len(group.trajectories):
            return None
        if len(original_rewards) == 0 or np.any(original_rewards != 0.0):
            return None
        original_task = int(group.task_id)
        if original_task <= 0 or original_task >= self._n_tasks:
            return None

        per_trajectory_heights = [
            self._verified_heights(trajectory) for trajectory in group.trajectories
        ]
        if any(heights is None or len(heights) == 0 for heights in per_trajectory_heights):
            return None

        for candidate in range(original_task - 1, -1, -1):
            threshold = float(self.thresholds[candidate])
            rewards = np.asarray(
                [float(any(height > threshold for height in heights))
                 for heights in per_trajectory_heights],
                dtype=np.float64,
            )
            successes = int(rewards.sum())
            if not (0 < successes < len(rewards)):
                continue

            rewritten = []
            for trajectory, heights, reward in zip(
                group.trajectories, per_trajectory_heights, rewards
            ):
                if reward == 1.0:
                    first_hit = next(
                        index for index, height in enumerate(heights)
                        if height > threshold
                    )
                    rewritten.append(list(trajectory[: first_hit + 1]))
                else:
                    rewritten.append(list(trajectory))
            return candidate, rewards, rewritten
        return None

    def update(
        self, task_id: int, trajectories: Sequence, weights: Sequence[float]
    ) -> None:
        self.actor.update(task_id, trajectories, weights)

    def _evaluate_slot(
        self,
        env,
        task_id: int,
        episode_seeds: np.ndarray,
        action_seed: int,
    ) -> dict[str, np.ndarray]:
        # Give every episode its own action stream. A single stream for the
        # whole batch loses common random numbers as soon as two policies have
        # different episode lengths: later episodes then start at different
        # positions in that stream. Pre-generating episode seeds keeps each
        # reset/action pair aligned across checkpoints and conditions.
        action_seed_rng = np.random.default_rng(action_seed)
        episode_action_seeds = action_seed_rng.integers(
            0, 2**63 - 1, size=len(episode_seeds), dtype=np.int64
        )
        maxima, successes, returns, times, policy_entropies = [], [], [], [], []
        for reset_seed, episode_action_seed in zip(
            episode_seeds, episode_action_seeds
        ):
            action_rng = np.random.default_rng(int(episode_action_seed))
            observation, _ = env.reset(seed=int(reset_seed))
            max_height = -np.inf
            native_return = 0.0
            native_success = False
            time_to_goal = MAX_EPISODE_STEPS  # right-censor failures at 500
            for step_index in range(1, MAX_EPISODE_STEPS + 1):
                probabilities = self.actor.probabilities(
                    normalize_observation(observation), task_id
                )
                policy_entropies.append(
                    -float(
                        np.sum(
                            probabilities
                            * np.log(np.maximum(probabilities, 1e-300))
                        )
                    )
                )
                action = int(action_rng.choice(3, p=probabilities))
                observation, reward, terminated, truncated, _ = env.step(action)
                max_height = max(max_height, tip_height(observation))
                native_return += float(reward)
                if terminated:
                    native_success = True
                    time_to_goal = step_index
                    break
                if truncated:
                    break
            maxima.append(max_height)
            successes.append(float(native_success))
            returns.append(native_return)
            times.append(float(time_to_goal))
        return {
            "max_heights": np.asarray(maxima, dtype=float),
            "native_successes": np.asarray(successes, dtype=float),
            "native_returns": np.asarray(returns, dtype=float),
            "censored_times": np.asarray(times, dtype=float),
            "mean_policy_entropy": float(np.mean(policy_entropies)),
        }

    def evaluate(self, n: int = 32, seed: Optional[int] = None) -> dict:
        """Evaluate with a fresh env and local fixed RNG streams.

        Training reset state, action RNG state, parameters, and update
        diagnostics are untouched.  ``mean_time_to_goal`` is the restricted
        mean with native failures right-censored at 500 steps; the
        success-only mean is reported separately and is ``None`` when there
        are no native successes.
        """

        if int(n) <= 0:
            raise ValueError("n must be positive")
        evaluation_seed = self.seed + 1_000_003 if seed is None else int(seed)
        seed_rng = np.random.default_rng(evaluation_seed)
        episode_seeds = seed_rng.integers(0, 2**31 - 1, size=int(n))
        action_seed = evaluation_seed + 1
        eval_env = gym.make("Acrobot-v1")
        try:
            if self.actor.is_shared:
                hardest = self._evaluate_slot(
                    eval_env, 0, episode_seeds, action_seed
                )
                pass_rates = [
                    float(np.mean(hardest["max_heights"] > threshold))
                    for threshold in self.thresholds
                ]
            else:
                pass_rates = []
                hardest = None
                for task_id, threshold in enumerate(self.thresholds):
                    batch = self._evaluate_slot(
                        eval_env, task_id, episode_seeds, action_seed
                    )
                    pass_rates.append(
                        float(np.mean(batch["max_heights"] > threshold))
                    )
                    if task_id == self._n_tasks - 1:
                        hardest = batch
                assert hardest is not None
        finally:
            eval_env.close()

        successes = hardest["native_successes"]
        successful_times = hardest["censored_times"][successes == 1.0]
        mean_success_time = (
            float(np.mean(successful_times)) if len(successful_times) else None
        )
        return {
            "thresholds": self.thresholds.astype(float).tolist(),
            "pass_rates": pass_rates,
            "native_success_rate": float(np.mean(successes)),
            "mean_native_return": float(np.mean(hardest["native_returns"])),
            "mean_time_to_goal": float(np.mean(hardest["censored_times"])),
            "mean_success_time_to_goal": mean_success_time,
            "mean_policy_entropy": float(hardest["mean_policy_entropy"]),
            "time_to_goal_censoring": "native failures assigned 500 steps",
            "episodes_per_task": int(n),
            "evaluation_seed": evaluation_seed,
        }

    def eval_pass_rates(self, n: int = 32, seed: Optional[int] = None) -> np.ndarray:
        return np.asarray(self.evaluate(n=n, seed=seed)["pass_rates"], dtype=float)

    def diagnostics(self) -> dict:
        return {
            "environment": "Acrobot-v1",
            "max_episode_steps": MAX_EPISODE_STEPS,
            "thresholds": self.thresholds.astype(float).tolist(),
            "policy": self.actor.diagnostics(),
        }

    def close(self) -> None:
        self.env.close()
