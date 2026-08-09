"""Independent neural MountainCar core for the V1 transfer study.

The study uses the unmodified Gymnasium ``MountainCar-v0`` dynamics and eight
nested post-transition success predicates.  Policies never receive a target
or task identifier in shared mode.  This module deliberately does not depend
on the older MountainCar adapter or the Acrobot experiment engine.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
except ImportError as error:  # pragma: no cover - optional dependency
    raise ImportError("install gymnasium[classic-control] to run this study") from error


THRESHOLDS = (-0.375, -0.250, -0.125, 0.0, 0.125, 0.250, 0.375, 0.500)
N_TASKS = 8
N_ACTIONS = 3
N_ROLLOUTS = 16
MAX_EPISODE_STEPS = 200
TRAINING_ACTION_SEED_OFFSET = 1_000_000
TRAINING_EPISODE_SEED_OFFSET = 2_000_000
EVALUATION_ACTION_SEED_OFFSET = 1_000_000
REGISTERED_DEVELOPMENT_SEEDS = (17_000, 17_001, 17_002)
RESERVED_CONFIRMATORY_SEEDS = tuple(range(18_000, 18_020))
OBSERVATION_LOW = np.array([-1.2, -0.07], dtype=np.float64)
OBSERVATION_HIGH = np.array([0.6, 0.07], dtype=np.float64)
_AUTHORIZATION_NONCE = object()


@dataclass(frozen=True)
class _RegisteredSeedAuthorization:
    seed: int
    development_lock_sha256: str
    nonce: object


def _issue_registered_seed_authorization(
    *, seed: int, development_lock_sha256: str
) -> _RegisteredSeedAuthorization:
    """Issue the internal capability used after runner-side lock validation."""

    if type(seed) is not int or seed not in REGISTERED_DEVELOPMENT_SEEDS:
        raise ValueError("authorization is restricted to registered development seeds")
    if (
        not isinstance(development_lock_sha256, str)
        or len(development_lock_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in development_lock_sha256
        )
    ):
        raise ValueError("development lock SHA-256 is malformed")
    return _RegisteredSeedAuthorization(
        seed=seed,
        development_lock_sha256=development_lock_sha256,
        nonce=_AUTHORIZATION_NONCE,
    )


def _state_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: (
            item.tolist() if isinstance(item, np.ndarray) else int(item)
        ),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_observation(observation: Sequence[float]) -> np.ndarray:
    """Map the official two-dimensional observation box to ``[-1, 1]^2``."""

    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (2,):
        raise ValueError(
            f"MountainCar observation must have shape (2,), got {obs.shape}"
        )
    return 2.0 * (obs - OBSERVATION_LOW) / (OBSERVATION_HIGH - OBSERVATION_LOW) - 1.0


def practical_maxrl_weights(rewards: Sequence[float]) -> np.ndarray:
    """Return the dropped-group practical-MaxRL coefficients for ``N=16``.

    For binary rewards and ``K=sum rewards``, the coefficients are
    ``1{K>0}(r_i/K - 1/N)``.  All-fail and all-pass groups therefore produce
    no parameter update.
    """

    values = np.asarray(rewards, dtype=np.float64)
    if values.shape != (N_ROLLOUTS,):
        raise ValueError(f"registered groups contain exactly {N_ROLLOUTS} rewards")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("MaxRL rewards must be binary")
    successes = float(values.sum())
    if successes == 0.0:
        return np.zeros(N_ROLLOUTS, dtype=np.float64)
    return values / successes - 1.0 / N_ROLLOUTS


@dataclass(frozen=True)
class MountainCarTransition:
    obs_before: np.ndarray
    action: int
    obs_after: np.ndarray
    native_reward: float
    native_terminated: bool
    truncated: bool


@dataclass(frozen=True)
class MountainCarGroup:
    task_id: int
    rewards: np.ndarray
    trajectories: list[list[MountainCarTransition]]
    infos: list[dict]

    @property
    def transitions(self) -> int:
        return int(sum(len(trajectory) for trajectory in self.trajectories))


class MountainCarNeuralActor:
    """One-hidden-layer categorical actor with exact capacity controls.

    Each slot contains ``W_in`` (2 by H), a hidden bias (H), and ``W_out``
    (H by 3), with no output bias.  Thus each slot has exactly ``6H`` trainable
    parameters:

    * shared H64: 384 total and 384 active;
    * disjoint H8 x 8: 384 total and 48 active;
    * disjoint H64 x 8: 3,072 total and 384 active.
    """

    SHARED = "shared_h64"
    DISJOINT_TOTAL = "disjoint_total_h8x8"
    DISJOINT_ACTIVE = "disjoint_active_h64x8"
    _HIDDEN = {SHARED: 64, DISJOINT_TOTAL: 8, DISJOINT_ACTIVE: 64}

    def __init__(
        self,
        *,
        mode: str = SHARED,
        learning_rate: float = 3e-4,
        parameter_seed: int = 0,
        action_seed: Optional[int] = None,
    ):
        if mode not in self._HIDDEN:
            raise ValueError(f"unknown registered actor mode {mode!r}")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.mode = mode
        self.hidden_size = self._HIDDEN[mode]
        self.n_slots = 1 if mode == self.SHARED else N_TASKS
        self.learning_rate = float(learning_rate)

        parameter_rng = np.random.default_rng(parameter_seed)
        self.action_rng = np.random.default_rng(
            parameter_seed + 1 if action_seed is None else action_seed
        )
        self.W_in = parameter_rng.normal(
            0.0, 1.0 / np.sqrt(2.0), size=(self.n_slots, 2, self.hidden_size)
        )
        self.b_hidden = np.zeros((self.n_slots, self.hidden_size), dtype=np.float64)
        # Every architecture starts with the identical uniform action policy.
        self.W_out = np.zeros(
            (self.n_slots, self.hidden_size, N_ACTIONS), dtype=np.float64
        )
        self.update_calls = 0
        self.applied_updates = 0
        self.slot_update_calls = np.zeros(self.n_slots, dtype=np.int64)

        expected = {
            self.SHARED: (384, 384),
            self.DISJOINT_TOTAL: (384, 48),
            self.DISJOINT_ACTIVE: (3072, 384),
        }[mode]
        if (self.parameter_count, self.active_parameter_count) != expected:
            raise RuntimeError("registered MountainCar capacity identity failed")

    @property
    def is_shared(self) -> bool:
        return self.mode == self.SHARED

    @property
    def parameter_count(self) -> int:
        return int(self.W_in.size + self.b_hidden.size + self.W_out.size)

    @property
    def active_parameter_count(self) -> int:
        return int(6 * self.hidden_size)

    def _slot(self, task_id: int) -> int:
        task = int(task_id)
        if not 0 <= task < N_TASKS:
            raise IndexError(f"task_id {task} outside [0, {N_TASKS})")
        return 0 if self.is_shared else task

    def parameter_vector(self) -> np.ndarray:
        return np.concatenate(
            (self.W_in.ravel(), self.b_hidden.ravel(), self.W_out.ravel())
        ).copy()

    def slot_parameter_vector(self, task_id: int) -> np.ndarray:
        slot = self._slot(task_id)
        return np.concatenate(
            (
                self.W_in[slot].ravel(),
                self.b_hidden[slot].ravel(),
                self.W_out[slot].ravel(),
            )
        ).copy()

    def parameter_sha256(self) -> str:
        return hashlib.sha256(self.parameter_vector().tobytes()).hexdigest()

    def parameter_state(self) -> dict[str, np.ndarray]:
        """Copy trainable arrays without counters or RNG state."""

        return {
            "W_in": self.W_in.copy(),
            "b_hidden": self.b_hidden.copy(),
            "W_out": self.W_out.copy(),
        }

    def load_parameter_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Restore a shape-checked parameter snapshot, leaving RNG/counters alone."""

        expected = {
            "W_in": self.W_in.shape,
            "b_hidden": self.b_hidden.shape,
            "W_out": self.W_out.shape,
        }
        if set(state) != set(expected):
            raise ValueError("parameter snapshot keys are invalid")
        copied = {}
        for key, shape in expected.items():
            value = np.asarray(state[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"parameter snapshot {key} is invalid")
            copied[key] = value.copy()
        self.W_in = copied["W_in"]
        self.b_hidden = copied["b_hidden"]
        self.W_out = copied["W_out"]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    @staticmethod
    def _forward(
        observation: np.ndarray,
        W_in: np.ndarray,
        b_hidden: np.ndarray,
        W_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(observation @ W_in + b_hidden)
        logits = hidden @ W_out
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return hidden, exponentials / exponentials.sum()

    def probabilities(
        self, observation: Sequence[float], task_id: int = 0
    ) -> np.ndarray:
        slot = self._slot(task_id)
        _, probabilities = self._forward(
            normalize_observation(observation),
            self.W_in[slot],
            self.b_hidden[slot],
            self.W_out[slot],
        )
        return probabilities

    def act(
        self,
        observation: Sequence[float],
        task_id: int = 0,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        generator = self.action_rng if rng is None else rng
        return int(
            generator.choice(N_ACTIONS, p=self.probabilities(observation, task_id))
        )

    @staticmethod
    def _step(step) -> tuple[np.ndarray, int]:
        if isinstance(step, MountainCarTransition):
            return normalize_observation(step.obs_before), int(step.action)
        if isinstance(step, Mapping):
            observation = step.get("obs_before", step.get("observation"))
            if observation is None:
                raise ValueError("transition mapping lacks obs_before")
            return normalize_observation(observation), int(step["action"])
        if len(step) != 2:
            raise ValueError("tuple transitions must be (observation, action)")
        return normalize_observation(step[0]), int(step[1])

    def group_gradient(
        self,
        task_id: int,
        trajectories: Sequence[Sequence],
        weights: Sequence[float],
    ) -> tuple[dict[str, np.ndarray], dict]:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(trajectories),) or not np.isfinite(values).all():
            raise ValueError("weights must be finite and match the rollout group")
        slot = self._slot(task_id)
        W_in = self.W_in[slot].copy()
        b_hidden = self.b_hidden[slot].copy()
        W_out = self.W_out[slot].copy()
        grad_in = np.zeros_like(W_in)
        grad_hidden = np.zeros_like(b_hidden)
        grad_out = np.zeros_like(W_out)
        entropy_sum = 0.0
        score_terms = 0
        weighted_terms = 0

        for trajectory, weight in zip(trajectories, values):
            for step in trajectory:
                observation, action = self._step(step)
                if not 0 <= action < N_ACTIONS:
                    raise ValueError("MountainCar action must be 0, 1, or 2")
                hidden, probabilities = self._forward(
                    observation, W_in, b_hidden, W_out
                )
                entropy_sum -= float(
                    np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
                )
                score_terms += 1
                if weight == 0.0:
                    continue
                output_score = -probabilities.copy()
                output_score[action] += 1.0
                hidden_score = (W_out @ output_score) * (1.0 - hidden * hidden)
                grad_out += weight * np.outer(hidden, output_score)
                grad_in += weight * np.outer(observation, hidden_score)
                grad_hidden += weight * hidden_score
                weighted_terms += 1

        gradient = {"W_in": grad_in, "b_hidden": grad_hidden, "W_out": grad_out}
        gradient_norm = float(
            np.sqrt(sum(float(np.sum(value * value)) for value in gradient.values()))
        )
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("non-finite MountainCar policy gradient")
        return gradient, {
            "slot": slot,
            "gradient_norm": gradient_norm,
            "update_norm": self.learning_rate * gradient_norm,
            "n_trajectories": len(trajectories),
            "n_score_terms": score_terms,
            "n_weighted_score_terms": weighted_terms,
            "weight_l1": float(np.abs(values).sum()),
            "mean_policy_entropy": (entropy_sum / score_terms if score_terms else None),
            "frozen_group_parameters": True,
        }

    def update(
        self,
        task_id: int,
        trajectories: Sequence[Sequence],
        weights: Sequence[float],
    ) -> dict:
        """Apply one plain-SGD ascent step from a frozen rollout group."""

        gradient, diagnostics = self.group_gradient(task_id, trajectories, weights)
        slot = int(diagnostics["slot"])
        self.W_in[slot] += self.learning_rate * gradient["W_in"]
        self.b_hidden[slot] += self.learning_rate * gradient["b_hidden"]
        self.W_out[slot] += self.learning_rate * gradient["W_out"]
        self.update_calls += 1
        self.slot_update_calls[slot] += 1
        applied = bool(float(diagnostics["update_norm"]) > 0.0)
        if applied:
            self.applied_updates += 1
        return {
            "task_id": int(task_id),
            **diagnostics,
            "applied": applied,
        }

    def diagnostics(self) -> dict:
        return {
            "mode": self.mode,
            "hidden_size": self.hidden_size,
            "n_slots": self.n_slots,
            "parameter_count": self.parameter_count,
            "active_parameter_count": self.active_parameter_count,
            "parameter_sha256": self.parameter_sha256(),
            "parameter_norm": float(np.linalg.norm(self.parameter_vector())),
            "update_calls": int(self.update_calls),
            "applied_updates": int(self.applied_updates),
            "slot_update_calls": self.slot_update_calls.astype(int).tolist(),
        }


class MountainCarSparseGoalSpace:
    """Eight fixed sparse positional goals on ``MountainCar-v0`` dynamics."""

    def __init__(
        self,
        *,
        actor: Optional[MountainCarNeuralActor] = None,
        mode: str = MountainCarNeuralActor.SHARED,
        learning_rate: float = 3e-4,
        seed: int = 0,
        registered_seed_authorization: Optional[_RegisteredSeedAuthorization] = None,
    ):
        if type(seed) is not int:
            raise TypeError("MountainCar V1 seed must be a primitive Python int")
        if seed in RESERVED_CONFIRMATORY_SEEDS:
            raise RuntimeError("confirmatory seeds have no MountainCar V1 core path")
        if seed in REGISTERED_DEVELOPMENT_SEEDS and not (
            type(registered_seed_authorization) is _RegisteredSeedAuthorization
            and registered_seed_authorization.seed == seed
            and registered_seed_authorization.nonce is _AUTHORIZATION_NONCE
        ):
            raise RuntimeError(
                "registered development seed lacks validated-lock authorization"
            )
        if (
            seed not in REGISTERED_DEVELOPMENT_SEEDS
            and registered_seed_authorization is not None
        ):
            raise RuntimeError(
                "registered-seed authorization supplied for an excluded seed"
            )
        self.seed = seed
        self.actor = actor or MountainCarNeuralActor(
            mode=mode,
            learning_rate=learning_rate,
            parameter_seed=self.seed,
            action_seed=self.seed + TRAINING_ACTION_SEED_OFFSET,
        )
        self.episode_rng = np.random.default_rng(
            self.seed + TRAINING_EPISODE_SEED_OFFSET
        )
        self.env = gym.make("MountainCar-v0")
        if (
            self.env.spec is None
            or self.env.spec.id != "MountainCar-v0"
            or self.env.spec.max_episode_steps != MAX_EPISODE_STEPS
            or tuple(self.env.observation_space.shape) != (2,)
            or int(self.env.action_space.n) != N_ACTIONS
        ):
            self.env.close()
            raise RuntimeError("unexpected Gymnasium MountainCar-v0 contract")

    @property
    def n_tasks(self) -> int:
        return N_TASKS

    def _episode(self, task_id: int) -> tuple[list[MountainCarTransition], dict]:
        task = int(task_id)
        if not 0 <= task < N_TASKS:
            raise IndexError(f"task_id {task} outside [0, {N_TASKS})")
        episode_seed = int(self.episode_rng.integers(0, 2**31 - 1))
        observation, _ = self.env.reset(seed=episode_seed)
        trajectory: list[MountainCarTransition] = []
        max_position = float(observation[0])
        success = False
        for _ in range(MAX_EPISODE_STEPS):
            before = np.asarray(observation, dtype=np.float64).copy()
            action = self.actor.act(before, task)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            after = np.asarray(observation, dtype=np.float64).copy()
            max_position = max(max_position, float(after[0]))
            trajectory.append(
                MountainCarTransition(
                    before,
                    action,
                    after,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                )
            )
            if float(after[0]) >= THRESHOLDS[task]:
                success = True
                break
            if terminated or truncated:
                break
        terminal = trajectory[-1]
        positions_before_final = [float(trajectory[0].obs_before[0])] + [
            float(transition.obs_after[0]) for transition in trajectory[:-1]
        ]
        return trajectory, {
            "episode_seed": episode_seed,
            "n_steps": len(trajectory),
            "max_position": max_position,
            "max_position_before_final": max(positions_before_final),
            "pre_final_position": float(terminal.obs_before[0]),
            "final_position": float(terminal.obs_after[0]),
            "native_terminated": bool(terminal.native_terminated),
            "native_truncated": bool(terminal.truncated),
            "native_reward_sum": float(
                sum(transition.native_reward for transition in trajectory)
            ),
            "success": success,
        }

    def rollout_group(
        self, task_id: int, n_rollouts: int = N_ROLLOUTS
    ) -> MountainCarGroup:
        if int(n_rollouts) != N_ROLLOUTS:
            raise ValueError(
                f"registered rollout groups contain exactly {N_ROLLOUTS} episodes"
            )
        trajectories, infos, rewards = [], [], []
        for _ in range(N_ROLLOUTS):
            trajectory, info = self._episode(task_id)
            trajectories.append(trajectory)
            infos.append(info)
            rewards.append(float(info["success"]))
        return MountainCarGroup(
            int(task_id), np.asarray(rewards, dtype=np.float64), trajectories, infos
        )

    def evaluate(self, *, n: int, seed: int) -> dict:
        """Evaluate with fixed common random numbers without touching training RNG.

        Episodes run to the native 200-step horizon (or the native flag), then
        the maximum reached position is scored.  Shared actors receive the same
        episode and action RNG streams for all thresholds, making nestedness an
        exact mechanical invariant rather than a sampling expectation.
        """

        if int(n) <= 0:
            raise ValueError("evaluation episode count must be positive")
        episode_state_before = copy.deepcopy(self.episode_rng.bit_generator.state)
        action_state_before = copy.deepcopy(self.actor.action_rng.bit_generator.state)
        parameter_hash_before = self.actor.parameter_sha256()
        episode_seeds = np.random.default_rng(int(seed)).integers(
            0, 2**31 - 1, size=int(n), dtype=np.int64
        )
        evaluation_action_seed_root = int(seed) + EVALUATION_ACTION_SEED_OFFSET
        action_seeds = np.random.default_rng(evaluation_action_seed_root).integers(
            0, 2**31 - 1, size=int(n), dtype=np.int64
        )
        pass_rates = []
        max_position_samples = []
        try:
            for task_id, threshold in enumerate(THRESHOLDS):
                evaluation_env = gym.make("MountainCar-v0")
                task_max_positions = []
                try:
                    for episode_seed, action_seed in zip(episode_seeds, action_seeds):
                        action_rng = np.random.default_rng(int(action_seed))
                        observation, _ = evaluation_env.reset(seed=int(episode_seed))
                        max_position = float(observation[0])
                        for _ in range(MAX_EPISODE_STEPS):
                            action = self.actor.act(
                                observation, task_id, rng=action_rng
                            )
                            observation, _, terminated, truncated, _ = (
                                evaluation_env.step(action)
                            )
                            max_position = max(max_position, float(observation[0]))
                            if terminated or truncated:
                                break
                        task_max_positions.append(max_position)
                finally:
                    evaluation_env.close()
                max_position_samples.append(task_max_positions)
                pass_rates.append(
                    sum(position >= threshold for position in task_max_positions)
                    / int(n)
                )
        finally:
            # The evaluator never intentionally mutates these states.  Restore
            # them defensively so an exception cannot perturb later training.
            self.episode_rng.bit_generator.state = episode_state_before
            self.actor.action_rng.bit_generator.state = action_state_before

        rates = np.asarray(pass_rates, dtype=np.float64)
        episode_preserved = _state_hash(
            self.episode_rng.bit_generator.state
        ) == _state_hash(episode_state_before)
        action_preserved = _state_hash(
            self.actor.action_rng.bit_generator.state
        ) == _state_hash(action_state_before)
        parameter_preserved = self.actor.parameter_sha256() == parameter_hash_before
        return {
            "pass_rates": rates.tolist(),
            "mean_pass": float(rates.mean()),
            "hardest_pass": float(rates[-1]),
            "max_position_samples": max_position_samples,
            "evaluation_episode_seeds": episode_seeds.astype(int).tolist(),
            "evaluation_action_seeds": action_seeds.astype(int).tolist(),
            "training_episode_rng_preserved": episode_preserved,
            "training_action_rng_preserved": action_preserved,
            "training_parameters_preserved": parameter_preserved,
            "shared_nested_pass_rates": (
                bool(np.all(rates[:-1] >= rates[1:])) if self.actor.is_shared else None
            ),
            "evaluation_seed": int(seed),
            "evaluation_action_seed_root": evaluation_action_seed_root,
            "evaluated_parameter_sha256": parameter_hash_before,
            "episodes_per_task": int(n),
        }

    def close(self) -> None:
        self.env.close()
