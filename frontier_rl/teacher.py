"""FrontierTeacher: the validated curriculum sampler.

Utility (PROOFS.md P1): u(p) = (1-(1-p)^N) - p — half the exact expected
coefficient L1 mass of the MaxRL estimator per group, peaking at
p* ≈ ln(N)/N.
Posterior: decayed Beta per task (decay 0.7, VALIDATION.md V2b).
Sampling: Thompson draw → u^gamma (V6: gamma tracks task-graph
connectivity — 4 for chained/shared-skill pools, 1 for flat pools) →
mix with uniform floor (P7: the floor bounds posterior staleness).

Validated defaults are the constructor defaults; every knob's provenance
is in its docstring line.
"""

from __future__ import annotations

import copy

import numpy as np


class FrontierTeacher:
    def __init__(self, n_tasks: int, n_rollouts: int = 16, *,
                 decay: float = 0.7,      # V2b: tracking > memory
                 floor: float = 0.1,      # P7/V3: staleness insurance
                 gamma: float = 1.0,      # V6: raise to ~4 on chained pools
                 seed: int = 0):
        if not isinstance(n_tasks, (int, np.integer)) or n_tasks <= 0:
            raise ValueError(f"n_tasks must be a positive integer, got {n_tasks!r}")
        if not isinstance(n_rollouts, (int, np.integer)) or n_rollouts < 2:
            raise ValueError(f"n_rollouts must be an integer >= 2, got {n_rollouts!r}")
        if not np.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be finite and in [0, 1], got {decay!r}")
        if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be finite and in [0, 1], got {floor!r}")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma must be positive and finite, got {gamma!r}")
        self.n_tasks = int(n_tasks)
        self.n_rollouts = int(n_rollouts)
        self.decay = float(decay)
        self.floor = float(floor)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_tasks)
        self.beta = np.ones(n_tasks)
        self.visits = np.zeros(n_tasks, dtype=np.int64)

    # -- evidence ---------------------------------------------------------
    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        """Update the task's posterior from one group's binary rewards.

        Only requested-task evidence belongs here — feeding relabeled
        successes back inflates the posterior (V4 + GPU A/B/C config C).
        """
        if (
            isinstance(task_id, (bool, np.bool_))
            or not isinstance(task_id, (int, np.integer))
            or not 0 <= task_id < self.n_tasks
        ):
            raise ValueError(f"task_id must lie in [0, {self.n_tasks}), got {task_id!r}")
        rewards = np.asarray(rewards, dtype=float)
        if rewards.ndim != 1 or rewards.size == 0:
            raise ValueError("rewards must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("rewards must contain only finite values")
        if np.any((rewards != 0.0) & (rewards != 1.0)):
            raise ValueError("rewards must be binary values in {0, 1}")
        k = float(np.sum(rewards))
        n = float(len(rewards))
        self.alpha[task_id] = 1.0 + (self.alpha[task_id] - 1.0) * self.decay + k
        self.beta[task_id] = 1.0 + (self.beta[task_id] - 1.0) * self.decay + (n - k)
        self.visits[task_id] += 1

    # -- sampling ---------------------------------------------------------
    def utility(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("pass rates must be finite and lie in [0, 1]")
        return np.maximum((1.0 - (1.0 - p) ** self.n_rollouts) - p, 0.0)

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = self.utility(p) ** self.gamma
        if u.sum() <= 1e-12:
            u = np.ones(self.n_tasks)
        probs = u / u.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1.0 - self.floor) * probs + self.floor * uniform

    def sample_tasks(self, batch: int) -> np.ndarray:
        if (
            isinstance(batch, (bool, np.bool_))
            or not isinstance(batch, (int, np.integer))
            or batch < 0
        ):
            raise ValueError(f"batch must be a non-negative integer, got {batch!r}")
        return self.rng.choice(self.n_tasks, size=batch, p=self.distribution())

    # -- introspection / persistence --------------------------------------
    def pass_rate_estimates(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def metrics(self) -> dict:
        p = self.pass_rate_estimates()
        seen = self.visits > 0
        out = {"teacher/visited_frac": float(seen.mean())}
        if seen.any():
            out["teacher/frac_dead"] = float((p[seen] < 0.05).mean())
            out["teacher/frac_mastered"] = float((p[seen] > 0.9).mean())
        return out

    def state_dict(self) -> dict:
        return {"version": 2,
                "n_tasks": self.n_tasks,
                "config": {"n_rollouts": self.n_rollouts, "decay": self.decay,
                           "floor": self.floor, "gamma": self.gamma},
                "alpha": self.alpha.copy(), "beta": self.beta.copy(),
                "visits": self.visits.copy(),
                "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        if "n_tasks" in state and int(state["n_tasks"]) != self.n_tasks:
            raise ValueError(f"state has {state['n_tasks']} tasks, teacher expects {self.n_tasks}")
        config = state.get("config")
        if config is not None:
            expected = self.state_dict()["config"]
            mismatches = [
                key for key, value in expected.items()
                if key not in config or config[key] != value
            ]
            if mismatches:
                detail = ", ".join(
                    f"{key}: checkpoint={config.get(key)!r}, current={expected[key]!r}"
                    for key in mismatches
                )
                raise ValueError(f"teacher configuration mismatch on resume ({detail})")
        alpha = np.asarray(state["alpha"], dtype=float)
        beta = np.asarray(state["beta"], dtype=float)
        visits = np.asarray(state["visits"], dtype=np.int64)
        expected = (self.n_tasks,)
        if alpha.shape != expected or beta.shape != expected or visits.shape != expected:
            raise ValueError(
                f"teacher state arrays must have shape {expected}, got "
                f"{alpha.shape}, {beta.shape}, and {visits.shape}"
            )
        if (
            not np.all(np.isfinite(alpha))
            or not np.all(np.isfinite(beta))
            or np.any(alpha <= 0)
            or np.any(beta <= 0)
            or np.any(visits < 0)
        ):
            raise ValueError("teacher state contains invalid posterior parameters")
        self.alpha = alpha.copy()
        self.beta = beta.copy()
        self.visits = visits.copy()
        if "rng_state" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
