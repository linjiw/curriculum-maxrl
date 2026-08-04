"""FrontierTeacher: the practical-MaxRL-aligned curriculum sampler.

For the practical centered/drop estimator, the normalized expected
coefficient half-mass is

    ν_N(p) = (1-p) - (1-p)^N = pass@N - pass@1.

The full expected absolute coefficient mass is ``2ν_N``.  This exact
finite-group activity profile peaks at ``1 - N**(-1/(N-1))`` (approximately
``ln(N)/N``), but it is not the activity profile of raw MaxRL or the full
control-variate estimator.  In particular, full CV retains a negative-only
update at ``K=0``, while the practical estimator drops that group.

Sampling uses a decayed Beta posterior, a Thompson draw, ``ν_N**gamma``,
and a uniform floor.  Decay, gamma, and the floor are empirical controls,
not consequences of the coefficient-mass derivation.
"""

from __future__ import annotations

import numpy as np


class FrontierTeacher:
    def __init__(self, n_tasks: int, n_rollouts: int = 16, *,
                 decay: float = 0.7,      # empirical tracking control
                 floor: float = 0.1,      # posterior-staleness insurance
                 gamma: float = 1.0,      # empirical concentration control
                 seed: int = 0):
        self.n_tasks = n_tasks
        self.n_rollouts = n_rollouts
        self.decay = decay
        self.floor = floor
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_tasks)
        self.beta = np.ones(n_tasks)
        self.visits = np.zeros(n_tasks, dtype=np.int64)

    # -- evidence ---------------------------------------------------------
    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        """Update the task's posterior from one group's binary rewards.

        Only requested-task evidence belongs here: relabeled outcomes come
        from a different proposal and feeding them back inflates the
        requested task's posterior.
        """
        k = float(np.sum(rewards))
        n = float(len(rewards))
        self.alpha[task_id] = 1.0 + (self.alpha[task_id] - 1.0) * self.decay + k
        self.beta[task_id] = 1.0 + (self.beta[task_id] - 1.0) * self.decay + (n - k)
        self.visits[task_id] += 1

    # -- sampling ---------------------------------------------------------
    def utility(self, p: np.ndarray) -> np.ndarray:
        """Return ``ν_N(p)``, practical MaxRL's coefficient half-mass."""
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
        return {"alpha": self.alpha.copy(), "beta": self.beta.copy(),
                "visits": self.visits.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.alpha = np.asarray(state["alpha"], dtype=float)
        self.beta = np.asarray(state["beta"], dtype=float)
        self.visits = np.asarray(state["visits"], dtype=np.int64)
