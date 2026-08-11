"""FrontierTeacher: the validated curriculum sampler.

Utility (PROOFS.md P1): u(p) = (1-(1-p)^N) - p — the exact expected
advantage mass of the MaxRL estimator per group, peaking at p* ≈ ln(N)/N.
Posterior: decayed Beta per task (decay 0.7, VALIDATION.md V2b).
Sampling: Thompson draw → u^gamma (V6: gamma tracks task-graph
connectivity — 4 for chained/shared-skill pools, 1 for flat pools) →
mix with uniform floor (P7: the floor bounds posterior staleness).

Validated defaults are the constructor defaults; every knob's provenance
is in its docstring line.
"""

from __future__ import annotations

import numpy as np


class FrontierTeacher:
    def __init__(self, n_tasks: int, n_rollouts: int = 16, *,
                 decay: float = 0.7,      # V2b: tracking > memory
                 floor: float = 0.1,      # P7/V3: staleness insurance
                 gamma: float = 1.0,      # V6: raise to ~4 on chained pools
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

        Only requested-task evidence belongs here — feeding relabeled
        successes back inflates the posterior (V4 + GPU A/B/C config C).
        """
        k = float(np.sum(rewards))
        n = float(len(rewards))
        self.alpha[task_id] = 1.0 + (self.alpha[task_id] - 1.0) * self.decay + k
        self.beta[task_id] = 1.0 + (self.beta[task_id] - 1.0) * self.decay + (n - k)
        self.visits[task_id] += 1

    # -- sampling ---------------------------------------------------------
    def utility(self, p: np.ndarray) -> np.ndarray:
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


class UniformTeacher(FrontierTeacher):
    """No-curriculum control with the same posterior bookkeeping.

    Keeping the observations and posterior identical to the adaptive teachers
    makes calibration and dead/mastered diagnostics comparable across arms.
    """

    def distribution(self) -> np.ndarray:
        return np.full(self.n_tasks, 1.0 / self.n_tasks)


class LearnabilityTeacher(FrontierTeacher):
    """Compute-blind learnability baseline, ``u(p) = p(1-p)``.

    This is the N=2 slice of the estimator-derived utility.  It intentionally
    shares the posterior, Thompson sampling, concentration, and uniform floor
    with :class:`FrontierTeacher`, isolating the score shape in an N-ablation.
    It is a learnability baseline, not an ALP-GMM implementation: ALP-GMM
    estimates temporal learning progress and fits a mixture model.
    """

    def utility(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p * (1.0 - p)


class StagedDifficultyTeacher(FrontierTeacher):
    """Hand-ordered promotion curriculum for a known difficulty ranking.

    The teacher samples uniformly from the unlocked prefix.  It unlocks the
    next task once the current frontier's posterior mean reaches
    ``promotion_threshold`` after ``min_frontier_groups`` observations.  A
    uniform floor keeps locked tasks observable and matches the coverage
    safeguard used by the adaptive arms.

    This deliberately simple baseline represents the common hand-designed
    easy-to-hard schedule; its ordering must come from environment metadata,
    never from evaluation outcomes.
    """

    def __init__(self, n_tasks: int, n_rollouts: int = 16, *,
                 difficulty_order=None, initial_tasks: int = 1,
                 promotion_threshold: float = 0.7,
                 min_frontier_groups: int = 5, floor: float = 0.1,
                 decay: float = 0.7, seed: int = 0):
        super().__init__(n_tasks, n_rollouts, decay=decay, floor=floor,
                         gamma=1.0, seed=seed)
        if difficulty_order is None:
            difficulty_order = np.arange(n_tasks)
        order = np.asarray(difficulty_order, dtype=np.int64)
        if order.shape != (n_tasks,) or set(order.tolist()) != set(range(n_tasks)):
            raise ValueError("difficulty_order must be a permutation of task ids")
        if not 1 <= initial_tasks <= n_tasks:
            raise ValueError("initial_tasks must lie in [1, n_tasks]")
        if not 0.0 <= promotion_threshold <= 1.0:
            raise ValueError("promotion_threshold must lie in [0, 1]")
        if min_frontier_groups < 1:
            raise ValueError("min_frontier_groups must be positive")
        self.difficulty_order = order
        self.active_count = int(initial_tasks)
        self.promotion_threshold = float(promotion_threshold)
        self.min_frontier_groups = int(min_frontier_groups)

    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        super().observe(task_id, rewards)
        while self.active_count < self.n_tasks:
            frontier = int(self.difficulty_order[self.active_count - 1])
            if self.visits[frontier] < self.min_frontier_groups:
                break
            p_hat = self.alpha[frontier] / (self.alpha[frontier] + self.beta[frontier])
            if p_hat < self.promotion_threshold:
                break
            self.active_count += 1

    def distribution(self) -> np.ndarray:
        active = self.difficulty_order[:self.active_count]
        staged = np.zeros(self.n_tasks, dtype=float)
        staged[active] = 1.0 / len(active)
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1.0 - self.floor) * staged + self.floor * uniform

    def metrics(self) -> dict:
        out = super().metrics()
        out["teacher/active_frac"] = self.active_count / self.n_tasks
        out["teacher/frontier_task"] = int(
            self.difficulty_order[self.active_count - 1])
        return out

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["active_count"] = self.active_count
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.active_count = int(state["active_count"])


def allocate_rollouts_greedy(p_hat: np.ndarray, total_budget: int, *,
                             n_min: int = 1, n_max: int = 64) -> np.ndarray:
    """Allocate a fixed rollout budget by exact discrete water-filling.

    For task ``i`` with pass probability ``p_i``, increasing its group size
    from ``N`` to ``N+1`` changes ``pass@N`` by
    ``p_i (1-p_i)^N``.  These marginal gains decrease with N, so repeatedly
    assigning the next rollout to the largest marginal is optimal for the
    separable concave objective ``sum_i pass@N_i`` (and equivalently for
    ``sum_i [pass@N_i - pass@1_i]``).

    The returned integer counts always sum exactly to ``total_budget``.
    Infeasible bounds are rejected instead of silently violating the budget.
    """
    p = np.asarray(p_hat, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_hat must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_hat entries must be finite probabilities")
    if n_min < 0 or n_max < n_min:
        raise ValueError("require 0 <= n_min <= n_max")
    minimum = n_min * len(p)
    maximum = n_max * len(p)
    if not minimum <= total_budget <= maximum:
        raise ValueError(
            f"budget {total_budget} is infeasible for [{minimum}, {maximum}]")

    allocation = np.full(len(p), n_min, dtype=np.int64)
    for _ in range(total_budget - minimum):
        marginal = p * (1.0 - p) ** allocation
        marginal[allocation >= n_max] = -1.0
        allocation[int(np.argmax(marginal))] += 1
    return allocation
