"""Curriculum teachers: map per-task statistics -> a sampling distribution.

All teachers see only what a real RLVR trainer sees: the empirical rewards of
past rollouts (never the true pass rate).  Statistics are tracked with an
exponential moving average plus a Beta posterior for uncertainty.

Teachers
--------
UniformTeacher       no curriculum (baseline).
ZPDBandTeacher       samples tasks whose estimated pass rate lies in a target
                     band (ADARFT/DAPO-style difficulty targeting).
ALPTeacher           absolute-learning-progress bandit (TSCL / ALP-GMM style):
                     samples proportional to |d p̂ / dt|.
MaxRLFrontierTeacher legacy heuristic: pass@N * (1-p), retained to reproduce
                     the original proposal tables.
AdvMassTeacher       current derived utility: pass@N - p, half the exact
                     expected coefficient L1 mass of practical Algorithm 1.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TaskStats:
    """Online statistics for one task, from observed rollout rewards only."""
    ema_pass: float = 0.0
    ema_initialized: bool = False
    alpha_beta: tuple[float, float] = (1.0, 1.0)  # Beta posterior params
    prev_ema: float = 0.0
    visits: int = 0

    def update(self, rewards: np.ndarray, ema_w: float = 0.3):
        mean_r = float(rewards.mean())
        self.prev_ema = self.ema_pass if self.ema_initialized else mean_r
        if not self.ema_initialized:
            self.ema_pass = mean_r
            self.ema_initialized = True
        else:
            self.ema_pass = (1 - ema_w) * self.ema_pass + ema_w * mean_r
        a, b = self.alpha_beta
        # decay old evidence so the posterior tracks the moving policy
        decay = 0.9
        self.alpha_beta = (1.0 + (a - 1.0) * decay + rewards.sum(),
                           1.0 + (b - 1.0) * decay + (len(rewards) - rewards.sum()))
        self.visits += 1


class Teacher:
    def __init__(self, n_tasks: int, seed: int = 0):
        self.n_tasks = n_tasks
        self.stats = [TaskStats() for _ in range(n_tasks)]
        self.rng = np.random.default_rng(seed)

    def sample_tasks(self, batch_size: int) -> np.ndarray:
        probs = self.distribution()
        return self.rng.choice(self.n_tasks, size=batch_size, p=probs)

    def distribution(self) -> np.ndarray:
        raise NotImplementedError

    def observe(self, task_id: int, rewards: np.ndarray):
        self.stats[task_id].update(rewards)


class UniformTeacher(Teacher):
    def distribution(self) -> np.ndarray:
        return np.full(self.n_tasks, 1.0 / self.n_tasks)


class ZPDBandTeacher(Teacher):
    """Prefer tasks with estimated pass rate inside [lo, hi]."""

    def __init__(self, n_tasks: int, seed: int = 0, lo: float = 0.05, hi: float = 0.85,
                 explore_frac: float = 0.15):
        super().__init__(n_tasks, seed)
        self.lo, self.hi, self.explore_frac = lo, hi, explore_frac

    def distribution(self) -> np.ndarray:
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            if not st.ema_initialized:
                w[i] = 1.0  # unvisited counts as in-band (optimism)
            elif self.lo <= st.ema_pass <= self.hi:
                w[i] = 1.0
        if w.sum() == 0:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


class ALPTeacher(Teacher):
    """Absolute learning progress bandit (softmax over |Δ ema_pass|)."""

    def __init__(self, n_tasks: int, seed: int = 0, explore_frac: float = 0.2):
        super().__init__(n_tasks, seed)
        self.explore_frac = explore_frac
        self.alp = np.zeros(n_tasks)

    def observe(self, task_id: int, rewards: np.ndarray):
        st = self.stats[task_id]
        st.update(rewards)
        self.alp[task_id] = 0.7 * self.alp[task_id] + 0.3 * abs(st.ema_pass - st.prev_ema)

    def distribution(self) -> np.ndarray:
        w = self.alp.copy()
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


class MaxRLFrontierTeacher(Teacher):
    """Legacy heuristic-frontier curriculum.

    This was the original proposal and is retained for reproducibility:

        u(p) = (1 - (1-p)^N) * (1 - p)

    It is not the exact expected coefficient mass. Use ``AdvMassTeacher`` for
    the derived utility ``pass@N-p``.
    """

    def __init__(self, n_tasks: int, seed: int = 0, n_rollouts: int = 16,
                 explore_frac: float = 0.1):
        super().__init__(n_tasks, seed)
        self.n_rollouts = n_rollouts
        self.explore_frac = explore_frac

    def distribution(self) -> np.ndarray:
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            a, b = st.alpha_beta
            p = self.rng.beta(a, b)
            w[i] = (1.0 - (1.0 - p) ** self.n_rollouts) * (1.0 - p)
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


class AdvMassTeacher(Teacher):
    """Teacher driven by the exact MaxRL coefficient-mass utility.

    THEORY.md section 2: for a group of N rollouts, expected coefficient L1
    mass is 2*(pass@N(p) - p). The normalized utility used here,
    pass@N(p)-p, is the probability the prompt is solvable within N attempts
    but not within one. Thompson sampling over the Beta pseudo-posterior
    supplies uncertainty-driven exploration.

    ``power`` (VALIDATION.md V6): sample ∝ u^power. This soft distribution is
    an exploration/coverage heuristic, not the one-step mass maximizer (which
    would put all non-floor probability on argmax u). Sharper concentration
    wins empirically on chain-structured pools (γ≈4 saturates); use 1–2 on
    flat pools.
    """

    def __init__(self, n_tasks: int, seed: int = 0, n_rollouts: int = 16,
                 explore_frac: float = 0.1, power: float = 1.0):
        super().__init__(n_tasks, seed)
        self.n_rollouts = n_rollouts
        self.explore_frac = explore_frac
        self.power = power

    def distribution(self) -> np.ndarray:
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            a, b = st.alpha_beta
            p = self.rng.beta(a, b)
            w[i] = (1.0 - (1.0 - p) ** self.n_rollouts) - p
        w = np.maximum(w, 0.0) ** self.power
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


def _validate_rollout_budget_inputs(
        p_hat: np.ndarray, total_budget: int, n_min: int,
        n_max: int) -> np.ndarray:
    """Validate a bounded integer rollout-allocation problem."""
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.ndim != 1:
        raise ValueError(f"p_hat must be one-dimensional, got shape {p_hat.shape}")
    if not np.all(np.isfinite(p_hat)):
        raise ValueError("p_hat must contain only finite values")
    if np.any((p_hat < 0.0) | (p_hat > 1.0)):
        raise ValueError("p_hat values must lie in [0, 1]")
    if (isinstance(total_budget, (bool, np.bool_))
            or not isinstance(total_budget, (int, np.integer))
            or total_budget < 0):
        raise ValueError(f"total_budget must be a non-negative integer, got {total_budget!r}")
    if (isinstance(n_min, (bool, np.bool_))
            or isinstance(n_max, (bool, np.bool_))
            or not isinstance(n_min, (int, np.integer))
            or not isinstance(n_max, (int, np.integer))):
        raise ValueError("n_min and n_max must be integers")
    if n_min < 1 or n_max < n_min:
        raise ValueError(f"require 1 <= n_min <= n_max, got {n_min} and {n_max}")

    n_items = len(p_hat)
    if n_items == 0:
        if total_budget == 0:
            return p_hat
        raise ValueError("a non-zero budget cannot be allocated to an empty prompt set")
    min_budget, max_budget = n_min * n_items, n_max * n_items
    if not min_budget <= total_budget <= max_budget:
        raise ValueError(
            f"total_budget={total_budget} is infeasible for {n_items} prompts with "
            f"bounds [{n_min}, {n_max}]; expected [{min_budget}, {max_budget}]"
        )
    return p_hat


def _allocate_rollouts_inverse_probability(
        p_hat: np.ndarray, total_budget: int, n_min: int,
        n_max: int) -> np.ndarray:
    """Allocate proportionally to inverse pass rate under exact box constraints."""
    p_hat = _validate_rollout_budget_inputs(p_hat, total_budget, n_min, n_max)
    if len(p_hat) == 0:
        return np.empty(0, dtype=int)

    raw = 1.0 / np.maximum(p_hat, 1.0 / n_max)
    scaled = raw / raw.sum() * total_budget
    n = np.clip(np.round(scaled), n_min, n_max).astype(int)

    # Settle rounding/clipping drift while preserving the hard bounds.
    order = np.argsort(-raw, kind="stable")
    while n.sum() > total_budget:
        movable = [i for i in order[::-1] if n[i] > n_min]
        n[movable[0]] -= 1
    while n.sum() < total_budget:
        movable = [i for i in order if n[i] < n_max]
        n[movable[0]] += 1
    return n


def allocate_rollouts_greedy(p_hat: np.ndarray, total_budget: int,
                             n_min: int = 4, n_max: int = 64) -> np.ndarray:
    """Optimal rollout allocation for the advantage-mass objective.

    Maximizing sum_i [(1-(1-p_i)^{N_i}) - p_i] s.t. sum N_i = B is concave in
    each N_i, so greedy water-filling is optimal: the marginal mass of the
    (N+1)-th rollout on prompt i is p_i(1-p_i)^N — the probability that
    rollout is the group's *first success*.  Repeatedly award the next
    rollout to the prompt with the largest marginal.
    """
    p_hat = _validate_rollout_budget_inputs(p_hat, total_budget, n_min, n_max)
    m = len(p_hat)
    if m == 0:
        return np.empty(0, dtype=int)
    # Use the supplied probabilities exactly. Clipping changes marginal
    # ordering and can violate the water-filling optimality claim at p=0/1.
    p = p_hat
    n = np.full(m, n_min, dtype=int)
    remaining = total_budget - n_min * m
    marginal = p * (1 - p) ** n  # value of the next rollout per prompt
    for _ in range(remaining):
        i = int(np.argmax(marginal))
        n[i] += 1
        if n[i] >= n_max:
            marginal[i] = -1.0
        else:
            marginal[i] = p[i] * (1 - p[i]) ** n[i]
    return n


def allocate_rollouts_adaptive(teacher: Teacher, task_ids: np.ndarray,
                               total_budget: int, n_min: int = 4,
                               n_max: int = 64) -> np.ndarray:
    """Compute-indexed curriculum: split a fixed rollout budget across the
    selected tasks so that harder tasks get more rollouts.

    Practical Algorithm 1 has expected truncation order N-1, so giving a hard
    task a larger N raises both its chance of a live group and its likelihood
    approximation order. This historical heuristic allocates
    ~1/max(p_hat, 1/n_max); use ``allocate_rollouts_greedy`` for the exact
    coefficient-mass optimum.
    """
    task_ids = np.asarray(task_ids)
    if task_ids.ndim != 1:
        raise ValueError(f"task_ids must be one-dimensional, got shape {task_ids.shape}")
    if not np.issubdtype(task_ids.dtype, np.integer):
        raise ValueError("task_ids must contain integers")
    if np.any((task_ids < 0) | (task_ids >= teacher.n_tasks)):
        raise ValueError(f"task_ids must lie in [0, {teacher.n_tasks})")

    p_hat = np.array([
        teacher.stats[int(t)].ema_pass
        if teacher.stats[int(t)].ema_initialized else 0.5
        for t in task_ids
    ], dtype=float)
    return _allocate_rollouts_inverse_probability(
        p_hat, total_budget, n_min, n_max
    )
