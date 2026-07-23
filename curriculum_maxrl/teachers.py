"""Curriculum teachers: map per-task statistics -> a sampling distribution.

All teachers see only what a real RLVR trainer sees: empirical rewards from
past rollouts (never the true pass rate). Statistics are tracked with an
exponential moving average plus discounted Beta pseudo-counts. These are not
an exact posterior for a moving policy.

Teachers
--------
UniformTeacher       no curriculum (baseline).
ZPDBandTeacher       samples tasks whose estimated pass rate lies in a target
                     band (ADARFT/DAPO-style difficulty targeting).
ALPTeacher           absolute-learning-progress bandit (TSCL / ALP-GMM style):
                     samples proportional to |d p̂ / dt|.
MaxRLFrontierTeacher legacy score pass@N*(1-p), equal to the exact practical
                     coefficient-mass family at index N+1.
AdvMassTeacher       exact half-mass score pass@N-p for N practical rollouts.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TaskStats:
    """Online statistics for one task, from observed rollout rewards only."""
    ema_pass: float = 0.0
    ema_initialized: bool = False
    alpha_beta: tuple[float, float] = (1.0, 1.0)  # discounted pseudo-counts
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
        # decay old evidence so pseudo-counts track the moving policy
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
    """Legacy near-neighbor of the exact coefficient-mass curriculum.

    For the MaxRL estimator with group size N, a prompt contributes gradient
    signal only when K >= 1, which happens with probability
    pass@N = 1 - (1-p)^N — exactly w_N(p) * p, the estimator's effective
    weight on the pass-rate gradient.  Utility:

        h_N(p) = (1 - (1-p)^N) * (1 - p) = u_{N+1}(p)

    It is exactly the practical mass family with an off-by-one group-size
    index. It is retained for historical comparisons; ``AdvMassTeacher`` is
    the exact N-rollout score. Discounted Beta pseudo-count draws provide
    Thompson-style randomized exploration.
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
    """Teacher driven by the exact expected scalar coefficient mass.

    THEORY.md section 2: for a group of N rollouts, the expected total
    |advantage| the MaxRL estimator emits on a prompt with pass rate p is
    2*(pass@N(p) - p) = 2*((1-(1-p)^N) - p) — the probability the prompt is
    solvable within N attempts but not within one. Sampling proportional to
    this quantity is a smooth stochastic priority rule; hard argmax would
    maximize known one-step mass absent coverage or variance constraints.
    Discounted Beta pseudo-count draws supply randomized exploration.

    ``power`` (VALIDATION.md V6): sample ∝ u^power. Learning compounds — steps on the
    highest-mass task unlock the next — so sharper concentration wins on
    chain-structured pools (γ≈4 saturates); use 1–2 on flat pools.
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


def allocate_rollouts_greedy(p_hat: np.ndarray, total_budget: int,
                             n_min: int = 4, n_max: int = 64) -> np.ndarray:
    """Exact myopic allocation for fixed supplied pass rates and bounds.

    With a feasible integer budget and fixed known/supplied p_i, maximizing
    half-mass sum_i u_{N_i}(p_i) (equivalently total expected coefficient mass
    2*sum_i u_{N_i}) is separable and discretely concave. Greedy water-filling
    is exact. The half-mass marginal is p_i(1-p_i)^N; total mass has an
    irrelevant factor two. This is not a long-horizon curriculum theorem.
    """
    p = np.asarray(p_hat, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_hat must be a one-dimensional array")
    if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("p_hat entries must be finite probabilities in [0, 1]")
    m = len(p)
    if m == 0:
        return np.array([], dtype=int)
    if (int(n_min) != n_min or int(n_max) != n_max
            or int(total_budget) != total_budget):
        raise ValueError("bounds and total_budget must be integers")
    n_min, n_max, total_budget = int(n_min), int(n_max), int(total_budget)
    if n_min < 1 or n_max < n_min:
        raise ValueError("require 1 <= n_min <= n_max")
    if not (m * n_min <= total_budget <= m * n_max):
        raise ValueError(
            "total_budget must satisfy len(p_hat)*n_min <= budget <= "
            "len(p_hat)*n_max"
        )
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
    """Historical 1/p̂ rollout heuristic, retained for old experiments.

    For practical drop-both MaxRL, effective order is N-1; raw/always-CV
    variants have order N. The proved one-step coefficient-mass rule is
    allocate_rollouts_greedy. This helper clips and renormalizes the older
    heuristic to the requested budget.
    """
    p_hat = np.array([
        max(teacher.stats[t].ema_pass if teacher.stats[t].ema_initialized else 0.5, 1e-3)
        for t in task_ids
    ])
    raw = 1.0 / np.maximum(p_hat, 1.0 / n_max)
    scaled = raw / raw.sum() * total_budget
    n = np.clip(np.round(scaled), n_min, n_max).astype(int)
    # settle rounding drift: take from easiest, give to hardest
    order = np.argsort(-raw)
    while n.sum() > total_budget:
        movable = [i for i in order[::-1] if n[i] > n_min]
        if not movable:
            break
        n[movable[0]] -= 1
    while n.sum() < total_budget:
        movable = [i for i in order if n[i] < n_max]
        if not movable:
            break
        n[movable[0]] += 1
    return n
