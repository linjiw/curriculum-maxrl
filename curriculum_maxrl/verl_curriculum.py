"""Compatibility surface for the canonical verl integration.

The production implementation lives in ``verl_integration/curriculum.py``.
Re-exporting it here prevents the research prototype and the copy-to-verl
module from drifting apart.  ``allocate_rollout_budget`` is retained only for
historical phase-2 experiments; it is the old ``1/p`` heuristic, not the
proved coefficient-mass water-filling rule in ``teachers.py``.
"""

from __future__ import annotations

import numpy as np

from verl_integration.curriculum import (
    CurriculumIndexedDataset,
    CurriculumSampler,
    FrontierTeacher,
)

__all__ = [
    "CurriculumIndexedDataset",
    "CurriculumSampler",
    "FrontierTeacher",
    "allocate_rollout_budget",
]


def allocate_rollout_budget(p_hat: np.ndarray, total_budget: int,
                            n_min: int = 4, n_max: int = 64) -> np.ndarray:
    """Historical ``N_i ∝ 1/p̂_i`` heuristic within integer bounds.

    Use ``curriculum_maxrl.teachers.allocate_rollouts_greedy`` for the exact
    fixed-pass-rate, one-step coefficient-mass allocation proved in P3.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.ndim != 1 or len(p_hat) == 0:
        raise ValueError("p_hat must be a non-empty vector")
    if not np.isfinite(p_hat).all():
        raise ValueError("p_hat must be finite")
    if n_min < 1 or n_max < n_min:
        raise ValueError("require 1 <= n_min <= n_max")
    if not (len(p_hat) * n_min <= total_budget <= len(p_hat) * n_max):
        raise ValueError("total_budget is infeasible under the supplied bounds")

    raw = 1.0 / np.maximum(p_hat, 1.0 / n_max)
    scaled = raw / raw.sum() * total_budget
    allocation = np.clip(np.round(scaled), n_min, n_max).astype(int)
    order = np.argsort(-raw)
    while allocation.sum() > total_budget:
        movable = [i for i in order[::-1] if allocation[i] > n_min]
        allocation[movable[0]] -= 1
    while allocation.sum() < total_budget:
        movable = [i for i in order if allocation[i] < n_max]
        allocation[movable[0]] += 1
    return allocation
