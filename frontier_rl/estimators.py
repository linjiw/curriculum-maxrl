"""Group-level scalar weights used by the training framework.

There are three closely related MaxRL estimators worth keeping distinct:

``maxrl_success_weights``
    The paper's raw success-average estimator (Eq. 9).  It is unbiased for
    truncation order ``T=N``.
``maxrl_unbiased_cv_weights``
    Eq. 10 with its zero-mean score control variate retained even when every
    rollout fails.  It is also unbiased for ``T=N``.
``maxrl_weights``
    The practical Algorithm-1 form used by this project: both terms are
    dropped when ``K=0``.  This is unbiased for order ``T=N-1`` (not ``N``)
    and has zero scalar weights on all-fail and all-pass groups.

GRPO and RLOO are included as baselines.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def maxrl_weights(rewards: np.ndarray) -> np.ndarray:
    """Practical dropped-group MaxRL weights (paper Algorithm 1).

    ``w_i = 1{K>0}(r_i/K - 1/N)``.  Although the nonzero-group expression
    matches Eq. 10, dropping its control variate at ``K=0`` changes the
    population objective from truncation order ``N`` to order ``N-1``.
    """
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.zeros(n)
    return rewards / k - 1.0 / n


def maxrl_success_weights(rewards: np.ndarray) -> np.ndarray:
    """Raw success-average weights from MaxRL Eq. 9 (unbiased for ``T=N``)."""
    k = rewards.sum()
    if k == 0:
        return np.zeros(len(rewards))
    return rewards / k


def maxrl_unbiased_cv_weights(rewards: np.ndarray) -> np.ndarray:
    """MaxRL Eq. 10, retaining the control variate on all-fail groups.

    For ``K=0`` this deliberately returns ``-1/N`` for every rollout.  The
    corresponding average score has zero *unconditional* expectation; setting
    it to zero only on all-fail groups makes it outcome-dependent and loses
    the order-``N`` unbiasedness guarantee.
    """
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.full(n, -1.0 / n)
    return rewards / k - 1.0 / n


def rloo_weights(rewards: np.ndarray) -> np.ndarray:
    n = len(rewards)
    if n < 2:
        return rewards.copy()
    loo = (rewards.sum() - rewards) / (n - 1)
    return (rewards - loo) / n


def grpo_weights(rewards: np.ndarray) -> np.ndarray:
    n = len(rewards)
    std = rewards.std(ddof=1) if n > 1 else 1.0
    return (rewards - rewards.mean()) / (std + EPS) / n
