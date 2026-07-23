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


def maxrl_weights(rewards: np.ndarray, positive_part: bool = False) -> np.ndarray:
    """Practical dropped-group MaxRL weights (paper Algorithm 1).

    ``w_i = 1{K>0}(r_i/K - 1/N)``.  Although the nonzero-group expression
    matches Eq. 10, dropping its control variate at ``K=0`` changes the
    population objective from truncation order ``N`` to order ``N-1``.

    positive_part=True keeps only the success weights (1/K − 1/N, failures 0)
    — the weighted-RFT estimator for policies without tractable per-sample
    log-probs (flow/diffusion action heads, weighted SFT: COSMOS3_RESPONSE.md
    Q1). Two exact properties (MC-verified in test_framework.py):

      E[Σ w⁺·S] = Σ_{k=2}^{N} (1/k)·∇pass@k   (unbiased for the pass@k tail
                                               objective; the dropped failure
                                               term is a zero-mean baseline)
      E[Σ w⁺]   = pass@N − pass@1              (the teacher utility u(p) —
                                               P1 governs sampling exactly)

    All-pass groups (K=N) self-retire: every weight is 0.  Note this is a
    different object from ``maxrl_success_weights`` (r/K, Eq. 9's raw
    average): positive-part keeps Eq. 10's *centered* success weight.
    """
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.zeros(n)
    w = rewards / k - 1.0 / n
    if positive_part:
        w = np.where(rewards > 0, w, 0.0)
    return w


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
