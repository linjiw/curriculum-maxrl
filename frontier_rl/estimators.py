"""Binary group-estimator coefficients used by the framework.

The raw, full-control-variate, and practical MaxRL variants are deliberately
separate: they optimize closely related objectives but have different
finite-group activity profiles. GRPO and RLOO are included as baselines.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def maxrl_raw_weights(rewards: np.ndarray) -> np.ndarray:
    """Raw success-conditioned estimator; zero on all-fail groups."""
    k = rewards.sum()
    if k == 0:
        return np.zeros(len(rewards))
    return rewards / k


def maxrl_full_cv_weights(rewards: np.ndarray) -> np.ndarray:
    """Eq.-10 estimator with its control variate retained at ``K=0``.

    An all-fail group receives ``-1/N`` on every rollout rather than being
    dropped. This is the variance-reduced ``T=N`` estimator, distinct from
    the practical centered/drop convention in :func:`maxrl_weights`.
    """
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.full(n, -1.0 / n)
    return rewards / k - 1.0 / n


def maxrl_weights(rewards: np.ndarray, positive_part: bool = False) -> np.ndarray:
    """Practical centered/drop weights: ``r_i/K - 1/N`` for ``K>0``.

    positive_part=True keeps only the success weights (1/K − 1/N, failures
    0). This is a weighted-RFT surrogate for policies without tractable
    per-sample log-probabilities (for example, flow/diffusion action heads),
    not the centered estimator and not generally an unbiased gradient of the
    same objective. Its exact expected coefficient sum is
    ``pass@N - pass@1``; this coefficient statistic can guide allocation but
    does not establish the surrogate's full gradient geometry.

    The whole group is zeroed at ``K=0`` and self-retires at ``K=N``. This
    implementation is unbiased for truncation order ``T=N-1``; retaining the
    control variate at ``K=0`` instead gives :func:`maxrl_full_cv_weights`.
    """
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.zeros(n)
    w = rewards / k - 1.0 / n
    if positive_part:
        w = np.where(rewards > 0, w, 0.0)
    return w


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
