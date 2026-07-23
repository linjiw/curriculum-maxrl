"""Group advantage weights.

``maxrl_weights`` matches the paper's practical Algorithm 1: it uses the
Eq. (10) centered coefficients on live groups and drops all-fail groups. That
drop makes its expected population weight the order-(N-1) MaxRL weight, not
the order-N weight of the success-only estimator in Eq. (9). The explicit
``maxrl_eq10_weights`` helper retains the control variate on all-fail groups
and is unbiased for order N. See PROOFS.md Proposition 0.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def maxrl_weights(rewards: np.ndarray) -> np.ndarray:
    """Practical Algorithm 1 weights; zero vector when K = 0.

    For N >= 2 their expected gradient is the order-(N-1) truncated MaxRL
    gradient. This is the estimator used by the repository's experiments.
    """
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.zeros(n)
    return rewards / k - 1.0 / n


def maxrl_eq10_weights(rewards: np.ndarray) -> np.ndarray:
    """Exact Eq. (10) control-variate weights.

    The success-average term is zero when K=0, but the unconditional
    ``-1/N`` score control remains. Its expectation is therefore the same as
    Eq. (9): the order-N truncated MaxRL gradient.
    """
    n = len(rewards)
    k = rewards.sum()
    weights = np.full(n, -1.0 / n)
    if k > 0:
        weights += rewards / k
    return weights


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
