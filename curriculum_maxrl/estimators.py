"""Per-group advantage weights for REINFORCE / RLOO / GRPO / MaxRL.

Each function maps a binary reward vector r (n rollouts of one prompt) to
per-rollout scalar weights w such that the gradient estimate is
sum_j w_j * S_j, with S_j = grad log pi(z_j).  These mirror the formulas in
verl/trainer/ppo/core_algos.py of the MaxRL codebase, minus token masking.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def weights_reinforce(r: np.ndarray) -> np.ndarray:
    return r / len(r)


def weights_rloo(r: np.ndarray) -> np.ndarray:
    n = len(r)
    if n < 2:
        return r.copy()
    loo_mean = (r.sum() - r) / (n - 1)
    return (r - loo_mean) / n


def weights_grpo(r: np.ndarray) -> np.ndarray:
    n = len(r)
    std = r.std(ddof=1) if n > 1 else 1.0
    return (r - r.mean()) / (std + EPS) / n


def weights_maxrl(r: np.ndarray) -> np.ndarray:
    """Practical Algorithm 1 estimator from the paper.

    ``w_j = r_j/K - 1/N`` on live groups; the whole group is dropped when
    K=0. Dropping the Eq. (10) control term on all-fail groups changes the
    expected population weight from order N to order N-1. This function is
    retained because it is the implementation used for the reported runs.
    """
    n = len(r)
    k = r.sum()
    if k == 0:
        return np.zeros(n)
    return r / k - 1.0 / n


def weights_maxrl_eq10(r: np.ndarray) -> np.ndarray:
    """Exact variance-reduced estimator from paper Eq. (10).

    The success-average term is zero at K=0, while the unconditional average
    score control remains. Unlike practical Algorithm 1, this is unbiased for
    the order-N truncated MaxRL objective.
    """
    n = len(r)
    k = r.sum()
    weights = np.full(n, -1.0 / n)
    if k > 0:
        weights += r / k
    return weights


def _c_TN(K: int, N: int, T: int) -> float:
    """Per-success weight of the subset estimator (maclaurin.py c_sub_TN,
    paper appendix eq. 51): its success term is unbiased for the T-truncated
    objective with N rollouts, any T <= N. c_{N,N}(K) = 1/K recovers the
    success-average term in Eq. (9)."""
    from math import lgamma, log, exp
    if K == 0 or T <= 0:
        return 0.0

    def logcomb(a, kk):
        if kk < 0 or a < kk or a < 0:
            return float("-inf")
        return lgamma(a + 1) - lgamma(kk + 1) - lgamma(a - kk + 1)

    F = N - K
    logC_NT = logcomb(N, T)
    s = 0.0
    for k in range(1, min(T, K) + 1):
        lt = logcomb(K - 1, k - 1) + logcomb(F, T - k) - logC_NT - log(k)
        if lt > float("-inf"):
            s += exp(lt)
    return s


def weights_maxrl_t(r: np.ndarray, T: int) -> np.ndarray:
    """Legacy dropped-group subset estimator used by the adaptive-T runs.

    w_succ = c_{T,N}(K) - 1/N, w_fail = -1/N (same zero-mean control variate
    as Eq. (10)); the group is dropped at K=0. T=N recovers
    ``weights_maxrl``. Because the control is conditionally dropped, this is
    not unbiased for order T: its population multiplier is
    ``w_T(p) - (1-p)^(N-1)``.
    """
    n = len(r)
    k = int(r.sum())
    if k == 0:
        return np.zeros(n)
    c = _c_TN(k, n, min(T, n))
    return r * c - 1.0 / n


def weights_maxrl_t_eq10(r: np.ndarray, T: int) -> np.ndarray:
    """Unbiased T-truncated subset estimator with the full Eq. (10) control."""
    n = len(r)
    k = int(r.sum())
    c = _c_TN(k, n, min(T, n))
    return r * c - 1.0 / n
