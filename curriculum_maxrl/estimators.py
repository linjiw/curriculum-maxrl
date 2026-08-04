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


def weights_maxrl_raw(r: np.ndarray) -> np.ndarray:
    """Raw success-conditioned MaxRL estimator (Theorem 2 / Eq. 9).

    The group contributes the average score of its successful rollouts and
    is zero when no success is observed. With ``N`` rollouts this is
    unbiased for the ``T=N`` truncated objective, but it is not centered and
    therefore does not retire all-success groups.
    """
    k = r.sum()
    if k == 0:
        return np.zeros(len(r))
    return r / k


def weights_maxrl_full_cv(r: np.ndarray) -> np.ndarray:
    """Full variance-reduced MaxRL estimator (Eq. 10).

    Unlike the practical Algorithm-1 convention, the zero-mean control
    variate ``-1/N`` is retained on all-fail groups. Such groups therefore
    emit a negative-only sample update. The estimator remains unbiased for
    the ``T=N`` truncated objective.
    """
    n = len(r)
    k = r.sum()
    if k == 0:
        return np.full(n, -1.0 / n)
    return r / k - 1.0 / n


def weights_maxrl(r: np.ndarray) -> np.ndarray:
    """Practical centered/drop convention used by the local experiments.

    w_j = (r_j / K - 1/N); the whole group is dropped when K = 0.
    Unbiased for the truncated ML objective with T = N-1 (dropping the
    K=0 control variate shifts the order; see PROOFS.md Prop 1 correction).
    This is deliberately distinct from :func:`weights_maxrl_full_cv`, which
    retains the Eq.-10 control variate on all-fail groups.
    """
    n = len(r)
    k = r.sum()
    if k == 0:
        return np.zeros(n)
    return r / k - 1.0 / n


def _c_TN(K: int, N: int, T: int) -> float:
    """Per-success weight of the subset estimator (maclaurin.py c_sub_TN,
    paper appendix eq. 51): unbiased for the T-truncated objective with N
    rollouts, any T <= N.  c_{N,N}(K) = 1/K recovers Algorithm 1."""
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
    """Full-CV subset estimator with decoupled order ``1 <= T <= N``.

    w_succ = c_{T,N}(K) - 1/N, w_fail = -1/N (same zero-mean control variate
    as eq. 10), including at K = 0.  T = N recovers
    :func:`weights_maxrl_full_cv`; T = 1 is the control-variate form of plain
    REINFORCE and has the same expected gradient. Retaining the K=0 baseline
    is required for unbiasedness at every T.
    """
    n = len(r)
    if not 1 <= T <= n:
        raise ValueError(f"T must satisfy 1 <= T <= N={n}; got {T}")
    k = int(r.sum())
    if k == 0:
        return np.full(n, -1.0 / n)
    c = _c_TN(k, n, T)
    return r * c - 1.0 / n
