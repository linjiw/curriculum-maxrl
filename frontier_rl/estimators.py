"""Group advantage weights. MaxRL is the framework's estimator (P5: ~N x
RLOO's signal on frontier tasks; H6: the only one of the three that is safe
under a frontier curriculum). GRPO/RLOO are included for baselines."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def maxrl_weights(rewards: np.ndarray, positive_part: bool = False) -> np.ndarray:
    """w_i = r_i/K − 1/N; zero vector when K = 0 (paper eq. 10).

    positive_part=True keeps only the success weights (1/K − 1/N, failures 0)
    — the weighted-RFT estimator for policies without tractable per-sample
    log-probs (flow/diffusion action heads, weighted SFT: COSMOS3_RESPONSE.md
    Q1). Two exact properties (MC-verified in test_framework.py):

      E[Σ w⁺·S] = Σ_{k=2}^{N} (1/k)·∇pass@k   (unbiased for the pass@k tail
                                               objective; the dropped failure
                                               term is a zero-mean baseline)
      E[Σ w⁺]   = pass@N − pass@1              (the teacher utility u(p) —
                                               P1 governs sampling exactly)

    All-pass groups (K=N) self-retire: every weight is 0.
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
