"""Count-law activity from four sufficient statistics per curriculum unit.

``group_law_teacher.GroupLawPosterior`` keeps a decayed Dirichlet over the N+1
bins of the success count.  That is the reference semantics, but it is a poor
fit for a jitted trainer: the state is ``(n_units, N+1)`` and every update
touches a bin chosen at runtime.

For the two estimators a trainer actually deploys, the full law is unnecessary.
Realized mass is a low-order polynomial in the success count,

    M_MaxRL(k) = 2(1 - k/N) 1{k>0}   = 2*1{k>0} - (2/N)*k        affine on k>0
    M_RLOO(k)  = 2k(N-k)/(N(N-1))    = (2/(N(N-1)))*(N*k - k^2)  quadratic

so ``A_E(z) = sum_k P(K=k|z) M_E(k)`` is a linear functional of a handful of
moments of the count law.  Carrying those moments directly gives exact
agreement with the Dirichlet at a fixed four floats per unit:

    W   = total posterior mass          (pseudo-groups + decayed real groups)
    Z   = mass at k = 0                 -> P(K=0)
    S   = sum of k weighted by mass     -> E[K]
    S2  = sum of k^2 weighted by mass   -> E[K^2]   (RLOO only)

    A_MaxRL = 2(1 - Z/W - S/(W*N))
    A_RLOO  = 2(N*S - S2)/(W*N*(N-1))

GRPO's mass carries a ``sqrt(k(N-k))`` factor, which is not polynomial, so it
has no finite-moment reduction; score it with ``GroupLawPosterior`` instead.

Prior.  As in the reference teacher, the prior is the i.i.d. count law at
``p0`` carrying ``prior_mass`` pseudo-groups.  With no data the count-law score
and the plug-in score therefore agree exactly, and they separate only as
evidence of non-Binomial structure arrives.

Backends.  Every function here is elementwise arithmetic on array-likes, so the
same code runs under numpy and under ``jax.numpy``.  Pass ``xp=jnp`` to score
inside a jitted trainer; ``N`` is a Python int and stays static.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "prior_stats",
    "observe",
    "activity",
    "mean_pass_rate",
    "plugin_activity",
    "granularity_gap",
    "CountLawStats",
]

ESTIMATORS = ("maxrl", "rloo")


def prior_stats(n_rollouts: int, p0: float = 0.5, prior_mass: float = 1.0):
    """Moments of ``prior_mass`` pseudo-groups drawn i.i.d. at ``p0``.

    Returns ``(W, Z, S, S2)`` for K ~ Binomial(N, p0):
    E[K] = N p0, Var[K] = N p0 (1-p0), so E[K^2] = N p0 (1-p0) + (N p0)^2.
    """
    n = int(n_rollouts)
    if n < 2:
        raise ValueError("n_rollouts must be at least 2")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be a probability")
    if prior_mass < 0.0:
        raise ValueError("prior_mass must be non-negative")
    m0 = float(prior_mass)
    ek = n * p0
    return (
        m0,                                    # W
        m0 * (1.0 - p0) ** n,                  # Z
        m0 * ek,                               # S
        m0 * (n * p0 * (1.0 - p0) + ek * ek),  # S2
    )


def observe(stats, k, n_rollouts: int, p0: float = 0.5, prior_mass: float = 1.0,
            decay: float = 1.0, xp=np):
    """Fold one closed group of ``n_rollouts`` with ``k`` successes into ``stats``.

    Matches ``GroupLawPosterior.observe``: the *excess over the prior* decays,
    the prior itself does not.  ``k`` may be an array to update many units at
    once; ``stats`` is a 4-tuple of matching arrays.
    """
    W, Z, S, S2 = stats
    pW, pZ, pS, pS2 = prior_stats(n_rollouts, p0, prior_mass)
    d = float(decay)
    if not 0.0 < d <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    k = xp.asarray(k)
    is_zero = (k == 0)
    # decay the accumulated excess, then add this group
    W = pW + (W - pW) * d + 1.0
    Z = pZ + (Z - pZ) * d + xp.where(is_zero, 1.0, 0.0)
    S = pS + (S - pS) * d + k
    S2 = pS2 + (S2 - pS2) * d + k * k
    return (W, Z, S, S2)


def activity(stats, n_rollouts: int, estimator: str = "maxrl", xp=np):
    """``A_E(z) = sum_k P(K=k|z) M_E(k)`` -- exact, from the moments."""
    W, Z, S, S2 = stats
    n = int(n_rollouts)
    if estimator == "maxrl":
        # 2(P(K>0) - E[K]/N)
        return 2.0 * (1.0 - Z / W - S / (W * n))
    if estimator == "rloo":
        return 2.0 * (n * S - S2) / (W * n * (n - 1))
    raise ValueError(
        f"unknown or unsupported estimator {estimator!r}; GRPO has no finite-moment "
        "reduction -- use group_law_teacher.GroupLawPosterior")


def mean_pass_rate(stats, n_rollouts: int, xp=np):
    W, _, S, _ = stats
    return S / (W * int(n_rollouts))


def plugin_activity(stats, n_rollouts: int, estimator: str = "maxrl", xp=np):
    """The condemned comparator: the i.i.d. curve evaluated at the unit's mean."""
    n = int(n_rollouts)
    p = mean_pass_rate(stats, n, xp=xp)
    if estimator == "maxrl":
        return 2.0 * (1.0 - (1.0 - p) ** n - p)
    if estimator == "rloo":
        # E[M_RLOO(K)] under Binomial(N, p) = 2p(1-p)(N-1)/N * N/(N-1) = 2p(1-p)
        return 2.0 * p * (1.0 - p)
    raise ValueError(f"unknown or unsupported estimator {estimator!r}")


def granularity_gap(stats, n_rollouts: int, estimator: str = "maxrl", xp=np):
    """Plug-in minus truth.

    For MaxRL this is exactly ``2[P(K=0) - (1-p_bar)^N]``.  The expression is
    nonnegative in the registered aggregation regime (a mixture of
    conditionally-i.i.d. atomic tasks), but not under every possible count law.
    """
    return (plugin_activity(stats, n_rollouts, estimator, xp=xp)
            - activity(stats, n_rollouts, estimator, xp=xp))


class CountLawStats:
    """Vectorised sufficient-statistic posterior over many curriculum units.

    A thin convenience wrapper over the free functions above, for the numpy
    (trainer-side) path.  Inside a jitted trainer, carry the 4-tuple yourself
    and call the free functions with ``xp=jnp``.
    """

    def __init__(self, n_units: int, n_rollouts: int, p0: float = 0.5,
                 prior_mass: float = 1.0, decay: float = 1.0):
        self.n = int(n_rollouts)
        self.p0 = float(p0)
        self.prior_mass = float(prior_mass)
        self.decay = float(decay)
        pW, pZ, pS, pS2 = prior_stats(self.n, p0, prior_mass)
        ones = np.ones(int(n_units), dtype=float)
        self.stats = (pW * ones, pZ * ones, pS * ones, pS2 * ones)

    def observe(self, unit: int, k: int) -> None:
        """Fold one closed group into a single unit's posterior."""
        if not 0 <= int(k) <= self.n:
            raise ValueError(f"k={k} outside [0, {self.n}]")
        one = tuple(a[unit] for a in self.stats)
        new = observe(one, int(k), self.n, self.p0, self.prior_mass, self.decay)
        for arr, v in zip(self.stats, new):
            arr[unit] = v

    def activity(self, estimator: str = "maxrl"):
        return activity(self.stats, self.n, estimator)

    def plugin_activity(self, estimator: str = "maxrl"):
        return plugin_activity(self.stats, self.n, estimator)

    def granularity_gap(self, estimator: str = "maxrl"):
        return granularity_gap(self.stats, self.n, estimator)

    def mean_pass_rate(self):
        return mean_pass_rate(self.stats, self.n)
