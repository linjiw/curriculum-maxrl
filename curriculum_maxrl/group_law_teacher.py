"""Group-law activity: score a curriculum unit by its success-count law.

Motivation.  For a permutation-equivariant binary group estimator the realized
absolute coefficient mass of a group depends only on its success count K, so
the activity of a curriculum unit z is

    A_E(z) = sum_k P(K=k | z) * M_E(k).                                    (1)

The sufficient statistic is the COUNT LAW, not the mean pass rate.  When the
scored unit is the atomic task and rollouts are conditionally i.i.d., P(K|z) is
Binomial(N, p) and (1) collapses to the familiar p-curves -- but that collapse
is an assumption about the unit, not a property of the estimator.  Plugging a
coarse unit's mean pass rate into the collapsed curve over-predicts its activity
by exactly twice its excess all-fail probability.

This module estimates (1) directly.  It costs nothing extra: K is already
observed for every group a trainer draws.

Conditioning.  A naive implementation keeps a Dirichlet over N+1 bins, which is
badly conditioned at N=32.  Two estimators do not need it:

    M_MaxRL(k) = 2(1 - k/N) 1{k>0}      affine on k>0  -> needs (P(K>0), E[K])
    M_RLOO(k)  = 2k(N-k)/(N(N-1))       quadratic      -> needs (E[K], E[K^2])
    M_GRPO(k)  = (2/N)sqrt((N-1)/N) sqrt(k(N-k))       -> needs the law

so MaxRL and RLOO get exact closed-form paths from two moments each, and only
GRPO pays for the full Dirichlet.  All three agree with the deployed
implementations in ``estimators.py`` (see ``test_group_law_teacher.py``).

The prior is the i.i.d. law at ``p0``.  With no data the group-law score and the
plug-in score therefore agree exactly; they separate only as evidence of
non-Binomial structure arrives.
"""
from __future__ import annotations

from math import comb, sqrt

import numpy as np

ESTIMATORS = ("maxrl", "rloo", "grpo")


def realized_mass(k: int, n_rollouts: int, estimator: str = "maxrl") -> float:
    """M_E(k): absolute coefficient mass of a group with k successes of N.

    Zero-stabilizer conventions, matching ``estimators.py``.
    """
    n = int(n_rollouts)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    if estimator == "maxrl":
        return 0.0 if k == 0 else 2.0 * (1.0 - k / n)
    if estimator == "rloo":
        return 2.0 * k * (n - k) / (n * (n - 1)) if n > 1 else 0.0
    if estimator == "grpo":
        return (2.0 / n) * sqrt((n - 1) / n) * sqrt(k * (n - k)) if n > 1 else 0.0
    raise ValueError(f"unknown estimator {estimator!r}")


def mass_vector(n_rollouts: int, estimator: str = "maxrl") -> np.ndarray:
    return np.array([realized_mass(k, n_rollouts, estimator)
                     for k in range(n_rollouts + 1)], dtype=float)


def iid_activity(p, n_rollouts: int, estimator: str = "maxrl"):
    """A_E under Binomial(N, p) -- the conditional-i.i.d. slice of (1)."""
    n = int(n_rollouts)
    p_arr = np.asarray(p, dtype=float)
    M = mass_vector(n, estimator)
    out = np.zeros_like(p_arr)
    for k in range(n + 1):
        out = out + comb(n, k) * p_arr**k * (1.0 - p_arr) ** (n - k) * M[k]
    return out


class GroupLawPosterior:
    """Decayed Dirichlet over the success count of one curriculum unit.

    The prior is the i.i.d. count law at ``p0`` carrying ``prior_mass``
    pseudo-groups, so an unvisited unit scores exactly like the plug-in.
    """

    def __init__(self, n_rollouts: int, p0: float = 0.5,
                 prior_mass: float = 1.0, decay: float = 0.7):
        if not 0.0 <= p0 <= 1.0:
            raise ValueError("p0 must be a probability")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.n = int(n_rollouts)
        self.decay = float(decay)
        self.prior = np.array(
            [prior_mass * comb(self.n, k) * p0**k * (1.0 - p0) ** (self.n - k)
             for k in range(self.n + 1)], dtype=float)
        self.alpha = self.prior.copy()

    def observe(self, k: int) -> None:
        """Fold in one group with k successes, decaying prior evidence."""
        if not 0 <= int(k) <= self.n:
            raise ValueError(f"k={k} outside [0, {self.n}]")
        self.alpha = self.prior + (self.alpha - self.prior) * self.decay
        self.alpha[int(k)] += 1.0

    # -- the count law ----------------------------------------------------
    def mean_law(self) -> np.ndarray:
        return self.alpha / self.alpha.sum()

    def sample_law(self, rng: np.random.Generator) -> np.ndarray:
        return rng.dirichlet(self.alpha)

    # -- functionals ------------------------------------------------------
    def activity(self, estimator: str = "maxrl",
                 law: np.ndarray | None = None) -> float:
        """A_E(z) from (1).  Exact for every estimator."""
        pi = self.mean_law() if law is None else law
        if estimator == "maxrl":
            # affine on k>0: 2(P(K>0) - E[K]/N)
            q = float(pi[1:].sum())
            ek = float(pi @ np.arange(self.n + 1))
            return 2.0 * (q - ek / self.n)
        if estimator == "rloo":
            k = np.arange(self.n + 1)
            ek = float(pi @ k)
            ek2 = float(pi @ (k * k))
            return 2.0 * (self.n * ek - ek2) / (self.n * (self.n - 1))
        return float(pi @ mass_vector(self.n, estimator))

    def mean_pass_rate(self, law: np.ndarray | None = None) -> float:
        pi = self.mean_law() if law is None else law
        return float(pi @ np.arange(self.n + 1)) / self.n

    def plugin_activity(self, estimator: str = "maxrl",
                        law: np.ndarray | None = None) -> float:
        """The naive comparator: evaluate the i.i.d. curve at the unit's mean."""
        return float(iid_activity(self.mean_pass_rate(law), self.n, estimator))

    def granularity_gap(self, estimator: str = "maxrl",
                        law: np.ndarray | None = None) -> float:
        """Plug-in minus truth.

        For MaxRL this is exactly ``2[P(K=0) - (1-p_bar)^N]``.  Its sign is
        nonnegative for mixtures of conditionally-i.i.d. atomic tasks, not for
        an arbitrary count law (under-dispersion at zero can reverse it).
        """
        return self.plugin_activity(estimator, law) - self.activity(estimator, law)
