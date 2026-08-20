"""Conformance: the four-moment scorer must equal the reference Dirichlet exactly.

``count_law_stats`` is the form a jitted trainer carries; ``group_law_teacher.
GroupLawPosterior`` is the reference semantics.  For MaxRL and RLOO they must
agree to floating point on every stream, including decayed and non-Binomial
ones.  If this file fails, the deployed score is not the theory's score.
"""
from __future__ import annotations

import numpy as np
import pytest

from . import count_law_stats as cls
from .group_law_teacher import GroupLawPosterior, iid_activity

TOL = 1e-12
ESTIMATORS = ("maxrl", "rloo")


def _streams(rng, n, n_groups):
    """A batch of deliberately non-Binomial count streams."""
    return {
        "all_fail": [0] * n_groups,
        "all_pass": [n] * n_groups,
        "unanimous_mixed": [0 if i % 2 else n for i in range(n_groups)],  # Fig 1 Level B
        "frontier": [1] * n_groups,
        "binomial_p50": list(rng.binomial(n, 0.5, n_groups)),
        "binomial_p05": list(rng.binomial(n, 0.05, n_groups)),
        "bimodal": list(rng.binomial(n, rng.choice([0.02, 0.98], n_groups))),
        "uniform_k": list(rng.integers(0, n + 1, n_groups)),
    }


@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("decay", [1.0, 0.7, 0.35])
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_matches_reference_dirichlet(n, decay, estimator):
    rng = np.random.default_rng(0xC0FFEE + n)
    for name, ks in _streams(rng, n, 12).items():
        ref = GroupLawPosterior(n, p0=0.5, prior_mass=1.0, decay=decay)
        st = cls.prior_stats(n, p0=0.5, prior_mass=1.0)
        # agree before any evidence
        assert abs(cls.activity(st, n, estimator)
                   - ref.activity(estimator)) < TOL, f"{name}: prior mismatch"
        for k in ks:
            ref.observe(k)
            st = cls.observe(st, k, n, p0=0.5, prior_mass=1.0, decay=decay)
            got, want = cls.activity(st, n, estimator), ref.activity(estimator)
            assert abs(got - want) < TOL, (
                f"{name} n={n} decay={decay} {estimator}: {got!r} != {want!r}")
            assert abs(cls.mean_pass_rate(st, n) - ref.mean_pass_rate()) < TOL
            assert abs(cls.granularity_gap(st, n, estimator)
                       - ref.granularity_gap(estimator)) < TOL


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_prior_equals_plugin_with_no_evidence(n):
    """With no data the count law and the plug-in must be the same number."""
    for p0 in (0.05, 0.5, 0.9):
        st = cls.prior_stats(n, p0=p0, prior_mass=1.0)
        for est in ESTIMATORS:
            assert abs(cls.activity(st, n, est)
                       - cls.plugin_activity(st, n, est)) < TOL
            assert abs(cls.granularity_gap(st, n, est)) < TOL
        # and it is the i.i.d. curve at p0
        assert abs(cls.activity(st, n, "maxrl") - iid_activity(p0, n, "maxrl")) < TOL


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_level_b_has_zero_activity_and_maximal_gap(n):
    """Fig. 1's counterexample, in the deployed estimator's own arithmetic.

    A unit whose groups are unanimous -- half all-pass, half all-fail -- has
    mean pass rate 1/2 and realized activity zero, while the plug-in scores it
    near the frontier maximum.
    """
    st = cls.prior_stats(n, p0=0.5, prior_mass=0.0)  # no prior, pure evidence
    for i in range(200):
        st = cls.observe(st, 0 if i % 2 else n, n, p0=0.5, prior_mass=0.0)
    assert abs(cls.mean_pass_rate(st, n) - 0.5) < TOL
    for est in ESTIMATORS:
        assert abs(cls.activity(st, n, est)) < TOL, f"{est} activity must vanish"
    # the gap is exactly twice the excess all-fail probability
    gap = cls.granularity_gap(st, n, "maxrl")
    assert abs(gap - 2.0 * (0.5 - 0.5 ** n)) < TOL


@pytest.mark.parametrize("n", [8, 16])
def test_gap_is_nonnegative_for_binomial_mixtures(n):
    """The corollary's sign, in the regime it is stated for.

    A coarse unit aggregates atomic instances: one instance is drawn per group
    and shared by its N members, so the count law is a *mixture* of Binomials.
    Then Pr(K=0) = E_x[(1-p_x)^N] >= (1-p_bar)^N by Jensen, and the plug-in
    never under-predicts.  This is the aggregation regime, and it is the only
    one in which the >= 0 claim may be made.

    The claim is about the *true* law, so build the stats from it exactly: a
    finite sample of groups fluctuates around it and its empirical gap may dip
    slightly negative, which is estimator variance, not a counterexample.
    """
    rng = np.random.default_rng(7)
    for _ in range(2000):
        m = int(rng.integers(1, 6))
        ps = rng.random(m)                       # atomic instances in the cell
        w = rng.dirichlet(np.ones(m))            # how often each is drawn
        # one instance per group, shared by all N members -> mixture of Binomials
        stats = (1.0,
                 float(w @ (1.0 - ps) ** n),                       # Pr(K=0)
                 float(w @ (n * ps)),                              # E[K]
                 float(w @ (n * ps * (1 - ps) + (n * ps) ** 2)))   # E[K^2]
        assert cls.granularity_gap(stats, n, "maxrl") >= -1e-15


@pytest.mark.parametrize("n", [8, 16])
def test_gap_inverts_under_anticorrelated_groups(n):
    """The boundary: outside the mixture regime the sign guarantee is false.

    The identity A_N(Q) = 2(Pr[K>0] - E[K]/N) needs no assumption at all, but
    the *sign* of the plug-in's error does.  A law that is under-dispersed at
    zero -- fewer all-fail groups than Binomial -- makes the plug-in
    UNDER-predict.  Recorded so the manuscript scopes the corollary to
    over-dispersed units rather than claiming it universally.
    """
    st = cls.prior_stats(n, p0=0.5, prior_mass=0.0)
    for _ in range(200):                        # every group has exactly one hit
        st = cls.observe(st, 1, n, p0=0.5, prior_mass=0.0)
    assert abs(cls.mean_pass_rate(st, n) - 1.0 / n) < TOL
    # Pr(K=0) is 0, but Binomial(N, 1/N) predicts ~1/e of groups all-fail
    gap = cls.granularity_gap(st, n, "maxrl")
    assert gap < -0.5, f"expected a large negative gap, got {gap}"
    assert abs(gap - 2.0 * (0.0 - (1.0 - 1.0 / n) ** n)) < TOL


def test_conditionally_iid_collapses_onto_the_plugin():
    """The theory's own boundary: an atomic, conditionally-i.i.d. unit gives a
    gap that vanishes as evidence accumulates, so at atomic units the count law
    buys nothing.  This is the reason a minimax arm must score a coarse unit."""
    n, p = 8, 0.3
    rng = np.random.default_rng(11)
    st = cls.prior_stats(n, p0=0.5, prior_mass=1.0)
    for k in rng.binomial(n, p, 4000):
        st = cls.observe(st, int(k), n, p0=0.5, prior_mass=1.0)
    assert abs(cls.granularity_gap(st, n, "maxrl")) < 0.02
    assert abs(cls.mean_pass_rate(st, n) - p) < 0.02


@pytest.mark.parametrize("n", [8, 32])
def test_vectorised_wrapper_matches_scalar_path(n):
    rng = np.random.default_rng(3)
    n_units = 6
    bank = cls.CountLawStats(n_units, n, p0=0.5, prior_mass=1.0, decay=0.7)
    refs = [GroupLawPosterior(n, p0=0.5, prior_mass=1.0, decay=0.7)
            for _ in range(n_units)]
    for _ in range(25):
        u = int(rng.integers(n_units))
        k = int(rng.integers(0, n + 1))
        bank.observe(u, k)
        refs[u].observe(k)
    for est in ESTIMATORS:
        got = bank.activity(est)
        want = np.array([r.activity(est) for r in refs])
        assert np.allclose(got, want, rtol=0, atol=TOL)


def test_jax_backend_agrees_when_available():
    """Same arithmetic under jax.numpy, at float32 tolerance."""
    jnp = pytest.importorskip("jax.numpy")
    n, rng = 8, np.random.default_rng(5)
    ks = [int(v) for v in rng.integers(0, n + 1, 30)]

    st_np = cls.prior_stats(n, p0=0.5, prior_mass=1.0)
    st_jx = tuple(jnp.asarray(v, dtype=jnp.float32) for v in st_np)
    for k in ks:
        st_np = cls.observe(st_np, k, n, decay=0.7)
        st_jx = cls.observe(st_jx, jnp.asarray(k, dtype=jnp.float32), n,
                            decay=0.7, xp=jnp)
    for est in ESTIMATORS:
        a_np = cls.activity(st_np, n, est)
        a_jx = float(cls.activity(st_jx, n, est, xp=jnp))
        assert abs(a_np - a_jx) < 1e-5, f"{est}: {a_np} vs {a_jx}"


def test_grpo_is_refused_with_a_pointer():
    with pytest.raises(ValueError, match="GroupLawPosterior"):
        cls.activity(cls.prior_stats(8), 8, "grpo")
