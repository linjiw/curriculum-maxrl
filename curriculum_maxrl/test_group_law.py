"""Contract tests for the group-law form of practical-MaxRL coefficient mass.

Proposition (arbitrary group law).  For ANY joint binary law Q on {0,1}^N,
with no independence and no identical distribution assumed,

    A_N(Q) = E_Q[ sum_i |w_i| ] = 2 ( Pr_Q(K>0) - E_Q[K]/N ).

Corollary (conditional-i.i.d. scalarization).  If Q = Bernoulli(p)^{otimes N}
then A_N(p) = 2{1 - p - (1-p)^N} = 2(pass@N - pass@1).

Corollary (task granularity).  If a curriculum scores an aggregate z whose
concrete tasks X ~ nu(.|z) each have conditionally i.i.d. rollouts, then
plugging the aggregate mean pass rate into A_N over-predicts, and

    A_N(p_bar_z) - 2 E_X[u_N(p_X)] = 2 [ Pr(K=0|z) - (1-p_bar_z)^N ] >= 0.

These are exercised against the deployed `weights_maxrl`, not a re-derivation,
so a change to the estimator convention breaks them.
"""
from itertools import product

import numpy as np
import pytest

from estimators import coefficient_activity, weights_maxrl


def realized_mass(r):
    return float(np.abs(weights_maxrl(np.asarray(r, dtype=float))).sum())


def exact_mass_under_law(n, prob):
    """E[sum|w|], Pr(K>0), E[K]/N by exhaustive enumeration of {0,1}^n."""
    tot = q = ek = 0.0
    for outcome in product((0, 1), repeat=n):
        w = prob(outcome)
        if w == 0.0:
            continue
        k = sum(outcome)
        tot += w * realized_mass(outcome)
        q += w * (k > 0)
        ek += w * k / n
    return tot, q, ek


# --------------------------------------------------------------- mass form
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("k", range(0, 7))
def test_realized_mass_is_2_one_minus_k_over_n(n, k):
    """M(K) = 2(1 - K/N) for K>0, and 0 for K=0 -- the whole basis."""
    if k > n:
        pytest.skip("k <= n")
    r = [1] * k + [0] * (n - k)
    expected = 0.0 if k == 0 else 2.0 * (1.0 - k / n)
    assert realized_mass(r) == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------- arbitrary (dependent) laws
def _laws(n, rng):
    """A deliberately nasty spread of joint laws, most of them dependent."""
    # 1. fully dependent: all-0 or all-1
    yield lambda o, n=n: (0.3 if sum(o) == 0 else 0.7 if sum(o) == n else 0.0)
    # 2. exchangeable mixture of Bernoullis (beta-binomial-like)
    ps = [0.05, 0.4, 0.9]
    wts = [0.5, 0.2, 0.3]
    def mix(o, ps=ps, wts=wts):
        return sum(w * np.prod([p if x else 1 - p for x in o])
                   for w, p in zip(wts, ps))
    yield mix
    # 3. anti-correlated pair structure: first rollout flips the rest
    def anti(o, n=n):
        head, tail = o[0], o[1:]
        p_tail = 0.2 if head else 0.8
        return 0.5 * np.prod([p_tail if x else 1 - p_tail for x in tail])
    yield anti
    # 4. non-identically distributed independent rollouts
    marg = rng.uniform(0.05, 0.95, size=n)
    def hetero(o, marg=marg):
        return float(np.prod([m if x else 1 - m for m, x in zip(marg, o)]))
    yield hetero
    # 5. random dense joint law
    raw = rng.random(2 ** n) + 1e-3
    raw /= raw.sum()
    idx = {o: i for i, o in enumerate(product((0, 1), repeat=n))}
    yield lambda o, raw=raw, idx=idx: float(raw[idx[o]])


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_mass_identity_holds_for_arbitrary_group_laws(n):
    rng = np.random.default_rng(20260818 + n)
    for law in _laws(n, rng):
        total = sum(law(o) for o in product((0, 1), repeat=n))
        assert total == pytest.approx(1.0, abs=1e-9), "law must be normalized"
        mass, q, ek = exact_mass_under_law(n, law)
        assert mass == pytest.approx(2.0 * (q - ek), abs=1e-12)


# --------------------------------------------------- i.i.d. reduction parity
@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.37, 0.5, 0.9, 1.0])
def test_iid_reduction_matches_closed_form(n, p):
    def iid(o, p=p):
        return float(np.prod([p if x else 1 - p for x in o]))
    mass, _, _ = exact_mass_under_law(n, iid)
    assert mass == pytest.approx(2.0 * coefficient_activity(p, n), abs=1e-12)
    assert mass == pytest.approx(2.0 * (1 - p - (1 - p) ** n), abs=1e-12)


# ------------------------------------------------------ task-granularity gap
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
def test_granularity_gap_equals_twice_excess_all_fail(n):
    """The plug-in over-prediction is exactly 2 x excess silence, and >= 0."""
    rng = np.random.default_rng(4242 + n)
    for _ in range(40):
        m = rng.integers(2, 8)
        w = rng.random(m); w /= w.sum()
        px = rng.random(m)
        p_bar = float(w @ px)
        true_half = float(w @ (1 - px - (1 - px) ** n))       # E_X[u_N(p_X)]
        plug = float(coefficient_activity(p_bar, n))          # u_N(p_bar)
        pr_silent = float(w @ (1 - px) ** n)                  # Pr(K=0 | z)
        gap = 2 * plug - 2 * true_half
        assert gap == pytest.approx(
            2.0 * (pr_silent - (1 - p_bar) ** n), abs=1e-12)
        assert gap >= -1e-12, "Jensen: plug-in can only over-predict"


def test_granularity_gap_vanishes_without_heterogeneity():
    """A homogeneous aggregate pays nothing: the reduction is then exact."""
    for n in (2, 8, 32):
        for p in (0.02, 0.3, 0.8):
            w = np.array([0.25, 0.25, 0.5])
            px = np.full(3, p)
            p_bar = float(w @ px)
            gap = 2 * (float(coefficient_activity(p_bar, n))
                       - float(w @ (1 - px - (1 - px) ** n)))
            assert gap == pytest.approx(0.0, abs=1e-15)


def test_harder_peaked_score_pays_more_for_the_same_heterogeneity():
    """At its own peak, u_32 is far more curvature-exposed than u_2=p(1-p).

    Not a claim about any experiment; it is the curvature statement behind
    Cor. 2, checked numerically.
    """
    def peak(n):
        return 1.0 - n ** (-1.0 / (n - 1))
    spread = 0.05
    for n in (16, 32, 64):
        pk = peak(n)
        w = np.array([0.5, 0.5])
        px = np.clip(np.array([pk - spread, pk + spread]), 1e-9, 1 - 1e-9)
        p_bar = float(w @ px)
        gap_n = 2 * (float(coefficient_activity(p_bar, n))
                     - float(w @ (1 - px - (1 - px) ** n)))
        # same absolute heterogeneity placed at p(1-p)'s own peak
        px2 = np.array([0.5 - spread, 0.5 + spread])
        gap_2 = 2 * (float(coefficient_activity(0.5, 2))
                     - float(w @ (1 - px2 - (1 - px2) ** 2)))
        assert gap_n > gap_2
