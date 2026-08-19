"""Contract tests for count-law activity estimation.

Four things must hold, or the method is not what the theory says it is:

1. M_E(k) matches the DEPLOYED estimators for every k and N.
2. The fast paths (affine for MaxRL, quadratic for RLOO) equal the full
   sum_k pi(k) M_E(k) on arbitrary laws.
3. Under conditionally i.i.d. data the estimate converges to the familiar
   p-curve -- the group-law teacher must not lose the atomic case.
4. On a heterogeneous unit it converges to the TRUE mixture activity while the
   plug-in converges to something larger, and the difference is exactly twice
   the excess all-fail probability.
"""
import numpy as np
import pytest

from estimators import weights_grpo, weights_maxrl, weights_rloo
from group_law_teacher import (GroupLawPosterior, iid_activity, mass_vector,
                               realized_mass)

DEPLOYED = {"maxrl": weights_maxrl, "rloo": weights_rloo, "grpo": weights_grpo}


# ---------------------------------------------------------------- 1. masses
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("est", ["maxrl", "rloo", "grpo"])
def test_mass_matches_deployed_estimator(n, est):
    for k in range(n + 1):
        r = np.array([1.0] * k + [0.0] * (n - k))
        deployed = float(np.abs(DEPLOYED[est](r)).sum())
        # GRPO's deployed form carries the EPS denominator stabilizer
        tol = 5e-6 if est == "grpo" else 1e-9
        assert realized_mass(k, n, est) == pytest.approx(deployed, abs=tol)


# ------------------------------------------------------------ 2. fast paths
@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("est", ["maxrl", "rloo", "grpo"])
def test_fast_path_equals_full_sum(n, est):
    rng = np.random.default_rng(20260819 + n)
    post = GroupLawPosterior(n)
    M = mass_vector(n, est)
    for _ in range(20):
        pi = rng.random(n + 1)
        pi /= pi.sum()
        assert post.activity(est, law=pi) == pytest.approx(float(pi @ M), abs=1e-12)


# --------------------------------------------------- 3. atomic i.i.d. limit
@pytest.mark.parametrize("p", [0.05, 0.2, 0.5, 0.8])
@pytest.mark.parametrize("est", ["maxrl", "rloo", "grpo"])
def test_converges_to_iid_curve_on_atomic_tasks(p, est):
    n = 16
    rng = np.random.default_rng(7)
    post = GroupLawPosterior(n, p0=0.5, prior_mass=1.0, decay=1.0)
    for _ in range(4000):
        post.observe(int(rng.binomial(n, p)))
    target = float(iid_activity(p, n, est))
    assert post.activity(est) == pytest.approx(target, abs=0.02)


def test_prior_reproduces_the_plugin_exactly():
    """With no data the two scores must agree, so the contrast is clean."""
    for n in (8, 16, 32):
        for p0 in (0.1, 0.5, 0.9):
            post = GroupLawPosterior(n, p0=p0)
            for est in ("maxrl", "rloo", "grpo"):
                assert post.activity(est) == pytest.approx(
                    post.plugin_activity(est), abs=1e-12)
            assert post.granularity_gap("maxrl") == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------- 4. heterogeneous unit: the gap
def test_heterogeneous_unit_recovers_truth_while_plugin_overshoots():
    """The counterexample: same mean pass rate, opposite activity."""
    n = 16
    rng = np.random.default_rng(11)
    # Level B: half the tasks mastered, half impossible.  Mean pass rate .5,
    # but every group is unanimous, so true activity is exactly zero.
    post = GroupLawPosterior(n, p0=0.5, prior_mass=1e-9, decay=1.0)
    for _ in range(4000):
        post.observe(n if rng.random() < 0.5 else 0)
    assert post.mean_pass_rate() == pytest.approx(0.5, abs=0.03)
    assert post.activity("maxrl") == pytest.approx(0.0, abs=1e-6)
    # the plug-in, seeing only the mean, reports the frontier value
    assert post.plugin_activity("maxrl") == pytest.approx(
        float(iid_activity(post.mean_pass_rate(), n, "maxrl")), abs=1e-12)
    assert post.plugin_activity("maxrl") > 0.9


@pytest.mark.parametrize("n", [8, 16, 32])
def test_gap_equals_twice_excess_all_fail(n):
    """plug-in - truth = 2[P(K=0) - (1-p_bar)^N], exactly, for MaxRL."""
    rng = np.random.default_rng(99 + n)
    for _ in range(30):
        pi = rng.random(n + 1)
        pi /= pi.sum()
        post = GroupLawPosterior(n)
        p_bar = float(pi @ np.arange(n + 1)) / n
        expected = 2.0 * (float(pi[0]) - (1.0 - p_bar) ** n)
        assert post.granularity_gap("maxrl", law=pi) == pytest.approx(
            expected, abs=1e-12)


def test_gap_is_nonnegative_for_iid_mixtures():
    """Jensen: mixing atomic tasks can only make the plug-in over-predict."""
    n = 16
    rng = np.random.default_rng(3)
    from math import comb
    for _ in range(50):
        m = rng.integers(2, 6)
        w = rng.random(m); w /= w.sum()
        px = rng.random(m)
        pi = np.zeros(n + 1)
        for wi, p in zip(w, px):
            pi += wi * np.array([comb(n, k) * p**k * (1 - p) ** (n - k)
                                 for k in range(n + 1)])
        post = GroupLawPosterior(n)
        assert post.granularity_gap("maxrl", law=pi) >= -1e-12
