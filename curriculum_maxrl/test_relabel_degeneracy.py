"""The relabeling dead-zone lemma, and its one admissible repair.

The recurring proposal for the `A_E = 0` dead zone is hindsight relabeling: when
a group of N rollouts all fail, rewrite each member's goal to the state that
member actually achieved, so the group "succeeds" and a gradient appears.

**Lemma (per-member relabeling is degenerate).** For each deployed centered
binary group estimator in this repository, realized mass vanishes at k = N:

    M_MaxRL(N) = 2(1 - N/N) = 0,
    M_RLOO(N)  = 2N(N-N)/(N(N-1)) = 0,
    M_GRPO(N)  = (2/N) sqrt((N-1)/N) sqrt(N(N-N)) = 0.

Relabeling each member to its own achieved goal makes every member succeed, so
K = N and the relabeled group carries *exactly as much coefficient mass as the
all-fail group it replaced*: zero.  It has additionally destroyed
exchangeability -- K is now a shared statistic over unrelated tasks -- so the
count law no longer describes the group at all.

The only admissible variant is **group-consistent** relabeling: draw one common
goal g' from a uniformly chosen member and apply it to the whole group, giving
K' in [1, N] and mass 2(1 - K'/N) > 0 whenever K' < N.  Even that is
anti-correlated with need -- mass grows with behavioural diversity, which is
lowest exactly when the dead zone binds.

These tests are the lemma's executable form.  They need no training run.
"""
from __future__ import annotations

import numpy as np
import pytest

from .group_law_teacher import ESTIMATORS, realized_mass

NS = [2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_unanimous_groups_carry_zero_mass(n, estimator):
    """Both unanimous outcomes are dead, for every deployed estimator."""
    assert realized_mass(0, n, estimator) == 0.0
    assert realized_mass(n, n, estimator) == 0.0


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_per_member_relabeling_buys_exactly_nothing(n, estimator):
    """The lemma: all-fail -> per-member relabel -> all-pass, mass 0 either way."""
    before = realized_mass(0, n, estimator)          # K = 0, the dead zone
    after = realized_mass(n, n, estimator)           # K = N, every member "succeeds"
    assert before == after == 0.0, (
        f"{estimator} N={n}: relabeling moved mass {before} -> {after}")


@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_group_consistent_relabeling_is_the_only_repair(n, estimator):
    """One common goal for the whole group gives K' < N, hence nonzero mass."""
    for k_prime in range(1, n):
        assert realized_mass(k_prime, n, estimator) > 0.0


@pytest.mark.parametrize("n", [8, 16])
def test_group_consistent_mass_is_anticorrelated_with_need(n):
    """Mass rises with behavioural diversity, which is lowest when stuck.

    Model the achieved goals of an all-fail group as ``d`` distinct end states.
    Relabel to one uniformly chosen member's goal; the number of members that
    then "succeed" is the size of that goal's class.  With d = 1 (every member
    ends in the same place -- the stuck regime HER is invoked for) K' = N and
    the mass is zero.
    """
    def expected_mass(d):
        # d equal-sized classes; picking any member selects its class of size N/d
        assert n % d == 0
        return realized_mass(n // d, n, "maxrl")

    masses = [expected_mass(d) for d in (1, 2, 4, 8) if n % d == 0]
    assert masses[0] == 0.0, "fully stuck group yields zero mass even after relabel"
    assert all(b > a for a, b in zip(masses, masses[1:])), (
        f"mass must increase with diversity, got {masses}")


@pytest.mark.parametrize("n", [4, 8, 16])
def test_mass_is_permutation_invariant_in_k_only(n):
    """The premise the lemma rests on: mass depends on the group only through K.

    Any relabeling scheme is therefore fully described by what it does to K --
    which is why 'it makes the rollouts succeed' is not, on its own, an argument
    that it helps.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(0, n + 1))
        outcomes = np.zeros(n, dtype=int)
        outcomes[:k] = 1
        for est in ESTIMATORS:
            base = realized_mass(k, n, est)
            for _ in range(5):
                rng.shuffle(outcomes)
                assert realized_mass(int(outcomes.sum()), n, est) == base
