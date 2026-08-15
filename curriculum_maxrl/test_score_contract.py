"""Cross-adapter contract tests for the deployed-N score.

The paper's central promise is an exact mapping from the deployed estimator to
the deployed sampler.  Two algebraically distinct forms coexist in this
repository for historical reasons:

    exact   u_N(p)      = 1 - (1-p)^N - p          (the paper's score)
    legacy  (1-(1-p)^N)(1-p) = u_{N+1}(p)          (historical maze/verl form)

These tests pin which adapter computes which, so the distinction can never
become silent.  They also guard the 2026-08-15 refactor that routed the free
adapters through the canonical helpers: every assertion below is bit-exact
against the historical inline expressions, so no completed result moves.

`curriculum_maxrl/verl_curriculum.py` is one of the 31 files hash-locked by
`autoresearch/iterate-260810-2240/E2C_CODE_MANIFEST.json`.  It is deliberately
NOT edited; its semantics are pinned here by test instead.
"""

from __future__ import annotations

import copy
import unittest

import numpy as np

from estimators import coefficient_activity, legacy_frontier_activity
import teachers as teachers_mod
import verl_curriculum


P_GRID = np.array([0.0, 1e-6, 0.01, 0.1, 0.25, 1.0 / 3.0, 0.5, 0.75, 0.9,
                   0.999, 1.0])
N_GRID = [1, 2, 3, 4, 8, 16, 32, 33, 64]


def historical_exact(p, n):
    """The inline expression that lived at teachers.py:186 before 2026-08-15."""
    return (1.0 - (1.0 - p) ** n) - p


def historical_legacy(p, n):
    """The inline expression that lived at teachers.py:148 before 2026-08-15."""
    return (1.0 - (1.0 - p) ** n) * (1.0 - p)


class FormulaIdentities(unittest.TestCase):
    def test_n2_reduces_to_learnability(self):
        """u_2(p) = p(1-p) exactly — the SFL/ProCuRL learnability slice."""
        for p in P_GRID:
            self.assertAlmostEqual(
                float(coefficient_activity(p, 2)), float(p * (1.0 - p)),
                places=15, msg=f"p={p}")

    def test_legacy_is_exactly_the_shifted_score(self):
        """(1-(1-p)^N)(1-p) == u_{N+1}(p) for every N on the grid."""
        for n in N_GRID:
            got = legacy_frontier_activity(P_GRID, n)
            want = coefficient_activity(P_GRID, n + 1)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-15,
                                       err_msg=f"N={n}")

    def test_peak_location_matches_closed_form(self):
        """argmax_p u_N(p) = 1 - N^(-1/(N-1)), the paper's p*_N."""
        for n in [2, 4, 8, 16, 32, 64]:
            p_star = 1.0 - n ** (-1.0 / (n - 1))
            grid = np.linspace(1e-9, 1 - 1e-9, 2_000_001)
            numeric = grid[int(np.argmax(coefficient_activity(grid, n)))]
            self.assertAlmostEqual(numeric, p_star, places=5, msg=f"N={n}")

    def test_scalar_and_vector_agree(self):
        for n in N_GRID:
            vec = coefficient_activity(P_GRID, n)
            for i, p in enumerate(P_GRID):
                self.assertAlmostEqual(float(coefficient_activity(p, n)),
                                       float(vec[i]), places=15)
            leg = legacy_frontier_activity(P_GRID, n)
            for i, p in enumerate(P_GRID):
                self.assertAlmostEqual(float(legacy_frontier_activity(p, n)),
                                       float(leg[i]), places=15)

    def test_shape_is_preserved(self):
        arr = np.asarray(P_GRID).reshape(-1, 1)
        self.assertEqual(coefficient_activity(arr, 8).shape, arr.shape)
        self.assertEqual(legacy_frontier_activity(arr, 8).shape, arr.shape)
        self.assertEqual(np.ndim(coefficient_activity(0.3, 8)), 0)

    def test_rejects_degenerate_rollout_count(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                coefficient_activity(0.3, bad)
            with self.assertRaises(ValueError):
                legacy_frontier_activity(0.3, bad)


class RefactorIsBitExact(unittest.TestCase):
    """The canonical helpers must reproduce the pre-refactor inline code."""

    def test_exact_helper_matches_historical_inline(self):
        for n in N_GRID:
            for p in P_GRID:
                self.assertEqual(float(coefficient_activity(p, n)),
                                 float(historical_exact(p, n)),
                                 msg=f"p={p} N={n}")

    def test_legacy_helper_matches_historical_inline(self):
        for n in N_GRID:
            for p in P_GRID:
                self.assertEqual(float(legacy_frontier_activity(p, n)),
                                 float(historical_legacy(p, n)),
                                 msg=f"p={p} N={n}")


def _replay_distribution(teacher, utility, power=None):
    """Recompute a Thompson teacher's distribution from the same RNG draws."""
    state = copy.deepcopy(teacher.rng.bit_generator.state)
    try:
        w = np.zeros(teacher.n_tasks)
        for i, st in enumerate(teacher.stats):
            a, b = st.alpha_beta
            p = teacher.rng.beta(a, b)
            w[i] = utility(p, teacher.n_rollouts)
        if power is not None:
            w = np.maximum(w, 0.0) ** power
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(teacher.n_tasks, 1.0 / teacher.n_tasks)
        return (1 - teacher.explore_frac) * probs + \
            teacher.explore_frac * uniform
    finally:
        teacher.rng.bit_generator.state = state


class AdapterFamilies(unittest.TestCase):
    """Pin which score family each adapter actually deploys."""

    def _seeded(self, cls, **kw):
        t = cls(n_tasks=12, seed=7, n_rollouts=16, **kw)
        rng = np.random.default_rng(3)
        for st in t.stats:  # spread the posteriors so weights differ per task
            st.alpha_beta = (1.0 + rng.integers(0, 9), 1.0 + rng.integers(0, 9))
        return t

    def test_maxrl_frontier_teacher_deploys_the_legacy_shifted_form(self):
        t = self._seeded(teachers_mod.MaxRLFrontierTeacher)
        state = copy.deepcopy(t.rng.bit_generator.state)
        got = t.distribution()
        t.rng.bit_generator.state = state
        want = _replay_distribution(t, legacy_frontier_activity)
        np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def test_adv_mass_teacher_deploys_the_exact_score(self):
        t = self._seeded(teachers_mod.AdvMassTeacher)
        state = copy.deepcopy(t.rng.bit_generator.state)
        got = t.distribution()
        t.rng.bit_generator.state = state
        want = _replay_distribution(t, coefficient_activity, power=t.power)
        np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def test_locked_verl_adapter_still_deploys_the_legacy_form(self):
        """verl_curriculum.py is hash-locked; pin it by test, never by edit."""
        t = verl_curriculum.FrontierTeacher(n_prompts=9, n_rollouts=16)
        p = np.linspace(0.0, 1.0, 9)
        np.testing.assert_allclose(t.utility(p),
                                   legacy_frontier_activity(p, 16),
                                   rtol=0, atol=1e-15)
        # and it is NOT the paper's exact score
        self.assertFalse(np.allclose(t.utility(p),
                                     coefficient_activity(p, 16)))


class DeployedExponentMapping(unittest.TestCase):
    """The configured rollout count must map to the advertised exponent."""

    def test_score_metadata_exponents(self):
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "_maze_train",
            pathlib.Path(__file__).with_name("maze_gpu") / "train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        for n in (2, 8, 16, 32):
            self.assertEqual(mod.score_metadata("frontier_un", n),
                             ("coefficient_activity", n))
            self.assertEqual(mod.score_metadata("frontier", n),
                             ("legacy_frontier_activity", n + 1))
            self.assertEqual(mod.score_metadata("learnability", n),
                             ("coefficient_activity", 2))

    def test_maze_score_arm_selects_the_exact_u32(self):
        """The MAZE-SCORE 'un' arm must deploy u_32, not u_33."""
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "_maze_train2",
            pathlib.Path(__file__).with_name("maze_gpu") / "train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        family, exponent = mod.score_metadata("frontier_un", 32)
        self.assertEqual(family, "coefficient_activity")
        self.assertEqual(exponent, 32)


if __name__ == "__main__":
    unittest.main()
