"""Cross-check that every encoding of the MAZE-SCORE block count agrees.

The block count N is written down in four independent places:

  1. ``EXPECTED_SEEDS``               analyze_maze_score.py
  2. ``MAX_EXACT_SIGN_FLIP_N``        analyze_maze_score.py
  3. ``#SBATCH --array=<lo>-<hi>%5``  hopper/sbatch/maze_score_array.sbatch
  4. a fail-closed seed regex         hopper/sbatch/maze_score_array.sbatch

Nothing in the repository compared them before this test.  The two failure
modes are both silent and both expensive:

  * the regex is narrower than the array  -> the array launches, every
    out-of-range task exits 2, and the campaign completes as a short run that
    still looks successful;
  * the array is wider than EXPECTED_SEEDS, or the pair count exceeds
    MAX_EXACT_SIGN_FLIP_N -> the campaign completes and the analyzer then
    refuses to read it, which under the prereg's no-rerun rule is
    unrecoverable.

See hopper/MAZE_SCORE_POWER_MEMO_2026-08-15.md.
"""

from __future__ import annotations

import pathlib
import re
import unittest

from analyze_maze_score import EXPECTED_SEEDS, MAX_EXACT_SIGN_FLIP_N

SBATCH = (pathlib.Path(__file__).resolve().parents[2]
          / "hopper" / "sbatch" / "maze_score_array.sbatch")


def _sbatch_text() -> str:
    return SBATCH.read_text(encoding="utf-8")


def _array_bounds() -> tuple[int, int, int]:
    m = re.search(r"^#SBATCH\s+--array=(\d+)-(\d+)%(\d+)\s*$",
                  _sbatch_text(), re.MULTILINE)
    if not m:
        raise AssertionError("no '#SBATCH --array=<lo>-<hi>%<throttle>' line")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _seed_guard() -> re.Pattern[str]:
    m = re.search(r'\[\[\s*"\$SEED"\s*=~\s*\^\((.+?)\)\$\s*\]\]', _sbatch_text())
    if not m:
        raise AssertionError("no fail-closed SEED regex guard in the sbatch")
    return re.compile(r"^(?:" + m.group(1) + r")$")


class SampleSizeContract(unittest.TestCase):

    def test_sbatch_file_exists(self):
        self.assertTrue(SBATCH.is_file(), f"missing {SBATCH}")

    def test_array_bounds_match_expected_seeds(self):
        lo, hi, _ = _array_bounds()
        self.assertEqual(
            (lo, hi), (min(EXPECTED_SEEDS), max(EXPECTED_SEEDS)),
            "sbatch --array bounds disagree with EXPECTED_SEEDS; a completed "
            "campaign would be unreadable or short")

    def test_expected_seeds_are_contiguous(self):
        seeds = list(EXPECTED_SEEDS)
        self.assertEqual(seeds, list(range(seeds[0], seeds[-1] + 1)))

    def test_seed_guard_accepts_every_expected_seed(self):
        guard = _seed_guard()
        for seed in EXPECTED_SEEDS:
            self.assertRegex(
                str(seed), guard,
                f"seed {seed} is in EXPECTED_SEEDS and in the array but the "
                f"sbatch guard rejects it; that task would exit 2 silently")

    def test_seed_guard_rejects_just_outside_the_range(self):
        guard = _seed_guard()
        for seed in (min(EXPECTED_SEEDS) - 1, max(EXPECTED_SEEDS) + 1):
            if seed < 0:
                continue
            self.assertNotRegex(
                str(seed), guard,
                f"seed {seed} is outside EXPECTED_SEEDS but the guard admits "
                f"it; the guard must stay fail-closed")

    def test_pair_count_within_exact_sign_flip_capacity(self):
        n = len(EXPECTED_SEEDS)
        self.assertLessEqual(
            n, MAX_EXACT_SIGN_FLIP_N,
            f"{n} blocks exceeds the exact sign-flip cap "
            f"({MAX_EXACT_SIGN_FLIP_N}); the analyzer would raise "
            f"AnalysisError after the campaign has already run")

    def test_exact_sign_flip_memory_stays_tractable(self):
        """Meet-in-the-middle stores 2**(n/2) + 2**(n-n/2) float64 sums."""
        n = MAX_EXACT_SIGN_FLIP_N
        gib = (2 ** (n // 2) + 2 ** (n - n // 2)) * 8 / 2 ** 30
        self.assertLess(
            gib, 1.0,
            f"exact sign-flip at the cap n={n} would need {gib:.2f} GiB; "
            f"raise the cap only with the memory budget in mind "
            f"(n=48 -> 0.25 GiB, n=52 -> 1.0 GiB, n=60 -> 16 GiB)")


if __name__ == "__main__":
    unittest.main()
