# MAZE-SCORE sample-size memo — 2026-08-15

**Outcome-blind.** No endpoint, result JSONL, telemetry, or checkpoint was
opened. Inputs are only: the preregistered SESOI, the historical contrast SD
range already recorded in the DRAFT prereg, the measured full-arm cost from
engineering job 9366552, and the analyzer source.

**Purpose.** Supply the outcome-blind sample-size decision the DRAFT prereg
defers ("The exact count remains an outcome-blind DRAFT item",
`MAZE_SCORE_PREREG.md:54-55`), before the protocol is frozen.

**Recommendation: 48 blocks (seeds 20–67), keeping the exact sign-flip test.**
This requires raising `MAX_EXACT_SIGN_FLIP_N` from 40 to 48 and reconciling
four encodings of N. It does **not** require replacing the preregistered exact
randomization test with a Monte-Carlo approximation.

---

## 1. The finding that matters most: n does not fix an effect at the SESOI

The frozen decision rule is a conjunction
(`curriculum_maxrl/maze_score/analyze_maze_score.py:529`):

```python
supported = mean >= SESOI and lower > 0.0 and adjusted_p < 0.05
```

Because the rule requires the **observed** mean to clear the SESOI, a true
effect of exactly +.005 gives a coin flip no matter how many blocks are run.
Simulating the actual rule (20,000 replications per cell):

| true effect | n=30 | n=40 | n=48 | n=60 | n=72 |
|---|---|---|---|---|---|
| **+.005 (= SESOI)** | 45.7% | 49.7% | 49.3% | 50.8% | 50.2% |
| +.0075 | 81.2% | 87.7% | 89.8% | 92.2% | 94.0% |
| +.010 | 97.2% | 98.9% | 99.4% | 99.8% | 100.0% |

(paired SD = .0135, the pessimistic end of the recorded .0077–.0135 range;
α = .05.)

**The +.005 row is flat.** Adding 42 blocks buys 4.5 points. This is not a
defect in the analyzer — it is what a "mean at least the SESOI" clause means —
but it must be stated plainly, because the natural reading of "SESOI" is "the
smallest effect the study can detect," and this design cannot detect that
effect better than chance at any n.

**Consequence for the prereg:** state the powered-for effect explicitly. This
design is powered for effects **at or above +.0075**, i.e. 1.5× the SESOI. The
SESOI remains the reporting threshold; it is not the detection threshold.
Alternatively, relax the conjunction while the prereg is still DRAFT — but the
conjunction is what makes a "supported" verdict mean *practically* and not
merely *statistically* significant, so I recommend keeping it and being
explicit.

## 2. Where n does buy something

Under Holm's worst case (α = .025) and the pessimistic SD = .0135, at the
powered-for effect +.0075:

| n | power | Δ vs previous | MIG-slice-h | wall @ %5 |
|---|---|---|---|---|
| 30 (current default) | 73.8% | — | 33.4 | 6.7 h |
| **40** | **86.2%** | **+12.4** | 44.6 | 8.9 h |
| **48 (recommended)** | **90.0%** | **+3.8** | 53.5 | 11.1 h |
| 60 | 92.5% | +2.5 | 66.9 | 13.4 h |
| 72 | 94.1% | +1.6 | 80.2 | 16.7 h |

The curve knees at 40 and is flat past 48. Going 48 → 72 costs **+26.7
MIG-slice-hours and +5.6 h wall for +4.1 points**.

At the optimistic SD (.0077) every design from 30 up is already ≥96% at
+.0075, so the choice only matters if the true dispersion is at the
pessimistic end — which is the assumption a sample-size decision should make.

Arithmetic: block cost ≤ 3 × 1,337 s = 4,011 s = 1.1142 MIG-slice-h
(conservative; it triple-counts one-time SFT prep). Waves = ⌈n/5⌉ under the
`%5` array throttle.

## 3. Why 48 and not 72: the exact test survives

`exact_sign_flip_p` is a meet-in-the-middle enumeration
(`analyze_maze_score.py:452-495`) storing 2^(n/2) + 2^(n−n/2) float64 sums.
`MAX_EXACT_SIGN_FLIP_N = 40` is a **memory** guard, not an algorithmic wall:

| n | exact-test RAM |
|---|---|
| 30 | 0.5 MB |
| 40 (current cap) | 16.8 MB |
| 44 | 67.1 MB |
| **48** | **268.4 MB** |
| 52 | 1.1 GB |
| 60 | 17.2 GB |
| 72 | 1,099.5 GB |

48 blocks costs 268 MB and one constant change. 72 blocks is arithmetically
impossible for the exact test and would force a seeded Monte-Carlo
substitution.

That substitution is the real cost the 72-block option was hiding. Replacing a
preregistered **exact** randomization p-value with a **sampled approximation**
changes the inferential instrument, not just the sample size, and it would have
to be justified in the paper against a project whose central asset is
procedural candor. Raising a memory cap from 40 to 48 does not.

**48 blocks recovers 90.0% of the 94.1% that 72 would give, keeps the exact
test, and costs two-thirds of the compute.**

## 4. Required pre-freeze reconciliation

Four independent encodings of N currently say 30 and must move together, plus
the memory cap. All five are outcome-blind edits and all are free while the
prereg is DRAFT:

| # | Site | Now | To |
|---|---|---|---|
| 1 | `analyze_maze_score.py:34` | `EXPECTED_SEEDS = tuple(range(20, 50))` | `range(20, 68)` |
| 2 | `analyze_maze_score.py:51` | `MAX_EXACT_SIGN_FLIP_N = 40` | `48` |
| 3 | `hopper/sbatch/maze_score_array.sbatch:12` | `--array=20-49%5` | `--array=20-67%5` |
| 4 | `hopper/sbatch/maze_score_array.sbatch:88` | `^(2[0-9]\|3[0-9]\|4[0-9])$` | must accept 20–67 |
| 5 | `hopper/MAZE_SCORE_PREREG.md` | "30 fresh independent seed blocks, 20–49" | 48 blocks, 20–67, + the §1 powered-for statement |

Site 4 is the dangerous one: the regex is a *fail-closed* seed guard, so if it
is not widened the array silently refuses seeds 50–67 and the campaign
completes as a 30-block run. Sites 1 and 3 disagreeing produces the opposite
failure — a completed campaign the analyzer refuses to read.

**Add a cross-check test** asserting that the sbatch array bounds, the sbatch
regex, `EXPECTED_SEEDS`, and `MAX_EXACT_SIGN_FLIP_N` are mutually consistent,
so this class of drift cannot recur silently. Without it, nothing in the
repository compares these four numbers.

## 5. Residual open items before freeze

These are outcome-blind and still unresolved; they are not sample-size
questions but they gate the same freeze:

1. **A fresh primary evaluation panel.** The repeatedly-used seed-12345 panel
   can only serve as a descriptive continuity check, not the primary endpoint.
2. **Mazes per level** is not pinned in the DRAFT.
3. **The powered-for effect statement** from §1 must appear in the prereg text,
   not only in this memo.

## 6. Verification

Power figures are a 20,000-replication simulation of the exact conjunction in
`analyze_maze_score.py:529`, with the sign-flip p approximated by its paired
t-test counterpart (permutation and t tests have essentially equal power for
symmetric paired differences; the approximation affects the third digit, not
the ranking). Script:
`$CLAUDE_JOB_DIR/tmp/power.py`, seed 20260815.

Memory figures are exact: `(2**(n//2) + 2**(n - n//2)) * 8` bytes.

Cost figures derive from job 9366552's measured 1,337 s single-arm wall clock
and 39,672 MiB peak, which requires retaining the `3g.40gb` MIG slice.
