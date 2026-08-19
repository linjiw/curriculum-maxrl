# P0 — Does scoring the count law recover what the plug-in loses?

**Status:** DRAFT, not frozen. Freeze target 2026-08-24. **No run may start
before the freeze commit.** Supersedes
`GRANULARITY_FLIP_PREREG_v1_SUPERSEDED.md`, which never launched.
**Schema:** `curriculum-maxrl/group-law-flip/v1`

## 1 · What is being tested

Corollary 2 is Tier 1: scoring a unit `z` by the i.i.d. curve at its mean pass
rate over-predicts its activity by exactly `2[Pr(K=0|z) − (1−p̄_z)^N] ≥ 0`,
verified to floating point on 288,000 real groups. Its **consequence for
learning** is Tier 3, supported post hoc only: MAZE-SCORE scored a level, lost
to `p(1−p)` by −.0032 [−.0054, −.0011], and the telemetry accounts for the
coefficient-mass error exactly.

This registers the counterfactual. Hold the substrate, estimator, budget,
warmstart and seeds fixed, and change **only the statistic the teacher scores**:

| arm | score of level `z` | posterior |
|---|---|---|
| `plugin` | `u_N(p̂_z)` — the i.i.d. curve at the unit's mean | Beta over `p_z` (identical to MAZE-SCORE's `un` arm) |
| `grouplaw` | `q̂_z − p̂_z` — half the realized count-law mass | count law over `K` |

Both are computed from the **same observed group outcomes**; the count-law arm
costs no extra rollouts, because `K` is already recorded for every group.
Implementation: `curriculum_maxrl/group_law_teacher.py`, whose prior is the
i.i.d. law, so the two arms agree exactly at initialization and separate only as
evidence of non-Binomial structure arrives (`test_group_law_teacher.py`, 45
tests, including the reduction to `u_N(p)` on atomic i.i.d. tasks and the exact
gap identity).

**No finite pool is needed.** v1 required one because a per-task posterior is
undefined when every maze is freshly generated. The count-law score is estimable
per level from the counts already observed, so the substrate is MAZE-SCORE's,
untouched.

## 2 · Design inputs, declared

The delivery threshold below was set from an **offline replay of the frozen
MAZE-SCORE telemetry**, using only group success counts — never an evaluation
endpoint, and never a number from this study, which has not run. Replaying the
`un` arm's realized visits:

- induced sampling distributions differ by **total-variation 0.126**;
- the correction shifts mass toward easier levels (plug-in top-3 levels 4,3,5;
  count-law top-3 levels 3,2,4), the direction Cor. 2 implies;
- per-level gaps are large where the score concentrates: at `p̂ = .227`,
  plug-in 1.546 against count-law 0.926.

This bounds the design; it does not predict the endpoint. In a live run the
count-law arm visits differently, so realized divergence may differ.

## 3 · Endpoints and decision rule (to be frozen 2026-08-24)

- **Primary:** paired per-block **time-integrated** `cov_auc_delta`, contrast
  `grouplaw − plugin` — the same currency, definition and ten-checkpoint
  structure as MAZE-SCORE, so the studies are directly comparable. Exactly ten
  timepoints enter; a missing, duplicate or extra one invalidates the cell.
- **Support requires all three:** exact two-sided paired sign-flip `p ≤ .05`;
  paired percentile-bootstrap 95% CI lower bound `> 0` (20,000 resamples);
  point estimate **≥ +.005** cov-AUC, identical to MAZE-SCORE's SESOI.
- **Practically ruled out** iff the CI upper bound `< +.005`.
- **Inconclusive** otherwise. Sign counts are descriptive.
- **Seeds 3001–3020**, twenty blocks, disjoint from every seed in this
  repository. Independent unit is the seed block.

**Treatment-delivery gate, pre-committed.** Mean total-variation distance
between the two arms' realized level-visit distributions, averaged over the run,
must be **≥ 0.05**. Below that the arms did not implement different curricula
and the endpoint is reported as *treatment not delivered*: inconclusive, and
explicitly **not** evidence against Cor. 2. The offline replay gives 0.126, so
the gate is expected to pass; it exists to make a null interpretable.

**Frozen secondary (descriptive, no decision).** Per-level measured gap
`2[P̂(K=0|z) − (1−p̄_z)^N]`, and its rank correlation with the per-level
coverage difference.

## 4 · Both verdict branches, drafted before data

**Supported.** §8 gains: *"Replacing the mean-pass-rate plug-in with the
realized count law, with substrate, estimator, budget and seeds fixed, recovers
Δ (CI …, k/20 blocks). The granularity gap therefore governs learning and not
only coefficient mass, and the MAZE-SCORE negative is its predicted sign."*
Tier 2, registered and confirmed. The count-law teacher becomes a method
contribution rather than a corollary of the theory. No claim is made that it
beats `p(1−p)`; that contrast is not in this design.

**Practically ruled out or inconclusive.** §8 gains: *"Scoring the realized
count law rather than the mean pass rate did not recover the loss (Δ …, CI …).
The granularity gap therefore accounts exactly for the coefficient-mass
prediction error without accounting for the learning outcome; we report it as a
calibration result, not a curriculum mechanism."* Tier 3, registered and
bounded. Pre-named diagnoses: (i) the realized-activity deficit is real but too
small a share of the update budget to move coverage; (ii) the count-law estimate
trades calibration for variance at this visit rate; (iii) both scores are
dominated by the uniform floor at this pool geometry. Cor. 2's Tier-1 status is
untouched either way — it is an identity about mass.

**Neither branch permits** adding seeds, changing the endpoint, or re-running an
arm for a scientific reason.

## 5 · Guards inherited from the AMaze checkpoint failure

1. Every run records its final training state; the analyzer refuses a cell below
   the full budget.
2. Completion is a `DONE` marker written on zero exit, never the presence of a
   checkpoint file.
3. The analyzer refuses an incomplete 2×20 matrix, refuses to run twice, and
   refuses a block whose arms disagree on warmstart or source hash.
4. Progress logging prints no endpoint value.

## 6 · What this cannot establish

One task family, one architecture, one `N`, one budget, one level partition. It
tests whether replacing the plug-in with the count law moves learning on this
substrate. It does not establish that either beats `p(1−p)`, nor that the result
transfers to a different partition or a finer unit.
