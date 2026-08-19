# P0 — The granularity flip: does scoring the estimator's own unit recover the loss?

**Status:** DRAFT, not yet frozen. Freeze target 2026-08-24 per the editorial
charter. **No run may start before the freeze commit.**
**Schema:** `curriculum-maxrl/granularity-flip/v1`
**Registers:** Corollary 2 (task granularity) as a *prospective* prediction
rather than the post-hoc reading it currently has.

## 1 · Why this experiment, and what it is worth

Corollary 2 states that a curriculum scoring an aggregate `z` by its mean pass
rate over-predicts its coefficient activity by exactly

    A_N(p̄_z) − 2·E_X[u_N(p_X)] = 2·[Pr(K=0 | z) − (1−p̄_z)^N]  ≥ 0.

The corollary is Tier 1 (proved, floating-point verified on 288,000 real
groups). Its *consequence for learning* is currently Tier 3 supported post hoc
only: MAZE-SCORE scored a level, lost to `p(1−p)` by −.0032
[−.0054, −.0011], and the telemetry accounts for the coefficient-mass error
exactly. What has never been tested is the counterfactual — **hold everything
fixed and score the concrete task instead.**

Both verdicts are worth having, which is why this is P0:

- **Confirm** → the corollary becomes a prospectively demonstrated statement
  about learning, not only about mass; the MAZE-SCORE negative becomes a
  *predicted sign*; and the paper's last abstract sentence is earned.
- **Refute** → the activity gap accounts for mass but not for learning. The
  corollary stays Tier 1 as an identity, its learning consequence is demoted to
  descriptive telemetry, and the paper says so in §8 with the diagnosis.

## 2 · A design deviation, stated rather than made silently

The charter specifies "hold substrate fixed; vary only posterior granularity —
per-level vs. per-task." **On the MAZE-SCORE substrate that is not
implementable, and the reason is a finding.**

`curriculum_maxrl/maze_gpu/train.py` runs an *infinite-data regime*: every step
calls `sample_task(level, rng)` to generate a **fresh** maze, which is never
seen again, and the teacher's posterior is a length-13 array indexed by level
(`Teacher.observe(level, rewards)`). There is no task identity to attach a
posterior to. A per-task posterior is not merely absent; it is undefined.

> **Consequence worth stating in the paper:** in a fresh-sample regime the
> granularity gap of Cor. 2 is not a bookkeeping defect that better
> instrumentation removes. It is structural, because the concrete task the
> estimator consumes is never observed twice.

The flip therefore requires a revisitable pool. **One substrate change, applied
byte-identically to both arms:** a finite pool of mazes replaces the fresh-maze
stream. Every other component — estimator, N, budget, warmstart, decay, floor,
Thompson mechanism, evaluation schedule, seeds — is held fixed. Granularity
remains the only manipulated variable, which is what the corollary is about.

## 3 · Design

Fixed pool, per seed block: **16 mazes per level × 13 levels = 208 tasks**,
generated once from the block's own RNG and shared byte-identically by both
arms (pool hash recorded per block; the analyzer refuses a block whose two arms
disagree on it). At 250 steps × 8 groups = 2,000 group draws per run, that is
≈9.6 expected visits per task under uniform sampling — enough for a per-task
Beta posterior to be determined, and deliberately far from the posterior
starvation diagnosed on GSM8K.

Both arms select **among the same 208 tasks**, so the action space is
identical. The single difference is the unit the score is computed on:

| arm | score of task *x* at level *z* | posterior units |
|---|---|---|
| `level` | `u_N( p̂_z )` — posterior pooled over the level's 16 mazes | 13 |
| `task`  | `u_N( p̂_x )` — posterior for that maze | 208 |

Shared and frozen: practical MaxRL at **N = 32**; discounted Beta with decay
0.7; uniform floor 0.15; Thompson-sampled pass rate; 250 update steps; 8 groups
per step; evaluation every 25 steps on the fixed held-out set; identical
per-block SFT warmstart checkpoint; identical learning rate.

**Seeds 3001–3020, twenty blocks.** Disjoint from every seed used anywhere in
this repository, including MAZE-SCORE's 20–67 and AMaze's 1001–1005 / 2001–2010.

## 4 · Endpoints and the decision rule (to be frozen 2026-08-24)

- **Primary:** paired, per-block **time-integrated** `cov_auc_delta` contrast
  `task − level`, where `cov_auc_delta` is the mean pass@8 coverage over the ten
  RL evaluations minus the post-SFT value — the same currency and definition as
  MAZE-SCORE, so the two studies are directly comparable. The time-integrated
  form is the primary by the factorial's power lesson (a single-eval endpoint
  already failed once); exactly ten checkpoint values enter, and a missing,
  duplicate, or extra timepoint invalidates the cell rather than being averaged.
- **Support requires all three** (conjunctive, as in MAZE-SCORE):
  1. exact two-sided paired sign-flip `p ≤ .05` on the block mean;
  2. paired percentile-bootstrap 95% CI lower bound `> 0` (20,000 resamples);
  3. point estimate **≥ +.005** cov-AUC (SESOI, identical to MAZE-SCORE's).
- **Practically ruled out** iff the CI upper bound `< +.005`.
- **Inconclusive** otherwise. Sign counts are descriptive.
- Independent unit is the **seed block**. Twenty blocks put the exact sign-flip
  floor at 2/2^20 and, at MAZE-SCORE's observed paired SD of ≈.0076, give a
  standard error of ≈.0017.

**Frozen secondary (mechanism, descriptive, no decision).** From the `level`
arm's telemetry, the measured per-level over-prediction
`2·[P̂(K=0 | z) − (1−p̄_z)^N]`, and its rank correlation with the per-level
coverage deficit. This is the quantity Cor. 2 names; it is reported whichever
way the primary lands.

## 5 · Both verdict branches, drafted before any data

**If supported.** §8 gains: *"Scoring the concrete task instead of the level,
with the substrate, estimator, budget and seeds held fixed, recovers Δ
(CI …, k/20 blocks). The granularity gap of Cor. 2 therefore governs learning
and not only coefficient mass, and the MAZE-SCORE negative is its predicted
sign."* The abstract's closing sentence is upgraded from "mean pass rate is not
a sufficient statistic for activity" to add "and scoring the estimator's own
unit recovers the loss." Tier assignment: **Tier 2, registered & confirmed.**
No claim is made that task-level scoring beats `p(1−p)` unless that contrast is
itself registered — it is not in this design.

**If practically ruled out or inconclusive.** §8 gains: *"Scoring the concrete
task rather than the level did not recover the loss (Δ …, CI …); an effect at or
above the registered SESOI is ruled out / the study does not establish one. The
granularity gap therefore accounts exactly for the coefficient-mass prediction
error (Tier 1, unaffected) without accounting for the learning outcome, and we
report it as a calibration result rather than a curriculum mechanism."* The
diagnosis to write is the one the data supports, from these pre-named
candidates: (i) the realized-activity deficit is real but too small a share of
the update budget to move coverage; (ii) per-task posteriors trade calibration
for variance at 9.6 visits/task; (iii) the pool discretization changed the
learning problem. Tier assignment: **Tier 3, registered & bounded.** The
corollary's Tier-1 status is untouched either way, because it is an identity
about mass.

**Neither branch permits** adding seeds, changing the endpoint, or re-running an
arm for a scientific reason.

## 6 · Implementation, and the fail-closed guards this campaign inherits

Required changes, all in a new module so the MAZE-SCORE evidence path stays
untouched: a finite-pool sampler; a task-indexed posterior; a `--score_unit
{level,task}` switch; per-group telemetry extended with the task id and the
pool hash.

Guards, carried over from the AMaze checkpoint failure of 2026-08-19 — that
campaign evaluated six of ten seeds at half budget because checkpointing was
keyed on ticks with no post-loop save:

1. Every run writes its final training state and records the stored
   `n_updates`; the analyzer refuses any cell below the full budget.
2. Completion is a `DONE` marker written only on a zero exit, never the mere
   presence of a checkpoint file.
3. The analyzer refuses an incomplete 2×20 matrix, refuses to run twice, and
   refuses a block whose arms disagree on the pool hash, the warmstart hash, or
   the source hash.
4. Progress logging must not print any endpoint value.

## 7 · What this cannot establish

One task family, one architecture, one `N`, one budget, one pool size. It tests
whether *granularity* moves learning on this substrate; it does not establish
that task-level scoring beats the canonical `p(1−p)` score, nor that the result
transfers to a fresh-sample regime — where, as §2 notes, per-task scoring is not
available at all.

---

## SUPERSEDED 2026-08-19, before any run

Never launched; no cell exists. Replaced by
`GROUP_LAW_FLIP_PREREG.md`, which tests the same corollary with a strictly
better design: the count-law score `q̂_z − p̂_z` is estimable from the success
counts a trainer already observes, so it needs **no finite pool** and runs on
the MAZE-SCORE substrate unchanged. The obstacle this v1 worked around — that a
per-task posterior is undefined in a fresh-sample regime — is dissolved rather
than accommodated. v1's finding about that regime is carried forward verbatim.
