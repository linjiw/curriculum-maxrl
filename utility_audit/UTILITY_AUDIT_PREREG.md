# Preregistration — branch-and-continue utility audit (SURPRISE experiment ①)

**Status:** FROZEN 2026-08-18, before any audit run.
**Source:** `SURPRISE_GUIDANCE_2026-08-18.md`, prediction ledger row ①.
**Substrate:** the exact-gradient CPU skill-chain testbed
(`curriculum_maxrl/testbed.py`, `SkillChainEnv`), where true pass rates are
computable and one gradient step is a pure function of `theta`.

## 1. The question

The paper's score, `u_N(p) = A_N(p)/2`, measures **availability**: how much
coefficient mass the deployed estimator can emit on a task. Curricula deploy
it as **utility**: what sampling should aim at. Three of our negatives — the
u64>u16 sweep, the AMaze replacement loss, and the oracle-tie — are consistent
with the two being different objects.

This audit measures **true continuation utility** directly and asks how well
each candidate score *predicts* it:

    U_H(x; θ) = J(Train_H(θ; x)) − J(θ)

where `Train_H(θ; x)` is `H` gradient steps of the deployed estimator on task
`x` alone, branched from a copy of `θ`, and `J` is a fixed population
objective over the pool.

## 2. Predictors under audit

| predictor | definition | what it represents |
|---|---|---|
| `p(1-p)` | learnability, the `N=2` slice | the literature's score |
| `u_N` | `1 - p - (1-p)^N` at deployed `N` | availability (the paper's score) |
| `u_64` | `u_N` at `N=64`, deployed `N` unchanged | the over-shooting winner from the sweep |
| **`A·C`** | `u_N(p_x) · C(x)`, `C(x)` = number of downstream tasks sharing skills with `x` whose pass rate would rise if `x`'s skills improved | availability × compounding — the guidance's factorization |
| `oracle` | `U_H` itself | ceiling |

`C(x)` is the structural compounding term. On the chain testbed it is
`(n_levels − level(x))`, the count of harder tasks in `x`'s chain: training
level `l` updates skills `1..l`, which every level `> l` in the chain also
requires. On the flat pool (§3) it is identically 1.

## 3. Two pools

- **structured**: `SkillChainEnv(n_chains=3, n_levels=12)` as shipped —
  tasks in a chain share prefix skills, so training one task raises the pass
  rate of every harder task in its chain.
- **flat**: same task count and same per-task difficulty *distribution*, but
  every task owns disjoint skills, so no training step on `x` changes any
  other task's pass rate. Built by giving each task its own private skill
  block whose length equals its chain level.

The ODE model in `curriculum_maxrl/VALIDATION.md` V6b predicts the
compounding term matters on the first and vanishes on the second.

## 4. Protocol

For each pool and each of **10 seeds**:

1. Reach a mid-training state: run the shipped uniform sampler with the
   deployed practical MaxRL estimator (`N=16`, drop all-fail) for
   `S_warm` steps so pass rates are spread across the pool.
2. At that state, for **every** task `x` in the pool: branch a deep copy of
   the env, train `H = 8` steps on `x` alone with the same estimator, `N=16`,
   `lr` as shipped, and record `U_H(x) = J_after − J_before` where
   `J = mean_x pass@8(x)` computed exactly from `true_pass_rates`
   (`pass@8 = 1 − (1−p)^8`).
3. Record every predictor's value at the *pre-branch* state.
4. Repeat at three warm-start depths `S_warm ∈ {400, 800, 1600}` so the
   audit spans early, mid, and late training; each depth is a separate
   stratum. *Depths were chosen before any utility was computed, from the
   pass-rate spread alone: at 400 uniform steps ~31% of tasks have p>.05 and
   ~6% are mastered; at 800, median p≈.10 with 19% mastered; at 1600, ~94%
   live and 55% mastered. A first draft used {20,40,80}, at which every task
   sits at p≈0 and all predictors trivially agree; that draft never ran a
   utility measurement and is superseded here.*

Every branch is deterministic given the seed; the branch RNG is derived from
`(seed, depth, task)` so no task's branch consumes another's stream.

## 5. Primary estimand and decision rule

**Primary:** Spearman rank correlation between each predictor and `U_H`,
computed **within** each (pool, seed, depth) stratum over the pool's 36
tasks, then averaged over depths within seed. This gives one number per
(predictor, pool, seed); the paired unit is the seed.

**The thesis test** is the paired contrast, on the structured pool,

    Δ_struct = ρ(A·C, U_H) − ρ(u_N, U_H)

over 10 seeds, exact two-sided sign-flip (2^10 assignments), α = .05,
SESOI +0.05 in rank correlation.

**The specificity test** is the same contrast on the flat pool, `Δ_flat`,
where the prediction is **no** advantage (`C ≡ 1` there, so `A·C = u_N` up to
scale and Δ_flat = 0 exactly by construction — this stratum is a built-in
sanity check on the code, not a statistical test).

**Frozen interpretation:**

| Δ_struct | verdict |
|---|---|
| ≥ +0.05 and p ≤ .05 | **factorization supported**: `C` carries predictive information availability lacks; "harder-peaked helps" is a consequence of compounding |
| CI upper < +0.05 | **factorization not needed**: `u_N` already ranks utility; retreat to "harder-peaked helps" as an empirical rule and drop the `A·C` object |
| otherwise | inconclusive at n=10; report interval, claim nothing |

Frozen secondaries, descriptive only: `ρ(u_64, U_H)` and `ρ(p(1-p), U_H)`
on both pools; the argmax-`p` of `U_H` per stratum versus `p*_16 = .169`
and `p*_64 = .064` (does true utility peak nearer the over-shoot?); the
same audit under `RLOO` in place of MaxRL as a single labelled robustness
column.

## 6. What this cannot establish

- It measures one-task-at-a-time utility at horizon `H=8`; sequential
  curricula compound across tasks and this audit does not simulate that.
- The compounding term `C` used here is the structural count, the simplest
  member of the family the guidance describes; a supported result licenses
  "the structural term helps," not "this is the right `C`."
- Exact gradients, tiny policy, synthetic pool. It tests the *ordering*
  claim in the setting where utility is measurable, which is the point.

## 7. Cost

36 tasks × 3 depths × 10 seeds × 2 pools × (8 branch steps of N=16) is
~2,160 branches, plus warmups of up to 1,600 steps; well under an hour on
one core.
