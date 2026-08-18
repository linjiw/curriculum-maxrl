# Preregistration — activity-matched, transfer-mismatched pairs (PI judgment Experiment ①)

**Status:** FROZEN 2026-08-18, before any utility measurement on this pool.
**Source:** `PI_JUDGMENT_2026-08-18.md` §七, Experiment 一.
**Substrate:** `utility_audit/branching_pool.py`, a skill forest with
heterogeneous branching, built today for this test.

## 1. Why a new pool was necessary

The utility audit returned "factorization not needed" (Δ = −0.021) on the
shipped three-linear-chain pool. That verdict cannot be read as evidence
against compounding, because on linear chains the test is unconstructable:
task level `l` requires skills `1..l`, so downstream-unlock count is a
deterministic function of level and level determines pass rate. Measured:
**corr(p, C) = +0.889**, and only 2 of 25 pass-rate bins contain more than one
distinct `C`. No structural term can carry information `u_N(p)` lacks there.

The branching pool breaks the confound by construction. A forest of trees with
per-tree branching factors `(3,1,3,1)` gives, at *identical* depth — hence the
same difficulty ladder — subtree sizes differing by up to an order of
magnitude: at depth 1, `C ∈ {4, 40}`; at depth 2, `{3, 13}`; at depth 3,
`{2, 4}`. Everything else (softmax skill policy, rollout, `apply_gradient`,
exact `true_pass_rates`) is inherited from `SkillChainEnv` unchanged, so the
estimator and learning rule are identical to the linear-chain audit.

Measured on this pool at warm=400: **34 pairs** with `u_N` matched to within
5% and `C` ratio ≥ 3×, the largest at 40×.

## 2. The test

At warm depth **400** uniform steps (chosen before any utility was measured,
from pass-rate spread and matched-pair count alone; at 1600 the pool is too
mastered and zero matched pairs survive), for each of **10 seeds**:

1. Form the **matched-pair set**: all task pairs `(i, j)` with
   `|u_N(p_i) − u_N(p_j)| / max(u_N) ≤ 0.05`, both `u_N > 10⁻⁶`, and
   `max(C_i,C_j)/min(C_i,C_j) ≥ 3`. Pairs are formed from `u_N` and `C`
   only — no utility is consulted.
2. For every task in every pair, measure true continuation utility
   `U_H(x) = J(Train_H(θ;x)) − J(θ)` by branch-and-continue at `H = 8`,
   exactly as in `UTILITY_AUDIT_PREREG.md` (deployed practical MaxRL, N=16,
   lr=0.5, private RNG stream per branch).
3. Within each pair, record `Δ = U_H(high-C) − U_H(low-C)`.

## 3. Primary estimand and decision rule

**Primary:** the seed-level mean of `Δ` over that seed's matched pairs, tested
across the 10 seeds by exact two-sided sign-flip (2¹⁰ assignments), α = .05,
with a 20,000-resample paired bootstrap CI.

**SESOI:** `Δ ≥ 0.002` in `J` units. `J = mean_x pass@8(x)`; the audit's
observed per-task `U_H` means are of order 0.003, so 0.002 is a difference of
the same magnitude as a typical task's entire one-shot contribution.

| condition | verdict |
|---|---|
| mean Δ ≥ +0.002 and p ≤ .05 | **transfer structure matters beyond activity.** Two tasks the deployed estimator scores identically differ in true utility by their downstream reach; a compounding term is warranted and the linear-chain null was a substrate artifact |
| CI upper < +0.002 | **activity suffices even under matched difficulty.** The linear-chain verdict generalises; drop the compounding object entirely |
| otherwise | inconclusive at n=10; report the interval, claim nothing |

**Frozen secondaries, descriptive only:** ρ(u_N, U_H) and ρ(u_N·C, U_H) over
the *full* pool at this warm depth, to compare against the linear-chain values
(+0.832 and +0.811); the same under RLOO; and Δ split by `C` ratio band
(3–5×, >5×).

## 4. What this cannot establish

- `C` here is the structural descendant count, the simplest member of its
  family; a supported result licenses "downstream reach carries utility
  information activity lacks", not "this is the right `C`".
- Depth-1 tasks carry both the largest `C` and the largest `p`; the matched
  set controls this by construction, but the pool is small (88 tasks) and the
  matched set is tens of pairs, not thousands.
- One-task-at-a-time utility at `H=8`, exact gradients, 88-task synthetic
  forest. This tests whether the *factors are separable in principle*, which
  is the question the linear chain could not pose.

---

## Amendment 2026-08-18 — the matching criterion was wrong (post-hoc, declared)

**Timing, stated plainly.** This amendment was written **after** running the
preregistered test and seeing its result. It is therefore *not* pre-data, and
the corrected test below is labelled exploratory. The original result stands
as preregistered and is reported in `BRANCHING_RESULT.md` regardless.

**The flaw.** §2 forms pairs by matching `u_N`. But `u_N` is **unimodal**: it
takes the same value on both sides of its peak. Matching on `u_N` alone
therefore pairs tasks of wildly different difficulty. Inspecting the `C`
ratio ≥ 5× band directly: mean pass rate is **0.852 on the high-`C` side and
0.011 on the low-`C` side** — a mastered task paired against a nearly-dead
one, at mean depths 1.06 and 3.94. The primary was not measuring "same
availability, different transfer"; over that band it was measuring "mastered
versus dead".

The PI judgment's Experiment ① specifies "相同的当前 pass rate；相同的
A_N(p)" — *same pass rate* **and** same activity. Only the second was
enforced. This is my error in operationalising the spec, and it is visible
from `p`, `C` and depth alone; but I did not go looking until the >5× band
came back negative, so I record the discovery as outcome-adjacent.

**The correction.** Pairs must additionally satisfy `|p_i − p_j| ≤ 0.02`.
Matching on `p` implies matching on `u_N` for any fixed `N`, so this is
strictly stronger and removes the branch ambiguity. Feasibility measured
before running: at warm 200 this yields ~1,000 pairs per seed with `C` ratio
≥ 3× (example: `p` 0.3866 vs 0.3864, `C` 40 vs 4); at 400, ~100–300; at 800
it collapses to ~0–30. **Warm depth moves to 200** for the corrected test,
chosen on pair count alone.

Everything else — `H=8`, 10 seeds, exact sign-flip, SESOI +0.002, the verdict
table, the frozen secondaries — is unchanged. Results appear under
`branch2-*` and are reported beside, never merged with, the preregistered run.
