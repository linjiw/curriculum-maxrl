# Preregistration v2 — activity-matched transfer-mismatched pairs, confirmatory

**Status:** FROZEN 2026-08-18, before any run on these seeds.
**Supersedes:** the exploratory corrected run in `BRANCHING_RESULT.md` §3.
**Why v2 exists:** v1 matched pairs on `u_N` alone. `u_N` is unimodal, so that
admitted pairs on opposite sides of the peak — mean pass rate 0.852 versus
0.011 in the largest-mismatch band. The corrected criterion was written after
seeing v1's result and is therefore post-hoc; this document re-freezes it on
**fresh seeds** so the finding can carry weight.

## 1. Design

Substrate `utility_audit/branching_pool.py`, unchanged (`depth=4`,
`branch_by_tree=(3,1,3,1)`, 88 tasks). Deployed practical MaxRL, `N=16`,
`lr=0.5`, `weights_maxrl` drop-K=0 — identical to the audit and to
`run_validation.py`.

**Seeds 5003–5022** (20, double v1's 10). Disjoint from every prior block:
audit 3001–3010, v1 4001–4010. Seeds 5001–5002 were consumed by a harness
smoke that computed utility on them, so they are excluded from the
confirmatory block rather than reused.

**Warm depth 200** uniform steps, fixed from v1's feasibility measurement
(pair counts ~1,000/seed at 200, ~100–300 at 400, ~0–30 at 800). Not re-tuned.

**Pair criterion**, formed from `p`, `u_N` and `C` only — no utility consulted:

- `|p_i − p_j| ≤ 0.02`  (same difficulty)
- `|u_N(p_i) − u_N(p_j)| ≤ 0.02`  **absolute** (same availability)

  Both are required and neither implies the other. The PI spec asks for equal
  pass rate *and* equal `A_N(p)`; near `p→0` the score is steep
  (`u′_N(0) = N−1 = 15`), so a 0.02 difference in `p` can move `u_N` by up to
  0.30. v1 used a tolerance *relative* to `max(u_N)`, which varies by seed and
  made pair counts swing from 2 to 756 across seeds; the absolute form gives
  12–504 pairs per seed (median 33, median `C` ratio 4.0) over the
  confirmatory block. Chosen on pair counts alone, before any utility on these
  seeds.
- both `u_N > 10⁻⁶`  (both tasks active)
- `max(C_i,C_j) / min(C_i,C_j) ≥ 3`  (transfer mismatched)

**Measurement.** For every task in every pair, true continuation utility
`U_H(x) = J(Train_H(θ;x)) − J(θ)` by branch-and-continue, `J = mean_x pass@8`,
private RNG stream per branch keyed on `(seed, warm, task)`.

**Horizons `H ∈ {4, 8, 20}`.** v1 used `H=8` only. The judgment's Experiment ②
asks whether the activity/utility gap is horizon-dependent; measuring three
horizons in the same run answers it without a separate campaign. `H=8` is the
primary; 4 and 20 are frozen secondaries.

## 2. Primary estimand and decision rule

**Primary:** seed-level mean of `Δ = U_8(high-C) − U_8(low-C)` over that
seed's matched pairs; across the 20 seeds, exact two-sided sign-flip
(2²⁰ = 1,048,576 assignments), α = .05, plus a 20,000-resample paired
bootstrap CI.

**SESOI +0.002** in `J` units, unchanged from v1.

| condition | verdict |
|---|---|
| mean Δ ≥ +0.002 and p ≤ .05 | **transfer matters beyond matched activity** |
| CI upper < +0.002 | activity suffices even under matched difficulty |
| otherwise | inconclusive at n=20 |

**Frozen secondaries, descriptive:** the same Δ at `H = 4` and `H = 20`;
Δ split by `C`-ratio band (3–5×, >5×); `ρ(u_N,U₈)`, `ρ(u_N·C,U₈)`, `ρ(C,U₈)`
over the full pool; and the identical primary under **RLOO**, whose
coefficient mass is `2p(1−p)` — v1 found the MaxRL effect 50× larger there,
and this re-tests that dissociation.

## 3. Pre-registered expectation

v1 exploratory gave MaxRL Δ = +0.00313 (9/10) and RLOO +0.00006 (9/10). If v2
reproduces the MaxRL effect above SESOI on fresh seeds, `activity ≠ utility`
is established causally on this substrate. If it does not, v1 was seed-lottery
and is reported as such.

## 4. What this cannot establish

- `C` is the structural descendant count; a positive result licenses
  "downstream reach carries utility information activity lacks", not that
  `A_N·C` is the right functional form. v1 already indicates it is not:
  `ρ(u_N,U)` and `ρ(u_N·C,U)` were .641 versus .638.
- 88-task synthetic forest, exact gradients, one-task-at-a-time utility.
- Nothing here enters the current ICLR submission, whose boundary is closed.
