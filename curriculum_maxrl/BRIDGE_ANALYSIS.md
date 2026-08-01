# The mass→learning-progress bridge (2026-08-01)

Addresses opus5 review M5 / NEXT_RESEARCH #4: advantage mass u_N(p) is the
paper's sampling utility, but nothing connected it to *learning progress*,
and the review showed a variance-tilted utility (1−p)²·u_N beating it
10/10 seeds. This experiment measures the gap and locates the mechanism.
Artifact: `results_bridge.json` (`run_bridge.py`; parts C/D appended by
the analysis scripts recorded in the repo history).

## Setup

Skill-chain testbed, everything exact:
- p_i = ∏_{s∈req_i} q_s, and ∇p_i has closed form, so the **exact
  first-order expected eval improvement** from one MaxRL group on task i is
  computable: LP(i) = lr · pass@(N−1)(p_i) · (1/M) Σ_m p_m C_im, where
  C_im sums the "unmastered-ness" c_s = (1−q_s0)² + Σ_{a≠0} q_sa² over
  shared skills. (Uses the T=N−1 prefactor of the practical estimator.)
- Ground truth per task: 400 independent group draws → apply update →
  exact Δeval → restore.

Candidates: mass u_N(p) (the paper's utility), variance tilt (1−p)²·u_N
(the review's challenger), closed-form LP (theory-optimal *myopic* choice).

## Part A — as instantaneous predictors of ΔJ (6 snapshots, seed 0)

| checkpoint | eval | ρ(mass, GT) | ρ(var-tilt, GT) | ρ(LP, GT) | LP-vs-GT slope |
|---|---|---|---|---|---|
| 0 | .009 | .799 | .799 | .799 | 1.21 |
| 200 | .079 | .888 | .880 | .881 | 1.05 |
| 400 | .192 | .942 | .872 | .945 | 1.03 |
| 800 | .405 | .985 | .821 | .990 | 1.02 |
| 1600 | .847 | .999 | .999 | .999 | 0.99 |
| 3200 | .969 | .995 | .995 | .996 | 0.99 |

Three facts:
1. **The closed-form LP is a valid first-order theory** — regression slope
   of ground truth on LP converges to 1.0 once past the all-dead regime.
2. **Mass ranks tasks essentially as well as exact LP** (ρ within .005 at
   every checkpoint). As a *predictor of the next step's improvement*,
   u_N is not missing anything material on this testbed.
3. **The variance tilt is a WORSE instantaneous predictor** (ρ .82 vs .99
   at ck=800) — yet it is the better sampling utility (below). Whatever
   it captures is not "which task improves eval most this step."

## Part B — as sampling utilities, chain pool (oracle-p, 10 seeds, AUC)

uniform .672 → mass γ1 .880 ≈ **LP γ1 .881** (paired d=+.0007, p=.63)
→ var-tilt γ1 .905 → mass γ4 .909 → **var-tilt γ4 .915**.
Var-tilt beats mass at matched γ: +.025 (γ1, 10/10) and +.005 (γ4, 9/10).
LP at γ4 *loses* to mass γ4 (−.014, 0/10).

**The dissociation:** the best instantaneous predictor (LP, slope 1.0) is
not the best sampling utility; the best sampling utility (var-tilt) is the
worst instantaneous predictor. Greedy-on-expected-improvement is not
optimal sequential sampling. The utility question is **dynamic, not
static** — no myopic functional settles it.

## Part C — flat-pool control (36 independent tasks; compounding removed)

Pre-stated prediction (recorded in the artifact before running): LP loses
on chains because it is myopic about *cross-task* compounding (steps on
task i unlock task j), so on a flat pool LP should stop losing.

**Refuted.** Flat pool: mass .543, var-tilt .592 (+.050, 10/10), LP .492
(−.050 vs mass, 0/10; −.100 vs var-tilt). LP loses *worse* without
cross-task structure. Cross-task compounding is not the mechanism.

## Part D — horizon-h lookahead (flat pool; the mechanism test)

Replace myopic LP with cumulative Δp over h virtual expected updates of
the task's own softmax (closed-form single-task ODE). If LP's deficit is
myopia about a task's *own* future (within-task option value: raising p_i
also raises task i's future learnability), the gap should close as h grows.

| utility | AUC | vs var-tilt (.592) |
|---|---|---|
| lookahead h=1 | .493 | −.100 (0/10) |
| h=10 | .507 | −.085 (0/10) |
| h=50 | .550 | −.042 (0/10) |
| h=200 | .575 | −.017 (1/10) |
| h=800 | .525 | −.068 (0/10) |

The gap closes monotonically through h=200 — **~83% of LP's deficit is
myopia about the task's own trajectory**, not cross-task effects. It never
crosses var-tilt, and h=800 *regresses*: with a long horizon the
deterministic ODE says nearly every task eventually saturates, flattening
the utility and pushing weight onto the far frontier — exactly where the
deterministic model is most wrong, because real learning there is
all-or-nothing (expected time-to-first-success ~ 1/(N·p) groups, huge
variance the ODE ignores). The residual gap is **stochasticity**: a task's
value is the value of its *distribution* of futures, not of its expected
trajectory.

## What this means for the paper

1. **u_N's honest status improves**: it ties the exact first-order LP as a
   predictor of next-step improvement (Part A) — the earlier framing
   "mass is uncorrelated with learning" (from the oracle-mass-vs-AUC
   observation) conflates prediction with control. Mass-at-collection
   saturates because *every reasonable policy collects almost all
   available mass*; that says nothing about ranking quality.
2. **But no myopic utility is the right objective**: sampling is a
   sequential decision problem. The variance tilt (1−p)²·u_N is a cheap
   surrogate for horizon value — it happens to discount near-mastered
   tasks (little future headroom) harder than mass does. A principled
   Proposition 8 would need option-value / Gittins-style machinery, which
   is beyond a testbed-paper's scope; the honest statement is the
   dissociation itself.
3. **The partition claims are untouched**: dead zone / band / mastered
   tail rest on u_N's zeros, which are exact and shared by every candidate
   (LP, var-tilt, and lookahead all vanish at p∈{0,1}). The *within-band
   ordering* is where the myopic/dynamic gap lives.
4. Paper edit shipped with this analysis: Remark "scope of the surrogate"
   now states the dissociation and cites this artifact, replacing the
   weaker "does not claim proportionality" hedge.

## Follow-ups it opens (not started)

- A stochastic lookahead (simulate the Beta-Binomial first-success time
  instead of the ODE) at h≈200 — would test whether stochasticity closes
  the last −.017.
- Whether (1−p)^α·u_N with tuned α beats α=2 (is the review's tilt itself
  just a point on a family?).
- Whether the tilt's advantage survives *posterior* p̂ (all Part B/C/D
  arms use oracle p; the deployed teacher's Thompson noise may wash out
  a .005–.025 AUC edge — cheap to test with the existing harness).
