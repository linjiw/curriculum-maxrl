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

## Part E — does the tilt survive the deployed machinery? (posterior p̂)

Same pools, but utilities computed from Thompson draws of the deployed
decayed-Beta posterior (α=0 is exactly the shipped AdvMassTeacher);
family (1−p̃)^α·u_N, α ∈ {0,1,2,3}, 10 seeds:

| pool | α=0 (deployed) | α=1 | α=2 | α=3 |
|---|---|---|---|---|
| chain | .743 | .770 (+.027, p=.009) | .777 (+.034, p=.001) | **.784** (+.041, p=.0009) |
| flat | .505 | **.520** (+.015, p=.034) | .518 (+.013, p=.046) | .508 (n.s.) |

**The edge survives posterior noise** — it is a deployable one-line
change, not an oracle artifact. α=2 is not special: chains reward more
tilt (monotone through α=3), the flat pool peaks at α≈1–2 and gives the
gain back at α=3. α ∈ [1,2] is the robust band across both pools.

Practical recommendation (not yet shipped): FrontierMax's utility line
`u = (1-(1-p̃)^N) - p̃` → `u *= (1-p̃)^α`, α=1 default. Honesty cost: this
adds one knob with a validated default to the "derived, not tuned"
story; the derivation-vs-tuning line moves from "the utility is derived"
to "the band and its zeros are derived; the within-band tilt is a
horizon-value correction, direction theory-motivated (Part D), exponent
tuned." Adopt only together with a GPU-rung validation (maze or E-LLM-3).

## Part F — stochastic lookahead (the residual is NOT mostly stochasticity)

Utility = mean cumulative Δp over M=32 *exact* stochastic virtual futures
of h groups (K~Bin(N,p); exact single-task MaxRL update — Σw=0 kills the
−q drift, so the update is `+lr(1−K/N)` on the correct logit and
`−lr·count/N` on sampled failure actions). Flat pool, 10 seeds:

| arm | AUC | vs det. h=200 | vs var-tilt (.592) |
|---|---|---|---|
| stoch h=50 | .571 | −.004 (n.s.) | −.021 (p=.002) |
| stoch h=200 | .580 | +.005 (7/10, p=.11) | −.013 (p=.0035) |

Stochasticity buys a small, non-significant improvement over the ODE and
**does not close the gap** — the pre-stated hypothesis is refuted as the
main explanation.

## Part G — budget-aware lookahead (also refuted)

Horizon = remaining_groups/n_tasks (shrinking as budget spends,
refreshed continuously): AUC .547 — *worse* than fixed h=200 (−.028,
0/10) and far below the tilt (−.045, 0/10). The correct per-task
horizon is evidently not the equal-share of remaining budget (a good
sampler concentrates, so its effective per-task horizon on chosen tasks
is much longer than the average).

## Parts H & I — the optimum's shape, and the resolution

Flat pool ⇒ tasks independent ⇒ the deterministic-optimal *open-loop*
allocation is a knapsack DP over per-task trajectories (quantum 10
groups, objective = **final** mean p). Its shape (part H artifact):
funds 30/36 tasks, allocation *increasing* in difficulty among funded
tasks (30 groups for p₀=.45 up to 540 for p₀=2e-4), hard-cutoff below
p₀≈2e-4 — "everyone funded to completion, pay each task its
time-to-learn, drop what doesn't fit." No proportional-to-anything
myopic index reproduces a shape with a budget-dependent cutoff.

Part I closes the loop: **MPC** (re-solve the DP from the current state
every 200 groups — planning *with* feedback):

| arm | AUC | final |
|---|---|---|
| mass (closed-loop) | .543 | .633 |
| var-tilt (closed-loop) | .592 | .730 |
| DP open-loop replay (H) | .542 | .660 |
| **MPC-DP (I)** | **.461** (−.13 vs tilt, 0/10) | **.762** (+.032 vs tilt, 8/10, p=.005) |

**The inversion resolves the mystery.** The planner — built to maximize
final performance — wins final and loses AUC; the tilt wins AUC. The
tilt's "mechanistically unexplained" edge was an *objective mismatch*
all along: parts B–G scored AUC, an objective none of the constructed
utilities targeted. "Which within-band utility is best" is **ill-posed
until the training objective is fixed** (time-averaged vs endpoint —
i.e., anytime performance vs a final checkpoint), and the answer
genuinely differs: endpoint favors planning-like allocations that
tolerate long dark periods on hard tasks; time-averaged favors
front-loading cheap wins. Note the same currency split runs through the
GPU experiments (maze "dose sets where the coverage dividend is spent";
Countdown early-stop banking the mean before the coverage bill) — this
is the testbed-exact version of that phenomenon.

## Where the utility question lands (final)

1. u_N ties the exact first-order improvement as a *predictor* (A).
2. No myopic or lookahead index we built is optimal for either
   objective; the true final-objective optimum is a *planning* solution
   (budget-dependent cutoff — not expressible as any state-local
   utility), approximable by MPC.
3. The practical recipe by objective: **AUC/anytime → (1−p)·u_N tilt**
   (cheap, posterior-robust, α∈[1,2]); **final-checkpoint → longer
   horizons / planning-like concentration** — and the paper's
   compounding-γ result (γ≈4 on chains) now reads as the same lesson:
   sharper concentration ≈ more planning-like.
4. The partition's boundaries (u_N's zeros) are shared by every
   candidate and remain the load-bearing theory claim.

## Follow-ups it opens (not started)

- GPU validation of the α-tilt before adopting it in FrontierMax
  (folded into sweep_un_form.sh P-U2 — running). Note P-U2's meter is
  matched-clock AUC — per the H/I resolution, that is the tilt's
  favorable objective; a final-checkpoint read should be reported
  beside it.
- Whittle-index analysis is now better-motivated: the restless-bandit
  value function is exactly what the DP computes; a closed-form index
  approximating it would unify the tilt (AUC) and the cutoff (final)
  as two discountings of one object.
