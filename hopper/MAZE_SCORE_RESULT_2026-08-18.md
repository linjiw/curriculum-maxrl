# MAZE-SCORE result — the deployed-$N$ shape does not transfer to neural scale

**Campaign:** `maze-score-v2-20260816-001`, attempt-001, 48 blocks (seeds 20–67),
protocol `maze_score_v2`, source manifest `d98fe3ed02acbbeb7c1e29d9…`.
**Prereg:** `hopper/MAZE_SCORE_PREREG.md`, FROZEN 2026-08-16, unmodified.
**Analyzer:** `curriculum_maxrl/maze_score/analyze_maze_score.py`, SHA-256
`197f1254…7bd5` — byte-identical to the hash recorded at freeze. Run **once**.
**Retrieval:** all 48 array tasks terminal `COMPLETED 0:0` across arrays
9389151 (seeds 20–43) and 9389243 (seeds 44–67); fetched to
`/data/robotixx/maze_score/campaign-20260818`, tree digest
`1f9eb70447b212b1…`, all 48 per-cell `SHA256SUMS` verified, 48/48 `COMPLETE`
markers, 144/144 result files. No endpoint was opened before the matrix was
whole.

## Primary: practically ruled out

Endpoint `cov_auc_delta` — mean pass@8 coverage over ten RL evaluations
(updates 25…250) minus the post-SFT value, paired within block.

| contrast | mean | 95% boot CI | exact sign-flip p | Holm p | pairs | decision |
|---|---|---|---|---|---|---|
| **`un` − `learn`** (primary) | **−.00324** | [−.00543, −.00111] | .00542 | .00542 | 15+/32−/1= | **practically ruled out** |
| `un` − `unif` (secondary) | +.00888 | [+.00657, +.01115] | 2.6e-9 | 5.3e-9 | 41+/6−/1= | supported |

Arm means: `learn` +.000285, `un` −.002950, `unif` −.011829.

The frozen rule is that "practically ruled out" applies only if the primary
interval's upper bound is below the +.005 SESOI. It is (−.00111). Per the
branch written before running:

> State that effects at or above the SESOI were ruled out under this protocol;
> do not claim universal non-transfer.

So: **at 1.26M parameters on procedural mazes at the deployed $N=32$, sampling
by $u_{32}$ does not beat sampling by its $N{=}2$ slice $p(1-p)$, and an
advantage at or above +.005 cov-AUC is ruled out.** Both adaptive samplers beat
uniform, so this is not a failure of curriculum learning; it is a failure of
*this score's shape* to be the better one at this scale.

No seeds were added, no endpoint substituted, no arm re-run. The Acrobot
positive stands at 640 parameters; it does not extend to here.

## Why: the curriculum names a coarser unit than the theory

Post-hoc, descriptive, computed after the primary and changing no
preregistered quantity (`curriculum_maxrl/maze_score/calibration.py` and
`group_law_audit.py`).

**Corrected mechanism.** `train.py` draws **one concrete maze per group** and
repeats that prompt N=32 times; the teacher's posterior pools over the many
mazes of a *level*. So rollouts inside a group *are* conditionally i.i.d. --
the conditionally i.i.d. unit is the maze, while the unit the curriculum
scores is the level. (An earlier draft of this document said the reverse; see
`PI_CORRECTION_GROUPLAW_GRANULARITY_2026-08-18.md`.)

**The identity never needed independence.** Realized mass is the deterministic
`M(K) = 2(1 - K/N)1{K>0}`, so for any joint binary group law
`A_N(Q) = 2(Pr(K>0) - E[K]/N)`. The familiar `2{1-p-(1-p)^N}` is only its
conditional-i.i.d. slice. What fails at a coarse task unit is the scalar
`p`-only reduction, and the failure is exact:

> **A_N(p̄_z) − 2·E_X[u_N(p_X)] = 2[Pr(K=0|z) − (1−p̄_z)^N] ≥ 0**
>
> the plug-in over-prediction equals twice the aggregate's *excess all-fail
> probability*.

**Verified to floating point**, not fitted: across 41,101 / 18,497 / 9,355
(seed, arm, level, window) cells at window widths 10 / 25 / 50 updates, the
maximum deviation is 2.8e-16 for `M̄ = 2(q̂ − p̂)` and 4.4e-16 for the
excess-silence identity (`hopper/MAZE_SCORE_GROUPLAW_AUDIT.json`).

| p̂ (LOO) | predicted A₃₂ | observed mass | realized/predicted | predicted silent | observed silent |
|---|---|---|---|---|---|
| .007 | .396 | .094 | .24 | 79.5% | 94.6% |
| **.113** | **1.731** | **.751** | **.43** | **2.2%** | **51.2%** |
| .224 | 1.551 | .920 | .59 | 0.03% | 31.6% |
| .406 | 1.188 | .942 | .79 | ~0% | 12.3% |
| .452 | 1.096 | .854 | .78 | ~0% | 12.1% |
| .727 | .546 | .505 | .93 | ~0% | 2.0% |
| .945 | .111 | .103 | .93 | ~0% | 0.4% |

**Why the harder-peaked score pays more.** Second order, the penalty is
≈ ½·|u_N''|·Var(p_X); at u_N's own peak |u_N''| = (N−1)/(1−p*_N) ≈ 34.7 for
N=32, against the exact and far milder `u_2(p̄) − E[u_2(P)] = Var(P)`. Raising
N moves activity toward harder tasks *and* makes the geometry more sensitive
to curriculum granularity.

Clustered on the 48 seed blocks (never on the 288,000 group draws):

| arm | realized/predicted | 95% CI (seed-clustered) | silent groups |
|---|---|---|---|
| `un` (u₃₂) | **.580** | [.570, .590] | **60.8%** |
| `learn` (p(1−p)) | **.703** | [.691, .715] | **32.1%** |
| `unif` | .587 | [.574, .601] | 68.0% |

Paired, `un` − `learn` = **−.123** [−.133, −.112], negative in **48/48**
blocks.

**Claim boundary.** This exactly accounts for the *coefficient-mass prediction
error* and is consistent with the endpoint contrast. It is not an
intervention, so it is not established as the sole causal mediator of the
endpoint. "Neural scale" is a property of this protocol, not an isolated
causal variable: capacity, function approximation, task heterogeneity and
curriculum granularity all change together. The windowed estimates pool
maze-to-maze heterogeneity with learner nonstationarity; the identity's
window-invariance bounds but does not remove that. No counterfactual
"a corrected score would have won" is asserted.

## What this does to the paper

The score's positive evidence is now explicitly bounded to the small,
exact-gradient regime. The paper reports a controlled positive at 640
parameters, a replication on two further platforms, and **three** preregistered
boundaries: peak location, standalone-signal bandwidth, and now neural scale.
The thesis that survives all of them is the one the paper already argues —
a deployed estimator induces an activity geometry, and activity is not learning
utility — with one addition this campaign earns, now stated as a theorem
rather than a diagnosis: *the estimator defines the coefficient map, the
curriculum defines the unit over which that map is averaged, and these
operations do not commute.*

Not done, deliberately: no extra seeds, no metric substitution, no promotion of
the branching $H{=}20$ secondary to rescue the headline.
