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

## Why: the i.i.d. assumption fails exactly where the score aims

Post-hoc, descriptive, computed after the primary and changing no
preregistered quantity (`curriculum_maxrl/maze_score/calibration.py`).

Prop. 1 gives the expected group mass $A_N(p)=2(1-p-(1-p)^N)$ **under
conditionally i.i.d. Bernoulli rollouts**. The telemetry records every group's
success count and realized mass, so the prediction is directly checkable. To
avoid circularity — mass is a deterministic function of $K$ — each group's pass
rate is estimated **leave-one-out** from the other groups on the same level in
the same 25-update window. 288,000 group draws:

| $\hat p$ (LOO) | predicted $A_{32}$ | observed mass | realized/predicted | predicted silent | observed silent |
|---|---|---|---|---|---|
| .007 | .396 | .094 | .24 | 79.5% | 94.6% |
| **.113** | **1.731** | **.751** | **.43** | **2.2%** | **51.2%** |
| .224 | 1.551 | .920 | .59 | 0.03% | 31.6% |
| .406 | 1.188 | .942 | .79 | ~0% | 12.3% |
| .452 | 1.096 | .854 | .78 | ~0% | 12.1% |
| .727 | .546 | .505 | .93 | ~0% | 2.0% |
| .945 | .111 | .103 | .93 | ~0% | 0.4% |

The shape is right (binned Pearson $r=.90$) but the level is not, and the error
is not uniform: **the realization ratio rises monotonically with $p$.** At
$\hat p\approx.11$ the Binomial model predicts 2.2% silent groups and 51.2% are
silent — a 24$\times$ under-estimate; at $\hat p\approx.22$ it is 1000$\times$.

The cause is over-dispersion. A group shares a *level*, not a *maze*. Outcomes
inside a group are therefore positively correlated, unanimity is far more
common than Binomial, and since $A_N$ is concave, $\mathbb E_{\text{maze}}
[A_N(p_{\text{maze}})] < A_N(\mathbb E[p_{\text{maze}}])$. The teacher scores
levels, so it pays exactly this concavity penalty — and the penalty is largest
where the pass rate is lowest.

Per arm, over identical seeds and SFT checkpoints:

| arm | mean $\hat p$ of selected levels | predicted mass | realized mass | realized/predicted | silent groups |
|---|---|---|---|---|---|
| `un` ($u_{32}$) | .171 | .812 | .444 | **.55** | **60.7%** |
| `learn` ($p(1-p)$) | .380 | .876 | .595 | **.68** | **32.2%** |
| `unif` | .189 | .490 | .262 | .54 | 68.0% |

$u_{32}$ *predicts* nearly as much activity as $p(1-p)$ (.81 vs .88) but
*realizes* a quarter less (.44 vs .60), because its peak
$p^\star_{32}=.106$ sits in the worst-calibrated band while $p(1-p)$ peaks at
.5, where the model is close to right. It loses almost twice as many groups to
silence.

**The identity is exact under its assumption; the assumption degrades toward
hard tasks; so the score systematically over-values the region it was derived
to prefer.** That is a scale-and-substrate boundary on the derivation, not a
refutation of the algebra, and it is consistent with every other boundary we
measured: the exponent sweep (harder peaks win where gradients are exact), the
AMaze starvation result (one Bernoulli per visit), and the branching audit
(activity ranks utility but locates it poorly).

## What this does to the paper

The score's positive evidence is now explicitly bounded to the small,
exact-gradient regime. The paper reports a controlled positive at 640
parameters, a replication on two further platforms, and **three** preregistered
boundaries: peak location, standalone-signal bandwidth, and now neural scale.
The thesis that survives all of them is the one the paper already argues —
a deployed estimator induces an activity geometry, and activity is not learning
utility — with one addition this campaign earns: *activity is not even realized
activity once rollouts stop being i.i.d.*

Not done, deliberately: no extra seeds, no metric substitution, no promotion of
the branching $H{=}20$ secondary to rescue the headline.
