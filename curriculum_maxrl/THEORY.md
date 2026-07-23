# Advantage-mass analysis: a derived curriculum signal for MaxRL

Deep-dive into the MaxRL paper's math (arXiv:2602.02710, full text) yielding a
derived coefficient-mass score for a heuristic teacher. The MaxRL and RLOO
mass identities are checked by 200k-trial Monte Carlo in the snippet below;
the complete analytic statements and assumptions are in `PROOFS.md`.

Here “derived” applies only to the scalar score `u_N`. Discounted pseudo-count
tracking, Thompson draws, proportional `u_N^gamma` sampling, decay, floor, and
`gamma` remain heuristic design choices rather than an exact optimizer.

## 1. Setup

A prompt with true pass rate `p` gets a group of `N` i.i.d. rollouts with
`K ~ Binomial(N, p)` successes. Each estimator assigns per-rollout advantage
weights `w_j`. Define the **scalar coefficient mass** (historically called
advantage mass here) as `Σ_j |w_j|`. The score functions `S_j` are multiplied
by these weights, so mass is an observable proxy/upper-bound factor for the
gradient budget; it is not the gradient norm or expected learning progress.

## 2. Exact expected advantage mass per estimator

**MaxRL** (Algorithm 1: `w_succ = 1/K − 1/N`, `w_fail = −1/N`, group dropped
at K=0). For K ≥ 1: `Σ|w| = K(1/K − 1/N) + (N−K)/N = 2(1 − K/N)`. Hence

```
E[Σ|w|] = 2·(P(K≥1) − E[K]/N) = 2·(pass@N(p) − p)   —  EXACT
        = 2·(pass@N − pass@1)
```

**The expected scalar coefficient mass on a prompt equals twice the
probability that it is solvable within N attempts but not within one.**
This is a compute-indexed coefficient-mass analogue of the zone of proximal
development: the estimator assigns expected scalar coefficient mass to the band of
prompts the student can *sometimes but not reliably* solve. It vanishes both
at p→0 (beyond frontier, group dropped) and p→1 (mastered), and peaks at

```
p* = 1 − N^(−1/(N−1)) ≈ ln(N)/N
```

(N=16 → p*≈0.17, N=32 → 0.11, N=128 → 0.038.) Larger group sizes shift the
**coefficient-mass peak** toward harder prompts at rate ln(N)/N. This does not
prove that the same p maximizes gradient norm or learning progress.

**RLOO** (`w_j = (r_j − LOO-mean)/N`): `Σ|w| = 2K(N−K)/(N(N−1))`, hence

```
E[Σ|w|] = 2·p·(1−p)   —  EXACT
```

RLOO's expected scalar coefficient mass is proportional to SFL's
"learnability" p(1−p) (Rutherford et al. 2024). The scalar score matches up
to a constant; the estimator and curriculum objective are otherwise distinct.

**GRPO** (`w_j = (r_j − mean)/(std+ε)/N`, degenerate groups K∈{0,N} give 0):

For a realized `K`, with the sample standard deviation used by the code,

```
Σ|w| = 2K(N-K) / [N²(sqrt(K(N-K)/(N(N-1)))+ε)].
```

At `ε=0` this is `2sqrt((N-1)K(N-K))/N^(3/2)`; its expectation is an exact
binomial sum rather than the displayed population approximation.

Numerically: at p=0.01, N=32 the population weight-function view of the paper
(`w(p)=1/√(p(1−p))` → mass ≈ 0.199) overstates the realized finite-sample
mass by 2× (exact: 0.100) because 72% of groups are all-fail and contribute
nothing. **The paper's population-level w(p) curves describe the
infinite-sample limit; at finite N the dropped practical MaxRL, RLOO, and GRPO
variants studied here are throttled by degenerate groups.** The always-retained
MaxRL control variate is an exception: it has nonzero sample weights at K=0.

## 3. Consequences for the curriculum design

1. **Derived coefficient-mass score.** Replace the heuristic frontier utility
   `u(p) = (1−(1−p)^N)(1−p)` with the exact half-mass score
   `u(p) = pass@N(p) − p = (1−(1−p)^N) − p`. Numerically the two are nearly
   close at configured N=16/32 (but can differ substantially at small N),
   which helps explain why the heuristic worked; the heuristic is the same
   family indexed by `N+1`. The derived form is (a) free of a hand-selected pass-rate
   band, (b) exactly half the expected scalar coefficient mass of the practical
   estimator, and (c) directly estimable from group statistics.

2. **Connection to SEC (Chen et al. 2025c, cited by the paper).** SEC drives
   a curriculum bandit with the *empirical* |advantage| as reward. For binary
   rewards our formulas are the exact expectations of SEC's signal per
   estimator. A Thompson-style teacher on `pass@N − p` is a model-based
   estimate of expected SEC-style coefficient mass, computed from discounted
   Beta pseudo-counts over p instead
   of noisy per-batch advantage sums. Under a changing policy this is
   Thompson-style randomization, not an exact stationary Bayesian posterior.

3. **Batch-level compute allocation rule.** Given budget B rollouts over a
   candidate pool with fixed known `p_i`, feasible integer bounds
   `1≤L_i≤N_i≤U_i`, and one group per selected prompt, allocating N_i per
   prompt to maximize half-mass (equivalently total expected mass after ×2)
   Σ_i (1−(1−p_i)^{N_i} − p_i) subject to Σ N_i = B is a concave (diminishing
   returns in N_i) resource-allocation problem → greedy/water-filling is
   optimal. The continuous derivative is
   `dM_i/dN = -(1-p_i)^N log(1-p_i)`; the discrete marginal relevant to
   integer allocation is `M_i(N+1) − M_i(N) = p_i(1−p_i)^N`. **Greedy rule:
   repeatedly give the
   next rollout to the prompt with the largest p_i(1−p_i)^{N_i}.** This
   replaces the heuristic `N_i ∝ 1/p̂_i` allocation in teachers.py with a
   rule that is provably optimal only for this one-step proxy when the inputs
   `p_i` are fixed and known. Plugging in moving estimates `p̂_i` is heuristic,
   and this is not a proof of an optimal long-horizon curriculum.
   Note `p(1−p)^N` is exactly the probability that rollout N+1 is the
   *first success* — "spend compute where the next sample is most likely to
   flip a dead group live."

4. **Metric alignment with the paper.** All headline results in the paper are
   coverage (pass@k) curves and the fraction-of-prompts-with-≥1-success
   dynamic (their Fig. 7). Our maze testbed so far only tracked mean pass;
   pass@k eval (Chen et al. 2021 unbiased estimator) is now added to the GPU
   testbed to make results comparable to the paper's claims — especially
   since MaxRL's advertised advantage (less pass@k collapse) is invisible in
   mean pass.

## 4. Decoupling T from N: corrected status

The paper's raw success estimator targets truncation order `T=N`. The
`c_{T,N}(K)` subset coefficient in this repository correctly decouples the
success term: `E[c_{T,N}(K)K]=1-(1-p)^T`. However, the historical
`weights_maxrl_t` implementation subtracts `1/N` only when `K>0`.
Proposition 1 in PROOFS.md shows that this outcome-dependent control-variate
drop changes its population weight to

```text
w_T(p) - (1-p)^(N-1).
```

It therefore does **not** target `J_T` exactly. At `T=N` it reduces to
effective order `N-1`; at `T=1` it does not recover plain RL. Exact decoupling
is available by retaining the `-1/N` weights on all-fail groups
(`weights_maxrl_t_unbiased`).

The historical adaptive-`T` experiment reported AUC 0.698 vs 0.704 for the
fixed setting under the coefficient-mass teacher (and 0.641 vs 0.653 under
uniform sampling). Those numbers compare shifted practical estimators, not the
claimed order-`T` objectives, so the earlier variance interpretation is not
established. Adaptive `T` is now marked inconclusive pending a corrected rerun.

## 5. Hindsight relabeling: manufacturing successes for the success-conditioned estimator

MaxRL's Theorem 1 expresses the ML gradient as an expected score conditioned
on success. The practical centered estimator uses both positively weighted
successful trajectories and negatively weighted failures in a live mixed
group, but an all-fail K=0 group is discarded and therefore supplies no
update. Hindsight
Experience Replay (HER) offers the complementary move: a failed trajectory is
a *success for the goal it actually reached*. Where task structure admits
relabeling (goal-conditioned tasks, nested prefixes), an eligible dead group
may yield a verifier-valid auxiliary group for an easier related task, at zero
extra generation cost but with additional rewriting, scoring, and optimizer
compute. With centered weights, an update occurs only when the
rewritten group is nondegenerate; a relabeler may also return no valid target.

The following historical five-seed exploratory snapshot is superseded by the
retained 12-seed V10 factorial in `VALIDATION.md`; its legacy “AUC” naming and
point estimates are kept only for chronology. On the skill-chain testbed (a failed level-l rollout with correct prefix j is
a success of the nested level-j task; relabel dead groups to the deepest
prefix achieved, apply the same success-conditioned weights):

| config | final (5 seeds) | AUC | relabeled groups |
|---|---|---|---|
| uniform+maxrl | 0.966 | 0.653 | 0 |
| **uniform+maxrl+hindsight** | 0.978 | **0.878** | 145 |
| advmass+maxrl | 0.979 | 0.704 | 0 |
| **advmass+maxrl+hindsight** | 0.984 | **0.883** | 129 |

In that historical snapshot, hindsight had the largest point improvement
(legacy AUC +0.22 versus +0.05 for the teacher) and stacked with it.
Interpretation: the teacher *avoids* spending compute beyond the frontier;
hindsight *recycles* whatever still lands there. Together they make the
frontier band effectively wider.

Bias caveat: the relabeled group is conditioned on the achieved outcome, so
it is not an unbiased estimator of the relabeled task's truncated-ML gradient
(same status as HER's auxiliary goals). Empirically it helps uniformly here;
at LLM scale the analogue is goal/prefix relabeling where verifiers admit it
(maze goals, sub-goals in multi-step proofs, partial-credit unit tests).

More precisely, exactness can be assessed only per accepted credited goal:
the conditional rewritten-group update moment must match the fresh target
group moment. Even if that holds, the goal-selection probability and hindsight
scale determine the overall auxiliary mixture. The current Acrobot relabeler
is therefore a verifier-valid auxiliary update, not exact hindsight-MaxRL; its
mixed-only acceptance makes full-law equality with a nondegenerate fresh group
impossible. See Proposition 6 in `PROOFS.md`.

**Archived exploratory ablations (driver/output not retained; advmass
teacher, 5 seeds).** Hindsight weight scale was monotone
on the toy — AUC 0.805 / 0.840 / 0.883 / 0.908 / 0.928 / 0.943 at scale
0.25→8. The toy's relabeled labels are semantically exact and the mean update
is strongly direction-aligned, but data-dependent goal selection still makes
the centered update biased in scale (PROOFS.md Proposition 6). Expect a knee
(then collapse) on real models where
over-weighted imitation of self-generated prefixes can entrench errors and
kill diversity. Default stays 1.0 (the natural unscaled multiplier); treat
scale as the imitation-strength knob the GPU/LLM runs must tune.

**Archived exploratory interaction (16-level regime; driver/output not
retained).** Hindsight partially
*substitutes* for the teacher: uniform+hindsight reaches 0.970 vs
advmass-alone 0.961 — recycling dead groups fixes much of what prompt
selection was avoiding. But they still stack (advmass+hindsight 0.978, best
in every regime tested). A possible teacher wall-clock advantage on real
models remains an untested compute hypothesis: it may avoid generating doomed
rollouts, whereas hindsight salvages them only after generation and adds
scoring/optimizer work. Division of labor: teacher =
don't waste compute; hindsight = salvage what still fails.

## 6. Verification snippet

```python
import numpy as np
rng = np.random.default_rng(1)
N = 32
for p in [0.005, 0.05, 0.2, 0.6, 0.95]:
    K = rng.binomial(N, p, size=200000)
    mass_maxrl = np.where(K >= 1, 2*(1 - K/N), 0.0).mean()
    assert abs(mass_maxrl - 2*((1-(1-p)**N) - p)) < 3e-3
    mass_rloo = (2*K*(N-K)/(N*(N-1))).mean()
    assert abs(mass_rloo - 2*p*(1-p)) < 3e-3
```

(Both asserts pass; GRPO's exact mass computed by binomial summation matches
Monte Carlo to 4 decimals.)
