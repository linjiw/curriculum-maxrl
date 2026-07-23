# Advantage-mass analysis: a derived curriculum signal for MaxRL

Deep-dive into the MaxRL paper's math (arXiv:2602.02710, full text) yielding a
*derived* (not heuristic) teacher utility. Coefficient-mass and expected-
gradient formulas are verified by exact finite-N enumeration in
`test_math_claims.py`; the Monte Carlo snippet at the bottom is a quick check.

## 1. Setup

A prompt with true pass rate `p` gets a group of `N` i.i.d. rollouts with
`K ~ Binomial(N, p)` successes. Each estimator assigns per-rollout advantage
weights `w_j`. Define the **advantage mass** of the group as `Σ_j |w_j|` —
the L1 mass of the scalar coefficients multiplying the rollout score
functions. It is a useful estimator-side surrogate, but not the policy-gradient
norm: score-vector norms and cancellation can change the realized update.

## 2. Exact expected advantage mass per estimator

**Practical MaxRL Algorithm 1** (`w_succ = 1/K − 1/N`,
`w_fail = −1/N`, group dropped at K=0). For K ≥ 1:
`Σ|w| = K(1/K − 1/N) + (N−K)/N = 2(1 − K/N)`. Hence

```
E[Σ|w|] = 2·(P(K≥1) − E[K]/N) = 2·(pass@N(p) − p)   —  EXACT
        = 2·(pass@N − pass@1)
```

**The expected MaxRL coefficient mass on a prompt equals twice the
probability that it is solvable within N attempts but not within one.**
This is a compute-indexed formalization of the zone of proximal development:
the estimator, by its own algebra, allocates signal exactly to the band of
prompts the student can *sometimes but not reliably* solve. It vanishes both
at p→0 (beyond frontier, group dropped) and p→1 (mastered), and peaks at

```
p* = 1 − N^(−1/(N−1)) ≈ ln(N)/N
```

(N=16 → p*≈0.17, N=32 → 0.11, N=128 → 0.038.) So **larger group sizes
automatically shift the optimal curriculum band toward harder prompts at rate
ln(N)/N** — the teacher and the objective are indexed by the same compute
knob, now with an exact constant.

**Objective-order caveat (exact).** The paper's Theorem 2 applies to the
success-only estimator in Eq. (9). Full Eq. (10) preserves that order-N
expectation only because its `−1/N` score control remains active when K=0.
Practical Algorithm 1 explicitly drops both terms at K=0. Exact enumeration
and Proposition 0 show that this practical variant has population weight
`w_{N−1}(p)`, so it optimizes the order-(N−1) truncated objective (and is
identically zero at N=1). The coefficient-mass identity above remains exact
for the practical coefficients. Full Eq. (10) has expected coefficient mass
`(1−p)^N + 2(pass@N−p)` because an all-fail group carries mass 1.

**RLOO** (`w_j = (r_j − LOO-mean)/N`): `Σ|w| = 2K(N−K)/(N(N−1))`, hence

```
E[Σ|w|] = 2·p·(1−p)   —  EXACT for every N ≥ 2
```

RLOO's advantage mass **is** SFL's "learnability" p(1−p) (Rutherford et al.
2024) up to the constant. The learnability curriculum literature and the
RLOO estimator are the same object seen from two sides.

**GRPO** (`w_j = (r_j − mean)/(std+ε)/N`, degenerate groups K∈{0,N} give 0):

```
Σ|w| | K = 2K(N−K) / (N²·sqrt(K(N−K)/(N(N−1))))   (ε→0)
E[Σ|w|] = Σ_{K=1}^{N−1} P(K) · [that quantity]
        → 2·sqrt(p(1−p)) as N→∞ for fixed p∈(0,1)
```

Numerically: at p=0.01, N=32 the population weight-function view of the paper
(`w(p)=1/√(p(1−p))` → mass ≈ 0.199) overstates the realized finite-sample
mass by 2× (exact: 0.100) because 72% of groups are all-fail and contribute
nothing. Multiplying the population limit by the non-degenerate-group
probability is not exact because the conditional K distribution also changes
the mass. **The paper's population-level w(p) curves describe the
infinite-sample limit; at finite N every estimator's realized coefficient
mass on hard prompts is throttled by pass@N.**

## 3. Consequences for the curriculum design

1. **Derived teacher utility.** Replace the heuristic frontier utility
   `u(p) = (1−(1−p)^N)(1−p)` with the exact advantage mass
   up to an irrelevant factor of two:
   `u(p) = pass@N(p) − p = (1−(1−p)^N) − p`. Numerically the two are nearly
   identical (max deviation ~1% of range), which retroactively explains why
   the heuristic worked; but the derived form is (a) parameter-free, (b) an
   exact target for the practical estimator's coefficient mass, and (c)
   directly estimable from group statistics.

2. **Connection to SEC (Chen et al. 2025c, cited by the paper).** SEC drives
   a curriculum bandit with the *empirical* |advantage| as reward. For binary
   rewards our formulas are the exact expectations of SEC's signal per
   estimator. A Thompson teacher on `pass@N − p` is "oracle SEC for MaxRL":
   same target, but computed from a Beta posterior over p instead of noisy
   per-batch advantage sums, and therefore usable *before* a prompt is ever
   sampled (posterior prior + optimism), where SEC needs at least one visit.

3. **Batch-level compute allocation rule.** Given budget B rollouts over a
   candidate pool, allocating N_i per prompt to maximize total advantage mass
   Σ_i (1−(1−p_i)^{N_i} − p_i) subject to Σ N_i = B is a concave (diminishing
   returns in N_i) resource-allocation problem → greedy/water-filling is
   optimal. Marginal value of one more rollout on prompt i:
   `ΔM_i(N) = (1−p_i)^N · p_i · [d/dN version]` — discrete marginal
   `M_i(N+1) − M_i(N) = p_i(1−p_i)^N`. **Greedy rule: repeatedly give the
   next rollout to the prompt with the largest p̂_i(1−p̂_i)^{N_i}.** This
   replaces the heuristic `N_i ∝ 1/p̂_i` allocation in teachers.py with a
   provably-optimal (for the mass objective) one.
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

## 4. Decoupling T from N: exact path implemented, old experiment not exact

The repo contains an unpublished-in-paper success coefficient
(`oversample_subset_vr_weights` / `c_sub_TN` in
`verl/trainer/ppo/maclaurin.py`, eq. 51 of the appendix): per-success weight
`c_{T,N}(K)` such that with N rollouts the success term is unbiased for the
**T-truncated** objective for any T ≤ N:

- `c_{N,N}(K) = 1/K` exactly (recovers Eq. 9's success term);
- `E[c_{T,N}(K)·K] = 1−(1−p)^T = w_T(p)·p` to 4 decimals for
  T ∈ {1,4,16}, p ∈ {0.05,0.3,0.7}, N = 16 (100k-trial MC).

To keep that expectation after variance reduction, the unconditional score
control must be applied to every group, including K=0. The new
`weights_maxrl_t_eq10` does this and is exhaustively verified. The historical
`weights_maxrl_t` function instead dropped the entire K=0 group. Its exact
population multiplier is

```
w_T(p) − (1−p)^(N−1),
```

not `w_T(p)`. At T=N it reduces to practical Algorithm 1's `w_{N−1}`.

The exact helper makes T a per-prompt knob independent of rollout budget and
opens a third integration axis beyond prompt selection and rollout allocation:

- easy prompts (p̂ high): T = 1 → plain RL weighting, minimal variance;
- frontier prompts: T = N → full ML weighting where the higher-order
  pass@k terms matter;
- beyond-frontier prompts kept in-batch for exploration: large N (better
  chance of a first success + tighter p̂ posterior) with moderate T
  (bounding the 1/p variance blow-up the truncation exists to control).

The teacher already estimates p̂ per prompt, so annealing T_i by difficulty
is mechanically cheap. This mirrors how the population weight
w_T(p) = (1−(1−p)^T)/p is
flat (≈T) for p ≪ 1/T and ≈1/p for p ≫ 1/T: choosing T_i ≈ 1/p̂_i puts every
prompt at the knee of its own weight curve.

**Empirical status: unresolved for the exact estimator.** The historical
experiment used the dropped-group `weights_maxrl_t` variant. With
T_i = clip(1/p̂_i, 1, N) slightly *underperforms* fixed T=N
(advmass teacher: AUC 0.698 vs 0.704; uniform: 0.641 vs 0.653; 5 seeds).
That is a valid negative for the tested heuristic, but not evidence about an
unbiased adaptive-T estimator. Re-running with `weights_maxrl_t_eq10` is
required before making the objective-curriculum claim.

## 5. Hindsight relabeling: manufacturing successes for the success-conditioned estimator

MaxRL's Theorem 1 says the ML gradient is the expected score function
*conditioned on success*. The implemented variance-reduced estimator uses
failures as a zero-mean control variate inside live groups, but emits exactly
zero when K=0. Hindsight Experience Replay (HER) offers the complementary move:
a failed trajectory can be a *success for the goal it actually reached*. Where
task structure admits
relabeling (goal-conditioned tasks, nested prefixes), each dead group can be
converted into a live group for an easier related task, at zero extra
generation cost.

On the skill-chain testbed (a failed level-l rollout with correct prefix j is
a success of the nested level-j task; relabel dead groups to the deepest
prefix achieved, apply the same success-conditioned weights):

| config | final (5 seeds) | AUC | relabeled groups |
|---|---|---|---|
| uniform+maxrl | 0.966 | 0.653 | 0 |
| **uniform+maxrl+hindsight** | 0.978 | **0.878** | 145 |
| advmass+maxrl | 0.979 | 0.704 | 0 |
| **advmass+maxrl+hindsight** | 0.984 | **0.883** | 129 |

**Largest single improvement found in this project** — bigger than the
teacher itself on learning speed (AUC +0.22 vs +0.05), and stacking with it.
Interpretation: the teacher *avoids* spending compute beyond the frontier;
hindsight *recycles* whatever still lands there. Together they make the
frontier band effectively wider.

Bias caveat: the relabeled goal is selected from the same dead group, so the
group is coupled and conditioned to contain a relabeled success. It is not an
unbiased estimator of the relabeled task's truncated-ML gradient (same status
as HER's auxiliary goals). V1 establishes directional alignment on the skill
chain (mean cosine 1.000), not equality in magnitude or joint sampling law.
Empirically it helps uniformly here; at LLM scale the analogue is goal/prefix
relabeling where verifiers admit it (maze goals, sub-goals in multi-step
proofs, partial-credit unit tests).

**Ablations (advmass teacher, 5 seeds).** Hindsight weight scale is monotone
on the toy — AUC 0.805 / 0.840 / 0.883 / 0.908 / 0.928 / 0.943 at scale
0.25→8 — because the toy's relabeled subtask is *exactly* correct and
gradients are exact; expect a knee (then collapse) on real models where
over-weighted imitation of self-generated prefixes can entrench errors and
kill diversity. Default stays 1.0 (the natural K=1 group weight); treat scale
as the imitation-strength knob the GPU/LLM runs must tune.

**Interaction with the teacher (16-level regime).** Hindsight partially
*substitutes* for the teacher: uniform+hindsight reaches 0.970 vs
advmass-alone 0.961 — recycling dead groups fixes much of what prompt
selection was avoiding. But they still stack (advmass+hindsight 0.978, best
in every regime tested), and the teacher retains its wall-clock advantage on
real models (it avoids *generating* doomed rollouts at all; hindsight only
salvages them after paying generation cost). Division of labor: teacher =
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
