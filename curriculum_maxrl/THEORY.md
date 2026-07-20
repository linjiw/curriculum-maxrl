# Advantage-mass analysis: a derived curriculum signal for MaxRL

Deep-dive into the MaxRL paper's math (arXiv:2602.02710, full text) yielding a
*derived* (not heuristic) teacher utility. All formulas verified by Monte
Carlo (200k trials) in this repo — see the session logs / reproduce with the
snippet at the bottom.

## 1. Setup

A prompt with true pass rate `p` gets a group of `N` i.i.d. rollouts with
`K ~ Binomial(N, p)` successes. Each estimator assigns per-rollout advantage
weights `w_j`. Define the **advantage mass** of the group as `Σ_j |w_j|` —
the total magnitude of learning signal the optimizer receives from this
prompt (the score functions `S_j` are multiplied by these weights, so mass ≈
the gradient budget the prompt commands).

## 2. Exact expected advantage mass per estimator

**MaxRL** (Algorithm 1: `w_succ = 1/K − 1/N`, `w_fail = −1/N`, group dropped
at K=0). For K ≥ 1: `Σ|w| = K(1/K − 1/N) + (N−K)/N = 2(1 − K/N)`. Hence

```
E[Σ|w|] = 2·(P(K≥1) − E[K]/N) = 2·(pass@N(p) − p)   —  EXACT
        = 2·(pass@N − pass@1)
```

**The expected MaxRL learning signal on a prompt equals twice the
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

**RLOO** (`w_j = (r_j − LOO-mean)/N`): `Σ|w| = 2K(N−K)/(N(N−1))`, hence

```
E[Σ|w|] = 2·p·(1−p)·N/(N−1) ≈ 2·p(1−p)   —  EXACT
```

RLOO's advantage mass **is** SFL's "learnability" p(1−p) (Rutherford et al.
2024) up to the constant. The learnability curriculum literature and the
RLOO estimator are the same object seen from two sides.

**GRPO** (`w_j = (r_j − mean)/(std+ε)/N`, degenerate groups K∈{0,N} give 0):

```
E[Σ|w|] = Σ_{K=1}^{N−1} P(K) · 2·sqrt( K(N−K)/(N(N−1)) ) / N · N ≈ 2·sqrt(p(1−p))·(1 − P(K∈{0,N}))
```

Numerically: at p=0.01, N=32 the population weight-function view of the paper
(`w(p)=1/√(p(1−p))` → mass ≈ 0.199) overstates the realized finite-sample
mass by 2× (exact: 0.100) because 72% of groups are all-fail and contribute
nothing. **The paper's population-level w(p) curves describe the
infinite-sample limit; at finite N every estimator's realized signal on hard
prompts is throttled by pass@N.** This sharpens the case for a teacher: no
choice of w(p) can put signal where groups die.

## 3. Consequences for the curriculum design

1. **Derived teacher utility.** Replace the heuristic frontier utility
   `u(p) = (1−(1−p)^N)(1−p)` with the exact advantage mass
   `u(p) = pass@N(p) − p = (1−(1−p)^N) − p`. Numerically the two are nearly
   identical (max deviation ~1% of range), which retroactively explains why
   the heuristic worked; but the derived form is (a) parameter-free, (b) an
   unbiased target for what the optimizer actually receives, and (c) directly
   estimable from group statistics.

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

## 4. The codebase already decouples T from N — enabling a *curriculum over the objective*

The paper's Algorithm 1 ties truncation order to group size (T = N). But the
repo contains an unpublished-in-paper estimator
(`oversample_subset_vr_weights` / `c_sub_TN` in
`verl/trainer/ppo/maclaurin.py`, eq. 51 of the appendix): per-success weight
`c_{T,N}(K)` such that with N rollouts the estimator is unbiased for the
**T-truncated** objective for any T ≤ N. Verified numerically here:

- `c_{N,N}(K) = 1/K` exactly (recovers Algorithm 1 at T = N);
- `E[c_{T,N}(K)·K] = 1−(1−p)^T = w_T(p)·p` to 4 decimals for
  T ∈ {1,4,16}, p ∈ {0.05,0.3,0.7}, N = 16 (100k-trial MC).

**Consequence:** T becomes a per-prompt knob independent of the rollout
budget. This opens a third integration axis beyond prompt selection and
rollout allocation — a **curriculum over the objective itself**:

- easy prompts (p̂ high): T = 1 → plain RL weighting, minimal variance;
- frontier prompts: T = N → full ML weighting where the higher-order
  pass@k terms matter;
- beyond-frontier prompts kept in-batch for exploration: large N (better
  chance of a first success + tighter p̂ posterior) with moderate T
  (bounding the 1/p variance blow-up the truncation exists to control).

The teacher already estimates p̂ per prompt, so annealing T_i by difficulty
is free. This mirrors how the population weight w_T(p) = (1−(1−p)^T)/p is
flat (≈T) for p ≪ 1/T and ≈1/p for p ≫ 1/T: choosing T_i ≈ 1/p̂_i puts every
prompt at the knee of its own weight curve.

**Empirical status: negative on the CPU testbed.** `weights_maxrl_t`
(validated: T=N recovers Algorithm 1 in 20 random cases) with
T_i = clip(1/p̂_i, 1, N) slightly *underperforms* fixed T=N
(advmass teacher: AUC 0.698 vs 0.704; uniform: 0.641 vs 0.653; 5 seeds).
Interpretation: lowering T only helps when the 1/p variance blow-up is the
binding constraint; on this testbed the group size (16–32) keeps variance
manageable, so shrinking T just weakens the beneficial hard-prompt
upweighting. Adaptive-T remains interesting only for regimes with very small
groups or extreme p̂ spreads — deprioritized.

## 5. Verification snippet

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
