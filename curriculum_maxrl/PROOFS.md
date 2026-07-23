# Formal results for curriculum-MaxRL

This note separates three MaxRL estimators that were previously conflated and
states exactly what the curriculum score does and does not prove.

Fix a task `x`. Let `p=p_θ(x)∈(0,1)` be its binary success probability,
`S_i=∇_θ log m_θ(z_i|x)` its rollout score, and let `N` rollouts be conditionally
i.i.d. with rewards `r_i∈{0,1}` and `K=Σ_i r_i`. We assume the reward/verifier is
independent of `θ`, the score identity is valid, and differentiation may pass
through the rollout expectation.

The order-`T` MaxRL objective and its gradient are

```text
J_T(p) = -Σ_{k=1}^T (1-p)^k/k,
∇J_T = w_T(p)∇p,       w_T(p) = [1-(1-p)^T]/p.
```

## Proposition 1 (Estimator conventions and the practical off-by-one)

Define the raw success-average estimator, its unbiased control-variate form,
and the practical dropped-group form:

```text
H_N = 1{K>0} (1/K) Σ_i r_i S_i,
C_N = H_N - (1/N) Σ_i S_i,
D_N = 1{K>0} [(1/K) Σ_i r_i S_i - (1/N) Σ_i S_i].
```

Then

```text
E[H_N] = E[C_N] = ∇J_N,
E[D_N] = ∇J_{N-1}                         (N≥2).
```

**Correction (verified 2026-07-23, 4M-trial MC).** The *practical* Algorithm-1
estimator (drop both terms when K=0) is unbiased for truncation order
**T = N−1**, not N: zeroing the control variate only on all-fail groups makes
it outcome-dependent, contributing −(1−p)^{N−1}∇p, and
w_N(p) − (1−p)^{N−1} = w_{N−1}(p). Eq. 9 (no CV) and Eq. 10 with the CV
retained at K=0 are both unbiased for T = N. MC at N=8, p≈0.23:
practical → 0.64678±0.0003 = ∇J^{N−1} (0.64683); eq9/eq10 → 0.675 = ∇J^N.
Credit: PR #1. None of the advantage-mass results below change (they are
statements about |w|, not about which J the mean gradient targets).

**Interpretation.** The learning signal a prompt commands equals twice the
probability it is *solvable within N attempts but not within one*. This is
the estimator's own zone-of-proximal-development functional — the teacher
utility is not a heuristic added on top of MaxRL but a quantity MaxRL already
computes implicitly.

For `N=1`, `D_1=0` identically. The repository's `maxrl_weights` implements
`D_N`; the paper's order-`N` unbiasedness theorem applies to `H_N`, and to
`C_N` only when its unconditional-score control variate is retained at `K=0`.

**Proof.** Conditional on `K>0`, the average successful score has expectation
`∇log p`. Therefore

```text
E[H_N] = P(K>0)∇log p
       = [1-(1-p)^N]∇p/p
       = ∇J_N.
```

The score identity gives `E[(1/N)Σ_i S_i]=0`, hence `E[C_N]=E[H_N]`. For the
dropped control variate, use the joint score identity on the all-fail event:

```text
E[1{K=0}(1/N)Σ_i S_i]
    = (1/N)∇P(K=0)
    = -(1-p)^(N-1)∇p.
```

Consequently

```text
E[1{K>0}(1/N)Σ_i S_i] = (1-p)^(N-1)∇p,
E[D_N] = {w_N(p)-(1-p)^(N-1)}∇p = w_{N-1}(p)∇p.  ∎
```

**Interpretation.** SFL's "learnability" curriculum objective `p(1-p)`
(Rutherford et al. 2024) is proportional to the expected coefficient mass of
the RLOO estimator. The curriculum literature and the estimator algebra
converge on the same scalar functional from opposite directions, and
Proposition 2 generalizes it to a compute-indexed MaxRL family
(`u_2(p)=p(1-p)` exactly and `u_1≡0`). The matching scalar priority does not
make the estimators or their population objectives identical.

For the subset estimator with nominal order `T≤N`, dropping the control
variate at `K=0` similarly changes its population weight to
`w_T(p)-(1-p)^(N-1)`. It targets `J_T` exactly only when that control variate is
retained on all groups.

## Proposition 2 (Exact coefficient mass of practical MaxRL)

For the scalar weights of `D_N`,

```text
A_N = Σ_i |w_i|,
E[A_N] = 2u_N(p),
u_N(p) = 1-(1-p)^N-p = (1-p)pass@(N-1,p).
```

For `N≥2`, `u_N` is strictly concave on `(0,1)` and has unique maximizer

```text
p*_N = 1-N^(-1/(N-1))
     = ln(N)/N + O((ln N)^2/N^2).
```

**Proof.** Given `K≥1`, there are `K` weights `1/K-1/N≥0` and `N-K`
weights `-1/N`, so

```text
A_N = K(1/K-1/N)+(N-K)/N = 2(1-K/N).
```

This also holds at `K=N`, where every weight is zero. At `K=0`, `A_N=0`.
Thus

```text
E[A_N] = 2E[(1-K/N)1{K≥1}]
       = 2(P(K≥1)-E[K]/N)
       = 2(1-(1-p)^N-p).
```

Finally, `u'_N=N(1-p)^(N-1)-1` and
`u''_N=-N(N-1)(1-p)^(N-2)<0`. Solving `u'_N=0` and expanding the exponential
gives the stated maximizer and asymptotic. ∎

**What is exact.** `A_N` is the `L1` mass of scalar estimator coefficients.
It is a useful observable proxy, not the gradient norm or guaranteed loss
decrease. If `||S_i||≤G`, then

```text
||Σ_i w_i S_i|| ≤ G A_N,
E||Σ_i w_i S_i|| ≤ 2G u_N(p).
```

Score norms, directions, cancellation, optimizer state, and parameter sharing
still determine actual learning progress.

## Proposition 3 (Myopic rollout allocation)

Fix known pass rates `p_i∈(0,1)`, integer bounds `1≤L_i≤N_i≤U_i`, and a
feasible budget `Σ_i L_i≤B≤Σ_i U_i`. The allocation maximizing one-step total
half-mass (equivalently, total expected coefficient mass after multiplying by
the constant two)

```text
Σ_i u_{N_i}(p_i)  subject to  Σ_i N_i=B
```

is obtained by starting at `N_i=L_i` and repeatedly assigning the next rollout
to a feasible task with largest marginal

```text
Δ_i(N_i) = u_{N_i+1}(p_i)-u_{N_i}(p_i)
         = p_i(1-p_i)^(N_i).
```

**Proof.** Each marginal is positive and strictly decreases with `N_i`.
Separable discrete concavity makes the greedy allocation exact by the usual
exchange argument. ∎

The marginal is the probability that rollout `N_i+1` is the group's first
success. This is a myopic coefficient-mass allocation theorem, not a theorem
about optimal long-horizon training or unknown/nonstationary `p_i`.

## Proposition 4 (RLOO coefficient mass equals learnability)

For the repository's RLOO normalization

```text
w_i = (r_i-r̄_{-i})/N,
```

and `N≥2`,

```text
E[Σ_i|w_i|] = 2p(1-p).
```

**Proof.** Conditional on `K`, a success has magnitude
`(N-K)/(N(N-1))` and a failure has magnitude `K/(N(N-1))`. Hence

```text
Σ_i|w_i| = 2K(N-K)/(N(N-1)).
```

Because `E[K(N-K)]=N(N-1)p(1-p)`, the result follows. ∎

This matches SFL-style Bernoulli learnability up to a constant. Practical
MaxRL coefficient mass equals RLOO's exactly at **`N=2`**, not `N=1`;
`u_1` is identically zero.

## Proposition 5 (Exact MaxRL/RLOO mass comparison)

For `N≥2` and every `p∈(0,1)`,

```text
E[A_MaxRL]/E[A_RLOO]
    = u_N(p)/(p(1-p))
    = [1-(1-p)^(N-1)]/p
    = Σ_{j=0}^{N-2}(1-p)^j ∈ [1,N-1].
```

The ratio tends to `N-1` as `p↓0` and to `1` as `p↑1`; equality holds for
all `p` when `N=2`. This compares scalar coefficient mass only. It does not by
itself prove that one estimator is safer, lower variance, or produces a larger
gradient norm.

## Proposition 6 (Honest characterization of hindsight updates)

Fix `θ`, the source-task mixture, and a relabeled goal `g`. Let `A_g` be the
positive-probability event that the source-group relabeler selects and accepts
`g`. (In an adaptive stream, read all probabilities below as conditional on
the current decision filtration.) Let

- `P_{θ,g}=m_θ(·|g)^⊗N` be the joint law of a fresh target-task group;
- `Q_{θ,g}=Law(rewritten group | A_g)` be the joint law after source-task
  failure conditioning, data-dependent goal selection, trajectory rewriting,
  relabeling, and acceptance; and
- `h_{θ,g}` be the actual practical centered group update, with **every**
  trajectory scored under goal `g`.

Then

```text
U_g^HS = E_{Q_{θ,g}}[h_{θ,g}],
∇J_{N-1,g} = E_{P_{θ,g}}[h_{θ,g}].
```

Thus, **per accepted `g` update**, equality of the update moments,
`E_Q[h_{θ,g}]=E_P[h_{θ,g}]`, is necessary and sufficient for exactness.
Equality of the full joint laws, `Q_{θ,g}=P_{θ,g}`, is a convenient
sufficient—not necessary—condition. If every trajectory score is bounded by
`G` under both laws, then for an unscaled auxiliary update and under the
conventional total-variation definition

```text
||U_g^HS-∇J_{N-1,g}||
    ≤ 4G·TV(Q_{θ,g},P_{θ,g}).
```

**Proof.** The first two identities are definitions plus Proposition 1. The
practical weights have `Σ_i|w_i|≤2(1-1/N)≤2`, so
`||h_{θ,g}||≤2(1-1/N)G≤2G`; the standard bounded expectation/TV inequality
gives the stated (slightly loose) bound. ∎

This statement is deliberately conditional. In an adaptive stream write
`π_t(g)=P(A_g|F_t)`; under the fixed setup above this reduces to
`π_g=P(A_g)`. If the auxiliary scale is `λ`, then the mean contribution per
source group is

```text
λ π_g E_Q[h_{θ,g}].
```

Across a discrete credited-goal pool, the conditional auxiliary-stream moment
is therefore

```text
E[U_t^HS|F_t]
  = λ Σ_g π_t(g) E_{Q_{t,g}}[h_{θ_t,g}].
```

Consequently `Q=P` does not by itself recover a desired global task-mixture
gradient: the selection probabilities `π_g` and scale `λ` still determine the
mixture and magnitude. Even if every per-goal moment were exact, preserving a
fixed goal mixture would require selection-frequency correction by
`ρ_g/π_t(g)`, not requested-task correction by `ρ_g/q_{t,g}`; neither factor
repairs `Q_{t,g}≠P_{θ_t,g}`. For the scaled per-accepted comparison,

```text
||λE_Q[h]-λE_P[h]|| ≤ 4|λ|G TV(Q,P).
```

If instead `λE_Q[h]` is compared with the unscaled fresh moment `E_P[h]`, a
separate scale term remains:

```text
||λE_Q[h]-E_P[h]||
  ≤ 4|λ|G TV(Q,P) + |λ-1| ||E_P[h]||.
```

Correct relabeled rewards and rewriting the target conditioning for all
positive- and negative-weight trajectories are necessary for semantic
validity, but they do not imply `Q=P`. The relabeler conditions on a dead
source group, chooses `g` from that same group, and guarantees a relabeled
success, so selection bias generally remains.

Even in the favorable special case
`Q=P_{θ,g}(·|K>0)`, the centered update satisfies

```text
E[D_N|K>0]
  = [pass@(N-1,p_g)/pass@(N,p_g)] ∇log p_g,
```

which is direction-aligned but not equal in scale to the ML gradient. Cosine
similarity of one does not establish unbiasedness.

The Acrobot relabeler accepts only mixed rewritten groups, `0<K<N`. Under the
corresponding favorable hypothetical law,

```text
Q=P_{θ,g}(·|0<K<N),
E[D_N|0<K<N]
  = pass@(N-1,p_g) / [1-(1-p_g)^N-p_g^N] · ∇log p_g.
```

This is again direction-aligned and generally scale-biased. The denominator
is the probability that a fresh target group is mixed; it differs from the
`K>0` acceptance probability above.

A narrower sufficient condition is available for success-only relabeling.
Within an accepted relabeled group, write `r'_i` for recomputed rewards and
`K'=Σ_i r'_i`, then draw index `J` with conditional probability `r'_J/K'`
(equivalently, use the success-average weights). If the marginal law of
`Z'_J`, after data-dependent goal selection and acceptance, is
`m_θ(·|g, success)` and its score is evaluated under `g`, then the per-accepted
update has expectation `∇log p_g`. Pooling every success across groups is not
equivalent because that weights groups by `K'`. This does not make the update
an unbiased `∇J_{N-1,g}` estimator or recover a global target mixture. The
selected-positive marginal-law assumption must be tested; verifier correctness
alone is insufficient.

## Proposition 7 (Uniform-floor visitation guarantee)

Suppose an adaptive teacher samples from distributions satisfying
`q_{t,i}≥f/m` for every task and history, where `m` is the pool size and
`0<f≤1`. Then for any task `i` and horizon `H`,

```text
P(no visit to i in the next H group draws | history)
    ≤ (1-f/m)^H ≤ exp(-fH/m).
```

The expected waiting time to revisit a task is at most `m/f` group draws.

**Proof.** Condition sequentially on the adaptive history. At every draw the
conditional miss probability is at most `1-f/m`; multiply those bounds and
use `1-a≤e^{-a}`. The waiting-time bound follows from the geometric tail. ∎

This proves visitation, not posterior accuracy, change detection, or regret.
The implementation's exponentially discounted Beta counts are useful
pseudo-counts for Thompson-style randomization, not an exact Bayesian
posterior for a drifting policy.

## Adaptive task sampling changes the optimized distribution

Let `F_t` be the decision filtration after the teacher has formed its current
distribution (including the current Thompson pseudo-draw) and before it draws
task `I_t`. If `q_{t,i}` is `F_t`-measurable and the group is then sampled from
the frozen current policy, Proposition 1 gives

```text
E[D_{N,I_t}|F_t] = Σ_i q_{t,i} ∇J_{N-1,i}.
```

The right side is the gradient of the current mixture only when `q_t` is
treated as a stop-gradient at that decision. Because `q_t` changes with the
history and indirectly with `θ`, there need not be one static objective whose
gradient the full adaptive process follows. If `q_t=ρ`, the conditional update
is already unbiased for the target mixture. Otherwise importance weights
`ρ_i/q_{t,i}` recover that target conditionally when `q_{t,i}>0` on its
support (special gradient cancellations can also make the unweighted mixtures
coincide). With a uniform floor `f`, those weights are bounded by `1/f` for a
uniform target distribution. Proportional-to-utility sampling is a smooth
heuristic priority rule; absent coverage or variance constraints, hard
`argmax` selection—not proportional sampling—maximizes known one-step utility.

## Proposition 8 (Conditional one-step progress; no unconditional curriculum theorem)

Let the fixed evaluation objective be

```text
F_ρ(θ) = Σ_i ρ_i J_{N-1,i}(θ),
g_ρ = ∇F_ρ(θ),
g_i = ∇J_{N-1,i}(θ),
```

and suppose `F_ρ` has `L`-Lipschitz gradient. Conditional on the decision
filtration and current parameters, sample `I~q`, obtain an unbiased practical
estimator `D_I` with `E[D_i]=g_i`, and take one ascent step
`θ⁺=θ+ηD_I`. Define

```text
G_q = Σ_i q_i g_i,
M_q = Σ_i q_i E||D_i||².
```

Then

```text
E[F_ρ(θ⁺)]
  ≥ F_ρ(θ) + η <g_ρ,G_q> - (Lη²/2) M_q.
```

Therefore a curriculum `q` has a no-worse guaranteed one-step lower bound than
a reference mixture `q⁰` when its target-gradient alignment gain is large
enough to pay any second-moment cost:

```text
η <g_ρ,G_q-G_q⁰> ≥ (Lη²/2)(M_q-M_q⁰).
```

Strict inequality gives a strictly better lower bound.

**Proof.** `L`-smoothness gives

```text
F_ρ(θ+v) ≥ F_ρ(θ) + <g_ρ,v> - (L/2)||v||².
```

Set `v=ηD_I`, take the conditional expectation, and use the definitions of
`G_q` and `M_q`. Subtracting the two lower bounds yields the comparison. ∎

This proposition supplies the missing bridge that experiments must test.
Coefficient mass `u_N(p_i)` does not determine either `<g_ρ,g_i>` or
`E||D_i||²`, so the exact identity in Proposition 2 cannot establish the
displayed inequality by itself.

For comparison, a target-preserving importance-weighted update

```text
D_I^IW = (ρ_I/q_I) D_I
```

has conditional mean `g_ρ` whenever `q_i>0` on the support of `ρ`. Its second
moment is

```text
E||D_I^IW||² = Σ_i (ρ_i²/q_i) E||D_i||².
```

If those second moments were known and positive, and there were no sampling
floor or other lower-bound constraint, the mixture minimizing this quantity
on the positive support of `ρ` would satisfy

```text
q_i ∝ ρ_i sqrt(E||D_i||²).
```

That second-moment-optimal rule (equivalently variance-optimal because the
importance-weighted mean is fixed) follows by Lagrange multipliers or
Cauchy--Schwarz. It is not generally the coefficient-mass teacher and says
nothing by itself about environment-transition cost. The mean and variance
statements are per requested group; unequal task transition or token costs
require a separate cost-aware analysis. It defines a distinct target-preserving
research direction rather than a proof of the
adaptive-objective method used in the current experiments.
