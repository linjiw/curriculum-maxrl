# Curriculum-MaxRL: Estimator-Derived Task Sampling and Verified Hindsight Recycling

*A local mechanism study for binary-reward reinforcement learning*

## Abstract

Under the practical centered MaxRL update studied here, rollout groups at two
extremes receive zero scalar coefficients: every rollout succeeds or every
rollout fails. We study whether the finite-sample algebra of MaxRL can define a
curriculum for reducing those groups, and
whether verifier-backed hindsight can recover useful auxiliary targets from
selected all-fail groups. The contribution is deliberately narrow: we derive
and test mechanisms for Curriculum-MaxRL on local tabular, tile-coded, and
small neural-control problems; we do not claim general state of
the art.

The analysis first separates three estimators that are easy to conflate. The
raw MaxRL success average and an always-retained control-variate estimator
target truncation order `T=N`, while the practical estimator that drops both
terms when a group has no successes targets `T=N-1`. For that practical
estimator, the expected `L1` mass of its scalar coefficients is exactly

```text
2 u_N(p) = 2 [1 - (1-p)^N - p]
          = 2 [pass@N(p) - pass@1(p)].
```

This is twice the probability that the first attempt fails but at least one of
the remaining `N-1` attempts succeeds. It is strictly concave and peaks at
`p*=1-N^(-1/(N-1))`, approximately `ln(N)/N`. Curriculum-MaxRL uses this
quantity as a task-priority score, estimated with discounted Beta
pseudo-counts, mixed with a uniform replay floor, and optionally sharpened by
a concentration exponent. All-fail groups may be relabeled to verifier-valid
achieved goals, but per-accepted-goal exactness requires an update-moment or
trajectory-law condition. Selection probabilities and auxiliary scale still
determine the global update; verifier correctness alone is insufficient.

Exact enumeration verifies the estimator identities to numerical precision.
In a paired 12-seed skill-chain component ablation, the proportional teacher
adds `+0.0718` checkpoint mean over uniform without hindsight, stronger
concentration adds `+0.0494`, and centered hindsight adds `+0.1050` under the
concentrated teacher. The teacher-by-hindsight and concentration-by-hindsight
interactions are negative (`-0.0586` and `-0.0420`): the components are
complementary but subadditive, not synergistic. All five contrasts survive a
Holm correction over the reported 15-contrast family.
On a corrected ten-seed tile-coded Gymnasium MountainCar mechanism study, concentrated
coefficient-mass sampling (`gamma=4`) improves mean-pass AUC over uniform by
`+0.141` (paired 95% bootstrap CI `[0.076, 0.202]`), and centered and
success-only hindsight add `+0.191 [0.155, 0.231]` and
`+0.197 [0.160, 0.238]`, respectively. These contrasts survive Holm
correction over the reported family of nine AUC comparisons. However, proportional
coefficient-mass sampling (`gamma=1`) is not separated in MountainCar from
uniform, a nearby legacy score, or Bernoulli learnability. A corrected
goal-conditioned gridworld and a synthetic skill chain provide consistent
local mechanism evidence. In the independently verified neural Acrobot V3
confirmation, the frontier-`u_16` teacher improved shared-H64 transition-AUC
over uniform by `+0.0363524` across 20 paired seeds (paired 95% bootstrap CI
`[0.0164536, 0.0553949]`; exact two-sided sign-flip `p=0.00263977`). The
observed mean and test result satisfied the preregistered `+0.03`/`p<=0.05`
decision rule; the interval does not establish that the population effect
exceeds `0.03`. Arm-level AUC means were `0.648669` for uniform and `0.685021`
for the teacher; final mean-pass values were `0.864258` and `0.916992`,
respectively, as secondary endpoints. This supports only a positive
shared-policy efficacy conclusion on the fixed eight-threshold family. It does
not test or establish transfer, capacity effects, hindsight, or Acrobot
performance in general. A subsequent effect-blind V4A feasibility study
passed integrity verification but failed its every-run hindsight-preview gate
in 3/9 scale-zero runs, so the optimizer-matched V4B factorial was not
authorized or run; V4A contributes no hindsight-effect estimate. A fresh V5A
full-grid feasibility study then completed 27/27 runs, passed all
learning-outcome-field-blind gates, selected `U*=250`, and independently
authorized V5B. V5B subsequently completed all 180 runs with zero run
failures and passed raw-data integrity checks (53,510 group records, 45,000
updates, and 1,080 checkpoints), but the frozen independent analyzer failed
its exact diagnostic-reconstruction rule. The runner used NumPy norm
reductions while the analyzer used Python scalar reductions; 377 of 720
diagnostic floats differed by at most `1.9984014443252818e-15` and 11 ULP.
Although these step norms are diagnostic rather than endpoints, exact
runner/analyzer dictionary agreement was the predeclared acceptance rule.
V5B is therefore a procedural NO-GO and no V5B primary contrast or hindsight
effect is reported. A post-hoc compatibility audit passed the remaining
checks but is non-authorizing. A separate capacity-matched neural MountainCar V1R2
development matrix also completed all reconstruction checks but returned
NO-GO: hardest-goal AUC was zero in all 15 runs and no all-pass groups occurred;
its confirmatory seeds remain untouched. The retained results therefore
support estimator-derived prioritization, one
narrow neural shared-policy efficacy claim, a still-open transfer-channel
hypothesis, and verified hindsight recycling as testable components; corrected
GPU, broader neural, and LLM experiments remain necessary.

Throughout this paper, “registered,” “sealed,” and “predeclared” denote local
source/runtime locks created before the corresponding seed block. They are not
externally timestamped preregistrations. The V3 and later manifests match
current source bytes; the historical V2 lock points to an earlier runner hash
whose exact bytes are not present at HEAD.

## 1. Problem statement

Consider a finite task pool `X={x_1,...,x_m}`. For task `x`, a policy
`m_theta(z|x)` produces a trajectory or response `z`, and a verifier returns a
binary reward `r(z,x) in {0,1}`. The success probability is

```text
p_theta(x) = E[r(z,x) | x].
```

Training generates groups of `N` conditionally independent rollouts for a
selected task. If `K` of those rollouts succeed, the practical dropped-group
estimator studied here produces no update at either extreme:

- `K=0`: no observed success anchors a success-conditioned update;
- `K=N`: the practical centered coefficients cancel to zero.

The research question is not simply whether hard tasks should receive more
weight. MaxRL already changes gradient weighting as a function of task
success probability. The sharper question is:

> Can the estimator's own finite-group coefficients identify where another
> rollout group is most likely to be informative, and can an all-fail group be
> turned into a semantically valid auxiliary update without overstating its
> statistical exactness?

Curriculum-MaxRL answers with two components:

1. **Frontier sampling:** prioritize tasks by a scalar coefficient-mass score
   derived from the practical MaxRL estimator.
2. **Verified hindsight recycling:** when a selected group is all-fail,
   optionally relabel its trajectories to achieved goals using the
   environment's verifier and a complete conditioning rewrite.

The first component reallocates future rollouts. It cannot retroactively
recover a realized all-fail group. The second changes what that group yields
by adding an auxiliary target, but may introduce selection bias. Keeping those
roles separate is central to the method and the claims.

The paper makes four contributions:

1. an estimator audit that identifies the `T=N-1` population objective of the
   practical dropped-group update;
2. an exact coefficient-mass identity and its compute-indexed frontier score;
3. a verifier and trajectory-law contract for hindsight recycling; and
4. corrected local experiments that separate score shape, concentration, and
   hindsight estimator, plus an independently implemented neural experiment
   and capacity-matched development controls whose failed gates delimit rather
   than establish parameter-sharing and transfer claims.

## 2. Motivation and related concepts

### The three channels

Everything the method does flows through three channels, and every experiment
we ran gains its effect through exactly one of them:

**Channel 1 — waste avoidance (the teacher).** Don't roll out where the
estimator will emit nothing. Worth a consistent but bounded +0.05–0.08 AUC —
bounded by the oracle ceiling, because allocation can only redistribute
signal that exists (a perfect sampler collects just 0.4% more advantage mass
than our posterior).

**Channel 2 — signal creation (hindsight).** Create verifier-valid auxiliary
targets from failures already paid for. This is the only channel that can
yield training signal from a realized all-fail group, although equality to a
fresh target update still requires the law or moment conditions proved below.
Its observed local gain varies with how much a relabeled skill can *compound*:
about +0.22 AUC on fixed task sets and +0.01 on one-shot task streams.

**Channel 3 — objective safety (MaxRL weighting underneath).** Channels 1–2
are not objective-agnostic add-ons: the identical teacher grew coverage under
MaxRL in every seed and amplified GRPO's collapse in every seed. The
objective decides whether a curriculum is safe at all.

One line: **the teacher allocates, hindsight creates, the objective decides
whether either is safe.** The regime map, practitioner playbook, and graded
claim inventory live in EVIDENCE.md; the interactive version of this section
(a live frontier-walk simulation) is on the project site.

### What problem this addresses

**Compute allocation in RLVR.** Rollout generation dominates the cost of RL
post-training, and on hard task distributions most of it buys nothing. Prior
fixes either pay for the waste differently (DAPO's dynamic sampling redraws
until a live group appears — the discards still cost GPU-hours), or gate on
heuristic difficulty bands with their own hyperparameters (ADARFT), or
target learnability p(1−p) — the right instinct, and exactly the N=2 member of the
real functional. Deriving the rule from the estimator's algebra gives the
band, its width, and its compute-scaling (`ln N/N`) without a separate
frontier-location parameter: the rollout budget `N` is the location knob.
The operational teacher still has explicitly reported tracking, replay-floor,
and optional concentration settings.

### 2.1 MaxRL

[MaxRL](https://arxiv.org/abs/2602.02710) interprets binary-reward policy
learning through a truncated maximum-likelihood objective. For truncation
order `T`, define

```text
J_T(p) = - sum_{k=1}^T (1-p)^k / k,
grad J_T = w_T(p) grad p,
w_T(p) = [1-(1-p)^T] / p.
```

As `T` grows, `w_T(p)` approaches the maximum-likelihood factor `1/p` and
places relatively more population weight on low-success tasks. This is an
implicit gradient-level curriculum. It acts only after a task has been
sampled, however, and its practical dropped-group form emits zero scalar
weights on both all-fail and all-pass groups.

### 2.2 Curriculum and learnability

Curriculum learning changes the distribution from which tasks are generated.
A common Bernoulli learnability score is `p(1-p)`, which is largest at
intermediate success probability. The score appears, for example, in
[SFL-style learnability sampling](https://arxiv.org/abs/2408.15099). We show
that the expected scalar coefficient mass of the repository's RLOO
normalization is `2p(1-p)`, and that practical MaxRL has exactly the same mass
when `N=2`. For general `N`, the MaxRL-derived score moves the frontier as a
function of the rollout budget.

This algebraic connection does not make the estimators or curriculum
objectives identical. It only identifies a shared scalar prioritization
quantity in the `N=2` case.

### 2.3 Hindsight relabeling

[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) turns a failed
attempt at one goal into a successful example for a goal that was actually
achieved. Curriculum-MaxRL uses the same conceptual move only when the source
group is all-fail. A relabel is accepted only if the environment can:

1. verify the relabeled reward under the achieved goal; and
2. rewrite every goal-conditioned trajectory, observation, or prompt so that
   all score-function terms are evaluated under the relabeled goal.

These conditions establish semantic validity, not statistical unbiasedness.
The data-dependent choice of achieved goal can still change the trajectory
law.

### 2.4 Dynamic sampling and filtering

[DAPO](https://arxiv.org/abs/2503.14476) oversamples prompts and filters
accuracy-zero and accuracy-one rollout groups when filling a training batch.
Curriculum-MaxRL acts at a different point: its frontier teacher changes future
task probabilities before generation, while verified hindsight may reuse a
selected `K=0` group as an auxiliary achieved-goal update. The approaches can
therefore spend and account for generation differently. The exact
DAPO/uniform equivalence reported later is specific to our sequential
synthetic simulator and matched generated-group budget; it is not a general
equivalence claim.

## 3. Estimator conventions

Fix a task and write `p=p_theta(x) in (0,1)`; endpoint statements may be read
by continuous limits. Let

```text
S_i = grad_theta log m_theta(z_i | x),
K   = sum_i r_i.
```

Assume the verifier does not depend on `theta`, the score identity holds, and
differentiation may pass through the rollout expectation. Three related
estimators must be distinguished.

### 3.1 Raw success average

```text
H_N = 1{K>0} (1/K) sum_i r_i S_i.
```

Conditioned on at least one success, the average successful score has
expectation `grad log p`. Therefore

```text
E[H_N] = [1-(1-p)^N] grad p / p = grad J_N.
```

### 3.2 Always-retained control variate

```text
C_N = H_N - (1/N) sum_i S_i.
```

The unconditional score term has mean zero, including on `K=0` groups, so
`E[C_N]=grad J_N`.

### 3.3 Practical dropped-group estimator

The implementation studied in this repository uses

```text
D_N = 1{K>0} [(1/K) sum_i r_i S_i - (1/N) sum_i S_i].
```

Equivalently, for `K>0` each rollout receives scalar weight
`r_i/K-1/N`; for `K=0` every weight is zero. Dropping the control variate
outcome-dependently changes the expectation:

```text
E[D_N] = grad J_{N-1},       N >= 2.
```

To see the shift, apply the joint score identity to the all-fail event:

```text
E[1{K=0}(1/N) sum_i S_i]
  = (1/N) grad P(K=0)
  = -(1-p)^(N-1) grad p.
```

The average-score term retained on `K>0` therefore has expectation
`(1-p)^(N-1) grad p`, and

```text
E[D_N]
  = [w_N(p)-(1-p)^(N-1)] grad p
  = w_{N-1}(p) grad p
  = grad J_{N-1}.
```

For `N=1`, `D_1` is identically zero. The order-`N` statement for `H_N` and
`C_N` therefore must not be transferred to `D_N`. This off-by-one distinction
is both a proof correction and an implementation contract.

## 4. Estimator-derived curriculum utility

### 4.1 Exact coefficient mass

Let `A_N` be the `L1` mass of the practical scalar coefficients. Conditional
on `1 <= K <= N`,

```text
A_N = K(1/K-1/N) + (N-K)/N = 2(1-K/N).
```

At `K=0` the group is dropped, and at `K=N` all coefficients are zero. Taking
expectations gives

```text
E[A_N] = 2 [P(K>=1) - E[K]/N]
       = 2 [1-(1-p)^N-p]
       = 2 u_N(p).
```

The half-mass

```text
u_N(p) = 1-(1-p)^N-p
```

is the probability that one designated first attempt fails while at least one
of the other `N-1` attempts succeeds. Its derivatives are

```text
u'_N(p)  = N(1-p)^(N-1)-1,
u''_N(p) = -N(N-1)(1-p)^(N-2),
```

so for `N>=2` it is strictly concave and has unique maximizer

```text
p*_N = 1-N^(-1/(N-1)) = ln(N)/N + O((ln N)^2/N^2).
```

This gives a compute-indexed frontier: changing the number of rollouts changes
the success-probability region that maximizes expected practical coefficient
mass. It does not maximize the probability of a merely mixed group, whose
binary-outcome probability has a different shape.

### 4.2 What the utility does not prove

`A_N` is a scalar coefficient statistic, not the gradient norm, variance,
loss decrease, or long-horizon value of training a task. If every trajectory
score has norm at most `G`, then

```text
||sum_i w_i S_i|| <= G A_N,
E ||sum_i w_i S_i|| <= 2G u_N(p).
```

This is an upper bound, not an equality. Score directions, cancellation,
representation sharing, optimizer state, and downstream transfer all matter.
Using `u_N(p)` to order heterogeneous tasks is therefore a **monotone-proxy
assumption**: taskwise score norms must not be so strongly negatively
associated with `u_N(p)` that they reverse its ordering. Positive association,
as observed when harder maze tasks also produce longer trajectories and larger
score norms, reinforces the ordering but does not turn it into a gradient-norm
theorem.
The concentration exponent used below is consequently an empirical task-graph
knob, not a theorem.

### 4.3 Relation to RLOO and finite-group scaling

For the repository's RLOO normalization and `N>=2`,

```text
E[sum_i |w_i^RLOO|] = 2p(1-p).
```

Conditional on `K`, its coefficient mass is
`2K(N-K)/[N(N-1)]`; using
`E[K(N-K)]=N(N-1)p(1-p)` gives the stated expectation.

For practical MaxRL,

```text
E[A_N] / E[A_RLOO]
  = [1-(1-p)^(N-1)]/p
  = sum_{j=0}^{N-2} (1-p)^j,
```

which lies between `1` and `N-1`, tends to `N-1` as `p` approaches zero,
and equals one for all `p` when `N=2`. This remains a coefficient-mass
comparison only.

### 4.4 Optional myopic rollout allocation

If pass rates `p_i in (0,1)` are known and fixed, task `i` has integer rollout
bounds `1 <= L_i <= N_i <= U_i`, and a budget `B` satisfies
`sum_i L_i <= B <= sum_i U_i`, then maximizing

```text
sum_i u_{N_i}(p_i)   subject to   sum_i N_i = B
```

is a separable discrete-concave problem. Starting at the lower bounds and
greedily assigning each of the remaining `B-sum_i L_i` rollouts to the largest marginal

```text
u_{N_i+1}(p_i)-u_{N_i}(p_i) = p_i(1-p_i)^(N_i)
```

is exact for that one-step proxy. The marginal is the probability that the
next rollout is the group's first success. This is not a long-horizon
allocation theorem and is not used in the main experiments.

The proof is the standard exchange argument for separable discrete concavity:
each task's marginal is positive and decreases geometrically with its assigned
rollout count, so an allocation that omits a larger feasible marginal in favor
of a smaller one can be improved by swapping those assignments.

### 4.5 Theorem boundary and empirical hypotheses

The mathematical results justify a scoring rule, not an optimization guarantee.
To keep the argument falsifiable, we separate proved statements from mechanisms
that experiments must establish:

| statement | status | scope |
|---|---|---|
| `E[D_N]=grad J_{N-1}` | proved | fixed task, frozen policy, independent on-policy rollouts, theta-independent verifier, valid score identity |
| `E[A_N]=2u_N(p)` and the unique maximizer `p*_N` | proved | scalar coefficients of the practical dropped-group estimator |
| greedy variable-`N` allocation is optimal | proved | only for the stated one-step separable proxy with fixed known pass rates and integer bounds |
| the uniform floor bounds revisit probability and expected wait | proved | visitation, not learning, tracking, or regret |
| the unscaled per-accepted-goal hindsight bias is bounded by `4G TV(Q,P)` | proved when whole-trajectory score norms are bounded by `G` | selection frequency, global mixture, auxiliary scale, and whether `TV(Q,P)` is small remain unresolved |
| the smooth target objective obeys the one-step alignment-minus-second-moment bound | proved under `L`-smoothness and conditional estimator unbiasedness | whether coefficient-mass sampling improves that bound is empirical |
| high `u_N` causes larger gradients or faster learning | empirical hypothesis | depends on score geometry, optimizer state, and sampled tasks |
| shared parameters transmit a frontier update to harder tasks | empirical causal hypothesis | requires a behaviorally adequate capacity-controlled intervention |
| a particular `gamma`, decay, floor, or hindsight scale is best | empirical hypothesis | no universal optimum follows from the coefficient identity |

Accordingly, a positive learning curve can validate a tested configuration but
cannot strengthen any theorem above, and a failed experiment does not falsify
the coefficient-mass identity. It instead falsifies or narrows the proposed
bridge from that identity to useful optimization.

## 5. Curriculum-MaxRL

### 5.1 Discounted evidence model

For each task `i`, Curriculum-MaxRL maintains discounted Beta pseudo-counts
`alpha_i,beta_i`, initialized to one. After observing `K` successes in `N`
requested rollouts,

```text
alpha_i <- 1 + d(alpha_i-1) + K,
beta_i  <- 1 + d(beta_i-1)  + (N-K),
```

where `d` is a decay factor. Because the policy changes, these are tracking
pseudo-counts rather than an exact Bayesian posterior.

At each sampling decision, draw

```text
p_tilde_i ~ Beta(alpha_i,beta_i)
```

and define

```text
s_i = u_N(p_tilde_i)^gamma,
q_i = (1-f) s_i / sum_j s_j + f/m.
```

If every `s_i` is numerically zero, the implementation falls back to uniform
sampling. The uniform floor `f` guarantees

```text
P(no visit to task i in H draws | history) <= exp(-fH/m)
```

and expected revisit time at most `m/f` draws. It does not guarantee posterior
accuracy, change detection, or regret.

The bound follows by conditioning on the adaptive history: each conditional
miss probability is at most `1-f/m`, so the `H`-draw miss probability is at
most `(1-f/m)^H <= exp(-fH/m)`; summing the geometric tail gives the waiting
time bound.

The default implementation uses `d=0.7`, `f=0.1`, and `gamma=1`; local
shared-skill experiments also test `gamma=4`. Results below show that
concentration matters in the tested shared-policy settings, while the best
`gamma` remains task dependent.

### 5.2 The optimized task distribution changes

Let the teacher choose task `i` with history-measurable probability `q_{t,i}`.
For the practical estimator,

```text
E[D_{N,I_t} | history] = sum_i q_{t,i} grad J_{N-1,i}.
```

Thus Curriculum-MaxRL follows a time-varying teacher-weighted update. It is an
unbiased gradient of a target mixture `rho` when `q_t=rho`; otherwise this is
not true in general. Importance weights `rho_i/q_{t,i}` recover the target
mixture when support is adequate. The experiments intentionally do not apply
that correction: they study adaptive training allocation, not unbiased
optimization of the original uniform task objective.

## 6. Verified hindsight recycling

### 6.1 Semantic contract

Only a source group with `K=0` is eligible. An environment-specific relabeler
selects an achieved goal `g`, recomputes rewards under the verifier, and
rewrites every goal-dependent trajectory field before any score is evaluated.
All-pass groups are not relabeled. Relabeled outcomes are also excluded from
the teacher's pseudo-count update; otherwise selected successes can inflate
the teacher's estimate of natural requested-task competence.

For episodic reach tasks, the corrected local adapters truncate a relabeled
successful trace at its first hit of `g`, matching the stopping rule of a fresh
rollout for that goal. This removes one avoidable source of mismatch but does
not establish equality of the full laws.

### 6.2 Statistical condition

Fix the current policy and source mixture. For a relabeled goal `g`, let `A_g`
be the event that the relabeler selects and accepts `g`, let `P_{theta,g}` be
the joint law of a fresh `N`-rollout target group, and let
`Q_{theta,g}=Law(rewritten group|A_g)`. Let `h_{theta,g}` be the practical
centered group update evaluated entirely under `g`. Then, per accepted update,

```text
U_g^HS            = E_Q[h_{theta,g}],
grad J_{N-1,g}    = E_P[h_{theta,g}].
```

Equality of these update moments is necessary and sufficient for per-accepted
exactness.
Full joint-law equality `Q=P` is sufficient but not necessary. If trajectory
score norms are bounded by `G`, then

```text
||U_g^HS-grad J_{N-1,g}|| <= 4G TV(Q,P).
```

The practical coefficients satisfy `sum_i |w_i|<=2`, hence
`||h_{theta,g}||<=2G`; the conventional bounded-expectation inequality for
total variation yields the factor four.

Selection probability and auxiliary scale still determine the global update,
so `Q=P` alone does not recover a desired task-mixture gradient. Even in the
favorable case `Q=P(.|K>0)`, the centered update is generally
direction-aligned but scale-shifted:

```text
E[D_N | K>0]
  = [pass@(N-1,p_g)/pass@(N,p_g)] grad log p_g.
```

Cosine similarity of one therefore does not prove unbiasedness. Acrobot's
mixed-only acceptance `0<K<N` has a different conditioning scale, given
explicitly in Proposition 6. A narrower success-only option uses weights
`r'_i/K'`. It recovers the per-accepted maximum-likelihood direction exactly
if the resulting selected-positive marginal is
`m_theta(.|g,success)` after selection and is scored under `g`. That does not
recover `grad J_{N-1,g}` or a global target mixture, and the marginal-law
assumption is testable but not automatic.

The implementation may multiply a hindsight update by a scalar `lambda_HS`.
The law statements above describe the unscaled estimator; changing
`lambda_HS` deliberately changes its magnitude. The default is one, but it is
not theoretically optimal or universal.

## 7. Algorithm

```text
Algorithm 1: Curriculum-MaxRL

Inputs:
  tasks x_1,...,x_m; group size N >= 2; decay 0 <= d <= 1;
  floor 0 < f <= 1; concentration gamma >= 0;
  hindsight mode in {none, centered, success-only};
  hindsight scale lambda_HS

Initialize alpha_i = beta_i = 1 and visit_i = 0 for every task i

repeat:
  draw p_tilde_i ~ Beta(alpha_i,beta_i) for every task
  s_i <- [1-(1-p_tilde_i)^N-p_tilde_i]^gamma
  if every s_i is zero, replace s by the uniform score vector
  q_i <- normalized s_i mixed with uniform floor f
  sample the requested task batch from q

  for each requested task i in that batch:

    generate N rollouts z_1,...,z_N for task i
    verify rewards r_1,...,r_N and set K <- sum_j r_j

    update alpha_i,beta_i from the original requested rewards only
    visit_i <- visit_i + 1

    if 0 < K < N:
      assign w_j <- r_j/K - 1/N
      add the weighted requested-task trajectories to the policy update

    else if K = 0:
      record an all-fail group
      if hindsight is enabled:
        use an environment-specific verifier to select achieved goal g
        rewrite every trajectory for g and recompute rewards r'_j
        optionally truncate successful traces at first verified hit of g

        set K' <- sum_j r'_j
        if the relabel is semantically valid and K' > 0:
          if mode = centered:
            w'_j <- lambda_HS (r'_j/K' - 1/N)
            # K'=N is valid but gives an all-zero centered update
          if mode = success-only:
            w'_j <- lambda_HS r'_j/K'
          add the weighted relabeled trajectories to the policy update

    else:  # K=N only
      record an all-pass group and add no practical centered update

  apply each weighted group update, or accumulate groups for the host
  optimizer, without mutating parameters inside a group's gradient calculation
```

The teacher and sampler states, including random-number state, are part of a
training checkpoint. The production integration also assigns contiguous
post-filter dataset indices and validates reward/index alignment before
updating the teacher.

## 8. Experimental study

### 8.1 Scope and protocol

The experiments are mechanism studies. They use small exact-gradient, tabular,
tile-coded, or one-hidden-layer neural policies so that estimator conventions,
task transfer, and relabeling can be inspected directly.

| study | task/policy | budget and seeds | reported metric |
|---|---|---|---|
| exact estimator audit | Bernoulli-logit score, exact binomial sum | multiple `N,p`; no sampling noise | maximum identity error |
| skill chain | 3 chains x 12 nested tasks, exact tabular score gradients | 400 teacher steps, 8 groups/step, `N=16`, 12 paired seeds | mean of 41 equally spaced checkpoints, including step zero |
| grid reach | radius-8 goal-conditioned tabular policy; teacher `gamma=4` | 150 group steps, 4 groups/step, `N=16`, 10 seeds | normalized mean-pass AUC over group steps |
| MountainCar tile-coded mechanism study | official [`MountainCar-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car/) dynamics, shared tile-coded policy, 10 nested position thresholds | at least 500,000 transitions/condition, `N=16`, 10 paired seeds | normalized mean-pass AUC over actual transitions |
| MountainCar neural V1R2 development | same official dynamics, 8 nested thresholds; shared H64 plus hardest-only and exact total-/active-capacity disjoint controls | 500,000 nominal transitions, `N=16`, 3 paired development seeds × 5 cells | hardest-goal AUC primary; outcome-blind adequacy gate; NO-GO |
| Acrobot V1/V2 development | official [`Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/) dynamics, one-layer neural actor, 8 nested tip-height thresholds | V1: 3 pilot seeds; V2: 3 paired development seeds x 6 cells, nominal 2,000,000 transitions/cell, `N=16` | normalized target-uniform mean-pass AUC over actual transitions; launch gates, not confirmatory inference |
| Acrobot V3 verified confirmation | same neural shared-H64 actor; uniform versus frontier-`u_N` teacher at `gamma=1`; hindsight off | 20 sealed paired seeds x 2 cells, nominal 2,000,000 transitions/cell, `N=16` | one predeclared shared-policy AUC contrast; decision supported |
| Acrobot V5A/V5B | same shared H64 actor; 3 learning rates × 3 hindsight scales | V5A: 27 fresh development runs; V5B: 180 fresh confirmatory runs at selected `U*=250` | V5A feasibility passed; V5B completed but the frozen exact-reconstruction rule failed, so the primary family is a procedural NO-GO and no contrast is claimed |

Grid evaluation uses 32 fixed episodes per ring and preserves training RNG
state. The tile-coded MountainCar evaluation uses 64 fixed episodes per
threshold and also preserves training RNG state. The MountainCar predicates—reach position
`x >= x*`—are custom binary tasks. Mean-pass AUC and flag pass are not the
standard Gymnasium return.

Acrobot evaluation uses 32 fresh episodes per threshold, fixed per-seed common
random numbers, and a full training-state fingerprint before and after
evaluation. Its predicates—strict post-transition tip-height crossings—are also
custom binary tasks over official dynamics. V1 pilot and V2 development data
are explicitly excluded from confirmatory inference. All 40 registered V3 runs
completed, and the independent verifier reproduced the source/runtime lock,
run invariants, paired AUCs, interval, exact test, and decision before the V3
result entered the retained evidence.

The tile-coded MountainCar study reports a family of nine AUC contrasts. We report paired percentile
bootstrap confidence intervals, exact two-sided sign-flip tests, and
Holm-adjusted `p` values over that family. Values in result tables are mean
plus or minus sample standard deviation unless stated otherwise.

### 8.2 Exact estimator audit

Exact binomial enumeration verifies

- `E[H_N]=E[C_N]=grad J_N`;
- `E[D_N]=grad J_{N-1}`; and
- `E[A_N]=2u_N(p)`

to maximum numerical error `1.33e-15` over the tested grid. This is an algebra
regression, not empirical evidence about optimization quality.

### 8.3 Skill-chain mechanism study

The component ablation matches trainer steps, sampled groups, and rollout
attempts across 12 paired seeds. Its primary metric is the arithmetic mean of
the analytic mean pass rate at 41 equally spaced checkpoints including step
zero; it is not called AUC. Conditions that select deeper tasks execute more
primitive skill decisions, and hindsight can add one reused-data policy update
for an all-fail group, so primitive environment work and optimizer compute are
not matched.

| configuration | checkpoint mean | final mean pass (mean only) |
|---|---:|---:|
| uniform, no hindsight | 0.660 ± 0.031 | 0.967 |
| frontier teacher, `gamma=1`, no hindsight | 0.732 ± 0.015 | 0.980 |
| frontier teacher, `gamma=4`, no hindsight | 0.781 ± 0.008 | 0.983 |
| uniform + centered hindsight | 0.866 ± 0.002 | 0.979 |
| frontier teacher, `gamma=1` + centered hindsight | 0.879 ± 0.003 | 0.984 |
| **Curriculum-MaxRL, `gamma=4` + centered hindsight** | **0.886 ± 0.002** | **0.986** |
| Curriculum-MaxRL, `gamma=4` + success-only hindsight | 0.885 ± 0.003 | 0.986 |

The complete prespecified family of paired checkpoint-mean contrasts is:

| contrast | mean [95% paired bootstrap CI] | exact sign-flip `p` | Holm-adjusted `p` |
|---|---:|---:|---:|
| `gamma=1` teacher - uniform, no hindsight | +0.0718 [+0.0531, +0.0901] | 0.00049 | 0.00732 |
| `gamma=4` - `gamma=1`, no hindsight | +0.0494 [+0.0422, +0.0563] | 0.00049 | 0.00732 |
| centered hindsight - none, under uniform | +0.2056 [+0.1873, +0.2215] | 0.00049 | 0.00732 |
| centered hindsight - none, under `gamma=1` teacher | +0.1470 [+0.1380, +0.1543] | 0.00049 | 0.00732 |
| teacher x centered-hindsight interaction | -0.0586 [-0.0779, -0.0390] | 0.00098 | 0.00732 |
| `gamma=4` - `gamma=1`, with centered hindsight | +0.00735 [+0.00556, +0.00901] | 0.00049 | 0.00732 |
| centered hindsight - none, under `gamma=4` teacher | +0.1050 [+0.1012, +0.1087] | 0.00049 | 0.00732 |
| full stack - uniform + centered hindsight | +0.0205 [+0.0190, +0.0223] | 0.00049 | 0.00732 |
| concentration x centered-hindsight interaction | -0.0420 [-0.0482, -0.0353] | 0.00049 | 0.00732 |
| success-only - centered hindsight, under `gamma=4` | -0.00145 [-0.00252, -0.00057] | 0.00830 | 0.00830 |
| centered scale `0.25` - scale `1` | -0.0545 [-0.0561, -0.0529] | 0.00049 | 0.00732 |
| centered scale `0.5` - scale `1` | -0.0278 [-0.0301, -0.0253] | 0.00049 | 0.00732 |
| centered scale `2` - scale `1` | +0.0215 [+0.0204, +0.0225] | 0.00049 | 0.00732 |
| centered scale `4` - scale `1` | +0.0378 [+0.0361, +0.0393] | 0.00049 | 0.00732 |
| centered scale `8` - scale `1` | +0.0502 [+0.0490, +0.0515] | 0.00049 | 0.00732 |

All 15 contrasts reject their nulls at familywise alpha `0.05`. Statistical
resolution should not be confused with practical magnitude: the
success-only/centered difference, for example, is only `0.00145`.

The teacher, concentration, and centered-hindsight simple effects in the
factorial subset are positive, while both difference-in-differences are
negative. The correct interpretation is **complementarity with diminishing
returns**: frontier sampling and hindsight each help, and the combined method
exceeds either matched component, but their gains are subadditive rather than
synergistic. A plausible mechanism is that hindsight already moves many failed
groups to useful prefixes, leaving less avoidable rollout waste for the
teacher, while stronger teacher concentration leaves fewer all-fail groups for
hindsight to recycle. The factorial identifies the interaction, not that
causal explanation.

The success-only condition is `0.00145` below centered hindsight in checkpoint
mean (`95% CI [-0.00252, -0.00057]`, Holm-adjusted `p=0.00830`). The effect is
statistically resolved in this low-noise analytic testbed but practically tiny,
and it does not alter the law caveat; MountainCar also does not separate the
two estimators.

The hindsight-scale ablation is monotone over the tested range:

| centered hindsight scale | checkpoint mean |
|---:|---:|
| 0.25 | 0.832 ± 0.004 |
| 0.50 | 0.858 ± 0.004 |
| 1.00 | 0.886 ± 0.002 |
| 2.00 | 0.908 ± 0.002 |
| 4.00 | 0.924 ± 0.002 |
| 8.00 | **0.936 ± 0.001** |

Scale two exceeds scale one by `+0.0215 [0.0204, 0.0225]` with
Holm-adjusted `p=0.00732`; scale eight exceeds scale one by
`+0.0502 [0.0490, 0.0515]` with the same adjusted `p`. Thus the default scale
of one is not the best tested setting on this toy and should not be presented
as universal. Performance is monotone through the largest tested scale, eight,
so this sweep identifies neither an optimum nor the point at which aggressive
relabel weighting becomes unstable.

In a separate matched-generation regime study, a frontier-heavy pool with
maximum initial pass probability `10^-5` leaves uniform, DAPO-style redraw,
and the plain teacher at `0.000` mean-pass AUC, while teacher plus hindsight
reaches `0.860` AUC and `0.981` final mean pass. In this sequential simulator,
the corrected DAPO-style implementation is exactly equivalent to uniform at a
matched generated-group budget. This result illustrates cold-start signal
creation in a synthetic nested task family; it is not a claim about DAPO in
general.

### 8.4 Corrected goal-conditioned gridworld

| configuration | mean-pass AUC | final mean pass |
|---|---:|---:|
| uniform + practical MaxRL | 0.583 ± 0.022 | 0.857 ± 0.021 |
| frontier teacher, `gamma=4` + practical MaxRL | 0.652 ± 0.020 | 0.912 ± 0.024 |
| **Curriculum-MaxRL, `gamma=4` + centered hindsight** | **0.702 ± 0.032** | **0.916 ± 0.038** |

The descriptive ten-seed mean is higher for the teacher than for uniform and
higher again with hindsight. The relabeler chooses one concrete achieved goal,
rewrites goal conditioning for every trajectory, and stops credited successes
at first hit.
Full relabeled/fresh joint-law equality is not claimed. This is a tabular
mechanism replication, not a robotics benchmark.

### 8.5 Corrected tile-coded Gymnasium MountainCar mechanism study

| configuration | mean-pass AUC | final mean pass | final flag pass |
|---|---:|---:|---:|
| flag-only, shared policy | 0.024 ± 0.006 | 0.024 ± 0.006 | 0.000 ± 0.000 |
| uniform curriculum, shared | 0.389 ± 0.071 | 0.684 ± 0.094 | 0.058 ± 0.079 |
| exact mass, `gamma=1`, shared | 0.414 ± 0.081 | 0.758 ± 0.127 | 0.208 ± 0.266 |
| legacy `u_{N+1}`, `gamma=1`, shared | 0.414 ± 0.078 | 0.745 ± 0.121 | 0.175 ± 0.274 |
| learnability, `gamma=1`, shared | 0.411 ± 0.037 | 0.697 ± 0.088 | 0.080 ± 0.143 |
| exact mass, `gamma=4`, shared | 0.530 ± 0.059 | 0.928 ± 0.056 | 0.664 ± 0.232 |
| exact `gamma=4` + centered hindsight, shared | 0.720 ± 0.029 | 0.969 ± 0.013 | 0.842 ± 0.062 |
| **exact `gamma=4` + success-only hindsight, shared** | **0.727 ± 0.023** | **0.970 ± 0.014** | **0.848 ± 0.058** |
| exact `gamma=4` + centered hindsight, per-bin parameters | 0.229 ± 0.031 | 0.284 ± 0.028 | 0.000 ± 0.000 |

The paired AUC analysis is:

| contrast | mean delta [95% bootstrap CI] | exact sign-flip `p` | Holm-adjusted `p` |
|---|---:|---:|---:|
| exact `gamma=1` - uniform | +0.025 [-0.035, +0.091] | 0.488 | 1.000 |
| exact `gamma=4` - uniform | **+0.141 [+0.076, +0.202]** | 0.0078 | **0.0469** |
| exact `gamma=4` - exact `gamma=1` | **+0.116 [+0.060, +0.172]** | 0.0078 | **0.0469** |
| exact `gamma=1` - legacy `gamma=1` | -0.000 [-0.023, +0.026] | 0.988 | 1.000 |
| exact `gamma=1` - learnability `gamma=1` | +0.003 [-0.054, +0.063] | 0.928 | 1.000 |
| centered hindsight - no hindsight | **+0.191 [+0.155, +0.231]** | 0.0020 | **0.0176** |
| success-only hindsight - no hindsight | **+0.197 [+0.160, +0.238]** | 0.0020 | **0.0176** |
| centered - success-only hindsight | -0.006 [-0.021, +0.008] | 0.436 | 1.000 |
| shared - per-bin centered | **+0.492 [+0.464, +0.522]** | 0.0020 | **0.0176** |

The family-corrected results support five local claims: `gamma=4` exceeds
uniform and `gamma=1`; each hindsight estimator exceeds no hindsight; and the
shared implementation exceeds the per-bin control. They do **not** separate
proportional exact mass from uniform, the legacy score, or learnability, and
they do not separate centered from success-only hindsight.

The shared/per-bin comparison diagnoses a transfer channel in this
implementation. It is not a pure parameter-sharing intervention: the per-bin
control changes capacity and data flow as well. The exact coefficient-mass score is
mathematically tied to practical coefficient mass, but this experiment does
not show that its small fixed-`N` shape difference is empirically superior to
nearby scores.

### 8.6 Independent neural Acrobot: development gates and verified V3

#### Environment and causal design

The independent adapter uses the official Gymnasium `Acrobot-v1` transition
dynamics and 500-step time limit. For post-transition observation `o`, it
computes tip height

```text
h(o) = -o_0 - (o_0 o_2 - o_1 o_3)
```

and defines eight nested tasks by strict crossings of thresholds
`[-1.5,-1.0,-0.5,0.0,0.25,0.5,0.7,1.0]`. The last threshold matches native
termination. These thresholds are a custom binary evaluation family, not eight
standard Gymnasium tasks.

The policy is a categorical actor with one tanh hidden layer, trained by plain
SGD using the sum of per-transition score-function gradients evaluated at one
frozen pre-group parameter vector. This is the exact realization of the sampled
Monte Carlo estimator, not an exact population gradient or a PPO update. The
shared H64 actor has 640 total and 640
active parameters and receives no task identifier. V2 crossed uniform versus
frontier-`u_16`, `gamma=1` sampling with three architectures: shared H64; eight
disjoint H8 actors with 640 total but 80 task-active parameters; and eight
disjoint H64 actors with 5,120 total but 640 task-active parameters. This
two-control design was intended to distinguish total-capacity and active-
capacity explanations. Hindsight was off, so this experiment tests the
frontier teacher rather than the full optional method.

#### V1 pilot: retained failed gate

V1 screened learning rates `[1e-4,3e-4,1e-3,3e-3]` on excluded seeds
`10000..10002`. Its frozen pooled-improvement rule selected `3e-3`, but the
declared launch gate failed: the teacher runs ended at mean-pass values
`0.9805`, `0.9961`, and `0.9961`, violating the `<0.95` headroom condition,
and the selected-rate pool had no all-fail groups after 200,000 transitions.
No V1 confirmation was run. V2 then used an effect-blind rule requiring
within-arm learning with headroom at the first checkpoint at or above one
million transitions; it selected `3e-4`, without using the teacher-minus-
uniform contrast.

#### V2 six-cell development: five gates pass, one decisive gate fails

All 18 V2 runs completed with finite values and retained accounting. Their
descriptive endpoints were:

| development cell (`n=3`) | transition AUC | final mean pass | mean SGD updates |
|---|---:|---:|---:|
| uniform, shared H64 | 0.6415 | 0.8568 | 258.3 |
| frontier teacher (`u_16`, `gamma=1`), shared H64 | 0.6807 | 0.9115 | 272.3 |
| uniform, disjoint total-matched H8 | 0.3350 | 0.3346 | 203.3 |
| frontier teacher (`u_16`, `gamma=1`), disjoint total-matched H8 | 0.3421 | 0.3424 | 216.7 |
| uniform, disjoint active-matched H64 | 0.3697 | 0.4076 | 225.0 |
| frontier teacher (`u_16`, `gamma=1`), disjoint active-matched H64 | 0.3730 | 0.4076 | 228.3 |

The source/runtime/invariant, post-warmup mixed-exposure, teacher-movement,
all-three-`K`-regimes, and projected-runtime gates passed. The every-cell
learning/headroom gate failed. At the first complete-group checkpoint at or
above one million transitions, qualifying seed counts were `3/3` and `3/3`
for shared uniform and shared teacher, but `0/3`, `0/3`, `0/3`, and `2/3` for
the two disjoint pairs. Hence the controls were not behaviorally adequate for
the intended causal interaction test, and the prespecified V2 six-cell
confirmation was not authorized.

Only after this effect-blind gate decision was fixed, the exploratory contrasts
were inspected. The shared teacher-minus-uniform AUC was `+0.0392` with all
three paired differences positive (`[0.0142,0.0478,0.0556]`). The corresponding
two-sided exact sign-flip `p=0.25`, the minimum attainable nonzero two-sided
value with three nonzero same-sign pairs. The sharing interactions were
`+0.0322` against the total-matched control and `+0.0359` against the active-
matched control. These values motivate another test; they do not confirm
efficacy or transfer, must not be pooled with later seeds, and cannot repair
the failed launch gate.

#### V3: one narrow, independently verified confirmation

V3 removes the failed causal controls and asks only whether frontier sampling
driven by the derived `u_16` coefficient-mass score improves a fixed
task-agnostic shared H64 policy over uniform sampling on this Acrobot threshold
family. It froze learning rate `3e-4`, `N=16`, `gamma=1`, no hindsight, 20
paired seeds `12000..12019`, and a nominal two-million-transition budget per
arm. Its only primary estimand was

```text
Delta = mean_seed(AUC_teacher,shared - AUC_uniform,shared).
```

The preregistered claim required the observed `Delta >= +0.03` and one exact
two-sided paired sign-flip test with `p <= 0.05`. The inferential unit was the
seed pair, and the randomization interpretation assumes paired differences are
sign-exchangeable under the sharp null. The paired bootstrap interval is
descriptive and was not a separate decision gate.

| V3 registered field | independently verified value |
|---|---:|
| valid paired seeds | **20 of 20** |
| shared teacher-minus-uniform AUC `Delta` | **+0.0363524** |
| 95% paired bootstrap interval | **[+0.0164536, +0.0553949]** |
| exact two-sided sign-flip `p` | **0.00263977** |
| registered efficacy decision | **supported (`true`)** |

The observed mean cleared the preregistered `+0.03` threshold and the exact
test rejected at `0.05`, so the registered decision is supported. Because the
bootstrap interval's lower endpoint is `0.0164536`, this result should not be
restated as evidence that the population mean effect exceeds `0.03`.

The arm-level means are secondary descriptive outcomes:

| V3 arm | transition AUC mean | final mean pass |
|---|---:|---:|
| uniform, shared H64 | 0.648669 | 0.864258 |
| frontier-`u_16` teacher, shared H64 | 0.685021 | 0.916992 |

The source-locked independent verifier reproduced all 20 paired effects, the
primary mean, interval, exact `2^20` sign-flip enumeration, and decision. The
result supports positive shared-policy curriculum efficacy only on this fixed
eight-threshold family. V3 contains no disjoint-policy control and no hindsight
arm, so it makes no transfer, capacity, hindsight, or general-Acrobot claim.

#### V4A: feasibility gate failure stops V4B

V4A was a source/runtime-locked, effect-blind feasibility study for the
planned optimizer-matched hindsight factorial. All nine runs used hindsight
scale zero, so Stage A did not estimate a hindsight effect. The independent
verifier validated the immutable lock, artifact, protocol, accounting, and
saved runner decision, and selected the registered fallback `U*=250` optimizer
updates. All gates except gate 3 passed. Gate 3 required at least ten positive,
finite, one-to-one, nonmutating hindsight previews in every run; exactly three
of nine runs had only `8`, `5`, and `6` previews. The projected serial runtime
for the proposed 90-run factorial was `3.452702` hours and passed its gate.

The registered consequence is a stop: V4B was not authorized and was not run.
This is evidence that the frozen feasibility requirement was not met, not
evidence for or against hindsight efficacy. The top-level verification field
`all_checks_passed=true` means that artifact integrity and the independent
recomputation passed; it does not override `gates.all_pass=false` or
`stage_b_factorial_authorized=false`.

The SHA-256 hashes are
`b19488783e1adba8cbac44ce8256c725a4470d8108c1192f9491ecc4882f1d8c`
for the lock,
`69b827dc425014f3b568186981e9c24d95158c72653125e0ade181272def2891`
for the Stage-A artifact, and
`c633e09df8e056f1589e631ff4d311913e1ac5594c3647790acc4b05990fca88`
for the independent report. The direct file-path analyzer command frozen in
the lock has a module-import defect. No locked V4 file was changed; from the
repository root the same hash-locked analyzer was run successfully as:

```bash
/tmp/curriculum-maxrl-gym/bin/python -m frontier_rl.examples.analyze_acrobot_hindsight_v4 \
  frontier_rl/examples/acrobot_hindsight_v4a_feasibility.json \
  --lock frontier_rl/examples/ACROBOT_HINDSIGHT_V4A_LOCK.json \
  --output frontier_rl/examples/acrobot_hindsight_v4a_verification.json
```

The invocation correction and its scope are retained in
`frontier_rl/examples/ACROBOT_HINDSIGHT_V4_ERRATA.md`.

#### V5A passes fresh feasibility; completed V5B is a procedural NO-GO

V5 did not revise V4A's stopped rule. It introduced fresh development seeds
`15000..15002` and ran the complete 3×3 grid of learning-rate multipliers
`{0.5,1,2}` and hindsight scales `{0,1,2}`. All 27 runs completed. The
independent analyzer reproduced the source/runtime lock, fixtures, schedule,
first-exact-update prefixes, natural relabel application, dead/mixed/all-pass
coverage, teacher activity, and runtime projection. Every launch gate passed,
and the fresh budget rule selected `U*=250`. The projected serial runtime for
the 180-run confirmatory matrix was `7.0557400375` hours.

The V5A authorization rule read no pass-rate curve, return, entropy, AUC, final
performance, or scale/learning-rate performance contrast. It therefore
establishes mechanics and feasibility, not efficacy. Its lock, artifact, and
verification SHA-256 hashes are respectively
`5c277413c5238f5839d281e09810537221a16737f831a498a3e0217ca5b1502e`,
`9cf741c91dcb82218cada9b451b76e0811c67aa4cbf1786ac0ba926806479b0a`,
and `a46b5e9f732b7f9e1796e2d4a2ff344c9ff738574c464b28631e884faaa6ba19`.

V5B completed all nine cells on fresh paired seeds `16000..16019`: all 180
runs finished with zero run failures, and a post-hoc forensic raw-integrity
audit validated 53,510 group records, 45,000 updates, and 1,080 checkpoints. Its
four update-indexed AUC contrasts, exact `2^20` sign-flip tests, Holm family,
and `0.03` materiality rules were frozen. The amendment and lock hashes are
`11975381874842bc3019074ea9d8168006c0517982ac11e00ad0b488e7671f36`
and `dfc930bbaf8e51c96fd1dab5851179457fce4f151def8c138ddf0cf17402bcf2`;
the completed artifact hash is
`c633886a121906ee2bceb03f3117e4bea5dc20ab314e43f9b702ef8d88f495ac`.

The frozen analyzer nevertheless failed deterministically before authorizing
the primary family. The runner computed step-norm diagnostics with NumPy
reductions, whereas the analyzer reconstructed them with Python scalar
reductions and then required exact equality of the resulting dictionaries.
A post-hoc forensic reduction audit found 377 mismatches among 720 diagnostic
floats; the maximum absolute discrepancy was `1.9984014443252818e-15` and the
maximum distance was 11 ULP. The protocol classifies step norms as diagnostics, but it
also makes exact runner/analyzer agreement an all-or-nothing acceptance rule.
That rule controls: the official V5B primary family is a **procedural NO-GO**,
and no cell outcome, contrast, sign, or hindsight-effect result is claimed.
The sealed runner artifact nevertheless contains precomputed case, contrast,
and decision subtrees, violating the protocol's literal "not computed" rule
even though none is reported or interpreted here. Those fields are
quarantined; V5C must separate raw execution output from verifier-gated primary
analysis.

A post-hoc tolerance-aware compatibility audit passed the remaining integrity
and reconstruction checks. It is diagnostic only and cannot rescue or
authorize the frozen family. A valid follow-up requires a reviewed
tolerance-aware verifier and a fresh V5C seed block; the existing V5B outcomes
must not be recycled into that decision. The failure boundary and post-hoc
diagnostics are recorded in the
[V5B verification erratum](frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md)
and
[forensic verification report](frontier_rl/examples/acrobot_hindsight_v5b_forensic_verification.json).

### 8.7 Neural MountainCar V1R2: verified development NO-GO

Neural V1R2 was designed as an independent transfer test, not a replication of
Section 8.5's tile-coded mean-pass mechanism study. It used eight nested
position thresholds and five conditions: frontier/shared H64,
uniform/shared H64, hardest-only/shared H64, uniform/disjoint-total H8×8, and
uniform/disjoint-active H64×8. The shared H64 and disjoint-total models each
have 384 total parameters; shared H64 and each active disjoint-active slot each
have 384 active parameters. All uniform conditions consumed an identical
outcome-independent task-schedule prefix.

All 15 development runs on seeds `17000..17002` completed, and the independent
analyzer reconstructed sampler traces, uniform schedules, raw rollouts,
evaluations, common random numbers, transition AUCs, parameter identities, and
the no-hindsight contract. The technical reconstruction passed, but the
predeclared feasibility decision was **NO-GO**:

| V1R2 development diagnostic | value |
|---|---:|
| pooled all-fail groups | 1,932 |
| pooled mixed groups | 474 |
| pooled all-pass groups | 0 |
| runs with nonzero hardest-goal AUC | 0 of 15 |

All four primary hardest-goal AUC contrasts were therefore exactly zero. The
supporting mean-pass AUC differences were `+0.0065104` (frontier minus
uniform), `+0.0119792` (frontier minus hardest-only), `+0.00546875` (uniform
shared minus disjoint-total), and `+0.00429688` (uniform shared minus
disjoint-active). These are descriptive development quantities and cannot
rescue a degenerate primary outcome.

The correct conclusion is not that the curriculum won or lost. The frozen
actor, optimizer, and 500,000-transition budget produced no native-goal
headroom and never exercised the all-pass side that coefficient-mass sampling
should downweight. Reserved confirmatory seeds `18000..18019` remain untouched
and unauthorized. This lack of native-goal headroom is directionally
consistent with Section 8.5's flag-only shared and centered-hindsight per-bin
controls, both of which ended at zero final-flag pass, while the successful
Section 8.5 arms combined intermediate-goal training with shared parameters.
Because the actor, optimizer, budget, and endpoints differ, that consistency
is neither a replication nor causal proof of the transfer channel.

The lock, development artifact, and verification hashes are
`b5edbc33048a8d3a8d7dbb992a23178ddf8424dd3c5be3165c87e6dc42a50a5c`,
`2e4803805009a3323307f6bdcfae17fb625008adb5361dc2310e414a19129180`,
and `fdefc9e4ee2887953c341d2f44c44001bc336598b089dbdc8035175e430148a0`.

### 8.8 Historical GPU maze records are not confirmatory evidence

Earlier 1.26M-parameter transformer maze runs are retained for provenance, but
the July 2026 audit found that they used the legacy `u_{N+1}` teacher score,
mixed `K=0` and `K=N` in a zero-weight counter, used the deepest response
budget for every level, scaled dense-hindsight loss with relabel count, and
reported unanchored step-indexed AUC despite wall-clock-matched endpoints.
Some historical relabel diagnostics also used path length where minimum BFS
depth was required. The seed-0 records were:

| historical configuration | final | best | legacy unanchored step-AUC |
|---|---:|---:|---:|
| frontier-ALP + MaxRL + dense hindsight | 0.258 | 0.269 | 0.236 |
| frontier-ALP + MaxRL | 0.244 | 0.257 | 0.233 |
| frontier + MaxRL + hindsight | 0.230 | 0.256 | 0.234 |
| uniform + MaxRL | 0.225 | 0.233 | 0.214 |
| uniform + GRPO | 0.230 | 0.237 | 0.216 |

These numbers motivate corrected experiments; they do not validate the
Curriculum-MaxRL coefficient-mass score, hindsight estimator, throughput mechanism, or an
objective-by-curriculum interaction. No reliability or state-of-the-art claim
is based on them.

## 9. Ablation findings

The experiments isolate several distinct questions.

1. **Does the `u_N` coefficient-mass score help under proportional sampling?**
   Not detectably in MountainCar: `u_N`, legacy `u_{N+1}`, learnability, and
   uniform are not separated at `gamma=1` after family correction.
2. **Does concentration matter?** Yes locally: coefficient-mass `gamma=4`
   exceeds coefficient-mass
   `gamma=1` by `+0.116 [0.060, 0.172]` AUC in MountainCar, and the skill chain
   moves from `0.732` to `0.781` checkpoint mean without hindsight. This is an
   empirical shared-policy concentration effect, not an isolated transfer
   effect or a consequence of the coefficient-mass
   theorem.
3. **Does hindsight matter after task prioritization?** Both centered and
   success-only variants exceed no hindsight in MountainCar. The experiment
   does not separate the two variants. In the lower-noise skill chain, centered
   is slightly above success-only, but the difference is only `0.00145`.
4. **Does the shared implementation outperform the per-bin control?** Yes by
   `+0.492 [0.464, 0.522]` AUC, while the per-bin flag-pass mean remains zero.
   Capacity and data flow are confounded, so this diagnoses a plausible
   transfer channel rather than identifying parameter sharing causally or
   establishing a universal architecture result.
5. **Can task selection alone solve a beyond-frontier cold start?** Not in the
   synthetic frontier-heavy regime. Hindsight supplies auxiliary prefix
   targets when every available source task is effectively unreachable.
6. **Should relabeled outcomes update the teacher?** No by default. A local
   feedback ablation inflated pseudo-count competence without establishing
   natural requested-task success. Curriculum-MaxRL updates teacher evidence
   from requested outcomes only.
7. **Are the teacher and hindsight synergistic?** No on the paired skill-chain
   factorial. Both components help and the full stack exceeds either matched
   component, but the teacher-by-hindsight and concentration-by-hindsight
   interactions are negative. The evidence supports complementary,
   diminishing returns rather than superadditive synergy.
8. **Is hindsight scale one optimal?** No on the tested skill chain. Scales
   `2`, `4`, and `8` exceed scale one, while scales `0.25` and `0.5` are worse;
   scale eight has the highest checkpoint mean at the tested boundary. Because
   the curve has not turned over, the sweep identifies neither an optimum nor
   a transferable default.
9. **Does the neural Acrobot study establish shared transfer?** No. V2's
   disjoint controls failed the predeclared behavioral-learning gate, so their
   small curriculum interactions are not a valid causal transfer test. V3
   independently confirms the narrower shared-policy efficacy claim
   (`Delta=+0.0363524`, exact `p=0.00263977`), but its shared-only design cannot
   identify transfer as the cause.
10. **Does capacity-matched neural MountainCar establish transfer?** No. V1R2
    had exact total- and active-capacity controls, but every hardest-goal AUC was
    zero and no all-pass groups occurred. The development gate correctly
    stopped confirmation; small positive supporting mean-pass deltas are not a
    substitute for primary-metric headroom.

## 10. Limitations

1. **Coefficient mass is a proxy.** The theorem concerns scalar `L1`
   coefficients, not realized gradient norm, variance, or improvement.
2. **The main evidence is local.** The corrected studies use exact-gradient,
   tabular, weak tile-coded, and a small one-hidden-layer neural policy. The
   Acrobot V1/V2 data remain excluded development evidence, while V3 supplies
   one independently verified neural shared-policy result on a fixed nested
   threshold family. The project does not establish broad neural-control,
   language-model, or real-robot performance.
3. **Both MountainCar metrics are custom.** They use nested binary thresholds
   on official dynamics, not standard episodic return. The positive tile-coded
   study's per-bin control is not capacity matched. Neural V1R2 adds exact
   capacity controls but is only a stopped three-seed development study with a
   zero hardest-goal primary metric.
4. **Hindsight remains distribution shifted.** Verifier validity, conditioning
   rewrite, and first-hit stopping remove semantic and protocol errors but do
   not imply `Q=P` or moment equality.
5. **Adaptive sampling changes the training objective.** Without importance
   correction, the method does not provide an unbiased gradient of a fixed
   uniform task mixture.
6. **The teacher assumes a finite indexed pool.** Streaming or procedural task
   sources require a parametric density or another generalizing difficulty
   model.
7. **Hyperparameters remain.** The score removes a hand-chosen success band,
   but decay, prior, floor, concentration, relabel rule, and hindsight scale
   remain design choices.
8. **Statistical power is limited.** MountainCar has ten paired seeds and a
   corrected comparison family, but some no-hindsight flag estimates remain
   wide. Grid is a descriptive local replication. The 12-seed skill-chain
   ablation has a paired corrected family but unusually low-noise analytic
   evaluation. Acrobot V3 has 20 paired seeds and supports its one registered
   efficacy decision, but it does not power unregistered transfer or hindsight
   claims. V5A is a feasibility stage; V5B completed but its frozen exact
   reconstruction rule failed, so its primary family is a procedural NO-GO;
   and neural MountainCar V1R2 did not authorize its 20-seed block.
9. **Corrected large-scale tests are missing.** The production integration is
   unit tested and passed a local patch-application check, but that check is not
   retained with an upstream MaxRL commit hash and there is no corrected
   multi-worker, multi-GPU end-to-end run.
10. **Historical GPU evidence is confounded.** It is retained only to explain
    which experiments must be rerun.
11. **Matched rollout attempts are not matched total compute.** In the skill
    chain, deeper sampled tasks execute more primitive decisions, and
    hindsight may add a reused-data optimizer update. Wall-clock and primitive
    compute comparisons require separate controls.
12. **The Acrobot control failure leaves the mechanism unidentified.** Matching
    total or active parameter counts did not make the disjoint policies learn
    comparably at the frozen budget and rate. V3 resolves shared-policy
    efficacy for its fixed family, but cross-task transfer causality remains
    unresolved.
13. **Local locks are not external preregistration.** They provide a
    machine-checkable within-repository chronology but no independent
    timestamp. Current V3 and later manifests match their files. V2's lock
    records a historical runner hash that differs from HEAD, so exact V2 runner
    reconstruction requires bytes not present in the current tree.

## 11. Next experiments

The next work should maximize information rather than expand the headline.

1. **Design and preregister V5C with a tolerance-aware verifier:** V5B finished
   180/180 runs with intact raw records, but its frozen analyzer failed exact
   equality on negligible cross-implementation norm-reduction differences.
   Preserve V5B as a procedural NO-GO. Review and test a finite/absolute/ULP
   tolerance contract before sealing fresh seeds; do not tune V5C from V5B
   cell outcomes or report V5B's frozen primary family post hoc.
2. **Repair the transfer intervention rather than reinterpret V2:** make the
   disjoint controls learn under an effect-blind development rule, for example
   through architecture-specific optimizer calibration or a longer frozen
   budget, then preregister a new teacher-by-sharing factorial on untouched
   seeds. Calibration must not use the interaction sign.
3. **Redesign neural MountainCar adequacy:** keep seeds `18000..18019`
   untouched. On fresh development seeds, calibrate actor/optimizer/budget with
   an outcome-blind rule requiring native-goal variation and all-fail, mixed,
   and all-pass exposure. Retain hardest-goal AUC and both exact capacity
   controls; only a separately reviewed V2 protocol may authorize confirmation.
4. **Score-shape factorial on one neural benchmark:** uniform, the `u_N`
   coefficient-mass score, legacy `u_{N+1}`, and learnability at common
   `gamma=1`, floor,
   initialization, response budget, and transition count.
5. **Concentration ablation after a base teacher effect is established:** the
   frontier-`u_N` rule at `gamma=1` versus `gamma=4`, with transfer structure
   measured rather than assumed.
6. **Hindsight factorial:** none, centered, and success-only, using verifier
   checks, complete conditioning rewrite, first-hit or BFS-correct stopping,
   cumulative separate `K=0`/`K=N` counters, and a wider preregistered scale
   sweep rather than assuming scale one.
7. **Objective-by-curriculum interaction:** MaxRL versus GRPO crossed with
   uniform versus frontier sampling under matched transitions. This is needed
   before making any compatibility or safety claim.
8. **Corrected GPU causal factorial:** materialize one identical post-
   initialization checkpoint per seed, preserve training RNG across evaluation,
   and cross the selected teacher with hindsight off/on under anchored compute
   accounting. This remains necessary before any scale claim.
9. **LLM-scale 2x2:** curriculum by `{MaxRL, GRPO}` on a fixed verifiable prompt
   pool when suitable multi-GPU hardware is available.
10. **Hindsight-law diagnostics:** estimate update-scale ratios, score moments,
   and relabeled/fresh distinguishability instead of relying on cosine alone.
11. **Streaming-task extension:** replace per-task pseudo-counts with a
   difficulty model over a continuous or generative task space.
12. **Adaptive group sizes:** test the fixed-pass-rate greedy allocation only
   after the rollout worker supports bounded per-task group sizes, and compare
   against variance-aware and long-horizon allocation baselines.

## 12. Conclusion

Curriculum-MaxRL begins with a correction, not a performance claim. The
practical dropped-group estimator used here targets order `N-1`, and its exact
expected scalar coefficient mass is `2[pass@N-pass@1]`. That identity yields a
finite-rollout frontier score with a closed-form peak and a clear limitation:
it predicts where nonzero scalar coefficients occur, not how much a policy
will improve.

Local experiments support three mechanisms. Concentrated frontier sampling
can improve a shared policy, verified hindsight can add signal beyond task
selection, and the corrected MountainCar shared implementation exceeds a
confounded per-bin control. That comparison diagnoses a plausible transfer
channel but does not isolate it causally. The independent neural Acrobot
development matrix sharpens this limitation: shared actors learned and showed
an excluded `+0.0392` teacher-minus-uniform AUC, but the disjoint controls did
not satisfy the behavioral gate, so the causal matrix stopped. The subsequent
source-locked V3 shared-only confirmation independently verified a
`+0.0363524` teacher-minus-uniform AUC over 20 paired seeds (95% paired
bootstrap CI `[0.0164536, 0.0553949]`, exact `p=0.00263977`) and supported its
registered efficacy decision. This is positive neural evidence for a shared
policy on the fixed threshold family, not evidence for transfer, capacity
effects, hindsight, or Acrobot in general. On the paired skill chain, the
components are complementary
but subadditive: hindsight reduces the remaining teacher gain, and stronger
teacher concentration reduces the remaining hindsight gain. The same
experiments also reject a stronger story: the exact
score is not empirically separated from nearby proportional priorities at
`gamma=1`, hindsight is not generally unbiased, and historical GPU runs do
not establish scale. The defensible contribution is therefore an audited
mathematical bridge from a practical estimator to a curriculum, plus a
reproducible local validation program for testing where that bridge holds.
V4A adds a transparent feasibility stop: integrity verification passed, but
the preview-count gate failed, so no V4B hindsight comparison was run and no
hindsight-effect conclusion follows from V4A. Fresh V5A then passed its stronger
full-grid gates and authorized V5B. V5B then completed 180/180 runs with intact
raw records, but its frozen analyzer failed exact diagnostic equality because
NumPy and Python scalar norm reductions differed by at most
`1.9984014443252818e-15` (11 ULP). The all-or-nothing rule makes this a
procedural NO-GO, so V5B adds no primary contrast or hindsight-effect result.
Neural MountainCar V1R2 adds a second transparent stop: complete
reconstruction passed, but the native-goal primary metric and all-pass regime
were absent, so confirmation remained untouched. Together these gates make the
current evidence more credible by preserving negative feasibility outcomes
instead of converting them into post-hoc performance claims.

## Reproducibility pointers

- Formal proofs: `curriculum_maxrl/PROOFS.md`
- Consolidated validation: `curriculum_maxrl/VALIDATION.md`
- Skill-chain driver: `frontier_rl/examples/run_skill_chain.py`
- Paired skill-chain ablation: `frontier_rl/examples/run_skill_chain_ablation.py`
- Skill-chain artifact: `frontier_rl/examples/skill_chain_component_ablation.json`
- Grid artifact: `frontier_rl/examples/grid_reach_validation.json`
- MountainCar artifact: `frontier_rl/examples/mountaincar_shared_validation.json`
- Acrobot V1 protocol and pilot: `frontier_rl/examples/ACROBOT_NEURAL_PROTOCOL.md`,
  `frontier_rl/examples/acrobot_neural_pilot.json`
- Acrobot V2 protocol, development artifact, and gate record:
  `frontier_rl/examples/ACROBOT_NEURAL_PROTOCOL_V2.md`,
  `frontier_rl/examples/acrobot_neural_v2_capacity_development.json`,
  `frontier_rl/examples/acrobot_neural_v2_development_gates.json`
- Verified Acrobot V3 protocol, source lock, artifact, and independent report:
  `frontier_rl/examples/ACROBOT_NEURAL_PROTOCOL_V3.md`,
  `frontier_rl/examples/ACROBOT_NEURAL_V3_LOCK.json`,
  `frontier_rl/examples/acrobot_neural_v3_shared_confirmatory.json`,
  `frontier_rl/examples/acrobot_neural_v3_verification.json`
- Completed V4A protocol, source lock, feasibility artifact, independent
  report, and invocation errata:
  `frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V4.md`,
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V4A_LOCK.json`,
  `frontier_rl/examples/acrobot_hindsight_v4a_feasibility.json`,
  `frontier_rl/examples/acrobot_hindsight_v4a_verification.json`,
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V4_ERRATA.md`
- Acrobot V5 protocol, V5A lock/artifact/verification, and V5B
  amendment/lock/completed artifact:
  `frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V5.md`,
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V5A_LOCK.json`,
  `frontier_rl/examples/acrobot_hindsight_v5a_feasibility.json`,
  `frontier_rl/examples/acrobot_hindsight_v5a_verification.json`,
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_AMENDMENT.json`,
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_LOCK.json`,
  `frontier_rl/examples/acrobot_hindsight_v5b_factorial.json`,
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md`,
  `frontier_rl/examples/acrobot_hindsight_v5b_forensic_verification.json`
- Neural MountainCar V1R2 result note, lock, development artifact, and
  verification:
  `frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_V1_RESULTS.md`,
  `frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_V1_LOCK.json`,
  `frontier_rl/examples/mountaincar_neural_transfer_v1_development.json`,
  `frontier_rl/examples/mountaincar_neural_transfer_v1_development_verification.json`
- External-review entry point: `REVIEW_NOTES.md`
- Production integration: `verl_integration/`
