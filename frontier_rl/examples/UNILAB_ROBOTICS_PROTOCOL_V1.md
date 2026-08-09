# UniLab robotics protocol V1: from reset-stream curriculum to grouped MaxRL

Status: **development protocol; locally recorded before claim-bearing runs**.
This is not an externally timestamped preregistration.

Execution target:

- Curriculum-MaxRL branch: `codex/curriculum-maxrl-research`
- UniLab source: fresh worktree `../UniLab-curriculum-maxrl`
- UniLab base commit: `515111162634dd14c67882f9c664d9a780c6f0c5`
- Hardware: Apple M4, 16 GB, CPU-only learner for the first smoke and
  mechanism gates

## 1. Claim boundary

There are two distinct experiments in this protocol.

1. **Reset-stream curriculum component.** UniLab's ordinary dense-reward PPO
   remains the learner. Completed episodes provide binary observations to a
   task-bin teacher, which assigns the next terrain bin. This can validate the
   scheduler, tracking, lifecycle, and sampling behavior. It does **not** test
   the grouped practical-MaxRL estimator `D_N`.
2. **Grouped estimator experiment.** A later learner collects `N` complete,
   independent episodes for the same task under one frozen policy, computes a
   binary verifier outcome for each, and applies the practical coefficients to
   each complete trajectory score. Only this stage instantiates the central
   estimator/teacher coupling.

Every report must identify which stage produced a number. A reset-stream PPO
result may not be described as “MaxRL training.”

## 2. Exact objects under test

For task `x`, success probability `p`, `N` complete rollouts, binary outcomes
`r_i`, count `K=Σr_i`, and true trajectory scores
`S_i=∇log π_θ(τ_i|x)`, the practical update is

```text
D_N = 1{K>0} Σ_i (r_i/K - 1/N) S_i,
E[D_N] = ∇J_{N-1},
J_T(p) = -Σ_{k=1}^T (1-p)^k/k.
```

Its exact expected scalar coefficient mass is

```text
E[Σ_i |w_i|] = 2u_N(p),
u_N(p) = 1-(1-p)^N-p = pass@N-pass@1,
p*_N = 1-N^(-1/(N-1)).
```

The scientific question is not whether this identity is true; it is. The
question is whether `u_N` predicts **useful target-distribution improvement per
unit environment cost** better than learnability, advantage magnitude, or
learning progress after score norms, cancellation, task-gradient conflict,
and nonstationarity enter.

## 3. Stage 0 — infrastructure gate (non-claim-bearing)

Before adaptive training:

- UniLab must emit exactly one episode-end event after both early termination
  and time-limit truncation are finalized and before autoreset.
- Normal timeout successes and early failures must both reach the teacher.
- The completed episode's old bin and final state must be recorded before a
  new bin is assigned.
- Teacher state must restore evidence, assigned bins, cached probabilities,
  and NumPy RNG so future samples are identical after resume.
- Existing default terrain behavior must remain unchanged.
- CPU-only flat and rough PPO one-update smokes must complete and write a
  checkpoint.

Passing Stage 0 establishes integration integrity only.

## 4. Stage 1 — fixed-policy base-rate and axis adequacy

The first exact-learner candidate is now UniLab's `StewartBalance` manipulation
task with the Motrix CPU backend. It is a cleaner first target than locomotion:
the observation is 15-dimensional, the action is two-dimensional, short
episodes are complete in ten control steps, and the task already distinguishes
successful stillness from a fall. Define tasks by the frozen
`init_ball_radius_ratio`; each ratio induces a nested uniform-disk reset
distribution. Overlap is allowed mathematically, but no outcome is cross-fed
between task trackers.

A development-only zero-action feasibility probe is recorded in
`unilab_stewart_base_rate_seed0_4.json`. Five RNG replicates, 64 CPU
environments, 100 simulator steps, and a 0.2-second horizon produced these
pooled task pass rates:

```text
radius ratio      .15    .20    .30    .40    .50    .60    .70    .80
pooled pass rate 1.000  0.571  0.255  0.150  0.103  0.069  0.050  0.034
```

This is a fixed-policy axis/lifecycle diagnostic, not a training result and not
evidence that the teacher helps. The apparent frontier is actually a failed
control-adequacy gate. Resets are uniform in disk area with maximum radius
`0.8*ratio`, while the native stillness radius is `0.12`; therefore the passive
probability of starting inside the goal is

```text
min(1, (0.12 / (0.8*ratio))^2).
```

For ratios `.20` through `.70`, this predicts `.5625, .2500, .1406, .0900,
.0625, .0459`, essentially the observed ladder. Mixed outcomes were mostly
reset geometry, not policy-controllable learnability. The stock disk/0.2-second
axis is rejected for claim-bearing comparisons.

The repaired development task uses fixed physical reset radii
`[.175,.275,.375,.475]*platform_radius`, random angle, a 6-second horizon, a
0.5-second eligibility delay, and ten *post-delay* stillness steps. UniLab's
`StewartBalanceGrouped` owner exposes the frozen radius as a 16th observation
and records the native success boolean before autoreset. The independent exact
runner retains 15 observations so it can load an audited stock-PPO warm start;
the initial ball displacement makes the radius identifiable at reset.

A post-hoc development sweep over one 3.28M-transition stock-PPO run selected
iteration 200 as the first viable warm start. Under 128 fixed stochastic
evaluation episodes per task it scored
`[.211,.133,.133,.039]` (macro `.129`), versus iteration 0's
`[.148,0,0,0]` (macro `.037`). Selection is development-only; a confirmatory
protocol must freeze the warm-start rule without inspecting confirmation
outcomes.

The Go2 rough-terrain axis remains the reset-stream curriculum-component test
in Stage 2. Its claim-bearing verifier must eventually be command aligned:

```text
success(e) = 1{episode reaches T_max}
             · 1{mean_t 1[||v_xy(t)-c_xy(t)|| <= eps_v
                            and |omega_z(t)-c_omega(t)| <= eps_w] >= q}.
```

`eps_v`, `eps_w`, and `q` may be calibrated only on development seeds, then
frozen. The initial Go2 implementation's timeout-without-early-termination
signal is a lifecycle smoke predicate, not the final performance verifier.

Launch an adaptive comparison only if every completion creates exactly one
event, at least three tasks have pass rates in `[0.05, 0.80]`, the hardest task
is above `0.02`, and fixed-policy evaluation leaves training RNG/state
unchanged. Failure means change the warm start, verifier, reset distribution,
or task axis using development data only. It is not evidence for or against the
teacher.

## 5. Stage 2 — dense-PPO curriculum-component comparison

This stage tests data scheduling under the same ordinary PPO learner. Match
environment transitions, optimizer updates, evaluation calls, and initial
policy. Report wall-clock separately.

Arms:

1. uniform terrain rows;
2. UniLab's distance promote/demote heuristic;
3. frontier teacher with `p(1-p)` utility;
4. estimator-shaped sampling with `u_8(p)`.

Arm 4 is an ablation of teacher shape. Because PPO supplies no frozen groups
or `D_8` trajectory weights, its use of `N=8` is nominal and does not inherit
the grouped-MaxRL optimality or gradient interpretation.

Shared teacher mechanics: discounted success/failure tracking, explicit
uniform floor `0.1`, `gamma=1`, fixed terrain-type columns, and identical
randomization. `gamma=4` is a separately registered concentration ablation,
not a default.

Budgets:

- seed-0 mechanism pilot: 0.5–1.0M transitions per arm, 30-minute wall cap;
- development: three paired seeds, 2M transitions per arm;
- confirmation (only after a frozen development decision): at least five new
  paired seeds.

Primary metric: macro mean command-aligned success AUC over actual environment
transitions under a fixed uniform evaluation mixture. Also report per-row/type
success, worst-row success, easy-row retention, fall rate, tracking error,
sampling probabilities, estimated pass rates, calibration error, and
wall-clock.

## 6. Stage 3 — exact grouped estimator/teacher experiment

Preferred first axis: the mixed-success Stewart radius-ratio tasks from Stage
1, now the repaired fixed-radius set `.175,.275,.375,.475`. The separate Torch
actor-only runner is `unilab_stewart_grouped.py`; PPO is not relabeled as exact.
It scores the detached pre-clip Gaussian latent action under the still-frozen
collection policy, sums the complete on-policy trajectory score, applies one
grouped update, and discards the wave. A deterministic regression check fails
if the actor-mean score gradient is accidentally cancelled by the
reparameterized sampling path. Dense reward never enters this actor update.

The first three paired development seeds used `N=8`, 120 groups, SGD `1e-5`,
128 fixed evaluation episodes per task, and 288,000 backend environment
transitions per arm. All arms learned substantially from macro `.129`; means
and sample standard deviations across the three seeds were:

```text
arm             transition AUC        final macro pass
uniform           .225 ± .031           .354 ± .070
learnability      .243 ± .020           .385 ± .007
u_8 mass          .232 ± .004           .376 ± .051
```

The estimator-specific teacher did validate its immediate mechanism: `u_8`
averaged 107 mixed/update-bearing groups, 13 all-fail groups, and 151.25
realized coefficient-mass units, versus uniform's 98.7, 21.3, and 140.25.
However, learnability had the highest mean AUC, teacher ordering changed by
seed, every descriptive paired interval crossed zero, and exact sign-flip
tests with only three pairs are uninformative. This is the intended scientific
boundary: scalar coefficient mass predicts whether the estimator emits an
update better than it predicts useful target-mixture progress. No teacher
superiority is claimed. Any confirmation must use fresh seeds and a frozen
checkpoint-selection rule.

Commanded speed/yaw bins on `Go2JoystickFlat` are the next transfer-rich axis.
Commands are already policy inputs, so one shared actor can transfer across
bins without changing network topology. Freeze one command for each complete
episode, and do not begin until the command-aligned verifier passes Stage 1
adequacy gates.

Collection/update contract:

1. Freeze policy parameters for the collection wave.
2. Sample a task bin and collect exactly `N=8` independent complete episodes
   under that bin.
3. Compute only the frozen binary verifier outcomes for teacher/actor weights;
   retain dense return for evaluation, not for the exact actor estimator.
4. Store each episode's complete `Σ_t log π(a_t|s_t,x)` graph or equivalent
   sufficient statistics.
5. Apply `w_i=1{K>0}(r_i/K-1/N)` to each complete trajectory score in one
   on-policy actor update. No PPO clipping or rollout reuse in the first exact
   implementation.
6. Update requested-task teacher evidence only; do not infer success from
   dense reward and do not feed auxiliary/relabel outcomes into it.

Minimum teacher-shape arms under the same `D_8` learner:

- uniform;
- learnability `p(1-p)`;
- exact coefficient mass `u_8(p)`;
- SEC-style mean absolute advantage or a strongest available learning-progress
  baseline.

Minimum objective interaction:

- practical `D_8`;
- a registered binary PPO/RLOO-style baseline;
- positive-part weights, with pass@1 and pass@8 both reported.

Instrumentation at every checkpoint:

- held-out `p_i`, tracker `p_hat_i`, predicted `u_N(p_hat_i)`;
- realized coefficient mass;
- actor gradient norm, squared norm, and signal-to-noise estimate;
- cosine with the fixed uniform-mixture evaluation gradient;
- cross-bin gradient cosine/transfer matrix;
- `K=0`, mixed, and `K=N` group rates;
- environment transitions, complete episodes, actor updates, and wall-clock.

Add a target-preserving arm using `rho_i/q_i` correction. Without that arm,
adaptive sampling changes the optimized task mixture and any benefit is not
pure sampling efficiency.

## 7. Discriminating interventions

1. **Group size:** `N in {2,4,8,16}` at matched total episodes. The sampled
   success frontier should move from `p*=0.5` toward about `0.17`. This is the
   cleanest signature unique to the compute-indexed statistic.
2. **Irreducible-noise negative control:** include mixed-success arms whose
   outcome depends on an unobservable random bit and cannot improve. Persistent
   selection falsifies “mixed success equals learnability.”
3. **Transfer:** compare one shared command-conditioned actor with
   capacity/data-adequate controls and measure the task-gradient cosine matrix.
   Current MountainCar controls do not establish causal necessity.
4. **Surrogate fidelity:** if PPO clipping or dense auxiliary losses are added,
   measure their gradient cosine with the exact unclipped grouped update before
   interpreting results as MaxRL.

## 8. Hindsight is a later, separate study

Joystick locomotion does not provide an obviously fresh-law-equivalent
achieved-goal relabel. Do not enable hindsight in Stages 1–3. A later UniLab
point-reach or manipulation task should compare no hindsight, achieved-goal
relabeling, and fresh rollouts under the credited goal. Verifier validity and
goal rewriting are necessary; equality of update moments is the actual
exactness criterion.

## 9. Statistics and stop rules

- Use paired warm starts and seed as the statistical unit.
- Separate development and confirmation seeds.
- Include initialization in AUC and integrate over actual transitions.
- Use paired bootstrap confidence intervals and exact paired sign-flip tests;
  Holm-correct the registered contrast family.
- Freeze absolute, relative, and ULP reconstruction tolerances before runs.
- Stop for inadequate headroom, missing mixed groups, corrupted lifecycle
  accounting, non-reconstructable diagnostics, or violated compute matching.
- Never inspect a stopped confirmatory outcome family to choose the next arm.

## 10. Closest prior art and positioning

Frontier/learnability sampling, procedural curricula, and hindsight are
established directions. Required comparators and framing include SFL/No
Regrets, PLR/PAIRED/ALP-GMM, SEC-style advantage teachers, DAPO dynamic
sampling, HER/USHER, and transfer-aware curricula. The proposed novelty is the
finite-group estimator-specific statistic and its tested coupling to the
matching learner—not the generic phrase “frontier RL.”

Primary-source starting points:

- SFL / No Regrets: <https://arxiv.org/abs/2408.15099>
- PLR: <https://arxiv.org/abs/2010.03934>
- PAIRED: <https://arxiv.org/abs/2012.02096>
- SEC: <https://arxiv.org/abs/2505.14970>
- DAPO: <https://arxiv.org/abs/2503.14476>
- HER: <https://arxiv.org/abs/1707.01495>
- USHER: <https://proceedings.mlr.press/v205/schramm23a.html>
- UniLab: <https://arxiv.org/abs/2605.30313>
