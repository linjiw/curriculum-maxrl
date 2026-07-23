# UniLab robotics research roadmap V2: preserve the target, repair the labels, then scale

Date: 2026-07-23

Status: **research and engineering roadmap; not a preregistration and not a
robotics-performance result**.

Audit boundary: the 40 environments registered by the sibling UniLab worktree
on `codex/curriculum-maxrl-unilab`, at commit
`b75b526223fc23442a7545eecadd3d16528f7f7d` plus the active calibrated-Stewart
and G1 lifecycle changes inspected on the date above. Re-resolve configs and
hash claim-bearing source before every experiment. This document does not
promote an uncommitted repair, a unit test, or an unfinished run into evidence.

## 1. Executive decision

UniLab contains several useful curriculum-like mechanisms, but they answer
different questions:

1. `StewartBalanceGrouped` is the only native case that currently instantiates
   the practical grouped estimator `D_8`. With exact `rho/q`, adaptive task
   sampling is a **fixed-target experimental-design** problem.
2. Rough-terrain rows, motion-reference frames, and Go2Arm command ranges alter
   the distribution on which the learner trains. They are **target-changing
   curricula** unless their updates and evaluation are corrected to a frozen
   target.
3. G1 penalty scaling changes the reward function during training. It is a
   **reward schedule**, not task sampling and not MaxRL.
4. No registered native UniLab environment implements achieved-goal hindsight
   relabeling. Hindsight is therefore a future, separate study.

The near-term path to better robotics evidence is consequently:

- finish the calibrated Stewart variance-mechanism gate without changing its
  protocol;
- repair invalid outcome labels, config composition, lifecycle accounting,
  checkpoint state, and observability before expensive locomotion training;
- run target-changing curriculum comparisons only against one frozen uniform
  evaluation distribution;
- port exact grouped `D_8` next to command-conditioned flat locomotion, where a
  shared policy has a plausible transfer channel;
- add hindsight only after a genuine goal-conditioned task exposes a verifier,
  a credited-goal rewrite, and a fresh-law comparator.

This ordering is intentionally falsifiable. A mechanism that fails its gate is
repaired or retired; it is not rescued by opening more seeds or switching the
primary metric.

## 2. Complete native environment inventory

The following grouping covers all 40 names returned by
`registry.list_registered_envs()`. “Adaptive-capable” means the environment
contains the mechanism; the composed training YAML may select a static mode or
disable it.

| Native case | Registered environments | Count | Mechanism actually present | Scientific class |
|---|---|---:|---|---|
| Exact grouped Stewart | `StewartBalanceGrouped` | 1 | Complete same-task groups, binary native verifier, practical `D_8`, finite-task teacher, exact optional `rho/q` | Fixed target only for uniform or corrected adaptive sampling; uncorrected adaptive arms change the task mixture |
| Rough terrain | `Go1JoystickRough`, `Go2JoystickRough`, `Go2WJoystickRough` | 3 | `TerrainSpawnManager`: distance promotion/demotion or discounted-Beta frontier row sampling | Target-changing reset distribution; ordinary dense-reward PPO, not grouped MaxRL |
| G1 walking | `G1WalkFlat`, `G1WalkRough`, `G1Walk23DofFlat`, `G1Walk23DofRough` | 4 | Episode-length-gated scaling of every negative reward coefficient | Reward schedule |
| Command-range expansion | `Go2ArmManipLoco` | 1 | Batch mean tracking reward can expand `vx`, `vy`, and yaw-command support | Target-changing task support |
| Motion/reference sampling | `G1MotionTracking`, `G1MotionTrackingDeploy`, `G1MotionTracking23Dof`, `G1MotionTracking23DofDeploy`, `G1BoxTracking`, `G1BoxTracking23Dof`, `G1FlipTracking`, `G1WallFlipTracking`, `G1ClimbTracking`, `G1FlipTracking23Dof`, `G1WallFlipTracking23Dof`, `G1ClimbTracking23Dof`, `G1MotionTrackingSAC`, `G1MotionTrackingSAC23Dof`, `G1FlipTrackingSAC`, `G1WallFlipTrackingSAC`, `G1FlipTrackingSAC23Dof`, `G1WallFlipTrackingSAC23Dof`, `G1WBTObs`, `G1WBTObs23Dof`, `X2WallFlipTracking` | 21 | Start, clip-start, frame-uniform, fixed mixed, or failure-count-adaptive reference-frame initialization | Static modes define a fixed reset target; `adaptive` changes the reset distribution without importance correction |
| No built-in adaptive curriculum | `Go1JoystickFlat`, `Go2FootStand`, `Go2JoystickFlat`, `Go2WJoystickFlat`, `A2JoystickFlat`, `AllegroInhandRotation`, `AllegroInhandRotationGrasp`, `SharpaInhandRotation`, `SharpaInhandRotationGrasp`, `StewartBalance` | 10 | Stationary configured task/command/reset distribution; ordinary PPO/APPO/off-policy/HORA training as configured | Fixed configured target, but no Curriculum-MaxRL mechanism |

There is no native hindsight row to add to this table. The two `*Grasp`
environments collect successful grasp states; that is data generation, not
goal relabeling and not HER.

The owner paths for the five active mechanisms are:

| Mechanism | UniLab owner path |
|---|---|
| Grouped Stewart learner/teacher | `src/unilab/algos/torch/grouped_maxrl/{runner,teacher,estimator,actor}.py` |
| Stewart grouped task | `src/unilab/envs/manipulation/stewart/grouped.py` |
| Rough terrain scheduling | `src/unilab/envs/locomotion/common/terrain_spawn.py` |
| Motion-reference scheduling | `src/unilab/envs/motion_tracking/g1/motion_loader.py` |
| G1 penalty schedule | `src/unilab/base/curriculum.py` and `src/unilab/envs/locomotion/g1/joystick.py` |
| Go2Arm command expansion | `src/unilab/envs/locomotion/go2_arm/manip_loco.py` |

### Motion-mode composition matters

The shared `MotionSampler` makes all 21 tracking environments capable of the
same modes, but defaults and YAML overrides differ. Generic motion, box,
climb, wall-flip, WBT, and X2 profiles inherit or declare `adaptive` in Python;
flip profiles default to `start`. Actual PPO/APPO/SAC YAMLs also override some
wall-flip and X2 cases to `start` or `uniform`, and SAC flip cases can use a
fixed `mixed` distribution. Every report must store the resolved
`sampling_mode`, not infer it from the registry name.

## 3. Core fixed-target mathematics: `D_8` targets `J_7`

For task `i`, let `p_i(theta)` be binary success probability. Freeze the policy
and task for eight independent complete episodes. Let

```text
R_ij in {0,1},
K_i = sum_j R_ij,
S_ij = grad_theta log m_theta(tau_ij | i).
```

The practical dropped-all-fail estimator is

```text
D_8,i = 1{K_i>0} sum_{j=1}^8 (R_ij/K_i - 1/8) S_ij.
```

Its expectation is not the order-8 gradient. It is

```text
E[D_8,i] = grad J_7(p_i),
J_7(p) = -sum_{k=1}^7 (1-p)^k/k.
```

The off-by-one follows because the score control variate is also dropped on
all-fail groups. The native Stewart runner correctly labels this practical
estimator order seven.

For a frozen target task mixture `rho`, define

```text
F_rho(theta) = sum_i rho_i J_7(p_i(theta)).
```

If task `I` is drawn from a full-support proposal `q`, then

```text
G = (rho_I/q_I) D_8,I,
E[G] = grad F_rho.
```

Thus corrected task sampling changes variance, not the optimized task
mixture. With

```text
v_i = E[||D_8,i||^2],
```

its second moment is

```text
M(q) = E[||G||^2] = sum_i rho_i^2 v_i/q_i.
```

If the `v_i` are known, positive, and there is no floor, the minimizer is

```text
q_i* proportional to rho_i sqrt(v_i).
```

This is importance-corrected experimental design, in the same broad variance
reduction family as importance sampling for stochastic gradients
([Alain et al.](https://arxiv.org/abs/1511.06481),
[Katharopoulos and Fleuret](https://proceedings.mlr.press/v80/katharopoulos18a.html)).
It is not a theorem that “intermediate success tasks learn fastest.”

The current Stewart prescription uses the safer mixture

```text
q_mix,i = (1-f) * rho_i sqrt(v_i) / sum_j rho_j sqrt(v_j) + f rho_i,
f = .30,
```

for uniform `rho`. This bounds `rho_i/q_i <= 1/f = 3.33`, but it is not the
exact hard-floor KKT solution. The exact solution of
`min_q M(q)` subject to `q_i >= f rho_i` has a water-filling form

```text
q_i = max(f rho_i, rho_i sqrt(v_i/lambda)),
```

with `lambda` chosen so probabilities sum to one. Comparing mixture-floor and
hard-floor allocation is a legitimate future ablation only after the current
registered study closes.

### Why `u_8` is a diagnostic, not the variance-optimal teacher

The exact expected absolute coefficient mass is

```text
E[sum_j |w_ij|] = 2u_8(p_i),
u_8(p) = 1-(1-p)^8-p.
```

`u_8` peaks near `p=.257`; SFL learnability `p(1-p)` peaks at `.5`
([Rutherford et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d0ed12c3fda52f2c241a0cebcf739a6-Abstract-Conference.html)).
Neither scalar contains trajectory-score norms, within-group gradient
cancellation, cross-task gradient conflict, or simulator cost. The native V2
result already showed why this distinction matters: `u_8` increased raw
coefficient mass, while exact target correction removed its expected weighted
mass advantage and exposed importance variance.

### What reducing `M(q)` can and cannot promise

If `F_rho` has an `L`-Lipschitz gradient and the learner takes
`theta+ = theta + eta G`, then

```text
E[F_rho(theta+) - F_rho(theta)]
  >= eta ||grad F_rho||^2 - (L eta^2/2) M(q).
```

Accurate variance allocation tightens this conditional one-step lower bound.
It does not guarantee a long-horizon learning gain when moment estimates are
noisy or stale, calibration consumes transitions, tasks have unequal costs,
the policy moves, or the optimizer changes the relevant norm. Those are
experimental questions.

## 4. Root analysis of each curriculum-bearing case

### 4.1 Stewart: the core algorithm and current highest-priority gate

`StewartBalanceGrouped` is currently the cleanest native test because it has a
short complete episode, a frozen binary success event, an explicit radius
task, a task-aware 16-dimensional policy, detached-action likelihood-ratio
scores, and an actor-only grouped runner. It does not use a critic, PPO
clipping, dense reward in the actor update, replay, or hindsight.

The central negative finding from V2 is informative: online selected-task
moment tracking saw only roughly eleven observations per task and lagged a
moving, heavy-tailed `v_i`. Same-policy probes indicated substantial
known-moment headroom, but the online sampler captured little of it by group
120. The next intervention is therefore better measurement, not another
success-rate heuristic.

The already recorded calibrated development protocol must remain unchanged:

- radii `.16, .18, .20, .225, .25, .275, .30, .325, .35, .375, .40`;
- `N=8`, 120 actor groups, SGD `1e-5`, fixed action standard deviation;
- `uniform_sham_calibration` versus `calibrated_gradvar_importance`;
- 32 calibration groups per task before groups 1 and 61;
- disjoint 16-group audits per task at actor groups 0, 60, and 120;
- `.30` mixture floor, exact unclipped `rho/q`;
- calibration cost charged to both arms and total-algorithm-transition AUC as
  the primary performance metric.

The mechanism passes only if all registered conditions pass on seeds 3--5:

1. mean audited `M(q_cal)/M(rho) <= .90` at groups 0 and 60;
2. the calibrated proposal captures at least 50% of the independently
   prescribed mixture's available gap;
3. the hierarchical-bootstrap 95% upper endpoint for achieved/uniform is
   below one;
4. realized training importance ESS fraction is at least `.80` in every seed.

Group 120 is a durability diagnostic. Performance remains exploratory unless
the mechanism gate passes. A failure stops confirmation and points to moment
estimation or refresh cadence; it does not falsify the known-`v` allocation
identity.

Regardless of the gate, report the fixed-uniform held-out plug-in
`mean_i J_7(p_i)` beside macro pass rate. It is an objective-aligned diagnostic,
not a replacement for the frozen primary AUC, and finite-evaluation plug-in
bias must be acknowledged. The sham arm's training-only AUC can quantify the
no-calibration transition counterfactual because streams are isolated, but it
cannot replace total-algorithm-transition accounting.

If the gate passes, freeze code, resolved config, checkpoint, raw analyzer,
and hashes before reserving new confirmation seeds. If it fails, the next
development block may compare only predeclared estimation repairs such as
adjacent-radius partial pooling, a robust mean-of-squared-norm estimator, or a
shorter refresh interval. Each repair must be judged on disjoint audit groups,
not on the calibration data that chose `q`.

### 4.2 Rough terrain: useful scheduler, invalid current competence label

`TerrainSpawnManager` has good engineering properties: finalized episode-end
events, fixed terrain-type columns once configured correctly, discounted
Beta pseudo-counts, an explicit uniform mixture, exact stored probabilities,
and checkpointed scheduler/RNG state. Its frontier modes still drive an
ordinary dense-reward learner and do not form same-task groups or apply
`D_8`.

Two blockers make present training scientifically uninterpretable.

**Blocker A: “success” currently means only survival to the time limit.**
`NpEnv.full_horizon_success` returns true for a time-limit truncation without
termination. All three rough implementations currently return an all-false
task-termination vector; terrain out-of-bounds is the only earlier truncation
in the owner configs. A zero-action robot that stays inside the terrain can
therefore be labeled successful without following its velocity/yaw command.
The integration tests that observe real Go1/Go2 timeouts verify event plumbing,
not competence.

Before any curriculum comparison, freeze a command-aligned verifier on
development data, for example

```text
success(e) = 1{reaches T_max without OOB}
             * 1{mean_t 1[||v_xy(t)-c_xy(t)|| <= eps_v
                            and |omega_z(t)-c_omega(t)| <= eps_w] >= zeta}
             * 1{upright/contact safety constraint passes}.
```

`eps_v`, `eps_w`, `zeta`, safety thresholds, and whether command changes split
the episode must be frozen before outcome comparisons. A regression test must
make zero-action/nonzero-command timeouts fail.

**Blocker B: the terrain generator and teacher have separate switches.** The
Go1 and Go2 owner YAMLs currently set both
`env.terrain_curriculum.enabled=false` and
`env.scene.terrain.generator.curriculum=false`. Existing Hydra tests turn on
the first switch without the second. With generator curriculum off, terrain
type and difficulty are sampled independently per patch; row and column no
longer mean ordered difficulty and fixed type. A valid run requires both:

```text
env.terrain_curriculum.enabled=true
env.scene.terrain.generator.curriculum=true
```

Go2W is code-capable through `TerrainSpawnManager`, but its rough YAML does not
expose the `terrain_curriculum` block, so it should not enter the first study
until composition and integration tests match Go1/Go2.

After both blockers close, the first dense-PPO study compares fixed uniform
rows, distance promotion/demotion, frontier `p(1-p)`, and `u_8`-shaped frontier
sampling. The last arm is only an estimator-shaped scheduler ablation. Evaluate
all arms on the same frozen uniform row-by-type grid, including initialization
in transition AUC. Report command-tracking success, tracking errors, falls/OOB,
per-arm visits and probabilities, easy-row retention, worst-row success,
transitions, and wall time. No arm may be called MaxRL training.

### 4.3 Motion tracking: broadest native curriculum surface, weakest accounting

Adaptive `MotionSampler` currently computes

```text
raw_b = failure_count_ema_b + alpha_uniform/B,
q_b = raw_b / sum_j raw_j,
```

where `failure_count_ema_b` is updated only on calls containing at least one
termination. This has five root problems:

1. It tracks raw failure counts without visits or exposure time, so a frequently
   sampled bin can create its own priority.
2. `adaptive_uniform_ratio=.1` is not a 10% mixture. Its effective total
   uniform mass is `.1/(sum_b failure_count_ema_b + .1)` and can vanish.
3. With no failure event the state does not decay or record successful
   exposure; `adaptive_alpha=.001` has a clock tied to failure-containing
   vector calls, not completed episodes.
4. Sampler evidence and global NumPy RNG are not checkpointed, so resume
   silently restarts the curriculum.
5. Entropy/top-bin fields exist but are not routed into training logs, and the
   adaptive hyperparameters other than mode are not exposed by the environment
   config.

There is also a smaller composition trap: `G1BoxTrackingEnv` reconstructs its
sampler after replacing the loader but passes only mode and environment count;
a future box `mixed` config would lose `sampling_start_ratio` and fall back to
zero.

The repair should define the statistical object first. For a reset-frame bin,
store starts/visits, completed episodes, failures, age, the resolved proposal,
and the exact credited frame. Estimate an exposure-normalized discounted
failure probability, then use an explicit mixture

```text
q = (1-f) normalize(score) + f/B.
```

Add owned RNG, fail-closed state serialization, resolved hyperparameters, and
per-bin logs. Decide whether the target is frame-uniform, clip-uniform, or a
fixed mixture; these differ when clips have different lengths.

The first falsifiable experiment is not “frontier MaxRL.” Use one G1 tracking
task with adequate CPU throughput and compare:

- static uniform frame initialization;
- current raw failure-count adaptive sampling as a historical comparator;
- repaired failure-rate sampling;
- an absolute-learning-progress proposal.

Absolute learning progress is the natural continuous-task baseline from
[ALP-GMM](https://proceedings.mlr.press/v100/portelas20a.html); replay priority
and staleness diagnostics should borrow from
[Prioritized Level Replay](https://arxiv.org/abs/2010.03934). Keep transitions,
optimizer updates, replay ratio, and evaluation calls matched. The primary
metric is reference-tracking success/error AUC under a fixed uniform evaluation
over frames or clips, not the adaptively sampled training return. Fail the
study if proposal reconstruction, minimum coverage, resume equivalence, or
per-bin exposure accounting fails.

For off-policy SAC variants, task-draw `rho/q` alone cannot restore a fixed
target: replay occupancy and behavior-policy mismatch remain. Establish the
reset scheduler first with a fixed-target evaluator; do not label a SAC run
target-preserving without an explicit replay/occupancy argument.

### 4.4 G1 walking: a lifecycle repair, then a reward-schedule ablation

`PenaltyCurriculum` identifies all negative reward coefficients, initializes
them at scale `.5`, and moves the common scale within `[.5,1]` according to an
EMA of episode length. Fifteen checked-in training configs enable it across
SAC, FlashSAC, TD3, and one rough PPO owner.

The audit found a lifecycle defect in the prior implementation. It updated
inside `update_state`, before `NpEnv.step` increments episode length and
computes time-limit truncation. Normal time-limit episodes were therefore
invisible, while early episodes were observed before finalization. The active
repair moves the update to `on_episode_end`, after all terminal flags and the
final step count are frozen, and adds exact-once tests for early and time-limit
episodes. Any earlier curriculum artifact should be treated as invalid for
schedule claims.

Two issues remain before training evidence:

- persist tracker state, current scale, and original reward weights in the
  environment checkpoint contract;
- define evaluation independently of the moving reward weights.

The falsifiable ablation is `schedule off` versus `fixed .5`, `fixed 1.0`, and
`adaptive`, with matched learner settings and final-policy evaluation on fixed
task metrics such as command tracking, fall rate, energy/torque, constraint
violations, and a frozen reporting reward. A schedule that raises its own
moving training return but not those fixed metrics has not improved the
robotic policy. No result from this family is evidence for the `D_8 -> J_7`
algorithm.

### 4.5 Go2Arm: support expansion, not fixed-target curriculum

`Go2ArmManipLoco` accumulates per-episode linear-velocity tracking reward. If
the reset batch mean exceeds `.8`, it expands global `vx`, `vy`, and yaw ranges
by configured steps until fixed maxima. Standard PPO owners disable this
mechanism; the MuJoCo PPO-HIM owner enables it from a narrower initial range.

The mechanism changes task support, aggregates heterogeneous reset episodes
into one global decision, and does not checkpoint the mutated command range or
per-episode curriculum accumulators. It should first gain owned state,
per-command-bin evidence, exact range logging, deterministic resume, and a
fixed full-range evaluator.

Then compare no expansion, the current threshold rule, absolute learning
progress, and a distribution-constrained self-paced baseline. This case is
conceptually closer to
[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html),
[GoalGAN](https://proceedings.mlr.press/v80/florensa18a.html), and
[self-paced deep RL](https://arxiv.org/abs/2004.11812) than to fixed-target
importance sampling. The primary result must be full-range locomotion plus
end-effector performance, with separately reported zero-command retention;
success on the current expanded training range is circular.

### 4.6 The ten stationary cases: controls and future axes

The ten environments without an adaptive curriculum are not missing data;
they are essential controls. Their configured command/reset/domain-randomized
mixture is stationary and can anchor evaluation.

- `Go1JoystickFlat`, `Go2JoystickFlat`, and `Go2WJoystickFlat` expose command
  conditioning and are promising task-bin donors. `Go2JoystickFlat` is the
  best next exact grouped-transfer candidate because command bins are visible
  to one shared actor without adding a task-ID-only shortcut.
- `Go2FootStand` has meaningful termination and fixed-pose metrics, but no
  existing difficulty axis. Do not invent one until a fixed-policy sweep shows
  controllable mixed-success bins.
- Allegro/Sharpa rotation and grasp-collection tasks have useful manipulation
  infrastructure but no frozen grouped binary objective. First separate
  rotation competence, object retention, and grasp-state generation.
- `StewartBalance` remains the dense-PPO owner and warm-start source;
  `StewartBalanceGrouped`, not the stock owner, is the exact experiment.
- `A2JoystickFlat` is a later cross-morphology generalization check, not a
  development target while the mechanism remains unsettled.

## 5. The next exact robotics port: command-conditioned `Go2JoystickFlat`

This is the first proposed expansion of the core algorithm, conditional on the
Stewart mechanism gate and a CPU feasibility screen.

### Task and verifier contract

Choose a small, frozen grid of complete-episode commands `(vx, vy, omega_z)`.
The actor must observe the command, and one command must remain fixed for all
eight episodes in a group. Define binary success from a predeclared fraction of
steps within linear/yaw tracking tolerances, plus upright and safety criteria.
Calibrate tolerances only on development seeds and then freeze them.

Reject the task axis unless a fixed checkpoint yields:

- at least four bins with pass rates in `[.05,.80]`;
- hardest-bin pass rate above `.02`;
- failures that change under policy intervention rather than reset noise only;
- evaluation that leaves actor, simulator, teacher, and training RNG state
  unchanged.

### Exact learner contract

1. Freeze policy parameters for a collection wave.
2. Draw one task bin from recorded `q`.
3. Collect the first complete episode from each of eight independent slots.
4. Score detached executed actions under the frozen command-conditioned policy.
5. Build `D_8` from the frozen binary verifier only.
6. Apply exact `rho/q` once and one actor update; discard the wave.
7. Keep dense return for diagnostics only; do not add it to the actor loss in
   the first port.

Required development arms are uniform and calibrated gradient-moment
importance sampling with sham-matched calibration. An uncorrected adaptive arm
may be included only as an explicitly target-changing diagnostic. Every report
must include uniform-target macro AUC, worst-bin success, gradient second
moment, ESS, importance maxima, mixed/all-fail/all-pass group rates, transition
and episode costs, and per-bin gradient-cosine matrices.

The cross-bin gradient matrix is central: a curriculum can help one shared
policy only when selected-bin updates transfer acceptably to the fixed target.
If gradients conflict, variance-optimal allocation can work exactly as
designed yet fail to improve target performance.

## 6. Hindsight: no native evidence and no Stewart shortcut

[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) rewrites goals
in off-policy goal-conditioned data. Correct rewards are necessary but do not
make a grouped MaxRL update fresh-law equivalent. Hindsight selection
conditions on the observed group, changes which goals appear, and can alter
both positive and negative trajectory laws; work such as
[USHER](https://proceedings.mlr.press/v205/schramm23a.html) exists precisely
because this bias needs explicit treatment.

For credited goal `g`, define

```text
P_theta,g = law of N fresh episodes requested under g,
Q_theta,g = law of the accepted rewritten group credited to g,
h_theta,g = the actual goal-rewritten grouped score update.
```

Exactness of a per-accepted-goal update requires

```text
E_Q[h_theta,g] = E_P[h_theta,g].
```

Matching verifier values is insufficient, and even per-goal equality does not
recover a global target mixture unless goal-selection frequencies are also
corrected.

Stewart radius tasks are distributions over initial ball position, not desired
goals attached to an otherwise unchanged trajectory. Re-crediting a trajectory
to a different radius generally does not reproduce the reset law of a fresh
episode. Do not use radius relabeling as HER.

A future hindsight study needs a true goal-conditioned manipulation/reach
environment with `desired_goal`, `achieved_goal`, a goal-dependent verifier,
and a policy-conditioning rewrite. Compare no hindsight, HER-style relabeling,
a bias-corrected method, and fresh rollouts under the credited goal. Measure
update-moment discrepancy against fresh groups before measuring performance.

## 7. Falsifiable staged program

| Priority | Stage | Required intervention | Pass condition | Consequence of failure |
|---:|---|---|---|---|
| P0 | Calibrated Stewart development | Finish the frozen seeds 3--5 protocol and independent analyzer | All four registered mechanism gates pass; group-120 durability reported separately | No confirmation; open a new estimation-only development block |
| P1 | Curriculum integrity | Land G1 finalized lifecycle tests; add state persistence; replace rough survival label; compose both terrain switches; repair MotionSampler statistics/state/logs | Exact-once outcomes, fixed verifier, reconstructable `q`, explicit floor, deterministic same-version resume | No long training on the affected family |
| P2 | Motion scheduler pilot | Uniform vs legacy count vs repaired rate vs ALP on one G1 task | Fixed-uniform-eval AUC improves with coverage/resume gates intact across paired development seeds | Retire or redesign adaptive frame priority; do not call it MaxRL |
| P3 | Rough dense-PPO scheduler | Uniform, distance, learnability, `u_8`-shape under ordered terrains | Command-aligned uniform-eval AUC and worst-row gains without easy-row collapse | Keep uniform/distance; diagnose labels or task axis |
| P4 | Exact command-conditioned Go2 | Native complete `D_8` port, uniform vs calibrated corrected allocation | Adequacy plus audited moment reduction; performance only after mechanism pass | Return to verifier/conditioning/variance estimation, not more heuristics |
| P5 | Reward/support curricula | G1 penalty ablation and Go2Arm full-range study | Gains on fixed physical metrics/full target range and exact resume | Treat schedule as training convenience, not an algorithmic result |
| P6 | Goal-conditioned hindsight | New task plus fresh-law moment audit | Rewritten and fresh update moments agree within frozen tolerance before efficacy | No exact-hindsight claim; keep as biased auxiliary ablation |
| P7 | Breadth | New morphology/manipulation confirmations with frozen protocols | Same registered mechanism and primary effect on fresh seed blocks | Narrow the domain claim to environments that passed |

### Common statistical contract

- Seed, not vector environment or task bin, is the independent replicate.
- Pair arms by initialization, checkpoint, evaluation stream, and total
  algorithm transitions; use disjoint sampling streams inside each arm.
- Separate development and confirmation seeds. Never tune a gate on a seed
  block and relabel that block confirmatory.
- Include initialization and all decision-path calibration cost in primary
  transition AUC. Report training-only AUC and wall time secondarily.
- Evaluate on a fixed target distribution, never the arm's adaptive training
  distribution.
- Report task-level outcomes before macro aggregation; tasks within one run
  are correlated observations, not independent seeds.
- Use paired intervals and exact paired randomization/sign tests when the
  registered design supports them. A three-seed development block is a
  mechanism screen, not a general efficacy claim.
- Fail closed on missing hashes, nonfinite probabilities, lost support,
  incorrect `rho/q`, duplicate/missed episode events, evaluation-state
  mutation, cost mismatch, or nonreconstructable metrics.

## 8. Performance-oriented algorithm research after P0

The following extensions preserve the core question better than adding more
success-rate heuristics:

1. **Out-of-sample moment estimation.** Compare independent calibration,
   adjacent-task partial pooling, and robust heavy-tail estimators using
   disjoint audits. The selection criterion is achieved `M(q)`, not training
   return.
2. **Refresh versus staleness.** Measure how quickly audit `v_i` and optimal
   `q` drift after actor updates. Choose refresh cadence from a new development
   block and charge its transitions.
3. **Hard-floor water filling.** Once estimates are adequate, compare the
   current convex mixture with the exact constrained optimizer at matched
   maximum importance weight.
4. **Optimizer-aware moments.** For preconditioned updates, audit the squared
   norm after the frozen preconditioner as well as raw Euclidean
   `||D_i||^2`. Do not change the norm post hoc to obtain a pass.
5. **Stratified task batches.** Compare one-task stochastic allocation with a
   target-preserving stratified cycle that collects predeclared groups per
   task. This can reduce between-task variance without large `rho/q`, at a
   potentially higher latency cost.
6. **Within-task control variates.** Reduce trajectory-score variance without
   changing task sampling, while preserving the exact estimator convention.
   Retaining the score control variate at all-fail groups changes the target
   from practical `J_7` to the corresponding order-8 estimator and must be
   labeled as a different objective.
7. **Transfer diagnostics.** Measure task-gradient norms and cosine matrices.
   Allocation should be interpreted jointly with target-gradient alignment,
   especially for command and motion curricula.
8. **Cost-aware design.** When episode lengths or simulator costs differ by
   task, record per-task transitions and wall time and solve an explicitly
   stated cost-constrained allocation problem. The equal-cost
   `rho sqrt(v)` rule cannot be imported unchanged.
9. **Group-size signature.** At matched total episodes, compare
   `N in {2,4,8,16}`. Practical `D_N` changes both its population target to
   `J_{N-1}` and its coefficient-mass frontier, so this is a mechanism study,
   not a hyperparameter sweep that can pool objectives.
10. **Irreducible-noise negative control.** Add mixed-success bins whose
    outcome depends on hidden randomness that the policy cannot influence.
    A teacher that persistently favors them falsifies the shortcut “mixed
    outcomes imply useful learnability.”
11. **Learner interaction.** After the exact actor-only result is stable,
    register `D_8` versus binary PPO/RLOO-style learner comparisons. Measure
    gradient cosine with the exact unclipped update before adding clipping,
    critics, dense auxiliary loss, or rollout reuse.

## 9. Literature routing, not literature decoration

The most relevant methods become comparators only in the cases whose objects
they actually match:

| Literature | Native use in this roadmap | Boundary |
|---|---|---|
| Gradient importance sampling ([Alain et al.](https://arxiv.org/abs/1511.06481); [Katharopoulos and Fleuret](https://proceedings.mlr.press/v80/katharopoulos18a.html)) | Calibrated Stewart and exact command-bin allocation | Fixed target requires exact proposal probability and `rho/q` |
| [SFL learnability](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d0ed12c3fda52f2c241a0cebcf739a6-Abstract-Conference.html) | Rough terrain and motion scheduler baseline | `p(1-p)` is not `v_i` and does not imply `D_8` variance reduction |
| [ALP-GMM](https://proceedings.mlr.press/v100/portelas20a.html) | Continuous motion-frame or command curriculum baseline | Absolute progress changes the training distribution |
| [Prioritized Level Replay](https://arxiv.org/abs/2010.03934) | Discrete terrain/frame replay, staleness, and coverage baseline | Replay priority is not an on-policy grouped estimator |
| [Reverse Curriculum](https://proceedings.mlr.press/v78/florensa17a.html), [GoalGAN](https://proceedings.mlr.press/v80/florensa18a.html), [self-paced deep RL](https://arxiv.org/abs/2004.11812) | Go2Arm command/goal support expansion | These are target-support curricula; evaluate on a frozen final target |
| [HER](https://arxiv.org/abs/1707.01495), [USHER](https://proceedings.mlr.press/v205/schramm23a.html) | Future goal-conditioned manipulation | Verifier correctness alone does not establish fresh-law grouped updates |

## 10. Claim ladder

Evidence should advance only one rung at a time:

1. **Infrastructure:** the Mac CPU path runs and lifecycle/cost/provenance
   contracts reconstruct.
2. **Estimator identity:** native `D_8` scores complete frozen-policy groups and
   targets `J_7`.
3. **Mechanism:** a calibrated corrected proposal lowers independently audited
   gradient second moment at adequate ESS.
4. **Local efficacy:** on fresh paired seeds, the fixed-target robotic metric
   improves at matched total algorithm cost.
5. **Transfer:** the same frozen mechanism and decision rule work on a second
   command-conditioned robotic family.
6. **Breadth:** multiple morphologies and manipulation/locomotion tasks confirm
   a predeclared effect.

Current native evidence reaches rung 2 and bounded pieces of rung 3 on Stewart;
the calibrated gate is designed to decide whether rung 3 is robust enough to
attempt confirmation. Rough terrain, motion tracking, G1 penalty scheduling,
and Go2Arm are promising engineering surfaces, but none may be used to skip
that ladder or support a general robotics claim.
