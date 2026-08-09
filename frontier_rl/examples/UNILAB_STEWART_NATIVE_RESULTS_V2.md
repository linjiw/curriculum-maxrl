# UniLab Stewart V2: native grouped MaxRL, target preservation, and variance tracking

Date: 2026-07-23

Status: **three-seed development evidence; no sampler-superiority or broad
robotics claim**.

This supersedes V1 for the repository's current UniLab claim.  V1 remains a
useful historical development study, but its independent runner used a stock
15-observation policy and selected a post-hoc warm start.  V2 lives natively in
the UniLab worktree, uses the registered 16-observation grouped task owner, and
fails closed on checkpoint provenance, task adequacy, episodewise evaluation,
sampling probabilities, importance arithmetic, and raw-artifact hashes.

## Native learner and immutable boundary

The Motrix CPU environment is `StewartBalanceGrouped`.  The shared policy sees
15 native observations plus normalized initial radius and emits two latent
Gaussian tilt commands; native inverse kinematics maps them to six platform
actuators.  One update freezes one radius and one policy, collects the first
complete episode from each of eight vector slots, and applies

```text
D_N = 1{K>0} sum_j (r_j/K - 1/N) S_j.
```

`S_j` is the complete detached-action likelihood-ratio trajectory score.
Practical `D_8` targets the order-7 MaxRL objective.  These runs use no critic,
PPO clipping, dense-reward actor term, trajectory replay, or hindsight.

All arms start from the final iteration-99 native PPO checkpoint:

```text
checkpoint SHA-256  e50ce701e87ed508c485ea1c4cc9d0828c9c8a341db6f4515644f25208cdd25c
owner-config SHA    adb9f48b936cea044df2babbc5403cfbdecb0c9644279eb7599ca0583368da5c
actor               16 -> 128 -> 128 -> 2, tanh, normalized observations
backend             Motrix CPU
```

The launcher rejects a mismatched checkpoint, owner config, architecture, task,
backend, or iteration before evaluating or training.

## A. Six-task teacher factorial

The ladder `[.20,.225,.25,.275,.30,.325]` passed its 512-episode-per-task
adequacy gate.  Each arm ran 120 groups, mean-only SGD at `1e-5`, and the same
fixed episodewise evaluation streams.

| arm | transition AUC | final uniform macro | mixed updates | raw coefficient mass |
|---|---:|---:|---:|---:|
| uniform | .2961 ± .0167 | .3225 ± .0221 | 101.0 | 129.0 |
| learnability | .2935 ± .0163 | .3220 ± .0128 | 103.0 | 125.0 |
| learnability + `rho/q` | .2933 ± .0152 | .3164 ± .0261 | 105.3 | 129.5 |
| `u_8` coefficient mass | .2872 ± .0224 | .3229 ± .0192 | 100.7 | 132.6 |
| `u_8` + `rho/q` | .2923 ± .0163 | .3082 ± .0156 | 100.0 | 132.8 |

The registered corrected-`u_8` minus uniform contrast was `-.0039` AUC and
`-.0143` final macro.  Three seeds do not establish harm or equivalence.  The
fixed-policy ladder offered only 3.35% proportional-`u_8` coefficient-mass
headroom, so the study was also opportunity-limited.

## B. Broader opportunity ladder

An independent 256-episode screen selected eleven radii from `.16` through
`.40`.  Its fixed-pass-rate plug-in predicted 15.06% more raw coefficient mass
and 7.89% more mixed-group probability than uniform.  The paired follow-up
verified that mechanism:

| arm | transition AUC | final uniform macro | mixed updates | raw / weighted mass | importance ESS fraction |
|---|---:|---:|---:|---:|---:|
| uniform | .3247 ± .0072 | .3404 ± .0040 | 82.3 | 102.7 / 102.7 | 1.000 |
| `u_8` | .3250 ± .0116 | .3416 ± .0154 | 88.0 | 110.1 / 110.1 | 1.000 |
| `u_8` + `rho/q` | .3252 ± .0085 | .3345 ± .0093 | 88.7 | 117.0 / 103.7 | .768 |

Corrected `u_8` increased raw mass by about 14%, close to the screen
prediction, but its performance contrast was only `+.0006` AUC and `-.0059`
final macro.  Importance correction cancels the expected mass gain exactly:

```text
E_q[(rho_I/q_I) mass_I] = sum_i rho_i E[mass_i].
```

It also pays importance variance.  This is a direct empirical reason not to
equate scalar coefficient mass with target-preserving efficiency.

## C. Gradient-second-moment allocation

For a fixed target distribution, the corrected task estimator has second
moment

```text
E ||(rho_I/q_I)D_I||^2 = sum_i rho_i^2 E||D_i||^2 / q_i,
```

so the unconstrained minimizer is

```text
q_i proportional to rho_i sqrt(E||D_i||^2).
```

The native `gradvar_importance` teacher estimates the raw conditional squared
gradient online and applies exact `rho/q`.  A `.30` uniform floor retains full
support and bounds every possible importance weight by 3.33.

| arm | transition AUC | final uniform macro | mixed updates | ESS fraction | observed corrected gradient second moment |
|---|---:|---:|---:|---:|---:|
| uniform | .3247 ± .0072 | .3404 ± .0040 | 82.3 | 1.000 | 9846.6 ± 221.5 |
| `u_8` + `rho/q` | .3267 ± .0134 | .3345 ± .0114 | 87.7 | .878 | 8979.3 ± 2815.2 |
| gradient moment + `rho/q` | .3242 ± .0109 | .3343 ± .0124 | 80.7 | .949 | 8833.0 ± 1541.5 |

The second-moment teacher improved the importance-weight mechanism but not
learning performance.  Versus uniform, its paired contrast was `-.0004` AUC
(descriptive interval `[-.0039,.0035]`) and `-.0062` final macro
(`[-.0199,.0107]`).  Versus corrected `u_8`, it raised mean ESS fraction from
`.878` to `.949` and lowered the mean maximum observed weight from 2.82 to
1.68.  Its observed corrected-gradient squared norm was 10.3% lower than
paired uniform on average, but only two of three seeds moved downward.

Same-policy probes diagnose the gap.  At group 60, the online distribution
captured a 7.33% plug-in second-moment reduction; computing a distribution
directly from the same probe moments with the same floor yielded 17.44%.  At
group 120 those values were 0.45% and 19.06%.  With only about eleven selected
observations per task, the heavy-tailed moment EMA is too noisy and lagged to
track the moving policy reliably.

## Correct boundary and next gate

V2 establishes that:

1. native complete-episode grouped MaxRL training is practical on a Mac CPU
   robotic simulator;
2. proportional-`u_8` sampling can generate the predicted raw estimator
   activity when the ladder contains real opportunity;
3. target correction cancels that mass advantage and exposes importance cost;
4. theorem-linked gradient-moment sampling gives healthier weights and some
   second-moment reduction, but the current online tracker does not yield a
   stable learning gain.

The next development gate is estimation, not a new heuristic score.  Seeds
3--5 compare uniform sham calibration with a calibrated gradient-moment arm.
Both collect 32 groups per task before group 1 and after group 60; only the
calibrated arm uses the estimates, freezing the `.30`-floor mixture for the
next 60 updates.  Disjoint 16-group-per-task audits at groups 0, 60, and 120
measure achieved versus prescribed second moment.  Calibration cost is charged
to both arms and total-algorithm-transition AUC is primary.  The exact two-arm
budget is 37,209,600 backend transitions.  Do not open confirmatory seeds until
the independent mechanism gate passes.

## Evidence provenance

The authoritative implementation, protocol, analyzer, and raw artifacts are
in the sibling UniLab branch `codex/curriculum-maxrl-unilab`.  Raw V2-C hashes:

```text
seed 0  281fb0b444a5e02042914da3a36b08b828771fa2896ce542faa679e4c8ab0a69
seed 1  46693714c18297225a0bde832537931b2c3dd2149193c8cf57da95ab9e209a9e
seed 2  a050cb60d55ffbcfe56830e52e61e7ed5c58c3ae843f32ce656159c4f235cf5c
```

Across V2-A, V2-B, and V2-C, nine seed-level multi-arm replicates (33 arm
runs) consumed 85,996,800 backend transitions including adequacy, fixed
evaluation, and gradient probes.
