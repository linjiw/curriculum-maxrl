# UniLab Stewart V1 results: a viable exact pipeline, not a teacher win

Status: **three-paired-seed development result; not confirmatory**.  All runs
used the UniLab Motrix CPU backend on an Apple M4.  The original UniLab checkout
was left untouched; integration work used base commit
`515111162634dd14c67882f9c664d9a780c6f0c5` in a separate worktree.

## 1. The first apparent frontier was spurious

The zero-action probe used the stock `StewartBalance` reset: uniform in disk
area, maximum physical radius `0.8*ratio`, zero initial velocity, native
stillness radius `0.12`, and five still transitions.  Its five-replicate pooled
pass rates looked attractive:

```text
ratio       .15    .20    .30    .40    .50    .60    .70    .80
pass rate 1.000   .571   .255   .150   .103   .069   .050   .034
```

But the passive reset probability is exactly

```text
P(start inside goal) = min(1, (0.12/(0.8*ratio))^2).
```

For ratios `.20`--`.70` this gives `.5625,.2500,.1406,.0900,.0625,.0459`,
essentially the observed ladder.  The stock 0.2-second experiment therefore
measured reset geometry, not policy-controllable learnability.

The exact grouped runner nevertheless served as a useful implementation audit.
At 120 groups and 9,600 matched backend transitions per arm, the seed-0 AUCs
were `.1855` uniform, `.1832` learnability, and `.1816` `u_8`; none moved
meaningfully.  The artifact is
`unilab_stewart_grouped_seed0_dev.json`.

## 2. Repairing the task and finding a warm start

The repaired task uses:

- exact reset-radius ratios `[.175,.275,.375,.475]`, with random angle;
- a 6-second complete-episode horizon;
- no success before 0.5 seconds;
- ten new stillness transitions required after that gate;
- fall or timeout as failure;
- one shared Gaussian actor.

The UniLab integration now provides this as `StewartBalanceGrouped`, including
an explicit native `episode_success` value before autoreset and a 16th policy
observation containing normalized task radius.  The independent exact runner
uses the same physical/verifier task but keeps the stock 15 observations so it
can load a stock-PPO checkpoint; the initial displacement itself identifies
the radius at reset.

The new 16-observation owner also completed its full configured CPU smoke:
100 PPO iterations, 204,800 transitions, and 11.0 seconds of training time.
Mean episode length reached 252.6 steps and final dense mean reward was 106.0.
Those are integration/controllability diagnostics, not a binary curriculum
comparison.

A single stock dense-PPO control ran 3,276,800 transitions in 160.6 seconds on
CPU (128 environments, seed 1).  Its final training return was not a sufficient
selection metric.  A post-hoc fixed-radius adequacy sweep found iteration 200
to be the useful checkpoint:

```text
checkpoint    per-radius pass rates                    macro
iteration 0   [.148, .000, .000, .000]                 .037
iteration 200 [.211, .133, .133, .039]                 .129
iteration 399 [.047, .047, .047, .031]                 .043
```

The selected checkpoint SHA-256 is
`ada83c1be1411700375c8f12e49f98e8af26edd82a35beb198692bbd6b6b6e15`.
This selection is explicitly development-only.  A future confirmation must
freeze a warm-start rule before inspecting fresh outcomes.

## 3. Exact grouped development comparison

Every arm starts from that identical checkpoint and uses:

- `N=8` complete frozen-policy episodes per group;
- practical `D_8` coefficients, hence the exact order-7 objective;
- the true complete trajectory score of the sampled latent Gaussian actions;
- no critic, PPO clipping/reuse, dense-reward actor term, or hindsight;
- SGD learning rate `1e-5`;
- 120 groups = 288,000 backend transitions per arm;
- 128 fixed stochastic evaluation episodes per task every 20 groups.

The action passed to the score is detached from the reparameterized sampling
path.  A deterministic regression check requires a nonzero actor-mean score
gradient; otherwise `raw_action-mean` would cancel it and the claimed
REINFORCE estimator would be false.

Three paired development seeds produced:

| sampler | transition AUC, mean ± SD | final macro pass | mixed/update groups | all-fail groups | realized coefficient mass |
|---|---:|---:|---:|---:|---:|
| uniform | .2249 ± .0314 | .3535 ± .0704 | 98.7 | 21.3 | 140.25 |
| learnability `p(1-p)` | **.2427 ± .0203** | **.3848 ± .0068** | 104.3 | 15.7 | 145.50 |
| exact mass `u_8(p)` | .2318 ± .0041 | .3763 ± .0506 | **107.0** | **13.0** | **151.25** |

Paired AUC differences were:

```text
learnability - uniform  +.0178  descriptive bootstrap [-.0085, +.0514]
u_8 - uniform           +.0069  descriptive bootstrap [-.0181, +.0361]
u_8 - learnability      -.0109  descriptive bootstrap [-.0285, +.0111]
```

With only three pairs, exact two-sided sign-flip p-values are necessarily
coarse (`.50`, `.75`, `.50`) and do not support a ranking.  The raw artifacts
and deterministic aggregation are:

- `unilab_stewart_grouped_fixed_radius_seed{0,1,2}_dev.json`;
- `analyze_unilab_stewart_grouped.py`;
- `unilab_stewart_grouped_fixed_radius_analysis_dev.json`.

## 4. Correct finding

The result supports two narrow statements.

1. **The Mac-CPU robotic simulator path is viable.**  A complete exact grouped
   on-policy estimator learned the repaired manipulation task, moving every
   arm from macro `.129` to roughly `.35`--`.38` at a matched budget.
2. **`u_8` predicts estimator activity better than useful performance.**  It
   reduced zero-update all-fail groups by about 39%, increased update-bearing
   groups by about 8%, and collected about 8% more scalar coefficient mass than
   uniform.  It did not have the best mean AUC.

This is not evidence that `u_8` is worse than learnability, nor that either
teacher beats uniform generally.  It is direct evidence against treating
scalar coefficient mass as gradient norm, alignment, variance reduction, or
learning progress.

## 5. Next experiment

The next registered study should keep this task and instrument the missing
bridge rather than merely add seeds:

1. freeze the grouped UniLab owner and warm-start selection rule;
2. use at least eight fresh paired seeds;
3. compare uniform, learnability, `u_8`, and a mean-absolute-advantage/learning-
   progress teacher;
4. record per-task held-out `p`, predicted `u_8`, score-gradient squared norm,
   cosine with a fixed uniform-mixture evaluation gradient, and cross-task
   gradient cosines;
5. add a target-preserving `rho/q` arm so adaptive-mixture improvement is not
   confused with changing the objective;
6. then intervene on `N in {2,4,8,16}` at matched total episodes.  Movement of
   the best sampling band with `p_N*=1-N^{-1/(N-1)}` is the distinctive test of
   the compute-indexed theory.
