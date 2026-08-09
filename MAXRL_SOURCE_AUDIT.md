# Official MaxRL source audit

Audited 2026-08-09, read-only. Official repository:
[tajwarfahim/maxrl](https://github.com/tajwarfahim/maxrl), pinned at commit
[`7197bbb46a2ecd866da52f6b401ff20a34fe9390`](https://github.com/tajwarfahim/maxrl/commit/7197bbb46a2ecd866da52f6b401ff20a34fe9390)
(2026-05-28).  The repository has no paper release tag, and this commit
postdates the arXiv paper, so every conclusion below is about this precise
source snapshot rather than an assumed paper-era tree.

## What the released experiments do

| Suite | Released training schedule | Evaluation | Seeds and learning rate | Retained evidence |
|---|---|---|---|---|
| ImageNet-256 | ResNet-50, 256 images/batch, `N=1024` categorical actions/image, 20 epochs (about 100k minibatches) | initial and every 1,000 steps; analytic `1-(1-p_y)^k` through `k=2048` | explicit seed 69; LR `.1` | checkpoints and aggregate W&B metrics; no per-example outcome ledger |
| Maze RL | 999,744 training mazes, 256 groups/step, `N=128` (32,768 trajectories/step), 10 released epochs (about 39,050 steps) on four GPUs | before training and every 250 steps; 256 test mazes ×2,048 generations | launcher has no experiment seed; dataloader fallback 1; LR `1e-4` | generation dump unset; no task-level outcomes or raw trajectories committed |
| SmolLM GSM8K | SmolLM2-360M, batch 256, `N=128`, released 200 epochs (about 5,800 steps) on eight GPUs | before training/every 100 steps; `n=32`, temperature `.6`, top-p `.95` | no experiment seed; data default 1, vLLM default 0; LR `1e-5` | aggregate validation/W&B only |
| Qwen3 math | Qwen3-1.7B default, Polaris-53K, batch 256, `N=16`, five epochs (about 1,035 steps) on 32 GPUs | before training/every 50 steps/last; released command includes AIME25 and MATH-500 | exported seed 79 is not passed; framework defaults apply; LR `1e-6` | no task outcomes or rollout ledger |

Primary launch sources: [ImageNet](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/imagenet/imagenet_training_script.sh),
[Maze](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/maze/maze_17.sh),
[SmolLM](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/smollm/smollm.sh),
and [Qwen3](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/qwen3_experiments/run_qwen3_training.sh).

The maze generator requests one million 17-by-17 Prim mazes with seed zero,
shuffles them, and assigns 256 to test.  The released source does not assert
uniqueness or train/test non-overlap, and the SFT checkpoint lacks frozen
dataset hashes, optimizer/RNG state, and a command that independently
recreates the stated 1,500-step stopping point.

## Direct source validation of our estimator analysis

The released MAXRL advantage path computes a centered binary reward divided
by the group reward mean (plus epsilon).  Therefore:

- for `K>0`, it is the practical centered weight up to a common scale;
- for `K=0`, every coefficient is exactly zero;
- the parsed `truncate_order` and `pass_k` values do not affect this path.

See [the implementation](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/core_algos.py#L402-L441)
and [trainer routing](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/ray_trainer.py#L270-L287).
This independently supports our convention mapping in the `epsilon -> 0`
idealization: centering only when `K>0` and dropping the complete all-failure
group corresponds to truncation order `T=N-1`, not `T=N`.  Finite epsilon
slightly perturbs the coefficient scale, so the released numerical path is not
literally the exact truncated-objective gradient.  The correction also does
not apply to a direct estimator or a non-dropped score-baseline control
variate.

The ImageNet path uses the same centered-by-mean structure under
`reinforce_with_p_normalization` in its
[sampling-based objective code](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/cifar10_experiments/sampling_based_rl_objective_experiments.py#L285-L318).

## The released LLM “pass@k” logger is a bootstrap proxy

The RL evaluator performs 1,000 with-replacement resamples and averages their
maximum reward; the trainer later exposes that `best@N/mean` value under a
`pass@N_accuracies` label.  See
[metric computation](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/metric_utils.py#L258-L296)
and [renaming in the trainer](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/ray_trainer.py#L1213-L1233).

For `c` binary successes in `n` retained samples, this targets approximately

`1-(1-c/n)^k`,

not the standard without-replacement estimator

`1-C(n-c,k)/C(n,k)`.

For example, when `n=k=32` and `c=1`, standard pass@32 is exactly one, while
the bootstrap proxy is about `.638`.  The maze SFT source separately contains
the correct combinatorial estimator, so the repository itself uses two
different conventions.

This source audit confirms the paper's conservative terminology: Countdown's
logged value must remain “VERL bootstrap best@16 coverage proxy.”  Every new
experiment should retain all per-task binary outcomes and recompute standard
pass@k independently.

## Reproduction and evidence gaps in the upstream release

- The experiment launchers do not enable validation-generation dumps, and
  `.gitignore` excludes the usual JSON/CSV/Parquet/W&B/checkpoint artifacts.
  The source therefore cannot recover our missing maze trajectories or
  Countdown task outcomes.
- The released commands do not reproduce every paper schedule: the maze
  launcher is about 39k steps versus paper discussions around 20k/default;
  SmolLM is about 5,800 steps versus roughly 1,500 reported; its `n=32`
  validation cannot produce final pass@128/pass@1024; Qwen releases two of the
  four reported evaluation benchmarks.
- Evidence is predominantly single-run.  Maze/SmolLM have no seed loop, and
  Qwen's exported seed is ineffective.
- Objectives with different advantage scales share one learning rate.  No
  estimator-specific development/confirmation protocol is released.
- There is no paper tag, complete environment lock, dataset-revision manifest,
  figure-data manifest, run registry, baseline launch matrix, or end-to-end
  reproduction command.
- ImageNet reports an eight-learning-rate sweep, but the repository does not
  contain a sweep driver, result ledger, or frozen selection rule.
- Validation can dominate small-budget comparisons: one released maze
  checkpoint uses 524,288 generated trajectories.  Training and evaluation
  cost should therefore be reported separately.

These limitations are not reasons to dismiss MaxRL's results.  They are
reasons not to inherit its artifact or metric conventions silently.

## What we should copy—and what we should not

Useful design ideas:

- exact probability and analytic pass@k in the ImageNet classification setup;
- fixed training/evaluation pools;
- explicit sampled-action/group budgets;
- throughput and wall-clock instrumentation already exposed by VERL.

Practices our release should keep stronger:

- source/runtime locks and exact data hashes;
- fresh paired seeds and declared independent units;
- arm-specific development rates when estimator scales differ;
- per-task binary outcomes, standard pass@k, raw group ledgers, optimizer-step
  counts, paid training transitions, validation sample counts, and failures;
- generated registry, figure inputs, and one-command verification.

MaxRL is a held-fixed estimator precedent, not a curriculum baseline.  Its
released suites sample fixed datasets; they do not implement `u_N` task
selection, ProCuRL, PLR, PAIRED, ACCEL, Gymnasium, or Acrobot.  Our fixed-pool
Acrobot tournament therefore answers a distinct question.

## Mac-scale extension suggested by this source: completed

We implemented the exact-probability idea as a CPU-only six-cell factorial on
`sklearn` Digits: `{practical MaxRL,RLOO} x {uniform,p(1-p),u_8}`, with exact
softmax success probabilities, a fixed 262,144-action budget per run,
estimator-specific learning-rate development, common random numbers, and 24
fresh paired confirmation blocks.  This preserves the strongest part of the
ImageNet design—noise-free `p` and analytic coverage—while directly testing
the estimator-by-sampler prediction on a Mac.

The result was an informative counterexample.  The registered interaction was
not supported (mean `+.01589`, 95% interval `[-.01686,+.04712]`, exact
`p=.350`).  MaxRL strongly favored `u_8` over `p(1-p)` (`+.20842`), but RLOO
also favored `u_8`, reversing the prediction; neither matched sampler beat
uniform.  The lesson is narrower and more defensible than the original
hypothesis: expected coefficient mass is exact estimator activity, not a
universal law of learning or curriculum optimality.  Full results and the
compact/full-replay boundary are in
[`curriculum_maxrl/digits_factorial/RESULTS.md`](curriculum_maxrl/digits_factorial/RESULTS.md).

The official maze release remains far beyond this Mac's practical scale—its
full schedule uses roughly 1.28 billion training trajectories. The separately
frozen paid-probe selection attachment supplied the more direct
environment-RL test of selection semantics and is now complete: all 12
development and 320/320 confirmation runs were valid. Its registered
`u_16-ProCuRL` fixed-paid-AUC contrast was `+.004894`, `t(79)=1.9773`,
`p=.05149`, below the `.02` SESOI and therefore unsupported.

At the frozen refresh cadence, each probed arm spent about 93.2% of its paid
transitions on probes. Ordinary-uniform fixed-paid AUC was `.65149`, compared
with `.33771` for ProCuRL-env, `.33942` for probe-sham uniform, and `.34261`
for `u_16`. This is a cadence-specific diagnosis of probe-cost domination in
an actor-only fixed-pool attachment, not evidence that full PPO ProCuRL is
inferior or that cheaper probing would have the same ordering.

The compact repository retains receipts and a content manifest; the
1,374,886,097-byte raw ledger remains external with SHA-256
`b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
The source/runtime/gate/artifact chain is internally bound, but no immutable
public pre-execution commit establishes timing. The registry generator emits
and checks exactly 562 records, including 441 Acrobot records, and is the owner
of the exact totals. All Mac experiments needed for this submission are now
complete; PLR, PAIRED, ACCEL, and full SFL are post-submission work.
