# PLR experiment decision for the current paper

Decision date: 2026-08-08. **No-go for the current paper claim.** A narrow
fixed-pool PLR selection-semantics attachment is technically possible on the
existing Acrobot/MaxRL scaffold, but it would add a critic and would not test
PLR's published unseen-level generalization result. An unqualified PLR
comparison should therefore remain a separate actor-critic generalization
study if the paper's claim expands. This is a methodological decision, not a
claim that PLR cannot run on a Mac.

Primary sources:

- [Prioritized Level Replay, ICML 2021](https://proceedings.mlr.press/v139/jiang21b.html)
- [Paper supplement](https://proceedings.mlr.press/v139/jiang21b/jiang21b-supp.pdf)
- [Official archived implementation, audited commit](https://github.com/facebookresearch/level-replay/tree/ccecf452ee3342217ece964aaf10c2831625f9b3)
- [Robust PLR / Dual Curriculum Design, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html)

## What original PLR requires

Original PLR prioritizes identifiable, resampleable training levels by an
estimate of their future learning potential. It does not require a procedural
generator or PPO: the level sampler can sit above a generic policy optimizer.
The paper finds TD/value error effective, and the released Procgen example
uses:

- a learned value function and GAE to compute per-level `value_l1` scores;
- rank-based score transformation;
- temperature `.1`;
- a staleness contribution with coefficient `.1`; and
- latest-score replacement rather than score smoothing.

The supplement reports staleness `.1` for Procgen and `.3` for MiniGrid, with
temperature `.1` in both; these are source settings, not universal defaults.
The sampler's `value_l1` name refers to the mean absolute GAE priority score.
The released PPO learner trains its critic with clipped squared value error,
not an L1 critic objective. It computes the level score with the pre-update
value function and updates sampler state before the learner update.

The procedural generator and held-out evaluation are not logical requirements
of fixed-pool PLR. Identifiable level IDs, a value/trajectory signal, level
scores, rank transform, staleness state, and a replay schedule are. Held-out
levels become necessary when reproducing PLR's published generalization claim.
Replacing value error with pass rate, `p(1-p)`, coefficient mass, or recent
improvement would be “PLR-style,” not the released PLR selection rule.

## Why the current Acrobot learner is the wrong host

The completed Acrobot tournament intentionally uses a task-blind
640-parameter policy, practical grouped MaxRL, and no critic. Its eight tasks
are nested terminal-height predicates over the same physical trajectories.
Consequently, one task-blind value function is ill-defined: an identical
physical state can have a different return under each hidden success
predicate. A PLR attachment would need a task-conditioned critic, such as
eight value heads or a threshold input, while keeping the actor task-blind.
That gives the teacher privileged task information and changes optimization,
update counts, and compute—not only task selection.

It would also leave the central PLR claim untested: the current eight
thresholds have no held-out level distribution or procedural generalization
target. A result could be driven by how the new critic represents reward
thresholds rather than by replaying levels with future learning potential.
Running such an arm as if it were a native PLR benchmark would invite a valid
reviewer objection and would not sharpen our estimator-conditioned sampler
thesis.

## Smallest defensible Acrobot attachment

If the paper prospectively expands to compare coefficient-derived selection
against temporal value-error selection, freeze a separate three-arm study:

1. `uniform + shadow critic`;
2. `u_16 + shadow critic`; and
3. `PLR(value_l1, rank, staleness) + shadow critic`.

Every arm must train a correspondingly seeded critic with the same
task-conditioned architecture, optimizer, target construction, and update
rule, using that arm's sampled trajectories. The critic must share no
parameters with the actor, so its only treatment-specific use is PLR task
selection. Keep the actor, practical MaxRL update, `N=16`, learning rate, task
thresholds, evaluation grid, and nominal two-million-transition budget fixed.

For the PLR arm, compute complete-trajectory GAE with the frozen pre-update
critic, set each level's score to mean absolute GAE, update sampler state, and
only then update the critic and actor. Freeze the released Procgen-style
`gamma=.999`, `lambda=.95`, rank transform, temperature `.1`, and staleness
coefficient `.1`. The released implementation's fill threshold `1.0` samples
all unseen levels before replay; do not describe that code path as the paper's
proportional annealing schedule.

Practical MaxRL forces 16 same-task trajectories per selection. The
source-nearest adaptation is to process those trajectories in a frozen ledger
order, increment the PLR episode clock for each completed trajectory, and let
the final completed trajectory supply the latest task score. A group-mean
score may be retained as a labeled robustness analysis, but it is not the
released latest-episode update.

Charge every training transition, complete the final group under the same
bounded-overshoot rule, and report transitions, groups, actor updates, critic
updates, elapsed time, scores, ranks, staleness, and task probabilities. Use
fresh paired seeds and a separate lock. The only permitted claim is a
fixed-pool comparison with PLR's value-error/rank/staleness selection
semantics; it is not native PLR, UED, robustness, or held-out generalization.
The launch decision must be made before reading the in-progress ProCuRL
outcome, not conditionally on that result.

## Correct later design

For an unqualified PLR/generalization comparison, freeze a separate study
with:

1. one actor-critic learner shared by uniform and PLR arms, using PPO if the
   goal is to stay close to the published experiments;
2. an identifiable fixed training pool plus a disjoint held-out pool drawn
   from the same level generator;
3. retained trajectories, returns, value predictions, GAE, per-level
   `value_l1` scores, ranks, staleness, sampling probabilities, and every
   actor/critic update;
4. charged scoring/replay/environment transitions and separate evaluation
   cost;
5. a named source setting with its environment-specific
   rank/temperature/staleness values, with any tuning confined to
   outcome-blind development seeds; and
6. fresh paired confirmation seeds and both train-pool sample efficiency and
   held-out generalization as predeclared outcomes.

A small MiniGrid or lightweight procedural maze family is a better CPU target
than thresholded Acrobot. It should compare domain randomization/uniform,
native PLR, and—only after the actor-critic estimator is analyzed—an explicitly
named coefficient-mass teacher. PAIRED and ACCEL remain out of scope until a
trainable environment generator or mutation operator is present.

## Paper consequence

For the current narrow claim, the required controls are uniform, the
`p(1-p)=u_2` score ablation, and the separately frozen paid-probe ProCuRL
selection-semantic study now in progress. That study uses `beta=20`, 20
rollout probes per task, refreshes estimates every 5,120 student transitions,
charges all probe transitions, and has 80 confirmation seeds; no outcome is
asserted here. PLR is required only for a broader assertion that coefficient
mass outperforms temporal value-error/staleness curricula. We do not make that
assertion, so PLR remains deferred for this paper.
