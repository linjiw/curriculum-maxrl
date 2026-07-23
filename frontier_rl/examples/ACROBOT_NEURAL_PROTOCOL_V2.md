# Neural Acrobot validation protocol — V2 development amendment

Status: frozen on 2026-07-22 after inspecting only the excluded V1 pilot seeds
`10000..10002` and before running V2 development seeds `11000..11002`, core
confirmatory seeds `0..19`, or scale confirmatory seeds `100..109`.

This document amends, but does not overwrite, the frozen V1 protocol in
[`ACROBOT_NEURAL_PROTOCOL.md`](ACROBOT_NEURAL_PROTOCOL.md). All environment,
verifier, neural actor, capacity-control, estimator, metric, multiplicity, and
artifact rules in V1 remain in force unless explicitly changed below.

## 1. Why V1 stopped

The retained V1 pilot artifact is
`acrobot_neural_pilot.json`, SHA-256
`746bce1711291cd6b6f0e2c5e2e2e0d890b19d1c5c86b9a3f2e57cdfd8f3d308`.
Its frozen V1 protocol has SHA-256
`42e6270b6d8e8e0d138b88fa1a4cb087a0a3799003b0199674a41e0a4919ba00`.

V1 selected learning rate `3e-3` by pooled final improvement, but its declared
launch gate failed (`pilot_selection.gates.all_pass=false`). The selected-rate
teacher runs ended at target-uniform mean pass rates `0.9805`, `0.9961`, and
`0.9961`, above the V1 nonsaturation boundary. After 200,000 transitions the
selected-rate pool also contained mixed and all-pass groups but no all-fail
groups. No V1 confirmatory run is authorized, and V1 is not retrospectively
reinterpreted as passing.

The quick core and scale artifacts are implementation smokes only. They are
never used for parameter selection, launch decisions, or inference.

## 2. Effect-blind V2 learning-rate rule

V2 chooses among the original candidate grid `[1e-4, 3e-4, 1e-3, 3e-3]` using
only within-arm learning and saturation, not a curriculum-minus-uniform effect.
A candidate is eligible when:

1. every pilot run at that rate is numerically valid; and
2. in **both** the shared-uniform and shared-teacher arms, at least two of three
   seeds improve target-uniform mean pass by at least `0.03` at the first
   complete-group evaluation checkpoint whose recorded transition count is at
   or above 1,000,000, while their value at that checkpoint remains strictly
   below `0.95`.

Choose the largest eligible rate. This rule selects `3e-4`. Rates `1e-3` and
`3e-3` fail the headroom rule, while `1e-4` is eligible but slower. At `3e-4`,
the V1 pilot final means were `0.6497` under uniform sampling and `0.6706` under
the teacher, so the policy was clearly learning without saturating. The
teacher-minus-uniform AUC values and signs are not inputs to this selection.

The V2 base learning rate is therefore frozen at `3e-4`.

## 3. Excluded capacity-development matrix

Before confirmatory data are touched, run the complete six-cell core matrix on
paired development seeds `11000..11002`:

```text
sampling:      uniform, exact u_16 teacher with gamma=1
architecture:  shared H64, disjoint-total 8xH8,
               disjoint-active 8xH64
hindsight:     off
learning rate: 3e-4
budget:        nominal 2,000,000 transitions per cell; complete final group,
               so the recorded actual count is at least 2,000,000
evaluation:    every 100,000 transitions, 32 episodes per task, including t=0
```

These 18 runs are exploratory development data. Their teacher-minus-uniform
contrasts, interaction contrasts, p-values, and confidence intervals are not
inputs to the launch decision and never enter confirmatory estimates.

The core confirmatory run is authorized only if all effect-blind gates pass:

1. Every run passes finite-value, binary-verifier, parameter-count, complete-
   transition, task/group/update accounting, and evaluation-state invariance
   checks. Its recorded hashes for the runner, adapter, teacher, estimators,
   interfaces, and this V2 protocol must exactly match
   `ACROBOT_NEURAL_V2_LOCK.json`. Confirmatory and scale artifacts must match
   those same development hashes; otherwise a new amendment is required.
2. In every cell, at least two of three seeds improve target-uniform mean pass
   by at least `0.03` at the first complete-group evaluation checkpoint whose
   recorded transition count is at or above 1,000,000, and remain below `0.95`
   at that checkpoint.
3. Every cell has mixed groups after 200,000 transitions and a post-warmup
   mixed-group fraction of at least `0.10`.
4. Every teacher cell has post-warmup mean total-variation distance from
   uniform greater than `0.05`.
5. Each cell observes all-fail, mixed, and all-pass groups somewhere over its
   complete two-million-transition run. All-fail groups need not persist after
   warmup because hindsight is off in the core causal comparison.
6. Observed serial throughput projects the full confirmatory matrix to no more
   than 24 hours on the development Mac.

There is deliberately no positive curriculum-effect or interaction gate. If
any gate fails, the failure is retained and the confirmatory run stops. No
fallback learning rate, architecture, threshold, or endpoint is selected from
the development treatment contrasts.

## 4. Confirmatory core freeze

If the development gates pass, run paired seeds `0..19` in the same six cells
at `3e-4` for a nominal two-million-transition budget, completing the final
group and recording the resulting actual count. Twenty pairs replace V1's twelve
because V1 measured a projected serial runtime of `1.958` hours for 12 pairs;
linear scaling projects about `3.26` hours for 20 pairs, within the frozen
24-hour limit. This is a precision improvement, not an outcome-dependent
sample-size change: it is frozen before V2 development or confirmatory results.

The primary metric and one five-contrast Holm family remain exactly V1's:

```text
CS - US
(CS - US) - (CD8 - UD8)
(CS - US) - (CD64 - UD64)
CS - CD8
CS - CD64
```

Curriculum efficacy still requires `CS-US >= 0.03`, positive, and Holm
significant. Strong transfer support still requires both interactions and both
shared-versus-disjoint contrasts to be positive and Holm significant. Exact
two-sided paired sign-flip tests and 20,000-resample paired-seed bootstrap
intervals are reported. Pilot and development seeds are excluded. Failure to
reject is called inconclusive, not equivalence.

## 5. Hindsight feasibility before the scale factorial

The V1 scale fallback to 250 updates is void because it was projected from the
rejected `3e-3` rate. At `3e-4`, the V1 teacher arms averaged 128.67 nonzero
updates per 1.004 million transitions, suggesting that 400 updates may fit the
four-million-transition cap, but this must be checked directly.

Using excluded development seeds `11000..11002`, run **scale-zero** shared-
teacher feasibility cells at learning-rate multipliers `[0.5, 1, 2]`, each to
400 nonzero live updates or the four-million-transition cap. Scale zero still
constructs every eligible relabel and records its nonmutating unscaled
auxiliary-gradient preview.

Freeze the scale update budget as follows:

1. use 400 only if every feasibility run reaches 400;
2. otherwise use 250 only if every run reaches at least 250;
3. otherwise stop the scale study.

In addition, every base-learning-rate feasibility seed must contain at least
ten eligible all-fail groups with verifier-valid, finite, nonzero auxiliary-
gradient previews. If this exposure gate fails, stop; do not switch to a lower
learning rate without another versioned amendment.

If feasible, the confirmatory scale factorial remains the V1 `3x3` grid over
learning-rate multipliers `[0.5,1,2]` and scales `[0,1,2]`, on paired seeds
`100..109`, matched by the frozen number of nonzero SGD updates. Its primary
metric remains update-indexed mean-pass AUC and its three-contrast Holm family
is unchanged. Update-indexed scale AUC is never pooled with transition-indexed
core AUC.

## 6. Claim boundary

V2 is development-informed, not “frozen before any learned Acrobot result.”
The adaptation history, failed V1 gate, hashes, seed partitions, and all pilot
outcomes remain visible. Confirmatory claims may use only the untouched seed
sets and the rules frozen here. A result from the V2 development matrix can
justify or stop the next run; it cannot support a paper performance claim.
