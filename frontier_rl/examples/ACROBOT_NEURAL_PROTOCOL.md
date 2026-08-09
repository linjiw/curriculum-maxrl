# Neural Acrobot validation protocol

Status: frozen before any learned-policy Acrobot result is inspected. The fixed
task thresholds were selected only from a random-policy calibration. Pilot data
are development data and are never described as confirmatory evidence.

## Question and causal design

The experiment asks whether Curriculum MaxRL helps because a frontier-focused
teacher directs updates toward useful shared behavior, rather than because it
changes parameter count, per-task active capacity, hindsight compute, or the
optimizer step size.

The independent environment is Gymnasium `Acrobot-v1` with its standard
three-action interface and 500-step time limit. From every post-transition
observation we compute the tip height

\[
h(o)=-o_0-(o_0o_2-o_1o_3).
\]

The eight nested tasks are strict crossings `h > threshold` for the fixed
thresholds

```text
[-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0].
```

The final threshold agrees with the native Acrobot termination predicate. An
episode for a requested task ends at its first verified crossing or at native
termination/truncation. The binary verifier drives training and curriculum
evidence; native return, native success, and time to native goal are secondary
evaluation outcomes.

Every condition uses a one-hidden-layer tanh categorical actor trained by plain
SGD. The score for a trajectory is the **sum** of its per-transition
`grad log pi(a|o)` terms evaluated at one frozen pre-group parameter vector.
There is no trajectory-length averaging, group averaging beyond the estimator's
own weights, clipping, momentum, or adaptive optimizer normalization. The
actor receives normalized physical observations but no task identifier. It is
therefore genuinely shared across the nested verifier thresholds.

## Core transition-matched experiment

The core experiment has a 2-by-3 design. Sampling is either uniform over the
eight tasks or the exact practical-MaxRL coefficient-mass teacher

\[
u_N(p)=1-(1-p)^N-p,
\]

with `N=16`, exponent `gamma=1`, discounted Beta evidence, decay `0.7`, and a
`0.1` uniform floor. Architecture is one of:

1. shared `H=64`: 640 total parameters and 640 active parameters;
2. disjoint-total `8 x H=8`: 640 total and 80 active for a requested task;
3. disjoint-active `8 x H=64`: 5,120 total and 640 active for a requested task.

The disjoint actors are capacity controls, not proposed methods. Each task uses
its own policy slot. Initialization is action-uniform in every condition: hidden
input weights are seeded, hidden biases and output weights are zero, and there
is no output bias. Parameter counts and active parameter counts are recorded.

Core conditions receive the same nominal transition budget. A rollout group is
never cut to hit the budget, so the final complete group can overshoot; all
metrics use the **actual** transition count. Hindsight is off in all six core
conditions. `gamma=4` is permitted only as a separately labelled secondary
follow-up after the `gamma=1` gate below; it is not substituted into the core
family.

The primary outcome is target-uniform mean-pass learning-curve AUC, computed by
normalized trapezoidal integration over actual transition checkpoints and
including the step-zero evaluation. Write `CS` and `US` for curriculum-teacher
and uniform shared-H64 conditions, `CD8` and `UD8` for their disjoint-total-H8
controls, and `CD64` and `UD64` for their disjoint-active-H64 controls. The one
predeclared five-contrast family is

```text
CS - US
(CS - US) - (CD8 - UD8)
(CS - US) - (CD64 - UD64)
CS - CD8
CS - CD64
```

All five receive Holm correction as one family at familywise alpha `0.05`.
Curriculum efficacy requires the first contrast to be positive, Holm
significant, and at least `+0.03` AUC. Strong transfer support requires all four
transfer/control contrasts to be positive and Holm significant. If only one
capacity control is supported, the conclusion is explicitly capacity-qualified.
Other within-cell comparisons are descriptive.

Secondary outcomes are final target-uniform mean pass rate, final hardest-task
and native success rates, native episodic return, censored time to native goal,
environment transitions, update counts, policy entropy, and gradient/update
norms. They are descriptive unless a later protocol explicitly promotes them.

The intended confirmatory core set is paired seeds `0..11` (twelve seeds).
Exact two-sided paired sign-flip tests and paired-seed bootstrap intervals with
20,000 resamples are used for all five contrasts. Evaluation uses fixed per-seed
common random numbers and must restore all training RNG state. The provisional
confirmatory budget is two million training transitions per condition; the
pilot freezes this budget and evaluation schedule before confirmatory data are
inspected.

## Optimizer-matched hindsight scale by learning rate

Hindsight is studied separately on the shared `H=64` actor under the exact
`gamma=1` teacher. The economical factorial grid is:

```text
learning-rate multiplier: [0.5, 1.0, 2.0]
hindsight scale:          [0.0, 1.0, 2.0]
```

Multipliers are applied to the base learning rate frozen by the pilot. Scale
zero is the no-hindsight baseline at that learning rate. Only an originally
all-fail (`K=0`) requested group is eligible for hindsight. The adapter chooses
the hardest lower threshold having `0 < K_b < N`, recomputes rewards using the
same strict verifier, truncates each credited success at its first crossing,
and retains the complete trace for credited failures. Because the actor is
task-agnostic, changing the verifier threshold does not require rewriting an
actor goal input. This validates verifier and conditioning compatibility; it
does **not** make an unbiasedness claim for the selected hindsight data law.
Hindsight is disabled for disjoint actors because the behavior and credited
policy slots would differ.

Every factorial condition is matched to 400 **nonzero SGD updates**, counting a
live requested-task update or an applied nonzero relabeled update. If the pilot
no-hindsight condition cannot reach 400 updates within the declared runtime,
the budget is reduced once to 250 before the scale study; if 250 is also not
feasible, the scale study stops rather than being retuned again. A four-million
transition safety cap prevents an unbounded run, and evaluation occurs every 50
nonzero updates. The final complete group can overshoot the cap. Reaching the
update budget, transitions-to-budget, and any cap censoring are recorded. At
scale zero, every eligible all-fail group is still relabeled and verifier-
checked, and its unscaled auxiliary-gradient diagnostics are computed without
applying or counting that auxiliary update. Thus the core experiment is
transition-matched, whereas this study is optimizer-update-matched; their AUCs
are not pooled.

The scale-study primary metric is target-uniform mean-pass AUC indexed by
nonzero optimizer updates. The one three-contrast Holm family is: scale 1 minus
scale 0 at the base learning rate; scale 2 minus scale 1 at the base learning
rate; and the iso-auxiliary-step difference-in-differences
`[(scale2, halfLR) - (scale0, halfLR)] - [(scale1, baseLR) - (scale0, baseLR)]`.
Final performance, transitions-to-budget, relabel count, gradient/update norms,
and native metrics are secondary. The intended confirmatory scale set is paired
seeds `100..109` (ten seeds).

## Development pilot and freeze rule

Pilot seeds are `10000..10002`, disjoint from both confirmatory seed sets.
Candidate base learning rates are `[1e-4, 3e-4, 1e-3, 3e-3]`. Each candidate is
run on both the
shared uniform and shared exact-teacher conditions at a shorter, identical
transition budget. Selection first rejects a rate if any parameter, probability,
gradient norm, update norm, or evaluation metric is non-finite. Among remaining
rates, it ranks the average pooled final improvement from step zero across the
two sampling rules; ties within `0.01` choose the smaller rate. This pooled rule
does not select the rate that maximizes the teacher-minus-uniform contrast.

The warmup boundary for diagnostics is 200,000 transitions. Before launching
the full core matrix, all of the following gates must pass:

1. every rollout/update/accounting identity holds, evaluation preserves the
   training RNG state (cadence invariance), all numerical values are finite,
   and verifier/relabel checks pass;
2. after warmup, the pooled selected-rate shared runs contain all three group
   regimes (`K=0`, `0<K<N`, `K=N`), at least 10% of groups are mixed, and the
   teacher conditions' mean total-variation distance from uniform exceeds
   `0.05`;
3. at least two of the three selected-rate curriculum-shared seed runs improve
   target-uniform mean pass by at least `0.03` by one million transitions while
   remaining below the saturation threshold `0.95`;
4. the mean paired shared `gamma=1` teacher-minus-uniform AUC direction is
   positive; and
5. observed serial throughput projects the full 12-seed, six-condition core
   matrix to no more than 24 hours on the development Mac.

These gates are only resource-allocation rules. Passing them is not evidence for the
confirmatory claim. If a gate fails, the result is retained and labelled, and
the confirmatory run is not silently retuned. A changed model, task definition,
or learning-rate grid requires a versioned new protocol.

## Exclusions, reproducibility, and reporting

There are no outcome-based seed exclusions. A run is invalid only for a logged
implementation failure, non-finite arithmetic, corrupted artifact, or failure
to execute the declared environment. The original record is retained. A rerun
uses the same seed only after the code/version change is recorded; cap-censored
scale runs are valid observations, not exclusions.

Every JSON artifact stores exact cases and budgets, raw per-seed curves and
counters, all actual transition and update coordinates, Python/NumPy/Gymnasium
versions, git commit and dirty-worktree status, and SHA-256 hashes of the runner,
adapter, teacher, estimator, and this protocol. Existing artifacts are not
overwritten unless the caller explicitly opts in. Quick runs and all pilot
artifacts are labelled exploratory.
