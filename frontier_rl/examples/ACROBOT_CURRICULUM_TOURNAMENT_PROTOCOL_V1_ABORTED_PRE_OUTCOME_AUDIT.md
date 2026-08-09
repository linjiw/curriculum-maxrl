# Source-locked Acrobot curriculum tournament

Status: **frozen before development or confirmatory execution**. Quick smoke
runs are explicitly excluded from evidence.

## Question and claim boundary

This CPU tournament compares three task samplers while holding the Acrobot
learner fixed: target-uniform sampling, Thompson-Beta learnability sampling
with utility `p(1-p)`, and the exact MaxRL frontier utility
`u16(p)=1-(1-p)^16-p`. The primary estimand is the paired difference in
target-uniform learning efficiency, `u16 - p(1-p)`. The experiment does not
test hindsight, architecture, capacity, transfer causality, PLR, ALP, PAIRED,
or ACCEL.

## Frozen learner and environment

- Official Gymnasium `Acrobot-v1`, including its 500-step limit.
- Ordered thresholds `[-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0]` and the
  existing strict post-transition tip-height verifier.
- One task-blind shared categorical actor with a tanh hidden layer of width
  64, 640 trainable parameters, learning rate `3e-4`, and plain SGD ascent.
- Practical dropped-group MaxRL weights
  `1{K>0}(r_i/K - 1/16)`, with no hindsight or relabel-derived update.
- Complete groups of `N=16` rollouts. A group is never cut at a transition
  boundary.
- Both adaptive arms use the same per-task discounted Beta tracker, Thompson
  draws, decay `0.7`, uniform floor `0.1`, and exponent `gamma=1`. They differ
  only in utility. The uniform arm samples every threshold with probability
  `1/8`.

The new wrapper imports the existing frozen neural Acrobot engine. It replaces
that module's teacher factory only inside a sequential, process-local context
and restores it in `finally`. Full task probabilities and posterior means are
added to every retained raw group record. Every evaluation checkpoint is also
retained as an explicit record.

## Development and launch gate

The registered development run uses all three arms, paired seeds
`19100..19102`, 200,000 actual transitions per run, evaluations at
initialization and after the first complete group crossing every 50,000
transitions, and 16 evaluation episodes per task. Development is used only for
these outcome-blind launch gates:

1. every run passes numerical, accounting, parameter-count, verifier, and
   evaluation-state checks;
2. every task is visited at least once when pooled across arms and seeds;
3. each adaptive arm exhibits a nonuniform requested-task distribution;
4. pooled groups include dead, mixed, and all-pass regimes; and
5. native-success checkpoint values exhibit variation.

No arm contrast, effect direction, p-value, confidence interval, or minimum
effect is a launch gate. The independent analyzer writes the gate artifact.
A full confirmation must name a passing gate artifact produced under the same
source lock. A deliberate alternative is an explicit, reason-bearing
`--v3-adequacy-waiver`; the raw artifact records that waiver and no implicit
waiver exists.

`--quick` is a mechanical smoke test only: one paired seed (`19100`), 8,000
transitions, a 4,000-transition evaluation interval, and two evaluation
episodes per task. It never uses a sealed seed and cannot create a launch
gate.

## Sealed confirmation

Run exactly 60 complete runs: three arms by exactly 20 paired seeds,
`19000..19019`. A repository seed-ledger audit found no prior training use of
this block. Each run receives a nominal budget of 2,000,000 actual environment
transitions; the final group is completed and the terminal coordinate may
overshoot by at most one 16-by-500-step group.

Evaluate at initialization and after the first completed group crossing every
100,000-transition boundary, with 32 fresh episodes per target. Within a seed,
all arms and checkpoints use the fixed evaluation seed root and therefore the
same evaluation common random numbers. Evaluation must preserve training
parameters, counters, and RNG state. Every run, including an invalid run, is
retained; there is no seed replacement, interim analysis, outcome-dependent
stopping, or tuning.

Before development or any sealed seed is executed, the runner must match the
exact source lock. Before every sealed run and after the final run it rechecks
the lock. The pinned runtime is CPython `3.12.13`, NumPy `2.5.1`, and
Gymnasium `1.3.0`. Any source/runtime/schedule mismatch blocks execution.

## Registered analysis

For seed `s` and arm `m`, let `A(s,m)` be normalized trapezoidal area under the
target-uniform mean pass-rate curve, against actual environment transitions,
from initialization through that run's complete terminal group. The sole
primary contrast is

`d_s = A(s,u16) - A(s,p(1-p))`.

The independent analyzer reports the mean of the 20 paired differences, a
20,000-resample paired-seed percentile bootstrap 95% interval, and the exact
two-sided paired sign-flip randomization p-value over all `2^20` assignments.
No multiplicity correction is applied to this one primary test.

Two secondary target-uniform AUC tests compare `p(1-p) - uniform` and
`u16 - uniform`; their exact two-sided sign-flip p-values form one Holm
step-down family at familywise alpha 0.05. They cannot rescue the primary.

Secondary descriptive endpoints include native-success AUC, native-return
AUC, final native success and return, realized practical-MaxRL absolute
coefficient mass per sampled group and per million transitions, and the
nonzero-mass group fraction. Their paired estimates and bootstrap intervals
are descriptive; they carry no confirmatory decision.

If any registered run is missing or invalid, or any raw ledger, checkpoint,
source-lock, runtime, seed, schedule, or saved derived metric fails independent
recomputation, confirmatory inference is not performed. The raw execution
artifact remains separate from the independent analysis report.
