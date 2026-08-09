# Acrobot ProCuRL selection-semantic experiment

Status: **implementation candidate; no quick, development, confirmation, or
canonical-lock creation may occur until the hardened implementation passes a
new independent pre-seal review.**  After approval, quick mode remains
engineering-only and cannot support a scientific claim.

The first sealed engineering/development wave is invalid and was aborted
before any development gate or arm contrast.  Its exact lock, quick artifact,
quick analysis, and development raw artifact are preserved under
`INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/`; the outcome-blind incident
record documents 73 entropy-aggregate replay mismatches with maximum absolute
difference `7.275957614183426e-12`.  Seeds 21100--21102 and 21200 are burned.
This replacement registration uses fresh development and quick seeds below;
the untouched confirmation block remains unchanged.

This experiment asks whether arbitrary-`N` MaxRL utility improves paid-compute
allocation over a source-faithful ProCuRL environment-selection rule when both
are attached to exactly the same small Acrobot learner.

## Frozen scientific question

The primary estimand is the paired-seed difference

`continuous-range-matched softmax u16 - ProCuRL p(1-p)`

in target-uniform mean success AUC over the first 2,000,000 **paid
transitions**.  Paid transitions include student and probe transitions and
exclude evaluation transitions.  The nominal-budget truncation is primary;
the complete atomic overshoot is retained and analyzed as sensitivity.

## Common learner and environment pool

- Gymnasium `Acrobot-v1`, official 500-step limit.
- Eight nested strict endpoint-height tasks at thresholds
  `[-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0]`.
- Shared task-blind one-hidden-layer H=64 categorical actor, 640 trainable
  parameters.
- Plain SGD ascent, learning rate `3e-4`.
- Groups of `N=16` student rollouts.
- Practical dropped-group MaxRL weights.  All-fail and all-pass student groups
  receive no update.  No hindsight relabeling.

The existing Acrobot engine and dependencies are byte-unchanged and included
in this experiment's independent lock.  This protocol does not modify the
already sealed Acrobot tournament.

## Four paired arms

All arms share the actor, optimizer, student rollout, evaluation, and logical
seed mapping.  Within a logical seed, common RNG roots are used wherever the
control flows permit it.

1. `procurl_env_b20_f5120`: initial 20-episode probe sweep per task, then one
   sweep for every crossed 5,120-student-transition boundary.  A sweep replaces
   the vector with `p_hat_j = successes_j / 20`; selection is
   `softmax(20 * p_hat * (1-p_hat))`.
2. `probe_sham_uniform_f5120`: identical probe episodes, cadence, and paid
   accounting, but estimates are discarded and selection is exactly `1/8`.
3. `ordinary_uniform`: no probes and exact `1/8` selection.
4. `u16_probe_range_matched_f5120`: identical probes and cadence; selection is
   `softmax(beta16 * (1-(1-p_hat)^16-p_hat))`, where
   `beta16 = 6.416133525771289`.  This is a **continuous-domain range match**:
   the unconstrained u16 maximum receives logit 5, matching the maximum
   ProCuRL logit `20 * 0.25 = 5`.  The empirical estimates lie on the 0.05
   lattice; there the largest attainable u16 logit is approximately
   `4.97730861318145`, not exactly 5.

Softmax is evaluated after subtracting the largest logit.  There is no prior,
decay, memory, floor, gate, or further temperature.  The latest complete sweep
fully replaces the previous estimates.

## Probe and refresh semantics

A probe sweep executes 20 independent stochastic episodes on each of the eight
tasks.  Each episode uses a fresh evaluation-only reset seed and action stream,
stops at the first strict crossing of that task's threshold (or the official
terminal/time limit), and contributes every simulator transition to paid
compute.  The actor parameters, actor training-action RNG, training-environment
reset RNG, optimizer counters, and student counters must be bitwise unchanged.
Probe action streams are isolated deterministic generators, rather than the
upstream implementation's global Python `random` stream.  Each sweep uses a
fresh Gymnasium environment that is closed in `finally`; evaluation likewise
uses a separate fresh environment and isolated action streams.  These choices
preserve the source selection semantics while making non-mutation auditable.

Probe and evaluation reset/action seeds use collision-free coordinate
encodings over `(logical seed, sweep, task, episode)` or `(logical seed,
episode)`.  They are not hashes reduced modulo a finite seed space.  Student
reset seeds still come from the separately registered environment-reset RNG.

The probed arms run a complete initial sweep before the first student-task
selection.  Thereafter, after collecting one complete 16-rollout student group,
let `t_before` and `t_after` be the student-transition counts around that group.
Before applying its update, execute exactly

`floor(t_after/5120) - floor(t_before/5120)`

complete sweeps with the pre-update actor.  Double crossings produce two
separate sweeps.  Apply the student update only after all required sweeps.  The
next task selection uses the newest complete estimate.

## Budgets, seeds, and evaluation

| mode | paired logical seeds | paid budget | evaluation trajectories |
|---|---:|---:|---:|
| confirmation | 21000--21079 (80) | 2,000,000 | 32 |
| development | 21300--21302 (3) | 400,000 | 32 |
| quick engineering | 21400 (1) | 100,000 | 2 |

The only supported execution environment is CPython 3.12.13 with NumPy 2.5.1
and Gymnasium 1.3.0.  Confirmation contains 320 runs and must not be launched
casually.  A run
starts another complete student group only while paid transitions are below its
nominal budget.  It retains the final complete group, every sweep required by
that group, its update, and the resulting paid-budget overshoot.

Scientific evaluation uses 32 shared, full-horizon trajectories with fixed
episode-specific reset and action streams across checkpoints and arms.
Success at all eight thresholds is derived from the same trajectory maxima;
the primary target is their unweighted mean.  Raw maximum heights and success
at the seven adjacent-threshold midpoints are retained descriptively.
Evaluation transitions are never charged.

Policy-entropy aggregation is also frozen: retain the 32 episode entropy sums
in ledger order and aggregate them with Python 3.12's built-in `sum(list)`.
The independent analyzer must apply that same ordered built-in sum, rather
than iterative `+=`, and compare it to the stored aggregate at absolute
tolerance `1e-12`.

Portable reanalysis is fail-closed on the same numerical semantics.  Before
importing or executing any locked analyzer byte, the verifier requires live
CPython 3.12.13, NumPy 2.5.1, and Gymnasium 1.3.0 and checks a frozen
32-element entropy vector.  Its Python 3.12 built-in-sum result must be
`0x1.12a4ae5d8b0d5p+14`; the known naive/older-interpreter result
`0x1.12a4ae5d8b0d3p+14` is rejected.  “Portable” therefore means that bundle
integrity can be checked independently in the pinned runtime, not that frozen
reanalysis is permitted under numerically different Python versions.

Evaluate at initialization, at the terminal state, and around every probe
sweep.  Immediately before each sweep, evaluate once; copy that unchanged
score to the post-sweep paid coordinate to encode the probe's no-learning
plateau.  Also evaluate after any atomic training step that crosses a regular
100,000-paid-transition checkpoint.  Repeated paid coordinates are retained to
represent zero-width post-update changes.

## Primary and secondary analysis

For each arm and seed, integrate target-uniform mean success by the trapezoid
rule on the nondecreasing recorded paid coordinates, linearly truncate the
crossing segment at the nominal budget, and divide by that budget.  Duplicate
coordinates remain in ledger order: their zero-width segment contributes zero
area, and the last duplicate starts the next positive-width segment.  The
primary paired contrast is `u16 - ProCuRL`.

- Primary test: two-sided paired t-test over 80 logical seeds.
- Support interval: 20,000-resample paired-seed percentile bootstrap, seed
  `31000`, quantiles 0.025/0.975 using NumPy's `linear` method.
- Decision: supported iff mean contrast is at least `+0.02` and two-sided
  `p <= 0.05`.
- Robustness: 1,000,000-draw deterministic Monte Carlo paired sign-flip test,
  seed `31001`, two-sided absolute statistic, and plus-one correction; it is
  reported but not substituted for the primary test.

The Holm-corrected secondary family is:

1. ProCuRL minus probe-sham;
2. u16 minus probe-sham;
3. ProCuRL minus ordinary uniform;
4. u16 minus ordinary uniform; and
5. probe-sham minus ordinary uniform.

Holm is the fixed step-down procedure at familywise alpha 0.05.  Secondary
paired bootstrap seeds are `31100, 31101, 31102, 31103, 31104` in exactly the
family order above, with the same resample count and linear quantiles.

Also report overshoot-inclusive AUC, student-transition AUC, final success,
student/probe transition fractions, sweep counts, optimizer updates, and the
following selection diagnostics: mean entropy, mean total variation from
uniform, mean maximum task probability, per-task mean assigned probability,
realized task fractions and their total variation from uniform, paid fractions,
and overshoot.  They are secondary or descriptive.

## Strict outcome-blind development gate

Development may start only after independent review, canonical lock creation,
and lock verification.  Confirmation additionally requires a fresh hashed
development artifact and independently recomputed passing gate.  The gate may
inspect no arm contrast, direction, confidence interval, p-value, or minimum
effect.  Every item below must pass:

1. every run is complete, numeric, finite, source-verified, parameter-valid,
   evaluation-RNG invariant, and exactly ledger-accounted;
2. every sweep has exactly 20 episodes per task and at most 80,000 transitions;
3. every `p_hat` is an integer multiple of 0.05;
4. probed arms have one initial sweep and exactly one later sweep per crossed
   5,120 boundary, including separate sweeps for double crossings;
5. probe sweeps change no actor parameters, optimizer count, or training RNG;
6. paid transitions recompute exactly as student plus probe transitions;
7. sham and ordinary selection are exactly uniform and ordinary has zero probes;
8. adaptive selection recomputes from the newest estimates and is non-uniform
   at least once across the pooled development artifact;
9. each probed run reaches at least 20,000 student transitions and at least one
   nonzero optimizer update;
10. dead, mixed, and all-pass student regimes all occur pooled across runs; and
11. native evaluation success varies pooled across checkpoints.

If this source cadence fails the development gate, the design is declared
inadequate; it is not silently relaxed.  A scaled cadence such as 80,000 student
transitions would require a new protocol, source lock, development seeds, and
confirmation seeds.

## Provenance limitation

The source audit and non-equivalence statement are in
`PROCURL_PRIMARY_SOURCE_PROVENANCE.md`.  There is no immutable public
pre-execution commit for this local registration; that limitation must be
disclosed.
