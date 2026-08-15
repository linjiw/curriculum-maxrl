# Preregistration: matched Frontier versus MaxMC development gate

**Protocol:** `ued-dev-frontier-vs-maxmc-4x8-b500-v1`

**Scope:** engineering/development selection only; never paper evidence

**Endpoint status at freeze:** no UED development endpoint was opened or used

The machine-readable authority is
`analysis/development_protocol_v1.json`. The analysis program, protocol,
campaign receipt, source bundle, environment, training driver, evaluator
driver, atomic run-package assembler, and sbatch file must all be
content-addressed in the campaign receipt before endpoint access. A change to
any of them requires a new campaign freeze and, after endpoint access, a new
protocol ID.

## Question and score-isolation pair

Does exact grouped Frontier coefficient activity improve AMaze validation
solved rate over group-matched MaxMC when all non-score settings, paired
training seeds, evaluator randomness, student PPO-update count, outer-cycle
count, and student interactions match?

The two authored templates are:

- Frontier: `maze_frontier_exact_grouped_n8.json`, SHA-256
  `b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508`;
- MaxMC: `maze_maxmc_group_matched_4x8_b500.json`, SHA-256
  `6ec2083745ccc585383170f0a14f464397614a4365ba644e5c9e7e4ef422d943`.

Both use one device, one student, four distinct levels per outer cycle, eight
evaluation streams per level, 256 rollout steps, a 500-level buffer, robust
PLR, replay probability 0.5, minimum fill 0.5, the same PPO/LSTM/environment
settings, and 30,000 student PPO updates. Each PPO update runs five epochs and
one minibatch per epoch, hence 150,000 optimizer step applications. The
intervention is `ued_score`: analytic Beta-posterior expected coefficient activity versus
MaxMC. Frontier uses `N=n_eval=8`, Beta(1,1), threshold zero, strict group
matching, overlay version `frontier-activity-v3`, and contract SHA-256
`5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`.

The paired development training seeds are exactly `(101, 102, 103, 104,
105)`. These seeds and the three validation mazes are retired from subsequent
confirmatory inference. No extra seed may be added because the first five look
unfavorable or inconclusive.

## Budget and terminal checkpoint

The fixed-update endpoint is 30,000 student PPO updates. In the pinned
implementation, `VmapTrainState.apply_gradients` assigns
`n_grad_updates=n_updates+1`; therefore both upstream fields end at 30,000 and
both are retained as integrity counters, but neither counts optimizer
applications. With five PPO epochs and one minibatch per epoch, the explicit
terminal `optimizer_step_applications` counter must equal
`30,000*5*1 = 150,000`. Each outer cycle executes `4*8*256 = 8,192` student
transitions. For every paired seed, the analyzer requires equal outer-cycle
counts, optimizer-step applications, and training transitions across arms, in
addition to exactly 30,000 student PPO updates. Evaluation transitions are
excluded.

There is a necessary implementation gate: upstream `ExperimentRunner.train`
stops on student PPO updates but saves `checkpoint.pkl` only at periodic outer
cycles. Its latest periodic checkpoint is not guaranteed to be the terminal
30,000-student-PPO-update state. A hashed campaign driver must therefore save a new
checkpoint after loop termination and atomically receipt its `n_updates`,
`n_grad_updates`, `optimizer_step_applications`, `n_iters`, transition count,
and SHA-256. The optimizer count must reconcile exactly as
`n_updates*student_n_epochs*student_n_minibatches`. The stock periodic checkpoint
is inadmissible. `from_last_checkpoint` must be false; this first comparison
has no scientific resume path.

Robust PLR does not update the student on every outer cycle. With four new
levels per cycle, a 500-slot buffer, minimum fill 0.5, and replay probability
0.5, planning assumes roughly 63 warm-up cycles plus about 60,000 post-warm-up
cycles to obtain 30,000 PPO updates: approximately 60,063 outer cycles and
492,036,096 student transitions per run. This is a resource estimate, not a
stopping rule; the receipted PPO-update target remains authoritative, and
paired arms must match their observed outer-cycle and transition counts. A
bounded terminal-chain Slurm smoke and a production-shape 100-update cost run
must precede campaign scheduling.

All Frontier logged rows and the terminal receipt must report `N=8`,
`n_eval=8`, group-size match true, zero incomplete groups, and zero duplicate
new groups. A nonzero counter invalidates the development comparison rather
than being silently filtered. The terminal checkpoint, source-format
`logs.csv` and `meta.json`, stdout/stderr, exact command, terminal Slurm
accounting, and external evaluation output must be closed by one verified
schema-2 `SHA256SUMS` plus `COMPLETE` package per run. The package must also
preserve and cryptographically revalidate the trainer-sidecar and
evaluator-package `SHA256SUMS`/`COMPLETE` pairs, `training-receipt.json`,
`evaluation-receipt.json`, the exact run context, and all 30 ordered
`evaluation-episodes.jsonl` records. Frontier additionally preserves the
training-sidecar-bound safe buffer snapshot. Every source manifest, COMPLETE
marker, receipt, context, checkpoint, and copied result is independently bound
in the exact run-manifest schema and again by the outer manifest; symlinks,
extra entries, unsafe relative paths, and partial packages are inadmissible.

## Endpoint and paired analysis

Only the terminal checkpoint is evaluated. For training seed `s`, both arms
use `minimax.evaluate --seed=100000+s --n_episodes=10` on exactly, and in this
order:

1. `Maze-SixteenRooms`
2. `Maze-Labyrinth`
3. `Maze-StandardMaze`

This shares the stochastic-policy evaluation randomness within a pair.
Each of these three pinned singleton mazes resolves to a 450-step horizon.
Thus the external evaluation budget is at most `3*10*450 = 13,500`
environment transitions per terminal checkpoint; the evaluator must verify the
resolved horizons before reading an endpoint. Periodic evaluation uses the
same 13,500-transition budget per call and is accounted separately from
student training.
Periodic in-training evaluations are diagnostics and never enter selection.
The primary run-level value is the unweighted mean of the three solved rates.
The independent unit is the paired training seed. The analyzer reports all
five paired differences, their mean, a two-sided paired Student-t 95% interval
(df 4), and the exact two-sided 32-assignment sign-flip p-value as descriptive
development summaries. There is one primary contrast and no multiplicity
adjustment. Ten policy episodes reduce evaluation noise but are not treated as
independent inferential units.

Requested secondary delivery diagnostics are currently blocked, outcome
blindly, by the source log schema. `logs.csv` exposes aggregate
posterior-weighted probability/trials and current total successes/trials, but
not the joint per-filled-slot analytic score, mean-plugin score, trial count,
age, and normalized replay probability. Consequently it is impossible to
recover analytic-versus-plugin Spearman rank correlation, top-k overlap,
Jensen gap, or replay mass by posterior trial-count quartile from those
aggregates. The analyzer deliberately does not unpickle model checkpoints.
Before collecting these diagnostics, a separately hashed driver must emit a
safe manifest-bound snapshot containing filled slot identity, success/trial
counts, both scores, age, and replay probability; the top-k definition and
deterministic quartile tie rule must then be frozen in a new protocol version.
Their absence does not alter the v1 primary keep rule, and no proxy is reported.

Advance Frontier only if every provenance, completeness, delivery, endpoint,
and matched-budget gate passes, the paired mean macro solved-rate difference
is strictly positive, and no validation maze has a mean paired regression
below `-0.05`. Otherwise do not advance this exact grouped variant. A positive
development result is a selection signal, not evidence that Frontier beats
robust PLR, ACCEL, or a published number.

## Outcome access and retries

The campaign receipt must enumerate exactly one pre-endpoint submission for
each of the ten seed-arm cells. The v1 analyzer rejects missing, duplicate, or
extra cells and any non-completed Slurm state. There is no endpoint-aware
retry. An infrastructure failure requires an outcome-blind, dated amendment
and a new protocol/campaign identity before resubmission. The analyzer first
validates all ten outer closures, embedded source closures, receipts,
provenance links, and paired budgets. It next validates the schema, order,
types, and raw-to-receipt aggregation of all 300 episode records. Only after
those gates pass may it parse numeric `evaluation.csv` cells, which must match
the raw reduction within the frozen tolerance. It never emits partial metrics.

Bounded local or Slurm engineering packages use the identical atomic closure
shape but are marked `endpoint_class=bounded_engineering_test`,
`paper_evidence=false`, and `analyzer_eligible=false`; the production analyzer
rejects them. On Slurm, assembly is post-terminal: the job publishes only
closed training/evaluation components, and a separate finalizer runs after
authoritative `COMPLETED 0:0` accounting is available. No in-job process may
claim its future terminal state.

Confirmatory seeds, the twelve-environment panel, robust PLR, and ACCEL remain
outside this development gate and sealed.
