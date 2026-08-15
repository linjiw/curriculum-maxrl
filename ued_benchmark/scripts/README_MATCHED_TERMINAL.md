# Matched terminal training and evaluation drivers

The v1 files and protocol described below are frozen engineering history.
`run_matched_terminal_v4.py` is a separate DRAFT v2 tie-aware driver: it accepts
only bounded local or Slurm engineering modes, marks every receipt non-evidence,
and requires both arms to publish `plr-replay-snapshot.json` with tie blocks,
score/replay effective support, and realized replay draw/distinct/duplicate
counters. It does not authorize v1 artifacts or the v1 evaluator/assembler to
be relabeled as v4 paper evidence.

`run_matched_terminal.py` and `evaluate_matched_terminal.py` implement the
terminal-state and external-evaluation gates required by
`../UED_MATCHED_DEV_PREREG.md`. They are development infrastructure, not paper
evidence, and they accept only the two config hashes named by
`../analysis/development_protocol_v1.json`.

## Run context

Both programs require the same content-addressed `run-context.json` and its
expected SHA-256. The context binds the protocol/run/arm/seed/job identities,
campaign-manifest digest, pinned upstream commit/tree, overlay contract, source
bundle and overlay manifests, environment manifest, both driver hashes, and
the future Slurm-wrapper hash. Production contexts use one of seeds 101--105,
the deterministic protocol run ID, and the active numeric `SLURM_JOB_ID`.
Engineering contexts must use a visibly prefixed `engineering-*` run ID and
`job_id=local-test` outside Slurm.

For an array task, the canonical job identity is
`SLURM_ARRAY_JOB_ID` + `_` + `SLURM_ARRAY_TASK_ID`; for a non-array job it is
`SLURM_JOB_ID`. The runtime context and campaign submission must carry that
exact identity. A production array still has to be submitted held: all ten
task IDs, the full campaign, and all per-task contexts are frozen before the
hold is released.

Both drivers also require `--campaign-manifest` and
`--expected-campaign-manifest-sha256` in the Slurm engineering and production
lanes. Before importing `minimax`, they verify the campaign digest against the
run context; the protocol and analyzer digests; the actual training,
evaluation, and assembler driver files; the full campaign/context provenance
projection; the one-GPU hardware shape; and the exact submission selected by
the run. Only the bounded local fixture lane may omit this campaign binding.
This makes the campaign flag `frozen_before_endpoint_access=true` an enforced
runtime gate rather than a post-hoc label.

The drivers also verify the applied source clone directly. Git `HEAD` and
`HEAD^{tree}` must match the pin, the NUL-delimited worktree status must contain
exactly the declared overlay files plus `.frontierrl_overlay.json`, and every
declared overlay file and marker is hashed. Any unrelated modified, deleted,
renamed, staged, or untracked source path is rejected.

## Terminal training

The training driver owns the `ExperimentRunner.step` loop and terminates on
student PPO `n_updates`, not outer iterations. It never loads a checkpoint.
After the loop it atomically saves and reloads the first `checkpoint.pkl`, then
reconciles terminal `n_updates`, upstream `n_grad_updates`, outer cycles,
student transitions, and `n_updates * epochs * minibatches` optimizer
applications.

The training output directory contains only these source artifacts:

- `checkpoint.pkl`
- `endpoint.json`
- `logs.csv`
- `meta.json`

`endpoint.json` has the exact schema expected by the frozen v1 analyzer. The
driver-only receipt is deliberately kept out of that strict package in a
separate atomic sidecar:

- both arms: `training-receipt.json`, `SHA256SUMS`, and `COMPLETE`;
- Frontier only: `frontier-buffer-snapshot.json` is an additional manifest
  payload.

The safe Frontier snapshot contains only filled-slot identities (canonical
level SHA-256), successes, trials, analytic and mean-plugin scores, Jensen gap,
age, and normalized replay probability. Before publication, the driver checks
the reconstructed replay vector against the pinned PLR implementation and the
recomputed analytic score against every stored filled-slot score with a fixed
`2e-6` float32 tolerance. The independent float64 reconstruction keeps its
stricter `1e-10` normalization gate; a separate 500-slot float32 regression
uses `2e-6`. Level-state leaves are materialized on the host once, then indexed
for every slot hash, avoiding repeated full-buffer transfers. Downstream
analysis never needs to unpickle a model checkpoint.

Small CPU tests use repeated, typed `--engineering-override FIELD=JSON`
arguments. The whitelist and numeric bounds exclude group size, seed, score,
resume, source, and provenance changes. Every accepted override is recorded;
production refuses all of them.

## External evaluation

The evaluator consumes the terminal checkpoint, endpoint, metadata, training
sidecar receipt, protocol, run context, and applied source. Production requires
exactly one visible GPU; local engineering requires exactly one CPU, while the
separately labeled Slurm engineering lane requires exactly one GPU. It
derives seed `100000 + training_seed` and exposes no CLI for changing the three
ordered mazes or ten episodes per maze.

The evaluator atomically publishes a separate closed package containing:

- `evaluation-episodes.jsonl`: exactly 30 ordered raw episode records;
- `evaluation.csv`: the six aggregate fields expected by the analyzer;
- `evaluation-receipt.json`, `SHA256SUMS`, and `COMPLETE`.

The primary real external evaluation has a budgeted maximum of
`3 * 10 * 450 = 13,500` environment transitions. The three pinned singleton
mazes each resolve to a 450-step horizon, and the driver checks that runtime
fact before evaluation. The pinned runner scans the full horizon, so its
effective real count is also 13,500. Both are recorded separately from student
training. Periodic in-training evaluation is likewise receipted as
`floor(outer_cycles / test_interval) * 13,500` budgeted transitions and
excluded from the student-training transition counter.
`--engineering-verify-independent-aggregate` is test-only: it runs a second
13,500-transition upstream `EvalRunner.run` and requires all six aggregates to
match the raw-lane reduction within `2e-6`; it is forbidden in production.
The deterministic synthetic evaluator is also engineering-only and tests
atomic package closure without executing an environment.

The training loop caches the last terminal counter summary and performs one
device-to-host state summary per outer cycle. It does not re-read the same
state in the loop condition. This matters because the 30,000-update robust-PLR
budget is expected to require about 60,063 outer cycles (roughly 492 million
student transitions) per run.

`assemble_matched_run.py` now performs the outcome-blind schema-2 revision. It
preserves the complete training and evaluation source closures inside the
atomic analyzer package: both receipts, both renamed `SHA256SUMS`/`COMPLETE`
pairs, all 30 raw episode records, the run context, and (for Frontier) the safe
snapshot, in addition to the original run artifacts. The analyzer independently
revalidates both embedded closures and every receipt/checkpoint/context link
before it parses numeric aggregate CSV cells.

`--engineering-test-mode` creates the same closed shape but marks it
`analyzer_eligible=false`. On Slurm, this assembler is a post-terminal Phase-B
operation: Phase A publishes closed trainer/evaluator components and exits;
only after authoritative `COMPLETED 0:0` `sacct` retrieval may Phase B create
`scheduler.json`, bind terminal stdout/stderr, and assemble. It is never valid
for the running job to predict its own terminal accounting.

## Verification

From the repository root, with the pinned CPU environment available:

```bash
python3 -m unittest -v \
  ued_benchmark.tests.test_matched_terminal_drivers \
  ued_benchmark.tests.test_assemble_matched_run
```

The suite runs bounded real training for both arms, terminal checkpoint and
counter checks, source-closure rejection, safe-snapshot checks, real external
evaluation plus independent aggregate parity for both arms, and two byte-equal
synthetic evaluation packages with a negative closure test. It opens no
existing experiment endpoint.
