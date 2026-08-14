# Status

**State:** active, frozen BARN evidence running
**Last updated:** 2026-08-14 04:09 America/New_York

## Bottom line

No BARN scientific evidence has run or been inspected. The unrelated active
MAZE/Hopper loop was classified out of scope and left scheduler-only. Work is
now aligned to the governing ICRA goal. The CPU-only Hopper dataset preparation
and outcome-blind timing smoke passed, and the complete protocol package was
frozen and committed at `23dacb88cf7b1f46dddf9d2453dbd7e0bcbbbf33`.
The amended source closure was committed at
`55d46ccb04ceef2707c382293248ad50087cbb58` and staged at SHA-256
`043d73a64cd63c2bc94e7f3c8fac4a97a3ff3e6b7671775a6402d0066db27760`.
The exact four-cell, 20-task campaign is now running. No endpoint or raw BARN
job log has been opened.

## Passed gates

- Official 300-course archive, manifest, and all 900 bound world/path/grid
  asset hashes verified.
- Required seed-20270811 split materialized: 240 train / 60 held out, exactly
  24/6 per each of 10 difficulty strata, no overlap.
- Hopper CPU container fingerprint job 9366688 completed successfully; ROS 2
  Humble, Gazebo Classic 11.10.2, Python, dpkg, pip, kernel, and headless boot
  receipts are checksum-verified locally.
- Real BARN campaign runner now supports frozen transition budgets, isolated
  evaluation, train-only smokes, per-episode accounting, balanced arm order,
  cell-specific ROS/Gazebo isolation, exact protocol enforcement, seven hash
  bindings, and atomic no-clobber publication.
- The engineering smoke chooses `barn-299` before package verification and
  loading. Its package verifier does not read held-out assets; only the runner
  hashes and loads the selected train course.
- Compute-node dataset preparation, train-only timing smoke, immutable
  engineering/evidence source staging, source-bound four-cell submission,
  terminal-state ledger finalization, CPU-only sealed postprocessing, and
  their local fail-closed tests are implemented.
- Final campaign postprocessing binds the finalized ledger SHA, performs all
  four blind selections and merges plus both frozen analyses on Hopper, and
  fetches only one checksum-closed all-cell package.
- Dataset preparation job 9366817 completed successfully after two bounded
  engineering failures: job 9366805 exposed the missing host `/usr/bin/time`,
  and job 9366814 exposed unsupported directory
  `renameat2(RENAME_NOREPLACE)` semantics on Hopper scratch. The retained
  preparation receipt SHA-256 is
  `216408ddfb6ef95c6d7cc912608aac0428240d09a562f20b03069408b1a9d76f`.
- Train-only smoke job 9366819 exposed an overwritten ROS `PYTHONPATH`; the
  corrected job 9366821 passed but did not yet report enough resource counters
  for a defensible projection. The repeated job 9366831 passed with a
  resource-only receipt: 50,570 training simulator steps in 330.497 seconds,
  16 training episodes, and 62.233 seconds for two train-course evaluations.
  Its retained receipt SHA-256 is
  `d9d251c819bbf602dae6c829e3c6755b514639f2fa1c3c9f83cd5b13d21c8738`.
- Both successful receipts state CPU-only execution, no held-out course reads,
  no paper endpoint, and no retained internal metric artifact. No success,
  reward, AUC, trajectory, or held-out result was used for the timing decision.

## Freeze result and active launch gate

- `prereg_icra.md` and `barn_protocol.json` are now exactly `FROZEN` in commit
  `23dacb88cf7b1f46dddf9d2453dbd7e0bcbbbf33`. The protocol SHA-256 is
  `36007d8c979b2dacccd595a43a4620dca7be24c1f50ef91a8a9ee4e869202cb2`;
  the preregistration SHA-256 is
  `975e3cced69807c86569acd167f5292d5cf8d8e1872b2f8b9a5f876cc464ab77`.
- The resource-only projection is 4.927 hours per arm: 1.815 hours for one
  million training transitions plus 3.112 hours for six 60-course evaluation
  checkpoints. A four-arm primary cell projects to 19.708 hours nominal and
  23.650 hours with 20% padding. Because that leaves only 0.350 hours under a
  24-hour allocation, the freeze recommendation is a 36-hour Slurm time limit;
  this changes no scientific transition, checkpoint, course, or seed budget.
- The frozen gate passed: 93 BARN contract tests, 17 core tests, four
  fail-closed shell workflow mocks, shell syntax, Python compilation, and
  whitespace checks all passed. The exact 39-file source closure is HEAD-clean.
- Evidence launch remains held only until a fresh evidence-mode stage creates
  the source bundle and SHA bound to this commit. Engineering bundle hashes
  must not be reused.
- The first source-bound primary submission created held array 9366866, but a
  transient remote ledger-install/acknowledgement failure left the canonical
  ledger absent. The array stayed `PENDING|JobHeldUser` and was canceled before
  allocation. Its incomplete campaign, pending ledger, and scheduler record
  are retained. An outcome-blind preregistration amendment records the event;
  the preregistration SHA-256 is now
  `f9dcc5f56ef890a7a32fd14244fd7073f50f27f7ad4ad5dea20efcb347f01864`.
- The submitter now reuses an exact hash-matched staging upload on resume and
  replaces only a partial transaction-owned upload. Its new regression
  interrupts the install after upload and proves same-job recovery. All four
  workflow mocks and all 93 BARN tests pass after the amendment.
- Campaign `barn-icra2027-20260814-002` has one five-seed attempt in each exact
  cell: primary array 9366868, N=2 array 9366873, N=4 array 9366878, and N=16
  array 9366883. The normalized 20-row ledger SHA-256 is
  `54fb6e79a833758227a30cd944ae654994d66e768c83aeace63725f83fa2364d`.
  At 04:09 EDT all 20 tasks were `RUNNING` under the frozen 36-hour limit.

## Next BARN actions

1. Monitor scheduler metadata only. Do not open raw logs or endpoints, fetch
   seed artifacts, select partial results, or analyze locally.
2. If any array task fails, retain the attempt and resubmit its entire
   five-seed cell under a new attempt ID with the identical frozen source.
3. After every recorded task is terminal and any scheduler-visible failure has
   been retried as a whole cell, run
   `finalize_barn_ledger.sh CAMPAIGN_ID SOURCE_BUNDLE SOURCE_SHA256`. Pass the
   printed `sha256=` value to
   `finalize_barn_campaign.sh CAMPAIGN_ID SOURCE_BUNDLE SOURCE_SHA256
   FINALIZED_LEDGER_SHA256` and accept only its sealed all-cell package.

## Scheduler state

All six BARN engineering jobs listed above are terminal. Held prelaunch job
9366866 was canceled before allocation. Evidence arrays 9366868, 9366873,
9366878, and 9366883 are running. The unrelated MAZE scheduler work remains
out of scope and was neither expanded nor inspected for scientific endpoints.
