# E2b recent-buffer treatment-delivery failure

The prospectively frozen E2b direction test became inconclusive on 2026-08-09.
The seed-2 replay arm stopped before its first optimizer update. This file
records the failure; it does not amend `E2B_PREREG.md`.

## Completed evidence

- Seed 1 B1, B2, and Rb completed 60/60 steps and fixed checkpoints.
- Seed 1 Rb passed all delivery gates across 329 scheduled groups.
- Seed 2 B1 and B2 completed 60/60 steps and fixed checkpoints.
- Seed 2 B2 emitted all 60 immutable schedule rows.
- Seed 2 Rb generated the matched first batch, then stopped at the strict
  auxiliary-token gate before recomputing old log probabilities or updating.
- Seed 3 was not launched after the preregistered inconclusive condition.

## Seed-2 step-1 failure

B2 requested five accepted slots with group response-token counts
`[570, 665, 573, 638, 585]`, totaling 3,031. The deterministically matched Rb
batch had the same 4,716 pre-update optimizer response tokens and all five
scheduled dataset-index slots. It contained only two informative source groups.

The exact dynamic-programming source matcher selected the best admissible five
sources with replacement, totaling 2,850 tokens. The resulting auxiliary
shortfall was 181 tokens, or 5.9716%, exceeding the frozen 5.0000% ceiling.
The corresponding total optimizer-token mismatch was 3.8380%. There were no
fallback slots and no informative target slots displaced.

The strict failure occurred before the audit-return path, so the failed run has
no replay JSONL row. A separately named, one-step post-failure diagnostic raised
only the logging ceiling and reproduced the same generated batch and selection.
Its audit is at:

`/data/robotixx/curriculum-maxrl-runtime/checkpoints/e2b_s2_step1_token_diagnostic_260809/replay_accounting.jsonl`

The original error is preserved in:

`autoresearch/iterate-260809-1533/e2_logs/e2b_buffer_replay_s2_260809.log`

## Interpretation

This is a support/cold-start failure, not an endpoint result. A recent buffer
cannot contribute at step 1, and the current batch did not span the requested
auxiliary-token dose closely enough. Per `E2B_PREREG.md`, no seed-2 retry, gate
change, or seed-3 continuation is allowed for E2b. A source-reservoir design
would be a new, prospectively frozen experiment.
