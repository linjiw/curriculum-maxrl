# Status

**State:** active
**Last updated:** 2026-08-14 00:39 America/New_York

## Current gates

- Official Hopper quick-start, storage, Slurm, GPU, monitoring, array-job, and
  Python-environment guidance read.
- Existing 3g.40gb MAZE-SCORE resource request matches ORC's recommended
  8-CPU/60-GB pairing.
- MAZE-SCORE's selected `frontier_un` dispatch is verified to compute exact
  `u_N`; the shifted formula belongs to the separate legacy `frontier` path.
- Exact and legacy score paths now have separate tested helpers; the v2 config
  records the score family and effective exponent.
- SSH access is working. The live account is `lwang44`, account `xiao`, with
  `gpu`, `interactive`, and `normal` QOS. The lock-addressed MAZE environment
  passes `pip check` and reports Python 3.10.20, torch 2.6.0+cu124, NumPy 2.2.6,
  and CUDA 12.4.
- Engineering job 9361275 was canceled while still pending after a clean-layout
  reproduction proved its partial source copy cannot import `estimators`.
- Historical endpoint/RNG behavior is preserved under `legacy_v1`; the new
  `maze_score_v2` path emits baseline plus exactly 25--250 once and separates
  SFT, task, rollout, teacher, and evaluation streams.
- The strict analyzer, 20 protocol tests, and Hopper wrapper mock suite pass.
- CPU workflow job 9366532 completed and was fetched with matching hashes.
- GPU import job 9366547 completed in 34 seconds on `gpu020`; its CUDA/runtime,
  source, environment, submission, and result receipts are fetched and verified.
- Full-arm cost job 9366552 is queued for priority on a `3g.40gb` slice. Its
  seed 99 is engineering-only and its scientific endpoint will not be inspected.
- MAZE-SCORE remains DRAFT and evidence-bearing launch is forbidden.
- The remaining MAZE gate is full-arm cost completion, outcome-blind sample
  freeze, clean commit/evidence bundle, and a complete campaign receipt.

## Decision log

| # | Decision | Verify | State |
|---|---|---|---|
| 1 | Keep official ORC resource/storage rules as the workflow contract | Cross-checked seven ORC guide pages | kept |
| 2 | Retract the over-broad formula blocker; keep MAZE-SCORE's exact `frontier_un` path | Audited sbatch key, `TEACHERS` dispatch, and both class formulas | kept |
| 3 | Cancel pending job 9361275 before allocation | Missing import and stdout-parent failures reproduced; `sacct` now records `CANCELLED` with zero elapsed | kept |
| 4 | Preserve historical loop semantics as legacy; create an explicit v2 evidence protocol | Step/read-path and RNG audit | kept |
| 5 | Use CPU I/O smoke before another GPU request | Exercises submit, scratch, accounting, fetch, and hashes without research compute | kept |
| 6 | Use explicit `maze_score_v2` rather than changing historical semantics | 20 protocol/analyzer tests pass | kept |
| 7 | Require a clean/FROZEN bundle for evidence | Stager and evidence sbatch fail closed on DRAFT/dirty state | kept |
| 8 | Advance through the smallest GPU first | Import-only job 9366547 completed and fetched before full-arm submission | kept |
| 9 | Queue one non-evidence full-arm cost profile, seed 99 | Job 9366552; scheduler state only until terminal | active |
