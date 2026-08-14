# Outcome-blind cancellation of campaign barn-icra2027-20260814-002

- Campaign ID: `barn-icra2027-20260814-002`.
- Frozen source bundle SHA-256:
  `043d73a64cd63c2bc94e7f3c8fac4a97a3ff3e6b7671775a6402d0066db27760`.
- Launch ledger SHA-256:
  `54fb6e79a833758227a30cd944ae654994d66e768c83fa2364d`.
- Arrays: primary `9366868`, N=2 `9366873`, N=4 `9366878`, and
  N=16 `9366883`; five seeds per array.
- Cancellation UTC: `2026-08-14T10:19:32Z`.
- Terminal accounting: all 20 tasks `CANCELLED`; primary tasks elapsed
  `02:12:11`, N=2 `02:11:40`, N=4 `02:11:10`, and N=16 `02:10:40`.
- Post-cancellation, an existence-only check found zero canonical `seed-N`
  blocks and zero canonical `COMPLETE` markers across the 20 ledger rows.
- Scientific endpoint, raw BARN log, result JSON, reward, success, AUC,
  trajectory, or held-out outcome inspected: **no**.

## Outcome-blind cause

Before any task became terminal, a source-closure audit found that the exact
bundled evidence seed job still published its checksum-closed directory with
`renameat2(RENAME_NOREPLACE)`.  Outcome-free engineering job `9366814` had
already established that Hopper `/scratch` rejects this directory operation
with `EINVAL`.  A fresh outcome-free probe against the campaign filesystem
and the exact staged bundle reproduced:

```json
{"available": true, "errno": 22, "error": "Invalid argument", "rc": -1}
```

The combined old/new primitive probe is retained at
`icra2027/receipts/barn_hopper_directory_publish_probe.json`, SHA-256
`2ebe0a818d82bc557d6e258a834246377373a789662c6674d46d464bb9a2c72a`.

The same unsupported operation was also present in the remote sealed-campaign
publisher.  Thus every task was structurally unable to expose its canonical
checksum-closed seed block even if computation succeeded, and the all-cell
sealer could not publish a package.  Waiting for the 36-hour limits or retrying
with the same source could not repair either failure.

## Decision and retained state

The exact four arrays were canceled using scheduler IDs only to avoid further
guaranteed-wasted CPU time.  The normalized ledger, Slurm accounting, and any
hidden incomplete work directories are retained.  Campaign `002` will not be
finalized, selected, merged, analyzed, or reused.

Before a replacement run, both directory-publication paths must use a
Hopper-supported, exclusive no-clobber claim followed by same-filesystem
atomic rename; collision and crash behavior must be covered by network-free
tests and an outcome-free Hopper probe.  The correction is a dated,
outcome-blind operational amendment.  It changes no arm, seed, split,
transition budget, checkpoint, evaluation panel, timeout, isolation,
analysis, retry-selection, or gate rule.  A new content-addressed source
bundle and a fresh campaign ID are mandatory.
