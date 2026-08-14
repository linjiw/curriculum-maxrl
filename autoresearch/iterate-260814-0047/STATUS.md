# Status

**State:** active, amended frozen BARN evidence running
**Last updated:** 2026-08-14 06:37 America/New_York

## Bottom line

The governing ICRA goal is active and aligned exclusively to real BARN
navigation evidence.  The scientific protocol remains frozen and byte-identical
at SHA-256
`36007d8c979b2dacccd595a43a4620dca7be24c1f50ef91a8a9ee4e869202cb2`.
No scientific endpoint, raw BARN log, partial result, selector, merger, or
analysis has been opened.

Campaign `barn-icra2027-20260814-002` was canceled outcome-blindly after a
source audit proved that its bundled directory-publication syscall is rejected
by Hopper scratch.  All 20 tasks were canceled after about 2 hours 11 minutes;
an existence-only check found zero canonical seed blocks and zero `COMPLETE`
markers.  Its ledger, scheduler accounting, and hidden incomplete work are
retained, and that campaign will never be finalized or analyzed.

A dated operational amendment replaced both unsupported directory publishers,
without changing any scientific setting.  Commit
`96ab585faedb041e3501fe71d732289d0d5c23fc` staged as the 40-file evidence
source closure
`b9e20a561c8edc93daec8638b15f031dd532eacb39f9d7488582c516ca3dc81c`.
Fresh campaign `barn-icra2027-20260814-003` has the exact primary and
N={2,4,16} cells, five seeds each, and all 20 tasks are now running CPU-only.

## Stable evidence inputs

- Official BARN archive SHA-256:
  `5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320`.
- 300-course manifest SHA-256:
  `1015a6a48ef44add7224200da2ace1cd6c8d7780275b30d7266a44dc88e9ec61`;
  all 900 bound world/path/grid hashes verify.
- Frozen 240/60 split SHA-256:
  `c0ed1d7024ebc240d96a023efb6a124e879fdb06d0342a5e5de7b6d6ed07d7d7`.
- CPU ROS 2/Gazebo container SHA-256:
  `cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0`.
- Analyzer SHA-256:
  `9469bdd52be8ceab9370dd982fd142faf48d58dea16726fce039ca52c5ea944f`.
- Amended preregistration SHA-256:
  `bd6910523d8494e6386b3bb1e816a8e9841becbbaf75809166addc382bd8f0d3`.
- Dataset-preparation receipt SHA-256:
  `216408ddfb6ef95c6d7cc912608aac0428240d09a562f20b03069408b1a9d76f`.
- Train-only timing receipt SHA-256:
  `d9d251c819bbf602dae6c829e3c6755b514639f2fa1c3c9f83cd5b13d21c8738`.
- Outcome-blind Hopper publication probe SHA-256:
  `2ebe0a818d82bc557d6e258a834246377373a789662c6674d46d464bb9a2c72a`.

## Outcome-blind operational correction

The exact campaign-002 bundle used directory
`renameat2(RENAME_NOREPLACE)` in both the seed publisher and remote campaign
sealer.  A fresh probe on Hopper's NFS-backed scratch returned `EINVAL` for
that primitive.  The replacement first hard-links the checksum-bound
`COMPLETE` file to an exclusive hidden sibling claim, verifies inode identity
and destination absence, and then performs one ordinary same-parent directory
rename.  The claim is retained as provenance and a retry fence.

The probe proved one successful claim, `EEXIST` for a second claimant,
successful ordinary rename, and preservation of an existing nonempty
destination.  Network-free tests cover empty/nonempty/file/symlink collisions,
handled failure, both crash windows, 24 concurrent seed races, and 16
concurrent sealed-package races.  The exact staged seed publisher then passed
on Hopper scratch before relaunch.

Verification after the amendment:

- 93 BARN runner/selector/merger/analyzer/verifier tests passed;
- 17 `frontier_rl` tests passed;
- all five source-stage, seed-publish, submit, ledger-finalize, and
  campaign-finalize mocks passed;
- Bash syntax, Python compilation, JSON parsing, and `git diff --check` passed.

## Live campaign

- Campaign ID: `barn-icra2027-20260814-003`.
- Source bundle:
  `/scratch/lwang44/maxrl/bundles/barn_source/b9e20a561c8edc93daec`.
- Launch ledger SHA-256:
  `0a1fc224e71ad2437fce35b40c6561c4b8aeb8750ef6af66af0c34bae731d576`;
  local and remote bytes match.
- Primary: array `9367009`, attempt `primary-attempt-001`.
- N=2: array `9367011`, attempt `n2-attempt-001`.
- N=4: array `9367020`, attempt `n4-attempt-001`.
- N=16: array `9367022`, attempt `n16-attempt-001`.
- At `2026-08-14T10:37:04Z`: all 20 tasks `RUNNING`, 8 CPUs and 24 GB
  each, no GPU/GRES, frozen 36-hour limit.

Primary job `9367009` remained held while its immutable staged ledger became
visible across Hopper login nodes, then the exact same transaction resumed,
installed the canonical digest, and released the same array.  No alternate
job, attempt, source, or ledger was introduced.

## Next actions

1. Monitor scheduler/accounting metadata only.  Do not open logs, endpoints,
   partial artifacts, selectors, mergers, or analyses.
2. If a task fails, retain the attempt and submit the entire affected
   five-seed cell under a new attempt ID with the identical source.
3. Only after every declared task is terminal, finalize the full ledger and
   invoke the source-bound CPU-only all-cell sealer.
4. Fetch only the checksum-closed sealed package, apply the frozen gate, and
   record the reproducible gate decision and milestone commit.

The unrelated MAZE/ICLR/E2c work remains out of scope and was not expanded.
