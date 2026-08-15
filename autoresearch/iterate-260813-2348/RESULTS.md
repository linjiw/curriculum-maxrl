# Results

**Snapshot:** 2026-08-14 00:39 America/New_York

**Scientific evidence launched:** no

**Engineering jobs:** CPU I/O PASS; GPU import PASS; full-arm cost QUEUED

## Kept changes

- Confirmed that MAZE-SCORE selects exact `u_N` through `frontier_un`; the
  shifted `u_(N+1)` implementation is a separate legacy path.
- Added canonical exact/legacy formula helpers and a backward-compatible
  `maze_score_v2` training protocol with frozen knobs, independent RNG streams,
  hashed shared warmstarts, endpoint-silent stdout, delivery telemetry, and
  exactly one evaluation at each completed update 25--250.
- Added a fail-closed 30-block analyzer with strict completeness/provenance
  checks, paired bootstrap, exact paired sign-flip inference, Holm correction,
  and supported/practically-ruled-out/inconclusive branches.
- Replaced mutable Hopper staging with content-addressed source bundles,
  hash-staged sbatch submissions, local/remote receipts, scheduler monitoring,
  endpoint-log protection, and checksum-verified retrieval.
- Rewrote the DRAFT preregistration around a candidate 30-block paired design,
  fresh evaluation panels, counterbalanced arm order, and a mandatory
  smoke/freeze/evidence ladder.

## Verification

- Local protocol/analyzer suite: 20 tests passed.
- Hopper wrapper mocked submit/fetch/log-gate/health/watch suite: PASS.
- Shell syntax and `git diff --check`: PASS at the last verification point.
- Lock-addressed environment:
  `/scratch/lwang44/envs/maze-score-ad774d459fa77bb6`.
- Environment lock SHA-256:
  `ad774d459fa77bb68c01c4a225db1e7faa3213216422eb5eabdf5b3c0e3d6224`.
- The exact lock/freeze/JSON receipts and a verified manifest are archived in
  `hopper_environment/`.
- Candidate engineering bundle:
  `/scratch/lwang44/maxrl/bundles/maze_score/f4359095fb05490192b4`.
- Bundle manifest SHA-256:
  `f4359095fb05490192b404ea03f9fc2413fc7fcd97b20571855b1c38160eaf80`.
- The exact bundle is archived at
  `hopper_bundle/f4359095fb05490192b4/`; its verified remote/local tree digest
  is `73e6cc51d53fd3d32ffe358a668e3446455e7b43670a73729b5776040fe786c2`.

## Hopper jobs

| Job | Purpose | Terminal result | Local record |
|---|---|---|---|
| 9361275 | obsolete full GPU smoke | canceled before allocation; elapsed 0 | `CANCELLED_JOB_9361275.md` |
| 9366532 | CPU submit/write/fetch smoke | completed, exit 0, 1 second | `hopper_smoke/` |
| 9366547 | import-only `1g.10gb` GPU smoke | completed, exit 0, 34 seconds | `hopper_gpu_import/` |
| 9366552 | full-arm `3g.40gb` cost/schema smoke, seed 99 | queued for priority | `hopper_full_arm/job-9366552.submission.tsv`; fetch result after terminal |

Job 9366547 ran on `gpu020.orc.gmu.edu`, saw an A100 MIG `1g.10gb`, and
verified Python 3.10.20, NumPy 2.2.6, torch 2.6.0+cu124, CUDA 12.4, formula
identities, and staged-source compilation. Its retrieved result tree digest is
`27b7fdd2c045cb4f361d377cb12de0facaf222ba368e53818c4b453556f7b319`.

## Current decision

Keep the v2 protocol, analyzer, and Hopper workflow. Do not launch the
MAZE-SCORE evidence array or BARN campaign. The immediate next gate is terminal
completion and verified retrieval of job 9366552, followed by an outcome-blind
cost/sample freeze and a clean committed evidence bundle. No partial or
engineering scientific endpoint has been used to change the design.
