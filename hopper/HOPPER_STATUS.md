# Hopper status and launch gates

**Snapshot:** 2026-08-14 13:10 EDT

**Overall:** the MAZE CPU/import/full-arm engineering ladder is verified through
endpoint-blind cost and schema. Historical UED import job `9366896` and bounded
one-update job `9366897` passed for their exact older bundle. The independently
audited terminal-chain successor bundle `06ffeeeb6998e8ddb1ce` is staged, and
its fresh exact-bundle import/JIT prerequisite job `9367063` is queued. These
are permanently non-evidence engineering results; every full UED campaign and
all other paper-evidence submissions remain `HOLD`.

## Verified facts

| Item | State | Evidence |
|---|---|---|
| SSH/account | PASS | `lwang44@hopper.orc.gmu.edu`; Slurm account `xiao`; `normal`, `interactive`, and `gpu` QOS observed |
| Lock-addressed MAZE environment | PASS for engineering | `/scratch/lwang44/envs/maze-score-ad774d459fa77bb6`; lock `ad774d459f...`; Python 3.10.20, torch 2.6.0+cu124, NumPy 2.2.6, CUDA 12.4; `pip check` passed |
| Local MAZE-SCORE v2 contract/analyzer | PASS | 20 focused synthetic tests passed on 2026-08-14; exact `frontier_un -> u_N` dispatch, v2 schedule/RNG contract, and analyzer checks are covered |
| Hopper wrapper | PASS | local mock submit/receipt/terminal-gated-fetch/log-gate/health suite passed; resource overrides are rejected and terminal receipts bind QOS/time/restarts/work/log/SubmitLine fields |
| CPU I/O smoke 9366532 | PASS | `COMPLETED`, exit 0; ran on `hop055`; fetched manifest verified; `COMPLETE` and `WORKFLOW_IO_SMOKE_PASS` present |
| GPU import smoke 9366547 | PASS | `COMPLETED`, exit 0 in 34 seconds on `gpu020`; A100 `1g.10gb` CUDA/import/formula/compile checks passed; result and receipts fetched with matching digest |
| Full-arm cost smoke 9366552 | PASS | `COMPLETED`, exit 0 in 22:22 on `gpu013`; endpoint-blind schema/cost package verified; peak GPU memory 39,672 MiB requires retaining `3g.40gb` |
| AMaze UED import replacement 9366815 | PASS | `COMPLETED`, exit 0 in 48 seconds on `gpu021`; exact `e675359647be418bd800` bundle, environment, source-faithful import, formula, and one-JIT closure audited; result-manifest SHA-256 `7416b652...` |
| AMaze UED one-update 9366863 | INFRASTRUCTURE FAIL / HOLD | `FAILED 1:0` after 15 seconds on `gpu021`; stopped before the driver because `/usr/bin/time` was absent; no result inspection or resubmission |
| AMaze UED exact import 9366896 | PASS / ENGINEERING ONLY | `COMPLETED 0:0` in 45 seconds on `gpu021`; exact `6c2ca94c...` source/environment/formula/one-JIT closure audited; result manifest `3a15f52d...`, fetched tree `0aee8fb1...` |
| AMaze UED bounded one-update 9366897 | PASS / ENGINEERING ONLY | `COMPLETED 0:0` in 1:42 on `gpu021`; one PPO update, one upstream gradient update, five optimizer applications, 64 Frontier trials, checkpoint reload, and complete provenance independently verified |
| AMaze UED terminal-chain successor | INDEPENDENT AUDIT GO / IMPORT QUEUED | exact bundle `06ffee...` is remotely staged; local two-phase E2E and independent audit have no P0/P1/P2 findings; fresh import/JIT job `9367063` is pending, and no successor one-update or terminal-chain job is yet authorized by a completed prerequisite |
| Tie-aware v4 remote contract | LOCAL CONTRACT GO / REMOTE HOLD | deterministic snapshot `da74eb3e0debc7781d6d` passes seven focused tests and twin closure, but independent audit finds four primary remote blockers: protected-overlay incompatibility, R2 `job-<id>` mismatch, system-Python GPU probing, and invalid mandatory MIG-gpumem accounting; it implements no submit operation |
| AMaze UED full campaign | **HOLD** | bounded jobs do not authorize 100-update, five-seed, matched-arm, endpoint analysis, or paper-evidence execution |
| GPU engineering job 9361275 | RETIRED | canceled while pending, elapsed `00:00:00`; no GPU allocation, result, or endpoint |
| MAZE-SCORE evidence | **COMPLETE — REPORTED** | All 48 blocks (seeds 20--67) terminal `COMPLETED 0:0` across arrays `9389151` (20--43) and `9389243` (44--67); fetched 2026-08-18 to `/data/robotixx/maze_score/campaign-20260818`, tree digest `1f9eb70447b212b1…`, 48/48 per-cell `SHA256SUMS` verified. Frozen analyzer (hash-matched) run **once**. Primary `un`−`learn` = −.00324 [−.00543,−.00111], **practically ruled out**; secondary `un`−`unif` = +.00888 supported. Written up in `hopper/MAZE_SCORE_RESULT_2026-08-18.md` |
| BARN evidence | FROZEN / RUNNING SEALED | campaign 002 was canceled outcome-blind after an unsupported publication operation; replacement campaign `barn-icra2027-20260814-003` runs the same frozen 20 CPU tasks under the amended hard-link publication contract; no partial log or endpoint may be opened |
| E2c | LOCAL ONLY | frozen protocol forbids moving it to Hopper |

The engineering bundle is
`/scratch/lwang44/maxrl/bundles/maze_score/f4359095fb05490192b4` with manifest
SHA-256 `f4359095fb05490192b404ea03f9fc2413fc7fcd97b20571855b1c38160eaf80`.
It records a dirty engineering source state and cannot pass the evidence
script's clean/FROZEN preflight. Job 9366547 verified this exact bundle and the
lock-addressed environment together inside a Slurm GPU allocation. A verified
local archive is retained at
`autoresearch/iterate-260813-2348/hopper_bundle/f4359095fb05490192b4/`.
The environment lock, freeze, and JSON receipts are likewise archived with a
verified manifest under `autoresearch/iterate-260813-2348/hopper_environment/`.

## Job ledger

### 9366532 — completed CPU workflow smoke

- Request: `normal`, 1 CPU, 1 GB, 5 minutes.
- Submitted with the hash-staged wrapper at `2026-08-14T04:19:00Z`.
- Compute receipt: `2026-08-14T04:19:07Z`, host `hop055.orc.gmu.edu`.
- Sbatch local/remote SHA-256:
  `9e236886e1afa35653e3e7ad010ab772151dd14a66d08abce64f22d5df30b359`.
- Remote result:
  `/scratch/lwang44/maxrl/tests/results/9366532`.
- Local verified copy:
  `autoresearch/iterate-260813-2348/hopper_smoke/job-9366532/`.
- Submission receipt and stdout are saved beside that directory. This is an
  infrastructure result only.

### 9361275 — canceled GPU engineering attempt

- Request: `gpuq`/`gpu`, one `3g.40gb` MIG, 8 CPUs, 60 GB, 40 minutes.
- Final accounting: `CANCELLED by 1224577940`, elapsed `00:00:00`.
- Outcome-blind pre-allocation audit proved the staged partial source could not
  import `estimators`; it also used stale-preserving `cp -n ... || true`, and
  its stdout parent was absent.
- It produced no endpoint and is excluded from every paper registry.
- Full receipt: `autoresearch/iterate-260813-2348/CANCELLED_JOB_9361275.md`.

### 9366547 — completed GPU import smoke

- Request: `gpuq`/`gpu`, one `1g.10gb` MIG, 2 CPUs, 15 GB, 10 minutes.
- Accounting: `COMPLETED`, exit 0, elapsed `00:00:34`; ran on
  `gpu020.orc.gmu.edu` from `2026-08-14T00:31:42` to `00:32:16` local time.
- Verified Python 3.10.20, NumPy 2.2.6, torch 2.6.0+cu124, CUDA 12.4, exact
  score identities, staged imports, and Python compilation.
- Sbatch SHA-256:
  `467c160e6d284ff68062205be1c731d3558522662f41f7f3356cdb5e6e273f9c`.
- Remote/local result tree digest:
  `27b7fdd2c045cb4f361d377cb12de0facaf222ba368e53818c4b453556f7b319`.
- Local verified copy:
  `autoresearch/iterate-260813-2348/hopper_gpu_import/job-9366547/`, with
  stdout and submission receipt beside it.

### 9366552 — completed full-arm cost smoke

- Request: `gpuq`/`gpu`, one `3g.40gb` MIG, 8 CPUs, 60 GB, 6 hours.
- Purpose: one complete engineering-only `un` arm at seed 99 to measure cost,
  completion, and schema; the seed is permanently excluded from inference.
- Accounting: `COMPLETED`, exit 0; `2026-08-14T02:54:21`--`03:16:43` EDT,
  1,342 seconds Slurm elapsed and 1,337 seconds in the timed job region.
- Peak host RSS was 1,368.47 MiB; peak `gres/gpumem` was 39,672 MiB. Slurm's
  zero energy/gpuutil counters are unavailable telemetry, not zero usage.
- The endpoint-blind audit verified one config row and exactly eleven expected
  evaluation-record timepoints `[0,25,...,250]`, without reading metric fields.
  It did not open or fetch stdout, result JSONL, telemetry JSONL, checkpoint,
  or any endpoint value.
- Safe local package:
  `/data/robotixx/maze_score/hopper_cost_audit/job-9366552/`; it contains only
  `profile.tsv`, `COMPLETE`, `SHA256SUMS`, the submission receipt, and the exact
  staged sbatch. Manifest SHA-256:
  `2e9d1f55cbc97bab97b91fe3e9dcc30f85a0d52e32c3c13f3605ee6a42405644`.

## MAZE-SCORE (core ICLR evidence): HOLD

Local v2 code is substantially stronger, but evidence submission is forbidden
until every row below is recorded as PASS together.

| Gate | Current state | Required evidence to clear |
|---|---|---|
| Scientific contract | OPEN | Change `MAZE_SCORE_PREREG.md` from DRAFT to FROZEN and fix the final outcome-blind block count; current 20--49 array is still a candidate design |
| Source identity | OPEN | Clean committed worktree; `stage_maze_score.sh evidence`; immutable bundle path, `SOURCE_STATE.json`, manifest hash, and successful manifest check |
| Environment identity | ENGINEERING PASS | Candidate path and lock are verified; before evidence, bind the recorded freeze/JSON hashes in the clean campaign receipt |
| Import/CUDA smoke | PASS | job 9366547 completed on `1g.10gb`; fetched runtime/manifest/receipt hashes all verify |
| Full-cost arm smoke | PASS | job 9366552 completed on `3g.40gb`; endpoint-blind cost/schema package and manifests verified |
| Runtime/sample freeze | OPEN | Retain `3g.40gb`, choose the outcome-blind block count/power policy, then amend/freeze prereg and array once |
| Campaign receipt | OPEN | Record source, environment, prereg, analyzer, sbatch, warmstart/config, campaign, attempt, and exact `--export` values before submission |
| Blind execution | OPEN | All array cells terminal and complete before any metric-bearing log, JSONL, or analyzer is opened |
| Retrieval/analysis | OPEN | Fetch one complete immutable campaign to a new local directory; verify all hashes/completeness, then run the frozen analyzer once |

The selected `frontier_un` path is already verified as exact `u_N`; the
separate legacy `frontier` path contains the shifted historical formula. The
remaining blocker is the frozen execution/evidence contract, not that formula
dispatch.

## AMaze UED / Frontier-PLR: engineering only

The executable benchmark is `facebookresearch/minimax` at commit
`d053054c5290a04c1c4cd8b55704d999cad73e30`. Final Frontier overlay contract
`5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`
passes 20 local formula/buffer/checkpoint/grouped-runner tests. The exact
score-isolation pair uses 4 levels x 8 streams, `N=8`, and a 500-level buffer
for both Frontier and matched MaxMC.

The current remotely verified Hopper engineering bundle is
`/scratch/lwang44/maxrl/bundles/ued_minimax/6c2ca94ca8109be2775c`, manifest
`6c2ca94ca8109be2775ce0f166e11f064466e4aaa3c2efb085587a0d3f13e93d`.
Its setup-addressed environment is
`/scratch/lwang44/envs/ued-minimax-v2-9ab83896f41c5294-dbd0494789fd70b8`
and includes Python 3.10.20, JAX/JAXlib 0.4.31, and exact Conda Git 2.45.2.

- Attempt 9366785 is a preserved infrastructure failure: exit 127 after 15
  seconds because the compute node lacked host Git. It stopped before source
  import and has no `COMPLETE` artifact or scientific result.
- Replacement import/formula/JIT job 9366815 binds the exact new bundle,
  environment freeze, environment manifest, setup, overlay, source, and sbatch
  hashes. It completed with exit 0 in 48 seconds on `gpu021`; the fetched tree,
  `COMPLETE`, runtime, receipt, source-faithful import, exact formulas, and one
  JIT all verify. Remote and local result-manifest SHA-256 is
  `7416b652ed46963e903ea438a2c5204d6574db8356d62b5ee9388a1c5e46c307`;
  the verified local copy is
  `/data/robotixx/ued_bench/hopper/import-smoke-job-9366815/`.
- Exact-bundle one-update job 9366863 was submitted from that closure but
  failed `1:0` after 15 seconds, before the bounded driver or PPO update,
  because compute node `gpu021` did not provide `/usr/bin/time`. Only the
  terminal bounded diagnostic was inspected; no partial result was inspected
  and no job was resubmitted.
- Fresh exact-bundle import/formula/JIT job 9366896 completed `0:0` in 45
  seconds on `gpu021`. Its verified local copy is
  `/data/robotixx/ued_bench/hopper/import-smoke-job-9366896/`; result-manifest
  SHA-256 is
  `3a15f52ddb0aa0b44f190f9701183c51884b91a0f1d850f327a53c3208f2a14c`
  and the complete fetched-tree digest is
  `0aee8fb1d99954ce9e1a94f4e01a20b77613fa588448e27e46eb35e0c75fdf6a`.
- Bounded exact-bundle one-update job 9366897 completed `0:0` in 1:42 on
  `gpu021`. Its verified local copy is
  `/data/robotixx/ued_bench/hopper/one-update-job-9366897/`; result-manifest
  SHA-256 is
  `4eaa676052cbc9006da1d285b03eda354cab27f3b7d72064b5138724c83691c8`
  and the complete fetched-tree digest is
  `e5f32761a3ee2b0ed25a9f8637f066b88a445769f037af1d579aefe376fed3d3`.
  The terminal state and an independent checkpoint load both give
  `n_updates=1`, `n_grad_updates=1`, five optimizer applications, 64 trials,
  four filled slots, and zero incomplete or duplicate-new groups. Checkpoint
  reload/static-signature and exact train-state continuity gates passed.
- The authorized replacement removes every host GNU-time assumption. The
  bounded driver records Python UTC/monotonic and `RUSAGE_SELF` diagnostics;
  terminal Slurm `sacct` remains authoritative external accounting. Local
  syntax/wrapper tests, the 22-test source suite, the grouped RNG contract, and
  the staged restricted-`PATH` one-update E2E pass. Schema 3 also excludes the
  two local Blackwell/JAX 0.6.2 probe trees from this JAX 0.4.31 bundle while
  retaining canonical overlay/config/test/driver hashes.
- The replacement was frozen from byte-identical builds and verified remotely:
  bundle ID
  `6c2ca94ca8109be2775c`, manifest SHA-256
  `6c2ca94ca8109be2775ce0f166e11f064466e4aaa3c2efb085587a0d3f13e93d`,
  overlay manifest SHA-256
  `e3a85cb9643f2bf2a9cc5337f15dcb7ddd87614d0342bcbe8a8a5b9e167ea1d4`,
  import sbatch SHA-256
  `fdf375950660be4179807a14fcf9dbfd815219e2bec525130a847b907d7ded7b`,
  and one-update sbatch SHA-256
  `fadcb628ca5c3c0282a4e78b1ce8d98011b487c00baf8cd22a86a9d12a7bed1f`.
  Jobs 9366896 and 9366897 verify this exact closure. Neither may be reused to
  gate any later bundle whose manifest differs.
- Any replacement one-update may execute only two outer cycles, 16,384
  transitions, one PPO update, `n_grad_updates=1`, and exactly five optimizer
  applications (five epochs x one minibatch); its receipt must also show 64
  Frontier trials, zero incomplete/duplicate-new groups, checkpoint reload
  continuity, and no OOD evaluation.
- The next terminal-chain training/evaluation/package smoke, if run, is also
  permanently non-evidence and bounded to one PPO update. It is an explicit
  two-phase workflow: the allocation may publish only `COMPONENTS_COMPLETE`;
  after `COMPLETED 0:0`, `hopper.sh terminal-receipt` captures scheduler-only
  accounting before any endpoint/log fetch. Four schema-2 fetch receipts then
  bind their pre-probe start times to that terminal receipt. The exact bundled
  finalizer requires the immutable submission receipt and scheduler
  `SubmitLine`, exact QOS/time/no-requeue/MIG allocation, closed components and
  marker, exact ordered Hopper-local scheduler timestamps/epoch/elapsed, an
  explicit no-`ALL` submission export, and an injection-free
  `python -I -B` 3.10.20 venv before it can produce
  an `analyzer_eligible=false` package. The local staged E2E and adversarial
  preterminal/local-CPU/receipt/closure/interpreter tests pass. No
  terminal-chain job has been submitted, and the new changed bundle
  must pass fresh exact-bundle import and one-update prerequisites; jobs
  9366896/9366897 cannot gate it. All 100-update, multi-seed, paired-arm,
  analyzer, and paper-evidence UED runs remain `HOLD`.
- The locally frozen successor candidate is bundle ID
  `06ffeeeb6998e8ddb1ce`, manifest SHA-256
  `06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff`.
  Two independent local builds were byte-identical and verified. Independent
  audit found no P0/P1/P2 and authorized only the bounded engineering ladder.
  The exact bundle is staged remotely and fresh import/JIT job `9367063` is
  pending scheduler priority. It has no fresh completed import or one-update
  gate yet; the old `6c2c...` jobs cannot satisfy its prerequisites.

## BARN (ICRA evidence): FROZEN, RUNNING SEALED

The official 300-course archive, 240/60 split, Gazebo container, runner,
machine protocol, analyzer, source bundle, timing projection, and outcome-blind
publication amendment are frozen. Campaign `barn-icra2027-20260814-002` was
canceled without endpoint access after Hopper rejected the original directory
publication primitive. Replacement campaign `barn-icra2027-20260814-003`
contains the same primary plus `N={2,4,16}` cells and five seeds per cell under
the amended hard-link claim protocol. It is the only live BARN evidence
campaign.

Scheduler-only snapshot at 2026-08-14 13:10 EDT: all 20 tasks remain
`RUNNING` after about 6h34m, with no terminal row, across arrays 9367009,
9367011, 9367020, and 9367022. No stdout, result tree, success value, or other
endpoint was accessed.

Monitor only scheduler/accounting metadata recorded by the BARN workflow. Do
not use generic `hopper.sh logs` or `fetch`, inspect one array task, select a
partial attempt, or launch another campaign. After all exact ledger entries are
terminal, use only `finalize_barn_ledger.sh` followed by
`finalize_barn_campaign.sh`; `BARN_CAMPAIGN_SEALED` is the sole authorized
outcome retrieval signal. Scientifically, this frozen campaign prioritizes ten
difficulty strata while all `N` episodes in a group share one sampled course,
so it is a stratum-priority heuristic unless sealed course-level diagnostics
support the required homogeneity assumption.

## Next authorized work

1. Make an outcome-blind MAZE sample-size decision. At the pessimistic
   historical paired SD, 30 blocks are underpowered; 60 are near 80% only at
   unadjusted alpha, while 72 are near 80% at the planned first-step adjusted
   alpha. Record quota policy and retry limits before changing the analyzer.
2. Retain `3g.40gb`; the cost smoke nearly filled it. Keep the tested 8 CPU/
   60G request unless a smaller CPU/RAM contract is separately re-smoked.
3. Commit/freeze the agreed protocol and stage a clean evidence bundle only
   after all remaining campaign-receipt hashes agree.
4. Monitor queued exact-bundle import job `9367063` by scheduler state only.
   On `COMPLETED 0:0`, fetch to a new destination and audit the entire closure;
   only then construct the exact successor one-update submission. Jobs 9366896
   and 9366897 are not cross-bundle prerequisites. Only after both fresh gates
   pass may the single bounded terminal-chain engineering smoke be submitted.
5. MAZE-SCORE is complete and reported; no further MAZE-SCORE submission is
   authorized. Keep every new BARN submission on `HOLD`. Preserve
   campaign 003 and use only its frozen ledger/finalization path after all tasks
   are terminal.
6. Keep v4 staging and submission on `HOLD`. Repair the four primary audited
   blockers in a new sibling identity, then repeat deterministic twin
   construction and an independent frozen audit before adding any remote
   command to this runbook.

Operational commands and retrieval rules are in
[`HOPPER_SETUP.md`](HOPPER_SETUP.md). Paper-priority rationale is in
[`../DESIGN_IMPROVEMENT_PLAN.md`](../DESIGN_IMPROVEMENT_PLAN.md).
