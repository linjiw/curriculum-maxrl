# ICRA 2027 BARN navigation campaign — preregistration

**Status:** FROZEN

Frozen outcome-blind on 2026-08-14 in the milestone commit containing this
line. No BARN scientific endpoint had run or been inspected at freeze. The
source bundle, runner, merger, selector, analyzer, machine protocol, receipts,
and this file must remain content-addressed together for evidence.

The prose protocol is mirrored by `icra2027/barn_protocol.json`, also marked
`FROZEN`, SHA-256
`36007d8c979b2dacccd595a43a4620dca7be24c1f50ef91a8a9ee4e869202cb2`.
Its SHA-256 is a seventh mandatory evidence binding alongside the manifest,
split, preregistration, analyzer, container, and source bundle.
The runner and blind merger reject any CLI or artifact whose exact cell,
schedule, seed, arm order, isolation mapping, or hash differs from that
machine-readable contract.

## Claim under test

At deployed group size `N=8`, sampling navigation-training strata by the
estimator-derived coefficient-activity utility

`u_N(p) = 1 - (1-p)^N - p`

improves time-integrated target-uniform success on a fixed held-out BARN course
set relative to uniform sampling and compute-blind `p(1-p)` learnability. A
hand-ordered staged curriculum is reported as a fourth baseline but is not an
Aug. 24 gate comparator.

The factor of two in expected total absolute advantage is immaterial to the
sampling distribution. The utility is hyperparameter-free conditional on N;
posterior decay and the uniform replay floor are shared stability settings,
not part of the central claim.

## Immutable domain, acquisition, and split

- Domain: the official 300-course Benchmark for Autonomous Robot Navigation
  (BARN) dataset, using the CPU-only Gazebo Classic adapter in
  `frontier_rl/adapters/barn_gazebo.py` and a shared lidar-to-velocity policy.
- Official acquisition URL:
  `https://www.cs.utexas.edu/~xiao/BARN/BARN_dataset.zip`.
- Archive SHA-256:
  `5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320`.
- Acquisition receipt:
  `icra2027/receipts/barn_dataset_acquisition.json`, SHA-256
  `a59655b7e39c75fb32d120e8c84cf28651e5ef88a76b66838b96401206416aca`.
- Manifest: `icra2027/barn_manifest.jsonl`, 300 unique courses, SHA-256
  `1015a6a48ef44add7224200da2ace1cd6c8d7780275b30d7266a44dc88e9ec61`.
  All 300 world, 300 path, and 300 grid hashes were reverified against the
  local archive extraction before freeze.
- Difficulty is the published optimal traversal time in seconds; longer is
  harder. The observed metadata range is 5.0266136575--6.8676534716 seconds.
- Split: `icra2027/barn_split.json`, SHA-256
  `c0ed1d7024ebc240d96a023efb6a124e879fdb06d0342a5e5de7b6d6ed07d7d7`,
  generated with seed `20270811`, 10 difficulty strata, and an 80/20 split.
  It contains 240 training and 60 held-out courses, exactly 24/6 per source
  stratum, with no overlap.
- All fixed-seed backend engineering used `barn-299`, which the frozen split
  assigns to training. No prospective held-out course was used for adapter
  tuning, determinism checks, or throughput measurement.

## Teacher unit and arms

The teacher task is one of 10 frozen training difficulty strata, not an
individual course. For each requested stratum, the adapter samples one of its
24 training courses uniformly and runs one rollout group there. All arms share
the same stratum definitions, within-stratum course RNG, simulator seeds,
policy, estimator, budgets, and evaluation stream.

Course-level teaching was considered and rejected before evidence: with 240
training courses and only five independent training seeds, its per-course Beta
posteriors would be too sparse to distinguish score shape. Strata retain the
published difficulty ordering while providing denser requested-task evidence.

The four primary arms are:

1. `ours_uN`: decayed Beta posterior, Thompson draw, exact `u_8`, and 10%
   uniform floor.
2. `uniform`: uniform stratum sampling with posterior bookkeeping retained for
   diagnostics.
3. `learnability`: the identical posterior, Thompson draw, concentration, and
   floor with utility `p(1-p)`. This is “learnability,” never “ALP-GMM.”
4. `staged`: unlock strata from shortest published traversal time upward,
   uniform over the unlocked prefix, promote at posterior success 0.7 after at
   least five frontier groups, and retain the same 10% global floor.

The easy-decile secondary endpoint uses the same published traversal-time
metadata; it is the six easiest held-out courses after stable `(difficulty,
env_id)` sorting.

## Execution environment and isolation

- Execution site: GMU Hopper `normal` partition, CPU-only. No BARN process may
  see or request a GPU; the local RTX 5090 remains embargoed for E2c.
- Frozen Apptainer image:
  `/scratch/lwang44/ros2-gazebo-classic.sif`, SHA-256
  `cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0`.
- Environment receipt:
  `icra2027/receipts/barn_hopper_environment.json`, SHA-256
  `cf78acad4b8d13c08644032dc957e86cfa46543ed4e004fe4196ff03e58c508c`.
  Fingerprint job 9366688 completed with exit 0 on `hop072`; it records ROS 2
  Humble, Gazebo Classic 11.10.2, Python 3.10.12, 1,542 Debian package rows,
  227 Python distributions, OS/kernel data, and a headless boot receipt.
- Compute-node dataset preparation job 9366817 completed with exit 0 on
  `hop072` and published the checksum-closed archive package. Its receipt is
  `icra2027/receipts/barn_hopper_dataset_prepare.json`, SHA-256
  `216408ddfb6ef95c6d7cc912608aac0428240d09a562f20b03069408b1a9d76f`.
- The outcome-blind, training-only timing smoke was job 9366831. Its redacted
  receipt is `icra2027/receipts/barn_hopper_training_smoke.json`, SHA-256
  `d9d251c819bbf602dae6c829e3c6755b514639f2fa1c3c9f83cd5b13d21c8738`.
  The derived resource projection is
  `icra2027/receipts/barn_hopper_feasibility_projection.json`, SHA-256
  `242e89a90832dd6aeb2b70e8fe94f1f9c1bac32c5d0daaf683d5e60005b59237`.
- One seed runs per Slurm process group. Seeds receive distinct
  `ROS_DOMAIN_ID`, Gazebo master ports, runtime roots, logs, and result paths.
  Campaign cells also receive disjoint ID/port ranges: domain bases
  `{primary:20, N2:50, N4:80, N16:110}` with `2*seed`, and master-port bases
  `{primary:13000, N2:14000, N4:15000, N16:16000}` with `4*seed`; isolated
  evaluation uses the immediately following ID and port.
- Arms run sequentially inside a seed to avoid simulator/process interference,
  but order is predeclared and counterbalanced. Primary seed orders are
  `{1:[ours,uniform,staged,learnability],
  2:[uniform,learnability,ours,staged],
  3:[learnability,staged,uniform,ours],
  4:[staged,ours,learnability,uniform],
  5:[uniform,ours,learnability,staged]}`. Two-arm ablations alternate order by
  seed (`ours` first for 1/3/5, learnability first for 2/4). Results are stored
  under canonical arm names and retain the executed order.
- Evidence jobs must bind the frozen manifest, split, preregistration,
  analyzer, container, and source-bundle SHA-256 values. The runner refuses
  full-evidence status when any binding is missing or inconsistent.

## Known simulator nondeterminism

Exact, acknowledged multi-step physics removed free-running physics timing but
did not make the ROS/Gazebo control loop bit-exact. On training course
`barn-299`, the same course, policy state, and seed succeeded twice but required
3,670 versus 3,260 simulator steps; commands and trajectory hashes also
differed. The two-episode measurement took 42.87 wall seconds on the local
Intel Core Ultra 7 265K. Receipt:
`icra2027/results/barn_backend_throughput_2026-08-14.json`, SHA-256
`7e4601feccc1482a46a34b2ca3927fb00eef9733e1194afe44ad847d7ca26d5d`.

Therefore the protocol does not claim fixed-seed trajectory or step-count
identity. It freezes physics settings, uses paired training and course-level
evaluation seeds across arms, keeps training seed as the independent unit, and
uses simulator transitions as the primary accounting currency. Outcome bits,
step counts, simulated seconds, course/stratum identity, and status are
retained for every training and evaluation episode; process wall/resource
telemetry is retained per arm/job. This
nondeterminism was documented before evidence and cannot be used to tune an
arm after launch.

## Primary schedule and budgets

The original grid-smoke defaults would imply 28,800 BARN episodes and about
171.48 local wall-hours per seed; they are infeasible and are not the evidence
schedule. The candidate outcome-blind evidence schedule is:

- paired training seeds `{1,2,3,4,5}`;
- deployed group size `N=8`;
- two requested strata per trainer update;
- primary training budget `1,000,000` Gazebo physics steps per arm;
- evaluate after initialization and after the first completed two-group
  trainer update whose cumulative training counter crosses each of
  `{200000,400000,600000,800000,1000000}` simulator steps;
- one stochastic episode per each of the 60 held-out courses at every
  checkpoint, with the same course-level seed across arms and checkpoints;
- complete the whole two-group trainer update that crosses the budget, record
  its overshoot, then stop and interpolate AUC at exactly 1,000,000 steps. No
  checkpoint or policy update occurs between the two jointly optimized groups;
- if one completed update crosses more than one pending threshold, evaluate
  once at that post-update policy and treat that checkpoint as covering every
  crossed threshold; do not duplicate identical evaluations;
- hard safety cap of 200 trainer updates; hitting the cap before the budget is
  an infrastructure failure, not a shortened scientific run;
- episode timeout 25.0 simulated seconds, fixed physics step size 0.005, fixed
  real-time update rate 2,000.

The outcome-blind, training-partition-only Hopper smoke measured 50,570
training physics steps in 330.496625 wall seconds (153.012153 steps/s) and two
smoke evaluation episodes in 62.232567 wall seconds (31.116284 s/episode). At
the frozen 1,000,000-step budget and planning case of six checkpoints times 60
held-out episodes, the linear resource projection is 4.927025 hours per arm,
19.708101 hours per four-arm primary seed, and 9.854050 hours per two-arm
ablation seed. A predeclared 20% operational pad gives 23.649721 and 11.824860
hours, respectively. Because a 24-hour kill limit would leave only 21.017
minutes above the padded primary projection, every evidence seed job requests
36 hours (`1-12:00:00`) on Hopper `normal`; the live-verified partition
`MaxTime` is seven days. The larger request changes only the scheduler kill
limit: the 1,000,000-step budget, 200-update safety cap, evaluation
schedule/panel, episode timeout, seeds, and arms remain unchanged.

This timing decision used only immutable provenance, completion state, and
resource counters from the declared training-only engineering smoke
(`heldout_courses_read=false`, `resource_counters_only=true`,
`paper_endpoint_emitted=false`, and
`internal_metric_artifact_retained=false`). No success, reward, AUC,
trajectory, status outcome, metric-bearing log, held-out asset, or scientific
evidence artifact was inspected. The extrapolation is based on one N=8 update
and two train-course evaluation episodes, so the 36-hour request supplies
operational headroom rather than claiming a runtime guarantee.

## Primary and secondary endpoints

Because exclusive like-for-like hardware cannot be guaranteed on Hopper, the
shared-hardware escape hatch is invoked before unblinding:

- **Primary:** area under held-out target-uniform mean success versus Gazebo
  physics steps, interpolated at the frozen 1,000,000-step budget.
- **Descriptive accounting view:** the same AUC at paired common training wall
  time. Evaluation wall time is excluded from training wall time and reported
  separately.

Secondary endpoints are final held-out target-uniform success, success/AUC by
frozen difficulty decile, easy-decile retention, all-fail and all-pass group
rates, posterior calibration, sampled-stratum distribution, training episodes,
physics steps, optimizer updates, wall time, and status/collision/timeout
counts. Secondary outcomes do not alter the gate.

## Mandatory N ablation

Run `N in {2,4,8,16}` at the same 1,000,000-step training budget per arm. The
confirmatory contrast is exact deployed `u_N` versus `p(1-p)` at each N, with
the same posterior/floor, seeds `{1,2,3,4,5}`, held-out panel, transition
checkpoints, timeout, and physics settings. The number of trainer updates is
allowed to differ because complete group cost changes with N; the transition
budget and 200-update safety cap stay fixed. The existing Acrobot result does
not replace this ablation.

Fresh two-arm cells are run for `N={2,4,16}`. The `N=8` ablation point reuses
the primary cell's `ours_uN` and `learnability` rows because their seeds,
teacher settings, held-out panel, transition budget, and evaluation schedule
are identical; a duplicate fresh N=8 cell is forbidden. The ablation analyzer
requires the three fresh merged cells plus the complete primary merge.

## Analysis and Aug. 24 gate

- Independent unit: training seed. Courses and evaluation episodes are repeated
  measurements, not independent replicates.
- Pair arms by training seed and fixed evaluation stream.
- Report paired mean delta, 95% paired percentile-bootstrap interval (20,000
  draws, seed 20270811), every seed delta, positive/tie counts, and the exact
  two-sided sign-flip p-value.
- No paper-level inference or gate decision from fewer than five complete
  paired seeds.
- Binding gate: continue the ICRA deadline only if `ours_uN` is directionally
  at least as good as both `uniform` and `learnability` at the frozen primary
  transition budget. `staged` is reported but cannot veto or pass this gate.
- Otherwise preserve the campaign and pivot to RA-L without deadline-driven
  expansion. A clean negative is not a failed experiment.
- Frozen analysis implementation is `icra2027/analyze_campaign.py`, SHA-256
  `9469bdd52be8ceab9370dd982fd142faf48d58dea16726fce039ca52c5ea944f`.

## Blinding, retries, and completeness

- Do not open partial BARN endpoint JSON, per-arm result data, or metric-bearing
  logs before all declared cells are terminal and the strict merger passes.
- Scheduler state, accounting, immutable hashes, resource telemetry, and
  non-metric completion markers may be inspected while jobs run.
- The runner writes its result atomically only after all declared arms complete.
  A failed/incomplete attempt is retained and the entire five-seed cell is
  rerun under a new attempt ID without changing seeds or settings.
- Every submission is recorded before execution in a normalized JSON ledger
  containing campaign ID, cell, attempt ID, seed, submission UTC, Slurm array
  job ID and array-task ID (plus the numeric task job ID once known), artifact
  path, and the seven expected hashes. The runner
  repeats that identity in the artifact and refuses an existing destination.
- The sole authorized evidence path is the source-bound sequence
  `stage_barn_campaign.sh evidence` -> four calls to
  `submit_barn_campaign.sh` -> `finalize_barn_ledger.sh` ->
  `finalize_barn_campaign.sh`. A held submission is resumed only by rerunning
  the exact same campaign/cell/attempt/source command; a different attempt is
  refused while that durable transaction is pending.
- A campaign ledger is not finalizable until it contains the exact declared
  cell set `{primary, ablation_n2, ablation_n4, ablation_n16}`, with all five
  seeds represented in every cell and every recorded Slurm task terminal.
  Selection, merging, analysis, or endpoint-bearing fetch from a proper subset
  is forbidden. Direct standalone selector, merger, analyzer, raw-log, or raw
  campaign-fetch use is not an authorized substitute for the all-cell sealer.
- If more than one complete attempt exists for a seed, the automated selector
  enumerates the full ledger and selects the earliest submitted structurally
  complete, hash-valid attempt; duplicate unselected artifacts are retained
  and excluded. Omitted ledger entries, unknown artifacts, identity mismatch,
  or endpoint-dependent selection are hard failures.
- The merger must reject missing/extra seeds or arms, duplicate seeds,
  provenance/config drift, incomplete histories, and any mismatch in the seven
  frozen hash bindings. Only its complete merged artifact may be analyzed.
- The sealer performs all four selections, all four merges, and both analyses
  in one CPU-only Slurm transaction and publishes one checksum-closed package.
  Once that package exists, the campaign ID is permanently closed to further
  attempts.
- Engineering smokes use training-partition courses only and are permanently
  stamped `engineering_smoke_not_paper_evidence`.

## Scope and stopping rules

- No PLR, HER, GCL/GACL, or hardware arm before the 4-arm five-seed matrix is
  complete or safely running.
- No local GPU use and no ICLR/MAZE-SCORE/E2c work in this campaign.
- Do not change score shape, seed list, split, promotion rule, transition
  budget, evaluation panel, or retry rule after freeze except through a dated,
  outcome-blind amendment recorded before the affected run.
- The Aug. 24 gate is binding. A tie or directional win versus both named
  controls permits the ICRA deadline; otherwise move to the RA-L timeline.

## Outcome-blind operational amendment — 2026-08-14

Before any BARN evidence task received a compute allocation, the first primary
submission transaction (campaign `barn-icra2027-20260814-001`, held array job
9366866) lost its remote ledger-install acknowledgement. The array remained
exactly `PENDING|JobHeldUser`; no seed task ran, no endpoint or raw job log was
opened, and the incomplete campaign was abandoned with its pending ledger
retained. Job 9366866 was canceled while still held.

The source-bound submitter was amended to make this pre-execution transaction
idempotent: an exact SHA-256-matching remote ledger staging file is reused on
resume, while a partial regular staging file is replaced from the durable
local proposal; the canonical ledger still must verify before release. The
network-free regression now interrupts the install after upload and proves
that the same held array and attempt resume without a second allocation or
upload. Evidence restarts under a new campaign ID and a newly content-addressed
source bundle. This amendment changes no scientific arm, seed, split, budget,
checkpoint, timeout, isolation, analysis, retry-selection, or gate rule; the
machine protocol remains unchanged.

## Venue constraints

The ICRA 2027 deadline is September 15, 2026, 11:59 PST. The manuscript is
double-column, double-anonymous, and limited to eight total pages including
references. AI-generated article content must be disclosed in acknowledgments;
review material may not be processed through an AI system.
