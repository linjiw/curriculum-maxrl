# Curriculum-MaxRL → ICRA 2027 progress report

**Date:** 2026-08-11
**Updated:** 2026-08-14
**Scope:** First bounded implementation pass plus current BARN/Hopper readiness
**Status:** CPU-only BARN engineering gates passed; protocol frozen at `23dacb8`; no BARN scientific evidence launched

## Current update — 2026-08-14

The earlier external-asset/backend blocker is closed. The official 300-course
BARN archive, all 900 adapter-consumed asset hashes, the deterministic 240/60
train/held-out split, ROS 2/Gazebo container, exact-step backend, real BARN
runner, frozen-shape analyzer, blind selector/merger, and source-bound Hopper
workflow are now implemented and verified. No BARN scientific endpoint has run
or been inspected.

### Outcome-blind Hopper engineering ledger

| job | purpose | terminal result | retained decision |
|---:|---|---|---|
| 9366805 | dataset preparation | failed `127:0` in 11 s | Missing host `/usr/bin/time`; switched all BARN jobs to Bash timing. |
| 9366814 | dataset preparation retry | failed `1:0` in 48 s | Hopper scratch rejected directory `renameat2(RENAME_NOREPLACE)`; changed publication to an atomic canonical-directory claim with `COMPLETE` hard-linked last. |
| 9366817 | dataset preparation | completed `0:0` in 42 s | Canonical checksum-closed package retained. |
| 9366819 | one-update train-only smoke | failed `1:0` in 39 s | Exact binaries built and no-asset guard passed; preserved ROS `PYTHONPATH` so `rclpy` remains importable. |
| 9366821 | corrected train-only smoke | completed `0:0` in 8 min 28 s | Runtime validation retained, but its receipt lacked simulator-step/phase timing counters and was not used for the final projection. |
| 9366831 | resource-instrumented train-only smoke | completed `0:0` in 7 min 15 s scheduler elapsed (422 s receipt elapsed) | Final outcome-blind feasibility receipt retained. |

Dataset preparation receipt SHA-256:
`216408ddfb6ef95c6d7cc912608aac0428240d09a562f20b03069408b1a9d76f`.
Training smoke receipt SHA-256:
`d9d251c819bbf602dae6c829e3c6755b514639f2fa1c3c9f83cd5b13d21c8738`.

Job 9366831 reported only resource counters: 50,570 training simulator steps,
16 training episodes, 330.497 training seconds, and 62.233 seconds for two
evaluations on train course `barn-299`. It was CPU-only, read no held-out
course, retained no metric artifact, and emitted no success, reward, AUC,
trajectory, held-out result, paper endpoint, or scientific comparison.

### Timing decision frozen before evidence

The resource receipt implies 153.012 training simulator steps/second, 1.815
hours for one million training transitions per arm, and 31.116 seconds per
evaluation episode. Six checkpoints over all 60 held-out courses project to
3.112 evaluation hours per arm, or 4.927 hours total per arm. Therefore:

- four-arm primary cell: 19.708 hours nominal, 23.650 hours with 20% padding;
- two-arm ablation cell: 9.854 hours nominal;
- margin under a 24-hour task after primary padding: only 0.350 hours;
- recommended Slurm request: 36 hours, leaving 12.350 hours beyond the padded
  primary estimate.

The 36-hour request is scheduler headroom only. It does not change the frozen
one-million-transition training budget, checkpoint cadence, evaluation course
count, arm definitions, or paired seeds.

### Freeze milestone and next action

The preregistration and machine protocol are now exactly `FROZEN` in commit
`23dacb88cf7b1f46dddf9d2453dbd7e0bcbbbf33`. The frozen protocol SHA-256 is
`36007d8c979b2dacccd595a43a4620dca7be24c1f50ef91a8a9ee4e869202cb2`;
the frozen preregistration SHA-256 is
`975e3cced69807c86569acd167f5292d5cf8d8e1872b2f8b9a5f876cc464ab77`;
the analyzer SHA-256 is
`9469bdd52be8ceab9370dd982fd142faf48d58dea16726fce039ca52c5ea944f`.
The milestone gate passed 93 BARN contract tests, 17 core tests, four
fail-closed shell workflow mocks, shell syntax, Python compilation, and
whitespace validation. No BARN evidence had been submitted or inspected at
freeze. The next authorized action is a fresh evidence-mode source stage from
that commit, followed by the exact source-bound four-cell submission; no
engineering bundle may be reused.

### Pre-execution ledger amendment — 2026-08-14

The first source-bound primary submission created held array job 9366866 for
campaign `barn-icra2027-20260814-001`, but the remote canonical-ledger install
or its acknowledgement did not complete. The fail-closed wrapper never
released the array. Scheduler-only inspection confirmed
`PENDING|JobHeldUser`; the job was canceled in that state with no compute
allocation, seed execution, endpoint inspection, or raw metric-log access.
The incomplete campaign and exact pending ledger are retained.

The source-bound submitter was hardened to reuse an exact SHA-256-matching
ledger staging file on same-attempt resume and to replace only a partial
transaction-owned upload from the durable local proposal. A network-free
regression now interrupts the install after upload and proves that recovery
uses the same held job without a second allocation or upload. All 93 BARN
tests and four workflow mocks pass after this change. The preregistration has a
dated outcome-blind operational amendment; its SHA-256 is now
`f9dcc5f56ef890a7a32fd14244fd7073f50f27f7ad4ad5dea20efcb347f01864`.
Scientific settings and the machine protocol are unchanged. Evidence will
restart under a new campaign ID and freshly staged source SHA.

## Historical 2026-08-11 bounded implementation snapshot

The sections below preserve the original 2026-08-11 snapshot; superseded
blockers and hashes are historical rather than the current campaign state.

## Bottom line

The ICRA track now has an outcome-blind, tested campaign scaffold rather than
only a strategy document. The domain decision is BARN-style mobile navigation,
with goal-conditioned hindsight as an optional arm and Isaac Lab as the
fallback. Four primary sampler arms run end-to-end on a CPU navigation smoke
adapter, held-out evaluation is paired and state-isolated, both wall-clock and
simulator-step comparisons are enforced by analysis, and the August 24 gate
cannot be triggered by a smoke or under-seeded artifact.

This is engineering progress, not robotics evidence. No BARN course assets,
Jackal packages, or Isaac Lab installation were found in the local workspace or
runtime. ROS 2 Humble and Nav2 are installed, so the missing boundary is the
lab-specific course/simulator/robot stack rather than the entire ROS base.

## Important correction to the supplied plan

The official [ICRA 2027 call for technical papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
confirms the September 15, 2026, 11:59 PST deadline, double-anonymous review,
and double-column format. It specifies **8 pages total, including references**,
not six content pages plus unlimited references. It also requires disclosure
of AI-generated article content in the acknowledgments and prohibits processing
manuscripts under review through an AI system. The writing plan should be
rebudgeted to eight total pages with references included.

## Retained implementation

### Campaign protocol

- `icra2027/prereg_icra.md` fixes the hypothesis, four arms, paired-seed unit,
  primary wall-clock meter, co-primary transition meter, secondary metrics,
  mandatory N-ablation, August 24 rule, and stopping rules. It remains a draft
  until the BARN asset and container hashes are filled and the file is reviewed
  and committed before the first full seed finishes.
- `icra2027/freeze_pool_split.py` consumes a course JSONL manifest and writes a
  deterministic difficulty-stratified train/held-out split with exact IDs,
  metadata, settings, and source SHA-256.
- `icra2027/analyze_campaign.py` computes paired common-budget AUC contrasts,
  paired bootstrap intervals, per-seed deltas, and exact sign-flip tests. It
  requires at least five full-domain paired seeds before making the August 24
  decision. Frozen analysis SHA-256:
  `4017958334fad6db74d594d2442ba18bffadd72adfd0412502875b5f467efdf3`.

### Teacher and allocation code

- Added a `UniformTeacher` whose posterior bookkeeping remains available for
  calibration diagnostics.
- Added `LearnabilityTeacher` with `p(1-p)`, sharing posterior, Thompson draw,
  floor, and concentration with the estimator-derived teacher. The code and
  prereg correctly call this a learnability baseline, not ALP-GMM.
- Added `StagedDifficultyTeacher`, a metadata-ordered easy-to-hard baseline with
  explicit promotion and floor behavior.
- Added exact discrete rollout water-filling using marginal value
  `p(1-p)^N`, with strict feasibility validation and exact budget conservation.

### Evaluation and smoke runner

- `GridReachSpace.evaluate_task` now uses per-task/per-episode held-out seeds
  and leaves the training RNG, episode counter, and simulator-step counter
  unchanged.
- `icra2027/navigation_campaign.py` runs all four arms with shared estimator and
  policy settings. It records target-uniform success, difficulty-bin success,
  easy retention, posterior calibration, dead groups, episodes, simulator
  steps, and training-only wall time.
- Every smoke artifact is stamped
  `engineering_smoke_not_paper_evidence`; the analysis refuses to treat it as
  a full BARN campaign.

## Smoke result: plumbing check only

One seed, 60 updates, four tasks/update, N=16, and 3,840 training episodes per
arm completed on the goal-conditioned grid adapter. These numbers must not
appear as ICRA empirical evidence.

| arm | episode-AUC | final held-out success | dead groups | sim steps |
|---|---:|---:|---:|---:|
| estimator-derived `u_16` | 0.3438 | 0.6133 | 14.6% | 39,964 |
| uniform | 0.3411 | 0.6055 | 18.8% | 38,940 |
| `p(1-p)` learnability | 0.2842 | 0.5078 | 16.7% | 29,754 |
| staged difficulty | 0.2497 | 0.3672 | 8.3% | 16,740 |

At paired common wall time, `u_16` minus uniform was +0.0100 AUC and minus
learnability was +0.0300, but minus staged was -0.0559. The matched-simulator-
step deltas were +0.0139, +0.0072, and -0.0628, respectively. This is a useful
protocol diagnostic: the staged arm's easier episodes make its common budget
much shorter, so its early concentration can win even while its fixed-episode
final is lower. The dual-budget analyzer preserved that unfavorable comparison
rather than reporting only the meter that favored `u_N`.

The saved analyzer verdict is correctly **not decision-ready**: one seed,
smoke domain, and the primary directional bar is not met.

Artifacts:

- `icra2027/results/navigation_smoke.json` — SHA-256
  `82bdf11f01d30fc5363c6ff42d955efcb91433709a3cc2f4f4bfb2e209367650`
- `icra2027/results/navigation_smoke_analysis.json` — SHA-256
  `1b9a5500843bb2747e5cb66818dcd023101de35f2f8c92141d283d4b599c87fb`

## Verification

- New ICRA campaign tests: **5/5 passed**.
- Existing `frontier_rl` and curriculum math/integration tests: **21/21
  passed** with the repository's required `PYTHONPATH=curriculum_maxrl`.
- Water-filling randomized invariant audit: **100/100 passed**.
- `python3 -m compileall -q frontier_rl icra2027`: passed.
- `git diff --check` on the retained scope: passed.

The bounded autoresearch keep criteria all passed. The changes were retained;
no discarded variant or failed code path remains in the campaign directory.

## Track A status

The E2c readiness-only driver was refreshed; no training job was launched.
All content-addressed code and asset integrity checks pass, held-out artifact
count remains zero, and B1 seed 3 is still the exact next stage. Launch remains
forbidden because GPU use was **10,263 MiB**, above the frozen **4,096 MiB**
ceiling. The receipt is
`autoresearch/iterate-260810-2240/E2C_LAUNCH_READINESS.json` (current SHA-256
`9bfb9980ed26ce55d44dda21be0ca12d3704ebb4bc0eabafc4ebc05e9b9d67ff`).

## Blockers and next actions

1. Supply or identify the BARN course manifest/assets and the lab's simulator
   launch/evaluation entry point. Required adapter outputs are only course ID,
   difficulty, binary success, simulator steps, and a rollout trajectory for
   the policy update.
2. Fill the BARN asset SHA-256 and container/image digest in
   `icra2027/prereg_icra.md`; generate and inspect `barn_split.json`; then
   commit the preregistration, analysis code, and split together.
3. Complete one end-to-end BARN seed for all four arms by August 17. If that
   cannot happen, activate the Isaac Lab fallback—but that runtime must first be
   installed or accessed elsewhere.
4. Launch the four-arm, five-paired-seed matrix. Do not add PLR, HER, or robot
   runs before the primary matrix is complete or safely underway.
5. Apply the August 24 gate exactly as frozen. A tie or win over both uniform
   and learnability permits the ICRA deadline; otherwise move to the RA-L
   timeline without forcing the result.
6. Keep polling E2c with the readiness-only command. Run the unchanged driver
   only after its own receipt reports authorization; do not schedule the ICRA
   campaign on that RTX 5090.

## Repository-safety note

The checkout began with extensive intentional user changes and untracked E2c
artifacts. No branch switch, merge, commit, reset, push, or publication was
performed. The nine-page ICLR release branch remains unreconciled exactly as
required by the existing branch handoff.
