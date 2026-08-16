# MAZE-SCORE: neural-scale test of the deployed-$N$ score shape — preregistration

**Status:** FROZEN 2026-08-16 by the clean commit containing this line.
Changes after this point require a dated outcome-blind amendment section.
Evidence submission of seeds 20--67 is authorized only from the clean
content-addressed bundle staged from that commit, whose ID and manifest
SHA-256 are recorded in the campaign receipt and in `HOPPER_STATUS.md`.

**Freeze record (all inputs outcome-blind).**

| Item | Value |
|---|---|
| Launch-ladder step 1, CPU I/O smoke | job `9366532` COMPLETED exit 0, `hop055`; retrieved and digest-verified |
| Launch-ladder step 2, GPU import smoke | job `9366547` COMPLETED exit 0 in 34 s, `gpu020`, A100 `1g.10gb` |
| Launch-ladder step 3, full-arm cost smoke | job `9366552` COMPLETED exit 0 in 22:22, `gpu013`, peak GPU memory 39,672 MiB (seed 99, permanently excluded from inference) |
| Sample count | 48 blocks, seeds 20--67; justification `hopper/MAZE_SCORE_POWER_MEMO_2026-08-15.md` SHA-256 `1280d988e62fb28ebc5bb57fe2f4cf86c60b52761cb3742b1923ef2015526c1b` |
| Analyzer | `curriculum_maxrl/maze_score/analyze_maze_score.py` SHA-256 `197f1254f73826744206a256f904346de3ff9010a393d5d051a549ba5b7d7bd5` |
| Array script | `hopper/sbatch/maze_score_array.sbatch` SHA-256 `4522e1def80de0b6465e309f167096c19a9df8ca43fd4a6574558ea5345615e7` |
| Trainer | `curriculum_maxrl/maze_gpu/train.py` SHA-256 `835173fb7d83cf2dd689664c604fcee08974caadb375e10fa9cf8dc1fa38bf19` |
| Environment | `/scratch/lwang44/envs/maze-score-ad774d459fa77bb6`; lock `ad774d459fa77bb68c01c4a225db1e7faa3213216422eb5eabdf5b3c0e3d6224`; freeze `70d7f2c337b75de70adf941dacefdb7d3f7ba1772ac7f32821c896a61e77f36a`; environment JSON `42efa0bf38cc6d4aca56eac21559dfc989c92abda49e9eaf5df4fbcf019bf393` |
| Retry policy | A cell that reaches a non-`COMPLETED` terminal Slurm state may be resubmitted at most **once**, into a new attempt directory, only for an infrastructure cause identified without opening any result JSONL or metric-bearing stdout. The cause and both attempt IDs are recorded as an amendment before resubmission. A second failure of the same cell ends the campaign at the achieved block count, which is reported as such. No cell is ever rerun for a scientific reason. |
| Array throttle | `%5`; expected wall about 11.1 h at 1.1142 MIG-slice-h per block |
**Environment:** GMU Hopper, one `3g.40gb` A100 MIG slice per array task, using
the lock-hash-addressed environment created by `hopper/setup_maze_env.sh`
(torch 2.6.0+cu124, numpy 2.2.6; exact path plus lock, full-freeze, and
environment-JSON hashes inserted at freeze).
**Code:** the content-addressed bundle recorded in the campaign receipt. The
evidence path must use `train.py --protocol maze_score_v2`; the historical
`legacy_v1` path and `fact_analyze.py::read_run` are reproduction-only. The
primary is computed only by the frozen
`curriculum_maxrl/maze_score/analyze_maze_score.py` in that same bundle.
**Relationship to E2c / GATE-DR:** none. Different machine, different question,
no shared artifacts.

## Question this answers

The strongest live criticism of the ICLR draft is that the score's *positive* evidence
exists only at 640 parameters (the fixed-pool Acrobot tournament). The 1.26M-parameter
maze factorial tested an *estimator* ordering (MaxRL vs GRPO), never the *score shape*.
So the paper currently supports "rollout-aware difficulty targeting helps at 640
parameters" and nothing larger.

**This study runs the Acrobot contrast at ~2,000x the parameter count:** does sampling
by the deployed-$N$ utility $u_N(p) = (1-(1-p)^N) - p$ beat its $N=2$ slice $p(1-p)$,
and uniform, on time-integrated coverage, with a 1.26M-parameter transformer at the
deployed $N=32$?

## Design

Three arms, estimator fixed to `maxrl` throughout (this is a score test, not an
estimator test):

| arm | `--teacher` | score |
|---|---|---|
| `un` | `frontier_un` | $u_{32}(p) = (1-(1-p)^{32}) - p$ |
| `learn` | `learnability` | $p(1-p)$ (the $N=2$ slice; SFL-style) |
| `unif` | `uniform` | none |

All three share the identical Beta posterior (decay 0.7), Thompson mechanism,
0.15 uniform floor, and level set; the *only* algorithmic difference between
`un` and `learn` is the utility function. The frozen contract test must dispatch
`frontier_un` to effective exponent 32 and `learnability` to exponent 2.

**Frozen design: 48 fresh independent seed blocks, 20–67.** Waves 1 and 2
used blocks 0–11. The count was fixed outcome-blind in
`hopper/MAZE_SCORE_POWER_MEMO_2026-08-15.md` from three inputs only — the
preregistered SESOI, the historical contrast SD range (.0077--.0135) already
recorded in this document, and the measured full-arm cost from engineering job
9366552. No endpoint, telemetry, or checkpoint value was opened.

**Powered-for effect: +.0075, i.e. 1.5x the SESOI.** Because the decision rule
requires the *observed* mean to clear the SESOI, a true effect of exactly
+.005 is a coin flip at every sample size (45.7% at n=30, 50.2% at n=72); the
SESOI is the *reporting* threshold, not the detection threshold. At the
pessimistic SD (.0135) and Holm's worst case, 48 blocks give 90.0% power at
+.0075, against 86.2% at 40 and 94.1% at 72. 48 is the last count at which the
preregistered *exact* sign-flip enumeration remains feasible (268 MB; 60 blocks
would need 17 GB and 72 blocks 1.1 TB), so larger designs would silently
substitute a sampled approximation for the exact randomization test. That
instrument change was judged a worse cost than the 4.1 power points forgone.

Within a block, all arms load the identical content-addressed SFT checkpoint.
Its hash is recorded in every run and checked by the analyzer. Creating versus
loading that checkpoint must be followed by the same phase-specific resets for
Python task generation, NumPy, rollout sampling, and evaluation sampling.
Selectors necessarily choose different level sequences, so the pairing claim
is shared initialization, per-level RNG construction, evaluation panel, and
seed—not an identical realized training-task sequence.

The SFT checkpoint is prepared before any arm runs. Arm execution order is
then counterbalanced deterministically by seed block and recorded in
`meta/arm_order.txt`:

| `seed mod 3` | first | second | third |
|---:|---|---|---|
| 0 | `un` | `learn` | `unif` |
| 1 | `learn` | `unif` | `un` |
| 2 | `unif` | `un` | `learn` |

The 30 consecutive candidate seeds contain ten blocks in each residue class,
so every arm occupies every process position exactly ten times. This order is
part of the frozen protocol; retries preserve the original seed and therefore
the original order.

Fixed optimizer/training protocol: `--steps 250 --eval-every 25 --lr 1e-4
--rollouts 32 --tasks-per-step 8 --sft-steps 600 --d-model 128 --n-layers 6`,
no hindsight. The v2 loop evaluates after exactly 25, 50, ..., 250 completed
updates, once each; it does not emit the historical first-update or duplicate
final records. Independent unit = seed/warmstart/evaluation block. Arms within
a block are paired observations, never independent replicates.

The primary evaluation panel is fresh relative to development. For block `s`,
the exact task-panel seed is `eval_tasks(s) = 202608130 + s`; it generates 32
held-out mazes per level that are never used for training or teacher updates.
Eight samples per maze support standard observed-set pass@8. The exact
evaluation-sampling base is `eval_samples(s) = 302608130 + s`. The post-SFT
evaluation uses that base, and the evaluation after `t` completed updates uses
a newly constructed generator seeded `eval_samples(s) + t`, for
`t in {25,50,...,250}`. Evaluation cannot advance the training or teacher RNG.
SFT, per-level RL task generation, rollout sampling, teacher Thompson draws,
and evaluation use separate phase-specific generator objects. The JSONL config
records both seed formulas, and the analyzer rejects any mismatch. The
repeatedly used seed-12345 panel is not a confirmatory endpoint.

## Safe launch ladder and freeze gate

The following order is mandatory. Completion means a terminal Slurm success,
local retrieval, and verification of each smoke's receipt and SHA-256 manifest;
queue submission alone is not completion.

1. **CPU I/O smoke:** submit `hopper/sbatch/workflow_io_smoke.sbatch` to verify
   Slurm submission, unique result directories, terminal accounting, manifest
   creation, retrieval, and local digest verification without allocating a GPU.
2. **GPU import smoke:** using the candidate immutable engineering bundle and
   lock-hash-addressed environment, submit
   `hopper/sbatch/maze_gpu_import_smoke.sbatch`. It may import code and report
   CUDA/runtime metadata, but it must not train or produce an endpoint.
3. **One full-arm cost smoke:** submit
   `hopper/sbatch/maze_full_arm_smoke.sbatch` on the reserved non-evidence seed
   99. It runs one complete 250-update `un` arm solely to establish wall time,
   memory, output shape, and retrieval cost. Seed 99 is permanently excluded
   from inference. Its scientific endpoint is not used to choose sample count,
   thresholds, hypotheses, or settings; only engineering cost/completeness
   fields may be inspected.
4. **Outcome-blind freeze:** use only the full-arm engineering cost receipt and
   pre-existing variance information to finalize the block count. Before any
   evidence submission, make the preregistration, analyzer seed set, Slurm
   array bounds/order, retry policy, clean content-addressed source bundle, and
   environment lock/full-freeze/JSON hashes agree; record all hashes and the
   three smoke receipts; then
   change this document's status to FROZEN in a clean commit and stage that
   exact evidence bundle.
5. **Evidence launch:** only after step 4 may
   `hopper/sbatch/maze_score_array.sbatch` be submitted for the frozen evidence
   seeds. No partial endpoint log or JSONL is inspected until every frozen cell
   has a verified terminal artifact locally.

A failure or protocol-relevant change returns the process to the earliest
affected engineering step. It never authorizes bypassing the freeze gate.
Steps 1--3 completed on 2026-08-14 (jobs 9366532, 9366547, 9366552) and step 4
completed on 2026-08-16; the 20--67 array and analyzer are the frozen 48-block
evidence implementation as of the commit carrying the FROZEN status above.

## Endpoints

- **Primary:** paired (same block) `cov_auc_delta` contrast **`un` − `learn`**, where
  `cov_auc_delta` = the unweighted mean of mean-per-level observed-set pass@8
  at completed updates `{25,50,...,250}` minus the post-SFT value. Exactly ten
  checkpoint values enter; missing, duplicate, or extra timepoints invalidate
  the cell rather than being silently averaged.
- **Secondary (one test):** paired `un` − `unif` on the same metric.
- **Descriptive:** `learn` − `unif`; final-step `delta_cov8`; per-level bands.

## Decision rule (frozen 2026-08-16)

For each paired contrast, compute a two-sided 95% paired percentile-bootstrap
interval (10,000 block resamples, seed 20260813) and an exact two-sided
sign-flip randomization p-value for the block-level mean. Holm-adjust the two
p-values for `{un-learn, un-unif}`. Support requires **all three**:

1. Holm-adjusted p-value `< .05`;
2. bootstrap lower bound `> 0`; and
3. point estimate at least **+.005** cov-AUC (SESOI, set at roughly a quarter
   of the wave-2 MaxRL−GRPO block-level effect of +.0195).

Sign counts remain descriptive. The analyzer must implement these rules before
any evidence file exists and must refuse campaign/source/config/warmstart
mismatches, partial blocks, NaNs, and nonterminal attempt files.

## Outcome branches (all written before running)

- **Supported:** the paper gains a neural-scale positive for the score shape; the
  Evidence-section sentence stating the score's support is small-scale by design is
  replaced by a statement of the tested scale, and Contribution 2 cites both scales.
- **Practically ruled out:** only if the primary interval upper bound is below
  +.005. State that effects at or above the SESOI were ruled out under this
  protocol; do not claim universal non-transfer.
- **Inconclusive:** every other nonsupport outcome. Report the point estimate
  and interval at equal prominence and retain the existing calibrated
  small-scale claim. Do not convert failure to reject into evidence of absence.

Every branch is reported. No re-running with altered settings, post-hoc metric
substitution, or seed extension is allowed. No claim extends beyond this task
family, model, and scale.

## Artifacts

Each immutable attempt retains the result JSONL, per-step delivery telemetry
(selected levels, group K, realized coefficient mass, silent groups, optimizer
update), content-addressed warmstart, source/environment/prereg hashes, stdout,
and Slurm accounting. Stdout contains progress only, never endpoint values.
After all expected cells are terminal, fetch into a new local campaign
directory, verify its SHA-256 manifest and completeness, then run
`curriculum_maxrl/maze_score/analyze_maze_score.py`. No endpoint log or JSONL is
inspected before the entire frozen matrix is present.

Infrastructure failures retain their original attempt directories. A retry is
allowed only for scheduler/node/I/O failure with the same seed, source,
environment, warmstart, and arguments, recorded before endpoint inspection.
OOM, timeout caused by the declared resource limit, NaN, or a scientific-code
exception is a failed/incomplete block unless a dated outcome-blind amendment
governs the whole campaign.

## Amendments

- **2026-08-16a (submission chunking; outcome-blind, no endpoint opened).**
  The frozen script declares `#SBATCH --array=20-67%5`, but Hopper's `gpu` QOS
  caps *submitted* jobs at 40 per user (`MaxSubmitJobsPU=40`, `MaxJobsPU=20`),
  so a single 48-task array is rejected at submit time with
  `QOSMaxSubmitJobPerUserLimit` before any allocation. Discovered with an empty
  queue, so it is a hard policy limit rather than transient contention.
  The campaign is therefore submitted as two chunks of the **same frozen
  script, bundle, environment, campaign ID, and attempt ID**, overriding only
  the array range: seeds 20--43 and 44--67. Their union is exactly the frozen
  48-block seed set, the in-script seed guard still refuses anything outside
  20--67, and no per-cell setting changes. The cap applies to submitted, not
  running, jobs, so the two 24-task chunks cannot even coexist in the queue:
  chunk 2 is submitted only once chunk 1 has drained to 16 or fewer tasks.
  Concurrency therefore stays at the frozen `%5` and total wall time is
  unchanged from the frozen single-array estimate; only the submission is
  split. Retrieval merges both array IDs into the single campaign
  matrix before analysis, and the analyzer's completeness check over
  `EXPECTED_SEEDS` remains the authority on whether the matrix is whole.

(none)
