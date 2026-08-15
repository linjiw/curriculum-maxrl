# MAZE-SCORE: neural-scale test of the deployed-$N$ score shape — preregistration

**Status:** DRAFT v2 — NOT AUTHORIZED FOR EVIDENCE SUBMISSION. The presence of
an executable array script or a staged bundle is not authorization to submit
seeds 20--49. While this document says DRAFT, only the non-evidence engineering
ladder below may run. Authorization begins only after its receipts, the final
outcome-blind sample count, retry policy, immutable source/environment hashes,
and analyzer hash are recorded together and this status is changed to FROZEN
in a clean commit. Changes after that point require a dated outcome-blind
amendment.
**Environment:** GMU Hopper, one `3g.40gb` A100 MIG slice per array task, using
the lock-hash-addressed environment created by `hopper/setup_maze_env.sh`
(torch 2.6.0+cu124, numpy 2.2.6; exact path/hash inserted at freeze).
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

**Candidate design: 30 fresh independent seed blocks, 20–49.** Waves 1 and 2
used blocks 0–11. Historical contrast SDs (.0077--.0135) imply that ten blocks
have only about 18--45% power for the +.005 SESOI; approximate 80% power spans
roughly 21--59 blocks. The exact count remains an outcome-blind DRAFT item until
one complete non-evidence arm establishes cost; no new endpoint value may be
used to select it.

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
   environment lock agree; record all hashes and the three smoke receipts; then
   change this document's status to FROZEN in a clean commit and stage that
   exact evidence bundle.
5. **Evidence launch:** only after step 4 may
   `hopper/sbatch/maze_score_array.sbatch` be submitted for the frozen evidence
   seeds. No partial endpoint log or JSONL is inspected until every frozen cell
   has a verified terminal artifact locally.

A failure or protocol-relevant change returns the process to the earliest
affected engineering step. It never authorizes bypassing the freeze gate. The
current 20--49 array and analyzer are a 30-block candidate implementation, not
an authorized evidence campaign while this document remains DRAFT.

## Endpoints

- **Primary:** paired (same block) `cov_auc_delta` contrast **`un` − `learn`**, where
  `cov_auc_delta` = the unweighted mean of mean-per-level observed-set pass@8
  at completed updates `{25,50,...,250}` minus the post-SFT value. Exactly ten
  checkpoint values enter; missing, duplicate, or extra timepoints invalidate
  the cell rather than being silently averaged.
- **Secondary (one test):** paired `un` − `unif` on the same metric.
- **Descriptive:** `learn` − `unif`; final-step `delta_cov8`; per-level bands.

## Candidate decision rule (becomes frozen with the campaign receipt)

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

(none)
