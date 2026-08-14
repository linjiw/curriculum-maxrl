# ICRA 2027 BARN navigation campaign

This directory contains the outcome-blind, CPU-only BARN/Gazebo evidence
pipeline governed by `CODEX_GOAL_ICRA_2026-08-11.md` and
`icra2027/prereg_icra.md`.

## Current state

The real 300-course BARN backend, immutable manifest and split, production
runner, strict blind merger, and transition-primary analyzer are implemented.
The compute-node dataset preparation (job 9366817) and outcome-blind,
training-only timing smoke (job 9366831) passed. Their immutable receipts and
the resulting 36-hour scheduler safety limit are bound into the frozen
preregistration and `barn_protocol.json`. No BARN paper endpoint was launched
or inspected before freeze.

Stable inputs:

- official archive SHA-256:
  `5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320`;
- 300-course manifest SHA-256:
  `1015a6a48ef44add7224200da2ace1cd6c8d7780275b30d7266a44dc88e9ec61`;
- 240/60 split SHA-256:
  `c0ed1d7024ebc240d96a023efb6a124e879fdb06d0342a5e5de7b6d6ed07d7d7`;
- CPU ROS 2 Humble / Gazebo Classic SIF SHA-256:
  `cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0`;
- Hopper dataset-preparation receipt SHA-256:
  `216408ddfb6ef95c6d7cc912608aac0428240d09a562f20b03069408b1a9d76f`;
- Hopper training-only smoke receipt SHA-256:
  `d9d251c819bbf602dae6c829e3c6755b514639f2fa1c3c9f83cd5b13d21c8738`.

## Protocol

- Primary: `ours_uN`, `uniform`, `learnability`, and `staged`, paired seeds
  1--5, N=8, 1,000,000 Gazebo steps per arm.
- Primary endpoint: held-out target-uniform success AUC interpolated at the
  exact transition budget. Wall time is descriptive.
- Gate comparators: uniform and learnability only; staged is reported.
- Mandatory ablation: fresh two-arm cells at N=2, 4, and 16; N=8 reuses the
  identical primary ours/learnability rows.
- All engineering smokes are restricted to training course `barn-299` and are
  permanently non-evidentiary.
- Evidence jobs request 36 hours on Hopper `normal`; this is only a scheduler
  kill limit. The 1,000,000-step budget and all scientific stopping rules stay
  fixed.

The machine-readable contract pins exact parameters, seed-specific balanced
execution order, disjoint ROS/Gazebo IDs, artifact completeness, and retry
selection. Full runs require seven hashes: manifest, split, preregistration,
analyzer, machine protocol, container, and source bundle.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=curriculum_maxrl:. \
  python3 -m pytest -q \
  icra2027/test_campaign.py \
  icra2027/test_barn_campaign.py \
  icra2027/test_verify_barn_smoke_package.py \
  icra2027/test_merge_barn_campaign.py \
  icra2027/test_select_barn_attempts.py

bash hopper/test_stage_barn_local.sh
bash hopper/test_submit_barn_campaign_local.sh
bash hopper/test_finalize_barn_ledger_local.sh
bash hopper/test_finalize_barn_campaign_local.sh
bash -n hopper/sbatch/barn_*.sbatch \
  hopper/stage_barn_campaign.sh \
  hopper/submit_barn_campaign.sh \
  hopper/finalize_barn_ledger.sh \
  hopper/finalize_barn_campaign.sh
git diff --check
```

## Engineering stage

Stage a DRAFT engineering closure and copy the four printed values into the
corresponding shell variables:

```bash
bash hopper/stage_barn_campaign.sh engineering

export BARN_SOURCE_BUNDLE_DIR=...
export BARN_SOURCE_SHA256=...
export BARN_DATASET_ARCHIVE=...
export BARN_DATASET_ARCHIVE_SHA256=...
```

Submit the CPU-only dataset preparation job, wait for it to finish, and then
submit the CPU-only one-update smoke:

```bash
./hopper/hopper.sh submit \
  hopper/sbatch/barn_dataset_prepare.sbatch \
  --export=ALL,BARN_SOURCE_BUNDLE_DIR="$BARN_SOURCE_BUNDLE_DIR",BARN_SOURCE_SHA256="$BARN_SOURCE_SHA256",BARN_DATASET_ARCHIVE="$BARN_DATASET_ARCHIVE",BARN_DATASET_ARCHIVE_SHA256="$BARN_DATASET_ARCHIVE_SHA256"
PREP_JOB_ID=...  # integer printed after "submitted"
./hopper/hopper.sh watch "$PREP_JOB_ID" 30 86400

./hopper/hopper.sh submit \
  hopper/sbatch/barn_training_smoke.sbatch \
  --export=ALL,BARN_SOURCE_BUNDLE_DIR="$BARN_SOURCE_BUNDLE_DIR",BARN_SOURCE_SHA256="$BARN_SOURCE_SHA256",BARN_DATASET_ARCHIVE="$BARN_DATASET_ARCHIVE",BARN_DATASET_ARCHIVE_SHA256="$BARN_DATASET_ARCHIVE_SHA256",BARN_TRAIN_COURSE=barn-299,BARN_SMOKE_SEED=20270811
SMOKE_JOB_ID=...  # integer printed after "submitted"
./hopper/hopper.sh watch "$SMOKE_JOB_ID" 30 86400
```

Both jobs publish only integrity, timing, and completion receipts. Jobs
9366817 and 9366831 are the receipts used for the frozen campaign. The smoke
verifier reads only the prepared package controls and the selected
`barn-299` declarations; the runner alone hashes and loads that train course.
The metric-bearing internal smoke artifact is destroyed on the compute node.

## Evidence stage and exact four-cell launch

The preregistration and machine protocol must be frozen and the full BARN
closure must match one commit before evidence staging. Never reuse the
engineering source values for evidence:

```bash
bash hopper/stage_barn_campaign.sh evidence

CAMPAIGN_ID=barn-icra2027-001
SOURCE_BUNDLE=...  # printed BARN_SOURCE_BUNDLE_DIR
SOURCE_SHA256=...  # printed BARN_SOURCE_SHA256
```

With the same `CAMPAIGN_ID`, `SOURCE_BUNDLE`, and `SOURCE_SHA256`, submit the
exact four preregistered cells through the source-bound wrapper. Every attempt
ID is unique; there is no fresh N=8 ablation because it reuses the primary
cell rows.

```bash
bash hopper/submit_barn_campaign.sh \
  "$CAMPAIGN_ID" primary primary-attempt-001 \
  "$SOURCE_BUNDLE" "$SOURCE_SHA256"
bash hopper/submit_barn_campaign.sh \
  "$CAMPAIGN_ID" ablation_n2 n2-attempt-001 \
  "$SOURCE_BUNDLE" "$SOURCE_SHA256"
bash hopper/submit_barn_campaign.sh \
  "$CAMPAIGN_ID" ablation_n4 n4-attempt-001 \
  "$SOURCE_BUNDLE" "$SOURCE_SHA256"
bash hopper/submit_barn_campaign.sh \
  "$CAMPAIGN_ID" ablation_n16 n16-attempt-001 \
  "$SOURCE_BUNDLE" "$SOURCE_SHA256"
```

Each call records and installs a five-seed ledger block before releasing its
held Slurm array. Monitor only the returned job IDs with `hopper.sh status` or
`hopper.sh watch`. Do not fetch seed artifacts, open raw endpoint logs, run a
selector on a subset, or inspect partial analysis. If a cell needs a retry,
resubmit that whole five-seed cell with a new attempt ID.

If a submission reports that its array remains held, rerun the exact same
campaign/cell/attempt/source command. The durable pending record resumes that
specific transaction; a different attempt is refused until it is resolved.
After `BARN_CAMPAIGN_SEALED`, the campaign ID is permanently closed to new
attempts.

## Outcome-blind finalization and retrieval

After every recorded array task is terminal, and after any scheduler-visible
failure has been retried as a whole cell, finalize the exact four-cell ledger:

```bash
bash hopper/finalize_barn_ledger.sh \
  "$CAMPAIGN_ID" "$SOURCE_BUNDLE" "$SOURCE_SHA256"

FINALIZED_LEDGER_SHA256=...  # copy sha256= from BARN_LEDGER_FINALIZED
bash hopper/finalize_barn_campaign.sh \
  "$CAMPAIGN_ID" "$SOURCE_BUNDLE" "$SOURCE_SHA256" \
  "$FINALIZED_LEDGER_SHA256"
```

The first command checks exact cell coverage, complete five-seed attempt
groups, terminal Slurm state, and closed artifacts without exposing outcomes.
The second submits the bundled CPU-only Slurm postprocessor, waits using
accounting metadata, performs all four frozen selections and merges plus both
analyses, seals one checksum-closed all-cell package, and fetches only that
package. Treat `BARN_CAMPAIGN_SEALED` as the sole successful retrieval signal;
do not substitute `hopper.sh fetch`, raw-log access, or local partial
postprocessing for this transaction.
