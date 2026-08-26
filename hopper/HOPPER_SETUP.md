# GMU Hopper runbook

**Last verified:** 2026-08-20

**Login:** `lwang44@hopper.orc.gmu.edu`

**Scratch root:** `/scratch/lwang44`

**Live state and launch gates:** [`HOPPER_STATUS.md`](HOPPER_STATUS.md)

This is the operational path for staging, submitting, monitoring, and
retrieving this project's Hopper jobs. Run commands from the repository root
on the lab machine unless a command explicitly says otherwise. Do not run
training on a Hopper login node.

## Operator quick start

The wrapper defaults to the login and scratch root above. It stages an
immutable copy of every submitted sbatch file and records local and remote
receipts.

```bash
# Verify SSH, Slurm, partitions, and scratch access.
./hopper/hopper.sh health

# Submit an audited sbatch file. Keep resource settings inside the file.
./hopper/hopper.sh submit hopper/sbatch/workflow_io_smoke.sbatch

# View all live jobs, or one job's queue and accounting state.
./hopper/hopper.sh status
./hopper/hopper.sh status JOB_ID

# Wait for terminal accounting without reading scientific output.
./hopper/hopper.sh watch JOB_ID 120 604800

# Inspect marker-only progress for a blinded multi-block campaign.
./hopper/hopper.sh campaign-status \
  /scratch/lwang44/maxrl/CAMPAIGN/attempts/attempt-001 EXPECTED_BLOCKS \
  /scratch/lwang44/maxrl/CAMPAIGN/incomplete/attempt-001

# Retrieve a terminal payload to a new path and verify its digest/manifest.
./hopper/hopper.sh fetch REMOTE_PATH NEW_LOCAL_PATH
```

Use `logs JOB_ID` only for engineering jobs whose streams are known not to
contain scientific endpoints. Maze evidence logs require the explicit
`--allow-endpoints` acknowledgement. Frozen group-law-flip logs are always
sealed: their only unblinding route is the experiment-specific complete-matrix
retrieval validator followed by the single-use analyzer.

The 2026-08-20 end-to-end wrapper regression used CPU I/O smoke job `9424207`:
it completed `0:0`, produced a terminal accounting receipt, and fetched with a
manifest-verified tree digest
`301db5dd484cfbf7217af0072aa38134410c228bb5ea9c58fd1e29359511041c`.
The verified engineering-only copy and its fetch/terminal receipts are at
`/data/robotixx/hopper_smokes/9424207`. This smoke also verified Hopper's valid
zero-second accounting form with a blank separate `StdErr` field.

## Non-negotiable boundaries

- E2c stays on its frozen local runtime; it never runs on Hopper.
- `/scratch/lwang44` is temporary, unbacked working storage. Retrieve and
  verify important results only through the retrieval transaction authorized
  by that workflow.
- A smoke result is engineering evidence only. It never enters a paper
  endpoint or a scientific run registry.
- Do not submit MAZE-SCORE or BARN evidence while its gate is `HOLD` in
  `HOPPER_STATUS.md`.
- For a blinded campaign, inspect scheduler state, resource use, hashes,
  manifests, and completion markers only. Do not inspect partial result JSONL,
  metric-bearing stdout, or analyzer output.

## Official Hopper contract used here

The wrapper follows the GMU ORC [Hopper Quick Start
Guide](https://wiki.orc.gmu.edu/mkdocs/Hopper_Quick_Start_Guide/), [Slurm
guide](https://wiki.orc.gmu.edu/mkdocs/Getting_Started_with_Slurm/), [GPU job
guide](https://wiki.orc.gmu.edu/mkdocs/Running_GPU_Jobs/), [storage
guide](https://wiki.orc.gmu.edu/mkdocs/Storage_on_the_Cluster/), and [best
practices](https://wiki.orc.gmu.edu/mkdocs/Hopper_Best_Practices_QA/).

- Submit batch jobs from a login node with `sbatch`; compute belongs in a
  Slurm allocation.
- CPU work uses `normal`. GPU work uses `gpuq`, `qos=gpu`, and an explicit
  `gres`.
- ORC's recommended single-job maxima are 2 CPUs/15 GB for `1g.10gb` and
  8 CPUs/60 GB for `3g.40gb`. The project templates match those values.
- `contrib-gpuq` is preemptible for non-contributors. It is not an evidence
  default and requires a preregistered retry policy before use.
- The ORC pages currently disagree on partition maxima: Quick Start lists
  `normal=3 days` and `gpuq=1 day`, while the general Slurm page lists longer
  limits. Project scripts stay at or below the stricter values. Check live
  partition state before requesting any increase.
- ORC documents `/home` as 60 GB and backed up. `/scratch` is unbacked, has a
  100-million-file limit, and purges files older than 90 days. Store active
  runs on scratch, then copy verified artifacts to persistent project/local
  storage.

The ORC docs normally expose scratch through `$SCRATCH`, but it was empty in
this account's login audit. Project tooling therefore uses the explicit,
validated path `/scratch/lwang44`.

## Wrapper setup and local validation

The defaults already select the correct account. These exports are optional
and make the target explicit:

```bash
export HOPPER_HOST=lwang44@hopper.orc.gmu.edu
export HOPPER_SCRATCH=/scratch/lwang44
```

Before changing or using the wrapper:

```bash
bash -n hopper/hopper.sh hopper/sbatch/*.sbatch
python3 -m pytest \
  curriculum_maxrl/maze_gpu/test_train_protocol.py \
  curriculum_maxrl/maze_score/test_analyze_maze_score.py -q
./hopper/test_hopper_local.sh
git diff --check
```

`test_hopper_local.sh` mocks SSH, Slurm, and transfer operations; it does not
contact Hopper.

Read-only connection and scheduler check:

```bash
./hopper/hopper.sh health
./hopper/hopper.sh status
./hopper/hopper.sh registry
```

The wrapper uses noninteractive SSH with a connection timeout and keepalive.
It intentionally leaves SSH/Slurm stderr visible so authentication, quota, and
scheduler failures are not mistaken for success.

## MAZE environment and source staging

The verified engineering environment is:

```text
/scratch/lwang44/envs/maze-score-ad774d459fa77bb6
lock SHA-256: ad774d459fa77bb68c01c4a225db1e7faa3213216422eb5eabdf5b3c0e3d6224
freeze SHA-256: 70d7f2c337b75de70adf941dacefdb7d3f7ba1772ac7f32821c896a61e77f36a
environment JSON SHA-256: 42efa0bf38cc6d4aca56eac21559dfc989c92abda49e9eaf5df4fbcf019bf393
```

`hopper/setup_maze_env.sh` creates the lock-prefix path from
`hopper/requirements-maze-hopper.lock`, uses scratch for Conda/pip caches,
performs `pip check`, and writes `LOCK_SHA256`, `ENVIRONMENT.freeze`, and
`ENVIRONMENT.json`. Before evidence, treat those receipts as immutable and bind
all three hashes into the campaign record; do not update packages in place.

Stage a complete candidate source bundle with:

```bash
bash hopper/stage_maze_score.sh engineering
```

The command prints the bundle ID, complete-manifest SHA-256, remote path,
source mode, and whether an identical bundle was reused. Engineering mode
records a dirty worktree and cannot be used for evidence. Evidence mode also
requires a clean committed tree and a preregistration whose literal status is
`FROZEN`; otherwise it exits before upload.

The candidate verified by GPU job 9366547 is bundle `f4359095fb05490192b4`,
manifest `f4359095fb05490192b404ea03f9fc2413fc7fcd97b20571855b1c38160eaf80`.
Do not silently substitute these engineering values into a later evidence
campaign; stage the clean frozen bundle and record its new values.

## What `submit` records

```bash
./hopper/hopper.sh submit hopper/sbatch/workflow_io_smoke.sbatch
```

Before calling `sbatch --parsable`, the wrapper:

1. validates all arguments and confines remote paths to
   `/scratch/lwang44`;
2. parses `#SBATCH --output`, expands `%u`, and creates its parent directory;
3. stages a unique script name containing its SHA-256 prefix and UTC stamp;
4. verifies the staged script hash; and
5. writes local and remote TSV receipts containing job ID, UTC time, host,
   local/remote script paths and hashes, stdout path, and exact sbatch
   arguments.

The local append-only receipt registry is `hopper/.job_registry`; remote
receipts are under `/scratch/lwang44/maxrl/receipts/`. Never replace a staged
script or reuse a scientific attempt directory.

## Status and endpoint-blind monitoring

Set `JOB_ID` to the integer printed by `submit`:

```bash
JOB_ID=9366532
./hopper/hopper.sh status "$JOB_ID"
./hopper/hopper.sh watch "$JOB_ID" 30 3600
```

`status` reports both `squeue` and `sacct`, including state, exit code,
elapsed/time limit, CPU/memory request, node, and submit/start/end times.
`watch` polls without opening stdout and returns failure for a non-completed
terminal state or a timeout. Its completed-state path was verified live
against job 9366532. Use `status` for the final accounting receipt.

For MAZE-SCORE and full-arm smoke names, `logs` refuses access unless
`--allow-endpoints` is explicit:

```bash
./hopper/hopper.sh logs "$JOB_ID" 80 --allow-endpoints
```

That flag is an acknowledgement, not routine permission. Do **not** use it
during a partial evidence matrix. Permitted blind monitoring is limited to
`status`, `watch`, resource telemetry, immutable hashes, and file/completion
counts that reveal no metric. An infrastructure incident that genuinely
requires unblinding must be recorded before opening the log and may invalidate
the affected scientific attempt. This MAZE acknowledgement never authorizes
opening a BARN evidence log.

Cancel a bad queued/running job directly with Slurm, then preserve its receipt:

```bash
ssh -o BatchMode=yes lwang44@hopper.orc.gmu.edu scancel "$JOB_ID"
./hopper/hopper.sh status "$JOB_ID"
```

## Non-evidence and MAZE retrieval

`fetch` accepts only a scratch path, requires a destination that does not yet
exist, verifies a remote `SHA256SUMS` when present, transfers into a hidden
partial destination, recomputes a file/tree digest locally, and only then
renames it to the requested destination.

For a non-evidence I/O smoke:

```bash
JOB_ID=9366532
RESULT_DEST="autoresearch/iterate-260813-2348/hopper_smoke/job-$JOB_ID"
./hopper/hopper.sh fetch \
  "/scratch/lwang44/maxrl/tests/results/$JOB_ID" \
  "$RESULT_DEST"
./hopper/hopper.sh fetch \
  "/scratch/lwang44/maxrl/tests/logs/maxrl-io-smoke_$JOB_ID.out" \
  "autoresearch/iterate-260813-2348/hopper_smoke/job-$JOB_ID.stdout.log"
```

The example above is non-evidence only. MAZE evidence must follow its own
frozen campaign retrieval contract after the complete matrix is terminal.
BARN evidence must not use `hopper.sh fetch` or `hopper.sh logs` at all; its
only outcome-bearing retrieval path is the sealed all-cell transaction in
[BARN source-bound workflow](#barn-source-bound-workflow).

## MAZE safe launch ladder

This ladder applies to MAZE only. Advance one rung only after its artifact and
terminal accounting are saved.

| Rung | Work | Resource | Required pass artifact |
|---|---|---|---|
| 0 | Local syntax, contract/analyzer tests, wrapper mock test | local | all tests pass |
| 1 | `hopper.sh health` | login/read-only | SSH, Slurm tools, and scratch writable |
| 2 | `workflow_io_smoke.sbatch` | `normal`, 1 CPU, 1 GB, 5 min | fetched `receipt.tsv`, `SHA256SUMS`, `COMPLETE`, exit 0 |
| 3 | Immutable source and lock-addressed environment | login/staging only | source manifest, clean source state, lock hash, `pip check` |
| 4 | `maze_gpu_import_smoke.sbatch` | `1g.10gb`, 2 CPUs, 15 GB, 10 min | CUDA/import/runtime receipt fetched and verified |
| 5 | `maze_full_arm_smoke.sbatch`, seed 99 only | `3g.40gb`, 8 CPUs, 60 GB | full-cost schema/runtime receipt; no endpoint analysis |
| 6 | Freeze protocol and campaign receipt | no compute | fixed sample count, retry rule, source/env/prereg/analyzer hashes |
| 7 | Evidence array | frozen request only | complete terminal matrix; no partial inspection |
| 8 | Retrieve, verify, then run the frozen MAZE analyzer | local | one complete MAZE campaign and analyzer report |

Rungs 0--5 are complete; engineering job 9366552 passed the full-arm cost
smoke. Rungs 6--8 remain blocked; see `HOPPER_STATUS.md`.

An evidence submission, once every HOLD is cleared, must pass the immutable
bundle/environment values as recorded `--export` arguments. The shape is:

```bash
./hopper/hopper.sh submit hopper/sbatch/maze_score_array.sbatch \
  --export=ALL,MAZE_BUNDLE_DIR=/scratch/lwang44/maxrl/bundles/maze_score/BUNDLE_ID,MAZE_SOURCE_MANIFEST_SHA256=SOURCE_SHA256,MAZE_ENV_DIR=/scratch/lwang44/envs/maze-score-LOCK_PREFIX,MAZE_ENV_LOCK_SHA256=LOCK_SHA256,MAZE_ENV_FREEZE_SHA256=FREEZE_SHA256,MAZE_ENV_JSON_SHA256=ENV_JSON_SHA256,MAZE_CAMPAIGN_ID=CAMPAIGN_ID,MAZE_ATTEMPT_ID=ATTEMPT_ID
```

This is a template, not authorization. Literal placeholder values must never
be submitted, and the current `HOLD` forbids running it.

## UED engineering ladder and full-campaign hold

The UED path is independent of the MAZE evidence ladder. Its current pinned
engineering environment is
`/scratch/lwang44/envs/ued-minimax-v2-9ab83896f41c5294-dbd0494789fd70b8`;
the exact source bundle verified by jobs 9366896 and 9366897 is
`/scratch/lwang44/maxrl/bundles/ued_minimax/6c2ca94ca8109be2775c`, manifest
`6c2ca94ca8109be2775ce0f166e11f064466e4aaa3c2efb085587a0d3f13e93d`.
Never mix a prerequisite from one bundle manifest with a later bundle.
The independently audited successor bundle is bundle ID
`06ffeeeb6998e8ddb1ce`, manifest
`06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff`.
It is remotely staged, and fresh exact-bundle import/JIT job `9367063` is
queued. Do not inspect its streams or result tree while pending/running. It
still needs a terminal verified import receipt and then a fresh one-update rung
before the terminal-chain smoke can be submitted.

| Rung | Work | Required terminal artifact |
|---|---|---|
| 0 | Local overlay/formula/grouped-runner/RNG and staging tests | all focused tests and two deterministic bundle builds pass |
| 1 | Exact-bundle import/formula/one-JIT smoke | terminal `COMPLETED 0:0`, fetched `COMPLETE` and verified manifest |
| 2 | Exact-bundle grouped one-update smoke | one PPO update, `n_grad_updates=1`, five optimizer applications, checkpoint reload and zero group-integrity errors |
| 3 | Frontier terminal-chain engineering smoke | phase A: one PPO update plus actual external 30-episode evaluation and atomic `COMPONENTS_COMPLETE`; phase B after terminal `sacct`: atomic engineering package with `analyzer_eligible=false` |
| 4 | Development campaign preparation | **HOLD** until source/package/protocol/campaign contracts are frozen and a new exact-bundle ladder passes |
| 5 | Five-seed paired-arm development campaign and analyzer | **HOLD**; never inferred from smoke success |

The tie-aware v4 path is not part of this executable ladder. Its latest bounded
remote-contract snapshot (`da74eb3e0debc7781d6d`) is a deterministic local
HOLD artifact and deliberately has no submit operation. Independent audit
requires a new identity after fixing protected-overlay compatibility, R2
`job-<id>` identity, closed-environment GPU probing, and Hopper MIG accounting.
Do not adapt the v3 commands below to v4 or stage that snapshot manually.

Job 9366896 completed rung 1 in 45 seconds on `gpu021`. Its fetched result
manifest is
`3a15f52ddb0aa0b44f190f9701183c51884b91a0f1d850f327a53c3208f2a14c`
and its verified local tree is
`/data/robotixx/ued_bench/hopper/import-smoke-job-9366896/`. Job 9366897
completed rung 2 in 1:42 on `gpu021`; its result manifest is
`4eaa676052cbc9006da1d285b03eda354cab27f3b7d72064b5138724c83691c8`
and its verified local tree is
`/data/robotixx/ued_bench/hopper/one-update-job-9366897/`.

Monitor every UED job with `status`/`watch` only until it is terminal. Do not
open or fetch its component tree or Slurm streams while it is pending or
running. After clean terminal accounting, fetch into destinations that do not
exist, then verify the complete outer manifest, nested prerequisites, closure,
provenance, counters, checkpoint receipt, and terminal `sacct`. An engineering
terminal-chain package uses an explicit two-phase handoff. Phase A is the Slurm
job: it uses the pinned environment Git, trains exactly one PPO update,
performs the actual external evaluation, and atomically publishes only closed
component packages. It cannot truthfully include terminal Slurm accounting
and therefore must not invoke the assembler. Phase B starts only after
scheduler state is `COMPLETED 0:0`: first capture the terminal allocation with
`terminal-receipt`, then use that receipt to gate every fetch of the component
tree, complete stdout/stderr, and immutable remote submission receipt. Each
schema-2 fetch receipt records its start before the first remote probe and must
bind the exact terminal receipt; the finalizer rejects any fetch that began
before both Slurm `End` and terminal-receipt capture. It also binds the
scheduler `SubmitLine`, QOS, time limit, no-requeue/restart state, work/log
paths, exact MIG TRES, and staged sbatch hash. It interprets Hopper scheduler
times in `America/New_York`, rejects ambiguous/nonexistent local timestamps,
and requires ordered Submit/Start/End values whose End epoch and elapsed delta
match the receipt exactly. The exact bundled finalizer runs
under an isolated clean Python 3.10.20 venv, derives `scheduler.json`, invokes
the assembler's engineering mode, then invokes its read-only validation mode.
It must not run the production 30,000-update analyzer or claim analyzer
eligibility. Actual external evaluation is required for this smoke, but its
metric values remain permanently excluded from paper claims.

The generic `fetch` command remains technically callable without a terminal
gate for unrelated workflows. The terminal-chain rule is therefore both an
operator prohibition before completion and a machine-enforced finalization
gate: an ungated or preterminal fetch receipt can never finalize this package.
Use the exact manifest-bound bundle copy of `hopper.sh` for Phase-A submission,
terminal capture, and every Phase-B fetch; its `submit` command rejects
resource, identity, output, chdir, and requeue overrides.

Phase A must use a single explicit `UED_*` export allowlist, never
`--export=ALL`. The sbatch has a direct `/bin/bash` shebang and then sanitizes
Python, loader, and JAX/XLA controls before invoking the pinned environment as
`python -I -B`. Keep the wrapper registry outside the immutable bundle; the
wrapper appends its local receipt there during submission. With every value
below copied from the exact staged bundle/environment and the fresh prerequisite
receipts, the submission shape is:

```bash
LOCAL_EXACT_BUNDLE=/absolute/exact/content-addressed-bundle
HOPPER_WRAPPER="$LOCAL_EXACT_BUNDLE/hopper/hopper.sh"
HOPPER_REGISTRY=/absolute/operator/receipts/ued-terminal-submissions.tsv
mkdir -p "$(dirname "$HOPPER_REGISTRY")"
TERMINAL_EXPORT_VALUES=(
  "UED_BUNDLE_DIR=$UED_BUNDLE_DIR"
  "UED_BUNDLE_MANIFEST_SHA256=$UED_BUNDLE_MANIFEST_SHA256"
  "UED_UPSTREAM_COMMIT=$UED_UPSTREAM_COMMIT"
  "UED_UPSTREAM_TREE=$UED_UPSTREAM_TREE"
  "UED_UPSTREAM_BUNDLE_SHA256=$UED_UPSTREAM_BUNDLE_SHA256"
  "UED_OVERLAY_MANIFEST_SHA256=$UED_OVERLAY_MANIFEST_SHA256"
  "UED_TERMINAL_CHAIN_SBATCH_SHA256=$UED_TERMINAL_CHAIN_SBATCH_SHA256"
  "UED_FRONTIER_CONFIG_SHA256=$UED_FRONTIER_CONFIG_SHA256"
  "UED_CONTRACT_SHA256=$UED_CONTRACT_SHA256"
  "UED_PROTOCOL_SHA256=$UED_PROTOCOL_SHA256"
  "UED_ANALYZER_SHA256=$UED_ANALYZER_SHA256"
  "UED_TRAINING_DRIVER_SHA256=$UED_TRAINING_DRIVER_SHA256"
  "UED_EVALUATION_DRIVER_SHA256=$UED_EVALUATION_DRIVER_SHA256"
  "UED_ASSEMBLER_SHA256=$UED_ASSEMBLER_SHA256"
  "UED_ENV_DIR=$UED_ENV_DIR"
  "UED_ENV_LOCK_SHA256=$UED_ENV_LOCK_SHA256"
  "UED_ENV_FREEZE_SHA256=$UED_ENV_FREEZE_SHA256"
  "UED_ENV_MANIFEST_SHA256=$UED_ENV_MANIFEST_SHA256"
  "UED_IMPORT_SMOKE_RESULT_DIR=$UED_IMPORT_SMOKE_RESULT_DIR"
  "UED_IMPORT_SMOKE_MANIFEST_SHA256=$UED_IMPORT_SMOKE_MANIFEST_SHA256"
  "UED_ONE_UPDATE_RESULT_DIR=$UED_ONE_UPDATE_RESULT_DIR"
  "UED_ONE_UPDATE_MANIFEST_SHA256=$UED_ONE_UPDATE_MANIFEST_SHA256"
)
TERMINAL_EXPORTS=$(IFS=,; printf '%s' "${TERMINAL_EXPORT_VALUES[*]}")
HOPPER_REGISTRY="$HOPPER_REGISTRY" bash "$HOPPER_WRAPPER" submit \
  "$LOCAL_EXACT_BUNDLE/hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch" \
  "--export=$TERMINAL_EXPORTS"
```

Record the printed numeric job ID and immutable remote submission-receipt path.
Do not inspect or fetch any endpoint or Slurm stream before terminal accounting.

The post-terminal command shape is:

```bash
JOB=REPLACE_NUMERIC_JOB_ID
NEW_LOCAL=/absolute/new/terminal-chain-job-REPLACE_NUMERIC_JOB_ID
LOCAL_EXACT_BUNDLE=/absolute/exact/content-addressed-bundle
REMOTE_SUBMISSION_RECEIPT=/scratch/lwang44/maxrl/receipts/REPLACE.tsv
mkdir -p "$NEW_LOCAL"
HOPPER_WRAPPER="$LOCAL_EXACT_BUNDLE/hopper/hopper.sh"
HOPPER_LOCAL_RESULTS_ROOT="$NEW_LOCAL" \
  bash "$HOPPER_WRAPPER" terminal-receipt "$JOB" "$NEW_LOCAL/terminal-sacct.tsv"

bash "$HOPPER_WRAPPER" fetch \
  "/scratch/lwang44/maxrl/tests/logs/ued-minimax-terminal-chain_${JOB}.out" \
  "$NEW_LOCAL/slurm.stdout" "$NEW_LOCAL/fetch-stdout.tsv" \
  "$NEW_LOCAL/terminal-sacct.tsv"
bash "$HOPPER_WRAPPER" fetch \
  "/scratch/lwang44/maxrl/tests/logs/ued-minimax-terminal-chain_${JOB}.err" \
  "$NEW_LOCAL/slurm.stderr" "$NEW_LOCAL/fetch-stderr.tsv" \
  "$NEW_LOCAL/terminal-sacct.tsv"
bash "$HOPPER_WRAPPER" fetch "$REMOTE_SUBMISSION_RECEIPT" \
  "$NEW_LOCAL/submission-receipt.tsv" "$NEW_LOCAL/fetch-submission.tsv" \
  "$NEW_LOCAL/terminal-sacct.tsv"

# Read only the exact post-terminal completion marker to obtain REMOTE_COMPONENTS.
REMOTE_COMPONENTS=/scratch/lwang44/maxrl/tests/ued-minimax-terminal-chain/REPLACE/job-$JOB
bash "$HOPPER_WRAPPER" fetch "$REMOTE_COMPONENTS" "$NEW_LOCAL/components" \
  "$NEW_LOCAL/fetch-components.tsv" "$NEW_LOCAL/terminal-sacct.tsv"

mkdir "$NEW_LOCAL/phase-b-home"
PHASE_B_BASE_PYTHON=/home/robotixx/miniconda3/envs/agenticrl/bin/python3.10
env -i HOME="$NEW_LOCAL/phase-b-home" PATH=/usr/bin:/bin LC_ALL=C \
  PYTHONNOUSERSITE=1 "$PHASE_B_BASE_PYTHON" -I -B -m venv "$NEW_LOCAL/phase-b-venv"
PHASE_B_PYTHON="$NEW_LOCAL/phase-b-venv/bin/python"
PHASE_B_PYTHON_REAL=$(readlink -f "$PHASE_B_PYTHON")
PHASE_B_PYTHON_SHA256=$(sha256sum "$PHASE_B_PYTHON_REAL" | awk '{print $1}')
env -i HOME="$NEW_LOCAL/phase-b-home" \
  PATH="$NEW_LOCAL/phase-b-venv/bin:/usr/bin:/bin" LC_ALL=C \
  PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
  "$PHASE_B_PYTHON" -I -B -m pip check
env -i HOME="$NEW_LOCAL/phase-b-home" \
  PATH="$NEW_LOCAL/phase-b-venv/bin:/usr/bin:/bin" LC_ALL=C \
  PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
  "$PHASE_B_PYTHON" -I -B -m pip freeze --all | LC_ALL=C sort \
  > "$NEW_LOCAL/phase-b.freeze"
PHASE_B_FREEZE_SHA256=$(sha256sum "$NEW_LOCAL/phase-b.freeze" | awk '{print $1}')
PHASE_B_VENV_CONFIG_SHA256=$(sha256sum "$NEW_LOCAL/phase-b-venv/pyvenv.cfg" | awk '{print $1}')
FINALIZER_SHA256=$(sha256sum \
  "$LOCAL_EXACT_BUNDLE/hopper/finalize_ued_minimax_terminal_chain.py" | awk '{print $1}')
BUNDLE_SHA256=$(sha256sum "$LOCAL_EXACT_BUNDLE/SHA256SUMS" | awk '{print $1}')
COMPONENTS_SHA256=$(sha256sum "$NEW_LOCAL/components/SHA256SUMS" | awk '{print $1}')
INPUT_CLOSURE_SHA256=$(sha256sum "$NEW_LOCAL/components/INPUT_CLOSURE.json" | awk '{print $1}')
TERMINAL_CHAIN_SBATCH_SHA256=$(sha256sum \
  "$LOCAL_EXACT_BUNDLE/hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch" | awk '{print $1}')
ASSEMBLER_SHA256=$(sha256sum \
  "$LOCAL_EXACT_BUNDLE/ued_benchmark/scripts/assemble_matched_run.py" | awk '{print $1}')

env -i HOME="$NEW_LOCAL/phase-b-home" \
  PATH="$NEW_LOCAL/phase-b-venv/bin:/usr/bin:/bin" LC_ALL=C \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
  "$PHASE_B_PYTHON" -I -B \
  "$LOCAL_EXACT_BUNDLE/hopper/finalize_ued_minimax_terminal_chain.py" \
  --job-id "$JOB" \
  --bundle-dir "$LOCAL_EXACT_BUNDLE" \
  --expected-bundle-manifest-sha256 "$BUNDLE_SHA256" \
  --components-dir "$NEW_LOCAL/components" \
  --expected-components-manifest-sha256 "$COMPONENTS_SHA256" \
  --expected-input-closure-sha256 "$INPUT_CLOSURE_SHA256" \
  --expected-sbatch-sha256 "$TERMINAL_CHAIN_SBATCH_SHA256" \
  --terminal-receipt "$NEW_LOCAL/terminal-sacct.tsv" \
  --submission-receipt "$NEW_LOCAL/submission-receipt.tsv" \
  --submission-fetch-receipt "$NEW_LOCAL/fetch-submission.tsv" \
  --components-fetch-receipt "$NEW_LOCAL/fetch-components.tsv" \
  --slurm-stdout "$NEW_LOCAL/slurm.stdout" \
  --stdout-fetch-receipt "$NEW_LOCAL/fetch-stdout.tsv" \
  --slurm-stderr "$NEW_LOCAL/slurm.stderr" \
  --stderr-fetch-receipt "$NEW_LOCAL/fetch-stderr.tsv" \
  --expected-assembler-sha256 "$ASSEMBLER_SHA256" \
  --expected-finalizer-sha256 "$FINALIZER_SHA256" \
  --python "$PHASE_B_PYTHON" \
  --expected-python-sha256 "$PHASE_B_PYTHON_SHA256" \
  --expected-python-version 3.10.20 \
  --expected-python-freeze-sha256 "$PHASE_B_FREEZE_SHA256" \
  --expected-python-venv-config-sha256 "$PHASE_B_VENV_CONFIG_SHA256" \
  --output-dir "$NEW_LOCAL/finalized"
```

These are placeholders, not a submission or finalization authorization.
Every destination must be absolute, canonical, and nonexistent; configure
`HOPPER_LOCAL_RESULTS_ROOT` to the existing parent used for the terminal audit.
`PHASE_B_BASE_PYTHON` must be the approved absolute Python 3.10.20 base; the
fresh venv must pass `pip check`, and its exact resolved binary, sorted
`pip freeze --all`, and `pyvenv.cfg` hashes must be supplied. Never replace
the bundled finalizer path with a working-tree copy.

Adding the terminal-chain sbatch, assembler, or schema changes the bundle
manifest. Therefore jobs 9366896 and 9366897 document the `6c2c...` closure
only and cannot gate the changed bundle. Run the exact-bundle import and
one-update rungs again before any terminal-chain submission. No full UED
campaign is authorized by this runbook while `HOPPER_STATUS.md` says `HOLD`.

## BARN source-bound workflow

The BARN path is separate from the MAZE ladder above. Do not invoke
`barn_seed_cpu.sbatch` directly and do not substitute generic `submit`, `logs`,
or `fetch` calls for the BARN wrappers.

Current state: campaign `barn-icra2027-20260814-002` was canceled
outcome-blind after an unsupported publication primitive was discovered.
Replacement campaign `barn-icra2027-20260814-003` is already running the exact
frozen 20-task matrix under the amended hard-link publication contract. The
submission examples below are historical/operator templates, not authorization
to resubmit or create another campaign. Monitor campaign 003 only through its
ledger workflow and finalize it only after every declared task is terminal.

### BARN engineering preparation

Stage the DRAFT engineering closure, copy its four printed `BARN_*` values,
and pass the same values to the CPU-only preparation job and then the
train-only smoke:

```bash
bash hopper/stage_barn_campaign.sh engineering

export BARN_SOURCE_BUNDLE_DIR=...
export BARN_SOURCE_SHA256=...
export BARN_DATASET_ARCHIVE=...
export BARN_DATASET_ARCHIVE_SHA256=...

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

The preparation job verifies all official archive assets on a compute node.
The smoke verifier reads only package controls and the `barn-299` manifest
declarations; only the runner hashes and loads that train course. Both jobs
publish redacted receipts, and the smoke destroys its metric-bearing internal
artifact.

### BARN evidence submission

After both receipts pass, freeze `icra2027/prereg_icra.md` and
`icra2027/barn_protocol.json`, commit the complete BARN closure, and stage a
new evidence-mode bundle. Engineering source values are not evidence values.

```bash
bash hopper/stage_barn_campaign.sh evidence

CAMPAIGN_ID=barn-icra2027-001
SOURCE_BUNDLE=...  # printed BARN_SOURCE_BUNDLE_DIR
SOURCE_SHA256=...  # printed BARN_SOURCE_SHA256
```

Use that same campaign ID and source pair for exactly these four calls. Attempt
IDs must be unique. N=8 is already represented by `primary` and is not a fifth
fresh ablation cell.

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

Each wrapper call submits one held seeds-1--5 array, installs its normalized
ledger rows, and releases the array only after the remote ledger is verified.
Monitor only the printed job IDs with `hopper.sh status` or `hopper.sh watch`.
Do not open raw logs or endpoints, fetch seed artifacts, select a subset, or
analyze locally. A retry is a complete five-seed cell submission with a new
attempt ID under the same campaign and source pair.

If a wrapper call reports that its array remains held, rerun the exact same
campaign/cell/attempt/source command to resume the durable transaction. A
different attempt is deliberately refused while that pending record exists.
Once `BARN_CAMPAIGN_SEALED` is published, that campaign ID is closed to all
further attempts.

### BARN ledger finalization and sealed retrieval

Wait until every recorded Slurm array task is terminal. If scheduler metadata
shows a failed task, submit a whole-cell retry before proceeding. Then run the
source-bound ledger finalizer and copy the `sha256=` field from its
`BARN_LEDGER_FINALIZED` line:

```bash
bash hopper/finalize_barn_ledger.sh \
  "$CAMPAIGN_ID" "$SOURCE_BUNDLE" "$SOURCE_SHA256"

FINALIZED_LEDGER_SHA256=...
bash hopper/finalize_barn_campaign.sh \
  "$CAMPAIGN_ID" "$SOURCE_BUNDLE" "$SOURCE_SHA256" \
  "$FINALIZED_LEDGER_SHA256"
```

The ledger finalizer requires the exact four campaign cells, complete
seeds-1--5 groups for every recorded attempt, and terminal accounting for all
tasks. Failed attempts may remain in the ledger after a full-cell retry. The
campaign finalizer submits the bundled CPU-only `normal` Slurm postprocessor,
waits using accounting metadata, performs all four outcome-blind selections
and merges plus the primary and ablation analyses, and seals their closed
package. Only after the postprocessor completes does the wrapper fetch that
single sealed all-cell package. `BARN_CAMPAIGN_SEALED` is the successful end
state; there is no authorized raw, partial, or per-cell retrieval step.

## Help

- ORC support: `orchelp@gmu.edu`
- Open OnDemand: [https://ondemand.orc.gmu.edu](https://ondemand.orc.gmu.edu)
  (VPN/campus network may be required)
