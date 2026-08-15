# Tie-aware v4 Hopper launch design

Date: 2026-08-14  
Scope: read-only design/audit; no remote query, staging, submission, cancellation, or endpoint inspection

## Decision

The next tie-aware v4 Hopper ladder is technically designable, but it is **not launch-ready today**. The finalized v4 core overlay can be reused byte-for-byte. The existing Hopper chain cannot: its stage manifest, GPU smoke, one-update job, terminal job, evaluator, assembler, and Phase-B finalizer are pinned to the frozen v3/v1 lineage. The safe path is a new sibling v4 engineering bundle followed, only after closure, by a different and separately authorized 100-update cost bundle.

Job `9367063` and its v3 bundle are immutable inputs to a different ladder. Nothing in a future v4 path may cancel, requeue, relabel, overwrite, depend on, or use a receipt from that job. Its pending/running output and logs remain unopened. Its eventual result can establish only the exact v3 bundle to which it belongs.

Current authorization state:

- **GO**: local implementation and independent audit of new sibling v4 tooling.
- **HOLD**: every v4 remote stage, Slurm submission, endpoint access, 100-update pilot, factorial run, development run, and paper claim.

## Evidence boundary

This design used only repository source, frozen manifests/protocols, local Hopper documentation, and closed safe accounting receipts. It did not inspect any held performance value, OOD result, pending-job stream, or partial result tree. It did not query Hopper. It did not inspect or modify BARN, MAZE-score, Blackwell/5090, or other GPU workstreams.

The only completed resource datum used here is the closed v3 one-update accounting package:

- `/data/robotixx/ued_bench/hopper/one-update-job-9366897/resource-accounting.json`, SHA-256 `1dbaad927677a48559f77eef8b1c2eb2a624982b910dd8d71e9f3a6f06e1cad4`.
- Completion marker SHA-256 `88c31c2293061b5e51482d917e91c988c55be9a60fe6abb27066633249eaebe3`.
- It records 16,384 training transitions, 65.807479368 in-process seconds, 248.96866066514312 transitions/s including first compilation, and 1,824,760 KiB (1.740 GiB) process maximum host RSS.
- Allocation: one A100 `1g.10gb` MIG, 2 CPUs, 15 GB host memory. Terminal `sacct` is explicitly not present in the inner resource receipt, so no peak GPU-memory conclusion is available.

These are v3 one-update engineering observations, not v4 steady-state measurements and not a performance endpoint.

## Frozen identities that must not drift

### Queued v3 chain

| Item | Frozen identity |
|---|---|
| Queued import/JIT job | `9367063` |
| Bundle namespace | `/scratch/lwang44/maxrl/bundles/ued_minimax/` |
| Bundle ID | `06ffeeeb6998e8ddb1ce` |
| Bundle manifest SHA-256 | `06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff` |
| Environment | `/scratch/lwang44/envs/ued-minimax-v2-9ab83896f41c5294-dbd0494789fd70b8` |
| Environment lock | `hopper/requirements-ued-minimax-hopper.lock`, `9ab83896f41c5294e0185593330898d3fcbe9187052931f2620b24b32da6c5c4` |
| Environment setup | `hopper/setup_ued_minimax_env.sh`, `dbd0494789fd70b8a2d677e0341ec8feab7623ade3133677cdf176dc75dcac2e` |

Do not modify the remote bundle, result root, job name, prerequisite record, submission receipt, or local fetch destination associated with `9367063`. Do not make v4 depend on its eventual result manifest. A new v4 bundle requires a fresh v4 import/JIT result from its own exact manifest.

### Finalized tie-aware v4 core

| File/identity | SHA-256 or value |
|---|---|
| Upstream commit | `d053054c5290a04c1c4cd8b55704d999cad73e30` |
| Upstream tree | `b0cace1fc54984e21a842f12d15d0b899e33d270` |
| `ued_benchmark/OVERLAY_CONTRACT_V4.json` | `3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b` |
| `ued_benchmark/OVERLAY_LINEAGE.json` | `784e2fd1f545d49c8d10c3f3aeda37aae51fa00127e2c14578702e275bfb6971` |
| `ued_benchmark/scripts/apply_minimax_overlay_v4.py` | `c2e5eb3dac02b86723ece485cd348832f1636198c781bae82c1d99df0167590b` |
| `ued_benchmark/overlay/minimax/util/rl/frontier_activity.py` | `63726251813bd9fafc2722409c4a2942c6ae2728327870797df47d01504738ca` |
| `ued_benchmark/overlay/minimax/util/rl/tie_aware_rank.py` | `1b9db20d05edd3212346e84d14606af91ae443c0665945a7b679ade161560244` |
| Fresh-clone applied-overlay manifest | `9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae` |
| `ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json` | `1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269` |
| v4 Frontier N=8 config | `0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2` |
| v4 group-matched MaxMC N=8 config | `a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6` |
| `ued_benchmark/scripts/run_grouped_one_update_v4.py` | `4acf85ab5b4a90a8f14fe94727e7973a5a47c14e85a22f18e8bd32641af88e78` |
| `ued_benchmark/scripts/run_matched_terminal_v4.py` | `171e1752980a31b121db427020708902ab046aeabe8af94160ff15cf2af5c9db` |

The overlay is GO for bounded engineering. Its default/opt-out behavior remains source-faithful; tie-aware rank replay is explicit for exact Frontier and group-matched MaxMC; the untouched official upstream MaxMC reference remains outside the overlay. The DRAFT v2 driver rejects production/matched-development execution and authorizes no performance endpoint.

One minor general-release issue remains intentionally outside the frozen core: direct constructors reject `temp <= 0` but a NaN can evade that comparison. Every authored config here pins finite `0.3`, and the terminal host validator fails closed. A v4 Hopper wrapper must additionally require `math.isfinite(plr_temp) and plr_temp > 0` before JIT. If constructor-level behavior is changed instead, that is a new overlay identity and requires a new contract, applicator, applied-manifest hash, tests, and ladder; it may not be silently called the finalized v4 above.

### N-factorial DRAFT

| File | SHA-256 |
|---|---|
| `ued_benchmark/analysis/development_protocol_v3_n_factorial_tie_aware_draft.json` | `81a57668d3cfdf595f13710df6152a437b8c4640791fbeeed2ef8c9e9486f26f` |
| `ued_benchmark/analysis/n_factorial_tie_aware_v4_draft_manifest.json` | `58e1ffd9c7e3d80992971b331c540d6c8976c9cd4082391fae92de0df4fd417f` |
| `ued_benchmark/analysis/validate_n_factorial_draft.py` | `7b520584d3803c933984be79a6a180eeb59a330e724c2773bd7a9b3c6569564d` |
| N=2 Frontier / MaxMC configs | `2e443515d3876ad8c8a632d9cc21f2a92288adf971cde0b2c4751679eed32791` / `81e2af766e588896c3013de23f22906446e27e18661e2dfc6b6f2cc4e284f1b3` |
| N=4 Frontier / MaxMC configs | `181ca0210ad988a699d408827b15941ef0a6b9c1588f4abd8c12dc1e6cc706b5` / `9033de1f79ee7f64ac980ebe28e90542e3cfe93dddf63374c20e1e52def824fb` |
| N=8 Frontier / MaxMC configs | `5cdaf48da9b6e3f2ab9dd0b9dd8c94eb7e49fe07d7744fc46b7b5f735b3a436d` / `105c6695baf86b894d65c6756fc5647d560c84daaef35ee3d3859c1eb9f68090` |

The factorial DRAFT explicitly forbids Hopper/GPU submission, endpoint access, production scheduling, and paper-evidence use. It fixes 32 streams and 8,192 transitions per outer cycle while jointly changing `N`, `n_eval`, `n_parallel`, and buffer size:

| N | `n_parallel` x `n_eval` | Buffer |
|---:|---:|---:|
| 2 | 16 x 2 | 2,000 |
| 4 | 8 x 4 | 1,000 |
| 8 | 4 x 8 | 500 |

Its exact 63-cycle warm-fill statement is conditional on accepting exactly `n_parallel` distinct new groups in every fill cycle. The receipt must report the actual replay-eligibility cycle and transition count whenever duplicates, incomplete/nonfinite groups, or another rejection break that condition.

The factorial N=8 files and the v2 matched N=8 files have different byte hashes. They are not interchangeable, even if parsed scientific fields appear equivalent. A campaign or receipt must bind one named config and its exact bytes.

## Why the current Hopper terminal chain cannot be relabeled v4

The current v3 tooling is immutable engineering history:

| Current file | SHA-256 | v4 incompatibility |
|---|---|---|
| `hopper/stage_ued_minimax.sh` | `73ad318fe21f6f99c92fe09ac6ec76c5dd6fe4b0d7fec8a5ca5939c59483ba55` | Builds schema-4 v3 bundle, copies v3 applicator, allows only v3 Frontier one-update/terminal/import endpoints, and sets `max_student_updates=1`. |
| `hopper/sbatch/ued_minimax_gpu_smoke.sbatch` | `dc2151530af2fd03d0917f96c5e1731f79647a679fe3396deb04135d69c59056` | Applies/checks the v3 overlay and JITs only the v3 Frontier formula/AMaze path. |
| `hopper/sbatch/ued_minimax_one_update_smoke.sbatch` | `f17e79d9fdc9436ac3858413b3dca98f81521ce537b5b6aee8e2f5c9813cf8e6` | Hardcodes v3 contract `5868...`, v3 Frontier config `b491...`, and `run_grouped_one_update.py`. |
| `hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch` | `5ed5186e010decdcc6bf97ff7dc820e0f4cf13e580e9b20c996a8dc561b13a14` | Hardcodes v1 protocol, v3 contract/config/driver, Frontier arm/seed 101, and v1 evaluator/assembler/analyzer. |
| `hopper/finalize_ued_minimax_terminal_chain.py` | `57eb4394cedf30cc1a5bfeca4734199652cbfcfd5fbcaaca08035e8001a2c5ec` | Assumes Frontier-only v1 run identity and `frontier-buffer-snapshot.json`; cannot validate the common v4 replay snapshot or MaxMC telemetry. |
| Local stage/import test | `cfa618f807e409b1ecc2257ed92f075a133a844b1bd41e8df6abc4ec93b0b177` | Pins the schema-4/v3 bundle. |
| Local one-update test | `6c288368081dd3464e8a56d9042867e156789b7072fcd055027496e2359da30a` | Pins v3 one-update identities and limits. |
| Local terminal test | `d57489b61f53ef32ea6f98450978390da2729067e47039114914cdd1652225c9` | Pins v1/v3 terminal components and Frontier-only finalization. |

There are also source-level gaps:

- `run_grouped_one_update_v4.py` is a valuable audited N=8 Frontier one-update gate, but it is deliberately Frontier-only, hardcoded to the v2 config identity, and capped at one update. It is not a generic N-factorial or cost driver.
- `run_matched_terminal_v4.py` is tied to the N=8 v2 protocol and intentionally requires local or Slurm engineering mode. It rejects production. It also locates `evaluate_matched_terminal.py` and `assemble_matched_run.py`, so using it unchanged binds v1-aware helpers.
- `evaluate_matched_terminal.py` imports `run_matched_terminal` (the v3 driver) and validates the v1 protocol identity. Its current SHA-256 is `09dbbcafdd3425ea1f751c766cb4cd010d69f438af02aa07f9fc00a3d943ab36`.
- `assemble_matched_run.py` imports the frozen v1 analyzer, expects `frontier-buffer-snapshot.json` only for Frontier, and expects no corresponding MaxMC snapshot. Its SHA-256 is `655d4d20da506a074b9365471a1900bdaf6468ddd991e313f6feb67fc84bde57`.
- v4 training emits `plr-replay-snapshot.json` for both arms. The v2 package contract names the packaged payload `training-plr-replay-snapshot.json`. A new assembler/finalizer must validate and map that common sidecar for both arms.
- The N-factorial DRAFT has no hashed launch driver. Its direct parser default xpid is `latest`, explicitly declared unsafe. It requires a campaign-bound unique `run_id`, `xpid`, output directory, and checkpoint path.

These are contract mismatches, not renaming problems. Overwriting any v3/v1 file would destroy reconstruction and can collide with job `9367063`.

## Required sibling files for a future v4 engineering bundle

The following names are proposed so implementation cannot silently mutate the v3 chain. Their final SHA-256 values do not exist yet and must be computed from reviewed bytes; literal placeholders are forbidden in a staged bundle.

| New sibling | Required role |
|---|---|
| `hopper/stage_ued_minimax_v4.sh` | Deterministically build a new v4 engineering namespace and schema; exclude Blackwell trees; bind every input below. |
| `hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch` | Apply/check v4, verify the expected applied manifest, exercise tie-aware rank distribution and one AMaze JIT on JAX 0.4.31. No training/evaluation. |
| `hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch` | Run only audited N=8 Frontier one-update driver, checkpoint round-trip, and common replay/tie diagnostics. |
| `hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch` | Parameterized `frontier|maxmc`; one update plus actual external engineering evaluation; publish closed Phase-A components only. |
| `hopper/finalize_ued_minimax_v4_terminal_chain.py` | Post-terminal-only fetch, authoritative `sacct`, common sidecar validation for either arm, and atomic permanently analyzer-ineligible assembly. |
| `hopper/test_ued_minimax_v4_local.sh` | Two-build/idempotence/hash/exclusion/namespace tests for the new stage bundle. |
| `hopper/test_ued_minimax_v4_one_update_local.sh` | Restricted-path one-update end-to-end receipt and checkpoint drift tests. |
| `hopper/test_ued_minimax_v4_terminal_chain_local.sh` | Both-arm Phase A/Phase B simulations, stale/mixed prerequisite rejection, sidecar closure, and analyzer-ineligible assertions. |
| `ued_benchmark/scripts/run_matched_terminal_hopper_v4.py` | Sibling of the frozen v4 driver which binds v4 evaluator/assembler paths and preserves the engineering-only gate. |
| `ued_benchmark/scripts/evaluate_matched_terminal_v4.py` | Import the v4 Hopper driver, bind v2 identity, and produce a v4-closed evaluation component without importing v1 training. |
| `ued_benchmark/scripts/assemble_matched_run_v4.py` | Structural engineering assembler for the common `plr-replay-snapshot.json`; do not import/invoke the v1 production analyzer. |
| `ued_benchmark/tests/test_hopper_terminal_v4.py` | Parsed-config, both-arm, sidecar, replay-integrity, campaign, and Phase-B closure tests. |

The engineering `BUNDLE_STATE.json` should mint a new schema (recommended `bundle_schema=5`) with exactly these allowed endpoints:

1. `v4_gpu_import_tie_formula_jit`
2. `v4_frontier_grouped_one_update`
3. `v4_frontier_terminal_chain_components`
4. `v4_maxmc_terminal_chain_components`

It must retain `max_student_updates=1`, `paper_evidence=false`, `endpoint_access_authorized=false`, `production_analyzer_invoked=false`, explicit-export-only Slurm submission, `-I -B`, no resume, two-phase terminal accounting, and new-result-directory-only publication. It must bind:

- all finalized v4 hashes listed above;
- v2 protocol and both v2 N=8 configs;
- upstream pin/commit/tree/git-bundle hashes;
- environment lock/setup/freeze/manifest hashes;
- every new driver, sbatch, finalizer, and test hash;
- exact expected fresh applied-overlay manifest `9b411f61...`;
- protected v3/v1 lineage hashes from `OVERLAY_LINEAGE.json` so the bundle proves it did not rewrite history.

The outer `SHA256SUMS` is the bundle authority. Build twice into new local destinations; require byte-identical outer manifest hashes and strict checksum validation before any independent audit.

Recommended remote namespace and job-name separation:

- Bundle: `/scratch/lwang44/maxrl/bundles/ued_minimax_v4_engineering/<bundle-id>`
- Results: `/scratch/lwang44/maxrl/tests/ued-minimax-v4-engineering/<bundle-id>/<rung>/<job-id>`
- Job names: `ued-v4-import`, `ued-v4-one-update`, `ued-v4-terminal-frontier`, `ued-v4-terminal-maxmc`

Never reuse `/scratch/lwang44/maxrl/bundles/ued_minimax/06ffee...` or any v3 result destination.

The existing lock-addressed environment can be reused only if its lock, setup, freeze, manifest, Python 3.10.20, Git 2.45.2, NumPy 1.25.2, JAX/JAXlib/CUDA plugin/PJRT 0.4.31, and active executable paths all verify exactly. Environment reuse does not waive the fresh v4 import/JIT rung.

## Ladder and closure gates

Every rung consumes only a closed result from the immediately preceding rung with the same v4 bundle-manifest SHA. `afterok` is scheduling convenience, not evidence; local post-terminal verification is the gate.

### Rung 0 — local bundle closure

Required before remote staging:

- Apply v4 to a fresh `d053054...` clone; check; reapply; check idempotence.
- Verify applied manifest SHA `9b411f61...` and every generated-file hash.
- Run the finalized JAX 0.4.31 CPU suite with `JAX_PLATFORMS=cpu` and `PYTHONDONTWRITEBYTECODE=1`.
- Run new wrapper tests for both arms, exact parsed config equality, finite-positive temperature, checkpoint static signature, common sidecar, and v3/v4 cross-resume rejection.
- Prove the disabled future telemetry path leaves PRNG use, training state, replay distribution, checkpoint, and update counts unchanged.
- Build the engineering bundle twice; require identical outer manifests, strict checksums, no symlinks, no unlisted file, and explicit exclusion of both Blackwell subtrees.
- Independent read-only audit must issue GO on the frozen bytes.

### Rung 1 — import/formula/JIT

One `1g.10gb` job, no training and no evaluation:

- Verify bundle/environment identities before importing code.
- Apply and check v4; verify exact applied manifest.
- JIT source-stable and tie-aware score/replay distributions, all-equal ties, distinct scores, filled/unfilled slots, signed zero, nonfinite failure, and one AMaze reset/step path.
- Check both N=8 parsed configs and both tie-aware flags, but do not inspect any performance endpoint.
- Publish a closed result plus terminal accounting; fetch only after `COMPLETED 0:0`.

### Rung 2 — exact grouped one update

Use the audited N=8 Frontier config and driver only. Execute the existing bounded two-cycle construction: one new-level warmup cycle, one forced replay cycle, exactly one PPO update, one upstream gradient update, five optimizer applications, and 16,384 transitions. Require:

- exact bundle/import/environment/input-closure hashes;
- no incomplete, duplicate-new, or nonfinite group rejection;
- checkpoint save/reload with exact static signature including tie-aware mode;
- cycle 1 replay draws zero;
- cycle 2 last and cumulative draw count exactly 4, with distinct count in `[1,4]` and duplicates `4-distinct`;
- `force_unique` recorded as buffer-update deduplication, never replay resampling;
- finite score/replay ESS and tie diagnostics;
- closed receipt and terminal `sacct` fetched only after completion.

This rung does not validate MaxMC training or terminal packaging.

### Rung 3 — terminal engineering, both arms

Run two separate jobs from the same manifest: Frontier seed 101 and MaxMC seed 101. Each is capped at one PPO update. Separate jobs prevent one arm's failure from contaminating the other and make resource accounting attributable.

Phase A on Slurm:

- train one real update with the exact arm config;
- save and round-trip the terminal checkpoint;
- perform the actual three-environment, 10-episode-per-environment engineering evaluation used to exercise packaging;
- write `training-receipt.json`, `plr-replay-snapshot.json`, evaluation components, provenance, and strict checksums;
- publish `COMPONENTS_COMPLETE`; never invoke a production analyzer.

Phase B locally, only after terminal `COMPLETED 0:0`:

- capture authoritative `sacct` with exact headers;
- fetch into a destination that did not exist;
- verify outer and nested manifests, job/submission/environment/source/config/driver hashes, and replay integrity;
- require checkpoint `replay_integrity` to equal the terminal receipt exactly;
- for both arms require finite filled scores, zero nonfinite rejection, finite ESS, and `draw=distinct+duplicate` for last and cumulative counters;
- assemble an atomic package with `paper_evidence=false`, `analyzer_eligible=false`, `endpoint_class=bounded_engineering_test`, and `production_analyzer_invoked=false`.

One Frontier-only terminal pass is GO for Frontier mechanics but **HOLD for matched-v4 readiness**. Both arm packages must close before the cost rung can be considered.

### Rung 4 — separately authorized 100-update cost pilot

This rung must not be added to the max-one-update engineering bundle. Mint a second immutable namespace, bundle state, protocol, config(s), driver, sbatch, tests, and result root:

- `hopper/stage_ued_minimax_v4_cost100.sh`
- `hopper/sbatch/ued_minimax_v4_cost100.sbatch`
- `hopper/finalize_ued_minimax_v4_cost100.py`
- `hopper/test_ued_minimax_v4_cost100_local.sh`
- `ued_benchmark/analysis/hopper_cost_protocol_v4_100.json`
- `ued_benchmark/scripts/run_cost100_v4.py`
- dedicated cost configs whose only scientific schedule difference from their named source configs is the frozen 100-update/cost-checkpoint budget; no command-line budget override and no `xpid=latest`.

The protocol begins as DRAFT/HOLD. It becomes runnable only through an explicit, content-addressed authorization recording `cost_pilot_authorized=true`, while keeping `performance_endpoint_authorized=false`, `OOD_endpoint_authorized=false`, `paper_evidence=false`, and `production_campaign_authorized=false`.

The first cost pair should be N=8 Frontier and N=8 group-matched MaxMC in separate jobs. This exercises both score paths at the primary matched layout. It does **not** size the full N-factorial. Before a six-cell factorial campaign, extend the same cost rung to the N=2 pair, because N=2 has the largest buffer and highest `n_parallel`; N=8 covers the largest group N. N=4 remains a required import/one-update shape even if it is not a separate 100-update sentinel.

Cost-run requirements:

- Normal production-shape `plr_replay_prob=0.5`; do not force replay after fill.
- Stop on exactly 100 student PPO updates, 100 upstream gradient updates, and 500 optimizer applications, or fail incomplete at the earlier hard cap/wall limit.
- `from_last_checkpoint=false`; no resume, silent extension, or retry under the same attempt identity.
- Recommended hard cap: 340 outer cycles, hence at most 2,785,280 training transitions. If exact fill or 100 updates are not reached, close an `INCOMPLETE_COST_PILOT` receipt and do not extrapolate it into a campaign request.
- Preserve the production test/checkpoint cadence needed to measure overhead, but seal all return/solved/OOD values. Only predeclared cost, integrity, tie-diversity, and calibration diagnostics may be released.
- A cost package cannot be used by the development analyzer and cannot establish that either method performs well.

The active calibration telemetry specification is still DRAFT and not stable enough to bind by hash. It requires a separately versioned observational telemetry overlay/writer rather than a mutation of the finalized core v4. The cost rung remains HOLD until that writer and its final protocol hash are frozen and audited, or until a protocol amendment explicitly narrows the cost pilot to sealed telemetry capture with no calibration inspection.

## Required telemetry

### Tie and replay diversity

Record per cycle and terminal aggregates for both arms:

- buffer capacity, filled count, distinct filled-score count;
- tie-block count, tied-member count, maximum block size, and fixed block-size histogram including singletons;
- mathematical tie-block mass preservation error computed in float64 diagnostics, with the frozen float32 tolerance and no claim of arbitrary-backend bitwise reduction identity;
- score-distribution ESS and replay-mixture ESS over the full filled distribution, plus entropy and maximum probability;
- expected unique slots in four with-replacement draws, clearly labeled a distribution expectation;
- realized last/cumulative replay draw, distinct-slot, and duplicate-draw counts, and a per-cycle distribution of realized unique counts;
- filled-score nonfinite count and candidate nonfinite rejection count;
- incomplete group, duplicate-new group, repeated-existing-level, and accepted-new-group counts;
- exact filled-count trajectory, observed replay-eligibility cycle, and observed transitions at eligibility;
- explicit `force_unique_resamples_replay=false` and sample identity `replay buffer slot index`.

Do not conflate distribution ESS/expected uniqueness with realized unique samples. Do not report a nominal 63-cycle fill when its exact acceptance gate fails.

### Frontier calibration and MaxMC comparator

The separately frozen telemetry writer must emit one closed record per attempted exact-N group, including new candidates that are not persisted. Minimum identities and fields:

- run/seed/arm/N, cycle/group index, level hash, level-chain identity, selection source (`new|replay|mutation`), slot/generation before and after, and disposition;
- the identical **pre-current-batch** posterior snapshot for concurrent occurrences of the same logical level;
- pre successes/trials, prior, pre-group analytic expected-activity prediction `q`, current `K`, current trials, and analyzer-derived realized activity `m_N(K)`;
- post counts and whether evidence was accepted/persisted;
- exact event ordering and closed JSONL, receipt, `SHA256SUMS`, and `COMPLETE`.

Integrity gates are zero duplicate event identity, zero duplicate-new corruption, zero partial group in the calibration population, zero nonfinite record, posterior continuity, and no current-group leakage. For Frontier, report predeclared descriptive bias, MAE, MSE/RMSE, fixed-bin ECE/MCE, and new/mutation versus replay partitions. These are adaptive training diagnostics, not independent-sample inference or benchmark performance.

MaxMC is not a probability forecast. For a replay group, only the stored score captured before the current group may be used for discrimination diagnostics. A current-group MaxMC score is forbidden as a predictor of that same group's target. Never attach Brier, reliability, or calibration-error labels to MaxMC.

### Resource and closure telemetry

Record without exposing held performance values:

- Slurm request and terminal `sacct`: job ID/name, state, exit, node, partition/QOS, MIG profile, `AllocTRES`, `ElapsedRaw`, `TotalCPU`, `MaxRSS`, `TRESUsageInMax`, and exact command/header receipt;
- in-process monotonic time and `getrusage` split by source verification, overlay apply, import/JIT, warm fill, replay training, periodic evaluation, checkpoint save, checkpoint reload, hashing, and packaging;
- first-compile time separately from post-JIT cycle p50/p90/p99;
- exact outer cycles, student updates, upstream gradient updates, optimizer applications, training transitions, evaluation transitions, and transitions/s for separately named phases;
- peak host RSS, GPU-memory authority/source, disk bytes/inodes, checkpoint bytes/hash, sidecar bytes/hash, final package bytes/hash;
- observed fill/replay/new decision counts and walltime remaining at every checkpoint;
- terminal-versus-checkpoint replay-integrity equality.

Absence of terminal GPU memory must be reported as unknown, not inferred from allocation success or host RSS.

## Bounded MIG and time plan

Keep one A100 `1g.10gb` MIG, 2 CPUs, and 15 GB for the first v4 ladder. That exact allocation completed v3 import and N=8 one-update work. It is a tested allocation, not proof that a smaller slice is safe and not proof that the N=2 buffer fits. Do not downsize until a closed v4 receipt contains authoritative peak GPU memory. If a shape OOMs, stop and mint a new audited resource rung; do not silently retry on another MIG profile.

| Rung | Proposed request | Safe basis and interpretation |
|---|---|---|
| v4 import/JIT | `1g.10gb`, 2 CPU, 15 GB, 10 min | Comparable v3 exact import completed in 45 s. New code/hash means a fresh result is still mandatory. |
| v4 Frontier one update | same, 30 min | Comparable v3 job completed in 1:42; its driver portion was 65.81 s. |
| v4 terminal engineering | same, 45 min **per arm** | Existing v3 script requests 30 min but has no completed safe terminal-chain timing receipt. Forty-five minutes is provisional headroom for real evaluation and component closure. |
| v4 N=8 100-update cost | same, 4 h **per arm** | Based only on the conservative one-update rate and the hard 340-cycle cap; must be replaced by measured phase timings. |

At 8,192 transitions per outer cycle:

| Scenario | Cycles | Transitions | Naive time at 248.969 transitions/s |
|---|---:|---:|---:|
| Exact lower bound: 63 fill + 100 replay-update cycles | 163 | 1,335,296 | 89.4 min |
| Planning centerline if the post-fill `p=0.5` replay coin behaves independently | 263 | 2,154,496 | 144.2 min |
| Proposed hard cost cap | 340 | 2,785,280 | 186.5 min |

The first line is a lower bound, not an expected run. The second is a planning heuristic, not a guarantee. The third is a protocol cap. None includes trustworthy v4 steady-state, evaluation, checkpoint, or packaging overhead. A 4-hour request leaves provisional margin but does not guarantee 100 updates; failure to reach the target closes as incomplete.

Maximum requested allocation exposure, not expected usage:

- N=8 engineering ladder after import: 0.5 MIG-h one-update plus 1.5 MIG-h for two 45-minute terminal jobs; import adds 0.167 MIG-h.
- N=8 paired 100-update cost rung: 8 MIG-h (two independent four-hour jobs).
- Optional N=2 paired resource-envelope extension before factorial launch: another 8 MIG-h.

Do not reserve or submit these resources until the corresponding protocol and bundle receive explicit authorization.

## Promotion and closure matrix

| Item | Current state | Gate to advance |
|---|---|---|
| Queued v3 job `9367063` | **GO, untouched independent work** | Let its existing owner monitor scheduler-only until terminal; no v4 dependency or action here. |
| Finalized v4 core overlay | **GO for bounded engineering** | Preserve exact hashes/default behavior; host rejects nonfinite/nonpositive temperature. |
| v2 N=8 configs/protocol | **GO as immutable inputs to a new engineering bundle** | Keep production/performance authorization false. |
| N-factorial DRAFT design | **GO for local validation only** | New successor engineering authorization and generic hashed driver are required before any Hopper use. |
| Current v3 Hopper chain as v4 launcher | **HOLD / incompatible** | Never relabel; implement sibling v4 chain. |
| New v4 engineering bundle construction | **GO to implement locally** | New files, exact hashes, two deterministic builds, full tests, independent audit. |
| v4 remote stage | **HOLD** | Frozen audited bundle plus explicit staging authorization and collision check. |
| v4 import/JIT | **HOLD** | Same-bundle staged closure; no reuse of job `9367063`. |
| v4 Frontier one-update | **HOLD** | Fresh same-bundle import result must pass. |
| v4 Frontier terminal engineering | **HOLD** | Same-bundle one-update, v4 evaluator/assembler/finalizer closure. |
| v4 matched terminal readiness | **HOLD** | Both Frontier and MaxMC terminal engineering packages pass. |
| 100-update cost protocol/bundle | **HOLD** | Separate frozen identities, dedicated configs, telemetry writer, tests, audit, and explicit cost-only authorization. |
| N=8 paired 100-update cost run | **HOLD** | Both-arm terminal readiness and cost-bundle import gate. |
| Full N-factorial resource readiness | **HOLD** | N=2 resource sentinel plus N=2/N=4/N=8 import/one-update shapes and exact six-cell generic driver. |
| Any multi-seed development/performance run | **HOLD** | Separately frozen campaign/protocol, resource evidence, endpoint authorization, and analyzer closure. |
| Paper/confirmatory evidence | **HOLD** | Cannot be inferred from engineering or cost receipts. |

## Exact future freeze checklist

Before the first v4 stage command exists, the launch review must be able to answer yes to every item:

1. Are all v3/v1 bytes and job `9367063` identities unchanged?
2. Does the new stage script use only the v4 namespace and create-only destinations?
3. Does its outer manifest bind the exact finalized v4 core, v2 protocol/configs, new wrappers, environment, upstream source bundle, tests, and protected lineage?
4. Does the bundle state allow only import, one-update, and two arm-specific terminal engineering endpoints with maximum one student update?
5. Do evaluator/assembler/finalizer understand `plr-replay-snapshot.json` for both arms and refuse the v1 snapshot schema?
6. Do driver/campaign/config/xpid/checkpoint/output identities fail closed on any mismatch, `latest`, overlap, resume, or nonfinite/nonpositive temperature?
7. Do tie/replay diagnostics separate distribution support from realized duplicate draws and accurately state that `force_unique` does not resample?
8. Does Phase A stop at closed components and Phase B require terminal `COMPLETED 0:0` plus authoritative `sacct` before fetching/assembly?
9. Are all engineering packages permanently non-paper and analyzer-ineligible?
10. Is the 100-update code absent from the max-one-update bundle and governed by a different, explicitly authorized cost-only manifest?
11. Is calibration telemetry observational, separately versioned, leakage-tested, and closed without changing PRNG/training/checkpoint behavior?
12. Were two fresh local bundle builds byte-identical, all focused pinned tests green, and an independent frozen-byte audit GO?

Until all twelve close, the correct launch decision is HOLD.

## File-write declaration

This report is the only file created by this audit. No existing file was edited. No remote state was read or changed, no job was submitted, and job `9367063` was not touched.

**FILE WRITES STOPPED.**
