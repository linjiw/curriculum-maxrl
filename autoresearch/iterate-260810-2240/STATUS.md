# Latest project status

**As of:** 2026-08-11 00:20 America/New_York

## Bottom line

The scientific core is strong enough for a focused paper; the remaining work
is causal accounting, artifact repair, and compression, not another broad
domain. The central thesis is:

> In RL with verifiable rewards, the advantage estimator conditions where a
> curriculum or recycler can emit and preserve update mass, and mean accuracy
> alone can hide a pass@k coverage cost.

The three defensible pillars are the exact coefficient-mass identity, the
balanced-maze time-integrated coverage ordering, and a reported three-seed
Countdown mean-up/bootstrap-coverage-proxy-down aggregate. The historical
proxy is not standard pass@k, and complete seed records are unavailable, so it
is not a replicated per-seed direction claim. GSM8K is a treatment-delivery
boundary result, not evidence for an LLM estimator-by-teacher interaction. The
saturation gate did not validate.

The checked-out `main` branch is behind the later remote research release
`origin/codex/curriculum-maxrl-research` at `9277141`, which already contains a
nine-page ICLR candidate and a broader frozen artifact release. That large
branch was not merged into the dirty E2c worktree; narrowly relevant audit and
metric-provenance repairs were ported selectively.

## Completed in this iteration

### Factorial independent-unit repair (E1)

- Added `curriculum_maxrl/maze_gpu_factorial/block_reanalysis.py` and its
  generated structured artifact `block_reanalysis.json`.
- Wave 2 registered P-F2 remains confirmed at 6/6 under uniform and 6/6 under
  FrontierMax, exact sign p=.03125 per sampler.
- Averaging the two repeated sampler contrasts within each independent seed
  block gives mean MaxRL-minus-GRPO covAUC +.0194979 with 95% t interval
  [+.0114755,+.0275202], 6/6 positive.
- Across both waves the block-level contrast is positive in 12/12 independent
  blocks, descriptive mean +.0217515 and interval [+.0166268,+.0268761].
- P-F3 met its registered pair-level bar, but block aggregation is four
  positive, one tie, and one negative; mean +.08333 with interval
  [-.00330,+.16996]. Easy-band localization is therefore suggestive.
- Added a checksummed seed-block plot and propagated the unit correction through
  the paper, verdict, provenance, evidence ledger, website, and root status.
- Also corrected known label drift: the historical two-epoch LLM replay is
  higher-dose, the gate's promising point used buggy decay, the strengthened
  GSM8K arm failed delivery, and the historical Countdown SFT overlap is now
  disclosed.

### E2c implementation readiness

- Froze `GOAL.md` and `E2C_PREREG.md` before any E2c reservoir or endpoint.
- Added an immutable reservoir collector: seed 424242, LR=0, 60x8x16 frozen-SFT
  generations, first 256 unique informative train groups retained.
- Added checksum enforcement, reservoir-only source selection, same-task source
  exclusion, CPU-to-device payload restoration, and reservoir-source auditing.
- Added a static preflight checking train/test task disjointness, original-target
  informativeness, group/token integrity, all three B2 schedules, and conditional
  token support.
- Reservoir preflight now validates every payload row and leading tensor
  dimension, train task/index uniqueness, repeated task/index/source identity,
  and absence of any hindsight/relabel metadata rather than trusting one row.
- Added a runtime delivery validator that must pass all three seeds before the
  endpoint evaluator can run, plus a seed-level endpoint contrast analyzer.
- Updated the reproducible runtime with an incremental E2c patch and added a
  GPU-safe driver that refuses to start above 4096 MiB shared memory use.
- Locked the executable protocol against ambient changes to seeds, steps,
  memory ceiling, model/data paths, and launcher settings. The preflight now
  rejects source-data/model fingerprint drift, wrong seed sets, changed gates,
  or a changed MaxRL commit.
- Added `E2C_LAUNCH_READINESS.json`, an outcome-blind receipt that validates
  frozen assets, runtime-code parity, complete-stage markers, checkpoints,
  logs, B2 schedules, stage ordering, endpoint absence, and current GPU state.
- Deep-audited B1/B2 seeds 1--2 for reuse: every run passes 59/59 logged
  configuration checks, all four step-60 weight files now have recorded
  SHA-256 fingerprints, and both 60-row B2 schedules pass fixed-slot/token
  metadata checks (329 accepted groups for seed 1; 308 for seed 2). Details are
  in `E2C_COMPARATOR_REUSE.md`.
- Hardened endpoint analysis to recompute mean@16 and standard observed-set
  pass@16 from retained 16-bit task outcomes, reject summary drift, require one
  held-out data checksum, and verify paired decoding/evaluation seeds. E2c does
  not inherit the historical VERL bootstrap-proxy ambiguity.
- The endpoint gate now also requires nine distinct raw-outcome artifacts,
  exact B1/B2/E2c checkpoint identities, evaluation seed `10000+s`, the frozen
  decoding budget, and one duplicate-free task-manifest fingerprint across all
  arms.
- Endpoint RNG is reset after checkpoint loading, and every future evaluation
  must fingerprint its exact model config/weights plus the frozen evaluator and
  verifier. The analyzer recomputes those hashes and anchors them to the final
  outcome-blind post-delivery receipt.
- Per-arm evaluator output is sealed during generation; the driver exposes an
  endpoint report only after the complete nine-arm matrix validates, preventing
  accidental partial-result inspection.
- Restored E2's frozen 25% displaced-live-slot diagnostic as an explicit
  delivery artifact. Crossing it preserves fixed-slot endpoints but forbids a
  pure extra-dose interpretation; it does not silently change the E2c arm.
- Deduplicated equivalent reservoir token-count candidates in the exact replay
  matcher. A parity test against the original dynamic program preserves source
  IDs; a five-repeat 256-source benchmark improved median selection time from
  0.5564 s to 0.2668 s (2.09x).

### E3/E4 artifact closure

- Moved the seed-1 `best@k` arrays into a checksummed structured artifact and
  relabeled the curve as a descriptive with-replacement VERL bootstrap proxy.
- Removed the unsupported crossing, paired timing, and per-seed direction
  claims from the manuscript. Historical multi-seed curves cannot be recovered
  without the missing task outcomes or checkpoints.
- Restored a tested SFT/evaluation identity audit and extended
  `data_integrity_check.json`: the stored historical count is 27/128 exposed
  tier-0 tasks and zero in tiers 1--2. The clean 101-task historical endpoint
  remains non-computable because source manifests and task outcomes are absent.
- Full evidence and decisions are in `E3_E4_AUDIT.md`.

## Verification

- Countdown protocol tests: 17/17 passed in the recovered runtime, including
  explicit rejection of seed drift and a complete nine-arm endpoint matrix
  test for delivery, checkpoint, evaluation-seed, and task-manifest pairing.
- Curriculum package tests: 25/25 passed with plugin autoload disabled to avoid
  the host ROS pytest plugin.
- Runtime imports for collector, replay, and trainer passed.
- A guarded normal-driver invocation revalidated and skipped all four complete
  comparator checkpoints, then stopped immediately at the B1-seed-3 GPU gate.
- The driver now reruns the deep outcome-blind receipt at three causal barriers:
  after all comparators, after reservoir preflight, and after all delivery
  audits but before any endpoint generation.
- Froze and verified a 31-file code/patch/patched-runtime manifest, including
  environment versions. Its checksum is embedded in the orchestrator; the
  orchestrator's own checksum is recorded in the preregistration.
- E2c patch applies and reverses cleanly against the recovered MaxRL runtime.
- Cold-start regression test passed with zero informative current groups; the
  replay source came exclusively from the checksummed reservoir with exact
  token match and no slot fallback.
- `bash reproduce.sh`: all artifact checks passed (the known optional fig2c raw
  log path was absent and correctly treated as a non-fatal miss).
- PDF compilation could not be run because no TeX engine is installed.

## Live execution state

The outcome-blind launch receipt passes every integrity check and identifies
`e2_clean_b1_s3_260809` as the exact next stage. Launch is not currently
authorized: the latest receipt found 14,743 MiB in use by unrelated work, above the
4,096 MiB safety ceiling. No Countdown GPU job was started and no partial E2c
endpoint exists.

This is now the sole completion blocker. The final content-addressed driver was
invoked normally at 00:20, passed its 31-file/environment manifest and all
readiness checks, revalidated/skipped the four reusable comparators, and exited
1 before B1 seed 3. The persistent external occupants recorded in the receipt
include the Cosmos critic daemon (7,350 MiB) and OpenPI evaluation (4,545 MiB);
they already exceed the frozen ceiling without counting CUDA overhead or other
processes. They are outside this project's authorized scope and were not
stopped.

Already complete and reusable: B1/B2 seeds 1 and 2. Still required, in order:

1. B1 and B2 seed 3, without held-out evaluation.
2. Frozen-SFT reservoir generation and three-schedule static preflight.
3. E2c replay training for seeds 1--3.
4. Three-seed delivery validation.
5. Only then, paired step-60 B1/B2/E2c held-out generation and analysis.

## Next best work

At any time, refresh the non-launching receipt with
`bash verl_integration/run_e2c_rtx5090.sh --readiness-only`. When it reports
`launch_authorized_now: true`, run `bash verl_integration/run_e2c_rtx5090.sh`;
its stop rules enforce the order above. E3 and E4 are now closed at the limit of the
surviving historical artifacts, and the later remote branch already contains a
nine-page candidate. Do not add a new domain or larger model before E2c is
resolved. Repository integration should reconcile the E2c worktree with
`9277141` deliberately rather than merging the large release branch blindly.
The file-level integration sequence and scientific conflict rules are frozen in
`BRANCH_RECONCILIATION.md`.

No smaller model, CPU surrogate, relaxed occupancy ceiling, partial endpoint,
or replacement protocol can satisfy the frozen goal. Completion requires the
external GPU state to change, followed by the unchanged driver command above.
