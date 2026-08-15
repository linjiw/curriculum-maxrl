# Status

**State:** active

**Snapshot:** 2026-08-14T17:10:31Z

## Current phase

Engineering validation of the frozen Frontier overlay and its GPU launch
ladder. Pinned `minimax` DR and PLR CPU smokes, the final Frontier unit/grouped
suite, full-horizon three-update Frontier and bridge smokes, and a Frontier
checkpoint resume all pass under `/data/robotixx/ued_bench`. No claim-bearing
benchmark endpoint has been inspected.

## Passed gates

- `facebookresearch/minimax` pinned at
  `d053054c5290a04c1c4cd8b55704d999cad73e30` (Apache-2.0).
- Archived `facebookresearch/level-replay` pinned at
  `ccecf452ee3342217ece964aaf10c2831625f9b3` (CC-BY-NC-4.0; reference only).
- Python 3.10.19 / JAX 0.4.31 source-faithful CPU environment resolves and
  passes a 70-package compatibility check.
- One-update DR smoke: 16 environment steps, checkpoint/log/meta written,
  14.74 seconds wall clock, 746368 KiB peak RSS.
- PLR smoke: buffer warm-up followed by three robust replay updates, 64 total
  environment steps, checkpoint/log/meta written, 21.84 seconds wall clock,
  966232 KiB peak RSS.
- Final Frontier overlay contract
  `5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`
  passes 20 focused and parsed-config grouped tests. Exact grouped Frontier and
  the 32-by-1 bridge each reach three PPO updates; the exact arm resumes to
  update four with posterior trials continuing to 160.
- Two focused grouped-RNG tests pass: the eight copies of each task share the
  intended reset key/state, while all 32 streams receive distinct step keys
  and the pinned categorical policy path produces non-broadcast batched draws.
  This validates RNG separation, not empirical outcome independence.
- A separate local Blackwell probe with JAX/JAXlib 0.6.2 and CUDA 12.9 wheels
  detects the RTX 5090 as compute capability 12.0, completes a 1024-square
  matrix JIT, and reproduces the analytic Beta-posterior Frontier activity to
  float32 tolerance. In an isolated clone, a two-line `jax.tree_map` API patch
  then makes a real AMaze reset and one environment step JIT on the GPU; three
  probe-contract tests and `pip check` pass.
- A complete 35-call `jax.tree_map` modernization overlay passes 16 archived
  upstream tests, 22 Frontier/project tests, and five static contract tests.
  Its exact one-update CPU protocol matches JAX 0.4.31 to maximum aggregate
  absolute error `5.960464477539063e-08`, with exact counters and checkpoint
  round trips.
- The manuscript now states the exact deterministic Beta-posterior priority
  `E[u_N(p)] = 1 - (b)_N/(a+b)_N - a/(a+b)` and its closed-form Jensen gap.
  The paper's coefficient test independently verifies both by 256-point
  quadrature on six posterior/group-size settings.
- A matched-development preregistration/analyzer freezes five paired seeds,
  three validation mazes, 30,000 student PPO updates, 150,000 optimizer-step
  applications, terminal package requirements, and the keep rule. A new
  loop-owning terminal driver and separate evaluator save the true terminal
  checkpoint, raw 30-episode evaluation outcomes, budget receipts, and a safe
  per-slot Frontier snapshot. The schema-2 atomic assembler retains both
  source closures and refuses incomplete or drifted packages. The drivers
  derive and verify the three 450-step validation horizons (13,500
  transitions/evaluation), bind the exact frozen campaign and every analysis
  executable before endpoint code, and pass the complete 31-test
  train/evaluate/assemble/analyze suite. Full launch remains held on a bounded
  two-phase terminal-chain Slurm smoke and a 100-update cost/package smoke.
- The bounded two-phase terminal chain now has an independently audited,
  byte-reproducible engineering bundle: ID `06ffeeeb6998e8ddb1ce`, manifest
  `06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff`.
  Independent review found no P0/P1/P2 issue after wrapper, staged Phase-A/B,
  timing/provenance negatives, 31 driver/package/analyzer tests, and 22 overlay
  tests passed. The exact bundle and unchanged environment are verified on
  Hopper. Fresh prerequisite import/JIT job `9367063` is pending for priority;
  no stdout, result, or performance endpoint has been opened.
- Tie-aware overlay v4 is independently green for bounded engineering at
  contract `3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b`.
  Its deterministic segmented transform preserves tie-block rank mass, makes
  actual replay probabilities permutation-equivariant, leaves the default
  path exact, and passes 34 core/legacy tests plus one one-update and six
  terminal/two-arm tests. Its v2 protocol remains DRAFT and production false.
- The outcome-blind `N={2,4,8}` factorial DRAFT is now hash-closed: protocol
  `81a57668d3cfdf595f13710df6152a437b8c4640791fbeeed2ef8c9e9486f26f`,
  manifest `58e1ffd9c7e3d80992971b331c540d6c8976c9cd4082391fae92de0df4fd417f`.
  Independent pinned-CPU validation passed 12/12 tests, all six matched configs
  and unique generated XPIDs, cross-`N` checkpoint rejection, and exact
  within-pair budget/checkpoint gates. This authorizes no endpoint or Hopper
  run; cross-`N` effects are explicitly joint `N/n_eval/n_parallel/buffer`
  effects rather than pure estimator-`N` effects.
- The separate calibration-telemetry DRAFT now specifies pre-batch Frontier
  forecasts, realized coefficient activity, MaxMC discrimination-only
  telemetry, slot lifecycles, and exact provenance. Independent review caught
  and blocked an invalid equality between robust-PLR outer cycles and PPO
  updates. The repaired contract freezes target updates plus a cycle cap,
  records realized `n_iters` and `n_updates` separately, derives transitions
  from cycles and optimizer applications from updates, and permits matched
  arms to report unequal realized exposure. Its 37 local tests and 25-artifact
  preflight pass, but independent re-audit found a second P1: the group ledger
  can contain only new or mixed-source cycles while the receipt claims a PPO
  update, because no per-cycle pre/post runner counter binds update attribution.
  The refrozen repair adds sibling-invariant pre/post runner counters, strict
  branch/delta/continuity and terminal-target equations, receipt binding, and
  P2 role/immutability/type hardening; 47 local tests pass. Independent
  re-audit gives scoped GO with no P0/P1 and independently rejects 11/11 clock
  attacks. The final P2 refreeze now enforces target-plus-one cycle feasibility,
  bounded counter products, device/inode alias rejection, exact package
  revalidation inside comparisons, and reachable dispositions; 48 tests pass
  under protocol `4053c520...` and analyzer `19b07d2f...`. Final independent
  re-audit reports `P0=P1=P2=0`. This is static specification/analyzer GO only:
  implementation, Cost-100 dependency, endpoints, and paper use remain held
  until a separately versioned telemetry overlay, writer, driver, campaign,
  and runtime audit exist.
- An adversarial compact-manuscript review found the coefficient theory
  defensible but holds competitive-method framing. The immediate artifact P0
  is fixed: `reproduce.sh --build` now targets `main_iclr2027.tex`, the paper
  reports the actual local 53-row registry (35 maze/11 Countdown/7 GSM8K), and
  the trace is contiguous at 62/62 claims (35 exact, 27 rounded). Non-build
  reproduction passed in a disposable copy. Although `pdflatex` is unavailable,
  the pinned cached Tectonic path below now supplies the verified PDFs.
- The focused main-text rewrite is complete. A contribution/evidence map now
  separates the proved coefficient identity, the scoped direct fixed-pool
  teacher, and engineering-only coefficient-activity PLR. The direct Acrobot
  score test leads Evidence; the estimator-only maze result is explicitly
  secondary; paid-probe, recycling, Countdown, and gate details are preserved
  only in the appendix. AMaze, robotics, and PLR/ACCEL performance remain
  explicit `HOLD` claims. Static LaTeX/reference/62-row-trace checks pass.
- A pinned Python 3.11.13/Matplotlib 3.11.1/font environment and Tectonic
  0.16.9 cache now close the deterministic manuscript-artifact path. Two
  isolated figure renders and two cached TeX builds at fixed
  `SOURCE_DATE_EPOCH` are byte-identical. The compact and website PDFs hash to
  `36a6c1fb...` (15 pages); the extended draft hashes to `25023b85...` (25
  pages). The conclusion is on page 8 and references begin on page 9.
  `PAPER_BUILD_RECEIPT.md` binds all 13 embedded figure PDFs, package/font
  records, Tectonic URL mapping and 483-member cache tree, inputs, outputs,
  logs, and non-fatal warnings without local account/path metadata. External
  tool assets still require a checksum-bound release bootstrap or container
  for unrelated-host clean-checkout reproduction. Independent audit then
  rebuilt all figures/PDFs from output-free staging, fault-injected publication
  rollback, and returned `GO` with no P0/P1/P2 finding; its frozen record is
  `PAPER_BUILD_AUDIT.md`.

## Active holds

- The source-faithful JAX 0.4.31 stack cannot compile for the RTX 5090. A sole
  authorized JAX 0.6.2 GPU PPO update completed with exact structural counters
  and checkpoints, but failed the frozen numerical gate: the
  `params/fc_pi_1/bias` absolute-sum error was `2.0395550519458627e-04`
  against `atol=5e-05`. A subsequently frozen, non-updating component capture
  localizes the first CPU/GPU failure to the cycle-one recurrent LSTM carry
  (9 elements, maximum absolute error `1.2597441673278809e-4`); tasks, actions,
  observations, and minibatch permutation are exact, clipping is inactive, and
  the Adam proposal matches the analytic post-clipping formula. Keep the 5090
  training gate closed. A separately frozen forward-only trace now localizes
  the earliest error to the time-zero LSTM input GEMM: convolution/features
  pass, while all four input-affine gates fail and the maximum error is
  `1.825392246246338e-4`. A highest-precision forward-only dot probe then passes
  every recurrent stage (final-carry maximum `5.9604645e-8`) while CPU
  default/highest records remain byte-exact. A separately hashed two-line
  compatibility overlay then completed one CPU PPO update with recovered
  546/546 aggregate and 24/24 floating-statistic parity, but its wrapper failed
  while writing the required outer receipt after consuming the frozen update
  budget. The raw base receipt/checkpoints were audited read-only; the gate is
  `INCOMPLETE/HOLD`, and no 5090 update was attempted. 5090 training and
  performance endpoints remain held.
- The paper is not yet a competitive PLR/minimax method paper. Its direct
  positive score evidence is the small Acrobot family; the existing neural
  maze study compares estimators rather than task scores; and the manuscript's
  direct Thompson teacher is not the same algorithm as coefficient-activity
  PLR with replay rank, staleness, grouping, and tie handling. Keep AMaze,
  ACCEL-superiority, N-factorial, and robotics claims out until the exact
  method/evidence identities close.
- Hopper A100 import/JIT attempt 9366785 failed before source import with exit
  127 because `git` was available on the login node but absent on compute node
  `gpu021`. Its immutable submission bindings passed independent audit and it
  produced no `COMPLETE` artifact. The replacement environment must contain
  an exact Git build; do not reuse or relabel the failed attempt.
- Replacement bundle `6c2ca94ca8109be2775c` removed the host-GNU-time
  dependency and retained exact Conda Git 2.45.2. Fresh import/JIT job 9366896
  completed `0:0`; its closure passed. Dependent one-update job 9366897 then
  completed `0:0` in 1:42 and independently verified exactly one PPO update,
  one upstream gradient counter, five optimizer applications, 64 posterior
  trials, zero delivery counters, and checkpoint/Optax/PLR continuity. This
  clears only the bounded overlay-training rung, not the terminal campaign.
- The full 30k terminal chain remains no-go: local source/package/schema gates
  pass, but no two-phase terminal Slurm run has yet combined real training,
  actual external evaluation, authoritative post-exit `sacct`, terminal logs,
  and atomic analyzer-ineligible assembly. A representative 100-update cost
  run is also still missing.
- No AMaze development performance run is authorized under the current v1
  ranking path. Upstream stable sorting gives exactly tied scores different
  inverse-rank weights; at `temp=.3`, rank 1 carries about 87.16% of the
  score-only distribution. A new N=8 group has only nine possible posterior
  scores, so buffer slot order can dominate tied replay choices. The terminal
  Slurm smoke may still validate engineering. The independently green v4
  transform removes this bias in a separate DRAFT, but it still needs a fresh
  content-addressed Hopper runtime ladder, cost rung, and production protocol
  before endpoint access.
- `force_unique=true` does not enforce unique replay tasks: sampling is with
  replacement and deduplication acts only on buffer updates. Under the
  score-only `.3` rank distribution, four draws have about 1.468 distinct
  levels in expectation and repeat rank 1 four times with probability .577
  before staleness mixing. The v4 gate must log realized replay diversity; any
  without-replacement sampler is a separate matched ablation, not part of the
  score-only primary.
- Tie-aware overlay v4 is now independently green for bounded engineering:
  contract `3d5f3827...`, applied manifest `9b411f61...`, no P0/P1, 34+1+6
  core/E2E tests, hostile permutation/ESS checks, default/sampling parity, and
  exact receipt/checkpoint closure. Its new v2 protocol is deliberately DRAFT
  with production authorization false. No development result may run until
  that protocol, the 100-update cost rung, and a fresh v4 Hopper ladder are
  frozen separately.
- A sibling-only v4 Hopper candidate is locally content-closed at bundle ID
  `d602ce7854f8f3e99352`, manifest
  `d602ce7854f8f3e99352025b97eed2fde32733c0dd23297d5c28b1051e7aeaf0`.
  Twin bundles are byte-identical; local import, Frontier one-update, both-arm
  terminal Phase-A/B (60 sealed evaluation rows), and fresh-clone 22/22 JAX
  tests pass. Independent audit grants only `LOCAL-ONLY GO`: no staging or
  submission until Hopper export/MIG/sacct, fixed Conda, and a manifest-bound
  non-test Phase-B Python environment are verified. Cost, production,
  analyzer, endpoint, and paper-evidence paths remain disabled.
- A subsequent remote-contract hardening snapshot is frozen locally at bundle
  ID `da74eb3e0debc7781d6d`, root manifest
  `da74eb3e0debc7781d6d785f9406acec953a02cfcc3674afeb70c0f438619cc8`.
  Its twin closure, seven focused adversarial tests, Slurm-spool simulation,
  syntax, and prior two-arm R3 fixture pass. Independent audit nevertheless
  keeps the full ladder and every remote action on `HOLD`: the widened sibling
  overlay conflicts with the preserved d602 R1/R2 closure, R2 disagrees on
  `job-<id>` identity, the GPU probe starts under system Python rather than the
  byte-closed environment, and MIG `gpumem` is incorrectly treated as a
  required Slurm counter. The frozen snapshot deliberately implements no
  submit operation.
- The six-cell `N={2,4,8}` package is also engineering-only. Its direct parser
  defaults to unsafe `xpid=latest`, five development seeds cannot yield a
  two-sided exact sign-flip p-value below `.0625`, and its current protocol
  forbids performance/OOD access, production scheduling, and paper-evidence
  labeling. Do not submit these configs directly or reinterpret the joint
  layout factor as estimator `N` alone.
- Hopper MAZE full-arm engineering job 9366552 completed in 22:22. Its strictly
  endpoint-blind cost/schema audit passed; peak GPU memory was 39,672 MiB on
  `3g.40gb`, so that slice must not be reduced. Paper evidence remains held.
- A separate frozen BARN process canceled campaign
  `barn-icra2027-20260814-002` outcome-blind after an unsupported publication
  operation and launched replacement campaign `barn-icra2027-20260814-003`
  with the same 20 sealed CPU tasks. This UED iteration has
  inspected no BARN log or endpoint and must not interfere. Its stratum-pooled
  score is a heuristic unless post-merge per-course data support the required
  within-stratum homogeneity assumption; it is not source-faithful PLR.
  A scheduler-only check at 13:10 EDT found all 20 tasks still `RUNNING` after
  about 6h34m; no task stream or result was accessed.
- Upstream `minimax` uses removed JAX APIs. Keep the JAX 0.4.31 baseline path
  separate from any current-JAX/Blackwell compatibility patch.

## Gate order

1. Read paper/repositories and identify exact benchmark/config/metric.
2. Pin upstream revisions, licenses, dependency locks, and evaluator assets.
3. Audit local RTX 5090 and Hopper without disrupting active workloads.
4. Pass CPU import/unit smoke, then a bounded local GPU smoke.
5. Reproduce one upstream baseline on development seeds.
6. Implement FrontierRL as a minimal score-only intervention with tests.
7. Run matched development comparisons; keep/discard by frozen metric.
8. Freeze confirmatory seeds/budget and only then schedule Hopper replications.
