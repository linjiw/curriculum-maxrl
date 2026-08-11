# Autoresearch results: status recovery, E1 repair, and E2c readiness

Date: 2026-08-10

## Decision

The most promising new research direction is E2c, an immutable-reservoir,
fixed-slot, dose-matched control for Countdown hindsight. E2 and E2b are not
negative endpoint results: both are treatment-delivery inconclusive. E2c is
designed to remove their source-support failure without weakening a prior gate.

The shared GPU was unavailable, so the bounded feasible experiment in this
iteration was the full E1 independent-unit reanalysis plus an executable CPU
test of E2c's exact cold-start failure mode.

A later remote research release was also discovered during the artifact audit.
It confirms that the historical Countdown scalar is a with-replacement VERL
bootstrap `best@k` proxy, not standard pass@k, and that the raw outcomes are
unrecoverable locally. The active worktree now preserves that distinction. E2c
will instead retain every binary task outcome and recompute standard
observed-set pass@16 before reporting a contrast.

## Metric comparison

| check | prior state | result |
|---|---|---|
| E2 current-batch source support | stopped before step 12 | reservoir cold-start unit test passes at step 1 with no current live source |
| Reservoir provenance | first reward row and task overlap | every tensor/non-tensor payload row, source/index/task identity, split uniqueness, and relabel-metadata absence gated |
| E2b seed-2 token support | best source missed 5% gate at 5.9716% | exact synthetic reservoir case delivers 0% mismatch; real three-seed support awaits frozen reservoir |
| Factorial independent unit | 24 repeated sampler contrasts described as blocks | 12 independent seed blocks, positive 12/12 |
| Wave-2 primary covAUC | 6/6 per sampler | preserved; block mean +.01950, CI excludes zero |
| Wave-2 easy-band localization | 10/12 called confirmed | registered pair bar met; block CI includes zero, now suggestive |
| Historical Countdown curve | hard-coded and called unbiased pass@k | checksummed seed-1 bootstrap proxy; descriptive only |
| Historical clean tier 0 | 27/128 overlap disclosed only in prose | machine-readable audit summary; 101-task reanalysis explicitly unavailable |
| E2c endpoint integrity | analyzer trusted summaries | raw-outcome recomputation + frozen data/task/checkpoint/seed pairing gate; full nine-arm matrix test |
| Endpoint byte identity | model path and shared seed only | post-load RNG reset; evaluator/verifier/model hashes recorded, recomputed, and matched to no-outcome delivery receipt |
| Partial endpoint exposure | evaluator printed each completed arm | per-arm output sealed until the complete nine-arm analyzer passes |
| E2c source matcher | duplicate token counts expanded identical DP states | exact source-selection parity; median 0.5564s → 0.2668s (2.09x) |
| E2c executable protocol | preregistered values remained ambient-overridable | seed/step/gate/asset/commit values locked and tested; hostile ambient override dry run passed |
| Dirty-worktree reproducibility | filenames and patch-presence tokens | 31 research/patch/runtime files and environment versions content-addressed; manifest locked by driver checksum |
| Comparator reuse | `.complete` marker plus final-step grep | 59/59 logged config checks per run, exact optimizer steps/scheduler, four full model fingerprints, semantic B2 schedules; seeds 1--2 valid for reuse |
| Delivery interpretation | displaced slots counted but threshold not emitted | frozen 25% diagnostic restored; crossing restricts interpretation to fixed-slot direction substitution without suppressing endpoints |
| Next-stage readiness | inferred from scattered markers and logs | outcome-blind receipt passes integrity and resolves B1 seed 3 as next; 2026-08-11 00:20 check held by 14,743 MiB GPU use |

## Keep/discard decision

Keep the E2c implementation: all local regression, provenance, checksum,
schedule-support, shell-syntax, import, patch, and package tests pass, and it
directly eliminates the previously observed empty-source cold start. Keep the
E1 repair: it preserves the primary result while removing an invalid
independence claim.

Keep the executable protocol hardening and readiness receipt. They close a
real preregistration-to-execution gap without using endpoint information: the
driver now revalidates completed stages and ignores attempted ambient changes
to the scientific settings. The receipt found no integrity error and no
held-out E2c artifact.

Keep the comparator-reuse audit. It replaces indirect marker evidence with
logged-configuration parity and immutable checkpoint/schedule fingerprints;
all checks pass, so rerunning seeds 1--2 would add cost without repairing a
scientific mismatch.

Do not claim E2c treatment delivery or endpoints yet. A real reservoir and the
third B2 schedule do not exist, and the GPU-safe launch guard correctly stopped
execution while unrelated jobs occupied the device.

## Final CPU completion audit

Keep all frozen E2c hardening: 25/25 package tests, the independent
mass-formula enumeration, the 31-file plus exact-environment manifest, the
outcome-blind deep receipt, and a normal guarded driver invocation all pass.
The normal invocation exits before B1 seed 3 solely because 14,743 MiB exceeds
the frozen 4,096 MiB ceiling. No E2c endpoint artifact exists. Further local
changes would either duplicate verified work or alter the prospectively frozen
experiment, so the remaining work is an external GPU-state dependency.
