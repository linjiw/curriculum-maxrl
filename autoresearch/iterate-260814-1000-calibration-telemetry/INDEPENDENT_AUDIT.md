# Independent calibration-telemetry audit

**Verdict on the audited 37-test candidate:** HOLD for promotion or run authorization  
**P0:** none  
**P1:** terminal PPO-update attribution is not bound to the cycle ledger  
**Scope:** static/synthetic only; no endpoint, GPU, Hopper, or performance data
was accessed.

This report is preserved as the finding that triggered the repair. A newer
candidate ultimately closed both the clock P1 and its P2 hardening. Its final
zero-finding static verdict is recorded in `INDEPENDENT_REAUDIT_FINAL.md`;
this historical HOLD applies only to the obsolete hashes below.

## Frozen candidate audited

- protocol:
  `8f786b4b66fe1f255b3bf00c05ad0d7378f614a88a26fe4d0440bc6076202511`
- analyzer:
  `59abc1fb54c0c32dd709349462a26597a29d6b1981771a5f3226679a74c7818f`
- tests:
  `d78202599fb106c183151e2c7be730568b97cc5ff0d461e37e4c2752e714f3b4`

The authored 37-test suite and 25-artifact preflight pass. The repaired dual
clock is otherwise correct: cycles drive transitions/groups; PPO updates drive
optimizer applications; `n_iters=cycles`; `n_updates=n_grad_updates=student
PPO updates`; matched arms may have unequal exposure.

## Promotion blocker

Two hostile closed packages were accepted:

1. two all-new/non-replay cycles with a receipt claiming one PPO update;
2. one cycle mixing replay and new groups, also claiming one PPO update.

The event ledger contains no pre/post per-cycle update counters or bound update
flag. Receipt update totals are internally self-consistent, but are never
derived from the cycles that purchased the logged groups. Both fixtures are
impossible under the validated robust-PLR runner, where the warm-up/new cycle
has zero updates and the forced-replay cycle has one.

Required repair:

- add sibling-invariant per-cycle pre/post `n_iters`, `n_updates`,
  `n_grad_updates`, and optimizer-application counters (or a separately
  hash-closed equivalent ledger);
- enforce continuity and a per-cycle update delta in `{0,1}`;
- bind every cycle to its actual runner branch and reject impossible
  mixed-source cycles;
- bind receipt totals to final post-cycle counters;
- require the terminal cycle to move from `target-1` to `target`, with no
  post-target cycle;
- move the positive unequal-exposure fixture's extra exploratory cycle before
  its terminal replay and add hostile tests for all bypasses above.

## Gates that held

Independent checks retained the exact activity/Beta math, same-cycle sibling
snapshots, prior/post continuity, canonical duplicate-new and level ownership,
event ordering, slot generations, eviction termination, partial/nonfinite
rejection, MaxMC previous-score continuity, fixed bins, training-seed
independence, package closure, expected external hashes, provenance, and the
remaining budget equations.

P2 hardening: distinguish campaign artifact roles where required; do not let
`compare_matched_runs` trust a mutable `package_validated` marker; normalize
malformed type/overflow failures to the contract's `TelemetryError`.
