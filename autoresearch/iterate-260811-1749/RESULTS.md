# Iteration results

**Verdict:** KEEP

All five keep criteria in `GOAL.md` passed. The retained scope is documented in
`ICRA2027_PROGRESS_REPORT_2026-08-11.md`.

## Evidence

- 5/5 new campaign tests passed.
- 21/21 relevant pre-existing tests passed.
- 100/100 randomized rollout-allocation invariants passed.
- Four-arm one-seed CPU navigation smoke completed and produced structured raw
  and analysis artifacts.
- Analyzer returned `decision_ready: false` and exposed the staged arm's early
  common-budget advantage.
- Official CFP check corrected the draft from 6+references to 8 total pages.
- E2c readiness integrity passed but GPU gate remained closed at 10,263 MiB.

## Remaining external dependencies

- BARN course assets/manifest and lab-specific simulator/Jackal entry point.
- A BARN training container or environment digest.
- Shared RTX 5090 occupancy below the independent E2c ceiling.

