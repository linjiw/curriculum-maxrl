# Goal: restore the ICRA BARN evidence path

**Started:** 2026-08-14 00:47 America/New_York  
**Iteration bound:** 25 modify/verify/keep decisions  
**Governing brief:** `CODEX_GOAL_ICRA_2026-08-11.md`

## Objective

Advance the outcome-blind ICRA 2027 navigation campaign from a candidate BARN
adapter to a coherent, reproducible pre-evidence package.  Reconcile the real
BARN runner, frozen split, transition-primary analyzer, acquisition and runtime
fingerprints, preregistration, and Hopper CPU launch interface before any
scientific endpoint is submitted or inspected.

## Keep metric

Retain an iteration only when it clears a named preregistration or execution
gate and passes its focused tests.  The package is launch-ready only when:

1. the manifest and split are reproducible and content-addressed;
2. the production runner consumes the frozen split and emits analyzer-compatible
   full-domain artifacts without mutating evaluation/training state;
3. the analyzer implements transition-matched AUC as primary and applies the
   Aug. 24 directional gate to uniform and learnability exactly;
4. BARN acquisition and actual execution-environment receipts are committed;
5. fixed-seed simulator nondeterminism and throughput are measured and recorded;
6. the preregistration and execution bundle are internally consistent before
   the first evidence-bearing run.

## Safety gates

- No local RTX 5090 use; all BARN work is CPU-only.
- No MAZE-SCORE, E2c, ICLR paper, website, or publication work in this loop.
- Do not inspect partial scientific endpoints or launch evidence while the
  preregistration remains DRAFT.
- Existing Hopper engineering job 9366552 may be monitored by scheduler state
  only; it is not progress on this ICRA goal.
- Preserve unrelated dirty-worktree files; never commit `.codex/`.
- Do not push, publish, or deploy.

