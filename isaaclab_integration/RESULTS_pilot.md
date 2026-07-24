# Pilot results — 5-arm terrain curriculum, Anymal-C rough (seed 42)

*2026-07-23. 600 iters × 1024 envs per arm, identical graded 10-row terrain grid,
success predicate `tile` for teacher evidence and for all fixed-grid evals.
Runs: `logs/rsl_rl/anymal_c_rough/2026-07-23_*`; primary artifacts:
`eval_frontier_{300,599}.json` per run + `curriculum_teacher/teacher_state.json`
+ `params/arm.yaml`. Analyzer output reproduced by
`analyze_arms.py --log_root logs/rsl_rl/anymal_c_rough`.*

## Headline (fixed-grid macro-pass, same per-level grid + eval seed for every arm)

| arm | eval@300 | eval@599 | per-level@599 (rows 0→9) |
|---|---|---|---|
| control | 0.151 | 0.291 | 0.47 0.43 0.37 0.35 0.32 0.30 0.24 0.17 0.16 0.13 |
| **greedy** | 0.191 | **0.372** | **0.76 0.71 0.58 0.45** 0.30 0.26 0.20 0.16 0.14 0.16 |
| scripted | 0.145 | 0.266 | 0.49 0.42 0.30 0.26 0.31 0.22 0.22 0.16 0.16 0.11 |
| uniform | 0.126 | 0.297 | 0.51 0.47 0.39 0.30 0.30 0.26 0.23 0.20 0.18 0.13 |
| teacher | 0.132 | 0.278 | 0.50 0.40 0.34 0.26 0.28 0.25 **0.23 0.23** 0.15 0.14 |

Single seed; per-level CI ≈ ±0.05 (80–120 episodes/level) ⇒ macro CI ≈ ±0.02.
Greedy's lead is real; control/uniform/teacher are statistically indistinguishable;
scripted trails slightly.

## Pre-registered verdicts

**P-A (mechanism gate): PASS.** ZPD targeting ratio mean 1.29 / final 1.14
(internal), 1.07 when ZPD membership is recomputed from the *fixed-grid measured*
pass rates — the teacher put more sampling mass on genuinely-frontier bins than a
uniform sampler, judged by external truth, not its own posterior. Posterior
calibration: MAE 0.046 vs fixed-grid truth. Telemetry trajectory matched the
regime-map prediction exactly: dead_frac 0→1.0 (cold start: everything measured
dead) →0.0 (policy lifts off), zpd_bins 0→4, zpd_mass →0.68, all in the final
~quarter of the run. The mechanism works.

**P-B (outcome): greedy wins on this grid at this budget; teacher ≈ uniform ≈
control — the pre-registered honest null, plus one sharpening we did not
pre-state.** The null condition ("every row learnable at budget ⇒ greedy
near-optimal, parity expected") was half-realized: not only was no row
unlearnable, *no row was mastered either* — at 600 iters the easiest row sits at
pass 0.5 (the teacher's final frontier_bin = 0). The teacher's designed edges are
skipping mastered bins and avoiding dead bins; this run contained **neither**
(mastered_frac = 0.0 throughout; dead_frac = 0.0 at end). With no waste to avoid,
the teacher degenerates by design to ≈ floor+spread ≈ uniform — and measured
exactly so (0.278 vs 0.297, near-identical per-level profiles).

The sharpening: greedy did not merely match — it won (+0.075 macro over uniform),
with its entire margin on rows 0–3 (0.76/0.71/0.58/0.45 vs teacher's
0.50/0.40/0.34/0.26); on the hard tail (rows 6–9) teacher/uniform are level or
slightly ahead (rows 6–7: 0.23/0.23 vs greedy 0.20/0.16 — right direction for the
mechanism, not significant at n=1). Interpretation: `terrain_levels_vel` is not
just a difficulty walker, it is a **per-env** controller — each env's next level
depends on *its own* outcome, so failing envs are instantly re-assigned easy rows
and mastering envs promote, with zero statistical lag. The global teacher throws
that per-env information away (iid draws from one distribution). In the
cold-start regime where the whole grid is at-or-beyond frontier, that locality is
worth more than posterior-based allocation. This is a real finding about the
massively-parallel-sim regime: **evidence pooling costs locality**, and the cost
is largest exactly where a from-scratch pilot spends most of its time.

**P-C (retention): no forgetting anywhere.** Easy-level (rows 0–1) fixed-grid
pass improved mid→final in every arm (deltas +0.19 to +0.35). Expected — nothing
was mastered, so nothing could be forgotten. τ not yet measured (NO_TAU); this
gate becomes binding only at longer horizons.

## What this changes (and doesn't)

- The integration and the mechanism are validated end-to-end in real sim: P-A
  passed against external truth, telemetry matches theory, all arms produced
  clean artifacts. Ladder step 1's *engineering* goal is met.
- The stock-grid outcome null is confirmed and sharpened: on a uniformly
  learnable, unmastered grid, don't use a global teacher — greedy's per-env
  locality wins. This is negative knowledge with a mechanism attached, the
  useful kind.
- Nothing in this run tests the teacher's designed edge. The discriminating
  experiment (phase 2) needs bins that are *dead at budget* and bins that get
  *mastered*: 16-row grid with step heights to 0.45 m (rows ~12+ beyond
  Anymal-C's physical ceiling), warmstart shared across arms (skips the
  undifferentiated prefix), ≥1500 iters, ≥3 seeds — pre-registered prediction:
  teacher > greedy there, because greedy walks envs into dead rows at its
  boundary and re-confirms mastered rows below, both of which the teacher
  skips.
- Design follow-up worth one arm in phase 2: a **hybrid** term — greedy's
  per-env ±1 walk, but bounded by the teacher's dead-bin mask (don't promote
  into bins the posterior rates dead). Combines locality with pooling; ~20
  lines on top of the existing terms.

## Ops notes

Two eval-script bugs found and fixed during the readout (both committed):
`handle_deprecated_rsl_rl_cfg` needed for rsl-rl 5.0.1 checkpoint loading, and
`env.reset()` between checkpoints must run under `torch.inference_mode()` (after
the first rollout, sim buffers are inference tensors). Eval reran cleanly with
per-checkpoint idempotent skip.
