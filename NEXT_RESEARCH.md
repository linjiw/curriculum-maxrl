# Next research steps (post review-3, 2026-07-31)

> **Status 2026-08-01:** #4 (bridge) DONE — see
> `curriculum_maxrl/BRIDGE_ANALYSIS.md`: mass ties exact first-order LP
> as a predictor; the review's variance tilt is a *horizon-value*
> surrogate and survives posterior noise (α∈[1,2] robust band); no
> myopic utility is the right sequential objective. #3 (u_N maze rerun)
> RUNNING — `maze_gpu/sweep_un_form.sh` queued behind the H6 seed-2 pair
> job, with pre-registered P-U1/P-U2 (incl. first GPU test of the tilt).
> NEW: E-LLM-3 (Jugs) selected via `BENCHMARK_SURVEY.md`; infrastructure
> + feasibility harness built and queued (`jugs_llm/`). #1 (GSM8K
> steering-controlled replication) and #2 (gate dose sweep) are next in
> the GPU queue after feasibility.

Ranked by (decisiveness for the paper's thesis) × (cost). Each item names
the claim it would settle, the design, and the pre-registration hook.

## 1. Steering-controlled GSM8K replication (decisive for Q2 at LLM scale)

**Claim at stake:** the LLM-scale teacher×estimator interaction (GRPO+teacher
regresses; MaxRL+teacher doesn't) is currently 1-of-2 seeds and tracks
delivered steering intensity.

**Design (per §6.7's own diagnosis):**
- 3 seeds × {grpo+teacher, grpo+uniform-with-replacement, grpo} — the
  with-replacement control cell separates the prompt-repetition confound
  (~34% unique prompts under the teacher vs ~43% uniform) from steering.
- Fix delivered treatment: raise steering intensity by warm-starting the
  posterior from the registered run's final posterior (the "map is real"
  result says it transfers), or restrict the pool to 1–2k prompts so
  3,200 draws are no longer starved.
- Pre-register: regression reproduces iff min dead-sampled fraction < .50;
  eval every 5 steps at n≥16 on the 1319-prompt test split; quote nothing
  inside the measured noise floor (mean@4 SD .0094 at n=4/256 — n=16/1319
  brings it to ~.002).
- Also resolves: the k-sweep vs trainer-val 3× level discrepancy (run both
  harnesses on one checkpoint, reconcile before quoting).

**Cost:** ~9 × 20 A10G-hours.

## 2. gate_max_p dose sweep on Countdown (turns the dial claim into a curve)

**Claim at stake:** "the gate threshold traces a monotone mean-vs-coverage
frontier" currently rests on 3 arms (none / under-gated-by-bug / full).

**Design:** corrected decay, gate_max_p ∈ {0.3, 0.5, 0.7, 0.9}, 3 seeds,
frontier tier as the primary meter (tier 1 saturates). Pre-register
monotonicity of (mean gain, coverage) in gate_max_p. This also replaces
the bug-sampled operating point with a designed one.

**Cost:** ~12 × 6 A10G-hours.

## 3. u_N-form maze rerun (repairs contribution 1's only real-gradient rung)

**Claim at stake:** the maze teacher runs the legacy heuristic
(1−(1−p)^N)(1−p), not the derived u_N — so the flagship "real gradients"
rung doesn't test the theorem. §6.1 shows the two are within noise on
chains; show it (or refute it) where it matters.

**Design:** swap `maze_gpu/train.py:79` to u_N, rerun champion + teacher-only,
3 seeds, matched clock AND matched steps (report both, per the step-matching
lesson). Pre-register: within-noise equivalence to the legacy form.

**Cost:** ~6 × 2 GPU-hours.

## 4. Mass → learning-progress bridge (the theory gap a referee will hit)

**Claim at stake:** Remark "scope of the surrogate" concedes mass ≠ expected
improvement; the oracle-mass result (Thompson collects 99.6% of oracle mass
but 0.700 vs 0.851 AUC) shows mass-at-collection saturates while outcomes
don't. What does predict outcome?

**Design (CPU, cheap, high theory value):**
- On the skill-chain testbed, log per-group realized gradient norm, cosine
  to the eval-improvement direction, and post-step Δeval. Regress Δeval on
  (a) mass, (b) mass × score-geometry factors, (c) the variance-tilted
  utility the review found beating u_N 10/10.
- Goal: either a Proposition 8 (mass × per-task step-size bound → first-order
  improvement bound under stated assumptions) or an honest section stating
  which factor dominates in which regime.
- The compounding ODE model already explains γ>1; fold it in.

**Cost:** CPU-days; mostly analysis.

## 5. One more seed each for the starred single-seed claims

Per EVIDENCE.md's own rule ("worth one more seed before external claims"):
- maze GRPO+teacher arm (the amplification cell) — seeds 1–2;
- maze efficiency checkpoint pair — second checkpoint pair from seed 1;
- coverage-meter claim (L6 0.125→0.438) — recompute on all 3 champion seeds.

**Cost:** ~8 GPU-hours total.

## 6. Hindsight exactness probe with discriminating power

**Claim at stake:** Prop 6's empirical support is now behavioral only (the
placebo battery). Build the probe the cosine table failed to be: an
environment where the conditional laws provably DON'T match (goal-dependent
execution), measure the predicted degradation as a function of the mismatch
(TV distance is computable on gridworld), and verify the sup‖∇log m‖·TV
bias bound's shape.

**Cost:** CPU-days. Turns Prop 6 from "characterized" into "tested".

## Standing hygiene rules (from three review rounds)

1. Ship no number whose artifact isn't committed (the 85% timing figure is
   now `gsm8k_partial/generation_timing.json`; V5 is now
   `results_baselines_regimes.json`).
2. Every efficiency/multiplier claim states its target convention inline.
3. Min-vs-mean: any "driven to X" telemetry claim reports the run mean
   beside the extremum.
4. Single-seed results are labeled in the sentence that states them, not
   in a footnote.
5. When an internal doc retracts a claim, grep the paper, site, PAPER.md,
   and EVIDENCE.md's claim table the same day (this round found five
   stragglers from commits that had already "fixed" them).
