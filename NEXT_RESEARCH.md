# Next research steps (post review-3, 2026-07-31)

> **Status 2026-08-10:** The active critical path has moved to E2c; see
> `autoresearch/iterate-260810-2240/GOAL.md`. E2 current-batch replay and E2b
> recent-buffer replay both stopped at prospectively frozen treatment-delivery
> gates, so neither supports an endpoint direction claim. E2c uses a
> checksummed, train-only frozen-SFT source reservoir and forbids held-out
> evaluation until all three replay arms pass delivery. Its preregistration,
> collector/replay implementation, static preflight, delivery validator, paired
> endpoint analyzer, and GPU-safe driver are implemented and CPU-tested. The
> endpoint analyzer now recomputes standard observed-set pass@16 from retained
> binary outcomes, avoiding the historical VERL bootstrap-best@k proxy. The
> scientific launcher values and source-asset fingerprints are now executable
> locks rather than ambient defaults. The outcome-blind launch receipt passes
> all integrity checks, reports no held-out E2c artifact, and identifies B1
> seed 3 as the next stage; refresh it with
> `bash verl_integration/run_e2c_rtx5090.sh --readiness-only`.
> Reused B1/B2 seeds 1--2 now also pass 59/59 logged configuration checks per
> run with full checkpoint and B2 schedule fingerprints; see
> `autoresearch/iterate-260810-2240/E2C_COMPARATOR_REUSE.md`.
> E3's
> historical multi-seed curve and E4's 101-task clean-tier endpoint are
> irrecoverable from surviving aggregates; the seed-1 curve is now structured
> and descriptive, and the overlap blocker is machine-readable. A later remote
> research release (`9277141`) already contains a nine-page candidate and must
> be reconciled deliberately with this E2c worktree. The
> shared RTX 5090 was occupied at the 2026-08-10 launch check. In parallel, the
> mandatory factorial independent-unit repair is complete: wave 2 remains 6/6
> per sampler and +.01950 [+.01148,+.02752] at n=6 independent blocks; the
> easy-band registered pair-level bar was met but its block-level interval
> includes zero. The ranked list below is historical context where superseded
> by this update.

> **Status 2026-08-02:** #4 (bridge) DONE through part I + GPU verdicts
> — `curriculum_maxrl/BRIDGE_ANALYSIS.md`. Final: u_N ties the exact
> first-order objective as predictor; "best within-band utility" is
> objective-dependent (AUC vs final, parts H/I) AND testbed-sensitive
> (the tilt won 10/10 on CPU, lost 0/3 on the maze — refuted per prereg,
> not adopted). #3 (u_N maze rerun) DONE: P-U1 CONFIRMED (exact u_N ≡
> legacy form on the maze, closing opus5 M4 with data). H6 seed-2 pair
> DONE: GRPO teacher-deficit direction now 2/2 seeds (dose effect, not
> sign flip; regression *shape* still 1-of-2). E-LLM-3 (Jugs) is
> launch-ready (pool, reward, JugsHindsight, launcher, SFT script all
> tested); feasibility RUNNING. Next: E-LLM-3 numeric prereg → cells;
> then #1 (steering-controlled GSM8K) and #2 (gate dose sweep).

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
