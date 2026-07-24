# Phase 2 — from diagnosis to a competitive frontier curriculum

*2026-07-24. Follows RESULTS_pilot.md. rsl_rl PPO is the fixed baseline learner
throughout; all improvement happens in the curriculum layer.*

## 1. The task, re-read as an RL-agent problem

Before tuning anything we re-derived what "task" and "difficulty" actually are
in `Isaac-Velocity-Rough-Anymal-C-v0`, from the terrain generator source:

- **The agent** (ANYmal-C, 12 joints, PPO at 50 Hz) is ONE shared policy serving
  4096 parallel episodes; goal-conditioning (commanded velocity) is in the obs,
  terrain is only observable through the height scanner.
- **A "task bin" is a terrain ROW.** The generator varies difficulty ∈ [0,1]
  linearly over `num_rows=10` (`terrain_generator.py:236+`): row r has
  difficulty ≈ (r + η)/10, η~U(0,1) per tile. Physical meaning per family:
  stairs 5→23 cm, boxes 5→20 cm, rough noise 2→10 cm, slopes 0→22°.
- **A terrain COLUMN fixes the terrain FAMILY** (`_add_curriculum_terrains`):
  cols 0–3 pyramid stairs (down), 4–7 inverted stairs (up), 8–11 boxes,
  12–15 random rough, 16–17 slope, 18–19 inverted slope. And
  `terrain_types[env] = env_index // (num_envs/20)` is **fixed for the whole
  run** — an env can never change family, only row.
- So the real task space is a **6 × 10 grid** (family × difficulty), the policy
  is shared across all of it, and difficulty is smooth within a family but
  NOT comparable across families: row 9 rough (10 cm noise) is far easier than
  row 9 stairs (23 cm steps).

## 2. Pilot diagnosis — three mechanisms, each now quantified

**D1 — the teacher was flat (concentration failure).** Final pilot posterior:
p̂ = [0.43 … 0.12] across rows → learnability u ∈ [0.245, 0.106] — a **2.3:1**
contrast. After γ=1 normalization + 0.1 floor: sampling probs 0.131…0.071,
**effective_bins 9.5–9.6 of 10 for the entire run** (logged; never below 8.8).
The teacher we fielded was, numerically, a ±30% modulated uniform sampler. The
pilot's teacher≈uniform result is thereby *explained*, not just observed: on a
smooth low-contrast difficulty axis, γ=1 learnability cannot concentrate.
γ=4 (their own validated tight-chain setting) lifts the same posterior to 29:1
contrast / effective_bins 6.2. Terrain rows ARE a chain (competence at row k
transfers to k+1 — same skill, higher amplitude), so the compounding argument
for γ=4 applies; the pilot's γ=1 followed the "flat pool" default, which §1
now shows was the wrong regime call.

**D2 — greedy's locality is worth more than allocation in this regime.**
Greedy is a per-env reflected random walk: env fails → its own next level −1,
succeeds → +1 — difficulty conditions on the *individual* env's outcome with
zero lag. Modeling it as a Markov chain on the measured pilot p-curve gives a
stationary distribution concentrated on rows 0–2 (effective_bins ≈ 2.3) that
*tracks the policy upward* — i.e. greedy achieves ~4× the concentration of our
teacher, placed adaptively, using per-env information the pooled posterior
throws away. Its measured mean-level trajectory (3.5 → 0.03 → climb to 2.87)
is exactly this chain relaxing then walking. A competence-dynamics simulation
of the pilot (skill grows ∝ practiced u; 5 seeds) reproduces the ordering
quantitatively: greedy AUC 0.122 > teacher(γ=1) 0.113 > uniform 0.068 — and
predicts the fixes: **teacher(γ=4) 0.139 and hybrid 0.126 both ≥ greedy**.

**D3 — the 10-bin aggregate posterior mixes 6 incomparable families.** The
teacher's p̂(row) averages stairs and rough; the frontier row *within* family
differs (sim with realistic per-family slopes: argmax u at row 4 for stairs vs
row 9 for rough vs aggregate argmax 6). Aggregation both flattens the utility
curve further (D1) and mis-places the band per family. The fix is free: bin =
(family, row), 60 bins, `terrain_types` already stores family. Evidence per
bin drops 6× — mitigated by the evidence-scaled decay and by 60 bins still
receiving ~200 resets/bin/iteration-window at 1024 envs.

## 3. What we changed (implemented, 19/19 CPU tests)

1. **`teacher_g4` arm** — same FrontierTerrainTeacher, γ=4. One-parameter
   regime correction per D1. (Kept as a separate pre-registered arm rather
   than silently changing `teacher`.)
2. **`hybrid` arm (`HybridTerrainTeacher`)** — greedy's per-env ±1 walk
   (locality, D2) + the pooled posterior as *guardrails*: promotions into
   bins rated dead (p̂<0.05) or mastered (p̂>0.9) with ≥8 effective
   observations are skipped (2-step past if open, else stay); failure
   retreats are NEVER blocked (test-caught trap: a masked retreat pins an
   env on a dead bin); 10% of resets draw from the teacher distribution
   (posterior coverage + easy-row replay). This is the "division of labor"
   design: locality does band-tracking, pooling does waste-avoidance —
   each mechanism doing only what it demonstrably won at in the pilot.
3. Family-aware binning (D3) is **deferred to phase 2b** — it changes the
   bin space and therefore the eval probe; 2a isolates D1/D2 fixes on the
   unchanged 10-bin space so results stay comparable with the pilot.

## 4. Phase-2a experiment (stock grid, pilot-comparable)

Arms: `teacher_g4`, `hybrid` (+ reuse pilot's greedy/teacher/uniform runs as
the baseline triplet — same seed, same grid, same budget, same eval).
600 iters × 1024 envs, seed 42, `tile` predicate, fixed-grid eval @300/599.

Pre-registered predictions (from the D2 simulation + D1 arithmetic):
- **2a-P1:** teacher_g4 effective_bins drops to ~6 (vs 9.5) and its fixed-grid
  macro-pass@599 ≥ teacher(γ=1)'s 0.278 + 0.03.
- **2a-P2:** hybrid ≥ greedy − 0.02 on macro-pass@599 (locality preserved;
  masks should be near-inactive on this grid — blocked_bins ≈ 0 after
  warm-up — so hybrid ≈ greedy here; its edge is reserved for grids with
  real dead/mastered bins).
- **2a-P3 (falsifier):** if teacher_g4 still shows effective_bins > 8.5, the
  concentration diagnosis is wrong — stop tuning γ and move to phase 2b.

## 5. Phase-2b experiment (the discriminating grid)

16-row grid, `step_height_range=(0.05, 0.45)` (+boxes/slopes scaled): rows
~12+ exceed ANYmal-C's step ceiling → genuinely dead bins; 1500 iters from a
shared 300-iter warmstart → easy rows reach mastery. This manufactures BOTH
waste regimes the teacher/hybrid masks are designed for, on the axis where
greedy must fail (its walker has no concept of "this row is impossible" — the
chain keeps pushing envs into the wall at its top boundary, and keeps
re-confirming mastered rows at the bottom).

Arms: greedy, teacher_g4, hybrid, uniform. ≥3 seeds.
- **2b-P1:** hybrid > greedy on macro-pass AUC (masks now active:
  blocked_bins > 0 sustained; greedy wastes its boundary envs on dead rows).
- **2b-P2:** teacher_g4 > uniform (dead-row mass avoided: dead rows get
  ≤ floor share).
- **2b-P3 (honest null):** if greedy still wins, per-env locality dominates
  even with dead bins — the correct conclusion becomes "port locality INTO
  the frontier framework" (per-env posteriors / local Thompson), and the
  hybrid IS that port's first draft; report as such.

## 6. Family-aware teacher (phase 2b arm 5, if 2a confirms D1)

`FrontierTerrainTeacher` with `bin = family * n_rows + row` (60 bins), family
read from `terrain_types[env] // cols_per_family`; eval probe pins
(family, row) pairs. Prediction: per-family frontier placement beats aggregate
placement on mixed grids (the D3 sim gap). ~40 lines: the bin-mapping helper +
a `family_aware: bool` param; the FrontierBinTeacher core needs nothing.

## 7. Why rsl_rl stays the baseline learner

Everything above moves the *data distribution*; PPO/GAE stays stock rsl_rl.
That isolation is what makes arm differences attributable to the curriculum.
Estimator-side work (MaxRL success-conditioned weighting) remains gated to a
sparse-success manipulation task (ladder step 3) where a real verifier exists —
per D8 and their §9.5; dense-reward locomotion tests the teacher, not the
estimator.
