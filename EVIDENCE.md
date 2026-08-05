# The evidence, reorganized: where the strength actually comes from

> **Currency note (2026-08-06):** this synthesis predates the balanced
> factorial. The "H6 estimator main effect, 9/9 runs, p=0.0079" claim
> below **failed its pre-registered factorial at the registered
> endpoint and is retracted** — see `maze_gpu_factorial/FACTORIAL_VERDICT.md`
> and paper §6.3b for what survives (exact-rung 10/10; exploratory
> covAUC 12/12, wave-2 confirmation running). Sections quoting H6 as
> established are historical.

*Synthesis pass over ~35 experiments, 7 propositions, 3 testbeds, and 2 external
ports. Not a chronology — a decomposition. REPORT.md tells the story of the
project; this document tells you how the method works, when to use which part,
and what each claim rests on.*

---

## 1. The three channels (decomposition of the method's strength)

Everything the method does flows through exactly three channels. Every
experiment we ran gains its effect through one of them, and knowing *which* is
what lets you predict where the method will and won't help.

### Channel 1 — WASTE AVOIDANCE (the teacher)
*"Don't roll out where the estimator will emit nothing."*

- Mechanism: sample ∝ u(p) = pass@N − pass@1 (P1: the estimator's exact
  expected signal). Zero at mastered (p→1) and unreachable (p→0) tasks.
- What it buys, measured: dead groups 5.8→3.4 of 8 (maze — historical
  zero-weight-group counter, mechanism-open per the EXPERIMENTS.md audit:
  it pools K=0 with K=N and cannot isolate dead-group waste); 22–35% more
  optimization steps per GPU-hour (frontier rollouts also end earlier);
  6/6 paired-seed wins vs uniform.
- Ceiling: the ORACLE bound — CORRECTED (Opus5 review B3 + our control
  battery, `frontier_rl/examples/hindsight_controls.json`). The published
  "oracle" carried a 10% floor handicap; the honest no-floor γ-matched
  oracle reaches 0.8885, TYING the full stack (0.8895). Creation still
  adds on top of perfect allocation (+0.005, oracle+HS 0.8935) but the
  channels substitute more than they compose: "beats the oracle by 0.039"
  is retracted; "+0.005 on top of the best sampler including an oracle"
  is the honest number. Realized Thompson-teacher gain is +0.05–0.08 AUC
  on CPU and ~+0.01 on the maze.
- When it's the dominant channel: mixed-difficulty pools with real spread
  (the balanced regime), and any setting where rollouts are the cost center.

### Channel 2 — SIGNAL CREATION (hindsight recycling)
*"Manufacture verified successes from the failures you already paid for."*

- Mechanism: a dead group's rollouts are relabeled to the sub-goals they
  actually achieved; the same success-conditioned weights apply. P6 + V1:
  where the env's relabel is exact, these gradients are *indistinguishable
  from fresh unbiased groups* (cosine 0.956 vs 0.958; mean → 1.000).
- What it buys, measured: the only channel that breaks the oracle ceiling
  (0.890 > 0.851); the only channel that scores at all in frontier-heavy
  regimes (0.98 vs exactly 0.00 for every pure sampler including DAPO);
  MountainCar flag 0.000 → 1.000.
- Its own boundary, measured: the gain is proportional to how much a
  relabeled skill can COMPOUND. Fixed task set (CPU skill chain): +0.22 AUC.
  One-shot tasks (infinite-data maze): +0.01, reliable but small. This is
  the single most important regime variable in the whole project.
- Contracts (violations measured, not hypothesized): exactness (true success
  under the env's verifier) and conditioning-rewrite (goal-embedded
  trajectories must be rewritten — skipping it made hindsight HURT,
  0.600 < 0.658, on the gridworld).

### Channel 3 — OBJECTIVE SAFETY (MaxRL weighting underneath)
*"The curriculum is only safe on a likelihood-shaped objective."*

- Mechanism: P5 — MaxRL concentrates ≈(N−1)× more signal than RLOO on
  frontier tasks as p→0, and unlike GRPO its weight function doesn't invert
  at p→1.
- Decomposition (control battery, 5 seeds): of the +0.22 hindsight gap
  on fixed pools, an extra-gradient placebo (replaying LIVE gradients on
  dead-group slots, zero relabel information) captures 83%, lr×2 captures
  68%, and random-target relabeling 69%. The relabel DIRECTION carries
  +0.037 beyond the strongest placebo — real, exactness-dependent (fake
  labels are actively harmful), but the headline "+0.22 from signal
  creation" decomposes into ~0.18 gradient-dose effect + ~0.04 direction
  information. Both numbers ship together from now on.
- What it buys, measured: the H6 reversal. The identical teacher GREW
  coverage under MaxRL every seed (pass@8 0.316→0.348); GRPO decayed
  coverage every seed, and in the seed run with a teacher the collapse
  was AMPLIFIED (0.332→0.269, easy-retention lost — single-seed arm).
  GRPO's inverted weighting was silently maintaining easy tasks; the
  curriculum removes that maintenance.
- This is a compatibility theorem in empirical form: **channels 1+2 are not
  objective-agnostic add-ons.** Ship them on GRPO and you make it worse.

**The one-line synthesis: the teacher allocates, hindsight creates, the
objective decides whether either is safe.**

## 2. The regime map (when each channel dominates)

| regime | ch.1 teacher | ch.2 hindsight | evidence |
|---|---|---|---|
| easy-heavy pool | small (+0.01–0.03) | small | V5 row 1: everything ≈0.93+ |
| balanced spread | moderate (+0.05) | large on fixed sets | V5 row 2, CPU main tables |
| frontier-heavy (p≈0 pool) | **zero** (nothing to allocate) | **categorical** (0→0.93+) | V5 row 3, MountainCar |
| infinite/one-shot tasks | moderate | small (+0.01, no compounding) | maze GPU, F3/F4 |
| fixed task set | moderate | **largest** (+0.22, compounds) | CPU, MountainCar shared |
| starved (tiny eval budget) | zero | zero | SONIC F3 + our V3 agreement |

Second-order effects, all measured:
- **Parameter sharing is the transfer channel** (MountainCar per-bin 0.000
  vs shared 1.000; maze-size cliff). No sharing ⇒ no curriculum can work.
- **γ concentration tracks structure**: γ≈4 on tight chains (compounding,
  V6/V6b ODE), γ=1 on broad/flat pools (GPU maze non-transfer — predicted).
- **Capacity interacts with the frontier**: the teacher walks the frontier to
  the policy's per-step-execution ceiling and stalls there (q-diagnosis);
  more capacity resumes the march (L6 coverage 0.188→0.438).

## 3. The meter lesson (how to even see the method working)

Three separate times, the *metric* hid what the method was doing:

1. **Fixed-step comparisons hid the teacher's speed** (it runs 22–35% more
   steps per hour) → matched wall-clock protocol.
2. **Peakedness hid the teacher's targeting** (ZPD utilities are diffuse by
   design — SONIC_RESPONSE Q5) → targeting-ratio criterion.
3. **pass@1 hid the deep-frontier march entirely** (L6 "stalled at 0.01–0.05"
   while coverage@64 went 0.125→0.438) → coverage currency.

Generalization: **likelihood-shaped training moves the distribution's tail
first; any single-sample metric under-reports it.** Evaluate in coverage/
efficiency currency or you will kill working runs.

## 4. Practitioner's playbook (how to actually play it)

Decision procedure distilled from every ablation:

1. **Choose the difficulty axis** so bins share parameters (goal-condition,
   never partition). Check: does competence at bin k move eval at bin k+1?
   If not, fix the representation before any curriculum work.
2. **Teacher config**: learnability p(1−p) if you have no natural group size
   N (dense-PPO, hazard-style p); advmass with your real N if you have
   episodic groups. Decay 0.7 (evidence-scaled half-life if throughput
   varies). Floor 0.1. Thompson if stochasticity is acceptable; mean+k·std
   if not. γ=1 unless the pool is a tight chain.
3. **Hindsight**: ON wherever the env can relabel exactly; expect gains ∝
   task-set fixedness. Both contracts enforced (unit-test the conditioning
   rewrite). Never feed relabels to the teacher's posterior (V4 + GPU C).
4. **Objective check**: MaxRL/likelihood weighting underneath. If the team
   insists on GRPO, do NOT ship the curriculum (H6).
5. **Metrics from day one**: coverage@k (not just pass@1), easy-decile
   retention, dead-group rate, teacher p̂-vs-eval calibration.
6. **When the frontier stalls**: run the q-diagnosis (per-step accuracy →
   geometric reach). Capacity problem ⇒ wider/longer; curriculum problem ⇒
   check sharing + relabel contracts.

## 5. Claim inventory (every load-bearing claim and its strongest evidence)

| claim | strongest single piece of evidence | grade |
|---|---|---|
| u(p) = estimator's exact expected signal | P1 proof + 200k-trial MC | proved |
| teacher beats uniform (matched clock) | 6/6 paired seeds; step-matched, teacher-only is n.s. (+0.006, p≈0.36) — the clock gain is largely throughput | multi-seed, decomposed |
| hindsight direction carries information | control battery: +0.037 beyond dose-matched replay, tightest arm (±.0025); random-target loses to replay 5/5 — cosine table RETIRED as evidence (no discriminating power, Opus5 B5) | controlled |
| full stack ties matched oracle; creation adds +0.005 on top | `hindsight_controls.json`: no-floor γ-matched oracle 0.8885 ≈ full stack 0.8895; oracle+HS 0.8935. ("beats the oracle 0.890 vs 0.851" RETRACTED — floor handicap) | multi-seed CPU |
| creation is the only live channel in dead regimes | V5 frontier-heavy w/ uniform+HS control (0.931 ≈ teacher+HS 0.928, both vs 0.000): allocation contributes nothing there; artifact `results_baselines_regimes.json` | controlled |
| curricula require likelihood weighting | H6: estimator main effect 9/9 runs, perm p=0.0079; LLM-scale interaction 1-of-2 seeds, tracks steering intensity | multi-seed / 1-of-2-seeds* |
| compounding drives hindsight's size | CPU +0.22 (83% captured by replay placebo — ship both numbers) vs maze +0.01 | cross-regime, decomposed |
| coverage is the right meter | L6 0.125→0.438 invisible to pass@1 | single-ckpt* |
| efficiency grows with difficulty | 11× at L5 / tie L2–3 / 3× worse L4 at the stated absolute-0.25 convention (old 1.2×/0.5× cells not reproducible from artifact — corrected) | single-seed* |
| sharing is the transfer channel | MountainCar per-bin 0 vs shared 1.000 | controlled |
| γ tracks structure | V6 + ODE model + GPU non-transfer *prediction* | pre-registered |

\* = worth one more seed before external claims; flagged in REPORT.

## 6. What we'd still like to know (ranked)

1. Fixed-prompt-set at LLM scale (GSM8K) — the regime map says this is where
   ch.2 compounds; the single most valuable missing cell.
2. One more seed on the efficiency multipliers (cheap, de-stars two claims).
3. Does the wide model's L6 coverage convert to pass@1 with more budget
   (capacity × duration interaction)?
4. Streaming teacher on a real procedural source (only validated on synthetic
   continuous goals so far).
