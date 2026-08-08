# The evidence, reorganized: where the strength actually comes from

> **Currency note (updated 2026-08-06):** this synthesis predates the
> balanced factorial. The "H6 estimator main effect, 9/9 runs,
> p=0.0079" claim below **failed its pre-registered factorial at the
> registered endpoint and is retracted** — see
> `maze_gpu_factorial/FACTORIAL_VERDICT.md` and paper §6.3b for what
> survives (exact-rung 10/10; covAUC then **confirmed as the wave-2
> registered primary, 6/6 per sampler, p=.031 each, 24/24 across
> waves**). Also superseded here: the "gate strength is a dial"
> reading — the designed-strength sweep refuted it (P-R1, 3 seeds;
> `countdown_reviewer_arms/PROVENANCE.md`); and the "+0.22 hindsight
> gain" framing — at LLM scale a dose-matched replay control exceeds
> recycling on both meters (ARM B, 2/3 seeds interim). Sections
> quoting these as established are historical.

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

- Mechanism: an all-fail group's rollouts can be credited to verifier-valid
  goals they actually achieved, producing an auxiliary selected-data stream.
  Equality with a fresh target-task update requires equality of the relevant
  update moments; full rewritten-group law equality is sufficient. Verifier
  validity and conditioning rewrite are necessary but not sufficient.
- Retained local evidence: under `γ=4`, centered hindsight added `+0.1050`
  AUC on the 12-seed skill chain. In tile-coded MountainCar it added `+0.191`
  (centered) or `+0.197` (success-only) AUC over the matched teacher arms;
  final custom flag-pass was about 0.84, not a standard-return result. The
  corrected grid study ordered 0.583 uniform < 0.652 teacher < 0.702 full
  stack, but was group-step rather than transition matched.
- A frontier-heavy toy shows the clean categorical mechanism: pure sampling
  cannot create signal when every pool task is unreachable, whereas valid
  auxiliary goals can ignite learning. This is not yet a broad robotics or
  language-model result. The small one-shot-maze hindsight effect is from the
  audited historical GPU protocol and remains hypothesis-generating.

### Channel 3 — LEARNER INTERACTION (the weighting underneath)
*"A curriculum changes the data stream; the learner determines the update."*

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

**The one-line synthesis: the teacher reallocates groups, hindsight adds
auxiliary targets, and the learner converts those data into updates.**

## 2. The regime map (when each channel dominates)

| observed regime | ch.1 teacher | ch.2 hindsight | evidence boundary |
|---|---|---|---|
| mixed nested thresholds | local positive effect | larger local effect | corrected CPU/Gym studies |
| frontier-heavy pool | no bootstrap in the limit | categorical ignition possible | synthetic mechanism only |
| fixed recurring goals | compounding is plausible | largest retained local effects | skill chain + tile-coded MountainCar |
| one-shot procedural tasks | unresolved under corrected protocol | possibly smaller | historical maze hypothesis only |
| arm-starved budget | posterior cannot localize | also hard to estimate | arithmetic/design constraint |

Second-order hypotheses worth testing:
- **Shared transfer:** shared parameters are a plausible competence-transfer
  channel, but the MountainCar shared/per-bin comparison confounds capacity
  and data flow, and Acrobot's capacity controls were behaviorally inadequate.
- **Concentration:** `γ=4` helped in two tightly shared nested-task studies;
  it is an empirical knob, not a theorem or a default for broad pools.
- **Capacity:** historical maze probes suggest a capacity/frontier
  interaction, but corrected multi-seed confirmation is still missing.

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

1. **Choose and test the task axis.** Use one goal-conditioned policy where
   plausible, then measure whether training bin `k` changes held-out behavior
   at `k+1`. Include capacity- and data-adequate controls before calling the
   effect transfer.
2. **Teacher config**: learnability p(1−p) if you have no natural group size
   N (dense-PPO, hazard-style p); advmass with your real N if you have
   episodic groups. Decay 0.7 (evidence-scaled half-life if throughput
   varies). Floor 0.1. Thompson if stochasticity is acceptable; mean+k·std
   if not. Start with `γ=1`; register `γ=4` as a distinct concentration arm
   when nested shared tasks make it plausible.
3. **Hindsight**: enable only with a binary verifier, goal-conditioning
   rewrite, first-hit truncation where applicable, and an adequacy pilot.
   Compare relabeled and fresh target-task update moments; never feed relabels
   to requested-task teacher state.
4. **Learner check**: register the estimator/surrogate and include an
   objective-by-teacher ablation. Do not generalize the historical GRPO
   interaction before a corrected run.
5. **Metrics from day one**: standard environment score, fixed-target pass
   rates, `K=0`/mixed/`K=N` counts, coefficient mass, gradient norm/alignment,
   actual transitions, optimizer updates, and wall-clock.
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

The historical maze coverage and efficiency multipliers remain useful for
experiment design, not for external performance claims.

## 6. What we'd still like to know (ranked)

1. Fixed-prompt-set at LLM scale (GSM8K) — the regime map says this is where
   ch.2 compounds; the single most valuable missing cell.
2. One more seed on the efficiency multipliers (cheap, de-stars two claims).
3. Does the wide model's L6 coverage convert to pass@1 with more budget
   (capacity × duration interaction)?
4. Streaming teacher on a real procedural source (only validated on synthetic
   continuous goals so far).
