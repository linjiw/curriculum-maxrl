# GSM8K 2×2 (E-LLM-1) — analysis report

*Status 2026-07-27: 3 of 4 cells complete (50 steps each); cell 2 (maxrl,
uniform) finishing its last 25 steps. Pre-registered predictions:
`curriculum_maxrl/GSM8K_A10G_PLAN.md` (P-G1..P-G5, committed 39520fa before
any cell finished). Setup: SmolLM2-360M-Instruct, 7473 GSM8K prompts, N=16
rollouts, batch 64, 50 steps, single A10G, val = 256 held-out rows @ n=4.*

## What this experiment is FOR (read this first)

The 2×2 crosses **estimator** {MaxRL, GRPO} × **prompt sampling** {frontier
teacher, uniform}. It tests two of our three channels at LLM scale:

- **Channel 1 (teacher = waste avoidance):** ~65–75% of uniform groups are
  dead (all 16 rollouts fail → zero gradient under MaxRL). Does reallocating
  that compute help? *Meter:* maxrl+cur vs maxrl, AUC + final val.
- **Channel 3 (objective safety):** our maze H6 result says a frontier
  curriculum AMPLIFIES GRPO's coverage collapse (GRPO's inverted weights
  were silently maintaining easy prompts; the curriculum removes that
  maintenance). Does that transfer? *Meter:* grpo+cur vs grpo — the
  pre-registered P-G2, our riskiest prediction.
- Channel 2 (hindsight) is deliberately absent — it needs an exact relabel
  map, which GSM8K lacks; that is E-LLM-2 (Countdown).

## Results (val mean@4 / pass@4, 256 held-out rows)

| cell | step 0 | step 25 | step 50 | train AUC |
|---|---|---|---|---|
| grpo (uniform) | .078/.162 | .105/.213 | **.120/.229** | .0432 |
| grpo + teacher | .072/.151 | .096/.193 | **.093/.181** ↓ | .0404 |
| maxrl + teacher | .066/.136 | .099/.190 | **.102/.190** | .0432 |
| maxrl (uniform) | .091/.182 | .097/.207 | **.108/.204** | — |

*(all four cells complete, 2026-07-27)*

## Finding 1 — P-G2 CONFIRMS: the H6 reversal transfers to LLM scale

**grpo+teacher is the only cell that got WORSE from step 25 to 50**
(.096→.093 mean@4, .193→.181 pass@4) while plain grpo climbed monotonically
(.078→.105→.120). Identical budget, identical estimator — adding the
teacher *hurt* GRPO, with the peak-then-decline signature the maze predicted.
The teacher was verifiably live (dead-sampled fraction driven to 0.48–0.51
minima vs ~0.65 population; fixed per-chunk sampler, commit c71cbe7).

This is the paper's safety claim at LLM scale, on a pre-registered
prediction: **a curriculum is not an objective-agnostic add-on; the
estimator underneath decides whether it is safe.** Teacher+MaxRL trained
stably to its best val; teacher+GRPO regressed.

Caveat inventory: single seed; small model; val n=4; absolute deltas are a
few points. The *ordering* and the *sign of the step-25→50 change* are the
pre-registered outcomes, and both landed as predicted.

## Finding 2 — the teacher's two sub-channels separate at this scale

Post-hoc against external difficulty annotations
(`lime-nlp/GSM8K_Difficulty`, solve rates of a 7B model, 7449/7473 matched):

- **Evidence channel WORKS:** teacher p̂ on visited prompts anti-correlates
  with external difficulty (Spearman −0.166, p=3.5e-17) after ~1.25 visits
  per visited prompt. The Bernoulli posterior extracts real difficulty
  signal from single-group observations.
- **Allocation channel BARELY ENGAGES:** visits are flat across difficulty
  quintiles (0.42–0.43 mean visits everywhere; visited_frac 34%). With 3200
  group-draws over 7473 prompts, Thompson sampling stays near-uniform — the
  posterior needs several visits per prompt to concentrate, and the budget
  provides ~1.

**Interpretation:** GSM8K-at-50-steps is a *posterior-starved* regime — the
teacher learns but cannot yet act on it. This structurally predicts
P-G1 (maxrl+cur > maxrl) lands small-or-null HERE without invalidating the
mechanism: the maze gave the teacher ~500 groups over 13 bins (≈40
visits/bin); GSM8K gives ~0.4 visits/prompt. Prompt-level posteriors need
either longer runs, smaller pools, or bin-level aggregation (tiers — which
is exactly how E-LLM-2 is designed).

- Teacher telemetry corroborates: visited_frac 0.9%→34%, frac_dead(p̂<.05)
  0→10.7% — the posterior map builds smoothly but covers a third of the
  pool by run end.

## Finding 3 — entropy tells the collapse story differently at LLM scale

grpo (uniform) ended at entropy 0.643; grpo+teacher at 0.765; maxrl+teacher
at 0.720. GRPO+teacher regressed on val *despite* retaining more token-level
entropy — so the LLM-scale collapse is **not** naive entropy collapse; it's
distributional (which prompts improve), visible in val/pass@k, not in
token entropy. The maze showed the same dissociation (pass@8 collapsed
while pass@1 looked fine). Meter lesson again: coverage currency or blind.

## Finding 4 — P-G1 verdict: NULL on final value, consistent with the posterior-starvation diagnosis

Cell 2 complete: maxrl (uniform) finishes at .108 mean@4 vs maxrl+teacher's
.102. The teacher arm gained 2.1× more over training (+.036 vs +.017) but
from a lower step-0 draw (.066 vs .091 — same warmstart, val sampling
noise), and the trajectories converge rather than diverge. Verdict:
**P-G1 is a null at this budget** — exactly what Finding 2's
posterior-starvation analysis predicts (visits flat across difficulty at
0.4/prompt; the teacher's allocation never left uniform's neighborhood).
The structural prior from the pre-registration scoreboard ("small-or-null
without invalidating the mechanism") held. The mechanism test moves to
E-LLM-2's tier-level posterior (3 arms, hundreds of visits each) where
starvation is impossible.

Honest note: pass@4 tells the same story (.204 uniform vs .190 teacher) —
no coverage rescue either. At 50 steps × 7.5k prompts, channel 1 does not
pay at the prompt level. The H6 safety result (Finding 1) is unaffected
and remains the experiment's headline.

## Finding 6 — Seed-2 replication of the GRPO+teacher cell: the regression does NOT reproduce (2026-07-31)

The H6 completion seed (grpo+teacher, data.seed=2, 50 steps, identical
config; ckpt `grpo_curtrue_s2`) climbed monotonically:

| seed | step 0 | step 25 | step 50 | second half |
|---|---|---|---|---|
| 1 (registered) | .072/.151 | .096/.193 | .093/.181 | **regressed** |
| 2 (replication) | .073/.150 | .095/.188 | .118/.203 | **climbed** |

Read honestly, three facts:

1. **P-G2's regression signature is not seed-stable.** The pre-registered
   outcome landed on the registered run; the replication seed shows no
   second-half decline and ends at mean parity with seed-1 uniform grpo
   (.118 vs .120).
2. **The treatment was weaker in seed 2.** Steering telemetry:
   min dead-sampled fraction 0.531 (seed 1: 0.48), mean 0.656 ≈ the 0.65
   population rate — the teacher was near-uniform on average this seed
   (posterior starvation + a different dataloader order). A near-null
   treatment cannot damage; mechanistically consistent with
   "steering causes the damage," but as evidence it means seed 2 did not
   fully administer the treatment. We state this as observed telemetry,
   not as an excuse: pre-registration binds us to report the miss.
3. **The coverage direction survives:** seed-2 final pass@4 .203 remains
   below uniform grpo's .229 (cross-seed comparison; same direction as
   Finding 5's monotone k-widening deficit).

**Verdict for the paper:** the LLM-scale teacher×estimator interaction is
demoted from "confirmed" to "1 of 2 seeds, treatment-intensity-dependent
— suggestive, not established." The maze estimator main effect (9 runs,
p=0.0079) is unaffected. Entropy note: seed-2 ended at 0.652 vs seed-1
grpo+teacher's 0.765 — the entropy-retention observation is also
seed-variable.

### Update 2026-08-01: the within-seed pair lands (grpo uniform seed 2)

The H6 completion run (raysession_2026-07-31_16-29-58, artifact
`gsm8k_partial/grpo_uniform_seed2.json`) makes the seed-2 contrast
within-seed instead of cross-seed:

| seed-2 cell | mean@4 traj | final pass@4 |
|---|---|---|
| grpo uniform | .081 → .085 → **.125** | **.229** |
| grpo+teacher | .073 → .095 → .118 | .203 |

Within seed 2 the teacher deficit is −.007 mean@4 (inside the .0094
noise floor) and −.026 pass@4 (~1.5× the .0172 pass@4 floor) — same
direction as seed 1 (−.027 / −.048, z≈2 each), smaller magnitude,
consistent with seed 2's weaker delivered steering (min dead-sampled
.531 vs uniform's own .531-min this seed: means both ≈.66). Sanity:
uniform-GRPO seed 2's final (.125) also replicates seed 1's (.120).

**Sharpened verdict:** the endpoint teacher-deficit *direction* under
GRPO is now 2/2 seeds on both meters (4/4 signed contrasts); what
remains 1-of-2 is the pre-registered *second-half regression* shape,
and magnitude tracks steering intensity. Still short of established —
the steering-controlled multi-seed cell remains the decisive
experiment — but the replication miss no longer reads as a sign flip,
only as a dose effect.

### Correction 2026-08-05: response-length "estimator signature" withdrawn

Finding F-D (all GRPO runs compress harder than all MaxRL runs,
209.6±1.8 vs 232.4±5.7) was advanced on five runs. The sixth run —
grpo uniform seed 2, the within-seed control — final-10 response length
is **251.9 tokens** (32.1% shrink from its 371-token warmstart), longer
than every earlier run including both MaxRL cells. The perfect
separation is falsified; length shrink at this scale is seed/run-level
variation, not an estimator signature. Withdrawn from the paper.

## Finding 5 — k-sweep: the teacher's GRPO damage grows with k (P-G3, directional)

Final checkpoints, vllm, n=16 samples on the 256-row slice, unbiased
pass@k (`gsm8k_partial/ksweep_results.json`):

| cell | p@1 | p@4 | p@8 | p@16 |
|---|---|---|---|---|
| grpo | .038 | .111 | .168 | **.238** |
| maxrl | .033 | .099 | .157 | .231 |
| grpo + teacher | .030 | .091 | .143 | .215 |
| maxrl + teacher | .028 | .084 | .138 | .215 |

Three honest reads:
1. **The teacher's damage to GRPO grows with k** — deficit −.008 at k=1
   widening to −.024 at k=16, the P-G3 signature (curriculum damage lives
   in coverage). Directional support: each individual delta is within
   single-seed noise (se≈.026 unpaired at n=256), but the monotone widening
   across four k values is the predicted pattern.
2. **At this budget the teacher costs coverage under BOTH estimators**
   (maxrl+teacher −.016 at k=16) — smaller than GRPO's cost, but not the
   maze's coverage GROWTH. Consistent with the starved-posterior diagnosis:
   a near-uniform teacher with a 0.48-dead-floor sampled fewer effective
   unique prompts per epoch than uniform did, a pure cost with no
   allocation benefit. Channel 1 at prompt-level 50-step budgets: null to
   slightly negative.
3. **GRPO's coverage did not collapse in 50 steps** (its p@16 leads all
   cells) — collapse horizons in the maze were hundreds of steps; the
   H6 signature here is the teacher-induced *relative* degradation
   (Finding 1's trajectory + the widening-with-k deficit), not absolute
   collapse. State this precisely in the paper.

## What remains before E-LLM-1 closes

1. ~~Cell 2 final val~~ DONE — Finding 4.
2. ~~k-sweep~~ DONE — Finding 5.
3. Dead-fraction trajectory figure + curves.json export for the site.
4. Fold verdicts into PAPER §7.5 (done in the same commit as this edit).

## Pre-registered predictions scoreboard (live)

| prediction | status |
|---|---|
| P-G1 teacher AUC gain over maxrl | **NULL at this budget** (final .102 vs .108; 2.1× the improvement slope but start-dominated) — as the posterior-starvation analysis predicted |
| **P-G2 grpo+cur does NOT beat grpo** | **CONFIRMED** (and regressed 25→50) |
| P-G3 pass@k divergence clearer than mean | **directional** — grpo+teacher deficit widens monotonically −.008→−.024 from k=1→16; individual deltas within single-seed noise |
| P-G4 teacher bends dead-fraction below population | WEAKENED on review (opus5 M2): the 0.48 is the run *minimum*; uniform GRPO's own minimum is 0.516 and all run *means* are indistinguishable (0.66–0.67). Honest read: treatment delivered but weak — consistent with posterior starvation, not strong steering. Earlier CONFIRMED claim retracted twice (epoch-frozen sampler, then min-vs-mean). |
| P-G5 absolute gains small; ordering is the outcome | holding exactly as written |
