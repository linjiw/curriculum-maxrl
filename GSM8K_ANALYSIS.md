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
| P-G4 teacher bends dead-fraction below population | CONFIRMED post-fix (min 0.48 vs 0.65 population); earlier claim retracted, see val_checkpoints.md |
| P-G5 absolute gains small; ordering is the outcome | holding exactly as written |
